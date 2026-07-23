# GLM5 TensorCast 适配设计

## 修订记录

| 日期 | 修订版本 | 修改描述 | 作者 | RFC文档 |
| -- | -- | -- | -- | -- |
| 2026-04-30 | 1.0 | 初稿完成，归档 TensorCast 适配 GLM5 系列模型及 Sparse MLA/DSA Indexer 融合算子设计 | minghang_c | - |
| 2026-07-22 | 1.1 | 补充 GLM-5.2 IndexShare 执行、cache 分配、MTP 和层复用设计 | minghang_c | - |
| 2026-07-22 | 1.2 | 补充旧版 Transformers 的 IndexShare 兼容 wrapper 与 MTP/repetition 修正 | minghang_c | - |
| 2026-07-23 | 1.3 | 澄清连续 shared layer 的 IndexShare top-k 转发语义 | minghang_c | - |

## 背景描述

GLM5 系列模型在注意力、MoE 和归一化路径上与 TensorCast 已支持的 DeepSeek-V3.2 稀疏 MLA 路径高度相似，但存在若干关键差异。如果仅复用原有 DeepSeek 路径，TensorCast 在 GLM5 编译和仿真时会遇到以下问题：

1. GLM5 的 HuggingFace 配置类型为 `glm_moe_dsa`，模型结构中使用 `GlmMoeDsaAttention`、`GlmMoeDsaMoE` 和 `GlmMoeDsaDecoderLayer`，原有 TensorCast model profile 无法识别这些模块。
2. GLM5 sparse attention 配置使用 `index_topk` 表示 DSA indexer 的 top-k 上限，而 DeepSeek-V3.2 路径使用 `topk_limit`。直接访问 `topk_limit` 会在 GLM5 构建或 `torch.compile` 过程中触发属性缺失。
3. 原有 DSA indexer 路径由多个 eager ATen 算子和 `dsa_index`、`dsa_index_cache` 等分散语义算子组成，trace 中无法呈现真实硬件语义上的单个 LightningIndexer/DSA Indexer block，也不利于性能模型统一建模。
4. GLM5 decoder block 中 RMSNorm 使用非默认 `eps`，原有 pattern matching 将 `eps` 固定在 pattern 闭包内，导致非默认 epsilon 场景无法稳定融合到 `tensor_cast.rms_norm`、`tensor_cast.add_rms_norm`、`tensor_cast.add_rms_norm2`。
5. Sparse MLA 的 DSA indexer cache dtype 需要跟随 attention quantization 配置。FP8 attention 下 indexer cache 应使用 FP8 dtype，非 FP8/GLM5 bf16 路径下应保持 bf16/fp16 语义，否则性能模型和内存估算会偏离实际路径。
6. GLM-5.2 在 `glm_moe_dsa` 架构上引入 IndexShare。`shared` sparse-attention layer 应复用前序 `full` layer 生成的 top-k indices，而不是每层执行 DSA indexer；若仍按逐层 indexer 建模，会高估 indexer 时延和 indexer cache 内存。

本特性的目标是让 TensorCast 能够正确编译和仿真 GLM5 系列模型，覆盖 sparse attention、MLA、MoE、MTP、RMSNorm fusion 以及 DSA indexer 相关性能估算。设计上尽量复用既有 DeepSeek sparse MLA 抽象，只在模型注册、配置兼容层、DSA indexer 语义算子和 pattern matching 能力上补齐差异，避免为 GLM5 复制一套独立实现。对于 GLM-5.2，IndexShare 在既有 `Glm5SparseAttention` wrapper 上表达跨层 top-k 数据流，不新增第二套 sparse MLA 或 indexer 实现。

## 方案设计

### 整体设计思路

本方案将 GLM5 适配拆分为五个层次：

1. **模型识别层**：新增 GLM5 model profile，将 HF `model_type="glm_moe_dsa"` 映射到 TensorCast 已有 MoE、MLA 和 MTP 包装机制。
2. **Sparse MLA 兼容层**：在 `DeepseekSparseAttention` / `DeepseekSparseAttentionIndexer` 中吸收 GLM5 与 DeepSeek 配置字段差异，统一向后端暴露 `topk_limit` 和 `topk_indices`。
3. **语义算子层**：用新的 `tensor_cast.dsa_indexer.default` 表达完整 DSA Indexer 语义块，替代拆散的 eager indexer 计算、cache 写入和 `dsa_index` / `dsa_index_cache` 流程。
4. **编译与性能建模层**：补齐 RMSNorm 非默认 epsilon pattern matching、DSA indexer analytic performance model、request length 感知的 sparse attention 估算，以及 indexer cache dtype 推导。
5. **GLM-5.2 IndexShare 层**：根据 `indexer_types` 在 full layer 运行 indexer、在 shared layer 转发已有 top-k，并通过 Transformers 兼容 wrapper、MTP 和 MLP-only repetition 保持跨层数据依赖及 cache 分配流程正确。

核心原则是：GLM5 不新增一套并行 MLA 实现，而是通过 model profile 接入既有 TensorCast sparse MLA wrapper；配置差异在 wrapper 边界收敛；trace 和性能模型统一面向 `dsa_indexer` 语义算子。GLM-5.2 的 shared layer 不重新产生该语义算子或其成本，而是复用对应 source full layer 的 top-k 结果。

### 模型注册与模块映射

新增 `tensor_cast/transformers/builtin_model/glm5.py`，注册 `glm_moe_dsa` 模型画像：

- `moe_module_name="GlmMoeDsaMoE"`
- `moe_num_experts_key="n_routed_experts"`
- `moe_gate_returns_raw_logits=True`
- `mla_module_name="GlmMoeDsaAttention"`
- `mla_module_class_type=Glm5SparseAttention`
- `mtp_block_module_name="GlmMoeDsaDecoderLayer"`
- `custom_expert_module_type=MoeExpertMLP`

该映射说明 GLM5 的 sparse attention 路径可以复用 `DeepseekSparseAttention` wrapper，但 MoE gate 输出、expert 结构和 MTP block 名称需要按 GLM5 HF 模块命名接入 TensorCast transformation pipeline。

同时，`tensor_cast/transformers/transformations.py` 调整 MTP layer type 扩展逻辑：当 HF config 存在 `layer_types` 或 `mlp_layer_types` 时，复制最后一个已有 layer type 作为新增 MTP layer 的类型，而不是把整个列表作为元素追加。这样 GLM5 这类同时维护 attention layer type 与 MLP layer type 的模型，在 MTP 扩展后仍保持配置列表结构正确。

### Sparse MLA 配置兼容

GLM5 HF 原生配置暴露 `index_topk`，不暴露 `topk_limit`。DeepSeek-V3.2 本地兼容配置则统一到 `topk_limit`。为避免模型 profile 层制造伪字段，本方案在 Sparse MLA wrapper 内提供 `_resolve_sparse_topk_limit(...)`：

1. 优先使用显式传入的 `topk_limit`。
2. 其次读取 indexer 自身的 `topk_limit`。
3. 再读取 attention config 或 indexer config 上的 `topk_limit`。
4. 最后读取 GLM5 风格的 `index_topk`。
5. 均不存在时抛出 `AttributeError("topk_limit")`。

`DeepseekSparseAttentionIndexer` 在构造时缓存解析后的 top-k 值，避免 `torch.compile` 路径中动态属性查找造成不稳定行为。`DeepseekV32Config` 保留窄范围的 `index_topk -> topk_limit` 兼容逻辑，仅用于旧 DeepSeek JSON fixture，不作为 GLM5 真实运行路径的依赖。

### GLM-5.2 IndexShare

GLM-5.2 沿用 `model_type="glm_moe_dsa"`，并通过 HF config 的 `indexer_types` 描述各 decoder layer 的 indexer 行为。支持的值为：

- `full`：该 layer 是 source layer，执行既有 DSA indexer 并生成 `topk_indices`。
- `shared`：该 layer 不执行 indexer，复用最近一个前序 `full` layer 传递的 `prev_topk_indices`。

`get_glm5_indexer_types(...)` 和 `resolve_glm5_indexer_source_layer(...)` 负责解析并校验该配置。配置长度不足、未知类型，或没有前序 full source 的 shared layer 均会抛出 `ValueError`，避免以错误的跨层依赖继续仿真。

`Glm5SparseAttention._run_sparse_attention_indexer(...)` 根据已解析的 source layer 选择执行或复用路径：source layer 调用父类的 DSA indexer；shared layer 直接返回 `prev_topk_indices`，缺失该输入时失败。为传递该状态，GLM5 attention forward 始终返回 `(attn_output, attn_weights, next_topk_indices)`；当下一层为 shared 时，第三项传递当前 layer 使用的 active top-k：full layer 转发新生成的 top-k，shared layer 在连续 shared 链中原样转发复用的 top-k。仅当后续 shared layer 不再需要该状态时，第三项为 `None`。因此 `full -> shared -> shared` 会持续传递同一份 top-k。基类 MLA 的默认输出仍保持既有二元语义。

MTP 扩展不简单复制最后一个 `indexer_types` 值。主 decoder stack 之外的 MTP proposal block 无法消费主栈的 `prev_topk_indices`，因此 `extend_glm5_indexer_types_for_mtp(...)` 为启用 IndexShare 的 GLM5.2 追加 `full`，使每个新增 MTP block 独立执行 indexer；空列表或 MTP layer 数为零时保持不变。

在 transformation 阶段，`patch_mla(...)` 将 source layer index、indexer type、`skip_topk` 和 `next_skip_topk` 标记写入 GLM5 attention wrapper。启用 repetition 时，完整 decoder layer 的 `CopyLayerWrapper` 只能重放第一个 tensor 输出，不能保留 IndexShare 所需的辅助 `topk_indices`。因此，`maybe_reuse_layers(...)` 对启用 IndexShare 的 GLM5.2 仅复用无跨层状态的 MLP 子模块，decoder attention 和 top-k 数据流仍逐层执行；如果 layer 不暴露标准 `.mlp` 子模块，则回退到既有完整 layer 复用逻辑。未启用 IndexShare 的 GLM5 继续采用既有完整 layer 复用行为。

### Transformers 兼容 wrapper

不同 Transformers 版本的 GLM-5 decoder 对 `prev_topk_indices` 和三元 attention 输出的支持不同。GLM5 model profile 注册 `patch_glm5_model(...)`，仅在原生 decoder 尚不支持 `prev_topk_indices` 时安装兼容 wrapper；已具备该接口的版本不改写执行路径。

兼容模式下，`Glm5ModelCompat` 在 decoder loop 中以 `None` 初始化 top-k state，并将每层 attention 输出第三项传入下一层。它沿用配置的 `use_cache` 默认值、按需创建 `DynamicCache`，并按原生路径构造 causal mask 与 RoPE，最终返回 `BaseModelOutputWithPast`。`Glm5DecoderLayerCompat` 执行原始 layernorm、attention、MLP 和 residual 路径，要求 attention 明确返回 `(attention_output, attention_weights, topk_indices)`；缺少第三项即抛出 `ValueError`，成功时向 compatibility loop 返回 `(hidden_states, topk_indices)`。这样旧版 Transformers 的 decoder 也能执行 GLM-5.2 full/shared IndexShare，而不会静默丢弃辅助输出。

兼容 wrapper 需要与 repetition 和 MTP 共存：`RegionMarkerWrapper` 持有真实 layer 时只包装其内部 layer，`CopyLayerWrapper` 没有独立可执行 decoder/MTP block，不能重复 patch。MTP patch 会定位 RegionMarker 包装下的真实 `mtp_block`，跳过 copy layer，避免重复包装或错误替换。

### `mlapo` / `mlapo_quant` 与 `qa_normed` 传递

DSA Indexer 需要复用 MLA 主路径中 query low-rank projection 后的 normalized query activation。为避免在 indexer wrapper 中重复执行 `q_a_proj` 和 `q_a_layernorm`，`tensor_cast.mlapo.default` 与 `tensor_cast.mlapo_quant.default` 的返回值扩展为四元组：

1. `q_states`
2. `kv_c_normed`
3. `k_rot`
4. `qa_normed`

当 `q_lora_rank` 不存在时，`qa_normed` 使用空 last-dimension tensor 作为 shape-only 占位，调用侧再转换为 `None`。Sparse attention wrapper 将 `qa_normed` 传入 indexer，减少重复计算，也让 `dsa_indexer` 语义块能够完整表达 query/key 准备和 top-k 选择。

### DSA Indexer 融合语义算子

新增 `torch.ops.tensor_cast.dsa_indexer.default`，用一个 TensorCast 自定义算子描述完整 DSA Indexer 语义。该算子声明 `indexer_cache` 为 mutable argument，输入包括：

- `hidden_states`
- `qa_normed`
- RoPE `cos` / `sin`
- `indexer_cache`
- `slot_mapping`
- `block_tables`
- `seq_lens`
- `wq_b_weight`
- `wk_weight`
- `weights_proj_weight`
- `k_norm_weight`
- `num_heads`
- `head_dim`
- `qk_rope_head_dim`
- `topk_limit`

返回 `topk_indices`，shape 为 `(batch, seq_len, min(topk_limit, active_seq_len))`。在 FakeTensor / `torch.compile` tracing 场景下，算子不从 `seq_lens` 提取 Python int，直接使用 `topk_limit` 作为静态 shape 上限，避免数据依赖破坏编译。

语义上，`dsa_indexer` 同时覆盖两类路径：

- **DeepSeek-V3.2 FP8 路径**：包含 query/key projection、RoPE、`rotate_activation`、activation quantization、FP8 key cache/scale cache 写入、FP8 index score、top-k selection。
- **GLM5/BF16 路径**：移除 FP8 特有的 activation rotation、quantization、scale cache 和 FP8 score shaping，使用 bf16/fp16 cache scoring、head weight mixing、head reduction 和 top-k selection。

`DeepseekSparseAttentionIndexer.forward()` 不再展开 eager indexer 计算，而是提取 quant 或非 quant linear 权重，直接调用 `tensor_cast.dsa_indexer`。`DeepseekSparseAttention._get_backend_kwargs()` 统一向 MLA backend 传递：

- `topk_limit`
- `topk_indices`

`tensor_cast.multihead_latent_attention.default` 与 `tensor_cast.multihead_latent_attention_quant.default` 接收上述 sparse 信息，并在语义描述和性能模型中将 sparse attention mask/top-k 约束纳入 attention 长度估算。

### 性能模型设计

`tensor_cast/performance_model/__init__.py` 新增 `_estimate_dsa_indexer_breakdown(...)`，将 DSA Indexer 拆成可独立归类的 compute 和 memory bucket：

1. q projection MMA：`qa_normed @ wq_b`
2. k projection MMA：`hidden_states @ wk`
3. head routing projection MMA：`hidden_states @ weights_proj`
4. RoPE GP operations
5. FP8-only `rotate_activation` 与 activation quantization GP operations
6. q-k index score MMA
7. head weight multiply、head reduce、FP8-only relu/q-scale/k-scale shaping
8. top-k selection GP operations
9. key cache read/write bytes
10. FP8 scale cache read/write bytes

`dsa_indexer` 的性能注册根据 `indexer_cache.dtype` 区分 BF16/FP16 风格路径与 FP8 路径。这样同一个语义算子可以覆盖 GLM5 bf16 sparse MLA 和 DeepSeek FP8 sparse MLA，同时保持不同 dtype 下的成本估算差异。

此外，multihead latent attention 相关性能模型将 `seq_lens` 概念统一为 `request_total_seq_lens`，并在 `query_lens` 或 `request_total_seq_lens` 缺失时提供默认值。Sparse attention 的 prefill/decode 估算会根据 `topk_indices.shape[-1]` 或 `topk_limit` 限制实际 attention length。cache traffic 在有真实 request length 时使用逻辑请求长度，而不是直接使用预分配 cache buffer shape，避免长上下文预分配导致性能估算放大。

### RMSNorm 非默认 epsilon pattern matching

原有 RMSNorm pattern 将 `eps=1e-6` 固化在 Python 闭包中，导致 GLM5 中 `eps=1e-5` 等非默认值无法匹配。方案将 `eps` 改为 pattern/replacement 的显式 scalar 参数，并通过 `scalar_workaround={"eps": 1e-6}` 提供默认 tracing 输入。

涉及的 pattern 包括：

- `RMSNormPattern`
- `AddRMSNormPattern`
- `AddRMSNorm2Pattern`
- `AddRMSNormDynamicQuant2MXFP4Pattern`

`PatternMatchPass` 和 pattern 注册接口的 `example_inputs` 类型从 `List[torch.Tensor]` 放宽为 `List[Any]`，并透传 `scalar_workaround`。这样 pattern matching 能同时覆盖默认 epsilon 和 GLM5 非默认 epsilon 的 RMSNorm fusion 场景。

### KV cache dtype 与并行线性边界修正

`tensor_cast/core/input_generator.py` 中 DSA indexer cache dtype 改为读取 attention quant config：

- attention quant 为 FP8 时，indexer cache 使用对应 FP8 dtype。
- 非 FP8 或无 quant config 时，保持模型默认 dtype。

GLM-5.2 启用 IndexShare 时，`get_sparse_attention_indexer_cache_info(...)` 仅为 `full` source layer 分配 indexer cache，并据此累计 `indexer_cache_per_token`；`shared` layer 只读取已传递的 top-k，不拥有或别名 source cache。该层选择规则与 DeepSeek-V4 的 cache compression 独立：V4 `compress_ratio` 仅决定一个已分配 indexer cache 的 block 数，不能改变 GLM-5.2 哪些 layer 分配 cache 的结论。

`tensor_cast/layers/parallel_linear.py` 中 Row/Parallel linear 的 all-reduce 条件增加 `tp_group.world_size > 1` 保护，避免单卡或无实际 tensor parallel 切分时错误触发跨组 all-reduce 判断。该修正保证 GLM5 适配过程中的 MoE/MLA linear wrapper 在不同 parallel 配置下行为一致。

## 使用说明

### 模型使用方式

适配完成后，GLM5 系列模型可通过既有 TensorCast text generation CLI 走标准模型构建、编译和仿真流程。例如：

```bash
python -m cli.inference.text_generate zai-org/GLM-5.2 \
  --device ATLAS_800_A3_752T_128G_DIE \
  --num-devices 16 \
  --tp-size 16 \
  --dp-size 1 \
  --ep-size 16 \
  --context-length 61000 \
  --query-length 4096 \
  --num-queries 1 \
  --num-mtp-tokens 3 \
  --compile \
  --quantize-linear-action W4A8_STATIC \
  --dump-input-shapes
```

TensorCast 会通过 HF `AutoConfig` 识别 `model_type="glm_moe_dsa"`，加载 GLM5 原生配置，并通过 model profile 将 GLM5 attention/MoE/MTP 模块接入既有 transformation pipeline。启用 MTP、repetition 或 `torch.compile` 时，兼容 wrapper 仍会显式保留 IndexShare top-k state；因此 `throughput_optimizer` 的并行 TP/EP/MTP 搜索可复用无状态 MLP 区域，而不会因 decoder 辅助输出丢失导致模型构建或编译失败。

### Trace 与性能分析结果

GLM5 sparse MLA 路径中，trace 表应体现以下关键语义块：

- `tensor_cast.mlapo.default` 或 `tensor_cast.mlapo_quant.default`
- `tensor_cast.dsa_indexer.default`
- `tensor_cast.multihead_latent_attention.default` 或 `tensor_cast.multihead_latent_attention_quant.default`

对于 GLM-5.2 IndexShare，只有 `full` source layer 出现并计入 `tensor_cast.dsa_indexer.default`；`shared` layer 使用上游 top-k 进入 MLA backend，不应重复产生 indexer 算子、indexer latency 或 indexer cache traffic。旧的拆分式 `tensor_cast.dsa_index.default` / `tensor_cast.dsa_index_cache.default` 不再作为 DSA Indexer 主路径暴露。`dsa_indexer` 的性能模型会根据 `indexer_cache.dtype` 自动区分 FP8 与 BF16/FP16 行为，不需要用户新增 CLI 参数。

### 配置约束

1. GLM5 真实运行路径依赖 HF 原生 `GlmMoeDsaConfig` 的 `index_topk` 字段，不要求配置中存在 `topk_limit`。
2. DeepSeek-V3.2 本地配置继续通过 `topk_limit` 表达 sparse top-k 上限，并保留对旧 `index_topk` fixture 的兼容转换。
3. `dsa_indexer` 是 TensorCast shape/performance 语义算子，不执行真实数值 kernel；其职责是保持 trace 语义、shape 推导和性能估算一致。
4. `seq_lens` / `request_total_seq_lens` 可用于限制 active sequence length。FakeTensor tracing 时不会读取 tensor value，因此 top-k 输出宽度采用静态 `topk_limit`。
5. RMSNorm pattern 支持非默认 `eps`，但仍要求计算图结构符合已注册 RMSNorm/AddRMSNorm pattern。
6. GLM-5.2 的 `indexer_types` 必须覆盖所有参与编译的 decoder 和已扩展 MTP layer，且仅可包含 `full` 与 `shared`。每个 `shared` layer 必须能解析到前序 `full` source layer。
7. 当 HuggingFace 配置不可获取时，shape-grid 使用 `zai-org/GLM-5.2` 的静态 fallback config（`model_key="glm52"`）；其 MLA projection 维度沿用 GLM5 的 q/kv low-rank 配置。

## 测试设计

### 单元测试

1. **GLM5 / DeepSeek sparse top-k 配置兼容**
   - 验证 `DeepseekSparseAttentionIndexer.topk_limit` 可从 wrapper、indexer、config 或 GLM5 `index_topk` 正确解析。
   - 验证 DeepSeek config 忽略 GLM5-only 字段，并将旧 `index_topk` fixture 兼容到 `topk_limit`。
   - 验证 GLM5 风格配置在 `topk_limit=None` 时可以 fallback 到 `index_topk`。

2. **DSA Indexer 语义算子**
   - 验证 `DeepseekSparseAttentionIndexer.forward()` 调用 `torch.ops.tensor_cast.dsa_indexer`。
   - 验证 `slot_mapping`、`block_tables`、`seq_lens` 按约定顺序传入算子。
   - 验证 `dsa_indexer` 返回 query-major top-k shape：`(batch, seq_len, topk)`。
   - 验证 active sequence length 小于 `topk_limit` 时，非 FakeTensor 路径使用 `min(topk_limit, active_seq_len)`。
   - 验证提供 `seq_lens` 时算子仍可被 `torch.compile` tracing。

3. **性能模型**
   - 验证 `_estimate_dsa_indexer_breakdown(...)` 在 BF16 路径下的 MMA、GP、cache traffic 计算。
   - 验证 request total sequence length 会影响 score length 与 cache traffic。
   - 验证 FP8 路径额外包含 `rotate_activation`、activation quantization、scale cache traffic 和 FP8 score shaping 成本。
   - 验证 `multihead_latent_attention` 的 sparse top-k 限制会影响 prefill/decode attention length 估算。

4. **RMSNorm pattern matching**
   - 构造非默认 `eps` 的 RMSNorm module，验证 pattern replacement 后 runtime table 中出现：
     - `tensor_cast.rms_norm.default`
     - `tensor_cast.add_rms_norm.default`
     - `tensor_cast.add_rms_norm2.default`

5. **Input generator / cache dtype**
   - 验证 DSA indexer cache dtype 跟随 attention quant config。
   - 验证 FP8 attention quant 下使用 FP8 cache dtype。

6. **GLM-5.2 IndexShare**
   - 验证 full layer 执行 indexer，shared layer 复用 `prev_topk_indices`，且缺失该输入时失败。
   - 验证未知 indexer type、越界 layer index 和无 full source 的 shared 配置被拒绝。
   - 验证 MTP 扩展的新增 proposal block 使用独立的 full indexer。
   - 验证仅 full source layer 分配 indexer cache；启用 IndexShare 时只复用无状态 MLP，不将 decoder attention 包装为 `CopyLayerWrapper`。
   - 验证旧版 decoder 的兼容 wrapper 传递 `prev_topk_indices`、拒绝缺少第三项 top-k 的 attention 输出，并覆盖 `full -> shared -> shared` 链中 active top-k 的连续转发；验证已原生支持该参数的 decoder 不安装 wrapper，以及 config 默认 cache 行为保持一致。
   - 验证兼容 wrapper 与 `RegionMarkerWrapper`/`CopyLayerWrapper` 共存：代表 layer 的真实 owner 被包装，copy layer 保持不可执行状态，MTP 仅 patch 实际 block。
   - 验证 `zai-org/GLM-5.2` 可在 HuggingFace 查询失败时解析到静态 shape-grid config。

### 集成测试

重点运行以下回归测试覆盖 GLM5.2 IndexShare、MLA、输入 cache、layer repetition 和静态模型配置路径：

```bash
uv run --frozen --with pytest --with parameterized python -m pytest \
  tests/regression/tensor_cast/test_glm5.py \
  tests/regression/tensor_cast/test_mla.py \
  tests/regression/tensor_cast/test_input_generator.py \
  tests/regression/tensor_cast/test_repetition.py \
  tests/regression/cli/test_shape_grid_model_configs.py \
  tests/regression/cli/test_model_configs.py
```

GLM-5.2 IndexShare 提交的关联验证结果为 `99 passed, 2 deselected`。该结果覆盖 full/shared 执行、MTP `indexer_types` 扩展、仅 full layer 的 indexer cache 分配、repetition 隔离和 GLM-5.2 fallback config；不应与 GLM5 通用适配阶段的历史 `48 passed` 结果混用。

### 端到端验证

使用 GLM5 text generation compile 命令进行端到端验证，成功标准为：

1. 模型构建成功，HF `GlmMoeDsaConfig` 可被 TensorCast 识别。
2. 编译过程不再出现 `AttributeError("topk_limit")`。
3. runtime table 或 trace 的 full source layer 中出现 `tensor_cast.dsa_indexer.default`，shared layer 不重复执行该算子。
4. Sparse MLA 主路径中 `mlapo`/`mlapo_quant`、`dsa_indexer`、`multihead_latent_attention` 语义块顺序完整，并且 active top-k 可跨 `full -> shared -> shared` 等连续 shared 链路传递。
5. Indexer cache 仅为 full source layer 分配，性能结果不将 shared layer 计为新的 indexer work。
6. RMSNorm 非默认 epsilon block 能被融合为 TensorCast RMSNorm 语义算子。
7. 对不原生支持 `prev_topk_indices` 的 Transformers decoder，开启 `--compile`、repetition 和 MTP 后，compatibility loop 仍能保留 full/shared top-k 传递、cache/mask 行为和真实 MTP block；decoder layer 不被 copy replay，只有无状态 MLP 可复用。

该端到端验证用于证明 GLM-5.2 sparse attention / MLA / MoE compilation path 中的 DSA Indexer、IndexShare 与旧版 Transformers 兼容语义已经形成闭环。
