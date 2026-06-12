## 修订记录
| 日期 | 修订版本 | 修改描述 | 作者 | RFC文档 |
| -- | -- | -- | -- | -- |
| 2026-04-30 | 1.0 | 初稿完成，归档 TensorCast 适配 GLM5 系列模型及 Sparse MLA/DSA Indexer 融合算子设计 | minghang_c | - |

## 背景描述

GLM5 系列模型在注意力、MoE 和归一化路径上与 TensorCast 已支持的 DeepSeek-V3.2 稀疏 MLA 路径高度相似，但存在若干关键差异。如果仅复用原有 DeepSeek 路径，TensorCast 在 GLM5 编译和仿真时会遇到以下问题：

1. GLM5 的 HuggingFace 配置类型为 `glm_moe_dsa`，模型结构中使用 `GlmMoeDsaAttention`、`GlmMoeDsaMoE` 和 `GlmMoeDsaDecoderLayer`，原有 TensorCast model profile 无法识别这些模块。
2. GLM5 sparse attention 配置使用 `index_topk` 表示 DSA indexer 的 top-k 上限，而 DeepSeek-V3.2 路径使用 `topk_limit`。直接访问 `topk_limit` 会在 GLM5 构建或 `torch.compile` 过程中触发属性缺失。
3. 原有 DSA indexer 路径由多个 eager ATen 算子和 `dsa_index`、`dsa_index_cache` 等分散语义算子组成，trace 中无法呈现真实硬件语义上的单个 LightningIndexer/DSA Indexer block，也不利于性能模型统一建模。
4. GLM5 decoder block 中 RMSNorm 使用非默认 `eps`，原有 pattern matching 将 `eps` 固定在 pattern 闭包内，导致非默认 epsilon 场景无法稳定融合到 `tensor_cast.rms_norm`、`tensor_cast.add_rms_norm`、`tensor_cast.add_rms_norm2`。
5. Sparse MLA 的 DSA indexer cache dtype 需要跟随 attention quantization 配置。FP8 attention 下 indexer cache 应使用 FP8 dtype，非 FP8/GLM5 bf16 路径下应保持 bf16/fp16 语义，否则性能模型和内存估算会偏离实际路径。

本特性的目标是让 TensorCast 能够正确编译和仿真 GLM5 系列模型，覆盖 sparse attention、MLA、MoE、MTP、RMSNorm fusion 以及 DSA indexer 相关性能估算。设计上尽量复用既有 DeepSeek sparse MLA 抽象，只在模型注册、配置兼容层、DSA indexer 语义算子和 pattern matching 能力上补齐差异，避免为 GLM5 复制一套独立实现。

## 方案设计

### 整体设计思路

本方案将 GLM5 适配拆分为四个层次：

1. **模型识别层**：新增 GLM5 model profile，将 HF `model_type="glm_moe_dsa"` 映射到 TensorCast 已有 MoE、MLA 和 MTP 包装机制。
2. **Sparse MLA 兼容层**：在 `DeepseekSparseAttention` / `DeepseekSparseAttentionIndexer` 中吸收 GLM5 与 DeepSeek 配置字段差异，统一向后端暴露 `topk_limit` 和 `topk_indices`。
3. **语义算子层**：用新的 `tensor_cast.dsa_indexer.default` 表达完整 DSA Indexer 语义块，替代拆散的 eager indexer 计算、cache 写入和 `dsa_index` / `dsa_index_cache` 流程。
4. **编译与性能建模层**：补齐 RMSNorm 非默认 epsilon pattern matching、DSA indexer analytic performance model、request length 感知的 sparse attention 估算，以及 indexer cache dtype 推导。

核心原则是：GLM5 不新增一套并行 MLA 实现，而是通过 model profile 接入既有 TensorCast sparse MLA wrapper；配置差异在 wrapper 边界收敛；trace 和性能模型统一面向 `dsa_indexer` 语义算子。

### 模型注册与模块映射

新增 `tensor_cast/transformers/builtin_model/glm5.py`，注册 `glm_moe_dsa` 模型画像：

- `moe_module_name="GlmMoeDsaMoE"`
- `moe_num_experts_key="n_routed_experts"`
- `moe_gate_returns_raw_logits=True`
- `mla_module_name="GlmMoeDsaAttention"`
- `mla_module_class_type=DeepseekSparseAttention`
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

`tensor_cast/layers/parallel_linear.py` 中 Row/Parallel linear 的 all-reduce 条件增加 `tp_group.world_size > 1` 保护，避免单卡或无实际 tensor parallel 切分时错误触发跨组 all-reduce 判断。该修正保证 GLM5 适配过程中的 MoE/MLA linear wrapper 在不同 parallel 配置下行为一致。

## 使用说明

### 模型使用方式

适配完成后，GLM5 系列模型可通过既有 TensorCast text generation CLI 走标准模型构建、编译和仿真流程。例如：

```bash
python -m cli.inference.text_generate "zai-org/GLM-5" \
  --device ATLAS_800_A3_752T_128G_DIE \
  --num-devices 64 \
  --tp-size 4 \
  --dp-size 16 \
  --ep-size 64 \
  --context-length 65500 \
  --query-length 1 \
  --num-queries 80 \
  --compile
```

TensorCast 会通过 HF `AutoConfig` 识别 `model_type="glm_moe_dsa"`，加载 GLM5 原生配置，并通过 model profile 将 GLM5 attention/MoE/MTP 模块接入既有 transformation pipeline。

### Trace 与性能分析结果

GLM5 sparse MLA 路径中，trace 表应体现以下关键语义块：

- `tensor_cast.mlapo.default` 或 `tensor_cast.mlapo_quant.default`
- `tensor_cast.dsa_indexer.default`
- `tensor_cast.multihead_latent_attention.default` 或 `tensor_cast.multihead_latent_attention_quant.default`

旧的拆分式 `tensor_cast.dsa_index.default` / `tensor_cast.dsa_index_cache.default` 不再作为 DSA Indexer 主路径暴露。`dsa_indexer` 的性能模型会根据 `indexer_cache.dtype` 自动区分 FP8 与 BF16/FP16 行为，不需要用户新增 CLI 参数。

### 配置约束

1. GLM5 真实运行路径依赖 HF 原生 `GlmMoeDsaConfig` 的 `index_topk` 字段，不要求配置中存在 `topk_limit`。
2. DeepSeek-V3.2 本地配置继续通过 `topk_limit` 表达 sparse top-k 上限，并保留对旧 `index_topk` fixture 的兼容转换。
3. `dsa_indexer` 是 TensorCast shape/performance 语义算子，不执行真实数值 kernel；其职责是保持 trace 语义、shape 推导和性能估算一致。
4. `seq_lens` / `request_total_seq_lens` 可用于限制 active sequence length。FakeTensor tracing 时不会读取 tensor value，因此 top-k 输出宽度采用静态 `topk_limit`。
5. RMSNorm pattern 支持非默认 `eps`，但仍要求计算图结构符合已注册 RMSNorm/AddRMSNorm pattern。

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

### 集成测试

重点运行以下测试文件覆盖 GLM5 适配涉及的主要路径：

```bash
python -m pytest tests/test_tensor_cast/test_mla.py tests/test_tensor_cast/test_runtime.py tests/test_tensor_cast/test_deepseek_v32.py -q
python -m pytest tests/test_tensor_cast/test_pattern_match.py -k non_default_eps -v
python -m pytest tests/test_tensor_cast/test_input_generator.py -q
```

历史提交中的关联验证结果包括：

```bash
python -m pytest tests/test_tensor_cast/test_pattern_match.py -k non_default_eps -v
python -m pytest tests/test_tensor_cast/test_pattern_match.py -k "rms_norm and not rope and not non_default_eps" -v
python -m pytest tests/test_tensor_cast/test_mla.py tests/test_tensor_cast/test_runtime.py tests/test_tensor_cast/test_deepseek_v32.py -q
```

结果为 `48 passed`。

### 端到端验证

使用 GLM5 text generation compile 命令进行端到端验证，成功标准为：

1. 模型构建成功，HF `GlmMoeDsaConfig` 可被 TensorCast 识别。
2. 编译过程不再出现 `AttributeError("topk_limit")`。
3. runtime table 或 trace 中出现 `tensor_cast.dsa_indexer.default`。
4. Sparse MLA 主路径中 `mlapo`/`mlapo_quant`、`dsa_indexer`、`multihead_latent_attention` 语义块顺序完整。
5. RMSNorm 非默认 epsilon block 能被融合为 TensorCast RMSNorm 语义算子。

该端到端验证用于证明 GLM5 sparse attention / MLA / MoE compilation path 与 DSA Indexer 融合语义已经形成闭环。