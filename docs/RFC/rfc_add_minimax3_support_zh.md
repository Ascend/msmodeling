# RFC: MiniMax-M3 模型适配支持

---

| 项目         | 内容                                |
| :----------- | :---------------------------------- |
| **状态**     | 已批准                              |
| **作者**     | —                                   |
| **创建日期** | 2026-07-25                          |
| **最后更新日期** | 2026-08-06                      |
| **相关链接** | [TensorCast 适配 MiniMax-M3 设计文档](../design/minimax_m3_adaptation_design.md) |

## 1. 概述

本 RFC 提议在 TensorCast 中增加 MiniMax-M3 / MiniMax-M3-VL 模型的仿真建模支持。MiniMax-M3 与当前仓库已有的 MiniMax-M2 相比，核心差异在于：

1. 使用分层 sparse attention 配置区分 dense attention layer 与 MiniMax3 sparse attention layer；旧版 Transformers 主要来自 `layer_types`，新版 Transformers 可能来自 `text_config.sparse_attention_config`。
2. Sparse attention layer 额外包含 learned indexer，用于为每个 query token 选择重要 KV block。
3. Attention 路径包含 indexer cache、`siso_reshape_and_cache`、`minimax_indexer`、`minimax_sparse_attention` 等 MiniMax3 专用语义算子。
4. Dense MLP 和 MoE expert 都使用 MiniMax3 风格的 fused `gate_up_proj` + OAI SwiGLU，需要拆分成 TensorCast 更容易识别和融合的 `gate_proj` / `up_proj` / `down_proj` 结构。
5. LayerNorm 使用 Gemma RMSNorm 语义，`weight` 的有效值是 `1 + weight`，且 decoder layer 内需要把 `residual + norm` 建模为 fused `add_rms_norm2`。

本方案目标是在不引入新的 `ModelProfile` 字段、不修改上游 Transformers 源码的前提下，通过最小模型 profile、模型局部 patch、MiniMax3 专用 wrapper 和少量 virtual op 完成端到端仿真。

---

## 2. 动机

MiniMax-M3 已经在新版 Transformers 中提供原生 `minimax_m3_vl` 模型源码。如果 TensorCast 只依赖通用模型适配逻辑，会遇到以下问题：

1. **Sparse attention 无法按真实链路建模**：普通 attention 只能表达 Q/K/V + attention body，无法表达 MiniMax3 的 indexer 打分、TopK block 选择、sparse attention body。
2. **Indexer cache 与普通 KV cache 不同**：indexer 只缓存 index key，不缓存 value，因此需要 `siso_reshape_and_cache` 这类 single-input single-output cache op。
3. **MLP 结构不利于现有融合 pass 识别**：HF 源码中 dense MLP 和 expert MLP 都使用 fused `gate_up_proj`，直接 forward 会出现 matmul 后 `chunk`，不利于 `SinkSplitPass` 和 grouped matmul + SwiGLU 融合。
4. **RMSNorm 语义与普通 RMSNorm 有差异**：Gemma RMSNorm 需要 `1 + weight`，如果不显式处理，norm 数值语义和 profiling 算子边界都可能不一致。
5. **真实长度在 compile/meta tensor 下会丢失**：Roofline 需要使用每个 request 的真实 `seq_len/query_len/decode` 信息，否则 sparse attention 的上下文长度会退化为 meta tensor fallback，导致 decode/prefill 建模偏差。
6. **Transformers 配置字段随版本变化**：MiniMax3 sparse attention 字段可能从 `layer_types/index_*` 演进为 `sparse_attention_config.sparse_*`，如果 patch 不兼容新旧字段，会导致 sparse attention wrapper 或 MTP attention wrapper 漏替换。

因此需要针对 MiniMax3 增加一组模型内局部 patch 和专用语义算子。

---

## 3. 目标与非目标

### 3.1 目标

- 支持 `model_type=minimax_m3_vl` 的模型加载、patch、量化、shard、compile 和 analytic trace 输出。
- 支持 dense attention layer 与 sparse attention layer 的分层建模。
- 支持 MiniMax3 learned indexer 的 block score / TopK 选择成本建模。
- 支持 MiniMax3 sparse attention body 的 QK/PV、softmax、Q/O/topk/KV 访存建模。
- 支持 MiniMax3 OAI SwiGLU 及其量化后接 quantized down projection 的路径。
- 支持 MiniMax3 Gemma RMSNorm、Q/K/indexer norm、decoder layer `add_rms_norm2` 融合建模。
- 保持 `ModelProfile` 简洁，不新增 `ModelProfile` 属性，不把 MiniMax3 的特例扩散到通用 profile。
- 尽量复用现有 MoE、quant linear、shard、compile pass 能力，只在 MiniMax3 与现有结构不兼容处做局部适配。

### 3.2 非目标

- 不实现真实 NPU sparse attention kernel，只提供 TensorCast 语义 op 与 Roofline 成本模型。
- 不在 TensorCast 中复刻 Transformers 全量 MiniMax3 模型源码。
- 不把 vLLM/vLLM-Ascend 的部署 kernel 优化作为源码语义强行写死；相关差异仅作为 profiling calibration 或风险项记录。
- 不改变 MiniMax2、DeepSeek、GLM 等已有模型的 profile 和主要仿真链路。

---

## 4. 用例分析

### 4.1 Decode 仿真

典型命令：

```bash
python -m cli.inference.text_generate MiniMaxAI/MiniMax-M3 \
  --device ATLAS_800_A3_752T_128G_DIE \
  --num-devices 16 \
  --num-queries 8 \
  --query-length 1 \
  --context-length 4334 \
  --decode \
  --compile \
  --tp-size 16 \
  --ep-size 16 \
  --enable-shared-expert-tp \
  --dump-input-shapes \
  --quantize-linear-action DISABLED \
  --quantize-attention-action DISABLED
```

期望行为：

- Dense layer 出现 `tensor_cast.attention.default`。
- Sparse layer 出现 `siso_reshape_and_cache`、`tensor_cast.minimax_indexer.default`、`tensor_cast.minimax_sparse_attention.default`。
- Sparse attention 的长度使用真实 `seq_lens_values=[context_length] * batch`，而不是 meta tensor fallback。
- MoE 层中 routed experts 能正确触发 grouped matmul 路径。

### 4.2 Prefill 仿真

典型命令：

```bash
python -m cli.inference.text_generate MiniMaxAI/MiniMax-M3 \
  --device ATLAS1_350_425T_112G \
  --num-devices 8 \
  --num-queries 1 \
  --query-length 16384 \
  --context-length 115712 \
  --compile \
  --quantize-linear-action FP8 \
  --quantize-attention-action DISABLED \
  --tp-size 8 \
  --ep-size 8 \
  --dump-input-shapes \
  --enable-shared-expert-tp
```

期望行为：

- `minimax_indexer` 按所有语义可见历史 token/block 计算 score 和 TopK 近似成本。
- `minimax_sparse_attention` 按每个 query 可见 KV block 计算 sparse QK/PV 和 softmax GP。
- Q/O/topk metadata、K/V cache 访存、MMA/GP 分开建模，便于和实测 kernel 分项对齐。

---

## 5. 方案设计

### 5.1 整体架构

MiniMax3 适配分为六层：

1. **ModelProfile 注册层**：注册 `model_type=minimax_m3_vl`，指定 MoE block 名称、专家数量字段、自定义 expert adapter、gate router 和 patch method。
2. **模型 patch 层**：在 `patch_method_for_minimax_m3` 中完成 attention、RMSNorm、dense MLP、MoE gate 属性补齐和通用 MoE patch。
3. **Attention wrapper 层**：使用 `MiniMaxM3AttentionWrapper` 替换每层 `self_attn`，根据 `layer_types` 或 `sparse_attention_config` 路由 dense/sparse 两条路径；MTP 层超出主干配置长度时继承最后一层 sparse 配置。
4. **语义算子层**：新增 `minimax_indexer`、`minimax_sparse_attention`、`m3_swiglu`、`m3_swiglu_quant`、`fused_rope` 等 TensorCast op。
5. **性能模型层**：在 `performance_model/__init__.py` 中为 MiniMax3 专用 op 注册 Roofline 属性。
6. **输入生成层**：在 `AttentionMetadataTensorCast` 中携带 materialized `seq_lens_values/query_lens_values/is_decode_values`，供 sparse attention Roofline 使用真实长度。

### 5.2 ModelProfile 设计

当前注册：

```python
register_model_profile(
    ModelProfile(
        model_type="minimax_m3_vl",
        moe_module_name="MiniMaxM3VLSparseMoeBlock",
        moe_num_experts_key="num_local_experts",
        moe_gate_router=route_minimax_m3_gate,
        patch_method=patch_method_for_minimax_m3,
        language_layers_path_str="language_model.layers",
        custom_expert_module_type=MiniMaxM3MoeExpertMLP,
    )
)
```

设计原则：

- 只保留通用框架已支持且 MiniMax3 必需的字段。
- 不新增 `ModelProfile` 属性。
- `language_layers_path_str="language_model.layers"` 复用 Transformers 原生 MiniMax3-VL 结构。
- `custom_expert_module_type` 只解决 MiniMax3 expert fused `gate_up_proj` 与现有 MoE pass 不兼容的问题。

### 5.3 Patch Method 设计

`patch_method_for_minimax_m3` 的执行顺序：

```python
def patch_method_for_minimax_m3(model: TransformerModel) -> TransformerModel:
    model.model_config.cache_rotary_embedding = False
    model = patch_minimax_m3_attention(model)
    model = patch_minimax_m3_layernorm(model)
    model = patch_minimax_m3_dense_mlp(model)
    model = patch_minimax_m3_moe_gate_attrs(model)
    model = patch_moe(model)
    return model
```

各步骤含义：

- `cache_rotary_embedding=False`：MiniMax3 使用 partial/3D RoPE，通用 rotary cache rewrite 可能生成不兼容的 position embedding 形状。
- `patch_minimax_m3_attention`：把 HF attention 替换为 TensorCast 可建模的 dense/sparse wrapper。
- `patch_minimax_m3_layernorm`：替换 Gemma RMSNorm，并 monkey patch decoder layer forward 以产生 `add_rms_norm2`。
- `patch_minimax_m3_dense_mlp`：把 dense MLP 的 fused `gate_up_proj` 拆成 `gate_proj/up_proj`。
- `patch_minimax_m3_moe_gate_attrs`：把 MoE block 上的 `routed_scaling_factor` 补到 gate 对象，供自定义 router 读取。
- `patch_moe`：复用 TensorCast 通用 MoE patch、dispatch、grouped matmul、combine 逻辑。

当前实现还补充了两类配置兼容逻辑：

- `_get_minimax_m3_effective_text_config` 会从 `model.text_config`、`model._inner.hf_config`、`model.hf_config` 中找到真正包含 `hidden_size` 的语言模型 config，兼容 `hf_config.text_config` 这类嵌套结构。
- `_resolve_minimax_m3_sparse_attention_config` 同时支持旧字段 `layer_types/index_*` 和新版 `sparse_attention_config.sparse_*`，避免 Transformers 升级后误判没有 sparse layer。

### 5.4 Attention Wrapper 设计

`MiniMaxM3AttentionWrapper.forward` 显式拆分 attention 链路：

1. 主 attention 投影：`q_proj/k_proj/v_proj`。
2. Q/K norm：`q_norm/k_norm`，通过 wrapper 转为 `tensor_cast.rms_norm`。
3. RoPE：使用 `tensor_cast.fused_rope` 表达 Q/K partial RoPE。
4. KV cache 写入：使用 `tensor_cast.reshape_and_cache`。
5. Dense attention layer：直接发射 `tensor_cast.attention.default`。
6. Sparse attention layer：
   - indexer Q/K 投影与 norm；
   - indexer RoPE；
   - index K cache 写入：`tensor_cast.siso_reshape_and_cache`；
   - block 选择：`tensor_cast.minimax_indexer.default`；
   - sparse attention body：`tensor_cast.minimax_sparse_attention.default`；
   - 输出投影：`o_proj`。

MTP 层处理：

- 主干层按照 sparse 配置数组逐层判断。
- 如果 MTP 层 `layer_idx` 超出 sparse 配置数组长度，则继承最后一个主干层的 dense/sparse 类型。
- 这样可以让 `--num-mtp-tokens` 场景下的 MiniMax3 sparse attention 仍走 TensorCast wrapper，而不是回退到 HF 原始 attention view 逻辑。

这样设计的原因：

- projection、norm、RoPE、cache、o_proj 都是独立 profiling kernel 或已有 TensorCast op，应保留为显式算子。
- `minimax_indexer` 只覆盖 score/topk 选择边界。
- `minimax_sparse_attention` 只覆盖 sparse QK/PV/softmax body 边界。
- 避免把完整 attention 包成一个大 op，降低与实测 kernel 对齐和问题定位的难度。

### 5.5 Dense MLP 与 MoE Expert 设计

MiniMax3 HF 源码中 MLP 典型结构是：

```text
gate_up_proj(hidden)
chunk -> gate, up
OAI SwiGLU
down_proj
```

TensorCast 需要更接近现有 pass 可识别的结构：

```text
gate_proj(hidden)
up_proj(hidden)
m3_swiglu(gate, up)
down_proj
```

因此引入：

- `MiniMaxM3DenseMLPWrapper`：处理 dense MLP 层。
- `MiniMaxM3MoeExpertMLP`：处理 MoE expert 层。

量化路径：

```text
gate/up -> m3_swiglu_quant -> int8 activation + activation_scale -> TensorCastQuantLinear.down_proj
```

非量化路径：

```text
gate/up -> m3_swiglu -> down_proj
```

### 5.6 MoE Gate Router 设计

`route_minimax_m3_gate` 使用 MiniMax3 的 sigmoid top-k gate 语义：

1. 使用 gate weight 计算 `router_logits`。
2. TP 场景下按 token 维度 pad 后切分到当前 rank。
3. 调用 `tensor_cast.moe_gating_top_k_sigmoid`。
4. 使用 `routed_scaling_factor` 和 `e_score_correction_bias`。
5. 返回 `topk_indices` 和转换回 hidden dtype 的 `topk_weights`。

`patch_minimax_m3_moe_gate_attrs` 的作用是把 MoE block 上的 `routed_scaling_factor` 复制到 gate 上，因为通用 MoE router 接收到的是 gate 对象，不是完整 MoE block。

### 5.7 RMSNorm 与 RoPE 设计

RMSNorm 设计：

- `GemmaRMSNormFusedWrapper`：用于 decoder layer norm，使用 `effective_weight = 1 + weight`。
- `RMSNormFusedWrapper`：用于 Q/K norm 和 indexer Q/K norm。
- `_fused_decoder_layer_forward`：将 `residual + attention_output + post_attention_layernorm` 表达为 `tensor_cast.add_rms_norm2`。

RoPE 设计：

- `fused_rope` 是 profiling/meta op，不执行真实数值旋转。
- 返回原始 `query/key`，保持下游 tensor 有效，避免 `empty_like` 未初始化值传播。
- Roofline 由独立 fused_rope op 的输入输出 shape 表达。

### 5.8 Input Generator 与 Attention Metadata

MiniMax3 sparse attention 的 Roofline 需要真实长度：

- `seq_lens_values`：每个 request 的总可见长度。
- `query_lens_values`：每个 request 当前 query token 数。
- `is_decode_values`：每个 request 是否是 decode。

这些字段挂在 `AttentionMetadataTensorCast` 上，并由 `MiniMaxM3AttentionWrapper` 透传给 `minimax_indexer` 和 `minimax_sparse_attention`。

必要性：

- torch.compile / meta tensor 下，`seq_lens` 和 `query_lens` 可能是 FakeTensor，不能安全 `.tolist()`。
- 如果只用 tensor shape fallback，会把真实上下文长度误判为 query token 数，导致 decode 长上下文成本严重偏低。

### 5.9 Roofline 建模设计

#### 5.9.1 `m3_swiglu`

边界：

```text
gate, up -> MiniMax3 OAI SwiGLU output
```

建模：

```text
GP ops = numel(gate) * 7
```

含义：

- 近似覆盖 clamp、sigmoid、乘法、加法等逐元素操作。
- 不计入 gate/up projection 和 down projection matmul，这些由对应 linear/grouped matmul op 建模。

#### 5.9.2 `m3_swiglu_quant`

边界：

```text
gate, up -> OAI SwiGLU -> dynamic symmetric quant -> int8 activation + fp32 scale
```

建模：

- 先复用 `m3_swiglu` 的逐元素 GP。
- 再组合 `dynamic_quantize_symmetric` 的访存属性。
- scale shape 按最后一维量化：`scale_shape[-1] = 1`。

#### 5.9.3 `minimax_indexer`

边界：

```text
idx_q, index K cache, seq_lens, query_lens, block_table
  -> topk block indices
```

公式：

```text
index_qk_mma = sum_b(2 * Q_b * N * L_b * D)
block_reduce_gp = sum_b(Q_b * N * L_b)
topk_gp = ceil(log2(K)) * sum_b(Q_b * N * B_n)
bytes_total = idx_q bytes + visible index K bytes + block score bytes + topk bytes
```

变量：

- `Q_b`：第 b 个 request 的 query token 数。
- `L_b`：第 b 个 request 的可见总长度。
- `N`：indexer head 数。
- `D`：indexer head dim。
- `K`：topk blocks。
- `B_n=ceil(L_b/block_size)`：可见 block 数。

设计边界：

- index K cache 写入不计在 `minimax_indexer` 内，而由 `siso_reshape_and_cache` 单独建模。
- 不把部署 TopK tile shape 当成语义候选 block 数减少；语义上仍按所有可见 block 参与选择。

#### 5.9.4 `minimax_sparse_attention`

边界：

```text
query, KV cache, topk_idx, seq_lens, query_lens, block_table
  -> sparse attention output
```

公式：

```text
attended_pairs = sum_q(attended_tokens_q)
attn_mma = 4 * N_q * attended_pairs * D
attn_gp = 6 * N_q * attended_pairs
qo_bytes = 2 * dtype_size * T * N_q * D
topk_bytes = 4 * T * N_kv * K
kv_read_bytes = 2 * dtype_size * effective_attended_pairs * N_kv * D
```

变量：

| 变量 | 含义 | 量纲 |
| :--- | :--- | :--- |
| `N_q` | query head 数 | head |
| `N_kv` | KV head 数 | head |
| `T` | 当前 batch 中 query token 总数，即 `sum_b(Q_b)` | token |
| `D` | attention head dim | hidden dim |
| `K` | 每个 query 选择的 top-k KV block 数 | block |
| `attended_tokens_q` | 单个 query 实际可见/参与 attention 的 token 数 | token |
| `attended_pairs` | 全部 query 的语义 attention pair 总数，即 `sum_q(attended_tokens_q)` | token pair |
| `effective_attended_pairs` | 考虑 prefill kernel 内 KV block 复用后的有效 KV 读取 pair 数 | token pair |
| `dtype_size` | query/KV/output 数据类型的字节数 | byte |
| `B_n` | 可见 block 数，`ceil(L_b / block_size)` | block |
| `B_s` | sparse attention block size，即每个 KV block 包含的 token 数 | token/block |
| `context_len` | 当前 request 的上下文长度，与变量表中的 `L_b` 同义 | token |

其中当前 KV read 计算公式为：

```python
if _minimax_m3_is_prefill_request(Q_b, request_idx, is_decode_values):
    effective_attended_pairs = _estimate_minimax_m3_prefill_sparse_attention_kv_read_pairs(
        context_len,
        Q_b,
        K,
        B_s,
        attended_pairs,
    )
else:
    effective_attended_pairs = min(K * B_s, L_b)

kv_read_bytes = 2 * dtype_size * effective_attended_pairs * N_kv * D
```

---

## 6. 技术选型

### 6.1 选择 Wrapper + Virtual Op

选择原因：

- 不修改 Transformers 安装包源码。
- 保留原始模块参数和权重来源。
- 可以在 TensorCast graph 中显式标出关键算子边界。
- 便于 `--dump-input-shapes` 和 chrome trace 与实测 kernel 对齐。

替代方案：

- 直接运行 HF 原始 forward，让 torch graph 自然展开。

不选择原因：

- 原始 forward 中 `gate_up_proj + chunk` 不利于现有 MoE grouped matmul 融合。
- 原始 sparse attention 的 Python 逻辑和部署 kernel 边界不同，直接展开不利于 profiling 对齐。
- Indexer cache 和 sparse attention body 需要明确的语义 op 才能做独立 Roofline。

### 6.2 选择最小 ModelProfile

选择原因：

- MiniMax3 的特殊性主要在 attention/indexer/RMSNorm/MLP patch，不应污染 `ModelProfile`。
- 与 GLM、DeepSeek 等模型保持一致：通用字段表达通用行为，特例由 `patch_method` 和 wrapper 处理。

---

## 7. 兼容性与影响范围

### 7.1 对已有模型的影响

本方案预期不影响 MiniMax2、DeepSeek、GLM 等已有模型，原因：

- MiniMax3 profile 独立绑定 `model_type=minimax_m3_vl`。
- `patch_method_for_minimax_m3` 只在 MiniMax3 profile 命中时执行。
- 新增 op 只有 MiniMax3 wrapper 会调用。
- `m3_swiglu` 被加入 `SinkSplitPass` binary ops 只是让 MiniMax3 gate/up split 消费者与普通 `swiglu` 具备同等 rewrite 能力。

### 7.2 Transformers 版本要求

MiniMax3 依赖新版 Transformers 中的原生 `minimax_m3_vl` 源码，因此依赖约束需要提升到：

```text
Python>=3.10
torch>=2.5.0
transformers>=5.13.0,<5.14.0
CANN/Ascend 驱动：需与目标 Ascend NPU 软件栈配套，建议使用项目 CI 或部署环境已验证版本
```

升级后部分模型如 `deepseek_v32/deepseek_v4/mimo_v2_flash` 已被 Transformers 内置，需要 safe register 兼容，避免重复 `AutoConfig.register` 报错。

MiniMax3 sparse attention 配置解析同时兼容：

- 旧式 `layer_types`、`index_n_heads`、`index_head_dim`、`index_topk_blocks`、`index_block_size`、`index_local_blocks`。
- 新式 `sparse_attention_config.sparse_attention_freq`、`sparse_num_index_heads`、`sparse_index_dim`、`sparse_topk_blocks`、`sparse_block_size`、`sparse_local_block`。

新式 `sparse_attention_config` 字段约束如下：

| 字段 | 类型 | 默认值/回退 | 是否必填 | 取值约束 |
| :--- | :--- | :--- | :--- | :--- |
| `sparse_attention_freq` | `int` | 回退到 `layer_types` 推导 | 否 | 大于 0 时表示 sparse layer 出现频率；缺失时使用旧式逐层配置 |
| `sparse_num_index_heads` | `int` | 回退到 `index_n_heads` | 是 | 正整数，表示 indexer query/key head 数 |
| `sparse_index_dim` | `int` | 回退到 `index_head_dim` | 是 | 正整数，表示 indexer head dim |
| `sparse_topk_blocks` | `int` | 回退到 `index_topk_blocks` | 是 | 正整数，表示每个 query 选择的 KV block 数 |
| `sparse_block_size` | `int` | 回退到 `index_block_size` | 是 | 正整数，单位为 token；用于计算可见 block 数 |
| `sparse_local_block` | `int` | 回退到 `index_local_blocks` | 否 | 非负整数，表示强制保留的局部 block 数 |

---

## 8. 测试设计

### 8.1 单元测试

| 测试 | 验证点 |
| :--- | :--- |
| `test_minimax_indexer_uses_materialized_lengths_for_meta_tensors` | meta tensor 下使用真实 `seq_lens_values/query_lens_values` |
| `test_minimax_indexer_prefill_scores_all_visible_blocks` | prefill indexer 语义上按所有可见 block 参与 score/topk |
| `test_minimax_indexer_k_cache_read_is_shared_across_index_heads` | index K cache 读不随 index head 重复放大 |
| `test_minimax_indexer_perf_properties_use_input_dtype` | indexer Roofline 使用输入 dtype，而不是固定 fp32 |
| `test_minimax_sparse_attention_uses_materialized_context_length` | decode sparse attention 使用真实上下文长度，并将 KV read cap 到 `min(K * block_size, L_b)` |
| `test_minimax_sparse_attention_prefill_kv_pairs_use_selected_capacity_geometric_mean` | prefill sparse attention KV read 使用 selected capacity 与 semantic pairs 的几何平均 |
| `test_minimax_m3_nested_sparse_attention_config_is_resolved` | 新版 `sparse_attention_config` 能正确解析 |
| `test_m3_swiglu_bf16_has_gp_roofline_properties` | `m3_swiglu` GP ops 计数正确 |

### 8.2 建议验证命令

```bash
pytest tests/regression/tensor_cast/test_minimax_m3_sparse_attention.py -q
pre-commit run --files \
  tensor_cast/transformers/builtin_model/minimax_m3.py \
  tensor_cast/layers/minimax_m3_attention.py \
  tensor_cast/ops/minimax_m3_sparse_attention.py \
  tensor_cast/performance_model/__init__.py \
  tests/regression/tensor_cast/test_minimax_m3_sparse_attention.py
```

端到端 smoke：

```bash
python -m cli.inference.text_generate MiniMaxAI/MiniMax-M3 \
  --device TEST_DEVICE \
  --num-devices 1 \
  --num-queries 1 \
  --query-length 1 \
  --context-length 256 \
  --decode \
  --compile \
  --tp-size 1 \
  --ep-size 1 \
  --dump-input-shapes \
  --quantize-linear-action DISABLED \
  --quantize-attention-action DISABLED
```

---

## 9. 风险与应对

### 9.1 Sparse attention Roofline 与实测 kernel 边界不完全一致

风险：

- 部署侧可能将 score/topk、sparse QK/PV、merge 等拆成多个 kernel，也可能融合成单个 ACLNN op。
- TensorCast 语义 op 边界与 profiling kernel 边界不总是一一对应。

应对：

- RFC 中明确 `minimax_indexer` 和 `minimax_sparse_attention` 的语义边界。
- 对比实测时按同一口径汇总 kernel，例如 score/topk 合并为 indexer，sparse attention body 合并为 sparse attention。
- 不把部署 tile shape 强行当作源码语义候选数量。

### 9.2 KV 访存复用近似可能需要 profiling calibration

风险：

- Prefill sparse attention kernel 可能存在 query tile 内 KV block 复用。
- 完全逐 query 独立读会偏高，只读物理 cache 一次会偏低。

应对：

- 当前采用折中 `effective_attended_pairs` 近似。
- 当前实现采用 selected capacity 与 semantic attended pairs 的几何平均；后续如获得 Ascend/vLLM-Ascend kernel 源码或稳定 profiling 数据，可将该近似升级为明确的 profiling calibration，并在代码注释和测试中标注。

### 9.3 Transformers 上游实现变化

风险：

- `minimax_m3_vl` 结构、字段名、layer type 字符串可能随 Transformers 升级变化。

应对：

- 依赖约束锁定在已验证范围 `>=5.13.0,<5.14.0`。
- patch 逻辑通过 `_get_minimax_m3_effective_text_config` 和 `_resolve_minimax_m3_sparse_attention_config` 同时兼容旧式 `layer_types/index_*` 与新式 `sparse_attention_config.sparse_*`。
- 升级 Transformers 时必须重新跑 MiniMax3 smoke 与 shape trace 对比。

### 9.4 Monkey patch decoder layer forward 的维护成本

风险：

- `_fused_decoder_layer_forward` 需要与上游 decoder layer forward 语义保持一致。

应对：

- 只做安全的 AddRMSNorm 融合，不改变 attention/MLP 主语义。
- 若上游 decoder forward 结构变化，需要重新核对该函数。

---

## 附录 A：术语表

| 术语 | 含义 |
| :--- | :--- |
| Dense Attention | 普通全量注意力层，走 `tensor_cast.attention.default` |
| Sparse Attention | MiniMax3 稀疏注意力层，先由 indexer 选择 KV block，再执行 sparse attention body |
| Indexer | MiniMax3 learned block selector，使用 index Q/K 为每个 query 选择重要历史 block |
| Index K Cache | indexer 专用 key cache，只缓存 index key，不缓存 value |
| `siso_reshape_and_cache` | single-input single-output cache 写入 op，用于 index K cache |
| `minimax_indexer` | TensorCast MiniMax3 语义 op，覆盖 block score 和 TopK 选择 |
| `minimax_sparse_attention` | TensorCast MiniMax3 语义 op，覆盖 sparse QK/PV/softmax body |
| MTP | Multi-Token Prediction，多 token 预测层；MiniMax-M3 的 MTP 层需要继承主干层 dense/sparse 配置以保持 wrapper 替换一致 |
| Roofline | 屋顶线性能模型，用计算量、访存量和硬件峰值估算算子性能上界 |
| FakeTensor/meta tensor | PyTorch 编译或 meta 执行中的无真实数据张量；只携带 shape/dtype，可能缺失真实序列长度 |
| SinkSplitPass | TensorCast 中用于把特定 binary/activation 消费者从上游融合结构中拆分出来的图改写 pass |
| grouped matmul | MoE expert 场景中按专家分组执行的矩阵乘，用于批量处理不同 expert 的 token |
| OAI SwiGLU | MiniMax3 使用的 SwiGLU 变体，带 `alpha/limit` 参数 |
| Gemma RMSNorm | 有效权重为 `1 + weight` 的 RMSNorm 语义 |
