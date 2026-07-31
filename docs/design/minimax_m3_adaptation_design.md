# 特性设计：TensorCast 适配 MiniMax-M3 / MiniMax-M3-VL 模型

## 修订记录

| 日期 | 版本 | 修改描述 | 作者 | 关联文档 |
| --- | --- | --- | --- | --- |
| 2026-07-27 | 1.0 | 初稿完成，归档 MiniMax-M3 在 TensorCast 中的模型适配、算子边界、Roofline 建模和验证方案 | - | - |

## 1. 背景描述

MiniMax-M3 / MiniMax-M3-VL 是 MiniMax 系列中和 MiniMax-M2 差异较大的模型。当前 Transformers 已经提供原生 `model_type="minimax_m3_vl"` 的模型源码，因此 TensorCast 不需要复制完整上游模型实现，但仍需要在模型结构进入 TensorCast 编译、trace 和 Roofline 之前补齐若干关键语义边界。

如果直接复用 MiniMax-M2 或通用 MoE/Attention 适配路径，MiniMax-M3 会遇到以下问题：

1. **Sparse Attention 链路缺失**：MiniMax-M3 的部分 decoder layer 不是普通 dense attention，而是通过 learned indexer 为每个 query token 选择重要 KV block，再执行 sparse attention body。通用 attention op 无法表达 indexer score、TopK block 选择和 sparse QK/PV 边界。

2. **Indexer cache 与普通 KV cache 不同**：普通 attention 缓存 K/V 两路，MiniMax-M3 indexer 只需要缓存 index key，不存在 index value，因此需要 `siso_reshape_and_cache` 表达 single-input single-output cache 写入。

3. **Fused `gate_up_proj` 不利于现有 MoE/MLP pass 识别**：Transformers MiniMax-M3 源码中 dense MLP 和 MoE expert 都使用 fused `gate_up_proj(hidden)`，再 `chunk` 成 gate/up。TensorCast 现有 `SinkSplitPass`、grouped matmul + SwiGLU 融合更容易识别 split `gate_proj` / `up_proj` 结构。

4. **OAI SwiGLU 语义不同于普通 SwiGLU**：MiniMax-M3 使用带 `alpha`、`limit` 参数的 OAI SwiGLU 变体，且量化场景下需要把 activation scale 传给后续 quantized down projection。

5. **Gemma RMSNorm 语义需要显式建模**：MiniMax-M3 的 RMSNorm 有 `1 + weight` 的有效权重语义，decoder layer 内还需要把 residual add + norm 显式融合为 `add_rms_norm2`，方便 trace 和实测算子边界对齐。

6. **真实长度在 `torch.compile` / meta tensor 下容易丢失**：MiniMax-M3 sparse indexer 和 sparse attention 的成本强依赖每个 request 的 `seq_len`、`query_len`、decode/prefill 阶段。如果只从 FakeTensor shape fallback，会把长上下文 decode 或长 prefill 建模错误。

7. **Transformers 新旧配置字段不完全一致**：部分版本使用 `layer_types` 与 `index_*` 字段描述 sparse attention；升级到较新 Transformers 后，相关字段可能嵌套在 `text_config.sparse_attention_config` 中，字段名也变为 `sparse_*`。如果 patch 只识别旧字段，会导致 MiniMax3 attention wrapper 被跳过。

本设计目标是在保持 `ModelProfile` 最小化、不新增通用 profile 字段、不修改 Transformers 安装包源码的前提下，使 TensorCast 能稳定执行 MiniMax-M3 的 decode/prefill 仿真命令，并在 trace 中输出可与实测 kernel 组合对齐的 MiniMax3 专用语义算子。

---

## 2. 方案设计

### 2.1 整体设计思路

MiniMax-M3 适配分为六层：

1. **模型注册层**：通过 `ModelProfile(model_type="minimax_m3_vl")` 注册最小必要 profile，仅声明 MoE block、专家数字段、gate router、patch method 和自定义 expert adapter。

2. **模型 patch 层**：通过 `patch_method_for_minimax_m3` 在加载后替换 attention、RMSNorm、dense MLP，并补齐 MoE gate 属性，再复用通用 `patch_moe`。

3. **Attention wrapper 层**：使用 `MiniMaxM3AttentionWrapper` 包装每层 `self_attn`。dense layer 走普通 attention；sparse layer 显式发射 indexer cache、`minimax_indexer`、`minimax_sparse_attention`。

4. **MLP/MoE expert adapter 层**：将 Transformers 源码中的 fused `gate_up_proj` 拆成 TensorCast 更容易识别的 `gate_proj` / `up_proj` / `m3_swiglu` / `down_proj`。

5. **语义算子层**：新增或复用 `minimax_indexer`、`minimax_sparse_attention`、`m3_swiglu`、`m3_swiglu_quant`、`fused_rope`、`siso_reshape_and_cache` 等 virtual op，使 trace 中的核心边界可见。

6. **性能模型层**：在 `tensor_cast/performance_model/__init__.py` 为 MiniMax3 专用 op 注册 Roofline 属性，区分 MMA、GP、Q/O/topk metadata、K/V cache 访存和通信边界。

整体链路如下：

```text
HF MiniMax-M3-VL model
  -> TensorCast ModelProfile(minimax_m3_vl)
  -> patch_method_for_minimax_m3
  -> MiniMaxM3AttentionWrapper / MLP Wrapper / MoE Expert Adapter
  -> TensorCast virtual ops
  -> AnalyticPerformanceModel Roofline
  -> op-bound table / chrome trace / shape dump
```

### 2.2 模型注册与 profile

当前注册入口位于 `tensor_cast/transformers/builtin_model/minimax_m3.py`：

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

字段设计原则：

- `model_type="minimax_m3_vl"` 与 Transformers config 对齐。
- `moe_module_name` 指向上游 MiniMax-M3 的 sparse MoE block。
- `moe_num_experts_key="num_local_experts"` 复用通用 MoE 框架读取专家数。
- `moe_gate_router=route_minimax_m3_gate` 用于 MiniMax3 sigmoid top-k routing。
- `patch_method` 只承载 MiniMax3 的结构差异，不新增 `ModelProfile` 通用字段。
- `custom_expert_module_type=MiniMaxM3MoeExpertMLP` 只解决 MiniMax3 expert fused `gate_up_proj` 的结构问题。

### 2.3 Patch Method

Patch method 当前设计如下：

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

各步骤职责：

| 步骤 | 作用 | 必要性 |
| --- | --- | --- |
| `cache_rotary_embedding=False` | 关闭通用 rotary cache rewrite | MiniMax3 有 partial/3D RoPE，通用缓存形状可能不兼容 |
| `patch_minimax_m3_attention` | 替换每层 `self_attn` 为 `MiniMaxM3AttentionWrapper` | 显式区分 dense attention 与 sparse attention |
| `patch_minimax_m3_layernorm` | 替换 RMSNorm wrapper，并 patch decoder layer forward | 建模 Gemma RMSNorm 和 `add_rms_norm2` |
| `patch_minimax_m3_dense_mlp` | 替换 dense MLP fused `gate_up_proj` | 让 dense MLP 与 MoE expert 使用统一 split gate/up 结构 |
| `patch_minimax_m3_moe_gate_attrs` | 把 MoE block 的 `routed_scaling_factor` 挂到 gate | 通用 router 只拿到 gate 对象，需要从 gate 读取缩放因子 |
| `patch_moe` | 复用 TensorCast 通用 MoE patch | 继续使用 dispatch、grouped matmul、combine 等现有能力 |

### 2.4 Attention Wrapper 设计

`MiniMaxM3AttentionWrapper` 是 MiniMax3 与 MiniMax2 最大差异的核心。它不是重新实现完整 attention 数值，而是把 Transformers 原始 attention 拆成 TensorCast 可建模边界。

#### 2.4.0 Sparse attention 配置解析

当前 `patch_minimax_m3_attention` 不再只依赖 `layer_types`，而是先通过 `_get_minimax_m3_effective_text_config` 找到真正的语言模型 config，再通过 `_resolve_minimax_m3_sparse_attention_config` 统一解析 sparse attention 配置：

```text
优先路径：
  text_config.layer_types
  text_config.index_n_heads / index_head_dim / index_topk_blocks / index_block_size / index_local_blocks

兼容路径：
  text_config.sparse_attention_config.sparse_attention_freq
  text_config.sparse_attention_config.sparse_num_index_heads
  text_config.sparse_attention_config.sparse_index_dim
  text_config.sparse_attention_config.sparse_topk_blocks
  text_config.sparse_attention_config.sparse_block_size
  text_config.sparse_attention_config.sparse_local_block
```

这样做的原因是 Transformers 版本升级后，MiniMax3-VL config 可能出现 `hf_config.text_config` 这类嵌套结构。如果只从 `model.text_config` 直接读取旧字段，可能误判“没有 sparse layer”，进而让 sparse attention 和 MTP attention 走回 HF 原始实现。

MTP 处理策略：

- 主干 decoder 层按自己的 sparse/dense 配置判断。
- 如果 MTP 层的 `layer_idx` 超过 sparse 配置数组长度，则继承最后一个主干层的 sparse 配置。
- 这样可以避免 MTP sparse attention 未被 wrapper 替换，导致 TP-sharded indexer q/k shape 与 HF 原始 view 逻辑不匹配。

#### 2.4.1 Dense attention layer

Dense layer 走常规链路：

```text
hidden_states
  -> q_proj/k_proj/v_proj
  -> q_norm/k_norm
  -> fused_rope
  -> reshape_and_cache
  -> tensor_cast.attention.default
  -> o_proj
```

Dense layer 的目的不是引入 MiniMax3 专用 sparse op，而是保证 Q/K norm、RoPE、cache 写入和 output projection 在 trace 中仍然显式可见。

#### 2.4.2 Sparse attention layer

Sparse layer 在普通 Q/K/V attention 链路之外增加 indexer 分支：

```text
hidden_states
  -> q_proj/k_proj/v_proj
  -> q_norm/k_norm
  -> fused_rope
  -> reshape_and_cache

  -> indexer.q_proj / indexer.k_proj
  -> indexer q/k norm
  -> indexer fused_rope
  -> siso_reshape_and_cache
  -> tensor_cast.minimax_indexer.default
  -> tensor_cast.minimax_sparse_attention.default
  -> o_proj
```

设计边界：

- `minimax_indexer` 只表达 index score、block reduce、TopK block 选择。
- `minimax_sparse_attention` 只表达 sparse QK/PV/softmax attention body。
- Q/K/V projection、norm、RoPE、cache write、o_proj 都保留为独立 op，避免一个大 op 吃掉过多边界。

### 2.5 Indexer Cache 与 `siso_reshape_and_cache`

普通 KV cache 的输入包含 K 和 V，shape 典型为：

```text
KV cache: [2, num_blocks, block_size, kv_heads, head_dim]
```

MiniMax3 indexer 只需要 index key cache，不需要 index value，因此使用 single-input single-output cache：

```text
index_k:      [T, indexer_heads, index_head_dim]
index cache:  [num_blocks, block_size, index_head_dim]
```

`siso_reshape_and_cache` 的设计目的：

- 表达 index key 写入 cache 的独立成本。
- 避免错误地为 indexer 分支引入不存在的 value cache。
- 让 trace 中 indexer cache 写入和 `minimax_indexer` 打分成本分开统计。

### 2.6 Dense MLP 与 MoE Expert

Transformers MiniMax3 的 MLP/Expert 原始结构是：

```text
gate_up_proj(hidden)
chunk -> gate, up
OAI SwiGLU
down_proj
```

TensorCast 适配后结构是：

```text
gate_proj(hidden)
up_proj(hidden)
m3_swiglu(gate, up)
down_proj
```

量化场景：

```text
gate_proj/up_proj
  -> m3_swiglu_quant
  -> int8 activation + activation_scale
  -> TensorCastQuantLinear.down_proj(external_activation_scale)
```

这样设计的原因：

- 和 DeepSeek、GLM 等已有 MoE expert split gate/up 结构保持一致。
- 让 grouped matmul + SwiGLU 融合 pass 能识别 gate/up/down 边界。
- 量化场景下将 activation scale 显式传给后续 quant linear，避免接口不匹配。

### 2.7 MiniMax3 MoE Gate Router

`route_minimax_m3_gate` 建模 MiniMax3 sigmoid top-k routing：

1. 使用 gate weight 计算 `router_logits`。
2. TP 场景下按 token 维 pad 并切到当前 rank。
3. 调用 `tensor_cast.moe_gating_top_k_sigmoid.default`。
4. 使用 `e_score_correction_bias` 和 `routed_scaling_factor`。
5. 返回 `topk_indices` 和转换回 hidden dtype 的 `topk_weights`。

与通用 MoE 的关系：

- dispatch、all-to-all、grouped matmul、combine 仍复用通用 MoE pipeline。
- MiniMax3 仅特化 gate routing 和 expert MLP 结构。

### 2.8 RMSNorm 与 RoPE

RMSNorm 设计：

- decoder layer norm 使用 `GemmaRMSNormFusedWrapper`，有效权重为 `1 + weight`。
- Q/K norm 和 indexer Q/K norm 使用 `RMSNormFusedWrapper`。
- decoder layer forward 中 residual add + post-attention RMSNorm 被建模为 `tensor_cast.add_rms_norm2.default`。

RoPE 设计：

- 使用 `tensor_cast.fused_rope.default` 表达 Q/K RoPE 边界。
- `fused_rope` 是 profiling/meta op，不执行真实数值旋转。
- stub 返回原始 query/key，避免 `empty_like` 未初始化值向下游传播。

---

## 3. Roofline 性能模型设计

### 3.1 `m3_swiglu`

边界：

```text
gate, up -> MiniMax3 OAI SwiGLU output
```

建模：

```text
GP ops = numel(gate) * 7
```

含义：

- 覆盖 clamp、sigmoid、逐元素乘加等 OAI SwiGLU pointwise 成本。
- 不覆盖 gate/up projection 和 down projection matmul，这些由 linear/grouped matmul 单独建模。

### 3.2 `m3_swiglu_quant`

边界：

```text
gate, up -> OAI SwiGLU -> dynamic symmetric quant -> int8 activation + fp32 scale
```

建模方式：

- 先复用 `m3_swiglu` 的逐元素 GP 成本。
- 再组合 `dynamic_quantize_symmetric` 的访存和 scale 输出成本。
- `alpha`、`limit`、`group_size` 当前作为接口兼容参数，不改变 meta shape 推导。

### 3.3 `minimax_indexer`

边界：

```text
idx_q, index K cache, seq_lens, query_lens, block_table
  -> topk block indices
```

变量：

| 变量 | 含义 |
| --- | --- |
| `Q_b` | 第 b 个 request 的 query token 数 |
| `L_b` | 第 b 个 request 的可见总长度 |
| `N` | 当前 rank 上的 indexer head 数 |
| `D` | indexer head dim |
| `K` | topk block 数 |
| `B_s` | block size |
| `B_n=ceil(L_b/B_s)` | 可见 block 数 |

公式：

```text
index_qk_mma = sum_b(2 * Q_b * N * L_b * D)
block_reduce_gp = sum_b(Q_b * N * L_b)
topk_gp = ceil(log2(K)) * sum_b(Q_b * N * B_n)
bytes_total = idx_q bytes + visible index K bytes + block score bytes + topk bytes
```

设计说明：

- indexer 语义上对可见历史 token/block 计算重要性分数。
- index K cache 写入由 `siso_reshape_and_cache` 单独建模，不计入 `minimax_indexer`。
- 当前不把部署侧 TopK tile shape 直接解释为语义候选 block 数减少，避免把 kernel 内部 tile 当成模型结构。
- compute dtype 使用 op 输入 dtype，而不是硬编码 fp32，用于反映仿真图里真实输入 dtype。

### 3.4 `minimax_sparse_attention`

边界：

```text
query, KV cache, topk_idx, seq_lens, query_lens, block_table
  -> sparse attention output
```

核心公式：

```text
attended_pairs = sum_q(attended_tokens_q)
attn_mma = 4 * N_q * attended_pairs * D
attn_gp = 6 * N_q * attended_pairs
qo_bytes = 2 * dtype_size * T * N_q * D
topk_bytes = 4 * T * N_kv * K
kv_bytes = 2 * dtype_size * effective_attended_pairs * N_kv * D
```

其中：

- `attn_mma` 系数 4 表示 QK 和 PV 两个 matmul，每个 FMA 按 2 FLOPs。
- `attn_gp` 用每个 attention score 约 6 个 GP ops 近似 softmax / scale / reduce。
- `qo_bytes` 覆盖 query 输入和 output 写出。
- `topk_bytes` 覆盖 TopK index metadata 读。
- `kv_bytes` 覆盖 K/V cache 读。

当前实现中的 KV read 计算公式为：

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

kv_read_bytes = 2 * s * effective_attended_pairs * N_kv * D
```

---

## 4. 使用说明

### 4.1 Decode 仿真命令

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

预期 trace 特征：

- 前 3 层 dense attention 出现 `tensor_cast.attention.default`。
- 后续 sparse layer 出现 `tensor_cast.siso_reshape_and_cache.default`、`tensor_cast.minimax_indexer.default`、`tensor_cast.minimax_sparse_attention.default`。
- MoE 层出现 sigmoid gating、dispatch、grouped matmul、SwiGLU、combine。
- sparse attention 长度使用真实 `seq_lens_values/query_lens_values/is_decode_values`。

### 4.2 Prefill 仿真命令

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

预期 trace 特征：

- `minimax_indexer` 的输入包含 index query、index K cache、seq_lens、query_lens、block_table。
- `minimax_sparse_attention` 的输入包含 query、K cache、V cache、topk_idx、seq_lens、query_lens、block_table。
- 如果开启 `--dump-op-bound-results`，应能看到 `minimax_indexer` 和 `minimax_sparse_attention` 分别归因到 compute/memory/GP 的占比。

---

## 5. 测试设计

### 5.1 单元测试

当前重点测试文件：

```text
tests/regression/tensor_cast/test_minimax_m3_sparse_attention.py
tests/regression/tensor_cast/test_transformations.py
```

建议覆盖点：

| 测试点 | 目标 |
| --- | --- |
| materialized length | meta tensor 下仍使用真实 `seq_lens_values/query_lens_values` |
| indexer score all visible blocks | prefill indexer 不把部署 tile shape 当语义候选减少 |
| index K cache read | index K cache read 不随 index head 重复放大 |
| input dtype | indexer compute ops 使用输入 dtype |
| sparse attention decode | decode sparse attention 使用真实上下文长度，并将 KV read cap 到 `min(K * block_size, L_b)` |
| sparse attention prefill | prefill sparse attention 的 KV read 使用 selected capacity 与 semantic pairs 的几何平均 |
| nested sparse config | Transformers 新版 `sparse_attention_config` 能正确解析 |
| MTP sparse layer | MTP 层超出主干 sparse 配置长度时继承最后一层配置，避免 wrapper 漏替换 |
| m3_swiglu | OAI SwiGLU GP ops 和量化路径属性正确 |

### 5.2 建议验证命令

```bash
pytest tests/regression/tensor_cast/test_minimax_m3_sparse_attention.py -q
pytest tests/regression/tensor_cast/test_transformations.py -q
pre-commit run --files \
  tensor_cast/transformers/builtin_model/minimax_m3.py \
  tensor_cast/layers/minimax_m3_attention.py \
  tensor_cast/ops/minimax_m3_sparse_attention.py \
  tensor_cast/performance_model/__init__.py \
  tests/regression/tensor_cast/test_minimax_m3_sparse_attention.py
```

### 5.3 成功标准

1. MiniMax-M3 decode 和 prefill 命令能完成 `--compile` 仿真。
2. `--dump-input-shapes` 能输出 dense attention、sparse indexer、sparse attention、MoE 的关键 shape。
3. `--dump-op-bound-results` 中 MiniMax3 专用 op 的调用次数与模型层类型一致。
4. MiniMax2、GLM、DeepSeek 等已有模型的通用 MoE/attention 路径不受 MiniMax3 patch 影响。
5. pre-commit 和相关 regression tests 通过。
