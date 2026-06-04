# 特性设计：TensorCast 适配 DeepSeek-V4 模型（Flash/Pro）

## 修订记录

| 日期 | 修订版本 | 修改描述 | 作者 | RFC 文档 |
| -- | -- | -- | -- | -- |
| 2026-06-03 | 1.0 | 初稿完成，归档 TensorCast 适配 DeepSeek-V4 模型的设计与实现 | — | — |

---

## 1. 背景描述

DeepSeek-V4（Flash/Pro）是 DeepSeek 系列模型的最新版本，在 V3/V3.2 基础上引入了多项关键架构升级。如果直接复用原有 TensorCast V3.2 路径，V4 模型在编译和仿真时会遇到以下核心问题：

1. **Head Compression（HC）机制缺失**：V4 decoder 在 attention 和 FFN 前后增加了 HC pre/post 包装，使用 Sinkhorn-based 混合将 hidden state 从 `[B,S,D]` 展开到 `[B,S,Hc,D]`，再还原回去。原有 TensorCast 没有 HC 相关的语义算子和性能模型。

2. **分层的 KV 压缩策略**：V4 采用 per-layer 级别的 `compress_ratios` 配置，ratio=0 使用滑动窗口、ratio=4 增加索引式稀疏压缩（Indexer）、ratio=128 使用重度压缩。每个 ratio 对应不同的 KV cache 管理和稀疏注意力路径。

3. **共享 KV 注意力机制**：V4 的 attention 不再使用标准 MLA 的 kv_b 分解，而是使用单一 `wkv` 投影产生 shared KV 向量，配合 grouped O projection (`wo_a` / `wo_b`) 实现注意力计算。

4. **Lightning Indexer 增强**：V4 indexer 在 ratio=4 层执行 learned sparse attention selection，包含 FP4 activation quantization、rotate activation、compressor 和 quant_lightning_indexer 等新语义算子。

5. **Hash 路由 MoE**：V4 前 `num_hash_layers` 层使用 token-id based hash routing 替代 top-k softmax routing，对应 `moe_gating_top_k_hash` 语义算子。

6. **Clamped SwiGLU**：V4 expert 的 activation 使用 `clamp(gate) * SiLU(up)` 而非标准 SwiGLU，需要 `v4_clamped_swiglu` 语义算子。

7. **并行维度差异**：V4 的 O projection 使用独立的 `o_proj_tp_group`，与 attention 的 `tp_group` 可分离，需要在 TP plan 中单独处理。

本特性的目标是让 TensorCast 能够正确编译和仿真 DeepSeek-V4 Flash/Pro 模型，覆盖 HC 机制、分层 KV 压缩、shared KV attention、Lightning Indexer、hash routing MoE 和 clamped SwiGLU 等关键新特性。设计上尽量复用既有 V3.2 sparse MLA 抽象，只在新语义算子、HC wrapping、compressor 和 grouped O projection 等处补齐差异。

---

## 2. 方案设计

### 2.1 整体设计思路

本方案将 DeepSeek-V4 适配拆分为五个层次：

1. **模型注册层**：新增 `deepseek_v4` model profile，将 HF `model_type="deepseek_v4"` 映射到 TensorCast V4 包装机制。

2. **HC 语义算子层**：用 `hc_pre_inv_rms`、`hc_pre_sinkhorn`、`hc_post`、`hc_head` 四个语义算子表达完整的 Head Compression 流程，覆盖 Sinkhorn 混合、weighted reduction 和 HC head reduction。

3. **KV 压缩算子层**：用 `compressor` 表达 V4 分层 KV 压缩，`scatter_nd_update_mla` 表达滑动窗口 KV 写入。

4. **稀疏注意力算子层**：用 `quant_lightning_indexer` 表达 ratio=4 learned indexer，用 `sparse_attn_sharedkv` 表达共享 KV sparse attention。

5. **MoE 路由算子层**：用 `moe_gating_top_k` 和 `moe_gating_top_k_hash` 区分非 hash 和 hash 路由，用 `v4_clamped_swiglu` 表达 clamped SwiGLU activation。

### 2.2 模型注册与配置

新增 `tensor_cast/transformers/builtin_model/deepseek_v4.py`，注册 `deepseek_v4` 模型画像：

```python
ModelProfile(
    model_type="deepseek_v4",
    moe_module_name="DeepseekV4MoE",
    moe_num_experts_key="n_routed_experts",
    moe_gate_returns_raw_logits=False,
    mtp_block_module_name="DeepseekV4DecoderLayer",
    mla_module_name="DeepseekV4SparseAttention",
    mla_field_names_override=MlaFieldNames(kv_b_proj=None),
    mla_module_class_type=DeepseekV4SparseAttention,
    patch_method=patch_method_for_deepseek_v4,
    moe_gate_router=route_deepseek_v4_gate,
)
```

`DeepseekV4Config` 子类化 `DeepseekV3Config`，新增 V4 特有字段：

- `compress_ratios` / `layer_types`：per-layer KV 压缩策略
- `topk_limit` / `index_topk`：Lightning indexer top-k 上限
- `num_hash_layers`：使用 hash routing 的 MoE 层数
- `hc_mult`、`hc_sinkhorn_iters`、`hc_eps`：HC shape 和 Sinkhorn 配置
- `o_groups`、`o_lora_rank`：grouped O projection 形状
- `score_func`、`route_scale`：V4 路由语义
- `expert_dtype`：FP4 expert quant 选择
- `swiglu_limit`、`compress_rope_theta`：expert clamping 和压缩 RoPE

配置验证包括：

- `compress_ratios` 必须定义，长度不少于 `num_hidden_layers`
- 仅支持 ratio ∈ {0, 4, 128}
- `layer_types` 必须与 `compress_ratios` 一致

### 2.3 Head Compression 语义算子

V4 decoder 在 attention 和 FFN 前后各包装一次 HC pre/post：

- **HC Pre** (`hc_pre_inv_rms` + `hc_pre_sinkhorn`)：将 `[B,S,D]` 展开为 `[B,S,Hc,D]`，通过 Sinkhorn-based 混合生成 `pre`、`post`、`comb`，最后用 `pre` 进行 weighted sum reduction 回到 `[B,S,D]`。

- **HC Post** (`hc_post`): 计算 `post.unsqueeze(-1) * x + sum(comb * residual, dim=hc)`，residual 被折叠进输出，不需要额外 `+ residual`。

- **HC Head** (`hc_head`): 模型输出层将 `[B,S,Hc,D]` 还原为 `[B,S,D]`，包含 RMS + linear + sigmoid + weighted reduction。

关键设计决策：

- `hc_pre_sinkhorn` 将 Sinkhorn iterations 和 weighted reduction 合并为一个语义算子，确保成本模型一起统计
- `hc_post` 的 residual 被折叠进 op 输出，避免 caller 重复累加 residual

### 2.4 KV 压缩与注意力

#### 2.4.1 分层压缩策略

V4 的 per-layer `compress_ratio` 决定 KV 缓存和注意力模式：

| ratio | 层类型 | KV 缓存 | 注意力 |
|-------|--------|---------|--------|
| 0 | sliding_attention | 滑动窗口 KV | 窗口 top-k |
| 4 | compressed_sparse_attention | 窗口 + 索引压缩 KV | 窗口 + indexer top-k |
| 128 | heavily_compressed_attention | 窗口 + 重度压缩 KV | 窗口 + 压缩 top-k |

#### 2.4.2 Compressor 语义算子

`tensor_cast.compressor` 表达 V4 压缩器：

- 对 hidden_states 执行 `wkv`、`wgate` 投影
- 生成 coarse-grained KV stream
- 对最后 `rope_head_dim` 维度执行 RoPE（主压缩器）或 Hadamard+FP4（索引压缩器）
- 写入 kv_cache 并返回 `compressed_kv`

成本模型覆盖：

- prefill：完整压缩成本，按 `query_len` 分段统计
- decode：单 token 成本，根据 `start_pos` 是否整除 compress_ratio 决定是否压缩

#### 2.4.3 Scatter KV 写入

`tensor_cast.scatter_nd_update_mla` 将 window KV 写入滑动窗口缓存：

- prefill short（`sl <= W`）：单次 scatter
- prefill split tail（`sl > W`）：两次 scatter 处理循环缓冲语义
- decode：单行 scatter

返回功能句柄供后续 `sparse_attn_sharedkv` 使用，建立完整 KV producer 链。

#### 2.4.4 Sparse Attention with Shared KV

`tensor_cast.sparse_attn_sharedkv` 执行共享 KV sparse attention：

- 有效 KV 长度由 `topk_indices.shape[-1]` 限制
- 使用 online softmax block（block=64）实现
- 支持 attention sink

成本模型：

- 两个 GEMM per iter（Q*K^T 和 S*V）
- per-iter scalar work（max/reduce/scale）
- KV gather traffic 按实际 topk 宽度统计

### 2.5 Lightning Indexer

`tensor_cast.quant_lightning_indexer` 表达 ratio=4 learned indexer：

**输入**（wrapper 已明确执行的阶段）：

- `q_states`：RoPE 后的 q tensor
- `weights`：weights_proj 输出
- `indexer_cache`：索引压缩 KV cache

**语义**（在语义算子内完成）：

1. `rotate_activation(q)` + `fp4_act_quant(q)`：pointwise 激活和 FP4 量化
2. `einsum("bshd,btd->bsht", q, kv_cache)`：qK score
3. `relu + weighted sum across heads`
4. `all_reduce_sum`（TP > 1）
5. `prefill: mask += where(causal_mask, -inf, 0)`
6. `topk(min(topk_limit, active_seq_len))`
7. `prefill: validity_mask postprocess` / `decode: offset`

输出宽度使用动态 `min(topk_limit, active_seq_len)` 而非固定 topk_limit。

### 2.6 MoE 路由

#### 2.6.1 非 Hash 路由（`moe_gating_top_k`）

覆盖 V4 非 hash MoE 层的 top-k 路由尾部：

- bias add（可选）
- topk on scores
- weight gather from pre-bias scores
- normalize（当 score_func != softmax）
- route_scale multiply

#### 2.6.2 Hash 路由（`moe_gating_top_k_hash`）

覆盖 V4 hash 层的 token-id based 路由：

- hash-table expert lookup（tid2eid[input_ids]）
- weight gather from scores
- normalize（可选）
- route_scale multiply

#### 2.6.3 Clamped SwiGLU

`tensor_cast.v4_clamped_swiglu` 模型 V4 expert activation：

- `clamp(gate) * SiLU(up)` 而非标准 `SiLU(gate) * up`

### claude 性能模型设计

`tensor_cast/performance_model/builtin_model/deepseek_v4.py` 为 V4 语义算子提供成本模型：

| 算子 | 主要成本 |
|------|----------|
| `hc_pre_inv_rms` | bf16→fp32 cast + RMS inverse |
| `hc_pre_sinkhorn` | Sinkhorn iterations + weighted reduction |
| `hc_post` | HC 维度 contraction + residual fold |
| `hc_head` | RMS + linear + sigmoid + weighted reduction |
| `scatter_nd_update_mla` | KV 行写入 |
| `compressor` | 投影 MMA + softmax + norm + RoPE |
| `quant_lightning_indexer` | FP4 quant + einsum + all_reduce + topk |
| `sparse_attn_sharedkv` | GEMM + online softmax + KV gather |
| `moe_gating_top_k` | bias add + topk + gather + normalize |
| `moe_gating_top_k_hash` | hash lookup + gather + normalize |
| `v4_clamped_swiglu` | clamp + SiLU + multiply |

---

## 3. 使用说明

### 3.1 运行环境

- Python ≥ 3.10
- PyTorch ≥ 2.5（支持 `torch.compile`）
- 需要 Hugging Face 连接

### 3.2 仿真命令

DeepSeek-V4 Flash/Pro 仿真示例：

```bash
python -m cli.inference.text_generate "deepseek-ai/DeepSeek-V4-Flash" \
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

TensorCast 会通过 HF `AutoConfig` 识别 `model_type="deepseek_v4"`，加载 V4 原生配置，并通过 model profile 将 V4 attention/MoE/HC 模块接入既有 transformation pipeline。

### 3.3 Trace 与性能分析结果

V4 sparse attention 路径中，trace 表应体现以下关键语义块：

- `tensor_cast.hc_pre_inv_rms.default` / `tensor_cast.hc_pre_sinkhorn.default`
- `tensor_cast.rms_norm.default`
- `tensor_cast.DeepseekV4SparseAttention` wrapper 内：
  - `tensor_cast.scatter_nd_update_mla.default`
  - `tensor_cast.compressor.default`（ratio > 0）
  - `tensor_cast.quant_lightning_indexer.default`（ratio=4）
  - `tensor_cast.sparse_attn_sharedkv.default`
- `tensor_cast.hc_post.default`
- `tensor_cast.hc_pre_*`（FFN 阶段）
- `tensor_cast.moe_gating_top_k.default` / `tensor_cast.moe_gating_top_k_hash.default`

### 3.4 配置约束

1. V4 必须定义 `compress_ratios`，每个 ratio 值必须为 0、4 或 128。
2. `layer_types` 必须与 `compress_ratios` 一致。
3. `o_proj_tp_group.world_size` 必须 ≤ `o_groups`，否则抛出错误。
4. Hash routing 层数由 `num_hash_layers` 控制，默认 0。
5. `swiglu_limit > 0` 时使用 `v4_clamped_swiglu`，否则使用标准 SiLU。

---

## 4. 测试设计

### 4.1 单元测试

1. **V4 配置解析**
   - 验证 `compress_ratios` / `layer_types` 验证逻辑
   - 验证 ratio ∈ {0, 4, 128} 约束
   - 验证 `index_topk` → `topk_limit` 兼容

2. **HC 语义算子**
   - 验证 `hc_pre_*`、`hc_post`、`hc_head` 在 trace 中出现
   - 验证 HC 维度展开/还原 shape 一致性
   - 验证 Sinkhorn iterations 参数传递

3. **V4 Attention Wrapper**
   - 验证 `DeepseekV4SparseAttention` 正确处理 ratio=0/4/128
   - 验证 `scatter_nd_update_mla` 返回功能句柄
   - 验证 `sparse_attn_sharedkv` 接收正确 topk_indices

4. **Lightning Indexer**
   - 验证 `quant_lightning_indexer` 动态 topk 宽度
   - 验证 indexer_cache dtype 和 layout

5. **MoE 路由**
   - 验证 hash routing 使用 `tid2eid` lookup
   - 验证 `v4_clamped_swiglu` 行为

### 4.2 集成测试

```bash
python -m pytest tests/test_tensor_cast/test_deepseek_v4.py -v
python -m pytest tests/test_tensor_cast/test_mla.py -v
python -m pytest tests/test_tensor_cast/test_moe_layer.py -v
```

### 4.3 成功标准

1. 模型构建成功，HF `DeepseekV4Config` 可被 TensorCast 识别。
2. 编译过程不再出现 HC 相关属性缺失。
3. trace 中出现 `hc_pre_*`、`hc_post`、`hc_head` 等 HC 语义算子。
4. ratio=4 层出现 `quant_lightning_indexer`。
5. 已有模型（DeepSeek-V3.2、GLM-5 等）测试不受影响。
