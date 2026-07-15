# Qwen3.5 仿真支持设计文档

状态 (Status): Draft
作者 (Authors): @yuyinkai
创建日期 (Created): 2026-05-19
更新日期 (Updated): 2026-06-11

---

## 1. 概述

### 1.1 简介

Qwen3.5 是 Qwen 系列的最新多模态大语言模型，引入了 **GatedDeltaNet（线性注意力）** 作为标准 Self-Attention 的替代方案，在部分层中使用线性复杂度注意力机制以降低长序列推理的计算开销。本 RFC 描述在 msmodeling 仿真框架中完整支持 Qwen3.5 模型（包括 Qwen3.5-27B、Qwen3.5-397B-A17B 等变体）的设计方案。

Qwen3.5 相比已支持的 Qwen3 系列，引入了以下新特性：

- **GatedDeltaNet 线性注意力**：包含 Chunk Gated Delta Rule（prefill）和 Recurrent Gated Delta Rule（decode）两条路径
- **CausalConv1d**：深度可分离因果卷积，用于序列状态管理
- **Gated RMSNorm**：带门控的 RMSNorm（`output * (1.0 + weight.float())`）
- **W8A8 静态量化**：DynamicQuant + QuantBatchMatmulV3 的全链路 INT8 推理
- **混合注意力架构**：同一模型中部分层使用标准 FlashAttention，部分层使用 GatedDeltaNet
- **多模态视觉编码器**：VL 变体包含视觉编码器，shape 与纯文本模型不同
- **MTP（Multi-Task Prediction）**：多 token 预测，影响 cache_position 元数据和 lm_head TP 分片

核心价值：

- **算子覆盖度 100%**：所有 Qwen3.5 源码中的核心计算逻辑均需建模，无遗漏
- **仿真精度可控**：通过实测 profiling 数据校准，仿真与硬件算法时间误差控制在 10% 以内
- **架构可扩展**：支持 Qwen3.5-27B、Qwen3.5-397B-A17B 及未来 MoE/VL 变体
- **分解算子架构**：通过 7 个分解 LA 算子（`la_*`）替代单一 fused op，每个算子的性能模型独立注册，不影响其他模型

### 1.2 动机

msmodeling 仿真框架已支持 Qwen3 系列模型，但 Qwen3.5 引入了多项架构变更，导致现有仿真框架无法直接运行或精度严重不足：

| 痛点 | 说明 |
|------|------|
| 模型加载失败 | Qwen3.5 的 `model_type` 为 `"qwen3_5"` / `"qwen3_5_moe"`，框架未注册，无法加载 HF config |
| GatedDeltaNet 无法建模 | 原有单一 `linear_attention` fused op 无法区分 prefill/decode 路径，中间内存未计入 |
| RMSNorm 融合失败 | Qwen3.5 的 RMSNorm 使用 `output * (1.0 + weight)` 拓扑，与标准 `output * weight` 不匹配 |
| 视觉模块 TP 分片错误 | Qwen3.5 VL 的视觉编码器不应按 TP 切分，但原代码无条件切分导致 shape 错误 |
| 中间内存未计入 | Chunk Delta Rule 内部产生大量中间内存（~442MB），原单一 fused op 只统计输入输出 |
| CausalConv1d 建模不完整 | 原始仿真将 CausalConv1d 的 state update 和 conv body 合并为一个 stage |
| MTP 支持缺失 | 多 token 预测场景下，lm_head TP 分片和 cache_position 元数据未处理 |
| Vision TP 配置缺失 | 缺少单独的 vision_tp_size 参数，无法灵活控制视觉模块并行度 |

### 1.3 目标

- 完整支持 Qwen3.5 全系列模型（纯文本、MoE、VL 变体）的推理仿真
- 7 个分解 LA 算子（`la_*`）独立建模，每个算子通过 `@register_op_properties` 注册性能模型
- CausalConv1d 拆分为 state update（`linear_attn_causal_conv_update`）和 conv body（`linear_attn_causal_conv`）两个算子
- 支持 W8A8_STATIC 量化全链路建模（DynamicQuant + QuantBatchMatmulV3）
- 视觉模块正确跳过 TP 分片或使用独立的 `vision_tp_group`
- Chunk Delta Rule 的中间内存和 `extra_static_cost_count` 正确计入
- MTP 场景的 `mtp.lm_head` 使用独立的 `lmhead_tp_group` TP 分片
- 新增模型变体时只需添加 HF config 注册和 Monkey Patch，无需修改核心 estimator

**非目标**：

- 不做 Qwen3.5 的训练/微调仿真
- 不做 All-Reduce 通信的精确建模（TP 通信暂作为独立项，不在单层算法时间内）
- 不做 NPU 硬件特有操作（Transpose/Slice/ScatterUpdate/ZerosLike/GatherV3）的建模
- 不做多机多卡分布式推理仿真

---

## 2. Qwen3.5 模型架构分析

### 2.1 混合注意力架构

Qwen3.5 的 DecoderLayer 中，**部分层使用标准 Self-Attention（FlashAttention），部分层使用 GatedDeltaNet（线性注意力）**。具体哪些层使用哪种注意力由 `self_attn.layer_type` 字段决定。

```text
Qwen3_5DecoderLayer:
  ├── input_layernorm         → RMSNorm（标准）
  ├── self_attn:
  │   ├── [FlashAttention] q_proj/k_proj/v_proj → RoPE → FlashAttention → o_proj
  │   └── [GatedDeltaNet] in_proj_qkv/in_proj_z/in_proj_b/in_proj_a
  │       → linear_attn_causal_conv / linear_attn_causal_conv_update → split QKV
  │       → linear_attn_fused_gdn_gating (beta/g)
  │       → linear_attn_chunk_gated_delta_rule / linear_attn_recurrent_gated_delta_rule
  │       → linear_attn_gated_rmsnorm → out_proj
  ├── residual_add
  ├── post_attention_layernorm → RMSNorm
  ├── mlp: gate_proj/up_proj → SiLU(gate) * up → down_proj（SwiGLU）
  └── residual_add
```

### 2.2 GatedDeltaNet 分解算子流程

GatedDeltaNet 的 forward 被分解为 7 个 TensorCast LA 算子，通过 Monkey Patch 路由：

| # | TC 算子 | 源码操作 | 约束 |
|:---:|------|------|------|
| 1 | `linear_attn_apply_padding_mask` | `apply_mask_to_padding_states(hidden_states, attention_mask)` | memory-bound |
| 2 | `linear_attn_causal_conv` / `linear_attn_causal_conv_update` | `causal_conv1d_fn(mixed_qkv)` — conv body 或 state update | memory-bound |
| 3 | `linear_attn_fused_gdn_gating` | beta sigmoid + g (exp + softplus) + l2norm Q/K | compute-bound |
| 4 | `linear_attn_chunk_gated_delta_rule` | `chunk_gated_delta_rule(query, key, value, beta, g, chunk_size)` | compute-bound |
| 5 | `linear_attn_recurrent_gated_delta_rule` | `recurrent_gated_delta_rule(query, key, value, beta, g)` | compute-bound |
| 6 | `linear_attn_gated_rmsnorm` | Gated RMSNorm (rmsnorm × weight × SiLU × z) | compute-bound |
| 7 | (Linear 投影) | `in_proj_qkv/z/b/a` + `out_proj` | compute-bound |

### 2.3 Chunk/Recurrent 路径选择逻辑

在 `qwen3_5_moe.py` 的 `_patched_linear_attn_forward` 中，通过以下逻辑选择路径：

```python
has_previous_state = _has_previous_state(cache_params, cache_position, layer_idx)
use_recurrent = has_previous_state and _is_recurrent_decode_batch(seq_len, cache_position)

if use_recurrent:
    conv_op = linear_attn_causal_conv_update
    core_attn_out = linear_attn_recurrent_gated_delta_rule(...)
else:
    conv_op = linear_attn_causal_conv
    core_attn_out = linear_attn_chunk_gated_delta_rule(...)
```

- `tensor_cast_has_previous_state`：从 `cache_position` 元数据读取，由 input_generator 设置
- `_is_recurrent_decode_batch`：检查 `tensor_cast_query_lens` / `tensor_cast_is_decode` / `tensor_cast_num_mtp_tokens`
- MTP 场景下，decode batch 的 query_len 可以是 `1` 或 `1 + num_mtp_tokens`，两者都走 recurrent 路径
- 当 decode batch 有多个 token 且走 recurrent 路径时，会做 `flatten_decode_batch` 将 `(B, S, ...)` reshape 为 `(B*S, 1, ...)`

### 2.4 CausalConv1d 分解

CausalConv1d 被拆分为两个独立算子：

| 算子 | 说明 | 硬件实测 |
|------|------|------|
| `linear_attn_causal_conv` | Conv body：深度可分离因果卷积 + SiLU 激活（prefill 路径） | ~150.1us (`CausalConv1d` kernel) |
| `linear_attn_causal_conv_update` | State update：读取当前输入，移位并更新卷积状态缓冲区（decode 路径） | ~85.8us (`_causal_conv1d_update_kernel`) |

### 2.5 量化方案

Qwen3.5 使用 **W8A8_STATIC** 静态量化。LA 算子本身与量化解耦——`in_proj_qkv/z/b/a` 和 `out_proj` 等 Linear 层通过框架的 `QuantLinearBase` 自动处理量化，`linear_attn_chunk_gated_delta_rule` 等核心注意力算子保持 FP32/BF16 计算。

### 2.6 模型变体矩阵

| 变体 | model_type | Attention | MoE | Vision | MTP |
|------|-----------|----------|-----|----------|-----|
| Qwen3.5-27B | `qwen3_5` | Hybrid | 否 | 是（VL） | 是 |
| Qwen3.5-397B-A17B | `qwen3_5_moe` | Hybrid | 是 | 是（VL） | 是 |

---

## 3. 方案设计

### 3.1 总体架构

#### 3.1.1 修改文件清单

```text
tensor_cast/
├── ops/
│   └── la.py                                ← 新增：7 个 LA 分解算子定义
├── performance_model/
│   └── __init__.py                          ← 新增：7 个 LA 算子的 register_op_properties
│                                             保留：旧的 linear_attention op（qwen3_next 兼容）
├── transformers/
│   ├── transformations.py                   ← 新增：vision_tp_group + LA TP plan + MTP lm_head TP
│   ├── builtin_model/
│   │   ├── qwen3_5_moe.py                  ← 新增：Monkey Patch（共享给 qwen3_5）+ moe 注册
│   │   └── qwen3_5.py                      ← 新增：非 MoE 变体注册（复用 qwen3_5_moe 的 patch）
├── core/
│   └── input_generator.py                  ← 新增：cache_position 元数据 + preprocessor_config 加载
├── adapter/
│   └── context.py                          ← 新增：vision_tp_size 上下文参数
cli/
└── inference/
    ├── model_adapter.py                     ← 新增：--vision-tp-size 参数
    └── text_generate.py                    ← 新增：--vision-tp-size 参数
```

#### 3.1.2 核心设计：分解算子 + Per-Op Property

将单一 fused `linear_attention` op 拆分为 7 个独立的 LA 算子，每个算子通过 `@register_op_properties` 注册独立的性能模型。这与旧方案（12 stage 串行 estimator）的对比：

| 方案 | 优点 | 缺点 | 选择 |
|------|------|------|------|
| **分解算子 + per-op property** | 框架自动处理 stage 串行求和；每算子独立建模；与框架 Roofline 引擎一致 | 无法做 stage 间 `max(compute, memory)` 独立取 max 后求和 | ✅ 采用 |
| 单一 fused op + 自定义 estimator | 可以精确控制 12 stage 的独立 `max(compute, memory)` 求和 | 需要维护独立 estimator 逻辑，偏离框架默认 Roofline 引擎 | ❌ 不采用 |

实际上，由于每个分解算子已经足够细粒度（如 `linear_attn_causal_conv` 只包含一个 conv body，`linear_attn_fused_gdn_gating` 只包含 gating 计算），框架默认的 `max(compute, memory)` 已经能准确估算时间。`extra_static_cost_count` 机制额外补偿 chunk delta rule 的 kernel launch 开销。

#### 3.1.3 中间内存建模

每个 LA 算子的 `register_op_properties` 额外计入中间内存。以 `linear_attn_chunk_gated_delta_rule` 为例：

```python
# _add_linear_attention_chunk_scratch_memory:
# - Q/K/V/β/g 五种张量的 FP32 cast + contiguous + transpose scratch
# - F.pad 填充到 chunk_size=64 倍数 + chunk reshape 分块
# - 输出 FP32→BF16 cast + transpose scratch
# - extra_static_cost_count += 8 (kernel launch overhead)

# _add_linear_attention_state_memory:
# - last_recurrent_state 创建和读写 (batch_size * num_v_heads * head_k_dim * head_v_dim * fp32_bytes)
# - state_read_passes 和 state_write_passes 参数控制
```

#### 3.1.4 Cache Position 元数据

input_generator 在生成 Qwen3.5 模型的输入时，在 `cache_position` tensor 上附加以下元数据属性，供 `qwen3_5_moe.py` 的 `_has_previous_state` 和 `_is_recurrent_decode_batch` 使用：

| 属性 | 类型 | 说明 |
|------|------|------|
| `tensor_cast_query_lens` | `Tuple[int, ...]` | 每个 batch 的 query length |
| `tensor_cast_is_decode` | `Tuple[bool, ...]` | 每个 batch 是否为 decode |
| `tensor_cast_has_previous_state` | `bool` | 是否有 previous KV cache state |
| `tensor_cast_base_decode_query_len` | `int` | MTP 场景的 base decode query length |
| `tensor_cast_num_mtp_tokens` | `int` | MTP token 数量 |
| `tensor_cast_effective_decode_steps` | `int` | 有效 decode step 数 |

#### 3.1.5 Vision TP 和 MTP LMHead TP

- **Vision TP**：新增 `--vision-tp-size` CLI 参数和 `vision_tensor_parallel_size` 配置项。默认为 1（视觉模块不切分 TP）。通过 `vision_tp_group` 控制视觉模块的 TP 分片。
- **MTP LMHead TP**：MTP 场景下，`mtp.lm_head` 使用 `lmhead_tp_group`（默认等于 `tp_group`）进行 TP 分片，而非主 `lm_head` 的 TP 配置。

### 3.2 模型加载流程

#### 3.2.1 模型注册

1. `tensor_cast/transformers/builtin_model/__init__.py` 自动发现并导入所有 `.py` 模块
2. `qwen3_5_moe.py` 注册 `qwen3_5_moe` model_type，包含 MoE 配置和共享的 `patch_method_for_qwen3_5`
3. `qwen3_5.py` 注册 `qwen3_5` model_type（非 MoE），复用 `patch_method_for_qwen3_5`

#### 3.2.2 Monkey Patch 要点

| Patch 目标 | 说明 |
|------|------|
| `Qwen3_5GatedDeltaNet.forward` / `Qwen3_5MoeGatedDeltaNet.forward` | 替换 forward，将投影、conv、attention、norm 路由到 7 个 LA 分解算子 |
| `Qwen3_5DecoderLayer.forward` / `Qwen3_5MoeDecoderLayer.forward` | 替换 forward，根据 `layer_type` 选择 `linear_attention` 或 `full_attention` 路径 |
| `Qwen3_5TextModel._update_linear_attn_mask` / `Qwen3_5MoeTextModel._update_linear_attn_mask` | 替换 mask 更新逻辑，compile 模式下返回 None 保持计算图稳定 |
| `Qwen3_5Model.get_placeholder_mask` / `Qwen3_5MoeModel.get_placeholder_mask` | VL 场景下跳过 image_features token 数检查 |
| `_set_qwen3_5_linear_attn_tp_size(model)` | 在 patch 最后执行，设置线性注意力层的 `tensor_cast_tp_size` |

### 3.3 LA TP Plan

在 `transformations.py` 中，为 Qwen3.5 模型注册 LA 相关的 TP 分片计划：

| Linear 层 | TP 类型 | TP Group | head_num |
|------|------|------|------|
| `linear_attn.in_proj_qkv` | COLWISE_LINEAR | tp_group | `2 * num_key_heads + num_value_heads` |
| `linear_attn.in_proj_z` | COLWISE_LINEAR | tp_group | `num_value_heads` |
| `linear_attn.in_proj_b` | COLWISE_LINEAR | tp_group | `num_value_heads` |
| `linear_attn.in_proj_a` | COLWISE_LINEAR | tp_group | `num_value_heads` |
| `linear_attn.out_proj` | ROWWISE_LINEAR | o_proj_tp_group | `num_value_heads` |

要求 `num_key_heads` 和 `num_value_heads` 必须能被 `tp_size` 整除。

---

## 4. 测试设计

### 4.1 单元测试

| 测试文件 | 测试项 | 验证点 |
|------|------|------|
| test_runtime.py | `linear_attn_causal_conv_eager` | `linear_attn_causal_conv` 形状 + perf model（无 state memory） |
| test_runtime.py | `linear_attn_causal_conv_update_eager` | `linear_attn_causal_conv_update` 形状 + state memory |
| test_runtime.py | `linear_attn_apply_padding_mask_eager` | `linear_attn_apply_padding_mask` 形状 |
| test_runtime.py | `linear_attn_fused_gdn_gating_eager` | 4 个输出形状（query_out, key_out, beta, g） |
| test_runtime.py | `la_chunk_rule_includes_scratch_memory_and_extra_static` | `memory_readwrite_bytes > 0` + `extra_static_cost_count > 0` |
| test_runtime.py | `linear_attn_recurrent_gated_delta_rule_eager` | 形状 + state memory read/write |
| test_runtime.py | `linear_attn_gated_rmsnorm_eager` | 形状 |
| test_runtime.py | `extra_static_cost_count_combines` | `PerformanceProperties.combine` 正确累加 |
| test_runtime.py | `qwen3_5_linear_attention_uses_local_tp_heads` | TP=16 时，prefill 走 chunk 路径 + decode 走 recurrent 路径 + MTP decode 走 recurrent 路径 |
| test_runtime.py | `qwen3_5_linear_attention_w8a8_reuses_quant_linear` | W8A8_STATIC 下投影层使用 quantize/static_quant_linear |
| test_runtime.py | `qwen3_5_linear_attention_rejects_invalid_tp_size` | TP=32 不能整除 num_k_heads=16，报 ValueError |
| test_runtime.py | `qwen3_5_vision_tp_defaults_to_unsharded` | `vision_tp_group.world_size=1`，attn 未被分片 |
| test_runtime.py | `qwen3_5_vision_tp_can_be_enabled_explicitly` | `vision_tp_size=2` 时 attn 被分片 |
| test_runtime.py | `qwen3_5_mtp_lm_head_uses_lmhead_tp_plan` | `mtp.lm_head` 是 ColumnParallelLinear |
| test_runtime.py | `qwen3_5_linear_attention_with_padding_mask` | 非空 attention_mask 触发 `linear_attn_apply_padding_mask` |
| test_input_generator.py | `qwen3_5_decode_mtp_cache_position_metadata` | MTP decode 场景下 cache_position 元数据正确 |
| test_input_generator.py | `varlen_qwen3_5_cache_position_starts_at_context` | Varlen 场景下 cache_position 从 context 起始 |
| test_input_generator.py | `qwen3_vl_1080p_resize_to_1088x1920` | VL 图像 resize shape 正确 |
| test_input_generator.py | `read_preprocessor_config_invalid_json_returns_none` | 无效 JSON 返回 None |
| test_input_generator.py | `read_preprocessor_config_missing_file_returns_none` | 缺失文件返回 None |
| test_input_generator.py | `resolve_local_preprocessor_config_non_dir_returns_none` | 非目录路径返回 None |
| test_input_generator.py | `load_preprocessor_pixel_limits_no_config_json_returns_none` | 目录无 config 时 fallback 返回 None |

### 4.2 集成测试

| 测试项 | 测试内容 | 验证点 |
|------|------|------|
| 端到端推理 | Qwen3.5-27B 完整推理仿真 | 仿真运行成功，无 crash |
| 多 query 场景 | decode 模式（seq_len=1, has_previous_state） | 正确走 recurrent path |
| 长序列 prefill | seq_len=2171, prefill | 正确走 chunk path |
| W8A8_STATIC 量化 | 量化推理全链路 | 量化算子被正确调用 |
| TP=2 推理 | 双卡张量并行 | 通信算子正确建模 |
| MoE 变体 | Qwen3.5-397B-A17B | grouped_matmul_swiglu 正确调用 |

### 4.3 实测验证案例

以下是在 ATLAS_800_A3_560T_128G_DIE 双卡环境下的实测结果：

| 场景 | 输入长度 | 输出长度 | 实测耗时 | 命令                                                                                                                                                                                                                                                                                                          |
|---|---|---|---|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Prefill 长文本** | 3500 | 1000 | 447.1 ms | `python -m cli.inference.text_generate Qwen/Qwen3.5-27B --device ATLAS_800_A3_560T_128G_DIE --num-devices 2 --num-queries 1 --query-length 3510 --context-length 0  --quantize-linear-action W8A8_STATIC --tp-size 2 --compile`                                                                             |
| **Decode 长文本上下文** | 3500 | 1000 | 27.33 ms | `python -m cli.inference.text_generate Qwen/Qwen3.5-27B --device ATLAS_800_A3_560T_128G_DIE --num-devices 2 --num-queries 1 --query-length 4 --context-length 4250  --quantize-linear-action W8A8_STATIC --tp-size 2 --compile --num-mtp-tokens 3 --decode`                                                 |
| **Prefill 超长文本** | 16000 | 1000 | 2.099 s | `python -m cli.inference.text_generate Qwen/Qwen3.5-27B --device ATLAS_800_A3_560T_128G_DIE --num-devices 2 --num-queries 1 --query-length 16824 --context-length 0  --quantize-linear-action W8A8_STATIC --tp-size 2 --compile` |
| **Decode 超长上下文** | 16000 | 1000 | 28.161 ms | `python -m cli.inference.text_generate Qwen/Qwen3.5-27B --device ATLAS_800_A3_560T_128G_DIE --num-devices 2 --num-queries 3 --query-length 4 --context-length 16500  --quantize-linear-action W8A8_STATIC --tp-size 2 --compile --num-mtp-tokens 3 --decode`                                                |
| **VL Prefill** (1080p + 30 token) | 1080p | 30 | 1.712 s | `python -m cli.inference.text_generate Qwen/Qwen3.5-27B --device ATLAS_800_A3_560T_128G_DIE --num-devices 2 --num-queries 4 --query-length 30 --context-length 0 --image-height 1080 --image-width 1920 --image-batch-size 1 --quantize-linear-action W8A8_STATIC --tp-size 2 --compile --num-mtp-tokens 3` |
| **VL Decode** (1080p + 30 token) | 1080p | 30 | 39.84 ms | `python -m cli.inference.text_generate Qwen/Qwen3.5-27B --device ATLAS_800_A3_560T_128G_DIE --num-devices 2 --num-queries 8 --query-length 4 --context-length 16500  --quantize-linear-action W8A8_STATIC --tp-size 2 --compile --num-mtp-tokens 3 --decode`                                                |
| **Prefill 超长文本** | 16000 | 1000 | 504ms | `python -m cli.inference.text_generate Qwen/Qwen3.5-35B-A3B --device ATLAS_800_A3_560T_128G_DIE --num-devices 2 --num-queries 1 --query-length 8192 --context-length 0  --quantize-linear-action W4A8_STATIC --tp-size 2 --compile  --num-mtp-tokens 3` |
| **Decode 超长上下文** | 16000 | 1000 | 29.76 ms | `python -m cli.inference.text_generate Qwen/Qwen3.5-35B-A3B --device ATLAS_800_A3_560T_128G_DIE --num-devices 2 --num-queries 8 --query-length 4 --context-length 16500  --quantize-linear-action W4A8_STATIC --tp-size 2 --compile --num-mtp-tokens 3 --decode`                                            |

---

## 5. 缺点和风险

| 风险 | 影响 | 应对措施 |
|------|------|------|
| 模型架构变更 | Qwen3.5 新版本可能修改 GatedDeltaNet 实现 | Monkey Patch 需同步更新，通过 test_model_load 快速发现 |
| 中间内存估算偏差 | Chunk Delta Rule scratch memory 估算与实际有差距 | 通过 `extra_static_cost_count` 补偿 kernel launch 开销 |
| CausalConv1d tiled kernel 开销 | Roofline 模型无法完全覆盖 NPU tiled kernel 开销 | 标注为 P1 优化项，当前阶段可接受 ~125us 残差 |
| All-Reduce 通信 | TP=2 时每层有 4 次 All-Reduce（~682us），暂未在单层算法时间中建模 | 通信作为独立项，不影响单层算法精度 |
| Qwen3.5 上游依赖 | 依赖 transformers 主线 Qwen3.5 实现 | 需锁定 transformers 版本，Monkey Patch 需适配对应版本 |
| 视觉编码器复杂度 | VL 变体的视觉编码器包含 DeepStack 等特殊结构 | 复用现有 Qwen3-VL 的 Patch 经验 |

---

## 6. 现有技术

| 项目/工具 | 做法 | 借鉴与差异 |
|------|------|-----------|
| Qwen3 仿真（已有） | 标准 FlashAttention + RoPE + SwiGLU 建模 | Qwen3.5 在此基础上新增 GatedDeltaNet 和 CausalConv1d |
| Qwen3-VL 仿真（已有） | 视觉编码器 + DeepStack + 多模态融合 | 复用视觉模块的 Monkey Patch 和 TP 分片经验 |
| Qwen3Next 仿真（已有） | 使用旧的 `linear_attention` fused op | 保留旧 op 以保证兼容性 |
| Chunk Delta Rule 参考实现 | `modeling_qwen3_5.py` 官方源码 | 逐行对照，确保 FLOPs 和内存估算一致 |

---

## 附录

### 附录 A：参考资料

- [Qwen3.5 模型源码](https://huggingface.co/Qwen/Qwen3.5-27B) — `modeling_qwen3_5.py`
- Qwen3.5-27B 仿真算子覆盖度分析报告（内部文档）
- 实测 LINEARATTENTION 实测算子记录（内部文档）
- 代码修改对齐说明（内部文档）
- 视觉对比（内部文档）

### 附录 B：术语表

| 术语 | 说明 |
|------|------|
| GatedDeltaNet | Qwen3.5 的线性注意力机制 |
| Chunk Gated Delta Rule | 将序列按 chunk_size=64 分块做 intra-chunk attention + inter-chunk recurrence |
| Recurrent Gated Delta Rule | 逐 token 递归更新注意力状态 |
| CausalConv1d | 因果深度可分离卷积 |
| Gated RMSNorm | Qwen3.5 特有 RMSNorm 变体：`output * (1.0 + weight)` |
| W8A8_STATIC | 权重 INT8 + 激活 INT8 静态量化 |
| MTP | Multi-Task Prediction，多 token 预测 |
| LA | Linear Attention，线性注意力 |

### 附录 C：文档更新历史

| 版本 | 日期 | 变更内容 | 作者 |
|------|------|---------|------|
| v1.0 | 2026-05-19 | 初始版本（性能回归测试框架） | @yuyinkai |
| v2.0 | 2026-06-04 | 重写为 Qwen3.5 仿真支持设计文档 | @yuyinkai |
| v3.0 | 2026-06-11 | 重构为分解 LA 算子架构（7 个 `la_*` ops），新增 vision TP / MTP / cache_position metadata / preprocessor config 支持 | @yuyinkai |
