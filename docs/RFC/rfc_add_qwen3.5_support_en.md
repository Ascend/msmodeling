# Qwen3.5 Simulation Support Design Document

Status: Draft
Authors: @yuyinkai
Created: 2026-05-19
Updated: 2026-06-11

---

## 1. Overview

### 1.1 Introduction

Qwen3.5 is the latest multimodal large language model in the Qwen series, introducing **GatedDeltaNet (Linear Attention)** as an alternative to standard Self-Attention. It uses linear-complexity attention in some layers to reduce the computational cost of long-sequence inference. This RFC describes the design for full Qwen3.5 model support in the msmodeling simulation framework (including Qwen3.5-27B, Qwen3.5-397B-A17B, and other variants).

Compared to the already-supported Qwen3 series, Qwen3.5 introduces the following new features:

- **GatedDeltaNet Linear Attention**: Includes Chunk Gated Delta Rule (prefill) and Recurrent Gated Delta Rule (decode) paths
- **CausalConv1d**: Depthwise separable causal convolution for sequence state management
- **Gated RMSNorm**: Gated RMSNorm (`output * (1.0 + weight.float())`)
- **W8A8 Static Quantization**: End-to-end INT8 inference with DynamicQuant + QuantBatchMatmulV3
- **Hybrid Attention Architecture**: Some layers use standard FlashAttention, others use GatedDeltaNet
- **Multimodal Vision Encoder**: VL variants include a vision encoder with different shapes from text-only models
- **MTP (Multi-Task Prediction)**: Multi-token prediction affecting cache_position metadata and lm_head TP sharding

Core value:

- **100% operator coverage**: All core computational logic in Qwen3.5 source code is modeled
- **Controllable simulation accuracy**: Calibrated against real profiling data, with <10% error for simulation vs hardware kernel time
- **Extensible architecture**: Supports Qwen3.5-27B, Qwen3.5-397B-A17B, and future MoE/VL variants
- **Decomposed operator architecture**: 7 decomposed LA ops (`la_*`) replace a single fused op, each with independently registered performance models

### 1.2 Motivation

The msmodeling framework already supports Qwen3 series models, but Qwen3.5 introduces several architectural changes that prevent the existing framework from running directly or cause significant accuracy degradation:

| Pain Point | Description |
|------|------|
| Model loading failure | Qwen3.5's `model_type` is `"qwen3_5"` / `"qwen3_5_moe"`, not registered in the framework |
| GatedDeltaNet not modelable | The original single `linear_attention` fused op couldn't distinguish prefill/decode paths or account for intermediate memory |
| RMSNorm fusion failure | Qwen3.5 RMSNorm uses `output * (1.0 + weight)` topology, mismatched with standard `output * weight` |
| Vision module TP sharding error | Qwen3.5 VL vision encoder should not be sharded by TP, but old code unconditionally sharded it |
| Intermediate memory not accounted | Chunk Delta Rule generates ~442MB of intermediate memory, the single fused op only counted inputs/outputs |
| CausalConv1d modeling incomplete | Original simulation merged state update and conv body into one stage |
| MTP support missing | Multi-token prediction scenarios lacked lm_head TP sharding and cache_position metadata handling |
| Vision TP config missing | No separate `vision_tp_size` parameter, unable to flexibly control vision module parallelism |

### 1.3 Goals

- Full support for all Qwen3.5 model variants (text-only, MoE, VL) inference simulation
- 7 decomposed LA ops (`la_*`) independently modeled, each with `@register_op_properties` for performance properties
- CausalConv1d split into state update (`linear_attn_causal_conv_update`) and conv body (`linear_attn_causal_conv`) operators
- W8A8_STATIC quantization full-chain modeling (DynamicQuant + QuantBatchMatmulV3)
- Vision modules correctly skip TP sharding or use an independent `vision_tp_group`
- Chunk Delta Rule intermediate memory and `extra_static_cost_count` properly accounted
- MTP `mtp.lm_head` uses independent `lmhead_tp_group` for TP sharding
- Adding new model variants only requires HF config registration and Monkey Patch, no core estimator modifications

**Non-Goals**:

- No training/fine-tuning simulation for Qwen3.5
- No precise All-Reduce communication modeling (TP communication is treated as a separate item, not included in per-layer kernel time)
- No modeling of NPU-specific ops (Transpose/Slice/ScatterUpdate/ZerosLike/GatherV3)
- No multi-node distributed inference simulation

---

## 2. Qwen3.5 Model Architecture Analysis

### 2.1 Hybrid Attention Architecture

In Qwen3.5's DecoderLayer, **some layers use standard Self-Attention (FlashAttention), others use GatedDeltaNet (Linear Attention)**. Which layers use which attention type is determined by the `self_attn.layer_type` field.

```text
Qwen3_5DecoderLayer:
  ├── input_layernorm         → RMSNorm (standard)
  ├── self_attn:
  │   ├── [FlashAttention] q_proj/k_proj/v_proj → RoPE → FlashAttention → o_proj
  │   └── [GatedDeltaNet] in_proj_qkv/in_proj_z/in_proj_b/in_proj_a
  │       → linear_attn_causal_conv / linear_attn_causal_conv_update → split QKV
  │       → linear_attn_fused_gdn_gating (beta/g)
  │       → linear_attn_chunk_gated_delta_rule / linear_attn_recurrent_gated_delta_rule
  │       → linear_attn_gated_rmsnorm → out_proj
  ├── residual_add
  ├── post_attention_layernorm → RMSNorm
  ├── mlp: gate_proj/up_proj → SiLU(gate) * up → down_proj (SwiGLU)
  └── residual_add
```

### 2.2 GatedDeltaNet Decomposed Op Flow

The GatedDeltaNet forward is decomposed into 7 TensorCast LA ops, routed through Monkey Patch:

| # | TC Op | Source Op | Constraint |
|:---:|------|------|------|
| 1 | `linear_attn_apply_padding_mask` | `apply_mask_to_padding_states(hidden_states, attention_mask)` | memory-bound |
| 2 | `linear_attn_causal_conv` / `linear_attn_causal_conv_update` | `causal_conv1d_fn(mixed_qkv)` — conv body or state update | memory-bound |
| 3 | `linear_attn_fused_gdn_gating` | beta sigmoid + g (exp + softplus) + l2norm Q/K | compute-bound |
| 4 | `linear_attn_chunk_gated_delta_rule` | `chunk_gated_delta_rule(query, key, value, beta, g, chunk_size)` | compute-bound |
| 5 | `linear_attn_recurrent_gated_delta_rule` | `recurrent_gated_delta_rule(query, key, value, beta, g)` | compute-bound |
| 6 | `linear_attn_gated_rmsnorm` | Gated RMSNorm (rmsnorm × weight × SiLU × z) | compute-bound |
| 7 | (Linear Projection) | `in_proj_qkv/z/b/a` + `out_proj` | compute-bound |

### 2.3 Chunk/Recurrent Path Selection

In `qwen3_5_moe.py`'s `_patched_linear_attn_forward`, the path is selected as follows:

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

- `tensor_cast_has_previous_state`: read from `cache_position` metadata, set by input_generator
- `_is_recurrent_decode_batch`: checks `tensor_cast_query_lens` / `tensor_cast_is_decode` / `tensor_cast_num_mtp_tokens`
- In MTP scenarios, the decode batch's query_len can be `1` or `1 + num_mtp_tokens`, both use the recurrent path
- When a decode batch has multiple tokens and uses the recurrent path, `flatten_decode_batch` reshapes `(B, S, ...)` to `(B*S, 1, ...)`

### 2.4 CausalConv1d Decomposition

CausalConv1d is split into two independent operators:

| Op | Description | HW Measured |
|------|------|------|
| `linear_attn_causal_conv` | Conv body: depthwise separable causal convolution + SiLU activation (prefill path) | ~150.1us (`CausalConv1d` kernel) |
| `linear_attn_causal_conv_update` | State update: read current input, shift and update convolution state buffer (decode path) | ~85.8us (`_causal_conv1d_update_kernel`) |

### 2.5 Quantization Scheme

Qwen3.5 uses **W8A8_STATIC** static quantization. LA ops are decoupled from quantization — Linear layers like `in_proj_qkv/z/b/a` and `out_proj` are automatically quantized via the framework's `QuantLinearBase`, while core attention ops like `linear_attn_chunk_gated_delta_rule` remain in FP32/BF16.

### 2.6 Model Variant Matrix

| Variant | model_type | Attention | MoE | Vision | MTP |
|------|-----------|----------|-----|----------|-----|
| Qwen3.5-27B | `qwen3_5` | Hybrid | No | Yes (VL) | Yes |
| Qwen3.5-397B-A17B | `qwen3_5_moe` | Hybrid | Yes | Yes (VL) | Yes |

---

## 3. Design

### 3.1 Overall Architecture

#### 3.1.1 Modified File List

```text
tensor_cast/
├── ops/
│   └── la.py                                ← NEW: 7 LA decomposed operator definitions
├── performance_model/
│   └── __init__.py                          ← NEW: register_op_properties for 7 LA ops
│                                             KEPT: old linear_attention op (qwen3_next compat)
├── transformers/
│   ├── transformations.py                   ← NEW: vision_tp_group + LA TP plan + MTP lm_head TP
│   ├── builtin_model/
│   │   ├── qwen3_5_moe.py                  ← NEW: Monkey Patch (shared) + moe registration
│   │   └── qwen3_5.py                      ← NEW: Non-MoE variant registration (reuses patch)
├── core/
│   └── input_generator.py                  ← NEW: cache_position metadata + preprocessor_config
├── adapter/
│   └── context.py                          ← NEW: vision_tp_size context parameter
cli/
└── inference/
    ├── model_adapter.py                     ← NEW: --vision-tp-size argument
    └── text_generate.py                    ← NEW: --vision-tp-size argument
```

#### 3.1.2 Core Design: Decomposed Ops + Per-Op Property

The single fused `linear_attention` op is split into 7 independent LA ops, each with its own `@register_op_properties` for performance modeling. Comparison with the old approach (12-stage serial estimator):

| Approach | Pros | Cons | Verdict |
|------|------|------|------|
| **Decomposed ops + per-op property** | Framework handles serial stage summation automatically; each op independently modeled; consistent with framework Roofline engine | Cannot do per-stage `max(compute, memory)` + sum across stages | ✅ Adopted |
| Single fused op + custom estimator | Can precisely control 12-stage independent `max(compute, memory)` + sum | Requires maintaining independent estimator logic, deviates from default Roofline engine | ❌ Not adopted |

In practice, since each decomposed op is already fine-grained enough (e.g., `linear_attn_causal_conv` contains only conv body, `linear_attn_fused_gdn_gating` contains only gating computation), the framework's default `max(compute, memory)` can accurately estimate time. The `extra_static_cost_count` mechanism additionally compensates for kernel launch overhead in chunk delta rule.

#### 3.1.3 Intermediate Memory Modeling

Each LA op's `register_op_properties` additionally accounts for intermediate memory. Taking `linear_attn_chunk_gated_delta_rule` as an example:

```python
# _add_linear_attention_chunk_scratch_memory:
# - Q/K/V/β/g 5-tensor FP32 cast + contiguous + transpose scratch
# - F.pad padding to chunk_size=64 multiple + chunk reshape
# - Output FP32→BF16 cast + transpose scratch
# - extra_static_cost_count += 8 (kernel launch overhead)

# _add_linear_attention_state_memory:
# - last_recurrent_state create and read/write (batch_size * num_v_heads * head_k_dim * head_v_dim * fp32_bytes)
# - state_read_passes and state_write_passes parameter control
```

#### 3.1.4 Cache Position Metadata

When generating inputs for Qwen3.5 models, input_generator attaches the following metadata attributes to the `cache_position` tensor, used by `_has_previous_state` and `_is_recurrent_decode_batch` in `qwen3_5_moe.py`:

| Attribute | Type | Description |
|------|------|------|
| `tensor_cast_query_lens` | `Tuple[int, ...]` | Query length per batch |
| `tensor_cast_is_decode` | `Tuple[bool, ...]` | Whether each batch is decode |
| `tensor_cast_has_previous_state` | `bool` | Whether previous KV cache state exists |
| `tensor_cast_base_decode_query_len` | `int` | Base decode query length in MTP scenario |
| `tensor_cast_num_mtp_tokens` | `int` | Number of MTP tokens |
| `tensor_cast_effective_decode_steps` | `int` | Effective decode step count |

#### 3.1.5 Vision TP and MTP LMHead TP

- **Vision TP**: New `--vision-tp-size` CLI argument and `vision_tensor_parallel_size` config option. Defaults to 1 (vision module unsharded). Controlled via `vision_tp_group`.
- **MTP LMHead TP**: In MTP scenarios, `mtp.lm_head` uses `lmhead_tp_group` (defaults to `tp_group`) for TP sharding, separate from the main `lm_head` TP config.

### 3.2 Model Loading Flow

#### 3.2.1 Model Registration

1. `tensor_cast/transformers/builtin_model/__init__.py` auto-discovers and imports all `.py` modules
2. `qwen3_5_moe.py` registers `qwen3_5_moe` model_type with MoE config and shared `patch_method_for_qwen3_5`
3. `qwen3_5.py` registers `qwen3_5` model_type (non-MoE), reusing `patch_method_for_qwen3_5`

#### 3.2.2 Monkey Patch Highlights

| Patch Target | Description |
|------|------|
| `Qwen3_5GatedDeltaNet.forward` / `Qwen3_5MoeGatedDeltaNet.forward` | Replace forward, routing projection, conv, attention, norm to 7 decomposed LA ops |
| `Qwen3_5DecoderLayer.forward` / `Qwen3_5MoeDecoderLayer.forward` | Replace forward, selecting `linear_attention` or `full_attention` path based on `layer_type` |
| `Qwen3_5TextModel._update_linear_attn_mask` / `Qwen3_5MoeTextModel._update_linear_attn_mask` | Replace mask update logic; returns None in compile mode to keep graph stable |
| `Qwen3_5Model.get_placeholder_mask` / `Qwen3_5MoeModel.get_placeholder_mask` | Skip image_features token count check in VL scenarios |
| `_set_qwen3_5_linear_attn_tp_size(model)` | Executed at the end of patch, sets `tensor_cast_tp_size` on linear attention layers |

### 3.3 LA TP Plan

In `transformations.py`, LA-related TP sharding plans are registered for Qwen3.5 models:

| Linear Layer | TP Type | TP Group | head_num |
|------|------|------|------|
| `linear_attn.in_proj_qkv` | COLWISE_LINEAR | tp_group | `2 * num_key_heads + num_value_heads` |
| `linear_attn.in_proj_z` | COLWISE_LINEAR | tp_group | `num_value_heads` |
| `linear_attn.in_proj_b` | COLWISE_LINEAR | tp_group | `num_value_heads` |
| `linear_attn.in_proj_a` | COLWISE_LINEAR | tp_group | `num_value_heads` |
| `linear_attn.out_proj` | ROWWISE_LINEAR | o_proj_tp_group | `num_value_heads` |

Requires `num_key_heads` and `num_value_heads` to be divisible by `tp_size`.

---

## 4. Test Design

### 4.1 Unit Tests

| Test File | Test | Verification |
|------|------|------|
| test_runtime.py | `linear_attn_causal_conv_eager` | `linear_attn_causal_conv` shape + perf model (no state memory) |
| test_runtime.py | `linear_attn_causal_conv_update_eager` | `linear_attn_causal_conv_update` shape + state memory |
| test_runtime.py | `linear_attn_apply_padding_mask_eager` | `linear_attn_apply_padding_mask` shape |
| test_runtime.py | `linear_attn_fused_gdn_gating_eager` | 4 output shapes (query_out, key_out, beta, g) |
| test_runtime.py | `la_chunk_rule_includes_scratch_memory_and_extra_static` | `memory_readwrite_bytes > 0` + `extra_static_cost_count > 0` |
| test_runtime.py | `linear_attn_recurrent_gated_delta_rule_eager` | shape + state memory read/write |
| test_runtime.py | `linear_attn_gated_rmsnorm_eager` | shape |
| test_runtime.py | `extra_static_cost_count_combines` | `PerformanceProperties.combine` correctly accumulates |
| test_runtime.py | `qwen3_5_linear_attention_uses_local_tp_heads` | TP=16: prefill=chunk path, decode=recurrent path, MTP decode=recurrent path |
| test_runtime.py | `qwen3_5_linear_attention_w8a8_reuses_quant_linear` | W8A8_STATIC: projections use quantize/static_quant_linear |
| test_runtime.py | `qwen3_5_linear_attention_rejects_invalid_tp_size` | TP=32 not dividing num_k_heads=16, raises ValueError |
| test_runtime.py | `qwen3_5_vision_tp_defaults_to_unsharded` | `vision_tp_group.world_size=1`, attn unsharded |
| test_runtime.py | `qwen3_5_vision_tp_can_be_enabled_explicitly` | `vision_tp_size=2` shards attn |
| test_runtime.py | `qwen3_5_mtp_lm_head_uses_lmhead_tp_plan` | `mtp.lm_head` is ColumnParallelLinear |
| test_runtime.py | `qwen3_5_linear_attention_with_padding_mask` | Non-None attention_mask triggers `linear_attn_apply_padding_mask` |
| test_input_generator.py | `qwen3_5_decode_mtp_cache_position_metadata` | MTP decode: cache_position metadata correct |
| test_input_generator.py | `varlen_qwen3_5_cache_position_starts_at_context` | Varlen: cache_position starts at context |
| test_input_generator.py | `qwen3_vl_1080p_resize_to_1088x1920` | VL image resize shape correct |
| test_input_generator.py | `read_preprocessor_config_invalid_json_returns_none` | Invalid JSON returns None |
| test_input_generator.py | `read_preprocessor_config_missing_file_returns_none` | Missing file returns None |
| test_input_generator.py | `resolve_local_preprocessor_config_non_dir_returns_none` | Non-directory path returns None |
| test_input_generator.py | `load_preprocessor_pixel_limits_no_config_json_returns_none` | Dir without config returns None |

### 4.2 Integration Tests

| Test | Content | Verification |
|------|------|------|
| End-to-end inference | Qwen3.5-27B full inference simulation | Simulation runs successfully, no crash |
| Multi-query scenario | decode mode (seq_len=1, has_previous_state) | Correctly uses recurrent path |
| Long sequence prefill | seq_len=2171, prefill | Correctly uses chunk path |
| W8A8_STATIC quantization | Full quantization chain | Quantization ops correctly called |
| TP=2 inference | Dual-card tensor parallelism | Communication ops correctly modeled |
| MoE variant | Qwen3.5-397B-A17B | grouped_matmul_swiglu correctly called |

### 4.3 Real HW Verification

The following are real HW measurements on ATLAS_800_A3_560T_128G_DIE (dual-card):

| Scenario | Input Len | Output Len | HW Time | Command                                                                                                                                                                                                                                                                                                     |
|---|---|---|---|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Prefill long text** | 3500 | 1000 | 447.1 ms | `python -m cli.inference.text_generate Qwen/Qwen3.5-27B --device ATLAS_800_A3_560T_128G_DIE --num-devices 2 --num-queries 1 --query-length 3510 --context-length 0  --quantize-linear-action W8A8_STATIC --tp-size 2 --compile`                                                                             |
| **Decode long context** | 3500 | 1000 | 27.33 ms | `python -m cli.inference.text_generate Qwen/Qwen3.5-27B --device ATLAS_800_A3_560T_128G_DIE --num-devices 2 --num-queries 1 --query-length 4 --context-length 4250  --quantize-linear-action W8A8_STATIC --tp-size 2 --compile --num-mtp-tokens 3 --decode`                                                 |
| **Prefill very long text** | 16000 | 1000 | 2.099 s | `python -m cli.inference.text_generate Qwen/Qwen3.5-27B --device ATLAS_800_A3_560T_128G_DIE --num-devices 2 --num-queries 1 --query-length 16824 --context-length 0  --quantize-linear-action W8A8_STATIC --tp-size 2 --compile` |
| **Decode very long context** | 16000 | 1000 | 28.161 ms | `python -m cli.inference.text_generate Qwen/Qwen3.5-27B --device ATLAS_800_A3_560T_128G_DIE --num-devices 2 --num-queries 3 --query-length 4 --context-length 16500  --quantize-linear-action W8A8_STATIC --tp-size 2 --compile --num-mtp-tokens 3 --decode`                                                |
| **VL Prefill** (1080p + 30 token) | 1080p | 30 | 1.712 s | `python -m cli.inference.text_generate Qwen/Qwen3.5-27B --device ATLAS_800_A3_560T_128G_DIE --num-devices 2 --num-queries 4 --query-length 30 --context-length 0 --image-height 1080 --image-width 1920 --image-batch-size 1 --quantize-linear-action W8A8_STATIC --tp-size 2 --compile --num-mtp-tokens 3` |
| **VL Decode** (1080p + 30 token) | 1080p | 30 | 39.84 ms | `python -m cli.inference.text_generate Qwen/Qwen3.5-27B --device ATLAS_800_A3_560T_128G_DIE --num-devices 2 --num-queries 8 --query-length 4 --context-length 16500  --quantize-linear-action W8A8_STATIC --tp-size 2 --compile --num-mtp-tokens 3 --decode`                                                |
| **Prefill very long text** (35B-A3B) | 16000 | 1000 | 504 ms | `python -m cli.inference.text_generate Qwen/Qwen3.5-35B-A3B --device ATLAS_800_A3_560T_128G_DIE --num-devices 2 --num-queries 1 --query-length 8192 --context-length 0  --quantize-linear-action W4A8_STATIC --tp-size 2 --compile  --num-mtp-tokens 3` |
| **Decode very long context** (W4A8) | 16000 | 1000 | 29.76 ms | `python -m cli.inference.text_generate Qwen/Qwen3.5-35B-A3B --device ATLAS_800_A3_560T_128G_DIE --num-devices 2 --num-queries 8 --query-length 4 --context-length 16500  --quantize-linear-action W4A8_STATIC --tp-size 2 --compile --num-mtp-tokens 3 --decode`                                            |

---

## 5. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|------|------|
| Model architecture changes | New Qwen3.5 versions may modify GatedDeltaNet implementation | Monkey Patch must be updated synchronously; detected quickly by test_model_load |
| Intermediate memory estimation error | Chunk Delta Rule scratch memory estimation may differ from reality | Compensate kernel launch overhead via `extra_static_cost_count` |
| CausalConv1d tiled kernel overhead | Roofline model cannot fully cover NPU tiled kernel overhead | Marked as P1 optimization; ~125us residual acceptable at current stage |
| All-Reduce communication | 4 All-Reduce per layer (TP=2, ~682us), not yet modeled in per-layer kernel time | Communication as separate item, doesn't affect per-layer algorithm accuracy |
| Qwen3.5 upstream dependency | Depends on transformers mainline Qwen3.5 implementation | Must lock transformers version; Monkey Patch must adapt to corresponding version |
| Vision encoder complexity | VL variants contain special structures like DeepStack | Reuse existing Qwen3-VL Patch experience |

---

## 6. Related Work

| Project/Tool | Approach | Reference & Differences |
|------|------|-----------|
| Qwen3 simulation (existing) | Standard FlashAttention + RoPE + SwiGLU modeling | Qwen3.5 adds GatedDeltaNet and CausalConv1d on this basis |
| Qwen3-VL simulation (existing) | Vision encoder + DeepStack + multimodal fusion | Reuse vision module Monkey Patch and TP sharding experience |
| Qwen3Next simulation (existing) | Uses old `linear_attention` fused op | Old op retained for compatibility |
| Chunk Delta Rule reference implementation | `modeling_qwen3_5.py` official source code | Line-by-line cross-reference to ensure FLOPs and memory estimation consistency |

---

## Appendix

### Appendix A: References

- [Qwen3.5 Model Source Code](https://huggingface.co/Qwen/Qwen3.5-27B) — `modeling_qwen3_5.py`
- Qwen3.5-27B Simulation Operator Coverage Analysis Report (internal document)
- Measured LINEARATTENTION Kernel Records (internal document)
- Code Change Alignment Notes (internal document)
- Visual Comparison (internal document)

### Appendix B: Glossary

| Term | Description |
|------|------|
| GatedDeltaNet | Qwen3.5's linear attention mechanism |
| Chunk Gated Delta Rule | Sequence split into chunk_size=64 blocks for intra-chunk attention + inter-chunk recurrence |
| Recurrent Gated Delta Rule | Token-by-token recurrent attention state update |
| CausalConv1d | Causal depthwise separable convolution |
| Gated RMSNorm | Qwen3.5-specific RMSNorm variant: `output * (1.0 + weight)` |
| W8A8_STATIC | Weight INT8 + Activation INT8 static quantization |
| MTP | Multi-Task Prediction |
| LA | Linear Attention |

### Appendix C: Document History

| Version | Date | Changes | Author |
|------|------|---------|------|
| v1.0 | 2026-05-19 | Initial version (performance regression test framework) | @yuyinkai |
| v2.0 | 2026-06-04 | Rewritten as Qwen3.5 simulation support design document | @yuyinkai |
| v3.0 | 2026-06-11 | Refactored to decomposed LA operator architecture (7 `la_*` ops), added vision TP / MTP / cache_position metadata / preprocessor config support | @yuyinkai |
