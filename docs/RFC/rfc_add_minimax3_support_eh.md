# RFC: Add MiniMax-M3 Model Adaptation Support

---

| Item | Content |
| :--- | :--- |
| **Status** | Approved |
| **Author** | - |
| **Created** | 2026-07-25 |
| **Related Link** | TensorCast MiniMax-M3 adaptation design document |

## 1. Overview

This RFC proposes adding simulation and modeling support for MiniMax-M3 / MiniMax-M3-VL in TensorCast. Compared with the existing MiniMax-M2 support in the repository, MiniMax-M3 has the following key differences:

1. It uses layered sparse attention configuration to distinguish dense attention layers from MiniMax3 sparse attention layers. Older Transformers versions mainly expose this through `layer_types`, while newer Transformers versions may expose it through `text_config.sparse_attention_config`.
2. Sparse attention layers include an additional learned indexer, which selects important KV blocks for each query token.
3. The attention path contains MiniMax3-specific semantic operators such as indexer cache, `siso_reshape_and_cache`, `minimax_indexer`, and `minimax_sparse_attention`.
4. Dense MLP and MoE experts both use the MiniMax3-style fused `gate_up_proj` + OAI SwiGLU structure. This needs to be split into `gate_proj` / `up_proj` / `down_proj` so TensorCast can recognize and fuse the graph more easily.
5. LayerNorm uses Gemma RMSNorm semantics, where the effective value of `weight` is `1 + weight`. The decoder layer also needs to model `residual + norm` as fused `add_rms_norm2`.

The goal is to provide end-to-end simulation through a minimal model profile, local model patches, MiniMax3-specific wrappers, and a small number of virtual ops, without adding new `ModelProfile` fields or modifying upstream Transformers source code.

---

## 2. Motivation

MiniMax-M3 is already supported natively in newer Transformers releases through the `minimax_m3_vl` model source. If TensorCast only relies on generic model adaptation logic, the following issues appear:

1. **Sparse attention cannot be modeled according to the real execution chain**: regular attention only expresses Q/K/V + attention body, but cannot express MiniMax3 indexer scoring, TopK block selection, and sparse attention body.
2. **Indexer cache is different from regular KV cache**: the indexer only caches index keys and does not cache values, so it needs a single-input single-output cache op such as `siso_reshape_and_cache`.
3. **The MLP structure is not friendly to existing fusion passes**: in the HF source, both dense MLP and expert MLP use fused `gate_up_proj`. Direct forward execution produces `matmul` followed by `chunk`, which is not friendly to `SinkSplitPass` and grouped matmul + SwiGLU fusion.
4. **RMSNorm semantics differ from regular RMSNorm**: Gemma RMSNorm requires `1 + weight`. If this is not modeled explicitly, both numerical semantics and profiling operator boundaries may become inconsistent.
5. **Materialized lengths can be lost under compile/meta tensors**: the Roofline model needs the real per-request `seq_len/query_len/decode` information. Otherwise sparse attention context length may fall back to meta tensor shape, causing biased decode/prefill modeling.
6. **Transformers config fields change across versions**: MiniMax3 sparse attention fields may evolve from `layer_types/index_*` to `sparse_attention_config.sparse_*`. If the patch does not handle both forms, the sparse attention wrapper or MTP attention wrapper may be skipped.

Therefore, MiniMax3 needs a set of local model patches and dedicated semantic operators.

---

## 3. Goals And Non-Goals

### 3.1 Goals

- Support loading, patching, quantization, sharding, compilation, and analytic trace output for `model_type=minimax_m3_vl`.
- Support layered modeling of dense attention layers and sparse attention layers.
- Support cost modeling for MiniMax3 learned indexer block score and TopK selection.
- Support modeling of MiniMax3 sparse attention body, including QK/PV, softmax, Q/O/topk/KV memory access.
- Support MiniMax3 OAI SwiGLU and the quantized path into quantized down projection.
- Support MiniMax3 Gemma RMSNorm, Q/K/indexer norm, and decoder layer `add_rms_norm2` fusion modeling.
- Keep `ModelProfile` minimal, avoid adding new `ModelProfile` attributes, and avoid spreading MiniMax3-specific cases into the generic profile layer.
- Reuse existing MoE, quant linear, sharding, and compile pass capabilities as much as possible. Add local adaptation only where MiniMax3 is structurally incompatible with the existing framework.

### 3.2 Non-Goals

- Do not implement a real NPU sparse attention kernel. Only provide TensorCast semantic ops and Roofline cost models.
- Do not duplicate the full Transformers MiniMax3 model source inside TensorCast.
- Do not hard-code vLLM/vLLM-Ascend deployment kernel optimizations as source-level semantics. Related differences are only recorded as profiling calibration items or risks.
- Do not change the profiles or main simulation paths of existing models such as MiniMax2, DeepSeek, and GLM.

---

## 4. Use Cases

### 4.1 Decode Simulation

Typical command:

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

Expected behavior:

- Dense layers emit `tensor_cast.attention.default`.
- Sparse layers emit `siso_reshape_and_cache`, `tensor_cast.minimax_indexer.default`, and `tensor_cast.minimax_sparse_attention.default`.
- Sparse attention uses real lengths such as `seq_lens_values=[context_length] * batch`, rather than meta tensor fallback.
- Routed experts in MoE layers correctly trigger the grouped matmul path.

### 4.2 Prefill Simulation

Typical command:

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

Expected behavior:

- `minimax_indexer` estimates score and TopK cost over all semantically visible historical tokens/blocks.
- `minimax_sparse_attention` estimates sparse QK/PV and softmax GP based on visible KV blocks for each query.
- Q/O/topk metadata, K/V cache memory access, MMA, and GP are modeled separately, so they can be aligned with profiling kernels by category.

---

## 5. Design

### 5.1 Overall Architecture

MiniMax3 adaptation is split into six layers:

1. **ModelProfile registration layer**: register `model_type=minimax_m3_vl`, and specify the MoE block name, expert count field, custom expert adapter, gate router, and patch method.
2. **Model patch layer**: `patch_method_for_minimax_m3` applies attention, RMSNorm, dense MLP, MoE gate attribute fixes, and then reuses generic MoE patching.
3. **Attention wrapper layer**: `MiniMaxM3AttentionWrapper` replaces each layer's `self_attn`, routing to dense or sparse paths based on `layer_types` or `sparse_attention_config`. If an MTP layer exceeds the main layer config length, it inherits the sparse configuration of the last main layer.
4. **Semantic op layer**: add TensorCast ops such as `minimax_indexer`, `minimax_sparse_attention`, `m3_swiglu`, `m3_swiglu_quant`, and `fused_rope`.
5. **Performance model layer**: register Roofline properties for MiniMax3-specific ops in `performance_model/__init__.py`.
6. **Input generation layer**: carry materialized `seq_lens_values/query_lens_values/is_decode_values` in `AttentionMetadataTensorCast`, so sparse attention Roofline can use real lengths.

### 5.2 ModelProfile Design

Current registration:

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

Design principles:

- Keep only fields that are already supported by the generic framework and required by MiniMax3.
- Do not add new `ModelProfile` attributes.
- Reuse the native Transformers MiniMax3-VL structure through `language_layers_path_str="language_model.layers"`.
- Use `custom_expert_module_type` only to solve the incompatibility between MiniMax3 expert fused `gate_up_proj` and the existing MoE pass.

### 5.3 Patch Method Design

Execution order of `patch_method_for_minimax_m3`:

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

Step meanings:

- `cache_rotary_embedding=False`: MiniMax3 uses partial/3D RoPE. Generic rotary cache rewriting may generate incompatible position embedding shapes.
- `patch_minimax_m3_attention`: replace HF attention with a TensorCast dense/sparse wrapper.
- `patch_minimax_m3_layernorm`: replace Gemma RMSNorm and monkey patch decoder layer forward to emit `add_rms_norm2`.
- `patch_minimax_m3_dense_mlp`: split dense MLP fused `gate_up_proj` into `gate_proj/up_proj`.
- `patch_minimax_m3_moe_gate_attrs`: copy the MoE block's `routed_scaling_factor` to the gate object for the custom router.
- `patch_moe`: reuse TensorCast generic MoE patch, dispatch, grouped matmul, and combine logic.

The current implementation also adds two types of config compatibility:

- `_get_minimax_m3_effective_text_config` finds the actual language model config containing `hidden_size` from `model.text_config`, `model._inner.hf_config`, or `model.hf_config`. This supports nested structures such as `hf_config.text_config`.
- `_resolve_minimax_m3_sparse_attention_config` supports both old fields `layer_types/index_*` and newer fields `sparse_attention_config.sparse_*`, avoiding false "no sparse layer" detection after Transformers upgrades.

### 5.4 Attention Wrapper Design

`MiniMaxM3AttentionWrapper.forward` explicitly splits the attention chain:

1. Main attention projections: `q_proj/k_proj/v_proj`.
2. Q/K norm: `q_norm/k_norm`, converted to `tensor_cast.rms_norm` through wrappers.
3. RoPE: use `tensor_cast.fused_rope` to represent Q/K partial RoPE.
4. KV cache write: use `tensor_cast.reshape_and_cache`.
5. Dense attention layer: directly emit `tensor_cast.attention.default`.
6. Sparse attention layer:
   - indexer Q/K projection and norm;
   - indexer RoPE;
   - index K cache write: `tensor_cast.siso_reshape_and_cache`;
   - block selection: `tensor_cast.minimax_indexer.default`;
   - sparse attention body: `tensor_cast.minimax_sparse_attention.default`;
   - output projection: `o_proj`.

MTP layer handling:

- Main layers are classified according to the sparse configuration array.
- If an MTP layer's `layer_idx` exceeds the sparse configuration array length, it inherits the dense/sparse type of the last main layer.
- This allows MiniMax3 sparse attention under `--num-mtp-tokens` to still use the TensorCast wrapper instead of falling back to the original HF attention view logic.

Reasons for this design:

- Projection, norm, RoPE, cache, and `o_proj` are either independent profiling kernels or existing TensorCast ops, so they should remain explicit.
- `minimax_indexer` only covers the score/topk selection boundary.
- `minimax_sparse_attention` only covers the sparse QK/PV/softmax body boundary.
- Avoid wrapping the entire attention into one large op, which would make profiling alignment and debugging harder.

### 5.5 Dense MLP And MoE Expert Design

The typical MiniMax3 HF MLP structure is:

```text
gate_up_proj(hidden)
chunk -> gate, up
OAI SwiGLU
down_proj
```

TensorCast needs a structure closer to what existing passes can recognize:

```text
gate_proj(hidden)
up_proj(hidden)
m3_swiglu(gate, up)
down_proj
```

Therefore, the adaptation introduces:

- `MiniMaxM3DenseMLPWrapper`: handles dense MLP layers.
- `MiniMaxM3MoeExpertMLP`: handles MoE expert layers.

Quantized path:

```text
gate/up -> m3_swiglu_quant -> int8 activation + activation_scale -> TensorCastQuantLinear.down_proj
```

Non-quantized path:

```text
gate/up -> m3_swiglu -> down_proj
```

### 5.6 MoE Gate Router Design

`route_minimax_m3_gate` uses MiniMax3 sigmoid top-k gate semantics:

1. Compute `router_logits` using the gate weight.
2. In TP scenarios, pad along the token dimension and slice to the current rank.
3. Call `tensor_cast.moe_gating_top_k_sigmoid`.
4. Apply `routed_scaling_factor` and `e_score_correction_bias`.
5. Return `topk_indices` and `topk_weights` converted back to the hidden dtype.

`patch_minimax_m3_moe_gate_attrs` copies `routed_scaling_factor` from the MoE block to the gate because the generic MoE router receives the gate object rather than the full MoE block.

### 5.7 RMSNorm And RoPE Design

RMSNorm design:

- `GemmaRMSNormFusedWrapper`: used for decoder layer norm, with `effective_weight = 1 + weight`.
- `RMSNormFusedWrapper`: used for Q/K norm and indexer Q/K norm.
- `_fused_decoder_layer_forward`: represents `residual + attention_output + post_attention_layernorm` as `tensor_cast.add_rms_norm2`.

RoPE design:

- `fused_rope` is a profiling/meta op and does not perform real numerical rotation.
- It returns the original `query/key`, preserving valid downstream tensors and avoiding propagation of uninitialized `empty_like` values.
- Roofline is represented by the input/output shapes of the standalone `fused_rope` op.

### 5.8 Input Generator And Attention Metadata

MiniMax3 sparse attention Roofline requires real lengths:

- `seq_lens_values`: total visible length of each request.
- `query_lens_values`: current query token count of each request.
- `is_decode_values`: whether each request is in decode phase.

These fields are stored on `AttentionMetadataTensorCast` and passed by `MiniMaxM3AttentionWrapper` to `minimax_indexer` and `minimax_sparse_attention`.

Why this is necessary:

- Under torch.compile / meta tensors, `seq_lens` and `query_lens` may be FakeTensors, so `.tolist()` is not safe.
- If only tensor shape fallback is used, real context length may be mistaken for query token count, causing severe underestimation for long-context decode.

### 5.9 Roofline Modeling Design

#### 5.9.1 `m3_swiglu`

Boundary:

```text
gate, up -> MiniMax3 OAI SwiGLU output
```

Modeling:

```text
GP ops = numel(gate) * 7
```

Meaning:

- Approximately covers pointwise operations such as clamp, sigmoid, multiply, and add.
- Does not include gate/up projection or down projection matmul, which are modeled by their corresponding linear/grouped matmul ops.

#### 5.9.2 `m3_swiglu_quant`

Boundary:

```text
gate, up -> OAI SwiGLU -> dynamic symmetric quant -> int8 activation + fp32 scale
```

Modeling:

- First reuse the pointwise GP cost of `m3_swiglu`.
- Then combine memory access properties from `dynamic_quantize_symmetric`.
- Scale shape is quantized along the last dimension: `scale_shape[-1] = 1`.

#### 5.9.3 `minimax_indexer`

Boundary:

```text
idx_q, index K cache, seq_lens, query_lens, block_table
  -> topk block indices
```

Formula:

```text
index_qk_mma = sum_b(2 * Q_b * N * L_b * D)
block_reduce_gp = sum_b(Q_b * N * L_b)
topk_gp = ceil(log2(K)) * sum_b(Q_b * N * B_n)
bytes_total = idx_q bytes + visible index K bytes + block score bytes + topk bytes
```

Variables:

- `Q_b`: number of query tokens for request b.
- `L_b`: total visible length for request b.
- `N`: number of indexer heads.
- `D`: indexer head dimension.
- `K`: number of topk blocks.
- `B_n=ceil(L_b/block_size)`: number of visible blocks.

Modeling boundary:

- index K cache write is not counted inside `minimax_indexer`; it is modeled separately by `siso_reshape_and_cache`.
- Deployment TopK tile shape is not treated as a semantic reduction in candidate block count; semantically all visible blocks still participate in selection.

#### 5.9.4 `minimax_sparse_attention`

Boundary:

```text
query, KV cache, topk_idx, seq_lens, query_lens, block_table
  -> sparse attention output
```

Formula:

```text
attended_pairs = sum_q(attended_tokens_q)
attn_mma = 4 * N_q * attended_pairs * D
attn_gp = 6 * N_q * attended_pairs
qo_bytes = 2 * dtype_size * T * N_q * D
topk_bytes = 4 * T * N_kv * K
kv_bytes = 2 * dtype_size * effective_attended_pairs * N_kv * D
```

Current KV read formula:

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

## 6. Technical Choices

### 6.1 Wrapper + Virtual Op

Reasons:

- Do not modify the installed Transformers package source.
- Preserve original module parameters and weight sources.
- Explicitly mark key operator boundaries in the TensorCast graph.
- Make `--dump-input-shapes` and chrome trace easier to align with profiling kernels.

Alternative:

- Directly run the original HF forward and let the torch graph expand naturally.

Reason for rejection:

- Original forward contains `gate_up_proj + chunk`, which is unfriendly to existing MoE grouped matmul fusion.
- Original sparse attention Python logic and deployment kernel boundaries differ, making profiling alignment harder.
- Indexer cache and sparse attention body need explicit semantic ops for independent Roofline modeling.

### 6.2 Minimal ModelProfile

Reasons:

- MiniMax3-specific behavior is mainly in attention/indexer/RMSNorm/MLP patching and should not pollute `ModelProfile`.
- Keep consistency with GLM, DeepSeek, and other models: generic fields express generic behavior, while special cases are handled by `patch_method` and wrappers.

---

## 7. Compatibility And Impact

### 7.1 Impact On Existing Models

This design is expected not to affect existing models such as MiniMax2, DeepSeek, and GLM, because:

- MiniMax3 has an independent profile bound to `model_type=minimax_m3_vl`.
- `patch_method_for_minimax_m3` only executes when the MiniMax3 profile is matched.
- Newly added ops are only called by the MiniMax3 wrapper.
- Adding `m3_swiglu` to `SinkSplitPass` binary ops only lets MiniMax3 gate/up split consumers have the same rewrite capability as regular `swiglu`.

### 7.2 Transformers Version Requirement

MiniMax3 depends on native `minimax_m3_vl` source in newer Transformers releases, so the dependency constraint should be raised to:

```text
transformers>=5.13.0,<5.14.0
```

After the upgrade, some models such as `deepseek_v32/deepseek_v4/mimo_v2_flash` may already be registered upstream in Transformers. Safe-register compatibility is needed to avoid duplicate `AutoConfig.register` errors.

MiniMax3 sparse attention config parsing supports both:

- Old-style `layer_types`, `index_n_heads`, `index_head_dim`, `index_topk_blocks`, `index_block_size`, `index_local_blocks`.
- New-style `sparse_attention_config.sparse_attention_freq`, `sparse_num_index_heads`, `sparse_index_dim`, `sparse_topk_blocks`, `sparse_block_size`, `sparse_local_block`.

---

## 8. Test Design

### 8.1 Unit Tests

| Test | Validation Point |
| :--- | :--- |
| `test_minimax_indexer_uses_materialized_lengths_for_meta_tensors` | Use real `seq_lens_values/query_lens_values` under meta tensors |
| `test_minimax_indexer_prefill_scores_all_visible_blocks` | Prefill indexer semantically scores/selects over all visible blocks |
| `test_minimax_indexer_k_cache_read_is_shared_across_index_heads` | Index K cache read is not duplicated by index head count |
| `test_minimax_indexer_perf_properties_use_input_dtype` | Indexer Roofline uses input dtype instead of fixed fp32 |
| `test_minimax_sparse_attention_uses_materialized_context_length` | Decode sparse attention uses real context length and caps KV read at `min(K * block_size, L_b)` |
| `test_minimax_sparse_attention_prefill_kv_pairs_use_selected_capacity_geometric_mean` | Prefill sparse attention KV read uses the geometric mean between selected capacity and semantic pairs |
| `test_minimax_m3_nested_sparse_attention_config_is_resolved` | New `sparse_attention_config` can be parsed correctly |
| `test_m3_swiglu_bf16_has_gp_roofline_properties` | `m3_swiglu` GP op count is correct |

### 8.2 Suggested Verification Commands

```bash
pytest tests/regression/tensor_cast/test_minimax_m3_sparse_attention.py -q
pre-commit run --files \
  tensor_cast/transformers/builtin_model/minimax_m3.py \
  tensor_cast/layers/minimax_m3_attention.py \
  tensor_cast/ops/minimax_m3_sparse_attention.py \
  tensor_cast/performance_model/__init__.py \
  tests/regression/tensor_cast/test_minimax_m3_sparse_attention.py
```

End-to-end smoke:

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

## 9. Risks And Mitigations

### 9.1 Sparse Attention Roofline Boundary May Not Exactly Match Profiling Kernels

Risks:

- Deployment may split score/topk, sparse QK/PV, merge, and other work into multiple kernels, or fuse them into a single ACLNN op.
- TensorCast semantic op boundaries do not always have a one-to-one mapping with profiling kernel boundaries.

Mitigations:

- Clearly define the semantic boundaries of `minimax_indexer` and `minimax_sparse_attention` in the RFC.
- When comparing with profiling, aggregate kernels under the same semantic category, for example score/topk as indexer and sparse attention body as sparse attention.
- Do not force deployment tile shapes to become source-level semantic candidate counts.

### 9.2 KV Memory Reuse Approximation May Need Profiling Calibration

Risks:

- Prefill sparse attention kernels may reuse KV blocks inside query tiles.
- Fully independent per-query reads overestimate cost; reading physical cache only once underestimates cost.

Mitigations:

- Use the current approximate `effective_attended_pairs`.
- The current implementation uses the geometric mean between selected capacity and semantic attended pairs. If Ascend/vLLM-Ascend kernel source or stable profiling data becomes available, this approximation can be upgraded to explicit profiling calibration and documented in code comments and tests.

### 9.3 Upstream Transformers Implementation Changes

Risks:

- `minimax_m3_vl` structure, field names, or layer type strings may change with Transformers upgrades.

Mitigations:

- Lock the dependency to the verified range `>=5.13.0,<5.14.0`.
- Use `_get_minimax_m3_effective_text_config` and `_resolve_minimax_m3_sparse_attention_config` to support both old-style `layer_types/index_*` and new-style `sparse_attention_config.sparse_*`.
- Rerun MiniMax3 smoke tests and shape trace comparison whenever Transformers is upgraded.

### 9.4 Maintenance Cost Of Monkey Patching Decoder Layer Forward

Risks:

- `_fused_decoder_layer_forward` must remain semantically aligned with upstream decoder layer forward.

Mitigations:

- Only apply safe AddRMSNorm fusion and do not change the main attention/MLP semantics.
- Re-check this function if the upstream decoder forward structure changes.

---

## 10. Open Issues

1. The exact KV reuse strategy inside the Ascend sparse attention kernel has not been fully confirmed from public source.
2. The mapping between `minimax_sparse_attention` and different profiling kernel combinations needs to be further standardized in op_mapping documentation.
3. Long-context prefill indexer/sparse attention error still needs calibration with more cases.
4. Current MTP sparse attention inherits the last layer sparse configuration to avoid missing the wrapper. Whether MTP needs a more detailed op_mapping or performance calibration should be confirmed with profiling.

---

## Appendix A: Glossary

| Term | Meaning |
| :--- | :--- |
| Dense Attention | Regular full attention layer, using `tensor_cast.attention.default` |
| Sparse Attention | MiniMax3 sparse attention layer, where the indexer first selects KV blocks and then sparse attention body runs |
| Indexer | MiniMax3 learned block selector, using index Q/K to choose important historical blocks for each query |
| Index K Cache | Indexer-specific key cache; it caches index keys only and does not cache values |
| `siso_reshape_and_cache` | Single-input single-output cache write op for index K cache |
| `minimax_indexer` | TensorCast MiniMax3 semantic op covering block score and TopK selection |
| `minimax_sparse_attention` | TensorCast MiniMax3 semantic op covering sparse QK/PV/softmax body |
| OAI SwiGLU | MiniMax3 SwiGLU variant with `alpha/limit` parameters |
| Gemma RMSNorm | RMSNorm semantics whose effective weight is `1 + weight` |
