# RFC: Context Parallel Modeling for DeepSeek-Style Attention

## Metadata

| Item | Content |
|:---|:---|
| **Status** | Draft |
| **Author** | Codex |
| **Updated Date** | 2026-07-13 |
| **Related Links** | https://github.com/vllm-project/vllm/issues/30055 |

---

## 1. Problem Statement

During the alignment work between TensorCast and vLLM-Ascend for W8A8 DeepSeek-style sparse-attention models, we found differences in static weight memory, context-parallel sharding, and attention backend selection. GLM-5.1 is the current workload used for reproduction and validation, but this proposal is not GLM-5.1 specific. DSA CP should be modeled as a general context-parallel mechanism for DeepSeek-style / DSA-like sparse-attention structures.

The main gaps are:

- Static weight memory is underestimated. Existing TensorCast logic reports around `28.452 GB`, while the real engine observes around `38 GB`; another real configuration observes around `31 GB`. The difference is related to the DSA CP path and weight materialization behavior.
- Quantized weight classification is incomplete. A W8A8 model directory can still contain float tensors, and the real engine can additionally hold `weight_scale`, `weight_offset`, fp32 scales, transposed weights, derived weights, and temporary layouts.
- `kv_b_proj` handling is different. In the DSA CP path, the real engine does not shard `kv_b_proj` like normal TP. TensorCast needs to preserve the full layout in the corresponding path and derive `W_UK_T` / `W_UV` from it.
- The attention backend is different. The real engine uses SFA for the DSA sparse-attention path, while TensorCast currently models the path mainly through MLA/FIA-style operators.

The first phase only covers model-level core classes, linear-layer sharding semantics, and sequence-parallel communication rewrites. NZ layout, PCP/DCP, and ServingCast-level changes are out of scope.

## 2. Proposal

The proposal aligns three behaviors in TensorCast:

1. Static memory accounting for original and derived weights.
2. Token-dimension communication rewrites based on the sequence parallel pass.
3. SFA operator modeling for the DSA sparse-attention path.

Design principles:

- DSA CP is not a model-specific switch. It should be triggered by sequence-parallel capability plus DeepSeek-style structure detection.
- `ParallelConfig` should continue to describe parallel sizes only. Behavior flags should be inferred from model structure, transformer config, or runtime execution config.
- Sequence collectives should reuse the existing `tp_group`. The first phase should not introduce `sharded_cp_group`, so DSA CP is not modeled as a separate communication domain. MLA initialization, however, must use an effective singleton TP view when DSA projections are replicated; this makes head and derived-weight layout follow the same effective TP size as `q_b_proj` / `kv_b_proj` without mutating the shared group.
- SFA inputs should be modeled as bf16 according to the real profile. The presence of quantized linear weights should not imply quantized SFA inputs.
- The dense MLA quantized path should remain unchanged. Linear-weight quantization and attention-kernel input quantization must be modeled separately.

Existing baseline capabilities:

- The duplicated profiling datasource wrapping issue has been fixed.
- The TP sharding plan can keep `q_b_proj` / `kv_b_proj` replicated with `disable_tp`, and MLA can derive `W_UK_T` / `W_UV` from the resulting full layout.
- The current transformers baseline supports the sparse-attention three-value return path.

Capabilities not yet merged:

- SFA attention replacement.
- SFA profiling decomposer / op mapping.
- Static weight accounting based on `quant_model_description.json` or MemoryTracker.
- DeepSeek-style / current GLM5 workload structural logic layered on top of the sequence parallel pass.

## 3. Detailed Design

### 3.1 DSA CP Trigger

DSA CP should not introduce a standalone `--enable-dsa-cp` entry. The recommended conditions are:

- sequence parallel is enabled.
- the model structure has the required DeepSeek-style sparse-attention features.

`_has_dsa_structure` should stay lightweight. It should detect structural capability and should not rely on a `model_type` allowlist. This allows new models with the same structure to reuse the same path.

### 3.2 Sequence Parallel Communication Rewrite

In DSA CP scenarios, `all_reduce -> reduce_scatter` should not be implemented as a new communication-group capability. It should reuse TensorCast's existing `sequence_parallel_pass`.

The existing `sequence_parallel_pass` already captures the core vLLM-Ascend SP semantics:

- Rewrite `all_reduce -> rms_norm / add_rms_norm` into `reduce_scatter -> rms_norm / add_rms_norm -> all_gather`.
- Rewrite `all_reduce -> add_rms_norm2` into `reduce_scatter -> add_rms_norm2`, with selective `all_gather` insertion based on downstream consumers.
- Keep intermediate residual chains on the local token shard when possible, and insert `all_gather` only when full-token layout is needed.
- Both `reduce_scatter` and `all_gather` shard along the token/sequence dimension and reuse the original `all_reduce` `rank_group`, which is the normal `tp_group`.

Therefore, the first phase should not add `sharded_cp_group`. TensorCast needs to align model layers, linear layers, and attention layers with the local-token / full-token semantics produced by `sequence_parallel_pass` after DeepSeek-style / DSA-like structure is detected.

### 3.3 DSA Structural Layering

DSA CP should be treated as structural logic layered on top of sequence parallelism:

- The trigger remains `enable_sequence_parallel and _has_dsa_structure()`.
- Sequence communication still uses `tp_group`; no extra CP group is created. The effective MLA construction group is singleton for replicated DSA attention weights, while the shared/global `tp_group` remains unchanged for SP collectives.
- DSA-related projections such as `q_b_proj` / `kv_b_proj` should avoid normal TP sharding and preserve full layout.
- In the DSA CP prefill path, `o_proj` / `wo_b` uses a replicated full weight and does not perform the normal row-parallel feature all-reduce. The sequence pass slices the full-token attention result to the local-token range before this projection.
- Attention boundaries must make it explicit whether hidden states are in local-token shard layout or full-token layout. When full Q/K/V is required, all-gather semantics should be inserted or preserved.
- As long as `dsa_indexer` remains a single fused semantic op, its `hidden_states` input boundary must observe a full-token layout so downstream SFA sees the complete sequence semantics. If an upstream norm pattern has already inserted `all_gather`, no indexer-specific pattern is needed. If follow-up debugging finds a path where a local norm output feeds `dsa_indexer` directly, TensorCast should add a full-token consumer-boundary pattern for that case.
- This graph/layout constraint does not mean the indexer top-k query must be computed on full tokens. The performance model should split the fused `dsa_indexer` cost internally: model the top-k query side on the local token shard, and model cache/compressor update plus full-sequence boundary work on full tokens.

#### 3.3.1 Token-Layout Contract Across Decoder Layers

The token dimension has two explicit layout states:

- **Full-token**: the rank observes the complete prefill token range, for example `4096` tokens.
- **Local-token**: the token range is sequence-sharded by the TP/SP group, for example `4096 / 16 = 256` tokens.

The prefill decoder-layer contract is:

| Boundary or component | Required layout | Rationale |
|:---|:---|:---|
| Attention input, Q/K/V projection, DSA indexer semantic boundary, and attention computation | Full-token | Sparse-attention semantics and the current fused indexer boundary require the complete sequence view. |
| Attention output before `o_proj` | Full-token, full-feature | For the validated GLM workload this is `[4096, 64, 256]`; DSA attention is computed over the complete query-token range and all heads. |
| Token slice before `o_proj` | Full-token to local-token | Slice only the token dimension, for example `[4096, 64, 256] -> [256, 64, 256]`. This is a graph-layout operation, not a feature reduce-scatter. |
| Replicated `o_proj` | Local-token, full-feature input | Flatten to `[256, 16384]` and apply the full `[16384, 6144]` projection weight. No TP feature all-reduce or second token sharding is performed. |
| Post-attention norm and residual | Local-token | Norm and elementwise residual work do not require the complete token range. |
| MoE gate, top-k, routed experts, and shared experts | Local-token | All MoE branches consume the same already-sharded token range. Expert dispatch may redistribute experts, but must not restore the TP full-token range. |
| Shared-expert TP reduction | Local-token | The all-reduce combines feature shards and preserves the local token dimension; it is not a second sequence reduce-scatter. |
| MoE output and next-layer residual merge | Local-token | Routed and shared branches must meet with identical local-token shapes. |
| Next-layer input norm | Local-token | The norm is performed before returning to the attention domain. |
| Normalized value entering the next attention | Local-token to full-token | Insert all-gather only on output 0 (the normalized attention input). Output 1 (the residual) always remains local for the next replicated `o_proj`/norm boundary. |

The steady-state prefill flow is therefore:

```text
full-token attention
    -> slice(token dim) to local-token
    -> local-token full-weight o_proj (no feature all-reduce)
    -> local-token post-attention norm/residual
    -> local-token gate/top-k
    -> local-token routed experts + local-token shared experts
    -> local-token residual merge and next input norm
    -> all_gather(token dim) on the normalized attention input only
    -> next full-token attention
```

This contract prevents three common modeling errors:

1. Restoring full tokens before the MoE gate or shared expert, which overstates MoE compute and communication.
2. Propagating local tokens directly into the next attention, which understates attention/indexer compute.
3. Gathering the residual output between layers. In an expanded graph this causes a second gather at the next boundary and can inflate `4096` tokens to `65536`; only the normalized output may return to the full-token domain.

The shared-expert path needs special care because both feature-parallel and sequence-parallel communication can use the same TP group. Communication intent must be inferred from the layout transition, not from the collective name alone: a shared-expert all-reduce on local-token input preserves token layout. The replicated DSA `o_proj` has no feature all-reduce; its full-token-to-local-token transition is the explicit pre-projection token slice.

#### 3.3.2 Expanded and Repetition Graph Equivalence

Layer repetition is a graph-size optimization and must not change the token-layout contract. An expanded graph carries the state through every decoder layer explicitly. A repetition graph carries the same state through `region_end -> copy_region* -> region_begin` boundaries and multiplies the representative region at runtime.

Both forms must produce the same effective operator shapes and communication counts after repetition expansion:

- attention and the fused DSA indexer boundary remain full-token for every layer;
- gate, routed experts, and shared experts remain local-token for every MoE layer;
- every attention-to-FFN transition slices the attention output to local-token before the replicated full-weight `o_proj`;
- every FFN-to-next-attention transition gathers only the normalized attention input.

Region markers are layout-transparent. They may prevent cross-layer fusion from being represented identically—for example, one graph may contain `add_rms_norm2`, while another contains equivalent `add + rms_norm` operations—but they must not alter full-token/local-token shapes or communication semantics.

### 3.4 Linear-Layer Sharding Semantics

RowParallelLinear and ColumnParallelLinear should share consistent constraints:

- `disable_tp=True` means the layer weight is not sharded by the current TP group.
- `gather_slice_data` should be constrained by `disable_tp`, so a full-layout weight is not gathered or sliced incorrectly.
- `slice_input_by_last_dim` should not be changed. Under sequence-parallel pass, it represents input slicing behavior and should not be reused for DSA CP weight layout semantics.

### 3.5 MLA Derived Weights

When DSA CP keeps `kv_b_proj` replicated:

- `kv_b_proj.weight` uses the full layout.
- MLA initialization uses an effective `world_size=1`, so `_num_heads_per_rank` naturally follows full-head semantics without a separate `disable_tp` branch inside MLA.
- `W_UK_T` / `W_UV` are derived from the full `kv_b_proj`.

This path is important for explaining the difference between the `31 GB` and `38 GB` observations: besides original checkpoint weights, the real engine also holds derived weights, scale/offset tensors, and layout-conversion tensors.

### 3.6 SFA Attention Operator

SFA modeling should add only the bf16 sparse-attention operator:

- `tensor_cast.mla_sparse_attention.default`

It should not add:

- `tensor_cast.mla_sparse_attention_quant.default`

The reason is that the current real profile shows bf16 SFA inputs. `quant_config` only indicates that quantized linear weights exist; it does not imply quantized SFA kernel inputs.

The dense MLA path remains unchanged:

- `tensor_cast.multihead_latent_attention.default`
- `tensor_cast.multihead_latent_attention_quant.default`

### 3.7 Static Weight Accounting

Short term, the existing static accounting should include:

- float exception tensors.
- `weight_scale` / `weight_offset`.
- fp32 scales.
- transposed weights and temporary layouts.
- derived weights such as `W_UK_T` / `W_UV`.
- full `kv_b_proj` when `disable_tp=True` prevents normal TP sharding.

Long term, TensorCast should read `quant_model_description.json` to classify quantization status and evaluate moving static weight accounting toward the model-forward MemoryTracker path. This can reduce divergence between static formulas and real execution-time materialization.

### 3.8 MoE Router Token Layout

TensorCast must distinguish the gate/router branch selector from its literal name. In the current TensorCast MoE implementation, `moe_gate_returns_raw_logits=True` is the branch used when EP is enabled and the gate should run on the full packed-local token tensor first:

```python
if self.has_ep and self._inner.moe_config.gate_returns_raw_logits:
```

This branch matches the vLLM-Ascend decode profile shape:

```text
gate matmul:     [9, 6144] -> [9, 256]
pad + TP slice:  [9, 256]  -> [12, 256] -> [3, 256]  # dp8tp4
topk:            [3, 256]  -> [3, 8]
routed expert:   [3, 6144]
```

Therefore, for the current GLM5 workload, TensorCast should keep `moe_gate_returns_raw_logits=True`. It is required to model the "full-token gate, sliced-token topk/dispatch" decode path. The name should not be interpreted as only controlling whether raw logits are returned by the gate.

Prefill and decode differ because they enter MoE with different token-layout states:

- Prefill with sequence parallel / DSA CP: the attention output has already been reduced-scattered or sequence-sharded before MoE. The gate, topk, routed expert, and shared expert should all operate on the local token shard, for example `[256, 6144] -> [256, 256] -> [256, 8]`.
- Decode with MC2 / fused MC2: the gate matmul can run on full packed-local decode tokens first, then the MoE prepare path pads and slices both `hidden_states` and `router_logits` by TP before topk and routed expert dispatch. With `dp8tp4` and 9 tokens, this becomes `9 -> pad 12 -> slice 3`.

The corresponding vLLM-Ascend behavior is split across the generic MoE path:

- `shared_forward_impl` may compute router logits from the full `hidden_states`.
- `PrepareAndFinalizeWithMC2.prepare` pads and slices both `hidden_states` and `router_logits` by TP.
- `select_experts` / `moe_gating_top_k` consumes the sliced router logits.

TensorCast should model these as separate concerns:

- `moe_gate_returns_raw_logits=True` selects the full-token gate branch and allows route-time pad/slice of router logits.
- Sequence parallel / DSA CP decides whether the MoE input has already become a local token shard in prefill.
- The MoE communication prepare step decides whether decode topk/dispatch uses a TP-local slice even when the gate matmul saw full packed-local tokens.

## 4. Module Interaction Design

The high-level flow is:

```text
Model config / runtime config
        |
        v
DSA structure detection + sequence_parallel
        |
        v
SequenceParallelPass
        |-- all_reduce -> reduce_scatter
        |-- full-token attention -> local-token slice -> replicated o_proj
        |-- local norm / residual chain
        |-- all_gather normalized output only when full-token layout is needed
        |-- keep residual output local across layers
        |-- reuse tp_group
        |
        v
Linear layer wrappers
        |-- disable_tp
        |-- gather_slice_data constraint
        |
        v
MLA / sparse attention wrapper
        |-- effective singleton TP view for replicated DSA projections
        |-- full kv_b_proj when needed
        |-- W_UK_T / W_UV materialization
        |-- preserve/insert all_gather for dsa_indexer hidden_states when no upstream full-token boundary exists
        |-- SFA op dispatch
        |
        v
Performance model + memory tracker
        |-- SparseFlashAttention mapping
        |-- dsa_indexer top-k modeled on local tokens
        |-- dsa_indexer cache/compressor modeled on full tokens
        |-- profiling / analytic fallback
        |-- static and forward memory accounting
```

Module responsibilities:

- Structure detection: detect DeepSeek-style sparse-attention capability without a model-name allowlist.
- Sequence parallel pass: reuse `tp_group`, rewrite shardable collective chains, slice full-token DSA attention output before the replicated `o_proj`, gather only normalized outputs, and keep residual outputs local across layers.
- Linear layers: interpret `disable_tp` consistently across RowParallelLinear and ColumnParallelLinear.
- Attention layers: ensure the `hidden_states` input to `dsa_indexer` has full-token layout when required. Prefer reusing the `all_gather` inserted by upstream sequence-parallel norm patterns; add a dedicated consumer-boundary pattern only if debugging confirms a local norm output can feed `dsa_indexer` directly. Attention layers also dispatch the SFA op for the sparse-attention path and avoid interpreting weight quantization as SFA input quantization.
- Performance model: map the SFA op to the real `SparseFlashAttention` profile; use analytic fallback when profile data is missing. For the current fused `dsa_indexer`, split internal cost accounting so the top-k query side uses local token count, while cache/compressor update and full-sequence boundary work use full token count.
- Memory accounting: include quantization auxiliary tensors, derived weights, and full-layout weights.

## 5. Ongoing Work and Limitations

Already available in the current baseline:

- Profiling datasource wrapping fix.
- Basic `kv_b_proj.disable_tp` support for full-layout derived weights.
- Sparse-attention three-value return support in the current transformers baseline.
- DSA full-head MLA initialization through an effective singleton TP view while retaining the shared TP group for sequence collectives.
- Full-token attention to local-token replicated-`o_proj` rewrite, including distinct normalized/residual output handling.
- Equivalent execution with repetition enabled and disabled; the expanded 78-layer graph must match all 78 DSA boundaries without multiplying the residual token range.

Ongoing work:

- Extend core DSA CP regression coverage beyond the current shape/layout and repetition-equivalence cases.
- Re-implement SFA op dispatch and add decomposer / op mapping / fallback.
- Use `quant_model_description.json` to improve quantized-weight classification.

Limitations:

- NZ-format extra memory is not modeled in the first phase.
- vLLM-Ascend layer-sharding broadcast is not modeled in the first phase. Real DSA-CP can distribute full-layout weights such as `o_proj` / `q_b_proj` by layer and asynchronously broadcast them to the current rank before forward execution. This affects weight residency, temporary shard windows, and peak-memory estimation, but it does not block the core functional path in this RFC.
- PCP/DCP is not included in the first phase.
- ServingCast is not changed in the first phase.
- When SFA profiling data is missing, performance relies on analytic fallback and cannot fully match real profiles.
- Before static memory accounting moves to a MemoryTracker-aligned path, it can still diverge from real execution-time materialization.
