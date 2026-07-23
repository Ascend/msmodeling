# RFC: Profiling Interpolation Phase2 Operator Capability Expansion

## Metadata

| Item | Value |
| --- | --- |
| RFC name | Profiling Interpolation Phase2 Operator Capability Expansion |
| Status | Draft |
| Phase | Phase 2 of 3 |
| Target module | `tensor_cast/performance_model/profiling_database` |
| Baseline | Phase1 interpolation behavior after PR262 |
| Related PRs | #262, #366, #388, #389 |
| Prerequisites | [PR #555](https://gitcode.com/Ascend/msmodeling/pull/555), [PR #593](https://gitcode.com/Ascend/msmodeling/pull/593), and the implementation of [Issue #272](https://gitcode.com/Ascend/msmodeling/issues/272) |
| Updated | 2026-07-22 |

The Chinese and English RFCs describe the same capability scope and runtime semantics. Any change to default behavior, interpolation axes, regimes, fallback conditions, or acceptance criteria must update both documents. Their section numbers, capability and acceptance tables, and Section 7 test matrix are normative structural peers; prose line counts need not match because of translation, but an implementation PR must include a bilingual structure diff and identify any intentional wording-only difference.

## 1. Overview

Phase2 extends the Phase1 `InterpolatingDataSource`. It does not create another datasource or a second interpolation foundation.

Phase1 already provides the wrapper entry point, global disable switch, candidate index, interpolation math, latency guard, failure details, and compute and attention interpolation. Phase2 reuses these components and adds capability-specific targets, regimes, and dispatch for the non-communication operator classes in scope. A path with incomplete data or child coverage remains a diagnosable miss and is not presented as effective interpolation coverage.

### 1.1 Scope

| Operator category | Phase2 scope | Main operators or kernels |
| --- | --- | --- |
| MOE/DFC | Add a specialized target, regime, and interpolation path for DFC | `DispatchFFNCombine` |
| Elementwise | Add broadcast-aware and shared-token-aware elementwise interpolation | `Add`, `Mul`, `Div`, and their AiCore or alternate kernel variants |
| Attention | Add LightningIndexer and conditional SparseFlashAttention leaf-operator interpolation | `LightningIndexer`, `SparseFlashAttention` |
| Compute | Refine generic compute mappings and add quant-scale and MLA cache-update subcases | Concat, Gather, Cast, and LayerNorm variants; `DynamicQuant`, `DynamicBlockQuant`, `QuantBatchMatmulV3`, and `ScatterNdUpdate` |

This table lists only leaf-operator capabilities added or extended in Phase2: a specialized MOE path for DFC, elementwise support, and compute and attention subcategories. Upstream passes only independently queryable leaf operators to Phase2.

The rest of this RFC consistently uses the hierarchy "operator category -> subcategory/path." Kernel variants, axis groups, and concrete dtypes remain details of their owning subcategory rather than separate operator categories.

Versioned profiling databases and `op_mapping.yaml` files do not have to contain identical operators. Except for explicit accounting rules such as `zero_cost` and `accepted_miss`, a data-driven path returns a result only when its mapping, runtime target, and profiling CSV are all available.

### 1.2 Goals

Phase2 provides:

1. New or completed profiling interpolation paths for the subcategories added or adjusted in Phase2.
2. Explainable continuous axes and exact regime fields for every path.
3. Low-dimension-first 1D, 2D, or 3D interpolation when the available data supports it.
4. Stable fallback when candidates are insufficient, semantics differ, or data is missing.
5. Phase1-equivalent failure reasons, success details, latency guards, tests, and reports for the added paths.
6. Reuse of leaf-operator invocations emitted upstream, without deriving or aggregating children inside the interpolation module.

### 1.3 Non-goals

Phase2 does not include:

- a second wrapper interpolation path for communication;
- extrapolation or `QuerySource.EXTRAPOLATED`;
- a new public CandidateIndex, math backend, or datasource;
- parent decomposition, child aggregation, or overlap and pipeline latency aggregation;
- accuracy claims for FP8, MXFP4, or DynamicBlockQuant without real profiling data.

## 2. Design principles and Phase1 reuse

Phase2 uses the [Phase1 RFC](./rfc_profiling_operator_interpolation_phase1_en.md) as its runtime baseline. This document defines only the Phase2 delta.

### 2.1 Reused behavior

| Phase1 capability | Phase2 usage |
| --- | --- |
| `InterpolatingDataSource` | The only wrapper entry point and the Phase2 implementation owner. It handles independently queryable leaf operators only and does not perform upstream decomposition or aggregation. |
| `CandidateIndex` / `CandidateGroup` | Candidate organization, regime grouping, and interpolation. |
| `interpolation_math.py` | Existing 1D linear interpolation and 2D / 3D geometry checks. |
| `--disable-profiling-interpolation` | Global fallback to base `ProfilingDataSource`. |
| `interpolation_policy.kernel_overrides` | `max_interpolation_dim` narrows the maximum dimension for one kernel. |
| Failure and success details | Existing source, method, axes, boundary, candidate, and failure fields. |

Phase2 introduces no new interpolation mathematics. 1D continues to use linear interpolation between adjacent measured points. 2D and 3D continue to use linear simplex interpolation inside the convex hull, together with the Phase1 degeneracy, boundary, and latency-validity checks. This is mathematically aligned with the historical aiconfigurator implementation based on 1D linear interpolation and 2D/3D `griddata(method="linear")`. The current aiconfigurator `perf_interp v2` Grid/ScatteredSites and extrapolation capabilities are outside Phase2 and are not promised for a later phase by this RFC.

### 2.2 Phase2 principles

- Minimal implementation: add only the local logic required to build and match a capability target.
- Occam's razor: do not introduce a platform or future-facing interface without a concrete blocker.
- Exact first: preserve complete base exact results except where a specialized path requires semantic isolation, and check same-regime exact coordinates before interpolation. For Elementwise with a valid non-scalar output, the wrapper bypasses the base cross-dtype byte-ratio lookup and checks the full output shape with the exact profiling dtype.
- Low dimension first: within the configured ceiling, try the 1D axes listed by each subcategory before 2D and 3D axis groups. When one dimension has multiple groups, use the order explicitly listed by that subcategory.
- Same semantics: never interpolate across incompatible discrete regimes.
- Explainable failure: return a miss reason when a path or candidate set is invalid.
- No extrapolation: fall back when the target is outside the valid boundary or convex hull.

## 3. Phase2 runtime delta

### 3.1 Dispatch

```text
InterpolatingDataSource.lookup(op)
  |
  |-- receive one leaf op from upstream
  |     -> read mapping and classify exact-match ownership
  |
  |-- base exact eligible
  |     -> base.lookup(op)
  |          |-- complete result -> return base result
  |          `-- PARTIAL / None -> Phase2 fallback dispatch
  |
  |-- specialized exact ownership
  |     -> moe_fused / compute_scale / quantized_matmul / non-scalar elementwise
  |     -> skip semantically incomplete input-shape-only base exact lookup
  |     -> check exact coordinate in specialized CandidateIndex
  |          |-- exact coordinate -> return MEASURED
  |          `-- non-exact / miss -> Phase2 fallback dispatch
  |
  v
Phase2 fallback dispatch
  |-- MOE/DFC specialized path
  |     `-- query_mode: moe_fused
  |           -> DispatchFFNCombine
  |-- Elementwise
  |     `-- query_mode: elementwise
  |           -> Broadcast / Shared-token
  |-- Attention
  |     |-- query_mode: attention_lightning_indexer
  |     |     -> LightningIndexer
  |     `-- query_mode: attention_sparse_sharedkv
  |           -> SparseFlashAttention
  |-- Compute
  |     |-- ordinary mapping
  |     |     -> Phase1 generic compute
  |     |-- compute_subcategory: compute_scale
  |     |     -> Dynamic quant scale
  |     |-- compute_subcategory: quantized_matmul
  |     |     -> Quantized matrix compute
  |     |-- query_mode: scatter_nd_update_mla
  |     |     -> MLA cache update
  |     `-- unknown compute_subcategory
  |           -> fail closed
  |-- query_mode: attention_special ---> Phase1 attention
  `-- category: communication ---------> base owns the path; wrapper misses
```

`moe_fused`, `compute_scale`, `quantized_matmul`, and valid non-scalar Elementwise paths use specialized exact ownership. The DFC base mapping consumes only the activation input and cannot validate complete GMM weight shapes; the other three base queries lack output-signature, scale-mode, complete-quant, or exact-dtype semantics. Other upstream leaf operators first enter a semantically complete base exact lookup and try interpolation only after a miss. Phase2 neither derives one operator's shape from another invocation nor aggregates multiple operators.

### 3.2 Prerequisite PRs, Issue, and merge gate

The Phase2 leaf-operator interface depends on these upstream changes:

| Prerequisite | Capability provided | Phase2 dependency |
| --- | --- | --- |
| [PR #555](https://gitcode.com/Ascend/msmodeling/pull/555) | DSA CP frontend and runtime-layout foundation | It provides the DeepSeek V4 `quant_lightning_indexer`, `sparse_attn_sharedkv`, and `scatter_nd_update_mla` semantic entries and forms workloads with the actual TP/SP and prefill/decode semantics. |
| [PR #593](https://gitcode.com/Ascend/msmodeling/pull/593) | Profiling semantics and exact-match foundation for GLM5 DSA CP | It refines the shape, dtype, phase, cache, and SparseFlashAttention child semantics of GLM5 `mla_sparse_attention`; this parent path is not the same as the DeepSeek V4 `sparse_attn_sharedkv` leaf entry. |
| [Issue #272](https://gitcode.com/Ascend/msmodeling/issues/272) | Unified upstream decomposition and leaf-query contract | It defines a canonical leaf descriptor consumable across DeepSeek V4 and GLM5, or explicit versioned leaf entries. Exact lookup and interpolation must consume the same shape, dtype, phase, topk, and request/sequence semantics; multi-operator latency is aggregated outside the interpolation module. |

Issue #272 may be implemented by follow-up commits on PR #593 or by a separate companion PR, but its acceptance criteria must be present in the Phase2 target baseline. With PR #555 and the current PR #593 alone, leaf paths that depend on upstream decomposition cannot merge as completed Phase2 capabilities.

Before PR #388 and PR #389 merge, integration must:

1. rebase onto a target branch containing PR #555, PR #593, and the Issue #272 implementation;
2. keep `profiling_data_source.py` at zero diff from that target branch;
3. remove parent decomposition, child-description dependencies, and aggregation logic from `InterpolatingDataSource`;
4. retain leaf targets, candidate indexes, interpolation, and diagnostics;
5. prove with integration tests that one leaf first uses exact lookup, then interpolation after a miss, and that upstream aggregates latency exactly once.

If the prerequisite contract is missing, the affected path remains outside Phase2 scope. Wrapper-local decomposition or a Phase2 modification to `ProfilingDataSource` is not an acceptable temporary substitute.

### 3.3 Exact coordinates and result sources

When base exact lookup is enabled, a complete base result is returned first. The DFC base mapping matches only activation inputs, the base Elementwise query can scale latency by dtype byte ratio, and base lookup for `compute_scale` and `quantized_matmul` lacks output-signature, scale-mode, or complete quant semantics. These four paths therefore perform exact matching and interpolation in specialized wrapper indexes.

After entering the Phase2 fallback, DFC, specialized attention, cache-update, Elementwise, `compute_scale`, and `quantized_matmul` paths all check exact coordinates inside their CandidateIndex:

- if all continuous target axes equal a measured point in the same regime, return `QuerySource.MEASURED`;
- if a non-exact target is successfully interpolated, return `QuerySource.INTERPOLATED`;
- if candidates are insufficient, regimes differ, latency is invalid, or the target is out of range, return a miss;
- if the base result was a valid `PARTIAL`, Phase1 PARTIAL preservation applies after interpolation fails.

### 3.4 Phase2 operator coverage and dimension ceilings

Interpolation keeps the Phase1 low-dimension-first rule, which is not redefined here. A ceiling below is an implementation limit, not a claim that the current CSV data forms a valid boundary at that dimension.

#### 3.4.1 Operator paths implemented or adjusted in Phase2

This table lists only operator entries implemented, extended, or adjusted in Phase2, with all axis combinations for one operator kept in the same row. Operators that only reuse unchanged Phase1 behavior are omitted. Operators with incomplete implementations follow Section 3.4.2; implemented paths with only data or validation gaps remain here, with their boundaries documented in the corresponding capability sections.

For MOE/DFC, Phase2 adds only the specialized `DispatchFFNCombine` interpolation path.

Coordinate and semantic-field names in the table mean the following. Each subcategory defines whether a field is a continuous interpolation axis or an exact regime field:

- `axis_0`: the first shape dimension after removing an optional leading `batch=1`; all remaining dimensions must stay fixed when this axis is used;
- `axis_1`: the second output-shape dimension, used only for rank-2 elementwise outputs;
- `output_numel`: the product of all output-shape dimensions;
- `tokens`: token rows after flattening semantic token dimensions, with the exact source defined by each subcategory;
- `cache_blocks`: the first KV-cache shape dimension, representing the number of cache blocks;
- `topk`: the number of experts or indices selected per token;
- `M / K / N`: matrix rows, reduction width, and output columns. Non-matrix DynamicQuant uses only flattened rows `M` and final input width `K`.

| Operator category | Subcategory/path | Operator entry | Versions | Axis combinations | Ceiling | Basis |
| --- | --- | --- | --- | --- | ---: | --- |
| MOE/DFC | DispatchFFNCombine | `tensor_cast.dispatch_ffn_combine.default` | v0.15, v0.18 | `tokens` | 1D | The target path is reachable. v0.15 lacks EP data and v0.18 uses INT8 weights, so neither establishes data equivalence for the plain unquantized entry; the current path stably misses. |
| MOE/DFC | DispatchFFNCombine | `tensor_cast.dispatch_ffn_combine_quant.default` | v0.15, v0.18 | `tokens` | 1D | The v0.18 BF16-activation plus INT8-weight signature supports W8A8. v0.15 lacks EP data and stably misses. Complete GMM1/GMM2 weight shapes, hidden size, topk, EP, dtype, layout, and subtype are regime fields. |
| MOE/DFC | DispatchFFNCombine | `tensor_cast.dispatch_ffn_combine_quant_int4.default` | v0.15, v0.18 | `tokens` | 1D | The target path is reachable. There is currently no semantically matching INT4 measured row, so the path returns a stable miss. |
| MOE/DFC | DispatchFFNCombine | `tensor_cast.dispatch_ffn_combine_fp8.default` | v0.15, v0.18 | `tokens` | 1D | The target path is reachable. There is currently no semantically matching FP8 measured row, so the path returns a stable miss. |
| MOE/DFC | DispatchFFNCombine | `tensor_cast.dispatch_ffn_combine_mxfp4.default` | v0.15, v0.18 | `tokens` | 1D | The target path is reachable. There is currently no semantically matching MXFP4 measured row, so the path returns a stable miss. |
| Elementwise | Broadcast / Shared-token | `aten.add.Tensor` | v0.13, v0.15, v0.18 | `axis_0`; rank-2 also uses `axis_1` and `axis_0 + axis_1` | 2D | Uses direct output-shape dimensions; broadcast signature, rank, dtype, and kernel are regime fields. |
| Elementwise | Broadcast / Shared-token | `aten.mul.Tensor` | v0.13, v0.15, v0.18 | v0.13: `axis_0`; v0.15/v0.18: same as `aten.add.Tensor` | v0.13: 1D; v0.15/v0.18: 2D | v0.13 reuses generic compute; later versions enter the broadcast and shared-token path. |
| Elementwise | Broadcast / Shared-token | `aten.div.Tensor` | v0.13, v0.15, v0.18 | v0.13: `axis_0`; v0.15: same as `aten.add.Tensor`; v0.18: not interpolated | v0.13: 1D; v0.15: 2D; v0.18: zero-cost | v0.13 reuses generic compute, v0.15 uses the elementwise path, and v0.18 is explicitly treated as zero-cost by the mapping. |
| Attention | LightningIndexer | `tensor_cast.quant_lightning_indexer.default` | v0.18 | `tokens`; `cache_blocks`; `tokens + cache_blocks` | 2D | `topk` and request/sequence metadata grouping are regime fields; real CSV data currently has only one `topk` value. |
| Attention | SparseFlashAttention | `tensor_cast.sparse_attn_sharedkv.default` (DeepSeek V4); the GLM5 entry is unified by Issue #272 | v0.18 | `tokens` | 1D (after prerequisites) | v0.18 has measured CSV data, but rows with equal tokens / cache / topk still carry different request/sequence metadata. Interpolation is enabled only when the canonical leaf supplies equivalent grouping fields; otherwise the path returns a stable miss. |
| Compute | Concat | `aten.cat.default` | v0.13, v0.18; zero-cost in v0.15 | `axis_0` | 1D | Versions with `ConcatD.csv` use generic compute. |
| Compute | Concat | `tensor_cast.cat.default` | v0.13, v0.18; zero-cost in v0.15 | `axis_0` | 1D | Reuses the `aten.cat.default` kernel policy. |
| Compute | Gather | `aten.embedding.default` | v0.13, v0.15, v0.18 | `output_numel` | 1D | Gather policy uses output element count; alternates inherit the axis. |
| Compute | Cast | `aten.to.dtype` | v0.13, v0.15, v0.18 | `axis_0` | 1D | Cast and TensorMove variants use generic compute. |
| Compute | LayerNorm | `aten.native_layer_norm.default` | v0.18 | `axis_0` | 1D | `tc_input_count: 3` excludes non-tensor metadata. |
| Compute | Dynamic quant | `tensor_cast.dynamic_quantize_symmetric.default` | v0.13, v0.15, v0.18 | `M`; `K`; `M + K` | 2D | Output signature, scale mode, dtype, and format are regime fields. |
| Compute | Dynamic quant | `tensor_cast.dynamic_quantize_asymmetric.default` | v0.13, v0.15, v0.18 | `M`; `K`; `M + K` | 2D | The asymmetric offset output is isolated from symmetric candidates. |
| Compute | Dynamic block quant | `tensor_cast.dynamic_quantize_mxfp4.default` | v0.13, v0.15, v0.18 | `M`; `K`; `M + K` | 2D | The path exists, but no `DynamicBlockQuant.csv` is checked in. |
| Compute | Quantized matrix | `tensor_cast.static_quant_linear.default` | v0.13, v0.15, v0.18 | `M`; `K`; `N`; three 2D pairs; `M + K + N` | 3D | Reuses MatMul axes with INT8 dtype, layout, and output signature isolated. |
| Compute | Quantized matrix | `tensor_cast.static_quant_linear_int4.default` | v0.13, v0.15, v0.18 | Same as `static_quant_linear` | 3D | Packed INT4 weight semantics remain isolated. |
| Compute | Quantized matrix | `tensor_cast.fp8_linear.default` | v0.13, v0.15, v0.18 | Same as `static_quant_linear` | 3D | The path exists, but no same-dtype FP8 data is checked in. |
| Compute | Quantized matrix | `tensor_cast.mxfp4_linear.default` | v0.13, v0.15, v0.18 | Same as `static_quant_linear` | 3D | The path exists, but no same-dtype MXFP4 data is checked in. |
| Compute | MLA cache update | `tensor_cast.scatter_nd_update_mla.default` | v0.18 | `tokens` | 1D | Cache capacity and update tokens do not form an independent 2D grid. |

#### 3.4.2 Operators currently uncovered or intentionally not interpolated

Entries are grouped by implementation status and include only operators without a complete interpolation path or operators explicitly excluded from interpolation. Operators with an implemented path but missing CSV data, dtype coverage, or accuracy validation are not listed here; their data and validation boundaries remain in the corresponding capability sections and Section 8.

| Type | Included operators | Versions | Reason and current behavior |
| --- | --- | --- | --- |
| Outside Phase2 | `tensor_cast.matmul_all_reduce.default`, `tensor_cast.static_quant_linear_all_reduce.default`, `tensor_cast.static_quant_linear_int4_all_reduce.default`, `tensor_cast.fp8_linear_all_reduce.default`, `tensor_cast.mxfp4_linear_all_reduce.default` | All | These operators contain communication children. Communication operators are not interpolated by this interpolation module. |
| Absorbed by fusion; explicitly not interpolated | `tensor_cast.concat_and_cache_mla.default` | v0.18 | MLA cache-write latency is fused into and accounted by `KvRmsNormRopeCache`. Interpolating it separately would double count latency, so the mapping returns documented zero latency. This changes only if a future version restores a standalone kernel. |
| Conditional accepted miss | `aten.index.Tensor` | v0.18 | Some calls in the target workload are RoPE-cache artifacts introduced by TC CachingRotaryEmb and have no standalone NPU kernel, so the current mapping returns zero latency. Other contexts may contain a real `Index` kernel; unconditional accepted miss can undercount until call semantics can be distinguished. |

Operators marked `zero_cost: true`, ordinary communication, and internal `profiling.*` kernels are not listed as uncovered. The first two have explicit accounting ownership; an internal kernel becomes a gap only when it appears as a real top-level TC trace operation and affects B2B or holdout results.

## 4. Capability design

### 4.1 MOE/DFC

#### 4.1.1 DispatchFFNCombine

The specialized `DispatchFFNCombine` path uses `query_mode: moe_fused`. Versioned mappings recognize its plain, W8A8, INT4, FP8, and MXFP4 semantic entries. A reachable entry does not imply that semantically reusable measured data exists.

##### Target and continuous axes

For a first activation shape `[d0, d1, ..., dn, hidden]`, `tokens = d0 * d1 * ... * dn`, representing the total token count in the activation. For example, `[2, 4, 72, 7168]` produces `tokens=576`.

Real DFC profiling CSV data records the activation as flattened `[tokens, hidden]`, so grouping information from the original leading dimensions cannot be recovered. The current path therefore uses only the database-backed `tokens` value as a continuous axis and does not synthesize additional axes from the runtime shape. This also matches the aiconfigurator MOE query: model and parallel settings select a bucket, then 1D interpolation is performed over `num_tokens` inside that bucket.

The current axis group is:

```text
1D: tokens
```

`topk`, hidden size, and EP size are exact regime fields. They are not continuous axes. `local_experts` must remain fixed when both the target and candidate expose it.

##### Regime and failure boundaries

Candidates are isolated by at least:

- kernel type;
- activation dtype and format;
- quant subtype and weight dtype/layout;
- complete GMM1/GMM2 weight shapes;
- hidden size;
- topk;
- EP size.

The target derives a `plain / w8a8 / int4 / fp8 / mxfp4` subtype from the TensorCast semantic entry and extracts complete GMM1/GMM2 shapes from the corresponding operator signature. A candidate extracts the same fields from the second and third CSV inputs. Missing weight shapes, an unknown subtype, or a mismatch rejects the candidate; equal activation dtype alone never permits latency reuse.

Current `DispatchFFNCombine.csv` data contains only BF16 activation plus INT8 weight signatures and therefore supports only the corresponding W8A8 path. The v0.15 CSV also lacks an `EP Size` field and cannot form a complete EP regime, so current effective data coverage is limited to v0.18 W8A8. Plain, INT4, FP8, MXFP4, and v0.15 targets return stable misses until semantically matching fields and measured data are added; they do not borrow W8A8 rows to increase hit rate.

EP size is a required regime field. The CSV must contain an `EP Size` column and every row must provide a valid positive integer. A missing column, blank value, or invalid value cannot match any target EP. If runtime EP is not configured, the path returns `ep_size_not_configured`.

The path misses when topk cannot be extracted, EP differs, candidates cannot form a boundary, or latency is invalid. It does not fall through to ordinary compute interpolation.

### 4.2 Elementwise

#### 4.2.1 Broadcast / Shared-token

The specialized Phase2 broadcast and shared-token path uses `query_mode: elementwise`. In v0.13, `aten.mul.Tensor` and `aten.div.Tensor` still use generic compute; other applicable versions map the kernels they actually contain, including `Add`, `Mul`, `Div`, and alternate variants such as `AddAiCore`, `MulAiCore`, or `RealDiv`.

##### Target and continuous axes

Continuous axes use direct output-shape dimensions:

- `axis_0`: the first output-shape dimension, or the next dimension when the leading dimension is `batch=1`;
- `axis_1`: the second dimension, used only when the output is rank 2 after removing an optional leading `batch=1`.

Axis groups are attempted in this order:

```text
1D: axis_0
1D: axis_1 (with axis_0 fixed)
2D: axis_0 + axis_1
```

The 1D `axis_0` fallback requires an equal output-shape tail; the 1D `axis_1` fallback requires a fixed `axis_0`. Only rank-2 outputs continue with 2D `axis_0 + axis_1`, and candidates must form a non-degenerate boundary. Outputs above rank 2 use only 1D `axis_0` with the complete tail fixed. The path no longer uses `output_numel + axis_0`, avoiding the nonlinear coordinate transform `output_numel = axis_0 * axis_1` before linear interpolation.

##### Regime, dtype behavior, and failure boundaries

Candidates are isolated by:

- kernel type;
- output rank;
- input count;
- broadcast pattern;
- target output dtype.

Candidate dtype must exactly match the target dtype. Phase2 does not scale measured latency by dtype byte width; missing same-dtype data returns a diagnostic miss. One interpolation also cannot mix candidates from the selected latency column and fallback latency columns.

Alternate kernels are tried in mapping order. If all kernels fail, details retain each attempted kernel and miss result.

### 4.3 Attention

#### 4.3.1 LightningIndexer

`tensor_cast.quant_lightning_indexer.default` uses:

```yaml
kernel_type: LightningIndexer
query_mode: attention_lightning_indexer
```

##### Target and continuous axes

The continuous axes are defined as follows:

- `tokens`: the number of token rows in the query; a 3D query `[T, H, D]` uses `T`, while a 4D query `[B, S, H, D]` uses `B * S`;
- `cache_blocks`: the first cache-tensor dimension;
- `topk`: the final output-shape dimension, used only as an exact regime field rather than a continuous axis.

Axis groups are attempted in this order:

```text
1D: tokens
1D: cache_blocks (with tokens fixed)
2D: tokens + cache_blocks
```

The quant operator places the cache tensor at a different input index from the ordinary path, so target construction follows the actual operator signature. Real `LightningIndexer.csv` data currently fixes `topk` at `2048`, so the RFC does not claim 3D support with `topk` as a third axis. This also matches aiconfigurator, where `index_topk` is a configuration condition rather than a continuous interpolation axis.

Real CSV rows can share the same `tokens / cache_blocks / topk` coordinate while exposing different request or sequence metadata shapes and materially different latencies. The implementation must extract an equivalent request/sequence grouping field from both target and CSV and place it in the regime. If the semantics cannot be aligned reliably, the path fails closed instead of merging those rows into one coordinate.

##### Regime and failure boundaries

The regime contains kernel type, query profiling dtype, cache profiling dtype, the cache-shape tail after its first dimension, the first two input formats, query rank, heads, head dimension, `topk`, and request/sequence grouping. Only the first cache dimension, `cache_blocks`, is continuous; cache dtype and tail must match exactly. If both the CSV and mapping provide an API path, the paths must match. A CSV without that column uses the remaining regime fields and does not require a synthetic value. If any required semantic field cannot be extracted equivalently from both target and CSV, the path fails closed.

A same-regime exact coordinate returns `MEASURED`; other targets can use up to 2D interpolation. `topk`, request/sequence grouping, heads, head dimension, query/cache dtype, cache tail, format, and API path must match. Success and failure details include the kernel, API path, target axes, and rejected rows.

#### 4.3.2 SparseFlashAttention

`SparseFlashAttention` is a sparse-attention kernel with measured data in the profiling database. The current direct TensorCast semantic entry, `tensor_cast.sparse_attn_sharedkv.default`, is provided by the DeepSeek V4 frontend in PR #555 and maps to this kernel through `query_mode: attention_sparse_sharedkv`. The GLM5 path in PR #593 currently produces a SparseFlashAttention child from the `mla_sparse_attention` parent and is not the same direct entry. Issue #272 must first unify the canonical leaf contract for both paths.

Phase2 does not introduce new sparse-attention operator semantics or decompose `mla_sparse_attention`. It performs exact lookup and interpolation only for independent leaf operators supplied by upstream. A canonical leaf carries at least query/cache shape, dtype, phase, selected topk, and request/sequence grouping fields that can be aligned with the CSV. Currently only v0.18 provides `SparseFlashAttention.csv`; v0.13 and v0.15 have no measured data and therefore cannot interpolate this kernel.

##### Target and continuous axes

After the semantic-alignment requirements below are satisfied, the only continuous axis is `tokens`, with the same definition as LightningIndexer: a 3D query `[T, H, D]` uses `T`, while a 4D query `[B, S, H, D]` uses `B * S`.

Axis groups are attempted in this order:

```text
1D: tokens
```

##### Regime and failure boundaries

Operator semantics make `topk_indices.shape[-1]` the effective accessed KV length, while `cache_blocks` is cache capacity rather than effective `past_kv`. aiconfigurator sparse-attention interpolation uses actual `past_kv / isl / batch` and does not substitute cache capacity for effective length. Because the current CSV cannot reliably recover an equivalent effective KV length, Phase2 does not interpolate along `cache_blocks`.

In the v0.18 CSV, rows with equal `tokens`, query/cache dtypes, heads, head dimension, full cache shape, and selected topk still have different request/sequence metadata shapes and materially different latencies. These rows must not be median-aggregated as one coordinate. The canonical leaf and CSV must expose equivalent sparse-phase and request/sequence grouping fields and include them in the regime. If those semantics are undefined, missing on either side, or ambiguous, the entire path returns a stable miss.

The complete regime includes at least query profiling dtype, cache profiling dtype, the first two input formats, heads, head dimension, full cache shape, selected topk, sparse phase, and request/sequence grouping. SparseFlashAttention candidates are not mixed with LightningIndexer or Phase1 FusedInferAttentionScore candidates.

Both attention helpers receive independent leaf operators from upstream. When the wrapper is disabled, the specialized base paths provided by the prerequisites perform exact lookup without falling through to generic compute.

### 4.4 Compute

#### 4.4.1 Generic compute mapping refinements

These changes add no `query_mode`, target builder, or interpolation math. They continue to use the Phase1 generic compute path. Phase2 changes a mapping only when operator semantics and versioned profiling data justify it:

##### Mapping scope

- `aten.cat.default` and `tensor_cast.cat.default` map to `ConcatD` in versions that contain `ConcatD.csv` instead of being unconditionally marked `zero_cost`;
- embedding mappings add `GatherV3` and `GatherV2AiCore` as alternate kernels for `GatherV2` in the relevant version;
- cast mappings add `TensorMove`, `CastAiCore`, and `TensorMoveAiCore` as version-specific alternate kernels;
- `aten.native_layer_norm.default` maps to `LayerNormV3`, uses `LayerNormV3WithImplMode` as an alternate, and sets `tc_input_count: 3` to exclude normalized-shape and epsilon metadata.

##### Failure boundaries

Alternate kernels are tried in mapping order after the primary kernel misses. A version without the corresponding CSV or candidate returns a stable miss and does not borrow data from another version.

#### 4.4.2 Quant-scale and quantized matmul

Quant-scale does not introduce a `query_mode`. It uses `compute_subcategory` to select specialized compute semantics.

##### Target and continuous axes

`compute_scale` derives `M = d0 * ... * dn` and `K` from a quantization input `[d0, ..., dn, K]`. `quantized_matmul` reuses the Phase1 MatMul target: inputs are interpreted as `M x K` and `K x N`, producing `M`, `K`, and `N`.

| `compute_subcategory` | Kernel | Axes | Maximum |
| --- | --- | --- | ---: |
| `compute_scale` | `DynamicQuant`, `DynamicBlockQuant` | `M`, `K`, `M + K` | 2D |
| `quantized_matmul` | `QuantBatchMatmulV3` | `M`, `K`, `N`, and their combinations | 3D |

##### Dynamic quant

The continuous axes are defined as follows:

- `M`: the product of all input dimensions except the final dimension;
- `K`: the final input dimension.

Scale output shapes identify:

- `per_tensor`;
- `per_token`;
- `per_channel`;
- `per_block`, only for `DynamicBlockQuant`.

The regime includes input dtype and format, output count, every output dtype and format, every auxiliary scale mode, and block size. A two-output symmetric row cannot satisfy a three-output asymmetric target.

Base compute exact matching checks only input shapes and cannot distinguish output signatures or scale modes. `compute_scale` therefore bypasses base exact matching and checks same-regime exact coordinates in its specialized index.

FP16 DynamicQuant matches the CSV `DT_FLOAT16` dtype. This local rule does not change Phase1 dtype compatibility for other compute paths.

##### Quantized matrix compute

`quantized_matmul` reuses the Phase1 MatMul M/K/N target and interpolation math. `tc_input_count: 2` aligns TensorCast `(x, weight, scale, ...)` with the two matrix inputs in `QuantBatchMatmulV3.csv`. Phase1 `scale_matrix` denotes the M/K 2D overhead of applying a scale during static quantization; it has different semantics and is not reused as this selector.

Static quant, INT4, FP8, and MXFP4 linear mappings may point to `QuantBatchMatmulV3`, but candidates remain isolated by actual input dtype and layout. Measured INT8 rows cannot satisfy FP8 or MXFP4 targets.

`tensor_cast.quantize.default` applies an existing scale through `AscendQuantV2`. It remains on generic compute and does not enter `compute_scale`.

##### Data and failure boundaries

The v0.13, v0.15, and v0.18 databases contain `DynamicQuant.csv` and `QuantBatchMatmulV3.csv`. DynamicQuant has BF16 rows, and v0.18 also has FP16 rows. Current QuantBatchMatmulV3 matrix inputs are INT8.

The repository does not contain `DynamicBlockQuant.csv` or demonstrated equivalent FP8 and MXFP4 QuantBatchMatmulV3 rows. Therefore:

- mapping and target construction can recognize DynamicBlockQuant, but lookup stably misses without a CSV;
- FP8 and MXFP4 linear do not reuse INT8 data;
- coverage and accuracy claims for those dtypes require real profiling data plus holdout and B2B evidence.

#### 4.4.3 MLA cache update

`tensor_cast.scatter_nd_update_mla.default` is a compute cache-update subcase:

```yaml
kernel_type: ScatterNdUpdate
alternate_kernel_types: [ScatterNdUpdateAiCore]
query_mode: scatter_nd_update_mla
```

##### Input normalization

TensorCast arguments are `(update, cache, index)`, while the ScatterNdUpdate CSV uses `(cache, index, update)`. Target construction normalizes the order before matching candidates.

##### Target and continuous axes

The continuous axis is defined as follows:

- `tokens`: the first update-shape dimension for ranks up to 2; for higher ranks, the product of all dimensions except the final dimension.

The final dimension remains the update tail. The current path interpolates only 1D `tokens`.

##### Regime and failure boundaries

The regime includes cache, index, and update dtypes, the complete cache shape, update tail, and all three input formats. Cache capacity is not a continuous axis in this path, so candidates with different complete cache shapes cannot be shared. `ScatterNdUpdateAiCore` is tried after the primary kernel misses.

This path receives an independent cache-update leaf operator. After a base exact miss, a same-regime exact coordinate in the specialized index returns `MEASURED`; non-exact coordinates enter 1D interpolation.

## 5. Mapping and minimal policy

Phase2 uses existing `op_mapping.yaml` fields and does not add a separate configuration file.

| Field | Purpose |
| --- | --- |
| `kernel_type` | Primary profiling CSV and kernel |
| `alternate_kernel_types` | Ordered variants tried after the primary kernel misses |
| `query_mode` | Select a DFC, Elementwise, Attention, or Compute/cache-update path |
| `compute_subcategory` | Select `compute_scale` or `quantized_matmul` |
| `tc_input_count` | Match only the first N TensorCast inputs against the CSV |
| `interpolation_policy.kernel_overrides.<kernel>.max_interpolation_dim` | Narrow a kernel's maximum interpolation dimension |

| Operator category | Subcategory/path | Mapping selector |
| --- | --- | --- |
| MOE/DFC | DispatchFFNCombine | `query_mode: moe_fused` |
| Elementwise | Broadcast / Shared-token | `query_mode: elementwise` |
| Attention | LightningIndexer | `query_mode: attention_lightning_indexer` |
| Attention | SparseFlashAttention | `query_mode: attention_sparse_sharedkv` |
| Compute | Generic compute mapping refinements | `kernel_type`, `alternate_kernel_types`, `tc_input_count` |
| Compute | Quant-scale / Dynamic quant | `compute_subcategory: compute_scale` |
| Compute | Quant-scale / Quantized matrix compute | `compute_subcategory: quantized_matmul`, `tc_input_count: 2` |
| Compute | MLA cache update | `query_mode: scatter_nd_update_mla` |

`attention_special` and ordinary compute mappings without a Phase2 selector continue to use Phase1 paths.

An unknown `compute_subcategory` returns `compute_subcategory_unknown`. Base-owned leaves first use the exact lookup provided by the prerequisites and enter a wrapper path only after a miss. `moe_fused`, `compute_scale`, `quantized_matmul`, and valid non-scalar Elementwise paths instead use their specialized CandidateIndex for both exact matching and interpolation. Phase2 does not add a second global switch.

## 6. Result and diagnostic additions

Phase2 reuses Phase1 QueryResult semantics, latency guards, and candidate geometry checks. It adds only capability-specific location data.

### 6.1 Successful results

A successful leaf-operator interpolation records at least:

- `kernel_type`, `query_mode`, or `compute_subcategory`;
- `method`, `interpolation_dim`, and `axes`;
- boundary, candidate count, and matched rows;
- `fallback_from` and `interpolation_path`;
- capability fields such as EP, scale mode, or attention subcategory.

A specialized exact coordinate returns `MEASURED` and records `method=exact_coordinate` plus the matched row.

### 6.2 Failed results

A failure must record more than `None`. It should distinguish:

- missing mapping or kernel;
- unavailable target or required field;
- missing CSV;
- regime mismatch;
- insufficient or degenerate candidates;
- invalid latency;
- out-of-range target;
- a required dimension blocked by `max_interpolation_dim`;
- an upstream leaf missing required semantic fields.

Miss details contain target axes, target regime, attempted kernels, rejected-row counts, and the last geometry diagnostics. The caller then applies Phase1 PARTIAL preservation or analytic fallback.

### 6.3 Latency source

Candidate latency must be positive and finite. Paths with source-pure grouping, including elementwise, generic compute, and quant-scale, do not mix the selected latency column with fallback latency columns in one interpolation.

Zero latency is returned only by existing `zero_cost` mappings or evidence-backed `accepted_miss` mappings and is not a normal interpolation candidate.

## 7. Tests and acceptance

### 7.1 Test scope

Phase2 regression tests live in:

```text
tests/regression/tensor_cast/test_specialized_operator_interpolation.py
```

They run with:

```text
tests/regression/tensor_cast/test_profiling_interpolation_phase1.py
tests/regression/tensor_cast/test_interpolating_data_source.py
tests/benchmark/ops/perf_database/test_op_mapping_schema.py
```

### 7.2 Capability test matrix

| Operator category | Subcategory/path | Required behavior |
| --- | --- | --- |
| MOE/DFC | DispatchFFNCombine | v0.18 W8A8 1D token interpolation, complete GMM1/GMM2 weight-shape isolation, real-CSV duplicate-coordinate protection, out-of-range and insufficient-candidate fallback, topk / EP / local-expert isolation, missing/blank/invalid EP rejection, strict plain / W8A8 / INT4 / FP8 / MXFP4 subtype isolation, and stable misses when matching subtype data is unavailable |
| Elementwise | Broadcast / Shared-token | Direct-shape 2D, 1D fallback, max dimension, v0.13 generic-compute compatibility, full-output-shape exact coordinates, broadcast signature, alternate kernels, cross-dtype rejection, and latency-source purity |
| Attention | LightningIndexer | Guarded 2D, query/cache dtype and cache-tail isolation, topk and request/sequence grouping isolation, duplicate-coordinate rejection, max dimension, real quant mapping, API-path mismatch, missing-CSV details, and base generic exact rejection |
| Attention | SparseFlashAttention | Canonical leaf contract, token-only 1D, query/cache dtype, cache-shape / selected-topk / sparse-phase / request-and-sequence grouping isolation, no cross-semantic median aggregation of duplicate coordinates, fail-closed behavior when semantics are unavailable, and specialized exact |
| Compute | Generic compute mapping refinements | ConcatD data presence, Gather and Cast alternate order, LayerNorm metadata exclusion, and `tc_input_count` |
| Compute | Quant-scale / Dynamic quant | DynamicQuant 2D, output signature, scale mode, FP16, specialized exact, max dimension, and DynamicBlockQuant mode |
| Compute | Quant-scale / Quantized matrix compute | QuantBatchMatmul 3D, `tc_input_count`, and no INT8 reuse for FP8 or MXFP4 |
| Compute | MLA cache update | Argument reordering, complete-cache-shape isolation, real-CSV duplicate-coordinate protection, 1D interpolation, alternate kernel, and specialized exact |

Every subcategory or path also covers applicable target-construction failures, insufficient candidates, regime mismatch, invalid latency, out-of-range targets, success details, and miss details.

### 7.3 Acceptance criteria

Before merge, Phase2 must satisfy:

1. Every implemented leaf mapping that declares a Phase2 selector reaches its intended target builder; data gaps produce stable misses, while unverified upstream leaf sequences follow Section 3.4.2.
2. Same-regime exact and interpolated coordinates return the correct source.
3. `max_interpolation_dim` blocks dimensions above the ceiling without changing low-dimension-first order.
4. Specialized leaf operators use the correct base exact lookup and cannot produce a false generic-compute hit.
5. Candidates do not cross incompatible EP, topk, complete GMM weight shapes, shape tails, complete cache shapes, request/sequence grouping, API paths, dtypes, quant subtypes, or output signatures.
6. A DFC candidate is used only when its activation/weight dtypes, layout, complete GMM1/GMM2 weight shapes, and EP match the target subtype. Current effective data coverage is limited to v0.18 W8A8; v0.15 and plain, INT4, FP8, and MXFP4 paths return stable misses.
7. The SparseFlashAttention canonical leaf, sparse phase, and request/sequence grouping are equivalently extractable from target and CSV. The interpolation path remains disabled until this condition is met.
8. Prerequisite integration tests prove that upstream passes independently queryable leaf operators and that Phase2 performs neither decomposition nor multi-operator aggregation.
9. Phase1 compute, attention, PARTIAL, disable switch, and latency-guard regressions pass.
10. The Chinese and English RFCs match mappings, implementation, and tests.

Unit tests prove code paths and boundary behavior. Accuracy or runtime hit-rate claims also require holdout, endpoint B2B, or real profiling ground truth. Without that evidence, a report may claim only that the path is executable, not that it is accurate or improves performance.

A minimal report includes software and data versions, the database path, enabled configuration, source distribution, interpolation axes and dimensions, error statistics, failure-reason distribution, and unverified scope. Report formatting belongs to test reports rather than this RFC.

## 8. Compatibility, rollback, and known limits

### 8.1 Compatibility

- Phase2 does not change the public QueryResult type or datasource construction.
- Relative to a target branch containing the prerequisites, Phase2 does not modify `ProfilingDataSource`.
- Existing mappings without Phase2 fields retain Phase1 or base behavior.
- Versioned mappings follow the operator semantics of each version and need not be identical. An entry wired without a CSV stably misses; an entry without a complete child set follows Section 3.4.2 rather than claiming coverage.
- Missing optional CSV fields either use an explicitly defined compatibility path or produce a stable miss. They do not break database loading.
- Communication remains on the base exact and alpha-beta path.

### 8.2 Rollback

Users can disable the wrapper with `--disable-profiling-interpolation`. Maintainers can remove a specialized mapping, lower `max_interpolation_dim`, or add a stricter regime to narrow one path.

Rollback does not delete profiling data or change the analytic performance model.

### 8.3 Known limits

Section 3.4.1 lists paths implemented or adjusted in Phase2, while Section 3.4.2 lists only operators with incomplete implementations or explicit interpolation exclusions. CSV, dtype, and accuracy-validation boundaries remain in their corresponding capability sections. An implementation ceiling does not prove that the current database has enough candidate density. If a path produces no `INTERPOLATED` result at an endpoint, the evidence proves only that its code path and failure boundaries exist, not that the path is active in a real workload or sufficiently accurate.

### 8.4 Risk controls

| Scope | Risk | Control |
| --- | --- | --- |
| Common | Incorrect mapping classification | Explicit `query_mode` / `compute_subcategory`; specialized targets do not fall into generic compute |
| Common | Cross-semantic candidate reuse | Regime isolation for complete GMM weight shapes, shape tails, complete cache shapes, request/sequence grouping, API paths, EP, topk, dtypes, and output signatures |
| Common | Unstable high-dimensional accuracy | Low dimension first, per-kernel `max_interpolation_dim`, and holdout before retaining a higher ceiling |
| MOE/DFC | Cross-configuration reuse when DFC quant subtypes, FFN intermediate widths, or EP semantics differ | Add subtype, complete GMM1/GMM2 weight shapes, weight dtype/layout, and EP to the regime. Only v0.18 W8A8 currently has complete data; v0.15 and other subtypes stably miss |
| Elementwise / Broadcast / Shared-token | Base exact lookup scales across dtype, or nonlinear coordinates introduce error | Bypass base Elementwise lookup for valid non-scalar outputs; match full output shape and profiling dtype exactly; rank-2 uses direct shape dimensions and does not rescale latency by byte width |
| Attention / LightningIndexer | Synthetic 3D exceeds real data dimensionality, while equal 2D coordinates can hide different cache or sequence semantics | Treat query/cache dtype, cache tail, `topk`, and request/sequence grouping as regime fields, cap interpolation at `tokens + cache_blocks` 2D, and fail closed when semantics cannot be aligned |
| Attention / SparseFlashAttention | Cache capacity substitutes for effective KV length, or duplicate coordinates aggregate different sparse phases, requests, or sequence semantics | Use `tokens` 1D only; match query/cache dtype, full cache shape, selected topk, sparse phase, and request/sequence grouping exactly. Fail closed for the entire path when the canonical leaf or semantic fields are incomplete |
| Upstream leaf input | Duplicate accounting, missing semantics, or exact/interpolation workload drift | Issue #272 is a merge gate; base and wrapper consume the same leaf-operator descriptor |
| Common | Sparse profiling data | No extrapolation or cross-dtype borrowing; return a diagnostic miss |

Phase2 is complete when its operator paths, boundaries, and fallback behavior are explicit and testable. Paths without real data remain recognizable stable misses until profiling data is added and accuracy is evaluated.
