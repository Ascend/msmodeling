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
| Updated | 2026-08-03 |

The Chinese and English RFCs describe the same capability scope and runtime semantics. Any change to default behavior, interpolation axes, regimes, fallback conditions, or acceptance criteria must update both documents. Their section numbers, capability and acceptance tables, and Section 7 test matrix are normative structural peers; prose line counts need not match because of translation, but an implementation PR must include a bilingual structure diff and identify any intentional wording-only difference.

## 1. Overview

Phase2 extends the Phase1 `InterpolatingDataSource`. It does not create another datasource or a second interpolation foundation.

Phase1 already provides the wrapper entry point, global disable switch, candidate index, interpolation math, latency guard, failure details, and compute and attention interpolation. Phase2 reuses these components and adds capability-specific targets, regimes, and dispatch for the non-communication operator classes in scope. A path with incomplete data or child coverage remains a diagnosable miss and is not presented as effective interpolation coverage.

### 1.1 Scope

| Operator category | Phase2 scope | Main operators or kernels |
| --- | --- | --- |
| MOE/DFC | Add a specialized target, regime, and interpolation path for DFC | `DispatchFFNCombine` |
| Elementwise | Add broadcast-aware and shared-token-aware elementwise interpolation | `Add`, `Mul`, `Div`, and their AiCore or alternate kernel variants |
| Attention | Add LightningIndexer and SparseFlashAttention leaf-operator interpolation | `LightningIndexer`, `SparseFlashAttention` |
| Compute | Refine generic compute mappings and add quant-scale and MLA cache-update subcases | Concat, Gather, Cast, and LayerNorm variants; `DynamicQuant`, `DynamicBlockQuant`, `QuantBatchMatmulV3`, and `ScatterNdUpdate` |

This table lists only leaf-operator capabilities added or extended in Phase2: a specialized MOE path for DFC, elementwise support, and compute and attention subcategories. Composite paths obtain independently queryable leaf descriptors from existing decomposers.

The rest of this RFC consistently uses the hierarchy "operator category -> subcategory/path." Kernel variants, axis groups, and concrete dtypes remain details of their owning subcategory rather than separate operator categories.

Versioned profiling databases and `op_mapping.yaml` files do not have to contain identical operators. Except for explicit accounting rules such as `zero_cost` and `accepted_miss`, a data-driven path returns a result only when its mapping, runtime target, and profiling CSV are all available.

### 1.2 Goals

Phase2 provides:

1. New or completed profiling interpolation paths for the subcategories added or adjusted in Phase2.
2. Explainable continuous axes and exact regime fields for every path.
3. Low-dimension-first 1D, 2D, or 3D interpolation when the available data supports it.
4. Stable fallback when candidates are insufficient, semantics differ, or data is missing.
5. Phase1-equivalent failure reasons, success details, latency guards, tests, and reports for the added paths.
6. Reuse of existing `ProfilingDataSource` decomposers and `SubKernelSpec` objects, without adding decomposition algorithms to the interpolation module.

### 1.3 Non-goals

Phase2 does not include:

- a second wrapper interpolation path for communication;
- extrapolation or `QuerySource.EXTRAPOLATED`;
- a new public CandidateIndex, math backend, or datasource;
- new parent-decomposition rules or overlap and pipeline latency models;
- accuracy claims for FP8, MXFP4, or DynamicBlockQuant without real profiling data.

## 2. Design principles and Phase1 reuse

Phase2 uses the [Phase1 RFC](./rfc_profiling_operator_interpolation_phase1_en.md) as its runtime baseline. This document defines only the Phase2 delta.

### 2.1 Reused behavior

| Phase1 capability | Phase2 usage |
| --- | --- |
| `InterpolatingDataSource` | The only wrapper entry point and the Phase2 implementation owner. It reuses existing decomposers to obtain leaf descriptors and defines no new decomposition rules. |
| `CandidateIndex` / `CandidateGroup` | Candidate organization, regime grouping, and interpolation. |
| `interpolation_math.py` | Existing 1D linear interpolation and 2D / 3D geometry checks. |
| `--disable-profiling-interpolation` | Global fallback to base `ProfilingDataSource`. |
| `interpolation_policy.kernel_overrides` | `max_interpolation_dim` narrows the maximum dimension for one kernel. |
| Failure and success details | Existing source, method, axes, boundary, candidate, and failure fields. |

Phase2 introduces no new interpolation mathematics. 1D continues to use linear interpolation between adjacent measured points. 2D and 3D continue to use linear simplex interpolation inside the convex hull, together with the Phase1 degeneracy, boundary, and latency-validity checks. This is mathematically aligned with the historical aiconfigurator implementation based on 1D linear interpolation and 2D/3D `griddata(method="linear")`. The current aiconfigurator `perf_interp v2` Grid/ScatteredSites and extrapolation capabilities are outside Phase2 and are not promised for a later phase by this RFC.

### 2.2 Phase2 principles

- Minimal implementation: add only the local logic required to build and match a capability target.
- Occam's razor: do not introduce a platform or future-facing interface without a concrete blocker.
- Exact first with one owner: every operator first calls base `ProfilingDataSource.lookup()`. A complete result is returned unchanged; only `PARTIAL` or a miss enters interpolation. The wrapper neither bypasses base lookup nor performs a second exact lookup in its candidate indexes.
- Low dimension first: within the configured ceiling, try the 1D axes listed by each subcategory before 2D and 3D axis groups. When one dimension has multiple groups, use the order explicitly listed by that subcategory.
- Same semantics: never interpolate across incompatible discrete regimes.
- Explainable failure: return a miss reason when a path or candidate set is invalid.
- No extrapolation: fall back when the target is outside the valid boundary or convex hull.

## 3. Phase2 runtime delta

### 3.1 Dispatch

```text
InterpolatingDataSource.lookup(op)
  |
  |-- base.lookup(op)
  |     |-- complete result -> return base result unchanged
  |     `-- PARTIAL / None -> Phase2 fallback dispatch
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
  |     `-- composite leaf attention_params
  |           -> LightningIndexer / SparseFlashAttention
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

Phase2 handles only fallback work that base lookup did not complete. Specialized CandidateIndexes filter candidates and interpolate; they do not repackage same-coordinate candidates as `MEASURED`. Phase2 does not derive one operator's shape from another invocation.

### 3.2 Prerequisite PRs, Issue, and merge gate

The Phase2 leaf-operator interface depends on these upstream changes:

| Prerequisite | Capability provided | Phase2 dependency |
| --- | --- | --- |
| [PR #555](https://gitcode.com/Ascend/msmodeling/pull/555) | DSA CP frontend and runtime-layout foundation | It provides the DeepSeek V4 `quant_lightning_indexer`, `sparse_attn_sharedkv`, and `scatter_nd_update_mla` semantic entries and forms workloads with the actual TP/SP and prefill/decode semantics. |
| [PR #593](https://gitcode.com/Ascend/msmodeling/pull/593) | Profiling semantics and exact-match foundation for GLM5 DSA CP | It refines the shape, dtype, phase, cache, and SparseFlashAttention child semantics of GLM5 `mla_sparse_attention`; this parent path is not the same as the DeepSeek V4 `sparse_attn_sharedkv` leaf entry. |
| [Issue #272](https://gitcode.com/Ascend/msmodeling/issues/272) | Unified upstream decomposition and leaf-query contract | It defines a canonical leaf descriptor consumable across DeepSeek V4 and GLM5, or explicit versioned leaf entries. Exact lookup and interpolation must consume the same shape, dtype, phase, topk, and request/sequence semantics; multi-operator latency is aggregated outside the interpolation module. |

The Phase2 target baseline must contain these leaf-descriptor and runtime-field contracts. The implementation consumes that contract directly and keeps no versioned substitute interface in the wrapper.

Phase2 consumes the existing `SubKernelSpec`, `attention_params`, and `cache_params` contracts provided by these prerequisites. Decomposition rules, leaf shapes, and runtime parameters remain defined by the existing decomposers in `ProfilingDataSource`; Phase2 adds neither decomposition algorithms nor changes to `ProfilingDataSource`. When the parent base query returns `PARTIAL` or misses, the wrapper calls the same existing decomposer, reuses base sub-kernel lookup first for each leaf, interpolates only a leaf miss, and aggregates latency once under the existing composite semantics.

If the prerequisite contract or required leaf semantics are incomplete, the path fails closed instead of guessing runtime semantics in the wrapper.

### 3.3 Result sources

- A complete base result keeps its original `QuerySource`; the wrapper does not rewrite it.
- After base returns `PARTIAL` or misses, a successful specialized path returns only `QuerySource.INTERPOLATED`.
- A single same-coordinate candidate, insufficient candidates, regime mismatch, invalid latency, or an out-of-range target returns a miss; there is no wrapper-local exact fallback.
- If base returned `PARTIAL`, the existing Phase1 fallback rule applies when interpolation also fails.

### 3.4 Phase2 operator coverage and dimension ceilings

Interpolation keeps the Phase1 low-dimension-first rule, which is not redefined here. A ceiling below is an implementation limit, not a claim that the current CSV data forms a valid boundary at that dimension.

#### 3.4.1 Operator paths implemented or adjusted in Phase2

This table lists only operator entries implemented, extended, or adjusted in Phase2, with all axis combinations for one operator kept in the same row. Operators that only reuse unchanged Phase1 behavior are omitted. Operators with incomplete implementations follow Section 3.4.2; implemented paths with only data or validation gaps remain here, with their boundaries documented in the corresponding capability sections.

For MOE/DFC, Phase2 adds only the specialized `DispatchFFNCombine` interpolation path.

Coordinate and semantic-field names in the table mean the following. Each subcategory defines whether a field is a continuous interpolation axis or an exact regime field:

- `io_numel`: the sum of all tensor-input and tensor-output element counts for one Elementwise invocation;
- `axis_0`: the first continuous shape dimension used by Phase1 generic compute, with all other shape dimensions fixed;
- `output_numel`: the product of all output-shape dimensions;
- `tokens`: token rows after flattening semantic token dimensions, with the exact source defined by each subcategory;
- `q_tokens`: the total query tokens processed by one attention leaf;
- `effective_kv_len`: KV length weighted by each request's query-token work;
- `topk`: the number of experts or indices selected per token;
- `M / K / N`: matrix rows, reduction width, and output columns. Non-matrix DynamicQuant uses only flattened rows `M` and final input width `K`.

| Operator category | Subcategory/path | Operator entry | Versions | Axis combinations | Ceiling | Basis |
| --- | --- | --- | --- | --- | ---: | --- |
| MOE/DFC | DispatchFFNCombine | `tensor_cast.dispatch_ffn_combine.default` | v0.15, v0.18 | `tokens` | 1D | The target path is reachable. v0.15 lacks EP data and v0.18 uses INT8 weights, so neither establishes data equivalence for the plain unquantized entry; the current path stably misses. |
| MOE/DFC | DispatchFFNCombine | `tensor_cast.dispatch_ffn_combine_quant.default` | v0.15, v0.18 | `tokens` | 1D | The v0.18 seven-input physical dtype signature supports the current W8A8 data. v0.15 lacks EP data and stably misses. Complete GMM1/GMM2 weight shapes, hidden size, topk, EP, activation format, and the seven-input dtype signature are regime fields. |
| MOE/DFC | DispatchFFNCombine | `tensor_cast.dispatch_ffn_combine_quant_int4.default` | v0.15, v0.18 | `tokens` | 1D | The target path is reachable. There is currently no semantically matching INT4 measured row, so the path returns a stable miss. |
| MOE/DFC | DispatchFFNCombine | `tensor_cast.dispatch_ffn_combine_fp8.default` | v0.15, v0.18 | `tokens` | 1D | The target path is reachable. There is currently no semantically matching FP8 measured row, so the path returns a stable miss. |
| MOE/DFC | DispatchFFNCombine | `tensor_cast.dispatch_ffn_combine_mxfp4.default` | v0.15, v0.18 | `tokens` | 1D | The target path is reachable. There is currently no semantically matching MXFP4 measured row, so the path returns a stable miss. |
| Elementwise | Broadcast / Shared-token | `aten.add.Tensor` | v0.13, v0.15, v0.18 | `io_numel` | 1D | Total input/output elements represent I/O work; broadcast signature, rank, dtype, and kernel are regime fields. |
| Elementwise | Broadcast / Shared-token | `aten.mul.Tensor` | v0.13, v0.15, v0.18 | v0.13 reuses generic compute; v0.15/v0.18 use `io_numel` | 1D | The specialized path uses one explainable continuous axis. |
| Elementwise | Broadcast / Shared-token | `aten.div.Tensor` | v0.13, v0.15, v0.18 | v0.13 reuses generic compute; v0.15 uses `io_numel`; v0.18 is not interpolated | 1D; v0.18 zero-cost | v0.18 is explicitly unaccounted by the mapping. |
| Attention | LightningIndexer | `LightningIndexer` leaf decomposed from `tensor_cast.dsa_indexer.default` | v0.18 | `q_tokens`; `effective_kv_len`; their pair | 2D | Phase, topk, heads, layouts, and cache mode are regime fields; missing runtime sequence values fail closed. |
| Attention | SparseFlashAttention | `SparseFlashAttention` leaf decomposed from `mla_sparse_attention*` | v0.18 | `q_tokens`; `effective_kv_len`; their pair | 2D | Uses the same workload semantics as LightningIndexer with additional sparse-block and sparse-index regime fields. |
| Compute | Concat | `aten.cat.default` | v0.13, v0.18; zero-cost in v0.15 | `output_numel` | 1D | Versions with `ConcatD.csv` use generic compute. |
| Compute | Concat | `tensor_cast.cat.default` | v0.13, v0.18; zero-cost in v0.15 | `output_numel` | 1D | Reuses the `aten.cat.default` kernel policy. |
| Compute | Gather | `aten.embedding.default` | v0.13, v0.15, v0.18 | `output_numel` | 1D | Gather policy uses output element count; alternates inherit the axis. |
| Compute | Cast | `aten.to.dtype` | v0.13, v0.15, v0.18 | `axis_0` | 1D | Cast and TensorMove variants use generic compute. |
| Compute | LayerNorm | `aten.native_layer_norm.default` | v0.18 | `axis_0` | 1D | `tc_input_count: 3` excludes non-tensor metadata. |
| Compute | Dynamic quant | `tensor_cast.dynamic_quantize_symmetric.default` | v0.13, v0.15, v0.18 | `M`; `K`; `M + K` | 2D | Output signature, scale mode, dtype, and format are regime fields. |
| Compute | Dynamic quant | `tensor_cast.dynamic_quantize_asymmetric.default` | v0.13, v0.15, v0.18 | `M`; `K`; `M + K` | 2D | The asymmetric offset output is isolated from symmetric candidates. |
| Compute | Dynamic block quant | `tensor_cast.dynamic_quantize_mxfp4.default` | v0.13, v0.15, v0.18 | `M`; `K`; `M + K` | 2D | The path exists, but no `DynamicBlockQuant.csv` is checked in. |
| Compute | Quantized matrix | `tensor_cast.static_quant_linear.default` | v0.13, v0.15, v0.18 | `M`; `K`; `N` and combinations | v0.13/v0.15: 3D; v0.18: 1D | `M/K/N` come from rank-2 activation and output shapes; dtype, layout, and output signature are isolated. |
| Compute | Quantized matrix | `tensor_cast.static_quant_linear_int4.default` | v0.13, v0.15, v0.18 | Same as `static_quant_linear` | v0.13/v0.15: 3D; v0.18: 1D | The path does not infer logical dimensions from the function name or packed physical weight shape. |
| Compute | Quantized matrix | `tensor_cast.fp8_linear.default` | v0.13, v0.15, v0.18 | Same as `static_quant_linear` | v0.13/v0.15: 3D; v0.18: 1D | The path exists, but no same-dtype FP8 data is checked in. |
| Compute | Quantized matrix | `tensor_cast.mxfp4_linear.default` | v0.13, v0.15, v0.18 | Same as `static_quant_linear` | v0.13/v0.15: 3D; v0.18: 1D | The path exists, but no same-dtype MXFP4 data is checked in. |
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
- the complete dtype signature of all seven physical DFC inputs;
- complete GMM1/GMM2 weight shapes;
- hidden size;
- topk;
- EP size.

The target reuses the existing DFC physical-input projection to convert TensorCast arguments into seven physical inputs: activation, two GMM weights, route, two workspaces, and probabilities. It then extracts the complete dtype signature and GMM1/GMM2 weight shapes. A candidate extracts equivalent fields from the seven CSV input shapes, dtypes, and formats. CSV formats are used only to restore and validate physical shapes; the target does not infer a complete format signature that is absent from its runtime semantics. A missing physical input, dtype signature, or weight shape, or any mismatch between target and candidate, rejects the candidate; equal activation dtype alone never permits latency reuse.

Current `DispatchFFNCombine.csv` data uses a seven-input dtype signature of BF16 activation, two INT8 weights, INT32 route, two INT64 workspaces, and FLOAT probabilities. It therefore supports only W8A8 targets with the same physical signature. The v0.15 CSV also lacks an `EP Size` field and cannot form a complete EP regime, so current effective data coverage is limited to v0.18 W8A8. Plain, INT4, FP8, MXFP4, and v0.15 targets return stable misses because their physical dtype signature or EP does not match; they do not borrow W8A8 rows to increase hit rate.

EP size is a required regime field. The CSV must contain an `EP Size` column and every row must provide a valid positive integer. A missing column, blank value, or invalid value cannot match any target EP. If runtime EP is not configured, the path returns `ep_size_not_configured`.

The path misses when topk cannot be extracted, EP differs, candidates cannot form a boundary, or latency is invalid. It does not fall through to ordinary compute interpolation.

### 4.2 Elementwise

#### 4.2.1 Broadcast / Shared-token

The specialized Phase2 broadcast and shared-token path uses `query_mode: elementwise`. In v0.13, `aten.mul.Tensor` and `aten.div.Tensor` still use generic compute; other applicable versions map the kernels they actually contain, including `Add`, `Mul`, `Div`, and alternate variants such as `AddAiCore`, `MulAiCore`, or `RealDiv`.

##### Target and continuous axes

The specialized path uses one continuous axis:

- `io_numel = sum(input_tensor.numel) + sum(output_tensor.numel)`, the total tensor I/O element count for one Elementwise invocation.

The current path attempts:

```text
1D: io_numel
```

This axis covers unary, binary, broadcast, and shared-token calls without introducing an unstable second axis. Output rank, input count, and broadcast pattern remain regime fields, so equal total element counts cannot mix structurally different calls.

##### Regime, dtype behavior, and failure boundaries

Candidates are isolated by:

- kernel type;
- output rank;
- input count;
- broadcast pattern;
- target output dtype.

Candidate dtype must exactly match the target dtype. Phase2 does not scale measured latency by dtype byte width; missing same-dtype data returns a diagnostic miss. One interpolation also cannot mix candidates from the preferred latency column and alternate latency columns.

Alternate kernels are tried in mapping order. If all kernels fail, details retain each attempted kernel and miss result.

### 4.3 Attention

#### 4.3.1 LightningIndexer

`LightningIndexer` is emitted by the existing decomposer for `tensor_cast.dsa_indexer.default` as `SubKernelSpec(query_mode="attention")`, with runtime sequence semantics carried in `attention_params`. Phase2 keeps no second direct-signature parser. A legacy `query_mode: attention_lightning_indexer` entry is not accepted as an interpolation entry and, after a base miss, returns `runtime_attention_leaf_required` because canonical runtime context is unavailable.

##### Target and continuous axes

Let `actual_seq_lengths_values` be cumulative query offsets. Their differences produce per-request query lengths `q_i`; `actual_seq_lengths_kv_values` provides the corresponding KV lengths `kv_i`. The continuous axes are:

- `q_tokens = sum(q_i)`, the total query tokens processed by the leaf;
- `effective_kv_len = sum(q_i * kv_i) / q_tokens`, the effective KV length weighted by query work.

Axis groups are attempted in this order:

```text
1D: q_tokens (effective_kv_len fixed)
1D: effective_kv_len (q_tokens fixed)
2D: q_tokens + effective_kv_len
```

Cache block count is physical capacity, not the KV length accessed by this invocation, so it is not an interpolation axis. `topk` remains an exact regime field rather than a third axis.

##### Regime and failure boundaries

The regime contains kernel type, query dtype, prefill/decode/mixed phase, query rank, head dimension, sparse mode, KV-head count, input/cache layout, `topk`, block size, head count, and KV-cache mode. Batch size is diagnostic only, not an axis or regime field; batches with equivalent workload semantics may share candidates.

Non-monotonic cumulative offsets, an offset total different from `q_tokens`, invalid KV lengths, incomplete runtime metadata, or any missing required regime field fail closed. The wrapper performs no same-coordinate `MEASURED` fallback and attempts at most 2D interpolation after a base miss.

Coordinate-level leave-one-coordinate-out holdout on the real v0.18 CSV recovers 87.53% with 1D and 98.50% with conditional 1D+2D. The combined median relative error is 2.07% and P90 is 8.98%. This validates the axis choice and the incremental value of 2D; it is not an end-to-end model-error claim.

By phase, the conditional 1D+2D recovery rates are 97.79% for decode, 99.03% for mixed, and 97.93% for prefill.

#### 4.3.2 SparseFlashAttention

`SparseFlashAttention` is emitted as an attention leaf by the existing decomposers for `tensor_cast.mla_sparse_attention.default` and its quantized variant. Phase2 reuses the same runtime-workload construction as LightningIndexer and adds no parent-decomposition algorithm. Only v0.18 currently provides usable `SparseFlashAttention.csv`; other versions return a stable miss when data is absent.

##### Target and continuous axes

The continuous axes are the same as LightningIndexer:

Axis groups are attempted in this order:

```text
1D: q_tokens (effective_kv_len fixed)
1D: effective_kv_len (q_tokens fixed)
2D: q_tokens + effective_kv_len
```

##### Regime and failure boundaries

The regime inherits LightningIndexer fields and adds sparse block size, sparse-index pattern, and valid-index count. SparseFlashAttention candidates never mix with LightningIndexer or Phase1 FusedInferAttentionScore candidates. Legacy CSV rows without complete runtime metadata are rejected.

Coordinate-level holdout on the real v0.18 CSV recovers 84.43% with 1D and 91.33% with conditional 1D+2D. The combined median relative error is 1.15% and P90 is 19.39%. Points recovered only by 2D have 0.49% median error and 1.82% P90, showing a clear second-axis gain; the combined P90 tail remains a known data risk.

By phase, the conditional 1D+2D recovery rates are 91.60% for decode, 94.48% for mixed, and 79.95% for prefill. The lower prefill recovery remains a known coverage risk.

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

`compute_scale` derives `M = math.prod(shape[:-1])` and `K = shape[-1]` from a quantization input `[d0, ..., dn, K]`. `quantized_matmul` does not infer logical dimensions from a packed physical weight shape. It requires rank-2 activation `[M, K]` and rank-2 output `[M, N]`, directly producing `M`, `K`, and `N`.

| `compute_subcategory` | Kernel | Axes | Maximum |
| --- | --- | --- | ---: |
| `compute_scale` | `DynamicQuant`, `DynamicBlockQuant` | `M`, `K`, `M + K` | 2D |
| `quantized_matmul` | `QuantBatchMatmulV3` | `M`, `K`, `N`, and their combinations | Up to 3D in v0.13/v0.15; 1D in v0.18 policy |

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

Base lookup always runs first. Only after base returns `PARTIAL` or misses does `compute_scale` build a regime with complete output signature and scale mode and attempt interpolation. The wrapper does not return a local `MEASURED` result.

FP16 DynamicQuant matches the CSV `DT_FLOAT16` dtype. This local rule does not change Phase1 dtype compatibility for other compute paths.

##### Quantized matrix compute

`quantized_matmul` reuses Phase1 M/K/N interpolation math, but its target coordinates come from rank-2 activation and output shapes. `tc_input_count: 2` still controls the candidate CSV input signature. Input dtype/format, output dtype/format, and output count are regime fields. Packed physical weight shape is neither a continuous coordinate nor inferred from the function name. Phase1 `scale_matrix` denotes the M/K 2D overhead of applying a scale during static quantization; it has different semantics and is not reused as this selector.

A mapping may keep an existing base-only `query_mode`, such as `mtp_projection`, together with `compute_subcategory: quantized_matmul`. The base query owns the exact lookup; `compute_subcategory` selects the wrapper interpolation fallback only after that lookup misses.

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

This path accepts either the independent cache-update entry or a `scatter_cache_write` leaf emitted by an existing decomposer. Only a base sub-kernel miss enters 1D interpolation. Full cache shape remains a regime field so different cache capacities cannot collapse to one coordinate.

## 5. Mapping and minimal policy

Phase2 uses existing `op_mapping.yaml` fields and does not add a separate configuration file.

| Field | Purpose |
| --- | --- |
| `kernel_type` | Primary profiling CSV and kernel |
| `alternate_kernel_types` | Ordered variants tried after the primary kernel misses |
| `query_mode` | Select DFC, Elementwise, Phase1 Attention, or independent cache-update paths; runtime attention leaves use `SubKernelSpec.query_mode: attention` |
| `compute_subcategory` | Select `compute_scale` or `quantized_matmul` |
| `tc_input_count` | Match only the first N TensorCast inputs against the CSV |
| `expected_input_formats` | Required by `quantized_matmul`; declares the complete ordered input-format regime used to isolate packed weights |
| `interpolation_policy.kernel_overrides.<kernel>.max_interpolation_dim` | Narrow a kernel's maximum interpolation dimension |

| Operator category | Subcategory/path | Mapping selector |
| --- | --- | --- |
| MOE/DFC | DispatchFFNCombine | `query_mode: moe_fused` |
| Elementwise | Broadcast / Shared-token | `query_mode: elementwise` |
| Attention | LightningIndexer | `composite: true`, `decomposer: true`, and a leaf with `SubKernelSpec.query_mode: attention` plus `attention_params` |
| Attention | SparseFlashAttention | `composite: true`, `decomposer: true`, and a leaf with `SubKernelSpec.query_mode: attention` plus `attention_params` |
| Compute | Generic compute mapping refinements | `kernel_type`, `alternate_kernel_types`, `tc_input_count` |
| Compute | Quant-scale / Dynamic quant | `compute_subcategory: compute_scale` |
| Compute | Quant-scale / Quantized matrix compute | `compute_subcategory: quantized_matmul`, `tc_input_count: 2`, `expected_input_formats` |
| Compute | MLA cache update | `query_mode: scatter_nd_update_mla` |

`attention_special` and ordinary compute mappings without a Phase2 selector continue to use Phase1 paths.

An unknown `compute_subcategory` returns `compute_subcategory_unknown`. Every top-level operator first uses base lookup; composite leaves also use existing base sub-kernel queries before interpolation. Only misses enter CandidateIndex interpolation, and CandidateIndex does not own exact matching. Phase2 adds no second global switch.

## 6. Result and diagnostic additions

Phase2 reuses Phase1 QueryResult semantics, latency guards, and candidate geometry checks. It adds only capability-specific location data.

### 6.1 Successful results

A successful leaf-operator interpolation records at least:

- `kernel_type`, `query_mode`, or `compute_subcategory`;
- `method`, `interpolation_dim`, and `axes`;
- boundary, candidate count, and matched rows;
- `fallback_from` and `interpolation_path`;
- capability fields such as EP, scale mode, or attention subcategory.

Base exact hits keep their base details. Successful interpolation records the actual axes, dimension, candidates, and boundary and does not emit `method=exact_coordinate`.

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

Candidate latency must be positive and finite. Paths with source-pure grouping, including elementwise, generic compute, and quant-scale, do not mix the preferred latency column with alternate latency columns in one interpolation.

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
| MOE/DFC | DispatchFFNCombine | v0.18 W8A8 1D token interpolation, seven-input physical-dtype and complete GMM1/GMM2 weight-shape isolation, real-CSV duplicate-coordinate protection, out-of-range and insufficient-candidate fallback, topk / EP / local-expert isolation, missing/blank/invalid EP rejection, strict physical-signature isolation across plain / W8A8 / INT4 / FP8 / MXFP4, and stable misses when matching data is unavailable |
| Elementwise | Broadcast / Shared-token | `io_numel` 1D, broadcast signature, alternate kernels, cross-dtype rejection, latency-source purity, and no local measured exact after a base miss |
| Attention | LightningIndexer | Conditional 1D+2D over `q_tokens / effective_kv_len`, phase/topk/head/layout/cache-mode isolation, fail-closed runtime sequence fields, and real-CSV golden and holdout tests |
| Attention | SparseFlashAttention | The same workload axes as LightningIndexer, additional sparse-block/index regime fields, rejection of legacy metadata, and real-CSV golden and holdout tests |
| Compute | Generic compute mapping refinements | ConcatD data presence, Gather and Cast alternate order, LayerNorm metadata exclusion, and `tc_input_count` |
| Compute | Quant-scale / Dynamic quant | DynamicQuant 2D, output signature, scale mode, FP16, max dimension, DynamicBlockQuant mode, and a real-CSV golden test |
| Compute | Quant-scale / Quantized matrix compute | M/K/N from rank-2 activation/output, versioned max dimension, `tc_input_count`, no INT8 reuse for FP8 or MXFP4, and a real-CSV golden test |
| Compute | MLA cache update | Argument reordering, complete-cache-shape isolation, real-CSV duplicate-coordinate protection, 1D interpolation, alternate kernel, and composite-leaf coverage |

Every subcategory or path also covers applicable target-construction failures, insufficient candidates, regime mismatch, invalid latency, out-of-range targets, success details, and miss details.

### 7.3 Acceptance criteria

Before merge, Phase2 must satisfy:

1. Every implemented leaf mapping that declares a Phase2 selector reaches its intended target builder; data gaps produce stable misses, while unverified upstream leaf sequences follow Section 3.4.2.
2. Complete base hits preserve their source; after a base miss, the wrapper returns only `INTERPOLATED` or a miss and never a local `MEASURED` result.
3. `max_interpolation_dim` blocks dimensions above the ceiling without changing low-dimension-first order.
4. Specialized leaves use existing base queries, and fallback consumes only base misses without revalidating base exact results.
5. Candidates do not cross incompatible EP, topk, complete GMM weight shapes, shape tails, complete cache shapes, request/sequence grouping, API paths, dtypes, DFC physical-input signatures, or output signatures.
6. A DFC candidate is used only when its seven-input physical dtype signature, activation format, complete GMM1/GMM2 weight shapes, and EP match the target. Current effective data coverage is limited to v0.18 W8A8; v0.15 and plain, INT4, FP8, and MXFP4 paths return stable misses.
7. LightningIndexer and SparseFlashAttention cumulative query offsets, KV lengths, and regime fields are equivalently extractable from target and CSV; incomplete semantics return stable misses.
8. Composite handling uses `SubKernelSpec` objects emitted by existing decomposers and adds no decomposition algorithm. Each leaf uses base query first, interpolation only after a miss, and latency is aggregated once.
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
| MOE/DFC | Cross-configuration reuse when DFC physical-input dtypes, FFN intermediate widths, or EP semantics differ | Add the seven-input physical dtype signature, complete GMM1/GMM2 weight shapes, activation format, and EP to the regime. Only v0.18 W8A8 currently has complete data; v0.15 and other physical signatures stably miss |
| Elementwise / Broadcast / Shared-token | Multiple shape axes lack one stable cross-pattern meaning | Use only total I/O work `io_numel` in 1D; isolate output rank, input count, broadcast pattern, and dtype in the regime |
| Attention / LightningIndexer | Cache capacity substitutes for actual KV work, or prefill and decode candidates mix | Derive `q_tokens/effective_kv_len` from runtime cumulative query offsets and KV lengths; isolate phase, topk, heads, layouts, and cache mode; fail closed on missing fields |
| Attention / SparseFlashAttention | Different sparse configurations aggregate at one workload coordinate | Reuse `q_tokens/effective_kv_len`, additionally isolate sparse block, index pattern, and valid-index count, and reject legacy metadata rows |
| Upstream leaf input | Duplicate accounting, missing semantics, or exact/interpolation workload drift | Reuse one leaf descriptor from the prerequisites; base and wrapper consume the same shapes, dtypes, and runtime parameters |
| Common | Sparse profiling data | No extrapolation or cross-dtype borrowing; return a diagnostic miss |

Phase2 is complete when its operator paths, boundaries, and fallback behavior are explicit and testable. Paths without real data remain recognizable stable misses until profiling data is added and accuracy is evaluated.
