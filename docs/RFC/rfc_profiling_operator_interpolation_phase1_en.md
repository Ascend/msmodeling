# RFC: Profiling Operator Interpolation Module Design (Phase 1)

## Metadata

| Field | Content |
|---|---|
| RFC name | Profiling Operator Interpolation Module Design (Phase 1: compute and attention_special multidimensional interpolation) |
| Status | Draft |
| Phase | Phase 1 / 3 |
| Target module | `tensor_cast/performance_model/profiling_database` |
| Target branch | `develop` |
| Author(s) | @zhenyu_zhang |
| Created | 2026-05-26 |
| Updated | 2026-06-13 |
| Related Issue/PR | #229 |

## Document Alignment Note

The Chinese RFC is the detailed master. This English version covers the same design boundaries, default behavior, rollback strategy, test requirements and compatibility conclusions, but some long Chinese design sections are split into smaller English sections for readability. If the two versions ever diverge, the Chinese design constraints are authoritative and the English version must be updated without introducing different defaults or acceptance criteria.

## 1. Overview

This RFC defines the first phase of the profiling operator interpolation module. Phase 1 is not limited to a single MatMul/GEMM M/K 2D case. It builds a runnable, reversible, observable interpolation framework and delivers two core capabilities:

1. Multidimensional interpolation for `compute` operators, focusing on MatMul/GEMM M/K/N 1D, 2D and 3D interpolation.
2. 1D, 2D and 3D interpolation paths for `attention_special`, based on the current query semantics `(batch, seq, heads, head_dim)`. `seq/avg_seq_len` is the baseline axis. `batch`, `heads` and `head_dim` require runtime field derivation, CSV enrichment or synthetic test CSVs. Runtime must still verify field derivability, candidate availability, boundary/grid completeness and latency validity.

Phase 1 intentionally keeps the scope narrow: communication operators are not wrapped by another interpolation layer, extrapolation is not enabled, and `fallback_order` is not exposed as user configuration. The phase introduces `scipy.interpolate.griddata` as a general 2D/3D interpolation backend, following the capability pattern used by aiconfigurator.

The main query flow is:

```text
profiling datasource entry
  -> exact lookup first
  -> exact miss / partial enters interpolation fallback
  -> compute or attention_special allowlist
  -> ComputeIndex / AttentionIndex candidate lookup
  -> 1D linear / scipy.griddata 2D-3D / regular grid fallback
  -> QuerySource.INTERPOLATED
  -> observable details / shape_match_info / debug log
```

## 2. Three-Phase Plan

### Phase 1: compute and attention_special multidimensional interpolation

Phase 1 delivers:

- Wire `InterpolatingDataSource` into the profiling path.
- Add `--disable-profiling-interpolation` as a base `ProfilingDataSource` rollback switch.
- Add the common math module `interpolation_math.py`.
- Use `scipy.interpolate.griddata` as the general 2D/3D interpolation backend.
- Add a candidate index layer for compute and attention_special candidates.
- Support compute 1D/2D/3D interpolation, focusing on MatMul/GEMM M/K/N.
- Support attention_special 1D/2D/3D interpolation paths. The 2D/3D paths must be implemented and tested in Phase 1. If production CSVs cannot reliably express batch or the third axis, runtime downgrades to `seq/avg_seq_len` 1D or another provably safe lower-dimensional combination.
- Emit `source`, `confidence`, `details`, `shape_match_info` and debug logs.
- Reserve the extrapolation guard interface but keep extrapolation disabled. Targets outside boundary/convex hull only record the failure reason and fall back. Phase 1 never returns `EXTRAPOLATED`.

### Phase 2: MoE/DFC, elementwise and composite expansion

Phase 2 extends the Phase 1 framework:

- `query_mode: moe_fused` and `DispatchFFNCombine` interpolation.
- Elementwise 1D / conditional 2D / conditional 3D interpolation.
- Composite decomposition reuses each sub-kernel's interpolation capability.
- FP8 quant scale compute subcases, such as `compute_scale` and `scale_matrix`, when the M/K schema is stable.

Communication remains outside wrapper interpolation. It continues to use the existing base exact / alpha-beta(message_bytes) path.

### Phase 3: extrapolation, accuracy evaluation and engineering convergence

Phase 3 delivers:

- Opt-in extrapolation, disabled by default.
- Ratio guard / max bound protection.
- Holdout accuracy evaluation and error reports.
- Performance budgets, cache policy, stale rebuild and concurrent access convergence.
- Benchmarks for exact hit, 1D/2D/3D interpolation, schema miss, policy miss, index build, stale rebuild and memory overhead.
- Metrics, warnings and debug log convergence.
- Policy field cleanup and documentation convergence.

### Overall Architecture Across Three Phases

```text
User / CLI / Config
  |
  |-- --performance-model profiling
  |-- --profiling-database <dir>
  |-- --disable-profiling-interpolation                  [Phase 1]
  v
ModelRunner / PerformanceModel Factory
  |
  |-- disable_profiling_interpolation = true
  |      |
  |      v
  |   ProfilingDataSource                                [base CSV/existing path]
  |      |
  |      |-- CSV exact match ----------------------------> QueryResult(source=MEASURED)
  |      |-- base internal derived path -----------------> QueryResult(source=INTERPOLATED)
  |      |     for example communication alpha-beta/message_bytes
  |      |-- partial composite hit ----------------------> QueryResult(source=PARTIAL)
  |      |-- miss ---------------------------------------> None
  |
  |-- disable_profiling_interpolation = false
         |
         v
      InterpolatingDataSource                            [Phase 1]
         |
         | 1. call base.lookup(op) first
         v
      ProfilingDataSource                                [existing]
         |
         |-- MEASURED -----------------------------------> return base result
         |-- INTERPOLATED -------------------------------> return base result
         |     base communication interpolation remains authoritative
         |-- EXTRAPOLATED -------------------------------> return base result if ever produced by base
         |     Phase 1 wrapper itself must not produce EXTRAPOLATED
         |-- PARTIAL ------------------------------------> try wrapper interpolation fallback
         |-- None ---------------------------------------> try wrapper interpolation fallback
                |
                v
      Interpolation Fallback Dispatcher                  [Phase 1]
         |
         |-- category/query_mode allowlist
         |-- policy merge:
         |     built-in defaults
         |       + global interpolation_policy
         |       + operator_mappings.<op>.interpolation_policy
         |
         +---------------------------------------------------------------+
         |                                                               |
         v                                                               v
  communication                                                   compute
  category == communication                                       category == compute
  [base only; no wrapper 1D/2D/3D interpolation]                  [Phase 1]
         |                                                               |
         |-- base exact / _query_comm_csv                                |-- CandidateIndex / ComputeIndex
         |-- base alpha-beta(message_bytes) may return INTERPOLATED      |-- axes: M / K / N
         |-- wrapper returns base result or None/PARTIAL unchanged       |-- regime keys:
         |                                                               |     kernel_type, dtype, format,
         |                                                               |     layout, transpose, exact fields
         |                                                               |-- algorithms:
         |                                                               |     1D linear
         |                                                               |     2D/3D scipy.griddata(linear)
         |                                                               |
         |                                                               |-- FP8 quant scale subcases:
         |                                                               |     compute_scale / scale_matrix
         |                                                               |     [Phase 2 if schema/policy is not
         |                                                               |      stable enough in Phase 1]
         |                                                               |
         |                                                               v
         |                                                        QueryResult(source=INTERPOLATED)
         |
         +---------------------------------------------------------------+
         |
         v
  attention_special
  query_mode == attention_special                                [Phase 1]
         |
         |-- CandidateIndex / AttentionIndex
         |-- axes from current query semantics and CSV enrichment:
         |     seq / avg_seq_len mandatory
         |     batch / heads / head_dim only when derivable and varying
         |-- regime keys:
         |     kernel_type, dtype, sparse_mode, kv_heads,
         |     quant_mode, layout, mask/cache regime
         |-- supported paths:
         |     direct attention_special ops
         |     attention sub-kernel fallback from decomposer
         |-- algorithms:
         |     1D linear
         |     2D/3D scipy.griddata(linear)
         |-- optional axis_transform:
         |     sqrt(seq) for known O(seq^2) kernels
         |
         v
  QueryResult(source=INTERPOLATED)


Phase 2 Extensions
  |
  |-- moe_fused / DispatchFFNCombine
  |-- elementwise 1D / conditional 2D / conditional 3D
  |-- composite sub-kernel interpolation reuse
  |-- FP8 quant scale compute subcases


Phase 3 Extensions
  |
  |-- ExtrapolationPolicy
  |     |-- allow_extrapolation default false
  |     |-- opt-in only
  |     |-- ratio guard / max bound
  |     |-- QuerySource.EXTRAPOLATED
  |     |-- lower confidence than interpolation
  |
  |-- Accuracy / Holdout Evaluation
  |-- Performance / Cache Governance
  |-- Observability Convergence
  |-- Policy / Documentation Cleanup
```

## 3. Phase 1 Scope

In scope:

1. Default profiling path wraps `ProfilingDataSource` with `InterpolatingDataSource`.
2. Exact hit remains the highest priority and returns `MEASURED`.
3. Interpolation is only a fallback for `None` or `PARTIAL`.
4. Compute operators support 1D/2D/3D interpolation, with MatMul/GEMM M/K/N as the main target.
5. Attention_special supports 1D/2D/3D code paths and tests.
6. Interpolation success returns `QuerySource.INTERPOLATED`.
7. Output includes `details`, `shape_match_info`, `confidence`, source and debug logs.
8. `confidence` is display-only in Phase 1. It does not drive result selection or fallback.

Out of scope:

1. No wrapper interpolation for communication operators.
2. No extrapolated result. `QuerySource.EXTRAPOLATED` is not returned in Phase 1.
3. No full MoE/DFC/elementwise/composite multidimensional implementation.
4. No user-configurable `fallback_order`; the order is fixed as `exact -> 1D -> 2D -> 3D`.
5. No requirement to modify every `op_mapping.yaml` entry at once.

## 4. Current State and Problems

The current profiling datasource is reliable for exact matches and a few existing derived paths, but it has limited behavior after exact miss:

- Measured CSV rows are used only when the runtime query matches the expected schema and regime.
- Compute operators such as MatMul/GEMM often have nearby measured points across M/K/N axes, but the current path cannot reuse them for multidimensional interpolation.
- Attention_special sub-kernels can often derive `seq/avg_seq_len`, while batch/head/head_dim information is only partially available from runtime fields or enriched CSVs.
- Existing fallback behavior is conservative but opaque: users cannot easily see whether a result is measured, interpolated by an existing base path, partially decomposed or missing.
- Communication already has alpha-beta/message_bytes behavior in the base datasource, so wrapping communication with another shape interpolation layer would mix two different models and increase risk.

Phase 1 addresses only the first safe slice: compute and attention_special interpolation behind exact-first lookup, explicit allowlists, no extrapolation and a rollback switch.

## 5. Main Path Design

`--performance-model profiling` should use `InterpolatingDataSource` by default. The wrapper calls base lookup first. Exact measured results are returned unchanged.

Add the rollback CLI:

```text
--disable-profiling-interpolation
```

and the matching user config field:

```python
disable_profiling_interpolation: bool = False
```

When the switch is enabled, profiling uses the base `ProfilingDataSource` path without the wrapper's compute/attention interpolation fallback. This is not a guarantee that every base result is CSV-exact: the base path may still contain existing specialized derived paths, such as communication alpha-beta/message_bytes handling.

Phase 1 wrapper behavior is restricted to compute and attention_special allowlist scenarios. Unsupported operators keep the base `ProfilingDataSource` behavior.

## 6. Query Flow

```text
lookup(op)
  -> base.lookup(op)
      -> MEASURED: return base result
      -> INTERPOLATED: return base result
         # possible from existing base communication paths
      -> EXTRAPOLATED: return base result if base ever produces it
         # Phase 1 wrapper itself must not produce EXTRAPOLATED
      -> PARTIAL: try wrapper interpolation fallback
      -> None: try wrapper interpolation fallback
  -> if op is communication:
         return base result
  -> if op is not compute or attention_special allowlist:
         return base result
  -> build interpolation target
  -> query candidate index
  -> try 1D / 2D / 3D by fixed default order
  -> if interpolation succeeds:
         return QueryResult(source=INTERPOLATED)
     else:
         return base result
```

The fixed attempt order is:

```text
exact -> 1D -> 2D -> 3D
```

This means the runtime first searches for candidates where only one continuous axis needs interpolation. If low-dimensional candidates cannot form a boundary because the fixed axes are too strict, the search expands to 2D and then 3D. This order is chosen to minimize error and behavior change. The index layer should avoid full table rescans per dimension and should return boundary/grid availability for the current regime.

`PARTIAL` means the base `ProfilingDataSource` produced a partial result. Phase 1 does not merge wrapper interpolation into missing sub-kernels. Instead, `PARTIAL` is treated as an entry point for a whole-wrapper retry: if the wrapper can independently build a complete interpolation result, it returns `QuerySource.INTERPOLATED` with `details["fallback_from"] = "partial"`; otherwise the original base `PARTIAL` is returned unchanged.

## 7. Candidate Index Layer

`InterpolatingDataSource` remains the datasource wrapper. The candidate index is an internal helper. It is not a new performance model and does not inherit from `DataSourcePerformanceModel`.

Recommended structure:

```text
DataSourcePerformanceModel
  ^
  |
InterpolatingDataSource
  |-- ComputeIndex
  |-- AttentionIndex
```

Candidate point shape:

```python
CandidatePoint(
    regime_key=...,
    axes={"M": 1024, "K": 4096},
    latency=...,
    row_meta={...},
)
```

Cache policy:

- Build lazily. Do not scan all CSVs in `InterpolatingDataSource.__init__`.
- Reuse DataFrames returned by `ProfilingDataSource._load_csv()`.
- Store the index cache inside the wrapper instance; no global cache.
- Do not use `id(df)` as the cache key. Use a stable content fingerprint such as `(kernel_type, df_content_hash, policy_hash, index_kind, tc_input_count)`. `df_content_hash` should cover DataFrame row count, column names, index and values, not only first/last row, so that middle-row changes rebuild the index. CSV path, mtime or profiling database version may also be included when available.
- If base `_csv_cache` is explicitly cleared, the wrapper index cache must also be cleared.

`policy_hash` is not a user-facing field. It is a deterministic hash generated by the index builder from the normalized effective interpolation policy after merging built-in defaults, global `interpolation_policy`, and operator-level `interpolation_policy`. Recommended serialization is canonical JSON with sorted keys and stable representations for missing values before applying SHA-256. It must change whenever effective axes, exact fields, max interpolation dimension, axis transform, extrapolation flags, or duplicate aggregation behavior changes.

`regime_key` must use a stable field order. Missing fields must be encoded explicitly as `None` or `"missing"` to avoid splitting equivalent candidates unpredictably.

## 8. Math Module

Add:

```text
tensor_cast/performance_model/profiling_database/interpolation_math.py
```

Phase 1 functions:

| Function | Responsibility |
|---|---|
| `find_boundary(values, target)` | Find the lower/upper boundary values containing target. Out-of-range targets fail in Phase 1. |
| `linear_interp(x, x0, y0, x1, y1)` | 1D linear interpolation. |
| `griddata_linear_interp(points, values, target)` | General 2D/3D linear interpolation using `scipy.interpolate.griddata(..., method="linear")`. |
| `validate_latency(value)` | Reject NaN, infinite and negative latency. |
| `estimate_confidence(...)` | Produce a display-only confidence value based on dimension, boundary width, candidate density, transform and downgrade information. |
| `build_interpolation_details(...)` | Build normalized details with method, dimension, axes, target, boundary, candidates, confidence and failure reason. |
| `check_extrapolation_guard(target, bounds, policy)` | Reserved extrapolation guard. Phase 1 only returns guard status/reason and never produces extrapolated results. |

`griddata` constraints:

- Only `method="linear"` is enabled in Phase 1.
- Do not use `griddata` for extrapolation.
- If target is outside the convex hull, `griddata` returns NaN, or scipy raises Qhull/ValueError, the attempt fails and falls back.
- 2D griddata needs at least three non-collinear points. 3D griddata needs at least four non-coplanar points.
- Phase 1 keeps only `griddata_linear_interp(points, values, target)` as the public 2D/3D math interface. Dedicated regular-grid bilinear/trilinear functions are deferred as an open follow-up if they later provide clear accuracy, performance or explainability value.

## 9. Compute Multidimensional Interpolation

MatMul/GEMM axes:

```text
activation: [..., M, K]
weight:     [K, N] or [N, K]
output:     [..., M, N]
```

Continuous axes:

- `M`: token/row dimension.
- `K`: reduction dimension.
- `N`: output column dimension.

Exact/regime fields:

- kernel type.
- dtype.
- format.
- layout.
- canonicalized transpose / weight layout result, without becoming stricter than exact lookup.
- quant regime.
- other fields that change kernel semantics.

`input_formats` is CSV physical-format metadata and should remain visible in `row_meta/details`. Phase 1 should not reject all canonicalizable FRACTAL_NZ candidates only because the runtime tensor is logical ND. Candidate groups may still be separated by CSV `input_formats`, and if both ND and non-ND groups are available for the same target, the ND group should be preferred. Cross-format reuse is allowed only when shape canonicalization proves the same M/K/N semantics.

M/K/N extraction:

| Scenario | M | K | N | Requirement |
|---|---|---|---|---|
| ND `[M,K] x [K,N]` | input0[0] | input0[1] | input1[1] | `source_layout=KN` only goes to row_meta/details unless exact lookup also distinguishes it. |
| ND `[M,K] x [N,K]` | input0[0] | input0[1] | input1[0] | F.linear / transposed weight form. `source_layout=NK` must not split buckets more strictly than exact lookup. |
| `BatchMatMulV2` / `TransposeBatchMatMul` 3D `(H,M,K) x (H,K,N)` | input0[1] | input0[2] | input1[2] | `H` is a batch/head regime or exact field. If input1 is `(H,N,K)`, parse `N=input1[1]`. |
| `_FLATTEN_BATCH_KERNELS` batched / flatten batch | `prod(batch_dims) * M` | input0[-1] | Parse only by exact-supported rules | Applies only to kernels where exact lookup already supports flatten batch. MatMul kernels must not use this rule today. |
| FRACTAL_NZ weight | parsed from input0 | parsed from input0 | restore with `fractal_nz_to_nd()` first | Never extract axes from tiled shape directly. |
| Unknown weight layout | none | none | none | Do not enter M/K/N multidimensional interpolation. Fall back or keep safe 1D behavior. |

CSV candidates and runtime targets must use the same extraction rule. If `K` can be inferred from both input0 and input1 and the values differ, the candidate is invalid.

`Accelerator Core` appears in some CSVs, but current exact lookup does not use it as a match condition. Phase 1 interpolation must not put it in `regime_key`; otherwise interpolation would be stricter than exact lookup. If AIC/AIV/MIX_AIC distinction becomes necessary later, exact lookup should be extended first and the interpolation index should follow.

RoPE, SwiGlu and block padding special cases must stay aligned with `_inputs_match`. If Phase 1 does not implement dedicated multidimensional axis extractors for these kernels, they must be excluded from the multidimensional allowlist.

Compute query order:

```text
exact -> 1D -> 2D -> 3D
```

1D examples:

- `M` varies, `K/N` fixed.
- `K` varies, `M/N` fixed.
- `N` varies, `M/K` fixed.

2D examples:

- `M/K` vary, `N` fixed.
- `M/N` vary, `K` fixed.
- `K/N` vary, `M` fixed.

3D:

- `M/K/N` all vary.
- Use `griddata_linear_interp()` for the valid 3D point set. Dedicated regular-grid trilinear interpolation is deferred from Phase 1.
- Downgrade or fall back on insufficient or degenerate candidates.

## 10. Attention_special Multidimensional Interpolation

Current `attention_special` query semantics are `(batch, seq, heads, head_dim)`. Phase 1 should implement and test 1D/2D/3D paths instead of staying at 1D. Runtime still downgrades if production CSVs lack fields or candidate points.

Continuous axes:

- `seq` or `avg_seq_len`.
- `batch`, only when runtime and CSV can derive it reliably.
- `heads`, only when multiple values exist and can form a boundary around the target.
- `head_dim`, only when multiple values exist and can form a boundary around the target.

Before using `Runtime actual_seq_lengths_shape` or `Runtime actual_seq_lengths_values` as a batch source, implementation must inspect the current CSV distribution and verify that the field represents batch size rather than a scalar value, ndim or one seq_len value. If the field is always batch=1 or cannot be reliably interpreted, production `seq + batch` 2D should not trigger, though synthetic/enrichment CSVs must still test the code path.

Exact/regime fields:

- kernel type.
- dtype.
- sparse mode / mask mode.
- KV-head regime.
- quant mode.
- input layout.
- KV cache mode.
- attention state.
- other fields that change attention kernel semantics.

`q_tokens` is not part of the attention `regime_key`, otherwise the same attention scenario can be split into overly narrow candidate groups. It remains in target/candidate axes as a non-interpolation axis for exact filtering. Phase 1 does not interpolate along `q_tokens` and must not aggregate candidates from different `q_tokens` values.

Covered entry points:

1. Direct `query_mode: attention_special` mappings such as `tensor_cast.attention.default` and `tensor_cast.attention_quant.default`.
2. Attention sub-kernel fallback produced by composite decomposers, such as MLA decomposition generating `FusedInferAttentionScore` sub-queries. Full composite interpolation is Phase 2 scope.

Recommended axis priority:

```text
seq/avg_seq_len -> batch(if derivable and varying) -> heads/head_dim(if varying)
```

Phase 1 must implement:

- 1D: `seq/avg_seq_len`.
- 2D: `seq + batch`, `seq + heads`, `seq + head_dim`, selected by available fields.
- 3D: `seq + batch + heads` or `seq + batch + head_dim`, selected by available fields.

These paths are required capabilities, not only reserved interfaces. Runtime must verify field existence, derivability, multi-value candidates, boundary/plane completeness and valid latency. If any condition fails, downgrade or return the base result.

Axis transform:

- Default to raw seq linear interpolation.
- For known O(seq^2) attention kernels, `sqrt(seq)` may be used through built-in or kernel override policy.
- In 2D/3D, the same transform must be applied to both CSV candidate coordinates and runtime target coordinates.
- Record `axis_transform` in details and lower confidence. Confidence remains display-only.

Decode attention note:

- Decode attention commonly has `query_len=1`; the interpolation axis is not the one-token query length but the average context length or past-KV length represented by `avg_seq_len` or equivalent runtime metadata.
- If `avg_seq_len` is estimated from token totals, the implementation must align with the current profiling matcher's block-padding tolerance. The existing FIA matching logic accepts a small block-level gap, for example the `_fia_avg_seq_len_gap` style tolerance around 16 tokens. Phase 1 must not make interpolation stricter than exact lookup by rejecting candidates that exact lookup would accept after block padding.
- The original and transformed coordinates should both be recorded in details when `sqrt(seq)` is used, so block-padding effects remain debuggable.

## 11. Minimal Policy

Phase 1 keeps policy minimal:

| Field | Phase 1 usage |
|---|---|
| `axes` | Allowed continuous axes. Compute defaults to `["M", "K", "N"]`; attention_special defaults to `["seq", "batch", "heads", "head_dim"]`, with runtime choosing the actual dimension by derivability and candidates. |
| `exact_fields` / `regime_keys` | Discrete fields that must match exactly. |
| `max_interpolation_dim` | Maximum interpolation dimension. Compute/MatMul can use 3. Attention_special defaults to 3 but only uses 2D/3D when CSV enrichment and candidates are sufficient. |
| `allow_extrapolation` | Reserved extrapolation field. Phase 1 treats it as an internal guard parameter, not an effective user option. Default false; no extrapolated result is returned. If YAML already contains it, treat as reserved/no-effect. |
| `axis_transform` | Used only when needed, for example attention `sqrt(seq)`. |

Unsupported in Phase 1:

- User-configurable `fallback_order`.
- Communication multidimensional policy.
- Extrapolated result return.

Policy merge order:

```text
built-in default policy
  -> global interpolation_policy
  -> operator_mappings.<op>.interpolation_policy
```

Most operators do not need explicit policy in Phase 1. Conservative defaults should be synthesized from category/query_mode.

## 12. Observability

Interpolation success returns:

```python
source = QuerySource.INTERPOLATED
```

`details` should include:

```python
{
    "method": "griddata_linear",
    "interpolation_dim": 2,
    "axes": ["M", "K"],
    "target": {"M": M, "K": K, "N": N},
    "boundary": {
        "M": [M_lo, M_hi],
        "K": [K_lo, K_hi],
    },
    "corner_points": [...],
    "confidence": 0.7,
    "axis_transform": None,
    "fallback_from": "exact_miss",
}
```

`shape_match_info` should record the interpolation rule, source, target shape, matched candidate shapes and fallback reason.

Final user-visible output must show source and confidence for profiling results, for example:

```text
op_name                                      source        confidence  method              axes       latency_us
torch.ops.npu.npu_transpose_batchmatmul     MEASURED      1.00        exact               -          120.3
torch.ops.npu.npu_matmul                    INTERPOLATED  0.70        griddata_linear     M,K        145.8
```

`confidence` is not a statistical confidence interval in Phase 1. It is an observability tag. It must not affect result selection, warning behavior, policy acceptance or analytic fallback.

## 13. Extrapolation Interface

Phase 1 reserves the extrapolation interface but keeps it disabled:

```text
allow_extrapolation = false
```

Therefore:

- Do not estimate targets outside the measured range.
- Do not return `QuerySource.EXTRAPOLATED`.
- If target is outside boundary, record `outside_boundary` and fall back.

Phase 3 may enable opt-in extrapolation with ratio guard / max bound.

## 14. Communication Handling

Communication operators do not enter wrapper interpolation in Phase 1. This does not mean communication has no interpolation-like behavior at all. The existing base `ProfilingDataSource` may return `INTERPOLATED` for alpha-beta/message_bytes paths. Phase 1 only avoids adding another 1D/2D/3D wrapper interpolation layer on top of communication.

## 15. Candidate Aggregation and Validation

Candidate aggregation is split into semantic filtering and interpolation-structure validation:

1. Canonicalize CSV rows and runtime target using the same shape rules.
2. Filter by policy/regime fields.
3. Validate latency.
4. Group candidates by axis tuple.
5. Aggregate duplicate points, defaulting to mean latency.
6. Build 1D boundary, 2D grid/convex hull or 3D grid/convex hull.
7. Run interpolation only when the structure is complete.
8. Record raw count, filtered count, aggregated count, dropped count, boundary, corner points and method in details.

The first filter excludes semantically inconsistent candidates. The second validation checks whether the remaining candidates can form a valid interpolation structure.

## 16. Test Plan

Math tests:

- `find_boundary()` hit, boundary hit and out-of-range failure.
- `linear_interp()` midpoint / quarter point.
- `griddata_linear_interp()` 2D and 3D success.
- `griddata_linear_interp()` outside convex hull failure.
- `griddata_linear_interp()` covers both regular-grid and irregular point sets. Dedicated bilinear/trilinear functions are not Phase 1 test targets.
- `validate_latency()` rejects NaN, infinity and negative values.
- `estimate_confidence()` is display-only.

Candidate index tests:

- Build M/K/N buckets from compute CSV rows.
- Validate `BatchMatMulV2` and `TransposeBatchMatMul` `(H,M,K) x (H,K,N)` extraction.
- Ensure `_FLATTEN_BATCH_KERNELS` flattening is not accidentally applied to MatMul kernels.
- Ensure RoPE/SwiGlu kernels do not accidentally enter M/K/N multidimensional interpolation without dedicated extractors.
- Build attention `seq/avg_seq_len`, `seq+batch`, `seq+heads/head_dim`, `seq+batch+heads/head_dim` buckets.
- Ensure `Accelerator Core` is not a regime key unless exact lookup uses it.
- Ensure transpose/layout does not split buckets more strictly than exact lookup.

Integration tests:

1. Exact hit returns `MEASURED`.
2. Compute exact miss with complete candidates returns `INTERPOLATED`.
3. Attention_special exact miss with complete candidates returns `INTERPOLATED`.
4. Missing candidates fall back to base behavior.
5. `--disable-profiling-interpolation` disables wrapper interpolation.
6. Communication operators do not enter wrapper interpolation.
7. Details and shape_match_info include dimension, axes, method, candidates and confidence.
8. Attention `axis_transform=sqrt` applies to both candidates and target.
9. `actual_seq_lengths_shape/values` that cannot reliably express batch downgrades rather than being misread.

Holdout tests:

- Compute holdout: build a MatMul/GEMM 2D/3D grid, remove one target point and verify interpolation.
- Attention holdout: cover `avg_seq_len` 1D and synthetic/enriched 2D/3D grids with batch/head/head_dim.

The implementation should produce a visible holdout / accuracy-loss report, either markdown or JSON. At minimum it should list target axes, removed measured latency, interpolated latency, absolute error, relative error, source, confidence, method and candidate count for every holdout sample.

## 17. Implementation Steps

Recommended Phase 1 steps:

1. Add user config and CLI rollback switch.
2. Wrap the profiling datasource construction path with `InterpolatingDataSource`.
3. Add `interpolation_math.py`.
4. Add the candidate index structure.
5. Implement compute M/K/N target extraction and 1D/2D/3D lookup.
6. Implement attention_special `seq/avg_seq_len`, `batch`, `heads/head_dim` target extraction and 1D/2D/3D lookup, with downgrade when CSV fields or candidates are insufficient.
7. Populate `QueryResult.details` and `shape_match_info`.
8. Add debug log and final source/confidence display.
9. Add unit tests, datasource integration tests and synthetic holdout tests.

## 18. Open Questions

Phase 1 keeps only a few review questions open:

1. Whether the wrapper should be enabled by default as recommended, with `--disable-profiling-interpolation` as rollback.
2. Whether attention CSV enrichment should add an explicit `Runtime batch_size` column to improve production 2D/3D hit rate for the batch axis.
3. Whether the candidate index should be a generic `CandidateIndex` or split into `ComputeIndex` and `AttentionIndex`.
4. Whether dedicated regular-grid bilinear/trilinear functions are worth adding later. Phase 1 uses `griddata_linear_interp` only.
5. Cross-format shape canonicalization/transform remains unresolved. For example, if measured CSV points only exist in NZ/FRACTAL_NZ format but runtime queries are ND, interpolation reuse should only be considered after exact lookup rules and data validation prove a reliable canonicalization.

## 19. Migration and Compatibility

### 19.1 Default Behavior

Phase 1 wraps the profiling datasource construction path with `InterpolatingDataSource` by default, but it does not blindly interpolate every existing `op_mapping.yaml` entry. Runtime behavior is still constrained by:

- Exact-first lookup. `MEASURED`, existing base `INTERPOLATED` and future base `EXTRAPOLATED` results are returned as-is.
- Wrapper fallback only covers the compute and attention_special allowlists.
- Entries without explicit `interpolation_policy` use conservative built-in defaults.
- Schema, regime-key, candidate-structure and latency validation failures downgrade or fall back to base behavior.

Existing entries can continue to run unchanged. New policy fields are optional and are used only to narrow or explicitly document interpolation behavior.

### 19.2 QueryResult and Downstream Compatibility

`QueryResult` remains backward compatible:

- `source` continues to distinguish `MEASURED`, `INTERPOLATED`, `PARTIAL` and other result origins.
- `confidence` is a display/observability field. It does not select winners or change scheduling decisions.
- `details` and `shape_match_info` may gain optional keys such as interpolation dimension, axes, method, candidate counts, boundary and fallback reason.
- Downstream consumers should read these fields as optional dictionaries and must not assume a fixed details key set.

Reports such as throughput_optimizer may display the new information, but they must not change optimization decisions solely because confidence or a details key is missing.

### 19.3 CSV Schema Compatibility

Phase 1 does not require all existing CSVs to be versioned or enriched immediately:

- Existing CSV exact hits keep exact behavior.
- Missing high-dimensional attention fields downgrade to `seq/avg_seq_len` 1D or fall back.
- New candidate fields are optional columns; old CSVs with missing columns must not fail to load.
- If a later phase introduces required CSV columns or a CSV version, it must also provide compatible readers and migration tooling.

### 19.4 Rollout and Rollback

Phase 1 provides `--disable-profiling-interpolation` as the hard rollback switch. Recommended rollout:

1. Shadow/dry-run: record candidates and predicted latency, but keep using the base result.
2. Allowlist enablement: return wrapper `INTERPOLATED` only for validated compute / attention_special kernels or model families.
3. Default enablement: keep the disable switch and debug details while monitoring hit rate, fallback reason and holdout error.

If behavior regresses, users can disable the wrapper immediately, and maintainers can narrow impact through policy allowlists or stricter regime keys.

## 20. Risks and Mitigation

| Risk | Impact | Mitigation |
|---|---|---|
| Default wrapper changes old behavior | Exact miss may return interpolation instead of the old fallback | Exact-first behavior, compute/attention_special allowlist, and `--disable-profiling-interpolation`. |
| Shape normalization diverges from exact lookup | Incorrect cross-regime interpolation | Reuse or strictly align current exact lookup normalizers; test special cases. |
| 2D/3D candidates are incomplete or degenerate | Invalid interpolation result | Downgrade or fall back. Never extrapolate in Phase 1. Record reasons in details/debug log. |

## 21. Acceptance Criteria

1. Compute multidimensional interpolation is end-to-end usable: in `--performance-model profiling`, MatMul/GEMM compute operators with CSV exact miss and complete candidates can return `QuerySource.INTERPOLATED` for M/K/N 1D, 2D and 3D interpolation.
2. Attention_special multidimensional interpolation is end-to-end usable: direct `query_mode: attention_special` operators and attention sub-kernel fallback from decomposers can execute `seq/avg_seq_len` 1D, `seq+batch` or `seq+heads/head_dim` 2D, and `seq+batch+heads/head_dim` 3D code paths. Production queries return `INTERPOLATED` only when fields are derivable, candidates are complete and regime matches; otherwise they downgrade or fall back predictably.
3. Interpolation results are reversible and explainable: users can switch back to the base `ProfilingDataSource` behavior with `--disable-profiling-interpolation`, and output/report/details/debug logs show `source`, `confidence`, interpolation axes, method, candidates and fallback reason. Confidence is displayed only and does not affect result selection.
