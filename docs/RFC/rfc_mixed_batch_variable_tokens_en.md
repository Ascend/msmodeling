# RFC: Current Mixed-Batch Variable-Token Modeling in Throughput Optimizer

## Metadata

| Item | Content |
|:---|:---|
| **Status** | Approved |
| **Author** | stormchasingg |
| **Updated Date** | 2026-06-26 |
| **Related Links** | |

---

## 1. Summary

This document records the current implementation of variable-token throughput optimization in the codebase.

Compared with the original RFC version, the current implementation still supports mixed-batch modeling for variable-length prefill workloads, but the code structure has been significantly simplified and renamed:

- the CLI uses the required `--input-length` argument for both fixed-length and distribution modes
- `--input-length` accepts either a positive integer or an existing length-distribution YAML file path
- distribution loading is handled by `load_length_distribution()`, while workload construction lives in `OptimizerData`
- mixed-batch execution is implemented through `_get_batched_forward_info()`
- final reporting no longer relies on a dedicated summary subclass
- batched detail-row expansion is handled directly inside `OptimizerSummary`

The current implementation only applies to:

- `cli.inference.throughput_optimizer`
- disaggregation mode
- prefill-only runs
- `TTFT` constrained searches

It does not apply to:

- PD ratio optimization mode
- Monte Carlo sampling
- request arrival distribution modeling

## 2. Current Scope and Entry Conditions

### 2.1 CLI behavior

The CLI now exposes a single required input-length argument:

- `--input-length`

Its accepted values are:

- a positive integer, which selects fixed-length mode
- an existing YAML file path, which selects variable-length distribution mode

This validation is enforced by `check_positive_integer_and_string()` in `serving_cast/service/utils.py` and used by `cli/inference/throughput_optimizer.py`.

### 2.2 Distribution-file mode

There is no separate `--length-distribution` CLI switch.

When `--input-length` is a file path, the CLI treats it as a length-distribution YAML file. For example:

- `--input-length serving_cast/example/length_distribution.yaml`

This mode is currently restricted to:

- `--disagg`
- `--ttft-limits`
- no `--tpot-limits`
- no PD ratio optimization

If these conditions are not satisfied, the CLI rejects the run.

At runtime, `args.input_length` remains:

- `int` in fixed-length mode
- `str` in distribution-file mode

`ParallelRunner` loads the `LengthDistribution` object from the file path and passes it into `OptimizerData.length_distribution`. In this mode, `OptimizerData.input_length` is set to `None`.

## 3. Data Model

### 3.1 Distribution types

`serving_cast/service/utils.py` defines:

- `LengthBin`
- `LengthDistribution`

Each bin contains:

- `min_tokens`
- `max_tokens`
- `weight`

Validation rules:

- `min_tokens >= 0`
- `max_tokens > min_tokens`
- `weight > 0`
- adjacent bins must not overlap

Weights are not required to sum to `1`.
The implementation normalizes them internally when building representative rows.

### 3.2 OptimizerData fields

`OptimizerData` currently contains both fixed-length and distribution-mode fields:

- `input_length`
- `length_distribution`
- `output_length`
- `batch_size`
- `ttft_limits`
- `tpot_limits`
- `prefix_cache_hit_rate`
- other serving and search parameters

Distribution mode is identified by:

- `optimizer_data.length_distribution is not None`

## 4. Variable-Token Workload Construction

### 4.1 Representative rows

`OptimizerData.get_representative_rows()` converts each length bin into a representative row.

The current implementation uses the bin midpoint by default and returns rows with:

- `num_input_tokens`
- `query_len`
- `request_ratio`

Semantics:

- `num_input_tokens` is the representative original input-token count
- `query_len` is the effective prefill length after applying prefix-cache reduction
- `request_ratio` is the normalized bin weight

### 4.2 Effective input length

`OptimizerData.get_effective_input_length()` behaves differently by mode:

- fixed-length mode:
  - returns the scalar effective input length after prefix-cache reduction
- distribution mode:
  - returns the weighted average of representative `query_len`

Chunk planning is handled by `OptimizerData.get_prefill_chunk_plan()`, which uses the effective input length and `max_batched_tokens`.

For distribution mode, the CLI performs a preflight chunked-prefill check by building an `OptimizerData(length_distribution=...)` object and calling `get_prefill_num_chunks()`. Distribution mode does not currently support chunked prefill, so users should increase `--max-batched-tokens` when the distribution's effective prefill length would require multiple chunks.

### 4.3 Integer sample allocation

`OptimizerData.build_concurrency_samples(concurrency)` expands the distribution into a concrete mixed batch.

The implementation:

1. computes ideal sample counts from `concurrency * request_ratio`
2. takes `floor(...)` as the base allocation
3. assigns the remaining requests using the largest-remainder method

Returned rows contain:

- `num_input_tokens`
- `query_len`
- `request_ratio`
- `samples`

This produces a deterministic mixed-batch composition for a given `concurrency`.

## 5. Execution Path

### 5.1 Fixed-length path

`BaseThroughputOptimizer._get_forward_info()` is still the standard path for:

- fixed-length prefill
- decode

It constructs a single `RequestInfo` template and runs:

- `generate_inputs`

### 5.2 Mixed-batch path

`BaseThroughputOptimizer._get_batched_forward_info()` is the current mixed-batch path.

It:

1. calls `optimizer_data.build_concurrency_samples(concurrency)`
2. expands those rows into a real heterogeneous `List[RequestInfo]`
3. repeats each row according to `samples`
4. runs inference with:
   - `generate_inputs_varlen`

Request fields are aligned with `RequestInfo` semantics:

- `num_input_tokens` for original input-token count
- `query_len` for actual prefill computation length

## 6. Disaggregation Integration

`DisaggThroughputOptimizer.get_inference_info()` now supports both modes:

- fixed-length
- variable-token mixed-batch

The branch condition is:

- `variable_input_mode = optimizer_data.length_distribution is not None`

### 6.1 Mixed-batch prefill

Under variable-token prefill:

1. `_get_batched_forward_info()` is used
2. `latency_ms` is computed from model execution time plus `serving_cost`
3. throughput is computed from the true batch token count:

```text
total_input_tokens = Σ(num_input_tokens * samples)
token/s = total_input_tokens / ttft * 1000
```

This replaces the old scalar formula based on one `input_length`.

### 6.2 Summary rows

The resulting DataFrame contains:

- one aggregate row
- multiple composition detail rows

The aggregate row uses:

- `num_input_tokens = "all"`
- `request_ratio = 1.0`
- `samples = concurrency`

Detail rows reuse the same configuration fields but clear performance columns such as:

- `ttft`
- `tpot`
- `token/s`
- `token/s/device`
- `percentage_breakdowns`

## 7. Final Report and Table Rendering

### 7.1 Summary class structure

The current implementation does not use a dedicated summary subclass for mixed-batch mode.

Instead, `OptimizerSummary` itself handles both:

- regular fixed-length final output
- mixed-batch final output

### 7.2 Best-row selection

`OptimizerSummary._prepare_agg_disagg_results()` still performs the base filtering and ranking:

- filter by `ttft` and `tpot` limits
- sort by `token/s`
- keep the best row for each `parallel`

This selection happens on the aggregate rows.

### 7.3 Composition-row expansion

If `args.input_length` is a string path, `OptimizerSummary._get_agg_disagg_final_out()` dispatches to:

- `_get_agg_disagg_final_out_batched()`

That path:

1. selects the best aggregate rows
2. calls `_expand_composition_rows()`
3. appends the matching detail rows from `self._summary_df`

The matching keys are:

- `parallel`
- `batch_size`
- `concurrency`
- `num_devices`

Ordering rules:

- aggregate row first
- detail rows after
- detail rows sorted by `num_input_tokens`

### 7.4 Batched final table

The mixed-batch final table is rendered by:

- `_get_disagg_table_buf_batched()`

This table is currently prefill-only and shows:

- `Top`
- `num_devices`
- `num_input_tokens`
- `request_ratio`
- `samples`
- `concurrency`
- `TTFT (ms)`
- `Throughput (token/s)`
- `parallel`
- `batch_size`

`input_length` and `output_length` are intentionally not shown in the batched final table because the composition rows are centered on:

- original representative token count
- request ratio
- allocated sample count

Performance columns on detail rows are rendered as `-`.

## 8. Module Interaction Diagram

```bash
CLI Argument Parsing (throughput_optimizer.py)
    │
    ├─ Required --input-length
    │   ├─ positive integer → fixed-length mode
    │   └─ existing YAML path → distribution-file mode
    │
    ├─ Is input_length a YAML path?
    │   ├─ No
    │   │   └─ Use scalar input_length path
    │   │
    │   └─ Yes
    │       ├─ Validate:
    │       │   ├─ disagg only
    │       │   ├─ prefill only (--ttft-limits set)
    │       │   └─ no --tpot-limits / no PD ratio optimization
    │       ├─ load_length_distribution(input_length)
    │       ├─ Build OptimizerData(length_distribution=...)
    │       ├─ Check that chunked prefill is not required
    │       └─ Use distribution-aware prefill path
    │
    └─ ParallelRunner(args)
        │
        ├─ Is input_length a YAML path?
        │   ├─ No
        │   │   └─ OptimizerData(input_length=<int>)
        │   │
        │   └─ Yes
        │       └─ OptimizerData(input_length=None,
        │                         length_distribution=load_length_distribution(input_length))
        │
        └─ run_disagg()
            │
            ├─ For each TP/parallel candidate
            │   └─ _get_df_list()
            │       └─ DisaggThroughputOptimizer.run()
            │           │
            │           ├─ Binary-search batch size
            │           └─ For each candidate batch
            │               └─ get_inference_info()
            │                   │
            │                   ├─ length_distribution is None?
            │                   │   ├─ Yes → _get_forward_info()
            │                   │   └─ No  → _get_batched_forward_info()
            │                   │            │
            │                   │            ├─ build_concurrency_samples(concurrency)
            │                   │            ├─ Expand rows into heterogeneous RequestInfo list
            │                   │            └─ run_inference(generate_inputs_varlen)
            │                   │
            │                   ├─ Compute TTFT / throughput
            │                   └─ Build:
            │                       ├─ one aggregate row
            │                       └─ multiple composition detail rows
            │
            └─ OptimizerSummary.report_final_result(args)
                │
                ├─ args.input_length is YAML path?
                │   ├─ No  → _get_agg_disagg_final_out()
                │   │         └─ _get_disagg_table_buf()
                │   │
                │   └─ Yes → _get_agg_disagg_final_out_batched()
                │             │
                │             ├─ _prepare_agg_disagg_results()
                │             ├─ _expand_composition_rows()
                │             └─ _get_disagg_table_buf_batched()
                │
                └─ Print overall best configuration + final table
```

## 9. Ongoing Work and Limitations

The following directions are already identified and are still in progress:

1. variable-token mixed-batch modeling for aggregation mode
2. variable-token mixed-batch modeling for decode-only scenarios

Beyond that, current limitations include:

1. distribution mode only works for disaggregation prefill with `TTFT` limits
2. distribution mode does not currently support chunked prefill
3. PD ratio optimization does not support variable-token mixed-batch modeling
4. best-row selection still happens on aggregate rows before detail-row expansion

## 10. Notes for Future Changes

If the implementation evolves again, the following areas are most sensitive and should be updated together:

- CLI contract for `--input-length` integer/path parsing
- `OptimizerData` naming and workload-construction helpers
- `BaseThroughputOptimizer` mixed-batch execution entry
- `DisaggThroughputOptimizer` summary row schema
- `OptimizerSummary` batched final-report formatting

In particular, any future reintroduction of:

- a separate distribution CLI argument
- summary subclasses
- decode-mode batched reporting
- aggregation-mode variable-token support

should be documented as a separate follow-up RFC update.

---

# Implementation Update: Aggregation and Chunked Prefill

**Updated Date:** 2026-08-04

This update records behavior added after the original RFC description. Where this section conflicts with the earlier
scope or limitation statements, this section describes the current implementation.

## 1. Updated Scope

Length-distribution input now supports:

- aggregation mode, including TTFT and TPOT constrained runs
- disaggregation prefill-only mode
- aggregation and disaggregation prefill splitting under `max_batched_tokens`

The following scenarios remain unsupported or incomplete:

- disaggregation decode-only variable-token modeling
- PD ratio optimization with variable-token input
- vLLM-style Prefill/Decode overlap for variable-token aggregation chunks
- request arrival distributions and Monte Carlo sampling

## 2. Concurrency-Aware Chunk Planning

`PrefillChunk` now includes request-completion information:

```python
@dataclass
class PrefillChunk:
    index: int
    query_len: int
    seq_len: int
    is_last_chunk: bool = False
```

For fixed-length input, `get_prefill_chunk_plan()` retains the original single-request splitting behavior. Each index
represents one homogeneous chunk shape that is expanded through concurrency during execution.

For distribution input, `get_prefill_chunk_plan(concurrency)` requires model-rank concurrency and calls
`build_concurrency_samples(concurrency)`. It expands the sampled requests and fills each chunk up to
`max_batched_tokens`. Multiple heterogeneous `PrefillChunk` entries can share the same index and are executed
together.

For example:

```text
sampled query lengths = [50, 50, 150, 150]
max_batched_tokens = 200

index 0:
  query_len=50,  seq_len=50,  is_last_chunk=True
  query_len=50,  seq_len=50,  is_last_chunk=True
  query_len=100, seq_len=100, is_last_chunk=False

index 1:
  query_len=50,  seq_len=150, is_last_chunk=True
  query_len=150, seq_len=150, is_last_chunk=True
```

`get_prefill_num_chunks(chunk_plan)` derives the count from the plan currently being executed:

```python
max(chunk.index for chunk in chunk_plan) + 1 if chunk_plan else 0
```

It accepts an existing plan rather than regenerating one from concurrency, so the reported count cannot diverge from
the plan passed to execution. The `max(index) + 1` calculation also does not depend on the last list element, because
several request entries can belong to the same chunk index.

## 3. Chunked Mixed-Batch Execution

`BaseThroughputOptimizer._get_batched_forward_info()` now accepts an optional chunk plan.

The method consistently returns:

```python
tuple[list[tuple[ModelRunnerMetrics, int]], list[dict]]
```

Each result tuple contains the execution metrics and the number of requests completed by that execution. Without a
chunk plan, the original heterogeneous request construction is preserved, including `num_input_tokens`; the batch is
executed once and returned as a one-element result list. Its completed-request count is the sum of `samples` on the
current DP rank. Completion count remains ServingCast scheduling metadata rather than being added to the cached
`ModelRunnerMetrics` type.

With a chunk plan, it:

1. groups adjacent `PrefillChunk` entries by index
2. converts each group into a heterogeneous `list[RequestInfo]`
3. invokes `run_inference(..., generate_inputs_func=generate_inputs_varlen)` once per index
4. counts completed requests using `is_last_chunk`
5. stops issuing later chunk executions after a result reports negative available device memory

Before grouping, execution validates that indices start at zero and are contiguous and non-decreasing. Invalid plans
such as `[0, 1, 0]`, `[1, 2]`, or `[0, 2]` raise `ValueError` instead of being silently reordered. This protects the
sequential `seq_len` semantics and ensures that equal indices form one adjacent execution group.

For the example above, the two inference inputs are:

```text
index 0: [(query_len=50, seq_len=50),
          (query_len=50, seq_len=50),
          (query_len=100, seq_len=100)]

index 1: [(query_len=50, seq_len=150),
          (query_len=150, seq_len=150)]
```

## 4. Aggregation Metrics

The aggregation variable-token path is now:

```text
get_inference_info
  -> get_prefill_chunk_plan(model-rank concurrency)
  -> _get_batched_full_prefill_metrics
  -> _get_batched_forward_info(chunk_plan=...)
  -> run_inference(generate_inputs_varlen), once per chunk index
```

`_get_batched_full_prefill_metrics()` accumulates chunk latency in index order. Requests marked with
`is_last_chunk=True` receive the accumulated completion timestamp as their first-token time. Average TTFT is:

```text
TTFT = sum(completed requests at an index * accumulated time at that index)
       / total sampled requests
```

For two 10 ms chunks that each complete two of four requests, TTFT is 15 ms. Aggregation mode does not apply
`serving_cost`, because there is no PD transfer in this mode. `prefill_latency` records the first chunk model latency,
`prefill_last_latency` records the last chunk model latency, and Prefill memory uses the minimum available memory
across all chunks.

Decode latency is still calculated separately using the weighted effective input length. The variable-token path does
not enter `_simulate_chunked_prefill()` or `DecodeFirstWithSlack`, so Prefill and Decode do not overlap in the current
aggregation approximation. Fixed-length aggregation chunked-prefill continues to use that scheduler.

## 5. Reporting Changes

Aggregation results now include the same distribution composition structure as disaggregation results:

- one aggregate row with `num_input_tokens="all"`
- detail rows containing representative `num_input_tokens`, normalized `request_ratio`, and integer `samples`
- TTFT, TPOT, throughput, and breakdown fields cleared on detail rows

Best-row selection is performed on aggregate rows before matching detail rows are appended. The aggregation final
table includes TPOT in addition to TTFT and throughput.

Rows allocated zero integer samples are omitted by `build_concurrency_samples()` and therefore do not appear in the
final table. This can happen to low-weight bins when model-rank concurrency is small.

The obsolete CLI preflight construction of `OptimizerData` and its disabled chunk-rejection block have been removed.
The CLI now only loads and validates the length-distribution file at this stage; the concurrency-aware plan is built
later by the selected optimizer, where DP-rank concurrency is available.

## 6. Disaggregation Chunk Execution and Remaining Boundaries

Disaggregation variable-token prefill now passes its concurrency-aware chunk plan to
`_get_batched_forward_info()`. It consumes the same `list[(metrics, completed_requests)]` representation as
aggregation mode and invokes `run_inference()` once per chunk index.

Disaggregation keeps the existing top-level branch structure. Within the variable-input branch it:

1. initializes cumulative Prefill latency with `serving_cost`, matching fixed-input disaggregation chunk handling
2. adds each chunk's model latency in execution order
3. weights the current cumulative latency by the requests completed at that chunk
4. divides the request-time sum by the sampled request count on the current DP rank to obtain TTFT
5. calculates throughput from global input tokens and the total Prefill phase latency
6. selects the minimum available device memory and aggregates normalized breakdowns across executed chunks

For example, with `serving_cost=3 ms`, a 10 ms chunk completing two requests and a following 20 ms chunk completing
two requests produce completion times of 13 ms and 33 ms. The average TTFT is `(2 * 13 + 2 * 33) / 4 = 23 ms`, while
the Prefill phase latency used for throughput is 33 ms. `latency_ms / concurrency` is not TTFT because batch latency is
observed by every request rather than divided among requests.

If execution stops because of Prefill OOM before all sampled requests complete, request-weighted TTFT is incomplete;
the implementation retains cumulative phase latency for diagnostics and lets the memory early-stop reason take
precedence.

Variable-token aggregation does not yet model a vLLM continuous-batching step containing both Prefill and Decode
requests. Supporting that behavior requires a mixed scheduler representation and a single mixed forward call rather
than separately modeled Prefill and Decode latency.

## 7. Module Interaction

```text
CLI (--input-length=<distribution.yaml>, --max-batched-tokens)
  │
  ├─ load_length_distribution()
  │
  └─ ParallelRunner
       │
       ├─ Build OptimizerData(length_distribution=..., max_batched_tokens=...)
       │
       ├─ run_agg()
       │    └─ AggThroughputOptimizer.get_inference_info(optimizer_data)
       │         ├─ optimizer_data.get_prefill_chunk_plan(DP-rank concurrency)
       │         ├─ optimizer_data.get_prefill_num_chunks(chunk_plan)
       │         └─ _get_batched_full_prefill_metrics(..., chunk_plan)
       │              └─ _get_batched_forward_info(..., chunk_plan)
       │
       └─ run_disagg()
            └─ DisaggThroughputOptimizer.get_inference_info(optimizer_data)
                 ├─ optimizer_data.get_prefill_chunk_plan(DP-rank concurrency)
                 ├─ optimizer_data.get_prefill_num_chunks(chunk_plan)
                 └─ variable-input Prefill branch
                      └─ _get_batched_forward_info(..., chunk_plan)

Both paths call:

              BaseThroughputOptimizer._get_batched_forward_info()
                                      │
                    optimizer_data.build_concurrency_samples()
                                      │
                validate contiguous, non-decreasing chunk indices
                                      │
                   group PrefillChunk entries by index
                                      │
                         for each chunk index
                                      │
              build heterogeneous RequestInfo list for this index
                                      │
              ModelRunner.run_inference(generate_inputs_varlen)
                                      │
              append (metrics, completed_requests) to chunk_results
                                      │
                    device_memory_available_gb < 0?
                         ├─ Yes → break; do not execute later chunks
                         └─ No  → continue with the next chunk index ──┐
                                      ▲                               │
                                      └───────────────────────────────┘
                                      │
                         after loop or early stop
                                      │
              return list[(ModelRunnerMetrics, completed_requests)]
                                      │
                       ┌──────────────┴──────────────┐
                       ▼                             ▼
          Agg metric aggregation        Disagg metric aggregation
          - request-weighted TTFT        - request-weighted TTFT
          - separate Decode latency      - global input throughput
          - no serving_cost              - serving_cost once
          - tightest chunk memory        - breakdown/memory aggregation
                       │                             │
                       └──────────────┬──────────────┘
                                      ▼
                  OptimizerSummary aggregate row + detail rows
```

Cross-request chunk packing example:

```text
DP-rank requests: A=50, B=50, C=150, D=150 tokens
max_batched_tokens=200

chunk index 0 (200 query tokens):
  A: query_len=50,  seq_len=50,  is_last_chunk=True
  B: query_len=50,  seq_len=50,  is_last_chunk=True
  C: query_len=100, seq_len=100, is_last_chunk=False

  run_inference([A(50, 50), B(50, 50), C(100, 100)])
  completed_requests=2

chunk index 1 (200 query tokens):
  C: query_len=50,  seq_len=150, is_last_chunk=True
  D: query_len=150, seq_len=150, is_last_chunk=True

  run_inference([C(50, 150), D(150, 150)])
  completed_requests=2
```

Request C is split across two chunks, while each chunk can contain pieces from different requests. If both inference
calls take 10 ms, A and B complete at 10 ms and C and D complete at 20 ms, producing an average TTFT of 15 ms.

`ModelRunnerMetrics` remains limited to model-execution measurements. Request completion count is attached by the
ServingCast layer after each inference call because completion depends on `PrefillChunk.is_last_chunk`, not on the
model shape or the cached model-runner result.

## 8. Regression Coverage

The updated regression coverage verifies:

- single-execution and multi-chunk paths use the same result-list shape
- non-chunk completion count equals the sum of sampled requests
- variable aggregation splits inference by token budget and computes request-weighted TTFT
- disaggregation uses global input tokens for throughput while request completion counts remain per DP rank
- disaggregation adds `serving_cost` once, aggregates chunk memory, and computes request-weighted TTFT
- unordered or non-contiguous chunk indices are rejected before `groupby` execution
