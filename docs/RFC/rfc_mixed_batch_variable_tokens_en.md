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
