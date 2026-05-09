# `tools/perf_data_collection`

Utilities for parsing profiling outputs, back-filling runtime metadata, generating theory-driven shape grids, replaying operators with `msprof`, and validating the resulting performance database.

## Directory Layout

```text
tools/perf_data_collection/
  comm_bench/
  grid_generator/
  op_replay/
  parsers/
  fia_common.py
  fill_fia_runtime_metadata.py
  generate_shape_grid.py
  memory_estimator.py
  start_microbench.py
  README.md
```

Offline comparison scripts now live in the sibling directory
`tools/perf_data_analysis/`.

## Typical Workflow

1. Parse raw profiling output with `parsers/parse_kernel_details.py`.
2. If `FusedInferAttentionScore.csv` needs richer runtime fields, fill them with `fill_fia_runtime_metadata.py`.
3. Expand sparse operator coverage with `generate_shape_grid.py`.
4. Replay operators and write microbench results back with `start_microbench.py`.
5. Use `tools/perf_data_analysis/` and `comm_bench/` to compare traces,
   inspect gaps, and validate communication data.

## Top-level Files

| File | Purpose |
| --- | --- |
| `fia_common.py` | Shared helpers for FIA shapes and metadata parsing. |
| `fill_fia_runtime_metadata.py` | Merges FIA runtime JSONL metadata into `FusedInferAttentionScore.csv`. |
| `generate_shape_grid.py` | Appends theory-generated shape rows into database CSV files. |
| `memory_estimator.py` | Estimates tensor memory usage for theory-mode shape generation. |
| `start_microbench.py` | Runs replay under `msprof`, aggregates `op_summary_*.csv`, and writes results back to the database. |

## Subdirectories and Related Tools

### Related: `tools/perf_data_analysis/`

Offline analysis and reporting helpers in the sibling package
`tools/perf_data_analysis/`.

| File | Purpose |
| --- | --- |
| `compute_m6.py` | Compares end-to-end timing between TC trace data and profiling trace data. |
| `generate_op_comparison.py` | Aggregates TC vs profiling comparisons by operator. |
| `generate_per_shape_comparison.py` | Produces per-`(kernel_type, shape)` comparison CSV output. |

### `comm_bench/`

Communication microbench generation and validation tools.

| File | Purpose |
| --- | --- |
| `generate_comm_microbench.py` | Generates or directly runs HCCL microbench workloads. |
| `run_comm_bench.sh` | Batch collection entry script for communication benchmark data. |
| `validate_comm_alignment.py` | Checks whether measured communication results align with the parser model. |

### `grid_generator/`

Core engine for theory-mode shape expansion.

| File | Purpose |
| --- | --- |
| `config.py` | Loads and validates shape-grid configuration. |
| `config.yaml` | Routing and generation rules for theory-mode shape expansion. |
| `evaluator.py` | Safe expression evaluator used by theory-mode dimensions. |
| `model_configs.py` | Model architecture presets used to prune or derive shape candidates. |
| `runner.py` | Main theory-mode generation pipeline. |
| `shape_grids.py` | Shared grid definitions. |
| `theory_router.py` | Routes operators to the appropriate theory generator. |
| `utils.py` | Shared CSV, row, and shape helpers. |
| `generators/base.py` | Base interfaces and helpers for generator implementations. |
| `generators/fused_attention.py` | Theory generator for fused attention operators. |
| `generators/moe.py` | Theory generator for MoE-related operators. |

### `op_replay/`

Operator replay framework and per-operator replay scripts.

| File | Purpose |
| --- | --- |
| `common.py` | Shared CLI, path, tensor construction, and CSV utilities for replay. |
| `probe_dfc_constraints.py` | Probes shape constraints for `DispatchFFNCombine`. |
| `replay_framework.py` | Common replay framework used by individual operators. |
| `run_all_op.py` | Discovers and runs all `*_run.py` scripts. |
| `*_run.py` | Per-operator replay entry points. |

### `parsers/`

Profiling and trace parsing helpers.

| File | Purpose |
| --- | --- |
| `parse_kernel_details.py` | Converts `kernel_details*.csv` or profiling directories into per-operator CSV files. |
| `trace_to_csv.py` | Flattens TC Chrome trace JSON into CSV. |

## Main CLI Scripts

### `parsers/parse_kernel_details.py`

Parses one `kernel_details*.csv` file or a profiling directory and writes one CSV per operator into the target database directory.

```powershell
py -3 tools/perf_data_collection/parsers/parse_kernel_details.py `
  --profiling-path G:\path\to\profiling_dir `
  --database-path tensor_cast/performance_model/profiling_database/data/.../dev_0331
```

| Argument | Required | Description |
| --- | --- | --- |
| `--profiling-path` | Yes | A single `kernel_details*.csv` file or a profiling directory scanned recursively. |
| `--database-path` | No | Explicit output directory for generated operator CSV files. |
| `--device` | No | Device name used when deriving the output path. |
| `--vllm-version` | No | vLLM version or full version-directory name used when deriving the output path. |
| `--torch-version` | No | PyTorch version used when deriving the output path. |
| `--cann-version` | No | CANN version used when deriving the output path. |

### `fill_fia_runtime_metadata.py`

Back-fills runtime JSONL metadata into `FusedInferAttentionScore.csv`.

| Argument | Required | Description |
| --- | --- | --- |
| `--csv-path` | Yes | Path to `FusedInferAttentionScore.csv`. |
| `--jsonl-path` | Yes | Path to FIA runtime JSONL input. |
| `--output-path` | No | Output CSV path. Defaults to in-place overwrite. |
| `--metadata-tag` | No | Completeness tag written into matched rows. |

### `generate_shape_grid.py`

Appends theory-generated shape rows into existing database CSV files.

| Argument | Required | Description |
| --- | --- | --- |
| `--target-models` | No | Comma-separated model names used to prune GEMM `(N, K)` candidates. |
| `--data-dir` | No | Explicit CSV root directory. |
| `--device` | No | Device name used when deriving the database path. |
| `--vllm-version` | No | vLLM version used when deriving the database path. |
| `--torch-version` | No | PyTorch version used when deriving the database path. |
| `--cann-version` | No | CANN version used when deriving the database path. |
| `--rows` | No | Maximum appended rows per CSV. `0` means no cap. |
| `--seed` | No | Random seed for reproducible sampling. |
| `--max-hbm-gb` | No | Per-row HBM budget in GiB. `0` disables memory filtering. |

### `start_microbench.py`

Runs replay scripts under `msprof`, aggregates profiling metrics, and writes them back into matching operator CSV files.

| Argument | Required | Description |
| --- | --- | --- |
| `--database-path` | No | Explicit database directory to read and update. |
| `--device` | No | Device directory name used when deriving the target path. |
| `--vllm-version` | No | vLLM version used when deriving the target path. |
| `--torch-version` | No | PyTorch version used when deriving the target path. |
| `--cann-version` | No | CANN version used when deriving the target path. |
| `--prof-path` | No | Existing `PROF_*` directory to parse directly without launching `msprof`. |
| `--op` | No | Restrict updates to specific operator names. |
| `--dispatch-ffn-combine-ep-size` | No | EP size used for `DispatchFFNCombine` replay and row matching. |
| `--repeat-count` | No | Replay repeat count passed through to operator scripts. |
| `--update-mode` | No | `all` or `missing-only`. |
| `--fail-fast` | No | Stop immediately when one replay script fails. |
| `--prune-empty-duration-rows` | No | Delete rows whose replay and profiling durations are both invalid after writeback. |

### `parsers/trace_to_csv.py`

Converts a TC Chrome trace JSON file into flat CSV output.

| Argument | Required | Description |
| --- | --- | --- |
| `--trace` | Yes | Input TC Chrome trace JSON path. |
| `--output` | No | Output CSV path. Defaults to `stdout`. |

## Related Analysis Script Arguments

### `tools/perf_data_analysis/compute_m6.py`

| Argument | Required | Description |
| --- | --- | --- |
| `--tc-trace` | Yes | TC Chrome trace JSON file. |
| `--prof-trace` | Yes | Forward-pass profiling trace CSV file. |
| `--source-filter` | No | Comma-separated source filters. |
| `--json-output` | No | Output JSON path. |

### `tools/perf_data_analysis/generate_op_comparison.py`

| Argument | Required | Description |
| --- | --- | --- |
| `--trace-dir` | No | Directory containing forward-pass trace CSV files. |
| `--data-dir` | No | Database directory containing `op_mapping.yaml`. |
| `--output` | No | Output JSON path. |

### `tools/perf_data_analysis/generate_per_shape_comparison.py`

| Argument | Required | Description |
| --- | --- | --- |
| `--tc-trace` | Yes | TC Chrome trace JSON file. |
| `--prof-trace` | Yes | Profiling trace CSV file. |
| `--output` | No | Output CSV path. Defaults to `stdout`. |

## `comm_bench/` Script Arguments

### `comm_bench/generate_comm_microbench.py`

| Argument | Required | Description |
| --- | --- | --- |
| `--output-dir` | No | Directory to write per-op CSV files (requires `--do-run`). |
| `--ops` | No | Communication operators to include. |
| `--num-devices` | No | Number of devices in each communication group. |
| `--topology-tier` | No | Topology tier `0`, `1`, or `2`. |
| `--grid-shape` | No | Hardware grid shape. |
| `--dtype` | No | Tensor dtype. |
| `--bytes-grid` | No | Custom `message_bytes` grid. |
| `--do-run` | Yes | Run the benchmark directly (requires `torchrun`). |
| `--output-csv` | No | Single output CSV path, only valid for one operator. |
| `--bench-mode` | No | `kernel` or `event`. |

### `comm_bench/run_comm_bench.sh`

```bash
bash tools/perf_data_collection/comm_bench/run_comm_bench.sh [OUTPUT_DIR]
```

| Argument | Required | Description |
| --- | --- | --- |
| `OUTPUT_DIR` | No | Output directory for generated communication CSV files. |

### `comm_bench/validate_comm_alignment.py`

| Argument | Required | Description |
| --- | --- | --- |
| `--csv-dir` | Yes | Directory containing `hcom_*.csv`. |
| `--tolerance` | No | Acceptable ratio tolerance. Default is `2.0`. |
| `--verbose` | No | Print all checked rows. |

## Shared `op_replay/` Arguments

Most `op_replay/*_run.py` scripts share the following arguments:

| Argument | Required | Description |
| --- | --- | --- |
| `--database-path` | No | Explicit database directory. |
| `--device` | No | Device directory name. |
| `--vllm-version` | No | vLLM version or full version-directory name. |
| `--torch-version` | No | PyTorch version. |
| `--cann-version` | No | CANN version. |
| `--repeat-count` | No | Replay count per row. |
| `--update-mode` | No | `all` or `missing-only`. |

Environment variable:

- `MSMODELING_OP_REPLAY_REPEAT_COUNT` provides the default replay count when the CLI flag is not set.

### `op_replay/run_all_op.py`

| Argument | Required | Description |
| --- | --- | --- |
| `--database-path` | No | Explicit database directory. |
| `--device` | No | Device directory name. |
| `--vllm-version` | No | vLLM version. |
| `--torch-version` | No | PyTorch version. |
| `--cann-version` | No | CANN version. |
| `--repeat-count` | No | Replay count passed to each operator script. |
| `--update-mode` | No | Replay update mode forwarded to each operator script. |
| `--execution-mode` | No | `inprocess` or `subprocess`. |
| `--op` | No | Restrict execution to selected operators. |
| `--dispatch-ffn-combine-ep-size` | No | Forwarded only to `DispatchFFNCombine_run.py`. |
| `--continue-on-error` | No | Continue running after individual operator failures. |

### `op_replay/probe_dfc_constraints.py`

This script has no CLI arguments. Run it directly.

## `op_replay/*_run.py` Overview

| File | Kernel Type | Extra Arguments | Purpose |
| --- | --- | --- | --- |
| `AddRmsNormBias_run.py` | `AddRmsNormBias` | None | Replays the fused AddRmsNormBias operator. |
| `Add_run.py` | `Add` | None | Replays `torch.add`. |
| `ArgMaxV2_run.py` | `ArgMaxV2` | None | Replays `torch.argmax`. |
| `AscendQuantV2_run.py` | `AscendQuantV2` | None | Replays `torch_npu.npu_quantize`. |
| `DispatchFFNCombine_run.py` | `DispatchFFNCombine` | `--ep-size`, `--balanced`, `--no-balanced` | Replays the fused DFC operator. |
| `DynamicQuant_run.py` | `DynamicQuant` | None | Replays `torch_npu.npu_dynamic_quant`. |
| `FusedInferAttentionScore_run.py` | `FusedInferAttentionScore` | None | Replays FIA. |
| `GatherV2_run.py` | `GatherV2` | None | Replays embedding and gather workloads. |
| `Index_run.py` | `Index` | None | Replays index-based tensor access. |
| `InterleaveRope_run.py` | `InterleaveRope` | None | Replays interleaved rope. |
| `KvRmsNormRopeCache_run.py` | `KvRmsNormRopeCache` | None | Replays KV rope cache. |
| `MaskedFill_run.py` | `MaskedFill` | None | Replays `masked_fill_`. |
| `MatMulCommon_run.py` | `MatMulCommon` | None | Replays the generic matmul path. |
| `MatMulV2_run.py` | `MatMulV2` | None | Replays `MatMulV2`. |
| `MatMulV3_run.py` | `MatMulV3` | None | Replays `MatMulV3`. |
| `PadV3_run.py` | `PadV3` | None | Replays pad based on input and output shapes. |
| `QuantBatchMatmulV3_run.py` | `QuantBatchMatmulV3` | None | Replays quantized batch matmul. |
| `ReshapeAndCacheNdKernel_run.py` | `ReshapeAndCacheNdKernel` | None | Replays reshape-and-cache. |
| `RINGMLAPrefillBF16Kernel_run.py` | `RINGMLAPrefillBF16Kernel` | None | Replays MLA prefill kernel. |
| `RmsNorm_run.py` | `RmsNorm` | None | Replays RMSNorm. |
| `Slice_run.py` | `Slice` | None | Replays Slice. |
| `SoftmaxV2_run.py` | `SoftmaxV2` | None | Replays Softmax. |
| `Sort_run.py` | `Sort` | None | Replays Sort. |
| `split_qkv_rmsnorm_rope_kernel_run.py` | `split_qkv_rmsnorm_rope_kernel` | None | Replays the custom Triton QKV kernel. |
| `SwiGlu_run.py` | `SwiGlu` | None | Replays SwiGlu. |
| `TensorMove_run.py` | `TensorMove` | None | Replays tensor copy. |
| `Transpose_run.py` | `Transpose` | None | Replays Transpose. |

## Common Commands

Parse profiling output:

```powershell
py -3 tools/perf_data_collection/parsers/parse_kernel_details.py `
  --profiling-path G:\path\to\profiling_dir `
  --database-path tensor_cast/performance_model/profiling_database/data/.../dev_0331
```

Generate theory-driven shape rows:

```powershell
py -3 tools/perf_data_collection/generate_shape_grid.py `
  --target-models dsv3,qwen3-32b `
  --data-dir tensor_cast/performance_model/profiling_database/data/.../dev_0331 `
  --rows 2000 `
  --seed 20260409
```

Replay operators and write back results:

```powershell
py -3 tools/perf_data_collection/start_microbench.py `
  --database-path tensor_cast/performance_model/profiling_database/data/.../dev_0331 `
  --repeat-count 1 `
  --update-mode missing-only
```

## `start_microbench.py` Update Modes

- `all`: update every matched row and append unmatched profiling samples into the target CSV.
- `missing-only`: replay and fill only rows whose `Average Duration(us)` and `Profiling Average Duration(us)` are both invalid (`0` or empty); rows that already contain at least one valid duration are skipped, and unmatched profiling samples are reported but not appended.

## Empty-row Pruning

- Default behavior: rows with both `Average Duration(us)` and `Profiling Average Duration(us)` invalid are kept so `missing-only` mode can retry them later.
- `--prune-empty-duration-rows`: opt-in cleanup that removes those rows after writeback. Use it only when you intentionally want to delete unrecoverable empty rows from the database.
