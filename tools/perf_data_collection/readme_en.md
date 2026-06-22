# Performance Database Collection Tools

This guide explains how to use the tools under `tools/perf_data_collection/`: required environments, generated artifacts, and common troubleshooting.

Main tools in recommended order:

1. `parsers/parse_kernel_details.py`: parse profiling output into per-operator CSV files.
2. `generate_shape_grid.py`: append theory shape rows to expand CSV coverage.
3. `start_microbench.py`: run compute replay under `msprof`, aggregate results, and write durations back.
4. `comm_bench/generate_comm_microbench.py`: collect HCCL communication microbench data into `hcom_*.csv`.

## Scope

This directory generates and refreshes operator CSV data in the TensorCast performance database:

- Parse per-operator CSV from NPU profiling results.
- Append theory shape rows to expand CSV coverage.
- Replay operators on NPU or collect HCCL communication microbench data and write durations back.

Supported device directory names:

| Device directory |
| --- |
| `ATLAS_800_A2_376T_64G` |
| `ATLAS_800_A2_313T_64G` |
| `ATLAS_800_A2_280T_64G` |
| `ATLAS_800_A2_280T_64G_PCIE` |
| `ATLAS_800_A2_280T_32G_PCIE` |
| `ATLAS_800_A3_752T_128G_DIE` |
| `ATLAS_800_A3_560T_128G_DIE` |

The default device is `ATLAS_800_A3_752T_128G_DIE`. The batch collection script `comm_bench/run_comm_bench.sh` targets **`ATLAS_800_A3_752T_128G_DIE`** with hardware grid `48 8 2`. Other A3 variants (for example ROCE models) may use a different grid shape (such as `2 8 2`); adjust `--grid-shape` or use `generate_comm_microbench.py` manually.

## Prerequisites

Run commands from the repository root. Python **>= 3.10** is required.

Activate the Python environment first:

```bash
source try/bin/activate
```

Then use `python` to launch scripts below. On Windows you may use `py -3` instead (PowerShell line continuation with `` ` ``).

| Scenario | Requirements |
| --- | --- |
| Parse profiling | An existing `kernel_details*.csv` file or a profiling directory containing one. |
| Generate theory shapes | Target database directory already contains operator CSV files to extend. |
| Run compute replay | NPU available; CANN, `msprof`, `torch`, and `torch_npu` installed; some custom ops also need vLLM-Ascend custom OPP. |
| Collect HCCL comm data | Launch with `torchrun`; NPU/HCCL communication environment available. |

You can pass `--database-path` explicitly, or derive the path via `--device`, `--vllm-version`, `--torch-version`, and `--cann-version`.

Example database directories (paths already present in the repository):

```text
tensor_cast/performance_model/profiling_database/data/ATLAS_800_A3_752T_128G_DIE/vllm_ascend/vllm0.18.0_torch2.9.0_cann8.5
tensor_cast/performance_model/profiling_database/data/ATLAS_800_A3_752T_128G_DIE/vllm_ascend/vllm0.18.0_torch2.9.0_cann8.5_shape_generated
```

The first path is the base database after profiling parse; the second is the shape-expanded database with theory rows appended.

## Directory Layout

```text
tools/perf_data_collection/
  comm_bench/              # HCCL communication microbench collection
  grid_generator/          # theory shape generation core logic
  op_replay/               # compute operator replay framework and entry points
  parsers/                 # profiling parse entry points
  fia_common.py            # shared FIA shape/metadata helpers
  generate_shape_grid.py   # theory shape generation entry point
  memory_estimator.py      # HBM memory estimation
  start_microbench.py      # msprof orchestration, aggregation, and writeback
  readme.md                # Chinese guide
  readme_en.md             # English guide
```

## Recommended Workflow

### 1. Parse raw profiling output

```bash
python tools/perf_data_collection/parsers/parse_kernel_details.py \
  --profiling-path /path/to/profiling_dir \
  --database-path tensor_cast/performance_model/profiling_database/data/ATLAS_800_A3_752T_128G_DIE/vllm_ascend/vllm0.18.0_torch2.9.0_cann8.5
```

This creates or updates per-operator CSV files such as `MatMulV2.csv` and `FusedInferAttentionScore.csv` under the target database directory.

### 2. Generate theory shape rows

```bash
python tools/perf_data_collection/generate_shape_grid.py \
  --database-path tensor_cast/performance_model/profiling_database/data/ATLAS_800_A3_752T_128G_DIE/vllm_ascend/vllm0.18.0_torch2.9.0_cann8.5_shape_generated \
  --target-models dsv3,qwen3-32b \
  --rows 2000 \
  --seed 20260409
```

This appends theory-generated shape rows to existing CSV files. `--rows 0` means no per-CSV row cap.

### 3. Replay operators and write back durations

```bash
python tools/perf_data_collection/start_microbench.py \
  --database-path tensor_cast/performance_model/profiling_database/data/ATLAS_800_A3_752T_128G_DIE/vllm_ascend/vllm0.18.0_torch2.9.0_cann8.5_shape_generated \
  --repeat-count 1 \
  --update-mode missing-only
```

`start_microbench.py` runs `op_replay/run_all_op.py` under `msprof`, aggregates `op_summary_*.csv`, and writes durations back to database CSV files.

### 4. Collect ATLAS_800_A3_752T_128G_DIE HCCL communication data

```bash
bash tools/perf_data_collection/comm_bench/run_comm_bench.sh ./hccl_bench_data
```

This script targets **`ATLAS_800_A3_752T_128G_DIE`** (default `grid-shape 48 8 2`) and writes `hcom_*.csv`. For other A3 models, confirm the actual grid shape first; use `generate_comm_microbench.py` for finer control.

## Main Tools

| Stage | Tool | Main input | Main output | Use case |
| --- | --- | --- | --- | --- |
| Parse | `parsers/parse_kernel_details.py` | `kernel_details*.csv` or profiling directory | per-operator CSV | Convert real profiling results into database format. |
| Expand | `generate_shape_grid.py` | existing database CSV, model/shape rules | CSV with appended theory rows | Expand shape coverage before replay. |
| Writeback | `start_microbench.py` | database CSV, operator replay scripts | updated CSV, report files | Re-measure operator durations on NPU and refresh the database. |
| Comm collect | `comm_bench/generate_comm_microbench.py` | HCCL comm config, message-size grid | `hcom_*.csv` | Collect communication operator performance data. |
| Batch comm | `comm_bench/run_comm_bench.sh` | `ATLAS_800_A3_752T_128G_DIE` grid (`48 8 2`) and output dir | standard `hcom_*.csv` set | Batch collect comm data with 752T DIE defaults. |

### 1. Profiling parse: `parsers/parse_kernel_details.py`

| Argument | Required | Default | Description |
| --- | --- | --- | --- |
| `--profiling-path` | Yes | — | Single `kernel_details*.csv` file or profiling directory. |
| `--database-path` | No | auto-derived | Explicit output database directory. |
| `--device` | No | `ATLAS_800_A3_752T_128G_DIE` | Device directory name when deriving the path. |
| `--vllm-version` | No | — | vLLM version or full version-directory name. |
| `--torch-version` | No | — | PyTorch version when deriving the path. |
| `--cann-version` | No | — | CANN version when deriving the path. |

Example:

```bash
python tools/perf_data_collection/parsers/parse_kernel_details.py \
  --profiling-path ./PROF_001 \
  --database-path tensor_cast/performance_model/profiling_database/data/ATLAS_800_A3_752T_128G_DIE/vllm_ascend/vllm0.18.0_torch2.9.0_cann8.5
```

### 2. Theory shape expansion: `generate_shape_grid.py`

| Argument | Required | Default | Description |
| --- | --- | --- | --- |
| `--database-path` | No | auto-derived | Explicit CSV root directory. |
| `--target-models` | No | full grid | Comma-separated model names to prune GEMM `(N, K)` candidates. |
| `--device` | No | `ATLAS_800_A3_752T_128G_DIE` | Device directory name when deriving the path. |
| `--vllm-version` | No | — | vLLM version when deriving the path. |
| `--torch-version` | No | — | PyTorch version when deriving the path. |
| `--cann-version` | No | — | CANN version when deriving the path. |
| `--rows` | No | `1000` | Max appended rows per CSV; `0` means no cap. |
| `--seed` | No | — | Random sampling seed. |
| `--max-hbm-gb` | No | `32.0` | Per-row HBM budget in GiB; `0` disables filtering. |

### 3. Operator replay writeback: `start_microbench.py`

| Argument | Required | Default | Description |
| --- | --- | --- | --- |
| `--database-path` | No | auto-derived | Explicit database directory. |
| `--device` | No | `ATLAS_800_A3_752T_128G_DIE` | Device directory name when deriving the path. |
| `--vllm-version` | No | — | vLLM version when deriving the path. |
| `--torch-version` | No | — | PyTorch version when deriving the path. |
| `--cann-version` | No | — | CANN version when deriving the path. |
| `--ops` | No | all supported ops | Restrict replay to specific operator names. |
| `--dispatch-ffn-combine-ep-size` | No | `16` | EP size for `DispatchFFNCombine` replay and row matching. |
| `--repeat-count` | No | `1` | Replay repeat count forwarded to `run_all_op.py`. |
| `--update-mode` | No | `all` | `all` updates all matched rows; `missing-only` fills rows without valid durations. |
| `--fail-fast` | No | `false` | Stop immediately when one replay script fails. |
| `--prune-empty-duration-rows` | No | `false` | Delete rows whose replay/profiling durations remain invalid after writeback. |

Single-operator debug example:

```bash
python tools/perf_data_collection/start_microbench.py \
  --database-path tensor_cast/performance_model/profiling_database/data/ATLAS_800_A3_752T_128G_DIE/vllm_ascend/vllm0.18.0_torch2.9.0_cann8.5 \
  --ops MatMulV2 \
  --repeat-count 1 \
  --update-mode all
```

#### 3.1 Shared replay arguments

Most `op_replay/*_run.py` scripts share:

| Argument | Required | Default | Description |
| --- | --- | --- | --- |
| `--database-path` | No | auto-derived | Explicit database directory. |
| `--device` | No | `ATLAS_800_A3_752T_128G_DIE` | Device directory name. |
| `--vllm-version` | No | — | vLLM version or full version-directory name. |
| `--torch-version` | No | — | PyTorch version. |
| `--cann-version` | No | — | CANN version. |
| `--repeat-count` | No | `30` | Replay count per row; code default when omitted; **not** overridable via env var. |
| `--update-mode` | No | `all` | `all` or `missing-only`. |

Run one replay script directly:

```bash
python tools/perf_data_collection/op_replay/MatMulV2_run.py \
  --database-path tensor_cast/performance_model/profiling_database/data/ATLAS_800_A3_752T_128G_DIE/vllm_ascend/vllm0.18.0_torch2.9.0_cann8.5 \
  --repeat-count 10 \
  --update-mode missing-only
```

#### 3.2 Replay orchestrator: `op_replay/run_all_op.py`

| Argument | Required | Default | Description |
| --- | --- | --- | --- |
| `--database-path` | No | auto-derived | Explicit database directory. |
| `--ops` | No | all operators | Restrict to selected operators. |
| `--repeat-count` | No | not forwarded | Forwarded to each operator replay script when set. |
| `--update-mode` | No | `all` | Writeback mode. |
| `--execution-mode` | No | `subprocess` | `subprocess` or `inprocess`; `start_microbench.py` uses `inprocess`. |
| `--continue-on-error` | No | `false` | Continue after individual operator failures. |

#### 3.3 Per-operator replay entry points

| File | Kernel Type | Description |
| --- | --- | --- |
| `AddRmsNormBias_run.py` | `AddRmsNormBias` | Replay fused AddRmsNormBias. |
| `Add_run.py` | `Add` | Replay elementwise add. |
| `ArgMaxV2_run.py` | `ArgMaxV2` | Replay argmax. |
| `AscendQuantV2_run.py` | `AscendQuantV2` | Replay Ascend quantize. |
| `BatchMatMulV2_run.py` | `BatchMatMulV2` | Replay batch matmul. |
| `DispatchFFNCombine_run.py` | `DispatchFFNCombine` | Replay DFC fused operator. |
| `DynamicQuant_run.py` | `DynamicQuant` | Replay dynamic quant. |
| `FusedInferAttentionScore_run.py` | `FusedInferAttentionScore` | Replay FIA. |
| `GatherV2_run.py` | `GatherV2` | Replay gather/embedding. |
| `GroupedMatmul_run.py` | `GroupedMatmul` | Replay grouped matmul. |
| `GroupedMatmulSwigluQuant_run.py` | `GroupedMatmulSwigluQuant` | Replay grouped matmul + SwiGlu + quant. |
| `Index_run.py` | `Index` | Replay index. |
| `InterleaveRope_run.py` | `InterleaveRope` | Replay interleaved RoPE. |
| `KvRmsNormRopeCache_run.py` | `KvRmsNormRopeCache` | Replay KV RMSNorm RoPE cache. |
| `LightningIndexer_run.py` | `LightningIndexer` | Replay LightningIndexer. |
| `MaskedFill_run.py` | `MaskedFill` | Replay masked fill. |
| `MatMulCommon_run.py` | `MatMulCommon` | Replay generic matmul. |
| `MatMulV2_run.py` | `MatMulV2` | Replay MatMulV2. |
| `MatMulV3_run.py` | `MatMulV3` | Replay MatMulV3. |
| `MoeTokenPermute_run.py` | `MoeTokenPermute` | Replay MoE token permute. |
| `MoeTokenUnpermute_run.py` | `MoeTokenUnpermute` | Replay MoE token unpermute. |
| `PadV3_run.py` | `PadV3` | Replay pad. |
| `QuantBatchMatmulV3_run.py` | `QuantBatchMatmulV3` | Replay quant batch matmul. |
| `RINGMLAPrefillBF16Kernel_run.py` | `RINGMLAPrefillBF16Kernel` | Replay MLA prefill kernel. |
| `ReshapeAndCacheNdKernel_run.py` | `ReshapeAndCacheNdKernel` | Replay reshape-and-cache. |
| `RmsNorm_run.py` | `RmsNorm` | Replay RMSNorm. |
| `ScatterNdUpdate_run.py` | `ScatterNdUpdate` | Replay scatter update. |
| `Slice_run.py` | `Slice` | Replay slice. |
| `SoftmaxV2_run.py` | `SoftmaxV2` | Replay softmax. |
| `Sort_run.py` | `Sort` | Replay sort. |
| `SparseFlashAttention_run.py` | `SparseFlashAttention` | Replay sparse flash attention. |
| `SwiGlu_run.py` | `SwiGlu` | Replay SwiGlu. |
| `TensorMove_run.py` | `TensorMove` | Replay tensor copy. |
| `TransposeBatchMatMul_run.py` | `TransposeBatchMatMul` | Replay transpose batch matmul. |
| `Transpose_run.py` | `Transpose` | Replay transpose. |
| `split_qkv_rmsnorm_rope_kernel_run.py` | `split_qkv_rmsnorm_rope_kernel` | Replay custom QKV/RMSNorm/RoPE fused kernel. |

### 4. HCCL communication microbench: `comm_bench/generate_comm_microbench.py`

| Argument | Required | Default | Description |
| --- | --- | --- | --- |
| `--database-path` | No | no write | Per-op communication CSV output directory. |
| `--ops` | No | all comm ops | `all_reduce`, `all_gather`, `reduce_scatter`, `all_to_all`. |
| `--num-devices` | No | `16` | Devices per communication group; multiple values allowed. |
| `--topology-tier` | No | auto-derived | Topology tier: `0` inter-pod, `1` intra-pod, `2` die-level. |
| `--grid-shape` | No | `48 8 2` | Hardware grid; default matches **`ATLAS_800_A3_752T_128G_DIE`**; A3 ROCE variants may use `2 8 2`, etc. |
| `--dtype` | No | `torch.bfloat16` | Communication tensor dtype. |
| `--bytes-grid` | No | built-in grid | Custom `message_bytes` list. |
| `--output-csv` | No | — | Single CSV output path; only valid with one `--ops` value. |
| `--bench-mode` | No | `kernel` | `kernel` uses profiler stats; `event` uses NPU event timing. |

Direct collection example (`ATLAS_800_A3_752T_128G_DIE`, `grid-shape 48 8 2`):

```bash
torchrun --nproc_per_node=16 \
  tools/perf_data_collection/comm_bench/generate_comm_microbench.py \
  --database-path ./hccl_data \
  --ops all_reduce all_gather reduce_scatter all_to_all \
  --grid-shape 48 8 2 \
  --num-devices 16 2
```

### 5. Batch comm collection: `comm_bench/run_comm_bench.sh`

Wrapper for **`ATLAS_800_A3_752T_128G_DIE`** around `generate_comm_microbench.py`.

- Input: output directory and optional multi-node environment variables.
- Output: `hcom_*.csv` collected with 752T DIE defaults (`grid-shape 48 8 2`).
- Default output directory: `./hccl_bench_data/`.

```bash
bash tools/perf_data_collection/comm_bench/run_comm_bench.sh ./hccl_bench_data
```

Multi-node inter-pod collection (different `NODE_RANK` on each node):

```bash
NNODES=2 NODE_RANK=0 MASTER_ADDR=<master_ip> \
  bash tools/perf_data_collection/comm_bench/run_comm_bench.sh ./hccl_inter_pod

NNODES=2 NODE_RANK=1 MASTER_ADDR=<master_ip> \
  bash tools/perf_data_collection/comm_bench/run_comm_bench.sh ./hccl_inter_pod
```

## Environment Variables

Variables commonly used under `tools/perf_data_collection/`. Replay repeat count is configured **only** via the `--repeat-count` CLI flag; **`MSMODELING_OP_REPLAY_REPEAT_COUNT` is no longer supported**.

| Variable | Default | Description |
| --- | --- | --- |
| `VLLM_ASCEND_PATH` | sibling `../vllm-ascend` | vllm-ascend repo root for custom Triton kernels. |
| `ASCEND_CUSTOM_OPP_PATH` | — | Required for custom OPP operator replay; see `start_microbench.py` module doc. |
| `LD_LIBRARY_PATH` | — | Custom OPP `op_api/lib` lookup; some ops require it with `ASCEND_CUSTOM_OPP_PATH`. |
| `ASCEND_HOME_PATH` / `ASCEND_TOOLKIT_HOME` / `ASCEND_TOOLKIT_HOME_PATH` / `ASCEND_INSTALL_PATH` | auto-detected | CANN install root and version detection. |
| `MASTER_ADDR` / `MASTER_PORT` / `RANK` / `WORLD_SIZE` / `LOCAL_RANK` | injected by `torchrun` | Distributed launch for comm bench / DFC replay. |

Full cross-module list: [Environment Variables](../../tests/README.md#environment-variables).

Custom OPP example:

```bash
export ASCEND_CUSTOM_OPP_PATH=/path/to/vllm_ascend/_cann_ops_custom/vendors/vllm-ascend:${ASCEND_CUSTOM_OPP_PATH}
export LD_LIBRARY_PATH=/path/to/vllm_ascend/_cann_ops_custom/vendors/vllm-ascend/op_api/lib:${LD_LIBRARY_PATH}
```

## Output Files

| Tool | Main output |
| --- | --- |
| `parse_kernel_details.py` | `{KernelType}.csv` under the target database directory. |
| `generate_shape_grid.py` | theory shape rows appended to existing `{KernelType}.csv`. |
| `start_microbench.py` | updated `Average Duration(us)`, `MicroBench ...` columns, and reports under `reports/`. |
| `generate_comm_microbench.py` | `hcom_allReduce_.csv`, `hcom_allGather_.csv`, `hcom_reduceScatter_.csv`, `hcom_alltoallv_.csv`. |
| `run_all_op.py` | `op_replay/run_all_op_status.json` for `start_microbench.py` aggregation. |

Report directory under the target database path:

```text
<database-path>/reports/
  profile_update_report_<timestamp>.md
  duration_gap_hotspots_full_<timestamp>.csv
```

## `start_microbench.py` Update Modes

- `all`: update every matched row and append unmatched profiling samples into the target CSV.
- `missing-only`: replay and fill only rows whose `Average Duration(us)` and `Profiling Average Duration(us)` are both invalid (`0` or empty); skip rows that already have at least one valid duration.

## Empty-row Pruning

- Default: rows with both duration columns invalid are kept so `missing-only` can retry later.
- `--prune-empty-duration-rows`: opt-in cleanup after writeback; use only when you intentionally delete unrecoverable empty rows.

## FAQ

### Database directory not found

Prefer an explicit `--database-path`. When deriving the path, ensure `--device`, `--vllm-version`, `--torch-version`, and `--cann-version` match the on-disk layout.

### `msprof not found`

Ascend toolkit environment is not active. Load CANN first, then run `start_microbench.py`.

### Custom OPP environment missing

Some custom operators require `ASCEND_CUSTOM_OPP_PATH` and `LD_LIBRARY_PATH`. See [Environment Variables](#environment-variables).

### `missing-only` did not run replay

Rows with valid `Average Duration(us)` or `Profiling Average Duration(us)` are skipped. Use `--update-mode all` to force refresh.

### Communication collection wrote no CSV

Ensure `--database-path` is set and rank 0 can write to the target directory. For `run_comm_bench.sh`, the first positional argument is the output directory (default `./hccl_bench_data/`).

### Multi-node communication collection hangs

Run the script on all nodes simultaneously with consistent `NNODES`, `MASTER_ADDR`, `MASTER_PORT`, and `NPROC`; only `NODE_RANK` should differ per node.

## Documentation Maintenance

When editing either guide, keep both `readme.md` and `readme_en.md` in sync:

- Argument names match argparse definitions in code.
- Defaults match code defaults.
- Example commands are copy-paste ready from the repository root (after `source try/bin/activate`).
- Document NPU, CANN, `torchrun`, or custom OPP requirements for new tools.
