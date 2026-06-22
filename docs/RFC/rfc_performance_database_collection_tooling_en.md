# RFC: Standalone Performance Database Collection Tooling

## Metadata

| Item | Content |
| :--- | :--- |
| **Status** | Complete |
| **Author(s)** | Horacehxw, Codex |
| **Creation Date** | 2026-05-12 |
| **Related Links** | <https://gitcode.com/Ascend/msmodeling/pull/124> |

---

## 1. Overview

The performance database can only be trusted by runtime performance models when its CSV data, operator mappings, replay scripts, communication benchmarks, and validation reports come from a repeatable offline flow. This RFC defines `tools/perf_data_collection/` as the offline data-production toolchain, covering raw profiling parsing, FIA runtime metadata backfill, theory shape-grid generation, per-kernel replay, `msprof` writeback, HCCL communication benchmarking, and basic trace conversion.

The produced database is consumed by `EmpiricalPerformanceModel` / `ProfilingDataSource`: runtime only needs to read `op_mapping.yaml`, per-kernel compute CSVs, and optional HCCL communication CSVs. This RFC fully defines how those data artifacts are generated, validated, and promoted into a reviewable performance database, without requiring any other design document to understand the database generation flow.

This RFC defines the standalone collection and tooling boundary for the performance database: how raw NPU profiling results become versioned CSV data, how synthetic shapes and microbench measurements are added, how HCCL communication data is collected, and how data is validated before runtime query code consumes it.

### 1.1 Goals

- Provide an offline data-production contract that does not depend on temporary design or planning documents.
- Define the database directory structure, compute CSV schema, FIA-specific columns, communication CSV schema, and versioning rules.
- Clarify the boundary between `op_mapping.yaml`, raw profiling parsing, shape-grid generation, operator replay scripts, `msprof` orchestration, and CSV writeback.
- Record tooling boundaries and limitations so contributors know which steps are automated and which steps remain manually maintained.
- Give human reviewers a shared checklist for deciding whether CSV/YAML/report changes can enter a production database directory.
- Provide a rollout and test plan where most checks run without NPU hardware, while replay/HCCL checks remain isolated to hardware hosts.

### 1.2 Non-goals

- This RFC does not change `ProfilingDataSource` runtime query behavior or interpolation logic.
- This RFC does not define model-level alignment metrics or end-to-end simulation strategy; it only requires the produced data to be consumable by those systems.
- This RFC does not require an external database service, remote artifact repository, or online collection pipeline.
- This RFC does not claim replay coverage for every profiling kernel.
- This RFC does not make temporary planning documents stable interfaces.

## 2. Solution Design

### 2.1 Recommended Solution

Define the standard offline collection pipeline as:

```mermaid
flowchart TD
    A[Raw NPU profiling bundle] --> B[parsers/parse_kernel_details.py]
    B --> C[Per-kernel compute CSVs]
    C --> D[fill_fia_runtime_metadata.py]
    C --> E[generate_shape_grid.py]
    D --> E
    E --> F[op_replay/*_run.py]
    F --> G[start_microbench.py under msprof]
    G --> H[CSV writeback and reports]
    I[HCCL benchmark] --> J[hccl/{cann_version}/hcom_*.csv]
    J --> H
    K[op_mapping.yaml] --> F
    K --> H
    H --> L[Versioned profiling_database/data directory]
```

The flow remains file-based. CSV and YAML are reviewable and diffable, and they remain compatible with the runtime data source. Tools may generate or rewrite rows, but persistent data artifacts stay in versioned database directories.

#### 2.1.0 Tool Scope

This RFC defines the entry points and responsibilities under `tools/perf_data_collection/`:

| Scope | Entry point | Database responsibility |
| --- | --- | --- |
| Raw profiling parsing | `parsers/parse_kernel_details.py` | Generate per-kernel compute CSVs from `kernel_details*.csv` or a profiling directory |
| FIA runtime metadata backfill | `fill_fia_runtime_metadata.py` | Merge FIA JSONL runtime values into `FusedInferAttentionScore.csv` |
| Shape-grid expansion | `generate_shape_grid.py`, `grid_generator/`, `memory_estimator.py` | Append replayable theory shape rows and filter them by HBM budget |
| Compute operator replay | `op_replay/*_run.py`, `op_replay/run_all_op.py`, `start_microbench.py` | Replay CSV rows on an NPU host and write microbench latency back |
| HCCL communication collection | `comm_bench/generate_comm_microbench.py`, `comm_bench/run_comm_bench.sh`, `comm_bench/validate_comm_alignment.py` | Generate or validate `hccl/{cann_version}/hcom_*.csv` |
| Trace conversion support | `parsers/trace_to_csv.py` | Convert TensorCast Chrome traces to CSV for offline alignment and reports; it does not directly produce operator database CSVs |

#### 2.1.1 Database Directory Contract

The database has two persistent data classes:

| Data type | Directory | Content | Version scope |
| --- | --- | --- | --- |
| Compute/operator data | `tensor_cast/performance_model/profiling_database/data/{device}/vllm_ascend/{version_dir}/` | `op_mapping.yaml` and `{KernelType}.csv` | Device + vLLM-Ascend + PyTorch + CANN stack |
| Communication data | `tensor_cast/performance_model/profiling_database/data/{device}/hccl/{cann_version}/` | `hcom_*.csv` benchmark files, with optional topology/config metadata | Device + CANN/HCCL stack |

`{version_dir}` follows the helper convention:

```text
vllm{vllm_version}_torch{torch_version}_cann{cann_version}
```

Recommended directory examples include:

- `vllm0.13.0_torch2.8.0_cann8.3`
- `vllm0.15.0_torch2.9.0_cann8.5`
- `vllm0.18.0_torch2.9.0_cann8.5`
- `vllm0.18.0_torch2.9.0_cann8.5_shape_generated`, a generated/staging example that should not be mistaken for a complete measured directory.

Each compute directory owns one `op_mapping.yaml` for that stack. Communication CSVs are not copied into every compute directory; `op_mapping.yaml` points to the communication directory through `communication_data_ref`, for example:

```yaml
communication_data_ref: "../../hccl/v8.5/"
communication_fallback: analytic
```

This allows multiple vLLM-Ascend versions to reuse HCCL measurements when the device and CANN/HCCL stack are unchanged.

#### 2.1.2 Compute CSV Contract

Each compute CSV is named from the NPU profiling `Type` column after necessary normalization:

```text
{KernelType}.csv
```

The parser and runtime data source use a CSV contract based on shapes and metadata rather than a Python object schema. The base required columns are:

| Column group | Columns | Producer |
| --- | --- | --- |
| Kernel identity and shape | `OP State`, `Accelerator Core`, `Input Shapes`, `Input Data Types`, `Input Formats`, `Output Shapes`, `Output Data Types`, `Output Formats` | Raw profiling parser |
| Profiling latency | `Profiling Average Duration(us)`, `Profiling Median Duration(us)`, `Profiling Std Duration(us)` | Raw profiling parser |
| Profiling counters | `Profiling Average aicore_time(us)`, `Profiling Average aic_total_cycles`, `Profiling Average aic_mac_time(us)`, and related AIC/AIV utilization columns | Raw profiling parser |
| Microbench latency | `Average Duration(us)` | `start_microbench.py` writeback |
| Microbench counters | `MicroBench aicore_time(us)`, `MicroBench aic_total_cycles`, and corresponding `MicroBench ...` counter columns | `start_microbench.py` writeback |

Shape fields use semicolons to separate tensor slots and commas to separate dimensions:

```text
"136,7168;7168,3584"
```

Key rules:

- Slot count has semantics. Empty slots must be preserved for FIA, custom kernels, and operators with optional inputs or scalar parameters.
- `Input Data Types` and `Input Formats` align with `Input Shapes` by slot index.
- `FRACTAL_NZ` is a valid format; replay and runtime lookup normalize it when needed.
- Generated rows may initially have no latency or a zero latency. A row should be considered measured only after replay/writeback fills `Average Duration(us)` or profiling import fills `Profiling Average Duration(us)`.
- Runtime lookup selects latency columns in this order: `Average Duration(us)`, `Profiling Average Duration(us)`, then `Duration(us)`.

#### 2.1.3 FIA CSV Extension Contract

`FusedInferAttentionScore.csv` needs additional runtime metadata because shape alone cannot fully describe paged attention behavior. The parser and backfill tool use the following columns when available:

| Column | Meaning |
| --- | --- |
| `Runtime source_profile` | Source profiling subdirectory or source tag |
| `Runtime actual_seq_lengths_shape` / `Runtime actual_seq_lengths_values` | Query sequence-length metadata |
| `Runtime actual_seq_lengths_kv_shape` / `Runtime actual_seq_lengths_kv_values` | KV sequence-length metadata |
| `Runtime avg_seq_len` | Average KV sequence length used by attention lookup and interpolation |
| `Runtime block_table_shape` / `Runtime block_table_valid_blocks` | Paged cache block-table metadata |
| `Runtime num_heads` / `Runtime num_key_value_heads` | Runtime head configuration |
| `Runtime sparse_mode`, `Runtime input_layout`, `Runtime block_size` | FIA execution-mode fields |
| `Runtime attn_state`, `Runtime kv_cache_mode` | Optional state/cache-mode fields |
| `Runtime metadata_completeness` | Completeness tag, for example `profile_shapes_only`, `runtime_values`, `runtime_values_dumped`, or `shape_grid_scene_generated` |

The parser can recover FIA shape-level metadata from profiling bundles. Full runtime values require a JSONL dump plus `fill_fia_runtime_metadata.py`. When those values are missing, the replay script falls back to heuristic inference; such rows remain useful, but they should not be treated as fully equivalent reproductions of whole-model runtime state.

#### 2.1.4 Communication CSV Contract

Communication data is stored under the HCCL directory, one CSV per HCCL kernel. Standard filenames are:

- `hcom_allGather_.csv`
- `hcom_allReduce_.csv`
- `hcom_alltoallv_.csv`
- `hcom_reduceScatter_.csv`

Communication CSV columns are split between runtime-required columns and collection-standard columns. `ProfilingDataSource` runtime lookup requires `message_bytes`, `num_devices`, and one usable latency column. `topology_tier` participates in matching only when the CSV contains it and `DeviceProfile.comm_grid` can resolve it. `dtype` and `bandwidth_gbps` are collection-standard columns for audit and validation; lookup does not filter by dtype. `communication_fallback` records the intended fallback policy; actual MISS fallback is handled by the upper empirical model.

| Column | Level | Type | Meaning |
| --- | --- | --- | --- |
| `message_bytes` | Runtime-required | int | Per-rank message size used for lookup and interpolation |
| `num_devices` | Runtime-required | int | Number of ranks in the communication group |
| `Average Duration(us)` / `Profiling Average Duration(us)` / `Duration(us)` | One runtime-required latency column | float | Communication latency; runtime selects by the common latency-column priority |
| `topology_tier` | Conditional match | int | Device topology tier; used when topology can be resolved |
| `dtype` | Collection-standard | string | Measured dtype, such as `DT_BF16` or `INT8` |
| `bandwidth_gbps` | Collection-standard | float | Derived bandwidth for audit and validation |

`ProfilingDataSource` first searches for a communication row that exactly matches `message_bytes` and `num_devices`; when `topology_tier` can be resolved and the CSV contains that column, `topology_tier` must also match. On miss, it finds bracket rows within the same `num_devices` and available topology tier, fits an alpha-beta latency model from those matching rows, and clamps the predicted latency to the bracket latency bounds. If `DeviceProfile.comm_grid` is absent, topology filtering is skipped; in that case, the CSV should avoid ambiguous rows for the same message/device count.

#### 2.1.5 Relationship Between `op_mapping.yaml` and Tooling

`op_mapping.yaml` is both the runtime query contract and a tooling input. It maps TensorCast operators to NPU profiling kernel types and records the evidence chain for each mapping.

Important top-level fields:

| Field | Purpose |
| --- | --- |
| `version`, `device`, `cann_version`, `pytorch_version`, `op_plugin_version`, `collection_date` | Identify the software stack and collection batch |
| `communication_data_ref` | Relative path to HCCL communication CSV data |
| `communication_fallback` | Intended fallback policy; actual MISS fallback is handled by the upper empirical model |
| `interpolation_policy` | Optional interpolation behavior, such as sqrt transform for FIA |
| `operator_mappings` | Runtime mapping from TensorCast op name to kernel query rule |
| `torch_npu_reference` | Reference API metadata that can help create replay scripts |

Important `operator_mappings` fields:

| Field | Meaning |
| --- | --- |
| `kernel_type` | Primary `{KernelType}.csv` file stem |
| `alternate_kernel_types` | Candidate CSVs for version drift or kernel variants |
| `category: communication` | Route the query to communication lookup |
| `query_mode` | Use special query logic, such as `attention_special`, `elementwise`, or `moe_fused` |
| `composite`, `sub_kernels`, `decomposer` | Model one TensorCast op as multiple NPU kernels |
| `tc_input_count` | Truncate TensorCast inputs for matching when TC and NPU signatures differ |
| `zero_cost` | Mark shape-only or fusion-absorbed ops as measured zero latency |
| `accepted_miss` | Record expected misses so they do not block validation |
| `notes` | Evidence chain and review context |

The first phase does not require automatically consuming `torch_npu_reference.{KernelType}.microbench_api` to generate `op_replay/<KernelType>_run.py`. Replay scripts are maintained manually. When adding or updating a replay script, contributors should use the Microbench Run Script Generator workflow in `docs/perf_database/skills/microbench/SKILL.md`: generate or update the script from the operator CSV, `op_mapping.yaml` `microbench_api`, upstream interface docs/tests, and the local replay conventions. A future generator can build on this workflow, but this RFC treats `torch_npu_reference` as auxiliary metadata rather than first-phase automation.

#### 2.1.6 Raw Profiling Parser

`tools/perf_data_collection/parsers/parse_kernel_details.py` is the entry point for converting Ascend profiling output into per-kernel CSV files.

Parser behavior:

- `--profiling-path` may be a single `kernel_details*.csv` file or a profiling directory.
- Directory input is scanned recursively for CSV filenames containing `kernel_details`.
- `operator_details.csv` and `trace_view.json` are discovered for FIA bundle inspection and later enrichment.
- Kernel details must contain `Type`, `OP State`, `Accelerator Core`, shape/dtype/format columns, `Duration(us)`, and AIC/AIV counter columns.
- Known variant names are normalized, including `split_qkv_rmsnorm_rope_kernel_0` to `split_qkv_rmsnorm_rope_kernel`, and `muls_add_kernel_1` to `muls_add_kernel`.
- Rows are aggregated by `(normalized Type, Input Shapes, Output Shapes)`.
- Output includes average, median, and standard-deviation latency columns, plus averaged hardware counters.
- Each operator type is written as `{KernelType}.csv` under the target database directory.

The parser is conservative: it does not infer missing tensor slots and does not rewrite API semantics. It preserves the profiling representation and leaves shape/API alignment to `op_mapping.yaml`, `generate_shape_grid.py`, and replay scripts.

#### 2.1.7 FIA Runtime Metadata Enrichment

FIA enrichment has two layers:

1. `parsers/parse_kernel_details.py` can attach shape-level metadata from a profiling bundle and mark rows as `Runtime metadata_completeness=profile_shapes_only`.
2. `tools/perf_data_collection/fill_fia_runtime_metadata.py` can merge runtime JSONL dumps into `FusedInferAttentionScore.csv`.

JSONL backfill uses LEFT JOIN semantics:

- CSV key: query/key/value shapes, KV sequence item count, attention mask shape, and block-table shape.
- JSONL key: corresponding runtime tensor metadata.
- If multiple runtime records match one CSV row, the row expands to 1:N.
- Matched rows receive a configurable metadata tag; the default is `runtime_values_dumped`.

The minimum JSONL join fields are `query_shape`, `key_shape`, `value_shape`, `actual_seq_lengths_kv`, `atten_mask_shape`, and `block_table_shape`. Optional payload fields include `actual_seq_lengths`, `block_table_valid_blocks`, `num_heads`, `num_key_value_heads`, `sparse_mode`, `input_layout`, and `block_size`. Each line is one JSON object, for example:

```json
{"query_shape":[8192,16,128],"key_shape":[8192,16,128],"value_shape":[8192,16,128],"actual_seq_lengths_kv":[40,998],"atten_mask_shape":null,"block_table_shape":null,"num_heads":16,"num_key_value_heads":2,"sparse_mode":0,"input_layout":"TND","block_size":128}
```

This design makes metadata incompleteness explicit. Shape-only FIA rows remain usable, but validation reports should treat `profile_shapes_only` rows as lower confidence than rows with real runtime sequence values.

#### 2.1.8 Shape Grid Generation

`tools/perf_data_collection/generate_shape_grid.py` appends theory-generated rows to database CSV files. It does not replace real profiling data and does not produce measured latency by itself.

Shape-grid behavior:

- `grid_generator/config.yaml` routes kernel types to theory patterns.
- Template patterns cover GEMM, quantized matmul, elementwise, norm, quantization, RoPE, sampling, KV cache, and shape-manipulation categories.
- Fused attention, grouped matmul, and `DispatchFFNCombine` use more complex Python generators.
- `--target-models` uses known model configurations to prune GEMM `(N, K)` candidates.
- `--rows` caps rows per CSV; `--seed` makes sampling reproducible.
- `--max-hbm-gb` filters generated rows whose estimated input/output tensors exceed the memory budget.
- Files without a matching theory generator are skipped.

Generated rows should inherit stable structural metadata from source rows and leave performance values blank or zero. They become production data only after `start_microbench.py` or a later profiling import fills latency.

#### 2.1.9 Per-kernel `op_replay` Script Framework

`tools/perf_data_collection/op_replay/` stores per-kernel replay implementations. Each covered kernel should provide a matching `*_run.py` script for categories such as matmul, quantization, FIA, KV cache, softmax, sort, transpose, and vLLM-Ascend custom kernels.

Shared conventions:

- `common.py` handles database path resolution, software-stack version directory naming, dtype/format parsing, tensor construction, repeat-count handling, and invalid-row tracking.
- `replay_framework.py` provides an `OpReplay` helper for standard API-style kernels.
- `run_all_op.py` discovers `*_run.py`, supports `--execution-mode inprocess` for one outer `msprof` session, and writes `run_all_op_status.json`.
- Each operator script reads the matching `{KernelType}.csv`, replays each row on NPU, prints concise `[OK]` messages, and deletes invalid rows when replay case construction fails.
- When `--repeat-count` is omitted, `common.py` applies the built-in default (`30`); repeat count is CLI-only and `MSMODELING_OP_REPLAY_REPEAT_COUNT` is no longer supported.
- When adding or modifying `<KernelType>_run.py`, contributors should follow the Microbench Run Script Generator skill's core steps: read the target CSV, locate `torch_npu_reference.<KernelType>.microbench_api`, confirm the real API from upstream repo docs/tests, infer missing non-tensor arguments, reuse `common.py` / `replay_framework.py` conventions, and validate at least `py_compile` plus `--help`.

Coverage limitation: replay coverage can expand incrementally. A CSV may exist without a corresponding replay script, and some scripts only support their target recorded shape/API patterns. Custom operators also require correct `ASCEND_CUSTOM_OPP_PATH` and `LD_LIBRARY_PATH`. Regular CI coverage for NPU-dependent replay is necessarily limited; most tests cover imports, CLI help, pure parsing logic, and writeback units.

#### 2.1.10 `start_microbench.py` `msprof` Orchestration and Writeback

`tools/perf_data_collection/start_microbench.py` is the production entry point for replay measurement writeback.

Flow:

1. Resolve the target database directory through `--database-path` or device/version parameters.
2. Validate selected operators and required custom OPP environment variables.
3. Run `msprof --output=... python op_replay/run_all_op.py --execution-mode inprocess`.
4. Locate generated `PROF_*` directories and `op_summary_*.csv` files.
5. Aggregate rows by operator type and signature.
6. Write `Average Duration(us)` and `MicroBench ...` counter columns back to matching database CSV rows.
7. Generate a markdown update report and duration-gap CSV under `reports/`.

Row matching uses:

```text
Input Shapes
Input Data Types
Input Formats
Output Shapes
Output Data Types
```

`DispatchFFNCombine` also includes `EP Size` in the signature. Update modes:

| Mode | Behavior |
| --- | --- |
| `all` | Update all matching rows and append unmatched profiling samples |
| `missing-only` | Replay/fill only rows whose microbench and profiling latencies are invalid; unmatched samples are reported but not appended |

`--prof-path` can parse an existing `PROF_*` directory without launching `msprof`. `--prune-empty-duration-rows` deletes rows that still have no valid replay or profiling latency after writeback.

Timing caveat: `Average Duration(us)` is the microbench value aggregated from `op_summary_*.csv` task duration. It is the runtime lookup's preferred latency, but comparison reports should still keep `Profiling Average Duration(us)` to expose replay versus whole-model profiling gaps.

#### 2.1.11 HCCL Communication Benchmark

Communication collection is separate from compute replay because HCCL latency depends on rank-group size and topology.

`tools/perf_data_collection/comm_bench/generate_comm_microbench.py` supports:

- Operators: all-reduce, all-gather, reduce-scatter, all-to-all.
- `--bytes-grid` for the message-bytes grid.
- `--num-devices`.
- `--topology-tier`.
- `--grid-shape` for hardware topology.
- dtype selection.
- `--bench-mode kernel` or `--bench-mode event`.
- `--do-run` for direct execution under `torchrun`.
- `--output-dir` for per-op files, or `--output-csv` for a single file.

`tools/perf_data_collection/comm_bench/run_comm_bench.sh` is the batch wrapper. `validate_comm_alignment.py` checks whether measured rows align with the parser/query model under a configurable ratio tolerance.

Limitations:

- The query convention uses lowercase `hcom_` prefixes, for example `hcom_reduceScatter_.csv`; CamelCase graph-compiled names are version-drift clues but not primary HCCL CSV filenames.
- Communication dtype is recorded in CSV, but the lookup path primarily matches message size, device count, and topology tier.
- If topology tier cannot be resolved from `DeviceProfile`, communication data must be organized to avoid ambiguity by directory and row contents.

#### 2.1.12 Validation and Reports

Validation should cover three levels:

| Level | Checks | Representative command |
| --- | --- | --- |
| Static/tooling | Python syntax, CLI help, schema unit tests, shape parser tests, HCCL benchmark pure-logic tests | `pytest tests/tools/test_op_replay.py tests/tools/test_op_replay_common.py tests/tools/test_start_microbench.py tests/tools/test_generate_shape_grid.py tests/tools/test_fia_parser_backfill.py tests/tools/test_generate_comm_microbench.py tests/tools/test_validate_comm_alignment.py` |
| Database/query | `op_mapping.yaml` schema, `ProfilingDataSource` lookup, interpolation, FIA enriched lookup | `pytest tests/perf_database` |
| Hardware-dependent | NPU replay, HCCL benchmark, writeback from real `msprof` output | Run `pytest -m npu` and selected `start_microbench.py` / `comm_bench` commands on an NPU host |

Each data refresh should produce:

- List of created or updated CSV files.
- Replay status summary from `run_all_op_status.json`.
- Profile update report from `start_microbench.py`.
- Duration-gap hotspot CSV when both replay and whole-model profiling latency exist.
- List of skipped operators or missing replay scripts.
- List of generated rows that still lack valid latency.

Validation must not hide partial coverage. Accepted miss, zero-cost ops, interpolation, partial composite hit, and FIA metadata completeness must remain visible in reports or CSV metadata.

#### 2.1.13 Tool CLI Contract

| Tool | Required input | Main output | Key failure modes |
| --- | --- | --- | --- |
| `parsers/parse_kernel_details.py` | `--profiling-path`; output directory from `--database-path` or device/version parameters | Per-kernel `{KernelType}.csv` | Fails when input lacks required profiling columns |
| `fill_fia_runtime_metadata.py` | `--csv-path`, `--jsonl-path` | Enriched `FusedInferAttentionScore.csv`; overwrites in place by default, or uses `--output-path` | Fails when CSV/JSONL is missing, CSV header is empty, or JSONL is invalid |
| `generate_shape_grid.py` | `--data-dir`, or `--device --vllm-version [--torch-version --cann-version]` | Appends theory shape rows to CSVs | Fails when data dir is missing or contains no CSV; CSVs without a generator are skipped and reported |
| `start_microbench.py` | `--database-path`, or device/version parameters; optional `--prof-path` | Writes `Average Duration(us)` and `MicroBench ...` counters, generates `reports/` | `msprof` missing, OPP env missing, no `PROF_*` / `op_summary_*.csv`; duplicate signatures, unmatched rows, and gaps enter the report |
| `op_replay/run_all_op.py` | Database path parameters; optional `--op` | Executes matching `*_run.py` scripts and writes `run_all_op_status.json` | Single-op failure; default is fail-fast, `--continue-on-error` runs remaining scripts |
| `comm_bench/generate_comm_microbench.py` | `torchrun ... --do-run` | `hcom_*.csv` | Without `--do-run`, no collection is executed; `--output-csv` allows only one `--ops` value |
| `comm_bench/validate_comm_alignment.py` | `--csv-dir` | PASS/WARN/FAIL alignment report; nonzero exit on FAIL | Fails when `--csv-dir` is not a directory; ratios beyond tolerance fail; malformed rows are not counted, so reviewers should check 0-row reports |

#### 2.1.14 Human Review Checklist

Before a data refresh or tooling change is sent for human review, it should provide at least:

- Target software stack: device, vLLM-Ascend, PyTorch, CANN, op-plugin, and collection date.
- Raw input sources: profiling bundle name or path summary, whether `operator_details.csv` / `trace_view.json` exist, and whether FIA runtime JSONL exists.
- CSV schema changes: added, removed, or renamed columns; whether empty tensor slots are preserved; whether latency-column priority is still satisfied.
- `op_mapping.yaml` changes: added/modified mappings, evidence chains, `alternate_kernel_types`, `tc_input_count`, `accepted_miss`, `zero_cost`, and composite/decomposer impact.
- Shape-grid changes: `--target-models`, `--rows`, `--seed`, `--max-hbm-gb`, and whether generated-only rows have been replayed.
- Replay results: `run_all_op_status.json` summary, profile update report, duration-gap hotspot CSV, invalid rows, and duplicate signatures.
- HCCL results: `hcom_*.csv` file list, `num_devices` / `topology_tier` coverage, `bench-mode`, `validate_comm_alignment.py` tolerance, and result.
- Known limitations: missing replay scripts, incomplete FIA metadata, staging-only generated directories, and reasons hardware-dependent tests were not run.

#### 2.1.15 Data Lifecycle and Version Management

Recommended lifecycle:

1. Create or select the compute directory for a software stack.
2. Import raw profiling with `parsers/parse_kernel_details.py`.
3. Add or update `op_mapping.yaml` in the same software-stack directory.
4. Enrich FIA metadata if runtime JSONL values are available.
5. Generate shape rows in a staging copy or clearly named generated directory.
6. First replay selected operators with `start_microbench.py --update-mode missing-only`.
7. Review writeback reports, missing shapes, invalid rows, duplicate signatures, and duration gaps.
8. Submit CSV/YAML changes together with changelog or reports, then promote the directory.

Versioning rules:

- Isolate software-stack changes by `{version_dir}`. Do not silently mix data from different vLLM-Ascend, PyTorch, or CANN versions.
- Store HCCL data under `{device}/hccl/{cann_version}` and reuse it through `communication_data_ref`.
- Update `collection_date` when a database directory receives a new profiling import or a large replay refresh.
- Keep published version directories unless there is an explicit deprecation notice.
- Use clear names for generated-only directories so users do not confuse them with complete measured data.
- Do not commit raw `PROF_*` directories unless review explicitly requires them; commit derived CSVs and report artifacts.

### 2.2 Alternative Solutions

#### Option A: Use Only Whole-model Raw Profiling

Import only whole-model profiling rows, without shape-grid generation or microbenchmark replay.

This is simple and closest to real workloads, but shape coverage is sparse. It cannot fill important neighboring shapes and makes runtime lookup overly dependent on one captured model/profile.

#### Option B: Generate All Replay Scripts from `torch_npu_reference`

Read `op_mapping.yaml` and automatically generate all `op_replay/<KernelType>_run.py` files from `torch_npu_reference.{KernelType}.microbench_api`.

This is a better long-term direction, but it is outside the first-phase scope. Many kernels need non-tensor parameters, custom OPP settings, preserved slots, valid cache tensors, or version-dependent API behavior that cannot be safely inferred from a single API string.

#### Option C: Use an External Database

Move CSV/YAML data to SQLite, DuckDB, or a service database.

This would improve indexing and schema validation, but it would add deployment and review cost in the first phase. The recommended solution keeps the runtime data source and review flow centered on file artifacts.

#### Option D: Replace Measurement with Analytic Models

Avoid replay collection and use analytic formulas for both compute and communication.

Analytic models are appropriate fallbacks, especially for communication, but they cannot accurately capture fusion, CANN kernel selection, tensor formats, and custom-op behavior.

### 2.3 Solution Analysis

#### Advantages

- Keeps data production offline, repeatable, and reviewable.
- Adopts the runtime contract: `op_mapping.yaml` plus per-kernel CSV files.
- Splits raw profiling import, metadata enrichment, shape expansion, replay measurement, and validation into clear tooling responsibilities.
- Allows partial data to be explicit instead of blocking an entire database refresh.
- Reuses HCCL measurements across compatible compute-stack directories.
- Supports exact measured rows and generated rows that can be filled incrementally.

#### Limitations

- Replay scripts require manual maintenance, and replay coverage expands incrementally.
- The first phase does not auto-consume `torch_npu_reference` to generate replay scripts.
- NPU-dependent replay and communication tests have limited regular CI coverage.
- CSV schema is easy to diff, but also easy to damage through manual edits.
- FIA replay can remain only an approximate reproduction when runtime sequence metadata is missing.
- Shape-grid generation may produce rows that look structurally valid but cannot be replayed by a specific operator API.

## 3. Implementation Plan

### 3.1 Rollout Plan

The schedule is phase-ordered. Phases 0-2 can run on any development host. Phases 3-4 depend on prepared NPU host windows. Phase 5 starts only after replay and communication reports are available.

| Milestone | Owner role and order | Scope | Exit criteria |
| --- | --- | --- | --- |
| Phase 0: RFC and contract | RFC owner; first phase before implementation work | Adopt this RFC as the standalone collection-tooling contract | Contributors can execute the flow without temporary documents |
| Phase 1: Parser/schema baseline | Tooling owner; no-NPU phase after Phase 0 | Validate raw parsing, output column order, and software-stack directory naming | Existing no-NPU parser and data-source tests pass |
| Phase 2: Shape-grid staging | Data collection owner; no-NPU staging phase after Phase 1 | Use `--rows`, `--seed`, and `--max-hbm-gb` in a staging directory | Generated rows preserve slot/dtype/format contracts and report skipped kernels |
| Phase 3: Replay writeback | Data collection owner and NPU host owner; scheduled when an NPU host is available | Run selected `op_replay` scripts through `start_microbench.py` on an NPU host | `Average Duration(us)` is filled, invalid rows are reported, and gap reports are generated |
| Phase 4: Communication collection | HCCL/NPU owner; scheduled with the same or adjacent NPU host window | Collect or refresh HCCL CSVs for the target CANN version | `validate_comm_alignment.py` passes under the agreed tolerance |
| Phase 5: Promotion | Data reviewer and runtime owner; after Phase 3-4 report review | Review CSV/YAML/report changes and promote the database directory | Runtime lookup tests pass and known limitations are recorded in the refresh report |

### 3.2 Test Plan

Run no-NPU checks before any hardware collection:

```bash
python -m pytest tests/tools/test_op_replay.py \
  tests/tools/test_op_replay_common.py \
  tests/tools/test_start_microbench.py \
  tests/tools/test_generate_shape_grid.py \
  tests/tools/test_fia_parser_backfill.py \
  tests/tools/test_generate_comm_microbench.py \
  tests/tools/test_validate_comm_alignment.py

python -m pytest tests/perf_database
```

Run syntax and help checks for changed replay scripts:

```bash
python -m py_compile tools/perf_data_collection/op_replay/<KernelType>_run.py
python tools/perf_data_collection/op_replay/<KernelType>_run.py --help
```

Run hardware-dependent checks only on a prepared NPU host:

```bash
python tools/perf_data_collection/start_microbench.py \
  --database-path tensor_cast/performance_model/profiling_database/data/{device}/vllm_ascend/{version_dir} \
  --op MatMulV2 RmsNorm \
  --repeat-count 1 \
  --update-mode missing-only

bash tools/perf_data_collection/comm_bench/run_comm_bench.sh \
  tensor_cast/performance_model/profiling_database/data/{device}/hccl/{cann_version}
```

### 3.3 Follow-up Work

- Add an in-repo replay-script generator that consumes `torch_npu_reference.{KernelType}.microbench_api` and operator-specific metadata.
- Add stricter CSV schema validation and duplicate-signature checks before writeback.
- Expand NPU-marked integration tests to cover representative custom operators.
- Improve FIA runtime metadata capture to reduce replay rows that depend on heuristic inference.
- Add a single collection command that orchestrates parse, enrich, shape generation, replay, communication validation, and report packaging while preserving this file contract.

### 3.4 Completion Criteria

- `tools/perf_data_collection/` entry points, database directories, CSV schemas, and the `op_mapping.yaml` contract match this RFC.
- At least one target software-stack directory can be generated through parse, optional enrich, shape generation, replay writeback, and report generation.
- The HCCL directory can be generated by communication benchmark tooling and referenced through `communication_data_ref`.
- No-NPU tests cover parser behavior, shape grid, FIA backfill, `start_microbench` writeback logic, communication CSV validation, and data-source query behavior.
- On an NPU host, selected replay and communication benchmark commands can run and produce status and gap reports for human review.
