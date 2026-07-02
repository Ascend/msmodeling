# Benchmark Cases Runner

## Introduction

Benchmark Cases Runner is a batch execution tool for [Throughput Optimizer](./user_guide/msmodeling_throughput_optimizer_user_guide.md). While the Throughput Optimizer evaluates a single hardware/configuration combination per invocation, Benchmark Cases Runner drives multiple cases sequentially from a CSV input file, aggregates all results into a single output CSV, and enables cross-hardware and cross-scenario performance comparison without writing custom scripts.

> **Module Location**: This tool is located at `optix/run_throughput_optimizer_cases.py` in the msmodeling package.

**Use Cases:**

- **Multi-Configuration Evaluation**: Benchmark multiple device × card-count × model × input/output-length combinations in a single run
- **Cross-Hardware Comparison**: Compare optimal throughput across different device profiles under consistent SLO constraints
- **Regression Tracking**: Run a fixed set of benchmark cases and collect results into a standardized CSV for trend analysis

**Key Features:**

- CSV-driven case definition: each row specifies one benchmark case
- Sequential execution with per-case error isolation (one failure does not abort the batch)
- Incremental result writing with batch flush (crash-safe at batch granularity)
- Template CSV generation with multiple example rows for quick setup
- CSV validation mode to check configuration before running
- Unified result CSV with Decode/Prefill metrics aligned with Throughput Optimizer output

## Quick Start

### Generate a template CSV

```bash
python -m optix.run_throughput_optimizer_cases --write-template cases_template.csv
```

This creates a CSV file with all required columns and **three example rows** covering common scenarios (1-card agg, 8-card agg, 4-card disagg with MTP). Edit the file to define your benchmark cases.

### Validate your CSV before running

```bash
python -m optix.run_throughput_optimizer_cases --validate-csv cases.csv
```

This reads and validates all rows in the CSV, printing a summary of each parsed case (device, model, limits, mode, etc.) without actually executing the optimizer. Use this to catch configuration errors before a long batch run.

### Run benchmark cases from CSV

```bash
python -m optix.run_throughput_optimizer_cases --input-csv cases.csv --output-csv results.csv
```

Each row in `cases.csv` is executed sequentially. Results are written to `results.csv` after every batch of 10 cases, so partial results are preserved even if the run is interrupted. If a single case fails, the error is logged and execution continues with the next case; the failed case gets a row in the CSV with empty metric columns.

## CSV Input Format

The input CSV must include a header row with column names matching the template. Each subsequent row defines one benchmark case. List-type fields (e.g., `mtp_acceptance_rate`, `ep_sizes`) use semicolon (`;`) as the in-cell delimiter.

### Required Columns

| Column | Description | Example |
|--------|-------------|---------|
| `case_name` | Identifier for this case (auto-generated as `row_N` if empty) | `8card_agg_w8a8` |
| `device` | Device profile name (same values as `--device` in Throughput Optimizer) | `ATLAS_800_A3_752T_128G_DIE` |
| `num_devices` | Number of devices | `8` |
| `model_id` | Recommended: reviewed absolute local model path. Hugging Face or ModelScope ids are accepted but may execute remote code through `trust_remote_code=True` and are not security-guaranteed. | `/data/models/Qwen3-32B` |
| `input_length` | Input token length | `3500` |
| `output_length` | Output token length | `1500` |
| `ttft_limits` | TTFT SLO limit in ms (empty = no constraint) | `2000` |
| `tpot_limits` | TPOT SLO limit in ms (defaults to 50.0 ms if empty) | `50` |

### Optional Columns

| Column | Description | Default |
|--------|-------------|---------|
| `tp_sizes` | TP sizes to search (semicolon-separated) | `None` (default range) |
| `quantize_linear_action` | Linear quantization action | `None` (falls back to `W8A8_DYNAMIC`) |
| `quantize_attention_action` | Attention quantization action | `None` (falls back to `DISABLED`) |
| `ep_sizes` | EP sizes to search (semicolon-separated), aligns with `--ep-sizes` in Throughput Optimizer | `None` |
| `num_mtp_tokens` | Number of MTP tokens | `0` |
| `mtp_acceptance_rate` | MTP acceptance rates (semicolon-separated) | `0.9;0.6;0.4;0.2` |
| `compile` | Enable torch.compile (`true`/`false`) | `false` |
| `mode` | Run mode: `agg` or `disagg` | `agg` |
| `max_prefill_tokens` | Maximum prefill tokens | `8192` |
| `batch_range` | Batch size range (semicolon-separated min;max) | `None` |
| `serving_cost` | Serving cost | `0` |
| `jobs` | Number of parallel jobs | `8` |
| `log_level` | Log verbosity level | `info` |
| `mxfp4_group_size` | Group size for MXFP4 quantization | `32` |
| `reserved_memory_gb` | Reserved device memory in GB | `0` |
| `compile_allow_graph_break` | Allow graph breaks in compile (`true`/`false`) | `false` |

> [!Warning]
> `ttft_limits` and `tpot_limits` accept **at most one value** per case. If multiple values are provided (e.g., `50;100`), the tool raises an error. Split them into separate CSV rows instead. All limit values use **milliseconds (ms)** as the unit, consistent with Throughput Optimizer.

<!-- -->

> [!Note]
> If any required column (`device`, `num_devices`, `model_id`, `input_length`, `output_length`) is missing from the CSV header, the tool raises an error immediately rather than silently using wrong defaults.

## Example CSV

```csv
case_name,device,num_devices,model_id,input_length,output_length,ttft_limits,tpot_limits,tp_sizes,quantize_linear_action,quantize_attention_action,ep_sizes,num_mtp_tokens,mtp_acceptance_rate,compile,mode,max_prefill_tokens,batch_range,serving_cost,jobs,log_level,mxfp4_group_size,reserved_memory_gb,compile_allow_graph_break
8card_agg_w8a8,ATLAS_800_A3_752T_128G_DIE,8,Qwen/Qwen3-32B,3500,1500,,50,,W8A8_DYNAMIC,DISABLED,,0,,true,agg,8192,,0,8,info,32,0,false
4card_disagg_ep,ATLAS_800_A3_752T_128G_DIE,4,Qwen/Qwen3-32B,2000,500,,50,,W8A8_DYNAMIC,INT8,2,0,,true,disagg,8192,,0,8,info,32,0,false
1card_disagg_mtp,ATLAS_800_A3_752T_128G_DIE,1,Qwen/Qwen3-32B,16000,1000,,50,,W8A8_DYNAMIC,DISABLED,,3,0.9;0.6;0.4,true,disagg,16000,,0,8,info,32,0,false
```

This CSV defines three cases:

1. **8-card aggregation** with W8A8_DYNAMIC linear quantization and TPOT ≤ 50 ms
2. **4-card disaggregated** with W8A8_DYNAMIC + INT8 attention quantization and EP size 2
3. **1-card disaggregated prefill** with MTP (3 tokens) and compile enabled

## Output CSV

The output CSV contains a header row, a reference row listing valid quantization options, and one result row per benchmark case. Key columns include:

| Column | Description |
|--------|-------------|
| `Case_Name` | Case identifier |
| `Device Type` | Device profile name |
| `Number of Devices` | Number of devices |
| `Input Length` / `Output Length` | Token lengths |
| `Model` | Model identifier |
| `Decode_Linear Quant Type` | Linear quantization type for best decode config |
| `Decode_Attn Quant Type` | Attention quantization type for best decode config |
| `Decode_Use EP` | EP size used (shown when ep_size > 1) |
| `Decode_MTP Tokens` | Number of MTP tokens |
| `Decode_TPOT Target(ms)` | TPOT SLO target in ms |
| `Decode_Concurrency` | Best decode concurrency |
| `Decode_TPOT(ms)` | Best decode TPOT under SLO |
| `Decode_Total TPS` | Best decode total throughput |
| `Decode_TPS/Device` | Best decode per-device throughput |
| `Decode_Mem` / `Decode_Comm` / `Decode_Cube` / `Decode_Vec` | Performance breakdown percentages |
| `Decode_TP Size` / `Decode_PP Size` / `Decode_DP Size` | Best parallel config |
| `Prefill_Linear Quant Type` | Linear quantization type for best prefill config |
| `Prefill_Attn Quant Type` | Attention quantization type for best prefill config |
| `Prefill_Use EP` | EP size used (shown when ep_size > 1) |
| `Prefill_MTP Tokens` | Number of MTP tokens |
| `Prefill_TTFT Target(ms)` | TTFT SLO target in ms |
| `Prefill_Concurrency` | Best prefill concurrency |
| `Prefill_TTFT(ms)` | Best prefill TTFT under SLO |
| `Prefill_Total TPS` | Best prefill total throughput |
| `Prefill_TPS/Device` | Best prefill per-device throughput |
| `Prefill_Mem` / `Prefill_Comm` / `Prefill_Cube` / `Prefill_Vec` | Performance breakdown percentages |
| `Prefill_TP Size` / `Prefill_PP Size` / `Prefill_DP Size` | Best parallel config |
| `QuantizeLinearAction_options` | All valid linear quantization actions |
| `QuantizeAttentionAction_options` | All valid attention quantization actions |

> [!Note]
> If a case has no prefill result (e.g., disagg decode-only mode), the prefill metric columns are left empty. If a case has no decode result, the decode metric columns are left empty. If no configuration satisfies the SLO, the corresponding metric columns are present but the best values reflect the closest attempt. If a case fails entirely (e.g., invalid model, device not found), the metric columns are empty and the error is printed to stderr; the case still gets a row in the CSV.

## CLI Parameters

```bash
Options:
  -h, --help            show this help message and exit
  --input-csv INPUT_CSV
                        Path to input CSV file with benchmark cases (one case per row).
                        Header must include columns from the template.
  --write-template PATH
                        Write a template CSV with multiple example rows to PATH and exit.
  --output-csv OUTPUT_CSV
                        Path to output CSV file for results. Defaults to
                        'benchmark_cases_results.csv' when --input-csv is used.
  --test-conversion     Run internal conversion test and exit.
  --validate-csv PATH   Validate the input CSV file at PATH and print a summary
                        of parsed cases without executing the optimizer.
```

## Execution Behavior

- **Sequential execution**: Cases run one at a time in CSV row order.
- **Per-case error isolation**: If a single case fails (e.g., invalid model, device not found, runtime exception), the error is printed to stderr and the case gets a row with empty metrics in the output CSV. Execution continues with the next case.
- **Batch flush**: Results are flushed to disk every 10 cases. If the process is interrupted, results from completed batches are preserved.
- **In-process execution**: Each case calls `ParallelRunner` from `serving_cast.parallel_runner` in-process. Results from `ParallelRunner` are processed using `OptimizerSummary._prepare_agg_disagg_results()` for SLO filtering, which correctly handles disagg mode by applying only the relevant limit per phase (TTFT for prefill summaries, TPOT for decode summaries). Log level is configured via `logging.basicConfig` before each case, consistent with Throughput Optimizer.

## Relationship to Throughput Optimizer

Benchmark Cases Runner internally calls [Throughput Optimizer](./user_guide/msmodeling_throughput_optimizer_user_guide.md)'s `ParallelRunner` for each case. The mapping from CSV columns to Throughput Optimizer arguments is:

| CSV Column | Throughput Optimizer Argument |
|------------|-------------------------------|
| `device` | `--device` |
| `num_devices` | `--num-devices` |
| `model_id` | positional `model_id` |
| `input_length` | `--input-length` |
| `output_length` | `--output-length` |
| `ttft_limits` | `--ttft-limits` |
| `tpot_limits` | `--tpot-limits` |
| `compile` | `--compile` |
| `mode` | `--disagg` (when set to `disagg`) |
| `quantize_linear_action` | `--quantize-linear-action` |
| `quantize_non_expert_linear_action` | `--quantize-non-expert-linear-action` |
| `quantize_attention_action` | `--quantize-attention-action` |
| `tp_sizes` | `--tp-sizes` |
| `ep_sizes` | `--ep-sizes` |
| `num_mtp_tokens` | `--num-mtp-tokens` |
| `mtp_acceptance_rate` | `--mtp-acceptance-rate` |
| `max_prefill_tokens` | `--max-prefill-tokens` |
| `batch_range` | `--batch-range` |
| `serving_cost` | `--serving-cost` |
| `jobs` | `--jobs` |
| `log_level` | `--log-level` |
| `reserved_memory_gb` | `--reserved-memory-gb` |
| `compile_allow_graph_break` | `--compile-allow-graph-break` |
| `mxfp4_group_size` | `--mxfp4-group-size` |

The following Throughput Optimizer arguments are set to defaults since they are not exposed as CSV columns:

| Argument | Default Value | Reason |
|----------|---------------|--------|
| `image_batch_size` | `None` | Multimodal not supported in CSV |
| `image_height` | `None` | Multimodal not supported in CSV |
| `image_width` | `None` | Multimodal not supported in CSV |
| `prefix_cache_hit_rate` | `0.0` | Not commonly varied per case |
| `enable_optimize_prefill_decode_ratio` | `False` | PD ratio optimization not exposed |
| `prefill_devices_per_instance` | `None` | PD ratio optimization not exposed |
| `decode_devices_per_instance` | `None` | PD ratio optimization not exposed |
| `moe_dp_sizes` | `None` | MOE-DP search not exposed |

For detailed parameter descriptions, see the [Throughput Optimizer documentation](./user_guide/msmodeling_throughput_optimizer_user_guide.md).
