# Service Parameter Optimization User Guide

## Overview

**Service Parameter Optimization** (msmodeling optix) is a service-parameter optimization feature based on the PSO particle-swarm search algorithm. It automatically searches on real service frameworks to obtain the best throughput parameter combination that meets latency requirements such as TTFT/TPOT.

The tool consists of two core modules:

- **Parameter optimization module**: uses the PSO particle-swarm algorithm to generate service parameter combinations and iteratively approach the optimal solution.
- **Parameter validation module**: automatically starts the service process and benchmark process, runs tests, and collects performance results. Currently supported service frameworks include `vLLM` and `MindIE`; supported benchmark tools include `AISBench` and `vllm_benchmark`.

The tool has been validated with DeepSeek V3.1, GLM5, and Qwen3.5-27b. In principle, the supported model range is not limited.

## Audience and Reading Path

This document is intended for performance and deployment engineers who need to automatically optimize vLLM/MindIE service deployment parameters. We recommend reading in the following order:

1. [Environment Preparation and Installation](#environment-preparation-and-installation) — install msmodeling in a uv virtual environment and make sure vLLM/MindIE are deployed on the system.
2. [Quick Start](#quick-start) — run a default optimization round.
3. [Configuration File Description](#configuration-file-description) and [Command Parameters](#command-parameters) — learn all parameters and configuration items.
4. [Output File Description](#output-file-description) — filter optimal parameters based on business SLO.
5. [Appendix](#appendix) — environment variables, troubleshooting, and ternary derived type usage.

## Supported Products

> [!NOTE]
> For the specific Ascend product models, see the [Ascend Product Form Description](https://www.hiascend.com/document/detail/zh/AscendFAQ/ProduTech/productform/hardwaredesc_0001.html).
> For Atlas A2 training/inference series products, only the Atlas 800I A2 inference server is currently supported within the series.
> For Atlas inference series products, only the Atlas 300I Duo inference card + Atlas 800 inference server (model 3000) is currently supported within the series.

| Product type | Supported |
|--|:----:|
| Atlas A3 training series / Atlas A3 inference series | √ |
| Atlas A2 training series / Atlas A2 inference series | √ |
| Atlas 200I/500 A2 inference product | √ |
| Atlas inference series products | √ |
| Atlas training series products | × |
| Atlas 350 accelerator card | × |

## Plugin Support

Service Parameter Optimization supports user-defined search parameter configurations and benchmark tools by registering the corresponding plugin through the plugin interface. Built-in plugins ship with the `optix` package and are selected directly via `-e`/`-b`; custom plugins can also be registered through the plugin interface.

**Built-in plugins**:

| Plugin | Type | Description |
|---|---|---|
|`vllm`|Service framework (`-e`)|vLLM service framework adapter|
|`mindie`|Service framework (`-e`)|MindIE service framework adapter|
|`ais_bench`|Benchmark (`-b`)|AISBench benchmark adapter|
|`vllm_benchmark`|Benchmark (`-b`)|vllm_benchmark benchmark adapter|

**Extension plugin examples** (under `contrib/optix/`, install as needed):

| Plugin | Type | Provider | Description | Plugin directory |
|---|---|---|---|---|
|`custom_vllm`|Service framework (`-e`)|contrib example|Starts local and remote vLLM containers via Docker + SSH, demonstrating custom service framework integration|`contrib/optix/vllm_msserviceprofiler/`|
|`evalscopeperf`|Benchmark (`-b`)|contrib example|Wraps the `evalscope perf` command as a benchmark tool, demonstrating custom benchmark integration|`contrib/optix/evalscopeplugin/`|

> Plugins are registered through Python entry points. For custom plugin development and usage, see [Plugin Development Guide](msmodeling_optix_plugin_user_guide.md).

## Environment Preparation and Installation

### Environment Isolation Principle

It is recommended to install the optix optimization tool in a virtual environment. Running `uv sync` in the repository root automatically creates `.venv` and completes the installation.

Installing msmodeling also installs packages such as `torch` and `transformers`. Installing msModeling in the system Python may conflict with the existing `torch`/`transformers` versions on the system, which can result in:

- vLLM or MindIE failing to start or reporting inference errors
- Version mismatch with the validated versions on the Ascend inference stack

### Installation Steps

```bash
git clone https://gitcode.com/Ascend/msmodeling.git   # skip if already cloned
cd msmodeling
uv sync
```

> [!WARNING]
> Do not run `pip install vllm`, `mindie_llm`, or other deployment packages inside the msmodeling virtual environment, and do not install msmodeling in a system Python without a venv. If in doubt, see [Recommended Practice: Service Parameter Optimization Environment and Deployment Stack](../install_guide/msmodeling_optix_env_and_deployment_stack.md).

### Verify Installation

```bash
uv run msmodeling optix --help
```

If the `msmodeling optix` help information is printed successfully, the tool is installed correctly. Deployment stack checks and the real optimization run belong to [Quick Start](#quick-start).

### Uninstall

```bash
uv pip uninstall msmodeling
```

## Quick Start

1. **Modify the configuration file**: after [Environment Preparation and Installation](#environment-preparation-and-installation), modify the configuration file `config.toml` as needed, including optimization parameters, benchmark tool parameters, and service parameters. See [Configuration File Description](#configuration-file-description).

   > **Note:** If you are unsure which parameters to tune and what ranges to use, use the **`optix-param-recommend` parameter recommendation skill**: provide the hardware, model, workload, and optimization target, and it returns recommended parameters and search ranges. Review them against your actual deployment environment (GPU memory, number of cards, latency requirements, and so on) before writing them into `config.toml`.

2. **Start optimization**: by default, vLLM service parameter optimization based on `AISBench` is executed. For other usage, see [Command Parameters](#command-parameters).

    ```bash
    msmodeling optix
    ```

3. **View results**: the optimization duration depends on the model size, dataset size, and the number of optimization rounds. When finished, `data_storage_*.csv` files are generated and saved in the **`result/store` subdirectory of the current working directory** (can be changed via `[data_storage].store_dir`). See [Output File Description](#output-file-description).

## Configuration File Description

The optimization configuration file is located at `./msmodeling/optix/config.toml` by default. The file is organized by TOML sections, each of which corresponds to one function:

| TOML section | Purpose | Document section |
|---|---|---|
| Top level | Optimization parameters | [Optimization Parameters](#optimization-parameters) |
| `[vllm]` / `[mindie]` | vLLM / MindIE service parameters and optimization fields | [Service Parameters](#service-parameters) |
| `[ais_bench.command]` / `[vllm_benchmark.command]` | Benchmark tool parameters | [Benchmark Tool Parameters](#benchmark-tool-parameters) |
| `[deploy]` | Deployment root directory (optional) | [Advanced Configuration → Deployment Environment](#deployment-environment) |
| `[data_storage]` | Result storage and fine-tuning (optional) | [Advanced Configuration → Result Storage and Fine-tuning](#result-storage-and-fine-tuning) |
| `[health_check]` | Runtime log anomaly detection (optional) | [Advanced Configuration → Log Detection](#log-detection) |

### Optimization Parameters

> **Note**: it is recommended to explicitly set the following parameters in `config.toml`. `n_particles`, `iters`, the penalty coefficients, `ttft_slo`, `tpot_slo`, `service`, and so on all have code defaults, but on the first run it is recommended to configure them explicitly according to the business latency requirements (SLO) to avoid unexpected defaults.

| Parameter | Optional/required | Description |
|---|---|---|
|`n_particles`|Required|Number of optimization seeds, that is, the number of parameter combinations generated per round. Range: integer 1-1000. Recommended: 8-16.|
|`iters`|Required|Number of iteration rounds. Range: integer 1-1000. Recommended: 4-8.|
|`ttft_penalty`|Required|Penalty coefficient for `time_to_first_token` timeout. Set to 0 if there is no latency requirement for `time_to_first_token`. Range: [0, 100]. Recommended: 1.|
|`tpot_penalty`|Required|Penalty coefficient for `time_per_output_token` timeout. Set to 0 if there is no latency requirement for `time_per_output_token`. Range: [0, 100]. Recommended: 1.|
|`success_rate_penalty`|Required|Request success rate penalty coefficient. Range: integer 1-1000. Recommended: 5.|
|`ttft_slo`|Required|Latency limit for `time_to_first_token`. For example, set 2 to limit `time_to_first_token` within 2s. Range: (0, 100], unit s.|
|`tpot_slo`|Required|Latency limit for `time_per_output_token`. For example, set 0.05 to limit `time_per_output_token` within 50ms. Range: (0, 100], unit s.|
|`use_request_rate_calibration`|Optional|Whether to calibrate `REQUESTRATE` through real measurements. Default: `true`. See below.|

**The two modes of `use_request_rate_calibration`**: this switch determines the behavior of the two special fields `CONCURRENCY` and `REQUESTRATE` in PSO.

| Value | Description | CONCURRENCY | REQUESTRATE |
|---|---|---|---|
|`true` (default)|With `CONCURRENCY` fixed, search for a better `REQUESTRATE`|Fixed at `max`, not searched|Calibrated through real measurements with `max` as the upper bound|
|`false`|Out-of-the-box stage, aiming to search for a better `CONCURRENCY`|Searched by PSO within `[min, max]`|Fixed at `max`|

> [!NOTE]
> You can configure the number of seeds and iterations according to the estimated time. The time per seed is service startup + benchmark time. For example, if startup + benchmark takes 15min per seed and you can spend 8 hours on optimization (about 50 seeds in total), configure `n_particles=8` and `iters=4` (the number of seeds is roughly twice the number of iterations).

### Benchmark Tool Parameters

<details open>
<summary>Using the AISBench benchmark tool</summary>

### AISBench Tool Parameters

When using the ais_bench benchmark tool, modify the `[ais_bench.command]` parameters in `config.toml`, referring to the [AISBench Quick Start](https://github.com/AISBench/benchmark/blob/master/docs/source_zh_cn/get_started/quick_start.md).

| Parameter | Description |
|---|---|
|`models`|Specifies the model task, configurable per the [model configuration guide](https://github.com/AISBench/benchmark/blob/master/docs/source_zh_cn/base_tutorials/all_params/models.md).|
|`mode`|Run mode, configurable per the [run mode guide](https://github.com/AISBench/benchmark/blob/master/docs/source_zh_cn/base_tutorials/all_params/mode.md).|
|`others`|Additional parameters, such as `--num_prompts 1000`, separated by spaces. Empty by default.|

</details>

<details>
<summary>Using the vllm_benchmark benchmark tool</summary>

### vllm_benchmark Tool Parameters

When using the vllm_benchmark benchmark tool, modify the `[vllm_benchmark.command]` parameters in `config.toml`:

| Parameter | Optional/required | Description |
|---|---|---|
|`host`|Required|Host IP, must be consistent with `host` in `[vllm.command]`; for example `127.0.0.1`.|
|`port`|Required|Port, must be consistent with `port` in `[vllm.command]`.|
|`model`|Required|Model path, must be consistent with `model` in `[vllm.command]`.|
|`served_model_name`|Required|Model name, must be consistent with `served_model_name` in `[vllm.command]`.|
|`dataset_name`|Required|Dataset name.|
|`others`|Optional|Additional parameters, separated by spaces without internal spaces. For example `--dataset-path /path/to/data --num-prompts 1000`. Empty by default.|

</details>

### Service Parameters

<details open>
<summary>Using the vLLM service framework</summary>

### vLLM Service Parameters

When using the vLLM framework, modify the `[vllm.command]` parameters in `config.toml`:

```toml
[vllm.command]
host = "127.0.0.1"
port = "8000"
model = "/workspace/vllm/models/llama-2-7b-chat-hf"
served_model_name = "llama-2-7b-chat-hf"
others = ""
```

| Parameter | Optional/required | Description |
|---|---|---|
|`host`|Required|Host IP, must be consistent with `host` in `[vllm_benchmark.command]`.|
|`port`|Required|Port, must be consistent with `port` in `[vllm_benchmark.command]`.|
|`model`|Required|Model path.|
|`served_model_name`|Required|Model name.|
|`others`|Optional|Additional parameters, separated by spaces. For example `--tensor-parallel-size 2 --no-enable-prefix-caching`. Empty by default.|

### vLLM Custom Parameter Optimization

The optimization tool supports adding vLLM parameters to the optimization through `[[vllm.target_field]]`. Depending on how the parameter takes effect, there are two configuration styles:

- **vLLM environment variables**: declare the field in `[[vllm.target_field]]` with `config_position = "env"`. The tool writes the same-named uppercase environment variable automatically before starting the service in each round; it does not need to be written into `others` in `[vllm.command]`.
- **vLLM command-line parameters**: declare the field in `[[vllm.target_field]]`, then reference it in `others` in `[vllm.command]` to append it to the startup command.

> **Variable reference rule**: use `$FIELD_NAME_IN_UPPERCASE` in `others` to reference an optimization field; the tool replaces it with the actual value of the current iteration at runtime.

#### Example 1: vLLM environment variable optimization

When the parameter to optimize is itself a vLLM environment variable, just add it to `[[vllm.target_field]]`:

```toml
[[vllm.target_field]]
name = "VLLM_WORKER_MULTIPROC_METHOD"
config_position = "env"
dtype = "enum"
dtype_param = ["fork", "spawn"]
value = "fork"
```

Such parameters do not need to be referenced in `others` in `[vllm.command]`; keep `others = ""` or only fill in other command-line parameters.

#### Example 2: command-line enumerated numeric parameter (`gpu_memory_utilization`)

```toml
# Step 1: declare the optimization field
[[vllm.target_field]]
name = "GPU_MEMORY_UTILIZATION"
config_position = "env"
dtype = "enum"
dtype_param = [0.9, 0.91, 0.92]
value = 0.9

# Step 2: reference the variable in others in [vllm.command]
[vllm.command]
others = "--gpu-memory-utilization $GPU_MEMORY_UTILIZATION"
```

#### Example 3: command-line switch/compound string parameter (`--compilation-config`)

When the parameter is a complete CLI string, use `""` (disabled) and the enabled form as the two enum candidates. The tool skips the empty string automatically and does not append anything to the startup command.

> **Note**: TOML strings use the double quote `"` as the delimiter; if the string contains double quotes, escape them with `\"`.

```toml
[[vllm.target_field]]
name = "COMPILATION_CONFIG"
config_position = "env"
dtype = "enum"
dtype_param = ["", "--compilation-config '{\"cudagraph_mode\": \"FULL_DECODE_ONLY\"}'"]
value = "--compilation-config '{\"cudagraph_mode\": \"FULL_DECODE_ONLY\"}'"

[vllm.command]
others = "$COMPILATION_CONFIG"
```

### Common vLLM Optimization Fields

| Field | dtype | Description |
|---|---|---|
|`MAX_NUM_BATCHED_TOKENS`|`int`|Maximum number of tokens per batch (including prefill); affects memory usage and throughput|
|`MAX_NUM_SEQS`|`int`|Maximum number of sequences per batch, that is, the decode concurrency upper bound|
|`CONCURRENCY`|`int`|Benchmark concurrency; special field, see below|
|`REQUESTRATE`|`float`|Benchmark request rate; special field, see below|

> **Special behavior of `CONCURRENCY` / `REQUESTRATE`**: these two fields apply to both vLLM and MindIE. The tool rewrites them according to `use_request_rate_calibration`: when `true`, `CONCURRENCY` is fixed at `max` and `REQUESTRATE` is calibrated through real measurements with `max` as the upper bound; when `false`, `CONCURRENCY` is searched by PSO and `REQUESTRATE` is fixed at `max`. Therefore set `max` to the expected maximum value; when `min == max`, the field is treated as a constant and is not searched.

</details>

<details>
<summary>Using the MindIE service framework</summary>

### MindIE Service Parameters

When using the MindIE framework, modify the `[mindie.command]` parameters in `config.toml`, referring to the [MindIE server configuration parameter guide](https://www.hiascend.com/document/detail/zh/mindie/20RC1/mindieservice/servicedev/mindie_service0285.html). Service parameters can specify a search range directly, for example configuring the search space of `max_batch_size` to 10-400:

```toml
[[mindie.target_field]]
name = "max_batch_size"
config_position = "BackendConfig.ScheduleConfig.maxBatchSize"
min = 10
max = 400
dtype = "int"
```

Parameters can also be derived from another parameter, for example `max_prefill_batch_size` is related to `max_batch_size` (`max_prefill_batch_size = ratio * max_batch_size`, `0 < ratio < 1`):

```toml
[[mindie.target_field]]
name = "max_prefill_batch_size"
config_position = "BackendConfig.ScheduleConfig.maxPrefillBatchSize"  # This value must not exceed maxBatchSize
min = 0.1
max = 0.7
dtype = "ratio"
dtype_param = "max_batch_size"
```

### Common MindIE Optimization Fields

| Field | config_position | dtype | Description |
|---|---|---|---|
|`max_batch_size`|`BackendConfig.ScheduleConfig.maxBatchSize`|`int`|Maximum batch size, the main throughput tuning knob|
|`max_prefill_batch_size`|`BackendConfig.ScheduleConfig.maxPrefillBatchSize`|`ratio`|Derived from `max_batch_size` by ratio; ratio recommended 0.1-0.7|
|`prefill_time_ms_per_req`|`BackendConfig.ScheduleConfig.prefillTimeMsPerReq`|`range`|Per-request prefill time upper bound (ms), enumerated by step|
|`decode_time_ms_per_req`|`BackendConfig.ScheduleConfig.decodeTimeMsPerReq`|`range`|Per-request decode time upper bound (ms), enumerated by step|
|`support_select_batch`|`BackendConfig.ScheduleConfig.supportSelectBatch`|`bool`|Whether to enable time-based batch selection; when enabled, `prefill/decode_time_ms_per_req` are forced to 0|
|`max_queue_delay_microseconds`|`BackendConfig.ScheduleConfig.maxQueueDelayMicroseconds`|`range`|Maximum request queue delay (μs), enumerated by step|
|`max_preempt_count`|`BackendConfig.ScheduleConfig.maxPreemptCount`|`ratio`|Preemption upper bound, derived from `max_batch_size` by ratio|
|`CONCURRENCY`|`env`|`int`|Benchmark concurrency; special field, see the vLLM section|
|`REQUESTRATE`|`env`|`float`|Benchmark request rate; special field, see the vLLM section|

</details>

#### target_field Field Description

| Field | Description | Example |
|---|---|---|
| `name` | Optimization field name. Environment-variable fields must be uppercase; the tool writes the same-named uppercase environment variable before starting the service in each round. Command-line fields serve as the `$FIELD_NAME_IN_UPPERCASE` variable reference | `COMPILATION_CONFIG` |
| `config_position` | Where the field takes effect: `env` (environment variable) or a MindIE service configuration file path (for example `BackendConfig.ScheduleConfig.maxBatchSize`) | `env` |
| `dtype` | Value type, determines how PSO samples: `int`/`float`/`bool`/`enum`/`range`/`ratio`/`share`/`factories`/`times`/`ternary_factories`/`ternary_times`. See [target_field Supported dtype Types](#target_field-supported-dtype-types) | `enum` |
| `dtype_param` | Depends on dtype: for `enum` a candidate value list, for `range` a step integer, for `ratio` and derived types the dependent-field configuration | `["", "--compilation-config '{\"cudagraph_mode\": \"FULL_DECODE_ONLY\"}'"]` |
| `value` | Initial value, used to generate baseline data | `"--compilation-config '{\"cudagraph_mode\": \"FULL_DECODE_ONLY\"}'"` |
| `min` / `max` | Search interval bounds (effective for `int`/`float`/`range`); when `min == max` the field is a constant and is not searched; the actual candidates of `enum` come from `dtype_param`; derived types require `min`/`max` to be 0 | — |
| `constant` | Optional; when set, the field is fixed to that value and is not searched by PSO (written by the tool when rewriting `CONCURRENCY`/`REQUESTRATE`) | — |

#### target_field Supported dtype Types

| Category | dtype | Description | dtype_param format |
|---|---|---|---|
| Basic | `int` | Integer within [min, max] | — |
| Basic | `float` | Float within [min, max] | — |
| Basic | `bool` | Boolean switch (true when the value is > 0.5) | — |
| Basic | `enum` | Select from the candidate list (numeric or string) | Candidate value list, e.g. `[1, 2, 4, 8]` |
| Basic | `range` | Enumerate by step within [min, max] | Step integer, e.g. `10` |
| Binary derived | `ratio` | `int(ratio × target)` | Dependent field name (string), e.g. `"max_batch_size"` |
| Binary derived | `share` | `target.min + target.max - target.value` (complementary) | Dependent field name (string) |
| Binary derived | `factories` | `product ÷ target` | `{"target_name": "field name", "product": value, "dtype": "int"}` |
| Binary derived | `times` | `product × target` | `{"target_name": "field name", "product": value, "dtype": "int"}` |
| **Ternary derived** | **`ternary_factories`** | **`product ÷ (field_a × field_b)`** | **`{"target_names": ["A", "B"], "product": value, "dtype": "int"}`** |
| **Ternary derived** | **`ternary_times`** | **`product × field_a × field_b`** | **`{"target_names": ["A", "B"], "product": value, "dtype": "int"}`** |

> [!NOTE]
> The values of derived-type fields (`factories` / `times` / `ternary_factories` / `ternary_times`) are derived automatically from dependencies and **do not participate in particle-swarm search**; set both `min` and `max` to `0`. If any dependent field value is `0` (division scenarios) or `None`/`NaN` (multiplication scenarios), the derivation for that round is skipped, the field keeps its original value, and a warning is logged.
>
> For the complete usage of ternary derived types (`ternary_factories` / `ternary_times`, `priority_policy` priority repair), see the [Appendix](#ternary-derived-type-usage-examples).

<details>
<summary>Advanced Configuration</summary>

### Deployment Environment

When Service Parameter Optimization starts vLLM, MindIE, or benchmark tools, optix first strips the msmodeling virtual environment related variables and then looks up the deployment root directory according to the configuration below, whose `bin/` must contain `vllm`, `ais_bench`, and so on. The deployment environment configuration is located in `[deploy]` in `config.toml`:

| Parameter | Required | Description |
|------|------|------|
|`path_prefix`|Optional|Deployment root directory, used to override the default system PATH. If not set, the system PATH is used after stripping the msmodeling venv; same effect as `OPTIX_DEPLOY_PATH`|

```toml
# Enable only when the system PATH cannot find vllm, ais_bench, etc.
[deploy]
# path_prefix = "/path/to/custom-deploy-root"
```

### Result Storage and Fine-tuning

After the PSO stage, the tool selects several candidate groups from the results for the fine-tune stage; `pso_top_k` controls the number selected. The result storage and fine-tuning configuration is located in `[data_storage]` in `config.toml`:

| Parameter | Optional/required | Description |
|---|---|---|
|`store_dir`|Optional|Result CSV storage directory, default `store`.|
|`pso_top_k`|Optional|Number of candidate groups selected from PSO results; default is 0, which disables the feature. Selection logic: take the top-k with the smallest fitness; for deep fine-tuning scenarios, `3~5` is recommended.|

```toml
[data_storage]
pso_top_k = 0
```

### Log Detection

The tool monitors the running logs for anomaly information and distinguishes fatal errors from retryable errors to implement intelligent error handling and retry. Detectable error types include out of memory (OOM), device faults (NPU), network errors, and IO errors. Fatal errors (such as OOM or NPU faults) immediately stop the scheduler; retryable errors (such as network jitter or IO failures) trigger automatic retry (up to 3 times). The log detection configuration is located in `[health_check]` in `config.toml`:

| Parameter | Optional/required | Description |
|---|---|---|
|`log_snippet_length`|Optional|Length of the log snippet for showing error details. Range: 50-1000, default 200.|
|`service_errors.fatal_patterns`|Optional|Fatal error pattern list of the service framework, empty by default.|
|`service_errors.retryable_patterns`|Optional|Retryable error pattern list of the service framework, empty by default.|
|`benchmark_errors.fatal_patterns`|Optional|Fatal error pattern list of the benchmark tool, empty by default.|
|`benchmark_errors.retryable_patterns`|Optional|Retryable error pattern list of the benchmark tool, empty by default.|

```toml
[health_check]
log_snippet_length = 200

# Service framework: fatal errors (stop immediately, no retry)
[health_check.service_errors.fatal_patterns]
out_of_memory = ["out of memory", "OOM killed", "MemoryError"]
device_error = ["NPU error", "device fault", "Ascend error"]

# Service framework: retryable errors (retry up to 3 times)
[health_check.service_errors.retryable_patterns]
network_error = ["connection reset", "connection refused", "timeout"]
io_error = ["file not found", "permission denied", "IO error"]

# Benchmark tool: fatal errors (stop immediately, no retry)
[health_check.benchmark_errors.fatal_patterns]
out_of_memory = []
device_error = []

# Benchmark tool: retryable errors (retry up to 3 times)
[health_check.benchmark_errors.retryable_patterns]
network_error = []
io_error = []
```

> All pattern lists are empty (`[]`) by default, meaning no detection. It is recommended to fill in `service_errors` first, then add `benchmark_errors` based on the actual errors of the benchmark tool; avoid overly broad patterns such as `"error"`, `"exception"`, or `"failed"`, and prefer complete error strings (for example `"out of memory"` instead of `"memory"`) to avoid false matches.

</details>

## Command Parameters

### Command Format

```bash
msmodeling optix [options]
```

### Notes

- Before starting optimization, confirm that `vllm` or `mindie` and `ais_bench` or `vllm_benchmark` can run properly in the system deployment environment and are not deployed inside the msmodeling virtual environment.
- The model path, port, dataset path, and service startup parameters in the configuration file such as `config.toml` must be consistent with the actual deployment environment. See [Configuration File Description](#configuration-file-description).
- Service Parameter Optimization repeatedly starts the service framework and runs benchmarks; it usually takes a long time (4-10h). It is recommended to run it in a dedicated or resource-stable environment.
- When environment isolation is abnormal, the log prefix `[optix/env]` gives the cause and fix suggestions. See [Environment Variables and Troubleshooting](#environment-variables-and-troubleshooting).

### Parameter Description

| Parameter | Optional/required | Description |
|---|---|---|
|-e or --engine|Optional|Inference framework: <br>&#8226;`vllm`: use vLLM as the inference framework<br>&#8226;`mindie`: use MindIE as the inference framework<br/>Default: `vllm`.|
|-b or --benchmark_policy|Optional|Benchmark tool: <br>&#8226;`vllm_benchmark`: use vllm_benchmark as the test tool <br/>&#8226;`ais_bench`: use AISBench as the test tool<br/>Default: `ais_bench`. Select the appropriate inference framework and test framework.|
|-c or --config|Optional|Custom configuration file path (TOML). Three forms are supported: <br>&#8226;Absolute path: use the specified path directly;<br>&#8226;Relative path (with a directory separator): resolved relative to the current working directory;<br>&#8226;Filename only: looked up in the current working directory.<br/>If not specified, the tool searches the configuration file in the preset path order. The specified file must be valid TOML and has the highest configuration priority.|
|-lb or --load_breakpoint|Optional|Controls whether to resume the optimization from a breakpoint; configuring this parameter enables it, not configuring it disables it by default.|
|--backup|Optional|Decides whether to back up data during optimization; configuring this parameter enables backup. Values: <br>&#8226;True: enable backup<br>&#8226;False: disable backup.<br/>Default: `False`.|

### Usage Examples

<details open>
<summary>vLLM</summary>

For the vLLM scenario with the ais_bench benchmark tool:

```bash
msmodeling optix -e vllm
```

For the vLLM scenario with the vllm_benchmark benchmark tool:

```bash
msmodeling optix -e vllm -b vllm_benchmark
```

</details>

<details>
<summary>MindIE</summary>

For the MindIE scenario with the ais_bench benchmark tool:

```bash
msmodeling optix -e mindie
```

</details>

**Specify a custom configuration file**:

```bash
# Absolute path
msmodeling optix -c /data/configs/my_config.toml

# Relative path
msmodeling optix -e vllm -b vllm_benchmark -c ../configs/vllm_config.toml

# Filename in the current directory
msmodeling optix -c my_config.toml
```

> To set environment variables that act on the vLLM/MindIE service, just set them before running the tool (for example `export ASCEND_RT_VISIBLE_DEVICES=0`); the tool applies them automatically during optimization.

## Output File Description

Each row in the output CSV corresponds to one parameter combination, and the first several columns are performance indicators. You can filter the performance rows that meet the requirements and set the VLLM/MindIE parameters and the vllm_benchmark/AISBench parameters to the values in the CSV.

| Field | Description |
|---|---|
|`generate_speed`|Throughput.|
|`time_to_first_token`|TTFT latency, in seconds.|
|`time_per_output_token`|TPOT latency, in seconds.|
|`success_rate`|Request success rate returned by the test.|
|`throughput`|Test throughput, in requests/second.|
|`CONCURRENCY`|Concurrency.|
|`REQUESTRATE`|Request rate.|
|`error`|Records the reason why the parameters did not execute normally, recorded when an error is sent.|
|`backup`|Data record address, recorded when `--backup` is enabled.|
|`real_evaluation`|Indicates whether the data is obtained from real test results. `false` means the data is predicted by the GP model.|
|`fitness`|Optimization value of the search algorithm; the smaller the value, the better the parameter combination.|
|`num_prompts`|Records the number of requests sent by the benchmark tool in this optimization round.|

The remaining columns are the corresponding VLLM or MindIE `config.toml` parameters.

## Appendix

### Environment Variables and Troubleshooting

**Environment variables**

| Variable | Description |
|------|------|
| `OPTIX_DEPLOY_PATH` | Optional. Deployment environment root directory, whose `bin/` must contain `vllm`, `ais_bench`, and so on. Has higher priority than `[deploy] path_prefix` in `config.toml`; if not set, the system PATH is used |
| `MIES_INSTALL_PATH` | MindIE installation root directory, kept by the child process; no need to change it for isolation |

Priority from high to low: `OPTIX_DEPLOY_PATH`, `[deploy] path_prefix` in `config.toml`, then the system PATH after stripping the msmodeling venv.

**Daily startup**

When vLLM is already installed on the system, `OPTIX_DEPLOY_PATH` usually does not need to be set:

```bash
source /path/to/msmodeling/.venv/bin/activate
msmodeling optix -e vllm -b ais_bench
```

Set it when the PATH layout is special:

```bash
export OPTIX_DEPLOY_PATH=/path/to/custom-deploy-root
source /path/to/msmodeling/.venv/bin/activate
msmodeling optix -e vllm -b ais_bench
```

**`[optix/env]` log reference**

| Log | Meaning | Handling |
|------|------|------|
| `当前未检测到虚拟环境` | msmodeling not installed in a venv | Run `uv sync` in the repository root (it creates `.venv` automatically); do not install into the system Python, otherwise `torch`/`transformers` may break the deployment stack |
| `找不到部署命令：vllm` 或 `mindieservice_daemon` | No such command in the system PATH after stripping the venv | First confirm vLLM or MindIE is installed on the system; if necessary, set `OPTIX_DEPLOY_PATH` or `[deploy] path_prefix` |
| `命令 vllm 解析到 msmodeling 虚拟环境` | vllm mistakenly installed in the msmodeling venv | Run `pip uninstall vllm` in that venv and use the system vLLM |
| `部署命令 vllm → ...` 且路径在系统侧 | Normal | No action needed |

### Log Levels

The optimization tool uses [loguru](https://github.com/Delgan/loguru) to output structured logs. Each console line contains context fields such as `run_id` and `stage`. Set the log level **before** starting the tool:

```bash
export OPTIX_LOG_LEVEL=DEBUG
```

| Level | Visible content |
|---|---|
|`INFO` (default)|Milestones: baseline passed, service ready, iteration summary, best result; child process startup shows multi-line `command:` / `log:`|
|`DEBUG`|Parameter and configuration details, child process I/O; each line contains `file:line`; uncaught exceptions show full stack traces|
|`TRACE`|PSO particle-level details (fitness, particle position, per-evaluation parameters); same `file:line` as DEBUG|

VLLM/MindIE and benchmark logs are written to the result directory or `/tmp`. The startup log is a readable multi-line format:

```text
Starting service subprocess
  command: vllm serve model_path --host 127.0.0.1 --port 8080
  log: /tmp/ms_serviceparam_optimizer__abc123
```

When baseline fails, the CLI boundary outputs a single message containing `exit=`, `command:`, `log:` and the last few log lines. Before constructing plugins, the framework checks `PATH` against the `required_executable` declared by each type: a missing `-b` raises `BenchmarkUnavailableError`, and a missing `-e` raises `SimulatorUnavailableError`. Both happen before the optimization starts, without launching child processes or cleaning output directories.

### Troubleshooting

| Symptom | Possible cause | Suggestion |
|---|---|---|
|Exit code `1`, `No feasible solution found`|Baseline or all PSO candidate fitness is `inf`|Check the CSV `error` column; use `OPTIX_LOG_LEVEL=DEBUG`; check the service and benchmark commands|
|`Optimizer aborted` with a stack trace|An uncaught fatal error in `main()`|Fix configuration, path, or health-check rules based on the boundary single traceback|
|`BenchmarkResultError` / AISBench CSV not unique|Zero or multiple `performances/*/*.csv` under the benchmark output directory|Clean the output directory; make sure each evaluation produces exactly one CSV; terminate the whole optimization immediately|
|Console only has `run_id`, `stage`, few details|Default `INFO` does not print particle-level logs|Set `OPTIX_LOG_LEVEL=TRACE` to inspect PSO internals|
|`BenchmarkUnavailableError` fails at startup|The CLI declared by the selected `-b` plugin is not in `PATH`|Install the benchmark CLI or switch `-b`; happens before the optimization starts|
|`SimulatorUnavailableError` fails at startup|The CLI declared by the selected `-e` plugin is not in `PATH`|Install the inference framework CLI or switch `-e`; happens before the optimization starts|
|`BaselineRunError` with `exit=` / `log:`|Baseline child process failed|First check the console tail logs; open the full log file if needed|
|`--config` points to a nonexistent file|Wrong path or file not deployed|Check the path; raises `ConfigFileNotFoundError` (exit code `1`)|

### CLI Exit Codes

| Exit code | Meaning |
|---|---|
|`0`|A feasible optimal solution was found and output completed|
|`1`|An `OptimizerError` subclass (`ConfigFileNotFoundError`, `BenchmarkResultError`, `NoFeasibleSolutionError`, `BaselineRunError`, etc.) or an uncaught fatal error|

> All failure paths currently exit with code `1`; distinguish the failure type by the log message or the `OptimizerError` subclass instead of relying on different non-zero exit codes. Invalid TOML raises `InvalidConfigError`. When the exit code is non-zero, combine the console log and the CSV `error` column for troubleshooting.

### Ternary Derived Type Usage Examples

#### Scenario 1: dp derived from the total number of cards

`tp` and `pp` are tunable parameters, and `dp` is derived from the total number of cards (16) (`dp = 16 ÷ (tp × pp)`):

> [!NOTE]
> `ternary_factories` requires the product of the dependent fields to derive the derived field legally. For `dtype = "int"`, `product` must be divisible by the product of the dependent fields, otherwise priority repair is triggered. Restricting the enum candidates of `tp` and `pp` so that the product is divisible by `product` is the best practice.

```toml
# Method 1 (best practice): restrict the enum candidates of tp and pp so that tp × pp ≤ 16
[[mindie.target_field]]
name = "tp"
config_position = "BackendConfig.ModelDeployConfig.ModelConfig.0.tp"
min = 0
max = 1
dtype = "enum"
dtype_param = [1, 2, 4, 8]   # tp max is 8

[[mindie.target_field]]
name = "pp"
config_position = "BackendConfig.ModelDeployConfig.ModelConfig.0.pp"
min = 0
max = 1
dtype = "enum"
dtype_param = [1, 2]          # pp limited to 1 or 2, so tp × pp max is 8 × 2 = 16
```

```toml
# Method 2: configure min_value as the lower-bound protection after repair failure, with a warning
[[mindie.target_field]]
name = "dp"
config_position = "BackendConfig.ModelDeployConfig.ModelConfig.0.dp"
min = 0
max = 0
dtype = "ternary_factories"
dtype_param = {target_names = ["tp", "pp"], product = 16, dtype = "int", min_value = 1}
# If no repairable legal combination exists and the result is below min_value, it falls back to min_value=1 and prints a WARNING
```

**Priority repair strategy (`priority_policy`)**

When the `tp`/`pp` combination generated by PSO cannot legally derive `dp` (for example not divisible or out of range), the system attempts to repair it. The repair strategy is controlled by `priority_policy`:

| Strategy | Semantics | Use case |
|--------|------|----------|
| `balanced` (default) | Splits particles into two groups: the first half repairs in `target_names` order, the second half in reverse order, reducing the structural bias of a single decoding order | No explicit field preference; default |
| `fixed` | User explicitly specifies the repair order: high-priority fields stay as unchanged as possible, low-priority fields are adjusted first | The user knows which field should be more stable |

```toml
# balanced (default) strategy example
# Use case: the user does not specify which field is more important; the system distributes the repair direction evenly
[[mindie.target_field]]
name = "dp"
config_position = "BackendConfig.ModelDeployConfig.ModelConfig.0.dp"
min = 0
max = 0
dtype = "ternary_factories"
dtype_param = {
  target_names = ["tp", "pp"],
  product = 32,
  dtype = "int",
  priority_policy = "balanced"   # default is balanced; can be omitted
}
```

```toml
# fixed strategy example
# Use case: the user knows tp should stay stable, adjust pp first
[[mindie.target_field]]
name = "dp"
config_position = "BackendConfig.ModelDeployConfig.ModelConfig.0.dp"
min = 0
max = 0
dtype = "ternary_factories"
dtype_param = {
  target_names = ["tp", "pp"],
  product = 32,
  dtype = "int",
  priority_policy = "fixed",
  priority = ["tp", "pp"]        # tp has high priority: keep tp as much as possible, adjust pp first
}
# Example: tp=8, pp=3 (invalid):
#   stage1: fix tp=8, find the nearest legal value in pp candidates → pp=4, dp=1
#   if stage1 fails, stage2: both fields can be adjusted, searched in ascending distance order
```

> [!note] priority_policy notes
>
> - `balanced` is the default strategy and takes effect automatically when not configured.
> - Repair runs in two stages: stage1 fixes the high-priority field and adjusts the low-priority field; if stage1 fails, stage2 can adjust both fields.
> - When all candidates are illegal, the repair fails and falls back to min/max truncation with a warning.

Scenario 2: `seq_len` and `prefill_batch_size` are tunable parameters, and `max_prefill_tokens` is automatically set to twice their product (`max_prefill_tokens = 2 × seq_len × prefill_batch_size`):

```toml
[[mindie.target_field]]
name = "seq_len"
config_position = "BackendConfig.ModelConfig.seqLen"
min = 0
max = 1
dtype = "enum"
dtype_param = [512, 1024, 2048, 4096]

[[mindie.target_field]]
name = "prefill_batch_size"
config_position = "BackendConfig.ScheduleConfig.maxPrefillBatchSize"
min = 1
max = 16
dtype = "int"

[[mindie.target_field]]
name = "max_prefill_tokens"
config_position = "BackendConfig.ScheduleConfig.maxPrefillTokens"
min = 0         # set to 0 to make it a constant, not searched
max = 0
dtype = "ternary_times"
dtype_param = {target_names = ["seq_len", "prefill_batch_size"], product = 2, dtype = "int"}
# seq_len=1024, prefill_batch_size=4 → max_prefill_tokens = 2 × 1024 × 4 = 8192
```
