<a id="7-configuration-file-description"></a>

## 1 Configuration File Description

The optimization configuration file is located at `./msmodeling/optix/config.toml` by default. The file is organized by TOML sections, and each section corresponds to a feature:

| TOML Section | Purpose | Document Section |
| --- | --- | --- |
| Top level | Parameters required for optimization | [Optimization Parameters](#optimization-parameters) |
| Top-level `optimizer_strategy` | Optimization mode selection | [Optimization Modes: PSO and Agent](plugin_and_modes.md#6-optimization-modes-pso-and-agent) |
| `[vllm]`/`[mindie]` | vLLM/MindIE serving parameters and optimization fields | [Serving Parameters](#serving-parameters) |
| `[ais_bench.command]`/`[vllm_benchmark.command]` | Benchmark tool parameters | [Benchmark Tool Parameters](#benchmark-tool-parameters) |
| `[agent_optimizer]` | Agent mode run control (optional) | [Optimization Modes → How Agent Mode Works](plugin_and_modes.md#how-agent-mode-works) |
| `[deploy]` | Deployment environment root directory (optional) | [Advanced Configuration → Deployment Environment](#deployment-environment) |
| `[data_storage]` | Result storage and fine-tuning (optional) | [Advanced Configuration → Result Storage and Fine-Tuning](#result-storage-and-fine-tuning) |
| `[benchmark_early_exit]` | vLLM benchmark early exit policy (optional) | [Advanced Configuration → Benchmark Early Exit](#benchmark-early-exit) |
| `[health_check]` | Runtime log anomaly detection (optional) | [Advanced Configuration → Log Detection](#log-detection) |

<a id="optimization-parameters"></a>

### 1.1 Optimization Parameters

> **Note**: You are advised to fill in the following parameters explicitly in `config.toml`. Fields such as `n_particles`, `iters`, each penalty coefficient, `ttft_slo`, `tpot_slo`, and `service` have code default values. However, for the first run, you are advised to configure them explicitly according to the service latency requirements (SLOs) to avoid default values that do not meet expectations.

| Parameter | Optional/Required | Description |
| --- | --- | --- |
| n_particles | Required | Number of optimization seeds, that is, the number of parameter combinations generated in one group. Value range: an integer from 1 to 1000. You are advised to set it to 8 to 16. |
| iters | Required | Number of iteration rounds. Value range: an integer from 1 to 1000. You are advised to set it to 4 to 8. |
| ttft_penalty | Required | Timeout penalty coefficient for `time_to_first_token`, that is, the first token latency. If you have no latency requirement for `time_to_first_token`, set it to 0. Value range: [0, 100]. You are advised to set it to 1. |
| tpot_penalty | Required | Timeout penalty coefficient for `time_per_output_token`, that is, the non-first token latency. If you have no latency requirement for `time_per_output_token`, set it to 0. Value range: [0, 100]. You are advised to set it to 1. |
| success_rate_penalty | Required | Request success rate penalty coefficient. Value range: an integer from 1 to 1000. You are advised to set it to 5. |
| ttft_slo | Required | Latency limit of `time_to_first_token`. For example, if the limit of `time_to_first_token` is 2s, set it to 2. Value range: (0, 100], in seconds. |
| tpot_slo | Required | Latency limit of `time_per_output_token`. For example, if the limit of `time_per_output_token` is 50 ms, set it to 0.05. Value range: (0, 100], in seconds. |
| optimizer_strategy | Optional | Optimization mode. The value is `pso` (default) or `agent`. For details, see [Optimization Modes: PSO and Agent](plugin_and_modes.md#6-optimization-modes-pso-and-agent). |
| use_request_rate_calibration | Optional | Whether to perform real-world calibration on `REQUESTRATE`. The default is `true`. For details, see the following description. |
| benchmark_run_count | Optional | Number of benchmark runs for each candidate group in `standard` mode. The default is 1, and the value must be a positive integer. When the value is greater than 1, the first N−1 rounds are warm-up and the last round is scored. In this case, you must also set `use_request_rate_calibration=false`. |

For example, to warm up and then perform the formal measurement on the same service instance, set the following at the top level of the TOML file:

```toml
use_request_rate_calibration = false
benchmark_run_count = 2
```

All rounds use the same group of parameters. Keep the service running between rounds, and use only the metrics of the last round for scoring and result recording. This setting applies to the `standard` process of single-node and multi-node plugins, including the baseline, PSO candidates, validation, and fine-tuning. The `pd_disagg` mode still uses its own `benchmark_run_count` for each phase. When you simulate Decode, you also need to enable prefix caching and fix the request content, order, request count, and sampling parameters of each round.

Repeated stress testing requires the benchmark plugin to clear the results of the previous round before each round starts, or to use an independent results directory for each round, to ensure that only the metrics of the current round are read. When you use an external plugin, confirm that it meets this convention. Otherwise, the formal scoring may read the metrics of a warm-up round.

When `ttft_penalty > 0` and `tpot_penalty = 0`, the final selector chooses the group with the highest output token throughput from the verified candidates whose TTFT is strictly lower than `ttft_slo`. If no such candidate exists, it chooses the group whose TTFT is closest to the upper limit. This mode does not filter candidates by TPOT. Its SLO boundary rule is consistent with the existing TPOT-only and dual-constraint selectors.

**Two modes of `use_request_rate_calibration`**: This switch determines the behavior of the two special fields `CONCURRENCY` and `REQUESTRATE` in PSO.

| Value | Description | CONCURRENCY | REQUESTRATE |
| --- | --- | --- | --- |
| `true` (default) | With `CONCURRENCY` fixed, searches for a better `REQUESTRATE`. | Fixed to `max` and does not participate in the search. | First runs at the maximum stress rate, and then calibrates according to the measured throughput. |
| `false` | In the initial phase, searches for a better `CONCURRENCY`. | Participates in the PSO search within [min, max]. | Fixed to the maximum stress rate. |

`REQUESTRATE=0` means no rate limit. Therefore, the "maximum stress rate" takes a value according to the service semantics: if the range contains 0, the value is 0. Otherwise, the value is the configured numeric value `max`. In agent mode, this switch must be `false`. For the reason, see [Optimization Modes: PSO and Agent](plugin_and_modes.md#6-optimization-modes-pso-and-agent).

> [!NOTE]
> You can configure the number of seeds and the number of iterations according to the estimated time. The time consumed by a single seed is the time to start the service plus the time for the test data. For example, if it takes 15 min for you to start the service and complete the test, and you are willing to spend 8 hours on optimization, you can run about 50 seeds in total. You are advised to set `n_particles=8` and `iters=4` (the number of seeds is about twice the number of iterations).

<a id="benchmark-tool-parameters"></a>

### 1.2 Benchmark Tool Parameters

<details open>
<summary>Using the AISBench Benchmark Tool</summary>

<a id="aisbench-tool-parameters"></a>

### 1.3 AISBench Tool Parameters

When you use the ais_bench benchmark tool, modify the `[ais_bench.command]` parameters in `config.toml`. You can refer to [AISBench Quick Start](https://github.com/AISBench/benchmark/blob/master/docs/source_zh_cn/get_started/quick_start.md) for modification.

| Parameter | Description |
| --- | --- |
| models | Specifies the model task. You can configure it according to [Model Configuration Description](https://github.com/AISBench/benchmark/blob/master/docs/source_zh_cn/base_tutorials/all_params/models.md). |
| mode | Run mode. You can configure it according to [Run Mode Description](https://github.com/AISBench/benchmark/blob/master/docs/source_zh_cn/base_tutorials/all_params/mode.md). |
| others | Concatenates other parameters, such as `--num_prompts 1000`. Use spaces to separate parameters. The default is empty. |

</details>

<details>
<summary>Using the vllm_benchmark Benchmark Tool</summary>

<a id="vllm_benchmark-tool-parameters"></a>

### 1.4 `vllm_benchmark` Tool Parameters

When you use the vllm_benchmark benchmark tool, modify the `[vllm_benchmark.command]` parameters in `config.toml`:

| Parameter | Optional/Required | Description |
| --- | --- | --- |
| --mode | Optional | Optimization process. `standard` is the existing general optimization process, which can include PSO search and FineTune, and is also the default value. `pd_disagg` searches Prefill and Decode separately and then recommends a ratio. It is enabled only when you explicitly specify `--mode pd_disagg`. If you do not pass this parameter, the legacy optimization process is used. |
| host | Required | Host IP address. It must be consistent with the `host` in `[vllm.command]`. You can set it to `127.0.0.1`. |
| port | Required | Port number. It must be consistent with the `port` in `[vllm.command]`. |
| model | Required | Model path. It must be consistent with the `model` in `[vllm.command]`. |
| served_model_name | Required | Model name. It must be consistent with the `served_model_name` in `[vllm.command]`. |
| dataset_name | Required | Data set name. |
| others | Optional | Concatenates other parameters. Note that you must use spaces to separate parameters, and that parameters cannot contain spaces. For example, `--dataset-path /path/to/data --num-prompts 1000`. The default is empty. |

</details>

<a id="serving-parameters"></a>

### 1.5 Serving Parameters

<details open>
<summary>Using the vLLM Serving Framework</summary>

<a id="vllm-serving-parameters"></a>

### 1.6 vLLM Serving Parameters

When you use the vLLM framework, modify the `[vllm.command]` parameters in `config.toml`:

```toml
[vllm.command]
host = "127.0.0.1"
port = "8000"
model = "/workspace/vllm/models/llama-2-7b-chat-hf"
served_model_name = "llama-2-7b-chat-hf"
others = ""
```

| Parameter | Optional/Required | Description |
| --- | --- | --- |
| host | Required | Host IP address. It must be consistent with the `host` in `[vllm_benchmark.command]`. |
| port | Required | Port number. It must be consistent with the `port` in `[vllm_benchmark.command]`. |
| model | Required | Model path. |
| served_model_name | Required | Model name. |
| others | Optional | Appends fixed parameters. Use spaces to separate parameters. For example, `--tensor-parallel-size 2 --no-enable-prefix-caching`. The default is empty. |

<a id="vllm-custom-parameter-optimization"></a>

### 1.7 vLLM Custom Parameter Optimization

The optimization tool supports adding vLLM parameters to the optimization through `[[vllm.target_field]]`. Depending on how a parameter takes effect, there are two configuration methods:

- **vLLM environment variables**: Only declare them in `[[vllm.target_field]]` with `config_position = "env"`. The tool automatically writes an environment variable with the same name in uppercase before starting the service in each optimization round. You do not need to write it in `others` of `[vllm.command]`.
- **vLLM CLI parameters**: Declare them in `[[vllm.target_field]]` with `config_position = "run"`. The tool automatically renders the field name as a `--kebab-case` parameter with the same name and appends it to `vllm serve`. You do not need to reference it in `others`.

For a boolean switch with `config_position = "run"`, when the value of the field at execution is `false`, the switch with the same name in `others` is turned off. For example, `enforce_eager=false` removes `--enforce-eager`. This rule applies to vLLM serving parameter rendering in both PSO and Agent modes.

If you need to inject fixed environment variables into the vLLM service process, configure them by key/value in `[vllm.env]`:

```toml
[vllm.env]
ASCEND_RT_VISIBLE_DEVICES = "8,9"
```

These variables enter the vLLM child process environment directly and are not concatenated into CLI parameters. If a variable with the same name also appears in the `env` field of `[[vllm.target_field]]`, the optimization field overwrites the fixed value round by round.

> **Variable reference rule**: `others` is only used for fixed parameters. To explicitly expand a field of the environment variable type, use `$` followed by the field name in uppercase.

#### Example 1: vLLM Environment Variable Optimization

When the parameter to be optimized is itself a vLLM environment variable, only add it to `[[vllm.target_field]]`:

```toml
[[vllm.target_field]]
name = "VLLM_WORKER_MULTIPROC_METHOD"
config_position = "env"
dtype = "enum"
dtype_param = ["fork", "spawn"]
value = "fork"
```

#### Example 2: CLI Parameter Optimization

After you declare `config_position = "run"`, the tool automatically renders the field name as a `--kebab-case` parameter with the same name and appends it to `vllm serve`. You do not need to reference it in `others`.

Ordinary scalar parameter:

```toml
[[vllm.target_field]]
name = "GPU_MEMORY_UTILIZATION"
config_position = "run"
dtype = "enum"
dtype_param = [0.9, 0.91, 0.92]
value = 0.9
```

Generated:

```bash
--gpu-memory-utilization 0.9
```

JSON container parameter:

```toml
[[vllm.target_field]]
name = "COMPILATION_CONFIG"
config_position = "run"
dtype = "enum"
dtype_param = ["", '{"cudagraph_mode": "FULL_DECODE_ONLY"}']
value = '{"cudagraph_mode": "FULL_DECODE_ONLY"}'
```

Generated:

```bash
--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'
```

An empty string candidate indicates that the parameter is not appended. A TOML single-quoted literal can directly contain JSON double quotes. If you use a double-quoted string, write the double quotes inside the JSON as `\"`.

<a id="common-vllm-optimization-fields"></a>

### 1.8 Common vLLM Optimization Fields

| Field Name | `dtype` | Description |
| --- | --- | --- |
| MAX_NUM_BATCHED_TOKENS | int | Maximum number of tokens per batch (including prefill). It affects device memory and throughput. |
| MAX_NUM_SEQS | int | Maximum number of sequences per batch, that is, the upper bound of decode concurrency |
| CONCURRENCY | int | Benchmark concurrency. This is a special field. For its behavior, see the following description. |
| REQUESTRATE | float | Benchmark send rate. This is a special field. For its behavior, see the following description. |

> **Special behavior of `CONCURRENCY`/`REQUESTRATE`**: These two fields apply to both vLLM and MindIE. The tool rewrites them automatically according to `use_request_rate_calibration`. When it is `true`, `CONCURRENCY` is fixed to `max`, the tool first runs at the maximum stress rate, and then calibrates `REQUESTRATE` according to the measured throughput. When it is `false`, `CONCURRENCY` participates in the PSO search and `REQUESTRATE` is fixed to the maximum stress rate. `REQUESTRATE=0` indicates no rate limit: if the range contains 0, the maximum stress rate is 0. Otherwise, it is the numeric value `max`. When `min == max`, the field is regarded as a constant and does not participate in the search.

</details>

<details>
<summary>Using the MindIE Serving Framework</summary>

<a id="mindie-serving-parameters"></a>

### 1.9 MindIE Serving Parameters

When you use the MindIE framework, modify the `[mindie.command]` parameters in `config.toml`. You can refer to [MindIE Server Configuration Parameters](https://www.hiascend.com/document/detail/zh/mindie/20RC1/mindieservice/servicedev/mindie_service0285.html) for modification. For serving parameters, you can directly specify the parameter range. For example, to configure the optimization search space of `max_batch_size` as 10 to 400:

```toml
[[mindie.target_field]]
name = "max_batch_size"
config_position = "BackendConfig.ScheduleConfig.maxBatchSize"
min = 10
max = 400
dtype = "int"
```

You can also set a parameter to be related to another parameter. For example, `max_prefill_batch_size` is related to `max_batch_size` (`max_prefill_batch_size = ratio * max_batch_size`, `0 < ratio < 1`):

```toml
[[mindie.target_field]]
name = "max_prefill_batch_size"
config_position = "BackendConfig.ScheduleConfig.maxPrefillBatchSize"  # This value must not exceed maxBatchSize
min = 0.1
max = 0.7
dtype = "ratio"
dtype_param = "max_batch_size"
```

<a id="common-mindie-optimization-fields"></a>

### 1.10 Common MindIE Optimization Fields

| Field Name | `config_position` | `dtype` | Description |
| --- | --- | --- | --- |
| max_batch_size | BackendConfig.ScheduleConfig.maxBatchSize | int | Maximum batch size, the main throughput adjustment point |
| max_prefill_batch_size | BackendConfig.ScheduleConfig.maxPrefillBatchSize | ratio | Derived proportionally from `max_batch_size`. You are advised to set the ratio to 0.1 to 0.7. |
| prefill_time_ms_per_req | BackendConfig.ScheduleConfig.prefillTimeMsPerReq | range | Upper limit of the prefill duration per request (ms), enumerated by step |
| decode_time_ms_per_req | BackendConfig.ScheduleConfig.decodeTimeMsPerReq | range | Upper limit of the decode duration per request (ms), enumerated by step |
| support_select_batch | BackendConfig.ScheduleConfig.supportSelectBatch | bool | Whether to enable batch selection by duration. When enabled, `prefill/decode_time_ms_per_req` is automatically set to 0. |
| max_queue_delay_microseconds | BackendConfig.ScheduleConfig.maxQueueDelayMicroseconds | range | Maximum request queuing latency (μs), enumerated by step |
| max_preempt_count | BackendConfig.ScheduleConfig.maxPreemptCount | ratio | Preemption upper limit, derived proportionally from `max_batch_size` |
| CONCURRENCY | env | int | Benchmark concurrency. This is a special field. For details, see the description in the vLLM section. |
| REQUESTRATE | env | float | Benchmark send rate. This is a special field. For details, see the description in the vLLM section. |

</details>

#### `target_field` Configuration File Description

| Field | Meaning | Example |
| --- | --- | --- |
| name | Optimization field name. For environment variable fields, use uppercase. The tool automatically writes an environment variable with the same name in uppercase before starting the service in each round. For CLI fields, the tool automatically renders it as a `--kebab-case` parameter with the same name. | `COMPILATION_CONFIG` |
| config_position | Effective position: `env` (environment variable only, not concatenated into the CLI), `run` (CLI parameter, the tool automatically appends a `--kebab-case` parameter with the same name, no need to reference it in `others`), or the MindIE service configuration file path (such as `BackendConfig.ScheduleConfig.maxBatchSize`) | `run` |
| dtype | Value type, which determines the PSO sampling method: `int`/`float`/`bool`/`enum`/`range`/`ratio`/`share`/`factories`/`times`/`ternary_factories`/`ternary_times`. For the meaning of each type, see [`dtype` Types Supported by `target_field`](#dtype-types-supported-by-target_field). | `enum` |
| dtype_param | Depends on dtype: for `enum` it is the candidate value list, for `range` it is the step integer, and for `ratio` and derived types it is the dependent field configuration. | `["", '{"cudagraph_mode": "FULL_DECODE_ONLY"}']` |
| value | Initial value, used to generate baseline data | `'{"cudagraph_mode": "FULL_DECODE_ONLY"}'` |
| min/max | Lower and upper bounds of the search interval (effective for `int`/`float`/`range`). When `min == max`, the field is regarded as a constant and does not participate in the search. For `enum`, the actual candidates are given by `dtype_param`. For derived types, set both `min` and `max` to 0. | — |
| constant | Optional. After you set a value, the field is fixed to that value and does not participate in the PSO search (the tool writes it when it automatically rewrites `CONCURRENCY`/`REQUESTRATE`). | — |

#### `dtype` Types Supported by `target_field`

| Category | `dtype` | Meaning | `dtype_param` Format |
| --- | --- | --- | --- |
| Basic type | int | Takes an integer in [min, max] | — |
| Basic type | float | Takes a floating-point number in [min, max] | — |
| Basic type | bool | Boolean switch (true when the parameter value is greater than 0.5) | — |
| Basic type | enum | Selects a value from the candidate list (numeric or string values are supported) | Candidate value list, for example [1, 2, 4, 8] |
| Basic type | range | Enumerates values in [min, max] by a positive step, and includes max only when aligned with the step (for example, min=1, max=9, step=3 gives [1, 4, 7]) | Positive step integer, for example 10 |
| Binary derived | ratio | `int(ratio × target)` | Dependent field name (string), for example `"max_batch_size"` |
| Binary derived | share | `target.min + target.max - target.value` (complementary) | Dependent field name (string) |
| Binary derived | factories | `product ÷ target` | `{"target_name": "field name", "product": value, "dtype": "int"}` |
| Binary derived | times | `product × target` | `{"target_name": "field name", "product": value, "dtype": "int"}` |
| Ternary derived | ternary_factories | `product ÷ (field_a × field_b)` | `{"target_names": ["A", "B"], "product": value, "dtype": "int"}` |
| Ternary derived | ternary_times | `product × field_a × field_b` | `{"target_names": ["A", "B"], "product": value, "dtype": "int"}` |

> [!NOTE]
> The values of derived type fields (`factories`/`times`/`ternary_factories`/`ternary_times`) are derived automatically from the dependency relationship and **do not participate in the particle swarm search**. You must set both `min` and `max` to 0. If the value of any dependent field is 0 (in the division case) or `None`/`NaN` (in the multiplication case), the derivation is skipped for the current round, the field keeps its original value, and a warning log is output.
>
> For the complete usage of ternary derived types (`ternary_factories`/`ternary_times` and `priority_policy` priority repair), see [Appendix](#ternary-derived-type-examples).

<details>
<summary>Advanced Configuration</summary>

<a id="deployment-environment"></a>

### 1.11 Deployment Environment

When measured serving optimization starts vLLM, MindIE, or a benchmark tool, optix first removes variables related to the msmodeling virtual environment, and then locates the deployment root directory according to the following configuration. The `bin/` directory should contain `vllm`, `ais_bench`, and so on. The deployment environment configuration is located in `[deploy]` in `config.toml`:

| Parameter | Required | Description |
| --- | --- | --- |
| path_prefix | Optional | Deployment root directory, used to override the default system PATH. If it is not set, the tool directly uses the system PATH after stripping the msmodeling venv, which has the same effect as `OPTIX_DEPLOY_PATH`. |

```toml
# Enable this only when the system PATH cannot find vllm, ais_bench, and so on
[deploy]
# path_prefix = "/path/to/custom-deploy-root"
```

<a id="result-storage-and-fine-tuning"></a>

### 1.12 Result Storage and Fine-Tuning

When only the TTFT penalty is enabled (`ttft_penalty > 0` and `tpot_penalty = 0`), the final candidate comparison first selects the candidates whose TTFT is lower than the SLO, and then chooses the one with the highest generation throughput. If all candidates exceed the TTFT SLO, it chooses the one with the lowest TTFT. When FineTune is disabled, the tool stops the services that it manages itself after the retest. External services with `manage_simulator_lifecycle=false` are managed by the caller.

After the PSO phase ends, the tool selects several groups of candidates from the results to enter the fine-tuning phase. `pso_top_k` controls the number selected. The result storage and fine-tuning configuration is located in `[data_storage]` in `config.toml`:

| Parameter | Optional/Required | Description |
| --- | --- | --- |
| store_dir | Optional | Result CSV storage directory. The default is `store`. |
| pso_top_k | Optional | Number of candidate groups selected from the PSO results. The default is 0, which disables FineTune but still retests the best PSO candidate. Selection logic: take the top-k with the smallest fitness. For deep fine-tuning scenarios, you are advised to set it to 3 to 5. |

```toml
[data_storage]
pso_top_k = 0
```

<a id="benchmark-early-exit"></a>

### 1.13 Benchmark Early Exit

Benchmark early exit is used to terminate ordinary PSO candidates whose performance is significantly worse than historical complete results in advance, reducing the time of measured serving optimization. This feature is disabled by default and currently supports only `-e vllm`.

Early exit applies only to ordinary PSO evaluation. Baseline, request-rate calibration, refinement, and re-testing of the optimal candidate are not terminated in advance, ensuring that the baseline and final results are complete.

When you enable it for the first time, you are advised to first use `action = "report"` to verify the policy:

```toml
[benchmark_early_exit]
enabled = true
action = "report"
metrics_url = "http://127.0.0.1:8000/metrics"
```

| Parameter | Optional/Required | Default Value | Description |
| --- | --- | --- | --- |
| enabled | Optional | `false` | Whether to enable early exit. |
| action | Optional | `"terminate"` | `report` only records the time when early exit should have occurred and does not terminate the current Case. `terminate` actually stops the current benchmark and service. |
| metrics_url | Optional | `"http://127.0.0.1:8000/metrics"` | vLLM Prometheus metrics address. It should be consistent with the actual service listening address and port. |

After you enable it, the system first establishes a reference from the baseline or another complete Case. After a subsequent ordinary PSO candidate completes warmup, if several consecutive valid windows are significantly worse than the reference, the system generates an early exit decision.

You are advised to enable it as follows:

1. Set `action = "report"` and complete one optimization round.
2. Check `data_storage_*.csv` and `metrics_samples_*.csv`.
3. After confirming that there are no obvious misjudgments, change `action` to `"terminate"`.
4. Continue optimization with the same model, data set, and service load.

> [!NOTE]
> `report` mode does not shorten the run time. It is only used to verify the early exit decision and generate the `metrics_samples_*.csv` sampling trace. Only `terminate` mode actually saves time.

**Note:**

- Currently, only the `vllm` engine is supported. Extended engines such as `mindie` and `vllm_pd` are not supported.
- When a Case is short, no available reference exists, or `/metrics` is temporarily inaccessible, the system does not perform early exit.
- After you modify the model, the data set, or the service load, you should re-establish a complete reference using the same criteria.
- Early exit is used to identify candidates for which the service is normal but the performance is poor. It cannot replace handling of service startup failures, health checks, or request failures.

<a id="log-detection"></a>

### 1.14 Log Detection

Checks for exception information in logs, distinguishes fatal errors from retryable errors, and implements intelligent error handling and retry mechanisms. The detectable error types include out of memory (OOM), device faults (NPU), network errors, and IO errors. Fatal errors (such as OOM and NPU faults) immediately stop the scheduler, and retryable errors (such as network jitter and IO failures) trigger automatic retries (up to 3 times). The log detection configuration is located in `[health_check]` in `config.toml`:

| Parameter | Optional/Required | Description |
| --- | --- | --- |
| log_snippet_length | Optional | Log snippet length, used to display error details. Value range: 50 to 1000. The default is 200. |
| service_errors.fatal_patterns | Optional | List of fatal error patterns of the serving framework. The default is empty. |
| service_errors.retryable_patterns | Optional | List of retryable error patterns of the serving framework. The default is empty. |
| benchmark_errors.fatal_patterns | Optional | List of fatal error patterns of the benchmark tool. The default is empty. |
| benchmark_errors.retryable_patterns | Optional | List of retryable error patterns of the benchmark tool. The default is empty. |

```toml
[health_check]
log_snippet_length = 200

# Serving framework: fatal errors (stop immediately, no retry)
[health_check.service_errors.fatal_patterns]
out_of_memory = ["out of memory", "OOM killed", "MemoryError"]
device_error = ["NPU error", "device fault", "Ascend error"]

# Serving framework: retryable errors (retry up to 3 times)
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

> By default, all pattern lists are empty (`[]`), which means no detection. You are advised to fill in `service_errors` first, and then add `benchmark_errors` according to the actual errors reported by the benchmark tool. Avoid overly broad patterns such as `"error"`, `"exception"`, and `"failed"`. Prefer complete error strings (such as `"out of memory"` rather than `"memory"`) to avoid false matches.

</details>

<a id="ternary-derived-type-examples"></a>

### 1.15 Ternary Derived Type Examples

#### Scenario 1: `dp` Automatically Derived from the Total Number of Devices

`tp` and `pp` are adjustable parameters, and `dp` is automatically derived from the total number of devices (16) (`dp = 16 ÷ (tp × pp)`):

> [!NOTE]
> `ternary_factories` requires that the product of the dependent fields can legally derive the derived field. For `dtype = "int"`, `product` must be divisible by the product of the dependent fields. Otherwise, priority repair is triggered. Limiting the enum candidates of `tp` and `pp` so that the product is divisible by `product` is the best practice.

```toml
# Method 1 (best practice): Limit the enum candidates of tp and pp to ensure tp × pp ≤ 16
[[mindie.target_field]]
name = "tp"
config_position = "BackendConfig.ModelDeployConfig.ModelConfig.0.tp"
min = 0
max = 1
dtype = "enum"
dtype_param = [1, 2, 4, 8]   # The maximum value of tp is 8

[[mindie.target_field]]
name = "pp"
config_position = "BackendConfig.ModelDeployConfig.ModelConfig.0.pp"
min = 0
max = 1
dtype = "enum"
dtype_param = [1, 2]          # pp is limited to 1 or 2, ensuring that the maximum value of tp × pp, 8 × 2 = 16, is not exceeded

[[mindie.target_field]]
name = "dp"
config_position = "BackendConfig.ModelDeployConfig.ModelConfig.0.dp"
min = 0
max = 0
dtype = "ternary_factories"
dtype_param = {target_names = ["tp", "pp"], product = 16, dtype = "int"}
# Example: tp=4, pp=2 → dp = 16 ÷ (4 × 2) = 2
#        tp=8, pp=2 → dp = 16 ÷ (8 × 2) = 1
```

```toml
# Method 2: Configure min_value as the lower-bound protection after repair failure, and output a warning
[[mindie.target_field]]
name = "dp"
config_position = "BackendConfig.ModelDeployConfig.ModelConfig.0.dp"
min = 0
max = 0
dtype = "ternary_factories"
dtype_param = {target_names = ["tp", "pp"], product = 16, dtype = "int", min_value = 1}
# If there is no valid combination to repair and the result is lower than min_value, the value degrades to min_value=1, and a WARNING is output
```

**Priority Repair Policy (`priority_policy`)**

When the `tp` and `pp` combination generated by PSO cannot legally derive `dp` (for example, it is not divisible or is out of bounds), the system attempts to repair it. The repair policy is controlled by `priority_policy`:

| Policy Name | Semantics | Applicable Scenario |
| --- | --- | --- |
| `balanced` (default) | Divides particles into two equal groups: the first half is repaired in the order of `target_names`, and the second half is repaired in reverse order, reducing the structural bias caused by a single decoding order | Used by default when you have no explicit field priority preference |
| `fixed` | You explicitly specify the repair order: high-priority fields are kept unchanged as much as possible, and low-priority fields are adjusted first | Used when you clearly know which field should be more stable |

```toml
# Example of the balanced (default) policy
# Applicable when you have not specified which field is more important. The system automatically balances the repair directions
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
  priority_policy = "balanced"   # balanced is the default, and you can omit it
}
```

```toml
# Example of the fixed policy
# Applicable when you clearly know that tp should remain stable and pp should be adjusted first
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
  priority = ["tp", "pp"]        # tp has a high priority: keep tp as much as possible and adjust pp first
}
# Example: tp=8, pp=3 (invalid):
#   stage1: Fix tp=8 and find the nearest valid value in the pp candidates → pp=4, dp=1
#   If stage1 fails, proceed to stage2: both fields can be adjusted, and the search is performed in ascending order of distance
```

> [!note] priority_policy description
>
> - `balanced` is the default policy and takes effect automatically when it is not configured.
> - Repair is divided into two stages: in stage1, high-priority fields are fixed and low-priority fields are adjusted. If stage1 fails, both fields can be adjusted in stage2.
> - When all candidates are invalid, the repair fails, the value degrades to min/max truncation, and a warning is output.

Scenario 2: `seq_len` and `prefill_batch_size` are adjustable parameters, and `max_prefill_tokens` is automatically set to twice the product of the two (`max_prefill_tokens = 2 × seq_len × prefill_batch_size`):

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
min = 0         # Set it to 0 to make it a constant that does not participate in the search
max = 0
dtype = "ternary_times"
dtype_param = {target_names = ["seq_len", "prefill_batch_size"], product = 2, dtype = "int"}
# seq_len=1024, prefill_batch_size=4 → max_prefill_tokens = 2 × 1024 × 4 = 8192
```
