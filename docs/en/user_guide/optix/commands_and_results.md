<a id="8-command-parameters"></a>

## 1 Command Parameters

<a id="command-format"></a>

### 1.1 Command Format

```bash
msmodeling optix [options]
```

<a id="precautions"></a>

### 1.2 Precautions

- Before you start optimization, confirm that `vllm` or `mindie` and `ais_bench` or `vllm_benchmark` can run properly in the system deployment environment and are not deployed in the msmodeling virtual environment.
- The model path, port, data set path, and service startup parameters in the configuration file, such as `config.toml`, must be consistent with the actual deployment environment. For details, see [Configuration File Description](configuration.md#7-configuration-file-description).
- Measured serving optimization repeatedly starts the serving framework and runs benchmarks, which usually takes a long time (4 to 10h). You are advised to run it in an exclusive or resource-stable environment.
- When environment isolation is abnormal, the log prefix `[optix/env]` provides the reason and repair suggestions. For details, see [Environment Variables and Troubleshooting](troubleshooting.md#environment-variables-and-troubleshooting).

<a id="parameters"></a>

### 1.3 Parameters

| Parameter | Optional/Required | Description |
| --- | --- | --- |
| `-e` or `--engine` | Optional | Specifies the inference framework. Valid values:<br>&#8226;`vllm`: uses vLLM as the inference framework<br>&#8226;`mindie`: uses MindIE as the inference framework<br/>The default value is `vllm`. |
| `-b` or `--benchmark-policy` | Optional | Specifies the benchmark tool. Valid values:<br>&#8226;`vllm_benchmark`: uses vllm_benchmark as the test tool <br/>&#8226;`ais_bench`: uses AISBench as the test tool<br/>The default value is `ais_bench`.<br/>You must select the inference framework and test framework yourself. |
| `-c` or `--config` | Optional | Specifies a custom configuration file path (in the TOML format). The following three forms are supported:<br>&#8226;Absolute path: use the specified path directly.<br>&#8226;Relative path (containing a directory separator): resolve it relative to the current working directory.<br>&#8226;File name only: search for it in the current working directory.<br/>By default, this parameter is not specified, and the tool automatically searches for the configuration file in the preset path order.<br/>The specified file must be in valid TOML format and has the highest configuration priority. |
| `--load-breakpoint` | Optional | Controls whether to resume the optimization process from a breakpoint. If this parameter is configured, the feature is enabled. By default, it is not configured, which means the feature is disabled. |
| `--backup` | Optional | Determines whether to back up data during optimization. If this parameter is configured, backup is enabled. By default, it is disabled. |

<a id="examples"></a>

### 1.4 Examples

<details open>
<summary>vLLM</summary>

When you use the ais_bench benchmark tool in the vLLM scenario, see the following example:

```bash
msmodeling optix -e vllm
```

When you use the vllm_benchmark benchmark tool in the vLLM scenario, see the following example:

```bash
msmodeling optix -e vllm -b vllm_benchmark
```

</details>

<details>
<summary>MindIE</summary>

When you use the ais_bench benchmark tool in the MindIE scenario, see the following example:

```bash
msmodeling optix -e mindie
```

</details>

**Specifying a Custom Configuration File**:

```bash
# Absolute path
msmodeling optix -c /data/configs/my_config.toml

# Relative path
msmodeling optix -e vllm -b vllm_benchmark -c ../configs/vllm_config.toml

# File name in the current directory
msmodeling optix -c my_config.toml
```

> To set environment variables that take effect for the vLLM service, preferentially use `[vllm.env]`. To set environment variables that take effect for the MindIE service or the entire tool process, set them before you run the tool (for example, `export ASCEND_RT_VISIBLE_DEVICES=0`). The tool automatically inherits them during optimization.

<a id="built-in-two-step-pd-disaggregation-optimization"></a>

### 1.5 Built-In Two-Step PD Disaggregation Optimization

OptiX supports configuring `pd_disagg.full_service`. Through the existing `vllm_pd` plugin, it automatically chains one-side search, ratio recommendation, complete service deployment, and load fine-tuning. For details, see [Running the Complete Process with a Single Command](../../quick_start/optix_pso_pd_disaggregation.md#running-the-complete-process-with-one-command).

When you use it step by step, OptiX supports searching serving parameters for vLLM Prefill-Decode (PD) disaggregated deployment and recommending the P/D instance ratio. First, use `--mode pd_disagg` to search the P and D parameters separately. Then you deploy the complete PD service. Finally, use the ordinary `standard` mode to fine-tune the concurrency and request rate.

For the first run, you can copy the configuration and run it by following the concrete example in the [PD Disaggregation Optimization Quick Start Guide](../../quick_start/optix_pso_pd_disaggregation.md). For complete parameter descriptions, ratio interpretation, and advanced configuration, see [Use Case: OptiX Two-Step PD Disaggregation Optimization](../../use_cases/optix_pd_step_by_step.md). For other PD disaggregation or PD hybrid deployment optimization, you can use the existing plugins listed in [Plugin Support](plugin_and_modes.md#3-plugin-support).

<a id="9-output-file-description"></a>

## 2 Output File Description

Each row in the output CSV corresponds to one group of parameters. The first column `case_id` identifies the Case, and the following six columns are performance metrics. You can filter the performance rows that meet your requirements as needed, and change the vLLM/MindIE parameters and the vllm_benchmark/AISBench parameters to the data in the CSV.

| Field | Description |
| --- | --- |
| case_id | Case identifier that associates the main results with the metrics sampling trace. |
| generate_speed(tokens/s) | Output token throughput, in tokens per second. |
| time_to_first_token(s) | TTFT latency, in seconds. |
| time_per_output_token(s) | TPOT latency, in seconds. |
| success_rate | Request success rate, a dimensionless ratio from 0 to 1. |
| throughput(requests/s) | Test throughput, in requests per second. |
| input_token_throughput(tokens/s) | Input token throughput, in tokens per second. |
| CONCURRENCY | Concurrency. |
| REQUESTRATE(requests/s) | Send rate, in requests per second. |
| error | Records the reason why the parameters were not executed properly this time. It is recorded when a send error occurs. |
| backup | Data record address. It is recorded when `--backup` is enabled. |
| real_evaluation | Indicates whether the data is obtained from real test results. `false` indicates that the group of data is predicted by the GP model. |
| fitness | Optimization value of the optimization algorithm. The smaller the value, the better the effect of the group of parameters. |
| num_prompts | Records the number of requests sent by the benchmark tool in this optimization. |
| would_early_exit | Whether the early exit condition is met. It may also be `true` in `report` mode. |
| early_exit | Whether the current Case was actually terminated in advance. It may be `true` only in `terminate` mode. |
| early_exit_reason | Reason for triggering early exit. |
| early_exit_decision_elapsed_seconds(s) | Time from the start of the Case to the formation of the early exit decision, in seconds. |
| estimated_time_saved_seconds(s) | Estimated time that can be saved in `report` mode, in seconds. |
| estimated_time_saved_ratio | Proportion of the estimated saved time to the duration of a complete Case. |
| result_source | Result source. The actual early exit result is `early_exit_metrics`. |
| usable_as_best | Whether the result can participate in the selection of the optimal result and the reference. Only results that were actually terminated in advance are `false`. In `report` mode, Cases that meet the conditions run completely, and this field is still `true`. |

Other result columns not listed in the preceding table include internal sampling and warmup metadata for early exit (such as `metrics_window_*` and `warmup_*`). When early exit is not enabled, these fields are empty. They are followed by the corresponding vLLM or MindIE `config.toml` parameters.

When `action = "report"`, a `metrics_samples_*.csv` file is also generated. You can associate the sampling trace with the main results in `data_storage_*.csv` through `case_id`. Actual early exit results do not participate in the selection of the optimal candidate, the reference, or refinement.
