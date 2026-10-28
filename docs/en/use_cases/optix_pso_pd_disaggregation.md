# Use Case: Searching for PD Disaggregation Serving Parameters and Instance Ratios with PSO

New to this feature? See the [Quick Start Guide](../quick_start/optix_pso_pd_disaggregation.md) first to copy the configuration, run the commands, and check the results of a concrete example. This document provides the complete parameter descriptions and advanced configurations.

When you deploy a Prefill-Decode (PD) disaggregation service, you need to determine two things: which parameters each Prefill (P) and Decode (D) instance uses, and how many instances of each to deploy. This document describes how to use OptiX to measure the processing capability of P and D separately, obtain the serving parameters and the instance ratio, and then adjust the concurrency and the request rate for the complete deployed service.

The whole procedure consists of three steps:

1. Run `pd_disagg`. OptiX starts vLLM in sequence, searches for the serving parameters of a single P instance and a single D instance, and calculates the ratio.
2. Based on the search results, use your existing deployment tool to start the P and D instances and the Proxy, and configure request forwarding and KV cache transmission.
3. Run the `standard` fine-tuning mode to benchmark the complete PD service through the Proxy and adjust the load parameters.

If the complete PD service is already deployed and you only need to adjust the benchmark concurrency and request rate, you can start directly from [Step 3](#step-3-fine-tuning-the-complete-pd-service).

## Preparing the Environment and Configuration Files

First install OptiX as described in [Environment Preparation and Installation](../user_guide/msmodeling_optix_user_guide.md#4-environment-preparation-and-installation), and confirm that `vllm` and the benchmark tool you choose are available in the runtime environment.

This document uses `vllm_benchmark` as an example, with 16 devices, 4 devices per P instance, and 2 devices per D instance, an input of 1024 tokens, an output of 256 tokens, and 500 requests per benchmark run. These values are used only to explain the configuration method. When you use the tool in practice, replace them with your own hardware, load, and latency targets. For the configuration template and other common fields, see [optix/config.toml](../../../optix/config.toml) and [Configuration File Description](../user_guide/msmodeling_optix_user_guide.md#7-configuration-file-description).

## Step 1: Searching for P/D Serving Parameters

### 1. Setting Common Serving Information and the Run Mode

Edit `pd_disagg_config.toml`:

```toml
optimizer_strategy = "pso"
output = "./result"
skip_pso = false
manage_simulator_lifecycle = true
use_request_rate_calibration = false

[vllm.command]
host = "127.0.0.1"
port = "8000"
model = "/path/to/model"
served_model_name = "pd-model"
others = ""

[vllm_benchmark.command]
dataset_name = "random"
```

Replace `/path/to/model` with the weight directory. This example assumes that OptiX and the vLLM used for the search run on the same machine. Ensure that the port is not in use.

| Parameter | Meaning of the Setting in This Example |
|---|---|
| skip_pso = false | Run the PSO serving parameter search. In `pd_disagg` mode, this value must be `false`. |
| manage_simulator_lifecycle = true | OptiX starts and stops the vLLM service used for the search. In `pd_disagg` mode, this value must be `true`. |
| use_request_rate_calibration = false | Do not add request rate calibration rounds. During the search, the concurrency can vary and the request rate uses the maximum value: if the range contains 0, use 0 (no rate limit). Otherwise, use `max`. This setting is also used to enable the Decode warm-up benchmark runs described later. |

### 2. Setting the Number of Devices and the Benchmark Parameters for Both Phases

In the same file, add or modify:

```toml
[pd_disagg]
total_devices = 16
prefill_devices_per_instance = 4
decode_devices_per_instance = 2
top_k = 3
use_full_device = true
phase_output_dir = "pd_disagg"

[pd_disagg.prefill]
engine = "vllm"
benchmark_policy = "vllm_benchmark"
n_particles = 8
iters = 4
benchmark_run_count = 1
ttft_slo = 2.0
ttft_penalty = 1
tpot_penalty = 0
fine_tune_mode = "pd_mixed"

[pd_disagg.prefill.simulator_command_overrides]
others = "--tensor-parallel-size 4 --no-enable-prefix-caching"

[pd_disagg.prefill.benchmark_command_overrides]
others = "--seed 1024 --num-prompts 500 --random-input-len 1024 --random-output-len 1"

[pd_disagg.decode]
engine = "vllm"
benchmark_policy = "vllm_benchmark"
n_particles = 12
iters = 6
benchmark_run_count = 2
tpot_slo = 0.05
ttft_penalty = 0
tpot_penalty = 1
fine_tune_mode = "pd_disaggregation"

[pd_disagg.decode.simulator_command_overrides]
others = "--tensor-parallel-size 2 --enable-prefix-caching"

[pd_disagg.decode.benchmark_command_overrides]
others = "--seed 1024 --num-prompts 500 --random-input-len 1024 --random-output-len 256"
```

The P phase sets the output length to 1 so that requests end after the first token, which mainly measures the processing capability of Prefill. The D phase enables prefix caching and runs two consecutive benchmark runs on the same group of candidates: the first run fills the cache, and the second run records the metrics. The two runs share one vLLM instance, and the warm-up result of the first round is not counted toward the search score or the ratio.

The two Decode rounds must issue identical requests. Therefore, the example sets `--seed 1024` in the **benchmark command**. The input and output lengths, the number of requests, the concurrency, and the request rate must also remain consistent. Setting the seed only in `vllm serve` cannot make the benchmark requests identical. Cache warm-up can reduce the effect of Prefill on the standalone D benchmark, but the performance of the complete PD pipeline still needs to be verified in Step 3.

| Parameter | Meaning and How to Modify It |
|---|---|
| total_devices | The total number of devices available for the final deployment. It is used to calculate the instance ratio and does not allocate specific device IDs. |
| prefill_devices_per_instance/decode_devices_per_instance | The number of devices used by each P/D instance. In this example, they correspond to `--tensor-parallel-size 4/2`. Set the total number of devices and these two parameters together. When all three are 0, only the theoretical ratio is calculated. |
| use_full_device | When set to `true`, only combinations that use up exactly the total number of devices are recommended. When set to `false`, remaining devices are allowed. |
| top_k | Controls the number of candidates retained in the phases and the number of fine-tuning rounds. The default value is 3. Setting it to 0 skips the fine-tuning in the P/D phases, but the optimal PSO result is still verified and the ratio is still calculated. When the device count constraint allows, an integer instance recommendation is still provided. |
| n_particles/iters | The number of PSO particles and iterations. Increasing them usually produces more candidate benchmark runs. In addition, the baseline, verification, and fine-tuning steps also add overhead. You can set these parameters separately for P and D. Items that are omitted inherit the top-level values with the same names. |
| benchmark_run_count | The number of benchmark runs for each group of candidates on the same service instance. The default value is 1. When it is greater than 1, the first N−1 runs are warm-up runs and the last run is scored. `use_request_rate_calibration=false` is also required. |
| ttft_slo/tpot_slo | The latency target of the current phase. In this example, P focuses on TTFT and D focuses on TPOT. 0.05 seconds is 50 milliseconds. |
| ttft_penalty/tpot_penalty | How latency affects the score. When set to 0, this item does not affect which candidate is better. In the example, P is not scored by TPOT and D is not scored by TTFT. |
| simulator_command_overrides | Modifies the vLLM startup configuration of the current phase, such as `port`, `model`, and `others`, without affecting the other phase. |
| benchmark_command_overrides | Modifies the benchmark command configuration of the current phase, such as the input and output lengths in the example. |
| phase_output_dir | The name of the P/D result directory, which is located in `output`. The default value is `pd_disagg`. |

A phase-level override of `others` replaces the entire string. If the common configuration still contains necessary startup parameters such as quantization, keep them in the `others` of both phases as well.

### 3. Configuring the Search Space for Serving Parameters and Loads

Continue to add the following search fields to `pd_disagg_config.toml`. P and D each have their own complete `target_field` list. After you configure a non-empty list for a phase, that phase uses this list and no longer merges the search fields of the common plugin. Therefore, you need to list the serving parameters, the concurrency, and the request rate.

```toml
[[pd_disagg.prefill.target_field]]
name = "MAX_NUM_BATCHED_TOKENS"
config_position = "run"
min = 8192
max = 65536
dtype = "int"
value = 8192

[[pd_disagg.prefill.target_field]]
name = "MAX_NUM_SEQS"
config_position = "run"
min = 8
max = 128
dtype = "int"
value = 32

[[pd_disagg.prefill.target_field]]
name = "CONCURRENCY"
config_position = "env"
min = 1
max = 256
dtype = "int"
value = 32

[[pd_disagg.prefill.target_field]]
name = "REQUESTRATE"
config_position = "env"
min = 0
max = 0
dtype = "float"
value = 0

[[pd_disagg.decode.target_field]]
name = "MAX_NUM_BATCHED_TOKENS"
config_position = "run"
min = 128
max = 4096
dtype = "int"
value = 512

[[pd_disagg.decode.target_field]]
name = "MAX_NUM_SEQS"
config_position = "run"
min = 32
max = 512
dtype = "int"
value = 64

[[pd_disagg.decode.target_field]]
name = "CONCURRENCY"
config_position = "env"
min = 1
max = 1000
dtype = "int"
value = 100

[[pd_disagg.decode.target_field]]
name = "REQUESTRATE"
config_position = "env"
min = 1
max = 1000
dtype = "float"
value = 100
```

The `value` of each field is the initial value used by the baseline. `min` and `max` define the adjustable range, and `dtype` defines the numeric type. `config_position="run"` writes the serving parameters into the vLLM startup command. `config_position="env"` is used here for the concurrency and the request rate, which the benchmark consumes. If you also use placeholders such as `$MAX_MODEL_LEN` in `others`, declare that field in the `target_field` of the corresponding phase as well.

In this example, the request rate of P is fixed and no rate limit is applied. The request rate of D keeps an adjustable range because when `top_k>0`, the fine-tuning in the D phase first measures the QPS without a rate limit and then adjusts the concurrency using that QPS as the request rate. The `REQUESTRATE` range of D must cover the expected QPS. If the measured value exceeds 1000, increase the upper limit. When `top_k=0`, this fine-tuning step is not performed and this range is not checked.

The `pd_disaggregation` fine-tuning of Decode adjusts the concurrency by TPOT first. After the concurrency can no longer change, if `ttft_penalty>0` is set, the request rate is adjusted by binary search to make the measured TTFT fall within [0.8 × `ttft_slo`, `ttft_slo`]. For example, if the TTFT exceeds the target at a request rate of 2.4 req/s, the rate is lowered to 1.2 first. If the TTFT is then below 0.8 times the target, 1.8 is tried next. When neither rate boundary has been found, the rate is halved or doubled first. After both boundaries are found, the midpoint is taken. After the next candidate rate is constrained by the configuration range and the data type, if the difference between it and the current rate does not exceed 3% of the larger of the two values, the process stops immediately and the benchmark of that candidate is not run. This tolerance is defined by `REQUEST_RATE_CONVERGENCE_TOLERANCE`. The process also ends when the rate range, the data type precision limit, or the maximum number of fine-tuning rounds is reached. Request rate convergence does not mean that the TTFT target is met. Therefore, the final metrics still need to be checked when the target interval is unreachable. This branch is not controlled by `step_size` or `slo_coefficient`. In this example, `ttft_penalty=0` for Decode. Therefore, request rate continuation is not enabled. This TTFT interval is the stop condition for request rate adjustment. The final recommendation is still selected from the candidates that meet the TPOT target by highest request throughput. Therefore, the recommended result is not guaranteed to fall within this TTFT interval.

### How to Modify the Configuration When Using AISBench

Change the `benchmark_policy` of both phases to `"ais_bench"`, and then change the preceding `benchmark_command_overrides` to the AISBench configuration. For example, for the P phase:

```toml
[pd_disagg.prefill.benchmark_command_overrides]
models = "your_prefill_model_config"
mode = "perf"
others = "--datasets your_prefill_dataset_config"
```

For the D phase, specify the corresponding configuration in `[pd_disagg.decode.benchmark_command_overrides]` in the same way. For `models` and `datasets`, enter the short names of the configurations that AISBench can resolve. `datasets` is passed through `others`. The Python model configuration of AISBench also needs the correct service address, port, and served model name. If you use `ais_bench` and need to enable steady-state testing, you can append `--summarizer stable_stage` to `others` in `[ais_bench.command]`, for example, `others = "--datasets your_dataset_config --summarizer stable_stage"`.

Set the output length of P to 1 token. The model and dataset configurations of D should ensure that the warm-up and the formal benchmark runs generate identical requests. The configuration method of AISBench differs from that of `vllm_benchmark`. Do not reuse parameters such as `--random-input-len` directly. The OptiX command fields that you can override are `models`, `mode`, `work_dir`, and `others`.

Check whether the short names can be resolved in the environment where AISBench runs, and then start the search. Replace the following short names with your own configurations and check once for P and D respectively:

```bash
ais_bench --models your_prefill_model_config --datasets your_prefill_dataset_config --search
```

### 4. Starting the Search and Viewing the Results

After you save the configuration, run:

```bash
msmodeling optix --mode pd_disagg -c ./pd_disagg_config.toml
```

OptiX first completes the Prefill search, then the Decode search, and finally calculates the instance ratio. It reads the request throughput reported by the benchmarks of the two phases directly as the QPS, in req/s. If a phase does not return a valid throughput greater than 0, troubleshoot the benchmark results of that phase first.

In this example, the results are saved in `result/pd_disagg/<run_id>/`, where `<run_id>` is the identifier of this run. Focus on the following files:

| File | Content to View |
|---|---|
| pd_disagg_summary.json | The run status, the P/D QPS, the theoretical ratio `pd_ratio`, the recommended number of instances `instances`, and the artifact paths. On success, `status` is `service_search_completed`. |
| prefill/phase_result.json, decode/phase_result.json | The benchmark metrics of the two phases, as well as `service_params` used for deployment and `benchmark_params` used for the benchmark. |
| pd_ratio_candidates.csv | The QPS, the ratio, and the feasible instance allocation of different P/D candidate combinations, which helps you compare other combinations. |
| P/D phase CSV | The parameters and performance data of each candidate. You can read the specific path from the `artifacts` field of the summary. |

You can also format and view the summary directly:

```bash
python -m json.tool result/pd_disagg/<run_id>/pd_disagg_summary.json
```

`pd_ratio` indicates **the number of P instances / the number of D instances** and is calculated as follows:

```text
pd_ratio = QPS of a single D instance / QPS of a single P instance
```

For example, if a single P instance processes 20 requests per second and a single D instance processes 10 requests per second, then `pd_ratio=0.5`, that is, in theory each P instance is paired with two D instances. In this example, each P instance uses 4 devices and each D instance uses 2 devices. Therefore, with 16 devices you can deploy `2P:4D`, and the processing capability of both sides is 40 req/s. The numbers here are only used to demonstrate the calculation.

For the actual deployment, look at `instances.prefill` and `instances.decode`. The recommended combination first satisfies the device count constraint and then gets as close as possible to the theoretical ratio. `ratio_error` is the absolute value of the difference between the actual P/D ratio and the theoretical value. If a total device count is configured but `instances` is empty, check `allocation_warning` to see whether the device count per instance can be combined into the total device count, or whether you can disable `use_full_device`.

## Step 2: Deploying P and D Instances and the Proxy

After you obtain the search results, deploy in the following order:

1. Read `service_params` from the two `phase_result.json` files, apply the serving parameters to the corresponding P and D startup commands, and keep the fixed parameters in the phase configuration, such as TP and quantization.
2. Start the P and D instances in the numbers given in `instances`, and use your existing deployment tool to configure the Proxy, request routing, and KV cache transmission.
3. From the machine that runs OptiX, access the Proxy and confirm that requests complete normally before you proceed to the next benchmark step.

For example, if the Proxy exposes an OpenAI-compatible API, you can first view the served model name (replace the address and port, and include the corresponding request header when authentication is enabled):

```bash
curl http://<proxy-host>:<proxy-port>/v1/models
```

`benchmark_params` in the results of Step 1 records the load used in the single-phase tests. After the complete service goes through the Proxy and KV cache transmission, its capacity may be different. Therefore, you need to re-measure the concurrency and the request rate next.

## Step 3: Fine-Tuning the Complete PD Service

### 1. Specifying the Proxy Address and the Fine-Tuning Range

Edit `pd_fine_tune.toml`. Set the following top-level fields, and replace the corresponding vLLM and benchmark configurations:

```toml
optimizer_strategy = "pso"
output = "./result/pd_fine_tune"
skip_pso = true
manage_simulator_lifecycle = false
use_request_rate_calibration = false
fine_tune_mode = "pd_mixed"
max_fine_tune = 10
ttft_slo = 2.0
tpot_slo = 0.05

[vllm]
target_field = []

[vllm.command]
host = "pd-service-host"
port = "8000"
model = "/path/to/model"
served_model_name = "pd-model"
others = ""

[vllm_benchmark.command]
dataset_name = "random"
others = "--seed 1024 --num-prompts 500 --random-input-len 1024 --random-output-len 256"

[[vllm_benchmark.target_field]]
name = "CONCURRENCY"
config_position = "env"
min = 1
max = 1000
dtype = "int"
value = 100

[[vllm_benchmark.target_field]]
name = "REQUESTRATE"
config_position = "env"
min = 1
max = 1000
dtype = "float"
value = 100
```

Delete the existing `[[vllm.target_field]]` block from the template, keep the preceding `target_field=[]`, and define the load parameters only under benchmark. Replace `pd-service-host:8000` with the Proxy address that OptiX can access, and make `served_model_name` consistent with the model name that the Proxy provides externally.

When you use `vllm_benchmark`, fill in the address, port, model path, and served model name in `[vllm.command]`. When the configuration is loaded, these four items are copied to `[vllm_benchmark.command]`. Therefore, if you modify only the fields with the same names under benchmark, they are overwritten.

| Parameter | Role in This Step |
|---|---|
| skip_pso = true | Skip PSO and go directly to fine-tuning after the baseline benchmark. Keep `optimizer_strategy` as `"pso"`. |
| manage_simulator_lifecycle = false | Use the PD service that is already running. OptiX only runs the benchmark and does not start, monitor, or stop the service. |
| fine_tune_mode = "pd_mixed" | Adjust the concurrency of the complete service by TPOT and the request rate by TTFT. When the request rate is fixed and TTFT exceeds the target, the tool tries to lower the concurrency. |
| max_fine_tune | The maximum number of fine-tuning attempts. In this example, it is 10. When you check the benchmark process for the first time, you can reduce it. The algorithm may also end early. |
| ttft_slo/tpot_slo | The latency targets of the complete service, which you should fill in according to your service requirements. |
| CONCURRENCY.value/REQUESTRATE.value | In this example, the benchmark starts from a concurrency of 100 and a request rate of 100 req/s, and then adjusts within the respective `min` to `max` ranges. |

If you want to adjust only the concurrency, you can set the `min`, `max`, and `value` of `REQUESTRATE` to the same value. When they are all set to 0, the request rate is fixed and no rate limit is applied.

### 2. Running Fine-Tuning

Confirm that the Proxy is ready, and run:

```bash
msmodeling optix --mode standard -e vllm -b vllm_benchmark -c ./pd_fine_tune.toml
```

This phase first measures the initial load and then tries to adjust the concurrency and the request rate. The results are written separately to `result/pd_fine_tune/`. When you view the result CSV, judge whether your service targets are met based on TTFT, TPOT, the success rate, and the throughput. For the meaning of the fields, see [Output File Description](../user_guide/msmodeling_optix_user_guide.md#9-output-file-description). This process does not require you to install `contrib/optix/vllm_pd_simulator`.

## FAQ

| Symptom | Check Method |
|---|---|
| `pd_disagg` reports a lifecycle or PSO configuration error at startup | In the first phase, set `skip_pso=false` and `manage_simulator_lifecycle=true`. These settings are the opposite of those in Step 3. |
| The Decode configuration fails validation after two benchmark runs | Check whether the top-level `use_request_rate_calibration` is `false`. |
| The search ends but no integer instance is recommended | Check the three device count fields and `allocation_warning`. The total number of devices cannot necessarily be obtained by combining the device counts of a single P and a single D instance. |
| After you change the benchmark address, the tool still connects to the old port | When you use `vllm_benchmark`, change `[vllm.command]`. When you use AISBench, check its Python model configuration. |
| AISBench cannot find the model or dataset configuration | First run `ais_bench --models ... --datasets ... --search` with the corresponding short names to confirm that they can be resolved uniquely. |

For more log descriptions and troubleshooting methods, see the [appendix of the main guide](../user_guide/msmodeling_optix_user_guide.md#10-appendix).
