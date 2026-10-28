# OptiX PSO PD Disaggregation Search: vLLM + AISBench Operation Guide 

This document uses **DeepSeek-V4-Flash + vLLM + AISBench** as an example to describe the PD disaggregation parameter search, the instance ratio, and how to benchmark the complete service. You can directly copy the [search configuration](../use_cases/examples/optix_pd_disaggregation/search.toml) and the [complete service benchmark configuration](../use_cases/examples/optix_pd_disaggregation/fine_tune.toml), and modify them as described in the following steps.

The operation sequence is as follows: **search the parameters of P (which processes the input) and D (which generates the subsequent tokens) separately → deploy the PD service and the Proxy based on the results → benchmark the complete service through the Proxy**. If a working PD service is already available and you only need to adjust the benchmark load, start directly from [Step 5](#step-5-benchmarking-the-complete-pd-service-through-the-proxy).

## Step 1: Confirming the Example Configuration and Preparing the Environment

This example uses the following model, serving parameters, and benchmark load:

| Item | Value Used in This Example |
|---|---|
| Model weight directory | `/dl/weight/DeepSeek-V4-Flash-w8a8-mtp` |
| Externally served model name/search port | `dsv4`/8900 |
| Parallel strategy of a single instance | DP=1, TP=8, with EP enabled. Both P and D use this strategy. |
| Fixed serving parameters | Ascend quantization, MTP 3 tokens, maximum model length 196608, memory utilization 0.9, and block size 128. For the graph compilation and `additional-config` settings, see `others` in the search template. |
| P serving parameters | `MAX_NUM_BATCHED_TOKENS`: 8192 to 65536, initial value 8192. `MAX_NUM_SEQS`: 24 or 32, initial value 32. |
| D serving parameters | `MAX_NUM_BATCHED_TOKENS`: 4096 or 8192, initial value 8192. `MAX_NUM_SEQS`: 32 to 512, initial value 64. |
| Concurrency candidates | 20, 24, 28, and 32 for both P and D, initial value 24 |
| Number of test requests | Concurrency × 4, that is, 80, 96, 112, and 128, initial value 96 |
| Request rate | `REQUESTRATE=0`, with no limit on the sending rate |
| Dataset | `gsm8k_gen_0_shot_cot_str_perf` |
| Search scale | `n_particles=10` and `iters=4`, run separately for P and D |
| Latency targets | Time to first token (TTFT) of P: 3 seconds. Time per output token (TPOT) of D: 0.05 seconds (50 milliseconds). |

**196608 is the maximum model length allowed by the service, not the input length of each GSM8K request.** This example uses the real GSM8K dataset.

This document uses **24 devices, with 8 devices for each P/D instance**, to demonstrate the instance ratio. Replace 24 with the total number of devices that are actually available.

### 1.1 Preparing the vLLM Deployment Environment

Perform the operations on a **Linux server or in a vLLM-Ascend container** that supports DSv4, and prepare the Ascend driver, CANN, vLLM-Ascend, and the model weights. First check the following:

```bash
npu-smi info
python3 --version
command -v vllm
vllm serve --help
```

The output should show the NPU devices, the Python version, and the vLLM help information. The `python3` used to install AISBench in the following steps must be Python 3.10 to 3.12. If vLLM is installed in a separate virtual environment, enter that deployment environment first. Do not enter the `.venv` of msmodeling.

This example includes quantization, MTP, and graph compilation settings. Therefore, vLLM-Ascend in the deployment environment must support these features and the startup parameters in the template.

### 1.2 Installing AISBench

Run `ais_bench --help` in the preceding deployment environment. If the help information is displayed, proceed directly to 1.3. If AISBench is not installed yet, create a separate environment for AISBench to avoid changing the dependencies of vLLM:

```bash
mkdir -p ~/optix-workspace
python3 -m venv ~/optix-workspace/aisbench-env
cd ~/optix-workspace
git clone https://github.com/AISBench/benchmark.git
cd benchmark
~/optix-workspace/aisbench-env/bin/python -m pip install -e . --use-pep517
~/optix-workspace/aisbench-env/bin/python -m pip install -r requirements/api.txt -r requirements/extra.txt
export PATH="$HOME/optix-workspace/aisbench-env/bin:$PATH"
ais_bench --help
```

After the source installation, keep the `benchmark` directory, because you need to modify the model and dataset configurations in it in Step 2.

### 1.3 Installing OptiX

OptiX is installed together with msmodeling and uses the `.venv` of the repository. Keep it separate from the preceding vLLM and AISBench deployment environments to prevent the dependency versions from affecting each other.

If `uv` is not installed yet, install it first:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version
```

If the msmodeling repository is not available yet, run:

```bash
mkdir -p ~/optix-workspace
cd ~/optix-workspace
git clone https://gitcode.com/Ascend/msmodeling.git
cd msmodeling
```

If the repository already exists, go directly to its root directory. Install and verify OptiX in that directory:

```bash
uv sync --python 3.12
uv run msmodeling optix --help
```

`uv sync` creates the `.venv` in the repository and installs OptiX. After the OptiX help information is displayed, continue with the following deployment command check. **Do not install vLLM and AISBench into this `.venv`.**

### 1.4 Confirming That OptiX Can Find the Deployment Commands

When OptiX starts services and runs benchmarks, it strips the `.venv` of msmodeling from the environment and then looks for `vllm` and `ais_bench`. Therefore, succeeding with `--help` in the current shell alone is not sufficient.

If the two commands are located in the system PATH, for example in `/usr/local/bin/`, no additional setting is required. If they are installed in a separate environment, for example in `/opt/vllm-env/bin/`, replace the directory in the following example with the root directory of the actual deployment environment, and then run:

```bash
export OPTIX_DEPLOY_PATH=/opt/vllm-env
export PATH="$OPTIX_DEPLOY_PATH/bin:$PATH"
```

For `OPTIX_DEPLOY_PATH`, enter `/opt/vllm-env`, not the path that ends with `bin`. The second line also enables the current shell to find vLLM. If AISBench is installed separately as described in 1.2, keep its `bin/` in PATH as well, because OptiX continues to look for `ais_bench` in PATH. Continue to use the current terminal for the following operations. If you open a new terminal, set these environment variables again.

Return to the root directory of the msmodeling repository and perform the check in the same way that OptiX actually looks for the commands. This step only checks the command paths and does not start a service:

```bash
uv run python - <<'PY'
from optix.deploy_env import resolve_deploy_context, validate_deploy_stack
ctx, env = resolve_deploy_context()
validate_deploy_stack(engine="vllm", benchmark="ais_bench", env=env, ctx=ctx)
PY
```

The log should show the paths of `deployment command vllm` and `deployment command ais_bench` respectively, and both should be outside the `.venv` of msmodeling. If `Deployment command not found` is reported, check whether the corresponding commands are installed in the deployment environment and whether `OPTIX_DEPLOY_PATH` is set correctly.

Run the following OptiX commands in the repository root directory. Place the Python configuration of AISBench in the **AISBench package directory in the deployment environment**. For a source installation, this is the corresponding configuration directory in the `benchmark` repository that you kept.

## Step 2: Preparing the AISBench Configurations for P and D

### 2.1 Copying the Model Configuration Provided by AISBench

First run the following in the deployment environment:

```bash
ais_bench --models vllm_api_stream_chat --datasets gsm8k_gen_0_shot_cot_str_perf --search
```

The output shows the actual paths of the model and dataset Python configurations. Enter the directory that contains `vllm_api_stream_chat.py` and make three copies:

```bash
cp vllm_api_stream_chat.py optix_dsv4_prefill.py
cp vllm_api_stream_chat.py optix_dsv4_decode.py
cp vllm_api_stream_chat.py optix_dsv4_proxy.py
```

The three files are used for P, D, and the final Proxy benchmark respectively. Use the API type defined in the copied files, and modify the service address and the benchmark load separately as follows.

### 2.2 Modifying the Fields in the New Files

Open the new files and modify the following fields in the model configuration in the `models` list. `model` is the **externally served model name**, while `path` is the **tokenizer/model directory**. The two are different.

| Field | `optix_dsv4_prefill.py` | `optix_dsv4_decode.py` | `optix_dsv4_proxy.py` |
|---|---|---|---|
| abbr | `optix-dsv4-prefill` | `optix-dsv4-decode` | `optix-dsv4-proxy` |
| path | `/dl/weight/DeepSeek-V4-Flash-w8a8-mtp` | The same as in `optix_dsv4_prefill.py` | The same model directory that the benchmark machine can read |
| model | `dsv4` | `dsv4` | The externally served model name of the Proxy, which is `dsv4` in this example |
| host_ip | `127.0.0.1` | `127.0.0.1` | The actual Proxy address |
| host_port | 8900 | 8900 | The actual Proxy port |
| max_out_len | **Change it to 1**. | Set the output length according to the target of this benchmark. | Keep it consistent with the D configuration. |
| batch_size | 24 | 24 | 24 |
| request_rate | 0 | 0 | 0 |

Write `batch_size` and `request_rate` as direct assignments such as `batch_size=24,` and `request_rate=0,`, and keep the trailing comma. OptiX updates them according to the `CONCURRENCY` and `REQUESTRATE` of the current candidate. If `url` in the file is not empty, update it to the corresponding endpoint as well. For the other fields, use the values in the copied files. For details, see the [official AISBench model configuration example](https://github.com/AISBench/benchmark/blob/master/ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py).

**The two benchmark rounds of D must use identical input requests.** Keep the dataset, the order, and the number of requests consistent. If random sampling is used, fix the request set first. The `--seed` of the service cannot replace this step. D and the Proxy use the same output length and sampling parameters, which you set according to the target of the current benchmark.

The `path` in the GSM8K dataset Python configuration should point to an available data directory, and you should verify it when you use the dataset for the first time. For details, see the [official AISBench GSM8K configuration](https://github.com/AISBench/benchmark/blob/master/ais_bench/benchmark/configs/datasets/gsm8k/gsm8k_gen_0_shot_cot_str_perf.py).

Finally, check whether the new short names can be resolved:

```bash
ais_bench --models optix_dsv4_prefill --datasets gsm8k_gen_0_shot_cot_str_perf --search
ais_bench --models optix_dsv4_decode --datasets gsm8k_gen_0_shot_cot_str_perf --search
ais_bench --models optix_dsv4_proxy --datasets gsm8k_gen_0_shot_cot_str_perf --search
```

For `--models`, enter the short name of the file, that is, the file name without `.py`. Do not enter an absolute path. `--search` only checks whether the configurations can be found. Before the formal search, you should also confirm that these configurations can send requests normally.

## Step 3: Copying the PD Disaggregation Template and Running the Search

```bash
cp docs/zh/use_cases/examples/optix_pd_disaggregation/search.toml ./pd_search.toml
```

The template already contains the model, the main startup parameters, and the complete search fields of P and D for this example. Check the following items in particular:

| Configuration Item | What to Check |
|---|---|
| `[pd_disagg.prefill.simulator_command_overrides]` | The model path, `served_model_name="dsv4"`, and `port="8900"` of P are consistent with the P configuration of AISBench. |
| `[pd_disagg.decode.simulator_command_overrides]` | The model path, model name, and port of D are consistent with the D configuration of AISBench. |
| `[pd_disagg] total_devices` | Change the example value 24 to the total number of devices available for the final deployment. |
| The two `devices_per_instance` entries in `[pd_disagg]` | Both P and D use 8 devices in this example, which corresponds to TP=8 in the template. |
| `[pd_disagg.prefill.benchmark_command_overrides] models` | P uses `optix_dsv4_prefill` created in Step 2. |
| `[pd_disagg.decode.benchmark_command_overrides] models` | D uses `optix_dsv4_decode` instead. |

**The serving startup parameters and benchmark configurations of P and D are already written in their respective phases.** To modify P, change the configuration under `pd_disagg.prefill`. To modify D, change the configuration under `pd_disagg.decode`. The `[vllm.command]` and `[ais_bench.command]` at the beginning of the file are used for plugin initialization. Keep them consistent with the model, address, and short name of P.

**The `target_field` of each phase is a complete list and does not automatically merge the common search fields.** The template already writes the complete `MAX_NUM_BATCHED_TOKENS`, `MAX_NUM_SEQS`, `CONCURRENCY`, `NUM_PROMPTS`, and `REQUESTRATE` under P and D respectively.

Use one `[[pd_disagg.<phase>.target_field]]` block for each parameter. `min` and `max` are the lower and upper limits of the search, `dtype_param` gives the enumeration candidates or the linkage rule, and `value` is the initial value. The two serving parameters use `config_position="run"`, and the benchmark parameters use `env`. `NUM_PROMPTS` is calculated as concurrency × 4 through `times`. When you add a parameter, append a block in the same format under the corresponding phase and keep the other existing fields.

The template already distinguishes the two phases. P outputs 1 token and is scored by TTFT. D enables prefix caching and runs two consecutive rounds, where the first round is the warm-up and the second round is scored by TPOT. `top_k=0` skips the candidate fine-tuning, but the optimal result is still verified and the ratio is still recommended.

The `others` of the two phases keep the model startup parameters, and the two batch parameters are generated from their respective search fields. `others` **replaces the entire startup parameter string**. Therefore, when you modify it, keep the complete parameters required by that phase.

The search requires a group of 8 available devices, and P and D run in sequence. `total_devices` only participates in the final ratio calculation and does not allocate specific device IDs. If you need to limit the Ascend device IDs, add the following at the end of the file:

```toml
[vllm.env]
ASCEND_RT_VISIBLE_DEVICES = "0,1,2,3,4,5,6,7"
```

After you save the file, run:

```bash
uv run msmodeling optix --mode pd_disagg -e vllm -b ais_bench -c ./pd_search.toml
```

The 10 × 4 search scale in this example is used for both phases. In addition, there is overhead for the baseline, the verification, and the D cache warm-up.

The results are stored in `result/pd_dsv4/pd_disagg/<run_id>/`. Replace `<run_id>` with the actual directory name. Open the following three files first:

| File | What to View |
|---|---|
| pd_disagg_summary.json | Whether `status` is `service_search_completed`, and how many instances `instances.prefill` and `instances.decode` each recommend deploying |
| prefill/phase_result.json | The parameters, actual TTFT, success rate, and request throughput of P |
| decode/phase_result.json | The parameters, actual TPOT, success rate, and request throughput of D |

After the search is complete, also confirm whether the TTFT of P meets 3 seconds, whether the TPOT of D meets 0.05 seconds, and whether the request success rate is 1. A completed status does not mean that the latency targets are met.

**When the number of devices is insufficient, the theoretical ratio and the actual deployment recommendation differ.** Assume that a single P processes 30 req/s and a single D processes 10 req/s (this is only for demonstration, not a measurement of DSv4):

| Comparison Item | Theoretical Balanced Solution | Recommendation with the Current Resources |
|---|---|---|
| Number of instances | 1P + 3D | 1P + 2D |
| Required devices (8 devices per instance) | 32 devices | 24 devices |
| Processing capability of both sides | P: 30 req/s, D: 30 req/s in total | P: 30 req/s, D: 20 req/s in total, and D becomes the bottleneck |

`pd_ratio=1/3` in the summary indicates the theoretical ratio. `instances.prefill=1` and `instances.decode=2` indicate that you are advised to deploy **1P + 2D** with the current resources. The results keep both ratios, but the tool does not automatically generate a textual explanation such as "32 devices are theoretically required while only 24 are available".

Prefer the feasible combination closest to the theoretical ratio. The actual throughput of the complete service still needs to be verified in Step 5.

## Step 4: Deploying P, D, and the Proxy Based on the Results

Use the PD deployment solution for your current environment to configure the P/D instances, the KV cache transmission, and the Proxy. The OptiX search results provide the parameters and the numbers, while the deployment tool still starts the complete cluster.

| Deployment Item | How to Fill It in |
|---|---|
| Number of P/D instances | Use `instances.prefill` and `instances.decode` from the summary. |
| Fixed parameters of a single instance | Keep the parameters used in this search, such as TP=8, quantization, and MTP, and add the PD communication parameters required by the deployment solution. |
| Batch parameters | Use the `MAX_NUM_BATCHED_TOKENS` and `MAX_NUM_SEQS` from the search results of P and D respectively. For the other parameters, use the startup configuration of that phase. |
| Device IDs and ports | Assign devices and independent ports to each instance, and configure the routing in the Proxy. |

For example, `MAX_NUM_BATCHED_TOKENS=16384` of P corresponds to `--max-num-batched-tokens 16384`, and `MAX_NUM_SEQS=64` of D corresponds to `--max-num-seqs 64`. `CONCURRENCY`, `NUM_PROMPTS`, and `REQUESTRATE` are used for benchmarks and are not added to the vLLM startup command. The concurrency of the complete service still needs to be verified through the Proxy in Step 5.

If the recommendation is `1P + 2D`, you can assign devices 0 to 7 to P1, devices 8 to 15 to D1, and devices 16 to 23 to D2. This device ID assignment is only for demonstration.

Complete one normal inference through the Proxy before you continue with the benchmark. For example, if the externally served model name of the Proxy is still `dsv4`, you can replace the following example address and verify it:

```bash
curl http://192.0.2.10:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"dsv4","messages":[{"role":"user","content":"Hello"}],"max_tokens":16}'
```

## Step 5: Benchmarking the Complete PD Service Through the Proxy

Even if you reuse a deployed service, this step still checks whether `vllm` and `ais_bench` are available at startup. Therefore, you need to complete the deployment command check in Step 1 first.

Open `optix_dsv4_proxy.py` created in Step 2, change `host_ip` and `host_port` to the IP address and port of the Proxy, and enter the model name that the Proxy provides externally in `model` (`dsv4` in this example). If `url` in the file is not empty, update it to the actual API address of the Proxy as well.

For `path`, enter the same model/tokenizer directory that the benchmark machine can read. Keep `max_out_len` and the sampling parameters consistent with the D configuration in Step 2.

**The benchmark address of AISBench is determined by this Python file. Modifying `[vllm.command] host/port` in the TOML file does not automatically synchronize the change to AISBench.**

Then copy the configuration and run:

```bash
cp docs/zh/use_cases/examples/optix_pd_disaggregation/fine_tune.toml ./pd_fine_tune.toml
uv run msmodeling optix --mode standard -e vllm -b ais_bench -c ./pd_fine_tune.toml
```

This step uses the deployed service, skips PSO, and starts fine-tuning from a concurrency of 24 and 96 requests. The request rate is fixed at 0, that is, **no limit on the sending rate**. In this step, `CONCURRENCY.dtype` is `int`, and the concurrency is adjusted within the integer range of 20 to 32 for at most 10 fine-tuning rounds. Step 3, by contrast, uses the concurrency candidates 20, 24, 28, and 32. In both steps, the number of requests changes with the concurrency by a factor of 4. This step checks TTFT and TPOT, and does not search a limited request rate.

View the measured rows (`real_evaluation=true`) in `result/pd_dsv4_fine_tune/store/data_storage_*.csv`:

| Column Name | How to Judge It in This Example |
|---|---|
| success_rate | Equal to 1 |
| time_to_first_token(s) | No more than 3 seconds |
| time_per_output_token(s) | No more than 0.05 seconds |
| throughput(requests/s) | Among the measured rows that meet both the success rate and the latency targets, select the row with the higher throughput |
| CONCURRENCY, NUM_PROMPTS | Record the concurrency and the number of requests corresponding to this row |

If the targets are still not met when the concurrency drops to 20, no suitable load has been found in the current range. You can expand the concurrency range or configure a limited request rate instead. For details, see the [detailed description of PD disaggregation search](../use_cases/optix_pso_pd_disaggregation.md).

## FAQ

| Symptom | Check First |
|---|---|
| The deployment commands cannot be found, or they resolve to the msmodeling virtual environment | Confirm as described in Step 1 that the commands are installed outside the `.venv` of msmodeling. For a separate environment, set `OPTIX_DEPLOY_PATH` to the root directory of the environment that contains `bin/`, and then run the check command in 1.4. |
| A new model short name cannot be found | Check whether the file is in a model configuration directory that AISBench can search and whether the file name is unique. Run `--search` first. |
| The request is sent to an old address, or the model name is incorrect | Check `host_ip`, `host_port`, `model`, and any non-empty `url` in the model Python file. |
| The modified serving parameters do not take effect | Check whether the search field of the corresponding phase is `config_position="run"`, and check the actual startup command in the log. |
| The warm-up of D has no effect | Confirm that prefix caching is enabled, that the two rounds use the same inputs, order, and number of requests, and that the cache capacity is sufficient. |
| The steady-state statistics of D fail or no valid data is available | Check the AISBench version and the test duration. 80 to 128 requests do not guarantee a sufficient steady-state window. If necessary, increase `NUM_PROMPTS.dtype_param.product` of D and verify again. |
| `instances` is empty | Check `allocation_warning`, and check whether the total number of devices can be combined from 8 devices per instance. If devices may be left unused, you can set `use_full_device=false`. |
