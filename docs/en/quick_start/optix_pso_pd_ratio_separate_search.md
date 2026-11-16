# OptiX PSO: Search P/D Parameters Separately and Calculate the Deployment Ratio

This document describes how to run PD co-deployment optimization in the separate deployment environments for P and D, measure the processing capability of Prefill (P) and Decode (D) separately with benchmark workloads, and then manually calculate the PD ratio.

It applies to the following two deployment methods, which can also be combined:

| Deployment Method | P-Side Operation Location | D-Side Operation Location | Service Plugin Used |
|---|---|---|---|
| Heterogeneous devices | The machine where the devices actually used for P reside | The machine where the devices actually used for D reside | Use `vllm` for a single machine and `multihost_infer` for multiple machines |
| Deploy one P/D service unit across multiple machines | The primary node of the P service group | The primary node of the D service group | Multi-machine PD co-deployment plugin `multihost_infer` |

Operation sequence: **Prepare the environments on both sides → Search the P/D parameters separately (for D, warm up each candidate group before scoring) → Confirm the P/D throughput → Calculate the ratio → Deploy the complete PD service and verify it**.

This document treats a deployment that can be started and stopped independently and provides a single service entry point externally as a "service unit". A single-machine service unit can occupy multiple devices on one machine. A multi-machine service unit contains the primary node and its workers. The QPS, number of devices, number of machines, and P:D ratio in the following sections are all counted for the entire service unit, and the DP, TP, and EP inside the unit reuse the configuration measured this time.

## Step 1: Preparing the P and D Environments and the Test Workload

### 1.1 Determine the Test Configuration

Record the deployment information for P and D separately, and determine the same set of service workloads:

| Item | How to Fill In |
|---|---|
| Model | Both sides use the same model. Record the weight path, quantization method, externally served model name, and tokenizer |
| Devices and topology | Record the device model, number of machines per unit, number of devices, DP/TP/EP, and other deployment parameters for both sides separately |
| Input requests | Fix the input request set, order, and sampling method. P and D use their corresponding input workloads |
| P output length | 1 token |
| D output length | The target service output length, for example 512 tokens |
| Latency target | P checks TTFT and D checks TPOT, for example 3 seconds and 0.05 seconds |
| Request rate | This document uses 0, that is, the send rate is not limited, and the processing capability is searched through concurrency |
| Number of requests | The example in this document is 512. Set it according to the actual test duration and cache capacity |
| Search scale | The example in this document is 8 particles and 4 iterations, and both sides can be set separately |

In the following commands, the model directory, port, number of devices, parallel strategy, and search range must all be replaced with actual values. **P and D can use different hardware, different parallel strategies, and different batching parameters, and the configuration of each side must remain consistent with the final deployment.**

### 1.2 Install OptiX and the Benchmark Tool

In the single-machine deployment environment or the multi-machine primary node environment of P and D, follow the [environment preparation steps](optix_pso_pd_disaggregation.md#11-preparing-the-vllm-deployment-environment) to prepare vLLM, AISBench, and OptiX, and complete the deployment command check in step 1 of that guide.

The subsequent `uv` commands are all run in the root directory of the msmodeling repository. The service and AISBench use their respective deployment environments.

For the multi-machine side, install the plugin separately:

```bash
uv pip install --python .venv/bin/python -e contrib/optix/multihost_infer
uv run --no-sync msmodeling optix --help
```

After you install the extension plugin, this document uniformly uses `uv run --no-sync` to run OptiX, which retains the plugins installed in the current environment.

### 1.3 Prepare the AISBench Configuration for P and D

Refer to [Prepare the AISBench configuration](optix_pso_pd_disaggregation.md#step-2-preparing-the-aisbench-configurations-for-p-and-d), copy the model configuration in the deployment environment of each side, and name them `optix_ratio_prefill.py` and `optix_ratio_decode.py`. Modify the following fields:

| Field | P Configuration | D Configuration |
|---|---|---|
| abbr | `optix-ratio-prefill` | `optix-ratio-decode` |
| path | The model/tokenizer directory readable on this side | The same model/tokenizer directory readable on this side |
| model | The externally served model name of the service on this side, for example `pd-model` | The externally served model name of the service on this side, for example `pd-model` |
| host_ip | The entry address on this side. On the primary node local machine, you can enter `127.0.0.1` | The entry address on this side. On the primary node local machine, you can enter `127.0.0.1` |
| host_port | Example in this document: 8900 | Example in this document: 8900 |
| max_out_len | 1 | The target output length, for example 512 in this document |
| batch_size | Initial concurrency, for example 32 in this document | Initial concurrency, for example 32 in this document |
| request_rate | 0 | 0 |

`batch_size` and `request_rate` use direct assignment forms such as `batch_size=32,` and `request_rate=0,`. When a non-empty `url` exists in the configuration, set it to the interface address on this side as well.

This document uses `gsm8k_gen_0_shot_cot_str_perf` for demonstration. Point the data path in its configuration to the fixed request set used this time, and both sides use the same version of the data and processing configuration. The warm-up and formal measurement of D also need to keep the request content, order, number of requests, concurrency, output length, and sampling parameters consistent.

Check short-name resolution on the corresponding machine:

```bash
# P side
ais_bench --models optix_ratio_prefill --datasets gsm8k_gen_0_shot_cot_str_perf --search
```

```bash
# D side
ais_bench --models optix_ratio_decode --datasets gsm8k_gen_0_shot_cot_str_perf --search
```

## Step 2: Configuring the Parameter Search for P and D Separately

### 2.1 P-Side Search Configuration

Create `prefill_search.toml` on the P side and fill in the following complete configuration. The example unit uses TP=8 and DP=1. Append fixed parameters required for deployment, such as EP, quantization, and MTP, to `others`.

```toml
optimizer_strategy = "pso"
n_particles = 8
iters = 4
output = "./result/pd_ratio_prefill"
skip_pso = false
manage_simulator_lifecycle = true
use_request_rate_calibration = false
benchmark_run_count = 1
fine_tune_mode = "pd_mixed"

ttft_slo = 3
tpot_slo = 0.05
ttft_penalty = 1
tpot_penalty = 0
success_rate_slo = 1
success_rate_penalty = 5

[data_storage]
pso_top_k = 0

[vllm.command]
host = "127.0.0.1"
port = "8900"
model = "/data/models/your-model"
served_model_name = "pd-model"
others = "--tensor-parallel-size 8 --data-parallel-size 1 --trust-remote-code --no-enable-prefix-caching"

[[vllm.target_field]]
name = "MAX_NUM_BATCHED_TOKENS"
config_position = "run"
min = 8192
max = 65536
dtype = "int"
value = 8192

[[vllm.target_field]]
name = "MAX_NUM_SEQS"
config_position = "run"
dtype = "enum"
dtype_param = [24, 32]
value = 32

[[vllm.target_field]]
name = "CONCURRENCY"
config_position = "env"
min = 16
max = 128
dtype = "enum"
dtype_param = [16, 32, 64, 128]
value = 32

[[vllm.target_field]]
name = "REQUESTRATE"
config_position = "env"
min = 0
max = 0
dtype = "float"
value = 0

[ais_bench.command]
models = "optix_ratio_prefill"
mode = "perf"
others = "--datasets gsm8k_gen_0_shot_cot_str_perf --num-prompts 512"
```

The P configuration disables prefix caching, outputs 1 token, and searches for throughput under the constraints of TTFT and success rate. `pso_top_k=0` is used to complete the PSO search and the optimal candidate verification, and candidate retesting is performed in step 4.

### 2.2 D-Side Search Configuration

On the D side, save the same configuration as the basis as `decode_search.toml`, replace the corresponding fields according to the following table, and keep the remaining configuration:

| Configuration Location | Value to Fill In on the D Side |
|---|---|
| Top-level `output` | `"./result/pd_ratio_decode"` |
| Top-level `benchmark_run_count` | 2, with the first round warming up and the second round scoring on the same service |
| Top-level `ttft_penalty`/`tpot_penalty` | 0/1 |
| Top-level `tpot_slo` | The target TPOT, 0.05 in this document |
| `[vllm.command] model` | The D-side model directory |
| `[vllm.command] others` | The fixed parallel parameters for D, with `--enable-prefix-caching` enabled |
| `MAX_NUM_BATCHED_TOKENS` field | `dtype="enum"`, `dtype_param=[4096,8192]`, `value=8192`. Delete the original `min` and `max` of this field |
| `MAX_NUM_SEQS` field | `dtype="int"`, `min=32`, `max=512`, `value=64`. Delete the original `dtype_param` of this field |
| `[ais_bench.command] models` | `"optix_ratio_decode"` |

Example of `others` for single-machine D:

```toml
others = "--tensor-parallel-size 8 --data-parallel-size 1 --trust-remote-code --enable-prefix-caching"
```

On the D side, each candidate group is automatically evaluated for two rounds on the same service, with the first round filling the cache and the second round recording the formal metrics. Keep `use_request_rate_calibration=false`, and fix the request content, order, number of requests, output length, and sampling parameters for both rounds. When you need more warm-up rounds, set `benchmark_run_count` to a larger positive integer, and the last round scores.

## Step 3: Running the Search on the Respective Devices

### 3.1 Heterogeneous Single-Machine Deployment

Run the following on the machine where the P devices reside:

```bash
uv run --no-sync msmodeling optix --mode standard -e vllm -b ais_bench \
  -c ./prefill_search.toml --backup
```

Run the following on the machine where the D devices reside:

```bash
uv run --no-sync msmodeling optix --mode standard -e vllm -b ais_bench \
  -c ./decode_search.toml --backup
```

The two sides can run independently. Keep the configuration files, device information, and the results under `result/pd_ratio_prefill/` and `result/pd_ratio_decode/` separately.

### 3.2 Use the Multi-Machine PD Co-Deployment Plugin

For a P or D service unit that requires multiple machines, perform this section on the primary node of that unit. When both sides use multiple machines, prepare two sets of cluster configurations separately and run them on their respective primary nodes.

First follow [multi-machine plugin environment preparation](../../../contrib/optix/multihost_infer/multihost_inference_optimization/README.md#环境准备) to complete the deployment:

1. The primary node runs on the same machine as OptiX. The primary node environment can log in to each worker host via passwordless SSH.
2. Each node has a consistent model directory and a working vLLM-Ascend environment, and the target model and parallel topology have been verified to perform inference normally.
3. Prepare the Ascend environment script and the `ascend-infer` conda environment according to the plugin-provided `template.sh`. When you customize the environment, adjust the paths and environment names in the template to actual values.
4. Complete the multi-node communication verification described in the plugin documentation.

Taking a P unit composed of two 8-device machines as an example, create `prefill_cluster.toml`:

```toml
docker_use_sudo = false

[vllm_mix]
chips_per_node = 8
data_parallel_rpc_port = 13389

[vllm_mix.env]
OMP_PROC_BIND = false
OMP_NUM_THREADS = 1
HCCL_BUFFSIZE = 1024
TASK_QUEUE_ENABLE = 1

[[vllm_mix.node]]
host = "192.0.2.10"
nic_name = "eth0"

[[vllm_mix.workers]]
host = "192.0.2.11"
ssh_port = 22
ssh_user = "your-user"
nic_name = "eth0"
# Fill in the container ID when the worker uses a container and omit this item when running on the host
# docker_container_id = "your-worker-container-id"
```

Replace the example addresses, communication NICs, SSH user, and container information with actual values. For more workers, append `[[vllm_mix.workers]]` in the same structure.

Fill in the parallel parameters of the entire service unit in `[vllm.command] others` of the corresponding search configuration. The following is an example for 2 nodes, 8 devices per node, TP=8, global DP=2, and PP=PCP=1:

```toml
# P unit
others = "--tensor-parallel-size 8 --data-parallel-size 2 --trust-remote-code --no-enable-prefix-caching"
```

```toml
# D unit
others = "--tensor-parallel-size 8 --data-parallel-size 2 --trust-remote-code --enable-prefix-caching"
```

Parameters such as EP, quantization, and MTP use the verified model deployment configuration. Configure the parallel strategy according to the node placement rules of the plugin: the global DP is an integer multiple of the number of nodes, the per-node occupancy satisfies `DP / number of nodes × TP × PP × PCP <= devices per node`, and the TP, PP, and PCP groups are placed within a node. This document fixes these parameters, and searches the batching parameters and concurrency during the search period.

Run the following on the primary node of the P unit:

```bash
export CLUSTER_CONFIG_PATH="$PWD/prefill_cluster.toml"
uv run --no-sync msmodeling optix --mode standard -e multihost_infer -b ais_bench \
  -c ./prefill_search.toml --backup
```

On the primary node of the D unit, prepare `decode_cluster.toml`, fill in the node information for D, and then run:

```bash
export CLUSTER_CONFIG_PATH="$PWD/decode_cluster.toml"
uv run --no-sync msmodeling optix --mode standard -e multihost_infer -b ais_bench \
  -c ./decode_search.toml --backup
```

Record the complete resource occupancy of as many machines as a P/D unit contains. For example, the aforementioned two-node unit is recorded as "2 machines, 16 devices, global DP=2", and the measured QPS belongs to the entire unit.

## Step 4: Fixing the Candidate and Measuring the QPS Used for the Ratio

### 4.1 Select and Deploy a Candidate

Open `result/pd_ratio_*/store/data_storage_*.csv` on both sides, view the measured records with `real_evaluation=true`, and record the service parameters and concurrency. On the P side, select the parameters with higher throughput from the records whose TTFT and success rate meet the target. On the D side, select the parameters with higher throughput from the records whose TPOT and success rate meet the target. The D CSV metrics come from the last formal measurement round of each candidate group and can be used directly for the ratio. When you need an independent retest, redeploy, warm up, and measure again according to the following steps.

| Column to Record | Purpose |
|---|---|
| `MAX_NUM_BATCHED_TOKENS`, `MAX_NUM_SEQS` | Rebuild the corresponding service parameters |
| `CONCURRENCY` | Set the retest concurrency |
| `success_rate` | Select the measurement results with a success rate of 1 |
| `time_to_first_token(s)` | The TTFT of P |
| `time_per_output_token(s)` | The TPOT of D |
| `throughput(requests/s)` | Request throughput, uniformly in req/s |

After you stop the previous candidate service, restart a service unit according to the candidate parameters. Single-machine example:

```bash
# P-side example, replace with the candidate parameters on this side and the complete fixed deployment parameters
vllm serve /data/models/your-model --served-model-name pd-model --port 8900 \
  --tensor-parallel-size 8 --data-parallel-size 1 --trust-remote-code \
  --no-enable-prefix-caching --max-num-batched-tokens 16384 --max-num-seqs 32
```

```bash
# D-side example, replace with the candidate parameters on this side and the complete fixed deployment parameters
vllm serve /data/models/your-model --served-model-name pd-model --port 8900 \
  --tensor-parallel-size 8 --data-parallel-size 1 --trust-remote-code \
  --enable-prefix-caching --max-num-batched-tokens 8192 --max-num-seqs 64
```

On the multi-machine side, locate the `Reproduce/` directory of the candidate in `result/pd_ratio_*/back_up/` according to the search log, and verify that the service parameters in the script are consistent with the CSV candidate. Copy `start_node.sh` to `/tmp/optix-ratio-replay/` on the original primary node, and copy scripts such as `start_work_0.sh` to the directories with the same name on the corresponding workers. When a worker uses a container, both the directory and the execution terminal are inside the original container.

First run the following in the primary node terminal:

```bash
cd /tmp/optix-ratio-replay
bash start_node.sh
```

Then run the corresponding script in another terminal on each worker. Taking the first worker as an example:

```bash
cd /tmp/optix-ratio-replay
bash start_work_0.sh
```

Keep the services on each node running, and check the entry in the benchmark terminal:

```bash
curl --fail http://127.0.0.1:8900/health
```

Continue after you confirm that `Application startup complete` appears in the primary node log. During the retest, keep the parallel strategy, fixed model parameters, and node communication configuration in the script. When you change the candidate, stop the previous service and restart it.

### 4.2 Measure the P Throughput

In the AISBench model configuration of P, set `batch_size` to the candidate concurrency, `request_rate=0`, and `max_out_len=1`. After the service is ready, run the following:

```bash
ais_bench --models optix_ratio_prefill --datasets gsm8k_gen_0_shot_cot_str_perf \
  --mode perf --num-prompts 512 --work-dir ./pd_ratio_measure/prefill --debug
```

Among the results with a success rate of 1 and a compliant TTFT, record the request throughput as `QPS_P`, and keep the corresponding service parameters and concurrency.

### 4.3 Warm Up and Measure the D Throughput

In the AISBench model configuration of D, set `batch_size` to the candidate concurrency, `request_rate=0`, and `max_out_len` to the target output length. Run the following two rounds of benchmarking, and **keep the same service unit running between the two rounds**:

```bash
# Round 1: fill the prefix cache with the fixed request set
ais_bench --models optix_ratio_decode --datasets gsm8k_gen_0_shot_cot_str_perf \
  --mode perf --num-prompts 512 --work-dir ./pd_ratio_measure/decode_warmup --debug

# Round 2: perform the formal measurement with the same request set and record the steady-state results
ais_bench --models optix_ratio_decode --datasets gsm8k_gen_0_shot_cot_str_perf \
  --mode perf --num-prompts 512 --work-dir ./pd_ratio_measure/decode_measured \
  --summarizer stable_stage --debug
```

Reserve enough space for the prefix cache according to this request set, and confirm the warm-up effect with the prefix cache hit metrics of the service. For a multi-DP unit, warm up each replica, and confirm that requests can hit the cache of the corresponding replica during the formal measurement. Continue the formal benchmarking until valid steady-state statistics are obtained.

Start every candidate from a newly launched service, and perform the formal measurement only after the warm-up is complete. Among the formal results with a success rate of 1 and a compliant TPOT, select the candidate with higher request throughput, record it as `QPS_D`, and keep its service parameters and concurrency.

Summarize the final measurement records of both sides:

| Item | P Unit | D Unit |
|---|---|---|
| Device model | Actual model | Actual model |
| Number of machines/devices per unit | Measured deployment occupancy | Measured deployment occupancy |
| DP/TP/EP, and so on | Measured configuration | Measured configuration |
| Batching parameters and test concurrency | Final candidate | Final candidate |
| Success rate | 1 | 1 |
| Latency | Compliant TTFT | Compliant TPOT |
| Request throughput | `QPS_P`, req/s | `QPS_D`, req/s |

## Step 5: Manually Calculating the PD Ratio

Calculate the number of service units by balancing the processing capability of both sides:

```text
Number of P units × QPS_P ≈ Number of D units × QPS_D
P:D ≈ QPS_D:QPS_P
Theoretical throughput of the candidate combination ≈ min(Number of P units × QPS_P, Number of D units × QPS_D)
```

For example, if a P unit delivers 30 req/s and a D unit delivers 10 req/s, the theoretical ratio is **1P:3D**. These two numbers are used only to demonstrate the calculation.

For heterogeneous deployment, calculate the two types of devices separately:

```text
Number of P units × devices per P unit <= number of corresponding devices available for P
Number of D units × devices per D unit <= number of corresponding devices available for D
```

For multi-machine deployment, also calculate the number of machines by the complete service unit. For example, if each P and D unit uses 2 machines with 8 devices each, then 1P:3D corresponds to 2 machines for P and 6 machines for D, for a total of 8 machines and 64 devices. The parallel strategy used during measurement is maintained inside each unit.

List adjacent integer ratios according to the actual resources, and compare the total processing capability of both sides. The following still uses the 30/10 req/s example:

| Ratio | Total P Capability | Total D Capability | Theoretical Throughput |
|---|---|---|---|
| 1P:2D | 30 req/s | 20 req/s | 20 req/s |
| 1P:3D | 30 req/s | 30 req/s | 30 req/s |
| 2P:3D | 60 req/s | 30 req/s | 30 req/s |

Select a combination allowed by the resources, and record the number of units, total number of machines, number of devices of each type, service parameters, and theoretical throughput as the initial configuration of the complete PD deployment.

## Step 6: Deploying the Complete PD Service and Verifying the Ratio

Use the existing PD deployment tool to deploy P, D, and the Proxy in the quantities selected in step 5:

1. Each unit uses the service parameters and fixed parallel strategy determined in step 4.
2. Configure the PD communication parameters, KV cache transmission, and Proxy routing.
3. Complete normal inference through the Proxy, and then run benchmarking with the same service input, target output length, and sampling parameters.

Refer to [Benchmark the complete PD service through the Proxy](optix_pso_pd_disaggregation.md#step-5-benchmarking-the-complete-pd-service-through-the-proxy), copy the complete service benchmark configuration, set the AISBench model short name and address to the actual Proxy, and run the following on the host or the benchmark machine:

```bash
uv run --no-sync msmodeling optix --mode standard -e vllm -b ais_bench \
  -c ./pd_fine_tune.toml
```

This configuration uses `skip_pso=true` and `manage_simulator_lifecycle=false` to adjust the benchmark workload for the deployed service. Filter the results by TTFT, TPOT, and success rate, and compare the throughput of the complete link. Adjust the number of P/D units and the concurrency according to the verification results, and keep the final compliant deployment ratio and parameters.
