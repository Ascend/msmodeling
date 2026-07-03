# TensorCast and ServingCast Quick Start

<br>

## 1. Overview

msModeling provides single-model performance simulation and service-level inference simulation. This guide is for first-time TensorCast and ServingCast users. It walks through environment checks, LLM text-generation simulation, throughput optimization, and end-to-end service simulation, helping you understand the main inputs, outputs, and usage scenarios.

### 1.1 Before You Start

**Experience Map (core operations take about 10 minutes)**

> **Recommended order**: Step 1 is the environment baseline. Step 2 runs TensorCast single-model simulation. Steps 3 and 4 cover ServingCast throughput optimization and service simulation, and can be selected as needed.

| Step | Stage | Core Module | Reference Operation Time | Suggested Concept Study |
| :---: | :---: | :--- | :---: | :---: |
| **1** | **Environment setup** | `msModeling` | 2 minutes | 5 minutes |
| **2** | **Single-model simulation** | `TensorCast` | 1 minute | 10 minutes |
| **3** | **Throughput optimization** | `ServingCast / Throughput Optimizer` | 2 minutes | 15 minutes |
| **4** | **Service simulation** | `ServingCast` | 2 minutes | 15 minutes |

### 1.2 Environment Preparation

👉 **Important: Complete the environment installation and configuration in the [msModeling Install Guide](../install_guide/msmodeling_install_guide.md) first.**

> [!CAUTION]
> This guide assumes commands are run from the msModeling repository root. If you run them from another directory, set `PYTHONPATH` first. Otherwise, errors such as `No module named cli` or `No module named tensor_cast` may occur.

## 2. Steps

> [!NOTE]
> The commands below can be copied and run directly. Use `TEST_DEVICE` to complete the workflow first, then replace it with the target hardware device and workload model.

### 2.1 Environment: Confirm Runtime Setup

Before starting, complete the environment setup in the [msModeling Install Guide](../install_guide/msmodeling_install_guide.md), including repository cloning, virtual environment creation, dependency installation, and `PYTHONPATH` configuration.

The following commands assume you are in the msModeling repository root. If not, set:

```bash
export PYTHONPATH=/path/to/msmodeling:$PYTHONPATH
```

TensorCast reads model configuration files from Hugging Face. If the environment cannot access Hugging Face directly, set a mirror:

```bash
export HF_ENDPOINT="https://hf-mirror.com"
```

Use the following commands to confirm the command-line entry points are available:

```bash
python -m cli.inference.text_generate --help
python -m serving_cast.main --help
```

If the commands do not print help information, check that the virtual environment is activated, dependencies are installed, and `PYTHONPATH` points to the msModeling repository root.

### 2.2 Single-Model Simulation: Run TensorCast Text Generation

TensorCast performs performance modeling for PyTorch programs. It does not execute the model on a real accelerator. Instead, it intercepts the computation graph and estimates operator latency, memory usage, and overall inference performance based on the target device profile.

> [!NOTE]
> TensorCast prints operator-level performance summaries, total execution time, TPS/Device, and memory usage by default. If `--chrome-trace` is specified, it can also generate a Chrome Trace file for timeline analysis.

#### 2.2.1 Run LLM Text-Generation Simulation

```bash
python -m cli.inference.text_generate Qwen/Qwen3-32B \
    --num-queries 2 \
    --query-length 3500 \
    --device TEST_DEVICE
```

#### 2.2.2 Check Simulation Results

If the command succeeds, the terminal prints output similar to:

```text
Model compilation and execution time: 0.192 s
----------------------------------------------  --------------  ------------  ----------
                     Name                       analytic total  analytic avg  # of Calls
----------------------------------------------  --------------  ------------  ----------
tensor_cast.static_quant_linear.default              884.004ms       1.973ms         448
tensor_cast.attention.default                        259.855ms       4.060ms          64
...
Total time for analytic: 1.744s
[analytic] TPS/Device: 4013 token/s
Total device memory: 64.000 GB
```

Key metrics:

- `analytic total`: Estimated total operator latency.
- `analytic avg`: Estimated average latency per operator call.
- `# of Calls`: Number of operator calls.
- `TPS/Device`: Tokens per second per device.
- `Total device memory`: Estimated memory usage, including weights, KV cache, and activations.

Success criteria:

- The terminal prints an operator-level performance table.
- The output includes `Total time for analytic` or `[analytic] TPS/Device`.
- The output includes memory estimation, such as `Total device memory`.

#### 2.2.3 Generate Chrome Trace (Optional)

To inspect a more fine-grained timeline, add `--chrome-trace`:

```bash
python -m cli.inference.text_generate Qwen/Qwen3-32B \
    --num-queries 2 \
    --query-length 3500 \
    --device TEST_DEVICE \
    --chrome-trace ./tensorcast_trace.json
```

After generation, open the trace file with `chrome://tracing` or MindStudio Insight.

### 2.3 Throughput Optimization: Run the ServingCast Throughput Optimizer

The ServingCast throughput optimizer searches for the best parallel strategy and batch configuration under SLO constraints such as TTFT and TPOT. It helps estimate the maximum serving throughput of a target model on target hardware.

> [!NOTE]
> PD colocated means Prefill and Decode run in the same instance. It is suitable for quickly evaluating overall service throughput. To evaluate Prefill and Decode separately, see the [Throughput Optimizer Guide](../user_guide/msmodeling_throughput_optimizer_user_guide.md).

#### 2.3.1 Run Throughput Optimization

The following command quickly evaluates a PD colocated scenario. For the first run, no explicit search dimensions are specified, so the tool uses the default TP search range. If the run takes too long, reduce `--num-devices` or specify `--tp-sizes` in advanced usage to narrow the search space.

```bash
python -m cli.inference.throughput_optimizer Qwen/Qwen3-32B \
    --device TEST_DEVICE \
    --num-devices 8 \
    --input-length 3500 \
    --output-length 1500 \
    --quantize-linear-action W8A8_DYNAMIC \
    --quantize-attention-action DISABLED \
    --tpot-limits 50
```

#### 2.3.2 Check Optimization Results

If the command succeeds, the terminal prints candidate configurations and throughput metrics. Focus on:

- `TP` / `DP`: Recommended parallel strategy.
- `batch size`: Batch size that satisfies the SLO constraints.
- `TTFT` / `TPOT`: Time to first token and time per output token.
- `token throughput`: System-level token throughput.

Success criteria:

- The terminal prints candidate or best configurations.
- The output includes throughput, TTFT, and TPOT metrics.
- No model configuration loading failure or parameter conflict is reported.

### 2.4 Service Simulation: Run End-to-End Serving Simulation

ServingCast service simulation uses YAML files to describe instance groups, request workloads, and serving limits. It can simulate end-to-end serving scenarios with multiple instances and requests, and outputs system-level metrics such as E2E_TIME, TTFT, TPOT, request throughput, and token throughput.

#### 2.4.1 Inspect Example Configurations

The repository includes example configurations that can be used directly:

```bash
ls serving_cast/example/instances.yaml serving_cast/example/common.yaml
```

The two configuration files are used as follows:

| Config File | Purpose |
| --- | --- |
| `instances.yaml` | Describes one or more instance groups, such as role, instance count, and TP/DP parallelism. |
| `common.yaml` | Describes global settings, such as model structure, request workload, serving limits, and simulation parameters. |

The example configurations use `TEST_DEVICE` and `Qwen/Qwen3-32B` by default. To change the model, request length, or request count, edit `common.yaml`. To change instance count, device count, or parallel strategy, edit `instances.yaml`.

#### 2.4.2 Run Service Simulation

```bash
python -m serving_cast.main \
    --instance_config_path=./serving_cast/example/instances.yaml \
    --common_config_path=./serving_cast/example/common.yaml
```

#### 2.4.3 Check Service Simulation Results

When simulation finishes, the console prints a performance summary similar to:

```text
         E2E_TIME(s)  TTFT(s)  TPOT(s)  INPUT_TOKENS  OUTPUT_TOKENS  OUTPUT_TOKEN_THROUGHPUT(tok/s)
AVERAGE     1052.591    0.378    0.301        1500.0         3500.0                           3.327
MIN         1050.000    0.300    0.300        1500.0         3500.0                           2.978
MAX         1175.500    0.600    0.336        1500.0         3500.0                           3.334
======== Overall Summary ========
request_throughput(req/s)      0.082
input_token_throughput(tok/s)  122.399
output_token_throughput(tok/s) 285.598
```

Metric descriptions:

- `E2E_TIME`: End-to-end latency for a single request.
- `TTFT`: Time to first token.
- `TPOT`: Time per output token after the first token.
- `request_throughput`: System-level request throughput.
- `input_token_throughput` / `output_token_throughput`: Aggregated token throughput.

Success criteria:

- The terminal prints a request-level statistics table.
- The output includes `Overall Summary`.
- The output includes `request_throughput`, `input_token_throughput`, or `output_token_throughput`.

## 3. Validate Results and Next Steps

If the commands above succeed, you have completed the core TensorCast and ServingCast workflow:

- TensorCast: single-model text-generation performance simulation with operator latency, TPS/Device, and memory estimates.
- Throughput Optimizer: throughput optimization under SLO constraints with recommended parallel strategy and throughput metrics.
- ServingCast: end-to-end service simulation with TTFT, TPOT, request throughput, and token throughput.

Common issues:

- If model configuration cannot be downloaded, check network access to Hugging Face or set the `HF_ENDPOINT` mirror.
- If `cli`, `tensor_cast`, or `serving_cast` cannot be imported, confirm that you are in the repository root or that `PYTHONPATH` is set correctly.
- If throughput optimization takes too long, reduce the search range, such as lowering `--num-devices` or explicitly specifying `--tp-sizes`.

Continue with:

- [TensorCast User Guide](../user_guide/msmodeling_tensor_cast_user_guide.md)
- [ServingCast User Guide](../user_guide/msmodeling_serving_cast_user_guide.md)
- [Throughput Optimizer Guide](../user_guide/msmodeling_throughput_optimizer_user_guide.md)
- [ServingCast Simulation Guide](../user_guide/msmodeling_serving_cast_simulation_user_guide.md)
