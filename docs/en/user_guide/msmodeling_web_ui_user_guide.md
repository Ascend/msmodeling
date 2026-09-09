# Web UI User Guide

This document is intended for everyday Modeling users and developers who are about to integrate the project. It helps you quickly understand what the tool can do, how to launch simulations from the Web UI or the CLI, how to interpret results, and how to configure parameters for different business scenarios.

If you only want to start the frontend page, run:

```bash
python -m web_ui.web_ui_start --port 2345
```

After startup, open `http://127.0.0.1:2345` in your browser.

---

## Reading Navigation

| Goal | Recommended Section |
| --- | --- |
| Starting the Web UI for the first time | [3. Web UI Quick Start](#web-ui-quick-start) |
| Configuring LLM/VL simulation | [4. LLM/VL Simulation Guide](#llm-vl-simulation) |
| Configuring video generation simulation | [5. Video Generation Simulation Guide](#video-generation-simulation) |
| Using the throughput optimizer | [6. Optimizer Throughput Optimization Guide](#optimizer-guide) |
| Interpreting results and exporting data | [7. Reading Result Charts and Detail Tables](#results-guide) |
| Troubleshooting common issues | [9. FAQ](#faq) |

---

## 1 Tool Positioning

Modeling is a simulation tool for model inference performance analysis. Its core capabilities include:

- Predict operator latency, memory usage, communication overhead, and overall inference time based on device profiles without real hardware or a full environment for running a real LLM.
- Support LLM text inference, VL multimodal inference, Diffusion Transformer inference for video generation, and serving throughput optimization.
- Support side-by-side comparison across multiple chips to help determine the performance differences of the same model on different devices.
- Support analysis of parameter combinations such as concurrency, TP, quantization, MTP, Prefix Cache, Ulysses, DiT Cache, PD aggregation, PD disaggregation, and PD ratio.
- The Web UI provides visual charts, detail tables, case selection, Excel export, and history cache, while the CLI suits scripted batch experiments.

The entry points most relevant to you in the repository are as follows:

| Entry Point                                     | Purpose | Recommended Use Case |
|-------------------------------------------------|---|---|
| python -m web_ui.web_ui_start                   | Start the Gradio frontend | Interactive configuration, result visualization, and use by non-developers |
| python -m cli.inference.text_generate           | LLM/VL forward inference simulation | Single or scripted LLM/VL performance analysis |
| python -m cli.inference.video_generate          | Video generation model simulation | Scenarios such as Diffusion Transformer/Wan/HunyuanVideo |
| python -m cli.inference.throughput_optimizer    | Serving throughput optimization | Find the optimal parallelism and batch under TTFT/TPOT/SLO constraints |

---

## 2 Environment Preparation

For the complete environment setup steps (cloning the repository, creating a virtual environment, installing dependencies, and configuring `PYTHONPATH` and Hugging Face access), see the [msModeling Installation Guide](../install_guide/msmodeling_install_guide.md).

If the environment is already set up, starting the Web UI from the repository root generally requires no additional configuration. The tool reads model configurations, commonly from Hugging Face, ModelScope, or a local model directory. If Hugging Face is inaccessible from your network, you can select `modelscope` in the `remote-source` field of the Web UI, or configure the `HF_ENDPOINT` mirror as described in the installation guide.

---

<a id="web-ui-quick-start"></a>

## 3 Web UI Quick Start

### 3.1 Starting the Local Page

```bash
python -m web_ui.web_ui_start --port 2345
```

Suitable for local use. Open it in your browser:

```text
http://127.0.0.1:2345
```

### 3.2 Web UI Page Description

The current Web UI mainly contains three types of workspaces:

| Page | Capability |
|---|---|
| Simulator - LLM Forward | LLM text inference simulation, supporting concurrency lists, TP lists, quantization, MTP, Prefix Cache, fine-grained parallelism, operator and memory analysis |
| Simulator - VL Forward | Multimodal VL inference simulation, adding image parameters such as image batch, height, and width on top of the LLM parameters |
| Video Generation | Video generation model inference simulation, supporting parameters such as Ulysses, CFG, DiT Cache, and Chrome Trace |
| Optimizer | Serving throughput optimization, supporting three deployment modes: `PD aggregation`, `PD disaggregation`, and `PD ratio` |

### 3.3 Basic Web UI Operation Process

1. Select the model, the primary chip, and optional competitor chips.
2. Fill in parameters such as the number of devices, concurrency, length, quantization, and parallelism.
3. Click "Preview Configuration" or "Preview Command" to confirm the CLI command to be run.
4. Click "Start Running".
5. View the summary conclusion, charts, memory analysis, bandwidth bottlenecks, operator details, and exported results.
6. If you set a concurrency list or TP list, select a specific case in the detail analysis area, for example, `Concurrency=32 | TP=2`, and then view the memory and operator data of that case.

---

<a id="llm-vl-simulation"></a>

## 4 LLM/VL Simulation Guide

Both LLM and VL simulations ultimately call:

```bash
python -m cli.inference.text_generate <model_id> [options]
```

VL adds image input parameters on top of the LLM simulation.

### 4.1 Key Concepts

| Concept                              | Description |
|--------------------------------------|---|
| num-queries                          | Number of concurrent requests, affecting batch, KV Cache, memory, and throughput |
| query-length                         | Number of new tokens in this run. Prefill is usually larger, and decode is usually 1 or a small value |
| context-length                       | Existing context length, affecting KV Cache and attention cost |
| decode                               | Enable autoregressive decode mode |
| tp-size                              | Number of Tensor Parallel workers |
| dp-size                              | Number of Data Parallel workers. You can enter `auto` in the Web UI |
| ep-size                              | Number of Expert Parallel workers, commonly used for MoE models |
| num-mtp-tokens                       | Number of MTP tokens, available for models that support MTP, such as DeepSeek |
| prefix-cache-hit-rate                | Prefix Cache hit rate, in the range [0,1), used to estimate the reuse benefit of prefill tokens |
| quantize-linear-action               | Quantization method for Linear layers, for example, `W8A8_DYNAMIC`, `FP8`, or `MXFP4` |
| quantize-non-expert-linear-action    | Quantization override for non-expert Linear layers, mainly used for DeepSeek V4. It applies to attention projections, dense MLP, and shared experts, while routed MoE experts still use `quantize-linear-action` |
| quantize-attention-action            | Quantization method for KV Cache/Attention, for example, `DISABLED`, `INT8`, or `FP8` |
| image-height/image-width             | VL image size |

### 4.2 Minimal LLM Example: Single-Chip Decode

```bash
python -m cli.inference.text_generate Qwen/Qwen3-32B \
  --device ATLAS_800_A2_280T_32G_PCIE \
  --num-devices 1 \
  --num-queries 32 \
  --query-length 1 \
  --context-length 4500 \
  --decode \
  --quantize-linear-action W8A8_DYNAMIC \
  --quantize-attention-action DISABLED
```

Suitable for quickly observing the single-device inference time, `TPS/Device`, memory, and operator proportions of a chip in a typical decode scenario.

### 4.3 Prefill Example: Long-Input Throughput and Bottleneck Analysis

```bash
python -m cli.inference.text_generate Qwen/Qwen3-32B \
  --device ATLAS_800_A2_280T_32G_PCIE \
  --num-devices 8 \
  --num-queries 8 \
  --query-length 3500 \
  --context-length 0 \
  --compile \
  --tp-size 8 \
  --quantize-linear-action W8A8_DYNAMIC \
  --quantize-attention-action INT8
```

This scenario focuses on the processing cost of the initial input segment, suitable for comparing:

- Whether prefill is affected by communication bottlenecks at different TP values.
- Whether Attention quantization reduces memory and bandwidth pressure.
- The impact of `compile` on graph compilation and execution time.

### 4.4 Concurrency List Example: Plotting Concurrency Curves

You can enter the following in the Web UI:

```text
Concurrency list: [16,32,64]
TP degree: 1
```

This is equivalent to running multiple experiments with different `--num-queries` values. The result area plots the relationship between concurrency and inference time and throughput, helping you find the optimal concurrency range.

For batch experiments from the CLI, you can use a script loop:

```bash
for nq in 16 32 64; do
  python -m cli.inference.text_generate Qwen/Qwen3-32B \
    --device ATLAS_800_A2_280T_32G_PCIE \
    --num-devices 8 \
    --num-queries $nq \
    --query-length 8 \
    --context-length 4500 \
    --decode \
    --tp-size 1 \
    --quantize-linear-action MXFP4 \
    --quantize-attention-action DISABLED
done
```

### 4.5 TP List Example: Iterating Multiple TP Values for the Same Model

You can enter the following in the Web UI:

```text
Number of deployed devices: 8
Request concurrency: 32
TP list: [1,2,4,8]
```

The tool iterates over multiple TP values at the same concurrency and outputs a chart of inference time against the TP count. The horizontal axis is the TP count, and the vertical axis is the inference time.

It helps answer:

- Whether computation speeds up as TP increases.
- Whether communication overhead cancels out the computation gains.
- The most suitable TP range for the current chip and model.

### 4.6 Concurrency List + TP List Example

You can enter the following in the Web UI:

```text
Number of deployed devices: 8
Concurrency list: [16,32,64]
TP list: [1,2]
```

The tool iterates over concurrency values for each TP and outputs concurrency curves for each TP. The results can be understood as:

| TP | Cases to Run |
|---|---|
| 1 | Concurrency 16, 32, and 64 |
| 2 | Concurrency 16, 32, and 64 |

Afterward, the memory, bandwidth, and operator detail areas show case selection options, for example:

```text
Concurrency=16 | TP=1
Concurrency=32 | TP=1
Concurrency=64 | TP=1
Concurrency=16 | TP=2
Concurrency=32 | TP=2
Concurrency=64 | TP=2
```

When viewing details, select the chip first and then the specific case. Otherwise, you may confuse the memory and operator data of different concurrency and TP values.

### 4.7 DeepSeek/MTP Example

```bash
python -m cli.inference.text_generate deepseek-ai/DeepSeek-R1 \
  --device ATLAS_800_A2_280T_32G_PCIE \
  --num-devices 8 \
  --num-queries 32 \
  --query-length 3 \
  --context-length 3500 \
  --decode \
  --num-mtp-tokens 2 \
  --tp-size 8 \
  --ep-size 8 \
  --quantize-linear-action W8A8_DYNAMIC \
  --compile
```

Note: `query-length` must be greater than the number of MTP tokens. Otherwise, there are not enough generated tokens to support MTP analysis.

### 4.8 Prefix Cache Example

```bash
python -m cli.inference.text_generate Qwen/Qwen3-32B \
  --device ATLAS_800_A2_280T_32G_PCIE \
  --num-devices 8 \
  --num-queries 32 \
  --query-length 512 \
  --context-length 4096 \
  --prefix-cache-hit-rate 0.5 \
  --tp-size 4 \
  --quantize-linear-action W8A8_DYNAMIC
```

`prefix-cache-hit-rate=0.5` indicates an approximate token-level estimate of 50% prefix hits. The higher the hit rate, the shorter the effective prefill length, and the TTFT and prefill-side memory pressure usually decrease.

### 4.9 VL Example: Image-Input Inference

```bash
python -m cli.inference.text_generate Qwen/Qwen3-VL-235B-A22B-Instruct \
  --device ATLAS_800_A2_280T_32G_PCIE \
  --num-devices 8 \
  --num-queries 4 \
  --query-length 16 \
  --context-length 200 \
  --decode \
  --tp-size 8 \
  --image-batch-size 1 \
  --image-height 720 \
  --image-width 1080 \
  --quantize-linear-action W8A8_DYNAMIC \
  --quantize-attention-action INT8
```

For VL scenarios, focus on the following:

- The impact of image size changes on memory.
- The memory peak after image batch is combined with text concurrency.
- The time proportion of operators related to the vision tower or multimodal projection.

---

<a id="video-generation-simulation"></a>

## 5 Video Generation Simulation Guide

Video generation entry point:

```bash
python -m cli.inference.video_generate <model_id> [options]
```

This tool simulates the Diffusion Transformer forward process and is commonly used for performance estimation of video generation models such as Wan and HunyuanVideo.

### 5.1 Key Parameters

| Parameter                | Description |
|--------------------------|---|
| --batch-size             | Video generation batch |
| --seq-len                | Text prompt token length |
| --height/--width         | Video resolution |
| --frame-num              | Number of frames |
| --sample-step            | Number of denoise steps |
| --dtype                  | `float16`, `float32`, or `bfloat16` |
| --world-size             | Total number of devices |
| --ulysses-size           | Ulysses sequence parallel size, which must divide `world-size` |
| --use-cfg                | Enable CFG |
| --cfg-parallel           | Use CFG parallelism |
| --dit-cache              | Enable DiT block cache |
| --cache-step-range       | Step range in which DiT Cache takes effect, in the `start,end` format |
| --cache-step-interval    | Refresh the cache every N steps. 1 is equivalent to no reuse |
| --cache-block-range      | Block cache range, in the `start,end` format |

### 5.2 Minimal Video Simulation Example

```bash
python -m cli.inference.video_generate Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --device ATLAS_800_A2_280T_32G_PCIE \
  --batch-size 1 \
  --seq-len 128 \
  --height 720 \
  --width 1280 \
  --frame-num 81 \
  --sample-step 50 \
  --dtype float16 \
  --quantize-linear-action W8A8_DYNAMIC
```

### 5.3 Ulysses Parallel Example

```bash
python -m cli.inference.video_generate Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --device ATLAS_800_A2_280T_32G_PCIE \
  --batch-size 1 \
  --seq-len 128 \
  --height 720 \
  --width 1280 \
  --frame-num 129 \
  --sample-step 50 \
  --world-size 8 \
  --ulysses-size 4 \
  --dtype float16
```

Configuration requirement:

```text
world-size % ulysses-size == 0
```

If this is not satisfied, the program reports an error. The Web UI also validates it in advance.

### 5.4 CFG and CFG Parallel Example

```bash
python -m cli.inference.video_generate Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --device ATLAS_800_A2_280T_32G_PCIE \
  --batch-size 1 \
  --seq-len 128 \
  --height 720 \
  --width 1280 \
  --frame-num 81 \
  --sample-step 30 \
  --world-size 8 \
  --ulysses-size 4 \
  --use-cfg \
  --cfg-parallel
```

`--use-cfg` simulates classifier-free guidance. `--cfg-parallel` is suitable for comparing the impact of CFG on communication and parallelism efficiency.

### 5.5 DiT Cache Example

```bash
python -m cli.inference.video_generate Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --device ATLAS_800_A2_280T_32G_PCIE \
  --batch-size 1 \
  --seq-len 128 \
  --height 720 \
  --width 1280 \
  --frame-num 81 \
  --sample-step 50 \
  --dit-cache \
  --cache-step-range 10,40 \
  --cache-step-interval 5 \
  --cache-block-range 0,20
```

Notes:

- `--cache-step-range 10,40` means cache reuse is attempted from the 10th to the 40th denoise step.
- `--cache-step-interval 5` means the cache refreshes every 5 steps and is reused for the remaining steps.
- `--cache-step-interval 1` basically disables cache reuse.

### 5.6 Chrome Trace Export

```bash
python -m cli.inference.video_generate Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --device ATLAS_800_A2_280T_32G_PCIE \
  --batch-size 1 \
  --seq-len 128 \
  --chrome-trace trace/video.json
```

After generation, you can open it in the Chrome browser:

```text
chrome://tracing
```

---

<a id="optimizer-guide"></a>

## 6 Optimizer Throughput Optimization Guide

Throughput optimization entry point:

```bash
python -m cli.inference.throughput_optimizer <model_id> [options]
```

The Optimizer does not run only one fixed parallelism configuration. Given the model, device, device count, input and output lengths, SLO constraints, and search space, it automatically searches for better parallelism, batch size, concurrency, and throughput.

### 6.1 Three Deployment Modes

The deployment mode names in the Web UI are:

| Web UI Name          | CLI Parameters | Applicable Scenario |
|----------------------|---|---|
| PD aggregation       | Default. Do not add `--disagg` or `--enable-optimize-prefill-decode-ratio` | prefill and decode are deployed together on the same type of instance. Run a baseline and compare across multiple chips first |
| PD disaggregation    | Add `--disagg` | Analyze prefill and decode separately, evaluating their capabilities under TTFT or TPOT constraints |
| PD ratio             | Add `--enable-optimize-prefill-decode-ratio` and specify the number of devices per P/D instance | In a PD-disaggregation architecture, find the ratio of prefill to decode instances |

### 6.2 PD Aggregation: Offline Throughput Optimization

When TTFT/TPOT constraints are not set, the tool focuses more on maximum throughput:

```bash
python -m cli.inference.throughput_optimizer Qwen/Qwen3-32B \
  --device ATLAS_800_A2_280T_32G_PCIE \
  --num-devices 8 \
  --input-length 3500 \
  --output-length 1500 \
  --compile \
  --quantize-linear-action W8A8_DYNAMIC \
  --quantize-attention-action INT8
```

It helps answer:

- What the theoretical maximum throughput of this model is with 8 devices.
- What the optimal TP/DP and batch roughly are.
- Which chip has higher optimal throughput in a multi-chip comparison.

### 6.3 PD Aggregation: Online Service SLO Constraints

Set both TTFT and TPOT:

```bash
python -m cli.inference.throughput_optimizer Qwen/Qwen3-32B \
  --device ATLAS_800_A2_280T_32G_PCIE \
  --num-devices 8 \
  --input-length 3500 \
  --output-length 1500 \
  --compile \
  --quantize-linear-action W8A8_DYNAMIC \
  --quantize-attention-action INT8 \
  --ttft-limits 2000 \
  --tpot-limits 50
```

Suitable for online service capacity evaluation:

- Whether TTFT meets the first-token response target.
- Whether TPOT meets the sustained generation speed target.
- What the optimal batch and concurrency are under the constraints.

### 6.4 Limiting the TP Search Space

By default, the Optimizer searches available TP values. You can also limit them manually:

```bash
python -m cli.inference.throughput_optimizer Qwen/Qwen3-32B \
  --device ATLAS_800_A2_280T_32G_PCIE \
  --num-devices 8 \
  --input-length 3500 \
  --output-length 1500 \
  --tp-sizes 1 2 4 8 \
  --batch-range 1 256 \
  --jobs 8
```

In the Web UI, you can enter a value in `TP parallel size list`:

```text
[1,2,4,8]
```

`batch-range` supports two meanings:

| Syntax                   | Meaning |
|--------------------------|---|
| --batch-range 256        | min defaults to 1 and max is 256 |
| --batch-range 1 256      | min is 1 and max is 256 |

### 6.5 PD Disaggregation: Prefill-Side TTFT Analysis

```bash
python -m cli.inference.throughput_optimizer Qwen/Qwen3-32B \
  --device ATLAS_800_A2_280T_32G_PCIE \
  --num-devices 8 \
  --input-length 3500 \
  --output-length 1500 \
  --compile \
  --quantize-linear-action W8A8_DYNAMIC \
  --quantize-attention-action DISABLED \
  --disagg \
  --ttft-limits 2000
```

This mode focuses on how many requests the prefill phase can handle under TTFT constraints.

### 6.6 PD Disaggregation: Decode-Side TPOT Analysis

```bash
python -m cli.inference.throughput_optimizer Qwen/Qwen3-32B \
  --device ATLAS_800_A2_280T_32G_PCIE \
  --num-devices 8 \
  --input-length 3500 \
  --output-length 1500 \
  --compile \
  --quantize-linear-action W8A8_DYNAMIC \
  --quantize-attention-action DISABLED \
  --disagg \
  --tpot-limits 50
```

This mode focuses on the sustained output capability of the decode phase under TPOT constraints.

### 6.7 PD Ratio: Optimizing the Prefill/Decode Instance Ratio

```bash
python -m cli.inference.throughput_optimizer deepseek-ai/DeepSeek-V3.1 \
  --device ATLAS_800_A2_280T_32G_PCIE \
  --num-devices 16 \
  --input-length 3500 \
  --output-length 1500 \
  --compile \
  --quantize-linear-action W8A8_DYNAMIC \
  --quantize-attention-action DISABLED \
  --enable-optimize-prefill-decode-ratio \
  --prefill-devices-per-instance 4 \
  --decode-devices-per-instance 2 \
  --ttft-limits 2000 \
  --tpot-limits 50 \
  --log-level info
```

The core idea of PD ratio is to calculate the Prefill QPS and Decode QPS separately, and then find a more balanced prefill/decode instance ratio.

For a rough understanding:

```text
Prefill QPS = prefill_concurrency / ttft_ms * 1000
Decode QPS  = decode_concurrency / (tpot_ms * output_length) * 1000
PD Ratio = Decode QPS / Prefill QPS
Balanced QPS = min(Prefill QPS, Decode QPS)
```

When `PD ratio > 1`, the decode side is relatively stronger and may need more prefill instances. When `PD ratio < 1`, the decode side may become the bottleneck.

### 6.8 Interpreting Optimizer Output

Typical output includes:

| Field              | Description |
|--------------------|---|
| Best Throughput    | Optimal token/s under the current constraints |
| TTFT               | Time To First Token, the first-token latency |
| TPOT               | Time Per Output Token, the per-output-token latency |
| concurrency        | Concurrency corresponding to the optimal configuration |
| parallel           | Parallelism configuration, for example, `tp4pp1dp2` |
| batch_size         | Optimal batch size |
| pd_ratio           | Instance ratio in PD ratio mode |
| balanced_qps       | System QPS after P/D balancing in PD ratio mode |

The Web UI also displays:

- Comparison of the optimal throughput across chips.
- Comparison of the optimal TTFT/TPOT across chips.
- Side-by-side comparison of fixed configurations.
- Key metrics table for PD ratio.
- Per-chip Pareto details.

---

<a id="results-guide"></a>

## 7 Reading Result Charts and Detail Tables

### 7.1 LLM/VL Results

Read the results in the following order:

1. Summary conclusion: check the total time, `TPS/Device`, and whether there are failures or warnings.
2. Inference time curves: check whether the time keeps decreasing as concurrency or TP increases.
3. Memory analysis: check the proportions of model weights, KV Cache, activations, and reserved memory.
4. Bandwidth bottlenecks: check memory bound, communication bound, and compute bound.
5. Operator details: sort by total time to locate the dominant operators.
6. Operator classification statistics: determine the optimization direction from categories such as GEMM, Attention, and Communication.

If you configured a concurrency list or TP list, select a case before viewing the details.

### 7.2 Video Results

Focus on:

- The relationship between total analytic time and sample steps.
- The proportion of communication operators after Ulysses.
- Whether CFG/CFG Parallel introduces extra all-gather or batch expansion.
- Whether DiT Cache significantly reduces the compute time of repeated blocks.

### 7.3 Optimizer Results

Recommended reading order:

1. Recommendation conclusion: check the optimal chip, throughput, parallelism, batch, and concurrency.
2. Per-chip optimal comparison: used to compare competitor chips with the primary chip.
3. Fixed-configuration comparison: ensure the comparison runs under the same configuration instead of comparing only the respective optimal points.
4. PD ratio: for a PD-disaggregation architecture, check the Balanced QPS and the prefill/decode instance ratio.
5. Per-chip Pareto: determine whether there are alternative points with higher throughput but slightly worse latency.

---

## 8 Parameter Selection Recommendations

### 8.1 If You Don't Know Where to Start

Initial LLM decode values:

```text
num-devices: 8
num-queries: 32
query-length: 1
context-length: 4500
decode: true
tp-size: 8
quantize-linear-action: W8A8_DYNAMIC
quantize-attention-action: DISABLED
```

Initial LLM prefill values:

```text
num-devices: 8
num-queries: 8
query-length: 3500
context-length: 0
decode: false
tp-size: 8
quantize-linear-action: W8A8_DYNAMIC
quantize-attention-action: INT8
```

Initial Optimizer values for online service:

```text
input-length: 3500
output-length: 1500
ttft-limits: 2000
tpot-limits: 50
tp-sizes: [1,2,4,8]
batch-range: [1,256]
jobs: 8
```

### 8.2 How to Choose TP

Rules of thumb:

- If the model weights are too large to fit, increase TP first.
- If single-device compute is the obvious bottleneck, increasing TP may bring significant gains.
- If communication accounts for a high proportion, further increasing TP may reduce the gains.
- For small models or small batches, an overly large TP may slow things down because of communication and synchronization overhead.

You are advised to first run [1,2,4,8] with the TP list in the Web UI, and then narrow the search range based on the curves.

### 8.3 How to Choose Concurrency

Rules of thumb:

- If concurrency is too low, device utilization may be insufficient.
- As concurrency increases gradually, throughput usually improves, but latency and memory also rise.
- If concurrency is too high, you may hit memory bottlenecks, an oversized KV Cache, or unacceptable latency.

You are advised to run [16,32,64,128] in the first round, and then scan finely around the optimal range.

### 8.4 How to Choose Quantization

| Scenario | Recommendation |
|---|---|
| Quick baseline | `W8A8_DYNAMIC` |
| No quantization impact desired | `DISABLED` |
| Obvious memory pressure | Try `INT8` attention or `FP8` |
| MXFP4 scheme evaluation | Use `MXFP4`, and adjust `mxfp4-group-size` when necessary |

Note: The simulation tool focuses on performance and resource estimation and does not replace real accuracy evaluation. The quality of the quantized model still needs to be verified through accuracy tests.

---

<a id="faq"></a>

## 9 FAQ

### 9.1 Browser Cannot Open the Web UI After Startup

Check:

- Whether you used the correct address: `http://127.0.0.1:2345`.
- Whether the port is occupied. You can change it to `--port 2346`.

### 9.2 Device Name Invalid

`--device` must come from `DeviceProfile.all_device_profiles`. The Web UI automatically loads the brand and chip lists from device profiles. From the CLI, you can view the choices in the error message, or select an available chip in the Web UI first.

### 9.3 Invalid TP/DP/EP Configuration

Common causes:

- `num-devices` is not divisible by `tp-size`.
- `world-size` is not divisible by `ulysses-size`.
- `TP * DP * EP` exceeds the number of deployed devices.
- Some fine-grained TP/DP parameters do not match the total device count.

Suggested handling: first run a simple configuration, for example, `tp-size=1, dp-size=auto, ep-size=1`, and then gradually increase the parallelism complexity.

### 9.4 Optimizer Finds No Feasible Solution

Common causes:

- The TTFT or TPOT constraints are too strict.
- `max-batched-tokens` is smaller than the effective input length.
- The batch search range is too small.
- The device count is insufficient or the TP search space is unsuitable.
- The reserved memory is too large, leaving insufficient available memory.

Suggested handling:

1. First remove the TTFT/TPOT constraints and check whether an offline optimum can be found.
2. Loosen `tpot-limits` or `ttft-limits`.
3. Increase the upper limit of `batch-range`.
4. Check whether `tp-sizes` contains feasible values.
5. Reduce `reserved-memory-gb` or use a stronger device profile.

### 9.5 Results Come from Cache and You Want to Rerun

The Web UI reads the cache in `.msmodeling_ui/results.sqlite3` and `.msmodeling_ui/logs/` based on the task hash. If you need a full rerun, you can clear the corresponding cache directory, or adjust a parameter that affects the simulation to generate a new task hash.

### 9.6 Chart Titles Obscure Content

The new Web UI places chart titles in a separate title position outside the image area instead of using the Gradio overlay title in the upper-left corner. If you still see the old style, confirm that the browser has not loaded an old page, and restart the Web UI.

---

## 10 Recommended Workflow Examples

### 10.1 Example A: Comparing the LLM Decode Capability of Two Chips

Web UI:

```text
Model: Qwen/Qwen3-32B
Primary chip: ATLAS_800_A2_280T_32G_PCIE
Comparison chip: Select another chip
Number of deployed devices: 8
Concurrency list: [16,32,64]
TP list: [1,2,4,8]
Number of generated tokens: 8
Context length: 4500
Decode mode: enabled
Quantization: MLP=W8A8_DYNAMIC, Attention=DISABLED
```

Observe:

- Which chip has lower inference time at the same TP and concurrency.
- Whether some chip shows more obvious communication bottlenecks at high TP.
- Whether the bottlenecks in the memory and operator details are consistent.

### 10.2 Example B: Evaluating the Impact of VL Image Size

First round:

```text
image-height: 720
image-width: 1080
```

Second round:

```text
image-height: 1024
image-width: 1024
```

Keep the other parameters unchanged and compare:

- The change in total inference time.
- The change in memory usage.
- The change in the time of Vision-related operators.

### 10.3 Example C: Video Generation Ulysses Scalability

Test in sequence:

```text
world-size=8, ulysses-size=1
world-size=8, ulysses-size=2
world-size=8, ulysses-size=4
world-size=8, ulysses-size=8
```

Observe:

- Whether the total time decreases as Ulysses increases.
- Whether the proportion of communication operators rises.
- Whether there is an optimal Ulysses value instead of "the larger, the better".

### 10.4 Example D: Online Service Capacity Evaluation

Web UI Optimizer:

```text
Deployment mode: PD aggregation
Model: Qwen/Qwen3-32B
Number of deployed devices: 8
Input length: 3500
Output length: 1500
TP degree list: [1,2,4,8]
Batch range: [1,256]
TTFT: 2000
TPOT: 50
Quantization: MLP=W8A8_DYNAMIC, Attention=INT8
```

Focus on the following in the output:

- Whether a feasible solution exists.
- Whether the optimal throughput, TTFT, and TPOT all meet the targets.
- Whether the optimal parallel and batch match the deployment expectations.

### 10.5 Example E: PD Ratio Deployment Planning

Web UI Optimizer:

```text
Deployment mode: PD ratio
Number of deployed devices: 16
Prefill devices per instance: 4
Decode devices per instance: 2
Input length: 3500
Output length: 1500
TTFT: 2000
TPOT: 50
```

Observe:

- Balanced QPS.
- Which of Prefill QPS and Decode QPS is lower.
- Whether the recommended P/D instance count and total device count match the actual cluster plan.

---

## 11 Additional Notes for Developers

If you plan to modify the Web UI, you are advised to first read:

```text
web_ui/README.md
```

Core file relationships:

```text
web_ui/__init__.py          Package entry point, lazily exposes launch_app
web_ui/app.py               Page layout and event binding
web_ui/components.py        Reusable components and result areas
web_ui/callbacks.py         Form building, validation, execution, and result collation
web_ui/command_builder.py   CLI command and task matrix generation
web_ui/runner.py            Caching, subprocess execution, and progress streaming
web_ui/parsers.py           Log parsing
web_ui/result_store.py      SQLite and log caching
web_ui/charts.py            Chart drawing
web_ui/styles.py            Shared CSS, theme helpers, and header styles
web_ui/schemas.py           Data classes shared among builders, runners, parsers, and stores
web_ui/utils.py             Shared parsing, hashing, and normalization helpers
web_ui/time_tracker.py      Tracks and displays simulation time information
web_ui/web_ui_start.py      Web UI server startup entry point
```

After modifying frontend features, you are advised to run:

```bash
python -m py_compile web_ui/__init__.py web_ui/app.py web_ui/callbacks.py web_ui/command_builder.py web_ui/components.py web_ui/charts.py web_ui/parsers.py web_ui/result_store.py web_ui/runner.py web_ui/schemas.py web_ui/styles.py web_ui/time_tracker.py web_ui/utils.py web_ui/web_ui_start.py
```

---

## 12 Quick Command Index

Start the Web UI:

```bash
python -m web_ui.web_ui_start --port 2345
```

LLM decode:

```bash
python -m cli.inference.text_generate Qwen/Qwen3-32B --device ATLAS_800_A2_280T_32G_PCIE --num-devices 8 --num-queries 32 --query-length 1 --context-length 4500 --decode --tp-size 8
```

VL:

```bash
python -m cli.inference.text_generate Qwen/Qwen3-VL-235B-A22B-Instruct --device ATLAS_800_A2_280T_32G_PCIE --num-devices 8 --num-queries 4 --query-length 16 --context-length 200 --decode --tp-size 8 --image-batch-size 1 --image-height 720 --image-width 1080
```

Video:

```bash
python -m cli.inference.video_generate Wan-AI/Wan2.2-T2V-A14B-Diffusers --device ATLAS_800_A2_280T_32G_PCIE --batch-size 1 --seq-len 128 --height 720 --width 1280 --frame-num 81 --sample-step 50
```

Optimizer:

```bash
python -m cli.inference.throughput_optimizer Qwen/Qwen3-32B --device ATLAS_800_A2_280T_32G_PCIE --num-devices 8 --input-length 3500 --output-length 1500 --tp-sizes 1 2 4 8 --batch-range 1 256 --ttft-limits 2000 --tpot-limits 50
```
