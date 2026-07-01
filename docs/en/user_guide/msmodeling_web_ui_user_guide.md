# Web UI User Guide

This document is intended for daily users of Modeling and developers who are about to integrate the project. Its goal is to help you quickly understand what the tool can do, how to launch simulations from the Web UI or CLI, how to interpret results, and how to configure parameters for different business scenarios.

If you only want to start the frontend page, simply run:

```bash
python -m web_ui.web_ui_start --port 2345
```

After starting, open `http://127.0.0.1:2345` in your browser.

---

## Reading Guide

| Goal | Recommended Section |
| --- | --- |
| First time launching Web UI | [3. Web UI Quick Start](#web-ui-quick-start) |
| Configuring LLM / VL simulations | [4. LLM / VL Simulation Guide](#llm-vl-simulation) |
| Configuring video generation simulations | [5. Video Generation Simulation Guide](#video-generation-simulation) |
| Using the throughput optimizer | [6. Optimizer Throughput Tuning Guide](#optimizer-guide) |
| Interpreting results and exporting data | [7. How to Read Result Charts and Detail Tables](#results-guide) |
| Troubleshooting common issues | [9. FAQ](#faq) |

---

## 1. Tool Positioning

Modeling is a simulation tool for model inference performance analysis. Its core capabilities include:

- Predicting operator latency, memory usage, communication overhead, and overall inference time based on device profiles, without requiring real hardware or a full runtime environment for large models.
- Supporting LLM text inference, VL multimodal inference, video generation Diffusion Transformer inference, and service throughput tuning.
- Supporting cross-chip comparisons to help evaluate performance differences of the same model on different devices.
- Supporting parameter combination analysis for concurrency, TP, quantization, MTP, Prefix Cache, Ulysses, DiT Cache, PD Aggregated, PD Disaggregated, PD Ratio, and more.
- The Web UI provides visual charts, detail tables, case selection, Excel export, and historical caching; the CLI is suitable for scripted batch experiments.

The most relevant entry points in the repository are as follows:

| Entry Point | Purpose | Recommended Scenario |
|---|---|---|
| `python -m web_ui.web_ui_start` | Launch the Gradio frontend | Interactive configuration, result visualization, non-developer users |
| `python -m cli.inference.text_generate` | LLM / VL forward inference simulation | One-off or scripted LLM/VL performance analysis |
| `python -m cli.inference.video_generate` | Video generation model simulation | Diffusion Transformer / Wan / HunyuanVideo scenarios |
| `python -m cli.inference.throughput_optimizer` | Service throughput tuning | Finding optimal parallel and batch under TTFT/TPOT/SLO constraints |

---

## 2. Environment Setup

For complete environment setup steps (cloning the repository, creating a virtual environment, installing dependencies, setting `PYTHONPATH` and Hugging Face access), please refer to the [msModeling Installation Guide](../install_guide/msmodeling_install_guide.md).

If the environment is already set up, launching the Web UI from the repository root generally requires no additional configuration. The tool reads model configurations, with common sources including Hugging Face, ModelScope, or local model directories. If the network cannot access Hugging Face, you can select `modelscope` in the Web UI's `remote-source`, or set the `HF_ENDPOINT` mirror as described in the installation guide.

---

<a id="web-ui-quick-start"></a>

## 3. Web UI Quick Start

### 3.1 Launch the Local Page

```bash
python -m web_ui.web_ui_start --port 2345
```

Suitable for local use. Open in your browser:

```text
http://127.0.0.1:2345
```

### 3.2 Web UI Page Overview

The current Web UI mainly includes three types of workspaces:

| Page | Capabilities |
|---|---|
| Simulator - LLM Forward | LLM text inference simulation, supporting concurrency list, TP list, quantization, MTP, Prefix Cache, parallel breakdown, operator and memory analysis |
| Simulator - VL Forward | Multimodal VL inference simulation, adding image batch, height, width and other image parameters on top of LLM parameters |
| Video Generation | Video generation model inference simulation, supporting Ulysses, CFG, DiT Cache, Chrome Trace and other parameters |
| Optimizer | Service throughput tuning, supporting three deployment modes: `PD Aggregated`, `PD Disaggregated`, `PD Ratio` |

### 3.3 Basic Web UI Workflow

1. Select the model, primary chip, and optional competitor chip.
2. Fill in parameters such as number of devices, concurrency, length, quantization, and parallelism.
3. Click "Preview Configuration" or "Preview Command" to confirm the CLI command that will be executed.
4. Click "Start Run".
5. View the summary conclusion, charts, memory analysis, bandwidth bottleneck, operator details, and export results.
6. If a concurrency list or TP list is configured, select a specific case in the detail analysis area, for example `Concurrency=32 | TP=2`, then view the memory and operator data for that case.

---

<a id="llm-vl-simulation"></a>

## 4. LLM / VL Simulation Guide

Both LLM and VL simulations ultimately invoke:

```bash
python -m cli.inference.text_generate <model_id> [options]
```

where VL adds image input parameters on top of the LLM simulation.

### 4.1 Key Concepts

| Concept | Description |
|---|---|
| `num-queries` | Number of concurrent requests, affecting batch, KV Cache, memory, and throughput |
| `query-length` | Number of newly added tokens. Prefill is usually large; decode is usually 1 or a small value |
| `context-length` | Existing context length, affecting KV Cache and attention cost |
| `decode` | Enable autoregressive decode mode |
| `tp-size` | Tensor Parallel size |
| `dp-size` | Data Parallel size; can be set to `auto` in the Web UI |
| `ep-size` | Expert Parallel size, commonly used for MoE models |
| `num-mtp-tokens` | Number of MTP tokens, available for models that support MTP such as DeepSeek |
| `prefix-cache-hit-rate` | Prefix Cache hit rate, value range `[0,1)`, used to estimate the benefit of prefill token reuse |
| `quantize-linear-action` | Linear layer quantization method, such as `W8A8_DYNAMIC`, `FP8`, `MXFP4` |
| `quantize-non-expert-linear-action` | Non-expert Linear layer quantization override, mainly used for DeepSeek V4; applies to attention projections, dense MLP, and shared experts; routed MoE experts still use `quantize-linear-action` |
| `quantize-attention-action` | KV Cache / Attention quantization method, such as `DISABLED`, `INT8`, `FP8` |
| `image-height/image-width` | VL image dimensions |

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

Suitable for quickly observing single-device inference time, TPS/Device, memory usage, and operator breakdown for a given chip under a typical decode scenario.

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

This scenario focuses on the cost of processing the first input segment, suitable for comparing:

- Whether prefill is affected by communication bottlenecks under different TP configurations.
- Whether attention quantization reduces memory and bandwidth pressure.
- The impact of `compile` on graph compilation and execution time.

### 4.4 Concurrency List Example: Plotting Concurrency Curves

In the Web UI, you can fill in:

```text
Concurrency list: [16,32,64]
TP parallel size: 1
```

This is equivalent to running multiple experiments with different `--num-queries` values. The results area will plot the relationship between concurrency count and inference time, throughput, and so on, suitable for finding the optimal concurrency range.

If using the CLI for batch experiments, you can use a script loop:

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

### 4.5 TP List Example: Sweeping Multiple TP Values for the Same Model

In the Web UI, you can fill in:

```text
Number of devices: 8
Request concurrency: 32
TP list: [1,2,4,8]
```

The tool will sweep through multiple TP values at the same concurrency and output a chart of TP count versus inference time. The x-axis is the TP count, and the y-axis is the inference time.

Suitable for answering:

- Whether increasing TP accelerates computation.
- Whether communication overhead cancels out computation gains.
- The optimal TP range for the current chip and model.

### 4.6 Concurrency List + TP List Example

In the Web UI, you can fill in:

```text
Number of devices: 8
Concurrency list: [16,32,64]
TP list: [1,2]
```

The tool will sweep concurrency for each TP and output a concurrency curve for each TP. The results can be understood as:

| TP | Cases that will run |
|---|---|
| 1 | Concurrency 16, 32, 64 |
| 2 | Concurrency 16, 32, 64 |

Subsequently, the memory, bandwidth, and operator detail areas will show case selectors, for example:

```text
Concurrency=16 | TP=1
Concurrency=32 | TP=1
Concurrency=64 | TP=1
Concurrency=16 | TP=2
Concurrency=32 | TP=2
Concurrency=64 | TP=2
```

When viewing details, please select a chip first and then a specific case; otherwise, it is easy to confuse memory and operator data across different concurrency and TP configurations.

### 4.7 DeepSeek / MTP Example

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

Note: `query-length` must be greater than the number of MTP tokens; otherwise there will not be enough generated tokens to carry out MTP analysis.

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

`prefix-cache-hit-rate=0.5` means estimating a 50% prefix hit at the token level. The higher the hit rate, the shorter the effective prefill length, and typically the lower the TTFT and prefill-side memory pressure.

### 4.9 VL Example: Image Input Inference

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

For VL scenarios, it is recommended to focus on:

- The impact of image dimension changes on memory usage.
- The peak memory when image batch is combined with text concurrency.
- The latency proportion of operators related to the vision tower or multimodal projection.

---

<a id="video-generation-simulation"></a>

## 5. Video Generation Simulation Guide

Video generation entry point:

```bash
python -m cli.inference.video_generate <model_id> [options]
```

This tool simulates the Diffusion Transformer forward process, commonly used for performance estimation of video generation models such as Wan and HunyuanVideo.

### 5.1 Key Parameters

| Parameter | Description |
|---|---|
| `--batch-size` | Video generation batch |
| `--seq-len` | Text prompt token length |
| `--height / --width` | Video resolution |
| `--frame-num` | Number of frames |
| `--sample-step` | Number of denoise steps |
| `--dtype` | `float16`, `float32`, `bfloat16` |
| `--world-size` | Total number of devices |
| `--ulysses-size` | Ulysses sequence parallel size, must evenly divide `world-size` |
| `--use-cfg` | Enable CFG |
| `--cfg-parallel` | Use CFG parallel |
| `--dit-cache` | Enable DiT block cache |
| `--cache-step-range` | Step range for DiT Cache to take effect, format `start,end` |
| `--cache-step-interval` | Refresh cache every N steps; `1` is equivalent to no reuse |
| `--cache-block-range` | Block cache range, format `start,end` |

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

If this is not satisfied, the program will report an error. The Web UI will also validate this in advance.

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

`--use-cfg` simulates classifier-free guidance. `--cfg-parallel` is suitable for comparing the impact of CFG on communication and parallel efficiency.

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

Explanation:

- `--cache-step-range 10,40` means attempting to reuse cache from denoise step 10 through step 40.
- `--cache-step-interval 5` means refreshing the cache every 5 steps, with the remaining steps reusing it.
- `--cache-step-interval 1` effectively disables cache reuse.

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

## 6. Optimizer Throughput Tuning Guide

Throughput tuning entry point:

```bash
python -m cli.inference.throughput_optimizer <model_id> [options]
```

The Optimizer does not just run a single fixed parallel configuration; instead, given a model, device, number of devices, input/output lengths, SLO constraints, and a search space, it automatically searches for better parallel configurations, batch size, concurrency, and throughput.

### 6.1 Three Deployment Modes

The deployment mode names in the Web UI are:

| Web UI Name | CLI Parameter | Applicable Scenario |
|---|---|---|
| `PD Aggregated` | Default, without `--disagg`, without `--enable-optimize-prefill-decode-ratio` | Prefill and Decode are co-deployed in the same instance type; suitable for baselines and cross-chip comparisons |
| `PD Disaggregated` | Add `--disagg` | Prefill and Decode disaggregated analysis; separately evaluating capacity under TTFT or TPOT constraints |
| `PD Ratio` | Add `--enable-optimize-prefill-decode-ratio`, and specify the number of devices per P/D instance | Under a PD disaggregated architecture, finding the optimal Prefill-to-Decode instance ratio |

### 6.2 PD Aggregated: Offline Throughput Tuning

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

Suitable for answering:

- Given 8 devices, what is the theoretical maximum throughput of this model.
- What the optimal TP/DP and batch approximately look like.
- In cross-chip comparisons, which chip achieves higher optimal throughput.

### 6.3 PD Aggregated: Online Service SLO Constraints

Setting both TTFT and TPOT:

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

- Whether TTFT can meet the first-token response target.
- Whether TPOT can meet the sustained generation speed target.
- The optimal batch and concurrency under the given constraints.

### 6.4 Restricting the TP Search Space

By default, the Optimizer will search available TP values. You can also manually restrict them:

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

In the Web UI, the `TP Parallel Size List` can be filled in as:

```text
[1,2,4,8]
```

`batch-range` supports two meanings:

| Syntax | Meaning |
|---|---|
| `--batch-range 256` | min defaults to 1, max is 256 |
| `--batch-range 1 256` | min is 1, max is 256 |

### 6.5 PD Disaggregated: Prefill-Side TTFT Analysis

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

This mode focuses on how many requests the Prefill stage can handle under TTFT constraints.

### 6.6 PD Disaggregated: Decode-Side TPOT Analysis

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

This mode focuses on the sustained output capability of the Decode stage under TPOT constraints.

### 6.7 PD Ratio: Prefill / Decode Instance Ratio Tuning

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

The core idea of PD Ratio is to compute the Prefill QPS and Decode QPS separately, then find a more balanced Prefill / Decode instance ratio.

Approximate understanding:

```text
Prefill QPS = prefill_concurrency / ttft_ms * 1000
Decode QPS  = decode_concurrency / (tpot_ms * output_length) * 1000
PD Ratio    = Decode QPS / Prefill QPS
Balanced QPS = min(Prefill QPS, Decode QPS)
```

When `PD Ratio > 1`, the Decode side is relatively stronger, and more Prefill instances may be needed; when `PD Ratio < 1`, the Decode side may become the bottleneck.

### 6.8 Optimizer Output Interpretation

Typical output includes:

| Field | Description |
|---|---|
| `Best Throughput` | Optimal token/s under the current constraints |
| `TTFT` | Time To First Token, first-token latency |
| `TPOT` | Time Per Output Token, per-output-token latency |
| `concurrency` | Concurrency corresponding to the optimal configuration |
| `parallel` | Parallel configuration, such as `tp4pp1dp2` |
| `batch_size` | Optimal batch |
| `pd_ratio` | Instance ratio in PD Ratio mode |
| `balanced_qps` | System QPS after P/D balancing in PD Ratio mode |

The Web UI also displays:

- Optimal throughput comparison across chips.
- Optimal TTFT / TPOT comparison across chips.
- Fixed-configuration cross-chip comparison.
- PD Ratio key metrics table.
- Single-chip Pareto details.

---

<a id="results-guide"></a>

## 7. How to Read Result Charts and Detail Tables

### 7.1 LLM / VL Results

It is recommended to read in the following order:

1. Summary conclusion: first check the total time, TPS/Device, and whether there are any failures or warnings.
2. Inference time chart: check whether increasing concurrency or TP still reduces latency.
3. Memory analysis: check the proportion of model weights, KV Cache, activations, and reserved memory.
4. Bandwidth bottleneck: check for memory bound, communication bound, and compute bound.
5. Operator details: sorted by total latency to identify the main operators.
6. Operator category statistics: determine optimization direction from categories such as GEMM, Attention, and Communication.

If a concurrency list or TP list is configured, be sure to select a case before viewing details.

### 7.2 Video Results

Key areas of focus:

- The relationship between total analytic time and sample steps.
- The proportion of communication operators after Ulysses.
- Whether CFG / CFG Parallel introduces additional all-gather or batch expansion.
- Whether DiT Cache significantly reduces the computation time of repeated blocks.

### 7.3 Optimizer Results

Recommended reading order:

1. Recommended conclusion: check the optimal chip, throughput, parallel configuration, batch, and concurrency.
2. Optimal comparison across chips: used for cross-chip comparison between the primary chip and competitors.
3. Fixed-configuration cross-chip comparison: ensure the comparison is done under the same configuration, not just comparing each chip's individual optimum.
4. PD Ratio: if using a PD disaggregated architecture, check the Balanced QPS and Prefill / Decode instance ratio.
5. Single-chip Pareto: determine whether there are alternative points with higher throughput but slightly worse latency.

---

## 8. Parameter Selection Recommendations

### 8.1 When You Don't Know Where to Start

LLM decode initial values:

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

LLM prefill initial values:

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

Optimizer online service initial values:

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

- If the model weights are too large to fit in memory: prioritize increasing TP.
- If a single device has a clear compute bottleneck: increasing TP may yield significant gains.
- If communication proportion is high: continuing to increase TP may have diminishing returns.
- For small models or small batches: excessively large TP may slow things down due to communication and synchronization overhead.

It is recommended to first run a TP list of `[1,2,4,8]` in the Web UI, then narrow down the search range based on the curves.

### 8.3 How to Choose Concurrency

Rules of thumb:

- Too low concurrency: device utilization may be insufficient.
- Gradually increasing concurrency: throughput usually improves, but latency and memory also increase.
- Excessively high concurrency: may trigger memory bottlenecks, excessive KV Cache, or unacceptable latency.

It is recommended to use `[16,32,64,128]` for the first round, then perform a finer sweep around the optimal range.

### 8.4 How to Choose Quantization

| Scenario | Recommendation |
|---|---|
| Quick baseline | `W8A8_DYNAMIC` |
| Do not want to introduce quantization effects | `DISABLED` |
| Significant memory pressure | Try `INT8` attention or `FP8` |
| MXFP4 solution evaluation | Use `MXFP4`, adjust `mxfp4-group-size` if necessary |

Note: The simulation tool focuses on performance and resource estimation, and does not replace real accuracy evaluation. Model quality after quantization must still be verified through accuracy testing.

---

<a id="faq"></a>

## 9. FAQ

### 9.1 Browser Cannot Open After Web UI Launch

Check:

- Whether the correct address is used: `http://127.0.0.1:2345`.
- Whether the port is occupied; you can switch to `--port 2346`.

### 9.2 Invalid Device Name

`--device` must come from `DeviceProfile.all_device_profiles`. The Web UI automatically loads the brand and chip list from device profiles. In the CLI, you can check the choices in the error message, or select an available chip in the Web UI first.

### 9.3 Invalid TP / DP / EP Configuration

Common causes:

- `num-devices` is not evenly divisible by `tp-size`.
- `world-size` is not evenly divisible by `ulysses-size`.
- `TP * DP * EP` exceeds the number of deployed devices.
- Certain fine-grained TP/DP parameters do not match the total number of devices.

Recommended approach: first run with a simple configuration, such as `tp-size=1, dp-size=auto, ep-size=1`, then gradually increase parallel complexity.

### 9.4 Optimizer Has No Feasible Solution

Common causes:

- TTFT or TPOT constraints are too strict.
- `max-batched-tokens` is smaller than the effective input length.
- The batch search range is too small.
- Insufficient number of devices or an unsuitable TP search space.
- Excessive reserved memory leading to insufficient available memory.

Recommended approach:

1. First remove TTFT/TPOT constraints to see if an offline optimum can be found.
2. Relax `tpot-limits` or `ttft-limits`.
3. Increase the upper limit of `batch-range`.
4. Check whether `tp-sizes` includes feasible values.
5. Reduce `reserved-memory-gb` or use a stronger device profile.

### 9.5 Results Come from Cache and You Want to Re-run

The Web UI reads cache from `.msmodeling_ui/results.sqlite3` and `.msmodeling_ui/logs/` based on the task hash. If you need to completely re-run, you can clear the corresponding cache directories, or adjust a parameter that affects the simulation to generate a new task hash.

### 9.6 Chart Title Overlapping Content

The current version of the Web UI places chart titles in a separate title area outside the image region, no longer using the Gradio overlay title in the upper-left corner. If you still see the old style, confirm that the browser is not loading the old page, and restart the Web UI.

---

## 10. Recommended Workflow Examples

### 10.1 Example A: Comparing LLM Decode Capabilities of Two Chips

Web UI:

```text
Model: Qwen/Qwen3-32B
Primary chip: ATLAS_800_A2_280T_32G_PCIE
Competitor chip: select another chip
Number of devices: 8
Concurrency list: [16,32,64]
TP list: [1,2,4,8]
Generated token count: 8
Context length: 4500
Decode mode: enabled
Quantization: MLP=W8A8_DYNAMIC, Attention=DISABLED
```

Observe:

- Which chip has lower inference time under the same TP and same concurrency.
- Whether one chip has more obvious communication bottlenecks at high TP.
- Whether the bottlenecks in memory and operator details are consistent.

### 10.2 Example B: Evaluating the Impact of VL Image Dimensions

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

Keep other parameters unchanged and compare:

- Changes in total inference time.
- Changes in memory usage.
- Changes in latency of vision-related operators.

### 10.3 Example C: Video Generation Ulysses Scalability

Test sequentially:

```text
world-size=8, ulysses-size=1
world-size=8, ulysses-size=2
world-size=8, ulysses-size=4
world-size=8, ulysses-size=8
```

Observe:

- Whether total latency decreases as Ulysses increases.
- Whether the proportion of communication operators increases.
- Whether there is an optimal Ulysses rather than bigger being better.

### 10.4 Example D: Online Service Capacity Evaluation

Web UI Optimizer:

```text
Deployment mode:PD Aggregated
Model: Qwen/Qwen3-32B
Number of devices: 8
Input length: 3500
Output length: 1500
TP parallel size list: [1,2,4,8]
Batch range: [1,256]
TTFT: 2000
TPOT: 50
Quantization: MLP=W8A8_DYNAMIC, Attention=INT8
```

Key areas to check in the output:

- Whether a feasible solution exists.
- Whether the optimal throughput, TTFT, and TPOT all meet the targets simultaneously.
- Whether the optimal parallel and batch match deployment expectations.

### 10.5 Example E: PD Ratio Deployment Planning

Web UI Optimizer:

```text
Deployment mode: PD Ratio
Number of devices: 16
Devices per Prefill instance: 4
Devices per Decode instance: 2
Input length: 3500
Output length: 1500
TTFT: 2000
TPOT: 50
```

Observe:

- Balanced QPS.
- Whether Prefill QPS or Decode QPS is lower.
- Whether the recommended number of P/D instances and total devices match actual cluster planning.

---

## 11. Developer Notes

If you want to modify the Web UI, it is recommended to first read:

```text
web_ui/README.md
```

Core file relationships:

```text
web_ui/__init__.py          Package entry point, lazily exposes launch_app
web_ui/app.py               Page layout and event bindings
web_ui/components.py        Reusable components and result areas
web_ui/callbacks.py         Form building, validation, execution, result organization
web_ui/command_builder.py   CLI command and task matrix generation
web_ui/runner.py            Cache, subprocess execution, progress streaming
web_ui/parsers.py           Log parsing
web_ui/result_store.py      SQLite and log caching
web_ui/charts.py            Chart rendering
web_ui/styles.py            Shared CSS, theme helpers, and header styles
web_ui/schemas.py           Data classes shared between builder, runner, parser, and store
web_ui/utils.py             Shared parsing, hashing, and normalization helpers
web_ui/time_tracker.py      Tracking and displaying simulation time information
web_ui/web_ui_start.py      Web UI server launch entry point
```

After modifying frontend functionality, it is recommended to run:

```bash
python -m py_compile web_ui/__init__.py web_ui/app.py web_ui/callbacks.py web_ui/command_builder.py web_ui/components.py web_ui/charts.py web_ui/parsers.py web_ui/result_store.py web_ui/runner.py web_ui/schemas.py web_ui/styles.py web_ui/time_tracker.py web_ui/utils.py web_ui/web_ui_start.py
```

---

## 12. Quick Command Index

Launch Web UI:

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
