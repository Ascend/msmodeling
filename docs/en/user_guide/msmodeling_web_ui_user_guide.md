# Web UI User Guide

> **⚠ Security Notice**
>
> The Web UI service has **no authentication mechanism** and binds to the loopback address (`127.0.0.1`/`::1`). A loopback binding only blocks *remote* network access. The TCP loopback is **visible to all users on the local host**. Any user on the same host can call all APIs, submit jobs, and read results.
>
> **This service is intended for single-user, single-host use only.** Do not run it on a multi-user shared host. If the service process runs with high privileges (root/administrator), it poses a local privilege escalation risk.

This document is intended for everyday Modeling users and developers who are about to integrate the project. It helps you quickly understand what the tool can do, how to launch simulations from the Web UI or the CLI, how to interpret results, and how to configure parameters for different business scenarios.

If you only want to start the Web UI, run the launcher in the repository root (with the venv activated):

```bash
python web_ui/main.py
```

After startup, open `http://127.0.0.1:5173` in your browser. The frontend automatically proxies `/api` requests to the backend at `http://127.0.0.1:8000`.

---

## Reading Navigation

| Goal | Recommended Section |
| --- | --- |
| Starting the Web UI for the first time | [3. Web UI Quick Start](#web-ui-quick-start) |
| Configuring LLM/VL simulation | [4. LLM/VL Simulation Guide](#llm-vl-simulation) |
| Configuring video generation simulation | [5. Video Generation Simulation Guide](#video-generation-simulation) |
| Using the throughput optimizer | [6. Optimizer Throughput Optimization Guide](#optimizer-guide) |
| Interpreting results and exporting data | [7. Reading Result Charts and Detail Tables](#results-guide) |

---

## 1 Tool Positioning

Modeling is a simulation tool for model inference performance analysis. Its core capabilities include:

- Predict operator latency, memory usage, communication overhead, and overall inference time based on device profiles without real hardware or a full environment for running a real LLM.
- Support LLM text inference, VL multimodal inference, Diffusion Transformer inference for video generation, and serving throughput optimization.
- Support side-by-side comparison across multiple chips to help determine the performance differences of the same model on different devices.
- Support analysis of parameter combinations such as concurrency, TP, quantization, MTP, Prefix Cache, Ulysses, DiT Cache, PD aggregation, PD disaggregation, and PD ratio.
- The Web UI provides visual charts, detail tables, case selection, CSV export, job history, Chrome trace download, and history cache, while the CLI suits scripted batch experiments.

The entry points most relevant to you in the repository are as follows:

| Entry Point | Purpose | Recommended Use Case |
| --- | --- | --- |
| python web_ui/main.py | Start the Vue 3 + FastAPI Web UI (a single command starts both the frontend and the backend). | Interactive configuration, result visualization, and use by non-developers |
| python -m cli.inference.text_generate | LLM/VL forward inference simulation | Single or scripted LLM/VL performance analysis |
| python -m cli.inference.video_generate | Video generation model simulation | Scenarios such as Diffusion Transformer/Wan/HunyuanVideo |
| python -m cli.inference.image_generate | Image generation model simulation (Transformer denoising stage) | Scenarios such as Diffusion Transformer/FLUX/Qwen-Image-Edit |
| python -m cli.inference.throughput_optimizer | Serving throughput optimization | Find the optimal parallelism and batch under TTFT/TPOT/SLO constraints. |

---

## 2 Environment Preparation

For the complete environment setup steps (cloning the repository, creating a virtual environment, installing dependencies, and configuring `PYTHONPATH` and Hugging Face access), see the [msModeling Installation Guide](../install_guide/msmodeling_install_guide.md).

If the environment is already set up, starting the Web UI from the repository root generally requires no additional configuration. The tool reads model configurations, commonly from Hugging Face, ModelScope, or a local model directory. If Hugging Face is inaccessible from your network, you can select `modelscope` in the `remote-source` field of the Web UI, or configure the `HF_ENDPOINT` mirror as described in the installation guide.

### 2.1 Additional Web UI Dependencies

The Web UI uses a frontend-backend separated architecture. In addition to the preceding Python dependencies, you also need to install the frontend dependencies:

**Backend Dependencies** (already included in the repository root `pyproject.toml` and installed together with the main project):

- FastAPI, uvicorn, sqlmodel, alembic, pydantic, and so on

**Frontend Dependencies** (requires Node.js ≥ 18 and npm):

```bash
cd web_ui/frontend
npm install
```

This command installs frontend libraries such as Vue 3, Element Plus, ECharts, and Pinia. You only need to run it once. Subsequent runs of `npm run dev` automatically detect dependency changes.

> **Node.js installation**: If Node.js is not installed, download an LTS version from [nodejs.org](https://nodejs.org/), or use a version manager such as nvm or fnm.

---

<a id="web-ui-quick-start"></a>

## 3 Web UI Quick Start

### 3.1 Starting the Local Page

The Web UI uses a frontend-backend separated architecture.

**Before the first startup**, install the frontend dependencies (only once):

```bash
cd web_ui/frontend && npm install
```

Also ensure that the Python dependencies are installed (see the [Installation Guide](../install_guide/msmodeling_install_guide.md) for details):

```bash
uv sync  # or pip install -e .
```

Then run the launcher in the repository root (with the venv activated):

```bash
python web_ui/main.py
```

The launcher starts the frontend (Vite dev server, default port 5173) and the backend (FastAPI, default port 8000) concurrently. The output of both is merged and displayed with `[backend]`/`[frontend]` prefixes, and Ctrl+C cleans up the entire process tree at the same time.

Open it in your browser:

```text
http://127.0.0.1:5173
```

The frontend Vite dev server automatically proxies `/api` requests to the backend at `http://127.0.0.1:8000`.

### 3.2 Web UI Page Description

The Web UI is a single-page application (SPA). The top navigation bar provides the following functions:

| Navigation Button | Description |
| --- | --- |
| Home | Return to the workspace (Console). |
| Documentation | The embedded version of this document |
| History | View the history list, status, and results of submitted jobs. |
| Language switch | Switch between Chinese/English in real time. |
| Theme switch | Switch between light/dark in real time. |

The main workspace, **Console**, uses a **Tab + vertically split pane** layout, and the three modules share the same page:

| Tab | Capability |
| --- | --- |
| Text Generation | LLM/VL forward inference simulation, supporting concurrency lists, TP lists, quantization, MTP, Prefix Cache, fine-grained parallelism, operator and memory analysis |
| Video Generation | Video generation model inference simulation, supporting parameters such as Ulysses, CFG, DiT Cache, and Chrome Trace |
| Throughput Optimizer | Serving throughput optimization, supporting three deployment modes: `PD aggregation`, `PD disaggregation`, and `PD ratio` |

The workspace of each Tab is divided into an upper and a lower part:

- **Upper part**: the configuration form (fields are dynamically generated from the TypeScript configuration, groups are collapsible, and you can hover over a field name to view its Chinese and English description)
- **Lower part**: the result pane (changes with the job status: idle placeholder → running → successful result/failure details)
- A draggable splitter in the middle lets you adjust the display ratio between the form and the results

### 3.3 Basic Web UI Operation Process

1. Select the model, the primary chip, and optional competitor chips.
2. Fill in parameters such as the number of devices, concurrency, length, quantization, and parallelism.
3. Click the **▶ Run** button to submit the job. After a successful submission, a Toast notification appears (including the job ID).
4. The result pane automatically switches to the running state, showing a spinning icon and progress text. You can view the logs or cancel the job at any time.
5. After the job completes, the result pane shows the summary conclusion, scatter/line charts, memory analysis, operator details, and so on.
6. If you set multi-value fields (such as a concurrency list or a TP list), the system automatically expands them into multiple cases, and the result area groups and displays them in a multi-case view.
7. Click **History** in the top navigation bar to view all historical jobs. Click a job to view its results directly.

> **Tip**: You cannot switch Tabs while a job is running (a warning appears). Wait for the job to complete or cancel it before switching.

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
| --- | --- |
| num-queries                          | Number of concurrent requests, affecting batch, KV Cache, memory, and throughput |
| query-length                         | Number of new tokens in this run. Prefill is usually larger, and decode is usually 1 or a small value |
| context-length                       | Existing context length, affecting KV Cache and attention cost |
| decode                               | Enabling autoregressive decode mode |
| tp-size                              | Number of Tensor Parallel workers |
| dp-size                              | Number of Data Parallel workers. You can enter `auto` in the Web UI |
| ep-size                              | Number of Expert Parallel workers, commonly used for MoE models |
| num-mtp-tokens                       | Number of MTP tokens, available for models that support MTP, such as DeepSeek |
| prefix-cache-hit-rate                | Prefix Cache hit rate, in the range [0,1), used to estimate the reuse benefit of prefill tokens |
| quantize-linear-action               | Quantization method for Linear layers, for example, `W8A8_DYNAMIC`, `FP8`, `MXFP4`, or `HIF8` |
| quantize-non-expert-linear-action    | Quantization override for non-expert Linear layers, mainly used for DeepSeek V4. It applies to attention projections, dense MLP, and shared experts, while routed MoE experts still use `quantize-linear-action` |
| quantize-attention-action            | Quantization method for KV Cache/Attention, for example, `DISABLED`, `INT8`, `FP8`, or `HIF8` |
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
  --quantize-attention-action disabled
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
  --quantize-attention-action int8
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
    --quantize-linear-action mxfp4 \
    --quantize-attention-action disabled
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
| --- | --- |
| 1 | Concurrency 16, 32, and 64 |
| 2 | Concurrency 16, 32, and 64 |

Afterward, the result pane automatically switches to the **multi-case view**: the Summary table lists the core metrics of each case (such as concurrency, TP, inference time, and memory), and you can click a row to drill down to the complete results of that case (memory distribution chart, operator time table, and so on).

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
  --quantize-attention-action int8
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
| --- | --- |
| --batch-size             | Video generation batch |
| --seq-len                | Text prompt token length |
| --height/--width         | Video resolution |
| --frame-num              | Number of frames |
| --sample-step            | Number of denoise steps |
| --dtype                  | `float16`, `float32`, or `bfloat16` |
| --num-devices            | Total number of devices |
| --ulysses-size           | Ulysses sequence parallel size; without CFG parallel, must equal `--num-devices`; with CFG parallel, `--num-devices` must be twice this value |
| --use-cfg                | Enable CFG |
| --cfg-parallel           | Use CFG parallelism |
| --dit-cache              | Enable DiT block cache |
| --cache-step-range       | Step range in which DiT Cache takes effect, in the `start,end` format |
| --cache-step-interval    | Refresh the cache every N steps. 1 is equivalent to no reuse. |
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
  --num-devices 4 \
  --ulysses-size 4 \
  --dtype float16
```

Parallel configuration requirements:

```text
without CFG parallel: num-devices == ulysses-size
with CFG parallel:    --use-cfg and num-devices == 2 * ulysses-size
```

The CLI and Web UI reject invalid combinations before inference.

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
  --num-devices 8 \
  --ulysses-size 4 \
  --use-cfg \
  --cfg-parallel
```

`--use-cfg` simulates classifier-free guidance and is required with `--cfg-parallel`. With CFG parallel, device ranks are paired as `[u, U + u]` for each Ulysses rank `u` in `[0, U)`, where `U` is `--ulysses-size`.

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
  --chrome-trace-file trace/video.json
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
| --- | --- | --- |
| PD aggregation       | Default. Do not add `--disagg` or `--enable-optimize-prefill-decode-ratio`. | Prefill and decode are deployed together on the same type of instance. Run a baseline and compare across multiple chips first. |
| PD disaggregation    | Add `--disagg` | Analyze prefill and decode separately, evaluating their capabilities under TTFT or TPOT constraints. |
| PD ratio             | Add `--enable-optimize-prefill-decode-ratio` and specify the number of devices per P/D instance. | In a PD-disaggregation architecture, find the ratio of prefill to decode instances. |

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
  --quantize-attention-action int8
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
  --quantize-attention-action int8 \
  --ttft-limit 2000 \
  --tpot-limit 50
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
| --- | --- |
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
  --quantize-attention-action disabled \
  --disagg \
  --ttft-limit 2000
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
  --quantize-attention-action disabled \
  --disagg \
  --tpot-limit 50
```

This mode focuses on the sustained output capability of the decode phase under TPOT constraints.

### 6.7 PD Ratio: Optimizing the Prefill/Decode Instance Ratio

```bash
python -m cli.inference.throughput_optimizer Qwen/Qwen3-32B \
  --device ATLAS_850_486T_112G_FM16 \
  --num-devices 8 \
  --input-length 500 \
  --output-length 100 \
  --compile \
  --quantize-linear-action W8A8_DYNAMIC \
  --quantize-attention-action disabled \
  --enable-optimize-prefill-decode-ratio \
  --prefill-devices-per-instance 4 \
  --decode-devices-per-instance 4 \
  --ttft-limit 10000 \
  --tpot-limit 3000 \
```

The core idea of PD ratio is to calculate the Prefill QPS and Decode QPS separately, and then find a more balanced prefill/decode instance ratio.

For a rough understanding:

```text
Prefill QPS = prefill_concurrency / prefill_phase_makespan_ms * 1000
Decode QPS  = decode_concurrency / (tpot_ms * output_length) * 1000
PD Ratio = Decode QPS / Prefill QPS
Balanced QPS = min(Prefill QPS, Decode QPS)
```

Prefill QPS uses the Prefill phase makespan (`prefill_phase_makespan_ms`). For PP>1 (pipeline parallelism), the throughput of the result row is calculated based on the steady-state wave interval; in that case, Prefill QPS is taken as `token/s / input_length` (full prompt token basis, consistent with the throughput numerator).

When `PD ratio > 1`, the decode side is relatively stronger and may need more prefill instances. When `PD ratio < 1`, the decode side may become the bottleneck.

### 6.8 Interpreting Optimizer Output

Typical output includes:

| Field              | Description |
| --- | --- |
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

The Web UI displays results with modular components, and each module has its own result view. The result pane supports light/dark themes, and charts adapt automatically.

### 7.1 Text Generation Results

The result pane displays the following from top to bottom:

1. **Summary metric cards**: key values such as `batch_size`/`execution_time`/`peak_usage`/`total_device` are visible at a glance.
2. **Simulator program time**: displays the simulator wall-clock time independently (including compilation, not the model execution time).
3. **TPS/Device bar chart**: one bar per device when comparing multiple chips.
4. **Memory distribution chart**: a visual breakdown of `total_device`/`model_weight`/`kv_cache`/`peak_usage`/`available`.
5. **Operator bottleneck distribution (OpBound)**: shows the proportions of memory bound/communication bound/compute bound in compact text.
6. **Operator time table**: Name/total/avg/# of Calls, sorted in descending order of time, with expandable input shapes and bound analysis.
7. **Chrome Trace download**: provides JSON download links indexed by case/seq (requires `--chrome-trace-file` to be enabled).

If you configured multi-value fields (multiple devices, multiple quantizations, a concurrency list, and so on), the results automatically switch to the **multi-case view**: the Summary table lists the core metrics of each case, and you can click a row to drill down to the complete results of a single case.

### 7.2 Video Results

Focus on:

- The relationship between total analytic time and sample steps.
- The proportion of communication operators after Ulysses.
- Whether CFG/CFG Parallel introduces extra all-gather or batch expansion.
- Whether DiT Cache significantly reduces the compute time of repeated blocks.
- The operator time table/charts and the Chrome Trace download (they share the same display components as Text Generation).

Multiple cases also show a Summary table plus drill-down.

### 7.3 Optimizer Results

The Optimizer displays different views depending on the deployment mode:

**PD Aggregation (`AggregatedView`)**:

- Scatter chart: Throughput vs Concurrency/TPOT, colored and grouped by parallel strategy
- Bar chart comparing the optimal throughput across devices (with multiple devices)
- Sweep ranking table: `rank`/`throughput`/`TTFT`/`TPOT`/`concurrency`/`num_devices`/`parallel`/`batch_size`
- CSV export

**PD Disaggregation (`DisaggregatedView`)**:

- Prefill table (TTFT-oriented) + Best configuration card
- Decode table (TPOT-oriented) + Best configuration card
- CSV export

**PD Ratio (`PDRatioView`)**:

- PD Ratio table: PD Ratio/Balanced QPS/P/D QPS/TTFT/TPOT/parallel configuration
- Best PD ratio card

**Scatter Charts (`OptimizerCurves`)**:

- The data source is the full set of explored points (raw records), colored by parallel strategy
- Automatically filters out out-of-memory (OOM) points and duplicate rows
- Mode-aware: 2 charts for aggregation/4 charts for disaggregation/2 charts for PD ratio
- Automatically adapts to light/dark themes

**Multi-Case View (`ThroughputMultiCaseResult`)**:

- Summary table (one row per case: device + metrics)
- Click to drill down to the complete results of a single case (including scatter charts + mode views)

### 7.4 Job Logs

Click the "Logs" button in the workspace or on the job status page to open the log drawer (`JobLogDrawer`):

| Function | Description |
| --- | --- |
| Full logs | The main job log (banner + interleaved output of all cases) |
| Per-case logs | Independent logs filtered by case (switched with radio buttons) |
| Log search | Case-insensitive line filtering (shows the number of matching lines/total lines) |
| ANSI rendering | Terminal color output → HTML (preserves bold/color/italic/underline) |

### 7.5 History

Click **History** in the top navigation bar to enter the History page:

| Function | Description |
| --- | --- |
| Job list | Table display: Job ID/module/label/status/creation time/completion time |
| Status labels | Color-coded: succeeded (green)/failed (red)/running (blue)/canceled (yellow) |
| Filtering | Filter by module/status, and search by Job ID/label. |
| Pagination | Choose 10/20/50/100 entries per page. |
| Actions | View results (succeeded)/view status (running)/view details (failed). |

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
quantize-attention-action: disabled
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
quantize-attention-action: int8
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
| --- | --- |
| Quick baseline | `W8A8_DYNAMIC` |
| No quantization impact desired | `disabled` |
| Obvious memory pressure | Try `int8` attention or `fp8` |
| mxfp4 scheme evaluation | Use `mxfp4`, and adjust `mxfp4-group-size` when necessary. |

For A5 HIF8 evaluation, select an A5 device and set `quantize-linear-action` to `HIF8`. The default recipe uses dynamic per-token activations, per-output-channel weights, and FP32 scales. Use `quantize-non-expert-linear-action` for non-expert overrides.

Attention/KV is configured independently through `quantize-attention-action`. Selecting `HIF8` stores the KV cache in HIF8 and performs Attention computation in BF16 (performance simulation only); it does not select HIF8 Attention matrix computation. The Linear quantization option does not automatically change the Attention/KV setting.

Note: The simulation tool focuses on performance and resource estimation and does not replace real accuracy evaluation. The quality of the quantized model still needs to be verified through accuracy tests.

---

## 9 Additional Notes for Developers

If you plan to modify the Web UI, you are advised to first read the design document:

```text
https://gitcode.com/Ascend/msmodeling.wiki/blob/main/design/web_ui_refactor_design.md
```

### 9.1 Architecture Overview

The Web UI uses a frontend-backend separated architecture:

```text
Browser (Vue 3 SPA)  ──HTTP/JSON──▶  FastAPI backend  ──subprocess──▶  CLI core
```

- **Frontend**: Vue 3 + Element Plus + Pinia + ECharts + Vite. The backend mounts the build artifacts with StaticFiles
- **Backend**: FastAPI + SQLite (WAL mode) + Alembic migrations. Jobs run in independent subprocesses
- **Frontend source**: `web_ui/frontend/`
- **Backend source**: `web_ui/backend/`

### 9.2 Core File Relationships

**Frontend**:

```text
web_ui/frontend/src/
├── App.vue                    # Root component (app-bar + router-view)
├── main.ts                    # Entry point (Vue + Element Plus + Pinia)
├── router/index.ts            # Routes (Console / History / JobResult / Docs)
├── pages/                     # Route pages (Console / History / JobResult / JobStatus / Docs)
├── components/
│   ├── workspace/             # Workspace (ModuleWorkspace + ResultPane)
│   ├── form/                  # Dynamic form (SchemaForm + SchemaFormItem)
│   ├── result/                # Result components (text / video / throughput subdirectories)
│   └── job-status/            # Job status card + log drawer
├── composables/               # Composables (useJobRunner / useFormValidation, and so on)
├── stores/                    # Pinia stores (formState / telemetry)
├── services/                  # API layer (axios wrapper)
├── config/forms/              # Form configuration source of truth (.ts files)
└── styles/theme.css           # CSS variable themes
```

**Backend**:

```text
web_ui/backend/
├── main.py                    # FastAPI app + lifespan + uvicorn entry point
├── db.py                      # SQLite engine + Alembic migrations
├── api/
│   ├── routers/               # API routes (jobs / cases / modules / options)
│   ├── schemas.py             # Pydantic response models
│   └── errors.py              # Exception handling
├── models/                    # Data entities + ORM definitions
├── services/
│   ├── job_manager.py         # Asynchronous job management
│   ├── job_runner.py          # Job execution (ThreadPoolExecutor + subprocess)
│   ├── result_view.py         # Result assembly (Top-N + SLO + multi-case)
│   ├── ranking.py             # Ranking computation
│   ├── repositories.py        # Data access layer
│   ├── schema_registry.py     # Form schema snapshot + hash
│   └── capture.py             # Log capture
├── runners/                   # Runner adapters (text_generate / video_generate / throughput_optimizer)
└── migrations/                # Alembic migrations
```

### 9.3 Web Startup

```bash
# Install the frontend dependencies on first use (only once)
cd web_ui/frontend && npm install

# Start (a single command starts the frontend and backend concurrently)
python web_ui/main.py
```

### 9.4 Form Configuration Development

Form fields are defined in `web_ui/frontend/src/config/forms/*.ts` (the source of truth). During the build, `npm run gen:schemas` generates data-only JSON for the backend schema_registry to load. After you modify fields, you must bump the version number.

---

## 10 Quick Command Index

Start the Web UI:

```bash
# Install the frontend dependencies on first use (only once)
cd web_ui/frontend && npm install

# Start (a single command starts the frontend and backend concurrently)
python web_ui/main.py
```

Open `http://127.0.0.1:5173` in your browser.

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
python -m cli.inference.throughput_optimizer Qwen/Qwen3-32B --device ATLAS_800_A2_280T_32G_PCIE --num-devices 8 --input-length 3500 --output-length 1500 --tp-sizes 1 2 4 8 --batch-range 1 256 --ttft-limit 2000 --tpot-limit 50
```
