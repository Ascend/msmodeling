# Web UI User Guide

> **⚠ Security Notice**
>
> The Web UI service has **no authentication** and binds to the loopback address (`127.0.0.1` / `::1`). Loopback only blocks *remote* network access — TCP loopback is visible to **all users on the same host**. Any local user can call every API, submit jobs, and read results.
>
> **This service is designed for single-user, single-machine use only.** Do NOT run it on shared or multi-user hosts. If the server process runs with elevated privileges (root / admin), this constitutes a local privilege-escalation risk.

This document is intended for daily users of Modeling and developers who are about to integrate the project. Its goal is to help you quickly understand what the tool can do, how to launch simulations from the Web UI or CLI, how to interpret results, and how to configure parameters for different business scenarios.

If you only want to start the Web UI, run the launcher from the repo root (with the venv activated):

```bash
python web_ui/main.py
```

After starting, open `http://127.0.0.1:5173` in your browser. The frontend will automatically proxy `/api` requests to the backend at `http://127.0.0.1:8000`.

---

## Reading Guide

| Goal | Recommended Section |
| --- | --- |
| First time launching Web UI | [3. Web UI Quick Start](#web-ui-quick-start) |
| Configuring LLM / VL simulations | [4. LLM / VL Simulation Guide](#llm-vl-simulation) |
| Configuring video generation simulations | [5. Video Generation Simulation Guide](#video-generation-simulation) |
| Using the throughput optimizer | [6. Optimizer Throughput Tuning Guide](#optimizer-guide) |
| Interpreting results and exporting data | [7. How to Read Result Charts and Detail Tables](#results-guide) |

---

## 1. Tool Positioning

Modeling is a simulation tool for model inference performance analysis. Its core capabilities include:

- Predicting operator latency, memory usage, communication overhead, and overall inference time based on device profiles, without requiring real hardware or a full runtime environment for large models.
- Supporting LLM text inference, VL multimodal inference, video generation Diffusion Transformer inference, and service throughput tuning.
- Supporting cross-chip comparisons to help evaluate performance differences of the same model on different devices.
- Supporting parameter combination analysis for concurrency, TP, quantization, MTP, Prefix Cache, Ulysses, DiT Cache, PD Aggregated, PD Disaggregated, PD Ratio, and more.
- The Web UI provides visual charts, detail tables, case selection, CSV export, job history, Chrome trace download, and historical caching; the CLI is suitable for scripted batch experiments.

The most relevant entry points in the repository are as follows:

| Entry Point | Purpose | Recommended Scenario |
| --- | --- | --- |
| `python web_ui/main.py` | Launch the Vue 3 + FastAPI Web UI (single launcher starts both frontend and backend) | Interactive configuration, result visualization, non-developer users |
| `python -m cli.inference.text_generate` | LLM / VL forward inference simulation | One-off or scripted LLM/VL performance analysis |
| `python -m cli.inference.video_generate` | Video generation model simulation | Diffusion Transformer / Wan / HunyuanVideo scenarios |
| `python -m cli.inference.image_generate` | Image generation model simulation (Transformer denoising stage) | Diffusion Transformer / FLUX / Qwen-Image-Edit scenarios |
| `python -m cli.inference.throughput_optimizer` | Service throughput tuning | Finding optimal parallel and batch under TTFT/TPOT/SLO constraints |

---

## 2. Environment Setup

For complete environment setup steps (cloning the repository, creating a virtual environment, installing dependencies, setting `PYTHONPATH` and Hugging Face access), please refer to the [msModeling Installation Guide](../install_guide/msmodeling_install_guide.md).

If the environment is already set up, launching the Web UI from the repository root generally requires no additional configuration. The tool reads model configurations, with common sources including Hugging Face, ModelScope, or local model directories. If the network cannot access Hugging Face, you can select `modelscope` in the Web UI's `remote-source`, or set the `HF_ENDPOINT` mirror as described in the installation guide.

### 2.1 Additional Web UI Dependencies

The Web UI uses a frontend-backend separation architecture. In addition to the Python dependencies above, you also need to install frontend dependencies:

**Backend dependencies** (already included in the repository root `pyproject.toml`, installed with the main project):

- FastAPI, uvicorn, sqlmodel, alembic, pydantic, etc.

**Frontend dependencies** (requires Node.js ≥ 18 and npm):

```bash
cd web_ui/frontend
npm install
```

This installs Vue 3, Element Plus, ECharts, Pinia, and other frontend libraries. Only needs to be run once. Subsequent `npm run dev` will detect dependency changes automatically.

> **Node.js installation**: If Node.js is not installed, download the LTS version from [nodejs.org](https://nodejs.org/), or use a version manager such as nvm or fnm.

---

<a id="web-ui-quick-start"></a>

## 3. Web UI Quick Start

### 3.1 Launch the Local Page

The Web UI uses a frontend-backend separation architecture.

**Before first launch**, install frontend dependencies (only once):

```bash
cd web_ui/frontend && npm install
```

Also ensure Python dependencies are installed (see [Install Guide](../install_guide/msmodeling_install_guide.md)):

```bash
uv sync  # or pip install -e .
```

Then run the launcher from the repo root (with the venv activated):

```bash
python web_ui/main.py
```

The launcher concurrently starts the frontend (Vite dev server, default port 5173) and the backend (FastAPI, default port 8000), streams both outputs with `[backend]` / `[frontend]` prefixes, and tree-kills both on Ctrl+C.

Open in your browser:

```text
http://127.0.0.1:5173
```

The frontend Vite dev server will automatically proxy `/api` requests to the backend at `http://127.0.0.1:8000`.

### 3.2 Web UI Page Overview

The Web UI is a single-page application (SPA). The top navigation bar provides the following:

| Navigation Button | Description |
| --- | --- |
| Home | Return to the workspace (Console) |
| Docs | Embedded user guide |
| History | View submitted job history, status, and results |
| Locale Switch | Chinese / English real-time switching |
| Theme Switch | Light / Dark real-time switching |

The main workspace **Console** uses a **Tab + vertical split** layout. Three modules share the same page:

| Tab | Capabilities |
| --- | --- |
| Text Generation | LLM / VL forward inference simulation, supporting concurrency list, TP list, quantization, MTP, Prefix Cache, parallel breakdown, operator and memory analysis |
| Video Generation | Video generation model inference simulation, supporting Ulysses, CFG, DiT Cache, Chrome Trace and other parameters |
| Throughput Optimizer | Service throughput tuning, supporting three deployment modes: `PD Aggregated`, `PD Disaggregated`, `PD Ratio` |

Each Tab's workspace is divided into two parts:

- **Upper part**: Configuration form (fields dynamically generated from TypeScript config, grouped into collapsible sections, hover over field names to see bilingual tooltips)
- **Lower part**: Result pane (changes with job status: idle placeholder → running → success result / failure details)
- A draggable divider between them allows resizing the form and result areas

### 3.3 Basic Web UI Workflow

1. Select the model, primary chip, and optional competitor chip.
2. Fill in parameters such as number of devices, concurrency, length, quantization, and parallelism.
3. Click the **▶ Run** button to submit the job. A Toast notification will appear on success (with the job ID).
4. The result pane automatically switches to the running state, showing a spinning icon and progress text. You can view logs or cancel the job at any time.
5. After the job completes, the result pane displays the summary, scatter plots / charts, memory analysis, operator details, etc.
6. If multi-value fields are set (e.g., concurrency list, TP list), the system automatically expands them into multiple cases, and the results are displayed in a multi-case view.
7. Click **History** in the top navigation bar to view all historical jobs and their results.

> **Note**: Tab switching is blocked while a job is running (a warning toast appears). You must wait for it to complete or cancel it before switching tabs.

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
| --- | --- |
| `num-queries` | Number of concurrent requests, affecting batch, KV Cache, memory, and throughput |
| `query-length` | Number of newly added tokens. Prefill is usually large; decode is usually 1 or a small value |
| `context-length` | Existing context length, affecting KV Cache and attention cost |
| `decode` | Enable autoregressive decode mode |
| `tp-size` | Tensor Parallel size |
| `dp-size` | Data Parallel size; can be set to `auto` in the Web UI |
| `ep-size` | Expert Parallel size, commonly used for MoE models |
| `num-mtp-tokens` | Number of MTP tokens, available for models that support MTP such as DeepSeek |
| `prefix-cache-hit-rate` | Prefix Cache hit rate, value range `[0,1)`, used to estimate the benefit of prefill token reuse |
| `quantize-linear-action` | Linear layer quantization method, such as `W8A8_DYNAMIC`, `fp8`, `mxfp4` |
| `quantize-non-expert-linear-action` | Non-expert Linear layer quantization override, mainly used for DeepSeek V4; applies to attention projections, dense MLP, and shared experts; routed MoE experts still use `quantize-linear-action` |
| `quantize-attention-action` | KV Cache / Attention quantization method, such as `disabled`, `int8`, `fp8` |
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
  --quantize-attention-action disabled
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
  --quantize-attention-action int8
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
    --quantize-linear-action mxfp4 \
    --quantize-attention-action disabled
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
| --- | --- |
| 1 | Concurrency 16, 32, 64 |
| 2 | Concurrency 16, 32, 64 |

Subsequently, the result pane automatically switches to the **multi-case view**: a Summary table lists the core metrics for each case (concurrency, TP, inference time, memory, etc.). Clicking a row drills down to the full result for that case (memory distribution chart, operator timing table, etc.).

The old manual case selector has been replaced by automatic multi-case expansion + drill-down interaction.

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
  --quantize-attention-action int8
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
| --- | --- |
| `--batch-size` | Video generation batch |
| `--seq-len` | Text prompt token length |
| `--height / --width` | Video resolution |
| `--frame-num` | Number of frames |
| `--sample-step` | Number of denoise steps |
| `--dtype` | `float16`, `float32`, `bfloat16` |
| `--num-devices` | Total number of devices |
| `--ulysses-size` | Ulysses sequence parallel size, must evenly divide `--num-devices` |
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
  --num-devices 8 \
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
  --num-devices 8 \
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
  --chrome-trace-file trace/video.json
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
| --- | --- | --- |
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
  --quantize-attention-action int8
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
  --quantize-attention-action int8 \
  --ttft-limit 2000 \
  --tpot-limit 50
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
| --- | --- |
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
  --quantize-attention-action disabled \
  --disagg \
  --ttft-limit 2000
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
  --quantize-attention-action disabled \
  --disagg \
  --tpot-limit 50
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
  --quantize-attention-action disabled \
  --enable-optimize-prefill-decode-ratio \
  --prefill-devices-per-instance 4 \
  --decode-devices-per-instance 2 \
  --ttft-limit 2000 \
  --tpot-limit 50 \
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
| --- | --- |
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

The Web UI uses modular result components. Different modules have dedicated result views. The result pane supports both light and dark themes, and charts adapt automatically.

### 7.1 Text Generation Results

The result pane displays the following from top to bottom:

1. **Summary metric cards**: batch_size / execution_time / peak_usage / total_device — key values at a glance.
2. **Simulator run time**: standalone display of the simulator wall-clock time (including compile; not the model execution time).
3. **TPS per Device bar chart**: one bar per chip for multi-chip comparisons.
4. **Memory distribution chart**: visual breakdown of total_device / model_weight / kv_cache / peak_usage / available.
5. **Operator bottleneck distribution (OpBound)**: compact text showing memory bound / communication bound / compute bound proportions.
6. **Operator timing table**: Name / total / avg / # of Calls, sorted by total latency descending, with expandable input shapes and bound analysis.
7. **Chrome Trace downloads**: JSON download links per case / seq index (requires `--chrome-trace-file`).

If multi-value fields are configured (multiple devices, multiple quantization methods, concurrency lists, etc.), the result automatically switches to the **multi-case view**: a Summary table lists core metrics for each case; click to drill down to the full single-case result.

### 7.2 Video Results

Key areas of focus:

- The relationship between total analytic time and sample steps.
- The proportion of communication operators after Ulysses.
- Whether CFG / CFG Parallel introduces additional all-gather or batch expansion.
- Whether DiT Cache significantly reduces the computation time of repeated blocks.
- Operator timing table / chart and Chrome Trace downloads (shared display components with Text Generation).

Multi-case results also show a Summary table + drill-down.

### 7.3 Optimizer Results

The Optimizer displays different views depending on the deployment mode:

**PD Aggregated (AggregatedView)**:

- Scatter plot: Throughput vs Concurrency / TPOT, colored by parallel strategy
- Cross-device optimal throughput comparison bar chart (for multi-device cases)
- Sweep ranking table: rank / throughput / TTFT / TPOT / concurrency / num_devices / parallel / batch_size
- CSV export

**PD Disaggregated (DisaggregatedView)**:

- Prefill table (TTFT-oriented) + Best configuration card
- Decode table (TPOT-oriented) + Best configuration card
- CSV export

**PD Ratio (PDRatioView)**:

- PD Ratio table: PD Ratio / Balanced QPS / P/D QPS / TTFT / TPOT / parallel configuration
- Best PD ratio card

**Scatter Plot (OptimizerCurves)**:

- Data source: all raw exploration points (raw records), colored by parallel strategy
- Automatically filters out-of-memory points (OOM) and duplicate rows
- Mode-aware: 2 charts for Aggregated / 4 charts for Disaggregated / 2 charts for PD Ratio
- Light / dark theme auto-adaptation

**Multi-case View (ThroughputMultiCaseResult)**:

- Summary table (one row per case: device + metrics)
- Click to drill down to single-case full results (scatter plot + mode view)

### 7.4 Job Logs

Click the "Logs" button in the workspace or job status page to open the log drawer (JobLogDrawer):

| Feature | Description |
| --- | --- |
| Full log | Main job log (banner + all cases interleaved output) |
| Per-case log | Independent log filtered by case (radio switch) |
| Log search | Case-insensitive line filter (shows matching lines / total lines) |
| ANSI rendering | Terminal colors → HTML (preserves bold / color / italic / underline) |

### 7.5 History

Click **History** in the top navigation bar to enter the History page:

| Feature | Description |
| --- | --- |
| Job list | Table display: Job ID / Module / Label / Status / Created / Completed |
| Status labels | Color-coded: success (green) / failed (red) / running (blue) / cancelled (yellow) |
| Filtering | Filter by module / status; search by Job ID / label |
| Pagination | Select 10 / 20 / 50 / 100 per page |
| Actions | View result (succeeded) / View status (running) / View details (failed) |

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
quantize-attention-action: disabled
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
quantize-attention-action: int8
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
| --- | --- |
| Quick baseline | `W8A8_DYNAMIC` |
| Do not want to introduce quantization effects | `disabled` |
| Significant memory pressure | Try `int8` attention or `fp8` |
| mxfp4 solution evaluation | Use `mxfp4`, adjust `mxfp4-group-size` if necessary |

Note: The simulation tool focuses on performance and resource estimation, and does not replace real accuracy evaluation. Model quality after quantization must still be verified through accuracy testing.

---

## 9. Developer Notes

If you want to modify the Web UI, it is recommended to first read the design document:

```text
docs/design/web_ui_refactor_design.md
```

### 9.1 Architecture Overview

The Web UI uses a frontend-backend separation architecture:

```text
Browser (Vue 3 SPA)  ──HTTP/JSON──▶  FastAPI Backend  ──subprocess──▶  CLI Core
```

- **Frontend**: Vue 3 + Element Plus + Pinia + ECharts + Vite. Build artifacts are served by the backend via StaticFiles.
- **Backend**: FastAPI + SQLite (WAL mode) + Alembic migrations. Jobs run in isolated subprocesses.
- **Frontend source**: `web_ui/frontend/`
- **Backend source**: `web_ui/backend/`

### 9.2 Core File Relationships

**Frontend**:

```text
web_ui/frontend/src/
├── App.vue                    # Root component (app-bar + router-view)
├── main.ts                    # Entry (Vue + Element Plus + Pinia)
├── router/index.ts            # Routes (Console / History / JobResult / Docs)
├── pages/                     # Route pages (Console / History / JobResult / JobStatus / Docs)
├── components/
│   ├── workspace/             # Workspace (ModuleWorkspace + ResultPane)
│   ├── form/                  # Dynamic form (SchemaForm + SchemaFormItem)
│   ├── result/                # Result components (text / video / throughput subdirs)
│   └── job-status/            # Job status card + log drawer
├── composables/               # Composable functions (useJobRunner / useFormValidation etc.)
├── stores/                    # Pinia stores (formState / telemetry)
├── services/                  # API layer (axios wrappers)
├── config/forms/              # Form config source of truth (.ts files)
└── styles/theme.css           # CSS variable theme
```

**Backend**:

```text
web_ui/backend/
├── main.py                    # FastAPI app + lifespan + uvicorn entry
├── db.py                      # SQLite engine + Alembic migrations
├── api/
│   ├── routers/               # API routes (jobs / cases / modules / options)
│   ├── schemas.py             # Pydantic response models
│   └── errors.py              # Error handling
├── models/                    # Data entities + ORM definitions
├── services/
│   ├── job_manager.py         # Async job management
│   ├── job_runner.py          # Job execution (ThreadPoolExecutor + subprocess)
│   ├── result_view.py         # Result assembly (Top-N + SLO + multi-case)
│   ├── ranking.py             # Rank calculation
│   ├── repositories.py        # Data access layer
│   ├── schema_registry.py     # Form schema snapshots + hash
│   └── capture.py             # Log capture
├── runners/                   # Runner adapters (text_generate / video_generate / throughput_optimizer)
└── migrations/                # Alembic migrations
```

### 9.3 Web Startup

```bash
# Install frontend dependencies first (only once)
cd web_ui/frontend && npm install

# Start the launcher (concurrently runs frontend on :5173 and backend on :8000)
python web_ui/main.py
```

### 9.4 Form Configuration Development

Form field definitions live in `web_ui/frontend/src/config/forms/*.ts` (source of truth). At build time, `npm run gen:schemas` generates data-only JSON for the backend schema_registry to load. After modifying fields, you must bump the version number.

---

## 10. Quick Command Index

Launch Web UI:

```bash
# Install frontend dependencies first (only once)
cd web_ui/frontend && npm install

# Start (single command, runs both frontend and backend)
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
