# TensorCast User Guide

For the complete model list and feature details, see [Model Support and Feature Support Matrix](./support_matrix/support_matrix_user_guide.md).

## Reading Path

| Goal | Recommended Section |
| --- | --- |
| 1. Quickly run an LLM text generation simulation | [2.1 Quick Start: Text Generation](#21-quick-start-text-generation) |
| 2. Understand timing, call count, and memory metrics in the output | [2.2 Result (Text Generation)](#22-result-text-generation) |
| 3. Run video generation model simulation | [2.3 Quick Start: Video Generation](#23-quick-start-video-generation) |
| 4. Run image generation model simulation | [2.5 Quick Start: Image Generation](#25-quick-start-image-generation) |
| 5. View or customize hardware device profiles | [3 Supported Devices and Custom Devices](#3-supported-devices-and-custom-devices) |

## 1 Introduction

TensorCast is a performance simulation and analysis framework for PyTorch programs. It empowers developers and researchers to predict the performance of their neural network models on specific hardware configurations without needing access to the physical machine.

At its core, TensorCast operates as a "virtual machine" or a runtime simulator. Instead of executing computations on a live accelerator, it intercepts a PyTorch program's computational graph and simulates its execution on a user-defined MachineConfig. This configuration specifies the target hardware's characteristics, such as theoretical compute power (TFLOPS), memory bandwidths, cache hierarchies, and interconnect speeds. In order to accurately estimate the optimal performance of the model on the given HW, TensorCast provides a model optimization pipeline including automatic model sharding, quantization and FX-graph optimization converting the source program into an optimal one before conducting the analysis.

By running a model on this "virtual" hardware, TensorCast provides detailed performance insights, including:

- Out-of-the-box support for Huggingface transformer models.
- Support various hardware accelerator devices with simple configurations.
- Operator-level execution time: Estimated using extensible models like analytic roofline model, empirical data, or ML-based predictors.
- Memory footprint: Tracks total and peak memory allocation.
- Computational characteristics: Analyzes FLOPs (Floating Point Operations) and memory access volume for each operator.
- Advanced Scheduling Simulation: Models complex execution patterns like concurrent computations across multiple streams.

The final output includes both comprehensive summary tables and detailed Chrome Trace files, allowing for deep visualization and identification of performance bottlenecks.

Before first use, see [Quick Start: Environment Setup and First Simulation](../install_guide/msmodeling_install_guide.md) to set up the environment and run an LLM inference simulation.

## 2 At a Glance

### 2.1 Quick Start: Text Generation

**What it does:** Simulate LLM inference performance for a batch of queries.

#### Prefill Scenario

To run prefill for Qwen3-32B on A2 with two requests, each with 3500 input tokens, run:

```bash
python -m cli.inference.text_generate Qwen/Qwen3-32B --num-queries 2 --query-length 3500 --context-length 3500 --device TEST_DEVICE --compile
```

In prefill mode, omit `--decode`; `--query-length` is the new input length and `--context-length` is the context length for each query.

You can also use different quantization schemes for linear layers. For example, use W8A8 dynamic quantization with a 4500-token context prefix:

```bash
python -m cli.inference.text_generate Qwen/Qwen3-32B --num-queries 2 --query-length 3500 --context-length 4500 --device TEST_DEVICE --quantize-linear-action W8A8_DYNAMIC --compile
```

#### Decode Scenario

The decode scenario is similar; only adjust the input length `--query-length` and the requested context length `--context-length`. When MTP is not enabled, `--query-length` is usually `1`; when `--num-mtp-tokens` is enabled, set `--query-length` to `1 + --num-mtp-tokens`.

```bash
python -m cli.inference.text_generate Qwen/Qwen3-32B --num-queries 10 --query-length 1 --context-length 4500 --decode --device TEST_DEVICE --quantize-linear-action W8A8_STATIC --compile
```

**Output:** A performance summary table; optionally a Chrome trace file if `--chrome-trace-file` is set.

### 2.2 Result (Text Generation)

Example output (truncated):

```text
Model compilation and execution time: 0.192 s
----------------------------------------------  --------------  ------------  ----------
                     Name                       analytic total  analytic avg  # of Calls
----------------------------------------------  --------------  ------------  ----------
tensor_cast.static_quant_linear.default              884.004ms       1.973ms         448
tensor_cast.attention.default                        259.855ms       4.060ms          64
aten.mul.Tensor                                      198.215ms     237.668us         834
aten._to_copy.default                                100.528ms     195.580us         514
tensor_cast.dynamic_quantize_symmetric.default        76.519ms     170.802us         448
...
Total time for analytic: 1.744s
[analytic] Execution time: 1.744174 s
[analytic] TPS/Device: 4013 token/s
Total device memory: 64.000 GB
  Model weight size: 31.981 GB
  KV cache: 1.719 GB
  Model activation size: 0.601 GB
  Reserved memory: 0.000 GB
  Memory available: 29.699 GB
```

Note: `Model compilation and execution time` is the simulator's runtime on the host, not the real model compile or execution time on hardware.

Metric descriptions:

- `analytic total`: Estimated total time spent by the operator.
- `analytic avg`: Average time per operator call.
- `# of Calls`: Number of times the operator is invoked.
- `Total time for analytic`: Sum of analytic operator time.
- `TPS/Device`: Tokens per second per device.
- `Total device memory` and breakdowns: Estimated memory usage by weights, KV cache, and activations.

### 2.3 Quick Start: Video Generation

**What it does:** Simulate the diffusion transformer forward pass for video generation models. The following example uses the Wan2.2 Diffusers remote model ID; on first run, the required model configuration files are downloaded according to the configured model source.

**Command:**

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

To run the block sparse attention (BSA) backend, retain the dense example above for the default path and add the BSA-only options:

```bash
python -m cli.inference.video_generate Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --device ATLAS_800_A2_280T_32G_PCIE \
  --batch-size 1 \
  --seq-len 128 \
  --attention-backend block_sparse_attention \
  --attention-block-size 128 \
  --attention-sparsity 0.5
```

**Key flags:** `model_id`, `--device`, `--batch-size`, `--seq-len`, `--height`, `--width`, `--frame-num`, `--sample-step`, `--dtype`, `--quantize-linear-action`, `--chrome-trace-file`, `--attention-backend`, `--attention-block-size`, `--attention-sparsity`

**Output:** A performance summary table; optionally a Chrome trace file if `--chrome-trace-file` is set.

### 2.4 Result (Video Generation)

Example output (truncated; actual values vary by device profile, model configuration, and input shape):

```text
Model compilation and execution time: 96.06264850008301s
----------------------------------------------  --------------  ------------  ----------
                     Name                       analytic total  analytic avg  # of Calls
----------------------------------------------  --------------  ------------  ----------
tensor_cast.attention.default                        1363.587s     340.897ms        4000
tensor_cast.static_quant_linear.default               231.521s      11.576ms       20000
aten._to_copy.default                                 150.176s       3.398ms       44200
aten.mul.Tensor                                       138.593s       3.448ms       40200
aten.add.Tensor                                        76.611s       3.802ms       20150
tensor_cast.dynamic_quantize_symmetric.default         42.740s       2.137ms       20000
aten.native_layer_norm.default                         35.517s       5.871ms        6050
aten.pow.Tensor_Scalar                                 35.240s       4.405ms        8000
aten.mean.dim                                          17.631s       2.204ms        8000
aten.copy_.default                                     17.618s       2.202ms        8000
aten.gelu.default                                      15.846s       7.730ms        2050
...
Total time for analytic: 2145.882s
```

Note: `Model compilation and execution time` is the simulator's runtime on the host, not the real model compile or execution time on hardware.

Metric descriptions:

- `analytic total`: Estimated total time spent by the operator.
- `analytic avg`: Average time per operator call.
- `# of Calls`: Number of times the operator is invoked.
- `Total time for analytic`: Sum of analytic operator time.

### 2.5 Quick Start: Image Generation

**What it does:** Simulate the diffusion transformer denoising workload for image generation models. The first version only simulates the Transformer denoising stage; prompt encoding, VAE, scheduler, and image I/O are excluded, and no real image is produced. The following example uses the FLUX.1-dev remote model ID; on first run, the required model configuration files are downloaded according to the configured model source.

**Command:**

```bash
python -m cli.inference.image_generate black-forest-labs/FLUX.1-dev \
  --device ATLAS_800_A2_280T_32G_PCIE \
  --batch-size 1 \
  --output-image-size 512 512 \
  --text-seq-len 512 \
  --sample-step 50 \
  --dtype float16 \
  --quantize-linear-action W8A8_DYNAMIC
```

**Key flags:** `model_id` / `--model-id`, `--device`, `--batch-size`, `--output-image-size`, `--text-seq-len`, `--source-image-size`, `--sample-step`, `--use-cfg`, `--num-devices`, `--ulysses-size`, `--cfg-parallel`, `--dit-cache`, `--chrome-trace-file`

**Output:** A performance summary table; optionally a Chrome trace file if `--chrome-trace-file` is set.

### 2.6 Result (Image Generation)

`image_generate` reports the critical path and logical measured work of the Transformer denoising stage, and only outputs a performance summary table and an optional Chrome trace. The output metrics are the same as for video generation:

- `analytic total`: Estimated total time spent by the operator.
- `analytic avg`: Average time per operator call.
- `# of Calls`: Number of times the operator is invoked.
- `Total time for analytic`: Sum of analytic operator time.
- `Model compilation and execution time`: The simulator's runtime on the host, not the real model compile or execution time on hardware.

Actual values vary by device profile, model configuration, and input shape.


## 3 Supported Devices and Custom Devices

We provide built-in support for the following device profiles (defined in `tensor_cast/device.py`):

- `TEST_DEVICE`
- `ATLAS_800_A2_376T_64G`
- `ATLAS_800_A2_313T_64G`
- `ATLAS_800_A2_280T_64G`
- `ATLAS_800_A2_280T_64G_PCIE`
- `ATLAS_800_A2_280T_32G_PCIE`
- `ATLAS_800_A3_752T_128G_DIE`
- `ATLAS_800_A3_560T_128G_DIE`
- `ATLAS_800_A3_560T_128G_DIE_ROCE`
- `ATLAS_350_425T_112G`
- `ATLAS_350_425T_84G`

### 3.1 Custom Device Types

For other hardware, define a custom device profile as a Python file under `tensor_cast/device_profiles`. TensorCast will load it automatically, and you can then reference the profile name from the CLI. Custom device guide: [device_profiles/README.md](../../../tensor_cast/device_profiles/README.md)

## 4 Detailed Usage

### Model Source Security Recommendation

We recommend local safe mode: download and review the model repository first,
then pass `model_id` as a complete absolute local path, for example
`/data/models/Qwen3-32B`. Local path loading validates ownership, symlinks, and
permissions before use. Avoid symlinked directories, shared writable
directories, or unreviewed model files.

The tools still support Hugging Face or ModelScope model ids such as
`Qwen/Qwen3-32B`, with `--remote-source` selecting the source. This model id
mode may execute remote Python code when falling back to
`trust_remote_code=True`; msmodeling does not provide security guarantees for
remote model code, and the runtime prints a `trust_remote_code` risk warning.

### 4.1 Run text generation with given query length

We provide a `text_generate.py` command line interface to simulate the text generation. The script supports text generation with a batch of queries with the same input length and optionally same context length. The table summary of op performance breakdown is provided by default. An option is also provided to dump the chrome trace.

Its general usage is shown below:

```text
usage: text_generate.py [-h]
                        [--device {TEST_DEVICE,ATLAS_800_A2_376T_64G,ATLAS_800_A2_313T_64G,ATLAS_800_A2_280T_64G,ATLAS_800_A2_280T_64G_PCIE,ATLAS_800_A2_280T_32G_PCIE,ATLAS_800_A3_752T_128G_DIE,ATLAS_800_A3_560T_128G_DIE,ATLAS_800_A3_560T_128G_DIE_ROCE,ATLAS_350_425T_112G,ATLAS_350_425T_84G}]
                        [--num-devices NUM_DEVICES] [--reserved-memory-gb RESERVED_MEMORY_GB]
                        [--log-level {debug,info,warning,error,critical}] --num-queries NUM_QUERIES
                        --query-length QUERY_LENGTH [--context-length CONTEXT_LENGTH] [--decode]
                        [--prefix-cache-hit-rate PREFIX_CACHE_HIT_RATE] [--num-mtp-tokens NUM_MTP_TOKENS]
                        [--no-repetition] [--compile] [--compile-allow-graph-break]
                        [--compilation-config [{enable_multistream,enable_sequence_parallel,enable_matmul_allreduce,enable_dispatch_ffn_combine} ...]]
                        [--quantize-linear-action {DISABLED,W8A16_STATIC,W8A8_STATIC,W4A8_STATIC,W8A16_DYNAMIC,W8A8_DYNAMIC,W4A8_DYNAMIC,FP8,MXFP4}]
                        [--quantize-non-expert-linear-action {DISABLED,W8A16_STATIC,W8A8_STATIC,W4A8_STATIC,W8A16_DYNAMIC,W8A8_DYNAMIC,W4A8_DYNAMIC,FP8,MXFP4}]
                        [--quantize-lmhead] [--mxfp4-group-size mxfp4_GROUP_SIZE]
                        [--quantize-attention-action {DISABLED,INT8,FP8}] [--graph-log-path GRAPH_LOG_DIR]
                        [--dump-input-shapes] [--dump-op-bound-results] [--chrome-trace-file CHROME_TRACE]
                        [--num-hidden-layers-override NUM_HIDDEN_LAYERS_OVERRIDE] [--tp-size TP_SIZE]
                        [--dp-size DP_SIZE] [--ep-size EP_SIZE] [--o-proj-tp-size O_PROJ_TP_SIZE]
                        [--o-proj-dp-size O_PROJ_DP_SIZE] [--mlp-tp-size MLP_TP_SIZE] [--mlp-dp-size MLP_DP_SIZE]
                        [--lmhead-tp-size LMHEAD_TP_SIZE] [--lmhead-dp-size LMHEAD_DP_SIZE]
                        [--moe-tp-size MOE_TP_SIZE] [--moe-dp-size MOE_DP_SIZE] [--word-embedding-tp {col,row}]
                        [--enable-redundant-experts] [--enable-shared-expert-tp] [--enable-external-shared-experts]
                        [--host-external-shared-experts] [--vision-tp-size VISION_TP_SIZE]
                        [--image-batch-size IMAGE_BATCH_SIZE] [--image-height IMAGE_HEIGHT]
                        [--image-width IMAGE_WIDTH]
                        [--remote-source {huggingface,modelscope}] [--performance-model {analytic,profiling}]
                        [--profiling-database-path PROFILING_DATABASE]
                        [--export-empirical-metrics EXPORT_EMPIRICAL_METRICS]
                        model_id

Run a simulated LLM inference pass and dump the perf result.
```

Main parameters:

| Parameter | Category | Required/Optional | Description |
| --- | --- | --- | --- |
| `model_id` | General Options | Required | Model ID or local model path.<br>1. Type: Str.<br>2. Reference values: Hugging Face ID, ModelScope ID, or local absolute path, such as `Qwen/Qwen3-32B` or `/data/models/Qwen3-32B`.<br>3. Default: none.<br>4. When a remote model ID is used, remote code may be executed through `trust_remote_code=True`. |
| `--device` | General Options | Optional | Specifies the device profile for simulation.<br>1. Type: Str.<br>2. Reference values: registered `DeviceProfile` names, including `TEST_DEVICE`, `ATLAS_800_A2_376T_64G`, `ATLAS_800_A2_313T_64G`, `ATLAS_800_A2_280T_64G`, `ATLAS_800_A2_280T_64G_PCIE`, `ATLAS_800_A2_280T_32G_PCIE`, `ATLAS_800_A3_752T_128G_DIE`, `ATLAS_800_A3_560T_128G_DIE`, `ATLAS_800_A3_560T_128G_DIE_ROCE`, `ATLAS_350_425T_112G`, `ATLAS_350_425T_84G`.<br>3. Default: `TEST_DEVICE`. |
| `--num-devices` | General Options | Optional | Specifies the number of devices participating in simulation.<br>1. Type: Int.<br>2. Valid range: positive integer.<br>3. Default: `1`. |
| `--reserved-memory-gb` | General Options | Optional | Specifies device memory reserved for system use, in GB.<br>1. Type: Float.<br>2. Valid range: non-negative number; set to `0` to disable memory reservation.<br>3. Default: `0.0`. |
| `--log-level` | General Options | Optional | Specifies the log level.<br>1. Type: Str.<br>2. Reference values: `debug`, `info`, `warning`, `error`, `critical`.<br>3. Default: `error`. |
| `--num-queries` | LLM Options | Required | Number of queries in this simulation.<br>1. Type: Int.<br>2. Valid range: positive integer.<br>3. Default: none. |
| `--query-length` | LLM Options | Required | New input token length for each query.<br>1. Type: Int.<br>2. Valid range: positive integer.<br>3. Default: none. |
| `--context-length` | LLM Options | Optional | Existing context token length for each query.<br>1. Type: Int.<br>2. Valid range: non-negative integer.<br>3. Default: `0`. |
| `--decode` | LLM Options | Optional | Enables autoregressive decode mode; omit it for prefill mode.<br>1. Type: Bool.<br>2. Valid range: flag option.<br>3. Default: `False`. |
| `--prefix-cache-hit-rate` | LLM Options | Optional | Specifies the prefix cache hit rate for prefill token reuse approximation.<br>1. Type: Float.<br>2. Valid range: `[0, 1)`.<br>3. Default: `0.0`. |
| `--num-mtp-tokens` | LLM Options | Optional | Specifies the number of Multi-Token Prediction (MTP) tokens. `0` means disabled.<br>1. Type: Int.<br>2. Valid range: non-negative integer.<br>3. Default: `0`.<br>4. Only models with MTP capability are supported, such as DeepSeek. |
| `--no-repetition` | LLM Options | Optional | Disables transformer repetition-pattern optimization and preserves the original model behavior.<br>1. Type: Bool.<br>2. Valid range: flag option.<br>3. Default: `False`. |
| `--compile` | Optimization Options | Optional | Invokes `torch.compile()` on the model before inference.<br>1. Type: Bool.<br>2. Valid range: flag option.<br>3. Default: `False`. |
| `--compile-allow-graph-break` | Optimization Options | Optional | Allows graph breaks during `torch.compile()`.<br>1. Type: Bool.<br>2. Valid range: flag option.<br>3. Default: `False`. |
| `--compilation-config` | Optimization Options | Optional | Dynamically enable specified compilation features. Multiple options can be provided at once.<br>1. Type: List[Str] (`nargs="*"`).<br>2. Choices: `enable_multistream` (enable multi-stream scheduling), `enable_sequence_parallel` (enable sequence parallel graph rewrite pass), `enable_matmul_allreduce` (enable matmul-allreduce fusion), `enable_dispatch_ffn_combine` (enable dispatch_ffn_combine fusion).<br>3. Default: when omitted, all compilation features remain disabled (`False`).<br>4. Example: `--compilation-config enable_multistream enable_sequence_parallel`. |
| `--quantize-linear-action` | Quantization Options | Optional | Specifies the linear layer quantization mode.<br>1. Type: Str.<br>2. Reference values: `DISABLED`, `W8A16_STATIC`, `W8A8_STATIC`, `W4A8_STATIC`, `W8A16_DYNAMIC`, `W8A8_DYNAMIC`, `W4A8_DYNAMIC`, `FP8`, `MXFP4`.<br>3. Default: `W8A8_DYNAMIC`. |
| `--quantize-non-expert-linear-action` | Quantization Options | Optional | Specifies a separate quantization mode for non-expert linear layers, such as attention projections, dense MLP, and shared experts.<br>1. Type: Str.<br>2. Reference values: `DISABLED`, `W8A16_STATIC`, `W8A8_STATIC`, `W4A8_STATIC`, `W8A16_DYNAMIC`, `W8A8_DYNAMIC`, `W4A8_DYNAMIC`, `FP8`, `MXFP4`.<br>3. Default: `DISABLED`. |
| `--quantize-lmhead` | Quantization Options | Optional | Enables quantization for lm head.<br>1. Type: Bool.<br>2. Valid range: flag option.<br>3. Default: `False`. |
| `--mxfp4-group-size` | Quantization Options | Optional | Specifies the group size for mxfp4 quantization.<br>1. Type: Int.<br>2. Valid range: positive integer.<br>3. Default: `32`. |
| `--quantize-attention-action` | Quantization Options | Optional | Specifies KV cache quantization mode.<br>1. Type: Str.<br>2. Reference values: `DISABLED`, `INT8`, `FP8`.<br>3. Default: `DISABLED`. |
| `--graph-log-path` | Debugging Options | Optional | Directory for compiled graph dumps when debugging the compile path. Each pass writes files under this directory.<br>1. Type: Str.<br>2. Valid range: directory path.<br>3. Default: `None`. |
| `--dump-input-shapes` | Debugging Options | Optional | Dumps input shape information for troubleshooting model input configuration.<br>1. Type: Bool.<br>2. Valid range: flag option.<br>3. Default: `False`. |
| `--dump-op-bound-results` | Debugging Options | Optional | Dumps per-operator memory, communication, MMA, and GP bound ratios in the result table.<br>1. Type: Bool.<br>2. Valid range: flag option.<br>3. Default: `False`. |
| `--chrome-trace-file` | Debugging Options | Optional | Specifies the Chrome trace output path for exporting the performance timeline.<br>1. Type: Str.<br>2. Valid range: file path.<br>3. Default: `None`. |
| `--num-hidden-layers-override` | Debugging Options | Optional | Overrides the number of hidden layers for debugging only.<br>1. Type: Int.<br>2. Valid range: non-negative integer.<br>3. Default: `0`. |
| `--tp-size` | Parallelism Options | Optional | Specifies the tensor parallel size for the whole model.<br>1. Type: Int.<br>2. Valid range: positive integer.<br>3. Default: `1`. |
| `--dp-size` | Parallelism Options | Optional | Specifies the data parallel size for the whole model.<br>1. Type: Int.<br>2. Valid range: positive integer.<br>3. Default: `None`. |
| `--ep-size` | Parallelism Options | Optional | Specifies expert parallel size for experts.<br>1. Type: Int.<br>2. Valid range: positive integer.<br>3. Default: `1`. |
| `--o-proj-tp-size` | Parallelism Options | Optional | Specifies TP size for the attention `o_proj` layer.<br>1. Type: Int.<br>2. Valid range: positive integer.<br>3. Default: `None`. |
| `--o-proj-dp-size` | Parallelism Options | Optional | Specifies DP size for the attention `o_proj` layer.<br>1. Type: Int.<br>2. Valid range: positive integer.<br>3. Default: `None`. |
| `--mlp-tp-size` | Parallelism Options | Optional | Specifies TP size for MLP layers and can override `--tp-size`.<br>1. Type: Int.<br>2. Valid range: positive integer.<br>3. Default: `None`. |
| `--mlp-dp-size` | Parallelism Options | Optional | Specifies DP size for MLP layers and can override `--dp-size`.<br>1. Type: Int.<br>2. Valid range: positive integer.<br>3. Default: `None`. |
| `--lmhead-tp-size` | Parallelism Options | Optional | Specifies TP size for lm head and can override `--tp-size`.<br>1. Type: Int.<br>2. Valid range: positive integer.<br>3. Default: `None`. |
| `--lmhead-dp-size` | Parallelism Options | Optional | Specifies DP size for lm head and can override `--dp-size`.<br>1. Type: Int.<br>2. Valid range: positive integer.<br>3. Default: `None`. |
| `--moe-tp-size` | Parallelism Options | Optional | Specifies TP size for experts and can override `--tp-size`.<br>1. Type: Int.<br>2. Valid range: positive integer.<br>3. Default: `None`. |
| `--moe-dp-size` | Parallelism Options | Optional | Specifies DP size for experts and can override `--dp-size`.<br>1. Type: Int.<br>2. Valid range: positive integer.<br>3. Default: `1`. |
| `--word-embedding-tp` | Parallelism Options | Optional | Enables word embedding tensor parallel and specifies the parallel mode.<br>1. Type: Str.<br>2. Reference values: `col`, `row`.<br>3. Default: `None`, meaning embedding TP is disabled. |
| `--enable-redundant-experts` | Parallelism Options | Optional | Enables redundant expert configuration.<br>1. Type: Bool.<br>2. Valid range: flag option.<br>3. Default: `False`.<br>4. When enabled alone, each device hosts 1 additional redundant expert.<br>5. When enabled together with `--enable-external-shared-experts`, the allocation logic is the same as external shared experts. If routing experts are already evenly distributed across devices and no redundant experts are needed for padding, each device hosting routing experts hosts 1 additional redundant expert. |
| `--enable-shared-expert-tp` | Parallelism Options | Optional | Enables vLLM-style tensor parallel for shared experts.<br>1. Type: Bool.<br>2. Valid range: flag option.<br>3. Default: `False`.<br>4. Shared experts use dense MLP TP with delayed `down_proj` reduction. |
| `--enable-external-shared-experts` | Parallelism Options | Optional | Enables external shared experts.<br>1. Type: Bool.<br>2. Valid range: flag option.<br>3. Default: `False`.<br>4. When enabled, devices are allocated between external shared experts and routing experts at a `1:top_k` ratio. Redundant experts are used to pad routing experts if needed.<br>5. For example, if `world_size=64`, `top_k=8`, and the number of routing experts is 256, 8 devices host external shared experts and the remaining 56 devices distribute the 256 routing experts: 32 devices host 5 routing experts each, and 24 devices host 4 routing experts plus 1 redundant expert each. |
| `--host-external-shared-experts` | Parallelism Options | Optional | Specifies that the current device hosts external shared experts.<br>1. Type: Bool.<br>2. Valid range: flag option.<br>3. Default: `False`. |
| `--vision-tp-size` | Parallelism Options | Optional | Specifies tensor parallel size for vision modules.<br>1. Type: Int.<br>2. Valid range: positive integer.<br>3. Default: `1`, meaning vision modules are not sharded. |
| `--image-batch-size` | MultiModal Options | Optional | Specifies image processing batch size.<br>1. Type: Int.<br>2. Valid range: positive integer.<br>3. Default: `None`. |
| `--image-height` | MultiModal Options | Optional | Specifies input image height.<br>1. Type: Int.<br>2. Valid range: positive integer.<br>3. Default: `None`. |
| `--image-width` | MultiModal Options | Optional | Specifies input image width.<br>1. Type: Int.<br>2. Valid range: positive integer.<br>3. Default: `None`. |
| `--remote-source` | Options | Optional | Specifies the remote model source.<br>1. Type: Str.<br>2. Reference values: `huggingface`, `modelscope`.<br>3. Default: `huggingface`. |
| `--performance-model` | Options | Optional | Specifies one or more performance models. This parameter can be repeated.<br>1. Type: List[Str].<br>2. Reference values: `analytic`, `profiling`.<br>3. Default: `analytic` when omitted.<br>4. `analytic` is a roofline model and does not require profiling data; `profiling` is an empirical performance model backed by profiling CSV data and requires `--profiling-database-path`. |
| `--profiling-database-path` | Options | Optional | Specifies the profiling database path for the `profiling` performance model.<br>1. Type: Str.<br>2. Valid range: directory path containing `op_mapping.yaml` and CSV files for each kernel type.<br>3. Default: `None`. |
| `--export-empirical-metrics` | Options | Optional | Exports M1-M5 metrics as JSON for offline M6 computation.<br>1. Type: Str.<br>2. Valid range: JSON file path.<br>3. Default: `None`.<br>4. Developer-only option; requires `--performance-model profiling`. |

For VL models, use `--image-batch-size`, `--image-height`, and `--image-width` together to describe the number and resolution of input images. Omit them for text-only models.

Run `python -m cli.inference.text_generate --help` for details.

### 4.2 Run video generation inference for diffusion models

We provide a `video_generate.py` command line interface to simulate the forward pass and performance of diffusion transformer models. The script supports simulating the inference process of video generation models (e.g., Stable Video Diffusion-like architectures) with configurable input dimensions, sampling steps, and parallelism settings. A detailed table summary of operator performance breakdown is provided by default. An option is also provided to dump the performance timeline as a Chrome Trace file.

Its general usage is shown below:

```text
usage: video_generate.py [-h] [-V] [-v] [-q]
                         [--device {TEST_DEVICE,...}]
                         --batch-size BATCH_SIZE --seq-len SEQ_LEN [--chrome-trace-file CHROME_TRACE]
                         [--height HEIGHT] [--width WIDTH] [--frame-num FRAME_NUM] [--sample-step SAMPLE_STEP]
                         [--log-level {debug,info,warning,error,critical}] [--dtype {float16,float32,bfloat16}]
                         [--remote-source {huggingface,modelscope}]
                         [--quantize-linear-action {DISABLED,W8A16_STATIC,W8A8_STATIC,W4A8_STATIC,W8A16_DYNAMIC,W8A8_DYNAMIC,W4A8_DYNAMIC,FP8,MXFP4}]
                         [--quantize-attention-action {DISABLED,INT8,FP8}] [--use-cfg]
                         [--attention-backend {dense,block_sparse_attention}] [--attention-block-size ATTENTION_BLOCK_SIZE]
                         [--attention-sparsity ATTENTION_SPARSITY] [--num-devices WORLD_SIZE]
                         [--ulysses-size ULYSSES_SIZE] [--cfg-parallel] [--compile]
                         [--compile-allow-graph-break] [--dit-cache]
                         [--cache-step-range CACHE_STEP_RANGE] [--cache-step-interval CACHE_STEP_INTERVAL]
                         [--cache-block-range CACHE_BLOCK_RANGE]
                         model_id

Run a simulated diffusion transformer forward and dump perf stats.
```

Full `--help` also includes `--version/-V`, `--verbose/-v`, and `--quiet/-q`. Default `--log-level` is `error`. This command does not provide `--debug` or `--log-file`. The model source is the positional `model_id` or `--model-id`.

Main parameters:

| Parameter | Category | Required/Optional | Description |
| --- | --- | --- | --- |
| `model_id` | positional arguments | Required | Video generation model ID or local model path.<br>1. Type: Str.<br>2. Reference values: Diffusers model directory, remote repo ID, or remote repo ID plus subfolder. It must contain `transformer/config.json` or a compatible transformer config.<br>3. Default: none.<br>4. A reviewed local absolute path is recommended; remote model IDs are not security-guaranteed. |
| `--device` | options | Optional | Specifies the device profile for simulation.<br>1. Type: Str.<br>2. Reference values: registered `DeviceProfile` names, including `TEST_DEVICE`, `ATLAS_800_A2_376T_64G`, `ATLAS_800_A2_313T_64G`, `ATLAS_800_A2_280T_64G`, `ATLAS_800_A2_280T_64G_PCIE`, `ATLAS_800_A2_280T_32G_PCIE`, `ATLAS_800_A3_752T_128G_DIE`, `ATLAS_800_A3_560T_128G_DIE`, `ATLAS_800_A3_560T_128G_DIE_ROCE`, `ATLAS_350_425T_112G`, `ATLAS_350_425T_84G`.<br>3. Default: `TEST_DEVICE`. |
| `--batch-size` | options | Required | Specifies the input batch size.<br>1. Type: Int.<br>2. Valid range: positive integer.<br>3. Default: none. |
| `--seq-len` | options | Required | Specifies text sequence length.<br>1. Type: Int.<br>2. Valid range: positive integer.<br>3. Default: none. |
| `--chrome-trace-file` | options | Optional | Specifies the Chrome trace JSON output path for exporting the performance timeline.<br>1. Type: Str.<br>2. Valid range: file path.<br>3. Default: `None`. |
| `--height` | options | Optional | Specifies input video or image frame height.<br>1. Type: Int.<br>2. Valid range: positive integer.<br>3. Default: `400`. |
| `--width` | options | Optional | Specifies input video or image frame width.<br>1. Type: Int.<br>2. Valid range: positive integer.<br>3. Default: `832`. |
| `--frame-num` | options | Optional | Specifies the number of video frames.<br>1. Type: Int.<br>2. Valid range: positive integer.<br>3. Default: `81`. |
| `--sample-step` | options | Optional | Specifies the number of diffusion sampling steps.<br>1. Type: Int.<br>2. Valid range: positive integer.<br>3. Default: `1`. |
| `--log-level` | options | Optional | Specifies the log level.<br>1. Type: Str.<br>2. Reference values: `debug`, `info`, `warning`, `error`, `critical`.<br>3. Default: `error`. |
| `--dtype` | options | Optional | Specifies model compute data type.<br>1. Type: Str.<br>2. Reference values: `float16`, `float32`, `bfloat16`.<br>3. Default: `float16`. |
| `--remote-source` | options | Optional | Specifies the remote source for non-local Diffusers repo IDs.<br>1. Type: Str.<br>2. Reference values: `huggingface`, `modelscope`.<br>3. Default: `huggingface`. |
| `--quantize-linear-action` | options | Optional | Specifies linear layer quantization mode.<br>1. Type: Str.<br>2. Reference values: `DISABLED`, `W8A16_STATIC`, `W8A8_STATIC`, `W4A8_STATIC`, `W8A16_DYNAMIC`, `W8A8_DYNAMIC`, `W4A8_DYNAMIC`, `FP8`, `MXFP4`.<br>3. Default: `W8A8_DYNAMIC`. |
| `--quantize-attention-action` | options | Optional | Specifies attention computation quantization mode.<br>1. Type: Str.<br>2. Reference values: `DISABLED`, `INT8`, `FP8`.<br>3. Default: `DISABLED`. |
| `--use-cfg` | options | Optional | Enables classifier-free guidance simulation path.<br>1. Type: Bool.<br>2. Valid range: flag option.<br>3. Default: `False`. |
| `--attention-backend` | Attention Options | Optional | Selects the attention backend.<br>1. Type: Str.<br>2. Reference values: `dense`, `block_sparse_attention`.<br>3. Default: `dense`. |
| `--attention-block-size` | Attention Options | Optional | Sets the BSA block size.<br>1. Type: Int.<br>2. Valid range: positive integer.<br>3. Default: `128`. |
| `--attention-sparsity` | Attention Options | Optional | Sets the ratio of KV blocks skipped by BSA.<br>1. Type: Float.<br>2. Valid range: `[0.0, 1.0)`.<br>3. Default: `0.0`. |
| `--num-devices` | Parallel Options | Optional | Specifies the total number of devices participating in distributed simulation.<br>1. Type: Int.<br>2. Valid range: positive integer.<br>3. Default: `1`. |
| `--ulysses-size` | Parallel Options | Optional | Specifies Ulysses parallel size.<br>1. Type: Int.<br>2. Valid range: positive integer.<br>3. Default: `1`. |
| `--cfg-parallel` | Parallel Options | Optional | Enables CFG parallel strategy.<br>1. Type: Bool.<br>2. Valid range: flag option.<br>3. Default: `False`. |
| `--compile` | Optimization Options | Optional | Compiles the primary transformer before simulation.<br>1. Type: Bool.<br>2. Valid range: flag option.<br>3. Default: `False`.<br>4. Uses `dynamic=False, fullgraph=True`; when DiT cache is active, the cache transformer uses the same policy. |
| `--compile-allow-graph-break` | Optimization Options | Optional | Allows graph breaks during compilation.<br>1. Type: Bool.<br>2. Valid range: flag option.<br>3. Default: `False`.<br>4. Changes compilation to `fullgraph=False` for both the primary and DiT cache transformers. |
| `--dit-cache` | Cache Options | Optional | Enables DiT block cache.<br>1. Type: Bool.<br>2. Valid range: flag option.<br>3. Default: `False`. |
| `--cache-step-range` | Cache Options | Optional | Specifies sampling step range for cache.<br>1. Type: Str.<br>2. Format: `start,end`, inclusive interval.<br>3. Default: `None`.<br>4. Required when `--dit-cache` is set. |
| `--cache-step-interval` | Cache Options | Optional | Specifies cache update step interval.<br>1. Type: Int.<br>2. Valid range: positive integer.<br>3. Default: `1`, which disables cache update reuse. |
| `--cache-block-range` | Cache Options | Optional | Specifies block range for cache.<br>1. Type: Str.<br>2. Format: `start,end`, start inclusive and end exclusive.<br>3. Default: `None`. |

BSA behavior and limits:

- The BSA route accepts only Q, K, and V that are all 4D tensors with exactly matching shapes; the mask must be `None` or a 4D bool tensor with shape `(batch, 1, query_len, key_len)`. It requires dropout `0`, non-causal attention, no custom scale, and GQA disabled.
- Unsupported calls use dense fallback so models mixing self-attention and cross-attention remain compatible. Each SDPA context emits one aggregated warning summary. Stable fallback reason codes are `non_4d_qkv`, `qkv_shape_mismatch`, `unsupported_attention_mask`, `dropout`, `causal`, `custom_scale`, and `gqa`.
- BSA cannot be combined with attention quantization. The following complete command is invalid and fails before execution:

  ```bash
  python -m cli.inference.video_generate example-model --batch-size 1 --seq-len 128 --attention-backend block_sparse_attention --quantize-attention-action int8
  ```

  Linear quantization remains independent and is not disallowed by this rule.
- The dense backend accepts BSA-only defaults (`--attention-block-size 128` and `--attention-sparsity 0.0`) but explicitly rejects non-default BSA-only values; it never silently ignores them.
- The estimator rounds query and KV block counts up with `ceil`. Retained KV blocks are `max(1, ceil(KV_blocks * (1 - sparsity)))`. Boundary blocks are charged as complete `block_size x block_size` padded tiles, so a non-divisible sequence can cost more than token-exact dense attention even when sparsity is `0.0`. Route generation scans the complete block-pair space. Memory is an analytic approximation: Q, K, and V are each read once in full, and K/V bytes are not scaled by sparsity.

> **Note:** When simulating HunyuanVideo1.5 from the official raw Tencent repository, use `tencent/HunyuanVideo-1.5/transformer/<t2v_variant>` on Hugging Face or `Tencent-Hunyuan/HunyuanVideo-1.5/transformer/<t2v_variant>` on ModelScope. Supported T2V selectors are `480p_t2v`, `480p_t2v_distilled`, and `720p_t2v`. Raw repository roots, local raw layouts, I2V variants, and SR variants are not supported. TensorCast validates the downloaded JSON configuration and maps it to the built-in Diffusers simulation model; it does not execute the Tencent `hyvideo` package or convert checkpoint weights.

Run `python -m cli.inference.video_generate --help` for details.

### 4.3 Run image generation inference for diffusion models

We provide an `image_generate.py` command line interface to simulate the diffusion transformer denoising workload and performance of image generation models. The script supports simulating the transformer denoising inference process of image generation models (e.g., FLUX, Qwen-Image-Edit) with configurable batch size, output image size, text condition length, and parallelism settings. A detailed table summary of operator performance breakdown is provided by default. An option is also provided to dump the performance timeline as a Chrome Trace file.

The first version only simulates the Transformer denoising stage that enters the simulated device; prompt encoding, VAE, scheduler, and image I/O are excluded, and no real image is produced.

Its general usage is shown below:

```text
msmodeling inference image-generate MODEL --batch-size <N> --output-image-size HEIGHT WIDTH --text-seq-len <N>
```

Full `--help` also includes `--version/-V`, `--verbose/-v`, and `--quiet/-q`. Default `--log-level` is `error`. This command does not provide `--debug` or `--log-file`. The model source is the positional `model_id` or `--model-id`. `--num-devices` is the official parallel-size flag; `--world-size` and `--chrome-trace` remain hidden compatibility aliases.

Main parameters:

| Parameter | Category | Required/Optional | Description |
| --- | --- | --- | --- |
| `model_id` / `--model-id` | positional / options | Required (either) | Image generation model ID or local model path. Positional or `--model-id`.<br>1. Type: Str.<br>2. Reference values: Diffusers model directory or an exactly allowed remote repo ID, such as `black-forest-labs/FLUX.1-dev`, `Qwen/Qwen-Image-Edit`.<br>3. Default: none.<br>4. A reviewed local absolute path is recommended; remote model IDs are not security-guaranteed. |
| `--device` | options | Optional | Specifies the device profile for simulation.<br>1. Type: Str.<br>2. Reference values: registered `DeviceProfile` names, including `TEST_DEVICE`, `ATLAS_800_A2_376T_64G`, `ATLAS_800_A2_313T_64G`, `ATLAS_800_A2_280T_64G`, `ATLAS_800_A2_280T_64G_PCIE`, `ATLAS_800_A2_280T_32G_PCIE`, `ATLAS_800_A3_752T_128G_DIE`, `ATLAS_800_A3_560T_128G_DIE`, `ATLAS_800_A3_560T_128G_DIE_ROCE`, `ATLAS_350_425T_112G`, `ATLAS_350_425T_84G`.<br>3. Default: `TEST_DEVICE`. |
| `--batch-size` | options | Required | Specifies the base batch size of the workload, not the prompt count or source image count.<br>1. Type: Int.<br>2. Valid range: positive integer.<br>3. Default: none. |
| `--output-image-size` | options | Required | Specifies the output image size, provided exactly once; used only to derive the shape, does not output an image.<br>1. Type: Tuple[Int, Int] (`HEIGHT WIDTH`).<br>2. Valid range: two positive integers.<br>3. Default: none. |
| `--text-seq-len` | options | Required | The effective text condition length entering the Transformer.<br>1. Type: Int.<br>2. Valid range: positive integer.<br>3. Default: none.<br>4. Not a character count, tokenizer input length, or template length; text encoding is not performed in this first version. |
| `--source-image-size` | options | Optional | Specifies the source image size, repeatable per source; accepts sizes only, not paths or pixels. Available only for editing kinds.<br>1. Type: Tuple[Int, Int] (`HEIGHT WIDTH`).<br>2. Valid range: two positive integers.<br>3. Default: none. |
| `--sample-step` | options | Optional | Specifies the number of identical Transformer workload iterations to run.<br>1. Type: Int.<br>2. Valid range: positive integer.<br>3. Default: `1`. |
| `--use-cfg` | options | Optional | Enables the video-style classifier-free guidance workload approximation.<br>1. Type: Bool.<br>2. Valid range: flag option.<br>3. Default: `False`. |
| `--dtype` | options | Optional | Specifies model compute data type.<br>1. Type: Str.<br>2. Reference values: `float16`, `float32`, `bfloat16`.<br>3. Default: `float16`. |
| `--remote-source` | options | Optional | Specifies the remote model source; participates in exact pair matching.<br>1. Type: Str.<br>2. Reference values: `huggingface`, `modelscope`.<br>3. Default: `huggingface`. |
| `--quantize-linear-action` | Quantization Options | Optional | Specifies linear layer quantization mode.<br>1. Type: Str.<br>2. Reference values: `DISABLED`, `W8A16_STATIC`, `W8A8_STATIC`, `W4A8_STATIC`, `W8A16_DYNAMIC`, `W8A8_DYNAMIC`, `W4A8_DYNAMIC`, `FP8`, `MXFP4`.<br>3. Default: `DISABLED`. |
| `--mxfp4-group-size` | Quantization Options | Optional | Specifies the group size for MXFP4 quantization.<br>1. Type: Int.<br>2. Valid range: positive integer.<br>3. Default: `32`. |
| `--quantize-attention-action` | Quantization Options | Optional | Specifies attention computation quantization mode.<br>1. Type: Str.<br>2. Reference values: `DISABLED`, `INT8`, `FP8`.<br>3. Default: `DISABLED`. |
| `--compile` | Optimization Options | Optional | Compiles the primary transformer before simulation.<br>1. Type: Bool.<br>2. Valid range: flag option.<br>3. Default: `False`.<br>4. Uses `dynamic=False, fullgraph=True`; when DiT cache is active, the cache transformer uses the same policy. |
| `--compile-allow-graph-break` | Optimization Options | Optional | Allows graph breaks during compilation.<br>1. Type: Bool.<br>2. Valid range: flag option.<br>3. Default: `False`.<br>4. Changes compilation to `fullgraph=False` for both the primary and DiT cache transformers. |
| `--num-devices` | Parallel Options | Optional | Specifies the total number of devices participating in distributed simulation.<br>1. Type: Int.<br>2. Valid range: positive integer.<br>3. Default: `1`.<br>4. Must equal `--ulysses-size` (`2 * --ulysses-size` when `--cfg-parallel` is enabled). Legacy `--world-size` is still accepted. |
| `--ulysses-size` | Parallel Options | Optional | Specifies Ulysses parallel size.<br>1. Type: Int.<br>2. Valid range: positive integer.<br>3. Default: `1`. |
| `--cfg-parallel` | Parallel Options | Optional | Enables CFG parallel strategy.<br>1. Type: Bool.<br>2. Valid range: flag option.<br>3. Default: `False`.<br>4. Only enabled with `--use-cfg`; then `--num-devices` must equal `2 * --ulysses-size`. |
| `--dit-cache` | Cache Options | Optional | Enables DiT block cache.<br>1. Type: Bool.<br>2. Valid range: flag option.<br>3. Default: `False`. |
| `--cache-step-range` | Cache Options | Optional | Specifies sampling step range for cache.<br>1. Type: Str.<br>2. Format: `start,end`, inclusive interval.<br>3. Default: `None`.<br>4. Required when `--dit-cache` is set and `--cache-step-interval > 1`. |
| `--cache-step-interval` | Cache Options | Optional | Specifies cache update step interval.<br>1. Type: Int.<br>2. Valid range: positive integer.<br>3. Default: `1`, which disables cache update reuse. |
| `--cache-block-range` | Cache Options | Optional | Specifies block range for cache.<br>1. Type: Str.<br>2. Format: `start,end`, start inclusive and end exclusive.<br>3. Default: `None`. |
| `--chrome-trace-file` | options | Optional | Specifies the Chrome trace JSON output path for exporting the performance timeline.<br>1. Type: Str.<br>2. Valid range: file path.<br>3. Default: `None`.<br>4. Generated only after a successful Runtime run. Legacy `--chrome-trace` is still accepted. |

> **Note:** The Core currently registers no production image model kind; support for models such as `black-forest-labs/FLUX.1-dev` and `Qwen/Qwen-Image-Edit` is provided by the corresponding model-extension PRs. Until then, passing a real model ID fails explicitly (fail-closed).

Run `python -m cli.inference.image_generate --help` for details.
