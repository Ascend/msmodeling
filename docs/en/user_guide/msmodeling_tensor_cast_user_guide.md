# Model Inference Performance Simulation User Guide

For the complete model list and feature details, see [Model and Feature Support Matrix](./support_matrix/support_matrix_user_guide.md).

## Reading Paths

| Goal | Recommended Section |
| --- | --- |
| 1. Quickly run LLM text generation simulation | [2.1 Quick Start: Text Generation](#21-quick-start-text-generation) |
| 2. Understand the latency, invocation count, and memory metrics in the output | [2.2 Result (Text Generation)](#22-result-text-generation) |
| 3. Run video generation model simulation | [2.3 Quick Start: Video Generation](#23-quick-start-video-generation) |
| 4. Run image generation model simulation | [2.5 Quick Start: Image Generation](#25-quick-start-image-generation) |
| 5. View or customize hardware device profiles | [3 Supported Devices and Custom Devices](#3-supported-devices-and-custom-devices) |

## 1 Introduction

TensorCast is a performance simulation and analysis framework for PyTorch programs. It allows developers and researchers to predict the performance of their neural network models on specific hardware configurations without accessing a physical machine.

At its core, TensorCast acts as a "virtual machine" or runtime simulator. Instead of executing computations on a real accelerator, it intercepts the computational graph of a PyTorch program and simulates its execution on a user-defined MachineConfig. This configuration specifies the characteristics of the target hardware, such as theoretical compute power (TFLOPS), memory bandwidth, cache hierarchy, and interconnect speed. To accurately estimate the optimal performance of a model on given hardware, TensorCast provides a model optimization pipeline that includes automatic model sharding, quantization, and FX graph optimization, converting the source program into an optimal form before the analysis.

By running models on "virtual" hardware, TensorCast provides detailed performance insights, including:

- Out-of-the-box support for Hugging Face transformer models.
- Support for multiple hardware accelerator devices through simple configuration.
- Operator-level execution time: estimated using extensible models, such as analytic roofline models, empirical data, or machine learning-based predictors.
- Memory footprint: tracks total and peak memory allocation.
- Computational characteristics: analyzes the FLOPs (floating point operations) and memory access volume of each operator.
- Advanced scheduling simulation: models complex execution patterns, such as concurrent computation across multiple streams.

The final output includes comprehensive summary tables and detailed Chrome Trace files, enabling in-depth visualization and identification of performance bottlenecks.

Before first use, see [Quick Start: Environment Setup and Your First Simulation](../install_guide/msmodeling_install_guide.md) to set up the environment and run an LLM inference simulation.

## 2 At a Glance

### 2.1 Quick Start: Text Generation

**What it does:** Simulates the LLM inference performance of a batch of queries.

#### Prefill Scenario

To run prefill for Qwen3-32B on `TEST_DEVICE` with two requests, each with a 3,500-token input length, run the following command:

```bash
python -m cli.inference.text_generate Qwen/Qwen3-32B --num-queries 2 --query-length 3500 --context-length 3500 --device TEST_DEVICE --compile
```

In prefill mode, do not add `--decode`. `--query-length` specifies the new input length, and `--context-length` specifies the context length of each request.

You can also quantize the linear layers with various quantization schemes, such as `W8A8` dynamic quantization, using a 4500-token context as the prefix:

```bash
python -m cli.inference.text_generate Qwen/Qwen3-32B --num-queries 2 --query-length 3500 --context-length 4500 --device TEST_DEVICE --quantize-linear-action W8A8_DYNAMIC --compile
```

#### Decode Scenario

Running the decode scenario is similar. You only need to adjust the input length `--query-length` and the context length `--context-length` of the requests. When MTP is disabled, `--query-length` is usually 1. When `--num-mtp-tokens` is enabled, set `--query-length` to `1 + --num-mtp-tokens`.

```bash
python -m cli.inference.text_generate Qwen/Qwen3-32B --num-queries 10 --query-length 1 --context-length 4500 --decode --device TEST_DEVICE --quantize-linear-action W8A8_STATIC --compile
```

**Output:** A performance summary table. If `--chrome-trace` is set, a Chrome trace file is optionally generated.

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

Note: `Model compilation and execution time` is the runtime of the simulator on the host machine, not the actual model compilation or execution time on the hardware.

Metric descriptions:

- `analytic total`: Estimated total time spent by operators.
- `analytic avg`: Average time per operator invocation.
- `# of Calls`: Number of times an operator is invoked.
- `Total time for analytic`: Sum of analytic operator time.
- `TPS/Device`: Tokens per second per device.
- `Total device memory` and its breakdown: Estimated memory usage of weights, KV cache, and activations.

### 2.3 Quick Start: Video Generation

**What it does:** Simulates the diffusion transformer forward pass of video generation models. The following example uses the Wan2.2 Diffusers remote model ID. On the first run, the required model configuration files are pulled according to the configuration.

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

**Key parameters:** `model_id`, `--device`, `--batch-size`, `--seq-len`, `--height`, `--width`, `--frame-num`, `--sample-step`, `--dtype`, `--quantize-linear-action`, `--chrome-trace`

**Output:** A performance summary table. If `--chrome-trace` is set, a Chrome trace file is optionally generated.

### 2.4 Result (Video Generation)

Example output (truncated. The actual values vary with the device configuration, model configuration, and input dimensions):

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

Note: `Model compilation and execution time` is the runtime of the simulator on the host machine, not the actual model compilation or execution time on the hardware.

Metric descriptions:

- `analytic total`: Estimated total time spent by operators.
- `analytic avg`: Average time per operator invocation.
- `# of Calls`: Number of times an operator is invoked.
- `Total time for analytic`: Sum of analytic operator time.

### 2.5 Quick Start: Image Generation

**What it does:** Simulates the diffusion transformer denoising workload of image generation models. The first release simulates only the Transformer denoising-stage workload; prompt encoding, VAE, scheduler, and actual image generation are not executed. The following example uses the FLUX.1-dev remote model ID; on the first run, the required model configuration files are pulled according to the configuration.

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

**Key parameters:** `model_id` / `--model-id`, `--device`, `--batch-size`, `--output-image-size`, `--text-seq-len`, `--source-image-size`, `--sample-step`, `--use-cfg`, `--num-devices`, `--ulysses-size`, `--cfg-parallel`, `--dit-cache`, `--chrome-trace-file`

**Output:** `Runtime execution time`, followed by a performance summary table. If `--chrome-trace-file` is set, a Chrome trace is exported and its path is printed afterward.

### 2.6 Result (Image Generation)

`image_generate` reports the critical path and logical measured workload of the Transformer denoising stage. It prints `Runtime execution time`, followed by a performance summary table; if `--chrome-trace-file` is set, it then exports a Chrome trace and prints its path.

- `Runtime execution time`: Runtime of the simulator on the host machine, not the actual model compilation or execution time on the hardware.
- `analytic total`: Estimated total time spent by operators.
- `analytic avg`: Average time per operator invocation.
- `# of Calls`: Number of times an operator is invoked.
- `Total time for analytic`: Sum of analytic operator time.

Specific values vary with the device configuration, model configuration, and input dimensions.

## 3 Supported Devices and Custom Devices

We provide built-in support for the following device configurations (defined in `tensor_cast/device.py`):

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

For other hardware, define a custom device configuration as a Python file in the `tensor_cast/device_profiles` directory. TensorCast loads it automatically, and you can then reference the configuration name in the CLI. Custom device guide: [device_profiles/README.md](../../../tensor_cast/device_profiles/README.md)

## 4 Detailed Usage

You are advised to use the local safe mode: download and review the model repository first, and then set `model_id` to a full local absolute path, for example
`/data/models/Qwen3-32B`. Local path loading validates the path owner, symbolic links, and permissions before running. You are advised not to use symbolic link directories,
shared writable directories, or model files from unreviewed sources.

The tool still supports passing a Hugging Face or ModelScope model ID directly, such as `Qwen/Qwen3-32B`, and you can select the source with
`--remote-source`. In this model ID mode, remote Python code may be executed during the `trust_remote_code=True` fallback.
msModeling does not guarantee the security of remote code. At runtime, it prints a `trust_remote_code` risk notice.

### 4.1 Text Generation

We provide the `text_generate.py` CLI to simulate text generation with text-only or VL image-and-text inputs. This script supports text generation simulation for a batch of queries with the same input length and optionally the same context length. A table summary of the operator performance breakdown is provided by default. You can also choose to export a Chrome trace.

Its general usage is as follows:

```text
usage: text_generate.py [-h]
                        [--device {TEST_DEVICE,ATLAS_800_A2_376T_64G,ATLAS_800_A2_313T_64G,ATLAS_800_A2_280T_64G,ATLAS_800_A2_280T_64G_PCIE,ATLAS_800_A2_280T_32G_PCIE,ATLAS_800_A3_752T_128G_DIE,ATLAS_800_A3_560T_128G_DIE,ATLAS_800_A3_560T_128G_DIE_ROCE,ATLAS_350_425T_112G,ATLAS_350_425T_84G}]
                        [--num-devices NUM_DEVICES] [--enable-multistream] [--reserved-memory-gb RESERVED_MEMORY_GB]
                        [--log-level {debug,info,warning,error,critical}] --num-queries NUM_QUERIES
                        --query-length QUERY_LENGTH [--context-length CONTEXT_LENGTH] [--decode]
                        [--prefix-cache-hit-rate PREFIX_CACHE_HIT_RATE] [--num-mtp-tokens NUM_MTP_TOKENS]
                        [--disable-repetition] [--compile] [--compile-allow-graph-break]
                        [--enable-sequence-parallel]
                        [--quantize-linear-action {DISABLED,W8A16_STATIC,W8A8_STATIC,W4A8_STATIC,W8A16_DYNAMIC,W8A8_DYNAMIC,W4A8_DYNAMIC,FP8,MXFP4}]
                        [--quantize-non-expert-linear-action {DISABLED,W8A16_STATIC,W8A8_STATIC,W4A8_STATIC,W8A16_DYNAMIC,W8A8_DYNAMIC,W4A8_DYNAMIC,FP8,MXFP4}]
                        [--quantize-lmhead] [--mxfp4-group-size MXFP4_GROUP_SIZE]
                        [--quantize-attention-action {DISABLED,INT8,FP8}] [--graph-log-url GRAPH_LOG_URL]
                        [--dump-input-shapes] [--dump-op-bound-results] [--chrome-trace CHROME_TRACE]
                        [--num-hidden-layers-override NUM_HIDDEN_LAYERS_OVERRIDE] [--tp-size TP_SIZE]
                        [--dp-size DP_SIZE] [--ep-size EP_SIZE] [--o-proj-tp-size O_PROJ_TP_SIZE]
                        [--o-proj-dp-size O_PROJ_DP_SIZE] [--mlp-tp-size MLP_TP_SIZE] [--mlp-dp-size MLP_DP_SIZE]
                        [--lmhead-tp-size LMHEAD_TP_SIZE] [--lmhead-dp-size LMHEAD_DP_SIZE]
                        [--moe-tp-size MOE_TP_SIZE] [--moe-dp-size MOE_DP_SIZE] [--word-embedding-tp {col,row}]
                        [--enable-redundant-experts] [--enable-shared-expert-tp] [--enable-dispatch-ffn-combine]
                        [--enable-external-shared-experts] [--host-external-shared-experts]
                        [--vision-tp-size VISION_TP_SIZE] [--image-batch-size IMAGE_BATCH_SIZE]
                        [--image-height IMAGE_HEIGHT] [--image-width IMAGE_WIDTH]
                        [--remote-source {huggingface,modelscope}] [--performance-model {analytic,profiling}]
                        [--profiling-database PROFILING_DATABASE]
                        [--export-empirical-metrics EXPORT_EMPIRICAL_METRICS]
                        model_id

Run a simulated LLM inference pass and dump the perf result.
```

The main parameters are as follows:

| Parameter                            | Category | Optional/Required | Description |
| -----------------------------------  | --- | --- | --- |
| model_id                             | General Options | Required | Model ID or local model path.<br>1. Type: Str.<br>2. Reference values: Hugging Face ID, ModelScope ID, or a local absolute path, for example `Qwen/Qwen3-32B` or `/data/models/Qwen3-32B`.<br>3. Default: None.<br>4. When using a remote model ID, remote code may be executed through `trust_remote_code=True`. |
| --device                             | General Options | Optional | Specifies the device configuration used for the simulation.<br>1. Type: Str.<br>2. Reference values: registered `DeviceProfile` names, including `TEST_DEVICE`, `ATLAS_800_A2_376T_64G`, `ATLAS_800_A2_313T_64G`, `ATLAS_800_A2_280T_64G`, `ATLAS_800_A2_280T_64G_PCIE`, `ATLAS_800_A2_280T_32G_PCIE`, `ATLAS_800_A3_752T_128G_DIE`, `ATLAS_800_A3_560T_128G_DIE`, `ATLAS_800_A3_560T_128G_DIE_ROCE`, `ATLAS_350_425T_112G`, `ATLAS_350_425T_84G`.<br>3. Default: `TEST_DEVICE`. |
| --num-devices                        | General Options | Optional | Specifies the number of devices participating in the simulation.<br>1. Type: Int.<br>2. Value range: positive integer.<br>3. Default: 1. |
| --enable-multistream                 | General Options | Optional | Enables compile-time multi-stream simulation in the `--compile` path.<br>1. Type: Bool.<br>2. Value range: on/off flag.<br>3. Default: `True`. |
| --reserved-memory-gb                 | General Options | Optional | Specifies the amount of device memory reserved for the system on each device, in GB.<br>1. Type: Float.<br>2. Value range: non-negative number. Set to 0 to reserve no system memory.<br>3. Default: 0.0. |
| --log-level                          | General Options | Optional | Specifies the log level.<br>1. Type: Str.<br>2. Reference values: `debug`, `info`, `warning`, `error`, `critical`.<br>3. Default: `error`. |
| --num-queries                        | LLM Options | Required | Number of queries in this simulation.<br>1. Type: Int.<br>2. Value range: positive integer.<br>3. Default: None. |
| --query-length                       | LLM Options | Required | New input token length of each query.<br>1. Type: Int.<br>2. Value range: positive integer.<br>3. Default: None. |
| --context-length                     | LLM Options | Optional | Existing context token length of each query.<br>1. Type: Int.<br>2. Value range: non-negative integer.<br>3. Default: 0. |
| --decode                             | LLM Options | Optional | Enables autoregressive decode mode. When not set, the simulation runs in prefill mode.<br>1. Type: Bool.<br>2. Value range: on/off flag.<br>3. Default: `False`. |
| --prefix-cache-hit-rate              | LLM Options | Optional | Specifies the prefix cache hit rate, used to approximate prefill token reuse.<br>1. Type: Float.<br>2. Value range: [0, 1).<br>3. Default: 0.0. |
| --num-mtp-tokens                     | LLM Options | Optional | Specifies the number of Multi-Token Prediction (MTP) tokens. 0 disables MTP.<br>1. Type: Int.<br>2. Value range: non-negative integer.<br>3. Default: 0.<br>4. Supported only by models with MTP capability, for example DeepSeek. |
| --disable-repetition                 | LLM Options | Optional | Disables the transformer repetition pattern optimization and preserves the original model behavior.<br>1. Type: Bool.<br>2. Value range: on/off flag.<br>3. Default: `False`. |
| --compile                            | Optimization Options | Optional | Calls `torch.compile()` on the model before inference.<br>1. Type: Bool.<br>2. Value range: on/off flag.<br>3. Default: `False`. |
| --compile-allow-graph-break          | Optimization Options | Optional | Allows graph breaks during `torch.compile()`.<br>1. Type: Bool.<br>2. Value range: on/off flag.<br>3. Default: `False`. |
| --enable-sequence-parallel           | Optimization Options | Optional | Enables the sequence parallel graph rewrite pass during compilation.<br>1. Type: Bool.<br>2. Value range: on/off flag.<br>3. Default: `False`. |
| --quantize-linear-action             | Quantization Options | Optional | Specifies the quantization scheme for linear layers.<br>1. Type: Str.<br>2. Reference values: `DISABLED`, `W8A16_STATIC`, `W8A8_STATIC`, `W4A8_STATIC`, `W8A16_DYNAMIC`, `W8A8_DYNAMIC`, `W4A8_DYNAMIC`, `FP8`, `MXFP4`.<br>3. Default: `W8A8_DYNAMIC`. |
| --quantize-non-expert-linear-action  | Quantization Options | Optional | Specifies an independent quantization scheme for non-expert linear layers, such as attention projections, dense MLPs, and shared experts.<br>1. Type: Str.<br>2. Reference values: `DISABLED`, `W8A16_STATIC`, `W8A8_STATIC`, `W4A8_STATIC`, `W8A16_DYNAMIC`, `W8A8_DYNAMIC`, `W4A8_DYNAMIC`, `FP8`, `MXFP4`.<br>3. Default: `DISABLED`. |
| --quantize-lmhead                    | Quantization Options | Optional | Enables quantization for the lm head.<br>1. Type: Bool.<br>2. Value range: on/off flag.<br>3. Default: `False`. |
| --mxfp4-group-size                   | Quantization Options | Optional | Specifies the group size for `MXFP4` quantization.<br>1. Type: Int.<br>2. Value range: positive integer.<br>3. Default: 32. |
| --quantize-attention-action          | Quantization Options | Optional | Specifies the quantization scheme for the KV cache.<br>1. Type: Str.<br>2. Reference values: `DISABLED`, `INT8`, `FP8`.<br>3. Default: `DISABLED`. |
| --graph-log-url                      | Debugging Options | Optional | Specifies the output path for the compilation graph log. Use it only when debugging the compile path.<br>1. Type: Str.<br>2. Value range: file or directory path.<br>3. Default: `None`. |
| --dump-input-shapes                  | Debugging Options | Optional | Outputs the input shape information to help troubleshoot the model input configuration.<br>1. Type: Bool.<br>2. Value range: on/off flag.<br>3. Default: `False`. |
| --dump-op-bound-results              | Debugging Options | Optional | Outputs the operator-level memory, communication, MMA, and GP bound ratios in the result table.<br>1. Type: Bool.<br>2. Value range: on/off flag.<br>3. Default: `False`. |
| --chrome-trace                       | Debugging Options | Optional | Specifies the Chrome trace output path for exporting the performance timeline.<br>1. Type: Str.<br>2. Value range: file path.<br>3. Default: `None`. |
| --num-hidden-layers-override         | Debugging Options | Optional | Overrides the number of hidden layers in the model. For debugging only.<br>1. Type: Int.<br>2. Value range: non-negative integer.<br>3. Default: 0. |
| --tp-size                            | Parallelism Options | Optional | Specifies the tensor parallel size for the entire model.<br>1. Type: Int.<br>2. Value range: positive integer.<br>3. Default: 1. |
| --dp-size                            | Parallelism Options | Optional | Specifies the data parallel size for the entire model.<br>1. Type: Int.<br>2. Value range: positive integer.<br>3. Default: `None`. |
| --ep-size                            | Parallelism Options | Optional | Specifies the expert parallel size for experts.<br>1. Type: Int.<br>2. Value range: positive integer.<br>3. Default: 1. |
| --o-proj-tp-size                     | Parallelism Options | Optional | Specifies the TP size for the attention `o_proj` layer.<br>1. Type: Int.<br>2. Value range: positive integer.<br>3. Default: `None`. |
| --o-proj-dp-size                     | Parallelism Options | Optional | Specifies the DP size for the attention `o_proj` layer.<br>1. Type: Int.<br>2. Value range: positive integer.<br>3. Default: `None`. |
| --mlp-tp-size                        | Parallelism Options | Optional | Specifies the TP size for MLP layers, overriding `--tp-size`.<br>1. Type: Int.<br>2. Value range: positive integer.<br>3. Default: `None`. |
| --mlp-dp-size                        | Parallelism Options | Optional | Specifies the DP size for MLP layers, overriding `--dp-size`.<br>1. Type: Int.<br>2. Value range: positive integer.<br>3. Default: `None`. |
| --lmhead-tp-size                     | Parallelism Options | Optional | Specifies the TP size for the lm head, overriding `--tp-size`.<br>1. Type: Int.<br>2. Value range: positive integer.<br>3. Default: `None`. |
| --lmhead-dp-size                     | Parallelism Options | Optional | Specifies the DP size for the lm head, overriding `--dp-size`.<br>1. Type: Int.<br>2. Value range: positive integer.<br>3. Default: `None`. |
| --moe-tp-size                        | Parallelism Options | Optional | Specifies the TP size for experts, overriding `--tp-size`.<br>1. Type: Int.<br>2. Value range: positive integer.<br>3. Default: `None`. |
| --moe-dp-size                        | Parallelism Options | Optional | Specifies the DP size for experts, overriding `--dp-size`.<br>1. Type: Int.<br>2. Value range: positive integer.<br>3. Default: 1. |
| --word-embedding-tp                  | Parallelism Options | Optional | Enables tensor parallelism for word embeddings and specifies the parallel mode.<br>1. Type: Str.<br>2. Reference values: `col`, `row`.<br>3. Default: `None`, which means embedding TP is disabled. |
| --enable-redundant-experts           | Parallelism Options | Optional | Enables the redundant expert configuration.<br>1. Type: Bool.<br>2. Value range: on/off flag.<br>3. Default: `False`.<br>4. When enabled alone, each device hosts one additional redundant expert.<br>5. When enabled together with `--enable-external-shared-experts`, the allocation logic is the same as for external shared experts. If the routing experts are already evenly distributed across devices and no redundant experts are needed to pad them, each device hosting routing experts hosts one additional redundant expert. |
| --enable-shared-expert-tp            | Parallelism Options | Optional | Enables vLLM-style tensor parallelism for shared experts.<br>1. Type: Bool.<br>2. Value range: on/off flag.<br>3. Default: `False`.<br>4. Shared experts use dense MLP TP and defer the `down_proj` reduction. |
| --enable-dispatch-ffn-combine        | Parallelism Options | Optional | Enables the dispatch_ffn_combine fusion mode during compilation.<br>1. Type: Bool.<br>2. Value range: on/off flag.<br>3. Default: `False`. |
| --enable-external-shared-experts     | Parallelism Options | Optional | Enables external shared experts.<br>1. Type: Bool.<br>2. Value range: on/off flag.<br>3. Default: `False`.<br>4. When enabled, devices are allocated between external shared experts and routing experts at a ratio of `1:top_k`. Redundant experts are used to pad the routing experts if needed.<br>5. For example, with `world_size=64`, `top_k=8`, and 256 routing experts, 8 devices host the external shared experts and the remaining 56 devices distribute the 256 routing experts: 32 devices host 5 routing experts each, and 24 devices host 4 routing experts and 1 redundant expert each. |
| --host-external-shared-experts       | Parallelism Options | Optional | Specifies that the current device hosts the external shared experts.<br>1. Type: Bool.<br>2. Value range: on/off flag.<br>3. Default: `False`. |
| --vision-tp-size                     | Parallelism Options | Optional | Specifies the tensor parallel size for the vision module.<br>1. Type: Int.<br>2. Value range: positive integer.<br>3. Default: 1, which means the vision module is not sharded. |
| --image-batch-size                   | MultiModal Options | Optional | Specifies the batch size for image processing.<br>1. Type: Int.<br>2. Value range: positive integer.<br>3. Default: `None`. |
| --image-height                       | MultiModal Options | Optional | Specifies the height of the input images.<br>1. Type: Int.<br>2. Value range: positive integer.<br>3. Default: `None`. |
| --image-width                        | MultiModal Options | Optional | Specifies the width of the input images.<br>1. Type: Int.<br>2. Value range: positive integer.<br>3. Default: `None`. |
| --remote-source                      | Options | Optional | Specifies the remote model source.<br>1. Type: Str.<br>2. Reference values: `huggingface`, `modelscope`.<br>3. Default: `huggingface`. |
| --performance-model                  | Options | Optional | Specifies the performance model. You can specify one or more models repeatedly.<br>1. Type: List[Str].<br>2. Reference values: `analytic`, `profiling`.<br>3. Default: `analytic` when not specified.<br>4. `analytic` is the Roofline model and does not require profiling data. `profiling` is an empirical performance model based on the profiling CSV database and requires `--profiling-database`. |
| --profiling-database                 | Options | Optional | Specifies the profiling database path when using the `profiling` performance model.<br>1. Type: Str.<br>2. Value range: path to a directory that contains `op_mapping.yaml` and CSV files for each kernel type.<br>3. Default: `None`. |
| --export-empirical-metrics           | Options | Optional | Exports the M1-M5 metrics JSON for offline M6 calculation.<br>1. Type: Str.<br>2. Value range: JSON file path.<br>3. Default: `None`.<br>4. For development and debugging only. Requires `--performance-model profiling`. |

`--enable-multistream` enables compile-time multi-stream simulation in the `--compile` path. This capability is enabled by default. Therefore, existing compile commands keep their current behavior.

For VL models, you can set `--image-batch-size`, `--image-height`, and `--image-width` together to describe the number and resolution of the input images. For text-only models, you can omit these parameters.

Run `python -m cli.inference.text_generate --help` for details.

#### Practical cases and result analysis

The following cases use Hugging Face model IDs to verify the execution mode, representative operators, input shapes, performance, memory, and bottlenecks from `text_generate` output. `text_generate` constructs simulated inputs; it does not read text or image files or return natural-language generations. The examples disable linear-layer quantization; output values illustrate the analysis method and are not fixed acceptance thresholds.

`Stats breakdowns` is the normalized share of analytically estimated operator time grouped by the dominant resource, not measured hardware utilization. `memory_bound`, `communication_bound`, `compute_bound_mma`, and `compute_bound_gp` represent memory-access-, communication-, matrix-compute-, and general-purpose-compute-bound components, respectively.

`Model compilation and execution time` is the simulator runtime on the host; `Total time for analytic` is the sum of `analytic total` in the operator table; `[analytic] Execution time` is the target-device latency predicted by the analytic model, and `TPS/Device` is calculated from this latency.

##### Case 1: DeepSeek-V4-Flash text-only incremental prefill

**Case goal:**

Simulate eight text-only requests on 16 A3 dies with `TP=2`, `DP=8`, and `EP=16`, then analyze the execution mode, operator hotspots, memory usage, and performance bottlenecks for incremental or chunked Prefill.

**Full command:**

```bash
python -m cli.inference.text_generate deepseek-ai/DeepSeek-V4-Flash \
  --device ATLAS_800_A3_752T_128G_DIE \
  --num-devices 16 \
  --tp-size 2 \
  --dp-size 8 \
  --ep-size 16 \
  --num-queries 8 \
  --query-length 128 \
  --context-length 2048 \
  --quantize-linear-action DISABLED \
  --compile \
  --compilation-config enable_dispatch_ffn_combine
```

**Key output** (truncated)

```text
Number of Queries per DP rank: 1
Model compilation and execution time: 15.923 s
----------------------------------------------  --------------  ------------  ----------
                     Name                       analytic total  analytic avg  # of Calls
----------------------------------------------  --------------  ------------  ----------
tensor_cast.dispatch_ffn_combine.default              35.837ms     833.409us          43
aten.mm.default                                         9.640ms      22.418us         430
tensor_cast.sparse_attn_sharedkv.default               2.626ms      61.072us          43
tensor_cast.compressor.default                         1.836ms      29.609us          62
tensor_cast.quant_lightning_indexer.default          137.760us       6.560us          21
tensor_cast.v4_clamped_swiglu.default                118.037us       2.745us          43
...
Total time for analytic: 67.209ms
[analytic] Execution time: 0.067209 s
[analytic] TPS/Device: 952.3 token/s
Total device memory: 64.000 GB
  Model weight size: 42.669 GB
  KV cache: 0.024 GB
  Model activation size: 0.103 GB
  Memory available: 21.204 GB
Stats breakdowns:
  analytic_OpBound: memory_bound: 99.19, communication_bound: 0.81, compute_bound_mma: 0.00, compute_bound_gp: 0.00
```

**Result analysis:**

1. Omitting `--decode` leaves `decode=False`, so the CLI uses the Prefill path. `--context-length 2048` specifies existing context and does not change the execution mode. Processing another `128` tokens is incremental or chunked prefill with a sequence length of `2048 + 128 = 2176`. `DP=8` distributes eight requests across eight DP ranks, matching `Number of Queries per DP rank: 1`; `TP × DP × PP = 2 × 8 × 1 = 16`, and `EP=16` shards the routed experts.
2. `--compilation-config enable_dispatch_ffn_combine` explicitly enables the fused MoE FFN path. Identify hotspots by comparing `analytic total`, rather than the per-call `analytic avg`. The fused `dispatch_ffn_combine` operator accounts for approximately `53.3%` of the total analytic time and should be examined first; matrix multiplication and sparse attention are the secondary hotspots.
3. `memory_bound: 99.19` is the dominant modeled bottleneck category, indicating that the analytic model estimates this case to be primarily memory-access bound. `Memory available: 21.204 GB` is positive, so the parallel configuration fits under the current device profile and simulation assumptions.

##### Case 2: Qwen3-VL image-and-text Prefill and Decode

**Case goal:**

Simulate eight initial image-and-text Prefill requests on one A3 die. Each request contains `32` text tokens and one `720 × 1080` image. Then simulate single-token Decode after Prefill to verify input shapes and language/vision paths, and analyze memory usage and performance bottlenecks.

**Full command:**

```bash
python -m cli.inference.text_generate Qwen/Qwen3-VL-8B-Instruct \
  --device ATLAS_800_A3_752T_128G_DIE \
  --num-devices 1 \
  --num-queries 8 \
  --query-length 32 \
  --context-length 0 \
  --image-batch-size 1 \
  --image-height 720 \
  --image-width 1080 \
  --quantize-linear-action DISABLED \
  --compile \
  --dump-input-shapes
```

**Key output** (truncated)

```text
Number of Queries per DP rank: 8
Model compilation and execution time: 6.411 s
-------------------------------------  ----------------------------------------------------------------------------  --------------  ------------  ----------
                 Name                                                  Input Shapes                                  analytic total  analytic avg  # of Calls
-------------------------------------  ----------------------------------------------------------------------------  --------------  ------------  ----------
aten.mm.default                        [6256, 4096], [4096, 24576]                                                        172.452ms       4.790ms          36
aten.mm.default                        [6256, 12288], [12288, 4096]                                                        86.316ms       2.398ms          36
tensor_cast.attention.default          [2992, 1152], [1, 2992, 16, 72], [1, 2992, 16, 72]                                  42.969ms     198.933us         216
aten.addmm.default                     [4304], [23936, 1152], [1152, 4304]                                                 24.484ms     906.821us          27
tensor_cast.attention.default          [6256, 4096], [49, 128, 8, 128], [49, 128, 8, 128], [8, 7], [9], [8], [8]            12.607ms     350.198us          36
aten.convolution.default               [23936, 3, 2, 16, 16], [1152, 3, 2, 16, 16], [1152]                                326.944us     326.944us           1
...
Total time for analytic: 619.789ms
[analytic] Execution time: 0.619789 s
[analytic] TPS/Device: 413 token/s
Total device memory: 64.000 GB
  Model weight size: 16.455 GB
  KV cache: 0.861 GB
  Model activation size: 0.862 GB
  Memory available: 45.821 GB
Stats breakdowns:
  analytic_OpBound: memory_bound: 24.06, communication_bound: 0.00, compute_bound_mma: 74.40, compute_bound_gp: 1.55
```

**Result analysis:**

1. Omitting `--decode` and setting `--context-length 0` selects initial Prefill. With the default `DP=1`, one DP rank handles all eight requests, matching `Number of Queries per DP rank: 8`.
2. The processor resizes each image from `720 × 1080` to `704 × 1088`. Based on the current processor's input-construction rules, each request has an internal `image_grid_thw` of `[1, 44, 68]`, representing `2992` patches. A `2 × 2` spatial merge plus two boundary tokens produces `2992 / 4 + 2 = 750` image tokens. Therefore, the model-side query and sequence length for each request are both `32 + 750 = 782`, and the fused sequence across eight requests has a leading dimension of `8 × 782 = 6256`.
3. The tool derives internal tensor shapes of `pixel_values.shape=[23936, 1536]` and `image_grid_thw.shape=[8, 3]` from the image parameters; neither is a CLI image-input parameter. In the operator table above, the leading dimensions `2992`, `23936`, and `6256` correspond to the vision sequence, flattened image patches, and fused multimodal sequence, respectively, and can be used to verify this derivation. If the model revision or processor configuration changes, use the actual operator shapes reported by `--dump-input-shapes`.
4. Convolutional patch embedding and vision attention confirm that the image enters the vision tower.
5. `compute_bound_mma: 74.40` is the dominant modeled bottleneck category. `Memory available: 45.821 GB` is positive, so the configuration fits under the current device profile and simulation assumptions. `TPS/Device` counts only the configured `32` text query tokens per request and excludes image tokens; do not compare it directly with text-only TPS.

**Decode configuration:**

The following command independently simulates next-token Decode after image-and-text Prefill. Image tokens are included in `--context-length`, so the image parameters are omitted:

```bash
python -m cli.inference.text_generate Qwen/Qwen3-VL-8B-Instruct \
  --device ATLAS_800_A3_752T_128G_DIE \
  --num-devices 1 \
  --num-queries 8 \
  --query-length 1 \
  --context-length 782 \
  --decode \
  --quantize-linear-action DISABLED \
  --compile \
  --dump-input-shapes
```

`--context-length 782` represents the `32 + 750 = 782` tokens already in the KV cache after image-and-text Prefill: text tokens and image tokens. Decode uses `--query-length 1` to process the current decode token, giving a sequence length of `782 + 1 = 783`.

During Prefill, the image is encoded into tokens that are included in the KV cache. Decode executes only the language path and does not construct `pixel_values` or `image_grid_thw`, nor does it execute the vision tower. When troubleshooting with `--log-level info`, the no-image-input message is expected; do not also set image parameters, or the image tokens will be counted twice.

**Decode key output** (truncated)

```text
Number of Queries per DP rank: 8
Model compilation and execution time: 2.925 s
-------------------------------------  ----------------------------------------------------------------------  --------------  ------------  ----------
                 Name                                               Input Shapes                               analytic total  analytic avg  # of Calls
-------------------------------------  ----------------------------------------------------------------------  --------------  ------------  ----------
aten.mm.default                        [8, 4096], [4096, 24576]                                                        7.062ms     196.169us          36
tensor_cast.attention.default          [8, 4096], [49, 128, 8, 128], [49, 128, 8, 128], [8, 7], [9], [8], [8]         1.060ms      29.432us          36
...
Total time for analytic: 16.927ms
[analytic] Execution time: 0.016927 s
[analytic] TPS/Device: 472.6 token/s
Total device memory: 64.000 GB
  KV cache: 0.861 GB
  Memory available: 47.757 GB
Stats breakdowns:
  analytic_OpBound: memory_bound: 100.00, communication_bound: 0.00, compute_bound_mma: 0.00, compute_bound_gp: 0.00
```

The leading operator-input dimension is `8`, representing one token for each of eight requests. This configuration does not construct image tensors, so Decode executes only the language path; when reproducing the case, confirm in the complete operator table that vision operators such as convolutional patch embedding and vision attention no longer appear.

##### Text-only and VL input configuration differences

| Stage or parameter | Text-only input | VL image-and-text input |
| --- | --- | --- |
| Prefill `--query-length` | Number of text tokens in the current Prefill | `--query-length` represents text tokens only; the tool derives image tokens internally and adds them to the model-side Prefill sequence |
| Prefill image parameters | Omitted | Set `--image-batch-size`, `--image-height`, and `--image-width` together |
| Decode `--context-length` | Number of text tokens already stored in the KV cache | Total number of text and image tokens already stored in the KV cache |

#### FAQ

##### Why does the vision path not run after the image parameters are set?

The vision path runs only during Prefill for a VL model. Ensure that `--decode` is omitted and that `--image-batch-size`, `--image-height`, and `--image-width` are all provided. If any parameter is omitted, a VL model runs without image input; a non-VL model ignores these parameters. Decode executes only the language path. Use `--log-level info` to view the corresponding message.

##### Why does the operator shape differ from the input image size?

The model processor adjusts the image to dimensions compatible with its patch and spatial-merge requirements. In this case, `720 × 1080` is adjusted to `704 × 1088`. If the model revision or processor configuration changes, use the actual operator shapes reported by `--dump-input-shapes`.

##### What do the length parameters mean in Prefill and Decode?

For Prefill, omit `--decode`. Initial Prefill normally uses `--context-length 0`, with `--query-length` representing the token count of the complete text prompt. For chunked or incremental Prefill, `--context-length` represents the model-side tokens already stored in the KV cache, while `--query-length` represents the new tokens in the current chunk.

For Decode, set `--decode`. Standard single-token Decode normally uses `--query-length 1`. When speculative decoding is enabled, the CLI aligns the Decode `--query-length` to `N + 1`, where `N` is the number of speculative tokens. For MTP, prefer the unified interface `--speculative-method mtp --num-speculative-tokens N`; the legacy `--num-mtp-tokens N` option is retained for backward compatibility. `--context-length` represents the model-side tokens already stored in the KV cache before the current Decode step. For a VL model, this length also includes image tokens produced during Prefill.

##### What should be done if the command succeeds but memory is insufficient?

`Memory available < 0` means that the target device cannot hold the current configuration. Adjust the model, parallel strategy, `--num-queries`, `--image-batch-size`, or sequence length.

### 4.2 Video Generation

We provide the `video_generate.py` CLI to simulate the forward pass and performance of diffusion transformer models. This script supports simulating the inference of video generation models (for example, Stable Video Diffusion-like architectures) with configurable input dimensions, sampling steps, and parallelism settings. A detailed table summary of the operator performance breakdown is provided by default. You can also choose to export the performance timeline as a Chrome Trace file.

Its general usage is as follows:

```text
usage: video_generate.py [-h]
                         [--device {TEST_DEVICE,ATLAS_800_A2_376T_64G,ATLAS_800_A2_313T_64G,ATLAS_800_A2_280T_64G,ATLAS_800_A2_280T_64G_PCIE,ATLAS_800_A2_280T_32G_PCIE,ATLAS_800_A3_752T_128G_DIE,ATLAS_800_A3_560T_128G_DIE,ATLAS_800_A3_560T_128G_DIE_ROCE,ATLAS_350_425T_112G,ATLAS_350_425T_84G}]
                         --batch-size BATCH_SIZE --seq-len SEQ_LEN [--chrome-trace CHROME_TRACE] [--height HEIGHT]
                         [--width WIDTH] [--frame-num FRAME_NUM] [--sample-step SAMPLE_STEP]
                         [--log-level {debug,info,warning,error,critical}] [--dtype {float16,float32,bfloat16}]
                         [--remote-source {huggingface,modelscope}]
                         [--quantize-linear-action {DISABLED,W8A16_STATIC,W8A8_STATIC,W4A8_STATIC,W8A16_DYNAMIC,W8A8_DYNAMIC,W4A8_DYNAMIC,FP8,MXFP4}]
                         [--use-cfg] [--world-size WORLD_SIZE]
                         [--ulysses-size ULYSSES_SIZE] [--cfg-parallel] [--dit-cache]
                         [--cache-step-range CACHE_STEP_RANGE] [--cache-step-interval CACHE_STEP_INTERVAL]
                         [--cache-block-range CACHE_BLOCK_RANGE]
                         model_id

Run a simulated diffusion transformer forward and dump perf stats.
```

The main parameters are as follows:

| Parameter                 | Category | Optional/Required | Description |
| ------------------------  | --- | --- | --- |
| model_id                  | positional arguments | Required | Video generation model ID or local model path.<br>1. Type: Str.<br>2. Reference values: a Diffusers model directory, a remote repo ID, or a repo ID with subdirectories, which must contain `transformer/config.json` or a compatible transformer configuration.<br>3. Default: None.<br>4. You are advised to use a reviewed local absolute path. Remote model IDs do not provide security guarantees. |
| --device                  | options | Optional | Specifies the device configuration used for the simulation.<br>1. Type: Str.<br>2. Reference values: registered `DeviceProfile` names, including `TEST_DEVICE`, `ATLAS_800_A2_376T_64G`, `ATLAS_800_A2_313T_64G`, `ATLAS_800_A2_280T_64G`, `ATLAS_800_A2_280T_64G_PCIE`, `ATLAS_800_A2_280T_32G_PCIE`, `ATLAS_800_A3_752T_128G_DIE`, `ATLAS_800_A3_560T_128G_DIE`, `ATLAS_800_A3_560T_128G_DIE_ROCE`, `ATLAS_350_425T_112G`, `ATLAS_350_425T_84G`.<br>3. Default: `TEST_DEVICE`. |
| --batch-size              | options | Required | Specifies the input batch size.<br>1. Type: Int.<br>2. Value range: positive integer.<br>3. Default: None. |
| --seq-len                 | options | Required | Specifies the text sequence length.<br>1. Type: Int.<br>2. Value range: positive integer.<br>3. Default: None. |
| --chrome-trace            | options | Optional | Specifies the Chrome trace JSON output path for exporting the performance timeline.<br>1. Type: Str.<br>2. Value range: file path.<br>3. Default: `None`. |
| --height                  | options | Optional | Specifies the height of the input video or image frames.<br>1. Type: Int.<br>2. Value range: positive integer.<br>3. Default: 400. |
| --width                   | options | Optional | Specifies the width of the input video or image frames.<br>1. Type: Int.<br>2. Value range: positive integer.<br>3. Default: 832. |
| --frame-num               | options | Optional | Specifies the number of video frames.<br>1. Type: Int.<br>2. Value range: positive integer.<br>3. Default: 81. |
| --sample-step             | options | Optional | Specifies the number of diffusion sampling steps.<br>1. Type: Int.<br>2. Value range: positive integer.<br>3. Default: 1. |
| --log-level               | options | Optional | Specifies the log level.<br>1. Type: Str.<br>2. Reference values: `debug`, `info`, `warning`, `error`, `critical`.<br>3. Default: `info`. |
| --dtype                   | options | Optional | Specifies the computation data type of the model.<br>1. Type: Str.<br>2. Reference values: `float16`, `float32`, `bfloat16`.<br>3. Default: `float16`. |
| --remote-source           | options | Optional | Specifies the remote model source for a non-local Diffusers repo ID.<br>1. Type: Str.<br>2. Reference values: `huggingface`, `modelscope`.<br>3. Default: `huggingface`. |
| --quantize-linear-action  | options | Optional | Specifies the quantization scheme for linear layers.<br>1. Type: Str.<br>2. Reference values: `DISABLED`, `W8A16_STATIC`, `W8A8_STATIC`, `W4A8_STATIC`, `W8A16_DYNAMIC`, `W8A8_DYNAMIC`, `W4A8_DYNAMIC`, `FP8`, `MXFP4`.<br>3. Default: `W8A8_DYNAMIC`. |
| --use-cfg                 | options | Optional | Enables the simulation path related to classifier-free guidance.<br>1. Type: Bool.<br>2. Value range: on/off flag.<br>3. Default: `False`. |
| --world-size              | Parallel Options | Optional | Specifies the total number of devices participating in the distributed simulation.<br>1. Type: Int.<br>2. Value range: positive integer.<br>3. Default: 1. |
| --ulysses-size            | Parallel Options | Optional | Specifies the Ulysses parallel size.<br>1. Type: Int.<br>2. Value range: positive integer.<br>3. Default: 1. |
| --cfg-parallel            | Parallel Options | Optional | Enables the CFG parallel strategy.<br>1. Type: Bool.<br>2. Value range: on/off flag.<br>3. Default: `False`. |
| --dit-cache               | Cache Options | Optional | Enables the DiT block cache.<br>1. Type: Bool.<br>2. Value range: on/off flag.<br>3. Default: `False`. |
| --cache-step-range        | Cache Options | Optional | Specifies the range of sampling steps for which the cache is enabled.<br>1. Type: Str.<br>2. Format: `start,end`, a closed interval.<br>3. Default: `None`.<br>4. Required when `--dit-cache` is set. |
| --cache-step-interval     | Cache Options | Optional | Specifies the step interval for cache updates.<br>1. Type: Int.<br>2. Value range: positive integer.<br>3. Default: 1, which means cache update reuse is disabled. |
| --cache-block-range       | Cache Options | Optional | Specifies the range of blocks for which the cache is enabled.<br>1. Type: Str.<br>2. Format: `start,end`, left-closed and right-open.<br>3. Default: `None`. |

Run `python -m cli.inference.video_generate --help` for details.

### 4.3 Image Generation

We provide the `image_generate.py` command-line interface to simulate the diffusion transformer denoising workload and performance of image generation models. This script supports simulating the Transformer denoising inference process of image generation models (such as FLUX and Qwen-Image-Edit), with configurable batch size, output image size, text condition length, and parallel settings. It provides a detailed operator performance breakdown summary by default. The performance timeline can also be exported to a Chrome Trace file.

Only the Transformer denoising stage entering the simulation device is simulated in the first release; prompt encoding, VAE, scheduler, and image I/O are excluded, and no actual image is generated.

Its general usage is as follows:

```text
msmodeling inference image-generate MODEL --batch-size <N> --output-image-size HEIGHT WIDTH --text-seq-len <N>
```

The full `--help` output also includes `--version/-V`, `--verbose/-v`, and `--quiet/-q`. The default `--log-level` is `error`. This tool does not provide `--debug` or `--log-file`. The model source is provided as the positional argument `model_id` or with `--model-id`. `--num-devices` is the formal parallel scale; `--world-size` and `--chrome-trace` are hidden compatibility aliases.

The main parameters are as follows:

| Parameter | Category | Optional/Required | Description |
| --- | --- | --- | --- |
| `model_id` / `--model-id` | positional / options | Required (choose one) | Image generation model ID or local model path. Can be provided as a positional argument or with `--model-id`.<br>1. Type: Str.<br>2. Reference values: a Diffusers model directory or an exact allowed remote repo ID, such as `black-forest-labs/FLUX.1-dev` or `Qwen/Qwen-Image-Edit`.<br>3. Default: None.<br>4. You are advised to use a reviewed local absolute path; remote model IDs do not provide security guarantees. |
| `--device` | options | Optional | Specifies the device configuration used for the simulation.<br>1. Type: Str.<br>2. Reference values: any registered `DeviceProfile` name; built-in values are listed in Section 3, Supported Devices and Custom Devices.<br>3. Default: `TEST_DEVICE`. |
| `--batch-size` | options | Required | Specifies the batch size of the base workload, not the number of prompts or source images.<br>1. Type: Int.<br>2. Value range: positive integer.<br>3. Default: None. |
| `--output-image-size` | options | Required | Specifies the output image size and must appear exactly once; it is used only to derive shapes and does not output an image.<br>1. Type: Tuple[Int, Int] (`HEIGHT WIDTH`).<br>2. Value range: two positive integers.<br>3. Default: None. |
| `--text-seq-len` | options | Required | Specifies the actual text condition length entering the Transformer.<br>1. Type: Int.<br>2. Value range: positive integer.<br>3. Default: None.<br>4. It is not the character count, tokenizer input length, or template length; this first release does not perform text encoding. |
| `--source-image-size` | options | Optional | Specifies the source image size and can be repeated for each source; accepts sizes only, not paths or pixels. Available only for the editing kind.<br>1. Type: Tuple[Int, Int] (`HEIGHT WIDTH`).<br>2. Value range: two positive integers.<br>3. Default: None. |
| `--sample-step` | options | Optional | Specifies the number of identical Transformer workload iterations to execute.<br>1. Type: Int.<br>2. Value range: positive integer.<br>3. Default: `1`. |
| `--use-cfg` | options | Optional | Enables a video-style classifier-free guidance workload approximation.<br>1. Type: Bool.<br>2. Value range: on/off flag.<br>3. Default: `False`. |
| `--dtype` | options | Optional | Specifies the computation data type of the model.<br>1. Type: Str.<br>2. Reference values: `float16`, `float32`, `bfloat16`.<br>3. Default: `float16`. |
| `--remote-source` | options | Optional | Specifies the remote model source; it participates in exact-pair matching.<br>1. Type: Str.<br>2. Reference values: `huggingface`, `modelscope`.<br>3. Default: `huggingface`. |
| `--quantize-linear-action` | Quantization Options | Optional | Specifies the quantization scheme for linear layers.<br>1. Type: Str.<br>2. Reference values: `DISABLED`, `W8A16_STATIC`, `W8A8_STATIC`, `W4A8_STATIC`, `W8A16_DYNAMIC`, `W8A8_DYNAMIC`, `W4A8_DYNAMIC`, `FP8`, `MXFP4`.<br>3. Default: `DISABLED`.<br>4. Case and underscore/hyphen forms are both accepted, for example `W8A8_DYNAMIC` and `w8a8-dynamic`, or `DISABLED` and `disabled`. |
| `--mxfp4-group-size` | Quantization Options | Optional | Specifies the group size for MXFP4 quantization.<br>1. Type: Int.<br>2. Value range: positive integer.<br>3. Default: `32`. |
| `--quantize-attention-action` | Quantization Options | Optional | Specifies the quantization scheme for attention computation.<br>1. Type: Str.<br>2. Reference values: `DISABLED`, `INT8`, `FP8`.<br>3. Default: `DISABLED`.<br>4. Case and underscore/hyphen forms are both accepted, for example `DISABLED` and `disabled`, or `INT8` and `int8`. |
| `--compile` | Optimization Options | Optional | Compiles the main transformer before simulation.<br>1. Type: Bool.<br>2. Value range: on/off flag.<br>3. Default: `False`.<br>4. Uses `dynamic=False, fullgraph=True`; the cache transformer uses the same strategy when DiT cache is enabled. |
| `--compile-allow-graph-break` | Optimization Options | Optional | Allows graph breaks during compilation.<br>1. Type: Bool.<br>2. Value range: on/off flag.<br>3. Default: `False`.<br>4. Changes the compilation mode of the main transformer and DiT cache transformer to `fullgraph=False`. |
| `--num-devices` | Parallel Options | Optional | Specifies the total number of devices participating in distributed simulation.<br>1. Type: Int.<br>2. Value range: positive integer.<br>3. Default: `1`.<br>4. Must equal `--ulysses-size`, or `2 * --ulysses-size` when `--cfg-parallel` is enabled. The old name `--world-size` is still parsed. |
| `--ulysses-size` | Parallel Options | Optional | Specifies the Ulysses parallel size.<br>1. Type: Int.<br>2. Value range: positive integer.<br>3. Default: `1`. |
| `--cfg-parallel` | Parallel Options | Optional | Enables the CFG parallel strategy.<br>1. Type: Bool.<br>2. Value range: on/off flag.<br>3. Default: `False`.<br>4. Enabled only with `--use-cfg`; in this case, `--num-devices` must equal `2 * --ulysses-size`. |
| `--dit-cache` | Cache Options | Optional | Enables the DiT block cache.<br>1. Type: Bool.<br>2. Value range: on/off flag.<br>3. Default: `False`. |
| `--cache-step-range` | Cache Options | Optional | Specifies the sampling-step range for which the cache is enabled.<br>1. Type: Str.<br>2. Format: `start,end`, a closed interval.<br>3. Default: `None`.<br>4. Required when `--dit-cache` is set and `--cache-step-interval > 1`. |
| `--cache-step-interval` | Cache Options | Optional | Specifies the step interval for cache updates.<br>1. Type: Int.<br>2. Value range: positive integer.<br>3. Default: `1`, which means cache update reuse is disabled. |
| `--cache-block-range` | Cache Options | Optional | Specifies the range of blocks for which the cache is enabled.<br>1. Type: Str.<br>2. Format: `start,end`, left-closed and right-open.<br>3. Default: `None`. |
| `--chrome-trace-file` | options | Optional | Specifies the Chrome trace JSON output path for exporting the performance timeline.<br>1. Type: Str.<br>2. Value range: file path.<br>3. Default: `None`.<br>4. Generated only after Runtime succeeds. The old name `--chrome-trace` is still parsed. |

> **Note:** Denoising workload simulation currently supports FLUX.1-dev and Qwen-Image-Edit (the three variants `Qwen/Qwen-Image-Edit`, `Qwen/Qwen-Image-Edit-2509`, and `Qwen/Qwen-Image-Edit-2511`).

> **Limitation:** Qwen-Image-Edit currently does not support Ulysses sequence parallelism (`--ulysses-size > 1`), because input sharding has not been implemented. When `--ulysses-size` is greater than `1`, simulation fails with a Qwen-specific error before Runtime and does not fall back to the non-parallel path.

Run `python -m cli.inference.image_generate --help` for details.
