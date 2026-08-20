# Model Inference Performance Simulation User Guide

For the complete model list and feature details, see [Model and Feature Support Matrix](./support_matrix/support_matrix_user_guide.md).

## Reading Paths

| Goal | Recommended Section |
| --- | --- |
| 1. Quickly run LLM text generation simulation | [2.1 Quick Start: Text Generation](#21-quick-start-text-generation) |
| 2. Understand the latency, invocation count, and memory metrics in the output | [2.2 Result (Text Generation)](#22-result-text-generation) |
| 3. Run video generation model simulation | [2.3 Quick Start: Video Generation](#23-quick-start-video-generation) |
| 4. View or customize hardware device profiles | [3 Supported Devices and Custom Devices](#3-supported-devices-and-custom-devices) |

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

We provide the `text_generate.py` CLI to simulate text generation. This script supports text generation simulation for a batch of queries with the same input length and optionally the same context length. A table summary of the operator performance breakdown is provided by default. You can also choose to export a Chrome trace.

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
