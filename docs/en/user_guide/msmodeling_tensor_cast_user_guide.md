# Model Inference Performance Simulation User Guide 

For the complete model list and feature details, see the [Model and Feature Support Matrix](./support_matrix/support_matrix_user_guide.md).

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
- Computational characteristics: analyzes the floating point operations (FLOPs) and memory access volume of each operator.
- Advanced scheduling simulation: models complex execution patterns, such as concurrent computation across multiple streams.

The final output includes comprehensive summary tables and detailed Chrome Trace files, enabling in-depth visualization and identification of performance bottlenecks.

Before first use, see the [msModeling Installation Guide](../install_guide/msmodeling_install_guide.md) to set up the environment and run an LLM inference simulation.

## 2 At a Glance

### 2.1 Quick Start: Text Generation

**What it does:** Simulates the LLM inference performance of a batch of queries.

#### Prefill Scenario

To run prefill for Qwen3-32B on `TEST_DEVICE` with two requests, each with a 3,500-token input length, run the following command:

```bash
python -m cli.inference.text_generate Qwen/Qwen3-32B --num-queries 2 --query-length 3500 --context-length 3500 --device TEST_DEVICE --compile
```

You can also add the `--prefill` flag explicitly (this is equivalent to the preceding command and lets you identify the phase at a glance):

```bash
python -m cli.inference.text_generate Qwen/Qwen3-32B --prefill --num-queries 2 --query-length 3500 --context-length 3500 --device TEST_DEVICE --compile
```

In prefill mode, do not add `--decode` (or specify `--prefill` explicitly). `--query-length` specifies the new input length, and `--context-length` specifies the context length of each request. `--prefill` and `--decode` are mutually exclusive, and passing both causes an error. When neither is passed, the simulation runs in prefill mode by default (backward compatible).

You can also quantize the linear layers with various quantization schemes, such as `W8A8` dynamic quantization, using a 4500-token context as the prefix:

```bash
python -m cli.inference.text_generate Qwen/Qwen3-32B --num-queries 2 --query-length 3500 --context-length 4500 --device TEST_DEVICE --quantize-linear-action W8A8_DYNAMIC --compile
```

#### Decode Scenario

Running the decode scenario is similar. You only need to adjust the input length `--query-length` and the context length `--context-length` of the requests. When MTP is disabled, `--query-length` is usually 1. When MTP is enabled, set `--query-length` to `1 + N`, where `N` is the number of MTP tokens. You are advised to use the unified interface `--speculative-method mtp --num-speculative-tokens N`. The original `--num-mtp-tokens` is still supported for compatibility. In this case, `--query-length` in decode must also be aligned to `N + 1` (the CLI can align it automatically).

```bash
python -m cli.inference.text_generate Qwen/Qwen3-32B --num-queries 10 --query-length 1 --context-length 4500 --decode --device TEST_DEVICE --quantize-linear-action W8A8_STATIC --compile
```

**Output:** A performance summary table. If `--chrome-trace-file` is set, a Chrome trace file is optionally generated.

#### Speculative Decoding (Dflash/DSpark)

TensorCast supports Dflash or DSpark draft speculative decoding simulation through `--speculative-method {dflash,dspark}`, which is mutually exclusive with MTP (`--num-mtp-tokens`). When enabled, each forward pass includes the complete target forward pass and the draft block forward pass. Draft linear layers do not participate in `--quantize-linear-action` quantization.

> Note: For MTP, you are advised to use the unified interface `--speculative-method mtp --num-speculative-tokens N` (semantically equivalent to the original `--num-mtp-tokens`). The legacy entry point `--num-mtp-tokens` cannot be combined with the new entry points `--speculative-method` and `--num-speculative-tokens`. When you select `mtp`, you must also configure `--num-speculative-tokens`. The original `--num-mtp-tokens` is still supported on its own for compatibility and will be gradually deprecated in later versions. The new entry point `--speculative-method` accepts `{mtp,dflash,dspark}`.

**Example (Decode, Dflash):**

```bash
python -m cli.inference.text_generate Qwen/Qwen3-32B \
  --num-queries 8 \
  --query-length 8 \
  --context-length 4500 \
  --decode \
  --device TEST_DEVICE \
  --speculative-method dflash \
  --num-speculative-tokens 7 \
  --num-draft-layers 6 \
  --compile
```

Notes:

- You must pass `--speculative-method` explicitly. Setting only dependent parameters such as `--num-speculative-tokens` does not enable speculative decoding.
- `--query-length` must be no less than the draft block length (`block_size = --num-speculative-tokens + 1`, or the built-in default when it is not specified). When `--speculative-method` is set, an explicit `--num-speculative-tokens 0` fails parsing. To disable speculative decoding, omit `--speculative-method`. When `--num-speculative-tokens` is omitted, dflash and dspark still use the built-in `block_size`.
- `--num-draft-layers` and `--draft-model-config-path` apply only to `dflash` and `dspark` and cannot be used together with `--speculative-method mtp`.
- When you select `dspark`, `--dspark-markov-rank` and `--dspark-markov-head` are optional (defaults: 256 and `vanilla`).
- `--acceptance-length` is used only for decode throughput conversion in `throughput_optimizer`. `text_generate` does not support this parameter.
- The default draft configuration is stored in `tensor_cast/runtime_configs/draft_configs/dflash_draft_builtin.json`. For field descriptions and customization methods, see [`tensor_cast/runtime_configs/draft_configs/README.md`](../../../tensor_cast/runtime_configs/draft_configs/README.md).

**Output:** A performance summary table. If `--chrome-trace-file` is set, a Chrome trace file is optionally generated.

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

Metrics:

- `analytic total`: estimated total time spent by operators
- `analytic avg`: average time per operator invocation
- `# of Calls`: number of times an operator is invoked
- `Total time for analytic`: sum of analytic operator time
- `TPS/Device`: tokens per second per device
- `Total device memory` and its breakdown: estimated memory usage of weights, KV cache, and activations

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

To use the block sparse attention (BSA) backend, add the BSA-specific options to the preceding dense example:

```bash
python -m cli.inference.video_generate Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --device ATLAS_800_A2_280T_32G_PCIE \
  --batch-size 1 \
  --seq-len 128 \
  --attention-backend block_sparse_attention \
  --attention-block-size 128 \
  --attention-sparsity 0.5
```

**Key parameters:** `model_id`, `--device`, `--batch-size`, `--seq-len`, `--height`, `--width`, `--frame-num`, `--sample-step`, `--dtype`, `--quantize-linear-action`, `--chrome-trace-file`, `--attention-backend`, `--attention-block-size`, `--attention-sparsity`

**Output:** A performance summary table. If `--chrome-trace-file` is set, a Chrome trace file is optionally generated.

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

Metrics:

- `analytic total`: estimated total time spent by operators
- `analytic avg`: average time per operator invocation
- `# of Calls`: number of times an operator is invoked
- `Total time for analytic`: sum of analytic operator time

### 2.5 Quick Start: Image Generation

**What it does:** Simulates the diffusion transformer denoising workload of image generation models. This first release simulates only the Transformer denoising workload and does not run prompt encoding, VAE, the scheduler, or actual image generation. The following example uses the FLUX.1-dev remote model ID. On the first run, the required model configuration files are pulled according to the configuration.

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

**Key parameters:** `model_id`/`--model-id`, `--device`, `--batch-size`, `--output-image-size`, `--text-seq-len`, `--source-image-size`, `--sample-step`, `--use-cfg`, `--num-devices`, `--ulysses-size`, `--cfg-parallel`, `--dit-cache`, `--chrome-trace-file`

**Output:** `Runtime execution time`, followed by a performance summary table. If `--chrome-trace-file` is set, a Chrome trace file is exported at the end and its path is printed.

### 2.6 Result (Image Generation)

`image_generate` reports the critical path and the logical measured workload of the Transformer denoising stage. It outputs `Runtime execution time` and a performance summary table in sequence. If `--chrome-trace-file` is set, it then exports a Chrome trace and prints its path.

- `Runtime execution time`: the runtime of the simulator on the host machine, not the actual model compilation or execution time on the hardware
- `analytic total`: estimated total time spent by operators
- `analytic avg`: average time per operator invocation
- `# of Calls`: number of times an operator is invoked
- `Total time for analytic`: sum of analytic operator time

The actual values vary with the device configuration, model configuration, and input dimensions.

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
- `ATLAS_850_486T_112G`
- `ATLAS_850_486T_128G`
- `ATLAS_850E_486T_96G`
- `ATLAS_850E_425T_96G`
- `ATLAS_850_486T_112G_ROCE`
- `ATLAS_850_486T_128G_ROCE`
- `ATLAS_850E_486T_96G_ROCE`
- `ATLAS_850E_425T_96G_ROCE`
- `ATLAS_850_486T_112G_FM16`
- `ATLAS_850_486T_128G_FM16`
- `ATLAS_850E_486T_96G_FM16`
- `ATLAS_850E_425T_96G_FM16`
- `ATLAS_950_486T_128G`
- `ATLAS_950_486T_96G`
- `ATLAS_950_486T_144G`

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

We provide the `text_generate.py` CLI to simulate text generation. It supports text-only input and VL image-text input. This script supports simulation for a batch of queries with the same input length and optionally the same context length. A table summary of the operator performance breakdown is provided by default. You can also choose to export a Chrome trace.

Its general usage is as follows:

```text
usage: text_generate.py [-h]
                        [--device <NAME>]
                        [--num-devices NUM_DEVICES] [--reserved-memory-gb RESERVED_MEMORY_GB]
                        [--log-level {debug,info,warning,error,critical}] --num-queries NUM_QUERIES
                        --query-length QUERY_LENGTH [--context-length CONTEXT_LENGTH] [--prefill] [--decode]
                        [--prefix-cache-hit-rate PREFIX_CACHE_HIT_RATE] [--num-mtp-tokens NUM_MTP_TOKENS]
                        [--no-repetition] [--compile] [--compile-allow-graph-break]
                        [--speculative-method {mtp,dflash,dspark}] [--num-speculative-tokens NUM_SPECULATIVE_TOKENS]
                        [--dspark-markov-rank DSPARK_MARKOV_RANK]
                        [--dspark-markov-head {vanilla,gated,rnn}]
                        [--num-draft-layers NUM_DRAFT_LAYERS]
                        [--draft-model-config-path DRAFT_MODEL_CONFIG_PATH]
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
                        [--remote-source {huggingface,modelscope}] [--performance-model {analytic,calibrated,profiling}]
                        [--profiling-database-path PROFILING_DATABASE] [--disable-profiling-interpolation]
                        [--analytic-calibration-profile ANALYTIC_CALIBRATION_PROFILE]
                        [--analytic-calibration-stack ANALYTIC_CALIBRATION_STACK]
                        [--export-empirical-metrics EXPORT_EMPIRICAL_METRICS]
                        model_id

Run a simulated LLM inference pass and dump the perf result.
```

The main parameters are as follows:

| Parameter                            | Category | Optional/Required | Description |
| -----------------------------------  | --- | --- | --- |
| model_id                             | General Options | Required | Model ID or local model path.<br>1. Type: Str.<br>2. Reference values: Hugging Face ID, ModelScope ID, or a local absolute path, for example `Qwen/Qwen3-32B` or `/data/models/Qwen3-32B`.<br>3. Default: none.<br>4. When using a remote model ID, remote code may be executed through `trust_remote_code=True`. |
| --device                             | General Options | Optional | Specifies the device configuration used for the simulation.<br>1. Type: Str.<br>2. Reference values: any registered `DeviceProfile` name. See section 3, "Device Types", for the built-in values.<br>3. Default: `TEST_DEVICE`. |
| --num-devices                        | General Options | Optional | Specifies the number of devices participating in the simulation.<br>1. Type: Int.<br>2. Value range: positive integer.<br>3. Default: 1. |
| --reserved-memory-gb                 | General Options | Optional | Specifies the amount of device memory reserved for the system on each device, in GB.<br>1. Type: Float.<br>2. Value range: non-negative number. Set to 0 to reserve no system memory.<br>3. Default: 0.0. |
| --log-level                          | General Options | Optional | Specifies the log level.<br>1. Type: Str.<br>2. Reference values: `debug`, `info`, `warning`, `error`, `critical`.<br>3. Default: `error`. |
| --num-queries                        | LLM Options | Required | Number of queries in this simulation.<br>1. Type: Int.<br>2. Value range: positive integer.<br>3. Default: none. |
| --query-length                       | LLM Options | Required | New input token length of each query.<br>1. Type: Int.<br>2. Value range: positive integer.<br>3. Default: none. |
| --context-length                     | LLM Options | Optional | Existing context token length of each query.<br>1. Type: Int.<br>2. Value range: non-negative integer.<br>3. Default: 0. |
| --decode                             | LLM Options | Optional | Enables autoregressive decode mode. When not set, the simulation runs in prefill mode.<br>1. Type: Bool.<br>2. Value range: on/off flag.<br>3. Default: `False`.<br>4. Mutually exclusive with `--prefill`. |
| --prefill                            | LLM Options | Optional | Explicitly specifies the prefill phase, which lets you identify the phase at a glance.<br>1. Type: Bool.<br>2. Value range: on/off flag.<br>3. Default: `False`.<br>4. Mutually exclusive with `--decode`. When neither is passed, the simulation runs in prefill mode by default (backward compatible). |
| --prefix-cache-hit-rate              | LLM Options | Optional | Specifies the prefix cache hit rate, used to approximate prefill token reuse.<br>1. Type: Float.<br>2. Value range: [0, 1).<br>3. Default: 0.0. |
| --num-mtp-tokens                     | LLM Options | Optional | Specifies the number of Multi-Token Prediction (MTP) tokens. 0 disables MTP.<br>1. Type: Int.<br>2. Value range: non-negative integer.<br>3. Default: 0.<br>4. Supported only by models with MTP capability, for example DeepSeek.<br>5. Legacy MTP entry point. It cannot be combined with `--speculative-method` or `--num-speculative-tokens`.<br>6. You are advised to use the unified interface `--speculative-method mtp --num-speculative-tokens N` first (semantically equivalent). This parameter is still supported on its own for compatibility and will be gradually deprecated in later versions. |
| --no-repetition                      | LLM Options | Optional | Disables the transformer repetition pattern optimization and preserves the original model behavior.<br>1. Type: Bool.<br>2. Value range: on/off flag.<br>3. Default: `False`. |
| --speculative-method                 | LLM Options | Optional | Specifies the speculative decoding method.<br>1. Type: Str.<br>2. Values: `mtp`, `dflash`, or `dspark`.<br>3. Default: not specified (disabled).<br>4. New speculative decoding entry point. It cannot be combined with the legacy MTP entry point `--num-mtp-tokens`.<br>5. When you select `mtp`, you must also configure `--num-speculative-tokens`. This parameter is the prerequisite switch for dependent parameters such as `--num-speculative-tokens`. |
| --num-speculative-tokens             | LLM Options | Optional | Number of speculative tokens, or depth `n` (excluding the anchor and bonus tokens).<br>1. Type: Int (`text_generate` accepts only a single value).<br>2. Semantics: when the value is `>= 1`, the internal `block_size = n + 1`. When the value is **omitted**, dflash and dspark use the `block_size` from the built-in configuration or from `--draft-model-config-path`. When `--speculative-method` is set, an **explicit** 0 fails parsing (to disable speculative decoding, omit `--speculative-method`, and to enable it, pass `n >= 1`). For `mtp`, `n` is the number of MTP tokens and must be explicitly configured as `>= 1`.<br>3. Default: not specified (dflash and dspark use the `block_size` from the built-in configuration or from `--draft-model-config-path`).<br>4. Requires `--speculative-method` to be set first. |
| --dspark-markov-rank                 | LLM Options | Optional | Markov embedding dimension.<br>1. Type: Int.<br>2. Value range: non-negative integer. 0 disables MarkovHead.<br>3. Default: 256.<br>4. Requires `--speculative-method dspark`. |
| --dspark-markov-head                 | LLM Options | Optional | Markov head type.<br>1. Type: Str.<br>2. Reference values: `vanilla`, `gated`, `rnn`.<br>3. Default: `vanilla`.<br>4. Requires `--speculative-method dspark`. |
| --num-draft-layers                   | LLM Options | Optional | Overrides `num_hidden_layers` of the draft model.<br>1. Type: Int.<br>2. Value range: non-negative integer. 0 means the default value from the configuration is used.<br>3. Default: 0.<br>4. Requires `--speculative-method dflash` or `dspark` (cannot be used together with `mtp`). |
| --draft-model-config-path            | LLM Options | Optional | Specifies an external draft `config.json` (or a directory that contains it) to override the built-in draft profile.<br>1. Type: Str.<br>2. Value range: file path or directory path.<br>3. Default: `None` (uses `tensor_cast/runtime_configs/draft_configs/dflash_draft_builtin.json`).<br>4. Requires `--speculative-method dflash` or `dspark` (cannot be used together with `mtp`). For the default configuration and field descriptions, see [`tensor_cast/runtime_configs/draft_configs/README.md`](../../../tensor_cast/runtime_configs/draft_configs/README.md). |
| --compile                            | Optimization Options | Optional | Calls `torch.compile()` on the model before inference.<br>1. Type: Bool.<br>2. Value range: on/off flag.<br>3. Default: `False`. |
| --compile-allow-graph-break          | Optimization Options | Optional | Allows graph breaks during `torch.compile()`.<br>1. Type: Bool.<br>2. Value range: on/off flag.<br>3. Default: `False`. |
| --compilation-config                 | Optimization Options | Optional | Dynamically enables the specified compilation features. You can specify multiple options at a time.<br>1. Type: List[Str] (`nargs="*"`).<br>2. Values: `enable_multistream` (enables multi-stream scheduling), `enable_sequence_parallel` (enables the sequence parallel graph rewrite pass), `enable_matmul_allreduce` (enables matmul-allreduce fusion), and `enable_dispatch_ffn_combine` (enables dispatch_ffn_combine fusion).<br>3. Default: when not specified, all compilation features remain disabled (`False`).<br>4. Example: `--compilation-config enable_multistream enable_sequence_parallel`.<br>5. Both uppercase and lowercase values and underscore or hyphen spellings are accepted, for example `enable_multistream` and `enable-multistream`. |
| --quantize-linear-action             | Quantization Options | Optional | Specifies the quantization scheme for linear layers.<br>1. Type: Str.<br>2. Reference values: `DISABLED`, `W8A16_STATIC`, `W8A8_STATIC`, `W4A8_STATIC`, `W8A16_DYNAMIC`, `W8A8_DYNAMIC`, `W4A8_DYNAMIC`, `FP8`, `MXFP4`.<br>3. Default: `W8A8_DYNAMIC`.<br>4. Both uppercase and lowercase values and underscore or hyphen spellings are accepted, for example `W8A8_DYNAMIC` and `w8a8-dynamic`, or `DISABLED` and `disabled`. |
| --quantize-non-expert-linear-action  | Quantization Options | Optional | Specifies an independent quantization scheme for non-expert linear layers, such as attention projections, dense MLPs, and shared experts.<br>1. Type: Str.<br>2. Reference values: `DISABLED`, `W8A16_STATIC`, `W8A8_STATIC`, `W4A8_STATIC`, `W8A16_DYNAMIC`, `W8A8_DYNAMIC`, `W4A8_DYNAMIC`, `FP8`, `MXFP4`.<br>3. Default: `DISABLED`.<br>4. Both uppercase and lowercase values and underscore or hyphen spellings are accepted, for example `DISABLED` and `disabled`. |
| --quantize-lmhead                    | Quantization Options | Optional | Enables quantization for the lm head.<br>1. Type: Bool.<br>2. Value range: on/off flag.<br>3. Default: `False`. |
| --mxfp4-group-size                   | Quantization Options | Optional | Specifies the group size for `mxfp4` quantization.<br>1. Type: Int.<br>2. Value range: positive integer.<br>3. Default: 32. |
| --quantize-attention-action          | Quantization Options | Optional | Specifies the quantization scheme for the KV cache.<br>1. Type: Str.<br>2. Reference values: `DISABLED`, `INT8`, `FP8`.<br>3. Default: `DISABLED`.<br>4. Both uppercase and lowercase values and underscore or hyphen spellings are accepted, for example `DISABLED` and `disabled`, or `INT8` and `int8`. |
| --graph-log-path                     | Debugging Options | Optional | Specifies the output **directory** for the compilation graph log. Use it only when debugging the compile path. Each pass writes multiple dump files to this directory.<br>1. Type: Str.<br>2. Value range: directory path.<br>3. Default: `None`. |
| --dump-input-shapes                  | Debugging Options | Optional | Outputs the input shape information to help troubleshoot the model input configuration.<br>1. Type: Bool.<br>2. Value range: on/off flag.<br>3. Default: `False`. |
| --dump-op-bound-results              | Debugging Options | Optional | Outputs the operator-level memory, communication, MMA, and GP bound ratios in the result table.<br>1. Type: Bool.<br>2. Value range: on/off flag.<br>3. Default: `False`. |
| --chrome-trace-file                  | Debugging Options | Optional | Specifies the Chrome trace output path for exporting the performance timeline.<br>1. Type: Str.<br>2. Value range: file path.<br>3. Default: `None`. |
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
| --moe-tp-size                        | Parallelism Options | Optional | Specifies the TP size for experts, overriding `--tp-size`.<br>1. Type: Int.<br>2. Value range: positive integer.<br>3. Default: `None`.<br>4. Note: When the sequence parallel or dispatch_ffn_combine compilation option is enabled, EP parallelism requires this value to be 1 (that is, EP × moe_dp == tp × dp). Otherwise, an error is reported during configuration parsing. |
| --moe-dp-size                        | Parallelism Options | Optional | Specifies the DP size for experts, overriding `--dp-size`.<br>1. Type: Int.<br>2. Value range: positive integer.<br>3. Default: 1. |
| --word-embedding-tp                  | Parallelism Options | Optional | Enables tensor parallelism for word embeddings and specifies the parallel mode.<br>1. Type: Str.<br>2. Reference values: `col`, `row`.<br>3. Default: `None`, which means embedding TP is disabled. |
| --enable-redundant-experts           | Parallelism Options | Optional | Enables the redundant expert configuration.<br>1. Type: Bool.<br>2. Value range: on/off flag.<br>3. Default: `False`.<br>4. When enabled alone, each device hosts one additional redundant expert.<br>5. When enabled together with `--enable-external-shared-experts`, the allocation logic is the same as for external shared experts. If the routing experts are already evenly distributed across devices and no redundant experts are needed to pad them, each device hosting routing experts hosts one additional redundant expert. |
| --enable-shared-expert-tp            | Parallelism Options | Optional | Enables vLLM-style tensor parallelism for shared experts.<br>1. Type: Bool.<br>2. Value range: on/off flag.<br>3. Default: `False`.<br>4. Shared experts use dense MLP TP and defer the `down_proj` reduction. |
| --enable-external-shared-experts     | Parallelism Options | Optional | Enables external shared experts.<br>1. Type: Bool.<br>2. Value range: on/off flag.<br>3. Default: `False`.<br>4. When enabled, devices are allocated between external shared experts and routing experts at a ratio of `1:top_k`. Redundant experts are used to pad the routing experts if needed.<br>5. For example, with `world_size=64`, `top_k=8`, and 256 routing experts, 8 devices host the external shared experts and the remaining 56 devices distribute the 256 routing experts: 32 devices host 5 routing experts each, and 24 devices host 4 routing experts and 1 redundant expert each. |
| --host-external-shared-experts       | Parallelism Options | Optional | Specifies that the current device hosts the external shared experts.<br>1. Type: Bool.<br>2. Value range: on/off flag.<br>3. Default: `False`. |
| --vision-tp-size                     | Parallelism Options | Optional | Specifies the tensor parallel size for the vision module.<br>1. Type: Int.<br>2. Value range: positive integer.<br>3. Default: 1, which means the vision module is not sharded. |
| --image-batch-size                   | MultiModal Options | Optional | Specifies the batch size for image processing.<br>1. Type: Int.<br>2. Value range: positive integer.<br>3. Default: `None`. |
| --image-height                       | MultiModal Options | Optional | Specifies the height of the input images.<br>1. Type: Int.<br>2. Value range: positive integer.<br>3. Default: `None`. |
| --image-width                        | MultiModal Options | Optional | Specifies the width of the input images.<br>1. Type: Int.<br>2. Value range: positive integer.<br>3. Default: `None`. |
| --remote-source                      | Options | Optional | Specifies the remote model source.<br>1. Type: Str.<br>2. Reference values: `huggingface`, `modelscope`.<br>3. Default: `huggingface`. |
| --performance-model                  | Options | Optional | Specifies the performance model. You can specify one or more models repeatedly.<br>1. Type: List[Str].<br>2. Reference values: `analytic`, `calibrated`, `profiling`.<br>3. Default: `analytic` when not specified.<br>4. `analytic` is the Roofline model and does not require profiling data. `calibrated` is an analytic model calibrated by `--analytic-calibration-profile`. `profiling` is an empirical performance model based on the profiling CSV database and requires `--profiling-database-path`. |
| --profiling-database-path            | Options | Optional | Specifies the profiling database path when using the `profiling` performance model.<br>1. Type: Str.<br>2. Value range: path to a directory that contains `op_mapping.yaml` and CSV files for each kernel type.<br>3. Default: `None`. |
| --disable-profiling-interpolation    | Options | Optional | Disables profiling interpolation and uses only exact and partial matches.<br>1. Type: Bool.<br>2. Value range: a switch parameter.<br>3. Default: `False`.<br>4. Requires `--performance-model profiling`. |
| --analytic-calibration-profile       | Options | Optional | Specifies the SQLite profile used to calibrate the analytic model.<br>1. Type: Str.<br>2. Value range: SQLite file path.<br>3. Default: `None`.<br>4. Calibrates analytic latency without requiring profiling OP mapping. |
| --analytic-calibration-stack         | Options | Optional | Specifies the actual software stack of a multi-stack analytic profile. A single-stack profile is selected automatically.<br>1. Type: Str.<br>2. Value range: software stack name.<br>3. Default: `None`, selected automatically. |
| --export-empirical-metrics           | Options | Optional | Exports the M1-M5 metrics JSON for offline M6 calculation.<br>1. Type: Str.<br>2. Value range: JSON file path.<br>3. Default: `None`.<br>4. For development and debugging only. Requires `--performance-model profiling`. |

For VL models, you can set `--image-batch-size`, `--image-height`, and `--image-width` together to describe the number and resolution of the input images. For text-only models, you can omit these parameters.

Run `python -m cli.inference.text_generate --help` for details.

#### Practice Cases and Result Analysis

The following two cases use Hugging Face model IDs to demonstrate how to verify the execution mode, representative operators, input shapes, performance, memory, and bottlenecks from `text_generate` output. `text_generate` constructs only simulated input. It does not read actual text or image files and does not return natural language generation results. The examples disable linear layer quantization. The output values are used only to illustrate the analysis method and are not fixed acceptance thresholds.

`Stats breakdowns` is the normalized proportion of estimated operator time that the analytic performance model classifies by dominant resource. It is not measured hardware utilization. `memory_bound`, `communication_bound`, `compute_bound_mma`, and `compute_bound_gp` represent the memory-bound, communication-bound, matrix-compute-bound, and general-compute-bound components, respectively.

`Model compilation and execution time` is the runtime of the simulator on the host machine. `Total time for analytic` is the accumulated value of `analytic total` in the operator table. `[analytic] Execution time` is the target device latency predicted by the analytic model, and `TPS/Device` is calculated based on that latency.

##### Case 1: DeepSeek-V4-Flash Text-Only Incremental Prefill

**Case Objective:**

Simulate 8 text-only requests on 16 A3 dies with `TP=2`, `DP=8`, and `EP=16`, and analyze the execution mode, hot spot operators, memory, and bottlenecks of incremental or chunked prefill.

**Full Command:**

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

**Key Output** (truncated)

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

**Result Analysis:**

1. When `--decode` is not specified, `decode=False` and the CLI enters the prefill path. `--context-length 2048` indicates only the existing context and does not change the execution mode. In this case, the simulation continues to process 128 new tokens, which is incremental or chunked prefill, and the sequence length is 2048 + 128 = 2176. `DP=8` evenly distributes the 8 requests across 8 DP ranks, which is consistent with `Number of Queries per DP rank: 1`. `TP × DP × PP = 2 × 8 × 1 = 16`, and `EP=16` shards the routed experts.
2. `--compilation-config enable_dispatch_ffn_combine` explicitly enables the fused MoE FFN path. When locating hot spots, compare `analytic total` rather than the per-invocation `analytic avg`. The single fused `dispatch_ffn_combine` operator accounts for about `53.3%` of the total analytic time and is the primary analysis target in this example. Matrix multiplication and sparse attention are secondary hot spots.
3. `memory_bound: 99.19` is the dominant modeled bottleneck category, which indicates that the analytic model estimates that this example is mainly limited by memory access. `Memory available: 21.204 GB` is positive, which means that the device memory can accommodate this parallel configuration under the current device profile and simulation assumptions.

##### Case 2: Qwen3-VL Image-Text Prefill and Decode

**Case Objective:**

Simulate 8 first image-text prefill requests on 1 A3 die, where each request contains 32 text tokens and one 720 × 1080 image. Then simulate the single-token decode after prefill to verify the input shapes, the language and vision paths, memory, and bottlenecks.

**Full Command:**

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

**Key Output** (truncated)

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

**Result Analysis:**

1. When `--decode` is not specified and `--context-length 0` is set, the CLI performs the first prefill. The default `DP=1`. Therefore, all 8 requests are processed by the same DP rank, which is consistent with `Number of Queries per DP rank: 8`.
2. The processor resizes the image from 720 × 1080 to 704 × 1088. According to the input construction rules of the current processor, the internal `image_grid_thw` of each request is [1, 44, 68], which corresponds to 2992 patches. After a 2 × 2 spatial merge and the addition of two boundary tokens, 2992 / 4 + 2 = 750 image tokens are generated. Therefore, the model-side query and sequence length of each request are both 32 + 750 = 782, and the first dimension of the fused sequence for 8 requests is 8 × 782 = 6256.
3. The internal tensor shapes that the tool constructs from the image parameters are `pixel_values.shape=[23936, 1536]` and `image_grid_thw.shape=[8, 3]`, which are not CLI image input parameters. In the preceding operator table, the first dimensions 2992, 23936, and 6256 correspond to the vision sequence, the expanded image patches, and the fused multimodal sequence, respectively, and can be used to verify that derivation. When the model revision or processor configuration changes, rely on the actual operator shapes displayed by `--dump-input-shapes`.
4. `convolution` patch embedding and vision attention indicate that the image has entered the vision tower.
5. `compute_bound_mma: 74.40` is the dominant modeled bottleneck category. `Memory available: 45.821 GB` is positive, which means that the device memory can accommodate this configuration under the current device profile and simulation assumptions. `TPS/Device` is calculated only from the 32 text query tokens configured for each request and does not include image tokens. Therefore, it cannot be directly compared with text-only TPS.

**Decode Configuration:**

The following command simulates only the next-token decode after image-text prefill. The image tokens are already included in `--context-length`. Therefore, the image parameters are not required:

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

`--context-length 782` indicates that the KV cache already contains 32 + 750 = 782 tokens after image-text prefill, corresponding to text tokens and image tokens, respectively. Decode uses `--query-length 1` to process the current decoding token. Therefore, the sequence length is 782 + 1 = 783.

The image was already encoded into tokens and included in the KV cache during prefill. Decode executes only the language path and no longer constructs `pixel_values` or `image_grid_thw` or runs the vision tower. When you use `--log-level info` for troubleshooting, the no-image-input notice is expected. Do not set the image parameters at the same time. Otherwise, the image tokens are counted twice.

**Decode Key Output** (truncated)

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

The first dimension of the operator input is 8, which corresponds to 8 requests each processing 1 token. This configuration does not construct image tensors. Therefore, decode executes only the language path. To verify this, check the complete operator table and confirm that vision operators such as `convolution` and vision attention from the prefill stage no longer appear.

##### Differences Between Text-Only and VL Input Configurations

| Stage or Parameter | Text-Only Input | VL Image-Text Input |
| --- | --- | --- |
| Prefill `--query-length` | Indicates the number of text tokens in the current prefill. | `--query-length` indicates only the number of text tokens. The tool derives image tokens internally from the image parameters and adds them to the model-side prefill sequence length. |
| Prefill image parameters | Omit them. | Set `--image-batch-size`, `--image-height`, and `--image-width` together. |
| Decode `--context-length` | Indicates the number of existing text tokens in the KV cache. | Indicates the total number of existing text tokens and image tokens in the KV cache. |

#### FAQ

##### Why Does the Vision Path Remain Unused After I Set the Image Parameters?

The vision path runs only during the prefill stage of VL models. Check that `--decode` is not set and that `--image-batch-size`, `--image-height`, and `--image-width` are all provided. If any parameter is missing, a VL model treats the input as having no image, and a non-VL model ignores these parameters. The decode stage executes only the language path. You can add `--log-level info` to view the corresponding notice.

##### Why Does the Input Image Size Differ from the Operator Shape?

The image is resized according to the model processor configuration to a size that satisfies the patch and spatial merge requirements. For example, 720 × 1080 in this case is resized to 704 × 1088. When the model revision or processor configuration changes, rely on the actual operator shapes displayed by `--dump-input-shapes`.

##### What Do the Length Parameters Mean in Prefill and Decode?

In the prefill stage, do not set `--decode` (or specify `--prefill` explicitly). The first prefill usually sets `--context-length 0` and uses `--query-length` to indicate the number of tokens in the complete text prompt. For chunked or incremental prefill, `--context-length` indicates the number of existing model-side tokens in the KV cache, and `--query-length` indicates the number of new tokens in the current chunk.

In the decode stage, set `--decode` (in this case you cannot set `--prefill`, because the two are mutually exclusive). A normal single-token decode usually sets `--query-length 1`. When speculative decoding is enabled, the CLI aligns the decode `--query-length` to `N + 1`, where `N` is the number of speculative tokens. For MTP, you are advised to use the unified interface `--speculative-method mtp --num-speculative-tokens N`. The original `--num-mtp-tokens N` is retained only for backward compatibility. `--context-length` indicates the number of existing model-side tokens in the KV cache before this decode round. For VL models, this length must also include the image tokens generated during prefill.

##### How Do I Handle Insufficient Device Memory When the Command Succeeds?

`Memory available < 0` indicates that the target device memory cannot accommodate the current configuration. Adjust the model, the parallel strategy, `--num-queries`, `--image-batch-size`, or the sequence length.

##### Why Do the Subsequent Stage Tracks Look Empty in the PP Trace?

When you enable pipeline parallelism (`--pp-size > 1`) and export a Chrome Trace, each stage is arranged on the timeline in single-request serial execution order. Stage 0 events are at the start of the timeline, and subsequent stages follow in order. When the initial viewport of the trace viewer covers only the early part of the timeline, the tracks of subsequent stages exist, but their events are all to the right of the viewport. Therefore, they look like empty tracks. This is normal display behavior and does not mean that the stages did not run or that the export is incomplete. You can confirm this in the following ways:

- The `PP Overview` track in the trace provides a `PP Pipeline Total` slice that covers the complete pipeline time range, and shows the actual time ranges of each stage and of the inter-stage communication as nested slices.
- Each track name carries a time range suffix, for example `PP Stage 1 [22.221-42.063 ms]`.
- When the export completes, the output prints a summary of the time range layout of each stage.
- In the trace viewer, select Fit all or zoom out so that the time range covers the complete pipeline.

### 4.2 Video Generation

We provide the `video_generate.py` CLI to simulate the forward pass and performance of diffusion transformer models. This script supports simulating the inference of video generation models (for example, Stable Video Diffusion-like architectures) with configurable input dimensions, sampling steps, and parallelism settings. A detailed table summary of the operator performance breakdown is provided by default. You can also choose to export the performance timeline as a Chrome Trace file.

Its general usage is as follows:

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

The complete `--help` output also includes `--version/-V`, `--verbose/-v`, and `--quiet/-q`. The default `--log-level` is `error`. This tool does not provide `--debug` or `--log-file`. The model source is the positional parameter `model_id` or `--model-id`.

The main parameters are as follows:

| Parameter                 | Category | Optional/Required | Description |
| ------------------------  | --- | --- | --- |
| model_id                  | Positional Arguments | Required | Video generation model ID or local model path.<br>1. Type: Str.<br>2. Reference values: a Diffusers model directory, a remote repo ID, or a repo ID with subdirectories, which must contain `transformer/config.json` or a compatible transformer configuration.<br>3. Default: none.<br>4. You are advised to use a reviewed local absolute path. Remote model IDs do not provide security guarantees. |
| --device                  | Options | Optional | Specifies the device configuration used for the simulation.<br>1. Type: Str.<br>2. Reference values: any registered `DeviceProfile` name. See section 3, "Device Types", for the built-in values.<br>3. Default: `TEST_DEVICE`. |
| --batch-size              | Options | Required | Specifies the input batch size.<br>1. Type: Int.<br>2. Value range: positive integer.<br>3. Default: none. |
| --seq-len                 | Options | Required | Specifies the text sequence length.<br>1. Type: Int.<br>2. Value range: positive integer.<br>3. Default: none. |
| --chrome-trace-file       | Options | Optional | Specifies the Chrome trace JSON output path for exporting the performance timeline.<br>1. Type: Str.<br>2. Value range: file path.<br>3. Default: `None`. |
| --height                  | Options | Optional | Specifies the height of the input video or image frames.<br>1. Type: Int.<br>2. Value range: positive integer.<br>3. Default: 400. |
| --width                   | Options | Optional | Specifies the width of the input video or image frames.<br>1. Type: Int.<br>2. Value range: positive integer.<br>3. Default: 832. |
| --frame-num               | Options | Optional | Specifies the number of video frames.<br>1. Type: Int.<br>2. Value range: positive integer.<br>3. Default: 81. |
| --sample-step             | Options | Optional | Specifies the number of diffusion sampling steps.<br>1. Type: Int.<br>2. Value range: positive integer.<br>3. Default: 1. |
| --log-level               | Options | Optional | Specifies the log level.<br>1. Type: Str.<br>2. Reference values: `debug`, `info`, `warning`, `error`, `critical`.<br>3. Default: `error`. |
| --dtype                   | Options | Optional | Specifies the computation data type of the model.<br>1. Type: Str.<br>2. Reference values: `float16`, `float32`, `bfloat16`.<br>3. Default: `float16`. |
| --remote-source           | Options | Optional | Specifies the remote model source for a non-local Diffusers repo ID.<br>1. Type: Str.<br>2. Reference values: `huggingface`, `modelscope`.<br>3. Default: `huggingface`. |
| --quantize-linear-action  | Options | Optional | Specifies the quantization scheme for linear layers.<br>1. Type: Str.<br>2. Reference values: `DISABLED`, `W8A16_STATIC`, `W8A8_STATIC`, `W4A8_STATIC`, `W8A16_DYNAMIC`, `W8A8_DYNAMIC`, `W4A8_DYNAMIC`, `FP8`, `MXFP4`.<br>3. Default: `W8A8_DYNAMIC`.<br>4. Both uppercase and lowercase values and underscore or hyphen spellings are accepted, for example `W8A8_DYNAMIC` and `w8a8-dynamic`, or `DISABLED` and `disabled`. |
| --quantize-attention-action | Options | Optional | Specifies the quantization scheme for attention computation.<br>1. Type: Str.<br>2. Reference values: `DISABLED`, `INT8`, `FP8`.<br>3. Default: `DISABLED`.<br>4. Both uppercase and lowercase values and underscore or hyphen spellings are accepted, for example `DISABLED` and `disabled`, or `INT8` and `int8`. |
| --use-cfg                 | Options | Optional | Enables the simulation path related to classifier-free guidance.<br>1. Type: Bool.<br>2. Value range: on/off flag.<br>3. Default: `False`. |
| --attention-backend       | Attention Options | Optional | Selects the attention backend.<br>1. Type: Str.<br>2. Reference values: `dense`, `block_sparse_attention`.<br>3. Default: `dense`.<br>4. Both uppercase and lowercase values and underscore or hyphen spellings are accepted, for example `block_sparse_attention` and `block-sparse-attention`. |
| --attention-block-size    | Attention Options | Optional | Sets the BSA block size.<br>1. Type: Int.<br>2. Value range: positive integer.<br>3. Default: 128. |
| --attention-sparsity      | Attention Options | Optional | Sets the proportion of KV blocks that BSA skips.<br>1. Type: Float.<br>2. Value range: [0.0, 1.0).<br>3. Default: 0.0. |
| --num-devices             | Parallel Options | Optional | Specifies the total number of devices participating in the distributed simulation.<br>1. Type: Int.<br>2. Value range: positive integer.<br>3. Default: 1. 4. Must equal `--ulysses-size`. With `--cfg-parallel`, it must equal `2 * --ulysses-size`. |
| --ulysses-size            | Parallel Options | Optional | Specifies the Ulysses parallel size.<br>1. Type: Int.<br>2. Value range: positive integer.<br>3. Default: 1. |
| --cfg-parallel            | Parallel Options | Optional | Enables the CFG parallel strategy.<br>1. Type: Bool.<br>2. Value range: on/off flag.<br>3. Default: `False`. 4. Requires `--use-cfg`. With this option, `--num-devices` must equal `2 * --ulysses-size`. |
| --compile                 | Optimization Options | Optional | Compiles the main transformer before simulation.<br>1. Type: Bool.<br>2. Value range: on/off flag.<br>3. Default: `False`.<br>4. Uses `dynamic=False, fullgraph=True`. When DiT cache is enabled, the cache transformer uses the same strategy. |
| --compile-allow-graph-break | Optimization Options | Optional | Allows graph breaks during compilation.<br>1. Type: Bool.<br>2. Value range: on/off flag.<br>3. Default: `False`.<br>4. Changes the compilation mode of the main transformer and the DiT cache transformer to `fullgraph=False`. |
| --dit-cache               | Cache Options | Optional | Enables the DiT block cache.<br>1. Type: Bool.<br>2. Value range: on/off flag.<br>3. Default: `False`. |
| --cache-step-range        | Cache Options | Optional | Specifies the range of sampling steps for which the cache is enabled.<br>1. Type: Str.<br>2. Format: `start,end`, a closed interval.<br>3. Default: `None`.<br>4. Required when `--dit-cache` is set. |
| --cache-step-interval     | Cache Options | Optional | Specifies the step interval for cache updates.<br>1. Type: Int.<br>2. Value range: positive integer.<br>3. Default: 1, which means cache update reuse is disabled. |
| --cache-block-range       | Cache Options | Optional | Specifies the range of blocks for which the cache is enabled.<br>1. Type: Str.<br>2. Format: `start,end`, left-closed and right-open.<br>3. Default: `None`. |

BSA behavior and limitations:

- The BSA route accepts only inputs where Q, K, and V are all 4D with exactly the same shape. The mask must be `None` or a 4D bool tensor with the shape (`batch`, 1, `query_len`, `key_len`). Dropout must be 0, the attention must be non-causal, no custom scale may be provided, and GQA must be disabled.
- Unsupported calls use the dense fallback to support models that mix self-attention and cross-attention. Each SDPA context outputs only one aggregated warning summary. The stable fallback reason codes are `non_4d_qkv`, `qkv_shape_mismatch`, `unsupported_attention_mask`, `dropout`, `causal`, `custom_scale`, and `gqa`.
- BSA cannot be used together with attention quantization. The following complete command is invalid and reports an error before execution:

  ```bash
  python -m cli.inference.video_generate example-model --batch-size 1 --seq-len 128 --attention-backend block_sparse_attention --quantize-attention-action int8
  ```

  Linear quantization is not prohibited by this rule and can still be used independently.
- The dense backend accepts only the default values of BSA-only parameters (`--attention-block-size 128` and `--attention-sparsity 0.0`). Non-default BSA-only values are explicitly rejected and are not silently ignored.
- The estimator rounds up the number of query and KV blocks with `ceil`. The retained KV blocks are `max(1, ceil(KV_blocks * (1 - sparsity)))`. Boundary blocks are charged as full `block_size x block_size` padded tiles. Therefore, when the sequence length is not divisible, the estimate may be higher than token-exact dense attention even when sparsity is 0.0. Route generation scans the complete block-pair space. Memory uses an analytic approximation: Q, K, and V are each read completely once, and the K/V bytes are not scaled by sparsity.

> **Note:** To simulate the official original Tencent HunyuanVideo1.5 model, use `tencent/HunyuanVideo-1.5/transformer/<t2v_variant>` for Hugging Face and `Tencent-Hunyuan/HunyuanVideo-1.5/transformer/<t2v_variant>` for ModelScope. The supported T2V variants are `480p_t2v`, `480p_t2v_distilled`, and `720p_t2v`.

Run `python -m cli.inference.video_generate --help` for details.

### 4.3 Image Generation

We provide the `image_generate.py` CLI to simulate the diffusion transformer denoising workload and performance of image generation models. This script supports simulating the Transformer denoising inference of image generation models (for example, FLUX and Qwen-Image-Edit) with configurable batch, output image size, text condition length, and parallelism settings. A detailed table summary of the operator performance breakdown is provided by default. You can also choose to export the performance timeline as a Chrome Trace file.

This first release simulates only the Transformer denoising stage that enters the simulated device. Prompt encoding, VAE, the scheduler, and image I/O are excluded, and no real images are generated.

Its general usage is as follows:

```text
msmodeling inference image-generate MODEL --batch-size <N> --output-image-size HEIGHT WIDTH --text-seq-len <N>
```

The complete `--help` output also includes `--version/-V`, `--verbose/-v`, and `--quiet/-q`. The default `--log-level` is `error`. This tool does not provide `--debug` or `--log-file`. The model source is the positional parameter `model_id` or `--model-id`. `--num-devices` is the formal parallel size, and `--world-size` and `--chrome-trace` are hidden compatibility aliases.

The main parameters are as follows:

| Parameter                 | Category | Optional/Required | Description |
| ------------------------  | --- | --- | --- |
| model_id/--model-id     | Positional/Options | Required (choose one) | Image generation model ID or local model path. It can be a positional parameter or `--model-id`.<br>1. Type: Str.<br>2. Reference values: a Diffusers model directory or an exactly allowed remote repo ID, for example `black-forest-labs/FLUX.1-dev` or `Qwen/Qwen-Image-Edit`.<br>3. Default: none.<br>4. You are advised to use a reviewed local absolute path. Remote model IDs do not provide security guarantees. |
| --device                  | Options | Optional | Specifies the device configuration used for the simulation.<br>1. Type: Str.<br>2. Reference values: any registered `DeviceProfile` name. See section 3, "Device Types", for the built-in values.<br>3. Default: `TEST_DEVICE`. |
| --batch-size              | Options | Required | Specifies the batch size of the base workload, not the number of prompts or source images.<br>1. Type: Int.<br>2. Value range: positive integer.<br>3. Default: none. |
| --output-image-size       | Options | Required | Specifies the output image size. It appears exactly once and is used only to derive shapes. No image is output.<br>1. Type: Tuple[Int, Int] (`HEIGHT WIDTH`).<br>2. Value range: two positive integers.<br>3. Default: none. |
| --text-seq-len            | Options | Required | Length of the text condition that actually enters the Transformer.<br>1. Type: Int.<br>2. Value range: positive integer.<br>3. Default: none.<br>4. It is not a character count, a tokenizer input length, or a template length. This first release does not perform text encoding. |
| --source-image-size       | Options | Optional | Specifies the source image size. It can be passed repeatedly per source and accepts only sizes, not paths or pixels. It is available only for the editing kind.<br>1. Type: Tuple[Int, Int] (`HEIGHT WIDTH`).<br>2. Value range: two positive integers.<br>3. Default: none. |
| --sample-step             | Options | Optional | Specifies the number of times to execute the same Transformer workload iteration.<br>1. Type: Int.<br>2. Value range: positive integer.<br>3. Default: 1. |
| --use-cfg                 | Options | Optional | Enables the video-style classifier-free guidance workload approximation.<br>1. Type: Bool.<br>2. Value range: on/off flag.<br>3. Default: `False`. |
| --dtype                   | Options | Optional | Specifies the computation data type of the model.<br>1. Type: Str.<br>2. Reference values: `float16`, `float32`, `bfloat16`.<br>3. Default: `float16`. |
| --remote-source           | Options | Optional | Specifies the remote model source. It participates in exact pair matching.<br>1. Type: Str.<br>2. Reference values: `huggingface`, `modelscope`.<br>3. Default: `huggingface`. |
| --quantize-linear-action  | Quantization Options | Optional | Specifies the quantization scheme for linear layers.<br>1. Type: Str.<br>2. Reference values: `DISABLED`, `W8A16_STATIC`, `W8A8_STATIC`, `W4A8_STATIC`, `W8A16_DYNAMIC`, `W8A8_DYNAMIC`, `W4A8_DYNAMIC`, `FP8`, `MXFP4`.<br>3. Default: `DISABLED`.<br>4. Both uppercase and lowercase values and underscore or hyphen spellings are accepted, for example `W8A8_DYNAMIC` and `w8a8-dynamic`, or `DISABLED` and `disabled`. |
| --mxfp4-group-size        | Quantization Options | Optional | Specifies the group size for `MXFP4` quantization.<br>1. Type: Int.<br>2. Value range: positive integer.<br>3. Default: 32. |
| --quantize-attention-action | Quantization Options | Optional | Specifies the quantization scheme for attention computation.<br>1. Type: Str.<br>2. Reference values: `DISABLED`, `INT8`, `FP8`.<br>3. Default: `DISABLED`.<br>4. Both uppercase and lowercase values and underscore or hyphen spellings are accepted, for example `DISABLED` and `disabled`, or `INT8` and `int8`. |
| --compile                 | Optimization Options | Optional | Compiles the main transformer before simulation.<br>1. Type: Bool.<br>2. Value range: on/off flag.<br>3. Default: `False`.<br>4. Uses `dynamic=False, fullgraph=True`. When DiT cache is enabled, the cache transformer uses the same strategy. |
| --compile-allow-graph-break | Optimization Options | Optional | Allows graph breaks during compilation.<br>1. Type: Bool.<br>2. Value range: on/off flag.<br>3. Default: `False`.<br>4. Changes the compilation mode of the main transformer and the DiT cache transformer to `fullgraph=False`. |
| --num-devices             | Parallel Options | Optional | Specifies the total number of devices participating in the distributed simulation.<br>1. Type: Int.<br>2. Value range: positive integer.<br>3. Default: 1.<br>4. It must equal `--ulysses-size` (or `2 * --ulysses-size` when `--cfg-parallel` is enabled). The legacy name `--world-size` is still parsed. |
| --ulysses-size            | Parallel Options | Optional | Specifies the Ulysses parallel size.<br>1. Type: Int.<br>2. Value range: positive integer.<br>3. Default: 1. |
| --cfg-parallel            | Parallel Options | Optional | Enables the CFG parallel strategy.<br>1. Type: Bool.<br>2. Value range: on/off flag.<br>3. Default: `False`.<br>4. It is enabled only under `--use-cfg`, in which case `--num-devices` must equal `2 * --ulysses-size`. |
| --dit-cache               | Cache Options | Optional | Enables the DiT block cache.<br>1. Type: Bool.<br>2. Value range: on/off flag.<br>3. Default: `False`. |
| --cache-step-range        | Cache Options | Optional | Specifies the range of sampling steps for which the cache is enabled.<br>1. Type: Str.<br>2. Format: `start,end`, a closed interval.<br>3. Default: `None`.<br>4. Required when `--dit-cache` is set and `--cache-step-interval > 1`. |
| --cache-step-interval     | Cache Options | Optional | Specifies the step interval for cache updates.<br>1. Type: Int.<br>2. Value range: positive integer.<br>3. Default: 1, which means cache update reuse is disabled. |
| --cache-block-range       | Cache Options | Optional | Specifies the range of blocks for which the cache is enabled.<br>1. Type: Str.<br>2. Format: `start,end`, left-closed and right-open.<br>3. Default: `None`. |
| --chrome-trace-file       | Options | Optional | Specifies the Chrome trace JSON output path for exporting the performance timeline.<br>1. Type: Str.<br>2. Value range: file path.<br>3. Default: `None`.<br>4. It is generated only after a successful runtime. The legacy name `--chrome-trace` is still parsed. |

> **Note:** The denoising workload simulation currently supports FLUX.1-dev and Qwen-Image-Edit (`Qwen/Qwen-Image-Edit`, `Qwen/Qwen-Image-Edit-2509`, and `Qwen/Qwen-Image-Edit-2511`).
> **Limitation:** Qwen-Image-Edit currently does not support Ulysses sequence parallelism (`--ulysses-size > 1`), and the corresponding input sharding is not yet implemented. When you pass `--ulysses-size` greater than 1, the simulation fails with a Qwen-specific error before the runtime and does not fall back to a non-parallel path.

Run `python -m cli.inference.image_generate --help` for details.
