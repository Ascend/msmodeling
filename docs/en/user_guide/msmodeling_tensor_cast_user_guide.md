# TensorCast

## Supported Matrix

**Core capabilities**

| Area | Support | Notes |
| --- | --- | --- |
| Runtime output | Supported | Perf summary, Chrome trace |
| Device modeling | Supported | Interconnect modeling |
| Device profiles | Supported | Custom device profiles (user-defined) |
| Perf model | Supported | Empirical model, analytic model |

**Models & optimization**

| Area | Support | Notes |
| --- | --- | --- |
| Text models (families) | Supported | Qwen3, Qwen3-Next, GLM-4, DeepSeek V3, DeepSeek V3.2, ERNIE 4.5, Ling, MiMo v2, MinMax M2, MoE supported |
| Vision-language models | Supported | Qwen3-VL, GLM-4V, InternVL |
| Video generation models (Diffusers DiT) | Supported | Wan, HunyuanVideo, HunyuanVideo1.5 |
| Auto sharding | Supported | DP, TP, EP, automatic parallel strategy search |
| Quantization (linear) | Supported | W8A16/W8A8/W4A8 (static & dynamic), FP8, MXFP4 |
| Quantization (attention) | Supported (text only) | INT8 |
| Serving simulation | Supported | Multi-instance multi-request end-to-end serving simulation, outputs TTFT/TPOT/throughput |
| Throughput optimization | Supported | Automatic configuration search under SLO constraints with cross-hardware comparison tables |

## Introduction

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

## At a Glance

### Quick Start: Text Generation

**What it does:** Simulate LLM inference performance for a batch of queries.

**Command:**

```bash
python -m cli.inference.text_generate Qwen/Qwen3-32B --num-queries 2 --query-length 3500 --device TEST_DEVICE
```

**Key flags:** `--context-length`, `--decode`, `--quantize-linear-action`, `--chrome-trace`, `--device`

**Output:** A performance summary table; optionally a Chrome trace file if `--chrome-trace` is set.

### Result (Text Generation)

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

### Quick Start: Video Generation

**What it does:** Simulate diffusion transformer forward pass for video generation models.

**Command:**

```bash
python -m cli.inference.video_generate docs/fixtures/hunyuanvideo_mock_model --batch-size 1 --seq-len 16 --height 576 --width 1024 --frame-num 14 --sample-step 25 --device TEST_DEVICE
```

**Key flags:** `--height`, `--width`, `--frame-num`, `--sample-step`, `--chrome-trace`, `--device`

**Output:** A performance summary table; optionally a Chrome trace file if `--chrome-trace` is set.

### Result (Video Generation)

Example output (truncated):

```text
Model compilation and execution time: 25.44349410000723s
----------------------------------------------  --------------  ------------  ----------
                     Name                       analytic total  analytic avg  # of Calls
----------------------------------------------  --------------  ------------  ----------
aten.addmm.default                                      8.546s       1.280ms        6675
tensor_cast.attention.default                           7.943s       5.125ms        1550
aten.mul.Tensor                                         2.597s     126.510us       20525
aten._to_copy.default                                   2.450s     142.242us       17225
tensor_cast.static_quant_linear.default                 2.266s     323.720us        7000
...
Total time for analytic: 29.350s
```

Note: `Model compilation and execution time` is the simulator's runtime on the host, not the real model compile or execution time on hardware.

Metric descriptions:

- `analytic total`: Estimated total time spent by the operator.
- `analytic avg`: Average time per operator call.
- `# of Calls`: Number of times the operator is invoked.
- `Total time for analytic`: Sum of analytic operator time.

## Supported Accelerators

We provide built-in support for the following device profiles (defined in `tensor_cast/device.py`):

- `TEST_DEVICE`
- `ATLAS_800_A2_376T_64G`
- `ATLAS_800_A2_313T_64G`
- `ATLAS_800_A2_280T_64G`
- `ATLAS_800_A2_280T_64G_PCIE`
- `ATLAS_800_A2_280T_32G_PCIE`
- `ATLAS_800_A3_752T_128G_DIE`
- `ATLAS_800_A3_560T_128G_DIE`

### Custom device types

For other hardware, define a custom device profile as a Python file under `tensor_cast/device_profiles`. TensorCast will load it automatically, and you can then reference the profile name from the CLI. Custom device guide: [device_profiles/README.md](../../../tensor_cast/device_profiles/README.md)

## Detailed Usage

### Run text generation with given query length

We provide a `text_generate.py` command line interface to simulate the text generation. The script supports text generation with a batch of queries with the same input length and optionally same context length. The table summary of op performance breakdown is provided by default. An option is also provided to dump the chrome trace.

Its general usage is shown below:

```text
usage: text_generate.py [-h]
                        [--device {TEST_DEVICE,ATLAS_800_A2_376T_64G,ATLAS_800_A2_313T_64G,ATLAS_800_A2_280T_64G,ATLAS_800_A2_280T_64G_PCIE,ATLAS_800_A2_280T_32G_PCIE,ATLAS_800_A3_752T_128G_DIE,ATLAS_800_A3_560T_128G_DIE}]
                        [--num-devices NUM_DEVICES] [--reserved-memory-gb RESERVED_MEMORY_GB] [--log-level {debug,info,warning,error,critical}] --num-queries NUM_QUERIES --query-length QUERY_LENGTH
                        [--context-length CONTEXT_LENGTH] [--decode] [--num-mtp-tokens NUM_MTP_TOKENS] [--disable-repetition] [--compile] [--compile-allow-graph-break]
                        [--quantize-linear-action {DISABLED,W8A16_STATIC,W8A8_STATIC,W4A8_STATIC,W8A16_DYNAMIC,W8A8_DYNAMIC,W4A8_DYNAMIC,FP8,MXFP4}] [--quantize-lmhead] [--mxfp4-group-size MXFP4_GROUP_SIZE]
                        [--quantize-attention-action {DISABLED,INT8,FP8}] [--graph-log-url GRAPH_LOG_URL] [--dump-input-shapes] [--chrome-trace CHROME_TRACE]
                        [--num-hidden-layers-override NUM_HIDDEN_LAYERS_OVERRIDE] [--tp-size TP_SIZE] [--dp-size DP_SIZE] [--ep-size EP_SIZE] [--o-proj-tp-size O_PROJ_TP_SIZE]
                        [--o-proj-dp-size O_PROJ_DP_SIZE] [--mlp-tp-size MLP_TP_SIZE] [--mlp-dp-size MLP_DP_SIZE] [--lmhead-tp-size LMHEAD_TP_SIZE] [--lmhead-dp-size LMHEAD_DP_SIZE]
                        [--moe-tp-size MOE_TP_SIZE] [--moe-dp-size MOE_DP_SIZE] [--word-embedding-tp] [--enable-redundant-experts] [--enable-external-shared-experts] [--host-external-shared-experts]
                        [--image-batch-size IMAGE_BATCH_SIZE] [--image-height IMAGE_HEIGHT] [--image-width IMAGE_WIDTH] [--remote-source {huggingface,modelscope}]
                        model_id

Run a simulated LLM inference pass and dump the perf result.
```

Run `python -m cli.inference.text_generate --help` for details.

### Run video generation inference for diffusion models

We provide a `video_generate.py` command line interface to simulate the forward pass and performance of diffusion transformer models. The script supports simulating the inference process of video generation models (e.g., Stable Video Diffusion-like architectures) with configurable input dimensions, sampling steps, and parallelism settings. A detailed table summary of operator performance breakdown is provided by default. An option is also provided to dump the performance timeline as a Chrome Trace file.

Its general usage is shown below:

```text
usage: video_generate.py [-h]
                         [--device {TEST_DEVICE,ATLAS_800_A2_376T_64G,ATLAS_800_A2_313T_64G,ATLAS_800_A2_280T_64G,ATLAS_800_A2_280T_64G_PCIE,ATLAS_800_A2_280T_32G_PCIE,ATLAS_800_A3_752T_128G_DIE,ATLAS_800_A3_560T_128G_DIE}]
                         --batch-size BATCH_SIZE --seq-len SEQ_LEN [--chrome-trace CHROME_TRACE] [--height HEIGHT] [--width WIDTH] [--frame-num FRAME_NUM] [--sample-step SAMPLE_STEP]
                         [--log-level {debug,info,warning,error,critical}] [--dtype {float16,float32,bfloat16}]
                         [--quantize-linear-action {DISABLED,W8A16_STATIC,W8A8_STATIC,W4A8_STATIC,W8A16_DYNAMIC,W8A8_DYNAMIC,W4A8_DYNAMIC,FP8,MXFP4}] [--use-cfg] [--world-size WORLD_SIZE]
                         [--ulysses-size ULYSSES_SIZE] [--cfg-parallel] [--dit-cache] [--cache-step-range CACHE_STEP_RANGE] [--cache-step-interval CACHE_STEP_INTERVAL]
                         [--cache-block-range CACHE_BLOCK_RANGE]
                         model_id

Run a simulated diffusion transformer forward and dump perf stats.
```

Run `python -m cli.inference.video_generate --help` for details.

## Advanced Notes

### External Shared Experts & Redundant Experts Implementation

The following outlines the implementation logic for External Shared Experts and Redundant Experts.

1. Redundant Experts Only:
Each device will host an additional redundant expert.

2. External Shared Experts Only:
Devices are allocated between external shared experts and routing experts at a ratio of 1:`top_k`. Redundant experts are used to pad routing experts if needed.
For example, if `world_size` is 64, `top_k` is 8, and number of routing experts is 256, 8 devices are assigned to host external shared experts.
The remaining 56 devices are used to distribute 256 routing experts. 32 devices host 5 routing experts each. 24 devices host 4 routing experts and 1 redundant expert.

3. Both External Shared Experts & Redundant Experts Enabled:
The allocation logic is identical to the "External Shared Experts Only" mode, with one addition: If no redundant experts are needed to pad routing experts (i.e., routing experts are evenly distributed across devices), each device hosting routing experts will host an additional redundant expert.

### Run Prefill

To run a prefill of Qwen3-32B with two requests with 3500-token input length each on A2. You can run the following command:

```bash
python -m cli.inference.text_generate Qwen/Qwen3-32B --num-queries 2 --query-length 3500 --context-length 3500 --device TEST_DEVICE
```

In prefill mode, omit `--decode`; `--query-length` is the new input length and `--context-length` sets the context length for each query.

You can also quantize the linear with various quantization schemes, such as W8A8 dynamic quantization and with 4500-token context as the prefix:

```bash
python -m cli.inference.text_generate Qwen/Qwen3-32B --num-queries 2 --query-length 3500 --context-length 4500 --device TEST_DEVICE --quantize-linear-action W8A8_DYNAMIC
```

### Run Decode

Running decode is similar by tweaking the input length and context length. Usually, the input length is 1.

```bash
python -m cli.inference.text_generate Qwen/Qwen3-32B --num-queries 10 --query-length 1 --context-length 4500 --device TEST_DEVICE --quantize-linear-action W8A8_STATIC
```

## TODO List (Roadmap)

- [ ] Models (planned, compilable): Kimi-K2, Qwen3-235B, GLM-4.5
- [ ] Auto sharding: CP, SP
- [ ] Quantization (attention): FP8
- [ ] Compiler: Complete fusion support for models (Qwen3 Dense, DeepSeek)
