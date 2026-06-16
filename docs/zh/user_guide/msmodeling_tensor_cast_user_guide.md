# TensorCast 使用指南

## 支持矩阵

**核心能力**

| 领域 | 支持情况 | 说明 |
| --- | --- | --- |
| 运行时输出 | 支持 | 性能汇总、Chrome trace |
| 设备建模 | 支持 | 互连建模 |
| 设备画像 | 支持 | 自定义 DeviceProfile（用户定义） |
| 性能模型 | 支持 | 经验模型、解析模型 |

**模型与优化**

| 领域 | 支持情况 | 说明 |
| --- | --- | --- |
| 文本模型（系列） | 支持 | Qwen3、Qwen3-Next、GLM-4、DeepSeek V3、DeepSeek V3.2、ERNIE 4.5、Ling、MiMo v2、MiniMax M2，支持 MoE |
| 视觉-语言模型 | 支持 | Qwen3-VL、GLM-4V、InternVL |
| 视频生成模型（Diffusers DiT） | 支持 | Wan、HunyuanVideo、HunyuanVideo1.5 |
| 自动分片 | 支持 | DP、TP、EP，自动并行策略搜索 |
| 量化（Linear） | 支持 | W8A16/W8A8/W4A8（静态与动态）、FP8、MXFP4 |
| 量化（Attention） | 支持（仅文本） | INT8 |
| 服务仿真 | 支持 | 多实例、多请求端到端服务仿真，输出 TTFT/TPOT/吞吐 |
| 吞吐优化 | 支持 | 在 SLO 约束下自动搜索配置，并支持跨硬件对比表 |

完整模型列表与特性明细请参见《[模型支持与特性支持矩阵](./support_matrix/support_matrix_user_guide.md)》。

## 阅读路径

| 目标 | 推荐章节 |
| --- | --- |
| 快速跑通 LLM 文本生成仿真 | [快速入门：文本生成](#快速入门文本生成) |
| 理解输出中的耗时、调用次数和显存指标 | [结果（文本生成）](#结果文本生成) |
| 运行视频生成模型仿真 | [快速入门：视频生成](#快速入门视频生成) |
| 查看或自定义硬件设备画像 | [支持的加速器](#支持的加速器) |
| 配置 prefill / decode 场景 | [运行 Prefill](#运行-prefill) 与 [运行 Decode](#运行-decode) |

## 简介

TensorCast 是一个面向 PyTorch 程序的性能仿真与分析框架。它使开发者和研究人员能够在无需访问物理机器的情况下，预测其神经网络模型在特定硬件配置上的性能表现。

从本质上讲，TensorCast 充当"虚拟机"或运行时仿真器。它不会在实际加速器上执行计算，而是拦截 PyTorch 程序的计算图，并在用户定义的 MachineConfig 上仿真其执行过程。该配置指定了目标硬件的特性，例如理论算力（TFLOPS）、内存带宽、缓存层次结构以及互连速度。为了准确估算模型在给定硬件上的最优性能，TensorCast 提供了模型优化流水线，包括自动模型分片、量化以及 FX 图优化，在开展分析前将源程序转换为最优形式。

通过在"虚拟"硬件上运行模型，TensorCast 提供详细的性能洞察，包括：

- 开箱即用的 Huggingface transformer 模型支持。
- 通过简单配置支持多种硬件加速器设备。
- 算子级执行时间：使用可扩展模型进行估算，例如解析 roofline 模型、经验数据或基于机器学习的预测器。
- 内存占用：跟踪总内存分配量和峰值内存分配量。
- 计算特征：分析每个算子的 FLOPs（浮点运算次数）和内存访问量。
- 高级调度仿真：建模复杂执行模式，例如跨多个 stream 的并发计算。

最终输出包括全面的汇总表和详细的 Chrome Trace 文件，便于深入可视化并识别性能瓶颈。

首次使用前，请先参阅《[msModeling 安装指南](../install_guide/msmodeling_install_guide.md)》完成环境搭建与第一次仿真。

## 概览

### 快速入门：文本生成

**功能说明：** 仿真一批 query 的 LLM 推理性能。

**命令：**

```bash
python -m cli.inference.text_generate Qwen/Qwen3-32B --num-queries 2 --query-length 3500 --device TEST_DEVICE
```

**关键参数：** `--context-length`、`--decode`、`--quantize-linear-action`、`--chrome-trace`、`--device`

**输出：** 性能汇总表；若设置了 `--chrome-trace`，还可选输出 Chrome trace 文件。

### 结果（文本生成）

示例输出（已截断）：

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

说明：`Model compilation and execution time` 是仿真器在宿主机上的运行时间，而非模型在硬件上的真实编译或执行时间。

指标说明：

- `analytic total`：算子估算总耗时。
- `analytic avg`：算子每次调用的平均耗时。
- `# of Calls`：算子被调用的次数。
- `Total time for analytic`：解析算子耗时的总和。
- `TPS/Device`：每设备每秒 token 数。
- `Total device memory` 及其细分项：权重、KV cache 和 activation 的估算内存占用。

### 快速入门：视频生成

**功能说明：** 仿真视频生成模型的 diffusion transformer 前向传播。

**命令：**

```bash
python -m cli.inference.video_generate docs/fixtures/hunyuanvideo_mock_model --batch-size 1 --seq-len 16 --height 576 --width 1024 --frame-num 14 --sample-step 25 --device TEST_DEVICE
```

**关键参数：** `--height`、`--width`、`--frame-num`、`--sample-step`、`--chrome-trace`、`--device`

**输出：** 性能汇总表；若设置了 `--chrome-trace`，还可选输出 Chrome trace 文件。

### 结果（视频生成）

示例输出（已截断）：

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

说明：`Model compilation and execution time` 是仿真器在宿主机上的运行时间，而非模型在硬件上的真实编译或执行时间。

指标说明：

- `analytic total`：算子估算总耗时。
- `analytic avg`：算子每次调用的平均耗时。
- `# of Calls`：算子被调用的次数。
- `Total time for analytic`：解析算子耗时的总和。

## 支持的加速器

我们为以下设备配置提供内置支持（定义于 `tensor_cast/device.py`）：

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

### 自定义设备类型

对于其他硬件，可在 `tensor_cast/device_profiles` 下以 Python 文件形式定义自定义设备配置。TensorCast 会自动加载，之后即可在 CLI 中引用该配置名称。自定义设备指南：[device_profiles/README.md](../../../tensor_cast/device_profiles/README.md)

## 详细用法

### 按指定 query 长度运行文本生成

我们提供 `text_generate.py` 命令行接口来仿真文本生成。该脚本支持对一批具有相同输入长度、可选相同 context 长度的 query 进行文本生成仿真。默认提供算子性能分解的表格汇总。也可选择导出 Chrome trace。

其一般用法如下：

```text
usage: text_generate.py [-h]
                        [--device {TEST_DEVICE,ATLAS_800_A2_376T_64G,ATLAS_800_A2_313T_64G,ATLAS_800_A2_280T_64G,ATLAS_800_A2_280T_64G_PCIE,ATLAS_800_A2_280T_32G_PCIE,ATLAS_800_A3_752T_128G_DIE,ATLAS_800_A3_560T_128G_DIE,ATLAS_800_A3_560T_128G_DIE_ROCE,ATLAS_350_425T_112G,ATLAS_350_425T_84G}]
                        [--num-devices NUM_DEVICES] [--reserved-memory-gb RESERVED_MEMORY_GB] [--log-level {debug,info,warning,error,critical}] --num-queries NUM_QUERIES --query-length QUERY_LENGTH
                        [--context-length CONTEXT_LENGTH] [--decode] [--num-mtp-tokens NUM_MTP_TOKENS] [--disable-repetition] [--compile] [--compile-allow-graph-break]
                        [--quantize-linear-action {DISABLED,W8A16_STATIC,W8A8_STATIC,W4A8_STATIC,W8A16_DYNAMIC,W8A8_DYNAMIC,W4A8_DYNAMIC,FP8,MXFP4}] [--quantize-lmhead] [--mxfp4-group-size MXFP4_GROUP_SIZE]
                        [--quantize-attention-action {DISABLED,INT8,FP8}] [--graph-log-url GRAPH_LOG_URL] [--dump-input-shapes] [--chrome-trace CHROME_TRACE]
                        [--num-hidden-layers-override NUM_HIDDEN_LAYERS_OVERRIDE] [--tp-size TP_SIZE] [--dp-size DP_SIZE] [--ep-size EP_SIZE] [--o-proj-tp-size O_PROJ_TP_SIZE]
                        [--o-proj-dp-size O_PROJ_DP_SIZE] [--mlp-tp-size MLP_TP_SIZE] [--mlp-dp-size MLP_DP_SIZE] [--lmhead-tp-size LMHEAD_TP_SIZE] [--lmhead-dp-size LMHEAD_DP_SIZE]
                        [--moe-tp-size MOE_TP_SIZE] [--moe-dp-size MOE_DP_SIZE] [--word-embedding-tp {col,row}] [--enable-redundant-experts] [--enable-external-shared-experts] [--host-external-shared-experts]
                        [--image-batch-size IMAGE_BATCH_SIZE] [--image-height IMAGE_HEIGHT] [--image-width IMAGE_WIDTH] [--remote-source {huggingface,modelscope}]
                        model_id

Run a simulated LLM inference pass and dump the perf result.
```

运行 `python -m cli.inference.text_generate --help` 查看详情。

### 运行 diffusion 模型的视频生成推理

我们提供 `video_generate.py` 命令行接口来仿真 diffusion transformer 模型的前向传播与性能。该脚本支持仿真视频生成模型（例如类 Stable Video Diffusion 架构）的推理过程，可配置输入维度、采样步数及并行设置。默认提供详细的算子性能分解表格汇总。也可选择将性能时间线导出为 Chrome Trace 文件。

其一般用法如下：

```text
usage: video_generate.py [-h]
                         [--device {TEST_DEVICE,ATLAS_800_A2_376T_64G,ATLAS_800_A2_313T_64G,ATLAS_800_A2_280T_64G,ATLAS_800_A2_280T_64G_PCIE,ATLAS_800_A2_280T_32G_PCIE,ATLAS_800_A3_752T_128G_DIE,ATLAS_800_A3_560T_128G_DIE,ATLAS_800_A3_560T_128G_DIE_ROCE,ATLAS_350_425T_112G,ATLAS_350_425T_84G}]
                         --batch-size BATCH_SIZE --seq-len SEQ_LEN [--chrome-trace CHROME_TRACE] [--height HEIGHT] [--width WIDTH] [--frame-num FRAME_NUM] [--sample-step SAMPLE_STEP]
                         [--log-level {debug,info,warning,error,critical}] [--dtype {float16,float32,bfloat16}]
                         [--quantize-linear-action {DISABLED,W8A16_STATIC,W8A8_STATIC,W4A8_STATIC,W8A16_DYNAMIC,W8A8_DYNAMIC,W4A8_DYNAMIC,FP8,MXFP4}] [--use-cfg] [--world-size WORLD_SIZE]
                         [--ulysses-size ULYSSES_SIZE] [--cfg-parallel] [--dit-cache] [--cache-step-range CACHE_STEP_RANGE] [--cache-step-interval CACHE_STEP_INTERVAL]
                         [--cache-block-range CACHE_BLOCK_RANGE]
                         model_id

Run a simulated diffusion transformer forward and dump perf stats.
```

运行 `python -m cli.inference.video_generate --help` 查看详情。

## 进阶说明

### External Shared Experts 与 Redundant Experts 实现

以下概述 External Shared Experts 与 Redundant Experts 的实现逻辑。

1. 仅 Redundant Experts：
每个设备会额外托管一个 redundant expert。

2. 仅 External Shared Experts：
设备按 1:`top_k` 的比例分配给 external shared experts 与 routing experts。如有需要，使用 redundant experts 对 routing experts 进行填充。
例如，若 `world_size` 为 64、`top_k` 为 8、routing experts 数量为 256，则有 8 个设备用于托管 external shared experts。
其余 56 个设备用于分布 256 个 routing experts。32 个设备各托管 5 个 routing experts。24 个设备各托管 4 个 routing experts 和 1 个 redundant expert。

3. 同时启用 External Shared Experts 与 Redundant Experts：
分配逻辑与"仅 External Shared Experts"模式相同，另有一项补充：若无需 redundant experts 来填充 routing experts（即 routing experts 在各设备间均匀分布），则每个托管 routing experts 的设备会额外托管一个 redundant expert。

### 运行 Prefill

要在 A2 上对 Qwen3-32B 运行 prefill，两个请求各 3500 token 输入长度，可运行以下命令：

```bash
python -m cli.inference.text_generate Qwen/Qwen3-32B --num-queries 2 --query-length 3500 --context-length 3500 --device TEST_DEVICE
```

Prefill 模式下不添加 `--decode`；`--query-length` 表示新输入长度，`--context-length` 表示每个请求的 context 长度。

也可使用多种量化方案对线性层进行量化，例如 W8A8 动态量化，并以 4500 token 的 context 作为前缀：

```bash
python -m cli.inference.text_generate Qwen/Qwen3-32B --num-queries 2 --query-length 3500 --context-length 4500 --device TEST_DEVICE --quantize-linear-action W8A8_DYNAMIC
```

### 运行 Decode

运行 decode 类似，只需调整输入长度和 context 长度。通常输入长度为 1。

```bash
python -m cli.inference.text_generate Qwen/Qwen3-32B --num-queries 10 --query-length 1 --context-length 4500 --device TEST_DEVICE --quantize-linear-action W8A8_STATIC
```

## 待办列表（Roadmap）

- [ ] 模型（规划中，可编译）：Kimi-K2、Qwen3-235B、GLM-4.5
- [ ] 自动分片：CP、SP
- [ ] 量化（Attention）：FP8
- [ ] 编译器：完善 Qwen3 Dense、DeepSeek 等模型的融合支持
