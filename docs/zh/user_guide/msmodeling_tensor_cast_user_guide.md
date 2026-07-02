# TensorCast 使用指南

完整模型列表与特性明细请参见《[模型支持与特性支持矩阵](./support_matrix/support_matrix_user_guide.md)》。

## 阅读路径

| 目标 | 推荐章节 |
| --- | --- |
| 1. 快速跑通 LLM 文本生成仿真 | [2.1 快速入门：文本生成](#21-快速入门文本生成) |
| 2. 理解输出中的耗时、调用次数和显存指标 | [2.2 结果（文本生成）](#22-结果文本生成) |
| 3. 运行视频生成模型仿真 | [2.3 快速入门：视频生成](#23-快速入门视频生成) |
| 4. 查看或自定义硬件设备画像 | [3 支持的设备与自定义设备](#3-支持的设备与自定义设备) |

## 1 简介

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

首次使用前，请先参阅《[快速上手：环境搭建与第一次仿真](../install_guide/msmodeling_install_guide.md)》完成环境搭建并运行一次 LLM 推理仿真。

## 2 概览

### 2.1 快速入门：文本生成

**功能说明：** 仿真一批 query 的 LLM 推理性能。

#### Prefill 场景

要在 TEST_DEVICE 上对 Qwen3-32B 运行 prefill，两个请求各 3500 token 输入长度，可运行以下命令：

```bash
python -m cli.inference.text_generate Qwen/Qwen3-32B --num-queries 2 --query-length 3500 --context-length 3500 --device TEST_DEVICE --compile
```

Prefill 模式下不添加 `--decode`；`--query-length` 表示新输入长度，`--context-length` 表示每个请求的 context 长度。

也可使用多种量化方案对线性层进行量化，例如 W8A8 动态量化，并以 4500 token 的 context 作为前缀：

```bash
python -m cli.inference.text_generate Qwen/Qwen3-32B --num-queries 2 --query-length 3500 --context-length 4500 --device TEST_DEVICE --quantize-linear-action W8A8_DYNAMIC --compile
```

#### Decode 场景

Decode 场景的运行方式类似，仅需调整输入长度 `--query-length` 和请求的 context 长度 `--context-length`。未启用 MTP 时，`--query-length` 通常为 `1`；启用 `--num-mtp-tokens` 时，`--query-length` 应设置为 `1 + --num-mtp-tokens`。

```bash
python -m cli.inference.text_generate Qwen/Qwen3-32B --num-queries 10 --query-length 1 --context-length 4500 --decode --device TEST_DEVICE --quantize-linear-action W8A8_STATIC --compile
```

**输出：** 性能汇总表；若设置了 `--chrome-trace`，还可选输出 Chrome trace 文件。

### 2.2 结果（文本生成）

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

### 2.3 快速入门：视频生成

**功能说明：** 仿真视频生成模型的 diffusion transformer 前向传播。

**命令：**

```bash
python -m cli.inference.video_generate docs/fixtures/hunyuanvideo_mock_model --batch-size 1 --seq-len 16 --height 576 --width 1024 --frame-num 14 --sample-step 25 --device TEST_DEVICE
```

**关键参数：** `--height`、`--width`、`--frame-num`、`--sample-step`、`--chrome-trace`、`--device`

**输出：** 性能汇总表；若设置了 `--chrome-trace`，还可选输出 Chrome trace 文件。

### 2.4 结果（视频生成）

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

## 3 支持的设备与自定义设备

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

### 3.1 自定义设备类型

对于其他硬件，可在 `tensor_cast/device_profiles` 下以 Python 文件形式定义自定义设备配置。TensorCast 会自动加载，之后即可在 CLI 中引用该配置名称。自定义设备指南：[device_profiles/README.md](../../../tensor_cast/device_profiles/README.md)

## 4 详细用法

推荐使用本地安全模式：先下载并审核模型仓库，然后将 `model_id` 填为完整本地绝对路径，例如
`/data/models/Qwen3-32B`。本地路径加载会在运行前校验路径属主、软链接和权限；不建议使用软链接目录、
共享可写目录或未审核来源中的模型文件。

工具仍支持直接传入 Hugging Face 或 ModelScope 的 model id，例如 `Qwen/Qwen3-32B`，并可通过
`--remote-source` 选择来源。该 model id 模式可能在 `trust_remote_code=True` fallback 时执行远端
Python 代码，msmodeling 不对远端代码安全性做保证；运行时会打印 `trust_remote_code` 风险提示。

### 4.1 文本生成

我们提供 `text_generate.py` 命令行接口来仿真文本生成。该脚本支持对一批具有相同输入长度、可选相同 context 长度的 query 进行文本生成仿真。默认提供算子性能分解的表格汇总。也可选择导出 Chrome trace。

其一般用法如下：

```text
usage: text_generate.py [-h]
                        [--device {TEST_DEVICE,ATLAS_800_A2_376T_64G,ATLAS_800_A2_313T_64G,ATLAS_800_A2_280T_64G,ATLAS_800_A2_280T_64G_PCIE,ATLAS_800_A2_280T_32G_PCIE,ATLAS_800_A3_752T_128G_DIE,ATLAS_800_A3_560T_128G_DIE,ATLAS_800_A3_560T_128G_DIE_ROCE,ATLAS_350_425T_112G,ATLAS_350_425T_84G}]
                        [--num-devices NUM_DEVICES] [--reserved-memory-gb RESERVED_MEMORY_GB]
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

主要参数说明如下：

| 参数名称 | 分类 | 可选/必选 | 参数说明 |
| --- | --- | --- | --- |
| `model_id` | General Options | 必选 | 模型 ID 或本地模型路径。<br>1. 类型：Str。<br>2. 参考值：Hugging Face ID、ModelScope ID 或本地绝对路径，例如 `Qwen/Qwen3-32B` 或 `/data/models/Qwen3-32B`。<br>3. 默认值：无。<br>4. 使用远端模型 ID 时，可能通过 `trust_remote_code=True` 执行远端代码。 |
| `--device` | General Options | 可选 | 指定用于仿真的设备配置。<br>1. 类型：Str。<br>2. 参考值：已注册 `DeviceProfile` 名称，包括 `TEST_DEVICE`、`ATLAS_800_A2_376T_64G`、`ATLAS_800_A2_313T_64G`、`ATLAS_800_A2_280T_64G`、`ATLAS_800_A2_280T_64G_PCIE`、`ATLAS_800_A2_280T_32G_PCIE`、`ATLAS_800_A3_752T_128G_DIE`、`ATLAS_800_A3_560T_128G_DIE`、`ATLAS_800_A3_560T_128G_DIE_ROCE`、`ATLAS_350_425T_112G`、`ATLAS_350_425T_84G`。<br>3. 默认值：`TEST_DEVICE`。 |
| `--num-devices` | General Options | 可选 | 指定参与仿真的设备数量。<br>1. 类型：Int。<br>2. 取值范围：正整数。<br>3. 默认值：`1`。 |
| `--reserved-memory-gb` | General Options | 可选 | 指定每张设备预留给系统使用的显存大小，单位为 GB。<br>1. 类型：Float。<br>2. 取值范围：非负数；设置为 `0` 表示不预留系统显存。<br>3. 默认值：`0.0`。 |
| `--log-level` | General Options | 可选 | 指定日志级别。<br>1. 类型：Str。<br>2. 参考值：`debug`、`info`、`warning`、`error`、`critical`。<br>3. 默认值：`error`。 |
| `--num-queries` | LLM Options | 必选 | 本次仿真的 query 数量。<br>1. 类型：Int。<br>2. 取值范围：正整数。<br>3. 默认值：无。 |
| `--query-length` | LLM Options | 必选 | 每个 query 的新输入 token 长度。<br>1. 类型：Int。<br>2. 取值范围：正整数。<br>3. 默认值：无。 |
| `--context-length` | LLM Options | 可选 | 每个 query 的已有上下文 token 长度。<br>1. 类型：Int。<br>2. 取值范围：非负整数。<br>3. 默认值：`0`。 |
| `--decode` | LLM Options | 可选 | 启用自回归 decode 模式；不设置时按 prefill 模式运行。<br>1. 类型：Bool。<br>2. 取值范围：开关参数。<br>3. 默认值：`False`。 |
| `--prefix-cache-hit-rate` | LLM Options | 可选 | 指定 prefix cache 命中率，用于 prefill token 复用近似。<br>1. 类型：Float。<br>2. 取值范围：`[0, 1)`。<br>3. 默认值：`0.0`。 |
| `--num-mtp-tokens` | LLM Options | 可选 | 指定 Multi-Token Prediction（MTP）token 数量，`0` 表示不启用。<br>1. 类型：Int。<br>2. 取值范围：非负整数。<br>3. 默认值：`0`。<br>4. 仅支持具备 MTP 能力的模型，例如 DeepSeek。 |
| `--disable-repetition` | LLM Options | 可选 | 禁用 transformer 重复模式优化，保留原始模型行为。<br>1. 类型：Bool。<br>2. 取值范围：开关参数。<br>3. 默认值：`False`。 |
| `--compile` | Optimization Options | 可选 | 在推理前对模型调用 `torch.compile()`。<br>1. 类型：Bool。<br>2. 取值范围：开关参数。<br>3. 默认值：`False`。 |
| `--compile-allow-graph-break` | Optimization Options | 可选 | 允许 `torch.compile()` 过程中出现 graph break。<br>1. 类型：Bool。<br>2. 取值范围：开关参数。<br>3. 默认值：`False`。 |
| `--enable-sequence-parallel` | Optimization Options | 可选 | 编译期间启用 sequence parallel 图改写 pass。<br>1. 类型：Bool。<br>2. 取值范围：开关参数。<br>3. 默认值：`False`。 |
| `--quantize-linear-action` | Quantization Options | 可选 | 指定线性层量化方式。<br>1. 类型：Str。<br>2. 参考值：`DISABLED`、`W8A16_STATIC`、`W8A8_STATIC`、`W4A8_STATIC`、`W8A16_DYNAMIC`、`W8A8_DYNAMIC`、`W4A8_DYNAMIC`、`FP8`、`MXFP4`。<br>3. 默认值：`W8A8_DYNAMIC`。 |
| `--quantize-non-expert-linear-action` | Quantization Options | 可选 | 为 attention 投影、dense MLP、shared experts 等非 expert 线性层指定独立量化方式。<br>1. 类型：Str。<br>2. 参考值：`DISABLED`、`W8A16_STATIC`、`W8A8_STATIC`、`W4A8_STATIC`、`W8A16_DYNAMIC`、`W8A8_DYNAMIC`、`W4A8_DYNAMIC`、`FP8`、`MXFP4`。<br>3. 默认值：`DISABLED`。 |
| `--quantize-lmhead` | Quantization Options | 可选 | 对 lm head 启用量化。<br>1. 类型：Bool。<br>2. 取值范围：开关参数。<br>3. 默认值：`False`。 |
| `--mxfp4-group-size` | Quantization Options | 可选 | 指定 MXFP4 量化的 group size。<br>1. 类型：Int。<br>2. 取值范围：正整数。<br>3. 默认值：`32`。 |
| `--quantize-attention-action` | Quantization Options | 可选 | 指定 KV cache 量化方式。<br>1. 类型：Str。<br>2. 参考值：`DISABLED`、`INT8`、`FP8`。<br>3. 默认值：`DISABLED`。 |
| `--graph-log-url` | Debugging Options | 可选 | 指定编译图日志输出路径，仅在 compile 路径调试时使用。<br>1. 类型：Str。<br>2. 取值范围：文件或目录路径。<br>3. 默认值：`None`。 |
| `--dump-input-shapes` | Debugging Options | 可选 | 输出输入 shape 信息，便于排查模型输入配置。<br>1. 类型：Bool。<br>2. 取值范围：开关参数。<br>3. 默认值：`False`。 |
| `--dump-op-bound-results` | Debugging Options | 可选 | 在结果表中输出算子级 memory、communication、MMA、GP bound 比例。<br>1. 类型：Bool。<br>2. 取值范围：开关参数。<br>3. 默认值：`False`。 |
| `--chrome-trace` | Debugging Options | 可选 | 指定 Chrome trace 输出路径，用于导出性能时间线。<br>1. 类型：Str。<br>2. 取值范围：文件路径。<br>3. 默认值：`None`。 |
| `--num-hidden-layers-override` | Debugging Options | 可选 | 覆盖模型 hidden layers 数量，仅用于调试。<br>1. 类型：Int。<br>2. 取值范围：非负整数。<br>3. 默认值：`0`。 |
| `--tp-size` | Parallelism Options | 可选 | 指定全模型 tensor parallel 并行规模。<br>1. 类型：Int。<br>2. 取值范围：正整数。<br>3. 默认值：`1`。 |
| `--dp-size` | Parallelism Options | 可选 | 指定全模型 data parallel 并行规模。<br>1. 类型：Int。<br>2. 取值范围：正整数。<br>3. 默认值：`None`。 |
| `--ep-size` | Parallelism Options | 可选 | 指定 experts 的 expert parallel 并行规模。<br>1. 类型：Int。<br>2. 取值范围：正整数。<br>3. 默认值：`1`。 |
| `--o-proj-tp-size` | Parallelism Options | 可选 | 指定 attention `o_proj` 层的 TP 并行规模。<br>1. 类型：Int。<br>2. 取值范围：正整数。<br>3. 默认值：`None`。 |
| `--o-proj-dp-size` | Parallelism Options | 可选 | 指定 attention `o_proj` 层的 DP 并行规模。<br>1. 类型：Int。<br>2. 取值范围：正整数。<br>3. 默认值：`None`。 |
| `--mlp-tp-size` | Parallelism Options | 可选 | 指定 MLP 层的 TP 并行规模，可覆盖 `--tp-size`。<br>1. 类型：Int。<br>2. 取值范围：正整数。<br>3. 默认值：`None`。 |
| `--mlp-dp-size` | Parallelism Options | 可选 | 指定 MLP 层的 DP 并行规模，可覆盖 `--dp-size`。<br>1. 类型：Int。<br>2. 取值范围：正整数。<br>3. 默认值：`None`。 |
| `--lmhead-tp-size` | Parallelism Options | 可选 | 指定 lm head 的 TP 并行规模，可覆盖 `--tp-size`。<br>1. 类型：Int。<br>2. 取值范围：正整数。<br>3. 默认值：`None`。 |
| `--lmhead-dp-size` | Parallelism Options | 可选 | 指定 lm head 的 DP 并行规模，可覆盖 `--dp-size`。<br>1. 类型：Int。<br>2. 取值范围：正整数。<br>3. 默认值：`None`。 |
| `--moe-tp-size` | Parallelism Options | 可选 | 指定 experts 的 TP 并行规模，可覆盖 `--tp-size`。<br>1. 类型：Int。<br>2. 取值范围：正整数。<br>3. 默认值：`None`。 |
| `--moe-dp-size` | Parallelism Options | 可选 | 指定 experts 的 DP 并行规模，可覆盖 `--dp-size`。<br>1. 类型：Int。<br>2. 取值范围：正整数。<br>3. 默认值：`1`。 |
| `--word-embedding-tp` | Parallelism Options | 可选 | 启用 word embedding 张量并行并指定并行模式。<br>1. 类型：Str。<br>2. 参考值：`col`、`row`。<br>3. 默认值：`None`，表示不启用 embedding TP。 |
| `--enable-redundant-experts` | Parallelism Options | 可选 | 启用冗余 expert 配置。<br>1. 类型：Bool。<br>2. 取值范围：开关参数。<br>3. 默认值：`False`。<br>4. 单独启用时，每张设备会额外托管 1 个 redundant expert。<br>5. 与 `--enable-external-shared-experts` 同时启用时，分配逻辑与外置 shared experts 相同；若 routing experts 已在各设备间均匀分布、无需 redundant experts 填充，则每个托管 routing experts 的设备会额外托管 1 个 redundant expert。 |
| `--enable-shared-expert-tp` | Parallelism Options | 可选 | 启用 vLLM 风格的 shared experts 张量并行。<br>1. 类型：Bool。<br>2. 取值范围：开关参数。<br>3. 默认值：`False`。<br>4. shared experts 使用 dense MLP TP，并延迟执行 `down_proj` 规约。 |
| `--enable-dispatch-ffn-combine` | Parallelism Options | 可选 | 编译期间启用 dispatch_ffn_combine 融合模式。<br>1. 类型：Bool。<br>2. 取值范围：开关参数。<br>3. 默认值：`False`。 |
| `--enable-external-shared-experts` | Parallelism Options | 可选 | 启用外置 shared experts。<br>1. 类型：Bool。<br>2. 取值范围：开关参数。<br>3. 默认值：`False`。<br>4. 启用后，设备按 `1:top_k` 的比例分配给 external shared experts 与 routing experts；如有需要，会使用 redundant experts 对 routing experts 进行填充。<br>5. 例如 `world_size=64`、`top_k=8`、routing experts 数量为 256 时，8 个设备托管 external shared experts，其余 56 个设备分布 256 个 routing experts：32 个设备各托管 5 个 routing experts，24 个设备各托管 4 个 routing experts 和 1 个 redundant expert。 |
| `--host-external-shared-experts` | Parallelism Options | 可选 | 指定当前设备承载外置 shared experts。<br>1. 类型：Bool。<br>2. 取值范围：开关参数。<br>3. 默认值：`False`。 |
| `--vision-tp-size` | Parallelism Options | 可选 | 指定 vision 模块的 tensor parallel 并行规模。<br>1. 类型：Int。<br>2. 取值范围：正整数。<br>3. 默认值：`1`，表示 vision 模块不切分。 |
| `--image-batch-size` | MultiModal Options | 可选 | 指定图像处理 batch size。<br>1. 类型：Int。<br>2. 取值范围：正整数。<br>3. 默认值：`None`。 |
| `--image-height` | MultiModal Options | 可选 | 指定输入图像高度。<br>1. 类型：Int。<br>2. 取值范围：正整数。<br>3. 默认值：`None`。 |
| `--image-width` | MultiModal Options | 可选 | 指定输入图像宽度。<br>1. 类型：Int。<br>2. 取值范围：正整数。<br>3. 默认值：`None`。 |
| `--remote-source` | Options | 可选 | 指定远端模型来源。<br>1. 类型：Str。<br>2. 参考值：`huggingface`、`modelscope`。<br>3. 默认值：`huggingface`。 |
| `--performance-model` | Options | 可选 | 指定性能模型，可重复指定一个或多个模型。<br>1. 类型：List[Str]。<br>2. 参考值：`analytic`、`profiling`。<br>3. 默认值：未指定时使用 `analytic`。<br>4. `analytic` 为 Roofline 模型，无需 profiling 数据；`profiling` 为基于 profiling CSV 数据库的经验性能模型，需配合 `--profiling-database` 使用。 |
| `--profiling-database` | Options | 可选 | 使用 `profiling` 性能模型时指定 profiling 数据库路径。<br>1. 类型：Str。<br>2. 取值范围：包含 `op_mapping.yaml` 和各 kernel 类型 CSV 文件的目录路径。<br>3. 默认值：`None`。 |
| `--export-empirical-metrics` | Options | 可选 | 导出 M1-M5 metrics JSON，用于离线 M6 计算。<br>1. 类型：Str。<br>2. 取值范围：JSON 文件路径。<br>3. 默认值：`None`。<br>4. 仅开发调试使用，需配合 `--performance-model profiling` 使用。 |

对于 VL 模型，可同时设置 `--image-batch-size`、`--image-height` 和 `--image-width` 来描述输入图像数量与分辨率；纯文本模型可省略这些参数。

运行 `python -m cli.inference.text_generate --help` 查看详情。

### 4.2 视频生成

我们提供 `video_generate.py` 命令行接口来仿真 diffusion transformer 模型的前向传播与性能。该脚本支持仿真视频生成模型（例如类 Stable Video Diffusion 架构）的推理过程，可配置输入维度、采样步数及并行设置。默认提供详细的算子性能分解表格汇总。也可选择将性能时间线导出为 Chrome Trace 文件。

其一般用法如下：

```text
usage: video_generate.py [-h]
                         [--device {TEST_DEVICE,ATLAS_800_A2_376T_64G,ATLAS_800_A2_313T_64G,ATLAS_800_A2_280T_64G,ATLAS_800_A2_280T_64G_PCIE,ATLAS_800_A2_280T_32G_PCIE,ATLAS_800_A3_752T_128G_DIE,ATLAS_800_A3_560T_128G_DIE,ATLAS_800_A3_560T_128G_DIE_ROCE,ATLAS_350_425T_112G,ATLAS_350_425T_84G}]
                         --batch-size BATCH_SIZE --seq-len SEQ_LEN [--chrome-trace CHROME_TRACE] [--height HEIGHT]
                         [--width WIDTH] [--frame-num FRAME_NUM] [--sample-step SAMPLE_STEP]
                         [--log-level {debug,info,warning,error,critical}] [--dtype {float16,float32,bfloat16}]
                         [--remote-source {huggingface,modelscope}]
                         [--quantize-linear-action {DISABLED,W8A16_STATIC,W8A8_STATIC,W4A8_STATIC,W8A16_DYNAMIC,W8A8_DYNAMIC,W4A8_DYNAMIC,FP8,MXFP4}]
                         [--quantize-attention-action {DISABLED,INT8,FP8}] [--use-cfg] [--world-size WORLD_SIZE]
                         [--ulysses-size ULYSSES_SIZE] [--cfg-parallel] [--dit-cache]
                         [--cache-step-range CACHE_STEP_RANGE] [--cache-step-interval CACHE_STEP_INTERVAL]
                         [--cache-block-range CACHE_BLOCK_RANGE]
                         model_id

Run a simulated diffusion transformer forward and dump perf stats.
```

主要参数说明如下：

| 参数名称 | 分类 | 可选/必选 | 参数说明 |
| --- | --- | --- | --- |
| `model_id` | positional arguments | 必选 | 视频生成模型 ID 或本地模型路径。<br>1. 类型：Str。<br>2. 参考值：Diffusers 模型目录、远端 repo ID 或带子目录的 repo ID，需包含 `transformer/config.json` 或兼容的 transformer 配置。<br>3. 默认值：无。<br>4. 推荐使用已审核的本地绝对路径；远端模型 ID 不提供安全保证。 |
| `--device` | options | 可选 | 指定用于仿真的设备配置。<br>1. 类型：Str。<br>2. 参考值：已注册 `DeviceProfile` 名称，包括 `TEST_DEVICE`、`ATLAS_800_A2_376T_64G`、`ATLAS_800_A2_313T_64G`、`ATLAS_800_A2_280T_64G`、`ATLAS_800_A2_280T_64G_PCIE`、`ATLAS_800_A2_280T_32G_PCIE`、`ATLAS_800_A3_752T_128G_DIE`、`ATLAS_800_A3_560T_128G_DIE`、`ATLAS_800_A3_560T_128G_DIE_ROCE`、`ATLAS_350_425T_112G`、`ATLAS_350_425T_84G`。<br>3. 默认值：`TEST_DEVICE`。 |
| `--batch-size` | options | 必选 | 指定输入 batch 大小。<br>1. 类型：Int。<br>2. 取值范围：正整数。<br>3. 默认值：无。 |
| `--seq-len` | options | 必选 | 指定文本序列长度。<br>1. 类型：Int。<br>2. 取值范围：正整数。<br>3. 默认值：无。 |
| `--chrome-trace` | options | 可选 | 指定 Chrome trace JSON 输出路径，用于导出性能时间线。<br>1. 类型：Str。<br>2. 取值范围：文件路径。<br>3. 默认值：`None`。 |
| `--height` | options | 可选 | 指定输入视频或图像帧高度。<br>1. 类型：Int。<br>2. 取值范围：正整数。<br>3. 默认值：`400`。 |
| `--width` | options | 可选 | 指定输入视频或图像帧宽度。<br>1. 类型：Int。<br>2. 取值范围：正整数。<br>3. 默认值：`832`。 |
| `--frame-num` | options | 可选 | 指定视频帧数。<br>1. 类型：Int。<br>2. 取值范围：正整数。<br>3. 默认值：`81`。 |
| `--sample-step` | options | 可选 | 指定 diffusion 采样步数。<br>1. 类型：Int。<br>2. 取值范围：正整数。<br>3. 默认值：`1`。 |
| `--log-level` | options | 可选 | 指定日志级别。<br>1. 类型：Str。<br>2. 参考值：`debug`、`info`、`warning`、`error`、`critical`。<br>3. 默认值：`info`。 |
| `--dtype` | options | 可选 | 指定模型计算数据类型。<br>1. 类型：Str。<br>2. 参考值：`float16`、`float32`、`bfloat16`。<br>3. 默认值：`float16`。 |
| `--remote-source` | options | 可选 | 指定非本地 Diffusers repo ID 的远端模型来源。<br>1. 类型：Str。<br>2. 参考值：`huggingface`、`modelscope`。<br>3. 默认值：`huggingface`。 |
| `--quantize-linear-action` | options | 可选 | 指定线性层量化方式。<br>1. 类型：Str。<br>2. 参考值：`DISABLED`、`W8A16_STATIC`、`W8A8_STATIC`、`W4A8_STATIC`、`W8A16_DYNAMIC`、`W8A8_DYNAMIC`、`W4A8_DYNAMIC`、`FP8`、`MXFP4`。<br>3. 默认值：`W8A8_DYNAMIC`。 |
| `--quantize-attention-action` | options | 可选 | 指定 attention 计算量化方式。<br>1. 类型：Str。<br>2. 参考值：`DISABLED`、`INT8`、`FP8`。<br>3. 默认值：`DISABLED`。 |
| `--use-cfg` | options | 可选 | 启用 classifier-free guidance 相关仿真路径。<br>1. 类型：Bool。<br>2. 取值范围：开关参数。<br>3. 默认值：`False`。 |
| `--world-size` | Parallel Options | 可选 | 指定参与分布式仿真的总设备数。<br>1. 类型：Int。<br>2. 取值范围：正整数。<br>3. 默认值：`1`。 |
| `--ulysses-size` | Parallel Options | 可选 | 指定 Ulysses 并行规模。<br>1. 类型：Int。<br>2. 取值范围：正整数。<br>3. 默认值：`1`。 |
| `--cfg-parallel` | Parallel Options | 可选 | 启用 CFG 并行策略。<br>1. 类型：Bool。<br>2. 取值范围：开关参数。<br>3. 默认值：`False`。 |
| `--dit-cache` | Cache Options | 可选 | 启用 DiT block cache。<br>1. 类型：Bool。<br>2. 取值范围：开关参数。<br>3. 默认值：`False`。 |
| `--cache-step-range` | Cache Options | 可选 | 指定启用 cache 的采样步范围。<br>1. 类型：Str。<br>2. 格式：`start,end`，闭区间。<br>3. 默认值：`None`。<br>4. 设置 `--dit-cache` 时必填。 |
| `--cache-step-interval` | Cache Options | 可选 | 指定 cache 更新步间隔。<br>1. 类型：Int。<br>2. 取值范围：正整数。<br>3. 默认值：`1`，表示不启用 cache 更新复用。 |
| `--cache-block-range` | Cache Options | 可选 | 指定启用 cache 的 block 范围。<br>1. 类型：Str。<br>2. 格式：`start,end`，左闭右开。<br>3. 默认值：`None`。 |

运行 `python -m cli.inference.video_generate --help` 查看详情。
