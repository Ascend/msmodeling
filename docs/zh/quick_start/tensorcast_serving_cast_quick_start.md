# TensorCast 与 ServingCast 快速入门

<br>

## 1. 概述

msModeling 提供单模型性能仿真与服务级推理仿真能力。本文档面向首次体验 TensorCast 和 ServingCast 的用户，带您从环境检查开始，依次跑通 LLM 文本生成仿真、吞吐优化和端到端服务仿真，快速理解工具的核心输入、输出与适用场景。

### 1.1 前言

**体验地图（核心操作约 10 分钟）**

> **执行顺序建议**：步骤 1 为基础环境检查；步骤 2 用于体验 TensorCast 单模型仿真；步骤 3 和步骤 4 用于体验 ServingCast 的吞吐优化与服务仿真，可按需选择。

| 步骤 | 环节 | 核心模块 | 参考操作耗时 | 建议原理学习 |
| :---: | :---: | :--- | :---: | :---: |
| **1** | **环境准备** | `msModeling` | 2 分钟 | 5 分钟 |
| **2** | **单模型仿真** | `TensorCast` | 1 分钟 | 10 分钟 |
| **3** | **吞吐优化** | `ServingCast / Throughput Optimizer` | 2 分钟 | 15 分钟 |
| **4** | **服务仿真** | `ServingCast` | 2 分钟 | 15 分钟 |

### 1.2 环境准备

👉 **【重要】请先按《[msModeling 安装指南](../install_guide/msmodeling_install_guide.md)》完成环境安装和配置。**

> [!CAUTION]注意
> 本文档默认在 msModeling 仓库根目录执行命令。若在其他目录运行，请先设置 `PYTHONPATH`，否则可能出现 `No module named cli`、`No module named tensor_cast` 等模块导入错误。

## 2. 操作步骤

> [!NOTE]说明
> 后续命令均支持直接复制执行。建议先使用 `TEST_DEVICE` 跑通流程，确认输出符合预期后，再替换为实际目标硬件设备和业务模型。

### 2.1【环境】确认运行环境

开始体验前，请先完成《[msModeling 安装指南](../install_guide/msmodeling_install_guide.md)》中的环境搭建，包括克隆仓库、创建虚拟环境、安装依赖以及配置 `PYTHONPATH`。

本文后续命令默认在 msModeling 仓库根目录执行。若不在仓库根目录运行，请先设置：

```bash
export PYTHONPATH=/path/to/msmodeling:$PYTHONPATH
```

TensorCast 运行 Hugging Face 模型时需要读取模型配置。如果当前环境无法直接访问 Hugging Face，可设置镜像：

```bash
export HF_ENDPOINT="https://hf-mirror.com"
```

可使用以下命令快速确认命令行入口可访问：

```bash
python -m cli.inference.text_generate --help
python -m serving_cast.main --help
```

若上述命令无法正常输出帮助信息，请优先检查虚拟环境是否已激活、依赖是否安装完成，以及 `PYTHONPATH` 是否指向 msModeling 仓库根目录。

### 2.2【单模型仿真】运行 TensorCast 文本生成仿真

TensorCast 面向 PyTorch 程序进行性能建模。它不会在真实加速器上执行模型，而是拦截计算图，并基于目标设备画像估算算子耗时、显存占用和整体推理性能。

> [!NOTE]知识点：TensorCast 输出
> TensorCast 默认输出算子级性能汇总、总执行时间、TPS/Device 与显存占用。若指定 `--chrome-trace`，还可以生成 Chrome Trace 文件用于可视化分析。

#### 2.2.1 执行 LLM 文本生成仿真

```bash
python -m cli.inference.text_generate Qwen/Qwen3-32B \
    --num-queries 2 \
    --query-length 3500 \
    --device TEST_DEVICE
```

#### 2.2.2 查看仿真结果

命令执行成功后，终端会输出类似以下内容：

```text
Model compilation and execution time: 0.192 s
----------------------------------------------  --------------  ------------  ----------
                     Name                       analytic total  analytic avg  # of Calls
----------------------------------------------  --------------  ------------  ----------
tensor_cast.static_quant_linear.default              884.004ms       1.973ms         448
tensor_cast.attention.default                        259.855ms       4.060ms          64
...
Total time for analytic: 1.744s
[analytic] TPS/Device: 4013 token/s
Total device memory: 64.000 GB
```

关键指标说明：

- `analytic total`：算子估算总耗时。
- `analytic avg`：算子每次调用的平均耗时。
- `# of Calls`：算子被调用的次数。
- `TPS/Device`：每设备每秒 token 数。
- `Total device memory`：权重、KV cache、activation 等显存估算结果。

成功标准：

- 终端输出算子级性能表。
- 输出 `Total time for analytic` 或 `[analytic] TPS/Device`。
- 输出显存估算结果，例如 `Total device memory`。

#### 2.2.3 生成 Chrome Trace（可选）

如需查看更细粒度的时间线，可增加 `--chrome-trace` 参数：

```bash
python -m cli.inference.text_generate Qwen/Qwen3-32B \
    --num-queries 2 \
    --query-length 3500 \
    --device TEST_DEVICE \
    --chrome-trace ./tensorcast_trace.json
```

生成后，可通过 `chrome://tracing` 或 MindStudio Insight 打开 trace 文件。

### 2.3【吞吐优化】运行 ServingCast 吞吐优化器

ServingCast 的吞吐优化器可在 TTFT、TPOT 等 SLO 约束下，自动搜索最优并行策略和 batch 配置，帮助评估给定模型在目标硬件上的最大服务吞吐。

> [!NOTE]知识点：PD 混部
> PD 混部表示 Prefill 与 Decode 运行在同一实例中，适合快速评估整体服务吞吐。若需要分别评估 Prefill 与 Decode，可继续阅读《[吞吐优化指南](../user_guide/msmodeling_throughput_optimizer_user_guide.md)》。

#### 2.3.1 执行吞吐优化

以下命令用于快速体验 PD 混部场景。首次体验时不额外指定搜索维度，工具会使用默认 TP 搜索范围；如果运行耗时较长，可减少 `--num-devices` 或在进阶使用时显式指定 `--tp-sizes` 缩小搜索范围。

```bash
python -m cli.inference.throughput_optimizer Qwen/Qwen3-32B \
    --device TEST_DEVICE \
    --num-devices 8 \
    --input-length 3500 \
    --output-length 1500 \
    --quantize-linear-action W8A8_DYNAMIC \
    --quantize-attention-action DISABLED \
    --tpot-limits 50
```

#### 2.3.2 查看优化结果

命令执行成功后，终端会先打印输入配置和最优配置摘要，随后展示候选并行配置表。例如：

```text
Input Configuration:
  Model: Qwen/Qwen3-32B
  Devices: 8 TEST_DEVICE
  TPOT Limits: 50.0 ms

Overall Best Configuration:
  Best Throughput: 2161.56 tokens/s
  TTFT: 13848.08 ms
  TPOT: 49.98 ms

Top 4 PD Aggregated Configurations:
| Top | Throughput (token/s) | TTFT (ms) | TPOT (ms) | concurrency | num_devices | parallel           | batch_size |
|  1  | 2161.56              | 13848.08  | 49.98     | 128         | 8           | TP=4 | PP=1 | DP=2 | 64         |
```

重点关注以下字段：

- `TP` / `DP`：推荐的并行策略。
- `concurrency`：当前候选配置支持的并发请求数。
- `batch size`：满足 SLO 约束下的批大小。
- `TTFT` / `TPOT`：首 token 时间与每输出 token 时间。
- `Throughput (token/s)`：系统级输出 token 吞吐，数值越大表示吞吐越高。

成功标准：

- 终端输出 `Overall Best Configuration` 或候选配置表。
- 输出 `Throughput`、`TTFT`、`TPOT` 等指标。
- 没有出现模型配置加载失败或参数冲突报错。

### 2.4【服务仿真】运行 ServingCast 端到端仿真

ServingCast 服务仿真基于 YAML 配置描述实例组、请求负载和服务限制，可模拟多实例、多请求的端到端 serving 场景，并输出 E2E_TIME、TTFT、TPOT、请求吞吐和 token 吞吐等系统级指标。

#### 2.4.1 查看示例配置

仓库内置了可直接使用的示例配置：

```bash
ls serving_cast/example/instances.yaml serving_cast/example/common.yaml
```

两个配置文件的作用如下：

| 配置文件 | 作用 |
| --- | --- |
| `instances.yaml` | 描述一个或多个实例组，例如角色、实例数量、TP/DP 并行方式等。 |
| `common.yaml` | 描述全局配置，例如模型结构、请求负载、服务限制与仿真参数。 |

示例配置默认使用 `TEST_DEVICE` 和 `Qwen/Qwen3-32B`。如果需要切换模型、请求长度或请求数量，可修改 `common.yaml`；如果需要调整实例数量、设备数或并行策略，可修改 `instances.yaml`。

#### 2.4.2 执行服务仿真

```bash
python -m serving_cast.main \
    --instance_config_path=./serving_cast/example/instances.yaml \
    --common_config_path=./serving_cast/example/common.yaml
```

#### 2.4.3 查看服务仿真结果

仿真结束后，控制台会打印类似以下的性能摘要：

```text
         E2E_TIME(s)  TTFT(s)  TPOT(s)  INPUT_TOKENS  OUTPUT_TOKENS  OUTPUT_TOKEN_THROUGHPUT(tok/s)
AVERAGE     1052.591    0.378    0.301        1500.0         3500.0                           3.327
MIN         1050.000    0.300    0.300        1500.0         3500.0                           2.978
MAX         1175.500    0.600    0.336        1500.0         3500.0                           3.334
======== Overall Summary ========
request_throughput(req/s)      0.082
input_token_throughput(tok/s)  122.399
output_token_throughput(tok/s) 285.598
```

指标说明：

- `E2E_TIME`：单请求端到端延迟。
- `TTFT`：首 token 时间。
- `TPOT`：首 token 之后每个输出 token 的时间。
- `benchmark_duration`：本次仿真的总耗时。
- `request_throughput`：系统级请求吞吐。
- `input_token_throughput` / `output_token_throughput`：聚合 token 吞吐。

成功标准：

- 终端输出请求级统计表。
- 输出 `Overall Summary`。
- 输出 `request_throughput`、`input_token_throughput` 或 `output_token_throughput`。

## 3. 结果校验与下一步

如果上述命令执行成功，说明已完成 TensorCast 与 ServingCast 的核心流程体验：

- TensorCast：已完成单模型文本生成性能仿真，并获得算子级耗时、TPS/Device 与显存估算。
- Throughput Optimizer：已完成 SLO 约束下的吞吐优化，并获得推荐并行策略与吞吐指标。
- ServingCast：已完成端到端服务仿真，并获得 TTFT、TPOT、请求吞吐和 token 吞吐。

常见问题：

- 如果提示无法下载模型配置，请确认网络可访问 Hugging Face，或设置 `HF_ENDPOINT` 镜像。
- 如果提示无法找到 `cli`、`tensor_cast` 或 `serving_cast` 模块，请确认当前目录为仓库根目录，或已正确设置 `PYTHONPATH`。
- 如果吞吐优化运行耗时较长，可先减小搜索范围，例如减少 `--num-devices` 或显式指定 `--tp-sizes`。

更多用法请继续阅读：

- 《[TensorCast 使用指南](../user_guide/msmodeling_tensor_cast_user_guide.md)》
- 《[ServingCast 使用指南](../user_guide/msmodeling_serving_cast_user_guide.md)》
- 《[吞吐优化指南](../user_guide/msmodeling_throughput_optimizer_user_guide.md)》
- 《[服务仿真指南](../user_guide/msmodeling_serving_cast_simulation_user_guide.md)》
