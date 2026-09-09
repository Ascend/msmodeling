# 服务化细粒度仿真 使用指南

> **实验性功能**：本指南描述的服务化细粒度仿真能力仍在持续迭代中，接口与行为可能发生变化，仿真结果仅供评估参考。

## 1 简介

ServingCast 仿真基于 YAML 配置，模拟多实例、多请求的端到端 serving 场景，并输出系统级指标，如吞吐、延迟（TTFT、TPOT）。它通过 YAML 配置描述实例组、模型结构、请求负载和服务限制，输出 TTFT、TPOT、吞吐量、请求数等性能指标，帮助用户在实际部署前分析服务能力与配置瓶颈。

**适用场景：**

- **系统行为验证**：在实际部署前，验证某套服务配置的预期性能
- **多实例基准测试**：仿真复杂服务拓扑，例如独立的 Prefill 与 Decode 集群
- **负载分析**：在特定请求模式与负载特征下评估系统性能
- **资源规划**：确定满足目标吞吐所需的实例数量及其配置

**主要特性：**

- 基于 YAML 的实例与负载配置
- 支持异构实例组
- 全面指标：端到端延迟、TTFT、TPOT、token 吞吐

## 2 环境要求

运行 ServingCast 仿真前，请先完成环境搭建（推荐 Python 3.10+），详见《[msModeling 安装指南](../install_guide/msmodeling_install_guide.md)》。

## 3 输入配置

服务仿真依赖两个 YAML 配置文件：

| 配置文件 | 作用 |
| --- | --- |
| `instance_config_path` | 描述一个或多个实例组，例如角色、实例数量、TP/DP 并行方式等。 |
| `common_config_path` | 描述全局配置，例如模型结构、请求负载、服务限制与仿真参数。 |

### 3.1 MTP 支持契约

`model_config.num_mtp_tokens` 表示 speculative token 数量。`model_config.mtp_acceptance_rate` 默认为 `[0.9, 0.6, 0.4, 0.2]`，因此 `num_mtp_tokens <= 4` 时无需覆盖默认值；使用更多 speculative token 或自定义接受率时，需提供至少相同数量、位于 `[0, 1]` 的有限值。Decode 查询窗口固定为 `num_mtp_tokens + 1`，请求序列长度和 KV 增量只保留 accepted tokens。

| 模式 | MTP 支持 |
| --- | --- |
| 聚合与 P/D 分离 serving | 支持 |
| interpolation | 暂不支持；设置 `model_config.enable_interpolate: false` |
| multi-process prediction | 暂不支持；设置 `model_config.enable_multi_process: false` |
| 同一步混合 Prefill/Decode | 串行拆分为 Prefill-only 与 Decode-only 同构调用；不代表真实 mixed-batch latency |

## 4 运行仿真

其一般用法如下所示：

```text
usage: python -m serving_cast.main [-h] --instance_config_path INSTANCE_CONFIG_PATH [INSTANCE_CONFIG_PATH ...] --common_config_path COMMON_CONFIG_PATH

Run a service inference simulation driven by YAML configuration files.

required arguments:
  --instance_config_path INSTANCE_CONFIG_PATH [INSTANCE_CONFIG_PATH ...]
                        Path to a YAML file that declares one or more instance groups.
                        Each group defines a homogeneous pool of nodes (role, count, TP/DP parallelism)
                        and can be mixed-and-matched in a single benchmark run.
  --common_config_path COMMON_CONFIG_PATH
                        Path to a YAML file with global settings: model architecture,
                        request-generation workload, and serving limits.

optional arguments:
  -h, --help            show this help message and exit
  --enable_profiling    Enable profiling during simulation (default: False)
  --profiling_output_path PROFILING_OUTPUT_PATH
                        Path to directory where profiling results will be saved (default: ./profiling_results)
```

参数说明：

| 参数 | 可选/必选 | 说明 |
| --- | --- | --- |
| `--instance_config_path` | 必选 | 一个或多个实例配置文件路径。格式：YAML 文件路径列表。每个文件声明一个或多个实例组，例如角色、实例数量、TP/DP 并行方式等。默认值：无。 |
| `--common_config_path` | 必选 | 全局配置文件路径。格式：YAML 文件路径。用于描述模型结构、请求负载、服务限制与仿真参数。默认值：无。 |
| `--enable_profiling` | 可选 | 开启 profiling，输出更细粒度的系统性能信息。取值范围：开关参数。默认值：`False`。 |
| `--profiling_output_path` | 可选 | 指定 profiling 结果目录。格式：目录路径。默认值：`./profiling_results`。 |

示例：

- 基本用法

```bash
python -m serving_cast.main --instance_config_path=./serving_cast/example/instances.yaml --common_config_path=./serving_cast/example/common.yaml
```

- PD 分离用法

`serving_cast/example/pd_disaggregation/` 提供了 Prefill 与 Decode 拆分部署的完整案例（实例拓扑、资源配置、请求负载、KV 传输参数与结果解释），可用一条命令运行：

```bash
bash serving_cast/example/pd_disaggregation/run_pd_disaggregation.sh
```

字段含义、P:D 配比调整方式与常见失败原因详见《[PD 分离仿真示例](../../../serving_cast/example/pd_disaggregation/README.md)》。

### 4.1 结果

仿真结束后，控制台会打印类似以下的性能摘要：

```text
         E2E_TIME(s)  CLIENT_TTFT(s)  SERVER_TTFT(s)  ADMISSION_WAIT(s)  TPOT(s)  INPUT_TOKENS  OUTPUT_TOKENS  OUTPUT_TOKEN_THROUGHPUT(tok/s)
AVERAGE     1052.591           0.378           0.301              0.077    0.301        1500.0         3500.0                           3.327
MIN         1050.000           0.300           0.300              0.000    0.300        1500.0         3500.0                           2.978
MAX         1175.500           0.600           0.336              0.264    0.336        1500.0         3500.0                           3.334
MEDIAN      1050.100           0.400           0.300              0.100    0.300        1500.0         3500.0                           3.334
P75         1050.125           0.400           0.300              0.100    0.300        1500.0         3500.0                           3.334
P90         1050.200           0.500           0.300              0.200    0.300        1500.0         3500.0                           3.334
P99         1175.500           0.600           0.336              0.264    0.336        1500.0         3500.0                           3.334
======== Overall Summary ========
benchmark_duration(s)          1225.500
total_requests                 100.000
request_throughput(req/s)      0.082
total_input_tokens             150000.000
input_token_throughput(tok/s)  122.399
total_output_tokens            350000.000
output_token_throughput(tok/s) 285.598
```

指标说明：

- E2E_TIME：单请求的端到端延迟（发出请求 → 最后一个 token）
- CLIENT_TTFT：请求离开客户端到首 token
- SERVER_TTFT：请求被服务端接纳到首 token
- ADMISSION_WAIT：请求离开客户端到服务端接纳；`CLIENT_TTFT = SERVER_TTFT + ADMISSION_WAIT`
- TPOT：首 token 之后的标准请求级每输出 token 时间；仅一个输出 token 时为零
- OUTPUT_TOKEN_THROUGHPUT：单请求的输出 token 速率
- request_throughput：系统级请求速率
- `input_token_throughput` / `output_token_throughput`：聚合 token 吞吐
