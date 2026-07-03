# ServingCast仿真 使用指南

## 1 简介

ServingCast 仿真用于基于实例配置和全局配置评估模型服务的端到端性能。它通过 YAML 配置描述实例组、模型结构、请求负载和服务限制，输出 TTFT、TPOT、吞吐量、请求数等性能指标，帮助用户在实际部署前分析服务能力与配置瓶颈。

## 2 环境要求

运行 ServingCast 仿真前，请先完成环境搭建（推荐 Python 3.10+），详见《[msModeling 安装指南](../install_guide/msmodeling_install_guide.md)》。

## 3 输入配置

服务仿真依赖两个 YAML 配置文件：

| 配置文件 | 作用 |
| --- | --- |
| `instance_config_path` | 描述一个或多个实例组，例如角色、实例数量、TP/DP 并行方式等。 |
| `common_config_path` | 描述全局配置，例如模型结构、请求负载、服务限制与仿真参数。 |

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

### 4.1 结果

仿真结束后，控制台会打印类似以下的性能摘要：

```text
         E2E_TIME(s)  TTFT(s)  TPOT(s)  INPUT_TOKENS  OUTPUT_TOKENS  OUTPUT_TOKEN_THROUGHPUT(tok/s)
AVERAGE     1052.591    0.378    0.301        1500.0         3500.0                           3.327
MIN         1050.000    0.300    0.300        1500.0         3500.0                           2.978
MAX         1175.500    0.600    0.336        1500.0         3500.0                           3.334
MEDIAN      1050.100    0.400    0.300        1500.0         3500.0                           3.334
P75         1050.125    0.400    0.300        1500.0         3500.0                           3.334
P90         1050.200    0.500    0.300        1500.0         3500.0                           3.334
P99         1175.500    0.600    0.336        1500.0         3500.0                           3.334
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
- TTFT：首 token 时间（Time-to-first-token）
- TPOT：首 token 之后每个输出 token 的时间（Time-per-output-token）
- OUTPUT_TOKEN_THROUGHPUT：单请求的输出 token 速率
- request_throughput：系统级请求速率
- `input_token_throughput` / `output_token_throughput`：聚合 token 吞吐
