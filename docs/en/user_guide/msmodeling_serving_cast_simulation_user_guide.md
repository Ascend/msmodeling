# Fine-Grained Serving Simulation User Guide

> **Experimental feature**: The fine-grained serving simulation capability described in this guide is still under continuous iteration. Interfaces and behavior may change, and the simulation results are for evaluation reference only.

## 1 Introduction

The ServingCast simulation is based on YAML configuration. It simulates end-to-end serving scenarios with multiple instances and requests, and outputs system-level metrics such as throughput and latency (TTFT, TPOT). It describes instance groups, model architecture, request workload, and serving limits through YAML configuration, and outputs performance metrics such as TTFT, TPOT, throughput, and request count, helping you analyze service capacity and configuration bottlenecks before actual deployment.

**Applicable Scenarios:**

- **System behavior validation**: Verify the expected performance of a service configuration before actual deployment.
- **Multi-instance benchmarking**: Simulate complex service topologies, such as independent prefill and decode clusters.
- **Load analysis**: Evaluate system performance under specific request patterns and load characteristics.
- **Resource planning**: Determine the number of instances and their configuration required to meet the target throughput.

**Key Features:**

- YAML-based instance and workload configuration
- Support for heterogeneous instance groups
- Comprehensive metrics: end-to-end latency, TTFT, TPOT, and token throughput

## 2 Environment Requirements

Before running a ServingCast simulation, complete the environment setup (Python 3.10+ recommended). For details, see [msModeling Installation Guide](../install_guide/msmodeling_install_guide.md).

## 3 Input Configuration

The service simulation relies on two YAML configuration files:

| Configuration File    | Description |
| --------------------  | --- |
| instance_config_path  | Describes one or more instance groups, for example, role, number of instances, and TP/DP modes. |
| common_config_path    | Describes global configuration, for example, model architecture, request workload, serving limits, and simulation parameters. |

## 4 Running Simulation

Its general usage is shown below:

```text
usage: python -m serving_cast.main [-h] --instance_config_path INSTANCE_CONFIG_PATH [INSTANCE_CONFIG_PATH ...] --common_config_path COMMON_CONFIG_PATH

Run a service inference simulation driven by YAML configuration files.

required arguments:
  --instance_config_path INSTANCE_CONFIG_PATH [INSTANCE_CONFIG_PATH ...]
                        Path to a YAML file that declares one or more instance groups.
                        Each group defines a homogeneous pool of nodes (role, count, TP/DP modes)
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

Parameters:

| Parameter                | Optional/Required | Description |
| -----------------------  | --- | --- |
| --instance_config_path   | Required | Paths to one or more instance configuration files. Format: list of YAML file paths. Each file declares one or more instance groups, such as role, number of instances, and TP/DP modes. Default: none. |
| --common_config_path     | Required | Path to the global configuration file. Format: YAML file path. Used to describe model architecture, request workload, serving limits, and simulation parameters. Default: none. |
| --enable_profiling       | Optional | Enables profiling to output more fine-grained system performance information. Value range: switch parameter. Default: `False`. |
| --profiling_output_path  | Optional | Specifies the profiling result directory. Format: directory path. Default: `./profiling_results`. |

Example:

- Basic usage

```bash
python -m serving_cast.main --instance_config_path=./serving_cast/example/instances.yaml --common_config_path=./serving_cast/example/common.yaml
```

### 4.1 Result

After the simulation finishes, the console prints a performance summary similar to the following:

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

Metric descriptions:

- `E2E_TIME`: end-to-end latency per request (issue → last token)
- `TTFT`: time-to-first-token
- `TPOT`: time-per-output-token after the first token
- `OUTPUT_TOKEN_THROUGHPUT`: per-request output-token rate
- `request_throughput`: system-wide request rate
- `input_token_throughput`/`output_token_throughput`: aggregate token throughput
