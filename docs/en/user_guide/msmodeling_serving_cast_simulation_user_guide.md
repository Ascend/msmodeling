# ServingCast Simulation Guide

> **Experimental feature**: The service-level fine-grained simulation capability described in this guide is still evolving. Interfaces and behavior may change, and simulation results are for evaluation reference only.

## 1 Introduction

ServingCast simulation uses YAML configuration to simulate end-to-end serving scenarios with multiple instances and requests, and outputs system-level metrics such as throughput and latency (TTFT, TPOT). It describes instance groups, model structure, request workloads, and serving limits through YAML files, and outputs performance metrics such as TTFT, TPOT, throughput, and request count to help users analyze serving capacity and configuration bottlenecks before actual deployment.

**Use Cases:**

- **System Behavior Validation**: Validate the expected performance of a serving configuration before actual deployment
- **Multi-Instance Benchmarking**: Simulate complex serving topologies, such as separate Prefill and Decode clusters
- **Workload Analysis**: Evaluate system performance under specific request patterns and load characteristics
- **Resource Planning**: Determine the required number of instances and their configurations to meet target throughput

**Key Features:**

- YAML-driven configuration for instances and workload
- Support for heterogeneous instance groups
- Comprehensive metrics: E2E latency, TTFT, TPOT, token throughput

## 2 Environment Requirements

Before running ServingCast simulation, complete environment setup first. Python 3.10+ is recommended. For details, see [Quick Start: Environment Setup and First Simulation](../install_guide/msmodeling_install_guide.md).

## 3 Input Configuration

Service simulation depends on two YAML configuration files:

| Configuration File | Purpose |
| --- | --- |
| `instance_config_path` | Describes one or more instance groups, such as role, instance count, and TP/DP parallelism. |
| `common_config_path` | Describes global settings, such as model structure, request workload, serving limits, and simulation parameters. |

## 4 Run Simulation

Its general usage is shown below:

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

Parameter descriptions:

| Parameter | Required/Optional | Description |
| --- | --- | --- |
| `--instance_config_path` | Required | Path to one or more instance configuration files. Format: YAML file path list. Each file declares one or more instance groups, such as role, instance count, and TP/DP parallelism. Default: none. |
| `--common_config_path` | Required | Path to the global configuration file. Format: YAML file path. It describes model structure, request workload, serving limits, and simulation parameters. Default: none. |
| `--enable_profiling` | Optional | Enables profiling and outputs more fine-grained system performance information. Valid range: flag option. Default: `False`. |
| `--profiling_output_path` | Optional | Specifies profiling result directory. Format: directory path. Default: `./profiling_results`. |

Example:

- Basic usage

```bash
python -m serving_cast.main --instance_config_path=./serving_cast/example/instances.yaml --common_config_path=./serving_cast/example/common.yaml
```

### 4.1 Result

After the simulation finishes, the console prints a performance summary similar to:

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

- E2E_TIME: End-to-end latency for a single request (request issued → last token)
- TTFT: Time-to-first-token
- TPOT: Time-per-output-token after the first token
- OUTPUT_TOKEN_THROUGHPUT: Per-request output-token rate
- request_throughput: System-level request rate
- `input_token_throughput` / `output_token_throughput`: Aggregate token throughput
