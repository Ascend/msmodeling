# ServingCast

## Introduction

ServingCast is a suite of tools for system-level inference serving simulation and throughput optimization.

## Components

ServingCast consists of two main components:

### 1. Throughput Optimizer

The Throughput Optimizer automatically searches for the optimal model configuration (parallelism strategy, batch size) to maximize token throughput under specified SLO constraints (e.g., limits on TTFT, TPOT).

**Use Cases:**

- **Hardware Planning**: Determine the optimal parallelism strategy (TP/DP/PP) and batch size for a given model on specific hardware before deployment
- **SLO-Constrained Optimization**: Find the configuration that maximizes throughput while meeting latency requirements (TTFT/TPOT limits)
- **Disaggregated Serving Design**: Optimize Prefill and Decode phases independently and calculate the optimal P:D instance ratio

**Key Features:**

- Aggregation mode: Optimizes combined Prefill-Decode serving architecture
- Disaggregation mode: Separates Prefill and Decode phases for independent optimization
- PD Ratio Optimization: Calculates optimal Prefill-to-Decode instance ratio for maximum system throughput

See [Throughput Optimizer Guide](./msmodeling_throughput_optimizer_user_guide.md) for detailed usage.

### 2. Serving Simulation

The Serving Simulation simulates end-to-end serving scenarios with multiple instances and requests based on YAML configuration files, outputting system-level metrics like throughput, latency (TTFT, TPOT).

**Use Cases:**

- **System Behavior Validation**: Validate the expected performance of a serving configuration before actual deployment
- **Multi-Instance Benchmarking**: Simulate complex serving setups with multiple instance groups (e.g., separate Prefill and Decode clusters)
- **Workload Analysis**: Evaluate system performance under specific request patterns and load characteristics
- **Resource Planning**: Determine the required number of instances and their configurations to meet target throughput

**Key Features:**

- YAML-driven configuration for instances and workload
- Support for heterogeneous instance groups
- Comprehensive metrics: E2E latency, TTFT, TPOT, token throughput

See [Serving Simulation Guide](./msmodeling_serving_cast_simulation_user_guide.md) for detailed usage.
