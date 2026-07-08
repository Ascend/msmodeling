# Model and Feature Support Matrix

This document summarizes the currently supported model types and simulation features, making it easier to quickly understand the tool capability boundaries.

## Reading Notes

- If you only want to check the currently supported models, see "Currently Supported Models".
- If you need to check whether a simulation or optimization capability is available, see "Feature Support".
- For detailed module usage, read this together with the following user guides:
  - [TensorCast User Guide](../msmodeling_tensor_cast_user_guide.md)
  - [Throughput Optimizer Guide](../msmodeling_throughput_optimizer_user_guide.md)
  - [Service Parameter Optimizer](../optix_user_guide.md)
  - [Web UI User Guide](../msmodeling_web_ui_user_guide.md)
  - [TensorCast New Model Adaptation Guide](../msmodeling_tensor_cast_new_model_adaptation_user_guide.md)
  - [Custom Plugin Developer Guide](../optix_plugin_user_guide.md)
- For the experimental serving simulation capability, read this together with the [ServingCast Simulation Guide](../msmodeling_serving_cast_simulation_user_guide.md).

## Currently Supported Models

> Note: OptiX supports all models that can be deployed in the target serving framework, and is not limited by the model scope listed below.

| Model Type | Model Family | Supported Models |
| --- | --- | --- |
| Text Models | DeepSeek | DeepSeek V4, DeepSeek V3.2, DeepSeek V3 |
| Text Models | Kimi | Kimi-K2.6, Kimi-K2.5, Kimi-K2 (supported through the DeepSeek V3 compatibility path) |
| Text Models | Qwen | Qwen3.5, Qwen3.5 MoE, Qwen3-Next, Qwen3 Dense, Qwen3 MoE |
| Text Models | GLM | GLM5.1, GLM5, GLM-4 MoE |
| Text Models | ERNIE | ERNIE 4.5 MoE |
| Text Models | Bailing / MiMo / MiniMax | Bailing MoE, MiMo v2 Flash, MiniMax M2 |
| Vision-Language Models | VL Models | Qwen3-VL, Qwen3-VL MoE, GLM-4V, GLM-4V MoE, InternVL |
| Video Generation Models | Diffusers DiT Video Generation Models | Wan, HunyuanVideo, HunyuanVideo1.5 |

## Feature Support

### Model Inference Performance Simulation

| Feature | Description |
| --- | --- |
| Multi-Hardware Simulation | Supports built-in device profiles for Ascend hardware such as Atlas 800 A2/A3 and Atlas 350, as well as custom device profiles, to estimate operator latency, communication overhead, and memory usage in multi-device scenarios without real hardware. |
| LLM Prefill / Decode Staged Simulation | Models prefill and decode as separate computation paths, covering attention, KV cache growth, and per-token generation overhead. |
| Prefix Cache Simulation | Approximates prefill reuse benefits from prefix cache hits and evaluates their impact on time to first token. |
| MTP Speculative Decoding Simulation | Models the extra draft/verify computation introduced by Multi-Token Prediction and evaluates its impact on latency and throughput. |
| Compilation and Graph Optimization | Rewrites the forward computation graph through compilation, fusing typical subgraph patterns such as RMSNorm and Grouped Matmul into unified performance operators to better reflect real deployment execution patterns. |
| Multi-Stream Compute-Communication Overlap | Schedules compute and communication on separate execution streams in the compile path, modeling cross-stream synchronization and end-to-end overlap benefits. |
| Quantization Simulation | Models quantization compute and memory costs for Linear, non-expert Linear, LMHead, and Attention paths, supporting strategies such as W8A8, W4A8, FP8, and MXFP4. |
| Parallelism and MoE Extensions | Models communication and compute overhead under global TP / DP / EP and fine-grained parallelism (Embedding TP, Vision TP, and more), covering redundant experts, external/shared expert, and other MoE deployment patterns. |
| VL Multimodal Input | Incorporates image batch, resolution, and vision-encoder parallelism into forward simulation for joint text and multimodal analysis. |
| Model Configuration Sources | Supports local model directories and remote model configuration from Hugging Face, ModelScope, and other sources. |
| Performance Model Selection | Supports Roofline-based analytic estimation and profiling-data-based performance modeling, with optional combined comparison across estimation paths. |
| Chrome Trace / Debug | Outputs operator-level timelines, shapes, graph structure, and bound analysis for bottleneck identification, validation, and visualization. |
| Video Generation DiT Simulation | Simulates multi-step denoising for Diffusers DiT video models such as Wan and HunyuanVideo, covering resolution, frame count, sampling steps, and quantization. |
| Ulysses Parallel Simulation | Models attention communication and compute overhead under sequence-dimension Ulysses parallelism for multi-device video DiT inference analysis. |
| CFG Simulation | Models conditional and unconditional forward paths under Classifier-Free Guidance and evaluates the guidance overhead per denoising step. |
| CFG Parallel Simulation | Models cross-device coordination and result gathering when running CFG paths in parallel, analyzing gains over serial guidance execution. |
| DiT Cache Simulation | Models cache reuse of intermediate results across sampling steps and block ranges during denoising and evaluates the impact of effective cache windows on total runtime. |

### Service-Level Performance Simulation

| Feature | Description |
| --- | --- |
| LLM / VLM Constrained Throughput Optimization | Automatically searches for optimal parallel strategies, batch settings, and token throughput for LLM and VLM under service-level constraints such as TTFT, TPOT, and serving cost. |
| PD Colocated Mode | Jointly evaluates Prefill and Decode in the same instance to quickly obtain overall serving throughput and parallel configuration recommendations. |
| PD Disaggregated Mode | Searches optimal Prefill and Decode instance configurations separately for PD-disaggregated deployment evaluation. |
| PD Ratio Optimization | Searches for the optimal Prefill / Decode instance ratio under a fixed hardware budget to balance resource allocation and serving capacity. |
| Parallel Strategy Search | Performs combinatorial search across TP, EP, and MOE-DP. |
| MTP Configuration Search | Searches MTP token count, acceptance rate, and related settings to evaluate the impact of speculative decoding on serving throughput. |
| Batch and Concurrency Search | Automatically searches batch size and request concurrency combinations that satisfy SLOs while respecting service constraints such as maximum batched tokens per step. |
| Chunked Prefill Simulation | Automatically splits long prompts into multiple prefill chunks when effective input length exceeds the per-step prefill token budget, improving TTFT, Prefill throughput, colocated scheduling, and memory modeling; can be combined with Prefix Cache analysis. |
| Prefix Cache Simulation | Models prefix cache hit rates and evaluates their impact on serving capacity. |
| Variable-Length Workload Simulation | Evaluates serving capacity under variable input-length distributions. |
| Cross-Hardware Comparison | Runs parallel searches across multiple device profiles under the same workload and SLO constraints and outputs best-configuration and throughput comparisons. |
| Compile and Fusion Options | Enables compile, Sequence Parallel, MoE fusion, Embedding TP, and related optimization paths during optimization to better reflect real serving deployments. |
| Multi-Stream Compute-Communication Overlap | In compile-enabled throughput optimization, schedules compute and communication on separate streams and models synchronization overhead and overlap benefits end to end. |
| Result Visualization and Export | Outputs candidate-configuration curves, raw optimization results, and operator-level traces for comparing parallel and batch combinations. |

### Service-Level Fine-Grained Simulation

> Note: This module is an experimental feature. See the [Service-Level Fine-Grained Simulation User Guide](../msmodeling_serving_cast_simulation_user_guide.md) for details.

| Feature | Description |
| --- | --- |
| YAML-Driven Serving Simulation | Simulates queuing, scheduling, and end-to-end serving flows across multiple instances and requests based on instance groups and global service configuration. |
| PD Colocated / PD Disaggregated Topology | Supports both colocated and disaggregated instance role combinations to cover different PD deployment topologies. |
| System Metrics Output | Aggregates request-level and system-level results, outputting serving metrics such as TTFT, TPOT, throughput, and E2E latency. |
| Profiling Collection | Collects profiling data during simulation for further analysis of operator latency and system behavior. |

### Web UI

| Feature | Description |
| --- | --- |
| LLM / VL Forward Simulation | Configures LLM and VL forward simulation through page forms, supporting concurrency lists, TP lists, quantization, MTP, Prefix Cache, and operator/memory analysis. |
| Video Generation Simulation | Provides a visual configuration entry point for video generation models, covering Ulysses, CFG, DiT Cache, quantization, and trace capabilities. |
| Throughput Optimization Experiments | Launches service throughput optimization from the page, supporting PD colocated, PD disaggregated, and PD ratio modes plus multi-device comparison. |
| Command Preview and Task Cache | Supports pre-run configuration preview and caches historical task results, logs, and detailed cases. |
| Result Display and Export | Parses simulation logs and generates charts, tables, memory analysis, bandwidth bottlenecks, operator details, and Excel export. |

### Service-Level Measured Optimization

| Feature | Description |
| --- | --- |
| Serving Framework Measured Optimization | Combines PSO particle swarm optimization with Early Rejection to automatically search for optimal deployment parameters on real serving frameworks under latency constraints. |
| Multi-Engine and Benchmark Policies | Supports inference engines such as vLLM and MindIE, with measured optimization under multiple benchmark evaluation policies. |
| Custom Configuration and Resume from Checkpoint | Supports custom optimization-space configuration and resuming optimization tasks from the last checkpoint after interruption. |
