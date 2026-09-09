# Model and Feature Support Matrix

This document summarizes the currently supported model types and simulation features, helping you quickly confirm the capability boundaries of the tool.

## Reading Notes

- If you are only interested in the currently supported models, see "Currently Supported Models".
- To confirm whether a CLI capability is available, see "Feature Support".
- For detailed usage of each module, read the following user guides:
  - [Model Inference Performance Simulation User Guide](../msmodeling_tensor_cast_user_guide.md)
  - [Serving Performance Simulation User Guide](../msmodeling_throughput_optimizer_user_guide.md)
  - [Measured Serving Optimization User Guide](../optix_user_guide.md)
  - [Web UI User Guide](../msmodeling_web_ui_user_guide.md)
  - [TensorCast New Model Adaptation Development Guide](../msmodeling_tensor_cast_new_model_adaptation_user_guide.md)
  - [OptiX Plugin Developer Guide](../optix_plugin_user_guide.md)
- For experimental serving simulation capabilities, see [Fine-Grained Serving Simulation User Guide](../msmodeling_serving_cast_simulation_user_guide.md).

## Currently Supported Models

> Note: OptiX supports all models that can be deployed on target serving frameworks and is not limited to the model scope in the following table.

| Model Type | Model Family | Supported Models |
| --- | --- | --- |
| Text models | DeepSeek | DeepSeek V4, DeepSeek V3.2, DeepSeek V3 |
| Text models | Kimi | Kimi-K2.6, Kimi-K2.5, Kimi-K2 (supported through the DeepSeek V3 compatibility path) |
| Text models | Qwen | Qwen3.5, Qwen3.5 MoE, Qwen3-Next, Qwen3 Dense, Qwen3 MoE |
| Text models | GLM | GLM5.1, GLM5, GLM-4 MoE |
| Text models | ERNIE | ERNIE 4.5 MoE |
| Text models | Bailing/MiMo/MiniMax | Bailing MoE, MiMo v2 Flash, MiniMax M2 |
| Vision-language models | VL Models | Qwen3-VL, Qwen3-VL MoE, GLM-4V, GLM-4V MoE, InternVL |
| Video generation models | Diffusers DiT | Wan, HunyuanVideo, HunyuanVideo1.5 |

## Feature Support

### Model Inference Performance Simulation

| Feature | Description |
| --- | --- |
| Multi-hardware simulation | Supports built-in profiles for Ascend devices such as Atlas 800 A2/A3 and Atlas 350, as well as custom device profiles, estimating operator latency, communication overhead, and memory usage in multi-device scenarios without real hardware. |
| Phased LLM prefill/decode simulation | Differentiates the two compute paths of prefill and decode, and models the attention, KV cache growth, and per-token generation overhead of each phase. |
| Prefix cache simulation | Approximately models the prefill reuse benefit of prefix cache hits and evaluates the impact of the cache on first-token latency. |
| MTP speculative decoding simulation | Models the extra draft/verify computation of Multi-Token Prediction (MTP) and evaluates the impact of MTP on latency and throughput. |
| Compilation and graph optimization | Compiles and rewrites the forward computation graph, fusing typical subgraphs such as RMSNorm and Grouped Matmul into unified performance operators to more closely match the execution behavior in real deployments. |
| Multi-stream compute-communication overlap | In the compilation path, splits computation and communication across different execution streams through multi-stream scheduling, modeling inter-stream synchronization overhead and the end-to-end benefit of compute-communication overlap. |
| Quantization simulation | Models the quantization compute and memory-access cost of paths such as Linear, non-Expert Linear, LMHead, and Attention, and supports strategy combinations such as `W8A8`, `W4A8`, `FP8`, and `MXFP4`. |
| Parallelism and MoE extensions | Models the communication and compute overhead after splitting with global TP/DP/EP and fine-grained parallelism (Embedding TP, Vision TP, and so on), covering MoE deployment modes such as redundant experts and external/shared experts. |
| VL multimodal input | Incorporates visual inputs such as image batch and resolution, as well as the vision encoder parallelism strategy, into forward simulation, supporting joint analysis of text and multimodal inputs. |
| Model configuration sources | Supports local model directories and loading model configurations from remote sources such as Hugging Face and ModelScope. |
| Performance model switching | Supports Roofline-based analytical estimation and performance modeling based on measured data, allowing you to combine and compare results from different estimation paths as needed. |
| Chrome Trace/Debug | Outputs operator-level timeline, shape, graph structure, and bound analysis information for bottleneck identification, result validation, and visual analysis. |
| Video generation DiT simulation | Supports multi-step denoising simulation for Diffusers DiT video models such as Wan and HunyuanVideo, covering resolution, frame count, sampling steps, and quantization configuration. |
| Ulysses parallelism simulation | Models the attention communication and compute overhead under sequence-dimension Ulysses parallel splitting, supporting multi-device video DiT inference performance analysis. |
| CFG simulation | Models the forward overhead of the conditional and unconditional paths under Classifier-Free Guidance (CFG) and evaluates the impact of guidance on single-step denoising time. |
| Parallel CFG simulation | Models the cross-device collaboration and result aggregation behavior when the two CFG paths execute in parallel, and analyzes the benefit of parallel guidance over serial execution. |
| DiT cache simulation | Models the cache strategy that reuses intermediate results by sampling step and block range during denoising, and evaluates the impact of the effective cache range on total time. |

### Serving Performance Simulation

| Feature | Description |
| --- | --- |
| Throughput optimization under LLM/VLM constraints | Under service-level latency and service cost constraints such as TTFT and TPOT, automatically searches for the optimal parallel strategy, batch configuration, and token throughput that satisfy the SLO for LLM and VLM workloads. |
| PD aggregation | Jointly evaluates prefill and decode in the same instance to quickly obtain overall serving throughput and parallel configuration recommendations. |
| PD disaggregation | Searches for the optimal configurations of prefill and decode instances separately, suitable for evaluating PD disaggregated deployment scenarios. |
| PD ratio optimization | Searches for the optimal prefill/decode instance ratio under a fixed hardware scale, balancing resource investment and serving capability between the two instance types. |
| Parallel strategy search | Performs combinatorial search across dimensions such as TP, EP, and MoE-DP. |
| MTP configuration search | Searches configurations such as the MTP token count and acceptance rate to evaluate the impact of speculative decoding on serving throughput. |
| Batch and concurrency search | Automatically searches for batch size and request concurrency combinations that satisfy the SLO, considering service constraints such as the maximum number of batched tokens per step. |
| Chunked prefill simulation | When the effective input length exceeds the per-step prefill token budget, automatically splits the long prompt into multiple prefill chunks for step-by-step modeling, providing a more accurate evaluation of TTFT, P-phase throughput, aggregation scheduling behavior, and memory usage. This can be analyzed in combination with Prefix Cache. |
| Prefix Cache simulation | Supports modeling of the prefix cache hit rate and evaluates the impact of the cache on serving capability. |
| Variable-length load simulation | Evaluates serving capability at different request lengths based on variable-length input distributions. |
| Cross-hardware comparison | Under the same workload and SLO constraints, searches multiple chip profiles in parallel and outputs the optimal configuration and throughput comparison. |
| Compilation and fusion options | Enables optimization paths such as compilation, Sequence Parallel, MoE fusion, and Embedding TP during the optimization process to more closely match real serving deployment modes. |
| Multi-stream compute-communication overlap | In the compiled throughput optimization path, splits computation and communication across different execution streams through multi-stream scheduling, modeling inter-stream synchronization overhead and the end-to-end benefit of compute-communication overlap. |
| Result visualization and export | Outputs candidate configuration curves, raw optimization results, and operator-level traces for comparing different parallelism and batch combinations. |

### Fine-Grained Serving Simulation

> Note: This module is an experimental feature. See [Fine-Grained Serving Simulation User Guide](../msmodeling_serving_cast_simulation_user_guide.md).

| Feature | Description |
| --- | --- |
| YAML-driven serving simulation | Simulates the queueing, scheduling, and end-to-end serving process of multiple instances and requests based on instance groups and global service configuration. |
| PD aggregation/PD disaggregation topologies | Supports combinations of the aggregation and disaggregation instance roles, covering service simulation under different PD deployment topologies. |
| System metrics output | Aggregates request-level and system-level simulation results and outputs service metrics such as TTFT, TPOT, throughput, and E2E latency. |
| Profiling collection | Collects profiling data during simulation for further analysis of operator latency and system behavior. |

### Web UI

| Feature | Description |
| --- | --- |
| LLM/VL forward simulation | Configures LLM and VL forward simulation through page forms, supporting concurrency lists, TP lists, quantization, MTP, prefix cache, and operator/memory analysis. |
| Video generation simulation | Provides a visual configuration entry for video generation models, covering capabilities such as Ulysses, CFG, DiT Cache, quantization, and trace. |
| Throughput optimization experiments | Launches serving throughput optimization from the page, supporting three deployment modes: PD aggregation, PD disaggregation, and PD ratio, as well as multi-chip comparison. |
| Command preview and task cache | Supports previewing configuration results before running, and caches the results, logs, and detail cases of historical tasks. |
| Result display and export | Parses simulation logs and generates charts, tables, memory analysis, bandwidth bottlenecks, and operator details, supporting Excel export. |

### Measured Serving Optimization

| Feature | Description |
| --- | --- |
| Framework-based measurement optimization | Combines Particle Swarm Optimization (PSO) and Early Rejection to automatically search for the optimal deployment parameters that satisfy latency constraints on real serving frameworks. |
| Multiple engines and evaluation strategies | Supports inference engines such as vLLM and MindIE, as well as measurement-based optimization under multiple benchmark evaluation strategies. |
| Custom configuration and breakpoint resume | Supports custom optimization space configuration and can resume optimization tasks from checkpoints after an interruption. |
