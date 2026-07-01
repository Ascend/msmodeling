# Model and Feature Support Matrix

This document summarizes the currently supported model types and simulation features, making it easier to quickly understand the tool capability boundaries.

## Reading Notes

- If you only want to check the currently supported models, see "Currently Supported Models".
- If you need to check whether a CLI capability is available, see "Feature Support".
- If you need service-level throughput optimization or serving simulation capabilities, read this together with the [ServingCast User Guide](../msmodeling_serving_cast_user_guide.md).

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

| Module | Feature | Support | Notes |
| --- | --- | --- | --- |
| TensorCast | Multi-DeviceProfile Simulation | Supported | Select built-in or custom DeviceProfile through `--device`; supports `--num-devices` and `--reserved-memory-gb` |
| TensorCast | Prefill / Decode Simulation | Supported | Configure inference phases with `--query-length`, `--context-length`, and `--decode` |
| TensorCast | MTP Simulation | Supported | Configure MTP token count with `--num-mtp-tokens` |
| TensorCast | Torch Compile | Supported | Enable compile path with `--compile` and `--compile-allow-graph-break` |
| TensorCast | Quantization | Supported | Supports Linear Quantization, LMHead Quantization, and Attention Quantization |
| TensorCast | Chrome Trace / Debug | Supported | Supports `--chrome-trace`, `--graph-log-url`, `--dump-input-shapes`, and `--num-hidden-layers-override` |
| TensorCast | Parallelism and MoE Extensions | Supported | Supports TP / DP / EP, fine-grained TP / DP, Embedding TP, redundant experts, and external shared experts |
| TensorCast | VL Input and Model Source | Supported | Supports image batch / height / width configuration, and `huggingface` or `modelscope` via `--remote-source` |
| Throughput Optimizer | SLO-Constrained Optimization | Supported | Automatically searches optimal parallel strategy, batch, and throughput under TTFT / TPOT constraints |
| Throughput Optimizer | PD Modes | Supported | Supports PD aggregation, PD disaggregation, PD Ratio, and optimal Prefill / Decode instance ratio search |
| Throughput Optimizer | Cross-Hardware Comparison | Supported | Supports best-configuration search and cross-hardware comparison across multiple DeviceProfiles |
| ServingCast | Serving Simulation | Supported | Simulates multi-instance, multi-request end-to-end serving scenarios with YAML configuration |
| ServingCast | System Metrics | Supported | Outputs throughput, TTFT, TPOT, E2E latency, and other serving-level metrics |
| Web UI | Visual Configuration | Supported | Configure model, device, parallelism, quantization, and workload parameters through the page |
| Web UI | Result Display and Export | Supported | Supports charts, tables, detail analysis, history cache, and result export |
| OptiX | Serving Parameter Optimization | Supported | Uses the PSO particle swarm optimization algorithm to tune and validate parameters for serving frameworks such as vLLM and MindIE |
