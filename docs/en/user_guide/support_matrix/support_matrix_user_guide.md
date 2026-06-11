# Model and Feature Support Matrix

This document summarizes the model types and simulation features currently supported by TensorCast. For usage details, see [TensorCast User Guide](../msmodeling_tensor_cast_user_guide.md).

> [!NOTE]
> TensorCast is designed for performance modeling and resource estimation. Accuracy-related simulation or quantization results are for reference only. Final accuracy should be verified with real model evaluation.

## TensorCast Model Support

| Model Type | Model Family | Supported Models |
| --- | --- | --- |
| Text Models | DeepSeek | DeepSeek V3, DeepSeek V3.2, Kimi-K2 (supported through the DeepSeek V3 compatibility path) |
| Text Models | ERNIE | ERNIE 4.5 MoE |
| Text Models | GLM | GLM-4 MoE, GLM5 |
| Text Models | Qwen | Qwen3 MoE, Qwen3-Next, Qwen3.5, Qwen3.5 MoE |
| Text Models | Bailing / MiMo / MiniMax | Bailing MoE, MiMo v2 Flash, MiniMax M2 |
| Vision-Language Models | VL Models | GLM-4V MoE, InternVL, Qwen3-VL, Qwen3-VL MoE |
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
