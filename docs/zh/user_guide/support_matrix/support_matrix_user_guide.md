# 模型支持与特性支持矩阵

本文档汇总当前已支持的模型类型与仿真特性，便于快速确认工具能力边界。

## 阅读说明

- 如果只关心目前已经支持的模型，请查看“目前已经支持的模型”。
- 如果需要确认某项 CLI 能力是否可用，请查看“特性支持”。
- 如果需要服务级吞吐优化或服务仿真能力，请结合《[ServingCast 使用指南](../msmodeling_serving_cast_user_guide.md)》阅读。

## 目前已经支持的模型

> 说明：OptiX 支持所有可在目标服务框架中部署的模型，不受下表模型范围限制。

| 模型类型 | 模型系列 | 支持模型 |
| --- | --- | --- |
| 文本模型 | DeepSeek 系列 | DeepSeek V4、DeepSeek V3.2、DeepSeek V3 |
| 文本模型 | Kimi 系列 | Kimi-K2.6、Kimi-K2.5、Kimi-K2（通过 DeepSeek V3 兼容路径支持） |
| 文本模型 | Qwen 系列 | Qwen3.5、Qwen3.5 MoE、Qwen3-Next、Qwen3 Dense、Qwen3 MoE |
| 文本模型 | GLM 系列 | GLM5.1、GLM5、GLM-4 MoE |
| 文本模型 | ERNIE 系列 | ERNIE 4.5 MoE |
| 文本模型 | Bailing / MiMo / MiniMax | Bailing MoE、MiMo v2 Flash、MiniMax M2 |
| 视觉-语言模型 | VL 模型 | Qwen3-VL、Qwen3-VL MoE、GLM-4V、GLM-4V MoE、InternVL |
| 视频生成模型 | Diffusers DiT 视频生成模型 | Wan、HunyuanVideo、HunyuanVideo1.5 |

## 特性支持

| 模块 | 特性 | 支持情况 | 说明 |
| --- | --- | --- | --- |
| TensorCast | 多 DeviceProfile 仿真 | 支持 | 通过 `--device` 选择内置或自定义 DeviceProfile；支持 `--num-devices` 与 `--reserved-memory-gb` |
| TensorCast | Prefill / Decode 仿真 | 支持 | 通过 `--query-length`、`--context-length` 与 `--decode` 配置不同推理阶段 |
| TensorCast | MTP 仿真 | 支持 | 通过 `--num-mtp-tokens` 配置 MTP token 数 |
| TensorCast | Torch Compile | 支持 | 通过 `--compile` 与 `--compile-allow-graph-break` 启用编译路径 |
| TensorCast | Quantization | 支持 | 支持 Linear Quantization、LMHead Quantization 与 Attention Quantization |
| TensorCast | Chrome Trace / Debug | 支持 | 支持 `--chrome-trace`、`--graph-log-url`、`--dump-input-shapes` 与 `--num-hidden-layers-override` |
| TensorCast | 并行与 MoE 扩展 | 支持 | 支持 TP / DP / EP、细粒度 TP / DP、Embedding TP、redundant experts 与 external shared experts |
| TensorCast | VL 输入与模型来源 | 支持 | 支持 image batch / height / width 配置，并可通过 `--remote-source` 选择 `huggingface` 或 `modelscope` |
| Throughput Optimizer | SLO 约束寻优 | 支持 | 基于 TTFT / TPOT 约束自动搜索最优并行策略、batch 与吞吐 |
| Throughput Optimizer | PD 模式 | 支持 | 支持 PD 混部、PD 分离、PD 配比，并可寻找最优 Prefill / Decode 实例配比 |
| Throughput Optimizer | 跨硬件对比 | 支持 | 支持多 DeviceProfile 下的最优配置搜索与 cross-hardware comparison |
| ServingCast | 服务级仿真 | 支持 | 基于 YAML 配置模拟多实例、多请求 end-to-end serving 场景 |
| ServingCast | 系统指标输出 | 支持 | 输出吞吐、TTFT、TPOT、E2E latency 等服务级指标 |
| Web UI | 可视化配置 | 支持 | 支持通过页面配置模型、芯片、并行、量化和 workload 参数 |
| Web UI | 结果展示与导出 | 支持 | 支持曲线、表格、明细分析、历史缓存与结果导出 |
| OptiX | 服务化参数寻优 | 支持 | 基于 PSO 粒子寻优算法对 vLLM、MindIE 等服务框架进行参数寻优与验证 |
