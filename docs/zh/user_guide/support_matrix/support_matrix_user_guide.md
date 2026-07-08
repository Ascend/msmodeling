# 模型支持与特性支持矩阵

本文档汇总当前已支持的模型类型与仿真特性，便于快速确认工具能力边界。

## 阅读说明

- 如果只关心目前已经支持的模型，请查看“目前已经支持的模型”。
- 如果需要确认某项 CLI 能力是否可用，请查看“特性支持”。
- 如需了解各模块详细用法，请结合以下使用指南阅读：
  - 《[模型推理性能仿真 使用指南](../msmodeling_tensor_cast_user_guide.md)》
  - 《[服务化性能仿真 使用指南](../msmodeling_throughput_optimizer_user_guide.md)》
  - 《[服务化实测寻优 使用指南](../optix_user_guide.md)》
  - 《[Web UI 使用说明](../msmodeling_web_ui_user_guide.md)》
  - 《[TensorCast 新模型适配开发指导](../msmodeling_tensor_cast_new_model_adaptation_user_guide.md)》
  - 《[OptiX 插件开发指导](../optix_plugin_user_guide.md)》
- 如需了解实验性服务仿真能力，请结合《[服务化细粒度仿真 使用指南](../msmodeling_serving_cast_simulation_user_guide.md)》阅读。

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

### 模型推理性能仿真

| 特性 | 说明 |
| --- | --- |
| 多硬件仿真 | 支持 Atlas 800 A2/A3、Atlas 350 等昇腾设备内置画像及自定义设备画像，在无真实硬件条件下估算多卡场景下的算子耗时、通信开销与显存占用。 |
| LLM Prefill / Decode 分阶段仿真 | 区分预填充与解码两阶段计算路径，分别建模 attention、KV cache 增长与逐 token 生成开销。 |
| Prefix Cache 仿真 | 近似建模前缀缓存命中带来的 prefill 复用收益，评估缓存对首 token 时延的影响。 |
| MTP 投机解码仿真 | 建模 Multi-Token Prediction 的 draft/verify 额外计算，评估 MTP 对延迟与吞吐的影响。 |
| 编译与图优化 | 对前向计算图进行编译改写，将 RMSNorm、Grouped Matmul 等典型子图融合为统一性能算子，更贴近实际部署下的执行形态。 |
| 多流通算掩盖 | 在编译路径中通过多流调度将计算与通信拆分到不同执行流，建模流间同步开销与算通掩盖带来的端到端收益。 |
| 量化仿真 | 分别建模 Linear、非 Expert Linear、LMHead、Attention 等路径的量化计算与访存成本，支持 W8A8、W4A8、FP8、MXFP4 等多种策略组合。 |
| 并行与 MoE 扩展 | 建模全局 TP / DP / EP 及细粒度并行（Embedding TP、Vision TP 等）切分后的通信与计算开销，覆盖 redundant experts、external/shared expert 等 MoE 部署形态。 |
| VL 多模态输入 | 将图像 batch、分辨率等视觉输入与视觉编码器并行策略纳入前向仿真，支持文本与多模态联合分析。 |
| 模型配置来源 | 支持本地模型目录，以及从 Hugging Face、ModelScope 等远程源加载模型配置。 |
| 性能模型切换 | 支持基于 Roofline 的解析估算与基于实测数据的性能建模，可按需组合对比不同估算路径的结果。 |
| Chrome Trace / Debug | 输出算子级 timeline、shape、图结构与 bound 分析信息，用于瓶颈定位、结果校验与可视化分析。 |
| 视频生成 DiT 仿真 | 支持 Wan、HunyuanVideo 等 Diffusers DiT 视频模型的多步去噪仿真，覆盖分辨率、帧数、采样步数与量化配置。 |
| Ulysses 并行仿真 | 建模序列维度 Ulysses 并行切分下的 attention 通信与计算开销，支持多卡视频 DiT 推理性能分析。 |
| CFG 仿真 | 建模 Classifier-Free Guidance 条件下条件路与非条件路的前向开销，评估 guidance 对单步去噪耗时的影响。 |
| CFG 并行仿真 | 建模 CFG 双路并行执行时的跨卡协同与结果汇聚行为，分析并行 guidance 相对串行执行的收益。 |
| DiT Cache 仿真 | 建模去噪过程中按采样 step 与 block 范围复用中间结果的 cache 策略，评估 cache 生效区间对总耗时的影响。 |

### 服务化性能仿真

| 特性 | 说明 |
| --- | --- |
| LLM / VLM 约束下吞吐优化 | 在 TTFT、TPOT 等服务级时延与服务成本约束下，对 LLM 与 VLM 自动搜索满足 SLO 的最优并行策略、batch 配置与 token 吞吐。 |
| PD 混部 | 在同一实例中联合评估 Prefill 与 Decode，快速获得整体服务吞吐与并行配置建议。 |
| PD 分离 | 分别搜索 Prefill 与 Decode 实例的最优配置，适用于 PD 分离部署场景评估。 |
| PD 配比寻优 | 在固定硬件规模下搜索最优 Prefill / Decode 实例配比，平衡两类实例的资源投入与服务能力。 |
| 并行策略搜索 | 在 TP、EP、MOE-DP 等维度上进行组合搜索。 |
| MTP 配置搜索 | 搜索 MTP token 数量、接受率等配置，评估投机解码对服务吞吐的影响。 |
| Batch 与并发搜索 | 自动搜索满足 SLO 的 batch 大小与请求并发组合，并考虑单步最大 batched tokens 等服务约束。 |
| Chunked Prefill 仿真 | 当有效输入长度超过单步 prefill token 预算时，自动将长 prompt 拆分为多个 prefill chunk 分步建模，更准确评估 TTFT、P 阶段吞吐、混部调度行为与显存占用；可与 Prefix Cache 组合分析。 |
| Prefix Cache 仿真 | 支持前缀缓存命中率建模，评估缓存对服务能力的影响。 |
| 变长负载仿真 | 基于变长输入分布评估不同请求长度下的服务能力。 |
| 跨硬件对比 | 在同一 workload 与 SLO 约束下，对多种芯片画像并行搜索并输出最优配置与吞吐对比。 |
| 编译与融合选项 | 在寻优过程中启用编译、Sequence Parallel、MoE 融合与 Embedding TP 等优化路径，更贴近真实服务部署形态。 |
| 多流通算掩盖 | 在启用编译的吞吐寻优路径中，通过多流调度将计算与通信拆分到不同执行流，建模流间同步开销与算通掩盖带来的端到端收益。 |
| 结果可视化与导出 | 输出候选配置曲线、原始寻优结果与算子级 trace，便于对比不同并行与 batch 组合。 |

### 服务化细粒度仿真

> 说明：本模块为实验性功能，详见《[服务化细粒度仿真 使用指南](../msmodeling_serving_cast_simulation_user_guide.md)》。

| 特性 | 说明 |
| --- | --- |
| YAML 驱动服务仿真 | 基于实例组与服务全局配置，模拟多实例、多请求的排队调度与端到端 serving 流程。 |
| PD 混部 / PD 分离拓扑 | 支持混部、分离两类实例角色组合，覆盖不同 PD 部署拓扑下的服务仿真。 |
| 系统指标输出 | 聚合请求级与系统级仿真结果，输出 TTFT、TPOT、吞吐与 E2E 延迟等服务指标。 |
| Profiling 采集 | 在仿真过程中采集 profiling 数据，用于进一步分析算子耗时与系统行为。 |

### Web UI

| 特性 | 说明 |
| --- | --- |
| LLM / VL 前向仿真 | 以页面表单方式配置 LLM 与 VL 前向仿真，支持并发列表、TP 列表、量化、MTP、Prefix Cache 与算子/显存分析。 |
| 视频生成仿真 | 提供视频生成模型的可视化配置入口，覆盖 Ulysses、CFG、DiT Cache、量化与 trace 等能力。 |
| 吞吐寻优实验 | 以页面方式发起服务吞吐寻优，支持 PD 混部、PD 分离、PD 配比三种部署模式与多芯片对比。 |
| 命令预览与任务缓存 | 支持运行前预览配置结果，并缓存历史任务的结果、日志与明细 case。 |
| 结果展示与导出 | 解析仿真日志并生成曲线、表格、显存分析、带宽瓶颈与算子明细，支持 Excel 导出。 |

### 服务化实测寻优

| 特性 | 说明 |
| --- | --- |
| 服务框架实测寻优 | 结合 PSO 粒子寻优与 Early Rejection，在真实服务框架上自动搜索满足时延约束的最优部署参数。 |
| 多引擎与评测策略 | 支持 vLLM、MindIE 等推理引擎，以及多种 benchmark 评测策略下的实测寻优。 |
| 自定义配置与断点续跑 | 支持自定义寻优空间配置，并可在中断后从检查点继续寻优任务。 |
