# RFC: Throughput Optimizer 多硬件工作负载复用

## 元数据

| 项目 | 内容 |
|:---|:---|
| **状态** | Draft |
| **作者** | jia_ya_nan, Codex |
| **创建日期** | 2026-07-31 |
| **更新日期** | 2026-08-03 |
| **相关模块** | `cli/inference/throughput_optimizer.py`、`serving_cast/parallel_runner.py`、`serving_cast/service/`、`tensor_cast/runtime.py` |
| **相关 RFC** | [PD 配比吞吐量寻优](rfc_pd_ratio_throughput_optimization_zh.md)、[Pipeline Parallel 仿真支持](rfc_pipeline_parallel_support_zh.md) |

---

## 1. 概述

`throughput_optimizer` 已支持一次通过 `--device DEVICE [DEVICE ...]` 比较多个 `DeviceProfile`。当前入口
`run_multi_device_loop()` 逐个 profile 创建 `ParallelRunner`；每个 runner 再为每个并行配置创建
`ModelRunner` 并执行完整 forward。即使模型、请求 shape、batch、逻辑并行策略和负载完全相同，模型解析、输入构造、
算子捕获、计算量/访存量/通信量统计及内存生命周期分析仍会被重复执行。

本 RFC 将一次 forward 的仿真拆为两个显式阶段：

1. **工作负载分析（hardware-independent analysis）**：构造模型和输入，捕获逻辑算子序列、算子性能属性、通信语义、
   流/依赖关系以及内存生命周期；产物为可重放的 `WorkloadTrace`。
2. **硬件性能估算（hardware-specific estimation）**：对同一个 `WorkloadTrace` 绑定目标 `DeviceProfile` 和对应
   performance model，估算计算、访存和通信时间，计算硬件容量相关内存余量，并继续由既有调度器求得 TTFT、TPOT 和吞吐。

复用的最小单位不是一次完整优化，而是一个可精确定义的 forward workload（包括 Prefill/Decode、并发量和形状）。
这样不同硬件因内存或 SLO 导致二分搜索访问不同 batch 时，只有交集 workload 命中缓存；不共享的 case 仍会按需分析，
不会为了预热而穷举所有 batch。

## 2. 目标与非目标

### 2.1 目标

- 同一 CLI 进程的多硬件运行中，相同 `WorkloadKey` 只分析一次；其他 profile 复用同一份工作量、访存量、逻辑通信量
  和推理阶段信息。
- 每个 profile 必须独立执行其 performance model 和内存容量计算，保留独立的性能结果、early-stop 和跨硬件对比表。
- 覆盖 aggregation、disaggregation Prefill、disaggregation Decode、chunked Prefill、PD ratio 和已有的并行搜索。
- 对 analytic 及 profiling performance model 均复用**工作负载**，但不复用某个 profile 的 latency 或 profiling 查询结果。
- 在同次调用拥有至少两个 DeviceProfile 时默认启用进程级内存缓存、single-flight 去重和命中统计；单硬件路径保持原有行为。
- 用 schema 版本、模型/并行/量化配置和精确 request shape 识别可复用 workload；无法证明硬件无关时宁可绕过复用。
- 复用实现不得降低已有 `--jobs` 的硬件估算/并行搜索并发度；候选与基线必须使用相同的搜索空间和 jobs 进行比较。
- 在至少两个 profile、多个重复输入 case 的基准中，相比关闭复用的原有路径取得不低于 15% 的端到端收益。

### 2.2 非目标

- 本期不建立跨 CLI 进程或跨机器的磁盘缓存；缓存生命周期仅限本次 `throughput_optimizer` 进程。
- 本期不改变 DeviceProfile、通信算法、性能模型公式、并行搜索空间或 OptimizerSummary 的指标口径。
- 不承诺不同 `WorkloadKey`（例如不同 batch、有效输入长度、逻辑并行策略）的复用，也不以近似 hash 换取命中率。
- 不把同名 device profile 当作可互换 profile；每个 profile 都必须重新估时和验算可用内存。

## 3. 现状与问题定位

当前链路如下：

```text
CLI --device A B
  -> run_multi_device_loop()
       -> A: ParallelRunner -> ProcessPool -> ModelRunner.run_inference()
       -> B: ParallelRunner -> ProcessPool -> ModelRunner.run_inference()  # 再次完整 forward
  -> cross-hardware summary
```

`ModelRunner.run_inference()` 在 `Runtime` 中记录 `OpInvokeInfo`，随后 `Runtime.replay_op_invoke_infos()` 对每个 op
调用当前 profile 的 `PerformanceModel.process_op()`；`MemoryTracker` 同时依据事件和张量 def-use 关系分析峰值内存。
`BaseThroughputOptimizer._forward_record_cache` 仅缓存单个 optimizer 实例中的 `ForwardLatencyRecord`，其中包含已绑定
hardware 的延迟和可用内存，既不能跨 profile 共享，也无法证明其适用于其他设备。

问题不是现有 forward-shape cache 缺少一个更大的字典，而是缓存层级过高：它缓存的是硬件结果，而需要复用的是其下游的
逻辑 workload。因此本 RFC 不共享 `ForwardLatencyRecord`，而引入其前置、无 profile 的 `WorkloadTrace`。

## 4. 核心语义与复用边界

### 4.1 两阶段契约

| 阶段 | 输入 | 产物 | 是否包含 DeviceProfile |
|:---|:---|:---|:---:|
| WorkloadAnalyzer | 模型及其 revision/config、请求/shape、逻辑并行、量化/编译/图改写配置 | `WorkloadTrace`、逻辑内存画像、阶段元数据 | 否 |
| HardwareEstimator | `WorkloadTrace`、目标 `DeviceProfile`、performance-model 配置和数据源 | 每 profile 的 event results、breakdown、执行时间、可用内存 | 是 |
| Serving scheduler | 各 forward 的 profile-specific `ForwardLatencyRecord` | TTFT、TPOT、token/s、SLO/OOM 结论 | 是 |

`WorkloadTrace` 必须保留顺序和依赖，而不是只保存 FLOPs 汇总。原因是同一总 FLOPs 在不同 stream、collective、通信 group、
pipeline stage 或 fusion 关系下会有不同总延迟。trace 中的每个事件至少包含：

- 可重建的 op 描述（operator、参数/输出的 tensor shape、dtype、stride、标量参数和必要的逻辑 group 信息）；
- `OpInvokeInfo.PerformanceProperties`：按 dtype 的 MMA/GP ops、read/write/readwrite bytes、static cost；
- 逻辑通信类型、消息大小、逻辑 rank/group，以及 stream/dependency token；
- region/reference 关系、内存 alias/def-use 元数据和峰值动态内存画像；
- Prefill/Decode、request shape、并发量、chunk 位置、MTP 语义等阶段元数据。

Estimator 由 trace 重建 performance-model 所需的 op invocation，再调用目标 profile 的现有 analytic 或 empirical estimator。
因此设备 A 的性能结果、通信算法选择、profiling CSV 查询、capacity 和 OOM 结论绝不会进入 device B 的缓存值。

### 4.2 可复用条件

只有完全相同的 `WorkloadKey` 才可复用。key 不包含 device 名称、理论算力、HBM 容量、链路带宽和通信拓扑参数；这些属于
`EstimationKey`。它必须包含以下 hardware-independent 输入：

| 类别 | 必须纳入 key 的字段 |
|:---|:---|
| 实现版本 | `WORKLOAD_TRACE_SCHEMA_VERSION` 和影响 workload 的编译配置 |
| 模型 | 规范化 model source、量化配置以及影响模型图的模型参数 |
| 逻辑并行 | world size、TP/DP/PP/EP/MoE-TP/MoE-DP、MTP、word/vision embedding 并行及会改变图的相关开关 |
| 请求 | Prefill/Decode、model concurrency、`query_len`、`seq_len`、图像 batch/高/宽、length-distribution 的规范化摘要 |
| workload 行为 | effective prefix-cache 结果、chunk plan、动态 shape、compile/graph break、所有 compilation passes、输入生成和 scheduler 语义版本 |

SLO、设备内存大小、reserved memory、性能模型类型、profiling database 路径以及 device name 不进入 `WorkloadKey`；它们只影响
搜索路径或估算结果。`ForwardShapeKey` 保留为上层索引，但不再单独视为可复用性的充分条件。

### 4.3 设备敏感保护

部分未来能力可能在图构造期基于物理拓扑选择不同的 collective 拆分、kernel/fusion 或模型结构。为避免把这类 case 错误当作
硬件无关，`WorkloadAnalyzer` 输出 `reuse_eligibility`：

- `reusable`：trace 只包含逻辑通信和与 profile 无关的图语义，可跨 profile 估算。
- `topology_scoped`：key 追加 topology-structure 约束，只在结构等价的 profile 间共享。
- `disabled`：无法安全冻结/重建 workload；本 case 退回旧的完整仿真路径并记录原因。

首期必须覆盖当前 optimizer 使用的 normal runtime、chunked Prefill、variable-length Prefill、MTP 和 PP runtime。
任何新模型 wrapper 或 runtime 特性未实现可重放序列化时，默认标记为 `disabled`，不得静默复用。

## 5. 方案设计

### 5.1 组件与数据模型

新增以下内部组件（命名为建议，可在实现设计阶段调整）：

```text
serving_cast/service/
  workload_cache.py             # WorkloadKey、缓存、single-flight、统计和失效校验
  workload_execution.py         # Analyzer/Estimator 适配 ModelRunner metrics
tensor_cast/
  runtime_workload.py           # RuntimeWorkloadTrace 的冻结、重建及 schema
```

关键数据结构：

```python
@dataclass(frozen=True)
class WorkloadKey:
    digest: str
    schema_version: int
    logical_execution_fingerprint: str
    request_fingerprint: str

@dataclass(frozen=True)
class RuntimeWorkloadTrace:
    key: WorkloadKey
    events: tuple[FrozenRuntimeEvent, ...]
    memory_plan: FrozenMemoryPlan
    static_model_memory_bytes: int
    phase_metadata: WorkloadPhaseMetadata
    reuse_eligibility: ReuseEligibility

@dataclass(frozen=True)
class EstimationKey:
    workload_digest: str
    device_fingerprint: str
    performance_model_fingerprint: str
```

最终形态的 `FrozenRuntimeEvent` 是不可变、可 pickle 的 IR，不能持有原始 FakeTensor 或 `DeviceProfile` 对象。它保存重建
estimator 所需的 tensor metadata 与逻辑值；这样一个 trace 可由 workload cache service 安全传给任意 profile 的估算 worker，
而不会因 Python 对象身份、原始 tensor 存储或某一 worker 的全局状态而失效。

`FrozenRuntimeEvent` 的实现必须使用独立的 tensor metadata 描述和逻辑 object id 重建 estimator 输入，不能将
`OpInvokeInfo`、FakeTensor、region 或 `Runtime` 实例直接放入跨进程缓存。原型的进程内原始对象复用仅用于验证可行性，
不得作为默认运行路径。

### 5.2 Runtime 与 ModelRunner 改造

现有 `Runtime` 的 recording 与 replay 混在一次 `__exit__()` 中。本 RFC 将其重构为以下可保持旧 API 的内部步骤：

1. `record_program()`：执行模型 fake forward，记录 `OpInvokeInfo`、region 和 multistream anchor。
2. `freeze_workload()`：解析 op properties、通信逻辑、memory def-use/alias 和逻辑峰值，产出版本化 `RuntimeWorkloadTrace`。
3. `estimate(trace, device_profile, perf_models)`：重建事件，调用现有 performance models，生成 event results、breakdown 和
   profile-specific memory availability。

`ModelRunner.run_inference()` 仍返回 `ModelRunnerMetrics`，以避免 Agg/Disagg 优化器重写。新增内部入口
`analyze_inference_workload()` 和 `estimate_inference_workload()`；旧入口等价于“分析后立刻用本 device 估算”，故单硬件行为
保持兼容。PP 的 `PipelineRunner` 也必须提供同样的 trace/estimate 分界，确保 stage schedule 和 logical send/recv 进入 trace。

`MemoryTracker` 的 device-capacity 计算与逻辑 allocation plan 分离：trace 保存峰值动态内存、KV/activation/alias 相关量；
estimator 再结合当前 profile 的 `memory_size_bytes`、当前权重大小和 `reserved_memory_gb` 计算
`device_memory_available_gb`。这保证小内存 profile 仍可独立 OOM，而不污染其他 profile。

### 5.3 缓存与并发模型

缓存范围为**本次 CLI 进程**。`run_multi_device_loop()` 创建一个 `WorkloadCacheCoordinator`，所有 profile 的
`ParallelRunner` 和其 `ProcessPoolExecutor` worker 都通过 IPC 访问它；现有 `--jobs` 值保持不变。协调器是唯一的 miss owner：

```text
hardware workers (A / B / ...)
   -> get_or_analyze(WorkloadKey)
        -> HIT: 返回 FrozenTrace
        -> WAIT: 等待相同 key 的 in-flight Future
        -> MISS: 由 coordinator 的 capture worker 执行一次分析，发布 FrozenTrace
   -> HardwareEstimator(profile) -> ForwardLatencyRecord(profile)
   -> existing optimizer/scheduler -> summary(profile)
```

协调器使用 `Future`/condition 实现 single-flight，因此即使 Prefill 与 Decode、或两个 profile 的 worker 同时请求同一 key，
也只会触发一次分析。capture 由独立、受控的 worker 执行；默认并发上限为 1，以隔离模型全局配置和 `torch.compile` 状态。
硬件估算与 optimizer 搜索继续在原有多个 worker 中并行执行，且只接收 `FrozenWorkloadTrace`，不构建模型、不执行 forward、
不触发 `torch.compile`。实现可在后续通过无状态 capture worker 放宽 capture 并发度，不能破坏“一 key 一次”的保证。

冻结 trace 使用 IPC 传递；worker 可保留小型只读 LRU 减少重复反序列化，但命中、失效和 in-flight 状态以 coordinator 为准。
协调器按条目数和序列化后字节数实行 LRU；达到上限时只淘汰无等待者的已完成条目。淘汰只带来重新分析，不影响正确性。

### 5.4 Optimizer 接入

- `BaseThroughputOptimizer._compute_forward_latency_record()` 由直接调用 `_get_forward_info()` 改为请求
  `WorkloadExecutionProvider`；provider 在未命中时分析，在每个调用 profile 上估时，并生成与当前完全相同的
  `ForwardLatencyRecord`。
- 保留 `_forward_record_cache`，但其 key 升级为 `EstimationKey` 或等价的 profile-aware key，防止把 device A 的
  latency record 返回给 device B。
- `ForwardLatencyTable`、Agg 的 scheduler、Disagg wave/chunk 逻辑和 `PDRatioThroughputOptimizer` 继续消费
  `ForwardLatencyRecord`，不感知 workload cache 的存在。
- `run_multi_device_loop()` 不再以“一个 profile 一套完整仿真”为边界，而是以“多 profile 共享一个 coordinator”为边界；
  现有每 profile 的 `OptimizerSummary`、终端输出和 `render_cross_hardware_summary()` 保持不变。
- profiling mode 在估算阶段为每个 profile 创建自己的 `ProfilingDataSource`/`EmpiricalPerformanceModel`；只共享 trace，
  不共享其 `op_records`、插值 cache 或 CSV 结果。

### 5.5 CLI、日志与可观测性

现有多硬件用法不变：

```bash
python -m cli.inference.throughput_optimizer \
  --device TEST_DEVICE ATLAS_800_A3_752T_128G_DIE \
  --num-devices 8 --input-length 2048 --output-length 512 \
  --tp-sizes 1 2 4 8 Qwen/Qwen3-32B
```

不新增或修改用户可见 CLI 参数。多硬件调用默认创建 coordinator；其 frozen trace LRU 使用实现内置的 128 条目上限，
并在运行摘要中输出实际占用字节数。缓存策略如需调优，作为内部实现配置处理，不扩展当前 CLI 表面。

每次运行结束输出不含业务指标的 cache 摘要，例如：

```text
Workload cache: enabled, capture_jobs=1, estimate_jobs=8, analysis=42, hit=126,
  wait=8, bypass=1, evicted=0, bytes=96.4 MiB, analysis_time=18.21s,
  estimation_time=4.83s
```

对每个 bypass 仅记录一次结构化 reason（如 `profile-dependent graph selection`、`unsupported frozen op` 或
`cache version mismatch`）。其中 `estimate_jobs` 必须等于用户传入（或现有默认）的 `--jobs`，用于证明复用没有把估算或
搜索串行化。

### 5.6 版本、配置和失效校验

虽然首期缓存不落盘，仍必须在取值前执行如下校验，确保 workload schema 和请求/配置边界明确，并为未来持久化缓存保留契约：

1. schema version 不一致：拒绝条目；
2. model、逻辑并行、量化/编译配置或 request shape 不一致：拒绝条目；
3. reuse eligibility 与目标 profile 的 topology structure 不兼容：拒绝条目或使用 topology-scoped key；
4. trace 反序列化、op rebuild 或 estimator 验证失败：删除该条目，降级为一次完整仿真，并在 debug 日志记录。

缓存版本失效是正常 miss，不是用户可见错误。任何 digest 计算失败、模型 revision 不可得或检测到不支持的可变全局状态时，
默认 bypass，不允许使用宽松 key。

## 6. 运行流程

```text
CLI parse / validate --device A B ...
  |
  +-- create WorkloadCacheCoordinator (one per CLI invocation)
  |
  +-- optimizer requests forward case
        |  WorkloadKey = hash(model + logical parallel + exact request + schema)
        |
        +-- cache miss: Analyzer records model/input once -> Frozen WorkloadTrace
        |                 (compute, memory, logical comm, phase and dependency metadata)
        |
        +-- for each DeviceProfile:
              HardwareEstimator(WorkloadTrace, DeviceProfile, perf model)
                -> device-specific duration/breakdowns/memory availability
                -> existing ForwardLatencyRecord cache
                -> Agg / Disagg / PD-ratio scheduling
                -> independent OptimizerSummary and cross-hardware rows
```

对输入分布和 chunked Prefill，optimizer 生成的每个 representative composition 或 chunk/wave shape 都是独立的
`WorkloadKey`。对 PD ratio，Prefill 侧和 Decode 侧分别请求自己的 workload；若不同 profile 上的 P 或 D case 相同，
仍共享对应 trace。因设备容量导致的 early-stop 不会取消其他 profile 的分析/估算。

## 7. 一致性、兼容性与回滚

结果的一致性策略是“重放同一逻辑 trace 到原 performance model”，不是重新实现性能公式。对于给定 profile，启用复用的
`execution_time_s`、breakdown、memory fields 和 scheduler 输入应与优化前的逐 profile 完整仿真一致；验收允许汇总指标有
不超过 1% 的差异，以覆盖浮点聚合顺序差异。实现阶段若某 op/event 无法完全复放，必须 bypass 该 workload，而不是使用近似汇总。

同次调用拥有至少两个 device target 时，shared reuse 默认启用；单 profile 保留现有本地 forward cache 行为，避免引入
无收益的跨进程传输。回滚通过回退到引入本特性前的版本完成，不保留额外 CLI 分支，避免扩大用户接口。

`--chrome-trace`、`--dump-op-bound-results` 和 empirical metrics 必须来自每个 profile 的 estimate replay；输出路径仍按
当前多硬件命名规则区分 profile，不能复用 capture trace 作为 profile 的性能 trace。

## 8. 备选方案与取舍

| 方案 | 结论 | 原因 |
|:---|:---|:---|
| 只扩大 `ForwardLatencyRecord` 缓存 | 不采用 | record 已含 latency、breakdown 和 memory availability，天然绑定 DeviceProfile。 |
| 仅按 FLOPs/bytes 汇总再套 roofline | 不采用 | 会丢失算子、通信算法、stream、依赖和 PP/融合语义，无法满足 1% 一致性目标。 |
| 每个 profile 各自保留 op-trace cache | 不采用 | 不会减少多 profile 的首次分析，无法满足“一次分析”验收。 |
| 进程内原始 `Runtime` 对象 + `InlineExecutor` | 明确不采用 | 原始对象不能安全跨进程传递；为共享它而替换 `ProcessPoolExecutor` 会取消 `--jobs` 并行。3.1 的 235B case 已出现首 profile 约五分钟未完成的回退。 |
| 进程级 frozen trace + profile estimator（本 RFC） | 采用 | 复用边界准确，保留现有 estimator 和 scheduler，可控地兼容分析/估算并行。 |
| 跨进程持久化磁盘缓存 | 后续考虑 | 需要模型制品、权限、容量、跨版本清理和安全设计，不是首期性能目标的必要条件。 |

## 9. 实施计划

1. **Runtime 基础设施**：定义可 pickle 的 frozen trace schema 和 `Runtime` 的
   capture/estimate 分界；冻结事件只能使用 tensor metadata 和逻辑 object id，禁止引用 FakeTensor、region、Runtime 或
   DeviceProfile。以单 profile direct path 的 capture+replay 等价测试为门槛。
2. **工作负载缓存**：实现独立 coordinator、single-flight capture worker、序列化 trace LRU、失效验证、统计和 bypass
   机制；覆盖并发 miss、capture 失败与重试。明确 IPC（pipe、共享内存或等价机制）的传输和内存所有权。
3. **Serving 接入**：为 `BaseThroughputOptimizer`、`ForwardLatencyTable` 和 `ParallelRunner` 注入 provider；estimate 和
   optimizer search 继续使用现有 `ProcessPoolExecutor` 与原 `--jobs` 并发，保持 Agg/Disagg 接口和输出不变。
4. **多硬件编排**：使 `run_multi_device_loop()` 默认共享 coordinator，并实现每 profile trace 输出和含
   `capture_jobs` / `estimate_jobs` 的统计摘要；不得引入 `InlineExecutor` 顺序回放路径。
5. **验证与发布门禁**：补齐测试矩阵，以优化前同机同版本结果建立基线并记录收益和一致性报告；性能门禁通过后将多硬件场景
   的默认行为固定为开启。

预计涉及文件：

| 文件 | 改动 |
|:---|:---|
| `tensor_cast/runtime.py`、新增 `tensor_cast/runtime_workload.py` | 捕获、冻结及按 profile 重放 runtime workload。 |
| `tensor_cast/core/model_runner.py` | 新增分析/估算入口并保持 `ModelRunnerMetrics` 兼容。 |
| `tensor_cast/performance_model/`、`memory_tracker.py` | 支持从 frozen event 估算与逻辑内存计划/容量计算分离。 |
| `serving_cast/service/workload_cache.py`、`workload_execution.py` | key、缓存、coordinator、统计和 provider。 |
| `serving_cast/service/base_throughput_optimizer.py`、`latency_table.py` | 以 workload provider 生成 profile-aware latency record。 |
| `serving_cast/parallel_runner.py`、`optimizer_curve_plots.py` | 跨 profile 共享 coordinator 和任务编排。 |
| `cli/inference/throughput_optimizer.py` | 保持 CLI 兼容，不新增复用专用参数。 |
| `tests/regression/...`、`tests/smoke/...` | 单测、回归、集成和性能基准。 |

## 10. 测试与验收设计

### 10.1 正确性测试

| 场景 | 验证点 |
|:---|:---|
| 二个 profile、相同 Prefill forward case | analyzer spy 仅调用一次；两个 estimator 均调用一次；两个结果带各自 device 名称。 |
| Agg（含 chunked Prefill） | 启用/关闭复用的 TTFT、TPOT、token/s、Prefill/Decode breakdown 逐 profile 差异不超过 1%。 |
| Disagg Prefill / Decode | 变量输入分布、prefix cache、MTP 和 OOM 分支均按 key 分离且结果一致。 |
| PD ratio | P/D 独立 workload 能跨 profile 命中；PD ratio、balanced QPS 和每硬件结果与关闭复用一致。 |
| TP/DP/PP/EP/MoE 搜索 | 不同逻辑并行或 world size 必须产生不同 key；相同配置必须命中。 |
| analytic 与 profiling | 不共享 profile-specific result；profile B 必须调用自身的 analytic/profiling estimator。 |
| topology-sensitive/bypass | 不兼容 eligibility 时零共享且日志含原因，结果等于完整路径。 |

### 10.2 失效与稳健性测试

- 修改 schema、source、model、量化/编译、request shape、chunk plan、输入分布、图像参数、prefix-cache、MTP 或并行参数时
  必须 miss；只修改 device 性能/容量时必须继续命中 workload、但重算 latency/memory。
- 两个并发请求同一 key 时，capture spy 必须等于 1；capture 失败时所有 waiter 获得确定失败并允许后续重试。
- LRU 淘汰后重新请求应重新分析；缓存淘汰不得改变结果。
- frozen trace 不能保留 DeviceProfile、原始 tensor identity 或不可 pickle 的 worker 对象。

### 10.3 性能验收方法

在固定 commit、同一 `.venv`、关闭 Chrome trace 的环境执行至少三次 warm run，报告中位数。基线与候选必须使用完全相同的
`--jobs`、模型 revision、DeviceProfile、搜索空间、输入 case、预热策略和日志级别；候选不得将任一 estimate 或 optimizer
search worker 串行化。使用两个及以上已注册的 DeviceProfile、多个重复 input case、相同模型/长度/并行搜索空间，比较：

```bash
# 基线：优化前 commit 的原有逐 profile 完整仿真
python -m cli.inference.throughput_optimizer ... --device A B

# 候选：包含本特性的 commit，默认启用复用
python -m cli.inference.throughput_optimizer ... --device A B
```

记录总耗时、analysis/estimation 耗时、unique workload 数、hit 数和峰值 cache 大小。候选运行时间须比基线至少低 15%；
同时逐 profile 对齐最优行及阶段级指标，差异不得超过 1%。如果目标模型的 analysis 时间占比不足以得到 15%，报告必须说明
该模型的占比、case 数与收益限制，不能以未测量结论代替基准证据。

重点基准至少包含以下双 profile case，并在记录中保留 baseline/candidate 的完整命令和三个 warm-run 的原始耗时：

```bash
python -m cli.inference.throughput_optimizer Qwen/Qwen3-235B-A22B \
  --device ATLAS_350_425T_84G ATLAS_350_425T_112G \
  --quantize-linear-action W8A8_DYNAMIC --quantize-attention-action INT8 \
  --num-devices 4 --input-length 3500 --output-length 1024 \
  --prefix-cache-hit-rate 0 --tpot-limits 50 --compile \
  --reserved-memory-gb 0 --log-level info
```

## 11. 风险与缓解措施

| 风险 | 缓解措施 |
|:---|:---|
| frozen IR 漏掉影响 estimator 的参数 | 不做 FLOPs 近似；通过 direct-vs-replay 回归和无法重建即 bypass 的策略保证正确性。 |
| trace 内存过大 | 条目/字节双限额、LRU、统计输出；首期不持久化。 |
| 多进程与 Torch/compile 全局状态冲突 | coordinator 集中 capture、默认单分析 worker；worker 只做无状态 estimate replay。 |
| 为共享对象牺牲 `--jobs` 并发 | 禁止传递原始 Runtime 和引入 `InlineExecutor`；仅传递 frozen trace，estimate/search 保持原 `ProcessPoolExecutor` 并发，并以相同 jobs 的基准作为发布门禁。 |
| 设备在建图期影响图语义 | eligibility 判定、topology-scoped key 和默认 bypass。 |
| profiling 数据库设备相关 | trace 可共享、每 profile 的 `ProfilingDataSource`/cache 绝不共享。 |
| 调试难以判断命中来源 | 结构化 hit/miss/bypass reason、结束摘要，以及与优化前 commit 的对比基线。 |
