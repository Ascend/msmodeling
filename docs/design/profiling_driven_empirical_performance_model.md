# Design Document: 工具支持基于实测算子性能的建模

## Revision History (修订记录)

| Date (日期)  | Version (修订版本) | Change Description (修改描述) | Author (作者) | RFC Document (RFC文档) |
| ---------- | -------------- | ------------------------- | ----------- | -------------------- |
| 2026-06-12 | 1.0            | 初稿完成，工具支持基于实测算子性能的建模      | -           | -                    |
|            |                |                           |             |                      |
|            |                |                           |             |                      |

---

## 1. Background (背景描述)

当前msmodeling主要基于roofline进行算子性能估算，虽能覆盖通用场景下的性能分析需求，但是在面对特定硬件平台、复杂算子实现及组合场景时，估算与真实评估性能存在一定偏差。因此需要建立一套基于实测性能数据的建模能力体系，直接读取并使用内置算子性能数据进行性能计算。

## 2. Design (方案设计)

## 2.1 整体架构

```text
┌────────────────────────────────────────────────────────────────────────────┐
│                           TensorCast Runtime                               │
│                                                                            │
│  ┌─────────────────────┐     ┌──────────────────────────────────────────┐  │
│  │  Runtime            │     │  用户可配置选择 PerformanceModel         │  │
│  │  (TorchDispatchMode)│────▶│                                          │  │
│  │                     │     │  ┌──────────────────────────────────┐    │  │
│  │  拦截所有算子调用   │     │  │ EmpiricalPerformanceModel        │    │  │
│  │  生成 OpInvokeInfo  │     │  │  DataSource.lookup(OpInvokeInfo) │    │  │
│  └─────────────────────┘     │  │  未命中 → fallback (Analytic)    │    │  │
│                              │  └──────────────────────────────────┘    │  │
│                              │  ┌──────────────────────────────────┐    │  │
│                              │  │ AnalyticPerformanceModel（现有） │    │  │
│                              │  │  Roofline 理论模型               │    │  │
│                              │  └──────────────────────────────────┘    │  │
│                              └──────────────────────────────────────────┘  │
│                                           │                                │
└───────────────────────────────────────────┼────────────────────────────────┘
                                            │查询优先级：精确匹配 → 插值（未来扩展） → 外推（未来扩展） 
                      ┌─────────────────────┼────────────────────────┐
                      │                     ▼                        │
                      │       DataSource（抽象接口）                 │
                      │  ┌──────────────────────────────────────┐    │
                      │  │  ProfilingDataSource                 │    │
                      │  │  加载op_mapping.yaml+{KernelType}.csv│    │
                      │  │  查询逻辑func_name → mapping         │    │
                      │  │             →CSV row→latency_us      │    │
                      │  └──────────────────────────────────────┘    │
                      │  ┌──────────────────────────────────────┐    │
                      │  │  InterpolatingDataSource（未来扩展） │    │
                      │  └──────────────────────────────────────┘    │
                      │  ┌───────────────────────────────────────┐   │
                      │  │  存储层                               │   │
                      │  │  计算: vllm_ascend/{version}/*.csv    │   │
                      │  │  通信: hccl/{cann_version}/*.csv      │   │
                      │  └───────────────────────────────────────┘   │
                      └──────────────────────────────────────────────┘

```

## 2.2 模块结构

```tezt
tensor_cast/performance_model/
├── analytic.py                           # AnalyticPerformanceModel（现有）
├── comm_analytic.py                      # CommAnalyticModel（现有）
├── memory_tracker.py                     # MemoryTracker（现有）
├── empirical.py                          # 新增EmpiricalPerformanceModel处理算子逻辑
└── profiling_database/                   # 新增：DataSource + 数据存储
    ├── __init__.py
    ├── data_source.py                    # DataSource ABC + QueryResult
    ├── profiling_data_source.py          # ProfilingDataSource，进行算子数据查询
    ├── interpolating_data_source.py      # InterpolatingDataSource，插值计算（能力启用待未来扩展）
    └── data/                             # 性能数据存储
        └── {设备名}/
           ├── vllm_ascend/
           │   └── vllm{vllm_asecend版本}_torch{pytorch版本}_cann{CANN版本}/ # 存放某设备类型下，各软件版本的性能数据库
           │       ├── op_mapping.yaml    # 记录从TensorCast虚拟算子映射到真实设备profiling数据中NPU内核类型的yaml文件
           │       └── {KernelType}.csv   # 维护每个kernel算子的性能数据库
           └── hccl/{CANN版本}/           # 通信算子性能数据库（和 CANN 版本绑定，跨 vLLM 版本复用）


```

## 2.3 核心模块设计

### 2.3.1 EmpiricalPerformanceModel

类的主要功能：接受 `DataSource` 实例，`process_op()` 先查性能数据库数据，未命中回退至 `fallback_model`（默认 `AnalyticPerformanceModel`），使用roofline模型结果。

```python
# tensor_cast/performance_model/empirical.py
class EmpiricalPerformanceModel(PerformanceModel):
    def __init__(self, device_profile, data_source: DataSource,
                 fallback_model: Optional[PerformanceModel] = None):
        self.data_source = data_source
        self.fallback_model = fallback_model

    @property
    def fallback_model(self) -> PerformanceModel:
        if self._fallback_model is None:
            from .analytic import AnalyticPerformanceModel
            self._fallback_model = AnalyticPerformanceModel(self.device_profile)
        return self._fallback_model

    @override
    def process_op(self, op_invoke_info):
        # process_op主要逻辑
        result = self.data_source.lookup(op_invoke_info)
        if result is not None:
            return Result(execution_time_s=result.latency_us * 1e-6, ...)
        return self.fallback_model.process_op(op_invoke_info)
```

### 2.3.2 DataSource抽象基类

```python
# tensor_cast/performance_model/profiligdatabase/data_source.py
class DataSource(ABC):
    """通用数据源抽象基类。
    TensorCast 只通过 OpInvokeInfo 查询，不感知底层映射关系和数据格式。"""

    @abstractmethod
    def lookup(self, op_invoke_info: "OpInvokeInfo") -> Optional[QueryResult]:
        """
        从 OpInvokeInfo 查询算子性能。
        返回 None 表示未命中，
        返回QuerySource.MEASURED或QuerySource.INTERPOLATED表示查询时命中，使用QueryResult.latency_us作为结果
        返回QuerySource.PARTIAL表示查询一些可拆分的复杂算子时，当decomposer场景中部分命中，使用已经累加的partial经验时长作为结果
        """
        ...
```

### 2.3.3 ProfilingDataSource

`ProfilingDataSource`类继承自`DataSource`，主要功能为读取配置文件，加载性能数据库信息，分派算子查询路径。核心函数`lookup`方法，会去掉 `torch.ops.` 前缀，得到规范化算子名，然后在 `operator_mappings.yaml` 中查找算子的映射配置。分派查询路径的顺序如下：

| 路径                | `op_mapping.yaml` 触发条件          | 核心方法                  | 返回行为                                                                                                                                 |
| ----------------- | ------------------------------- | --------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| Composite         | `composite: true`               | `_lookup_composite`   | 复合算子的查询方式，按照先用定义的decomposer再查配置文件的顺序获取分解的子算子，再分别获取各个计算算子与通信算子的时长并累加。当decomposer查询子算子的场景，当部分子算子MISS 时可返回 `PARTIAL`。                   |
| Communication     | `category: communication`       | `_lookup_comm`        | 查询通信算子时长。通过通信量、device个数、通信拓扑层级进行查找数据，若可以精准查询，则返回对应时长，结果类型为`MEASURED`；若无法查询到，根据device个数与通信拓扑层级进行查找，以通信量进行插值推测，最终返回结果类型为`INTERPOLATED` |
| Attention special | `query_mode: attention_special` | `_lookup_attention`   | 目前主要用于算子 `FusedInferAttentionScore`；存在 `alternate_kernel_types` 时按优先级查询对应算子在attention相关维度的数据。                                        |
| Elementwise       | `query_mode: elementwise`       | `_lookup_elementwise` | 适用于`aten.mul`等逐元素操作的算子。按输出 shape 严格匹配，也支持按 dtype 字节比缩放时长。                                                                            |
| MoE fused         | `query_mode: moe_fused`         | `_lookup_moe`         | 目前主要用于算子DispatchFFNCombine。根据shape与ep_size进行匹配查询数据。                                                                                  |
| Zero cost         | `zero_cost: true`               | 视为算子耗时为0              | 返回 `0.0 us`，设为精准查询。                                                                                                                  |
| Accepted miss     | `accepted_miss: <reason>`       | 视为算子耗时为0              | 返回 `0.0 us` 和相应解释说明。用于 TensorCast 中存在、但 NPU profiling 中没有独立 kernel 的算子（如TensorCast存在，但是NPU profiling时已经被融到其他算子中的场景）。                 |
| Compute           | 默认路径，配置 `kernel_type`           | `_lookup_compute`     | 按优先级查询 `alternate_kernel_types`中对应算子对应shape的数据。                                                                                      |

## 3. Usage Instructions (使用说明)

### 3.1 CLI接口

在 `cli/inference/text_generate.py` 中新增参数

```python
    parser.add_argument("--performance-model",
                        action="append",
                        default=None,
                        help="性能模型类型，可多次指定。"
                             "'analytic': Roofline 模型（默认，无需数据）。"
                             "'profiling': 基于实测 Profiling 数据库的 EmpiricalPerformanceModel "
                             "（需要 --profiling-database）。")
    parser.add_argument("--profiling-database", type=str, default=None,
                        help="性能数据库路径（profiling 模式生效），"
                             "指向包含 op_mapping.yaml 和 CSV 数据文件的目录")
    # 默认值处理
    if args.performance_model is None:
        args.performance_model = ["analytic"]
```

| CLI 选项                 | 是否必选 | 默认值        | 说明                                                                                                                                                                                                        |
| ---------------------- | ---- | ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--performance-model`  | 否    | `analytic` | 指定仿真建模的性能模型类型，模型类型有`analytic`和`profiling`。当性能模型为`profiling`时，需同时指定参数`--profiling-database`。两种性能模型均不需要物理NPU设备。一次仿真可指定一个或多个性能模型，当指定多个性能模型时使用方法为`--performance-model analytic --performance-model profiling` |
| `--profiling-database` | 否    | None       | 性能数据库路径（`--performance-model`为profiling 模式时生效）                                                                                                                                                            |

## 4. Test Design (测试设计)

### 4.1 单元测试

运行以下测试用例，均可通过

- pytest tests\regression\tensor_cast\test_empirical.py

- pytest tests\benchmark\ops\perf_database\test_empirical.py

- pytest tests\benchmark\ops\perf_database\test_profiling_data_source.py

### 4.2 集成测试

参见3.1 CLI接口，运行text_generate命令时，指定参数`--performance-model profiling`和 `--profiling-database $DATA_DIR`，使用性能数据库中实测数据接入仿真建模，精度差距小于20%。
