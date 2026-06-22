# RFC: Chunked Prefill Latency 查表优化方案

# 1. 概述

| 项目 | 内容 |
| --- | --- |
| 状态 (Status) | Draft |
| 作者 (Authors) | @jiayanan, Codex |
| 创建日期 (Created) | 2026-06-06 |
| 更新日期 (Updated) | 2026-06-06 |
| 相关 Issue/PR | TBD |

---

## 1.1 简介

本 RFC 提议为 `throughput_optimizer` 的 Chunked Prefill 建模路径引入 Forward Latency 查表机制：在进入调度和吞吐计算前，先收集当前候选并发下会用到的 prefill / decode forward shape，按顺序精确计算并缓存 `ModelRunner` 结果，后续调度循环只查表读取 latency、memory 和 breakdown。若任意 forward record 的 `memory_left_gb < 0`，当前候选立即停止，不再计算后续 chunk / decode shape。

该方案解决长上下文、小 `max_batched_tokens` 场景下优化器运行时间被重复 `ModelRunner.run_inference()` 调用放大的问题。例如输入长度为 128k、chunk size 为 1k 时，单请求会产生 128 个 prefill chunk；这 128 个不同 `seq_len` 的 shape 仍需逐个精确建模，但同一个候选或同一个 optimizer 实例内不应重复建模相同 shape。若再叠加并发搜索、PD 分离 prefill wave 和调度循环，当前朴素实现还会出现大量重复 forward 建模。查表优化不改变 Chunked Prefill 的语义，只优化吞吐优化器自身的计算方式。

## 1.2 动机

Chunked Prefill 首版建模已经允许 `effective_input_length > max_batched_tokens` 的长 prompt 继续仿真，但其性能成本随 chunk 数增长：

```text
K = ceil(effective_input_length / max_batched_tokens)
```

当 `input_length = 128k`、`max_batched_tokens = 1k` 时，`K = 128`。在精确建模下，每个 chunk 的 `seq_len` 不同：

```text
(query_len=1k, seq_len=1k)
(query_len=1k, seq_len=2k)
...
(query_len=1k, seq_len=128k)
```

这些 shape 不能简单合并为一次完整 prefill，也不能只用第一个 chunk 的 latency 乘以 128，因为 attention 的上下文长度逐 chunk 增长。因此精确建模不能减少唯一 chunk shape 的数量。另一方面，同一个候选 batch size / concurrency 下，调度公式会反复访问相同 shape；在 PD 分离 prefill 路径中，同一个 chunk shape 还可能按 wave 多次调用 `ModelRunner`。这使优化器运行时间远高于必要成本。

不做该优化的影响：

- 长上下文极小 chunk 场景下，`throughput_optimizer` 搜索耗时显著增加。
- PD Ratio 优化需要分别搜索 P / D 资源，P 阶段的 chunked prefill 成本会被进一步放大。
- 用户为了让优化器跑得动，可能被迫增大 `max_batched_tokens` 或缩小搜索空间，从而影响配置评估质量。

## 1.3 目标

目标：

- 为 Chunked Prefill 引入精确 latency table，预先计算当前调度计划需要的唯一 forward shape。
- 让 `AggThroughputOptimizer` 和 `DisaggThroughputOptimizer` 复用统一查表能力。
- 精确查表保持现有建模结果不变，只减少重复 `ModelRunner` 调用。
- 负 memory 早停作为默认行为，不新增 Chunked Prefill 专用开关参数。

非目标：

- 不把多个 chunk 合并成一次完整 prefill 计算。
- 不改变 Chunked Prefill 的 TTFT / TPOT / throughput 语义。
- 不复刻某个线上推理后端的全部 scheduler 细节。
- 不修改 tensor_cast 底层 op 建模和 `ModelRunner` 基础接口。

# 2. 用例分析

## 2.1 长上下文 Chunked Prefill 搜索

用户希望评估如下场景：

```text
input_length = 131072
output_length = 1024
max_batched_tokens = 1024
deployment = aggregation
```

优化器应保持 128 个 prefill chunk 的精确语义，但不应在同一个候选并发下对相同 `(query_len, seq_len, concurrency)` 重复调用 `ModelRunner`。调度循环只从查表结果读取 latency；若某个 chunk 已经 `memory_left_gb < 0`，后续 chunk 不再继续建模。

## 2.2 PD 分离 prefill 阶段

PD 分离 prefill 当前按 chunk 和 wave 累加 latency。若 `wave_concurrency` 对同一 chunk shape 多次出现，应该只计算一次：

```text
ForwardShapeKey(
  is_decode=False,
  model_concurrency=wave_concurrency,
  query_len=chunk.query_len,
  seq_len=chunk.seq_len,
)
```

后续 wave 只查表累加。

## 2.3 PD Ratio 优化

PD Ratio 优化会依赖 P 阶段 QPS：

```text
P QPS = p_concurrency / ttft * 1000
```

当 P 阶段使用 Chunked Prefill 时，查表优化应降低 P 侧搜索耗时；D 阶段继续复用 decode latency cache。最终 PD ratio 计算公式不变。

## 2.4 DFX 要求

- 兼容性：精确查表结果应与当前逐步调用结果一致。
- 可维护性：shape key、table 构建和查表逻辑独立封装，避免散落在 agg / disagg 公式中。
- 可测试性：可通过 fake `ModelRunner` 验证唯一 shape 只计算一次。
- 可靠性：table key 必须覆盖所有影响 forward shape 的动态字段，避免错误复用。
- 性能：长上下文下，重复 `ModelRunner` 调用次数应从按 step / wave 计数下降到按唯一 shape 计数；OOM 候选应在首次负 memory 后提前停止。

# 3. 方案设计

## 3.1 总体方案

### 3.1.1 msmodeling 当前状态

当前 Chunked Prefill 相关路径：

```text
serving_cast/service/utils.py
serving_cast/service/base_throughput_optimizer.py
serving_cast/service/agg_throughput_optimizer.py
serving_cast/service/disagg_throughput_optimizer.py
cli/inference/throughput_optimizer.py
```

当前实现中：

- `OptimizerData.get_prefill_chunk_plan()` 负责生成 chunk plan。
- `BaseThroughputOptimizer._get_forward_info()` 将单个 shape 转成 `RequestInfo` 并调用 `ModelRunner.run_inference()`。
- `AggThroughputOptimizer` 内部已有 `_get_or_compute_latency()`，但 cache 逻辑只在 agg 路径中使用。
- `DisaggThroughputOptimizer` chunked prefill 路径直接循环 `_get_forward_info()`，同一 shape 在多个 wave 中可能重复计算。
- agg chunked prefill 的时间模拟循环会按 step 查询 latency，虽然已有 cache，但还没有“先生成调度计划、再批量预热 shape 表”的统一机制。

### 3.1.2 推荐方案

新增统一的 Forward Latency Table：

```text
OptimizerData + chunk plan
  -> 生成调度计划或 prefill wave 计划
  -> 收集唯一 ForwardShapeKey
  -> ForwardLatencyTable(..., optimizer_data).prefetch(keys)
  -> 调度/吞吐计算阶段 table.get(key)
  -> 输出 TTFT / TPOT / throughput / memory / breakdown
```

核心原则：

- 精确查表只改变执行方式，不改变 latency 来源。
- table 以当前 optimizer 配置为作用域，模型配置、设备、量化配置和 parallel config 必须一致。
- record 在单个 optimizer 实例内复用，避免二分 / 指数搜索不同 batch size 时重复计算相同 chunk shape。
- shape key 显式包含会影响 `RequestInfo` 的动态 shape 信息。
- agg / disagg 统一通过 table 获取 forward metrics。
- `--jobs` 继续作为 optimizer 外层搜索任务的并行预算，不作为 latency table 的早停开关。

### 3.1.3 ForwardShapeKey

新增不可变 key：

```python
@dataclass(frozen=True)
class ForwardShapeKey:
    is_decode: bool
    model_concurrency: int
    query_len: int
    seq_len: int
    image_batch_size: int | None = None
    image_height: int | None = None
    image_width: int | None = None
```

字段说明：

- `is_decode`：区分 prefill 和 decode。
- `model_concurrency`：传给 `RequestInfo.concurrency` 的模型级并发数。
- `query_len`：本次 forward 新计算 token 数。
- `seq_len`：本次 forward 的上下文长度。
- image 字段：用于 VL 模型，避免不同图像输入复用同一 text-only shape。

不放入 key 的字段：

- model id、device、quantization、parallel config：由当前 table 所属 optimizer 配置固定。
- `max_batched_tokens`：只影响调度和 chunk plan，最终通过 `query_len` / `seq_len` 体现。
- prefix cache hit rate：只影响 `effective_input_length` 和 chunk plan，最终也通过 shape 体现。

### 3.1.4 ForwardLatencyTable

新增 table 抽象：

```python
@dataclass
class ForwardLatencyRecord:
    latency_ms: float
    memory_left_gb: float
    breakdowns: str
    raw_breakdowns: dict[str, dict[str, float]]

class ForwardLatencyTable:
    def __init__(self, optimizer: BaseThroughputOptimizer, optimizer_data: OptimizerData):
        self.optimizer = optimizer
        self.optimizer_data = optimizer_data
        self.memory_exceeded_key: ForwardShapeKey | None = None
        self.records: dict[ForwardShapeKey, ForwardLatencyRecord] = {}

    def prefetch(self, keys: Iterable[ForwardShapeKey]) -> None:
        missing_keys = [key for key in unique(keys) if key not in self.records]
        for key in missing_keys:
            record = self._compute(key)
            self.records[key] = record
            if record.memory_left_gb < 0:
                self.memory_exceeded_key = key
                break

    def get(self, key: ForwardShapeKey) -> ForwardLatencyRecord:
        if key not in self.records:
            self.prefetch([key])
        if key not in self.records:
            raise RuntimeError(...)
        return self.records[key]
```

`_compute(key)` 复用现有 `_get_forward_info()`：

```text
ForwardShapeKey
  -> RequestInfo(...)
  -> ModelRunner.run_inference(...)
  -> ForwardLatencyRecord
```

decode 仍需应用 MTP acceptance rate：

```text
decode_latency_ms = raw_latency_ms / (sum(mtp_acceptance_rate[:num_mtp_tokens]) + 1)
```

因此 `ForwardLatencyRecord.latency_ms` 保存 raw forward latency，不缓存已经按 MTP acceptance rate 折算后的
decode latency。读取 decode record 时，按当前 `OptimizerData` 动态折算，避免同一 shape 在不同 MTP 配置下误复用。

预取规则：

- `prefetch()` 按 key 顺序串行计算，行为与普通精确查表一致。
- negative-memory 早停默认开启且不可关闭，确保首次 `memory_left_gb < 0` 后不会启动后续 key 的计算。
- `--jobs` 只控制外层搜索任务并行，不传递到 latency table。

### 3.1.5 Disagg prefill 查表流程

disagg prefill 不需要复杂调度计划，chunk 和 wave 可直接枚举：

```text
keys = []
for chunk in chunk_plan:
  wave_size = max(max_batched_tokens // chunk.query_len, 1)
  remaining = concurrency
  while remaining > 0:
    wave_concurrency = min(wave_size, remaining)
    keys.append(ForwardShapeKey(False, wave_concurrency, chunk.query_len, chunk.seq_len))
    remaining -= wave_concurrency

table.prefetch(keys)

latency_ms = serving_cost
for key in keys:
  record = table.get(key)
  latency_ms += record.latency_ms
  memory_left = min(memory_left, record.memory_left_gb)
  if record.memory_left_gb < 0:
    break
```

breakdown 聚合保留原有 wave 等权语义：每个 wave 先基于 `raw_breakdowns` 归一化为比例，再对所有 wave
求平均。重复 wave 可以复用同一 table record，但仍按实际 wave 次数参与 breakdown 平均。

收益：

- 原先同一 key 可能在多个 wave 中重复调用 `_get_forward_info()`。
- 查表后同一 key 只计算一次。
- 对 `wave_concurrency=1` 的 128k / 1k 场景，计算次数从 `128 * concurrency` 下降到 `128`。

### 3.1.6 Agg chunked prefill 查表流程

agg 模式需要调度 prefill 和 decode 的混部 step。为实现“一次性把 chunk 都跑了”，将当前时间模拟拆成两段：

```text
阶段 1：生成逻辑调度计划，只产生 shape key，不计算 latency
阶段 2：prefetch 唯一 key
阶段 3：replay 调度计划，通过 table.get(key) 累计时间和指标
```

调度计划结构：

```python
@dataclass(frozen=True)
class ScheduleStep:
    prefill_key: ForwardShapeKey | None
    decode_key: ForwardShapeKey | None
    p_step: int
    d_step: int
```

由于默认 `Scheduler.decide()` 只依赖：

```text
ready_decode
pending_prefill
chunk_query_len
max_batched_tokens
```

不依赖 latency，因此可以先进行 dry-run，生成完整 step plan 和唯一 key 集合。replay 阶段再读取真实 latency：

```text
prefill_latency = table.get(prefill_key).latency_ms if prefill_key else 0
decode_latency = table.get(decode_key).latency_ms if decode_key else 0
step_latency = scheduler.step_latency(prefill_latency, decode_latency)
current_time += step_latency
```

这样可以保留现有 TTFT / TPOT / makespan 统计逻辑，同时将 `ModelRunner` 调用集中到 prefetch 阶段。

### 3.1.7 精确查表

精确查表：

- 计算所有唯一 `ForwardShapeKey`。
- 作为默认且唯一的查表机制。
- 输出结果与当前逐步调用一致。
- 适用于最终结果和所有 chunked prefill 搜索场景。

### 3.1.8 输出结果

建议新增或复用以下输出字段：

```text
prefill_num_chunks
max_batched_tokens
effective_input_length
chunked_prefill_forward_shape_count
```

建议在日志中输出：

```text
Chunked prefill latency table: enabled
Prefill chunks: 128
Forward shapes: 128
Stop on negative memory: true
```

## 3.2 技术选型（可选）

### 方案 A：统一精确 latency table（推荐首版必选）

优点：

- 不改变建模结果。
- 可以同时修复 agg / disagg 的重复调用问题。
- 实现边界清晰，容易单测。

缺点：

- 对 128k / 1k 这类场景仍需计算约 128 个不同 prefill shape；只靠去重不能减少唯一 shape 数。
- 唯一 shape 仍会按顺序预取；这是保证首次负 memory 后不再启动后续计算的必要约束。

### 方案 B：`--jobs` 并行预取精确查表（不推荐）

优点：

- 不改变建模语义，仍对每个唯一 shape 调用 `ModelRunner` 做精确建模。
- 对 128k / 1k 这类唯一 shape 多的场景，可以降低查表预取阶段的 wall-clock 时间。
- 复用现有 `--jobs`，用户无需学习新参数。

缺点：

- 总计算量不变，只是并行执行。
- 多 worker 可能增加模型构建开销和内存占用。
- 需要协调外层搜索并行和内层 table prefetch 并行，避免嵌套并行过载。
- 无法保证首次 `memory_left_gb < 0` 后不再启动后续 key 的计算，因此不作为当前实现方案。

### 方案 C：把所有 chunk 合并成一个 `run_inference(List[RequestInfo])`（不推荐）

优点：

- 表面上只调用一次 `ModelRunner`。

缺点：

- 语义错误：这会把 128 个串行 prefill chunk 变成一个 batch 中的 128 个 request。
- TTFT、KV cache 增长、step latency 和 scheduler 行为都会被改变。
- 不能作为 Chunked Prefill 的精确建模。

## 3.3 安全隐私与 DFX 设计

兼容性：

- 精确查表输出 TTFT、TPOT、吞吐、memory 应与逐步调用结果一致。
- `effective_input_length <= max_batched_tokens` 的单 chunk 场景保持现有路径和结果。
- 不改变 `--max-batched-tokens`、prefix cache、MTP、PD ratio 现有语义。

可靠性：

- table key 必须覆盖所有影响 forward shape 的动态字段。
- `ForwardShapeKey` 必须包含 VL 图像相关 shape 字段，避免旧 text-only cache key 误复用图像输入结果。
- MTP acceptance rate 不放入 `ForwardShapeKey`；record 缓存 raw latency，decode 读取时再按当前 MTP 配置折算。
- 任意 forward record 的 `memory_left_gb < 0` 时，当前候选应提前停止，不继续计算或 replay 后续 chunk / decode shape。

性能：

- 精确查表应保证同一 `ForwardShapeKey` 只调用一次 `ModelRunner`。
- disagg prefill 的重复 wave 应复用同一 record。
- agg 时间模拟应先 prefetch，再 replay，避免循环中反复触发 compute。
- 精确查表通过去重和 optimizer 实例级 record cache 降低重复建模成本。

可维护性：

- table 放在 `serving_cast/service` 公共层，agg / disagg 只构造 key 和读取 record。
- `_get_forward_info()` 继续负责底层 `RequestInfo` 构造，避免 duplicated request shape 逻辑。
- `--jobs` 继续由 outer runner 管理，用于搜索任务并行。

## 3.4 编程与调用设计

### 3.4.1 编程模型基本设计

涉及模块：

| 模块 | 变更 |
| --- | --- |
| `serving_cast/service/base_throughput_optimizer.py` | 提供统一 `_make_forward_shape_key()` / `_compute_forward_latency_record()` |
| `serving_cast/service/latency_table.py` | 新增 `ForwardShapeKey`、`ForwardLatencyRecord`、`ForwardLatencyTable` |
| `serving_cast/service/agg_throughput_optimizer.py` | chunked prefill 时间模拟拆分为 plan / prefetch / replay |
| `serving_cast/service/disagg_throughput_optimizer.py` | chunked prefill wave 先收集 key，再查表累加 |
| `tests/regression/serving_cast/test_service/` | 增加查表、agg/disagg 行为测试 |

开发约束：

- 使用现有 `RequestInfo` 和 `ModelRunner.run_inference()`。
- 精确查表不能改变模型 forward shape。
- 默认配置必须保持结果兼容。
- 不新增 CLI 参数或 table-level 早停开关。
- latency table 预取保持顺序执行，保证 OOM 早停不会启动后续 shape。

验收标准：

- disagg chunked prefill 中，同一 key 的 fake forward 调用次数为 1。
- agg chunked prefill 中，调度 replay 阶段不再触发新的 `ModelRunner` 调用。
- 精确查表下现有 chunked prefill UT 结果不变。
- 任意 record `memory_left_gb < 0` 后不再计算后续 missing keys。

### 3.4.2 接口定义与设计

#### 3.4.2.1 `ForwardShapeKey`

接口描述：描述一次 forward latency 建模所需的唯一 shape。

接口原型：

```python
@dataclass(frozen=True)
class ForwardShapeKey:
    is_decode: bool
    model_concurrency: int
    query_len: int
    seq_len: int
    image_batch_size: int | None = None
    image_height: int | None = None
    image_width: int | None = None
```

输入/输出参数：

| 参数名称 | 输入/输出 | 类型 | 描述 | 取值范围 |
| --- | --- | --- | --- | --- |
| `is_decode` | 输入 | bool | 是否为 decode forward | `True` / `False` |
| `model_concurrency` | 输入 | int | 模型级并发 | `> 0` |
| `query_len` | 输入 | int | 新计算 token 数 | `> 0` |
| `seq_len` | 输入 | int | 上下文长度 | `>= query_len` |
| `image_batch_size` | 输入 | int \| None | VL 图像 batch size | `None` 或 `> 0` |
| `image_height` | 输入 | int \| None | VL 图像高度 | `None` 或 `> 0` |
| `image_width` | 输入 | int \| None | VL 图像宽度 | `None` 或 `> 0` |

#### 3.4.2.2 `ForwardLatencyTable`

接口描述：批量预热并查询 forward latency record。

接口原型：

```python
class ForwardLatencyTable:
    def __init__(self, optimizer: BaseThroughputOptimizer, optimizer_data: OptimizerData) -> None: ...
    def prefetch(self, keys: Iterable[ForwardShapeKey]) -> None: ...
    def get(self, key: ForwardShapeKey) -> ForwardLatencyRecord: ...
```

输入/输出参数：

| 参数名称 | 输入/输出 | 类型 | 描述 | 取值范围 |
| --- | --- | --- | --- | --- |
| `optimizer_data` | 输入 | OptimizerData | 本次查询所需的 MTP、图像输入和调度配置 | 当前 optimizer run 内固定 |
| `keys` | 输入 | Iterable | 需要预热的 shape key | 非空或空集合 |
| `key` | 输入 | ForwardShapeKey | 需要查询的 shape key | 已预热或可 lazy compute |
| `ForwardLatencyRecord.latency_ms` | 输出 | float | raw forward latency，单位 ms；decode MTP 折算在读取时完成 | `>= 0` |
| `ForwardLatencyRecord.memory_left_gb` | 输出 | float | 设备剩余显存，单位 GiB | 任意 float |
| `ForwardLatencyRecord.breakdowns` | 输出 | str | 性能 breakdown 文本 | 字符串 |
| `ForwardLatencyRecord.raw_breakdowns` | 输出 | dict | 性能 breakdown 原始数据，用于 disagg 按 wave 聚合 | 字典 |

异常处理：

- `model_concurrency <= 0`、`query_len <= 0`、`seq_len < query_len`：抛出 `ValueError`。

### 3.4.3 使用说明

默认精确查表：

```bash
python -m cli.inference.throughput_optimizer Qwen/Qwen3-32B \
  --device TEST_DEVICE \
  --num-devices 1 \
  --input-length 131072 \
  --output-length 1024 \
  --max-batched-tokens 1024
```

使用 `--jobs` 并行外层搜索任务：

```bash
python -m cli.inference.throughput_optimizer Qwen/Qwen3-32B \
  --device TEST_DEVICE \
  --num-devices 1 \
  --input-length 131072 \
  --output-length 1024 \
  --max-batched-tokens 1024 \
  --jobs 8
```

PD 分离 prefill：

```bash
python -m cli.inference.throughput_optimizer Qwen/Qwen3-32B \
  --device TEST_DEVICE \
  --num-devices 4 \
  --input-length 131072 \
  --output-length 1024 \
  --max-batched-tokens 1024 \
  --ttft-limits 1000 \
  --disagg
```

# 4. 测试设计

单元测试：

- `ForwardShapeKey`
  - 相同字段生成相同 key。
  - 不同 `query_len` / `seq_len` / `model_concurrency` 不会误命中。
  - VL image 字段不同时不会误命中。
- `ForwardLatencyTable`
  - `prefetch()` 对重复 key 只调用一次 `_compute()`。
  - `get()` 对已存在 key 不触发 `_compute()`。
  - `get()` 对不存在 key 可 lazy compute。
  - 非法 key 抛出明确异常。
  - 按 key 顺序串行计算 missing keys。
  - 任意 record `memory_left_gb < 0` 后不再计算后续 missing keys。

disagg 测试：

- `input_length=10, max_batched_tokens=4, concurrency=4` 时，重复 wave 复用同一 key。
- fake forward 验证 `_get_forward_info()` 调用次数等于唯一 key 数。
- 精确查表下输出 `ttft` 与原逐步调用一致。
- memory 使用所有 record 的最小 `memory_left_gb`。
- 任意 wave 的 `memory_left_gb < 0` 后停止后续 wave 计算。
- breakdown 聚合逻辑与原逻辑一致。

agg 测试：

- 调度 dry-run 生成的 `ScheduleStep` 与原模拟循环的 p/d 决策一致。
- replay 阶段只调用 `table.get()`，不直接调用 `_get_forward_info()`。
- 精确查表下 `ttft`、`tpot`、`token/s` 与原模拟循环一致。
- `DecodeFirstWithSlack` 决策不依赖 latency，因此可先 plan 再 replay。
- `output_length=1` 时不生成 decode key。
- decode key 复用 MTP acceptance rate 后的 latency。

CLI / 回归测试：

- `--jobs 1` 与默认外层搜索行为一致。
- `--jobs 4` 仍可用于 outer runner 并行，不改变 latency table 顺序早停行为。
- 非 chunked 场景现有 UT 结果不变。
- prefix cache + chunked prefill + 精确查表结果不变。
- PD ratio 优化中 P 阶段可使用查表结果，D 阶段不受影响。

# 5. 缺点和风险（可选）

风险：

- table key 覆盖不完整会导致错误复用 shape。
- agg 模式拆分 plan / replay 后，代码结构比当前单循环更复杂。
- 如果未来 Scheduler 决策依赖历史 latency 或 deadline，dry-run 计划生成需要调整。
- 为保证负 memory 早停，latency table 不做内部并行预取；128k / 1k 这类唯一 shape 很多且都不 OOM 的场景仍需顺序完成所有唯一 shape。

应对措施：

- 默认精确查表，不改变现有结果。
- key 中显式包含所有动态 shape 字段，并为 VL 场景补充 image 字段。
- Scheduler 接口文档中说明：若策略依赖 latency，则需要使用 lazy table replay 或在 plan 阶段提供估算 latency。
- 为 agg / disagg 分别增加 fake runner 单测，锁定调用次数和结果一致性。
- 依赖 optimizer 实例级 record cache 降低跨 batch size 搜索的重复建模成本。

# 6. 现有能力（可选）

现有能力：

- `OptimizerData.get_prefill_chunk_plan()` 已能生成 chunk plan。
- `AggThroughputOptimizer._get_or_compute_latency()` 已具备局部 cache。
- `ModelRunner.run_inference()` 已能基于 `RequestInfo` 计算单个 forward shape。
- `cli/inference/throughput_optimizer.py` 已提供 `--jobs` 参数。
- `serving_cast/parallel_runner.py` 已使用 `ProcessPoolExecutor` 并行运行搜索任务。

本 RFC 的差异：

- 将局部 cache 抽象为公共 Forward Latency Table。
- disagg prefill 也复用查表能力。
- agg chunked prefill 支持先 plan、再 prefetch、后 replay。
- 同一个 optimizer 实例内跨 batch size 搜索复用精确查表 record。
- negative-memory 早停默认开启且不可关闭。

# 7. 未解决问题（可选）

- 是否需要将精确查表 record 复用到比单个 optimizer 实例更长的生命周期。目前建议先限定在单个 optimizer 实例内。

---

附录

## 参考资料链接

- `msmodeling/docs/RFC/rfc_chunked_prefill_support_zh.md`
- `msmodeling/serving_cast/service/utils.py`
- `msmodeling/serving_cast/service/base_throughput_optimizer.py`
- `msmodeling/serving_cast/service/agg_throughput_optimizer.py`
- `msmodeling/serving_cast/service/disagg_throughput_optimizer.py`
- `msmodeling/cli/inference/throughput_optimizer.py`

## 术语表

| 术语 | 说明 |
| --- | --- |
| Chunked Prefill | 将一次长 prompt prefill 拆成多个 token chunk 分多轮完成 |
| ForwardShapeKey | 一次 forward latency 建模的唯一 shape 标识 |
| ForwardLatencyTable | 保存 forward shape 到 latency / memory / breakdown 的查表结构 |
| 精确查表 | 对所有唯一 shape 调用 `ModelRunner`，结果与逐步调用一致 |
| TTFT | Time To First Token，首 token 延迟 |
| TPOT | Time Per Output Token，平均每输出 token 延迟 |
| `max_batched_tokens` | 单个 prefill / mixed step 的总 token budget，同时作为单请求 prefill chunk 的最大 token 数 |
