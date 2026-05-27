# RFC: msmodeling Chunked Prefill 建模支持方案

状态 (Status): Draft
作者 (Authors): @jiayanan, Codex
创建日期 (Created): 2026-05-18
更新日期 (Updated): 2026-05-22
相关 Issue/PR: 待补充（RFC 合入前需关联 Issue/PR）

---

## 1.1 简介

本 RFC 提议在 msmodeling 的服务化吞吐建模链路中补齐 Chunked Prefill 能力，使 `throughput_optimizer` 可以在长 prompt 或较小 prefill token budget 场景下，将一次 prefill 拆分为多个 chunk 进行建模。

该能力主要面向混部部署（代码中对应 `agg` / `aggregation`）和 PD 分离部署中的 prefill 阶段。首版目标是给出稳定、可解释的 TTFT、TPOT、吞吐趋势，而不是完全复刻线上推理框架的所有调度细节。

## 1.2 动机

当前 msmodeling 的混部模式以 `max_prefill_tokens` 作为单个 prefill wave 的 token budget，并要求：

```text
max_prefill_tokens >= effective_input_length
```

当长上下文场景中 `effective_input_length` 大于 `max_prefill_tokens` 时，`throughput_optimizer` 会直接报错退出。这与实际服务化系统存在差距：常见推理后端通常可以通过 Chunked Prefill 将长 prompt 拆成多个 prefill step，在有限 token budget 下完成请求。

本 RFC 同时将原 CLI 参数 `--max-prefill-tokens` 重命名为 `--max-batched-tokens`，使参数名同时覆盖 prefill-only step 和 prefill/decode 混部 step 的总 token budget。

## 1.3 目标

目标：

- 为 `cli.inference.throughput_optimizer` 增加 Chunked Prefill 建模能力。
- 支持 `max_batched_tokens < effective_input_length` 的长 prompt 场景，直接使用 `--max-batched-tokens` 作为单次 prefill chunk 的最大 token 数。
- 与 prefix cache 共存：先计算 `effective_input_length`，再对剩余 prefill token 做 chunk 切分。
- 在结果中输出 chunked prefill 的关键参数，便于分析 `max_batched_tokens` 对 TTFT、TPOT、吞吐和显存的影响。
- 保持兼容语义：当 `effective_input_length <= max_batched_tokens` 时，chunk 数为 1，建模结果与原完整 prefill 逻辑一致；当 `effective_input_length > max_batched_tokens` 时，自动切分 prefill chunk。

# 2. 用例分析

## 2.1 长上下文混部部署

用户希望评估如下场景：

```text
input_length = 32768
output_length = 1024
max_batched_tokens = 8192
deployment = aggregation
```

现有逻辑会因为 `max_batched_tokens < input_length` 退出。调整后，工具应自动按 `max_batched_tokens` 将 prompt 拆分为多个 prefill chunk，并继续输出 TTFT、TPOT、吞吐、显存余量等指标。

## 2.2 Prefix Cache 与 Chunked Prefill 组合

已有 prefix cache 逻辑会先计算：

```text
cached_prefix_tokens = floor(input_length * prefix_cache_hit_rate)
effective_input_length = input_length - cached_prefix_tokens
```

Chunked Prefill 应作用于 `effective_input_length`，而不是原始 `input_length`。例如：

```text
input_length = 32768
prefix_cache_hit_rate = 0.5
effective_input_length = 16384
max_batched_tokens = 8192
num_prefill_chunks = 2
```

decode 仍沿用原始 `input_length`，避免误把 prefix cache 命中解释为 decode 上下文变短。

## 2.3 PD 分离与 PD Ratio 优化

PD 分离中，Chunked Prefill 只影响 prefill 阶段：

- `disaggregation-prefill`：TTFT / P QPS 应纳入 chunked prefill。
- `disaggregation-decode`：TPOT / D QPS 不直接受 chunked prefill 影响。
- `enable-optimize-prefill-decode-ratio`：P 阶段 QPS 需要使用 chunked prefill 后的 prefill throughput，再与 D 阶段 QPS 做 ratio 计算。

## 2.4 DFX 要求

- 兼容性：当 `effective_input_length <= max_batched_tokens` 时，不影响已有完整 prefill 建模结果。
- 可维护性：chunk 切分和 latency cache key 独立封装，避免散落在优化器公式里。
- 可测试性：chunk plan、参数校验、agg（混部）/disagg（分离）路径、prefix cache 组合均可单测。
- 可靠性：`max_batched_tokens <= 0` 等场景应在入口明确报错。
- 性能：较小 `max_batched_tokens` 会增加 `ModelRunner` 调用次数，需通过缓存控制开销。

# 3.方案设计

## 3.1 总体方案

### 3.1.1 msmodeling 当前状态

当前 `throughput_optimizer` 混部模式（代码路径为 `agg` / `AggThroughputOptimizer`）的关键逻辑在：

```text
msmodeling/serving_cast/service/agg_throughput_optimizer.py
msmodeling/serving_cast/service/base_throughput_optimizer.py
msmodeling/serving_cast/service/utils.py
msmodeling/cli/inference/throughput_optimizer.py
```

现有语义：

- `OptimizerData.get_effective_input_length()` 已支持 prefix cache。
- 混部模式通过 `prefill_batch_size = max_batched_tokens // effective_input_length` 估算一个 prefill wave 可容纳的请求数。
- `_get_forward_info(is_decode=False)` 使用 `query_len = seq_len = effective_input_length` 跑一次完整 prefill。
- 当 `max_batched_tokens < effective_input_length` 且非 disagg / PD ratio 模式时，CLI 直接返回错误。

因此当前 `throughput_optimizer` 的服务化性能估算仍停留在完整 prefill 建模，尚未将 chunked prefill 暴露为可配置、可分析的建模能力。

### 3.1.2 推荐方案

将原 `--max-prefill-tokens` 重命名为 `--max-batched-tokens`，并扩展其语义：它既是单个 prefill / mixed step 的总 token budget，也是单请求 prefill chunk 的最大 token 数。首版不新增额外 chunk size 参数，也不增加独立 enable 开关。

```python
max_batched_tokens: int
```

核心流程：

```text
CLI / Web UI
  -> OptimizerData(max_batched_tokens, ...)
  -> OptimizerData.get_effective_input_length()
  -> OptimizerData.get_prefill_chunk_plan()
  -> AggThroughputOptimizer / DisaggThroughputOptimizer
  -> BaseThroughputOptimizer._get_forward_info(... query_len_override, seq_len_override ...)
  -> ModelRunner.run_inference(RequestInfo(...))
  -> 输出 TTFT / TPOT / throughput / chunk 参数
```

默认兼容策略：

- `effective_input_length <= max_batched_tokens`
  - chunk plan 只有 1 个 chunk，`query_len = seq_len = effective_input_length`
  - 结果与现有完整 prefill 逻辑一致
- `effective_input_length > max_batched_tokens`
  - 自动使用 `max_batched_tokens` 作为 chunk size 切分 prefill
  - 允许长 prompt 在较小 token budget 下继续建模
- 混部模式下，`max_batched_tokens` 表示 prefill chunk token 与 decode token 共享的混部 step 总 token budget

### 3.1.3 Chunk Plan

新增 `OptimizerData.get_prefill_chunk_plan()`：

```python
@dataclass(frozen=True)
class PrefillChunk:
    index: int
    query_len: int
    seq_len: int

def get_prefill_chunk_plan(self) -> list[PrefillChunk]:
    effective_input_length = self.get_effective_input_length(is_decode=False)
    chunk_size = self.max_batched_tokens

    chunks = []
    consumed = 0
    index = 0
    while consumed < effective_input_length:
        query_len = min(chunk_size, effective_input_length - consumed)
        seq_len = consumed + query_len
        chunks.append(PrefillChunk(index=index, query_len=query_len, seq_len=seq_len))
        consumed += query_len
        index += 1
    return chunks
```

说明：

- `query_len` 表示本次 prefill chunk 新计算的 token 数。
- `seq_len` 表示当前 chunk 完成后的有效上下文长度。
- 首版沿用当前 prefix cache 近似，即 chunk plan 基于 `effective_input_length` 生成，不额外把 cached prefix token 加回 `seq_len`。

### 3.1.4 混部模式 TTFT 语义

混部模式下需要明确以下调度语义：

- 未完成的 prefill chunk 不会向用户返回 token，也不应记录为请求的 prefill 完成。
- 只有最后一个 prefill chunk 完成，并产出用户可见的首 token 后，前端才认为该请求的 prefill 阶段结束。
- 已完成 prefill 的请求可以立即进入 decode，不需要等待同一轮或全部并发请求都完成 prefill。
- 因此 TTFT 应按请求或请求组的最后一个 prefill chunk 完成时间统计，而不是按所有请求 prefill 全部完成的全局 barrier 统计。

定义：

```text
E = effective_input_length
C = max_batched_tokens                  # 单请求 prefill chunk 的最大 token 数
K = ceil(E / C)                         # 单请求 chunk 数
B = max_batched_tokens                  # 混部 step 的总 token budget
S = floor(B * 1.15)                     # 混部 step 调度 slack 上限
N = concurrency
```

`max_batched_tokens` 在混部模式中同时约束 prefill token 与 decode token。为了避免 decode 优先调度后，少量 decode token 导致一个完整 prefill chunk 无法调度，或必须缩短 prefill chunk 才能调度，从而增加额外运行成本，首版提议混部 step 的调度判断允许 15% slack。对于任意混部 step，应满足：

```text
prefill_concurrency * chunk_query_len + decode_concurrency <= floor(max_batched_tokens * 1.15)
```

其中 decode 阶段每个活跃请求在一个 decode step 中消耗 1 个 token budget。也就是说，当已有 `D` 个请求处于 decode 阶段时，当前 chunk 可调度的 prefill 并发为：

```text
prefill_capacity(D, chunk_query_len) = max(0, floor((floor(max_batched_tokens * 1.15) - D) / chunk_query_len))
```

例如 `max_batched_tokens = 4000`、当前 chunk 的 `chunk_query_len = 4000`、`decode_concurrency = 512` 时：

```text
4000 + 512 = 4512 < 4000 * 1.15
```

因此同一个混部 step 中可以同时运行 1 个请求的完整 prefill chunk 和 512 个请求的 decode。若按严格 `max_batched_tokens` 判断，该 step 会因为 decode 已占用 512 个 token 而无法调度 prefill；15% slack 可以避免 prefill chunk 被迫等待或被缩短，缩短整体运行时间。

该 slack 只用于默认调度策略的混部 step 判断，不改变 chunk plan：单个 prefill chunk 的 `query_len` 仍不超过 `max_batched_tokens`。若 `prefill_concurrency * chunk_query_len + decode_concurrency > floor(max_batched_tokens * 1.15)`，默认调度策略仍应减少 prefill 并发或将该 step 视为 decode-only。

为避免将调度策略硬编码进吞吐模拟主循环，混部 step 的请求选择和 latency 合并方式应抽象为可插拔 `Scheduler`。轻量级时间模拟只维护请求状态、调用 latency model、更新时间线；具体每个 step 选择多少 prefill / decode 请求，由 Scheduler 决定。

调度策略接口：

```python
@dataclass(frozen=True)
class StepDecision:
    p_step: int
    d_step: int

@dataclass(frozen=True)
class SchedulerState:
    ready_decode: int
    pending_prefill: int
    chunk_query_len: int
    max_batched_tokens: int

class Scheduler(ABC):
    @abstractmethod
    def decide(self, state: SchedulerState) -> StepDecision:
        ...

    @abstractmethod
    def step_latency(self, prefill_latency: float, decode_latency: float) -> float:
        ...
```

首版默认策略为 `DecodeFirstWithSlack`：

```python
class DecodeFirstWithSlack(Scheduler):
    slack_ratio = 1.15

    def decide(self, state: SchedulerState) -> StepDecision:
        limit = floor(state.max_batched_tokens * self.slack_ratio)
        d_step = min(state.ready_decode, state.max_batched_tokens)
        p_step = max(
            0,
            min(
                state.pending_prefill,
                floor((limit - d_step) / state.chunk_query_len),
            ),
        )
        return StepDecision(p_step=p_step, d_step=d_step)

    def step_latency(self, prefill_latency: float, decode_latency: float) -> float:
        return max(prefill_latency, decode_latency)
```

后续可以在不修改模拟主循环的前提下增加 `PrefillFirst`、`Balanced`、EDF 或 token-proportional 等策略。首版不新增 `--scheduler` CLI 参数，默认使用 `DecodeFirstWithSlack`；是否暴露策略选择留作后续扩展。

首版不引入全局 prefill barrier。混部吞吐优化主路径采用轻量级时间模拟：从候选并发 `N` 个请求同时进入系统开始，在请求组粒度推进 prefill chunk 与 decode step。较早完成最后一个 prefill chunk 的请求可以在后续 step 直接参与 decode，后续请求仍继续推进 prefill chunk。具体模拟逻辑见 3.1.5。

当 `effective_input_length <= max_batched_tokens` 时仍走现有完整 prefill 公式，保证短 prompt 场景行为不变。

### 3.1.5 混部模式 TPOT 与吞吐估算

Chunked Prefill 下，TPOT 不能简单按“所有请求完成 prefill 后再统一 decode”计算。首版回退到轻量级时间模拟：不完全复刻线上 scheduler，只在请求组粒度模拟 prefill chunk 完成、首 token 产生、decode token 产生和请求完成时间。

定义：

```text
O = output_length                         # 用户请求的输出 token 总数
R = max(O - 1, 0)                         # 首 token 已由最后一个 prefill chunk 产生后的剩余 decode token 数
B = max_batched_tokens
K = len(chunk_plan)
N = concurrency
t = 0
scheduler = DecodeFirstWithSlack()
```

模拟状态：

```text
pending_prefill: 等待执行 prefill chunk 的请求组，初始为 N 个请求、chunk_index=0
ready_decode: 已产出首 token 且仍有剩余 decode token 的请求组
finished: 已完成全部 O 个输出 token 的请求
```

每个模拟 step 按如下策略推进：

```text
while finished < N:
  if pending_prefill is not empty:
    chunk = next pending prefill chunk
    state = SchedulerState(
      ready_decode=count(ready_decode),
      pending_prefill=count(pending_prefill with same chunk),
      chunk_query_len=chunk.query_len,
      max_batched_tokens=B,
    )
  else:
    state = SchedulerState(
      ready_decode=count(ready_decode),
      pending_prefill=0,
      chunk_query_len=B,
      max_batched_tokens=B,
    )

  decision = scheduler.decide(state)
  P_step = decision.p_step
  D_step = decision.d_step

  prefill_step_latency = latency(P_step, chunk.query_len, chunk.seq_len, is_decode=False) if P_step > 0 else 0
  decode_step_latency = latency(D_step, is_decode=True) if D_step > 0 else 0
  step_latency = scheduler.step_latency(prefill_step_latency, decode_step_latency)

  t += step_latency
  update selected prefill groups
  update selected decode groups
```

调度说明：

- `D_step` 表示本 step 参与 decode 的请求数，每个 decode 请求消耗 1 个 token budget，由 Scheduler 返回。
- `P_step` 表示本 step 参与 prefill 的请求数，由 Scheduler 根据当前策略和 token budget 返回。
- 默认 `DecodeFirstWithSlack` 采用 decode 优先、prefill 使用 slack 后剩余 budget 的策略，用于保证已经产出首 token 的请求 TPOT 稳定；若 `P_step == 0` 且仍有 pending prefill，则本 step 为 decode-only。
- 若 `ready_decode` 为空，则 `D_step = 0`，由于每个 prefill chunk 的 `query_len <= max_batched_tokens`，至少可以推进 1 个 prefill chunk。
- `S = floor(B * 1.15)` 是默认 Scheduler 的内部调度上限，只作用于 prefill/decode 混部 step，用于避免 decode 优先调度导致 prefill chunk 无法调度或被缩短。
- `step_latency` 由 Scheduler 合并 prefill / decode latency。默认 `DecodeFirstWithSlack` 使用 `max(prefill_step_latency, decode_step_latency)`，表示 prefill chunk 与 decode step 在同一个混部 step 内并行推进；后续策略可使用 `prefill_step_latency + decode_step_latency` 或专门的 mixed forward shape。

状态更新时间：

- prefill 组完成非最后一个 chunk 后，进入下一 chunk 的 `pending_prefill`。
- prefill 组完成最后一个 chunk 后，记录该组所有请求的 `first_token_time = t`；若 `R > 0`，进入 `ready_decode`，否则记录 `finish_time = t` 并直接进入 `finished`。
- decode 组在每个 decode step 产出 1 个 token，记录 token 产生时间；当剩余 decode token 为 0 时，记录 `finish_time = t` 并进入 `finished`。
- 请求组可按 `(chunk_index, remaining_decode_tokens, first_token_time)` 聚合，只在 step 内按 `P_step` / `D_step` 切分，不需要逐 token 复刻线上 scheduler。

最终指标：

```text
ttft = average(first_token_time for request in requests)
tpot = 0 if R == 0 else average((finish_time - first_token_time) / R for request in requests)
makespan = max(finish_time for request in requests)
request_throughput = 1000 * N / makespan
output_throughput = 1000 * N * O / makespan
```

这里的 `ttft` 是候选并发窗口内所有请求的平均首 token 时间；`tpot` 是首 token 之后的平均 decode token 间隔；`output_throughput` 使用从 `t=0` 到最后一个请求完成的 `makespan` 计算，因此能体现“较早完成 prefill 的请求先 decode、后续请求继续 prefill”的重叠效果。

### 3.1.6 PD 分离模式

PD 分离的 prefill 阶段复用相同 chunk plan：

- `decode_flag = False` 时，使用 chunked prefill 计算 TTFT / P throughput。
- `decode_flag = True` 时，保持现有 decode 逻辑，不额外读取 chunk 参数。

PD Ratio 优化中：

```text
P QPS = p_concurrency / chunked_prefill_ttft * 1000
D QPS = d_concurrency / (tpot * max(output_length - 1, 1)) * 1000
pd_ratio = D_QPS / P_QPS
```

### 3.1.7 参数语义

首版不增加独立 enable 开关，也不增加 chunk size 搜索开关。`--max-batched-tokens` 同时表达混部 step 的总 token budget 和单请求 prefill chunk 的最大 token 数：

- `--max-batched-tokens N`：使用 `N` 作为单个 prefill / mixed step 的总 token budget。
- `N` 必须为正整数。
- 当 `effective_input_length <= N` 时，chunk 数为 1，prefill chunk plan 等价于非 chunked prefill。
- 当 `effective_input_length > N` 时，自动按 `N` 切分 prefill chunk，最后一个 chunk 可小于 `N`。
- 默认 Scheduler 为 `DecodeFirstWithSlack`，混部 step 调度判断允许 15% slack，即 prefill token 与 decode token 总和可放宽到 `floor(N * 1.15)`；该 slack 不改变 prefill chunk 的最大长度。
- 原 `--max-prefill-tokens` 改名为 `--max-batched-tokens`；首版 RFC 不引入额外 chunk size 参数。

### 3.1.8 输出结果

在结果中追加以下字段：

```text
prefill_num_chunks
max_batched_tokens
effective_input_length
```

## 3.2 技术选型（可选）

### 方案 A：固定 `max_batched_tokens`，复用现有 ModelRunner（推荐首版）

优点：

- 改动集中在 `OptimizerData`、`BaseThroughputOptimizer`、agg（混部）/disagg（分离）optimizer。
- 不需要改底层算子建模。
- 与 prefix cache 已有 effective length 语义兼容。
- 可快速解除 `max_batched_tokens < effective_input_length` 的限制。

缺点：

- 不能精确模拟线上 scheduler 的请求间 chunk 交错和 decode 抢占。
- 首版仅内置默认 `DecodeFirstWithSlack`，其他策略需要后续补充实现。
- 多 chunk 会增加 `ModelRunner` 调用次数，需要 cache。

## 3.3 安全隐私与DFX设计

兼容性：

- `effective_input_length <= max_batched_tokens` 时，保持完整 prefill 建模结果。
- `effective_input_length > max_batched_tokens` 时，不再返回错误，而是自动启用 chunked prefill。
- 已有 CLI 参数、prefix cache、MTP、PD ratio 的默认行为不变。

可靠性：

- `max_batched_tokens <= 0` 报错。
- `effective_input_length < 1` 继续沿用 prefix cache 的错误处理。

性能：

- latency cache key 需从 `batch_size` 扩展为：

```text
(is_decode, concurrency, query_len, seq_len)
```

可维护性：

- chunk plan 生成放在 `OptimizerData`，避免 agg（混部）/disagg（分离）重复实现。
- 调度策略通过 `Scheduler` 抽象承载，轻量级时间模拟只负责状态推进，避免把 decode-first、slack ratio 和 latency 合并方式硬编码在主循环中。
- `_get_forward_info()` 只负责把 override 后的 `query_len` / `seq_len` 转成 `RequestInfo`。
- agg（混部）/disagg（分离）只负责选择 Scheduler 并调用模拟过程。

## 3.4 编程与调用设计

### 3.4.1 编程模型基本设计

涉及模块：

| 模块 | 变更 |
| --- | --- |
| `cli/inference/throughput_optimizer.py` | 将 `--max-prefill-tokens` 改名为 `--max-batched-tokens`，并更新校验 |
| `serving_cast/service/utils.py` | `OptimizerData` 增加 chunked prefill 字段和 chunk plan helper |
| `serving_cast/service/scheduler.py` | 定义 `Scheduler`、`SchedulerState`、`StepDecision` 和默认 `DecodeFirstWithSlack` |
| `serving_cast/service/base_throughput_optimizer.py` | `_get_forward_info()` 支持 prefill query/seq override |
| `serving_cast/service/agg_throughput_optimizer.py` | 混部模式使用 chunk plan 计算 TTFT、TPOT 与吞吐 |
| `serving_cast/service/disagg_throughput_optimizer.py` | prefill 模式使用 chunk plan，decode 模式保持不变 |
| `web_ui/command_builder.py` | Web UI 支持生成 `--max-batched-tokens` |

开发约束：

- 使用现有 `RequestInfo` / `ModelRunner` 接口。
- 不直接修改 tensor_cast 底层 op 建模。

可验收设计：

- `effective_input_length <= max_batched_tokens` 时输出与现有基线一致。
- `effective_input_length > max_batched_tokens` 时可正常输出 chunked prefill 结果。
- `max_batched_tokens` 变小时，`prefill_num_chunks` 增大，TTFT 和吞吐趋势能够体现更多 prefill step 带来的影响。

### 3.4.2 接口定义与设计

#### 3.4.2.1 CLI 参数

接口描述：在 `cli.inference.throughput_optimizer` 中将原 `--max-prefill-tokens` 改名为 `--max-batched-tokens`，并用该参数自动决定 prefill chunk plan。

接口原型：

```bash
python -m cli.inference.throughput_optimizer MODEL_ID \
  --max-batched-tokens 8192
```

输入/输出参数：

| 参数名称 | 输入/输出 | 类型 | 描述 | 取值范围 |
| --- | --- | --- | --- | --- |
| `--max-batched-tokens` | 输入 | int | 单个 prefill / mixed step 的总 token budget；同时作为单请求 prefill chunk 的最大 token 数；混部调度判断允许 15% slack | `> 0` |
| `prefill_num_chunks` | 输出 | int | 单请求 prefill chunk 数 | `>= 1` |
| `max_batched_tokens` | 输出 | int | 实际使用的 token budget 和最大 prefill chunk token 数 | `> 0` |

异常处理：

- 指定 `--max-batched-tokens` 且其值小于等于 0：返回非零退出码。
- `effective_input_length > max_batched_tokens` 不再报错，而是自动启用 chunked prefill。

变更说明：

- `--max-batched-tokens` 是 `--max-prefill-tokens` 的重命名。
- 不新增额外 chunk size 参数，避免同一个 token budget 与 chunk size 出现双参数配置不一致。

调用参考代码：

```bash
python -m cli.inference.throughput_optimizer Qwen/Qwen3-32B \
  --device TEST_DEVICE \
  --num-devices 1 \
  --input-length 32768 \
  --output-length 1024 \
  --tpot-limits 50 \
  --max-batched-tokens 8192
```

#### 3.4.2.2 `OptimizerData`

接口描述：集中表达 chunked prefill 配置和派生值。

接口原型：

```python
@dataclass
class OptimizerData:
    max_batched_tokens: int

    def get_prefill_chunk_plan(self) -> list[PrefillChunk]: ...
```

输入/输出参数：

| 参数名称 | 输入/输出 | 类型 | 描述 | 取值范围 |
| --- | --- | --- | --- | --- |
| `max_batched_tokens` | 输入 | int | 用户指定 token budget；同时作为最大 prefill chunk size | `> 0` |
| `PrefillChunk.query_len` | 输出 | int | 本 chunk 新 prefill token 数 | `>= 1` |
| `PrefillChunk.seq_len` | 输出 | int | 本 chunk 完成后的上下文长度 | `>= query_len` |

### 3.4.3 使用说明

基础用法：

```bash
python -m cli.inference.throughput_optimizer Qwen/Qwen3-32B \
  --device TEST_DEVICE \
  --num-devices 1 \
  --input-length 32768 \
  --output-length 1024 \
  --tpot-limits 1000 \
  --max-batched-tokens 8192
```

指定更小 token budget：

```bash
python -m cli.inference.throughput_optimizer Qwen/Qwen3-32B \
  --device TEST_DEVICE \
  --num-devices 1 \
  --input-length 32768 \
  --output-length 1024 \
  --tpot-limits 1000 \
  --max-batched-tokens 4096
```

与 prefix cache 组合：

```bash
python -m cli.inference.throughput_optimizer Qwen/Qwen3-32B \
  --device TEST_DEVICE \
  --num-devices 1 \
  --input-length 32768 \
  --output-length 1024 \
  --tpot-limits 1000 \
  --prefix-cache-hit-rate 0.5 \
  --max-batched-tokens 8192
```

# 4.测试设计

单元测试：

- 参数校验
  - `max_batched_tokens <= 0` 非法。
  - `max_batched_tokens > 0` 合法。
- `OptimizerData.get_prefill_chunk_plan()`
  - `effective_input_length=4096, max_batched_tokens=8192` 生成 1 个 chunk，等价于完整 prefill。
  - `effective_input_length=32768, max_batched_tokens=8192` 生成 4 个 chunk。
  - `effective_input_length=10000, max_batched_tokens=4096` 生成 `[4096, 4096, 1808]`。
  - prefix cache 生效后基于 `effective_input_length` 生成 chunk plan。
- `BaseThroughputOptimizer._get_forward_info()`
  - prefill override 后 `RequestInfo.query_len` / `seq_len` 符合 chunk plan。
  - decode 路径不受 prefill chunk 参数影响。
- `DecodeFirstWithSlack`
  - `max_batched_tokens=4000, ready_decode=512, pending_prefill=1, chunk_query_len=4000` 时返回 `p_step=1, d_step=512`。
  - 超过 `floor(max_batched_tokens * 1.15)` 时减少 `p_step` 或返回 `p_step=0`。
  - `step_latency(prefill_latency, decode_latency)` 返回二者较大值。
- 轻量级时间模拟
  - 使用注入的 Scheduler 产生 `StepDecision`，主循环不直接硬编码 decode-first、slack ratio 或 latency 合并规则。
  - 可通过 fake Scheduler 验证状态推进、首 token 时间、finish 时间和 makespan 计算。

混部模式测试：

- `effective_input_length <= max_batched_tokens` 时，与非 chunked prefill 的 prefill latency 调用等价。
- `effective_input_length > max_batched_tokens` 时，自动使用 chunk plan 且可以正常输出结果。
- `K=1` 场景与非 chunked prefill 的 prefill latency 调用等价。
- `K>1` 场景会按 chunk 数多次调用 prefill latency。
- 未完成的 prefill chunk 不产生首 token；只有最后一个 prefill chunk 完成后才统计该请求的 TTFT。
- 已完成 prefill 的请求组可进入 decode，不需要等待全部请求完成 prefill。
- TPOT 通过轻量级时间模拟计算：较早完成 prefill 的请求组能在后续 prefill chunk 推进期间消耗 decode token。
- 混部 step 的 token budget 同时包含 prefill chunk token 和 decode token，调度判断允许 15% slack。例如 `max_batched_tokens=4000`、当前 `chunk_query_len=4000`、`decode_batch=512` 时，`4000 + 512 = 4512 < 4000 * 1.15`，可同时调度 1 个 prefill 请求和 512 个 decode 请求。
- 超过 15% slack 的混部 step 仍需减少 prefill 并发或退化为 decode-only。
- 验证替换 Scheduler 后不需要修改轻量级时间模拟主循环。
- 验证 chunked prefill 场景不使用“所有请求 prefill 完成后统一 decode”的全局 barrier 公式。
- 输出吞吐使用最后一个请求组完成 decode 的 `makespan` 计算。
- cache key 区分不同 `max_batched_tokens`、query_len、seq_len。

PD 分离测试：

- `--disagg --ttft-limits ... --max-batched-tokens N` 使用 chunk plan。
- `--disagg --tpot-limits ... --max-batched-tokens N` decode 结果与完整 prefill 场景一致。
- `--enable-optimize-prefill-decode-ratio` 中 P QPS 使用 chunked prefill TTFT，D QPS 不变。

CLI / Web UI 测试：

- CLI 能解析 `--max-batched-tokens`。
- 非法参数返回非零退出码并输出明确错误信息。
- Web UI command builder 能生成 `--max-batched-tokens`。
- 输出结果包含 `max_batched_tokens`、`prefill_num_chunks`。

回归测试：

- `effective_input_length <= max_batched_tokens` 的现有 UT 结果不变。
- prefix cache 现有测试继续通过。
- MTP decode 现有测试继续通过。

# 5.缺点和风险（可选）

- 首版公式是服务化近似，不完全模拟线上 scheduler 的请求间交错、decode 抢占和动态 chunk 调整。
- `max_batched_tokens` 变小会增加 `ModelRunner` 调用次数，影响优化器运行时间。
- 不同推理后端对 chunked prefill 的调度语义不完全一致；首版只做 msmodeling 内部建模，不承诺完全对应某个后端版本。
- 输出新增列可能影响依赖固定列集合的下游脚本，需要在 release note 中说明。
- prefix cache 与 chunked prefill 的组合仍是 token 粒度近似，不代表真实 block cache 行为。

应对措施：

- `effective_input_length <= max_batched_tokens` 时保持完整 prefill 建模，保护短 prompt 场景行为。
- 输出中显式记录 chunk 参数，避免用户误解结果来源。
- 后续可在 `serving_cast/service` 内引入更高精度的 mixed-step TPOT 模型。

# 6.现有能力（可选）

msmodeling 当前能力：

- `OptimizerData.get_effective_input_length()` 已支持 prefix cache。
- `AggThroughputOptimizer` 已用 `max_batched_tokens` 表达 prefill wave token budget。

本 RFC 的差异：

- 优先在 msmodeling 的 `RequestInfo` 粒度上完成 chunk prefill 建模。
- 不新增 chunk size 参数，直接使用 `--max-batched-tokens` 自动决定 chunk plan，方便与 prefix cache、并发参数一起分析。

---

附录

## 参考资料链接

- `msmodeling/serving_cast/service/agg_throughput_optimizer.py`
- `msmodeling/serving_cast/service/base_throughput_optimizer.py`
- `msmodeling/serving_cast/service/utils.py`
- `msmodeling/docs/RFC/rfc_prefix_cache_support_zh.md`
- `msmodeling/docs/RFC/rfc_q2_modeling_tecnical_plan.md`

## 术语表

| 术语 | 说明 |
| --- | --- |
| Prefill | 首 token 前，对 prompt token 执行上下文计算并写入 KV cache 的阶段 |
| Chunked Prefill | 将一次长 prompt prefill 拆成多个 token chunk 分多轮完成 |
| ISL | Input Sequence Length，输入长度 |
| OSL | Output Sequence Length，输出长度 |
| TTFT | Time To First Token，首 token 延迟 |
| TPOT | Time Per Output Token，平均每输出 token 延迟 |
| `max_batched_tokens` | 单个 prefill / mixed step 的总 token budget，混部时同时包含 prefill chunk token 和 decode token；混部调度判断允许 15% slack |
| `effective_input_length` | prefix cache 生效后仍需 prefill 的 token 数 |

## 文档更新计划

- 更新 `msmodeling/docs/en/throughput_optimizer.md`，补充 chunked prefill 参数说明。
- 如实现 Web UI 参数，更新相关 Web UI 使用文档。
- 在 release note 中说明默认兼容行为与新增输出列。
