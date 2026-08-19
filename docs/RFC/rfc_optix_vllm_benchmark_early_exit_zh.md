# 1. Overview（概述）

Status（状态）：Approved
Author(s)（作者）：huqixing
Created（创建日期）：2026-08-08
Updated（更新日期）：2026-08-08
Related Issue/PR (相关 Issue/PR): https://gitcode.com/Ascend/msmodeling/pull/542

---

## 1.1 Summary（简介）

本提案为 Optix vLLM benchmark 引入自适应早停能力。系统持续采集 vLLM
`/metrics`，先判断负载是否进入可评估阶段，再用当前候选的 metrics 窗口与历史最优
完整 Case 保存的 metrics 窗口进行同口径比较。连续多个窗口明显变差时，根据配置仅
报告早停结论，或终止当前 benchmark。

方案遵循以下原则：

- **metrics 对 metrics**：reference 和 observed 都来自 vLLM `/metrics` 的窗口统计，
  不使用完整 benchmark 汇总值直接对比运行中窗口。
- **完整结果保护**：基线和 refinement 完整评测不执行早停，只采集 metrics，确保
  reference 和最终候选具有完整 benchmark 结果。
- **负载就绪优先、时间兜底**：warmup 可在负载达到目标后提前结束，最迟 90 秒结束；
  不等待性能指标“稳定”。
- **连续窗口判定**：单个坏窗口不触发早停，降低瞬时抖动导致误杀的概率。
- **可观测、可回放**：主结果表记录早停摘要，明细表保留每次 metrics 采样，支持离线
  绘图和复核。
- **故障降级**：metrics 暂时不可用时跳过本次判断，不主动终止 benchmark。

## 1.2 Motivation（动机）

Optix 通过多轮启动推理服务、运行 benchmark 和计算 fitness 搜索服务参数。对于
vLLM 长输入、长输出场景，单个 benchmark 可能持续数十分钟，而明显劣于历史最优
结果的候选仍需完整运行，导致寻优时间随粒子数和迭代数快速增长。

仅配置固定 warmup 时间存在两个问题：小模型可能早已满载，继续等待会损失早停收益；
大模型或低请求速率场景又可能尚未进入有效负载，过早比较会误判。同时，如果使用历史
完整 benchmark 的全程平均值对比当前 `/metrics` 短窗口，两者数据来源和统计周期不同，
也会引入系统性偏差。

因此需要一套能够自动判断负载阶段、统一 reference/observed 口径、对偶发波动具有
容错能力，并且可以通过完整采样轨迹复核的早停机制。

## 1.3 Goals（目标）

### 目标

- 减少明显较差候选消耗的 benchmark 时间。
- 让 warmup 无需客户手工估计模型启动后的固定等待时长。
- 避免 reference 与 observed 指标来源、统计窗口不同造成误判。
- 支持先以 `report` 模式验证策略，再切换为 `terminate` 获得实际时间收益。
- 为 warmup、正式窗口和早停决策提供可追踪的数据证据。
- 保持已有配置兼容，客户只需配置常用开关，高级参数可按需覆盖。

### 非目标

- 不替代 vLLM benchmark 对完整 Case 的最终性能统计。
- 不保证异常 benchmark 或服务故障可以作为正常的“低性能 Case”参与比较。
- 不在本提案中定义 benchmark 最低成功率的全局有效性门槛。
- 不支持 MindIE；当前实现依赖 vLLM `/metrics`。
- 不将性能稳定性作为 warmup 结束条件。
- 不在当前版本中解决多 DP replica 的分组负载判定。

# 2. Use Case Analysis（用例分析）

## 2.1 长时间寻优

用户以固定模型、数据集和硬件运行多轮 PSO。历史最优完整 Case 已产生有效 metrics
窗口，后续候选进入正式观察阶段后持续显著低于 reference。系统应尽早记录或执行
早停，而不是等待完整 benchmark 结束。

功能要求：

- 仅对普通 PSO evaluation 候选执行早停。
- baseline、refinement 和最优候选复测必须完整运行。
- 早停结果不得参与 best/reference 选择。
- 早停判断不能阻塞 Scheduler 的健康检查和进程回收。

## 2.2 策略灰度验证

用户首次接入早停时使用 `action = "report"`。候选仍完整运行，但结果表记录“若启用
终止，本应在何时早停”以及估算可节省的时间。用户可以结合 metrics 明细曲线确认是否
存在过早或过晚判断，再切换为 `terminate`。

可靠性要求：

- report 和 terminate 使用相同的 warmup、窗口及判断逻辑。
- report 模式必须保留完整 benchmark 结果。
- 预计节省时间必须基于真实完整 Case 时长计算。

## 2.3 Case 运行轨迹复核

当用户怀疑早停误杀、漏停或 warmup 过长时，可使用 `case_id` 关联主结果和 metrics
明细，观察 running、waiting、generate speed、TTFT、TPOT、reference 和 observed
随时间的变化。

可观测性要求：

- 每次成功或失败的 metrics 采样均应立即落盘。
- Case 异常退出时，已经采集的数据不能丢失。
- 主结果必须记录 warmup 结束时间和原因。
- calibration 与 evaluation 必须能够通过 phase/pass 区分。

## 2.4 短 Case 与 metrics 故障

- Case 在 warmup 结束前完成：记录 `case_completed_before_warmup`，不执行早停。
- `/metrics` 暂时无法访问：记录失败采样并跳过本次判断，benchmark 继续运行。
- 当前没有可用 reference：继续采集当前 Case 的窗口，为后续完整 Case 生成 reference，
  但不执行早停。

# 3. Design（方案设计）

## 3.1 Overall Design（总体方案）

### 3.1.1 总体架构

```text
vLLM /metrics
      |
      v
VllmMetricsClient              解析 counter / histogram / gauge
      |
      v
EarlyExitController
  |-- 5 秒采样
  |-- 自适应 warmup
  |-- 30 秒正式窗口
  |-- reference / observed 比较
  `-- report 或 terminate
      |
      +--------------------+
      |                    |
      v                    v
data_storage_*.csv   metrics_samples_*.csv（仅 report）
Case 摘要和结果       调试用的每次采样时间序列
```

`Scheduler` 负责在单个 Case 启动时创建或更新控制器、传入运行参数、按调度循环触发
采样，并将早停异常转换为正常的候选结束流程。`DataStorage` 负责为每个 Case 分配稳定
的 `case_id`，并分别保存主结果和采样明细。

### 3.1.2 生效范围与阶段

早停仅在以下条件同时满足时执行：

1. `[benchmark_early_exit].enabled = true`；
2. 当前 engine 为 `vllm`；
3. 当前处于可执行早停的 evaluation pass；
4. 已存在可用的历史完整 Case reference。

不同评测阶段的行为如下：

| 阶段 | metrics 采集 | warmup | 早停决策 | 说明 |
|---|---|---|---|---|
| baseline | 是 | 是 | 否 | 必须完整运行，用于建立初始结果与窗口 reference |
| PSO evaluation | 是 | 是 | 是 | 主要早停对象 |
| request-rate calibration | 是 | 否 | 否 | 只观察，不混入正式评估 |
| calibration 后 evaluation | 是 | 重新开始 | 是 | 窗口状态从 evaluation pass 重新计时 |
| refinement / 最优候选复测 | 是 | 是 | 否 | 必须获得完整 benchmark 指标 |

被保护的完整评测仍执行 metrics 采样，以便从完整 Case 中选取代表窗口，但
`Scheduler.disable_early_exit()` 禁止其产生终止决策。

### 3.1.3 metrics 数据模型

采集端兼容 vLLM 指标的多种命名别名，并将同名、不同 label 的样本求和。核心指标
分为三类：

| 类型 | 指标 | 窗口计算方式 |
|---|---|---|
| Counter | 输出 token 总数 | `(current - previous) / elapsed` 得到 generate speed |
| Counter | 完成、失败请求总数 | 用窗口增量计算 success rate |
| Histogram | TTFT sum / count | `sum_delta / count_delta` |
| Histogram | TPOT sum / count | `sum_delta / count_delta` |
| Gauge | running、waiting 请求数 | 使用当前快照值描述负载 |

Counter 发生回退时，视为服务重启或指标重置，该窗口对应增量返回无效，不参与正式
判断。Histogram 在窗口内没有新增样本时，对应 TTFT 或 TPOT 为空；其他维度仍可按
规则使用。

### 3.1.4 自适应 warmup

warmup 内部策略如下，客户无需为不同模型估计固定时间：

| 策略 | 固定值 |
|---|---:|
| metrics 采样间隔 | 5 秒 |
| 最短 warmup | 15 秒 |
| 最长 warmup | 90 秒 |
| 负载阈值比例 | 80% |
| load-ready 连续样本 | 2 个 |
| 最少有效 warmup 样本 | 3 个 |

单 DP 场景下：

```text
effective_target = min(MAX_NUM_SEQS, CONCURRENCY)
load_threshold = ceil(effective_target * 0.8)
load_ready = running_requests >= load_threshold
```

`waiting_requests` 仅用于观测，不参与 `load_ready` 判定。该选择避免“刚出现排队就结束
warmup”，而此时 running 负载可能尚未达到有效目标。

warmup 有三种结束结果：

1. 经过至少 15 秒、取得至少 3 个有效负载样本，并连续 2 个样本满足
   `load_ready`：结束原因为 `load_ready`。
2. 未满足负载条件但经过 90 秒：强制结束，原因为 `max_warmup_timeout`。
3. benchmark 在上述条件前完成：原因为 `case_completed_before_warmup`，不进入正式
   早停判断。

性能波动是否稳定不属于 warmup 必要条件。warmup 结束快照作为第一个正式窗口的起点，
防止 warmup 数据混入 observed 窗口。

### 3.1.5 正式观察窗口

warmup 结束后，以固定窗口生成 observed：

```text
window_seconds = 30
min_output_tokens = 128
min_completed_requests = 1
```

窗口只有满足最少输出 token，并在 completed counter 可用时满足最少完成请求数，才是
有效窗口。无效窗口会被记录，但既不增加也不清零连续坏窗口计数。

每个有效窗口产生：

- `evaluation_generate_speed`；
- `evaluation_time_to_first_token`；
- `evaluation_time_per_output_token`；
- `evaluation_success_rate`。

### 3.1.6 reference 生成与选择

完整 Case 结束后，从其所有 generate speed 大于 0 的有效正式窗口中，按 generate
speed 升序排列并选择较低中位数：

```text
representative_index = (window_count - 1) // 2
```

使用较低中位数可以降低单个高峰窗口对 reference 的抬升，同时避免采用最差窗口过度
放宽早停标准。代表窗口的 generate speed、TTFT、TPOT、success rate 和有效窗口数量
写入主结果表。

后续 Case 启动时，从满足以下条件的历史结果中选择 fitness 最小的完整 Case：

- benchmark 参数口径匹配，包括 `num_prompts`；
- fitness、TTFT、TPOT 和 generate speed 有效；
- 未被实际早停；
- `usable_as_best = true`；
- 已保存有效的 `metrics_window_generate_speed`。

该 Case 的代表 metrics 窗口构成 reference。这样 observed 与 reference 均是 vLLM
metrics 窗口，不与完整 benchmark 的全程平均值混用。

### 3.1.7 坏窗口与早停判定

每个有效 observed 窗口依次执行两类判断：

1. **综合成本判断**：

   ```text
   observed_score >= reference_score * relative_score_threshold
   ```

2. **生成速度判断**：

   ```text
   observed_generate_speed
       <= reference_generate_speed * relative_generate_speed_threshold
   ```

score 复用 Optix fitness cost 的维度和权重；在运行中 metrics 缺少某个延迟维度时，
仅使用当前可获得的维度。score 数值越大表示成本越差。

默认参数为：

```text
relative_score_threshold = 3.0
relative_generate_speed_threshold = 0.5
consecutive_bad_windows = 3
```

任意一个条件满足即将当前窗口标记为坏窗口。有效好窗口会将连续坏窗口计数清零；连续
3 个坏窗口后形成早停决定。在最长 warmup 90 秒结束的典型场景下，最早约在 120、
150、180 秒形成三个正式窗口并作出决定。

### 3.1.8 action 语义

| action | 行为 | 适用阶段 |
|---|---|---|
| `report` | 设置 `would_early_exit=true`，完整运行 benchmark；结束后计算预计节省时间 | 策略验证、阈值调试 |
| `terminate` | 设置 `early_exit=true`，停止当前 benchmark 和服务 | 正式节省寻优时间 |

无论哪种 action，满足条件的 Case 都会设置 `usable_as_best=false`，不会成为后续最优
候选或 reference。`report` 模式保留完整 benchmark 结果，同时补充：

```text
estimated_time_saved_seconds = case_duration - decision_elapsed
estimated_time_saved_ratio = estimated_time_saved_seconds / case_duration
```

### 3.1.9 数据持久化

`data_storage_<run_time>.csv` 在原有完整 benchmark 指标之外增加以下类别：

- 关联：`case_id`；
- 决策：`would_early_exit`、`early_exit`、`early_exit_reason`、
  `early_exit_decision_elapsed_seconds`；
- 预计收益：`estimated_time_saved_seconds`、`estimated_time_saved_ratio`；
- 对比数据：reference / observed generate speed、score、SLO violations；
- reference 窗口：metrics window generate speed、TTFT、TPOT、success rate、样本数；
- warmup：结束时间、原因、样本数、有效目标、负载阈值、running、waiting、load ratio、
  是否强制结束。

新增字段不会替代原有 benchmark 汇总指标。实际早停结果的 `result_source` 为
`early_exit_metrics`，且不可作为最优结果。

仅当 `action = "report"` 时生成 `metrics_samples_<run_time>.csv`，并在每次采样后立即
追加一行。即使 Case 异常退出，已写入的轨迹仍然保留。`terminate` 模式仍在内存中采集
metrics 用于 warmup、早停判断和 reference，但不生成该调试轨迹表。主要字段包括：

- `case_id`、优化阶段、迭代号、benchmark phase 和 pass；
- 采样时间、Case 内经过时间和采样序号；
- counter 总值、窗口增量、running 和 waiting；
- 5 秒派生指标以及正式 30 秒窗口指标；
- warmup 状态、load-ready 计数、结束事件和原因；
- reference、observed、坏窗口计数和最终决定；
- scrape 是否成功及错误类型。

## 3.2 Technology Selection（技术选型）

### 固定 warmup 时间

固定 90 秒配置简单，但小模型可能早已满载，浪费早停机会；不同模型又需要客户反复
调参。本方案保留 90 秒作为上限，以负载就绪作为提前结束条件。

### 等待性能稳定后结束 warmup

吞吐稳定判定容易受采样窗口、突发流量和长尾请求影响，也可能显著延长 warmup，削弱
早停收益。因此性能稳定性只用于离线曲线分析，不参与在线控制。

### waiting 大于 0 即认为负载就绪

waiting 可能在 running 负载很低时出现，无法证明服务已达到有效并发。本方案只使用
`running >= load_threshold`，waiting 保留为观测指标。

### 完整 benchmark 汇总值对比运行中 metrics

两者统计周期和数据来源不同，会引入系统性偏差。本方案在完整 Case 结束后额外保存其
代表 metrics 窗口，统一使用 metrics 对 metrics。

### 单窗口触发

单窗口响应快，但容易受编译、调度和网络抖动影响。本方案默认要求连续三个坏窗口，并
允许客户在充分验证后覆盖该值。

## 3.3 Security, Privacy, and DFX Design（安全、隐私与 DFX 设计）

### metrics 不可用

- URL 仅接受带有效 host 的 HTTP/HTTPS 地址。
- 网络或读取失败转换为 `MetricsUnavailableError`。
- 第一次失败记录 warning，连续失败降为 debug，避免刷屏。
- 失败采样写入明细表；当前 benchmark 不因 metrics 故障终止。
- 恢复成功后自动继续 warmup 或正式窗口采集。

### 兼容性

- 功能默认关闭，不影响已有配置。
- 仅新增 CSV 列和独立 metrics 明细文件，不删除既有结果字段。
- 旧版 TOML 中的高级早停参数仍可加载并覆盖默认值。
- vLLM 指标名称通过别名集合兼容不同版本；无法识别的指标维度按缺失处理。

### 安全与隐私

- 仅对用户配置的 metrics URL 发起 GET 请求，不发送模型输入或输出文本。
- CSV 只记录性能计数、状态和参数，不记录 prompt 内容。
- 落盘值使用既有 CSV 清洗逻辑，降低公式注入风险。

### 可维护性与可测试性

- metrics 获取与解析集中在 `VllmMetricsClient`。
- warmup、窗口和决策状态集中在 `EarlyExitController`。
- Scheduler 只负责生命周期、阶段保护和结果合并。
- 固定 warmup 策略由命名常量定义，避免 magic number 分散在流程中。
- metrics client、状态机、Scheduler 和 DataStorage 均可通过 mock 独立测试。

## 3.4 Programming and Integration Design（编程与集成设计）

### 3.4.1 Basic Programming Model Design（基本编程模型）

本能力运行在 Optix Python 进程中，不要求修改 vLLM。集成依赖如下：

- vLLM 服务必须提供 Prometheus 文本格式的 `/metrics`。
- Optix Scheduler 必须能够取得当前 `MAX_NUM_SEQS` 和 benchmark concurrency。
- benchmark 生命周期由现有 Simulator/Benchmark 插件管理。
- DataStorage 目录必须可写，以保存主结果和采样明细。

当前负载公式按单 DP 场景设计。多 DP 环境需要按 replica 对 gauges 和并发目标进行
分组，在完成该设计前不应直接套用聚合值。

### 3.4.2 API Definition and Design（接口定义）

#### `BenchmarkEarlyExitConfig`

客户常用配置只需要以下三项：

```toml
[benchmark_early_exit]
enabled = true
action = "terminate"
metrics_url = "http://127.0.0.1:8000/metrics"
```

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `enabled` | bool | `false` | 是否启用 vLLM benchmark 早停 |
| `action` | enum | `terminate` | `report` 或 `terminate`；`report` 用于调试并生成 metrics 采样轨迹 |
| `metrics_url` | string | `http://127.0.0.1:8000/metrics` | vLLM metrics 地址 |

为兼容旧配置和高级调试，下列字段仍可显式覆盖；省略时使用代码默认值：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `window_seconds` | 30 | 正式观察窗口长度 |
| `min_output_tokens` | 128 | 有效窗口最少输出 token |
| `min_completed_requests` | 1 | 有效窗口最少完成请求 |
| `relative_generate_speed_threshold` | 0.5 | observed/reference 速度下限 |
| `relative_score_threshold` | 3.0 | observed/reference 成本上限 |
| `consecutive_bad_windows` | 3 | 连续坏窗口门槛 |
| `timeout_seconds` | 1.0 | 单次 `/metrics` 请求超时 |

TOML 显式值优先于代码默认值。`enabled` 默认关闭；启用后 `action` 默认 `terminate`，
使早停直接生效。需要验证阈值和观察 metrics 曲线时，显式改为 `report`。

#### `VllmMetricsClient`

输入为 metrics URL 或测试注入的 fetch callable，输出 `VllmMetricsSnapshot`。网络故障
统一转换为 `MetricsUnavailableError`，由 Scheduler 进行降级处理。

#### `EarlyExitController`

输入为配置、metrics client、reference、fitness evaluator 及当前运行参数。公开操作为：

- `reset()`：在新的 benchmark pass 开始时清空窗口状态；
- `observe(phase)`：采集但不做决定；
- `check(phase)`：采集并在满足条件时返回 `EarlyExitDecision`；
- `representative_window()`：返回完整 Case 的代表窗口和有效窗口数。

### 3.4.3 Usage Instructions（使用说明）

推荐分两步启用：

1. 使用 `action = "report"` 完整跑一轮，通过主结果中的预计节省时间和 metrics 曲线
   验证阈值。
2. 确认无误后改为 `action = "terminate"`，在后续寻优中获得实际时间收益。

运行约束：

- 修改 benchmark 数据规模后，应重新建立相同口径的 baseline/reference。
- 服务故障、请求大量失败与“正常但性能差”是不同状态，不应仅依赖早停处理。
- request rate 应符合目标业务负载；持续远高于服务能力会造成长期排队并显著放大 TTFT。

# 4. Test Design（测试设计）

## 4.1 单元与回归测试

| 类别 | 验证内容 |
|---|---|
| metrics 解析 | counter、histogram、gauge、label 聚合、别名和 counter 回退 |
| 采样节流 | 5 秒内不重复抓取，窗口到期后才生成正式指标 |
| warmup | load-ready 连续样本、15 秒下限、90 秒兜底、短 Case 结束 |
| 窗口有效性 | 最少 token、最少完成请求、缺失 TTFT/TPOT |
| reference | 完整 Case 代表窗口、历史最优筛选、早停结果排除 |
| 决策 | score、generate speed、连续坏窗口、好窗口清零 |
| action | report 不终止、terminate 触发终止、结果不可作为 best |
| 阶段 | calibration 只观察、evaluation 决策、完整评测保护 |
| 持久化 | 主表字段、仅 report 生成 metrics 轨迹、case_id 关联、失败采样保留 |
| 配置 | action 枚举、默认策略值、旧字段兼容 |

## 4.2 端到端验证

在真实 vLLM 环境至少验证：

1. baseline 完整运行并产生代表 metrics 窗口；
2. 后续相近候选不误触发早停；
3. 明显较差候选在连续三个坏窗口后产生 report；
4. report 的预计节省时间与 metrics 曲线人工复核一致；
5. 切换 terminate 后，差候选被实际停止且不参与最优结果；
6. `/metrics` 短暂失败时 benchmark 继续运行；
7. `case_id` 可以关联主结果与完整采样轨迹。

## 4.3 验收标准

- 默认关闭早停时，现有 Optix vLLM 寻优行为不变。
- 客户仅配置 enabled、action、metrics URL 即可启用功能。
- baseline 和 refinement 不被早停，且能保存代表 metrics 窗口。
- warmup 在负载连续就绪后结束，或最迟 90 秒结束；waiting 不参与结束判断。
- reference 与 observed 均来自同口径 `/metrics` 正式窗口。
- 连续三个坏窗口才形成决定；report 与 terminate 语义符合本提案。
- 早停 Case 不参与 best/reference 选择。
- 主结果表和 metrics 明细表可通过 `case_id` 完整关联。
- metrics 不可用不会主动终止 benchmark。
- 配置、早停控制器、Scheduler 和 DataStorage 回归测试全部通过。

# 5. Drawbacks and Risks（缺点与风险）

| 风险或限制 | 影响 | 应对 |
|---|---|---|
| reference 本身不稳定 | 可能误杀或漏停 | 使用完整 Case、较低中位数窗口和 report 灰度验证 |
| benchmark 成功率过低但仍产出结果 | 可能污染完整结果 | 由独立的 benchmark 有效性校验处理，不将服务故障等同于低性能 |
| 高 request rate 导致持续排队 | TTFT 和窗口指标显著恶化 | 使用与目标场景一致的到达率，并通过轨迹复核 |
| 多 DP 指标聚合 | effective target 与 gauge 可能不再同口径 | 当前限制单 DP；后续按 replica 分组 |
| vLLM 指标变更 | 部分维度缺失 | 维护别名集合，缺失维度降级而非终止 |
| report 与 terminate 行为差异 | report 估算收益不等同于真实回收耗时 | 正式启用前进行 terminate 端到端验证 |
| 较短 benchmark | 可用正式窗口不足 | 无 reference 时不早停，完整 Case 继续保留 benchmark 结果 |

本方案还会增加一个逐行追加的 metrics CSV。对于粒子数多、Case 时间长的任务，文件
行数会明显增长，但相较模型推理开销很小，并换取了异常退出后的可追踪性。

# 6. Existing Technology（现有技术）

本方案复用以下现有能力：

- vLLM Prometheus `/metrics` 提供累积 counter、histogram 和负载 gauge。
- Optix `PerformanceTuner.calculate_cost()` 提供与完整寻优一致的综合成本计算。
- Scheduler 已有 Simulator/Benchmark 生命周期、健康检查和阶段管理。
- DataStorage 已有主结果 CSV、历史结果筛选和 CSV 值清洗能力。

与只对 benchmark 最终 JSON 做判断的方案不同，本方案在 benchmark 运行中持续读取
服务端指标；与仅使用吞吐阈值的方案不同，本方案在指标可用时也可复用 TTFT、TPOT 和
成功率维度。

# 7. Unresolved Questions（未解决问题）

- 多 DP 场景应如何按 replica 关联 `MAX_NUM_SEQS`、concurrency 和 running gauges。
- benchmark 最低成功率应由全局健康检查、结果解析还是早停控制器负责。
- 不同模型和负载是否需要分层默认阈值，或继续使用统一默认值配合 report 灰度。
- metrics 明细的长期归档、压缩和自动图表生成是否应作为独立能力提供。
- report 估算节省时间是否需要加入真实进程回收和下一 Case 启动成本。

---

Appendix（附录）

## A. 术语

| 术语 | 含义 |
|---|---|
| reference | 历史最优完整 Case 保存的代表 metrics 窗口 |
| observed | 当前 Case 的正式 metrics 窗口 |
| warmup | benchmark 启动后、正式早停观察前的负载建立阶段 |
| 有效窗口 | 满足最少输出 token 和完成请求要求的正式窗口 |
| 坏窗口 | score 或 generate speed 相对 reference 达到早停条件的有效窗口 |
| would early exit | 条件已满足，无论 action 是否实际终止 |
| early exit | `action=terminate` 时实际终止了当前 Case |

## B. 代码位置

| 模块 | 职责 |
|---|---|
| `optix/optimizer/early_exit.py` | metrics 解析、warmup、窗口、reference 与决策 |
| `optix/optimizer/scheduler.py` | 生命周期、阶段保护、终止和结果合并 |
| `optix/optimizer/store.py` | 主结果、metrics 明细和 reference 选择 |
| `optix/config/config.py` | 配置模型与结果字段 |
| `tests/regression/optix/test_optimizer/test_early_exit.py` | 早停核心回归 |
| `tests/regression/optix/test_optimizer/test_schedule.py` | Scheduler 集成回归 |
| `tests/regression/optix/test_optimizer/test_store.py` | 数据持久化与 reference 回归 |

## C. 文档更新计划

| 文档 | 更新内容 |
|---|---|
| Optix 用户指南 | 增加基础配置、report/terminate 使用方式和结果字段说明 |
| Optix 设计文档 | 增加 Scheduler 与 EarlyExitController 的集成关系 |
| 故障处理指南 | 增加 metrics 不可用、benchmark 大量失败和服务异常的区分方法 |
