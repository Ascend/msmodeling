# OptiX 服务参数微调能力详细设计

## 修订记录

| 日期 | 修订版本 | 修改描述 | 作者 | RFC文档 |
| -- | -- | -- | -- | -- |
| 2026-07-22 | 1.0 | 初稿完成 | 许锦涛 | 待确认 |

## 背景描述

OptiX 是 msmodeling 中面向推理服务参数自动寻优的工具，当前主流程由 baseline、PSO 搜索和候选 fine tune 三段组成。baseline 用于确认默认服务配置可运行并得到初始指标，PSO 用于在参数空间内搜索候选服务参数，fine tune 用于对候选结果做局部微调并选出最终推荐配置。

在 PD 混部和 PD 分离推理场景中，用户经常需要在已有服务或固定服务侧参数的前提下，只调整压测侧的 `CONCURRENCY`、`REQUESTRATE`、`NUM_PROMPTS` 等字段，观察 TTFT、TPOT、throughput 的变化。原有流程存在以下问题：

1. PSO 是必经阶段，无法直接从 baseline 进入微调，导致只想做局部验证时仍要承担完整 PSO 评测成本。
2. fine tune 每轮可能走完整 `scheduler.run()`，在服务侧参数不变时仍重复启动和停止服务，PD 分离场景耗时较高。
3. PD 分离需要先探测 QPS，再固定请求速率并按 TPOT 调整并发；原有 PD 混部微调逻辑同时调整请求速率和并发，不适配该策略。
4. benchmark 中 `REQUESTRATE=0` 表示不限速，不是普通数值 0；原有比例调整在从不限速降压时容易失效。
5. 外部服务已提前启动时，OptiX 缺少只运行 benchmark、不管理 simulator 生命周期的模式。

本特性的核心价值是将 OptiX 的微调能力从“PSO 后处理”扩展为可独立使用、可按场景选择算法的服务参数局部调优能力，降低 PD 分离调参成本，并提升微调过程的可控性和可解释性。

目标如下：

1. 支持 `skip_pso=true`，允许跳过 PSO，直接基于 baseline 进行微调。
2. 支持 `fine_tune_mode`，区分 `pd_mixed` 和 `pd_disaggregation` 两种微调策略。
3. 支持 PD 分离微调：先以 `REQUESTRATE=0` 探测 QPS，再固定请求速率，只按 TPOT 二分调整并发。
4. 支持 `scheduler.run_benchmark_only()`，在服务侧参数不变时只重启 benchmark，不重复重启 simulator 服务。
5. 支持 `manage_simulator_lifecycle=false`，允许对外部已启动服务执行 benchmark-only 微调。
6. 补齐配置校验、异常处理和回归测试，避免非法组合进入运行阶段。

## 方案设计

### 场景用例

| 场景 | 触发条件 | 输入 | 输出 |
| -- | -- | -- | -- |
| 常规 PSO 后微调 | `skip_pso=false` | baseline 配置、PSO 参数空间、benchmark 配置 | PSO top-k 候选及其微调结果，最终推荐参数 |
| baseline 直接微调 | `skip_pso=true` | baseline 配置、fine tune 配置 | baseline 及其微调结果，最终推荐参数 |
| PD 混部微调 | `fine_tune_mode="pd_mixed"` | TTFT/TPOT SLO、并发和请求速率字段 | 根据 TPOT 调整并发，根据 TTFT 调整请求速率 |
| PD 分离微调 | `fine_tune_mode="pd_disaggregation"` | TPOT SLO、throughput/QPS 指标、并发和请求速率字段 | 固定请求速率下的并发搜索结果 |
| 外部服务微调 | `manage_simulator_lifecycle=false` 且 `skip_pso=true` | 已启动服务、benchmark 参数 | 只重跑 benchmark 的微调结果 |

### 整体思路

本方案在保持 OptiX 既有 baseline、PSO、DataStorage 和插件体系不变的基础上，将候选微调逻辑抽象为 `_fine_tune_from_candidate()`，让候选来源可以是 PSO top-k，也可以是 baseline。`skip_pso` 只改变候选来源，不改变最终选择、结果保存和错误处理方式。

微调算法由 `FineTune.fine_tune_mode` 控制。`pd_mixed` 复用并增强原有策略，根据 TPOT 调整并发，根据 TTFT 调整请求速率，并补充 `REQUESTRATE=0` 不限速语义和最小步长退避。`pd_disaggregation` 新增专用策略，先通过 QPS 探测得到固定请求速率，再只调整并发，避免 PD 分离场景下请求速率和并发同时变化导致归因不清。

执行层新增 `Scheduler.run_benchmark_only()`，用于服务已启动且服务侧参数不变的微调轮次。该方法只停止和重启 benchmark，更新 benchmark 侧 `data_field` 和命令，然后复用 `monitoring_status()` 等待 benchmark 完成。`Scheduler.save_result(stop_service=False)` 支持保存结果时不停止服务，使一次候选的多轮微调可以共享同一服务生命周期。

配置层新增校验，确保 `skip_pso=false` 时 `n_particles` 和 `iters` 必须大于 0，`skip_pso=true` 时允许两者为 0；`manage_simulator_lifecycle=false` 只允许和 `skip_pso=true` 搭配，避免 PSO 搜索误以为可以修改外部服务配置。

### 系统架构

![image.png](https://raw.gitcode.com/user-images/assets/8428112/5d3f41ff-92b7-4732-91ca-189dc8529fec/image.png 'image.png')

上图展示微调能力涉及的主要模块。配置决定候选来源和微调模式，`PSOOptimizer` 负责流程编排，`FineTune` 负责参数生成，`Scheduler` 负责真实评测和结果落盘。

### 核心流程

![image.png](https://raw.gitcode.com/user-images/assets/8428112/f86334e3-9291-48e1-99d7-9d0ee4ae0c18/image.png 'image.png')

该流程说明 `skip_pso` 和 `fine_tune_mode` 是两个正交开关。前者决定候选来自 baseline 还是 PSO，后者决定候选进入哪种微调算法。

### 配置设计

| 配置项 | 类型 | 默认值 | 取值范围 | 说明 |
| -- | -- | -- | -- | -- |
| `skip_pso` | bool | `false` | `true`/`false` | 是否跳过 PSO，直接对 baseline 做微调。 |
| `fine_tune_mode` | str | `"pd_mixed"` | `"pd_mixed"`、`"pd_disaggregation"` | 微调模式。非法值在配置校验阶段报错。 |
| `n_particles` | int | `5` | `[0, 1000)` | PSO 粒子数。`skip_pso=false` 时必须大于 0。 |
| `iters` | int | `10` | `[0, 1000)` | PSO 迭代轮数。`skip_pso=false` 时必须大于 0。 |
| `max_fine_tune` | int | `30` | 运行时限制到 `MAX_ITER_NUM` 内 | 每个候选最多微调轮数。默认值从 10 调整为 30。 |
| `use_request_rate_calibration` | bool | `true` | `true`/`false` | PSO 阶段是否使用请求速率校准。`skip_pso=true` 的 PD 分离路径不依赖该开关。 |
| `manage_simulator_lifecycle` | bool | `true` | `true`/`false` | 是否由 OptiX 管理 simulator 服务生命周期。为 `false` 时要求 `skip_pso=true`。 |
| `data_storage.pso_top_k` | int | 既有默认值 | 非负整数 | PSO 后进入微调的候选数量。`skip_pso=true` 时不使用。 |

推荐的 PD 分离 baseline 直接微调配置如下：

```toml
skip_pso = true
n_particles = 0
iters = 0
fine_tune_mode = "pd_disaggregation"
tpot_penalty = 1
ttft_penalty = 0
use_request_rate_calibration = false
manage_simulator_lifecycle = true
max_fine_tune = 30
```

如果服务由外部提前启动，只希望 OptiX 运行 benchmark，可配置：

```toml
skip_pso = true
n_particles = 0
iters = 0
fine_tune_mode = "pd_disaggregation"
manage_simulator_lifecycle = false
```

### 算法设计

#### PD 混部微调

`pd_mixed` 模式基于上一轮 `PerformanceIndex` 生成下一轮参数：

1. 如果 `time_per_output_token > tpot_slo`，按超限比例降低并发。
2. 如果 `time_per_output_token < tpot_slo * (1 - slo_coefficient)`，按余量比例提高并发。
3. 如果 `time_to_first_token > ttft_slo`，按超限比例降低请求速率。
4. 如果 `time_to_first_token < ttft_slo * (1 - slo_coefficient)`，按 `step_size` 提高请求速率。
5. 连续两次调整方向相反时，优先使用历史值计算中间点，减少震荡。
6. 当并发发生变化时，请求速率重置为初始请求速率，避免同一轮同时改变两个压力维度。

`REQUESTRATE=0` 被定义为不限速。如果请求速率当前为 0 且需要继续增压，则认为已经到达语义最大值，不再调整；如果需要降压，则从有限上界开始回退到正请求速率。轻微超 SLO 时，即使比例变化小于普通阈值，也会启用最小步长退避，避免超线后提前停止。

#### PD 分离微调

`pd_disaggregation` 模式将请求速率和并发解耦：

1. 使用候选参数生成 probe 参数，并将 `REQUESTRATE` 设置为 0，表示不限速压测。
2. 运行一次 probe，读取 `PerformanceIndex.throughput` 作为 QPS。该字段是唯一 QPS 来源，不使用 `generate_speed` 替代。
3. 将 QPS 向下保留 1 位小数，作为后续固定请求速率：

   ```text
   fixed_request_rate = floor(throughput * 10) / 10
   ```

4. 后续每轮固定 `REQUESTRATE=fixed_request_rate`，只调整 `CONCURRENCY`。
5. 如果 `TPOT > tpot_slo`，将当前并发作为上界；没有下界时下一轮取当前并发的一半，否则取上下界中点。
6. 如果 `TPOT <= tpot_slo`，将当前并发作为下界；没有上界时下一轮取当前并发的 2 倍，否则取上下界中点。
7. 参数到达边界、变化不足、重复评测、缺少指标或达到 `max_fine_tune` 时停止。

示例：

```text
baseline: CONCURRENCY=16, REQUESTRATE=8
probe:    CONCURRENCY=16, REQUESTRATE=0 -> throughput=12.39
fixed:    REQUESTRATE=floor(12.39 * 10) / 10 = 12.3
round 1:  TPOT 未超时 -> CONCURRENCY=32, REQUESTRATE=12.3
round 2:  TPOT 超时   -> CONCURRENCY=(16 + 32) / 2 = 24, REQUESTRATE=12.3
```

### 执行与生命周期设计

`Scheduler.run_benchmark_only()` 负责 benchmark-only 评测，流程如下：

1. 停止上一轮 benchmark。
2. 将 `simulate_run_info` 映射到 benchmark 的 `data_field`。
3. 调用 benchmark 的 `update_command()` 和 `prepare()`。
4. 启动 benchmark。
5. 调用 `monitoring_status(monitor_service=...)` 等待 benchmark 结束。
6. 读取 `benchmark.get_performance_index()` 并写入 `last_outcome`。

`monitor_service=true` 时，运行期同时检查 simulator 和 benchmark；`monitor_service=false` 时，只检查 benchmark，适用于外部服务模式。`save_result(stop_service=false)` 用于微调中间轮次，保存结果但不停止服务；候选微调结束后 `_stop_after_fine_tune()` 统一清理资源。外部服务模式下只停止 benchmark，不停止外部 simulator。

### 最终选择逻辑

常规 `pd_mixed` 模式沿用 `best_params()` 的既有规则，根据 penalty 和 SLO 组合选择满足约束且吞吐更高的结果。`pd_disaggregation` 模式新增专用选择逻辑：

1. 优先选择 `TPOT <= tpot_slo` 的结果。
2. 在满足 TPOT 的结果中，优先按 `throughput` 选择最大值；如果 `throughput` 缺失，则回退使用 `generate_speed`。
3. 如果所有结果均超过 TPOT SLO，则选择 TPOT 相对超限比例最小的结果。

### 异常与边界处理

| 异常或边界 | 处理方式 |
| -- | -- |
| `fine_tune_mode` 非法 | `Settings.validate_pso_settings()` 抛出配置错误。 |
| `skip_pso=false` 且 `n_particles<=0` 或 `iters<=0` | 配置校验失败。 |
| `manage_simulator_lifecycle=false` 且 `skip_pso=false` | 配置校验失败。 |
| PD 分离缺少 `REQUESTRATE` | probe 阶段抛出 `ValueError`，该候选微调停止。 |
| PD 分离缺少正 `throughput` | 固定请求速率初始化失败，该候选微调停止。 |
| PD 分离缺少 `CONCURRENCY` 或 TPOT | 并发二分阶段失败，该候选微调停止。 |
| benchmark-only 运行失败 | `last_outcome` 标记失败，fitness 记为 `inf`，结果仍落盘。 |
| baseline 失败 | 抛出 `BaselineRunError`，按生命周期模式清理资源。 |
| PSO 首轮 shape broadcast 错误 | 记录 warning 并重试一次，其他 `ValueError` 继续抛出。 |

### 影响范围

| 模块或文件 | 影响说明 |
| -- | -- |
| `optix/config/config.py` | 新增 `skip_pso`、`fine_tune_mode`、`manage_simulator_lifecycle`，调整 `n_particles`、`iters` 校验，`max_fine_tune` 默认值改为 30。 |
| `optix/config.toml` | 补充 PD 分离微调示例配置，并演示 `NUM_PROMPTS` 随 `CONCURRENCY` 派生。 |
| `optix/optimizer/experience_fine_tunning.py` | 新增 PD 分离算法、QPS 处理、`REQUESTRATE=0` 语义、最小步长退避。 |
| `optix/optimizer/optimizer.py` | 新增 `skip_pso` 编排、候选微调抽象、PD 分离专用选择逻辑、外部服务生命周期处理。 |
| `optix/optimizer/scheduler.py` | 新增 `run_benchmark_only()`、`monitor_service` 开关、`save_result(stop_service=...)`。 |
| `optix/optimizer/store.py` | 增强结果文件路径日志，便于定位输出。 |
| `tests/regression/optix/` | 新增配置校验、微调算法、编排流程和 scheduler benchmark-only 测试。 |

性能影响：

1. `skip_pso=true` 可将真实评测次数从 `n_particles * iters + fine tune` 降低为 `1 + max_fine_tune` 以内。
2. benchmark-only 微调避免重复重启 simulator，单轮耗时不再包含服务启动和 ready 等待时间。
3. PD 分离需要额外一次 QPS probe，但后续请求速率固定，搜索路径更稳定。

兼容性影响：

1. 默认 `skip_pso=false`、`fine_tune_mode="pd_mixed"`、`manage_simulator_lifecycle=true`，保持原有主流程。
2. `n_particles=0` 或 `iters=0` 仅在 `skip_pso=true` 时合法，旧配置如果错误配置为 0 会更早暴露。
3. 输出 CSV 字段结构不变，新增逻辑继续通过 `DataStorage.save()` 记录参数、指标、错误和耗时。

## 使用说明

### 使用入口

通过 `msmodeling optix` 启动 OptiX，使用 `-e` 指定 simulator engine，使用 `-b` 指定 benchmark policy，使用 `-c` 指定 TOML 配置文件：

```bash
msmodeling optix -e vllm -b vllm_benchmark -c optix/config.toml
```

如果需要备份运行数据，可继续使用既有 `--backup` 参数：

```bash
msmodeling optix -e mindie -b ais_bench -c ./my_optix_config.toml --backup
```

### 配置示例

PD 分离微调推荐配置：

```toml
skip_pso = true
n_particles = 0
iters = 0
fine_tune_mode = "pd_disaggregation"
ttft_penalty = 0
tpot_penalty = 1
tpot_slo = 0.05
use_request_rate_calibration = false
manage_simulator_lifecycle = true
max_fine_tune = 30

[data_storage]
pso_top_k = 1

[[vllm.target_field]]
name = "CONCURRENCY"
config_position = "env"
min = 1
max = 300
dtype = "int"
value = 100

[[vllm.target_field]]
name = "NUM_PROMPTS"
config_position = "env"
min = 0
max = 0
dtype = "times"
dtype_param = { target_name = "CONCURRENCY", product = 4, dtype = "int" }
value = 0

[[vllm.target_field]]
name = "REQUESTRATE"
config_position = "env"
min = 0
max = 10000
dtype = "float"
value = 0
```

外部服务模式示例：

```toml
skip_pso = true
n_particles = 0
iters = 0
fine_tune_mode = "pd_disaggregation"
manage_simulator_lifecycle = false
```

该模式要求 simulator 服务已经由用户或外部系统启动，OptiX 只运行 benchmark 并读取指标。

### 参数说明

| 参数 | 必选/可选 | 默认值 | 说明 |
| -- | -- | -- | -- |
| `skip_pso` | 可选 | `false` | 是否跳过 PSO。开启后直接基于 baseline 做微调。 |
| `fine_tune_mode` | 可选 | `"pd_mixed"` | 微调模式。`pd_mixed` 适用于 PD 混部，`pd_disaggregation` 适用于 PD 分离。 |
| `n_particles` | 可选 | `5` | PSO 粒子数。跳过 PSO 时可以配置为 0。 |
| `iters` | 可选 | `10` | PSO 迭代轮数。跳过 PSO 时可以配置为 0。 |
| `max_fine_tune` | 可选 | `30` | 每个候选最大微调轮数。 |
| `manage_simulator_lifecycle` | 可选 | `true` | 是否由 OptiX 启停和监控 simulator。 |
| `use_request_rate_calibration` | 可选 | `true` | 是否在 PSO 阶段固定并发并校准请求速率。 |
| `ttft_penalty` | 可选 | `3.0` | TTFT 约束权重。PD 分离只关注 TPOT 时可配置为 0。 |
| `tpot_penalty` | 可选 | `3.0` | TPOT 约束权重。PD 分离建议配置为非 0。 |
| `ttft_slo` | 可选 | `0.5` | TTFT SLO，单位秒。 |
| `tpot_slo` | 可选 | `0.05` | TPOT SLO，单位秒。 |
| `step_size` | 可选 | `0.6` | PD 混部比例调整步长。 |

### 使用约束

1. `fine_tune_mode` 只支持 `"pd_mixed"` 和 `"pd_disaggregation"`。
2. `skip_pso=false` 时，`n_particles` 和 `iters` 必须大于 0。
3. `manage_simulator_lifecycle=false` 时必须同时配置 `skip_pso=true`。
4. PD 分离模式要求 benchmark 返回正数 `PerformanceIndex.throughput`，该值作为 QPS 使用。
5. PD 分离模式要求目标字段中存在 `REQUESTRATE` 和 `CONCURRENCY` 或同类并发字段。
6. `REQUESTRATE=0` 表示不限速，只建议用于 probe 或明确需要不限速压测的场景。
7. 外部服务模式下，OptiX 不负责启动、停止或恢复 simulator 服务，用户需要自行保证服务可用。

### 兼容与迁移

已有 OptiX 配置默认不需要迁移。未显式配置 `skip_pso` 时仍执行 PSO；未显式配置 `fine_tune_mode` 时仍使用 PD 混部策略；未显式配置 `manage_simulator_lifecycle` 时仍由 OptiX 管理服务生命周期。

如果旧配置误将 `n_particles` 或 `iters` 配置为 0，且没有启用 `skip_pso=true`，新版本会在配置加载阶段报错。该行为是为了避免运行时进入无效 PSO 状态。

回滚时，可将配置恢复为：

```toml
skip_pso = false
fine_tune_mode = "pd_mixed"
n_particles = 5
iters = 10
manage_simulator_lifecycle = true
```

## 测试设计

### 测试范围

本特性测试覆盖配置校验、微调算法、OptiX 流程编排、scheduler benchmark-only 执行，以及错误和边界条件。测试文件主要位于：

| 测试文件 | 覆盖内容 |
| -- | -- |
| `tests/regression/optix/test_config/test_config_config.py` | Settings 新增字段和非法组合校验。 |
| `tests/regression/optix/test_optimizer/test_experience_fine_tunning.py` | PD 混部参数调整、`REQUESTRATE=0`、PD 分离 QPS probe 和并发二分。 |
| `tests/regression/optix/test_optimizer/test_pso_optimizer.py` | `skip_pso` 编排、候选微调、PD 分离最终选择、外部服务模式。 |
| `tests/regression/optix/test_optimizer/test_schedule.py` | benchmark-only 执行、monitor_service、save_result 不停服务。 |

### 测试用例

| 用例名 | 测试类型 | 前置条件 | 操作方式 | 预期结果 |
| -- | -- | -- | -- | -- |
| UT-跳过 PSO 允许零粒子 | 单元测试 | `skip_pso=true` | 构造 `Settings(n_particles=0, iters=0)` | 配置创建成功。 |
| UT-未跳过 PSO 拒绝零粒子 | 单元测试 | `skip_pso=false` | 构造 `Settings(n_particles=0)` 或 `Settings(iters=0)` | 抛出 `ValidationError`。 |
| UT-外部服务模式配置校验 | 单元测试 | `manage_simulator_lifecycle=false` | 分别搭配 `skip_pso=true/false` | 仅 `skip_pso=true` 合法。 |
| UT-非法微调模式 | 单元测试 | `fine_tune_mode="unknown"` | 构造 Settings | 抛出 `ValidationError`。 |
| UT-不限速请求速率降压 | 单元测试 | `REQUESTRATE=0` | 调用 `FineTune.update_field()` 降低请求速率 | 从有限上界回退到正请求速率。 |
| UT-轻微 TPOT 超线退避 | 单元测试 | `TPOT` 略高于 SLO | 调用 `handle_concurrency()` | 仍按最小步长降低并发。 |
| UT-轻微 TTFT 超线退避 | 单元测试 | `TTFT` 略高于 SLO | 调用 `handle_request_rate()` | 仍按最小步长降低请求速率。 |
| UT-PD 分离 QPS 初始化 | 单元测试 | `throughput=12.39` | 调用 `init_pd_disaggregation_request_rate()` | 固定请求速率为 `12.3`。 |
| UT-PD 分离缺少 QPS | 异常测试 | `throughput=None` | 初始化固定请求速率 | 抛出 `ValueError`。 |
| UT-PD 分离 probe | 单元测试 | 参数中存在 `REQUESTRATE` | 调用 `prepare_pd_disaggregation_probe()` | probe 参数中 `REQUESTRATE=0`。 |
| UT-PD 分离并发二分 | 单元测试 | 固定请求速率已初始化 | 分别输入 TPOT 超线和未超线指标 | 超线降低并发，未超线提高并发或取中点。 |
| UT-skip_pso 编排 | 单元测试 | `skip_pso=true` | 调用 `PSOOptimizer.run_plugin()` | 不创建 PSO，不读取 top-k，只 refine baseline。 |
| UT-PD 混部 benchmark-only | 集成式单元测试 | 服务已由候选评测启动 | 调用 `_fine_tune_from_candidate()` | 后续微调调用 `run_benchmark_only()`，不重复 `scheduler.run()`。 |
| UT-PD 分离 probe 后微调 | 集成式单元测试 | `fine_tune_mode=pd_disaggregation` | 调用 `_fine_tune_from_candidate()` | 先 `scheduler.run()` probe，再 `run_benchmark_only()` 微调。 |
| UT-外部服务只跑 benchmark | 集成式单元测试 | `manage_simulator_lifecycle=false` | 调用 baseline 或 PD 分离微调 | 只调用 benchmark-only，不停止 simulator。 |
| UT-save_result 不停服务 | 单元测试 | `stop_service=false` | 调用 `Scheduler.save_result()` | 保存结果但不调用 `stop_target_server()`。 |
| UT-benchmark-only 监控 | 单元测试 | benchmark 支持 success 或 health | 调用 `run_benchmark_only()` | 更新命令、启动 benchmark、写入 `last_outcome`。 |
