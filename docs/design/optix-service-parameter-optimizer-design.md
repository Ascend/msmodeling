# 特性设计：服务化参数实测寻优功能

## 背景描述

大模型推理服务在上线前通常需要根据模型、硬件、数据集和服务框架反复调整参数，例如 MindIE 的
`BackendConfig.ScheduleConfig.maxBatchSize`、`maxPrefillBatchSize`、`maxPrefillTokens`，以及 vLLM 的
`MAX_NUM_BATCHED_TOKENS`、`MAX_NUM_SEQS`、并发数和请求速率等。手工调参依赖经验，验证周期长，且不同服务框架、
benchmark 工具和日志格式之间缺少统一编排能力，导致参数搜索、服务启动、压测、指标采集和失败恢复难以标准化。

`optix` 提供服务化参数自动寻优能力，定位为 LLM 推理性能调优工具。该工具通过统一 CLI 入口读取
TOML 配置，按用户选择的推理引擎和 benchmark 工具生成待优化参数空间，使用 PSO（Particle Swarm Optimization，
粒子群优化）搜索候选配置，并由调度器负责服务启动、benchmark 执行、健康检查、结果落盘、断点续跑和日志备份。

当前方案解决的问题包括：

1. 参数空间分散：MindIE JSON 配置、vLLM 命令行参数、benchmark 命令参数和环境变量缺少统一模型。
2. 搜索约束复杂：并发、request rate、TP/DP/MoE 并行度、比例字段和派生字段之间存在依赖，直接搜索容易产生非法组合。
3. 实验过程脆弱：服务启动失败、benchmark 失败、OOM、设备异常、网络抖动、IO 异常等需要记录
4. 结果不可追溯：每次参数、性能指标、fitness、错误信息、备份目录需要统一记录，支持后续分析和断点续跑。

核心价值：

1. 将推理服务参数调优从人工试错转为自动化搜索，降低调优门槛。
2. 支持 MindIE 与 vLLM 两类推理服务，以及 AISBench 与 vLLM benchmark 两类评测入口。
3. 通过统一数据模型和插件接口，把服务控制、benchmark、健康检查和性能目标解耦。
4. 将每次实验结果持久化为 CSV，便于断点续跑、最佳配置筛选和问题复盘。

目标：

1. 提供 `msmodeling optix` CLI，支持选择推理引擎、benchmark 策略、自定义 TOML、断点续跑和备份开关。
2. 支持通过 `target_field` 描述待优化参数，包括 `int`、`float`、`bool`、`enum`、`range`、`ratio`、`factories`、
   `times`、`ternary_factories`、`ternary_times`、`share` 等类型。
3. 支持 PSO 搜索、历史数据加载、重复参数跳过、候选结果二次 fine tune 和基于 TTFT/TPOT/吞吐/成功率的 fitness 计算。
4. 支持服务与 benchmark 运行期健康检查，并根据日志规则区分 fatal 与 retryable 错误。
5. 支持结果持久化和可选备份，确保每轮调优的参数、指标和异常可追溯。

非目标：

1. 不负责训练模型或修改模型权重。
2. 不保证在无 MindIE、vLLM、AISBench 或 NPU 运行环境时完成真实端到端压测。
3. 不替代 TensorCast/ServingCast 的性能仿真模型；本功能面向真实服务或服务框架的参数寻优编排。
4. 不在当前版本提供 Web UI，交互入口以 CLI 和 TOML 配置为主。

## 方案设计

### 场景用例

主要用户是推理服务部署和性能调优人员。典型路径如下：

1. 用户准备 MindIE 或 vLLM 服务环境、模型路径、数据集路径和 benchmark 工具。
2. 用户在 TOML 中配置优化参数、SLO、benchmark 命令、推理引擎命令和健康检查日志模式。
3. 用户执行 `msmodeling optix --engine mindie --benchmark_policy ais_bench --config ./config.toml`。
4. 工具启动默认配置服务并运行 baseline benchmark，得到初始性能指标和 fitness。
5. PSO 生成候选参数，Scheduler 按候选参数更新服务配置或环境变量，启动服务，运行 benchmark，采集指标。
6. DataStorage 将参数、性能指标、fitness、错误信息、backup 路径和 benchmark 规模写入 CSV。
7. 工具从历史结果中选取候选配置做 fine tune，最终输出最优参数和性能指标日志。

输入：

1. CLI 参数：`--engine`、`--benchmark_policy`、`--config`、`--backup`、`--load_breakpoint`。
2. TOML 配置：优化参数、SLO、输出目录、健康检查规则、MindIE/vLLM/AISBench/vLLM benchmark 配置。
3. 外部运行环境：MindIE 或 vLLM 服务命令、AISBench 或 vLLM benchmark 命令、模型和数据集路径。

输出：

1. `result/store/data_storage_<timestamp>.csv` 形式的实验记录。
2. 可选 `result/bak/` 下的服务配置、benchmark 输出和日志备份。
3. 日志中的最优参数、TTFT、TPOT、generate speed、success rate、throughput 等信息。

### 整体思路

整体采用“配置模型 + 插件接口 + 调度器 + PSO 优化器”的分层设计。配置层使用 pydantic-settings 将 TOML 与环境变量合并
为 `Settings` 对象；插件层将推理服务抽象为 `SimulatorInterface`，将 benchmark 工具抽象为 `BenchmarkInterface`；
调度层统一执行服务启动、benchmark、健康检查、重试、停止和结果保存；优化层负责生成候选参数、计算 fitness、筛选最佳结果。

参数空间通过 `OptimizerConfigField` 描述，每个字段包含名称、配置位置、取值范围、数据类型、当前值和派生规则。PSO 只处理
连续空间向量，`map_param_with_value()` 负责把连续向量转换成真实参数，并处理枚举选择、布尔转换、比例值、派生值和非法组合修复。
这种设计避免把服务框架特有约束写死在 PSO 算法里。

运行期间，Scheduler 每次接收一组候选参数，将其映射为 `simulate_run_info`，再调用 simulator 和 benchmark 的统一接口。
MindIE simulator 会修改 JSON 配置中的 `BackendConfig.*` 字段并启动服务；vLLM simulator 会通过命令行和环境变量注入参数。
benchmark 运行完成后返回统一的 `PerformanceIndex`，由 `PerformanceTuner.minimum_algorithm()` 转换为需要最小化的 fitness。

错误处理采用 hook 化健康检查。服务启动轮询和运行期监控会读取 simulator/benchmark 最新日志，根据 TOML 中配置的 fatal/retryable
模式生成 `HealthCheckResult`。fatal 错误直接终止当前候选并向上抛出，retryable 错误触发 Scheduler 重试，最多重试 3 次。

### 系统架构

![plantuml-diagram (4).png](https://raw.gitcode.com/user-images/assets/8428112/8aeb8f11-1b8f-4d08-87e2-6639847d0bdd/plantuml-diagram__4_.png 'plantuml-diagram (4).png')

该图表示 optix 的主要协作关系：CLI 负责装配对象，PSO 负责搜索，Scheduler 负责执行一次真实评测，Simulator 和 Benchmark
负责对接外部服务与压测工具，DataStorage 负责记录实验过程。

### 核心流程

![plantuml-diagram (3).png](https://raw.gitcode.com/user-images/assets/8428112/81009f08-a6d1-4d99-a66e-997ce723c46c/plantuml-diagram__3_.png 'plantuml-diagram (3).png')

该流程体现了一次完整寻优任务。baseline 用于确认默认服务可运行并提供初始指标，PSO 阶段负责全局搜索，fine tune 阶段对 top
候选做局部改进，最终按 SLO 和吞吐策略选择结果。

### 参数映射与约束修复

`OptimizerConfigField` 是参数空间的核心数据模型：

| 字段 | 说明 |
| -- | -- |
| `name` | 参数名称，同时用于 CSV 列名和环境变量名称 |
| `config_position` | 参数写入位置，支持 MindIE JSON 路径或 `env` |
| `min` / `max` | 搜索空间上下界 |
| `dtype` | 参数类型或派生规则 |
| `value` | 当前值 |
| `dtype_param` | 枚举列表、依赖字段名、乘除关系或派生规则配置 |
| `constant` | 固定值；固定字段不进入 PSO 维度 |

参数映射规则：

1. `int`、`float`、`bool`：将连续粒子值转换成目标类型。
2. `enum`：将连续区间切分为枚举段，映射到最近枚举值；支持数字和字符串枚举。
3. `range`：配置加载阶段转换为枚举列表。
4. `ratio`：按 `self_ratio * target.value` 计算实际值。
5. `factories`：按 `product / target.value` 计算实际值。
6. `times`：按 `product * target.value` 计算实际值。
7. `ternary_factories`：按 `product / (field_a * field_b)` 计算实际值，并支持最小值、最大值和整除约束。
8. `ternary_times`：按 `product * field_a * field_b` 计算实际值。
9. `share`：按 `target.min + target.max - target.value` 计算互补值。

针对 `ternary_factories`，方案提供优先级感知修复：

1. 使用 `DecodeContext` 记录粒子编号、粒子数和迭代轮次。
2. `priority_policy=fixed` 时按显式优先级修复。
3. `priority_policy=balanced` 时按粒子编号和迭代轮次切换字段优先级，降低固定修复顺序导致的搜索偏置。
4. 修复分两阶段：先固定高优先级字段只调整低优先级字段；失败后同时调整两个字段。
5. 若修复失败，则根据上下界做 clamp；如果整除约束仍不满足，则抛出异常。

此外，参数映射包含业务后处理：

1. `maxPrefillBatchSize` 为 0 时强制设为 1。
2. `supportSelectBatch` 为 false 时，将 `prefillTimeMsPerReq` 和 `decodeTimeMsPerReq` 置 0。
3. `CONCURRENCY`、`MAXCONCURRENCY`、`REQUESTRATE` 在 PSO 搜索期间可被固定或按 benchmark 结果二次调整。

### Fitness 设计

`PerformanceIndex` 统一表示 benchmark 指标：

| 指标 | 说明 |
| -- | -- |
| `generate_speed` | 输出 token 生成速度 |
| `time_to_first_token` | TTFT，单位为秒 |
| `time_per_output_token` | TPOT，单位为秒 |
| `success_rate` | 成功请求比例 |
| `throughput` | 请求吞吐 |

`PerformanceTuner.minimum_algorithm()` 将指标转换为最小化目标：

1. `generate_speed` 越高，`generate_speed_target / generate_speed` 越低。
2. TTFT、TPOT 和 success rate 使用指数惩罚，超过 SLO 时 fitness 快速增大。
3. 默认权重为 `w_gen=0.4`、`w_ft=0.2`、`w_pot=0.3`、`w_succ=0.1`。
4. 指标缺失、吞吐无效或成功率无效时返回 `inf`，避免非法结果被选为最优。

最终最佳参数选择策略：

1. 若未启用 TTFT/TPOT penalty，则选择 `generate_speed` 最大的候选。
2. 若只启用 TPOT penalty，则优先选择满足 TPOT 阈值的最高吞吐候选，否则选择 TPOT 违约最小的候选。
3. 若同时启用 TTFT 和 TPOT penalty，则优先选择同时满足阈值的最高吞吐候选，否则选择综合违约最小的候选。

### 服务与 Benchmark 插件

内置注册表由 `register_ori_functions()` 初始化：

| 注册名 | 类型 | 实现类 | 说明 |
| -- | -- | -- | -- |
| `mindie` | simulator | `Simulator` | 修改 MindIE JSON 配置并启动 MindIE 服务 |
| `vllm` | simulator | `VllmSimulator` | 通过 `vllm serve` 启动服务 |
| `ais_bench` | benchmark | `AisBench` | 使用 `ais_bench` 运行 MindIE benchmark |
| `vllm_benchmark` | benchmark | `VllmBenchMark` | 使用 `vllm bench serve` 运行 vLLM benchmark |

插件扩展接口：

1. 新增服务引擎需继承 `SimulatorInterface`，实现 `base_url`、`update_command()`，必要时重写 `update_config()`、`health()`、`stop()`。
2. 新增 benchmark 需继承 `BenchmarkInterface`，实现 `update_command()` 和 `get_performance_index()`。
3. 使用 `register_simulator(name, cls)` 或 `register_benchmarks(name, cls)` 注册，注册名会进入 CLI `--engine` 或
   `--benchmark_policy` 候选集合。

### 调度与健康检查

Scheduler 的职责是执行一次候选参数评测：

1. 根据 `params` 和 `target_field` 生成 `simulate_run_info`。
2. 将字段同步给 simulator 和 benchmark，触发命令或配置更新。
3. 启动服务并在 `wait_start_time` 内轮询健康状态。
4. 启动 benchmark，并在 `particles_time_out` 内监控服务和 benchmark 状态。
5. 采集 `PerformanceIndex`。
6. 保存结果，按需备份配置和日志，停止服务和 benchmark。

健康检查使用 hook 机制：

| Hook 点 | 作用 |
| -- | -- |
| `ServiceHookPoint.STARTUP_POLLING` | 服务启动阶段检查服务日志 |
| `ServiceHookPoint.RUNTIME_MONITOR` | 服务运行阶段检查服务日志 |
| `BenchmarkHookPoint.RUNTIME_MONITOR` | benchmark 运行阶段检查 benchmark 日志 |

错误分类：

1. fatal：如 OOM、设备错误，立即停止当前流程并抛出 `FatalError`。
2. retryable：如网络、IO 短暂异常，抛出 `RetryableError` 并触发重试。
3. unknown：hook 自身异常时按 fatal 处理。

日志匹配规则由 TOML 的 `[health_check.*]` 配置提供，匹配结果携带最新日志片段，便于问题定位。

### 数据模型与文件变更

新增或涉及的主要模块：

| 路径 | 作用 |
| -- | -- |
| `pyproject.toml` | optix 独立包配置和 `msmodeling` 入口 |
| `cli/main.py` | CLI 顶层入口，转发 `msmodeling inference` / `msmodeling optix` |
| `optix/config.toml` | 默认配置模板 |
| `optix/config/config.py` | Settings、参数字段、性能指标、参数映射和派生字段规则 |
| `optix/config/custom_command.py` | MindIE/vLLM/AISBench/vLLM benchmark 命令构造 |
| `optix/optimizer/optimizer.py` | PSOOptimizer、fine tune 编排和主函数 |
| `optix/optimizer/scheduler.py` | 服务、benchmark、健康检查、重试和保存调度 |
| `optix/optimizer/store.py` | CSV 持久化、历史数据加载和最佳结果筛选 |
| `optix/optimizer/health_check.py` | 健康检查 hook 和错误分类 |
| `optix/optimizer/interfaces/` | simulator、benchmark、custom process 抽象接口 |
| `optix/optimizer/plugins/` | 内置 MindIE/vLLM simulator 与 AISBench/vLLM benchmark |
| `tests/regression/optix/` | 当前 optix 单元测试 |

运行期输出：

| 输出 | 默认位置 | 说明 |
| -- | -- | -- |
| 结果目录 | `./result` | `Settings.output` 默认值 |
| CSV 存储 | `./result/store/data_storage_<timestamp>.csv` | 每轮参数和指标记录 |
| simulator 输出 | `./result/simulator` | simulator 相关输出 |
| vLLM 输出 | `./result/vllm` | vLLM 服务和 benchmark 输出 |
| MindIE 输出 | `./result/mindie` | MindIE 相关输出 |
| AISBench 输出 | `./result/ais_bench` | AISBench 输出 |
| 备份目录 | `./result/bak` | `--backup` 开启后创建 |

### 功能与性能影响

功能影响：

1. 新增独立 `optix` Python 包和 `msmodeling optix` CLI。
2. 新增服务参数自动寻优能力，支持 MindIE/vLLM 与 AISBench/vLLM benchmark 组合。
3. 新增插件注册机制，允许扩展服务引擎和 benchmark。
4. 新增健康检查 hook，支持按日志规则区分 fatal/retryable 错误。

性能影响：

1. 搜索成本与 `n_particles * iters` 近似线性相关，每个候选会触发一次或两次 benchmark。
2. `sample_size` 可降低 benchmark 请求规模，用于加速初筛。
3. `MAX_ITER_NUM=200` 限制 `n_particles`、`iters` 和 `max_fine_tune` 的上限，避免配置错误导致无限搜索。
4. 重复参数跳过可减少无效评测。

可靠性影响：

1. 服务和 benchmark 进程由 `CustomProcess` 统一管理，启动前会清理残留进程，停止时会清理子进程。
2. MindIE simulator 会在启动前备份原始配置，停止时恢复默认配置。
3. `--backup` 可保存日志和配置，便于失败后排查。
4. 数据存储使用 CSV，便于人工查看，但并发多进程写入不是当前设计目标。

安全性影响：

1. CLI 检测到 root 用户时输出安全告警，不建议以 root 运行。
2. CSV 写入前通过 `sanitize_csv_value()` 规避公式注入风险。
3. 进程启动依赖用户配置的命令和 `others` 字段，使用方需要保证配置来源可信。

兼容性影响：

1. 默认配置读取多个路径，兼容安装目录、用户目录、运行目录和环境变量指定配置。
2. `--config` 追加自定义 TOML，并允许额外字段，便于灰度新增配置。
3. 对 MindIE 默认安装路径和 `MIES_INSTALL_PATH` 均有兼容逻辑。
4. vLLM 和 AISBench 依赖命令在 `PATH` 中可发现。

## 使用说明

### 安装与入口

`optix` 已并入 `msmodeling` 主 wheel，通过根目录 `pyproject.toml` 打包。安装后提供入口：

```bash
msmodeling optix [OPTIONS]
```

常用命令：

```bash
msmodeling optix \
  --engine mindie \
  --benchmark_policy ais_bench \
  --config ./config.toml \
  --backup
```

```bash
msmodeling optix \
  --engine vllm \
  --benchmark_policy vllm_benchmark \
  --config ./config.toml \
  --load_breakpoint
```

### CLI 参数

| 参数 | 可选/必选 | 默认值 | 说明 |
| -- | -- | -- | -- |
| `--engine` / `-e` | 可选 | `mindie` | 推理服务引擎，内置支持 `mindie`、`vllm`，也支持已注册自定义 simulator |
| `--benchmark_policy` / `-b` | 可选 | `ais_bench` | benchmark 策略，内置支持 `ais_bench`、`vllm_benchmark`，也支持已注册自定义 benchmark |
| `--config` / `-c` | 可选 | `None` | 自定义 TOML 配置路径，支持绝对路径、相对路径和当前目录文件名 |
| `--backup` | 可选 | `False` | 是否将服务配置、benchmark 输出和日志备份到 `output/bak` |
| `--load_breakpoint` / `-lb` | 可选 | `False` | 是否从历史 CSV 加载真实评测结果并断点续跑 |

### 配置加载优先级

配置由 `Settings` 统一加载，默认配置路径包括：

1. `INSTALL_PATH/model_eval_state.toml`
2. `~/model_eval_state.toml`
3. `RUN_PATH/model_eval_state.toml`
4. `INSTALL_PATH/config.toml`
5. `INSTALL_PATH/optix/config.toml`
6. `~/config.toml`
7. `RUN_PATH/config.toml`
8. `MODEL_EVAL_STATE_CONFIG_PATH` 或 `model_eval_state_config_path` 指定路径
9. CLI `--config` 指定路径

环境变量前缀为 `model_eval_state_`。输出目录也可通过 `MODEL_EVAL_STATE_OUTPUT` 或
`model_eval_state_output` 指定。是否启用模拟模式由 `MODEL_EVAL_STATE_SIMULATE` 或
`model_eval_state_simulate` 控制。

### 主要配置参数

| 参数 | 类型 | 默认值 | 说明 |
| -- | -- | -- | -- |
| `n_particles` | int | `5` | PSO 粒子数量，代码上限 200 |
| `iters` | int | `10` | PSO 迭代次数，代码上限 200 |
| `ttft_slo` | float | `0.5` | TTFT SLO，单位秒 |
| `tpot_slo` | float | `0.05` | TPOT SLO，单位秒 |
| `success_rate_slo` | float | `1.0` | 成功率 SLO |
| `ttft_penalty` | float | `3.0` | TTFT 指数惩罚系数 |
| `tpot_penalty` | float | `3.0` | TPOT 指数惩罚系数 |
| `success_rate_penalty` | float | `5.0` | 成功率指数惩罚系数 |
| `sample_size` | Optional[int] | `None` | benchmark 采样请求数；为空时使用 benchmark 原始请求数 |
| `data_storage.pso_top_k` | int | `3` | PSO 后进入 fine tune 的 top 结果数量 |

### target_field 配置

MindIE 示例：

```toml
[[mindie.target_field]]
name = "max_batch_size"
config_position = "BackendConfig.ScheduleConfig.maxBatchSize"
min = 10
max = 1000
dtype = "int"

[[mindie.target_field]]
name = "max_prefill_batch_size"
config_position = "BackendConfig.ScheduleConfig.maxPrefillBatchSize"
min = 0.1
max = 0.7
dtype = "ratio"
dtype_param = "max_batch_size"

[[mindie.target_field]]
name = "CONCURRENCY"
config_position = "env"
min = 1
max = 1001
dtype = "int"
value = 100
```

vLLM 示例：

```toml
[[vllm.target_field]]
name = "MAX_NUM_BATCHED_TOKENS"
config_position = "env"
min = 8192
max = 65536
dtype = "int"
value = 8192

[[vllm.target_field]]
name = "MAX_NUM_SEQS"
config_position = "env"
min = 32
max = 512
dtype = "int"
value = 64
```

字段约束：

1. `config_position="env"` 表示写入环境变量，同时可替换命令中的 `$NAME`。
2. `config_position` 以 `BackendConfig` 开头时表示写入 MindIE JSON 配置。
3. `min == max` 或 `constant` 不为空的字段会作为固定值，不进入 PSO 搜索维度。
4. `range` 类型需要配置 `dtype_param` 作为步长，加载后会转换为枚举。
5. 派生字段依赖的目标字段必须同时出现在 `target_field` 中。

### Benchmark 配置

AISBench：

```toml
[ais_bench.command]
models = "models"
datasets = "datasets"
mode = "perf"
num_prompts = 3000
```

vLLM benchmark：

```toml
[vllm_benchmark.command]
host = "127.0.0.1"
port = "8000"
model = "/path/to/model"
served_model_name = "model_name"
dataset_name = "random"
num_prompts = 3000
others = ""
```

### 健康检查配置

```toml
[health_check]
log_snippet_length = 200

[health_check.service_errors.fatal_patterns]
out_of_memory = ["out of memory", "OOM killed"]
device_error = ["NPU error", "device fault"]

[health_check.service_errors.retryable_patterns]
network_error = ["connection reset", "timeout"]
io_error = ["file not found", "permission denied"]

[health_check.benchmark_errors.fatal_patterns]
out_of_memory = ["out of memory"]
device_error = ["device fault"]

[health_check.benchmark_errors.retryable_patterns]
network_error = ["connection refused", "timeout"]
io_error = ["IO error"]
```

使用约束：

1. 建议使用普通用户运行，不建议 root 运行。
2. MindIE 场景需要可访问 MindIE 配置文件；默认路径为
   `/usr/local/Ascend/mindie/latest/mindie-service/conf/config.json`，也支持 `MIES_INSTALL_PATH`。
3. vLLM 场景要求 `vllm` 命令在 `PATH` 中。
4. AISBench 场景要求 `ais_bench` 命令在 `PATH` 中。
5. 部分真实 benchmark 需要 NPU、模型文件和数据集文件，当前工具不负责准备这些资源。
6. `others` 字段会被拆分并拼接到命令行，配置来源需要可信。
7. CSV 历史数据用于断点续跑时，会按 benchmark 请求数和真实评测标记过滤。

兼容与迁移：

1. 该功能位于 `optix`，对 `tensor_cast/`、`serving_cast/` 主流程无直接运行时影响。
2. 旧用户可继续通过默认 `config.toml` 使用，新增字段通过 pydantic-settings 允许扩展。
3. 回滚路径是停止 optix 进程并恢复原始服务配置；MindIE simulator 停止时会写回默认配置。
4. 若启用 `--backup`，可从 `output/bak` 查看每轮变更前后的日志和配置。

## 测试设计

### 单元测试

当前单元测试位于 `tests/regression/optix/`，由 `bash scripts/run_regression.sh` 统一收集执行。

| 用例名 | 测试类型 | 前置条件 | 操作方式 | 预期结果 |
| -- | -- | -- | -- | -- |
| UT-CLI optix 转发 | 单元测试 | mock `optix.optimizer.optimizer.main` | 执行 `msmodeling optix` | 调用 optix 主函数 |
| UT-CLI 参数透传 | 单元测试 | mock `sys.argv` | 执行 `msmodeling optix --some-arg value` | 剩余参数正确传给 optix |
| UT-参数字段校验 | 单元测试 | 构造 `OptimizerConfigField` | 测试 min/max、constant、dtype 转换 | 非法边界抛错，固定值自动识别 |
| UT-range 转 enum | 单元测试 | 配置 `dtype="range"` | 调用配置加载逻辑 | 生成离散枚举候选 |
| UT-连续参数映射 | 单元测试 | 构造 numpy 粒子位置 | 调用 `map_param_with_value()` | int/bool/enum 正确转换 |
| UT-ratio/factories/times | 单元测试 | 构造依赖字段 | 调用 `update_optimizer_value()` | 派生字段按规则计算 |
| UT-ternary_factories 修复 | 单元测试 | 构造 TP/PP/DP 约束 | 调用修复函数 | 优先级修复或 clamp 行为符合预期 |
| UT-PSO 边界构造 | 单元测试 | 构造含固定字段的 target_field | 调用 `constructing_bounds()`、`dimensions()` | 固定字段不进入 PSO 维度 |
| UT-重复参数跳过 | 单元测试 | mock scheduler 和 `_seen_params` | 调用 `_skip_if_duplicate()` | 重复参数写入 inf fitness 并跳过真实评测 |
| UT-fitness 计算 | 单元测试 | 构造 `PerformanceIndex` | 调用 `minimum_algorithm()` | 指标缺失返回 inf，正常指标返回有限 cost |
| UT-best_params 策略 | 单元测试 | 构造候选指标列表 | 调用 `best_params()` | 按 penalty/SLO/吞吐选择候选 |
| UT-Scheduler 健康检查初始化 | 单元测试 | mock simulator/benchmark | 创建 Scheduler | 默认 hook 注册完成 |
| UT-fatal 错误处理 | 单元测试 | 构造 fatal `ErrorContext` | 调用 `_handle_error()` | 抛出 `FatalError` |
| UT-retryable 错误处理 | 单元测试 | 构造 retryable `ErrorContext` | 调用 `run_target_server()` | 触发重试，成功后退出 |
| UT-DataStorage 保存 | 单元测试 | 临时 store 目录 | 调用 `save()` | CSV 表头和值写入正确 |
| UT-插件注册 | 单元测试 | 自定义接口子类 | 调用注册函数 | 注册表写入成功，非法类型抛错 |

### 集成测试

| 用例名 | 测试类型 | 前置条件 | 操作方式 | 预期结果 |
| -- | -- | -- | -- | -- |
| IT-MindIE + AISBench dry-run | 集成测试 | mock MindIE 配置、mock ais_bench 输出 | 创建 `Simulator`、`AisBench`、`Scheduler` | 配置写入、benchmark 指标解析、CSV 保存成功 |
| IT-vLLM + vLLM benchmark dry-run | 集成测试 | mock `vllm serve` 和 benchmark JSON 输出 | 运行 Scheduler 单轮评测 | 命令参数替换、结果解析和进程停止成功 |
| IT-自定义 TOML 覆盖 | 集成测试 | 准备默认配置和自定义配置 | 执行 `msmodeling optix --config custom.toml` | 自定义配置覆盖默认配置 |
| IT-断点续跑 | 集成测试 | 准备历史 `data_storage_*.csv` | 使用 `--load_breakpoint` 运行 | 历史 position/cost 被加载并用于 PSO 初始化 |
| IT-健康检查重试 | 集成测试 | mock 日志出现 retryable pattern | 执行单轮 Scheduler | 触发重试且保存错误上下文 |
| IT-健康检查 fatal | 集成测试 | mock 日志出现 fatal pattern | 执行单轮 Scheduler | 当前候选终止，不继续重试 |

### 端到端测试

真实 E2E 依赖 MindIE/vLLM、模型文件、benchmark 工具和硬件资源，建议在具备 NPU 的环境中执行，并使用 `@pytest.mark.npu`
或独立流水线隔离。

| 用例名 | 测试类型 | 前置条件 | 操作方式 | 预期结果 |
| -- | -- | -- | -- | -- |
| E2E-MindIE 自动寻优 | 端到端测试 | NPU、MindIE、模型、AISBench、数据集可用 | 执行 `msmodeling optix --engine mindie --benchmark_policy ais_bench --backup` | 完成 baseline、PSO、fine tune，生成 CSV 和备份 |
| E2E-vLLM 自动寻优 | 端到端测试 | vLLM 服务可运行，benchmark 数据可用 | 执行 `msmodeling optix --engine vllm --benchmark_policy vllm_benchmark` | 生成最优参数记录，服务进程被清理 |
| E2E-自定义参数空间 | 端到端测试 | 配置包含派生字段和 env 字段 | 使用自定义 TOML 运行 | 输出参数均满足依赖约束 |

### 异常与边界测试

| 用例名 | 测试类型 | 前置条件 | 操作方式 | 预期结果 |
| -- | -- | -- | -- | -- |
| 异常-配置文件不存在 | 异常测试 | 指定不存在 TOML | 执行 `--config missing.toml` | 输出错误并返回 |
| 异常-TOML 格式错误 | 异常测试 | 准备非法 TOML | 执行 `--config invalid.toml` | 抛出配置校验错误 |
| 异常-MindIE 配置不存在 | 异常测试 | `mindie.config_path` 不存在 | 创建 `Simulator` | 抛出 `FileNotFoundError` |
| 异常-vLLM 命令缺失 | 异常测试 | PATH 中无 `vllm` | 创建 `VllmCommand` | 抛出命令缺失错误 |
| 异常-AISBench 命令缺失 | 异常测试 | PATH 中无 `ais_bench` | 创建 `AisBenchCommand` | 抛出命令缺失错误 |
| 边界-粒子数过大 | 边界测试 | `n_particles > 200` | 创建 PSOOptimizer | 自动截断为 200 |
| 边界-迭代数过大 | 边界测试 | `iters > 200` | 创建 PSOOptimizer | 自动截断为 200 |
| 边界-request rate 固定 | 边界测试 | `REQUESTRATE.min == max` | 调用 `run_with_request_rate()` | 跳过第二次 request rate 评测 |
| 边界-备份目录超限 | 边界测试 | `bak` 大于 1GB | 调用 `set_back_up_path()` | 禁用本轮备份路径 |

### 性能与兼容性测试

| 用例名 | 测试类型 | 前置条件 | 操作方式 | 预期结果 |
| -- | -- | -- | -- | -- |
| 性能-采样模式 | 性能测试 | benchmark 请求数大于 `sample_size` | 运行 PSO 搜索 | 搜索阶段请求数被临时降至 `sample_size` |
| 性能-重复参数跳过 | 性能测试 | PSO 生成重复粒子 | 运行 `op_func()` | 重复参数不启动服务和 benchmark |
| 兼容-默认配置路径 | 兼容测试 | 多路径配置存在 | 初始化 Settings | 按 pydantic-settings 顺序合并配置 |
| 兼容-MIES_INSTALL_PATH | 兼容测试 | 设置 `MIES_INSTALL_PATH` | 获取 MindIE 配置路径 | 使用 MindIE 新安装路径 |
| 兼容-命令 JSON 参数 | 兼容测试 | `others` 包含 JSON 参数 | 调用 `CustomProcess.before_run()` | 合并参数被拆分为独立 CLI 参数 |

完成标准：

1. `tests/regression/optix/` 现有 UT 全部通过。
2. `bash scripts/run_regression.sh` 能收集并运行 optix regression 用例。
3. 在具备外部依赖的环境中完成至少一组 MindIE + AISBench 或 vLLM + vLLM benchmark 端到端验证。
4. 失败场景下 CSV、日志和备份信息足够定位问题。
