# 特性设计：服务化参数实测寻优功能适配pd分离参数搜索

## 修订记录

| 日期 | 修订版本 | 修改描述 | 作者 | RFC文档 |
| -- | -- | -- | -- | -- |
| 2026-09-08 | 1.1 | 统一 P/D QPS 为 benchmark 实测请求吞吐，补充资源受限时的实例推荐说明 | 待确认 | [OptiX 实测寻优适配 PD 分离场景](../RFC/rfc_optix_pd_disaggregation_real_tuning_zh.md) |
| 2026-09-08 | 1.2 | 明确 P/D 搜索中 `REQUESTRATE=0` 代表不限速和最大施压速率 | 待确认 | [OptiX 实测寻优适配 PD 分离场景](../RFC/rfc_optix_pd_disaggregation_real_tuning_zh.md) |
| 2026-09-08 | 1.3 | 允许 `top_k=0` 跳过 P/D 阶段 FineTune，保留搜索和配比推荐 | 待确认 | [OptiX 实测寻优适配 PD 分离场景](../RFC/rfc_optix_pd_disaggregation_real_tuning_zh.md) |
| 2026-09-08 | 1.4 | Decode 搜索增加 prefix cache 预热轮次，明确仅以最后一轮 benchmark 计算 QPS | 待确认 | [OptiX 实测寻优适配 PD 分离场景](../RFC/rfc_optix_pd_disaggregation_real_tuning_zh.md) |
| 2026-09-09 | 1.5 | 完善固定请求速率时的 TTFT 并发调节，并允许 PD 分离实测 QPS 作为可继续调节的初始速率 | 待确认 | [OptiX 实测寻优适配 PD 分离场景](../RFC/rfc_optix_pd_disaggregation_real_tuning_zh.md) |
| 2026-09-09 | 1.7 | 将校准模式与多轮 benchmark 的冲突提前为配置校验错误 | 待确认 | [OptiX 实测寻优适配 PD 分离场景](../RFC/rfc_optix_pd_disaggregation_real_tuning_zh.md) |
| 2026-09-09 | 1.8 | 强制 PD 服务参数搜索执行 PSO，禁止 `pd_disagg` 模式与 `skip_pso=true` 组合 | 待确认 | [OptiX 实测寻优适配 PD 分离场景](../RFC/rfc_optix_pd_disaggregation_real_tuning_zh.md) |
| 2026-09-09 | 1.9 | 明确 PD 服务参数搜索必须管理 simulator 生命周期，禁止静默忽略全局配置 | 待确认 | [OptiX 实测寻优适配 PD 分离场景](../RFC/rfc_optix_pd_disaggregation_real_tuning_zh.md) |
| 2026-09-10 | 1.10 | `--mode` 缺省时固定进入旧版单阶段流程，PD 搜索仅允许 CLI 显式启用 | 待确认 | [OptiX 实测寻优适配 PD 分离场景](../RFC/rfc_optix_pd_disaggregation_real_tuning_zh.md) |
| 2026-09-10 | 1.11 | 将默认模式值由 `service_param` 更名为 `standard`，明确其为可包含 PSO 和 FineTune 的通用寻优流程 | 待确认 | [OptiX 实测寻优适配 PD 分离场景](../RFC/rfc_optix_pd_disaggregation_real_tuning_zh.md) |

## 背景描述

PD 分离部署需要先分别确定 Prefill、Decode 的服务化参数，再依据两侧单实例处理能力推荐实例配比，最后在完整 PD 服务上调整并发和请求速率。
本版采用两个相互独立的入口：

- `pd_disagg` 只搜索 P/D 服务参数、计算 QPS 并推荐实例配比。
- 用户启动 PD 服务后，使用普通 `standard` 模式直接执行 benchmark-only FineTune。

适用范围：本详设仅针对需要计算并推荐 P/D 实例配比的 PD 分离服务搜索。无需配比推荐的普通 PD 分离服务，以及 PD 混部服务，直接使用仓内已有 OptiX 插件寻优，不进入 `pd_disagg` 流程。

目标：

- P/D 搜索完成后输出服务参数、QPS、PD ratio、整数实例推荐和阶段产物。
- Decode 搜索在同一服务实例上先填充 prefix cache，再以相同负载复测，使实测吞吐近似 D 节点 QPS。
- 微调阶段固定 `skip_pso=true`、`manage_simulator_lifecycle=false`，只运行 benchmark。
- 微调只调整 benchmark 侧 `CONCURRENCY` 和 `REQUESTRATE`，不依赖多机 simulator 插件。
- 请求速率固定且 TTFT 超过 SLO 时，通过降低并发改善 TTFT；请求速率可调时仍优先使用请求速率调节 TTFT。
- PD 分离 FineTune 将不限速探测得到的 QPS 作为请求速率初值，并在二分并发无法继续移动后按 TTFT 调整请求速率。
- 保持默认 OptiX 服务参数寻优行为不变。

非目标：不内置 PD 服务启动；不接管用户服务生命周期；不在微调阶段自动读取 P/D 搜索产物；不改变 P/D 搜索算法。

## 方案设计

### 场景用例

| 场景 | 触发条件 | 输入 | 输出 |
| -- | -- | -- | -- |
| P/D 服务参数搜索 | 执行 `--mode pd_disagg` | P/D vLLM 配置、搜索空间、SLO、设备约束 | P/D 最优参数、QPS、PD ratio、实例推荐 |
| 用户启动 PD 服务 | P/D 搜索完成 | 推荐参数和实例配比 | OptiX 主机可访问的 PD 服务 |
| benchmark 微调 | PD 服务已启动 | 普通 OptiX 微调配置、benchmark 负载范围 | 最优并发、请求速率和端到端指标 |

### 整体思路

`PdDisaggOrchestrator` 只执行 Prefill 和 Decode 两个阶段。两个阶段分别复用现有 `PSOOptimizer`、`Scheduler`、`DataStorage` 和 vLLM/benchmark 插件，完成服务参数搜索后计算 QPS 和实例配比，以 `service_search_completed` 状态结束。最终解析模式为 `pd_disagg` 时要求顶层 `skip_pso=false` 且 `manage_simulator_lifecycle=true`，否则在创建阶段 runner 前报错，避免服务参数搜索静默退化为 baseline + FineTune，或忽略用户配置后仍管理服务。`OptixPhaseRunner` 将已校验的生命周期配置透传给 `PSOOptimizer`，并在直接调用时再次防御性校验。`pd_disagg.top_k=0` 时两个阶段不执行 FineTune，仍保留 PSO 最优候选的完整验证和配比计算。每个阶段可通过 `benchmark_run_count` 指定同一服务实例上的 benchmark 总执行次数；当全局 `use_request_rate_calibration=false` 时，前 N-1 轮只预热，只有第 N 轮进入适应度、CSV 和阶段 QPS。

运行模式只由 CLI `--mode` 控制。省略该参数时固定使用默认值 `standard`，完整复用引入 PD 流程前的通用寻优逻辑，包括 baseline、可选 PSO 和 FineTune；只有显式指定 `--mode pd_disagg` 才进入 P/D 服务参数搜索和配比推荐。配置文件不提供启用 PD 模式的第二入口。

Decode 示例把 `benchmark_run_count` 设为 2，并在服务端启用 prefix caching。第一轮使用固定 seed 的请求填充 prefix cache，第一轮结束后只停止 benchmark，不重启 Decode 服务；第二轮使用完全相同的服务参数、并发、请求速率、请求数、输入输出长度、请求顺序和 prompt，再以第二轮 `throughput` 近似 D 节点 QPS。多轮 benchmark 仅允许 `use_request_rate_calibration=false`；校准模式下任一阶段配置 `benchmark_run_count>1` 会在配置加载时直接报错。

微调不再进入 `PdDisaggOrchestrator`。用户根据第一步结果启动 PD 服务，随后使用独立配置执行普通 `standard` 模式。配置将 `skip_pso` 设为 `true`、`manage_simulator_lifecycle` 设为 `false`，并清空 vLLM 服务侧搜索字段。优化器建立 benchmark 基线后直接执行 `pd_mixed` FineTune，所有评测均走 `Scheduler.run_benchmark_only(..., monitor_service=false)`。`pd_mixed` 优先根据 TPOT 调整并发，再根据 TTFT 调整请求速率；若请求速率通过 `constant` 或 `min=max` 固定，TTFT 超过 SLO 时改为降低并发，TTFT 已满足时不额外使用该回退规则。

benchmark 的 host、port、model 和数据集由 `vllm_benchmark` 或其他 benchmark 插件配置提供。OptiX 不检查服务由什么工具启动，也不会停止用户服务；服务不可达或指标无效时按普通微调流程报错。

### 系统架构

![image.png](https://raw.gitcode.com/user-images/assets/8428112/32a37aa3-0403-4e93-ba82-6f607ddb9ed9/image.png 'image.png')
第一步和第二步没有恢复关系；两者仅通过用户选择的服务参数和实际部署衔接。

### 核心流程

1. 用户配置 `skip_pso=false`、`manage_simulator_lifecycle=true` 并执行 `msmodeling optix --mode pd_disagg`；若最终解析模式为 `pd_disagg` 但任一配置不满足，工具直接报配置错误。
2. 编排器预检 Prefill/Decode 的 simulator、benchmark 和部署环境。
3. 顺序搜索 P/D 服务参数。`top_k=0` 时跳过两阶段 FineTune；`top_k>0` 时对排名靠前的候选执行阶段 FineTune。Prefill benchmark 输出长度配置为 1 token。Decode 对每个候选复用同一服务连续运行两次相同 benchmark，第一轮填充 prefix cache，第二轮才采集指标。P/D QPS 均取各自有效 benchmark 返回的请求吞吐 `throughput`（req/s）。
4. 计算 `pd_ratio=D_QPS/P_QPS`，枚举可行实例数，写入 P/D 阶段产物、`pd_ratio_candidates.csv` 和 `pd_disagg_summary.json`。
5. summary 状态更新为 `service_search_completed`，本次 `pd_disagg` 执行结束。
6. 用户使用推荐参数和实例配比启动 PD 服务。
7. 用户执行普通 `standard` 模式，并通过配置跳过 PSO、关闭 simulator 生命周期管理、只声明 benchmark 搜索字段。
8. 优化器运行 benchmark 基线和 `pd_mixed` FineTune，输出最优并发、请求速率和性能指标。

### 数据模型与产物

- `PdDisaggConfig` 包含设备约束、输出目录、Prefill 配置和 Decode 配置，不包含恢复或 Final 阶段字段。`top_k` 取值范围为大于等于 0；0 表示跳过 P/D 阶段 FineTune，但仍保留一组最优候选用于配比和实例推荐。每个 `PdDisaggPhaseConfig` 增加 `benchmark_run_count`，类型为正整数、默认值为 1；当全局 `use_request_rate_calibration=true` 时，Prefill 和 Decode 的该值都必须为 1。
- P/D `phase_result.json` 分别记录服务参数、benchmark 参数、性能指标、QPS 和 QPS 来源；两阶段 QPS 的单位均为 req/s，来源均为 `benchmark_throughput`。
- `pd_disagg_summary.json` 记录 P/D 结果、PD ratio、实例推荐和产物路径。
- 微调结果按普通 `standard` 模式的输出规则单独存储。

### 功能、性能与可靠性影响

默认 `standard` 模式仍执行原有通用寻优逻辑。Decode 配置 `benchmark_run_count=2` 后，每个候选增加一轮 benchmark，但服务仍只启动一次，因此新增耗时主要是一轮 benchmark 而非一次服务启动。Decode QPS 使用 prefix cache 预热后的 benchmark 实测请求吞吐，避免将 token/s 与 Prefill 的 req/s 直接比较。微调阶段不启动或监控 simulator，只承担 benchmark 基线和 FineTune 耗时。用户负责外部 PD 服务的参数一致性、可达性和生命周期。

影响范围包括 `optix/config/config.py`、`optix/optimizer/pd_disagg.py`、`optix/config.toml`、OptiX 用户指南、本详设和 `tests/regression/optix/`。

## 使用说明

第一步，新建 `pd_disagg_config.toml`。以下示例使用 vLLM 和 `vllm_benchmark`，模型、端口、设备数和搜索范围需按实际环境调整：

```toml
n_particles = 8
iters = 4
skip_pso = false
manage_simulator_lifecycle = true
use_request_rate_calibration = false

[vllm.command]
host = "127.0.0.1"
port = "8000"
model = "/path/to/model"
served_model_name = "model_name"
others = ""

[vllm_benchmark.command]
host = "127.0.0.1"
port = "8000"
model = "/path/to/model"
served_model_name = "model_name"
dataset_name = "random"
others = ""

[pd_disagg]
total_devices = 16
prefill_devices_per_instance = 4
decode_devices_per_instance = 2
top_k = 0 # 跳过 P/D 搜索阶段的 FineTune
use_full_device = true
phase_output_dir = "pd_disagg"

[pd_disagg.prefill]
engine = "vllm"
benchmark_policy = "vllm_benchmark"
benchmark_run_count = 1
n_particles = 8
iters = 4
ttft_penalty = 1
tpot_penalty = 0
ttft_slo = 2.0
fine_tune_mode = "pd_mixed"

[pd_disagg.prefill.simulator_command_overrides]
others = "--tensor-parallel-size 4 --no-enable-prefix-caching"

[pd_disagg.prefill.benchmark_command_overrides]
others = "--num-prompts 500 --random-input-len 1024 --random-output-len 1"

[[pd_disagg.prefill.target_field]]
name = "MAX_NUM_BATCHED_TOKENS"
config_position = "env"
min = 8192
max = 65536
dtype = "int"
value = 8192

[[pd_disagg.prefill.target_field]]
name = "MAX_NUM_SEQS"
config_position = "env"
min = 8
max = 128
dtype = "int"
value = 32

[[pd_disagg.prefill.target_field]]
name = "CONCURRENCY"
config_position = "env"
min = 1
max = 256
dtype = "int"
value = 32

[[pd_disagg.prefill.target_field]]
name = "REQUESTRATE"
config_position = "env"
min = 0
max = 0
dtype = "float"
value = 0

[pd_disagg.decode]
engine = "vllm"
benchmark_policy = "vllm_benchmark"
benchmark_run_count = 2
n_particles = 12
iters = 6
ttft_penalty = 0
tpot_penalty = 1
tpot_slo = 0.05
fine_tune_mode = "pd_disaggregation"

[pd_disagg.decode.simulator_command_overrides]
others = "--tensor-parallel-size 2 --enable-prefix-caching"

[pd_disagg.decode.benchmark_command_overrides]
others = "--seed 1024 --num-prompts 500 --random-input-len 1024 --random-output-len 256"

[[pd_disagg.decode.target_field]]
name = "MAX_NUM_BATCHED_TOKENS"
config_position = "env"
min = 128
max = 4096
dtype = "int"
value = 512

[[pd_disagg.decode.target_field]]
name = "MAX_NUM_SEQS"
config_position = "env"
min = 32
max = 512
dtype = "int"
value = 64

[[pd_disagg.decode.target_field]]
name = "CONCURRENCY"
config_position = "env"
min = 1
max = 1000
dtype = "int"
value = 100

[[pd_disagg.decode.target_field]]
name = "REQUESTRATE"
config_position = "env"
min = 0
max = 1000
dtype = "float"
value = 0
```

P/D 搜索配置 `use_request_rate_calibration=false` 时，`CONCURRENCY` 在范围内参与 PSO，`REQUESTRATE` 固定为最大施压速率。`REQUESTRATE=0` 表示不限速：范围包含 `0` 时固定为 `0`，否则固定为数值 `max`。`top_k=0` 只跳过本步的 P/D 阶段 FineTune，不影响 PSO、QPS 和配比推荐，也不影响用户拉起完整 PD 服务后单独执行的 benchmark-only 微调。

`benchmark_run_count` 表示单个候选在同一服务实例上执行 benchmark 的总次数，必须大于等于 1，默认值为 1。该配置只允许与 `use_request_rate_calibration=false` 组合；校准模式下配置大于 1 会在 `Settings` 跨字段校验时直接报错，不再运行时降为 1。Decode 推荐配置为 2：第一轮只填充 prefix cache，第二轮指标才写入 DataStorage 并参与 fitness、Decode QPS 和 PD 配比。两轮之间不会重启服务，但会停止并重新拉起 benchmark 进程、清理上一轮结果文件。若任意预热轮失败，则该候选失败，不能回退使用预热指标。

Decode 双跑成立的前提是服务端启用 prefix caching，且 benchmark 两轮生成完全一致的 workload。使用 `dataset_name="random"` 时应显式配置同一个 `--seed`；服务启动参数中的 `--seed` 不能替代 benchmark 的 seed。Prefill 仍可关闭 prefix caching 并保持单次 benchmark。该双跑只用于第一步独立 Decode 搜索，以近似隔离 D 节点能力；用户启动完整 PD 服务后的第二步 benchmark-only 微调直接测量真实 PD 链路，不受此双跑配置约束。

`top_k>0` 且 Decode 启用 `pd_disaggregation` 阶段 FineTune 时，会先临时使用 `REQUESTRATE=0` 进行不限流探测，并将实测 QPS 作为后续 FineTune 的初始请求速率。并发仍按 TPOT 是否超过 SLO 维护上下界并二分调整；并发可以继续移动时优先调整并发，并发到达边界或二分结果不再变化后，若 `ttft_penalty>0` 且 TTFT 偏离 SLO 区间，则按 `pd_mixed` 的比例步长规则继续调整请求速率。初始 QPS 后续不再被强制回填为固定值。Decode 必须配置 `min < max` 的 `REQUESTRATE` 范围，且范围应覆盖预期实测 QPS；`top_k=0` 时不做该校验。

P/D 配比计算中，Prefill 和 Decode 均直接使用各自 benchmark 返回的正数 `throughput`（req/s）作为 QPS。Decode 不再使用 `CONCURRENCY/TPOT_s`；若任一阶段缺少有效吞吐，配比计算失败并提示检查 benchmark 指标解析。

最终实例数仍由设备约束决定。例如 `total_devices=16`，且 P、D 单实例都使用 8 卡时，在至少各部署一个实例并要求用满设备的前提下，唯一可行推荐为 `1P:1D`；理论 `pd_ratio` 与实际比例的差异通过 `ratio_error` 展示。

执行 P/D 服务参数搜索：

```bash
msmodeling optix --mode pd_disagg -c ./pd_disagg_config.toml
```

成功后 `pd_disagg_summary.json` 的状态为 `service_search_completed`。用户根据 P/D `phase_result.json` 的 `service_params` 和 summary 的 `instances` 启动 PD 服务。

第二步，新建独立的 `pd_fine_tune.toml`：

```toml
skip_pso = true
manage_simulator_lifecycle = false
fine_tune_mode = "pd_mixed"

[vllm]
target_field = []

[vllm_benchmark.command]
host = "pd-service-host"
port = "8000"
model = "model_path"
served_model_name = "model_name"
dataset_name = "random"
others = "--num-prompts 500 --random-input-len 1024 --random-output-len 256"

[[vllm_benchmark.target_field]]
name = "CONCURRENCY"
config_position = "env"
min = 1
max = 1000
dtype = "int"
value = 100

[[vllm_benchmark.target_field]]
name = "REQUESTRATE"
config_position = "env"
min = 1
max = 1000
dtype = "float"
value = 100
```

执行：

```bash
msmodeling optix --mode standard -e vllm -b vllm_benchmark -c ./pd_fine_tune.toml
```

关键配置：

| 配置 | 微调取值 | 说明 |
| -- | -- | -- |
| `skip_pso` | `true` | 基线完成后直接进入 FineTune |
| `manage_simulator_lifecycle` | `false` | 不启动、监控或停止 simulator |
| `fine_tune_mode` | `pd_mixed` | 根据 TPOT 调并发、根据 TTFT 调速率；速率固定且 TTFT 超标时降低并发 |
| `vllm.target_field` | `[]` | 排除服务侧搜索字段 |
| `vllm_benchmark.target_field` | 并发、请求速率字段 | 指定 FineTune 可调整的 benchmark 参数 |

`manage_simulator_lifecycle=false` 必须与 `skip_pso=true` 同时配置，否则配置校验失败。benchmark URL 必须从 OptiX 主机访问到用户启动的 PD 服务。不需要安装或注册 `vllm_pd` 多机 simulator 插件。

`REQUESTRATE` 的 `constant` 非空或 `min=max` 时视为固定速率。固定速率不会因 TTFT 调整而改变；当 TTFT 超过 SLO 且并发可调时，FineTune 按 TTFT 超标比例降低 `CONCURRENCY`。PD 分离模式的请求速率后续调节需要 `ttft_penalty>0`；为 0 时仅使用不限速探测得到的初始 QPS，并保持原二分并发逻辑。

## 测试设计

| 用例名 | 测试类型 | 前置条件 | 操作方式 | 预期结果 |
| -- | -- | -- | -- | -- |
| UT-PD 配置精简 | 单元测试 | 无 | 构造 `PdDisaggConfig` 并传入已移除字段 | 只包含 P/D 搜索和配比字段，已移除字段明确报错 |
| UT-P/D 阶段 FineTune 可选 | 边界测试 | `top_k=0` | 执行 P/D 搜索 | 两阶段均跳过 FineTune，保留 PSO 最优结果并正常输出配比和 summary |
| UT-Decode 请求速率范围 | 边界测试 | `fine_tune_mode=pd_disaggregation` | 使用固定或不包含实测 QPS 的范围 | 在运行微调前或固定速率时明确报错 |
| UT-固定速率 TTFT 回退 | 单元测试 | `fine_tune_mode=pd_mixed`、请求速率固定 | 构造 TPOT 达标且 TTFT 超标的指标 | 请求速率不变，并发降低 |
| UT-PD 分离速率续调 | 单元测试 | `fine_tune_mode=pd_disaggregation`、`ttft_penalty>0` | 先用实测 QPS 初始化，再令二分并发收敛且 TTFT 超标 | 首轮使用实测 QPS，后续降低请求速率，并发仍按二分逻辑调整 |
| UT-最大施压速率 | 边界测试 | `use_request_rate_calibration=false` | 分别使用包含和不包含 0 的 `REQUESTRATE` 范围 | 包含 0 时固定为 0，否则固定为数值 max |
| UT-benchmark 次数配置 | 边界测试 | 构造 P/D 阶段配置 | 省略、配置 1、2、0 | 默认值为 1；1、2 可用；0 校验失败 |
| UT-Decode cache 预热 | 单元测试 | `use_request_rate_calibration=false`、Decode `benchmark_run_count=2` | 两轮返回不同吞吐 | 服务只启动一次、benchmark 执行两次，仅第二轮进入结果和 QPS |
| UT-校准模式配置冲突 | 异常测试 | `use_request_rate_calibration=true`、任一阶段 `benchmark_run_count>1` | 加载配置 | 直接校验失败并指出非法阶段，不进入候选评测 |
| UT-Decode 预热失败 | 异常测试 | 第一轮 benchmark 失败 | 执行候选评测 | 候选失败，不执行第二轮，也不保存预热指标 |
| UT-Prefill TPOT=0 | 边界测试 | `tpot_penalty=0` | 筛选候选 | Prefill 保留 TPOT=0，Decode 仍过滤 |
| UT-P/D QPS | 单元测试 | 构造 P/D 指标 | 计算 QPS 和 ratio | P/D 均使用 benchmark throughput，单位均为 req/s |
| UT-资源受限配比 | 边界测试 | 总计 16 卡，P/D 单实例均为 8 卡，理论配比不为 1 | 枚举实例分配 | 仍只推荐 `1P:1D`，并正确记录 `ratio_error` |
| UT-服务搜索停止边界 | 单元测试 | mock phase runner | 执行 `pd_disagg` | 只运行 P/D，输出配比后结束 |
| UT-运行模式兼容 | 单元测试 | 命令省略 `--mode` | 执行寻优入口 | 默认进入 `standard`，执行原有通用流程，不进入 P/D 编排器 |
| UT-PD 跳过 PSO 冲突 | 异常测试 | 最终解析模式为 `pd_disagg` | 配置 `skip_pso=true` 后启动 | 创建阶段 runner 前直接报错；`standard` 模式仍允许 `skip_pso=true` |
| UT-PD 生命周期冲突 | 异常测试 | 最终解析模式为 `pd_disagg` | 配置 `manage_simulator_lifecycle=false` 后启动 | 创建阶段 runner 前直接报错；`standard` benchmark-only 微调仍允许为 `false` |
| IT-P/D 搜索成功 | 集成测试 | P/D mock 成功 | 执行编排器 | 生成 P/D 产物、配比 CSV 和完成 summary |
| IT-P/D 搜索失败 | 异常测试 | Decode mock 失败 | 执行编排器 | 保留 Prefill 产物并记录失败阶段 |
| IT-benchmark-only 微调 | 集成测试 | mock benchmark 和已运行服务 | 执行普通微调配置 | 跳过 PSO，不运行 simulator，只调整 benchmark 字段 |
| IT-benchmark 失败 | 异常测试 | 服务不可达 | 执行普通微调配置 | 返回评测失败，不尝试启停服务 |
| E2E-两步 mock | 端到端测试 | mock P/D simulator 和 benchmark | 先执行 `pd_disagg`，再执行 `standard` | 分别输出配比推荐和最优负载参数 |
| E2E-真实外部服务 | 端到端测试 | NPU 环境和用户已启动 PD 服务 | 运行两步命令 | 无多机 simulator 插件时完成微调，服务不被 OptiX 停止 |
| 兼容性回归 | 兼容性测试 | 现有 OptiX 配置 | 运行 OptiX regression | 默认服务参数寻优行为不变 |

真实外部服务 E2E 必须在具备 NPU 和 PD 部署的环境执行，本地 mock 测试不能代替该验收。
