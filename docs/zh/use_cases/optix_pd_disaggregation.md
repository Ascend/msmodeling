# 使用案例：OptiX PD 分离两步寻优

## 场景与目标

本文适用于使用 vLLM 部署 Prefill-Decode（PD）分离服务、需要搜索 P/D 服务参数并推荐实例配比的场景。

PD 分离模式拆分为“P/D 服务参数搜索与配比推荐”和“外部 PD 服务 benchmark-only 微调”两步。
该流程仅用于需要推荐 P/D 实例配比的场景；其他 PD 分离或 PD 混部寻优直接使用仓内现有插件。

流程如下：先分别搜索 Prefill 和 Decode 服务参数并获取配比推荐，再使用自有部署工具启动完整 PD 服务，最后对该服务的并发和请求速率进行微调。

## 前提条件与配置准备

1. 按[主指南的环境准备与安装](../user_guide/msmodeling_optix_user_guide.md#环境准备与安装)安装 OptiX，并准备可用的 vLLM 和测评工具。
2. 准备模型路径、服务地址、总卡数、P/D 单实例用卡数和业务 TTFT/TPOT 目标。
3. 从仓库的 [optix/config.toml](../../../optix/config.toml) 复制配置为 `pd_disagg_config.toml`，按实际环境修改公共服务和测评参数，并参考其中注释的 `[pd_disagg]` 示例配置两个阶段。通用字段见[配置文件说明](../user_guide/msmodeling_optix_user_guide.md#配置文件说明)。

以下 TOML 代码块展示需修改的配置片段，应合并到对应配置文件中；同名字段或表应修改已有定义，避免重复添加。第一步使用 `pd_disagg_config.toml`，第二步使用独立的 `pd_fine_tune.toml`。

## 第一步：搜索 P/D 服务参数并推荐配比

### 配置阶段测评方式

在 `pd_disagg_config.toml` 的顶层设置：

```toml
skip_pso = false
manage_simulator_lifecycle = true
use_request_rate_calibration = false
```

Prefill 阶段必须把 benchmark 的输出长度配置为 1 token，使请求在首 token 后结束，尽量消除 Decode 过程对结果的影响。
OptiX 直接把 P/D 两阶段 benchmark 各自返回的 `throughput`（单位 req/s）作为 QPS，不使用并发数和 TTFT/TPOT 计算或兜底。任一阶段未返回有效正数 `throughput` 时，配比计算失败。

独立 Decode 搜索应启用 vLLM prefix caching，并在 `use_request_rate_calibration=false` 时设置 `decode.benchmark_run_count=2`。每组候选只启动一次 Decode 服务：第一轮相同请求用于填充 prefix cache，第二轮才采集有效指标；第一轮不会进入 CSV、fitness、Decode QPS 或配比计算。random 数据集应显式固定 benchmark `--seed`，确保两轮 prompt 和请求顺序一致。该双跑只用于第一步近似隔离 D 节点能力，不影响用户启动完整 PD 服务后的第二步 benchmark-only 微调。

### 配置卡数、搜索预算与搜索空间

下表字段均位于 `[pd_disagg]` 下，`phase` 表示 `prefill` 或 `decode`。

关键配置如下：

| 配置 | 默认值 | 说明 |
|---|---|---|
| `total_devices` | `0` | 总设备数；为 0 时只输出浮点配比。|
| `prefill_devices_per_instance` / `decode_devices_per_instance` | `0` | 单实例设备数；设备相关三个字段必须同时配置。|
| `top_k` | `3` | P/D 阶段候选数量；`0` 表示跳过阶段 FineTune，仍输出 PSO 最优点对应的配比和至少一组实例推荐。|
| `use_full_device` | `true` | 是否要求实例组合用满设备。|
| `phase_output_dir` | `pd_disagg` | 相对 `output` 的阶段产物目录。|
| `prefill.engine` / `decode.engine` | `vllm` | 内置 PD 流程限定值。|
| `prefill.benchmark_policy` / `decode.benchmark_policy` | `vllm_benchmark` | 可换为 `ais_bench` 或其他已注册 benchmark。|
| `prefill.n_particles` / `prefill.iters` | 顶层同名值 | Prefill 独立 PSO 粒子数和迭代数。|
| `decode.n_particles` / `decode.iters` | 顶层同名值 | Decode 独立 PSO 粒子数和迭代数。|
| `prefill.benchmark_run_count` / `decode.benchmark_run_count` | `1` | 同一服务实例上的 benchmark 总次数，必须大于等于 1；大于 1 时要求 `use_request_rate_calibration=false`，否则配置校验失败。前 N-1 轮只预热。|
| `phase.simulator_command_overrides` | 空 | 可选，只覆盖该阶段 vLLM 启动命令配置中已声明的字段。|
| `phase.benchmark_command_overrides` | 空 | 可选，只覆盖该阶段 benchmark 命令配置中已声明的字段。|
| `phase.target_field` | 空 | 该阶段完整搜索空间；非空时替代从插件继承的字段。|

Prefill 和 Decode 可分别设置搜索预算；任一字段未填写时仅该字段回退到顶层配置：

```toml
[pd_disagg.prefill]
n_particles = 8
iters = 4
benchmark_run_count = 1

[pd_disagg.decode]
n_particles = 12
iters = 6
benchmark_run_count = 2
```

`[vllm.command]` 是 Prefill/Decode 的公共基线；两个阶段可分别使用
`simulator_command_overrides` 覆盖 `host`、`port`、`model`、`served_model_name` 或 `others`。
覆盖使用阶段本地副本，不会回写全局配置，也不会污染另一个阶段。例如：

```toml
[pd_disagg.prefill.simulator_command_overrides]
others = "--tensor-parallel-size 4 --no-enable-prefix-caching"

[pd_disagg.decode.simulator_command_overrides]
others = "--tensor-parallel-size 2 --enable-prefix-caching"
```

若 `others` 使用 `$MAX_MODEL_LEN` 等搜索参数占位符，必须在对应阶段自己的
`[[pd_disagg.<phase>.target_field]]` 中声明。
P/D 的 `target_field` 是两份相互独立的完整搜索空间；只要任一阶段显式配置了它，就不会再合并
`[[vllm.target_field]]`。因此服务参数和 `CONCURRENCY`、`REQUESTRATE` 等负载参数都要在该阶段列全：

```toml
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
min = 1
max = 1000
dtype = "float"
value = 100
```

仅当 `top_k>0` 启用 Decode 阶段 FineTune 时，Decode 才会先临时使用不限流速率探测 QPS，再固定该速率调整并发，因此 `REQUESTRATE` 必须配置为覆盖预期 QPS 的可调范围。`top_k=0` 时跳过该 FineTune 和相应范围校验。

同一个 benchmark 也可以在 P 和 D 阶段使用不同命令参数。`vllm_benchmark` 使用 random 数据集时可按下例配置：

```toml
[pd_disagg.prefill.benchmark_command_overrides]
others = "--num-prompts 500 --random-input-len 1024 --random-output-len 1"

[pd_disagg.decode.benchmark_command_overrides]
others = "--seed 1024 --num-prompts 500 --random-input-len 1024 --random-output-len 256"
```

Decode 双跑时，两轮必须保持相同的 benchmark 参数、请求数、输入输出长度、并发和请求速率；服务启动命令中的 `--seed` 不能替代上述 benchmark `--seed`。若 `use_request_rate_calibration=true`，Prefill 和 Decode 的 `benchmark_run_count` 都必须保持为 1，配置大于 1 会直接校验失败。

如果选用 `ais_bench`，仍使用同一覆盖机制，配置键必须是 `[ais_bench.command]`
已定义的 `models`、`mode`、`work_dir` 或 `others`；核心流程不会判断 benchmark 名称。

### 执行搜索与查看配比

第一步执行：

```bash
msmodeling optix --mode pd_disagg -c ./pd_disagg_config.toml
```

第一步只使用 vLLM 搜索 Prefill/Decode 服务化参数，输出 `service_search_completed` summary、P/D QPS、`pd_ratio` 和整数实例推荐。该模式必须配置 `skip_pso=false` 和 `manage_simulator_lifecycle=true`；否则启动时直接报错，不会使用未经搜索的 baseline 计算配比，也不会静默忽略生命周期配置。配置 `pd_disagg.top_k=0` 可跳过 P/D 搜索阶段内的 FineTune，仍保留 PSO 最优结果和配比推荐。用户随后使用自有部署工具启动 PD 服务。

浮点 `pd_ratio` 表示未考虑卡数的理论 `P_instances / D_instances`；最终 `instances` 必须满足总卡数和单实例用卡约束。例如总计 16 卡、P/D 单实例均为 8 卡且要求用满设备时，唯一可行推荐是 `1P:1D`；其与理论配比的差异记录在 `ratio_error`。

输出目录为 `result/pd_disagg/<run_id>/`。PD 模式写入 P/D CSV、`phase_result.json`、
`pd_ratio_candidates.csv` 和 `service_search_completed` summary。第二步按普通 `standard` 模式的输出规则单独存储，CSV 字段见[结果文件说明](../user_guide/msmodeling_optix_user_guide.md#结果文件说明)。

## 部署完整 PD 服务

第一步的阶段 JSON 分别记录 `service_params` 和 `benchmark_params`。用户必须将 `service_params` 和推荐卡数应用到实际 PD 部署，再将 benchmark 插件的服务 URL 指向当前 OptiX 主机能够访问的 PD Proxy 地址。具体 URL 字段由所选 benchmark 插件定义。

## 第二步：微调已部署 PD 服务的负载

第二步新建独立的普通 OptiX 微调配置。使用 `vllm_benchmark` 时，在 `[vllm.command]` 中填写已运行 PD 服务的 Proxy 地址和模型信息；测评数据集等参数配置在 `[vllm_benchmark.command]` 中。

配置加载后，`vllm_benchmark.command` 的 `host`、`port`、`model` 和 `served_model_name` 会被 `[vllm.command]` 中的同名字段覆盖，即使设置了 `manage_simulator_lifecycle=false` 也如此。因此这四项应在 `[vllm.command]` 中配置，只修改 benchmark 中的同名字段不会生效。此处的 `[vllm.command]` 用于提供测评目标信息，不会启动或停止 PD 服务。

```toml
skip_pso = true
manage_simulator_lifecycle = false
fine_tune_mode = "pd_mixed"

[vllm]
target_field = []

[vllm.command]
host = "pd-service-host"
port = "8000"
model = "model_path"
served_model_name = "model_name"

[vllm_benchmark.command]
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

然后执行：

```bash
msmodeling optix --mode standard -e vllm -b vllm_benchmark -c ./pd_fine_tune.toml
```

该步在 benchmark 基线测试后直接 FineTune，只调整并发和请求速率，不启动、监控或停止 simulator。无需安装 `contrib/optix/vllm_pd_simulator`。

通用日志与故障处理见[主指南附录](../user_guide/msmodeling_optix_user_guide.md#附录)。
