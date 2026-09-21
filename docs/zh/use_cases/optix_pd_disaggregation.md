# 使用案例：搜索 PD 分离服务参数与实例配比

部署 Prefill-Decode（PD）分离服务时，需要确定两件事：每个 Prefill（P）和 Decode（D）实例使用什么参数，以及各部署多少个实例。本文介绍如何用 OptiX 分别测量 P、D 的处理能力，得到服务参数和实例配比，再对部署好的完整服务调整并发和请求速率。

整个操作分为三步：

1. 运行 `pd_disagg`，由 OptiX 依次启动 vLLM，搜索 P、D 单实例的服务参数并计算配比。
2. 按搜索结果，用现有部署工具启动 P、D 实例和 Proxy，配置请求转发与 KV cache 传输。
3. 运行 `standard` 微调模式，通过 Proxy 压测完整 PD 服务，调整负载参数。

如果完整 PD 服务已经部署好，只需要调整压测并发和请求速率，可以直接从[第三步](#第三步微调完整-pd-服务)开始。

## 准备环境和配置文件

先按[环境准备与安装](../user_guide/msmodeling_optix_user_guide.md#环境准备与安装)安装 OptiX，并确认运行环境中可以使用 `vllm` 和所选压测工具。

本文以 `vllm_benchmark` 为例，使用 16 卡、P 单实例 4 卡、D 单实例 2 卡，输入 1024 token、输出 256 token、每次测评 500 个请求。这些数值用于说明配置方法，实际使用时请替换为自己的硬件、负载和时延目标。配置模板及其他通用字段见 [optix/config.toml](../../../optix/config.toml) 和[配置文件说明](../user_guide/msmodeling_optix_user_guide.md#配置文件说明)。

## 第一步：搜索 P/D 服务参数

### 1. 设置公共服务信息和运行方式

编辑 `pd_disagg_config.toml`：

```toml
optimizer_strategy = "pso"
output = "./result"
skip_pso = false
manage_simulator_lifecycle = true
use_request_rate_calibration = false

[vllm.command]
host = "127.0.0.1"
port = "8000"
model = "/path/to/model"
served_model_name = "pd-model"
others = ""

[vllm_benchmark.command]
dataset_name = "random"
```

将 `/path/to/model` 替换为权重目录。这里假设 OptiX 和用于搜索的 vLLM 在同一台机器上；端口需要保证未被占用。

| 参数 | 本例设置的含义 |
|---|---|
| `skip_pso = false` | 执行 PSO 服务参数搜索。`pd_disagg` 模式要求此值为 `false`。 |
| `manage_simulator_lifecycle = true` | 由 OptiX 启停用于搜索的 vLLM 服务。`pd_disagg` 模式要求此值为 `true`。 |
| `use_request_rate_calibration = false` | 不追加请求率校准轮次；搜索时并发可变，请求速率使用最大值：范围包含 `0` 时取 `0`（不限速），否则取 `max`。也用于启用后面的 Decode 预热测评。 |

### 2. 设置设备数和两个阶段的测评参数

在同一文件中添加或修改：

```toml
[pd_disagg]
total_devices = 16
prefill_devices_per_instance = 4
decode_devices_per_instance = 2
top_k = 3
use_full_device = true
phase_output_dir = "pd_disagg"

[pd_disagg.prefill]
engine = "vllm"
benchmark_policy = "vllm_benchmark"
n_particles = 8
iters = 4
benchmark_run_count = 1
ttft_slo = 2.0
ttft_penalty = 1
tpot_penalty = 0
fine_tune_mode = "pd_mixed"

[pd_disagg.prefill.simulator_command_overrides]
others = "--tensor-parallel-size 4 --no-enable-prefix-caching"

[pd_disagg.prefill.benchmark_command_overrides]
others = "--seed 1024 --num-prompts 500 --random-input-len 1024 --random-output-len 1"

[pd_disagg.decode]
engine = "vllm"
benchmark_policy = "vllm_benchmark"
n_particles = 12
iters = 6
benchmark_run_count = 2
tpot_slo = 0.05
ttft_penalty = 0
tpot_penalty = 1
fine_tune_mode = "pd_disaggregation"

[pd_disagg.decode.simulator_command_overrides]
others = "--tensor-parallel-size 2 --enable-prefix-caching"

[pd_disagg.decode.benchmark_command_overrides]
others = "--seed 1024 --num-prompts 500 --random-input-len 1024 --random-output-len 256"
```

P 阶段把输出长度设为 `1`，使请求在首 token 后结束，主要测量 Prefill 的处理能力。D 阶段启用 prefix caching，并对同一组候选连续测评两次：第一次填充缓存，第二次记录指标。两次测评共用一个 vLLM 实例，第一轮预热结果不计入搜索成绩或配比。

Decode 的两轮请求需要一致，所以示例在 **benchmark 命令**中设置了 `--seed 1024`。输入输出长度、请求数、并发和请求率也要保持一致。仅在 `vllm serve` 中设置 seed 无法固定压测请求。缓存预热可以减少 Prefill 对独立 D 测评的影响，完整 PD 链路的性能仍需在第三步验证。

| 参数 | 含义与修改方法 |
|---|---|
| `total_devices` | 最终部署可使用的总卡数，用于实例配比计算，不负责分配具体卡号。 |
| `prefill_devices_per_instance` / `decode_devices_per_instance` | 每个 P/D 实例使用的卡数。本例分别与 `--tensor-parallel-size 4/2` 对应；总卡数和这两项应同时设置。三项均为 `0` 时只计算理论配比。 |
| `use_full_device` | `true` 时只推荐恰好用满总卡数的组合；`false` 时允许剩余设备。 |
| `top_k` | 控制阶段候选保留与微调数量，默认 `3`。设为 `0` 可跳过 P/D 阶段内的微调，仍会验证 PSO 最优结果并计算配比；卡数约束允许时仍有整数实例推荐。 |
| `n_particles` / `iters` | PSO 粒子数和迭代数。增加通常会带来更多候选测评；此外还有基线、验证和微调的开销。P/D 可分别设置，缺省项继承顶层同名值。 |
| `benchmark_run_count` | 每组候选在同一服务实例上的测评次数，默认 `1`。大于 `1` 时前 N−1 次预热、最后一次计分，且要求 `use_request_rate_calibration=false`。 |
| `ttft_slo` / `tpot_slo` | 当前阶段的时延目标。本例 P 关注 TTFT，D 关注 TPOT；`0.05` 秒即 `50` 毫秒。 |
| `ttft_penalty` / `tpot_penalty` | 时延对评分的影响；设为 `0` 时该项不影响候选优劣。示例中 P 不按 TPOT 评分，D 不按 TTFT 评分。 |
| `simulator_command_overrides` | 修改当前阶段的 vLLM 启动配置，如 `port`、`model`、`others`，不影响另一个阶段。 |
| `benchmark_command_overrides` | 修改当前阶段的压测命令配置，如示例中的输入输出长度。 |
| `phase_output_dir` | P/D 结果目录名，位于 `output` 下，默认 `pd_disagg`。 |

`others` 的阶段覆盖会替换整个字符串。如果公共配置中还有量化等必要启动参数，应在两个阶段的 `others` 中一并保留。

### 3. 配置服务参数和负载的搜索范围

继续在 `pd_disagg_config.toml` 中添加下列搜索字段。P 和 D 的 `target_field` 分别是一份完整列表：某阶段配置了非空列表后，该阶段就使用这份列表，不再合并公共插件的搜索字段。因此服务参数、并发和请求率都需要列出。

```toml
[[pd_disagg.prefill.target_field]]
name = "MAX_NUM_BATCHED_TOKENS"
config_position = "run"
min = 8192
max = 65536
dtype = "int"
value = 8192

[[pd_disagg.prefill.target_field]]
name = "MAX_NUM_SEQS"
config_position = "run"
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
config_position = "run"
min = 128
max = 4096
dtype = "int"
value = 512

[[pd_disagg.decode.target_field]]
name = "MAX_NUM_SEQS"
config_position = "run"
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

每个字段的 `value` 是基线使用的初始值，`min`、`max` 定义可调范围，`dtype` 定义数值类型。`config_position="run"` 将服务参数写入 vLLM 启动命令，`config_position="env"` 用于这里的并发和请求率，由 benchmark 消费。若另外在 `others` 中使用 `$MAX_MODEL_LEN` 等占位符，也要在相应阶段的 `target_field` 中声明该字段。

本例 P 的请求率固定为不限速。D 的请求率保留可调范围，是因为 `top_k>0` 时，D 阶段微调会先不限速测量 QPS，再以该 QPS 为请求率调整并发。D 的 `REQUESTRATE` 范围需要覆盖预期 QPS；如果实测值超过 `1000`，应扩大上限。`top_k=0` 时不执行这一步微调，也不检查该范围。

### 使用 AISBench 时怎么改

将两个阶段的 `benchmark_policy` 改为 `"ais_bench"`，再把上述 `benchmark_command_overrides` 改为 AISBench 的配置。例如 P 阶段：

```toml
[pd_disagg.prefill.benchmark_command_overrides]
models = "your_prefill_model_config"
mode = "perf"
others = "--datasets your_prefill_dataset_config"
```

D 阶段同样在 `[pd_disagg.decode.benchmark_command_overrides]` 中指定对应配置。`models` 和 `datasets` 填写 AISBench 能解析的配置短名；`datasets` 通过 `others` 传入。AISBench 的模型 Python 配置还需填写正确的服务地址、端口和服务模型名。如果使用 `ais_bench` 并需要开启稳态测试，可在 `[ais_bench.command]` 的 `others` 中追加 `--summarizer stable_stage`，例如 `others = "--datasets your_dataset_config --summarizer stable_stage"`。

P 的输出长度设为 1 token；D 的模型与数据集配置应保证预热和正式测评生成相同请求。AISBench 的配置方式与 `vllm_benchmark` 不同，不要直接沿用 `--random-input-len` 等参数。可覆盖的 OptiX 命令字段为 `models`、`mode`、`work_dir` 和 `others`。

先在运行 AISBench 的环境中检查短名能否解析，再启动搜索。下列短名需替换为自己的配置，P/D 分别检查一次：

```bash
ais_bench --models your_prefill_model_config --datasets your_prefill_dataset_config --search
```

### 4. 启动搜索并查看结果

保存配置后执行：

```bash
msmodeling optix --mode pd_disagg -c ./pd_disagg_config.toml
```

OptiX 先完成 Prefill 搜索，再完成 Decode 搜索，最后计算实例配比。它直接读取两阶段 benchmark 的请求吞吐量作为 QPS，单位是 req/s；若某阶段没有返回大于 0 的有效吞吐量，需先排查该阶段的压测结果。

本例结果保存在 `result/pd_disagg/<run_id>/`，`<run_id>` 为本次运行标识。重点查看以下文件：

| 文件 | 查看内容 |
|---|---|
| `pd_disagg_summary.json` | 运行状态、P/D QPS、理论配比 `pd_ratio`、推荐实例数 `instances`，以及产物路径。成功时 `status` 为 `service_search_completed`。 |
| `prefill/phase_result.json`、`decode/phase_result.json` | 两个阶段的测评指标，以及用于部署的 `service_params` 和用于压测的 `benchmark_params`。 |
| `pd_ratio_candidates.csv` | 不同 P/D 候选组合的 QPS、配比和可行的实例分配，便于比较其他组合。 |
| P/D 阶段 CSV | 各次候选的参数和性能数据；具体路径可从 summary 的 `artifacts` 字段读取。 |

可以直接格式化查看 summary：

```bash
python -m json.tool result/pd_disagg/<run_id>/pd_disagg_summary.json
```

`pd_ratio` 表示 **P 实例数 / D 实例数**，计算式为：

```text
pd_ratio = 单个 D 实例的 QPS / 单个 P 实例的 QPS
```

例如单个 P 实例每秒处理 20 个请求，单个 D 实例每秒处理 10 个请求，则 `pd_ratio=0.5`，即理论上每个 P 实例搭配两个 D 实例。本例 P 每实例 4 卡、D 每实例 2 卡，16 卡可部署 `2P:4D`，两侧处理能力均为 40 req/s。这里的数字仅用于演示计算。

实际部署看 `instances.prefill` 和 `instances.decode`。推荐组合先满足卡数约束，再优先接近理论配比；`ratio_error` 是实际 P/D 比值与理论值之差的绝对值。如果有总卡数配置但 `instances` 为空，查看 `allocation_warning`，检查单实例卡数能否组合成总卡数，或是否允许关闭 `use_full_device`。

## 第二步：部署 P、D 实例和 Proxy

取得搜索结果后，按以下顺序部署：

1. 从两个 `phase_result.json` 读取 `service_params`，将服务参数应用到相应的 P、D 启动命令，同时保留阶段配置中的 TP、量化等固定参数。
2. 按 `instances` 中的数量启动 P、D 实例，并用现有部署工具配置 Proxy、请求路由和 KV cache 传输。
3. 从运行 OptiX 的机器访问 Proxy，确认请求可以正常完成，再进行下一步压测。

例如 Proxy 暴露了 OpenAI 兼容接口，可先查看服务模型名（替换地址和端口；启用鉴权时带上相应请求头）：

```bash
curl http://<proxy-host>:<proxy-port>/v1/models
```

第一步结果中的 `benchmark_params` 记录了单阶段测试所用的负载。完整服务经过 Proxy 和 KV cache 传输后，承载能力可能不同，接下来需要重新测量并发和请求率。

## 第三步：微调完整 PD 服务

### 1. 填写 Proxy 地址和微调范围

编辑 `pd_fine_tune.toml`。设置以下顶层字段，并替换对应的 vLLM 和 benchmark 配置：

```toml
optimizer_strategy = "pso"
output = "./result/pd_fine_tune"
skip_pso = true
manage_simulator_lifecycle = false
use_request_rate_calibration = false
fine_tune_mode = "pd_mixed"
max_fine_tune = 10
ttft_slo = 2.0
tpot_slo = 0.05

[vllm]
target_field = []

[vllm.command]
host = "pd-service-host"
port = "8000"
model = "/path/to/model"
served_model_name = "pd-model"
others = ""

[vllm_benchmark.command]
dataset_name = "random"
others = "--seed 1024 --num-prompts 500 --random-input-len 1024 --random-output-len 256"

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

删除模板中已有的 `[[vllm.target_field]]` 块，保留上面的 `target_field=[]`，只在 benchmark 下定义负载参数。`pd-service-host:8000` 替换为 OptiX 能访问的 Proxy 地址，`served_model_name` 与 Proxy 对外提供的模型名一致。

使用 `vllm_benchmark` 时，地址、端口、模型路径和服务模型名需要填写在 `[vllm.command]`。加载配置时，这四项会复制到 `[vllm_benchmark.command]`，因此仅修改 benchmark 下的同名字段会被覆盖。

| 参数 | 本步的作用 |
|---|---|
| `skip_pso = true` | 跳过 PSO，完成基线测评后直接微调；`optimizer_strategy` 保持为 `"pso"` 即可。 |
| `manage_simulator_lifecycle = false` | 使用已运行的 PD 服务，OptiX 只执行压测，不启动、监控或停止该服务。 |
| `fine_tune_mode = "pd_mixed"` | 对完整服务按 TPOT 调整并发、按 TTFT 调整请求率。请求率固定且 TTFT 超标时，会尝试降低并发。 |
| `max_fine_tune` | 微调最多尝试的次数。本例为 `10`；先检查压测流程时可减小，算法也可能提前结束。 |
| `ttft_slo` / `tpot_slo` | 完整服务的时延目标，应按业务要求填写。 |
| `CONCURRENCY.value` / `REQUESTRATE.value` | 本例从并发 100、请求率 100 req/s 开始测评，再在各自的 `min`～`max` 范围内调整。 |

如只想调整并发，可将 `REQUESTRATE` 的 `min`、`max`、`value` 都设为相同值；都设为 `0` 时表示固定不限速。

### 2. 运行微调

确认 Proxy 已就绪，执行：

```bash
msmodeling optix --mode standard -e vllm -b vllm_benchmark -c ./pd_fine_tune.toml
```

这一阶段先测初始负载，再尝试调整并发和请求率，结果单独写入 `result/pd_fine_tune/`。查看结果 CSV 时，结合 TTFT、TPOT、成功率和吞吐量判断是否满足业务目标，字段含义见[结果文件说明](../user_guide/msmodeling_optix_user_guide.md#结果文件说明)。本流程无需安装 `contrib/optix/vllm_pd_simulator`。

## 常见问题

| 现象 | 检查方法 |
|---|---|
| `pd_disagg` 启动时报生命周期或 PSO 配置错误 | 第一阶段应设置 `skip_pso=false`、`manage_simulator_lifecycle=true`；这与第三步的设置相反。 |
| Decode 配置两次测评后校验失败 | 检查顶层 `use_request_rate_calibration` 是否为 `false`。 |
| 搜索结束但没有整数实例推荐 | 检查三个卡数字段和 `allocation_warning`；总卡数不一定能由 P/D 单实例卡数组合得到。 |
| 修改了压测地址仍连接旧端口 | 使用 `vllm_benchmark` 时改 `[vllm.command]`；使用 AISBench 时检查其模型 Python 配置。 |
| AISBench 找不到模型或数据集配置 | 先用对应短名执行 `ais_bench --models ... --datasets ... --search`，确认能唯一解析。 |

更多日志说明与排查方法见[主指南附录](../user_guide/msmodeling_optix_user_guide.md#附录)。
