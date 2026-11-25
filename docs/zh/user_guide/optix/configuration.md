# OptiX 配置参数参考

本文档介绍OptiX的config.toml配置文件，涵盖寻优参数、测评工具参数、服务化参数与高级配置各配置段的参数取值和说明，适用于使用OptiX进行模型服务化性能寻优的配置人员。

[使用指南](../msmodeling_optix_user_guide.md) | [插件与模式](plugin_and_modes.md) | [配置参数](configuration.md) | [命令与结果](commands_and_results.md) | [环境与排障](troubleshooting.md)

<a id="7-配置文件说明"></a>

## 1. 配置文件说明

寻优配置文件默认位于`./msmodeling/optix/config.toml`。文件按 TOML 段组织，各段对应一项功能：

| TOML 段                                               | 用途                      | 文档章节 |
|------------------------------------------------------|-------------------------|---|
| 顶层                                                   | 寻优所需参数                  | [寻优参数](#寻优参数) |
| 顶层 `optimizer_strategy` | 寻优模式选择 | [寻优模式：PSO 与 Agent](plugin_and_modes.md#6-寻优模式pso-与-agent) |
| `[vllm]` / `[mindie]` | vLLM / MindIE 服务化参数与寻优字段 | [服务化参数](#服务化参数) |
| `[ais_bench.command]`  / `[vllm_benchmark.command]`| 测评工具参数                  | [测评工具参数](#测评工具参数) |
| `[agent_optimizer]`                                 | Agent 模式运行控制（可选）       | [寻优模式 → Agent 模式工作方式](plugin_and_modes.md#agent-模式工作方式) |
| `[deploy]`                                           | 部署环境根目录（可选）             | [高级配置 → 部署环境](#部署环境) |
| `[data_storage]`                                     | 结果存储与精调（可选）             | [高级配置 → 结果存储与精调](#结果存储与精调) |
| `[benchmark_early_exit]`                             | vLLM benchmark 早停策略（可选）    | [高级配置 → Benchmark 早停](#benchmark-早停) |
| `[health_check]`                                     | 运行时日志异常检测（可选）           | [高级配置 → 日志检测](#日志检测) |

<a id="寻优参数"></a>
<a id="11-寻优参数"></a>

## 2. 寻优参数

> **注意**：以下参数建议在 `config.toml` 中显式填写。`n_particles`、`iters`、各惩罚系数、`ttft_slo`、`tpot_slo`、`service` 等字段均有代码默认值，但首次运行建议按业务时延要求（SLO）显式配置，避免默认值不符合预期。

|参数|可选/必选| 说明                                                                                                         |
|---|---|------------------------------------------------------------------------------------------------------------|
|`n_particles`|必选| 寻优种子数，即一组生成的参数组合数，取值范围：1-1000 的整数。建议设为 8~16。                                                               |
|`iters`|必选| 迭代轮次数，取值范围：1-1000 的整数。建议设为 4~8。                                                                            |
|`ttft_penalty`|必选| `time_to_first_token` 即首 token 时延超时惩罚系数，若对 `time_to_first_token` 没有时延要求设置为 0 即可。取值范围：[0, 100]。建议设为 1。      |
|`tpot_penalty`|必选| `time_per_output_token` 即非首 token 时延超时惩罚系数，若对 `time_per_output_token` 没有时延要求设置为 0 即可。取值范围：[0, 100]。建议设为 1。 |
|`success_rate_penalty`|必选| 请求成功率惩罚系数，取值范围为：1-1000 的整数。建议设为 5。                                                                         |
|`ttft_slo`|必选| `time_to_first_token` 的限制时延。如对 `time_to_first_token` 限制为 2s 内，则设为 2，取值范围：(0, 100]，单位 s。                    |
|`tpot_slo`|必选| `time_per_output_token` 的限制时延。如对 `time_per_output_token` 限制为 50ms 内，则设为 0.05，取值范围：(0, 100]，单位 s。           |
|`optimizer_strategy`|可选| 寻优模式，取值 `pso`（默认）或 `agent`，请参见[寻优模式：PSO 与 Agent](plugin_and_modes.md#6-寻优模式pso-与-agent)。|
|`use_request_rate_calibration`|可选| 是否对 `REQUESTRATE` 做实测校准，默认 `true`。详见下方说明。                                                                  |
|`benchmark_run_count`|可选| `standard` 模式下每组候选的测评次数，默认 `1`，须为正整数。大于 `1` 时，前 N−1 轮预热、最后一轮计分；须同时设置 `use_request_rate_calibration=false`。|

例如，在同一服务实例上先预热再正式测量，在 TOML 顶层设置：

```toml
use_request_rate_calibration = false
benchmark_run_count = 2
```

各轮使用同一组参数，轮次之间保留服务运行，只将最后一轮指标用于评分和结果记录。该设置适用于单机和多机插件的 standard 流程，包括基线、PSO 候选、验证及微调；`pd_disagg` 模式仍使用各阶段自己的 `benchmark_run_count`。模拟 Decode 时还需开启 prefix caching，并固定各轮请求内容、顺序、请求数和采样参数。

重复压测要求压测插件在每轮开始前清理上一轮结果，或使用独立的轮次结果目录，确保只读取当前轮指标。使用外部插件时请确认其满足该约定，否则正式计分可能读取到预热轮指标。

当 `ttft_penalty > 0` 且 `tpot_penalty = 0` 时，最终选择器会从 TTFT 严格低于 `ttft_slo` 的已验证候选中选择输出 token 吞吐最高的一组；若没有这样的候选，则选择 TTFT 最接近上限的一组。此模式不按 TPOT 筛选候选，SLO 边界规则与现有 TPOT-only 和双约束选择器一致。

**`use_request_rate_calibration` 两种模式**：该开关决定 `CONCURRENCY` 与 `REQUESTRATE` 两个特殊字段在 PSO 中的行为。

|取值| 说明                            |CONCURRENCY|REQUESTRATE|
|---|-------------------------------|---|---|
|`true`（默认）| 在 `CONCURRENCY` 固定的情况下，搜索较优的 `REQUESTRATE` |固定为 `max`，不参与搜索|先以最大施压速率实测，再按实测吞吐校准|
|`false`| 开箱阶段，目标搜索较优的 `CONCURRENCY`    |在 `[min, max]` 内参与 PSO 搜索|固定为最大施压速率|

`REQUESTRATE=0` 表示不限速，因此“最大施压速率”按业务语义取值：范围包含 `0` 时取 `0`，否则取配置的数值 `max`。Agent 模式下该开关必须为 `false`，原因见[寻优模式：PSO 与 Agent](plugin_and_modes.md#6-寻优模式pso-与-agent)。

> [!NOTE]
> 用户可根据预估时间自行配置种子和迭代次数。单个种子使用时间为拉起服务 + 测试数据。比如用户拉起服务 + 完成测试需 15min，且愿意用 8h 来进行寻优，则一共可跑约 50 个种子，建议配置 `n_particles=8`、`iters=4`（种子数为迭代次数的 2 倍左右）。

<a id="测评工具参数"></a>
<a id="12-测评工具参数"></a>

## 3. 测评工具参数

<details open>
<summary>使用 AISBench 测评工具</summary>

<a id="aisbench-工具参数"></a>
<a id="13-aisbench-工具参数"></a>

### 3.1 AISBench 工具参数

使用 ais_bench 测评工具时，需修改 `config.toml` 中的 `[ais_bench.command]` 参数，可参照 [AISBench 快速入门](https://github.com/AISBench/benchmark/blob/master/docs/source_zh_cn/get_started/quick_start.md)进行修改。

|参数| 说明                                                                                                                               |
|---|----------------------------------------------------------------------------------------------------------------------------------|
|`models`| 指定模型任务，可根据[模型配置说明](https://github.com/AISBench/benchmark/blob/master/docs/source_zh_cn/base_tutorials/all_params/models.md)进行配置。 |
|`mode`| 运行模式。可根据[运行模式说明](https://github.com/AISBench/benchmark/blob/master/docs/source_zh_cn/base_tutorials/all_params/mode.md)进行配置。     |
|`others`| 拼接其他参数，如 `--num_prompts 1000` 等，参数间使用空格分隔。默认为空。                                                                         |

</details>

<details>
<summary>使用 vllm_benchmark 测评工具</summary>

<a id="vllm_benchmark-工具参数"></a>
<a id="14-vllm_benchmark-工具参数"></a>

### 3.2 vllm_benchmark 工具参数

使用 vllm_benchmark 测评工具时，需修改 `config.toml` 中的 `[vllm_benchmark.command]` 参数：

|参数|可选/必选|说明|
|---|---|---|
|`host`|必选|主机 ip，需与 `[vllm.command]` 中的 `host` 保持一致，可设为 `127.0.0.1`。|
|`port`|必选|端口号，需与 `[vllm.command]` 中的 `port` 保持一致。|
|`model`|必选|模型路径，需与 `[vllm.command]` 中的 `model` 保持一致。|
|`served_model_name`|必选|模型名称，需与 `[vllm.command]` 中的 `served_model_name` 保持一致。|
|`dataset_name`|必选|数据集名称。|
|`others`|可选|拼接其他参数，注意参数间使用空格分隔，参数内部不能留有空格。如 `--dataset-path /path/to/data --num-prompts 1000`。默认为空。|

</details>

<a id="服务化参数"></a>
<a id="15-服务化参数"></a>

## 4. 服务化参数

<details open>
<summary>使用 vLLM 服务框架</summary>

<a id="vllm-服务化参数"></a>
<a id="16-vllm-服务化参数"></a>

### 4.1 vLLM 服务化参数

使用 vLLM 框架时，需修改 `config.toml` 中的 `[vllm.command]` 参数：

```toml
[vllm.command]
host = "127.0.0.1"
port = "8000"
model = "/workspace/vllm/models/llama-2-7b-chat-hf"
served_model_name = "llama-2-7b-chat-hf"
others = ""
```

|参数|可选/必选|说明|
|---|---|---|
|`host`|必选|主机 ip，需与 `[vllm_benchmark.command]` 中的 `host` 保持一致。|
|`port`|必选|端口号，需与 `[vllm_benchmark.command]` 中的 `port` 保持一致。|
|`model`|必选|模型路径。|
|`served_model_name`|必选|模型名称。|
|`others`|可选|追加固定参数，参数间使用空格分隔。如 `--tensor-parallel-size 2 --no-enable-prefix-caching`。默认为空。|

<a id="vllm-自定义参数寻优"></a>
<a id="17-vllm-自定义参数寻优"></a>

### 4.2 vLLM 自定义参数寻优

寻优工具支持通过 `[[vllm.target_field]]` 添加 VLLM 参数参与寻优。根据参数生效方式不同，配置方式分为两类：

- **VLLM 环境变量**：只需在 `[[vllm.target_field]]` 中声明，且 `config_position = "env"`。工具会在每轮寻优启动服务前自动写入同名大写环境变量，不需要写入 `[vllm.command]` 的 `others`。
- **VLLM 命令行参数**：在 `[[vllm.target_field]]` 中声明，且 `config_position = "run"`；工具会自动将字段名渲染为同名 `--kebab-case` 参数并追加到 `vllm serve`，无需在 `others` 中引用。

对于 `config_position = "run"` 的布尔开关，执行字段值为 `false` 时会关闭 `others` 中的同名开关，例如 `enforce_eager=false` 会移除 `--enforce-eager`；该规则适用于 PSO 和 Agent 模式的 vLLM 服务参数渲染。

如需给 vLLM 服务进程注入固定环境变量，可在 `[vllm.env]` 中按 key/value 配置：

```toml
[vllm.env]
ASCEND_RT_VISIBLE_DEVICES = "8,9"
```

这些变量会直接进入 vLLM 子进程环境，不会拼成命令行参数；如果同名变量同时出现在 `[[vllm.target_field]]` 的 `env` 字段中，寻优字段会按轮次覆盖固定值。

> **变量引用规则**：`others` 仅用于固定参数；如需显式展开环境变量类字段，可使用 `$字段名大写`。

<a id="示例一vllm-环境变量寻优"></a>
<a id="171-示例一vllm-环境变量寻优"></a>

#### 4.2.1 示例一：VLLM 环境变量寻优

待寻优参数本身是 VLLM 环境变量时，只需添加到 `[[vllm.target_field]]`：

```toml
[[vllm.target_field]]
name = "VLLM_WORKER_MULTIPROC_METHOD"
config_position = "env"
dtype = "enum"
dtype_param = ["fork", "spawn"]
value = "fork"
```

<a id="示例二命令行参数寻优"></a>
<a id="172-示例二命令行参数寻优"></a>

#### 4.2.2 示例二：命令行参数寻优

声明 `config_position = "run"` 后，工具会自动把字段名渲染为同名 `--kebab-case` 参数并追加到 `vllm serve`，无需在 `others` 中引用。

普通标量参数：

```toml
[[vllm.target_field]]
name = "GPU_MEMORY_UTILIZATION"
config_position = "run"
dtype = "enum"
dtype_param = [0.9, 0.91, 0.92]
value = 0.9
```

生成：

```bash
--gpu-memory-utilization 0.9
```

JSON 容器参数：

```toml
[[vllm.target_field]]
name = "COMPILATION_CONFIG"
config_position = "run"
dtype = "enum"
dtype_param = ["", '{"cudagraph_mode": "FULL_DECODE_ONLY"}']
value = '{"cudagraph_mode": "FULL_DECODE_ONLY"}'
```

生成：

```bash
--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'
```

空字符串候选表示不追加该参数。TOML 单引号字面量可以直接包含 JSON 双引号；若使用双引号字符串，则需将 JSON 内部双引号写成 `\"`。

<a id="vllm-常用寻优字段"></a>
<a id="18-vllm-常用寻优字段"></a>

### 4.3 VLLM 常用寻优字段

|字段名|dtype|说明|
|---|---|---|
|`MAX_NUM_BATCHED_TOKENS`|`int`|单批最大 token 数（含 prefill），影响显存与吞吐|
|`MAX_NUM_SEQS`|`int`|单批最大序列数，即 decode 并发上界|
|`CONCURRENCY`|`int`|测评并发数，特殊字段，行为见下方说明|
|`REQUESTRATE`|`float`|测评发送速率，特殊字段，行为见下方说明|

> **`CONCURRENCY` / `REQUESTRATE` 特殊行为**：这两个字段对 vLLM 与 MindIE 都适用，工具会按 `use_request_rate_calibration` 自动改写：`true` 时把 `CONCURRENCY` 固定为 `max`，先以最大施压速率运行，再按实测吞吐校准 `REQUESTRATE`；`false` 时 `CONCURRENCY` 参与 PSO 搜索，`REQUESTRATE` 固定为最大施压速率。`REQUESTRATE=0` 表示不限速：范围包含 `0` 时最大施压速率为 `0`，否则为数值 `max`。当 `min == max` 时字段被视为常量、不参与搜索。

</details>

<details>
<summary>使用 MindIE 服务框架</summary>

<a id="mindie-服务化参数"></a>
<a id="19-mindie-服务化参数"></a>

### 4.4 MindIE 服务化参数

使用 MindIE 框架时，需修改 `config.toml` 中的 `[mindie.command]` 参数，可参考 [MindIE server配置参数说明](https://www.hiascend.com/document/detail/zh/mindie/20RC1/mindieservice/servicedev/mindie_service0285.html)进行修改。服务化参数可直接指定参数的范围，如配置 `max_batch_size` 的寻优搜索空间为 10~400：

```toml
[[mindie.target_field]]
name = "max_batch_size"
config_position = "BackendConfig.ScheduleConfig.maxBatchSize"
min = 10
max = 400
dtype = "int"
```

也可设置参数与另一参数相关，如 `max_prefill_batch_size` 与 `max_batch_size` 相关（`max_prefill_batch_size = ratio * max_batch_size`，`0 < ratio < 1`）：

```toml
[[mindie.target_field]]
name = "max_prefill_batch_size"
config_position = "BackendConfig.ScheduleConfig.maxPrefillBatchSize"  # 该值不得超过 maxBatchSize
min = 0.1
max = 0.7
dtype = "ratio"
dtype_param = "max_batch_size"
```

<a id="mindie-常用寻优字段"></a>
<a id="110-mindie-常用寻优字段"></a>

### 4.5 MindIE 常用寻优字段

|字段名|config_position|dtype|说明|
|---|---|---|---|
|`max_batch_size`|`BackendConfig.ScheduleConfig.maxBatchSize`|`int`|最大批大小，吞吐主调节点|
|`max_prefill_batch_size`|`BackendConfig.ScheduleConfig.maxPrefillBatchSize`|`ratio`|按比例随 `max_batch_size` 推导，比例建议 `0.1~0.7`|
|`prefill_time_ms_per_req`|`BackendConfig.ScheduleConfig.prefillTimeMsPerReq`|`range`|单请求 prefill 耗时上限（ms），按步长枚举|
|`decode_time_ms_per_req`|`BackendConfig.ScheduleConfig.decodeTimeMsPerReq`|`range`|单请求 decode 耗时上限（ms），按步长枚举|
|`support_select_batch`|`BackendConfig.ScheduleConfig.supportSelectBatch`|`bool`|是否启用按耗时选批，开启后 `prefill/decode_time_ms_per_req` 自动置 0|
|`max_queue_delay_microseconds`|`BackendConfig.ScheduleConfig.maxQueueDelayMicroseconds`|`range`|请求排队最大时延（μs），按步长枚举|
|`max_preempt_count`|`BackendConfig.ScheduleConfig.maxPreemptCount`|`ratio`|抢占上限，按比例随 `max_batch_size` 推导|
|`CONCURRENCY`|`env`|`int`|测评并发数，特殊字段，见 VLLM 段说明|
|`REQUESTRATE`|`env`|`float`|测评发送速率，特殊字段，见 VLLM 段说明|

</details>

<a id="target_field-配置文件说明"></a>
<a id="1101-target_field-配置文件说明"></a>

#### 4.5.1 target_field配置文件说明

| 字段 | 含义                                                                                                                                                                                          | 示例 |
|---|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---|
| `name` | 寻优字段名。环境变量类需大写，工具每轮启动服务前自动写入同名大写环境变量；命令行类由工具自动渲染为同名 `--kebab-case` 参数                                                                                                                                   | `COMPILATION_CONFIG` |
| `config_position` | 生效位置：`env`（仅环境变量，不拼命令行）、`run`（命令行参数，工具自动追加同名 `--kebab-case` 参数，无需在 `others` 引用）或 MindIE 服务配置文件路径（如 `BackendConfig.ScheduleConfig.maxBatchSize`） | `run` |
| `dtype` | 取值类型，决定 PSO 采样方式：`int`/`float`/`bool`/`enum`/`range`/`ratio`/`share`/`factories`/`times`/`ternary_factories`/`ternary_times`，各类型含义见 [target_field 支持的 dtype 类型](#target_field-支持的-dtype-类型) | `enum` |
| `dtype_param` | 依 dtype 而定：`enum` 为候选值列表、`range` 为步长整数、`ratio` 与派生类为依赖字段配置                                                                                                                                  | `["", '{"cudagraph_mode": "FULL_DECODE_ONLY"}']` |
| `value` | 初始值，用于生成基线数据                                                                                                                                                                                | `'{"cudagraph_mode": "FULL_DECODE_ONLY"}'` |
| `min` / `max` | 搜索区间上下界（`int`/`float`/`range` 生效）；`min == max` 时字段视为常量、不参与搜索；`enum` 的实际候选由 `dtype_param` 给出，派生类型需将 `min`/`max` 均设为 0                                                                        | — |
| `constant` | 可选；设值后字段固定为该值、不参与 PSO 搜索（`CONCURRENCY`/`REQUESTRATE` 自动改写时由工具写入）                                                                                                                            | — |

<a id="target_field-支持的-dtype-类型"></a>
<a id="1102-target_field-支持的-dtype-类型"></a>

#### 4.5.2 target_field 支持的 dtype 类型

|分类|dtype|含义|dtype_param 格式|
|---|---|---|---|
| 基础类型 | `int` | 在 [min, max] 内取整数 | — |
| 基础类型 | `float` | 在 [min, max] 内取浮点数 | — |
| 基础类型 | `bool` | 布尔开关（参数值 > 0.5 时为 true） | — |
| 基础类型 | `enum` | 从候选列表中选值（支持数值或字符串） | 候选值列表，如 `[1, 2, 4, 8]` |
| 基础类型 | `range` | 按正步长在 [min, max] 内枚举，仅在步长对齐时包含 max（如 min=1、max=9、step=3 时为 [1, 4, 7]） | 正整数步长，如 `10` |
| 二元派生 | `ratio` | `int(比例 × target)` | 依赖字段名（字符串），如 `"max_batch_size"` |
| 二元派生 | `share` | `target.min + target.max - target.value`（互补） | 依赖字段名（字符串） |
| 二元派生 | `factories` | `product ÷ target` | `{"target_name": "字段名", "product": 值, "dtype": "int"}` |
| 二元派生 | `times` | `product × target` | `{"target_name": "字段名", "product": 值, "dtype": "int"}` |
| 三元派生 | `ternary_factories` | `product ÷ (field_a × field_b)` | `{"target_names": ["A", "B"], "product": 值, "dtype": "int"}` |
| 三元派生 | `ternary_times` | `product × field_a × field_b` | `{"target_names": ["A", "B"], "product": 值, "dtype": "int"}` |

> [!NOTE]
> 派生类型字段（`factories` / `times` / `ternary_factories` / `ternary_times`）的值由依赖关系自动推导，**不参与粒子群搜索**，需将 `min` 和 `max` 均设为 `0`。若任一依赖字段值为 `0`（除法场景）或 `None`/`NaN`（乘法场景），本轮推导跳过，字段保持原值并输出警告日志。
>
> 三元派生类型的完整用法（`ternary_factories` / `ternary_times`、`priority_policy` 优先级修复）见[附录](#三元派生类型使用示例)。

<details>
<summary>高级配置</summary>

## 5. 高级配置

<a id="部署环境"></a>
<a id="111-部署环境"></a>

### 5.1 部署环境

服务化实测寻优拉起 vLLM、MindIE 或测评工具时，optix 会先去掉 msmodeling 虚拟环境相关变量，再按下面配置找部署根目录，`bin/` 下应有 `vllm`、`ais_bench` 等，部署环境配置位于 `config.toml` 中的 `[deploy]`：

|参数|可选/必选|说明|
|------|------|------|
|`path_prefix`|可选|部署根目录，其 `bin/` 会前置到剥离 msmodeling 虚拟环境后的 PATH；未找到命令时继续搜索原有 PATH。`OPTIX_DEPLOY_PATH` 优先于此配置|

```toml
# 仅当系统 PATH 找不到 vllm、ais_bench 等时再打开
[deploy]
# path_prefix = "/path/to/custom-deploy-root"
```

<a id="结果存储与精调"></a>
<a id="112-结果存储与精调"></a>

### 5.2 结果存储与精调

仅启用 TTFT 惩罚（`ttft_penalty > 0`、`tpot_penalty = 0`）时，最终判优时优先选择 TTFT 低于 SLO 的候选，再在其中选择生成吞吐最高者；如果所有候选均超出 TTFT SLO，则选择 TTFT 最低者。关闭 FineTune 时，工具在复测后停止自己管理的服务；`manage_simulator_lifecycle=false` 的外部服务由调用方管理。

PSO 阶段结束后，工具会从结果中选出若干组候选进入精调（fine-tune）阶段，`pso_top_k` 控制选取数量，结果存储与精调配置位于 `config.toml` 中的 `[data_storage]`：

|参数|可选/必选| 说明                                                                    |
|---|---|-----------------------------------------------------------------------|
|`store_dir`|可选| 结果 CSV 存储目录，默认 `store`。                                               |
|`pso_top_k`|可选| 从 PSO 结果中选取的候选组数，默认为 0，关闭 FineTune，但仍复测最佳 PSO 候选。选取逻辑：取 fitness 最小的 top-k；深度精调场景建议 `3~5`。 |

```toml
[data_storage]
pso_top_k = 0
```

<a id="benchmark-早停"></a>
<a id="113-benchmark-早停"></a>

### 5.3 Benchmark 早停

Benchmark 早停用于提前结束性能明显差于历史完整结果的普通 PSO 候选，减少服务化实测寻优时间。该功能默认关闭，目前仅支持 `-e vllm`。

早停只作用于普通 PSO evaluation。baseline、request-rate calibration、refinement 和最优候选复测不会被提前终止，以保证基准和最终结果完整。

首次启用时，建议先使用 `action = "report"` 验证策略：

```toml
[benchmark_early_exit]
enabled = true
action = "report"
metrics_url = "http://127.0.0.1:8000/metrics"
```

|参数|可选/必选|默认值|说明|
|---|---|---|---|
|`enabled`|可选|`false`|是否启用早停。|
|`action`|可选|`"terminate"`|`report` 仅记录本应早停的时刻，不终止当前 Case；`terminate` 实际停止当前 benchmark 和服务。|
|`metrics_url`|可选|`"http://127.0.0.1:8000/metrics"`|vLLM Prometheus metrics 地址，应与实际服务监听地址和端口一致。|

启用后，系统先从 baseline 或其他完整 Case 建立 reference。后续普通 PSO 候选经过 warmup 后，若连续多个有效窗口显著差于 reference，系统生成早停决定。

推荐按以下步骤启用：

1. 设置 `action = "report"`，完成一轮寻优；
2. 检查 `data_storage_*.csv` 和 `metrics_samples_*.csv`；
3. 确认不存在明显误判后，将 `action` 改为 `"terminate"`；
4. 使用相同模型、数据集和业务负载继续寻优。

> [!NOTE]
> `report` 模式不会缩短运行时间，只用于验证早停判断，并生成 `metrics_samples_*.csv` 采样轨迹。`terminate` 模式才会实际节省时间。

**注意：**

- 当前仅支持 `vllm` engine，不支持 `mindie` 或 `vllm_pd` 等扩展 engine。
- 短 Case、没有可用 reference 或 `/metrics` 暂时不可访问时，系统不会执行早停。
- 修改模型、数据集或业务负载后，应重新建立相同口径的完整 reference。
- 早停用于识别“服务正常但性能较差”的候选，不能替代服务启动失败、健康检查或请求失败处理。

<a id="日志检测"></a>
<a id="114-日志检测"></a>

### 5.4 日志检测

检查日志中出现的异常信息，区分致命错误和可重试错误，实现智能错误处理和重试机制。可检测的错误类型包括内存溢出（OOM）、设备故障（NPU）、网络错误和 IO 错误等。致命错误（如 OOM、NPU 故障）会立即停止调度器，可重试错误（如网络抖动、IO 失败）会触发自动重试（最多 3 次），日志检测配置位于 `config.toml` 中的 `[health_check]`：

|参数|可选/必选|说明|
|---|---|---|
|`log_snippet_length`|可选|日志片段长度，用于显示错误详情。取值范围：50-1000，默认为 200。|
|`service_errors.fatal_patterns`|可选|服务化框架致命错误模式列表，默认为空。|
|`service_errors.retryable_patterns`|可选|服务化框架可重试错误模式列表，默认为空。|
|`benchmark_errors.fatal_patterns`|可选|测评工具致命错误模式列表，默认为空。|
|`benchmark_errors.retryable_patterns`|可选|测评工具可重试错误模式列表，默认为空。|

```toml
[health_check]
log_snippet_length = 200

# 服务化框架：致命错误（立即停止，不重试）
[health_check.service_errors.fatal_patterns]
out_of_memory = ["out of memory", "OOM killed", "MemoryError"]
device_error = ["NPU error", "device fault", "Ascend error"]

# 服务化框架：可重试错误（最多重试 3 次）
[health_check.service_errors.retryable_patterns]
network_error = ["connection reset", "connection refused", "timeout"]
io_error = ["file not found", "permission denied", "IO error"]

# 测评工具：致命错误（立即停止，不重试）
[health_check.benchmark_errors.fatal_patterns]
out_of_memory = []
device_error = []

# 测评工具：可重试错误（最多重试 3 次）
[health_check.benchmark_errors.retryable_patterns]
network_error = []
io_error = []
```

> 默认所有模式列表为空（`[]`），即不检测。建议先填 `service_errors`，再按测评工具实际报错补 `benchmark_errors`；避免使用 `"error"`、`"exception"`、`"failed"` 等过宽模式，优先用完整错误串（如 `"out of memory"` 而非 `"memory"`），避免误匹配。

</details>

<a id="三元派生类型使用示例"></a>
<a id="115-三元派生类型使用示例"></a>

## 6. 三元派生类型使用示例

<a id="场景一dp-由总卡数自动推导"></a>
<a id="1151-场景一dp-由总卡数自动推导"></a>

### 6.1 场景一：dp 由总卡数自动推导

`tp`、`pp` 为可调参数，`dp` 由总卡数（16）自动推导（`dp = 16 ÷ (tp × pp)`）：

> [!NOTE]
> `ternary_factories` 要求各依赖字段的乘积能合法推出派生字段。对于 `dtype = "int"`，`product` 必须能被依赖字段乘积整除，否则会触发优先级修复。限制 `tp`、`pp` 的枚举候选使乘积可整除 `product` 是最佳实践。

```toml
# 方式一（最佳实践）：限制 tp 和 pp 的枚举候选值，保证 tp × pp ≤ 16
[[mindie.target_field]]
name = "tp"
config_position = "BackendConfig.ModelDeployConfig.ModelConfig.0.tp"
min = 0
max = 1
dtype = "enum"
dtype_param = [1, 2, 4, 8]   # tp 最大为 8

[[mindie.target_field]]
name = "pp"
config_position = "BackendConfig.ModelDeployConfig.ModelConfig.0.pp"
min = 0
max = 1
dtype = "enum"
dtype_param = [1, 2]          # pp 限制为 1 或 2，保证 tp × pp 最大 8 × 2 = 16 不超出

[[mindie.target_field]]
name = "dp"
config_position = "BackendConfig.ModelDeployConfig.ModelConfig.0.dp"
min = 0
max = 0
dtype = "ternary_factories"
dtype_param = {target_names = ["tp", "pp"], product = 16, dtype = "int"}
# 示例：tp=4, pp=2 → dp = 16 ÷ (4 × 2) = 2
#        tp=8, pp=2 → dp = 16 ÷ (8 × 2) = 1
```

```toml
# 方式二：配置 min_value 作为修复失败后的下界保护，并输出警告
[[mindie.target_field]]
name = "dp"
config_position = "BackendConfig.ModelDeployConfig.ModelConfig.0.dp"
min = 0
max = 0
dtype = "ternary_factories"
dtype_param = {target_names = ["tp", "pp"], product = 16, dtype = "int", min_value = 1}
# 如果没有可修复的合法组合，且结果低于 min_value，会降级至 min_value=1，并输出 WARNING
```

**优先级修复策略（`priority_policy`）**

当 PSO 生成的 `tp`、`pp` 组合不能合法推出 `dp`（如不能整除、超界）时，系统会尝试修复。修复策略由 `priority_policy` 控制：

| 策略名 | 语义 | 适用场景 |
|--------|------|----------|
| `balanced`（默认） | 将粒子均分两组：前半用 `target_names` 顺序修复，后半用反序修复，降低单一解码顺序带来的结构性偏置 | 用户没有明确字段优先偏好，默认使用 |
| `fixed` | 用户显式指定修复顺序：高优先级字段尽量保持不动，优先调整低优先级字段 | 用户明确知道哪个字段更应该稳定 |

```toml
# balanced（默认）策略示例
# 适用：用户没有指定哪个字段更重要，系统自动均衡分配修复方向
[[mindie.target_field]]
name = "dp"
config_position = "BackendConfig.ModelDeployConfig.ModelConfig.0.dp"
min = 0
max = 0
dtype = "ternary_factories"
# 默认即为 balanced，可不写 priority_policy。
dtype_param = {target_names = ["tp", "pp"], product = 32, dtype = "int", priority_policy = "balanced"}
```

```toml
# fixed 策略示例
# 适用：用户明确知道 tp 应保持稳定，优先调整 pp
[[mindie.target_field]]
name = "dp"
config_position = "BackendConfig.ModelDeployConfig.ModelConfig.0.dp"
min = 0
max = 0
dtype = "ternary_factories"
# tp 高优先：尽量保留 tp，首先调整 pp。
dtype_param = {target_names = ["tp", "pp"], product = 32, dtype = "int", priority_policy = "fixed", priority = ["tp", "pp"]}
# 示例： tp=8、pp=3（非法）：
#   stage1：固定 tp=8，在 pp 候选中找最近合法值 → pp=4， dp=1
#   stage1 失败时再 stage2：两个字段均可调整，按距离升序搜索
```

> [!note] priority_policy说明
>
> - `balanced` 是默认策略，不配置时自动生效。
> - 修复分两阶段：stage1 固定高优先字段、调整低优先字段；stage1 失败后 stage2 两个字段均可调整。
> - 全部候选都不合法时，修复失败，降级至 min/max 截断，并输出 warning。

场景二：`seq_len`、`prefill_batch_size` 为可调参数，`max_prefill_tokens` 自动设为二者之积的 2 倍（`max_prefill_tokens = 2 × seq_len × prefill_batch_size`）：

```toml
[[mindie.target_field]]
name = "seq_len"
config_position = "BackendConfig.ModelConfig.seqLen"
min = 0
max = 1
dtype = "enum"
dtype_param = [512, 1024, 2048, 4096]

[[mindie.target_field]]
name = "prefill_batch_size"
config_position = "BackendConfig.ScheduleConfig.maxPrefillBatchSize"
min = 1
max = 16
dtype = "int"

[[mindie.target_field]]
name = "max_prefill_tokens"
config_position = "BackendConfig.ScheduleConfig.maxPrefillTokens"
min = 0         # 设为 0 使其成为常量，不参与搜索
max = 0
dtype = "ternary_times"
dtype_param = {target_names = ["seq_len", "prefill_batch_size"], product = 2, dtype = "int"}
# seq_len=1024, prefill_batch_size=4 → max_prefill_tokens = 2 × 1024 × 4 = 8192
```
