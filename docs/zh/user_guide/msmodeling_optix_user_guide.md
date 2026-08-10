# 服务化实测寻优使用指南

## 简介

**服务化实测寻优**（msmodeling optix）是一项基于 PSO 粒子寻优算法的服务化参数实测寻优功能，支持在真实服务框架上自动搜索，获取符合TTFT/TPOT等时延要求的最佳吞吐参数组合。

工具主要包括两大核心功能模块：

- **参数寻优模块**：利用 PSO 粒子寻优算法自动生成服务化参数组合，不断逼近最优解。
- **参数验证模块**：自动化启动服务化进程与测评工具进程，进行参数测试，获取性能结果。当前已支持的服务框架包括 `vLLM` 和 `MindIE`，测评工具包括 `AISBench`、`vllm_benchmark`。

目前工具已基于 DeepSeek V3.1、GLM5 和 Qwen3.5-27b 通过验证，理论上不限制支持模型范围。

## 适用对象与阅读路径

本文适用于需要对 vLLM、MindIE 服务化部署参数进行自动寻优的性能工程师和部署工程师。建议按以下顺序阅读：

1. [环境准备与安装](#环境准备与安装) — 在 uv 虚拟环境中安装 msmodeling，确认系统已部署 vLLM/MindIE。
2. [快速入门](#快速入门) — 完成一次默认寻优。
3. [配置文件说明](#配置文件说明) 和 [命令参数说明](#命令参数说明) — 了解全部参数与配置项。
4. [结果文件说明](#结果文件说明) — 根据业务 SLO 筛选最优参数。
5. [附录](#附录) — 运维与排障与三元派生类型用法按需查阅。

## 产品支持情况

> [!NOTE]
> 昇腾产品的具体型号，请参见《[昇腾产品形态说明](https://www.hiascend.com/document/detail/zh/AscendFAQ/ProduTech/productform/hardwaredesc_0001.html)》。
> 针对 Atlas A2 训练系列产品/Atlas A2 推理系列产品，当前仅支持该系列产品中的 Atlas 800I A2 推理服务器。
> 针对 Atlas 推理系列产品，当前仅支持该系列产品中的 Atlas 300I Duo 推理卡 + Atlas 800 推理服务器（型号：3000）。

|产品类型|是否支持|
|--|:----:|
|Atlas A3 训练系列产品/Atlas A3 推理系列产品|√|
|Atlas A2 训练系列产品/Atlas A2 推理系列产品|√|
|Atlas 200I/500 A2 推理产品|√|
|Atlas 推理系列产品|√|
|Atlas 训练系列产品|×|
|Atlas 350 加速卡|×|

## 插件支持

服务化实测寻优支持用户自定义搜索参数配置以及测试工具，只需适配插件模式注册对应插件即可。内置插件随 `optix` 包发布，通过 `-e`/`-b` 直接选用；也可通过插件模式接入自定义插件。

**内置插件**：

|插件名称|类型|插件描述|
|---|---|---|
|`vllm`|服务框架（`-e`）|vLLM 服务化框架适配|
|`mindie`|服务框架（`-e`）|MindIE 服务化框架适配|
|`ais_bench`|测评工具（`-b`）|AISBench 测评工具适配|
|`vllm_benchmark`|测评工具（`-b`）|vllm_benchmark 测评工具适配|

**扩展插件示例**（位于 `contrib/optix/`，按需安装）：

|插件名称|类型|提供者|插件描述|插件目录|
|---|---|---|---|---|
|`custom_vllm`|服务框架（`-e`）|contrib 示例|通过 Docker + SSH 拉起本地与远端 vLLM 容器，演示自定义服务框架接入|`contrib/optix/vllm_msserviceprofiler/`|
|`evalscopeperf`|测评工具（`-b`）|contrib 示例|封装 `evalscope perf` 命令作为测评工具，演示自定义测评工具接入|`contrib/optix/evalscopeplugin/`|

> 插件通过 Python entry points 注册，自定义插件开发与使用详见[插件开发操作步骤](msmodeling_optix_plugin_user_guide.md)。

## 环境准备与安装

### 环境隔离原则

服务化实测寻优 optix 推荐装在虚拟环境里。在仓库根目录执行 `uv sync` 即可自动创建 `.venv` 并完成安装。

安装 msmodeling 会同时装上 `torch`、`transformers` 等包，如果在系统 Python 里安装 msModeling，可能会与系统里原有的 `torch`、`transformers` 版本冲突，导致：

- vLLM、MindIE 起不来或推理报错
- 和 Ascend 推理栈上已验证的版本对不上

### 安装步骤

```bash
git clone https://gitcode.com/Ascend/msmodeling.git   # 如已拉取，则不用重复拉取
cd msmodeling
uv sync
```

> [!WARNING]
> 不要在 msmodeling 虚拟环境里 `pip install vllm`、`mindie_llm` 等部署包，也不要在未建 venv 的系统 Python 里安装 msmodeling，若有疑问，可参考[推荐实践：服务化实测寻优环境与部署栈](../install_guide/msmodeling_optix_env_and_deployment_stack.md)

### 验证安装

```bash
uv run msmodeling optix --help
```

能正常打印 `msmodeling optix` 帮助信息即表示工具安装成功；部署栈检查与正式寻优见[快速入门](#快速入门)。

### 卸载

```bash
uv pip uninstall msmodeling
```

## 快速入门

1. **修改配置文件**：完成[环境准备与安装](#环境准备与安装)，按实际情况修改配置文件 `config.toml`，包括寻优参数、测评工具参数、服务化参数。详见 [配置文件说明](#配置文件说明)。

   > **说明：** 若不确定该调哪些参数及参数范围，可使用 **`optix-param-recommend` 参数推荐 skill**，输入硬件、模型、负载和优化目标后即可获得推荐参数与搜索范围，再按实际部署环境（显存、卡数、时延要求等）核对并修改后写入 `config.toml`。

2. **启动寻优**：默认执行的是基于 `AISBench` 的 `vLLM` 服务化参数寻优，其它用法详见 [命令参数说明](#命令参数说明)。

    ```bash
    msmodeling optix
    ```

3. **查看结果**：寻优时间由模型大小、数据集大小和寻优次数等决定，结束后会生成 `data_storage_*.csv` 文件并保存在**当前工作目录下的 `result/store`** 子目录中（可用 `[data_storage].store_dir` 修改），详见 [结果文件说明](#结果文件说明)。

## 配置文件说明

寻优配置文件默认位于`./msmodeling/optix/config.toml`。文件按 TOML 段组织，各段对应一项功能：

| TOML 段                                               | 用途                      | 文档章节 |
|------------------------------------------------------|-------------------------|---|
| 顶层                                                   | 寻优所需参数                  | [寻优参数](#寻优参数) |
| `[vllm]` / `[mindie]` | vLLM / MindIE 服务化参数与寻优字段 | [服务化参数](#服务化参数) |
| `[ais_bench.command]`  / `[vllm_benchmark.command]`| 测评工具参数                  | [测评工具参数](#测评工具参数) |
| `[deploy]`                                           | 部署环境根目录（可选）             | [高级配置 → 部署环境](#部署环境) |
| `[data_storage]`                                     | 结果存储与精调（可选）             | [高级配置 → 结果存储与精调](#结果存储与精调) |
| `[health_check]`                                     | 运行时日志异常检测（可选）           | [高级配置 → 日志检测](#日志检测) |

### 寻优参数

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
|`use_request_rate_calibration`|可选| 是否对 `REQUESTRATE` 做实测校准，默认 `true`。详见下方说明。                                                                  |

**`use_request_rate_calibration` 两种模式**：该开关决定 `CONCURRENCY` 与 `REQUESTRATE` 两个特殊字段在 PSO 中的行为。

|取值| 说明                            |CONCURRENCY|REQUESTRATE|
|---|-------------------------------|---|---|
|`true`（默认）| 在 `CONCURRENCY` 固定的情况下，搜索较优的 `REQUESTRATE` |固定为 `max`，不参与搜索|以 `max` 为上界做实测校准|
|`false`| 开箱阶段，目标搜索较优的 `CONCURRENCY`    |在 `[min, max]` 内参与 PSO 搜索|固定为 `max`|

> [!NOTE]
> 用户可根据预估时间自行配置种子和迭代次数。单个种子使用时间为拉起服务 + 测试数据。比如用户拉起服务 + 完成测试需 15min，且愿意用 8 小时来进行寻优，则一共可跑约 50 个种子，建议配置 `n_particles=8`、`iters=4`（种子数为迭代次数的 2 倍左右）。

### 测评工具参数

<details open>
<summary>使用 AISBench 测评工具</summary>

### AISBench 工具参数

使用 ais_bench 测评工具时，需修改 `config.toml` 中的 `[ais_bench.command]` 参数，可参照 [AISBench 快速入门](https://github.com/AISBench/benchmark/blob/master/docs/source_zh_cn/get_started/quick_start.md)进行修改。

|参数| 说明                                                                                                                               |
|---|----------------------------------------------------------------------------------------------------------------------------------|
|`models`| 指定模型任务，可根据[模型配置说明](https://github.com/AISBench/benchmark/blob/master/docs/source_zh_cn/base_tutorials/all_params/models.md)进行配置。 |
|`mode`| 运行模式。可根据[运行模式说明](https://github.com/AISBench/benchmark/blob/master/docs/source_zh_cn/base_tutorials/all_params/mode.md)进行配置。     |
|`others`| 拼接其他参数，如 `--num_prompts 1000` 等，参数间使用空格分隔。默认为空。                                                                         |

</details>

<details>
<summary>使用 vllm_benchmark 测评工具</summary>

### vllm_benchmark 工具参数

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

### 服务化参数

<details open>
<summary>使用 vLLM 服务框架</summary>

### vLLM 服务化参数

使用 VLLM 框架时，需修改 `config.toml` 中的 `[vllm.command]` 参数：

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
|`others`|可选|拼接其他参数，参数间使用空格分隔。如 `--tensor-parallel-size 2 --no-enable-prefix-caching`。默认为空。|

### VLLM 自定义参数寻优

寻优工具支持通过 `[[vllm.target_field]]` 添加 VLLM 参数参与寻优。根据参数生效方式不同，配置方式分为两类：

- **VLLM 环境变量**：只需在 `[[vllm.target_field]]` 中声明，且 `config_position = "env"`。工具会在每轮寻优启动服务前自动写入同名大写环境变量，不需要写入 `[vllm.command]` 的 `others`。
- **VLLM 命令行参数**：先在 `[[vllm.target_field]]` 中声明，再在 `[vllm.command]` 的 `others` 中通过变量引用拼接到启动命令。

> **变量引用规则**：在 `others` 中使用 `$字段名大写` 的格式引用寻优字段，工具运行时会自动将其替换为当前迭代的实际值。

#### 示例一：VLLM 环境变量寻优

待寻优参数本身是 VLLM 环境变量时，只需添加到 `[[vllm.target_field]]`：

```toml
[[vllm.target_field]]
name = "VLLM_WORKER_MULTIPROC_METHOD"
config_position = "env"
dtype = "enum"
dtype_param = ["fork", "spawn"]
value = "fork"
```

此类参数无需在 `[vllm.command]` 的 `others` 中引用，保持 `others = ""` 或仅填写其他命令行参数即可。

#### 示例二：命令行枚举数值参数（以 `gpu_memory_utilization` 为例）

```toml
# 第一步：声明寻优字段
[[vllm.target_field]]
name = "GPU_MEMORY_UTILIZATION"
config_position = "env"
dtype = "enum"
dtype_param = [0.9, 0.91, 0.92]
value = 0.9

# 第二步：在 [vllm.command] 的 others 中引用变量
[vllm.command]
others = "--gpu-memory-utilization $GPU_MEMORY_UTILIZATION"
```

#### 示例三：命令行开关型/复合字符串参数（以 `--compilation-config` 为例）

当参数本身是一段完整的 CLI 字符串时，可将"不启用"（空字符串 `""`）和"启用"两种形态作为枚举候选值。工具遇到空字符串时会自动跳过，不向启动命令追加任何内容。

> **注意**：TOML 字符串使用双引号 `"` 作为边界符，若字符串内容中包含双引号，需使用 `\"` 转义。

```toml
[[vllm.target_field]]
name = "COMPILATION_CONFIG"
config_position = "env"
dtype = "enum"
dtype_param = ["", "--compilation-config '{\"cudagraph_mode\": \"FULL_DECODE_ONLY\"}'"]
value = "--compilation-config '{\"cudagraph_mode\": \"FULL_DECODE_ONLY\"}'"

[vllm.command]
others = "$COMPILATION_CONFIG"
```

### VLLM 常用寻优字段

|字段名|dtype|说明|
|---|---|---|
|`MAX_NUM_BATCHED_TOKENS`|`int`|单批最大 token 数（含 prefill），影响显存与吞吐|
|`MAX_NUM_SEQS`|`int`|单批最大序列数，即 decode 并发上界|
|`CONCURRENCY`|`int`|测评并发数，特殊字段，行为见下方说明|
|`REQUESTRATE`|`float`|测评发送速率，特殊字段，行为见下方说明|

> **`CONCURRENCY` / `REQUESTRATE` 特殊行为**：这两个字段对 vLLM 与 MindIE 都适用，工具会按 `use_request_rate_calibration` 自动改写：`true` 时把 `CONCURRENCY` 固定为 `max`、`REQUESTRATE` 以 `max` 为上界做实测校准；`false` 时 `CONCURRENCY` 参与 PSO 搜索、`REQUESTRATE` 固定为 `max`。因此配置时 `max` 要设成期望的最大值；当 `min == max` 时字段被视为常量、不参与搜索。

</details>

<details>
<summary>使用 MindIE 服务框架</summary>

### MindIE 服务化参数

使用 MindIE 框架时，需修改 `config.toml` 中的 `[mindie.command]` 参数，可参考 [MindIE server 配置参数说明](https://www.hiascend.com/document/detail/zh/mindie/20RC1/mindieservice/servicedev/mindie_service0285.html)进行修改。服务化参数可直接指定参数的范围，如配置 `max_batch_size` 的寻优搜索空间为 10~400：

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

### MindIE 常用寻优字段

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

#### target_field 配置文件说明

| 字段 | 含义                                                                                                                                                                                          | 示例 |
|---|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---|
| `name` | 寻优字段名。环境变量类需大写，工具每轮启动服务前自动写入同名大写环境变量；命令行类用作 `$字段名大写` 变量引用                                                                                                                                   | `COMPILATION_CONFIG` |
| `config_position` | 生效位置：`env`（环境变量）或MindIE服务配置文件路径（如 `BackendConfig.ScheduleConfig.maxBatchSize`）                                                                                                              | `env` |
| `dtype` | 取值类型，决定 PSO 采样方式：`int`/`float`/`bool`/`enum`/`range`/`ratio`/`share`/`factories`/`times`/`ternary_factories`/`ternary_times`，各类型含义见 [target_field 支持的 dtype 类型](#target_field-支持的-dtype-类型) | `enum` |
| `dtype_param` | 依 dtype 而定：`enum` 为候选值列表、`range` 为步长整数、`ratio` 与派生类为依赖字段配置                                                                                                                                  | `["", "--compilation-config '{\"cudagraph_mode\": \"FULL_DECODE_ONLY\"}'"]` |
| `value` | 初始值，用于生成基线数据                                                                                                                                                                                | `"--compilation-config '{\"cudagraph_mode\": \"FULL_DECODE_ONLY\"}'"` |
| `min` / `max` | 搜索区间上下界（`int`/`float`/`range` 生效）；`min == max` 时字段视为常量、不参与搜索；`enum` 的实际候选由 `dtype_param` 给出，派生类型需将 `min`/`max` 均设为 0                                                                        | — |
| `constant` | 可选；设值后字段固定为该值、不参与 PSO 搜索（`CONCURRENCY`/`REQUESTRATE` 自动改写时由工具写入）                                                                                                                            | — |

#### target_field 支持的 dtype 类型

|分类|dtype|含义|dtype_param 格式|
|---|---|---|---|
| 基础类型 | `int` | 在 [min, max] 内取整数 | — |
| 基础类型 | `float` | 在 [min, max] 内取浮点数 | — |
| 基础类型 | `bool` | 布尔开关（参数值 > 0.5 时为 true） | — |
| 基础类型 | `enum` | 从候选列表中选值（支持数值或字符串） | 候选值列表，如 `[1, 2, 4, 8]` |
| 基础类型 | `range` | 按步长在 [min, max] 内枚举 | 步长整数，如 `10` |
| 二元派生 | `ratio` | `int(比例 × target)` | 依赖字段名（字符串），如 `"max_batch_size"` |
| 二元派生 | `share` | `target.min + target.max - target.value`（互补） | 依赖字段名（字符串） |
| 二元派生 | `factories` | `product ÷ target` | `{"target_name": "字段名", "product": 值, "dtype": "int"}` |
| 二元派生 | `times` | `product × target` | `{"target_name": "字段名", "product": 值, "dtype": "int"}` |
| **三元派生** | **`ternary_factories`** | **`product ÷ (field_a × field_b)`** | **`{"target_names": ["A", "B"], "product": 值, "dtype": "int"}`** |
| **三元派生** | **`ternary_times`** | **`product × field_a × field_b`** | **`{"target_names": ["A", "B"], "product": 值, "dtype": "int"}`** |

> [!NOTE]
> 派生类型字段（`factories` / `times` / `ternary_factories` / `ternary_times`）的值由依赖关系自动推导，**不参与粒子群搜索**，需将 `min` 和 `max` 均设为 `0`。若任一依赖字段值为 `0`（除法场景）或 `None`/`NaN`（乘法场景），本轮推导跳过，字段保持原值并输出警告日志。
>
> 三元派生类型的完整用法（`ternary_factories` / `ternary_times`、`priority_policy` 优先级修复）见[附录](#三元派生类型使用示例)。

<details>
<summary>高级配置</summary>

### 部署环境

服务化实测寻优拉起 vLLM、MindIE 或测评工具时，optix 会先去掉 msmodeling 虚拟环境相关变量，再按下面配置找部署根目录，`bin/` 下应有 `vllm`、`ais_bench` 等，部署环境配置位于 `config.toml` 中的 `[deploy]`：

|参数|必选|说明|
|------|------|------|
|`path_prefix`|可选|部署根目录，用来覆盖默认系统 PATH。不设则剥离 msmodeling venv 后直接走系统 PATH，效果同 `OPTIX_DEPLOY_PATH`|

```toml
# 仅当系统 PATH 找不到 vllm、ais_bench 等时再打开
[deploy]
# path_prefix = "/path/to/custom-deploy-root"
```

### 结果存储与精调

PSO 阶段结束后，工具会从结果中选出若干组候选进入精调（fine-tune）阶段，`pso_top_k` 控制选取数量，结果存储与精调配置位于 `config.toml` 中的 `[data_storage]`：

|参数|可选/必选| 说明                                                                    |
|---|---|-----------------------------------------------------------------------|
|`store_dir`|可选| 结果 CSV 存储目录，默认 `store`。                                               |
|`pso_top_k`|可选| 从 PSO 结果中选取的候选组数，默认为 0，功能不开启。选取逻辑：取 fitness 最小的 top-k；深度精调场景建议 `3~5`。 |

```toml
[data_storage]
pso_top_k = 0
```

### 日志检测

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

## 命令参数说明

### 命令格式

```bash
msmodeling optix [options]
```

### 注意事项

- 启动寻优前，确认 `vllm` 或 `mindie` 以及 `ais_bench` 或 `vllm_benchmark` 已在系统部署环境里能正常运行，且未部署在 msmodeling 虚拟环境中。
- 配置文件如 `config.toml` 中的模型路径、端口、数据集路径和服务启动参数需与实际部署环境保持一致，详见[配置文件说明](#配置文件说明)。。
- 服务化实测寻优会反复拉起服务化框架并执行测评，耗时通常较长（4-10h），建议在独占或资源稳定的环境中运行。
- 环境隔离异常时，日志前缀 `[optix/env]` 会给出原因与修复建议，详见[环境变量与排障](#环境变量与排障)。

### 参数说明

|参数|可选/必选|说明|
|---|---|---|
|-e或--engine|可选|指定推理框架，可取值：<br>&#8226;vllm：使用VLLM作为推理框架<br>&#8226;mindie：使用MindIE作为推理框架<br/>默认值为`vllm`。|
|-b或--benchmark_policy|可选|指定测评工具，可取值：<br>&#8226;vllm_benchmark：使用vllm_benchmark作为测试工具 <br/>&#8226;ais_bench：使用AISBench作为测试工具<br/>默认值为`ais_bench`。<br/>用户需自行选择适配的推理框架以及测试框架。|
|-c或--config|可选|指定自定义配置文件路径（TOML格式）。支持以下三种形式：<br>&#8226;绝对路径：直接使用指定路径；<br>&#8226;相对路径（含目录分隔符）：相对于当前工作目录解析；<br>&#8226;仅文件名：在当前工作目录下查找。<br/>默认不指定，工具按预设路径顺序自动搜索配置文件。<br/>指定文件必须为有效 TOML 格式，且具有最高配置优先级。|
|-lb或--load_breakpoint|可选|控制是否从断点恢复寻优过程，配置本参数表示开启，默认未配置表示关闭。|
|--backup|可选|决定是否在寻优过程中备份数据，配置本参数表示开启备份，可取值：<br>&#8226;True：开启备份<br>&#8226;False：不开启备份。<br/>默认值为`False`。|

### 使用示例

<details open>
<summary>vLLM</summary>

vLLM场景下使用ais_bench测评工具可参考

```bash
msmodeling optix -e vllm
```

vLLM场景下使用vllm_benchmark测评工具可参考

```bash
msmodeling optix -e vllm -b vllm_benchmark
```

</details>

<details>
<summary>MindIE</summary>

mindIE场景下使用ais_bench测评工具可参考

```bash
msmodeling optix -e mindie
```

</details>

**指定自定义配置文件**：

```bash
# 绝对路径
msmodeling optix -c /data/configs/my_config.toml

# 相对路径
msmodeling optix -e vllm -b vllm_benchmark -c ../configs/vllm_config.toml

# 当前目录下的文件名
msmodeling optix -c my_config.toml
```

> 如需设置环境变量作用于 vLLM/MindIE 服务，只需在运行工具前设置即可（如 `export ASCEND_RT_VISIBLE_DEVICES=0`），工具会在寻优过程中自动设置。

## 结果文件说明

输出 CSV 中的每一行对应一组参数，前几列为性能指标。用户可以根据需求筛选满足要求的性能行，将 VLLM/MindIE 参数以及 vllm_benchmark/AISBench 的参数改为 CSV 中的数据即可。

|字段|说明|
|---|---|
|`generate_speed`|吞吐。|
|`time_to_first_token`|TTFT 时延，单位为秒。|
|`time_per_output_token`|TPOT 时延，单位为秒。|
|`success_rate`|测试返回请求成功率。|
|`throughput`|测试吞吐，单位为请求数/秒。|
|`CONCURRENCY`|并发数。|
|`REQUESTRATE`|发送速率。|
|`error`|记录这次参数没有正常执行的原因，在发送错误时记录。|
|`backup`|数据记录地址，当开启 `--backup` 时记录。|
|`real_evaluation`|标记数据是否由真实测试结果得到。`false` 代表该组数据由 GP 模型预测得到。|
|`fitness`|寻优算法优化值，该值越小代表该组参数效果越好。|
|`num_prompts`|记录这次寻优测评工具发送的请求数。|

其余列为对应的 VLLM 或 MindIE 的 `config.toml` 参数。

## 附录

### 环境变量与排障

**环境变量**

| 变量 | 说明 |
|------|------|
| `OPTIX_DEPLOY_PATH` | 可选。部署环境根目录，其 `bin/` 下应有 `vllm`、`ais_bench` 等。优先级高于 `config.toml` 的 `[deploy] path_prefix`；不设则用系统 PATH |
| `MIES_INSTALL_PATH` | MindIE 安装根目录，子进程会保留，不用为隔离而改 |

优先级从高到低：`OPTIX_DEPLOY_PATH`、`config.toml` 的 `[deploy] path_prefix`、仅剥离 msmodeling venv 后走系统 PATH。

**日常启动**

系统里已经装好 vLLM 时，通常不用设 `OPTIX_DEPLOY_PATH`：

```bash
source /path/to/msmodeling/.venv/bin/activate
msmodeling optix -e vllm -b ais_bench
```

PATH 布局特殊时再设：

```bash
export OPTIX_DEPLOY_PATH=/path/to/custom-deploy-root
source /path/to/msmodeling/.venv/bin/activate
msmodeling optix -e vllm -b ais_bench
```

**`[optix/env]` 日志对照**

| 日志 | 含义 | 处理 |
|------|------|------|
| `当前未检测到虚拟环境` | 没用 venv 装 msmodeling | 在仓库根目录执行 `uv sync`（会自动创建 `.venv`）；别装到系统 Python，否则 `torch`、`transformers` 会冲掉部署栈 |
| `找不到部署命令：vllm` 或 `mindieservice_daemon` | 剥离 venv 后系统 PATH 里没有命令 | 先确认系统已装 vLLM 或 MindIE；必要时设 `OPTIX_DEPLOY_PATH` 或 `[deploy] path_prefix` |
| `命令 vllm 解析到 msmodeling 虚拟环境` | msmodeling venv 里误装了 vllm | 在该 venv 里 `pip uninstall vllm`，改用系统里的 vLLM |
| `部署命令 vllm → ...` 且路径在系统侧 | 正常 | 不用改 |

### 日志级别与查看

寻优工具使用 [loguru](https://github.com/Delgan/loguru) 输出结构化日志。控制台每行包含 `run_id`、`stage` 等上下文字段。请在启动工具**之前**设置日志级别：

```bash
export OPTIX_LOG_LEVEL=DEBUG

```

|级别|可见内容|
|---|---|
|`INFO`（默认）|里程碑：baseline 通过、服务就绪、迭代摘要、最优结果；子进程启动为多行 `command:` / `log:`|
|`DEBUG`|参数与配置细节、子进程 I/O；每行含 `file:line`；未捕获异常含完整堆栈|
|`TRACE`|PSO 粒子级细节（fitness、粒子位置、单次评测参数）；与 DEBUG 相同含 `file:line`|

VLLM/MindIE 及测评日志写入结果目录或 `/tmp`。启动日志为多行可读格式：

```text
Starting service subprocess
  command: vllm serve model_path --host 127.0.0.1 --port 8080
  log: /tmp/ms_serviceparam_optimizer__abc123
```

baseline 失败时 CLI 边界仅输出一次含 `exit=`、`command:`、`log:` 及日志末尾数行的消息。框架在构造插件前按各类声明的 `required_executable` 检查 `PATH`：`-b` 缺失时抛出 `BenchmarkUnavailableError`，`-e` 缺失时抛出 `SimulatorUnavailableError`，均发生在寻优开始前，不会拉起子进程或清理输出目录。

### 故障排查

|现象|可能原因|处理建议|
|---|---|---|
|退出码 `1`，提示 `No feasible solution found`|baseline 或 PSO 全部候选 fitness 为 `inf`|查看 CSV `error` 列；使用 `OPTIX_LOG_LEVEL=DEBUG`；检查服务与测评命令|
|`Optimizer aborted` 并带堆栈|`main()` 未捕获的致命错误|根据边界单次 traceback 修复配置、路径或健康检查规则|
|`BenchmarkResultError` / AISBench CSV 不唯一|测评输出目录下 0 个或多个 `performances/*/*.csv`|清理输出目录；确保每次评测只产生一份 CSV；立即终止整次寻优|
|控制台仅有 `run_id`、`stage`，细节较少|默认 `INFO` 不打印粒子级日志|设置 `OPTIX_LOG_LEVEL=TRACE` 查看 PSO 内部|
|`BenchmarkUnavailableError` 启动即失败|所选 `-b` 插件声明的 CLI 不在 `PATH`|安装 benchmark CLI 或更换 `-b`；发生在寻优开始前|
|`SimulatorUnavailableError` 启动即失败|所选 `-e` 插件声明的 CLI 不在 `PATH`|安装推理框架 CLI 或更换 `-e`；发生在寻优开始前|
|`BaselineRunError` 含 `exit=` / `log:`|baseline 子进程失败|先看控制台末尾日志；需要时再打开完整 log 文件|
|`--config` 指向不存在文件|路径错误或文件未部署|检查路径；抛出 `ConfigFileNotFoundError`（退出码 `1`）|

### CLI 退出码

|退出码|含义|
|---|---|
|`0`|找到可行最优解并完成输出|
|`1`|`OptimizerError` 子类（`ConfigFileNotFoundError`、`BenchmarkResultError`、`NoFeasibleSolutionError`、`BaselineRunError` 等）或未捕获致命错误|

> 当前所有失败路径均以退出码 `1` 退出；请通过日志消息或 `OptimizerError` 子类区分失败类型，而非依赖不同非零退出码。非法 TOML 抛出 `InvalidConfigError`。退出码非零时，请结合控制台日志与 CSV `error` 列排查。

### 三元派生类型使用示例

#### 场景一：dp 由总卡数自动推导

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
dtype_param = {
  target_names = ["tp", "pp"],
  product = 32,
  dtype = "int",
  priority_policy = "balanced"   # 默认即为 balanced，可不写
}
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
dtype_param = {
  target_names = ["tp", "pp"],
  product = 32,
  dtype = "int",
  priority_policy = "fixed",
  priority = ["tp", "pp"]        # tp 高优先：尽量保留 tp，首先调整 pp
}
# 示例： tp=8、pp=3（非法）：
#   stage1：固定 tp=8，在 pp 候选中找最近合法値 → pp=4， dp=1
#   stage1 失败时再 stage2：两个字段均可调整，按距离升序搜索
```

> [!note] priority_policy 说明
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
