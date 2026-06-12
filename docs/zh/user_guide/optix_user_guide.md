# 服务化自动寻优工具

## 简介

**服务化自动寻优工具**（msmodeling optix）是一个基于PSO粒子寻优算法的服务化参数自动寻优工具，支持对 `VLLM` 和 `MindIE` 进行自动寻优，获取符合时延要求的最佳吞吐参数组合。

## 适用对象与阅读路径

本文适用于需要对 vLLM、MindIE 服务化部署参数进行自动寻优的性能工程师和部署工程师。建议按以下顺序阅读：

1. 先阅读“使用前准备”和“工具安装”，确认服务框架与测评工具可正常运行。
2. 再阅读“快速入门”和“命令参数说明”，完成一次默认寻优。
3. 最后阅读“配置文件说明”和“输出结果文件说明”，根据业务 SLO 调整搜索空间并筛选推荐参数。

工具主要包括两大核心功能模块：

- **参数寻优模块**：利用PSO粒子寻优算法自动生成服务化参数组合，不断逼近最优解；同时，Early Rejection算法通过理论建模、调优经验及部分实测数据对服务化参数完成早期评估；

- **参数验证模块**：自动化启动服务化进程与测评工具进程，进行参数测试，获取性能结果。当前已支持的测评工具包括 `AISBench`、`vllm_benchmark`。

> [!NOTE]
>
> 旧版 benchmark 工具将逐步由 AISBench 替代，推荐优先使用 AISBench。若当前环境仍保留 `vllm_benchmark` 适配能力，可按本文对应章节配置。

服务化自动寻优工具能够基于以上功能模块，自动推荐吞吐较优的服务化参数组合。

目前工具已基于llama3-8b和qwen3-8b通过验证，理论上不限制支持模型范围，后续计划扩大支持范围的验证。

**基本概念**

- `VLLM`、`MindIE`：服务化框架，支持对模型进行服务化部署。
- `vllm_benchmark`、`AISBench`：推理性能评测工具，支持对服务化进行推理性能评测。

## 产品支持情况<a name="ZH-CN_TOPIC_0000002479925980"></a>

> [!NOTE]
>
>昇腾产品的具体型号，请参见《[昇腾产品形态说明](https://www.hiascend.com/document/detail/zh/AscendFAQ/ProduTech/productform/hardwaredesc_0001.html)》。

|产品类型| 是否支持 |
|--|:----:|
|Atlas 350 加速卡|x|
|Atlas A3 训练系列产品/Atlas A3 推理系列产品|√|
|Atlas A2 训练系列产品/Atlas A2 推理系列产品|√|
|Atlas 200I/500 A2 推理产品|√|
|Atlas 推理系列产品|√|
|Atlas 训练系列产品|x|

> [!NOTE]
>
>针对Atlas A2 训练系列产品/Atlas A2 推理系列产品，当前仅支持该系列产品中的Atlas 800I A2 推理服务器。
>针对Atlas 推理系列产品，当前仅支持该系列产品中的Atlas 300I Duo 推理卡+Atlas 800 推理服务器（型号：3000）。

## 使用前准备

**环境准备**
准备好能正常运行服务化（如[VLLM Server](https://docs.vllm.ai/projects/ascend/en/latest/quick_start.html)/[MindIE Service](https://gitcode.com/Ascend/MindIE-Motor/blob/master/docs/zh/user_guide/quick_start.md)）和测评工具（如`vllm_benchmark/AISBench`，参见[测评工具部署](https://gitee.com/aisbench/benchmark/blob/master/README.md)）的环境。

## 工具安装

寻优工具是 msmodeling 项目的子工具。安装时请进入包含 OptiX 源码与 `pyproject.toml` 的目录后执行可编辑安装：

```bash
git clone https://gitcode.com/Ascend/msmodeling.git # 如已拉取，则不用重复拉取
cd <optix_source_dir>
pip install -e .  # 安装寻优工具
```

> [!NOTE]
> 若当前分支未包含 OptiX 源码目录，请切换到包含 OptiX 代码的发布分支或使用对应发布包；仅复制文档文件无法提供 `msmodeling optix` 命令。

## 工具卸载

```bash
pip uninstall optix
```

## 快速入门

1. 完成[使用前准备](#使用前准备)章节要求。

2. 修改配置文件：启动寻优前需用户按照实际情况配置 `config.toml`，包括寻优参数、测评工具参数、服务化参数。参考[配置文件说明](#配置文件说明)章节完成配置。也可通过 `-c` 参数将配置文件放在任意路径，具体见[命令参数说明](#命令参数说明)。

3. 启动寻优：完成上述步骤后，执行以下命令，一键启动自动寻优：

    ```bash
    msmodeling optix
    ```

    默认执行的是基于`AISBench`的`MindIE`服务化参数寻优。

4. 查看结果：寻优时间由模型大小和数据集大小决定，一般在4~8小时完成，结束后会生成`data_storage_*.csv`的文件并保存在当前目录的`result/store`子目录中，其中记录了各组参数的性能，详细介绍请参见[输出结果文件说明](#输出结果文件说明)。

## 命令参数说明

**功能说明**

工具结合参数验证、参数寻优模块，通过真机实测给出可靠的服务化参数推荐值。

**注意事项**

- 启动寻优前需保证所选服务框架（`vllm` 或 `mindie`）和测评工具（`ais_bench` 或 `vllm_benchmark`）可独立运行。
- `config.toml` 中的模型路径、端口、数据集路径和服务启动参数需与实际部署环境保持一致。
- 自动寻优会反复拉起服务并执行测评，耗时通常较长，建议在独占或资源稳定的环境中运行。

**命令格式**

```bash
msmodeling optix [options]
```

**参数说明**

|参数|可选/必选|说明|
|---|---|---|
|-lb或--load_breakpoint|可选|控制是否从断点恢复寻优过程，配置本参数表示开启，默认未配置表示关闭。|
|-d或--deploy_policy|可选|设置部署策略，即单机或多机部署，可取值：<br>&#8226;single：单机部署<br>&#8226;multiple：多机部署。<br/>默认值为`single`。|
|--backup|可选|决定是否在寻优过程中备份数据，配置本参数表示开启备份，可取值：<br>&#8226;True：开启备份<br>&#8226;False：不开启备份。<br/>默认值为`False`。|
|-b或--benchmark_policy|可选|指定测评工具，可取值：<br>&#8226;vllm_benchmark：使用vllm_benchmark作为测试工具 <br/>&#8226;ais_bench：使用AISBench作为测试工具<br/>默认值为`ais_bench`。<br/>用户需自行选择适配的推理框架以及测试框架。|
|-e或--engine|可选|指定推理框架，可取值：<br>&#8226;vllm：使用VLLM作为推理框架<br>&#8226;mindie：使用MindIE作为推理框架<br/>默认值为`mindie`。|
|-c或--config|可选|指定自定义配置文件路径（TOML格式）。支持以下三种形式：<br>&#8226;绝对路径：直接使用指定路径；<br>&#8226;相对路径（含目录分隔符）：相对于当前工作目录解析；<br>&#8226;仅文件名：在当前工作目录下查找。<br/>默认不指定，工具按预设路径顺序自动搜索配置文件。<br/>指定文件必须为有效 TOML 格式，且具有最高配置优先级。|

**使用示例（vllm服务化参数寻优）**

1. 修改配置文件：启动寻优前需用户按照实际情况配置 `config.toml`，包括寻优参数、测评工具参数、服务化参数。参考[配置文件说明](#配置文件说明)章节完成配置。

2. 如果需要设置环境变量作用于vllm/mindie服务，只需在运行工具前设置环境变量即可，例如：

    ```bash
    export ASCEND_RT_VISIBLE_DEVICES=0
    ```

    工具会在寻优过程中自动设置。

3. 前置条件准备就绪后，执行以下命令，一键启动自动寻优：

    ```bash
    msmodeling optix -e vllm
    ```

    若在VLLM场景下使用`vllm_benchmark`测评工具可参考

    ```bash
    msmodeling optix -e vllm -b vllm_benchmark
    ```

**使用示例（mindie服务化参数寻优）**

1. 修改配置文件：启动寻优前需用户按照实际情况配置 `config.toml`，包括寻优参数、测评工具参数、服务化参数。参考[配置文件说明](#配置文件说明)章节完成配置。
2. 如果需要设置环境变量作用于vllm/mindie服务，只需在运行工具前设置环境变量即可，例如：

    ```bash
    export ASCEND_RT_VISIBLE_DEVICES=0
    ```

    工具会在寻优过程中自动设置。

3. 前置条件准备就绪后，执行以下命令，一键启动自动寻优：

    ```bash
    msmodeling optix
    ```

**使用示例（指定自定义配置文件）**

如果配置文件不在默认搜索路径中，可通过 `-c` 参数显式指定：

```bash
# 绝对路径
msmodeling optix -c /data/configs/my_config.toml

# 当前目录下的文件名
msmodeling optix -c my_config.toml

# 相对路径
msmodeling optix -e vllm -b vllm_benchmark -c ../configs/vllm_config.toml
```

指定的配置文件具有最高优先级，会覆盖默认路径下的同名配置项。

**输出说明**

自动寻优完成后，输出csv格式的结果文件，在当前目录下生成result/store文件夹存放。详情介绍请参见[输出结果文件说明](#输出结果文件说明)。

## 输出结果文件说明

输出csv中的每一行对应一组参数，前四列为性能指标。用户可以根据需求筛选满足要求的性能行，将VLLM/MindIE参数以及vllm_benchmark/AISBench的参数改为csv中的数据即可。

| 字段 | 说明 |
| --- | --- |
| generate_speed | 吞吐。 |
| time_to_first_token | TTFT 时延，单位为秒。 |
| time_per_output_token | TPOT 时延，单位为秒。 |
| success_rate | 测试返回请求成功率。 |
| throughput | 测试吞吐，单位为请求数/秒。 |
| CONCURRENCY | 并发数。 |
| REQUESTRATE | 发送速率。 |
| error | 记录这次参数没有正常执行的原因，在发送错误时记录。 |
| backup | 数据记录地址，当开启--backup时记录。 |
| real_evaluation | 标记数据是否由真实测试结果得到。false代表该组数据由gp模型预测得到。 |
| fitness | 寻优算法优化值，该值越小代表该组参数效果越好 |
| num_prompts | 记录这次寻优测评工具发送的请求数。 |

其余列为对应的VLLM或MindIE的config.toml参数。

## 附录

### 配置文件说明

**寻优参数**： `n_particles` （寻优种子数）、`iters` （迭代轮次数）、 `tpot_slo` （`time_per_output_token`的限制时延）等。
用户可根据预估时间来自行配置种子和迭代次数。我们单个种子使用时间为拉起服务+测试数据。比如用户拉起服务+完成测试需9-10min，且愿意用8小时来进行寻优，则一共可跑约50个种子，建议用户配置5 * 10。设置种子数为10，迭代次数为5，建议用户配置种子数为迭代次数的2倍左右。

> **注意**：以下寻优参数均为必填项，不可删除或省略，否则运行时会报错。

|参数|可选/必选|说明|
|---|---|---|
|n_particles|必选|寻优种子数，即一组生成的参数组合数，取值范围为：1-1000的整数。建议设为 15 ~ 30。 |
|iters|必选|迭代轮次数，取值范围为：1-1000的整数。建议设为 5 ~ 10。 |
|ttft_penalty|必选|`time_to_first_token` 即首token时延超时惩罚系数，若对 `time_to_first_token` 没有时延要求设置为0即可。取值范围：【0, 100】。建议设为1。|
|tpot_penalty|必选|`time_per_output_token` 即非首token时延超时惩罚系数，若对`time_per_output_token`没有时延要求设置为0即可。取值范围：【0, 100】。建议设为1。|
|success_rate_penalty|必选|请求成功率惩罚系数，取值范围为：1-1000的整数。建议设为5。 |
|ttft_slo|必选|`time_to_first_token`的限制时延。如对`time_to_first_token`限制为2s内，则设为2，取值范围：(0, 100]，单位s。|
|tpot_slo|必选|`time_per_output_token`的限制时延。如对`time_per_output_token`限制为50ms内，则设为0.05，取值范围：(0, 100]，单位s。 |
|service|必选|标注多机启动时为主机或从机，多机场景下从机设为 `slave`，可取值：<br>&#8226;master：主机<br/>&#8226;slave：从机，<br/>默认值为`master`。|
|sample_size|可选|对原始数据集采样大小，用采样后的数据进行调优，可增加寻优效率，取值范围为：1000-10000的整数，建议设为原数据集请求的1 / 3。|

**测评工具参数**：
若使用`AISBench`测评，需修改以下参数，可以参照[AISBench 使用说明](https://gitee.com/aisbench/benchmark/blob/master/README.md)进行修改。

|参数|说明|
|---|---|
|models| 指定模型任务，可根据[模型配置说明](https://gitee.com/aisbench/benchmark/blob/master/doc/users_guide/models.md)进行配置。|
|datasets| 指定数据集任务，可根据[数据集准备指南](https://gitee.com/aisbench/benchmark/blob/master/doc/users_guide/datasets.md)进行配置 |
|mode| 运行模式。可根据[运行模式说明](https://gitee.com/aisbench/benchmark/blob/master/doc/users_guide/mode.md)进行配置。|
|num_prompts| 控制运行数据集的条数，`mode`为`perf`时有效。|

若使用`vllm_benchmark`测评，需修改以下参数：

|参数|可选/必选|说明|
|---|---|---|
|host|必选| 主机ip，需与`[vllm.command]`中的`host`保持一致，可设为`127.0.0.1`。|
|port|必选| 端口号，需与`[vllm.command]`中的`port`保持一致。|
|model|必选| 模型路径，需与`[vllm.command]`中的`model`保持一致。|
|served_model_name|必选| 模型名称，需与`[vllm.command]`中的`served_model_name`保持一致。|
|dataset_name|必选| 数据集名称。|
|dataset_path|必选| 数据集路径。|
|num_prompts|必选| 控制运行数据集的条数。<br>取值范围：1-10000的整数。|
|others|可选| 拼接其他参数，注意参数间使用空格分隔，参数内部不能留有空格。如`--ignore-eos --custom-output-len 1500`。默认为空。|

**VLLM服务化参数**：
使用VLLM框架时，需修改`config.toml`中的`[vllm.command]`参数，如：

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
|host|必选| 主机ip，需与`[vllm_benchmark.command]`中的`host`保持一致，可设为`127.0.0.1`。|
|port|必选| 端口号，需与`[vllm_benchmark.command]`中的`port`保持一致。|
|model|必选| 模型路径，需与`[vllm_benchmark.command]`中的`model`保持一致。|
|served_model_name|必选| 模型名称，需与`[vllm_benchmark.command]`中的`served_model_name`保持一致。|
|others|可选| 拼接其他参数，注意参数间使用空格分隔，参数内部不能留有空格。如：`--tensor-parallel-size 2 --no-enable-prefix-caching`。默认为空。|

### VLLM自定义参数寻优

寻优工具支持通过 `[[vllm.target_field]]` 添加任意 VLLM 启动参数参与寻优。配置方式分为两步：**声明寻优字段** + **在 `others` 中引用变量**。

> **变量引用规则**：在 `others` 中使用 `$字段名大写` 的格式引用寻优字段，工具运行时会自动将其替换为当前迭代的实际值。

#### 示例一：枚举数值参数（以 `gpu_memory_utilization` 为例）

**第一步**：声明寻优字段。

```toml
[[vllm.target_field]]
name = "GPU_MEMORY_UTILIZATION"
config_position = "env"
dtype = "enum"
dtype_param = [0.9, 0.91, 0.92]
value = 0.9
```

**第二步**：在 `[vllm.command]` 的 `others` 中引用变量。

```toml
[vllm.command]
# ... 其他必填参数 ...
others = "--gpu-memory-utilization $GPU_MEMORY_UTILIZATION"
```

#### 示例二：开关型/复合字符串参数（以编译配置 `--compilation-config` 为例）

当参数本身是一段完整的 CLI 字符串时，可将"不启用"（空字符串 `""`）和"启用"两种形态作为枚举候选值。工具遇到空字符串时会自动跳过，不向启动命令追加任何内容。

**第一步**：声明寻优字段。

> **注意**：TOML 字符串使用双引号 `"` 作为边界符，若字符串内容中包含双引号，需使用 `\"` 转义，否则会解析报错。

```toml
[[vllm.target_field]]
name = "COMPILATION_CONFIG"
config_position = "env"
dtype = "enum"
dtype_param = ["", "--compilation-config '{\"cudagraph_mode\": \"FULL_DECODE_ONLY\"}'"]
value = "--compilation-config '{\"cudagraph_mode\": \"FULL_DECODE_ONLY\"}'"
```

**第二步**：在 `[vllm.command]` 的 `others` 中引用变量。

```toml
[vllm.command]
# ... 其他必填参数 ...
others = "$COMPILATION_CONFIG"
```

**MindIE服务化参数**： 可以参考[MindIE server 配置参数说明](https://www.hiascend.com/document/detail/zh/mindie/20RC1/mindieservice/servicedev/mindie_service0285.html)进行修改。
服务化参数可直接指定参数的范围，如配置服务化参数 `max_batch_size` 的寻优搜索空间为 10 ~ 400，则可设置：

```toml
[[mindie.target_field]]
"name": "max_batch_size",    # 服务化参数名称
"config_position": "BackendConfig.ScheduleConfig.maxBatchSize",    # 服务化参数在MindIE Server中的位置
"min": 10,    # 最小值
"max": 400,    # 最大值
"dtype": "int"    # 数据类型
```

此外，也可设置参数与另一参数相关，如 `max_prefill_batch_size` 与 `max_batch_size` 相关，`max_prefill_batch_size = ratio * max_batch_size (0 < ratio < 1)`则可设置：

```toml
[[mindie.target_field]]
"name": "max_prefill_batch_size",
"config_position": "BackendConfig.ScheduleConfig.maxPrefillBatchSize",
"min": 0,
"max": 1,
"dtype": "ratio",
"dtype_param": "max_batch_size"    # 表明该参数与max_batch_size相关
```

此外，`target_field` 支持的所有 `dtype` 类型如下：

| 分类 | dtype | 含义 | dtype_param 格式 |
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

> [!note] 说明
>
> 派生类型字段（`factories` / `times` / `ternary_factories` / `ternary_times`）的值由依赖关系自动推导，**不参与粒子群搜索**，需将 `min` 和 `max` 均设为 `0`。若任一依赖字段值为 `0`（除法场景）或 `None`/`NaN`（乘法场景），本轮推导跳过，字段保持原值并输出警告日志。

**三元派生类型使用示例**

场景一：`tp`、`pp` 为可调参数，`dp` 由总卡数（16）自动推导（`dp = 16 ÷ (tp × pp)`）：

> [!note] 约束说明
>
> `ternary_factories` 要求各依赖字段的乘积能合法推出派生字段。对于 `dtype = "int"`，`product` 必须能被依赖字段乘积整除，否则会触发优先级修复。
>
> - **int 类型内置保护**：结果不足 1 或不能整除时优先尝试修复源字段；修复失败后按 min/max 降级处理，并输出 WARNING。
> - **显式设置范围**：在 `dtype_param` 中配置 `min_value` / `max_value` 可覆盖上下界。
> - **最佳实践**：限制 `tp`、`pp` 的枚举候选使乘积可整除 `product`，避免依赖降级处理。

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
> - `balanced` 通过将粒子按解码顺序分层，降低单一字段顺序导致的结构性偏置，但不能保证全局最优。
> - `fixed` 适合用户明确知道哪个字段更应该稳定的场景，例如 tp 由硬件资源决定时。
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
# 当 seq_len=1024, prefill_batch_size=4 时，max_prefill_tokens = 2 × 1024 × 4 = 8192
```

**日志检测**：检查日志中出现的异常信息，区分致命错误和可重试错误，实现智能错误处理和重试机制。可检测的错误类型包括内存溢出（OOM）、设备故障（NPU）、网络错误和IO错误等。致命错误（如OOM、NPU故障）会立即停止调度器，可重试错误（如网络抖动、IO失败）会触发自动重试（最多3次）。

|参数|可选/必选|说明|
|---|---|---|
|log_snippet_length|可选|日志片段长度，用于显示错误详情。取值范围：50-1000，默认为200。|
|service_errors.fatal_patterns|可选|服务化框架致命错误模式列表，默认为空。常见致命错误包括内存溢出、设备故障等。|
|service_errors.retryable_patterns|可选|服务化框架可重试错误模式列表，默认为空。常见可重试错误包括网络错误、IO错误等。|
|benchmark_errors.fatal_patterns|可选|测评工具致命错误模式列表，默认为空。|
|benchmark_errors.retryable_patterns|可选|测评工具可重试错误模式列表，默认为空。|

配置示例：

```toml
[health_check]
log_snippet_length = 200

[health_check.service_errors.fatal_patterns]
out_of_memory = ["out of memory", "OOM killed", "MemoryError"]
device_error = ["NPU error", "device fault", "Ascend error"]

[health_check.service_errors.retryable_patterns]
network_error = ["connection reset", "connection refused", "timeout"]
io_error = ["file not found", "permission denied", "IO error"]
```

### 插件模式

现在寻优工具支持用户自定义搜索参数配置以及测试工具，用户可以根据自己的需求配置。只需适配我们的插件模式，注册对应的插件即可，详情请参见[插件开发操作步骤](./optix_plugin_user_guide.md)。

### 日志说明

寻优过程中默认日志为INFO级别，如果用户想看每一轮具体的日志，可以在使用工具前设置

```bash
export MODELEVALSTATE_LEVEL=DEBUG
```

对于每一轮的运行状态会进行输出，我们将具体的VLLM/MindIE日志重定向在/tmp目录下，可以根据打印信息获取具体文件路径查看服务运行状态。
