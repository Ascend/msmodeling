# OptiX 命令与结果参考

[使用指南](../msmodeling_optix_user_guide.md) | [插件与模式](plugin_and_modes.md) | [配置参数](configuration.md) | [命令与结果](commands_and_results.md) | [环境与排障](troubleshooting.md)

<a id="8-命令参数说明"></a>

## 1. 命令参数说明

<a id="命令格式"></a>

### 1.1 命令格式

```bash
msmodeling optix [options]
```

<a id="注意事项"></a>

### 1.2 注意事项

- 启动寻优前，确认 `vllm` 或 `mindie` 以及 `ais_bench` 或 `vllm_benchmark` 已在系统部署环境里能正常运行，且未部署在 msmodeling 虚拟环境中。
- 配置文件如 `config.toml` 中的模型路径、端口、数据集路径和服务启动参数需与实际部署环境保持一致，请参见[配置文件说明](configuration.md#7-配置文件说明)。
- 服务化实测寻优会反复拉起服务化框架并执行测评，耗时通常较长（4-10h），建议在独占或资源稳定的环境中运行。
- 环境隔离异常时，日志前缀 `[optix/env]` 会给出原因与修复建议，请参见[环境变量与排障](troubleshooting.md#环境变量与排障)。

<a id="参数说明"></a>

### 1.3 参数说明

|参数|可选/必选|说明|
|---|---|---|
|`--mode`|可选|选择寻优流程。`standard` 为默认的通用寻优流程，`pd_disagg` 分别搜索 P、D 参数并推荐配比。|
|-e或--engine|可选|指定推理框架，可取值：<br>&#8226;vllm：使用VLLM作为推理框架<br>&#8226;mindie：使用MindIE作为推理框架<br/>默认值为`vllm`。|
|-b或--benchmark-policy|可选|指定测评工具，可取值：<br>&#8226;`vllm_benchmark`：使用 vllm_benchmark 作为测评工具 <br/>&#8226;`ais_bench`：使用 AISBench 作为测评工具<br/>默认值为`ais_bench`。<br/>用户需自行选择适配的推理框架以及测试框架。|
|-c或--config|可选|指定自定义配置文件路径（TOML格式）。支持以下三种形式：<br>&#8226;绝对路径：直接使用指定路径；<br>&#8226;相对路径（含目录分隔符）：相对于当前工作目录解析；<br>&#8226;仅文件名：在当前工作目录下查找。<br/>默认不指定，工具按预设路径顺序自动搜索配置文件。<br/>指定文件必须为有效 TOML 格式，且具有最高配置优先级。|
|--load-breakpoint|可选|控制是否从断点恢复寻优过程，配置本参数表示开启，默认未配置表示关闭。|
|--backup|可选|决定是否在寻优过程中备份数据。配置本参数表示开启备份，默认关闭。|

<a id="使用示例"></a>

### 1.4 使用示例

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

> 如需设置环境变量作用于 vLLM 服务，优先使用 `[vllm.env]`；如需作用于 MindIE 服务或整个工具进程，可在运行工具前设置（如 `export ASCEND_RT_VISIBLE_DEVICES=0`），工具会在寻优过程中自动继承。

<a id="内置-pd-分离两步寻优"></a>

### 1.5 内置 PD 分离两步寻优

OptiX 支持配置 `pd_disagg.full_service`，通过现有 `vllm_pd` 插件自动串联单侧搜索、配比推荐、完整服务部署和负载微调，见[命令行一键寻优指南](../../quick_start/optix_pso_pd_disaggregation.md#一个命令运行完整流程)。

分步使用时，OptiX 支持为 vLLM 的 PD（Prefill-Decode）分离部署搜索服务参数并推荐 P/D 实例配比：先通过 `--mode pd_disagg` 分别搜索 P/D 参数，再由用户部署完整 PD 服务，最后通过普通 `standard` 模式微调并发和请求速率。

例如，在仓库根目录使用准备好的配置文件启动 P/D 搜索：

```bash
msmodeling optix --mode pd_disagg -e vllm -b ais_bench -c ./pd_search.toml
```

首次使用可按[PD 分离寻优新手操作指南](../../quick_start/optix_pso_pd_disaggregation.md)中的具体例子复制配置并运行。完整参数解释、配比解读和进阶配置见[PD 分离分步执行案例](../../use_cases/optix_pd_step_by_step.md)。其他 PD 分离或 PD 混部寻优可使用[插件支持](plugin_and_modes.md#3-插件支持)中列出的现有插件。

<a id="9-结果文件说明"></a>

## 2. 结果文件说明

输出 CSV 中的每一行对应一组参数。第一列 `case_id` 用于标识 Case，后续六列为性能指标。用户可以根据需求筛选满足要求的性能行，将 VLLM/MindIE 参数以及 vllm_benchmark/AISBench 的参数改为 CSV 中的数据即可。

|字段|说明|
|---|---|
|`case_id`|关联主结果和 metrics 采样轨迹的 Case 标识。|
|`generate_speed(tokens/s)`|输出 token 吞吐，单位为 token/秒。|
|`time_to_first_token(s)`|TTFT 时延，单位为秒。|
|`time_per_output_token(s)`|TPOT 时延，单位为秒。|
|`success_rate`|请求成功率，无量纲比例，取值为 0～1。|
|`throughput(requests/s)`|测试吞吐，单位为请求数/秒。|
|`input_token_throughput(tokens/s)`|输入 token 吞吐，单位为 token/秒。 |
|`CONCURRENCY`|并发数。|
|`REQUESTRATE(requests/s)`|发送速率，单位为请求数/秒。|
|`error`|记录这次参数没有正常执行的原因，在发送错误时记录。|
|`backup`|数据记录地址，当开启 `--backup` 时记录。|
|`real_evaluation`|标记数据是否由真实测试结果得到。`false` 代表该组数据由 GP 模型预测得到。|
|`fitness`|寻优算法优化值，该值越小代表该组参数效果越好。|
|`num_prompts`|记录这次寻优测评工具发送的请求数。|
|`would_early_exit`|是否满足早停条件；`report` 模式下也可能为 `true`。|
|`early_exit`|当前 Case 是否被实际提前终止；仅 `terminate` 模式可能为 `true`。|
|`early_exit_reason`|触发早停的原因。|
|`early_exit_decision_elapsed_seconds(s)`|从 Case 开始到形成早停决定的时间，单位为秒。|
|`estimated_time_saved_seconds(s)`|`report` 模式下估算的可节省时间，单位为秒。|
|`estimated_time_saved_ratio`|估算可节省时间占完整 Case 时长的比例。|
|`result_source`|结果来源；实际早停结果为 `early_exit_metrics`。|
|`usable_as_best`|该结果能否参与最优结果和 reference 选择；仅实际提前终止的结果为 `false`。`report` 模式下满足条件的 Case 会完整运行，该字段仍为 `true`。|

上表未列出的结果列还包括早停内部采样和预热元数据（如 `metrics_window_*` 和 `warmup_*`）；未启用早停时，这些字段为空。随后为对应的 VLLM 或 MindIE 的 `config.toml` 参数。

当 `action = "report"` 时，还会生成 `metrics_samples_*.csv`。可以通过 `case_id` 将采样轨迹与 `data_storage_*.csv` 中的主结果关联。实际早停结果不会参与最优候选、reference 或 refinement 选择。
