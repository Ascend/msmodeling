# OptiX 插件、安装与寻优模式

[使用指南](../msmodeling_optix_user_guide.md) | [插件与模式](plugin_and_modes.md) | [配置参数](configuration.md) | [命令与结果](commands_and_results.md) | [环境与排障](troubleshooting.md)

<a id="3-插件支持"></a>

## 1. 插件支持

服务化实测寻优支持用户自定义搜索参数配置以及测评工具，只需适配插件模式注册对应插件即可。内置插件随 `optix` 包发布，通过 `-e`/`-b` 直接选用；也可通过插件模式接入自定义插件。

**内置插件**：

|插件名称|类型|插件描述|
|---|---|---|
|`vllm`|服务框架（`-e`）|vLLM 服务化框架适配|
|`mindie`|服务框架（`-e`）|MindIE 服务化框架适配|
|`ais_bench`|测评工具（`-b`）|AISBench 测评工具适配|
|`vllm_benchmark`|测评工具（`-b`）|vllm_benchmark 测评工具适配|

**扩展插件**（位于 `contrib/optix/`，按需安装）：

|插件名称|类型|提供者|插件描述|插件目录|
|---|---|---|---|---|
|`custom_vllm`|服务框架（`-e`）|contrib 示例|通过 Docker + SSH 拉起本地与远端 vLLM 容器，演示自定义服务框架接入|`contrib/optix/vllm_msserviceprofiler/`|
|`multihost_infer`|服务框架（`-e`）|小巧灵突击队|vLLM 多机混合部署插件，基于多节点 DP（Data Parallel）+ headless 方案，node 主节点对外提供 API、work 节点仅作工作节点|`contrib/optix/multihost_infer/`|
|`vllm_pd`|服务框架（`-e`）|小巧灵突击队|vLLM PD（Prefill-Decode）分离部署集群插件，通过 SSH 远程管理多节点启停与健康检查|`contrib/optix/vllm_pd_simulator/`|
|`evalscopeperf`|测评工具（`-b`）|小巧灵突击队|封装 `evalscope perf` 命令作为测评工具，演示自定义测评工具接入|`contrib/optix/evalscopeplugin/`|

> 插件通过 Python entry points 注册，自定义插件开发与使用请参见[插件开发操作步骤](../msmodeling_optix_plugin_user_guide.md)。

<a id="4-环境准备与安装"></a>

## 2. 环境准备与安装

<a id="环境隔离原则"></a>

### 2.1 环境隔离原则

确认系统已部署 vLLM/MindIE：例如一个vLLM服务容器，容器内安装服务化实测寻优 optix 工具，推荐装在虚拟环境里。在仓库根目录执行 `uv sync` 即可自动创建 `.venv` 并完成安装。

安装 msmodeling 会同时装上 `torch`、`transformers` 等包，如果在系统 Python 里安装 msModeling，可能会与系统里原有的 `torch`、`transformers` 版本冲突，导致：

- vLLM、MindIE 起不来或推理报错
- 和 Ascend 推理栈上已验证的版本对不上

<a id="安装步骤"></a>

### 2.2 安装步骤

```bash
git clone https://gitcode.com/Ascend/msmodeling.git   # 如已拉取，则不用重复拉取
cd msmodeling
uv sync
```

> [!WARNING]
> 不要在 msmodeling 虚拟环境里 `pip install vllm`、`mindie_llm` 等部署包，也不要在未建 venv 的系统 Python 里安装 msmodeling，若有疑问，可参考[推荐实践：服务化实测寻优环境与部署栈](../../install_guide/msmodeling_optix_env_and_deployment_stack.md)

<a id="验证安装"></a>

### 2.3 验证安装

```bash
uv run msmodeling optix --help
```

能正常打印 `msmodeling optix` 帮助信息即表示工具安装成功；部署栈检查与正式寻优见[快速入门](#5-快速入门)。

<a id="卸载"></a>

### 2.4 卸载

```bash
uv pip uninstall msmodeling
```

<a id="5-快速入门"></a>

## 3. 快速入门

单机标准寻优的完整操作请参见[单机标准寻优指南](../../quick_start/optix_single_node.md)。

1. **修改配置文件**：完成[环境准备与安装](#4-环境准备与安装)，按实际情况修改配置文件 `config.toml`，包括寻优参数、测评工具参数、服务化参数。请参见 [配置文件说明](configuration.md#7-配置文件说明)。

   > **说明：** 若不确定该调哪些参数及参数范围，可使用 **`optix-assistant` 寻优 skill**（统一入口，AGENTS.md 中已路由）的 PSO 模式，输入硬件、模型、负载和优化目标后即可获得推荐参数与搜索范围，再按实际部署环境（显存、卡数、时延要求等）核对并由其写入 `config.toml`。

2. **启动寻优**：默认执行的是基于 `AISBench` 的 `vLLM` 服务化参数寻优（PSO 模式），其他用法请参见 [命令参数说明](commands_and_results.md#8-命令参数说明)；如需改用 Agent 模式，见[寻优模式：PSO 与 Agent](#6-寻优模式pso-与-agent)。

    ```bash
    msmodeling optix
    ```

3. **查看结果**：寻优时间由模型大小、数据集大小和寻优次数等决定，结束后会生成 `data_storage_*.csv` 文件并保存在**当前工作目录下的 `result/store`** 子目录中（可用 `[data_storage].store_dir` 修改），请参见 [结果文件说明](commands_and_results.md#9-结果文件说明)。

<a id="6-寻优模式pso-与-agent"></a>

## 4. 寻优模式：PSO 与 Agent

服务化实测寻优提供两种寻优模式，通过配置文件顶层 `optimizer_strategy` 字段选择，默认为 `pso`：

|寻优模式|配置取值|寻优方式|特点|适用场景|
|---|---|---|---|---|
|PSO 模式（默认）|`optimizer_strategy = "pso"`|由工具内置的 PSO 粒子寻优算法自动生成参数组合并迭代寻优|全自动运行，无需人工介入，搜索规模由 `n_particles` 与 `iters` 直接控制|常规自动寻优，以及无人值守的长时间寻优|
|Agent 模式|`optimizer_strategy = "agent"`|由 AI agent 逐轮生成候选参数，工具负责候选校验、执行实测与结果写回，agent 依据每轮结果决策下一轮方向|每个候选带有可解释的选择依据，可吸收历史经验与失败信息，支持人工确认或修改候选|需要结合调优经验进行方向决策，或希望人工参与候选评审的场景|

两种模式共用[配置文件说明](configuration.md#7-配置文件说明)中的寻优参数、测评工具参数和服务化参数，差异主要体现在搜索控制方式上：

- PSO 模式由 `n_particles` 与 `iters` 控制搜索规模；Agent 模式由 `[agent_optimizer]` 配置节控制轮数、单轮 trial 数量与时间预算。
- Agent 模式要求 `use_request_rate_calibration = false`，此时 `CONCURRENCY` 参与搜索、`REQUESTRATE` 固定为最大施压速率，以便逐 trial 观测并发与时延的关系；若保持默认 `true`，候选中的 `CONCURRENCY` 会被固定为 `max`，并发搜索维度失效。

<a id="agent-模式工作方式"></a>

### 4.1 Agent 模式工作方式

PD 分离 Agent 模式的安装与使用步骤见[从环境准备到 Agent 驱动的 PD 分离寻优](../../quick_start/optix_agent_pd_disaggregation.md)。

Agent 模式下，工具不在内部运行 PSO，而是从 `[agent_optimizer].run_dir` 对应的运行目录读取 AI agent 生成的候选文件 `candidates.round-N.json`，校验通过后逐个执行实测，并写回 `results.round-N.json` 与 `progress.round-N.json`；仅当下一轮候选文件尚未生成时，本轮执行完成即退出。收敛退出需同时满足趋势条件（连续 `convergence_rounds` 轮改善幅度低于 1%）与覆盖条件（已完成轮次不少于 `min_rounds`、已完成 trial 不少于 `min_trials`），并受 `max_rounds` 轮数上限约束，每轮 trial 执行受 `time_limit_minutes` 时长限制。

`[agent_optimizer]` 配置节字段说明（均可选）：

|参数|默认值|说明|
|---|---|---|
|`run_id`|`"default"`|寻优运行标识，候选与结果文件按该标识组织于运行目录中。|
|`run_dir`|空|运行目录，留空时自动解析为 `.agent_optimizer/runs/{run_id}`。|
|`max_rounds`|`32`|最大寻优轮数，达到后自然终止。|
|`max_trials`|`12`|单轮执行的 trial 数量上限，超出部分不执行。|
|`time_limit_minutes`|`1440`|单轮执行 trial 的时长上限（分钟），默认 1440；全局预算由 agent 跨轮累计。|
|`convergence_rounds`|`6`|触发收敛检测所需的连续无显著改善轮数。|
|`min_rounds`|`4`|收敛前至少完成的寻优轮数。|
|`min_trials`|`4`|收敛前至少完成的 trial 数。Agent 模式每轮 1 个 trial，与 `min_rounds` 等价，取较大者生效。|

推荐通过 **`optix-assistant` 寻优 skill** 使用 Agent 模式。该 skill 提供配置预检（config_preflight）、上下文采集（collect_context）、配置参数检查（check_search_space）与最优配置导出（export_best_config）等配套脚本，并对候选生成、失败分析与判优规则遵循统一的行为契约。典型使用流程：

1. 按[环境准备与安装](#4-环境准备与安装)完成安装，并确认部署栈可用；
2. 安装并启动 AI Agent 工具，推荐使用 [MindStudioAgent](https://gitcode.com/Ascend/msagent)，或按 [OpenCode 官方文档](https://dev.opencode.ai/v2/docs)安装 OpenCode 并配置模型服务。进入 msmodeling 仓库目录后，在 Agent 对话中输入“请读取本仓库的 `.agents/skills/optix-assistant/SKILL.md`，使用 optix-assistant 进行 Agent 模式寻优”，然后按提示选择全自动或交互式执行方式；
3. 完成配置预检与上下文采集后，agent 按目标路由读取方法卡；基础配置通过 `config_writer` 登记到 TOML，每轮更新 `context.search_space`，候选 `params` 只写取值。校验后执行实测；每轮进程退出后，将 context.json 复制为 context.round-N.json，再准备下一轮。收敛后按最佳结果所属轮次的 context 归档导出配置，归档缺失或损坏时拒绝导出。
