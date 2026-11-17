# OptiX 环境与故障排查

[使用指南](../msmodeling_optix_user_guide.md) | [插件与模式](plugin_and_modes.md) | [配置参数](configuration.md) | [命令与结果](commands_and_results.md) | [环境与排障](troubleshooting.md)

<a id="10-附录"></a>

## 1. 附录

<a id="环境变量与排障"></a>

### 1.1 环境变量与排障

**环境变量**

| 变量 | 说明 |
|------|------|
| `OPTIX_DEPLOY_PATH` | 可选。部署环境根目录，其 `bin/` 会前置到剥离 msmodeling 虚拟环境后的 PATH；未找到命令时继续搜索原有 PATH。优先于 `config.toml` 的 `[deploy] path_prefix` |
| `MIES_INSTALL_PATH` | MindIE 安装根目录，子进程会保留，不用为隔离而改 |

部署根目录优先使用 `OPTIX_DEPLOY_PATH`，其次使用 `config.toml` 的 `[deploy] path_prefix`。命令先从该目录的 `bin/` 查找，再搜索剥离 msmodeling 虚拟环境后的原有 PATH。

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

<a id="日志级别与查看"></a>

### 1.2 日志级别与查看

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

<a id="故障排查"></a>

### 1.3 故障排查

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
|启用了早停但 Case 未被停止|配置为 `action = "report"`|验证判断结果后改为 `action = "terminate"`|
|日志提示 early exit unsupported|engine 不是 `vllm`|当前仅在 `-e vllm` 下启用早停|
|普通候选一直完整运行|尚无完整 reference、Case 太短或有效窗口数据不足|确认 baseline 已完成，并检查 metrics 采样轨迹|
|baseline、calibration 或 refinement 未早停|这是完整结果保护机制|无需处理|
|`/metrics` 获取失败|`metrics_url`、监听地址或端口不正确|验证 vLLM `/metrics` 地址；单次采集失败不会主动终止 benchmark|

<a id="cli-退出码"></a>

### 1.4 CLI 退出码

|退出码|含义|
|---|---|
|`0`|找到可行最优解并完成输出|
|`1`|`OptimizerError` 子类（`ConfigFileNotFoundError`、`BenchmarkResultError`、`NoFeasibleSolutionError`、`BaselineRunError` 等）或未捕获致命错误|

> 当前所有失败路径均以退出码 `1` 退出；请通过日志消息或 `OptimizerError` 子类区分失败类型，而非依赖不同非零退出码。非法 TOML 抛出 `InvalidConfigError`。退出码非零时，请结合控制台日志与 CSV `error` 列排查。
