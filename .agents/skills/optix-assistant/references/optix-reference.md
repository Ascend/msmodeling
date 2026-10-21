# OptiX 参考手册（agent 前置知识）

读完后应能不读 optix 源码就正确驱动它。事实来源：msmodeling 仓库
`optix/` 代码与 `docs/zh/user_guide/msmodeling_optix_user_guide.md`。

本手册只承载上游文档未覆盖的参考性内容；流程与判据的归属：
run 目录协议与 progress 状态机、失败分类判据见
`../workflows/execution-validation.md`，候选生成与判优契约见
`../workflows/data-analysis.md`，benchmark 配置见 `benchmark.md`。

## CLI

```bash
msmodeling optix -e <vllm|mindie> -b <ais_bench|vllm_benchmark> -c <config.toml> [-lb] [--backup]
```

- `-e` 引擎、`-b` benchmark、`-c` 配置文件路径（默认 `optix/config.toml`）。
- `-lb/--load_breakpoint` 从上次中断处继续；`--backup` 备份数据。
- agent 寻优**没有独立 CLI 开关**，由 config.toml 的 `optimizer_strategy = "agent"` 驱动。

## config.toml 关键字段

- `optimizer_strategy`：`"agent"` 时 optix 读 run 目录的 `candidates.round-N.json`
  执行候选；`"pso"`（默认）时走内部 PSO、忽略 run 目录候选。agent 寻优必须是 `"agent"`。
- `ttft_slo` / `tpot_slo`（秒）：SLO 判定阈值。
- `[vllm.command]`：`host`、`port`、`model`（模型路径）、`served_model_name`、`others`
  （固定 CLI flag 字符串，如 `"--max-model-len 2560 --trust-remote-code"`）。
- `[[vllm.target_field]]`：搜索空间字段定义，字段：`name`、`config_position`（`env`）、
  `min`/`max`、`dtype`（`int`/`float`/`ratio`/`range`/`bool`）、`value`（当前值）。
- `[agent_optimizer]`：`run_id`、`run_dir`（空 = 自动 `.agent_optimizer/runs/{run_id}`）、
  `max_rounds`、`max_trials`、`time_limit_minutes`
  （单轮 trial 时间上限）、`convergence_rounds`、`min_rounds`、`min_trials`。
- `[health_check]`：serve/benchmark 日志的 fatal/retryable 错误模式表，控制重试策略。

## 参数命名与映射规则（重要，写错会生成非法 serve 命令）

候选/search_space 里写的是**短参数名**，optix 负责映射成真实 vLLM flag：

- 并行参数：`tp`→`--tensor-parallel-size`，`dp`→`--data-parallel-size`，
  `pp`→`--pipeline-parallel-size`，`pcp`→`--prefill-context-parallel-size`。其余参数名按"小写+下划线转连字符"映射
  （如 `max_num_batched_tokens`→`--max-num-batched-tokens`）。
- **容器 flag 的子键参数**（不要自己拼 JSON）：
  - `cudagraph_mode`、`cudagraph_capture_sizes` → 进 `--compilation-config '{...}'`
  - `num_speculative_tokens` → 进 `--speculative-config '{...}'`；
    投机容器内的 eager 用独立名 `speculative_enforce_eager`（与主模型顶层
    `--enforce-eager` 语义不同，勿混用）
  - 正确写法：候选里直接写 `num_speculative_tokens: 3`，optix 自动包装；
    **错误写法**：把 `'--speculative-config {...}'` 整串当作某个参数的值
    （会被当成普通值再拼一次 flag 名，产出非法命令）。
- `CONCURRENCY`、`REQUESTRATE` 是 benchmark 专用字段（渲染为 `vllm bench serve` 的
  `--max-concurrency` / `--request-rate`），永远不会进 serve 命令。
- bool 参数：`true` → 只出 flag（`--flag`）；显式 `false` → 不输出该 flag，并移除 `others` 中同名开关（单机、多机及 PSO 共用）；候选省略字段仍按有效字段定义解析，不能用省略代替关闭。
- 固定不变的启动参数（模型路径、`--max-model-len`、`--trust-remote-code` 等）写进
  `[vllm.command].others`，不进搜索空间。

## 候选 → 真实命令的物化链路

`strategies.py::_params_to_target_field` → `custom_command.py::VllmCommand/VllmBenchmarkCommand`
→ `custom_process.py::_prepare_run_params`：

1. 候选 `params` 先过 `validate_candidate`（校验规则见 `../workflows/data-analysis.md`「候选校验规则」），
   非法候选直接跳过、不消耗 trial。
2. 参数落到 `OptimizerConfigField`：config.toml 已有字段保留类型、位置和派生关系，可调域采用当轮 search_space 定义，再走 `apply_value`（按 dtype 转换）；
   不在 config.toml 里的参数按 run 目录 `context.json` 的 search_space schema 动态建字段
   （list/dict 值先 JSON 序列化成字符串再进字段）。
3. serve 命令渲染（`VllmCommand`）：`config_position="run"` 的字段渲染为 CLI flag；
   `CONCURRENCY`/`REQUESTRATE`（`BENCHMARK_ONLY_FIELDS`）被跳过，绝不进 serve 命令。
4. benchmark 命令渲染（`VllmBenchmarkCommand`）：argv 固定含字面占位 token
   `--max-concurrency $CONCURRENCY --request-rate $REQUESTRATE`；启动前
   `_prepare_run_params` 把 `$CONCURRENCY`/`$REQUESTRATE` 元素替换为本 trial 的字段值；
   `REQUESTRATE` 值 ≤0 时连同 `--request-rate` flag 一起从命令中删除。
5. `others` 字符串内的 `$VAR` 占位符按变量名边界做正则替换（可嵌在 JSON 值内，
   如 `--speculative-config '{"num_speculative_tokens": $NUM_SPECULATIVE_TOKENS,...}'`）。
6. serve 命令去重与合并：容器 flag（`--speculative-config`/`--compilation-config`）多处出现
   合并为一个（`others` 为底座、搜索侧键值覆盖）；普通 flag 重复时后出现的值生效；
   `--speculative-config` 合并后若 `num_speculative_tokens==0` 或缺 `method`/`model`，
   整个容器被丢弃（= 关闭投机，vLLM 对孤儿容器会启动失败）。
7. argv[0] 为 `vllm`/`ais_bench` 时，`materialize_command` 按部署 PATH 解析成绝对路径
   （解析细节见"环境与部署"）。

## 性能指标映射（benchmark 结果 → PerformanceIndex）

**vllm_benchmark**（`plugins/benchmark.py::VllmBenchMark.get_performance_index`）：读取
`[vllm_benchmark.command].result_dir` 下所有 `*.json`（即 `vllm bench serve --save-result`
的输出），逐文件覆盖，最终值来自最后成功解析的文件：

| vllm bench JSON 字段 | PerformanceIndex 字段 | 换算 |
|---|---|---|
| `output_throughput` | `generate_speed` | 直取（tok/s），缺省 0 |
| `mean_ttft_ms` | `time_to_first_token` | /1000（毫秒→秒），缺省 0 |
| `mean_tpot_ms` | `time_per_output_token` | /1000（毫秒→秒），缺省 0 |
| `completed` ÷ `num_prompts` | `success_rate` | 比值；`num_prompts` ≤0 时为 0 |
| `request_throughput` | `throughput` | 直取（req/s），字段缺失时取 3.0 |

- 每个 trial 前 optix 清空 result_dir（`before_run`），防止读到上一轮残留数据；
  JSON 解析失败的文件只记 error 日志并跳过。
- PerformanceIndex 的 TTFT/TPOT **单位是秒**（results/progress 文件里同样如此），
  与 vllm bench JSON 的毫秒字段差 1000 倍，对比原始 JSON 时注意换算。

**ais_bench**（`AisBench.get_performance_index`）：从 `[ais_bench].output_path` 下唯一的
CSV + 同名 JSON 取数。TTFT/TPOT 由 `[ais_bench.performance_config]` 指定 CSV 指标名与
算法列（默认 `TTFT`/`TPOT` + `average` 列），值带 `ms`/`us` 后缀时自动换算为秒；
`throughput` = JSON `Request Throughput.total`，`success_rate` =
`Success Requests`/`Total Requests`，`generate_speed` = `Output Token Throughput.total`。

## 轮次时序与 agent 策略内部行为（实测结论）

optix 在 round-N 执行完时立即读取 `candidates.round-N+1.json`，文件不存在则退出
（日志 "Candidate file not found ... Stopping at round N"），不等分析侧生成下一轮
候选。**逐轮重新拉起一次 `msmodeling optix` 是正常用法**，每轮启动前确保本轮
candidates 文件已就位。

编排器（`agentic/orchestrator.py::AgentOptimizer`）的其他内部行为：

- **resume 扫描**：启动时读各 `results.round-N.json`，只有含 ≥1 个无 error 且
  fitness 有限的 trial 的轮才算"已完成"，恢复时跳过；全败轮会被自动重跑。
- **baseline checkpoint**：`baseline.json` 存在时基线不重跑，直接加载
  （fitness + 5 个性能字段）。
- **轮内执行**（`agentic/strategies.py::AgentCandidateStrategy`）：候选校验 →
  只执行 `valid[:max_trials]`，trial 串行；`time_limit_minutes` 到点中断本轮；
  每 trial 更新 `progress.round-N.json`；轮末写 `results.round-N.json`
  （含 `validation` 节：每个候选的 valid/errors/warnings，被拒候选的拒绝原因在这）。
- **轮最佳选取**：先取 `slo_violated=false` 的 trial 池再按 fitness 最小；
  池为空才退回全部 trial——与"达标优先"判优契约一致。
- **收敛双门**：连续 `convergence_rounds` 轮 improvement ≤1% **且**覆盖达标
  （≥`min_rounds` 轮、≥`min_trials` 个 trial）才提前停；只满足趋势会继续跑。
- **并发/请求率钉死**（`optimizer.py::adapter_target_field`）：REQUESTRATE 两种模式
  都恒钉在 max；`use_request_rate_calibration=true` 时 CONCURRENCY 也被钉死为 max
  （候选并发值被忽略并打 warning，且每 trial 会跑两遍 benchmark：先 max 并发测吞吐，
  再以 `throughput × 1.05` 作为 request-rate 复测）——skill 的 preflight 已强制
  `use_request_rate_calibration = false`，不要打开。

## 进程生命周期（serve/benchmark 起停）

来自 `plugins/simulate.py` 与 `interfaces/custom_process.py`：

- **启动前清场**：按 `process_name` 清理残留进程（`kill_residual_process`）；
  MindIE 每次启动前额外 `pkill -9 mindie` + `npu-smi info`。
- **benchmark 输出目录清理**：每个 trial 前清空输出目录（vllm_benchmark 清
  `result_dir`，ais_bench 清 `output_path`），保证指标只来自本次运行。
- **启动等待**：每秒轮询健康检查，上限 `wait_start_time` 秒；服务进程在启动期
  退出立即判失败，超时记 `TimeoutError`。
- **停止 vLLM**：优先按 PID 定点回收（psutil 递归 terminate 子进程树 → 等 5s →
  存活者 kill → 再等 5s）；失败回退 `pkill -15/-9 -f "vllm serve"` 最多 3 轮，
  以 `pgrep -f "vllm serve"` 判定是否清完，清不完打残留进程日志。
- **trial 日志落盘**：serve/benchmark 日志写在临时文件，清理前复制到
  `<run_dir>/trial_logs/<candidate_id>.log`（成功/失败各一份）。

## 环境与部署

- optix 必须装在 uv 虚拟环境（仓库根 `uv sync`），严禁装进系统 Python；
  vLLM/MindIE/测评工具用系统环境。
- serve/benchmark 不在系统 PATH 时，用 `[deploy] path_prefix` 或 `OPTIX_DEPLOY_PATH`
  指定部署根目录（优先级与用户手册一致：`OPTIX_DEPLOY_PATH` > `path_prefix` > 系统 PATH）。
- 子进程环境构造（`deploy_env.py::build_deploy_env`）：剥掉 `VIRTUAL_ENV`/`PYTHONHOME`
  （conda 隔离时连 `CONDA_PREFIX`/`CONDA_DEFAULT_ENV`），把 msmodeling venv 下的路径段
  从 `PATH`/`PYTHONPATH`/`LD_LIBRARY_PATH` 中过滤掉，再把 `<部署根>/bin` 前置到 PATH。
- `materialize_command` 只解析 argv[0]：`vllm`/`ais_bench` 按部署 PATH 解析为绝对路径；
  解析结果落在 msmodeling venv 内直接报错（防误装进 venv），其余 argv 原样透传。
- MindIE 可执行文件解析顺序：`$MIES_INSTALL_PATH/bin/mindieservice_daemon` →
  默认 `/usr/local/Ascend/mindie/latest/mindie-service/bin/mindieservice_daemon` →
  PATH 上的 `mindie_llm_server`。
- CLI 入口：`msmodeling optix` 与 `python -m optix` 等价；`-c` 自定义 TOML 追加在默认
  `optix/config.toml` 之后加载，同名键以自定义文件为准。

## 多机混部

完整约定见 [multihost.md](multihost.md)。参数装配复用 `AgentCandidateStrategy._params_to_target_field()`，命令渲染复用 `optix/serve_plan.py`。scheduler 在多机路径通过 `multihost_adapter.py` 转换为原插件的 env/others 协议。节点通信、脚本生成、SSH 和生命周期仍由原插件负责。
