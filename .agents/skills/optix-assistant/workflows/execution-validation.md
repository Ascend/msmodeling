# Execution Validation（执行验证）

本文件写 agent 模式每轮循环中「后台执行与进度监控 → 读取结果」的具体操作；
整体编排见 `./agent.md`。

执行后端只支持 optix——它不可用时没有替代方案，直接停止。本流程负责启动 optix、
监控进度、读取结果；**不手写 vllm serve / benchmark 命令**——serve 拉起、
benchmark 执行、候选校验、失败分类、进程清理全部由 optix 完成。

## 输入

读取 `context.json`、其 `source.config_toml` 指向的配置和 `candidates.round-N.json`。启动前执行 `python3 <SKILL_ROOT>/scripts/validate_candidates.py --run-dir <run-dir> --round N`，检查当前空间、参数与 TOML 配置；`config.toml` 在 Round 0 装配完成后冻结，Round 0 之后不得修改 TOML，新增字段规则见 `./data-analysis.md`「每轮搜索空间」。校验器按 `source.config_toml_sha256` 复核 config.toml 未被修改。optix 按 `context.search_space` 补默认值、覆盖候选值并校验，再装配 TOML 字段和额外字段。候选必须符合空间定义并满足
组合约束；生成候选时已做过筛选，执行时由 optix 用 `validate_candidate` 再校验
一遍，无效候选自动跳过，本流程不重复校验。

多机的引擎 CLI 使用 `-e multihost_infer`，配置与预检仍使用 `[vllm]` / `--engine vllm`。
保持同一绝对路径的 `CLUSTER_CONFIG_PATH` 生效；采集与每次预检传 `--cluster-config`。

## 启动前检查（全部通过才启动）

1. 重跑只读 config 门禁：
   `python <SKILL_ROOT>/scripts/config_preflight.py --check-only --config <config.toml 绝对路径> --engine <engine> --run-id <run_id>`，
   必须退出 0（无 `ERROR`）才启动。所有会写入 TOML 的 `config_preflight.py --set-*`
   和 `config_writer.py` 命令仅用于 Round 0 装配，须在 `collect_context` 记录
   `config_toml_sha256` 之前完成；**R1 起不得通过任何写入命令修改 TOML**。
   R1 起若门禁错误要求修改 TOML（包括模型路径、SLO、benchmark、agent 模式、
   run_id 或 `use_request_rate_calibration`），终止当前 run；修正配置后重新装配新 run。
   搜索参数类缺失按 `./data-analysis.md`「每轮搜索空间」更新 `context.search_space`，不修改 TOML。
2. benchmark 为 ais_bench 时，先跑 `ais_bench --models <短名> --datasets <短名> --search`
   预检（毫秒级拦截配置类故障，细则见 `../references/benchmark.md` §3）。
3. agent 模式**不**经 `apply_launch_config` / `launch_env.sh`（那是 PSO）。
   启动前只需确认本机已有的部署相关 env（如 `OPTIX_DEPLOY_PATH`）仍生效；
   不要 `source` 不存在的 `launch_env.sh`。
4. 确认 `[agent_optimizer]` 的 `run_id`/`run_dir` 与当前 run 对齐。

## 执行

1. **启动 optix**：作为后台任务启动，日志落盘，禁用裸 `nohup &`
   （裸 nohup 会脱离平台跟踪，崩溃时收不到通知）。启动方式按平台能力分两支：
   - 平台提供受跟踪的后台任务 API（后台任务退出时自动通知 agent）→ 优先使用，
     保留"整轮结束/崩溃"的通知来源：
     ```bash
     msmodeling optix -e <engine> -b <benchmark> -c <config.toml> > /tmp/optix-round-N.log 2>&1
     ```
   - 平台只有同步 shell 工具（部分平台的 shell 工具在调用超时时会清理整个
     进程组——裸 `nohup` 只挡 SIGHUP、挡不住进程组级 SIGKILL）→
     用 `setsid` 完全脱离会话与进程组：
     ```bash
     setsid msmodeling optix -e <engine> -b <benchmark> -c <config.toml> > /tmp/optix-round-N.log 2>&1 < /dev/null &
     ```
     代价是进程彻底脱离平台跟踪、失去退出通知，由第 2 条的等待方式兜底；
     拉起前先检查并清理上一轮可能残留的孤儿 serve 进程（`pgrep -f vllm`），
     防止残留进程占着显存，影响对下一轮结果的判断。
   optix 读取 run 目录的 `candidates.round-N.json` 执行本轮 trial，写回
   `results.round-N.json` / `progress.round-N.json`。
2. **等待方式（强制）**：启动后是否释放控制权，按平台的事件能力分三支——
   **先判定平台能力，再执行对应分支；无法确认时一律按分支 3 执行**（阻塞等待
   只是不能中途交互，永远安全；错判成前两支就是无人收结果的早退事故）：
   - **平台有后台任务退出通知**（后台任务退出时自动回调 agent）：
     启动后立即结束当前回复，**释放控制权**（不阻塞等待）；用户可自由发消息
     （询问进度、查看参数、调整方向等）。
   - **平台无退出通知、但有定时唤醒**（cron/提醒类机制能再次激活 agent）：
     用 `setsid` 启动后立即结束当前回复，并**同时**建好定时巡检任务
     （查 optix 进程存活 + `progress.round-N.json`），靠定时唤醒兜底推进。
   - **两者都无**（平台工具列表里既没有受跟踪的后台任务 API，也没有
     cron/提醒类定时唤醒）：**禁止释放控制权**——此时结束回复等于没人再收
     trial 结果（一次性 headless 会话如 `opencode run` 就是这种情况，回复结束
     进程即退出）。用 `setsid` 启动后在当前回复内阻塞等待：
     `sleep <300~600>` → 查两件事——optix 进程是否存活
     （`pgrep -f "msmodeling opti[x]"`，字符类防自匹配）、`progress.round-N.json`
     的 `status`/`stage` 是否推进——循环直到本轮 `completed/round_done` 或
     **optix 进程消失**（未 round_done 而进程不在 = 异常退出，立即按第 3 条
     诊断，不等停滞阈值）。
3. **事件响应（三个分支通用）**：
   - **收到 trial 完成事件 / 巡检发现 completed 计数变化 → 立即汇报该 trial**
     （模板见 `./agent.md` C3）；
   - **判定整轮结果**（退出通知、巡检发现 optix 退出、或阻塞轮询发现
     round_done）：
     - 正常退出 → 读 `results.round-N.json` 汇总（`./agent.md` C5 模板）；
     - 非零退出/崩溃 → 读 `/tmp/optix-round-N.log` 按「失败分类」节
       诊断（`./agent.md` C4 模板）；
   - 疑似卡住 → 先查 optix 进程是否存活：进程不在且本轮未 `round_done` =
     异常退出，立即诊断；进程在但无进展才适用停滞判据（见第 4 条）。
4. **监控细节**：以 `progress.round-N.json` 的 `status`/`stage` 为准（optix 写入，
   本流程不改写）；不要依赖 benchmark 的 tqdm 输出（stderr/缓冲不可靠），
   需要更细粒度时读服务日志的 `Running: N reqs` /
   `Engine 000: Avg generation throughput` 行。
   本阶段在 subagent 中执行时，trial 级实时汇报简化为轮次完成后向主 agent
   返回汇总，由主 agent 向用户汇报：
   - `starting/preparing`：本轮初始化（候选校验、baseline 准备）；
   - `running/trial_cold_start`：冷启动（权重加载→图编译→KV 初始化），
     大模型单阶段可达 5-10 分钟（`enforce_eager=false` 后更长）；日志中
     HCCL/TBE/共享内存广播等编译信息属正常阶段，不得误判卡死——日志反复出现
     `shm_broadcast "No available shared memory broadcast block found in 60s"`
     属编译/量化正常阶段，**不是死锁**；
   - `running/trial_completed`：一个 trial 完成；
   - `completed/round_done`：本轮全部完成。
   **停滞判据**：进程存活的前提下，连续 30 分钟无任何阶段/状态变化才告警
   疑似卡死（进程已消失是异常退出，不适用本条）。
5. **轮次时序（实测结论）**：optix 在 round-N 执行完时立即查找
   `candidates.round-N+1.json`，文件不存在即退出（日志
   "Candidate file not found ... Stopping at round N"），不等分析侧生成下一轮
   候选。因此**逐轮重新拉起一次 `msmodeling optix` 是正常用法**，不是异常；
   每轮启动前确保本轮 candidates 文件和 context.json 已就位，之前的成功轮已归档 context。上一进程退出前不更新 context、不预生成下一轮候选。若旧轮没有成功 trial，按 `./agent.md`「同轮失败与重试计数」修复重试或暂停；不得直接推进新轮。
6. **清理分工**：trial 结束后的 serve 进程停止、显存释放等清理由 optix 负责；
   本流程不执行任何手工清理命令。agent 模式无 `unset_env.sh` 义务
   （未跑 apply_launch_config）；若 shell 里手动 export 过临时变量，整轮结束自行清理。

## 本轮 context 归档

多机运行时自动保存执行输入的 `context.round-N.json`；进程退出后核对它与结果对应，已有归档不再复制覆盖。单机按下面规则归档。

确认 OptiX 进程已退出，并核对本轮 `results.round-N.json` 已写入后，在修改 context 或准备下一轮前执行：

```bash
cp -- "<run-dir>/context.json" "<run-dir>/context.round-N.json"
```

- 保留 `context.json` 作为工作文件；归档保存该轮执行时的完整内容。执行期间不得修改 context。
- 同轮失败重试结束后，归档随该轮新结果同步更新；重试次数与停止条件按 `./agent.md`「同轮失败与重试计数」执行。成功轮已有归档不得覆盖，重复结束通知核对归档后跳过复制。
- 崩溃未产生本轮结果时保留当前 context 用于恢复，不推进下一轮。归档缺失时须确认当前 context 仍是该轮执行输入才可补存，不得用后续轮次内容补造。
- 导出及历史分析读取对应轮次归档。context 只保存 TOML 路径；TOML 在 Round 0 装配后冻结，跨轮不得修改固定配置，导出读取的 TOML 即 Round 0 基线。确需变更固定配置时终止当前 run 重新装配。

## 读取结果

agent 先用每条 trial 的 `candidate_id` 回查同轮 `candidates.round-N.json`，通过
`direction_id` 读取方向的 `trial_type`：`diagnosis` 单独汇报，仅作为诊断证据。
只有 `optimization` 进入 agent 的正常判优、验收和收敛分析；引擎与摘要的原始统计
未过滤诊断。最终导出脚本自动完成上述关联与过滤，不能只依赖结果 metadata。

- 以 `results.round-N.json` 为准（optix 直接写入，含 `fitness`、`performance`、
  `slo_violated`、结构化 `failure`）；原始日志只通过 `evidence_ref` 字段引用，
  不直接读。
- **核对实际命令与意图一致**：每轮结果落盘后，从 optix 日志（含 serve 拉起
  命令行与 benchmark 子进程命令）确认实际执行的部署/压测参数就是本轮候选与
  空间装配的预期——重点核对固定项（并行度、max-model-len、prefix cache 开关、
  模型路径等）是否按预期出现且取值正确、有无预期之外的 flag。发现不一致时
  先停止推进，排查空间条目与命令渲染的偏差（如条目名无法映射到真实 flag），
  修复后按同轮失败规则重试。
- 失败分类直接用 optix 已经分好的结果（`failure` 的
  `category/sub_category/suggested_action/offending_params`），只读、不自己再
  分类一遍（判据见下「失败分类」）；分类不可信时按 `./agent.md`「失败恢复 ·
  Layer 2」升级处理。
- 读取前校验一致性：`round` 与文件名一致、candidate_id 在 run 内唯一；
  一致性不符时报告异常，不猜值修补。
- 原始 optix 日志（`/tmp/optix-round-N.log`）只在整轮异常退出时用于诊断，
  不作为结果依据。

## 失败分类（Layer 1：正则匹配，不需要 agent 参与）

optix（agent 模式的 `strategies.py`）在 trial 失败时已调
`validation.py::classify_failure`，用正则表 `FAILURE_CLASSIFICATION_TABLE` 产出
`failure` 结构体
（`category/sub_category/message/evidence_line/suggested_action/offending_params`）。
常见失败（OOM/timeout/segfault/connection_refused/SLO 违规）在这一层就能定，主 agent
仅对回查候选后确认为 `trial_type=optimization` 的结果读 `failure.suggested_action`，
作为后续轮次避开失败参数区域的依据（落点就是各轮 `results.round-N.json` 的 `failure` 块）；
`diagnosis` 的失败只保留为诊断证据，不作为正常寻优的失败规避依据。

> **分类可信度判据（来自实测经验）**：`failure.message` 若无可读的异常行（日志
> tail 被截断到几乎为空，仅剩片段），**一律视为分类不可信**，直接触发 Layer 2
> 深挖（`./agent.md`「失败恢复 · Layer 2」）——不要基于残缺证据的方向（如
> `oom/weight`）做失败规避决策。已修复版本中，分类改为按**尾部异常行优先**逐行
> 匹配，`failure.message` 分级截断、强制保留最后一行异常，并附 `evidence_line`
> （命中的那一行）；若看到 `category="oom"` 且 trial 秒退（~18s 内、服务从未
> 就绪），优先怀疑是 speculative-config 等启动期配置错误而非真实 OOM。

> **症状模式对照**：Layer 2 深挖前先对照
> `<SKILL_ROOT>/knowledge/modeling-diagnosis/symptom-patterns.md`（D1-D9），
> 命中即复用其诊断动作与处置规程（P1-P3，见
> `<SKILL_ROOT>/knowledge/modeling-diagnosis/procedures.md`）；判据细则
> 见 `./agent.md`「失败恢复 · Layer 2」。
