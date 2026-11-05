# PD agent 部署搜索

本流程使用 `-e vllm_pd`，复用原插件，不接入 `--mode pd_disagg`。沿用 agent 的知识路由、候选分析、SLO 和失败恢复约定。

## 实验边界

执行前确认设备清单、模型和软件版本、传输配置、负载、SLO、搜索预算和独立复测预算。固定主机与设备身份；跨轮允许重新划分 P/D 设备池。PP/PCP 固定为 1。原插件对 dense 模型的 DP 固定渲染为 1，因此拒绝其 DP>1 候选，使用多个独立实例扩容。部署控制的设备、网络与端口环境变量不可作为候选覆盖。TP 设备片段不能跨节点，TP/DP 与实例数必须满足原插件的分组规则。

MTP 配置在 Round 0 冻结实验前准备，沿用 `workflows/environment-preparation.md`：Agent 明确选择
后端支持的 method，使用 `config_writer.py --prepare-speculation <JSON> --run-dir <计划run目录>` 写入
`[vllm.command].others`，不写 `[vllm_pd].vllm_others`。两侧 γ 通过 `engine=vllm_pd` 的
`num_speculative_tokens_prefill/decode` 字段登记，初始为 0；正数候选覆盖公共关闭兜底。
`GPU_MEMORY_UTILIZATION_prefill/decode` 必须是 run 字段。不得在冻结后原地改变方法底座。

Round 0 使用 `collect_context.py --engine vllm --deployment pd`，其余模型、设备和负载选项按 `--help` 填入已确认值。脚本保存不可覆盖的 `pd_experiment.json`；后续轮次仍使用同一 run 目录。context 不自动生成搜索空间。

可在 run 目录登记 `pd_search_plan.json`，例如：

```json
{"layouts":[{"prefill":{"devices":4,"tp":2,"dp":1},"decode":{"devices":4,"tp":2,"dp":1}}]}
```

这是待覆盖范围，不代表配置一定合法；执行前还要进行完整资源校验。未登记计划时，不宣称布局覆盖完整。

## 每轮操作

1. 按知识与实测证据维护搜索空间，生成一个候选。保留现有 `params`、分析与引用字段，增加 `deployment.nodes`：每项包含基础配置资源节点索引 `source_node`、`role` 和 `gpu_ids`。可将一个节点的不同设备分配给不同角色；不得扩展设备清单。
2. 原控制进程退出后，通过统一入口准备并执行一轮：

   ```bash
   python .agents/skills/optix-assistant/scripts/run_pd_round.py \
     --run-dir <run-dir> --round N
   ```

   入口读取固定的 `candidates.round-N.json`，自动生成部署配置、重新采集 context、执行配置与候选预检，
   随后将同一配置传给 `-e vllm_pd -c ...`。不得手工串联另一份 `-c` 或沿用旧 context。
   `--prepare-only` 只执行准备和全部预检，不启动服务；同轮重试使用新的 `--attempt 2`，保留失败证据。
   启动准备或复测分别传 `--purpose bootstrap` / `verification_baseline` / `verification_winner`。
3. `execution.round-N.attempt-M.json` 绑定候选、轮次、用途、配置与 context 指纹以及解析后的服务、benchmark、
   优化器设置与有效 P/D 参数计划。启动前复查实际加载的设置和待下发参数；候选、配置或 context 不一致则停止，不会自动换配置继续运行。
   轮次配置继续作为执行快照保留；文件由程序选择，不由 agent 分别传递。`[vllm.command].others` 从基础配置保留，
   生成器不会猜测缺失的投机 method。若要改变冻结底座，建立新实验。
4. 实测启动前生成唯一归档身份（轮次、尝试、用途、候选和 case），结果保存时禁止改成其他候选。
   归档 results 与 context，检查清理，再分析下一轮。搜索预算跨进程累计；实际启动失败消耗次数，
   启动前预检拒绝不消耗。不得通过新建 run 目录重置同一实验预算。

`pd_deployments/<归档身份>/deployment.json` 保存启动前尝试记录及实际脚本快照；身份包含轮次、配置尝试、用途、候选、case 和服务启动重试序号，`pd_search_summary.json` 汇总覆盖、失败及重复布局。预算耗尽后汇报未覆盖布局和最佳已测方案。

## 独立复测与导出

预先约定至少两次基线与两次胜出配置的完整复测、独立验证预算和 median fitness 汇总口径。先执行一次 `export_best_config.py` 固定搜索胜出方案。复测候选的 params 必须完整复制原始基线或胜出 trial 的有效参数；配置生成脚本从对应归档原样重放 TOML，不重新规范化。复测仍使用相同负载及冻结配置；每次复测使用不同的新轮次，或在同轮使用不同的 `--attempt`；配置和对应清单保存在 `<run-dir>/`，不会覆盖已有文件。生成时用 `--purpose verification_baseline` 或 `--purpose verification_winner` 标注，不能将搜索中选出的成绩当作独立重复。沿用上游 trial_type 契约，diagnosis trial 不参与最佳导出，也不计入有效复测次数。

复测启动前写入独立预算 `pd_verification_budget.json`，例如：

```json
{"required_repetitions":2,"aggregation":"median_fitness","max_trials":6,"time_limit_seconds":3600}
```

预算首次执行时冻结，失败尝试同样计数；预算耗尽不得通过重新启动重置。示例数值需按实际部署耗时调整并事先确认。

验证清单 JSON 包含 `required_repetitions`、`baseline` 和 `winner`。每侧包含 `expected`（`purpose`、`config_sha256`、完整有效 `params`、`benchmark_spec`）和 `references`（如 `results.round-5.json#trials[0]`）。指纹来自待复测配置；重复记录必须与之完全一致。脚本进一步核对预期字段与 `baseline.json`、胜出结果的真实归档一致；不得为使检查通过修改预期部署或负载。复测记录不参与重新选择搜索胜出方案。

```bash
python .agents/skills/optix-assistant/scripts/summarize_pd_search.py \
  --run-dir <run-dir> --verification-plan <verification-plan.json>
python .agents/skills/optix-assistant/scripts/export_best_config.py --run-dir <run-dir>
```

无复测证据时可省略 `--verification-plan`，状态保持 pending。汇总保留每次指标、median 和范围；verified 表示重复证据满足约定，不自动表示有性能提升。

导出使用胜出轮归档，产物为 `serve_command_pd.json`、`best_deployment/` 和 handoff。包含原插件实际生成的 P/D/代理脚本、代理依赖、配置与节点清单。缺失必要产物时拒绝导出。final_report 说明资源布局、覆盖范围、失败、复测状态和尚未完成的集群验收。

## 集群验收

离线测试不证明真实 KV 传输、代理路由、设备释放或性能收益。设备到位后按 RFC 执行固定布局闭环、单侧参数隔离、跨轮重分配、失败恢复、双方复测及导出复现。


## 五步容量辅助流程（显式启用）

主干：**先跑通 → 测业务基线 → 按需少量测 P、D → 完整 PD 寻优 → 复测导出**。不要求用户提供调优经验；已有配置优先复用，建议起点与硬约束分开。原固定布局流程仍可独立使用。

1. **准备与跑通。** 保留原环境准备、知识路由、候选分析和预检。主 run 采集冻结设备清单后，在 `pd_workflow.json` 登记已确认预算（格式如下）；`pd_workflow.py bootstrap` 输出有界启动布局建议。可提供 `--params <JSON对象文件>` 固定已知参数，硬约束同时登记在策略文件的 `fixed_params`。模型/后端规则仍需预检，不能认为建议保证装得下模型。agent 按证据选一个候选，使用原 `generate_pd_config.py --purpose bootstrap` 生成配置，再重采集 context、执行原 OptiX 入口。该用途只执行准备评估，不写正式 `baseline.json`，也不进入搜索轮次；失败后在预算内换候选，清理不明时停止。
2. **业务基线。** 跑通后使用目标业务负载与普通 search 用途进入原 agent 流程，先建立正式基线。有效指标、有限评分和成功请求是基线成立条件，SLO 可尚未达标。无效旧检查点会拒绝复用，需保留失败证据后修复。此时冻结每实例 TP/DP；后续只改变实例数量、卡组放置与服务参数。不要把启动探测或 P/D 阶段指标写进业务基线。
3. **可选阶段探测。** 首版支持单主机 TP 实例、`vllm_benchmark` 的显式 random 协议。P 关闭 prefix caching、输出 1 token；D 启用 prefix caching、固定 benchmark seed、先预热后计分。输入长度及 D 输出长度匹配业务负载。agent 各提出少量候选，不运行 PSO；阶段使用普通 vLLM 参数名，回到完整 PD 时映射为相应 `_prefill` / `_decode` 字段，并发/请求率不写入服务参数；DP>1 或不支持该协议的场景跳过探测，继续完整 PD 搜索。阶段的独立 TOML、context、candidate 和 resource 文件由 Agent 准备并传入，不覆盖主 `context.json`；context 和 candidate 的副本及测量结果写入 `pd_tasks/<任务ID>/`。
4. **完整 PD 搜索。** `pd_workflow.py layouts --params <完整有效参数JSON>` 输出合法布局，默认用满设备，可加 `--allow-unused-devices`。枚举有数量/组合上限，明确报告截断，不宣称穷尽所有卡位排列。可追加 `--prefill-capacity <P任务attempt.json> --decode-capacity <D任务attempt.json>`；同一实验且参数/规格匹配时按容量公式排序，不匹配则输出未使用容量线索。agent 参考这些证据选择；容量公式只是线索。先固定服务参数比较布局，再给有希望的布局可比较的预算交替调整 P、D 参数；参数变化后旧容量不作为精确值。仍是一进程一部署候选。
5. **复测与导出。** 沿用前文基线/胜出独立复测和实际脚本导出；阶段记录不参与端到端最佳结果评选。报告阶段跳过/失败、未覆盖布局、预算和实机未验证项。

预算文件示例（数值仅说明格式，运行前按用户预算确认）：

```json
{
  "version": 1,
  "fixed_params": {},
  "budgets": {
    "total": {"max_trials": 24, "time_limit_seconds": 14400},
    "bootstrap": {"max_trials": 4, "time_limit_seconds": 2400},
    "baseline": {"max_trials": 2, "time_limit_seconds": 1200},
    "prefill_probe": {"max_trials": 2, "time_limit_seconds": 1200},
    "decode_probe": {"max_trials": 2, "time_limit_seconds": 1200},
    "search": {"max_trials": 10, "time_limit_seconds": 7200},
    "verification": {"max_trials": 4, "time_limit_seconds": 2400}
  }
}
```

策略首次执行后冻结；失败尝试与子任务计入总账，复测额度预留，重启不清零。原 `pd_budget.json` 搜索上限和 `pd_verification_budget.json` 复测约定继续生效，取共同约束。时间限制触发后先退出执行再清理，清理耗时也记录；不承诺清理必定能在预算内完成。

异常中断且未记录结束时间的尝试，在确认清理后记为 `aborted`，仍计入次数；因为真实运行时长不可还原，标记 `elapsed_unknown`，不把停机时间计入时间预算。搜索汇总通过 `unknown_elapsed_attempts` 提示这一不确定性，预算时间因此可能低估实际运行时间。

辅助入口（默认项目根目录）：

```bash
python .agents/skills/optix-assistant/scripts/pd_workflow.py bootstrap --run-dir <主run目录>
python .agents/skills/optix-assistant/scripts/pd_workflow.py layouts --run-dir <主run目录> --params <固定规格JSON>
python .agents/skills/optix-assistant/scripts/pd_workflow.py status --run-dir <主run目录>
```

阶段执行使用独立配置中的 `pd_disagg.<phase>` 测量覆盖项与 target_field，但不调用内置 PSO 入口：

```bash
python .agents/skills/optix-assistant/scripts/pd_workflow.py probe \
  --run-dir <主run目录> --phase prefill --task-id p1 \
  --config <阶段TOML> --context <阶段context.json> \
  --candidate <单个agent候选JSON文件路径> --resource <设备JSON文件路径>
```

阶段 TOML 基于运行环境准备并配置 `pd_disagg.<phase>` 测量覆盖项和目标字段；context 是独立的单服务上下文，candidate 是该上下文下经校验的单个候选。四个参数均为文件路径，需先准备文件；工具不会自动生成这些输入。设备 JSON 的内容为 `{"ssh_ip":"设备主机地址","gpu_ids":[0,1]}`；阶段服务参数中 TP 必须匹配卡数。命令必须在该设备主机、使用共享主 run 目录串行执行；不自动搬运模型或部署运行环境。agent 不能在控制节点直接操作另一台主机的卡号。软件/硬件事实缺失时容量记录不可声明跨任务缓存命中；D 缓存真实命中与设备释放仍需实机验收。

阶段探测仅支持内置本机 vLLM 插件；服务启动前持久记录独立进程组，清理确认要求同一主机、用户和网络命名空间中，已记录的进程组及服务监听端口均已消失。中断后在原环境执行 `python .agents/skills/optix-assistant/scripts/pd_workflow.py recover-phase --run-dir <主run目录>`，再继续工作；检查失败不得复用资源。旧版没有清理身份的阶段记录无法自动确认，需人工检查并处理。
