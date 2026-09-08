# Agent 模式 workflow（执行流程）

> **分工**：本文件 = agent 分支的**执行流程**（启动模式选择 → 配置就绪 → Round 0 → 每轮循环 A-E → 收敛退出）。
> 候选生成、判优、失败处理等**判据与禁止规则**，以及完成/输出衔接标准，见 `../references/agent-mode.md`（行为契约）。
> 收敛与预算数值的真源 = 仓库 `config.toml [agent_optimizer]`，本文件不背书旧数值（见「步骤 E：收敛检测」）。

---

## 执行约定（cwd 与路径锚点）

本文所有命令默认在**项目根目录**执行：`optix/config.toml`、`.agent_optimizer/runs/`、optix 启动都以项目根为锚（`[agent_optimizer].run_dir` 留空，由 optix 自动解析为 `<项目根>/.agent_optimizer/runs/<run_id>`）。

脚本路径前缀 `<SKILL_ROOT>` 指本 SKILL 所在目录：`.agents/skills/optix-assistant`。全局安装场景（Codex：`$CODEX_HOME/skills/optix-assistant`；Claude Code：`~/.claude/skills/optix-assistant`）执行前替换为该目录的**绝对路径**。

`--run-dir` 统一传 `<项目根>/.agent_optimizer/runs/<run_id>`。

## 目标

用 AI agent 驱动服务化参数调优。Agent 负责读上下文、生成候选、分析结果和决策下轮方向；Python 脚本负责确定性的校验、执行、摘要汇总和结果导出。

---

## 启动：模式选择

**Agent 进入本分支后第一条消息必须是模式选择，禁止跳过，禁止默认 autonomous。**

```
请选择寻优模式：

1. 全自动 — 候选自动生成执行，每完成一个 trial 实时汇报，每轮有总结。不询问确认。
2. 交互式 — 每轮候选生成后展示给用户确认（确认/修改/跳过），执行过程与全自动相同。

请输入 1 或 2：
```

**两种模式共同行为**：
- 选定后全程不再问"要继续吗？""这个候选可以吗？"
- 执行过程中每个 trial 完成实时汇报
- 每轮有总结
- 收敛后自动输出最终报告

**两种模式区别**：
- 全自动：候选生成 → 校验 → 后台执行 → 进度汇报 → 总结 → 下一轮，全自动流转
- 交互式：候选生成后暂停，展示候选列表+每个候选的 rationale/expected_effect/risk，用户选择：1.确认执行 2.修改某个候选 3.跳过本轮

---

## 工作流

### 配置就绪（强制，每次运行必做，禁止跳过）

**通用骨架的唯一宿主 = `../SKILL.md`「共享前置流程」**：0a benchmark 询问、`config_preflight.py
--discover` / `--check-only`、共享问询清单（使用卡数 / 模型路径与服务模型名 / ais_bench 短名 /
输入输出长度，含"禁止猜值"原则与探测边界）、写入职责分工、生效配置确认表——本文件不重复维护，
以 SKILL.md 为准；命令字段明细见 `config_preflight.py --help` 与 `../references/benchmark.md`。

**agent 专属差异**（在通用骨架之上追加，全自动/交互式两种模式都强制）：

- 寻优目标：`tpot_slo` / `ttft_slo`（单位秒）必填且 > 0 → 必须问用户目标值（即使有默认值也要
  确认，禁止静默填）→ `config_preflight --ttft-slo <x> --tpot-slo <y>`；用户无概念时按
  `../references/benchmark.md` §6.3 典型场景模板给建议值，标注"按典型场景建议，请确认"
- 轮数 / trial / 时间预算 → 与用户确认后写入 optix 实际读取的 config.toml：
  `config_preflight --set-max-rounds <N> --set-max-trials <N> --set-time-limit-minutes <N>`
  （写入 `[agent_optimizer]`，是 optix 运行终止的真实依据；`time_limit_minutes` ≤ 1440（24h））
- agent 模式 → 确认是否开启 → `config_preflight --set-agent-mode`
  （自动对齐 run_id + 清空 run_dir + 写入 `use_request_rate_calibration=false`）
- **单一事实源契约**：`--check-only` / `--set-*` 回填、`apply_launch_config` 写入、optix 启动，
  必须全部使用同一个**显式绝对路径**（`--config <绝对路径>`），禁止各自再发现——避免 preflight
  校验的文件与 optix 实际消费的文件不一致；指定卡号（可选）经 `ASCEND_RT_VISIBLE_DEVICES`
  （可写 `launch_env.sh`）与 `--world-size` 交叉校验

> **run-id 自动化**：run-id 是内部标识（复用已有或自动生成 `run-<时间戳>`），**用户无需提供**。
> config.toml 模板的 `run_id = "default"` 是占位（无实义）——首次 `--set-agent-mode` 会自动生成真实 run-id 写入并取代它；已有真实 run-id（run-xxx）再 `--set-agent-mode` 会沿用（同一实验续跑）。agent 捕获 config_preflight 输出的 `run-id:`，后续 `collect_context --run-id <run-id>` 保持一致；optix 才能在同一目录找到 `candidates.round-N.json`。

> **边界**：完整性问题（缺模型路径/SLO）由本步骤引导用户；`apply_launch_config.py` 负责把**匹配场景**的 env/additional_config 应用（命中 presets 时）。

### Round 0：初始化与自检

```
1. collect_context.py → context.json（多源融合搜索空间：config.toml + known_patterns + 模型推导 + vendor preset 官方推荐）
   ↓
2. experience_injector.py → experience_report.md（经验注入，强制）
   ↓
3. check_search_space.py → checklist_report.md（自检）
   ↓
4. 生成 candidates.round-1.json
```

#### 步骤 1：准备 run 目录并采集上下文

```bash
python <SKILL_ROOT>/scripts/collect_context.py \
  --run-dir .agent_optimizer/runs/<run_id> \
  --engine vllm --benchmark ais_bench --world-size <N> --num-nodes <N> --num-per-node <N> --memory-gb <N>
```

> **预算不在 collect_context 设置**：轮数 / trial / 时长的运行时终止由 config.toml `[agent_optimizer]`
> 决定（经 config_preflight `--set-max-rounds/--set-max-trials/--set-time-limit-minutes` 写入）；
> collect_context 的同名旗标仅作信息记录，默认值与 `[agent_optimizer]` 保持一致。

如果有本 skill PSO 分支的推荐结果（assistant_handoff / 旧 `config_skill_handoff` 同构），优先使用 `--recommend-result`。

如果模型命中 `presets/ascend_vllm_presets.json`（vLLM Ascend 官方推荐配置目录），collect_context.py 会自动注入 `knowledge.vendor_preset`，并把官方推荐的缺失参数补进搜索空间（`source="vendor_preset"`）。命中场景的官方参数作为 **Round 1 基线**与 strong_suggest 默认值。

**命中后必跑：应用匹配场景的 launch 配置**（env + additional_config 不会自动生效，需显式应用）：

```bash
python <SKILL_ROOT>/scripts/apply_launch_config.py \
  --run-dir .agent_optimizer/runs/<run_id>
```

该脚本会把匹配场景的 `additional_config` flag 合并进 `config.toml [<engine>.command].others`（幂等 + TOML 校验 + 备份），并把 `env` 生成到 `<run-dir>/launch_env.sh`，同时生成配套清理脚本 `<run-dir>/unset_env.sh`（unset 本脚本写入的全部 key）。D1 启动 optix 前先 `source` 该脚本（见步骤 C）；**每轮 optix 运行结束或切换 preset 后必须 `source unset_env.sh`**，防止残留环境变量污染下一轮 trial 的结果归因。

#### 步骤 1.5：经验注入（强制，collect_context 之后、check_search_space 之前）

```bash
python <SKILL_ROOT>/scripts/experience_injector.py \
  --run-dir .agent_optimizer/runs/<run_id>
```

产出 `experience_report.md`，包含理论推导+硬约束预检+历史经验参考。**agent 必须在 Round 1 候选生成前读取此报告**（步骤 3 之前不可跳过）。

> **设计意图**：collect_context 完成搜索空间装配后，experience_injector 基于 context.json 中的模型/硬件信息做三道事：
> 1. 理论推导（Roofline / KV 预算 / TP 候选）—— 给参数的物理安全边界
> 2. 硬约束预检 —— 标记互斥/强依赖/数值违规组合，候选生成时避开
> 3. 历史经验匹配 —— 内置种子经验（来自 experience_seed + known_patterns + 同类历史案例）给出参数推荐倾向
>
> Agent 在生成 candidates.round-1.json 时必须交叉引用此报告的"候选生成建议"章节，不可忽略。

**命中后必读：引擎级固定参数（`context.engine_fixed_params`）**。preset 的
`additional_config`（如 `speculative-config.enforce_eager=true`）只写进
config.toml，不参与搜索；collect_context 已把它们登记为 `engine_fixed_params`，
其中 `flip_required=true` 的参数（当前为 `speculative-config.enforce_eager`，
注入搜索空间名为 `speculative_enforce_eager`）已被注入搜索空间。**Round 1 必须
包含至少一个单变量反转候选**（如 `speculative_enforce_eager=false`），用来质疑
preset 默认值是否为目标硬件的次优值。注意：裸名 `enforce_eager` 是主模型顶层
`--enforce-eager`（见 custom_command FLAG_NAME_MAP/JSON_SUBKEY_MAP 注释），与
投机容器内的 eager 语义不同，勿混用。
未做反转验证前，禁止把任何引擎级固定参数造成的性能上界表述为"物理极限"。

#### 步骤 2：搜索空间自检

```bash
python <SKILL_ROOT>/scripts/check_search_space.py \
  --run-dir .agent_optimizer/runs/<run_id>
```

**自检后有错误的处理**：暂停，报告错误，等用户修正后继续。仅有警告：输出报告后继续。

#### 步骤 3：生成首轮候选

参见 `../references/agent-mode.md`「候选生成规则」章节。首轮固定 4 个候选（基线 + 3 个方向探索）。

---

### 每轮循环（Round N）

```
1. 读取数据源
2. 生成 candidates.round-N.json
3. 后台执行 trial（optix agent 模式自行校验候选，无效的自动跳过）
4. 定时检查进度 + 实时汇报
5. summarize_history.py 追加 summary.md
6. 收敛检测
7. 下轮方向决策
```

#### 步骤 A：读取数据源

每轮候选生成前必须读取：

1. `context.json` — 搜索空间、约束、model_info、knowledge（仅 Round 0 全量读一次；后续轮搜索空间不变，无需重读）
2. `summary.md` — 所有过往轮次摘要、趋势、失败分析
3. `results.round-*.json` — 结构化 trial 数据（尤其是上一轮的 `failure` 字段，含 `category/sub_category/suggested_action/offending_params`）。执行由 `msmodeling optix` agent 模式（`optimizer_strategy="agent"`）完成，results 内含结构化 `failure` 块，**原始 vLLM/benchmark log 不内联**（由 optix scheduler 持久化）。主 agent 读取 results 时只接触结构化字段，不接触原始 log。
4. `known_patterns.json` — 人类维护的模型族级规则（本 skill `knowledge/` 目录）
5. **`experience_report.md`（强制）** — 由 `experience_injector.py` 在 Round 0 生成的**经验注入报告**，内含：
   - 理论推导结论（Roofline、KV 预算、TP 候选安全范围）
   - 硬约束预检结果（候选生成时必须避开的违规组合）
   - 历史经验匹配（内置种子经验中与当前模型/硬件匹配的参数倾向）

   Agent 生成候选时**必须交叉引用此报告**，每条候选的 rationale 中需说明是否参考了 `experience_report.md` 的建议。尤其 Round 1 的基线候选应优先采用报告中的"Round 1 基线候选建议"。

#### 步骤 B：候选生成

参见 `../references/agent-mode.md`「候选生成规则」章节。

**执行前必展示候选参数（强制，两种模式都要）**：生成 `candidates.round-N.json` 后、启动执行前，必须用**表格**向用户展示本轮候选的参数清单（让用户无需翻 JSON），例如：

```
Round 1 候选参数（4 个）
候选ID | MAX_NUM_BATCHED_TOKENS | MAX_NUM_SEQS | tp | dp | cudagraph_mode     | GPU_MEMORY_UTILIZATION
r1-a   | 16384                  | 48           | 4  | 4  | FULL_DECODE_ONLY    | 0.85
r1-b   | 32768                  | 64           | 4  | 4  | FULL_DECODE_ONLY    | 0.85
```

**表格上方必须先写"每个候选的选择依据"**（rationale + 引用 summary.md/results 的具体
数据），方便交互模式下用户快速判断确认/修改/跳过，而不是只看到参数表。

**如果是交互式模式**：生成后展示候选列表给用户确认。

> **候选校验由 optix 执行**：`msmodeling optix` agent 模式读 `candidates.round-N.json` 时自行调 `validate_candidate`（搜索空间/约束/运行时 flag 校验），**无效候选自动跳过**，agent 无需单独校验步骤。生成候选时仍需主动避开 combo safety 风险（KV cache 溢出、并发-批次冲突）。

#### 步骤 C：后台执行（强制执行契约）

**C1：启动（平台后台任务 + 进度监控，禁用裸 `nohup &`）**

**关键原则**：不要用分离进程（`nohup ... &`）——它脱离平台跟踪，退出（崩溃/完成）时 agent **收不到任何通知**，实时汇报无从实现。改用**平台可跟踪后台任务 + 事件监控**：

1. **后台任务**：把 `msmodeling optix` 作为当前平台的**可跟踪异步任务**启动（Claude Code：Bash `run_in_background: true`）。**任务退出时平台自动通知 agent**（成功/崩溃/超时）——这是"整轮结束/崩溃"的汇报来源。
2. **进度监控**：同时启动一个 Monitor 观察 `progress.round-N.json`（每完成一个 trial 该文件更新一次），**`completed` 计数变化时发一条事件**——这是"每个 trial 完成"的实时汇报来源。
3. **（兜底）定时唤醒**：若平台不支持事件监控，用定时任务（Claude Code：CronCreate）每 N 分钟检查进度并汇报。

```bash
# Claude Code 实现示例：
# 1) 后台跑 optix —— 退出（成功/崩溃）时自动收到通知
Bash(run_in_background=true): msmodeling optix -e <engine> -b <benchmark> -c <config.toml> > /tmp/optix-round-N.log 2>&1
# 2) Monitor 轮询进度文件 —— 每个 trial 完成发一条事件
Monitor: 每 ~5 秒读 progress.round-N.json，completed 数增加时 emit 一条 "trial 完成" 事件
```

**执行前置条件（config.toml 契约）**：
- `optimizer_strategy = "agent"` 必须开启 —— **optix 没有 `--agent-mode` 标志**，agent 寻优由该配置项驱动；未开启时 optix 走内部 PSO，会忽略本 run 目录的候选。
- `[agent_optimizer]` 的 `run_id`/`run_dir` 必须与 agent-optimizer 的 run 目录对齐：`run_id` 与 `collect_context.py --run-id` 一致（`run_dir` 留空时自动解析为 `.agent_optimizer/runs/{run_id}`）。
- `<engine>` = context 的引擎（vllm/mindie），`<benchmark>` = benchmark（ais_bench/vllm_benchmark），`<config.toml>` = 配置路径（默认 `optix/config.toml`）。
- optix 会读取 run_dir 下的 `candidates.round-N.json` 执行本轮 trial，写回 `results.round-N.json` / `progress.round-N.json`；仅当 `candidates.round-N+1.json` 尚未生成时，本轮执行完即退出（否则会继续跑下一轮）。

**C1 前必跑：config.toml 门禁（强制）**。启动 optix 前先执行：

```bash
python <SKILL_ROOT>/scripts/config_preflight.py --check-only --config <config.toml> --engine <engine> --run-id <run_id>
```

必须退出 0（无 `ERROR`）才启动；TOML 解析失败或完整性缺失（模型路径/SLO/benchmark/agent 模式/`[agent_optimizer].run_id` 对齐）时**禁止继续**，用 `config_preflight.py` 的 `--set-*` 回填修正（搜索参数类修正走 `config_writer.py`，禁止手写 TOML）。

**C1 前必跑（二）—— benchmark 配置预检（ais_bench 时强制，毫秒级拦截配置类故障）**：
optix 启动时会执行 `ais_bench --models <name> --search` 探测模型配置；若配置不在 AISBench 包内搜索路径或缺少 `--datasets`，服务会在启动早期失败。启动前先手动预检：

```bash
# benchmark = ais_bench 时（[ais_bench.command].models/datasets 需能独立解析）
ais_bench --models <models 短名> --datasets <datasets 短名> --search
# 期望输出包含两行的 Task Type/Task Name/Config File Path 表格；报 "match config file failed"
# 或 "You must specify --datasets" 即配置有问题，先修再启动（见 ../references/benchmark.md §3）
```

**C1 前必跑（三）—— `use_request_rate_calibration`（agent 模式，已由 preflight 门禁强制）**：
`config_preflight --set-agent-mode` 已自动写入 `use_request_rate_calibration = false`，无需手改 TOML。
该值为 `false` 时 optix 走 `scheduler.run`：候选 CONCURRENCY 逐 trial 生效（如 64/128/256），REQUESTRATE 固定 max，
闭环才能测出并发-SLO 拐点，与 known_patterns 的 CONCURRENCY 探索方向一致；为 `true` 时 optix 会把候选
CONCURRENCY 钉死为 max（optimizer.py fixed_target 逻辑），**agent 模式的并发维度静默失效**。
若 `config_preflight --check-only` 报"use_request_rate_calibration 需为 false"的 ERROR，
重跑 `config_preflight --set-agent-mode` 自动修正即可。

**C1 前：应用匹配场景的 env 与 additional_config（命中 `knowledge.vendor_preset` 时）**。Round 0 的 `apply_launch_config.py` 已把 `additional_config` 合并进 `config.toml [<engine>.command].others`（幂等）并生成 `<run-dir>/launch_env.sh`。启动前：

1. **env**：`source <run-dir>/launch_env.sh`（optix 的 Popen 继承环境变量，vLLM 即生效）。
   ```bash
   source .agent_optimizer/runs/<run_id>/launch_env.sh
   ```
   整轮结束 / 切换 preset 后必须清理，防止旧值残留污染后续 trial：
   ```bash
   source .agent_optimizer/runs/<run_id>/unset_env.sh
   ```
2. **config.toml**：重跑 validate_toml 门禁确认 `others` 合并后仍合法（apply_launch_config 已校验，保险起见再跑一次）。

**启动后汇报**：

```
🚀 Round N 已启动，M 个 trial 后台执行中。
   进度文件: .agent_optimizer/runs/<run_id>/progress.round-N.json
   我会在每个 trial 完成时实时汇报；整轮结束/崩溃也会收到通知。
```

**C2：执行契约（强制）**

1. 启动后立即结束当前回复，**释放控制权**（不阻塞等待）
2. 用户可自由发消息（询问进度、查看参数、调整方向等）
3. **收到 Monitor 事件（trial 完成）→ 立即汇报该 trial**（C3 模板）
4. **收到后台任务退出通知 → 判定整轮结果**：
   - 正常退出 → 读 `results.round-N.json` 汇总（C5 模板）
   - 非零退出/崩溃 → 读 `/tmp/optix-round-N.log` 按 `../references/agent-mode.md`「失败处理」分类诊断（C4 模板）
5. 超时阈值内无任何事件（后台任务疑似卡住）→ 用定时唤醒兜底检查

**进度文件状态机（Monitor 判定依据）**：`progress.round-N.json` 由 optix 写入，
首 trial 前即存在，字段 `status`/`stage` 取值：

| status | stage | 含义 |
|--------|-------|------|
| starting | preparing | 本轮初始化（候选校验、baseline 准备） |
| running | trial_cold_start | 某 trial 正在冷启动：权重加载→Graph capture→KV cache 初始化 |
| running | trial_completed | 刚完成一个 trial，等待下一个 |
| completed | round_done | 本轮全部完成 |

**冷启动说明**：vLLM 加载大模型 + MTP + CUDA Graph capture 单阶段可达
5-10 分钟（`enforce_eager=false` 后更长），日志反复出现
`shm_broadcast "No available shared memory broadcast block found in 60s"`
属编译/量化正常阶段，**不是死锁**。只有**连续 30 分钟无任何阶段/状态变化**
才告警"疑似卡死"。

**进度判定命令**：不要依赖 benchmark 的 tqdm 输出（stderr/缓冲不可靠），以
`progress.round-N.json` 的 `status/stage` 为准；需要更细粒度时读服务日志的
`Running: N reqs` / `Engine 000: Avg generation throughput` 行。

**C3：trial 完成汇报模板**

每 trial 完成汇报**必须包含 TPOT / TTFT / 吞吐**（从 `progress.round-N.json` 的 `trials[].performance` 读取，全自动也汇报）：

```
📊 Round N 进度: X/M 完成
   ✅ rN-a: fitness=X | 吞吐=X tok/s | TTFT=Xms | TPOT=Xms | 成功率=X% | 耗时 Xmin
   🔄 rN-b: 运行中 (已 Xmin)
   ⏳ rN-c, rN-d: 等待中
```

**C4：trial 失败汇报模板**

```
❌ Round N, rN-x 失败 [category/sub_category]
   offending_params: MAX_NUM_BATCHED_TOKENS=98304, GPU_MEMORY_UTILIZATION=0.94
   建议: reduce_batch
   下轮自动避开此参数组合
```

**C5：全部完成汇报模板**

```
✅ Round N 全部完成!
   最佳: rN-x (fitness=X, throughput=X tok/s, TPOT=Xms)
   [如有失败] 失败: rN-y [category/sub_category]

   📝 分析: [本轮关键发现，引用具体数据]
   → Round N+1 方向: [下轮方向简述，每个候选的选择依据]

   → 生成 Round N+1 候选中...
```

#### 步骤 D：总结

```bash
python <SKILL_ROOT>/scripts/summarize_history.py \
  --run-dir .agent_optimizer/runs/<run_id> --round N
```

`summarize_history.py` 追加内容到累积式 `summary.md`。

**关键约束**：agent 必须先写出"下轮方向建议"节（包含每个候选的 rationale 和引用数据），再生成 `candidates.round-N+1.json`。禁止无依据的猜测。

#### 步骤 E：收敛检测

每轮结束后检查（以下收敛与预算参数均在 config.toml `[agent_optimizer]` 中配置，
下述数值为仓库 `config.toml` 当前默认，与 `config.py` 的 `AgentOptimizerConfig` 兜底值一致；
调参只改 config.toml，勿在此文档留旧值；轮数 / trial / 时间预算可用 config_preflight
`--set-max-rounds/--set-max-trials/--set-time-limit-minutes` 写入）：

- **趋势条件**：比较本轮 best_fitness 和全局 best_fitness（SLO 达标优先，见
  `../references/agent-mode.md`「fitness 公式解读」），连续 `convergence_rounds` 轮改善 < 1%（默认 **6**）。
- **覆盖条件（必须同时满足）**：`optix` 会用 `search_space_coverage` 检查——
  已执行轮次 ≥ `min_rounds`（默认 4）、已完成 trial ≥ `min_trials`（默认 12）、
  且 `engine_fixed_params` 中所有 `flip_required` 参数已完成反转验证。
  任一不满足时输出"覆盖率不完整"及原因，**继续下一轮**，不收敛。
- 两个条件都满足才输出 "收敛退出" 并停止；跑满 `[agent_optimizer].max_rounds`
  （默认 32）自然终止。
- **全局时间预算**：`[agent_optimizer].time_limit_minutes` 是 optix 运行的时间
  上限（默认 1440 分钟，preflight 校验其 ≤ 1440）。agent 跨轮累计各轮耗时，
  预算耗尽前完成当前 trial 并执行 `export_best_config.py` +
  生成 `final_report.md` 优雅收尾（不硬中断）。

**注意**：collect_context 不控制上述参数——其同名预算旗标仅作信息记录，
运行时以 config.toml `[agent_optimizer]` 为准。

---

