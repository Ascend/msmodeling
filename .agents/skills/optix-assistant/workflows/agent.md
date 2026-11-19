# Agent 模式 workflow（编排层）

> **分工**：本文件只写 agent 分支的整体安排（模式选择 → 各阶段怎么串 → 汇报要求 →
> 失败恢复 → 收敛收尾），不写每步的具体操作。
> 各阶段的具体步骤在三个阶段流程文件里（见「阶段归属」表，进入对应阶段时完整阅读）。
> 候选生成与判优的规则见 `./data-analysis.md`「候选生成规则」「fitness 公式解读」；
> 失败分类（Layer 1）判据见 `./execution-validation.md`「失败分类」。
> 收敛与预算的数值以仓库 `config.toml [agent_optimizer]` 为准（判定细则见
> `./data-analysis.md`「收敛检测」）。
>
> agent 分支的运行时依赖（`optimizer_strategy="agent"`、config.toml
> `[agent_optimizer]` 配置节、candidates/results/progress 文件、收敛状态机）已在
> msmodeling-x 完整跑通（3 轮 12 trial 实测，run-id `run-20260901-qwen35`）。

---

## 执行约定（工作目录与路径）

本文所有命令默认在**项目根目录**执行：`optix/config.toml`、`.agent_optimizer/runs/`、optix 启动都以项目根目录为基准（`[agent_optimizer].run_dir` 留空，由 optix 自动解析为 `<项目根>/.agent_optimizer/runs/<run_id>`）。

脚本路径前缀 `<SKILL_ROOT>` 指本 SKILL 所在目录：`.agents/skills/optix-assistant`。全局安装场景（Codex：`$CODEX_HOME/skills/optix-assistant`；Claude Code：`~/.claude/skills/optix-assistant`）执行前替换为该目录的**绝对路径**。

`--run-dir` 统一传 `<项目根>/.agent_optimizer/runs/<run_id>`。

PD 分离场景先读 [PD 部署搜索约定](../references/pd.md)，其跨轮布局、预算和复测约定优先于本文通用收敛说明。

## 目标

用 AI agent 驱动服务化参数调优。Agent 负责读上下文、生成候选、分析结果和决策下轮方向；Python 脚本负责确定性的校验、执行、摘要汇总和结果导出。

## 阶段归属（三个阶段流程文件）

agent 分支的详细操作拆为三个阶段文件。这里只定义文件职责；执行顺序以「每轮循环」为准。

| 阶段 | 流程文件 | 覆盖内容 |
|---|---|---|
| Round 0 环境准备 | `./environment-preparation.md` | 环境与案例收集、方法卡核对、基础配置集中写入、上下文采集与配置检查、进入首轮优化 |
| Round N 数据分析与候选生成 | `./data-analysis.md` | 数据读取、knowledge 路由、方向与候选生成、候选校验、轮次总结、收敛检测 |
| Round N 执行验证 | `./execution-validation.md` | trial 执行、监控、结果回写、失败分类（Layer 1） |

总体顺序：

```text
Round 0: 环境与案例收集 → 方法卡核对 → TOML 写入与预检 → context（环境事实）→ 配置复检

Round N: analysis → knowledge routing → candidate validation → execution
                              ↓
              已有瓶颈证据 → 直接进入 knowledge routing
              证据不足     → diagnosis → knowledge routing
```

除 diagnosis 外，图中步骤均为必做；后续步骤不得在前置产物缺失时启动。

**阶段执行方式：优先 subagent**。当前平台具备 subagent 能力时，每个阶段（以及每轮的
"数据分析 → 执行验证"）应在独立 subagent 中执行——阶段产出全部落在 run 目录里
约定格式的 JSON 文件中，主 agent 只收发结构化结论，不把长日志和大文件读进主上下文：

- **输入**：run_id / run 目录绝对路径、config.toml 绝对路径、轮次 N、本阶段流程文件路径；
- **输出**：约定格式的 JSON 文件（schema 见 `../schemas/`）+ 返回主 agent 的简短结论
  （≤200 字：阶段状态、关键指标、是否卡住及原因），禁止回传原始日志；
- **主 agent 保留**：模式/预算等与用户的交互、轮次编排、收敛决策和 final_report。
- 平台不具备 subagent 能力时降级为当前上下文顺序执行，流程与产物不变。

中间文件的字段约束见 `../schemas/`（context / candidates / results / progress /
best-result）；`summary.md` 等 Markdown 是由这些 JSON 生成的
展示文件，不作判定依据。

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

## 每轮循环（Round N）

Round 0 完成环境准备后，Round N 按 A → B → C → D → E 执行；除 diagnosis 外不得跳过步骤。
详细操作按「阶段归属」加载对应文件。

### Round 0（一次性）

```text
环境与用户约束（含负载口径声明，S6）→ case-knowledge 案例收集 → 目标路由与读卡
→ 基础配置及运行节集中写入 → 配置预检 → collect_context → 配置复检
→ 噪声校准 → 维度索引初始化 → 进入 Round 1
```

Round 0 不建立搜索空间。配置或负载检查通过后，在 Round 1 步骤 B 由 agent 建立空间并生成候选。

**负载口径声明与瓶颈预警（S6，Round 0 强制）**
1. 启动前必须从用户/负载定义确认并写入 `summary.md` 的**口径指纹**：prefix cache 开关
   （`--enable-prefix-caching` / `--no-enable-prefix-caching`）、数据集前缀共享结构与设计命中率
   （如 pc0 = 内容互不相关、命中率 0；pc90 = 组内共享前缀占比 0.9）、并发口径、输入/输出长度、
   temperature。案例结论与历史负知识引用一律先匹配该指纹（见 `./data-analysis.md`「知识参考优先级」）。
   案例收集完成时（环境准备步骤 4）对 `knowledge/case-knowledge/cases/negative/` 做一次
   **负结果条目匹配**（`search_cases.py --negative --model <前缀>`）：命中条目按
   `./data-analysis.md`「知识参考优先级」第 6 条裁决——指纹一致 → 该参数默认排除出本 run
   空间（R1 确需触碰时在 rationale 给出与条目场景的差异依据，并按单变量对照设计）；
   指纹不一致 → 降级「仅参考」，不得作为排除依据。匹配命中与排除决定写入 `candidates_analysis`。
2. 基线部署完成后（Round 1 步骤 B 前）立即做一次瓶颈定位与 SLO 余量诊断，写进 `candidates_analysis`：
   ① prefill-bound 还是 decode-bound（对照 TTFT/TPOT 与 prefill 工作量估算）；② TTFT 余量 =
   (SLO − 实测)/SLO。**余量 <20% 且输入 ≥16K 时，必须显式向用户报告「负载口径（prefix cache 命中
   结构）可能是收益最大的优化方向，decode 侧优化空间正被 prefill 串行/SLO 限制」并给出口径调整建议**——
   不得默默接受口径继续只搜 decode 参数（实例：pc0 + 64K 输入下 TTFT 余量 13%，γ/capture 类
   decode 侧的全部优化空间被 P 侧串行 prefill 抵消，该 run 唯一有效增量是 γ=3；换 pc90 口径后同参数族
   连续兑现）。若用户确认口径不可变，在 summary 记录该约束，后续瓶颈结论须绑定此口径。

**噪声校准（S3，强制，完成前不得进 Round 1）**
同一次部署内用同一配置连测 3 次，把三个指标的相对标准差写入
run 目录的 `noise_band.json`（`context.json` 是框架独占资产，见 SKILL.md「PD 分离部署的强制约束」第 4 条；
历史 run 的 `context.noise_band` 仅作回退）：

```json
"noise_band": {"generate_speed_sigma": 0.0052, "tpot_sigma": 0.004, "ttft_sigma": 0.02,
               "samples": 3, "measured_round": 0}
```

- 没有这个数，任何 < 2σ 的差异都只能写"不可分辨"（`metrics_gate.py compare` 会这么判）。
- σ 的大小取决于硬件、模型与负载，不同场景差别很大，**只能用本 run 实测**，
  不能套用历史数值。校准成本是 3 次压测（不含部署），比后面为判不出效应而
  反复重测同一维度便宜得多。
- 校准值随负载形态变化即失效（换并发、换输入输出长度后重新校准），
  并在 `noise_band` 里记 `measured_round`。
- **σ > 5% 视为校准失败**（`metrics_gate.py compare` 会拒绝判定，退出码 2）：先检查
  是否混入不同配置/部署或 warmup 样本，重测后再进 Round 1。确属高噪声场景需要人工确认继续时，
  用 `compare --noise-ack-wide "<依据>"`，并把依据写进 summary——确认后若
  2σ ≥ 目标效应或所需样本量超预算，脚本给 `noise_dominant`/`unmeasurable`
  终止判定（退出码 4）：终止 run 并在 final_report 记录「噪声过大、不可判定」，
  不要再按 `n_required` 补测。

**无基线冷起点质量（S6）**：无基线/新模型分支下，起点不得用纯保守默认值拼装
（MNS 64 / GMU 0.85 / γ=0 / MNBT_d 512 一类——实测这类起点离最优远，且会多耗 2 轮以上
预算爬坡）。起点 = **安全上界探测起点**：MNS/GMU 按方法卡建议区间取上界（request-scheduling
§5.1 可行上界、memory-budget 0.9），模型自带 MTP 头时 γ=3 起步（mtp.md §5.2/§5.3）；
上界起点的 OOM 等风险由 Round 0 预检与首轮失败重试机制处理。基线候选只允许出现一次；同配置
重复压测仅用于 Round 0 噪声校准与漂移确认，不得占用搜索轮（见 `./data-analysis.md`
「每轮候选数量」）。

**维度索引初始化（S5）**：`check_dimensions.py --run-dir <run> --init-index` 生成
`dimension_index.json`，每个可动维度一条 `{"status": ..., "reason": ...}`；
`status` 只允许 `active` / `covered` / `retired`，标 `retired` 必须写 `reason`。
后续每轮结束后更新（`check_dimensions.py --run-dir <run>` 会指出未覆盖与超预算的维度，
单维度 trial 上限默认由 `budgets.max_trials ÷ 可动维度数` 推导）。

**预算区分（S5）**：`[agent_optimizer].time_limit_minutes` 是**单轮**上限。
Round 0 必须在 `summary.md` 顶部写下任务起始墙钟与总预算，把 baseline 部署
与每轮重部署（PD 场景下部署通常占墙钟一半以上）计入"已用"。

### Round N 主循环

| 步骤 | 必做内容 | 产物 / 进入条件 | 详情 |
|---|---|---|---|
| A | 读取 `context.json`、历史 `results`、`summary` 和配置文件 | 数据输入就绪 | `./data-analysis.md`「输入」 |
| B | 判断瓶颈；证据不足时执行 diagnosis；按 `goals → method-index → relationships → methods` 路由并读取命中方法卡；建立或更新 context.search_space（`config.toml` 已冻结，新增字段规则见 `../references/parameter-rules.md`「运行中新增字段」）并在 directions 中说明变更；生成候选并校验 | `candidates.round-N.json`（含三个分析字段与顶层 routing_review） | `./data-analysis.md`「本轮分析记录」「流程」 |
| C | 校验候选，执行 trial，监控并实时汇报；进程退出后归档本轮 context | `progress.round-N.json`、`results.round-N.json`、`context.round-N.json` | `./execution-validation.md` |
| D | 汇总本轮指标、失败和下轮方向 | `summary.md` | `./data-analysis.md`「流程」 |
| E | 执行收敛检测；未收敛则返回下一轮 A | 收敛报告或 Round N+1 | `./data-analysis.md`「收敛检测」 |

B 步骤固定顺序：`瓶颈证据 → [必要时 diagnosis] → goals → method-index → relationships → 方法卡 → directions 的三个分析字段 → 候选（同一 JSON 落盘）`。
缺少配置预检、目标路由、组合校验、方法卡读取或分析记录时，不得生成或执行候选。

## 汇报要求（主 agent 职责，全自动/交互式均强制）

**启动后汇报**：

```
🚀 Round N 已启动，M 个 trial 后台执行中。
   进度文件: .agent_optimizer/runs/<run_id>/progress.round-N.json
   我会在每个 trial 完成时实时汇报；整轮结束/崩溃也会收到通知。
```

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
   按同轮失败规则修复重试；达到失败上限则暂停
```

**C5：本轮成功并完成归档后的汇报模板**

```
✅ Round N 全部完成!
   最佳: rN-x (fitness=X, throughput=X tok/s, TPOT=Xms)
   [如有失败] 失败: rN-y [category/sub_category]

   📝 分析: [本轮关键发现，引用具体数据]
   → Round N+1 方向: [下轮方向简述，每个候选的选择依据]

   → 生成 Round N+1 候选中...
```

## 失败恢复（主 agent 职责）

单 trial 失败的处置由数据分析阶段完成（见 `./data-analysis.md`「生成前必须完成的检查」）；
失败分类（Layer 1）与其可信度判据见 `./execution-validation.md`「失败分类」。
本节规定 Layer 2 升级与整轮级恢复。

### Layer 2 — 拉起子 agent 深挖根因（仅兜底）

仅当满足下列触发条件之一时，主 agent 用 Agent 工具拉起子 agent 深挖根因：

- `failure.category == "unknown"`（正则未命中，是没见过的错误模式）
- 连续 2+ 个 trial 同类失败（分类虽对，但参数怎么避仍撞，根因没挖到）
- `failure.message` 无可用证据行（截断），分类不可信（判据见
  `./execution-validation.md`「失败分类」）

子 agent 的输入输出要求：

- **输入**：optix 已落盘的原始 log 文件路径（从 `failure` 字段或 run_dir 下日志
  找到，子 agent 自行读文件）+ 候选参数 + 已有 `failure` 分类（如有）。
  **禁止把原始 log 全文喂给主 agent。**
- **输出**：必须按下面的 JSON 结构返回，主 agent 收到后写回 `failure` 字段：

```json
{"root_cause": "<一句话根因>",
 "category_override": "<可空，覆盖 classify 的 category>",
 "offending_params": {"<PARAM>": "<value>"},
 "suggested_action": "<reduce_batch|increase_tp|fix_parallel|retry|...>",
 "confidence": "high|med|low",
 "evidence_excerpt": "<=200字，log 关键片段，截断>"}
```

- **`evidence_excerpt` 必须 ≤200 字**——这是子 agent 回传时唯一允许进入主 agent
  上下文的 log 内容，超过就违背了"不把原始日志读进主上下文"的设计目的。
- **症状模式优先匹配**：深挖前先对照
  `<SKILL_ROOT>/knowledge/modeling-diagnosis/symptom-patterns.md` 的症状模式
  （D1-D9）——命中则直接复用其诊断动作与处置规程（P1-P3，见
  `<SKILL_ROOT>/knowledge/modeling-diagnosis/procedures.md`），缩短定位链路；
  未命中再按上述通用契约独立排查。

### 同轮失败与重试计数

每轮只有一个候选。实际执行候选 trial 后无成功结果，计为一次失败尝试；执行前门禁拒绝不计。失败后保持 Round N，不创建 Round N+1。

1. 分析 `failure.category` 与错误证据
2. 若 category 为 `unknown` 或同类连续失败 → 触发 Layer 2 子 agent（见上）
3. 回退到最近成功的参数组合
4. 在允许调整的参数中选择修复方向，保持用户冻结项
5. 根因与余量判断参考：处置规程见
   `<SKILL_ROOT>/knowledge/modeling-diagnosis/procedures.md`（P1-P3）；结合实际配置、显存观测及方法卡确定降级方向
   （减 batch 还是提并行度），空间扩展须确认越界安全（见 `./data-analysis.md`
   「参数范围扩展」）
6. 首次失败后最多自动修复并重试一次；同轮两次执行均失败则暂停，不再自动重试。修复前先完成本次 context 归档；覆盖失败候选前，在新的 `candidates_analysis` 中保留上次失败摘要、修复依据及本轮执行次数（首次为 1，重试为 2）。
7. Agent 在重试前按 `./data-analysis.md`「时间预算」检查累计耗时，再重新校验候选；进程退出后按执行流程同步结果与 context 归档。恢复任务时核对候选分析中的执行次数、candidate_id 与结果；候选尚无对应结果时按执行流程恢复，不能把旧结果计为本次失败。失败计数不因重启 Agent 清零；已有两次失败时继续保持暂停，等待用户明确恢复。

达到两次失败时报告：

```
Round N 已连续执行失败 2 次（首次执行 + 1 次重试），自动恢复已暂停。建议检查:
  - 搜索空间是否有可行解
  - config.toml 配置是否正确
  - 硬件资源是否充足
  - （如仍有 unknown 类失败）人工查看 `<run-dir>/trial_logs/<candidate_id>.log`
    （optix 在每个 trial 删除原始日志前自动落盘，成功/失败各一份）
```

## 收敛与收尾（主 agent 职责）

- 收敛决策（趋势 + 覆盖 + **维度覆盖** 三条件判定）由主 agent 按 `./data-analysis.md`「收敛检测」执行。
- **宣称"已收敛"的前置门禁（S5）**：`check_dimensions.py --run-dir <run>` 必须 exit 0。
  未 exit 0 时只能写"预算耗尽/提前终止"，并列出未覆盖或超预算的维度，
  不得写"已收敛"。
- 收敛退出或跑满预算后执行：

  ```bash
  python <SKILL_ROOT>/scripts/export_best_config.py \
    --run-dir .agent_optimizer/runs/<run_id>
  ```

  多机混部输出 `serve_command_multihost.json`、`best_deployment/start_node.sh` 和各 worker 脚本，见 `../references/multihost.md`；final_report 展示节点对应关系和实际脚本。下面的单命令产物说明适用于单机。

  生成产物：
  - `best_result.json`：当前最优 trial
  - `optimizer_config_handoff.json`：给 `ms-serviceparam-optimizer-config` skill
    使用的推荐参数（含 `serve_command`；有环境覆盖时含 `serve_env`）
  - `serve_command.json`：最优配置渲染的 vLLM serve 命令（argv + command；有环境覆盖时含 `env`，null 表示删除变量）
  - `serve_command.sh`：可直接执行的 `vllm serve` 启动脚本（final_report 从它复制命令）

- **负结果沉淀（强制，紧随 export_best_config，在 final_report 之前）**：

  ```bash
  python <SKILL_ROOT>/scripts/export_negative_knowledge.py \
    --run-dir .agent_optimizer/runs/<run_id>
  ```

  从 `dimension_index.json` 的 retired 条目 + `results.round-*.json` 对照实测生成负知识
  条目，写入 `knowledge/case-knowledge/cases/negative/`（场景指纹 + 置信度 + 未验证场景，
  见 `../knowledge/case-knowledge/SKILL.md`「negative 分区」）。无 retired 条目时为 no-op；
  σ 缺失或幅度 < 2σ 时脚本默认跳过（`--force` 可入库但标 confidence: low）。禁止手工编造
  条目；导出摘要（写入/跳过的条目数）写入 final_report「失败统计」。

- Agent 生成 `final_report.md`，包含：
  - 基本信息（模型、引擎、硬件、轮次/trial 数、总耗时）
  - 最优结果（throughput、TPOT、对比基线提升）
  - 最优参数
  - **完整 vLLM serve 启动命令**（可直接执行，含所有搜索参数 + 引擎固定参数 +
    环境变量；来源：`serve_command.sh` / `optimizer_config_handoff.json` 的
    `serve_command` / `serve_env` 字段）
  - 每个关键参数的探索过程（引用具体轮次和数据）
  - 关键发现
  - 失败统计
  - **统计可信度声明（S1/S3 强制）**：
    ① `metrics_gate.py breakdown --run-dir <run>` 的四分量占比、`ρ(fitness, 吞吐)`
    与 `OK/WEAK/DEGRADED` 判定（判 DEGRADED 时必须写明"本轮 fitness 只反映吞吐"）；
    ② 每条"A 比 B 好 x%"的结论附带 `metrics_gate.py compare` 的
    `VERDICT`（`real_gain` / `within_noise` / `insufficient`）与 σ、n；
    ③ `noise_band.json` 的实际值（历史 run 的 `context.noise_band` 等效）；
    ④ 最优配置的采信数字取同配置复测样本的中位数，
    最优 trial 单轮值与复测中位数偏差 > 3σ 时判离群，按中位数报告并标注离群轮次。
    没有这四项的提升声明一律删除，改写为"不可分辨"。

  **生成约束**：
  1. 每个参数的"探索过程"节必须引用具体轮次和数据，禁止虚泛描述。读取
     `summary.md` 和 `results.round-*.json` 进行填充。
  2. 必须包含完整 vLLM serve 命令（从 `serve_command.sh` 复制，含
     `--tensor-parallel-size`/`--max-num-batched-tokens`/`--api-server-count` 等搜索
     参数、`--seed`/`--quantization` 等固定参数，以及前置 env）。命令必须是
     **可直接 `bash` 执行**的一版（JSON 容器值带引号），不要只贴参数表。
