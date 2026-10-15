# RFC: optix 寻优工具 Agent 模式（Agent-Driven Service Parameter Tuning）

Status (状态): Approved  
Author(s) (作者): @wendellX  
Created (创建日期): 2026-09-07  
Updated (更新日期): 2026-09-07  
Related Issue/PR (相关 Issue/PR): #778 (must be linked for background tracking / 必须关联 Issue/PR 以便追踪背景)

---

## 1. Overview (概述)

### 1.1 Summary (简介)

本 RFC 描述 msmodeling 寻优工具 optix 的 **Agent 模式**（AI-agent 驱动的服务化参数闭环调优）：由 AI agent 承担"读上下文 → 生成候选 → 分析结果 → 决策下轮方向"的智力环节，由 optix 执行层与配套脚本承担"确定性校验 → 真实/仿真压测 → 进度与结果持久化 → 收敛判定"的机械环节，两者通过 `candidates.round-N.json` / `results.round-N.json` / `progress.round-N.json` / `summary.md` 等文件契约协作。

### 1.2 Motivation (动机)

optix（`msmodeling optix`）原通过 PSO 粒子寻优在真实/仿真环境下自动调节 MindIE / vLLM 服务化参数。PSO 模式虽然全自动，但在以下场景存在结构性短板：

| 痛点 | 说明 |
|------|------|
| **黑盒不可解释** | PSO 的粒子更新不产生可读的"为什么调这个方向"的推理；用户难以判断收敛结果是否可信、难以在过程中干预 |
| **经验难注入** | 社区沉淀的模型族级规则（known_patterns）、官方推荐 preset、历史失败教训难以在 PSO 内部被结构化利用 |
| **失败难归因** | trial 失败时只有零散日志，缺少统一的失败分类（OOM / timeout / config 错误 / SLO 违例）与"下轮怎么避"的可执行建议 |
| **方向决策盲区** | 引擎级固定参数（如 preset 强制的 `enforce_eager`）是否是当前硬件次优值、并发维度是否被校准逻辑静默钉死——PSO 无感知 |

Agent 模式的价值：把"方向决策"交给可解释的 agent（携带经验报告、失败分类与各轮结构化 failure 证据），把"确定性执行"留在代码层（同一套校验/压测/收敛逻辑，agent 与 PSO 复用）。不做此模式，上述调优场景只能退回人工试参或不可解释的黑盒寻优。

### 1.3 Goals (目标)

**目标**：

- 提供 agent 驱动的多轮闭环调优：逐轮生成候选（含 rationale / expected_effect / risk）→ 执行压测 → 判优 → 收敛退出
- 候选与结果、进度以**文件契约**持久化（`candidates/results/progress/summary`），执行与决策解耦，支持中断续跑（resume）
- 收敛采用**趋势 + 搜索空间覆盖双门**：仅"连续 N 轮无改善"不足以收敛，还须满足最低轮数/trial 数，保证最低采样量
- 失败处理两层化：确定性正则分类（Layer 1）为主、子 agent 深挖（Layer 2）兜底，结构化失败信息随各轮 results 落盘，作为下轮候选生成时规避失败参数区域的依据
- 与 PSO 共享评估管线（fitness 判定、TP/并发钉死语义、上下文采集、经验注入），避免两套口径漂移

**非目标**：

- 不改变 optix 的 PSO 寻优能力与 `config.toml` 主流程（Agent 模式是其上的策略分支，`optimizer_strategy="agent"`）
- 不做参数组合的效果预测（候选的 expected_effect 是方向性估计，非保证）
- 不取代 benchmark / 仿真执行器本身（沿用 vllm_benchmark / ais_bench 与既有 scheduler）
- 不支持无模型无硬件的纯离线寻优（Agent 模式需真实或仿真压测闭环）

# 2. Use Case Analysis (用例分析)

**用例 U1 — 可解释的多轮调优（主用例）**：用户希望了解每个候选为何被选中、每轮探索了什么方向、为什么收敛。Agent 每轮生成候选时携带 rationale 并引用经验报告/历史数据；每轮结束输出总结与下轮方向；收敛后产出含完整 vLLM serve 命令的 final_report。

- 功能点：模式选择（全自动/交互式）→ 配置就绪门禁 → Round 0 上下文采集与经验注入 → 每轮 候选生成 → 后台执行 → 实时汇报 → 总结 → 收敛检测
- 关键性能指标：每轮固定 1 个候选；收敛默认触发条件为连续 6 轮改善 <1% 且覆盖门满足；时间预算上限 1440 分钟可配
- DFX 要求：可解释性（每个候选有 rationale / expected_effect / risk；final_report 引用具体轮次数据）；可干预性（交互式模式下用户可确认/修改/跳过候选）；可靠性（执行中断可 resume，全败轮自动重跑）

**用例 U2 — 失败导向的调优**：trial 失败（OOM / timeout / 启动期 config 错误）时，执行层产出结构化 `failure`（category / sub_category / message / evidence_line / suggested_action / offending_params），agent 依据它在生成下轮候选时避开失败参数区域。

- 功能点：失败确定性分类（正则表）→ 基于各轮 results 的 failure 证据规避失败参数区域 → 连续同类失败或 unknown 时升级子 agent 深挖 → 同轮失败修复重试（每轮固定 1 候选，同轮最多自动修复重试一次，两次执行均失败即暂停等用户介入，见 `workflows/agent.md`「同轮失败与重试计数」与 `workflows/data-analysis.md`）
- DFX 要求：失败证据可审计（evidence_line 指向日志命中行；failure.message 强制保留尾部异常行）；子 agent 回传 evidence_excerpt ≤200 字防上下文爆炸

**用例 U3 — 方法卡指导参数选择**：Round 0 采集环境事实与已确认负载；agent 按目标路由读取方法卡、参考匹配的官方案例，基础配置登记到 config.toml；首次优化时建立 context.search_space，后续逐轮维护参数定义和默认值，候选只写取值。脚本在执行前校验；每轮进程退出后归档完整 context。

- 功能点：经验报告强制生成与必读 → vendor preset 命中后应用 launch env / additional_config → 每轮候选交叉引用经验报告
- 安全/隐私要求：本地运行，不向外部服务上传模型路径/配置/日志
- DFX 要求：单一事实源（经验报告为 Round 0 产物，后续轮不重复采集）；幂等（apply_launch_config 可重复执行；env 残留用 unset_env.sh 清理）

**用例 U4 — 跨轮状态保持与续跑**：多轮寻优可中断（断电/人为停止）后从已完成的轮次继续，不重跑已完成 trial。

- 功能点：`AgentOptimizer._scan_completed_rounds` 扫描 results.round-N.json 识别已完成轮 → baseline.json 检查点加载（跳过重复 baseline 实测）→ 从下一轮继续
- DFX 要求：全败轮不计入完成轮（恢复后自动重跑）；基线实测结果落盘为 checkpoint，避免重复冷启动

# 3. Design (方案设计)

## 3.1 Overall Design (总体方案)

### 3.1.1 系统架构

Agent 模式采用"**决策/执行分层**"架构，自下而上分三层：

```text
┌──────────────────────────────────────────────────────────────────┐
│ 用户（自然语言交互：全自动 / 交互式模式选择、候选确认、进度询问）     │
└───────────────────────────────┬──────────────────────────────────┘
                                │
┌───────────────────────────────▼──────────────────────────────────┐
│ 入口层：optix-assistant skill（.agents/skills/optix-assistant/）  │
│   SKILL.md ─ 三模式路由（config / PSO / agent）                    │
│   workflows/agent.md ─ agent 执行流程（Round 0 → 每轮 A-E）        │
│   workflows/data-analysis.md ─ 候选与判优规则                    │
│   scripts/ ─ collect_context / config_writer /                    │
│               check_search_space / config_preflight /             │
│               summarize_history / export_best_config / estimation │
└───────────────────────────────┬──────────────────────────────────┘
                                │ 文件契约：candidates/results/progress/
                                │ context.json / summary.md / baseline.json
┌───────────────────────────────▼──────────────────────────────────┐
│ 执行层：optix/optimizer/agentic/（optimizer_strategy="agent"）     │
│   orchestrator.py  AgentOptimizer ─ 跨轮编排/resume/并发钉死/双门收敛│
│   strategies.py    AgentCandidateStrategy ─ 单轮候选执行与判优      │
│   validation.py    validate_candidate + classify_failure(失败分类)  │
│   coverage.py      search_space_coverage(收敛覆盖门)               │
│   candidates.py    Candidate/TrialResult/TrialFailure 数据契约     │
└───────────────────────────────┬──────────────────────────────────┘
                                │ 复用 PSO 评估管线（fitness/scheduler/target_field）
┌───────────────────────────────▼──────────────────────────────────┐
│ 评测层：vllm_benchmark / ais_bench（真实或仿真压测）                 │
└──────────────────────────────────────────────────────────────────┘
```

**决策/执行职责划分**（本 RFC 核心设计决策）：

| 环节 | 归属 | 理由 |
|---|---|---|
| 意图识别 / 模式选择 / 候选确认 | Agent（skill 层） | 需自然语言与用户意图理解 |
| 候选生成（方向/数值/rationale） | Agent（skill 层） | 需综合方法卡、历史与结构化失败证据做可解释决策 |
| 上下文采集 / 配置写入 / 参数检查 | 脚本（skill scripts/） | 确定性规则，脚本比 agent 更可靠可测 |
| 候选校验（合法域/约束/运行时 flag） | `validation.py` | 单一事实源，防 agent 与执行层规则漂移 |
| 候选执行 / 进度写盘 / 结果落盘 | `strategies.py` + scheduler | 机械执行，确定性 |
| 失败分类（确定性部分） | `validation.py::classify_failure` | 正则表驱动，Layer 1 无 agent 成本 |
| 失败深挖（未知/连续同类失败） | 子 agent（仅 Layer 2 兜底） | 需读原始 log 推断根因，但证据回传受限 |
| 收敛判定 | `orchestrator.py`（趋势+覆盖双门） | 确定性判据，不受 agent 状态影响 |

### 3.1.2 文件契约（run 目录数据流）

run 目录统一为 `<项目根>/.agent_optimizer/runs/<run_id>`，round 数据文件命名带轮号：

| 文件 | 写入方 | 消费方 | 内容 |
|---|---|---|---|
| `context.json` | collect_context 采集事实、agent 优化时建立和维护空间 | orchestrator / agent 每轮 | 环境事实、负载、约束、model_info、source.config_toml、search_space |
| `context.round-N.json` | Agent（skill 层） | 历史分析、export_best_config.py | 当轮完整 context 归档；成功轮归档保留不覆盖 |
| `candidates.round-N.json` | Agent（skill 层） | strategies.py | 候选：candidate_id / params（纯值）/ rationale / expected_effect / risk |
| `results.round-N.json` | strategies.py | agent / summarize_history / orchestrator | trial 结果 + validation + 结构化 failure |
| `progress.round-N.json` | strategies.py | agent（Monitor 实时汇报） | status/stage 状态机 + 每 trial 指标 |
| `summary.md` | summarize_history.py（累积追加） | agent（每轮方向决策） | 各轮摘要、趋势、失败分析 |
| `baseline.json` | orchestrator（_ensure_baseline） | orchestrator（resume 时加载） | 基线实测 checkpoint |

**候选执行前置条件**：`optimizer_strategy="agent"` 已开启（optix 无 `--agent-mode` 标志，由该配置项驱动）；`[agent_optimizer]` 的 run_id/run_dir 与 agent run 目录对齐；`use_request_rate_calibration=false`（否则候选 CONCURRENCY 被钉死为 max，并发维度静默失效）。仅当 `candidates.round-N+1.json` 尚未生成时，本轮执行完即退出（否则继续跑下一轮）。

### 3.1.3 核心运行流程

```text
配置就绪（config_preflight 门禁：模型路径/SLO/benchmark/agent 模式/run_id 对齐）
   ↓
Round 0：
   1. collect_context → context.json（环境事实、负载和配置来源，不初始化搜索空间）
   2. Agent 按目标路由读取方法卡 → config_writer 登记参数
   3. case-knowledge 检索官方案例，核对并登记适用配置
   4. check_search_space → checklist_report.md；config_preflight 复检
   5. Agent 生成 candidates.round-1.json（基线，含方向分析与证据引用）
   ↓
每轮循环 Round N（A-E）：
   A. 读取数据源（context/config.toml/summary/results/方法卡）
   B. Agent 更新 context.search_space，在 candidates_analysis 说明变更，生成 candidates.round-N.json（每轮 1 个候选）→ 校验 → 表格展示
   C. optix agent 模式后台执行：strategies 校验候选→逐 trial 冷启动→压测→写 progress/results
      （config_preflight --check-only 门禁退出 0 才启动；ais_bench 先 --search 预检）
   D. summarize_history 追加 summary.md
   E. 收敛检测：趋势条件（连续 convergence_rounds 轮改善<1%）∧ 覆盖条件（满足 min_rounds/min_trials）→ 双门满足才收敛
   ↓（跑满 max_rounds 或 time_limit 自然终止）
完成：export_best_config → best_result.json / optimizer_config_handoff.json /
      serve_command.json / serve_command.sh → Agent 生成 final_report.md（含完整 serve 命令）
```

### 3.1.4 失败处理两层架构

```text
Layer 1（默认，确定性，无 agent 成本）：
  执行层失败 → classify_failure（FAILURE_CLASSIFICATION_TABLE 正则逐行匹配，尾部异常行优先）
  → failure 结构体 {category/sub_category/message/evidence_line/suggested_action/offending_params}
  → 主 agent 读 failure.suggested_action，生成下轮候选时避开失败参数区域

Layer 2（子 agent 深挖，仅下列触发条件）：
  ① failure.category == "unknown"（没见过的错误模式）
  ② 连续 2+ trial 同类失败（分类对但方向仍撞，根因未挖到）
  ③ failure.message 无可用证据行（分类不可信）
  子 agent 输入 = 原始 log 文件路径（非全文），输出 = 强制 schema
  （root_cause / category_override / offending_params / suggested_action /
    confidence / evidence_excerpt≤200 字）
```

### 3.1.5 收敛双门判定

| 门 | 判定 | 默认参数（config.toml `[agent_optimizer]`） |
|---|---|---|
| 趋势门 | 连续 `convergence_rounds` 轮改善 < 1%（SLO 达标优先排序，fitness 仅兜底） | convergence_rounds=6 |
| 覆盖门 | 已执行轮 ≥ `min_rounds` ∧ 已完成 trial ≥ `min_trials` | min_rounds=4, min_trials=4（每轮 1 trial，两门等价，取较大者生效） |

覆盖门要求足够的轮次与 trial，避免仅凭少量结果判断收敛。只有双门都满足才 `convergence` 退出；跑满 `max_rounds`（默认 32）自然终止；`time_limit_minutes`（默认 1440，≤1440 校验）限制每次单轮执行，OptiX 在下一 trial 启动前检查期限，当前 trial 完成后才停止。Agent 在新轮、重试和恢复前，对各轮进度文件的 `cumulative_elapsed_s` 求和，管理全局 trial 时间预算；该字段包含成功、失败及重试的已完成 trial 耗时，不包含 baseline、分析或等待时间。预算耗尽时，Agent 执行导出与最终报告。

## 3.2 Security, Privacy, and DFX Design (安全隐私与DFX设计)

### 3.2.1 兼容性

- **配置兼容**：`optimizer_strategy="agent"` 与既有 `optimizer_strategy="pso"` 并存于同一 config 模型；`[agent_optimizer]` 为新增可选节，不配置时回退 PSO 行为
- **评估管线兼容**：Agent 模式复用 PSO 的 scheduler / fitness / benchmark 插件，无第二套评测实现
- **脚本兼容**：config_writer / config_preflight 的门禁与写入职责分层（config 模式不执行 preflight；agent 模式专属门禁），与三模式路由共存

### 3.2.2 可维护性

- **单一事实源**：环境与负载（context.json）、基础参数定义（config.toml target_field）、本轮有效定义与默认值（context.search_space）、历史输入（context.round-N.json，进程退出后复制 context.json 归档）、收敛预算数值（config.toml `[agent_optimizer]`，文档不背书旧值）、候选校验规则（validation.py 单一宿主，skill 与执行层共享判定，改一处两处生效）
- **估算物理量单一宿主**：`estimation.py`（dtype_bytes / per_token_kv_bytes / estimate_weight_gb / kv_budget_gb / tp_candidates），供 recommend_params 使用，agent 候选域由本轮 context.search_space 定义，并与 TOML 基础域保持一致
- **文档分层**：SKILL.md 纯路由（不混执行细节）；workflows/agent.md 管流程；workflows/data-analysis.md 与 execution-validation.md 管候选、判优和失败处理规则

### 3.2.3 可测试性

- agentic 执行层为纯 Python 模块，可脱离 LLM 单测：coverage.py 明确标注"stdlib-only 以便密封单测"；candidates/validation 的校验与失败分类均有测试覆盖（test_custom_command 等）
- 跨轮编排（AgentOptimizer）可注入临时 run_dir 构造，便于测试 resume/收敛路径
- estimation.py 配套 test_estimation.py（dtype 映射/GQA/MLA/weight 单位/KV 预算/TP 候选/paged 分档）

### 3.2.4 可靠性

- **resume**：已完成轮次扫描 + baseline checkpoint；全败轮不计入完成轮，恢复后自动重跑
- **收敛检查**：同时满足改善趋势与最低轮数/trial 数要求
- **防并发静默失效**：`use_request_rate_calibration=false` 门禁 + 钉死告警回调（on_pin_concurrency）
- **失败证据保尾**：classify_failure 按尾部异常行优先匹配，message 强制保留最后异常行
- **超时自愈**：trial 冷启动 5-10 分钟为正常；连续 30 分钟无 stage 变化才告警"疑似卡死"；时间预算耗尽优雅收尾

### 3.2.5 安全与隐私

- 全部本地执行；不上传模型路径、config、日志、压测数据至外部服务
- 子 agent 回传主 agent 的 log 内容仅限 evidence_excerpt ≤200 字（防上下文爆炸与敏感信息扩散）
- 原始 log 由 optix scheduler 持久化于 run_dir 本地，供审计与人工排查

## 3.3 Programming and Integration Design (编程与调用设计)

### 3.3.1 Basic Programming Model Design (编程模型基本设计)

**运行环境**：Python 3.10+（optix 运行环境）、NPU/GPU 服务器（真实压测）或仿真模式；入口命令 `msmodeling optix`。Agent 模式由 skill（optix-assistant）承载决策，optix CLI 承载执行：

```bash
# 执行层入口（skill 层经 config_preflight 门禁后调用）
msmodeling optix -e <engine> -b <benchmark> -c <config.toml>   # optimizer_strategy="agent" 时走 agentic
```

**数据契约**：决策层（skill/agent）与执行层（optix）之间只通过 run_dir 下的 JSON/MD 文件交互（见 3.1.2），无进程内 API 耦合——skill 是独立仓库目录的脚本集合，optix 是 Python 包，两者经文件契约解耦，可独立演进与测试。

**开发约束**：

- 候选文件必须符合 `candidates.round-N.json` 命名与字段结构（candidate_id/params/rationale/expected_effect/risk）
- 候选 params 必须是 search_space 内合法参数，非法候选由执行层校验自动跳过（不中断整轮）
- `use_request_rate_calibration` 必须为 false 才能测并发拐点；需按序 source/unset launch_env.sh 防 env 残留
- 禁止手写 TOML：搜索参数经 config_writer 写，运行节经 config_preflight `--set-*` 写

**可验收设计**：功能验收 = Agent 模式跑通至少一轮并写回 results（SKILL.md 完成标准）；行为验收 = 候选校验/失败分类/收敛双门的确定性部分可被单测锁定，不随 LLM 输出变化。

### 3.3.2 API Definition and Design (接口定义与设计)

Agent 模式对外不新增 Python API，能力经 CLI 配置项与 skill 脚本暴露。主要"接口"如下。

#### 3.3.2.1 optimizer_strategy（config.toml 顶层配置）

* **接口描述**：选择寻优策略；`"agent"` 时 optimizer.py 的 `_run_optimizer` 走 `AgentOptimizer` 跨轮编排，否则走 PSO run 插件。
* **接口原型**：`optimizer_strategy: Literal["pso", "agent"]`（config.toml 顶层）
* **输入/输出参数**：

| Parameter Name (参数名称) | Input/Output (输入/输出) | Type (类型) | Description (描述) | Value Range (取值范围) |
| --- | --- | --- | --- | --- |
| optimizer_strategy | Input | string | 寻优策略 | `"pso"` / `"agent"` |

* **异常处理**：非合法值由 config 模型 Literal 校验拒绝（PR#778 意见 #4）
* **约束说明**：agent 模式依赖 run_dir 下 candidates.round-N.json 存在；无候选文件时 orchestrator 告警并停止
* **变更说明**：既有 PSO 配置不受影响（默认仍为 pso 行为）

#### 3.3.2.2 [agent_optimizer] 配置节（AgentOptimizerConfig）

* **接口描述**：Agent 模式运行参数（跨轮编排读取），见 `optix/config/config.py::AgentOptimizerConfig`。
* **接口原型**：TOML 节 `[agent_optimizer]`
* **输入/输出参数**：

| Parameter Name (参数名称) | Input/Output (输入/输出) | Type (类型) | Description (描述) | Value Range (取值范围) |
| --- | --- | --- | --- | --- |
| run_id | Input | string | run 标识（内部，用户无需提供） | 任意非空串；占位 `"default"` 会被 `--set-agent-mode` 自动替换 |
| max_rounds | Input | int | 最大轮数（自然终止上限） | 默认 32 |
| max_trials | Input | int | 单轮最大 trial 数 | 默认 12 |
| time_limit_minutes | Input | int | 单轮执行时间上限（≤1440 校验；全局预算由 agent 跨轮累计） | 默认 1440 |
| run_dir | Input | string | run 目录；留空自动解析 `.agent_optimizer/runs/<run_id>` | 默认 "" |
| convergence_rounds | Input | int | 趋势门：连续无改善轮数阈值 | 默认 6 |
| min_rounds | Input | int | 覆盖门：最小已执行轮数 | 默认 4 |
| min_trials | Input | int | 覆盖门：最小已完成 trial 数（每轮 1 trial，与 min_rounds 等价，取较大者生效） | 默认 4 |

* **异常处理**：time_limit_minutes > 1440 由 config_preflight 拒绝
* **约束说明**：run_id 须与 collect_context `--run-id` 对齐；run_dir 缺省时由 orchestrator 按 run_id 自动解析
* **变更说明**：该节为新增可选配置；缺省整节时用 pydantic 默认值

#### 3.3.2.3 run 目录文件契约（candidates / results / progress）

* **接口描述**：决策层（agent）与执行层（strategies.py）之间的数据交换格式。
* **接口原型**：JSON 文件（命名带轮号，见 3.1.2）
* **输入/输出参数**：

| Parameter Name (参数名称) | Input/Output (输入/输出) | Type (类型) | Description (描述) | Value Range (取值范围) |
| --- | --- | --- | --- | --- |
| candidates.round-N.json | Input（执行层读） | JSON 数组 | 候选列表；字段 candidate_id/params/rationale/expected_effect/risk/source | candidate_id 形如 rN-a |
| results.round-N.json | Output（执行层写） | JSON | trial 结果 + validations + failure | — |
| progress.round-N.json | Output（执行层写） | JSON | status/stage 状态机 + per-trial 指标 | status ∈ starting/running/completed；stage ∈ preparing/trial_cold_start/trial_completed/round_done |

* **异常处理**：无效候选不中断整轮（validation 过滤后跳过）；results 内 failure 结构化（无原始 log 内联）
* **约束说明**：progress 首 trial 前即写盘（防冷启动期无信号）；最终状态保留（不 unlink），Monitor 据此区分"完成"与"卡死"
* **变更说明**：见 3.1.2

### 3.3.3 Usage Instructions (使用说明)

**使用入口**（skill 层，用户视角）：

1. 发起"agent 模式寻优"意图 → 入口 skill 判定模式 → 全自动/交互式二选一（禁止默认 autonomous）
2. 配置就绪：回答模型路径/服务模型名/卡数/benchmark/输入输出长度/SLO 目标（必问禁猜）→ config_preflight 门禁回填
3. 按 `workflows/agent.md` Round 0 → 每轮 A-E 流转；交互式模式下每轮候选以表格展示（rationale 在上、参数表在下）供确认/修改/跳过
4. 收敛或跑满预算 → export_best_config + final_report（含可直接 bash 执行的 serve 命令）

**约束与限制**：

- agent 模式与 PSO 模式互斥（同一 config.toml 一次一种 optimizer_strategy）
- 依赖候选文件的生成节奏：本轮执行完仅当下一轮候选文件已生成才继续（决策-执行解耦的边界）
- 交互式模式每轮需用户介入，不适合无人值守；无人值守用全自动
- `time_limit_minutes` ≤ 1440；收敛参数改 config.toml `[agent_optimizer]`，不写死在文档

# 4. Test Design (测试设计)

| 层级 | 覆盖对象 | 用例/方式 |
|---|---|---|
| 单元测试 | agentic/validation.py | validate_candidate 合法/非法域；classify_failure 失败分类（spec_config/oom/timeout 等）；派生类型契约（DERIVED_CANDIDATE_SETTABLE / DERIVED_ENGINE_ONLY，意见 #2） |
| 单元测试 | agentic/coverage.py | search_space_coverage：min_rounds/min_trials 未满足时不 ok，均满足才 ok（stdlib-only 密封单测） |
| 单元测试 | agentic/orchestrator.py | AgentOptimizer：resume 扫描（全败轮不计完成轮）、baseline checkpoint 加载/重跑、并发钉死告警回调（意见 #9/#12） |
| 单元测试 | config 层 | custom_command 渲染契约（VLLM_SERVE_CAPACITY_ENV_FIELDS 缺字段告警，意见 #14）；_update_ratio_field target 缺失两形态（意见 #13）；AgentOptimizerConfig 默认值 |
| 单元测试 | estimation.py | dtype_bytes 映射（fp8/float8/int4/int8/bf16/fp32）、GQA 分摊/MQA 复制/MLA latent、weight 单位 1e9、KV 预算、TP 候选、paged 分档（意见 #10） |
| 集成验证 | skill scripts | collect_context → config_writer → check_search_space → config_preflight 门禁链（--check-only 退出码） |
| 端到端验证 | 全链路 | 真机 3 轮 12 trial 实测（run-id run-20260901-qwen35 已跑通）；候选生成→执行→results 写回→收敛 |

# 5. Drawbacks and Risks (Optional) (缺点和风险)

| 风险 | 影响 | 应对 |
|---|---|---|
| Agent 决策质量依赖 LLM | 方向建议可能次优；LLM 输出不可复现 | 决策与执行分层：所有确定性环节（校验/执行/收敛/失败分类）代码化并有测试；agent 只做方向决策且每步留 rationale 供审计 |
| 每轮候选依赖 agent 生成文件 | agent 停止则寻优停止 | 支持 resume（已完成轮不重跑）；文件契约使中断可续 |

# 6. Existing Technology (Optional) (现有技术)

- **Optuna / Hyperopt 类自动调参框架**：本方案借鉴其"优化器与目标解耦、多轮 trial 管理"思想；差异在于本方案的方向决策由**可解释 LLM agent** 承担（可输出 rationale），而非纯采样/TPE 代理模型，且面向的是真实服务压测（每 trial 分钟级成本），因此额外引入失败分类与收敛覆盖门来控制昂贵 trial 的浪费。
- **vLLM 官方参数指南 / Ascend 官方 preset**：本方案的 known_patterns 与 vendor_preset 是其 skill 化、可编程化表达（单一数据源 JSON），并在 Round 1 作为基线种子注入，区别于静态文档。
- **LLM-as-optimizer（如 OPRO 等论文范式）**：方向一致（LLM 生成候选并反思）；差异在于本方案非研究原型——机械环节（校验/执行/收敛）全部下沉代码层并有单测，LLM 只做有边界的方向决策，且失败有结构化分类与审计证据。

---

Appendix (附录)

* **References (参考资料链接)**
  * [optix-assistant SKILL.md](../../.agents/skills/optix-assistant/SKILL.md)（寻优统一入口与三模式路由）
  * [agent 模式执行流程](../../.agents/skills/optix-assistant/workflows/agent.md)（Round 0 → 每轮 A-E）
  * [PSO 模式文档](../../.agents/skills/optix-assistant/references/pso-mode.md)
  * 执行层源码：`optix/optimizer/agentic/`（orchestrator / strategies / validation / coverage / candidates）
  * 配置模型：`optix/config/config.py::AgentOptimizerConfig`
  * [estimation 物理量单一宿主](../../.agents/skills/optix-assistant/scripts/estimation.py)（PR#778 意见 #10）
* **Glossary (术语表)**
  * **PSO**：粒子群寻优（optix 内置全自动寻优策略）
  * **candidate / trial / round**：候选（一轮内一个参数组合）/ trial（一次候选的完整执行与评测）/ round（一轮候选集合的执行）
  * **vendor_preset**：vLLM Ascend 官方推荐配置（presets/ascend_vllm_presets.json），命中场景作为 Round 1 基线与 strong_suggest 默认
  * **known_patterns**：人类维护的模型族级规则（knowledge/known_patterns.json）
  * **SLO**：TTFT/TPOT 时延上限（硬约束，判优第一排序键）
* **Documentation Update Plan (文档更新计划)**
  * 本文档为 Agent 模式全链路设计的权威描述；SKILL.md / workflows/agent.md 为运行期行为文档（入口/流程），不重复承载设计决策
  * 收敛与预算数值的运行时真源 = config.toml `[agent_optimizer]`，本 RFC 与行为文档均不背书硬编码旧值
  * 新引擎级固定参数/新失败分类模式沉淀时，同步更新本 RFC 与相关行为文档
