# Agent 模式行为契约（候选 / 判优 / 失败规则）

> **本文件 = agent 分支的行为契约（判断标准与禁止规则）**，与 `../workflows/agent.md`（执行流程）配套，
> 两文件一起构成原 agent-mode.md 全文；进入 agent 分支前必须完整阅读两者，禁止只读目录就开跑。

> 主体自 `agent-optimizer/SKILL.md` 全文平移（该 skill 已冻结）。差异项（路径前缀替换、
> 写入分工、assistant_handoff 引用、措辞适配与源文档笔误修正）均已处理并 diff 校验，
> 除差异项外与源文档逐字一致。

> **运行时依赖（已合入并验证）**：本契约对应的执行环节（`optimizer_strategy="agent"`、config.toml
> `[agent_optimizer]` 配置节、candidates/results/progress 文件、收敛状态机）依赖 optix 的
> agent 模式实现，已在 msmodeling-x 完整跑通（3 轮 12 trial 实测，run-id `run-20260901-qwen35`）。

## 候选生成规则

### 生成前必须完成的检查

1. **交叉比对 knowledge 与 search_space**：每条 known_patterns 建议的参数是否存在于 search_space 中。如有 gap，报告用户。
2. **并行策略覆盖**（MoE 模型）：search_space 需包含 EP/DP/TP 中至少一种可调维度。
3. **读取上一轮 failure 字段**：提取 `offending_params`，更新 failing_params 黑名单。
4. **引擎级固定参数反转验证**：读取 `context.engine_fixed_params`，凡
   `flip_required=true` 且 `verified=false` 的参数，本轮候选必须覆盖相反值
   （单变量反转）；未覆盖则在下轮方向建议中说明原因。

### 黑名单机制

Agent 维护 `failing_params` 列表，记入内存（跨轮持久）：

```
{param_name: str, direction: str, threshold: Any, reason: str}
```

示例：
- `{param_name: "CONCURRENCY", direction: "gt", threshold: 128, reason: "r1-c TTFT=39.56s"}`
- `{param_name: "MAX_NUM_BATCHED_TOKENS", direction: "gt", threshold: 98304, reason: "r2-b OOM/kv_cache"}`

生成下轮候选时自动过滤所有命中黑名单的参数组合（参数值在 threshold ±20% 范围内视为命中）。

### 知识参考优先级

1. `do_not_tune` — 参数固定，不得调整
2. `strong_suggest` — 默认启用，可调但需 rationale
3. `explore` — 优先探索方向
4. `informational` — 仅供参考

**vendor_preset（官方推荐）**：命中 `knowledge.vendor_preset` 时，其 `params` 视为 strong_suggest 级别 —— Round 1 基线取官方推荐值；官方值只作起点，实测（results.round-*.json）优先于官方先验。`source_url` 用于 final_report 溯源。

**hardware 是参考不是门槛**：presets 的 `hardware` 是官方推荐硬件，卡数不匹配**不阻断注入**（tp/dp 本就在搜索空间不会被覆盖，其余推荐值跨卡数可用）。`knowledge.vendor_preset.hardware` 里 `matched=False` 表示官方在推荐硬件上验证、你实际卡数不同 —— 并行参数（tp/dp）按你的 world_size 调整，其余推荐值仍可作种子。

### 候选格式

```json
{
  "round": 1,
  "candidates": [
    {
      "candidate_id": "r1-a",
      "params": {"MAX_NUM_BATCHED_TOKENS": 65536, "CONCURRENCY": 64},
      "rationale": "基线。引用 summary.md 数据和黑名单约束。",
      "expected_effect": "吞吐 ~1000 tok/s",
      "risk": "低"
    }
  ]
}
```

### 动态候选数量

| 上轮情况 | 候选数量 |
|---------|---------|
| 首轮 | 4 |
| 改善明显（throughput +5% 以上） | 3-4（集中深挖） |
| 轻微改善或持平 | 4 |
| 无改善或倒退 | 5-6（扩大探索） |

---

## fitness 公式解读（判优前必读）

optix 的 fitness 是**软约束加权标量**（`optix/optimizer/performance_tunner.py`），
越小越优：

```text
fitness = 0.4 * (gen_speed_target / speed)         # 吞吐项，量级通常 2-13
        + 0.2 * exp(ttft_penalty * (TTFT/slo - 1)) # TTFT 惩罚，ttft_penalty=0 时恒为 0.2
        + 0.3 * exp(tpot_penalty * (TPOT/slo - 1)) # TPOT 惩罚，边界附近 ≈0.3
        + 0.1 * exp(success_rate_penalty * (1/success_rate - 1))
```

**何时 fitness 排名不可信**：
- `ttft_penalty=0` 时 TTFT 项是常数，fitness 对 TTFT 完全无感知；
- 吞吐项量级（`0.4*5300/speed`，speed 400-700 时约 2.9-5.3）远大于 SLO 惩罚项时，
  高吞吐但 TPOT 违例的候选会拿到更优 fitness —— 这是"软约束"的固有缺陷；
- 因此**硬约束场景（TPOT/TTFT 有明确上限）必须以 `slo_violated` 为第一排序键**：
  本轮最佳、`export_best_config.py`、汇总报告都优先"达标 + 吞吐最大"，
  纯 fitness 只作兜底（无任何达标候选时）。

**引擎级固定参数与"物理极限"声明**：任何"已达物理极限"的结论，必须先声明
`engine_fixed_params` 中每个 `flip_required` 参数已验证（反转值已测过），
否则一律视为"受预设默认值影响的暂态上界"。

---

## 失败处理

### 失败分类（两层）

**Layer 1 — 确定性分类（默认，无 agent 成本）**：执行方（`msmodeling optix` agent 模式的 `strategies.py`）失败时已调 `validation.py::classify_failure`，用正则表 `FAILURE_CLASSIFICATION_TABLE` 产出 `failure` 结构体（`category/sub_category/message/evidence_line/suggested_action/offending_params`）。常见失败（OOM/timeout/segfault/connection_refused/SLO 违规）在此层即可定，主 agent 直接读 `failure.suggested_action` 更新黑名单。

> **分类可信度判据（实测沉淀）**：`failure.message` 若无可读的异常行（日志 tail 被截断到几乎为空，仅剩片段），**一律视为分类不可信**，直接触发 Layer 2 深挖——不要基于残缺证据的方向（如 `oom/weight`）做黑名单决策。已修复版本中，分类改为按**尾部异常行优先**逐行匹配，`failure.message` 分级截断、强制保留最后一行异常，并附 `evidence_line`（命中的那一行）；若看到 `category="oom"` 且 trial 秒退（~18s 内、服务从未就绪），优先怀疑是 speculative-config 等启动期配置错误而非真实 OOM。

**Layer 2 — 子 agent 升级路径（仅兜底）**：仅当满足下列触发条件之一时，主 agent 用 Agent 工具拉起子 agent 深挖根因：

- `failure.category == "unknown"`（正则未命中，是没见过的错误模式）
- 连续 2+ 个 trial 同类失败（分类虽对，但参数怎么避仍撞，根因没挖到）
- `failure.message` 无可用证据行（截断），分类不可信（见上）

子 agent 契约：
- **输入**：执行方持久化的原始 log 文件路径（由 `failure` 关联或 run_dir 下日志定位，子 agent 自行读盘）+ 候选参数 + 已有 `failure` 分类（如有）。**禁止把原始 log 全文喂给主 agent。**
- **输出**：强制 schema，由主 agent 吸收后落回 `failure` 字段：

```json
{"root_cause": "<一句话根因>",
 "category_override": "<可空，覆盖 classify 的 category>",
 "offending_params": {"<PARAM>": "<value>"},
 "suggested_action": "<reduce_batch|increase_tp|fix_parallel|retry|...>",
 "confidence": "high|med|low",
 "evidence_excerpt": "<=200字，log 关键片段，截断>"}
```

- **`evidence_excerpt` 必须 ≤200 字**——这是子 agent 回传时唯一允许进入主 agent 上下文的 log 内容，否则会重复 Layer 0 堵源头的效果前功尽弃。

### 单 trial 失败

读 `failure.suggested_action` / `offending_params` → 记入 failing_params 黑名单 → 继续执行其他 trial → 下轮避开。

### 整轮全失败

1. 分析共同 `failure.category`（如多数 OOM → batch 过大）
2. 若 category 为 `unknown` 或同类连续失败 → 触发 Layer 2 子 agent
3. 回退到最近成功的参数组合
4. 降级方向：减半 batch、降低并发
5. 最多自动调整 3 轮，仍全败则暂停等用户介入

### 连续 2 轮全失败

暂停并报告：

```
已连续 2 轮全部失败，无法自动恢复。建议检查:
  - 搜索空间是否有可行解
  - config.toml 配置是否正确
  - 硬件资源是否充足
  - （如仍有 unknown 类失败）人工查看 `<run-dir>/trial_logs/<candidate_id>.log`
    （optix 在每个 trial 删除原始日志前自动落盘，成功/失败各一份）
```

---

## 完成

所有轮次完成后（收敛退出或跑满 max_rounds）：

```bash
python <SKILL_ROOT>/scripts/export_best_config.py \
  --run-dir .agent_optimizer/runs/<run_id>
```

Agent 生成 `final_report.md`，包含：
- 基本信息（模型、引擎、硬件、轮次/trial 数、总耗时）
- 最优结果（throughput、TPOT、对比基线提升）
- 最优参数
- **完整 vLLM serve 启动命令**（可直接执行，含所有搜索参数 + 引擎固定参数 +
  环境变量；来源：`export_best_config.py` 生成的 `serve_command.sh` /
  `optimizer_config_handoff.json` 的 `serve_command` 字段）
- 每个关键参数的探索过程（引用具体轮次和数据）
- 关键发现
- 失败统计

**生成约束**：
1. `final_report.md` 中每个参数的"探索过程"节必须引用具体轮次和数据，禁止虚泛描述。读取 `summary.md` 和 `results.round-*.json` 进行填充。
2. `final_report.md` **必须包含完整 vLLM serve 命令**（从 `serve_command.sh` 复制，
   含 `--tensor-parallel-size`/`--max-num-batched-tokens`/`--api-server-count` 等搜索
   参数、`--seed`/`--quantization` 等固定参数，以及前置 env）。命令必须是**可直接
   `bash` 执行**的一版（JSON 容器值带引号），不要只贴参数表。

---

## 输出衔接

`export_best_config.py` 生成：
- `best_result.json`：当前最优 trial
- `optimizer_config_handoff.json`：给 `ms-serviceparam-optimizer-config` skill 使用的推荐参数（含 `serve_command` 字段）
- `serve_command.json`：最优配置渲染的 vLLM serve 命令（argv + command 字符串）
- `serve_command.sh`：可直接执行的 `vllm serve` 启动脚本（final_report 从它复制命令）
