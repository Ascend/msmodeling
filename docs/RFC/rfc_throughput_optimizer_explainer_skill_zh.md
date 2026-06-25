# RFC: `throughput-optimizer-explainer` Skill 设计方案

## 元数据

| 项目 | 内容 |
|:---|:---|
| **状态** | Draft |
| **作者** | lutean |
| **创建日期** | 2026-06-24 |
| **更新日期** | 2026-06-24 |
| **相关链接** | [Skill 文档](../../.agents/skills/throughput-optimizer-explainer/SKILL.md)、[Executor RFC](rfc_throughput_optimizer_executor_skill_zh.md) |

---

## 1. 概述

本 RFC 描述 `throughput-optimizer-explainer` Skill 的设计与落地方案。该 Skill 用于解释 `python -m cli.inference.throughput_optimizer` 的输出，判断结果是否合理，比较硬件或并行策略差异，并在需要时把 optimizer best row 映射为 `python -m cli.inference.text_generate` 的 Prefill/Decode 验证命令。

该 Skill 的核心原则是先判断证据等级，再给出结论。只有宏观指标时，只做部署、阶段和策略层面的推断；只有提供 `text_generate --dump-op-bound-results` 或真实 profiler/trace 时，才进行 operator 级归因。

## 2. 目标与非目标

目标：

- 解释 aggregation、disaggregation 和 PD ratio 模式下的 optimizer 结果。
- 判断 throughput、TTFT、TPOT、PD ratio、batch、concurrency 和并行策略是否基本合理。
- 对比硬件或策略时，优先使用 phase breakdown，再使用 op-bound 和宏观指标。
- 将 aggregation best row 拆成 Prefill forward、Decode forward 和调度公式解释。
- 生成 Prefill/Decode `text_generate` 验证命令，并可追加 `--dump-op-bound-results`。
- 明确区分 TensorCast 模拟归因和真实 profiler/kernel 证据。

非目标：

- 不执行 `throughput_optimizer` 搜索；搜索由 `throughput-optimizer-executor` 负责。
- 不直接执行 `text_generate` 验证命令；执行由 `text-generate-executor` 负责。
- 不把 `text_generate` TPS 直接等同于 aggregation throughput。
- 不在缺少 op-bound 或 profiler 数据时断言具体 operator 或 kernel 瓶颈。

## 3. 证据等级

| 等级 | 输入 | 可支持结论 |
|:---|:---|:---|
| `macro_only` | 普通 optimizer summary 或 extractor JSON | 策略层解释、SLO/容量推断、Prefill/Decode 假设 |
| `optimizer_phase_breakdown` | `--dump-original-results` 中的 `percentage_breakdowns` | 阶段级 Cube/Vec/Comm/Mem 对比 |
| `text_generate_phase_breakdown` | `text_generate` Stats breakdown | 验证 optimizer 阶段假设 |
| `text_generate_op_bound` | `text_generate --dump-op-bound-results` 表格 | TensorCast 模拟 operator 级归因 |
| `profiler_trace` | 真实 profiler、chrome trace 或汇总 trace | 在 trace 支撑范围内做 operator/kernel 级判断 |

输出必须说明结论基于哪一种证据。如果证据不足，应给出最小验证动作，而不是扩大结论。

## 4. 工作流

1. 识别 optimizer 模式：aggregation、disaggregation 或 PD ratio。
2. 提取可比条件：model、device、num devices、input/output length、SLO、quantization、compile、prefix cache、MTP 和 search space。
3. 提取 best row 和 top candidates：throughput、TTFT、TPOT、concurrency、batch、parallel、PD ratio、QPS 和 breakdown。
4. 先判定证据等级，再解释结果。
5. aggregation 必须拆成 Prefill forward、Decode forward 和 scheduling formula。
6. disaggregation 直接映射到 Prefill 或 Decode 阶段。
7. 缺少 phase breakdown 且用户需要瓶颈分析时，生成 `text_generate` 验证命令。
8. 需要 operator 级模拟归因时，追加 `--dump-op-bound-results` 并检查 top total-time operators。
9. 输出合理性等级：`basically reasonable`、`partly explainable`、`suspicious` 或 `insufficient evidence`。

## 5. 脚本设计

| 脚本 | 功能 |
|:---|:---|
| `scripts/parse_optimizer_output.py` | 解析 raw optimizer 输出、dump 表、`text_generate` breakdown 和 op-bound 表为 JSON |
| `scripts/build_text_generate_commands.py` | 从 normalized best row JSON 生成 aggregation 或 disaggregation 的 `text_generate` 验证命令 |
| `scripts/compare_phase_breakdowns.py` | 比较 Cube/Vec/Comm/Mem 阶段占比，或用 `--op-bound` 比较 op-bound 表 |

脚本只负责结构化解析、命令生成和差异计算；最终结论仍由 agent 根据证据等级和用户问题组织。

## 6. Handoff 边界

- 来自 `throughput-optimizer-executor`：用户追问结果合理性、硬件差异、瓶颈归因或 best row 验证映射时，转入本 Skill。
- 到 `text-generate-executor`：本 Skill 生成验证命令后，如果用户要实际运行命令，再交由 `text-generate-executor`。
- 与 profiler 数据：用户提供真实 profiler 或 chrome trace 时，本 Skill 可以基于 trace 支撑范围进行更强证据的归因。

## 7. 测试与验收

| 测试场景 | 验收标准 |
|:---|:---|
| 只有 optimizer summary | 能给出宏观合理性判断，并说明 operator 证据不足 |
| aggregation best row | 能拆成 Prefill 和 Decode 两条 `text_generate` 验证命令 |
| disaggregation Prefill/Decode | 能按阶段生成对应验证命令 |
| phase breakdown 对比 | 能基于 Cube/Vec/Comm/Mem 解释硬件或策略差异 |
| op-bound 输出 | 能说明这是 TensorCast 模拟归因，并按 top total-time operators 解释 |
| 证据不足 | 能给出最小下一步验证动作 |

## 8. 修改文件

| 文件 | 说明 |
|:---|:---|
| `.agents/skills/throughput-optimizer-explainer/SKILL.md` | Skill 主说明、证据规则、工作流、映射规则 |
| `.agents/skills/throughput-optimizer-explainer/references/*.md` | 映射规则、证据等级、瓶颈解释和输出模板 |
| `.agents/skills/throughput-optimizer-explainer/scripts/*.py` | 解析、命令生成和对比辅助脚本 |
| `.agents/skills/README.md` | skills 索引与 quick start |
| `AGENTS.md` | 项目级 Skills 体系索引 |
