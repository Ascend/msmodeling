# RFC: `throughput-optimizer-executor` Skill 设计方案

## 元数据

| 项目 | 内容 |
|:---|:---|
| **状态** | Draft |
| **作者** | lutean |
| **创建日期** | 2026-05-30 |
| **更新日期** | 2026-05-30 |
| **相关链接** | [Executor Skill](../../.agents/skills/throughput-optimizer-executor/SKILL.md)、[Explainer Skill](../../.agents/skills/throughput-optimizer-explainer/SKILL.md)、[Explainer RFC](rfc_throughput_optimizer_explainer_skill_zh.md) |

---

## 1. 概述

本 RFC 描述 `throughput-optimizer-executor` Skill 的设计与落地方案。该 Skill 用于把用户关于部署规划、硬件对比、并行搜索、PD 聚合/分离/配比优化等自然语言诉求，转换为 `python -m cli.inference.throughput_optimizer` 命令，并在用户确认后执行仿真、解析结果、总结最优并行策略。

该 Skill 重点服务“搜索和规划”场景：用户通常不知道 TP/EP/MOE-DP、batch range、concurrency 或 P/D 配比的最佳组合，需要工具自动搜索候选策略并输出 throughput、TTFT、TPOT 等指标。与 `text-generate-executor` 的单点验证不同，本 Skill 面向跨候选、跨硬件和跨部署模式的吞吐建模。

## 2. 目标与非目标

目标：

- 支持用户通过对话生成 `python -m cli.inference.throughput_optimizer` 命令。
- 支持 aggregation、disaggregation phase evaluation、P/D instance ratio planning 三类核心模式。
- 支持单硬件和多硬件 profile 对比。
- 支持 TP、EP、MOE-DP 等并行搜索空间的保守默认策略和用户自定义策略。
- 支持 TTFT、TPOT 或两者共同作为 SLO 目标。
- 执行后提取最优策略、batch、concurrency、throughput、TTFT、TPOT、Top candidates、PD ratio 结果。

非目标：

- 不修改 throughput optimizer 的搜索算法或性能模型。
- 不自动决定量化策略；只提供推荐默认值并要求用户确认。
- 不保证建模结果等同真实线上性能；结果必须作为部署规划参考，并要求真实负载验证。
- 不负责深入解释结果合理性、硬件差异或 Cube/Vec/Comm/Mem 瓶颈；这些问题交由 `throughput-optimizer-explainer`。
- 不负责直接执行 `text_generate` 单点复验；复验命令由 `throughput-optimizer-explainer` 映射后，实际执行交由 `text-generate-executor`。

## 3. 用例分析

| 用例 | 用户输入示例 | Skill 行为 | 输出 |
|:---|:---|:---|:---|
| 聚合部署搜索 | “Qwen 32B，A3，8 卡，输入 4k 输出 1k，找最佳 TP” | 使用 aggregation 模式，构造 TP 搜索命令 | 最优 TP、batch、concurrency、throughput、TTFT/TPOT |
| 多硬件对比 | “比较 ATLAS_800_A2_280T_64G 和 ATLAS_800_A3_560T_128G_DIE” | `--device` 传入多个 profile，共享 `--num-devices` | 跨硬件 summary 和最佳行 |
| PD 分离能力评估 | “分别看 prefill 和 decode 能力” | 使用 `--disagg`，按 SLO 运行 P/D 阶段优化 | Prefill/Decode 阶段结果 |
| P/D 实例配比规划 | “给定 P 4 卡 D 4 卡，算 PD ratio” | 使用 `--enable-optimize-prefill-decode-ratio` 和 P/D devices per instance | P/D QPS、PD ratio、组合 Top N |
| MoE 搜索 | “DeepSeek MoE 搜 TP、EP、MOE-DP” | 询问搜索范围，构造 MoE 并行搜索参数 | MoE 最优候选和并行策略 |

DFX 要求：

- 兼容性：仅通过 CLI 参数驱动现有 optimizer。
- 可维护性：把问参流程和参数解释拆分为 references 文档。
- 可测试性：通过固定 prompt 验证命令生成、模式选择、校验规则和失败处理。
- 可靠性：执行前必须展示命令和约束检查，避免组合互斥参数。
- 上下文隔离：除非用户明确要求沿用上一轮配置，否则每次新规划诉求都按新的参数收集流程处理，避免历史对话污染命令生成。

## 4. 方案设计

### 4.1 总体设计

Skill 目录结构：

```text
throughput-optimizer-executor/
├── SKILL.md
└── references/
    ├── dialog-flow.md
    └── throughput-optimizer-params.md

```

核心流程：

```text
用户部署规划诉求
        ↓
识别模式：aggregation / disagg / PD ratio / multi-hardware / MoE
        ↓
渐进式收集模型、硬件、设备数、输入输出长度、SLO、搜索空间
        ↓
仅当用户不确定或要求推荐时提供建议值，并标注来源
        ↓
应用默认规则和互斥参数校验
        ↓
展示命令和假设，等待用户确认
        ↓
执行 throughput_optimizer
        ↓
解析表格和 summary
        ↓
输出最佳策略、Top candidates、跨硬件或 PD ratio 摘要
```

### 4.2 模式选择规则

- “PD 混部 / PD 聚合 / 聚合部署”映射为 aggregation，不添加 `--disagg`。
- “PD 分离”需要进一步询问目标：阶段能力评估或 P/D 实例配比规划。
- 阶段能力评估使用 `--disagg`，并根据 SLO 或用户选择运行 Prefill、Decode 或两者。
- P/D 实例配比规划使用 `--enable-optimize-prefill-decode-ratio`，并要求 `--prefill-devices-per-instance` 与 `--decode-devices-per-instance`。
- `--enable-optimize-prefill-decode-ratio` 与 `--disagg` 互斥。

### 4.3 参数收集策略

首轮必须收集：

- `model_id`
- `--device`，支持多个硬件 profile
- `--num-devices`
- 输入长度
- 输出长度
- 部署模式
- SLO 目标：TTFT、TPOT 或两者

按模型和场景扩展：

- Dense 模型：优先保守 TP 搜索。
- MoE 模型：先询问是否只搜索 TP，或扩展 EP/MOE-DP 搜索。
- 多模态模型：询问 image height / width 等视觉输入参数。
- Prefix cache：询问 `--prefix-cache-hit-rate` 与 `--max-prefill-tokens` 影响。
- MTP：询问 `--num-mtp-tokens` 并确认模型支持。若后续 handoff 到 `text_generate` 做 Decode MTP 单点复验，`--query-length` 需设置为 `1 + --num-mtp-tokens`，因为 decode query length 为 `1`。

### 4.4 默认值和确认规则

- 不静默选择量化策略。
- 推荐量化默认值：`--quantize-linear-action W8A8_DYNAMIC` 与 `--quantize-attention-action DISABLED`。
- `--compile` 默认推荐开启，但仍允许用户调整。
- 用户不清楚搜索空间时，采用保守搜索：dense 先搜 TP；MoE 先搜 TP，再建议 EP/MOE-DP 作为扩展。
- 多硬件对比时，一个共享 `--num-devices` 应用于所有 profile，需要提醒用户该约束。
- 用户只提供模型和规划场景时，不生成完整命令，必须先渐进式询问缺失核心参数。
- 不默认复用历史对话中的硬件、设备数、输入输出长度、部署模式、SLO、batch range、TP/EP/MOE-DP 搜索空间等参数；仅当用户明确说“沿用/继续/保持上一轮配置”时才复用。
- 仅当用户表示“不确定”或主动要求建议时给出建议值。
- 建议值必须标注来源：CLI 默认、用户上一轮配置、当前轮检查过的本地测试/示例、文档示例、throughput_optimizer 结果或显式启发式。
- 不得声称某配置或搜索计划来自回归覆盖、仓库默认或本地示例，除非当前轮检查了具体文件或命令输出，并在摘要中说明来源。
- 显式启发式建议必须说明其为启发式，并简要说明实践理由。

### 4.5 校验规则

执行前校验：

- `--device` 可包含多个 profile，多 profile 使用空格分隔。
- 多硬件运行共用同一个 `--num-devices`。
- `--enable-optimize-prefill-decode-ratio` 不可与 `--disagg` 同时使用。
- PD ratio 模式必须提供 P/D 每实例设备数。
- 显式 TP、EP、MOE-DP 不应超过对应阶段的设备数。
- `--batch-range` 必须是 `[max]` 或 `[min max]`，且为正整数、`min <= max`。
- aggregation 模式下 `--max-prefill-tokens` 不应小于 prefix cache 后的有效输入长度。
- Decode MTP 结果解释或 handoff 到 `text_generate` 时，需要说明 MTP forward 时间口径：它对应原主模型 forward 加 MTP layer forward；若用户要估算开启 MTP 后的单 token 时延，可除以 `1 + sum(每步 MTP 的接受率)`，接受率由用户按预期自行给定。

### 4.6 结果解析与摘要

执行成功后输出：

- 完整命令。
- 模式：aggregation、disagg phase evaluation 或 PD ratio planning。
- 最优并行策略。
- batch size、concurrency、throughput。
- TTFT、TPOT。
- 多硬件对比 summary。
- Top candidates 表格摘要。
- PD ratio 模式下的 P QPS、D QPS、PD ratio 和 P/D 配置。
- 统一免责声明：结果来自 throughput modeling，仅供部署规划参考。

执行失败后输出：

- 完整命令。
- 关键错误行。
- 错误分类：参数、环境、模型/设备假设、搜索空间冲突。
- 最小修正建议。

### 4.7 Handoff 边界

- `throughput-optimizer-executor` 负责问参、生成命令、执行 optimizer 和总结最优候选。
- 用户追问“结果是否合理”“为什么硬件表现不同”“Cube/Vec/Comm/Mem 如何解释”“best row 如何映射到 text_generate”时，交由 `throughput-optimizer-explainer`。
- `throughput-optimizer-explainer` 可以生成 Prefill/Decode 验证命令；如果用户要实际运行这些命令，再交由 `text-generate-executor`。

## 5. 实施计划

| 阶段 | 内容 | 状态 |
|:---|:---|:---|
| P0 | 创建 `throughput-optimizer-executor` Skill 主说明 | 已完成 |
| P0 | 提供问参流程和参数说明 references | 已完成 |
| P0 | 提供结果提取脚本 `extract_throughput_optimizer_result.py` | 已完成 |
| P1 | 补充典型场景验收 prompt | 待执行 |
| P1 | 与 `throughput-optimizer-explainer` 明确 handoff 边界 | 已完成 |
| P2 | 增强多硬件和 PD ratio 输出结构化摘要 | 可选 |

## 6. 测试与验收

| 测试场景 | 验收标准 |
|:---|:---|
| Aggregation TP 搜索 | 能生成不含 `--disagg` 的聚合部署命令 |
| PD 分离能力评估 | 能识别 phase evaluation 并添加 `--disagg` |
| PD ratio 规划 | 能添加 `--enable-optimize-prefill-decode-ratio` 并要求 P/D devices per instance |
| 互斥参数校验 | 同时出现 `--disagg` 与 PD ratio 参数时阻止执行 |
| 多硬件对比 | 能生成多个 `--device` profile，并说明共享 `--num-devices` |
| MoE 搜索 | 能追问 EP/MOE-DP 搜索范围，不默认扩大搜索空间 |
| 上下文隔离 | 用户只给模型和规划场景时不复用历史配置，不生成完整命令，而是询问缺失核心参数 |
| 建议来源标注 | 用户不确定时可以给建议，但每个建议值必须标注来源；未检查本地文件时不得声称来自回归覆盖 |
| 失败处理 | 能展示命令、关键错误和最小修正建议 |

## 7. 修改文件

| 文件 | 说明 |
|:---|:---|
| `skills/throughput-optimizer-executor/SKILL.md` | Skill 主说明和执行规则 |
| `skills/throughput-optimizer-executor/references/dialog-flow.md` | 问参流程和模式分支 |
| `skills/throughput-optimizer-executor/references/throughput-optimizer-params.md` | 参数说明和默认规则 |

## 8. 后续演进

- 增强 optimizer 输出解析，统一 Overall Best、Top candidates、PD ratio 的 JSON schema。
- 增加更多典型硬件/模型场景的验收 prompt。
- 通过 `throughput-optimizer-explainer` 建立显式解释链路：从 optimizer best row 生成 Prefill/Decode 验证命令，并按需 handoff 到 `text-generate-executor` 执行。
