# 单一事实来源矩阵（Source of Truth Matrix）

本文件是 msmodeling 项目的**文档优先级权威**。当多份文档对同一事项给出不同描述时，按本矩阵的优先级裁决：高优先级为准，低优先级视为过期，并应尽快修正。

> **适用对象**：AI agent、人类贡献者、CI 门禁脚本。三者发生分歧时，均以本矩阵为准。

---

## 1. 文档优先级

| 优先级 | 文档 / 目录 | 角色 | 性质 |
|:---:|------|------|------|
| 1 | `spec/` | 项目正式规范（**最高权威**） | 规范源 |
| 2 | `docs/RFC/` | 设计提案（先于代码，经评审合入） | 规范源（设计层） |
| 3 | `docs/design/` | 实现设计（RFC 落地细节） | 规范源（实现层） |
| 4 | `CONTRIBUTING.md` | 人类贡献规范（环境、commit、PR、测试、代码风格） | 规范源（流程层） |
| 5 | `AGENTS.md` | AI agent 入口（项目概览 + skill 索引 + 指向 spec/） | 入口 |
| 6 | `CLAUDE.md` | Claude Code adapter（指向 AGENTS.md + spec/） | 入口（adapter） |
| 7 | `.agents/skills/*/SKILL.md` | 任务级执行规范 | 执行规范 |
| 8 | `README.md` | 人类入口（安装、快速上手） | 入口 |
| 9 | `origin/master` 实际代码 | 最终产物 | 产物 |

**关键区分**：

- **规范源**（1-4、7）：定义"应该怎样"，是设计意图。
- **入口文件**（5、6、8）：定义"从哪开始读"，不是设计文档。入口文件之间若冲突，以规范源为准。
- **产物**（9）：定义"实际怎样"。当代码与规范源冲突时，**代码是事实但不是权威**——应视作 bug 并修复其中一方（多数情况下修正代码以符合 spec；若 spec 确实过期，走 RFC 流程更新 spec）。

---

## 2. 冲突解决规则

### 2.1 AI agent 读取顺序

AI agent 在执行任务时，应按以下顺序确认事实：

1. **先读入口**：`AGENTS.md`（项目概览 + skill 索引）→ 定位相关 skill
2. **再读 spec**：若任务涉及规范约束，读 `spec/` 下对应文件
3. **再读 skill**：读 `.agents/skills/{skill}/SKILL.md` 获取执行步骤
4. **以代码验证**：涉及"当前实现是什么"时，按任务读取对应代码版本——开发/测试读当前工作树，PR 检视读 PR head，确认项目基线状态读 `origin/master`；不要仅依赖文档描述

### 2.2 冲突发现时的处置

| 冲突类型 | 处置 |
|------|------|
| spec/ 与入口文件（AGENTS.md / CLAUDE.md / README.md）冲突 | 以 `spec/` 为准；入口文件视作过期，应在本 PR 或后续 PR 修正 |
| spec/ 与 `CONTRIBUTING.md` 冲突 | 以 `spec/` 为准；`CONTRIBUTING.md` 视作过期，应修正 |
| spec/ 与 `docs/RFC/` 冲突 | 以更高优先级方为准；若 RFC 是更新提案，应走 RFC 合入流程更新 spec/ |
| spec/ 与 skill SKILL.md 冲突 | 以 `spec/` 为准；skill 视作过期，应修正 |
| spec/ 与实际代码冲突 | **代码是事实但不是权威**——判断哪方过期，修正过期方；若修正 spec/，需走 RFC |
| 入口文件之间冲突（如 AGENTS.md 与 CLAUDE.md） | 以 `AGENTS.md` 为准（CLAUDE.md 是 adapter，仅转发） |

### 2.3 何时该写 spec vs RFC vs design

| 文档类型 | 何时写 | 位置 | 评审要求 |
|------|------|------|------|
| **spec/** | 项目不变量、强制规范、工作流定义、治理规则 | `spec/{子目录}/{name}.md` | 必须经 PR 评审合入，修改需新 PR |
| **RFC** | 新功能/新架构/破坏性变更的设计提案 | `docs/RFC/rfc_{name}_zh.md` 或 `rfc_{name}_en.md` | 必须经 RFC 评审，合入后相关 spec 应同步更新 |
| **design** | RFC 落地的实现细节、模块划分、接口设计 | `docs/design/{name}_design.md` | 评审同 PR，是 RFC 与代码之间的桥梁 |
| **skill** | 可重复执行的任务流程（含触发词、步骤、产出） | `.agents/skills/{name}/SKILL.md` | 遵循 [Skill 开发规范](../../AGENTS.md) |

**经验法则**：

- 改"规则"→ 写 spec/
- 提"新设计"→ 写 RFC → 合入后更新 spec/
- 写"怎么实现"→ 写 design
- 写"AI 怎么执行任务"→ 写 skill

---

## 3. spec/ 目录结构

```text
spec/
├── governance/                 # 治理规则
│   ├── source-of-truth-matrix.md   # 本文件
│   └── ai-collaboration.md         # AI 协作规范（后续补齐）
├── foundations/                # 基础规范（不变量，后续补齐）
├── workflows/                  # 工作流（动态流程，后续补齐）
│   ├── issue-workflow.md
│   ├── pr-workflow.md
│   └── review-workflow.md
└── delivery/                   # 交付（后续补齐）
    └── release-process.md
```

**新增 spec 文件的原则**：

- 只写**强制规范**（"必须/禁止/应该"），不写可选建议
- 每条规范应可被代码、CI 或 agent 检查/遵守
- 规范变更必须走 PR，commit message 含 `spec(scope): 描述`

---

## 4. 维护责任

| 文档 | 维护者 | 更新触发 |
|------|------|------|
| `spec/` | SIG「文档与Skill」chair | 规则变更、RFC 合入、工作流调整 |
| `AGENTS.md` | SIG「文档与Skill」chair | skill 新增/移除、项目结构变化、spec 目录变化 |
| `CONTRIBUTING.md` | SIG「测试与基础设施」+「文档与Skill」 | 流程变更、环境变更、规范变更 |
| `.agents/skills/` | 各 skill 对应 SIG | skill 流程变更 |
| `docs/RFC/` | RFC 提出者 | 新设计提案 |

> **SIG 路由**：文件归属哪个 SIG 检视，由 [`sig_ownership.json`](../../.agents/skills/sig-review/sig_ownership.json) 最长前缀匹配决定。本文件归 SIG「文档与Skill」。
