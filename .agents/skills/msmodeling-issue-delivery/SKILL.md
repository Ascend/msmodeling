---
name: msmodeling-issue-delivery
description: 编排需求分析、设计、开发、本地验证、GitCode PR 和 openLiBing CI，将指定 MindStudio-Modeling Issue 推进到 ready-for-review。
metadata:
  version: 1.0.0
  source: issue-25-ai-native
---

# Issue 端到端交付

## 适用场景

- “实现 Issue 25”
- “从这个 Issue 开始自动完成开发和 CI”
- “继续上次未完成的 Issue 工作流”

## 前置读取

1. `AGENTS.md`
2. `spec/workflows/development-workflow.md`
3. `spec/workflows/pr-workflow.md`
4. `spec/workflows/ci-recovery-workflow.md`
5. `spec/foundations/audit-record-contract.md`
6. 与任务匹配的领域 Skill

## 组合能力

- 远端 Issue：`gitcode-issue-review`
- PR 创建：`gitcode-pr-create`
- 本地门禁：`gitcode-precommit` 和项目测试命令
- 安全检查：`gitcode-security-check`
- CI：`msmodeling-ci-recovery`

> 独立检视由 reviewer 通过 `sig-review` 技能独立触发，不属于交付自动化范围。

## 执行流程

1. 解析 canonical/source/operation target，创建 `run-id`，使用 CLI 读取 Issue 和已有审计评论。
2. 核验正式 Issue 是否仍开放、是否已有闭环 PR、canonical `master` 是否已包含实现。
3. 输出需求分析：范围、非目标、验收、依赖、风险和待确认项。
4. guided 模式下等待批准；autonomous 模式必须存在已记录的明确授权。
5. 输出方案设计和文件级计划；架构变更写 RFC，具体设计写 `docs/design/`。
6. 从最新 canonical `master` 创建非 master 分支；Fork 模式 push 到 source repository。
7. 按计划实现，每完成一项立即验证，不批量假定完成。
8. 执行受影响 pytest、pre-commit、安全检查和必要构建。
9. 使用 sign-off commit，推送前报告变更和验证证据。
10. 用户确认或授权覆盖后，推送 source branch，并通过 CLI 向 canonical repository 创建 Draft PR。
11. 对 canonical PR 执行 CI Recovery，失败则分析、修复、验证、推送并再次监控。
12. 提交作者自检，确认文档、风险、未覆盖项和 CI。
13. 用户确认后将 PR 切换为 Ready。

## 人工检查点

默认在需求分析、设计、实现完成、push、PR 创建、CI 修复和 Ready 前确认。`autonomous` 只能跳过普通检查点，不能跳过
`spec/foundations/ai-execution-policy.md` 的硬停止点。

## 完成标准

- canonical PR 为 ready-for-review；
- canonical PR 的 CI 通过或存在明确阻塞记录；
- Issue/PR 审计链完整；
- 作者自检未冒充独立评审；
- 未自动审批或合并。
