---
name: msmodeling-ci-recovery
description: 使用 GitCode CLI 和 gitcode-pipeline-analyzer 监控 MindStudio-Modeling openLiBing PR 流水线，分析失败、修复并循环验证。
metadata:
  version: 1.1.0
  source: cicd-auto-recovery
---

# CI 监控与恢复

## 适用场景

- “PR 创建后默认触发 CI 并监控至全绿”
- “看 PR 123 流水线”
- “分析 CI 失败并修复”
- “持续监控到全绿”

## 前置读取

- `spec/workflows/ci-recovery-workflow.md`
- `.agents/skills/gitcode-pipeline-analyzer/SKILL.md`

## 工作流程

1. 使用 `gitcode pr view` 和 `gitcode pr comments --json` 确认 PR 与最新 head。
2. **PR 创建或推送后默认立即**触发 openLiBing CI，但以 `PR number + head SHA` 为幂等键避免重复流水线：先查当前 head 是否已有有效运行——`ci-pipeline-running` 则只监控，`ci-pipeline-passed` 则结束，`ci-pipeline-failed` 则进入诊断；只有当前 head 不存在有效运行时才评论 `compile` 触发。不得改用 GitCode Actions 猜测。
3. 文档 CI（`docs-ci-pipeline-*`）由后台自动触发，不需要评论 `compile`。`docs-ci-pipeline-failed` 时从评论区提取报错信息修复文档，commit、push 后自动重新触发。
4. 运行 pipeline analyzer，优先选择当前 head 的最新有效流水线。
5. 输出 stage/job 状态、失败任务、首个直接错误和关联日志。
6. 分类根因：直接、连带、基线、基础设施或证据不足。
7. 读取相关代码并尽可能本地复现。
   - **ruff format/check 本地复现**：必须用 `--config pre-commit/pyproject.toml`（line-length=120、quote-style=preserve），不用默认配置（line-length=88），否则本地通过但 CI 仍失败。
8. **CI 修复循环默认自动**：诊断根因并本地复现后无需逐次向用户确认，直接修复——运行受影响门禁、commit、push、评论 `compile` 重新触发流水线，等待 `ci-pipeline-running` label 后进入监控。这是 CI 恢复的默认执行模式，覆盖 guided 模式"push 前确认"的一般规则，但仍受硬停止约束：不得以强制测试通过代替根因修复，不得绕过质量门禁。
9. `ci-pipeline-passed` 结束循环；`ci-pipeline-failed` 回到步骤 4 诊断并修复。
10. 在 PR 评论中记录本轮 run、根因、验证和修复 commit。
11. 重复直到通过或 blocked。**只有**证据充分的责任/基础设施问题、无法自动定位根因（证据不足）或触及硬停止点时才停下交用户；blocked 时在 PR 评论记录证据、责任和下一步。
12. **完成后报告**：CI 全绿后向用户报告本轮共新增 N 个修复 commit（列出每个 commit 摘要和 SHA），提示用户关注；无修复（首次即通过）时报告"CI 一次通过，无额外修复 commit"。

## 安全规则

- GitCode PR 和评论只经 CLI。
- openLiBing 访问只允许使用 pipeline analyzer。
- 日志输出必须脱敏。
- 不得使用“强制测试通过”代替修复。

## 完成标准

CI 全绿（以 `ci-pipeline-passed` label 为准），或存在证据充分、责任和下一步清晰的 blocker 记录。

## 后台服务与通知

PR 流水线状态与阻塞流转通知由常驻服务 后台合入管理服务 统一驱动（基于 `/merge`、CI label 和 diff_comment），agent 不在 PR 评论写 `/next` 治理评论，避免双重通知。CI 修复循环内的 commit、push 和重新触发均由本技能自动完成。协议见 `spec/governance/next-comment-protocol.md`。
