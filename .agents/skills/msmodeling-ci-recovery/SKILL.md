---
name: msmodeling-ci-recovery
description: 使用 GitCode CLI 和 gitcode-pipeline-analyzer 监控 MindStudio-Modeling openLiBing PR 流水线，分析失败、修复并循环验证。
metadata:
  version: 1.0.0
  source: issue-25-ai-native
---

# CI 监控与恢复

## 适用场景

- “看 PR 123 流水线”
- “分析 CI 失败并修复”
- “持续监控到全绿”

## 前置读取

- `spec/workflows/ci-recovery-workflow.md`
- `.agents/skills/gitcode-pipeline-analyzer/SKILL.md`

## 工作流程

1. 使用 `gitcode pr view` 和 `gitcode pr comments --json` 确认 PR 与最新 head。
2. 评论区没有流水线记录时，在 PR 评论 `compile` 触发 openLiBing 流水线；通过 `ci-pipeline-running` label 确认启动，`ci-pipeline-failed`/`ci-pipeline-passed` 确认结果：`passed` 表示 CI 通过，结束循环；`failed` 表示 CI 失败，进入诊断与修复。不得改用 GitCode Actions 猜测。
3. 文档 CI（`docs-ci-pipeline-*`）由后台自动触发，不需要评论 `compile`。`docs-ci-pipeline-failed` 时从评论区提取报错信息修复文档，commit、push 后自动重新触发。
4. 运行 pipeline analyzer，优先选择当前 head 的最新有效流水线。
5. 输出 stage/job 状态、失败任务、首个直接错误和关联日志。
6. 分类根因：直接、连带、基线、基础设施或证据不足。
7. 读取相关代码并尽可能本地复现。
   - **ruff format/check 本地复现**：必须用 `--config pre-commit/pyproject.toml`（line-length=120、quote-style=preserve），不用默认配置（line-length=88），否则本地通过但 CI 仍失败。
8. guided 模式下确认修复方案；autonomous 模式按已记录授权处理。
9. 修复后运行受影响门禁、commit、push，在 PR 评论 `compile` 触发流水线，等待 `ci-pipeline-running` label 出现后进入监控，直到 `ci-pipeline-passed`（结束循环）或 `ci-pipeline-failed`（进入诊断修复）。
10. 在 PR 评论中记录本轮 run、根因、验证和修复 commit。
11. 重复直到通过或 blocked。

## 安全规则

- GitCode PR 和评论只经 CLI。
- openLiBing 访问只允许使用 pipeline analyzer。
- 日志输出必须脱敏。
- 不得使用“强制测试通过”代替修复。

## 完成标准

CI 全绿（以 `ci-pipeline-passed` label 为准），或存在证据充分、责任和下一步清晰的 blocker 记录。

## `/govern` 治理集成

CI 阻塞责任移交时（标记阻塞、转交修复等），在 PR 评论写一行 `/next <login> <verb>` 触发通知 + 看板。协议见 `spec/governance/next-comment-protocol.md`。
