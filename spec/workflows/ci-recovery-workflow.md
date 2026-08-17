# openLiBing CI 恢复工作流

## 平台边界

canonical repository 的 PR 流水线由 openLiBing 执行，通过 GitCode PR 机器人评论反馈。不得假设已启用
GitCode Actions。Fork 内部 staging PR 可以没有 CI，但不能替代 canonical PR 的 CI 证据。

## 状态机

```text
idle -> triggered -> running -> passed
                  \-> failed -> diagnosed -> fixed -> pushed -> running
                  \-> blocked
```

## 步骤

1. 使用 `gitcode pr comments <PR> --json` 获取流水线评论。
2. canonical PR 创建或推送后**默认立即**触发 openLiBing CI，但以 `PR number + head SHA` 为幂等键避免重复流水线：先查当前 head 是否已有有效运行——已 `running` 则只监控，已 `passed` 则结束，已 `failed` 则进入诊断；只有当前 head 不存在有效运行时才评论 `compile` 触发。通过 `ci-pipeline-running`/`ci-pipeline-passed`/`ci-pipeline-failed` label 判定；不得伪造成功。Fork staging
   PR 可以记录为 `not-applicable`。
3. 必须使用 `gitcode-pipeline-analyzer` 选择最新有效运行并读取任务状态和日志。
4. 优先提取第一条直接失败、测试摘要和与改动最相关的错误。
5. 分类为：
   - 当前改动直接导致；
   - 当前改动触发的连带失败；
   - 基线问题；
   - 基础设施或权限问题；
   - 证据不足。
6. 本地复现或通过代码证据验证根因。
   - **ruff format/check 复现**：必须使用 `--config pre-commit/pyproject.toml`，不得使用默认配置。`pre-commit/pyproject.toml` 设定了 `line-length=120`、`quote-style=preserve` 等规则，与默认配置（line-length=88）不一致；用默认配置本地通过但 CI 仍会失败。
7. **CI 修复循环默认自动**：诊断根因并本地复现后，无需逐次向用户确认，直接修复——运行受影响门禁、commit、push、在 PR 评论 `compile` 重新触发流水线，等待 `ci-pipeline-running` label 后回到步骤 1。这是 CI 恢复的默认执行模式，覆盖 guided 模式下"push 前确认"的一般规则，但仍受硬停止约束：不得以"强制测试通过"代替根因修复，不得绕过质量门禁。
8. `ci-pipeline-passed` 则结束循环；`ci-pipeline-failed` 则回到步骤 4 诊断并修复。
9. 重复直到通过或进入 blocked。**只有**遇到以下情形才停下交给用户：证据充分的责任/基础设施问题、无法自动定位根因（证据不足）、或修复会触及硬停止点。blocked 时必须在 PR 评论记录证据、责任和下一步。
10. **完成后报告**：CI 全绿后，向用户报告本轮 CI 修复共新增 N 个 commit（列出每个 commit 的摘要和 SHA），提示用户关注这些修复提交。无修复（首次即通过）时报告"CI 一次通过，无额外修复 commit"。

## 文档 CI

文档 CI（`docs-ci-pipeline-*`）由后台自动触发，不需要评论 `compile`。通过以下 label 确认状态：

- `docs-ci-pipeline-running`：文档 CI 执行中
- `docs-ci-pipeline-success`：文档 CI 通过
- `docs-ci-pipeline-failed`：文档 CI 失败

文档 CI 失败时，从 PR 评论区提取报错信息（通常由 `ascend-robot` 发布），按报错修复文档文件后 commit、push 即可重新触发，无需评论 `compile`。

## 审计

每轮记录：

- pipeline URL/run 标识；
- commit SHA；
- 失败任务和摘要；
- 根因分类；
- 修复文件和验证；
- 新 commit；
- 下一轮结果。

基础设施失败只有在有明确证据时才能这样分类。重试不能代替根因分析。
