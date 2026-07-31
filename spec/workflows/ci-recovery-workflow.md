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
2. canonical PR 没有流水线评论时，在 PR 评论 `compile` 触发 openLiBing 流水线，通过 `ci-pipeline-running` label 确认启动；不得伪造成功。Fork staging
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
7. guided 模式下确认修复方案；autonomous 模式下按授权范围修复。
8. 运行受影响门禁、commit、push，在 PR 评论 `compile` 触发流水线，等待 `ci-pipeline-running` label 后回到步骤 1；`ci-pipeline-passed` 则结束，`ci-pipeline-failed` 则诊断并修复。
9. 重复直到通过或进入 blocked。

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
