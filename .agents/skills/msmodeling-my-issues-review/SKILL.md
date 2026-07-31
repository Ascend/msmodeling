---
name: msmodeling-my-issues-review
description: 查询 GitCode 中当前用户负责的 MindStudio-Modeling 开放 Issue，结合代码给出接受、拒绝、需补充或阻塞的完整评审结论。
metadata:
  version: 1.0.0
  source: issue-25-ai-native
---

# 我的 Issue 评审

## 适用场景

- “分析我负责的开放 Issue”
- “评审 Issue 25”
- “哪些 Issue 可以开始开发”

## 前置条件

阅读 `spec/workflows/issue-review-workflow.md` 和 `spec/foundations/gitcode-cli-contract.md`。

## 工作流程

1. 执行 `gitcode auth status` 获取当前登录用户名，不显示 Token。
2. 查询：

   ```bash
   gitcode issue list -R <TARGET_REPO> \
     --state open --assignee <username> --json
   ```

3. 用户未指定编号时，先展示候选列表；没有候选时明确返回空结果。
4. 对目标 Issue 执行 `issue view --comments --json`、`issue comments --json`、`issue prs --json`，并补充 `issue relations -R <TARGET_REPO> --json` 和 `milestone list -R <TARGET_REPO> --json` 以确认关联与里程碑，避免仅凭标题判断。
5. 检查 canonical `master` 的本地跟踪分支和当前实现，确认 Issue 是否仍有效、是否已被 PR 或代码闭环。
6. 输出：
   - 当前理解；
   - 代码与远端证据；
   - 缺失信息；
   - 验收标准；
   - 实现建议；
   - 风险；
   - 结论：接受、拒绝、需补充或阻塞；
   - 下一步。
7. 用户确认后使用 `gitcode issue comment --body-file ... --json` 提交评审。

## 安全规则

- 未经授权不修改标签、负责人、里程碑或状态；用户未明确要求时不覆写已有 label 或 milestone。
- 评审结论必须区分事实与假设，issue 评论和关联 PR 作为证据，而非标题。
- 不因 Issue 已关闭就判定已实现，必须检查 PR 和 `origin/master`。
- 不存在或不可访问的 Issue 必须终止，禁止编造。

## 完成标准

结论有代码和远端事实支撑，且无论结论类型都包含可执行建议。
