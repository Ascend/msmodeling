# PR 工作流

## 创建前

- 工作树和分支符合预期；
- 变更范围与 Issue 一致；
- 相关本地门禁通过；
- commit 已 sign-off；
- PR body 不含凭证和本地绝对路径；
- 已确认 base 为 `master`。
- 已确认 canonical target、source repository、head 分支和 operation target。

## 创建

1. 正式交付使用 `gitcode pr create -R Ascend/msmodeling --body-file ... --json`。
2. Fork 模式显式指定 canonical target、Fork source、head 和 base；主仓分支模式显式确认 source 与 target
   均为 canonical。
3. 首次提交建议创建 Draft。
4. PR body 至少包含：
   - 背景与关联 Issue；
   - 修改内容；
   - 验证证据；
   - 风险与回滚；
   - AI 参与说明；
   - 未覆盖项。
5. 创建后使用 `gitcode pr view --json` 回读验证。
6. 在 Issue 评论中记录 PR 编号和 URL。

## Ready

只有满足以下条件才允许从 Draft 切换为 Ready：

- 本地门禁完成；
- CI 通过；
- PR 位于 canonical repository；Fork 内部 staging PR 不能进入最终 Ready 完成态；
- 作者自检完成；
- 文档同步完成；
- 没有未披露 blocker；
- 用户确认或 autonomous 授权覆盖该动作。

审批和合并必须由有权限的独立主体完成。
