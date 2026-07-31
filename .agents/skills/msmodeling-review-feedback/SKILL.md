---
name: msmodeling-review-feedback
description: 拉取 MindStudio-Modeling PR 的行内和总体检视意见，分类、修复、验证并通过 GitCode CLI 回复和解决讨论。
metadata:
  version: 1.1.0
  source: issue-25-ai-native
---

# PR 检视意见处理

## 适用场景

- "处理 PR 123 的评审意见"
- "按检视意见修改代码"
- "分析哪些评论合理"

## 前置条件

1. `gitcode version` 和 `gitcode auth status` 通过。
2. 已知 PR 编号和 target repository。

## 权威数据源

- PR 评论（含行内检视意见）：`gitcode pr comments <PR> -R <repo> --json`
- PR diff：`gitcode pr diff <PR> -R <repo>`
- 回复 discussion：`gitcode pr reply <PR> --discussion <id> --body <text> -R <repo>`
- 提交汇总：`gitcode pr comment <PR> -R <repo> --body <text>`

## 工作流程

### 1. 拉取检视意见

```bash
gitcode pr comments <PR编号> -R <TARGET_REPO> --json
```

从返回的评论列表中筛选 `comment_type` 为 `diff_comment`（行级检视意见）的评论。每条含 `id`、`discussion_id`、`diff_file`、`diff_position`、`resolved`（True/False）和 `body`。

### 2. 分类

将意见按下表分严重度，并按文件分组（同文件的多个意见一次性处理，减少跳转）。对每条意见判断：接受、需澄清、替代方案、拒绝或延期。

| 严重度 | 别名 | 动作 |
|--------|------|------|
| 阻塞 | P0 / 需修改 | 必须修，不修无法合入 |
| 高 | P1 | 强烈建议修 |
| 中 | P2 | 合理则修，可延期 |
| 建议 | P3 | 记录，按需处理 |

不明确意见先通过 `gitcode pr reply` 请求澄清，不猜测修改：

```bash
gitcode pr reply <PR编号> -R <TARGET_REPO> --discussion <discussion_id> --body "请澄清：这里期望的行为是什么？"
```

### 3. 修复

对接受项修改代码并运行受影响测试。行级检视意见的 `diff_file` 与 `diff_position` 指向 PR diff，定位本地行时不要直接套用行号，按 [sig-review/ref/line-mapping.md](../sig-review/ref/line-mapping.md) 解析 hunk 求新版本行号；拿不准时按评论内容而非行号定位。

### 4. 回复每条意见（提交前必须执行）

每条 diff_comment 处理完后，必须通过 `gitcode pr reply` 回复，说明处理结果。回复要极简但明确：

```bash
# 已修复
gitcode pr reply <PR编号> -R <TARGET_REPO> --discussion <discussion_id> --body "已修复，见 commit <sha>"

# 延期
gitcode pr reply <PR编号> -R <TARGET_REPO> --discussion <discussion_id> --body "延期，原因：依赖 xxx 先行合入"

# 拒绝
gitcode pr reply <PR编号> -R <TARGET_REPO> --discussion <discussion_id> --body "未采纳，原因：与 spec xxx 冲突，以 spec 为准"
```

> **回复要求**：每条 diff_comment 必须有回复，不能默默修改后不回复。reviewer 需要知道每条意见的处理结果。

### 5. 解决讨论（resolve）

> **已知缺口**：GitCode API v5 尚未公开 resolve 端点，gitcode CLI 无 `pr resolve` 子命令。当前无法通过 CLI/API 将 diff_comment 从「未解决」标记为「已解决」。
>
> **临时方案**：回复意见后，提醒用户在 GitCode 网页点击「已解决」按钮。当检视意见数 = 已解决数时，PR 才被后台允许合入。
>
> **待办**：gitcode CLI 增加 `gitcode pr resolve <PR> --comment <id>` 子命令后，替换为 CLI 操作。

### 6. Commit、Push 和 CI

commit 和 push 后运行 `msmodeling-ci-recovery` 监控 CI 闭环。

### 7. 提交汇总

```bash
gitcode pr comment <PR编号> -R <TARGET_REPO> --body-file "$TMPDIR/feedback-summary.md"
```

汇总内容：

```
## 检视意见处理汇总

- 已修复：N 条
- 延期：N 条（原因）
- 拒绝：N 条（原因）
- 验证：受影响测试已通过
- CI：<状态>
- 待手动 resolve：N 条（已在网页标记 / 待标记）
```

## 安全规则

- 不盲从 reviewer 建议；与 spec 冲突时以 spec 为准并说明。
- 禁止 force push，除非用户明确授权且使用 `--force-with-lease`。
- 不把行号当作唯一定位依据，需结合评论内容和当前代码。
- 每条 diff_comment 必须有回复，不能默默修改不回复。

## 完成标准

所有意见有明确状态和回复；接受项已验证；CI 已闭环或记录阻塞；汇总已提交；待手动 resolve 的意见已提醒用户。
