# 行内检视意见 → 本地行号映射

本文件为 `sig-review`（Step 4 行号二次检查）和 `msmodeling-review-feedback`（修复时定位本地行）共享的行号计算参考。GitCode inline comment 的 `position` 指向 PR diff，而非本地工作区行号；直接套用会错位。

## CLI 契约

```bash
gitcode pr comment <PR> -R <TARGET_REPO> --path <file> --position <new-file-line> --body-file <finding.md> --json
```

- `--path`：PR diff 中的文件路径；
- `--position`：新版本文件的实际行号，不是 unified diff 内偏移；
- 删除行没有新版本行号，不能作为 inline comment 目标。

提交前执行 `gitcode schema "pr comment"` 与 `gitcode pr diff <PR> -R <TARGET_REPO>`，schema 定义与本文不一致时停止并报告兼容性问题。

## hunk 解析（求新版本行号）

对每个文件单独解析 unified diff 的 hunk：

```text
@@ -old_start,old_count +new_start,new_count @@
```

维护 `old_line` 和 `new_line`：

- 上下文行：两者同时加一；
- `+` 行：当前目标行号为 `new_line`，然后 `new_line + 1`；
- `-` 行：仅 `old_line + 1`；
- 新 hunk：重置为 hunk header 中的起始行。

只有 `+` 行或确实属于变更范围的上下文行才适合作为 finding 锚点。优先锚定导致问题的新增行。

## 定位策略

1. CLI 获取 PR、head SHA 和权威 diff；
2. 验证目标文件存在于 diff；
3. 解析目标 hunk，计算新版本实际行号；
4. 读取本地 PR head 文件，核对该行内容与 finding 一致；
5. 检查已有评论，避免重复；
6. 无法可靠定位时，改为总体 review comment，并明确说明无法锚定。

## 安全约束

- 禁止使用“估计位置”“大约行号”或向上/向下偏移试错；
- 禁止直接调用 GitCode API；
- 禁止将同一 finding 同时作为 inline 和总体评论重复提交；
- 评论不得复制 diff 中的凭证或其他敏感内容。
