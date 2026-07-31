---
name: msmodeling-issue-draft
description: 将开发者的模糊问题或需求澄清、分析并整理为高质量 MindStudio-Modeling Issue，确认后通过 GitCode CLI 提交。
metadata:
  version: 1.0.0
  source: issue-25-ai-native
---

# msmodeling Issue 草拟

## 适用场景

- “我发现一个问题，帮我提 Issue”
- “把这个想法整理成需求”
- “分析代码后提交 RFC”

## 前置条件

1. 阅读 `spec/workflows/issue-workflow.md`。
2. 阅读 `spec/foundations/gitcode-cli-contract.md`。
3. 读取 `.gitcode/ISSUE_TEMPLATE/` 中对应模板。
4. 执行 `gitcode version`、`gitcode auth status` 和 `gitcode schema "issue create"`。
5. 解析 repository context；正式 Issue target 为 `Ascend/msmodeling`，Fork 调测 target 必须显式指定。

## 工作流程

1. 从用户描述中提取问题、期望、影响、环境、复现、范围和约束。
2. 只询问会改变 Issue 可实施性或验收结果的缺失信息。
3. 使用 `gitcode issue list -R <TARGET_REPO> --state all --json` 做重复检查；必要时本地过滤。
4. 分析相关代码、测试、文档和历史，明确标注事实、推断和待确认项。
5. 按 Bug、Feature 或 RFC 模板生成完整草稿。
6. 使用 `gitcode label list` 获取真实标签；没有合适标签时不添加。
7. 将正文写入临时 UTF-8 文件，执行：

   ```bash
   gitcode issue create -R <TARGET_REPO> \
     --title "<title>" --body-file <file> --dry-run --json
   ```

8. 向用户展示最终标题、正文和元数据，得到明确确认后正式创建。
9. 使用 `gitcode issue view <number> --json` 回读验证。

## 模板与高级字段

优先使用 `.gitcode/ISSUE_TEMPLATE/` 中仓库定义的模板；仓库无对应模板时按下述骨架起草，正文写入临时 UTF-8 文件后用 `--body-file` 提交。

Bug 骨架：

```markdown
## Problem

## Reproduction
1.
2.

## Expected

## Actual

## Environment
- GitCode CLI:
- OS:
- Shell:

## Impact
```

Feature 骨架：

```markdown
## Background

## Proposal

## Acceptance Criteria
- [ ] ...

## Alternatives
```

高级字段（按需，以仓库实际支持的为准，先查 `gitcode schema "issue create"`）：

```bash
gitcode issue create -R <TARGET_REPO> --title "<title>" --body-file <file> --security-hole --json
gitcode issue create -R <TARGET_REPO> --title "<title>" --body-file <file> --issue-type "需求" --issue-severity "高" --json
```

## 输出

- 重复检查结论；
- 最终 Issue 草稿；
- dry-run 结果；
- 创建后的 Issue 编号和 URL。

## 安全规则

- 禁止直接调用 GitCode API。
- 禁止在正文中写入 Token、本地绝对路径和未公开漏洞细节。
- 用户确认前不得创建 Issue。
- 创建前必须展示 `<TARGET_REPO>`；canonical 配置不能替代写入授权。
- CLI 返回不确定结果时，先查询远端，不得直接重试创建。

## 完成标准

- 信息足以实施或明确列出待确认项；
- 验收标准可测试；
- 已执行 dry-run；
- 用户确认最终草稿；
- CLI 创建和回读成功。
