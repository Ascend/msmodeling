---
name: msmodeling-issue-draft
description: 将开发者的模糊问题或需求澄清、分析并整理为高质量 MindStudio-Modeling Issue，确认后通过 GitCode CLI 提交。
metadata:
  version: 1.1.0
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
4. 读取 [sig-review/sig_ownership.json](../sig-review/sig_ownership.json)（SIG 目录归属表和 chair 名单）。
5. 执行 `gitcode version`、`gitcode auth status` 和 `gitcode schema "issue create"`。
6. 解析 repository context；正式 Issue target 为 `Ascend/msmodeling`，Fork 调测 target 必须显式指定。

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

8. **SIG 路由**：根据 Issue 标题、正文和标签推断归属 SIG（规则见下文），输出 chair 列表。
9. 向用户展示最终标题、正文、元数据和 SIG 路由结果，先 dry-run 预览，确认后正式创建。

   ```bash
   # dry-run 预览（含 assignee）
   gitcode issue create -R <TARGET_REPO> \
     --title "<title>" --body-file <file> \
     --assignee <chair> --label feature --dry-run --json

   # 用户确认后正式创建
   gitcode issue create -R <TARGET_REPO> \
     --title "<title>" --body-file <file> \
     --assignee <chair> --label feature --json
   ```

   > fork 用户在主仓可能无 assignee 写权限（返回 403）。无权限时省略 `--assignee`，改为创建后评论 @chair。
10. 使用 `gitcode issue view <number> --json` 回读验证。
11. **飞书通知**（可选）：若 `gitcode lark doctor` 通过，向 chair 所在飞书群发送通知：

    ```bash
    gitcode lark send --text "Issue #<number> <title> 已指派给 <chair>（SIG: <sig_name>）<url>"
    ```

    无 lark 配置时跳过；assignee 的站内信由 GitCode 自动发送。

## SIG 路由规则

Issue 无文件 diff，路由依据按优先级从高到低：

| 优先级 | 依据 | 示例 |
|--------|------|------|
| 1 | `sig:XXX` 标签直通 | 标签 `sig:模型适配` → chair: ChenHuiwen |
| 2 | 正文提及 `sig_ownership.json` 中的路径 | 正文含 `tensor_cast/layers/` → 模型适配 |
| 3 | 标题/正文关键词匹配 | 标题含"吞吐"/"throughput" → Throughput寻优 |
| 4 | `fallback_sigs` 顶层目录推断 | 正文主要涉及 `optix/` → Optix |
| 5 | 无匹配 | 不指派，评论建议归属 SIG |

**关键词→SIG 映射**（大致正确即可，chair 不对会自行转交）：

| 关键词 | SIG |
|--------|-----|
| 检视/review/PR 检视 | 文档与Skill |
| 测试/test/CI/pre-commit/gitleaks | 测试与基础设施 |
| 文档/doc/Skill/AGENTS | 文档与Skill |
| 设备/device/DeviceProfile/拓扑 | ServingCast |
| 吞吐/throughput/optimizer | Throughput寻优 |
| optix/服务化寻优 | Optix |
| profiling/算子/kernel/microbench | 实测算子查询 |
| perf_data/工具链 | 实测算子工具链 |
| 模型适配/attention/MLA/MoE/transformer | 模型适配 |
| 视频/diffuser/video | 视频生成 |
| UI/web_ui | UI |

**路由结果处理**：

- 单 SIG → 指派 chair 为 assignee
- 跨 SIG → 指派多个 chair，评论说明
- chair == Issue 作者 → 改指派 reviewer
- 无匹配 → 不指派，评论 @ 对应 chair 建议归属

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
- SIG 路由结果（归属 SIG、chair、assignee）；
- dry-run 结果；
- 创建后的 Issue 编号和 URL；
- 飞书通知状态（已发送 / 跳过）。

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
- SIG 路由结果已展示（或标注无匹配）；
- 用户确认最终草稿；
- CLI 创建和回读成功；
- assignee 已指派（或无权限时评论 @chair）；
- 飞书通知已发送或已记录跳过原因。
