# AI Native 工作流指南

## 准备

```bash
gitcode version
gitcode auth status
python scripts/ai/check_gitcode_cli.py --json
python scripts/ai/resolve_repository_context.py --json
```

Windows PowerShell 请使用 `gitcode`，不要使用别名 `gc`。

## 两种模式

- `guided`：默认，在 Issue 提交、分析结论、设计、远端写入、push、CI 修复和 Ready 前确认。
- `autonomous`：用户明确限定仓库、Issue、分支和目标后连续执行。

两种模式都不能自动越过安全问题、权限不足、门禁绕过、破坏性操作、审批或合并。

## 独立场景

| 说法示例 | 路由 |
|---|---|
| “我遇到一个模型适配问题，帮我整理并提交 Issue” | `msmodeling-issue-draft` |
| “分析我负责的开放 Issue” | `msmodeling-my-issues-review` |
| “实现 Issue 25，每一步先确认” | `msmodeling-issue-delivery` |
| “检视 PR 123，并提交准确的行内意见” | `sig-review` |
| “分析 PR 123 的流水线并修复” | `msmodeling-ci-recovery` |
| “处理 PR 123 尚未解决的检视意见” | `msmodeling-review-feedback` |

## 全自动授权示例

```text
以 Ascend/msmodeling 的 Issue 123 为范围，从 upstream/master 创建 codex/issue-123 分支，
push 到当前 origin Fork，并向 Ascend/msmodeling 创建 Draft PR。按 autonomous 模式推进到
ready-for-review，允许根据 openLiBing 失败修改代码并重复推送；不要审批、合并、关闭 Issue 或强制推送。
```

AI 会把授权范围写入 Issue/PR 评论。出现硬停止条件时仍会暂停。

## Fork 与主仓模式

Fork 模式：

```bash
git fetch upstream
git switch -c feat/example upstream/master
git push origin feat/example
gitcode pr create -R Ascend/msmodeling \
  --fork <fork-owner>/msmodeling --head feat/example --base master
```

主仓分支模式省略 `--fork`，但 source branch、canonical target 和 base 仍需显式确认。Fork 内部临时 PR
可以没有 CI；最终 canonical PR 必须完成 openLiBing。

## 暂停与恢复

可随时说“暂停并留下恢复点”。AI 应记录最后成功阶段、验证证据、工作树状态、未完成项和下一条安全命令。
之后说“继续 Issue 123 的 run-id …”，AI 会先通过 CLI 核验远端记录，再从第一个未完成阶段继续。

## 审计记录

每个完整工作流使用唯一 `run-id`。关键阶段在 Issue 或 PR 评论记录输入事实、动作、证据、结论、用户决策、
产物和下一阶段。评论不包含 Token、本地绝对路径或敏感漏洞细节。
