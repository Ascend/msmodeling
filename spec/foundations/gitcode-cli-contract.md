# GitCode CLI 操作契约

## 1. 仓库身份

- canonical repository：`Ascend/msmodeling`
- 默认分支：`master`
- source repository：从当前可写 Git remote 动态识别，通常为 Fork 的 `origin`
- operation target：当前 CLI 命令实际读写的 `owner/repository`
- 跨平台命令：`gitcode`
- 最低推荐版本：`0.8.0`

`.agents/repository-contract.json` 是仓库身份的机器可读契约。个人 Fork 名称不得写入项目强制规范。

Windows PowerShell 不使用 `gc`，因为它是 `Get-Content` 的内置别名。

## 2. 启动检查

所有 GitCode 工作流开始前执行：

```bash
gitcode version
gitcode auth status
gitcode schema "<required command>"
python scripts/ai/resolve_repository_context.py --json
```

开发版 CLI 没有语义版本时，必须通过所需命令 schema 的逐项验证。

## 3. 目标选择

- 正式 Issue、PR、Review 和 CI 默认面向 canonical repository。
- 本地分支 push 到 source repository；有主仓写权限时 source 可以等于 canonical。
- Fork 临时调测必须显式把 operation target 设置为 Fork。
- 所有远端写操作都必须显式传入 `-R <operation-target>`，并在执行前展示目标仓。
- canonical 默认值只适用于只读操作，不代表写授权。
- 跨仓 PR 必须显式提供 canonical target、Fork source、head 和 base。

可用以下命令校验写入上下文：

```bash
python scripts/ai/resolve_repository_context.py \
  --write --repo Ascend/msmodeling --json
```

## 4. 远端边界

以下操作必须经 CLI：

- Issue：list/view/create/edit/comment/label/close/reopen；
- PR：list/view/diff/create/edit/comment/review/reply/label/ready/resolve/unresolve；
- Commit 和关联关系查询；
- GitCode 流水线状态查询。

Typed command 未覆盖的能力只能通过 `gitcode api` 兜底。Skill 和脚本禁止直接访问 `api.gitcode.com`。

openLiBing 不属于 GitCode 实体写入面。canonical PR 的日志只允许由 `gitcode-pipeline-analyzer` 以只读方式访问。
Fork 内部临时 PR 没有 CI 时可以记录为 staging `not-applicable`，但不能据此把最终交付标记为 CI 通过。

## 5. 输出和正文

- 机器消费优先 `--json`。
- 多行正文使用 `--body-file`、`--comment-file` 或 stdin。
- 支持 `--dry-run` 的创建操作先 dry-run。
- 资源编号、URL、head SHA 和评论 ID 从 JSON 结果提取，不解析彩色文本。

## 6. 认证安全

AI 允许执行 `gitcode auth status`，禁止：

- `gitcode auth token`；
- `gitcode auth status --show-token`；
- 读取认证配置文件；
- 打印 `GC_TOKEN` 或 `GITCODE_TOKEN`；
- 要求用户把 Token 粘贴到对话。

## 7. inline comment 契约

当前 CLI 的 `gitcode pr comment --path --position` 中：

- `--path` 是 PR diff 中的文件路径；
- `--position` 是新版本文件的实际行号；
- 不是 unified diff 内的偏移量。

提交前必须确认目标行属于 PR 新增或修改范围，并先用 `gitcode pr diff` 获取权威 diff。
