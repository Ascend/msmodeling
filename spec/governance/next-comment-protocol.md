# `/next` 评论协议

成员在 PR/Issue 评论中写一行 `/next` 声明责任移交 + 请求动作。仓外本地工具（不上库）扫描这些行并用飞书群 @ 通知责任人 + 发布看板。

> **PR 侧 vs Issue 侧**：PR 合入流程的责任移交通知（分配 reviewer、CI 通过后通知 approver、催审、返工通知作者等）已由后台合入管理服务基于 `/merge`、CI label 和 diff_comment 统一接管，agent 不在 PR 评论写 `/next`，避免双重通知。`/next` 协议在 **Issue 侧**（Issue 评审、分类、退回、转交）仍由 agent 主动写入。

## 语法

```
/next <gitcode_login> <verb> [自由文本备注]
```

- 仅解析**评论正文首列**以 `/next ` 开头的行；其余自由文本忽略。
- 一条评论可含多行 `/next`，每行独立处理。
- `<verb>` 必须在下方 verb 表中，否则跳过。
- PR 实体只认 PR 列为 ✓ 的 verb；Issue 实体只认 Issue 列为 ✓ 的 verb。
- `<gitcode_login>` 必须在操作者的 `people.local.yaml` 映射里，否则记 `unroutable`、不通知。
- `/next` 是**通知触发 + 责任移交信号**，不等于状态机/SLA/审批。审批/合入仍按仓库规范（`/lgtm`、`/approve`、独立 reviewer、CI 门禁）。

## 去重

去重键 = `comment_id:verb:login`。同一条评论即使被编辑或被工具多次扫描，只通知一次。若同一评论有多行 `/next`，按 `(comment_id, verb, login)` 各通知一次。

## verb 表（7）

| verb | PR | Issue | 含义 | 超期阈值(h) |
|---|---|---|---|---|
| `review` | ✓ | ✓ | 请检视/评审 | 24 |
| `ack` | ✓ | ✓ | 确认接手 | 8 |
| `approve` | ✓ | ✓ | 请审批 | 48 |
| `return` | ✓ | ✓ | 退回作者 | 24 |
| `forward` | ✓ | ✓ | 责任转交 | 24 |
| `block` | ✓ | ✓ | 标记阻塞 | 0（不催） |
| `reject` | — | ✓ | 标记 wontfix | 0（不催） |

共用 6 个（`review` `ack` `approve` `return` `forward` `block`），Issue 专有 `reject`。检视通过后交审批用 `/next <approver> approve 检视已通过`，无需单独的 lgtm verb——reviewer 直接用 GitCode 原生 `/lgtm` 命令通过即可。

## 超期

超期 = 已停留时长 > 阈值。`block`/`reject` 阈值 0 = 永不催（终态/等待态）。静默时段 22:00–08:00（Asia/Shanghai）超期通知顺延；新通知不顺延（即时）。阈值在仓外 `policy.yaml` 可调，此处仅作示例。

## 范围

本协议只定义**评论格式 + verb 枚举 + 去重规则**。扫描器、飞书通知、看板、定时器、`people.local.yaml` 映射、`policy.yaml` 阈值均为仓外本地工具实现，不上库。
