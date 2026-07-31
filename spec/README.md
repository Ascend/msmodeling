# msmodeling 规范入口

`spec/` 是 MindStudio-Modeling 的强制规范源。AI、人工贡献者、Skills、自动化脚本与文档发生冲突时，按
[单一事实来源矩阵](./governance/source-of-truth-matrix.md) 裁决。

## 最小阅读集

所有 AI 开发任务先读：

1. [AI 协作治理](./governance/ai-collaboration.md)
2. [AI 执行授权](./foundations/ai-execution-policy.md)
3. [GitCode CLI 契约](./foundations/gitcode-cli-contract.md)
4. 与任务对应的 `workflows/` 文档

## 目录

```text
spec/
├── README.md
├── governance/
│   ├── source-of-truth-matrix.md
│   └── ai-collaboration.md
├── foundations/
│   ├── ai-execution-policy.md
│   ├── audit-record-contract.md
│   └── gitcode-cli-contract.md
└── workflows/
    ├── issue-workflow.md
    ├── issue-review-workflow.md
    ├── development-workflow.md
    ├── pr-workflow.md
    ├── review-workflow.md
    └── ci-recovery-workflow.md
```

## 任务路由

| 任务 | 必读规范 | 执行 Skill |
|---|---|---|
| 模糊需求转 Issue | `issue-workflow.md` | `msmodeling-issue-draft` |
| 评审自己负责的 Issue | `issue-review-workflow.md` | `msmodeling-my-issues-review` |
| 指定 Issue 完成交付 | `development-workflow.md`、`pr-workflow.md`、`ci-recovery-workflow.md` | `msmodeling-issue-delivery` |
| PR 检视 | `review-workflow.md` | `sig-review` |
| CI 失败闭环 | `ci-recovery-workflow.md` | `msmodeling-ci-recovery` |
| 处理检视意见 | `review-workflow.md` | `msmodeling-review-feedback` |

## 基本规则

- canonical repository 为 `Ascend/msmodeling`，默认分支为 `master`。
- source repository 从当前 Git remote 动态识别；operation target 在远端写入前显式指定。
- GitCode 远端实体操作必须通过 `gitcode` CLI。
- 面向 canonical repository 的最终 PR 必须完成 openLiBing CI，日志通过 `gitcode-pipeline-analyzer` 分析。
- 默认使用 `guided` 模式；`autonomous` 必须由用户明确授权。
- 没有证据不得推进状态，没有独立检视不得把作者自检表述为审批。
- 强制规则写入 `spec/`，架构提案写入 `docs/RFC/`，实现设计写入 `docs/design/`。
