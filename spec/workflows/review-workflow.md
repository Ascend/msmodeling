# PR 检视与反馈工作流

## 多角色检视

根据变更启用以下角色：

- 需求与架构；
- 逻辑正确性；
- TensorCast/ServingCast/OptiX 领域语义；
- 性能、显存和仿真精度；
- 测试与回归；
- 安全；
- 文档、Specs 和 Skills；
- CI 与交付。

一个 AI 可以执行多角色分析，但必须标注为单一执行主体的多视角检查。

## 启动合入流程

合入流程由常驻服务 后台合入管理服务 驱动：在 PR 评论区评论 `/merge` 即启动加速流程，后台合入管理服务 自动完成 SIG 路由、reviewer 分配、CI 与 `lgtm`/`approved` 标签监控和催审，全程无需手动指派 assignee 或操作 GitCode 网页。

| 动作 | 谁做 | 命令 | 后续 |
|------|------|------|------|
| 请求检视 / 启动合入 | PR 作者或任何成员 | 评论 `/merge` | 后台合入管理服务 路由 SIG、分配 reviewer、飞书通知 |
| 检视通过 | reviewer | 评论 `/lgtm` | 后台合入管理服务 检测后自动通知 approver |
| 审批通过 | approver | 评论 `/approve` | 后台合入管理服务 检测后标记可合入 |

> **前置**：评论 `/merge` 前确认当前 head 的 CI 已全绿（`ci-pipeline-passed`/`docs-ci-pipeline-success`）；CI 未通过时先走 `ci-recovery-workflow.md` 修复至全绿，不启动合入。

agent **不执行 assignee 指派、不打 `sig:XXX` 标签**——SIG 路由、assignee 分配和标签由后台合入管理服务统一完成，agent 不获取变更文件或计算 SIG 归属。

## 检视步骤

1. CLI 读取 PR、评论、关联 Issue 和权威 diff。
2. 参考 `sig_ownership.json` 了解变更所属 SIG（信息性，不执行指派或打标签）。
3. 读取变更文件的必要上下文和相关规范。
4. 生成候选 finding。
5. 对每条 finding 核验准确性、影响、行号、严重度、建议和重复情况。
6. 行级 finding 使用 CLI inline comment；总体检视结论通过 `gitcode pr comment` 提交，检视通过时在同一 PR 评论 `/lgtm`，由 后台合入管理服务 自动通知 approver，不手动指派 approver。
7. 给出风险等级和合入建议，但未经授权不审批。

## Finding 格式

```text
[阻塞|高|中|建议] 问题标题
影响：
证据：
建议：
```

## 反馈处理

1. CLI 拉取 inline、discussion 和关联 Issue 评论。
2. 按严重度和文件形成修复清单。
3. 对合理意见修复并验证；不合理或不明确意见提供证据并回复。
4. 使用 `gitcode pr reply` 回复 discussion。
5. 每条意见回复后均使用 CLI resolve（含拒绝/延期，已解决评论 reviewer 仍可查阅）。
6. 汇总已修复、延期、拒绝及验证结果。
