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

## 检视步骤

1. CLI 读取 PR、评论、关联 Issue 和权威 diff。
2. 使用 `sig_ownership.json` 路由相关 SIG。
3. 读取变更文件的必要上下文和相关规范。
4. 生成候选 finding。
5. 对每条 finding 核验准确性、影响、行号、严重度、建议和重复情况。
6. 行级 finding 使用 CLI inline comment；总体结论使用 `gitcode pr review`。
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
5. 只有确认解决后才使用 CLI resolve。
6. 汇总已修复、延期、拒绝及验证结果。
