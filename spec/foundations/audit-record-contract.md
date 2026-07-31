# AI 审计记录契约

## 1. run-id

每次端到端或独立远端工作流生成唯一 `run-id`。推荐格式：

```text
issue-<number>-<yyyymmdd>-<short-id>
pr-<number>-<yyyymmdd>-<short-id>
```

## 2. 阶段标记

Issue/PR 评论使用以下隐藏标记：

```markdown
<!-- ai-native-run: run-id=<id> phase=<phase> mode=<guided|autonomous> -->
```

## 3. 阶段记录

```markdown
## AI 研发记录：<阶段>

- 输入事实：
- canonical/source/operation target：
- 执行动作：
- 关键证据：
- 结论：
- 用户决策：
- 产物：
- 下一阶段：
```

字段没有内容时写“不适用”，不得删除字段后伪装成完整记录。

## 4. 必须记录的事件

- 工作流开始、暂停、恢复和终止；
- 需求分析、设计选择和开发计划批准；
- 分支、commit、push、Issue 和 PR 创建；
- canonical repository、source repository、operation target 及跨仓关系；
- 本地构建、测试、pre-commit、安全检查和结果；
- 每轮 CI 的链接、状态、根因和修复 commit；
- Review findings、回复、解决或延期理由；
- Ready、审批建议和最终阻塞结论。

只读探查无需逐命令评论，但必须在阶段摘要中列出主要证据来源。

## 5. 安全

记录中禁止包含：

- Token、密码、私钥和认证文件内容；
- 未公开安全漏洞的利用细节；
- 用户本地绝对路径；
- 与任务无关的个人信息；
- 完整环境变量转储。

## 6. 恢复和幂等

恢复前必须读取现有评论，定位相同 `run-id` 的最后成功阶段。创建 Issue、PR 或重复评论前先验证资源是否已存在。
远端写操作返回不确定结果时，先查询远端事实再决定是否重试。
