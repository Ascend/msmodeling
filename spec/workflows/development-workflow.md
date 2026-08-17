# AI 开发交付工作流

## 状态机

```text
verified
-> analyzed
-> designed
-> planned
-> in-progress
-> locally-verified
-> draft-pr
-> ci-passed
-> self-checked
-> ready-for-review
```

## 阶段

### 1. 远端事实核验

解析 canonical/source/operation target，使用 CLI 读取正式 Issue、评论和关联 PR，确认问题未被其他实现闭环。
以 canonical `master` 的本地跟踪分支（通常为 `upstream/master`）判断基线是否已包含对应代码。

### 2. 需求分析

输出范围、非目标、验收标准、依赖、风险和待确认项，并记录到 Issue。

### 3. 方案设计

重大架构变更写 RFC，具体实现写 design。列出候选方案和权衡，得到用户选择。

### 4. 开发计划

计划必须包含文件、实现步骤、测试、文档和回滚。任何时刻只推进一个明确步骤。

### 5. 分支与实现

- 不在 `master` 直接开发；
- Fork 模式从最新 canonical 基线建分支并 push 到 source repository；
- 主仓模式从最新 canonical 基线建主仓开发分支；
- 遵守 AGENTS 中的业务架构约束；
- 修改行为时同步测试和文档。

### 6. 本地门禁

按改动范围执行：

```bash
uv run pre-commit run --all-files
uv run pytest <affected tests>
python build.py test
python build.py
```

全量测试或构建成本不适合当前环境时，必须运行风险匹配的最小集合并记录未运行项，不能写“全部通过”。

### 7. Commit、Push 和 PR

Commit 使用 Conventional Commits 并 sign-off。push 面向 source repository，最终 PR 面向 canonical repository，
按 `pr-workflow.md` 执行。

### 8. CI 和自检

PR 创建后**默认立即**按 `ci-recovery-workflow.md` 触发 CI 并监控至闭环：触发、监控、失败诊断和修复（含 commit、push、重新触发）全程默认自动，直到全绿或 blocked，无需用户逐次催促或确认 push。CI 全绿后向用户报告本轮新增的修复 commit 数。作者自检必须包含本地证据、CI、风险、文档同步和未覆盖项。

## 完成标准

工作流只有在 canonical PR 达到 CI 通过后才推进到 `ready-for-review`。Fork 内部 PR 只作为 staging 证据。
独立评审、审批和合并不属于作者自动交付的默认完成范围。
