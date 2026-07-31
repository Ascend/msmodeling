# AI Native 架构

## 分层

```text
开发者
  │ 自然语言意图、确认、暂停、接管
  ▼
AGENTS.md ── 统一入口与任务路由
  ▼
spec/ ── 授权、远端边界、审计、状态机
  ▼
工作流 Skills ── Issue / Delivery / PR Review / CI / Feedback
  ├── 领域 Skills ── DeviceProfile / op mapping / 模型适配 / OptiX / 性能优化
  ├── GitCode Skills ── Issue / PR / inline review / pipeline
  └── 本地工具 ── Git / pytest / pre-commit / build
  ▼
GitCode CLI + openLiBing analyzer
  ▼
Issue / PR 可审计记录
```

## 设计性质

- **端到端且可拆分**：工作流负责编排，每个原子 Skill 仍可单独调用。
- **人机共治**：默认逐阶段确认；明确授权后可连续推进；人可随时暂停、改变计划或接管。
- **事实优先**：远端状态来自 CLI，PR 变更来自 `gitcode pr diff`，代码事实来自正确的工作树或基线。
- **证据驱动**：测试、构建、CI 和评审结论必须关联证据，未知项不写成通过。
- **最小权限**：AI 不自动审批、合并、强推、关闭 Issue 或绕过门禁。

## 仓库身份

- canonical repository：`Ascend/msmodeling`，是正式 Issue、PR、Review 和 CI 的目标。
- source repository：从当前可写 Git remote 动态识别，可以是个人 Fork 或主仓开发分支。
- operation target：每次 GitCode CLI 操作实际使用的仓库；写操作必须显式选择和确认。

Fork 内部 PR 只用于 staging，不能替代 canonical PR 的 openLiBing 结果。

## Superpowers 与 OpenSpec

当前方案不要求引入 Superpowers。交互检查点、自动模式、暂停和恢复已由执行规范与工作流 Skills 定义。

当前方案也不引入 OpenSpec 运行时依赖。仓库已有 `spec/`、RFC、design 三层事实源，继续使用它们能避免
双写和冲突。结构化需求、验收条件与方案文档仍是必需产物，只是采用仓库原生格式。
