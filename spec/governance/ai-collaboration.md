# AI 协作治理规范

## 1. 目标

AI 在本仓库中是受项目规范、权限边界和质量门禁约束的工程参与者。AI 可以分析、设计、开发、验证和操作
GitCode，但不能替代维护者承担授权、审批和最终质量责任。

## 2. 入口关系

- `AGENTS.md`：所有 AI 客户端的仓库级入口和任务路由。
- `CLAUDE.md`：Claude 适配器，仅转发到 `AGENTS.md` 和本规范。
- `.agents/skills/`：任务执行能力，不是项目规则源。
- `spec/`：强制规则。
- `docs/RFC/`、`docs/design/`：架构和实现设计。

客户端适配文件不得复制整套规则。发生冲突时以 `spec/` 为准。

## 3. 能力分层

### 3.1 领域 Skills

领域 Skills 处理 DeviceProfile、op mapping、模型适配、吞吐优化和 OptiX 等业务任务，不得直接操作 GitCode
远端。

### 3.2 GitCode Skills

GitCode Skills 只负责通过 `gitcode` CLI 读取或修改远端实体，不替代业务判断和项目质量门禁。

### 3.3 工作流 Skills

工作流 Skills 面向完整作业目标，组合领域 Skills、GitCode Skills、本地 Git、构建、测试和审计记录。每个
被组合的能力仍应可以独立触发。

## 4. 主体责任

- AI 作者自检不能作为独立 reviewer 结论。
- 多角色分析可以由一个 AI 执行，但必须明确标注为“多角色分析”，不能伪装为多个独立主体。
- 需要独立评审时，由不同的人或不同执行主体完成，并在记录中保留主体标识。
- `/lgtm`、`/approve`、审批和合并遵循项目权限，不因 AI 结论自动授予。

## 5. 事实核验

- Issue/PR 当前状态：通过 GitCode CLI 读取远端事实。
- 当前实现：读取任务工作树。
- 项目基线：读取 `origin/master`。
- PR 变更：使用 `gitcode pr diff`，本地代码只用于补充上下文。
- CI：读取 PR 流水线机器人评论，并使用 `gitcode-pipeline-analyzer`。
- 不存在或不可访问的资源必须明确报错，禁止补全或编造。

## 6. 变更同步

规则、工作流或 Skill 发生变化时，至少检查：

- `AGENTS.md`
- `spec/`
- `.agents/README.md`
- `README.md`
- `CONTRIBUTING.md`
- `.gitcode` Issue/PR 模板
- `docs/ai-native/`
