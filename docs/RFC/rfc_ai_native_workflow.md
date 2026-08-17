# RFC：msmodeling AI Native 工作流

- 状态：实施中
- canonical repository：`Ascend/msmodeling`

## 1. 背景

仓库已有领域 Skills、SIG 检视和工程规范，但缺少统一的 AI 入口、执行授权、远端边界、审计契约以及从
Issue 到 PR/CI 的编排层。部分能力直接调用 GitCode API，导致认证方式、命令语义和审计方式不一致。

本 RFC 定义 msmodeling 的 AI Native 工作流体系。`gitcode` CLI 在其中是**底座工具**——与 GitCode
平台交互的统一基础设施层，负责远端操作的认证、命令语义和审计一致性；工作流的核心是 Skills 编排与
`spec/` 契约，CLI 只为远端写操作提供受控通道，不充当工作流的驱动中心。

## 2. 目标与非目标

目标：

- 让开发者用自然语言独立触发 Issue、开发、PR 检视、反馈处理和 CI 恢复；
- 让端到端流程组合这些独立能力，并允许人工随时暂停、修改、接管或恢复；
- 所有 GitCode 远端操作统一走 `gitcode` CLI 底座；
- 在 Issue/PR 评论中记录关键决策、证据和状态；
- 保留 openLiBing，并使用 `gitcode-pipeline-analyzer` 分析流水线。

非目标：

- 不自动获得 reviewer、approver 或合并权限；
- 不绕过测试、门禁和安全规则；
- 不在本 RFC 中迁移到 GitCode Actions；
- 不把作者自检伪装成独立评审。

## 3. 架构

```text
自然语言意图
  -> AGENTS.md 路由
  -> spec 授权、边界和工作流
  -> 工作流 Skill 编排
       -> 领域 Skills
       -> GitCode CLI Skills（底座）
       -> 本地 Git / 测试 / 构建
       -> openLiBing analyzer
  -> Issue / PR 审计记录
```

架构分为五层：

1. `AGENTS.md`：跨客户端统一入口和意图路由；
2. `spec/`：不变量、授权、审计和状态机；
3. `msmodeling-*` 工作流 Skills：面向作业目标的编排；
4. `gitcode-*` 与领域 Skills：可独立调用的原子能力；
5. CLI、Git、pytest、pre-commit、build 和 openLiBing：执行与证据层。

## 4. 组件

| 组件 | 职责 |
|---|---|
| `AGENTS.md` | 识别自然语言意图并路由，不复制完整实现步骤 |
| `spec/foundations/` | 执行授权、审计记录和 CLI 边界 |
| `spec/workflows/` | 各场景状态机、输入、门禁和完成条件 |
| `.agents/skills/msmodeling-*` | 组合原子能力，维护阶段推进 |
| `.agents/skills/gitcode-*` | GitCode CLI 底座原子操作 |
| `.agents/skills/sig-review` | 评论 /merge 启动合入和项目专项检查（SIG 路由由 后台合入管理服务 驱动） |
| `.agents/repository-contract.json` | canonical、source 发现和 operation target 规则 |
| `scripts/ai/` | 静态边界、Skills 元数据和 CLI 能力校验 |
| `.agents/gitcode-skills.lock.json` | 上游 Skills 来源、版本和本仓适配 |

## 5. 关键决策

### 5.1 GitCode CLI 是远端交互底座

`gitcode` CLI 是与 GitCode 平台交互的唯一底座工具，负责认证、命令语义和审计的统一。Typed command
优先；只有 CLI 没有 typed command 时才允许 `gitcode api`。Skill 和仓库脚本不得直接访问
`api.gitcode.com`。执行前通过 `version`、`auth status` 和 `schema` 做能力协商。CLI 是基础设施层，
不构成工作流的核心驱动。

### 5.2 三层仓库身份

canonical repository 固定为 `Ascend/msmodeling`；source repository 从当前可写 Git remote 动态识别；
operation target 由每次远端操作显式选择。Fork 用于开发和临时调测，正式 Issue、PR、Review 和 CI 面向
canonical repository。有主仓权限时 source 可以等于 canonical。

### 5.3 guided 默认，autonomous 显式授权

默认在关键阶段确认。用户可以为当前仓库、Issue、分支和目标授权连续执行；安全、权限、破坏性操作、
门禁绕过、审批和合并始终停止。

### 5.4 审计记录是工作流状态

每次执行生成 `run-id`，在 Issue/PR 评论中记录阶段、事实、动作、证据、用户决策、产物和下一阶段。
恢复时以远端评论和本地 Git 状态交叉核验，写操作遵循幂等检查。

### 5.5 openLiBing 保持不变

仓库当前不使用 GitCode Actions。AI 从 PR 机器人评论识别 openLiBing 任务，并通过
`gitcode-pipeline-analyzer` 获取和分析日志；修复后以新提交重新触发和监控。

## 6. 状态与恢复

工作流状态不依赖本地私有数据库。远端审计评论保存 `run-id` 和成功阶段，本地 Git 保存分支、commit 和
工作树状态。恢复步骤：

1. 通过 CLI 读取 Issue/PR 和评论；
2. 找到相同 `run-id` 的最后成功阶段；
3. 核验资源状态、head SHA、分支和工作树；
4. 从第一个未完成门禁继续；
5. 若远端与本地证据冲突，停止并请求人工裁决。

## 7. CLI 能力协商

`scripts/ai/check_gitcode_cli.py` 依次解析显式 `--binary`、`GITCODE_BIN` 和 `PATH`。发布版要求至少
`0.8.0`；开发版以实际 schema 为准。检查只读取版本、认证状态和命令 schema，不读取 Token。

`scripts/ai/resolve_repository_context.py` 读取仓库契约，从 Git remote 解析 source repository。只读操作可默认
使用 canonical；远端写操作缺少显式 `--repo` 时 fail-fast。

## 8. 行内评论

`gitcode-pr-inline-review` 将 `--position` 定义为新版本文件实际行号。评论前从 `gitcode pr diff` 解析
hunk 的新文件行范围，只允许定位到新增或修改行。无法确定行号时改为总体评论，不尝试多个位置。

## 9. openLiBing 诊断

`msmodeling-ci-recovery` 通过 CLI 读取 PR 评论并定位 openLiBing 任务。日志获取和诊断委托给
`gitcode-pipeline-analyzer`。每轮保存任务链接、状态、失败签名、根因、修复 commit 和本地验证。

## 10. 工作流

- 模糊描述到 Issue：补充问题 → 代码核验 → 草稿 → 人工确认 → CLI 创建；
- 我的 Issue 评审：识别当前用户 → 查询负责人 Issue → 证据分析 → 接受/拒绝/补充；
- Issue 交付：核验 → 分析 → 设计 → 计划 → 开发 → 本地门禁 → Draft PR → CI → Ready；
- PR 检视：SIG 路由 → 多角色检查 → 行内评论 → 总体风险和合入建议；
- CI 恢复：读取评论 → 日志分析 → 根因 → 修复 → 本地复验 → 推送 → 重复监控；
- Review 反馈：归类意见 → 验证合理性 → 修改或解释 → 回复 → 复验。

## 11. 校验

```bash
python scripts/ai/check_gitcode_cli.py --json
python scripts/ai/validate_skills.py --json
python scripts/ai/validate_remote_boundary.py --json
uv run pytest tests/regression/scripts/test_ai_native_validators.py
uv run pre-commit run --all-files
```

`validate_remote_boundary.py` 只验证仓库内 AI 执行资产，不禁止文档引用 GitCode 网页链接。它禁止 Skills
中的直接 API 主机、旧 `review_api.py` 和旧私有 Token 配置路径。

## 12. 风险与缓解

| 风险 | 缓解 |
|---|---|
| CLI 版本或参数变化 | lock 文件、最低版本、运行时 schema 校验 |
| 重复创建或评论 | run-id、先查询后写、失败后核验远端 |
| 行内评论错位 | 使用新版本实际行号并校验 diff 新增行 |
| AI 越权 | guided 默认、autonomous 范围绑定、硬停止点 |
| 文档漂移 | 事实源矩阵、Skills 元数据与自动校验 |
| CI 无限重试 | 每轮记录根因；相同失败或无新证据时停止并请求决策 |

## 13. 验收标准

- 所有目标场景既可独立触发，也可由端到端工作流编排；
- `AGENTS.md`、规范、Skills、贡献指南和模板一致；
- 仓库校验器能发现 Skills 元数据缺失和 GitCode 远端边界绕过；
- CLI 兼容性检查能验证版本、认证和所需 command schema；
- 示例覆盖本仓实际业务模块；
- 后续 PR 保留可恢复的实施证据。
