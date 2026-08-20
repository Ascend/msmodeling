# MindStudio-Modeling Agent Guide

本文件是 Codex、Claude、Cursor、OpenCode 和其他通用 AI 代理的仓库级统一入口。

AI 首次处理本仓任务时必须完整阅读本文件。强制规则以 [spec/README.md](spec/README.md) 为入口；人类贡献规范见
[CONTRIBUTING.md](CONTRIBUTING.md)。

## 1. 项目与远端

MindStudio-Modeling（msmodeling）是昇腾 AI 模型性能仿真与分析框架：

| 组件 | 定位 | 目录 |
|---|---|---|
| TensorCast | PyTorch 模型性能仿真、性能模型和模型适配 | `tensor_cast/` |
| ServingCast | 系统级推理服务仿真和吞吐优化 | `serving_cast/` |
| OptiX | 服务化实测参数自动寻优 | `optix/` |
| Web UI | 可视化配置、运行和结果展示 | `web_ui/` |

GitCode 配置：

- canonical repository：`Ascend/msmodeling`
- 默认分支：`master`
- source repository：从当前可写 Git remote 动态识别，贡献者通常使用 Fork 的 `origin`
- operation target：每次远端写操作必须显式指定；正式 Issue、PR、Review 和 CI 面向 canonical repository
- GitCode 命令：`gitcode`
- canonical PR 流水线：openLiBing，通过 PR 机器人评论反馈

禁止在 `master` 直接开发。

## 2. 读取顺序

1. 本文件；
2. [spec/README.md](spec/README.md)；
3. 与任务对应的 `spec/workflows/`；
4. 对应 `.agents/skills/<skill>/SKILL.md`；
5. 任务相关代码、测试和文档；
6. `.loop/memory/lessons.md` 中相关经验。

事实源冲突时按 [source-of-truth-matrix.md](spec/governance/source-of-truth-matrix.md) 裁决。

## 3. AI Native 路由

以下自然语言意图必须自动进入对应工作流，不等待用户知道 Skill 名称：

| 用户意图 | 工作流 Skill |
|---|---|
| 描述问题、需求或想法并希望提 Issue | `msmodeling-issue-draft` |
| 分析自己负责的开放 Issue | `msmodeling-my-issues-review` |
| 指定 Issue 完成分析、开发、PR 和 CI | `msmodeling-issue-delivery` |
| 指定 PR 做深度检视、行内评论和风险评估 | `sig-review` |
| 查看、分析或修复 PR 流水线 | `msmodeling-ci-recovery` |
| 处理 PR 检视意见 | `msmodeling-review-feedback` |
| 请求检视或启动合入流程 | `sig-review` |

| Skill | 触发词 | 用途 |
|-------|--------|------|
| `device_config` | `/device_config` 或"我要导入新的设备拓扑" | 通过自然语言将硬件规格转换为 TensorCast `DeviceProfile` |
| `op-mapping` | `/op_mapping` 或"生成 op_mapping.yaml" | 将 TensorCast 仿真算子映射到 NPU profiling 内核类型 |
| `microbench` | `/microbench` 或"生成 xxx_run.py" | 从 profiling CSV 生成可在 NPU 上重放的 run script |
| `profiling-database-axis-density` | "新增采集轴"、"修改轴密度"、"检查数据库轴密度" | 定标 profiling database 轴范围、密度和 exact 锚点 |
| `msmodeling-env-installer` | "安装 msmodeling 环境"、"uv sync" | 安装并验证当前仓库开发环境、依赖和必要环境变量 |
| `gitcode-cli-installer` | "安装 gitcode-cli"、"gitcode 认证"、"gc auth login" | 安装和认证 gitcode CLI（npm + auth login + lark），一次性配置 |
| `model-adaptation` | "接入新模型"、"生成 ModelProfile"、"处理 doctor report" | 从仿真命令和 raw profiling 出发，完成 TensorCast 新模型适配流程 |
| `text-generate-executor` | "跑 text_generate"、"验证 best row"、"导出 trace" | 生成并执行 `python -m cli.inference.text_generate` 单点验证命令 |
| `throughput-optimizer-executor` | "搜索最佳 TP/EP"、"硬件对比"、"PD 配比优化" | 生成并执行 `python -m cli.inference.throughput_optimizer` 吞吐规划命令 |
| `throughput-optimizer-explainer` | "结果是否合理"、"为什么硬件不同"、"Cube/Vec/Comm/Mem 瓶颈" | 解释 optimizer 结果，并将 best row 映射到 `text_generate` 验证命令 |
| `optix-deploy` | "部署 optix"、"安装服务化自动寻优工具" | 安装并验证 msmodeling optix 服务化自动寻优工具 |
| `optix-config` | "配置 config.toml"、"设置 MindIE/vLLM 寻优字段" | 自动修改 optix `config.toml` 的寻优参数、target 和 benchmark 配置 |
| `optix-param-recommend` | "推荐 optix 参数"、"生成寻优范围" | 根据硬件、模型、负载和目标推荐 MindIE/vLLM 寻优参数与配置片段 |
| `sig-review` | "请求检视"、"启动合入"、"检视PR {number}"、"review PR {number}"、"分析PR {number}的检视意见" | 评论 /merge 启动合入流程（后台 MergeTrack 工具跟踪后续状态）、代码检视、检视意见分析，不指派 assignee，支持 cursor/claude code/opencode/codex 等各类 agent |

完整工作流负责组合能力；Issue、PR、Pipeline 和领域 Skills 仍可被独立触发。

## 4. 执行模式

默认模式为 `guided`：在 Issue 最终提交、需求结论、设计方案、远端写操作、source branch 首次 push、PR 创建和 Ready
等关键阶段确认。**CI 修复循环为默认自动例外**：PR 创建后默认立即触发 CI 并持续监控，失败后自动诊断、修复、
commit、push、重新触发，循环直到全绿或 blocked，不逐次确认 push；CI 全绿后报告本轮新增修复 commit 数；只有
blocked（证据充分的责任/基础设施问题、无法自动定位根因）才停下交用户。CI 自动修复不得绕过质量门禁或以强制
测试通过代替根因修复。

用户可以明确授权 `autonomous`。授权必须绑定当前仓库、Issue、分支和目标，并记录在 Issue/PR 评论。

以下硬停止点不因 autonomous 消失：

- 范围或公共行为发生实质变化；
- 凭证、隐私或安全漏洞；
- 绕过测试或质量门禁；
- 关闭 Issue、强制推送、强制测试通过、审批或合并；
- 规范、代码和远端事实冲突；
- 破坏性操作或需要新权限。

详见 [ai-execution-policy.md](spec/foundations/ai-execution-policy.md)。

## 5. GitCode CLI 唯一远端入口

所有 GitCode Issue、PR、评论、标签、评审和流水线状态操作必须通过 `gitcode` CLI。禁止 Skill 或脚本直接访问
GitCode REST API。

启动远端工作流前：

```bash
gitcode version
gitcode auth status
gitcode schema "<required command>"
python scripts/ai/resolve_repository_context.py --json
```

规则：

- Fork 开发时从 canonical 基线建分支并 push 到 source repository，再向 canonical repository 创建 PR；
- 有主仓权限时可以在主仓分支开发，但 PR target 仍为 canonical `master`；
- Fork 内部临时 Issue/PR 必须显式选择 Fork 作为 operation target，不能改变 canonical 身份；
- canonical 只定义正式目标，不构成远端写授权；远端写必须显示目标仓并确认；
- 读取类命令优先 `--json`；
- 多行正文使用 `--body-file` 或 `--comment-file`；
- 支持 `--dry-run` 的创建操作先预览；
- AI 禁止执行显示 Token 的命令、读取认证配置或要求用户在对话中粘贴 Token；
- typed command 未覆盖的能力只能通过 `gitcode api` 兜底；
- openLiBing 日志必须使用 `gitcode-pipeline-analyzer`，不得假设已启用 GitCode Actions。

详见 [gitcode-cli-contract.md](spec/foundations/gitcode-cli-contract.md)。

## 6. 可审计与可恢复

端到端和远端写工作流必须创建 `run-id`，在 Issue/PR 评论中记录阶段、证据、用户决策、产物和下一步。

恢复前读取现有评论，避免重复创建 Issue、PR 或评论。作者自检与独立评审必须使用不同主体标识。

详见 [audit-record-contract.md](spec/foundations/audit-record-contract.md)。

## 7. Skills 体系

### 7.1 工作流 Skills

| Skill | 作用 |
|---|---|
| `msmodeling-issue-draft` | 模糊需求澄清、代码分析、Issue 草稿、确认和 CLI 创建 |
| `msmodeling-my-issues-review` | 查询并评审当前用户负责的开放 Issue |
| `msmodeling-issue-delivery` | Issue 到 ready-for-review 的完整开发闭环 |
| `sig-review` | 评论 /merge 启动合入、深度检视、inline comment、风险评级和合入建议 |
| `msmodeling-ci-recovery` | openLiBing 流水线监控与修复循环 |
| `msmodeling-review-feedback` | 检视意见分析、修复、回复和解决 |

### 7.2 GitCode CLI Skills

已引入 Issue triage、PR create、pipeline、pre-commit 和 security Skills；Issue 创建、Issue 评审、PR 检视和反馈处理由 `msmodeling-*` / `sig-review` 工作流 Skill 承担。来源和适配记录在
`.agents/gitcode-skills.lock.json`。

### 7.3 业务领域 Skills

| Skill | 触发场景 |
|---|---|
| `device_config` | 新增或转换设备画像 |
| `op-mapping` | 生成或更新 `op_mapping.yaml` |
| `microbench` | 从 profiling 数据生成 NPU 重放脚本 |
| `profiling-database-axis-density` | 新增或调整 profiling database 采集轴和密度 |
| `msmodeling-env-installer` | 安装和验证开发环境 |
| `gitcode-cli-installer` | 安装和认证 gitcode CLI |
| `model-adaptation` | 接入新模型和生成 ModelProfile |
| `text-generate-executor` | 执行单点模型仿真验证 |
| `throughput-optimizer-executor` | 搜索并行策略和吞吐配置 |
| `throughput-optimizer-explainer` | 解释结果与瓶颈 |
| `optix-deploy` | 部署 OptiX |
| `optix-config` | 修改 OptiX 配置 |
| `optix-param-recommend` | 推荐 OptiX 参数范围 |

Skill 开发要求：

- 入口为 `SKILL.md`；
- frontmatter 包含 `name`、`description`、`metadata.version` 和 `metadata.source`；
- 使用仓库相对路径；
- 明确适用场景、流程、安全规则和完成标准；
- 示例命令可运行；
- 可选多代理能力不能成为正确性前提。

## 8. 代码架构约束

### DeviceProfile

`tensor_cast/device.py` 中 `DeviceProfile.__post_init__` 会向 `all_device_profiles` 注册：

- `name` 必须唯一；
- 写入前检查同名 profile；
- 用户自定义 profile 优先放入 `tensor_cast/device_profiles/`。

### CommGrid

- `grid.ndim == len(topologies)`；
- 每个 grid 维度至少为 2；
- `topologies` 的 key 是从 0 开始的 `start_dim`。

### 上游模型适配

- 通过 `tensor_cast/transformers/builtin_model/` wrapper 或 patch 层适配；
- 避免直接修改上游依赖；
- 优先 composition，谨慎使用 monkey patch。

### Performance Model

- `EmpiricalPerformanceModel` 基于 profiling；
- `AnalyticPerformanceModel` 基于算子复杂度；
- `op_mapping.yaml` 连接 TensorCast op 与 NPU kernel；
- hot path 禁止无必要的 `tensor.item()`，避免 CPU-NPU 同步。

## 9. 本地开发与验证

环境使用 Python 3.10+ 和 uv：

```bash
uv sync --group ci --group lint
```

按改动范围执行：

```bash
uv run pytest <affected-tests>
uv run pre-commit run --all-files
python build.py test
python build.py
```

`pre-commit` 的 `gitleaks-offline-scan` hook 用 `language: system` + `entry: ./gitleaks`，pre-commit 不自动安装；首次运行前用 `python scripts/ai/install_gitleaks.py` 准备本地二进制（幂等、平台感知、自动维护 `.gitignore`）。

AI Native 资产额外执行：

```bash
python scripts/ai/install_gitleaks.py --json
python scripts/ai/validate_skills.py
python scripts/ai/validate_remote_boundary.py
python scripts/ai/resolve_repository_context.py --json
python scripts/ai/check_gitcode_cli.py --json
```

测试或构建未运行时必须说明原因和残余风险，不得标记为通过。

## 10. Commit 与 PR

- Conventional Commits；
- 所有 commits 使用 sign-off；
- PR 聚焦单一目标；
- PR 关联 Issue；
- PR body 包含背景、修改、验证、风险、AI 参与说明和未覆盖项；
- 创建、编辑、评论和评审 PR 使用 GitCode CLI；
- CI 全绿和作者自检完成后才能进入 ready-for-review；
- 作者不得把自检当独立审批。

## 11. 目录

```text
MindStudio-Modeling/
├── .agents/
│   ├── gitcode-skills.lock.json
│   ├── repository-contract.json
│   └── skills/
├── .loop/memory/lessons.md
├── spec/
├── docs/
│   ├── RFC/
│   ├── design/
│   └── ai-native/
├── scripts/ai/
├── tensor_cast/
├── serving_cast/
├── optix/
├── tests/
├── AGENTS.md
├── CLAUDE.md
└── CONTRIBUTING.md
```
