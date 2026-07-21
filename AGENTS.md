# msmodeling Agent Guide

本项目使用 Claude Code 进行 AI 辅助开发。本文档定义了 AI agents 在 msmodeling 项目中的工作规范。

> **AI agents 必读**：首次在本项目中执行任务前，请完整阅读本文档。人类贡献者规范请参见 [CONTRIBUTING.md](CONTRIBUTING.md)，本文档仅包含 Agent 专有内容。

---

## Table of Contents

- [项目概览](#项目概览)
- [贡献规范（引用）](#贡献规范引用)
- [Skills 体系](#skills-体系)
- [Skill 开发规范](#skill-开发规范)
- [代码架构注意事项](#代码架构注意事项)
- [项目结构](#项目结构)
- [Review Checklist](#review-checklist)

---

## 项目概览

msmodeling（MindStudio Modeling）是一个全系统性能仿真与分析框架，包含两个核心组件：

| 组件 | 定位 | 关键模块 |
|------|------|----------|
| **TensorCast** | PyTorch 程序性能仿真器，无需物理硬件即可模拟模型在目标硬件上的执行 | `tensor_cast/` |
| **ServingCast** | 系统级推理服务仿真与吞吐优化 | `serving_cast/` |

**核心价值**：在无需物理设备的情况下预测模型性能、定位瓶颈、优化配置。

---

## Skills 体系

项目使用 Claude Code skills 来规范特定任务的执行方式。所有 skills 位于 [`.agents/skills/`](.agents/skills/) 目录，随代码一起版本管理。

> **提示**：如需在 Claude Code 中启用这些 skills，请将 `.agents/skills` 目录完整复制到 `~/.claude/skills`。

| Skill | 触发词 | 用途 |
|-------|--------|------|
| `device_config` | `/device_config` 或"我要导入新的设备拓扑" | 通过自然语言将硬件规格转换为 TensorCast `DeviceProfile` |
| `op-mapping` | `/op_mapping` 或"生成 op_mapping.yaml" | 将 TensorCast 仿真算子映射到 NPU profiling 内核类型 |
| `microbench` | `/microbench` 或"生成 xxx_run.py" | 从 profiling CSV 生成可在 NPU 上重放的 run script |
| `msmodeling-env-installer` | "安装 msmodeling 环境"、"创建 myenv" | 安装并验证当前仓库开发环境、依赖和必要环境变量 |
| `model-adaptation` | "接入新模型"、"生成 ModelProfile"、"处理 doctor report" | 从仿真命令和 raw profiling 出发，完成 TensorCast 新模型适配流程 |
| `text-generate-executor` | "跑 text_generate"、"验证 best row"、"导出 trace" | 生成并执行 `python -m cli.inference.text_generate` 单点验证命令 |
| `throughput-optimizer-executor` | "搜索最佳 TP/EP"、"硬件对比"、"PD 配比优化" | 生成并执行 `python -m cli.inference.throughput_optimizer` 吞吐规划命令 |
| `throughput-optimizer-explainer` | "结果是否合理"、"为什么硬件不同"、"Cube/Vec/Comm/Mem 瓶颈" | 解释 optimizer 结果，并将 best row 映射到 `text_generate` 验证命令 |
| `optix-deploy` | "部署 optix"、"安装服务化自动寻优工具" | 安装并验证 msmodeling optix 服务化自动寻优工具 |
| `optix-config` | "配置 config.toml"、"设置 MindIE/vLLM 寻优字段" | 自动修改 optix `config.toml` 的寻优参数、target 和 benchmark 配置 |
| `optix-param-recommend` | "推荐 optix 参数"、"生成寻优范围" | 根据硬件、模型、负载和目标推荐 MindIE/vLLM 寻优参数与配置片段 |
| `sig-review` | "请求检视"、"检视PR {number}"、"review PR {number}"、"分析PR {number}的检视意见" | GitCode PR 分配检视（SIG 路由 + 指派 chair）、代码检视、检视意见分析，支持 cursor/claude code/opencode/codex 等各类 agent |

### 何时使用哪个 Skill

- 用户想添加新硬件 → `device_config`
- 用户想做 TC op → NPU kernel 的映射 → `op-mapping`
- 用户已有 profiling CSV，想生成可重放的 microbench → `microbench`
- 用户想安装或验证当前仓库开发环境 → `msmodeling-env-installer`
- 用户想接入新的 HuggingFace-style 模型 → `model-adaptation`
- 用户想跑固定 `text_generate` 场景或复验 optimizer best row → `text-generate-executor`
- 用户想搜索部署策略、对比硬件或规划 PD 聚合/分离/配比 → `throughput-optimizer-executor`
- 用户想解释 optimizer 结果合理性、硬件差异、Cube/Vec/Comm/Mem 或 op-bound 归因 → `throughput-optimizer-explainer`
- 用户想部署 optix 服务化自动寻优工具 → `optix-deploy`
- 用户想修改 optix `config.toml` → `optix-config`
- 用户首次使用 optix 且需要推荐寻优参数和搜索范围 → `optix-param-recommend`
- 用户推送 PR 后想请求检视（自动分配 SIG chair） → `sig-review`（assign）
- 用户收到 PR 检视通知，要用 agent 检视 PR → `sig-review`（review）
- 用户收到检视意见，想分析哪些该改、怎么改 → `sig-review`（analyze）

---

## 贡献规范（引用）

以下规范的权威来源为 [CONTRIBUTING.md](CONTRIBUTING.md)，Agent 按需读取对应章节即可，无需全量加载：

| 规范 | 链接 | 要点速查 |
|------|------|----------|
| 环境搭建 | [CONTRIBUTING.md#3-搭建开发环境](CONTRIBUTING.md#3-搭建开发环境) | Python ≥ 3.10，使用 uv 管理 |
| pre-commit | [CONTRIBUTING.md#5-本地测试与检查](CONTRIBUTING.md#5-本地测试与检查) | `pre-commit run --all-files` |
| 代码规范 | [CONTRIBUTING.md#代码规范](CONTRIBUTING.md#代码规范) | 以 pre-commit 检查为准 |
| Commit 规范 | [CONTRIBUTING.md#commit-规范](CONTRIBUTING.md#commit-规范) | Conventional Commits，必须 sign-off |
| 测试要求 | [CONTRIBUTING.md#测试要求](CONTRIBUTING.md#测试要求) | 新功能必须附带测试，覆盖率 ≥ 80% |
| PR 规范 | [CONTRIBUTING.md#pr-规范](CONTRIBUTING.md#pr-规范) | 功能单一，避免超大 PR |

---

## Skill 开发规范

新增或修改 skill 时，遵循以下规范：

### 文件命名

- Skill 入口文件：`SKILL.md`（大写）
- 文件名必须是小写、snake_case，不含连字符

### SKILL.md 结构

每个 skill 必须包含以下 frontmatter 和章节：

```markdown
---
name: <skill-name-in-kebab-case>
description: <一句话描述触发场景>
metadata:
  version: <semver>
  source: local-session-analysis
---

# <Skill 中文名>

## 适用场景

## 默认策略

## 工作流程

## <具体执行内容>

## 安全规则

## 完成标准
```

### 目录结构

```text
.agents/skills/<skill_name>/
├── SKILL.md              # 必须：skill 定义
├── ref/                  # 可选：参考文档
│   ├── xxx.md
│   └── yyy.md
├── scripts/              # 可选：辅助脚本
│   └── xxx.py
└── <other files>         # 可选：模板、配置等
```

### 路径规范

- Skill 内部引用使用相对路径（`./` 或 `../`）
- Repo 级别引用使用相对于项目根的路径（如 `tensor_cast/device.py`）
- 不要使用硬编码的绝对路径

### 验证要求

Skill 实现后必须验证：

1. 可以通过 Claude Code skill 系统正确加载
2. 生成的输出可以被导入/执行（不产生 import error）
3. 文档中的示例命令可运行

---

## 代码架构注意事项

### 1. `DeviceProfile` 注册机制

`tensor_cast/device.py` 中的 `DeviceProfile.__post_init__` 会自动将 profile 注册到类变量 `all_device_profiles`。因此：

- `name` 必须唯一，重复会抛出 `ValueError`
- 写入前先检查是否已存在同名 profile
- 不确定时优先写入 `tensor_cast/device_profiles/*.py`（用户自定义 profile）而非 `device.py`

### 2. `CommGrid` 约束

- `grid.ndim == len(topologies)`
- 每个 grid 维度至少为 2
- `topologies` 的 key 是 `start_dim`（从 0 开始），不是任意层级编号

### 3. 避免直接修改 vLLM/上游依赖

TensorCast 引用上游模型代码时：

- 通过 `tensor_cast/transformers/builtin_model/` 中的 wrapper 或 patch 层适配
- 不建议直接在 `transformers` 库注入改动
- 模型特定行为优先通过 composition 而非 monkey-patching 实现

### 4. Performance Model 架构

- `EmpiricalPerformanceModel`：基于真实 profiling 数据的性能估算
- `AnalyticPerformanceModel`：基于算子复杂度分析的估算
- Profiling 数据通过 `op_mapping.yaml` 与 TC 算子建立映射关系
- 不要在 hot path 中使用 `tensor.item()`（会导致 CPU-NPU 同步开销）

---

## 项目结构

```text
msmodeling/
├── .agents/skills/          # Claude Code skills（随代码版本管理）
│   ├── README.md             # Skills 概览索引
│   ├── device_config/
│   ├── op-mapping/
│   ├── microbench/
│   ├── msmodeling-env-installer/
│   ├── model-adaptation/
│   ├── text-generate-executor/
│   ├── throughput-optimizer-executor/
│   ├── throughput-optimizer-explainer/
│   ├── optix-deploy/
│   ├── optix-config/
│   ├── optix-param-recommend/
│   └── sig-review/           # GitCode PR 检视（SIG 责任田路由 + AI 自动检视）
├── .loop/                   # AI 自主开发框架（经验池 + prompt 模板）
│   └── memory/lessons.md     # 跨会话经验教训库（团队共享，随代码版本管理）
├── spec/                    # 项目正式规范（最高权威）
│   ├── governance/           #   治理规则（source-of-truth-matrix.md 等）
│   └── foundations/          #   基础规范（后续补齐）
├── tensor_cast/             # 核心仿真框架
│   ├── device.py             # DeviceProfile 定义
│   ├── device_profiles/     # 用户自定义 profiles
│   ├── ops/                 # TC 虚拟算子
│   ├── transformers/        # 模型适配层
│   ├── performance_model/   # 性能模型
│   └── compilation/         # 编译相关
├── serving_cast/            # 服务仿真
├── tests/                   # 测试（UT + ST + benchmark）
│   └── perf_database/       # Profiling 数据库
├── docs/                    # 文档
│   ├── RFC/                 # 设计提案
│   ├── design/              # 实现设计
│   └── perf_database/      # 专项文档
├── cli/                     # CLI 入口
├── optix/                   # 服务化自动寻优工具
├── web_ui/                  # Web UI
├── pre-commit/              # pre-commit 配置
└── tools/                   # 辅助工具
```

---

## Review Checklist

提交 PR 前，确认以下条目：

### 代码质量

- [ ] 代码风格通过 pre-commit 检查（ruff、pylint、bandit 均无 ERROR）
- [ ] 无 magic numbers（已替换为命名常量）
- [ ] 命名符合规范（PascalCase/snake_case/ALL_UPPER_CASE）
- [ ] 导入在文件顶部，无循环依赖

### 测试

- [ ] 新功能附带测试，覆盖率 ≥ 80%
- [ ] Bug 修复附带回归测试
- [ ] UT 通过：`bash ./tests/run_ut.sh tensor_cast`

### 文档

- [ ] 新增/修改的功能反映在相关文档中
- [ ] 如果是 Skill 相关改动，更新 `.agents/skills/README.md` 对应章节

### Skill 相关（若涉及）

- [ ] SKILL.md frontmatter 完整（name、description、metadata.version、metadata.source）
- [ ] Skill 文件名合法（snake_case，无连字符）
- [ ] 验证 skill 可正确加载和执行

### Commit & PR

- [ ] Commit message 符合 Conventional Commits 格式
- [ ] **所有 commits 已 sign-off**（`git commit -s`）
- [ ] PR 从 fork 创建，描述完整
