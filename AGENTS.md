# msmodeling Agent & Contributor Guide

本项目使用 Claude Code 进行 AI辅助开发。本文档定义了 human contributors 和 AI agents 在 msmodeling 项目中的工作规范。

> **AI agents 必读**：首次在本项目中执行任务前，请完整阅读本文档。它解释了项目结构、skills 体系、代码规范和提交约定。

---

## Table of Contents

- [msmodeling Agent & Contributor Guide](#msmodeling-agent--contributor-guide)
  - [项目概览](#项目概览)
  - [Skills 体系](#skills-体系)
  - [环境与工具链](#环境与工具链)
    - [pre-commit](#pre-commit)
    - [Python 规范](#python-规范)
  - [Commit 规范](#commit-规范)
  - [测试规范](#测试规范)
  - [Skill 开发规范](#skill-开发规范)
  - [代码架构注意事项](#代码架构注意事项)
  - [项目结构](#项目结构)
  - [快速开始](#快速开始)
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
| `op_mapping` | `/op_mapping` 或"生成 op_mapping.yaml" | 将 TensorCast 仿真算子映射到 NPU profiling 内核类型 |
| `microbench` | `/microbench` 或"生成 xxx_run.py" | 从 profiling CSV 生成可在 NPU 上重放的 run script |

### 何时使用哪个 Skill

- 用户想添加新硬件 → `device_config`
- 用户想做 TC op → NPU kernel 的映射 → `op_mapping`
- 用户已有 profiling CSV，想生成可重放的 microbench → `microbench`

---

## 环境与工具链

### pre-commit

所有代码提交前必须通过 pre-commit 检查：

```bash
pip install pre-commit
pre-commit install  # 只需运行一次
```

提交时自动触发（若需手动运行）：

```bash
pre-commit run --all-files
```

检查项：trailing-whitespace、end-of-file-fixer、yaml/json 格式、ruff（lint + format）、codespell、pylint、bandit、typos。

> **注意**：pre-commit 的 `--fix` 标志会自动修复部分问题（如 ruff format），修复后请重新 `git add` 并提交。

### Python 规范

| 规范 | 要求 |
|------|------|
| Python 版本 | ≥ 3.10 |
| 行长度 | ≤ 120 字符 |
| 命名 | Classes: `PascalCase`；Functions/Methods: `snake_case`；Constants: `ALL_UPPER_CASE` |
| 导入 | 统一在文件顶部，例外：循环导入（inline import）、懒加载、TYPE_CHECKING 包裹 |
| Magic Numbers | 禁止，必须用命名常量 |
| Docstrings | 多行 docstring 的闭合 `"""` 必须独占一行（`D209`） |
| 打开文件 | 优先使用 context manager（`with open(...)` 而非 `try/finally`） |

配置位于 `pre-commit/pyproject.toml`，主配置通过 `ruff` 和 `pylint` 实现。

---

## Commit 规范

遵循 **Conventional Commits** 格式：

```text
<type>: <简短描述>

<详细说明（可选）>

Signed-off-by: Your Name <your.email@example.com>
```

**有效 type**：

- `feat`：新功能
- `fix`：Bug 修复
- `perf`：性能相关改动
- `refactor`：重构（无功能变化）
- `test`：测试相关
- `docs`：文档相关
- `chore`：杂项（依赖更新、CI 配置等）

**推荐格式示例**：

```text
feat(device_config): 添加 ATLAS_800_A3_560T_128G_DIE profile

- 新增单 die profile，算力 560T，显存 64GiB
- 复用现有 A3 die 互联拓扑

feat(skills): 新增 op_mapping SKILL

- 引入六阶段并行子 Agent 工作流
- 支持 TC op → NPU kernel 的自动化映射

fix(tensor_cast): 修正 DeepSeek V3.1 MoE 专家路由逻辑

- 修复 expert_indices 计算错误导致的精度问题
- 添加回归测试
```

**强制要求**：

- 所有 commits 必须 sign-off（`git commit -s`）
- PR 从 fork 仓库提交，而非直接推送到 main

---

## 测试规范

| 测试类型 | 位置 | 执行命令 |
|----------|------|----------|
| Unit Tests (UT) | `tests/test_tensor_cast/`、`tests/test_skill/` 等 | `bash ./tests/run_ut.sh tensor_cast` |
| System Tests (ST) | `tests/st/` | 通过 `pytest` 或项目 ST 框架运行 |

**要求**：

- 新功能必须附带对应测试
- Bug 修复必须包含回归测试
- UT 覆盖率新增代码应 ≥ 80%
- NPU 专用测试标记为 `@pytest.mark.npu`（默认跳过）

**跳过标记测试**：

```bash
pytest -m "not npu"  # 默认行为
```

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
│   ├── op_mapping/
│   └── microbench/
├── tensor_cast/             # 核心仿真框架
│   ├── device.py             # DeviceProfile 定义
│   ├── device_profiles/     # 用户自定义 profiles
│   ├── ops/                 # TC 虚拟算子
│   ├── transformers/        # 模型适配层
│   ├── performance_model/   # 性能模型
│   └── compilation/         # 编译相关
├── serving_cast/            # 服务仿真
├── tests/                   # 测试（UT + ST + skill_eval）
│   ├── perf_database/       # Profiling 数据库
│   └── skill_eval/         # Skill 评测框架
├── docs/                    # 文档
│   ├── RFC/                 # 设计提案
│   └── perf_database/      # 专项文档
├── cli/                     # CLI 入口
├── pre-commit/              # pre-commit 配置
└── tools/                   # 辅助工具
```

---

## 快速开始

### 首次设置

```bash
# 1. 克隆并设置环境
git clone https://gitcode.com/Ascend/msmodeling.git -b develop
cd msmodeling
pip install uv
uv venv --python 3.10 myenv
source myenv/bin/activate
uv pip install -r requirements.txt

# 2. 安装 pre-commit
pip install pre-commit
pre-commit install

# 3. 验证环境
python -c "import tensor_cast; print(tensor_cast.__version__)"
```

### 提交代码

```bash
# 1. 创建功能分支
git checkout -b feat/your-feature-name

# 2. 开发 + 测试
python -m pytest tests/test_tensor_cast/test_xxx.py -v
bash ./tests/run_ut.sh tensor_cast

# 3. 运行 pre-commit
pre-commit run --all-files

# 4. 提交（必须 sign-off）
git add .
git commit -s -m "feat(module): add feature description"

# 5. 推送到 fork 并创建 PR
git remote add myfork https://github.com/YOUR_USERNAME/msmodeling.git
git push -u myfork your-branch-name
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

- [ ] SKILL.md frontmatter 完整（name、description、version、source）
- [ ] Skill 文件名合法（snake_case，无连字符）
- [ ] 验证 skill 可正确加载和执行

### Commit & PR

- [ ] Commit message 符合 Conventional Commits 格式
- [ ] **所有 commits 已 sign-off**（`git commit -s`）
- [ ] PR 从 fork 创建，描述完整
