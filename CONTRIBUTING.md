# msModeling 贡献指南

感谢你对 msModeling（MindStudio Modeling）的关注！我们欢迎每一位开发者参与本项目的开发与建设。无论你是修复一个 bug、完善文档，还是提出一个全新的功能特性，你的贡献都将帮助 msModeling 成为更好的全系统性能仿真与分析框架。

本指南旨在帮助你了解如何高效地参与贡献，请在提交代码前仔细阅读。

---

## 目录

- [行为准则](#行为准则)
- [开源协议与合规](#开源协议与合规)
- [贡献方式](#贡献方式)
- [开发者如何贡献代码](#开发者如何贡献代码)
  - [1. 找到要参与的 Issue 或提出想法](#1-找到要参与的-issue-或提出想法)
  - [2. 环境准备（包括克隆仓库）](#2-环境准备包括克隆仓库)
  - [3. 配置开发环境](#3-配置开发环境)
  - [4. 创建功能分支并开发](#4-创建功能分支并开发)
  - [5. 本地测试与检查](#5-本地测试与检查)
  - [6. 提交代码](#6-提交代码)
  - [7. 推送并创建 Pull Request](#7-推送并创建-pull-request)
  - [8. 门禁流水线与 CI](#8-门禁流水线与-ci)
  - [9. 代码评审](#9-代码评审)
- [代码目录规范](#代码目录规范)
- [Commit 规范](#commit-规范)
- [PR 规范](#pr-规范)
- [代码规范](#代码规范)
- [测试要求](#测试要求)
- [设计文档与 RFC](#设计文档与-rfc)
- [质量愿景与工程实践](#质量愿景与工程实践)
- [AI 辅助编程](#ai-辅助编程)
- [社区角色与晋升机制](#社区角色与晋升机制)
  - [角色定义](#角色定义)
  - [晋升路径](#晋升路径)
  - [活跃度要求与荣誉退休](#活跃度要求与荣誉退休)
- [安全与漏洞响应](#安全与漏洞响应)
- [用户接口稳定性](#用户接口稳定性)
- [生态共建](#生态共建)
- [联系方式](#联系方式)

---

## 行为准则

我们致力于营造一个开放、友善、包容的社区环境。请在参与本项目时保持尊重与专业，遵循基本的社区礼仪。对于不当行为，维护者有权采取适当措施。

---

## 开源协议与合规

msModeling 采用 [木兰宽松许可证 第2版（Mulan PSL v2）](http://license.coscl.org.cn/MulanPSL2) 开源。

- 提交的**所有代码内容必须尊重业界开源规范及 License 要求**，确保知识产权合规。
- **贡献者许可协议 (CLA)**：参与项目贡献前，个人贡献应签署个人 CLA，公司贡献应签署企业 CLA。
- 禁止引入与项目 License 不兼容的第三方依赖。
- 引入第三方库时需确认其 License 兼容性，并在 PR 中说明。
- 新增源文件应在文件头部包含版权声明（参见 [LICENSE](docs/zh/legal/LICENSE) 中的模板）。

---

## 贡献方式

你可以通过以下多种方式参与贡献：

| 贡献方式 | 说明 |
|----------|------|
| 🐛 **提交 Bug** | 在 [Issues](https://gitcode.com/Ascend/msmodeling/issues) 中报告你发现的问题 |
| 💡 **提出建议** | 通过 Issue 或 Discussion 提出功能建议 |
| 📖 **完善文档** | 改进 README、使用说明、API 文档等 |
| 🧪 **补充测试** | 为现有模块补充单元测试或系统测试 |
| 🔧 **修复 Bug** | 认领 Issue 并提交修复 PR |
| 🚀 **开发新功能** | 通过 RFC 提案流程贡献新特性 |
| 🔌 **开发插件** | 在 `contrib/` 目录下贡献第三方扩展 |
| 🤝 **代码检视** | 参与 PR 评审，帮助提升代码质量 |
| 💬 **社区答疑** | 回复用户 Issue，帮助定位和解决问题 |

---

## 开发者如何贡献代码

### 1. 找到要参与的 Issue 或提出想法

- 浏览 [Issue 列表](https://gitcode.com/Ascend/msmodeling/issues)，寻找标有 `good first issue` 或 `help wanted` 的任务。
- 如果你有新的想法或发现了 bug，请先创建一个 Issue，描述你的方案或问题。等待维护者确认方向后再开始开发，避免返工。
- 对于较大的功能变更，建议先提交 [RFC（设计文档）](#设计文档与-rfc)。

### 2. 环境准备（包括克隆仓库）

按照《[msModeling 安装指南 — 源码安装](docs/zh/install_guide/msmodeling_install_guide.md#231-环境准备)》章节完成编译和测试环境的搭建。

> **说明：** 环境镜像的构建方法及配套软件版本由 MindStudio 统一镜像制作指南维护，本仓库不重复定义。

请先将仓库 Fork 到你的个人空间，然后在本地克隆你的 Fork 副本：

```bash
# Fork 仓库后，克隆你的 fork
git clone https://gitcode.com/YOUR_USERNAME/msmodeling.git -b master
cd msmodeling

# 添加上游仓库以便同步
git remote add upstream https://gitcode.com/Ascend/msmodeling.git
git fetch upstream
```

> **注意**：外部贡献者默认通过 Fork 提交 PR；具备权限的维护者也可在上游功能分支开发。任何角色都不得直接推送到 `master`。

### 3. 配置开发环境

本节按“**安装依赖 →（可选）切换镜像源 → 配置环境变量 → 安装辅助工具**”的顺序组织。可先按下表按自己的使用目的找到需要完成的步骤：

| 使用目的 | 需要完成的步骤 |
|----------|----------------|
| 仅体验工具（仿真、Throughput Optimizer、Web UI、OptiX 等） | 3.1（推荐）或 3.2 安装依赖 |
| 本地运行测试（pytest） | 安装依赖时补充 `ci` 依赖组（见 3.1 依赖组表格） |
| 参与代码贡献（本地 pre-commit 检查） | 安装依赖时补充 `lint` 依赖组 + 3.5 安装 pre-commit |
| 使用 AI Native 工作流 | 额外完成 3.6 安装 gitcode CLI |
| 依赖下载失败或速度慢 | 参考 3.3 切换 PyPI 镜像源 |
| 不在仓库根目录运行、无法访问 Hugging Face | 参考 3.4 配置环境变量 |

#### 3.1 方式一：uv（推荐）

如上编译容器已准备好 uv 和源设置，使用 uv 只需一条命令完成虚拟环境创建与依赖安装：

```bash
cd msmodeling
uv sync
# 可选：uv sync --group lint
```

`uv sync` 会自动创建 `.venv` 并以可编辑模式安装本项目，无需 `uv venv` 或 `pip install -e .`。

**按需安装可选依赖组**：`uv sync` 默认只安装运行与仿真所需依赖，按使用场景再决定是否安装可选依赖组：

| 依赖组 | 何时需要 | 安装命令 | 典型用途 |
| -------- | -------- | -------- | -------- |
| （默认） | 安装后运行仿真、Throughput Optimizer、Web UI、OptiX 等 | `uv sync` | 日常使用与体验 |
| `lint` | 参与代码贡献，需要本地 pre-commit 风格/提交检查时 | `uv sync --group lint` | `uv run pre-commit install`、`uv run pre-commit run --all-files` |
| `ci` | 本地运行 pytest、对齐 CI Gate / `scripts/run_*.sh` 测试环境时 | `uv sync --group ci` | `uv run pytest ...`、`./scripts/run_ci_gate.sh` |

仅体验工具时执行 `uv sync` 即可，无需安装 `lint` 或 `ci`。需要时再按上表补充：

```bash
# 开发贡献：安装 pre-commit 等 lint 依赖
uv sync --group lint

# 本地测试：安装 pytest 等 CI 依赖
uv sync --group ci
```

**使用环境**：完成后，可使用 `uv run ...` 执行命令；如需手动激活虚拟环境，可激活 `uv sync` 自动创建的 `.venv`。

> [!NOTE]
> 如果使用 `uv` 创建或管理虚拟环境，后续查看、升级、卸载也建议使用 `uv pip ...` 或 `uv run ...`。不要仅通过 `which pip` 判断当前环境，部分场景下 `pip` 可能指向非预期的 Python 环境。

#### 3.2 方式二：pip + requirements.txt（备选）

如果不使用 `uv`，也可以通过 Python 原生虚拟环境和 `requirements.txt` 安装依赖。CPU 环境建议先从 PyTorch CPU 源安装 `torch` 与 `torchvision`，再安装其余依赖。

```bash
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate

pip install "torch>=2.8,<=2.10" "torchvision>=0.23.0" --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -e .
```

如需使用 pip 运行测试或 CI 检查，请安装同时包含运行时和测试依赖的 `requirements-ci.txt`：

```bash
pip install -r requirements-ci.txt
pip install -e .
```

> [!NOTE]
> `pip install -e .` 会以源码可编辑模式安装 msModeling，并注册 `msmodeling` CLI。源码更新后无需重新复制文件，必要时重新执行安装命令即可。

如果依赖下载失败或速度较慢，可切换 PyPI 镜像源后重试，详见[切换 PyPI 镜像源](#33-切换-pypi-镜像源)。

> [!WARNING]
> Windows 上 PyTorch 2.10 可能运行不正常。如遇问题，建议使用 PyTorch 2.8 或更早版本。

#### 3.3 切换 PyPI 镜像源

国内网络环境下，依赖下载失败或速度较慢时，可临时切换 PyPI 镜像源。推荐优先使用阿里云镜像；若已有公司内网源或其他镜像配置，可继续沿用，不必强制更换。

**uv（推荐用于 `uv sync`）**

```bash
# 当前终端会话生效（推荐）
export UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple
uv sync

# 或仅对单次命令生效
UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple uv sync

# uv pip 安装/升级时临时指定镜像
uv pip install --upgrade -e . -i https://mirrors.aliyun.com/pypi/simple
```

**pip**

```bash
# 临时使用阿里云源（推荐）
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple

# 临时使用清华源
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 临时使用华为云源
pip install -r requirements.txt -i https://repo.huaweicloud.com/repository/pypi/simple
```

如某个镜像源同步不及时导致版本找不到，请更换其他镜像源或临时回退到官方源 `https://pypi.org/simple` 后重试。

#### 3.4 配置环境变量

msModeling 常用环境变量如下：

| 环境变量 | 可选/必选 | 说明 |
| -------- | -------- | ---- |
| PYTHONPATH | 可选 | 不在 msModeling 仓库根目录下运行时，可将该变量配置为仓库根目录，避免出现 `No module named cli`、`No module named tensor_cast` 等模块导入错误。 |
| HF_ENDPOINT | 可选 | 无法直接访问 Hugging Face 时，可配置 Hugging Face 镜像地址，例如 `https://hf-mirror.com`。 |
| OPTIX_DEPLOY_PATH | 可选 | 使用 OptiX 且系统 `PATH` 特殊时，可配置部署栈命令所在路径。通常无需配置。 |

如果不在 msModeling 根目录下运行，需要设置 `PYTHONPATH`：

```bash
# Linux / macOS
export PYTHONPATH=/path/to/msmodeling:$PYTHONPATH

# Windows PowerShell
$env:PYTHONPATH = "C:\path\to\msmodeling;$env:PYTHONPATH"
```

工具运行时可能需要从 Hugging Face 读取模型配置文件。如果无法直接访问，可以设置镜像：

```bash
# Linux / macOS
export HF_ENDPOINT="https://hf-mirror.com"

# Windows PowerShell
$env:HF_ENDPOINT = "https://hf-mirror.com"
```

在受限网络中，即使设置 `HF_ENDPOINT`，仍可能因代理策略、DNS、TLS 证书、镜像站不可达、模型仓库需要鉴权，或依赖库未使用该环境变量而下载失败。此时建议使用经审核的本地模型路径。

#### 3.5 安装 pre-commit

参与代码贡献时，需要安装一次 pre-commit 钩子（对应的 `lint` 依赖组见 3.1）：

```bash
uv sync --group lint
uv run pre-commit install    # 只需运行一次
```

#### 3.6 安装 gitcode CLI（AI Native 工作流远端操作入口，一次性）

使用 AI Native 工作流（远端操作 Issue、PR、流水线等）时，需要安装并认证 gitcode CLI：

```bash
npm config set @gitcode-cli:registry https://registry.npmjs.org
npm install -g @gitcode-cli/cli
gitcode auth login    # 交互式认证，令牌在 GitCode → 设置 → 私人令牌 → 生成（需 repo 读写权限）
gitcode version && gitcode auth status    # 验证
```

> Windows PowerShell 中 `gc` 是 `Get-Content` 的内置别名，请使用 `gitcode` 全称。

更多环境细节请参考 [README.md](README.md) 中的安装与使用指南部分。

### 4. 创建功能分支并开发

```bash
# 确保基于最新的 master 分支
git fetch upstream
git checkout -b feat/your-feature-name upstream/master
```

### 5. 本地测试与检查

**代码上库前必须在本地通过以下检查：**

1. **pre-commit 检查** — 确保代码风格、拼写等符合要求：

   ```bash
   uv run pre-commit run --all-files
   ```

2. **单元测试（UT）与系统测试（ST）** — 确保相关测试通过：

   由于本地测试脚本已迁移下线，本地运行测试需先同步测试依赖，然后手动触发 `pytest`。首先同步测试依赖：

   ```bash
   uv sync --group ci
   ```

   然后手动运行 `pytest` 触发对应测试，例如：

   ```bash
   uv run pytest tests/regression/tensor_cast/  # 运行 TensorCast 单元测试
   uv run pytest tests/regression/serving_cast/  # 运行 ServingCast 单元测试
   uv run pytest tests/smoke/  # 运行集成与冒烟测试 (ST)
   ```

3. **仓库要求的其他检查手段** — 如 skill 验证（确认 skill 可正确加载和执行）、模型仿真精度验证等。

> 💡 **提示**：本地测试通过后再提交，可以有效减少 CI 反复修复的成本，提升效率。

### 6. 提交代码

所有 commit 必须遵循 [Commit 规范](#commit-规范)，且**必须 sign-off**：

```bash
git add .
git commit -s -m "feat(tensor_cast): add new operator support

- 新增 xxx 算子仿真
- 补充对应 UT 用例"
```

### 7. 推送并创建 Pull Request

```bash
git push origin feat/your-feature-name
gitcode pr create \
  -R Ascend/msmodeling \
  --fork YOUR_USERNAME/msmodeling \
  --base master \
  --head feat/your-feature-name \
  --title "feat(scope): summary" \
  --body-file pr-body.md \
  --draft
```

AI 或自动化创建 PR、添加评论、设置标签和提交评审时必须使用 `gitcode` CLI。人工可使用 GitCode 页面，
但 PR 仍应遵循 [PR 规范](#pr-规范)。创建前可运行 `gitcode schema "pr create"` 核对当前 CLI 参数。
有主仓分支写权限时可以省略 `--fork`，但仍不得直接在 `master` 开发。

创建 PR 后，对 AI agent 说“请求检视”，即可按 SIG 路由执行深度检视；经确认后，AI 使用 CLI 提交
准确的行内评论和总体结论（详见 `.agents/skills/sig-review/SKILL.md`）。

### 8. 门禁流水线与 CI

PR 提交后，openLiBing 会在 PR 评论区反馈门禁流水线。AI 必须通过 `gitcode pr comments` 读取远端状态，
并使用 `gitcode-pipeline-analyzer` 获取和分析日志。**如果门禁流水线有报错**，例如：

- **pre-commit 检查失败**：请查看 CI 日志，定位失败的具体检查项（如 ruff lint error、typo 等），在本地修复后重新推送。
- **UT 失败**：请查看 CI 日志中的测试报告，找到失败的用例及错误堆栈，在本地复现并修复。
- **其他门禁失败**：认真阅读日志输出，定位根因，解决问题后再次推送。

修复后在本地复验、推送新 commit，并重复监控下一轮流水线。每轮应在 PR 留下任务链接、失败签名、根因、
修复 commit 和验证结果；无法获得新证据或需要绕过门禁时必须停止。

> ⚠️ **门禁必须全部通过后 PR 才能进入评审流程。** 如有疑问，可在 PR 评论区求助。

### 9. 代码评审

- Reviewer 会对你的 PR 进行评审，可能提出修改建议。请积极回应评审意见并更新代码。
- **意见解决与状态设置**：代码合入前，必须将所有检视意见解决并设置为**已解决（Resolved）**状态。若开发者由于未加入本项目而无权设置状态，请联系仓库管理员添加权限。
- **标签收集与合入条件**：代码合入需收集一个 `/lgtm` 标签与一个 `/approve` 标签。开发者通过检视后，可主动联系 Reviewer 收集 `/lgtm`，联系 Approver 收集 `/approve`。
- 标签集齐且检视意见均已解决后，由具有合入权限的 Approver 或 Maintainer 合入代码。

---

## 代码目录规范

| 类型 | 位置 | 说明 |
|------|------|------|
| 核心功能代码 | `tensor_cast/`、`serving_cast/`、`cli/` 等 | 项目主体功能模块 |
| 第三方开发插件 / 非主体功能 | `contrib/` | 所有非主体功能代码及第三方开发的插件应合入此目录 |
| 测试用例 | `tests/` | UT、ST、Skill 评测等 |
| 文档 | `docs/` | 使用说明、RFC 设计文档、专项文档等 |
| 工具脚本 | `tools/` | 辅助工具和脚本 |
| AI Agent Skills | `.agents/skills/` | Claude Code skills，随代码版本管理 |

---

## Commit 规范

遵循 **Conventional Commits** 格式：

```text
<type>(<scope>): <简短描述>

<详细说明（可选）>

Signed-off-by: Your Name <your.email@example.com>
```

**有效 type**：

| Type | 用途 |
|------|------|
| `feat` | 新功能 |
| `fix` | Bug 修复 |
| `perf` | 性能相关改动 |
| `refactor` | 重构（无功能变化） |
| `test` | 测试相关 |
| `docs` | 文档相关 |
| `chore` | 杂项（依赖更新、CI 配置等） |

**示例**：

```text
feat(device_config): 添加 ATLAS_800_A3_560T_128G_DIE profile

- 新增单 die profile，算力 560T，显存 64GiB
- 复用现有 A3 die 互联拓扑

Signed-off-by: Zhang San <zhangsan@example.com>
```

**强制要求**：

- ✅ 所有 commit 必须 sign-off（`git commit -s`）
- ✅ Commit message 应清晰描述变更内容
- ✅ 代码可读性好，Commit 信息与代码变更对应

---

## PR 规范

提交 PR 时，请确保 PR 描述**规范、完整**（正常情况下，请直接根据默认模板填写即可，重点强调背景、修改方案、验证结果与影响范围等要素）。

**PR 关联要求**：

为便于版本规划、问题追踪和合入管理，PR 合入前必须满足以下二选一要求：

- **关联 Issue**：建议贡献者根据本次修改内容自行创建 Issue，并在 PR 中完成关联。请在评审通过后及时推进合入，避免 PR 长时间悬置。
- **关联里程碑**：如 PR 关联里程碑，则该 PR 需要合入对应的商发分支，请确认目标分支与里程碑版本一致。

如你暂无关联 Issue、里程碑或将 diff comment 标记为已解决的权限，请通过[项目协作权限申请链接](https://gitcode.com/invite/link/ff088415445e4722837f)申请加入项目协作权限，我们会尽快审批。

**PR 最佳实践**：

- **功能单一**：每个 PR 应聚焦于单一功能或修复，避免多个不相关的修改混入同一 PR
- **避免超大 PR**：如非必要，**尽量避免提交修改超过 2000 行的 PR**。超大 PR 难以评审，请拆分为多个功能独立的 PR
- **及时回应评审**：收到评审意见后请及时回复和修改

---

## 代码规范

本项目的代码规范以 **pre-commit 检查为准**。所有代码在提交前必须通过 pre-commit 的自动化检查，涵盖 `ruff`（lint + format）、`pylint`、`bandit`、`codespell`、`typos` 等工具。具体配置详见 `pre-commit/pyproject.toml` 和 `.pre-commit-config.yaml`。

此外，也可参考 [昇腾社区 Python 编码风格指南](https://gitcode.com/Ascend/community/blob/master/docs/contributor/Ascend-python-coding-style-guide.md) 作为补充阅读，帮助了解更多编码最佳实践。

---

## 测试要求

| 测试类型 | 位置 | 执行命令 |
|----------|------|----------|
| 单元测试（UT） | `tests/regression/tensor_cast/`、`tests/regression/serving_cast/` 等 | `uv run pytest <位置>` (先执行 `uv sync --group ci` 同步依赖) |
| 系统/冒烟测试（ST） | `tests/smoke/`、`tests/regression/` 等 | `uv run pytest tests/smoke/` (通过 pytest 触发) |

**硬性要求**：

- 新功能**必须附带对应测试**
- Bug 修复**必须包含回归测试**
- **UT 覆盖率**：新增代码行覆盖率 ≥ **80%**，分支覆盖率 ≥ **60%**
- **已支持的模型仿真精度不得下降**（新增代码不能影响已有模型的仿真结果）
- NPU 专用测试标记为 `@pytest.mark.npu`（在无 NPU 环境中默认跳过）
- 鼓励社区开发者积极补充测试用例，完善测试覆盖

---

## 设计文档与 RFC

**设计文档应先于代码开发**。对于涉及以下情况的变更，请先在社区提交 RFC（Request for Comments）设计文档，尽早请求评审：

- 新增核心功能或模块
- 修改现有公共 API
- 架构性重构
- 性能模型的重大变更

RFC 文档放置于 `docs/RFC/` 目录。提交 RFC 后，请在 Issue 或 Discussion 中通知相关 Reviewer 参与评审。

---

## 质量愿景与工程实践

msModeling 致力于打造**高质量、高可靠、可持续演进**的开源项目。以下是我们的质量导向与工程实践要求：

### 代码质量

- 代码应符合仓库规范，尽可能靠近最佳实践
- 对腐化代码应及时重构，保持代码健康
- 配套测试用例完备，设计文档完善

### 主分支稳定性

- 主分支出现问题时，**采用 Revert-First 策略** — 先 revert 恢复主分支稳定，再排查和修复问题
- **冒烟测试出现问题须在 24 小时内处理**

### 发版与分支管理

- 发包前或商发分支拉取后，**受限合入代码**（仅部分 Approver 有权限），确保版本稳定性
- 分支演进需遵循团队既定策略

### 门禁纪律

- **处理门禁屏蔽或任何质量要求绕过时需谨慎**，必须记录合理、充分的理由
- 未经授权不得跳过或绕过质量门禁

### 安全与漏洞

- 漏洞问题按社区漏洞响应要求处理（参见 [安全与漏洞响应](#安全与漏洞响应)）

---

## AI 辅助编程

本仓提供跨 AI 客户端的统一入口 [AGENTS.md](AGENTS.md)、强制规范 [spec/](spec/README.md) 和
[AI Native 工作流指南](docs/ai-native/workflow-guide.md)。开发者可以独立触发 Issue、开发、PR 检视、
CI 恢复等场景，也可以让工作流 Skill 编排完整交付。

- 默认使用 `guided` 模式，在关键阶段由开发者确认；`autonomous` 必须明确限定仓库、Issue、分支和目标。
- AI 对 GitCode Issue、PR、评论、评审、标签和状态的操作必须使用 `gitcode` CLI。
- 关键决策、验证证据、暂停点和恢复点必须记录到 Issue/PR，且不得包含 Token 或本地敏感信息。
- 开发者可随时暂停、修改计划、接管或恢复；审批、合并、强推、关闭 Issue 和门禁绕过不会被自动执行。
- 交互式流程不依赖 Superpowers；需求与设计使用仓库已有的 `spec/`、RFC 和 design 体系，不要求 OpenSpec。

- **与 AI Agent 协作产出的代码，必须由开发者本人进行人工审视**
- 开发者须**对代码质量负全责**，做好把关
- AI 生成的代码同样需要满足本贡献指南中的所有质量要求（代码风格、测试覆盖、文档等）
- 不能以"AI 生成的代码"为由降低代码标准

---

## 社区角色与晋升机制

### 角色定义

| 角色 | 职责 | 权限 |
|------|------|------|
| **Contributor** | 贡献代码、文档、测试等 | 提交 PR |
| **Reviewer** | 代码检视，提出改进建议 | `/lgtm` 标签 |
| **Approver** | 批准代码合入，看护模块质量 | `/approve` 标签，合入权限 |
| **Maintainer** | 仓库整体技术方向与质量把控 | 仓库管理权限 |

### 晋升路径

社区采用**提名 + 多数投票**的晋升机制：

- **Contributor → Reviewer**：通过高质量的代码检视积累贡献，被提名并经投票多数通过后成为 Reviewer
- **Reviewer → Approver**：对所负责模块看护质量较好，被提名并经投票多数通过后成为 Approver
- **Approver → Maintainer**：对仓库整体贡献较大，被提名并经投票多数通过后成为 Maintainer

> 📖 晋升流程详情请参考：[昇腾社区角色定义与晋升机制](https://gitcode.com/Ascend/community/blob/master/docs/role-definition-and-promotion-mechanism.md)

### 活跃度要求与荣誉退休

为保障社区的活跃运转，各角色有最低活跃度要求：

| 角色 | 最低活跃度 | 响应时效 |
|------|-----------|---------|
| **Reviewer** | 每季度至少充分检视 **3 个以上** PR | 责任田内的代码建议在 **48 小时内**完成检视或委托检视 |
| **Approver** | 每季度至少参与 **3 次以上**的代码合入或技术讨论 | 责任田内的代码建议在 **48 小时内**完成检视或委托检视 |

**荣誉退休机制**：

- 若长时间未在社区交互，将被视为**荣誉退休**
- 荣誉退休后，如有时间可随时回归，我们始终欢迎
- **鼓励主动告知退休**，以便团队及时安排其他同学接手看护工作

---

## 安全与漏洞响应

- 若发现安全漏洞，**请勿通过公开 Issue 报告**，请通过安全邮件或私信联系维护者
- 漏洞问题按社区漏洞响应要求处理，及时修复并发布安全更新
- 所有安全相关的修复应遵循最小变更原则，并包含回归测试

---

## 用户接口稳定性

- 对用户接口界面应尽可能保持稳定，确保**历史功能兼容**
- 需下线的功能须**提前一个季度公告**，给用户充分的迁移时间
- 接口变更须在 CHANGELOG 中明确记录

---

## 生态共建

我们倡导积极参与构建活跃、健康的 msModeling 社区生态：

- 🙌 **及时回复用户问题** — 帮助新用户快速上手
- 👍 **为社区贡献者点赞** — 对帮助 msModeling 生态繁荣的努力表示感谢
- 📢 **分享使用经验** — 在 Discussion、技术群、博客中分享实践心得
- 🔗 **参与周会** — MindStudio Modeling 周会每周三 10:00–12:00（UTC+8），议程：[Etherpad](https://etherpad.ascend.osinfra.cn/p/sig-msit-modeling)

我们相信，每一次互动和贡献都在让社区变得更好！

---

## 联系方式

- **Issue 反馈**：[https://gitcode.com/Ascend/msmodeling/issues](https://gitcode.com/Ascend/msmodeling/issues)
- **讨论区**：[https://gitcode.com/Ascend/msmodeling/discussions](https://gitcode.com/Ascend/msmodeling/discussions)
- **周会议程**：[https://etherpad.ascend.osinfra.cn/p/sig-msit-modeling](https://etherpad.ascend.osinfra.cn/p/sig-msit-modeling)

---

*感谢你的贡献，让 msModeling 变得更好！* 🎉
