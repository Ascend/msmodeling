# RFC: `msmodeling-env-installer` Skill 设计方案

## 元数据

| 项目 | 内容 |
|:---|:---|
| **状态** | Draft |
| **作者** | lutean |
| **创建日期** | 2026-06-01 |
| **更新日期** | 2026-06-03 |
| **相关链接** | 无 |

---

## 1. 概述

本 RFC 描述 `msmodeling-env-installer` Skill 的设计与落地方案。该 Skill 用于将用户关于安装 msmodeling 开发环境、创建虚拟环境、安装当前仓库 `requirements.txt`、配置 `PYTHONPATH` 或 `HF_ENDPOINT` 的自然语言请求，转换为可执行、可验证、可回溯的环境安装流程。

由于仓库内后续可能存在其他环境安装工具，当用户只输入“安装环境”“安装依赖”“配置环境”等泛化请求时，本 Skill 不能默认接管。Agent 必须先询问用户是否要安装 msmodeling 当前仓库的环境依赖，确认后再执行。

该 Skill 面向“新机器或新仓库 checkout 后需要快速完成开发环境初始化”的场景。它不修改 msmodeling 业务代码，也不改变仓库依赖定义，而是围绕 README 推荐路径和本地脚本封装一套稳定的安装、校验和故障处理流程。

## 2. 目标与非目标

目标：

- 支持用户通过对话触发 msmodeling 环境安装流程。
- 对“安装环境”等泛化请求先确认是否安装 msmodeling 当前仓库环境依赖。
- 校验当前目录是否为 msmodeling 仓库根目录，并检查 `README.md` 与 `requirements.txt`。
- 检查 Python 版本，要求 `3.10+`。
- 默认使用本机检测到的 Python 主次版本创建虚拟环境，避免 `uv` 额外下载 Python；仅在用户指定或本机具备时使用其他版本。
- 检查并安装 `uv`，安装后解析真实 `uv` 可执行路径。
- 根据当前 shell/平台选择 PowerShell 或 Bash 自动化脚本。
- 支持 Windows PowerShell 自动化脚本 `scripts/install-current-project-deps.ps1`。
- 支持 Linux/macOS/WSL/Git Bash 自动化脚本 `scripts/install-current-project-deps.sh`。
- 支持已有干净 Python 环境的 fallback 路径：`pip install -r requirements.txt`。
- fallback 前检查环境中不包含 `torch_npu`、`torch-npu` 或 `cudatoolkit`。
- 支持按需设置当前会话的 `PYTHONPATH` 和 `HF_ENDPOINT=https://hf-mirror.com`。
- 安装完成后执行 `uv pip check --python <venv-python>` 或 `python -m pip check`。

非目标：

- 不修改 `requirements.txt`、README 或项目源码。
- 不自动解决所有平台特定的 PyTorch、NPU、CUDA、驱动兼容问题。
- 不默认持久化系统级环境变量。
- 不替代模型下载、性能仿真、推理验证等后续工作流。
- 不在未经用户确认时执行网络安装、删除环境或覆盖已有环境。

## 3. 用例分析

| 用例 | 用户输入示例 | Skill 行为 | 输出 |
|:---|:---|:---|:---|
| 明确安装 msmodeling 环境 | “帮我安装 msmodeling 的环境” | 检查仓库根目录、Python、`uv`，创建虚拟环境并安装依赖 | 安装结果、激活命令、`pip check` 结果 |
| 模糊环境安装请求 | “帮我安装环境” | 先询问是否安装 msmodeling 当前仓库环境依赖 | 等待用户确认 |
| 自定义环境名 | “新创建名为 lta 的虚拟环境” | 使用 `-EnvName lta` 或 `--env-name lta` 创建环境 | `lta` 激活命令和验证结果 |
| 已有环境 fallback | “我已有环境，只安装 requirements” | 检查 `torch_npu`、`torch-npu`、`cudatoolkit` 后执行 fallback | 依赖安装结果和 `pip check` 结果 |
| Windows 自动化 | “按 README 配置 msmodeling 环境” | 在 PowerShell 中运行 `.ps1` 脚本 | 脚本执行摘要和失败定位 |
| Bash 自动化 | “在 Git Bash 中安装 msmodeling 环境” | 运行 `.sh` 脚本 | 脚本执行摘要和激活命令 |
| 仓库外运行 | “命令找不到 msmodeling 包” | 提示切换到仓库根目录或设置 `PYTHONPATH` | 当前会话环境变量设置命令 |
| Hugging Face 不可达 | “模型下载连不上 HF” | 设置 `HF_ENDPOINT=https://hf-mirror.com` | 当前会话镜像配置和后续命令 |

DFX 要求：

- 兼容性：区分 Windows PowerShell、Linux/macOS shell、WSL/Git Bash 的命令差异。
- 可维护性：核心规则集中在 `SKILL.md`，平台自动化逻辑放入脚本。
- 可测试性：通过脚本帮助输出、语法检查、实际安装输出验证流程稳定性。
- 可靠性：安装前检查目录、Python 版本、依赖文件和命令可用性；安装后执行依赖一致性检查。
- 可诊断性：失败时保留完整命令、关键错误、失败阶段和最小修复建议。

## 4. 方案设计

### 4.1 总体设计

Skill 目录结构：

```text
msmodeling-env-installer/
├── SKILL.md
└── scripts/
    ├── install-current-project-deps.ps1
    └── install-current-project-deps.sh
```

核心流程：

```text
用户提出环境安装请求
        -> 若请求模糊，先确认是否安装 msmodeling 当前仓库环境依赖
        -> 确认当前目录是否为 msmodeling 仓库根目录
        -> 检查 Python 版本与 uv 可用性
        -> 根据当前环境选择 PowerShell 或 Bash 脚本
        -> 选择安装路径：新建虚拟环境 / 使用已有环境 fallback
        -> 安装 requirements.txt
        -> 按需设置 PYTHONPATH / HF_ENDPOINT
        -> 执行 pip check
        -> 输出激活命令、验证结果和后续建议
```

### 4.2 Skill 触发条件

直接触发本 Skill 的场景：

- 用户明确要求安装 msmodeling 环境依赖。
- 用户明确要求初始化 msmodeling 开发环境或按 msmodeling README 配置环境。
- 用户明确要求在当前 msmodeling 仓库创建 `myenv` 或指定名称的虚拟环境。
- 用户明确要求安装当前仓库的 `requirements.txt`。
- 用户明确要求检查当前 msmodeling Python 依赖。
- 用户明确要求为当前 msmodeling 会话配置 `PYTHONPATH`。
- 用户明确要求配置 `HF_ENDPOINT=https://hf-mirror.com`。

需要先确认的场景：

- 用户只说“安装环境”“安装依赖”“配置环境”“初始化环境”，但没有明确说明是 msmodeling、本仓库、`requirements.txt`、`myenv` 或 `uv`。
- 当前仓库后续可能存在其他环境安装工具时，不能默认选择本 Skill。

确认话术：

```text
你是要安装 msmodeling 当前仓库的环境依赖吗？确认后我会使用 msmodeling-env-installer 执行。
```

### 4.3 环境选择策略

根据当前运行环境选择自动化脚本：

| 当前环境 | 优先命令 |
|:---|:---|
| Windows PowerShell | `.\.agents\skills\msmodeling-env-installer\scripts\install-current-project-deps.ps1` |
| Linux/macOS Bash | `bash ./.agents/skills/msmodeling-env-installer/scripts/install-current-project-deps.sh` |
| WSL/Git Bash | `bash ./.agents/skills/msmodeling-env-installer/scripts/install-current-project-deps.sh` |

如果当前 shell 与操作系统不匹配，优先选择当前 shell 可直接执行的脚本。例如在 Windows Git Bash 中使用 `.sh`，在 Windows PowerShell 中使用 `.ps1`。

### 4.4 安装路径设计

跨平台通用安装命令：

```bash
pip install uv -i https://mirrors.ustc.edu.cn/pypi/web/simple
uv venv --python <detected-python-version> myenv
uv pip install --python <venv-python> -r requirements.txt -i https://mirrors.ustc.edu.cn/pypi/web/simple
```

其中 `<detected-python-version>` 默认使用本机检测到的 Python 主次版本，例如 `3.10`。这样可以避免 `uv` 为创建其他版本环境额外下载 Python。

激活命令按操作系统区分：

| 操作系统 | 激活命令 |
|:---|:---|
| Linux/macOS/WSL/Git Bash | `source myenv/bin/activate` |
| Windows PowerShell | `myenv\Scripts\Activate.ps1` |
| Windows cmd | `myenv\Scripts\activate.bat` |

PowerShell 自动化路径：

```powershell
.\.agents\skills\msmodeling-env-installer\scripts\install-current-project-deps.ps1
```

Bash 自动化路径：

```bash
bash ./.agents/skills/msmodeling-env-installer/scripts/install-current-project-deps.sh
```

两个脚本都需要在安装或调用 `uv` 后解析真实 `uv` 可执行路径，避免新安装后 `PATH` 未刷新的问题。脚本默认将 `UV_CACHE_DIR` 指向仓库内 `.uv-cache`，减少默认缓存目录不可写导致的安装失败。

PowerShell 参数：

| 参数 | 说明 |
|:---|:---|
| `-EnvName` | 虚拟环境目录名，默认 `myenv` |
| `-PythonVersion` | `uv venv` 使用的 Python 版本，默认使用检测到的本机 Python 主次版本 |
| `-UseExistingEnv` | 跳过新建 venv，使用已有环境安装依赖 |
| `-SetProjectEnv` | 为当前 PowerShell 会话设置 `PYTHONPATH` |
| `-UseHFMirror` | 为当前 PowerShell 会话设置 `HF_ENDPOINT` |
| `-UseProjectUvCache` | 默认启用，将 `UV_CACHE_DIR` 指向仓库内 `.uv-cache` |

Bash 参数：

| 参数 | 说明 |
|:---|:---|
| `--env-name <name>` | 虚拟环境目录名，默认 `myenv` |
| `--python-version <version>` | `uv venv` 使用的 Python 版本，默认使用检测到的本机 Python 主次版本 |
| `--use-existing-env` | 跳过新建 venv，使用已有环境安装依赖 |
| `--set-project-env` | 输出并为当前脚本进程设置 `PYTHONPATH` |
| `--use-hf-mirror` | 输出并为当前脚本进程设置 `HF_ENDPOINT` |
| `--no-project-uv-cache` | 不设置仓库内 `.uv-cache` 作为 `UV_CACHE_DIR` |

已有环境 fallback：

```bash
python -m pip install -r requirements.txt
python -m pip check
```

执行 fallback 前必须校验当前 Python 环境不包含 `torch_npu`、`torch-npu` 或 `cudatoolkit`。如果检测到任一包，应阻止直接 fallback，并建议用户新建虚拟环境或明确确认继续使用该环境。

环境变量配置按操作系统区分：

| 操作系统 | `PYTHONPATH` | `HF_ENDPOINT` |
| --- | --- | --- |
| Linux/macOS/WSL/Git Bash | `export PYTHONPATH="$(pwd):${PYTHONPATH:-}"` | `export HF_ENDPOINT="https://hf-mirror.com"` |
| Windows PowerShell | ``$env:PYTHONPATH = "$(Get-Location);$env:PYTHONPATH"`` | ``$env:HF_ENDPOINT = "https://hf-mirror.com"`` |

### 4.5 默认值和确认规则

- 默认虚拟环境名为 `myenv`。
- 用户指定环境名时必须尊重，例如 `-EnvName lta` 或 `--env-name lta`。
- 默认 Python 版本为检测到的本机 Python 主次版本；最低要求为 `3.10+`。
- 默认使用中科大 PyPI 镜像安装 `uv` 和依赖。
- 默认在当前会话设置 `UV_CACHE_DIR` 到仓库内 `.uv-cache`，除非用户已有显式 `UV_CACHE_DIR` 或传入禁用参数。
- 默认不覆盖已有环境；如果目标环境目录已存在，需要说明复用或重建的影响并让用户确认。
- 默认不持久化 `PYTHONPATH` 和 `HF_ENDPOINT` 到系统环境变量。
- 涉及网络安装时，应展示将执行的命令，并按当前工具权限请求用户授权。
- 用户请求不明确时，不默认执行 msmodeling 环境安装，必须先确认。

### 4.6 校验规则

执行前校验：

- 当前目录必须包含 `README.md` 和 `requirements.txt`。
- 能找到 `python`、`python3` 或 Windows `py -3` 启动器。
- Python 版本必须大于等于 `3.10.0`。
- 如果用户选择已有环境 fallback，必须检测当前环境中是否安装或可导入 `torch_npu`，以及是否安装 `torch-npu`、`torch_npu`、`cudatoolkit` 包。
- 如果缺少 `uv`，先通过 `python -m pip install uv -i https://mirrors.ustc.edu.cn/pypi/web/simple` 安装。
- 安装 `uv` 后必须解析真实 `uv` 可执行路径，不能假设当前 shell 的 `PATH` 已刷新。
- Windows 新建环境路径下必须能找到 `$EnvName\Scripts\python.exe`。
- Linux/macOS 新建环境路径下必须能找到 `$EnvName/bin/python`。
- Git Bash/MSYS/Cygwin 场景可使用 `$EnvName/Scripts/python.exe`。
- 通过 `uv` 创建环境时，依赖安装完成后优先执行 `uv pip check --python <venv-python>`。

失败分类：

- 仓库路径错误：缺少 `README.md` 或 `requirements.txt`。
- Python 不可用或版本过低。
- 已有环境 fallback 检测到 `torch_npu`、`torch-npu`、`torch_npu` 或 `cudatoolkit`。
- 网络或镜像不可达。
- `uv venv` 创建失败。
- `requirements.txt` 依赖解析或安装失败。
- Windows PyTorch 版本兼容问题。
- `pip check` 发现依赖冲突。

### 4.7 输出摘要

成功执行后输出：

- 完整安装命令或脚本命令。
- Python 版本和虚拟环境路径。
- 依赖安装结果。
- `pip check` 或 `uv pip check` 结果。
- 激活命令，例如 `myenv\Scripts\Activate.ps1` 或 `source myenv/bin/activate`。
- 当前会话中是否设置了 `PYTHONPATH` 或 `HF_ENDPOINT`。

失败时输出：

- 完整命令。
- 关键错误行。
- 失败阶段。
- 最小修复建议，例如切换 Python 版本、检查网络镜像、使用新环境、调整 PyTorch 版本。

## 5. 实施计划

| 阶段 | 内容 | 状态 |
|:---|:---|:---|
| P0 | 创建 `msmodeling-env-installer` Skill 主说明 | 已完成 |
| P0 | 提供 Windows PowerShell 自动化安装脚本 | 已完成 |
| P0 | 修复 `SKILL.md` 与 skills README 乱码 | 已完成 |
| P0 | 补充模糊“安装环境”请求的确认规则 | 已完成 |
| P1 | 补充 Linux/macOS/WSL/Git Bash 自动化脚本 | 已完成 |
| P1 | 补充典型场景验收 prompt | 待执行 |
| P2 | 增加结构化安装日志摘要和错误分类脚本 | 可选 |

## 6. 测试与验收

| 测试场景 | 验收标准 |
|:---|:---|
| 仓库根目录校验 | 不在仓库根目录时阻止安装，并提示切换目录 |
| 模糊触发请求 | 用户只说“安装环境”时先确认是否安装 msmodeling 当前仓库环境依赖 |
| Python 版本校验 | Python 低于 `3.10` 时阻止安装并提示升级 |
| uv 缺失安装 | 缺少 `uv` 时能自动安装或给出可执行命令 |
| uv 路径解析 | 新安装 `uv` 后能解析真实可执行路径，不依赖 `PATH` 立即刷新 |
| 项目内 uv cache | 默认使用仓库内 `.uv-cache` 或尊重已有 `UV_CACHE_DIR` |
| 新建默认环境 | 能生成并执行 `uv venv --python <detected-python-version> myenv` |
| 新建自定义环境 | 能按用户指定名称创建环境，例如 `lta` |
| PowerShell 脚本 | `.ps1` 语法解析通过，能在 Windows PowerShell 中执行 |
| Bash 脚本 | `.sh` 通过 `bash -n`，能在 Git Bash/Linux/macOS 中输出帮助并执行 |
| 依赖安装 | 能执行 `uv pip install --python <venv-python> -r requirements.txt` 并报告结果 |
| 已有环境 fallback | 指定已有环境时不强制创建 `myenv`，且先检查 `torch_npu`、`torch-npu`、`torch_npu` 和 `cudatoolkit` |
| 环境变量设置 | 能按需设置 `PYTHONPATH` 和 `HF_ENDPOINT`，并说明仅影响当前会话或脚本进程 |
| 依赖一致性验证 | uv 创建环境时优先执行 `uv pip check --python <venv-python>` |
| 失败处理 | 能展示命令、关键错误、失败阶段和最小修复建议 |

## 7. 修改文件

| 文件 | 说明 |
|:---|:---|
| `.agents\skills\msmodeling-env-installer\SKILL.md` | Skill 主说明、触发条件和执行规则 |
| `.agents\skills\msmodeling-env-installer\scripts\install-current-project-deps.ps1` | Windows PowerShell 自动化安装脚本 |
| `.agents\skills\msmodeling-env-installer\scripts\install-current-project-deps.sh` | Linux/macOS/WSL/Git Bash 自动化安装脚本 |
| `.agents\skills\README.md` | skills 索引与 quick start |
| `docs/RFC/rfc_msmodeling_env_installer_skill_zh.md` | 本 RFC 文档 |

## 8. 后续演进

- 补充更多典型场景验收 prompt。
- 将安装日志中的常见错误结构化分类，例如 Python 版本、网络镜像、依赖冲突、PyTorch 兼容性。
- 增加环境验证命令，例如最小 import 检查或核心 CLI `--help` 检查。
- 将 README 中的环境安装说明与 Skill 保持同步，避免脚本和文档漂移。
- 根据 Windows PyTorch 兼容性实践沉淀推荐版本矩阵。
