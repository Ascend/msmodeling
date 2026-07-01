---
name: optix-deploy
description: 当用户需要部署 msmodeling optix 服务化自动寻优工具时使用。负责安装与验证。
---

# 服务化参数自动寻优部署

## 工作范围

本 skill 负责将工具装好并验证能用。

其他内容由对应 skill 负责：
- 运行环境检查：`ms-serviceparam-optimizer-env-check`
- 首次参数范围推荐：`optix-param-recommend`
- 生成或修改 `config.toml`：`optix-config`

## 支持的硬件产品

寻优工具仅支持以下昇腾推理产品：

|产品类型| 是否支持 |
|--|:----:|
|Atlas A3 训练系列产品/Atlas A3 推理系列产品|  √   |
|Atlas A2 训练系列产品/Atlas A2 推理系列产品|  √   |
|Atlas 200I/500 A2 推理产品|  √   |
|Atlas 推理系列产品|  √   |
|Atlas 训练系列产品|  x   |

> - 目标运行环境不支持 Windows
> - Atlas 训练系列产品不支持，请确认使用的是推理系列产品

## 安装流程

用户说"安装""部署"时，直接执行以下步骤，**不要只给命令**。

完整说明见 [OptiX 使用指南 · 推荐实践：环境与部署栈](../../../docs/zh/user_guide/optix_user_guide.md#推荐实践环境与部署栈)。

**为何必须用 venv**：`uv sync` 会装上 `torch`、`transformers` 等，给 TensorCast 仿真用，不是 OptiX 真机寻优用的。装到系统 Python 会改掉 vLLM、MindIE 依赖的版本，服务可能起不来。在仓库根目录 `uv sync` 会自动创建 `.venv` 并隔离安装；vLLM、MindIE 继续用系统里那套。

### 0. 默认使用阿里云 PyPI 镜像

为减少安装等待时间，默认优先使用阿里云镜像：

```bash
https://mirrors.aliyun.com/pypi/simple/
```

> 注意：不要写成 `PYPI_MIRROR=... python -m pip install -i "$PYPI_MIRROR" ...`。
> 这种写法里 `"$PYPI_MIRROR"` 会在临时环境变量生效前被 shell 展开，结果可能变成空字符串，导致 pip 实际拿到空的 index URL。
>
> 优先直接把镜像地址写进 `-i` 参数；如果确实要用环境变量，必须先单独 `export`，再在后续命令里使用。

若用户已有公司内网镜像、离线源或明确指定其他源，则尊重用户配置，不强制改成阿里云镜像。`uv sync` 使用 `pyproject.toml` / `uv.lock` 中的索引配置；需要额外 PyPI 镜像时可通过 `UV_INDEX_URL` 等 uv 环境变量配置。

### 1. 检查仓库是否存在

寻优工具已集成在 msmodeling 项目根目录中。检查安装入口与默认配置：

```bash
ls optix/config.toml pyproject.toml
```

若仓库不存在，**直接克隆**：

```bash
git clone https://gitcode.com/Ascend/msmodeling.git
cd msmodeling
```

### 2. 安装 msmodeling / OptiX

msmodeling 仓库结构：

```
msmodeling/                  ← 仓库根目录
├── pyproject.toml           ← msmodeling 安装入口（含 optix CLI）
└── optix/                   ← 寻优工具源码（含 config.toml、optimizer 代码）
```

**在仓库根目录执行 `uv sync`**（勿在系统 Python 中安装）：

- 自动创建 `.venv`（若不存在）
- 以可编辑模式安装 msmodeling（含 `msmodeling optix` CLI）
- **无需** `uv venv` 或 `pip install -e .`

安装时不要依赖上一次 shell 的 `cd` 状态。应优先使用**单条命令**：

```bash
cd /path/to/msmodeling && uv sync
```

若用户未检测到 venv，输出警告（与 `[optix/env]` 场景 A 一致），说明应使用 `uv sync` 而非系统 Python。

> **禁止**：在 msmodeling venv 中 `pip install vllm`；在 vLLM 部署 venv 中对 msmodeling 仓库执行 `pip install -e .`（应使用 `uv sync`）。

### 3. 验证

```bash
uv run msmodeling optix --help
```

### 4. 汇报结果

告诉用户安装是否成功。若失败，给出具体缺失项和修复建议。

安装成功后额外提示（用一句话说清原因）：

- **OptiX 必须装在 uv venv**：`uv sync` 会带入 `torch`/`transformers`（仿真用），装到系统会 **污染依赖 → vLLM/MindIE 可能启动失败**
- **勿**在 msmodeling venv 里 `pip install vllm`；部署栈默认走 **系统环境**
- PATH 特殊时才配 `OPTIX_DEPLOY_PATH` 或 `[deploy] path_prefix`
- 下一步：运行环境检查、配置 `config.toml` 或开始参数推荐

## 卸载

卸载前说明会移除 Python 包，征得同意后执行：

```bash
uv pip uninstall msmodeling
```

## 要求

- 始终中文回答。
- 用户同意安装后，要实际检查路径、执行安装并验证 CLI，不要只给命令。
- 不要把运行、配置、结果解读内容塞进本 skill；交给专项 skill 或文档。
- 涉及会创建目录、安装依赖、卸载包的动作，先说明影响并征得用户同意。
