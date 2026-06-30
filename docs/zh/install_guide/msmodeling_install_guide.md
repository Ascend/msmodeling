# 快速上手：环境搭建与第一次仿真

## 本章目标

读完本章，你将能够搭建好开发环境，成功运行一次 LLM 推理仿真，并理解从命令行到结果输出的完整调用链。

## 适用对象与前置条件

本文适用于首次使用 msModeling 的开发者和测试人员。开始前请确认：

- 已安装 Python 3.10 或更高版本，推荐使用独立虚拟环境。
- 运行环境可访问 GitCode 与 Python 包源。
- 若需要直接拉取 Hugging Face 模型配置，运行环境可访问 Hugging Face；否则可按本文设置镜像。

## 环境搭建

### 1. 克隆仓库

```bash
git clone https://gitcode.com/Ascend/msmodeling.git
cd msmodeling
```

### 2. 推荐方式：uv

项目推荐使用 `uv` 管理虚拟环境和依赖。仓库包含 `pyproject.toml` 时，`scripts/` 下的脚本也会自动识别并使用 `uv`。

```bash
pip install uv
uv venv --python 3.13 .venv

# Linux / macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate

uv sync

# 可选：安装 lint 或 CI 相关依赖
uv sync --group lint
uv sync --group ci
```

完成后，可在已激活的虚拟环境中直接运行命令，也可以使用 `uv run ...` 执行命令。

### 3. 备选方式：pip + requirements.txt

如果不使用 `uv`，也可以通过 Python 原生虚拟环境和 `requirements.txt` 安装依赖。CPU 环境建议先从 PyTorch CPU 源安装 `torch` 与 `torchvision`，再安装其余依赖。

```bash
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate

pip install "torch>=2.7,<=2.10" "torchvision>=0.25.0" --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

> [!NOTE]
> `requirements.txt` 顶部也保留了 pip 安装路径和 CPU PyTorch 安装顺序说明。

如果依赖下载失败或速度较慢，可临时切换 PyPI 镜像源后重试：

```bash
# 临时使用清华源
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 或临时使用阿里云源
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple

# 或临时使用华为云源
pip install -r requirements.txt -i https://repo.huaweicloud.com/repository/pypi/simple
```

如某个镜像源同步不及时导致版本找不到，请更换其他镜像源或临时回退到官方源 `https://pypi.org/simple` 后重试。

<!---->

> [!WARNING]
> Windows 上 PyTorch 2.10 可能运行不正常。如遇问题，建议使用 PyTorch 2.8 或更早版本。

### 4. 设置 PYTHONPATH

如果不在 msmodeling 根目录下运行，需要设置：

```bash
# Linux / macOS
export PYTHONPATH=/path/to/msmodeling:$PYTHONPATH

# Windows PowerShell
$env:PYTHONPATH = "C:\path\to\msmodeling;$env:PYTHONPATH"
```

### 5. Hugging Face 访问

工具运行时需要从 Hugging Face 读取模型配置文件。如果无法直接访问，可以设置镜像：

```bash
# Linux / macOS
export HF_ENDPOINT="https://hf-mirror.com"

# Windows PowerShell
$env:HF_ENDPOINT = "https://hf-mirror.com"
```

在受限网络中，即使设置 `HF_ENDPOINT`，仍可能因代理策略、DNS、TLS 证书、镜像站不可达、模型仓库需要鉴权，或依赖库未使用该环境变量而下载失败。此时建议使用以下方案：

优先在可访问外网的环境中提前下载并审核模型仓库中的配置文件（仅需 `.json`、`.yaml`、`.yml`、`.txt` 后缀），再将 `model_id` 指向本地绝对路径：

   ```bash
   python -m cli.inference.text_generate /data/models/Qwen3-32B --num-queries 2 --query-length 3500 --device TEST_DEVICE
   ```

## 验证安装

依赖安装完成后，请在 **msModeling 仓库根目录** 下激活虚拟环境，执行以下命令确认 CLI 入口可用：

```bash
python -m cli.inference.text_generate --help
python -m cli.inference.throughput_optimizer --help
python -m serving_cast.main --help
```

若安装正常，上述命令应分别输出 `text_generate`、`throughput_optimizer`、`serving_cast` 的用法说明与参数列表，且不报 `ModuleNotFoundError`。

常见问题：

- 若 `--help` 无法显示帮助信息，请参考上文「验证安装」章节排查虚拟环境、`PYTHONPATH` 与依赖安装。
- 如果提示无法找到 `cli` 或 `tensor_cast` 模块，请确认当前目录为仓库根目录，或已正确设置 `PYTHONPATH`。
- 如果模型配置下载失败，请确认网络可访问 Hugging Face；若 `HF_ENDPOINT` 镜像仍不可用，请改用本地模型路径。
- 如果依赖安装失败，请先确认虚拟环境已激活，并重新执行 `uv sync`；若使用 pip 方式，请升级 `pip` 后重新执行 `pip install -r requirements.txt`，必要时切换 PyPI 镜像源。
