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

项目推荐使用 `uv` 管理虚拟环境和依赖。在仓库根目录执行 `uv sync` 会**自动创建** `.venv`、以可编辑模式安装本项目（含 `msmodeling` CLI），并同步 `uv.lock` 中的依赖；**无需**先执行 `uv venv` 或 `pip install -e .`。`scripts/` 下的脚本也会自动识别并使用 `uv`。

```bash
pip install uv
cd msmodeling
uv sync

# 可选：指定 Python 版本（默认使用本机可用版本）
# UV_PYTHON=3.13 uv sync

# 可选：安装 lint 或 CI 相关依赖
uv sync --group lint
uv sync --group ci
```

完成后可直接使用 `uv run ...`（推荐），或激活自动创建的虚拟环境：

```bash
# Linux / macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

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

## OptiX 与仿真环境分离<a name="optix-与仿真环境分离"></a>

若使用 [OptiX 服务化自动寻优](../user_guide/optix_user_guide.md)：

- msmodeling、OptiX **必须**装在 uv 虚拟环境里，例如 `.venv`：执行 `uv sync` 即可。安装会带上 `torch`、`transformers` 等，它们给仿真用，不是 OptiX 寻优用的。写进系统 Python 会冲掉部署栈里的版本，vLLM、MindIE 可能起不来或推理报错。
- vLLM、MindIE、测评工具默认走 **系统里已部署好的环境**，一般不必再建部署 venv。
- 不要在 msmodeling venv 里 `pip install vllm`。

OptiX 子进程会自动剥离 msmodeling venv，走系统 PATH；仅当 PATH 特殊时可配置 `OPTIX_DEPLOY_PATH`。详见《[OptiX 使用指南 · 推荐实践：环境与部署栈](../user_guide/optix_user_guide.md#推荐实践环境与部署栈)》。

## 第一次仿真

优先在可访问外网的环境中提前下载并审核模型仓库中的配置文件（仅需 `.json`、`.yaml`、`.yml`、`.txt` 后缀），再将 `model_id` 指向本地绝对路径：

   ```bash
   python -m cli.inference.text_generate /data/models/Qwen3-32B --num-queries 2 --query-length 3500 --device TEST_DEVICE
   ```

## 验证安装

依赖安装完成后，在 **msModeling 仓库根目录** 执行以下命令确认 CLI 入口可用（`uv run` 无需手动激活虚拟环境）：

```bash
uv run python -m cli.inference.text_generate --help
uv run python -m cli.inference.throughput_optimizer --help
uv run python -m serving_cast.main --help
uv run msmodeling optix --help
```

若安装正常，上述命令应分别输出 `text_generate`、`throughput_optimizer`、`serving_cast` 的用法说明与参数列表，且不报 `ModuleNotFoundError`。

常见问题：

- 若 `--help` 无法显示帮助信息，请参考上文「验证安装」章节排查虚拟环境、`PYTHONPATH` 与依赖安装。
- 如果提示无法找到 `cli` 或 `tensor_cast` 模块，请确认当前目录为仓库根目录，或已正确设置 `PYTHONPATH`。
- 如果模型配置下载失败，请确认网络可访问 Hugging Face；若 `HF_ENDPOINT` 镜像仍不可用，请改用本地模型路径。
- 如果依赖安装失败，请先确认虚拟环境已激活，并重新执行 `uv sync`；若使用 pip 方式，请升级 `pip` 后重新执行 `pip install -r requirements.txt`，必要时切换 PyPI 镜像源。
