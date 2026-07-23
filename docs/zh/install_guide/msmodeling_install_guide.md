# msModeling工具安装指南

## 1. 安装说明

msModeling 是面向大模型推理性能仿真、服务化吞吐优化和 OptiX 服务化实测寻优的工具。读完本文，您将能够完成环境安装、验证命令行入口，并运行一次基础仿真。

本文适用于首次使用 msModeling 的开发者和测试人员。开始前请确认：

- 已安装 Python 3.10 或更高版本，推荐使用独立虚拟环境。
- 运行环境可访问 GitCode 与 Python 包源。
- 若需要直接拉取 Hugging Face 模型配置，运行环境可访问 Hugging Face；否则请按本文配置镜像或使用本地模型路径。

msModeling 当前仅支持源码安装。本文不涉及在线安装包、离线安装包或 run 包安装方式。

## 2. 源码安装

### 2.1 克隆源码

执行如下命令下载 26.1.0 分支源码：

```bash
git clone -b 26.1.0 https://gitcode.com/Ascend/msmodeling.git
cd msmodeling
```

### 2.2 推荐方式：uv

项目推荐使用 `uv` 管理虚拟环境和依赖。仓库包含 `pyproject.toml` 时，`scripts/` 下的脚本也会自动识别并使用 `uv`。

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

完成后，可使用 `uv run ...` 执行命令；如需手动激活虚拟环境，可激活 `uv sync` 自动创建的 `.venv`。

> [!NOTE]
> 如果使用 `uv` 创建或管理虚拟环境，后续查看、升级、卸载也建议使用 `uv pip ...` 或 `uv run ...`。不要仅通过 `which pip` 判断当前环境，部分场景下 `pip` 可能指向非预期的 Python 环境。

### 2.3 备选方式：pip + requirements.txt

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

> [!NOTE]
> `pip install -e .` 会以源码可编辑模式安装 msModeling，并注册 `msmodeling` CLI。源码更新后无需重新复制文件，必要时重新执行安装命令即可。

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

> [!WARNING]
> Windows 上 PyTorch 2.10 可能运行不正常。如遇问题，建议使用 PyTorch 2.8 或更早版本。

### 2.4 配置环境变量

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

## 3. 验证安装

安装完成后，在已激活的 Python 环境中执行以下命令确认 CLI 入口可用。

```bash
python -m cli.inference.text_generate --help
python -m cli.inference.throughput_optimizer --help
python -m serving_cast.main --help
msmodeling optix --help
pip show msmodeling
```

若安装正常，上述命令应分别输出 `text_generate`、`throughput_optimizer`、`serving_cast` 和 `msmodeling optix` 的用法说明与参数列表，且不报 `ModuleNotFoundError`。

如需运行一次基础仿真，建议优先在可访问外网的环境中提前下载并审核模型仓库中的配置文件（仅需 `.json`、`.yaml`、`.yml`、`.txt` 后缀），再将 `model_id` 指向本地绝对路径：

```bash
python -m cli.inference.text_generate /data/models/Qwen3-32B --num-queries 2 --query-length 3500 --device TEST_DEVICE
```

若命令无法正常执行，请确认当前终端已激活安装 msModeling 的 Python 环境。

## 4. 卸载

可在安装 msModeling 的 Python 环境中执行如下命令卸载。

如果使用 `uv` 管理虚拟环境，执行：

```bash
uv pip uninstall msmodeling
```

如果使用 `pip + requirements.txt` 方式安装，执行：

```bash
pip uninstall msmodeling
```

> [!NOTE]
> 卸载前请确认当前终端使用的是安装 msModeling 的 Python 环境，避免卸载到其他环境中的同名包。若通过 `uv` 管理环境，优先使用 `uv pip uninstall msmodeling`。若不再需要源码目录，可在卸载后手动删除。

## 5. 升级

升级前可先查看当前环境中的版本信息：

```bash
# uv 环境
uv pip show msmodeling

# pip 环境
pip show msmodeling
```

进入 msModeling 仓库根目录，拉取目标版本源码后升级：

```bash
cd msmodeling
git fetch
git checkout 26.1.0
git pull

# uv 环境
uv pip install --upgrade -e .

# uv 环境临时指定镜像源
uv pip install --upgrade -e . -i https://mirrors.aliyun.com/pypi/simple

# pip 环境
pip install --upgrade -e .
```

升级版本时需要关注版本配套关系，请参见《[版本说明](https://gitcode.com/Ascend/release-management/blob/master/MindStudio/26.1.0/release_notes.md)》。

## 6. 附录

### 6.1 OptiX 与仿真环境分离<a name="optix-与仿真环境分离"></a>

若使用 [OptiX 服务化自动寻优](../user_guide/optix_user_guide.md)：

- msModeling、OptiX 必须装在独立虚拟环境里，例如 `.venv`。安装会带上 `torch`、`transformers` 等依赖，它们给仿真使用，不是 OptiX 寻优使用的部署栈。
- vLLM、MindIE、测评工具默认使用系统里已部署好的环境，一般不必再创建部署虚拟环境。
- 不要在 msModeling 虚拟环境里执行 `pip install vllm`。

OptiX 子进程会自动剥离 msModeling 虚拟环境，使用系统 `PATH`；仅当 `PATH` 特殊时可配置 `OPTIX_DEPLOY_PATH`。详见《[OptiX 使用指南 · 推荐实践：环境与部署栈](../user_guide/optix_user_guide.md#推荐实践环境与部署栈)》。

### 6.2 常见问题

- 若 `--help` 无法显示帮助信息，请优先排查虚拟环境、`PYTHONPATH` 与依赖安装。
- 如果提示无法找到 `cli` 或 `tensor_cast` 模块，请确认当前目录为仓库根目录，或已正确设置 `PYTHONPATH`。
- 如果模型配置下载失败，请确认网络可访问 Hugging Face；若 `HF_ENDPOINT` 镜像仍不可用，请改用本地模型路径。
- 如果依赖安装失败，请先确认虚拟环境已激活。若使用 `uv`，请重新执行 `uv sync`；若使用 pip 方式，请升级 `pip` 后依次重新执行 `pip install -r requirements.txt` 和 `pip install -e .`，必要时切换 PyPI 镜像源。
