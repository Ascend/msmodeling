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
export HF_ENDPOINT="https://hf-mirror.com"
```

## 第一次仿真

环境准备完成后，可以运行一次最小 LLM 推理仿真：

```bash
python -m cli.inference.text_generate Qwen/Qwen3-32B --num-queries 2 --query-length 3500 --device TEST_DEVICE
```

运行完成后，终端会输出算子级性能汇总、总执行时间、TPS/Device 以及显存占用等信息。更详细的使用方法请参考《[TensorCast 使用指南](../user_guide/msmodeling_tensor_cast_user_guide.md)》。

## 结果校验与下一步

如果命令执行成功，通常可以看到以下信息：

- 算子级性能表，包括 `analytic total`、`analytic avg` 与 `# of Calls`。
- `Total time for analytic`、`TPS/Device` 等总体性能指标。
- 权重、KV cache、activation 与可用显存等内存估算结果。

常见问题：

- 如果提示无法找到 `cli` 或 `tensor_cast` 模块，请确认当前目录为仓库根目录，或已正确设置 `PYTHONPATH`。
- 如果模型配置下载失败，请确认网络可访问 Hugging Face，或设置 `HF_ENDPOINT` 镜像。
- 如果依赖安装失败，请先确认虚拟环境已激活，并重新执行 `uv sync`；若使用 pip 方式，请升级 `pip` 后重新执行 `pip install -r requirements.txt`。
