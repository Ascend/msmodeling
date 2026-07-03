# Quick Start: Environment Setup and First Simulation

## Goal

After reading this guide, you will be able to set up the development environment, successfully run an LLM inference simulation, and understand the full flow from command line invocation to result output.

## Audience and Prerequisites

This guide is intended for developers and testers who are using msModeling for the first time. Before you start, make sure that:

- Python 3.10 or later is installed. An isolated virtual environment is recommended.
- The runtime environment can access GitCode and Python package indexes.
- If you need to pull Hugging Face model configuration files directly, the runtime environment can access Hugging Face; otherwise, configure a mirror as described below.

## Environment Setup

### 1. Clone the Repository

```bash
git clone https://gitcode.com/Ascend/msmodeling.git
cd msmodeling
```

### 2. Recommended: uv

The project recommends using `uv` to manage the virtual environment and dependencies. When the repository contains `pyproject.toml`, scripts under `scripts/` can also detect and use `uv` automatically.

```bash
pip install uv
uv venv --python 3.13 .venv

# Linux / macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate

uv sync

# Optional: install lint or CI dependencies
uv sync --group lint
uv sync --group ci
```

After setup, you can run commands directly in the activated virtual environment, or use `uv run ...`.

### 3. Alternative: pip + requirements.txt

If you do not use `uv`, install dependencies with Python's built-in virtual environment and `requirements.txt`. In CPU environments, install `torch` and `torchvision` from the PyTorch CPU index before installing the remaining dependencies.

```bash
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate

pip install "torch>=2.8,<=2.10" "torchvision>=0.23.0" --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

> [!NOTE]
> The top of `requirements.txt` also keeps notes about the pip installation path and CPU PyTorch installation order.

If dependency downloads fail or are slow, temporarily switch to a PyPI mirror and retry:

```bash
# Temporarily use Tsinghua mirror
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# Or temporarily use Alibaba Cloud mirror
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple

# Or temporarily use Huawei Cloud mirror
pip install -r requirements.txt -i https://repo.huaweicloud.com/repository/pypi/simple
```

If a mirror is not synchronized in time and a version cannot be found, switch to another mirror or temporarily fall back to the official index `https://pypi.org/simple`.

<!---->

> [!WARNING]
> PyTorch 2.10 may not run properly on Windows. If you encounter issues, use PyTorch 2.8 or earlier.

### 4. Set PYTHONPATH

If you do not run commands from the msmodeling repository root, set `PYTHONPATH` first:

```bash
# Linux / macOS
export PYTHONPATH=/path/to/msmodeling:$PYTHONPATH

# Windows PowerShell
$env:PYTHONPATH = "C:\path\to\msmodeling;$env:PYTHONPATH"
```

### 5. Hugging Face Access

The tool reads model configuration files from Hugging Face at runtime. If direct access is unavailable, set a mirror endpoint:

```bash
# Linux / macOS
export HF_ENDPOINT="https://hf-mirror.com"

# Windows PowerShell
$env:HF_ENDPOINT = "https://hf-mirror.com"
```

In restricted networks, downloads can still fail even with `HF_ENDPOINT` set because of proxy policies, DNS, TLS certificates, mirror availability, model repository authentication, or dependency libraries not using this environment variable. In this case, use the following approach:

Prefer downloading and reviewing the model repository configuration files in an environment with internet access first. Only files ending with `.json`, `.yaml`, `.yml`, or `.txt` are required. Then point `model_id` to the local absolute path:

   ```bash
   python -m cli.inference.text_generate /data/models/Qwen3-32B --num-queries 2 --query-length 3500 --device TEST_DEVICE
   ```

## Verify the Installation

After installing dependencies, activate the virtual environment under the **msModeling repository root** and run the following commands to verify that CLI entry points are available:

```bash
python -m cli.inference.text_generate --help
python -m cli.inference.throughput_optimizer --help
python -m serving_cast.main --help
```

If the installation is correct, the commands above should print usage and argument lists for `text_generate`, `throughput_optimizer`, and `serving_cast` respectively, without `ModuleNotFoundError`.

Troubleshooting:

- If `--help` cannot display help, check the virtual environment, `PYTHONPATH`, and dependency installation according to the "Verify the Installation" section above.
- If `cli` or `tensor_cast` cannot be found, confirm that the current directory is the repository root or that `PYTHONPATH` is configured correctly.
- If model configuration download fails, confirm that the network can access Hugging Face. If the `HF_ENDPOINT` mirror is still unavailable, use a local model path.
- If dependency installation fails, first confirm that the virtual environment is activated and rerun `uv sync`. If you use pip, upgrade `pip` and rerun `pip install -r requirements.txt`; switch PyPI mirrors if needed.
