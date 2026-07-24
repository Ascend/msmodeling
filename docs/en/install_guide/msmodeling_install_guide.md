# msModeling Installation Guide

## 1. Installation Notes

msModeling is a tool for large model inference performance simulation, service-level performance simulation, and OptiX service parameter optimization. After reading this guide, you will be able to install the environment, verify the command-line entry points, and run a basic simulation.

This guide is intended for developers and testers who are using msModeling for the first time. Before you start, make sure that:

- Python 3.10 or later is installed. An isolated virtual environment is recommended.
- The runtime environment can access GitCode and Python package indexes.
- If you need to pull Hugging Face model configuration files directly, the runtime environment can access Hugging Face; otherwise, configure a mirror or use a local model path as described below.

msModeling currently supports source installation only. Online installers, offline installers, and run packages are not covered in this guide.

## 2. Source Installation

### 2.1 Clone the Source Code

Run the following commands:

```bash
git clone https://gitcode.com/Ascend/msmodeling.git
cd msmodeling
```

### 2.2 Recommended: uv

The project recommends using `uv` to manage the virtual environment and dependencies. When the repository contains `pyproject.toml`, scripts under `scripts/` also automatically detect and use `uv`.

```bash
pip install uv
cd msmodeling
uv sync

# Optional: specify the Python version (defaults to an available local version)
# UV_PYTHON=3.13 uv sync

# Optional: install lint or CI dependency groups
uv sync --group lint
uv sync --group ci
```

After setup, use `uv run ...` to run commands. If you need to activate the virtual environment manually, activate the `.venv` automatically created by `uv sync`.

> [!NOTE]
> If you use `uv` to create or manage the virtual environment, use `uv pip ...` or `uv run ...` for later inspection, upgrade, and uninstallation. Do not rely only on `which pip` to determine the active environment, because `pip` may point to an unexpected Python environment in some cases.

### 2.3 Alternative: pip + requirements.txt

If you do not use `uv`, you can use Python's built-in virtual environment and `requirements.txt` to install dependencies. In CPU environments, install `torch` and `torchvision` from the PyTorch CPU index before installing the remaining dependencies.

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
> `pip install -e .` installs msModeling in editable source mode and registers the `msmodeling` CLI. After source code updates, you do not need to copy files again; rerun the installation command when needed.

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

> [!WARNING]
> PyTorch 2.10 may not run properly on Windows. If you encounter issues, use PyTorch 2.8 or earlier.

### 2.4 Configure Environment Variables

Common msModeling environment variables are as follows:

| Environment Variable | Optional/Required | Description |
| -------------------- | ----------------- | ----------- |
| PYTHONPATH | Optional | If commands are not run from the msModeling repository root, set this variable to the repository root to avoid errors such as `No module named cli` or `No module named tensor_cast`. |
| HF_ENDPOINT | Optional | If Hugging Face cannot be accessed directly, set this variable to a Hugging Face mirror, for example `https://hf-mirror.com`. |
| OPTIX_DEPLOY_PATH | Optional | If you use OptiX and the system `PATH` is non-standard, set this variable to the path that contains deployment stack commands. Usually this is not required. |

If you do not run commands from the msModeling repository root, set `PYTHONPATH`:

```bash
# Linux / macOS
export PYTHONPATH=/path/to/msmodeling:$PYTHONPATH

# Windows PowerShell
$env:PYTHONPATH = "C:\path\to\msmodeling;$env:PYTHONPATH"
```

The tool may need to read model configuration files from Hugging Face at runtime. If direct access is unavailable, set a mirror endpoint:

```bash
# Linux / macOS
export HF_ENDPOINT="https://hf-mirror.com"

# Windows PowerShell
$env:HF_ENDPOINT = "https://hf-mirror.com"
```

In restricted networks, downloads may still fail even with `HF_ENDPOINT` set because of proxy policies, DNS, TLS certificates, mirror availability, model repository authentication, or dependency libraries not using this environment variable. In this case, use a reviewed local model path.

## 3. Verify the Installation

After installation, run the following commands in the activated Python environment to verify that the CLI entry points are available.

```bash
python -m cli.inference.text_generate --help
python -m cli.inference.throughput_optimizer --help
python -m serving_cast.main --help
msmodeling optix --help
pip show msmodeling
```

If the installation is correct, the commands above should print usage and argument lists for `text_generate`, `throughput_optimizer`, `serving_cast`, and `msmodeling optix`, without `ModuleNotFoundError`.

To run a basic simulation, prefer downloading and reviewing the model repository configuration files in an environment with internet access first. Only files ending with `.json`, `.yaml`, `.yml`, or `.txt` are required. Then point `model_id` to the local absolute path:

```bash
python -m cli.inference.text_generate /data/models/Qwen3-32B --num-queries 2 --query-length 3500 --device TEST_DEVICE
```

If the command cannot run properly, confirm that the current terminal has activated the Python environment where msModeling is installed.

## 4. Uninstall

Run the following command in the Python environment where msModeling is installed.

If you use `uv` to manage the virtual environment, run:

```bash
uv pip uninstall msmodeling
```

If you use the `pip + requirements.txt` method, run:

```bash
pip uninstall msmodeling
```

> [!NOTE]
> Before uninstalling, confirm that the current terminal is using the Python environment where msModeling is installed, to avoid uninstalling a package with the same name from another environment. If the environment is managed by `uv`, prefer `uv pip uninstall msmodeling`. If the source directory is no longer needed, delete it manually after uninstallation.

## 5. Upgrade

Before upgrading, check the version in the current environment:

```bash
# uv environment
uv pip show msmodeling

# pip environment
pip show msmodeling
```

Enter the msModeling repository root, pull the target version source code, and upgrade:

```bash
cd msmodeling
git fetch
git checkout 26.1.0
git pull

# uv environment
uv pip install --upgrade -e .

# uv environment with a temporary mirror
uv pip install --upgrade -e . -i https://mirrors.aliyun.com/pypi/simple

# pip environment
pip install --upgrade -e .
```

When upgrading versions, pay attention to version compatibility. See the [Release Notes](https://gitcode.com/Ascend/release-management/blob/master/MindStudio/26.1.0/release_notes.md).

## 6. Appendix

### 6.1 OptiX and Simulation Environment Isolation<a name="optix-and-simulation-environment-isolation"></a>

If you use [OptiX service parameter optimization](../user_guide/optix_user_guide.md):

- Install msModeling and OptiX in an isolated virtual environment such as `.venv`. The installation brings in dependencies such as `torch` and `transformers`; they are used for simulation, not for the OptiX deployment stack.
- vLLM, MindIE, and benchmark tools use the system deployment environment by default. Usually, you do not need to create another deployment virtual environment.
- Do not run `pip install vllm` in the msModeling virtual environment.

OptiX child processes automatically strip the msModeling virtual environment and use the system `PATH`. Set `OPTIX_DEPLOY_PATH` only when `PATH` is non-standard. For details, see [OptiX User Guide - Recommended Practice: Environment and Deploy Stack](../user_guide/optix_user_guide.md#recommended-practice-environment-and-deploy-stack).

### 6.2 Troubleshooting

- If `--help` cannot display help, first check the virtual environment, `PYTHONPATH`, and dependency installation.
- If `cli` or `tensor_cast` cannot be found, confirm that the current directory is the repository root or that `PYTHONPATH` is configured correctly.
- If model configuration download fails, confirm that the network can access Hugging Face. If the `HF_ENDPOINT` mirror is still unavailable, use a local model path.
- If dependency installation fails, first confirm that the virtual environment is activated. If you use `uv`, rerun `uv sync`; if you use pip, upgrade `pip` and rerun `pip install -r requirements.txt` followed by `pip install -e .`. Switch PyPI mirrors if needed.
