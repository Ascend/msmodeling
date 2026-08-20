# msModeling Installation Guide

## 1. Installation Instructions

msModeling is a tool for LLM inference performance simulation, serving throughput optimization, and OptiX-based serving optimization through real-world testing. After reading this guide, you will be able to complete the environment installation, verify the CLI entry, and run a basic simulation.

This guide applies to developers and testers who use msModeling for the first time. Before you start, ensure that:

- Python 3.10 or later is installed. An independent virtual environment is recommended.
- The runtime environment can access GitCode and Python package sources.
- If you need to pull Hugging Face model configurations directly, ensure that the runtime environment can access Hugging Face. Otherwise, configure a mirror or use a local model path as described in this guide.

## 2. Installation Methods

### 2.1 Online Installation

If your device has internet access, you can automatically download and install the tool with one command. See the MindStudio [Download](https://www.hiascend.com/developer/software/mindstudio/download?versionId=147&ids=45%2C89dda9ba9de741349efa03687a487678%2C91%2C49%2C) page on the Ascend Community, select the "Inference Development" scenario, the corresponding CANN version, and the corresponding tool, and then select "Online Installation" as the installation method. The system guides you through the remaining operations.

### 2.2 Offline Installation

For devices in an environment without external network access, such as an enterprise intranet, download the complete offline installation package on a machine with internet access first, and then transfer it to the target device for installation. See the MindStudio [Download](https://www.hiascend.com/developer/software/mindstudio/download?versionId=147&ids=45%2C89dda9ba9de741349efa03687a487678%2C91%2C50%2C) page on the Ascend Community, select the "Inference Development" scenario, the corresponding CANN version, and the corresponding tool, and then select "Offline Installation" as the installation method to obtain the corresponding installation package and operation instructions.

### 2.3 Source Installation

If you want to use the features of the latest code or modify the source code to enhance features, download the code in this repository, compile and package the tool yourself, and complete the installation.

#### 2.3.1 Cloning the Source Code

Run the following commands to download the source code of the 26.1.0 branch:

```bash
git clone -b 26.1.0 https://gitcode.com/Ascend/msmodeling.git
cd msmodeling
```

#### 2.3.2 Recommended Method: `uv`

The project recommends using `uv` to manage virtual environments and dependencies. When the repository contains `pyproject.toml`, the scripts in `scripts/` also automatically detect and use `uv`.

```bash
pip install uv
cd msmodeling
uv sync

# Optional: specify the Python version (the version available on the local machine is used by default)
# UV_PYTHON=3.13 uv sync

# Optional: install lint or CI-related dependencies
uv sync --group lint
uv sync --group ci
```

After that, you can run commands with `uv run ...`. If you want to activate the virtual environment manually, activate `.venv`, which `uv sync` creates automatically.

> [!NOTE]
> If you use `uv` to create or manage virtual environments, you are advised to also use `uv pip ...` or `uv run ...` for subsequent viewing, upgrading, and uninstalling. Do not determine the current environment only with `which pip`, because `pip` may point to an unexpected Python environment in some scenarios.

#### 2.3.3 Alternative Method: `pip` + `requirements.txt`

If you do not use `uv`, you can also install dependencies with the native Python virtual environment and `requirements.txt`. In a CPU environment, you are advised to install `torch` and `torchvision` from the PyTorch CPU source first, and then install the remaining dependencies.

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
> `pip install -e .` installs msModeling in source editable mode and registers the `msmodeling` CLI. You do not need to copy files again after source code updates. Run the installation command again when necessary.

If dependency downloads fail or are slow, you can temporarily switch to a PyPI mirror and try again:

```bash
# Temporarily use the Tsinghua mirror
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# Or temporarily use the Alibaba Cloud mirror
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple

# Or temporarily use the Huawei Cloud mirror
pip install -r requirements.txt -i https://repo.huaweicloud.com/repository/pypi/simple
```

If a mirror does not synchronize in time and the version cannot be found, switch to another mirror or temporarily fall back to the official source `https://pypi.org/simple` and try again.

> [!WARNING]
> PyTorch 2.10 may not run properly on Windows. If you encounter issues, use PyTorch 2.8 or earlier.

#### 2.3.4 Configuring Environment Variables

The commonly used environment variables of msModeling are as follows:

| Environment Variable | Optional/Required | Description |
| -------------------- | ----------------- | ----------- |
| PYTHONPATH | Optional | If you do not run commands from the msModeling repository root, set this variable to the repository root to avoid module import errors such as `No module named cli` and `No module named tensor_cast`. |
| HF_ENDPOINT | Optional | If you cannot access Hugging Face directly, configure the Hugging Face mirror address, for example `https://hf-mirror.com`. |
| OPTIX_DEPLOY_PATH | Optional | If you use OptiX and the system `PATH` is special, configure the path where the deployment stack commands are located. You generally do not need to configure it. |

If you do not run commands from the msModeling root directory, set `PYTHONPATH`:

```bash
# Linux / macOS
export PYTHONPATH=/path/to/msmodeling:$PYTHONPATH

# Windows PowerShell
$env:PYTHONPATH = "C:\path\to\msmodeling;$env:PYTHONPATH"
```

The tool may need to read model configuration files from Hugging Face at runtime. If direct access is unavailable, set a mirror:

```bash
# Linux / macOS
export HF_ENDPOINT="https://hf-mirror.com"

# Windows PowerShell
$env:HF_ENDPOINT = "https://hf-mirror.com"
```

In a restricted network, even if you set `HF_ENDPOINT`, downloads may still fail because of proxy policies, DNS, TLS certificates, unreachable mirror sites, model repositories requiring authentication, or dependency libraries that do not use this environment variable. In this case, use a reviewed local model path.

## 3. Verifying the Installation

After the installation, run the following commands in an activated Python environment to verify that the CLI entry is available.

```bash
python -m cli.inference.text_generate --help
python -m cli.inference.throughput_optimizer --help
python -m serving_cast.main --help
msmodeling optix --help
pip show msmodeling
```

If the installation is successful, the preceding commands should output the usage instructions and parameter lists of `text_generate`, `throughput_optimizer`, `serving_cast`, and `msmodeling optix`, respectively, without reporting `ModuleNotFoundError`.

To run a basic simulation, you are advised to download and review the configuration files in the model repository in advance in an environment with external network access (only the `.json`, `.yaml`, `.yml`, and `.txt` suffixes are required), and then point `model_id` to a local absolute path:

```bash
python -m cli.inference.text_generate /data/models/Qwen3-32B --num-queries 2 --query-length 3500 --device TEST_DEVICE
```

If the command cannot run properly, verify that the current terminal has activated the Python environment where msModeling is installed.

## 4. Uninstalling

You can uninstall msModeling by running the following commands in the Python environment where it is installed.

If you use `uv` to manage the virtual environment, run:

```bash
uv pip uninstall msmodeling
```

If you installed msModeling with `pip + requirements.txt`, run:

```bash
pip uninstall msmodeling
```

> [!NOTE]
> Before uninstalling, verify that the current terminal uses the Python environment where msModeling is installed, to avoid uninstalling a package with the same name in another environment. If you manage the environment with `uv`, prefer `uv pip uninstall msmodeling`. If you no longer need the source directory, delete it manually after uninstalling.

## 5. Upgrading

Before upgrading, you can view the version information in the current environment:

```bash
# uv environment
uv pip show msmodeling

# pip environment
pip show msmodeling
```

### 5.1 Method 1: Overwrite Upgrade

Upgrading means uninstalling first and then installing. If you choose the overwrite upgrade, directly perform the installation as described in [2.1 Online Installation](#21-online-installation) and [2.2 Offline Installation](#22-offline-installation). The tool automatically uninstalls the old version and guides you through the overwrite installation.

### 5.2 Method 2: Source Upgrade

If you choose the source upgrade, complete the upgrade as follows:

1. Enter the msModeling repository root directory and pull the source code of the target version.

    ```bash
    cd msmodeling
    git fetch
    git checkout 26.1.0
    git pull
    ```

2. Select the corresponding environment and upgrade it to the target version.

    ```bash
    # uv environment
    uv pip install --upgrade -e .

    # uv environment with a temporary mirror source
    uv pip install --upgrade -e . -i https://mirrors.aliyun.com/pypi/simple

    # pip environment
    pip install --upgrade -e .
    ```

When upgrading versions, pay attention to the version compatibility relationships. See [Release Notes](https://gitcode.com/Ascend/release-management/blob/master/MindStudio/26.1.0/release_notes.md).

## 6. Appendix

### 6.1 OptiX and Simulation Environment Separation

If you use [OptiX for automatic serving optimization](../user_guide/optix_user_guide.md):

- msModeling and OptiX must be installed in an independent virtual environment, for example `.venv`. The installation brings in dependencies such as `torch` and `transformers`, which are used for simulation, not the deployment stack used for OptiX optimization.
- vLLM, MindIE, and benchmark tools use the environments already deployed in the system by default. Therefore, you generally do not need to create another deployment virtual environment.
- Do not run `pip install vllm` in the msModeling virtual environment.

OptiX child processes automatically strip the msModeling virtual environment and use the system `PATH`. Only when `PATH` is special can you configure `OPTIX_DEPLOY_PATH`. For details, see [OptiX User Guide - Recommended Practice: Environment and Deployment Stack](../user_guide/optix_user_guide.md#recommended-practice-environment-and-deployment-stack).

### 6.2 FAQ

- If `--help` cannot display help information, first troubleshoot the virtual environment, `PYTHONPATH`, and dependency installation.
- If the `cli` or `tensor_cast` module cannot be found, verify that the current directory is the repository root, or that `PYTHONPATH` is set correctly.
- If the model configuration download fails, verify that the network can access Hugging Face. If the `HF_ENDPOINT` mirror is still unavailable, use a local model path instead.
- If dependency installation fails, first verify that the virtual environment is activated. If you use `uv`, run `uv sync` again. If you use the pip method, upgrade `pip` and then run `pip install -r requirements.txt` and `pip install -e .` in sequence again, switching to a PyPI mirror when necessary.
