# msModeling Installation Guide

<br>

## 1. Installation Instructions

Ensure that the runtime environment meets the following conditions:

- Python 3.10 or later is installed. An independent virtual environment is recommended.
- The network can access GitCode and Python package sources.
- If you need to pull Hugging Face model configurations, ensure that the network can access Hugging Face.

The tool supports the following installation methods: [online installation](#21-online-installation), [offline installation](#22-offline-installation), and [source installation](#23-source-installation).

## 2. Installation Methods

### 2.1 Online Installation

**Linux/macOS Environment**

If your device has internet access, you can automatically download and install the tool with one command. See the MindStudio [Download](https://www.hiascend.com/developer/software/mindstudio/download) page on the Ascend Community, select the product version, select the "Inference Development" scenario, select the msModeling tool, and then select "Online Installation" as the installation method. The system guides you through the remaining operations.

**Windows Environment**

Run the following commands in sequence to complete the installation (you can obtain the whl package link from the [release page](https://gitcode.com/Ascend/msmodeling/releases)):

```bash
# 1. Create and activate a virtual environment (a virtual environment is required to prevent damage to the execution environment)
python -m venv .venv
.venv\Scripts\activate
# 2. Install msmodeling (the following is an example link)
pip install https://gitcode.com/Ascend/msmodeling/releases/download/tag_MindStudio_26.2.0.B110_001/msmodeling-26.2.0-py3-none-any.whl
```

### 2.2 Offline Installation

For devices in an environment without external network access, such as an enterprise intranet, download the complete offline installation package on a machine with internet access first, and then transfer it to the target device for installation.

**Linux/macOS Environment**

See the MindStudio [Download](https://www.hiascend.com/developer/software/mindstudio/download) page on the Ascend Community, select the product version, select the "Inference Development" scenario, select the msModeling tool, and then select "Offline Installation" as the installation method to obtain the corresponding installation package and operation instructions.

**Windows Environment**

Run the following commands in sequence to complete the installation (you can obtain the whl package link from the [release page](https://gitcode.com/Ascend/msmodeling/releases)):

```bash
# 1. Create and activate a virtual environment (a virtual environment is required to prevent damage to the execution environment)
python -m venv .venv
.venv\Scripts\activate
# 2. Install msmodeling (the following is an example file name)
pip install msmodeling-26.2.0-py3-none-any.whl
```

### 2.3 Source Installation

If you want to use the features of the latest code or modify the source code to enhance features, download the code in this repository, compile and package the tool yourself, and complete the installation.

#### 2.3.1 Preparing the Environment

Source code compilation uses the standard MindStudio build environment in all cases.

- For daily development or when you use a released image, see the [MindStudio Tool Development Environment Setup Guide](https://gitcode.com/Ascend/msot/blob/master/docs/en/common/dev_env_setup.md).
- When you need to reproduce the environment from a base OS, perform source build verification, or run unit test verification, you must refer to the [MindStudio Unified Build Image Creation Guide](https://gitcode.com/Ascend/msot/blob/master/docs/en/common/docker_image_build_guide.md) and build the environment image on site from the openEuler base image.

All subsequent source compilation and unit test commands in this document run in the specified image container or in a container of the environment image built on site. The CANN software package version, GCC version, and Python version follow the Unified Build Image Creation Guide, and this repository does not maintain them separately.

After the image is built, you must use the `ctr_in.py` command provided in Chapter 7 of the Unified Build Image Creation Guide to start and enter the container in an interactive terminal. Do not use a plain `docker run` to create a container, and do not use `docker exec <container_name> bash -c '<command>'` instead of an interactive environment. Otherwise, Python, GCC, and CANN environment initialization may be skipped.

After you enter the interactive container shell opened by `ctr_in.py`, run the following command to clone this repository:

```bash
cd ~
git clone https://gitcode.com/Ascend/msmodeling.git
```

#### 2.3.2 Compiling the Source Code

In the same interactive container shell opened by `ctr_in.py`, run the following command in the repository root directory to automatically download dependencies and build:

```bash
cd ~/msmodeling
python3 build.py
```

> [!CAUTION]
>
> Running `python3 build.py` automatically triggers `uv sync --frozen --group build`, which installs all dependencies (including `torch`, `transformers`, and so on) according to `pyproject.toml`/`uv.lock`. Do **not** manually preinstall any dependency by referring to `requirements.txt` or other configuration files. If you encounter dependency issues, still troubleshoot them according to `pyproject.toml` and `uv.lock`.

After the build succeeds, the installation package is generated in the `artifacts/` directory.

#### 2.3.3 Running Unit Tests (Optional)

This step is not required for installation. If you need to verify the basic functions of the code, you can run unit tests:

```bash
cd ~/msmodeling
python3 build.py test
```

If the command returns code 0 and no test case fails, the unit tests pass.

#### 2.3.4 Installing the Tool

Copy the whl package to the runtime environment (no copy is required for a local installation), and perform the following installation operations:

```bash
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate

pip install msmodeling-xxxxx.whl
```

## 3. Verifying the Installation

After the installation, run the following command in an activated Python environment:

```bash
msmodeling --help
```

If the command runs without errors and displays help information, the installation is successful.

## 4. Uninstalling

### 4.1 Installation Through the Ascend Community

1. Download the script.

   ```bash
   curl -O https://inst.obs.cn-north-4.myhuaweicloud.com/26.2.0/ms_install.py
   ```

   > [!NOTE]
   >
   > - You need a networked environment to download the script. If the environment does not allow network access or is offline, download the script in a networked environment first and then copy it to the target device.
   > - If the command does not respond or problems such as connection failures or SSL certificate errors occur, see the [FAQ](https://www.hiascend.com/developer/blog/details/02176213671719317003).

2. Perform the uninstallation.

   ```bash
   python ms_install.py uninstall msmodeling
   ```
   
   If the uninstallation is successful, the following information is printed:

   ```text
   Successfully uninstalled 1 tool (msmodeling)
   ```

### 4.2 Installation Through the `pip install` Command

Activate the Python virtual environment (venv) where msmodeling is installed, and run the following command:

```bash
pip uninstall msmodeling
```

> [!NOTE]
> Before uninstalling, verify that the current terminal environment is the Python environment where msModeling is installed, to avoid uninstalling a package with the same name in another environment.

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
    git checkout 26.2.0
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

When upgrading versions, pay attention to the version compatibility relationships. See [Release Notes](https://gitcode.com/Ascend/release-management/blob/master/MindStudio/26.2.0/release_notes.md).

## 6. Appendix

### 6.1 Serving Optimization Through Real-World Testing and Simulation Environment Separation

If you use [serving optimization through real-world testing with OptiX](../user_guide/msmodeling_optix_user_guide.md):

- msModeling and OptiX must be installed in an independent virtual environment, for example `.venv`. The installation brings in dependencies such as `torch` and `transformers`, which are used for simulation, not the deployment stack used for OptiX optimization.
- vLLM, MindIE, and benchmark tools use the environments already deployed in the system by default. Therefore, you generally do not need to create another deployment virtual environment.
- Do not run `pip install vllm` in the msModeling virtual environment.

OptiX child processes automatically strip the msModeling virtual environment and use the system `PATH`. Only when `PATH` is special can you configure `OPTIX_DEPLOY_PATH`. For details, see [Recommended Practice: Environment and Deployment Stack](msmodeling_optix_env_and_deployment_stack.md).

### 6.2 FAQ

- If `--help` cannot display help information, first troubleshoot the virtual environment, `PYTHONPATH`, and dependency installation.
- If the `cli` or `tensor_cast` module cannot be found, verify that the current directory is the repository root, or that `PYTHONPATH` is set correctly.
- If the model configuration download fails, verify that the network can access Hugging Face. If the `HF_ENDPOINT` mirror is still unavailable, use a local model path instead.
- If dependency installation fails, first verify that the virtual environment is activated. If you use `uv`, run `uv sync` again. If you use the pip method, upgrade `pip` and then run `pip install -r requirements.txt` and `pip install -e .` in sequence again, switching to a PyPI mirror when necessary.
