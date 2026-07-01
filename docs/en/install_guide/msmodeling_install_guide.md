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

The project recommends `uv` for the virtual environment and dependencies. From the repository root, `uv sync` **creates** `.venv` if needed, installs this project in editable mode (including the `msmodeling` CLI), and syncs dependencies from `uv.lock`. You do **not** need a separate `uv venv` or `pip install -e .`. Scripts under `scripts/` also detect and use `uv` automatically.

```bash
pip install uv
cd msmodeling
uv sync

# Optional: pin Python version (defaults to an available interpreter on the machine)
# UV_PYTHON=3.13 uv sync

# Optional: install lint or CI dependency groups
uv sync --group lint
uv sync --group ci
```

After setup, use `uv run ...` (recommended), or activate the auto-created virtual environment:

```bash
# Linux / macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 3. Alternative: pip + requirements.txt

If you do not use `uv`, install dependencies with Python's built-in virtual environment and `requirements.txt`. In CPU environments, install `torch` and `torchvision` from the PyTorch CPU index before installing the remaining dependencies.

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

## OptiX and Simulation Environment Isolation<a name="optix-and-simulation-environment-isolation"></a>

If you use [OptiX service parameter optimizer](../user_guide/optix_user_guide.md):

- Install msmodeling and OptiX in a uv venv from this guide, for example `.venv`: run `uv sync`. The install brings in `torch`, `transformers`, and similar packages for simulation, not for OptiX on-device runs. Installing into system Python can break vLLM and MindIE on the same machine.
- vLLM, MindIE, and benchmarks default to the system deploy environment. You usually do not need a separate deploy venv.
- Do not `pip install vllm` in the msmodeling venv.

OptiX strips the msmodeling venv for child processes and uses system PATH; set `OPTIX_DEPLOY_PATH` only when PATH is non-standard. See [OptiX User Guide · Environment and Deploy Stack](../user_guide/optix_user_guide.md#recommended-practice-environment-and-deploy-stack).

## First Simulation

Prefer downloading and reviewing the model repository configuration files in an environment with internet access first. Only files ending with `.json`, `.yaml`, `.yml`, or `.txt` are required. Then point `model_id` to the local absolute path:

   ```bash
   python -m cli.inference.text_generate /data/models/Qwen3-32B --num-queries 2 --query-length 3500 --device TEST_DEVICE
   ```

## Verify the Installation

After installing dependencies, run the following commands from the **msModeling repository root** to verify CLI entry points (`uv run` does not require manual activation):

```bash
uv run python -m cli.inference.text_generate --help
uv run python -m cli.inference.throughput_optimizer --help
uv run python -m serving_cast.main --help
uv run msmodeling optix --help
```

If the installation is correct, the commands above should print usage and argument lists for `text_generate`, `throughput_optimizer`, and `serving_cast` respectively, without `ModuleNotFoundError`.

Troubleshooting:

- If `--help` cannot display help, check the virtual environment, `PYTHONPATH`, and dependency installation according to the "Verify the Installation" section above.
- If `cli` or `tensor_cast` cannot be found, confirm that the current directory is the repository root or that `PYTHONPATH` is configured correctly.
- If model configuration download fails, confirm that the network can access Hugging Face. If the `HF_ENDPOINT` mirror is still unavailable, use a local model path.
- If dependency installation fails, first confirm that the virtual environment is activated and rerun `uv sync`. If you use pip, upgrade `pip` and rerun `pip install -r requirements.txt`; switch PyPI mirrors if needed.
