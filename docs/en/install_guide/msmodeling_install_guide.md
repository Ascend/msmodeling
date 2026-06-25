# Quick Start: Environment Setup and First Simulation

## Goal

After reading this guide, you will be able to set up the development environment, run your first LLM inference simulation, and understand the basic flow from command line invocation to result output.

## Environment Setup

### 1. Clone the Repository

```bash
git clone https://gitcode.com/Ascend/msmodeling.git
cd msmodeling
```

### 2. Create a Virtual Environment (Python 3.10+ Recommended)

```bash
pip install uv
uv venv --python 3.13 myenv

# Linux / macOS
source myenv/bin/activate

# Windows
myenv\Scripts\activate
```

### 3. Install Dependencies

```bash
uv pip install -r requirements.txt
```

> [!NOTE]
> If you already have a Python environment, you can also run `pip install -r requirements.txt` directly.

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

The tool reads model configuration files from Hugging Face. If direct access is unavailable, set a mirror endpoint:

```bash
export HF_ENDPOINT="https://hf-mirror.com"
```

## First Simulation

After the environment is ready, run a minimal LLM inference simulation:

```bash
python -m cli.inference.text_generate Qwen/Qwen3-32B --num-queries 2 --query-length 3500 --device TEST_DEVICE
```

The command prints operator-level performance summary, total execution time, TPS/Device, and estimated memory usage. For more details, see [TensorCast User Guide](../user_guide/msmodeling_tensor_cast_user_guide.md).
