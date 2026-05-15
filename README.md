# MindStudio Modeling

MindStudio-Modeling is a performance simulation and analysis framework for neural network inference workloads, consisting of two core components for predicting and optimizing model performance on target hardware:

1. **TensorCast**
    * **Core Purpose**: A PyTorch program performance simulator, functioning as a "virtual machine."
    * **Main Function**: Intercepts a model's PyTorch computational graph and simulates its execution on a user-defined hardware profile (`DeviceProfile`) without requiring physical hardware.
    * **Supported Tasks**:
        * **Text Generation**: Simulates Large Language Model (LLM) inference (e.g., Qwen) via `cli.inference.text_generate`.
        * **Video Generation**: Simulates the forward pass of diffusion transformer models (e.g., Stable Video Diffusion-like architectures) via `cli.inference.video_generate`.
    * **Output**: Provides operator-level performance breakdown, memory footprint analysis, FLOPs analysis, and can generate Chrome Trace files for visualization.

2. **ServingCast**
    * **Core Purpose**: A suite of tools for system-level inference serving simulation and throughput optimization.
    * **Main Function**:
        * **Service Simulation**: Driven by `main.py`, it simulates end-to-end serving scenarios with multiple instances and requests based on YAML configuration files, outputting system-level metrics like throughput, latency (TTFT, TPOT).
        * **Throughput Optimization**: Via `cli.inference.throughput_optimizer.py`, it automatically searches for the optimal model configuration (parallelism strategy, batch size) to maximize token throughput under specified Service Level Objective (SLO) constraints (e.g., limits on TTFT, TPOT).

**Core Value**: It enables developers to predict model performance, identify bottlenecks, and optimize configurations for target hardware without needing access to the physical devices.

<!-- toc -->

- [MindStudio Modeling](#mindstudio-modeling)
  - [Installation](#installation)
    - [Environment Setup](#environment-setup)
  - [Getting Started](#getting-started)
  - [License](#license)

<!-- tocstop -->

## Installation

```bash
git clone https://gitcode.com/Ascend/msmodeling.git -b develop
cd msmodeling

# 1. install uv, Create a virtual environment (Python >= 3.10), take Python 3.13 as an example
pip install uv
uv venv --python 3.13 myenv

# 2. activate env
## Linux or MacOS
source myenv/bin/activate
## Windows
myenv/Scripts/activate # (or myenv\Scripts\activate)

# 3. install dependencies
uv pip install -r requirements.txt
```

Alternatively, if you already have a python environment which does not contain `torch_npu` or `cudatoolkit`, you can just run:

```bash
pip install -r requirements.txt
```

**Supported Python versions:** 3.10+

> [!Warning]
> If you are using Windows, note that PyTorch 2.10 may not run properly on your system. For a solution, please refer to [this issue](https://github.com/pytorch/pytorch/issues/166628). If you have not yet installed PyTorch, for optimal compatibility, we strongly recommend using version 2.8 or earlier to ensure the program functions correctly.

### Environment Setup

If you are not using the tools within the msmodeling directory, please set the `PYTHONPATH` before running:

```bash
export PYTHONPATH=/path/to/msmodeling:$PYTHONPATH
```

> [!Warning]
> When the tool is running, it will read the model configuration file from Hugging Face. Please ensure that your device can access [Hugging Face](https://huggingface.co/). Or you can set: `export HF_ENDPOINT="https://hf-mirror.com"`

## Getting Started

For detailed usage, please refer to the two documentation files:

* [For service simulation and throughput optimization.](./docs/en/serving_cast_instruct.md)

* [For TensorCast performance simulation framework.](./docs/en/tensor_cast_instruct.md)

## Contributions

### Coding style

Use `pre-commit` to make sure the coding style aligns:

```bash
pip install pre-commit
cd /path/to/msmodeling
pre-commit install  # run once
```

Later commit will be checked by `pre-commit` automatically.

### Unit tests

```bash
cd /path/to/msmodeling
```

Make sure unit tests pass by running: `bash ./tests/run_ut.sh tensor_cast` or `bash ./tests/run_ut.sh serving_cast`. Please ensure that the ut coverage rate of the newly added code is greater than `80%`

## License

msmodeling has a MulanPSL2-style license, as found in the [LICENSE](LICENSE) file.
