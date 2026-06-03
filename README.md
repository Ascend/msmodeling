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
        * **Throughput Optimization**: Via `cli.inference.throughput_optimizer.py`, it automatically searches for the optimal model configuration (parallelism strategy, batch size) to maximize token throughput under specified Service Level Objective (SLO) constraints (e.g., limits on TTFT, TPOT). Supports benchmarking multiple `--device` profiles with cross-hardware comparison tables and terminal ASCII sweep curves for single-device analysis.

**Core Value**: It enables developers to predict model performance, identify bottlenecks, and optimize configurations for target hardware without needing access to the physical devices.

<!-- toc -->

- [MindStudio Modeling](#mindstudio-modeling)
  - [Installation](#installation)
    - [Recommended: uv](#recommended-uv)
    - [Alternative: pip + requirements.txt](#alternative-pip--requirementstxt)
    - [Environment Setup](#environment-setup)
  - [Getting Started](#getting-started)
  - [License](#license)

<!-- tocstop -->

## Installation

**Supported Python versions:** 3.10+

> [!Warning]
> If you are using Windows, note that PyTorch 2.10 may not run properly on your system. For a solution, please refer to [this issue](https://github.com/pytorch/pytorch/issues/166628). If you have not yet installed PyTorch, for optimal compatibility, we strongly recommend using version 2.8 or earlier to ensure the program functions correctly.

### Recommended: uv

```bash
git clone https://gitcode.com/Ascend/msmodeling.git -b develop
cd msmodeling

pip install uv
uv venv --python 3.13 .venv

# Linux or macOS
source .venv/bin/activate
# Windows
# .venv\Scripts\activate

uv sync
# optional: uv sync --group lint / uv sync --group ci
```

Use `uv run …` or an activated venv to run commands. Scripts under `scripts/` auto-detect uv when `pyproject.toml` is present.

### Alternative: pip + requirements.txt

```bash
git clone https://gitcode.com/Ascend/msmodeling.git -b develop
cd msmodeling

python -m venv .venv
source .venv/bin/activate   # adjust for Windows

pip install "torch>=2.7,<=2.10" --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

See comments at the top of `requirements.txt` for the CPU PyTorch step.

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
uv sync --group lint
uv run pre-commit install   # run once
```

Later commits will be checked by `pre-commit` automatically.

### Unit tests

```bash
cd /path/to/msmodeling
uv sync
```

Make sure unit tests pass by running: `bash ./tests/run_ut.sh tensor_cast` or `bash ./tests/run_ut.sh serving_cast`. Please ensure that the ut coverage rate of the newly added code is greater than `80%`

## Suggestions & Community

We welcome everyone to contribute to the community. If you have any questions or suggestions, please submit an [Issue](https://gitcode.com/Ascend/msmodeling/issues) and we will respond as soon as possible. Thank you for your support.

| Technical Chat Groups | Official Account | More Ways to Connect |
| :---: | :---: | :--- |
| <img src="https://raw.gitcode.com/user-images/assets/8428112/368af17d-bd72-4bb6-ae94-f10fac88fd00/30be980e7fd65b2486d251b48a7999f3.jpg" width="120"><br><sub>*Scan to join the technical chat*</sub> | <img src="https://raw.gitcode.com/Ascend/msinsight/raw/master/docs/zh/user_guide/figures/readme/officialAccount.jpg" width="120"><br><sub>*Scan to follow for the latest updates*</sub> |Scan the codes to join our technical chat and follow our official account—the fastest way for MindStudio users and developers to connect:<br> **Quick Q&A:** Discuss technical issues instantly with community members<br>**Stay Updated:** Be the first to receive notifications on version releases and feature updates<br> **Knowledge Sharing:** Exchange best practices with fellow developers  <br>🛠️ **Other Channels**:<br> Ascend Assistant：[![WeChat](https://img.shields.io/badge/WeChat-07C160?style=flat-square&logo=wechat&logoColor=white)](https://gitcode.com/Ascend/msit/blob/master/docs/zh/figures/readme/xiaozhushou.png)<br> Ascend Forum：[![Website](https://img.shields.io/badge/Website-%231e37ff?style=flat-square&logo=RSS&logoColor=white)](https://www.hiascend.com/forum/) |

## Weekly Meeting

- MindStudio Modeling Weekly Meeting: [https://etherpad.ascend.osinfra.cn/p/sig-msit-modeling](https://etherpad.ascend.osinfra.cn/p/sig-msit-modeling)
- Wednesday, 10:00 - 12:00 (UTC+8, [Convert to your timezone](https://dateful.com/convert/gmt8?t=15))

## About the MindStudio Team

The Huawei MindStudio end-to-end development toolchain team is dedicated to providing comprehensive Ascend AI application development solutions, empowering developers to efficiently complete training development, inference development, and operator development.

## License

msmodeling has a MulanPSL2-style license, as found in the [LICENSE](LICENSE) file.
