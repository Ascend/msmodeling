# Measured Serving Optimization User Guide

## Overview

**Measured serving optimization** (`msmodeling optix`) is a feature for measured optimization of serving parameters based on the Particle Swarm Optimization (PSO) algorithm. It supports automatic search on real serving frameworks such as vLLM and MindIE to obtain the best throughput parameter combination that meets latency requirements.

## Intended Audience and Reading Path

This document is intended for performance engineers and deployment engineers who need to automatically optimize vLLM and MindIE serving deployment parameters. You are advised to read it in the following order:

1. Read "Preparations" and "Tool Installation" first to confirm that the serving frameworks and benchmark tools can run properly.
2. Then read "Quick Start" and "Command Parameters" to complete one default optimization.
3. Finally, read "Configuration File Description" and "Output File Description" to adjust the search space based on service SLOs and filter recommended parameters.

The tool mainly consists of two core functional modules:

- **Parameter optimization module**: uses the PSO particle swarm optimization algorithm to automatically generate serving parameter combinations and continuously approach the optimal solution. In addition, the Early Rejection algorithm performs early evaluation of serving parameters based on theoretical modeling, tuning experience, and partial measured data.

- **Parameter verification module**: automatically starts the serving process and the benchmark tool process for parameter testing to obtain performance results. Currently, the supported benchmark tools include `AISBench` and `vllm_benchmark`.

> [!NOTE]
>
> The legacy benchmark tool will gradually be replaced by AISBench. You are advised to use AISBench first. If the current environment still retains the `vllm_benchmark` adaptation capability, you can configure it as described in the corresponding sections of this document.

Based on the preceding functional modules, measured serving optimization can automatically recommend serving parameter combinations with better throughput.

The tool has been validated on LLaMA3-8B and Qwen3-8B. In principle, it does not limit the supported model types, and broader validation coverage is planned for future releases.

**Basic Concepts**

- vLLM and MindIE: serving frameworks that support serving deployment of models.
- `vllm_benchmark` and `AISBench`: inference performance benchmark tools that support inference performance evaluation of serving.

## Supported Products<a name="ZH-CN_TOPIC_0000002479925980"></a>

> [!NOTE]
>
> For details about the specific models of Ascend products, see [Ascend Product Forms](https://www.hiascend.com/document/detail/en/AscendFAQ/ProduTech/productform/hardwaredesc_0001.html).

|Product Type| Supported (Yes/No)|
|--|:----:|
|Atlas 350 accelerator card|No|
|Atlas A3 training products/Atlas A3 inference products|Yes|
|Atlas A2 training products/Atlas A2 inference products|Yes|
|Atlas 200I/500 A2 inference products|Yes|
|Atlas inference products|Yes|
|Atlas training products|No|

> [!NOTE]
>
> For Atlas A2 training products/Atlas A2 inference products, only the Atlas 800I A2 inference server in this product series is currently supported.
> For Atlas inference products, only the Atlas 300I Duo inference card and Atlas 800 inference server (model 3000) in this product series are currently supported.

## Preparations

**Environment and Deployment Stack**

| Layer | Recommended Practice | Description |
|------|----------|------|
| **msModeling/OptiX** | **Must** be installed in a **uv virtual environment** | The installation brings along `torch`, `transformers`, and so on, which are used for TensorCast simulation rather than OptiX optimization. Installing them into the system Python will overwrite the packages of the same names in the deployment stack. |
| **vLLM/MindIE/benchmark tools** | **Use the system environment by default** | It is assumed that the serving and benchmark tools have been deployed on the machine as described in the official documentation. Generally, you do not need to create another deployment venv. |

When OptiX starts a serving or benchmark subprocess, it removes traces of the msModeling virtual environment from `PATH` and `PYTHONPATH`, and then uses the system PATH to find commands such as `vllm`, `mindieservice_daemon`, and `ais_bench`. You do not need to manually modify the subprocess environment variables or create a separate venv for the deployment stack.

If a command is not in the default PATH, or multiple runtimes are installed on the machine, you can specify the deployment root directory through `OPTIX_DEPLOY_PATH` or `[deploy] path_prefix` in `config.toml`.

For the complete steps, see [Recommended Practice: Environment and Deployment Stack](#recommended-practice-environment-and-deployment-stack). For common installation on the simulation side, see [Environment Setup Guide](../install_guide/msmodeling_install_guide.md#61-optix-and-simulation-environment-separation).

**Deployment Stack Preparation**

In the system environment, or in the path specified by `[deploy]`, confirm that the serving and benchmark tools can run properly. You can refer to [vLLM Server](https://docs.vllm.ai/projects/ascend/en/latest/quick_start.html), [MindIE Service](https://gitcode.com/Ascend/MindIE-Motor/blob/v3.1.0/docs/en/user_guide/quick_start_motor.md), and [AISBench Benchmark Tool Deployment](https://gitee.com/aisbench/benchmark/blob/master/README.md).

## Tool Installation

> [!IMPORTANT]
> **OptiX must be installed in a virtual environment.** Run `uv sync` in the repository root directory. It automatically creates `.venv` and completes the installation.
>
> Installing msModeling also installs packages such as `torch` and `transformers`. These dependencies are used for TensorCast simulation, and real-device optimization does not rely on them. If you install msModeling in the system Python, the existing versions of `torch` and `transformers` in the system are often changed. As a result:
>
> - vLLM and MindIE fail to start or report inference errors
> - The versions do not match those verified on the Ascend inference stack
> - Other deployment tools on the same machine also break
>
> Install msModeling only in the uv virtual environment. Continue to use the existing vLLM, MindIE, and benchmark tools in the system.

The optimization tool is integrated in the msModeling repository root directory. Install it as follows:

```bash
git clone https://gitcode.com/Ascend/msmodeling.git   # Skip this if the repository is already cloned
cd msmodeling
uv sync
```

> [!NOTE]
> `uv sync` automatically creates `.venv` and installs msModeling in editable mode (including the `msmodeling optix` CLI). You do not need to run `uv venv` or `pip install -e .`. If the current branch does not contain the OptiX source directory, switch to a release branch that contains the OptiX code or use the corresponding release package. Copying only the documentation files cannot provide the `msmodeling optix` command.
> [!WARNING]
> Do not install deployment packages such as `vllm` and `mindie_llm` in the msModeling virtual environment. Do not install msModeling in the system Python without creating a venv. For details, see [Recommended Practice: Environment and Deployment Stack](#recommended-practice-environment-and-deployment-stack).

## Recommended Practice: Environment and Deployment Stack

In a typical scenario, vLLM or MindIE is already deployed in the system, and msModeling is installed separately in a uv virtual environment.

**1. Install msModeling (`uv`)**

```bash
cd /path/to/msmodeling
uv sync
```

To verify: run `uv run msmodeling optix --help`.

**2. Confirm that the system deployment stack is available**

You can first run `deactivate` to exit the msModeling venv, and then check the commands in the system. For example:

```bash
which vllm
vllm --help
```

`which vllm` should resolve to a system path, for example `/usr/local/bin/vllm`, rather than the `.venv/bin/vllm` of msModeling.

For MindIE scenarios, confirm that `mindieservice_daemon` is available, or that the installation pointed to by `MIES_INSTALL_PATH` is correct.

**3. Optional: Specify the deployment root directory**

Configure this only when the system PATH cannot find the correct command:

```bash
export OPTIX_DEPLOY_PATH=/path/to/custom-deploy-root
```

You can also write it into `optix/config.toml`. For the field descriptions, see `[deploy]` in [Configuration File Description](#configuration-file-description):

```toml
[deploy]
path_prefix = "/path/to/custom-deploy-root"
```

**4. Run OptiX in the msModeling venv**

```bash
source /path/to/msmodeling/.venv/bin/activate
msmodeling optix -e vllm -b ais_bench
```

If you set `OPTIX_DEPLOY_PATH` in the preceding step, keep the export.

**5. Confirm the logs**

If the startup log contains information such as `[optix/env] ... deploy command vllm → /usr/local/bin/vllm`, the subprocess uses the system deployment stack instead of the msModeling venv.

> [!NOTE]
> By default, you do not need to create a dedicated deployment venv. Only when the PATH layout is special do you need `OPTIX_DEPLOY_PATH` or `[deploy] path_prefix`.

## Tool Uninstallation

Uninstall it in the msModeling virtual environment:

```bash
pip uninstall msmodeling
```

## Quick Start

0. **Confirm that the path returned by `which vllm` is not in the msModeling venv**. Example for the vLLM scenario:

    ```bash
    source /path/to/msmodeling/.venv/bin/activate
    which vllm
    ```

    Correct example: `/usr/local/bin/vllm`. Incorrect example: `/path/to/msmodeling/.venv/bin/vllm`. See [Environment Variables and Troubleshooting](#environment-variables-and-troubleshooting).

1. Complete the requirements described in [Preparations](#preparations) and [Recommended Practice: Environment and Deployment Stack](#recommended-practice-environment-and-deployment-stack).

2. Modify the configuration file. Before starting optimization, configure `config.toml` according to your environment, including optimization parameters, benchmark tool parameters, and serving parameters. See [Configuration File Description](#configuration-file-description) for details. You can also use the `-c` parameter to place the configuration file in any path. For details, see [Command Parameters](#command-parameters).

3. Start optimization. After completing the preceding steps, run the following command to start automatic optimization with one click:

    ```bash
    msmodeling optix
    ```

    By default, the tool optimizes vLLM serving parameters with `AISBench` as the benchmark tool.

4. View the results. The optimization time is determined by the model size and dataset size, typically 4 to 8 hours. After completion, the `data_storage_*.csv` file is generated and saved in the `result/store` subdirectory of the current directory. It records the performance of each parameter set. For details, see [Output File Description](#output-file-description).

## Command Parameters

**Function**

The tool combines the parameter verification and parameter optimization modules to provide reliable recommended values for serving parameters through real-device testing.

**Precautions**

- Before starting optimization, confirm that `vllm` or `mindie` and `ais_bench` or `vllm_benchmark` can run in the system deployment environment and are not installed in the msModeling virtual environment.
- The model path, port, dataset path, and service startup parameters in `config.toml` must be consistent with the actual deployment environment.
- Automatic optimization repeatedly starts services and runs benchmarks, which usually takes a long time. You are advised to run it in a dedicated or resource-stable environment.
- When an environment isolation exception occurs, the log prefix `[optix/env]` provides the cause and remediation suggestions. For details, see [Environment Variables and Troubleshooting](#environment-variables-and-troubleshooting).

**Syntax**

```bash
msmodeling optix [options]
```

**Parameters**

|Parameter|Mandatory (Yes/No)|Description|
|---|---|---|
|-`lb` or `--load_breakpoint`|No|Controls whether to resume the optimization process from a breakpoint. Including this parameter enables the feature; omitting it disables the feature.|
|`--backup`|No|Determines whether to back up data during optimization. Including this parameter enables backup. The options are as follows:<br>&#8226;`True`: enables backup<br>&#8226;`False`: disables backup.<br/>The default value is `False`.|
|-`b` or `--benchmark_policy`|No|Specifies a benchmark tool. The options are as follows:<br>&#8226;`vllm_benchmark`: uses `vllm_benchmark` as the test tool <br/>&#8226;`ais_bench`: uses AISBench as the test tool<br/>The default value is `ais_bench`.<br/>You need to select the inference framework and the test framework that are compatible with each other.|
|-`e` or `--engine`|No|Specifies an inference framework. The options are as follows:<br>&#8226;`vllm`: uses vLLM as the inference framework<br>&#8226;`mindie`: uses MindIE as the inference framework<br/>The default value is `vllm`.|
|-`c` or `--config`|No|Specifies a custom configuration file path (TOML format). The following three forms are supported:<br>&#8226;Absolute path: uses the specified path directly.<br>&#8226;Relative path (with directory separators): resolves the path relative to the current working directory.<br>&#8226;Filename only: searches in the current working directory.<br/>If not specified, the tool automatically searches for the configuration file in the preset path order.<br/>The specified file must be in valid TOML format and has the highest configuration priority.|

**Example (vLLM Serving Parameter Optimization)**

1. Modify the configuration file. Before starting optimization, configure `config.toml` according to your environment, including optimization parameters, benchmark tool parameters, and serving parameters. See [Configuration File Description](#configuration-file-description) for details.

2. To set environment variables for the vllm/mindie services, simply set them before running the tool. For example:

    ```bash
    export ASCEND_RT_VISIBLE_DEVICES=0
    ```

    The tool automatically applies these environment variables during optimization.

3. After the prerequisites are ready, run the following command to start automatic optimization with one click:

    ```bash
    msmodeling optix -e vllm
    ```

    To use `vllm_benchmark` as the benchmark tool in vLLM scenarios, see:

    ```bash
    msmodeling optix -e vllm -b vllm_benchmark
    ```

**Example (MindIE Serving Parameter Optimization)**

1. Modify the configuration file. Before starting optimization, configure `config.toml` according to your environment, including optimization parameters, benchmark tool parameters, and serving parameters. See [Configuration File Description](#configuration-file-description) for details.
2. To set environment variables for the vllm/mindie services, simply set them before running the tool. For example:

    ```bash
    export ASCEND_RT_VISIBLE_DEVICES=0
    ```

    The tool automatically applies these environment variables during optimization.

3. After the prerequisites are ready, run the following command to start automatic optimization with one click:

    ```bash
    msmodeling optix -e mindie
    ```

**Example (Specifying a Custom Configuration File)**

If the configuration file is not in the default search path, you can explicitly specify it with the `-c` parameter:

```bash
# Absolute path
msmodeling optix -c /data/configs/my_config.toml

# Filename in the current directory
msmodeling optix -c my_config.toml

# Relative path
msmodeling optix -e vllm -b vllm_benchmark -c ../configs/vllm_config.toml
```

The specified configuration file has the highest priority and overrides the configuration items with the same names in the default path.

**Output Description**

After automatic optimization is complete, a result file in CSV format is generated and stored in the `result/store` folder in the current directory. For details, see [Output File Description](#output-file-description).

## Output File Description

Each row in the output CSV corresponds to a parameter set, and the first four columns are performance metrics. You can filter the performance rows that meet your requirements and change the vLLM/MindIE parameters and the vllm_benchmark/AISBench parameters to the values in the CSV.

| Field | Description |
| --- | --- |
| generate_speed | Throughput. |
| time_to_first_token | TTFT latency, in seconds. |
| time_per_output_token | TPOT latency, in seconds. |
| success_rate | Success rate of requests returned by the test. |
| throughput | Test throughput, in requests per second. |
| CONCURRENCY | Concurrency. |
| REQUESTRATE | Send rate. |
| error | Records the reason why this parameter set did not execute properly, recorded when a sending error occurs. |
| backup | Data recording address, recorded when `--backup` is enabled. |
| real_evaluation | Marks whether the data is obtained from real test results. `false` indicates that this set of data is predicted by the gp model. |
| fitness | Optimization value of the optimization algorithm. A smaller value indicates a better parameter set. |
| num_prompts | Number of requests sent by the benchmark tool during this optimization. |

The remaining columns are the corresponding `config.toml` parameters of vLLM or MindIE.

## Appendixes

### Configuration File Description

**Deployment Environment `[deploy]`**

When starting vLLM, MindIE, or benchmark tool subprocesses, OptiX first removes the variables related to the msModeling virtual environment, and then finds the deployment root directory according to the following configuration. The `bin/` directory should contain `vllm`, `ais_bench`, and so on:

| Parameter | Mandatory | Description |
|------|------|------|
| path_prefix | No | Deployment root directory, used to override the default system PATH. If not set, the system PATH is used directly after stripping the msModeling venv, which has the same effect as the directory-level `OPTIX_DEPLOY_PATH`. |

Consistent with the comments in `optix/config.toml`:

```toml
# Enable this only when the system PATH cannot find vllm, ais_bench, and so on
[deploy]
# path_prefix = "/path/to/custom-deploy-root"
```

**Optimization Parameters**: `n_particles` (number of optimization seeds), `iters` (number of iteration rounds), and `tpot_slo` (latency constraint of `time_per_output_token`), and so on.
You can configure the number of seeds and iterations based on the estimated time. The time for a single seed is the time for starting the service plus testing the data. For example, if starting the service and completing the test takes 9 to 10 minutes, and you are willing to spend 8 hours on optimization, you can run about 50 seeds in total. You are advised to configure 5 × 10. Set the number of seeds to 10 and the number of iterations to 5. You are advised to set the number of seeds to about twice the number of iterations.

> **Note**: All of the following optimization parameters are mandatory and must not be deleted or omitted. Otherwise, an error is reported during running.

|Parameter|Mandatory (Yes/No)|Description|
|---|---|---|
|n_particles|Yes|Number of optimization seeds, that is, the number of parameter combinations generated in one group. Value range: an integer from 1 to 1000. You are advised to set it to 15 to 30. |
|iters|Yes|Number of iteration rounds. Value range: an integer from 1 to 1000. You are advised to set it to 5 to 10. |
|ttft_penalty|Yes|Penalty coefficient for `time_to_first_token`, that is, the first-token latency timeout penalty coefficient. Set it to 0 if there is no latency requirement for `time_to_first_token`. Value range: [0, 100]. You are advised to set it to 1.|
|tpot_penalty|Yes|Penalty coefficient for `time_per_output_token`, that is, the non-first-token latency timeout penalty coefficient. Set it to 0 if there is no latency requirement for `time_per_output_token`. Value range: [0, 100]. You are advised to set it to 1.|
|success_rate_penalty|Yes|Penalty coefficient for the request success rate. Value range: an integer from 1 to 1000. You are advised to set it to 5. |
|ttft_slo|Yes|Latency constraint of `time_to_first_token`. For example, if `time_to_first_token` is limited to 2 seconds, set the value to 2. Value range: (0, 100], in seconds.|
|tpot_slo|Yes|Latency constraint of `time_per_output_token`. For example, if `time_per_output_token` is limited to 50 ms, set the value to 0.05. Value range: (0, 100], in seconds. |
|service|Yes|Marks whether the node is the primary node or the secondary node in multi-node startup. In multi-node scenarios, set the secondary node to `slave`. The options are as follows:<br>&#8226;`master`: primary node<br/>&#8226;`slave`: secondary node,<br/>The default value is `master`.|
|sample_size|No|Sampling size of the original dataset. Using the sampled data for tuning can improve optimization efficiency. Value range: an integer from 1000 to 10000. You are advised to set it to 1/3 of the requests in the original dataset.|

**Benchmark Tool Parameters**:
If `AISBench` is used for the test, modify the following parameters. You can modify them by referring to the [AISBench Usage Description](https://gitee.com/aisbench/benchmark/blob/master/README.md).

|Parameter|Description|
|---|---|
|models| Specifies a model task. You can configure it according to the [Model Configuration Description](https://gitee.com/aisbench/benchmark/blob/master/doc/users_guide/models.md).|
|datasets| Specifies a dataset task. You can configure it according to the [Dataset Preparation Guide](https://gitee.com/aisbench/benchmark/blob/master/doc/users_guide/datasets.md).|
|mode| Operation mode. You can configure it according to the [Operation Mode Description](https://gitee.com/aisbench/benchmark/blob/master/doc/users_guide/mode.md).|
|num_prompts| Controls the number of dataset items to run. This parameter is valid when `mode` is set to `perf`.|

If `vllm_benchmark` is used for the test, modify the following parameters:

|Parameter|Mandatory (Yes/No)|Description|
|---|---|---|
|host|Yes| host IP address, which must be consistent with the `host` in `[vllm.command]`. You can set it to `127.0.0.1`.|
|port|Yes| Port number, which must be consistent with the `port` in `[vllm.command]`.|
|model|Yes| Model path, which must be consistent with the `model` in `[vllm.command]`.|
|served_model_name|Yes| Model name, which must be consistent with the `served_model_name` in `[vllm.command]`.|
|dataset_name|Yes| Dataset name.|
|dataset_path|Yes| Dataset path.|
|num_prompts|Yes| Controls the number of dataset items to run.<br>Value range: an integer from 1 to 10000.|
|others|No| Concatenates other parameters. Note that parameters are separated by spaces, and no space is allowed within a parameter. For example: `--ignore-eos --custom-output-len 1500`. The default value is empty.|

**vLLM Serving Parameters**:
When the vLLM framework is used, modify the `[vllm.command]` parameters in `config.toml`. For example:

```toml
[vllm.command]
host = "127.0.0.1"
port = "8000"
model = "/workspace/vllm/models/llama-2-7b-chat-hf"
served_model_name = "llama-2-7b-chat-hf"
others = ""
```

|Parameter|Mandatory (Yes/No)|Description|
|---|---|---|
|host|Yes| host IP address, which must be consistent with the `host` in `[vllm_benchmark.command]`. You can set it to `127.0.0.1`.|
|port|Yes| Port number, which must be consistent with the `port` in `[vllm_benchmark.command]`.|
|model|Yes| Model path, which must be consistent with the `model` in `[vllm_benchmark.command]`.|
|served_model_name|Yes| Model name, which must be consistent with the `served_model_name` in `[vllm_benchmark.command]`.|
|others|No| Concatenates other parameters. Note that parameters are separated by spaces, and no space is allowed within a parameter. For example: `--tensor-parallel-size 2 --no-enable-prefix-caching`. The default value is empty.|

### vLLM Custom Parameter Optimization

The optimization tool supports adding vLLM parameters to the optimization through `[[vllm.target_field]]`. Depending on how the parameter takes effect, there are two configuration types:

- vLLM environment variables: You only need to declare them in `[[vllm.target_field]]` with `config_position = "env"`. The tool automatically writes the uppercase environment variable with the same name before starting the service in each optimization round. You do not need to write it into `others` in `[vllm.command]`.
- vLLM CLI parameters: Declare them in `[[vllm.target_field]]` first, and then reference the variables in `others` of `[vllm.command]` to splice them into the startup command.

> **Variable reference rule**: In `others`, use the format `$UPPERCASE_FIELD_NAME` to reference an optimization field. The tool automatically replaces it with the actual value of the current iteration during running.

#### Example 1: vLLM Environment Variable Optimization

If the parameter to be optimized is itself a vLLM environment variable, you only need to add it to `[[vllm.target_field]]`. For example:

```toml
[[vllm.target_field]]
name = "VLLM_WORKER_MULTIPROC_METHOD"
config_position = "env"
dtype = "enum"
dtype_param = ["fork", "spawn"]
value = "fork"
```

Such parameters do not need to be referenced in `others` of `[vllm.command]`. You can keep `others = ""` or fill in only other CLI parameters.

#### Example 2: Enumerated Numeric Command-Line Parameter (Using `gpu_memory_utilization` as an Example)

**Step 1**: Declare the optimization field.

```toml
[[vllm.target_field]]
name = "GPU_MEMORY_UTILIZATION"
config_position = "env"
dtype = "enum"
dtype_param = [0.9, 0.91, 0.92]
value = 0.9
```

**Step 2**: Reference the variable in `others` of `[vllm.command]`.

```toml
[vllm.command]
# ... other mandatory parameters ...
others = "--gpu-memory-utilization $GPU_MEMORY_UTILIZATION"
```

#### Example 3: Switch/Composite String Command-Line Parameter (Using Compilation Configuration `--compilation-config` as an Example)

When the parameter itself is a complete CLI string, you can use the "not enabled" (empty string `""`) and "enabled" forms as enum candidate values. When the tool encounters an empty string, it automatically skips it and does not append anything to the startup command.

**Step 1**: Declare the optimization field.

> **Note**: TOML strings use double quotation marks `"` as delimiters. If the string content contains double quotation marks, escape them with `\"`. Otherwise, a parsing error occurs.

```toml
[[vllm.target_field]]
name = "COMPILATION_CONFIG"
config_position = "env"
dtype = "enum"
dtype_param = ["", "--compilation-config '{\"cudagraph_mode\": \"FULL_DECODE_ONLY\"}'"]
value = "--compilation-config '{\"cudagraph_mode\": \"FULL_DECODE_ONLY\"}'"
```

**Step 2**: Reference the variable in `others` of `[vllm.command]`.

```toml
[vllm.command]
# ... other mandatory parameters ...
others = "$COMPILATION_CONFIG"
```

**MindIE serving parameters**: You can modify them by referring to the [MindIE Server Configuration Parameter Description](https://www.hiascend.com/document/detail/zh/mindie/20RC1/mindieservice/servicedev/mindie_service0285.html).
Serving parameters can directly specify the range of a parameter. For example, to configure the optimization search space of the serving parameter `max_batch_size` as 10 to 400, you can set:

```toml
[[mindie.target_field]]
name = "max_batch_size" # Serving parameter name
config_position = "BackendConfig.ScheduleConfig.maxBatchSize" # Position of the serving parameter in the MindIE Server
min = 10 # Minimum value
max = 400 # Maximum value
dtype = "int" # Data type
```

In addition, you can also set a parameter to be related to another parameter. For example, `max_prefill_batch_size` is related to `max_batch_size`, that is, `max_prefill_batch_size = ratio * max_batch_size` (0 < `ratio` < 1). You can set:

```toml
[[mindie.target_field]]
name = "max_prefill_batch_size"
config_position = "BackendConfig.ScheduleConfig.maxPrefillBatchSize"
min = 0
max = 1
dtype = "ratio"
dtype_param = "max_batch_size" # Indicates that this parameter is related to max_batch_size
```

In addition, all `dtype` types supported by `target_field` are as follows:

| Category | `dtype` | Meaning | `dtype_param` Format |
|---|---|---|---|
| Basic type | `int` | Takes an integer within [min, max] | — |
| Basic type | `float` | Takes a floating-point number within [min, max] | — |
| Basic type | `bool` | Boolean switch (the parameter value is true when it is greater than 0.5) | — |
| Basic type | `enum` | Selects a value from the candidate list (supports numeric or string values) | Candidate value list, such as [1, 2, 4, 8] |
| Basic type | `range` | Enumerates within [min, max] by step | Step integer, such as 10 |
| Binary derived | `ratio` | `int(ratio × target)` | Dependent field name (string), such as `"max_batch_size"` |
| Binary derived | `share` | `target.min + target.max - target.value` (complement) | Dependent field name (string) |
| Binary derived | `factories` | `product ÷ target` | `{"target_name": "field name", "product": value, "dtype": "int"}` |
| Binary derived | `times` | `product × target` | `{"target_name": "field name", "product": value, "dtype": "int"}` |
| **Ternary derived** | **`ternary_factories`** | **`product ÷ (field_a × field_b)`** | **`{"target_names": ["A", "B"], "product": value, "dtype": "int"}`** |
| **Ternary derived** | **`ternary_times`** | **`product × field_a × field_b`** | **`{"target_names": ["A", "B"], "product": value, "dtype": "int"}`** |

> [!note] Description
>
> The values of derived fields (`factories`/`times`/`ternary_factories`/`ternary_times`) are automatically derived from dependency relationships and **do not participate in the particle swarm search**. You need to set both `min` and `max` to 0. If the value of any dependent field is 0 (division scenario) or `None`/`NaN` (multiplication scenario), the derivation of this round is skipped, the field keeps its original value, and a warning log is output.

**Usage Examples of Ternary Derived Types**

Scenario 1: `tp` and `pp` are adjustable parameters, and `dp` is automatically derived from the total number of devices (16), that is, `dp = 16 ÷ (tp × pp)`:

> [!note] Constraint Description
>
> `ternary_factories` requires that the product of the dependent fields can validly derive the derived field. For `dtype = "int"`, `product` must be divisible by the product of the dependent fields. Otherwise, priority repair is triggered.
>
> - **Built-in protection for the int type**: If the result is less than 1 or cannot be divided evenly, the tool first attempts to repair the source fields. If the repair fails, the value is degraded according to min/max, and a WARNING is output.
> - **Explicitly set the range**: Configuring `min_value`/`max_value` in `dtype_param` can override the upper and lower bounds.
> - **Best practice**: Limit the enum candidates of `tp` and `pp` so that the product is divisible by `product`, avoiding dependent degradation.

```toml
# Method 1 (best practice): Limit the enum candidates of tp and pp to ensure tp × pp ≤ 16
[[mindie.target_field]]
name = "tp"
config_position = "BackendConfig.ModelDeployConfig.ModelConfig.0.tp"
min = 0
max = 1
dtype = "enum"
dtype_param = [1, 2, 4, 8]   # The maximum value of tp is 8

[[mindie.target_field]]
name = "pp"
config_position = "BackendConfig.ModelDeployConfig.ModelConfig.0.pp"
min = 0
max = 1
dtype = "enum"
dtype_param = [1, 2]          # pp is limited to 1 or 2 to ensure that the maximum tp × pp of 8 × 2 = 16 is not exceeded

[[mindie.target_field]]
name = "dp"
config_position = "BackendConfig.ModelDeployConfig.ModelConfig.0.dp"
min = 0
max = 0
dtype = "ternary_factories"
dtype_param = {target_names = ["tp", "pp"], product = 16, dtype = "int"}
# Example: tp=4, pp=2 → dp = 16 ÷ (4 × 2) = 2
#          tp=8, pp=2 → dp = 16 ÷ (8 × 2) = 1
```

```toml
# Method 2: Configure min_value as the lower-bound protection after repair failure, and output a warning
[[mindie.target_field]]
name = "dp"
config_position = "BackendConfig.ModelDeployConfig.ModelConfig.0.dp"
min = 0
max = 0
dtype = "ternary_factories"
dtype_param = {target_names = ["tp", "pp"], product = 16, dtype = "int", min_value = 1}
# If there is no valid combination to repair and the result is lower than min_value, the value degrades to min_value=1, and a WARNING is output
```

**Priority Repair Strategy (`priority_policy`)**

When the `tp` and `pp` combinations generated by PSO cannot validly derive `dp` (for example, they cannot be divided evenly or exceed the bounds), the system attempts to repair them. The repair strategy is controlled by `priority_policy`:

| Strategy | Semantics | Applicable Scenario |
|--------|------|----------|
| `balanced` (default) | Divides the particles into two groups: the first half is repaired in `target_names` order, and the second half in reverse order, reducing the structural bias caused by a single decoding order | Used by default when you have no explicit field priority preference |
| `fixed` | You explicitly specify the repair order: high-priority fields remain unchanged as much as possible, and low-priority fields are adjusted first | When you clearly know which field should be more stable |

```toml
# Example of the balanced (default) strategy
# Applicable when you have not specified which field is more important. The system automatically balances the repair directions
[[mindie.target_field]]
name = "dp"
config_position = "BackendConfig.ModelDeployConfig.ModelConfig.0.dp"
min = 0
max = 0
dtype = "ternary_factories"
dtype_param = {
  target_names = ["tp", "pp"],
  product = 32,
  dtype = "int",
  priority_policy = "balanced"   # balanced is the default and can be omitted
}
```

```toml
# Example of the fixed strategy
# Applicable when you clearly know that tp should remain stable and pp should be adjusted first
[[mindie.target_field]]
name = "dp"
config_position = "BackendConfig.ModelDeployConfig.ModelConfig.0.dp"
min = 0
max = 0
dtype = "ternary_factories"
dtype_param = {
  target_names = ["tp", "pp"],
  product = 32,
  dtype = "int",
  priority_policy = "fixed",
  priority = ["tp", "pp"]        # tp has high priority: keep tp as much as possible and adjust pp first
}
# Example: tp=8, pp=3 (invalid):
#   stage1: Fix tp=8 and find the nearest valid value in the pp candidates → pp=4, dp=1
#   If stage1 fails, proceed to stage2: both fields can be adjusted, and search in ascending order of distance
```

> [!note] Notes on priority_policy
>
> - `balanced` is the default strategy and takes effect automatically when not configured.
> - `balanced` layers the particles by decoding order, reducing the structural bias caused by the order of a single field, but it cannot guarantee a global optimum.
> - `fixed` is suitable for scenarios where you clearly know which field should be more stable, for example, when tp is determined by hardware resources.
> - Repair is performed in two stages: stage1 fixes the high-priority field and adjusts the low-priority field. If stage1 fails, stage2 allows both fields to be adjusted.
> - When all candidates are invalid, the repair fails, the value degrades to min/max truncation, and a warning is output.

Scenario 2: `seq_len` and `prefill_batch_size` are adjustable parameters, and `max_prefill_tokens` is automatically set to twice the product of the two, that is, `max_prefill_tokens = 2 × seq_len × prefill_batch_size`:

```toml
[[mindie.target_field]]
name = "seq_len"
config_position = "BackendConfig.ModelConfig.seqLen"
min = 0
max = 1
dtype = "enum"
dtype_param = [512, 1024, 2048, 4096]

[[mindie.target_field]]
name = "prefill_batch_size"
config_position = "BackendConfig.ScheduleConfig.maxPrefillBatchSize"
min = 1
max = 16
dtype = "int"

[[mindie.target_field]]
name = "max_prefill_tokens"
config_position = "BackendConfig.ScheduleConfig.maxPrefillTokens"
min = 0         # Set to 0 to make it a constant that does not participate in the search
max = 0
dtype = "ternary_times"
dtype_param = {target_names = ["seq_len", "prefill_batch_size"], product = 2, dtype = "int"}
# When seq_len=1024 and prefill_batch_size=4, max_prefill_tokens = 2 × 1024 × 4 = 8192
```

**Log detection**: Detects abnormal information in logs, distinguishes fatal errors from retryable errors, and implements intelligent error handling and retry mechanisms. Detectable error types include out of memory (OOM), device faults (NPU), network errors, and I/O errors. Fatal errors (such as OOM and NPU faults) immediately stop the scheduler, and retryable errors (such as network jitter and I/O failures) trigger automatic retry (up to 3 times).

|Parameter|Mandatory (Yes/No)|Description|
|---|---|---|
|log_snippet_length|No|Log snippet length, used to display error details. Value range: 50-1000. The default value is 200.|
|service_errors.fatal_patterns|No|List of fatal error patterns of the serving framework. The default value is empty. Common fatal errors include out of memory and device faults.|
|service_errors.retryable_patterns|No|List of retryable error patterns of the serving framework. The default value is empty. Common retryable errors include network errors and I/O errors.|
|benchmark_errors.fatal_patterns|No|List of fatal error patterns of the benchmark tool. The default value is empty.|
|benchmark_errors.retryable_patterns|No|List of retryable error patterns of the benchmark tool. The default value is empty.|

Configuration example:

```toml
[health_check]
log_snippet_length = 200

[health_check.service_errors.fatal_patterns]
out_of_memory = ["out of memory", "OOM killed", "MemoryError"]
device_error = ["NPU error", "device fault", "Ascend error"]

[health_check.service_errors.retryable_patterns]
network_error = ["connection reset", "connection refused", "timeout"]
io_error = ["file not found", "permission denied", "IO error"]
```

### Plugin Mode

The optimization tool now supports user-defined search parameter configurations and benchmark tools. You can configure them according to your requirements. You only need to adapt to the plugin mode and register the corresponding plugin. For details, see [Plugin Development Procedure](./optix_plugin_user_guide.md).

### Environment Variables and Troubleshooting<a name="environment-variables-and-troubleshooting"></a>

**Environment Variables**

| Variable            | Description |
|---------------------|------|
| OPTIX_DEPLOY_PATH   | Optional. Root directory of the deployment environment. Its `bin/` directory should contain `vllm`, `ais_bench`, and so on. Its priority is higher than `[deploy] path_prefix` in `config.toml`. If it is not set, the system PATH is used. |
| MIES_INSTALL_PATH   | MindIE installation root directory. It is retained by subprocesses and does not need to be changed for isolation. |

Priority from high to low: `OPTIX_DEPLOY_PATH`, `[deploy] path_prefix` in `config.toml`, and the system PATH used after stripping only the msModeling venv.

**Daily Startup**

When vLLM is already installed in the system, you usually do not need to set `OPTIX_DEPLOY_PATH`:

```bash
source /path/to/msmodeling/.venv/bin/activate
msmodeling optix -e vllm -b ais_bench
```

Set it only when the PATH layout is special:

```bash
export OPTIX_DEPLOY_PATH=/path/to/custom-deploy-root
source /path/to/msmodeling/.venv/bin/activate
msmodeling optix -e vllm -b ais_bench
```

**`[optix/env]` Log Reference**

| Log | Meaning | Handling |
|------|------|------|
| `No virtual environment detected` | msModeling is not installed with a venv | Run `uv sync` in the repository root directory (it automatically creates `.venv`). Do not install it into the system Python. Otherwise, `torch` and `transformers` will overwrite the deployment stack. |
| `Deployment command not found: vllm` or `mindieservice_daemon` | No command is found in the system PATH after stripping the venv | First confirm that vLLM or MindIE is installed in the system. If necessary, set `OPTIX_DEPLOY_PATH` or `[deploy] path_prefix`. |
| `Command vllm resolves to the msmodeling virtual environment` | vllm is mistakenly installed in the msModeling venv | Run `pip uninstall vllm` in that venv and use the vLLM in the system. |
| `Deployment command vllm → ...` and the path is on the system side | Normal | No change is required. |

### Log Description

The default log level during optimization is `INFO`. If you want to view the specific logs of each round, set the following environment variable before using the tool:

```bash
export MODELEVALSTATE_LEVEL=DEBUG
```

The running status of each round is output. The specific vLLM/MindIE logs are redirected to the `/tmp` directory. You can obtain the specific file path from the printed information to view the service running status.
