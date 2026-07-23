# Service Parameter Optimizer

## Overview

**Service Parameter Optimizer** (`msmodeling optix`) is an automatic optimization tool for serving parameters based on Particle Swarm Optimization (PSO). It supports `vLLM` and `MindIE`, automatically tuning parameters to find the optimal throughput configuration that meets latency requirements.

## Audience and Reading Path

This guide is for performance and deployment engineers who need automatic tuning of vLLM or MindIE serving parameters. Recommended reading order:

1. Read [Recommended Practice: Environment and Deploy Stack](#recommended-practice-environment-and-deploy-stack) and [Tool Installation](#tool-installation): install msmodeling in a uv venv; assume vLLM/MindIE are on the **system** environment.
2. Read [Preparations](#preparations) and [Quick Start](#quick-start) to confirm the serving framework and benchmark tools run independently.
3. Read [Command Parameters](#command-parameters) and [Configuration File Description](#configuration-file-description) (including `[deploy]`) to run a default optimization.
4. For issues, see [Environment Variables and Troubleshooting](#environment-variables-and-troubleshooting); for SLO tuning, see [Output File Description](#output-file-description).

The tool consists of two core functional modules:

- **Parameter optimization module**: uses PSO to automatically generate serving parameter combinations and iteratively approach the optimal solution. In addition, the Early Rejection algorithm performs early-stage benchmarks on serving parameters based on theoretical modeling, tuning experience, and partial test data.

- **Parameter verification module**: automatically starts the serving process and benchmark tool to test parameters and obtain performance results. Currently, the supported benchmark tools include `ais_bench` and `vllm_benchmark`.

> [!NOTE]
>
> The benchmark tool is about to be replaced by AISBench and is no longer supported by Service Parameter Optimizer.

Based on the modules above, Service Parameter Optimizer can automatically recommend serving parameter combinations that deliver high throughput.

The tool has been validated on LLaMA3-8B and Qwen3-8B. In principle, it does not limit the supported model types, and broader validation coverage is planned for future releases.

**Concepts**

- `vLLM` and `MindIE`: serving frameworks, which support model deployment in serving scenarios.
- `vLLM_Benchmark` and `Ais_Bench`: inference performance benchmark tools for serving frameworks.

## Supported Products<a name="ZH-CN_TOPIC_0000002479925980"></a>

> [!NOTE]
>
>For details about Ascend product models, see [Ascend Product Models](https://www.hiascend.com/document/detail/zh/AscendFAQ/ProduTech/productform/hardwaredesc_0001.html).

|Product Type| Supported (Yes/No)|
|--|:----:|
|Atlas A3 training products and Atlas A3 inference products|  Yes  |
|Atlas A2 training products and Atlas A2 inference products|  Yes  |
|Atlas 200I/500 A2 inference products|  Yes  |
|Atlas inference products|  Yes  |
|Atlas training products|  No  |

> [!NOTE]
>
>For Atlas A2 training products/Atlas A2 inference products, only the Atlas 800I A2 inference server is supported.
>For Atlas inference products, only the Atlas 300I Duo inference card and Atlas 800 inference server (model 3000) are supported.

## Preparations

**Environment and deploy stack**

| Layer | Recommended | Notes |
|-------|-------------|-------|
| **msmodeling / OptiX** | **Must** use a **uv virtual environment** | Install brings in `torch`, `transformers`, and similar packages for TensorCast simulation, not for OptiX on-device runs. Installing into system Python overwrites packages the deploy stack relies on |
| **vLLM / MindIE / benchmarks** | **System environment by default** | Assume serving and benchmarks are already on the machine. You usually do not need a separate deploy venv |

When OptiX starts child processes, it removes msmodeling venv entries from `PATH` and `PYTHONPATH`, then resolves `vllm`, `mindieservice_daemon`, `ais_bench`, and similar commands from system PATH. You do not need to edit subprocess env vars or build a second venv for deploy.

If binaries are not on the default PATH, set `OPTIX_DEPLOY_PATH` or `config.toml` `[deploy] path_prefix`.

See [Recommended Practice: Environment and Deploy Stack](#recommended-practice-environment-and-deploy-stack) and the [Environment Setup Guide](../install_guide/msmodeling_install_guide.md#optix-and-simulation-environment-isolation).

**Deploy stack setup**

Confirm serving and benchmarks work in the system environment, or under the path set by `[deploy]`. See [vLLM Server](https://docs.vllm.ai/projects/ascend/en/latest/quick_start.html), [MindIE Service](https://gitcode.com/Ascend/MindIE-Motor/blob/master/docs/zh/user_guide/quick_start.md), and [AISBench deployment](https://gitee.com/aisbench/benchmark/blob/master/README.md).

## Tool Installation

> [!IMPORTANT]
> **Install OptiX inside a virtual environment.** Run `uv sync` at the repository root; it creates `.venv` automatically.
>
> Installing msmodeling also installs `torch`, `transformers`, and related packages. Those dependencies are for TensorCast simulation in this repo, not for OptiX on-device optimization. Installing msmodeling into system Python often changes the `torch` and `transformers` versions already used on the machine, which leads to:
>
> - vLLM or MindIE failing to start or throwing inference errors
> - a mismatch with validated Ascend inference stack versions
> - other deployment tools on the same Python breaking
>
> Install msmodeling only inside a uv venv. Keep vLLM, MindIE, and benchmarks on the system install you already have.

OptiX is integrated at the msmodeling repository root. Install as follows:

```bash
git clone https://gitcode.com/Ascend/msmodeling.git   # skip if already cloned
cd msmodeling
uv sync
```

> [!NOTE]
> `uv sync` creates `.venv`, installs msmodeling in editable mode (including the `msmodeling optix` CLI), and does not require `uv venv` or `pip install -e .`. If the current branch does not include the OptiX source tree, switch to a release branch that includes OptiX or use the corresponding release package. Copying documentation alone does not provide the `msmodeling optix` command.
> [!WARNING]
> **Do not** `pip install vllm` / `mindie_llm` in the msmodeling venv. **Do not** install msmodeling in system Python without a venv. See [Recommended Practice: Environment and Deploy Stack](#recommended-practice-environment-and-deploy-stack).

## Recommended Practice: Environment and Deploy Stack<a name="recommended-practice-environment-and-deploy-stack"></a>

Typical setup: **vLLM/MindIE already on the system**; msmodeling only in a uv venv.

**① Install msmodeling (uv)**

```bash
cd /path/to/msmodeling
uv sync
```

Verify: `uv run msmodeling optix --help`

**② Confirm system deploy stack (no separate deploy venv)**

```bash
deactivate 2>/dev/null || true
which vllm    # e.g. /usr/local/bin/vllm
vllm --help
```

**③ (Optional) explicit deploy root**

Only if system PATH cannot resolve the right binaries:

```bash
export OPTIX_DEPLOY_PATH=/path/to/custom-deploy-root
```

Or in `optix/config.toml`:

```toml
[deploy]
path_prefix = "/path/to/custom-deploy-root"
```

**④ Run OptiX from the msmodeling venv**

```bash
source /path/to/msmodeling/.venv/bin/activate
msmodeling optix -e vllm -b ais_bench
```

**⑤ Confirm logs**

`[optix/env] ... 部署命令 vllm → /usr/local/bin/vllm` (or your system/custom path) means child processes use the deploy stack, not the msmodeling venv.

> [!NOTE]
> By default you **do not** need a deploy-only venv. `OPTIX_DEPLOY_PATH` / `[deploy]` are optional overrides.

## Tool Uninstallation

Uninstall from the msmodeling virtual environment:

```bash
pip uninstall msmodeling
```

## Quick Start

0. **Confirm deploy commands are outside the msmodeling venv** (vLLM example):

    ```bash
    source /path/to/msmodeling/.venv/bin/activate
    which vllm
    # expected: /usr/local/bin/vllm (system path)
    # wrong: /path/to/msmodeling/.venv/bin/vllm → see troubleshooting scenario C
    ```

1. Complete [Preparations](#preparations) and [Recommended Practice: Environment and Deploy Stack](#recommended-practice-environment-and-deploy-stack).

2. Modify the configuration file. Before starting optimization, configure [`config.toml`](../../../optix/config.toml) according to your environment, including optimization parameters, benchmark tool parameters, and serving parameters. See [Configuration File Description](#configuration-file-description) for details.

3. Start optimization. After the preceding steps are complete, start automatic optimization with a single command:

    ```bash
    msmodeling optix
    ```

    By default, the tool optimizes `vLLM` serving parameters with `Ais_Bench` as the benchmark tool.

4. View the results. The optimization time depends on the model size and dataset size, typically taking 4 to 8 hours. Upon completion, the `data_storage_*.csv` file is generated and saved in the `result/store` subdirectory under the current directory. The file records the performance of each parameter set. For details, see [Output File Description](#output-file-description).

## Command Parameters

**Function**

By combining real-device testing with the parameter verification and optimization modules, the tool delivers reliable recommended values for serving parameters.

**Precautions**

- Before starting optimization, ensure the selected serving framework and benchmark tool run in the **system deploy environment** (or `[deploy]` path), not in the msmodeling venv.
- Model paths, ports, dataset paths, and service startup parameters in `config.toml` must match your deployment environment.
- Optimization repeatedly starts services and runs benchmarks; expect long runtimes and use a stable, dedicated environment when possible.
- On environment isolation failures, log lines prefixed with `[optix/env]` explain cause and fix; see [Environment Variables and Troubleshooting](#environment-variables-and-troubleshooting).

**Syntax**

```bash
msmodeling optix [options]
```

**Parameters**

|Parameter|Mandatory (Yes/No)|Description|
|---|---|---|
|-lb or --load_breakpoint|No|Specifies whether to resume optimization from a breakpoint. Including this parameter enables breakpoint resumption; omitting it disables this feature.|
|-d or --deploy_policy|No|Specifies a deployment policy. The options are as follows:<br>&#8226;`single`: single-node deployment<br>&#8226;`multiple`: multi-node deployment<br>The default value is `single`.|
|--backup|No|Specifies whether to back up data during optimization. The options are as follows:<br>&#8226; `True`: enables backup.<br>&#8226; `False`: disables backup.<br>The default value is `False`.|
|-b or --benchmark_policy|No|Specifies a benchmark tool. The options are as follows:<br>&#8226;`vllm_benchmark`: vllm_benchmark is used as the benchmark tool.<br>&#8226;`ais_bench`: ais_bench is used as the benchmark tool.<br>The default value is `ais_bench`.<br>You need to select a benchmark tool compatible with your inference framework.|
|-e or --engine|No|Specifies an inference framework. The options are as follows:<br>&#8226;`vllm`: vLLM is used as the inference framework.<br>&#8226;`mindie`: MindIE is used as the inference framework.<br>The default value is `vllm`.|
|-c or --config|No|Specifies a custom configuration file path (TOML format). Supports the following three forms:<br>&#8226;Absolute path: directly uses the specified path;<br>&#8226;Relative path (with directory separators): resolves relative to the current working directory;<br>&#8226;Filename only: searches in the current working directory.<br>If not specified, the tool searches for configuration files in preset path order.<br>The specified file must be in valid TOML format and has the highest configuration priority.|

**Example (vLLM Service Parameter Optimization)**

1. Modify the configuration file. Before starting optimization, configure [<idp:inline displayname="code" id="code151415418558">config.toml</idp:inline>](../../../optix/config.toml) according to your environment, including optimization parameters, benchmark tool parameters, and serving parameters. See [Configuration File Description](#configuration-file-description) for details.

2. To set environment variables for the vLLM/MindIE services, configure them before running the tool. For example:

    ```bash
    export ASCEND_RT_VISIBLE_DEVICES=0
    ```

    The tool automatically applies these environment variables during optimization.

3. Once the prerequisites are ready, run the following command to start automatic optimization:

    ```bash
    msmodeling optix -e vllm
    ```

    To use `vllm_benchmark` in the vLLM scenario:

    ```bash
    msmodeling optix -e vllm -b vllm_benchmark
    ```

**Example (MindIE Serving Parameter Optimization)**

1. Modify the configuration file. Before starting optimization, configure [<idp:inline displayname="code" id="code121419465514">config.toml</idp:inline>](../../../optix/config.toml) according to your environment, including optimization parameters, benchmark tool parameters, and serving parameters. See [Configuration File Description](#configuration-file-description) for details.
2. To set environment variables for the vLLM/MindIE services, configure them before running the tool. For example:

    ```bash
    export ASCEND_RT_VISIBLE_DEVICES=0
    ```

    The tool automatically applies these environment variables during optimization.

3. Once the prerequisites are ready, run the following command to start automatic optimization:

    ```bash
    msmodeling optix -e mindie
    ```

**Example (Specifying a Custom Configuration File)**

If the configuration file is not in the default search path, you can explicitly specify it using the `-c` parameter:

```bash
# Absolute path
msmodeling optix -c /data/configs/my_config.toml

# Filename in the current directory
msmodeling optix -c my_config.toml

# Relative path
msmodeling optix -e vllm -b vllm_benchmark -c ../configs/vllm_config.toml
```

The specified configuration file has the highest priority and overrides configurations with the same name in the default path.

**Output Description**

After automatic optimization is complete, a result file in CSV format is generated and stored in the `result/store` folder in the current directory. For details, see [Output File Description](#output-file-description).

## Output File Description

Each row in the output CSV corresponds to a parameter set, with the first four columns representing performance metrics. You can filter rows that meet your requirements and update vLLM/MindIE and vllm_benchmark/ais_bench parameters with the corresponding values from the CSV.

| Field| Description|
| --- | --- |
| generate_speed | Throughput|
| time_to_first_token | Time to first token (TTFT), in seconds.|
| time_per_output_token | Time per output token (TPOT), in seconds.|
| success_rate | Request success rate|
| throughput | Test throughput, in requests per second.|
| CONCURRENCY | Concurrency|
| REQUESTRATE | Send rate|
| error | Records error reasons if parameters failed to execute.|
| backup | Data backup path (recorded when `--backup` is enabled)|
| real_evaluation | Specifies whether data is obtained from the actual test result. `false` indicates that the data is obtained from the GP model prediction.|
| fitness | Optimization score. Lower values indicate better parameter sets.|
| num_prompts | Number of requests sent by the benchmark tool.|

Other columns are the `config.toml` parameters from vLLM or MindIE.

## Appendixes

### Configuration File Description

**Deploy environment `[deploy]`**

When OptiX spawns vLLM/MindIE and benchmark child processes, it strips msmodeling venv variables and resolves the deploy stack root (with `bin/vllm`, `bin/ais_bench`, etc.) from:

| Parameter | Mandatory | Description |
|-----------|-----------|-------------|
| `path_prefix` | Optional | **Optional override** for deploy root. If unset, child processes use **system PATH** after stripping the msmodeling venv. Equivalent to directory-level `OPTIX_DEPLOY_PATH`. |

```toml
# Optional: only when system PATH cannot resolve vllm/ais_bench
[deploy]
# path_prefix = "/path/to/custom-deploy-root"
```

**Optimization parameters**: `n_particles` (number of optimization particles), `iters` (number of iterations), and `tpot_slo` (latency constraint for `time_per_output_token`).
You can configure the number of particles and iterations based on the estimated time. Each particle requires time for service startup and testing. For example, if service startup and testing takes 9 to 10 minutes per run, and you allocate 8 hours for optimization, you can run approximately 50 particles in total. The recommended configuration is 5 × 10. That is, 10 particles and 5 iterations. As a rule of thumb, set the number of particles to about twice the number of iterations.

> **Note**: All of the following optimization parameters are mandatory and must not be deleted or omitted. Otherwise, an error will occur during running.

|Parameter|Mandatory (Yes/No)|Description|
|---|---|---|
|n_particles|Yes|Number of particles (parameter combinations). The value is an integer ranging from 1 to 1000. The recommended value ranges from 15 to 30.|
|iters|Yes|Number of iterations. The value is an integer ranging from 1 to 1000. The recommended value ranges from 5 to 10.|
|ttft_penalty|Yes|Penalty coefficient for `time_to_first_token` (TTFT) violations. Set it to `0` if there is no TTFT constraint. Value range: [0, 100]. The recommended value is `1`.|
|tpot_penalty|Yes|Penalty coefficient for `time_per_output_token` (TPOT) violations. Set it to `0` if there is no TPOT constraint. Value range: [0, 100]. The recommended value is `1`.|
|success_rate_penalty|Yes|Penalty coefficient for the request success rate. The value is an integer ranging from 1 to 1000. The recommended value is `5`.|
|ttft_slo|Yes|TTFT latency constraint (in seconds) For example, if TTFT is limited to 2s, set the value to `2`. Value range: (0, 100].|
|tpot_slo|Yes|TPOT latency constraint (in seconds) For example, if TPOT is limited to 50 ms, set the value to `0.05`. Value range: (0, 100].|
|service|Yes|Node role in multi-node deployment. The options are as follows:<br>&#8226;`master`: primary node<br>&#8226;`slave`: secondary node<br>The default value is `master`.|

**Benchmark tool parameters**:
If `ais_bench` is used for the test, modify the following parameters. For details, see [AISBench Quick Start](https://github.com/AISBench/benchmark/blob/master/docs/source_en/get_started/quick_start.md).

|Parameter|Description|
|---|---|
|models| Specifies a model task. You can configure it as described in [Model Configuration Description](https://github.com/AISBench/benchmark/blob/master/docs/source_en/base_tutorials/all_params/models.md).|
|datasets| Specifies a dataset task. For details, see [Dataset Preparation Guide](https://github.com/AISBench/benchmark/blob/master/docs/source_en/get_started/datasets.md).|
|mode| Specifies the operation mode. For details, see [Operation Mode Description](https://github.com/AISBench/benchmark/blob/master/docs/source_en/base_tutorials/all_params/mode.md).|
|num_prompts| Specifies the number of prompts to run from the dataset. This parameter is valid only when `mode` is set to `perf`.|

If `vllm_benchmark` is used for the test, modify the following parameters:

|Parameter|Mandatory (Yes/No)|Description|
|---|---|---|
|host|Yes| Host IP address, which must match `host` in `[vllm.command]`. The value can be `127.0.0.1`.|
|port|Yes| Port number, which must match `port` in `[vllm.command]`.|
|model|Yes| Model path, which must match `model` in `[vllm.command]`.|
|served_model_name|Yes| Model name, which must match `served_model_name` in `[vllm.command]`.|
|dataset_name|Yes| Dataset name|
|dataset_path|Yes| Dataset path|
|num_prompts|Yes| Specifies the number of prompts to run from the dataset. The value is an integer ranging from 1 to 10000.|
|others|No| Additional parameters. Use spaces to separate them, and no space is allowed within the parameters, for example, `--ignore-eos --custom-output-len 1500`. This parameter is left empty by default.|

**vLLM serving parameters**:
When the vLLM framework is used, you need to modify the `[vllm.command]` parameter in the `config.toml` file. For example:

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
|host|Yes| Host IP address, which must match `host` in `[vllm_benchmark.command]`. The value can be `127.0.0.1`.|
|port|Yes| Port number, which must match `port` in `[vllm_benchmark.command]`.|
|model|Yes| Model path, which must match `model` in `[vllm_benchmark.command]`.|
|served_model_name|Yes| Model name, which must match `served_model_name` in `[vllm_benchmark.command]`.|
|others|No| Additional parameters. Use spaces to separate them, and no space is allowed within the parameters, for example, `--tensor-parallel-size 2 --no-enable-prefix-caching`. This parameter is left empty by default.|

### vLLM Custom Parameter Optimization

The optimizer supports adding vLLM parameters for optimization through `[[vllm.target_field]]`. Configure the parameter according to how it takes effect:

- vLLM environment variables: declare the field in `[[vllm.target_field]]` and set `config_position = "env"`. The tool writes the uppercase environment variable before starting the service in each optimization round. Do not add it to `[vllm.command].others`.
- vLLM command-line parameters: declare the field in `[[vllm.target_field]]`, then reference it from `[vllm.command].others` so it is appended to the launch command.

> **Variable reference rule**: Use the format `$UPPERCASE_FIELD_NAME` in `others` to reference an optimization field. The tool automatically replaces it with the actual value of the current iteration.

#### Example 1: vLLM Environment Variable Optimization

If the target parameter is a vLLM environment variable, add it only to `[[vllm.target_field]]`. For example:

```toml
[[vllm.target_field]]
name = "VLLM_WORKER_MULTIPROC_METHOD"
config_position = "env"
dtype = "enum"
dtype_param = ["fork", "spawn"]
value = "fork"
```

Do not reference this type of parameter in `[vllm.command].others`. Keep `others = ""` or use it only for other command-line parameters.

#### Example 2: Command-Line Enumerated Numeric Parameter (Taking `gpu_memory_utilization` as an Example)

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

#### Example 3: Command-Line Switch/Composite String Parameter (Taking Compilation Config `--compilation-config` as an Example)

When the parameter itself is a complete CLI string, you can use an empty string `""` (not enabled) and the enabled form as enum candidates. The tool automatically skips empty strings and does not append anything to the launch command.

**Step 1**: Declare the optimization field.

> **Note**: TOML uses double quotation marks `"` as string delimiters. If the string content contains double quotation marks, use `\"` to escape them. Otherwise, a parsing error will occur.

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

**MindIE serving parameters**: Modify these parameters as described in [MindIE Server Configuration Parameter Description] (<https://www.hiascend.com/document/detail/zh/mindie/20RC1/mindieservice/servicedev/mindie_service0285.html>).
You can define search ranges directly using these parameters. For example, to set the optimization search space for `max_batch_size` to 10 to 400:

```toml
[[mindie.target_field]]
name = "max_batch_size" # Serving parameter name
config_position = "BackendConfig.ScheduleConfig.maxBatchSize" # Path to the serving parameters in MindIE Server config
min = 10 # Minimum value
max = 400 # Maximum value
dtype = "int" # Data type
```

You can also define parameters relative to others. For example, to set `max_prefill_batch_size` as a ratio of `max_batch_size`, that is, `max_prefill_batch_size = ratio * max_batch_size (0 < ratio < 1)`:

```toml
[[mindie.target_field]]
name = "max_prefill_batch_size"
config_position = "BackendConfig.ScheduleConfig.maxPrefillBatchSize"
min = 0
max = 1
dtype = "ratio"
dtype_param = "max_batch_size" # Indicates that max_prefill_batch_size is proportional to max_batch_size.
```

### Plugin Mode

Service Parameter Optimizer supports user-defined search parameter configuration and benchmark tools. You can configure them as needed by adapting to the plugin interface and registering the corresponding plugin. For details, see [Plugin Development Guide](./optix_plugin_user_guide.md).

### Environment Variables and Troubleshooting<a name="environment-variables-and-troubleshooting"></a>

**Environment variables**

| Variable | Description |
|----------|-------------|
| `OPTIX_DEPLOY_PATH` | **Optional**. Deploy root; **priority over** `config.toml` `[deploy] path_prefix`. If unset, **system PATH** is used after venv strip. |
| `MIES_INSTALL_PATH` | MindIE install root; **preserved** in child-process env. |

Priority: `OPTIX_DEPLOY_PATH` > `config.toml [deploy] path_prefix` > strip msmodeling venv only (rely on system PATH).

**Recommended launch (default: system vLLM, no OPTIX_DEPLOY_PATH)**

```bash
source /path/to/msmodeling/.venv/bin/activate
msmodeling optix -e vllm -b ais_bench
```

If PATH is non-standard:

```bash
export OPTIX_DEPLOY_PATH=/path/to/custom-deploy-root
source /path/to/msmodeling/.venv/bin/activate
msmodeling optix -e vllm -b ais_bench
```

**`[optix/env]` log reference**

| Log / symptom | Cause | Action |
|---------------|-------|--------|
| `[optix/env] 当前未检测到虚拟环境` | optix without venv / msmodeling installed in system Python | Run `uv sync` at the repository root (creates `.venv` automatically); do not install in system Python—`torch`/`transformers` may overwrite deploy stack and break vLLM/MindIE |
| `[optix/env] 找不到部署命令：vllm` (or `mindieservice_daemon`) | No deploy stack on system PATH after venv strip | Verify system vLLM/MindIE install; optionally set `OPTIX_DEPLOY_PATH` or `[deploy] path_prefix` |
| `[optix/env] 命令 vllm 解析到 msmodeling 虚拟环境` | vllm installed in msmodeling venv | `pip uninstall vllm` in msmodeling venv; use system-deployed vLLM |
| `[optix/env] msmodeling 运行于虚拟环境 ...；部署命令 vllm → ...` | Normal | No action |

### Log Description

The optimizer uses [loguru](https://github.com/Delgan/loguru) with structured context (`run_id`, `stage`, `engine`) on each console line. Configure the level **before** starting the tool:

```bash
# Preferred
export OPTIX_LOG_LEVEL=INFO

# Legacy alias (used only when OPTIX_LOG_LEVEL is unset)
export MODELEVALSTATE_LEVEL=DEBUG
```

| Level | What you see |
| --- | --- |
| `INFO` (default) | Milestones: baseline pass, service ready, iteration summaries, best result; subprocess start shows multiline `command:` / `log:` |
| `DEBUG` | Parameter dumps, subprocess I/O (`Popen`, log reads, benchmark CSV glob); each line includes `file:line`; uncaught errors include full traceback (`diagnose` / `backtrace`) |
| `TRACE` | Per-particle PSO detail (fitness, swarm positions, evaluation parameters); same `file:line` column as DEBUG |

Example — iteration summary only:

```bash
export OPTIX_LOG_LEVEL=INFO
msmodeling optix -e vllm
```

Example — debug a failing candidate configuration:

```bash
export OPTIX_LOG_LEVEL=DEBUG
msmodeling optix -e mindie --backup
```

Service and benchmark subprocess logs are written to files under the result tree (and `/tmp` in some setups). Startup logs use a readable multiline format:

```text
Starting service subprocess
  command: vllm serve model_path --host 127.0.0.1 --port 8080
  log: /tmp/ms_serviceparam_optimizer__abc123
```

Baseline failures at the CLI boundary show `exit=`, `command:`, `log:`, and the last lines of the subprocess log once (no nested wrappers). Before constructing plugins, OptiX checks `PATH` using each class's `required_executable` for `-b` (`BenchmarkUnavailableError`) and `-e` (`SimulatorUnavailableError`, e.g. missing `vllm`). These fail before optimization starts — no subprocess or output cleanup runs.

### Troubleshooting

| Symptom | Likely cause | What to do |
| --- | --- | --- |
| Process exits with code `1`, message `No feasible solution found` | Every candidate failed baseline or PSO (all fitness `inf`) | Check CSV `error` column; run with `OPTIX_LOG_LEVEL=DEBUG`; verify service and benchmark commands |
| `Optimizer aborted` with traceback | Unhandled fatal error in `main()` | Read the single boundary traceback; fix config, paths, or health-check patterns |
| `BenchmarkResultError` (`OptimizerError` subclass) / ambiguous AISBench CSV | Zero or multiple `performances/*/*.csv` under benchmark output | Clean output dir; ensure one AISBench run per evaluation; **terminates the whole CLI run** (not a single PSO particle) |
| Console shows `run_id` and `stage` but little detail | Default `INFO` hides particle-level logs | Set `OPTIX_LOG_LEVEL=TRACE` for PSO internals |
| `BenchmarkUnavailableError` on startup | CLI declared by the selected `-b` plugin is not on `PATH` | Install the benchmark CLI or choose another `-b` option; fix happens before optimization starts |
| `SimulatorUnavailableError` on startup | CLI declared by the selected `-e` plugin is not on `PATH` (e.g. `vllm`) | Install the inference CLI or choose another `-e` option; fix happens before optimization starts |
| `BaselineRunError` with `exit=` / `log:` / `log tail:` | Service subprocess failed during baseline | Read the printed tail first; open the log path only if you need more context |
| Console `run_id` or `stage` shows `-` on a boundary error | Fixed in recent releases — boundary logging runs inside `contextualize` | Upgrade; boundary errors should show real `run_id` and `stage` |
| `--config` points to a missing file | Wrong path or file not deployed | Verify path; raises `ConfigFileNotFoundError` (exit code `1`) |

**CLI exit codes**

| Code | Meaning |
| --- | --- |
| `0` | Feasible best configuration found and written |
| `1` | `OptimizerError` subclasses (`ConfigFileNotFoundError`, `BenchmarkResultError`, `NoFeasibleSolutionError`, `BaselineRunError`, etc.) or unhandled fatal error |

All failure paths currently exit with code `1`. Distinguish failure types by log message or `OptimizerError` subclass, not by exit code.

Domain failures use `optix.optimizer.errors.OptimizerError` and subclasses so callers can distinguish missing config, invalid TOML, baseline failure, and no feasible solution without parsing log text. `InvalidConfigError` is raised for malformed TOML.

When the process exits non-zero, use console logs together with the CSV `error` column for diagnosis.
