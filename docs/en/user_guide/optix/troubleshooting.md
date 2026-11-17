<a id="10-appendix"></a>

## 1 Appendix

<a id="environment-variables-and-troubleshooting"></a>

### 1.1 Environment Variables and Troubleshooting

**Environment Variables**

| Variable | Description |
| --- | --- |
| OPTIX_DEPLOY_PATH | Optional. Deployment environment root directory. Its `bin/` should contain `vllm`, `ais_bench`, and so on. Its priority is higher than `[deploy] path_prefix` in `config.toml`. If it is not set, the system PATH is used. |
| MIES_INSTALL_PATH | MindIE installation root directory. It is retained in child processes and does not need to be changed for isolation. |

Priority from high to low: `OPTIX_DEPLOY_PATH`, `[deploy] path_prefix` in `config.toml`, and the system PATH after only stripping the msmodeling venv.

**Daily Startup**

When vLLM is already installed on the system, you usually do not need to set `OPTIX_DEPLOY_PATH`:

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
| --- | --- | --- |
| `No virtual environment detected` | msmodeling was not installed with a venv | Run `uv sync` in the repository root directory (this automatically creates `.venv`). Do not install it in the system Python. Otherwise, `torch` and `transformers` overwrite the deployment stack. |
| `Deployment command not found: vllm` or `mindieservice_daemon` | The command is not in the system PATH after the venv is stripped | First confirm that vLLM or MindIE is installed on the system. If necessary, set `OPTIX_DEPLOY_PATH` or `[deploy] path_prefix`. |
| `Command vllm resolves to the msmodeling virtual environment` | vllm was mistakenly installed in the msmodeling venv | Run `pip uninstall vllm` in that venv and use the vLLM installed on the system instead. |
| `Deployment command vllm → ...` and the path is on the system side | Normal | No change is required. |

<a id="log-levels-and-log-viewing"></a>

### 1.2 Log Levels and Log Viewing

The optimization tool uses [loguru](https://github.com/Delgan/loguru) to output structured logs. Each console line contains context fields such as `run_id` and `stage`. Set the log level **before** you start the tool:

```bash
export OPTIX_LOG_LEVEL=DEBUG

```

| Level | Visible Content |
| --- | --- |
| `INFO` (default) | Milestones: baseline passed, service ready, iteration summary, and optimal result. Subprocess startup appears as multiple lines of `command:`/`log:`. |
| `DEBUG` | Parameter and configuration details and subprocess I/O. Each line contains `file:line`. Uncaught exceptions contain the complete stack. |
| `TRACE` | PSO particle-level details (fitness, particle position, and parameters of a single evaluation). Like DEBUG, it contains `file:line`. |

vLLM/MindIE and benchmark logs are written to the result directory or `/tmp`. The startup log is in a multi-line readable format:

```text
Starting service subprocess
  command: vllm serve model_path --host 127.0.0.1 --port 8080
  log: /tmp/ms_serviceparam_optimizer__abc123
```

When the baseline fails, the CLI boundary outputs only one message containing `exit=`, `command:`, `log:`, and the last few lines of the log. Before the framework constructs a plugin, it checks `PATH` according to the `required_executable` declared by each type. If `-b` is missing, it throws `BenchmarkUnavailableError`. If `-e` is missing, it throws `SimulatorUnavailableError`. Both occur before optimization starts, and no subprocess is started and no output directory is cleared.

<a id="troubleshooting"></a>

### 1.3 Troubleshooting

| Symptom | Possible Cause | Suggested Action |
| --- | --- | --- |
| Exit code 1, with the message `No feasible solution found` | The fitness of the baseline or all PSO candidates is `inf`. | Check the `error` column of the CSV. Use `OPTIX_LOG_LEVEL=DEBUG`. Check the service and benchmark commands. |
| `Optimizer aborted` with a stack | Fatal error not caught by `main()` | Fix the configuration, path, or health check rules according to the single boundary traceback. |
| `BenchmarkResultError`/the AISBench CSV is not unique. | 0 or more than one `performances/*/*.csv` in the benchmark output directory | Clear the output directory. Ensure that each evaluation produces only one CSV. Terminate the entire optimization immediately. |
| The console shows only `run_id` and `stage`, with few details. | The default `INFO` level does not print particle-level logs. | Set `OPTIX_LOG_LEVEL=TRACE` to view the PSO internals. |
| `BenchmarkUnavailableError` fails immediately at startup. | The CLI declared by the selected `-b` plugin is not in `PATH`. | Install the benchmark CLI or change `-b`. This occurs before optimization starts. |
| `SimulatorUnavailableError` fails immediately at startup. | The CLI declared by the selected `-e` plugin is not in `PATH`. | Install the inference framework CLI or change `-e`. This occurs before optimization starts. |
| `BaselineRunError` containing `exit=`/`log:` | The baseline subprocess failed. | First check the log at the end of the console. Open the complete log file when necessary. |
| `--config` points to a nonexistent file. | Incorrect path or the file is not deployed. | Check the path. `ConfigFileNotFoundError` is thrown (exit code 1). |
| Early exit is enabled but the Case is not stopped. | The configuration is `action = "report"`. | After verifying the decision result, change it to `action = "terminate"`. |
| The log indicates early exit unsupported. | The engine is not `vllm`. | Early exit is currently enabled only for `-e vllm`. |
| Ordinary candidates always run completely. | No complete reference exists yet, the Case is too short, or valid window data is insufficient. | Confirm that the baseline is complete, and check the metrics sampling trace. |
| Baseline, calibration, or refinement is not terminated early. | This is the complete result protection mechanism. | No action is required. |
| `/metrics` retrieval fails. | The `metrics_url`, listening address, or port is incorrect. | Verify the vLLM `/metrics` address. A single collection failure does not actively terminate the benchmark. |

<a id="cli-exit-codes"></a>

### 1.4 CLI Exit Codes

| Exit Code | Meaning |
| --- | --- |
| 0 | A feasible optimal solution is found and output is complete. |
| 1 | A subclass of `OptimizerError` (`ConfigFileNotFoundError`, `BenchmarkResultError`, `NoFeasibleSolutionError`, `BaselineRunError`, and so on) or an uncaught fatal error |

> Currently, all failure paths exit with exit code 1. Distinguish failure types through log messages or subclasses of `OptimizerError` rather than relying on different non-zero exit codes. Invalid TOML throws `InvalidConfigError`. When the exit code is non-zero, troubleshoot based on the console logs and the CSV `error` column.
