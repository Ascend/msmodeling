# Recommended Practice: Service Parameter Optimization Environment and Deployment Stack

Typical scenario: vLLM or MindIE is already deployed on the system, and msmodeling is installed separately in a `uv` virtual environment.

**① Install msmodeling (uv)**

```bash
cd /path/to/msmodeling
uv sync
```

Verify: `uv run msmodeling optix --help`

**② Confirm the system deployment stack is usable**

You can `deactivate` to exit the msmodeling venv, then check the system commands, for example:

```bash
which vllm
vllm --help
```

`which vllm` should resolve to a system path such as `/usr/local/bin/vllm`, not msmodeling's `.venv/bin/vllm`.

For the MindIE scenario, confirm that `mindieservice_daemon` is usable, or that the installation pointed to by `MIES_INSTALL_PATH` is correct.

> [!NOTE]
> For the full deployment of the stack, see [VLLM Server](https://docs.vllm.ai/projects/ascend/en/latest/quick_start.html), [MindIE Service](https://gitcode.com/Ascend/MindIE-Motor/blob/master/docs/zh/user_guide/quick_start_motor.md), and [AISBench Benchmark Tool Deployment](https://gitee.com/aisbench/benchmark/blob/master/README.md).

**③ Optional: specify the deployment root directory**

Configure this (and only this) when the system `PATH` cannot find the correct commands:

```bash
export OPTIX_DEPLOY_PATH=/path/to/custom-deploy-root
```

You can also write it into `optix/config.toml`. See [Deployment Environment `[deploy]`](../user_guide/msmodeling_optix_user_guide.md#deployment-environment) in the User Guide:

```toml
[deploy]
path_prefix = "/path/to/custom-deploy-root"
```

**④ Run the OptiX service parameter optimizer in the msmodeling venv**

```bash
source /path/to/msmodeling/.venv/bin/activate
msmodeling optix -e vllm -b ais_bench
```

If you set `OPTIX_DEPLOY_PATH` in the previous step, keep the export.

**⑤ Confirm the logs**

If the startup log shows something like `[optix/env] ... deployment command vllm → /usr/local/bin/vllm`, the subprocess is using the system deployment stack rather than the msmodeling venv.

> [!NOTE]
> By default there is no need to create a separate deployment venv. `OPTIX_DEPLOY_PATH` or `[deploy] path_prefix` is only needed when the `PATH` layout is unusual.