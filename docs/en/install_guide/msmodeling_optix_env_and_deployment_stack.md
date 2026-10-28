# Recommended Practice: Environment and Deployment Stack for Serving Optimization Through Real-World Testing

Typical scenario: vLLM or MindIE is already deployed in the system, and msModeling is installed separately in a uv virtual environment.

**1. Install msModeling (`uv`)**

```bash
cd /path/to/msmodeling
uv sync
```

To verify: run `uv run msmodeling optix --help`.

**2. Confirm That the System Deployment Stack Is Available**

You can first run `deactivate` to exit the msModeling venv, and then check the commands in the system. For example:

```bash
which vllm
vllm --help
```

`which vllm` should resolve to a system path, for example `/usr/local/bin/vllm`, rather than the `.venv/bin/vllm` in the msModeling venv.

For MindIE scenarios, confirm that `mindieservice_daemon` is available, or that the installation pointed to by `MIES_INSTALL_PATH` is correct.

> [!NOTE]
> For the full deployment of the deployment stack, see [VLLM Server](https://docs.vllm.ai/projects/ascend/en/latest/quick_start.html), [MindIE Service](https://gitcode.com/Ascend/MindIE-Motor/blob/master/docs/en/user_guide/quick_start.md), and [AISBench Benchmark Tool Deployment](https://gitee.com/aisbench/benchmark/blob/master/README.md).

**3. Optional: Specify the Deployment Root Directory**

Configure this only when the system PATH cannot find the correct command:

```bash
export OPTIX_DEPLOY_PATH=/path/to/custom-deploy-root
```

You can also write it into `optix/config.toml`. For field descriptions, see [Configuration File Description](../user_guide/msmodeling_optix_user_guide.md#7-configuration-file-description) in the *User Guide*, specifically "Advanced Configuration → Deployment Environment `[deploy]`":

```toml
[deploy]
path_prefix = "/path/to/custom-deploy-root"
```

**4. Run OptiX for Serving Optimization Through Real-World Testing in the msModeling venv**

```bash
source /path/to/msmodeling/.venv/bin/activate
msmodeling optix -e vllm -b ais_bench
```

If you set `OPTIX_DEPLOY_PATH` in the preceding step, keep it exported.

**5. Confirm the Logs**

If the startup log contains information such as `[optix/env] ... deploy command vllm → /usr/local/bin/vllm`, the subprocess uses the system deployment stack instead of the msModeling venv.

> [!NOTE]
> By default, you do not need to create a dedicated deployment venv. Only when the PATH layout is special do you need `OPTIX_DEPLOY_PATH` or `[deploy] path_prefix`.
