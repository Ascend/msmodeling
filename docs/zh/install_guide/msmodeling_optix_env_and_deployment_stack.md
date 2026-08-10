# 推荐实践：服务化实测寻优环境与部署栈

典型场景：系统里已经部署好 vLLM 或 MindIE，msmodeling 单独装在 uv 虚拟环境里。

**① 安装 msmodeling（uv）**

```bash
cd /path/to/msmodeling
uv sync
```

验证：`uv run msmodeling optix --help`

**② 确认系统部署栈可用**

可先 `deactivate` 退出 msmodeling venv，再检查系统里的命令，例如：

```bash
which vllm
vllm --help
```

`which vllm` 应落在系统路径，例如 `/usr/local/bin/vllm`，而不是 msmodeling 的 `.venv/bin/vllm`。

MindIE 场景请确认 `mindieservice_daemon` 可用，或 `MIES_INSTALL_PATH` 指向的安装正确。

> [!NOTE]
> 部署栈的完整部署可参考 [VLLM Server](https://docs.vllm.ai/projects/ascend/en/latest/quick_start.html)、[MindIE Service](https://gitcode.com/Ascend/MindIE-Motor/blob/master/docs/zh/user_guide/quick_start_motor.md)，以及 [AISBench 测评工具部署](https://gitee.com/aisbench/benchmark/blob/master/README.md)。

**③ 可选：指定部署根目录**

仅当系统 PATH 找不到正确命令时再配：

```bash
export OPTIX_DEPLOY_PATH=/path/to/custom-deploy-root
```

也可写入 `optix/config.toml`，字段说明见《使用指南》的[配置文件说明](../user_guide/msmodeling_optix_user_guide.md#配置文件说明)中的「高级配置 → 部署环境 `[deploy]`」：

```toml
[deploy]
path_prefix = "/path/to/custom-deploy-root"
```

**④ 在 msmodeling venv 中运行 服务化实测寻优 optix**

```bash
source /path/to/msmodeling/.venv/bin/activate
msmodeling optix -e vllm -b ais_bench
```

上一步若设置了 `OPTIX_DEPLOY_PATH`，保持 export 即可。

**⑤ 确认日志**

启动日志里出现 `[optix/env] ... 部署命令 vllm → /usr/local/bin/vllm` 一类信息，说明子进程走的是系统部署栈，而不是 msmodeling venv。

> [!NOTE]
> 默认不必再建部署专用 venv。只有 PATH 布局特殊时才需要 `OPTIX_DEPLOY_PATH` 或 `[deploy] path_prefix`。
