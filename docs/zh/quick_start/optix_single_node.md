# 单机标准寻优

本文以一台昇腾设备、vLLM Ascend 和 AISBench 为例。准备环境后，可选择 Agent 或命令行方式；无需 PD 集群和多机 SSH。

## 1. 准备环境

在设备宿主机或已安装 vLLM Ascend 的容器中，准备模型文件、vLLM、AISBench 和 msmodeling。安装与环境隔离参见[环境准备与安装](../user_guide/optix/plugin_and_modes.md#4-环境准备与安装)。模型路径和服务名按实际环境填写。

在运行环境中安装 msmodeling（已有仓库时进入实际目录）：

```bash
git clone https://gitcode.com/Ascend/msmodeling.git /workspace/msmodeling
cd /workspace/msmodeling
uv sync
```

先按 [uv 官方说明](https://docs.astral.sh/uv/getting-started/installation/)安装 uv。镜像中的 vLLM、驱动和模型需适配设备；以下示例假设当前环境可访问模型。

在运行寻优的环境中验证：

```bash
npu-smi info
vllm --help
cd /workspace/msmodeling
uv run msmodeling optix --help
```

仓库路径按实际情况替换。命令行方式还需准备 AISBench；可在独立环境安装，并只将测评命令加入 PATH：

```bash
uv venv --python 3.10 /opt/aisbench-venv
uv pip install --python /opt/aisbench-venv/bin/python 'setuptools<81' ais_bench_benchmark==3.1.20260630
mkdir -p /opt/optix-deploy/bin
ln -sfn /opt/aisbench-venv/bin/ais_bench /opt/optix-deploy/bin/ais_bench
export OPTIX_DEPLOY_PATH=/opt/optix-deploy
export PATH="/opt/optix-deploy/bin:$PATH"
ais_bench --help
```

重新进入终端后需重新设置上述两个环境变量。Agent 方式可让 Agent 检查并准备 AISBench。部署栈和工具环境隔离参见[环境与部署栈](../install_guide/msmodeling_optix_env_and_deployment_stack.md)。

## 2. 选择寻优方式

### 2.1 Agent 方式

选择支持读取文件、执行命令和持续运行任务的 Agent 工具，并配置可访问的模型服务，例如 [msAgent](https://gitcode.com/Ascend/msagent) 或 [OpenCode](https://opencode.ai/docs/)。以下以 OpenCode 为例，在安装 OptiX 的环境中执行：

```bash
cd /workspace/msmodeling
source .venv/bin/activate
opencode
```

仓库路径按实际情况替换。知识库缺失时按 Skill 提示准备。

填写以下任务说明并发送：

```text
请读取本仓库 .agents/skills/optix-assistant/SKILL.md，使用 Agent 模式进行单机 vLLM 服务参数寻优，使用 AISBench 测评。
设备：[可用设备编号及卡数]。
模型：[模型路径和服务模型名]。
负载：[数据集或输入长度、输出长度、请求数及固定并发或并发范围]。
目标：[吞吐目标、TTFT 和 TPOT 要求]。
预算：[最多实测次数和总时长，包含基线与复测]。
请检查环境、准备缺失的测评配置，展示计划，并在结束后提供结果和最佳配置路径。
```

按 Agent 提示选择全自动或交互式，并确认设备、负载和预算。寻优结束后，Agent 通常会给出报告和最佳配置路径；找不到时可直接询问 Agent。结果通常位于仓库的 `.agent_optimizer/runs/<run_id>/`：`results.round-N.json` 记录每轮候选和性能，`progress.round-N.json` 记录进度，`best_result.json` 保存最佳参数。实际产物以任务状态和 Agent 给出的路径为准。不要在对话中发送密码或私钥。

### 2.2 命令行方式

先准备 AISBench 模型配置。以下命令按前面的 Python 3.10 环境定位配置目录，复制模板并打开编辑：

```bash
cd /opt/aisbench-venv/lib/python3.10/site-packages/ais_bench/benchmark/configs/models/vllm_api
cp vllm_api_stream_chat.py optix_single.py
nano optix_single.py
```

将文件内容替换为以下示例，**模型路径、服务模型名和端口须按实际情况修改**：

```python
from ais_bench.benchmark.models import VLLMCustomAPIChat

models = [dict(
    attr="service", type=VLLMCustomAPIChat, abbr="optix-single",
    path="/mnt/share/weights/模型名",  # 改为容器内模型路径
    model="single-model",  # 服务对外模型名
    stream=True, host_ip="127.0.0.1", host_port=8900,
    url="", max_out_len=256, batch_size=4, request_rate=0,
    trust_remote_code=False,
)]
```

示例使用 AISBench 自带的 `gsm8k_gen_0_shot_cot_str_perf` 数据集配置；按[数据集配置](https://github.com/AISBench/benchmark)准备可用数据。检查两个短名能找到：

```bash
ais_bench --models optix_single --datasets gsm8k_gen_0_shot_cot_str_perf --search
```

回到 msmodeling 仓库，新建 `single_node.toml`：

```bash
cd /workspace/msmodeling
nano single_node.toml
```

复制以下完整示例。三个标注处须与上面的 AISBench 配置一致；卡数、启动参数、负载、搜索范围和时延目标须适配实际模型与设备。

```toml
optimizer_strategy = "pso"
n_particles = 4
iters = 2
skip_pso = false
manage_simulator_lifecycle = true
use_request_rate_calibration = false
ttft_slo = 3
tpot_slo = 0.05
ttft_penalty = 1
tpot_penalty = 1
success_rate_penalty = 5

[data_storage]
pso_top_k = 0

[vllm.command]
host = "127.0.0.1"
port = "8900"  # 与 AISBench host_port 一致
model = "/mnt/share/weights/模型名"  # 改为实际模型路径
served_model_name = "single-model"  # 与 AISBench model 一致
others = "--tensor-parallel-size 1 --trust-remote-code"  # 按模型和设备调整

[ais_bench.command]
models = "optix_single"
mode = "perf"
others = "--datasets gsm8k_gen_0_shot_cot_str_perf --num-prompts 32"

[[vllm.target_field]]
name = "MAX_NUM_SEQS"
config_position = "run"
min = 4
max = 16
dtype = "int"
value = 4

[[vllm.target_field]]
name = "CONCURRENCY"
config_position = "env"
min = 1
max = 4
dtype = "int"
value = 4

[[vllm.target_field]]
name = "REQUESTRATE"
config_position = "env"
min = 0
max = 0
dtype = "float"
value = 0
```

示例中的 `MAX_NUM_SEQS` 范围和并发仅用于说明，不代表所有模型都适用。确认数据、模型、卡数及启动参数后运行：

```bash
uv run msmodeling optix --mode standard -e vllm -b ais_bench -c ./single_node.toml
```

工具会启动服务并进行多轮测评。完成后查看配置的 `output` 和 `[data_storage].store_dir` 对应目录；默认结果 CSV 通常在当前工作目录的 `result/store/`。可按吞吐、TTFT、TPOT 和成功率筛选；字段含义见[命令与结果参考](../user_guide/optix/commands_and_results.md#9-结果文件说明)。失败时查看终端日志及[故障排查](../user_guide/optix/troubleshooting.md#故障排查)。
