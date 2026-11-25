# 命令行方式：PD 分离一键寻优

本文以四台 Atlas 800I A3、DeepSeek-V4-Flash、vLLM Ascend 和 AISBench 为例。在 A 节点的 `optix-pd` 容器中准备配置，运行一次命令，依次完成 P/D 参数搜索、配比推荐、完整服务部署和负载微调，再查看报告。

<a id="第-1-步确认示例配置准备环境"></a>
<a id="11-准备-vllm-部署环境"></a>

## 1. 准备环境和连接

先完成 [PD 分离寻优公共环境准备](optix_pd_environment.md)，包括安装 AISBench 和验证部署命令。四台机器使用相同模型版本，容器内模型路径为 `/mnt/share/weights/DeepSeek-V4-Flash-w8a8-mtp`；使用其他路径时，下面配置中的路径也要同步修改。

如果重新进入容器或打开终端，先按公共环境准备中的 4.3 节重新设置 `PATH` 和 `OPTIX_DEPLOY_PATH`。

安装 PD 插件并准备 Proxy 脚本：

```bash
cd /workspace/msmodeling
uv pip install --python .venv/bin/python -e contrib/optix/vllm_pd_simulator
bash contrib/optix/vllm_pd_simulator/vllm_pd_simulator/scripts/pd/download_proxy_scripts.sh
```

重新执行 `uv sync` 后需要再次安装插件。Proxy 脚本需能下载，或提前按脚本提示准备；多机通信环境和 KV Connector 需与 vLLM Ascend 版本匹配。

<a id="第-2-步准备-pd-的-aisbench-配置"></a>

## 2. 准备 AISBench 测评配置

已有可用 AISBench 模型配置时，如模型、服务地址、P/D 输出长度和负载均符合要求，可直接复用对应短名，并在第 3 节填写；否则按以下步骤准备。

在 A 节点容器内查找 AISBench 自带的模型配置：

```bash
ais_bench --models vllm_api_stream_chat \
  --datasets gsm8k_gen_0_shot_cot_str_perf --search
```

进入输出中 `vllm_api_stream_chat.py` 所在目录。按公共环境准备中的 Python 3.10 环境安装时，通常为 `/opt/aisbench-venv/lib/python3.10/site-packages/ais_bench/benchmark/configs/models/vllm_api/`。复制两份：

```bash
cp vllm_api_stream_chat.py optix_dsv4_prefill.py
cp vllm_api_stream_chat.py optix_dsv4_decode.py
```

将 `optix_dsv4_prefill.py` 改为以下示例，并按实际环境修改模型路径和服务模型名：

```python
from ais_bench.benchmark.models import VLLMCustomAPIChat

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr="optix-dsv4-prefill",
        path="/mnt/share/weights/DeepSeek-V4-Flash-w8a8-mtp",  # 模型目录
        model="dsv4",  # 服务模型名
        stream=True,
        host_ip="127.0.0.1",
        host_port=8900,
        url="",
        max_out_len=1,
        batch_size=24,
        request_rate=0,
        trust_remote_code=False,
    )
]
```

`optix_dsv4_decode.py` 使用相同内容，将 `abbr` 改为 `optix-dsv4-decode`，并将 `max_out_len` 改为 `256` 或实际输出长度。两份文件的 `path`、`model` 与第 3 节配置保持一致。完整服务测评复用 D 配置，工具会绑定实际 Proxy 地址。

确认 GSM8K 数据集配置指向测评环境中可用的数据，可参考 [AISBench 的 GSM8K 配置](https://github.com/AISBench/benchmark/blob/master/ais_bench/benchmark/configs/datasets/gsm8k/gsm8k_gen_0_shot_cot_str_perf.py)。D 的预热与测评应使用相同输入和顺序。

```bash
ais_bench --models optix_dsv4_prefill --datasets gsm8k_gen_0_shot_cot_str_perf --search
ais_bench --models optix_dsv4_decode --datasets gsm8k_gen_0_shot_cot_str_perf --search
```

两条命令都能找到对应配置后继续。

<a id="第-3-步复制-pd-分离模板运行搜索"></a>
<a id="一个命令运行完整流程"></a>

## 3. 填写四机完整流程配置

回到 A 节点容器内的 msmodeling 仓库根目录，新建 `pd_full.toml`：

```bash
cd /workspace/msmodeling
nano pd_full.toml
```

复制下面的完整示例到文件中，重点填写：

- **机器与网络**：四台机器的 SSH 地址、账号和端口，通信 IP、网卡、容器名及设备编号。
- **模型与测评**：三处模型路径、服务模型名和 P/D 的 AISBench 配置短名，保持与第 2 节一致。
- **目标与预算**：按业务填写 P、D 及完整服务的时延目标、负载范围和搜索规模。

搜索 `【必改】`、`【按实际修改】`、`【按目标修改】` 和 `【按预算修改】` 可定位这些位置。示例的量化、MTP、并行策略及启动参数须适配实际模型和部署栈。

本例每个 P/D 实例使用 8 个设备节点，四台全卡共 64 个；`role = "auto"` 让工具根据测得的处理能力分配 P/D 资源。`enabled = true` 开启完整流程。

<details>
<summary>配置约束与进阶选项</summary>

`pd_disagg.full_service.enabled` 默认为 `false`，只搜索 P/D 参数并推荐配比；设置为 `true` 后，继续部署完整服务并微调负载。可参考[完整流程配置模板](../use_cases/examples/optix_pd_disaggregation/full_pipeline.toml)，模板以 24 卡为例，请按实际资源调整。

- 设备池使用 `role="auto"` 时，按推荐实例数自动分成 P/D；`prefill`、`decode` 可限制设备角色，必须留足对应卡数。支持留卡，受 `use_full_device` 控制。该流程要求扁平 `vllm_pd.nodes`，不接受旧的手工 group 布局。
- P/D 服务参数及各自固定启动参数自动传递到部署插件；两侧 TP×DP 必须与各自每实例卡数一致，当前插件要求 PP=1，DP>1 适用于 MoE 模型。
- `full_service.target_field` 仅配置 `CONCURRENCY`、`REQUESTRATE` 和可选 `NUM_PROMPTS`，前两项必须存在；最后阶段固定配比与服务参数，复用同一套服务微调负载，不再跑 PSO。
- 仅适用于真实 PD 节点的 vLLM 参数可放入 `full_service.prefill_additional_config` / `decode_additional_config`，合并到对应角色的 `--additional-config`；例如 `decode_additional_config = { recompute_scheduler_enable = true }`。单侧搜索不启用它们，已搜索参数仍优先。
- 最后阶段单独配置端到端 TTFT/TPOT 目标和 benchmark 负载，使用完整输出长度。支持 `vllm_benchmark`、`ais_bench`；工具自动绑定 Proxy 地址。AISBench 可复用 D 模型配置，要求单个显式模型、地址为字面量，结束后恢复该配置文件。
- SSH、模型、vLLM、CANN、KV Connector 以及 Proxy 脚本须在执行前就绪，部署插件负责服务启停与健康检查。所有资源应为本次测试专用；单侧搜索依然在当前控制端配置的设备上依次执行。

</details>

```toml
# DSv4 + vLLM + AISBench：PD 分离一键完整寻优模板
# 配套指南：../../../quick_start/optix_pso_pd_disaggregation.md
n_particles = 10 # 【按预算修改】种子数
iters = 4 # 【按预算修改】迭代轮数
optimizer_strategy = "pso"
output = "./result/pd_dsv4"
skip_pso = false
manage_simulator_lifecycle = true
use_request_rate_calibration = false

# 插件初始化配置；P/D 实际压测配置分别写在下方阶段中。
[ais_bench.command]
models = "optix_dsv4_prefill"
mode = "perf"

# 插件初始化所需的服务信息；P/D 实际启动配置分别写在下方阶段中。
[vllm.command]
host = "127.0.0.1"
port = "8900"
model = "/mnt/share/weights/DeepSeek-V4-Flash-w8a8-mtp" # 【按实际修改】三处模型路径保持一致
served_model_name = "dsv4" # 【按实际修改】与 AISBench 的 model 一致

[pd_disagg]
total_devices = 64 # 【按实际修改】四台 A3 全卡共 64 个设备节点
prefill_devices_per_instance = 8
decode_devices_per_instance = 8
top_k = 0 # 不做候选微调，仍验证最优结果并计算实例配比
use_full_device = true

# Prefill：关闭 prefix caching，压测输出 1 个 token，关注 TTFT。
[pd_disagg.prefill]
benchmark_policy = "ais_bench"
ttft_slo = 3 # 【按目标修改】P 单侧 TTFT，单位秒
ttft_penalty = 1
tpot_penalty = 0
benchmark_run_count = 1

[[pd_disagg.prefill.target_field]]
name = "MAX_NUM_BATCHED_TOKENS"
config_position = "run"
min = 8192
max = 65536
dtype = "int"
value = 8192

[[pd_disagg.prefill.target_field]]
name = "MAX_NUM_SEQS"
config_position = "run"
dtype = "enum"
value = 32
dtype_param = [24, 32]

[[pd_disagg.prefill.target_field]]
name = "CONCURRENCY"
config_position = "env"
min = 20
max = 32
dtype = "enum"
dtype_param = [20, 24, 28, 32]
value = 24

[[pd_disagg.prefill.target_field]]
name = "NUM_PROMPTS"
config_position = "env"
dtype = "times"
dtype_param = {target_name = "CONCURRENCY", product = 4, dtype = "int"}
value = 96

[[pd_disagg.prefill.target_field]]
name = "REQUESTRATE"
config_position = "env"
min = 0
max = 0
dtype = "float"
value = 0

[pd_disagg.prefill.simulator_command_overrides]
host = "127.0.0.1"
port = "8900"
model = "/mnt/share/weights/DeepSeek-V4-Flash-w8a8-mtp" # 【按实际修改】三处模型路径保持一致
served_model_name = "dsv4" # 【按实际修改】与 AISBench 的 model 一致
others = "--data-parallel-size 1 --tensor-parallel-size 8 --quantization ascend --seed 1024 --enable-expert-parallel --max-model-len 196608 --async-scheduling --trust-remote-code --no-enable-prefix-caching --speculative-config '{\"num_speculative_tokens\":3,\"method\":\"mtp\",\"enforce_eager\":true}' --gpu-memory-utilization 0.9 --compilation-config '{\"cudagraph_mode\":\"FULL_DECODE_ONLY\"}' --block-size 128 --additional-config '{\"ascend_compilation_config\":{\"enable_npugraph_ex\":true, \"enable_static_kernel\":false},\"enable_dsa_cp\":true,\"multistream_overlap_shared_expert\":true}'"

[pd_disagg.prefill.benchmark_command_overrides]
models = "optix_dsv4_prefill"
mode = "perf"
others = "--datasets gsm8k_gen_0_shot_cot_str_perf --num-prompts $NUM_PROMPTS"

# Decode：开启 prefix caching，连续测两轮，关注 TPOT。
[pd_disagg.decode]
benchmark_policy = "ais_bench"
benchmark_run_count = 2
tpot_slo = 0.05 # 【按目标修改】TPOT，单位秒
ttft_penalty = 0
tpot_penalty = 1

[[pd_disagg.decode.target_field]]
name = "MAX_NUM_BATCHED_TOKENS"
config_position = "run"
dtype = "enum"
dtype_param = [4096, 8192]
value = 8192

[[pd_disagg.decode.target_field]]
name = "MAX_NUM_SEQS"
config_position = "run"
min = 32
max = 512
dtype = "int"
value = 64

[[pd_disagg.decode.target_field]]
name = "CONCURRENCY"
config_position = "env"
min = 20
max = 32
dtype = "enum"
dtype_param = [20, 24, 28, 32]
value = 24

[[pd_disagg.decode.target_field]]
name = "NUM_PROMPTS"
config_position = "env"
dtype = "times"
dtype_param = {target_name = "CONCURRENCY", product = 4, dtype = "int"}
value = 96

[[pd_disagg.decode.target_field]]
name = "REQUESTRATE"
config_position = "env"
min = 0
max = 0
dtype = "float"
value = 0

[pd_disagg.decode.simulator_command_overrides]
host = "127.0.0.1"
port = "8900"
model = "/mnt/share/weights/DeepSeek-V4-Flash-w8a8-mtp" # 【按实际修改】三处模型路径保持一致
served_model_name = "dsv4" # 【按实际修改】与 AISBench 的 model 一致
others = "--data-parallel-size 1 --tensor-parallel-size 8 --quantization ascend --seed 1024 --enable-expert-parallel --max-model-len 196608 --async-scheduling --trust-remote-code --enable-prefix-caching --speculative-config '{\"num_speculative_tokens\":3,\"method\":\"mtp\",\"enforce_eager\":true}' --gpu-memory-utilization 0.9 --compilation-config '{\"cudagraph_mode\":\"FULL_DECODE_ONLY\"}' --block-size 128 --additional-config '{\"ascend_compilation_config\":{\"enable_npugraph_ex\":true, \"enable_static_kernel\":false},\"enable_dsa_cp\":true,\"multistream_overlap_shared_expert\":true}'"

[pd_disagg.decode.benchmark_command_overrides]
models = "optix_dsv4_decode"
mode = "perf"
others = "--datasets gsm8k_gen_0_shot_cot_str_perf --num-prompts $NUM_PROMPTS --summarizer stable_stage"

# 自动完整链路：同一次运行中部署集群，并通过 Proxy 微调压测负载。
# 按实际环境修改以下主机、卡号、通信网卡；auto 角色允许按推荐配比自动分配。
[pd_disagg.full_service]
benchmark_policy = "ais_bench"
ttft_slo = 3.0 # 【按目标修改】完整服务 TTFT，单位秒
tpot_slo = 0.05 # 【按目标修改】TPOT，单位秒
enabled = true
decode_others = "--data-parallel-size 1 --tensor-parallel-size 8 --quantization ascend --seed 1024 --enable-expert-parallel --max-model-len 196608 --async-scheduling --trust-remote-code --enable-prefix-caching --speculative-config '{\"num_speculative_tokens\":3,\"method\":\"mtp\",\"enforce_eager\":true}' --gpu-memory-utilization 0.9 --compilation-config '{\"cudagraph_mode\":\"FULL_DECODE_ONLY\"}' --block-size 128 --additional-config '{\"ascend_compilation_config\":{\"enable_npugraph_ex\":true, \"enable_static_kernel\":false},\"enable_dsa_cp\":true,\"multistream_overlap_shared_expert\":true}'"

[pd_disagg.full_service.benchmark_command_overrides]
models = "optix_dsv4_decode"
mode = "perf"
others = "--datasets gsm8k_gen_0_shot_cot_str_perf --num-prompts $NUM_PROMPTS"

[[pd_disagg.full_service.target_field]]
name = "CONCURRENCY"
config_position = "env"
dtype = "int"
min = 4
max = 128
value = 24

[[pd_disagg.full_service.target_field]]
name = "NUM_PROMPTS"
config_position = "env"
dtype = "times"
dtype_param = { target_name = "CONCURRENCY", product = 4, dtype = "int" }
min = 16
max = 512
value = 96

[[pd_disagg.full_service.target_field]]
name = "REQUESTRATE"
config_position = "env"
dtype = "float"
min = 0
max = 1000
value = 0

[vllm_pd]
# 控制端须能免密 SSH 到各宿主机；容器场景填写各 host 的 docker_container_id。

[vllm_pd.proxy]
hosts = "pd-a"
service_port = 8000

[[cluster.hosts]]
id = "pd-a"  # 节点标识，与下方 hosts 对应
ssh_ip = "192.0.2.10"  # 【必改】宿主机 SSH 地址
ssh_port = 22  # 【按实际修改】SSH 端口
ssh_user = "worker"  # 【必改】SSH 账号
docker_container_id = "optix-pd"  # 【按实际修改】宿主机上的容器名
service_ip = "192.0.2.10"  # 【必改】此节点通信 IP
network_interface = "eth0"  # 【必改】此节点通信网卡

[[cluster.hosts]]
id = "pd-b"  # 节点标识，与下方 hosts 对应
ssh_ip = "192.0.2.11"  # 【必改】宿主机 SSH 地址
ssh_port = 22  # 【按实际修改】SSH 端口
ssh_user = "worker"  # 【必改】SSH 账号
docker_container_id = "optix-pd"  # 【按实际修改】宿主机上的容器名
service_ip = "192.0.2.11"  # 【必改】此节点通信 IP
network_interface = "eth0"  # 【必改】此节点通信网卡

[[cluster.hosts]]
id = "pd-c"  # 节点标识，与下方 hosts 对应
ssh_ip = "192.0.2.12"  # 【必改】宿主机 SSH 地址
ssh_port = 22  # 【按实际修改】SSH 端口
ssh_user = "worker"  # 【必改】SSH 账号
docker_container_id = "optix-pd"  # 【按实际修改】宿主机上的容器名
service_ip = "192.0.2.12"  # 【必改】此节点通信 IP
network_interface = "eth0"  # 【必改】此节点通信网卡

[[cluster.hosts]]
id = "pd-d"  # 节点标识，与下方 hosts 对应
ssh_ip = "192.0.2.13"  # 【必改】宿主机 SSH 地址
ssh_port = 22  # 【按实际修改】SSH 端口
ssh_user = "worker"  # 【必改】SSH 账号
docker_container_id = "optix-pd"  # 【按实际修改】宿主机上的容器名
service_ip = "192.0.2.13"  # 【必改】此节点通信 IP
network_interface = "eth0"  # 【必改】此节点通信网卡

[[vllm_pd.nodes]]
hosts = "pd-a"
role = "auto"  # 按推荐配比自动分配 P/D
gpu_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] # 【按实际修改】可用设备编号

[[vllm_pd.nodes]]
hosts = "pd-b"
role = "auto"  # 按推荐配比自动分配 P/D
gpu_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] # 【按实际修改】可用设备编号

[[vllm_pd.nodes]]
hosts = "pd-c"
role = "auto"  # 按推荐配比自动分配 P/D
gpu_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] # 【按实际修改】可用设备编号

[[vllm_pd.nodes]]
hosts = "pd-d"
role = "auto"  # 按推荐配比自动分配 P/D
gpu_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] # 【按实际修改】可用设备编号

```

注意三处 `model` 路径和两份 AISBench 配置中的 `path` 要一致；`served_model_name` 与 AISBench 的 `model` 保持一致。P 和 D 单侧时延目标与完整服务目标分别填写，单位均为秒。

## 4. 运行完整流程

在 A 节点容器的 msmodeling 仓库根目录执行：

```bash
uv run --no-sync msmodeling optix --mode pd_disagg -e vllm -b ais_bench -c ./pd_full.toml
```

工具会依次搜索 P、D 参数，推荐实例配比，部署完整 PD 服务并通过 Proxy 微调负载。搜索和测评可能需要较长时间。

## 5. 查看结果

完成后打开终端输出的 `final_result.md`。按本例的 `output` 设置，目录为 `result/pd_dsv4/pd_disagg/<run_id>/`。

| 想看什么 | 位置 |
|---|---|
| 最终报告、P/D 配比和服务参数 | `final_result.md` |
| 完整服务的部署配置 | `full_service/deployment.json` |
| P、D 单侧结果 | `prefill/phase_result.json`、`decode/phase_result.json` |
| 每次候选的实测记录 | 各阶段输出目录中的 `data_storage_*.csv` |

报告中 `status` 表示流程是否完成，`full_service.slo_met` 表示完整服务的指标是否满足目标。发生失败时查看报告中的失败阶段、原因及对应日志。完整服务阶段结束后工具会停止服务；需要再次部署时，可参考 `full_service/deployment.json` 中的配置和实际部署环境进行复现。报告还汇总最终并发、请求速率、实测指标、机器与设备分配及部署配置。未启用完整服务时不会生成该完整流程报告。

<a id="第-4-步根据结果部署-pd-和-proxy"></a>

## 6. 分步寻优与常见问题

需要分开运行 P/D 搜索、手工部署服务或单独调整测评负载时，参考 [PD 分离分步寻优详细案例](../use_cases/optix_pd_step_by_step.md)。

<a id="第-5-步通过-proxy-压测完整-pd-服务"></a>

已部署服务后需要通过 Proxy 调整负载时，参见[分步案例的完整服务微调](../use_cases/optix_pd_step_by_step.md#4-微调完整-pd-服务)。

在异构设备上分别搜索 P/D 参数，或使用多机 PD 混部插件测量服务单元并手动计算配比时，参见[分别搜索 P/D 参数并计算部署配比](optix_pso_pd_ratio_separate_search.md)。

| 现象 | 先检查 |
|---|---|
| SSH 或容器访问失败 | A 节点容器能否免密连接各宿主机；账号是否有 Docker 权限；容器名是否正确 |
| 找不到部署命令 | 按公共环境准备验证 vLLM 和 AISBench 的路径；重新进入容器后恢复环境变量 |
| AISBench 找不到模型配置 | 检查配置文件目录和短名，重新执行 `ais_bench --search` |
| 配置校验或资源分配失败 | 检查 IP、容器、网卡、设备编号、`total_devices` 和每实例设备数 |
| Proxy 或 P/D 启动失败 | 检查 Proxy 脚本、端口、KV Connector 和通信环境，查看报告与部署日志 |
