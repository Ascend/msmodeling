# OptiX PSO：分别搜索 P/D 参数并计算部署配比

本文介绍如何在 P、D 各自的部署环境中运行 PD 混部寻优，通过压测负载分别测量 Prefill（P）和 Decode（D）的处理能力，再手动计算 PD 配比。

适用于以下两种部署方式，也可以组合使用：

| 部署方式 | P 侧操作位置 | D 侧操作位置 | 使用的服务插件 |
|---|---|---|---|
| 异构设备 | 实际用于 P 的设备所在机器 | 实际用于 D 的设备所在机器 | 单机使用 `vllm`；多机使用 `multihost_infer` |
| 多机部署一个 P/D 服务单元 | P 服务组的主节点 | D 服务组的主节点 | 多机 PD 混部插件 `multihost_infer` |

操作顺序：**准备两侧环境 → 分别搜索 P/D 参数（D 每组候选先预热再计分）→ 确认 P/D 吞吐 → 计算配比 → 部署完整 PD 服务并验证**。

本文把一组可以独立启停、对外提供一个服务入口的部署作为一个“服务单元”。单机服务单元可以占用一台机器的多张卡；多机服务单元包含主节点及其 worker。后续的 QPS、卡数、机器数和 P:D 比例均按整个服务单元统计，单元内部的 DP、TP、EP 沿用本次测量的配置。

## 第 1 步：准备 P、D 环境和测试负载

### 1.1 确定本次测试配置

分别记录 P、D 的部署信息，并确定同一组业务负载：

| 项目 | 填写方式 |
|---|---|
| 模型 | 两侧使用同一模型，记录权重路径、量化方式、对外模型名和分词器 |
| 设备与拓扑 | 分别记录两侧的设备型号、每单元机器数、卡数、DP/TP/EP 等部署参数 |
| 输入请求 | 固定输入请求集、顺序和采样方式；P、D 使用对应的输入负载 |
| P 输出长度 | 1 token |
| D 输出长度 | 目标业务输出长度，例如 512 token |
| 时延目标 | P 检查 TTFT，D 检查 TPOT，例如 3 秒和 0.05 秒 |
| 请求率 | 本文使用 `0`，即不限发送速率，通过并发搜索处理能力 |
| 请求数 | 本文示例为 512，按实际测试时长和缓存容量设置 |
| 搜索规模 | 本文示例为 8 个粒子、4 次迭代，两侧可分别设置 |

以下命令中的模型目录、端口、卡数、并行策略和搜索范围均需替换为实际值。**P、D 可以使用不同硬件、不同并行策略和不同批处理参数，各自配置与最终部署保持一致。**

### 1.2 安装 OptiX 和压测工具

在 P、D 各自的单机部署环境或多机主节点环境中，按照[环境准备步骤](optix_pso_pd_disaggregation.md#11-准备-vllm-部署环境)准备 vLLM、AISBench 和 OptiX，完成该指南第 1 步的部署命令检查。

后续 `uv` 命令均在 msmodeling 仓库根目录执行。服务和 AISBench 使用各自的部署环境。

多机侧另安装插件：

```bash
uv pip install --python .venv/bin/python -e contrib/optix/multihost_infer
uv run --no-sync msmodeling optix --help
```

安装扩展插件后，本文统一使用 `uv run --no-sync` 运行 OptiX，保留当前环境中已安装的插件。

### 1.3 准备 P、D 的 AISBench 配置

参考[准备 AISBench 配置](optix_pso_pd_disaggregation.md#第-2-步准备-pd-的-aisbench-配置)，在两侧部署环境分别复制模型配置，命名为 `optix_ratio_prefill.py`、`optix_ratio_decode.py`。修改如下字段：

| 字段 | P 配置 | D 配置 |
|---|---|---|
| `abbr` | `optix-ratio-prefill` | `optix-ratio-decode` |
| `path` | 本侧可读取的模型/分词器目录 | 本侧可读取的同一模型/分词器目录 |
| `model` | 本侧服务对外模型名，示例为 `pd-model` | 本侧服务对外模型名，示例为 `pd-model` |
| `host_ip` | 本侧入口地址，主节点本机可填 `127.0.0.1` | 本侧入口地址，主节点本机可填 `127.0.0.1` |
| `host_port` | 本文示例 `8900` | 本文示例 `8900` |
| `max_out_len` | `1` | 目标输出长度，本文示例 `512` |
| `batch_size` | 初始并发，本文示例 `32` | 初始并发，本文示例 `32` |
| `request_rate` | `0` | `0` |

`batch_size`、`request_rate` 使用 `batch_size=32,`、`request_rate=0,` 这样的直接赋值形式。配置中存在非空 `url` 时，一并设置为本侧接口地址。

本文使用 `gsm8k_gen_0_shot_cot_str_perf` 演示。将其配置中的数据路径指向本次固定请求集；两侧使用相同版本的数据和处理配置。D 的预热、正式测量还需保持请求内容、顺序、请求数、并发、输出长度和采样参数一致。

在对应机器检查短名解析：

```bash
# P 侧
ais_bench --models optix_ratio_prefill --datasets gsm8k_gen_0_shot_cot_str_perf --search
```

```bash
# D 侧
ais_bench --models optix_ratio_decode --datasets gsm8k_gen_0_shot_cot_str_perf --search
```

## 第 2 步：分别配置 P、D 参数搜索

### 2.1 P 侧搜索配置

在 P 侧新建 `prefill_search.toml`，填入以下完整配置。示例单元使用 TP=8、DP=1；部署所需的 EP、量化、MTP 等固定参数追加到 `others` 中。

```toml
optimizer_strategy = "pso"
n_particles = 8
iters = 4
output = "./result/pd_ratio_prefill"
skip_pso = false
manage_simulator_lifecycle = true
use_request_rate_calibration = false
benchmark_run_count = 1
fine_tune_mode = "pd_mixed"

ttft_slo = 3
tpot_slo = 0.05
ttft_penalty = 1
tpot_penalty = 0
success_rate_slo = 1
success_rate_penalty = 5

[data_storage]
pso_top_k = 0

[vllm.command]
host = "127.0.0.1"
port = "8900"
model = "/data/models/your-model"
served_model_name = "pd-model"
others = "--tensor-parallel-size 8 --data-parallel-size 1 --trust-remote-code --no-enable-prefix-caching"

[[vllm.target_field]]
name = "MAX_NUM_BATCHED_TOKENS"
config_position = "run"
min = 8192
max = 65536
dtype = "int"
value = 8192

[[vllm.target_field]]
name = "MAX_NUM_SEQS"
config_position = "run"
dtype = "enum"
dtype_param = [24, 32]
value = 32

[[vllm.target_field]]
name = "CONCURRENCY"
config_position = "env"
min = 16
max = 128
dtype = "enum"
dtype_param = [16, 32, 64, 128]
value = 32

[[vllm.target_field]]
name = "REQUESTRATE"
config_position = "env"
min = 0
max = 0
dtype = "float"
value = 0

[ais_bench.command]
models = "optix_ratio_prefill"
mode = "perf"
others = "--datasets gsm8k_gen_0_shot_cot_str_perf --num-prompts 512"
```

P 配置关闭 prefix caching，输出 1 token，按 TTFT 和成功率约束搜索吞吐。`pso_top_k=0` 用于完成 PSO 搜索和最优候选验证，候选复测在第 4 步执行。

### 2.2 D 侧搜索配置

在 D 侧以同一份配置为基础保存为 `decode_search.toml`，按下表替换对应字段，保留其余配置：

| 配置位置 | D 侧填写值 |
|---|---|
| 顶层 `output` | `"./result/pd_ratio_decode"` |
| 顶层 `benchmark_run_count` | `2`，同一服务上第一轮预热、第二轮计分 |
| 顶层 `ttft_penalty` / `tpot_penalty` | `0` / `1` |
| 顶层 `tpot_slo` | 目标 TPOT，本文为 `0.05` |
| `[vllm.command] model` | D 侧模型目录 |
| `[vllm.command] others` | D 的固定并行参数，启用 `--enable-prefix-caching` |
| `MAX_NUM_BATCHED_TOKENS` 字段 | `dtype="enum"`、`dtype_param=[4096,8192]`、`value=8192`；删除该字段原有的 `min`、`max` |
| `MAX_NUM_SEQS` 字段 | `dtype="int"`、`min=32`、`max=512`、`value=64`；删除该字段原有的 `dtype_param` |
| `[ais_bench.command] models` | `"optix_ratio_decode"` |

单机 D 的 `others` 示例：

```toml
others = "--tensor-parallel-size 8 --data-parallel-size 1 --trust-remote-code --enable-prefix-caching"
```

D 侧每组候选自动在同一服务上测评两轮，第一轮填充缓存、第二轮记录正式指标。保持 `use_request_rate_calibration=false`，固定两轮请求内容、顺序、请求数、输出长度和采样参数。需要更多预热轮次时，将 `benchmark_run_count` 设为更大的正整数，最后一轮计分。

## 第 3 步：在各自设备上执行搜索

### 3.1 异构单机部署

在 P 设备所在机器执行：

```bash
uv run --no-sync msmodeling optix --mode standard -e vllm -b ais_bench \
  -c ./prefill_search.toml --backup
```

在 D 设备所在机器执行：

```bash
uv run --no-sync msmodeling optix --mode standard -e vllm -b ais_bench \
  -c ./decode_search.toml --backup
```

两侧可独立执行。分别保留配置文件、设备信息和 `result/pd_ratio_prefill/`、`result/pd_ratio_decode/` 下的结果。

### 3.2 使用多机 PD 混部插件

对需要多机的 P 或 D 服务单元，在该单元主节点执行本节。两侧都使用多机时，分别准备两套集群配置并在各自主节点运行。

先按[多机插件环境准备](../../../contrib/optix/multihost_infer/multihost_inference_optimization/README.md#环境准备)完成部署：

1. 主节点与 OptiX 同机运行；主节点环境可免密 SSH 登录各 worker 宿主机。
2. 各节点具有一致的模型目录和可用的 vLLM-Ascend 环境，已验证目标模型及并行拓扑能够正常推理。
3. 按插件 `template.sh` 准备 Ascend 环境脚本和 `ascend-infer` conda 环境；自定义环境时，将模板中的路径和环境名调整为实际值。
4. 完成插件文档中的多节点通信验证。

以 P 单元由两台 8 卡机器构成为例，创建 `prefill_cluster.toml`：

```toml
docker_use_sudo = false

[vllm_mix]
chips_per_node = 8
data_parallel_rpc_port = 13389

[vllm_mix.env]
OMP_PROC_BIND = false
OMP_NUM_THREADS = 1
HCCL_BUFFSIZE = 1024
TASK_QUEUE_ENABLE = 1

[[vllm_mix.node]]
host = "192.0.2.10"
nic_name = "eth0"

[[vllm_mix.workers]]
host = "192.0.2.11"
ssh_port = 22
ssh_user = "your-user"
nic_name = "eth0"
# worker 使用容器时填写容器 ID；在宿主机运行时省略此项。
# docker_container_id = "your-worker-container-id"
```

将示例地址、通信网卡、SSH 用户和容器信息替换为实际值；更多 worker 按同样结构追加 `[[vllm_mix.workers]]`。

在对应搜索配置的 `[vllm.command] others` 中填入整个服务单元的并行参数。以下为 2 节点、每节点 8 卡、TP=8、全局 DP=2、PP=PCP=1 的填写示例：

```toml
# P 单元
others = "--tensor-parallel-size 8 --data-parallel-size 2 --trust-remote-code --no-enable-prefix-caching"
```

```toml
# D 单元
others = "--tensor-parallel-size 8 --data-parallel-size 2 --trust-remote-code --enable-prefix-caching"
```

EP、量化、MTP 等参数使用已验证的模型部署配置。按照插件的节点放置规则配置并行策略：全局 DP 是节点数的整数倍，每节点占用满足 `DP / 节点数 × TP × PP × PCP <= 每节点卡数`，TP、PP、PCP 组放在节点内。本文搜索期间固定这些参数，搜索批处理参数和并发。

在 P 单元主节点执行：

```bash
export CLUSTER_CONFIG_PATH="$PWD/prefill_cluster.toml"
uv run --no-sync msmodeling optix --mode standard -e multihost_infer -b ais_bench \
  -c ./prefill_search.toml --backup
```

在 D 单元主节点准备 `decode_cluster.toml`，填写 D 的节点信息，然后执行：

```bash
export CLUSTER_CONFIG_PATH="$PWD/decode_cluster.toml"
uv run --no-sync msmodeling optix --mode standard -e multihost_infer -b ais_bench \
  -c ./decode_search.toml --backup
```

一个 P/D 单元包含几台机器，就记录几台机器的完整资源占用。例如上述两节点单元记录为“2 台、16 卡、全局 DP=2”，测得的 QPS 归属于整个单元。

## 第 4 步：固定候选，测量用于配比的 QPS

### 4.1 选择并部署候选

分别打开两侧 `result/pd_ratio_*/store/data_storage_*.csv`，查看 `real_evaluation=true` 的实测记录，记录服务参数和并发。P 侧从 TTFT、成功率达标的记录中选择吞吐较高的参数；D 侧从 TPOT、成功率达标的记录中选择吞吐较高的参数。D 的 CSV 指标来自每组候选的最后一轮正式测量，可直接用于配比。需要独立复测时，按下面步骤重新部署、预热和测量。

| 需要记录的列 | 用途 |
|---|---|
| `MAX_NUM_BATCHED_TOKENS`、`MAX_NUM_SEQS` | 重建对应服务参数 |
| `CONCURRENCY` | 设置本次复测并发 |
| `success_rate` | 选择成功率为 1 的测量结果 |
| `time_to_first_token(s)` | P 的 TTFT |
| `time_per_output_token(s)` | D 的 TPOT |
| `throughput(requests/s)` | 请求吞吐，统一使用 req/s |

停止上一组候选服务后，按候选参数重新启动一个服务单元。单机示例：

```bash
# P 侧示例，替换为本侧候选参数和完整固定部署参数。
vllm serve /data/models/your-model --served-model-name pd-model --port 8900 \
  --tensor-parallel-size 8 --data-parallel-size 1 --trust-remote-code \
  --no-enable-prefix-caching --max-num-batched-tokens 16384 --max-num-seqs 32
```

```bash
# D 侧示例，替换为本侧候选参数和完整固定部署参数。
vllm serve /data/models/your-model --served-model-name pd-model --port 8900 \
  --tensor-parallel-size 8 --data-parallel-size 1 --trust-remote-code \
  --enable-prefix-caching --max-num-batched-tokens 8192 --max-num-seqs 64
```

多机侧在 `result/pd_ratio_*/back_up/` 下，按搜索日志定位该候选的 `Reproduce/` 目录，核对脚本中的服务参数与 CSV 候选一致。将 `start_node.sh` 复制到原主节点的 `/tmp/optix-ratio-replay/`，将 `start_work_0.sh` 等脚本分别复制到对应 worker 的同名目录；worker 使用容器时，目录和执行终端均位于原容器内。

先在主节点终端执行：

```bash
cd /tmp/optix-ratio-replay
bash start_node.sh
```

随后在各 worker 的另一个终端执行对应脚本，以第一个 worker 为例：

```bash
cd /tmp/optix-ratio-replay
bash start_work_0.sh
```

保持各节点服务运行，在压测终端检查入口：

```bash
curl --fail http://127.0.0.1:8900/health
```

确认主节点日志出现 `Application startup complete` 后继续。复测期间保留脚本中的并行策略、模型固定参数和节点通信配置，更换候选时停止上一组服务后重新启动。

### 4.2 测量 P 吞吐

在 P 的 AISBench 模型配置中，将 `batch_size` 设置为候选并发，`request_rate=0`、`max_out_len=1`。服务就绪后执行：

```bash
ais_bench --models optix_ratio_prefill --datasets gsm8k_gen_0_shot_cot_str_perf \
  --mode perf --num-prompts 512 --work-dir ./pd_ratio_measure/prefill --debug
```

在成功率为 1、TTFT 达标的结果中，记录请求吞吐为 `QPS_P`，并保留对应服务参数和并发。

### 4.3 预热并测量 D 吞吐

在 D 的 AISBench 模型配置中，将 `batch_size` 设置为候选并发，`request_rate=0`、`max_out_len` 设置为目标输出长度。执行以下两轮压测，**两轮之间保持同一服务单元运行**：

```bash
# 第一轮：使用固定请求集填充 prefix cache。
ais_bench --models optix_ratio_decode --datasets gsm8k_gen_0_shot_cot_str_perf \
  --mode perf --num-prompts 512 --work-dir ./pd_ratio_measure/decode_warmup --debug

# 第二轮：使用相同请求集正式测量，记录稳态结果。
ais_bench --models optix_ratio_decode --datasets gsm8k_gen_0_shot_cot_str_perf \
  --mode perf --num-prompts 512 --work-dir ./pd_ratio_measure/decode_measured \
  --summarizer stable_stage --debug
```

根据本次请求集为 prefix cache 留足空间，结合服务的 prefix cache 命中指标确认预热效果。多 DP 单元对各副本完成预热，并确认正式测量时请求能够命中对应副本的缓存。正式压测持续到能够获得有效稳态统计。

每个候选都从新启动的服务开始，完成预热后再正式测量。在成功率为 1、TPOT 达标的正式结果中，选择请求吞吐较高的候选，记录为 `QPS_D`，保留其服务参数和并发。

汇总两侧最终测量记录：

| 项目 | P 单元 | D 单元 |
|---|---|---|
| 设备型号 | 实际型号 | 实际型号 |
| 每单元机器数 / 卡数 | 实测部署占用 | 实测部署占用 |
| DP / TP / EP 等 | 实测配置 | 实测配置 |
| 批处理参数与测试并发 | 最终候选 | 最终候选 |
| 成功率 | 1 | 1 |
| 时延 | 达标的 TTFT | 达标的 TPOT |
| 请求吞吐 | `QPS_P`，req/s | `QPS_D`，req/s |

## 第 5 步：手动计算 PD 配比

按两侧处理能力平衡计算服务单元数量：

```text
P 单元数 × QPS_P ≈ D 单元数 × QPS_D
P:D ≈ QPS_D:QPS_P
候选组合的理论吞吐 ≈ min(P 单元数 × QPS_P, D 单元数 × QPS_D)
```

例如 P 单元为 30 req/s、D 单元为 10 req/s，理论配比为 **1P:3D**。这两个数仅用于演示计算。

异构部署分别核算两类设备：

```text
P 单元数 × 每 P 单元卡数 <= 可用于 P 的对应设备卡数
D 单元数 × 每 D 单元卡数 <= 可用于 D 的对应设备卡数
```

多机部署同时按完整服务单元核算机器数。例如每个 P、D 单元都使用 2 台 8 卡机器，1P:3D 对应 P 使用 2 台、D 使用 6 台，共 8 台、64 卡；每个单元内部均保持测量时的并行策略。

根据实际资源列出相邻整数配比，比较两侧合计处理能力。以下仍采用 30 / 10 req/s 的示例：

| 配比 | P 合计能力 | D 合计能力 | 理论吞吐 |
|---|---|---|---|
| 1P:2D | 30 req/s | 20 req/s | 20 req/s |
| 1P:3D | 30 req/s | 30 req/s | 30 req/s |
| 2P:3D | 60 req/s | 30 req/s | 30 req/s |

选择资源允许的组合，记录单元数量、总机器数、各类设备卡数、服务参数和理论吞吐，作为完整 PD 部署的初始配置。

## 第 6 步：部署完整 PD 服务并验证配比

使用现有 PD 部署工具，按第 5 步选择的数量部署 P、D 和 Proxy：

1. 各单元使用第 4 步确定的服务参数及固定并行策略。
2. 配置 PD 通信参数、KV cache 传输和 Proxy 路由。
3. 通过 Proxy 完成正常推理，再使用同一业务输入、目标输出长度和采样参数进行压测。

参考[通过 Proxy 压测完整 PD 服务](optix_pso_pd_disaggregation.md#第-5-步通过-proxy-压测完整-pd-服务)，复制完整服务压测配置，将 AISBench 模型短名和地址设置为实际 Proxy，在主机或压测机执行：

```bash
uv run --no-sync msmodeling optix --mode standard -e vllm -b ais_bench \
  -c ./pd_fine_tune.toml
```

该配置使用 `skip_pso=true`、`manage_simulator_lifecycle=false`，对已部署服务调整压测负载。按 TTFT、TPOT 和成功率筛选结果，比较完整链路吞吐；根据验证结果调整 P/D 单元数及并发，保留最终达标的部署配比和参数。
