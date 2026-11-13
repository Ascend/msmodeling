# OptiX PSO PD 分离寻优：vLLM + AISBench 操作指南

本文以 **DeepSeek-V4-Flash + vLLM + AISBench** 为例，介绍 PD 分离参数搜索、实例配比和完整服务压测。可以直接复制[搜索配置](../use_cases/examples/optix_pd_disaggregation/search.toml)和[完整服务压测配置](../use_cases/examples/optix_pd_disaggregation/fine_tune.toml)，按下面步骤修改。

在异构设备上分别搜索 P/D 参数，或使用多机 PD 混部插件测量服务单元并手动计算配比时，参见[分别搜索 P/D 参数并计算部署配比](optix_pso_pd_ratio_separate_search.md)。

操作顺序：**分别搜索 P（处理输入）和 D（生成后续 token）的参数 → 按结果部署 PD 服务与 Proxy → 通过 Proxy 压测完整服务**。已有可用 PD 服务、只需调整压测负载时，直接从[第 5 步](#第-5-步通过-proxy-压测完整-pd-服务)开始。

## 一个命令运行完整流程

通过 `pd_disagg.full_service.enabled` 开关控制是否执行完整链路，默认 `false`。设为 `true` 后，同一条命令依次完成 P 搜索、D 搜索、配比推荐、完整 PD 服务部署和负载微调，中间无需人工部署或再次启动 OptiX。设为 `false` 时，只运行 P/D 搜索和配比推荐，跳过完整服务预检、部署和压测，其他完整服务配置可以保留。已有配置若省略该开关，现在也按关闭处理；需要完整流程时请显式补上 `enabled = true`。

```toml
[pd_disagg.full_service]
enabled = true  # false：只搜索 P/D 并推荐配比；true：继续部署完整 PD 并微调负载
```

开启时仍需填写下述完整流程模板中的负载搜索范围和部署信息。

首次准备时安装现有 PD 插件（重新 `uv sync` 后需重装）：

```bash
uv pip install --python .venv/bin/python -e contrib/optix/vllm_pd_simulator
```

复制 [完整流程配置模板](../use_cases/examples/optix_pd_disaggregation/full_pipeline.toml)，按环境填写搜索参数、模型、AISBench 短名、`cluster.hosts` 和 `vllm_pd.nodes` 中允许使用的设备、通信网卡及 Proxy 地址。模板按 24 卡、P/D 每实例 8 卡展示。路径从仓库根目录解析。

```bash
uv run --no-sync msmodeling optix --mode pd_disagg -c ./pd_full.toml
```

- 设备池使用 `role="auto"` 时，按推荐实例数自动分成 P/D；`prefill`、`decode` 可限制设备角色，必须留足对应卡数。支持留卡，受 `use_full_device` 控制。该流程要求扁平 `vllm_pd.nodes`，不接受旧的手工 group 布局。
- P/D 服务参数及各自固定启动参数自动传递到部署插件；两侧 TP×DP 必须与各自每实例卡数一致，当前插件要求 PP=1，DP>1 适用于 MoE 模型。
- 单侧模拟与最终部署需要不同的固定启动参数时，设置 `full_service.prefill_others` / `decode_others`。例如 D 单侧开启 prefix caching 模拟 Decode，而实际完整服务关闭缓存，可将最终 D 的完整 `others` 填入 `decode_others`，明确使用 `--no-enable-prefix-caching`。该字段整段替换对应角色继承的 `others`，需保留 TP/DP、量化等必要参数；搜索选中的服务参数仍会叠加且优先，未填写则保持原有继承方式。
- `full_service.target_field` 仅配置 `CONCURRENCY`、`REQUESTRATE` 和可选 `NUM_PROMPTS`，前两项必须存在；最后阶段固定配比与服务参数，复用同一套服务微调负载，不再跑 PSO。
- 完整服务阶段固定使用 `fine_tune_mode="pd_mixed"` 和 `use_request_rate_calibration=false`，在 `full_service.target_field` 的范围内联合调整并发与请求率。顶层请求率标定开关仍作用于 P/D 单侧搜索，不会将完整服务的并发固定到上限；需要固定完整服务并发时，将其 `CONCURRENCY` 的 `min`、`max`、`value` 设为同一值。
- 仅适用于真实 PD 节点的 vLLM 参数可放入 `full_service.prefill_additional_config` / `decode_additional_config`，合并到对应角色的 `--additional-config`；例如 `decode_additional_config = { recompute_scheduler_enable = true }`。单侧搜索不启用它们，已搜索参数仍优先。
- 最后阶段单独配置端到端 TTFT/TPOT 目标和 benchmark 负载，使用完整输出长度。支持 `vllm_benchmark`、`ais_bench`；工具自动绑定 Proxy 地址。AISBench 可复用 D 模型配置，要求单个显式模型、地址为字面量，结束后恢复该配置文件。
- `full_service.benchmark_run_count` 允许大于等于 `1` 的整数，默认 `2`：每个完整服务负载候选在同一套服务上先预热一轮，再取第二轮计分。配置为 N 时，前 N−1 轮预热，仅最后一轮计分；各轮候选负载参数不变，不因重复压测重启 P/D 或 Proxy。显式配置 `1` 时单轮直接计分，报告会注明未额外安排压测预热轮次。原先未填写该字段的完整服务配置会采用新的两轮默认值；单侧搜索默认值仍为 `1`。
- 完整服务 SLO 使用 `full_service.ttft_slo` / `tpot_slo` / `success_rate_slo`，应按模型、完整输入输出长度和业务要求设置。未填写 TTFT 目标时仍采用 `0.5` 秒，报告会注明使用默认值；模板显式填写的 `3.0` 秒仅是示例，不代表所有模型的通用标准。预热用于统一测量口径，实际是否消除首次运行影响仍需实测核对。
- SSH、模型、vLLM、CANN、KV Connector 以及 Proxy 脚本须在执行前就绪，部署插件负责服务启停与健康检查。所有资源应为本次测试专用；单侧搜索依然在当前控制端配置的设备上依次执行。

Proxy 地址预检用于排除空值和未指定地址，不能证明服务实际可达，也不限制 P/D 的监听网卡。P/D 启动模板仍使用 `--host 0.0.0.0`；节点的 `bind_ip` / `service_ip` 用于服务连接和路由，不会收窄 P/D 监听范围。本流程不自动配置接口认证，应在部署环境中通过网络隔离、访问控制或网关限定服务访问范围。静态端口检查会识别同一 SSH 执行上下文中的全网卡监听冲突，但不探测 DNS 别名或不同容器是否共享宿主机网络；这些情况仍需按实际网络拓扑分配端口。

完整服务拓扑分配并通过静态校验后，插件会在该阶段首次 SSH 操作前打印 `Configured PD deployment ports` 清单，包含 Proxy 服务端口、每个 P/D 进程的 SSH 地址与容器、HTTP 服务端口、KV 端口范围、配置中保留的 RPC 端口及 Ascend 基础端口；MoE 模型另列各组实际传给 vLLM 的 DP Coordinator 地址和端口。`configured_rpc` 是静态校验保留的节点 RPC 配置值，不代表启动模板一定启用该监听；`ASCEND_BASE_PORT` 仅为基础端口，不代表通信后端使用的全部端口。清单不包含后端动态选择的端口，也不探测远端实时占用；配置校验通过不等于端口空闲。部署前应在对应主机/容器的网络命名空间核对占用情况，例如使用 `ss -lntup`，必要时调整配置端口或处理已确认归属的残留服务。

插件已有启动前清理和正常退出、可捕获异常时的清理；SIGKILL、容器退出或主机故障可能使清理无法执行。当前没有 `--dry-run` 或 `--pd-cleanup-only` 入口。异常中断后，可结合本轮 `full_service/deployment.json`、生成的脚本和日志确认实际分配的节点、容器、卡号与端口，再定向排查和清理；不要将整个资源池都视为本轮已使用资源。

完整服务阶段若启动或压测失败，汇总和报告保留主流程的异常类型与原因；压测停止、配置恢复、服务停止各自独立尝试，失败项追加到 `full_service.cleanup_errors`，并在 `final_result.md` 的“清理异常（附加信息）”中逐项展示。若主流程成功而清理失败，整体仍标记失败，不输出成功的完整服务阶段结果。

开启 `enabled = true` 时，运行结束直接查看 **`output/pd_disagg/<run_id>/final_result.md`**，命令最后会打印该文件路径。报告在一个文件中集中展示：

- 完整流程状态，以及完整服务的吞吐、TTFT、TPOT、成功率和逐项 SLO 判定；
- 最终并发、请求率及压测命令覆盖，并发的配置范围、初始值及最终实测值来源；
- 最终选中的 P/D 服务参数、单侧负载与指标、理论配比及实际实例数；
- 实际机器、容器、物理卡、Proxy 地址与完整部署配置（内嵌 JSON，可直接核对和提取）。

资源统计分别展示设备池总卡数、`total_devices` 预算、已分配卡数、预算内余量和池内闲置。设备池允许大于预算：例如池内 32 卡、预算 24 卡、实际分配 24 卡时，预算内余量为 0，池内闲置为 8。总量按分配前设备池统计，不包含 Proxy 角色的设备；旧部署记录未保存池总量时，报告显示未知，不用已分配设备反推池总量。

当前未采集单侧搜索与部署节点的卡型号，也未自动关联控制端与 `cluster.hosts` 的主机标识，报告不保证硬件一致。单侧搜索在控制端配置的设备上运行；将参数用于不同型号、显存容量或带宽的部署机器前，应另行核对其适用性，不能仅凭完整服务 SLO 结果判断参数失配原因。

`final_result.md` 用于在部署环境中核对结果，保留原始信息，未自动脱敏。机器地址、容器标识、模型与输出路径、启动参数及环境配置可能出现在部署表、内嵌 JSON、压测配置、错误信息和追溯路径中；对外分享至 Issue、PR 评论或群聊前，请检查整份报告并脱敏。原始 `deployment.json` 和生成的启动脚本也应按相同原则处理。

若流程中途失败，同一报告会记录失败阶段、错误和已保存的阶段结果，不会把缺失指标标记为达标。`status="completed"` 表示流程完成，`full_service.slo_met` 单独表示性能目标是否全部达到。

原有 `pd_disagg_summary.json`、`full_service/deployment.json`、各阶段 `phase_result.json` 和 CSV 继续保留用于追溯，日常查看最终结果无需逐个打开。`enabled = false` 或未配置完整服务时不生成这份报告，分步模式的状态、文件和查看方式保持不变。完整服务结束后自动停止。

## 第 1 步：确认示例配置，准备环境

本例使用以下模型、服务参数和压测负载：

| 项目 | 本例使用的值 |
|---|---|
| 模型权重目录 | `/dl/weight/DeepSeek-V4-Flash-w8a8-mtp` |
| 对外模型名 / 搜索端口 | `dsv4` / `8900` |
| 单实例并行策略 | DP=1、TP=8、开启 EP；P 和 D 均使用该策略 |
| 固定服务参数 | Ascend 量化、MTP 3 token、最大模型长度 196608、显存利用率 0.9、block size 128；图编译和 additional-config 配置见搜索模板中的 `others` |
| P 服务参数 | `MAX_NUM_BATCHED_TOKENS`：8192～65536，初始值 8192；`MAX_NUM_SEQS`：24 或 32，初始值 32 |
| D 服务参数 | `MAX_NUM_BATCHED_TOKENS`：4096 或 8192，初始值 8192；`MAX_NUM_SEQS`：32～512，初始值 64 |
| 并发候选 | P、D 均为 20、24、28、32，初始值 24 |
| 测试请求数 | 并发 × 4，即 80、96、112、128；初始值 96 |
| 请求率 | `REQUESTRATE=0`，不限发送速率 |
| 数据集 | `gsm8k_gen_0_shot_cot_str_perf` |
| 搜索规模 | `n_particles=10`、`iters=4`，P、D 分别执行 |
| 时延目标 | P 的首 token 时延 3 秒；D 的每个输出 token 时延 0.05 秒（50 毫秒） |

**196608 是服务允许的最大模型长度，不是每条 GSM8K 请求的输入长度。** 本例使用 GSM8K 真实数据集。

本文用 **24 卡、每个 P/D 实例各 8 卡**演示实例配比，24 需要替换为实际可用的总卡数。

### 1.1 准备 vLLM 部署环境

在支持 DSv4 的 **Linux 服务器或 vLLM-Ascend 容器**中操作，准备好昇腾驱动、CANN、vLLM-Ascend 和模型权重。先检查：

```bash
npu-smi info
python3 --version
command -v vllm
vllm serve --help
```

应能看到 NPU 设备、Python 版本和 vLLM 帮助信息。下面安装 AISBench 使用的 `python3` 须为 Python 3.10～3.12。若 vLLM 安装在独立虚拟环境，先进入该部署环境；不要进入 msmodeling 的 `.venv`。

本例包含量化、MTP 和图编译设置，部署环境中的 vLLM-Ascend 需支持这些特性及模板中的启动参数。

### 1.2 安装 AISBench

在上述部署环境中执行 `ais_bench --help`。已能显示帮助信息时，直接进入 1.3；尚未安装时，为 AISBench 新建独立环境，避免改动 vLLM 的依赖：

```bash
mkdir -p ~/optix-workspace
python3 -m venv ~/optix-workspace/aisbench-env
cd ~/optix-workspace
git clone https://github.com/AISBench/benchmark.git
cd benchmark
~/optix-workspace/aisbench-env/bin/python -m pip install -e . --use-pep517
~/optix-workspace/aisbench-env/bin/python -m pip install -r requirements/api.txt -r requirements/extra.txt
export PATH="$HOME/optix-workspace/aisbench-env/bin:$PATH"
ais_bench --help
```

源码安装后保留 `benchmark` 目录，第 2 步需要修改其中的模型和数据集配置。

### 1.3 安装 OptiX

OptiX 随 msmodeling 安装，使用仓库自己的 `.venv`，与上述 vLLM、AISBench 部署环境分开，避免依赖版本相互影响。

如果还没有 `uv`，先安装：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version
```

还没有 msmodeling 仓库时，执行：

```bash
mkdir -p ~/optix-workspace
cd ~/optix-workspace
git clone https://gitcode.com/Ascend/msmodeling.git
cd msmodeling
```

已有仓库则直接进入自己的仓库根目录。在该目录安装并验证：

```bash
uv sync --python 3.12
uv run msmodeling optix --help
```

`uv sync` 会创建仓库内的 `.venv` 并安装 OptiX。显示 OptiX 帮助信息后，继续下面的部署命令检查。**不要把 vLLM、AISBench 安装到这个 `.venv` 中。**

### 1.4 确认 OptiX 能找到部署命令

OptiX 启动服务和压测时，会从环境中剥离 msmodeling 的 `.venv`，再查找 `vllm`、`ais_bench`。因此，仅在当前 shell 中执行 `--help` 成功还不够。

如果两条命令位于系统 PATH，例如 `/usr/local/bin/`，不用额外设置。若安装在独立环境，例如 `/opt/vllm-env/bin/`，将下方目录替换为实际部署环境根目录，再执行：

```bash
export OPTIX_DEPLOY_PATH=/opt/vllm-env
export PATH="$OPTIX_DEPLOY_PATH/bin:$PATH"
```

`OPTIX_DEPLOY_PATH` 填 `/opt/vllm-env`，不要填到 `bin`。第二行让当前 shell 也能找到 vLLM。AISBench 若按 1.2 独立安装，其 `bin/` 也要保留在 PATH 中；OptiX 会继续从 PATH 中查找 `ais_bench`。后续操作继续使用本终端，新开终端时需重新设置这些环境变量。

回到 msmodeling 仓库根目录，按 OptiX 实际查找方式检查；这一步只检查命令路径，不启动服务：

```bash
uv run python - <<'PY'
from optix.deploy_env import resolve_deploy_context, validate_deploy_stack
ctx, env = resolve_deploy_context()
validate_deploy_stack(engine="vllm", benchmark="ais_bench", env=env, ctx=ctx)
PY
```

日志中应分别显示 `deployment command vllm`、`deployment command ais_bench` 的路径，且均位于 msmodeling 的 `.venv` 之外。报 `Deployment command not found` 时，检查部署环境是否安装了对应命令，以及 `OPTIX_DEPLOY_PATH` 是否填对。

下面的 OptiX 命令均在仓库根目录执行。AISBench 的 Python 配置放在**部署环境中的 AISBench 包目录**；源码安装时，就是上面保留的 `benchmark` 仓库中的对应配置目录。

## 第 2 步：准备 P、D 的 AISBench 配置

### 2.1 复制 AISBench 自带的模型配置

先在部署环境中执行：

```bash
ais_bench --models vllm_api_stream_chat --datasets gsm8k_gen_0_shot_cot_str_perf --search
```

输出会显示模型和数据集 Python 配置的实际路径。进入其中 `vllm_api_stream_chat.py` 所在目录，复制三份：

```bash
cp vllm_api_stream_chat.py optix_dsv4_prefill.py
cp vllm_api_stream_chat.py optix_dsv4_decode.py
cp vllm_api_stream_chat.py optix_dsv4_proxy.py
```

这三份分别用于 P、D 和最后的 Proxy 压测。API 类型使用所复制文件中的定义，下面分别修改服务地址和压测负载。

### 2.2 修改新文件中的字段

打开新文件，在 `models` 列表的模型配置中修改下列字段。`model` 是**服务对外模型名**，`path` 是**分词器/模型目录**，两者不同。

| 字段 | `optix_dsv4_prefill.py` | `optix_dsv4_decode.py` | `optix_dsv4_proxy.py` |
|---|---|---|---|
| `abbr` | `optix-dsv4-prefill` | `optix-dsv4-decode` | `optix-dsv4-proxy` |
| `path` | `/dl/weight/DeepSeek-V4-Flash-w8a8-mtp` | 同左 | 压测机可读取的同一模型目录 |
| `model` | `dsv4` | `dsv4` | Proxy 对外模型名，本例为 `dsv4` |
| `host_ip` | `127.0.0.1` | `127.0.0.1` | 实际 Proxy 地址 |
| `host_port` | `8900` | `8900` | 实际 Proxy 端口 |
| `max_out_len` | **改为 `1`** | 按本次压测目标设置输出长度 | 与 D 配置保持一致 |
| `batch_size` | `24` | `24` | `24` |
| `request_rate` | `0` | `0` | `0` |

`batch_size`、`request_rate` 要写成 `batch_size=24,`、`request_rate=0,` 这样的直接赋值形式，保留末尾逗号。OptiX 会按当前候选的 `CONCURRENCY` 和 `REQUESTRATE` 更新它们。若文件中的 `url` 非空，也要更新到对应端点。其余字段使用所复制文件中的值，可参考 [AISBench 官方模型配置示例](https://github.com/AISBench/benchmark/blob/master/ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py)。

**D 的两轮测评必须使用相同输入请求**，保持数据集、顺序和请求数一致；有随机抽样时先固定请求集。服务的 `--seed` 不能代替这一步。D 和 Proxy 使用相同的输出长度和采样参数，按本次压测目标设置。

GSM8K 数据集 Python 配置中的 `path` 应指向可用的数据目录，首次使用时也要核对；可参考 [AISBench 官方 GSM8K 配置](https://github.com/AISBench/benchmark/blob/master/ais_bench/benchmark/configs/datasets/gsm8k/gsm8k_gen_0_shot_cot_str_perf.py)。

最后检查新短名能否解析：

```bash
ais_bench --models optix_dsv4_prefill --datasets gsm8k_gen_0_shot_cot_str_perf --search
ais_bench --models optix_dsv4_decode --datasets gsm8k_gen_0_shot_cot_str_perf --search
ais_bench --models optix_dsv4_proxy --datasets gsm8k_gen_0_shot_cot_str_perf --search
```

`--models` 填文件名去掉 `.py` 后的短名，不填绝对路径。`--search` 只检查能否找到配置，正式搜索前还应确认这些配置能正常发请求。

## 第 3 步：复制 PD 分离模板，运行搜索

```bash
cp docs/zh/use_cases/examples/optix_pd_disaggregation/search.toml ./pd_search.toml
```

模板已包含本例的模型、主要启动参数和 P/D 各自完整的搜索字段。主要核对以下位置：

| 配置位置 | 需要核对的内容 |
|---|---|
| `[pd_disagg.prefill.simulator_command_overrides]` | P 的模型路径、`served_model_name="dsv4"`、`port="8900"` 与 AISBench 的 P 配置一致 |
| `[pd_disagg.decode.simulator_command_overrides]` | D 的模型路径、模型名、端口与 AISBench 的 D 配置一致 |
| `[pd_disagg] total_devices` | 将示例 `24` 改成最终部署可用的总卡数 |
| `[pd_disagg]` 的两个 `devices_per_instance` | 本例 P/D 都用 8 卡，与模板中的 TP=8 对应 |
| `[pd_disagg.prefill.benchmark_command_overrides] models` | P 使用第 2 步创建的 `optix_dsv4_prefill` |
| `[pd_disagg.decode.benchmark_command_overrides] models` | D 改用 `optix_dsv4_decode` |

**P、D 的服务启动参数和压测配置已分别写在各自阶段下。** 修改 P 就改 `pd_disagg.prefill` 下的配置，修改 D 就改 `pd_disagg.decode` 下的配置。文件开头的 `[vllm.command]`、`[ais_bench.command]` 用于插件初始化，保持与 P 的模型、地址和短名一致。

**每个阶段的 `target_field` 都是一份完整列表，不会自动合并公共搜索字段。** 模板已在 P、D 下分别写全 `MAX_NUM_BATCHED_TOKENS`、`MAX_NUM_SEQS`、`CONCURRENCY`、`NUM_PROMPTS`、`REQUESTRATE`。

每个参数使用一个 `[[pd_disagg.<阶段>.target_field]]` 块。`min`、`max` 是搜索上下限，`dtype_param` 给出枚举候选或联动规则，`value` 是初始值。两个服务参数使用 `config_position="run"`，压测参数使用 `env`；`NUM_PROMPTS` 通过 `times` 按并发 × 4 计算。新增参数时，在对应阶段追加同样格式的块，保留其他已有字段。

模板已配置好两阶段的区别：P 输出 1 个 token、按 TTFT 评分；D 开启 prefix caching，连续测两轮，第一轮预热、第二轮按 TPOT 评分。`top_k=0` 省去候选微调，仍验证最优结果并推荐配比。

若模型或引擎在 D 单侧场景下不支持 prefix caching，应关闭该单侧缓存；关闭后 D 单侧吞吐包含 Prefill 成本。完整 PD 对应角色的缓存设置独立按实际部署要求填写，可通过 `full_service.decode_others` 与单侧区分；单侧容量用于配比估计，最终是否达标由完整服务实测确认。使用 AISBench `stable_stage` 时，请求数需大于并发数，并提供足够的后续补入请求，避免短测试没有有效稳态区间。

两阶段的 `others` 保留模型启动参数，两个批处理参数由各自的搜索字段生成。`others` 会**替换整个启动参数字符串**，修改时要保留该阶段所需的完整参数。

搜索时需要一组可用的 8 卡，P/D 依次运行。`total_devices` 只参与最终配比计算，不会分配具体卡号。需要限制昇腾卡号时，可在文件末尾添加：

```toml
[vllm.env]
ASCEND_RT_VISIBLE_DEVICES = "0,1,2,3,4,5,6,7"
```

保存后运行：

```bash
uv run msmodeling optix --mode pd_disagg -e vllm -b ais_bench -c ./pd_search.toml
```

本例的 `10 × 4` 搜索规模会分别用于两阶段，另外还有基线、验证和 D 缓存预热开销。

结果位于 `result/pd_dsv4/pd_disagg/<run_id>/`，`<run_id>` 换成实际目录名。先打开这三份文件：

| 文件 | 看什么 |
|---|---|
| `pd_disagg_summary.json` | `status` 是否为 `service_search_completed`；`instances.prefill/decode` 建议各部署几个实例 |
| `prefill/phase_result.json` | P 的参数、实际 TTFT、成功率和请求吞吐 |
| `decode/phase_result.json` | D 的参数、实际 TPOT、成功率和请求吞吐 |

搜索完成后，还要确认 P 的 TTFT 是否满足 3 秒、D 的 TPOT 是否满足 0.05 秒，且请求成功率为 1。完成状态不代表时延已经达标。

**卡数不足时，理论配比与实际部署建议会不同。** 假设单个 P 为 30 req/s、单个 D 为 10 req/s（仅用于演示，非 DSv4 实测）：

| 对比项 | 理论平衡方案 | 当前资源下的推荐 |
|---|---|---|
| 实例数量 | 1P + 3D | 1P + 2D |
| 所需卡数（每实例 8 卡） | 32 卡 | 24 卡 |
| 两侧处理能力 | P 为 30、D 合计 30 req/s | P 为 30、D 合计 20 req/s，D 成为瓶颈 |

summary 中的 `pd_ratio=1/3` 表示理论比例；`instances.prefill=1`、`instances.decode=2` 表示当前资源下建议部署 **1P + 2D**。结果保留两种比例，但不会自动生成“理论需要 32 卡、当前只有 24 卡”的文字说明。

推荐优先选择最接近理论比例的可行组合。完整服务的实际吞吐仍需第 5 步验证。

## 第 4 步：根据结果部署 P、D 和 Proxy

使用当前环境的 PD 部署方案，配置 P/D 实例、KV cache 传输和 Proxy。OptiX 搜索结果提供参数与数量，完整集群仍由部署工具启动。

**KV cache 传输说明：** 第 3 步的示例通过 PD 混部的单实例分别模拟 P、D 阶段，不涉及 P/D 实例间的 KV cache 传输，因此搜索模板未包含相关传输参数。手动拉起完整 PD 服务时，需要根据当前 vLLM-Ascend 版本和 PD 部署方案，在 P、D 各实例的启动命令中补充 KV cache 传输相关参数，并与 Proxy 的路由配置配套使用。

| 部署项 | 怎么填写 |
|---|---|
| P/D 实例数 | 按 summary 的 `instances.prefill`、`instances.decode` |
| 单实例固定参数 | 保留本次搜索使用的 TP=8、量化、MTP 等参数，并加入部署方案要求的 PD 通信参数 |
| 批处理参数 | 分别使用 P、D 搜索结果中的 `MAX_NUM_BATCHED_TOKENS` 和 `MAX_NUM_SEQS`；其他参数沿用该阶段的启动配置 |
| 卡号和端口 | 给各实例分配设备和独立端口，并在 Proxy 中配置路由 |

例如 P 的 `MAX_NUM_BATCHED_TOKENS=16384` 对应 `--max-num-batched-tokens 16384`；D 的 `MAX_NUM_SEQS=64` 对应 `--max-num-seqs 64`。`CONCURRENCY`、`NUM_PROMPTS`、`REQUESTRATE` 用于压测，不加入 vLLM 启动命令。完整服务的并发还需第 5 步通过 Proxy 验证。

若推荐为 `1P + 2D`，可为 P1 分配 0～7 号卡、D1 分配 8～15 号卡、D2 分配 16～23 号卡。此卡号分配仅作演示。

先通过 Proxy 完成一次正常推理，再继续压测。例如 Proxy 对外模型名仍为 `dsv4`，可替换下面的示例地址后验证：

```bash
curl http://192.0.2.10:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"dsv4","messages":[{"role":"user","content":"你好"}],"max_tokens":16}'
```

## 第 5 步：通过 Proxy 压测完整 PD 服务

即使复用已部署的服务，本步启动时仍会检查 `vllm`、`ais_bench` 是否可用，需要先完成第 1 步的部署命令检查。

打开第 2 步创建的 `optix_dsv4_proxy.py`，将 `host_ip`、`host_port` 改为 Proxy 的 IP 和端口，`model` 填写 Proxy 对外提供的模型名（本例为 `dsv4`）。如果该文件中的 `url` 非空，也要更新为 Proxy 的实际接口地址。

`path` 填写压测机可读取的同一模型/分词器目录。`max_out_len` 和采样参数与第 2 步的 D 配置保持一致。

**AISBench 的压测地址由这个 Python 文件决定，修改 TOML 中的 `[vllm.command] host/port` 不会自动同步到 AISBench。**

然后复制配置并运行：

```bash
cp docs/zh/use_cases/examples/optix_pd_disaggregation/fine_tune.toml ./pd_fine_tune.toml
uv run msmodeling optix --mode standard -e vllm -b ais_bench -c ./pd_fine_tune.toml
```

本步使用已部署的服务，跳过 PSO，从并发 24、请求数 96 开始微调。请求率固定为 `0`，即**不限发送速率**。本步的 `CONCURRENCY.dtype` 为 `int`，在 20～32 的整数范围内调整，最多微调 10 次；第 3 步则使用 20、24、28、32 四个并发候选。两步的请求数都随并发按 4 倍联动。本步检查 TTFT 和 TPOT，没有搜索有限请求率。

在 `result/pd_dsv4_fine_tune/store/data_storage_*.csv` 中查看实测行（`real_evaluation=true`）：

| 列名 | 本例如何判断 |
|---|---|
| `success_rate` | 等于 1 |
| `time_to_first_token(s)` | 不超过 3 秒 |
| `time_per_output_token(s)` | 不超过 0.05 秒 |
| `throughput(requests/s)` | 在成功率与时延都达标的实测行中，选择吞吐较高的行 |
| `CONCURRENCY`、`NUM_PROMPTS` | 记录这行对应的并发与请求数 |

若并发降到 20 仍不达标，当前范围内还没有找到合适负载。可另行扩大并发范围或配置有限请求率，具体方法见[PD 分离寻优详细说明](../use_cases/optix_pso_pd_disaggregation.md)。

## 常见问题

| 现象 | 先检查 |
|---|---|
| 找不到部署命令，或命令解析到 msmodeling 虚拟环境 | 按第 1 步确认命令安装在 msmodeling 的 `.venv` 之外；独立环境将 `OPTIX_DEPLOY_PATH` 设为包含 `bin/` 的环境根目录，再执行 1.4 的检查命令 |
| 找不到新的模型短名 | 文件是否放在 AISBench 可搜索的模型配置目录，文件名是否唯一；先执行 `--search` |
| 请求发到了旧地址或模型名错误 | 检查模型 Python 文件中的 `host_ip`、`host_port`、`model` 和非空 `url` |
| 服务参数修改没有生效 | 检查对应阶段的搜索字段是否为 `config_position="run"`，以及日志中的实际启动命令 |
| D 的预热没有效果 | 确认启用 prefix caching；两轮输入、顺序、请求数一致，且缓存容量足够 |
| D 的稳态统计失败或没有有效数据 | 核对 AISBench 版本和测试时长；80～128 个请求不保证足够形成稳态窗口，必要时增大 D 的 `NUM_PROMPTS.dtype_param.product` 后重新验证 |
| `instances` 为空 | 查看 `allocation_warning`；核对总卡数是否能由每实例 8 卡组合，允许留卡时可设 `use_full_device=false` |
