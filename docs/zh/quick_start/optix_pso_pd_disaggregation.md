# OptiX PSO PD 分离寻优：vLLM + AISBench 操作指南

本文以 **DeepSeek-V4-Flash + vLLM + AISBench** 为例，介绍 PD 分离参数搜索、实例配比和完整服务压测。可以直接复制[搜索配置](../use_cases/examples/optix_pd_disaggregation/search.toml)和[完整服务压测配置](../use_cases/examples/optix_pd_disaggregation/fine_tune.toml)，按下面步骤修改。

操作顺序：**分别搜索 P（处理输入）和 D（生成后续 token）的参数 → 按结果部署 PD 服务与 Proxy → 通过 Proxy 压测完整服务**。已有可用 PD 服务、只需调整压测负载时，直接从[第 5 步](#第-5-步通过-proxy-压测完整-pd-服务)开始。

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
