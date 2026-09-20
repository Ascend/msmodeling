# PD 分离（Prefill / Decode Disaggregation）仿真示例

本目录提供一个开箱即用的 **PD 分离** 服务化仿真案例：Prefill 与 Decode 使用相互独立的实例池和资源切分，
请求先在 Prefill 池完成首 token 计算，再把 KV Cache 传输到 Decode 池继续增量解码。

示例目标是让使用者用一条命令跑通一次完整的 PD 分离仿真，并能看懂每个配置项对结果的影响。

## 1. 目录内容

| 文件 | 作用 |
| --- | --- |
| `instances.yaml` | PD 分离实例拓扑：Prefill / Decode 两个资源池、并行切分、通信带宽 |
| `common.yaml` | 模型配置、请求负载配置、服务调度配置 |
| `run_pd_disaggregation.sh` | 一键运行脚本，封装 `python -m serving_cast.main` 命令 |
| `README.md` | 本说明文档 |

作为对照，`serving_cast/example/` 根目录下的 `instances.yaml` / `common.yaml` 是 **PD 融合**
（`pd_role: both`）的单实例示例，可与本示例结果横向对比。

## 2. 快速开始

前置条件：已按仓库标准方式安装依赖（`uv sync --group ci`），且首次运行可访问模型 Hub 以拉取
`Qwen/Qwen3-32B` 的 `config.json`（不下载权重）。

在仓库根目录执行：

```bash
bash serving_cast/example/pd_disaggregation/run_pd_disaggregation.sh
```

本示例共有 3 个实例（1 个 Prefill + 2 个 Decode），每个实例都要先做一次性能预热与插值采样，
因此首次运行会持续较长时间（分钟级到十几分钟，取决于机器性能），过程中打印的
`No op properties function defined for ...`、`Negative activation memory estimate ...`
等信息属于性能建模的正常提示，不影响结果。

脚本等价于下面这条命令，在没有 bash 的环境（例如 Windows PowerShell）中可直接使用：

```bash
python -m serving_cast.main \
  --instance_config_path=./serving_cast/example/pd_disaggregation/instances.yaml \
  --common_config_path=./serving_cast/example/pd_disaggregation/common.yaml \
  --output_json=./pd_disaggregation_summary.json
```

脚本支持的可选环境变量：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `PD_INSTANCE_CONFIG` | 本目录 `instances.yaml` | 替换实例拓扑配置 |
| `PD_COMMON_CONFIG` | 本目录 `common.yaml` | 替换模型 / 负载 / 服务配置 |
| `PD_OUTPUT_JSON` | 仓库根目录 `pd_disaggregation_summary.json` | 汇总结果 JSON 输出路径 |
| `PYTHON` | 有 `uv` 时为 `uv run python`，否则 `python3` | 指定解释器 |

脚本额外参数会透传给 `serving_cast.main`，例如开启 profiling：

```bash
bash serving_cast/example/pd_disaggregation/run_pd_disaggregation.sh --enable_profiling
```

## 3. PD 分离是怎么被识别的

`serving_cast` 不需要显式的 “开启 PD 分离” 开关，部署形态由 `instances.yaml` 中出现的 `pd_role` 组合推断
（见 `serving_cast/main.py` 的 `instance_group2pd_type`）：

| `pd_role` 组合 | 推断结果 | 使用的 Serving 实现 |
| --- | --- | --- |
| 只有 `both` | `pd_aggregation` | `PdAggregationServing` |
| 同时有 `prefill` 和 `decode`，且没有 `both` | `pd_disaggregation` | `PdDisaggregationServing` |
| 其它组合 | 非法 | 抛出 `ValueError("check instance's pd_role")` |

因此 **PD 分离的充要条件是：`prefill` 与 `decode` 两组同时存在，且不能混入 `both`**。

请求在 PD 分离下的完整流转：

1. 客户端按 `request_rate` 逐个尝试发送请求：获得并发槽后离开客户端（`LEAVES_CLIENT`）并进入
   Serving（`ARRIVES_SERVER`）；服务端并发（`max_concurrency`）已满时，请求停留在客户端排队，
   各项客户端计时指标从真正获得并发槽的时刻起算；
2. `PdDisaggregationServing` 把 `need_kv_transfer` 置为 `True`，通过负载均衡器选中一个 Prefill 实例；
3. Prefill 实例完成 prefill 后进入 `KVS_TRANSFERRING` 状态，按 `device2device_bandwidth`
   与 `device2device_rate` 计算 KV Cache 传输时延；
4. 传输完成后回调选出一个 Decode 实例，继续增量解码直至完成。

> 注意：`common.yaml` 中的 `enable_kv_transfer_modeling` 必须为 `True`，否则 KV 传输时延按 0 计算，
> PD 分离相对 PD 融合的额外开销将无法体现。

## 4. 配置字段说明

### 4.1 `instances.yaml`

顶层是 `instance_groups` 列表，本示例包含 Prefill 与 Decode 两组。

| 字段 | 本示例取值 | 说明 |
| --- | --- | --- |
| `num_instances` | Prefill 1 / Decode 2 | 该组的实例个数；两组的比值即 **P:D 配比** |
| `num_devices_per_instance` | Prefill 4 / Decode 2 | 单实例占用的设备数，需与 `parallel_config.world_size` 一致 |
| `device_type` | `TEST_DEVICE` | 设备画像名，取值来自 `tensor_cast` 的 `DeviceProfile` 注册表 |
| `pd_role` | `prefill` / `decode` | 该组承担的阶段，可选 `prefill`、`decode`、`both` |
| `parallel_config.world_size` | 4 / 2 | 实例内并行总规模，等于 `tp_size * dp_size` |
| `parallel_config.tp_size` | 4 / 2 | 张量并行度 |
| `parallel_config.dp_size` | 1 / 1 | 数据并行度，每个 DP rank 对应一个独立调度的 Engine |
| `parallel_config.mlp_tp_size` / `mlp_dp_size` | `null` | MLP 单独切分，`null` 表示复用 `tp_size` / `dp_size` |
| `parallel_config.lmhead_tp_size` / `lmhead_dp_size` | `null` | LM Head 单独切分，`null` 表示复用 `tp_size` / `dp_size` |
| `parallel_config.ep_size` | 1 | 专家并行度，稠密模型保持 1 |
| `parallel_config.moe_tp_size` / `moe_dp_size` | 4 与 1 / 2 与 1 | MoE 切分，必须满足 `moe_tp_size * moe_dp_size * ep_size == world_size` |
| `communication_config.host2device_bandwidth` | 10 GB/s | 输入预处理阶段 Host → Device 拷贝带宽（字节/秒） |
| `communication_config.host2device_rate` | 0.5 | Host → Device 有效带宽折算系数 |
| `communication_config.device2device_bandwidth` | 4 GB/s | **Prefill → Decode 的 KV Cache 传输带宽**，PD 分离的关键参数 |
| `communication_config.device2device_rate` | 0.5 | KV 传输有效带宽折算系数 |

本示例的资源划分思路：Prefill 计算密集，用较大 TP（TP=4）压低 TTFT；Decode 访存密集且并发度高，
用较小 TP（TP=2）配多实例提升整体吞吐。两组合计 8 卡，对应单机 8 卡部署。

### 4.2 `common.yaml`

| 字段 | 本示例取值 | 说明 |
| --- | --- | --- |
| `model_config.name` | `Qwen/Qwen3-32B` | 模型标识，可换成本地模型目录路径 |
| `model_config.quantize_linear_action` | `W8A8_DYNAMIC` | 线性层量化策略，直接影响权重显存与算力需求 |
| `model_config.predict_steps` | 5 | 插值采样点数量，越大越精确、预热越慢 |
| `model_config.enable_interpolate` | `True` | 开启插值，避免逐 step 实测，显著缩短仿真时间 |
| `model_config.interpolation_seed` | 1234 | 插值采样随机种子，固定后结果可复现 |
| `model_config.enable_preprocessing_modeling` | `True` | 是否建模输入预处理开销 |
| `model_config.enable_kv_transfer_modeling` | `True` | **PD 分离必须开启**，否则 KV 传输时延为 0 |
| `load_gen.load_gen_type` | `fixed_length` | 定长负载生成器 |
| `load_gen.num_requests` | 32 | 请求总数，调大统计更平稳，同时线性拉长仿真时间 |
| `load_gen.num_input_tokens` | 512 | 单请求输入长度，主要影响 Prefill 池压力与 TTFT |
| `load_gen.num_output_tokens` | 128 | 单请求输出长度，主要影响 Decode 池压力与 TPOT |
| `load_gen.request_rate` | 4.0 | 请求到达速率（req/s），泊松到达 |
| `serving_config.max_concurrency` | 8 | 全局在服并发上限，超过后新请求排队（同时决定预热采样批数） |
| `serving_config.block_size` | 128 | KV Cache 分页大小（token/block） |
| `serving_config.max_tokens_budget` | 1024 | 单次调度批的 token 预算上限（同时决定预热采样批数） |

## 5. 预期输出

运行结束后 stdout 会打印两块内容。第一块是分位数指标表（数值随环境与配置变化，此处仅示意形态）：

```text
         E2E_TIME(s)  TTFT(s)  TPOT(s)  INPUT_TOKENS  OUTPUT_TOKENS  OUTPUT_TOKEN_THROUGHPUT(tok/s)
AVERAGE        ...      ...      ...         512.0          128.0                              ...
MIN            ...      ...      ...         512.0          128.0                              ...
MAX            ...      ...      ...         512.0          128.0                              ...
MEDIAN         ...      ...      ...         512.0          128.0                              ...
P75            ...      ...      ...         512.0          128.0                              ...
P90            ...      ...      ...         512.0          128.0                              ...
P99            ...      ...      ...         512.0          128.0                              ...
```

第二块是整体汇总：

```text
======== Overall Summary ========
benchmark_duration(s)          ...
total_requests                 32.000
request_throughput(req/s)      ...
total_input_tokens             16384.000
input_token_throughput(tok/s)  ...
total_output_tokens            4096.000
output_token_throughput(tok/s) ...
```

指标含义：

- `E2E_TIME(s)`：端到端时延，`decode_done_time - leaves_client_time`；
- `CLIENT_TTFT(s)`：客户端首 token 时延，`prefill_done_time - leaves_client_time`；
- `SERVER_TTFT(s)`：服务端首 token 时延，`prefill_done_time - arrives_server_time`，PD 分离下由
  Prefill 池排队与算力决定；
- `ADMISSION_WAIT(s)`：离开客户端到到达服务端的时延（仅传输时延，仿真中近似为 0）；
- `TPOT(s)`：单输出 token 时延，由 Decode 池决定，同时包含 KV 传输带来的起步延迟；
- `OUTPUT_TOKEN_THROUGHPUT(tok/s)`：单请求维度的出词速率；
- `Overall Summary`：基于首个请求离开客户端到最后一个请求解码完成的墙钟跨度计算的整体吞吐。

> 客户端并发语义（与 AIPerf 对齐）：`request_rate` 的每个时间点表示一次“尝试发送”；
> 服务端并发已满时请求停留在客户端排队，只有真正获得并发槽后才记录 `leaves_client_time`
> 与 `arrives_server_time`。因此 `CLIENT_TTFT`、`ADMISSION_WAIT`、`E2E_TIME` 的起点均为
> 实际发送时刻，不含客户端排队时间。

同时会生成 `--output_json` 指定的 JSON 文件，结构为：

```json
{
  "per_metric_summary": { "E2E_TIME(s)": { "AVERAGE": 0.0, "MIN": 0.0 } },
  "overall_summary": { "benchmark_duration(s)": 0.0, "total_requests": 32.0 }
}
```

## 6. 常见调整

| 想验证什么 | 改哪里 |
| --- | --- |
| P:D 配比（如 1P4D、2P2D） | `instances.yaml` 两组的 `num_instances` |
| Prefill / Decode 并行策略 | 对应组的 `parallel_config`，同步修改 `num_devices_per_instance` 与 `world_size` |
| KV 传输开销敏感性 | `communication_config.device2device_bandwidth` / `device2device_rate` |
| 负载形态（长输入 / 长输出） | `common.yaml` 的 `num_input_tokens` / `num_output_tokens` |
| 压力大小 | `common.yaml` 的 `request_rate`、`num_requests`、`max_concurrency` |
| 换硬件 | `instances.yaml` 的 `device_type`，取值需为已注册的 `DeviceProfile` 名称 |
| 与 PD 融合对比 | 直接运行 `serving_cast/example/` 根目录下的配置作为基线 |

调整并行配置时务必同时满足：

- `world_size == tp_size * dp_size == num_devices_per_instance`
- `moe_tp_size * moe_dp_size * ep_size == world_size`

## 7. 常见失败原因

| 现象 | 原因 | 处理方式 |
| --- | --- | --- |
| `ValueError: check instance's pd_role` | `prefill` / `decode` 没有同时出现，或混入了 `both` | 保证两组同时存在且不含 `both` |
| `ValueError: <role> is not supported` | `pd_role` 拼写错误 | 只能取 `prefill`、`decode`、`both` |
| `ModuleNotFoundError`（如 `salabim`） | 未安装仿真依赖 | 在仓库根目录执行 `uv sync --group ci` |
| 拉取模型 config 超时或失败 | 无法访问模型 Hub | 配置 Hub 镜像，或把 `model_config.name` 改为本地模型目录 |
| 设备画像找不到 | `device_type` 未在 `DeviceProfile` 注册表中 | 使用已注册名称，或参考 `tensor_cast/device_profiles/` 新增画像 |
| 并行配置报错 | `world_size` 与 `tp_size * dp_size` 或 `num_devices_per_instance` 不一致 | 三者对齐 |
| `moe_tp_size * moe_dp_size * ep_size` 校验失败 | 修改了 `tp_size` 但没有同步 MoE 切分 | 保持三者乘积等于 `world_size` |
| KV Cache 可用 block 过少 / 显存不足 | 单卡显存装不下权重与 KV | 增大 `tp_size`、降低 `num_input_tokens`，或改用显存更大的 `device_type` |
| 运行时间过长 | 请求数或输出长度过大 | 降低 `num_requests`、`num_output_tokens` 或 `predict_steps` |
| PD 分离与 PD 融合结果几乎一致 | `enable_kv_transfer_modeling` 为 `False` | 置为 `True` |

## 8. 相关材料

- ServingCast 仿真用户指南：`docs/zh/user_guide/msmodeling_serving_cast_simulation_user_guide.md`
- PD 融合对照示例：`serving_cast/example/instances.yaml`、`serving_cast/example/common.yaml`
- PD 分离并行策略自动寻优：`python -m cli.inference.throughput_optimizer --disagg ...`
