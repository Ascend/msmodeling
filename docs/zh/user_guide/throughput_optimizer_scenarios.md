# Throughput Optimizer 渐进式场景案例

本文使用同一个 DeepSeek-V3.1 模型，依次运行 PD 混部、量化、并行策略搜索、MTP、PD 分离和 PD 配比案例。每节给出业务用途、参数含义、完整命令和简要结果解读。

## 1 准备与场景索引

先完成[安装指南](../install_guide/msmodeling_install_guide.md)中的环境配置。所有案例使用以下模型、设备画像和请求长度：

| 参数 | 本文取值 | 含义 |
| --- | --- | --- |
| 模型位置参数 | `deepseek-ai/DeepSeek-V3.1` | 模型 ID，也支持本地模型配置目录 |
| `--device` | `ATLAS_800_A3_752T_128G_DIE` | 参与仿真的设备画像 |
| `--num-devices` | 常规案例为 `64` | 常规混部和 PD 分离的实例设备数；PD 配比自动搜索实例规模时作为搜索上界，固定 P/D 单实例设备数时由两者之和决定总预算 |
| `--input-length` | `512` | 每个请求的输入 token 数 |
| `--output-length` | `128` | 每个请求的输出 token 数 |

模型配置已通过 ModelScope 下载到本地时，将各命令中的模型 ID 替换为该目录即可。案例使用默认的 `analytic` 性能模型，由模型结构、设备画像和调度仿真计算服务指标，仿真设备数与运行命令的机器所装 NPU 数量无关。

| 步骤 | 业务目标 | 在前面案例上增加的配置 | 主要输出 |
| --- | --- | --- | --- |
| [基础 PD 混部](#2-基础-pd-混部) | 评估同实例执行 P/D 的吞吐 | 模型、设备数、请求长度和 TPOT 约束 | Throughput、TTFT、TPOT |
| [量化](#3-w8a8c8-量化) | 评估量化后的吞吐和显存 | W8A8 动态线性层、INT8 Attention | 吞吐、权重/KV Cache 显存 |
| [并行策略搜索](#4-tpepmoe-dp-并行策略搜索) | 选择适合 MoE 模型的并行策略 | TP、EP、MoE-DP 候选列表 | 最优并行配置、候选排名 |
| [MTP](#5-mtp-投机解码) | 评估一次前向预测多个 token 的效果 | 投机方法、token 数、接受长度 | Throughput、TPOT、TTFT |
| [PD 分离](#6-pd-分离) | 分别评估 P/D 阶段能力 | `--disagg` 和两个阶段的 SLO | Prefill TTFT、Decode TPOT、阶段吞吐 |
| [PD 配比](#7-pd-配比) | 规划 P/D 实例规模与数量 | 配比开关、每实例设备数、目标 QPS | P/D QPS、PD Ratio、实例数 |

TTFT 是首 token 延迟，TPOT 是首 token 之后的平均输出 token 间隔。混部和 Decode 的 Throughput 以输出 token/s 表示；Prefill 的 Throughput 以处理的输入 token/s 表示。以下数值为仿真结果。

## 2 基础 PD 混部

适用于 Prefill 和 Decode 共用一个服务实例，先估计指定请求长度和 TPOT 下的吞吐。实例可以使用多个设备。第一步只搜索 TP，保留默认的 EP 和 MoE-DP 配置。

- `--tpot-limit 50`：按 `50 ms` 的逐 token 延迟上限筛选候选。
- `--tp-sizes 8 16 32 64`：搜索四种张量并行度；DP 随实例设备数和 TP 推导。
- `--quantize-linear-action DISABLED`、`--quantize-attention-action DISABLED`：明确本节未开启这两类量化。

```bash
python -m cli.inference.throughput_optimizer deepseek-ai/DeepSeek-V3.1 \
  --device ATLAS_800_A3_752T_128G_DIE --num-devices 64 \
  --input-length 512 --output-length 128 \
  --tp-sizes 8 16 32 64 \
  --quantize-linear-action DISABLED \
  --quantize-attention-action DISABLED \
  --tpot-limit 50 --log-level info
```

**运行摘要**：未量化基线输出 `Throughput=460.80 token/s`、`TTFT=300.36 ms`、`TPOT=49.87 ms`，最优 `TP=16、DP=4、EP=64、MoE-DP=1`，`batch_size=6`、并发为 `24`。该候选的 `weight_GB=28.94`、`kv_cache_GB=0.30`。

**输出与解读**：`Overall Best Configuration` 给出符合 `TPOT <= 50 ms` 的最优吞吐，`Top ... PD Aggregated Configurations` 列出对应 TP/DP、batch 和 concurrency。上述结果作为下一节量化比较的基线。此处仅约束 TPOT，TTFT 作为观察指标。

**参数关系**：请求长度和设备数取正整数；TP 候选应满足模型及设备划分约束。`concurrency` 是实例总并发，`batch_size` 是局部批大小；在本文 PP=1 的案例中，总并发按 `batch_size × DP` 计算。

## 3 W8A8C8 量化

适用于在相同设备数和请求长度下，评估量化对吞吐与显存的影响。本节保留第 2 节的 TP 搜索范围，只改变两项量化设置。

- `--quantize-linear-action W8A8_DYNAMIC`：线性层使用 W8A8 动态量化。
- `--quantize-attention-action INT8`：Attention/KV Cache 使用 INT8 量化。
- `W8A8C8` 是上述组合的简称；CLI 使用两个独立参数及其枚举值。

```bash
python -m cli.inference.throughput_optimizer deepseek-ai/DeepSeek-V3.1 \
  --device ATLAS_800_A3_752T_128G_DIE --num-devices 64 \
  --input-length 512 --output-length 128 \
  --tp-sizes 8 16 32 64 \
  --quantize-linear-action W8A8_DYNAMIC \
  --quantize-attention-action INT8 \
  --tpot-limit 50 --log-level info
```

**运行摘要**：已运行的量化案例输出 `Throughput=2195.50 token/s`、`TTFT=662.63 ms`、`TPOT=49.34 ms`，最优 `TP=8、DP=8、EP=64、MoE-DP=1`，并发为 `120`。

**结果解读**：该配置满足 `50 ms` 的 TPOT 上限。与第 2 节比较时，同时观察 `Throughput`、`weight_GB`、`kv_cache_GB` 和可行并发，判断量化在当前 workload 下的影响。

## 4 TP/EP/MoE-DP 并行策略搜索

适用于进一步选择 DeepSeek-V3.1 的 MoE 并行策略。在量化案例上展开三个搜索维度：

| 参数 | 业务含义 | 本节候选 |
| --- | --- | --- |
| `--tp-sizes` | Attention/非专家部分的张量并行度 | `8 16 32 64` |
| `--ep-sizes` | 专家分布到多少设备 | `8 16 32 64` |
| `--moe-dp-sizes` | MoE 数据并行副本数 | `1 2 4 8` |

```bash
python -m cli.inference.throughput_optimizer deepseek-ai/DeepSeek-V3.1 \
  --device ATLAS_800_A3_752T_128G_DIE --num-devices 64 \
  --input-length 512 --output-length 128 \
  --tp-sizes 8 16 32 64 --ep-sizes 8 16 32 64 \
  --moe-dp-sizes 1 2 4 8 \
  --quantize-linear-action W8A8_DYNAMIC \
  --quantize-attention-action INT8 \
  --tpot-limit 50 --log-level info
```

**参数关系**：三个候选列表构成 `4 × 4 × 4 = 64` 个待校验组合，合法的设备划分和模型配置进入仿真。本文 PP=1 时 `DP = 64 / TP`；MoE 在同一设备集合内划分 EP、MoE-TP 和 MoE-DP。增加候选数会增加搜索量。

**运行摘要与解读**：最优结果仍为 `TP=8、DP=8、EP=64、MoE-DP=1`，吞吐为 `2195.50 token/s`，TPOT 为 `49.34 ms`。在本节覆盖的候选中，扩大 EP/MoE-DP 搜索范围没有改变第 3 节的推荐配置；其他候选可从排名表中比较。

## 5 MTP 投机解码

适用于模型具备 MTP 能力、希望评估多 token 预测对服务吞吐的影响。当前主线通过统一投机接口配置 MTP：

| 参数 | 类型与含义 | 本例取值 |
| --- | --- | --- |
| `--speculative-method` | 投机方法；`mtp` 表示启用 MTP 建模 | `mtp` |
| `--num-speculative-tokens` | 正整数候选列表，不含 anchor/bonus token；使用 MTP 时显式提供 | `3` |
| `--acceptance-length` | 浮点数，表示平均接受的投机 token 数，参与 Decode 延迟折算 | `2` |

接受长度默认值为 `5.0`，计算时限制到 `[0, n]`，其中 `n` 为投机 token 数；本例显式设置为 `2`。

```bash
python -m cli.inference.throughput_optimizer deepseek-ai/DeepSeek-V3.1 \
  --device ATLAS_800_A3_752T_128G_DIE --num-devices 64 \
  --input-length 512 --output-length 128 \
  --quantize-linear-action W8A8_DYNAMIC \
  --quantize-attention-action INT8 \
  --speculative-method mtp --num-speculative-tokens 3 \
  --acceptance-length 2 --tpot-limit 50 --log-level info
```

**建模方式**：先计算启用 MTP 后本轮前向的耗时，再按平均输出 token 数折算：

```text
A = min(max(acceptance_length, 0), n)
每输出 token 的 Decode 折算耗时 = 本轮 MTP 前向耗时 / (1 + A)
```

本例 `n=3、A=2`，平均每轮输出 `1 + 2 = 3` 个 token。`--num-speculative-tokens` 支持多个候选，与 TP/EP/MoE-DP 一起扩大搜索空间；本节固定为 `3`，TP 使用默认搜索范围。

**运行摘要**：`Throughput=17837.34 token/s`、`TTFT=3138.08 ms`、`TPOT=49.55 ms`。

**结果解读**：该配置满足 `50 ms` 的 TPOT 上限。MTP 增加每轮前向的预测工作量，同时增加平均输出 token 数，优化器据此重新搜索并发并计算吞吐；TTFT 仍作为混部服务的首 token 指标输出。

## 6 PD 分离

适用于 Prefill 和 Decode 独立部署，分别寻找两个阶段的并行度和并发。保留 W8A8C8，关闭 MTP，增加 `--disagg`，并分别设置 TTFT 和 TPOT 上限。

```bash
python -m cli.inference.throughput_optimizer deepseek-ai/DeepSeek-V3.1 \
  --device ATLAS_800_A3_752T_128G_DIE --num-devices 64 \
  --input-length 512 --output-length 128 \
  --quantize-linear-action W8A8_DYNAMIC \
  --quantize-attention-action INT8 \
  --disagg --ttft-limit 2000 --tpot-limit 50 --log-level info
```

**参数与输出**：本命令分别输出 Prefill 和 Decode 两张候选表。`--ttft-limit 2000` 筛选 Prefill，`--tpot-limit 50` 筛选 Decode；仅给其中一个 SLO 时，只运行对应阶段。

| 阶段 | 最优配置 | 吞吐 | 阶段时延 |
| --- | --- | ---: | ---: |
| Prefill | TP=1、DP=64、EP=64，并发=512 | 143366.21 输入 token/s | TTFT=1828.49 ms |
| Decode | TP=4、DP=16、EP=64，并发=320 | 6425.59 输出 token/s | TPOT=49.80 ms |

**结果解读**：两个阶段分别满足 `TTFT <= 2000 ms` 和 `TPOT <= 50 ms`。Prefill 与 Decode 推荐的 TP 和并发不同，适合使用不同的实例配置。Prefill 吞吐统计输入 token，Decode 吞吐统计输出 token；两者的阶段 QPS 用于下一步规划 P/D 能力配比。

## 7 PD 配比

适用于在 P/D 单实例规模确定后，规划满足目标请求 QPS 的实例数量。本节将 P/D 单实例规模各设为 `32` 个设备，继续使用同一模型、量化配置和请求长度。

- `--enable-optimize-prefill-decode-ratio`：分别优化 P/D 并汇总配比。
- `--prefill-devices-per-instance 32`、`--decode-devices-per-instance 32`：指定各阶段单实例设备数，两个参数成对配置。
- `--target-qps 1`：目标为 `1 req/s`，启用容量规划。
- 固定 P/D 单实例设备数时，总设备预算由 P/D 两项之和决定；本例为 `32 + 32 = 64`，最终实例数读取容量规划候选表。

```bash
python -m cli.inference.throughput_optimizer deepseek-ai/DeepSeek-V3.1 \
  --device ATLAS_800_A3_752T_128G_DIE \
  --input-length 512 --output-length 128 \
  --quantize-linear-action W8A8_DYNAMIC \
  --quantize-attention-action INT8 \
  --target-qps 1 --enable-optimize-prefill-decode-ratio \
  --prefill-devices-per-instance 32 \
  --decode-devices-per-instance 32 --log-level info
```

**运行摘要**：`P QPS=144.21 req/s`、`D QPS=456.82 req/s`、`PD Ratio=3.17`。

**结果解读**：理论 P:D 实例比例为 `D QPS / P QPS = 456.82 / 144.21 ≈ 3.17`，即约 `3.17` 个 Prefill 实例对应一个 Decode 实例。实际整数实例数和总设备数读取容量规划候选表，并以两侧处理能力较小者作为均衡 QPS。本节命令未设置 TTFT/TPOT 上限，结果体现该命令下的阶段处理能力。

**模式关系**：PD 配比模式内部已分别执行 P/D 优化，因此与 `--disagg` 是两种独立入口。只计算固定设备预算下的配比时，省略 `--target-qps`；若固定 P/D 单实例设备数，总预算仍由两者之和决定。只有在省略两个单实例设备数、使用自动搜索实例规模时，`--num-devices` 才用于限制搜索范围。
