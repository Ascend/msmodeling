# 服务化性能仿真 使用指南

## 1 简介

吞吐优化器（throughput optimizer）是一款在 SLO（Service Level Objective，服务级别目标）约束下优化吞吐量的工具。它会在指定的 SLO 约束（例如对 TTFT、TPOT 的上限）下自动搜索最优模型配置（并行策略、批大小），以最大化 token 吞吐量。
本文适用于需要评估大模型服务部署方案、模型服务吞吐、并行策略和 SLO 约束配置的开发者、性能工程师和容量规划人员。开始前请先完成《[msModeling 安装指南](../install_guide/msmodeling_install_guide.md)》中的环境搭建，并确认目标模型配置可被加载。

## 2 主要场景

吞吐优化器可用于硬件规划、SLO 约束下的吞吐寻优，以及 PD 分离部署设计。按部署形态划分，主要支持以下三类场景：

| 模式 | 适用场景 | 关键参数 |
| --- | --- | --- |
| PD 混部 | Prefill 与 Decode 运行在同一实例中，适合快速评估整体吞吐 | `--tpot-limits`、`--ttft-limits`|
| PD 分离 | Prefill 与 Decode 分开部署，需要分别评估阶段能力 | `--disagg`、`--ttft-limits` 或 `--tpot-limits` |
| PD 配比 | 需要规划 Prefill 与 Decode 实例数量比例 | `--enable-optimize-prefill-decode-ratio`、`--prefill-devices-per-instance`、`--decode-devices-per-instance` |

### 2.1 PD 混部场景

PD 混部针对 Prefill-Decode 合一的服务架构优化吞吐量，两个阶段运行在同一实例上。优化器会在所有可行的 TP（Tensor Parallelism，张量并行）与 DP（Data Parallelism，数据并行）配置中搜索，在 SLO 约束下找到最佳吞吐量。

#### 示例

```bash
python -m cli.inference.throughput_optimizer Qwen/Qwen3-32B \
    --device TEST_DEVICE \
    --num-devices 8 \
    --input-length 3500 \
    --output-length 1500 \
    --compile \
    --quantize-linear-action W8A8_DYNAMIC \
    --quantize-attention-action DISABLED \
    --tpot-limits 50
```

#### 使用 Prefix Cache

若要在启用 prefix cache 的情况下估算 PD 混部吞吐量，请添加 `--prefix-cache-hit-rate`：

```bash
python -m cli.inference.throughput_optimizer Qwen/Qwen3-32B \
    --device TEST_DEVICE \
    --num-devices 8 \
    --input-length 3500 \
    --output-length 1500 \
    --compile \
    --quantize-linear-action W8A8_DYNAMIC \
    --quantize-attention-action DISABLED \
    --tpot-limits 50 \
    --prefix-cache-hit-rate 0.5
```

#### 约束

- `--max-batched-tokens` 设置单个 Prefill 或混合 Prefill/Decode 步骤的 token 预算。如果 `effective_input_length` 大于 `max_batched_tokens`，优化器会自动将 Prefill 拆分为多个分块（chunk）。请将 `--max-batched-tokens` 设置为与服务引擎调度预算一致。

### 2.2 PD 分离场景

PD 分离将 Prefill 与 Decode 阶段拆分为独立的优化运行，适用于需要分别刻画各阶段性能，或规划 PD 分离服务部署的场景。

#### 前置条件

启用 PD 分离需提供：

- `--disagg`：启用 PD 分离

#### Prefill 模式

在 TTFT（Time-to-First-Token，首 token 时间）约束下优化 Prefill 阶段吞吐量。此模式下应设置 `--disagg` 与 `--ttft-limits`。

```bash
python -m cli.inference.throughput_optimizer Qwen/Qwen3-32B \
    --device TEST_DEVICE \
    --num-devices 8 \
    --input-length 3500 \
    --output-length 1500 \
    --compile \
    --quantize-linear-action W8A8_DYNAMIC \
    --quantize-attention-action DISABLED \
    --disagg \
    --ttft-limits 2000
```

#### Decode 模式

在 TPOT（Time-per-Output-Token，每输出 token 时间）约束下优化 Decode 阶段吞吐量。此模式下应设置 `--disagg` 与 `--tpot-limits`。

```bash
python -m cli.inference.throughput_optimizer Qwen/Qwen3-32B \
    --device TEST_DEVICE \
    --num-devices 8 \
    --input-length 3500 \
    --output-length 1500 \
    --compile \
    --quantize-linear-action W8A8_DYNAMIC \
    --quantize-attention-action DISABLED \
    --disagg \
    --tpot-limits 50
```

### 2.3 PD 配比场景

PD 配比模式可独立优化 Prefill 与 Decode 阶段，再合并结果以找到使系统吞吐量最大的最优 Prefill / Decode 实例配比。该模式尤其适用于 Prefill 与 Decode 实例可独立扩缩的 PD 分离服务架构。

#### 前置条件

启用 PD 配比需提供：

- `--enable-optimize-prefill-decode-ratio`：启用 PD 配比模式
- `--prefill-devices-per-instance`：每个 Prefill 实例的设备数
- `--decode-devices-per-instance`：每个 Decode 实例的设备数

#### 示例

```bash
python -m cli.inference.throughput_optimizer deepseek-ai/DeepSeek-V3.1 \
    --device TEST_DEVICE \
    --input-length 3500 \
    --output-length 1500 \
    --compile \
    --quantize-linear-action W8A8_DYNAMIC \
    --quantize-attention-action DISABLED \
    --enable-optimize-prefill-decode-ratio \
    --prefill-devices-per-instance 16 \
    --decode-devices-per-instance 16 \
    --log-level info
```

#### 约束

- `--enable-optimize-prefill-decode-ratio` 不能与 `--disagg` 同时使用
- 启用 PD 配比时，必须同时指定 `--prefill-devices-per-instance` 与 `--decode-devices-per-instance`

## 3 结果说明

脚本会输出性能指标（吞吐量、TTFT、TPOT、并发度，以及模式相关字段如 QPS 或 PD 配比）。示例：

```bash
********************************************************************************
  ----------------------------------------------------------------------------
  Input Configuration:
    Model: Qwen/Qwen3-32B
    Quantize Linear action: W8A8_DYNAMIC
    Quantize Attention action: DISABLED
    Devices: 8 TEST_DEVICE
    TTFT Limits: None ms
    TPOT Limits: 50.0 ms
  ----------------------------------------------------------------------------
  Overall Best Configuration:
    Best Throughput: 2888.45 tokens/s
    TTFT: 16032.05 ms
    TPOT: 49.90 ms
  ----------------------------------------------------------------------------
Top 4 Aggregation Configurations:
+-----+----------------------+-----------+-----------+-------------+-------------+--------------------+------------+
| Top | Throughput (token/s) | TTFT (ms) | TPOT (ms) | concurrency | num_devices |      parallel      | batch_size |
+-----+----------------------+-----------+-----------+-------------+-------------+--------------------+------------+
|  1  |       2888.45        |  16032.05 |   49.90   |     175     |       8     | TP=8 | PP=1 | DP=1 |    175     |
|  2  |       2013.49        |  22512.86 |   49.56   |     130     |       8     | TP=4 | PP=1 | DP=2 |     65     |
|  3  |       1140.23        |  25817.73 |   49.44   |      76     |       8     | TP=2 | PP=1 | DP=4 |     19     |
|  4  |        549.89        |  14214.54 |   48.72   |      32     |       8     | TP=1 | PP=1 | DP=8 |     4      |
+-----+----------------------+-----------+-----------+-------------+-------------+--------------------+------------+
********************************************************************************
```

### 3.1 多硬件配置对比

单次运行可传入一个或多个 `--device` 值，在相同模型、工作负载与 SLO 设置下对多个 `DeviceProfile` 目标进行基准测试并对比其最优配置。

```bash
python -m cli.inference.throughput_optimizer Qwen/Qwen3-32B \
    --device ATLAS_800_A2_280T_64G ATLAS_800_A3_560T_128G_DIE \
    --num-devices 8 \
    --input-length 3500 \
    --output-length 1500 \
    --compile \
    --quantize-linear-action W8A8_DYNAMIC \
    --quantize-attention-action DISABLED \
    --tpot-limits 50
```

行为说明：

- 各设备配置依次优化。每个设备完成后会打印该设备的表格（格式与单设备运行相同）。
- 指定两个及以上设备时，工具还会额外打印：
  1. **硬件配置对比**表，包含核心建模参数（算力、内存、通信带宽及相关字段）。
  2. **跨硬件汇总**表，列出各设备的最优配置并排序，便于对比。
- 跨硬件汇总因模式而异：
  - PD 混部：在 TTFT/TPOT 限制下各设备的最佳吞吐量。
  - PD 分离：在设置相应限制时，分别输出 Prefill 与 Decode 的跨硬件表。
  - PD 配比模式：在 TTFT/TPOT 限制下各设备的最佳均衡 QPS，包含 PD 配比；若设置 `--num-devices`，还可包含 P/D 实例数量。

#### 使用多个 `--device` 时的示例输出

指定两个及以上设备配置时，优化器会先为每个配置打印单设备结果，再输出两张跨硬件表：

**硬件配置对比表**

该表展示所有请求设备的核心建模参数（算力、内存带宽、通信带宽等）：

```text
************************************************************************************************************
  Cross-hardware - device profile summary (modeling abstraction vs performance merge tables)
  Device profile parameter comparison (effective compute / memory BW / comm BW)
  --------------------------------------------------------------------------------------------------------
+-----------------------+-----------------------+-------------------------+---------------+-------------+-----------+----------------+
|         Device        | Cube Compute (TFLOPS) | Vector Compute (TFLOPS) | HBM BW (TB/s) | Memory (GB) | Comm Grid | Comm BW (GB/s) |
+-----------------------+-----------------------+-------------------------+---------------+-------------+-----------+----------------+
|      TEST_DEVICE      |         247.73        |           7.70          |     0.960     |     64.0    |  256 x 8  |   35 | 137.2   |
| ATLAS_800_A2_376T_64G |         247.73        |          15.40          |     0.960     |     64.0    |  128 x 8  |  17.5 | 137.2  |
+-----------------------+-----------------------+-------------------------+---------------+-------------+-----------+----------------+
```

**跨硬件汇总表（因模式而异）**

在当前 SLO 约束下，按设备列出最优配置的排序表。PD 混部示例如下：

```text
****************************************************************************************************
  Cross-hardware - PD Aggregated (best throughput config per device under TTFT/TPOT limits)
  ------------------------------------------------------------------------------------------------
+-----+-----------------------+----------------------+-----------+-----------+-------------+--------------------+-------+-------------+
| Top |         Device        | Throughput (token/s) | TTFT (ms) | TPOT (ms) | Concurrency |      Parallel      | Batch | num_devices |
+-----+-----------------------+----------------------+-----------+-----------+-------------+--------------------+-------+-------------+
|  1  | ATLAS_800_A2_376T_64G |       18435.99       |  4986.05  |   54.48   |     1184    | TP=1 | PP=1 | DP=8 |  148  |      8      |
|  2  |      TEST_DEVICE      |       18128.74       |  4973.39  |   53.39   |     1144    | TP=1 | PP=1 | DP=8 |  143  |      8      |
+-----+-----------------------+----------------------+-----------+-----------+-------------+--------------------+-------+-------------+
```

对于 PD 分离与 PD 配比模式，跨硬件汇总表包含对应阶段或 QPS 相关列。

#### 终端扫描曲线（单设备）

仅评估一个设备配置时，扫描结束后优化器可渲染**终端 ASCII 散点图**，用于查看吞吐量与并发度、延迟在各并行配置下的关系。

三种优化模式均会生成图表：

| 模式 | 图表 |
|------|-------|
| PD 混部 | 吞吐量 vs 并发度；吞吐量 vs TPOT |
| PD 分离（Prefill） | 吞吐量 vs 并发度；吞吐量 vs TTFT |
| PD 分离（Decode） | 吞吐量 vs 并发度；吞吐量 vs TPOT |
| PD 配比 | 吞吐量 vs 并发度；吞吐量 vs TPOT（Decode 侧 TPS） |

说明：

- 使用多个 `--device` 时**不会**打印终端曲线；请改用跨硬件汇总表。
- 曲线数据点会排除 OOM / 内存不足的配置。曲线**不受** TTFT/TPOT SLO 限制过滤，因此展示完整有效扫描结果，而表格仍报告满足 SLO 约束的最优结果。
- 渲染依赖可选的 `plotext`。若未安装 `plotext` 或绘图失败，仍会打印优化结果并记录警告。

示例（单设备，PD 混部）：

```bash
python -m cli.inference.throughput_optimizer Qwen/Qwen3-32B \
    --device ATLAS_800_A2_280T_64G \
    --num-devices 8 \
    --input-length 3500 \
    --output-length 1500 \
    --tpot-limits 50 \
    --batch-range 1 256
```

## 4 参数

```bash
Options:
  --input-length INPUT_LENGTH
                        The input length of the prompt. (default: None)
  --output-length OUTPUT_LENGTH
                        The expected output length. (default: None)
  --mtp-acceptance-rate MTP_ACCEPTANCE_RATE [MTP_ACCEPTANCE_RATE ...]
                        Acceptance rate list for MTP (default: [0.9, 0.6, 0.4, 0.2])
  --dump-original-results
                        If set, dump the original results for analysis. (default: False)

General Options:
  model_id              Model source. Recommended safe mode: a reviewed absolute local model path. Model id mode also accepts Hugging Face or
                        ModelScope ids, but may execute remote Python code through trust_remote_code=True and is not security-guaranteed.
  --device DEVICE [DEVICE ...]
                        Device profile(s) to evaluate. One or more registered DeviceProfile names.
                        Supported values: TEST_DEVICE, ATLAS_800_A2_376T_64G, ATLAS_800_A2_313T_64G,
                        ATLAS_800_A2_280T_64G, ATLAS_800_A2_280T_64G_PCIE, ATLAS_800_A2_280T_32G_PCIE,
                        ATLAS_800_A3_752T_128G_DIE, ATLAS_800_A3_560T_128G_DIE,
                        ATLAS_800_A3_560T_128G_DIE_ROCE, ATLAS_350_425T_112G, ATLAS_350_425T_84G.
                        Multiple values enable cross-hardware comparison tables.
                        Duplicate names are removed; input order is preserved.
                        If omitted, defaults to TEST_DEVICE. (default: TEST_DEVICE)
  --num-devices NUM_DEVICES
                        Specifies the total number of devices/processes to use. Must be a positive integer. A value of 1 indicates single-device
                        execution. (default: 1)
  --reserved-memory-gb RESERVED_MEMORY_GB
                        Amount of device memory (in gigabytes) reserved for system usage and unavailable for application. Set to 0 to disable
                        memory reservation. (default: 10.0)
  --log-level {debug,info,warning,error,critical}
                        Specifies the verbosity level for log output. Available levels: 'debug' (most verbose), 'info', 'warning', 'error',
                        'critical' (least verbose). (default: error)

Model & Quantization Options:
  --compile             If set, invoke torch.compile() on the model before inference. (default: False)
  --compile-allow-graph-break
                        If set, allows graph breaks during torch.compile() to improve compilation speed or handle unsupported ops. (default: False)
  --num-mtp-tokens {0,1,2,3,4,5,6,7,8,9}
                        Number of MTP tokens, 0 means disabled - only support models having MTP like DeepSeek (default: 0)
  --quantize-linear-action {DISABLED,W8A16_STATIC,W8A8_STATIC,W4A8_STATIC,W8A16_DYNAMIC,W8A8_DYNAMIC,W4A8_DYNAMIC,FP8,MXFP4}
                        Quantize all linear layers in the model from choices (currently only support symmetric quant) (default: W8A8_DYNAMIC)
  --quantize-non-expert-linear-action {DISABLED,W8A16_STATIC,W8A8_STATIC,W4A8_STATIC,W8A16_DYNAMIC,W8A8_DYNAMIC,W4A8_DYNAMIC,FP8,MXFP4}
                        Set a separate quantization type for non-expert linear layers, such as attention projections, dense MLP layers, and shared experts, while routed MoE experts keep --quantize-linear-action. This option is mainly intended for DeepSeek V4-style MoE models. (default: DISABLED)
  --mxfp4-group-size MXFP4_GROUP_SIZE
                        Group size for MXFP4 quantization (default: 32)
  --quantize-attention-action {DISABLED,INT8,FP8}
                        Quantize the KV cache with the given action (default: DISABLED)
  --tp-sizes [TP_SIZES ...]
                        Enable TP search. Optional explicit TP sizes. If no value is provided, defaults to powers of 2 up to world_size. (default: None)
  --ep-sizes [EP_SIZES ...]
                        Enable EP search. Optional explicit EP sizes. If no value is provided, defaults to powers of 2 up to world_size. (default: None)
  --moe-dp-sizes [MOE_DP_SIZES ...]
                        Enable MOE-DP search. Optional explicit MOE-DP sizes. If no value is provided, defaults to powers of 2 up to world_size. (default: None)
  --enable-shared-expert-tp
                        Enable vLLM-style tensor parallel for shared experts. (default: False)
  --enable-sequence-parallel
                        Enable the sequence parallel graph rewrite pass during compilation. (default: False)
  --enable-dispatch-ffn-combine
                        Enable dispatch_ffn_combine fusion pattern during compilation. (default: False)
  --word-embedding-tp {col,row}
                        Enable word embedding tensor parallel with mode {'col','row'}. If omitted, embedding TP is disabled. (default: None)

Debug Options:
  --chrome-trace CHROME_TRACE
                        Generate chrome trace file for visualization, for example trace.json. (default: None)

Performance Model Options:
  --performance-model {analytic,profiling}
                        Performance model type(s). 'analytic': Roofline model (default). 'profiling': empirical model backed by measured CSV data (requires --profiling-database). May be specified multiple times. (default: None)
  --profiling-database PROFILING_DATABASE
                        Path to the profiling CSV database directory for 'profiling' mode. e.g. tensor_cast/performance_model/profiling_database/data/ATLAS_800_A3_752T_128G_DIE/vllm_ascend/vllm0.18.0_torch2.9.0_cann8.5/ (default: None)

Service Options:
  --ttft-limits TTFT_LIMITS
                        TTFT constraints under which to search for the best throughput. None means no constraint. (default: None)
  --tpot-limits TPOT_LIMITS
                        TPOT constraints under which to search for the best throughput. None means no constraint. (default: None)
  --max-batched-tokens MAX_BATCHED_TOKENS
                        Max batched tokens for one prefill or mixed prefill/decode step. (default: 8192)
  --prefix-cache-hit-rate PREFIX_CACHE_HIT_RATE
                        Prefix cache hit rate for token-level prefill reuse approximation. Valid range: [0, 1). (default: 0.0)
  --batch-range BATCH_RANGE [BATCH_RANGE ...]
                        Batch size range: [min max] or [max] (default: 1 for min, no limit for max) (default: None)
  --serving-cost SERVING_COST
                        Serving cost represents the cost of service delivery (default: 0)
  --disagg              If set, run disaggregation mode. disagg means disaggregation mode. (default: False)
  --jobs JOBS           Number of parallel jobs. (default: 8)
  --max-search-combinations MAX_SEARCH_COMBINATIONS
                        Warn when TP/EP/MOE-DP/MTP search combinations exceed this value. Set 0 to disable the warning. (default: 100)
  --concurrency-search-strategy {exponential,linear_exponential}
                        Concurrency search strategy. The default is exponential. (default: exponential)

MultiModal Options:
  --image-batch-size IMAGE_BATCH_SIZE
                        Number of images per request. If omitted, reuse batch_size for backward compatibility. (default: None)
  --image-height IMAGE_HEIGHT
                        Height of the input images (default: None)
  --image-width IMAGE_WIDTH
                        Width of the input images (default: None)

PD Ratio Optimization Options:
  --enable-optimize-prefill-decode-ratio
                        Enable PD (Prefill-Decode) ratio optimization mode. This mode independently
                        optimizes Prefill and Decode phases, then combines results to find the optimal
                        P/D instance ratio. Cannot be used together with --disagg. (default: False)
  --prefill-devices-per-instance PREFILL_DEVICES_PER_INSTANCE
                        Number of devices per Prefill instance. Required when --enable-optimize-prefill-decode-ratio
                        is set. Determines the parallelism configuration search space for Prefill phase.
  --decode-devices-per-instance DECODE_DEVICES_PER_INSTANCE
                        Number of devices per Decode instance. Required when --enable-optimize-prefill-decode-ratio
                        is set. Determines the parallelism configuration search space for Decode phase.
```

主要参数说明如下：

| 参数名称 | 分类 | 可选/必选 | 参数说明 |
| --- | --- | --- | --- |
| `--device` | Options | 可选 | 指定一个或多个设备画像名称；传入多个设备时会输出跨硬件对比结果。<br>1. 类型：Str 或 List[Str]。<br>2. 参考值：`TEST_DEVICE`、`ATLAS_800_A2_376T_64G`、`ATLAS_800_A2_313T_64G`、`ATLAS_800_A2_280T_64G`、`ATLAS_800_A2_280T_64G_PCIE`、`ATLAS_800_A2_280T_32G_PCIE`、`ATLAS_800_A3_752T_128G_DIE`、`ATLAS_800_A3_560T_128G_DIE`、`ATLAS_800_A3_560T_128G_DIE_ROCE`、`ATLAS_350_425T_112G`、`ATLAS_350_425T_84G`。<br>3. 默认值：未指定时使用 `TEST_DEVICE`。<br>4. 支持一次传入多个已注册 `DeviceProfile` 名称，重复名称会去重并保留输入顺序。 |
| `--input-length` | Options | 必选 | 输入 prompt 的 token 长度。<br>1. 类型：Int。<br>2. 取值范围：正整数。<br>3. 默认值：无。 |
| `--output-length` | Options | 必选 | 期望生成的输出 token 长度。<br>1. 类型：Int。<br>2. 取值范围：正整数。<br>3. 默认值：无。 |
| `--mtp-acceptance-rate` | Options | 可选 | 指定 MTP token 的接受率列表。<br>1. 类型：List[Float]。<br>2. 取值范围：浮点数列表。<br>3. 默认值：`[0.9, 0.6, 0.4, 0.2]`。 |
| `--prefix-cache-hit-rate` | Options | 可选 | 指定 prefix cache 命中率。<br>1. 类型：Float。<br>2. 取值范围：`[0, 1)`。<br>3. 默认值：`0.0`。 |
| `--dump-original-results` | Options | 可选 | 输出原始搜索结果，便于进一步分析。<br>1. 类型：Bool。<br>2. 取值范围：开关参数。<br>3. 默认值：`False`。 |
| `model_id` | General Options | 必选 | 模型 ID 或已审核的本地模型绝对路径。<br>1. 类型：Str。<br>2. 参考值：Hugging Face ID、ModelScope ID 或本地绝对路径。<br>3. 默认值：无。<br>4. 使用远端模型 ID 时，可能通过 `trust_remote_code=True` 执行远端代码。 |
| `--num-devices` | General Options | 可选 | 指定参与仿真的设备总数。<br>1. 类型：Int。<br>2. 取值范围：正整数。<br>3. 默认值：`1`。 |
| `--reserved-memory-gb` | General Options | 可选 | 指定每张设备预留给系统使用的显存，单位为 GB。<br>1. 类型：Float。<br>2. 取值范围：非负数；设置为 `0` 表示不预留系统显存。<br>3. 默认值：`10.0`。 |
| `--log-level` | General Options | 可选 | 指定日志级别。<br>1. 类型：Str。<br>2. 参考值：`debug`、`info`、`warning`、`error`、`critical`。<br>3. 默认值：`error`。 |
| `--compile` | Model & Quantization Options | 可选 | 在推理前对模型调用 `torch.compile()`。<br>1. 类型：Bool。<br>2. 取值范围：开关参数。<br>3. 默认值：`False`。 |
| `--compile-allow-graph-break` | Model & Quantization Options | 可选 | 允许 `torch.compile()` 过程中出现 graph break。<br>1. 类型：Bool。<br>2. 取值范围：开关参数。<br>3. 默认值：`False`。 |
| `--num-mtp-tokens` | Model & Quantization Options | 可选 | 指定 MTP token 数量，`0` 表示不启用。<br>1. 类型：Int。<br>2. 取值范围：`0` 到 `9`。<br>3. 默认值：`0`。 |
| `--quantize-linear-action` | Model & Quantization Options | 可选 | 指定线性层量化方式。<br>1. 类型：Str。<br>2. 参考值：`DISABLED`、`W8A16_STATIC`、`W8A8_STATIC`、`W4A8_STATIC`、`W8A16_DYNAMIC`、`W8A8_DYNAMIC`、`W4A8_DYNAMIC`、`FP8`、`MXFP4`。<br>3. 默认值：`W8A8_DYNAMIC`。 |
| `--quantize-non-expert-linear-action` | Model & Quantization Options | 可选 | 为 attention 投影、dense MLP、shared experts 等非 expert 线性层指定独立量化方式。<br>1. 类型：Str。<br>2. 参考值：`DISABLED`、`W8A16_STATIC`、`W8A8_STATIC`、`W4A8_STATIC`、`W8A16_DYNAMIC`、`W8A8_DYNAMIC`、`W4A8_DYNAMIC`、`FP8`、`MXFP4`。<br>3. 默认值：`DISABLED`。<br>4. 主要用于 DeepSeek V4 风格 MoE 模型；路由 MoE experts 仍使用 `--quantize-linear-action`。 |
| `--mxfp4-group-size` | Model & Quantization Options | 可选 | 指定 MXFP4 量化的 group size。<br>1. 类型：Int。<br>2. 取值范围：正整数。<br>3. 默认值：`32`。 |
| `--quantize-attention-action` | Model & Quantization Options | 可选 | 指定 KV cache 量化方式。<br>1. 类型：Str。<br>2. 参考值：`DISABLED`、`INT8`、`FP8`。<br>3. 默认值：`DISABLED`。 |
| `--tp-sizes` | Model & Quantization Options | 可选 | 启用 TP 搜索，并可显式指定 TP 取值范围。<br>1. 类型：List[Int]。<br>2. 取值范围：正整数列表。<br>3. 默认值：`None`；仅传入参数但不指定取值时，默认搜索不超过 `world_size` 的 2 的幂。 |
| `--ep-sizes` | Model & Quantization Options | 可选 | 启用 EP 搜索，并可显式指定 EP 取值范围。<br>1. 类型：List[Int]。<br>2. 取值范围：正整数列表。<br>3. 默认值：`None`；仅传入参数但不指定取值时，默认搜索不超过 `world_size` 的 2 的幂。 |
| `--moe-dp-sizes` | Model & Quantization Options | 可选 | 启用 MOE-DP 搜索，并可显式指定 MOE-DP 取值范围。<br>1. 类型：List[Int]。<br>2. 取值范围：正整数列表。<br>3. 默认值：`None`；仅传入参数但不指定取值时，默认搜索不超过 `world_size` 的 2 的幂。 |
| `--enable-shared-expert-tp` | Model & Quantization Options | 可选 | 启用 vLLM 风格的 shared experts 张量并行。<br>1. 类型：Bool。<br>2. 取值范围：开关参数。<br>3. 默认值：`False`。<br>4. shared experts 使用 dense MLP TP，并延迟执行 `down_proj` 规约。 |
| `--enable-sequence-parallel` | Model & Quantization Options | 可选 | 编译期间启用 sequence parallel 图改写 pass。<br>1. 类型：Bool。<br>2. 取值范围：开关参数。<br>3. 默认值：`False`。 |
| `--enable-dispatch-ffn-combine` | Model & Quantization Options | 可选 | 编译期间启用 dispatch_ffn_combine 融合模式。<br>1. 类型：Bool。<br>2. 取值范围：开关参数。<br>3. 默认值：`False`。 |
| `--word-embedding-tp` | Model & Quantization Options | 可选 | 启用 word embedding 张量并行并指定并行模式。<br>1. 类型：Str。<br>2. 参考值：`col`、`row`。<br>3. 默认值：`None`，表示不启用 embedding TP。 |
| `--performance-model` | Performance Model Options | 可选 | 指定性能模型类型，可重复指定以对比多个模型。<br>1. 类型：Str。<br>2. 参考值：`analytic`、`profiling`。<br>3. 默认值：`None`，未指定时取 `analytic`。<br>4. `profiling` 模式需配合 `--profiling-database` 使用。 |
| `--profiling-database` | Performance Model Options | 条件必选 | 指定 profiling 模式使用的实测算子 CSV 数据库目录。<br>1. 类型：Str。<br>2. 取值范围：数据库目录路径，例如 `tensor_cast/performance_model/profiling_database/data/ATLAS_800_A3_752T_128G_DIE/vllm_ascend/vllm0.18.0_torch2.9.0_cann8.5/`。<br>3. 默认值：`None`；使用 `--performance-model profiling` 时必填。 |
| `--chrome-trace` | Debug Options | 可选 | 生成 Chrome Trace 文件，用于可视化分析算子级性能。<br>1. 类型：Str。<br>2. 参考值：Trace 文件路径，例如 `trace.json`。<br>3. 默认值：`None`，表示不生成 Chrome Trace 文件。 |
| `--ttft-limits` | Service Options | 可选 | 指定 TTFT 约束，用于在约束内搜索最优吞吐。<br>1. 类型：Float。<br>2. 取值范围：正数，单位 ms。<br>3. 默认值：`None`，表示不限制 TTFT。 |
| `--tpot-limits` | Service Options | 可选 | 指定 TPOT 约束，用于在约束内搜索最优吞吐。<br>1. 类型：Float。<br>2. 取值范围：正数，单位 ms。<br>3. 默认值：`None`，表示不限制 TPOT。 |
| `--max-batched-tokens` | Service Options | 可选 | 指定单个 prefill 或混合 prefill/decode step 的最大 batched tokens。<br>1. 类型：Int。<br>2. 取值范围：正整数。<br>3. 默认值：`8192`。 |
| `--batch-range` | Service Options | 可选 | 指定 batch size 搜索范围。<br>1. 类型：List[Int]。<br>2. 格式：`[min max]` 或 `[max]`。<br>3. 默认值：`None`；未指定 `min` 时默认从 `1` 开始搜索，未指定 `max` 时不设置上限。 |
| `--serving-cost` | Service Options | 可选 | 指定服务成本，用于成本相关指标计算。<br>1. 类型：Float。<br>2. 取值范围：非负数。<br>3. 默认值：`0`。 |
| `--disagg` | Service Options | 可选 | 启用 PD 分离模式。<br>1. 类型：Bool。<br>2. 取值范围：开关参数。<br>3. 默认值：`False`。 |
| `--jobs` | Service Options | 可选 | 指定并行搜索任务数。<br>1. 类型：Int。<br>2. 取值范围：正整数。<br>3. 默认值：`8`。 |
| `--max-search-combinations` | Service Options | 可选 | 当 TP / EP / MOE-DP / MTP 搜索组合数超过该值时输出警告。<br>1. 类型：Int。<br>2. 取值范围：非负整数；设置为 `0` 表示关闭该警告。<br>3. 默认值：`100`。 |
| `--concurrency-search-strategy` | Service Options | 可选 | 指定并发度搜索策略。<br>1. 类型：Str。<br>2. 参考值：`exponential`、`linear_exponential`。<br>3. 默认值：`exponential`。 |
| `--image-batch-size` | MultiModal Options | 可选 | 指定每个请求的图像数量。<br>1. 类型：Int。<br>2. 取值范围：正整数。<br>3. 默认值：`None`；未设置时复用 batch size。 |
| `--image-height` | MultiModal Options | 可选 | 指定输入图像高度。<br>1. 类型：Int。<br>2. 取值范围：正整数。<br>3. 默认值：`None`。 |
| `--image-width` | MultiModal Options | 可选 | 指定输入图像宽度。<br>1. 类型：Int。<br>2. 取值范围：正整数。<br>3. 默认值：`None`。 |
| `--prefill-devices-per-instance` | PD Ratio Optimization Options | 条件必选 | 启用 PD 配比优化时必填，指定每个 Prefill 实例的设备数。<br>1. 类型：Int。<br>2. 取值范围：正整数。<br>3. 默认值：无。<br>4. 用于确定 Prefill 阶段的并行配置搜索空间。 |
| `--decode-devices-per-instance` | PD Ratio Optimization Options | 条件必选 | 启用 PD 配比优化时必填，指定每个 Decode 实例的设备数。<br>1. 类型：Int。<br>2. 取值范围：正整数。<br>3. 默认值：无。<br>4. 用于确定 Decode 阶段的并行配置搜索空间。 |
| `--enable-optimize-prefill-decode-ratio` | PD Ratio Optimization Options | 可选 | 启用 Prefill/Decode 实例配比优化模式。<br>1. 类型：Bool。<br>2. 取值范围：开关参数。<br>3. 默认值：`False`。<br>4. 不能与 `--disagg` 同时使用。 |

### 搜索维度与范围

`throughput_optimizer` 根据提供的搜索参数决定搜索维度：

- `--tp-sizes`：启用 TP 搜索
- `--ep-sizes`：启用 EP 搜索
- `--moe-dp-sizes`：启用 MOE-DP 搜索

规则：

- 若未提供任何搜索参数，默认仅进行 TP 搜索，使用默认范围。
- 对于未参与搜索的维度，使用固定默认值：
  - `tp = num_devices`
  - `ep = num_devices`
  - `moe-dp = 1`
- 若提供搜索参数，需要显式给出取值。常用范围为：
  `powers of 2 up to world_size`
  （例如当 `num_devices=8` 时，可设置为 `[1, 2, 4, 8]`）。

示例：

```bash
# 仅搜索 TP（显式范围）
python -m cli.inference.throughput_optimizer Qwen/Qwen3-30B-A3B --device TEST_DEVICE --num-devices 8 --input-length 3500 --output-length 1500 --tpot-limits 50 --tp-sizes 1 2 4 8

# 搜索 TP/EP（MOE-DP 固定为 1）
python -m cli.inference.throughput_optimizer Qwen/Qwen3-30B-A3B --device TEST_DEVICE --num-devices 8 --input-length 3500 --output-length 1500 --tpot-limits 50 --tp-sizes 1 2 4 8 --ep-sizes 1 2 4 8

# 搜索 TP/EP/MOE-DP
python -m cli.inference.throughput_optimizer Qwen/Qwen3-30B-A3B --device TEST_DEVICE --num-devices 8 --input-length 3500 --output-length 1500 --tpot-limits 50 --tp-sizes 1 2 4 8 --ep-sizes 1 2 4 8 --moe-dp-sizes 1 2 4 8

# 仅搜索 EP（显式范围）
python -m cli.inference.throughput_optimizer Qwen/Qwen3-30B-A3B --device TEST_DEVICE --num-devices 8 --input-length 3500 --output-length 1500 --tpot-limits 50 --ep-sizes 1 2 4 8
```

### 性能模型选择

`throughput_optimizer` 默认使用解析（Roofline）模型估算算子延迟。通过 `--performance-model` 可切换为基于实测算子 CSV 数据的 profiling 模型：

- `--performance-model analytic`（默认）：纯解析 Roofline 模型，无需额外数据。
- `--performance-model profiling`：使用实测算子数据建模，**必须同时指定 `--profiling-database <目录>`**，否则启动时报错。当某算子 shape 在 CSV 中缺失时，先尝试插值，仍无法命中时回退到解析模型。

`--performance-model` 可重复指定（如同时传 `analytic` 与 `profiling`）以便对比不同模型的结果。

示例：

```bash
python -m cli.inference.throughput_optimizer Qwen/Qwen3-30B-A3B --device ATLAS_800_A3_752T_128G_DIE --num-devices 8 --input-length 3500 --output-length 1500 --tpot-limits 50 \
    --performance-model profiling \
    --profiling-database tensor_cast/performance_model/profiling_database/data/ATLAS_800_A3_752T_128G_DIE/vllm_ascend/vllm0.18.0_torch2.9.0_cann8.5/
```

## 5 补充说明

### 5.1 PD 混部下性能指标的计算方法

- TTFT：

  当 `effective_input_length <= max_batched_tokens` 时，保留原有完整 Prefill 公式。
  平均值为 `ttft = sum_for_ttft / concurrency`。对于 sum_for_ttft，假设 Prefill
  批大小为最大批处理 token 数除以有效输入长度。
  因此 `prefill_batch_size = max_batched_tokens // effective_input_length`。请求按
  prefill_batch_size 分步逐批处理。总 ttft 时间计算如下：

  `sum_for_ttft = (prefill_latency * prefill_batch_size) * (1 + calc_nums_for_ttft) * calc_nums_for_ttft / 2`

  例如，若有 12 个请求，max_batched_tokens 为 8192，input_length 为 2048，
  则 prefill_batch_size 为 4，12 个请求分 3 步处理。
  因此

  `sum_for_ttft = (prefill_latency * 4 ) * (1 + 3) * 3 / 2`

  `ttft = sum_for_ttft / 12`

  当 `effective_input_length > max_batched_tokens` 时，优化器会自动将 Prefill
  拆分为多个分块（chunk）。当前版本使用固定的 decode-first 混合调度器，并保留 15% 的 token
  预算余量；暂不暴露调度器选择的 CLI 参数。

- TPOT：

  TPOT 计算不考虑 bubble 时间。

  `tpot = (ttft + decode_latency * output_length) / output_length`

- 输出吞吐量
  `output_throughput = 1000 * (output_length * concurrency) / (ttft + tpot * output_length)`

### 5.2 PD 配比模式下性能指标的计算方法

PD 配比模式使用 QPS（Queries Per Second，每秒查询数）作为匹配 Prefill 与 Decode 能力的主要指标：

- **Prefill QPS（P QPS）**：

  P QPS 表示单个 Prefill 实例的请求处理能力。

  `P QPS = p_concurrency / ttft * 1000` (req/s)

  其中：
  - `p_concurrency`：Prefill 阶段的批大小（并发请求数）
  - `ttft`：首 token 时间，单位为毫秒

- **Decode QPS（D QPS）**：

  D QPS 表示单个 Decode 实例的请求处理能力。

  `D QPS = d_concurrency / (tpot * max(output_length - 1, 1)) * 1000` (req/s)

  其中：
  - `d_concurrency`：Decode 阶段的批大小（并发请求数）
  - `tpot`：每输出 token 时间，单位为毫秒
  - `max(output_length - 1, 1)`：生成首 token 之后的 Decode token 数

- **PD 配比**：

  PD 配比表示为实现均衡吞吐量，Prefill 与 Decode 实例之间的最优比例。

  `PD 配比 = D QPS / P QPS`

  含义：
  - PD 配比 = 1.0：一个 Prefill 实例可支撑一个 Decode 实例
  - PD 配比 = 2.0：一个 Prefill 实例可支撑两个 Decode 实例
  - PD 配比 = 0.5：需要两个 Prefill 实例才能支撑一个 Decode 实例

- **实例分布**：

  指定 `--num-devices` 时，会计算最优的 Prefill 与 Decode 实例数量：

  1. 计算在设备预算内可容纳的总实例数：
     `max_p_inst = total_devices / p_devices_per_instance`
     `max_d_inst = total_devices / d_devices_per_instance`

  2. 寻找 P:D 实例组合，使其：
     - 尽可能接近 PD 配比
     - 落在总设备预算内
     - 使系统整体吞吐量最大
