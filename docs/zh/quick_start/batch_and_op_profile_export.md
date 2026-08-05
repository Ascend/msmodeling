# Throughput Optimizer 批处理使用说明

## 1. 简介（Introduction）

本代码在 Throughput Optimizer CLI（`cli/inference/throughput_optimizer.py`）中集成了两大功能：

- **功能一：CSV 批处理功能集成** —— 通过 `--input-csv`/`--output-csv` 参数，从 CSV 文件加载多个 case 一次性执行，结果聚合到单个输出 CSV。
- **功能二：算子级 CSV 导出功能** —— 通过 `--export-op-profile` 参数，在批处理模式下额外导出每个 case 最优配置对应的算子级性能明细 CSV，便于算子级瓶颈定位。

**CLI 入口**：`python -m cli.inference.throughput_optimizer`

**适用场景**：

- **多配置批量评估**：在一次运行中评估多个 设备 × 卡数 × 模型 × 输入/输出长度 组合
- **算子级瓶颈定位**：在获得最优配置后，直接查看该配置下每个算子的性能明细，无需重新运行仿真

**关键特性**：

- CSV 驱动的 case 定义：每行定义一个 benchmark case
- 顺序执行 + 单 case 错误隔离（单个失败不会中断整批）
- 增量结果写入 + 批量刷新（每 10 个 case 刷新一次磁盘）
- 统一的结果 CSV，Decode/Prefill 指标列与 Throughput Optimizer 输出对齐
- 算子级 CSV 导出（可选）：不带该 flag 时行为与原功能完全一致

> **模块位置**：批处理辅助模块位于 `cli/inference/_batch_cases.py`，通过 `cli/inference/throughput_optimizer.py` CLI 间接调用。

## 2. 快速开始（Quick Start）

### 2.1 准备输入 CSV

输入 CSV 必须包含表头行，列名与模板一致。每行定义一个 benchmark case。最简单的方式是参考仓库自带的示例 CSV `cases.csv`（含 3 个 case）进行修改。

### 2.2 运行批处理

```bash
python -m cli.inference.throughput_optimizer --input-csv cases.csv --output-csv results.csv
```

`cases.csv` 中的每一行将顺序执行。结果每处理 10 个 case 刷新一次磁盘写入 `results.csv`，即使运行被中断，已完成批次的结果也会保留。如果单个 case 失败（包括解析阶段错误，如非法布尔值、非法模式、非法数字、非法枚举值；以及运行时错误，如无效模型、设备未找到），错误信息会打印到 stderr，执行继续到下一个 case；失败的 case 在 CSV 中会有一行记录，但指标列为空。

### 2.3 运行批处理（主结果 + 算子级导出）

```bash
python -m cli.inference.throughput_optimizer --input-csv cases.csv --output-csv results.csv --export-op-profile
```

在主结果 CSV（`results.csv`）之外，会在 `--output-csv` 同级目录下生成 `op_profiles/` 子目录，包含每个 case 最优配置对应的算子级性能明细 CSV。

### 2.4 最小示例

以仓库自带的 `cases.csv` 为例（3 个 case），在项目根目录执行：

```bash
# 仅主结果
python -m cli.inference.throughput_optimizer --input-csv cases.csv --output-csv results.csv

# 主结果 + 算子级导出
python -m cli.inference.throughput_optimizer --input-csv cases.csv --output-csv results.csv --export-op-profile
```

> **工作目录**：在项目根目录运行即可，无需切换其他子目录。

## 3. CSV 输入格式（CSV Input Format）

输入 CSV 必须包含一个表头行，列名与下文模板一致。每个后续行定义一个 benchmark case。列表类型字段（如 `mtp_acceptance_rate`、`ep_sizes`）在单元格内使用分号（`;`）作为分隔符。

输入 CSV 共 24 列（`CSV_CONFIG_HEADER`），分为 5 列必需列与 19 列可选列。

### 3.1 必需列（5 列）

> **重要**：如果表头缺失以下任一必需列（`device`、`num_devices`、`model_id`、`input_length`、`output_length`），工具会**立即报错**，而不是静默使用错误的默认值。

| 列名 | 含义 | 示例值 |
|------|------|--------|
| `device` | 设备 profile 名称（取值与 Throughput Optimizer 的 `--device` 一致） | `ATLAS_800_A3_752T_128G_DIE` |
| `num_devices` | 设备数量 | `8` |
| `model_id` | HuggingFace 模型 ID 或本地路径 | `Qwen/Qwen3-32B` |
| `input_length` | 输入 token 长度 | `3500` |
| `output_length` | 输出 token 长度 | `1500` |

### 3.2 可选列（19 列）

| 列名 | 含义 | 默认值 |
|------|------|--------|
| `case_name` | case 标识符（为空时自动生成为 `case_N`） | `case_N`（N 为 CSV 行号，从 2 开始） |
| `ttft_limits` | TTFT SLO 限制，单位 ms（为空表示无约束） | 空（无约束） |
| `tpot_limits` | TPOT SLO 限制，单位 ms（为空时默认 50.0 ms） | `50.0` |
| `tp_sizes` | 要搜索的 TP 尺寸（分号分隔） | `None`（默认范围，2 的幂直至 num_devices） |
| `quantize_linear_action` | 线性层量化动作 | `None`（回退到 `W8A8_DYNAMIC`） |
| `quantize_attention_action` | 注意力（KV cache）量化动作 | `None`（回退到 `DISABLED`） |
| `ep_sizes` | 要搜索的 EP 尺寸（分号分隔），与 Throughput Optimizer 的 `--ep-sizes` 对齐 | `None` |
| `num_mtp_tokens` | MTP token 数量（0 表示禁用） | `0` |
| `mtp_acceptance_rate` | MTP 接受率列表（分号分隔） | `0.9;0.6;0.4;0.2` |
| `compile` | 是否启用 torch.compile（合法值：`true`/`1`/`yes`/`false`/`0`/`no`；空值为 `false`；无效值抛出错误） | `false` |
| `mode` | 运行模式：`agg` 或 `disagg`（空值默认 `agg`；无效值抛出错误，该 case 被跳过） | `agg` |
| `max_batched_tokens` | 单次 prefill 或混合 prefill/decode 步的最大 batched tokens | `8192` |
| `batch_range` | batch size 范围（分号分隔，`min;max` 或 `max`） | `None` |
| `serving_cost` | serving_cost | `0` |
| `jobs` | 并行作业数 | `8` |
| `log_level` | 日志级别 | `info` |
| `mxfp4_group_size` | MXFP4 量化的 group size | `32` |
| `reserved_memory_gb` | 预留设备显存（GB） | 空（回退到 CLI `--reserved-memory-gb` 默认值 10.0） |
| `compile_allow_graph_break` | 是否允许 compile 图中断（合法值：`true`/`1`/`yes`/`false`/`0`/`no`；空值为 `false`；无效值抛出错误） | `false` |

> [!Warning]
> `ttft_limits` 与 `tpot_limits` 每个 case **至多接受一个值**。如果提供多个值（如 `50;100`），工具会报错。请将其拆分为不同的 CSV 行。
>
> 所有 limit 值的单位均为**毫秒（ms）**，与 Throughput Optimizer 保持一致。
>
> `tpot_limits` 为空时默认使用 `50.0` ms。

<!-- -->

> [!Note]
> 列表字段（如 `mtp_acceptance_rate`、`ep_sizes`、`tp_sizes`、`batch_range`）使用分号 `;` 作为单元格内分隔符，例如 `mtp_acceptance_rate=0.9;0.6;0.4`。

## 4. 示例 CSV（Example CSV）

以下是仓库自带的 `cases.csv` 内容（3 个 case）：

```csv
case_name,device,num_devices,model_id,input_length,output_length,ttft_limits,tpot_limits,tp_sizes,quantize_linear_action,quantize_attention_action,ep_sizes,num_mtp_tokens,mtp_acceptance_rate,compile,mode,max_batched_tokens,batch_range,serving_cost,jobs,log_level,mxfp4_group_size,reserved_memory_gb,compile_allow_graph_break
8card_agg_w8a8,ATLAS_800_A3_752T_128G_DIE,8,Qwen/Qwen3-32B,3500,1500,,50,,W8A8_DYNAMIC,DISABLED,,0,,true,agg,8192,,0,8,info,32,,false
4card_disagg_ep,ATLAS_800_A3_752T_128G_DIE,4,Qwen/Qwen3-32B,2000,500,,50,,W8A8_DYNAMIC,INT8,2,0,,true,disagg,8192,,0,8,info,32,,false
1card_disagg_mtp,ATLAS_800_A3_752T_128G_DIE,1,Qwen/Qwen3-32B,16000,1000,,50,,W8A8_DYNAMIC,DISABLED,,3,0.9;0.6;0.4,true,disagg,16000,,0,8,info,32,,false
```

该 CSV 定义了 3 个 case：

1. **`8card_agg_w8a8`**：8 卡聚合（agg）模式，使用 `W8A8_DYNAMIC` 线性量化，TPOT SLO ≤ 50 ms，输入 3500 / 输出 1500 tokens，开启 torch.compile。
2. **`4card_disagg_ep`**：4 卡分离（disagg）模式，使用 `W8A8_DYNAMIC` 线性量化 + `INT8` 注意力量化，EP size 为 2，输入 2000 / 输出 500 tokens。
3. **`1card_disagg_mtp`**：1 卡分离（disagg）模式，启用 MTP（3 tokens），MTP 接受率列表 `0.9;0.6;0.4`，输入 16000 / 输出 1000 tokens，`max_batched_tokens` 设为 16000。

## 5. 输出 CSV（Output CSV）

### 5.1 主结果 CSV（results.csv）

主结果 CSV 包含一个表头行、一个列出有效量化选项的参考行，以及每个 benchmark case 一行的结果。共 40 列，关键列含义如下：

| 列名 | 含义 |
|------|------|
| `Case_Name` | case 标识符 |
| `Device Type` | 设备 profile 名称 |
| `Number of Devices` | 设备数量 |
| `Input Length` / `Output Length` | token 长度 |
| `Model` | 模型标识符 |
| `Decode_Linear Quant Type` | 最优 decode 配置的线性量化类型 |
| `Decode_Attn Quant Type` | 最优 decode 配置的注意力量化类型 |
| `Decode_EP Size` | 使用的 EP size（仅当 ep_size > 1 时显示） |
| `Decode_MTP Tokens` | MTP token 数量 |
| `Decode_TPOT Target(ms)` | decode 的 TPOT SLO 目标（ms） |
| `Decode_Concurrency` | 最优 decode 并发数 |
| `Decode_TPOT(ms)` | SLO 约束下的最优 decode TPOT |
| `Decode_Total TPS` | 最优 decode 总吞吐 |
| `Decode_TPS/Device` | 最优 decode 单卡吞吐 |
| `Decode_Mem` / `Decode_Comm` / `Decode_Cube` / `Decode_Vec` | 性能拆解百分比 |
| `Decode_TP Size` / `Decode_PP Size` / `Decode_DP Size` | 最优 decode 并行配置 |
| `Prefill_Linear Quant Type` | 最优 prefill 配置的线性量化类型 |
| `Prefill_Attn Quant Type` | 最优 prefill 配置的注意力量化类型 |
| `Prefill_EP Size` | 使用的 EP size（仅当 ep_size > 1 时显示） |
| `Prefill_MTP Tokens` | MTP token 数量 |
| `Prefill_TTFT Target(ms)` | prefill 的 TTFT SLO 目标（ms） |
| `Prefill_Concurrency` | 最优 prefill 并发数 |
| `Prefill_TTFT(ms)` | SLO 约束下的最优 prefill TTFT |
| `Prefill_Total TPS` | 最优 prefill 总吞吐 |
| `Prefill_TPS/Device` | 最优 prefill 单卡吞吐 |
| `Prefill_Mem` / `Prefill_Comm` / `Prefill_Cube` / `Prefill_Vec` | 性能拆解百分比 |
| `Prefill_TP Size` / `Prefill_PP Size` / `Prefill_DP Size` | 最优 prefill 并行配置 |
| `QuantizeLinearAction_options` | 所有有效的线性量化动作枚举 |
| `QuantizeAttentionAction_options` | 所有有效的注意力量化动作枚举 |

> [!Note]
>
> - 如果某个 case 没有 prefill 结果（如 disagg decode-only 模式），prefill 指标列留空；如果没有 decode 结果，decode 指标列留空。
> - 如果没有配置满足 SLO，对应指标列仍存在，但最优值反映最接近的尝试。
> - 如果整个 case 失败（如无效模型、设备未找到，或解析阶段的非法字段值），指标列留空，错误打印到 stderr，但该 case 仍在 CSV 中保留一行。

### 5.2 算子级 CSV（op_profiles/）

当使用 `--export-op-profile` 时，会在 `--output-csv` 同级目录下生成 `op_profiles/` 子目录，其中包含每个 case 最优配置对应的算子级性能明细 CSV。

**算子级 CSV 表头（6 列）**：

| 列名 | 含义 |
|------|------|
| `Phase` | 阶段名称：`prefill` 或 `decode` |
| `Op_Name` | 算子名称（如 `tensor_cast.static_quant_linear.default`） |
| `Perf_Model` | 性能模型类型（如 `analytic`） |
| `Perf_Total_s` | 该算子总耗时（秒） |
| `Perf_Avg_s` | 该算子单次平均耗时（秒） |
| `Call_Times` | 该算子被调用次数 |

**文件命名规则**（根据 case 的模式与阶段生成不同文件）：

| 模式 | 生成文件 | 说明 |
|------|----------|------|
| `agg` | `<case_name>.csv` | 包含 prefill + decode 两阶段，通过 `Phase` 列区分 |
| `disagg` prefill 阶段 | `<case_name>_prefill.csv` | 仅含 prefill 阶段算子 |
| `disagg` decode 阶段 | `<case_name>_decode.csv` | 仅含 decode 阶段算子 |

> [!Note]
> 当最优配置行无对应 op_profile（如 early_stop 场景）时，会跳过该 summary 不写文件。

**算子级 CSV 前 5 行示例**（来自 `op_profiles/8card_agg_w8a8.csv`，agg 模式）：

```csv
Phase,Op_Name,Perf_Model,Perf_Total_s,Perf_Avg_s,Call_Times
prefill,tensor_cast.static_quant_linear_all_reduce.default,analytic,0.11920457142857176,0.0009312857142857168,128
prefill,tensor_cast.static_quant_linear.default,analytic,0.06757038297872339,0.0005278936170212765,128
prefill,tensor_cast.attention.default,analytic,0.027979883945841392,0.00043718568665377175,64
prefill,aten.add.Tensor,analytic,0.02415994608179724,0.00018874957876404094,128
```

当一个 case 的 `effective_input_length > max_batched_tokens` 时，prefill 阶段会被切分为多个 chunk 执行。每个 chunk 会独立调用 `_get_or_compute_latency` 产生一份算子级性能数据。为保证导出的算子级 CSV 反映**完整仿真结果**，所有 chunk 的算子数据会通过 `OpProfileSummary.merge()` 方法按以下规则聚合后导出。

**聚合规则（`OpProfileSummary.merge()`）**：

1. **分组键**：按 `(Op_Name, Perf_Model)` 分组。相同算子名与相同性能模型类型的行归为一组。
2. **`Perf_Total_s`**：所有 chunk 的 `Perf_Total_s` 之和（总耗时累加）。例如 chunk 1 的 `Perf_Total_s=0.0676`，chunk 2 的 `Perf_Total_s=0.0507`，则聚合后 `Perf_Total_s=0.1183`。
3. **`Call_Times`**：所有 chunk 的 `Call_Times` 之和（调用次数累加）。例如 chunk 1 的 `Call_Times=128`，chunk 2 的 `Call_Times=96`，则聚合后 `Call_Times=224`。
4. **`Perf_Avg_s`**：由聚合后的 `Perf_Total_s / Call_Times` 重新计算。
5. **排序**：聚合结果按 `Perf_Total_s` 降序排列，便于快速定位耗时最高的算子。

**聚合示例**：假设某 case 的 prefill 阶段被切分为 2 个 chunk，算子 `tensor_cast.static_quant_linear.default` 在两个 chunk 中的数据如下：

| 来源 | Perf_Total_s | Call_Times | Perf_Avg_s |
|------|--------------|------------|------------|
| chunk 1 | 0.0676 | 128 | 0.000528 |
| chunk 2 | 0.0507 | 96 | 0.000528 |
| **聚合后（导出）** | **0.1183** | **224** | **0.000528** |

## 6. CLI 参数（CLI Parameters）

批处理相关参数位于 "Batch Cases Options" 参数组：

```bash
Batch Cases Options:
  --input-csv INPUT_CSV
                        Path to input CSV with benchmark cases. When set, runs batch mode
                        (one case per row, aggregated output CSV).
  --output-csv OUTPUT_CSV
                        Path to output CSV for batch results.
                        Defaults to 'benchmark_cases_results.csv' when --input-csv is used.
  --export-op-profile   Export per-case operator-level CSV files under <output-dir>/op_profiles/.
                        Only effective with --input-csv.
```

| 参数 | 说明 |
|------|------|
| `--input-csv <路径>` | 输入 CSV 文件路径，每行一个 case。设置后进入批处理模式，表头必须包含模板列。 |
| `--output-csv <路径>` | 输出 CSV 文件路径。当使用 `--input-csv` 时，默认为 `benchmark_cases_results.csv`。 |
| `--export-op-profile` | 导出每个 case 的算子级 CSV 文件到 `<output-dir>/op_profiles/` 子目录。**仅在批处理模式（`--input-csv`）下生效**；单 case 模式下会打印 warning 并忽略。 |

> [!Note]
> `--export-op-profile` 是一个 `store_true` 开关（默认 `False`）。不带该 flag 时，批处理行为与原功能完全一致。

其余参数（`--device`、`--input-length`、`--output-length`、`--tp-sizes`、`--ep-sizes`、`--ttft-limits` 等）属于单 case 模式参数。在批处理模式下，这些参数由 CSV 各行覆盖，CLI 上的同名参数不会逐 case 生效。

## 7. 执行行为（Execution Behavior）

- **顺序执行**：case 按 CSV 行顺序逐个执行。
- **单 case 错误隔离**：如果单个 case 失败（包括解析阶段错误，如非法布尔值、非法模式、非法数字、非法枚举值；以及运行时错误，如无效模型、设备未找到、运行时异常），错误打印到 stderr，该 case 在输出 CSV 中得到一行记录（指标列为空），执行继续到下一个 case，不中断整批。
- **批量刷新**：结果每处理 10 个 case 刷新一次磁盘。如果进程被中断，已完成批次的结果会被保留。
- **进程内执行**：每个 case 在进程内调用 `serving_cast.parallel_runner.ParallelRunner`。`ParallelRunner` 返回的结果使用 `OptimizerSummary` 的公共 API 进行 SLO 过滤，正确处理 disagg 模式（prefill summary 仅应用 TTFT 限制，decode summary 仅应用 TPOT 限制）。
- **算子级导出零影响**：不带 `--export-op-profile` 时，与原批处理功能完全一致，不生成 `op_profiles/` 目录，`results.csv` 内容与带 flag 时逐字节一致。

## 8. 与 Throughput Optimizer 的关系（Relationship to Throughput Optimizer）

批处理模式内部为每个 case 调用 [Throughput Optimizer]的 `ParallelRunner`。CSV 列到 Throughput Optimizer 参数的映射如下：

| CSV 列 | Throughput Optimizer 参数 |
|--------|----------------------------|
| `device` | `--device` |
| `num_devices` | `--num-devices` |
| `model_id` | 位置参数 `model_id` |
| `input_length` | `--input-length` |
| `output_length` | `--output-length` |
| `ttft_limits` | `--ttft-limits` |
| `tpot_limits` | `--tpot-limits` |
| `compile` | `--compile` |
| `mode` | `--disagg`（当值为 `disagg` 时设置） |
| `quantize_linear_action` | `--quantize-linear-action` |
| `quantize_attention_action` | `--quantize-attention-action` |
| `tp_sizes` | `--tp-sizes` |
| `ep_sizes` | `--ep-sizes` |
| `num_mtp_tokens` | `--num-mtp-tokens` |
| `mtp_acceptance_rate` | `--mtp-acceptance-rate` |
| `max_batched_tokens` | `--max-batched-tokens` |
| `batch_range` | `--batch-range` |
| `serving_cost` | `--serving-cost` |
| `jobs` | `--jobs` |
| `log_level` | `--log-level` |
| `reserved_memory_gb` | `--reserved-memory-gb` |
| `compile_allow_graph_break` | `--compile-allow-graph-break` |
| `mxfp4_group_size` | `--mxfp4-group-size` |

以下 Throughput Optimizer 参数未作为 CSV 列暴露，使用默认值（从主 CLI 参数继承或使用与 Throughput Optimizer 一致的默认值）：

| 参数 | 默认值 |
|------|--------|
| `image_batch_size` | `None` |
| `image_height` | `None` | 
| `image_width` | `None` | 
| `prefix_cache_hit_rate` | `0.0` | 
| `enable_multistream` | `True` | 
| `enable_optimize_prefill_decode_ratio` | `False` |
| `prefill_devices_per_instance` | `None` | 
| `decode_devices_per_instance` | `None` | 
| `moe_dp_sizes` | `None` |
| `dump_original_results` | `False` |
| `concurrency_search_strategy` | `exponential` |

## 9. 注意事项

1. **必需列缺失会立即报错**：如果表头缺失 `device`、`num_devices`、`model_id`、`input_length`、`output_length` 任一列，工具会立即抛出错误，**不会**静默使用错误的默认值。

2. **`ttft_limits` 与 `tpot_limits` 每个 case 至多一个值**：如果在一个单元格内提供多个值（如 `50;100`），工具会报错。请将其拆分为不同的 CSV 行。

3. **所有 limit 值单位为毫秒（ms）**：`ttft_limits` 和 `tpot_limits` 的单位均为 ms，与 Throughput Optimizer 保持一致。

4. **`tpot_limits` 为空时默认 50.0 ms**：当 `tpot_limits` 单元格为空时，工具会自动使用 `50.0` ms 作为默认 TPOT SLO 限制。

5. **列表字段使用分号 `;` 作为单元格内分隔符**：如 `mtp_acceptance_rate=0.9;0.6;0.4`、`ep_sizes=2;4`、`tp_sizes=1;2;4`、`batch_range=1;256`。

6. **`mode` 字段取值 `agg` 或 `disagg`**：空值默认按 `agg` 处理；无效值（如 `disag`、`aggr`）会抛出错误，该 case 被跳过并写空指标行。

7. **布尔字段（`compile`、`compile_allow_graph_break`）严格校验**：合法值为 `true`/`1`/`yes`（→ True）、`false`/`0`/`no`（→ False）；空值为 `False`。无效值（如 `ture`、`flase`）会抛出错误，该 case 被跳过并写空指标行。

8. **`--export-op-profile` 仅在批处理模式（`--input-csv`）下生效**：在单 case 模式下使用该 flag 会打印 warning（`Warning: --export-op-profile is only effective with --input-csv; ignored.`）到 stderr 并忽略。

9. **`op_profiles/` 目录生成位置**：生成在 `--output-csv` 同级目录下。例如 `--output-csv results.csv` 会生成 `op_profiles/`。

10. **算子级 CSV 命名规则**：
    - agg 模式生成 `<case_name>.csv`（含 prefill + decode 两阶段，通过 `Phase` 列区分）；
    - disagg prefill 阶段生成 `<case_name>_prefill.csv`；
    - disagg decode 阶段生成 `<case_name>_decode.csv`。

11. **单 case 失败不中断批处理**：包括解析阶段错误（如非法布尔值、非法模式、非法数字、非法枚举值）和运行时错误（如无效模型、设备未找到）。错误打印到 stderr，该 case 在 CSV 中保留一行记录但指标列为空，执行继续到下一个 case。

12. **每 10 个 case 刷新一次磁盘**：中断后已完成批次的结果保留。

13. **工作目录为项目根目录**：直接在项目根目录运行 `python -m cli.inference.throughput_optimizer ...` 即可。

14. **不带 `--export-op-profile` 时与原功能完全一致**，`op_profiles/` 目录不会被创建或修改。

## 10. 附录

快速测试建议

1. **第一步：复制示例 CSV**。将 `cases.csv` 复制到自己的工作目录，按需修改设备、卡数、模型、输入/输出长度等字段。

2. **第二步：先跑仅主结果模式**。第一次使用建议不带 `--export-op-profile`，确认 CSV 格式正确、case 能正常执行：

   ```bash
   python -m cli.inference.throughput_optimizer --input-csv my_cases.csv --output-csv my_results.csv
   ```

3. **第三步：检查结果 CSV**。打开 `my_results.csv`，确认表头、参考行（量化选项枚举）与各 case 结果行。失败的 case 指标列会留空，错误信息见 stderr。

4. **第四步：按需启用算子级导出**。当需要定位算子级瓶颈时，加上 `--export-op-profile`：

   ```bash
   python -m cli.inference.throughput_optimizer --input-csv my_cases.csv --output-csv my_results.csv --export-op-profile
   ```

   然后查看 `op_profiles/` 目录下对应 case 的算子 CSV，按 `Perf_Total_s` 降序排序即可快速定位耗时最高的算子。

5. **第五步：注意 SLO 单位与默认值**。所有 limit 单位为 ms；`tpot_limits` 为空时默认 50.0 ms；`ttft_limits` 与 `tpot_limits` 每个 case 至多一个值。
