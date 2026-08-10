# RFC: Throughput Optimizer 当前 Mixed-Batch 变长 Token 建模实现

## 元数据

| 项目 | 内容 |
|:---|:---|
| **状态** | 已批准 |
| **作者** | stormchasingg |
| **更新日期** | 2026-06-26 |
| **相关链接** | |

---

## 1. 概述

本文档描述当前代码中 throughput optimizer 对变长 token 的 mixed-batch 建模实现。

与最初版本的 RFC 相比，当前实现仍然保留了 mixed-batch variable-token 的核心能力，但代码结构已经有较大重构：

- CLI 侧使用必填的 `--input-length` 同时承载 fixed-length 和 distribution 两种模式
- `--input-length` 支持正整数或已存在的 length-distribution YAML 文件路径
- 分布文件加载由 `load_length_distribution()` 完成，workload 构造主要集中在 `OptimizerData`
- mixed-batch 执行入口是 `_get_batched_forward_info()`
- 最终结果展示不再依赖单独的 summary 子类
- batched detail row 的展开逻辑直接收敛在 `OptimizerSummary` 中

当前实现仅覆盖：

- `cli.inference.throughput_optimizer`
- disaggregation 模式
- prefill-only
- `TTFT` 约束搜索

不覆盖：

- PD ratio 优化模式
- Monte Carlo 随机采样
- 请求到达分布建模

## 2. 当前适用范围与入口条件

### 2.1 CLI 行为

当前 CLI 暴露一个必填输入长度参数：

- `--input-length`

该参数支持：

- 正整数：进入 fixed-length 模式
- 已存在的 YAML 文件路径：进入 variable-length distribution 模式

该校验由 `serving_cast/service/utils.py` 中的 `check_positive_integer_and_string()` 完成，并由 `cli/inference/throughput_optimizer.py` 使用。

### 2.2 分布文件模式

当前不再有独立的 `--length-distribution` CLI 开关。

当 `--input-length` 是文件路径时，CLI 会将其视为 length-distribution YAML 文件。例如：

- `--input-length serving_cast/example/length_distribution.yaml`

当前该模式仅允许运行在：

- `--disagg`
- 指定 `--ttft-limits`
- 不指定 `--tpot-limits`
- 不启用 PD ratio 优化

若不满足这些条件，CLI 会直接拒绝执行。

运行时，`args.input_length` 会保持为：

- `int`：fixed-length 模式
- `str`：distribution 文件路径模式

`ParallelRunner` 会基于该文件路径加载 `LengthDistribution` 对象，并传入 `OptimizerData.length_distribution`。在该模式下，`OptimizerData.input_length` 会被设置为 `None`。

## 3. 数据模型

### 3.1 分布类型

`serving_cast/service/utils.py` 中定义了：

- `LengthBin`
- `LengthDistribution`

每个 bin 包含：

- `min_tokens`
- `max_tokens`
- `weight`

当前校验规则为：

- `min_tokens >= 0`
- `max_tokens > min_tokens`
- `weight > 0`
- 相邻 bin 不允许重叠

`weight` 不要求用户手工配置为和为 `1`，实现会在内部做归一化。

### 3.2 OptimizerData 字段

`OptimizerData` 当前同时容纳 fixed-length 和 distribution 两种模式所需字段：

- `input_length`
- `length_distribution`
- `output_length`
- `batch_size`
- `ttft_limits`
- `tpot_limits`
- `prefix_cache_hit_rate`
- 其他服务与搜索参数

distribution 模式的判断条件是：

- `optimizer_data.length_distribution is not None`

## 4. 变长 Token Workload 构造

### 4.1 representative rows

`OptimizerData.get_representative_rows()` 会把每个长度 bin 转成一个 representative row。

当前默认使用 bin 中点，并返回：

- `num_input_tokens`
- `query_len`
- `request_ratio`

字段语义为：

- `num_input_tokens`：该 bin 的代表原始输入 token 数
- `query_len`：考虑 prefix cache 后实际参与 prefill 计算的长度
- `request_ratio`：内部归一化后的 bin 权重

### 4.2 effective input length

`OptimizerData.get_effective_input_length()` 当前行为分两种：

- fixed-length 模式：
  - 返回考虑 prefix cache 后的标量 effective input length
- distribution 模式：
  - 返回 representative `query_len` 的加权平均值

chunk plan 由 `OptimizerData.get_prefill_chunk_plan()` 负责生成，依赖 effective input length 和 `max_batched_tokens`。

distribution 模式下，CLI 会先构造 `OptimizerData(length_distribution=...)` 并调用 `get_prefill_num_chunks()` 做 chunked prefill 预检查。当前 distribution 模式不支持 chunked prefill，因此当分布对应的 effective prefill length 需要拆成多个 chunk 时，用户需要增大 `--max-batched-tokens`。

### 4.3 整数 sample 分配

`OptimizerData.build_concurrency_samples(concurrency)` 负责把分布展开成一个具体 mixed batch。

当前算法：

1. 按 `concurrency * request_ratio` 计算理想 sample 数
2. 取 `floor(...)` 作为基础分配
3. 使用最大剩余法分配剩余请求

返回行包含：

- `num_input_tokens`
- `query_len`
- `request_ratio`
- `samples`

这保证了在给定 `concurrency` 下 mixed-batch 构成是确定且可复现的。

## 5. 执行路径

### 5.1 fixed-length 路径

`BaseThroughputOptimizer._get_forward_info()` 仍然是以下场景的标准路径：

- fixed-length prefill
- decode

该路径构造单个 `RequestInfo` 模板，并使用：

- `generate_inputs`

### 5.2 mixed-batch 路径

`BaseThroughputOptimizer._get_batched_forward_info()` 是当前 mixed-batch 的执行入口。

它会：

1. 调用 `optimizer_data.build_concurrency_samples(concurrency)`
2. 将这些分布行展开为真实 heterogeneous `List[RequestInfo]`
3. 按 `samples` 重复每一类请求
4. 使用：
   - `generate_inputs_varlen`
   进行推理

这里的字段命名已经对齐 `RequestInfo` 语义：

- `num_input_tokens`：原始输入 token 数
- `query_len`：本次实际参与 prefill 的长度

## 6. Disaggregation 接入

`DisaggThroughputOptimizer.get_inference_info()` 当前同时支持：

- fixed-length
- variable-token mixed-batch

分支条件是：

- `variable_input_mode = optimizer_data.length_distribution is not None`

### 6.1 mixed-batch prefill

在 variable-token prefill 场景下：

1. 调用 `_get_batched_forward_info()`
2. 用模型执行时间加 `serving_cost` 得到 `latency_ms`
3. 吞吐计算改为使用本次 batch 的真实 token 总量：

```text
total_input_tokens = Σ(num_input_tokens * samples)
token/s = total_input_tokens / ttft * 1000
```

这取代了旧的基于单个 `input_length` 的标量吞吐计算方式。

### 6.2 summary rows

当前输出的 DataFrame 包含：

- 1 条 aggregate row
- 多条 composition detail rows

aggregate row 约定为：

- `num_input_tokens = "all"`
- `request_ratio = 1.0`
- `samples = concurrency`

detail row 复用同一配置字段，但会清空性能列，例如：

- `ttft`
- `tpot`
- `token/s`
- `token/s/device`
- `percentage_breakdowns`

## 7. 最终报告与表格渲染

### 7.1 Summary 类结构

当前实现不再使用单独的 mixed-batch summary 子类。

现在由 `OptimizerSummary` 自身同时处理：

- regular fixed-length final output
- mixed-batch final output

### 7.2 最优行筛选

`OptimizerSummary._prepare_agg_disagg_results()` 仍然负责基础筛选与排序：

- 按 `ttft` / `tpot` 限制过滤
- 按 `token/s` 排序
- 每个 `parallel` 只保留最优 aggregate row

这个筛选仍然是在 aggregate row 上完成的。

### 7.3 composition row 展开

当 `args.input_length` 是字符串路径时，`OptimizerSummary._get_agg_disagg_final_out()` 会转到：

- `_get_agg_disagg_final_out_batched()`

该路径会：

1. 先选出最优 aggregate rows
2. 调用 `_expand_composition_rows()`
3. 从 `self._summary_df` 中把对应 detail rows 拼回去

当前用于匹配 detail rows 的键为：

- `parallel`
- `batch_size`
- `concurrency`
- `num_devices`

排序规则为：

- aggregate row 在前
- detail rows 在后
- detail rows 按 `num_input_tokens` 排序

### 7.4 batched 最终表

mixed-batch 最终表由：

- `_get_disagg_table_buf_batched()`

生成。

当前它是 prefill-only 表格，展示：

- `Top`
- `num_devices`
- `num_input_tokens`
- `request_ratio`
- `samples`
- `concurrency`
- `TTFT (ms)`
- `Throughput (token/s)`
- `parallel`
- `batch_size`

当前 batched 最终表里不再展示：

- `input_length`
- `output_length`

因为 composition row 更关心的是：

- 原始代表 token 数
- 分布占比
- 实际分配 sample 数

detail row 上的性能列统一显示为 `-`。

## 8. 模块交互关系

```bash
CLI 参数解析 (throughput_optimizer.py)
    │
    ├─ 必填 --input-length
    │   ├─ 正整数 → fixed-length 模式
    │   └─ 已存在 YAML 路径 → distribution 文件模式
    │
    ├─ input_length 是否为 YAML 路径？
    │   ├─ 否
    │   │   └─ 走标量 input_length 路径
    │   │
    │   └─ 是
    │       ├─ 校验：
    │       │   ├─ 仅支持 disagg
    │       │   ├─ 仅支持 prefill-only（设置 --ttft-limits）
    │       │   └─ 不允许 --tpot-limits / 不允许 PD ratio 优化
    │       ├─ load_length_distribution(input_length)
    │       ├─ 构造 OptimizerData(length_distribution=...)
    │       ├─ 检查当前分布是否需要 chunked prefill
    │       └─ 进入 distribution-aware prefill 路径
    │
    └─ ParallelRunner(args)
        │
        ├─ input_length 是否为 YAML 路径？
        │   ├─ 否
        │   │   └─ OptimizerData(input_length=<int>)
        │   │
        │   └─ 是
        │       └─ OptimizerData(input_length=None,
        │                         length_distribution=load_length_distribution(input_length))
        │
        └─ run_disagg()
            │
            ├─ 对每个 TP / parallel 候选配置
            │   └─ _get_df_list()
            │       └─ DisaggThroughputOptimizer.run()
            │           │
            │           ├─ 二分搜索 batch size
            │           └─ 对每个候选 batch
            │               └─ get_inference_info()
            │                   │
            │                   ├─ length_distribution is None?
            │                   │   ├─ 是 → _get_forward_info()
            │                   │   └─ 否 → _get_batched_forward_info()
            │                   │            │
            │                   │            ├─ build_concurrency_samples(concurrency)
            │                   │            ├─ 展开为 heterogeneous RequestInfo 列表
            │                   │            └─ run_inference(generate_inputs_varlen)
            │                   │
            │                   ├─ 计算 TTFT / throughput
            │                   └─ 构造：
            │                       ├─ 1 条 aggregate row
            │                       └─ 多条 composition detail rows
            │
            └─ OptimizerSummary.report_final_result(args)
                │
                ├─ args.input_length 是否为 YAML 路径？
                │   ├─ 否 → _get_agg_disagg_final_out()
                │   │         └─ _get_disagg_table_buf()
                │   │
                │   └─ 是 → _get_agg_disagg_final_out_batched()
                │             │
                │             ├─ _prepare_agg_disagg_results()
                │             ├─ _expand_composition_rows()
                │             └─ _get_disagg_table_buf_batched()
                │
                └─ 输出 overall best configuration + final table
```

## 9. 正在进行中与后续工作

当前已经明确但尚未完成的方向包括：

1. aggregation 模式下的 variable-token mixed-batch 建模
2. decode-only 场景下的 variable-token mixed-batch 建模

除此之外，当前实现的已知局限包括：

1. distribution 模式仅支持 disaggregation prefill with `TTFT`
2. distribution 模式当前不支持 chunked prefill
3. PD ratio 优化不支持 variable-token mixed-batch
4. 最优配置仍然先在 aggregate row 上筛选，再展开 detail rows

## 10. 后续变更提示

如果后续实现再次演化，最敏感、最需要同步更新文档的部分包括：

- `--input-length` 的整数/路径解析语义
- `OptimizerData` 的命名与 workload 构造辅助函数
- `BaseThroughputOptimizer` mixed-batch 执行入口
- `DisaggThroughputOptimizer` 的 summary row schema
- `OptimizerSummary` 的 batched final report 展示格式

尤其如果未来重新引入以下能力：

- 独立的 distribution CLI 参数
- summary 子类
- decode 模式 batched report
- aggregation 模式的 variable-token 支持

建议单独作为后续 RFC 变更记录。 

---

# 实现更新：Aggregation 与 Chunked Prefill

**更新日期：** 2026-08-04

本节记录原 RFC 完成后新增的行为。如果本节与前文的范围或限制描述冲突，以本节所述的当前实现为准。

## 1. 更新后的范围

长度分布输入当前支持：

- aggregation 模式，包括 TTFT 和 TPOT 约束场景
- disaggregation 的 prefill-only 模式
- aggregation 和 disaggregation 在 `max_batched_tokens` 限制下的 Prefill chunk 划分

当前仍不支持或尚未完整支持：

- disaggregation decode-only 变长输入建模
- 变长输入的 PD ratio 优化
- aggregation 变长 chunk 中类似 vLLM 的 Prefill/Decode 重叠
- 请求到达分布和 Monte Carlo 采样

## 2. 感知并发数的 Chunk Plan

`PrefillChunk` 使用 `is_last_chunk` 表示该 chunk 执行后对应请求是否完成：

```python
@dataclass
class PrefillChunk:
    index: int
    query_len: int
    seq_len: int
    is_last_chunk: bool = False
```

固定输入场景保留单请求拆分语义；执行阶段再按并发数扩展相同 shape。变长输入场景调用
`get_prefill_chunk_plan(concurrency)`，其中 `concurrency` 是单个 DP rank 的并发数。该方法通过
`build_concurrency_samples(concurrency)` 展开采样请求，并在 `max_batched_tokens` 预算内打包 chunk。同一个
`index` 可以包含多个不同 shape 的请求片段，它们会进入同一次推理。

`get_prefill_num_chunks(chunk_plan)` 直接从当前实际执行的 plan 计算：

```python
max(chunk.index for chunk in chunk_plan) + 1 if chunk_plan else 0
```

该接口不会按 concurrency 重新生成 plan，也不依赖列表最后一个元素的 index。

## 3. Chunked Mixed-Batch 执行

`BaseThroughputOptimizer._get_batched_forward_info()` 统一返回：

```python
tuple[list[tuple[ModelRunnerMetrics, int]], list[dict]]
```

每个结果包含一次模型执行指标和该次执行完成的请求数。非 chunk 路径保留原有异构请求构造方式，包括
`num_input_tokens`，执行一次并包装为单元素列表；其完成请求数等于当前 DP rank 的 `samples` 总和。

chunk 路径在执行前校验 index 必须从 0 开始、连续且非递减，再按 index 分组，为每组构造异构
`RequestInfo` 列表并调用一次 `run_inference(generate_inputs_varlen)`。完成请求数通过
`is_last_chunk=True` 统计。如果某次执行的可用显存为负数，后续 chunk 不再执行。

`completed_requests` 属于 ServingCast 的调度信息，不写入会被缓存复用的 `ModelRunnerMetrics`。

## 4. Aggregation 指标

Aggregation 变长输入链路为：

```text
get_inference_info
  -> get_prefill_chunk_plan(DP-rank concurrency)
  -> _get_batched_full_prefill_metrics
  -> _get_batched_forward_info(chunk_plan=...)
  -> 每个 chunk index 调用一次 run_inference(generate_inputs_varlen)
```

chunk 按 index 顺序执行。当前 chunk 完成的请求以累计模型时间作为 first-token time，平均 TTFT 为：

```text
TTFT = sum(当前 chunk 完成请求数 * 当前累计时间) / 总采样请求数
```

例如两个 chunk 均为 10 ms，并分别完成 4 个请求中的 2 个请求，则平均 TTFT 为 15 ms。Aggregation
没有 PD 传输，因此不计算 `serving_cost`。`prefill_latency` 记录首个 chunk 的模型延迟，
`prefill_last_latency` 记录最后一个 chunk 的模型延迟，Prefill 显存取所有 chunk 中的最小可用显存。

Decode 延迟仍按加权有效输入长度单独计算。变长输入路径不进入 `_simulate_chunked_prefill()` 或
`DecodeFirstWithSlack`，当前近似中 Prefill 与 Decode 不重叠；固定输入 aggregation 的 chunked-prefill
仍沿用原调度器。

## 5. Disaggregation 指标与 CLI

Disaggregation 变长 Prefill 将 concurrency-aware chunk plan 传给 `_get_batched_forward_info()`，并消费与
Aggregation 相同的 `list[(metrics, completed_requests)]`：

1. 累计 Prefill 延迟从 `serving_cost` 开始，与固定输入 Disaggregation chunk 路径一致；
2. 按执行顺序累加每个 chunk 的模型延迟；
3. 使用当前累计延迟乘以该 chunk 完成的请求数；
4. 除以当前 DP rank 的采样请求总数得到 TTFT；
5. 使用全局输入 token 数和 Prefill 阶段总延迟计算吞吐；
6. 跨已执行 chunk 聚合 breakdown，并选择最小可用显存。

例如 `serving_cost=3 ms`，第一个 10 ms chunk 完成 2 个请求，随后 20 ms chunk 再完成 2 个请求，则完成
时间分别为 13 ms 和 33 ms，平均 TTFT 为 `(2 * 13 + 2 * 33) / 4 = 23 ms`，吞吐所用的 Prefill 阶段
总延迟为 33 ms。

CLI 中过时的 `OptimizerData` 预构造和已禁用的 chunk 拒绝逻辑已经删除。CLI 在该阶段只加载并校验长度
分布文件；真正的 plan 在 optimizer 中得到 DP-rank concurrency 后生成。

## 6. 输出行为

Aggregation 与 Disaggregation 的变长输入结果均包含：

- 一条 `num_input_tokens="all"` 的汇总行；
- 多条包含代表性 `num_input_tokens`、归一化 `request_ratio` 和整数 `samples` 的明细行；
- 明细行不重复展示 TTFT、TPOT、吞吐和 breakdown。

整数采样结果为 0 的桶会被 `build_concurrency_samples()` 过滤，因此低权重桶在 DP-rank concurrency 较小
时可能不会出现在最终表格中。

## 7. 模块交互关系

```text
CLI (--input-length=<distribution.yaml>, --max-batched-tokens)
  │
  ├─ load_length_distribution()
  │
  └─ ParallelRunner
       │
       ├─ 构造 OptimizerData(length_distribution=..., max_batched_tokens=...)
       │
       ├─ run_agg()
       │    └─ AggThroughputOptimizer.get_inference_info(optimizer_data)
       │         ├─ optimizer_data.get_prefill_chunk_plan(DP-rank concurrency)
       │         ├─ optimizer_data.get_prefill_num_chunks(chunk_plan)
       │         └─ _get_batched_full_prefill_metrics(..., chunk_plan)
       │              └─ _get_batched_forward_info(..., chunk_plan)
       │
       └─ run_disagg()
            └─ DisaggThroughputOptimizer.get_inference_info(optimizer_data)
                 ├─ optimizer_data.get_prefill_chunk_plan(DP-rank concurrency)
                 ├─ optimizer_data.get_prefill_num_chunks(chunk_plan)
                 └─ 变长输入 Prefill 分支
                      └─ _get_batched_forward_info(..., chunk_plan)

两条路径均调用：

              BaseThroughputOptimizer._get_batched_forward_info()
                                      │
                    optimizer_data.build_concurrency_samples()
                                      │
                        校验 chunk index 连续且非递减
                                      │
                       按 index 对 PrefillChunk 分组
                                      │
                          遍历每个 chunk index
                                      │
                  为当前 index 构造异构 RequestInfo 列表
                                      │
              ModelRunner.run_inference(generate_inputs_varlen)
                                      │
              将 (metrics, completed_requests) 加入 chunk_results
                                      │
                    device_memory_available_gb < 0？
                         ├─ 是 → break，不再执行后续 chunk
                         └─ 否 → 继续下一个 chunk index ──────────────┐
                                      ▲                               │
                                      └───────────────────────────────┘
                                      │
                           循环结束或触发 early-stop
                                      │
              返回 list[(ModelRunnerMetrics, completed_requests)]
                                      │
                       ┌──────────────┴──────────────┐
                       ▼                             ▼
             Aggregation 指标聚合          Disaggregation 指标聚合
             - 请求加权 TTFT               - 请求加权 TTFT
             - 单独计算 Decode 延迟         - 全局输入 token 吞吐
             - 不计算 serving_cost          - serving_cost 只计算一次
             - 选择最紧张的 chunk 显存      - 聚合 breakdown 和显存
                       │                             │
                       └──────────────┬──────────────┘
                                      ▼
                         OptimizerSummary 汇总行 + 明细行
```

跨请求 chunk 打包示例：

```text
单 DP rank 请求：A=50、B=50、C=150、D=150 tokens
max_batched_tokens=200

chunk index 0（query token 总数为 200）：
  A: query_len=50,  seq_len=50,  is_last_chunk=True
  B: query_len=50,  seq_len=50,  is_last_chunk=True
  C: query_len=100, seq_len=100, is_last_chunk=False

  run_inference([A(50, 50), B(50, 50), C(100, 100)])
  completed_requests=2

chunk index 1（query token 总数为 200）：
  C: query_len=50,  seq_len=150, is_last_chunk=True
  D: query_len=150, seq_len=150, is_last_chunk=True

  run_inference([C(50, 150), D(150, 150)])
  completed_requests=2
```

请求 C 被拆到两个 chunk 中，而每个 chunk 又可以包含不同请求的片段。如果两次推理均耗时 10 ms，A、B
在 10 ms 完成，C、D 在 20 ms 完成，平均 TTFT 为 15 ms。

`ModelRunnerMetrics` 只保存模型执行指标。请求完成数由 ServingCast 在每次推理后根据
`PrefillChunk.is_last_chunk` 补充，因为它属于调度语义，不取决于模型 shape，也不应进入 ModelRunner 缓存。

## 8. 回归覆盖

当前回归测试覆盖：

- 非 chunk 和多 chunk 路径使用相同的结果列表结构；
- 非 chunk 完成请求数等于采样请求总数；
- Aggregation 按 token budget 划分变长 Prefill，并计算请求加权 TTFT；
- Disaggregation 使用全局输入 token 计算吞吐，而完成请求数保持单 DP rank 语义；
- Disaggregation 只添加一次 `serving_cost`，并聚合 chunk 显存和请求加权 TTFT；
- 在 `groupby` 执行前拒绝乱序或不连续的 chunk index。
