# RFC: Throughput Optimizer 当前 Mixed-Batch 变长 Token 建模实现

## 元数据

| 项目 | 内容 |
|:---|:---|
| **状态** | 已批准 |
| **作者** | stormchasingg |
| **更新日期** | 2026-05-15 |
| **相关链接** | |

---

## 1. 概述

本文档描述当前代码中 throughput optimizer 对变长 token 的 mixed-batch 建模实现。

与最初版本的 RFC 相比，当前实现仍然保留了 mixed-batch variable-token 的核心能力，但代码结构已经有较大重构：

- CLI 侧使用布尔开关 `--length-distribution`，而不是用户传入路径
- 分布解析与 workload 构造主要集中在 `OptimizerData`
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

当前 CLI 同时暴露：

- `--input-length`
- `--length-distribution`

并要求：

- 这两个参数必须二选一

该约束在 `cli/inference/throughput_optimizer.py` 中校验。

### 2.2 内置分布模式

`--length-distribution` 当前是布尔开关，不是文件路径参数。

开启后，CLI 会加载内置分布文件：

- `serving_cast/example/length_distribution.yaml`

当前该模式仅允许运行在：

- `--disagg`
- 指定 `--ttft-limits`
- 不指定 `--tpot-limits`
- 不启用 PD ratio 优化

若不满足这些条件，CLI 会直接拒绝执行。

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

`OptimizerData.get_max_effective_input_length()` 是 distribution 模式专用接口，CLI 用它做：

- `max_prefill_tokens` 检查

它会基于配置中的 bin 计算最大有效 prefill 长度。

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

当 `args.length_distribution is not None` 时，`OptimizerSummary._get_agg_disagg_final_out()` 会转到：

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
    ├─ --input-length / --length-distribution 二选一
    │
    ├─ 是否开启 --length-distribution？
    │   ├─ 否
    │   │   └─ 走标量 input_length 路径
    │   │
    │   └─ 是
    │       ├─ load_length_distribution()
    │       ├─ 构造 OptimizerData(length_distribution=...)
    │       ├─ 校验：
    │       │   ├─ 仅支持 disagg
    │       │   ├─ 仅支持 prefill-only（设置 --ttft-limits）
    │       │   └─ 不允许 --tpot-limits / 不允许 PD ratio 优化
    │       └─ 进入 distribution-aware prefill 路径
    │
    └─ ParallelRunner(args)
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
                ├─ length_distribution is None?
                │   ├─ 是 → _get_agg_disagg_final_out()
                │   │         └─ _get_disagg_table_buf()
                │   │
                │   └─ 否 → _get_agg_disagg_final_out_batched()
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

1. CLI 仅支持内置 YAML 文件，不支持自定义分布路径
2. distribution 模式仅支持 disaggregation prefill with `TTFT`
3. PD ratio 优化不支持 variable-token mixed-batch
4. 最优配置仍然先在 aggregate row 上筛选，再展开 detail rows

## 10. 后续变更提示

如果后续实现再次演化，最敏感、最需要同步更新文档的部分包括：

- `--length-distribution` 的 CLI 语义
- `OptimizerData` 的命名与 workload 构造辅助函数
- `BaseThroughputOptimizer` mixed-batch 执行入口
- `DisaggThroughputOptimizer` 的 summary row schema
- `OptimizerSummary` 的 batched final report 展示格式

尤其如果未来重新引入以下能力：

- 自定义 distribution 文件路径
- summary 子类
- decode 模式 batched report
- aggregation 模式的 variable-token 支持

建议单独作为后续 RFC 变更记录。 
