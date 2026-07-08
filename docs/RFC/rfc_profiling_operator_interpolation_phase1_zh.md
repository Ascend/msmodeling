# RFC: 实测算子接入插值模块设计（第一阶段）

## 元数据

| 字段 | 内容 |
|---|---|
| RFC 名称 | 实测算子接入插值模块设计（第一阶段：compute 与 attention_special 多维插值） |
| 状态 | Draft |
| 阶段 | Phase 1 / 3 |
| 目标模块 | `tensor_cast/performance_model/profiling_database` |
| 目标分支 | `develop` |
| 作者 | @zhenyu_zhang |
| 创建日期 | 2026-05-26 |
| 更新日期 | 2026-06-13 |
| 关联 Issue/PR | #229 |

## 文档对齐说明

中文版是本 RFC 的详版 master；英文版覆盖相同设计边界、默认行为、回退策略、测试要求和兼容性结论，但为便于英文读者阅读，将部分中文长章节拆成独立章节。两版若出现解释差异，以本文档的设计约束为准，并需要同步修正英文版，不允许引入不同的默认行为或验收标准。

## 1. 概述

本 RFC 是实测算子接入插值模块的第一阶段设计。第一阶段不只做单个 MatMul/GEMM M/K 2D 用例，而是完成一条可运行、可回退、可观测的插值主框架，并在第一阶段交付两类核心能力：

1. `compute` 类算子的多维插值，重点覆盖 MatMul/GEMM 的 M/K/N 1D/2D/3D 插值能力。
2. `attention_special` 子类的 1D/2D/3D 多维插值能力，基于当前 query 语义 `(batch, seq, heads, head_dim)` 尽最大可能实现；其中 `seq/avg_seq_len` 1D 是基础能力，`batch`、`heads`、`head_dim` 的 2D/3D 需要通过现有 Runtime 字段推导、CSV enrichment 补充或测试数据构造来验证，运行时仍必须检查字段可推导、候选点存在多个取值并能形成 boundary/grid。

第一阶段仍然保持范围收敛：不做 communication wrapper 二次插值，不启用外推，但参考 aiconfigurator 的插值能力引入 `scipy.interpolate.griddata` 作为通用多维插值后端；不把 `fallback_order` 暴露为配置项，不要求一次性改造所有 `op_mapping.yaml`。

总体查询闭环：

```text
profiling datasource 主路径接入
  -> exact lookup 优先
  -> exact miss / partial 时进入插值 fallback
  -> compute 或 attention_special allowlist
  -> ComputeIndex / AttentionIndex 查找候选点
  -> 1D linear / scipy.griddata 2D/3D / specialized grid fallback
  -> 返回 QuerySource.INTERPOLATED
  -> details / shape_match_info / debug log 可解释
```

## 2. 三阶段整体规划

### 阶段一：compute 与 attention_special 多维插值

阶段一交付：

- profiling 路径接入 `InterpolatingDataSource`。
- 增加 `--disable-profiling-interpolation` 回退开关。
- 新增公共数学模块 `interpolation_math.py`。
- 引入 `scipy.interpolate.griddata`，作为 2D/3D 通用插值后端。
- 新增候选点索引层，支持 compute 与 attention_special 的候选点组织。
- compute 类支持 1D/2D/3D 插值，重点交付 MatMul/GEMM 的 M/K/N 插值。
- attention_special 子类支持 1D/2D/3D 插值能力；第一阶段要实现可运行的 2D/3D 路径，当前 CSV 无法稳定表达 batch 或第三轴时降级到 `seq/avg_seq_len` 1D 或可证明可靠的低维组合。
- 输出 `source`、`confidence`、`details`、`shape_match_info` 和 debug 日志。
- 外推 guard 在第一阶段预留接口但默认关闭；目标点超出 boundary/凸包时只记录失败原因并回退，阶段一不返回 `EXTRAPOLATED`。

### 阶段二：MoE/DFC、elementwise 与 composite 扩展

阶段二在阶段一框架上扩展：

- `query_mode: moe_fused` 与 `DispatchFFNCombine` 插值。
- elementwise 的 1D / 条件 2D / 条件 3D 插值。
- composite 分解后复用子算子插值能力。
- FP8 量化 scale 类 compute 子场景，如 `compute_scale` / `scale_matrix` 对应的 M/K 数据表能力。

阶段二仍保持 communication 不做 wrapper 二次插值；communication 使用现有 base exact / alpha-beta(message_bytes) 路径。

### 阶段三：外推、精度评估、性能治理与工程化收敛

阶段三交付：

- opt-in 外推策略，默认关闭。
- ratio guard / max bound 外推保护。
- holdout 精度评估和误差报告。
- 性能预算、缓存策略、stale rebuild 和并发访问收敛。
- benchmark 覆盖 exact hit、1D/2D/3D 插值、schema miss、policy miss、index 构建、stale rebuild 和内存开销。
- metrics、warning、debug log 收敛。
- policy 字段精简和文档收敛。

### 三阶段整体架构图

```text
User / CLI / Config
  |
  |-- --performance-model profiling
  |-- --profiling-database <dir>
  |-- --disable-profiling-interpolation                  [Phase 1]
  v
ModelRunner / PerformanceModel Factory
  |
  |-- disable_profiling_interpolation = true
  |      |
  |      v
  |   ProfilingDataSource                                [base CSV/existing path]
  |      |
  |      |-- CSV 精确匹配 -------------------------------> QueryResult(source=MEASURED)
  |      |-- base 内部派生路径 --------------------------> QueryResult(source=INTERPOLATED)
  |      |     例如 communication alpha-beta/message_bytes
  |      |-- composite 部分命中 -------------------------> QueryResult(source=PARTIAL)
  |      |-- miss ---------------------------------------> None
  |
  |-- disable_profiling_interpolation = false
         |
         v
      InterpolatingDataSource                            [Phase 1]
         |
         | 1. 先调用 base.lookup(op)
         v
      ProfilingDataSource                                [existing]
         |
         |-- MEASURED -----------------------------------> return base result
         |-- INTERPOLATED -------------------------------> return base result
         |     base communication 插值仍由 base 路径负责
         |-- EXTRAPOLATED -------------------------------> 如果 base 未来返回该 source，则直接返回 base result
         |     Phase 1 wrapper 自身不得产生 EXTRAPOLATED
         |-- PARTIAL ------------------------------------> 尝试 wrapper interpolation fallback
         |-- None ---------------------------------------> 尝试 wrapper interpolation fallback
                |
                v
      Interpolation Fallback Dispatcher                  [Phase 1]
         |
         |-- category/query_mode allowlist
         |-- policy merge: built-in defaults
         |              + global interpolation_policy
         |              + operator_mappings.<op>.interpolation_policy
         |
         +---------------------------------------------------------------+
         |                                                               |
         v                                                               v
  communication                                                   compute
  category == communication                                       category == compute
  [existing / not wrapper interpolation]                          [Phase 1]
         |                                                               |
         |-- base exact / _query_comm_csv                                |-- CandidateIndex / ComputeIndex
         |-- alpha-beta(message_bytes)                                   |-- axes: M / K / N
         |-- no 1D/2D/3D wrapper policy                                  |-- regime keys:
         |                                                               |     kernel_type, dtype, format,
         |                                                               |     layout, transpose, other exact fields
         |                                                               |-- algorithms:
         |                                                               |     1D linear
         |                                                               |     2D/3D scipy.griddata(linear)
         |                                                               |
         |                                                               |-- FP8 quant scale subcases
         |                                                               |     compute_scale / scale_matrix
         |                                                               |     [Phase 2 if schema/policy is not
         |                                                               |      stable enough in Phase 1]
         |                                                               |
         |                                                               v
         |                                                        QueryResult(source=INTERPOLATED)
         |
         +---------------------------------------------------------------+
         |
         v
  attention_special
  query_mode == attention_special                                [Phase 1]
         |
         |-- CandidateIndex / AttentionIndex
         |-- axes from current query semantics and CSV enrichment:
         |     seq / avg_seq_len mandatory
         |     batch / heads / head_dim only when derivable and varying
         |-- regime keys:
         |     kernel_type, dtype, sparse_mode, kv_heads,
         |     quant_mode, layout, mask/cache regime
         |-- supported paths:
         |     direct attention_special ops
         |     attention sub-kernel fallback from decomposer
         |-- algorithms:
         |     1D linear
         |     2D/3D scipy.griddata(linear)
         |-- optional axis_transform:
         |     sqrt(seq) for known O(seq^2) kernels
         |
         v
  QueryResult(source=INTERPOLATED)


Phase 2 Extensions
  |
  |-- moe_fused / DispatchFFNCombine
  |     |-- axes: tokens and stable expert/EP-related continuous fields
  |     |-- exact/regime keys: top-k, expert layout, EP, dtype, kernel regime
  |
  |-- elementwise
  |     |-- default 1D: output elements or total bytes
  |     |-- conditional 2D/3D:
  |           bytes + rank / broadcast / batch / stride if stable
  |
  |-- composite
  |     |-- static sub_kernels from op_mapping.yaml
  |     |-- dynamic decomposer, including MLA/MLAPO full composite handling
  |     |-- sub-kernel miss reuses its category policy
  |
  |-- FP8 quant scale compute subcases
        |-- compute_scale: dynamic FP8 scale computation overhead
        |-- scale_matrix: static quantization matrix scale overhead
        |-- M/K policy when CSV schema is stable


Phase 3 Extensions
  |
  |-- ExtrapolationPolicy
  |     |-- allow_extrapolation default false
  |     |-- opt-in only
  |     |-- ratio guard / max bound
  |     |-- QuerySource.EXTRAPOLATED
  |     |-- lower confidence than interpolation
  |
  |-- Accuracy / Holdout Evaluation
  |     |-- compute holdout
  |     |-- attention holdout
  |     |-- later MoE/DFC/elementwise/composite holdout
  |
  |-- Observability Convergence
  |     |-- details normalization
  |     |-- shape_match_info normalization
  |     |-- confidence printed in final user-visible output
  |     |-- confidence print-only, not result selection
  |     |-- metrics:
  |           MEASURED / INTERPOLATED / EXTRAPOLATED / PARTIAL / MISS
  |
  |-- Policy / Documentation Cleanup
        |-- distinguish implemented fields from reserved fields
        |-- keep fallback_order reserved unless fixed order is insufficient
        |-- migration notes and regression checklist
```

## 3. 第一阶段目标与非目标

### 3.1 目标

1. 在 profiling datasource 主路径中接入插值 wrapper，并提供回退开关。
2. 保持 `MEASURED` 精确命中优先，插值只作为 exact miss 或 `PARTIAL` 的 fallback。
3. 新增公共数学模块，支持 boundary、1D linear、`scipy.interpolate.griddata` 2D/3D、finite 校验和 confidence 计算。
4. 新增候选点索引层，将 CSV 候选点组织成按 regime key 和连续轴查询的 bucket。
5. compute 类支持 1D/2D/3D 插值，MatMul/GEMM 明确支持 M/K/N 多维插值。
6. attention_special 子类支持 `seq/avg_seq_len` 1D，并实现 `batch`、`heads`、`head_dim` 参与的 2D/3D 插值路径；这些轴只有在 CSV 字段可稳定推导且候选点存在多个取值时才参与运行时插值。
7. 插值成功返回 `QuerySource.INTERPOLATED`，并通过 `details`、`shape_match_info`、`confidence` 和 debug log 记录系统判定过程。
8. 第一阶段仅预留外推 guard 接口但不启用外推结果；目标点超出 boundary 或凸包时记录失败原因并回退，外推留到阶段三启用。

### 3.2 非目标

第一阶段不做以下内容：

- 不实现 communication wrapper 插值。
- 不实现 MoE / DFC / elementwise / composite 的完整插值能力。
- 不启用外推，不返回 `QuerySource.EXTRAPOLATED`；阶段一只预留外推 guard 接口，不接入结果返回路径。
- 不使用 scipy/griddata 做外推；griddata 只用于实测点凸包或 boundary 范围内的插值。
- 不引入 scipy 之外的其他重依赖。
- 不将 `fallback_order` 作为可配置字段。
- 不把性能 benchmark 作为硬验收项。
- 不要求一次性改造所有 `op_mapping.yaml` 条目。

## 4. 当前状态与问题

当前 profiling 模型已有 `ProfilingDataSource`，并且主仓中已有基础 `InterpolatingDataSource` 原型，但普通 `--performance-model profiling` 主路径仍直接构造 `ProfilingDataSource`。因此在默认使用路径下，CSV exact miss 后不会稳定进入统一插值 fallback。

当前主仓已识别 `query_mode: attention_special`，基础语义是按 `(batch, seq, heads, head_dim)` 查询 attention 类 CSV；但插值能力尚未形成统一的多维候选点索引和可解释输出。

当前需要优先解决的问题是：

1. 主路径没有统一接入插值 wrapper。
2. 插值数学逻辑散落在 datasource 分支里，不利于扩展 2D/3D。
3. compute 类缺少统一 M/K/N 多维插值能力。
4. attention_special 缺少基于 `seq/avg_seq_len`、`batch`、`heads/head_dim` 的统一 1D/2D/3D 多维插值能力；现有 CSV 字段不完整时还缺少稳定推导、降级和可解释输出机制。
5. 查询结果对调用方可观测性不足，难以判断结果来自实测、插值还是 fallback。
6. policy 表达容易过宽，需要第一阶段先收敛到最小必要字段。

## 5. 方案设计

### 5.1 主路径接入

`--performance-model profiling` 默认接入 `InterpolatingDataSource`，外层 wrapper 只在 exact miss / partial 后尝试插值；命中实测结果时直接返回 base `ProfilingDataSource` 结果。

新增回退开关 `--disable-profiling-interpolation`，对应用户配置 `disable_profiling_interpolation: bool = False`。启用该开关后 profiling datasource 使用 base `ProfilingDataSource` 行为，不进入 wrapper 的 compute/attention 插值 fallback。这里不是说 base 路径所有结果都必须来自 CSV 精确命中；base 内部已有的 communication alpha-beta/message_bytes 等专门派生路径仍保持原状。

第一阶段 wrapper 只对 compute 与 attention_special allowlist 场景生效。未覆盖算子仍保持 base `ProfilingDataSource` 行为，避免默认开启后扩大行为变化范围。

### 5.2 查询流程

第一阶段查询流程如下：

```text
lookup(op)
  -> base.lookup(op)
      -> MEASURED: return base result
      -> INTERPOLATED: return base result
         # 可能来自现有 base communication 路径
      -> EXTRAPOLATED: 如果 base 未来返回该 source，则直接返回 base result
         # Phase 1 wrapper 自身不得产生 EXTRAPOLATED
      -> PARTIAL: 尝试 wrapper interpolation fallback
      -> None: 尝试 wrapper interpolation fallback
  -> if op is communication:
         return base result
  -> if op is not compute or attention_special allowlist:
         return base result
  -> build interpolation target
  -> query candidate index
  -> try 1D / 2D / 3D by fixed default order
  -> if interpolation success:
         return QueryResult(source=INTERPOLATED)
     else:
         return base result
```

关键原则：

- exact hit 永远优先。
- 插值只处理 miss / partial fallback。
- 插值失败不抛出业务异常，记录原因后回退到原行为。
- communication 不进入 wrapper 插值。communication 继续走 `ProfilingDataSource` 已有 exact / alpha-beta(message_bytes) 路径；该 base 路径内部已经可能返回 `QuerySource.INTERPOLATED`，因此这里的“不插值”只表示不再由 `InterpolatingDataSource` 额外做 1D/2D/3D wrapper 插值。

默认尝试顺序固定为：

```text
exact -> 1D -> 2D -> 3D
```

含义是优先寻找“只有一个连续轴需要插值”的候选点；如果低维候选因为固定轴太严格无法形成 boundary，再扩大到 2D / 3D。第一阶段不将该顺序暴露为 `fallback_order` 配置。实现上不要求逐个维度全量重查，索引层应一次性返回各轴的 boundary/grid 可用性，再选择最低可用维度；compute 轴优先级建议为 `M -> K -> N`。

`PARTIAL` 只表示 base `ProfilingDataSource` 已经产生了部分结果。第一阶段 wrapper 不做“对子 kernel 缺口局部补齐”的混合结果，而是把 `PARTIAL` 当作一次完整 wrapper 插值重试的入口：如果 wrapper 能独立生成完整插值结果，则返回 `QuerySource.INTERPOLATED` 并在 details 记录 `fallback_from="partial"`；如果 wrapper 插值失败，则返回 `None`，让上层 empirical 路径使用 analytic fallback，避免把只覆盖部分子 kernel 的 latency 当作完整 op latency。

### 5.3 InterpolatingDataSource 与候选点索引

`InterpolatingDataSource` 是 datasource wrapper，负责接入 base lookup、判断 fallback、调用索引和数学模块。

继承关系：

```text
DataSourcePerformanceModel
  ↑
InterpolatingDataSource
  └── 持有 ComputeIndex / AttentionIndex

ComputeIndex / AttentionIndex
  ↑
object
```

索引组件不是新的性能模型，也不继承 `DataSourcePerformanceModel`。它只是内部候选点索引组件，用于把 CSV 行预处理成按 regime key 和连续轴组织的 bucket，避免每次插值都全表扫描。

第一阶段建议可以实现一个通用 `CandidateIndex`，内部按 category/query_mode 分 bucket；也可以先实现 `ComputeIndex` 与 `AttentionIndex` 两个内部类。关键要求是索引层输出统一候选点结构，而不是把 1D/2D/3D 的 shape 过滤逻辑散落到各 query_mode 分支中。

候选点基础结构建议：

```python
CandidatePoint(
    regime_key=...,
    axes={"M": 1024, "K": 4096},
    latency=...,
    row_meta={...},
)
```

缓存策略：

- 索引懒加载构建，不在 `InterpolatingDataSource.__init__` 中全量扫描所有 CSV。
- 复用 `ProfilingDataSource._load_csv()` 返回的 DataFrame。
- 索引缓存保存在 wrapper 实例内，不使用全局缓存。
- 不使用 `id(df)` 作为缓存 key。建议使用稳定内容指纹，例如 `(kernel_type, df_content_hash, policy_hash, index_kind, tc_input_count)`；`df_content_hash` 应覆盖 DataFrame 行数、列名、索引和值内容，不能只取 first/last row，避免中间行变化后复用陈旧索引。如果可获得 CSV path、mtime 或 profiling database version，也可以纳入指纹。
- 当 base `_csv_cache` 被显式清理时，wrapper index cache 也必须同步清理。

`policy_hash` 不是用户配置字段，而是索引构建器根据归一化后的有效插值 policy 生成的确定性哈希。有效 policy 的来源包括内置默认值、全局 `interpolation_policy` 和单算子 `interpolation_policy`。建议使用按 key 排序的 canonical JSON 序列化，并稳定表达缺失值，再计算 SHA-256。只要有效 axes、exact fields、max interpolation dimension、axis transform、外推开关或重复点聚合策略发生变化，`policy_hash` 就必须变化。

### 5.4 公共数学模块

新增模块：

```text
tensor_cast/performance_model/profiling_database/interpolation_math.py
```

第一阶段参考 aiconfigurator 的 `aiconfigurator.sdk.interpolation` 能力，引入 `scipy.interpolate.griddata` 作为 2D/3D 通用插值后端。aiconfigurator 中 2D/3D 插值会先抽取相邻候选点，再调用 `griddata(..., method="linear")`；部分 3D 场景也保留“2D 后 1D”的专用路径。本阶段采用同样思路，但要与 msmodeling 的 `QueryResult.details`、`shape_match_info`、`source` 和 confidence 输出对齐。

第一阶段实现函数：

| 函数 | 职责 |
|---|---|
| `find_boundary(values, target)` | 在有序候选值中查找包住 target 的 lower / upper。阶段一只做区间内插值；target 超出范围时返回失败。 |
| `linear_interp(x, x0, y0, x1, y1)` | 一维线性插值。 |
| `griddata_linear_interp(points, values, target)` | 通用 2D/3D 线性插值后端，内部调用 `scipy.interpolate.griddata(..., method="linear")`。阶段一不开放其他 method。 |
| `validate_latency(value)` | 校验 latency 是 finite 且非负。 |
| `estimate_confidence(...)` | 根据插值维度、boundary 宽度、候选点密度、是否 transform 和是否降级给出基础 confidence。 |
| `build_interpolation_details(...)` | 生成 details 字段，记录轴、方法、候选点、boundary、confidence 和失败原因。 |

预留但第一阶段不启用：

| 函数 | 说明 |
|---|---|
| `check_extrapolation_guard(target, bounds, policy)` | 供阶段三外推使用。阶段一只返回 guard 判断和失败原因，不据此生成 `QuerySource.EXTRAPOLATED` 结果，也不改变插值回退路径。 |

阶段一中 `find_boundary` 或 griddata 凸包检查失败时仍记录 `outside_boundary` / `outside_convex_hull` 等原因并回退；`check_extrapolation_guard` 仅作为策略接口占位，用于表达 ratio guard / max bound 等后续外推保护。

griddata 使用约束：

- 默认只启用 `method="linear"`，函数接口不暴露 `cubic` 等其他 method。
- 不使用 griddata 做外推；target 超出候选点凸包、griddata 返回 `nan` 或抛出 Qhull/ValueError 时，当前插值尝试失败并回退。
- 2D griddata 至少需要 3 个非共线点，3D griddata 至少需要 4 个非共面点；点数不足、共线、共面或 rank 退化时先降级到更低维候选策略，无法降级时回退。
- 第一阶段只保留 `griddata_linear_interp(points, values, target)` 作为 2D/3D 公共接口。规则网格 bilinear/trilinear 的专用函数暂不实现；后续如果发现规则网格路径有明显精度、性能或可解释性收益，再作为遗留问题单独补充。
- `method="cubic"` 第一阶段不默认启用；如后续 attention/DSA 类场景需要，应通过 policy 显式声明并补充精度评估。

依赖要求：

- 当前仓库 `requirements.txt` 已包含 `scipy`，第一阶段无需新增第三方依赖。
- 运行时如 scipy 不可用，应在加载或首次使用插值模块时报出清晰错误；不应静默退回到不一致的插值实现。
- 文档和测试需要覆盖 scipy 依赖缺失、griddata 返回 `nan`、候选点退化等场景。

### 5.5 compute 类多维插值

第一阶段 compute 类支持 1D/2D/3D 插值。核心交付是 MatMul/GEMM M/K/N。

#### 轴定义

对 GEMM 语义：

```text
输入激活: [..., M, K]
权重:     [K, N] 或 [N, K]，取决于 transpose / layout
输出:     [..., M, N]
```

连续轴：

- `M`：输入激活的行数或 flatten 后 token/row 数。
- `K`：reduce 维度。
- `N`：输出列维度。

精确匹配字段：

- kernel type。
- dtype。
- format。
- layout。
- transpose / weight layout 的 canonicalized 结果；不能比 exact lookup 更严格。
- 其他会改变 kernel 语义的字段。

`input_formats` 是 CSV 物理格式信息，候选点需要保留在 `row_meta/details` 中，便于解释来源。第一阶段 target 不应因为 runtime 侧是逻辑 ND tensor 就把可 canonicalize 的 FRACTAL_NZ 候选全部排掉；但候选组仍按 CSV `input_formats` 分组，且当同一 target 同时存在 ND 与非 ND 候选组时，优先使用 ND 候选组。跨物理格式复用只允许发生在 shape canonicalization 能证明 M/K/N 语义一致的场景。

M/K/N 抽取规则：

| 场景 | M | K | N | 处理要求 |
|---|---|---|---|---|
| ND `[M,K] x [K,N]` | input0[0] | input0[1] | input1[1] | `source_layout=KN` 只进入 row_meta/details；只有 exact lookup 也区分该字段时才进入 regime key |
| ND `[M,K] x [N,K]` | input0[0] | input0[1] | input1[0] | 对应 F.linear / transpose 权重，`source_layout=NK` 只进入 row_meta/details，不能导致比 exact 更严格的 bucket 分裂 |
| `BatchMatMulV2` / `TransposeBatchMatMul` 3D `(H,M,K) x (H,K,N)` | input0[1] | input0[2] | input1[2] | `H` 作为 batch/head regime 或 exact 字段；若 input1 为 `(H,N,K)`，按转置形态解析 N=input1[1] |
| `_FLATTEN_BATCH_KERNELS` 中的 batched / flatten batch | `prod(batch_dims) * M` | input0[-1] | 按 exact lookup 支持的规则解析 | 仅适用于 exact lookup 已启用 flatten batch 的 kernel；MatMul 类当前不能套用该规则 |
| FRACTAL_NZ 权重 | input0 解析后 M | input0 解析后 K | 先 `fractal_nz_to_nd()` 再解析 | 不能直接用 tiled shape 抽轴 |
| 无法判断权重 layout | 无 | 无 | 无 | 不进入 M/K/N 多维插值，回退或只保留现有安全 1D |

CSV 候选点和 runtime target 必须使用同一套抽取规则。若 K 同时能从 input0 和 input1 推导，二者不一致时该候选点无效，不能进入 bucket。

`Accelerator Core` 虽然存在于部分 CSV 列中，但当前 exact lookup 没有把它作为匹配条件。第一阶段插值也不能把它放入 regime key；否则会出现 exact lookup 能命中的候选行在插值索引中被拆成不同 bucket 的问题。若后续确实需要区分 AIC/AIV/MIX_AIC，应先扩展 exact lookup，再同步到插值索引。

RoPE / SwiGlu / block padding 等 `_inputs_match` 特殊规则必须与 exact lookup 对齐。第一阶段如果没有为这些 kernel 实现专门的多维轴抽取器，应在多维 allowlist 中排除，避免误进入 M/K/N 2D/3D 插值路径。

#### 1D / 2D / 3D 策略

第一阶段 compute 查询顺序：

```text
exact -> 1D -> 2D -> 3D
```

1D：

- 只允许一个连续轴变化，例如 M 变化，K/N exact。
- 使用 `linear_interp`。

2D：

- 允许两个连续轴变化，例如 M/K 变化，N exact。
- 使用 `griddata_linear_interp`，规则矩形网格也先通过统一 `points/values/target` 接口表达。
- 目标点必须被四个角点围住并形成有效 boundary。

3D：

- 允许 M/K/N 三个连续轴变化。
- 使用 `griddata_linear_interp`。规则 3D 网格的 trilinear 专用实现不进入第一阶段。

3D 不是“先做 2D，再额外补一个 1D fallback”，而是一个三维插值算法：目标点在三维候选点集合内被邻近点包围。第一阶段统一使用 scipy 的线性单纯形插值。

#### 候选点要求

1D 需要两个 boundary 点。

2D 对规则网格需要四个角点：

```text
(M_lo, K_lo)  (M_lo, K_hi)
(M_hi, K_lo)  (M_hi, K_hi)
```

3D 对规则网格需要两个平面上的八个角点：

```text
N_lo plane: (M_lo,K_lo), (M_lo,K_hi), (M_hi,K_lo), (M_hi,K_hi)
N_hi plane: (M_lo,K_lo), (M_lo,K_hi), (M_hi,K_lo), (M_hi,K_hi)
```

对于 griddata 路径，候选点数量必须满足对应维度的最小要求，并且 target 必须位于候选点凸包内。缺少必要点、点集退化、griddata 返回 `nan` 或有限性校验失败时，不强行插值；记录原因并回退。

#### compute 子场景边界

第一阶段 compute 以通用 shape 解析和 MatMul/GEMM 语义为主。

`compute_scale` / `scale_matrix` 不写成 msmodeling 顶层算子类别，更准确的表达是：

```text
FP8 量化 scale 类 compute 子场景
```

其中：

- `compute_scale` 表示动态 FP8 量化中的 scale 计算开销。
- `scale_matrix` 表示静态量化对矩阵应用 scale 的开销。

这类子场景如果当前 CSV schema 能稳定表达 M/K，可归入 compute M/K 2D 能力；如果缺少明确字段，放到阶段二随 FP8 量化 scale 场景补充。

### 5.6 attention_special 子类多维插值

当前 `op_mapping.yaml` 已将 `attention_special` 描述为按 `(batch, seq, heads, head_dim)` 查询。第一阶段 attention 插值基于这个已有语义，而不是引入全新的 attention schema。但当前仓库中不同版本的 `FusedInferAttentionScore.csv` 字段并不一致：老版本可能只有基础 shape 列，enriched 版本有 `Runtime avg_seq_len`、`Runtime num_heads`、`Runtime num_key_value_heads`、`Runtime actual_seq_lengths_shape/values` 等列，但没有独立 `Runtime batch_size` 列。因此第一阶段不能只停留在 1D；需要实现 attention_special 的 2D/3D 查询、校验和插值代码路径，并通过字段推导、CSV enrichment 或测试 CSV 验证这些能力。生产数据缺少必要字段或候选点时，运行时降级到低维或回退。

#### 基础轴定义

连续轴候选：

- `seq`：sequence length 或平均 sequence length。
- `batch`：batch size；当前 CSV 没有独立 batch 列，只能从 `Runtime actual_seq_lengths_shape/values` 或 runtime `seq_lens.shape[0]` 推导，推导失败时不能作为插值轴。
- `heads`：attention head 数；同一模型内通常固定，只有 CSV 中存在多个不同取值并能 boundary 时才可作为连续轴。
- `head_dim`：每个 head 的维度；通常隐含在 Q shape 最后一维，同一模型内通常固定，只有 CSV 中存在多个不同取值并能 boundary 时才可作为连续轴。

`batch` 轴必须谨慎处理。实现前需要统计当前生产 CSV 中 `Runtime actual_seq_lengths_shape` 和 `Runtime actual_seq_lengths_values` 的实际分布，确认它们表达的是 batch 数，而不是标量值、ndim 或单条 seq_len。若字段长期只有 batch=1 或无法可靠解释，生产查询中 `seq + batch` 2D 不应触发；代码路径仍需要通过 synthetic/enrichment CSV 覆盖。

精确匹配字段：

- kernel type，例如 `FusedInferAttentionScore`。
- dtype。
- sparse mode / mask mode。
- kv head regime。
- quant mode。
- layout。
- 其他会改变 attention kernel 语义的字段。

`q_tokens` 不放入 attention regime key，避免把同一 attention 场景拆成过细的候选组；但 `q_tokens` 仍作为 target/candidate axes 中的非插值轴参与精确过滤。也就是说，第一阶段不会沿 `q_tokens` 插值，也不会把 `q_tokens=2` 与 `q_tokens=4` 的候选点混合聚合。

#### 子类范围

第一阶段 attention_special 覆盖两类入口：

1. 直接映射为 `query_mode: attention_special` 的 attention 算子，例如 `tensor_cast.attention.default`、`tensor_cast.attention_quant.default`。
2. composite decomposer 产生的 attention sub-kernel fallback，例如 MLA decomposition 中生成的 `FusedInferAttentionScore` attention 子查询。第一阶段只复用 attention sub-kernel 插值能力，不要求实现完整 composite 插值能力。

#### 维度策略

attention_special 默认策略：

```text
exact -> 1D -> 2D -> 3D
```

推荐轴优先级：

```text
seq/avg_seq_len -> batch(if derivable and varying) -> heads/head_dim(if varying)
```

原因：

- `seq` 通常是 attention latency 变化最敏感的连续轴。
- `batch` 具备连续意义，但当前 CSV 没有独立 batch 列，必须可从 actual_seq_lengths 或补充 enrichment 字段稳定推导。
- `heads` / `head_dim` 在同一模型 profiling 数据中通常固定，更常作为 regime key；只有 CSV 中能稳定表达、存在多个不同取值并且候选点足够时才参与插值。

#### 子类能力表

| 子类 / 场景 | 第一阶段能力 | 连续轴 | 说明 |
|---|---|---|---|
| FusedInferAttentionScore / raw attention | 1D/2D/3D 必做代码路径 | `seq/avg_seq_len`、`batch`、`heads` 或 `head_dim` | 当前 enriched CSV 主力轴是 `Runtime avg_seq_len`；batch 需从 actual_seq_lengths 推导或由 enrichment 补列。2D/3D 路径第一阶段必须实现，运行时按数据条件触发。 |
| attention_quant | 1D/2D/3D 必做代码路径 | `seq/avg_seq_len`、`batch`、`heads` 或 `head_dim` | quant mode / scale 相关字段必须 exact，不跨 quant regime 插值。2D/3D 只在量化 regime 一致且候选点完整时触发。 |
| MLA decomposer 产生的 attention sub-kernel | 1D/2D/3D 必做代码路径 | `avg_seq_len`、`batch`、`num_heads` 或 `head_dim` | 只对 decomposer 生成的 attention 子查询 fallback；完整 composite 插值放阶段二。 |
| decode attention | 1D/条件 2D | `avg_seq_len` 或 past KV 长度语义；可选 `batch` | decode 常见 query_len=1，插值轴不是 query seq=1，而是平均上下文长度 / past KV 长度。 |
| context attention | 1D 必做，条件 2D/3D | `seq/avg_seq_len`；可选 `batch`、`heads/head_dim` | 如果 CSV 能形成完整 2D/3D boundary，则启用对应维度。 |

注意：表中的 2D/3D 是第一阶段要实现的能力，不是仅预留接口；但不是每次查询都强制触发。运行时必须检查字段存在、字段可推导、候选点存在多个取值、boundary/平面完整、latency 有效。任一条件不满足时降级到低维或回退。

#### attention axis transform

attention latency 对 seq 可能呈非线性关系。第一阶段允许保留已有的 `axis_transform: sqrt` 思路，但只作为内置或 kernel override 的保守能力使用：

- 默认先使用原始 seq 线性插值。
- 对已知 O(seq^2) attention kernel，可配置或内置使用 `sqrt(seq)` 空间插值；如果进入 2D/3D，seq 轴也应先应用相同 transform 后再参与 griddata。该 transform 必须同时作用于 CSV 候选点坐标和 runtime target 坐标。
- 使用 transform 时必须在 details 中记录 `axis_transform` 和 confidence 降级。

decode attention 需要额外注意：

- decode attention 常见 `query_len=1`，插值轴不是单 token query length，而是 `avg_seq_len` 或等价 runtime metadata 表达的平均上下文长度 / past KV 长度。
- 如果 `avg_seq_len` 来自 token 总量估算，实现必须与当前 profiling matcher 的 block-padding 容差对齐。现有 FIA 匹配逻辑允许小的 block 级 gap，例如 `_fia_avg_seq_len_gap` 风格的 16 token 左右容差。第一阶段插值不能比 exact lookup 更严格，不能拒绝 exact lookup 在 block padding 后会接受的候选点。
- 使用 `sqrt(seq)` 时，details 应同时记录 transform 前后的 target / boundary 坐标，便于排查 block-padding 对插值的影响。

## 6. 最小 policy 设计

第一阶段 policy 只保留必要表达，不引入过宽字段。

必须支持：

| 字段 | 第一阶段用途 |
|---|---|
| `axes` | 声明允许插值的连续轴。compute 可为 `["M", "K", "N"]`；attention_special 第一阶段默认声明 `["seq", "batch", "heads", "head_dim"]`，运行时根据字段可推导性和候选点完整性选择实际维度。 |
| `exact_fields` / `regime_keys` | 声明必须精确匹配的离散字段，例如 kernel type、dtype、format、layout、transpose、sparse mode、quant mode。 |
| `max_interpolation_dim` | 限制最高插值维度。compute/MatMul 可为 3；attention_special 第一阶段默认上限为 3，但运行时必须满足 CSV enrichment 和候选点条件才能使用 2D/3D。 |
| `allow_extrapolation` | 外推策略预留字段；阶段一仅作为内部 guard 参数，不作为用户可生效配置。默认 false，不返回外推结果；如果 YAML 中已有该字段，应按 reserved/no-effect 处理。 |
| `axis_transform` | 仅 attention_special 需要时使用，例如 `sqrt(seq)`；不是所有算子默认启用。 |

暂不支持：

| 字段 | 原因 |
|---|---|
| `fallback_order` | 第一阶段固定顺序，避免 policy 过度复杂。 |
| communication 多维 policy | communication 不做 wrapper 插值。 |
| 外推结果返回 | 阶段三再启用；阶段一只保留 guard 接口和 policy 字段。 |

全局 policy 与单算子 policy 的关系：

```text
内置默认 policy
  -> 全局 interpolation_policy 覆盖通用默认
  -> operator_mappings.<op>.interpolation_policy 覆盖单算子特殊行为
```

第一阶段不要求所有 op_mapping 都显式配置 policy。未配置时按 query_mode/category 合成保守默认值；只有需要突破默认能力或声明特殊轴时才配置单算子 policy。

## 7. 可观测输出

插值成功时：

```python
source = QuerySource.INTERPOLATED
```

`details` 建议包含：

```python
{
    "method": "griddata_linear",
    "interpolation_dim": 2,
    "axes": ["M", "K"],
    "exact_fields": {...},
    "target": {"M": M, "K": K, "N": N},
    "boundary": {
        "M": [M_lo, M_hi],
        "K": [K_lo, K_hi],
    },
    "corner_points": [...],
    "confidence": 0.7,
    "axis_transform": None,
    "fallback_from": "exact_miss",
}
```

attention_special 需要额外记录：

```python
{
    "query_mode": "attention_special",
    "attention_axes": {
        "batch": batch,
        "seq": seq,
        "heads": heads,
        "head_dim": head_dim,
    },
    "regime_keys": {
        "sparse_mode": sparse_mode,
        "kv_heads": num_kv_heads,
        "quant_mode": quant_mode,
    }
}
```

`shape_match_info` 建议记录：

- `shape_match_rule = "interpolated_1d_linear"` / `"interpolated_2d_griddata_linear"` / `"interpolated_3d_griddata_linear"`。
- `source = "INTERPOLATED"`。
- target shape。
- matched candidate shapes。
- fallback reason。

debug log 至少记录：

```text
op=<name> query_mode=<mode> source=INTERPOLATED dim=<1|2|3> axes=<...> method=<...> confidence=...
```

confidence 第一阶段只用于最终回显、statistics、debug 和 metrics，不作为调度硬门槛。是否采用 profiling / interpolation 结果仍由 `source` 和是否命中决定；不能因为 confidence 低于某个阈值自动切回 analytic fallback，也不新增基于 confidence 的 warning / fallback / policy 分支。

confidence 第一阶段不是统计意义上的置信区间，而是初始观测标签。固定基础值只能表达来源和相对风险，后续阶段需要通过 holdout 评估按真实误差分布校准。第一阶段可把 boundary 宽度、候选点密度、是否使用 transform、是否从高维降级等信息写入 details，并用于轻量调整 confidence，但不能让 confidence 参与结果选择。

### 7.1 最终回显中的 confidence

第一阶段要求最终用户可见输出中展示每个算子的结果来源和置信度，方便用户判断该算子性能估计的可信程度。当前 msmodeling 已经在 `QueryResult` 中携带 `confidence`，并在 `EmpiricalPerformanceModel` 的 statistics 中写入；本阶段需要把该字段贯通到最终回显、summary 或 report 中。

建议最终回显至少包含以下字段：

```text
op_name
source
confidence
latency
interpolation_dim
interpolation_method
shape_match_rule
```

示例：

```text
Profiling interpolation summary
op_name                                      source        confidence  method              axes       latency_us
aten.mm.default                              INTERPOLATED  0.70        griddata_2d_linear  M,K        123.4
tensor_cast.attention.default               INTERPOLATED  0.60        griddata_3d_linear  seq,batch,head_dim  456.7
tensor_cast.reshape.default                  MEASURED      1.00        exact               -          0.0
```

若已有输出格式不适合直接加表格，也必须保证每个 op 的 statistics/debug 信息中包含：

```python
{
    "source": "INTERPOLATED",
    "confidence": 0.7,
    "interpolation_dim": 2,
    "method": "griddata_linear",
}
```

confidence 的建议含义：

| confidence | 建议含义 |
|---|---|
| `1.0` | exact / zero-cost / accepted miss 等确定性结果。 |
| `0.8-0.9` | 实测数据路径但经过组合、通信拟合或较轻量派生。 |
| `0.6-0.7` | 插值结果；维度越高、axis transform 越多，confidence 越低。 |
| `<0.6` | composite interpolation、较弱近似或未来外推结果；仅作为低置信度标签展示。 |

上述数值是观测标签，不是数学置信区间，也不是结果选择阈值。

## 8. 外推接口

第一阶段预留外推接口，但默认关闭。

语义：

- 插值：target 被实测点上下界包围，例如已测 M=4、6，查询 M=5。
- 外推：target 超出实测范围，例如已测最大 M=6，查询 M=7。

第一阶段行为：

```text
allow_extrapolation = false
```

因此：

- 不对范围外 target 推算。
- 不返回 `QuerySource.EXTRAPOLATED`。
- 如果 target 超出 boundary，记录 `outside_boundary` 并回退。

阶段三再启用 opt-in 外推，并要求 ratio guard / max bound 通过。

## 9. communication 处理

第一阶段不对 communication 做 wrapper 插值。

communication 当前应保持现有路径：

```text
ProfilingDataSource exact lookup
  -> base 内部已有 alpha-beta(message_bytes) 逻辑
  -> fallback
```

`InterpolatingDataSource` 不再对 communication 应用额外 1D/2D/3D policy。这样可以避免把通信消息大小拟合和算子 shape 插值混在同一层，降低第一阶段风险。

## 10. 候选点聚合与校验

候选点处理拆成两层：先做语义过滤，再做插值结构校验。

1. 使用同一套 shape 规则规范化 CSV 行和 runtime target。
2. 按 policy / regime 字段过滤候选点，确保 dtype、format、layout、transpose、attention regime、quant mode 等语义一致。
3. 校验 latency，拒绝 NaN、inf、负值和缺失值。
4. 按插值轴 tuple 聚合候选点。
5. 对重复点做聚合，默认使用 mean latency，并在 details 中记录 raw / aggregated 计数。
6. 构造 1D boundary、2D grid/convex hull 或 3D grid/convex hull。
7. 只有结构完整且 target 位于有效范围内时才执行插值。
8. details 中记录 raw count、filtered count、aggregated count、dropped count、boundary、corner points、method 和降级/回退原因。

第一层过滤负责排除语义不一致的候选点；第二层校验只判断剩余候选点是否能形成合法插值结构。不能把结构不足误判为 exact miss，也不能用低维候选点伪造高维 grid。

## 11. 测试设计

### 11.1 单元测试

建议新增：

```text
tests/regression/tensor_cast/test_interpolation_math.py
```

覆盖：

- `find_boundary` 正常 boundary。
- target 小于最小值 / 大于最大值时返回失败。
- `linear_interp`。
- `griddata_linear_interp` 2D/3D。
- griddata target 在凸包内时返回 finite latency，且与直接调用 `scipy.interpolate.griddata(..., method="linear")` 的结果一致。
- griddata target 超出凸包、点集退化或返回 `nan` 时失败并回退。
- `griddata_linear_interp` 覆盖规则网格和非规则点集；bilinear/trilinear 专用接口不作为第一阶段测试目标。
- latency finite / non-negative 校验。
- confidence 基础计算。

建议新增：

```text
tests/regression/tensor_cast/test_profiling_interpolation_phase1.py
```

覆盖：

- compute CSV 行构建 M/K/N bucket。
- `BatchMatMulV2`、`TransposeBatchMatMul` 的 `(H,M,K) x (H,K,N)` 轴抽取正确，`H` 作为 batch/head regime 或 exact 字段处理。
- `_FLATTEN_BATCH_KERNELS` 的 flatten batch 规则不误用于 MatMul 类 kernel。
- RoPE / SwiGlu 等没有专门多维轴抽取器的 compute kernel 不会意外进入 M/K/N 多维插值路径。
- attention_special CSV 行构建 `seq/avg_seq_len`、`seq+batch`、`seq+heads/head_dim`、`seq+batch+heads/head_dim` bucket；当生产 CSV 字段不足时，对应 bucket 为空并在运行时降级。
- dtype / format / layout / sparse mode / quant mode 不一致时不进入同一 bucket。
- 插值 regime key 不包含 exact lookup 未使用的 `Accelerator Core`；transpose/layout 不会导致比 exact 更严格的 bucket 分裂。
- 1D/2D/3D 候选点完整时返回有效 boundary/grid 结构。
- 缺少必要候选点时返回失败原因。

### 11.2 datasource 集成测试

建议新增：

```text
tests/regression/tensor_cast/test_profiling_interpolation_phase1.py
```

覆盖：

1. exact hit 优先返回 `MEASURED`。
2. compute exact miss 但候选点 boundary 完整时返回 `INTERPOLATED`。
3. attention_special exact miss 但候选点 boundary 完整时返回 `INTERPOLATED`。
4. 缺少候选点时不插值，回退 base 行为。
5. `--disable-profiling-interpolation` 后不进入 wrapper。
6. communication op 不进入 wrapper 插值。
7. details / shape_match_info 包含插值维度、轴、算法、候选点和 confidence。
8. attention 的 `axis_transform=sqrt` 同时作用于候选点坐标和 target 坐标。
9. `Runtime actual_seq_lengths_shape/values` 无法可靠表达 batch 时降级，不把单个 seq_len 值误当成 batch。

### 11.3 精度测试

第一阶段不要求完整离线精度平台，但需要 synthetic holdout case：

- compute holdout：CSV 包含 MatMul/GEMM 2D/3D 网格，移除目标点作为查询目标，插值结果与公式一致。
- attention holdout：CSV 至少覆盖 `avg_seq_len` 1D，并构造包含 batch/head_dim 或 heads 的 2D/3D 测试网格，移除目标点作为查询目标，验证结果与 `scipy.interpolate.griddata(..., method="linear")` 直接调用结果一致。即使真实 profiling 数据暂时缺少 batch/head_dim 多取值，也要用测试 CSV 覆盖第一阶段 2D/3D 代码路径。

holdout / 精度损失对比需要生成可见报告，至少包含每个 holdout 样本的目标 axes、被移除的真实 latency、插值 latency、绝对误差、相对误差、source、confidence、method 和候选点数量。报告可以是 markdown 或 JSON artifact，例如 `phase1_profiling_interpolation_holdout_report_zh.md`。

本轮实现期验证摘要：

- repo-tracked 回归覆盖 interpolation math、candidate index、wrapper datasource、CLI help、composite decomposition、op_mapping policy，以及 synthetic M1、M3、M5 指标聚合行为。
- 真实 profiling DB 的 enabled / disabled on/off 验证作为本 PR 的本地验证证据记录在 PR 描述中；完整 nightly/CI 非劣化门禁留后续。
- synthetic holdout 覆盖 compute 和 attention 的 1D/2D/3D 代码路径，验证结果与 `linear_interp` / `scipy.interpolate.griddata(..., method="linear")` 一致；生产默认仍受 kernel policy、regime key、候选点完整性和 no-extrapolation 约束。
- 本地补充跑过 Qwen3-32B prefill/decode、DeepSeek-V3 prefill/decode 四个 profiling 场景，导出 empirical metrics 和 chrome trace 后计算 M6；这些结果只用于说明 profiling model 的命中和覆盖情况，不解释为真实硬件吞吐提升。
- 已知边界包括 MatMul / batch-matmul 数据稀疏与路由问题、AddRmsNormBias / shared-token elementwise-like 覆盖不足，以及 full nightly on/off gate 尚未纳入本 PR；这些作为后续数据或阶段演进事项跟踪。

从用户视角，第一阶段要证明：

- 插值结果来源可见。
- 插值算法可解释。
- 结果可通过开关回退。

## 12. 实施步骤

建议阶段一拆成以下实现步骤：

1. 增加用户配置和 CLI。
2. 在 profiling datasource 构造路径接入 wrapper。
3. 新增 `interpolation_math.py`。
4. 新增候选点索引结构。
5. 实现 compute M/K/N target 提取与 1D/2D/3D 查询。
6. 实现 attention_special `seq/avg_seq_len`、`batch`、`heads/head_dim` target 提取与 1D/2D/3D 查询；当 CSV 字段和候选点不满足条件时降级。
7. 实现 `QueryResult` details / `shape_match_info` 填充。
8. 增加 debug log 和 warning。
9. 增加单元测试和集成测试。

## 13. 迁移与兼容性

### 13.1 默认行为

阶段一默认在 profiling datasource 构造路径中包裹 `InterpolatingDataSource`，但并不让所有现有 `op_mapping.yaml` entry 无条件插值。实际执行仍受以下条件约束：

- base `ProfilingDataSource` exact lookup 优先；`MEASURED`、base 内部 `INTERPOLATED`、未来 base `EXTRAPOLATED` 结果直接返回。
- wrapper fallback 仅覆盖 compute 与 attention_special allowlist。
- 若 entry 没有显式 `interpolation_policy`，使用内置保守默认 policy。
- 若 schema、regime key、候选点结构或 latency 校验不满足要求，稳定降级或回退 base 行为。

现有 entry 不需要立刻修改才能继续运行；新增 policy 字段只用于收窄或显式声明插值策略。

### 13.2 字段新增与 API 兼容

`QueryResult` 语义保持向后兼容：

- `source` 继续用于区分 `MEASURED`、`INTERPOLATED`、`PARTIAL` 等来源。
- `confidence` 是新增/强化的展示字段，默认只用于观测，不参与结果选择。
- `details` 与 `shape_match_info` 可新增插值维度、轴、method、候选点数量、boundary、fallback reason 等键。
- 下游消费端应按字典可选字段读取，不应假设 details key 集合固定。

throughput_optimizer 报告只应展示新增字段；不得因为 confidence 缺失或 details key 缺失改变调度决策。

### 13.3 CSV schema 兼容

阶段一不要求现有 CSV 统一升级版本号，也不要求一次性补齐 batch/head/head_dim 等 enrichment 字段：

- 已有 CSV 能 exact 命中时仍走 exact。
- 缺少高维候选字段时，attention_special 降级到 `seq/avg_seq_len` 1D 或回退。
- 新增 candidate point 字段应作为可选列处理；旧 CSV 缺列不能导致加载失败。
- 若后续阶段引入必需列或 CSV version，应先提供兼容读取和迁移脚本。

### 13.4 灰度与回退

阶段一提供 `--disable-profiling-interpolation` 作为硬回退开关。上线建议分三步：

1. shadow/dry-run：记录 interpolation candidate 和 predicted latency，但继续使用 base result。
2. allowlist enable：仅对 compute / attention_special 中已验证的 kernel 或 model family 返回 wrapper `INTERPOLATED`。
3. default enable：保留 disable switch 与 debug details，持续监控 hit rate、fallback reason 和 holdout error。

若出现行为回归，用户可立即关闭 wrapper；维护者可通过 policy allowlist 或 regime key 收窄影响范围。

## 14. 风险与缓解

| 风险 | 影响 | 缓解 |
|---|---|---|
| 默认开启改变旧行为 | exact miss 后可能返回插值结果而不是原 fallback | 提供 `--disable-profiling-interpolation`；第一阶段只对 compute / attention_special allowlist 生效 |
| 候选点语义不一致 | 可能跨 dtype / layout / transpose / attention regime 错误插值 | `exact_fields/regime_keys` 必须精确匹配；无法证明等价的 shape 不合并 |
| 多维候选点不足 | 2D/3D 缺角点、缺平面或 griddata 点集退化时结果不可靠 | 不强行插值，按固定顺序降级到低维或回退 base 行为，并在 details 记录原因 |

## 15. 未解决问题

第一阶段只保留少量需要评审确认的问题：

1. 默认是否按 RFC 建议开启 wrapper，并通过 `--disable-profiling-interpolation` 回退。
2. attention_special 是否需要在 enrichment 工具中补充显式 `Runtime batch_size`，以提升生产数据中 batch 轴 2D/3D 的触发率。
3. 候选点索引命名是否使用通用 `CandidateIndex`，还是分为 `ComputeIndex` / `AttentionIndex`。
4. 规则网格 bilinear/trilinear 专用函数是否值得在后续阶段补充；第一阶段统一使用 `griddata_linear_interp`。
5. 插值尚未覆盖跨格式 shape canonicalization/transform，例如算子库只有 NZ/FRACTAL_NZ 格式实测点而 runtime 查询是 ND 格式时，能否基于可靠 canonicalization 进行插值复用，需要后续结合 exact lookup 规则和数据验证再决定。

## 16. 第一阶段验收标准

第一阶段从用户视角验收以下能力：

1. compute 多维插值端到端可用：在 `--performance-model profiling` 流程中，MatMul/GEMM 等 compute 类算子 CSV exact miss 且候选点完整时，能够完成 M/K/N 维度上的 1D、2D、3D 插值，并返回 `QuerySource.INTERPOLATED`。
2. attention_special 多维插值端到端可用：直接 `query_mode: attention_special` 的 attention 算子，以及 composite decomposer 产生的 attention sub-kernel fallback，能够完成 `seq/avg_seq_len` 1D、`seq+batch` 或 `seq+heads/head_dim` 2D、`seq+batch+heads/head_dim` 3D 插值代码路径；生产查询在字段可推导、候选点完整且 regime 匹配时返回 `QuerySource.INTERPOLATED`，不满足时稳定降级或回退。
3. 插值结果可回退、可解释：用户可以通过 `--disable-profiling-interpolation` 回到 base `ProfilingDataSource` 行为；候选点不足或 schema 不满足时可预测地回退；回显、summary/report、details 或 debug log 中能看到 `source`、`confidence`、插值轴、算法、候选点和 fallback 原因，且 confidence 只展示、不参与结果选择。
