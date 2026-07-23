# RFC: Profiling 插值 Phase2 算子能力扩展

## 元数据

| 项目 | 内容 |
| --- | --- |
| RFC 名称 | Profiling 插值 Phase2 算子能力扩展 |
| 英文名 | Profiling Interpolation Phase2 Operator Capability Expansion |
| 状态 | Draft |
| 阶段 | Phase 2（共 3 阶段） |
| 目标模块 | `tensor_cast/performance_model/profiling_database` |
| 基线 | Phase1 插值能力及 PR262 合入后的行为 |
| 关联 PR | #262、#366、#388、#389 |
| 前置变更 | [PR #555](https://gitcode.com/Ascend/msmodeling/pull/555)、[PR #593](https://gitcode.com/Ascend/msmodeling/pull/593)、[Issue #272](https://gitcode.com/Ascend/msmodeling/issues/272) 的实现 |
| 更新日期 | 2026-07-22 |

中英文 RFC 描述同一套能力范围和运行语义。修改默认行为、插值轴、regime、回退条件或验收标准时，必须同步修改两份文档。两份文档的章节编号、能力与验收表格、Section 7 测试矩阵是规范性结构对照；翻译导致的行数不同不构成差异，但实现 PR 必须附中英文结构 diff，并列明所有仅措辞不同的有意差异。

## 1. 概述

Phase2 在 Phase1 的 `InterpolatingDataSource` 上扩展算子能力，不建立第二套 datasource，也不新增公共插值底座。

Phase1 已提供 wrapper 入口、全局关闭开关、候选索引、插值数学、latency guard、failure details 和 compute / attention 插值。Phase2 复用这些能力，为目标范围内的非通信算子补充专用 target、regime 和 dispatch；数据或子项不完整的路径保持可诊断 miss，不据此声明已形成有效插值覆盖。

### 1.1 能力范围

| 算子大类 | Phase2 能力范围 | 主要算子或 kernel |
| --- | --- | --- |
| MOE/DFC | 为 DFC 新增专用 target、regime 与插值路径 | `DispatchFFNCombine` |
| Elementwise | 新增 broadcast / shared-token 感知的 elementwise 插值路径 | `Add`、`Mul`、`Div` 及其 AiCore / alternate kernel 变体 |
| Attention | 新增 LightningIndexer 与条件性 SparseFlashAttention 叶子算子插值 | `LightningIndexer`、`SparseFlashAttention` |
| Compute | generic compute mapping 精修、quant-scale 与 MLA cache update | Concat / Gather / Cast / LayerNorm 变体、`DynamicQuant`、`DynamicBlockQuant`、`QuantBatchMatmulV3`、`ScatterNdUpdate` |

本表只列 Phase2 新增或扩展的叶子算子能力：为 DFC 新增 MOE 专用路径，补充 elementwise，并扩展 compute 与 attention 的子类和路径。上游只向 Phase2 传递可独立查询的叶子算子。

后文统一按“算子大类 -> 子类/路径”组织。kernel 变体、插值轴组合和具体 dtype 只在所属子类内说明，不提升为新的算子大类。

不同版本的 profiling 数据和 `op_mapping.yaml` 不必包含完全相同的算子。除 `zero_cost` 和 `accepted_miss` 这类显式计量规则外，只有 mapping、runtime target 和 profiling CSV 同时可用时，数据驱动路径才会产生结果。

### 1.2 目标

Phase2 完成以下工作：

1. 为上述本次新增或调整的子类建立或补齐可达的 profiling 插值路径。
2. 每条路径定义可解释的连续轴和必须精确匹配的 regime。
3. 在数据允许时按低维优先顺序完成 1D、2D 或 3D 插值。
4. 候选不足、语义不一致或数据缺失时稳定回退，不使用不兼容数据凑命中率。
5. 新路径具备与 Phase1 一致的 failure reason、success details、latency guard、测试和报告能力。
6. 复用上游生成的叶子算子调用，不在插值模块内重新推导或汇总子项。

### 1.3 非目标

Phase2 不包含：

- communication 的第二套 wrapper 插值；
- 外推或 `QuerySource.EXTRAPOLATED`；
- 新的公共 CandidateIndex、数学后端或 datasource；
- 父算子拆分、子项汇总或 overlap / pipeline latency 聚合；
- 在没有真实 profiling 数据时声明 FP8、MXFP4 或 DynamicBlockQuant 的插值精度。

## 2. 设计原则与 Phase1 复用边界

Phase2 以 [Phase1 RFC](./rfc_profiling_operator_interpolation_phase1_zh.md) 定义的运行契约为基线。本文只描述 Phase2 的差量。

### 2.1 复用内容

| Phase1 能力 | Phase2 用法 |
| --- | --- |
| `InterpolatingDataSource` | 唯一 wrapper 入口和 Phase2 实现主体；只处理可独立查询的叶子算子，不承担上游拆分或汇总 |
| `CandidateIndex` / `CandidateGroup` | 组织候选点、按 regime 分组、执行插值 |
| `interpolation_math.py` | 复用 1D 线性插值和 2D / 3D 几何检查 |
| `--disable-profiling-interpolation` | 全局回退到 base `ProfilingDataSource` |
| `interpolation_policy.kernel_overrides` | 通过 `max_interpolation_dim` 收紧单个 kernel 的最高维度 |
| failure / success details | 复用来源、方法、轴、边界、候选点和失败原因字段 |

Phase2 不引入新的插值数学。1D 继续使用相邻实测点线性插值，2D / 3D 继续使用凸包内的线性单纯形插值，并沿用 Phase1 的退化检查、边界检查和 latency 合法性检查。该数学口径与 aiconfigurator 历史实现的 1D linear 和 2D / 3D `griddata(method="linear")` 一致，都是线性插值；aiconfigurator 当前 `perf_interp v2` 的 Grid / ScatteredSites 和外推能力不属于 Phase2，也不在本文承诺后续阶段实现。

### 2.2 Phase2 原则

- 最小实现：每类算子只增加完成 target 和候选匹配所需的局部逻辑。
- 奥卡姆剃刀：没有真实 blocker 时不抽象新平台，不设计未来接口。
- exact 优先：除需要专用语义隔离的路径外，完整 base exact 结果保持不变；专用路径也在插值前检查同 regime 精确坐标。对具有合法非标量输出的 Elementwise，wrapper 绕过 base 的跨 dtype byte-ratio 查询，按完整输出 shape 和 exact profiling dtype 检查精确坐标。
- 低维优先：允许的维度内始终先尝试各子类列出的 1D 轴，再尝试 2D、3D 轴组合；同一维度存在多个组合时，按该子类明确列出的顺序执行。
- 同语义候选：离散 regime 不匹配时不插值。
- 失败可解释：路径不适用或候选不合法时返回 miss，并记录原因。
- 不外推：目标不在有效边界或凸包内时回退。

## 3. Phase2 runtime 差量设计

### 3.1 Dispatch

```text
InterpolatingDataSource.lookup(op)
  |
  |-- receive one leaf op from upstream
  |     -> read mapping and classify exact-match ownership
  |
  |-- base exact eligible
  |     -> base.lookup(op)
  |          |-- complete result -> return base result
  |          `-- PARTIAL / None -> Phase2 fallback dispatch
  |
  |-- specialized exact ownership
  |     -> moe_fused / compute_scale / quantized_matmul / non-scalar elementwise
  |     -> skip semantically incomplete input-shape-only base exact lookup
  |     -> check exact coordinate in specialized CandidateIndex
  |          |-- exact coordinate -> return MEASURED
  |          `-- non-exact / miss -> Phase2 fallback dispatch
  |
  v
Phase2 fallback dispatch
  |-- MOE/DFC 专用路径
  |     `-- query_mode: moe_fused
  |           -> DispatchFFNCombine
  |-- Elementwise
  |     `-- query_mode: elementwise
  |           -> Broadcast / Shared-token
  |-- Attention
  |     |-- query_mode: attention_lightning_indexer
  |     |     -> LightningIndexer
  |     `-- query_mode: attention_sparse_sharedkv
  |           -> SparseFlashAttention
  |-- Compute
  |     |-- ordinary mapping
  |     |     -> Phase1 generic compute
  |     |-- compute_subcategory: compute_scale
  |     |     -> Dynamic quant scale
  |     |-- compute_subcategory: quantized_matmul
  |     |     -> Quantized matrix compute
  |     |-- query_mode: scatter_nd_update_mla
  |     |     -> MLA cache update
  |     `-- unknown compute_subcategory
  |           -> fail closed
  |-- query_mode: attention_special ---> Phase1 attention
  `-- category: communication ---------> base owns this path; wrapper returns miss
```

`moe_fused`、`compute_scale`、`quantized_matmul` 和合法非标量 Elementwise 由专用 CandidateIndex 负责 exact：DFC 的 base mapping 只使用 activation 输入，无法校验完整 GMM 权重 shape；其余三类 base 查询缺少输出签名、scale mode、完整 quant 或 exact dtype 语义。其他上游叶子算子先进入语义完整的 base exact lookup，miss 后再尝试插值。Phase2 mapping 不根据其他调用推导该算子的 shape，也不负责多算子汇总。

### 3.2 前置 PR、Issue 与合入门槛

Phase2 的叶子算子接口依赖以下前置变更：

| 前置项 | 提供的能力 | Phase2 依赖点 |
| --- | --- | --- |
| [PR #555](https://gitcode.com/Ascend/msmodeling/pull/555) | DSA CP 前端与运行时布局基础 | 提供 DeepSeek V4 的 `quant_lightning_indexer`、`sparse_attn_sharedkv`、`scatter_nd_update_mla` 语义入口，并按真实 TP/SP、prefill/decode 语义形成 workload。 |
| [PR #593](https://gitcode.com/Ascend/msmodeling/pull/593) | GLM5 DSA CP 的 profiling 语义与 exact-match 基础 | 完善 GLM5 `mla_sparse_attention` 的 shape、dtype、phase、cache 与 SparseFlashAttention 子项语义；当前父算子路径不等同于 DeepSeek V4 的 `sparse_attn_sharedkv` 叶子入口。 |
| [Issue #272](https://gitcode.com/Ascend/msmodeling/issues/272) | 统一上游拆分与叶子算子查询契约 | 定义跨 DeepSeek V4 / GLM5 可消费的 canonical leaf descriptor 或明确的版本化叶子入口；exact lookup 与 interpolation 必须使用同一份 shape、dtype、phase、topk 和请求/序列语义，多算子 latency 在插值模块外汇总。 |

Issue #272 可以由 PR #593 的后续提交或独立配套 PR 完成，但其验收项必须先进入 Phase2 的目标基线。仅有 PR #555 和当前 PR #593、但 #272 尚未完成时，依赖上游拆分的叶子路径不能作为 Phase2 已完成能力合入。

PR #388 和 PR #389 合入前必须完成以下集成步骤：

1. rebase 到包含 PR #555、PR #593 和 Issue #272 实现的目标分支；
2. `profiling_data_source.py` 相对该目标分支保持零 diff；
3. 删除 `InterpolatingDataSource` 中的父算子拆分、子项描述依赖和汇总逻辑；
4. 保留叶子算子的 target、candidate index、插值和诊断；
5. 用集成测试证明同一个叶子算子先 exact lookup，miss 后再进入插值，且上游只汇总一次 latency。

前置契约未满足的路径必须保持在 Phase2 范围外，不能通过 wrapper-local 拆分或修改 `ProfilingDataSource` 临时补齐。

### 3.3 精确坐标与结果来源

base exact lookup 未被跳过时，base 完整结果优先返回。DFC base mapping 只匹配 activation，Elementwise base 查询会按 dtype byte ratio 缩放 latency，`compute_scale` / `quantized_matmul` base 查询缺少输出签名、scale mode 或完整 quant 语义。因此这四类路径由 wrapper 在专用索引内完成精确匹配和插值。

进入 Phase2 fallback 后，DFC、专用 attention、cache-update、Elementwise、`compute_scale` 和 `quantized_matmul` 路径都在 CandidateIndex 中检查精确坐标：

- target 的所有连续轴与同 regime 实测点相等时，返回 `QuerySource.MEASURED`；
- 非精确坐标且插值成功时，返回 `QuerySource.INTERPOLATED`；
- 候选不足、regime 不匹配、latency 非法或越界时返回 miss；
- base 原结果为合法 `PARTIAL` 时，插值失败后沿用 Phase1 的 PARTIAL 保留规则。

### 3.4 Phase2 算子覆盖与维度上限

插值尝试顺序沿用 Phase1 的低维优先规则，本节不再重复其通用定义。下表中的“最高维度”是代码允许尝试的上限，不保证当前 CSV 已形成相应候选边界。

#### 3.4.1 Phase2 本次实现或调整的算子路径

本表只列 Phase2 本次实现、扩展或调整的算子入口；同一算子的所有轴组合写在同一行。仅复用 Phase1 且本次未调整的算子不重复列出。实现不完整的算子按 3.4.2 处理；只有数据或验证缺口的已实现路径仍保留在本表，并在对应能力章节说明边界。

MOE/DFC 大类本次只新增 `DispatchFFNCombine` 专用插值路径。

表中坐标与语义字段含义如下。字段是否作为连续轴参与插值，或仅作为 regime 精确匹配，由对应子类定义：

- `axis_0`：去掉可选的首维 `batch=1` 后，shape 的第一维；使用该轴时其余维度必须保持一致；
- `axis_1`：仅用于 rank-2 elementwise 输出，表示输出 shape 的第二维；
- `output_numel`：输出 shape 各维的乘积；
- `tokens`：将语义上的 token 维展平后的 token 行数，具体取法由对应子类说明；
- `cache_blocks`：KV cache shape 的第一维，即 cache block 数；
- `topk`：每个 token 选择的 expert 或 index 数；
- `M / K / N`：矩阵语义中的行数、归约维和输出列数；非矩阵 DynamicQuant 只使用展平后的行数 `M` 与最后一维 `K`。

| 算子大类 | 子类/路径 | 算子入口 | 适用版本 | 轴组合 | 最高维度 | 依据 |
| --- | --- | --- | --- | --- | ---: | --- |
| MOE/DFC | DispatchFFNCombine | `tensor_cast.dispatch_ffn_combine.default` | v0.15、v0.18 | `tokens` | 1D | target 路径可达；v0.15 缺少 EP 字段，v0.18 权重为 INT8，均不能证明普通未量化入口的数据等价性，当前稳定 miss。 |
| MOE/DFC | DispatchFFNCombine | `tensor_cast.dispatch_ffn_combine_quant.default` | v0.15、v0.18 | `tokens` | 1D | v0.18 的 BF16 activation + INT8 weight 签名支持 W8A8；v0.15 缺少 EP 字段，稳定 miss。完整 GMM1/GMM2 权重 shape、hidden、topk、EP、dtype、layout 和 subtype 属于 regime。 |
| MOE/DFC | DispatchFFNCombine | `tensor_cast.dispatch_ffn_combine_quant_int4.default` | v0.15、v0.18 | `tokens` | 1D | target 路径可达；当前没有同语义 INT4 实测行，稳定 miss。 |
| MOE/DFC | DispatchFFNCombine | `tensor_cast.dispatch_ffn_combine_fp8.default` | v0.15、v0.18 | `tokens` | 1D | target 路径可达；当前没有同语义 FP8 实测行，稳定 miss。 |
| MOE/DFC | DispatchFFNCombine | `tensor_cast.dispatch_ffn_combine_mxfp4.default` | v0.15、v0.18 | `tokens` | 1D | target 路径可达；当前没有同语义 MXFP4 实测行，稳定 miss。 |
| Elementwise | Broadcast / Shared-token | `aten.add.Tensor` | v0.13、v0.15、v0.18 | `axis_0`；rank-2 时增加 `axis_1`、`axis_0 + axis_1` | 2D | 直接使用输出 shape 维度；broadcast signature、rank、dtype 和 kernel 属于 regime。 |
| Elementwise | Broadcast / Shared-token | `aten.mul.Tensor` | v0.13、v0.15、v0.18 | v0.13：`axis_0`；v0.15/v0.18：同 `aten.add.Tensor` | v0.13：1D；v0.15/v0.18：2D | v0.13 复用 generic compute，后续版本进入 broadcast / shared-token 路径。 |
| Elementwise | Broadcast / Shared-token | `aten.div.Tensor` | v0.13、v0.15、v0.18 | v0.13：`axis_0`；v0.15：同 `aten.add.Tensor`；v0.18：不插值 | v0.13：1D；v0.15：2D；v0.18：zero-cost | v0.13 复用 generic compute，v0.15 进入 elementwise 路径，v0.18 由 mapping 显式不计量。 |
| Attention | LightningIndexer | `tensor_cast.quant_lightning_indexer.default` | v0.18 | `tokens`；`cache_blocks`；`tokens + cache_blocks` | 2D | `topk` 和请求/序列元数据分组属于 regime；真实 CSV 的 `topk` 当前只有一个值。 |
| Attention | SparseFlashAttention | `tensor_cast.sparse_attn_sharedkv.default`（DeepSeek V4）；GLM5 入口由 Issue #272 统一 | v0.18 | `tokens` | 1D（条件满足后） | v0.18 已有实测 CSV，但相同 tokens / cache / topk 下仍有不同请求/序列元数据；只有 canonical leaf 能提供同义分组字段时才启用，否则稳定 miss。 |
| Compute | Concat | `aten.cat.default` | v0.13、v0.18；v0.15 为 zero-cost | `axis_0` | 1D | 有 `ConcatD.csv` 的版本走 generic compute。 |
| Compute | Concat | `tensor_cast.cat.default` | v0.13、v0.18；v0.15 为 zero-cost | `axis_0` | 1D | 与 `aten.cat.default` 共用 kernel policy。 |
| Compute | Gather | `aten.embedding.default` | v0.13、v0.15、v0.18 | `output_numel` | 1D | Gather policy 使用输出元素数，alternate kernel 继承该轴。 |
| Compute | Cast | `aten.to.dtype` | v0.13、v0.15、v0.18 | `axis_0` | 1D | Cast / TensorMove 变体走 generic compute。 |
| Compute | LayerNorm | `aten.native_layer_norm.default` | v0.18 | `axis_0` | 1D | `tc_input_count: 3` 排除非 tensor 元数据。 |
| Compute | Dynamic quant | `tensor_cast.dynamic_quantize_symmetric.default` | v0.13、v0.15、v0.18 | `M`；`K`；`M + K` | 2D | 输出签名、scale mode、dtype 和 format 属于 regime。 |
| Compute | Dynamic quant | `tensor_cast.dynamic_quantize_asymmetric.default` | v0.13、v0.15、v0.18 | `M`；`K`；`M + K` | 2D | asymmetric offset 输出与 symmetric 候选隔离。 |
| Compute | Dynamic block quant | `tensor_cast.dynamic_quantize_mxfp4.default` | v0.13、v0.15、v0.18 | `M`；`K`；`M + K` | 2D | 路径已实现，但当前无 `DynamicBlockQuant.csv`。 |
| Compute | Quantized matrix | `tensor_cast.static_quant_linear.default` | v0.13、v0.15、v0.18 | `M`；`K`；`N`；三种 2D 组合；`M + K + N` | 3D | 复用 MatMul 轴，INT8 dtype / layout / 输出签名精确隔离。 |
| Compute | Quantized matrix | `tensor_cast.static_quant_linear_int4.default` | v0.13、v0.15、v0.18 | 同 `static_quant_linear` | 3D | INT4 packed-weight 语义与其他 dtype 隔离。 |
| Compute | Quantized matrix | `tensor_cast.fp8_linear.default` | v0.13、v0.15、v0.18 | 同 `static_quant_linear` | 3D | 路径已实现，但当前无同 dtype FP8 数据。 |
| Compute | Quantized matrix | `tensor_cast.mxfp4_linear.default` | v0.13、v0.15、v0.18 | 同 `static_quant_linear` | 3D | 路径已实现，但当前无同 dtype MXFP4 数据。 |
| Compute | MLA cache update | `tensor_cast.scatter_nd_update_mla.default` | v0.18 | `tokens` | 1D | cache capacity 与 update tokens 未形成独立二维网格。 |

#### 3.4.2 当前未覆盖或明确不插值的算子

未覆盖项按实现状态归并，只列没有完整插值路径或明确不进入插值的算子。已经实现路径但缺少 CSV、dtype 数据或精度验证的算子不列入本表，其数据与验证边界在对应能力章节和第 8 章说明。

| 类型 | 包含算子 | 版本 | 未覆盖原因与当前行为 |
| --- | --- | --- | --- |
| 超出 Phase2 范围 | `tensor_cast.matmul_all_reduce.default`、`tensor_cast.static_quant_linear_all_reduce.default`、`tensor_cast.static_quant_linear_int4_all_reduce.default`、`tensor_cast.fp8_linear_all_reduce.default`、`tensor_cast.mxfp4_linear_all_reduce.default` | 全版本 | 上述算子均包含 communication 子项；communication 类算子不由本插值模块实现插值。 |
| 融合吸收，明确不插值 | `tensor_cast.concat_and_cache_mla.default` | v0.18 | MLA cache write 已被 `KvRmsNormRopeCache` 融合并由其计量。单独插值会重复计算 latency，因此 mapping 返回带说明的 0 latency。只有未来版本重新出现独立 kernel 时才需要调整。 |
| 条件性 accepted miss | `aten.index.Tensor` | v0.18 | 目标 workload 中部分调用是 TC CachingRotaryEmb 产生的 RoPE cache artifact，没有对应的独立 NPU kernel，当前 mapping 返回 0 latency。但其他场景可能存在真实 `Index` kernel；在能够区分调用语义前，无条件 accepted miss 存在漏算风险。 |

`zero_cost: true` 算子、普通 communication 和内部 `profiling.*` kernel 不列入未覆盖表：前两类已有明确计量所有权，内部 kernel 只有在真实顶层 TC trace 中出现并影响 B2B / holdout 时才构成覆盖缺口。

## 4. 能力设计

### 4.1 MOE/DFC

#### 4.1.1 DispatchFFNCombine

`DispatchFFNCombine` 专用路径使用 `query_mode: moe_fused`。当前 mapping 识别普通、W8A8、INT4、FP8 和 MXFP4 语义入口；入口可达不等于存在可复用的同语义实测数据。

##### Target 与连续轴

设首个 activation shape 为 `[d0, d1, ..., dn, hidden]`，`tokens = d0 * d1 * ... * dn`，表示当前 activation 的总 token 数。例如 `[2, 4, 72, 7168]` 得到 `tokens=576`。

真实 DFC profiling CSV 将 activation 记录为展平后的 `[tokens, hidden]`，不能恢复各前导维的分组信息。因此当前只使用数据库直接支持的 `tokens` 作为连续插值轴，不从运行时 shape 构造额外轴。该口径也与 aiconfigurator 的 MOE 查询一致：先按模型和并行配置分桶，再在桶内沿 `num_tokens` 做 1D 插值。

当前轴组合为：

```text
1D: tokens
```

`topk`、`hidden` 和 EP size 是 exact regime，不作为连续插值轴。`local_experts` 在 target 和候选都可提取时必须保持一致。

##### Regime 与失败边界

候选至少按以下字段隔离：

- kernel type；
- activation dtype / format；
- quant subtype、weight dtype / layout；
- 完整 GMM1 / GMM2 weight shape；
- hidden size；
- topk；
- EP size。

target 根据 TensorCast 语义入口生成 `plain / w8a8 / int4 / fp8 / mxfp4` subtype，并从对应算子签名提取 GMM1/GMM2 权重列表的完整 shape；候选从 CSV 的第二、第三个输入提取相同字段。任一权重 shape 缺失、subtype 无法识别或两侧不一致时拒绝候选，不能只因 activation dtype 相同而共享 latency。

当前 `DispatchFFNCombine.csv` 只有 BF16 activation + INT8 weight 签名，只能支持与该签名对应的 W8A8 路径。v0.15 CSV 还缺少 `EP Size` 字段，不能构造完整 EP regime，因此当前有效数据覆盖仅为 v0.18 W8A8。普通、INT4、FP8 和 MXFP4 target，以及缺少 EP 字段的 v0.15 target，在补充同语义字段和实测数据前稳定 miss，不使用 W8A8 行凑命中率。

EP size 是必需 regime。CSV 必须存在 `EP Size` 列，且每一行都必须提供合法正整数；缺列、空值或非法值都不能匹配任意目标 EP。runtime 未配置 EP 时，整条 MOE/DFC 路径返回 `ep_size_not_configured`。

无法提取 topk、候选 EP 不匹配、候选点不能形成 boundary 或 latency 非法时返回 miss，不落入 ordinary compute 插值。

### 4.2 Elementwise

#### 4.2.1 Broadcast / Shared-token

Phase2 的 broadcast / shared-token 专用路径使用 `query_mode: elementwise`。v0.13 的 `aten.mul.Tensor` 和 `aten.div.Tensor` 仍走 generic compute；其他适用版本按真实 kernel 配置 `Add`、`Mul`、`Div` 及 `AddAiCore`、`MulAiCore`、`RealDiv` 等 alternate kernel。

##### Target 与连续轴

连续轴直接取输出 shape 维度：

- `axis_0`：输出 shape 的第一维；首维为 `batch=1` 时取下一维；
- `axis_1`：仅在去除可选 `batch=1` 后输出为 rank 2 时使用，表示第二维。

轴组合按以下顺序尝试：

```text
1D: axis_0
1D: axis_1（axis_0 固定）
2D: axis_0 + axis_1
```

`axis_0` 的 1D fallback 要求输出 shape tail 一致；`axis_1` 的 1D fallback 要求 `axis_0` 固定。仅 rank-2 输出继续尝试 `axis_0 + axis_1` 2D，且候选必须形成非退化边界；rank 大于 2 的输出只使用固定完整 tail 的 `axis_0` 1D。这里不再使用 `output_numel + axis_0`，避免在线性插值前引入 `output_numel = axis_0 * axis_1` 的非线性坐标变换。

##### Regime、dtype 与失败边界

候选按以下字段隔离：

- kernel type；
- output rank；
- input count；
- broadcast pattern；
- target output dtype。

候选 dtype 必须与目标 dtype 精确一致。Phase2 不按 dtype 字节宽度缩放实测 latency；缺少同 dtype 数据时返回带原因的 miss。一次插值也不能混用 selected latency column 与 fallback latency column。

alternate kernel 按 mapping 顺序尝试。所有 kernel 都失败后，统一记录 attempted kernel 和每次 miss details。

### 4.3 Attention

#### 4.3.1 LightningIndexer

`tensor_cast.quant_lightning_indexer.default` 使用：

```yaml
kernel_type: LightningIndexer
query_mode: attention_lightning_indexer
```

##### Target 与连续轴

连续轴定义如下：

- `tokens`：query 实际包含的 token 行数；3D query `[T, H, D]` 取 `T`，4D query `[B, S, H, D]` 取 `B * S`；
- `cache_blocks`：cache tensor 的第一维；
- `topk`：输出 shape 的最后一维，仅用于精确匹配 regime，不作为连续轴。

轴组合按以下顺序尝试：

```text
1D: tokens
1D: cache_blocks（tokens 固定）
2D: tokens + cache_blocks
```

quant 路径的 cache tensor 位置与普通路径不同，target builder 按真实算子签名选择 cache input。真实 `LightningIndexer.csv` 中 `topk` 当前固定为 `2048`，因此不声明以 `topk` 为第三轴的 3D 能力；该选择也与 aiconfigurator 将 `index_topk` 作为配置条件而非连续插值轴的口径一致。

真实 CSV 在相同 `tokens / cache_blocks / topk` 坐标下仍存在不同的请求或序列元数据 shape，且 latency 差异明显。实现必须从 target 和 CSV 两侧提取同义的请求/序列分组字段并加入 regime；无法可靠对齐时 fail closed，不能把这些行合并为同一坐标。

##### Regime 与失败边界

regime 包含 kernel type、query profiling dtype、cache profiling dtype、cache shape 除首维外的 tail、前两个输入 format、query rank、heads、head_dim、`topk` 和请求/序列分组。只有 cache shape 首维 `cache_blocks` 是连续轴；cache dtype 和 tail 必须精确匹配。CSV 和 mapping 都提供 API path 时还必须匹配 API path；CSV 没有该列时，使用其余 regime 字段，不虚构必填列。任何必要语义字段无法从 target 与 CSV 两侧同义提取时，该路径 fail closed。

同 regime 精确坐标先返回 `MEASURED`，其余目标最多尝试 2D。`topk`、请求/序列分组、heads、head_dim、query/cache dtype、cache tail、format 和 API path 必须匹配。成功和失败 details 记录 kernel、API path、target axes 和 rejected rows。

#### 4.3.2 SparseFlashAttention

`SparseFlashAttention` 是 profiling database 中已有实测数据的稀疏 Attention kernel。当前直接 TensorCast 语义入口 `tensor_cast.sparse_attn_sharedkv.default` 由 PR #555 的 DeepSeek V4 前端提供，并通过 `query_mode: attention_sparse_sharedkv` 映射到该 kernel。PR #593 的 GLM5 路径当前从 `mla_sparse_attention` 父算子产生 SparseFlashAttention 子项，不等同于该直接入口；Issue #272 必须先统一两条路径的 canonical leaf contract。

Phase2 不新增稀疏 Attention 算子语义，也不拆分 `mla_sparse_attention`，只对上游传入的独立叶子算子执行精确查询和插值。canonical leaf 至少要携带 query/cache shape、dtype、phase、selected topk 和可与 CSV 对齐的请求/序列分组字段。当前仅 v0.18 提供 `SparseFlashAttention.csv`；v0.13 和 v0.15 缺少实测数据，不具备插值条件。

##### Target 与连续轴

满足后述语义对齐条件后，连续轴只保留 `tokens`：与 LightningIndexer 使用相同定义；3D query `[T, H, D]` 取 `T`，4D query `[B, S, H, D]` 取 `B * S`。

轴组合按以下顺序尝试：

```text
1D: tokens
```

##### Regime 与失败边界

算子语义表明实际访问量由 `topk_indices.shape[-1]` 决定，而 `cache_blocks` 只是 cache 容量，不等于有效 `past_kv`。aiconfigurator 的 sparse attention 使用实际 `past_kv / isl / batch`，不使用 cache 容量替代有效长度。当前 CSV 无法稳定恢复同义的有效 KV 长度，因此 Phase2 不沿 `cache_blocks` 插值。

v0.18 CSV 在相同 `tokens`、query/cache dtype、heads、head_dim、完整 cache shape 和 selected topk 下仍存在不同请求/序列元数据 shape，latency 差异明显。这些行不能作为同一坐标取中位数。canonical leaf 和 CSV 必须能够提取同义的 sparse phase 与请求/序列分组字段，并将其加入 regime；字段语义尚未定义、任一侧无法提取或出现歧义时整条路径稳定 miss。

完整 regime 至少包含 query profiling dtype、cache profiling dtype、前两个输入 format、heads、head_dim、完整 cache shape、selected topk、sparse phase 和请求/序列分组。SparseFlashAttention 候选不与 LightningIndexer 或 Phase1 FusedInferAttentionScore 候选混用。

两个 attention helper 都接收上游传入的独立叶子算子。关闭 wrapper 后，由前置变更提供的专用 base 路径执行 exact lookup，不回落到 generic compute 伪造命中。

### 4.4 Compute

#### 4.4.1 Generic compute mapping 精修

这类改动不增加 `query_mode`、target builder 或插值数学，继续使用 Phase1 generic compute 路径。Phase2 只在有明确算子语义和版本数据依据时补充 mapping：

##### Mapping 范围

- `aten.cat.default`、`tensor_cast.cat.default` 在本地存在 `ConcatD.csv` 的版本映射到 `ConcatD`，不再无条件标记 `zero_cost`；
- embedding 在对应版本为 `GatherV2` 补充 `GatherV3`、`GatherV2AiCore` alternate kernel；
- cast 在对应版本补充 `TensorMove`、`CastAiCore`、`TensorMoveAiCore` alternate kernel；
- `aten.native_layer_norm.default` 映射到 `LayerNormV3`，以 `LayerNormV3WithImplMode` 为 alternate，并通过 `tc_input_count: 3` 排除 normalized shape、epsilon 等非 tensor 元数据。

##### 失败边界

主 kernel 无结果时按 mapping 顺序尝试 alternate。版本没有相应 CSV 或候选时稳定 miss，不跨版本借用数据。

#### 4.4.2 Quant-scale 与 quantized matmul

Quant-scale 不增加 `query_mode`，通过 `compute_subcategory` 选择专用 compute 语义。

##### Target 与连续轴

`compute_scale` 从 quant 输入 shape `[d0, ..., dn, K]` 构造 `M = d0 * ... * dn` 和 `K`。`quantized_matmul` 复用 Phase1 MatMul target：输入矩阵按 `M x K` 与 `K x N` 解释，得到 `M`、`K`、`N`。

| `compute_subcategory` | Kernel | 轴 | 最高维度 |
| --- | --- | --- | ---: |
| `compute_scale` | `DynamicQuant`、`DynamicBlockQuant` | `M`、`K`、`M + K` | 2D |
| `quantized_matmul` | `QuantBatchMatmulV3` | `M`、`K`、`N` 及其组合 | 3D |

##### Dynamic quant

连续轴定义如下：

- `M`：输入 shape 除最后一维外，其余维度的乘积；
- `K`：输入 shape 的最后一维。

scale 输出 shape 用于判定：

- `per_tensor`；
- `per_token`；
- `per_channel`；
- `per_block`，仅用于 `DynamicBlockQuant`。

regime 包含 input dtype / format、输出个数、所有输出 dtype / format、所有辅助输出的 scale mode 和 block size。对称两输出行不能匹配非对称三输出 target。

base compute exact 只检查输入 shape，不能区分输出签名和 scale mode。因此 `compute_scale` 绕过 base exact，在专用索引内执行同 regime 精确坐标检查。

DynamicQuant 的 FP16 数据按 CSV 的 `DT_FLOAT16` 匹配，不改变 Phase1 其他 compute 路径的 dtype 兼容规则。

##### Quantized matrix compute

`quantized_matmul` 直接复用 Phase1 MatMul 的 M/K/N target 和插值数学。`tc_input_count: 2` 用于将 TensorCast `(x, weight, scale, ...)` 对齐到 `QuantBatchMatmulV3.csv` 的两个矩阵输入。Phase1 的 `scale_matrix` 表示静态量化应用 scale 的 M/K 2D 开销，语义不同，本次不复用该 selector。

static quant、INT4、FP8 和 MXFP4 linear mapping 可以指向同一个 `QuantBatchMatmulV3` kernel，但候选仍按真实输入 dtype 和 layout 隔离。INT8 实测行不能用于 FP8 或 MXFP4 target。

`tensor_cast.quantize.default` 是应用已有 scale 的 `AscendQuantV2` 路径，继续使用 generic compute，不进入 `compute_scale`。

##### 数据与失败边界

v0.13、v0.15 和 v0.18 数据库包含 `DynamicQuant.csv` 与 `QuantBatchMatmulV3.csv`。DynamicQuant 有 BF16 实测行，v0.18 还包含 FP16 行；QuantBatchMatmulV3 当前矩阵输入是 INT8。

仓库当前没有 `DynamicBlockQuant.csv`，也没有可证明等价的 FP8 / MXFP4 QuantBatchMatmulV3 实测行。因此：

- DynamicBlockQuant 可以被 mapping 和 target builder 识别，但没有 CSV 时稳定 miss；
- FP8 / MXFP4 linear 不复用 INT8 数据；
- 只有补充真实 profiling 数据并完成 holdout / B2B 后，才能声明这些 dtype 的插值覆盖和精度。

#### 4.4.3 MLA cache update

`tensor_cast.scatter_nd_update_mla.default` 属于 compute 下的 cache-update 子场景：

```yaml
kernel_type: ScatterNdUpdate
alternate_kernel_types: [ScatterNdUpdateAiCore]
query_mode: scatter_nd_update_mla
```

##### 输入规范化

TensorCast 参数顺序是 `(update, cache, index)`，ScatterNdUpdate CSV 使用 `(cache, index, update)`。target builder 先规整顺序，再匹配候选。

##### Target 与连续轴

连续轴定义如下：

- `tokens`：update shape 不高于 2D 时取第一维；高于 2D 时取除最后一维外其余维度的乘积。

最后一维保持为 update tail。当前只在 `tokens` 上做 1D 插值。

##### Regime 与失败边界

regime 包含 cache / index / update dtype、完整 cache shape、update tail 和三个输入 format。cache 第一维表示容量，不作为当前连续轴；完整 cache shape 不一致时不得共享候选。主 kernel miss 后再尝试 `ScatterNdUpdateAiCore`。

该路径接收独立的 cache-update 叶子算子。base exact miss 后，专用索引的同 regime 精确坐标返回 `MEASURED`，非精确坐标才进入 1D 插值。

## 5. Mapping 与最小 policy

Phase2 使用现有 `op_mapping.yaml` 字段，不增加独立配置文件。

| 字段 | 用途 |
| --- | --- |
| `kernel_type` | 主 profiling CSV / kernel |
| `alternate_kernel_types` | 主 kernel 无结果时按顺序尝试的变体 |
| `query_mode` | 选择 DFC、Elementwise、Attention 或 Compute/cache-update 专用路径 |
| `compute_subcategory` | 选择 `compute_scale` 或 `quantized_matmul` |
| `tc_input_count` | 只使用前 N 个 TensorCast 输入与 CSV 对齐 |
| `interpolation_policy.kernel_overrides.<kernel>.max_interpolation_dim` | 收紧 kernel 的最高插值维度 |

| 算子大类 | 子类/路径 | Mapping selector |
| --- | --- | --- |
| MOE/DFC | DispatchFFNCombine | `query_mode: moe_fused` |
| Elementwise | Broadcast / Shared-token | `query_mode: elementwise` |
| Attention | LightningIndexer | `query_mode: attention_lightning_indexer` |
| Attention | SparseFlashAttention | `query_mode: attention_sparse_sharedkv` |
| Compute | Generic compute mapping 精修 | `kernel_type`、`alternate_kernel_types`、`tc_input_count` |
| Compute | Quant-scale / Dynamic quant | `compute_subcategory: compute_scale` |
| Compute | Quant-scale / Quantized matrix compute | `compute_subcategory: quantized_matmul`、`tc_input_count: 2` |
| Compute | MLA cache update | `query_mode: scatter_nd_update_mla` |

`attention_special` 和未声明 Phase2 selector 的 ordinary compute 继续使用 Phase1 路径。

未知 `compute_subcategory` 返回 `compute_subcategory_unknown`。base-owned 叶子先使用前置变更提供的 exact lookup，miss 后再由 wrapper 进入专用路径；`moe_fused`、`compute_scale`、`quantized_matmul` 和合法非标量 Elementwise 则由专用 CandidateIndex 同时负责 exact 与 interpolation。Phase2 不增加第二个全局开关。

## 6. 结果与诊断增量

Phase2 沿用 Phase1 的 QueryResult、latency guard 和候选几何检查，只增加能力定位信息。

### 6.1 成功结果

叶子算子插值成功至少记录：

- `kernel_type`、`query_mode` 或 `compute_subcategory`；
- `method`、`interpolation_dim`、`axes`；
- `boundary`、candidate count、matched rows；
- `fallback_from` 和 `interpolation_path`；
- 能力特有字段，例如 EP、scale mode 或 attention subcategory。

专用索引精确坐标返回 `MEASURED`，details 中记录 `method=exact_coordinate` 和匹配行。

### 6.2 失败结果

失败不能只返回 `None`。至少应区分：

- mapping 或 kernel 缺失；
- target / required field 无法提取；
- CSV 不存在；
- regime 不匹配；
- candidate 不足或几何退化；
- latency 非法；
- target 越界；
- `max_interpolation_dim` 阻止所需维度；
- 上游叶子算子缺少必需语义字段。

miss details 记录 target axes、target regime、attempted kernel、rejected row 统计和最后一次几何诊断。上层仍可按 Phase1 规则进入 PARTIAL 保留或 analytic fallback。

### 6.3 Latency source

候选 latency 必须是正数且 finite。Elementwise、generic compute 和 quant-scale 等使用 source-pure 分组的路径，不在一次插值中混合 selected latency column 与 fallback latency column。

0 latency 只由已有 `zero_cost` 或有明确计量依据的 `accepted_miss` mapping 返回，不能作为普通插值候选。

## 7. 测试与验收

### 7.1 测试范围

Phase2 回归测试集中在：

```text
tests/regression/tensor_cast/test_specialized_operator_interpolation.py
```

同时必须运行：

```text
tests/regression/tensor_cast/test_profiling_interpolation_phase1.py
tests/regression/tensor_cast/test_interpolating_data_source.py
tests/benchmark/ops/perf_database/test_op_mapping_schema.py
```

### 7.2 能力测试矩阵

| 算子大类 | 子类/路径 | 必测行为 |
| --- | --- | --- |
| MOE/DFC | DispatchFFNCombine | v0.18 W8A8 1D tokens 插值、完整 GMM1/GMM2 weight shape 隔离、真实 CSV 重复坐标防误合并、范围与候选不足回退、topk / EP / local experts 隔离、EP 缺列/空值/非法值拒绝、plain / W8A8 / INT4 / FP8 / MXFP4 subtype 严格隔离、无同 subtype 数据时稳定 miss |
| Elementwise | Broadcast / Shared-token | 直接 shape 轴 2D、1D fallback、max dim、v0.13 generic compute 兼容、完整输出 shape 精确坐标、broadcast signature、alternate kernel、跨 dtype 拒绝、latency source purity |
| Attention | LightningIndexer | guarded 2D、query/cache dtype 与 cache tail 隔离、topk 与请求/序列分组隔离、重复坐标拒绝合并、max dim、真实 quant mapping、API path mismatch、missing CSV details、base generic exact 拒绝 |
| Attention | SparseFlashAttention | canonical leaf contract、tokens 1D、query/cache dtype、cache shape / selected-topk / sparse phase / 请求与序列分组隔离、重复坐标不跨语义取中位数、语义不可提取时 fail closed、专用 exact |
| Compute | Generic compute mapping 精修 | ConcatD 数据存在性、Gather / Cast alternate 顺序、LayerNorm metadata 排除与 `tc_input_count` |
| Compute | Quant-scale / Dynamic quant | DynamicQuant 2D、输出签名、scale mode、FP16、专用 exact、max dim、DynamicBlockQuant mode |
| Compute | Quant-scale / Quantized matrix compute | QuantBatchMatmul 3D、`tc_input_count`、FP8 / MXFP4 不复用 INT8 |
| Compute | MLA cache update | 参数重排、完整 cache shape 隔离、真实 CSV 重复坐标防误合并、1D 插值、alternate kernel、专用 exact |

每个子类或路径还要覆盖适用的 target 构造失败、candidate 不足、regime 不匹配、invalid latency、越界、成功 details 和失败 details。

### 7.3 验收标准

Phase2 合入前满足：

1. 声明 Phase2 selector 且列为已实现的叶子算子 mapping 可以到达对应 target builder；数据缺口稳定 miss，上游叶子序列未确认的路径按 3.4.2 处理。
2. 同 regime 精确坐标和插值坐标返回正确 source。
3. `max_interpolation_dim` 能阻止超出上限的路径，不改变低维优先顺序。
4. 专用叶子算子使用正确的 base exact lookup，不通过 generic compute 误命中。
5. 不跨不兼容的 EP、topk、完整 GMM weight shape、shape tail、完整 cache shape、请求/序列分组、API path、dtype、quant subtype 或输出签名混用候选。
6. DFC 只有与 CSV activation / weight dtype、layout、完整 GMM1/GMM2 weight shape 和 EP 严格对应的 subtype 可以使用候选；当前只有 v0.18 W8A8 具备有效数据，v0.15 及 plain、INT4、FP8、MXFP4 路径稳定 miss。
7. SparseFlashAttention 的 canonical leaf、sparse phase 和请求/序列分组在 target 与 CSV 两侧同义可提取；条件未满足时不得启用该插值路径。
8. 前置集成测试证明上游传入的是可独立查询的叶子算子，Phase2 不执行拆分或多算子汇总。
9. Phase1 compute、attention、PARTIAL、disable switch 和 latency guard 回归测试通过。
10. 中英文 RFC 与 mapping、代码和测试保持一致。

UT 证明路径和边界条件正确。涉及插值精度或 runtime 命中率的结论还需要 holdout、endpoint B2B 或真实 profiling ground truth；没有这些证据时，只能说明代码路径可用，不能说明精度或性能收益。

最小报告包含测试版本、数据目录、启用配置、source 分布、插值轴 / 维度、误差统计、failure reason 分布和未验证范围。具体报告格式由测试报告维护，不在 RFC 内固定模板。

## 8. 兼容性、回退与已知限制

### 8.1 兼容性

- Phase2 不修改 QueryResult 公共类型和 datasource 构造方式。
- Phase2 相对包含前置变更的目标分支不修改 `ProfilingDataSource`。
- 原有 mapping 未声明 Phase2 字段时保持 Phase1 / base 行为。
- 各版本 mapping 按该版本的算子语义维护，不要求强制一致；mapping 已接入但 CSV 缺失的入口稳定 miss，完整子项缺失的入口按 3.4.2 处理，不能伪造覆盖。
- 缺少可选 CSV 字段时，target builder 要么使用已定义的兼容路径，要么稳定 miss，不能导致数据库整体加载失败。
- communication 继续由 base exact / alpha-beta 路径负责。

### 8.2 回退

用户可以通过 `--disable-profiling-interpolation` 关闭整个 wrapper。维护者可以通过移除专用 mapping、收紧 `max_interpolation_dim` 或补充更严格 regime 缩小单条路径的影响范围。

回退不会删除 profiling 数据，也不改变 analytic performance model。

### 8.3 已知限制

3.4.1 列出本次已实现或调整的路径，3.4.2 只列实现不完整或明确不插值的算子。CSV、dtype 和精度验证等数据边界在对应能力章节说明。实现上限不等于当前数据库具备对应候选密度；路径没有在 endpoint 产生 `INTERPOLATED` 结果时，只能说明代码路径和失败边界可用，不能声明该路径已在真实 workload 生效或具备可接受精度。

### 8.4 风险控制

| 适用范围 | 风险 | 控制方式 |
| --- | --- | --- |
| 公共 | mapping 归类错误 | 显式 `query_mode` / `compute_subcategory`，专用 target 不落入 generic compute |
| 公共 | 候选跨语义混用 | regime key 隔离完整 GMM weight shape、shape tail、完整 cache shape、请求/序列分组、API path、EP、topk、dtype 和输出签名 |
| 公共 | 高维插值精度不稳定 | 低维优先，按 kernel 收紧 `max_interpolation_dim`，用 holdout 决定是否保留高维 |
| MOE/DFC | DFC quant subtype、FFN 中间宽度或 EP 语义缺失导致跨配置混用 | subtype、完整 GMM1/GMM2 weight shape、weight dtype / layout 和 EP 加入 regime；当前仅 v0.18 W8A8 具备完整数据，v0.15 或其他 subtype 稳定 miss |
| Elementwise / Broadcast / Shared-token | base exact 跨 dtype 缩放，或非线性坐标导致偏差 | 有合法非标量输出时绕过 base elementwise 查询；完整输出 shape 和 profiling dtype 精确匹配；rank-2 直接使用两个 shape 维度，不按字节宽度换算 latency |
| Attention / LightningIndexer | 合成 3D 超出真实数据维度，且相同二维坐标存在不同 cache 或序列语义 | query/cache dtype、cache tail、`topk` 和请求/序列分组作为 regime，当前最高使用 `tokens + cache_blocks` 2D；语义无法对齐时 fail closed |
| Attention / SparseFlashAttention | cache 容量替代有效 KV 长度，或不同 sparse phase / 请求 / 序列语义在重复坐标被聚合 | 当前只沿 `tokens` 1D；query/cache dtype、完整 cache shape、selected topk、sparse phase 和请求/序列分组精确匹配，canonical leaf 或语义字段不完整时整条路径 fail closed |
| 上游叶子输入 | 重复计费、语义缺失或 exact/interpolation workload 不一致 | Issue #272 作为合入门槛；base 与 wrapper 消费同一叶子算子描述 |
| 公共 | profiling 数据不足 | 不外推、不跨 dtype 借用，返回带原因的 miss |

Phase2 的完成标准是算子路径、边界和回退行为清楚且可测试。真实数据未覆盖的路径保留为可识别的稳定 miss，待补充 profiling 数据后再评估覆盖率和精度。
