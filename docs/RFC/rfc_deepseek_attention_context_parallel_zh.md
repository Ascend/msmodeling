# RFC: DeepSeek 风格注意力的 Context Parallel 建模方案

## Metadata

| Item | Content |
|:---|:---|
| **Status** | Draft |
| **Author** | Codex |
| **Updated Date** | 2026-07-13 |
| **Related Links** | https://github.com/vllm-project/vllm/issues/30055 |

---

## 1. 问题描述

在 W8A8 DeepSeek-style 稀疏注意力模型的仿真与真实引擎对齐过程中，发现 TensorCast 和 vLLM-Ascend 在静态权重显存、上下文并行切分、注意力算子选择上存在差异。GLM-5.1 是当前用于复现和校验的 workload，但本方案不是 GLM-5.1 专用方案，DSA CP 应建模为面向 DeepSeek-style / DSA-like 稀疏注意力结构的通用上下文并行机制。

当前主要差异如下：

- 静态权重显存低估：TensorCast 旧逻辑统计到的权重约为 `28.452 GB`，真实引擎观测到约 `38 GB`；部分配置下真实快照约为 `31 GB`，差异与 DSA CP 路径和权重量化方式相关。
- 权重量化分类不完整：W8A8 目录中仍存在 float tensor，且真实引擎还会额外持有 `weight_scale`、`weight_offset`、fp32 scale、转置权重、派生权重和临时布局。
- `kv_b_proj` 处理不一致：DSA CP 路径下真实引擎没有按普通 TP 方式切分 `kv_b_proj`，TensorCast 需要在相关路径中保留 full layout 并据此派生 `W_UK_T` / `W_UV`。
- 注意力后端不一致：真实引擎在 DSA 稀疏注意力路径使用 SFA，当前 TensorCast 仍主要通过 MLA/FIA 风格算子建模。

首阶段只覆盖模型主体功能类、线性层切分语义和 sequence parallel 相关通信改写，不包含 NZ 布局、PCP/DCP、ServingCast 层改造。

## 2. 方案描述

方案目标是在 TensorCast 中对齐三类行为：

1. 静态权重和派生权重的显存统计。
2. 复用 sequence parallel pass 的 token 维通信改写。
3. DSA 稀疏注意力路径的 SFA 算子建模。

设计原则：

- DSA CP 不是模型专用开关，应由 `sequence_parallel` 能力和 DSA-like 结构检测共同触发。
- `ParallelConfig` 继续只表达并行 size，不放行为开关；运行时行为由模型结构、transformer config 或执行配置推导。
- Sequence collective 复用现有 `tp_group`。首阶段不新增 `sharded_cp_group`，避免把 DSA CP 建模成独立通信域。但在 DSA projection 复制部署时，MLA 初始化需要使用有效 `world_size=1` 的 TP 视图，使 head 数和派生权重布局与 `q_b_proj` / `kv_b_proj` 的有效 TP size 一致，同时不修改共享 group。
- SFA 输入按真实 profile 的 bf16 语义建模，不因为模型中存在量化权重就引入量化 SFA 输入算子。
- Dense MLA 的量化路径继续保留；线性权重量化和注意力 kernel 输入量化需要分开建模。

当前基线中已经具备的能力：

- profiling datasource 的重复 wrapping 问题已修复。
- TP sharding plan 可以通过 `disable_tp` 保持 `q_b_proj` / `kv_b_proj` 为 replicated full layout，MLA 据此从完整权重派生 `W_UK_T` / `W_UV`。
- sparse attention 的三返回值路径已经由当前 transformers 基线支持。

当前还未合入的能力：

- SFA attention replacement。
- SFA profiling decomposer / op mapping。
- 基于 `quant_model_description.json` 或 MemoryTracker 的静态权重统计重构。
- DeepSeek-style / 当前 GLM5 workload 结构特性在 sequence parallel pass 基础上的叠加逻辑。

## 3. 方案详细设计

### 3.1 DSA CP 触发条件

DSA CP 不新增独立的 `--enable-dsa-cp` 入口。推荐条件为：

- 用户启用 sequence parallel。
- 模型结构中存在 DSA-like sparse attention 所需的结构特征。

`_has_dsa_structure` 应保持轻量，只判断结构能力，不依赖 `model_type` 白名单。这样可以让同类结构的新模型复用相同路径。

### 3.2 Sequence Parallel 通信改写

DSA CP 场景下的 `all_reduce -> reduce_scatter` 不应作为新的通信组能力实现，而应复用 TensorCast 已有的 `sequence_parallel_pass`。

现有 `sequence_parallel_pass` 已覆盖 vLLM-Ascend 的核心 SP 语义：

- 对 `all_reduce -> rms_norm / add_rms_norm` 模式，改写为 `reduce_scatter -> rms_norm / add_rms_norm -> all_gather`。
- 对 `all_reduce -> add_rms_norm2` 模式，先 `reduce_scatter`，并根据后续消费关系选择性插入 `all_gather`。
- 对 residual 链路，允许中间 residual 保持 local shard，再在需要 full-token 输出的位置补 `all_gather`。
- `reduce_scatter` 和 `all_gather` 均沿 token/sequence 维切分，并复用 `all_reduce` 原有的 `rank_group`，即普通 `tp_group`。

因此，首阶段设计不新增 `sharded_cp_group`。TensorCast 需要补齐的是：在 DeepSeek-style / DSA-like 结构被识别后，让模型层、线性层和注意力层的局部 token / full token 语义与 `sequence_parallel_pass` 的改写结果保持一致。

### 3.3 DSA 结构叠加逻辑

DSA CP 可以理解为在 sequence parallel pass 基础上叠加的结构化特性：

- 触发条件仍然是 `enable_sequence_parallel and _has_dsa_structure()`。
- Sequence 通信仍走 `tp_group`，不创建额外 CP group。复制 DSA attention 权重时，MLA 构造使用 singleton 有效 group；共享/全局 `tp_group` 仍用于 SP collective。
- `q_b_proj` / `kv_b_proj` 等 DSA 相关 projection 需要避免普通 TP 切分，保留 full layout。
- DSA CP prefill 路径中的 `o_proj` / `wo_b` 使用 replicated full weight，不执行普通 row-parallel feature all-reduce；sequence pass 在 projection 前将 full-token attention 输出切到 local-token 范围。
- attention 前后需要明确 hidden states 是 local-token shard 还是 full-token layout；需要 full Q/K/V 时插入或保留 all-gather 语义。
- 在当前不拆分 `dsa_indexer` 单体 op 的前提下，DSA indexer 输入边界需要保证 `hidden_states` 恢复 full-token layout，以便后续 SFA 获得完整 sequence 语义。如果上游已有 norm 后 `all_gather`，则不需要新增 indexer 专用 pattern；如果后续调试发现存在 norm 输出保持 local shard 且直接进入 `dsa_indexer` 的路径，再补 full-token consumer boundary pattern。
- 上述 graph/layout 约束不表示 indexer top-k query 必须按 full token 计算。性能模型内部需要区分：top-k query 侧按 local token shard 建模，cache/compressor 更新和 full-sequence 边界按 full token 建模。

#### 3.3.1 Decoder 层间 Token Layout 契约

Token 维需要显式区分两种布局状态：

- **Full-token**：当前 rank 可见完整 prefill token 范围，例如 `4096` tokens。
- **Local-token**：token 范围已经按 TP/SP group 做 sequence shard，例如 `4096 / 16 = 256` tokens。

Prefill decoder layer 的布局契约如下：

| 边界或组件 | 需要的布局 | 原因 |
|:---|:---|:---|
| Attention 输入、Q/K/V projection、DSA indexer 语义边界和 attention 计算 | Full-token | Sparse attention 语义和当前 fused indexer 边界需要完整 sequence 视图。 |
| `o_proj` 前的 attention 输出 | Full-token、full-feature | 当前 GLM workload 中为 `[4096, 64, 256]`；DSA attention 对完整 query-token 和全部 heads 做计算。 |
| `o_proj` 前的 token slice | Full-token 到 local-token | 只切 token 维，例如 `[4096, 64, 256] -> [256, 64, 256]`；这是图布局转换，不是 feature reduce-scatter。 |
| Replicated `o_proj` | Local-token、full-feature 输入 | 展平为 `[256, 16384]`，使用完整 `[16384, 6144]` 权重计算；不执行 TP feature all-reduce，也不再次切 token。 |
| Post-attention norm 和 residual | Local-token | Norm 和逐元素 residual 计算不需要完整 token 范围。 |
| MoE gate、top-k、routed experts 和 shared experts | Local-token | 所有 MoE 分支消费同一份已切分 token。Expert dispatch 可以重排 expert，但不能恢复 TP full-token 范围。 |
| Shared-expert TP reduction | Local-token | 该 all-reduce 用于合并 feature shard，并保持 local token 维；不能再次执行 sequence reduce-scatter。 |
| MoE 输出和下一层 residual merge | Local-token | Routed/shared 两个分支必须以相同 local-token shape 汇合。 |
| 下一层 input norm | Local-token | 在回到 attention domain 前先完成本地 norm。 |
| 进入下一层 attention 的 normalized value | Local-token 到 full-token | 只对 output 0（normalized attention input）插入 all-gather；output 1（residual）始终保持 local，供下一次 replicated `o_proj`/norm 边界使用。 |

稳定态 prefill 流程为：

```text
full-token attention
    -> 沿 token 维切到 local-token
    -> local-token full-weight o_proj（无 feature all-reduce）
    -> local-token post-attention norm/residual
    -> local-token gate/top-k
    -> local-token routed experts + local-token shared experts
    -> local-token residual merge 和下一层 input norm
    -> 只对 normalized attention input 执行 all_gather(token dim)
    -> 下一层 full-token attention
```

该契约可以避免三类常见错误：

1. 在 MoE gate/shared expert 前过早恢复 full tokens，导致 MoE 计算量和通信量高估。
2. 将 local tokens 直接传播到下一层 attention，导致 attention/indexer 计算量低估。
3. 对跨层 residual 输出执行 all-gather。展开图中这会在下一边界再次 gather，使 `4096` tokens 错误扩张到 `65536`；只有 normalized 输出可以回到 full-token 域。

Shared-expert 路径需要特别区分通信意图。Feature parallel 和 sequence parallel 可能复用同一个 TP group，因此不能仅根据 collective 名称判断布局变化：local-token 输入上的 shared-expert all-reduce 保持 token layout。Replicated DSA `o_proj` 没有 feature all-reduce；其 full-token 到 local-token 的转换由 projection 前显式 token slice 完成。

#### 3.3.2 展开图与 Repetition 图的等价性

Layer repetition 只是图规模优化，不能改变 token-layout 契约。展开图显式地在每个 decoder layer 间传播状态；repetition 图则通过 `region_end -> copy_region* -> region_begin` 边界传播同一状态，并在 runtime 展开 representative region。

Repetition 展开后，两种图必须得到相同的有效算子 shape 和通信次数：

- 每一层 attention 和 fused DSA indexer 边界保持 full-token；
- 每一个 MoE 层的 gate、routed experts 和 shared experts 保持 local-token；
- 每个 attention-to-FFN 边界在 replicated full-weight `o_proj` 前将 attention 输出切为 local-token；
- 每个 FFN-to-next-attention 边界只 gather normalized attention input。

Region marker 对 layout 是透明的。它可能阻止跨层 fusion 以完全相同的形式出现，例如一张图使用 `add_rms_norm2`，另一张图使用等价的 `add + rms_norm`；但它不能改变 full-token/local-token shape 或通信语义。

### 3.4 线性层切分语义

对 RowParallelLinear / ColumnParallelLinear 的约束应保持一致：

- `disable_tp=True` 表示该层不按当前 TP group 切分权重。
- `gather_slice_data` 需要受 `disable_tp` 约束，避免对已经保留 full layout 的权重再做错误 gather/slice。
- 不修改 `slice_input_by_last_dim`。在 sequence parallel pass 前提下，该变量对应输入切分行为，不应被 DSA CP 的权重布局逻辑复用。

### 3.5 MLA 派生权重

DSA CP 保持 `kv_b_proj` replicated 时：

- `kv_b_proj.weight` 使用 full layout。
- MLA 初始化使用有效 `world_size=1`，使 `_num_heads_per_rank` 自然采用 full-head 语义，无需在 MLA 内增加独立的 `disable_tp` 分支。
- `W_UK_T` / `W_UV` 从 full `kv_b_proj` 派生。

这部分是解释 `31 GB` 和 `38 GB` 显存差异的重要路径：真实引擎除了原始 checkpoint 权重，还会持有派生权重、scale/offset 和布局转换相关 tensor。

### 3.6 SFA 注意力算子

新增 SFA 建模时只引入 bf16 sparse attention 算子：

- `tensor_cast.mla_sparse_attention.default`

不引入：

- `tensor_cast.mla_sparse_attention_quant.default`

原因是当前真实 profile 中 SFA 输入为 bf16。`quant_config` 只能说明模型中存在量化线性权重，不能推导出 SFA kernel 输入被量化。

Dense MLA 路径保持现状：

- `tensor_cast.multihead_latent_attention.default`
- `tensor_cast.multihead_latent_attention_quant.default`

### 3.7 静态权重统计

短期可以在现有权重统计中补齐以下类别：

- float exception tensor。
- `weight_scale` / `weight_offset`。
- fp32 scale。
- 转置权重和临时布局。
- `W_UK_T` / `W_UV` 等派生权重。
- `disable_tp=True` 后不再 TP 切分的 full `kv_b_proj`。

中长期建议读取 `quant_model_description.json` 判断权重量化状态，并评估将静态权重统计迁移到模型 forward 阶段的 MemoryTracker 口径，以减少静态公式和真实执行路径之间的偏差。

### 3.8 MoE Router Token Layout

TensorCast 需要区分 `moe_gate_returns_raw_logits` 的字面含义和当前代码里的实际分支选择语义。在当前 TensorCast MoE 实现中，`moe_gate_returns_raw_logits=True` 是 EP 打开时选择 full packed-local token 上先执行 gate 的分支：

```python
if self.has_ep and self._inner.moe_config.gate_returns_raw_logits:
```

这个分支对应 vLLM-Ascend decode profile 中看到的 shape：

```text
gate matmul:     [9, 6144] -> [9, 256]
pad + TP slice:  [9, 256]  -> [12, 256] -> [3, 256]  # dp8tp4
topk:            [3, 256]  -> [3, 8]
routed expert:   [3, 6144]
```

因此，对于当前 GLM5 workload，TensorCast 应保持 `moe_gate_returns_raw_logits=True`。它用于建模 decode 阶段的 “full-token gate，sliced-token topk/dispatch” 路径，不能只按名字理解为是否返回 raw logits。

Prefill 和 decode 的差异来自进入 MoE 时 token layout 不同：

- Prefill 开启 sequence parallel / DSA CP 时，attention 输出在 MoE 前已经经过 `reduce_scatter` 或 sequence shard。此时 gate、topk、routed expert、shared expert 都应在 local token shard 上执行，例如 `[256, 6144] -> [256, 256] -> [256, 8]`。
- Decode 走 MC2 / fused MC2 时，gate matmul 可以先在 full packed-local decode tokens 上执行，然后 MoE prepare 对 `hidden_states` 和 `router_logits` 同时按 TP 做 pad/slice，再进入 topk 和 routed expert dispatch。对于 `dp8tp4` 且 9 tokens 的场景，就是 `9 -> pad 12 -> slice 3`。

vLLM-Ascend 中相关逻辑分散在通用 MoE 路径中：

- `shared_forward_impl` 可能先基于 full `hidden_states` 计算 router logits。
- `PrepareAndFinalizeWithMC2.prepare` 对 `hidden_states` 和 `router_logits` 同时按 TP 做 pad/slice。
- `select_experts` / `moe_gating_top_k` 消费切分后的 router logits。

TensorCast 应把这几类语义拆开建模：

- `moe_gate_returns_raw_logits=True` 选择 full-token gate 分支，并允许 route 阶段对 router logits 做 pad/slice。
- Sequence parallel / DSA CP 决定 prefill 阶段 MoE 输入是否已经是 local token shard。
- MoE communication prepare 决定 decode 阶段即使 gate matmul 看到 full packed-local tokens，topk/dispatch 是否仍使用 TP-local slice。

## 4. 模块交互设计

整体交互如下：

```text
Model config / runtime config
        |
        v
DSA structure detection + sequence_parallel
        |
        v
SequenceParallelPass
        |-- all_reduce -> reduce_scatter
        |-- full-token attention -> local-token slice -> replicated o_proj
        |-- local norm / residual chain
        |-- 仅在需要 full-token 时 all_gather normalized 输出
        |-- residual 输出跨层保持 local
        |-- reuse tp_group
        |
        v
Linear layer wrappers
        |-- disable_tp
        |-- gather_slice_data constraint
        |
        v
MLA / sparse attention wrapper
        |-- replicated DSA projection 使用 singleton 有效 TP 视图
        |-- full kv_b_proj when needed
        |-- W_UK_T / W_UV materialization
        |-- preserve/insert all_gather for dsa_indexer hidden_states when no upstream full-token boundary exists
        |-- SFA op dispatch
        |
        v
Performance model + memory tracker
        |-- SparseFlashAttention mapping
        |-- dsa_indexer top-k modeled on local tokens
        |-- dsa_indexer cache/compressor modeled on full tokens
        |-- profiling / analytic fallback
        |-- static and forward memory accounting
```

模块职责：

- 模型结构检测：只判断是否具备 DSA-like sparse attention 结构，不写模型名白名单。
- Sequence parallel pass：复用 `tp_group`，改写可切分 collective 链路，在 replicated `o_proj` 前切分 full-token DSA attention 输出，仅 gather normalized 输出，并让 residual 输出跨层保持 local。
- 线性层：统一解释 `disable_tp`，保证 RowParallelLinear 和 ColumnParallelLinear 的 gather/slice 行为一致。
- 注意力层：必要时保证 `dsa_indexer` 的 `hidden_states` 输入已恢复 full-token layout；优先复用上游 sequence parallel pattern 已插入的 `all_gather`，仅在调试确认存在 local norm output 直接进入 `dsa_indexer` 时新增 consumer boundary pattern。注意力层还需要根据 sparse attention 路径调度 SFA op，避免把权重量化误解释为 SFA 输入量化。
- 性能模型：将 SFA op 映射到真实 `SparseFlashAttention` profile；缺少 profile 时使用 analytic fallback。对于当前单体 `dsa_indexer`，内部成本需要拆开估算：top-k query 侧使用 local token 数，cache/compressor 更新和 full-sequence 边界使用 full token 数。
- 显存统计：补齐量化辅助 tensor、派生权重和 full-layout 权重。

## 5. 进行中的工作和局限性

已完成或当前基线已有：

- profiling datasource wrapping 修复。
- `kv_b_proj.disable_tp` 对 full-layout 派生权重的基础支持。
- sparse attention 三返回值路径适配当前 transformers 基线。
- DSA full-head MLA 初始化使用 singleton 有效 TP 视图，同时共享 TP group 继续用于 sequence collective。
- full-token attention 到 local-token replicated `o_proj` 的改写，以及 normalized/residual 两类输出的差异化处理。
- repetition 开启和关闭时执行等价；完整 78 层图需要命中全部 78 个 DSA 边界，且 residual token 范围不能被成倍放大。

进行中的工作：

- 在当前 shape/layout 和 repetition 等价性用例基础上继续扩展 DSA CP 回归覆盖。
- 重新实现 SFA op dispatch，并补 decomposer / op mapping / fallback。
- 用 `quant_model_description.json` 提升权重量化分类准确性。

局限性：

- 首阶段不建模 NZ 格式额外显存。
- 首阶段不建模 vLLM-Ascend layer-sharding broadcast。真实 DSA-CP 可将 `o_proj` / `q_b_proj` 等 full-layout 权重按 layer 分散驻留，并在 forward 前异步 broadcast 到当前 rank；这会影响权重驻留、临时 shard window 和峰值显存估计，但不影响本 RFC 的主体功能路径。
- 首阶段不接入 PCP/DCP。
- 首阶段不改 ServingCast。
- SFA profiling 数据缺失时，性能结果只能依赖 analytic fallback，无法完全对齐真实 profile。
- 静态显存统计在迁移到 MemoryTracker 口径前，仍可能和真实引擎执行期物化行为存在偏差。
