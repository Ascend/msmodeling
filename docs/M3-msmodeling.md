# MiniMax-M3 仿真适配设计文档

> *Simplicity is prerequisite for reliability.  —  Edsger W. Dijkstra*

---

## 1. 背景

[MiniMax-M3](https://huggingface.co/MiniMaxAI/MiniMax-M3) 是 MiniMax 的第三代多模态大语言模型，核心特点是 [Minimax Sparse Attention](https://arxiv.org/abs/2606.13392) 机制—— 在部分 GQA attention layer 中使用indexer选择top-k个 kv block 计算稀疏注意力，与标准 dense attention 混合。

---

## 2. MiniMax-M3 模型架构关键特征

> [!NOTE]
>
> Minimax-M3是VL类模型，本节仅描述其文本 backbone。

### 2.1 总体结构

```text
MiniMaxM3SparseForCausalLM
  └── model (MiniMaxM3Model)
      ├── embed_tokens (VocabParallelEmbedding)
      ├── dense_layer x 3 # 标准GQA + MLP
      │   ├── input_layernorm (GemmaRMSNorm)
      │   ├── self_attn
      │   │   ├── qkv_proj (QKVParallelLinear)           # 标准 QKV 投影
      │   │   ├── qk_norm (GemmaRMSNorm per_head)        # 按 head_dim 归一化，共享权重
      │   │   ├── rotary_emb (partial RoPE)              # 部分维度 RoPE, rotary_dim=64
      │   │   └── attn
      │   │   ├── o_proj (RowParallelLinear)             # 输出投影
      │   ├── mlp
      │   │   ├── gate_up_proj (MergedColumnParallelLinear)
      │   │   ├── down_proj (RowParallelLinear)
      │   │   └── [MoE] experts (FusedMoE) + gate (ReplicatedLinear)
      │   │       └── [MoE] shared_experts (MiniMaxM3MLP)
      │   └── post_attention_layernorm (RMSNorm / GemmaRMSNorm)
      ├── sparse_layer x 57  # Sparse Attention + MoE（有共享专家）
      │   ├── input_layernorm (GemmaRMSNorm)
      │   ├── self_attn (MiniMaxM3Attention)
      │   │   ├── qkv_proj (QKVParallelLinear)          # 标准 QKV 投影
      │   │   ├── q_norm / k_norm (RMSNorm per_head)    # 按 head_dim 归一化，共享权重
      │   │   ├── rotary_emb (partial RoPE)              # 部分维度 RoPE, rotary_dim=64
      │   │   ├── [sparse] index_q_proj / index_k_proj   # index 分支 Q/K 投影
      │   │   │   └── index_v_proj / index_o_proj        # 可选；M3-preview sparse layer 均关闭
      │   │   │   └── index_q/k_norm                     # index 分支 QK 归一化
      │   │   └── attn
      │   │   ├── o_proj (RowParallelLinear)             # 输出投影
      │   ├── post_attention_layernorm (GemmaRMSNorm)
      │   ├── moe
      │   │   ├── gate_up_proj (MergedColumnParallelLinear)
      │   │   ├── swiglu_oai
      │   │   ├── down_proj (RowParallelLinear)
      │   │   └── shared_experts
      └── final_norm (GemmaRMSNorm)
  └── lm_head (ParallelLMHead)
```

### 2.2 Sparse Attention 机制

![image-20260623203126364](https://liujiaxu-pic.oss-cn-beijing.aliyuncs.com/image-20260623203126364.png)

[Minimax Sparse Attention](https://arxiv.org/abs/2606.13392) 基于GQA实现，在块粒度（Block Granularity，默认每块 128 个 Token）上选择KV Block计算attention，从而降低计算量，支持更大的上下文长度，分为以下两个步骤

- **索引：** 在标准 GQA 层上增加了两个投影矩阵，分别计算。首先计算并为可见的 KV 块进行评分，通过最大池化（Max-pooling）得到每个block的分数，再利用 Top-$k$ 算子为每个 Query 和 GQA 组选择得分最高的 KV 块（默认选择 $k=16$ 块，即固定 2048 个 KV Token 的计算预算）。此外，当前 Query 所在的对角块会被强制保留。
- **计算稀疏注意力：** 仅对索引分支筛选出的 Top-$k$ 个 KV 块中的 Token 进行精确的 Softmax 稠密注意力计算。通过这种方式，它避免了全量序列的平方级计算瓶颈。

### 2.3 其他关键特征

| 特征 | 说明 |
|------|------|
| **QK Normalization** | `qk_norm_type="per_head"`：将 Q/K reshape 为多个`head_dim` 向量后逐 head 归一化；权重形状为`(head_dim,)`，各 head 共享权重 |
| **Partial RoPE** | `rotary_dim=64`，仅部分 head_dim 施加 RoPE |
| **Gemma 风格** | 使用`x * (1 + weight)` 的 GemmaRMSNorm |
| **MoE + Dense 混合** | `moe_layer_freq` 前 3 层为 0，layer 3-59 为 1：前 3 层 Dense MLP，后 57 层 MoE |
| **SwiGLU 变体** | `hidden_act="swigluoai"`，带`swiglu_alpha=1.702` 和`swiglu_limit=7.0` |

---

## 3. msmodeling 四层架构与适配思路

TensorCast 的性能建模采用 **"一算子一建模"** 架构，模型适配通过四层协作完成：

```text
模型定义层:  transformers/builtin_model/minimax_m3.py
   │  注册 ModelProfile + @register_custom_model
   ▼
抽象算子层:  layers/minimax_m3_attention.py (新建)
   │  封装 forward，调用虚拟算子
   ▼
虚拟算子层:  ops/minimax_m3_sparse_attention.py (新建)
   │  @register_tensor_cast_op → 定义 meta-only 算子签名
   ▼
性能模型层:  performance_model/__init__.py (追加注册)
   │  @register_op_properties → Roofline 拆解 (FLOPs + memory)
   └─ @register_op_estimator → 直接估算（可选）
```

**核心原则（参照 DSA-Indexer 的建模方法论）：**

1. **虚拟算子层**只做"身份锚定"，不参与任何计算或时间估算
2. **性能模型层**根据算子类型注册 FLOPs 和内存访问量的拆解函数
3. **拆解函数**严格依据虚拟算子 docstring 描述的算法步骤，把每一步划分到 MMA 或 GP 桶
4. **Roofline 模型**最终决定时间 = max(compute_time, memory_time)

---

## 4. Indexer 与 Sparse Attention 的性能建模公式

MiniMax-M3 sparse layer 的计算可拆成两个建模边界，本章推导各自的 FLOPs 与访存量公式：

1. **Indexer**：index key cache写入 → block score → top-k block 输出（projection / norm / RoPE 等操作复用现有op建模）
2. **Sparse Attention**：主 attention 根据 top-k block 访问标准 K/V cache

术语约定：

| 术语 | 说明 |
|------|------|
| **MMA** | Matrix Multiply-Accumulate，对应 GEMM/BMM/attention QK 和 PV 等矩阵乘累加计算；本指南按 FMA = 2 FLOPs 估算，例如 $[M, K] \times [K, N]$ 的计算量为 $2MKN$。 |
| **GP** | General Purpose elementwise/reduction work，表示非矩阵乘的通用计算，如 RMSNorm、RoPE、mask、max reduce、top-k 选择、softmax 近似等。 |
| **FLOPs** | 浮点操作数。性能模型按 dtype 将 MMA/GP 分桶，再结合吞吐和带宽做 roofline 估算。 |
| **访存量** | 逻辑读写字节数，包括 activation、weight、cache、score/top-k index 的读写；真实 kernel 有 cache reuse 时可用 lower/upper bound 修正。 |
| **lower / upper** | 同一数据可能被多个 head 复用时的访存下界/上界。`lower` 假设复用充分，`upper` 假设几乎无复用。 |

变量约定：

| 符号 | 含义 |
|------|------|
| $B$ | batch size |
| $Q_b$ | 第 $b$ 个 request 本轮 query token 数；decode 中通常为 1 |
| $L_b$ | 第 $b$ 个 request 当前总 KV 长度，包含 prefix 和本轮新增 token |
| $T$ | 本轮总 query token 数 |
| $H$ | hidden size；M3-preview 为 6144 |
| $B_n = \lceil L_b / B_s \rceil$ | 第 $b$ 个 request 的 KV block 数；在 $\sum_b$ 内按当前 request 的 $L_b$ 取值 |
| $N_q$ | 本 rank 上的 query heads |
| $N_{kv}$ | 本 rank 上的 KV heads |
| $N_{\mathrm{indexer}}$ / $D_{\mathrm{indexer}}$ | 本 rank 上的 index heads / index head dim；M3-preview 为总量 4 / 128 |
| $D$ | 标准 attention head dim；M3-preview 为 128。4.1 中 $D$ 是局部简写，含义见 4.1 开头说明 |
| $D_r$ | index RoPE 维度；M3-preview 可按`rotary_dim=64` |
| $K$ | `sparse_topk_blocks`；M3-preview 为 16 |
| $B_s$ | `sparse_block_size`；M3-preview 为 128 |
| $R$ | `sparse_local_block`；M3-preview 为 1 |
| $s$ | activation/cache dtype 字节数，bf16/fp16 时 $s=2$ |

### 4.1 Indexer

> **本节局部符号约定**：为简化公式，4.1 中的 $N$ 和 $D$ 分别表示
> $N_{\mathrm{indexer}}$ 和 $D_{\mathrm{indexer}}$，即 indexer head 数和
> indexer head dim；它们不表示 4.2 主 attention 中的 query/KV head 数或标准 head dim。

Indexer 的计算由三个阶段组成：index K cache write（§4.1.1）、block score（§4.1.2）和 top-k selection（§4.1.3）。Index 分支的 projection、norm、RoPE 不属于 Indexer 建模边界，其计算量与访存量另行统计。

#### 4.1.1 Index Cache Write

写入当前 token 的 index K cache。index K cache 为单 head，对应张量 shape `[T, 1, D]`。

计算量：

$$
\mathrm{MMA}=0,\qquad \mathrm{GP}=0
$$

访存量：

$$
\begin{aligned}
\mathrm{read\_idx\_k\_bytes\_current} &= TDs \\
\mathrm{write\_idx\_k\_cache\_bytes} &= TDs
\end{aligned}
$$

#### 4.1.2 Index Block Score

对应 vLLM 中 `_index_block_score_kernel`（`vllm/models/minimax_m3/common/ops/index_topk.py`）的 score 部分：`idx_q [T, N, D]` 与 index K cache 打分，然后按 `sparse_block_size` 聚合到 block score。index Q 有 $N$ 个 head；index K cache 为单 head（$[T, 1, D]$），各 indexer head 对同一份 K cache 独立打分。

计算量按 request 求和：

$$
\begin{aligned}
\mathrm{index\_qk\_mma}
  &= 2\sum_b Q_bNL_bD \\
\mathrm{block\_reduce\_gp}
  &\approx \sum_b Q_bNL_b
  && \text{score\_type = max}
\end{aligned}
$$

M3-preview 使用`sparse_score_type="max"`，这里只按 max reduce 建模。

访存量建议按逻辑访问量估算：

$$
\begin{aligned}
\mathrm{read\_idx\_q\_bytes}
  &= TNDs \\
\mathrm{read\_idx\_k\_cache\_bytes}
  &= \sum_b \frac{Q_b}{B_q} N L_b D s \\
\mathrm{write\_score\_bytes}
  &= 4\sum_b Q_bNB_n
\end{aligned}
$$

其中 $B_q = 64$ 是kernel中Q的tile大小：key cache被加载一次后被 $B_q$ 个 query token 共享，因此 K cache 读量按 query tile 数 $\lceil Q_b / B_q \rceil$ 而非 query token 数计；近似写成 $Q_b / B_q$。

#### 4.1.3 Top-k Selection

从 block score 中选择每个 indexer head 的 top-k block，输出 top-k block 索引供 4.2 的 sparse attention 使用（$T$ 个 token、$N$ 个 head、每个 head 选 $K$ 个 block）。

计算量：

$$
\mathrm{topk\_gp}
\approx c_{\mathrm{topk}}\sum_b Q_bNB_n
$$

其中`c_topk` 是 top-k 选择的近似常数。若性能模型只需要稳定的一阶估算， 可取 $c_{\mathrm{topk}}=K$；若要更接近选择算法，可取 $c_{\mathrm{topk}}=\lceil \log_2(\max(K, 2)) \rceil$。

访存量：

$$
\begin{aligned}
\mathrm{read\_score\_bytes} &= 4\sum_b Q_bNB_n \\
\mathrm{write\_topk\_bytes} &= 4TNK
\end{aligned}
$$

### 4.2 Sparse Attention

Sparse attention 使用 Indexer 输出的 top-k block 索引，从标准 K/V cache 中取 selected blocks 做主 attention。

计算量：

$$
\mathrm{MMA}_{\mathrm{sparse\_attn}}
  = 4N_qTDkB_k
$$

这里的系数 $4$ 来自 sparse attention 的两次矩阵乘：

$$
\underbrace{2N_qTDkB_k}_{QK^\top}
+
\underbrace{2N_qTDkB_k}_{PV}
= 4N_qTDkB_k
$$

每次矩阵乘按 FMA = 2 FLOPs 估算；$kB_k$ 表示每个 query token 访问的 selected token 上界。

访存量：

$$
\begin{aligned}
\mathrm{Bytes}_{Q+\mathrm{KV}}
  &=
    \underbrace{2sN_qTD}_{\text{Q read + O write}}
    +
    \underbrace{2sN_{kv}TkB_kD}_{\text{KV read}} \\
  &= 2sTD\left(N_q + N_{kv}kB_k\right)
\end{aligned}
$$

常量来源：

| 常量 | 来源 |
|------|------|
| 第一个 $2$ | Q 读一次，attention 输出 O 写一次，二者形状同为 $[T, N_q, D]$，因此为 $2sN_qTD$。 |
| 第二个 $2$ | KV 包含 K cache 和 V cache 两份数据，各读取一次；因此为 $2sN_{kv}TkB_kD$。 |
| $s$ | 每个元素的字节数，bf16/fp16 时 $s=2$。 |
| $kB_k$ | 每个 query token 实际访问的 selected token 上界：$k$ 个 block，每个 block $B_k$ 个 token。 |

---

## 5. 各层级详细修改方案

本章按第 3 章的四层架构顺序展开：**模型定义层 -> 抽象算子层 -> 虚拟算子层 -> 性能模型层**。适配目标是把 MiniMax-M3 sparse layer 显式拆成两个可建模边界：

1. `minimax_indexer`：index K cache write → top-k block 输出，公式见 4.1（projection / norm / RoPE 为 trace 显式 op）。
2. `minimax_sparse_attention`：根据 top-k block 访问标准 K/V cache 做 sparse attention，公式见 4.2。

### 5.1 模型定义层：注册 MiniMax-M3 ModelProfile

模型定义层负责识别 HF/SGLang 模型结构，并把模型里的 attention layer 替换或包装成 TensorCast 可追踪的抽象算子。建议新增：

```text
tensor_cast/transformers/builtin_model/minimax_m3.py
```

需要完成三件事：

1. 注册`MiniMaxM3` 的`ModelProfile` 和`@register_custom_model`。
2. 从`config.text_config.sparse_attention_config` 读取 sparse 参数。
3. 按 layer id 判断 dense/sparse layer，并把 attention 模块映射到抽象算子层。

关键配置字段：

| 字段 | 用途 |
|------|------|
| `hidden_size` | 第 4 章中的 $H$ |
| `num_attention_heads` | 标准 attention query head 总数 |
| `num_key_value_heads` | 标准 attention KV head 总数 |
| `head_dim` | 标准 attention head dim，即 4.2 的 $D$ |
| `rotary_dim` | index RoPE 维度 $D_r$ |
| `sparse_attention_freq` | 判断每层 dense / sparse |
| `sparse_num_index_heads` | indexer head 总数 $N_{\mathrm{indexer}}$ |
| `sparse_index_dim` | indexer head dim $D_{\mathrm{indexer}}$ |
| `sparse_topk_blocks` | top-k block 数 $K$ |
| `sparse_block_size` | block size $B_s$ |
| `sparse_local_block` | local block 数 $R$ |
| `sparse_score_type` | M3-preview 为`max` |

模型定义层不直接计算 FLOPs 或 bytes，只负责把这些静态配置传给后续 wrapper。对于 M3-preview，应固定以下判断：

```python
sparse_cfg = config.text_config.sparse_attention_config
is_sparse_layer = sparse_cfg["sparse_attention_freq"][layer_idx] == 1
```

注意：M3-preview 的 sparse layer 不建模 index value / index output 分支；第 5 章的适配路径只包含 index Q/K 和主 attention K/V。

### 5.2 抽象算子层：封装 MiniMax-M3 Attention Forward

抽象算子层负责把模型 forward 中的具体张量和 runtime metadata 组织成虚拟算子调用。建议新增：

```text
tensor_cast/layers/minimax_m3_attention.py
```

建议提供一个 wrapper，例如`MiniMaxM3AttentionWrapper`，职责如下：

| 职责 | 说明 |
|------|------|
| dense layer | 继续走已有 dense attention 建模路径 |
| sparse layer | 调用`minimax_indexer` 后，再调用`minimax_sparse_attention` |
| shape 提取 | 从 hidden states、QKV、seq lens、block table 中提取 $T, Q_b, L_b, B_n$ |
| config 传递 | 传递 $H, N_q, N_{kv}, D, K, B_s, R, N_{\mathrm{indexer}}, D_{\mathrm{indexer}}, D_r$ |

抽象算子层推荐的 sparse forward 逻辑：

```python
if is_sparse_layer:
    # indexer q/k proj, norm, RoPE are explicit ops before this call
    # index_keys = indexer_cache if indexer_cache is not None else idx_k_flat
    topk_idx = torch.ops.tensor_cast.minimax_indexer(
        idx_q_flat,
        index_keys,
        seq_lens,
        query_lens,
        block_table,
        topk_blocks=topk_blocks,
        block_size=block_size,
    )

    out = torch.ops.tensor_cast.minimax_sparse_attention(
        query,
        key_cache,
        value_cache,
        topk_idx,
        seq_lens,
        query_lens,
        block_table,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        topk_blocks=topk_blocks,
        block_size=block_size,
        local_blocks=local_blocks,
    )
else:
    out = torch.ops.tensor_cast.attention(...)
```

这里的`seq_lens` 和`query_lens` 很关键：性能模型应按 request 求和，不能用固定的平均序列长度替代。抽象算子层需要保证它们能传到性能模型层，至少能还原第 4 章公式中的 $Q_b$ 和 $L_b$。

### 5.3 虚拟算子层：定义 meta-only op 签名

虚拟算子层只做身份锚定和 shape 传播，不做真实计算，也不估算性能。建议新增：

```text
tensor_cast/ops/minimax_m3_sparse_attention.py
```

只需要注册两个 MiniMax-M3 专用 op。

#### 5.3.1 `minimax_indexer`

```python
@register_tensor_cast_op("minimax_indexer")
def _(
    idx_q: torch.Tensor,
    idx_k_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    query_lens: torch.Tensor,
    block_table: torch.Tensor,
    *,
    topk_blocks: int,
    block_size: int,
    seq_lens_values: list[int] | None = None,
    query_lens_values: list[int] | None = None,
    is_decode_values: list[bool] | None = None,
) -> torch.Tensor:
    """
    MiniMax-M3 indexer fused op.

    Boundary:
      index K cache write -> index QK block score -> top-k block indices.

    Index q/k projections, norm, and RoPE are explicit ops in
    MiniMaxM3AttentionWrapper.forward.

    Performance formula: see section 4.1.
    """
    total_tokens = idx_q.shape[0]
    num_indexer_heads = idx_q.shape[1]
    return torch.empty(
        (total_tokens, num_indexer_heads, topk_blocks),
        dtype=torch.int32,
        device="meta",
    )
```

返回形状不要求和 kernel 完全一致，只要性能模型能从 op 参数中拿到 $T, N, K$ 即可。若 runtime 的 top-k index 实际按`[N, T, K]` 或 paged layout 排布，docstring 中说明等价即可。

#### 5.3.2 `minimax_sparse_attention`

```python
@register_tensor_cast_op("minimax_sparse_attention")
def _(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    topk_idx: torch.Tensor,
    seq_lens: torch.Tensor,
    query_lens: torch.Tensor,
    block_table: torch.Tensor,
    *,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    topk_blocks: int,
    block_size: int,
    local_blocks: int,
) -> torch.Tensor:
    """
    MiniMax-M3 sparse attention fused op.

    Boundary:
      read Q + selected K/V cache -> sparse QK/PV attention -> output O.

    Performance formula: see section 4.2.
    """
    return torch.empty_like(query).contiguous()
```

同时在`tensor_cast/ops/__init__.py` 中导入：

```python
from . import minimax_m3_sparse_attention  # noqa: F401
```

不建议在虚拟算子层注册`minimax_attention_hybrid` 或 index value 相关 op。M3-preview 的 sparse layer 只需要上面两个边界，dense layer 复用已有 attention / linear / norm / rotary / MoE 算子即可。

### 5.4 性能模型层：注册 op properties

性能模型层负责把两个虚拟 op 映射到 Roofline 需要的 MMA、GP 和 memory bytes。修改位置：

```text
tensor_cast/performance_model/__init__.py
```

建议新增两个拆解函数，并注册到对应 op：

```python
@register_op_properties(torch.ops.tensor_cast.minimax_indexer.default)
def _estimate_minimax_indexer_breakdown(...):
    ...

@register_op_properties(torch.ops.tensor_cast.minimax_sparse_attention.default)
def _estimate_minimax_sparse_attention_breakdown(...):
    ...
```

#### 5.4.1 `minimax_indexer` 拆解

`minimax_indexer` 使用 4.1.4 的汇总口径，仅建模 index K cache write、block score 和 top-k。Projection / norm / RoPE 由 trace 中的显式 op 单独统计。

输入量定义：

```python
T = idx_q.shape[0]
N = idx_q.shape[1]
D = idx_q.shape[2]
K = topk_blocks
B_s = block_size
B_n = ceil(seq_lens / B_s)
```

计算桶：

```text
MMA = 2 * sum_b(Q_b * N * L_b * D)

GP  = sum_b(Q_b * N * L_b)
    + c_topk * sum_b(Q_b * N * B_n)
```

访存桶按 4.1.4 的`Bytes_indexer` 汇总。实现时可以先把各阶段拆成命名变量，再求和，方便 profiling 校正：

```python
bytes_cache_write = 2 * T * D * s
bytes_score = T * N * D * s + sum_b((Q_b / B_q) * N * L_b * D * s) + 4 * sum_b(Q_b * N * B_n)
bytes_topk = 4 * sum_b(Q_b * N * B_n) + 4 * T * N * K
bytes_total = bytes_cache_write + bytes_score + bytes_topk
```

#### 5.4.2 `minimax_sparse_attention` 拆解

`sparse attention` 使用 4.2 的口径。令每个 request 的 selected token 上界为：

```text
A_b = min(L_b, min(B_n, K + R) * B_s)
```

计算桶：

```text
MMA = 4 * sum_b(Q_b * N_q * A_b * D)
GP  = 6 * sum_b(Q_b * N_q * A_b)
```

访存桶：

```text
Q/O bytes = 2 * s * T * N_q * D
KV bytes  = 2 * s * sum_b(Q_b * A_b * N_kv * D)
topk bytes = 4 * T * N_kv * K
bytes_total = Q/O bytes + KV bytes + topk bytes
```

如果实现暂时使用 decode-only 快速估算，也可以令`A_b = min(L_b, (K + R) * B_s)`；但最终版本应保留 per-request 的 $Q_b, L_b$ 求和。

### 5.5 修改汇总表

| 层级 | 文件 | 主要改动 |
|------|------|----------|
| 模型定义层 | `tensor_cast/transformers/builtin_model/minimax_m3.py` | 注册 MiniMax-M3 profile，读取`text_config.sparse_attention_config`，按 layer id 标记 sparse layer |
| 抽象算子层 | `tensor_cast/layers/minimax_m3_attention.py` | 包装 attention forward，dense layer 复用已有 attention，sparse layer 调用`minimax_indexer` + `minimax_sparse_attention` |
| 虚拟算子层 | `tensor_cast/ops/minimax_m3_sparse_attention.py` | 注册两个 meta-only op：`minimax_indexer`、`minimax_sparse_attention` |
| 虚拟算子导入 | `tensor_cast/ops/__init__.py` | 导入`minimax_m3_sparse_attention` |
| 性能模型层 | `tensor_cast/performance_model/__init__.py` | 注册两个 op properties 拆解函数，实现第 4 章公式 |

## 6. 注册流程完整追溯

```text
用户输入: python -m cli.inference.text_generate \
    --model-id minimax/MiniMax-M3 \
    --num-queries 1 --query-length 128 \
    --decode
    │
    ▼
UserInputConfig.from_args(args)
    │  model_id="minimax/MiniMax-M3"
    │
    ▼
build_model(user_input)
    │  tensor_cast/core/model_builder.py:50
    │
    ▼
ConfigResolver.resolve()
    │  1. 从 HuggingFace 加载 config.json
    │  2. model_type = "minimax_m3_vl"
    │  3. 读取 text_config.sparse_attention_config
    │  4. ModelProfile 注册 → attention / MoE / MTP 等静态信息
    │
    ▼
TransformerModel(model_id, config)
    │  tensor_cast/transformers/model.py
    │
    ├── import tensor_cast.transformers.builtin_model
    │   → import builtin_model.minimax_m3  (自动加载)
    │      │  → register_model_profile(ModelProfile("minimax_m3_vl", ...))
    │      │  → @register_custom_model("minimax_m3_vl") 注册自定义函数
    │      ▼
    ├── get_custom_model("minimax_m3_vl") → custom_fn
    │
    ├── custom_fn(self):  ← 自定义 patch 流程
    │   ├── wrap_model(self)                      # CausalLmWrapper
    │   ├── maybe_enable_mtp(self)                # MTP
    │   ├── maybe_reuse_layers(self)              # 层复用
    │   ├── patch_rotary_emb(self)                # RoPE 替换
    │   ├── patch_minimax_m3_attention(self)      # ← 新: 替换 attention
    │   │   ├── 解析 sparse_attention_config
    │   │   ├── 识别 sparse_layer_ids
    │   │   └── 替换 self_attn → MiniMaxM3AttentionWrapper
    │   ├── patch_moe(self)                       # MoE 替换
    │   ├── quantize_model(self)                  # 量化
    │   └── shard_model(self)                     # TP/EP 切分
    │
    ▼
generate_inputs() → 构造输入
    │
    ▼
Runtime.__enter__()
    │  → 激活 TorchDispatchMode
    │
    ▼
model.forward(**inputs)
    │
    ▼
在每层 forward 中:
  MiniMaxM3AttentionWrapper.forward()
    │
    ├── dense layer:
    │     调用已有 tensor_cast.attention / linear / norm / rotary / MoE 算子
    │
    └── sparse layer:
          1. indexer q/k proj, norm, fused_rope（显式 op）
          2. 调用 torch.ops.tensor_cast.minimax_indexer(...)
          3. 调用 torch.ops.tensor_cast.reshape_and_cache（主 KV）
          4. 调用 torch.ops.tensor_cast.minimax_sparse_attention(...)
    │
    ▼
Runtime.__torch_dispatch__(...)
    │  → 分别为 minimax_indexer / minimax_sparse_attention 注册 OpInvokeInfo
    │
    ▼
Runtime.__exit__()
    │
    ├── repeat_op_invoke_infos()
    └── replay_op_invoke_infos()
        │
        ▼
    AnalyticPerformanceModel.process_op()
        │
        ├── get_op_estimator(op, device)
        │   → 未注册专用 estimator → _estimate_default
        │
        ├── _estimate_default()
        │   │
        │   ├── minimax_indexer.get_perf_properties()
        │   │   │
        │   │   └── _estimate_minimax_indexer_breakdown()
        │   │       │
        │   │       └── 拆解为 4.1.4:
        │   │           ├── index_qk_mma + block_reduce_gp
        │   │           ├── topk_gp
        │   │           └── Bytes_indexer
        │   │
        │   ├── minimax_sparse_attention.get_perf_properties()
        │   │   │
        │   │   └── _estimate_minimax_sparse_attention_breakdown()
        │   │       │
        │   │       └── 拆解为 4.2:
        │   │           ├── sparse_qk_mma + sparse_pv_mma
        │   │           ├── softmax_gp + mask_gp
        │   │           └── Q/O + selected K/V + topk index bytes
        │   │
        │   └── Roofline: time = max(compute_time, memory_time) + static_cost
        │
        ▼
    RuntimeEvent(op_invoke_info, perf_results)
    self.event_list.append(event)
    │
    ▼
_build_model_timelines() → token 依赖 + stream 并行
    │
    ▼
total_execution_time_s()
table_averages()
get_breakdowns()
export_chrome_trace()
```

---

## 7. 测试验证

### 7.1 最小验证测试

```python
# tests/regression/tensor_cast/test_minimax_m3.py

import torch
from tensor_cast import Runtime, device_profiles
from tensor_cast.performance_model.analytic import AnalyticPerformanceModel

def test_minimax_m3_sparse_attention_ops():
    """Verify sparse attention ops are registered and estimatable."""
    device = device_profiles.TEST_DEVICE
    perf_model = AnalyticPerformanceModel(device)

    with Runtime(perf_model, device) as runtime, torch.no_grad():
        seq_lens = torch.tensor([4096], device="meta")
        query_lens = torch.tensor([1], device="meta")
        block_table = torch.empty(1, 32, device="meta", dtype=torch.long)

        topk_idx = torch.ops.tensor_cast.minimax_indexer(
            torch.empty(1, 4, 128, device="meta"),
            torch.empty(1, 1, 128, device="meta"),
            seq_lens,
            query_lens,
            block_table,
            topk_blocks=16,
            block_size=128,
        )

        out = torch.ops.tensor_cast.minimax_sparse_attention(
            torch.empty(1, 64, 128, device="meta"),
            torch.empty(4096, 4, 128, device="meta"),
            torch.empty(4096, 4, 128, device="meta"),
            topk_idx,
            seq_lens,
            query_lens,
            block_table,
            num_q_heads=64,
            num_kv_heads=4,
            head_dim=128,
            topk_blocks=16,
            block_size=128,
            local_blocks=1,
        )

    # Verify results
    table = runtime.table_averages()
    assert "minimax_indexer" in table
    assert "minimax_sparse_attention" in table

    total_time = runtime.total_execution_time_s()
    print(f"Total execution time: {total_time}")
    print(table)
```

### 7.2 端到端仿真测试

```python
# 终端命令
python -m cli.inference.text_generate \
    --model-id minimax/MiniMax-M3 \
    --num-queries 1 \
    --query-length 128 \
    --decode \
    --tp-size 8 \
    --device ATLAS_800_A3_752T_128G_DIE
```

端到端对比测试用例：

| 硬件 | 并发数 | 输入长度 | 输出长度 | msmodeling 仿真 TTFT | msmodeling 仿真 TPOT | 硬件实测 TTFT | 硬件实测 TPOT | 误差 |
|------|--------|----------|----------|----------------------|----------------------|----------------|----------------|------|
| B200 | 1 | 64 | 1k |  |  |  |  |  |
| B200 | 4 | 64 | 1k |  |  |  |  |  |
| B200 | 32 | 64 | 1k |  |  |  |  |  |
| B200 | 64 | 64 | 1k |  |  |  |  |  |
| A5 | 1 | 64 | 1k |  |  |  |  |  |
| A5 | 4 | 64 | 1k |  |  |  |  |  |
| A5 | 32 | 64 | 1k |  |  |  |  |  |
| A5 | 64 | 64 | 1k |  |  |  |  |  |

对比方法：输入长度固定为 64 tokens，输出长度固定为 1k tokens；分别在 B200 和 A5 上采集硬件实测结果，并与 msmodeling 仿真的 TTFT、TPOT 做误差对比。

---

## 8. 参考

### 8.1 msmodeling

- DSA-Indexer 建模方法论：`notes/msmodeling-DSA-Indexer-性能仿真建模分析.md`
- 现有类似模型适配：`tensor_cast/transformers/builtin_model/minimax_m2.py`
- 自定义 patch 链参考：`tensor_cast/transformers/builtin_model/kimi_k25.py`
- Op 定义示例：`tensor_cast/ops/attention.py`, `tensor_cast/ops/mla.py`
- Op Properties 注册示例：`tensor_cast/performance_model/__init__.py` (`dsa_indexer`, `attention`)
- 性能模型基类：`tensor_cast/performance_model/base.py`

### 8.2 mm-sglang-triton

- MiniMax-M3 模型定义：`python/sglang/srt/models/minimax_m3.py`
- Sparse Attention 后端：`python/sglang/srt/layers/attention/minimax_sparse_backend.py`
- Sparse Attention Triton 算子：`python/sglang/srt/layers/attention/minimax_sparse_ops/`

### 8.3 Minimax-M3-preview

- 模型配置：`config.json`
