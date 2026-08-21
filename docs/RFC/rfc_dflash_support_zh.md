# RFC: Dflash 统一建模方案

## 1. Overview

Status (状态): Draft  
Author(s) (作者): linjingjie  
Created (创建日期): 2026-07-30  
Updated (更新日期): 2026-08-12（CLI 迁移为 `--speculative-method` / `--num-speculative-tokens` / `--acceptance-length`）
Related Issue/PR (相关 Issue/PR):  
Supersedes (替代): 双模式草案中「Mode A / Mode B」分裂方案；本 RFC **不再区分 Mode A / Mode B**。  

本文档为 Dflash 统一建模的**唯一规范说明**（含 Forward 契约、KVCache 构建与验收账本）。  
落地代码：`tensor_cast/layers/dflash.py`、`tensor_cast/layers/dflash_qwen3.py`。

---

## 1.1 Summary

本提案为 TensorCast 增加 **Dflash 并行 block 投机解码**的统一性能建模能力。

Dflash 草稿一次 forward，在长度为 `dflash_block_size` 的 block（1 个 anchor + `block_size-1` 个 MASK）上并行预测候选 token；target 对同一 block 做一次 verify；接受长度仅用于吞吐折算。

### 核心需求

启用 Draft-model 后，Decode 整图必须能以 **TensorCast 算子**形式完整计入：

- **算子调用次数**（含 draft 侧 N 次 GQA Attention、Linear/MLP，以及 target verify）；
- **算子调用时间**（走现有 performance model / profiling 映射，而非 `eager` matmul 旁路）。

### Draft 构图公式（本 RFC 硬性约定）

```text
Draft =
  内置 Qwen3Config
  + Qwen3DFlash 层骨架（DecoderLayer / Attention 投影与 MLP / RMSNorm / Rotary）
  + 一次 context_kv_proj（全层 context K/V）+ paged reshape_and_cache
  + TC attention 注册（attention_by_layers → torch.ops.tensor_cast.attention[*]）
  + Decoder 仅短 noise：apply_rope(Q,K_n) + cache + Attn（context 已在 cache）

不复用 target 的 MLA 导入 / patch_mla 路径。
```

统一设计要点：

1. **单一草稿族**：Qwen3 GQA + dense MLP；Attention **出口**复用项目已有 TC `attention` 算子；**入口**为 DFlash paged KV（context 预写 cache + noise 追加），不得退化为「仅 noise 自注意力」或对满长 context 每层重复 `k/v_proj`。
2. **内置草稿结构配置**：仓库内置 Qwen3 draft profile；不强制用户导入完整外部 draft config；`hidden_size` / `vocab_size` / `max_position_embeddings` 从 target 覆盖。
3. **共享 target 的 `embed_tokens` / `lm_head`**；**`rotary_emb` 由草稿自持**。
4. **Attention 类型由 config 的 `layer_types` 逐层决定**（见 3.1.5）：  
   - `layer_types[i] == "sliding_attention"` → 该层启用 `sliding_window`（config 缺省时默认 **2048**）；  
   - `layer_types[i] == "full_attention"` → 该层为全注意力，**不**按 SW 截断 context。  
   据此得到每层 `L_ctx` / `L_kv`（见 3.1.5），不得写死「全部层都是 sliding」。
5. **CLI**：必须以显式 `--speculative-method dflash` 启用（禁止仅凭 `--num-speculative-tokens >= 2` 隐式启用）；  
   `--num-speculative-tokens` 为 Dflash 从属；`--num-draft-layers` / `--draft-model-config-path` 为共享从属（`--speculative-method`）；  
   **`--acceptance-length` 仅存在于 `throughput_optimizer`**，供 Decode  
   `decode_latency = record.latency_ms / (dflash_acceptance_length + 1)` 折算（不参与构图 / 不驱动 KV CF；`text_generate` 无此 flag）。
6. Prefill / Decode 整图均为：`target 完整前向 + draft(block)`。
7. 建模路径 aux：`output_intermediate_hidden_states=True` 后按 `aux_hidden_state_layer_ids` **长度合成** clones（非按层 gather）；ids 只定长度/校验。
8. **draft 自有 Linear 不量化**（`quantize_model` 跳过 `draft.*`）。

本方案保证与真实 Dflash Draft **计算量 / 显存足迹**一致；不追求数值正确；不建模 DynamicCache / acceptance 驱动的 KV 裁剪回滚。

---

## 1.2 Motivation

- MTP 复用主模型 decoder 类，隐含同族；无法表达「target 为 Deepseek/Kimi（MLA）、draft 为 Qwen3 DFlash（GQA）」的真实算量子图。
- 若 draft Attention 使用 transformers `eager_attention_forward`，则 **不会发出** `torch.ops.tensor_cast.attention*`，统计上只剩 target（如 MLA）次数——无法满足核心需求。
- 若对 draft 误跑 target 的 `patch_mla`，会把 Qwen3 GQA 错改成 MLA，次数与时间均错误。
- 双模式（外部 config / 复用末层）语义冲突，维护成本高；完整外部 checkpoint config 对仿真过重。

---

## 1.3 Goals

### 目标

- 提供单一 `DflashConfig` + 内置 draft profile。
- Draft 按 **构图公式** 构建；**Prefill 与 Decode** 均为 target 完整前向 + draft 前向。
- **Draft 侧 Attention / Linear 进入 TC 算子图**，可计次、可估时；**draft Linear 不量化**。
- Forward 见 3.1.3：target intermediate → 按 `L_aux` 合成 aux → `cat` → `fc+hidden_norm` → **一次** `context_kv_proj` → 逐层 `k_norm` → **一次**融合 context `apply_rope` → `reshape_and_cache`×N；Decoder 仅对 block 长 noise 做 QKV/`apply_rope`/cache/Attn。
- transformation：跳过 draft 上的 target MLA/MoE **与 draft Linear 量化**；按草稿 GQA heads shard（含 `draft.context_kv_proj`）；cache 草稿 rotary。
- 吞吐优化：`dflash_acceptance_length`（**仅** `throughput_optimizer` CLI）参与 Decode 延迟折算（见 3.1.10）；Prefill **跑 draft 但不折算**。
- CLI：必须显式 `--speculative-method dflash` 才启用；`--num-speculative-tokens` 为 Dflash 从属；`--num-draft-layers` / `--draft-model-config-path` 为共享从属（`--speculative-method`）。

### 非目标

- 不实现真实数值推理、采样正确性、acceptance 控制流。
- 不加载草稿权重；不做 DynamicCache / crop / KV 回滚。
- 不保留「复用主模型末层 DecoderLayer」路径。
- **不**把 draft 当作「第二个完整 TransformerModel(target)」再跑一遍 MLA/MoE 导入。

---

# 2. 用例分析

## 2.1 Dflash 推理性能仿真

**功能点**

- CLI 以 `--speculative-method dflash` 启用 Dflash；结构来自内置 profile + target 维度；Dflash-only 从属须在 `--speculative-method dflash` 后；共享 draft 从属须在 `--speculative-method` 后。
- Decode：主模型前向（verify）+ 草稿并行 block；按接受长度折算（吞吐 CLI）。
- Prefill：主模型前向 + 草稿前向（aux 按 intermediate 合成），用于 TTFT；acceptance **不**折算。

**性能指标**

- 仿真延迟 / 吞吐相对实测误差。
- **Decode 一步算子账本**（见 3.1.8）：draft `attention*` 次数 = `num_draft_layers`；verify 侧仍为 target 原路径次数。

**DFX**

- 与 MTP 互斥；默认关闭。
- 逻辑集中：`dflash.py`、`dflash_qwen3.py`、内置 profile、`maybe_enable_dflash`、`patch_attention` / `shard_model` 对 draft 的注册与 TP。
- 单测：TC attention、融合 RoPE/`reshape_and_cache`、`L_ctx`/`L_kv`、共享 embed、禁止 MLA 碰 draft、折算。

---

# 3. 方案设计

## 3.1 总体方案

### 3.1.0 分层与挂接顺序

```text
配置层     DflashConfig + 内置 Qwen3Config profile
构建层     Qwen3DFlash 层 + context_kv_proj + set_shared(embed, lm_head)
注册层     draft _attn_implementation=tensor_cast；attention_by_layers 覆盖 draft layer_idx
Wrapper    Prefill/Decode: target → collect aux → draft(block)
```

挂接顺序：

```text
wrap_model
maybe_enable_mtp                 # 与 dflash 互斥
maybe_enable_dflash              # 构图 Draft（Qwen3 骨架 + KV 适配 + 共享 vocab）
maybe_reuse_layers               # 可选：同构 draft.layers
patch_model
patch_rotary_emb                 # cache 草稿自持 rotary（不共享 target RoPE）
patch_attention                  # 为 draft 层注册 AttentionTensorCast 槽位
patch_mla / patch_moe            # 必须跳过 draft.*（不复用 target MLA 导入）
quantize_model                   # 跳过 draft.* Linear（draft 不量化）
shard_model                      # draft 按 Qwen3 GQA heads；共享 lm_head 不双包
```

**明确禁止**：对 `draft.*` 调用与 target 相同的 `patch_mla` / MoE 替换逻辑。

### 3.1.1 Draft 构图公式

| 组成部分 | 含义 | 复用现有能力 |
|----------|------|----------------|
| 内置 `Qwen3Config` | block、N、heads、`intermediate_size`、SW、`layer_types`、`target_layer_ids` 等 | 配置解析 / `Qwen3Config(**dict)` |
| 标准 Qwen3 层骨架 | `q/k/v/o_proj`、`q_norm`/`k_norm`、`Qwen3MLP`、Pre-LN RMSNorm、`Qwen3RotaryEmbedding` | 与「Qwen3 作主模型」相同的模块类（`from_config` 或等价手建） |
| TC attention 注册 | `config._attn_implementation = "tensor_cast"`；`attention_by_layers[draft_layer_idx] = AttentionTensorCast()`；最终 `torch.ops.tensor_cast.attention` / `attention_quant` | `ALL_ATTENTION_FUNCTIONS["tensor_cast"]`、`flash_attention_forward`、`AttentionTensorCast`、performance model |
| DFlash KV / cache 适配 | 见 3.1.3：model 级写 context cache；层内仅 noise + TC attention | `context_kv_proj`、`apply_rope`、`reshape_and_cache`、`AttentionTensorCast` |
| 不复用 target MLA | target 可为 Kimi/Deepseek MLA；draft 始终 GQA | `patch_mla` 跳过 `draft.*`；`ensure_draft_kv_caches` 在 MLA target 下为 draft 层补 GQA cache |

落地形态（已实现）：

- `DflashDraftModel`：`fc` / `hidden_norm` / **`context_kv_proj`** / `layers`（`DflashDraftLayer`→`Qwen3DFlashDecoderLayer`）/ 自持 `rotary_emb`；`set_shared(embed, lm_head)`。
- `Qwen3DFlashAttention`：短 noise QKV + `tensor_cast.apply_rope` +（有 cache 时）仅 noise 写 cache + TC attention；无 cache 单测回退为 `cat(ctx, noise)`。
- `maybe_reuse_layers` 后非首层为 `CopyLayerWrapper`：context 路径通过 init 期缓存的 `_draft_k_norms` / `_draft_attn_layer_indices` 取到各层 `k_norm`。

### 3.1.2 内置草稿配置与 CLI 覆盖

**原则**：结构尺寸以内置 profile（或可选外部 config）为准；维度与词表跟 target 对齐；**CLI 显式参数可覆盖 config 中对应字段**（后写覆盖先写）。

| 类别 | 字段 | 取值来源 |
|------|------|----------|
| 本方案约定 | `model_type` | `"qwen3"` |
| | `block_size` | `8`（有效投机 `block_size-1`）；可被 `--num-speculative-tokens` 覆盖 |
| | `attention_bias` | `false` |
| | `hidden_act` | `"silu"` |
| | `use_sliding_window` | 与 `layer_types` / `sliding_window` 一致即可 |
| | `sliding_window` | 缺省 **2048**（仅对 `sliding_attention` 层生效） |
| | `layer_types` | **逐层** `"sliding_attention"` 或 `"full_attention"`（长度 = N）；决定该层是否启用 SW（见 3.1.5）。Builtin 可全 `full_attention` 或全 SW，**以实际 config 为准，不在代码里写死** |
| 无通用默认，由 builtin draft profile 给出 | `head_dim` | `128` |
| | `num_attention_heads` | `64` |
| | `num_key_value_heads` | `8` |
| | `intermediate_size` | `18432` |
| | `num_hidden_layers`（draft N） | `6`；可被 `--num-draft-layers` 覆盖（覆盖后需同步 `layer_types` 长度：截断或按末项填充） |
| | `aux_hidden_state_layer_ids` / `target_layer_ids` | 如 `[1, 12, 24, 35, 47, 58]` |
| | `rms_norm_eps` 等 | 与 builtin draft profile 一致 |
| **运行时从 target 覆盖** | `hidden_size` / `vocab_size` / `max_position_embeddings` | `text_config` |
| **吞吐折算（非结构）** | `dflash_acceptance_length` | **仅** `throughput_optimizer --acceptance-length`（默认 5，上界 clamp 到 `block_size-1`；`<0` 报错）；写入 `DflashConfig`；`text_generate` 无此 flag |

**CLI 覆盖优先级（结构字段）：**

```text
target 维度覆盖（H/V/max_pos）
  ↑ 最高（一致性强制）
CLI 显式参数（--num-speculative-tokens / --num-draft-layers / …）
  ↑ 覆盖 config 同名字段
builtin 或 --draft-model-config-path 加载的 config
  ↑ 基底
```

| CLI | 覆盖的 config / 语义 |
|-----|----------------------|
| `--speculative-method` | **启用开关**（`dflash`/`dspark`）；未传则禁止 draft 从属参数 |
| `--num-speculative-tokens`（≥2） | **Dflash 从属**：覆盖 `block_size`；**不**单独启用 Dflash |
| `--acceptance-length` | **从属（仅吞吐 CLI）**：**不覆盖**层结构；写入 config 供 Decode 折算 |
| `--num-draft-layers` | **共享从属**：覆盖 `num_hidden_layers`（draft N）；需 `--speculative-method` |
| `--draft-model-config-path` | **共享从属**：可选外部 config 替换 builtin；需 `--speculative-method` |

校验：`max(target_layer_ids) < target.num_hidden_layers`；`H`/`V` 与 target 一致；`len(layer_types) == num_hidden_layers`（覆盖 N 后必须成立）。

构图前设置：

```python
draft_hf_config._attn_implementation = "tensor_cast"  # 进入 TC attention，禁止默认 eager
```

### 3.1.3 DFlash Forward 契约（必须满足）

与当前落地实现（`dflash.py` / `dflash_qwen3.py`）一致：

```text
① 主模型完整前向；`output_intermediate_hidden_states=True`（与 MTP 同模式；**不用** forward hook）
     → 默认拿到 intermediate/last_hidden，再按 `aux_hidden_state_layer_ids` **长度**合成 L_aux 路 clones
       （`_synthesize_modeling_aux_hiddens`：ids 只定长度/校验，**不做** per-layer gather）
     → 若返回值已是 aux list，则走兼容分支并经一次 `as_bsh` 归一化
② 构建 target_context：tensor_cast.cat 对齐后的各路 aux（perf 映射 NPU ConcatD）
     （Prefill 返回 `(target_out, draft_logits)` 防 DCE；runner 解包 primary）
     → Prefill: [B, L_ctx_fc, L_aux·H]  # L_ctx_fc = max(L_ctx[*])；pad/truncate 到 L_ctx
     → Decode:  [B, block, L_aux·H]     # 不对满长 context 做 cat/fc；长上下文只在 KV cache
③ fc + hidden_norm → target_hidden              # Prefill: [B,L_ctx_fc,H]；Decode: [B,block,H]
④ context_kv_proj(target_hidden) → 一次 MatMul   # 见 3.1.3.1；拆成每层 (K_ctx, V_ctx)
⑤ noise = embed([anchor|MASK…])；再 context KV write（model 级，Decoder 之前）：
     每层：k_norm(K_ctx)                         # head_dim RMSNorm；权重=该层 self_attn.k_norm
     Tile/cat 各层 K → 一次 tensor_cast.apply_rope  # Prefill=满长；Decode=block 写长
     每层：reshape_and_cache(K_roped, V_ctx)      # 写入该层 draft GQA paged KV cache
⑥ 每层 DecoderLayer（Pre-LN；序列长 = block）：
     Q  ← q_norm(q_proj(noise))
     K_n← k_norm(k_proj(noise))；V_n ← v_proj(noise)
     apply_rope(Q, K_n)                          # 仅 block 长；不对 context K 再 RoPE
     reshape_and_cache(K_n, V_n)                 # 追加 noise 段到同一 cache
     Attn ← TC attention(Q, KV=cache)            # is_causal=False；seq_lens=L_ctx+block
                                                 # Decode 即使 fc 为 block，Attn 仍用配置 L_ctx
                                                 # sliding_window 仅当 layer_types[i]==sliding_attention
     再 o_proj → residual → MLP
⑦ norm → target.lm_head → DFlash 取 block 内 block-1 位置；DSpark 保留满 block
⑧ Prefill：返回 `(target_out, draft_logits)`；Decode：先 target sampler，再 draft propose
```

#### 3.1.3.1 `context_kv_proj` 与 draft KV cache 构建

**一次 fused Linear，同时产出全部 Draft 层的 context K 与 V**（本地宽约 `N×2×head_dim` 量级，随 TP 变化）：

```text
context_kv_proj: Linear(H → num_kv_heads × N_draft × 2 × head_dim, bias=False)
                 # 布局 head-major：[num_kv_heads, N_draft, 2, head_dim]
                 # 「×2」= K 与 V，不是「KV head 数再乘 2」
TP：ColumnParallel(head_num=num_kv_heads)，前缀 draft.context_kv_proj；
    本地 out 与各层 k_proj/v_proj 宽度一致（可 cat / 同 cache head 维）
```

拆分后按 `layer_types[i]` 截断到该层 `L_ctx[i]`（见 3.1.5）。

**Paged KV cache（GQA）**：

| 步骤 | 行为 | 实现要点 |
|------|------|----------|
| 分配 | 每层一份 GQA cache | `ensure_draft_kv_caches`；shape `(2, num_blocks, page_size, local_kv_heads, head_dim)`；优先保留 `input_generator` 已按 TP 分好的 cache（MLA target 时 draft 层需单独 GQA 槽） |
| Context 写 | Decoder **之前**一次性 ×N | `_fuse_context_rope_and_cache`：`k_norm` → 融合 `apply_rope` → `reshape_and_cache`；slot=`[0 .. L_ctx)` |
| Noise 写 | 每层 Decoder 内 | `AttentionTensorCast` 路径再调 `reshape_and_cache`；slot=`[L_ctx .. L_ctx+block)` |
| Attn 读 | FIA / TC attention | `seq_lens = L_ctx+block`，`query_lens = block`；context K 已在 cache 中且已 RoPE |

`reshape_and_cache` 为 **inplace**（`mutates_args=("kv_cache",)`，返回 `None`）；chrome-trace 的 Output 为 `None` 属预期，shape 看 Inputs。

**禁止**的错误融合（旧草案 / 已废弃代理）：

```text
# 旧 RFC：每层 new_empty 代理 context + Attention 内 cat(ctx, noise)
k_ctx_proxy = new_empty(L_ctx, ...)   # 不对满长做投影
K = cat(k_ctx_proxy, k_proj(noise))   # 无 paged cache / 无融合 context RoPE
```

当前实现：**有** `context_kv_proj` + paged cache；无 cache 的单测路径才回退 `cat(ctx, noise)`。

**正确语义**：context 段在 model 级写入 draft KV cache；Decoder Attention 只追加 noise 并从 cache 读满长 KV——**不是**对 `noise` 与 `target_hidden` 做 hidden 维拼接后再算 Attention。

### 3.1.4 模块清单（对齐落地实现）

| 模块 | 形状 / 归属 | 作用 |
|------|-------------|------|
| `fc` | `Linear(L_aux·H → H)`，草稿自有（**不量化**） | 融合建模 aux（默认 intermediate 合成 clones） |
| `hidden_norm` / `norm` | `Qwen3RMSNorm`，草稿自有 | fc 后 / 最终归一化（compile 可 fuse 为 `tensor_cast.rms_norm`） |
| `context_kv_proj` | `Linear(H → num_kv_heads·N·2·head_dim)`，草稿自有 | **一次**产出全部层 context K/V；TP head-major |
| `layers` × N | `DflashDraftLayer` + `Qwen3DFlash*` | 并行 block 预测；层内仅短 noise |
| `_draft_k_norms` | init 期对各层 `self_attn.k_norm` 的引用列表 | `reuse_layers`/`CopyLayerWrapper` 后仍可做逐层 context `k_norm` |
| `rotary_emb` | Qwen3 RoPE → `CachingRotaryEmb`，**草稿自持** | noise / 融合 context 查 cos·sin；**不**共享 target RoPE |
| `embed_tokens` / `lm_head` | **引用 target** | 与 target 共享词表模块 |
| draft KV cache | 每层 GQA paged：`kv_cache_by_layers[layer_idx]`；context 预写 + noise 追加 | `ensure_draft_kv_caches` + `reshape_and_cache`；不计 DynamicCache CF |

### 3.1.5 `layer_types` 与 `L_ctx` / `L_kv` 契约（算量真源）

是否启用 sliding window 由 **`layer_types[layer_idx]`** 决定，而非全局写死。

```python
# 与 3.1.5 契约一致
self.sliding_window = (
    config.sliding_window  # 缺省 2048
    if config.layer_types[layer_idx] == "sliding_attention"
    else None  # full_attention：不按 SW 截断
)
```

TensorCast 不做 draft KV cache CF；用下列规则代理每层可见 context 长度：

```text
block = dflash_block_size
q_len = block

# 逐层：
if layer_types[i] == "sliding_attention":
    L_ctx[i] = min(context_length, sliding_window)   # sliding_window 默认 2048
else:  # "full_attention"
    L_ctx[i] = context_length                        # 全上下文（建模口径）

L_kv[i] = L_ctx[i] + block
```

说明：

- **Prefill**：`fc` / `hidden_norm` / `context_kv_proj` 的 token 维用 `max(L_ctx[*])` 构图，再按层 slice 到 `L_ctx[i]`。
- **Decode**：`fc` / `hidden_norm` / `context_kv_proj` 的 token 维为 `block`（aux cat 同为 block）；**不**把短 aux pad 到满长 `L_ctx`。
- 每个 draft 层 TC `attention*`：`q_len = block`；KV 来自 paged cache，有效长度 `L_kv[i]=L_ctx[i]+block`（Decode 用配置的 `L_ctx`，与本次 fc 写长无关）；`sliding_window` 仅在 SW 层传入 TC attention。
- Context 段 **不再**在 Attention 内用 `new_empty`+`cat` 代理（有 cache 的主路径）；算量上 context 写 cache + Noise Attn 与「Q=block、KV=L_ctx+block」一致。
- Builtin / 外部 config 中的 `layer_types` 原样生效（可为全 `full_attention` 或混排）。
- **`dflash_acceptance_length` 不改变 `L_ctx` / 图结构**；只用于 3.1.10 吞吐量折算。

### 3.1.6 Prefill / Decode

| 阶段 | 内容 |
|------|------|
| Prefill / TTFT | **主模型完整前向 + draft 前向**；aux cat/fc 对齐到 `L_ctx`（`context_length`） |
| Decode 每步 | **主模型完整前向（verify）+ draft 前向**；aux cat/fc 对齐到 `block`；长上下文仅体现在 Attention KVCache（`seq_lens=L_ctx+block`） |

输出固定 shape；不做真实接受判定 / KV crop。

### 3.1.7 `DflashWrapper.forward` 伪代码

```python
block = dflash_block_size
# Prefill: aux_seq = max L_ctx；Decode: aux_seq = block（Attn 仍用配置 L_ctx）

# 1) 主模型前向（MTP 式 intermediate）+ 按 aux_ids 长度合成 L_aux clones
target_out, aux_hiddens = _run_target_collect_aux(
    ..., output_intermediate_hidden_states=True
)

# 2) ConcatD(aux) → noise embedding → context KV → decoder
#    （与 dflash.py forward 一致；勿把 prepare_context_kv 放到 embed 之前）
aux_seq = block if is_decode else max_L_ctx
target_context = draft.build_context_features(aux_hiddens, aux_seq, align_seq=not is_decode)
next_tokens = sampler(target_out, sampling_metadata)  # Prefill/Decode 均可先 sample
noise = draft.embed_tokens(...)                      # 共享 target.embed；[B, block, H]
pos_emb = draft.rotary_emb(noise, block_position_ids)
prepared = draft.prepare_context_kv(
    target_context, batch=B, block=block,
    attn_use_configured_context=is_decode, **tc_attn_kwargs,
)
draft_hidden = draft.run_noise_decoder(
    noise, block_position_ids, prepared,
    position_embeddings=pos_emb, **tc_attn_kwargs,
)
# prepare_context_kv：
#   target_hidden = hidden_norm(fc(target_context))
#   split context_kv_proj → per-layer (K,V)
#   ensure_draft_kv_caches(...)
#   _fuse_context_rope_and_cache(...)   # k_norm×N + apply_rope×1 + reshape_and_cache×N
# run_noise_decoder：
#   for layer: noise-only attn (短 apply_rope + cache + TC attention)

draft_logits = draft.lm_head(draft_hidden[:, 1:, :])  # DFlash 排除 anchor；DSpark 保留满 block
# Prefill: return (target_out, draft_logits)  # runner 取 primary
# Decode DFlash: return (draft_tokens, next_tokens)
# Decode DSpark: return (draft_tokens, next_tokens[, confidence])
```

`tc_attn_kwargs` 必须传入 `attention_by_layers`、`kv_cache_by_layers`（及 generator 侧 meta）；draft 用 `build_draft_attention_metadata` 生成 context/noise 两套 `slot_mapping`。`layer_idx` 覆盖全部 draft 层（含 target offset）。

### 3.1.8 算子账本（核心需求验收）

Decode **一步**期望（概念次数；TP/融合不改变「每层一次 attention 核心 op」的建模口径）：

| 区域 | 算子（示例） | 次数量级 |
|------|--------------|----------|
| Draft 准备 | `fc`（`L_aux·H → H`） | 1 |
| Draft 准备 | `hidden_norm` / `rms_norm` | 1（整段 `target_hidden`） |
| Draft 准备 | `context_kv_proj` MatMul | **1**（Prefill=`L_ctx_fc`；Decode=`block`；非每层满长 proj） |
| Draft 准备 | context `k_norm`（`rms_norm`） | **= N** |
| Draft 准备 | `tensor_cast.apply_rope`（context 融合） | **1**（序列维 ≈ Σ `L_ctx[i]` 或 `N·L_ctx`） |
| Draft 准备 | `reshape_and_cache`（context） | **= N**（Output=`None`，inplace） |
| Draft Decoder | `tensor_cast.apply_rope`（noise） | **= N**（序列长 = `block`） |
| Draft Decoder | `reshape_and_cache`（noise） | **= N** |
| Draft Decoder | `tensor_cast.attention` / `attention_quant` | **= N** |
| Draft Decoder | `q/k/v/o_proj`、MLP Linear | 每层一组（序列长 = `block`） |
| Draft | 共享 `embed` / `lm_head` | 各至少 1；**lm_head 序列维 = `block-1`**（先 Index 去 anchor） |
| Target verify | 原主模型 Attention/MLA 路径 | **= target 有效层数** |
| Target verify ArgMax | `Sampler`：bonus `[1,V]→[1]` + specs `[S,V]→[S]`（`S=block-1`） | **2**（与 NPU 拆分一致；非一次 `[S+1,V]`） |
| Draft ArgMax | `argmax(draft_logits)`（已是 `[B,S,V]`，`S=block-1`） | **1**（`[S,V]→[S]`，供下一轮 verify） |

验收失败信号：

- Trace / op-bound 中 **只有** target MLA/Attention、**没有** draft 的 `tensor_cast.attention*` → 仍走 eager，未完成 TC 注册。
- Draft 出现 `multihead_latent_attention*` → 误用了 target MLA 导入。
- 每层仍对满长 context 做 `k_proj`/`apply_rope`（N 次全长 RoPE）→ 未对齐 one-shot `context_kv_proj` + 融合 context RoPE。

### 3.1.9 Transformation 规则

| Pass | 行为 |
|------|------|
| `patch_attention` | 为 draft 各 `layer_idx` 注册 `AttentionTensorCast`；与 target 槽位一并或分段编号，但必须可被 draft 取到 |
| `patch_rotary_emb` | cache **草稿** rotary |
| `patch_mla` / `patch_moe` | **跳过 `draft.*`** |
| `quantize_model` | **跳过 `draft.*` Linear**（draft 不量化）；仅量化 target；共享 vocab 模块不冲突双量化 |
| `shard_model` | draft 按 **Qwen3** heads；含 `draft.context_kv_proj`（`head_num=num_key_value_heads`）与 `draft.layers.*.dflash_block.self_attn.{q,k,v,o}_proj` / MLP；共享 lm_head 不双包 |

VL（如 Kimi）：target 层解析用 `language_model.model.layers`；维度取 `text_config`。Draft **仍是 Qwen3 GQA**，与 target 是否 VL/MLA 无关。

### 3.1.10 性能折算与吞吐量优化（`dflash_acceptance_length`）

`dflash_acceptance_length` **不是**构图参数，而是 **吞吐优化专用标量**（与 MTP 的 acceptance 折算同角色）：

- 来源：**仅** `throughput_optimizer --acceptance-length`；未指定时默认 `5.0`；`DflashConfig` 上界 clamp 到 `block_size-1`，`<0` 抛错（`text_generate` 无此 CLI，启用时用 config 默认）。
- 消费方：`serving_cast`（如 `base_throughput_optimizer._fold_decode_latency_ms`）在 Dflash decode 场景下：

```python
average_tokens = dflash_acceptance_length + 1  # 不含 anchor；已 clamp
decode_latency = record.latency_ms / average_tokens
```

- `record.latency_ms` 已含 **target + draft（含 TC attention）** 满算子图；禁止再把 draft/verify 拆开相加。
- Prefill/TTFT **不**使用该折算；Prefill 仿真图**含** draft（与 Decode 同为 target+draft）。
- 该参数 **不**改变 `layer_types` / `L_ctx` / KV 图结构，也 **不**驱动 acceptance 控制流或 KV crop。

---

## 3.2 技术选型

| 选项 | 结论 |
|------|------|
| 单一 Qwen3 DFlash + 内置 profile + TC attention + `context_kv_proj` / paged KV | **选定（已落地）** |
| `layer_types` 逐层决定 SW / full | **选定**（见 3.1.5） |
| CLI 覆盖 `block_size` / `num_draft_layers`；`acceptance_length` 专用于吞吐折算 | **选定** |
| 复用 target MLA 导入到 draft | **否** |
| draft 使用 `eager_attention_forward` | **否**（无法满足计次/计时） |
| 写死全部层 sliding / 写死全部层 full | **否**（以 config `layer_types` 为准） |
| 双模式 Mode A/B | **否** |
| 强制外部完整 draft config | **否**（可选 override） |
| 完整二次 `TransformerModel(draft_id)` | **否**（会卷入无关 wrap/MLA） |

---

## 3.4 编程与调用设计

### 3.4.1 约束

- Target：可解析 decoder `layers` 的 decoder-only 或 VL 文本塔。
- Draft：`hidden_size` / `vocab_size` 与 target 一致；`_attn_implementation="tensor_cast"`。
- DecoderLayer.forward **必须**接收并消费 `target_hidden`（见 3.1.3）。

### 3.4.2 核心类型（示意）

```python
@dataclasses.dataclass
class DflashConfig:
    dflash_block_size: int = 8
    """来自 config.block_size；可被 --num-speculative-tokens 覆盖（须先 --speculative-method dflash）。"""
    num_draft_layers: int = 6
    """来自 config.num_hidden_layers；可被 --num-draft-layers 覆盖。"""
    dflash_acceptance_length: float = 5.0
    """吞吐优化折算标量（默认 5；上界 clamp 到 block-1；<0 报错）；不参与构图。"""
    sliding_window: Optional[int] = 2048
    """仅对 layer_types[i]==sliding_attention 的层生效；full_attention 层忽略。"""
    aux_hidden_state_layer_ids: Optional[List[int]] = None
    draft_model_config_path: Optional[str] = None  # 可选：替换 builtin 基底后再套 CLI 覆盖

# Draft = builtin Qwen3Config + Qwen3DFlash 骨架 + TC attn + context_kv_proj / paged cache
class DflashDraftModel(nn.Module):
    def set_shared(self, embed_tokens, lm_head): ...
    def build_context_features(self, ref_hidden, l_ctx): ...
    def _split_context_kv(self, target_hidden): ...
    def _fuse_context_rope_and_cache(self, context_kv_by_layer, ...): ...
    def forward(self, noise_embedding, target_context, position_ids, position_embeddings=None, **kwargs):
        target_hidden = self.hidden_norm(self.fc(target_context))
        context_kv_by_layer = self._split_context_kv(target_hidden)
        # ensure GQA caches; fuse context k_norm + apply_rope + reshape_and_cache × N
        context_kv_roped = self._fuse_context_rope_and_cache(...)
        for i, layer in enumerate(self.layers):
            noise_embedding = layer(
                noise_embedding,
                target_hidden=...,
                context_kv=context_kv_roped[i],
                draft_noise_attention_meta=...,
                **kwargs,
            )
        return self.norm(noise_embedding)

class DflashWrapper(ModelWrapperBase):
    # decode: build_context_features → draft → verify；query_length 可抬升到 block
```

```python
def maybe_enable_dflash(model):
    if not model.model_config.dflash_config:
        return model
    if model.model_config.mtp_config:
        raise ValueError("Dflash and MTP are mutually exclusive")
    dcfg = model.model_config.dflash_config
    draft_hf = load_builtin_draft_qwen3_config(dcfg)  # merge target H/V/max_pos
    draft_hf._attn_implementation = "tensor_cast"
    draft = build_dflash_draft_and_wrapper(...)  # DflashDraftModel + DflashWrapper
    draft.set_shared(*resolve_target_embed_and_lm_head(model))
    model._inner = DflashWrapper(dcfg, draft, model._inner, ...)
    return model
    # 随后 patch_attention 注册 draft 槽位；patch_mla 跳过 draft.*；
    # patch_rotary_emb cache draft.rotary_emb；shard 含 context_kv_proj
```

### 3.4.3 CLI

| 参数 | 类型 | 默认 | 入口 | 说明 |
|------|------|------|------|------|
| `--speculative-method` | choice | None | text_generate / throughput_optimizer | `dflash` / `dspark`；与 MTP 互斥 |
| `--num-speculative-tokens` | int | 0 | 同上 | 投机 token 数（不含 anchor）；`n>=1` → `block=n+1`；须先 `--speculative-method` |
| `--acceptance-length` | float | 5 | **仅** throughput_optimizer | Decode 折算标量；须先 `--speculative-method`；dflash 上界 `B-1` |
| `--num-draft-layers` | int | config N | 同上 | **共享从属**：须先 `--speculative-method` |
| `--draft-model-config-path` | str | None | 同上 | **共享从属**：须先 `--speculative-method` |

约束：

- **启用条件**：`speculative_method == "dflash"`；**禁止**仅凭 `--num-speculative-tokens` 隐式启用。
- **G3 依赖**（`validate_draft_spec_cli_args`）：  
  - 无 `--speculative-method` 却传 `--num-speculative-tokens`（及吞吐侧 `--acceptance-length`）→ fail-fast；  
  - 无 `--speculative-method` 却传 `--num-draft-layers` / `--draft-model-config-path` → fail-fast。
- 与 MTP / DSpark 互斥。
- Decode：`--query-length >= dflash_block_size`（不足可抬升到 block）。
- `--context-length` 参与各层 `L_ctx`（SW 层再与 `sliding_window` 取 min；full 层用满 `context_length`）。
- `layer_types` **只来自 config**（builtin 或 path）；本阶段不提供单独 CLI 改写逐层类型（若未来需要另开参数）。

```bash
python -m cli.inference.text_generate moonshotai/Kimi-K2.6 \
  --device ATLAS_800_A3_752T_128G_DIE \
  --num-devices 16 --tp-size 16 --dp-size 1 \
  --num-queries 2 --query-length 8 --context-length 1024 \
  --decode --speculative-method dflash --num-speculative-tokens 7 \
  --quantize-linear-action W4A8_DYNAMIC
```

---

# 4. 测试设计

| 用例 | 验证要点 |
|------|----------|
| `test_dflash_builtin_config` | 无外部 path 可构建；`layer_types` 长度 = N |
| `test_dflash_layer_types_sw_vs_full` | `sliding_attention` 层 `L_ctx=min(ctx,SW)`；`full_attention` 层 `L_ctx=ctx` |
| `test_dflash_cli_override_block_and_layers` | `--speculative-method dflash` + `--num-speculative-tokens` / `--num-draft-layers` 覆盖 config |
| `test_dflash_cli_dependent_args_require_method` | 无 `--speculative-method` 传从属参数 → fail-fast |
| `test_dflash_tc_attention_ops` | draft 路径出现 `tensor_cast.attention*`；次数 = N |
| `test_dflash_no_mla_on_draft` | `draft.*` 无 MLA / 无 `multihead_latent_attention*` |
| `test_dflash_kv_inject_shapes` | Q=`block`；有 cache 时 Attn 读 paged KV（`L_ctx+block`）；`context_kv_proj` 一次满长 |
| `test_dflash_shared_embed_lm_head` | 与 target 同模块引用 |
| `test_dflash_forward_contract` | 符合 3.1.3；融合 context RoPE×1 + noise RoPE×N；禁止 hidden 维 fused 旧路径 |
| `test_dflash_apply_rope_and_reshape_and_cache` | 发出 `apply_rope` / `reshape_and_cache`；context 融合与 noise 短序列可区分 |
| `test_dflash_acceptance_throughput_fold` | `dflash_acceptance_length` 仅折算；optimizer `/ (accept+1)` |
| `test_dflash_mtp_mutex` | 互斥 |
| `test_dflash_vl_layer_resolve` | Kimi 等嵌套 layers 可解析 |

E2E：`text_generate` + Dflash 时，op-bound / trace 同时含 draft GQA attention 与 target verify（MLA）。

---

# 5. Drawbacks and Risks

1. **内置 profile 漂移**：换真实 draft checkpoint 需更新 builtin。→ 可选 path override。  
2. **KV / RoPE 路径演进**：旧「`new_empty` proxy + Attention 内 cat」已废弃；落地为 `context_kv_proj` + 融合 context RoPE + paged cache。SW/full 混排时各层 `L_ctx` 不同，fc 侧用 `max(L_ctx)` 再 slice。→ 以 3.1.3 / 3.1.5 为准；变更另开 RFC。  
3. **KV 适配与 TC 出口脱节**：只构图不注册 / 仍调 eager → 计次失败。→ 单测 `test_dflash_tc_attention_ops` 阻断。  
4. **误 patch MLA**：跨族 target 时最大风险；且 MLA cache 形状与 draft GQA 不同。→ `patch_mla` 硬跳过 `draft.*`；`ensure_draft_kv_caches` / `input_generator` 为 draft 层保留 GQA cache。  
5. **共享 lm_head 的 TP/量化**：需识别同一 module，避免双包。  
6. **CLI 覆盖 N 与 `layer_types` 长度不一致**：覆盖 `--num-draft-layers` 后必须同步 `layer_types`。→ `__post_init__` / build 时校验或按末项填充。  
7. **`maybe_reuse_layers` + CopyLayerWrapper**：非代表层无 `dflash_block` 子树；context `k_norm` 依赖 init 期 `_draft_k_norms`。→ 勿在 reuse 后再从 `layers[i].dflash_block` 取模块。  

---

# 6. Existing Technology

- 实现：`tensor_cast/layers/dflash.py`、`dflash_qwen3.py`；`ops/attention.py`（`reshape_and_cache` inplace）
- TensorCast：`AttentionTensorCast`、`CachingRotaryEmb`、`torch.ops.tensor_cast.attention*` / `apply_rope`、quant/shard、MTP 整图折算模板
- Qwen3 HF：`Qwen3Config` / `Qwen3MLP` / `Qwen3RotaryEmbedding` / `Qwen3RMSNorm`
- VL：`language_layers_path_str`（如 Kimi `language_model.model.layers`）

---

## Appendix

### Glossary

| 术语 | 描述 |
|------|------|
| Dflash | 并行 block 投机；`DflashWrapper` 挂接 |
| Draft 构图公式 | 内置 Qwen3Config + Qwen3DFlash 骨架 + TC attention + `context_kv_proj` / paged KV；不复用 target MLA |
| `target_hidden` | aux 经 `fc+hidden_norm` 的 context，供 `context_kv_proj` |
| `context_kv_proj` | 一次 Linear 产出全部层 context K/V；`out = num_kv_heads·N·2·head_dim`（×2=K/V） |
| `noise_embedding` | `embed([anchor\|MASK…])`，供 Q 与 noise 段 K/V（长度=`block`） |
| draft KV cache | 每层 GQA paged；context 预写（融合 RoPE 后）+ noise 追加；Attn 从 cache 读 |
| `layer_types` | 逐层 `"sliding_attention"` / `"full_attention"`；决定是否启用 `sliding_window` |
| `L_ctx` / `L_kv` | SW 层：`min(context_length, sliding_window)`；full：`context_length`；`L_kv=L_ctx+block`（见 3.1.5） |
| `dflash_acceptance_length` | 吞吐量优化折算标量（不含 anchor）；不参与构图 |

### Documentation Update Plan

- 落地后更新中英文 user guide、CLI `--help`、web_ui。
- 旧双模式 RFC 标为 superseded，指向本文档。
- **以本文 3.1.3 落地契约为唯一 Forward / KVCache 规范**；旧「Attention 内 cat proxy」路径废弃。

### 与旧双模式草案 / 错误路径

| 旧概念 / 错误路径 | 本 RFC |
|-------------------|--------|
| Mode A / Mode B | 删除分裂；单一路径 |
| 外部完整 draft config 必填 | 内置 profile + 可选 override |
| 复用 target 末层 | 删除 |
| hidden 维 cat+proj 融合 | 禁止 |
| 每层 `new_empty` context proxy + Attention 内 `cat(ctx,noise)` | **改为** `context_kv_proj` + paged `reshape_and_cache`（无 cache 单测才 cat） |
| 每层对满长 context 做 `k_proj` / RoPE | **改为** 一次 `context_kv_proj` + 一次融合 context `apply_rope` |
| `eager_attention_forward` | 改为 **TC `attention*`** |
| 对 draft 跑 `patch_mla` | **禁止** |
| 自建双份 embed/lm_head | 引用 target |
| 写死全部 sliding | 由 config **`layer_types`** 逐层决定 SW / full |
| Decode 把 aux cat/fc pad 到满长 `L_ctx` | **改为** Decode 仅 `block`；Attn 用配置 `L_ctx`（见 3.1.5 / 3.1.6） |
| 无 CLI 覆盖 / block≥2 隐式启用 | 须先 `--speculative-method`；`--num-speculative-tokens` 为从属；`--num-draft-layers` / path 为共享从属；`--acceptance-length` **仅**吞吐 CLI |
