# Copyright (C) 2025 HuggingFace Inc. team.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
from typing import Optional, Tuple

import torch
from torch._subclasses.fake_tensor import is_fake

from ..utils import register_tensor_cast_op


# ============================================================
# V4 HC (Head Compression) ops
# ============================================================


@register_tensor_cast_op("hc_pre_inv_rms")
def _(x: torch.Tensor, hc_mult: int) -> torch.Tensor:
    """Semantic op for HC pre inverse-RMS stage.

    Input `x` is the HC-aware hidden state shaped `[B,S,Hc,D]`. Semantically
    this op corresponds to the reference `hc_pre` steps:
        x_flat = x.flatten(2).float()
        rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + eps)

    It returns the per-row inverse-RMS factor shaped `[B,S,1]`, which is then
    multiplied onto the HC mix projection before sinkhorn splitting.
    """
    batch_shape = x.shape[:-2]
    return torch.empty(*batch_shape, 1, dtype=torch.float32, device=x.device)


@register_tensor_cast_op("hc_pre_sinkhorn")
def _(
    x: torch.Tensor,
    hidden_states: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int,
    sinkhorn_iters: int = 1,
    hc_eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Semantic op for HC sinkhorn mixing + weighted reduction.

    Inputs:
      - `x`: HC mix tensor after Cast + Matmul + inverse-RMS scaling, shaped
        `[B,S,mix_hc]` where `mix_hc = (2 + hc_mult) * hc_mult` in the V4
        reference flow.
      - `hidden_states`: original HC-expanded hidden state shaped
        `[B,S,hc_mult,D]`. The op consumes it to produce the reduced hidden
        state `sum(pre.unsqueeze(-1) * hidden_states, dim=2).to(dtype)`.
      - `hc_scale`, `hc_base`: learned sinkhorn shaping parameters from the
        reference `hc_split_sinkhorn(mixes, hc_scale, hc_base, ...)` call.

    Returns:
      - reduced: `[B,S,D]` weighted-sum hidden state in `hidden_states.dtype`
      - post:    `[B,S,hc_mult]`
      - comb:    `[B,S,hc_mult,hc_mult]`
    """
    batch_shape = x.shape[:-1]
    hidden_size = hidden_states.shape[-1]
    return (
        torch.empty(*batch_shape, hidden_size, dtype=hidden_states.dtype, device=x.device),
        torch.empty(*batch_shape, hc_mult, dtype=x.dtype, device=x.device),
        torch.empty(*batch_shape, hc_mult, hc_mult, dtype=x.dtype, device=x.device),
    )


@register_tensor_cast_op("hc_post")
def _(
    x: torch.Tensor,
    residual: torch.Tensor,
    hc_weight: Optional[torch.Tensor],
    hc_combine: Optional[torch.Tensor],
    hc_mult: int,
) -> torch.Tensor:
    """Semantic op for HC post stage (model.py 683-686).

    Computes:
        y = post.unsqueeze(-1) * x.unsqueeze(-2)
            + sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)

    The `comb * residual` term folds the residual into the HC-mixed output,
    so the caller MUST NOT apply an extra `residual + y` afterwards (doing so
    would double-count the residual contribution).
    """
    batch_shape = x.shape[:-1]
    hidden = x.shape[-1]
    return torch.empty(*batch_shape, hc_mult, hidden, dtype=x.dtype, device=x.device)


@register_tensor_cast_op("hc_head")
def _(
    x: torch.Tensor,
    hc_head_fn: torch.Tensor,
    hc_head_scale: torch.Tensor,
    hc_head_base: torch.Tensor,
    hc_mult: int,
    hc_eps: float = 1e-6,
) -> torch.Tensor:
    """Semantic op for HC head reduction (model.py 728-735).

    Mirrors the reference `ParallelHead.hc_head(x, hc_fn, hc_scale, hc_base)`:
        x_flat = x.flatten(2).float()
        rsqrt = rsqrt(mean(x_flat^2) + norm_eps)
        mixes = linear(x_flat, hc_fn) * rsqrt
        pre = sigmoid(mixes * hc_scale + hc_base) + hc_eps
        y = sum(pre.unsqueeze(-1) * x, dim=2).to(x.dtype)

    Input `x` is the HC-expanded final hidden state shaped `[B,S,Hc,D]`. The
    op encapsulates the full reduction back to `[B,S,D]` so the upper layer
    does not need to spell out the linear / sigmoid / weighted-sum chain.
    """
    batch_shape = x.shape[:-2]
    hidden = x.shape[-1]
    return torch.empty(*batch_shape, hidden, dtype=x.dtype, device=x.device)


# ============================================================
# V4 MLA / attention ops
# ============================================================


@register_tensor_cast_op("scatter_nd_update_mla", mutates_args=("kv_cache",))
def _(
    kv: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    window_size: Optional[int] = None,
    seq_lens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Semantic op for writing V4 MLA window KV into cache.

    V4's main attention path (Flash/Pro) materializes KV explicitly, then performs a
    scatter-style cache update before sparse attention consumes the shared KV
    memory. This op keeps that write visible in the semantic graph without
    collapsing it back into the older `concat_and_cache_mla` abstraction.

    The op writes `kv` into `kv_cache` at `slot_mapping` positions and returns
    a functional handle to the updated cache (same shape/dtype as the input
    `kv_cache`). Returning the cache handle (instead of a `like(kv)` tensor)
    lets callers wire the post-write cache directly into the next consumer
    (e.g. `sparse_attn_sharedkv(q, kv_cache=...)`), establishing a real
    producer/consumer data edge for the entire `wkv -> kv_norm -> RoPE -> cat
    -> scatter -> sparse_attn` chain rather than relying on side-effect
    ordering alone.
    """
    return torch.empty_like(kv_cache)


@register_tensor_cast_op("compressor", mutates_args=("kv_cache",))
def _(
    hidden_states: torch.Tensor,
    kv_cache: torch.Tensor,
    compress_ratio: int,
    head_dim: int,
    rope_head_dim: int,
    rotate: bool,
    seq_lens: Optional[torch.Tensor] = None,
    query_lens: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Semantic op for V4 Compressor (Flash/Pro; mirrors `Compressor.forward`).

    Writes the coarse KV stream into the enclosing `kv_cache` AND returns:
        - `compressed_kv` (shape `[B, seq_len // ratio, head_dim]`) so the
          prefill caller can `torch.cat([kv, kv_compress], dim=1)` exactly like
          `model.py:524-526`.
        - `kv_cache_handle` (same shape/dtype as input `kv_cache`) so callers
          can rebind `kv_cache` and feed it into the next consumer
          (e.g. `sparse_attn_sharedkv`, `quant_lightning_indexer`). Without this
          explicit data edge, torch.compile DCE drops the compressor when the
          downstream consumer reads the pre-mutation `kv_cache` symbol — the
          same fix pattern used by `scatter_nd_update_mla` above.

    `rope_head_dim` matches reference `Compressor.rope_head_dim` and lets the
    cost model bill RoPE only on `kv[..., -rd:]` (model.py:367) and
    act_quant only on `kv[..., :-rd]` (model.py:372).

    `rotate` matches reference `Compressor.rotate`: True for indexer compressor
    (Hadamard + fp4 over full d), False for main KV compressor (block act_quant
    over nope d-rd only). See model.py:368-372.
    """
    batch, seq_len, _ = hidden_states.shape
    if query_lens is not None and not is_fake(query_lens) and query_lens.numel() > 0:
        # Packed single-token requests (`query_len == 1`) correspond to decode.
        # Keep the compressed output at one row so the semantic op matches the
        # reference decoder's incremental cache update shape.
        if bool(torch.all(query_lens == 1)):
            compressed_seq = 1
        else:
            compressed_seq = seq_len // compress_ratio if seq_len >= compress_ratio else 0
    else:
        compressed_seq = seq_len // compress_ratio if seq_len >= compress_ratio else 0

    compressed_kv = torch.empty(batch, compressed_seq, head_dim, dtype=hidden_states.dtype, device=hidden_states.device)
    return compressed_kv, torch.empty_like(kv_cache)


@register_tensor_cast_op("quant_lightning_indexer")
def _(
    q_states: torch.Tensor,
    weights: torch.Tensor,
    indexer_cache: torch.Tensor,
    topk_limit: int,
    tp_world_size: int = 1,
    seq_lens: Optional[torch.Tensor] = None,
    query_lens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Semantic op for V4 ratio=4 learned sparse indexer score/topk core (Flash/Pro).

    Emitted by `DeepseekV4SparseAttentionIndexer.forward(...)` after that wrapper
    has already made the reference `Indexer.forward` preprocessing stages explicit:
        q = wq_b(qa_normed).unflatten(...)
        apply_rotary_emb(q[..., -rd:])
        compressor(x, ...) -> writes indexer_cache (separate `tensor_cast.compressor` event)
        weights = weights_proj(hidden_states) * (softmax_scale * n_heads**-0.5)

    The reference (deepseek-ai/DeepSeek-V4-Flash/inference/model.py:402-433) also runs
    `rotate_activation(q)` and `fp4_act_quant(q, fp4_block_size, True)` between
    the q-RoPE and the compressor write. tensor_cast has no standalone semantic
    op for either; the wrapper does not surface them in the trace, and instead
    their FLOPs/bytes are charged inside this op's cost model so that the
    modeled latency matches the reference's actual runtime.

    This semantic op models the remaining learned sparse-indexer core:
        rotate_activation(q) + fp4_act_quant(q)         # absorbed cost-only
        local_score = einsum("bshd,btd->bsht", q, indexer_kv_cache)
        local_score = (local_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)
        if world_size > 1: score = all_reduce_sum(local_score)
        else: score = local_score
        if prefill (start_pos == 0):
            score += where(causal_mask, -inf, 0)
        topk_indices = topk(score, k=min(topk_limit, active_seq_len))
        if prefill:
            topk_indices = where(validity_mask, -1, topk_indices + offset)
        else:
            topk_indices += offset

    `q_states` is already TP-local and RoPE-processed. `weights` already carries
    the reference scaling term. `tp_world_size` tells the semantic/perf layer
    whether the post-head-reduction score must be all-reduced before top-k.

    Unlike the old fixed-width modeling, the output width follows the reference
    V4 indexer and is clamped by the active compressed-sequence length:
        min(topk_limit, active_seq_len)
    where `active_seq_len ~= end_pos // compress_ratio`. For V4 ratio=4 layers,
    callers pass the full `seq_lens`, so we conservatively model the active
    compressed length as `max(seq_lens) // 4`.
    """
    batch, seq = q_states.shape[:2]
    active_seq_len = seq
    if (
        seq_lens is not None
        and not is_fake(seq_lens)
        and getattr(seq_lens, "device", None) is not None
        and seq_lens.device.type != "meta"
    ):
        active_seq_len = int(seq_lens.max().item()) // 4
    topk = min(topk_limit, max(active_seq_len, 1))
    return torch.empty(batch, seq, topk, dtype=torch.long, device=q_states.device)


@register_tensor_cast_op("sparse_attn_sharedkv")
def _(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_indices: torch.Tensor,
    softmax_scale: float,
    head_dim: int,
    kv_dependency: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Semantic op for V4 sparse attention with shared KV.

    The effective per-token KV length attended is `topk_indices.shape[-1]` —
    callers pass the already-merged window+compress indices. `attn_sink` and
    `softmax_scale` are carried explicitly so the op record matches the
    reference sparse-attention call signature.
    """
    del attn_sink, softmax_scale
    if kv_dependency is not None:
        # Keep optional cache-update handles live in the graph without spelling
        # out a full-tensor arithmetic dependency in Python.
        _ = kv_dependency.shape
    batch_size, seq_length, num_heads, _ = q.shape
    # Reference V4 attention keeps the shared-KV / output stream in the model
    # working dtype; q may be transiently promoted for compute, but the attention
    # result feeding inverse-RoPE and O projection should stay aligned with the
    # KV / hidden-state stream dtype.
    return torch.empty(batch_size, seq_length, num_heads, head_dim, dtype=kv.dtype, device=q.device)


# ============================================================
# V4 MoE gating ops
# ============================================================


@register_tensor_cast_op("moe_gating_top_k")
def _(
    scores: torch.Tensor,
    top_k: int,
    normalize_weights: bool = True,
    route_scale: float = 1.0,
    bias: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Post-score topk routing tail for V4 non-hash MoE layers (Flash/Pro). Covers the
    steps that follow the gate matmul + score function in the reference
    Gate.forward (model.py:572-583):
        - optional bias add (scores + bias) for expert selection only
        - topk on the (possibly biased) scores
        - weight gather from the pre-bias scores
        - optional normalization (when score_func != softmax)
        - route_scale multiplication

    The gate matmul and score function (softmax / sigmoid / sqrt-softplus)
    are emitted as standalone ops in `MoELayer.route()` so each is billed
    against its real dtype (fp32 for the matmul, per reference) and its
    actual elementwise cost.

    Args:
        scores: (..., num_experts) post-score values (typically fp32).
        top_k: number of activated experts per token.
        normalize_weights: whether to bill the divide-by-sum normalize step.
        route_scale: routing scale factor (only its presence matters here;
            the value is unused in cost modeling).
        bias: optional per-expert bias added to scores before topk.

    Returns:
        topk_weights: (..., top_k)
        topk_indices: (..., top_k) int64
    """
    del normalize_weights, route_scale
    if bias is not None:
        _ = bias.shape
    out_shape = (*scores.shape[:-1], top_k)
    return (
        torch.empty(out_shape, dtype=scores.dtype, device=scores.device),
        torch.empty(out_shape, dtype=torch.int64, device=scores.device),
    )


@register_tensor_cast_op("moe_gating_top_k_hash")
def _(
    scores: torch.Tensor,
    top_k: int,
    normalize_weights: bool = True,
    route_scale: float = 1.0,
    input_ids: Optional[torch.Tensor] = None,
    tid2eid: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Post-score hash-routing tail for V4 hash layers (Flash/Pro). Covers the steps
    that follow the gate matmul + score function in the reference Gate.forward:
        - hash-table expert lookup keyed by token id (replaces topk on logits)
        - weight gather from the pre-bias scores
        - optional normalization (when score_func != softmax)
        - route_scale multiplication

    The gate matmul and score function (softmax / sigmoid / sqrt-softplus)
    are emitted as standalone ops in `MoELayer.route()` so each is billed
    against its real dtype (fp32 for the matmul, per reference) and its
    actual elementwise cost, instead of being lumped into a single fused
    estimator.

    Args:
        scores: (..., num_experts) post-score values (typically fp32).
        top_k: number of activated experts per token.
        normalize_weights: whether to bill the divide-by-sum normalize step.
        route_scale: routing scale factor (only its presence matters here;
            the value is unused in cost modeling).
        input_ids: token ids used by V4 hash routing.
        tid2eid: token-id -> expert-id lookup table used by V4 hash routing.

    Returns:
        topk_weights: (..., top_k)
        topk_indices: (..., top_k) int64
    """
    del normalize_weights, route_scale
    if input_ids is None:
        raise ValueError("DeepSeek V4 hash routing requires input_ids")
    if tid2eid is None:
        raise ValueError("DeepSeek V4 hash routing requires tid2eid")
    _ = input_ids.shape
    _ = tid2eid.shape[-1]
    out_shape = (*scores.shape[:-1], top_k)
    return (
        torch.empty(out_shape, dtype=scores.dtype, device=scores.device),
        torch.empty(out_shape, dtype=torch.int64, device=scores.device),
    )


@register_tensor_cast_op("v4_clamped_swiglu")
def _(
    gate: torch.Tensor,
    up: torch.Tensor,
    swiglu_limit: float,
) -> torch.Tensor:
    """DeepSeek V4 clamped SwiGLU activation.

    Models the reference expert activation that clamps gate/up projections before
    SiLU-gated multiplication. `swiglu_limit` is carried as an explicit op arg so
    traces and cost models can distinguish V4 experts from unclamped V3 SwiGLU.
    """
    del swiglu_limit
    return torch.empty_like(up)
