"""Analytic GLM5Next costs built on shared gated-delta KDA formulas.

These semantic blocks have no measured NPU mapping yet. Composite launch and
scratch estimates are explicit so a whole attention block is not charged as a
single kernel with only hidden-state IO.
"""

import math

import torch

from .. import (
    _accumulate_compute_ops,
    _elementwise_sigmoid_ops,
    _elementwise_silu_ops,
    _elementwise_softplus_ops,
    _rmsnorm_ops,
)
from ..linear_attention import (
    CHUNK_EXTRA_STATIC_KERNELS,
    chunk_gated_delta_rule_scratch_bytes,
    gated_delta_rule_ops,
)
from ..op_invoke_info import OpInvokeInfo


def _requests(query_lens, seq_lens, num_tokens):
    queries = query_lens.tolist()
    lengths = seq_lens.tolist()
    if len(queries) != len(lengths) or sum(queries) != num_tokens:
        raise ValueError("GLM5Next request lengths must match the projected token count")
    if any(q < 0 or s < q for q, s in zip(queries, lengths)):
        raise ValueError("GLM5Next requires 0 <= query length <= sequence length")
    return list(zip(queries, lengths))


@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.glm5_next_kda.default)
def kda_properties(info):
    q, _k, _v, forget, beta, gate, conv, a_log, _dt, _norm, state, qlens, slens, phases, lower, _eps = info.args
    heads, dim = q.shape[-2:]
    tokens = q.numel() // (heads * dim)
    requests = _requests(qlens, slens, tokens)
    if len(phases) != len(requests) or state.shape[0] != len(requests):
        raise ValueError("KDA state and phase metadata must have one entry per request")
    # state (10) is a preallocated mutation buffer; qlens/slens (11/12) are host
    # metadata.  Account state traffic below per active request instead of charging
    # the entire batch allocation through the generic tensor-IO helper.
    props = info.get_memory_access_properties(exclude_input_ids={10, 11, 12})
    # The base op plus conv, forget/beta preparation, KDA rule and gated RMSNorm
    # form four logical boundaries, hence three *extra* static-cost charges.
    props.extra_static_cost_count = 3

    # q, k and v are concatenated before the reference depthwise Conv1d: three
    # streams, H*D channels per stream, and T output positions per channel.
    conv_elements = tokens * 3 * heads * dim
    # Reuse the common SiLU budget used by other analytic operators.  GLM5Next
    # configures hidden_act="silu" for this causal depthwise convolution.
    conv_gp = conv_elements * 2 * conv.shape[-1] + _elementwise_silu_ops(conv_elements)
    _accumulate_compute_ops(props, q.dtype, gp_ops=conv_gp)

    # Forget gate always adds dt_bias per channel.  exp(A_log) is once per head,
    # not once per token/channel.  Reuse the shared sigmoid/softplus budgets;
    # compare/select are the two remaining operations in torch.where.
    if lower is not None:
        forget_gate_gp_per_element = 1 + 1 + _elementwise_sigmoid_ops(1) + 1
    else:
        forget_gate_gp_per_element = 1 + 2 + _elementwise_softplus_ops(1) + 1 + 1
    gate_ops = forget.numel() * forget_gate_gp_per_element + a_log.numel()
    # beta is sigmoid(b_proj), using the same sigmoid accounting as other ops.
    beta_ops = _elementwise_sigmoid_ops(beta.numel())
    # Reuse the common RMSNorm accounting, then add this module's affine weight,
    # sigmoid(gate), and final gated multiplication.
    norm_ops = _rmsnorm_ops(tokens * heads, dim)
    norm_ops += gate.numel() + _elementwise_sigmoid_ops(gate.numel()) + gate.numel()
    _accumulate_compute_ops(props, torch.float32, gp_ops=gate_ops + beta_ops + norm_ops)

    # The conv output is activation dtype; post-gate forget and beta are FP32.
    # Each intermediate is materialized once then consumed once, so the factor 2
    # is write + later read.  The factor 4 is the FP32 byte width.
    props.memory_readwrite_bytes += 2 * (conv_elements * q.element_size() + (forget.numel() + beta.numel()) * 4)
    per_request_state = state.shape[1]
    hidden_rule_gp = 0
    fp32_rule_mma = 0
    fp32_rule_gp = 0
    for (query_len, seq_len), decode in zip(requests, phases):
        if query_len == 0:
            continue
        # Every active request produces an updated conv history and recurrent
        # matrix.  It needs the previous packed state only when history exists.
        props.memory_write_bytes += per_request_state
        if seq_len > query_len:
            props.memory_read_bytes += per_request_state
        use_recurrent = bool(decode) and query_len == 1
        mma_ops, hidden_gp_ops, fp32_gp_ops = gated_delta_rule_ops(
            1,
            query_len,
            heads,
            dim,
            dim,
            use_recurrent=use_recurrent,
        )
        hidden_rule_gp += hidden_gp_ops
        fp32_rule_mma += mma_ops
        fp32_rule_gp += fp32_gp_ops
        if not use_recurrent:
            # Keep GLM's packed state accounting above, but share the chunk
            # workspace and launch-boundary model with other KDA families.
            props.memory_readwrite_bytes += chunk_gated_delta_rule_scratch_bytes(
                1,
                query_len,
                heads,
                dim,
                dim,
                64,
                q.dtype,
            )
            props.extra_static_cost_count = max(props.extra_static_cost_count, 3 + CHUNK_EXTRA_STATIC_KERNELS)
    _accumulate_compute_ops(props, q.dtype, gp_ops=hidden_rule_gp)
    _accumulate_compute_ops(props, torch.float32, mma_ops=fp32_rule_mma, gp_ops=fp32_rule_gp)
    return props


@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.glm5_next_kpool_indexer.default)
def kpool_properties(info):
    q, _k, _gates, weights, _ape, cache, qlens, slens, pool, topk, tail = info.args
    heads, dim = q.shape[-2:]
    tokens = q.numel() // (heads * dim)
    # cache (5) is a preallocated mutation buffer; qlens/slens (6/7) are host
    # metadata.  Cache traffic is accounted below from active token/sequence
    # lengths, rather than charging every allocated cache slot.
    props = info.get_memory_access_properties(exclude_input_ids={5, 6, 7})
    # Pooling, query-to-pool scoring, and selection/expansion are three logical
    # stages, so add two static costs to the base semantic op.
    props.extra_static_cost_count = 2
    # Reference packed cache layout is [key:D, compression logits:D, valid:1].
    # Each current token appends one such 2D+1-wide entry.
    props.memory_write_bytes += tokens * (2 * dim + 1) * cache.element_size()
    for query_len, seq_len in _requests(qlens, slens, tokens):
        if query_len == 0:
            continue
        pools = math.ceil(seq_len / pool)
        candidates = query_len * pools
        # Each active request scans its packed key/logit/valid cache once; the
        # resulting pooled candidates are then shared by that request's queries.
        props.memory_read_bytes += seq_len * (2 * dim + 1) * cache.element_size()
        # q@[pooled_key]^T computes one D-wide dot product per [query, pool,
        # head].  An FMA counts as two FLOP-equivalent MMA operations.
        _accumulate_compute_ops(props, torch.float32, mma_ops=2 * candidates * heads * dim)
        # Pool construction combines gate+APE, softmax, and a weighted key
        # reduction.  Eight is the fused FP32 vector-work budget per cached
        # key channel; the gate projection itself remains a separate Linear op.
        pool_build_gp = seq_len * dim * 8
        # Per [query, pool, head], score scaling/ReLU and the small head-weight
        # reduction are modeled as five FP32 GP operations.
        score_postprocess_gp = candidates * heads * 5
        # topk selects pools, not raw tokens: its fan-in is topk/pool.  Clamp
        # the logarithm to at least one comparison stage for small selections.
        selection_stages = max(1, math.ceil(math.log2(max(2, topk // pool))))
        topk_select_gp = candidates * selection_stages
        # Expanding selected pools produces topk raw indices; when tail is
        # enabled, at most pool-1 additional raw indices are appended.  The
        # factor two covers index expansion and validity/mask handling.
        output_width = topk + (pool - 1 if tail else 0)
        index_expand_gp = query_len * output_width * 2
        gp = pool_build_gp + score_postprocess_gp + topk_select_gp + index_expand_gp
        _accumulate_compute_ops(props, torch.float32, gp_ops=gp)
        # FP32 workspace contains pooled D-vectors and one score per candidate.
        # Factor 4 is bytes/FP32; factor 2 is materialization then consumption.
        props.memory_readwrite_bytes += 2 * 4 * (pools * dim + candidates)
    # weights_proj is a separate Linear.  This is only its output activation's
    # per-element scale/use in the indexer, kept in the activation dtype.
    _accumulate_compute_ops(props, weights.dtype, gp_ops=weights.numel())
    return props
