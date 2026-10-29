# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under the Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""qwen4_exp (Qwen3.8-Flash-Next) model-specific meta ops.

Op boundaries for the hyper-connection (HC), N-gram embedding and PLE
(Projected Lexical Embedding) building blocks. Linear projections and the
n-gram embedding table lookup stay as explicit torch modules (aten
``addmm``/``embedding`` ops carry their own performance formulas); these ops
capture the fused model-specific kernels:

- ``grouped_rmsnorm``: per-stream FP32 RMSNorm over ``group_size`` slices
  with a selectable scale convention (``scale_mode="one_plus_weight"`` for
  the Qwen4Exp ``x * (1 + weight)`` form of the official
  ``Qwen4ExpTextRMSNorm``). Shared by the HC mixers and the PLE norms.
- ``hc_gated_stream_mean``: sigmoid gate -> per-stream multiply -> mean over
  streams (official ``Qwen4ExpTextGatedResidual`` input mixer, everything
  between the lowrank projections).
- ``qwen4_ngram_hash``: EOS context padding, EOS-segment-aware right
  shifts, int64 XOR hash with seeded odd multipliers, per-head prime
  modulo and table offset (official ``Qwen4ExpTextNGramEmbedding`` hash
  path; the embedding lookup itself stays an ``nn.Embedding``).
- ``qwen4_ple_gate``: per-stream key-query dot, sign-preserving sqrt,
  sigmoid gate and shared-value broadcast (official
  ``Qwen4ExpTextPLELayer`` gating).
- ``qwen4_ple_short_conv``: causal depthwise dilated conv1d + SiLU with
  zero left padding of ``(kernel - 1) * dilation`` tokens (official
  ``Qwen4ExpTextPLELayer._short_conv`` prefill-from-scratch path).

Performance formulas: see tensor_cast/performance_model/builtin_model/qwen4_exp.py.
"""

import torch

from ..utils import register_tensor_cast_op


@register_tensor_cast_op("grouped_rmsnorm")
def _(
    x: torch.Tensor,
    weight: torch.Tensor,
    group_size: int,
    eps: float,
    scale_mode: str = "one_plus_weight",
) -> torch.Tensor:
    """Per-stream FP32 RMSNorm over ``group_size`` slices.

    Boundary: cast to FP32 -> grouped RMS statistics over each
    ``group_size`` slice -> elementwise scale -> cast back. ``scale_mode``
    selects the scale convention: ``"one_plus_weight"`` scales by
    ``1 + weight`` (the Qwen4Exp ``Qwen4ExpTextRMSNorm`` form). Used on the
    flattened stream bundle ``[B, T, hc_count * hidden_size]`` with
    ``group_size = hidden_size`` (one norm group per residual stream).
    """
    if scale_mode != "one_plus_weight":
        raise ValueError(f"Unsupported grouped_rmsnorm scale_mode: {scale_mode!r}")
    return torch.empty_like(x).contiguous()


@register_tensor_cast_op("hc_gated_stream_mean")
def _(
    mix_weight: torch.Tensor,
    normed: torch.Tensor,
    hc_count: int,
) -> torch.Tensor:
    """HC input mixer kernel (official ``Qwen4ExpTextGatedResidual``).

    Boundary: ``sigmoid(mix_weight)`` -> unflatten to
    ``[B, T, hc_count, hidden_size]`` -> gate-multiply the normed bundle ->
    mean over the stream axis -> ``[B, T, hidden_size]``. The lowrank
    ``input_mix_weight_down``/``up`` projections and the intermediate
    ``silu(down / hc_count)`` activation are explicit ops outside this
    kernel.
    """
    hidden_size = normed.shape[-1] // hc_count
    batch_shape = normed.shape[:-1]
    return torch.empty(*batch_shape, hidden_size, dtype=normed.dtype, device=normed.device)


@register_tensor_cast_op("qwen4_ngram_hash")
def _(
    input_ids: torch.Tensor,
    multipliers: torch.Tensor,
    head_vocab_sizes: torch.Tensor,
    head_offsets: torch.Tensor,
    ngram_size: int,
    heads_per_ngram: int,
    eos_token_id: int,
) -> torch.Tensor:
    """N-gram hash kernel (official ``Qwen4ExpTextNGramEmbedding``).

    Boundary: EOS context padding of ``ngram_size - 1`` tokens ->
    EOS-segment-aware right shifts -> int64 XOR hash with seeded odd
    multipliers -> per-head prime modulo -> cumulative table offset add.
    Output: ``[B, T, (ngram_size - 1) * heads_per_ngram]`` int64 row
    indices for the shared n-gram embedding table; the table lookup itself
    is the surrounding ``nn.Embedding`` (aten ``embedding`` op).

    The hash consumes the full ``[B, T]`` id window (prefill-from-scratch
    semantics with EOS left context); decode steps hash the current
    ``[B, 1]`` window against the EOS-saturated history maintained by the
    surrounding decode loop.
    """
    batch_size, seq_len = input_ids.shape
    ngram_heads = (ngram_size - 1) * heads_per_ngram
    return torch.empty(
        (batch_size, seq_len, ngram_heads),
        dtype=torch.long,
        device=input_ids.device,
    )


@register_tensor_cast_op("qwen4_ple_gate")
def _(
    key_normed: torch.Tensor,
    query_normed: torch.Tensor,
    value: torch.Tensor,
    hidden_size: int,
) -> torch.Tensor:
    """PLE stream gating kernel (official ``Qwen4ExpTextPLELayer``).

    Boundary: per-stream key-query dot scaled by ``1/sqrt(hidden_size)`` ->
    sign-preserving sqrt (``sqrt(|x|.clamp_min(1e-6)) * sign(x)``) ->
    sigmoid -> broadcast-multiply the shared value into every stream ->
    flatten to ``[B, T, hc_count * hidden_size]``. ``key_normed`` and
    ``query_normed`` are the grouped-RMSNormed projections
    ``[B, T, hc_count, hidden_size]``; ``value`` is the shared
    ``[B, T, hidden_size]`` value projection.
    """
    batch_shape = key_normed.shape[:-2]
    hc_count = key_normed.shape[-2]
    return torch.empty(
        (*batch_shape, hc_count * hidden_size),
        dtype=value.dtype,
        device=value.device,
    )


@register_tensor_cast_op("qwen4_ple_short_conv")
def _(
    x: torch.Tensor,
    kernel_size: int,
    dilation: int,
) -> torch.Tensor:
    """PLE causal depthwise conv kernel (official ``_short_conv``).

    Boundary: zero left padding of ``(kernel_size - 1) * dilation`` tokens
    -> causal depthwise conv1d (groups = channels, dilation) -> SiLU.
    Input and output are the flattened stream bundle
    ``[B, T, hc_count * hidden_size]``; the channel-first transpose and the
    residual add stay outside the kernel.
    """
    return torch.empty_like(x).contiguous()
