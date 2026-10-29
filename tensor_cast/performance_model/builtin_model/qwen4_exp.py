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

"""Analytic performance formulas for the qwen4_exp (Qwen3.8-Flash-Next)
model-specific ops: grouped RMSNorm, HC input mixer, N-gram hash, PLE
gating and the PLE causal depthwise conv.

Op boundaries are documented in tensor_cast/ops/qwen4_exp.py. Linear
projections and the n-gram embedding table lookup stay as aten
``addmm``/``embedding`` ops and carry their own formulas.
"""

import torch

from .. import (
    _accumulate_compute_ops,
    _elementwise_sigmoid_ops,
    _elementwise_silu_ops,
    _rmsnorm_ops,
)
from ..cost_utils import byte_count
from ..op_invoke_info import OpInvokeInfo


@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.grouped_rmsnorm.default)
def _(
    op_invoke_info: OpInvokeInfo,
) -> OpInvokeInfo.PerformanceProperties:
    x = op_invoke_info.args[0]
    weight = op_invoke_info.args[1]
    group_size = int(op_invoke_info.args[2])
    scale_mode = op_invoke_info.args[4] if len(op_invoke_info.args) > 4 else "one_plus_weight"
    if scale_mode != "one_plus_weight":
        raise ValueError(f"Unsupported grouped_rmsnorm scale_mode: {scale_mode!r}")

    dim = x.size(-1)
    numel = x.numel()
    num_rows = numel // max(dim, 1)
    groups_per_row = dim // max(group_size, 1)

    properties = op_invoke_info.get_memory_access_properties(exclude_input_ids={1})
    # The broadcast norm weight is read once, not once per row.
    properties.memory_read_bytes += byte_count(dim, weight.dtype)

    # FP32 cast + per-group RMS statistics + (1 + w) elementwise scale.
    rms_gp_ops = _rmsnorm_ops(num_rows * groups_per_row, group_size)
    scale_gp_ops = 2 * numel
    _accumulate_compute_ops(properties, torch.float32, gp_ops=rms_gp_ops + scale_gp_ops)
    # One FP32 intermediate round-trip (cast in, cast out), as in the
    # GLM5 hc_pre_inv_rms convention.
    properties.memory_readwrite_bytes += 8 * numel
    return properties


@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.hc_gated_stream_mean.default)
def _(
    op_invoke_info: OpInvokeInfo,
) -> OpInvokeInfo.PerformanceProperties:
    mix_weight = op_invoke_info.args[0]
    normed = op_invoke_info.args[1]
    hc_count = max(int(op_invoke_info.args[2]), 1)

    hc_hidden = normed.size(-1)
    hidden_size = hc_hidden // hc_count
    tokens = normed.numel() // max(hc_hidden, 1)

    properties = op_invoke_info.get_memory_access_properties()
    # Sigmoid gate -> per-stream multiply -> mean over the stream axis.
    gate_gp_ops = _elementwise_sigmoid_ops(tokens * hc_hidden)
    mul_gp_ops = tokens * hc_hidden
    # (hc_count - 1) additions + one division per output element.
    mean_gp_ops = tokens * hidden_size * hc_count
    _accumulate_compute_ops(
        properties,
        mix_weight.dtype,
        gp_ops=gate_gp_ops + mul_gp_ops + mean_gp_ops,
    )
    return properties


@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.qwen4_ngram_hash.default)
def _(
    op_invoke_info: OpInvokeInfo,
) -> OpInvokeInfo.PerformanceProperties:
    input_ids = op_invoke_info.args[0]
    ngram_size = int(op_invoke_info.args[4])
    heads_per_ngram = int(op_invoke_info.args[5])

    batch_size, seq_len = input_ids.shape
    ngram_heads = (ngram_size - 1) * heads_per_ngram

    properties = op_invoke_info.get_memory_access_properties(exclude_input_ids={1, 2, 3})
    # Seeded multipliers + per-head vocab sizes and offsets (int64 tables).
    properties.memory_read_bytes += byte_count(ngram_size + 2 * ngram_heads, torch.long)

    # int64 vector work per token, accounted under FP32 vector throughput
    # (device capability tables carry no separate INT64 vector rate):
    # - EOS-segment-aware right shifts: gather/compare/select per shift
    #   (cummax segment scan amortizes to a constant per token);
    # - XOR hash: n multiplies + (n - 1) xors per n-gram level (levels 2..N);
    # - per-head prime modulo + cumulative offset add.
    shift_gp_ops = 4 * ngram_size
    hash_gp_ops = sum(2 * n - 1 for n in range(2, ngram_size + 1))
    mod_gp_ops = 2 * ngram_heads
    _accumulate_compute_ops(
        properties,
        torch.float32,
        gp_ops=batch_size * seq_len * (shift_gp_ops + hash_gp_ops + mod_gp_ops),
    )
    return properties


@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.qwen4_ple_gate.default)
def _(
    op_invoke_info: OpInvokeInfo,
) -> OpInvokeInfo.PerformanceProperties:
    key_normed = op_invoke_info.args[0]
    value = op_invoke_info.args[2]
    hidden_size = int(op_invoke_info.args[3])

    hc_count = key_normed.size(-2)
    tokens = value.numel() // max(hidden_size, 1)

    properties = op_invoke_info.get_memory_access_properties()
    # Per-stream key-query dot (mul + add per dot element).
    dot_gp_ops = tokens * hc_count * hidden_size * 2
    # 1/sqrt(H) scale + sign-preserving sqrt chain (abs/clamp/sqrt/sign/mul)
    # + sigmoid gate, one per stream.
    act_gp_ops = tokens * hc_count * (1 + 5 + _elementwise_sigmoid_ops(1))
    # Broadcast the shared value into every stream.
    broadcast_gp_ops = tokens * hc_count * hidden_size
    _accumulate_compute_ops(
        properties,
        value.dtype,
        gp_ops=dot_gp_ops + act_gp_ops + broadcast_gp_ops,
    )
    return properties


@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.qwen4_ple_short_conv.default)
def _(
    op_invoke_info: OpInvokeInfo,
) -> OpInvokeInfo.PerformanceProperties:
    x = op_invoke_info.args[0]
    kernel_size = max(int(op_invoke_info.args[1]), 1)

    channels = x.size(-1)
    tokens = x.numel() // max(channels, 1)

    properties = op_invoke_info.get_memory_access_properties()
    # Causal depthwise conv (kernel_size MACs per output element, counted
    # as mul + add) + SiLU. The dilation only extends the zero-padded
    # window, not the MAC count. The zero-pad state is not materialized by
    # a fused kernel, so prefill-from-scratch carries no state traffic.
    conv_gp_ops = tokens * channels * kernel_size * 2 + _elementwise_silu_ops(tokens * channels)
    # Depthwise conv kernel weights, read once.
    properties.memory_read_bytes += byte_count(channels * kernel_size, x.dtype)
    _accumulate_compute_ops(properties, x.dtype, gp_ops=conv_gp_ops)
    return properties
