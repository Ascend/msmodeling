# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
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

"""Analytic models for shared manifold-constrained Hyper-Connection ops."""

import torch

from . import _accumulate_compute_ops, _rmsnorm_ops
from .op_invoke_info import OpInvokeInfo


@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.hc_pre_inv_rms.default)
def _(
    op_invoke_info: OpInvokeInfo,
) -> OpInvokeInfo.PerformanceProperties:
    x = op_invoke_info.args[0]
    hc_mult = max(int(op_invoke_info.args[1]), 1)
    hidden_size = x.size(-1)
    row_width = hc_mult * hidden_size
    num_rows = x.numel() // max(row_width, 1)
    properties = op_invoke_info.get_memory_access_properties()
    cast_gp_ops = num_rows * row_width
    rms_gp_ops = _rmsnorm_ops(num_rows, row_width)
    # The fused implementation retains one FP32 intermediate round-trip.
    properties.memory_readwrite_bytes += 8 * num_rows * row_width
    _accumulate_compute_ops(properties, torch.float32, gp_ops=cast_gp_ops + rms_gp_ops)
    return properties


@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.hc_pre_sinkhorn.default)
def _(
    op_invoke_info: OpInvokeInfo,
) -> OpInvokeInfo.PerformanceProperties:
    x = op_invoke_info.args[0]
    hidden_states = op_invoke_info.args[1]
    hc_mult = max(int(op_invoke_info.args[4]), 1)
    sinkhorn_iters = max(int(op_invoke_info.args[5]) if len(op_invoke_info.args) > 5 else 1, 1)
    hc_eps = float(op_invoke_info.args[6]) if len(op_invoke_info.args) > 6 else 1e-6
    row_width = x.size(-1)
    num_rows = x.numel() // max(row_width, 1)
    hidden_size = hidden_states.size(-1)
    properties = op_invoke_info.get_memory_access_properties()

    # Build pre/post/comb, then apply the initial softmax/column normalization.
    setup_gp_ops = num_rows * (hc_mult + hc_mult * 2 + hc_mult * hc_mult * 2)
    first_norm_gp_ops = num_rows * (hc_mult * hc_mult * 5 + hc_mult * 2)
    extra_iters = max(sinkhorn_iters - 1, 0)
    eps_per_iter_gp_ops = hc_mult * hc_mult * 2 if hc_eps != 0 else 0
    extra_iter_gp_ops = num_rows * extra_iters * (hc_mult * hc_mult * 4 + hc_mult * 2 + eps_per_iter_gp_ops)

    # Collapse the HC streams and cast back to the model working dtype.
    reduce_gp_ops = num_rows * hc_mult * hidden_size * 2
    cast_gp_ops = num_rows * hidden_size
    properties.memory_readwrite_bytes += num_rows * hidden_size * 8
    _accumulate_compute_ops(
        properties,
        torch.float32,
        gp_ops=setup_gp_ops + first_norm_gp_ops + extra_iter_gp_ops + reduce_gp_ops,
    )
    _accumulate_compute_ops(properties, hidden_states.dtype, gp_ops=cast_gp_ops)
    return properties


@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.hc_post.default)
def _(
    op_invoke_info: OpInvokeInfo,
) -> OpInvokeInfo.PerformanceProperties:
    x = op_invoke_info.args[0]
    hc_mult = max(int(op_invoke_info.args[4]), 1)
    hidden_size = x.size(-1)
    num_rows = x.numel() // max(hidden_size, 1)
    properties = op_invoke_info.get_memory_access_properties()
    comb_reduce_gp_ops = num_rows * hc_mult * hc_mult * hidden_size * 2
    post_gp_ops = num_rows * hc_mult * hidden_size * 2
    cast_gp_ops = num_rows * hc_mult * hidden_size
    properties.memory_readwrite_bytes += num_rows * hc_mult * hidden_size * 8
    _accumulate_compute_ops(properties, torch.float32, gp_ops=comb_reduce_gp_ops + post_gp_ops)
    _accumulate_compute_ops(properties, x.dtype, gp_ops=cast_gp_ops)
    return properties


@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.hc_head.default)
def _(
    op_invoke_info: OpInvokeInfo,
) -> OpInvokeInfo.PerformanceProperties:
    x = op_invoke_info.args[0]
    hc_mult = max(int(op_invoke_info.args[4]), 1)
    hidden_size = x.size(-1)
    row_width = hc_mult * hidden_size
    leading = 1
    for size in x.shape[:-2]:
        leading *= int(size)
    properties = op_invoke_info.get_memory_access_properties()
    rms_gp = _rmsnorm_ops(leading, row_width)
    mma_ops = leading * row_width * hc_mult * 2
    activate_gp = leading * hc_mult * 8
    reduce_gp = leading * hc_mult * hidden_size * 2
    properties.memory_readwrite_bytes += 8 * leading * hidden_size
    _accumulate_compute_ops(
        properties,
        torch.float32,
        mma_ops=mma_ops,
        gp_ops=rms_gp + activate_gp + reduce_gp,
    )
    return properties


@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.mhc_head.default)
def _(op_invoke_info: OpInvokeInfo) -> OpInvokeInfo.PerformanceProperties:
    """GLM5Next's mHC head is an unweighted mean across the stream axis."""
    streams = op_invoke_info.args[0]
    batch, seq, mult, hidden = streams.shape
    properties = op_invoke_info.get_memory_access_properties()
    # (mult - 1) additions and one reciprocal/multiply, conservatively mult GP.
    _accumulate_compute_ops(properties, streams.dtype, gp_ops=batch * seq * hidden * mult)
    return properties
