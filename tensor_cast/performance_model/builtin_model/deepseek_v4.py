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
import math
from typing import List, Optional, Tuple

import torch
from torch._subclasses.fake_tensor import is_fake

from .. import _accumulate_compute_ops, _rmsnorm_ops
from ..op_invoke_info import OpInvokeInfo
from ..utils import bytes_of_elements


def _safe_max_int(tensor: Optional[torch.Tensor]) -> Optional[int]:
    """Return ``int(tensor.max())`` or ``None`` if not safely materializable.

    During analytic / multistream tracing the tensor may be a fake / functional
    / symbolic tensor where ``.item()`` raises. We treat any failure (including
    ``is_fake``-detected fake tensors and meta tensors) as "value unknown" so
    callers can fall back to a shape-based estimate.
    """
    if tensor is None:
        return None
    try:
        if is_fake(tensor):
            return None
    except Exception:
        return None
    if getattr(tensor, "device", None) is not None and tensor.device.type == "meta":
        return None
    try:
        return int(tensor.max().item())
    except Exception:
        return None


@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.scatter_nd_update_mla.default)
def _(
    op_invoke_info: OpInvokeInfo,
) -> OpInvokeInfo.PerformanceProperties:
    assert len(op_invoke_info.args) >= 3
    kv = op_invoke_info.args[0]

    # Reference (model.py:518-530): this stage is a direct KV window-cache
    # assignment (`self.kv_cache[...] = kv`), not a generic indexed scatter.
    # The semantic op carries `slot_mapping` only because msmodeling routes the
    # write through the existing block-cache abstraction; the reference V4
    # implementation does not materialize or consume an index tensor here.
    #
    # Cost model therefore mirrors the reference assignment semantics:
    #   - read the source `kv` rows
    #   - write exactly the updated cache rows
    #   - do NOT charge reads of `slot_mapping`, `seq_lens`, or the full cache
    #     tensor / returned cache handle.
    #
    # Per call, rows written are:
    #   prefill short      = `seqlen`
    #   prefill split tail = `W-cutoff` then `cutoff`
    #   decode             = 1 (or packed multi-decode `sl` rows)
    # Caller is responsible for slicing `kv` to the rows that should land in
    # cache; this op just charges the reference-equivalent assignment traffic.
    properties = op_invoke_info.get_memory_access_properties(
        exclude_input_ids={1, 2, 4},
        exclude_output_ids={0},
    )
    batch, rows_written = kv.shape[0], kv.shape[1]
    per_row_bytes = kv.shape[-1] * kv.element_size()
    properties.memory_write_bytes += batch * rows_written * per_row_bytes
    return properties


@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.hc_pre_inv_rms.default)
def _(
    op_invoke_info: OpInvokeInfo,
) -> OpInvokeInfo.PerformanceProperties:
    # Reference flow (model.py:673-681 of the V4 Flash inference model):
    #     x = x.flatten(2).float()
    #     rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
    # This op covers the bf16->fp32 cast of x plus the RMS-style inverse
    # square root reduction along the flattened (hc_mult * hidden_size) axis.
    x = op_invoke_info.args[0]
    hc_mult = max(int(op_invoke_info.args[1]), 1)
    hidden_size = x.size(-1)
    row_width = hc_mult * hidden_size
    num_rows = x.numel() // max(row_width, 1)
    properties = op_invoke_info.get_memory_access_properties()
    # Up-cast pass over the full input tensor: one elementwise op per element.
    cast_gp_ops = num_rows * row_width
    rms_gp_ops = _rmsnorm_ops(num_rows, row_width)
    # Eager kernel chain materializes two fp32 intermediate buffers in HBM:
    # `x.flatten(2).float()` and `x_fp32.square()`, each numel*4 bytes written
    # then read by the next kernel. Bill 2 * (4 + 4) * num_rows * row_width.
    # FIXED: The bf16->fp32 cast and square() are fused in the reference kernel
    # (model.py:691). Only the fp32 accumulator output and the final cast back
    # are visible HBM traffic. Reduce from 16*B/elem to 8*B/elem (one fp32
    # output buffer round-trip) + the dtype-cast read/write.
    properties.memory_readwrite_bytes += 8 * num_rows * row_width
    _accumulate_compute_ops(properties, torch.float32, gp_ops=cast_gp_ops + rms_gp_ops)
    return properties


@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.hc_pre_sinkhorn.default)
def _(
    op_invoke_info: OpInvokeInfo,
) -> OpInvokeInfo.PerformanceProperties:
    # args: (x, hidden_states, hc_scale, hc_base, hc_mult, sinkhorn_iters, hc_eps)
    #
    # Reference kernel: hc_split_sinkhorn_kernel (kernel.py:372-427).
    # For each row the kernel does:
    #   1. one-shot per-row setup:
    #      - pre  = sigmoid(scale[0] * mixes[:hc] + base[:hc])           ~ hc * 1 (sigmoid)
    #      - post = 2 * sigmoid(scale[1] * mixes[hc:2hc] + base[hc:2hc]) ~ hc * 1 + hc * 1 (sigmoid+scale)
    #      - comb = scale[2] * mixes[2hc:] + base[2hc:]                  ~ hc^2 * 2 (mul+add)
    #   2. first softmax-style normalization step:
    #      row_max + exp + row_sum + divide + col_sum + divide
    #      plus eps additions in each divide (1 add per row/col per iter)
    #                                                                ~ hc^2 * 5 + hc * 2
    #   3. (sinkhorn_iters - 1) cheaper row+col normalizations:
    #      row_sum + divide + col_sum + divide (each with eps addition)
    #                                                                ~ hc^2 * 6 + hc * 4
    #
    # FIXED:
    # - first_norm: row_max(hc*(hc-1)) + exp(hc^2) + row_sum(hc*(hc-1))
    #   + divide(hc^2) + eps_add(hc^2) + col_sum(hc*(hc-1)) + divide(hc^2)
    #   = hc^2 * 5 + hc * 2  (was hc^2 * 6 + hc * 3, overcounted ~40%)
    # - extra_iter: row_sum + row_eps(hc^2) + row_div(hc^2) + col_sum + col_eps(hc^2) + col_div(hc^2)
    #   = hc^2 * 6 + hc * 4  (was hc^2 * 4 + hc * 2, eps was missing 1 add)
    # - memory: fused mul+sum kernel streams result directly without materializing
    #   two separate fp32 buffers; charge only the final output round-trip.
    x = op_invoke_info.args[0]
    hidden_states = op_invoke_info.args[1]
    hc_mult = max(int(op_invoke_info.args[4]), 1)
    sinkhorn_iters = max(int(op_invoke_info.args[5]) if len(op_invoke_info.args) > 5 else 1, 1)
    hc_eps = float(op_invoke_info.args[6]) if len(op_invoke_info.args) > 6 else 1e-6
    row_width = x.size(-1)
    num_rows = x.numel() // max(row_width, 1)
    hidden_size = hidden_states.size(-1)
    properties = op_invoke_info.get_memory_access_properties()

    # One-shot per-row construction: pre sigmoid + post (sigmoid then *2) + comb fill.
    setup_gp_ops = num_rows * (hc_mult * 1 + hc_mult * 2 + hc_mult * hc_mult * 2)
    # First normalization step (kernel.py:401-408): more accurate count.
    first_norm_gp_ops = num_rows * (hc_mult * hc_mult * 5 + hc_mult * 2)
    # Each extra iteration: row_sum + row_eps_add + row_div + col_sum + col_eps_add + col_div.
    extra_iters = max(sinkhorn_iters - 1, 0)
    eps_per_iter_gp_ops = hc_mult * hc_mult * 2 if hc_eps != 0 else 0
    extra_iter_gp_ops = num_rows * extra_iters * (hc_mult * hc_mult * 4 + hc_mult * 2 + eps_per_iter_gp_ops)
    # tilelang fused kernel keeps comb in register fragment across iterations,
    # so sinkhorn body adds no HBM traffic beyond the final pre/post/comb writes
    # already accounted for by get_memory_access_properties.

    # Weighted reduction `sum(pre.unsqueeze(-1) * hidden_states, dim=2)` plus
    # the trailing dtype cast back to `hidden_states.dtype`.
    # FIXED: The fused mul+sum kernel streams the fp32 result directly to the
    # cast kernel; only the final [num_rows, hidden] output buffer round-trips
    # through HBM (one write + one read). Remove the separate mul_buf_bytes.
    reduce_gp_ops = num_rows * hc_mult * hidden_size * 2
    cast_gp_ops = num_rows * hidden_size
    sum_buf_bytes = num_rows * hidden_size * 8
    properties.memory_readwrite_bytes += sum_buf_bytes

    _accumulate_compute_ops(
        properties,
        torch.float32,
        gp_ops=setup_gp_ops + first_norm_gp_ops + extra_iter_gp_ops,
    )
    _accumulate_compute_ops(
        properties,
        torch.float32,
        gp_ops=reduce_gp_ops,
    )
    _accumulate_compute_ops(
        properties,
        hidden_states.dtype,
        gp_ops=cast_gp_ops,
    )
    return properties


@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.hc_post.default)
def _(
    op_invoke_info: OpInvokeInfo,
) -> OpInvokeInfo.PerformanceProperties:
    """Cost of HcPost (model.py 683-686).

    `y = post.unsqueeze(-1) * x.unsqueeze(-2) + sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)`

    The `comb * residual` branch is a small-HC contraction [hc,hc] x [hc,d] ->
    [hc,d]. Model it as a fused mul+reduce over hc rather than a large GEMM or
    a materialized [n,hc,hc,d] broadcast.
    """
    x = op_invoke_info.args[0]
    # args: (x, residual, post, comb, hc_mult)
    hc_mult = max(int(op_invoke_info.args[4]), 1)
    hidden_size = x.size(-1)
    num_rows = x.numel() // max(hidden_size, 1)
    properties = op_invoke_info.get_memory_access_properties()
    # Fused mul+reduce over the HC axis: comb.unsqueeze(-1) * residual.unsqueeze(-2), then sum(dim=2).
    comb_reduce_gp_ops = num_rows * hc_mult * hc_mult * hidden_size * 2
    post_gp_ops = num_rows * hc_mult * hidden_size * 2
    # Final `y.type_as(x)` cast over [num_rows, hc, hidden] (model.py:686).
    cast_gp_ops = num_rows * hc_mult * hidden_size
    # FIXED: The fused `post*x + sum(comb*res,dim=hc)` kernel (model.py:698)
    # produces a single fp32 [num_rows, hc, hidden] output. Charge only that
    # round-trip instead of two separate branch buffers. The type_as cast
    # (bf16->bf16) is cheap and already covered by get_memory_access_properties.
    output_buf_bytes = num_rows * hc_mult * hidden_size * 8
    properties.memory_readwrite_bytes += output_buf_bytes
    _accumulate_compute_ops(properties, torch.float32, gp_ops=comb_reduce_gp_ops + post_gp_ops)
    _accumulate_compute_ops(properties, x.dtype, gp_ops=cast_gp_ops)
    return properties


@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.hc_head.default)
def _(
    op_invoke_info: OpInvokeInfo,
) -> OpInvokeInfo.PerformanceProperties:
    """Cost of HC head reduction (model.py 728-735).

    Mirrors the reference `ParallelHead.hc_head` flow:
      - x_flat = x.flatten(2).float()  (flatten HC into feature width)
      - rsqrt = rsqrt(mean(x_flat^2) + eps)         # RMS over Hc*D
      - mixes = linear(x_flat, hc_head_fn) * rsqrt  # [B,S,Hc] linear
      - pre = sigmoid(mixes * hc_scale + hc_base) + hc_eps
      - y = sum(pre.unsqueeze(-1) * x, dim=2)       # weighted reduction
    """
    x = op_invoke_info.args[0]
    hc_mult = max(int(op_invoke_info.args[4]), 1)
    hidden_size = x.size(-1)
    row_width = hc_mult * hidden_size
    # x has shape [..., hc_mult, hidden]; row_count is product of leading dims.
    leading = 1
    for s in x.shape[:-2]:
        leading *= int(s)
    properties = op_invoke_info.get_memory_access_properties()
    # 1) inverse-RMS over flattened row of width Hc*D
    rms_gp = _rmsnorm_ops(leading, row_width)
    # 2) linear: [leading, Hc*D] @ [Hc*D, Hc] -> [leading, Hc]
    mma_ops = leading * row_width * hc_mult * 2
    # 3) (mixes * rsqrt) then sigmoid(mixes * hc_scale + hc_base) + hc_eps.
    # ~ rsqrt-mul + scale + base + sigmoid(4) + eps = 8 ops/elem.
    activate_gp = leading * hc_mult * 8
    # 4) weighted reduction: pre.unsqueeze(-1) * x, then sum over hc dim
    reduce_gp = leading * hc_mult * hidden_size * 2
    # FIXED: The hc_head kernel (model.py:754-761) is fused: it streams the
    # fp32 x_flat from register, computes rms/linear/activation/reduction in
    # a single pass, and writes only the final [leading, D] fp32 output
    # round-trip. Remove all intermediate buffer charges (flatten/square/weighted
    # are in-register), keep only the final output buffer.
    output_buf_bytes = 8 * leading * hidden_size
    properties.memory_readwrite_bytes += output_buf_bytes
    _accumulate_compute_ops(
        properties,
        torch.float32,
        mma_ops=mma_ops,
        gp_ops=rms_gp + activate_gp + reduce_gp,
    )
    return properties


@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.compressor.default)
def _(
    op_invoke_info: OpInvokeInfo,
) -> OpInvokeInfo.PerformanceProperties:
    """V4 Flash Compressor cost aligned to model.py:316-377."""
    # args: (hidden_states, kv_cache, compress_ratio, head_dim, rope_head_dim, rotate, seq_lens?, query_lens?)
    hidden_states = op_invoke_info.args[0]
    kv_cache = op_invoke_info.args[1]
    compress_ratio = max(int(op_invoke_info.args[2]), 1)
    head_dim = int(op_invoke_info.args[3])
    rope_head_dim = max(int(op_invoke_info.args[4]), 0)
    nope_head_dim = max(head_dim - rope_head_dim, 0)
    rotate = bool(op_invoke_info.args[5])
    seq_lens = op_invoke_info.args[6] if len(op_invoke_info.args) > 6 else None
    query_lens = op_invoke_info.args[7] if len(op_invoke_info.args) > 7 else None
    batch, seq_len, hidden_size = hidden_states.shape
    # ratio==4 layers run with overlap=True (coff=2 in reference Compressor).
    coff = 2 if compress_ratio == 4 else 1
    overlap = coff == 2
    proj_out_dim = coff * head_dim

    # Per-request (seq_len_i, query_len_i) pairs. Falls back to one synthetic
    # request whose start_pos comes from `max(seq_lens) - seq_len` when only
    # seq_lens is available (legacy path), or pure prefill when neither is.
    per_req: List[Tuple[int, int]] = []
    seq_lens_list: Optional[List[int]] = None
    query_lens_list: Optional[List[int]] = None
    if (
        seq_lens is not None
        and not is_fake(seq_lens)
        and getattr(seq_lens, "device", None) is not None
        and seq_lens.device.type != "meta"
    ):
        try:
            seq_lens_list = [int(v) for v in seq_lens.tolist()]
        except Exception:
            seq_lens_list = None
    if (
        query_lens is not None
        and not is_fake(query_lens)
        and getattr(query_lens, "device", None) is not None
        and query_lens.device.type != "meta"
    ):
        try:
            query_lens_list = [int(v) for v in query_lens.tolist()]
        except Exception:
            query_lens_list = None
    if seq_lens_list is not None and query_lens_list is not None and len(seq_lens_list) == len(query_lens_list):
        per_req = [(s, q) for s, q in zip(seq_lens_list, query_lens_list) if q > 0]
    elif seq_lens_list is not None and len(seq_lens_list) > 0:
        # Legacy: derive a single combined request from max(seq_lens).
        per_req = [(max(seq_lens_list), seq_len)]
    else:
        per_req = [(seq_len, seq_len)]

    properties = op_invoke_info.get_memory_access_properties(
        exclude_input_ids={1, 4},
        exclude_output_ids={0, 1},
    )
    # wkv + wgate: two fp32 Linears x -> coff*d (model.py:323-324).
    proj_mma = batch * seq_len * hidden_size * proj_out_dim * 2
    mma_ops = 2 * proj_mma
    # fp32 weight reads (input x is auto-counted).
    properties.memory_read_bytes += 2 * hidden_size * proj_out_dim * 4
    # Eager `x = x.float()` (model.py:322) materializes a fp32 hidden_states
    # buffer consumed by wkv/wgate. Bill one round-trip (write+read=8 B/elem).
    properties.memory_readwrite_bytes += 8 * batch * seq_len * hidden_size

    gp_ops = 0
    # bf16 -> fp32 cast pass over hidden_states (model.py:322).
    gp_ops += batch * seq_len * hidden_size

    # Accumulate per-request costs. Each prefill request contributes full
    # prefill cost on its own `query_len_i` rows; each decode-token contributes
    # one decode-step cost with its own `start_pos`.
    total_post_compress_rows = 0  # sum over requests of compressed rows produced
    for total_seq_i, q_len_i in per_req:
        start_pos_i = max(total_seq_i - q_len_i, 0)
        is_prefill_i = start_pos_i == 0
        if is_prefill_i:
            eff_seq = q_len_i
            remainder = eff_seq % compress_ratio
            cutoff = eff_seq - remainder
            post_compress_run_i = eff_seq >= compress_ratio
            compressed_seq_i = (eff_seq // compress_ratio) if post_compress_run_i else 0
            state_rows = 0
            if overlap and cutoff >= compress_ratio:
                state_rows += compress_ratio
                gp_ops += batch * compress_ratio * proj_out_dim  # score+ape add
            if remainder > 0:
                state_rows += remainder
                gp_ops += batch * remainder * proj_out_dim  # score+ape add
            if state_rows > 0:
                properties.memory_write_bytes += 2 * batch * state_rows * proj_out_dim * 4
            if post_compress_run_i:
                # score.unflatten + ape (model.py:338).
                gp_ops += batch * cutoff * proj_out_dim
                window = 2 * compress_ratio if overlap else compress_ratio
                num_groups = compressed_seq_i
                if overlap:
                    gp_ops += 2 * batch * num_groups * window * proj_out_dim
                elems = batch * num_groups * window * proj_out_dim
                gp_ops += elems * 4  # softmax(dim=2) ~4 ops/elem
                gp_ops += elems * 2  # mul + sum-reduce over window dim
                # FIXED: softmax output [num_groups, window, d] is fused into the
                # weighted-sum kernel; only the final compressed [num_groups, d]
                # result round-trips HBM. Remove the 2 * 8 * elems intermediate charge.
                total_post_compress_rows += compressed_seq_i
        else:
            # Decode: for each of the q_len_i tokens packed in this request,
            # bill the single-token decode path with the token's own start_pos.
            for tok in range(q_len_i):
                tok_start_pos = start_pos_i + tok
                # score += ape (model.py:345).
                gp_ops += batch * proj_out_dim
                # Single-row state write (model.py:347-348/356-357), kv+score fp32.
                properties.memory_write_bytes += 2 * batch * proj_out_dim * 4
                if (tok_start_pos + 1) % compress_ratio == 0:
                    window = 2 * compress_ratio if overlap else compress_ratio
                    row_dim = head_dim if overlap else proj_out_dim
                    if overlap:
                        gp_ops += 2 * batch * window * head_dim
                        properties.memory_write_bytes += 2 * batch * compress_ratio * proj_out_dim * 4
                    elems = batch * window * row_dim
                    gp_ops += elems * 4  # softmax(dim=1)
                    gp_ops += elems * 2  # mul + sum-reduce
                    # FIXED: same as prefill — softmax output is fused into the
                    # weighted-sum kernel; only the compressed kv write round-trips.
                    total_post_compress_rows += 1

    if total_post_compress_rows > 0:
        # norm + RoPE on last rd dims + (rotate=True: Hadamard+fp4 over full d;
        #                                rotate=False: act_quant over nope d-rd)
        # + cache write (model.py:362-376). Aggregated across all requests.
        rows = batch * total_post_compress_rows
        # FIXED: kv.to(dtype) converts fp32 kv to bf16 (write bf16, read bf16 for
        # norm), and norm outputs bf16 kv that is read by RoPE (already counted
        # there via exclude_input_ids). Only charge the fp32->bf16 write+read
        # (4B/elem) and the norm output write (2B/elem). Remove the redundant
        # +kv_cache.element_size() term which double-counts the bf16 read.
        properties.memory_readwrite_bytes += rows * head_dim * 6
        gp_ops += _rmsnorm_ops(rows, head_dim)
        # RoPE only on kv[..., -rd:] (model.py:367).
        gp_ops += rows * rope_head_dim * 5
        if rotate:
            log2_d = max(int(math.log2(max(head_dim, 1))), 1)
            gp_ops += rows * head_dim * (log2_d + 1)  # Hadamard + scale multiply
            # FIXED: fp4_act_quant (kernel.py:129-183) does: reduce_absmax +
            # fast_round_scale (bit ops: log2_ceil + pow2) + clamp + cast + mul.
            # ~5-6 ops/elem (was 3, undercounted).
            gp_ops += rows * head_dim * 5
        else:
            gp_ops += rows * nope_head_dim * 3
        properties.memory_write_bytes += batch * total_post_compress_rows * head_dim * kv_cache.element_size()

    _accumulate_compute_ops(
        properties,
        torch.float32,
        mma_ops=mma_ops,
        gp_ops=gp_ops,
    )
    return properties


@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.apply_rope_inplace.default)
def _(
    op_invoke_info: OpInvokeInfo,
) -> OpInvokeInfo.PerformanceProperties:
    """In-place RoPE on x (model.py:232 apply_rotary_emb).

    Reference: cast x to fp32 -> view as complex pairs -> multiply by freqs_cis
    (conj if inverse) -> view back to real -> copy_ back into x.
    For V4 Flash we often rotate only the trailing `rope_head_dim` channels of a
    wider tensor (e.g. last 64 of head_dim 512). The op still mutates the full
    destination tensor/view in-place, but the math and fp32 intermediates scale
    with the rotated suffix width, not the full hidden width.
    """
    # args: (x, cos, sin, is_neox, inverse, rope_head_dim)
    x = op_invoke_info.args[0]
    sin = op_invoke_info.args[2]
    inverse = bool(op_invoke_info.args[4]) if len(op_invoke_info.args) > 4 else False
    rope_head_dim = int(op_invoke_info.args[5]) if len(op_invoke_info.args) > 5 else int(x.shape[-1])
    rope_head_dim = int(x.shape[-1]) if rope_head_dim < 0 else min(rope_head_dim, int(x.shape[-1]))
    # The semantic op is attached to the full-width tensor so downstream shape
    # propagation stays at head_dim=512, but the reference `apply_rotary_emb`
    # only touches the sliced suffix `x[..., -rope_head_dim:]`. Exclude the
    # auto-counted tensor accesses and bill the rotated slice explicitly.
    properties = op_invoke_info.get_memory_access_properties(
        exclude_input_ids={0, 1, 2},
        exclude_output_ids={0},
    )
    rotated_numel = x.numel() * rope_head_dim // int(x.shape[-1])
    # cos/sin are read once each (bf16 complex-pair buffer).
    # The x in-place read is already auto-counted (x is input 0). Only bill
    # the x in-place write explicitly here (output 0 write was excluded).
    # cos/sin reads: sin.numel() == rotated_numel (complex pairs).
    # FIXED: removed the redundant manual x read that was double-counted
    # against get_memory_access_properties.
    properties.memory_read_bytes += 2 * sin.numel() * sin.element_size()
    properties.memory_write_bytes += rotated_numel * x.element_size()
    # 6 fp32 ops per paired-rotation element (complex mul: 4 muls + 2 adds)
    # over rotated_numel/2 pairs, plus two cast passes over the rotated slice:
    # bf16->fp32 on the way in (`x.float()`) and fp32->orig dtype on the way out.
    rope_gp_ops = (rotated_numel // 2) * 6
    cast_gp_ops = rotated_numel * 2
    # inverse path: freqs_cis.conj() flips sign of imag (sin), one op per rotated element.
    conj_gp_ops = sin.numel() if inverse else 0
    # FIXED: The tilelang fused kernel (model.py:232) performs bf16->fp32 cast,
    # complex-mul, and view_as_real in a single fused pass. Only the fp32
    # intermediate (the cast result before complex-mul) materializes to HBM.
    # Reduce from 2 round-trips (16*B) to 1 (8*B).
    properties.memory_readwrite_bytes += 8 * rotated_numel
    _accumulate_compute_ops(
        properties,
        torch.float32,
        gp_ops=rope_gp_ops + cast_gp_ops + conj_gp_ops,
    )
    return properties


@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.quant_lightning_indexer.default)
def _(
    op_invoke_info: OpInvokeInfo,
) -> OpInvokeInfo.PerformanceProperties:
    """Cost of the V4 Flash lightning indexer score/topk core.

    By the time this op is emitted, the wrapper has already made these stages
    explicit in `DeepseekV4SparseAttentionIndexer.forward(...)`:
        q = wq_b(qa_normed) -> unflatten -> rope on q[..., -rd:]
        weights = weights_proj(x) * (head_dim**-0.5 * n_heads**-0.5)
        compressor(x, ...)  -> writes indexer_cache (separate trace event)

    The reference (ds-model-v4-flash inference/model.py:402-433) also performs
    `rotate_activation(q)` and `fp4_act_quant(q, fp4_block_size, True)` between
    the q-RoPE and the compressor write. tensor_cast has no standalone semantic
    op for either; the wrapper does not surface them in the trace, so their
    FLOPs/bytes are charged here as elementwise gp work over the full q tensor.

    This op therefore models, in reference order:
        rotate_activation(q)                            # gp over q
        fp4_act_quant(q)                                # gp over q
        local_score = einsum("bshd,btd->bsht", q, indexer_cache[:end_pos // ratio])
        local_score = (local_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)
        score = all_reduce_sum(local_score)             # when TP world_size > 1
        if prefill:
            score += where(causal_mask, -inf, 0)        # gp over score
        topk_idxs = topk(score, k=min(topk_limit, end_pos // ratio))
        if prefill:
            topk_idxs = where(validity_mask, -1, topk_idxs + offset)
        else:
            topk_idxs += offset
    The dominant FLOPs are the qK indexing einsum plus the weighted reduction.
    """
    # args: (q_states, weights, indexer_cache, topk_limit, tp_world_size, seq_lens, query_lens?)
    q_states = op_invoke_info.args[0]
    indexer_cache = op_invoke_info.args[2]
    topk_limit = int(op_invoke_info.args[3])
    tp_world_size = max(int(op_invoke_info.args[4]), 1)
    batch, seq_len, num_heads, head_dim = q_states.shape
    seq_lens = op_invoke_info.args[5]
    query_lens = op_invoke_info.args[6] if len(op_invoke_info.args) > 6 else None

    # Per-request prefill/decode split: each request contributes `query_lens_i`
    # rows of mma + reduction. Prefill rows additionally pay mask-add and
    # validity-mask topk postprocess. Without query_lens metadata we
    # conservatively assume the whole packed seq is prefill.
    seq_lens_list: Optional[List[int]] = None
    query_lens_list: Optional[List[int]] = None
    if (
        seq_lens is not None
        and not is_fake(seq_lens)
        and getattr(seq_lens, "device", None) is not None
        and seq_lens.device.type != "meta"
    ):
        try:
            seq_lens_list = [int(v) for v in seq_lens.tolist()]
        except Exception:
            seq_lens_list = None
    if (
        query_lens is not None
        and not is_fake(query_lens)
        and getattr(query_lens, "device", None) is not None
        and query_lens.device.type != "meta"
    ):
        try:
            query_lens_list = [int(v) for v in query_lens.tolist()]
        except Exception:
            query_lens_list = None

    cache_capacity = int(indexer_cache.shape[-2])
    if seq_lens_list is not None and query_lens_list is not None and len(seq_lens_list) == len(query_lens_list):
        request_cache_work = []
        for total_seq_len, query_len in zip(seq_lens_list, query_lens_list):
            if query_len <= 0:
                continue
            active_len = max(1, min(cache_capacity, total_seq_len // 4))
            topk_w = max(1, min(topk_limit, active_len))
            request_cache_work.append((total_seq_len, query_len, active_len, topk_w))
        if not request_cache_work:
            active_len = max(1, min(cache_capacity, seq_len))
            topk_w = max(1, min(topk_limit, active_len))
            request_cache_work = [(batch * seq_len, batch * seq_len, active_len, topk_w)]
    else:
        max_seq_len = _safe_max_int(seq_lens)
        active_cache_len_guess = max(max_seq_len // 4, 1) if max_seq_len is not None else seq_len
        active_len = max(1, min(cache_capacity, active_cache_len_guess))
        topk_w = max(1, min(topk_limit, active_len))
        request_cache_work = [(batch * seq_len, batch * seq_len, active_len, topk_w)]

    score_pair_count = sum(q * active_len for _, q, active_len, _ in request_cache_work)
    prefill_score_pair_count = sum(q * active_len for s, q, active_len, _ in request_cache_work if s == q)
    topk_work = sum(q * topk_w for _, q, _, topk_w in request_cache_work)
    # FIXED: prefill `where(validity_mask, -1, topk_idxs + offset)` is:
    #   1) topk_idxs + offset (1 add)
    #   2) compare(mask, ...) (1 compare)
    #   3) select(-1 or result) (1 select)
    #   ~3 ops per element (was 3, close enough). decode is still 1 add.
    topk_postprocess_gp = sum(q * topk_w * (3 if s == q else 1) for s, q, _, topk_w in request_cache_work)

    properties = op_invoke_info.get_memory_access_properties(exclude_input_ids={2})

    # Reference q-side stages between RoPE and the compressor write:
    #   q = rotate_activation(q)            -- elementwise Hadamard-style rotation
    #   fp4_act_quant(q, fp4_block_size, True) -- elementwise blockwise FP4 quant
    # Both are pointwise over the full (batch, seq_len, num_heads, head_dim) q tensor.
    q_elements = batch * seq_len * num_heads * head_dim
    log2_head_dim = max(int(math.log2(max(head_dim, 1))), 1)
    rotate_activation_gp = q_elements * (log2_head_dim + 1)
    # FIXED: fp4_act_quant (kernel.py:129-183) does: reduce_absmax +
    # fast_round_scale (bit ops: log2_ceil + pow2) + clamp + cast + mul.
    # ~5-6 ops/elem (was 1, severely undercounted).
    fp4_act_quant_gp = q_elements * 5

    # qK score einsum across the active compressed-cache prefix.
    qk_score_mma = batch * num_heads * head_dim * score_pair_count * 2
    score_reduce_gp = batch * num_heads * score_pair_count * 3
    score_elements_total = batch * score_pair_count
    if tp_world_size > 1 and num_heads > 0:
        all_reduce_score_bytes = score_elements_total * q_states.element_size()
        properties.memory_readwrite_bytes += all_reduce_score_bytes * 2
    # Prefill-only mask-add: applied per-request to its own `query_len_i` rows.
    score_mask_gp = batch * prefill_score_pair_count * 3
    topk_gp = batch * topk_work
    # Postprocess on topk indices: prefill performs `where(mask, -1, topk_idxs + offset)`,
    # decode performs `topk_idxs += offset`.
    topk_postprocess_gp *= batch
    _accumulate_compute_ops(
        properties,
        q_states.dtype,
        mma_ops=qk_score_mma,
        gp_ops=(
            rotate_activation_gp + fp4_act_quant_gp + score_reduce_gp + score_mask_gp + topk_gp + topk_postprocess_gp
        ),
    )
    return properties


@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.v4_clamped_swiglu.default)
def _(
    op_invoke_info: OpInvokeInfo,
) -> OpInvokeInfo.PerformanceProperties:
    gate, up, _ = op_invoke_info.args
    properties = op_invoke_info.get_memory_access_properties()
    dtype = gate.dtype if gate.dtype == up.dtype else torch.float32
    numel = up.numel()
    # FIXED: clamp(gate) [1 compare + 1 select] + clamp(up) [1 compare + 1 select]
    # + SiLU(gate) [sigmoid ~5 + mul = ~6] + multiply = 2+2+6+1 = ~11 ops/elem.
    _accumulate_compute_ops(properties, dtype, gp_ops=numel * 11)
    return properties


@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.moe_gating_top_k_hash.default)
def _(
    op_invoke_info: OpInvokeInfo,
) -> OpInvokeInfo.PerformanceProperties:
    """Cost of the post-score hash-routing tail.

    The gate matmul and score function are now standalone ops billed by
    their own handlers. This op only covers hash-table expert lookup,
    weight gather from the pre-bias scores, optional normalize and
    route-scale.
    """
    scores = op_invoke_info.args[0]
    top_k = int(op_invoke_info.args[1])
    normalize_weights = bool(op_invoke_info.args[2]) if len(op_invoke_info.args) > 2 else True
    num_experts = int(scores.shape[-1])
    num_tokens = scores.numel() // max(num_experts, 1)
    properties = op_invoke_info.get_memory_access_properties()
    hash_lookup_gp = num_tokens * top_k  # gather from tid2eid table
    weight_gather_gp = num_tokens * top_k  # gather from scores
    normalize_gp = num_tokens * top_k * 2 if normalize_weights else 0  # sum + div
    scale_gp = num_tokens * top_k  # route_scale multiply
    _accumulate_compute_ops(
        properties,
        scores.dtype,
        gp_ops=hash_lookup_gp + weight_gather_gp + normalize_gp + scale_gp,
    )
    return properties


@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.moe_gating_top_k.default)
def _(
    op_invoke_info: OpInvokeInfo,
) -> OpInvokeInfo.PerformanceProperties:
    """Cost of the post-score topk routing tail (V4 non-hash MoE).

    Mirrors the hash variant but replaces the hash-table lookup with the
    topk cost (and an optional bias-add prior to topk). Reference path:
    Gate.forward (model.py:572-583) for non-hash layers.
    """
    scores = op_invoke_info.args[0]
    top_k = int(op_invoke_info.args[1])
    normalize_weights = bool(op_invoke_info.args[2]) if len(op_invoke_info.args) > 2 else True
    bias = op_invoke_info.args[4] if len(op_invoke_info.args) > 4 else None
    num_experts = int(scores.shape[-1])
    num_tokens = scores.numel() // max(num_experts, 1)
    properties = op_invoke_info.get_memory_access_properties()
    bias_add_gp = num_tokens * num_experts if bias is not None else 0
    # topk over num_experts: model as O(num_experts) per token (selection net /
    # bitonic sort tail); k is small relative to num_experts.
    topk_gp = num_tokens * num_experts
    weight_gather_gp = num_tokens * top_k  # gather from pre-bias scores
    normalize_gp = num_tokens * top_k * 2 if normalize_weights else 0
    scale_gp = num_tokens * top_k
    _accumulate_compute_ops(
        properties,
        scores.dtype,
        gp_ops=bias_add_gp + topk_gp + weight_gather_gp + normalize_gp + scale_gp,
    )
    return properties


@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.sparse_attn_sharedkv.default)
def _(
    op_invoke_info: OpInvokeInfo,
) -> OpInvokeInfo.PerformanceProperties:
    """sparse_attn cost aligned to kernel.py:277-368 (block=64 online softmax)."""
    # args: (q, kv, attn_sink, topk_indices, softmax_scale, head_dim)
    q = op_invoke_info.args[0]
    kv = op_invoke_info.args[1]
    attn_sink = op_invoke_info.args[2]
    topk_indices = op_invoke_info.args[3]
    v_head_dim = int(op_invoke_info.args[5])

    # `sparse_attn` pads h<16 to 16 (kernel.py:359-362).
    raw_num_heads = int(q.size(2))
    num_heads = max(raw_num_heads, 16)
    padded_head_delta = num_heads - raw_num_heads
    q_head_dim = int(q.size(3))
    query_tokens = int(q.size(0) * q.size(1))
    sparse_topk = int(topk_indices.shape[-1])
    block = 64
    num_iters = (sparse_topk + block - 1) // block
    padded_topk = num_iters * block
    # Total active context = sum over iters; padded iters still pay full block.
    pipelined_ctx = padded_topk
    context_sum = query_tokens * pipelined_ctx

    properties = op_invoke_info.get_memory_access_properties(exclude_input_ids={1})
    # Two GEMMs per iter (Q*K^T and S*V), both [h, block]x[block, d].
    mma_ops = context_sum * num_heads * q_head_dim * 2 + context_sum * num_heads * v_head_dim * 2
    # Per-iter scalar work (kernel.py:321-343):
    #   acc_s init + scale + reduce_max + exp(score-max) + reduce_sum +
    #   sum_exp update + acc_o scale + final q/sink tail.
    #   Reduction ops are charged as (block - 1) comparisons/adds per head.
    per_iter_gp = (
        num_heads * block * 2  # acc_s init + acc_s *= scale
        + num_heads * (block - 1) * 2  # reduce_max
        + num_heads  # exp(prev-cur)
        + num_heads * block  # exp(score-max)
        + num_heads * (block - 1) * 2  # reduce_sum
        + num_heads * 2  # sum_exp = sum_exp*scale + scores_sum
        + num_heads * v_head_dim  # acc_o *= scores_scale
    )
    # Final per (batch, query) (kernel.py:345-350): sink exp + divide.
    per_query_gp = num_heads + num_heads * v_head_dim
    gp_ops = query_tokens * (num_iters * per_iter_gp + per_query_gp)

    _accumulate_compute_ops(properties, q.dtype, mma_ops=mma_ops, gp_ops=gp_ops)

    # Q/attn_sink padding traffic when the kernel internally pads h<16.
    if padded_head_delta > 0:
        properties.memory_read_bytes += query_tokens * padded_head_delta * q_head_dim * q.element_size()
        properties.memory_read_bytes += padded_head_delta * attn_sink.element_size()
        properties.memory_write_bytes += query_tokens * padded_head_delta * q_head_dim * q.element_size()
    # KV gather: each active idx reads one `d`-wide row; masked slots still
    # consume topk index bandwidth up to the padded block count.
    kv_row_bytes = bytes_of_elements(kv.size(-1), kv.dtype)
    properties.memory_read_bytes += query_tokens * sparse_topk * kv_row_bytes
    properties.memory_read_bytes += query_tokens * padded_topk * 4
    return properties
