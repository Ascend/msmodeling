import torch
import torch.fx as fx
from torch._inductor.pattern_matcher import Arg as _BaseArg, CallFunction, Match

from .... import config, ops  # noqa: F401


def _arg(name: str):
    try:
        return _BaseArg(name)
    except TypeError:
        del name
        return _BaseArg()


matmul_allreduce_pattern = CallFunction(
    torch.ops.tensor_cast.all_reduce.default,
    CallFunction(torch.ops.aten.mm.default, _arg("mat1"), _arg("mat2")),
    _arg("rank"),
    _arg("rank_group"),
)


static_quant_linear_allreduce_pattern = CallFunction(
    torch.ops.tensor_cast.all_reduce.default,
    CallFunction(
        torch.ops.tensor_cast.static_quant_linear.default,
        _arg("x"),
        _arg("w"),
        _arg("w_scale"),
        _arg("w_offset"),
        _arg("x_scale"),
        _arg("x_offset"),
        _arg("bias"),
        _arg("out_dtype"),
    ),
    _arg("rank"),
    _arg("rank_group"),
)


static_quant_linear_int4_allreduce_pattern = CallFunction(
    torch.ops.tensor_cast.all_reduce.default,
    CallFunction(
        torch.ops.tensor_cast.static_quant_linear_int4.default,
        _arg("x"),
        _arg("w"),
        _arg("w_scale"),
        _arg("w_offset"),
        _arg("x_scale"),
        _arg("x_offset"),
        _arg("bias"),
        _arg("out_dtype"),
    ),
    _arg("rank"),
    _arg("rank_group"),
)


fp8_linear_allreduce_pattern = CallFunction(
    torch.ops.tensor_cast.all_reduce.default,
    CallFunction(
        torch.ops.tensor_cast.fp8_linear.default,
        _arg("x"),
        _arg("w"),
        _arg("x_scale"),
        _arg("w_scale"),
        _arg("bias"),
        _arg("out_dtype"),
    ),
    _arg("rank"),
    _arg("rank_group"),
)


mxfp4_linear_allreduce_pattern = CallFunction(
    torch.ops.tensor_cast.all_reduce.default,
    CallFunction(
        torch.ops.tensor_cast.mxfp4_linear.default,
        _arg("x"),
        _arg("w"),
        _arg("x_scale"),
        _arg("w_scale"),
        _arg("bias"),
        _arg("out_dtype"),
    ),
    _arg("rank"),
    _arg("rank_group"),
)


def _is_allreduce_after_target(match: Match, linear_target) -> bool:
    allreduce_node = match.output_node()
    if allreduce_node.target != torch.ops.tensor_cast.all_reduce.default:
        return False
    if len(allreduce_node.args) != 3:
        return False
    linear_node = allreduce_node.args[0]
    if not isinstance(linear_node, fx.Node):
        return False
    if linear_node.target != linear_target:
        return False
    if len(linear_node.users) != 1:
        return False
    return True


def _is_matmul_allreduce(match: Match) -> bool:
    return _is_allreduce_after_target(match, torch.ops.aten.mm.default)


def _is_static_quant_linear_allreduce(match: Match) -> bool:
    return _is_allreduce_after_target(match, torch.ops.tensor_cast.static_quant_linear.default)


def _is_static_quant_linear_int4_allreduce(match: Match) -> bool:
    return _is_allreduce_after_target(match, torch.ops.tensor_cast.static_quant_linear_int4.default)


def _is_fp8_linear_allreduce(match: Match) -> bool:
    return _is_allreduce_after_target(match, torch.ops.tensor_cast.fp8_linear.default)


def _is_mxfp4_linear_allreduce(match: Match) -> bool:
    return _is_allreduce_after_target(match, torch.ops.tensor_cast.mxfp4_linear.default)


def _replace_with_fused_op(match: Match, fused_target, fused_args) -> None:
    graph = match.graph
    allreduce_node = match.output_node()
    with graph.inserting_before(allreduce_node):
        fused_node = graph.call_function(fused_target, args=fused_args)
    fused_node.meta.update(allreduce_node.meta)
    allreduce_node.replace_all_uses_with(fused_node)
    match.erase_nodes()


def _fuse_matmul_allreduce(match: Match, mat1, mat2, rank, rank_group):
    _replace_with_fused_op(
        match,
        torch.ops.tensor_cast.matmul_all_reduce,
        (mat1, mat2, None, rank, rank_group),
    )


def _fuse_static_quant_linear_allreduce(
    match: Match,
    x,
    w,
    w_scale,
    w_offset,
    x_scale,
    x_offset,
    bias,
    out_dtype,
    rank,
    rank_group,
):
    _replace_with_fused_op(
        match,
        torch.ops.tensor_cast.static_quant_linear_all_reduce,
        (x, w, w_scale, w_offset, x_scale, x_offset, bias, out_dtype, rank, rank_group),
    )


def _fuse_static_quant_linear_int4_allreduce(
    match: Match,
    x,
    w,
    w_scale,
    w_offset,
    x_scale,
    x_offset,
    bias,
    out_dtype,
    rank,
    rank_group,
):
    _replace_with_fused_op(
        match,
        torch.ops.tensor_cast.static_quant_linear_int4_all_reduce,
        (x, w, w_scale, w_offset, x_scale, x_offset, bias, out_dtype, rank, rank_group),
    )


def _fuse_fp8_linear_allreduce(
    match: Match,
    x,
    w,
    x_scale,
    w_scale,
    bias,
    out_dtype,
    rank,
    rank_group,
):
    _replace_with_fused_op(
        match,
        torch.ops.tensor_cast.fp8_linear_all_reduce,
        (x, w, x_scale, w_scale, bias, out_dtype, rank, rank_group),
    )


def _fuse_mxfp4_linear_allreduce(
    match: Match,
    x,
    w,
    x_scale,
    w_scale,
    bias,
    out_dtype,
    rank,
    rank_group,
):
    _replace_with_fused_op(
        match,
        torch.ops.tensor_cast.mxfp4_linear_all_reduce,
        (x, w, x_scale, w_scale, bias, out_dtype, rank, rank_group),
    )


def register_all_patterns():
    from . import register_pattern

    if not config.compilation.fusion_patterns.enable_matmul_allreduce:
        return

    register_pattern(
        name="matmul_allreduce_pattern",
        pattern=matmul_allreduce_pattern,
        handler=_fuse_matmul_allreduce,
        extra_check=_is_matmul_allreduce,
    )
    register_pattern(
        name="static_quant_linear_allreduce_pattern",
        pattern=static_quant_linear_allreduce_pattern,
        handler=_fuse_static_quant_linear_allreduce,
        extra_check=_is_static_quant_linear_allreduce,
    )
    register_pattern(
        name="static_quant_linear_int4_allreduce_pattern",
        pattern=static_quant_linear_int4_allreduce_pattern,
        handler=_fuse_static_quant_linear_int4_allreduce,
        extra_check=_is_static_quant_linear_int4_allreduce,
    )
    register_pattern(
        name="fp8_linear_allreduce_pattern",
        pattern=fp8_linear_allreduce_pattern,
        handler=_fuse_fp8_linear_allreduce,
        extra_check=_is_fp8_linear_allreduce,
    )
    register_pattern(
        name="mxfp4_linear_allreduce_pattern",
        pattern=mxfp4_linear_allreduce_pattern,
        handler=_fuse_mxfp4_linear_allreduce,
        extra_check=_is_mxfp4_linear_allreduce,
    )
