import torch

from .op_invoke_info import OpInvokeInfo
from .utils import bytes_of_elements


def elementwise_sigmoid_ops(numel: int) -> int:
    return numel * 4


def elementwise_softplus_ops(numel: int) -> int:
    return numel * 4


def elementwise_silu_ops(numel: int) -> int:
    return numel * 6


def rmsnorm_ops(num_rows: int, row_width: int) -> int:
    """Return the FP32 GP estimate for RMSNorm."""
    return num_rows * row_width * 5


def l2norm_ops(num_rows: int, row_width: int) -> int:
    """Return the FP32 GP estimate for L2 normalization."""
    return num_rows * row_width * 4


def accumulate_compute_ops(
    properties: OpInvokeInfo.PerformanceProperties,
    dtype: torch.dtype,
    mma_ops: int = 0,
    gp_ops: int = 0,
) -> None:
    """Add compute work to performance properties without charging extra IO."""
    if mma_ops == 0 and gp_ops == 0:
        return
    delta = OpInvokeInfo.PerformanceProperties(
        compute_ops={
            dtype: OpInvokeInfo.ComputeOps(mma_ops=mma_ops, gp_ops=gp_ops),
        }
    )
    properties.combine(delta, compute_only=True)


def byte_count(num_elements: int, dtype: torch.dtype) -> int:
    return int(bytes_of_elements(num_elements, dtype))
