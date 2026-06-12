import math
from typing import Optional

import torch

from ..utils import exact_division


def _get_sharded_shape(shape: torch.Tensor, dim: int, block_size: int):
    sharded_shape = list(shape)
    sharded_shape[dim] = block_size
    return sharded_shape


def _get_partial_sharded_padding(tensor: torch.Tensor, world_size: int, rank: int, dim: int = 0):
    """
    Splits a PyTorch tensor along a specified dimension into shards across multiple processes,
    with padding applied to ensure all shards have uniform size (using zeros for padding).

    Args:
        tensor: Input tensor to be sharded and padded.
        world_size: Total number of processes (total shards to split into).
        rank: Index of current process (determines which shard to return).
        dim: Dimension along which to shard. Must be 0, -1, or the last dimension (tensor.dim() - 1).

    Returns:
        torch.Tensor:
            Padded shard of the input tensor corresponding to the current rank,with uniform size across all processes.
    """
    assert dim in [0, -1, tensor.dim() - 1]

    size = tensor.shape[dim]
    block_size = math.ceil(size / world_size)

    start = rank * block_size
    stop = (rank + 1) * block_size

    if dim == 0:
        tensor = tensor[start:stop]
    else:
        tensor = tensor[..., start:stop]

    sharded_shape = _get_sharded_shape(tensor.shape, dim, block_size)
    tensor_zeros = torch.zeros(size=sharded_shape, dtype=tensor.dtype, device=tensor.device)
    if dim == 0:
        tensor_zeros[: tensor.shape[0]] = tensor
    else:
        tensor_zeros[..., : tensor.shape[-1]] = tensor

    return tensor_zeros


def _get_partial_sharded_by_unit(tensor: torch.Tensor, world_size: int, rank: int, dim: int = 0, unit_size: int = 1):
    """
    Splits a PyTorch tensor along a specified dimension into shards across multiple processes,
    with alignment to fixed-size units to ensure shards don't split units.

    Args:
        tensor: Input tensor to be sharded.
        world_size: Total number of processes (total shards to split into).
        rank: Index of current process (determines which shard to return).
        dim: Dimension along which to shard. Must be 0, -1, or the last dimension (tensor.dim() - 1).
        unit_size: Size of the fixed unit for alignment. Shard boundaries will always align with these units.

    Returns:
        torch.Tensor: Shard of the input tensor corresponding to the current rank.
    """
    assert dim in [0, -1, tensor.dim() - 1]

    size = tensor.shape[dim]
    unit_num = size // unit_size
    if unit_num >= world_size:
        block_size = size // world_size
        start = rank * block_size
        stop = (rank + 1) * block_size
    else:
        start = (rank // (world_size // unit_num)) * unit_size
        stop = ((rank // (world_size // unit_num)) + 1) * unit_size

    if dim == 0:
        tensor = tensor[start:stop]
    else:
        tensor = tensor[..., start:stop]

    return tensor


def get_partial_sharded(
    tensor: torch.Tensor,
    world_size: int,
    rank: int,
    dim: int = 0,
    unit_num: Optional[int] = None,
):
    size = tensor.shape[dim]
    if unit_num is not None:
        unit_size = exact_division(size, unit_num)
        if unit_num % world_size == 0 or world_size % unit_num == 0:
            return _get_partial_sharded_by_unit(tensor, world_size, rank, dim, unit_size)
        else:
            raise ValueError(
                f"The scenario where unit_num {unit_num} does not divide world_size {world_size}"
                f"and world_size {world_size} does not divide unit_num {unit_num} is not supported."
            )
    else:
        return _get_partial_sharded_padding(tensor, world_size, rank, dim)


def apply_static_quant_linear(x: torch.Tensor, module: torch.nn.Module) -> torch.Tensor:
    """Invoke a Linear-like module, preferring its statically-quantized inner path.

    Contract:
        ``module`` is expected to be either a plain ``torch.nn.Module`` (typically
        ``torch.nn.Linear`` or a quantized linear that exposes ``qweight``), or a
        :class:`ModelWrapperBase` whose ``_inner`` attribute holds such a module.
        No other wrapper conventions are supported.

    Behavior:
        * If the (optionally unwrapped) target carries a ``qweight`` attribute, it
          is treated as a statically-quantized linear and invoked directly so the
          quantized kernel is exercised without any wrapper-side dispatch.
        * Otherwise, the original ``module`` is called so wrapper-side logic such
          as dtype casting or hooks is preserved.
    """
    if isinstance(module, ModelWrapperBase):
        target = module._inner
    else:
        target = module
    qweight = getattr(target, "qweight", None)
    if qweight is None:
        return module(x)
    return target(x)


class ModelWrapperBase(torch.nn.Module):
    def __init__(self, wrapped: Optional[torch.nn.Module]):
        super().__init__()
        self._inner = wrapped

    def unwrap(self) -> torch.nn.Module:
        wrapped = self
        while isinstance(wrapped, ModelWrapperBase):
            wrapped = wrapped._inner
        return wrapped

    def __getattr__(self, item):
        try:
            return super().__getattr__(item)
        except AttributeError:
            if hasattr(self._inner, item):
                return getattr(self._inner, item)
            raise
