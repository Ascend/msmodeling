import torch

from ..utils import register_tensor_cast_op


@register_tensor_cast_op("static_false_mask")
def _static_false_mask(mask: torch.Tensor) -> torch.Tensor:
    """Mark a Boolean mask whose elements are known to be false."""
    return mask.clone()


def static_false_mask(mask: torch.Tensor) -> torch.Tensor:
    return torch.ops.tensor_cast.static_false_mask.default(mask)
