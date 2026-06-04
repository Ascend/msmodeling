from typing import Tuple

import torch

from ..utils import register_tensor_cast_op


@register_tensor_cast_op("apply_rope")
def _(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    q_embed, k_embed = torch.empty_like(query), torch.empty_like(key)
    q_embed = q_embed.transpose(1, 2)
    k_embed = k_embed.transpose(1, 2)
    return q_embed.contiguous(), k_embed.contiguous()


@register_tensor_cast_op("apply_rope_inplace", mutates_args=("x",))
def _(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox: bool = True,
    inverse: bool = False,
    rope_head_dim: int = -1,
) -> torch.Tensor:
    # In-place RoPE on the trailing `rope_head_dim` channels of x.
    # When rope_head_dim < 0, rotate the full last dimension.
    del cos, sin, is_neox, inverse, rope_head_dim
    return x
