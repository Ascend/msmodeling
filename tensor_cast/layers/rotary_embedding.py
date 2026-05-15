from typing import Optional, Tuple, Union

import torch

from ..patch_torch import support_autocast_for_meta


class CachingRotaryEmb(torch.nn.Module):
    """
    Cache the position embeddings so that we can do quick index_select without
    computing them again and again in each forward.

    cos and sin are stored separately to align with NPU profiling shapes.
    NPU stores cos/sin as (max_pos, rope_dim) each; previous implementation
    concatenated them into (max_pos, 2*rope_dim) which produced an
    aten.index.Tensor shape that never appears in profiling data.
    """

    def __init__(
        self,
        rotary_emb: torch.nn.Module,
        act_dtype: torch.dtype,
        max_position_embeddings: int,
        expand_to_3d_position_ids: bool = False,
    ):
        super().__init__()
        self.act_dtype = act_dtype
        self.use_3d_position_index = False
        x = torch.empty(max_position_embeddings, device="meta", dtype=act_dtype).unsqueeze(0)
        position_ids = torch.arange(0, max_position_embeddings, device="meta", dtype=torch.long).unsqueeze(0)
        if expand_to_3d_position_ids:
            # Expand to (3, 1, max_position_embeddings) for T/H/W dimensions
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        with support_autocast_for_meta():
            position_embeddings = rotary_emb(x, position_ids)
        self.cos_cache: Optional[torch.Tensor]
        self.sin_cache: Optional[torch.Tensor]
        if isinstance(position_embeddings, (tuple, list)) and len(position_embeddings) == 2:
            cos, sin = position_embeddings
            cos = cos.squeeze()
            sin = sin.squeeze()
            if expand_to_3d_position_ids and cos.ndim == 3:
                self.use_3d_position_index = True
            self.register_buffer("cos_cache", cos, persistent=False)
            self.register_buffer("sin_cache", sin, persistent=False)
        else:
            self.cos_cache = None
            self.sin_cache = None
            self.rotary_emb = rotary_emb

    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        if self.cos_cache is not None and x.dtype == self.act_dtype:
            if self.use_3d_position_index:
                batch_idx = torch.arange(position_ids.size(0), device=position_ids.device)[:, None, None]
                return (
                    self.cos_cache[batch_idx, position_ids],
                    self.sin_cache[batch_idx, position_ids],
                )

            if position_ids.ndim == 3:
                # position_ids is (3, batch, seq_len) for multimodal; use text positions [0]
                position_ids = position_ids[0]
            flat_ids = position_ids.flatten()
            cos = self.cos_cache.index_select(0, flat_ids).reshape(position_ids.size(0), -1, self.cos_cache.size(-1))
            sin = self.sin_cache.index_select(0, flat_ids).reshape(position_ids.size(0), -1, self.sin_cache.size(-1))
            return cos, sin
        else:
            return self.rotary_emb(x, position_ids)
