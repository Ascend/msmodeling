import dataclasses
from typing import Optional, TYPE_CHECKING

import torch

from .. import ops  # noqa: F401

if TYPE_CHECKING:
    from ..model_config import AttentionQuantConfig


# adapted from vLLM but trimmed to avoid redundancy
@dataclasses.dataclass
class AttentionMetadataBase:
    """Per-layer attention metadata"""

    query_start_loc: torch.Tensor
    """(batch_size + 1,), the start location of each request in query Tensor"""

    seq_lens: torch.Tensor
    """(batch_size,), the length of each request including both computed tokens
    and newly scheduled tokens"""

    query_lens: torch.Tensor
    """(batch_size,), the actual query length of each request"""

    block_table_tensor: Optional[torch.Tensor] = None
    """(batch_size, max_blocks_per_seq)"""
    slot_mapping: Optional[torch.Tensor] = None
    """(num_tokens,) The indices of the token slots that input tokens will be
    stored into."""


class AttentionBase(torch.nn.Module):
    attn_implmentation = None

    def __init__(self):
        super().__init__()
        self.quant_config: Optional[AttentionQuantConfig] = None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        kv_cache: Optional[torch.Tensor] = None,
        attention_meta: Optional[AttentionMetadataBase] = None,
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement forward().")


# adapted from vLLM
def flash_attention_forward(
    # Transformers args
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    **kwargs,
):
    attention_by_layers: Optional[dict[int, AttentionBase]] = kwargs.pop("attention_by_layers", None)
    is_vision_attention = False
    if attention_by_layers is None:
        # For VL models, the visual layer's attention_by_layers cannot be obtained from kwargs,
        # so it is retrieved from the module's _tensor_cast_context instead.
        is_vision_attention = True
        _tensor_cast_context = getattr(module, "_tensor_cast_context", None)
        if _tensor_cast_context is not None:
            attention_by_layers = _tensor_cast_context.get("attention_by_layers")

    assert attention_by_layers is not None, "Expect attention_by_layers to be provided."
    if is_vision_attention:
        assert _tensor_cast_context is not None, "Expect _tensor_cast_context to be provided."
        depth_layer_idx = _tensor_cast_context.get("depth_layer_idx")
        self_attn = attention_by_layers[depth_layer_idx]
        kv_cache = None
        attention_meta = None
        query, key, value = (x.transpose(1, 2) for x in (query, key, value))
        num_tokens = query.shape[0] * query.shape[1]
        # For subsequent time calculation, the key and value do not need to be reshaped
        query = query.reshape(num_tokens, -1)
    else:
        kv_cache_by_layers: Optional[dict[int, torch.Tensor]] = kwargs.pop("kv_cache_by_layers", None)
        attention_meta: AttentionMetadataBase = kwargs.pop("attention_meta", None)
        attention_meta_by_layers: Optional[dict[int, AttentionMetadataBase]] = kwargs.pop(
            "attention_meta_by_layers", None
        )
        assert attention_meta is None or attention_meta_by_layers is None, (
            "Only one of attention_meta and attention_meta_by_layers can be provided."
        )

        self_attn = attention_by_layers[module.layer_idx]
        kv_cache = kv_cache_by_layers[module.layer_idx] if kv_cache_by_layers else None
        attention_meta = attention_meta_by_layers[module.layer_idx] if attention_meta_by_layers else attention_meta
        # TODO: understand why we need these shape manipulation
        query, key, value = (x.transpose(1, 2) for x in (query, key, value))
        num_tokens = query.shape[0] * query.shape[1]
        query, key, value = (x.reshape(num_tokens, -1) for x in (query, key, value))
    # return (attn_output, attn_weights) while we ignore attn_weights
    return self_attn.forward(
        query,
        key,
        value,
        attention_mask,
        kv_cache=kv_cache,
        attention_meta=attention_meta,
        **kwargs,
    ), None


class AttentionMetadataTensorCast(AttentionMetadataBase):
    pass


class AttentionTensorCast(AttentionBase):
    attn_implmentation = "tensor_cast"

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        kv_cache: Optional[torch.Tensor] = None,
        attention_meta: Optional[AttentionMetadataBase] = None,
        **kwargs,
    ) -> torch.Tensor:
        query_start_loc = attention_meta.query_start_loc if attention_meta else None
        seq_lens = attention_meta.seq_lens if attention_meta else None
        query_lens = attention_meta.query_lens if attention_meta else None
        if attention_meta is not None:
            if self.quant_config is not None:
                kv_scale = self.quant_config.kv_scale
                kv_offset = self.quant_config.kv_offset
                key = torch.ops.tensor_cast.quantize(key, kv_scale, kv_offset, kv_cache.dtype)
                value = torch.ops.tensor_cast.quantize(value, kv_scale, kv_offset, kv_cache.dtype)
            if not (key.dtype == value.dtype == kv_cache.dtype):
                raise ValueError(
                    f"Expect key, value and kv_cache dtype match but got {key.dtype}, {value.dtype}, {kv_cache.dtype}"
                )
            torch.ops.tensor_cast.reshape_and_cache(key, value, kv_cache, attention_meta.slot_mapping)
            key = kv_cache[0]
            value = kv_cache[1]
        if self.quant_config is not None and attention_meta is not None:
            out_dtype = query.dtype
            query = torch.ops.tensor_cast.quantize(
                query,
                self.quant_config.query_scale,
                self.quant_config.query_offset,
                kv_cache.dtype,
            )
            return torch.ops.tensor_cast.attention_quant(
                query,
                key,
                value,
                attention_mask,
                attention_meta.block_table_tensor if attention_meta is not None else None,
                query_start_loc,
                seq_lens,
                query_lens,
                self.quant_config.query_scale,
                self.quant_config.query_offset,
                self.quant_config.kv_scale,
                self.quant_config.kv_offset,
                self.quant_config.attention_prob_scale,
                self.quant_config.attention_prob_offset,
                out_dtype,
            )
        else:
            return torch.ops.tensor_cast.attention(
                query,
                key,
                value,
                attention_mask,
                attention_meta.block_table_tensor if attention_meta is not None else None,
                query_start_loc,
                seq_lens,
                query_lens,
            )
