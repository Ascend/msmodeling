import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

WrappedForwardFactory = Callable[[Callable[..., Any]], Callable[..., Any]]
MakeWrappedForward = Callable[[Any], WrappedForwardFactory]
BlockSetter = Callable[[Any], None]
GetBlocksWithSetters = Callable[[Any], Sequence[Tuple[Any, BlockSetter]]]


@dataclass(frozen=True)
class DiTBlockCacheSpec:
    model_type: str
    get_blocks_with_setters: GetBlocksWithSetters
    make_wrapped_forward: MakeWrappedForward


_DIT_BLOCK_CACHE_SPECS: Dict[str, DiTBlockCacheSpec] = {}


def register_dit_block_cache_spec(class_name: str, spec: DiTBlockCacheSpec) -> None:
    if not class_name:
        raise ValueError("'class_name' must be a non-empty string.")
    _DIT_BLOCK_CACHE_SPECS[class_name] = spec


def get_dit_block_cache_spec(class_name: Optional[str]) -> Optional[DiTBlockCacheSpec]:
    if not class_name:
        return None
    return _DIT_BLOCK_CACHE_SPECS.get(class_name)


def _make_block_setter(container, index: int) -> BlockSetter:
    def _set_block(new_block: Any) -> None:
        container[index] = new_block

    return _set_block


def _module_list_blocks_with_setters(blocks) -> list[tuple[Any, BlockSetter]]:
    return [(block, _make_block_setter(blocks, idx)) for idx, block in enumerate(blocks)]


def replace_blocks_in_range(
    blocks_with_setters: Sequence[Tuple[Any, BlockSetter]],
    start: int,
    end: int,
    make_cache_block: Callable[[Any, int], Any],
) -> int:
    from .cache_agent.dit_block_cache import DiTBlockCache

    replaced = 0
    bounded_end = min(end, len(blocks_with_setters))
    for flat_idx in range(start, bounded_end):
        block, setter = blocks_with_setters[flat_idx]
        if isinstance(block, DiTBlockCache):
            continue
        setter(make_cache_block(block, flat_idx))
        replaced += 1
    return replaced


def _get_wan_blocks_with_setters(inner: Any) -> Sequence[Tuple[Any, BlockSetter]]:
    if not hasattr(inner, "blocks"):
        logger.warning("WanTransformer3DModel has no attribute 'blocks'.")
        return []
    pairs = _module_list_blocks_with_setters(inner.blocks)
    if not pairs:
        logger.warning("WanTransformer3DModel.blocks is empty.")
    return pairs


def _get_hunyuanvideo_blocks_with_setters(
    inner: Any,
) -> Sequence[Tuple[Any, BlockSetter]]:
    if not hasattr(inner, "transformer_blocks"):
        logger.warning("HunyuanVideoTransformer3DModel has no attribute 'transformer_blocks'.")
        return []
    if not hasattr(inner, "single_transformer_blocks"):
        logger.warning("HunyuanVideoTransformer3DModel has no attribute 'single_transformer_blocks'.")
        return []
    pairs = _module_list_blocks_with_setters(inner.transformer_blocks)
    pairs.extend(_module_list_blocks_with_setters(inner.single_transformer_blocks))
    if not pairs:
        logger.warning("HunyuanVideoTransformer3DModel transformer blocks are empty.")
    return pairs


def _get_hunyuanvideo15_blocks_with_setters(
    inner: Any,
) -> Sequence[Tuple[Any, BlockSetter]]:
    if not hasattr(inner, "transformer_blocks"):
        logger.warning("HunyuanVideo15Transformer3DModel has no attribute 'transformer_blocks'.")
        return []
    pairs = _module_list_blocks_with_setters(inner.transformer_blocks)
    if not pairs:
        logger.warning("HunyuanVideo15Transformer3DModel.transformer_blocks is empty.")
    return pairs


def _wan_make_wrapped_forward(agent: Any) -> WrappedForwardFactory:
    def _make_wrapped_forward(
        orig_forward_bound: Callable[..., Any],
    ) -> Callable[..., Any]:
        def _wrapped_forward(
            _self_block: Any,
            hidden_states: Any,
            encoder_hidden_states: Any,
            temb: Any,
            rotary_emb: Any,
        ) -> Any:
            return agent.apply(
                orig_forward_bound,
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                rotary_emb=rotary_emb,
            )

        return _wrapped_forward

    return _make_wrapped_forward


def _hunyuanvideo_make_wrapped_forward(agent: Any) -> WrappedForwardFactory:
    def _make_wrapped_forward(
        orig_forward_bound: Callable[..., Any],
    ) -> Callable[..., Any]:
        def _wrapped_forward(
            _self_block: Any,
            hidden_states: Any,
            encoder_hidden_states: Any,
            temb: Any,
            attention_mask: Any,
            image_rotary_emb: Any,
            token_replace_emb: Any,
            first_frame_num_tokens: Any,
        ) -> Any:
            return agent.apply(
                orig_forward_bound,
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                attention_mask=attention_mask,
                freqs_cis=image_rotary_emb,
                token_replace_emb=token_replace_emb,
                first_frame_num_tokens=first_frame_num_tokens,
            )

        return _wrapped_forward

    return _make_wrapped_forward


def _hunyuanvideo15_make_wrapped_forward(agent: Any) -> WrappedForwardFactory:
    def _make_wrapped_forward(
        orig_forward_bound: Callable[..., Any],
    ) -> Callable[..., Any]:
        def _wrapped_forward(
            _self_block: Any,
            hidden_states: Any,
            encoder_hidden_states: Any,
            temb: Any,
            encoder_attention_mask: Any,
            image_rotary_emb: Any,
        ) -> Any:
            return agent.apply(
                orig_forward_bound,
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                attention_mask=encoder_attention_mask,
                freqs_cis=image_rotary_emb,
            )

        return _wrapped_forward

    return _make_wrapped_forward


register_dit_block_cache_spec(
    "WanTransformer3DModel",
    DiTBlockCacheSpec(
        model_type="Wan",
        get_blocks_with_setters=_get_wan_blocks_with_setters,
        make_wrapped_forward=_wan_make_wrapped_forward,
    ),
)
register_dit_block_cache_spec(
    "HunyuanVideoTransformer3DModel",
    DiTBlockCacheSpec(
        model_type="HunyuanVideo",
        get_blocks_with_setters=_get_hunyuanvideo_blocks_with_setters,
        make_wrapped_forward=_hunyuanvideo_make_wrapped_forward,
    ),
)
register_dit_block_cache_spec(
    "HunyuanVideo15Transformer3DModel",
    DiTBlockCacheSpec(
        model_type="HunyuanVideo15",
        get_blocks_with_setters=_get_hunyuanvideo15_blocks_with_setters,
        make_wrapped_forward=_hunyuanvideo15_make_wrapped_forward,
    ),
)
