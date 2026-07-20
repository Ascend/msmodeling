from .mla import DeepseekSparseAttention

_INDEXER_FULL = "full"
_INDEXER_SHARED = "shared"


def get_glm5_indexer_types(config) -> list[str] | None:
    indexer_types = getattr(config, "indexer_types", None)
    if not isinstance(indexer_types, list) or not indexer_types:
        return None
    return indexer_types


def glm5_uses_indexshare(config) -> bool:
    indexer_types = get_glm5_indexer_types(config)
    return indexer_types is not None and _INDEXER_SHARED in indexer_types


def resolve_glm5_indexer_source_layer(indexer_types: list[str] | None, layer_idx: int) -> int:
    if not indexer_types:
        return layer_idx
    if layer_idx >= len(indexer_types):
        raise ValueError(f"GLM5 indexer_types has {len(indexer_types)} entries, cannot resolve layer {layer_idx}")
    for idx, indexer_type in enumerate(indexer_types):
        if indexer_type not in (_INDEXER_FULL, _INDEXER_SHARED):
            raise ValueError(f"Unsupported GLM5 indexer type '{indexer_type}' at layer {idx}")
    if _INDEXER_SHARED not in indexer_types:
        return layer_idx

    indexer_type = indexer_types[layer_idx]
    if indexer_type == _INDEXER_FULL:
        return layer_idx
    if indexer_type != _INDEXER_SHARED:
        raise ValueError(f"Unsupported GLM5 indexer type '{indexer_type}' at layer {layer_idx}")

    for source_idx in range(layer_idx - 1, -1, -1):
        if indexer_types[source_idx] == _INDEXER_FULL:
            return source_idx
        if indexer_types[source_idx] != _INDEXER_SHARED:
            raise ValueError(f"Unsupported GLM5 indexer type '{indexer_types[source_idx]}' at layer {source_idx}")
    raise ValueError(f"GLM5 shared indexer layer {layer_idx} has no preceding full indexer layer")


def get_glm5_indexer_flow_flags(indexer_types: list[str] | None, layer_idx: int) -> tuple[bool, bool]:
    if not indexer_types or _INDEXER_SHARED not in indexer_types:
        return False, False

    source_layer_idx = resolve_glm5_indexer_source_layer(indexer_types, layer_idx)
    skip_topk = source_layer_idx != layer_idx
    next_layer_idx = layer_idx + 1
    next_skip_topk = False
    if next_layer_idx < len(indexer_types):
        resolve_glm5_indexer_source_layer(indexer_types, next_layer_idx)
        next_skip_topk = indexer_types[next_layer_idx] == _INDEXER_SHARED
    return skip_topk, next_skip_topk


def extend_glm5_indexer_types_for_mtp(indexer_types: list[str], num_mtp_layers: int) -> None:
    if num_mtp_layers <= 0 or not indexer_types:
        return
    if _INDEXER_SHARED not in indexer_types:
        indexer_types.extend([indexer_types[-1]] * num_mtp_layers)
        return

    # MTP proposal blocks run independently from the main decoder stack and
    # therefore cannot consume a main-stack prev_topk_indices output. Every
    # appended MTP block must compute its own indexer.
    for indexer_type in indexer_types:
        if indexer_type not in (_INDEXER_FULL, _INDEXER_SHARED):
            raise ValueError(f"Unsupported GLM5 indexer type '{indexer_type}'")
    indexer_types.extend([_INDEXER_FULL] * num_mtp_layers)


class Glm5SparseAttention(DeepseekSparseAttention):
    def _run_sparse_attention_indexer(
        self, hidden_states, qa_normed, position_embeddings, attention_meta=None, **kwargs
    ):
        source_layer_idx = getattr(self, "indexer_source_layer_idx", self.layer_idx)
        if source_layer_idx == self.layer_idx:
            return super()._run_sparse_attention_indexer(
                hidden_states,
                qa_normed,
                position_embeddings,
                attention_meta,
                **kwargs,
            )

        prev_topk_indices = kwargs.get("prev_topk_indices")
        if prev_topk_indices is None:
            raise ValueError(
                f"GLM5 shared indexer layer {self.layer_idx} missing prev_topk_indices from source layer {source_layer_idx}"
            )
        return prev_topk_indices

    def _format_forward_output(self, attn_output, attn_weights, pre_attn_out) -> tuple:
        attrs = vars(self)
        inner = attrs.get("_inner")
        modules = attrs.get("_modules")
        if inner is None and isinstance(modules, dict):
            inner = modules.get("_inner")
        next_skip_topk = attrs.get("next_skip_topk", False) or getattr(inner, "next_skip_topk", False)
        next_topk = pre_attn_out if next_skip_topk else None
        return attn_output, attn_weights, next_topk

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)
