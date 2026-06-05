# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
"""
input_generation
"""

from dataclasses import dataclass
from importlib import import_module
from typing import Any, List, Tuple

import torch

from ..layers.attention import AttentionMetadataTensorCast
from ..layers.sampler import SamplingMetadata
from ..performance_model import bytes_of_tensor
from ..transformers.utils import get_attention_quant_config, logger
from ..utils import exact_division


@dataclass
class RequestInfo:
    query_len: int
    seq_len: int
    is_decode: bool = True
    context_length: int = 0
    num_input_tokens: int = None
    num_output_tokens: int = None
    concurrency: int = 1
    image_batch_size: int = None
    image_height: int = None
    image_width: int = None


def generate_inputs(model, requests: List[RequestInfo], block_size: int = 128):
    # TODO merge generate_inputs and generate_inputs_varlen
    # for now, unify the function signatures, Firstly.
    request = requests[0]
    concurrency = request.concurrency
    seq_len = request.seq_len
    query_len = request.query_len
    is_decode = request.is_decode
    image_kwargs = {}
    context_length = request.context_length
    if model.is_vl_model:
        image_kwargs = generate_image_inputs(
            model,
            request.image_batch_size,
            request.image_height,
            request.image_width,
            concurrency,
        )
        num_image_tokens = image_kwargs.pop("num_image_tokens", 0)
        seq_len += num_image_tokens
        if is_decode:
            # In the decode phase, the image input is removed, but the image token needs to be added to content_length
            image_kwargs = {}
        else:
            query_len += num_image_tokens
    else:
        if request.image_batch_size is not None or request.image_height is not None or request.image_width is not None:
            logger.warning("For non-VL models, the parameter input of the image is ignored")
    model_config = model.model_config
    num_mtp_tokens = model_config.mtp_config.num_mtp_layers if model_config.mtp_config else 0
    parallel_config = model_config.parallel_config
    batch_size = (concurrency + parallel_config.data_parallel_size - 1) // parallel_config.data_parallel_size

    max_context_length = seq_len + num_mtp_tokens + 1

    # Paged attention parameters (can be adjusted)
    num_blocks = (
        max_context_length * batch_size + block_size - 1
    ) // block_size  # Total number of blocks available in the KV cache

    # Prepare Attention Metadata for Paged Attention
    # `query_start_loc` indicates the start of each query in the concatenated input tensor.
    # Shape: [num_queries + 1] -> e.g., [0, 50, 100, 150] for 3 queries of length 50.
    query_start_loc = torch.arange(0, (batch_size + 1) * query_len, query_len, dtype=torch.long)

    # `seq_lens` is the total length (context + new tokens) for each sequence in the batch.
    seq_lens = torch.empty(batch_size, dtype=torch.long)
    seq_lens.fill_(seq_len)

    query_lens = torch.empty(batch_size, dtype=torch.long)
    query_lens.fill_(query_len)

    # `block_tables` map logical sequence blocks to physical blocks in the KV cache.
    max_num_blocks_per_seq = (seq_len + block_size - 1) // block_size

    block_table_tensor = torch.empty((batch_size, max_num_blocks_per_seq), dtype=torch.long, device="meta")

    slot_mapping = torch.empty((batch_size * query_len,), dtype=torch.long, device="meta")

    attn_meta = AttentionMetadataTensorCast(
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        query_lens=query_lens,
        block_table_tensor=block_table_tensor,
        slot_mapping=slot_mapping,
    )

    # The total number of new tokens to be processed in this batch, concatenated.
    # Note: Padding for TP/EP alignment has been moved to MoE layers
    # (see FusedMoETensorCast.forward() and ParallelMoELayer.forward())
    # to avoid inflating token counts for non-MoE operations.
    # This matches vLLM's behavior where scheduler handles global alignment
    # and grouped_matmul handles per-expert alignment internally.
    num_tokens = batch_size * query_len
    input_ids = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
    position_ids = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
    kv_cache_by_layers, kv_cache_per_token = _get_kv_cache_info(model, num_blocks, block_size)
    sampling_metadata = SamplingMetadata(
        query_start_loc=attn_meta.query_start_loc,
    )
    if is_decode:
        # do not prune logits
        sampling_metadata.selected_token_indices = None
    else:
        sampling_metadata.selected_token_indices = torch.arange(
            query_len - 1, batch_size * query_len, query_len, device="meta"
        )

    kwargs = {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "attention_meta": attn_meta,
        "kv_cache_by_layers": kv_cache_by_layers,
        "kv_cache_per_token": kv_cache_per_token,
        "sampling_metadata": sampling_metadata,
    }

    sparse_attention_indexer_cache = get_sparse_attention_indexer_cache_info(model, num_blocks, block_size)
    kwargs.update(sparse_attention_indexer_cache)

    if model.model_config.hf_config.model_type in (
        "qwen3_next",
        "qwen3_5",
        "qwen3_5_moe",
    ):
        kwargs["cache_position"] = torch.arange(
            context_length, context_length + num_tokens, dtype=torch.long, device="cpu"
        )
    kwargs.update(image_kwargs)
    return kwargs


def resize_image(
    model_id,
    model_type,
    image_height,
    image_width,
    patch_size,
    merge_size,
    temporal_patch_size,
):
    factor = patch_size * merge_size

    def build_qwen_resize_params():
        params = {
            "height": image_height,
            "width": image_width,
            "factor": factor,
        }
        min_pixels, max_pixels = _load_preprocessor_pixel_limits(model_id)
        if min_pixels is not None and max_pixels is not None:
            params["min_pixels"] = min_pixels
            params["max_pixels"] = max_pixels
        return params

    def build_glm_resize_params():
        return {
            "height": image_height,
            "width": image_width,
            "factor": factor,
            "num_frames": temporal_patch_size,
            "temporal_factor": temporal_patch_size,
        }

    resize_specs = {
        "glm4v_moe": (
            "transformers.models.glm4v.image_processing_glm4v",
            build_glm_resize_params,
        ),
        "qwen3_vl": (
            "transformers.models.qwen2_vl.image_processing_qwen2_vl",
            build_qwen_resize_params,
        ),
        "qwen3_vl_moe": (
            "transformers.models.qwen2_vl.image_processing_qwen2_vl",
            build_qwen_resize_params,
        ),
    }

    module_path, params_builder = resize_specs.get(model_type, resize_specs["qwen3_vl"])
    smart_resize = import_module(module_path).smart_resize
    return smart_resize(**params_builder())


def _load_preprocessor_pixel_limits(model_id: str):
    """
    Load image pixel limits from instantiated HF processor.
    """
    if not model_id:
        logger.warning("model_id is empty; falling back to smart_resize defaults.")
        return None, None

    try:
        from transformers import AutoImageProcessor

        image_processor = AutoImageProcessor.from_pretrained(model_id)
        size = getattr(image_processor, "size", None)
        if size is None or not hasattr(size, "get"):
            return None, None
        min_pixels = size.get("shortest_edge")
        max_pixels = size.get("longest_edge")
        return min_pixels, max_pixels
    except Exception:
        logger.warning(
            "Failed to load processor/image size for model_id=%s; falling back to smart_resize defaults.",
            model_id,
            exc_info=True,
        )
        return None, None


def generate_image_inputs(model, image_batch_size, image_height, image_width, concurrency):
    if image_batch_size is None or image_height is None or image_width is None:
        print("For vision-language models,without image input")
        return {}
    hf_config = model.model_config.hf_config
    vision_config = hf_config.vision_config
    patch_size = vision_config.patch_size
    merge_size = vision_config.spatial_merge_size or 2
    # Rescales the image
    temporal_patch_size = vision_config.temporal_patch_size or 2
    resized_height, resized_width = resize_image(
        getattr(model, "model_id", ""),
        hf_config.model_type,
        image_height,
        image_width,
        patch_size=patch_size,
        merge_size=merge_size,
        temporal_patch_size=temporal_patch_size,
    )

    # For images, the value of grid_t is 1.
    grid_t = 1
    grid_h, grid_w = resized_height // patch_size, resized_width // patch_size
    image_grid_thw = torch.tensor([[grid_t, grid_h, grid_w]], dtype=torch.long).expand(image_batch_size, 3)
    channel = vision_config.in_channels or 3
    hidden_dim = channel * temporal_patch_size * patch_size * patch_size
    tokens = grid_t * grid_h * grid_w
    pixel_values = torch.empty(
        image_batch_size * tokens,
        hidden_dim,
        dtype=model.model_config.dtype,
        device="meta",
    )
    # Calculate the token embedded in the text.
    merge_length = merge_size**2
    num_image_tokens = image_batch_size * (tokens // merge_length + 2)
    parallel_config = model.model_config.parallel_config
    batch_size = (concurrency + parallel_config.data_parallel_size - 1) // parallel_config.data_parallel_size
    pixel_values = pixel_values.repeat(batch_size, 1)
    image_grid_thw = image_grid_thw.repeat(batch_size, 1)
    return {
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
        "num_image_tokens": num_image_tokens,
    }


def _resolve_sparse_attention_kv_cache_width(model, attention_layer=None) -> int:
    """Resolve the per-token KV cache width for sparse-attention wrappers.

    This path covers standard MLA-like wrappers as well as V4's custom shared-KV
    sparse attention wrapper. Prefer runtime layer attributes when available,
    then fall back to the legacy DeepSeek MLA width formula.

    Args:
        model: The model wrapper.
        attention_layer: The attention layer instance, or None if decoder layers
            cannot be resolved. When None, falls back to
            ``model.text_config.kv_lora_rank + model.text_config.qk_rope_head_dim``.

    Returns:
        The per-token KV cache width in bytes.
    """
    if attention_layer is not None:
        for attr in ("_head_dim", "head_dim"):
            width = getattr(attention_layer, attr, None)
            if width is not None:
                return int(width)
    return int(model.text_config.kv_lora_rank + model.text_config.qk_rope_head_dim)


def _resolve_sparse_attention_indexer_cache_width(model, attention_layer) -> int | None:
    """Resolve the auxiliary indexer cache width for sparse-attention wrappers.

    For V4 ratio=4 layers this picks up the dedicated indexer-local head width
    (`index_head_dim`), which is intentionally different from the main KV cache
    width (`head_dim`).
    """
    for attr in ("_index_head_dim",):
        width = getattr(attention_layer, attr, None)
        if width is not None:
            return int(width)

    indexer = getattr(attention_layer, "indexer", None)
    if indexer is not None:
        width = getattr(indexer, "head_dim", None)
        if width is not None:
            return int(width)

    width = getattr(model.text_config, "index_head_dim", None)
    if width is not None:
        return int(width)
    return None


def _layer_uses_sparse_attention_indexer(attention_layer) -> bool:
    use_indexer = getattr(attention_layer, "use_indexer", None)
    if use_indexer is not None:
        return bool(use_indexer)
    return getattr(attention_layer, "indexer", None) is not None


def _resolve_decoder_attention_layer(layer):
    """Resolve a decoder layer's attention module through lightweight wrappers."""
    from ..layers.utils import ModelWrapperBase

    current = layer
    visited = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))

        attention_layer = getattr(current, "self_attn", None)
        if attention_layer is not None:
            while isinstance(attention_layer, ModelWrapperBase) and attention_layer._inner is not None:
                attention_layer = attention_layer._inner
            if isinstance(attention_layer, torch.nn.Module):
                nested_attention = attention_layer._modules.get("self_attn")
                if nested_attention is not None:
                    attention_layer = nested_attention
            return attention_layer

        representative = getattr(current, "representative", None)
        if representative is not None:
            current = representative
            continue

        current = current._inner if isinstance(current, ModelWrapperBase) else None

    return None


def _get_kv_cache_info(model, num_blocks: int, block_size: int) -> Tuple[dict[Any, Any], int]:
    model_config = model.model_config
    parallel_config = model.model_config.parallel_config
    decoder_layers = None
    if model_config.mla_config is not None:
        try:
            decoder_layers = _resolve_decoder_layers(model)
        except AttributeError:
            decoder_layers = None
    # Initialize the KV cache structure (also on 'meta' device).
    kv_cache_per_token = 0
    kv_cache_by_layers = {}
    for i in range(model.num_hidden_layers):
        kvcache_dtype = model_config.dtype
        if (attention_config := get_attention_quant_config(model, i)) is not None:
            kvcache_dtype = attention_config.get_quant_dtype()

        if model_config.mla_config is not None:
            # decoder_layers may be None if _resolve_decoder_layers raises
            # AttributeError (e.g., model not fully wrapped). In that case
            # attention_layer stays None and the fallback formula below is used.
            attention_layer = None
            if decoder_layers is not None and i < len(decoder_layers):
                attention_layer = _resolve_decoder_attention_layer(decoder_layers[i])
            kv_cache_width = _resolve_sparse_attention_kv_cache_width(
                model,
                attention_layer,
            )
            kv_cache_by_layers[i] = torch.empty(
                [
                    num_blocks,
                    block_size,
                    kv_cache_width,
                ],
                dtype=kvcache_dtype,
                device="meta",
            )
        else:
            # Shape: [2 (K/V), num_blocks, block_size, num_heads, head_dim]
            if model.text_config.num_key_value_heads >= parallel_config.tensor_parallel_size:
                kv_heads = exact_division(
                    model.text_config.num_key_value_heads,
                    parallel_config.tensor_parallel_size,
                )
            else:
                assert parallel_config.tensor_parallel_size % model.text_config.num_key_value_heads == 0
                kv_heads = 1

            kv_cache_by_layers[i] = torch.empty(
                [
                    2,
                    num_blocks,
                    block_size,
                    kv_heads,
                    model.head_dim,
                ],
                dtype=kvcache_dtype,
                device="meta",
            )
        kv_cache_per_token += bytes_of_tensor(kv_cache_by_layers[i]) / (num_blocks * block_size)
    return kv_cache_by_layers, kv_cache_per_token


def get_kv_cache_info(model, num_blocks, block_size):
    model_config = model.model_config
    tp_size = model_config.parallel_config.tensor_parallel_size
    decoder_layers = None
    if model_config.mla_config is not None:
        try:
            decoder_layers = _resolve_decoder_layers(model)
        except AttributeError:
            decoder_layers = None
    kv_cache_by_layers = {}
    kv_cache_per_token = 0
    for i in range(model.num_hidden_layers):
        kvcache_dtype = model_config.dtype
        attention_config = get_attention_quant_config(model, i)
        if attention_config is not None:
            kvcache_dtype = attention_config.get_quant_dtype()

        if model_config.mla_config is not None:
            # decoder_layers may be None if _resolve_decoder_layers raises
            # AttributeError (e.g., model not fully wrapped). In that case
            # attention_layer stays None and the fallback formula below is used.
            attention_layer = None
            if decoder_layers is not None and i < len(decoder_layers):
                attention_layer = _resolve_decoder_attention_layer(decoder_layers[i])
            kv_cache_width = _resolve_sparse_attention_kv_cache_width(
                model,
                attention_layer,
            )
            kv_cache_by_layers[i] = torch.empty(
                (
                    num_blocks,
                    block_size,
                    kv_cache_width,
                ),
                dtype=kvcache_dtype,
                device="meta",
            )
        else:
            n_kv = model.text_config.num_key_value_heads
            if n_kv >= tp_size:
                assert n_kv % tp_size == 0
                kv_heads = n_kv // tp_size
            else:
                assert tp_size % n_kv == 0
                kv_heads = 1
            kv_cache_by_layers[i] = torch.empty(
                (
                    2,
                    num_blocks,
                    block_size,
                    kv_heads,
                    model.head_dim,
                ),
                dtype=kvcache_dtype,
                device="meta",
            )
        kv_cache_per_token += bytes_of_tensor(kv_cache_by_layers[i]) / (num_blocks * block_size)

    return kv_cache_by_layers, kv_cache_per_token


def _resolve_decoder_layers(model):
    """Resolve the decoder layers ``ModuleList`` regardless of how the model
    is wrapped.

    msmodeling can wrap the underlying HF model in several layouts:
        * ``TransformerModel(_inner=CausalLmWrapper(_inner=HFModel))``
        * ``TransformerModel(_inner=ModelWrapper(_inner=HFModel))``
        * ``OptimizedModule(_orig_mod=TransformerModel(...))`` when --compile is on
        * ``MtpWrapper(_inner=...)`` when MTP is enabled

    Using ``model.model.layers`` works only when the deepest module follows the
    ``*ForCausalLM``-with-inner-``*Model`` layout. For modules registered via
    ``AutoModel.register(Cfg, *Model)`` (e.g. DeepseekV4Model), the deepest
    module IS the ``*Model`` itself and exposes ``.layers`` directly, so
    ``model.model`` raises AttributeError under torch.compile / dynamo.

    This helper peels off all known wrappers via ``model.unwrap()`` (when
    available) and then probes both layout variants.

    Returns:
        The decoder layers ModuleList.

    Raises:
        AttributeError: If neither ``unwrap().layers`` nor ``unwrap().model.layers``
            is available. Callers should catch this and fall back to the
            legacy formula (kv_lora_rank + qk_rope_head_dim).
    """
    inner = model.unwrap() if hasattr(model, "unwrap") else model
    if hasattr(inner, "layers"):
        return inner.layers
    nested = getattr(inner, "model", None)
    if nested is not None and hasattr(nested, "layers"):
        return nested.layers
    raise AttributeError(
        "Unable to locate decoder layers; neither `unwrap().layers` nor "
        "`unwrap().model.layers` is available on this model"
    )


def get_sparse_attention_indexer_cache_info(model, num_blocks, block_size):
    """Allocate per-layer auxiliary indexer caches for sparse-attention wrappers.

    Despite the older DSA-oriented naming in surrounding code, this helper is
    also used by DeepSeek V4's custom sparse attention path, whose ratio=4
    layers carry a distinct learned indexer.
    """
    model_config = model.model_config
    mla_config = model_config.mla_config
    if mla_config is None or not mla_config.mla_cls.requires_indexer_cache():
        return {}

    indexer_cache_by_layers = {}
    indexer_cache_per_token = 0
    try:
        decoder_layers = _resolve_decoder_layers(model)
    except AttributeError:
        decoder_layers = None
    for i in range(model.num_hidden_layers):
        attention_layer = (
            _resolve_decoder_attention_layer(decoder_layers[i])
            if decoder_layers is not None and i < len(decoder_layers)
            else None
        )
        if attention_layer is not None and not _layer_uses_sparse_attention_indexer(attention_layer):
            continue

        cache_width = _resolve_sparse_attention_indexer_cache_width(model, attention_layer)
        if cache_width is None:
            continue

        cache_dtype = model_config.dtype
        if (attention_config := get_attention_quant_config(model, i)) is not None:
            cache_dtype = attention_config.get_quant_dtype()
        indexer_cache_by_layers[i] = torch.empty(
            [
                num_blocks,
                block_size,
                cache_width,
            ],
            dtype=cache_dtype,
            device="meta",
        )
        indexer_cache_per_token += bytes_of_tensor(indexer_cache_by_layers[i]) / (num_blocks * block_size)

    return {
        "indexer_cache_by_layers": indexer_cache_by_layers,
        "indexer_cache_per_token": indexer_cache_per_token,
    }


def generate_inputs_varlen(model, requests: List[RequestInfo], block_size):
    """
    requests: List[RequestInfo], each dict represents a request, containing keys: query_len, seq_len, is_decode
    """
    model_config = model.model_config
    mtp = getattr(model_config, "mtp_config", None)
    num_mtp_tokens = mtp.num_mtp_layers if mtp else 0

    batch_size = len(requests)
    if batch_size == 0:
        return {}

    query_lens = [r.query_len for r in requests]
    seq_lens = [r.seq_len for r in requests]
    is_decode_list = [r.is_decode for r in requests]
    num_tokens = sum(query_lens)

    query_start_loc = [0]
    for ql in query_lens:
        query_start_loc.append(query_start_loc[-1] + ql)
    query_start_loc = torch.tensor(query_start_loc, dtype=torch.long)

    seq_lens_t = torch.tensor(seq_lens, dtype=torch.long)
    query_len_t = torch.tensor(query_lens, dtype=torch.long)

    num_blocks = (sum(seq_lens) + batch_size * (num_mtp_tokens + 1) + block_size - 1) // block_size
    max_num_blocks_per_seq = (max(seq_lens) + block_size - 1) // block_size
    block_table_tensor = torch.empty((batch_size, max_num_blocks_per_seq), dtype=torch.long, device="meta")
    slot_mapping = torch.empty((num_tokens,), dtype=torch.long, device="meta")

    attn_meta = AttentionMetadataTensorCast(
        query_start_loc=query_start_loc,
        query_lens=query_len_t,
        seq_lens=seq_lens_t,
        block_table_tensor=block_table_tensor,
        slot_mapping=slot_mapping,
    )

    input_ids = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
    position_ids = torch.empty([1, num_tokens], dtype=torch.long, device="meta")

    kv_cache_by_layers, kv_cache_per_token = get_kv_cache_info(model, num_blocks, block_size)

    sampling_meta = SamplingMetadata(query_start_loc=query_start_loc)
    selected_token_indices = []

    pos = 0
    for ql, decode in zip(query_lens, is_decode_list):
        if decode:
            selected_token_indices.extend(range(pos, pos + ql))
        else:
            selected_token_indices.append(pos + ql - 1)
        pos += ql
    sampling_meta.selected_token_indices = torch.tensor(selected_token_indices, dtype=torch.long, device="meta")

    kwargs = {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "attention_meta": attn_meta,
        "kv_cache_by_layers": kv_cache_by_layers,
        "sampling_metadata": sampling_meta,
        "kv_cache_per_token": kv_cache_per_token,
    }

    sparse_attention_indexer_cache = get_sparse_attention_indexer_cache_info(model, num_blocks, block_size)
    kwargs.update(sparse_attention_indexer_cache)

    if model.model_config.hf_config.model_type == "qwen3_next":
        kwargs["cache_position"] = torch.arange(num_tokens, dtype=torch.long, device="cpu")

    return kwargs


def get_inputs_num_bytes(model, requests: List[RequestInfo], block_size: int) -> int:
    """
    Get the number of bytes of the input tensors.
    """
    input_kwargs = generate_inputs_varlen(model, requests, block_size)
    inputs_num_bytes = 0
    inputs_num_bytes += bytes_of_tensor(input_kwargs["input_ids"])
    inputs_num_bytes += bytes_of_tensor(input_kwargs["position_ids"])
    inputs_num_bytes += bytes_of_tensor(input_kwargs["attention_meta"].query_start_loc)
    inputs_num_bytes += bytes_of_tensor(input_kwargs["attention_meta"].seq_lens)
    inputs_num_bytes += bytes_of_tensor(input_kwargs["attention_meta"].query_lens)
    inputs_num_bytes += bytes_of_tensor(input_kwargs["attention_meta"].block_table_tensor)
    inputs_num_bytes += bytes_of_tensor(input_kwargs["attention_meta"].slot_mapping)
    return inputs_num_bytes
