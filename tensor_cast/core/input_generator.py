# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
"""
input_generation
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from importlib import import_module
from pathlib import Path
from typing import Any, Optional

import torch

from ..layers.attention import AttentionMetadataTensorCast
from ..layers.sampler import SamplingMetadata
from ..performance_model import bytes_of_tensor
from ..transformers.utils import get_attention_quant_config, logger
from ..utils import exact_division

# Qwen2-VL / Qwen3-VL preprocessor_config.json defaults when no local config is available.
_QWEN_VL_DEFAULT_MIN_PIXELS = 65536
_QWEN_VL_DEFAULT_MAX_PIXELS = 16777216


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


def generate_inputs(model, requests: list[RequestInfo], block_size: int = 128):
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
    # total_kv_tokens mirrors the numerator behind num_blocks (max_context_length
    # per request, summed over the batch); V4 sizing compresses this footprint.
    total_kv_tokens = max_context_length * batch_size
    kv_cache_by_layers, kv_cache_per_token = _get_kv_cache_info(
        model, num_blocks, block_size, batch_size, total_kv_tokens
    )
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

    sparse_attention_indexer_cache = get_sparse_attention_indexer_cache_info(
        model, num_blocks, block_size, batch_size, total_kv_tokens
    )
    kwargs.update(sparse_attention_indexer_cache)

    if model.model_config.hf_config.model_type in (
        "qwen3_next",
        "qwen3_5",
        "qwen3_5_moe",
    ):
        cache_position = torch.arange(context_length, context_length + num_tokens, dtype=torch.long, device="cpu")
        cache_position.tensor_cast_query_lens = tuple(query_len for _ in range(batch_size))
        cache_position.tensor_cast_is_decode = tuple(is_decode for _ in range(batch_size))
        cache_position.tensor_cast_has_previous_state = context_length > 0
        cache_position.tensor_cast_base_decode_query_len = 1 if is_decode and num_mtp_tokens > 0 else query_len
        cache_position.tensor_cast_num_mtp_tokens = num_mtp_tokens
        cache_position.tensor_cast_effective_decode_steps = query_len if is_decode else 0
        kwargs["cache_position"] = cache_position
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
        min_pixels, max_pixels = _load_qwen_vl_pixel_limits(model_id)
        return {
            "height": image_height,
            "width": image_width,
            "factor": factor,
            "min_pixels": min_pixels,
            "max_pixels": max_pixels,
        }

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


def _read_preprocessor_config(path: Path):
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        logger.debug("Failed to read preprocessor config from %s.", path, exc_info=True)
        return None


def _resolve_local_preprocessor_config(model_id: str) -> Path | None:
    model_path = Path(model_id)
    if model_path.is_dir():
        config_path = model_path / "preprocessor_config.json"
        if config_path.is_file():
            return config_path

    try:
        from transformers.utils import cached_file

        cached_path = cached_file(model_id, "preprocessor_config.json", local_files_only=True)
        if cached_path:
            return Path(cached_path)
    except Exception:
        logger.debug(
            "No local cached preprocessor_config.json for model_id=%s.",
            model_id,
            exc_info=True,
        )
    return None


def _extract_pixel_limits(config: dict | None):
    if not config:
        return None, None
    size = config.get("size")
    if isinstance(size, Mapping):
        min_pixels = size.get("shortest_edge") or size.get("min_pixels")
        max_pixels = size.get("longest_edge") or size.get("max_pixels")
        if min_pixels is not None and max_pixels is not None:
            return min_pixels, max_pixels
    min_pixels = config.get("min_pixels") or config.get("shortest_edge")
    max_pixels = config.get("max_pixels") or config.get("longest_edge")
    return min_pixels, max_pixels


def _load_qwen_vl_pixel_limits(model_id: str) -> tuple[int, int]:
    min_pixels, max_pixels = _load_preprocessor_pixel_limits(model_id)
    if min_pixels is not None and max_pixels is not None:
        return min_pixels, max_pixels
    logger.info(
        "Using Qwen VL default pixel limits for model_id=%s (no local preprocessor_config.json).",
        model_id,
    )
    return _QWEN_VL_DEFAULT_MIN_PIXELS, _QWEN_VL_DEFAULT_MAX_PIXELS


@lru_cache(maxsize=128)
def _load_preprocessor_pixel_limits(model_id: str):
    """
    Load image pixel limits from a local HF processor config.
    """
    if not model_id:
        logger.warning("model_id is empty; Qwen VL resize will use built-in default pixel limits.")
        return None, None

    local_config = _resolve_local_preprocessor_config(model_id)
    if local_config is not None:
        min_pixels, max_pixels = _extract_pixel_limits(_read_preprocessor_config(local_config))
        if min_pixels is not None and max_pixels is not None:
            return min_pixels, max_pixels

    try:
        from transformers import AutoImageProcessor

        image_processor = AutoImageProcessor.from_pretrained(model_id, local_files_only=True)
        size = getattr(image_processor, "size", None)
        if size is None or not isinstance(size, Mapping):
            return None, None
        min_pixels = size.get("shortest_edge")
        max_pixels = size.get("longest_edge")
        return min_pixels, max_pixels
    except Exception:
        logger.debug(
            "No local image processor for model_id=%s; Qwen VL resize may use built-in defaults.",
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


def _is_v4_model(model) -> bool:
    """Return True when the loaded model is DeepSeek V4."""
    for config in (
        getattr(getattr(model, "model_config", None), "hf_config", None),
        getattr(model, "text_config", None),
    ):
        if config is not None and getattr(config, "model_type", None) == "deepseek_v4":
            return True
    return False


def _resolve_main_kv_cache_dtype(model, layer_idx: int) -> torch.dtype:
    """Resolve storage dtype for the primary (attention) KV cache.

    DeepSeek V4's reference inference model keeps the shared attention KV cache
    in the model working dtype (bf16/fp16) even when activations are FP8-quantized
    elsewhere (model.py:506-507, 527). Indexer cache may still use FP8; see
    ``_resolve_indexer_cache_dtype``.
    """
    model_config = model.model_config
    if _is_v4_model(model):
        return model_config.dtype

    kvcache_dtype = model_config.dtype
    if (attention_config := get_attention_quant_config(model, layer_idx)) is not None:
        kvcache_dtype = attention_config.get_quant_dtype()
    return kvcache_dtype


def _resolve_indexer_cache_dtype(model, layer_idx: int) -> torch.dtype:
    """Resolve storage dtype for sparse-attention indexer auxiliary cache."""
    model_config = model.model_config
    cache_dtype = model_config.dtype
    if (attention_config := get_attention_quant_config(model, layer_idx)) is not None:
        cache_dtype = attention_config.get_quant_dtype()
    return cache_dtype


def _get_kv_cache_info(
    model,
    num_blocks: int,
    block_size: int,
    batch_size: Optional[int] = None,
    total_kv_tokens: Optional[int] = None,
) -> tuple[dict[Any, Any], int]:
    model_config = model.model_config
    parallel_config = model.model_config.parallel_config
    decoder_layers = None
    if model_config.mla_config is not None:
        try:
            decoder_layers = _resolve_decoder_layers(model)
        except AttributeError:
            decoder_layers = None
    # Initialize the KV cache structure (also on 'meta' device).
    is_v4_model = _is_v4_model(model)
    kv_cache_per_token = 0
    kv_cache_by_layers = {}
    for i in range(model.num_hidden_layers):
        kvcache_dtype = _resolve_main_kv_cache_dtype(model, i)

        if model_config.mla_config is not None:
            # decoder_layers may be None if _resolve_decoder_layers raises
            # AttributeError (e.g., model not fully wrapped). In that case
            # attention_layer stays None and the fallback formula below is used.
            attention_layer = None
            if decoder_layers is not None and i < len(decoder_layers):
                attention_layer = _resolve_decoder_attention_layer(decoder_layers[i])
            if is_v4_model:
                kv_cache_shape = _resolve_v4_kv_cache_size(
                    model,
                    attention_layer,
                    num_blocks,
                    block_size,
                    batch_size,
                    total_kv_tokens,
                )
                kv_cache_by_layers[i] = torch.empty(
                    kv_cache_shape,
                    dtype=kvcache_dtype,
                    device="meta",
                )
            else:
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


def _resolve_v4_kv_cache_size(
    model,
    attention_layer=None,
    num_blocks: int = 1,
    block_size: int = 1,
    batch_size: Optional[int] = None,
    total_kv_tokens: Optional[int] = None,
) -> list[int]:
    """
    Resolve V4 KV cache shape based on compress_ratio.

    Per the reference implementation (ds-model-v4-pro/inference/model.py:473-474):
        kv_cache_size = window_size + (max_seq_len // compress_ratio if compress_ratio else 0)
        kv_cache = zeros(max_batch_size, kv_cache_size, head_dim)

    The reference allocates the cache PER request (``max_batch_size`` rows) and,
    along the sequence axis, only keeps a *compressed* footprint:

      Layer type | per-request sequence slots
      -----------|------------------------------------------------
      ratio=0    | window_size                  (pure sliding window)
      ratio=4    | window_size + seq_len // 4    (window + compressed KV)
      ratio=128  | window_size + seq_len // 128  (window + heavily compressed KV)

    msmodeling stores caches in a paged ``[num_blocks, block_size, head_dim]``
    layout, so we translate the compressed per-request footprint into a block
    count over the whole batch:

        total_slots = batch_size * window_size + total_kv_tokens // compress_ratio
        num_blocks  = ceil(total_slots / block_size)

    where ``total_kv_tokens`` is the sum of per-request sequence lengths in the
    batch (the number of real token positions that flow through the KV cache).

    NOTE: the previous implementation divided ``max_position_embeddings`` by the
    compress ratio and compared the single-request result against the whole
    batch-wide pool (``num_blocks * block_size``). Those two quantities are not
    dimensionally comparable, so the branch never triggered and every V4 layer
    fell back to the full (uncompressed) pool size, over-counting KV memory.

    Args:
        model: The model wrapper.
        attention_layer: The attention layer instance.
        num_blocks: Paged-pool block count, used for the non-V4 / fallback path.
        block_size: Size of each cache block.
        batch_size: Number of sequences in the batch. Required together with
            ``total_kv_tokens`` to apply V4 compressed sizing.
        total_kv_tokens: Sum of per-request sequence lengths across the batch.

    Returns:
        List representing the tensor shape: ``[num_blocks, block_size, head_dim]``.
    """
    head_dim = _resolve_sparse_attention_kv_cache_width(model, attention_layer)

    # window_size from config (sliding_window). Non-V4 MLA models (e.g. V3.2)
    # have no sliding window, so this stays 0 and we keep the full pool below.
    window_size = int(getattr(model.text_config, "sliding_window", 0) or 0)

    compress_ratio = 0
    if attention_layer is not None:
        compress_ratio = int(getattr(attention_layer, "compress_ratio", 0) or 0)

    # A V4 sparse layer either keeps a sliding window (ratio==0) or a
    # window + compressed cache (ratio>0). Standard MLA layers have neither.
    is_v4_sparse_layer = window_size > 0 or compress_ratio > 0
    if is_v4_sparse_layer and batch_size is not None and total_kv_tokens is not None:
        # Sliding-window ring buffer: window_size slots per request.
        window_slots = batch_size * window_size
        # Compressed KV: the reference Compressor pools every `compress_ratio`
        # consecutive tokens into a single cache row, so the compressed segment
        # holds total_kv_tokens // compress_ratio slots across the batch.
        compressed_slots = (total_kv_tokens // compress_ratio) if compress_ratio > 0 else 0
        total_slots = window_slots + compressed_slots
        adjusted_num_blocks = max(1, (total_slots + block_size - 1) // block_size)
        return [adjusted_num_blocks, block_size, head_dim]

    # Non-V4 MLA or missing batch info: keep the full paged pool.
    return [num_blocks, block_size, head_dim]


def get_kv_cache_info(model, num_blocks, block_size, batch_size=None, total_kv_tokens=None):
    return _get_kv_cache_info(model, num_blocks, block_size, batch_size, total_kv_tokens)


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


def get_sparse_attention_indexer_cache_info(model, num_blocks, block_size, batch_size=None, total_kv_tokens=None):
    """Allocate per-layer auxiliary indexer caches for sparse-attention wrappers.

    Despite the older DSA-oriented naming in surrounding code, this helper is
    also used by DeepSeek V4's custom sparse attention path, whose ratio=4
    layers carry a distinct learned indexer.

    For V4 the indexer cache is purely compressed (no sliding window): the
    reference allocates ``[max_batch_size, max_seq_len // compress_ratio,
    index_head_dim]`` (model.py:399). When ``batch_size`` and
    ``total_kv_tokens`` are provided we size the paged cache to
    ``total_kv_tokens // compress_ratio`` slots instead of the full pool.
    """
    model_config = model.model_config
    mla_config = model_config.mla_config
    if mla_config is None or not mla_config.mla_cls.requires_indexer_cache():
        return {}

    # Compressed indexer-cache sizing is a DeepSeek V4-only behavior. Other
    # sparse-attention models (e.g. DeepSeek V3.2 / DSA) also reach this helper
    # via requires_indexer_cache(), so we must not let the compression branch
    # alter their cache size. Gate it explicitly on the V4 model type.
    is_v4_model = _is_v4_model(model)
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

        cache_dtype = _resolve_indexer_cache_dtype(model, i)

        # Indexer cache is purely compressed (no window). Size it to
        # total_kv_tokens // compress_ratio slots when batch info is available,
        # otherwise fall back to the full paged pool for backward compatibility.
        # Only DeepSeek V4 uses this compressed sizing; every other model keeps
        # the full paged pool unchanged.
        indexer_num_blocks = num_blocks
        compress_ratio = (
            int(getattr(attention_layer, "compress_ratio", 0) or 0)
            if (is_v4_model and attention_layer is not None)
            else 0
        )
        if batch_size is not None and total_kv_tokens is not None and compress_ratio > 0:
            compressed_slots = total_kv_tokens // compress_ratio
            indexer_num_blocks = max(1, (compressed_slots + block_size - 1) // block_size)

        indexer_cache_by_layers[i] = torch.empty(
            [
                indexer_num_blocks,
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


def generate_inputs_varlen(model, requests: list[RequestInfo], block_size):
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

    total_kv_tokens = sum(seq_lens) + batch_size * (num_mtp_tokens + 1)
    num_blocks = (total_kv_tokens + block_size - 1) // block_size
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

    kv_cache_by_layers, kv_cache_per_token = get_kv_cache_info(
        model, num_blocks, block_size, batch_size, total_kv_tokens
    )

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

    sparse_attention_indexer_cache = get_sparse_attention_indexer_cache_info(
        model, num_blocks, block_size, batch_size, total_kv_tokens
    )
    kwargs.update(sparse_attention_indexer_cache)

    if model.model_config.hf_config.model_type in (
        "qwen3_next",
        "qwen3_5",
        "qwen3_5_moe",
    ):
        cache_positions = []
        first_context_length = 0
        for request in requests:
            context_length = request.context_length or max(request.seq_len - request.query_len, 0)
            if not cache_positions:
                first_context_length = context_length
            cache_positions.append(
                torch.arange(
                    context_length,
                    context_length + request.query_len,
                    dtype=torch.long,
                    device="cpu",
                )
            )
        cache_position = torch.cat(cache_positions)
        cache_position.tensor_cast_query_lens = tuple(query_lens)
        cache_position.tensor_cast_is_decode = tuple(is_decode_list)
        cache_position.tensor_cast_has_previous_state = first_context_length > 0
        cache_position.tensor_cast_base_decode_query_lens = tuple(
            1 if is_decode and num_mtp_tokens > 0 else query_len
            for query_len, is_decode in zip(query_lens, is_decode_list)
        )
        cache_position.tensor_cast_num_mtp_tokens = num_mtp_tokens
        cache_position.tensor_cast_effective_decode_steps = tuple(
            query_len if is_decode else 0 for query_len, is_decode in zip(query_lens, is_decode_list)
        )
        kwargs["cache_position"] = cache_position

    return kwargs


def get_inputs_num_bytes(model, requests: list[RequestInfo], block_size: int) -> int:
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
