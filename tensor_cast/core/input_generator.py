# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2025-2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""
input_generation
"""

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import import_module
from numbers import Integral
from pathlib import Path
from typing import Any, Optional

import torch

from ..layers.attention import AttentionMetadataTensorCast
from ..layers.deepseek_v41 import DeepseekV41SharedAttentionState, EngramHashState, resolve_csa2_layer_role
from ..layers.glm5 import get_glm5_indexer_types, glm5_uses_indexshare, resolve_glm5_indexer_source_layer
from ..layers.sampler import SamplingMetadata, SpecDecodeMetadata
from ..model_config import config_has_draft_spec
from ..performance_model import bytes_of_tensor
from ..transformers.utils import get_attention_quant_config, logger
from ..utils import exact_division
from .vision_geometry import VISION_IMAGE_BOUNDARY_TOKEN_COUNT

# Qwen2-VL / Qwen3-VL preprocessor_config.json defaults when no local config is available.
_QWEN_VL_DEFAULT_MIN_PIXELS = 65536
_QWEN_VL_DEFAULT_MAX_PIXELS = 16777216
_BAILING_V3_KDA_CONV_STATE_COUNT = 3
_DEFAULT_DRAFT_SLIDING_WINDOW = 2048


def _is_deepseek_v41_config(config) -> bool:
    """Recognize both the multimodal wrapper and its nested text config."""
    return getattr(config, "model_type", None) in {"deepseek_v41", "deepseek_v41_text"}


def _validate_deepseek_v41_image_chunk(
    model_type: str,
    has_image_input: bool,
    is_decode: bool,
    context_length: int,
) -> None:
    if model_type == "deepseek_v41" and has_image_input and not is_decode and int(context_length) != 0:
        raise ValueError("DeepSeek V4.1 image spans must be fully contained in the first prefill chunk")


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
    # CSA2 state belongs to the request/workload lifecycle, not the model
    # object.  Keeping it here prevents concurrent ModelRunner invocations from
    # sharing mutable compressor tails and top-k streams.
    csa2_state: Optional[DeepseekV41SharedAttentionState] = field(default=None, repr=False)
    engram_hash_state: Optional[EngramHashState] = field(default=None, repr=False)


def _build_spec_decode_metadata(
    query_start_loc: torch.Tensor,
    query_lens: torch.Tensor | list[int],
    num_speculative_tokens: int,
) -> SpecDecodeMetadata:
    """Build verify-window metadata for MTP / Dflash speculative decode.

    ``num_speculative_tokens`` is the number of draft/spec positions (excludes bonus).
    Window length is ``num_speculative_tokens + 1`` (specs + one bonus/anchor slot).
    """
    if num_speculative_tokens <= 0:
        raise ValueError("num_speculative_tokens must be positive for spec decode metadata")

    if isinstance(query_lens, torch.Tensor):
        if query_lens.device.type == "meta":
            raise ValueError("query_lens must be a Python list or a materialized tensor for spec decode metadata")
        query_lens_list = query_lens.tolist()
    else:
        query_lens_list = query_lens
    spec_window = num_speculative_tokens + 1
    # Spec decode metadata covers the per-request tail target+bonus verification window.
    # Longer decode query windows may carry earlier rows, but only the tail window participates
    # in lm-head verification/proposal selection.
    logits_indices = []
    query_start = 0

    for query_len in query_lens_list:
        query_len = int(query_len)
        if query_len < spec_window:
            raise ValueError(
                f"Spec decode query length must be at least num_speculative_tokens + 1 ({spec_window}), got {query_len}"
            )
        tail_start = query_start + query_len - spec_window
        tail_rows = list(range(tail_start, tail_start + spec_window))

        logits_indices.extend(tail_rows)
        query_start += query_len

    metadata_device = query_start_loc.device
    return SpecDecodeMetadata(
        logits_indices=torch.tensor(logits_indices, dtype=torch.long, device=metadata_device),
        num_active_requests=len(query_lens_list),
        num_speculative_tokens=num_speculative_tokens,
    )


def generate_inputs(model, requests: list[RequestInfo], block_size: int = 128):
    # TODO merge generate_inputs and generate_inputs_varlen
    # for now, unify the function signatures, Firstly.
    request = requests[0]
    concurrency = request.concurrency
    seq_len = request.seq_len
    query_len = request.query_len
    is_decode = request.is_decode
    image_kwargs = {}
    num_image_tokens = 0
    # Match the varlen entry so cached-prefix tokens count as previous context.
    context_length = request.context_length or max(seq_len - query_len, 0)
    if model.is_vl_model:
        has_image_input = all(
            value is not None for value in (request.image_batch_size, request.image_height, request.image_width)
        )
        _validate_deepseek_v41_image_chunk(
            model.model_config.hf_config.model_type,
            has_image_input,
            is_decode,
            context_length,
        )
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
    draft_block = int(model_config.draft_block_size()) if config_has_draft_spec(model_config) else 0
    # SpecDecodeMetadata uses specs+bonus window aligned with existing Sampler.
    # DSpark sample_from_anchor=True semantically proposes B tokens, but verify window
    # remains query_len=B; keep B-1 speculative slots as the DFlash-compatible engineering
    # approximation (RFC 3.4.5). Fold still clamps acceptance to B for DSpark.
    dflash_spec_tokens = max(draft_block - 1, 0)
    parallel_config = model_config.parallel_config
    batch_size = (concurrency + parallel_config.data_parallel_size - 1) // parallel_config.data_parallel_size

    speculative_tail = max(num_mtp_tokens, dflash_spec_tokens) + 1
    max_context_length = seq_len + speculative_tail

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
    compressed_slot_mapping = torch.empty((batch_size * query_len,), dtype=torch.long, device="meta")

    attn_meta = AttentionMetadataTensorCast(
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        query_lens=query_lens,
        seq_lens_values=[seq_len] * batch_size,
        query_lens_values=[query_len] * batch_size,
        is_decode_values=[is_decode] * batch_size,
        block_table_tensor=block_table_tensor,
        slot_mapping=slot_mapping,
        max_total_seq_len=int(seq_len),
        start_pos=int(context_length),
        compressed_slot_mapping=compressed_slot_mapping,
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
    if model.model_config.hf_config.model_type == "deepseek_v41" and "pixel_values" in image_kwargs:
        image_kwargs["image_token_types"] = torch.empty_like(input_ids, dtype=torch.int8)
        image_kwargs["image_token_mask"] = torch.empty_like(input_ids, dtype=torch.bool)
    # total_kv_tokens mirrors the numerator behind num_blocks (max_context_length
    # per request, summed over the batch); V4 sizing compresses this footprint.
    total_kv_tokens = max_context_length * batch_size
    # Decode Context Parallel stores only ``1 / dcp`` of each sequence's tokens on
    # a card, so where that shard is a real per-card saving the KV footprint (the
    # ``num_blocks`` sizing the physical cache tensors) shrinks accordingly. Per-token
    # cost is unaffected (both the tensor bytes and the ``num_blocks`` denominator
    # scale together inside ``_get_kv_cache_info``). The factor is layout-dependent
    # (``dcp`` for MLA, TP-replication-capped for GQA, 1 for SFA); see
    # ``dcp_kv_token_capacity_factor``.
    kv_num_blocks = dcp_sharded_num_blocks(model, num_blocks)
    kv_cache_by_layers, kv_cache_per_token = _get_kv_cache_info(
        model,
        kv_num_blocks,
        block_size,
        batch_size,
        total_kv_tokens,
        [seq_len] * batch_size,
        speculative_tail,
        num_blocks,
    )
    compressed_cache_by_source, compressed_scale_cache_by_source, compressed_cache_per_token = (
        _get_deepseek_v41_compressed_cache_info(model, block_size, batch_size, total_kv_tokens)
    )
    sampling_metadata = SamplingMetadata(
        query_start_loc=attn_meta.query_start_loc,
    )
    # Short decode windows cannot form the target+bonus verification window; omit spec metadata
    # so downstream lm-head and sampler logic uses ordinary decode selection for that step.
    # Fixed-shape generation uses one request template expanded by concurrency, so query_len is uniform.
    # MTP and Dflash both use SpecDecodeMetadata so Sampler emits verify ArgMax as
    # [1,V]→[1] (bonus) + [S,V]→[S] (speculative), matching NPU DFlash / MTP traces.
    if is_decode and num_mtp_tokens > 0 and query_len >= num_mtp_tokens + 1:
        sampling_metadata.spec_decode_metadata = _build_spec_decode_metadata(
            attn_meta.query_start_loc,
            [query_len] * batch_size,
            num_mtp_tokens,
        )
    elif is_decode and dflash_spec_tokens > 0 and query_len >= draft_block:
        sampling_metadata.spec_decode_metadata = _build_spec_decode_metadata(
            attn_meta.query_start_loc,
            [query_len] * batch_size,
            dflash_spec_tokens,
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
        "kv_cache_excluded_layer_indices": kv_cache_excluded_layer_indices(model),
        "sampling_metadata": sampling_metadata,
    }
    if _is_deepseek_v41_config(model.model_config.hf_config):
        kwargs["csa2_state"] = _get_deepseek_v41_state(model, requests)
        if is_decode:
            _seed_deepseek_v41_draft_windows(model, kwargs["csa2_state"], batch_size)
        kwargs["engram_hash_state"] = _get_deepseek_v41_engram_state(model, requests)
        kwargs["engram_token_ids"] = torch.empty_like(input_ids)
        if "image_token_mask" in image_kwargs:
            kwargs["engram_token_mask"] = torch.empty_like(input_ids, dtype=torch.bool)
        kwargs["csa2_cache_ownership"] = _get_deepseek_v41_cache_ownership(model)
        kwargs["compressed_kv_cache_by_source"] = compressed_cache_by_source
        kwargs["compressed_kv_scale_cache_by_source"] = compressed_scale_cache_by_source
        kwargs["compressed_kv_cache_per_token"] = compressed_cache_per_token

    sparse_attention_indexer_cache = get_sparse_attention_indexer_cache_info(
        model, num_blocks, block_size, batch_size, total_kv_tokens
    )
    kwargs.update(sparse_attention_indexer_cache)

    if model.model_config.hf_config.model_type in (
        "qwen3_next",
        "qwen3_5",
        "qwen3_5_moe",
        "qwen3_5_moe_text",
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
        "glm4v": (
            "transformers.models.glm4v.image_processing_glm4v",
            build_glm_resize_params,
        ),
        "glm4v_moe": (
            "transformers.models.glm4v.image_processing_glm4v",
            build_glm_resize_params,
        ),
        "glm5_next": (
            "transformers.models.glm5_next.image_processing_glm5_next",
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
        from transformers import AutoImageProcessor  # pylint: disable=no-name-in-module

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
        logger.info("Vision-language model is running without image input; skip the visual encoder.")
        return {}
    hf_config = model.model_config.hf_config
    vision_config = hf_config.vision_config
    if hf_config.model_type == "deepseek_v41":
        return _generate_deepseek_v41_image_inputs(
            model,
            image_batch_size,
            image_height,
            image_width,
            concurrency,
        )
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
    num_image_tokens = image_batch_size * (tokens // merge_length + VISION_IMAGE_BOUNDARY_TOKEN_COUNT)
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


def _is_integral_non_bool(value) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool)


def _resolve_sparse_attention_indexer_cache_width(model, attention_layer) -> int | None:
    """Resolve the auxiliary indexer cache width for sparse-attention wrappers.

    For V4 ratio=4 layers this picks up the dedicated indexer-local head width
    (`index_head_dim`), which is intentionally different from the main KV cache
    width (`head_dim`).
    """
    for attr in ("indexer_cache_width", "_index_head_dim", "indexer_head_dim"):
        width = getattr(attention_layer, attr, None)
        if _is_integral_non_bool(width):
            return int(width)

    indexer = getattr(attention_layer, "indexer", None)
    if indexer is not None:
        width = getattr(indexer, "head_dim", None)
        if _is_integral_non_bool(width):
            return int(width)

    width = getattr(model.text_config, "index_head_dim", None)
    if _is_integral_non_bool(width):
        return int(width)
    return None


def _layer_uses_sparse_attention_indexer(attention_layer) -> bool:
    is_sparse_layer = getattr(attention_layer, "is_sparse_layer", None)
    if isinstance(is_sparse_layer, bool):
        return is_sparse_layer
    use_indexer = getattr(attention_layer, "use_indexer", None)
    if isinstance(use_indexer, bool):
        return use_indexer
    return getattr(attention_layer, "indexer", None) is not None


def _resolve_decoder_attention_layer(layer, preserve_attention_wrapper: bool = False):
    """Resolve a decoder layer's attention module through lightweight wrappers."""
    from ..layers.utils import ModelWrapperBase

    current = layer
    visited = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))

        attention_layer = getattr(current, "self_attn", None)
        if attention_layer is None:
            attention_layer = getattr(current, "attention", None)
        if attention_layer is not None:
            if preserve_attention_wrapper:
                return attention_layer
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
    """Return True when the loaded model uses a V4 shared-KV layout."""
    for config in (
        getattr(getattr(model, "model_config", None), "hf_config", None),
        getattr(model, "text_config", None),
    ):
        if config is not None and getattr(config, "model_type", None) in (
            "deepseek_v4",
            "deepseek_v41",
            "deepseek_v41_text",
        ):
            return True
    return False


def _is_mimo_v2_flash_model(model) -> bool:
    """Return True when the model uses MiMo V2's asymmetric K/V cache rows."""
    for config in (
        getattr(getattr(model, "model_config", None), "hf_config", None),
        getattr(model, "text_config", None),
    ):
        if config is not None and getattr(config, "model_type", None) == "mimo_v2_flash":
            return True
    return False


def _get_glm5_indexshare_config(model):
    orig_mod = getattr(model, "_orig_mod", None)
    for candidate in (model, orig_mod):
        if candidate is None:
            continue
        inner = getattr(candidate, "_inner", None)
        for config in (
            getattr(inner, "hf_config", None),
            getattr(getattr(candidate, "model_config", None), "hf_config", None),
            getattr(candidate, "text_config", None),
        ):
            if config is not None and getattr(config, "model_type", None) == "glm_moe_dsa":
                return config
    return None


def _get_glm5_indexshare_indexer_types(model) -> list[str] | None:
    glm5_config = _get_glm5_indexshare_config(model)
    if glm5_config is None or not glm5_uses_indexshare(glm5_config):
        return None
    return get_glm5_indexer_types(glm5_config)


def _glm5_indexshare_layer_owns_indexer_cache(indexer_types: list[str] | None, layer_idx: int) -> bool:
    # GLM-5.2 shared layers reuse prev_topk_indices and must not alias source caches here.
    return indexer_types is None or resolve_glm5_indexer_source_layer(indexer_types, layer_idx) == layer_idx


def _draft_layer_count(model_config) -> int:
    """Return DFlash/DSpark draft depth; 0 for mocks / configs without a real count.

    ``MagicMock`` implements ``__int__`` (returns 1) and a callable
    ``has_draft_spec``, so only accept a real ``Integral`` from ``draft_num_layers()``.
    """
    if not config_has_draft_spec(model_config):
        return 0
    getter = getattr(model_config, "draft_num_layers", None)
    if not callable(getter):
        return 0
    value = getter()
    if isinstance(value, bool) or not isinstance(value, Integral):
        return 0
    return max(int(value), 0)


def _resolve_sparse_attention_indexer_num_blocks(
    is_v4_model: bool,
    attention_layer,
    num_blocks: int,
    block_size: int,
    batch_size: Optional[int] = None,
    total_kv_tokens: Optional[int] = None,
) -> int:
    # DeepSeek V4 compresses indexer-cache slots; other sparse-attention models keep the full paged pool.
    compress_ratio = (
        int(getattr(attention_layer, "compress_ratio", 0) or 0) if (is_v4_model and attention_layer is not None) else 0
    )
    if batch_size is None or total_kv_tokens is None or compress_ratio <= 0:
        return num_blocks

    compressed_slots = total_kv_tokens // compress_ratio
    return max(1, (compressed_slots + block_size - 1) // block_size)


def _resolve_draft_kv_cache_num_blocks(
    model,
    draft_layer_idx: int,
    num_blocks: int,
    block_size: int,
    batch_size: Optional[int] = None,
    total_kv_tokens: Optional[int] = None,
    request_seq_lens: Optional[list[int]] = None,
    draft_tail_tokens: int = 0,
    unsharded_num_blocks: Optional[int] = None,
) -> int:
    """Return draft-cache blocks according to the draft layer's attention type.

    ``request_seq_lens`` contains raw context lengths. ``draft_tail_tokens`` is
    separately retained after any sliding-window cap because the draft block is
    written after the context ring-buffer slots.
    """
    model_config = getattr(model, "model_config", None)
    dspark_config = getattr(model_config, "dspark_config", None)
    dflash_config = getattr(model_config, "dflash_config", None)
    if dspark_config is not None:
        draft_config = dspark_config
        force_sliding_window = True
    elif dflash_config is not None:
        draft_config = dflash_config
        force_sliding_window = False
    else:
        return num_blocks
    if batch_size is None or total_kv_tokens is None or batch_size <= 0 or total_kv_tokens <= 0:
        return num_blocks

    layer_types = getattr(draft_config, "layer_types", None)
    if not isinstance(layer_types, (list, tuple)):
        inner = getattr(model, "_inner", None)
        draft_hf = getattr(inner, "draft_hf_config", None)
        layer_types = getattr(draft_hf, "layer_types", None)
    layer_type = "sliding_attention" if force_sliding_window else "full_attention"
    if isinstance(layer_types, (list, tuple)) and 0 <= int(draft_layer_idx) < len(layer_types):
        layer_type = str(layer_types[int(draft_layer_idx)])

    try:
        draft_tail_tokens = max(int(draft_tail_tokens), 0)
    except (TypeError, ValueError):
        draft_tail_tokens = 0

    sliding_window = getattr(draft_config, "sliding_window", None)
    try:
        sliding_window = int(sliding_window)
    except (TypeError, ValueError):
        sliding_window = _DEFAULT_DRAFT_SLIDING_WINDOW
    if sliding_window <= 0:
        sliding_window = _DEFAULT_DRAFT_SLIDING_WINDOW

    if request_seq_lens is not None and len(request_seq_lens) == int(batch_size):
        return sum(
            max(
                1,
                (
                    (
                        min(max(int(seq_len), 0), sliding_window)
                        if layer_type == "sliding_attention"
                        else max(int(seq_len), 0)
                    )
                    + draft_tail_tokens
                    + block_size
                    - 1
                )
                // block_size,
            )
            for seq_len in request_seq_lens
        )

    if layer_type != "sliding_attention":
        # Draft attention never participates in target DCP. When individual request
        # lengths are unavailable, retain the target's unsharded capacity rather than
        # reusing the DCP-sharded block count passed to the main KV allocator.
        return max(int(unsharded_num_blocks or num_blocks), 1)

    # Paged block tables are per-request. Without individual sequence lengths,
    # reserve one window-sized allocation per active request rather than packing
    # partially used pages from distinct requests into the same physical block.
    per_request_slots = min(int(total_kv_tokens), sliding_window) + draft_tail_tokens
    return int(batch_size) * max(1, (per_request_slots + block_size - 1) // block_size)


# Backward-compatible wrapper for callers/tests that used the DSpark-specific name.
def _resolve_dspark_kv_cache_num_blocks(
    model,
    num_blocks: int,
    block_size: int,
    batch_size: Optional[int] = None,
    total_kv_tokens: Optional[int] = None,
    request_seq_lens: Optional[list[int]] = None,
    draft_tail_tokens: int = 0,
    unsharded_num_blocks: Optional[int] = None,
) -> int:
    return _resolve_draft_kv_cache_num_blocks(
        model,
        0,
        num_blocks,
        block_size,
        batch_size,
        total_kv_tokens,
        request_seq_lens,
        draft_tail_tokens,
        unsharded_num_blocks,
    )


def _is_mla_model(model) -> bool:
    """Return True when the model stores latent (MLA-style) KV instead of GQA heads."""
    return getattr(getattr(model, "model_config", None), "mla_config", None) is not None


def _model_uses_sparse_attention_indexer(model) -> bool:
    """Return True when this model allocates the auxiliary SFA indexer cache.

    Mirrors the gate in ``get_sparse_attention_indexer_cache_info``, so any model
    that gets an indexer cache (DeepSeek V3.2 ``deepseek_v32``, GLM-5 ``glm_moe_dsa``
    via ``Glm5SparseAttention``, DeepSeek V4) is recognised here by its KV layout
    rather than by model name.
    """
    mla_config = getattr(getattr(model, "model_config", None), "mla_config", None)
    requires_indexer_cache = getattr(getattr(mla_config, "mla_cls", None), "requires_indexer_cache", None)
    return bool(requires_indexer_cache()) if callable(requires_indexer_cache) else False


def _gqa_kv_replication_factor(model) -> int:
    """How many cards of a TP group redundantly hold the same GQA KV head.

    The paged cache in ``_get_kv_cache_info`` gives a card ``h_kv / tp`` KV heads,
    or 1 replicated head when ``h_kv < tp``. So a TP group stores each KV head
    ``max(1, tp / h_kv)`` times, and that replication is the only KV redundancy DCP
    can convert into capacity. Returns 1 when ``num_key_value_heads`` is unavailable
    (layout unverifiable -> claim no gain).
    """
    num_key_value_heads = getattr(getattr(model, "text_config", None), "num_key_value_heads", None)
    if not isinstance(num_key_value_heads, int) or num_key_value_heads <= 0:
        return 1
    tp_size = model.model_config.parallel_config.tensor_parallel_size
    return max(1, tp_size // num_key_value_heads)


def dcp_kv_token_capacity_factor(model) -> int:
    """Per-card KV token capacity multiplier under Decode Context Parallel.

    DCP shards the KV cache along the token dimension, so a card holds only
    ``1 / dcp_size`` of every sequence's tokens. Whether that buys extra token
    capacity depends on the KV layout, so this is gated on layout -- not model name:

    * **MLA (latent KV)**: the latent cache is not partitioned across TP heads, so
      per-card bytes per sequence really do fall by ``dcp_size``. Factor ``dcp_size``.
    * **GQA**: DCP re-partitions KV heads across the same TP domain, moving a rank
      from ``h_kv/tp`` heads over ``S`` to ``h_kv*dcp/tp`` heads over ``S/dcp``
      (see ``AttentionTensorCast.forward``). Per-card KV bytes are
      ``(h_kv*dcp/tp) * D * (S/dcp) == (h_kv/tp) * D * S`` -- invariant. The only
      real gain is de-duplicating KV heads that TP replicated when ``h_kv < tp``,
      capped by that replication: ``min(dcp_size, max(1, tp/h_kv))``. For
      ``h_kv >= tp`` (e.g. Qwen3-32B, ``h_kv=8``/``tp=8``) the factor is 1;
      returning ``dcp_size`` there would fabricate ``dcp`` times the real token
      capacity and make the optimizer over-estimate servable concurrency/context.

    Callers multiply the warmup ``num_blocks`` by the factor; per-token cost stays
    the true physical per-card cost. Returns 1 when DCP is off.

    Gated off for sparse-attention (SFA) models -- V4 plus every model that
    allocates an indexer cache: their main cache is sized by a sliding window +
    ``seq_len // compress_ratio`` rather than a plain ``S``, and the indexer cache
    is not sequence-sharded at all yet its per-token cost is folded into
    ``kv_cache_per_token_gb``, so a blanket multiplier would also inflate the
    un-sharded indexer part. Modeling SFA + DCP memory is an explicit RFC non-goal
    (rfc_context_parallel_dcp §1.3 #4).
    """
    dcp_size = model.model_config.parallel_config.decode_context_parallel_size
    if dcp_size <= 1:
        return 1
    if _is_v4_model(model) or _model_uses_sparse_attention_indexer(model):
        return 1
    if _is_mla_model(model):
        return dcp_size
    return min(dcp_size, _gqa_kv_replication_factor(model))


def dcp_sharded_num_blocks(model, num_blocks: int) -> int:
    """Per-card paged-block count after the DCP token shard.

    Rounds UP: blocks are the allocation unit, so a card holding the remainder of a
    ``dcp``-way split still needs a whole block for it. Flooring would model a
    slightly smaller per-card KV footprint than any real deployment can have (e.g.
    ``num_blocks=17``, factor 8 -> 2 blocks instead of 3). Always >= 1.
    """
    factor = dcp_kv_token_capacity_factor(model)
    return max(1, -(-num_blocks // factor))


def _resolve_main_kv_cache_dtype(model, layer_idx: int) -> torch.dtype:
    """Resolve storage dtype for the primary (attention) KV cache.

    Draft attention quantization isolation is handled centrally in
    ``get_attention_quant_config`` (returns ``None`` for draft layers), so
    this function does not need to know about draft boundaries.
    """
    model_config = model.model_config
    if _is_deepseek_v41_config(getattr(model, "text_config", None)):
        return torch.float8_e4m3fn
    kvcache_dtype = torch.bfloat16 if _is_v4_model(model) else model_config.dtype
    if (attention_config := get_attention_quant_config(model, layer_idx)) is not None:
        kvcache_dtype = attention_config.get_quant_dtype()
    return kvcache_dtype


def _v4_main_cache_storage(shape, dtype, rope_head_dim):
    """Physical V4 payload; scale metadata and backend padding are not modeled."""
    if dtype == torch.float8_e4m3fn:
        # One byte per non-RoPE value, two bytes per BF16 RoPE value.
        return [*shape[:-1], shape[-1] + rope_head_dim], torch.uint8
    return shape, dtype


def _resolve_indexer_cache_dtype(model, layer_idx: int, attention_layer=None) -> torch.dtype:
    """Resolve storage dtype for sparse-attention indexer auxiliary cache."""
    if _is_deepseek_v41_config(getattr(model, "text_config", None)):
        return torch.int4
    model_config = model.model_config
    if attention_layer is not None:
        dtype = getattr(attention_layer, "indexer_cache_dtype", None)
        if isinstance(dtype, torch.dtype):
            return dtype
    cache_dtype = model_config.dtype
    if (attention_config := get_attention_quant_config(model, layer_idx)) is not None:
        cache_dtype = attention_config.get_quant_dtype()
    return cache_dtype


_DTYPE_BYTE_WIDTHS = {
    torch.bool: 1,
    torch.uint8: 1,
    torch.int8: 1,
    torch.int16: 2,
    torch.int32: 4,
    torch.int64: 8,
    torch.float16: 2,
    torch.bfloat16: 2,
    torch.float32: 4,
    torch.float64: 8,
    torch.complex64: 8,
    torch.complex128: 16,
}


def glm5_next_kda_state_shape(text_config, dtype, tp_size, batch_size) -> tuple[int, int]:
    """Packed per-request state; capacity is independent of context length."""
    heads = exact_division(text_config.linear_num_heads, tp_size)
    dim = text_config.linear_head_dim
    recurrent_bytes = heads * dim * dim * 4
    # HF keeps kernel_size entries, including the current convolution position.
    dtype_bytes = _DTYPE_BYTE_WIDTHS.get(dtype)
    if dtype_bytes is None:
        raise ValueError(f"Unsupported GLM5Next KDA state dtype: {dtype}")
    conv_bytes = 3 * heads * dim * text_config.linear_conv_kernel_dim * dtype_bytes
    return int(batch_size or 1), recurrent_bytes + conv_bytes


def _resolve_bailing_v3_kda_cache_bytes(attention_layer, model_config, parallel_config, batch_size) -> int:
    local_heads = exact_division(
        int(attention_layer.num_heads),
        parallel_config.tensor_parallel_size,
    )
    head_dim = int(attention_layer.head_dim)
    conv_kernel_size = int(attention_layer.conv_size)
    request_batch_size = int(batch_size or 1)

    # FLA KDA keeps its final recurrent state in FP32 regardless of the
    # activation dtype. Its Q/K/V short-convolution states retain the model
    # working dtype.
    recurrent_state_element_size = torch.empty((), dtype=torch.float32).element_size()
    recurrent_state_bytes = request_batch_size * local_heads * head_dim * head_dim * recurrent_state_element_size
    conv_state_bytes = (
        _BAILING_V3_KDA_CONV_STATE_COUNT
        * request_batch_size
        * local_heads
        * head_dim
        * (conv_kernel_size - 1)
        * torch.empty((), dtype=model_config.dtype).element_size()
    )
    return recurrent_state_bytes + conv_state_bytes


def kv_cache_excluded_layer_indices(model) -> set[int]:
    """Return layer indices whose kv_cache placeholder must be excluded from
    KV-cache byte/per-token accounting.

    Qwen3.5 hybrid attention keeps ``linear_attention`` layers in
    ``kv_cache_by_layers`` (the placeholders are still required by pipeline
    parallel stage validation and layer-indexed forward), but these layers do
    not own a real paged KV cache. Counting their placeholders inflates both
    ``kv_cache_bytes`` and ``kv_cache_per_token`` and is the root cause of
    ``device_memory_available_gb == 0``. Centralising the judgement here keeps
    every consumer (``_get_kv_cache_info`` and ``ModelRunner``) on a single,
    consistent accounting scope.

    Kimi K3 hybrid attention follows the same principle: KDA layers
    (``KimiDeltaAttention``, ~69/93 layers) use a recurrent state, not a paged
    KV cache. They are identified by ``is_linear_attn`` (bool) on each decoder
    layer (set by ``KimiDecoderLayer``, propagated through ``CopyLayerWrapper``
    by P1.8). Without this exclusion, KDA layers' KV cache is counted in the
    memory budget, inflating it by ~74% (69/93 layers are false positives).
    """
    model_type = getattr(
        getattr(getattr(model, "model_config", None), "hf_config", None),
        "model_type",
        None,
    )
    if model_type in ("qwen3_5", "qwen3_5_moe", "qwen3_5_moe_text"):
        layer_types = getattr(getattr(model, "text_config", None), "layer_types", None)
        if not isinstance(layer_types, list):
            return set()
        return {idx for idx, layer_type in enumerate(layer_types) if layer_type == "linear_attention"}

    # Kimi K3: KDA layers (KimiDeltaAttention, ~69/93 layers) use a recurrent
    # state, not a paged KV cache. Their indices are listed in
    # ``text_config.linear_attn_config['kda_layers']`` (1-based, matching the
    # config's ``full_attn_layers`` which includes ``num_hidden_layers`` as the
    # last entry). Convert to 0-based to match the ``range(num_hidden_layers)``
    # convention used by ``_get_kv_cache_info`` and ``ModelRunner``.
    if model_type != "kimi_k3":
        return set()
    linear_attn_cfg = getattr(
        getattr(model, "text_config", None),
        "linear_attn_config",
        None,
    )
    if not isinstance(linear_attn_cfg, dict):
        return set()
    kda_layers = linear_attn_cfg.get("kda_layers")
    if not isinstance(kda_layers, (list, tuple)) or not kda_layers:
        return set()
    n_layers = getattr(
        getattr(model, "text_config", None),
        "num_hidden_layers",
        0,
    )
    # Detect 1-based vs 0-based indexing. K3's config lists layers starting from
    # 1 (e.g. kda_layers=[1,2,3,5,...,91], full_attn_layers=[4,8,...,92,93]).
    # The max-based check (max >= n_layers) is unreliable here because
    # kda_layers tops out at 91 while n_layers=93 (only full_attn_layers
    # reaches 93). Instead, probe whether index 0 appears in EITHER list: if
    # it does not, the indexing is 1-based and we subtract 1 to convert to the
    # 0-based convention used by range(num_hidden_layers).
    full_attn_layers = linear_attn_cfg.get("full_attn_layers")
    union_indices = set(kda_layers)
    if isinstance(full_attn_layers, (list, tuple)):
        union_indices.update(full_attn_layers)
    offset = 0 if 0 in union_indices else -1
    excluded = {idx + offset for idx in kda_layers if isinstance(idx, int) and idx + offset >= 0}
    logger.info(
        "kv_cache_excluded_layer_indices: kimi_k3 excluded %d KDA layers "
        "from KV cache accounting (out of %d total layers, offset=%d).",
        len(excluded),
        n_layers,
        offset,
    )
    return excluded


def _shard_kv_heads(total_kv_heads: int, tensor_parallel_size: int) -> int:
    """Return the number of KV heads owned by one tensor-parallel rank."""
    if total_kv_heads >= tensor_parallel_size:
        if total_kv_heads % tensor_parallel_size != 0:
            raise ValueError(
                f"total_kv_heads {total_kv_heads} is not divisible by tensor_parallel_size {tensor_parallel_size}"
            )
        return total_kv_heads // tensor_parallel_size

    if tensor_parallel_size % total_kv_heads != 0:
        raise ValueError(
            f"tensor_parallel_size {tensor_parallel_size} is not divisible by total_kv_heads {total_kv_heads}"
        )
    return 1


def _get_kv_cache_info(
    model,
    num_blocks: int,
    block_size: int,
    batch_size: Optional[int] = None,
    total_kv_tokens: Optional[int] = None,
    request_seq_lens: Optional[list[int]] = None,
    draft_tail_tokens: int = 0,
    unsharded_num_blocks: Optional[int] = None,
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
    excluded_layers = kv_cache_excluded_layer_indices(model)
    kv_cache_per_token = 0
    kv_cache_by_layers = {}
    draft_enabled = config_has_draft_spec(model_config)
    num_draft_layers = int(model_config.draft_num_layers()) if draft_enabled else 0
    num_target_layers = max(int(model.num_hidden_layers) - num_draft_layers, 0)
    draft_hf = None
    if draft_enabled:
        inner = getattr(model, "_inner", None)
        draft_hf = getattr(inner, "draft_hf_config", None)
        if draft_hf is None:
            # PipelineModel has no _inner; the speculative wrapper lives on the
            # last stage and exposes the draft HF config from there.
            draft_hf = getattr(model, "draft_hf_config", None)
    native_draft = getattr(getattr(model, "_inner", None), "draft", None)
    if native_draft is None:
        native_draft = getattr(model, "draft", None)
    native_dspark = bool(getattr(native_draft, "native_dspark", False))
    if draft_enabled and draft_hf is None and not native_dspark:
        # UnsupportedPPConfigurationError instead of a bare AttributeError:
        # the serving layer skips candidates raising it, while PP>1 re-raises
        # other exceptions loud and would abort the whole multi-candidate
        # search. Deferred import avoids a circular import.
        from ..pipeline_parallel import UnsupportedPPConfigurationError

        raise UnsupportedPPConfigurationError(
            "DFlash/DSpark is enabled but the draft HF config is not reachable "
            f"from {type(model).__name__}; refusing to allocate draft KV cache "
            "with the target model layout."
        )

    for i in range(model.num_hidden_layers):
        kvcache_dtype = _resolve_main_kv_cache_dtype(model, i)
        is_draft_layer = draft_enabled and i >= num_target_layers
        layer_num_blocks = num_blocks

        if (
            not is_draft_layer
            and getattr(model.text_config, "model_type", None) == "glm5_next_text"
            # MTP appends decoder layers using the final target layer type,
            # while model.text_config still describes only the target layers.
            and model.text_config.layer_types[min(i, len(model.text_config.layer_types) - 1)] == "linear_attention"
        ):
            kv_cache_by_layers[i] = torch.empty(
                glm5_next_kda_state_shape(
                    model.text_config, model_config.dtype, parallel_config.tensor_parallel_size, batch_size
                ),
                dtype=torch.uint8,
                device="meta",
            )
            # Count persistent state in total memory, but not in per-token KV
            # cache or KV transfer. This follows the Bailing KDA convention.
            continue

        if is_draft_layer:
            # Generic DFlash/DSpark uses Qwen3 GQA and owns separate K/V tensors.
            # DeepSeek V4.1 native DSpark instead owns one FP8 latent window per
            # draft layer, matching DSparkAttention's [B, window, head_dim] cache.
            layer_num_blocks = _resolve_draft_kv_cache_num_blocks(
                model,
                i - num_target_layers,
                num_blocks,
                block_size,
                batch_size,
                total_kv_tokens,
                request_seq_lens,
                draft_tail_tokens,
                unsharded_num_blocks,
            )
            if native_dspark:
                kv_cache_by_layers[i] = torch.empty(
                    [layer_num_blocks, block_size, int(model.head_dim)],
                    dtype=torch.float8_e4m3fn,
                    device="meta",
                )
            else:
                # Generic DFlash/DSpark requires its own draft config. The
                # validation above deliberately prevents falling back to the
                # target model's KV layout.
                draft_kv_heads = int(draft_hf.num_key_value_heads)
                draft_head_dim = int(
                    getattr(draft_hf, "head_dim", draft_hf.hidden_size // draft_hf.num_attention_heads)
                )
                kv_heads = _shard_kv_heads(draft_kv_heads, parallel_config.tensor_parallel_size)
                kv_cache_by_layers[i] = torch.empty(
                    [2, layer_num_blocks, block_size, kv_heads, draft_head_dim],
                    dtype=kvcache_dtype,
                    device="meta",
                )
        elif model_config.mla_config is not None:
            # decoder_layers may be None if _resolve_decoder_layers raises
            # AttributeError (e.g., model not fully wrapped). In that case
            # attention_layer stays None and the fallback formula below is used.
            attention_layer = None
            if decoder_layers is not None and i < len(decoder_layers):
                attention_layer = _resolve_decoder_attention_layer(decoder_layers[i])
            if attention_layer is not None and type(attention_layer).__name__ == "BailingMoeV3KimiDeltaAttention":
                kv_cache_by_layers[i] = torch.empty(
                    _resolve_bailing_v3_kda_cache_bytes(
                        attention_layer,
                        model_config,
                        parallel_config,
                        batch_size,
                    ),
                    dtype=torch.uint8,
                    device="meta",
                )
                # KDA state scales per request, not per token.
                continue
            if is_v4_model:
                kv_cache_shape = _resolve_v4_kv_cache_size(
                    model,
                    attention_layer,
                    num_blocks,
                    block_size,
                    batch_size,
                    total_kv_tokens,
                )
                rope_head_dim = int(getattr(model.text_config, "qk_rope_head_dim", 0))
                kv_cache_shape, kvcache_dtype = _v4_main_cache_storage(kv_cache_shape, kvcache_dtype, rope_head_dim)
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
        elif _is_mimo_v2_flash_model(model):
            pattern = model.text_config.hybrid_layer_pattern
            if not pattern:
                raise ValueError("MiMo V2 hybrid_layer_pattern must not be empty")
            is_swa = bool(pattern[i % len(pattern)])
            if is_swa:
                total_kv_heads = int(model.text_config.swa_num_key_value_heads)
                qk_head_dim = int(model.text_config.swa_head_dim)
                v_head_dim = int(model.text_config.swa_v_head_dim)
            else:
                total_kv_heads = int(model.text_config.num_key_value_heads)
                qk_head_dim = int(model.text_config.head_dim)
                v_head_dim = int(model.text_config.v_head_dim)

            kv_heads = _shard_kv_heads(total_kv_heads, parallel_config.tensor_parallel_size)

            # MiMo uses D_qk != D_v. Store one exact-size packed [K | V]
            # row instead of padding V to D_qk in a conventional [2, ...] cache.
            kv_cache_by_layers[i] = torch.empty(
                [
                    1,
                    num_blocks,
                    block_size,
                    kv_heads,
                    qk_head_dim + v_head_dim,
                ],
                dtype=kvcache_dtype,
                device="meta",
            )
        else:
            # Shape: [2 (K/V), num_blocks, block_size, num_heads, head_dim]
            kv_heads = _shard_kv_heads(
                int(model.text_config.num_key_value_heads),
                parallel_config.tensor_parallel_size,
            )

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
        # linear_attention layers keep a placeholder for PP/forward indexing but
        # do not own a real paged KV cache, so they must not inflate per-token cost.
        if i not in excluded_layers:
            kv_cache_per_token += bytes_of_tensor(kv_cache_by_layers[i]) / (layer_num_blocks * block_size)

    # Decode Context Parallel slices the KV cache along the token (sequence)
    # dimension: each device stores only ``1 / dcp_size`` of every sequence's
    # tokens, while the per-token byte cost on a device is UNCHANGED (the physical
    # slot ``[2, N_blk, block_size, h_kv/tp, D]`` stores the same bytes per token).
    # The saving is therefore "more tokens fit per device" (token capacity grows
    # by ``dcp_size``), NOT "each token costs less". So ``kv_cache_per_token`` is
    # left as the true physical per-card cost here; the DCP capacity multiplier is
    # applied where ``num_blocks`` is derived (serving warmup), via
    # ``dcp_kv_token_capacity_factor``. This keeps the reported per-token metric
    # and the KV-transfer byte accounting (get_kv_cache_num_bytes) physically
    # correct, instead of dividing the per-token cost as a prior version did.
    return kv_cache_by_layers, kv_cache_per_token


def _get_deepseek_v41_cache_ownership(model) -> dict[str, dict[int, Optional[int]]]:
    """Expose CSA2 logical-to-physical owner mappings to model invocations."""
    if not _is_deepseek_v41_config(model.text_config):
        return {}
    compressed = {}
    index = {}
    for layer_id in range(int(model.text_config.num_hidden_layers)):
        role = resolve_csa2_layer_role(model.text_config, layer_id)
        compressed[layer_id] = role.compressed_kv_owner
        index[layer_id] = role.index_k_owner
    return {"compressed_kv_owner_by_layer": compressed, "index_k_owner_by_layer": index}


def _get_deepseek_v41_state(model, requests: list[RequestInfo]) -> DeepseekV41SharedAttentionState:
    """Return workload-bound CSA2 state for one request lifecycle.

    The first request owns the batch state object; callers that reuse the same
    RequestInfo instances for subsequent decode calls therefore retain tails,
    while a different request list gets an isolated state without mutating the
    model object.
    """
    if not requests:
        return DeepseekV41SharedAttentionState()
    state = requests[0].csa2_state
    if state is None:
        if any(request.csa2_state is not None for request in requests):
            raise ValueError("All requests in a CSA2 batch must share one workload state")
        state = DeepseekV41SharedAttentionState()
        for request in requests:
            request.csa2_state = state
    elif any(request.csa2_state is not state for request in requests):
        raise ValueError("All requests in a CSA2 batch must share one workload state")
    return state


def _get_deepseek_v41_engram_state(model, requests: list[RequestInfo]) -> EngramHashState:
    """Return request-owned Engram history shared by packed requests."""
    if not _is_deepseek_v41_config(model.text_config):
        return EngramHashState()
    if not requests:
        return EngramHashState()
    state = requests[0].engram_hash_state
    if state is None:
        state = EngramHashState()
        for request in requests:
            request.engram_hash_state = state
    elif any(request.engram_hash_state is not state for request in requests):
        raise ValueError("Packed DeepSeek V4.1 requests must share one Engram hash state")
    return state


def _seed_deepseek_v41_draft_windows(model, state, batch_size: int) -> None:
    """Represent historical draft KV for standalone meta decode simulations.

    A real prefill publishes its windows to this same request state. Allocate
    only missing planes so subsequent invocations consume previous updates.
    """
    draft = getattr(getattr(model, "_inner", None), "draft", None)
    if not bool(getattr(draft, "native_dspark", False)):
        return
    for layer_id, layer in enumerate(draft.layers):
        if layer_id not in state.draft_kv_by_layer:
            state.draft_kv_by_layer[layer_id] = torch.empty(
                batch_size,
                layer.attn.window_size,
                layer.attn.head_dim,
                dtype=torch.float8_e4m3fn,
                device="meta",
            )


def reset_deepseek_v41_state(model=None, requests: Optional[list[RequestInfo]] = None) -> None:
    """Reset request-owned CSA2 state while retaining the legacy call shape."""
    del model
    if requests:
        seen = set()
        for request in requests:
            state = request.csa2_state
            if state is not None and id(state) not in seen:
                state.reset()
                seen.add(id(state))
            request.csa2_state = None


def _get_deepseek_v41_compressed_cache_info(
    model, block_size: int, batch_size: Optional[int], total_kv_tokens: Optional[int]
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], int]:
    """Allocate one compressed-KV cache per configured CSA2 source layer."""
    config = getattr(model, "text_config", None)
    if not _is_deepseek_v41_config(config):
        return {}, {}, 0
    total_tokens = max(int(total_kv_tokens or 0), 1)
    head_dim = int(getattr(config, "head_dim", getattr(model, "head_dim", 1)))
    dtype = torch.int4
    scale_dtype = torch.float8_e4m3fn
    caches: dict[int, torch.Tensor] = {}
    scale_caches: dict[int, torch.Tensor] = {}
    per_token = 0
    for source in tuple(getattr(config, "kv_source_layer_ids", ())):
        ratio = max(int(config.compress_ratios[int(source)]), 1)
        slots = max(1, (total_tokens + ratio - 1) // ratio)
        # ``total_kv_tokens`` already covers the whole packed batch. Do not
        # multiply by batch again, or memory grows quadratically with batch size.
        blocks = max(1, (slots + block_size - 1) // block_size)
        cache = torch.empty((blocks, block_size, head_dim), dtype=dtype, device="meta")
        scale_cache = torch.empty((blocks, block_size, (head_dim + 16 - 1) // 16), dtype=scale_dtype, device="meta")
        caches[int(source)] = cache
        scale_caches[int(source)] = scale_cache
        per_token += (bytes_of_tensor(cache) + bytes_of_tensor(scale_cache)) / max(blocks * block_size, 1)
    return caches, scale_caches, per_token


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

    if _is_deepseek_v41_config(model.model_config.hf_config):
        # V4.1 owns compressed KV in a separate source cache.  This tensor is
        # therefore the window-only ring buffer; compressed source caches are
        # allocated by _get_deepseek_v41_compressed_cache_info().
        if batch_size is not None:
            window_slots = max(int(batch_size), 1) * max(window_size, 1)
            window_blocks = max(1, (window_slots + block_size - 1) // block_size)
            return [window_blocks, block_size, head_dim]
        return [num_blocks, block_size, head_dim]

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


def get_kv_cache_info(
    model,
    num_blocks,
    block_size,
    batch_size=None,
    total_kv_tokens=None,
    request_seq_lens=None,
    draft_tail_tokens=0,
    unsharded_num_blocks=None,
):
    return _get_kv_cache_info(
        model,
        num_blocks,
        block_size,
        batch_size,
        total_kv_tokens,
        request_seq_lens,
        draft_tail_tokens,
        unsharded_num_blocks,
    )


def _resolve_decoder_layers(model):
    """Resolve the decoder layers ``ModuleList`` regardless of how the model
    is wrapped.

    msmodeling can wrap the underlying HF model in several layouts:
        * ``TransformerModel(_inner=CausalLmWrapper(_inner=HFModel))``
        * ``TransformerModel(_inner=ModelWrapper(_inner=HFModel))``
        * ``TransformerModel(_inner=ConditionalGeneration(language_model=...))``
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
    if hasattr(inner, "stages") and hasattr(inner, "num_hidden_layers"):
        layers = [None] * int(inner.num_hidden_layers)
        for stage in inner.stages:
            stage_layers = _resolve_decoder_layers(stage.model)
            layer_start = int(stage.stage_spec.layer_start)
            for offset, layer in enumerate(stage_layers):
                layer_idx = layer_start + offset
                if 0 <= layer_idx < len(layers):
                    layers[layer_idx] = layer
        # Only raise for missing decoder layers; MTP proposal layers live in
        # MtpWrapper.mtp (not in any stage's decoder ModuleList) so their
        # indices legitimately remain None. Downstream _get_kv_cache_info
        # handles None entries with the generic KV formula, which is correct
        # for MTP blocks (they use standard MLA attention).
        plan = getattr(inner, "plan", None)
        decoder_count = int(plan.num_hidden_layers) if plan is not None else len(layers)
        missing_decoder = [idx for idx in range(decoder_count) if layers[idx] is None]
        if missing_decoder:
            raise AttributeError(f"Unable to locate pipeline decoder layer(s): {missing_decoder}")
        return layers
    language_model = getattr(inner, "language_model", None)
    if language_model is not None and hasattr(language_model, "layers"):
        return language_model.layers
    raise AttributeError(
        "Unable to locate decoder layers in `unwrap().layers`, "
        "`unwrap().model.layers`, or `unwrap().language_model.layers`"
    )


def get_sparse_attention_indexer_cache_info(model, num_blocks, block_size, batch_size=None, total_kv_tokens=None):
    """Allocate per-layer auxiliary indexer caches for sparse-attention wrappers.

    Despite the older DSA-oriented naming in surrounding code, this helper is
    also used by DeepSeek V4's custom sparse attention path, whose ratio=4
    layers carry a distinct learned indexer.

    GLM-5.2 IndexShare decides whether a layer owns cache; V4 compression
    decides how many blocks each allocated cache needs. DFlash/DSpark draft
    layers are Qwen3 GQA and are skipped: they are not part of the target
    sparse-indexer stack.

    For V4 the indexer cache is purely compressed (no sliding window): the
    reference allocates ``[max_batch_size, max_seq_len // compress_ratio,
    index_head_dim]`` (model.py:399). When ``batch_size`` and
    ``total_kv_tokens`` are provided we size the paged cache to
    ``total_kv_tokens // compress_ratio`` slots instead of the full pool.
    """
    model_config = model.model_config
    mla_config = model_config.mla_config
    try:
        decoder_layers = _resolve_decoder_layers(model)
    except AttributeError:
        decoder_layers = None

    has_sparse_indexer = False
    if decoder_layers is not None:
        has_sparse_indexer = any(
            _layer_uses_sparse_attention_indexer(
                _resolve_decoder_attention_layer(layer, preserve_attention_wrapper=True)
            )
            for layer in decoder_layers
        )

    mla_requires_indexer = (
        mla_config is not None and mla_config.mla_cls is not None and mla_config.mla_cls.requires_indexer_cache()
    )
    if not has_sparse_indexer and not mla_requires_indexer:
        return {}

    is_v4_model = _is_v4_model(model)
    # GLM-5.2 IndexShare is a layer-selection rule: shared layers reuse top-k and own no cache.
    glm5_indexer_types = _get_glm5_indexshare_indexer_types(model)
    num_draft_layers = _draft_layer_count(model_config)
    num_target_layers = max(int(model.num_hidden_layers) - num_draft_layers, 0)
    indexer_cache_by_layers = {}
    indexer_scale_cache_by_layers = {}
    indexer_cache_per_token = 0
    for i in range(model.num_hidden_layers):
        # Qwen3 GQA draft is not part of the target GLM5/DSA indexer stack.
        if num_draft_layers and i >= num_target_layers:
            continue
        try:
            owns_indexer_cache = _glm5_indexshare_layer_owns_indexer_cache(glm5_indexer_types, i)
        except ValueError as err:
            raise ValueError(
                f"Invalid GLM5 indexer_types for layer {i}/{model.num_hidden_layers}: {glm5_indexer_types}"
            ) from err
        if not owns_indexer_cache:
            continue

        attention_layer = (
            _resolve_decoder_attention_layer(
                decoder_layers[i],
                preserve_attention_wrapper=True,
            )
            if decoder_layers is not None and i < len(decoder_layers)
            else None
        )
        if attention_layer is not None and not _layer_uses_sparse_attention_indexer(attention_layer):
            continue

        cache_width = _resolve_sparse_attention_indexer_cache_width(model, attention_layer)
        if cache_width is None:
            continue

        cache_dtype = _resolve_indexer_cache_dtype(model, i, attention_layer)

        indexer_num_blocks = _resolve_sparse_attention_indexer_num_blocks(
            is_v4_model,
            attention_layer,
            num_blocks,
            block_size,
            batch_size,
            total_kv_tokens,
        )

        indexer_cache_by_layers[i] = torch.empty(
            [
                indexer_num_blocks,
                block_size,
                cache_width,
            ],
            dtype=cache_dtype,
            device="meta",
        )
        is_v41 = _is_deepseek_v41_config(model.model_config.hf_config) or _is_deepseek_v41_config(
            getattr(model, "text_config", None)
        )
        if is_v41:
            indexer_scale_cache_by_layers[i] = torch.empty(
                [indexer_num_blocks, block_size, (cache_width + 32 - 1) // 32],
                dtype=torch.float8_e8m0fnu,
                device="meta",
            )
            role = resolve_csa2_layer_role(model.text_config, i)
            owner = role.index_k_owner
            if owner is not None and owner != i and owner in indexer_cache_by_layers:
                indexer_cache_by_layers[i] = indexer_cache_by_layers[owner]
                if owner in indexer_scale_cache_by_layers:
                    indexer_scale_cache_by_layers[i] = indexer_scale_cache_by_layers[owner]
                continue
        cache_bytes = bytes_of_tensor(indexer_cache_by_layers[i])
        if is_v41:
            cache_bytes += bytes_of_tensor(indexer_scale_cache_by_layers[i])
        indexer_cache_per_token += cache_bytes / max(num_blocks * block_size, 1)

    return {
        "indexer_cache_by_layers": indexer_cache_by_layers,
        "indexer_scale_cache_by_layers": indexer_scale_cache_by_layers,
        "indexer_cache_per_token": indexer_cache_per_token,
    }


def generate_inputs_varlen(model, requests: list[RequestInfo], block_size):
    """
    requests: List[RequestInfo], each dict represents a request, containing keys: query_len, seq_len, is_decode
    """
    model_config = model.model_config
    mtp = getattr(model_config, "mtp_config", None)
    num_mtp_tokens = mtp.num_mtp_layers if mtp else 0
    draft_block = int(model_config.draft_block_size()) if config_has_draft_spec(model_config) else 0
    dflash_spec_tokens = max(draft_block - 1, 0)

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

    max_total_seq_len = int(max(seq_lens))
    speculative_tail = max(num_mtp_tokens, dflash_spec_tokens) + 1
    total_kv_tokens = sum(seq_lens) + batch_size * speculative_tail
    num_blocks = (total_kv_tokens + block_size - 1) // block_size
    # Decode Context Parallel stores only ``1 / dcp`` of each sequence's tokens on
    # a card, so where that shard is a real per-card saving the physical KV footprint
    # shrinks accordingly (per-token cost is invariant; bytes and the num_blocks
    # denominator scale together in ``_get_kv_cache_info``). The factor is layout-
    # dependent -- ``dcp`` for MLA, capped by TP KV replication for GQA, 1 for SFA;
    # see ``dcp_kv_token_capacity_factor``.
    kv_num_blocks = dcp_sharded_num_blocks(model, num_blocks)
    max_num_blocks_per_seq = (max_total_seq_len + block_size - 1) // block_size
    block_table_tensor = torch.empty((batch_size, max_num_blocks_per_seq), dtype=torch.long, device="meta")
    slot_mapping = torch.empty((num_tokens,), dtype=torch.long, device="meta")
    compressed_slot_mapping = torch.empty((num_tokens,), dtype=torch.long, device="meta")

    attn_meta = AttentionMetadataTensorCast(
        query_start_loc=query_start_loc,
        query_lens=query_len_t,
        seq_lens=seq_lens_t,
        query_lens_values=list(query_lens),
        seq_lens_values=list(seq_lens),
        is_decode_values=list(is_decode_list),
        block_table_tensor=block_table_tensor,
        slot_mapping=slot_mapping,
        max_total_seq_len=max_total_seq_len,
        start_pos=(seq_lens[0] - query_lens[0]) if len(set(s - q for s, q in zip(seq_lens, query_lens))) == 1 else None,
        compressed_slot_mapping=compressed_slot_mapping,
    )

    input_ids = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
    position_ids = torch.empty([1, num_tokens], dtype=torch.long, device="meta")

    kv_cache_by_layers, kv_cache_per_token = get_kv_cache_info(
        model,
        kv_num_blocks,
        block_size,
        batch_size,
        total_kv_tokens,
        seq_lens,
        speculative_tail,
        num_blocks,
    )
    compressed_cache_by_source, compressed_scale_cache_by_source, compressed_cache_per_token = (
        _get_deepseek_v41_compressed_cache_info(model, block_size, batch_size, total_kv_tokens)
    )

    sampling_meta = SamplingMetadata(query_start_loc=query_start_loc)
    # Pure decode batches use the common verification window. Mixed MTP batches
    # instead record one selection row for prefill and a window for each decode.
    # Short decode windows use ordinary per-request selection.
    # When enabled, pass the per-request query_lens list so packed offsets honor varlen batches.
    use_mtp_spec_decode = (
        num_mtp_tokens > 0 and all(is_decode_list) and all(query_len >= num_mtp_tokens + 1 for query_len in query_lens)
    )
    use_dflash_spec_decode = (
        dflash_spec_tokens > 0 and all(is_decode_list) and all(query_len >= draft_block for query_len in query_lens)
    )
    if num_mtp_tokens > 0 and any(is_decode_list) and not all(is_decode_list):
        indices = []
        lengths = []
        offset = 0
        for ql, decode in zip(query_lens, is_decode_list):
            length = num_mtp_tokens + 1 if decode else 1
            if ql < length:
                raise ValueError("Mixed MTP decode requires a full verification window")
            indices.extend(range(offset + ql - length, offset + ql))
            lengths.append(length)
            offset += ql
        sampling_meta.spec_decode_metadata = SpecDecodeMetadata(
            torch.tensor(indices, device=query_start_loc.device, dtype=torch.long),
            len(lengths),
            num_mtp_tokens,
            tuple(lengths),
        )
        sampling_meta.selected_token_indices = None
    elif use_mtp_spec_decode:
        sampling_meta.spec_decode_metadata = _build_spec_decode_metadata(
            query_start_loc,
            query_lens,
            num_mtp_tokens,
        )
        sampling_meta.selected_token_indices = None
    elif use_dflash_spec_decode:
        sampling_meta.spec_decode_metadata = _build_spec_decode_metadata(
            query_start_loc,
            query_lens,
            dflash_spec_tokens,
        )
        sampling_meta.selected_token_indices = None
    elif num_mtp_tokens > 0 and all(is_decode_list):
        sampling_meta.selected_token_indices = None
    else:
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
        "kv_cache_excluded_layer_indices": kv_cache_excluded_layer_indices(model),
        "sampling_metadata": sampling_meta,
        "kv_cache_per_token": kv_cache_per_token,
    }
    if _is_deepseek_v41_config(model.model_config.hf_config):
        kwargs["csa2_state"] = _get_deepseek_v41_state(model, requests)
        if all(request.is_decode for request in requests):
            _seed_deepseek_v41_draft_windows(model, kwargs["csa2_state"], batch_size)
        kwargs["engram_hash_state"] = _get_deepseek_v41_engram_state(model, requests)
        kwargs["engram_token_ids"] = torch.empty_like(input_ids)
        kwargs["csa2_cache_ownership"] = _get_deepseek_v41_cache_ownership(model)
        kwargs["compressed_kv_cache_by_source"] = compressed_cache_by_source
        kwargs["compressed_kv_scale_cache_by_source"] = compressed_scale_cache_by_source
        kwargs["compressed_kv_cache_per_token"] = compressed_cache_per_token

    sparse_attention_indexer_cache = get_sparse_attention_indexer_cache_info(
        model, num_blocks, block_size, batch_size, total_kv_tokens
    )
    kwargs.update(sparse_attention_indexer_cache)

    if model.model_config.hf_config.model_type in (
        "qwen3_next",
        "qwen3_5",
        "qwen3_5_moe",
        "qwen3_5_moe_text",
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


def _deepseek_v41_image_grid(image_height: int, image_width: int, vision_config) -> tuple[int, int, int, int]:
    """Return ViT and LLM grid sizes using the reference V4.1 resize rules."""
    patch = int(vision_config.patch_size)
    ratio = int(vision_config.downsample_ratio)
    width = int(image_width)
    height = int(image_height)
    max_wh_ratio = getattr(vision_config, "max_wh_ratio", None)
    if max_wh_ratio is not None and width > height * float(max_wh_ratio):
        width = int(height * float(max_wh_ratio))
    min_pixels = int(getattr(vision_config, "min_pixels", 0) or 0)
    if 0 < width * height < min_pixels:
        scale = math.sqrt(min_pixels / (width * height))
        width = int(width * scale)
        height = int(height * scale)

    best_width = math.ceil(width / patch) * patch
    best_height = math.ceil(height / patch) * patch

    def grids(pixel_height: int, pixel_width: int) -> tuple[int, int, int, int]:
        vit_h, vit_w = pixel_height // patch, pixel_width // patch
        llm_h = math.ceil(vit_h / ratio)
        llm_w = math.ceil(vit_w / ratio)
        return vit_h, vit_w, llm_h, llm_w

    vit_h, vit_w, llm_h, llm_w = grids(best_height, best_width)
    max_tokens = int(getattr(vision_config, "max_image_tokens", 1024))
    if llm_h * (llm_w + 1) + VISION_IMAGE_BOUNDARY_TOKEN_COUNT > max_tokens:
        aspect = height / width
        max_w_float = math.sqrt((max_tokens - 2) / aspect + 0.25) - 0.5
        max_h_float = max_w_float * aspect
        cell = patch * ratio
        if max_w_float < 1.0:
            best_height, best_width = ((max_tokens - 2) // 2 * cell, cell)
        elif max_h_float < 1.0:
            best_height, best_width = (cell, (max_tokens - 3) * cell)
        else:
            scale = min(
                math.floor(max_w_float) * cell / width,
                math.floor(max_h_float) * cell / height,
            )
            best_height = math.floor(height * scale / patch) * patch
            best_width = math.floor(width * scale / patch) * patch
        vit_h, vit_w, llm_h, llm_w = grids(best_height, best_width)
    return vit_h, vit_w, llm_h, llm_w


def _generate_deepseek_v41_image_inputs(
    model,
    image_batch_size: int,
    image_height: int,
    image_width: int,
    concurrency: int,
) -> dict[str, Any]:
    vision_config = model.model_config.hf_config.vision_config
    vit_h, vit_w, llm_h, llm_w = _deepseek_v41_image_grid(image_height, image_width, vision_config)
    patch = int(vision_config.patch_size)
    channels = int(getattr(vision_config, "in_channels", 3))
    parallel_config = model.model_config.parallel_config
    batch_size = (concurrency + parallel_config.data_parallel_size - 1) // parallel_config.data_parallel_size
    num_images = batch_size * int(image_batch_size)
    pixel_values = torch.empty(
        num_images,
        vit_h * vit_w,
        channels,
        patch,
        patch,
        dtype=model.model_config.dtype,
        device="meta",
    )
    image_span_tokens = llm_h * (llm_w + 1) + VISION_IMAGE_BOUNDARY_TOKEN_COUNT
    return {
        "pixel_values": pixel_values,
        "vision_grid_height": vit_h,
        "vision_grid_width": vit_w,
        "num_image_tokens": int(image_batch_size) * image_span_tokens,
    }


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
