import logging
import threading
from contextlib import contextmanager
from typing import Optional

import diffusers
import torch
import torch.nn.functional as F
from aenum import extend_enum
from diffusers.models.attention_dispatch import AttentionBackendName, _AttentionBackendRegistry

from ..model_config import AttentionBackend
from ..parallel_group import ParallelGroup

_thread_local = threading.local()
logger = logging.getLogger(__name__)


if not hasattr(diffusers.models.attention_dispatch.AttentionBackendName, "TENSOR_CAST"):
    extend_enum(
        diffusers.models.attention_dispatch.AttentionBackendName,
        "TENSOR_CAST",
        "tensor_cast",
    )


def set_sp_group(sp_group: Optional[ParallelGroup]):
    _thread_local.sp_group = sp_group


def get_sp_group() -> Optional[ParallelGroup]:
    return getattr(_thread_local, "sp_group", None)


def get_attention_quant_config():
    return getattr(_thread_local, "attention_quant_config", None)


def get_attention_route_plan():
    return getattr(_thread_local, "attention_route_plan", None)


def _get_attention_stats():
    return getattr(_thread_local, "attention_stats", None)


def _get_block_sparse_attention_fallback_reason(
    query,
    key,
    value,
    route_plan,
    attention_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
    canonical_layout=False,
):
    if route_plan is None or route_plan.backend != AttentionBackend.block_sparse_attention:
        return None
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        return "non_4d_qkv"
    if query.shape != key.shape or query.shape != value.shape:
        return "qkv_shape_mismatch"
    if not _has_supported_attention_mask(attention_mask, query, key, canonical_layout):
        return "unsupported_attention_mask"
    if dropout_p != 0.0:
        return "dropout"
    if is_causal:
        return "causal"
    if scale is not None:
        return "custom_scale"
    if enable_gqa:
        return "gqa"
    return None


def _record_attention_call(reason):
    stats = _get_attention_stats()
    route_plan = get_attention_route_plan()
    if stats is None or route_plan is None or route_plan.backend != AttentionBackend.block_sparse_attention:
        return
    if reason is None:
        stats["block_sparse_attention_calls"] += 1
    else:
        stats["dense_fallback_calls"] += 1
        reasons = stats["dense_fallback_reasons"]
        reasons[reason] = reasons.get(reason, 0) + 1


def _has_supported_attention_mask(attention_mask, query, key, canonical_layout) -> bool:
    if attention_mask is None:
        return True
    if not isinstance(attention_mask, torch.Tensor) or attention_mask.dtype != torch.bool or attention_mask.ndim != 4:
        return False

    sequence_dim = 1 if canonical_layout else -2
    return attention_mask.shape == (
        query.shape[0],
        1,
        query.shape[sequence_dim],
        key.shape[sequence_dim],
    )


def _run_attention(
    query,
    key,
    value,
    attention_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
    canonical_layout=False,
):
    quant_config = get_attention_quant_config()
    route_plan = get_attention_route_plan()
    fallback_reason = _get_block_sparse_attention_fallback_reason(
        query,
        key,
        value,
        route_plan,
        attention_mask,
        dropout_p,
        is_causal,
        scale,
        enable_gqa,
        canonical_layout,
    )
    _record_attention_call(fallback_reason)
    if (
        fallback_reason is None
        and route_plan is not None
        and route_plan.backend == AttentionBackend.block_sparse_attention
    ):
        route_metadata = torch.ops.tensor_cast.attention_route_generate(
            query, key, route_plan.block_size, route_plan.sparsity
        )
        return torch.ops.tensor_cast.block_sparse_attention(
            query, key, value, attention_mask, route_metadata, route_plan.block_size, route_plan.sparsity
        )
    if quant_config is None:
        return torch.ops.tensor_cast.attention(query, key, value, attention_mask, None, None, None, None)

    out_dtype = query.dtype
    quant_dtype = quant_config.get_quant_dtype()
    query = torch.ops.tensor_cast.quantize(query, quant_config.query_scale, quant_config.query_offset, quant_dtype)
    key = torch.ops.tensor_cast.quantize(key, quant_config.kv_scale, quant_config.kv_offset, quant_dtype)
    value = torch.ops.tensor_cast.quantize(value, quant_config.kv_scale, quant_config.kv_offset, quant_dtype)
    return torch.ops.tensor_cast.attention_quant(
        query,
        key,
        value,
        attention_mask,
        None,
        None,
        None,
        None,
        quant_config.query_scale,
        quant_config.query_offset,
        quant_config.kv_scale,
        quant_config.kv_offset,
        quant_config.attention_prob_scale,
        quant_config.attention_prob_offset,
        out_dtype,
    )


@_AttentionBackendRegistry.register(AttentionBackendName.TENSOR_CAST)
def _attention(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
    **kwargs,
):
    sp_group = get_sp_group()
    if sp_group is None:
        return _run_attention(
            query,
            key,
            value,
            attn_mask,
            dropout_p,
            is_causal,
            scale,
            enable_gqa,
            True,
        )

    route_plan = get_attention_route_plan()
    if (
        _get_block_sparse_attention_fallback_reason(
            query,
            key,
            value,
            route_plan,
            attn_mask,
            dropout_p,
            is_causal,
            scale,
            enable_gqa,
            True,
        )
        == "non_4d_qkv"
    ):
        return _run_attention(
            query,
            key,
            value,
            attn_mask,
            dropout_p,
            is_causal,
            scale,
            enable_gqa,
            True,
        )

    ulysses_size = sp_group.world_size

    # all-to-all: (b, s, h, w) -> (b, s * p, h, w / p)
    # In cross attention, query shape is not equal to key, value shape
    batch_size, seq_per_rank, num_heads, head_dim = query.shape
    batch_size_kv, seq_per_rank_kv, num_heads_kv, head_dim_kv = key.shape
    input_tensor_q = torch.ones(
        (batch_size, seq_per_rank, num_heads // ulysses_size, head_dim),
        dtype=query.dtype,
        device=query.device,
    )
    input_tensor_kv = torch.ones(
        (batch_size_kv, seq_per_rank_kv, num_heads_kv // ulysses_size, head_dim_kv),
        dtype=query.dtype,
        device=query.device,
    )
    input_split_sizes = [1 for _ in range(ulysses_size - 1)]
    output_split_sizes = [1 for _ in range(ulysses_size - 1)]

    _ = sp_group.all_to_all(
        input_tensor_q,
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
    )
    _ = sp_group.all_to_all(
        input_tensor_kv,
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
    )
    _ = sp_group.all_to_all(
        input_tensor_kv,
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
    )
    query = query.view(batch_size, seq_per_rank * ulysses_size, num_heads // ulysses_size, head_dim)
    key = key.view(
        batch_size_kv,
        seq_per_rank_kv * ulysses_size,
        num_heads_kv // ulysses_size,
        head_dim_kv,
    )
    value = value.view(
        batch_size_kv,
        seq_per_rank_kv * ulysses_size,
        num_heads_kv // ulysses_size,
        head_dim_kv,
    )
    out = _run_attention(
        query,
        key,
        value,
        attn_mask,
        dropout_p,
        is_causal,
        scale,
        enable_gqa,
        True,
    )

    _ = sp_group.all_to_all(
        input_tensor_q,
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
    )
    out = out.view(batch_size, seq_per_rank, num_heads, head_dim)
    return out


# scaled_dot_product_attention is not capturable by torch_dispatch;
# override it with our custom tensor_cast attention op instead.
@contextmanager
def use_custom_sdpa(quant_config=None, route_plan=None):
    if (
        route_plan is not None
        and route_plan.backend == AttentionBackend.block_sparse_attention
        and quant_config is not None
    ):
        raise ValueError("block_sparse_attention does not support attention quantization")

    original_sdpa = F.scaled_dot_product_attention
    original_quant_config = get_attention_quant_config()
    original_route_plan = get_attention_route_plan()
    original_attention_stats = _get_attention_stats()
    attention_stats = {
        "block_sparse_attention_calls": 0,
        "dense_fallback_calls": 0,
        "dense_fallback_reasons": {},
    }

    def _custom_sdpa(
        q,
        k,
        v,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
        enable_gqa=False,
    ):
        route_plan = get_attention_route_plan()
        canonical_qkv = None
        if q.ndim == k.ndim == v.ndim == 4:
            canonical_qkv = (q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        if (
            canonical_qkv is not None
            and route_plan is not None
            and route_plan.backend == AttentionBackend.block_sparse_attention
        ):
            q_canonical, k_canonical, v_canonical = canonical_qkv
            output = _run_attention(
                q_canonical,
                k_canonical,
                v_canonical,
                attn_mask,
                dropout_p,
                is_causal,
                scale,
                enable_gqa,
                True,
            )
            return output.transpose(1, 2)
        return _run_attention(q, k, v, attn_mask, dropout_p, is_causal, scale, enable_gqa)

    _thread_local.attention_quant_config = quant_config
    _thread_local.attention_route_plan = route_plan
    _thread_local.attention_stats = attention_stats
    F.scaled_dot_product_attention = _custom_sdpa
    try:
        yield attention_stats
    finally:
        F.scaled_dot_product_attention = original_sdpa
        _thread_local.attention_quant_config = original_quant_config
        _thread_local.attention_route_plan = original_route_plan
        _thread_local.attention_stats = original_attention_stats
        if attention_stats["dense_fallback_calls"]:
            reasons = ", ".join(
                f"{reason}={count}" for reason, count in sorted(attention_stats["dense_fallback_reasons"].items())
            )
            logger.warning(
                "Block sparse attention fallback: block_sparse_attention_calls=%d, dense_fallback_calls=%d, reasons=%s",
                attention_stats["block_sparse_attention_calls"],
                attention_stats["dense_fallback_calls"],
                reasons,
            )
