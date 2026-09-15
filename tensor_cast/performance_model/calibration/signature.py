"""Build calibration keys from TensorCast operation semantics.

The signature intentionally uses only ``OpInvokeInfo``, ``DeviceProfile`` and
the raw analytic result.  In particular, it never resolves an NPU kernel name
or reads ``op_mapping.yaml``.
"""

from dataclasses import dataclass
from math import prod
from typing import Any, Mapping, Optional

import torch

from ..base import PerformanceModel
from ..op_invoke_info import OpInvokeInfo


def _op_name(op_invoke_info: OpInvokeInfo) -> str:
    return str(op_invoke_info.func).removeprefix("torch.ops.")


def _tensor_args(value: Any) -> list[torch.Tensor]:
    """Collect tensors from positional arguments without inspecting values."""
    tensors: list[torch.Tensor] = []
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (tuple, list)):
        for item in value:
            tensors.extend(_tensor_args(item))
    return tensors


def _dtype_name(tensor: Optional[torch.Tensor]) -> Optional[str]:
    if tensor is None:
        return None
    return str(tensor.dtype).removeprefix("torch.")


def _matmul_features(tensors: list[torch.Tensor]) -> dict[str, Any]:
    if len(tensors) < 2 or tensors[0].ndim < 2 or tensors[1].ndim < 2:
        return {}
    lhs, rhs = tensors[:2]
    features = {
        "m": int(lhs.shape[-2]),
        "k": int(lhs.shape[-1]),
        "n": int(rhs.shape[-1]),
    }
    if any(value <= 0 for value in features.values()):
        return {}
    if lhs.ndim > 2:
        features["batch"] = int(prod(lhs.shape[:-2]))
    return _with_mm_derived_features(features)


def _with_mm_derived_features(features: dict[str, Any]) -> dict[str, Any]:
    k = features.get("k")
    n = features.get("n")
    m = features.get("m")
    if not all(isinstance(value, int) and value > 0 for value in (m, k, n)):
        return features
    features.update(
        {
            "min_kn": min(k, n),
            "max_kn": max(k, n),
            "mn": m * n,
            "aspect": max(k, n) / min(k, n),
        }
    )
    return features


def _linear_features(op_name: str, op_invoke_info: OpInvokeInfo) -> dict[str, Any]:
    if len(op_invoke_info.args) < 2:
        return {}
    x, weight = op_invoke_info.args[:2]
    if not isinstance(x, torch.Tensor) or not isinstance(weight, torch.Tensor) or x.ndim < 2 or weight.ndim < 2:
        return {}
    m = int(prod(x.shape[:-1]))
    k = int(x.shape[-1])
    if m <= 0 or k <= 0:
        return {}
    if op_name == "tensor_cast.static_quant_linear_int4.default":
        pack_factor = weight.element_size() * 8 // 4
        logical_elements = weight.numel() * pack_factor
        if logical_elements % k:
            return {}
        n = int(logical_elements // k)
    else:
        n = int(weight.shape[-1])
    return _with_mm_derived_features({"m": m, "k": k, "n": n})


_GMM_VARIANTS = {
    "tensor_cast.grouped_matmul_quant.default": ("plain", "w8a8"),
    "tensor_cast.grouped_matmul_quant_swiglu.default": ("swiglu_quant", "w8a8"),
}


def _grouped_weight_dims(weight: torch.Tensor, k: int, output_n: int, *, swiglu: bool) -> tuple[int, str]:
    """Return logical GMM N and the physical weight orientation.

    TensorCast keeps each expert weight as a rank-2 ``[K, N]`` tensor.  CSV
    rows may expose either ``[E, K, N]`` or ``[E, N, K]`` (and FRACTAL_NZ
    restores to one of those orientations), so the activation K and output N
    are used as the disambiguating evidence.
    """
    if weight.ndim != 2:
        return 0, "unknown"
    first, second = int(weight.shape[0]), int(weight.shape[1])
    logical_n = output_n * 2 if swiglu else output_n
    if (first, second) == (k, logical_n):
        return logical_n, "kn"
    if (first, second) == (logical_n, k):
        return logical_n, "nk"
    return 0, "unknown"


def _grouped_features(op_name: str, op_invoke_info: OpInvokeInfo) -> dict[str, Any]:
    variant_info = _GMM_VARIANTS.get(op_name)
    if variant_info is None or len(op_invoke_info.args) < 2:
        return {}
    variant, quantization = variant_info
    x_values, weight_values = op_invoke_info.args[:2]
    if not isinstance(x_values, (list, tuple)) or not isinstance(weight_values, (list, tuple)):
        return {}
    x_tensors = [value for value in x_values if isinstance(value, torch.Tensor)]
    weight_tensors = [value for value in weight_values if isinstance(value, torch.Tensor)]
    if not x_tensors or len(x_tensors) != len(weight_tensors):
        return {}
    if any(tensor.ndim != 2 for tensor in [*x_tensors, *weight_tensors]):
        return {}
    m_values = [int(tensor.shape[0]) for tensor in x_tensors]
    k = int(x_tensors[0].shape[1])
    if any(value < 0 for value in m_values) or sum(m_values) <= 0 or k <= 0:
        return {}
    if any(int(tensor.shape[1]) != k for tensor in x_tensors):
        return {}
    output = op_invoke_info.out
    output_n = int(output.shape[-1]) if isinstance(output, torch.Tensor) and output.ndim >= 2 else 0
    if output_n <= 0:
        return {}
    swiglu = variant == "swiglu_quant"
    n_values = []
    orientations = []
    for weight in weight_tensors:
        n_value, orientation = _grouped_weight_dims(weight, k, output_n, swiglu=swiglu)
        if not n_value:
            return {}
        n_values.append(n_value)
        orientations.append(orientation)
    if len(set(n_values)) != 1 or len(set(orientations)) != 1:
        return {}
    nonzero = [value for value in m_values if value > 0]
    m_total = sum(m_values)
    mean_nonzero = sum(nonzero) / len(nonzero) if nonzero else 0.0
    max_m = max(nonzero, default=0)
    imbalance = max_m / mean_nonzero if mean_nonzero else 0.0
    distribution = "balanced" if max(m_values) - min(m_values) <= 1 else "skewed"
    return {
        "mm_kind": "grouped",
        "gmm_variant": variant,
        "quantization": quantization,
        "compute_dtype": _dtype_name(x_tensors[0]),
        "weight_dtype": _dtype_name(weight_tensors[0]),
        "output_dtype": _dtype_name(output) if isinstance(output, torch.Tensor) else None,
        "m_total": m_total,
        "k": k,
        "n": output_n,
        "gemm_n": n_values[0],
        "num_experts": len(x_tensors),
        "nonzero_experts": len(nonzero),
        "max_m": max_m,
        "min_nonzero_m": min(nonzero, default=0),
        "imbalance_ratio": imbalance,
        "distribution": distribution,
        "weight_orientation": orientations[0],
    }


_LINEAR_QUANTIZATION = {
    "tensor_cast.static_quant_linear.default": "w8a8",
    "tensor_cast.static_quant_linear_int4.default": "w4a8",
    "tensor_cast.fp8_linear.default": "fp8",
    "tensor_cast.mxfp4_linear.default": "mxfp4",
}


def _attention_features(tensors: list[torch.Tensor], kwargs: Mapping[str, Any]) -> dict[str, Any]:
    if len(tensors) < 2:
        return {}
    query, key = tensors[:2]
    if query.ndim < 2 or key.ndim < 2:
        return {}
    features: dict[str, Any] = {
        "q_tokens": int(prod(query.shape[:-2])) if query.ndim >= 3 else int(query.shape[0]),
        "kv_len": int(prod(key.shape[:-2])) if key.ndim >= 3 else int(key.shape[0]),
        "head_dim": int(query.shape[-1]),
    }
    if query.ndim >= 3:
        features["heads"] = int(query.shape[-2])
    if key.ndim >= 3:
        features["kv_heads"] = int(key.shape[-2])
    phase = kwargs.get("phase")
    if isinstance(phase, str) and phase:
        features["phase"] = phase
    return features


_MLA_CORE_OPS = {
    "tensor_cast.multihead_latent_attention.default",
    "tensor_cast.multihead_latent_attention_quant.default",
    "tensor_cast.mla_sparse_attention.default",
    "tensor_cast.mla_sparse_attention_quant.default",
}
_MLA_PROJECTION_OPS = {
    "tensor_cast.mla_kv_projection.default",
    "tensor_cast.mla_kv_projection_quant.default",
    "tensor_cast.mla_q_absorb_projection.default",
    "tensor_cast.mla_q_absorb_projection_quant.default",
    "tensor_cast.mla_v_up_projection.default",
    "tensor_cast.mla_v_up_projection_quant.default",
}


def _mla_runtime_features(op_name: str, op_invoke_info: OpInvokeInfo) -> dict[str, Any]:
    args = op_invoke_info.args
    q, projected_kv, absorbed_q, kv_cache = args[:4]
    is_sparse = "mla_sparse_attention" in op_name
    query_lens = args[7] if len(args) > 7 else None
    seq_lens = args[6] if len(args) > 6 else None
    topk = args[10] if len(args) > 10 and isinstance(args[10], int) else None
    topk_indices = args[11] if len(args) > 11 else None
    if topk is None and isinstance(topk_indices, torch.Tensor) and topk_indices.ndim:
        topk = int(topk_indices.shape[-1])
    q_tokens = int(q.shape[0]) if isinstance(q, torch.Tensor) and q.ndim else 0
    if isinstance(absorbed_q, torch.Tensor) and absorbed_q.shape[0] > 0:
        heads = int(absorbed_q.shape[1])
        kv_heads = 1
    elif isinstance(projected_kv, torch.Tensor) and projected_kv.shape[0] > 0:
        heads = int(projected_kv.shape[1])
        # Dense MLA prefill is decompressed to one K/V head per query head.
        # Sparse MLA keeps the latent K/V representation for SFA, even when
        # the surrounding TC tensor carries the projected head axis.
        kv_heads = 1 if is_sparse else heads
    else:
        heads = int(q.shape[1]) if isinstance(q, torch.Tensor) and q.ndim >= 2 else 0
        kv_heads = 1
    kv_lora_rank = int(args[9]) if len(args) > 9 and isinstance(args[9], int) else 0
    if is_sparse:
        head_dim = kv_lora_rank or (int(q.shape[-1]) if isinstance(q, torch.Tensor) else 0)
    elif isinstance(projected_kv, torch.Tensor) and projected_kv.shape[0] > 0 and isinstance(q, torch.Tensor):
        rope_dim = int(kv_cache.shape[-1]) - kv_lora_rank if isinstance(kv_cache, torch.Tensor) else 0
        head_dim = int(q.shape[-1]) - rope_dim
    else:
        head_dim = kv_lora_rank or (int(q.shape[-1]) if isinstance(q, torch.Tensor) else 0)
    phase = op_invoke_info.kwargs.get("phase")
    decode_values = op_invoke_info.kwargs.get("is_decode_values")
    if phase not in {"prefill", "decode"} and isinstance(decode_values, (list, tuple)) and decode_values:
        if all(value is True for value in decode_values):
            phase = "decode"
        elif all(value is False for value in decode_values):
            phase = "prefill"
    if phase not in {"prefill", "decode"} and isinstance(query_lens, torch.Tensor) and query_lens.numel():
        phase = "decode" if int(query_lens.max().item()) <= 1 else "prefill"
    features: dict[str, Any] = {
        "q_tokens": q_tokens,
        "heads": heads,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "phase": phase,
        "is_sparse": is_sparse,
    }
    if isinstance(seq_lens, torch.Tensor) and seq_lens.numel():
        values = [int(value) for value in seq_lens.detach().cpu().flatten().tolist()]
        features["kv_lengths"] = tuple(values)
        features["effective_kv_len"] = sum(values) / len(values) if values else 0
        block_size = int(kv_cache.shape[-2]) if isinstance(kv_cache, torch.Tensor) and kv_cache.ndim >= 2 else 0
        if block_size > 0:
            features["block_size"] = block_size
            features["valid_blocks"] = tuple((value + block_size - 1) // block_size for value in values)
    if topk is not None:
        features["topk"] = topk
    return {key: value for key, value in features.items() if value is not None}


def _mla_projection_features(op_name: str, op_invoke_info: OpInvokeInfo) -> dict[str, Any]:
    args = op_invoke_info.args
    if len(args) < 2 or not isinstance(args[0], torch.Tensor) or not isinstance(args[1], torch.Tensor):
        return {}
    activation, weight = args[:2]
    m = int(prod(activation.shape[:-1]))
    k = int(activation.shape[-1])
    if "mla_v_up_projection" in op_name:
        n = int(weight.shape[-1])
        mm_kind = "transpose_batch"
    elif activation.ndim >= 3 or "mla_q_absorb_projection" in op_name:
        n = int(weight.shape[-1])
        mm_kind = "batch"
    else:
        n = int(weight.shape[-1])
        mm_kind = "dense"
    if "quant" in op_name:
        quantization = "w8a8"
    else:
        quantization = "none"
    return {
        "m": m,
        "k": k,
        "n": n,
        "mm_kind": "weight_quant" if quantization != "none" else mm_kind,
        "quantization": quantization,
        "compute_dtype": _dtype_name(activation),
        "projection_role": op_name.split("tensor_cast.", 1)[-1].split("_projection", 1)[0],
    }


def _communication_features(
    op_name: str,
    op_invoke_info: OpInvokeInfo,
    tensors: list[torch.Tensor],
    raw_result: PerformanceModel.Result,
) -> dict[str, Any]:
    statistics = raw_result.statistics if isinstance(raw_result.statistics, dict) else {}
    features: dict[str, Any] = {"collective": op_name.rsplit(".", 1)[0].removeprefix("tensor_cast.")}
    group_size = statistics.get("group_size")
    rank_group = op_invoke_info.args[-1] if op_invoke_info.args else None
    if not isinstance(group_size, (int, float)) and isinstance(rank_group, (list, tuple)):
        group_size = len(rank_group)
    if isinstance(group_size, (int, float)):
        features["group_size"] = int(group_size)
    raw_message_bytes = statistics.get("message_size_bytes")
    if not isinstance(raw_message_bytes, (int, float)) and tensors:
        raw_message_bytes = int(tensors[0].numel() * tensors[0].element_size())
    if isinstance(raw_message_bytes, (int, float)):
        raw_message_bytes = int(raw_message_bytes)
        features["analytic_message_bytes"] = raw_message_bytes
        message_bytes = raw_message_bytes
        if op_name == "tensor_cast.reduce_scatter.default" and features.get("group_size", 0) > 1:
            message_bytes //= int(features["group_size"])
        # HCCL all-to-allv rows use the full per-rank input list.  The raw
        # analytic statistic deliberately excludes the local self-transfer.
        if op_name == "tensor_cast.all_to_all.default" and tensors:
            message_bytes = int(tensors[0].numel() * tensors[0].element_size())
        features["message_bytes"] = message_bytes
    topology_tier = statistics.get("topology_tier")
    if isinstance(topology_tier, int):
        features["topology_tier"] = topology_tier
    return features


@dataclass(frozen=True)
class CalibrationSignature:
    """Normalized key used by analytic calibration data sources."""

    tc_op: str
    op_family: str
    device: str
    dtype: Optional[str]
    features: Mapping[str, Any]


def build_calibration_signature(
    op_invoke_info: OpInvokeInfo,
    raw_result: PerformanceModel.Result,
    *,
    device_name: str,
) -> CalibrationSignature:
    """Build a semantic signature without consulting profiling OP mappings."""
    op_name = _op_name(op_invoke_info)
    tensors = _tensor_args(op_invoke_info.args)
    first_tensor = tensors[0] if tensors else None

    if op_name in {"aten.mm.default", "aten.bmm.default", "aten.matmul.default"}:
        family = "matmul"
        features = _matmul_features(tensors)
        features["mm_kind"] = "batch" if op_name == "aten.bmm.default" else "dense"
        features["quantization"] = "none"
        features["compute_dtype"] = _dtype_name(first_tensor)
    elif op_name in _GMM_VARIANTS:
        family = "matmul"
        features = _grouped_features(op_name, op_invoke_info)
        features["compute_dtype"] = features.get("compute_dtype") or _dtype_name(first_tensor)
    elif op_name in _LINEAR_QUANTIZATION:
        family = "matmul"
        features = _linear_features(op_name, op_invoke_info)
        features["mm_kind"] = "weight_quant"
        features["quantization"] = _LINEAR_QUANTIZATION[op_name]
        features["compute_dtype"] = _dtype_name(first_tensor)
    elif op_name in _MLA_PROJECTION_OPS:
        family = "matmul"
        features = _mla_projection_features(op_name, op_invoke_info)
    elif op_name in _MLA_CORE_OPS:
        family = "attention"
        features = _mla_runtime_features(op_name, op_invoke_info)
    elif op_name.startswith("tensor_cast.attention"):
        family = "attention"
        features = _attention_features(tensors, op_invoke_info.kwargs)
    elif op_name in {
        "tensor_cast.all_reduce.default",
        "tensor_cast.all_gather.default",
        "tensor_cast.reduce_scatter.default",
        "tensor_cast.all_to_all.default",
    }:
        family = "communication"
        features = _communication_features(op_name, op_invoke_info, tensors, raw_result)
    else:
        family = "other"
        features = {}

    return CalibrationSignature(
        tc_op=op_name,
        op_family=family,
        device=device_name,
        dtype=_dtype_name(first_tensor),
        features=features,
    )
