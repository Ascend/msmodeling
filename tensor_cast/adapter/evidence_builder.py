from typing import Any, Dict, Iterable, List, Optional

from .context import AdaptationContext
from .hints import HintLedger, UserHint
from .insight import ObservedKernel, RawInsightSummary


_RAW_TO_TC_OP = {
    "FusedInferAttentionScore": ("tensor_cast.attention.default", "medium"),
    "MoeGatingTopK": ("tensor_cast.moe_gating_top_k_softmax.default", "medium"),
    "RmsNorm": ("tensor_cast.rms_norm.default", "low"),
}

_PROFILING_ONLY_KERNELS = {
    "DispatchFFNCombine",
    "QuantBatchMatmulV3",
    "DynamicQuant",
    "MatMulV2",
    "AddRmsNormBias",
}


def _hinted_mapping(hints: Iterable[UserHint], profiling_name: str) -> Optional[Dict[str, Any]]:
    for hint in hints:
        if hint.kind != "op_mapping_hint":
            continue
        if hint.data.get("profiling_op") != profiling_name:
            continue
        tc_op = hint.data.get("tc_op")
        if not tc_op:
            continue
        return {
            "name": tc_op,
            "confidence": hint.confidence,
            "source": f"user_hint:{profiling_name}",
        }
    return None


def _expected_op_from_kernel(kernel: ObservedKernel, hints: Iterable[UserHint]) -> Optional[Dict[str, Any]]:
    hinted = _hinted_mapping(hints, kernel.normalized_name)
    if hinted is not None:
        hinted["count"] = kernel.occurrences
        return hinted

    if kernel.normalized_name in _RAW_TO_TC_OP:
        op_name, confidence = _RAW_TO_TC_OP[kernel.normalized_name]
        return {
            "name": op_name,
            "count": kernel.occurrences,
            "confidence": confidence,
            "source": f"raw_insight:{kernel.normalized_name}",
        }
    if kernel.normalized_name in _PROFILING_ONLY_KERNELS:
        return {
            "name": f"profiling.{kernel.normalized_name}",
            "count": kernel.occurrences,
            "confidence": "low",
            "source": f"raw_insight:{kernel.normalized_name}",
        }
    return None


def build_evidence_draft(
    context: AdaptationContext,
    raw_insight: RawInsightSummary,
    hints: Optional[HintLedger] = None,
    case_name: Optional[str] = None,
    top_n: int = 20,
) -> Dict[str, Any]:
    hint_items = [] if hints is None else hints.hints
    major_ops: List[Dict[str, Any]] = []
    seen_ops = set()
    for kernel in raw_insight.top_kernels(top_n):
        expected = _expected_op_from_kernel(kernel, hint_items)
        if expected is None:
            continue
        key = (expected["name"], expected.get("source"))
        if key in seen_ops:
            continue
        seen_ops.add(key)
        major_ops.append(expected)

    generated_case_name = case_name or _default_case_name(context)
    return {
        "version": 1,
        "model": {
            "model_id": context.model_id,
            "raw_command": context.raw_command,
        },
        "cases": [
            {
                "name": generated_case_name,
                "input": _evidence_input_from_context(context),
                "observed_kernels": [kernel.to_dict() for kernel in raw_insight.top_kernels(top_n)],
                "expected": {
                    "total_forward": {
                        "time_s": raw_insight.total_wall_duration_ms / 1000.0,
                        "rel_tolerance": 0.2,
                        "source": "raw_insight:Totals.wall_duration_ms",
                    },
                    "major_ops": major_ops,
                },
                "notes": [
                    "Generated from raw Insight profiling and optional user hints.",
                    "raw Insight Totals wall duration is used as expected total_forward time.",
                    "Low-confidence profiling.* entries are placeholders for fused or profiling-only kernels.",
                ],
            }
        ],
    }


def _default_case_name(context: AdaptationContext) -> str:
    model_name = context.model_id.rstrip("/").split("/")[-1].lower().replace("_", "-")
    phase = "decode" if context.normalized_args.get("decode") else "prefill"
    quant = context.normalized_args.get("quantize_linear_action")
    suffix = f"-{str(quant).lower()}" if quant else ""
    return f"{model_name}-{phase}{suffix}"


def _evidence_input_from_context(context: AdaptationContext) -> Dict[str, Any]:
    data = dict(context.normalized_args)
    aliases = {
        "compile": "do_compile",
        "compile_allow_graph_break": "allow_graph_break",
        "num_devices": "world_size",
        "query_length": "query_len",
    }
    for source, target in aliases.items():
        if source in data and target not in data:
            data[target] = data[source]
    data.setdefault("model_id", context.model_id)
    return data
