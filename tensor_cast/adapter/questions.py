from typing import Any, Dict, Iterable, List, Optional


def build_human_questions(
    evidence_draft: Optional[Dict[str, Any]] = None,
    hint_conflicts: Optional[Iterable[Dict[str, Any]]] = None,
    limit: int = 8,
) -> List[Dict[str, Any]]:
    questions: List[Dict[str, Any]] = []

    for conflict in hint_conflicts or []:
        questions.append(
            {
                "kind": "resolve_hint_conflict",
                "priority": "high" if conflict.get("severity") == "error" else "medium",
                "question": (
                    f"Hint conflict {conflict.get('category')}: {conflict.get('message')} "
                    "If you have just confirmed the value, update the hints file; otherwise remove or lower confidence."
                ),
                "evidence": conflict,
            }
        )

    for case in _evidence_cases(evidence_draft):
        observed_by_source = {
            kernel.get("normalized_name"): kernel
            for kernel in case.get("observed_kernels", [])
            if isinstance(kernel, dict)
        }
        for op in case.get("expected", {}).get("major_ops", []):
            confidence = str(op.get("confidence", "high")).lower()
            name = str(op.get("name", ""))
            if confidence not in {"low", "medium"} and not name.startswith("profiling."):
                continue
            source = str(op.get("source", ""))
            profiling_name = source.split(":", maxsplit=1)[1] if ":" in source else name
            observed = observed_by_source.get(profiling_name, {})
            questions.append(
                {
                    "kind": "confirm_op_mapping",
                    "priority": "medium" if confidence == "medium" else "low",
                    "question": (
                        f"Raw Insight op {profiling_name!r} appears {observed.get('occurrences', op.get('count'))} times. "
                        f"Current TensorCast mapping candidate is {name!r} with {confidence} confidence. "
                        "If you just confirmed it, add an op_mapping_hint with optional count or shape variants; "
                        "if not, leave it as low-confidence evidence."
                    ),
                    "evidence": {
                        "case": case.get("name"),
                        "expected_op": op,
                        "observed_kernel": observed,
                    },
                }
            )

    return questions[:limit]


def _evidence_cases(evidence_draft: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not evidence_draft:
        return []
    cases = evidence_draft.get("cases", [])
    return [case for case in cases if isinstance(case, dict)]
