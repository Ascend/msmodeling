import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Union


def build_st_case_from_dicts(
    evidence_case: Dict[str, Any],
    actual_summary: Dict[str, Any],
    model: Dict[str, Any],
    status: str = "draft",
    verification_issues: List[Dict[str, Any]] = None,
    operator_top_n: int = 10,
) -> Dict[str, Any]:
    case_name = str(evidence_case.get("name") or actual_summary.get("case_name") or "adapter-case")
    total_time = float(actual_summary.get("total_forward_time_s") or 0.0)
    user_input = dict(evidence_case.get("input", {}))
    if "model_id" not in user_input and model.get("model_id"):
        user_input["model_id"] = model["model_id"]
    return {
        "type": "text",
        "name": case_name,
        "status": status,
        "description": f"Generated adapter guardrail for {case_name}",
        "initial_time_s": total_time,
        "baseline_time_s": total_time,
        "initial_tolerance": 0.1,
        "baseline_tolerance": 0.8,
        "operator_top_n": operator_top_n,
        "operator_tolerance": 0.1,
        "user_input": user_input,
        "operators": _top_operator_entries(actual_summary.get("ops", {}), operator_top_n),
        "verification_issues": [] if verification_issues is None else verification_issues,
    }


def _top_operator_entries(ops: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
    entries = []
    for name, data in ops.items():
        entries.append(
            {
                "name": name,
                "total_time_s": float(data.get("total_time_s") or 0.0),
                "num_calls": int(data.get("count") or 0),
            }
        )
    return sorted(entries, key=lambda item: item["total_time_s"], reverse=True)[:limit]


def build_st_cases_from_report(report: Dict[str, Any], operator_top_n: int = 10) -> List[Dict[str, Any]]:
    actuals = {item.get("case_name"): item for item in report.get("actual_summaries", [])}
    verification_by_case = {item.get("case_name"): item for item in report.get("verification_reports", [])}
    model = dict(report.get("evidence_model", {}))
    cases = []
    for evidence_case in report.get("evidence_cases", []):
        actual = actuals.get(evidence_case.get("name"))
        if actual is None:
            continue
        verification = verification_by_case.get(evidence_case.get("name"), {})
        status = "verified" if report.get("passed") and verification.get("passed", True) else "draft"
        cases.append(
            build_st_case_from_dicts(
                evidence_case,
                actual,
                model,
                status=status,
                verification_issues=list(verification.get("issues", [])),
                operator_top_n=operator_top_n,
            )
        )
    return cases


def write_st_cases(
    cases: Iterable[Dict[str, Any]],
    output_path: Union[str, Path],
) -> List[Path]:
    path = Path(output_path)
    cases = list(cases)
    written = []
    if path.suffix == ".json" and len(cases) == 1:
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(path, cases[0])
        return [path]
    path.mkdir(parents=True, exist_ok=True)
    for case in cases:
        case_name = str(case.get("name", "adapter-case")).replace("/", "-")
        target = path / f"{case_name}.json"
        _write_json(target, case)
        written.append(target)
    return written


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=False)
        handle.write("\n")
