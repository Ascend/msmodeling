import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Union


def build_st_case_from_verification(report: Dict[str, Any], operator_top_n: int = 10) -> Dict[str, Any]:
    """Build a benchmark-regression-compatible ST case from a verify report.

    The emitted JSON only contains keys the performance-regression loader
    understands (``TextPerfRegressionCase`` fields), so a passed case can be
    committed directly under ``tests/benchmark/models/cases/``. Verification
    context is recorded in the description; failed verifications never emit a
    case.
    """
    case_name = str(report.get("case_name") or "adapter-case")
    simulation = report.get("simulation", {})
    total_time = float(simulation.get("total_forward_time_s") or 0.0)
    check_summary = ", ".join(
        f"{check.get('category')}={check.get('actual_count')}" for check in report.get("key_op_checks", [])
    )
    description = (
        f"Adapter run-through guardrail for {case_name} "
        f"(model-adapter verify passed; key op counts: {check_summary or 'n/a'})."
    )
    return {
        "type": "text",
        "name": case_name,
        "description": description,
        "initial_time_s": total_time,
        "baseline_time_s": total_time,
        "initial_tolerance": 0.1,
        "baseline_tolerance": 0.8,
        "operator_top_n": operator_top_n,
        "operator_tolerance": 0.1,
        "user_input": dict(report.get("case_input", {})),
        "operators": _top_operator_entries(report.get("actual_summary", {}).get("ops", {}), operator_top_n),
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


def build_st_cases_from_verification(report: Dict[str, Any], operator_top_n: int = 10) -> List[Dict[str, Any]]:
    if not report.get("passed"):
        return []
    return [build_st_case_from_verification(report, operator_top_n=operator_top_n)]


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
        json.dump(data, handle, indent=2, sort_keys=False, ensure_ascii=False)
        handle.write("\n")
