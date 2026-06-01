"""Generate config combinations for parameterized tests."""

from collections.abc import Iterable


def build_case_matrix(**dimensions: Iterable[object]) -> list[dict[str, object]]:
    """Build cartesian product matrix from named dimensions."""
    cases: list[dict[str, object]] = [{}]
    for key, values in dimensions.items():
        value_list = list(values)
        next_cases: list[dict[str, object]] = []
        for case in cases:
            for value in value_list:
                item = dict(case)
                item[key] = value
                next_cases.append(item)
        cases = next_cases
    return cases


def build_latency_thresholds(*, ttft_ms: float, tpot_ms: float, tolerance_ms: float = 0.1) -> dict[str, float]:
    """Create threshold config shared by serving latency tests."""
    return {
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "tolerance_ms": tolerance_ms,
    }
