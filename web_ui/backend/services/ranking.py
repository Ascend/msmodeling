"""Capture-time ranking (``result_records.rank``).

Pure domain. For the optimizer, ``rank=1`` is the overall best across devices:
max ``throughput_token_s``; ties -> lower ``ttft_ms`` -> lower ``tpot_ms`` ->
device name ascending. ``result.best_config`` is then a pure lookup of the
rank=1 record — no read-time enrichment.
"""

from __future__ import annotations

import math
from typing import Any, Mapping


def _num(summary: Mapping[str, Any], key: str) -> float:
    """Coerce a summary field to float; missing/NaN/unparseable -> -inf for ranking."""
    value = summary.get(key)
    try:
        f = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("-inf")
    return f if not math.isnan(f) else float("-inf")


def _tt(summary: Mapping[str, Any], key: str) -> float:
    """Latency tiebreaker: missing/NaN -> +inf (worse)."""
    value = summary.get(key)
    try:
        f = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("inf")
    return f if not math.isnan(f) else float("inf")


def optimizer_rank_key(summary: Mapping[str, Any]) -> tuple:
    """Sort key for a single optimizer summary: higher throughput first, then
    lower TTFT, lower TPOT, then device name ascending.

    Returned as a tuple suitable for ``sorted(key=...)`` after negating
    throughput. Callers rank 1..N in sorted order.
    """
    config = summary.get("config") if isinstance(summary.get("config"), Mapping) else {}
    device = str(config.get("device") or summary.get("device") or "")
    # PD-ratio records carry balanced_qps (not throughput_token_s); rank by the
    # mode-appropriate primary metric so rank=1 is the real best, not arbitrary.
    primary = "balanced_qps" if summary.get("mode") == "pd_ratio" else "throughput_token_s"
    return (
        -_num(summary, primary),
        _tt(summary, "ttft_ms"),
        _tt(summary, "tpot_ms"),
        device,
    )


def assign_optimizer_ranks(records: list[Any]) -> list[int]:
    """Return the 1-based rank of each record in input order.

    ``records`` items expose a ``summary`` mapping (or are mappings themselves).
    Rank 1 = best. Ties keep stable input order. Non-optimizer (text/video)
    records are not ranked — callers pass only optimizer records here.
    """
    decorated = list(enumerate(records))
    decorated.sort(key=lambda pair: optimizer_rank_key(_summary_of(pair[1])))
    ranks_by_index = [0] * len(records)
    for position, (original_index, _record) in enumerate(decorated, start=1):
        ranks_by_index[original_index] = position
    return ranks_by_index


def _summary_of(record: Any) -> Mapping[str, Any]:
    if isinstance(record, Mapping):
        return record.get("summary", record) if "summary" in record else record
    return getattr(record, "summary", {}) or {}
