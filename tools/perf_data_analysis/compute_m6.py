"""Compute M6: empirical E2E prediction ratio."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

DEFAULT_SOURCE_FILTER = {"MEASURED", "INTERPOLATED"}


def _sum_kernels_with_dedup(events: list) -> tuple:
    """Sum kernel durations with hcom deduplication.

    kernel_details.csv records each hcom on both Stream N/A and a hardware
    stream with identical (start_time, duration). Deduplicate by
    (int(start), kernel_type) keeping the max duration.
    AicpuKernel entries are tracked separately and excluded from
    compute_us and hcom_us.

    Args:
        events: list of (start, end, kernel_type, input_shapes)

    Returns:
        (compute_us, hcom_us, aicpu_us, kernel_count, kernel_type_durations)
    """
    compute_us = 0.0
    hcom_us = 0.0
    aicpu_us = 0.0
    kernel_count = 0
    kernel_type_durations: dict[str, float] = defaultdict(float)
    hcom_seen: dict[tuple, float] = {}

    for start, end, ktype, _ in events:
        dur = end - start
        if ktype.endswith("AicpuKernel"):
            kernel_count += 1
            aicpu_us += dur
        elif ktype.startswith("hcom_"):
            key = (int(start), ktype)
            if key in hcom_seen:
                if dur > hcom_seen[key]:
                    hcom_us += dur - hcom_seen[key]
                    kernel_type_durations[ktype] += dur - hcom_seen[key]
                    hcom_seen[key] = dur
            else:
                hcom_seen[key] = dur
                hcom_us += dur
                kernel_type_durations[ktype] += dur
                kernel_count += 1
        else:
            kernel_count += 1
            compute_us += dur
            kernel_type_durations[ktype] += dur

    return compute_us, hcom_us, aicpu_us, kernel_count, dict(kernel_type_durations)


def _load_tc_trace(
    tc_trace_path: Path,
    source_filter: set[str],
) -> float:
    """Load TC chrome trace, sum empirical HIT durations.

    Only counts events where args.source ∈ source_filter and dur > 0.

    Returns:
        empirical_hit_us
    """
    with tc_trace_path.open() as f:
        data = json.load(f)

    total = 0.0
    for event in data.get("traceEvents", []):
        if event.get("ph") != "X":
            continue
        args = event.get("args", {})
        source = args.get("source", "")
        if source not in source_filter:
            continue
        dur = event.get("dur", 0)
        if dur <= 0:
            continue
        if args.get("kernel_type", ""):
            total += dur

    return total


def _load_prof_trace(prof_trace_path: Path) -> tuple[float, float, float]:
    """Load prof trace CSV, sum durations with hcom dedup.

    Returns:
        (real_per_fwd_us, compute_us, hcom_us)
    """
    events = []
    with prof_trace_path.open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            ktype = (row.get("Type") or "").strip()
            if not ktype:
                continue
            try:
                start = float((row.get("Start Time(us)") or "0").strip())
                dur = float((row.get("Duration(us)") or "0").strip())
            except ValueError:
                continue
            events.append((start, start + dur, ktype, ""))

    compute_us, hcom_us, _, _, _ = _sum_kernels_with_dedup(events)
    return compute_us + hcom_us, compute_us, hcom_us


def compute_m6(
    tc_trace: str,
    prof_trace: str,
    source_filter: set[str] | None = None,
) -> dict:
    """Compute M6 by comparing TC chrome trace vs prof forward pass trace.

    Args:
        tc_trace: Path to TC chrome trace JSON (from --chrome-trace).
        prof_trace: Path to clean forward pass CSV (pre-extracted).
        source_filter: Set of QuerySource names to include from TC trace.
            Default: {"MEASURED", "INTERPOLATED"}.

    Returns:
        dict with m6_ratio, empirical_hit_us, real_per_fwd_us,
        and component breakdowns.
    """
    if source_filter is None:
        source_filter = DEFAULT_SOURCE_FILTER

    tc_trace_path = Path(tc_trace)
    prof_trace_path = Path(prof_trace)

    if not tc_trace_path.exists():
        raise FileNotFoundError(f"TC trace not found: {tc_trace_path}")
    if not prof_trace_path.exists():
        raise FileNotFoundError(f"Prof trace not found: {prof_trace_path}")

    empirical_hit_us = _load_tc_trace(tc_trace_path, source_filter)
    real_per_fwd_us, prof_compute, prof_hcom = _load_prof_trace(prof_trace_path)

    m6_ratio = empirical_hit_us / real_per_fwd_us if real_per_fwd_us > 0 else 0.0

    return {
        "m6_ratio": m6_ratio,
        "empirical_hit_us": round(empirical_hit_us, 2),
        "real_per_fwd_us": round(real_per_fwd_us, 2),
        "selected_fwd_compute_us": round(prof_compute, 2),
        "selected_fwd_hcom_us": round(prof_hcom, 2),
        "tc_trace": tc_trace,
        "prof_trace": prof_trace,
        "source_filter": sorted(source_filter),
    }


def _format_report(result: dict) -> str:
    lines = [
        "=" * 60,
        "M6: Empirical E2E Prediction Ratio",
        "=" * 60,
        "",
        f"TC trace:        {result['tc_trace']}",
        f"Prof trace:      {result['prof_trace']}",
        f"Source filter:   {result['source_filter']}",
        "",
        f"Empirical HIT total: {result['empirical_hit_us']:>12,.1f} us "
        f"({result['empirical_hit_us'] / 1e3:,.1f} ms)",
        f"Real per-fwd:        {result['real_per_fwd_us']:>12,.1f} us "
        f"({result['real_per_fwd_us'] / 1e3:,.1f} ms)",
        f"  Compute:           {result['selected_fwd_compute_us']:>12,.1f} us",
        f"  hcom:              {result['selected_fwd_hcom_us']:>12,.1f} us",
        "",
        f"M6 = {result['m6_ratio']:.3f}  (Empirical / Real)",
    ]

    return "\n".join(lines)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute M6: compare TC chrome trace vs profiling forward pass."
    )
    parser.add_argument(
        "--tc-trace",
        required=True,
        help="Path to TC chrome trace JSON (from --chrome-trace)",
    )
    parser.add_argument(
        "--prof-trace",
        required=True,
        help="Path to clean forward pass CSV (pre-extracted from kernel_details)",
    )
    parser.add_argument(
        "--source-filter",
        default=None,
        help="Comma-separated QuerySource names to include "
        "(default: MEASURED,INTERPOLATED). Use MEASURED to exclude interpolated.",
    )
    parser.add_argument("--json-output", default=None)
    return parser


def main() -> None:
    args = build_argparser().parse_args()

    source_filter = DEFAULT_SOURCE_FILTER
    if args.source_filter:
        source_filter = {s.strip() for s in args.source_filter.split(",")}

    result = compute_m6(
        tc_trace=args.tc_trace,
        prof_trace=args.prof_trace,
        source_filter=source_filter,
    )
    print(_format_report(result))
    if args.json_output:
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nJSON output written to: {output_path}")


if __name__ == "__main__":
    main()
