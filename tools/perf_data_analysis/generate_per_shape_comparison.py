"""Generate per-(kernel_type, shape) TC vs Profiling comparison CSV.

Compares TC predicted duration (from chrome trace) against actual NPU
kernel duration (from forward-pass trace CSV), grouped by
(kernel_type, normalized_shape_key).

Inputs:
  - TC chrome trace JSON (--chrome-trace output with simulation_shapes field)
  - Prof forward-pass trace CSV (pre-extracted from kernel_details.csv)

Output:
  CSV with columns: kernel_type, shape_key, tc_dur_us, prof_dur_us,
  delta_pct, tc_count, prof_count, source

Usage:
    python3.10 tools/perf_data_analysis/generate_per_shape_comparison.py \\
        --tc-trace results/dsv3_dc_trace.json \\
        --prof-trace docs/perf_database/forward_pass_traces/dsv3_dc_1tok.csv \\
        --output results/dsv3_dc_per_shape.csv
"""

import argparse
import ast
import contextlib
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

OUTPUT_COLUMNS = [
    "kernel_type",
    "shape_key",
    "tc_dur_us",
    "prof_dur_us",
    "delta_pct",
    "tc_count",
    "prof_count",
    "source",
]


def _normalize_shape_key(shapes_str: str) -> str:
    """Normalize a shape string to a canonical key for matching.

    TC shapes: "100,200;200,300" or [[100,200],[200,300]]
    Prof shapes: "100,200;200,300" (semicolon-separated dims)

    Returns semicolon-joined dims like "100,200;200,300".
    """
    s = shapes_str.strip().strip('"').replace(" ", "")
    if not s:
        return ""
    # Already in "dim1,dim2;dim3,dim4" format
    if ";" in s or (s and "[" not in s):
        return s
    # Parse [[100,200],[200,300]] format
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list):
            return ";".join(",".join(str(d) for d in shape) for shape in parsed)
    except (ValueError, SyntaxError):
        pass
    return s


def _parse_tc_trace(
    trace_path: str,
) -> dict[tuple[str, str], dict]:
    """Parse TC trace, return {(kernel_type, shape_key): {dur, count, source}}.

    Composite ops with sub_kernel_durations are expanded into individual
    sub-kernel entries. Each sub-kernel inherits the parent's simulation_shapes.
    """
    with open(trace_path) as f:
        data = json.load(f)

    result: dict[tuple[str, str], dict] = defaultdict(
        lambda: {"dur_us": 0.0, "count": 0, "source": ""}
    )

    for event in data.get("traceEvents", []):
        if event.get("ph") != "X":
            continue

        args = event.get("args", {})
        kernel_type = args.get("kernel_type", "")
        if not kernel_type:
            continue

        dur = event.get("dur", 0)
        source = args.get("source", "")
        simulation_shapes = args.get("simulation_shapes", "")
        shape_key = _normalize_shape_key(simulation_shapes)

        # Check for composite with sub_kernel_durations
        skd_raw = args.get("sub_kernel_durations")
        skd = None
        if skd_raw:
            try:
                skd = ast.literal_eval(skd_raw)
            except (ValueError, SyntaxError):
                skd = None
            if not (
                isinstance(skd, list)
                and all(isinstance(x, (list, tuple)) and len(x) == 2 for x in skd)
            ):
                skd = None

        if skd:
            # Expand composite into individual sub-kernel rows
            for sk_name, sk_dur in skd:
                key = (sk_name, shape_key)
                result[key]["dur_us"] += sk_dur
                result[key]["count"] += 1
                result[key]["source"] = source
        else:
            # Non-composite or legacy trace without sub_kernel_durations
            sub_kts = [s.strip() for s in kernel_type.split(",") if s.strip()]
            if len(sub_kts) > 1:
                per_sub = dur / len(sub_kts)
                for sk in sub_kts:
                    key = (sk, shape_key)
                    result[key]["dur_us"] += per_sub
                    result[key]["count"] += 1
                    result[key]["source"] = source
            else:
                key = (kernel_type, shape_key)
                result[key]["dur_us"] += dur
                result[key]["count"] += 1
                result[key]["source"] = source

    return dict(result)


def _parse_prof_trace(
    prof_path: str,
) -> dict[tuple[str, str], dict]:
    """Parse prof CSV, return {(kernel_type, shape_key): {dur, count}}.

    hcom dedup: group by (int(start_time), Type), take max duration.
    AicpuKernel excluded.
    """
    raw_events: list[tuple[str, str, float, float]] = []  # (type, shapes, dur, start)

    with open(prof_path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            ktype = (row.get("Type") or "").strip()
            if not ktype or ktype.endswith("AicpuKernel"):
                continue
            dur = float((row.get("Duration(us)") or "0").strip())
            shapes = (row.get("Input Shapes") or "").strip().strip('"')
            start = float((row.get("Start Time(us)") or "0").strip())
            raw_events.append((ktype, shapes, dur, start))

    # hcom dedup
    hcom_groups: dict[tuple[int, str, str], float] = {}
    result: dict[tuple[str, str], dict] = defaultdict(
        lambda: {"dur_us": 0.0, "count": 0}
    )

    for ktype, shapes, dur, start in raw_events:
        shape_key = _normalize_shape_key(shapes)
        if ktype.startswith("hcom_"):
            key = (int(start), ktype, shape_key)
            hcom_groups[key] = max(hcom_groups.get(key, 0.0), dur)
        else:
            rkey = (ktype, shape_key)
            result[rkey]["dur_us"] += dur
            result[rkey]["count"] += 1

    for (_, ktype, shape_key), dur in hcom_groups.items():
        rkey = (ktype, shape_key)
        result[rkey]["dur_us"] += dur
        result[rkey]["count"] += 1

    return dict(result)


def generate_per_shape_comparison(
    tc_trace: str,
    prof_trace: str,
    output_path: str | None = None,
) -> list[dict]:
    """Generate per-(kernel_type, shape) comparison.

    Args:
        tc_trace: Path to TC chrome trace JSON.
        prof_trace: Path to prof forward-pass trace CSV.
        output_path: Output CSV path. If None, prints to stdout.

    Returns:
        List of comparison rows (dicts).
    """
    tc_data = _parse_tc_trace(tc_trace)
    prof_data = _parse_prof_trace(prof_trace)

    all_keys = set(tc_data.keys()) | set(prof_data.keys())
    rows = []
    for key in all_keys:
        kt, shape_key = key
        tc = tc_data.get(key, {"dur_us": 0, "count": 0, "source": ""})
        prof = prof_data.get(key, {"dur_us": 0, "count": 0})

        tc_dur = tc["dur_us"]
        prof_dur = prof["dur_us"]
        delta_pct = (
            (tc_dur - prof_dur) / prof_dur * 100
            if prof_dur > 0
            else (float("inf") if tc_dur > 0 else 0)
        )

        rows.append(
            {
                "kernel_type": kt,
                "shape_key": shape_key,
                "tc_dur_us": round(tc_dur, 2),
                "prof_dur_us": round(prof_dur, 2),
                "delta_pct": round(delta_pct, 1),
                "tc_count": tc["count"],
                "prof_count": prof["count"],
                "source": tc.get("source", ""),
            }
        )

    rows.sort(key=lambda r: -max(r["tc_dur_us"], r["prof_dur_us"]))

    # Write output
    cm = (
        open(output_path, "w", newline="")
        if output_path
        else contextlib.nullcontext(sys.stdout)
    )
    with cm as out_file:
        writer = csv.DictWriter(out_file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate per-(kernel_type, shape) TC vs Profiling comparison."
    )
    parser.add_argument(
        "--tc-trace", required=True, help="Path to TC chrome trace JSON"
    )
    parser.add_argument(
        "--prof-trace", required=True, help="Path to prof forward-pass trace CSV"
    )
    parser.add_argument(
        "--output", default=None, help="Output CSV path (default: stdout)"
    )
    args = parser.parse_args()

    rows = generate_per_shape_comparison(args.tc_trace, args.prof_trace, args.output)
    if not args.output:
        return
    print(f"Generated {len(rows)} per-shape comparison rows → {args.output}")


if __name__ == "__main__":
    main()
