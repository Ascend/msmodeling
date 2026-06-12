"""Convert TC chrome trace JSON to human-readable CSV.

Extracts per-op events from the chrome trace and writes a flat CSV
with one row per op invocation. Useful for manual inspection, Excel
analysis, or as input to other tools.

Usage:
    python3.10 tools/perf_data_collection/parsers/trace_to_csv.py \\
        --trace results/dsv3_dc_trace.json \\
        --output results/dsv3_dc_ops.csv

    # Print to stdout (pipe to less, head, etc.)
    python3.10 tools/perf_data_collection/parsers/trace_to_csv.py \\
        --trace results/dsv3_dc_trace.json
"""

import argparse
import contextlib
import csv
import json
import sys
from pathlib import Path

COLUMNS = [
    "op_name",
    "kernel_type",
    "simulation_shapes",
    "dur_us",
    "source",
    "confidence",
    "composite",
    "sub_kernel_durations",
    "kernel_shapes",
    "shape_match_rule",
    "sub_kernel_shapes",
]


def trace_to_csv(trace_path: str, output_path: str | None = None) -> None:
    """Convert chrome trace JSON to CSV.

    Args:
        trace_path: Path to TC chrome trace JSON.
        output_path: Path to output CSV. If None, prints to stdout.
    """
    with open(trace_path) as f:
        data = json.load(f)

    cm = (
        open(output_path, "w", newline="")
        if output_path
        else contextlib.nullcontext(sys.stdout)
    )
    with cm as out_file:
        writer = csv.DictWriter(out_file, fieldnames=COLUMNS)
        writer.writeheader()

        for event in data.get("traceEvents", []):
            if event.get("ph") != "X":
                continue

            args = event.get("args", {})
            writer.writerow(
                {
                    "op_name": event.get("name", ""),
                    "kernel_type": args.get("kernel_type", ""),
                    "simulation_shapes": args.get("simulation_shapes", ""),
                    "dur_us": event.get("dur", 0),
                    "source": args.get("source", ""),
                    "confidence": args.get("confidence", ""),
                    "composite": args.get("composite", ""),
                    "sub_kernel_durations": args.get("sub_kernel_durations", ""),
                    "kernel_shapes": args.get("kernel_shapes", ""),
                    "shape_match_rule": args.get("shape_match_rule", ""),
                    "sub_kernel_shapes": args.get("sub_kernel_shapes", ""),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert TC chrome trace JSON to human-readable CSV."
    )
    parser.add_argument(
        "--trace", required=True, help="Path to TC chrome trace JSON"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path (default: stdout)",
    )
    args = parser.parse_args()
    trace_to_csv(args.trace, args.output)


if __name__ == "__main__":
    main()
