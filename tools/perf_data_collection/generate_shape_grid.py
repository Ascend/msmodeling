"""Generate replayable shape grids from real optimizer database queries."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence


CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parents[1]
OP_REPLAY_DIR = CURRENT_DIR / "op_replay"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from cli.logo import print_logo  # noqa: E402
from grid_generator.runner import discover_replay_supported_ops, run_query_mode  # noqa: E402


DEFAULT_ROWS = 1000


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


def build_argparser() -> argparse.ArgumentParser:
    supported = ", ".join(discover_replay_supported_ops(OP_REPLAY_DIR))
    parser = argparse.ArgumentParser(
        description=(
            "Run internal throughput-optimizer sweeps for the target HuggingFace model(s), "
            "capture profiling-database queries, and append replayable query or generic fallback rows. "
            "Pass --optimizer-args-file to generate shapes for your actual optimizer scenarios instead "
            "of (or in addition to) the internal model-structure sweeps."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Replay-supported operators: {supported}",
    )
    parser.add_argument(
        "--database-path",
        type=Path,
        required=True,
        help="Performance-database directory containing op_mapping.yaml and operator CSV files.",
    )
    parser.add_argument(
        "--rows",
        type=_positive_int,
        default=DEFAULT_ROWS,
        help=(
            "Maximum new valid and unique rows appended to each target operator CSV in this run. "
            "Existing, duplicate, and rejected rows do not consume the budget. "
            "Exact demands originating from --optimizer-args-file never consume this budget; "
            "target-model exact demands and coverage/fallback candidates do. "
            f"Default: {DEFAULT_ROWS}."
        ),
    )
    target_group = parser.add_argument_group("Workload Source (one of both may be given)")
    target_group.add_argument(
        "--target-models",
        nargs="+",
        default=None,
        help=(
            "One or more HuggingFace model IDs for the internal sampling sweeps. "
            "Comma-separated IDs are also accepted. Required unless --optimizer-args-file is given."
        ),
    )
    target_group.add_argument(
        "--optimizer-args-file",
        type=Path,
        default=None,
        help=(
            "YAML or JSON file describing one or more actual throughput-optimizer scenarios "
            "(field names mirror the optimizer CLI in snake_case). Unknown or unsupported fields "
            "fail closed. Exact query demand from these scenarios is always generated and never "
            "truncated by --rows."
        ),
    )
    parser.add_argument(
        "--ops",
        nargs="+",
        default=None,
        help=(
            "Optional replay-supported kernel types that define the final output set. "
            "Unqueried selections use generic theory fallback; default: model-queried operators."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for deterministic coverage-candidate ordering. Default: 0.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help=(
            "Optional output path for the machine-readable JSON generation report. "
            "By default the report is written into the automatic query cache directory."
        ),
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_argparser().parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    print_logo()
    data_dir = args.database_path.resolve()
    result = run_query_mode(
        args,
        data_dir=data_dir,
        op_replay_dir=OP_REPLAY_DIR,
        repo_root=REPO_ROOT,
    )
    totals = result.report.get("totals", {})
    print(
        "Query-driven shape generation complete: "
        f"workloads={result.workloads.succeeded}/{result.workloads.attempted}, "
        f"cached={result.workloads.cached}, elapsed={result.workloads.elapsed_seconds:.1f}s, "
        f"captured_demands={result.captured_demands}, "
        f"appended_rows={result.total_appended_rows}, "
        f"updated_csvs={len(result.generated_files)}"
        + (
            f", preflight_rejected={totals.get('preflight_rejected', 0)}"
            if totals.get("preflight_rejected")
            else ""
        )
        + "."
    )
    unsupported = result.report.get("unsupported_kernel_demands") or {}
    if unsupported:
        print(
            "Warning: captured demands target kernels without database/replay support and were not generated: "
            + ", ".join(f"{name} x{count}" for name, count in unsupported.items())
        )
    if result.workloads.failed_workloads:
        print(
            f"Warning: {len(result.workloads.failed_workloads)} workload(s) failed; "
            "successful workloads were still used."
        )
    if result.skipped_files:
        print(f"No new replay row was produced for {len(result.skipped_files)} supported CSV(s).")
    if result.report_path is not None:
        print(f"Generation report: {result.report_path}")


if __name__ == "__main__":
    main()
