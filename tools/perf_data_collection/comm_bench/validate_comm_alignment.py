"""C9 comm data alignment: validate HCCL microbenchmark CSVs against CommAnalyticModel.

Loads per-op CSVs from an hccl/ directory and compares each measured Duration(us)
against the CommAnalyticModel analytic prediction for the same
(message_bytes, num_devices, topology_tier) combination.

Alignment criteria (configurable via --tolerance):
  - PASS  : 1/tolerance ≤ ratio ≤ tolerance   (default tolerance=2.0)
  - WARN  : ratio outside tolerance but within tolerance*2
  - FAIL  : ratio outside tolerance*2

CSV filename → op_type mapping (matches op_mapping.yaml kernel_type):
  hcom_allReduce_.csv      → all_reduce
  hcom_allGather_.csv      → all_gather
  hcom_reduceScatter_.csv  → reduce_scatter
  hcom_alltoallv_.csv      → all_to_all

Usage:
  python validate_comm_alignment.py --csv-dir ./hccl/v8.5/
  python validate_comm_alignment.py --csv-dir ./hccl/v8.5/ --tolerance 1.5 --verbose
"""

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cli.logo import print_logo

# ---------------------------------------------------------------------------
# A3 topology constants (mirrors device.py ATLAS_800_A3_752T_128G_DIE)
# ---------------------------------------------------------------------------

@dataclass
class _Topology:
    bandwidth_bytes_ps: float   # unidirectional, bytes/s
    latency_s: float
    comm_efficiency: float = 0.7


# topology_tier → InterconnectTopology for ATLAS_800_A3
_A3_TOPOLOGIES: Dict[int, _Topology] = {
    0: _Topology(bandwidth_bytes_ps=196e9, latency_s=5.5e-6),   # inter_pod (2-level CLOS)
    1: _Topology(bandwidth_bytes_ps=196e9, latency_s=0.5e-6),   # intra_pod (1-level CLOS)
    2: _Topology(bandwidth_bytes_ps=224e9, latency_s=0.2e-6),   # die_level (SIO)
}

# CSV filename → (op_type, message_bytes_column_meaning)
_CSV_TO_OP: Dict[str, str] = {
    "hcom_allReduce_.csv":     "all_reduce",
    "hcom_allGather_.csv":     "all_gather",
    "hcom_reduceScatter_.csv": "reduce_scatter",
    "hcom_alltoallv_.csv":     "all_to_all",
}

_DEFAULT_TOLERANCE = 2.0


# ---------------------------------------------------------------------------
# Analytic prediction (mirrors CommAnalyticModel formulas)
# ---------------------------------------------------------------------------

def analytic_predict_us(
    op_type: str,
    message_bytes: int,
    num_devices: int,
    topology_tier: int,
    topologies: Optional[Dict[int, _Topology]] = None,
) -> float:
    """Return analytic latency estimate in microseconds.

    Mirrors CommAnalyticModel ring/tree/recursive algorithm selection.
    message_bytes semantics vary per op to match benchmark CSV conventions:
      - all_reduce: total tensor size (formula uses 2*(n-1)*m/n)
      - all_gather: per-rank chunk size (formula uses (n-1)*m)
      - reduce_scatter: total tensor size (formula uses (n-1)*m/n)
    Both sides (CSV measured values and analytic predictions) use the same
    convention per op, so the ratio comparison remains valid.
    """
    if topologies is None:
        topologies = _A3_TOPOLOGIES

    topo = topologies[topology_tier]
    bw = topo.bandwidth_bytes_ps * topo.comm_efficiency   # effective bytes/s
    lat = topo.latency_s
    n = num_devices
    m = message_bytes

    if n <= 1:
        return 0.0

    if op_type == "all_reduce":
        # Ring:  2*(N-1)*lat + 2*(N-1)*M/N / bw
        # Tree:  2*log2(N)*lat + 2*M / bw
        time_ring = 2 * (n - 1) * lat + 2 * (n - 1) * m / n / bw
        time_tree = 2 * math.log2(n) * lat + 2 * m / bw
        return min(time_ring, time_tree) * 1e6

    elif op_type == "all_gather":
        # Ring:      (N-1)*lat + (N-1)*M / bw
        # Recursive: log2(N)*lat + (N-1)*M / bw
        time_ring = (n - 1) * lat + (n - 1) * m / bw
        time_rec  = math.log2(n) * lat + (n - 1) * m / bw
        return min(time_ring, time_rec) * 1e6

    elif op_type == "reduce_scatter":
        # Ring:      (N-1)*lat + (N-1)*M/N / bw
        # Recursive: log2(N)*lat + (N-1)*M/N / bw
        time_ring = (n - 1) * lat + (n - 1) * m / n / bw
        time_rec  = math.log2(n) * lat + (n - 1) * m / n / bw
        return min(time_ring, time_rec) * 1e6

    elif op_type == "all_to_all":
        # Pairwise: (N-1)*lat + M / bw
        # Bruck:    log2(N)*lat + M / bw
        time_pairwise = (n - 1) * lat + m / bw
        time_bruck    = math.log2(n) * lat + m / bw
        return min(time_pairwise, time_bruck) * 1e6

    else:
        raise ValueError(f"Unknown op_type: {op_type!r}")


# ---------------------------------------------------------------------------
# Per-row alignment result
# ---------------------------------------------------------------------------

@dataclass
class AlignmentRow:
    op_type: str
    message_bytes: int
    num_devices: int
    topology_tier: int
    measured_us: float
    predicted_us: float

    @property
    def ratio(self) -> float:
        if self.predicted_us == 0:
            return float("inf")
        return self.measured_us / self.predicted_us

    def status(self, tolerance: float) -> str:
        r = self.ratio
        if 1.0 / tolerance <= r <= tolerance:
            return "PASS"
        if 1.0 / (tolerance * 2) <= r <= tolerance * 2:
            return "WARN"
        return "FAIL"


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def load_comm_csv(csv_path: Path) -> List[dict]:
    """Load a microbenchmark comm CSV.  Returns list of row dicts."""
    rows = []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _parse_row(row: dict, op_type: str) -> Optional[AlignmentRow]:
    """Parse a CSV row into an AlignmentRow.  Returns None on parse error."""
    try:
        message_bytes = int(row["message_bytes"])
        num_devices   = int(row["num_devices"])
        topology_tier = int(row.get("topology_tier", 1))
        duration_col  = "Average Duration(us)" if "Average Duration(us)" in row else "Duration(us)"
        measured_us   = float(row[duration_col])
    except (KeyError, ValueError):
        return None

    predicted_us = analytic_predict_us(op_type, message_bytes, num_devices, topology_tier)
    return AlignmentRow(
        op_type=op_type,
        message_bytes=message_bytes,
        num_devices=num_devices,
        topology_tier=topology_tier,
        measured_us=measured_us,
        predicted_us=predicted_us,
    )


# ---------------------------------------------------------------------------
# Alignment report
# ---------------------------------------------------------------------------

@dataclass
class AlignmentReport:
    rows: List[AlignmentRow]
    tolerance: float

    @property
    def pass_count(self) -> int:
        return sum(1 for r in self.rows if r.status(self.tolerance) == "PASS")

    @property
    def warn_count(self) -> int:
        return sum(1 for r in self.rows if r.status(self.tolerance) == "WARN")

    @property
    def fail_count(self) -> int:
        return sum(1 for r in self.rows if r.status(self.tolerance) == "FAIL")

    @property
    def mean_ratio(self) -> float:
        if not self.rows:
            return float("nan")
        return sum(r.ratio for r in self.rows) / len(self.rows)

    @property
    def max_ratio(self) -> float:
        if not self.rows:
            return float("nan")
        return max(r.ratio for r in self.rows)

    @property
    def min_ratio(self) -> float:
        if not self.rows:
            return float("nan")
        return min(r.ratio for r in self.rows)

    def ok(self) -> bool:
        return self.fail_count == 0


def validate_csv(csv_path: Path, op_type: str, tolerance: float) -> AlignmentReport:
    """Validate a single comm CSV against analytic predictions."""
    raw_rows = load_comm_csv(csv_path)
    alignment_rows = []
    for raw in raw_rows:
        row = _parse_row(raw, op_type)
        if row is not None:
            alignment_rows.append(row)
    return AlignmentReport(rows=alignment_rows, tolerance=tolerance)


def validate_directory(
    csv_dir: Path,
    tolerance: float = _DEFAULT_TOLERANCE,
) -> Tuple[Dict[str, AlignmentReport], bool]:
    """Validate all known comm CSVs in csv_dir.

    Returns (reports_by_op, all_ok).
    """
    reports: Dict[str, AlignmentReport] = {}
    all_ok = True

    for filename, op_type in _CSV_TO_OP.items():
        csv_path = csv_dir / filename
        if not csv_path.exists():
            print(f"  [SKIP] {filename} not found in {csv_dir}")
            continue
        report = validate_csv(csv_path, op_type, tolerance)
        reports[op_type] = report
        if not report.ok():
            all_ok = False

    return reports, all_ok


# ---------------------------------------------------------------------------
# CLI output
# ---------------------------------------------------------------------------

def _print_report(op_type: str, report: AlignmentReport, verbose: bool) -> None:
    print(f"\n{'='*70}")
    print(f"  {op_type}  ({len(report.rows)} rows, tolerance={report.tolerance:.1f}x)")
    print(f"  PASS={report.pass_count}  WARN={report.warn_count}  FAIL={report.fail_count}")
    if report.rows:
        print(f"  ratio: mean={report.mean_ratio:.2f}x  min={report.min_ratio:.2f}x  max={report.max_ratio:.2f}x")
    print(f"{'='*70}")

    if verbose or report.fail_count > 0 or report.warn_count > 0:
        header = f"  {'status':<6}  {'msg_bytes':>12}  {'n_dev':>5}  {'tier':>4}  {'measured':>10}  {'predicted':>10}  {'ratio':>6}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for row in report.rows:
            status = row.status(report.tolerance)
            if not verbose and status == "PASS":
                continue
            print(
                f"  {status:<6}  {row.message_bytes:>12}  {row.num_devices:>5}  "
                f"{row.topology_tier:>4}  {row.measured_us:>10.2f}  "
                f"{row.predicted_us:>10.2f}  {row.ratio:>6.2f}x"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="C9 comm data alignment: validate HCCL CSVs vs CommAnalyticModel.",
    )
    parser.add_argument(
        "--csv-dir", required=True,
        help="Directory containing hcom_*.csv microbenchmark files",
    )
    parser.add_argument(
        "--tolerance", type=float, default=_DEFAULT_TOLERANCE,
        help=f"Acceptable ratio range [1/tol, tol] (default: {_DEFAULT_TOLERANCE})",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print all rows, not just WARN/FAIL",
    )
    args = parser.parse_args()
    print_logo()

    csv_dir = Path(args.csv_dir)
    if not csv_dir.is_dir():
        print(f"ERROR: {csv_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    print(f"C9 Comm Alignment Validation")
    print(f"  csv_dir   : {csv_dir}")
    print(f"  tolerance : {args.tolerance:.1f}x")

    reports, all_ok = validate_directory(csv_dir, tolerance=args.tolerance)

    for op_type, report in reports.items():
        _print_report(op_type, report, verbose=args.verbose)

    print(f"\n{'='*70}")
    total_rows = sum(len(r.rows) for r in reports.values())
    total_pass = sum(r.pass_count for r in reports.values())
    total_warn = sum(r.warn_count for r in reports.values())
    total_fail = sum(r.fail_count for r in reports.values())
    print(f"  TOTAL: {total_rows} rows  PASS={total_pass}  WARN={total_warn}  FAIL={total_fail}")
    print(f"  Result: {'OK' if all_ok else 'FAILED'}")
    print(f"{'='*70}")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
