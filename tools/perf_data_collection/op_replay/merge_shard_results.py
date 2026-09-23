#!/usr/bin/env python3

# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Merge several sharded microbench database directories into one.

Each shard is an independent DB copy that a single ``start_microbench`` process
wrote back to. The standalone CLI keeps its general min-of-N merge behavior.
The public multi-device runner additionally supplies case and operator ownership
so every row is taken only from the worker that was assigned to measure it.

Usage::

    python merge_shard_results.py --output <merged_dir> <shard_0> <shard_1> ...

Only CSVs that exist in at least one shard are written. CSVs present in some
shards but missing from shard 0 are still merged (shard 0 is only the source of
the canonical column header when present).
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import sys
from pathlib import Path
from typing import Mapping, Sequence

try:
    from .common import case_belongs_to_shard, normalize_op_name
    from .operator_metadata import supports_case_sharding
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from common import case_belongs_to_shard, normalize_op_name
    from operator_metadata import supports_case_sharding

try:
    from signature_utils import get_case_shard_key, get_sig
except ImportError:
    from ..signature_utils import get_case_shard_key, get_sig


DURATION_COLUMNS = (
    "Average Duration(us)",
    "Profiling Average Duration(us)",
    "Profiling Median Duration(us)",
)


def _parse_float(value: str | None) -> float:
    if value is None:
        return 0.0
    try:
        return float(str(value).strip())
    except (ValueError, TypeError):
        return 0.0


def _row_best_duration(row: dict[str, str]) -> float:
    """Return the smallest positive duration across the measured columns, or 0."""
    best = 0.0
    for col in DURATION_COLUMNS:
        v = _parse_float(row.get(col))
        if math.isfinite(v) and v > 0 and (best == 0.0 or v < best):
            best = v
    return best


def _merge_one_csv(
    csv_name: str,
    shard_dirs: list[Path],
    out_path: Path,
    *,
    case_shard_count: int = 1,
    operator_shard_index: int | None = None,
) -> int:
    """Merge ``csv_name`` across shards. Returns number of merged rows."""
    header: list[str] | None = None
    # sig (str) -> row dict. Insertion order preserves a stable row order.
    merged: dict[str, dict[str, str]] = {}

    op_name = Path(csv_name).stem
    for shard_index, shard_dir in enumerate(shard_dirs):
        if operator_shard_index is not None and shard_index != operator_shard_index:
            continue
        p = shard_dir / csv_name
        if not p.is_file():
            continue
        # Preserve BOM-aware reading: csv.DictReader handles the BOM only if the
        # file is opened with utf-8-sig.
        with open(p, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                if header is None:
                    header = list(reader.fieldnames)
                elif reader.fieldnames != header:
                    # Column drift between shards — keep the wider header set.
                    for col in reader.fieldnames:
                        if col not in header:
                            header.append(col)
            for row in reader:
                # DictReader may return extra keys (None-keyed trailing column);
                # drop them so they don't leak into the writer.
                row = {k: v for k, v in row.items() if k is not None}
                if (
                    case_shard_count > 1
                    and supports_case_sharding(op_name)
                    and not case_belongs_to_shard(
                        get_case_shard_key(row, op_name),
                        case_shard_count,
                        shard_index,
                    )
                ):
                    continue
                sig = get_sig(row, as_str=True, op_name=op_name)
                if not sig:
                    continue
                existing = merged.get(sig)
                if existing is None:
                    merged[sig] = row
                    continue
                # Same signature: keep the row with the smaller positive duration.
                cur_best = _row_best_duration(existing)
                new_best = _row_best_duration(row)
                if new_best > 0 and (cur_best == 0.0 or new_best < cur_best):
                    # Replace duration columns but keep the first-seen identity
                    # columns (Input Shapes, Runtime fields, etc.) stable.
                    for col in DURATION_COLUMNS:
                        if col in row:
                            existing[col] = row[col]
                # Backfill any columns the existing row is missing.
                for col, val in row.items():
                    if col not in existing or (not existing[col] and val):
                        existing[col] = val

    if header is None or not merged:
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Write with UTF-8 BOM to match the existing DB CSV convention.
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        for row in merged.values():
            # Ensure every header column is present.
            for col in header:
                row.setdefault(col, "")
            writer.writerow(row)
    return len(merged)


def merge_shard_directories(
    shards: list[Path],
    out_dir: Path,
    *,
    case_shard_count: int | None = None,
    operator_shard_assignments: Mapping[str, int] | None = None,
    operators: Sequence[str] | None = None,
) -> int:
    """Merge shard database directories into ``out_dir``.

    Returns the number of rows written across all operator CSV files.
    """
    shard_dirs = [shard for shard in shards if shard.is_dir()]
    if not shard_dirs:
        raise ValueError("no shard directories provided")
    if case_shard_count is not None and (case_shard_count <= 0 or case_shard_count != len(shard_dirs)):
        raise ValueError("case_shard_count must match the number of shard directories")
    resolved_shard_count = case_shard_count or 1
    assignments = dict(operator_shard_assignments or {})
    if any(index < 0 or index >= len(shard_dirs) for index in assignments.values()):
        raise ValueError("operator shard assignment is outside the shard directory range")
    selected_operators = (
        None if operators is None else {normalize_op_name(operator_name) for operator_name in operators}
    )

    # Collect the union of CSV file names across shards.
    csv_names: set[str] = set()
    for d in shard_dirs:
        csv_names.update(
            path.name
            for path in d.glob("*.csv")
            if selected_operators is None or normalize_op_name(path.stem) in selected_operators
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for csv_name in sorted(csv_names):
        n = _merge_one_csv(
            csv_name,
            shard_dirs,
            out_dir / csv_name,
            case_shard_count=resolved_shard_count,
            operator_shard_index=assignments.get(Path(csv_name).stem),
        )
        if n:
            total += n
            print(f"  {csv_name}: {n} rows")
    # Recursively copy non-CSV files (e.g. op_mapping.yaml, reports/,
    # msprof artifacts) from shard 0 so sub-directory products are preserved.
    for src in shard_dirs[0].rglob("*"):
        if src.is_file() and src.suffix != ".csv":
            target = out_dir / src.relative_to(shard_dirs[0])
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, target)
    print(f"merged {total} rows across {len(csv_names)} CSVs -> {out_dir}")
    return total


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge sharded microbench DB directories by row signature."
    )
    parser.add_argument("--output", required=True, type=Path, help="Merged output DB directory.")
    parser.add_argument(
        "shards",
        nargs="+",
        type=Path,
        help="Shard DB directories (order: shard 0 first; its header is canonical).",
    )
    args = parser.parse_args()

    try:
        merge_shard_directories(args.shards, args.output)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
