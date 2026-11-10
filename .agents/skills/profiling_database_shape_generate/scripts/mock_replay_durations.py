# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of the software at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""模式A步骤1（本地验证）：仅为无时长的数据行填充占位时长，产出 mock 验证数据库。

复制目标性能数据库到 --output-dir，对任一时长列（Average Duration(us) /
Profiling Average Duration(us)）缺失或 <=0 的行填充固定占位时长，模拟"已完成回放"；
已有实测值的时长列保持原值不变。mock 库仅供 cli.inference.text_generate 本地
命中率/插值验证，严禁回传正式数据库或用于发布。
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

LATENCY_COLS = ("Average Duration(us)", "Profiling Average Duration(us)")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Copy a profiling database and fill placeholder durations into not-yet-replayed rows.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--database-path", type=Path, required=True, help="Source database directory.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Mock database output directory.")
    parser.add_argument("--placeholder-us", type=float, default=100.0, help="Fixed placeholder duration (us).")
    parser.add_argument("--overwrite", action="store_true", help="Remove an existing output directory first.")
    return parser


def mock_csv(path: Path, placeholder_us: float) -> tuple[int, int, bool]:
    """Fill placeholder into rows lacking any latency value; return (total_rows, mocked_rows, has_latency_cols).

    行级 union 语义判定待 mock 行：任一时长列缺失/<=0 即视为待 mock 行；该行中
    缺失/<=0 的时长列填占位值，已有实测值的时长列保持不动。
    """
    frame = pd.read_csv(path, encoding="utf-8-sig")
    total = len(frame)
    available = [col for col in LATENCY_COLS if col in frame.columns]
    if not available:
        return total, 0, False

    def to_number(value: object) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number

    pending = None
    for col in available:
        numeric = frame[col].map(to_number)
        col_pending = numeric.isna() | (numeric <= 0)
        pending = col_pending if pending is None else (pending | col_pending)
    mocked_count = int(pending.sum()) if pending is not None else 0
    if mocked_count == 0:
        return total, 0, True

    for col in available:
        numeric = frame[col].map(to_number)
        col_pending = numeric.isna() | (numeric <= 0)
        frame.loc[col_pending, col] = placeholder_us
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return total, mocked_count, True


def main() -> None:
    args = build_argparser().parse_args()
    source = args.database_path.resolve()
    output = args.output_dir.resolve()
    if not source.is_dir():
        raise SystemExit(f"database directory not found: {source}")
    if not (source / "op_mapping.yaml").is_file():
        raise SystemExit(f"not a profiling database (missing op_mapping.yaml): {source}")
    if output.exists():
        if not args.overwrite:
            raise SystemExit(f"output directory already exists: {output} (pass --overwrite to replace)")
        shutil.rmtree(output)

    shutil.copytree(source, output)
    print(f"[mock_replay_durations] copied {source} -> {output}")

    total_mocked = 0
    missing_latency_cols: list[str] = []
    for csv_path in sorted(output.glob("*.csv")):
        total, mocked, has_latency_cols = mock_csv(csv_path, args.placeholder_us)
        if mocked:
            total_mocked += mocked
            print(f"  {csv_path.name}: mocked {mocked}/{total} rows -> {args.placeholder_us}us")
        if not has_latency_cols:
            missing_latency_cols.append(csv_path.name)
    if total_mocked == 0:
        print("  no pending rows found; nothing was mocked")
    if missing_latency_cols:
        print(
            "  WARNING: no latency column in these CSVs (rows left unmocked; queries against them "
            "will MISS unless the kernel is zero_cost/accepted_miss): " + ", ".join(missing_latency_cols)
        )

    print(
        "\n[mock_replay_durations done] mock database ready for local verification only:\n"
        f"  {output}\n"
        "Next: run scripts/verify_db_coverage.py --database-path <mock dir> --cases-yaml <cases yaml> "
        "[--cases <ids>]"
    )


if __name__ == "__main__":
    main()
