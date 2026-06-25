#!/usr/bin/env python3
import argparse
import json
import re
import sys
from pathlib import Path


KEYS = ["Mem", "Comm", "Cube", "Vec"]
BOUND_KEYS = ["memory_bound", "communication_bound", "compute_bound_mma", "compute_bound_gp"]
PCT_KEYS = ["memory_pct", "comm_pct", "mma_pct", "gp_pct"]


def parse_breakdown(value) -> dict:
    if isinstance(value, dict):
        return {key: float(value.get(key, value.get(key.lower(), 0.0))) for key in KEYS}
    result = {key: 0.0 for key in KEYS}
    if not isinstance(value, str):
        return result
    for key in KEYS:
        match = re.search(rf"\b{key}\b\s*:?\s*([-+]?\d+(?:\.\d+)?)", value, re.IGNORECASE)
        if match:
            result[key] = float(match.group(1))
    return result


def read_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def normalize_op_rows(payload, field: str) -> list[dict]:
    rows = payload if isinstance(payload, list) else payload.get(field, [])
    normalized = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        item = dict(row)
        if "total_s" not in item:
            models = item.get("perf_models") or {}
            if models:
                model_data = next(iter(models.values()))
                item.update({key: model_data.get(key) for key in ("total_s", "avg_s", *PCT_KEYS)})
        item["total_s"] = float(item.get("total_s") or 0.0)
        item["bound"] = item.get("bound") or "unknown"
        item["name"] = item.get("name") or "<unknown>"
        for key in PCT_KEYS:
            value = item.get(key)
            item[key] = float(value) if value is not None else None
        normalized.append(item)
    return sorted(normalized, key=lambda row: row["total_s"], reverse=True)


def bound_distribution(rows: list[dict]) -> dict:
    totals = {key: 0.0 for key in BOUND_KEYS}
    totals["unknown"] = 0.0
    counts = {key: 0 for key in totals}
    for row in rows:
        bound = row.get("bound") or "unknown"
        if bound not in totals:
            totals[bound] = 0.0
            counts[bound] = 0
        totals[bound] += row["total_s"]
        counts[bound] += 1
    return {"total_s_by_bound": totals, "count_by_bound": counts}


def row_summary(row: dict | None) -> dict | None:
    if row is None:
        return None
    return {
        "name": row["name"],
        "bound": row["bound"],
        "total_s": row["total_s"],
        "memory_pct": row.get("memory_pct"),
        "comm_pct": row.get("comm_pct"),
        "mma_pct": row.get("mma_pct"),
        "gp_pct": row.get("gp_pct"),
    }


def compare_op_bounds(left_payload: dict, right_payload: dict, field: str, top_n: int) -> dict:
    left_rows = normalize_op_rows(left_payload, field)
    right_rows = normalize_op_rows(right_payload, field)
    left_by_name = {row["name"]: row for row in left_rows}
    right_by_name = {row["name"]: row for row in right_rows}
    top_names = []
    for row in left_rows[:top_n] + right_rows[:top_n]:
        if row["name"] not in top_names:
            top_names.append(row["name"])

    top_diffs = []
    for name in top_names:
        left = left_by_name.get(name)
        right = right_by_name.get(name)
        diff = {
            "name": name,
            "left": row_summary(left),
            "right": row_summary(right),
            "delta_total_s_right_minus_left": (right["total_s"] if right else 0.0) - (left["total_s"] if left else 0.0),
        }
        for key in PCT_KEYS:
            left_value = left.get(key) if left else None
            right_value = right.get(key) if right else None
            diff[f"delta_{key}_right_minus_left"] = (
                right_value - left_value if right_value is not None and left_value is not None else None
            )
        top_diffs.append(diff)

    return {
        "left_distribution": bound_distribution(left_rows),
        "right_distribution": bound_distribution(right_rows),
        "top_diffs": top_diffs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare Cube/Vec/Comm/Mem phase breakdowns.")
    parser.add_argument("left")
    parser.add_argument("right")
    parser.add_argument("--field", default="breakdown", help="JSON field containing the breakdown.")
    parser.add_argument("--op-bound", action="store_true", help="Compare parsed text_generate op-bound tables.")
    parser.add_argument("--op-field", default="text_generate_op_bounds", help="JSON field containing op-bound rows.")
    parser.add_argument("--top-n", type=int, default=10, help="Number of top ops from each side to compare.")
    args = parser.parse_args()

    left_payload = read_json(args.left)
    right_payload = read_json(args.right)
    if args.op_bound:
        json.dump(
            compare_op_bounds(left_payload, right_payload, args.op_field, args.top_n),
            sys.stdout,
            indent=2,
        )
        sys.stdout.write("\n")
        return 0

    left = parse_breakdown(left_payload.get(args.field))
    right = parse_breakdown(right_payload.get(args.field))
    diff = {key: right[key] - left[key] for key in KEYS}
    ratio = {key: (right[key] / left[key] if left[key] else None) for key in KEYS}
    json.dump(
        {"left": left, "right": right, "delta_right_minus_left": diff, "ratio_right_over_left": ratio},
        sys.stdout,
        indent=2,
    )
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
