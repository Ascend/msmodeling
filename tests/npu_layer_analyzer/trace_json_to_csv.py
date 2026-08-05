#!/usr/bin/env python3
"""将 Chrome Trace Event JSON 转为 kernel_details.csv 格式。

用于将仿真 profiling（如 torch.profiler 导出的 trace.json）
转为 NPU Layer Analyzer / NPU Forward Inspector 可识别的 CSV 格式。
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path


def extract_size_from_tensor(tensor_str: str) -> str:
    """从 tensor(...) 字符串中提取 size。"""
    m = re.search(r"size=\(([^)]+)\)", tensor_str)
    if m:
        return m.group(1).replace(" ", "")
    return ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Chrome Trace Event JSON → kernel_details.csv")
    parser.add_argument("--input", "-i", required=True, help="输入 trace JSON")
    parser.add_argument("--output", "-o", default=None, help="输出 CSV 路径")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    input_path = Path(args.input)
    if not input_path.is_file():
        raise SystemExit(f"输入文件不存在: {input_path}")

    output_path = Path(args.output) if args.output else input_path.parent / f"{input_path.stem}_kernel_details.csv"

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    events = data.get("traceEvents", [])
    # 只保留 X 类型（完整事件）
    x_events = [e for e in events if e.get("ph") == "X" and e.get("cat") == "analytic"]

    if not x_events:
        raise SystemExit("未找到 analytic 类型的 X 事件。")

    rows = []
    task_id = 0
    seen_keys: set[tuple] = set()  # 去重：同 name + 同 ts + 同 input_shapes 只保留第一个

    for ev in x_events:
        full_name = ev.get("name", "")
        # Name = 完整原始名, Type = 第一个 . 前的类别
        name = full_name
        parts = full_name.split(".")
        op_type = parts[0] if len(parts) >= 2 else full_name

        tid = ev.get("tid", 0)
        ts = ev.get("ts", 0)
        dur = ev.get("dur", 0)
        args = ev.get("args", {})

        input_shapes = args.get("simulation_shapes", "")
        output_str = args.get("Output", "")
        output_shapes = extract_size_from_tensor(output_str)

        # 去重：同 name + 同 ts + 同 dur + 同 tid + 同 input_shapes 视为重复
        dedup_key = (full_name, ts, dur, tid, input_shapes)
        if dedup_key in seen_keys:
            continue
        seen_keys.add(dedup_key)

        # 补充 Category
        category = args.get("cat", ev.get("cat", ""))

        rows.append(
            {
                "Stream ID": str(tid),
                "Task ID": str(task_id),
                "Name": name,
                "Type": op_type,
                "Start Time(us)": str(ts),
                "Duration(us)": str(dur),
                "Input Shapes": input_shapes,
                "Output Shapes": output_shapes,
                "Category": category,
                "Full Name": full_name,
            }
        )
        task_id += 1

    fieldnames = [
        "Stream ID",
        "Task ID",
        "Name",
        "Type",
        "Start Time(us)",
        "Duration(us)",
        "Input Shapes",
        "Output Shapes",
        "Category",
        "Full Name",
    ]

    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(
        f"转换完成: {len(rows)} 行（去重前 {len(x_events)} 行，去除 {len(x_events) - len(rows)} 个重复） → {output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
