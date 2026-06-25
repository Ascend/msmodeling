#!/usr/bin/env python3
import argparse
import json
import re
import sys
from pathlib import Path


FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
TIME_RE = re.compile(rf"^\s*({FLOAT_RE})\s*(s|ms|us|µs|ns)\s*$", re.IGNORECASE)
BOUND_KEYS = {
    "memory": "memory_pct",
    "comm": "comm_pct",
    "mma": "mma_pct",
    "gp": "gp_pct",
}


def read_text(path: str | None) -> str:
    if path:
        return Path(path).read_text(encoding="utf-8")
    return sys.stdin.read()


def parse_float(pattern: str, text: str):
    match = re.search(pattern, text, re.IGNORECASE)
    return float(match.group(1)) if match else None


def detect_mode(text: str) -> str:
    lowered = text.lower()
    if "pd ratio configurations" in lowered or "pd ratio:" in lowered:
        return "pd_ratio"
    if "aggregation configurations" in lowered or "percentage_breakdowns(p)" in lowered:
        return "aggregation"
    if "disaggregation" in lowered and "ttft" in lowered and "tpot" in lowered:
        return "disaggregation"
    return "unknown"


def parse_key_values(text: str) -> dict:
    fields = {
        "best_throughput_tokens_per_s": rf"Best Throughput:\s*({FLOAT_RE})\s*tokens/s",
        "ttft_ms": rf"TTFT:\s*({FLOAT_RE})\s*ms",
        "tpot_ms": rf"TPOT:\s*({FLOAT_RE})\s*ms",
        "pd_ratio": rf"PD Ratio:\s*({FLOAT_RE})",
        "prefill_qps": rf"Prefill QPS:\s*({FLOAT_RE})\s*req/s",
        "decode_qps": rf"Decode QPS:\s*({FLOAT_RE})\s*req/s",
    }
    return {name: value for name, pat in fields.items() if (value := parse_float(pat, text)) is not None}


def split_row_by_positions(row: str, positions: list[int]) -> list[str]:
    values = []
    for left, right in zip(positions, positions[1:]):
        values.append(row[left + 1 : right].strip().replace("\x1b[0m", "").replace("\x1b[1m", ""))
    return values


def parse_pretty_tables(text: str) -> list[dict]:
    lines = text.splitlines()
    tables = []
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped.startswith("Top ") and stripped.endswith(":"):
            title = stripped[:-1]
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("|"):
                i += 1
            if i >= len(lines):
                break
            header_line = lines[i].rstrip("\n")
            positions = [idx for idx, ch in enumerate(header_line) if ch == "|"]
            headers = split_row_by_positions(header_line, positions)
            rows = []
            i += 1
            while i < len(lines):
                raw = lines[i].rstrip("\n")
                row_stripped = raw.strip()
                if not row_stripped:
                    i += 1
                    continue
                if row_stripped.startswith("Top ") and row_stripped.endswith(":"):
                    break
                if not row_stripped.startswith("|"):
                    if rows:
                        break
                    i += 1
                    continue
                if set(row_stripped) <= {"+", "-", "|"}:
                    i += 1
                    continue
                values = split_row_by_positions(raw, positions)
                if len(values) == len(headers):
                    rows.append(dict(zip(headers, values)))
                i += 1
            tables.append({"title": title, "headers": headers, "rows": rows})
            continue
        i += 1
    return tables


def parse_dump_rows(text: str) -> list[dict]:
    rows = []
    headers = None
    for line in text.splitlines():
        if "percentage_breakdowns" in line and "device_name" in line:
            headers = re.split(r"\s{2,}", line.strip())
            continue
        if not headers or not line.strip():
            continue
        parts = re.split(r"\s{2,}", line.strip(), maxsplit=len(headers) - 1)
        if len(parts) == len(headers):
            rows.append(dict(zip(headers, parts)))
    return rows


def parse_text_generate_breakdowns(text: str) -> dict:
    output = {}
    in_section = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "Stats breakdowns:":
            in_section = True
            continue
        if not in_section or ":" not in stripped:
            continue
        name, rest = stripped.split(":", 1)
        values = {}
        for item in rest.split(","):
            if ":" not in item:
                continue
            key, val = item.split(":", 1)
            key = key.strip()
            val = val.strip()
            try:
                values[key] = float(val)
            except ValueError:
                print(
                    f"warning: unable to parse text_generate breakdown value {name.strip()}.{key}={val!r}",
                    file=sys.stderr,
                )
        if values:
            output[name.strip()] = values
    return output


def parse_time_seconds(value: str):
    match = TIME_RE.match(value)
    if not match:
        return None
    amount = float(match.group(1))
    unit = match.group(2).lower()
    if unit == "s":
        return amount
    if unit == "ms":
        return amount / 1_000
    if unit in {"us", "µs"}:
        return amount / 1_000_000
    if unit == "ns":
        return amount / 1_000_000_000
    return None


def parse_percent(value: str):
    try:
        return float(value.strip().rstrip("%"))
    except ValueError:
        return None


def parse_int(value: str):
    try:
        return int(value.strip())
    except ValueError:
        return None


def is_separator(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and set(stripped) <= {"-", " "}


def is_op_bound_header(parts: list[str]) -> bool:
    lowered = [part.lower() for part in parts]
    return (
        "name" in lowered
        and "# of calls" in lowered
        and (any(part.startswith("bound") for part in lowered) or any(part.endswith(" %") for part in lowered))
    )


def split_aligned_table_row(line: str) -> list[str]:
    return re.split(r"\s{2,}", line.strip())


def first_model_name(row: dict) -> str | None:
    models = row.get("perf_models") or {}
    if not models:
        return None
    return next(iter(models))


def promote_first_model_fields(row: dict) -> dict:
    model_name = first_model_name(row)
    if not model_name:
        return row
    model_data = row["perf_models"][model_name]
    row["perf_model"] = model_name
    for key in ("total_s", "avg_s", "memory_pct", "comm_pct", "mma_pct", "gp_pct"):
        if key in model_data:
            row[key] = model_data[key]
    return row


def parse_op_bound_row(headers: list[str], values: list[str]) -> dict:
    row = {"perf_models": {}}
    for header, value in zip(headers, values):
        lowered = header.lower()
        if lowered == "name":
            row["name"] = value
        elif lowered.startswith("bound"):
            row["bound"] = value
        elif lowered == "# of calls":
            row["call_times"] = parse_int(value)
        elif lowered.endswith(" total"):
            model = header[: -len(" total")]
            row["perf_models"].setdefault(model, {})["total_s"] = parse_time_seconds(value)
            row["perf_models"][model]["total_raw"] = value
        elif lowered.endswith(" avg"):
            model = header[: -len(" avg")]
            row["perf_models"].setdefault(model, {})["avg_s"] = parse_time_seconds(value)
            row["perf_models"][model]["avg_raw"] = value
        else:
            for suffix, out_key in BOUND_KEYS.items():
                marker = f" {suffix} %"
                if lowered.endswith(marker):
                    model = header[: -len(marker)]
                    row["perf_models"].setdefault(model, {})[out_key] = parse_percent(value)
                    break
    return promote_first_model_fields(row)


def parse_text_generate_op_bounds(text: str) -> list[dict]:
    rows = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        headers = split_aligned_table_row(lines[i])
        if not is_op_bound_header(headers):
            i += 1
            continue

        i += 1
        while i < len(lines) and is_separator(lines[i]):
            i += 1

        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            if not stripped or stripped.startswith("Total time for"):
                break
            if is_separator(line):
                i += 1
                continue
            values = split_aligned_table_row(line)
            if len(values) == len(headers):
                row = parse_op_bound_row(headers, values)
                if row.get("name"):
                    rows.append(row)
            i += 1
        break
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Parse throughput_optimizer or text_generate output into JSON.")
    parser.add_argument("path", nargs="?", help="Optional input file. Reads stdin if omitted.")
    args = parser.parse_args()

    text = read_text(args.path)
    payload = {
        "mode": detect_mode(text),
        "best": parse_key_values(text),
        "tables": parse_pretty_tables(text),
        "dump_rows": parse_dump_rows(text),
        "text_generate_breakdowns": parse_text_generate_breakdowns(text),
        "text_generate_op_bounds": parse_text_generate_op_bounds(text),
    }
    json.dump(payload, sys.stdout, ensure_ascii=True, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
