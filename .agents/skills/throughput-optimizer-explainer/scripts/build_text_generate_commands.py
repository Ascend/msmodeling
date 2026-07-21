#!/usr/bin/env python3
import argparse
import json
import math
import shlex
import sys
from pathlib import Path


def read_json(path: str | None) -> dict:
    text = Path(path).read_text(encoding="utf-8-sig") if path else sys.stdin.read()
    return json.loads(text)


def parse_parallel(label: str | None) -> dict:
    result = {}
    if not label:
        return result
    for part in label.split("|"):
        if "=" not in part:
            continue
        key, value = part.strip().split("=", 1)
        try:
            result[key.strip().lower().replace("-", "_")] = int(value.strip())
        except ValueError:
            print(
                f"Warning: cannot parse parallel parameter '{key.strip()}' value '{value.strip()}' as int, skipping",
                file=sys.stderr,
            )
    return result


def add_if(cmd: list[str], flag: str, value):
    if value is not None:
        cmd.extend([flag, str(value)])


def base_cmd(data: dict, parallel: dict, include_op_bound: bool = False) -> list[str]:
    cmd = ["python", "-m", "cli.inference.text_generate", str(data["model"])]
    for key, flag in [
        ("device", "--device"),
        ("num_devices", "--num-devices"),
        ("quantize_linear_action", "--quantize-linear-action"),
        ("quantize_attention_action", "--quantize-attention-action"),
        ("mxfp4_group_size", "--mxfp4-group-size"),
    ]:
        add_if(cmd, flag, data.get(key))
    if data.get("compile"):
        cmd.append("--compile")
    add_if(cmd, "--tp-size", parallel.get("tp"))
    add_if(cmd, "--dp-size", parallel.get("dp"))
    add_if(cmd, "--ep-size", parallel.get("ep"))
    add_if(cmd, "--moe-tp-size", parallel.get("moe_tp"))
    add_if(cmd, "--moe-dp-size", parallel.get("moe_dp"))
    if include_op_bound:
        cmd.append("--dump-op-bound-results")
    return cmd


def shell_join(cmd: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in cmd)


def build_aggregation(data: dict, include_op_bound: bool = False) -> dict:
    input_length = int(data["input_length"])
    output_length = int(data["output_length"])
    max_batched_tokens = int(data.get("max_batched_tokens", 8192))
    hit_rate = float(data.get("prefix_cache_hit_rate", 0.0))
    num_mtp_tokens = int(data.get("num_mtp_tokens", 0))
    concurrency = int(data["concurrency"])
    effective_input_length = max(1, input_length - math.floor(input_length * hit_rate))
    prefill_batch_size = max_batched_tokens // effective_input_length
    if prefill_batch_size < 1:
        raise ValueError("max_batched_tokens must be >= effective_input_length for aggregation validation")
    parallel = parse_parallel(data.get("parallel"))

    prefill = base_cmd(data, parallel, include_op_bound=include_op_bound)
    prefill.extend(
        [
            "--num-queries",
            str(prefill_batch_size),
            "--query-length",
            str(effective_input_length),
            "--context-length",
            "0",
        ]
    )

    decode = base_cmd(data, parallel, include_op_bound=include_op_bound)
    decode.extend(
        [
            "--num-queries",
            str(concurrency),
            "--query-length",
            str(num_mtp_tokens + 1),
            "--context-length",
            str(input_length + output_length // 2),
            "--decode",
        ]
    )

    return {
        "mode": "aggregation",
        "effective_input_length": effective_input_length,
        "prefill_batch_size": prefill_batch_size,
        "partial_prefill_wave": concurrency % prefill_batch_size,
        "prefill_command": shell_join(prefill),
        "decode_command": shell_join(decode),
    }


def build_disaggregation(data: dict, include_op_bound: bool = False) -> dict:
    input_length = int(data["input_length"])
    output_length = int(data["output_length"])
    concurrency = int(data["concurrency"])
    hit_rate = float(data.get("prefix_cache_hit_rate", 0.0))
    num_mtp_tokens = int(data.get("num_mtp_tokens", 0))
    phase = data.get("phase")
    if phase not in {"prefill", "decode"}:
        raise ValueError("disaggregation input must include phase='prefill' or phase='decode'")
    parallel = parse_parallel(data.get("parallel"))
    cmd = base_cmd(data, parallel, include_op_bound=include_op_bound)
    cmd.extend(["--num-queries", str(concurrency)])
    if phase == "prefill":
        effective_input_length = max(1, input_length - math.floor(input_length * hit_rate))
        cmd.extend(["--query-length", str(effective_input_length), "--context-length", "0"])
        return {"mode": "disaggregation", "phase": phase, "command": shell_join(cmd)}
    cmd.extend(
        [
            "--query-length",
            str(num_mtp_tokens + 1),
            "--context-length",
            str(input_length + output_length // 2),
            "--decode",
        ]
    )
    return {"mode": "disaggregation", "phase": phase, "command": shell_join(cmd)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build text_generate validation commands from a best-row JSON.")
    parser.add_argument("path", nargs="?", help="Optional JSON file. Reads stdin if omitted.")
    parser.add_argument("--mode", choices=["aggregation", "disaggregation"], required=True)
    parser.add_argument(
        "--include-op-bound",
        action="store_true",
        help="Append --dump-op-bound-results to generated text_generate commands.",
    )
    args = parser.parse_args()
    data = read_json(args.path)
    result = (
        build_aggregation(data, include_op_bound=args.include_op_bound)
        if args.mode == "aggregation"
        else build_disaggregation(data, include_op_bound=args.include_op_bound)
    )
    json.dump(result, sys.stdout, ensure_ascii=True, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
