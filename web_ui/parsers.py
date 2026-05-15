from __future__ import annotations

import re
from typing import Any

from .schemas import ExperimentResult, ExperimentTask

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

TIME_RE = re.compile(r"^([0-9.]+)(ns|us|ms|s)$")


def time_to_us(text: str) -> float:
    text = text.strip()
    m = TIME_RE.match(text)
    if not m:
        return 0.0
    value = float(m.group(1))
    unit = m.group(2)
    return {
        "ns": value / 1000.0,
        "us": value,
        "ms": value * 1000.0,
        "s": value * 1_000_000.0,
    }[unit]


def time_to_seconds(text: str) -> float:
    return time_to_us(text) / 1_000_000.0


def _strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text or "")


def _extract_execution_error(log: str, fallback: str | None = None) -> str | None:
    lines = [_strip_ansi(line).strip() for line in (log or "").splitlines()]
    if "huggingface.co" in (log or "") and "couldn't find them in the cached files" in (log or ""):
        return (
            "Unable to download model files from HuggingFace and no local cache was found. "
            "Check network access or use a cached/local model source."
        )
    candidates = [
        line
        for line in lines
        if line
        and (
            "ModuleNotFoundError" in line
            or "ImportError" in line
            or "CalledProcessError" in line
            or "No matching distribution found" in line
            or "Permission denied" in line
            or "Error while finding module specification" in line
            or "OSError:" in line
            or "couldn't connect to 'https://huggingface.co'" in line
            or "couldn't find them in the cached files" in line
            or line.startswith(("ERROR:", "Traceback"))
        )
    ]
    if candidates:
        return candidates[-1]
    for line in reversed(lines):
        if line:
            return line
    return fallback


def _optimizer_no_result_reason(task: ExperimentTask) -> str:
    ttft = task.params.get("ttft_limits")
    tpot = task.params.get("tpot_limits")
    ttft_text = f"{ttft:g} ms" if isinstance(ttft, (int, float)) else "unlimited"
    tpot_text = f"{tpot:g} ms" if isinstance(tpot, (int, float)) else "unlimited"
    return (
        f"No valid deployment was found under the current limits "
        f"(TTFT={ttft_text}, TPOT={tpot_text})."
        "Try relaxing the latency limits, increasing num-devices, "
        "changing quantization, or reducing input/output length."
    )


def _parse_optimizer_row(cells: list[str]) -> dict[str, Any] | None:
    if len(cells) < 8 or not cells[0].isdigit():
        return None
    try:
        return {
            "rank": int(cells[0]),
            "throughput_token_s": float(cells[1]),
            "ttft_ms": float(cells[2]),
            "tpot_ms": float(cells[3]),
            "concurrency": int(cells[4]),
            "num_devices": int(cells[5]),
            "parallel": " | ".join(part for part in cells[6:-1] if part),
            "batch_size": int(cells[-1]),
        }
    except ValueError:
        return None


def _parse_table(lines: list[str]) -> list[dict[str, Any]]:
    rows = []
    for line in lines:
        stripped = _strip_ansi(line.rstrip())
        if not stripped or stripped.startswith(("-", "+")):
            continue
        if "analytic total" in stripped and "# of Calls" in stripped:
            continue
        if stripped.startswith("|"):
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            optimizer_row = _parse_optimizer_row(cells)
            if optimizer_row is not None:
                rows.append(optimizer_row)
            continue
        m = re.match(
            r"^(.*?)\s+([0-9.]+(?:ns|us|ms|s))\s+([0-9.]+(?:ns|us|ms|s))\s+(\d+)$",
            stripped,
        )
        if m:
            rows.append(
                {
                    "name": m.group(1).strip(),
                    "analytic_total_raw": m.group(2),
                    "analytic_total_us": time_to_us(m.group(2)),
                    "analytic_avg_raw": m.group(3),
                    "analytic_avg_us": time_to_us(m.group(3)),
                    "num_calls": int(m.group(4)),
                }
            )
    return rows


def parse_text_generate(task: ExperimentTask, log: str, status: str, error: str | None = None) -> ExperimentResult:
    summary = {}
    warnings = []
    infos = []
    table_lines = []
    in_table = False
    for line in log.splitlines():
        stripped = _strip_ansi(line.strip())
        if stripped.startswith("WARNING"):
            warnings.append(stripped)
        elif stripped.startswith("INFO"):
            infos.append(stripped)
        if stripped.startswith("Number of Queries per DP rank:"):
            summary["queries_per_dp_rank"] = int(stripped.split(":", 1)[1].strip())
        elif stripped.startswith("Model compilation and execution time:"):
            summary["run_time_s"] = float(stripped.split(":", 1)[1].strip().split()[0])
        elif stripped.startswith("Total time for analytic:"):
            summary["analytic_total_time_s"] = time_to_seconds(stripped.split(":", 1)[1].strip())
            in_table = False
        elif stripped.startswith("TPS/Device:"):
            summary["tps_per_device"] = float(stripped.split(":", 1)[1].strip().split()[0])
        elif stripped.startswith("Total device memory:"):
            summary["total_device_memory_gb"] = float(stripped.split(":", 1)[1].strip().split()[0])
        elif stripped.startswith("Model weight size:"):
            summary["model_weight_size_gb"] = float(stripped.split(":", 1)[1].strip().split()[0])
        elif stripped.startswith("KV cache:"):
            summary["kv_cache_gb"] = float(stripped.split(":", 1)[1].strip().split()[0])
        elif stripped.startswith("Model activation size:"):
            summary["model_activation_size_gb"] = float(stripped.split(":", 1)[1].strip().split()[0])
        elif stripped.startswith("Reserved memory:"):
            summary["reserved_memory_gb"] = float(stripped.split(":", 1)[1].strip().split()[0])
        elif stripped.startswith("Memory available:"):
            summary["memory_available_gb"] = float(stripped.split(":", 1)[1].strip().split()[0])
            summary["memory_fit_status"] = "oom_risk" if summary["memory_available_gb"] < 0 else "fit"
        elif "analytic_OpBound:" in stripped:
            parts = stripped.split("analytic_OpBound:", 1)[1].strip().split(",")
            for part in parts:
                if ":" in part:
                    k, v = part.split(":", 1)
                    summary[k.strip()] = float(v.strip())
        if "Name" in stripped and "analytic total" in stripped and "# of Calls" in stripped:
            in_table = True
            table_lines = []
        elif in_table:
            table_lines.append(line)
    tables = {"op_breakdown": _parse_table(table_lines)}
    if task.params.get("decode"):
        summary["stage"] = "decode"
    else:
        summary["stage"] = "prefill"
    summary["bottleneck_type"] = _pick_bottleneck(summary)
    if status != "success":
        summary["execution_error"] = _extract_execution_error(log, error)
    return ExperimentResult(
        task.sim_type,
        status,
        task.params,
        task.command,
        task.task_hash,
        task.label,
        summary,
        tables,
        warnings,
        infos,
        log,
        error,
    )


def _pick_bottleneck(summary: dict[str, Any]) -> str | None:
    candidates = {
        "memory_bound": summary.get("memory_bound"),
        "communication_bound": summary.get("communication_bound"),
        "compute_bound_mma": summary.get("compute_bound_mma"),
        "compute_bound_gp": summary.get("compute_bound_gp"),
    }
    valid = {k: v for k, v in candidates.items() if isinstance(v, (int, float))}
    if not valid:
        return None
    return max(valid, key=valid.get)


def parse_video_generate(task: ExperimentTask, log: str, status: str, error: str | None = None) -> ExperimentResult:
    summary = {
        "cfg_mode": "cfg_parallel"
        if task.params.get("cfg_parallel") and task.params.get("use_cfg")
        else ("batch_concat" if task.params.get("use_cfg") else "disabled")
    }
    warnings = []
    infos = []
    table_lines = []
    in_table = False
    for line in log.splitlines():
        stripped = _strip_ansi(line.strip())
        if stripped.startswith("WARNING"):
            warnings.append(stripped)
        elif stripped.startswith("INFO"):
            infos.append(stripped)
        if stripped.startswith("Model compilation and execution time:"):
            summary["run_time_s"] = time_to_seconds(stripped.split(":", 1)[1].strip())
        elif stripped.startswith("Total time for analytic:"):
            summary["analytic_total_time_s"] = time_to_seconds(stripped.split(":", 1)[1].strip())
            in_table = False
        elif "Enabled dit_block_cache" in stripped:
            summary["dit_cache_effective"] = True
            m = re.search(
                r"replaced\s+(\d+)\s+blocks\s+in\s+range\s+\[(\d+),\s*(\d+)\)\s+out of\s+(\d+)",
                stripped,
            )
            if m:
                summary["replaced_blocks"] = int(m.group(1))
                summary["replaced_range_start"] = int(m.group(2))
                summary["replaced_range_end"] = int(m.group(3))
                summary["total_blocks"] = int(m.group(4))
        elif "DiT cache is disabled because" in stripped:
            summary["dit_cache_effective"] = False
            summary["dit_cache_disable_reason"] = stripped.split("because", 1)[1].strip().rstrip(".")
        if "Name" in stripped and "analytic total" in stripped and "# of Calls" in stripped:
            in_table = True
            table_lines = []
        elif in_table:
            table_lines.append(line)
    summary.setdefault("dit_cache_effective", bool(task.params.get("dit_cache", False)))
    tables = {"op_breakdown": _parse_table(table_lines)}
    if status != "success":
        summary["execution_error"] = _extract_execution_error(log, error)
    return ExperimentResult(
        task.sim_type,
        status,
        task.params,
        task.command,
        task.task_hash,
        task.label,
        summary,
        tables,
        warnings,
        infos,
        log,
        error,
    )


def parse_optimizer(task: ExperimentTask, log: str, status: str, error: str | None = None) -> ExperimentResult:
    summary = {}
    warnings = []
    infos = []
    table_lines = []
    in_table = False
    for line in log.splitlines():
        stripped = _strip_ansi(line.strip())
        if stripped.startswith("WARNING"):
            warnings.append(stripped)
        elif stripped.startswith("INFO"):
            infos.append(stripped)
        if stripped.startswith("Best Throughput:"):
            summary["best_throughput"] = float(stripped.split(":", 1)[1].strip().split()[0])
        elif stripped.startswith("TTFT:"):
            summary["best_ttft_ms"] = float(stripped.split(":", 1)[1].strip().split()[0])
        elif stripped.startswith("TPOT:"):
            summary["best_tpot_ms"] = float(stripped.split(":", 1)[1].strip().split()[0])
        elif stripped.startswith("TTFT Limits:"):
            raw = stripped.split(":", 1)[1].strip().split()[0]
            summary["ttft_limits_ms"] = None if raw == "None" else float(raw)
        elif stripped.startswith("TPOT Limits:"):
            raw = stripped.split(":", 1)[1].strip().split()[0]
            summary["tpot_limits_ms"] = None if raw == "None" else float(raw)
        if stripped.startswith("| Top | Throughput"):
            in_table = True
            table_lines = [line]
        elif in_table:
            table_lines.append(line)

    summary.setdefault("ttft_limits_ms", task.params.get("ttft_limits"))
    summary.setdefault("tpot_limits_ms", task.params.get("tpot_limits"))

    rows = _parse_table(table_lines)
    if rows:
        top1 = rows[0]
        summary["best_parallel"] = top1["parallel"]
        summary["best_batch_size"] = top1["batch_size"]
        summary["best_concurrency"] = top1["concurrency"]
    else:
        if status == "success":
            summary["no_result_reason"] = _optimizer_no_result_reason(task)
            status = "no_result"
        elif error:
            summary["execution_error"] = _extract_execution_error(log, error)

    return ExperimentResult(
        task.sim_type,
        status,
        task.params,
        task.command,
        task.task_hash,
        task.label,
        summary,
        {"top_configs": rows},
        warnings,
        infos,
        log,
        error,
    )


def parse_result(task: ExperimentTask, log: str, status: str, error: str | None = None) -> ExperimentResult:
    if task.sim_type == "text_generate":
        return parse_text_generate(task, log, status, error)
    if task.sim_type == "video_generate":
        return parse_video_generate(task, log, status, error)
    return parse_optimizer(task, log, status, error)
