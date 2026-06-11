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
    """Parse standard optimizer table row (non-PD-Ratio mode)."""
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


def _parse_pd_ratio_row(cells: list[str]) -> dict[str, Any] | None:
    """Parse PD Ratio mode table row."""
    # Expected columns: Top, PD Ratio, Balanced QPS, P QPS, D QPS, TTFT, TPOT,
    #                    P Parallel, D Parallel, P Devices/Instance, D Devices/Instance,
    #                    P Batch Size, D Batch Size, P Concurrency, D Concurrency
    if len(cells) < 15 or not cells[0].isdigit():
        return None
    try:
        return {
            "rank": int(cells[0]),
            "pd_ratio": float(cells[1]),
            "balanced_qps": float(cells[2]),
            "p_qps": float(cells[3]),
            "d_qps": float(cells[4]),
            "ttft_ms": float(cells[5]),
            "tpot_ms": float(cells[6]),
            "p_parallel": cells[7].strip(),
            "d_parallel": cells[8].strip(),
            "prefill_devices_per_instance": int(cells[9]),
            "decode_devices_per_instance": int(cells[10]),
            "p_batch_size": int(cells[11]),
            "d_batch_size": int(cells[12]),
            "p_concurrency": int(cells[13]),
            "d_concurrency": int(cells[14]),
            "throughput_token_s": float(cells[2]),  # Use balanced_qps as throughput
        }
    except (ValueError, IndexError):
        return None


def _parse_disagg_row(cells: list[str], is_prefill: bool) -> dict[str, Any] | None:
    """Parse PD Disaggregated mode table row.

    Prefill columns: Top, Throughput, QPS, TTFT, concurrency, num_devices, parallel, batch_size
    Decode columns: Top, Throughput, QPS, TPOT, concurrency, num_devices, parallel, batch_size
    """
    if len(cells) < 8 or not cells[0].isdigit():
        return None
    try:
        result = {
            "rank": int(cells[0]),
            "throughput_token_s": float(cells[1]),
            "qps": float(cells[2]),
            "concurrency": int(cells[4]),
            "num_devices": int(cells[5]),
            "parallel": " | ".join(part for part in cells[6:-1] if part),
            "batch_size": int(cells[-1]),
        }
        if is_prefill:
            result["ttft_ms"] = float(cells[3])
        else:
            result["tpot_ms"] = float(cells[3])
        return result
    except (ValueError, IndexError):
        return None


def _extract_parallel_config(parallel_str: str) -> dict[str, Any]:
    """Extract TP, PP, DP from parallel string like 'TP=2 | PP=1 | DP=2'."""
    result = {"parallel": parallel_str, "tp": None, "pp": None, "dp": None}
    for part in parallel_str.split("|"):
        part = part.strip()
        if "=" in part:
            key, val = part.split("=", 1)
            key = key.strip().lower()
            val = val.strip()
            try:
                if key == "tp":
                    result["tp"] = int(val)
                elif key == "pp":
                    result["pp"] = int(val)
                elif key == "dp":
                    result["dp"] = int(val)
            except ValueError:
                pass
    return result


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

    # Detect mode by checking command parameters
    is_pd_ratio_mode = "--enable-optimize-prefill-decode-ratio" in task.command
    is_disagg_mode = "--disagg" in task.command

    for line in log.splitlines():
        stripped = _strip_ansi(line.strip())
        if stripped.startswith("WARNING"):
            warnings.append(stripped)
        elif stripped.startswith("INFO"):
            infos.append(stripped)

        # Parse PD Ratio mode - Overall Best Configuration section
        if is_pd_ratio_mode:
            if stripped.strip().startswith("PD Ratio:"):
                # PD Ratio: 0.28 (P Instance:D Instance)
                match = re.search(r"PD Ratio:\s+([0-9.]+)", stripped)
                if match:
                    summary["pd_ratio"] = float(match.group(1))
            elif stripped.strip().startswith("Prefill QPS:"):
                # Prefill QPS: 82.11 req/s  (TTFT: 998.63 ms, Parallel: TP=2 | PP=1 | DP=2, Batch: 41, Concurrency: 82)
                match = re.search(r"Prefill QPS:\s+([0-9.]+)", stripped)
                if match:
                    summary["p_qps"] = float(match.group(1))
                ttft_match = re.search(r"TTFT:\s+([0-9.]+)\s*ms", stripped)
                if ttft_match:
                    summary["best_ttft_ms"] = float(ttft_match.group(1))
                parallel_match = re.search(r"Parallel:\s*(.+?),\s*Batch:", stripped)
                if parallel_match:
                    summary["p_parallel"] = parallel_match.group(1).strip()
                batch_match = re.search(r"Batch:\s+(\d+),\s*Concurrency:", stripped)
                if batch_match:
                    summary["p_batch_size"] = int(batch_match.group(1))
                concurrency_match = re.search(r"Concurrency:\s+(\d+)", stripped)
                if concurrency_match:
                    summary["p_concurrency"] = int(concurrency_match.group(1))
            elif stripped.strip().startswith("Decode QPS:"):
                # Decode QPS:  23.36 req/s  (TPOT: 27.54 ms, Parallel: TP=4 | PP=1 | DP=1, Batch: 128, Concurrency: 128)
                match = re.search(r"Decode QPS:\s+([0-9.]+)", stripped)
                if match:
                    summary["d_qps"] = float(match.group(1))
                tpot_match = re.search(r"TPOT:\s+([0-9.]+)\s*ms", stripped)
                if tpot_match:
                    summary["best_tpot_ms"] = float(tpot_match.group(1))
                parallel_match = re.search(r"Parallel:\s*(.+?),\s*Batch:", stripped)
                if parallel_match:
                    summary["d_parallel"] = parallel_match.group(1).strip()
                batch_match = re.search(r"Batch:\s+(\d+),\s*Concurrency:", stripped)
                if batch_match:
                    summary["d_batch_size"] = int(batch_match.group(1))
                concurrency_match = re.search(r"Concurrency:\s+(\d+)", stripped)
                if concurrency_match:
                    summary["d_concurrency"] = int(concurrency_match.group(1))
            elif stripped.strip().startswith("Balanced QPS:") or stripped.strip().startswith("Balanced:"):
                # Some formats might have Balanced QPS on its own line
                match = re.search(r"Balanced[^:]*:\s+([0-9.]+)", stripped)
                if match:
                    summary["balanced_qps"] = float(match.group(1))
                    # Use balanced_qps as best_throughput for compatibility
                    summary["best_throughput"] = float(match.group(1))

            # Detect PD Ratio table
            if "Top" in stripped and "PD Ratio" in stripped and "Balanced QPS" in stripped:
                in_table = True
                table_lines = [line]
            elif in_table:
                table_lines.append(line)

        # Parse standard mode (non-PD-Ratio)
        else:
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

    # Parse table rows
    rows = []
    is_prefill_table = False  # Track if current table is prefill (for disagg mode)
    for line in table_lines:
        stripped = _strip_ansi(line.rstrip())
        if not stripped or stripped.startswith(("-", "+")):
            continue
        # Detect table type for disagg mode
        if is_disagg_mode and "TTFT" in stripped:
            is_prefill_table = True
        elif is_disagg_mode and "TPOT" in stripped:
            is_prefill_table = False
        if stripped.startswith("|"):
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if is_pd_ratio_mode:
                row = _parse_pd_ratio_row(cells)
            elif is_disagg_mode:
                row = _parse_disagg_row(cells, is_prefill_table)
            else:
                row = _parse_optimizer_row(cells)
            if row is not None:
                rows.append(row)

    if rows:
        top1 = rows[0]
        if is_pd_ratio_mode:
            # For PD Ratio mode, use d_parallel for best_parallel
            summary["best_parallel"] = top1.get("d_parallel", top1.get("p_parallel", ""))
            summary["best_batch_size"] = top1.get("d_batch_size", top1.get("p_batch_size", 0))
            summary["best_concurrency"] = top1.get("d_concurrency", top1.get("p_concurrency", 0))
            # Ensure PD Ratio specific fields are set
            summary.setdefault("balanced_qps", top1.get("balanced_qps"))
            summary.setdefault("p_qps", top1.get("p_qps"))
            summary.setdefault("d_qps", top1.get("d_qps"))
            summary.setdefault("pd_ratio", top1.get("pd_ratio"))
            summary.setdefault("prefill_devices_per_instance", top1.get("prefill_devices_per_instance"))
            summary.setdefault("decode_devices_per_instance", top1.get("decode_devices_per_instance"))
        elif is_disagg_mode:
            # For PD Disaggregated mode
            summary["best_parallel"] = top1.get("parallel", "")
            summary["best_batch_size"] = top1.get("batch_size", 0)
            summary["best_concurrency"] = top1.get("concurrency", 0)
            summary.setdefault("qps", top1.get("qps"))
            summary.setdefault("best_ttft_ms", top1.get("ttft_ms"))
            summary.setdefault("best_tpot_ms", top1.get("tpot_ms"))
        else:
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
