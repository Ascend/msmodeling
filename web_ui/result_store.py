from __future__ import annotations

import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, TYPE_CHECKING

from .schemas import ExperimentResult, ExperimentTask

if TYPE_CHECKING:
    import os

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _resolve_log_path(path_str: str) -> Path:
    if not path_str:
        return Path("")
    normalized = path_str.replace("\\", "/")
    return Path(normalized)


def _extract_optimizer_top1_from_log(raw_log: str) -> dict[str, Any]:
    """Extract top 1 configuration from standard optimizer log (8-column table)."""
    if not raw_log:
        return {}
    raw_log = ANSI_RE.sub("", raw_log)
    pattern = re.compile(
        r"^\|\s*1\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*"
        r"\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(.*?)\s*\|\s*(\d+)\s*\|$",
        flags=re.MULTILINE,
    )
    m = pattern.search(raw_log)
    if not m:
        return {}
    return {
        "best_throughput": float(m.group(1)),
        "best_ttft_ms": float(m.group(2)),
        "best_tpot_ms": float(m.group(3)),
        "best_concurrency": int(m.group(4)),
        "best_parallel": m.group(6).strip(),
        "best_batch_size": int(m.group(7)),
    }


def _extract_pd_ratio_top1_from_log(raw_log: str) -> dict[str, Any]:
    """Extract top 1 configuration from PD Ratio mode log (15-column table)."""
    if not raw_log:
        return {}
    raw_log = ANSI_RE.sub("", raw_log)

    # Try to extract from the "Overall Best Configuration" section first
    # This section has explicit key-value pairs
    result = {}

    # Extract PD Ratio
    pd_match = re.search(r"PD Ratio:\s+([0-9.]+)", raw_log)
    if pd_match:
        result["pd_ratio"] = float(pd_match.group(1))

    # Extract Prefill QPS and TTFT
    p_qps_match = re.search(r"Prefill QPS:\s+([0-9.]+)", raw_log)
    if p_qps_match:
        result["p_qps"] = float(p_qps_match.group(1))
    ttft_match = re.search(r"TTFT:\s+([0-9.]+)\s*ms", raw_log)
    if ttft_match:
        result["best_ttft_ms"] = float(ttft_match.group(1))

    # Extract Decode QPS and TPOT
    d_qps_match = re.search(r"Decode QPS:\s+([0-9.]+)", raw_log)
    if d_qps_match:
        result["d_qps"] = float(d_qps_match.group(1))
    tpot_match = re.search(r"TPOT:\s+([0-9.]+)\s*ms", raw_log)
    if tpot_match:
        result["best_tpot_ms"] = float(tpot_match.group(1))

    # Extract Balanced QPS - this is the key metric for best_throughput
    balanced_match = re.search(r"Balanced[^:]*:\s+([0-9.]+)", raw_log)
    if balanced_match:
        balanced_qps = float(balanced_match.group(1))
        result["balanced_qps"] = balanced_qps

    # Extract Decode parallel info (used for best_parallel)
    d_parallel_match = re.search(r"D Parallel:\s*([^,\n]+)", raw_log)
    if d_parallel_match:
        result["best_parallel"] = d_parallel_match.group(1).strip()

    # Extract batch sizes and concurrency
    p_batch_match = re.search(r"P Batch:\s*(\d+)", raw_log)
    if p_batch_match:
        result["p_batch_size"] = int(p_batch_match.group(1))
    d_batch_match = re.search(r"D Batch:\s*(\d+)", raw_log)
    if d_batch_match:
        result["best_batch_size"] = int(d_batch_match.group(1))
    d_concurrency_match = re.search(r"D Concurrency:\s*(\d+)", raw_log)
    if d_concurrency_match:
        result["best_concurrency"] = int(d_concurrency_match.group(1))

    # If we have the key fields, return the result
    if result.get("best_throughput") is not None:
        return result

    # Fallback: try to parse from the PD Ratio table (15 columns)
    # Table header: | Top | PD Ratio | Balanced QPS | P QPS | D QPS | TTFT | TPOT | P Parallel | D Parallel | P Devices/Instance | D Devices/Instance | P Batch Size | D Batch Size | P Concurrency | D Concurrency |
    pattern = re.compile(
        r"^\|\s*1\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*"
        r"\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|"
        r"\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|$",
        flags=re.MULTILINE,
    )
    m = pattern.search(raw_log)
    if m:
        balanced_qps = float(m.group(2))
        return {
            "pd_ratio": float(m.group(1)),
            "balanced_qps": balanced_qps,
            "p_qps": float(m.group(3)),
            "d_qps": float(m.group(4)),
            "best_ttft_ms": float(m.group(5)),
            "best_tpot_ms": float(m.group(6)),
            "best_parallel": m.group(8).strip(),
            "prefill_devices_per_instance": int(m.group(9)),
            "decode_devices_per_instance": int(m.group(10)),
            "p_batch_size": int(m.group(11)),
            "best_batch_size": int(m.group(12)),
            "p_concurrency": int(m.group(13)),
            "best_concurrency": int(m.group(14)),
        }

    return {}


def _infer_optimizer_no_result_reason_from_params(params: dict[str, Any]) -> str:
    ttft = params.get("ttft_limits")
    tpot = params.get("tpot_limits")
    ttft_text = f"{ttft:g} ms" if isinstance(ttft, (int, float)) else "unlimited"
    tpot_text = f"{tpot:g} ms" if isinstance(tpot, (int, float)) else "unlimited"
    return (
        f"No valid deployment was found under the current limits "
        f"(TTFT={ttft_text}, TPOT={tpot_text})."
        "Try relaxing the latency limits, increasing num-devices, "
        "changing quantization, or reducing input/output length."
    )


def _enrich_optimizer_summary(
    summary: dict[str, Any],
    tables: dict[str, Any],
    raw_log: str = "",
    params: dict[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    if not isinstance(summary, dict):
        summary = {}
    if not isinstance(tables, dict):
        tables = {}
    params = params or {}

    top_rows = tables.get("top_configs") or []
    if isinstance(top_rows, list) and top_rows:
        top1 = top_rows[0]
        if isinstance(top1, dict):
            # For PD Ratio mode, fields have different names
            is_pd_ratio = top1.get("pd_ratio") is not None
            if is_pd_ratio:
                # PD Ratio mode uses different field names
                if top1.get("d_parallel") and summary.get("best_parallel") in (None, ""):
                    summary["best_parallel"] = top1.get("d_parallel")
                if top1.get("d_batch_size") and summary.get("best_batch_size") in (None, ""):
                    summary["best_batch_size"] = top1.get("d_batch_size")
                if top1.get("d_concurrency") and summary.get("best_concurrency") in (None, ""):
                    summary["best_concurrency"] = top1.get("d_concurrency")
                # throughput_token_s is used in PD Ratio mode
                if top1.get("throughput_token_s") and summary.get("best_throughput") in (None, ""):
                    summary["best_throughput"] = top1.get("throughput_token_s")
            else:
                # Standard mode
                if top1.get("parallel") and summary.get("best_parallel") in (None, ""):
                    summary["best_parallel"] = top1.get("parallel")
                if top1.get("batch_size") and summary.get("best_batch_size") in (None, ""):
                    summary["best_batch_size"] = top1.get("batch_size")
                if top1.get("concurrency") and summary.get("best_concurrency") in (None, ""):
                    summary["best_concurrency"] = top1.get("concurrency")
                if top1.get("throughput_token_s") and summary.get("best_throughput") in (None, ""):
                    summary["best_throughput"] = top1.get("throughput_token_s")
            # Common fields
            if top1.get("ttft_ms") and summary.get("best_ttft_ms") in (None, ""):
                summary["best_ttft_ms"] = top1.get("ttft_ms")
            if top1.get("tpot_ms") and summary.get("best_tpot_ms") in (None, ""):
                summary["best_tpot_ms"] = top1.get("tpot_ms")

    if (
        any(summary.get(k) in (None, "") for k in ["best_parallel", "best_batch_size", "best_concurrency"])
        or not top_rows
    ):
        # Detect if this is PD Ratio mode
        is_pd_ratio_mode = (
            params.get("enable_optimize_prefill_decode_ratio", False)
            or "PD Ratio" in raw_log
            or summary.get("pd_ratio") is not None
            or (top_rows and top_rows[0].get("pd_ratio") is not None)
        )
        # Use the appropriate extraction function based on mode
        if is_pd_ratio_mode:
            extracted = _extract_pd_ratio_top1_from_log(raw_log)
        else:
            extracted = _extract_optimizer_top1_from_log(raw_log)
        summary.update({k: v for k, v in extracted.items() if summary.get(k) in (None, "")})

    summary.setdefault("ttft_limits_ms", params.get("ttft_limits"))
    summary.setdefault("tpot_limits_ms", params.get("tpot_limits"))

    has_result = summary.get("best_throughput") not in (None, "")
    if error and not summary.get("execution_error"):
        summary["execution_error"] = error
    if not has_result and not summary.get("no_result_reason") and not summary.get("execution_error"):
        summary["no_result_reason"] = _infer_optimizer_no_result_reason_from_params(params)

    return summary


class ResultStore:
    def __init__(self, root: str | os.PathLike | None = None):
        self.root = Path(root or ".msmodeling_ui")
        self.root.mkdir(parents=True, exist_ok=True)
        self.logs_dir = self.root / "logs"
        self.logs_dir.mkdir(exist_ok=True)
        self.db_path = self.root / "results.sqlite3"
        self._init_db()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    task_hash TEXT PRIMARY KEY,
                    sim_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    label TEXT NOT NULL,
                    params_json TEXT NOT NULL,
                    summary_json TEXT NOT NULL,
                    tables_json TEXT NOT NULL,
                    warnings_json TEXT NOT NULL,
                    infos_json TEXT NOT NULL,
                    log_path TEXT NOT NULL,
                    error TEXT,
                    created_at REAL NOT NULL
                )
                """
            )
            conn.commit()

    def get_cached_result(self, task: ExperimentTask) -> ExperimentResult | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT sim_type,status,label,params_json,summary_json,tables_json,"
                "warnings_json,infos_json,log_path,error FROM runs WHERE task_hash=?",
                (task.task_hash,),
            ).fetchone()
        if not row:
            return None
        log_file = _resolve_log_path(row[8])
        raw_log = log_file.read_text(encoding="utf-8") if log_file.exists() else ""
        summary = json.loads(row[4])
        tables = json.loads(row[5])
        if row[0] == "throughput_optimizer":
            summary = _enrich_optimizer_summary(summary, tables, raw_log, json.loads(row[3]), row[9])
        return ExperimentResult(
            sim_type=row[0],
            status=row[1],
            params=json.loads(row[3]),
            command=task.command,
            task_hash=task.task_hash,
            label=row[2],
            summary=summary,
            tables=tables,
            warnings=json.loads(row[6]),
            infos=json.loads(row[7]),
            raw_log=raw_log,
            error=row[9],
            source="cache",
        )

    def save_result(self, result: ExperimentResult):
        log_path = self.logs_dir / f"{result.task_hash}.log"
        log_path.write_text(result.raw_log or "", encoding="utf-8")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO runs(task_hash, sim_type, status, label, params_json, summary_json,
                    tables_json, warnings_json, infos_json, log_path, error, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result.task_hash,
                    result.sim_type,
                    result.status,
                    result.label,
                    json.dumps(result.params, ensure_ascii=False),
                    json.dumps(result.summary, ensure_ascii=False),
                    json.dumps(result.tables, ensure_ascii=False),
                    json.dumps(result.warnings, ensure_ascii=False),
                    json.dumps(result.infos, ensure_ascii=False),
                    str(log_path),
                    result.error,
                    time.time(),
                ),
            )
            conn.commit()

    def query_rows(self, sim_type: str | None = None) -> list[dict[str, Any]]:
        query = (
            "SELECT sim_type,status,label,params_json,summary_json,warnings_json,"
            "infos_json,created_at,task_hash,error FROM runs"
        )
        args: tuple[Any, ...] = ()
        if sim_type:
            query += " WHERE sim_type=?"
            args = (sim_type,)
        query += " ORDER BY created_at DESC"
        rows = []
        with self._connect() as conn:
            for row in conn.execute(query, args).fetchall():
                params = json.loads(row[3])
                summary = json.loads(row[4])
                top_configs: list[dict[str, Any]] = []
                if row[0] == "throughput_optimizer":
                    try:
                        raw_row = conn.execute(
                            "SELECT tables_json, log_path FROM runs WHERE task_hash=?",
                            (row[8],),
                        ).fetchone()
                        tables = json.loads(raw_row[0]) if raw_row and raw_row[0] else {}
                        log_file = _resolve_log_path(raw_row[1]) if raw_row and raw_row[1] else Path("")
                        raw_log = log_file.read_text(encoding="utf-8") if log_file.exists() else ""
                    except Exception:
                        tables = {}
                        raw_log = ""
                    top_configs = tables.get("top_configs") or []
                    summary = _enrich_optimizer_summary(summary, tables, raw_log, json.loads(row[3]), row[9])
                rows.append(
                    {
                        "sim_type": row[0],
                        "status": row[1],
                        "label": row[2],
                        **params,
                        **summary,
                        "top_configs": top_configs,
                        "warning_count": len(json.loads(row[5])),
                        "info_count": len(json.loads(row[6])),
                        "created_at": row[7],
                        "task_hash": row[8],
                    }
                )
        return rows
