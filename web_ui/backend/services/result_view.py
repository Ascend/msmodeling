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

"""Result view builder.

Assembles the ``result`` view envelope from a job's ``result_records[]`` per
``contracts/result-rendering.md``. The backend builds this at fetch time; the module
result component reads from this assembled envelope (not the raw ``records[]``).

* text_generate / video_generate: 1 record → ``result`` ≈ that record's summary+tables
* throughput_optimizer: N records → aggregated ``result`` with best_config,
  sweep_rows, cross_hardware, and mode-specific disagg/pd_ratio views
"""

from __future__ import annotations

import logging
import math
from typing import Any


logger = logging.getLogger(__name__)


def assemble_result_envelope(
    module_id: str,
    records: list[dict[str, Any]],
    form_schema_version: str | None = None,
    input_config: dict[str, Any] | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Assemble the ``result`` envelope for a job from its result records.

    This is called by ``GET /api/jobs/{id}/result`` to build the backend
    ``result`` field that the module result component renders.

    Args:
        module_id: The module identifier
        records: Raw result records from the database (with rank, config, summary, tables)
        form_schema_version: Pinned form schema version (for deterministic reopen)
        input_config: Original job params (optional, for context)

    Returns:
        The assembled ``result`` envelope per ``contracts/result-rendering.md``.
    """
    if not records:
        return {
            "mode": "empty",
            "input_config": input_config or {},
            "best_config": None,
            "sweep_rows": [],
            "cross_hardware": [],
        }

    if module_id == "text_generate":
        return _assemble_text_result(records, input_config, job_id)
    elif module_id == "video_generate":
        return _assemble_video_result(records, input_config, job_id)
    elif module_id in ("throughput_optimizer", "throughput_optimizer_benchmark"):
        return _assemble_throughput_result(records, input_config, job_id)
    else:
        logger.warning(f"Unknown module_id {module_id}, returning empty envelope")
        return {
            "mode": "empty",
            "input_config": input_config or {},
            "best_config": None,
            "sweep_rows": [],
            "cross_hardware": [],
        }


def _single_text_envelope(record: dict[str, Any]) -> dict[str, Any]:
    """The metrics envelope for one text-gen record (its ``tables``)."""
    tables = record.get("tables", {})
    return {
        "batch_size": tables.get("batch_size"),
        "run_time_s": tables.get("run_time_s"),
        "execution_time_s": tables.get("execution_time_s"),
        "tps_per_model": tables.get("tps_per_model"),
        "memory_gb": tables.get("memory_gb"),
        "breakdowns_raw": tables.get("breakdowns_raw"),
        "breakdowns_percent": tables.get("breakdowns_percent"),
        "perf_model_name": tables.get("perf_model_name"),
        "op_breakdown": tables.get("op_breakdown", []),
        "dump_input_shapes": tables.get("dump_input_shapes", False),
        "dump_op_bound_results": tables.get("dump_op_bound_results", False),
    }


def _assemble_text_result(
    records: list[dict[str, Any]],
    input_config: dict[str, Any] | None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Text generate: 1 record → flat envelope; N records (Phase D2 multi-case)
    → ``{multi_case: true, cases: [...]}`` with one entry per case.
    """
    if not records:
        return {}

    if len(records) == 1:
        from services.trace_store import trace_path

        return {
            "mode": "text_generation",
            "input_config": input_config or {},
            "case_hash": records[0].get("case_hash"),
            **_single_text_envelope(records[0]),
            "chrome_trace": {"available": trace_path(job_id, records[0]["seq"]).exists() if job_id else False},
        }

    # Multi-case: one entry per case (its config + summary + metrics envelope).
    from services.trace_store import trace_path

    return {
        "mode": "text_generation",
        "multi_case": True,
        "input_config": input_config or {},
        "cases": [
            {
                "config": r.get("config", {}),
                "summary": r.get("summary", {}),
                "case_hash": r.get("case_hash"),
                **_single_text_envelope(r),
                "chrome_trace": {"available": trace_path(job_id, r["seq"]).exists() if job_id else False},
            }
            for r in records
        ],
    }


def _single_video_envelope(record: dict[str, Any]) -> dict[str, Any]:
    """The metrics envelope for one video-gen record (its ``tables``)."""
    tables = record.get("tables", {})
    return {
        "execution_time_s": tables.get("execution_time_s", {}),
        "breakdowns": tables.get("breakdowns", {}),
        "table_rows": tables.get("table_rows", []),
        "op_breakdown": tables.get("op_breakdown", []),
    }


def _assemble_video_result(
    records: list[dict[str, Any]],
    input_config: dict[str, Any] | None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Video generate: 1 record → flat envelope; N records (Phase D2 multi-case)
    → ``{multi_case: true, cases: [...]}``.
    """
    if not records:
        return {}

    from services.trace_store import trace_path

    if len(records) == 1:
        return {
            "mode": "video_generation",
            "input_config": input_config or {},
            "case_hash": records[0].get("case_hash"),
            **_single_video_envelope(records[0]),
            "chrome_trace": {"available": trace_path(job_id, records[0]["seq"]).exists() if job_id else False},
        }

    return {
        "mode": "video_generation",
        "multi_case": True,
        "input_config": input_config or {},
        "cases": [
            {
                "config": r.get("config", {}),
                "summary": r.get("summary", {}),
                "case_hash": r.get("case_hash"),
                **_single_video_envelope(r),
                "chrome_trace": {"available": trace_path(job_id, r["seq"]).exists() if job_id else False},
            }
            for r in records
        ],
    }


def _sweep_row_from_record(record: dict[str, Any]) -> dict[str, Any]:
    """Extract a flat sweep row from a result record (for disagg tables)."""
    summary = record.get("summary", {})
    config = record.get("config", {})
    return {
        "rank": record.get("rank"),
        "device": config.get("device"),
        "parallel": config.get("parallel"),
        "concurrency": config.get("concurrency"),
        "throughput_token_s": summary.get("throughput_token_s"),
        "qps": summary.get("qps"),
        "ttft_ms": summary.get("ttft_ms"),
        "tpot_ms": summary.get("tpot_ms"),
        "input_throughput_token_s": summary.get("input_throughput_token_s"),
        "per_card_input_throughput_token_s": summary.get("per_card_input_throughput_token_s"),
        "per_card_output_throughput_token_s": summary.get("per_card_output_throughput_token_s"),
        "num_devices": config.get("num_devices"),
        "batch_size": config.get("batch_size"),
        # Memory columns
        "model_weight_size_gb": summary.get("model_weight_size_gb"),
        "kv_cache_size_gb": summary.get("kv_cache_size_gb"),
        "model_activation_size_gb": summary.get("model_activation_size_gb"),
        "device_memory_available_gb": summary.get("device_memory_available_gb"),
    }


def _reassign_local_ranks(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Re-rank records WITHIN a single case (copy, don't mutate originals).

    The persisted rank is the global rank across all cases (computed by
    job_runner; the single source of truth). When rendering multi_case, each
    case's best_config lookup needs rank=1 WITHIN the case, so a per-case rank is
    recomputed on demand (local copies only; the persisted value is never
    mutated). Uses assign_optimizer_ranks so PD-ratio (balanced_qps) vs
    aggregation (throughput_token_s) is handled correctly.
    """
    import copy
    from services.ranking import assign_optimizer_ranks

    local = [copy.copy(r) for r in records]
    ranks = assign_optimizer_ranks(local)
    for r, rank in zip(local, ranks):
        r["rank"] = rank
    return local


# ---------------------------------------------------------------------------
# Read-time Top-N alignment with the CLI (SLO filter + per-parallelism dedup).
#
# The CLI print path (serving_cast/service/optimizer_summary.py) collapses the
# raw OptimizerSummary DataFrame before printing "Top N": it drops rows that
# violate the TTFT/TPOT limits, then keeps only the best row per parallelism
# (agg/disagg: per ``parallel``; pd_ratio: per ``(parallel_p, parallel_d)`` and
# again per ``balanced_qps`` rounded to 2 dp). The web_ui capture path stores
# the RAW rows; these helpers replay that same collapse at fetch time so the
# rendered envelope matches the CLI's Top-N (and best_config never picks an
# SLO-violating row). Pure Python on purpose — result_view stays pandas-free.
# ---------------------------------------------------------------------------


def _record_limits(record: dict[str, Any]) -> tuple[float, float]:
    """``(tpot_limit, ttft_limit)`` for this record; ``+inf`` when unset/missing.

    Mirrors the CLI's ``data_config.tpot_limits or float("inf")``. Limits are
    carried per-record in ``config`` (set in ``_run_throughput_sweep`` case_tag),
    so multi-case jobs with different limits per case filter correctly.
    """
    cfg = record.get("config", {})
    tpot = cfg.get("tpot_limits")
    ttft = cfg.get("ttft_limits")
    # Mirror the CLI's ``value or float("inf")``: 0/False/missing are "unset".
    tpot_limit = float(tpot) if isinstance(tpot, (int, float)) and tpot else float("inf")
    ttft_limit = float(ttft) if isinstance(ttft, (int, float)) and ttft else float("inf")
    return tpot_limit, ttft_limit


def _latency_or_inf(value: Any) -> float:
    """Mirrors ``pd.to_numeric(...).fillna(inf)``: missing/NaN/unparseable -> +inf.

    A ``+inf`` latency only fails the ``<= limit`` mask when a real limit is set,
    exactly like the CLI.
    """
    try:
        f = float(value)
    except (TypeError, ValueError):
        return float("inf")
    return f if not math.isnan(f) else float("inf")


def _passes_slo(record: dict[str, Any]) -> bool:
    """SLO mask — mode-aware for disagg:

    * **aggregation**: check BOTH limits (original CLI semantics).
    * **disagg_prefill**: only ``ttft_limit`` applies (tpot is not meaningful for
      the prefill phase; the row's ``tpot_ms`` is ``None`` which would otherwise
      coerce to ``+inf`` and fail a finite ``tpot_limit``).
    * **disagg_decode**: only ``tpot_limit`` applies (ttft is not meaningful for
      the decode phase; symmetric to prefill).

    ``pd_ratio`` reuses the same fields (``ttft_ms`` == CLI ``ttft_p``,
    ``tpot_ms`` == CLI ``tpot_d``) and always checks both.
    """
    tpot_limit, ttft_limit = _record_limits(record)
    summary = record.get("summary", {})
    mode = summary.get("mode") or record.get("config", {}).get("mode") or "aggregation"
    if mode == "disagg_prefill":
        return _latency_or_inf(summary.get("ttft_ms")) <= ttft_limit
    if mode == "disagg_decode":
        return _latency_or_inf(summary.get("tpot_ms")) <= tpot_limit
    return (
        _latency_or_inf(summary.get("tpot_ms")) <= tpot_limit and _latency_or_inf(summary.get("ttft_ms")) <= ttft_limit
    )


def _num_summary(record: dict[str, Any], key: str) -> float:
    """Numeric ``summary[key]`` for sorting; missing/NaN/unparseable -> -inf
    (sorts last under ``reverse=True``).
    """
    try:
        value = float(record.get("summary", {}).get(key))
        return value if not math.isnan(value) else float("-inf")
    except (TypeError, ValueError):
        return float("-inf")


def _topn_agg_disagg(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """SLO mask, then keep the best (max ``token/s``) row per ``(parallel, phase)``.

    Prefill and decode rows are treated as distinct groups even when their
    ``parallel`` strings match (a disagg run often explores the same parallel
    strategy in both phases) — collapsing across phases would silently drop
    one phase's best row and make the result panel disagree with the runner's
    per-phase log output.
    """
    passed = [r for r in records if _passes_slo(r)]
    passed.sort(key=lambda r: _num_summary(r, "throughput_token_s"), reverse=True)
    seen: set[tuple[Any, str]] = set()
    survivors: list[dict[str, Any]] = []
    for record in passed:
        parallel = record.get("config", {}).get("parallel")
        mode = record.get("summary", {}).get("mode") or record.get("config", {}).get("mode") or "aggregation"
        key = (parallel, mode)
        if key in seen:
            continue
        seen.add(key)
        survivors.append(record)
    # Already sorted desc by throughput; the CLI sorts once more (no-op here),
    # kept explicit for parity clarity.
    survivors.sort(key=lambda r: _num_summary(r, "throughput_token_s"), reverse=True)
    return survivors


def _pd_rank_key(record: dict[str, Any]) -> tuple:
    """Sort key mirroring ``serving_cast.utils.PD_RATIO_RANK_KEYS``.

    Pandas ``sort_values`` places NaN last for BOTH asc and desc. We collapse to
    a single ascending tuple sort: desc numeric keys are negated (largest real
    value -> most negative -> first) with missing sent to ``+inf`` (last); asc
    numeric keys send missing to ``+inf`` (last). Field path mapping:
    ``ttft_p``->``summary.ttft_ms``, ``tpot_d``->``summary.tpot_ms``,
    ``batch_size_p/d``->``config.p/d_batch_size``,
    ``concurrency_p/d``->``config.p/d_concurrency``.
    """
    summary = record.get("summary", {})
    config = record.get("config", {})

    def _num(mapping: dict[str, Any], key: str) -> float | None:
        try:
            f = float(mapping.get(key))
        except (TypeError, ValueError):
            return None
        return f if not math.isnan(f) else None

    def _desc(mapping: dict[str, Any], key: str) -> float:
        f = _num(mapping, key)
        return -f if f is not None else float("inf")

    def _asc_num(mapping: dict[str, Any], key: str) -> float:
        f = _num(mapping, key)
        return f if f is not None else float("inf")

    def _asc_str(mapping: dict[str, Any], key: str) -> str:
        v = mapping.get(key)
        return "" if v is None else str(v)

    return (
        _desc(summary, "balanced_qps"),
        _desc(summary, "d_qps"),
        _desc(summary, "p_qps"),
        _asc_num(summary, "ttft_ms"),  # ttft_p
        _asc_num(summary, "tpot_ms"),  # tpot_d
        _desc(config, "d_batch_size"),  # batch_size_d
        _desc(config, "p_batch_size"),  # batch_size_p
        _desc(config, "d_concurrency"),  # concurrency_d
        _desc(config, "p_concurrency"),  # concurrency_p
        _asc_str(config, "parallel_p"),
        _asc_str(config, "parallel_d"),
    )


def _topn_pd_ratio(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Replay ``OptimizerSummary._prepare_pd_ratio_results``: SLO mask, best per
    ``(parallel_p, parallel_d)`` by ``PD_RATIO_RANK_KEYS``, then best per
    ``balanced_qps`` rounded to 2 dp.
    """
    passed = [r for r in records if _passes_slo(r)]
    passed.sort(key=_pd_rank_key)

    def _rounded_balanced_qps(record: dict[str, Any]) -> float | None:
        try:
            return round(float(record.get("summary", {}).get("balanced_qps")), 2)
        except (TypeError, ValueError):
            return None

    seen_pairs: set[tuple[Any, Any]] = set()
    stage1: list[dict[str, Any]] = []
    for record in passed:
        key = (
            record.get("config", {}).get("parallel_p"),
            record.get("config", {}).get("parallel_d"),
        )
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        stage1.append(record)

    stage1.sort(key=_pd_rank_key)
    seen_q: set[Any] = set()
    survivors: list[dict[str, Any]] = []
    for record in stage1:
        q = _rounded_balanced_qps(record)
        if q in seen_q:
            continue
        seen_q.add(q)
        survivors.append(record)
    return survivors


def _apply_cli_topn_filter(records: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    """Dispatch the CLI Top-N collapse by optimizer mode. Safe on empty input."""
    if not records:
        return []
    if mode == "pd_ratio":
        return _topn_pd_ratio(records)
    return _topn_agg_disagg(records)


def _assemble_throughput_result(
    records: list[dict[str, Any]],
    input_config: dict[str, Any] | None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Throughput optimizer has N records → aggregated envelope with ranking.

    Builds:
    * best_config: the rank=1 record (max throughput_token_s; ties → lower ttft,
      then lower tpot, then device)
    * sweep_rows: all records (rank-ordered for display)
    * cross_hardware: best_config per selected device
    * disagg_prefill/disagg_decode/pd_ratio_rows: mode-specific fields
    """
    if not records:
        return {
            "mode": "aggregation",
            "input_config": input_config or {},
            "best_config": None,
            "sweep_rows": [],
            "cross_hardware": [],
            "pd_pairs": None,
        }

    # Multi-case grouping (Phase D2): if records span multiple cases (distinct
    # tpot/ttft/quant combos), build a per-case envelope. Each case's records
    # are fed back through this assembler (so each case keeps its own
    # best_config / sweep_rows / cross_hardware).
    _CASE_FIELDS = (
        "device",
        "tpot_limits",
        "ttft_limits",
        "quantize_linear_action",
        "quantize_attention_action",
    )
    case_keys = [tuple(r.get("config", {}).get(k) for k in _CASE_FIELDS) for r in records]
    if len(set(case_keys)) > 1:
        from services.trace_store import trace_path

        groups: dict[tuple, list[dict[str, Any]]] = {}
        order: list[tuple] = []
        for r, key in zip(records, case_keys):
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(r)

        def _case_input_config(case_records: list[dict[str, Any]]) -> dict[str, Any]:
            """Extract input_config from a case's records for PD-pair calculation.

            Multi-case grouping passes None for input_config, but each case's
            records carry their own input/output lengths in config. Extract them
            so PD-pair calculation works for each case independently.

            Falls back to the top-level input_config if the case records don't
            have the lengths (which can happen when the CLI doesn't write them
            to the config).
            """
            first = case_records[0] if case_records else {}
            config = first.get("config", {})
            # Try to extract from config first (CLI may write them)
            in_len = config.get("input_length") or config.get("input-length")
            out_len = config.get("output_length") or config.get("output-length")
            # Fall back to top-level input_config if not in config
            if in_len is None or out_len is None:
                top_ic = input_config or {}
                in_len = in_len if in_len is not None else top_ic.get("input-length")
                out_len = out_len if out_len is not None else top_ic.get("output-length")
            return {
                "input-length": in_len,
                "output-length": out_len,
            }

        return {
            "mode": "throughput_optimization",
            "multi_case": True,
            "input_config": input_config or {},
            "cases": [
                {
                    "case_config": dict(zip(_CASE_FIELDS, key)),
                    # case_hash for per-case log lookup (all recs in a case share it).
                    "case_hash": groups[key][0].get("case_hash"),
                    # Pass the RAW case records: the single-case path below owns
                    # both the Top-N SLO/dedup filter and the per-case local
                    # re-rank (so rank=1 = best for THIS case among survivors;
                    # job_runner's rank is global across all cases).
                    # Extract per-case input_config so PD-pair calculation works.
                    **_assemble_throughput_result(groups[key], _case_input_config(groups[key]), job_id),
                    "chrome_trace": {
                        "available": (
                            any(trace_path(job_id, r["seq"]).exists() for r in groups[key] if "seq" in r)
                            if job_id
                            else False
                        )
                    },
                }
                for key in order
            ],
        }

    # Infer mode from the first record (or default to aggregation)
    # Runner stores mode in summary.mode or config.mode, not tables.mode
    first_record = records[0]
    mode = first_record.get("summary", {}).get("mode") or first_record.get("config", {}).get("mode", "aggregation")

    # Read-time Top-N alignment with the CLI: collapse raw records to the CLI's
    # SLO-filtered, per-parallelism-deduped Top-N, then re-rank the survivors so
    # rank=1 is the true best among them. The persisted/global rank was computed
    # over the RAW set and may point at a since-filtered-out row, so a local
    # re-rank over the survivors is required for best_config (rank=1) to stay
    # correct. See _apply_cli_topn_filter / _reassign_local_ranks.
    records = _apply_cli_topn_filter(records, mode)
    if not records:
        return {
            "mode": mode,
            "input_config": input_config or {},
            "best_config": None,
            "sweep_rows": [],
            "cross_hardware": [],
            "disagg_prefill": None,
            "disagg_decode": None,
            "pd_ratio_rows": None,
            "pd_pairs": None,
            "note": "No configurations satisfy the current TTFT/TPOT filters.",
        }
    records = _reassign_local_ranks(records)

    # best_config = the rank=1 survivor. ``records`` was re-ranked just above, so
    # exactly one record carries rank=1; next() always finds it (no default —
    # fail fast if that invariant ever breaks, rather than silently emit a None
    # best_config / empty cross_hardware).
    best_config_record = next(r for r in records if r.get("rank") == 1)
    summary = best_config_record.get("summary", {})
    config = best_config_record.get("config", {})
    if summary.get("mode") == "pd_ratio":
        # PD-ratio records carry balanced_qps / pd_ratio / p_qps / d_qps and
        # parallel_p / parallel_d — NOT throughput_token_s / parallel.
        best_config = {
            "device": config.get("device"),
            "parallel_p": config.get("parallel_p"),
            "parallel_d": config.get("parallel_d"),
            "p_devices_per_instance": config.get("p_devices_per_instance"),
            "d_devices_per_instance": config.get("d_devices_per_instance"),
            "p_batch_size": config.get("p_batch_size"),
            "d_batch_size": config.get("d_batch_size"),
            "balanced_qps": summary.get("balanced_qps"),
            "pd_ratio": summary.get("pd_ratio"),
            "p_qps": summary.get("p_qps"),
            "d_qps": summary.get("d_qps"),
            "ttft_ms": summary.get("ttft_ms"),
            "tpot_ms": summary.get("tpot_ms"),
        }
    else:
        out = summary.get("throughput_token_s")
        inp = summary.get("input_throughput_token_s")
        best_config = {
            "device": config.get("device"),
            "parallel": config.get("parallel"),
            "concurrency": config.get("concurrency"),
            "throughput_token_s": out,
            "qps": summary.get("qps"),
            "ttft_ms": summary.get("ttft_ms"),
            "tpot_ms": summary.get("tpot_ms"),
            "input_throughput_token_s": inp,
            "e2e_throughput_token_s": out + inp if out is not None and inp is not None else None,
            "per_card_input_throughput_token_s": summary.get("per_card_input_throughput_token_s"),
            "per_card_output_throughput_token_s": summary.get("per_card_output_throughput_token_s"),
            "num_devices": config.get("num_devices"),
            "batch_size": config.get("batch_size"),
            # Memory columns
            "model_weight_size_gb": summary.get("model_weight_size_gb"),
            "kv_cache_size_gb": summary.get("kv_cache_size_gb"),
            "model_activation_size_gb": summary.get("model_activation_size_gb"),
            "device_memory_available_gb": summary.get("device_memory_available_gb"),
        }

    # Build sweep_rows (all records, rank-ordered)
    sweep_rows = []
    for record in records:
        summary = record.get("summary", {})
        config = record.get("config", {})
        out = summary.get("throughput_token_s")
        inp = summary.get("input_throughput_token_s")
        sweep_rows.append(
            {
                "device": config.get("device"),
                "rank": record.get("rank"),
                "parallel": config.get("parallel"),
                "concurrency": config.get("concurrency"),
                "throughput_token_s": out,
                "qps": summary.get("qps"),
                "ttft_ms": summary.get("ttft_ms"),
                "tpot_ms": summary.get("tpot_ms"),
                "input_throughput_token_s": inp,
                "e2e_throughput_token_s": out + inp if out is not None and inp is not None else None,
                "per_card_input_throughput_token_s": summary.get("per_card_input_throughput_token_s"),
                "per_card_output_throughput_token_s": summary.get("per_card_output_throughput_token_s"),
                "num_devices": config.get("num_devices"),
                "batch_size": config.get("batch_size"),
                # Memory columns
                "model_weight_size_gb": summary.get("model_weight_size_gb"),
                "kv_cache_size_gb": summary.get("kv_cache_size_gb"),
                "model_activation_size_gb": summary.get("model_activation_size_gb"),
                "device_memory_available_gb": summary.get("device_memory_available_gb"),
            }
        )

    # Sort sweep_rows by rank
    sweep_rows.sort(key=lambda r: r["rank"])

    # Build cross_hardware (best record per selected device). best_config is
    # always set here (rank=1 survivor), so no None-guard is needed.
    cross_hardware = []
    devices_in_config = _get_devices_in_config(input_config or {})
    for device_name in devices_in_config:
        # Find the best record for this device
        best_for_device = None
        best_rank = float("inf")
        for record in records:
            if record.get("config", {}).get("device") == device_name:
                rank = record.get("rank", float("inf"))
                if rank < best_rank:
                    best_rank = rank
                    best_for_device = record

        if best_for_device:
            summary = best_for_device.get("summary", {})
            config = best_for_device.get("config", {})
            cross_hardware.append(
                {
                    "device": device_name,
                    "throughput_token_s": summary.get("throughput_token_s"),
                    "qps": summary.get("qps"),
                    "ttft_ms": summary.get("ttft_ms"),
                    "tpot_ms": summary.get("tpot_ms"),
                    "parallel": config.get("parallel"),
                    "concurrency": config.get("concurrency"),
                }
            )

    # Sort cross_hardware by throughput (descending)
    cross_hardware.sort(key=lambda r: r.get("throughput_token_s", 0), reverse=True)

    # Mode-specific fields
    disagg_prefill = None
    disagg_decode = None
    pd_ratio_rows = []
    pd_pairs: list[dict[str, Any]] | None = None

    # Disaggregated: partition records by phase (now correctly labeled per-row
    # by _agg_disagg_row: ttft→disagg_prefill, tpot→disagg_decode).
    if mode.startswith("disagg"):
        prefill_recs = [r for r in records if r.get("summary", {}).get("mode") == "disagg_prefill"]
        decode_recs = [r for r in records if r.get("summary", {}).get("mode") == "disagg_decode"]
        if prefill_recs:
            prefill_recs.sort(key=lambda r: r.get("rank", 999))
            disagg_prefill = [_sweep_row_from_record(r) for r in prefill_recs]
        if decode_recs:
            decode_recs.sort(key=lambda r: r.get("rank", 999))
            disagg_decode = [_sweep_row_from_record(r) for r in decode_recs]
        # PD-pair system estimate (upper bound) — needs both phase tables and
        # the original lengths from input_config (multi-case passes None and
        # therefore skips this).
        if disagg_prefill and disagg_decode and input_config:
            pd_pairs = _pd_pair_rows(
                disagg_prefill,
                disagg_decode,
                input_config.get("input-length"),
                input_config.get("output-length"),
            )
    elif mode == "pd_ratio":
        # Extract pd_ratio rows
        for record in records:
            summary = record.get("summary", {})
            if "pd_ratio" in summary or "balanced_qps" in summary:
                pd_ratio_rows.append(
                    {
                        "device": record.get("config", {}).get("device"),
                        "parallel_p": record.get("config", {}).get("parallel_p"),
                        "parallel_d": record.get("config", {}).get("parallel_d"),
                        "pd_ratio": summary.get("pd_ratio"),
                        "balanced_qps": summary.get("balanced_qps"),
                        "p_qps": summary.get("p_qps"),
                        "d_qps": summary.get("d_qps"),
                        "ttft_ms": summary.get("ttft_ms"),
                        "tpot_ms": summary.get("tpot_ms"),
                        # P/D split columns
                        "p_devices_per_instance": record.get("config", {}).get("p_devices_per_instance"),
                        "d_devices_per_instance": record.get("config", {}).get("d_devices_per_instance"),
                        "p_batch_size": record.get("config", {}).get("p_batch_size"),
                        "d_batch_size": record.get("config", {}).get("d_batch_size"),
                        "p_concurrency": record.get("config", {}).get("p_concurrency"),
                        "d_concurrency": record.get("config", {}).get("d_concurrency"),
                    }
                )

    # Chrome trace availability
    from services.trace_store import trace_path

    chrome_trace_available = (
        any(trace_path(job_id, r["seq"]).exists() for r in records if "seq" in r) if job_id else False
    )

    # 7-metrics safety net: ensure the throughput metrics are always present
    # in best_config and sweep_rows (even if the runner didn't write them).
    # Uses setdefault so existing values are preserved.
    _BENCHMARK_METRICS = (
        "input_throughput_token_s",
        "e2e_throughput_token_s",
        "per_card_input_throughput_token_s",
        "per_card_output_throughput_token_s",
    )
    if best_config:
        for key in _BENCHMARK_METRICS:
            best_config.setdefault(key, None)
    for row in sweep_rows:
        for key in _BENCHMARK_METRICS:
            row.setdefault(key, None)

    return {
        "mode": mode,
        "input_config": input_config or {},
        "best_config": best_config,
        "sweep_rows": sweep_rows,
        "cross_hardware": cross_hardware,
        "disagg_prefill": disagg_prefill,
        "disagg_decode": disagg_decode,
        "pd_ratio_rows": pd_ratio_rows if pd_ratio_rows else None,
        "pd_pairs": pd_pairs,
        "chrome_trace": {"available": chrome_trace_available},
    }


def _get_devices_in_config(config: dict[str, Any]) -> list[str]:
    """Extract device list from input config (for cross-hardware)."""
    device = config.get("device")
    if isinstance(device, list):
        return device
    if isinstance(device, str):
        return [device]
    return []


def _to_float(value: Any) -> float | None:
    """Coerce to float; None for missing/NaN/unparseable."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def _pd_pair_rows(
    prefill_rows: list[dict[str, Any]],
    decode_rows: list[dict[str, Any]],
    input_length: Any,
    output_length: Any,
) -> list[dict[str, Any]]:
    """Pure: cartesian prefill×decode pairs with system-level upper-bound metrics.

    Disagg sweeps measure each phase INDEPENDENTLY (each row assumes all
    devices on that phase), so a pair's "system" numbers are an UPPER BOUND:
    the steady-state capability if both sides could simultaneously sustain
    their measured throughput. The bottleneck side (lower phase QPS) sets it.

    Pairs where either side lacks a computable QPS are skipped — without QPS
    there is no system metric to derive. Rows are sorted by system QPS desc
    (ties: lower TTFT, then lower TPOT) and ranked 1..N, so rank=1 is the
    recommended pair. ``e2e_throughput_token_s`` = system output + input
    throughput (None when either length is unknown). Does not mutate its inputs.
    """
    in_len = _to_float(input_length)
    out_len = _to_float(output_length)
    pairs: list[dict[str, Any]] = []
    for p in prefill_rows:
        p_qps = _to_float(p.get("qps"))
        if p_qps is None or p_qps <= 0:
            continue
        for d in decode_rows:
            d_qps = _to_float(d.get("qps"))
            if d_qps is None or d_qps <= 0:
                continue
            sys_qps = min(p_qps, d_qps)
            sys_out = sys_qps * out_len if out_len else None
            sys_in = sys_qps * in_len if in_len else None
            pairs.append(
                {
                    "rank": 0,  # assigned after sort
                    "parallel_p": p.get("parallel"),
                    "parallel_d": d.get("parallel"),
                    "p_qps": p_qps,
                    "d_qps": d_qps,
                    "system_qps": sys_qps,
                    "system_output_throughput_token_s": sys_out,
                    "system_input_throughput_token_s": sys_in,
                    "e2e_throughput_token_s": sys_out + sys_in if sys_out is not None and sys_in is not None else None,
                    "p_utilization": sys_qps / p_qps,
                    "d_utilization": sys_qps / d_qps,
                    "bottleneck": "p" if p_qps < d_qps else ("d" if d_qps < p_qps else "balanced"),
                    "ttft_ms": p.get("ttft_ms"),
                    "tpot_ms": d.get("tpot_ms"),
                }
            )

    def _sort_key(row: dict[str, Any]) -> tuple:
        ttft = _to_float(row.get("ttft_ms"))
        tpot = _to_float(row.get("tpot_ms"))
        return (
            -row["system_qps"],
            float("inf") if ttft is None else ttft,
            float("inf") if tpot is None else tpot,
        )

    pairs.sort(key=_sort_key)
    for i, row in enumerate(pairs, start=1):
        row["rank"] = i
    return pairs
