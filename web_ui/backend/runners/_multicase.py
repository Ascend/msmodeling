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

"""Shared multi-case helpers.

Used by runner adapters to expand multi-value fields into the cartesian product
of concrete cases and run them sequentially.
"""

from __future__ import annotations

import itertools
import os
import traceback
from pathlib import Path
from typing import Any, Callable

from services.case_validation import validate_case_for_module

FieldParser = Callable[[Any], list]

#: Upper bound on cases a single job may expand into.
MAX_CASES = 100

_REPO_ROOT = Path(__file__).resolve().parents[3]


def resolve_model_id_path(model_id: Any) -> Any:
    """Resolve relative model_id to a local path when available.

    Resolution order:
      1. ``tests/assets/model_config/<model_id>`` — bundled test/model configs.
         Works with both simple names (``deepseek_v3``) and HuggingFace-style
         ids (``Wan-AI/Wan2.1-T2V-14B-Diffusers``) — the ``/`` in the id becomes
         a nested directory under ``model_config``.
      2. ``<repo_root>/<model_id>`` — repo-relative path.
      3. Original value unchanged (remote HuggingFace model id).

    Args:
        model_id: Model identifier (path, local name, or repo id).

    Returns:
        Resolved absolute path string, or original value if no local match.
    """
    if not model_id or os.path.isabs(model_id) or os.path.isdir(model_id):
        return model_id
    # 1. Try tests/assets/model_config/<model_id> (preserves "/" as nested dir)
    local_candidate = _REPO_ROOT / "tests" / "assets" / "model_config" / model_id
    if local_candidate.is_dir():
        return str(local_candidate)
    # 2. Try <repo_root>/<model_id>
    repo_candidate = _REPO_ROOT / model_id
    if repo_candidate.is_dir():
        return str(repo_candidate)
    # 3. Fall back to original (remote HuggingFace id)
    return model_id


def expand_module_cases_strict(module_id: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand params into per-case dicts, raising on bad/oversized input.

    Args:
        module_id: Module identifier.
        params: Form params.

    Returns:
        List of case param dicts.

    Raises:
        ValueError: If multi-value field is unparseable or exceeds MAX_CASES.
    """
    if module_id == "text_generate":
        from runners.text_generate import _expand_cases

        return _expand_cases(params)
    if module_id == "video_generate":
        from runners.video_generate import _VIDEO_MULTI_FIELDS

        return expand_cases(params, _VIDEO_MULTI_FIELDS)
    if module_id == "throughput_optimizer":
        from runners.throughput_optimizer import _THROUGHPUT_MULTI_FIELDS

        return expand_cases(params, _THROUGHPUT_MULTI_FIELDS)
    return [params]


def as_list(value: Any) -> list:
    """Normalize multi-select value into a list."""
    if value is None:
        return []
    if isinstance(value, list):
        return [v for v in value if v is not None]
    return [value]


def parse_int_list(value: Any) -> list[int]:
    """Parse free-text comma/space list into ints."""
    if value is None:
        return []
    if isinstance(value, list):
        return [int(v) for v in value]
    if isinstance(value, int):
        return [value]
    parts = [p for p in str(value).replace(",", " ").split() if p]
    return [int(p) for p in parts]


def parse_float_list(value: Any) -> list[float]:
    """Parse free-text comma/space list into floats."""
    if value is None:
        return []
    if isinstance(value, list):
        return [float(v) for v in value]
    if isinstance(value, (int, float)):
        return [float(value)]
    parts = [p for p in str(value).replace(",", " ").split() if p]
    return [float(p) for p in parts]


def aggregate_runtime_events(event_list: Any, perf_model_name: str | None = None) -> list[dict]:
    """Aggregate Runtime event_list into structured per-op table.

    Args:
        event_list: List of runtime events.
        perf_model_name: Performance model name (None for counts only).

    Returns:
        List of dicts with name, perf_model, perf_total, perf_avg, call_times.
    """
    aggregated: dict[str, dict[str, float]] = {}
    for event in event_list or []:
        name = str(event.op_invoke_info.func)
        entry = aggregated.setdefault(name, {"total": 0.0, "count": 0})
        entry["count"] += 1
        if perf_model_name is None:
            continue
        result = (event.perf_results or {}).get(perf_model_name)
        if result is not None:
            entry["total"] += getattr(result, "execution_time_s", 0.0)
    items: list[dict] = []
    for name, entry in aggregated.items():
        count = entry["count"]
        total = entry["total"]
        items.append(
            {
                "name": name,
                "perf_model": perf_model_name,
                "perf_total": total,
                "perf_avg": total / count if count else 0.0,
                "call_times": count,
            }
        )
    items.sort(key=lambda x: x["perf_total"], reverse=True)
    return items


def compute_case_hash(module_id: str, version: str | None, case_params: dict[str, Any]) -> str | None:
    """Compute stable hash of case params.

    Args:
        module_id: Module identifier.
        version: Schema version (None disables hashing).
        case_params: Case parameters.

    Returns:
        SHA256 hash string or None if version is unknown.
    """
    if not version:
        return None
    from services.params_hash import compute_params_hash

    return compute_params_hash(module_id, version, case_params)


def expand_cases(
    params: dict[str, Any],
    fields: dict[str, FieldParser],
    *,
    max_cases: int | None = None,
) -> list[dict[str, Any]]:
    """Expand multi-value fields into cartesian product of case param dicts.

    Args:
        params: Original params.
        fields: Dict mapping field_id to parser function.
        max_cases: Override for MAX_CASES limit.

    Returns:
        List of case param dicts.
    """
    keys = list(fields)
    sources: dict[str, list] = {}
    for k in keys:
        parsed = fields[k](params.get(k))
        # An empty/missing field collapses to a single None element (so omitted
        # fields don't fan out, and empty strings don't reach the runner).
        sources[k] = parsed if parsed else [None]
    # Determine the case limit: explicit arg > params > no limit
    limit = max_cases if max_cases is not None else params.get("max-search-combinations")
    # Cap the cartesian product incrementally so a runaway form (many multi-value
    # fields) can't fan out into thousands of sequentially-run cases.
    if limit is not None:
        total = 1
        for k in keys:
            total *= len(sources[k])
            if total > limit:
                raise ValueError(f"too many cases: {total} > {limit}")
    cases: list[dict[str, Any]] = []
    for combo in itertools.product(*(sources[k] for k in keys)):
        case = dict(params)
        for k, v in zip(keys, combo):
            case[k] = v
        cases.append(case)
    return cases


def run_cases(
    params: dict[str, Any],
    fields: dict[str, FieldParser],
    run_one: Callable[[dict[str, Any]], dict[str, Any]],
    *,
    cached_hashes: set[str] | None = None,
    case_hash_ctx: tuple[str, str | None] | None = None,
    job_id: str | None = None,
    provided: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Expand fields and run each case via run_one.

    Args:
        params: Original params.
        fields: Dict mapping field_id to parser function.
        run_one: Function to run a single case.
        cached_hashes: Set of cached case hashes for dedup.
        case_hash_ctx: Tuple of (module_id, version) for hash computation.
        job_id: Job ID for trace path synthesis.
        provided: Set of explicitly provided field names (for wants_provided validators).

    Returns:
        Tuple of (records, skipped_hashes).
    """
    from runners._cli_command import build_cli_command_string

    cached = cached_hashes or set()
    cases = expand_cases(params, fields)

    module_id = case_hash_ctx[0] if case_hash_ctx else None

    # Case divider: a prominent separator line (80 `=` chars) between cases,
    # replacing the `[case i/n]` stamp (which had display issues in some log
    # viewers). Each case's CLI command is still logged, just without the stamp.
    _divider = "\n" + "=" * 80

    def _ch(cp: dict[str, Any]) -> str | None:
        return compute_case_hash(case_hash_ctx[0], case_hash_ctx[1], cp) if case_hash_ctx else None

    # Trace path synthesis when chrome_trace is enabled
    def _synth_trace_path(case_params: dict[str, Any], case_hash: str | None) -> None:
        """If chrome_trace is True, replace it with the computed path.
        If False, convert to None (CLI expects string or None, not boolean).
        """
        trace_val = case_params.get("chrome-trace-file")
        if trace_val is True and job_id and case_hash:
            from services.trace_store import legacy_hash_path

            case_params["chrome-trace-file"] = str(legacy_hash_path(job_id, case_hash))
        elif trace_val is False:
            # Frontend sends boolean False, but CLI expects string or None
            case_params["chrome-trace-file"] = None

    records: list[dict[str, Any]] = []
    skipped: list[str] = []

    # Multi-case expansion summary: the parent process logged the reference
    # command for the ORIGINAL params (before expansion). List the actual
    # per-case commands here so the job log reflects what will really be
    # executed. No `[case i/n]` stamp.
    if module_id and len(cases) > 1:
        print(f"{_divider}", flush=True)
        print(
            f"[{module_id}] Expanding into {len(cases)} case(s) based on multi-field values:",
            flush=True,
        )
        for case_params in cases:
            print(f"  {build_cli_command_string(module_id, case_params)}", flush=True)
        print(flush=True)

    if len(cases) == 1:
        ch = _ch(cases[0])
        if ch and ch in cached:
            print(f"{_divider}", flush=True)
            print(f"Cached (hash {ch[:8]}…)", flush=True)
            skipped.append(ch)
            return records, skipped
        case_params = cases[0]
        _synth_trace_path(case_params, ch)
        cfg = {k: case_params.get(k) for k in fields}
        from services.capture import capture_case_log

        # === Case-level validation (RFC §3.6) ===
        if module_id:
            error, error_fields = validate_case_for_module(module_id, case_params, provided or set())
            if error:
                records.append(
                    {
                        "config": cfg,
                        "summary": {"error": error, "error_fields": error_fields, "validation_failed": True},
                        "tables": {},
                        "case_hash": ch,
                        "case_log": "",
                    }
                )
                return records, skipped

        buf = None
        try:
            with capture_case_log() as buf:
                # Divider + CLI command. No case index label.
                print(f"{_divider}", flush=True)
                print(
                    f"CLI: {build_cli_command_string(module_id, case_params) if module_id else '(unknown module)'}",
                    flush=True,
                )
                print(flush=True)
                rec = run_one(case_params)
            rec["case_hash"] = ch
            rec["case_log"] = buf.getvalue()
            records.append(rec)
        except Exception as e:
            # Mirror the multi-case path: record this case as failed (error in
            # summary + a traceback in the log) instead of letting the exception
            # abort the run with no failure record. The traceback is appended to
            # the case_log so the per-case view shows why it failed.
            traceback.print_exc()
            records.append(
                {
                    "config": cfg,
                    "summary": {"error": str(e)},
                    "tables": {},
                    "case_hash": ch,
                    "case_log": (buf.getvalue() + traceback.format_exc()) if buf is not None else "",
                }
            )
        return records, skipped

    keys = list(fields)
    from services.capture import capture_case_log

    for case_params in cases:
        ch = _ch(case_params)
        if ch and ch in cached:
            print(f"{_divider}", flush=True)
            print(f"Cached (hash {ch[:8]}…)", flush=True)
            skipped.append(ch)
            continue
        _synth_trace_path(case_params, ch)
        cfg = {k: case_params.get(k) for k in keys}

        # === Case-level validation (RFC §3.6) ===
        if module_id:
            error, error_fields = validate_case_for_module(module_id, case_params, provided or set())
            if error:
                records.append(
                    {
                        "config": cfg,
                        "summary": {"error": error, "error_fields": error_fields, "validation_failed": True},
                        "tables": {},
                        "case_hash": ch,
                        "case_log": "",
                    }
                )
                continue  # Skip execution, move to next case

        buf = None
        try:
            with capture_case_log() as buf:
                # Divider + CLI command + key values. No case index label.
                print(f"{_divider}", flush=True)
                if module_id:
                    print(f"CLI: {build_cli_command_string(module_id, case_params)}", flush=True)
                print(" ".join(f"{k}={v}" for k, v in cfg.items()), flush=True)
                print(flush=True)
                rec = run_one(case_params)
            rec["case_hash"] = ch
            rec["case_log"] = buf.getvalue()
            records.append(rec)
        except Exception as e:
            traceback.print_exc()
            records.append(
                {
                    "config": cfg,
                    "summary": {"error": str(e)},
                    "tables": {},
                    "case_hash": ch,
                    "case_log": (buf.getvalue() + traceback.format_exc()) if buf is not None else "",
                }
            )
    return records, skipped
