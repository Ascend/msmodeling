"""Shared multi-case helpers (Phase D2).

Used by the video / throughput (and reusable by text) adapters to expand a set
of multi-value fields into the cartesian product of concrete cases and run them
sequentially in-process: each case is prefaced by a ``===== Case i/N =====``
header in the job log, and a case that raises is captured as a failed case
(error in its summary + a traceback in the log) without aborting the loop.

Single-value submits collapse to one case (backward compatible — the adapter's
normal single-case path).
"""

from __future__ import annotations

import itertools
import os
import traceback
from pathlib import Path
from typing import Any, Callable

FieldParser = Callable[[Any], list]

#: Upper bound on the number of cases a single job may expand into. Free-text
#: multi-value fields (num_queries/tp_size/...) are comma lists, so pasting many
#: values or combining multi-selects can otherwise fan out into thousands of
#: sequentially-run cases (workers run them in-process, one by one).
MAX_CASES = 100

_REPO_ROOT = Path(__file__).resolve().parents[3]


def resolve_model_id_path(model_id: Any) -> Any:
    """Resolve a relative model_id against the repo root.

    The backend worker runs from ``web/backend``, so a repo-relative model_id
    (e.g. ``tests/assets/...``) must be joined to the repo root to be found.
    Absolute paths, existing dirs, and remote repo ids are returned unchanged
    (remote ids are resolved later by each runner's model loader).
    """
    if model_id and not os.path.isabs(model_id) and not os.path.isdir(model_id):
        candidate = _REPO_ROOT / model_id
        if candidate.is_dir():
            return str(candidate)
    return model_id


def as_list(value: Any) -> list:
    """Normalize a multi-select value (list | str | None) into a list."""
    if value is None:
        return []
    if isinstance(value, list):
        return [v for v in value if v is not None]
    return [value]


def parse_int_list(value: Any) -> list[int]:
    """Parse a free-text comma/space list (or a real list / single int) into ints."""
    if value is None:
        return []
    if isinstance(value, list):
        return [int(v) for v in value]
    if isinstance(value, int):
        return [value]
    parts = [p for p in str(value).replace(",", " ").split() if p]
    return [int(p) for p in parts]


def parse_float_list(value: Any) -> list[float]:
    """Parse a free-text comma/space list (or list / single number) into floats."""
    if value is None:
        return []
    if isinstance(value, list):
        return [float(v) for v in value]
    if isinstance(value, (int, float)):
        return [float(value)]
    parts = [p for p in str(value).replace(",", " ").split() if p]
    return [float(p) for p in parts]


def aggregate_runtime_events(event_list: Any, perf_model_name: str | None = None) -> list[dict]:
    """Aggregate a Runtime/runner event_list into the structured per-op table
    (mirrors ModelRunner._aggregate_runtime_events, usable by text + video):
    [{name, perf_model, perf_total, perf_avg, call_times}], sorted by total desc.
    With ``perf_model_name=None`` totals are 0 (counts still populated).
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
    """Stable hash of one case's concrete params = sha256(module, version, params).
    Returns None when version is unknown (case hashing disabled — no dedup).
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
    """Expand ``fields`` (each a multi-value field + its parser) into the
    cartesian product of concrete case param dicts. An empty/missing field
    collapses to the original single value (so omitted fields don't fan out).

    ``max_cases`` overrides the hardcoded ``MAX_CASES`` limit. If ``None``,
    the limit is read from ``params["max_search_combinations"]`` (if present);
    if that's also absent, no limit is enforced.
    """
    keys = list(fields)
    sources: dict[str, list] = {}
    for k in keys:
        parsed = fields[k](params.get(k))
        # An empty/missing field collapses to a single None element (so omitted
        # fields don't fan out, and empty strings don't reach the runner).
        sources[k] = parsed if parsed else [None]
    # Determine the case limit: explicit arg > params > no limit
    limit = max_cases if max_cases is not None else params.get("max_search_combinations")
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
) -> tuple[list[dict[str, Any]], list[str]]:
    """Expand ``fields`` and run each case via ``run_one(case_params)``.

    Single case -> one record (no header). Multiple cases -> loop with a per-case
    header + error capture (failed cases recorded, loop continues).

    Case-level dedup: when ``cached_hashes`` + ``case_hash_ctx`` are provided, a
    case whose ``compute_case_hash`` is in ``cached_hashes`` is SKIPPED (appended
    to the returned ``skipped`` list) instead of run — the main process clones its
    records from a prior succeeded job. Each run record carries its ``case_hash``.

    Per-case CLI logging: the parent process logs a "CLI (reference, unexpanded)"
    line using the ORIGINAL params (before multi-case expansion). For runners that
    fan out, the ACTUAL per-case CLI commands are logged here so the job log
    reflects what is really executed.

    Returns ``(records, skipped_hashes)``.
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
        """If chrome_trace is True, replace it with the computed path."""
        if case_params.get("chrome_trace") is True and job_id and case_hash:
            from services.trace_store import legacy_hash_path

            case_params["chrome_trace"] = str(legacy_hash_path(job_id, case_hash))

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

    for i, case_params in enumerate(cases, 1):
        ch = _ch(case_params)
        if ch and ch in cached:
            print(f"{_divider}", flush=True)
            print(f"Cached (hash {ch[:8]}…)", flush=True)
            skipped.append(ch)
            continue
        _synth_trace_path(case_params, ch)
        cfg = {k: case_params.get(k) for k in keys}
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
