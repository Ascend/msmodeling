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

"""Persist nightly pytest outcomes and skip completed node ids on resume.

Does not checkpoint the xdist queue. In-flight tests (no terminal journal line)
are omitted from the skip set and run again.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback
    fcntl = None  # type: ignore[assignment]

_PROGRESS_JOURNAL_ENV: Final = "MSMODELING_NIGHTLY_PROGRESS_JOURNAL"
_RESUME_ENV: Final = "MSMODELING_NIGHTLY_RESUME"
_RESUME_ACTIVE_ENV: Final = "MSMODELING_NIGHTLY_RESUME_ACTIVE"
_WAVE_A_MARKER: Final = "not npu and not benchmark and not network"
_WAVE_B_MARKER: Final = "not npu and (benchmark or network)"
SESSION_REL: Final = ".pytest_cache/nightly/progress.session.json"
DURATION_HISTORY_REL: Final = ".pytest_cache/nightly/duration-history.json"
SLOW_FIRST_SEED_PATH: Final = Path(__file__).resolve().parent / "slow_first_seed.txt"
PROGRESS_JOURNALS: Final[dict[str, str]] = {
    "non-benchmark": ".pytest_cache/nightly/progress.non-benchmark.jsonl",
    "benchmark": ".pytest_cache/nightly/progress.benchmark.jsonl",
}
_SLOW_FIRST_ENV: Final = "MSMODELING_NIGHTLY_SLOW_FIRST"
_DEFAULT_SLOW_FIRST: Final = 16
_XDIST_WORKERS_ENV: Final = "MSMODELING_NIGHTLY_XDIST_WORKERS"
_DEFAULT_XDIST_WORKERS: Final = 128
_WAVE_A_COLLECTED_KEY: Final = "wave_a_collected_count"
_COMPLETED_OUTCOMES: Final[frozenset[str]] = frozenset({"passed", "failed", "error", "skipped", "xfailed", "xpassed"})

_journal_path: Path | None = None
_worker_id = "master"
_write_enabled = False
_skip_enabled = False
_reorder_enabled = False
_pending_outcome: dict[str, str] = {}
_pending_call_duration: dict[str, float] = {}


def resolve_resume_enabled() -> bool:
    """True when MSMODELING_NIGHTLY_RESUME is a truthy boolean env value."""
    raw = (os.environ.get(_RESUME_ENV) or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def compute_session_identity(repo_root: Path) -> dict[str, str]:
    """HEAD + dirty tree + wave markers + lock/pyproject fingerprint."""
    head = _git_output(repo_root, "rev-parse", "HEAD").strip()
    dirty = _sha256_text(_git_output(repo_root, "status", "--porcelain"))
    return {
        "head": head,
        "dirty_sha256": dirty,
        "wave_a_marker": _WAVE_A_MARKER,
        "wave_b_marker": _WAVE_B_MARKER,
        "dep_sha256": _dependency_fingerprint(repo_root),
    }


def session_matches(saved: dict[str, str], current: dict[str, str]) -> bool:
    """Require an exact identity match before skipping completed tests."""
    keys = ("head", "dirty_sha256", "wave_a_marker", "wave_b_marker", "dep_sha256")
    return all(saved.get(key, "") == current.get(key, "") for key in keys)


def load_completed_node_ids(journal_path: Path) -> set[str]:
    """Node ids that already have a terminal outcome. In-flight ids are omitted."""
    completed: set[str] = set()
    if not journal_path.is_file():
        return completed
    for raw_line in journal_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        node_id = str(event.get("node_id") or "")
        outcome = str(event.get("outcome") or "").lower()
        if node_id and outcome in _COMPLETED_OUTCOMES:
            completed.add(node_id)
    return completed


def load_node_durations(journal_path: Path) -> dict[str, float]:
    """Last recorded duration_sec per node id."""
    durations: dict[str, float] = {}
    if not journal_path.is_file():
        return durations
    for raw_line in journal_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        node_id = str(event.get("node_id") or "")
        if not node_id:
            continue
        try:
            durations[node_id] = float(event.get("duration_sec") or 0.0)
        except (TypeError, ValueError):
            continue
    return durations


def load_duration_history(path: Path) -> dict[str, float]:
    """Persistent node_id → call seconds across nightly wipes."""
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict):
        return {}
    history: dict[str, float] = {}
    for key, value in raw.items():
        try:
            history[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return history


def write_duration_history(path: Path, durations: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(durations, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def merge_duration_history(repo_root: Path) -> dict[str, float]:
    """Fold current journals into history so a full-run wipe keeps slow-first data."""
    history_path = repo_root / DURATION_HISTORY_REL
    merged = load_duration_history(history_path)
    for rel in PROGRESS_JOURNALS.values():
        merged.update(load_node_durations(repo_root / rel))
    if merged:
        write_duration_history(history_path, merged)
    return merged


def resolve_xdist_worker_cap() -> int:
    """Max Wave A xdist workers. Default 128; never exceed ``os.cpu_count()``."""
    raw = (os.environ.get(_XDIST_WORKERS_ENV) or "").strip()
    cap = _DEFAULT_XDIST_WORKERS
    if raw:
        try:
            cap = max(1, int(raw))
        except ValueError:
            cap = _DEFAULT_XDIST_WORKERS
    cpu = os.cpu_count() or 1
    return max(1, min(cap, cpu))


def estimate_wave_a_remaining(repo_root: Path) -> int | None:
    """Remaining Wave A node ids for resume worker sizing. ``None`` if unknown."""
    completed = load_completed_node_ids(repo_root / PROGRESS_JOURNALS["non-benchmark"])
    session = _load_session(repo_root / SESSION_REL) or {}
    raw_count = (session.get(_WAVE_A_COLLECTED_KEY) or "").strip()
    if raw_count.isdigit():
        return max(0, int(raw_count) - len(completed))
    # Duration history merges both waves and does not retain marker membership,
    # so it cannot provide a safe Wave A total. Keep the configured worker cap
    # until a Wave A collection records an exact count.
    return None


def resolve_wave_a_worker_count(repo_root: Path, *, resume_active: bool) -> int:
    """``min(cap, remaining)`` on resume so leftover runs do not spawn hundreds of workers."""
    cap = resolve_xdist_worker_cap()
    if not resume_active:
        return cap
    remaining = estimate_wave_a_remaining(repo_root)
    if remaining is None:
        return cap
    return max(1, min(cap, remaining))


def resolve_slow_first_count() -> int:
    """How many historically slow node ids to move to the front of the queue."""
    raw = (os.environ.get(_SLOW_FIRST_ENV) or "").strip()
    if not raw:
        return _DEFAULT_SLOW_FIRST
    try:
        return max(0, int(raw))
    except ValueError:
        return _DEFAULT_SLOW_FIRST


def load_slow_first_seed(path: Path | None = None) -> tuple[str, ...]:
    """Committed node-id prefixes for cold-start slow-first."""
    seed_path = path if path is not None else SLOW_FIRST_SEED_PATH
    if not seed_path.is_file():
        return ()
    prefixes: list[str] = []
    for raw_line in seed_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        prefixes.append(line)
    return tuple(prefixes)


def _node_id(item: Any) -> str:
    return str(getattr(item, "nodeid", "") or "")


def _move_items_to_front(items: list[Any], slow: list[Any]) -> list[str]:
    slow_ids = {_node_id(item) for item in slow}
    rest = [item for item in items if _node_id(item) not in slow_ids]
    items[:] = slow + rest
    return [_node_id(item) for item in slow]


def apply_slow_first(
    items: list[Any],
    durations: dict[str, float],
    count: int,
    seed: tuple[str, ...] = (),
) -> list[str]:
    """Move up to *count* slow items to the front. Return those node ids.

    Prefer measured durations. If this machine has no history yet, fall back to
    committed seed prefixes so the first nightly on a new host still starts the
    known compile/model tails first.
    """
    if count <= 0 or not items:
        return []
    ranked = sorted(
        ((float(durations.get(_node_id(item)) or 0.0), index, item) for index, item in enumerate(items)),
        key=lambda row: (-row[0], row[1]),
    )
    slow = [item for duration, _index, item in ranked[:count] if duration > 0.0]
    if slow:
        return _move_items_to_front(items, slow)
    if not seed:
        return []
    picked: list[Any] = []
    picked_ids: set[str] = set()
    for prefix in seed:
        for item in items:
            node_id = _node_id(item)
            if node_id in picked_ids or not node_id.startswith(prefix):
                continue
            picked.append(item)
            picked_ids.add(node_id)
            if len(picked) >= count:
                return _move_items_to_front(items, picked)
    if not picked:
        return []
    return _move_items_to_front(items, picked)


def prepare_progress_session(repo_root: Path, *, resume: bool, logger: logging.Logger) -> bool:
    """Validate or reset the progress session. Return True when skip-on-resume is active."""
    identity = compute_session_identity(repo_root)
    session_path = repo_root / SESSION_REL
    if resume:
        saved = _load_session(session_path)
        if saved is not None and session_matches(saved, identity):
            completed = 0
            for rel in PROGRESS_JOURNALS.values():
                completed += len(load_completed_node_ids(repo_root / rel))
            logger.info(
                "Nightly resume enabled: skipping %d completed node id(s) from progress journal",
                completed,
            )
            return True
        logger.warning("Nightly resume refused: session fingerprint mismatch or missing; starting a full run")
    merge_duration_history(repo_root)
    _wipe_progress_journals(repo_root)
    _write_session(session_path, identity)
    return False


def pytest_configure(config: Any) -> None:
    """Enable JSONL writes on workers/serial pytest; skip completed ids when resume is active."""
    global _journal_path, _worker_id, _write_enabled, _skip_enabled, _reorder_enabled
    global _pending_outcome, _pending_call_duration

    _pending_outcome = {}
    _pending_call_duration = {}
    raw_path = (os.environ.get(_PROGRESS_JOURNAL_ENV) or "").strip()
    _journal_path = Path(raw_path) if raw_path else None
    _skip_enabled = (os.environ.get(_RESUME_ACTIVE_ENV) or "").strip() == "1" and _journal_path is not None
    worker_input = getattr(config, "workerinput", None)
    if isinstance(worker_input, dict):
        _worker_id = str(worker_input.get("workerid", "worker"))
        _write_enabled = _journal_path is not None
        # xdist workers collect the item list used by worksteal; reorder must happen here.
        _reorder_enabled = _journal_path is not None
        return

    num_processes = getattr(config.option, "numprocesses", None)
    _worker_id = "master"
    _write_enabled = _journal_path is not None and not num_processes
    _reorder_enabled = _journal_path is not None


def pytest_collection_modifyitems(config: Any, items: list[Any]) -> None:
    """Deselect completed resume ids, then move a few historically slow tests first."""
    collected_before_skip = len(items)
    if _skip_enabled and _journal_path is not None:
        completed = load_completed_node_ids(_journal_path)
        if completed:
            kept: list[Any] = []
            deselected: list[Any] = []
            for item in items:
                if str(getattr(item, "nodeid", "")) in completed:
                    deselected.append(item)
                else:
                    kept.append(item)
            if deselected:
                config.hook.pytest_deselected(items=deselected)
                items[:] = kept
    if (
        _journal_path is not None
        and _journal_path.name == Path(PROGRESS_JOURNALS["non-benchmark"]).name
        and _worker_id in {"master", "gw0"}
    ):
        # In xdist, every worker performs the full collection and the controller
        # performs none. Limit the write to gw0 (or master in serial pytest) so
        # all xdist workers do not rewrite the same session field.
        _update_session_field(
            _journal_path.parent / Path(SESSION_REL).name,
            _WAVE_A_COLLECTED_KEY,
            str(collected_before_skip),
        )
    if not _reorder_enabled or _journal_path is None:
        return
    count = resolve_slow_first_count()
    if count <= 0:
        return
    history = load_duration_history(_journal_path.parent / Path(DURATION_HISTORY_REL).name)
    seed = () if any(float(history.get(_node_id(item)) or 0.0) > 0.0 for item in items) else load_slow_first_seed()
    moved = apply_slow_first(items, history, count, seed=seed)
    if moved and _worker_id in {"master", "gw0"}:
        preview = ", ".join(moved[:8])
        extra = "" if len(moved) <= 8 else f", ... (+{len(moved) - 8})"
        logging.getLogger("nightly").info(
            "Nightly slow-first: moved %d historically slow node id(s) to the front: %s%s",
            len(moved),
            preview,
            extra,
        )


def pytest_runtest_logreport(report: Any) -> None:
    """Flush a terminal outcome after setup-fail / call-fail / teardown (pass or fail)."""
    if not _write_enabled or _journal_path is None:
        return
    node_id = str(getattr(report, "nodeid", "") or "")
    if not node_id:
        return
    when = str(getattr(report, "when", "") or "")
    outcome = _report_outcome(report)

    if when == "setup":
        if outcome in {"failed", "error", "skipped", "xfailed"}:
            _append_outcome(node_id, outcome=outcome, phase="setup", report=report)
            _pending_outcome.pop(node_id, None)
            _pending_call_duration.pop(node_id, None)
        return

    if when == "call":
        if outcome in {"failed", "error", "skipped", "xfailed", "xpassed"}:
            _append_outcome(node_id, outcome=outcome, phase="call", report=report)
            _pending_outcome.pop(node_id, None)
            _pending_call_duration.pop(node_id, None)
            return
        if outcome == "passed":
            _pending_outcome[node_id] = "passed"
            _pending_call_duration[node_id] = float(getattr(report, "duration", 0.0) or 0.0)
        return

    if when == "teardown":
        call_duration = _pending_call_duration.pop(node_id, None)
        if outcome in {"failed", "error"}:
            _append_outcome(node_id, outcome="error", phase="teardown", report=report)
        elif _pending_outcome.get(node_id) == "passed":
            _append_outcome(
                node_id,
                outcome="passed",
                phase="teardown",
                report=report,
                duration_sec=call_duration,
            )
        _pending_outcome.pop(node_id, None)


def pytest_collectreport(report: Any) -> None:
    """Collection errors are terminal and must not be skipped as in-flight."""
    if not _write_enabled or _journal_path is None:
        return
    if not getattr(report, "failed", False):
        return
    node_id = str(getattr(report, "nodeid", "") or "")
    if node_id:
        _append_outcome(node_id, outcome="error", phase="collect", report=report)


def _report_outcome(report: Any) -> str:
    raw = str(getattr(report, "outcome", "") or "").lower()
    if raw in _COMPLETED_OUTCOMES:
        return raw
    if getattr(report, "failed", False):
        return "failed"
    if getattr(report, "skipped", False):
        return "skipped"
    if getattr(report, "passed", False):
        return "passed"
    return raw


def _append_outcome(
    node_id: str,
    *,
    outcome: str,
    phase: str,
    report: Any,
    duration_sec: float | None = None,
) -> None:
    if _journal_path is None:
        return
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "worker": _worker_id,
        "node_id": node_id,
        "phase": phase,
        "outcome": outcome,
        "duration_sec": float(duration_sec if duration_sec is not None else getattr(report, "duration", 0.0) or 0.0),
    }
    _append_jsonl(_journal_path, payload)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    encoded = (json.dumps(payload, ensure_ascii=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        if fcntl is not None:
            fcntl.flock(fd, fcntl.LOCK_EX)
        remaining = memoryview(encoded)
        while remaining:
            written = os.write(fd, remaining)
            remaining = remaining[written:]
        os.fsync(fd)
    finally:
        if fcntl is not None:
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _wipe_progress_journals(repo_root: Path) -> None:
    for rel in PROGRESS_JOURNALS.values():
        (repo_root / rel).unlink(missing_ok=True)


def _load_session(path: Path) -> dict[str, str] | None:
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    return {str(key): str(value) for key, value in raw.items()}


def _write_session(path: Path, identity: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _update_session_field(path: Path, key: str, value: str) -> None:
    saved = _load_session(path) or {}
    saved[key] = value
    _write_session(path, saved)


def _dependency_fingerprint(repo_root: Path) -> str:
    hasher = hashlib.sha256()
    for name in ("pyproject.toml", "uv.lock"):
        path = repo_root / name
        hasher.update(name.encode())
        hasher.update(b"\0")
        if path.is_file():
            hasher.update(path.read_bytes())
        hasher.update(b"\0")
    return hasher.hexdigest()


def _git_output(repo_root: Path, *args: str) -> str:
    git_bin = shutil.which("git")
    if git_bin is None:
        return ""
    try:
        proc = subprocess.run(
            [git_bin, "-C", str(repo_root), *args],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return ""
    if proc.returncode != 0:
        return ""
    return proc.stdout


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# Imported by main.py for env keys / paths without instantiating pytest hooks.
PROGRESS_JOURNAL_ENV: Final = _PROGRESS_JOURNAL_ENV
RESUME_ACTIVE_ENV: Final = _RESUME_ACTIVE_ENV
WAVE_A_MARKER: Final = _WAVE_A_MARKER
WAVE_B_MARKER: Final = _WAVE_B_MARKER
XDIST_WORKERS_ENV: Final = _XDIST_WORKERS_ENV
