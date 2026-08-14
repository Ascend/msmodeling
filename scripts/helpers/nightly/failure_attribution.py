"""First-bad commit attribution via Shanghai day-walk + linear/bisect in worktrees."""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Final
from zoneinfo import ZoneInfo

from scripts.helpers.nightly.pytest_parser import normalize_stdout_node_id
from scripts.helpers.nightly.report_models import AttributionConclusion, FailureBlame

logger = logging.getLogger(__name__)

_ATTRIBUTION_TZ: Final = ZoneInfo("Asia/Shanghai")
_MAX_LOOKBACK_DAYS: Final = 7
_GIT_LOG_FIELD_SEP: Final = "\x1f"
_UNKNOWN_COMMIT: Final = "unknown"
_UNKNOWN_AUTHOR: Final = "unknown"
_BISECT_NODE_ENV: Final = "MSMODELING_BISECT_NODE"
_BISECT_PYTHON_ENV: Final = "MSMODELING_BISECT_PYTHON"
_BISECT_DEADLINE_UNIX_ENV: Final = "MSMODELING_ATTRIBUTION_DEADLINE_UNIX"
_PYTEST_UNCOLLECTIBLE_EXIT_CODES: Final = frozenset({2, 5})
_WORKTREE_NAME_PREFIX: Final = "nightly-attrib-"
_NODE_PYTEST_TIMEOUT_SECONDS: Final = 600
_NODE_PYTEST_TIMEOUT_EXIT_CODE: Final = 124
# When ``git rev-list --count good..bad`` is at most this, walk oldest→newest
# (first fail = first-bad). Above this, use git bisect. At 16, linear worst-case
# is still cheap vs bisect process overhead on tiny ranges (~log2(16)=4 steps).
LINEAR_VS_BISECT_MAX_COMMITS: Final = 16
_NEED_HUMAN_LOOKBACK: Final = "Still failing after 7-day lookback; needs human follow-up"
_NEED_HUMAN_INCOMPLETE: Final = "Attribution incomplete; needs human follow-up"
_CANNOT_REPRODUCE_LABEL: Final = "Flaky / not reproduced at HEAD"
_UNCOLLECTIBLE_LABEL: Final = "Test not collectable at current commit; needs human follow-up"
_ATTRIBUTION_TIMEOUT_LABEL: Final = "Attribution pytest timed out; needs human follow-up"
ATTRIBUTION_BUDGET_SKIP_LABEL: Final = "Attribution skipped due to timeout"

_GIT: str | None = None


def _git_bin() -> str:
    global _GIT
    if _GIT is None:
        path = shutil.which("git")
        if path is None:
            msg = "git not found"
            raise RuntimeError(msg)
        _GIT = path
    return _GIT


@dataclass(frozen=True, slots=True)
class FirstBadResult:
    node_id: str
    commit_id: str
    author: str
    subject: str
    conclusion: AttributionConclusion
    good_commit: str | None = None
    detail: str = ""

    @property
    def attributed(self) -> bool:
        return self.conclusion == AttributionConclusion.FIRST_BAD

    @property
    def needs_human(self) -> bool:
        return self.conclusion in {
            AttributionConclusion.NEED_HUMAN,
            AttributionConclusion.UNCOLLECTIBLE,
        }


class LookbackExhaustedError(Exception):
    """No validated good within the Shanghai calendar lookback window."""

    def __init__(self, node_id: str, *, bad: str, lookback_days: int, reason: str = "") -> None:
        self.node_id = node_id
        self.bad = bad
        self.lookback_days = lookback_days
        self.reason = reason or _NEED_HUMAN_LOOKBACK
        super().__init__(f"{self.reason} ({node_id}; bad={bad})")


LookbackExhausted = LookbackExhaustedError


def _run_git(
    repo_root: Path,
    *args: str,
    env: dict[str, str] | None = None,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            [_git_bin(), *args],
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess([_git_bin(), *args], 124, "", "attribution budget exhausted")


def _git_stdout(repo_root: Path, *args: str) -> str:
    return _run_git(repo_root, *args).stdout.strip()


def _require_git(proc: subprocess.CompletedProcess[str], description: str) -> None:
    if proc.returncode != 0:
        stderr = proc.stderr.strip()
        message = f"{description}: {stderr}" if stderr else description
        raise RuntimeError(message)


def default_max_workers(failed_count: int) -> int:
    """Concurrency for attribution workers: min(failed, max(1, cpu//2))."""
    if failed_count <= 0:
        return 1
    return min(failed_count, max(1, (os.cpu_count() or 1) // 2))


def run_node_pytest(
    repo_root: Path,
    node_id: str,
    *,
    python_exe: str,
    timeout_seconds: float | None = None,
) -> int:
    """Run one pytest node at *repo_root*; return pytest exit code."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)
    effective_timeout = _NODE_PYTEST_TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds
    try:
        proc = subprocess.run(
            [python_exe, "-m", "pytest", node_id, "-x", "--tb=line", "-q"],
            cwd=repo_root,
            env=env,
            check=False,
            timeout=effective_timeout,
        )
    except subprocess.TimeoutExpired:
        logger.error(
            "Attribution pytest timed out after %ss for %s",
            effective_timeout,
            node_id,
        )
        return _NODE_PYTEST_TIMEOUT_EXIT_CODE
    return proc.returncode


def _node_pytest_timeout_seconds(*, deadline: float | None) -> float | None:
    """Seconds for one attribution pytest; None keeps the module default."""
    if deadline is None:
        return None
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return 0.01
    return min(float(_NODE_PYTEST_TIMEOUT_SECONDS), remaining)


def _budget_exhausted(deadline: float | None) -> bool:
    return deadline is not None and time.monotonic() >= deadline


def _budget_remaining_seconds(deadline: float | None) -> float | None:
    if deadline is None:
        return None
    return max(0.0, deadline - time.monotonic())


def _unix_deadline_from_monotonic(deadline: float | None) -> float | None:
    """Map a monotonic deadline to ``time.time()`` for child processes."""
    remaining = _budget_remaining_seconds(deadline)
    if remaining is None:
        return None
    return time.time() + remaining


def _monotonic_deadline_from_unix_env() -> float | None:
    raw = (os.environ.get(_BISECT_DEADLINE_UNIX_ENV) or "").strip()
    if not raw:
        return None
    try:
        unix_deadline = float(raw)
    except ValueError:
        return None
    remaining = unix_deadline - time.time()
    if remaining <= 0:
        return time.monotonic() - 1.0
    return time.monotonic() + remaining


def _run_node_pytest_with_deadline(
    repo_root: Path,
    node_id: str,
    *,
    python_exe: str,
    deadline: float | None,
) -> int:
    if _budget_exhausted(deadline):
        return _NODE_PYTEST_TIMEOUT_EXIT_CODE
    return run_node_pytest(
        repo_root,
        node_id,
        python_exe=python_exe,
        timeout_seconds=_node_pytest_timeout_seconds(deadline=deadline),
    )


def _budget_skip_results(node_ids: tuple[str, ...]) -> tuple[FirstBadResult, ...]:
    return tuple(_need_human_result(node_id, detail=ATTRIBUTION_BUDGET_SKIP_LABEL) for node_id in node_ids)


def _attribution_skip_exit(exit_code: int) -> bool:
    """True when a single-node attribution pytest result is inconclusive (skip bisect/linear)."""
    return exit_code in _PYTEST_UNCOLLECTIBLE_EXIT_CODES or exit_code == _NODE_PYTEST_TIMEOUT_EXIT_CODE


def _bisect_exit_code(pytest_exit_code: int) -> int:
    if _attribution_skip_exit(pytest_exit_code):
        return 125
    return 0 if pytest_exit_code == 0 else 1


def bisect_check() -> None:
    """Entry point for ``git bisect run`` — reads node id from env; cwd is worktree."""
    node_id = os.environ.get(_BISECT_NODE_ENV, "")
    if not node_id:
        sys.exit(125)
    deadline = _monotonic_deadline_from_unix_env()
    if _budget_exhausted(deadline):
        sys.exit(255)
    python_exe = os.environ.get(_BISECT_PYTHON_ENV) or sys.executable
    sys.exit(
        _bisect_exit_code(
            _run_node_pytest_with_deadline(
                Path.cwd(),
                node_id,
                python_exe=python_exe,
                deadline=deadline,
            )
        )
    )


def _commit_metadata(repo_root: Path, commit: str) -> tuple[str, str, str]:
    raw = _git_stdout(
        repo_root,
        "log",
        "-1",
        f"--format=%h{_GIT_LOG_FIELD_SEP}%an{_GIT_LOG_FIELD_SEP}%s",
        commit,
    )
    if not raw or _GIT_LOG_FIELD_SEP not in raw:
        return _UNKNOWN_COMMIT, _UNKNOWN_AUTHOR, "(unknown)"
    commit_id, author, subject = raw.split(_GIT_LOG_FIELD_SEP, 2)
    return commit_id, author, subject


def _need_human_result(node_id: str, *, detail: str, good_commit: str | None = None) -> FirstBadResult:
    return FirstBadResult(
        node_id=node_id,
        commit_id=_UNKNOWN_COMMIT,
        author=_UNKNOWN_AUTHOR,
        subject=detail,
        conclusion=AttributionConclusion.NEED_HUMAN,
        good_commit=good_commit,
        detail=detail,
    )


def _build_cannot_reproduce(repo_root: Path, node_id: str, bad: str) -> FirstBadResult:
    commit_id, author, _subject = _commit_metadata(repo_root, bad)
    return FirstBadResult(
        node_id=node_id,
        commit_id=commit_id,
        author=author,
        subject=_CANNOT_REPRODUCE_LABEL,
        conclusion=AttributionConclusion.CANNOT_REPRODUCE,
        detail=_CANNOT_REPRODUCE_LABEL,
    )


def find_day_candidate(
    repo_root: Path,
    *,
    day: date,
    tz: ZoneInfo,
    tip: str,
) -> str | None:
    """Last first-parent commit on that Shanghai calendar day at/before tip; None if empty."""
    start = datetime(day.year, day.month, day.day, tzinfo=tz)
    end = start + timedelta(days=1)
    raw = _git_stdout(
        repo_root,
        "rev-list",
        "--first-parent",
        "--max-count=1",
        f"--after={int(start.timestamp()) - 1}",
        f"--before={int(end.timestamp())}",
        tip,
    )
    return raw or None


def _is_shallow_repo(repo_root: Path) -> bool:
    git_dir = _git_stdout(repo_root, "rev-parse", "--git-dir")
    if not git_dir:
        return False
    git_path = Path(git_dir)
    if not git_path.is_absolute():
        git_path = repo_root / git_path
    return (git_path / "shallow").is_file()


def find_good_commit(
    worktree: Path,
    node_id: str,
    *,
    bad: str,
    python_exe: str,
    max_lookback_days: int = _MAX_LOOKBACK_DAYS,
    now: datetime | None = None,
    deadline: float | None = None,
) -> str:
    """Day-walk Asia/Shanghai from previous calendar day; raise LookbackExhausted if none."""
    tz = _ATTRIBUTION_TZ
    current = now.astimezone(tz) if now is not None else datetime.now(tz)
    today = current.date()
    saw_any_candidate = False

    for offset in range(1, max_lookback_days + 1):
        if _budget_exhausted(deadline):
            raise LookbackExhausted(
                node_id,
                bad=bad,
                lookback_days=max_lookback_days,
                reason=ATTRIBUTION_BUDGET_SKIP_LABEL,
            )
        day = today - timedelta(days=offset)
        candidate = find_day_candidate(worktree, day=day, tz=tz, tip=bad)
        if candidate is None:
            continue
        saw_any_candidate = True
        checkout = _run_git(worktree, "checkout", "--detach", candidate)
        if checkout.returncode != 0:
            logger.warning(
                "checkout %s failed during day-walk: %s",
                candidate[:12],
                checkout.stderr.strip(),
            )
            continue
        exit_code = _run_node_pytest_with_deadline(
            worktree,
            node_id,
            python_exe=python_exe,
            deadline=deadline,
        )
        if _budget_exhausted(deadline):
            raise LookbackExhausted(
                node_id,
                bad=bad,
                lookback_days=max_lookback_days,
                reason=ATTRIBUTION_BUDGET_SKIP_LABEL,
            )
        if _attribution_skip_exit(exit_code):
            if exit_code in _PYTEST_UNCOLLECTIBLE_EXIT_CODES:
                logger.info("Skipping uncollectible commit %s for %s", candidate[:12], node_id)
            else:
                logger.info("Skipping timed-out attribution at commit %s for %s", candidate[:12], node_id)
            continue
        if exit_code == 0:
            return candidate

    if not saw_any_candidate or _is_shallow_repo(worktree):
        raise LookbackExhausted(
            node_id,
            bad=bad,
            lookback_days=max_lookback_days,
            reason=(
                "Insufficient history or shallow clone; cannot complete day-walk attribution; needs human follow-up"
            ),
        )
    raise LookbackExhausted(node_id, bad=bad, lookback_days=max_lookback_days)


def _count_commits(repo_root: Path, good: str, bad: str) -> int:
    raw = _git_stdout(repo_root, "rev-list", "--count", f"{good}..{bad}")
    try:
        return int(raw)
    except ValueError:
        return 0


def linear_first_bad(
    worktree: Path,
    node_id: str,
    *,
    good: str,
    bad: str,
    python_exe: str,
    deadline: float | None = None,
) -> FirstBadResult:
    """Oldest→newest walk of good..bad; first failing collectible commit is first-bad."""
    commits = _git_stdout(worktree, "rev-list", "--reverse", f"{good}..{bad}").splitlines()
    if not commits:
        return _need_human_result(node_id, detail=_NEED_HUMAN_INCOMPLETE, good_commit=good)

    for commit in commits:
        if _budget_exhausted(deadline):
            return _need_human_result(node_id, detail=ATTRIBUTION_BUDGET_SKIP_LABEL, good_commit=good)
        checkout = _run_git(worktree, "checkout", "--detach", commit)
        if checkout.returncode != 0:
            logger.warning("linear checkout %s failed: %s", commit[:12], checkout.stderr.strip())
            continue
        exit_code = _run_node_pytest_with_deadline(
            worktree,
            node_id,
            python_exe=python_exe,
            deadline=deadline,
        )
        if _budget_exhausted(deadline):
            return _need_human_result(node_id, detail=ATTRIBUTION_BUDGET_SKIP_LABEL, good_commit=good)
        if _attribution_skip_exit(exit_code):
            continue
        if exit_code != 0:
            commit_id, author, subject = _commit_metadata(worktree, commit)
            return FirstBadResult(
                node_id=node_id,
                commit_id=commit_id,
                author=author,
                subject=subject,
                conclusion=AttributionConclusion.FIRST_BAD,
                good_commit=good,
                detail=f"Introducing commit {commit_id}",
            )

    return _need_human_result(node_id, detail=_NEED_HUMAN_INCOMPLETE, good_commit=good)


def bisect_first_bad(
    main_root: Path,
    worktree: Path,
    node_id: str,
    *,
    good: str,
    bad: str,
    python_exe: str,
    deadline: float | None = None,
) -> FirstBadResult:
    """Bisect good..bad inside *worktree* using main checkout's attribution module."""
    if _budget_exhausted(deadline):
        return _need_human_result(node_id, detail=ATTRIBUTION_BUDGET_SKIP_LABEL, good_commit=good)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(main_root)
    env[_BISECT_NODE_ENV] = node_id
    env[_BISECT_PYTHON_ENV] = python_exe
    unix_deadline = _unix_deadline_from_monotonic(deadline)
    if unix_deadline is not None:
        env[_BISECT_DEADLINE_UNIX_ENV] = str(unix_deadline)

    start = _run_git(worktree, "bisect", "start", bad, good, env=env)
    if start.returncode != 0:
        logger.warning("git bisect start failed: %s", start.stderr.strip())
        return _need_human_result(node_id, detail=_NEED_HUMAN_INCOMPLETE, good_commit=good)

    try:
        remaining = _budget_remaining_seconds(deadline)
        proc = _run_git(
            worktree,
            "bisect",
            "run",
            python_exe,
            "-m",
            "scripts.helpers.nightly.failure_attribution",
            env=env,
            timeout=remaining,
        )
        if _budget_exhausted(deadline) or proc.returncode == 124:
            return _need_human_result(node_id, detail=ATTRIBUTION_BUDGET_SKIP_LABEL, good_commit=good)
        if proc.returncode == 0:
            bad_commit = _git_stdout(worktree, "rev-parse", "HEAD")
            commit_id, author, subject = _commit_metadata(worktree, bad_commit)
            return FirstBadResult(
                node_id=node_id,
                commit_id=commit_id,
                author=author,
                subject=subject,
                conclusion=AttributionConclusion.FIRST_BAD,
                good_commit=good,
                detail=f"Introducing commit {commit_id}",
            )
        logger.warning("git bisect run failed (exit %d): %s", proc.returncode, proc.stderr.strip())
        return _need_human_result(node_id, detail=_NEED_HUMAN_INCOMPLETE, good_commit=good)
    finally:
        reset = _run_git(worktree, "bisect", "reset", env=env)
        if reset.returncode != 0:
            logger.warning("git bisect reset failed: %s", reset.stderr.strip())


def find_first_bad(
    main_root: Path,
    worktree: Path,
    node_id: str,
    *,
    good: str,
    bad: str,
    python_exe: str,
    deadline: float | None = None,
) -> FirstBadResult:
    """Choose linear or bisect from ``LINEAR_VS_BISECT_MAX_COMMITS``."""
    count = _count_commits(worktree, good, bad)
    if count <= LINEAR_VS_BISECT_MAX_COMMITS:
        logger.info(
            "Using linear first-bad for %s (%d commits ≤ %d)",
            node_id,
            count,
            LINEAR_VS_BISECT_MAX_COMMITS,
        )
        return linear_first_bad(
            worktree,
            node_id,
            good=good,
            bad=bad,
            python_exe=python_exe,
            deadline=deadline,
        )
    logger.info(
        "Using bisect first-bad for %s (%d commits > %d)",
        node_id,
        count,
        LINEAR_VS_BISECT_MAX_COMMITS,
    )
    return bisect_first_bad(
        main_root,
        worktree,
        node_id,
        good=good,
        bad=bad,
        python_exe=python_exe,
        deadline=deadline,
    )


def _worktree_path(parent: Path, node_id: str) -> Path:
    digest = hashlib.sha1(node_id.encode("utf-8"), usedforsecurity=False).hexdigest()[:12]
    return parent / f"{_WORKTREE_NAME_PREFIX}{digest}"


def _add_worktree(repo_root: Path, path: Path, commit: str) -> None:
    if path.exists():
        _run_git(repo_root, "worktree", "remove", "--force", str(path))
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
    proc = _run_git(repo_root, "worktree", "add", "--detach", str(path), commit)
    _require_git(proc, f"git worktree add failed for {path}")


def _remove_worktree(repo_root: Path, path: Path) -> None:
    proc = _run_git(repo_root, "worktree", "remove", "--force", str(path))
    if proc.returncode != 0:
        logger.warning("git worktree remove failed for %s: %s", path, proc.stderr.strip())
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    _run_git(repo_root, "worktree", "prune")


def _attribute_one_at(
    main_root: Path,
    work_root: Path,
    node_id: str,
    *,
    bad: str,
    python_exe: str,
    deadline: float | None = None,
) -> FirstBadResult:
    if _budget_exhausted(deadline):
        return _need_human_result(node_id, detail=ATTRIBUTION_BUDGET_SKIP_LABEL)
    exit_at_bad = _run_node_pytest_with_deadline(
        work_root,
        node_id,
        python_exe=python_exe,
        deadline=deadline,
    )
    if exit_at_bad in _PYTEST_UNCOLLECTIBLE_EXIT_CODES:
        return FirstBadResult(
            node_id=node_id,
            commit_id=_UNKNOWN_COMMIT,
            author=_UNKNOWN_AUTHOR,
            subject=_UNCOLLECTIBLE_LABEL,
            conclusion=AttributionConclusion.UNCOLLECTIBLE,
            detail=_UNCOLLECTIBLE_LABEL,
        )
    if exit_at_bad == _NODE_PYTEST_TIMEOUT_EXIT_CODE:
        if _budget_exhausted(deadline):
            return _need_human_result(node_id, detail=ATTRIBUTION_BUDGET_SKIP_LABEL)
        return _need_human_result(node_id, detail=_ATTRIBUTION_TIMEOUT_LABEL)
    if exit_at_bad == 0:
        return _build_cannot_reproduce(work_root, node_id, bad)

    try:
        good = find_good_commit(
            work_root,
            node_id,
            bad=bad,
            python_exe=python_exe,
            deadline=deadline,
        )
    except LookbackExhausted as exc:
        return _need_human_result(node_id, detail=exc.reason)

    return find_first_bad(
        main_root,
        work_root,
        node_id,
        good=good,
        bad=bad,
        python_exe=python_exe,
        deadline=deadline,
    )


def _attribute_one_worktree(
    repo_root: Path,
    node_id: str,
    *,
    bad: str,
    python_exe: str,
    worktree_parent: Path,
    deadline: float | None = None,
) -> FirstBadResult:
    wt_path = _worktree_path(worktree_parent, node_id)
    _add_worktree(repo_root, wt_path, bad)
    try:
        return _attribute_one_at(
            repo_root,
            wt_path,
            node_id,
            bad=bad,
            python_exe=python_exe,
            deadline=deadline,
        )
    finally:
        if wt_path.exists():
            _run_git(wt_path, "bisect", "reset")
        _remove_worktree(repo_root, wt_path)


def _attribute_serial(
    repo_root: Path,
    node_ids: tuple[str, ...],
    *,
    bad: str,
    python_exe: str,
    deadline: float | None = None,
) -> tuple[FirstBadResult, ...]:
    """Checkout → attribute one node → reset → next (main tree; no worktrees)."""
    results: list[FirstBadResult] = []
    try:
        for index, node_id in enumerate(node_ids):
            if _budget_exhausted(deadline):
                results.extend(_budget_skip_results(node_ids[index:]))
                break
            checkout = _run_git(repo_root, "checkout", "--detach", bad)
            if checkout.returncode != 0:
                logger.warning("serial checkout bad failed: %s", checkout.stderr.strip())
                results.append(_need_human_result(node_id, detail=_NEED_HUMAN_INCOMPLETE))
                continue
            try:
                results.append(
                    _attribute_one_at(
                        repo_root,
                        repo_root,
                        node_id,
                        bad=bad,
                        python_exe=python_exe,
                        deadline=deadline,
                    )
                )
            except Exception:
                logger.exception("serial attribution failed for %s", node_id)
                results.append(_need_human_result(node_id, detail=_NEED_HUMAN_INCOMPLETE))
            reset = _run_git(repo_root, "checkout", "--detach", bad)
            if reset.returncode != 0:
                logger.warning("serial reset to bad failed: %s", reset.stderr.strip())
    finally:
        _run_git(repo_root, "checkout", "--detach", bad)
    return tuple(results)


def _probe_worktree(repo_root: Path, bad: str, parent: Path) -> bool:
    probe = parent / f"{_WORKTREE_NAME_PREFIX}probe"
    try:
        _add_worktree(repo_root, probe, bad)
    except RuntimeError as exc:
        logger.warning("git worktree add failed (%s); falling back to serial attribution", exc)
        return False
    _remove_worktree(repo_root, probe)
    return True


def _attribute_parallel_worktrees(
    repo_root: Path,
    node_ids: tuple[str, ...],
    *,
    bad: str,
    python_exe: str,
    max_workers: int,
    worktree_parent: Path,
    deadline: float | None = None,
) -> tuple[FirstBadResult, ...]:
    results_by_node: dict[str, FirstBadResult] = {}
    workers = max(1, max_workers)
    pending = list(node_ids)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        inflight: dict[Future[FirstBadResult], str] = {}
        while pending or inflight:
            while pending and len(inflight) < workers:
                if _budget_exhausted(deadline):
                    for node_id in pending:
                        results_by_node[node_id] = _need_human_result(
                            node_id,
                            detail=ATTRIBUTION_BUDGET_SKIP_LABEL,
                        )
                    pending.clear()
                    break
                node_id = pending.pop(0)
                future = pool.submit(
                    _attribute_one_worktree,
                    repo_root,
                    node_id,
                    bad=bad,
                    python_exe=python_exe,
                    worktree_parent=worktree_parent,
                    deadline=deadline,
                )
                inflight[future] = node_id
            if not inflight:
                break
            done, _ = wait(inflight, return_when=FIRST_COMPLETED)
            for future in done:
                node_id = inflight.pop(future)
                try:
                    results_by_node[node_id] = future.result()
                except Exception:
                    logger.exception("attribution worker failed for %s", node_id)
                    results_by_node[node_id] = _need_human_result(
                        node_id,
                        detail=_NEED_HUMAN_INCOMPLETE,
                    )
    return tuple(results_by_node[node_id] for node_id in node_ids)


def attribute_failures(
    repo_root: Path,
    failed_nodes: tuple[str, ...],
    *,
    python_exe: str,
    max_workers: int,
    deadline: float | None = None,
) -> tuple[FirstBadResult, ...]:
    """Attribute every failed nodeid; never cancel siblings on lookback miss.

    When *deadline* (``time.monotonic``) is set, stop starting new nodes once
    the budget is exhausted and mark remaining nodes as need-human timeout skips.
    """
    if not failed_nodes:
        return ()
    if _budget_exhausted(deadline):
        normalized_early = tuple(normalize_stdout_node_id(node) for node in failed_nodes)
        return _budget_skip_results(normalized_early)

    bad = _git_stdout(repo_root, "rev-parse", "HEAD")
    if not bad:
        raise RuntimeError("Cannot resolve HEAD for attribution")

    normalized = tuple(normalize_stdout_node_id(node) for node in failed_nodes)

    with tempfile.TemporaryDirectory(prefix="nightly-attrib-wt-") as tmp:
        parent = Path(tmp)
        if _probe_worktree(repo_root, bad, parent):
            return _attribute_parallel_worktrees(
                repo_root,
                normalized,
                bad=bad,
                python_exe=python_exe,
                max_workers=max_workers,
                worktree_parent=parent,
                deadline=deadline,
            )
        return _attribute_serial(
            repo_root,
            normalized,
            bad=bad,
            python_exe=python_exe,
            deadline=deadline,
        )


def results_to_failure_blames(
    results: tuple[FirstBadResult, ...],
    *,
    failure_reasons: dict[str, str] | None = None,
) -> tuple[FailureBlame, ...]:
    """Map attribution results to Feishu/console blame rows."""
    reasons = failure_reasons or {}
    return tuple(
        FailureBlame(
            node_id=result.node_id,
            commit_id=result.commit_id,
            author=result.author,
            subject=result.subject,
            conclusion=result.conclusion,
            last_reason=reasons.get(result.node_id) or result.detail,
        )
        for result in results
    )


if __name__ == "__main__":
    bisect_check()
