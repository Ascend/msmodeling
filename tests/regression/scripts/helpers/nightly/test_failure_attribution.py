"""Tests for nightly.failure_attribution — day-walk, linear/bisect, worktrees, need-human."""

from __future__ import annotations

import subprocess
import time
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from scripts.helpers.nightly import failure_attribution as fa
from scripts.helpers.nightly.failure_attribution import (
    ATTRIBUTION_BUDGET_SKIP_LABEL,
    LINEAR_VS_BISECT_MAX_COMMITS,
    FirstBadResult,
    LookbackExhausted,
    _attribution_skip_exit,
    _bisect_exit_code,
    _commit_metadata,
    attribute_failures,
    default_max_workers,
    find_day_candidate,
    find_first_bad,
    find_good_commit,
    linear_first_bad,
    results_to_failure_blames,
)
from scripts.helpers.nightly.report_models import AttributionConclusion
from tests.helpers.fake_subprocess import FakeCompleted


def test_commit_metadata_parses_git_log_fields(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    sep = "\x1f"

    def _fake_run(repo: object, *args: str, env: object = None) -> FakeCompleted:
        if args and args[0] == "log":
            return FakeCompleted(0, f"abc1234{sep}alice{sep}fix nightly bisect\n", "")
        return FakeCompleted(0, "", "")

    monkeypatch.setattr(
        "scripts.helpers.nightly.failure_attribution._run_git",
        _fake_run,
    )
    commit_id, author, subject = _commit_metadata(tmp_path, "fullsha")
    assert commit_id == "abc1234"
    assert author == "alice"
    assert subject == "fix nightly bisect"


def test_bisect_exit_code_maps_uncollectible_to_skip() -> None:
    assert _bisect_exit_code(0) == 0
    assert _bisect_exit_code(1) == 1
    assert _bisect_exit_code(2) == 125
    assert _bisect_exit_code(5) == 125
    assert _bisect_exit_code(124) == 125


def test_attribution_skip_exit_treats_timeout_as_inconclusive() -> None:
    assert _attribution_skip_exit(0) is False
    assert _attribution_skip_exit(1) is False
    assert _attribution_skip_exit(2) is True
    assert _attribution_skip_exit(5) is True
    assert _attribution_skip_exit(124) is True


def test_linear_first_bad_skips_timeout_without_first_bad(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "scripts.helpers.nightly.failure_attribution._git_stdout",
        lambda _repo, *args: "abc\ndef" if args[0] == "rev-list" else "",
    )
    monkeypatch.setattr(
        "scripts.helpers.nightly.failure_attribution._run_git",
        lambda *_a, **_k: FakeCompleted(0, "", ""),
    )
    monkeypatch.setattr(
        "scripts.helpers.nightly.failure_attribution.run_node_pytest",
        lambda *_a, **_k: 124,
    )
    result = linear_first_bad(
        tmp_path,
        "tests/a.py::test_x",
        good="goodsha",
        bad="badsha",
        python_exe="python",
    )
    assert result.conclusion == AttributionConclusion.NEED_HUMAN
    assert result.commit_id == "unknown"


def test_attribute_one_at_head_timeout_needs_human(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from scripts.helpers.nightly.failure_attribution import _attribute_one_at

    monkeypatch.setattr(
        "scripts.helpers.nightly.failure_attribution.run_node_pytest",
        lambda *_a, **_k: 124,
    )
    result = _attribute_one_at(tmp_path, tmp_path, "tests/a.py::test_x", bad="badsha", python_exe="python")
    assert result.conclusion == AttributionConclusion.NEED_HUMAN
    assert "timed out" in result.detail.lower()


def test_default_max_workers_caps_at_half_cpus(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "scripts.helpers.nightly.failure_attribution.os.cpu_count",
        lambda: 8,
    )
    assert default_max_workers(10) == 4
    assert default_max_workers(2) == 2
    assert default_max_workers(0) == 1


def test_linear_vs_bisect_threshold_is_documented_constant() -> None:
    assert LINEAR_VS_BISECT_MAX_COMMITS == 16


def test_find_day_candidate_returns_last_first_parent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: list[tuple[str, ...]] = []

    def _fake_stdout(repo: object, *args: str) -> str:
        captured.append(args)
        if args[0] == "rev-list":
            return "cafebabe"
        return ""

    monkeypatch.setattr(
        "scripts.helpers.nightly.failure_attribution._git_stdout",
        _fake_stdout,
    )
    tz = ZoneInfo("Asia/Shanghai")
    tip = "deadbeef"
    result = find_day_candidate(tmp_path, day=date(2026, 8, 7), tz=tz, tip=tip)
    assert result == "cafebabe"
    assert captured[0][0] == "rev-list"
    assert "--first-parent" in captured[0]
    assert tip in captured[0]


def test_find_good_commit_day_walk_skips_empty_and_uses_shanghai_previous_day(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    tz = ZoneInfo("Asia/Shanghai")
    now = datetime(2026, 8, 8, 1, 0, tzinfo=tz)
    days_seen: list[date] = []

    def _fake_candidate(repo: object, *, day: date, tz: ZoneInfo, tip: str) -> str | None:
        days_seen.append(day)
        if day == date(2026, 8, 6):
            return "goodsha"
        return None

    checkouts: list[str] = []

    def _fake_run(repo: object, *args: str, env: object = None) -> FakeCompleted:
        if args[:2] == ("checkout", "--detach"):
            checkouts.append(args[2])
            return FakeCompleted(0, "", "")
        return FakeCompleted(0, "", "")

    monkeypatch.setattr(
        "scripts.helpers.nightly.failure_attribution.find_day_candidate",
        _fake_candidate,
    )
    monkeypatch.setattr(
        "scripts.helpers.nightly.failure_attribution._run_git",
        _fake_run,
    )
    monkeypatch.setattr(
        "scripts.helpers.nightly.failure_attribution.run_node_pytest",
        lambda *_a, **_k: 0,
    )
    monkeypatch.setattr(
        "scripts.helpers.nightly.failure_attribution._is_shallow_repo",
        lambda _root: False,
    )

    good = find_good_commit(
        tmp_path,
        "tests/a.py::test_x",
        bad="badsha",
        python_exe="python",
        now=now,
    )
    assert good == "goodsha"
    assert days_seen[0] == date(2026, 8, 7)
    assert date(2026, 8, 6) in days_seen
    assert checkouts == ["goodsha"]


def test_find_good_commit_skips_uncollectible_and_continues(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    tz = ZoneInfo("Asia/Shanghai")
    now = datetime(2026, 8, 8, 12, 0, tzinfo=tz)
    candidates = {
        date(2026, 8, 7): "skipsha",
        date(2026, 8, 6): "goodsha",
    }

    monkeypatch.setattr(
        "scripts.helpers.nightly.failure_attribution.find_day_candidate",
        lambda _repo, *, day, tz, tip: candidates.get(day),
    )
    monkeypatch.setattr(
        "scripts.helpers.nightly.failure_attribution._run_git",
        lambda *_a, **_k: FakeCompleted(0, "", ""),
    )

    calls: list[str] = []

    def _run_tracked(root: object, node: str, *, python_exe: str, timeout_seconds: float | None = None) -> int:
        del root, node, python_exe, timeout_seconds
        calls.append("run")
        return 2 if len(calls) == 1 else 0

    monkeypatch.setattr(
        "scripts.helpers.nightly.failure_attribution.run_node_pytest",
        _run_tracked,
    )
    monkeypatch.setattr(
        "scripts.helpers.nightly.failure_attribution._is_shallow_repo",
        lambda _root: False,
    )

    good = find_good_commit(
        tmp_path,
        "tests/a.py::test_x",
        bad="badsha",
        python_exe="python",
        now=now,
    )
    assert good == "goodsha"
    assert calls == ["run", "run"]


def test_find_good_commit_raises_lookback_exhausted_after_seven_bad_days(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    tz = ZoneInfo("Asia/Shanghai")
    now = datetime(2026, 8, 8, 12, 0, tzinfo=tz)

    monkeypatch.setattr(
        "scripts.helpers.nightly.failure_attribution.find_day_candidate",
        lambda _repo, *, day, tz, tip: f"sha-{day.isoformat()}",
    )
    monkeypatch.setattr(
        "scripts.helpers.nightly.failure_attribution._run_git",
        lambda *_a, **_k: FakeCompleted(0, "", ""),
    )
    monkeypatch.setattr(
        "scripts.helpers.nightly.failure_attribution.run_node_pytest",
        lambda *_a, **_k: 1,
    )
    monkeypatch.setattr(
        "scripts.helpers.nightly.failure_attribution._is_shallow_repo",
        lambda _root: False,
    )

    with pytest.raises(LookbackExhausted) as exc_info:
        find_good_commit(
            tmp_path,
            "tests/a.py::test_x",
            bad="badsha",
            python_exe="python",
            now=now,
        )
    assert exc_info.value.lookback_days == 7
    assert exc_info.value.node_id == "tests/a.py::test_x"
    assert "needs human follow-up" in exc_info.value.reason


def test_results_to_failure_blames_prefers_pytest_reason() -> None:
    results = (
        FirstBadResult(
            node_id="tests/a.py::test_x",
            commit_id="abc",
            author="alice",
            subject="fix test",
            conclusion=AttributionConclusion.FIRST_BAD,
            detail="Introducing commit abc",
        ),
    )
    blames = results_to_failure_blames(
        results,
        failure_reasons={"tests/a.py::test_x": "AssertionError: boom"},
    )
    assert blames[0].last_reason == "AssertionError: boom"


def test_find_first_bad_uses_linear_when_count_small(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "scripts.helpers.nightly.failure_attribution._count_commits",
        lambda *_a, **_k: 3,
    )
    called: list[str] = []

    def _linear(*_a: object, **_k: object) -> FirstBadResult:
        called.append("linear")
        return FirstBadResult(
            node_id="tests/a.py::test_x",
            commit_id="abc",
            author="a",
            subject="s",
            conclusion=AttributionConclusion.FIRST_BAD,
            good_commit="good",
        )

    def _bisect(*_a: object, **_k: object) -> FirstBadResult:
        called.append("bisect")
        raise AssertionError("bisect should not run")

    monkeypatch.setattr("scripts.helpers.nightly.failure_attribution.linear_first_bad", _linear)
    monkeypatch.setattr("scripts.helpers.nightly.failure_attribution.bisect_first_bad", _bisect)
    result = find_first_bad(
        tmp_path,
        tmp_path,
        "tests/a.py::test_x",
        good="good",
        bad="bad",
        python_exe="python",
    )
    assert called == ["linear"]
    assert result.attributed


def test_find_first_bad_uses_bisect_when_count_large(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "scripts.helpers.nightly.failure_attribution._count_commits",
        lambda *_a, **_k: LINEAR_VS_BISECT_MAX_COMMITS + 1,
    )
    called: list[str] = []

    def _linear(*_a: object, **_k: object) -> FirstBadResult:
        called.append("linear")
        raise AssertionError("linear should not run")

    def _bisect(*_a: object, **_k: object) -> FirstBadResult:
        called.append("bisect")
        return FirstBadResult(
            node_id="tests/a.py::test_x",
            commit_id="abc",
            author="a",
            subject="s",
            conclusion=AttributionConclusion.FIRST_BAD,
            good_commit="good",
        )

    monkeypatch.setattr("scripts.helpers.nightly.failure_attribution.linear_first_bad", _linear)
    monkeypatch.setattr("scripts.helpers.nightly.failure_attribution.bisect_first_bad", _bisect)
    result = find_first_bad(
        tmp_path,
        tmp_path,
        "tests/a.py::test_x",
        good="good",
        bad="bad",
        python_exe="python",
    )
    assert called == ["bisect"]
    assert result.attributed


def test_attribute_failures_creates_one_worktree_per_node_and_keeps_main_head(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    main_head_reads: list[str] = []
    worktrees_added: list[str] = []
    worktrees_removed: list[str] = []

    def _fake_stdout(repo: object, *args: str) -> str:
        if args[:2] == ("rev-parse", "HEAD"):
            main_head_reads.append("HEAD")
            return "badsha"
        if args[0] == "log":
            return "badsha12\x1falice\x1fsubject"
        return ""

    def _fake_run(repo: object, *args: str, env: object = None) -> FakeCompleted:
        if args[:2] == ("worktree", "add"):
            worktrees_added.append(args[3])
            Path(args[3]).mkdir(parents=True, exist_ok=True)
            return FakeCompleted(0, "", "")
        if args[:2] == ("worktree", "remove"):
            worktrees_removed.append(args[-1])
            return FakeCompleted(0, "", "")
        if args[:2] == ("worktree", "prune"):
            return FakeCompleted(0, "", "")
        if args[0] == "bisect":
            return FakeCompleted(0, "", "")
        if args[:2] == ("checkout", "--detach"):
            return FakeCompleted(0, "", "")
        if args[0] == "log":
            return FakeCompleted(0, "badsha12\x1falice\x1fsubject\n", "")
        if args[:2] == ("rev-parse", "HEAD"):
            return FakeCompleted(0, "badsha\n", "")
        return FakeCompleted(0, "", "")

    monkeypatch.setattr(
        "scripts.helpers.nightly.failure_attribution._git_stdout",
        _fake_stdout,
    )
    monkeypatch.setattr(
        "scripts.helpers.nightly.failure_attribution._run_git",
        _fake_run,
    )
    monkeypatch.setattr(
        "scripts.helpers.nightly.failure_attribution.run_node_pytest",
        lambda *_a, **_k: 0,
    )

    nodes = ("tests/a.py::test_x", "tests/b.py::test_y")
    results = attribute_failures(
        tmp_path,
        nodes,
        python_exe="python",
        max_workers=2,
    )
    assert len(results) == 2
    assert all(r.conclusion == AttributionConclusion.CANNOT_REPRODUCE for r in results)
    assert all(r.subject == "Flaky / not reproduced at HEAD" for r in results)
    # probe + 2 nodes
    assert len(worktrees_added) >= 2
    assert main_head_reads


def test_attribute_failures_lookback_miss_does_not_cancel_siblings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "scripts.helpers.nightly.failure_attribution._git_stdout",
        lambda *_a, **_k: "badsha",
    )
    monkeypatch.setattr(
        "scripts.helpers.nightly.failure_attribution._probe_worktree",
        lambda *_a, **_k: True,
    )
    monkeypatch.setattr(
        "scripts.helpers.nightly.failure_attribution._add_worktree",
        lambda _root, path, _commit: path.mkdir(parents=True, exist_ok=True),
    )
    monkeypatch.setattr(
        "scripts.helpers.nightly.failure_attribution._remove_worktree",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        "scripts.helpers.nightly.failure_attribution.run_node_pytest",
        lambda *_a, **_k: 1,
    )

    def _find_good(worktree: object, node_id: str, **_k: object) -> str:
        if node_id.endswith("test_x"):
            raise LookbackExhausted(node_id, bad="badsha", lookback_days=7)
        return "goodsha"

    monkeypatch.setattr(
        "scripts.helpers.nightly.failure_attribution.find_good_commit",
        _find_good,
    )
    monkeypatch.setattr(
        "scripts.helpers.nightly.failure_attribution.find_first_bad",
        lambda *_a, **_k: FirstBadResult(
            node_id="tests/b.py::test_y",
            commit_id="abc1234",
            author="bob",
            subject="fix",
            conclusion=AttributionConclusion.FIRST_BAD,
            good_commit="goodsha",
        ),
    )

    results = attribute_failures(
        tmp_path,
        ("tests/a.py::test_x", "tests/b.py::test_y"),
        python_exe="python",
        max_workers=2,
    )
    by_node = {r.node_id: r for r in results}
    assert by_node["tests/a.py::test_x"].needs_human
    assert by_node["tests/b.py::test_y"].attributed


def test_attribute_failures_falls_back_serial_when_worktree_probe_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(
        "scripts.helpers.nightly.failure_attribution._git_stdout",
        lambda *_a, **_k: "badsha",
    )
    monkeypatch.setattr(
        "scripts.helpers.nightly.failure_attribution._probe_worktree",
        lambda *_a, **_k: False,
    )

    serial_called: list[str] = []

    def _serial(repo: object, nodes: tuple[str, ...], **_k: object) -> tuple[FirstBadResult, ...]:
        serial_called.extend(nodes)
        return tuple(
            FirstBadResult(
                node_id=node,
                commit_id="unknown",
                author="unknown",
                subject="Flaky / not reproduced at HEAD",
                conclusion=AttributionConclusion.CANNOT_REPRODUCE,
            )
            for node in nodes
        )

    monkeypatch.setattr(
        "scripts.helpers.nightly.failure_attribution._attribute_serial",
        _serial,
    )
    with caplog.at_level("WARNING"):
        results = attribute_failures(
            tmp_path,
            ("tests/a.py::test_x",),
            python_exe="python",
            max_workers=1,
        )
    assert serial_called == ["tests/a.py::test_x"]
    assert results[0].conclusion == AttributionConclusion.CANNOT_REPRODUCE


def test_run_node_pytest_timeout_returns_timeout_exit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def _hang(*_a: object, **_k: object) -> object:
        raise subprocess.TimeoutExpired(cmd=["pytest"], timeout=1)

    monkeypatch.setattr(fa.subprocess, "run", _hang)
    assert fa.run_node_pytest(tmp_path, "tests/a.py::test_x", python_exe="python") == 124


def test_attribute_failures_exhausted_deadline_skips_all_nodes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "scripts.helpers.nightly.failure_attribution._git_stdout",
        lambda *_a, **_k: "badsha",
    )
    monkeypatch.setattr(
        "scripts.helpers.nightly.failure_attribution._probe_worktree",
        lambda *_a, **_k: False,
    )
    pytest_calls: list[str] = []

    def _pytest(_repo: object, node_id: str, **_k: object) -> int:
        pytest_calls.append(node_id)
        return 0

    monkeypatch.setattr(
        "scripts.helpers.nightly.failure_attribution.run_node_pytest",
        _pytest,
    )
    monkeypatch.setattr(
        "scripts.helpers.nightly.failure_attribution._run_git",
        lambda *_a, **_k: FakeCompleted(0, "", ""),
    )

    nodes = ("tests/a.py::test_x", "tests/b.py::test_y")
    results = attribute_failures(
        tmp_path,
        nodes,
        python_exe="python",
        max_workers=1,
        deadline=time.monotonic() - 1.0,
    )
    assert pytest_calls == []
    assert len(results) == 2
    assert all(r.conclusion == AttributionConclusion.NEED_HUMAN for r in results)
    assert all(r.subject == ATTRIBUTION_BUDGET_SKIP_LABEL for r in results)


def test_attribute_failures_deadline_stops_before_later_serial_nodes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "scripts.helpers.nightly.failure_attribution._git_stdout",
        lambda *_a, **_k: "badsha",
    )
    monkeypatch.setattr(
        "scripts.helpers.nightly.failure_attribution._probe_worktree",
        lambda *_a, **_k: False,
    )
    monkeypatch.setattr(
        "scripts.helpers.nightly.failure_attribution._run_git",
        lambda *_a, **_k: FakeCompleted(0, "", ""),
    )

    clock = {"now": 1000.0}

    def _mono() -> float:
        return clock["now"]

    pytest_calls: list[str] = []

    def _pytest(_repo: object, node_id: str, **_k: object) -> int:
        pytest_calls.append(node_id)
        clock["now"] += 10.0
        return 0

    monkeypatch.setattr(time, "monotonic", _mono)
    monkeypatch.setattr(
        "scripts.helpers.nightly.failure_attribution.time.monotonic",
        _mono,
    )
    monkeypatch.setattr(
        "scripts.helpers.nightly.failure_attribution.run_node_pytest",
        _pytest,
    )

    results = attribute_failures(
        tmp_path,
        ("tests/a.py::test_x", "tests/b.py::test_y", "tests/c.py::test_z"),
        python_exe="python",
        max_workers=1,
        deadline=1005.0,
    )
    assert pytest_calls == ["tests/a.py::test_x"]
    assert results[0].conclusion == AttributionConclusion.CANNOT_REPRODUCE
    assert results[1].subject == ATTRIBUTION_BUDGET_SKIP_LABEL
    assert results[2].subject == ATTRIBUTION_BUDGET_SKIP_LABEL


def test_find_good_commit_raises_budget_skip_when_deadline_exhausted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "scripts.helpers.nightly.failure_attribution.find_day_candidate",
        lambda *_a, **_k: "cand",
    )
    pytest_calls: list[str] = []
    monkeypatch.setattr(
        "scripts.helpers.nightly.failure_attribution.run_node_pytest",
        lambda *_a, **_k: pytest_calls.append("x") or 1,
    )
    with pytest.raises(LookbackExhausted) as exc_info:
        find_good_commit(
            tmp_path,
            "tests/a.py::test_x",
            bad="badsha",
            python_exe="python",
            deadline=time.monotonic() - 1.0,
        )
    assert exc_info.value.reason == ATTRIBUTION_BUDGET_SKIP_LABEL
    assert pytest_calls == []


def test_linear_first_bad_stops_when_deadline_exhausted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "scripts.helpers.nightly.failure_attribution._git_stdout",
        lambda _repo, *args: "c1\nc2\nc3" if args[0] == "rev-list" else "",
    )
    monkeypatch.setattr(
        "scripts.helpers.nightly.failure_attribution._run_git",
        lambda *_a, **_k: FakeCompleted(0, "", ""),
    )
    clock = {"now": 1000.0}
    monkeypatch.setattr(
        "scripts.helpers.nightly.failure_attribution.time.monotonic",
        lambda: clock["now"],
    )
    pytest_calls: list[str] = []

    def _pytest(_repo: object, node_id: str, **_k: object) -> int:
        del node_id
        pytest_calls.append("run")
        clock["now"] += 10.0
        return 0

    monkeypatch.setattr(
        "scripts.helpers.nightly.failure_attribution.run_node_pytest",
        _pytest,
    )
    result = linear_first_bad(
        tmp_path,
        "tests/a.py::test_x",
        good="good",
        bad="bad",
        python_exe="python",
        deadline=1005.0,
    )
    assert pytest_calls == ["run"]
    assert result.subject == ATTRIBUTION_BUDGET_SKIP_LABEL
    assert result.needs_human


def test_first_bad_result_includes_good_commit_field() -> None:
    result = FirstBadResult(
        node_id="tests/a.py::test_x",
        commit_id="abc",
        author="a",
        subject="s",
        conclusion=AttributionConclusion.FIRST_BAD,
        good_commit="good",
    )
    assert result.good_commit == "good"
    assert result.attributed
