"""Tests for nightly progress journal and resume skip set."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING

from scripts.helpers.nightly import progress_journal
from scripts.helpers.nightly.progress_journal import (
    apply_slow_first,
    compute_session_identity,
    estimate_wave_a_remaining,
    load_completed_node_ids,
    load_slow_first_seed,
    prepare_progress_session,
    resolve_wave_a_worker_count,
    session_matches,
    write_duration_history,
)

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _report(**kwargs: object) -> SimpleNamespace:
    defaults: dict[str, object] = {
        "nodeid": "tests/a.py::test_ok",
        "when": "call",
        "outcome": "passed",
        "failed": False,
        "skipped": False,
        "passed": True,
        "duration": 0.5,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_completed_node_ids_ignore_malformed_and_keep_terminal_only(tmp_path: Path) -> None:
    journal = tmp_path / "progress.jsonl"
    journal.write_text(
        "\n".join(
            [
                "{not-json",
                json.dumps({"node_id": "tests/a.py::test_pass", "outcome": "passed"}),
                json.dumps({"node_id": "tests/a.py::test_fail", "outcome": "failed"}),
                json.dumps({"node_id": "tests/a.py::inflight", "outcome": ""}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    assert load_completed_node_ids(journal) == {
        "tests/a.py::test_pass",
        "tests/a.py::test_fail",
    }


def test_session_matches_requires_all_fingerprint_fields() -> None:
    base = {
        "head": "abc",
        "dirty_sha256": "d1",
        "wave_a_marker": "not npu and not benchmark and not network",
        "wave_b_marker": "not npu and (benchmark or network)",
        "dep_sha256": "dep",
    }
    assert session_matches(base, dict(base))
    changed = dict(base)
    changed["head"] = "def"
    assert not session_matches(base, changed)


def test_prepare_progress_session_refuses_mismatch_and_wipes(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    logger = __import__("logging").getLogger("progress-test")
    progress = tmp_path / ".pytest_cache/nightly/progress.non-benchmark.jsonl"
    progress.parent.mkdir(parents=True)
    progress.write_text(
        json.dumps({"node_id": "tests/a.py::old", "outcome": "passed"}) + "\n",
        encoding="utf-8",
    )
    session = tmp_path / ".pytest_cache/nightly/progress.session.json"
    session.write_text(json.dumps({"head": "other"}), encoding="utf-8")

    with caplog.at_level(__import__("logging").WARNING):
        active = prepare_progress_session(tmp_path, resume=True, logger=logger)

    assert active is False
    assert not progress.exists()
    saved = json.loads(session.read_text(encoding="utf-8"))
    assert saved["head"] == compute_session_identity(tmp_path)["head"]
    assert "resume refused" in caplog.text


def test_prepare_progress_session_resumes_when_fingerprint_matches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger = __import__("logging").getLogger("progress-test")
    identity = compute_session_identity(tmp_path)
    session = tmp_path / ".pytest_cache/nightly/progress.session.json"
    session.parent.mkdir(parents=True)
    session.write_text(json.dumps(identity), encoding="utf-8")
    progress = tmp_path / ".pytest_cache/nightly/progress.non-benchmark.jsonl"
    progress.write_text(
        json.dumps({"node_id": "tests/a.py::kept", "outcome": "passed"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(progress_journal, "compute_session_identity", lambda _repo: identity)

    active = prepare_progress_session(tmp_path, resume=True, logger=logger)
    assert active is True
    assert progress.exists()


def test_call_failure_is_written_immediately(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    journal = tmp_path / "progress.jsonl"
    monkeypatch.setenv("MSMODELING_NIGHTLY_PROGRESS_JOURNAL", str(journal))
    monkeypatch.delenv("MSMODELING_NIGHTLY_RESUME_ACTIVE", raising=False)
    progress_journal.pytest_configure(SimpleNamespace(option=SimpleNamespace(numprocesses=None)))
    progress_journal.pytest_runtest_logreport(
        _report(
            nodeid="tests/a.py::test_broken",
            when="call",
            outcome="failed",
            failed=True,
            passed=False,
        )
    )
    event = json.loads(journal.read_text(encoding="utf-8"))
    assert event["node_id"] == "tests/a.py::test_broken"
    assert event["outcome"] == "failed"
    assert event["phase"] == "call"


def test_passed_is_written_only_after_teardown(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    journal = tmp_path / "progress.jsonl"
    monkeypatch.setenv("MSMODELING_NIGHTLY_PROGRESS_JOURNAL", str(journal))
    progress_journal.pytest_configure(SimpleNamespace(option=SimpleNamespace(numprocesses=None)))
    progress_journal.pytest_runtest_logreport(_report(when="setup", outcome="passed"))
    progress_journal.pytest_runtest_logreport(_report(when="call", outcome="passed"))
    assert not journal.exists()
    progress_journal.pytest_runtest_logreport(_report(when="teardown", outcome="passed", duration=0.01))
    event = json.loads(journal.read_text(encoding="utf-8"))
    assert event["outcome"] == "passed"
    assert event["phase"] == "teardown"
    assert event["duration_sec"] == 0.5


def test_killed_after_call_pass_is_not_completed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    journal = tmp_path / "progress.jsonl"
    monkeypatch.setenv("MSMODELING_NIGHTLY_PROGRESS_JOURNAL", str(journal))
    progress_journal.pytest_configure(SimpleNamespace(option=SimpleNamespace(numprocesses=None)))
    progress_journal.pytest_runtest_logreport(_report(when="call", outcome="passed"))
    assert load_completed_node_ids(journal) == set()


def test_collection_modifyitems_deselects_completed_ids(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    journal = tmp_path / "progress.jsonl"
    journal.write_text(
        json.dumps({"node_id": "tests/a.py::done", "outcome": "passed"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("MSMODELING_NIGHTLY_PROGRESS_JOURNAL", str(journal))
    monkeypatch.setenv("MSMODELING_NIGHTLY_RESUME_ACTIVE", "1")
    progress_journal.pytest_configure(SimpleNamespace(option=SimpleNamespace(numprocesses=None)))

    deselected: list[object] = []
    config = SimpleNamespace(hook=SimpleNamespace(pytest_deselected=lambda items: deselected.extend(items)))
    items = [
        SimpleNamespace(nodeid="tests/a.py::done"),
        SimpleNamespace(nodeid="tests/a.py::todo"),
    ]
    progress_journal.pytest_collection_modifyitems(config, items)
    assert [item.nodeid for item in items] == ["tests/a.py::todo"]
    assert [item.nodeid for item in deselected] == ["tests/a.py::done"]


def test_apply_slow_first_moves_only_known_slow_ids_to_front() -> None:
    items = [
        SimpleNamespace(nodeid="tests/a.py::fast"),
        SimpleNamespace(nodeid="tests/a.py::unknown"),
        SimpleNamespace(nodeid="tests/a.py::slow"),
        SimpleNamespace(nodeid="tests/a.py::mid"),
    ]
    moved = apply_slow_first(items, {"tests/a.py::slow": 30.0, "tests/a.py::mid": 5.0, "tests/a.py::fast": 0.1}, 2)
    assert moved == ["tests/a.py::slow", "tests/a.py::mid"]
    assert [item.nodeid for item in items] == [
        "tests/a.py::slow",
        "tests/a.py::mid",
        "tests/a.py::fast",
        "tests/a.py::unknown",
    ]


def test_apply_slow_first_uses_committed_seed_when_history_is_empty() -> None:
    items = [
        SimpleNamespace(nodeid="tests/fast.py::t"),
        SimpleNamespace(nodeid="tests/regression/tensor_cast/test_kimi_k3.py::TestKimiK3::test_kimi_k3_text_prefill"),
        SimpleNamespace(
            nodeid="tests/regression/tensor_cast/test_deepseek_v4.py::TestDeepseekV4ModelNightly::test_model_initialization"
        ),
        SimpleNamespace(nodeid="tests/other.py::t"),
    ]
    moved = apply_slow_first(
        items,
        {},
        2,
        seed=(
            "tests/regression/tensor_cast/test_deepseek_v4.py::TestDeepseekV4ModelNightly::test_model_initialization",
            "tests/regression/tensor_cast/test_kimi_k3.py::TestKimiK3",
        ),
    )
    assert moved == [
        "tests/regression/tensor_cast/test_deepseek_v4.py::TestDeepseekV4ModelNightly::test_model_initialization",
        "tests/regression/tensor_cast/test_kimi_k3.py::TestKimiK3::test_kimi_k3_text_prefill",
    ]
    assert items[0].nodeid.endswith("test_model_initialization")


def test_apply_slow_first_prefers_history_over_seed() -> None:
    items = [
        SimpleNamespace(nodeid="tests/a.py::seeded"),
        SimpleNamespace(nodeid="tests/a.py::measured"),
    ]
    moved = apply_slow_first(
        items,
        {"tests/a.py::measured": 40.0},
        1,
        seed=("tests/a.py::seeded",),
    )
    assert moved == ["tests/a.py::measured"]


def test_load_slow_first_seed_skips_comments(tmp_path: Path) -> None:
    path = tmp_path / "seed.txt"
    path.write_text("# comment\n\ntests/slow.py::case\n", encoding="utf-8")
    assert load_slow_first_seed(path) == ("tests/slow.py::case",)


def test_collection_modifyitems_slow_first_after_resume_deselect(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    journal = tmp_path / "progress.jsonl"
    journal.write_text(
        json.dumps({"node_id": "tests/a.py::done", "outcome": "passed"}) + "\n",
        encoding="utf-8",
    )
    write_duration_history(
        tmp_path / "duration-history.json",
        {"tests/a.py::slow": 12.0, "tests/a.py::fast": 0.2},
    )
    monkeypatch.setenv("MSMODELING_NIGHTLY_PROGRESS_JOURNAL", str(journal))
    monkeypatch.setenv("MSMODELING_NIGHTLY_RESUME_ACTIVE", "1")
    monkeypatch.setenv("MSMODELING_NIGHTLY_SLOW_FIRST", "1")
    progress_journal.pytest_configure(SimpleNamespace(option=SimpleNamespace(numprocesses=None)))
    items = [
        SimpleNamespace(nodeid="tests/a.py::fast"),
        SimpleNamespace(nodeid="tests/a.py::done"),
        SimpleNamespace(nodeid="tests/a.py::slow"),
    ]
    config = SimpleNamespace(hook=SimpleNamespace(pytest_deselected=lambda items: None))
    progress_journal.pytest_collection_modifyitems(config, items)
    assert [item.nodeid for item in items] == ["tests/a.py::slow", "tests/a.py::fast"]


def test_xdist_worker_configure_enables_slow_first_reorder(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    journal = tmp_path / "progress.non-benchmark.jsonl"
    journal.write_text("", encoding="utf-8")
    write_duration_history(tmp_path / "duration-history.json", {"tests/a.py::slow": 12.0})
    monkeypatch.setenv("MSMODELING_NIGHTLY_PROGRESS_JOURNAL", str(journal))
    monkeypatch.delenv("MSMODELING_NIGHTLY_RESUME_ACTIVE", raising=False)
    monkeypatch.setenv("MSMODELING_NIGHTLY_SLOW_FIRST", "1")
    progress_journal.pytest_configure(
        SimpleNamespace(
            workerinput={"workerid": "gw0"},
            option=SimpleNamespace(numprocesses=8),
        )
    )
    items = [
        SimpleNamespace(nodeid="tests/a.py::fast"),
        SimpleNamespace(nodeid="tests/a.py::slow"),
    ]
    progress_journal.pytest_collection_modifyitems(SimpleNamespace(), items)
    assert [item.nodeid for item in items] == ["tests/a.py::slow", "tests/a.py::fast"]
    session = json.loads((tmp_path / "progress.session.json").read_text(encoding="utf-8"))
    assert session["wave_a_collected_count"] == "2"


def test_xdist_non_primary_worker_does_not_overwrite_wave_a_count(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    journal = tmp_path / "progress.non-benchmark.jsonl"
    session = tmp_path / "progress.session.json"
    session.write_text(json.dumps({"wave_a_collected_count": "8290"}), encoding="utf-8")
    monkeypatch.setenv("MSMODELING_NIGHTLY_PROGRESS_JOURNAL", str(journal))
    progress_journal.pytest_configure(
        SimpleNamespace(
            workerinput={"workerid": "gw1"},
            option=SimpleNamespace(numprocesses=8),
        )
    )

    progress_journal.pytest_collection_modifyitems(
        SimpleNamespace(),
        [SimpleNamespace(nodeid="tests/a.py::one"), SimpleNamespace(nodeid="tests/a.py::two")],
    )

    saved = json.loads(session.read_text(encoding="utf-8"))
    assert saved["wave_a_collected_count"] == "8290"


def test_wave_a_remaining_is_unknown_without_exact_collection_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MSMODELING_NIGHTLY_XDIST_WORKERS", "128")
    monkeypatch.setattr(progress_journal.os, "cpu_count", lambda: 256)
    nightly = tmp_path / ".pytest_cache/nightly"
    nightly.mkdir(parents=True)
    write_duration_history(
        nightly / "duration-history.json",
        {
            "tests/wave_a.py::done": 1.0,
            "tests/wave_a.py::pending": 2.0,
            "tests/wave_b.py::pending": 3.0,
        },
    )
    (nightly / "progress.non-benchmark.jsonl").write_text(
        json.dumps({"node_id": "tests/wave_a.py::done", "outcome": "passed"}) + "\n",
        encoding="utf-8",
    )

    assert estimate_wave_a_remaining(tmp_path) is None
    assert resolve_wave_a_worker_count(tmp_path, resume_active=True) == 128


def test_wave_a_worker_count_caps_and_shrinks_on_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MSMODELING_NIGHTLY_XDIST_WORKERS", "160")
    monkeypatch.setattr(progress_journal.os, "cpu_count", lambda: 640)
    nightly = tmp_path / ".pytest_cache/nightly"
    nightly.mkdir(parents=True)
    (nightly / "progress.session.json").write_text(
        json.dumps({"wave_a_collected_count": "8290"}),
        encoding="utf-8",
    )
    (nightly / "progress.non-benchmark.jsonl").write_text(
        "\n".join(json.dumps({"node_id": f"tests/a.py::t{i}", "outcome": "passed"}) for i in range(8278)) + "\n",
        encoding="utf-8",
    )
    assert resolve_wave_a_worker_count(tmp_path, resume_active=False) == 160
    assert estimate_wave_a_remaining(tmp_path) == 12
    assert resolve_wave_a_worker_count(tmp_path, resume_active=True) == 12
