"""Tests for ci_gate.sync — target branch resolution and incremental map merge."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from _pytest.monkeypatch import MonkeyPatch

from scripts.helpers._config import Config, ConfigError
from scripts.helpers.ci_gate import sync
from scripts.helpers.ci_gate.diff import resolve_ref_commit
from tests.helpers.fake_subprocess import FakeCompleted

_DEFAULT_CFG = Config(
    test_map_path="/tmp/test_map.json",
    base_branch="master",
    line_threshold=60.0,
    branch_threshold=40.0,
    benchmark_parallel=False,
    feishu_webhook_url="",
    msmodeling_cache=".msmodeling_cache",
    weights_prune=False,
)


def test_resolve_target_branch_prefers_cli() -> None:
    assert sync.resolve_target_branch(cli_target="origin/develop", cfg=_DEFAULT_CFG) == "origin/develop"


def test_resolve_target_branch_uses_env(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("MSMODELING_TEST_MAP_TARGET_BRANCH", "develop")
    assert sync.resolve_target_branch(cli_target=None, cfg=_DEFAULT_CFG) == "develop"


def test_resolve_target_branch_falls_back_to_base_branch(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.delenv("MSMODELING_TEST_MAP_TARGET_BRANCH", raising=False)
    assert sync.resolve_target_branch(cli_target=None, cfg=_DEFAULT_CFG) == "master"


def test_resolve_ref_commit_returns_sha(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: FakeCompleted(0, "abc123deadbeef\n", ""))
    assert resolve_ref_commit(tmp_path, "master") == "abc123deadbeef"


def test_can_incremental_sync_blocks_non_ancestor(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("scripts.helpers.ci_gate.diff.is_git_ancestor", lambda *_a, **_k: False)
    assert sync.can_incremental_sync(tmp_path, "oldcommit111", "newcommit222") is False


def test_can_incremental_sync_allows_up_to_date() -> None:
    assert sync.can_incremental_sync(Path("/tmp"), "samecommit000", "samecommit000") is True


def test_ephemeral_target_checkout_creates_and_cleans_branch(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def _fake_run(cmd: list[str], **_kwargs: object) -> FakeCompleted:
        calls.append(cmd)
        if cmd[1:3] == ["symbolic-ref", "--short"]:
            return FakeCompleted(0, "main\n", "")
        if cmd[1:3] == ["rev-parse", "HEAD"]:
            return FakeCompleted(0, "savedsha000\n", "")
        return FakeCompleted(0, "", "")

    monkeypatch.setattr("subprocess.run", _fake_run)
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.diff.resolve_target_head",
        lambda *_a, **_k: "targetsha111",
    )
    from scripts.helpers.ci_gate.diff import ephemeral_target_checkout

    with ephemeral_target_checkout(tmp_path, "develop") as target_head:
        assert target_head == "targetsha111"
        assert any(cmd[1:3] == ["checkout", "-B"] for cmd in calls)
    assert any(cmd[1:3] == ["checkout", "main"] for cmd in calls)
    assert any(cmd[1:3] == ["branch", "-D"] for cmd in calls)


def test_apply_incremental_test_map_update_replaces_touched_source_paths(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("scripts.helpers.common.build_test_map.REPO_ROOT", tmp_path)
    (tmp_path / "cli").mkdir()
    (tmp_path / "cli" / "main.py").write_text("x", encoding="utf-8")

    existing = {
        "tests/regression/cli/test_run.py::test_run": {"cli/main.py": ["run"]},
        "tests/smoke/test_bar.py::test_x": {"tensor_cast/foo.py": ["bar"]},
    }
    fresh = {
        "tests/regression/cli/test_run.py::test_run": {"cli/main.py": ["run_new"]},
        "tests/smoke/test_baz.py::test_z": {"tensor_cast/baz.py": ["baz"]},
    }
    updated = sync.apply_incremental_test_map_update(
        existing,
        fresh,
        frozenset({"cli/main.py", "tensor_cast/foo.py"}),
    )
    assert updated["tests/regression/cli/test_run.py::test_run"]["cli/main.py"] == ["run_new"]
    assert "tensor_cast/foo.py" not in updated.get("tests/smoke/test_bar.py::test_x", {})
    assert "tests/smoke/test_baz.py::test_z" not in updated


def test_apply_incremental_test_map_update_replaces_touched_test_file_nodes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("scripts.helpers.common.build_test_map.REPO_ROOT", tmp_path)
    (tmp_path / "cli").mkdir()
    (tmp_path / "cli" / "main.py").write_text("x", encoding="utf-8")
    (tmp_path / "cli" / "util.py").write_text("x", encoding="utf-8")
    existing = {
        "tests/regression/cli/test_run.py::test_run": {"cli/main.py": ["run"]},
    }
    fresh = {
        "tests/regression/cli/test_run.py::test_run": {
            "cli/main.py": ["run_new"],
            "cli/util.py": ["helper"],
        },
    }
    updated = sync.apply_incremental_test_map_update(
        existing,
        fresh,
        frozenset({"tests/regression/cli/test_run.py"}),
    )
    assert updated["tests/regression/cli/test_run.py::test_run"] == {
        "cli/main.py": ["run_new"],
        "cli/util.py": ["helper"],
    }


def test_apply_incremental_test_map_update_drops_removed_test_nodes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("scripts.helpers.common.build_test_map.REPO_ROOT", tmp_path)
    (tmp_path / "cli").mkdir()
    (tmp_path / "cli" / "main.py").write_text("x", encoding="utf-8")
    existing = {
        "tests/regression/cli/test_run.py::test_run": {"cli/main.py": ["run"]},
        "tests/regression/cli/test_run.py::test_old": {"cli/main.py": ["legacy"]},
    }
    fresh = {
        "tests/regression/cli/test_run.py::test_run": {"cli/main.py": ["run"]},
    }
    updated = sync.apply_incremental_test_map_update(
        existing,
        fresh,
        frozenset({"tests/regression/cli/test_run.py"}),
    )
    assert "tests/regression/cli/test_run.py::test_run" in updated
    assert "tests/regression/cli/test_run.py::test_old" not in updated


def test_sync_test_map_once_up_to_date_skips_pytest(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    map_path = tmp_path / "test_map.json"
    map_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "built_from_commit": "abc123",
                "map": {
                    "tests/regression/cli/test_run.py::test_run": {"cli/main.py": ["run"]},
                },
            }
        ),
        encoding="utf-8",
    )
    cfg = Config(
        test_map_path=str(map_path),
        base_branch="master",
        line_threshold=60.0,
        branch_threshold=40.0,
        benchmark_parallel=False,
        feishu_webhook_url="",
        msmodeling_cache=".msmodeling_cache",
        weights_prune=False,
    )
    monkeypatch.setattr("scripts.helpers.ci_gate.sync.resolve_target_head", lambda *_a, **_k: "abc123")
    pytest_ran = False

    def _fail_pytest(*_args: object, **_kwargs: object) -> int:
        nonlocal pytest_ran
        pytest_ran = True
        return 1

    monkeypatch.setattr(sync, "run_test_map_pytest", _fail_pytest)
    with caplog.at_level("INFO"):
        exit_code = sync.sync_test_map_once(cfg, target_branch="master", logger=logging.getLogger("test"))
    assert exit_code == 0
    assert pytest_ran is False
    assert "up to date" in caplog.text


def test_sync_test_map_watch_continues_after_sync_failure(
    monkeypatch: MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    sync._shutdown_flag[0] = False
    slept = False

    def _fail_once(*_args: object, **_kwargs: object) -> int:
        return 1

    def _sleep_then_stop(_seconds: float) -> None:
        nonlocal slept
        slept = True
        sync._shutdown_flag[0] = True

    monkeypatch.setattr(sync, "sync_test_map_once", _fail_once)
    monkeypatch.setattr("scripts.helpers.ci_gate.sync.time.sleep", _sleep_then_stop)

    with caplog.at_level(logging.ERROR):
        exit_code = sync.sync_test_map_watch(
            _DEFAULT_CFG,
            target_branch="master",
            interval_seconds=60,
            logger=logging.getLogger("test"),
        )

    assert exit_code == 0
    assert slept is True
    assert "failed" in caplog.text.lower()


def test_parse_sync_interval_default_when_unset(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.delenv("MSMODELING_TEST_MAP_SYNC_INTERVAL", raising=False)
    assert sync._parse_sync_interval() == 60.0


def test_parse_sync_interval_reads_env(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("MSMODELING_TEST_MAP_SYNC_INTERVAL", "120")
    assert sync._parse_sync_interval() == 120.0


def test_parse_sync_interval_rejects_invalid(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("MSMODELING_TEST_MAP_SYNC_INTERVAL", "not-a-number")
    with pytest.raises(ConfigError):
        sync._parse_sync_interval()


def test_log_sync_env_logs_target_and_interval(
    caplog: pytest.LogCaptureFixture,
) -> None:
    logger = logging.getLogger("test_sync_env")
    with caplog.at_level(logging.INFO):
        sync._log_sync_env(logger, "origin/master", 30.0)
    assert "MSMODELING_TEST_MAP_TARGET_BRANCH = origin/master" in caplog.text
    assert "MSMODELING_TEST_MAP_SYNC_INTERVAL = 30.0" in caplog.text


def test_build_arg_parser_requires_once_or_watch() -> None:
    parser = sync.build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
    args = parser.parse_args(["--once"])
    assert args.once is True
    assert args.watch is False


def test_build_test_map_pytest_cmd_includes_marker_and_cov() -> None:
    cmd = sync.build_test_map_pytest_cmd("python")
    assert cmd[0] == "python"
    assert "-m" in cmd
    assert "tests/smoke/" in cmd
    assert "tests/regression/" in cmd


def test_run_test_map_pytest_invokes_subprocess(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    recorded: list[list[str]] = []

    def _fake_run(cmd: list[str], **_kwargs: object) -> FakeCompleted:
        recorded.append(cmd)
        return FakeCompleted(0, "", "")

    monkeypatch.setattr("scripts.helpers.ci_gate.sync.subprocess.run", _fake_run)
    exit_code = sync.run_test_map_pytest(tmp_path, "python")
    assert exit_code == 0
    assert recorded[0][0] == "python"


def test_collect_fresh_map_returns_map(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.sync.load_gate_policy",
        lambda _root: type("Policy", (), {"roots": ("cli/",)})(),
    )
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.sync.collect_allowed_node_ids",
        lambda _marker: frozenset({"tests/smoke/test_a.py::test_a"}),
    )
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.sync.collect_test_map",
        lambda **_kwargs: {"tests/smoke/test_a.py::test_a": {"cli/main.py": ["run"]}},
    )
    result = sync._collect_fresh_map()
    assert result == {"tests/smoke/test_a.py::test_a": {"cli/main.py": ["run"]}}


def test_run_pytest_and_collect_fresh_map_returns_none_on_pytest_failure(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(sync, "run_test_map_pytest", lambda *_a, **_k: 1)
    assert sync._run_pytest_and_collect_fresh_map(tmp_path, "python") is None


def test_run_pytest_and_collect_fresh_map_collects_on_success(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(sync, "run_test_map_pytest", lambda *_a, **_k: 0)
    monkeypatch.setattr(sync, "_collect_fresh_map", lambda: {"tests/a.py::test_x": {}})
    assert sync._run_pytest_and_collect_fresh_map(tmp_path, "python") == {"tests/a.py::test_x": {}}


def test_full_rebuild_test_map_writes_map(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    map_path = tmp_path / "test_map.json"
    cfg = Config(
        test_map_path=str(map_path),
        base_branch="master",
        line_threshold=60.0,
        branch_threshold=40.0,
        benchmark_parallel=False,
        feishu_webhook_url="",
        msmodeling_cache=".msmodeling_cache",
        weights_prune=False,
    )
    fresh_map = {"tests/smoke/test_a.py::test_a": {"cli/main.py": ["run"]}}
    written: list[tuple[Path, dict[str, object]]] = []

    class _FakeCheckout:
        def __enter__(self) -> str:
            return "targetsha111"

        def __exit__(self, *_args: object) -> None:
            return None

    monkeypatch.setattr(sync, "ephemeral_target_checkout", lambda *_a, **_k: _FakeCheckout())
    monkeypatch.setattr(sync, "_run_pytest_and_collect_fresh_map", lambda *_a, **_k: fresh_map)
    monkeypatch.setattr(
        sync,
        "write_test_map",
        lambda path, mapping, *, built_from_commit: written.append(
            (path, {"map": mapping, "commit": built_from_commit})
        ),
    )
    logger = logging.getLogger("test_full_rebuild")
    with caplog.at_level(logging.INFO):
        exit_code = sync._full_rebuild_test_map(
            cfg,
            target_branch="master",
            target_head="targetsha111",
            logger=logger,
            reason="test rebuild",
        )
    assert exit_code == 0
    assert written[0][0] == map_path
    assert written[0][1]["commit"] == "targetsha111"


def test_main_once_requires_test_map_path(monkeypatch: MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    monkeypatch.setenv("MSMODELING_OFFLINE", "1")
    monkeypatch.delenv("MSMODELING_TEST_MAP_PATH", raising=False)
    with caplog.at_level(logging.ERROR):
        exit_code = sync.main(["--once"])
    assert exit_code == 1
    assert "MSMODELING_TEST_MAP_PATH is required" in caplog.text


def test_main_once_runs_sync(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    map_path = tmp_path / "test_map.json"
    map_path.write_text('{"schema_version":1,"built_from_commit":"abc","map":{}}', encoding="utf-8")
    monkeypatch.setenv("MSMODELING_OFFLINE", "1")
    monkeypatch.setenv("MSMODELING_TEST_MAP_PATH", str(map_path))
    monkeypatch.setattr(sync, "sync_test_map_once", lambda *_a, **_k: 0)
    assert sync.main(["--once", "--target-branch", "master"]) == 0
