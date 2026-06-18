"""Tests for ci_gate.diff — resolve_base_ref, fetch_diff_line_map, classify_changes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from scripts.helpers._config import ConfigError
from scripts.helpers.ci_gate.diff import (
    DiffEntry,
    GitDiffResult,
    _classify_rename,
    _fetch_deepen,
    _parse_fetch_remote_branch,
    classify_changes,
    fetch_diff_line_map,
    resolve_base_ref,
    resolve_head_commit,
)
from scripts.helpers.ci_gate.gate_policy import default_test_discovery
from scripts.helpers.common.coverage_config import product_roots
from tests.helpers.fake_subprocess import FakeCompleted

_DEFAULT_ROOTS = product_roots()


def _diff_result(
    *,
    entries: tuple[DiffEntry, ...] = (),
    line_map: dict[str, set[int]] | None = None,
) -> GitDiffResult:
    return GitDiffResult(line_map=line_map or {}, entries=entries)


# ---------------------------------------------------------------------------
# resolve_base_ref
# ---------------------------------------------------------------------------


def test_resolve_head_commit_returns_full_sha(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: FakeCompleted(0, "abc123deadbeef\n", ""))
    assert resolve_head_commit(tmp_path) == "abc123deadbeef"


def test_resolve_head_commit_git_failure_raises_config_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: FakeCompleted(1, "", "fatal: not a git repository"))
    with pytest.raises(ConfigError, match=r"Cannot resolve HEAD commit"):
        resolve_head_commit(tmp_path)


def test_resolve_base_ref_merge_base_success_returns_sha(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: FakeCompleted(0, "abc123\n", ""))
    result = resolve_base_ref(tmp_path, "main")
    assert result == "abc123"


def test_resolve_base_ref_fallback_to_origin_returns_sha(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = iter(
        [
            FakeCompleted(1, "", ""),
            FakeCompleted(0, "def456\n", ""),
        ]
    )
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: next(calls))
    result = resolve_base_ref(tmp_path, "main")
    assert result == "def456"


def test_resolve_base_ref_both_fail_raises_config_error_with_branch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: FakeCompleted(1, "", "not found"))
    with pytest.raises(ConfigError, match=r"Cannot resolve base ref.*'nonexistent'.*not found either"):
        resolve_base_ref(tmp_path, "nonexistent")


@pytest.mark.parametrize(
    ("ref", "expected"),
    [
        ("origin/master", ("origin", "master")),
        ("center/develop", ("center", "develop")),
        ("master", ("origin", "master")),
    ],
)
def test_parse_fetch_remote_branch_splits_remote_and_branch(ref: str, expected: tuple[str, str]) -> None:
    assert _parse_fetch_remote_branch(ref) == expected


def test_fetch_deepen_invokes_git_fetch_with_remote_and_branch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    recorded: list[list[str]] = []

    def _fake_run(cmd: list[str], **_kwargs: object) -> FakeCompleted:
        recorded.append(cmd)
        return FakeCompleted(0, "", "")

    monkeypatch.setattr("subprocess.run", _fake_run)
    _fetch_deepen(tmp_path, "origin/master")
    assert recorded[-1][-4:] == ["fetch", "--depth=50", "origin", "master"]


def test_resolve_base_ref_deepens_with_split_fetch_before_retry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = iter(
        [
            FakeCompleted(1, "", ""),
            FakeCompleted(1, "", ""),
            FakeCompleted(0, "", ""),
            FakeCompleted(0, "mergebase\n", ""),
        ]
    )
    recorded: list[list[str]] = []

    def _fake_run(cmd: list[str], **_kwargs: object) -> FakeCompleted:
        recorded.append(cmd)
        return next(calls)

    monkeypatch.setattr("subprocess.run", _fake_run)
    result = resolve_base_ref(tmp_path, "master")
    assert result == "mergebase"
    assert recorded[2][-4:] == ["fetch", "--depth=50", "origin", "master"]


# ---------------------------------------------------------------------------
# fetch_diff_line_map
# ---------------------------------------------------------------------------


def test_fetch_diff_line_map_parses_hunks_into_line_sets(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    diff_output = "+++ b/cli/main.py\n@@ -0,0 +5,3 @@\n+++ b/tensor_cast/ops.py\n@@ -10,0 +20,2 @@\n"
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: FakeCompleted(0, diff_output, ""))
    result = fetch_diff_line_map(tmp_path, "abc123")
    assert result["cli/main.py"] == {5, 6, 7}
    assert result["tensor_cast/ops.py"] == {20, 21}


def test_fetch_diff_line_map_single_line_hunk_returns_single_line(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "subprocess.run",
        lambda *a, **kw: FakeCompleted(0, "+++ b/a.py\n@@ -0,0 +1 @@\n", ""),
    )
    result = fetch_diff_line_map(tmp_path, "abc123")
    assert result["a.py"] == {1}


def test_fetch_diff_line_map_pure_deletion_hunk_records_touch_line(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    diff_output = "+++ b/tensor_cast/ops.py\n@@ -10,3 +9,0 @@\n"
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: FakeCompleted(0, diff_output, ""))
    result = fetch_diff_line_map(tmp_path, "abc123")
    assert result["tensor_cast/ops.py"] == {9}


def test_fetch_diff_line_map_empty_diff_returns_empty_dict(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: FakeCompleted(0, "", ""))
    result = fetch_diff_line_map(tmp_path, "abc123")
    assert result == {}


def test_fetch_diff_parses_added_and_deleted_entries(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from scripts.helpers.ci_gate.diff import fetch_diff

    diff_output = (
        "diff --git a/cli/old.py b/cli/old.py\n"
        "deleted file mode 100644\n"
        "--- a/cli/old.py\n"
        "+++ /dev/null\n"
        "diff --git a/tests/smoke/test_new.py b/tests/smoke/test_new.py\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/tests/smoke/test_new.py\n"
    )
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: FakeCompleted(0, diff_output, ""))
    result = fetch_diff(tmp_path, "abc123")
    assert any(entry.status == "D" and entry.old_path == "cli/old.py" for entry in result.entries)
    assert any(entry.status == "A" and entry.new_path == "tests/smoke/test_new.py" for entry in result.entries)


# ---------------------------------------------------------------------------
# classify_changes
# ---------------------------------------------------------------------------


def test_classify_changes_helpers_path_not_new_test(tmp_path: Path) -> None:
    result = classify_changes(
        tmp_path,
        "abc123",
        _diff_result(entries=(DiffEntry(status="A", old_path=None, new_path="tests/helpers/assert_utils.py"),)),
        roots=_DEFAULT_ROOTS,
    )
    assert result.new_test == ()


def test_classify_changes_added_test_populates_new_test(tmp_path: Path) -> None:
    result = classify_changes(
        tmp_path,
        "abc123",
        _diff_result(entries=(DiffEntry(status="A", old_path=None, new_path="tests/smoke/test_new.py"),)),
        roots=_DEFAULT_ROOTS,
    )
    assert "tests/smoke/test_new.py" in result.new_test


def test_classify_changes_modified_source_includes_lines(tmp_path: Path) -> None:
    diff_map = {"cli/main.py": {10, 11}}
    result = classify_changes(
        tmp_path,
        "abc123",
        _diff_result(
            entries=(DiffEntry(status="M", old_path="cli/main.py", new_path="cli/main.py"),),
            line_map=diff_map,
        ),
        roots=_DEFAULT_ROOTS,
    )
    assert len(result.modified_source) == 1
    assert result.modified_source[0][0] == "cli/main.py"
    assert result.modified_source[0][1] == frozenset({10, 11})


def test_classify_changes_modified_test_populates_modified_test(tmp_path: Path) -> None:
    result = classify_changes(
        tmp_path,
        "abc123",
        _diff_result(
            entries=(
                DiffEntry(
                    status="M", old_path="tests/regression/cli/test_foo.py", new_path="tests/regression/cli/test_foo.py"
                ),
            ),
        ),
        roots=_DEFAULT_ROOTS,
    )
    assert result.modified_test == ("tests/regression/cli/test_foo.py",)
    assert result.new_test == ()
    assert result.modified_source == ()


def test_classify_changes_deleted_test_populates_del_test(tmp_path: Path) -> None:
    result = classify_changes(
        tmp_path,
        "abc123",
        _diff_result(entries=(DiffEntry(status="D", old_path="tests/smoke/test_old.py", new_path=None),)),
        roots=_DEFAULT_ROOTS,
    )
    assert "tests/smoke/test_old.py" in result.del_test


def test_classify_changes_config_triggers_full_suite(tmp_path: Path) -> None:
    result = classify_changes(
        tmp_path,
        "abc123",
        _diff_result(entries=(DiffEntry(status="M", old_path="pyproject.toml", new_path="pyproject.toml"),)),
        roots=_DEFAULT_ROOTS,
    )
    assert result.config == ("pyproject.toml",)


def test_classify_changes_requirements_txt_triggers_full_suite(tmp_path: Path) -> None:
    result = classify_changes(
        tmp_path,
        "abc123",
        _diff_result(entries=(DiffEntry(status="M", old_path="requirements.txt", new_path="requirements.txt"),)),
        roots=_DEFAULT_ROOTS,
    )
    assert result.config == ("requirements.txt",)


def test_classify_changes_uv_lock_triggers_full_suite(tmp_path: Path) -> None:
    result = classify_changes(
        tmp_path,
        "abc123",
        _diff_result(entries=(DiffEntry(status="M", old_path="uv.lock", new_path="uv.lock"),)),
        roots=_DEFAULT_ROOTS,
    )
    assert result.config == ("uv.lock",)


def test_classify_changes_gate_policy_yaml_does_not_trigger_full_suite(tmp_path: Path) -> None:
    result = classify_changes(
        tmp_path,
        "abc123",
        _diff_result(
            entries=(
                DiffEntry(status="M", old_path="tests/.ci/gate_policy.yaml", new_path="tests/.ci/gate_policy.yaml"),
            ),
        ),
        roots=_DEFAULT_ROOTS,
    )
    assert result.config == ()


def test_classify_changes_agents_skill_scripts_not_unscoped(tmp_path: Path) -> None:
    result = classify_changes(
        tmp_path,
        "abc123",
        _diff_result(
            entries=(
                DiffEntry(
                    status="M",
                    old_path=".agents/skills/optix-config/scripts/auto_config.py",
                    new_path=".agents/skills/optix-config/scripts/auto_config.py",
                ),
            ),
        ),
        roots=_DEFAULT_ROOTS,
    )
    assert result.unscoped_source == ()
    assert result.modified_source == ()


def test_classify_changes_deleted_config_triggers_full_suite(tmp_path: Path) -> None:
    result = classify_changes(
        tmp_path,
        "abc123",
        _diff_result(entries=(DiffEntry(status="D", old_path="requirements.txt", new_path=None),)),
        roots=_DEFAULT_ROOTS,
    )
    assert result.config == ("requirements.txt",)


def test_classify_changes_renamed_config_triggers_full_suite(tmp_path: Path) -> None:
    result = classify_changes(
        tmp_path,
        "abc123",
        _diff_result(
            entries=(
                DiffEntry(
                    status="R100",
                    old_path="pytest.ini",
                    new_path="setup.cfg",
                ),
            ),
        ),
        roots=_DEFAULT_ROOTS,
    )
    assert result.config == ("pytest.ini", "setup.cfg")


def test_classify_changes_deleted_conftest_triggers_config_not_del_source(tmp_path: Path) -> None:
    result = classify_changes(
        tmp_path,
        "abc123",
        _diff_result(
            entries=(DiffEntry(status="D", old_path="tests/regression/conftest.py", new_path=None),),
        ),
        roots=_DEFAULT_ROOTS,
    )
    assert result.config == ("tests/regression/conftest.py",)
    assert result.del_source == ()


def test_classify_changes_added_conftest_triggers_config_not_unscoped(tmp_path: Path) -> None:
    result = classify_changes(
        tmp_path,
        "abc123",
        _diff_result(
            entries=(DiffEntry(status="A", old_path=None, new_path="tests/regression/conftest.py"),),
        ),
        roots=_DEFAULT_ROOTS,
    )
    assert result.config == ("tests/regression/conftest.py",)
    assert result.unscoped_source == ()
    assert result.new_source == ()


def test_classify_changes_deleted_source_populates_del_source(tmp_path: Path) -> None:
    result = classify_changes(
        tmp_path,
        "abc123",
        _diff_result(entries=(DiffEntry(status="D", old_path="cli/old_main.py", new_path=None),)),
        roots=_DEFAULT_ROOTS,
    )
    assert "cli/old_main.py" in result.del_source


# ---------------------------------------------------------------------------
# _classify_rename (PR review: old/new path boundary)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("old_path", "new_path", "score", "expected"),
    [
        (
            "tests/smoke/test_foo.py",
            "tests/regression/test_foo.py",
            100,
            {
                "del_test": ["tests/smoke/test_foo.py"],
                "new_test": ["tests/regression/test_foo.py"],
                "del_source": [],
                "renames": [],
                "modified": {},
            },
        ),
        (
            "tests/smoke/test_foo.py",
            "scripts/helpers/foo.py",
            100,
            {
                "del_test": ["tests/smoke/test_foo.py"],
                "new_test": [],
                "del_source": [],
                "renames": [],
                "modified": {},
            },
        ),
        (
            "tests/regression/test_bar.py",
            "tests/helpers/test_bar.py",
            100,
            {
                "del_test": ["tests/regression/test_bar.py"],
                "new_test": [],
                "del_source": [],
                "renames": [],
                "modified": {},
            },
        ),
        (
            "tensor_cast/old_module.py",
            "tests/regression/test_old_module.py",
            100,
            {
                "del_test": [],
                "new_test": ["tests/regression/test_old_module.py"],
                "del_source": ["tensor_cast/old_module.py"],
                "renames": [],
                "modified": {},
            },
        ),
        (
            "tensor_cast/foo.py",
            "tensor_cast/bar.py",
            100,
            {
                "del_test": [],
                "new_test": [],
                "del_source": [],
                "renames": [("tensor_cast/foo.py", "tensor_cast/bar.py", 100)],
                "modified": {},
            },
        ),
        (
            "tensor_cast/foo.py",
            "tensor_cast/bar.py",
            85,
            {
                "del_test": [],
                "new_test": [],
                "del_source": [],
                "renames": [("tensor_cast/foo.py", "tensor_cast/bar.py", 85)],
                "modified": {"tensor_cast/bar.py": frozenset({10, 11})},
            },
        ),
    ],
)
def test_classify_rename_paths(
    old_path: str,
    new_path: str,
    score: int,
    expected: dict[str, object],
) -> None:
    discovery = default_test_discovery()
    diff = {"tensor_cast/bar.py": {10, 11}}

    del_test, new_test, del_source, renames, modified = _classify_rename(
        old_path,
        new_path,
        score,
        diff,
        discovery,
        _DEFAULT_ROOTS,
    )

    assert del_test == expected["del_test"]
    assert new_test == expected["new_test"]
    assert del_source == expected["del_source"]
    assert renames == expected["renames"]
    assert modified == expected["modified"]


def test_classify_changes_rename_test_to_non_test_records_del_test_only(tmp_path: Path) -> None:
    result = classify_changes(
        tmp_path,
        "abc123",
        _diff_result(
            entries=(
                DiffEntry(
                    status="R100",
                    old_path="tests/smoke/test_foo.py",
                    new_path="scripts/helpers/foo.py",
                ),
            ),
        ),
        discovery=default_test_discovery(),
        roots=_DEFAULT_ROOTS,
    )
    assert result.del_test == ("tests/smoke/test_foo.py",)
    assert result.new_test == ()
    assert result.renames == ()


def test_classify_changes_rename_product_to_test_records_del_source_and_new_test(tmp_path: Path) -> None:
    result = classify_changes(
        tmp_path,
        "abc123",
        _diff_result(
            entries=(
                DiffEntry(
                    status="R100",
                    old_path="tensor_cast/old.py",
                    new_path="tests/regression/test_old.py",
                ),
            ),
        ),
        discovery=default_test_discovery(),
        roots=_DEFAULT_ROOTS,
    )
    assert result.del_source == ("tensor_cast/old.py",)
    assert result.new_test == ("tests/regression/test_old.py",)
    assert result.renames == ()


def test_classify_changes_rename_product_populates_renames(tmp_path: Path) -> None:
    result = classify_changes(
        tmp_path,
        "abc123",
        _diff_result(
            entries=(
                DiffEntry(
                    status="R100",
                    old_path="tensor_cast/foo.py",
                    new_path="tensor_cast/bar.py",
                ),
            ),
        ),
        discovery=default_test_discovery(),
        roots=_DEFAULT_ROOTS,
    )
    assert result.renames == (("tensor_cast/foo.py", "tensor_cast/bar.py", 100),)
    assert result.del_test == ()
    assert result.new_test == ()
