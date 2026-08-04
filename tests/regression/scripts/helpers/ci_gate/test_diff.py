"""Tests for ci_gate.diff — resolve_base_ref, fetch_diff_line_map."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from scripts.helpers._config import ConfigError
from scripts.helpers.ci_gate.diff import (
    _fetch_deepen,
    _parse_fetch_remote_branch,
    cleanup_all_ephemeral_checkouts,
    fetch_changed_paths,
    fetch_diff,
    fetch_diff_line_map,
    fetch_ref,
    is_git_ancestor,
    resolve_base_ref,
    resolve_head_commit,
    resolve_ref_commit,
    resolve_remote_ref,
    resolve_target_head,
)
from tests.helpers.fake_subprocess import FakeCompleted

# ---------------------------------------------------------------------------
# resolve_base_ref
# ---------------------------------------------------------------------------


def test_resolve_head_commit_returns_full_sha(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: FakeCompleted(0, "abc123deadbeef\n", ""))
    assert resolve_head_commit(tmp_path) == "abc123deadbeef"


def test_resolve_head_commit_git_failure_raises_config_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "subprocess.run",
        lambda *a, **kw: FakeCompleted(1, "", "fatal: not a git repository"),
    )
    with pytest.raises(ConfigError, match=r"Cannot resolve HEAD commit"):
        resolve_head_commit(tmp_path)


def test_resolve_ref_commit_returns_full_sha(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: FakeCompleted(0, "deadbeef\n", ""))
    assert resolve_ref_commit(tmp_path, "master") == "deadbeef"


def test_fetch_changed_paths_returns_names(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "subprocess.run",
        lambda *a, **kw: FakeCompleted(0, "cli/main.py\ntensor_cast/foo.py\n", ""),
    )
    assert fetch_changed_paths(tmp_path, "base", "head") == frozenset({"cli/main.py", "tensor_cast/foo.py"})


def test_is_git_ancestor_true_when_git_succeeds(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: FakeCompleted(0, "", ""))
    assert is_git_ancestor(tmp_path, "aaa", "bbb") is True


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


def test_fetch_diff_git_failure_raises_config_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: FakeCompleted(1, "", "fatal: bad revision"))
    with pytest.raises(ConfigError, match=r"git diff failed"):
        fetch_diff(tmp_path, "abc123")


def test_fetch_ref_delegates_to_fetch_deepen(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    recorded: list[str] = []

    def _fake_deepen(_repo_root: Path, ref: str) -> None:
        recorded.append(ref)

    monkeypatch.setattr("scripts.helpers.ci_gate.diff._fetch_deepen", _fake_deepen)
    fetch_ref(tmp_path, "origin/master")
    assert recorded == ["origin/master"]


def test_resolve_remote_ref_returns_explicit_remote_branch() -> None:
    assert resolve_remote_ref("origin/master") == "origin/master"


def test_resolve_remote_ref_adds_origin_prefix_for_bare_branch() -> None:
    assert resolve_remote_ref("master") == "origin/master"


def test_resolve_target_head_fetches_and_resolves_sha(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[str] = []

    def _fake_fetch(_repo_root: Path, ref: str) -> None:
        calls.append(ref)

    monkeypatch.setattr("scripts.helpers.ci_gate.diff.fetch_ref", _fake_fetch)
    monkeypatch.setattr("scripts.helpers.ci_gate.diff.resolve_remote_ref", lambda ref: "origin/master")
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.diff.resolve_ref_commit",
        lambda _root, _ref: "deadbeef",
    )
    assert resolve_target_head(tmp_path, "master") == "deadbeef"
    assert calls == ["master"]


def test_cleanup_all_ephemeral_checkouts_restores_each_repo(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from scripts.helpers.ci_gate import diff as diff_mod

    repo_a = tmp_path / "repo_a"
    repo_b = tmp_path / "repo_b"
    repo_a.mkdir()
    repo_b.mkdir()
    diff_mod._ephemeral_checkouts[str(repo_a.resolve())] = diff_mod._EphemeralCheckoutState(
        "msmodeling-sync/1",
        "main",
    )
    diff_mod._ephemeral_checkouts[str(repo_b.resolve())] = diff_mod._EphemeralCheckoutState(
        "msmodeling-sync/2",
        "develop",
    )
    cleaned: list[Path] = []

    def _fake_cleanup(repo_root: Path) -> None:
        cleaned.append(repo_root)
        diff_mod._ephemeral_checkouts.pop(str(repo_root.resolve()), None)

    monkeypatch.setattr(diff_mod, "_cleanup_ephemeral_checkout", _fake_cleanup)
    cleanup_all_ephemeral_checkouts()
    assert len(cleaned) == 2
    assert diff_mod._ephemeral_checkouts == {}


# ---------------------------------------------------------------------------
# ephemeral checkout cleanup
# ---------------------------------------------------------------------------


def test_cleanup_ephemeral_checkout_force_checkout_then_deletes_branch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from scripts.helpers.ci_gate import diff as diff_mod

    work_branch = "msmodeling-sync/99999"
    diff_mod._ephemeral_checkouts[str(tmp_path.resolve())] = diff_mod._EphemeralCheckoutState(
        work_branch,
        "main",
    )
    git_calls: list[tuple[str, ...]] = []

    def _fake_run_git(_repo_root: Path, *args: str) -> FakeCompleted:
        git_calls.append(args)
        if args == ("checkout", "main"):
            return FakeCompleted(1, "", "checkout failed")
        if args == ("checkout", "-f", "main"):
            return FakeCompleted(0, "", "")
        if args == ("branch", "-D", work_branch):
            return FakeCompleted(0, "", "")
        return FakeCompleted(0, "", "")

    monkeypatch.setattr(diff_mod, "_run_git", _fake_run_git)
    diff_mod._cleanup_ephemeral_checkout(tmp_path)

    assert ("checkout", "main") in git_calls
    assert ("checkout", "-f", "main") in git_calls
    assert ("branch", "-D", work_branch) in git_calls
    assert str(tmp_path.resolve()) not in diff_mod._ephemeral_checkouts


def test_cleanup_ephemeral_checkout_skips_delete_when_checkout_unrecoverable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from scripts.helpers.ci_gate import diff as diff_mod

    work_branch = "msmodeling-sync/88888"
    diff_mod._ephemeral_checkouts[str(tmp_path.resolve())] = diff_mod._EphemeralCheckoutState(
        work_branch,
        "main",
    )
    git_calls: list[tuple[str, ...]] = []

    def _fake_run_git(_repo_root: Path, *args: str) -> FakeCompleted:
        git_calls.append(args)
        if args[0] == "checkout":
            return FakeCompleted(1, "", "checkout failed")
        return FakeCompleted(0, "", "")

    monkeypatch.setattr(diff_mod, "_run_git", _fake_run_git)
    diff_mod._cleanup_ephemeral_checkout(tmp_path)

    assert ("branch", "-D", work_branch) not in git_calls
    assert str(tmp_path.resolve()) not in diff_mod._ephemeral_checkouts
