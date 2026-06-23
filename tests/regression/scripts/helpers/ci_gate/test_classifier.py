"""Tests for ci_gate.classifier — change classification from git diff."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from scripts.helpers.ci_gate.models import ChangeSet

from scripts.helpers._paths import REPO_ROOT
from scripts.helpers.ci_gate.classifier import classify_changes, expand_rename_entries
from scripts.helpers.ci_gate.diff import DiffEntry, GitDiffResult
from scripts.helpers.ci_gate.policy import load_gate_policy

_POLICY = load_gate_policy(REPO_ROOT)


def _diff_result(
    *,
    entries: tuple[DiffEntry, ...] = (),
    line_map: dict[str, set[int]] | None = None,
) -> GitDiffResult:
    return GitDiffResult(line_map=line_map or {}, entries=entries)


def _classify(diff: GitDiffResult) -> ChangeSet:
    return classify_changes(diff, _POLICY)


def test_classify_changes_helpers_path_not_new_test() -> None:
    result = _classify(
        _diff_result(entries=(DiffEntry(status="A", old_path=None, new_path="tests/helpers/assert_utils.py"),)),
    )
    assert result.new_test == ()


def test_classify_changes_added_test_populates_new_test() -> None:
    result = _classify(
        _diff_result(entries=(DiffEntry(status="A", old_path=None, new_path="tests/smoke/test_new.py"),)),
    )
    assert "tests/smoke/test_new.py" in result.new_test


def test_classify_changes_modified_source_includes_lines() -> None:
    result = _classify(
        _diff_result(
            entries=(DiffEntry(status="M", old_path="cli/main.py", new_path="cli/main.py"),),
            line_map={"cli/main.py": {10, 11}},
        ),
    )
    assert len(result.modified_source) == 1
    assert result.modified_source[0][0] == "cli/main.py"
    assert result.modified_source[0][1] == frozenset({10, 11})


def test_classify_changes_gate_policy_validate_only() -> None:
    result = _classify(
        _diff_result(
            entries=(
                DiffEntry(
                    status="M",
                    old_path="tests/.ci/gate_policy.yaml",
                    new_path="tests/.ci/gate_policy.yaml",
                ),
            ),
        ),
    )
    assert result.config == ()
    assert result.unscoped_python == ()


def test_classify_changes_agents_skill_scripts_are_unscoped() -> None:
    result = _classify(
        _diff_result(
            entries=(
                DiffEntry(
                    status="M",
                    old_path=".agents/skills/optix-config/scripts/auto_config.py",
                    new_path=".agents/skills/optix-config/scripts/auto_config.py",
                ),
            ),
        ),
    )
    assert result.unscoped_python == (".agents/skills/optix-config/scripts/auto_config.py",)
    assert result.modified_source == ()


def test_classify_changes_rename_product_is_delete_plus_add() -> None:
    result = _classify(
        _diff_result(
            entries=(
                DiffEntry(
                    status="R100",
                    old_path="tensor_cast/foo.py",
                    new_path="tensor_cast/bar.py",
                ),
            ),
        ),
    )
    assert result.del_source == ("tensor_cast/foo.py",)
    assert result.new_source == ("tensor_cast/bar.py",)


def test_expand_rename_partial_adds_modified_lines() -> None:
    entries = expand_rename_entries(
        (
            DiffEntry(
                status="R85",
                old_path="tensor_cast/foo.py",
                new_path="tensor_cast/bar.py",
            ),
        ),
        {"tensor_cast/bar.py": {10, 11}},
    )
    statuses = [entry.status for entry in entries]
    assert statuses == ["D", "A", "M"]


def test_classify_changes_rename_test_to_product_records_del_test_and_new_source() -> None:
    result = _classify(
        _diff_result(
            entries=(
                DiffEntry(
                    status="R100",
                    old_path="tests/smoke/test_foo.py",
                    new_path="tensor_cast/foo.py",
                ),
            ),
        ),
    )
    assert result.del_test == ("tests/smoke/test_foo.py",)
    assert result.new_test == ()
    assert result.new_source == ("tensor_cast/foo.py",)


def test_entry_paths_rejects_missing_path() -> None:
    from scripts.helpers._config import ConfigError
    from scripts.helpers.ci_gate.classifier import _entry_paths

    with pytest.raises(ConfigError, match="old_path"):
        _entry_paths(DiffEntry(status="D", old_path=None, new_path=None))
    with pytest.raises(ConfigError, match="new_path"):
        _entry_paths(DiffEntry(status="M", old_path="a.py", new_path=None))
