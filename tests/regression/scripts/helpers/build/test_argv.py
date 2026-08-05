"""Tests for scripts.helpers.build.argv."""

from __future__ import annotations

import pytest

from scripts.helpers.build.argv import BuildOptions, BuildSuite, parse_argv


def test_parse_argv_defaults() -> None:
    options = parse_argv([])
    assert options == BuildOptions(
        is_test=False,
        is_local=False,
        version=None,
        version_explicit=False,
        extras={},
        suite=BuildSuite.CI_GATE,
    )


def test_parse_argv_test_token() -> None:
    options = parse_argv(["test"])
    assert options.is_test is True
    assert options.is_local is False


def test_parse_argv_local_token_parsed() -> None:
    options = parse_argv(["local"])
    assert options.is_local is True
    assert options.is_test is False


def test_parse_argv_local_and_test() -> None:
    options = parse_argv(["local", "test"])
    assert options.is_local is True
    assert options.is_test is True


def test_parse_argv_version_flag() -> None:
    options = parse_argv(["--version", "2.3.4"])
    assert options.version == "2.3.4"
    assert options.version_explicit is True


def test_parse_argv_short_version_flag() -> None:
    options = parse_argv(["-v", "9.9.9"])
    assert options.version == "9.9.9"
    assert options.version_explicit is True


def test_parse_argv_extra_key_value() -> None:
    options = parse_argv(["test", "--extra", "test_map_path=/tmp/map.json"])
    assert options.extras == {"test_map_path": "/tmp/map.json"}


def test_parse_argv_build_rejects_any_extra() -> None:
    with pytest.raises(SystemExit) as exc_info:
        parse_argv(["-e", "foo=1"])
    assert exc_info.value.code == 2


def test_parse_argv_test_rejects_unknown_extra() -> None:
    with pytest.raises(SystemExit) as exc_info:
        parse_argv(["test", "-e", "bad=1"])
    assert exc_info.value.code == 2


def test_parse_argv_test_allows_whitelist_extras() -> None:
    options = parse_argv(
        [
            "test",
            "-e",
            "test_map_path=/tmp/map.json",
            "-e",
            "base_branch=main",
            "-e",
            "offline=1",
            "-e",
            "weights_prune=0",
        ],
    )
    assert options.extras == {
        "test_map_path": "/tmp/map.json",
        "base_branch": "main",
        "offline": "1",
        "weights_prune": "0",
    }


def test_parse_argv_build_rejects_test_extra_without_test_token() -> None:
    with pytest.raises(SystemExit) as exc_info:
        parse_argv(["-e", "test_map_path=x"])
    assert exc_info.value.code == 2


def test_parse_argv_unknown_token_exits_2() -> None:
    with pytest.raises(SystemExit) as exc_info:
        parse_argv(["nightly"])
    assert exc_info.value.code == 2


def test_parse_argv_duplicate_token_exits_2() -> None:
    with pytest.raises(SystemExit) as exc_info:
        parse_argv(["test", "test"])
    assert exc_info.value.code == 2


def test_parse_argv_extra_without_equals_exits_2() -> None:
    with pytest.raises(SystemExit) as exc_info:
        parse_argv(["--extra", "bad"])
    assert exc_info.value.code == 2


def test_parse_argv_extra_empty_key_exits_2() -> None:
    with pytest.raises(SystemExit) as exc_info:
        parse_argv(["--extra", "=value"])
    assert exc_info.value.code == 2


def test_parse_argv_duplicate_extra_key_exits_2() -> None:
    with pytest.raises(SystemExit) as exc_info:
        parse_argv(["--extra", "a=1", "--extra", "a=2"])
    assert exc_info.value.code == 2


def test_parse_argv_suite_default_ci_gate() -> None:
    options = parse_argv(["test"])
    assert options.suite == BuildSuite.CI_GATE


def test_parse_argv_suite_full() -> None:
    options = parse_argv(["test", "--suite", "full"])
    assert options.suite == BuildSuite.FULL


@pytest.mark.parametrize("suite", ["smoke", "regression", "benchmark", "ci_gate"])
def test_parse_argv_suite_named(suite: str) -> None:
    options = parse_argv(["test", "--suite", suite])
    assert options.suite.value == suite


def test_parse_argv_suite_without_test_exits_2() -> None:
    with pytest.raises(SystemExit) as exc_info:
        parse_argv(["--suite", "full"])
    assert exc_info.value.code == 2


def test_parse_argv_suite_unknown_exits_2() -> None:
    with pytest.raises(SystemExit) as exc_info:
        parse_argv(["test", "--suite", "nightly"])
    assert exc_info.value.code == 2
