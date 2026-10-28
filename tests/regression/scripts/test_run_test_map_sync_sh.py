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

"""Shell entry checks for run_test_map_sync.sh before dependency install."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

from scripts.helpers._paths import REPO_ROOT

_SCRIPT = REPO_ROOT / "scripts" / "run_test_map_sync.sh"
_BASH = shutil.which("bash") or "bash"
_HEAD = "f897ac382618a7ecab53ffe4f80e8bcd93ce2700"


def _write_map(path: Path, commit: str | None) -> None:
    payload: dict[str, object] = {"schema_version": 1, "map": {}}
    if commit is not None:
        payload["built_from_commit"] = commit
    path.write_text(json.dumps(payload), encoding="utf-8")


def _fake_tools(
    tmp_path: Path,
    *,
    head: str = _HEAD,
    fetch_ok: bool = True,
    rev_parse_ok: bool = True,
    uv_exit: int = 99,
) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True)
    fetch_status = "0" if fetch_ok else "1"
    rev_parse_status = "0" if rev_parse_ok else "1"
    (bin_dir / "git").write_text(
        "\n".join(
            (
                "#!/bin/bash",
                'printf "%s\\n" "$*" >> "${GIT_LOG}"',
                'if [[ "$*" == *rev-parse* ]]; then',
                f"  [[ {rev_parse_status} -eq 0 ]] || exit 1",
                '  printf "%s\\n" "${FAKE_HEAD}"',
                "  exit 0",
                "fi",
                'if [[ "$*" == *fetch* ]]; then',
                f"  exit {fetch_status}",
                "fi",
                "exit 1",
                "",
            )
        ),
        encoding="utf-8",
    )
    (bin_dir / "uv").write_text(
        "\n".join(
            (
                "#!/bin/bash",
                'printf "%s\\n" "$*" >> "${UV_LOG}"',
                f"exit {uv_exit}",
                "",
            )
        ),
        encoding="utf-8",
    )
    (bin_dir / "python-stub").write_text(
        '#!/bin/bash\nprintf "%s\\n" "$*" >> "${PY_LOG}"\nexit 0\n',
        encoding="utf-8",
    )
    for name in ("git", "uv", "python-stub"):
        (bin_dir / name).chmod(0o755)
    return bin_dir


def _run(
    tmp_path: Path,
    args: list[str],
    *,
    commit: str | None = _HEAD,
    head: str = _HEAD,
    fetch_ok: bool = True,
    rev_parse_ok: bool = True,
    uv_exit: int = 99,
    python: str | None = None,
    extra_env: dict[str, str] | None = None,
    map_text: str | None = None,
    map_bytes: bytes | None = None,
    omit_map: bool = False,
    isolated_path: bool = False,
    python_fallback: bool = False,
) -> subprocess.CompletedProcess[str]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    map_path = tmp_path / "test_map.json"
    if omit_map:
        map_path.unlink(missing_ok=True)
    elif map_bytes is not None:
        map_path.write_bytes(map_bytes)
    elif map_text is None:
        _write_map(map_path, commit)
    else:
        map_path.write_text(map_text, encoding="utf-8")
    bin_dir = _fake_tools(
        tmp_path,
        head=head,
        fetch_ok=fetch_ok,
        rev_parse_ok=rev_parse_ok,
        uv_exit=uv_exit,
    )
    if python_fallback:
        real_python = shutil.which("python3")
        assert real_python is not None
        (bin_dir / "python").symlink_to(real_python)
    env = dict(os.environ)
    env.pop("PYTHON", None)
    env.pop("UV_NO_SYNC", None)
    env.pop("MSMODELING_TEST_MAP_TARGET_BRANCH", None)
    # CI exports BASH_ENV, which sources profile scripts. Those scripts rewrite PATH
    # with tr/sed and hide the dirname/readlink entries this test just installed.
    env.pop("BASH_ENV", None)
    env.pop("ENV", None)
    if isolated_path:
        tools = tmp_path / "coreutils"
        tools.mkdir()
        for name in ("dirname", "readlink"):
            found = shutil.which(name)
            assert found is not None
            (tools / name).symlink_to(found)
        env["PATH"] = f"{bin_dir}{os.pathsep}{tools}"
    else:
        env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
    env["MSMODELING_TEST_MAP_PATH"] = str(map_path)
    env["MSMODELING_TEST_BASE_BRANCH"] = "master"
    env["GIT_LOG"] = str(tmp_path / "git.log")
    env["UV_LOG"] = str(tmp_path / "uv.log")
    env["PY_LOG"] = str(tmp_path / "python.log")
    env["FAKE_HEAD"] = head
    if python is not None:
        env["PYTHON"] = python
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [_BASH, "--noprofile", "--norc", str(_SCRIPT), *args],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _assert_skipped(result: subprocess.CompletedProcess[str], tmp_path: Path) -> None:
    assert result.returncode == 0, result.stderr
    assert "skipping dependency install" in result.stdout
    assert not (tmp_path / "uv.log").exists()
    assert not (tmp_path / "python.log").exists()


def _assert_continued(result: subprocess.CompletedProcess[str], tmp_path: Path) -> str:
    assert result.returncode == 0, result.stderr
    assert "skipping dependency install" not in result.stdout
    log = (tmp_path / "python.log").read_text(encoding="utf-8")
    assert "ci_gate/sync.py" in log
    return log


def _stub(tmp_path: Path) -> str:
    return str(tmp_path / "bin" / "python-stub")


def test_once_skips_install_when_commit_matches(tmp_path: Path) -> None:
    result = _run(tmp_path, ["--once"])

    _assert_skipped(result, tmp_path)
    assert _HEAD[:12] in result.stdout
    git_log = (tmp_path / "git.log").read_text(encoding="utf-8")
    assert "fetch --depth=50 origin master" in git_log
    assert "rev-parse origin/master^{commit}" in git_log


def test_once_uses_separate_target_branch_flag(tmp_path: Path) -> None:
    result = _run(tmp_path, ["--once", "--target-branch", "origin/release"])

    _assert_skipped(result, tmp_path)
    git_log = (tmp_path / "git.log").read_text(encoding="utf-8")
    assert "fetch --depth=50 origin release" in git_log
    assert "rev-parse origin/release^{commit}" in git_log


def test_once_uses_equals_target_branch_flag(tmp_path: Path) -> None:
    result = _run(tmp_path, ["--once", "--target-branch=origin/release"])

    _assert_skipped(result, tmp_path)
    assert "fetch --depth=50 origin release" in (tmp_path / "git.log").read_text(encoding="utf-8")


def test_once_target_branch_env_is_trimmed_and_cli_wins(tmp_path: Path) -> None:
    from_env = _run(
        tmp_path / "env",
        ["--once"],
        extra_env={"MSMODELING_TEST_MAP_TARGET_BRANCH": "  origin/release  "},
    )
    _assert_skipped(from_env, tmp_path / "env")
    assert "fetch --depth=50 origin release" in (tmp_path / "env" / "git.log").read_text(encoding="utf-8")

    overridden = _run(
        tmp_path / "cli",
        ["--once", "--target-branch", "origin/hotfix"],
        extra_env={"MSMODELING_TEST_MAP_TARGET_BRANCH": "origin/release"},
    )
    _assert_skipped(overridden, tmp_path / "cli")
    cli_log = (tmp_path / "cli" / "git.log").read_text(encoding="utf-8")
    assert "fetch --depth=50 origin hotfix" in cli_log
    assert "origin release" not in cli_log


def test_once_bare_base_branch_uses_origin(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        ["--once"],
        extra_env={"MSMODELING_TEST_BASE_BRANCH": "  develop  "},
    )

    _assert_skipped(result, tmp_path)
    git_log = (tmp_path / "git.log").read_text(encoding="utf-8")
    assert "fetch --depth=50 origin develop" in git_log
    assert "rev-parse origin/develop^{commit}" in git_log


def test_once_strips_commit_whitespace(tmp_path: Path) -> None:
    result = _run(tmp_path, ["--once"], map_text=json.dumps({"built_from_commit": f"  {_HEAD}  ", "map": {}}))

    _assert_skipped(result, tmp_path)


def test_once_interval_flags_do_not_hide_once(tmp_path: Path) -> None:
    separate = _run(tmp_path / "separate", ["--interval", "30", "--once"])
    equals = _run(tmp_path / "equals", ["--once", "--interval=5"])

    _assert_skipped(separate, tmp_path / "separate")
    _assert_skipped(equals, tmp_path / "equals")


def test_once_falls_back_to_python_when_python3_is_absent(tmp_path: Path) -> None:
    result = _run(tmp_path, ["--once"], isolated_path=True, python_fallback=True)

    _assert_skipped(result, tmp_path)


def test_once_installs_when_commit_moved(tmp_path: Path) -> None:
    result = _run(tmp_path, ["--once", "--target-branch", "develop"], head="a" * 40, python=_stub(tmp_path))

    log = _assert_continued(result, tmp_path)
    assert "--once" in log
    assert "--target-branch" in log
    assert "develop" in log
    assert not (tmp_path / "uv.log").exists()


def test_once_installs_when_short_commit_does_not_match(tmp_path: Path) -> None:
    result = _run(tmp_path, ["--once"], commit=_HEAD[:12], python=_stub(tmp_path))

    _assert_continued(result, tmp_path)


def test_once_installs_when_map_is_unreadable(tmp_path: Path) -> None:
    for index, map_text in enumerate(("{", "[]", '{"built_from_commit": 1}', '{"built_from_commit": "   "}')):
        case = tmp_path / str(index)
        result = _run(case, ["--once"], python=_stub(case), map_text=map_text)
        _assert_continued(result, case)
        assert not (case / "git.log").exists()


def test_once_installs_when_map_is_not_utf8(tmp_path: Path) -> None:
    result = _run(tmp_path, ["--once"], python=_stub(tmp_path), map_bytes=b"\xff")

    _assert_continued(result, tmp_path)
    assert not (tmp_path / "git.log").exists()


def test_once_installs_when_target_branch_trims_to_empty(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        ["--once"],
        python=_stub(tmp_path),
        extra_env={"MSMODELING_TEST_BASE_BRANCH": "   "},
    )

    _assert_continued(result, tmp_path)
    assert not (tmp_path / "git.log").exists()


def test_once_installs_when_map_file_is_missing(tmp_path: Path) -> None:
    result = _run(tmp_path, ["--once"], python=_stub(tmp_path), omit_map=True)

    _assert_continued(result, tmp_path)
    assert not (tmp_path / "git.log").exists()


def test_once_installs_when_built_from_commit_missing(tmp_path: Path) -> None:
    result = _run(tmp_path, ["--once"], commit=None, python=_stub(tmp_path))

    _assert_continued(result, tmp_path)


def test_once_installs_when_fetch_fails(tmp_path: Path) -> None:
    result = _run(tmp_path, ["--once"], fetch_ok=False, python=_stub(tmp_path))

    _assert_continued(result, tmp_path)


def test_once_installs_when_rev_parse_fails(tmp_path: Path) -> None:
    result = _run(tmp_path, ["--once"], rev_parse_ok=False, python=_stub(tmp_path))

    _assert_continued(result, tmp_path)


def test_once_installs_when_target_branch_is_incomplete(tmp_path: Path) -> None:
    missing_value = _run(tmp_path / "missing", ["--once", "--target-branch"], python=_stub(tmp_path / "missing"))
    empty_branch = _run(
        tmp_path / "empty",
        ["--once"],
        python=_stub(tmp_path / "empty"),
        extra_env={"MSMODELING_TEST_MAP_TARGET_BRANCH": "origin/"},
    )

    _assert_continued(missing_value, tmp_path / "missing")
    _assert_continued(empty_branch, tmp_path / "empty")
    assert not (tmp_path / "empty" / "git.log").exists()


def test_once_installs_when_no_interpreter_can_read_the_map(tmp_path: Path) -> None:
    result = _run(tmp_path, ["--once"], python=_stub(tmp_path), isolated_path=True)

    _assert_continued(result, tmp_path)
    assert not (tmp_path / "git.log").exists()


def test_once_still_runs_uv_sync_when_commit_moved(tmp_path: Path) -> None:
    result = _run(tmp_path, ["--once"], head="b" * 40, uv_exit=0)

    assert result.returncode == 0, result.stderr
    assert "skipping dependency install" not in result.stdout
    uv_log = (tmp_path / "uv.log").read_text(encoding="utf-8")
    assert "sync --frozen --group ci" in uv_log
    assert "run python" in uv_log
    assert "ci_gate/sync.py" in uv_log


def test_watch_installs_even_when_commit_matches(tmp_path: Path) -> None:
    watch = _run(tmp_path / "watch", ["--watch"], python=_stub(tmp_path / "watch"))
    both = _run(tmp_path / "both", ["--once", "--watch"], python=_stub(tmp_path / "both"))

    for case, result in ((tmp_path / "watch", watch), (tmp_path / "both", both)):
        log = _assert_continued(result, case)
        assert "--watch" in log
        assert not (case / "git.log").exists()
