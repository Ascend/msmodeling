"""Tests for AI-native repository validation helpers."""

from __future__ import annotations

import io
import json
import subprocess
import tarfile
from pathlib import Path

import pytest

from scripts.ai.check_gitcode_cli import (
    CheckResult,
    _resolve_binary,
    _run,
    check_cli,
    main as check_cli_main,
    parse_semantic_version,
)
from scripts.ai.install_gitleaks import (
    GITLEAKS_VERSION,
    InstallResult,
    PlatformAsset,
    _binary_version,
    _ensure_gitignore,
    _extract_member,
    detect_asset,
    install,
    main as install_gitleaks_main,
)
from scripts.ai.resolve_repository_context import (
    load_contract,
    main as resolve_repository_main,
    parse_repository_slug,
    resolve_context,
    validate_repository_slug,
)
from scripts.ai.validate_remote_boundary import (
    main as validate_remote_boundary_main,
    validate_boundary,
    validate_repository_identity,
)
from scripts.ai.validate_skills import (
    SEMVER_PATTERN,
    main as validate_skills_main,
    validate_skills,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _fake_completed(returncode: int = 0, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


def test_parse_semantic_version() -> None:
    assert parse_semantic_version("gitcode version 0.8.0") == (0, 8, 0)
    assert parse_semantic_version("gitcode version v1.2.3") == (1, 2, 3)
    assert parse_semantic_version("gitcode version dev") is None


def test_inline_review_contract_uses_new_file_line() -> None:
    ref = REPO_ROOT / ".agents" / "skills" / "sig-review" / "ref" / "line-mapping.md"
    text = ref.read_text(encoding="utf-8")
    assert "新版本文件的实际行号" in text
    assert "禁止使用“估计位置”" in text


def test_all_skills_have_required_frontmatter() -> None:
    findings = validate_skills(REPO_ROOT)
    errors = [f for f in findings if f.severity == "error"]
    assert errors == [], errors


def test_skill_metadata_version_uses_semver() -> None:
    assert SEMVER_PATTERN.fullmatch("1.2.3")
    assert SEMVER_PATTERN.fullmatch("0.1.0-beta.1+build.7")
    assert not SEMVER_PATTERN.fullmatch("1.2")
    assert not SEMVER_PATTERN.fullmatch("v1.2.3")


def test_skills_do_not_bypass_gitcode_cli() -> None:
    assert validate_boundary(REPO_ROOT) == []
    assert validate_repository_identity(REPO_ROOT) == []


def test_repository_identity_detects_contributor_fork(tmp_path: Path) -> None:
    agents = tmp_path / ".agents"
    agents.mkdir()
    (agents / "repository-contract.json").write_text(
        '{"canonical_repository": "Ascend/msmodeling"}',
        encoding="utf-8",
    )
    (tmp_path / "AGENTS.md").write_text(
        "Do not pin contributor/msmodeling here.",
        encoding="utf-8",
    )

    findings = validate_repository_identity(
        tmp_path,
        contributor_repository="contributor/msmodeling",
    )

    assert len(findings) == 1
    assert findings[0].path == "AGENTS.md"


def test_repository_contract_uses_canonical_upstream() -> None:
    contract = load_contract(REPO_ROOT)
    assert contract["canonical_repository"] == "Ascend/msmodeling"
    assert contract["source_repository"]["strategy"] == "git-remote"
    assert contract["operation_target"]["write_requires_explicit_repository"] is True


def test_parse_repository_slug_supports_fork_remotes() -> None:
    assert parse_repository_slug("git@gitcode.com:contributor/msmodeling.git") == "contributor/msmodeling"
    assert parse_repository_slug("https://gitcode.com/Ascend/msmodeling.git") == "Ascend/msmodeling"


def test_repository_context_separates_source_and_target() -> None:
    context = resolve_context(
        REPO_ROOT,
        operation_repository="Ascend/msmodeling",
        write=True,
    )
    assert context["canonical_repository"] == "Ascend/msmodeling"
    assert context["source_repository"] != ""
    assert context["operation_target"] == "Ascend/msmodeling"
    assert context["canonical_target"] is True
    assert context["target_pull_request_ci_required"] is True


def test_repository_context_requires_explicit_write_target() -> None:
    with pytest.raises(ValueError, match="explicit --repo"):
        resolve_context(REPO_ROOT, operation_repository=None, write=True)


def test_repository_slug_rejects_unsafe_values() -> None:
    with pytest.raises(ValueError, match="invalid owner/repository"):
        validate_repository_slug("Ascend/msmodeling --state closed")


# ---------------------------------------------------------------------------
# Coverage for scripts/ai/ CLI entry points and helpers (PR #623)
# ---------------------------------------------------------------------------


def test_resolve_binary_prefers_explicit_then_env(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _resolve_binary("/usr/bin/true") == "/usr/bin/true"
    monkeypatch.setenv("GITCODE_BIN", "/usr/bin/true")
    assert _resolve_binary(None) == "/usr/bin/true"
    monkeypatch.delenv("GITCODE_BIN", raising=False)
    # shutil.which fallback: mock so the test does not depend on a real gitcode binary in CI
    monkeypatch.setattr("scripts.ai.check_gitcode_cli.shutil.which", lambda cmd: "/usr/bin/true")
    assert _resolve_binary(None) == "/usr/bin/true"
    # nothing resolvable -> FileNotFoundError (covers the error branch)
    monkeypatch.setattr("scripts.ai.check_gitcode_cli.shutil.which", lambda cmd: None)
    with pytest.raises(FileNotFoundError):
        _resolve_binary(None)


def test_run_invokes_subprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    def fake_run(cmd, **kw):  # type: ignore[no-untyped-def]
        captured["cmd"] = cmd
        return _fake_completed(0, "out", "err")

    monkeypatch.setattr("scripts.ai.check_gitcode_cli.subprocess.run", fake_run)
    proc = _run("/usr/bin/true", "version")
    assert proc.returncode == 0
    assert captured["cmd"] == ["/usr/bin/true", "version"]


def test_check_cli_handles_non_dict_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    # schema returns a JSON array (non-dict) -> AttributeError must be caught
    def fake_run(cmd, **kw):  # type: ignore[no-untyped-def]
        if "version" in cmd:
            return _fake_completed(0, "gitcode version 0.8.0", "")
        if "auth" in cmd:
            return _fake_completed(0, "ok", "")
        if "schema" in cmd:
            return _fake_completed(0, '[{"name": "repo"}]', "")
        return _fake_completed(1, "", "")

    monkeypatch.setattr("scripts.ai.check_gitcode_cli._run", lambda b, *a: fake_run([b, *a]))
    result = check_cli("/usr/bin/true")
    assert isinstance(result, CheckResult)
    # schema mismatch (array payload) reported, no crash
    assert not result.ok
    assert any("schema mismatch" in e for e in result.errors)


def test_check_cli_main_json_emits_result(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    monkeypatch.setattr("sys.argv", ["check_cli", "--json", "--binary", "/usr/bin/true"])
    monkeypatch.setattr(
        "scripts.ai.check_gitcode_cli.check_cli",
        lambda b: CheckResult(
            binary=b,
            version_text="dev",
            semantic_version=None,
            development_build=True,
            auth_ok=True,
            schemas={},
            errors=[],
            ok=True,
        ),
    )
    rc = check_cli_main()
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is True


def test_detect_asset_returns_platform_asset() -> None:
    asset = detect_asset(GITLEAKS_VERSION)
    assert isinstance(asset, PlatformAsset)
    assert asset.binary_member
    assert asset.archive


def test_binary_version_missing_and_present(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert _binary_version(tmp_path / "missing") is None
    binary = tmp_path / "gitleaks"
    binary.write_text("", encoding="utf-8")
    monkeypatch.setattr(
        "scripts.ai.install_gitleaks.subprocess.run",
        lambda cmd, **kw: _fake_completed(0, GITLEAKS_VERSION, ""),
    )
    assert _binary_version(binary) == GITLEAKS_VERSION


def test_ensure_gitignore_is_idempotent(tmp_path: Path) -> None:
    assert _ensure_gitignore(tmp_path) is True
    assert _ensure_gitignore(tmp_path) is False
    assert "/gitleaks" in (tmp_path / ".gitignore").read_text(encoding="utf-8")


def test_extract_member_tar_archive(tmp_path: Path) -> None:
    member_content = b"#!/bin/sh\nexit 0\n"
    archive = tmp_path / "gitleaks.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        info = tarfile.TarInfo("gitleaks")
        info.size = len(member_content)
        tf.addfile(info, io.BytesIO(member_content))
    asset = PlatformAsset("gitleaks.tar.gz", False, "gitleaks")
    target = _extract_member(archive, asset, tmp_path)
    assert target.exists()
    assert target.read_bytes() == member_content


def test_install_reports_present_when_version_matches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "gitleaks").write_text("", encoding="utf-8")
    monkeypatch.setattr("scripts.ai.install_gitleaks._binary_version", lambda b: GITLEAKS_VERSION)
    result = install(tmp_path, GITLEAKS_VERSION, force=False)
    assert isinstance(result, InstallResult)
    assert result.ok is True
    assert result.installed is False


def test_install_main_json_emits_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr("sys.argv", ["install_gitleaks", "--repo-root", str(tmp_path), "--json"])
    monkeypatch.setattr(
        "scripts.ai.install_gitleaks.install",
        lambda root, ver, force: InstallResult(
            binary=str(root / "gitleaks"),
            version=ver,
            installed=False,
            gitignore_updated=False,
            ok=True,
            errors=[],
        ),
    )
    rc = install_gitleaks_main()
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is True


def test_resolve_repository_main_json(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    monkeypatch.setattr("sys.argv", ["resolve", "--json", "--repo", "Ascend/msmodeling"])
    rc = resolve_repository_main()
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is True
    assert out["canonical_repository"] == "Ascend/msmodeling"


def test_resolve_repository_main_requires_write_target(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr("sys.argv", ["resolve", "--json", "--write"])
    rc = resolve_repository_main()
    assert rc == 1
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is False


def test_validate_remote_boundary_main_json(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    monkeypatch.setattr("sys.argv", ["validate_remote_boundary", "--json"])
    rc = validate_remote_boundary_main()
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is True


def test_validate_skills_main_json(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    monkeypatch.setattr("sys.argv", ["validate_skills", "--json"])
    rc = validate_skills_main()
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is True
