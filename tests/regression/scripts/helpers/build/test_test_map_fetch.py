"""Tests for scripts.helpers.build.test_map_fetch and defaults."""

from __future__ import annotations

from typing import TYPE_CHECKING
from urllib.error import HTTPError

import pytest

from scripts.helpers.build.test_map_fetch import (
    MapFetchError,
    download_test_map,
    resolve_test_map_path,
)
from scripts.helpers.defaults import build_test_map_url, default_test_map_path

if TYPE_CHECKING:
    from pathlib import Path


def test_build_test_map_url_keeps_branch_slash() -> None:
    url = build_test_map_url("poc/AiClusterHub")
    assert url.endswith("/sync/poc/AiClusterHub/test_map.json")


def test_default_test_map_path_is_branch_scoped() -> None:
    path = default_test_map_path(cache_dir=".msmodeling_cache", base_branch="poc/AiClusterHub")
    assert path.as_posix() == ".msmodeling_cache/test_map/poc/AiClusterHub/test_map.json"


def test_resolve_uses_existing_valid_file(tmp_path: Path) -> None:
    map_file = tmp_path / "map.json"
    map_file.write_text('{"schema_version": 1}', encoding="utf-8")
    assert resolve_test_map_path(configured=str(map_file), base_branch="master") == map_file


def test_resolve_redownloads_corrupt_configured_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("not-json", encoding="utf-8")
    dest = tmp_path / "cache" / "test_map" / "master" / "test_map.json"

    def fake_download(*, base_branch: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text('{"ok": true}', encoding="utf-8")

    monkeypatch.setattr(
        "scripts.helpers.build.test_map_fetch.download_test_map",
        fake_download,
    )
    with caplog.at_level("WARNING"):
        path = resolve_test_map_path(
            configured=str(bad),
            base_branch="master",
            cache_dir=tmp_path / "cache",
        )
    assert path == dest
    assert "not valid UTF-8 JSON" in caplog.text


def test_resolve_downloads_when_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    dest = tmp_path / "cache" / "test_map" / "master" / "test_map.json"

    def fake_download(*, base_branch: str, dest: Path) -> None:
        assert base_branch == "master"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text('{"ok": true}', encoding="utf-8")

    monkeypatch.setattr(
        "scripts.helpers.build.test_map_fetch.download_test_map",
        fake_download,
    )
    with caplog.at_level("WARNING"):
        path = resolve_test_map_path(
            configured=str(tmp_path / "missing.json"),
            base_branch="master",
            cache_dir=tmp_path / "cache",
        )
    assert path == dest
    assert path.is_file()
    assert "not a readable file" in caplog.text


def test_resolve_reuses_valid_branch_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dest = tmp_path / "cache" / "test_map" / "master" / "test_map.json"
    dest.parent.mkdir(parents=True)
    dest.write_text('{"cached": true}', encoding="utf-8")
    calls: list[str] = []

    def fake_download(*, base_branch: str, dest: Path) -> None:
        calls.append(base_branch)
        raise AssertionError("should not download")

    monkeypatch.setattr(
        "scripts.helpers.build.test_map_fetch.download_test_map",
        fake_download,
    )
    path = resolve_test_map_path(configured=None, base_branch="master", cache_dir=tmp_path / "cache")
    assert path == dest
    assert calls == []


def test_download_http_error_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(_url: str, timeout: float = 0) -> object:
        assert timeout == 30
        raise HTTPError(_url, 404, "Not Found", hdrs=None, fp=None)  # type: ignore[arg-type]

    monkeypatch.setattr("scripts.helpers.build.test_map_fetch.urllib.request.urlopen", fake_urlopen)
    with pytest.raises(MapFetchError, match="HTTP 404"):
        download_test_map(base_branch="master", dest=tmp_path / "test_map.json")


def test_download_rejects_non_object_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class _Resp:
        def read(self) -> bytes:
            return b"[1, 2]"

        def __enter__(self) -> _Resp:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    monkeypatch.setattr(
        "scripts.helpers.build.test_map_fetch.urllib.request.urlopen",
        lambda _url, timeout=0: _Resp(),
    )
    with pytest.raises(MapFetchError, match="root must be an object"):
        download_test_map(base_branch="master", dest=tmp_path / "test_map.json")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("", None),
        ("# comment only", None),
        ("UV_INDEX_URL=https://example.com  # trailing", ("UV_INDEX_URL", "https://example.com")),
        ("  HF_ENDPOINT = https://hf-mirror.com ", ("HF_ENDPOINT", "https://hf-mirror.com")),
        ("MSMODELING_CACHE=.msmodeling_cache", ("MSMODELING_CACHE", ".msmodeling_cache")),
        ("=novalue", None),
    ],
)
def test_parse_defaults_env_line_matches_shell_grammar(
    raw: str,
    expected: tuple[str, str] | None,
) -> None:
    from scripts.helpers.defaults import parse_defaults_env_line

    assert parse_defaults_env_line(raw) == expected


def test_parse_defaults_env_ignores_unknown_keys(tmp_path: Path) -> None:
    from scripts.helpers.defaults import _parse_defaults_env

    path = tmp_path / "defaults.env"
    path.write_text(
        "\n".join(
            [
                "# header",
                "UV_INDEX_URL=https://a.example  # inline",
                "HF_ENDPOINT=https://b.example",
                "MSMODELING_TEST_BASE_BRANCH=master",
                "MSMODELING_CACHE=.cache",
                "NOT_A_DEFAULT=should_ignore",
            ],
        )
        + "\n",
        encoding="utf-8",
    )
    parsed = _parse_defaults_env(path)
    assert parsed == {
        "UV_INDEX_URL": "https://a.example",
        "HF_ENDPOINT": "https://b.example",
        "MSMODELING_TEST_BASE_BRANCH": "master",
        "MSMODELING_CACHE": ".cache",
    }
