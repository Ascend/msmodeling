"""Tests for ci_gate.comments — GitCode PR comment helpers."""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import TYPE_CHECKING
from urllib.parse import parse_qs, urlparse

if TYPE_CHECKING:
    import pytest

from scripts.helpers.ci_gate.comments import (
    GitCodeCommentConfig,
    _build_headers,
    _request_json,
    build_all_exempt_tests_comment_body,
    build_exemption_drift_comment_body,
    build_shadowed_defs_comment_body,
    build_unscoped_python_comment_body,
    load_gitcode_comment_config,
    maybe_post_all_exempt_tests_comment,
    maybe_post_exemption_drift_comment,
    maybe_post_shadowed_defs_comment,
    maybe_post_unscoped_python_comment,
    post_all_exempt_tests_comment,
    post_exemption_drift_comment,
    post_shadowed_defs_comment,
    post_unscoped_python_comment,
)
from scripts.helpers.ci_gate.models import GateError
from scripts.helpers.common.ast_utils import ShadowWarning

_MARKER = "<!-- msmodeling-ci-gate:unscoped-python -->"
_DRIFT_MARKER = "<!-- msmodeling-ci-gate:exemption-drift -->"
_SHADOW_MARKER = "<!-- msmodeling-ci-gate:shadowed-defs -->"
_ALL_EXEMPT_MARKER = "<!-- msmodeling-ci-gate:all-exempt-tests -->"


def _install_urlopen_mock(monkeypatch: pytest.MonkeyPatch, handler: object) -> None:
    monkeypatch.setattr(urllib.request, "urlopen", handler)


def test_request_json_parses_json_response(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {"id": 42, "body": "hello"}

    class _FakeResponse:
        def read(self) -> bytes:
            return json.dumps(payload).encode()

        def __enter__(self) -> _FakeResponse:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    def _fake_urlopen(_request: urllib.request.Request, *, timeout: float) -> _FakeResponse:
        del timeout
        return _FakeResponse()

    _install_urlopen_mock(monkeypatch, _fake_urlopen)
    result = _request_json(
        "GET",
        "https://api.atomgit.com/api/v5/repos/Ascend/msmodeling/pulls/1/comments",
        headers=_build_headers("pat"),
    )
    assert result == payload


def test_request_json_empty_body_returns_empty_dict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeResponse:
        def read(self) -> bytes:
            return b""

        def __enter__(self) -> _FakeResponse:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    _install_urlopen_mock(monkeypatch, lambda *_a, **_kw: _FakeResponse())
    result = _request_json("GET", "https://example.com", headers=_build_headers("pat"))
    assert result == {}


def test_build_all_exempt_tests_comment_body_lists_paths() -> None:
    body = build_all_exempt_tests_comment_body(("tests/smoke/test_a.py", "tests/regression/test_b.py"))
    assert _ALL_EXEMPT_MARKER in body
    assert "`tests/smoke/test_a.py`" in body
    assert "only exempt test node(s)" in body


def test_build_unscoped_python_comment_body_lists_paths() -> None:
    body = build_unscoped_python_comment_body(("scripts/helpers/foo.py", "misc/bar.py"))
    assert _MARKER in body
    assert "`scripts/helpers/foo.py`" in body
    assert "`misc/bar.py`" in body


def test_build_exemption_drift_comment_body_lists_errors() -> None:
    errors = (
        GateError(
            category="exemption_drift",
            path="tensor_cast/ops.py",
            symbol="add",
            detail="exemption tensor_cast/ops.py::add references deleted source file",
        ),
    )
    body = build_exemption_drift_comment_body(errors)
    assert _DRIFT_MARKER in body
    assert "`tensor_cast/ops.py::add`" in body
    assert "deleted source file" in body


def test_build_shadowed_defs_comment_body_lists_warnings() -> None:
    warnings = (ShadowWarning("cli/main.py", 1, "foo", 4),)
    body = build_shadowed_defs_comment_body(warnings)
    assert _SHADOW_MARKER in body
    assert "`cli/main.py:1`" in body
    assert "shadowed by line 4" in body


def test_load_gitcode_comment_config_returns_none_when_incomplete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key in ("GITCODE_OWNER", "GITCODE_REPO", "GITCODE_PR_NUMBER", "GITCODE_PAT"):
        monkeypatch.delenv(key, raising=False)
    assert load_gitcode_comment_config() is None


def test_load_gitcode_comment_config_parses_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GITCODE_OWNER", "Ascend")
    monkeypatch.setenv("GITCODE_REPO", "msmodeling")
    monkeypatch.setenv("GITCODE_PR_NUMBER", "42")
    monkeypatch.setenv("GITCODE_PAT", "secret")
    config = load_gitcode_comment_config()
    assert config == GitCodeCommentConfig(owner="Ascend", repo="msmodeling", pr_number=42, pat_token="secret")


def test_maybe_post_unscoped_python_comment_skips_without_config(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.comments.load_gitcode_comment_config",
        lambda _cfg=None: None,
    )
    with caplog.at_level("DEBUG"):
        maybe_post_unscoped_python_comment(("misc/foo.py",))
    assert "GitCode PR comment not posted" in caplog.text
    assert "Unscoped Python" not in caplog.text


def test_post_unscoped_python_comment_creates_when_marker_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = GitCodeCommentConfig(owner="Ascend", repo="msmodeling", pr_number=7, pat_token="pat")
    calls: list[tuple[str, str, object]] = []

    def _fake_urlopen(request: urllib.request.Request, *, timeout: float) -> object:
        del timeout
        method = request.method or "GET"
        url = request.full_url
        payload = None
        if isinstance(request.data, (bytes, bytearray)):
            payload = json.loads(request.data.decode())
        calls.append((method, url, payload))
        if method == "GET":
            return _urlopen_response(json.dumps([]).encode())
        return _urlopen_response(b"{}")

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    post_unscoped_python_comment(("misc/foo.py",), config=config)
    assert calls[0][0] == "GET"
    assert calls[1] == (
        "POST",
        "https://api.atomgit.com/api/v5/repos/Ascend/msmodeling/pulls/7/comments",
        {"body": build_unscoped_python_comment_body(("misc/foo.py",))},
    )


def _urlopen_response(body: bytes) -> object:
    class _FakeResponse:
        def read(self) -> bytes:
            return body

        def __enter__(self) -> _FakeResponse:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    return _FakeResponse()


def test_post_unscoped_python_comment_updates_existing_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = GitCodeCommentConfig(owner="Ascend", repo="msmodeling", pr_number=7, pat_token="pat")
    body = build_unscoped_python_comment_body(("misc/foo.py",))
    calls: list[tuple[str, str, object]] = []

    def _fake_urlopen(request: urllib.request.Request, *, timeout: float) -> object:
        del timeout
        method = request.method or "GET"
        url = request.full_url
        payload = None
        if isinstance(request.data, (bytes, bytearray)):
            payload = json.loads(request.data.decode())
        calls.append((method, url, payload))
        if method == "GET":
            return _urlopen_response(json.dumps([{"id": 99, "body": f"{_MARKER}\nold"}]).encode())
        return _urlopen_response(b"{}")

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    post_unscoped_python_comment(("misc/foo.py",), config=config)
    assert calls[-1] == (
        "PATCH",
        "https://api.atomgit.com/api/v5/repos/Ascend/msmodeling/pulls/comments/99",
        {"body": body},
    )


def test_post_unscoped_python_comment_paginates_pr_comments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = GitCodeCommentConfig(owner="Ascend", repo="msmodeling", pr_number=7, pat_token="pat")
    filler = [{"id": index, "body": f"comment {index}"} for index in range(100)]
    calls: list[tuple[str, str]] = []

    def _fake_urlopen(request: urllib.request.Request, *, timeout: float) -> object:
        del timeout
        method = request.method or "GET"
        url = request.full_url
        calls.append((method, url))
        if method != "GET":
            return _urlopen_response(b"{}")
        page = parse_qs(urlparse(url).query).get("page", ["1"])[0]
        if page == "1":
            return _urlopen_response(json.dumps(filler).encode())
        if page == "2":
            return _urlopen_response(json.dumps([{"id": 200, "body": f"{_MARKER}\nold"}]).encode())
        return _urlopen_response(json.dumps([]).encode())

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    post_unscoped_python_comment(("misc/foo.py",), config=config)
    get_urls = [url for method, url in calls if method == "GET"]
    assert len(get_urls) == 2
    assert parse_qs(urlparse(get_urls[0]).query)["page"] == ["1"]
    assert parse_qs(urlparse(get_urls[1]).query)["page"] == ["2"]
    assert calls[-1][0] == "PATCH"


def test_post_all_exempt_tests_comment_creates_comment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = GitCodeCommentConfig(owner="Ascend", repo="msmodeling", pr_number=3, pat_token="pat")
    methods: list[str] = []

    def _fake_urlopen(request: urllib.request.Request, *, timeout: float) -> object:
        del timeout
        methods.append(request.method or "GET")
        if methods[-1] == "GET":
            return _urlopen_response(json.dumps([]).encode())
        return _urlopen_response(b"{}")

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    post_all_exempt_tests_comment(("tests/smoke/test_a.py",), config=config)
    assert methods == ["GET", "POST"]


def test_post_exemption_drift_comment_creates_comment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = GitCodeCommentConfig(owner="Ascend", repo="msmodeling", pr_number=5, pat_token="pat")
    errors = (
        GateError(
            category="exemption_drift",
            path="tensor_cast/ops.py",
            symbol="add",
            detail="deleted",
        ),
    )
    methods: list[str] = []

    def _fake_urlopen(request: urllib.request.Request, *, timeout: float) -> object:
        del timeout
        methods.append(request.method or "GET")
        if methods[-1] == "GET":
            return _urlopen_response(json.dumps([]).encode())
        return _urlopen_response(b"{}")

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    post_exemption_drift_comment(errors, config=config)
    assert methods == ["GET", "POST"]


def test_post_shadowed_defs_comment_creates_comment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = GitCodeCommentConfig(owner="Ascend", repo="msmodeling", pr_number=9, pat_token="pat")
    warnings = (ShadowWarning("cli/main.py", 1, "foo", 4),)
    methods: list[str] = []

    def _fake_urlopen(request: urllib.request.Request, *, timeout: float) -> object:
        del timeout
        methods.append(request.method or "GET")
        if methods[-1] == "GET":
            return _urlopen_response(json.dumps([]).encode())
        return _urlopen_response(b"{}")

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    post_shadowed_defs_comment(warnings, config=config)
    assert methods == ["GET", "POST"]


def test_maybe_post_exemption_drift_comment_skips_without_config(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.comments.load_gitcode_comment_config",
        lambda _cfg=None: None,
    )
    errors = (
        GateError(
            category="exemption_drift",
            path="tensor_cast/ops.py",
            symbol="add",
            detail="deleted",
        ),
    )
    with caplog.at_level("DEBUG"):
        maybe_post_exemption_drift_comment(errors)
    assert "GitCode PR comment not posted" in caplog.text
    assert "Exemption drift" not in caplog.text


def test_maybe_post_shadowed_defs_comment_logs_http_errors(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = GitCodeCommentConfig(owner="Ascend", repo="msmodeling", pr_number=7, pat_token="pat")
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.comments.load_gitcode_comment_config",
        lambda _cfg=None: config,
    )

    def _raise(*_args: object, **_kwargs: object) -> None:
        raise urllib.error.URLError("network down")

    monkeypatch.setattr(urllib.request, "urlopen", _raise)
    warnings = (ShadowWarning("cli/main.py", 1, "foo", 4),)
    with caplog.at_level("WARNING"):
        maybe_post_shadowed_defs_comment(warnings)
    assert "GitCode PR comment failed (non-blocking)" in caplog.text


def test_maybe_post_all_exempt_tests_comment_logs_logic_errors(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = GitCodeCommentConfig(owner="Ascend", repo="msmodeling", pr_number=7, pat_token="pat")
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.comments.load_gitcode_comment_config",
        lambda _cfg=None: config,
    )

    def _raise(*_args: object, **_kwargs: object) -> None:
        raise ValueError("unexpected marker state")

    monkeypatch.setattr("scripts.helpers.ci_gate.comments.post_all_exempt_tests_comment", _raise)
    with caplog.at_level(logging.ERROR, logger="scripts.helpers.ci_gate.comments"):
        maybe_post_all_exempt_tests_comment(("tests/smoke/test_a.py",))
    assert "GitCode PR comment logic error (non-blocking)" in caplog.text


def test_maybe_post_unscoped_python_comment_logs_http_errors(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = GitCodeCommentConfig(owner="Ascend", repo="msmodeling", pr_number=7, pat_token="pat")
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.comments.load_gitcode_comment_config",
        lambda _cfg=None: config,
    )

    def _raise(*_args: object, **_kwargs: object) -> None:
        raise urllib.error.URLError("network down")

    monkeypatch.setattr("scripts.helpers.ci_gate.comments.post_unscoped_python_comment", _raise)
    with caplog.at_level("WARNING"):
        maybe_post_unscoped_python_comment(("misc/foo.py",))
    assert "GitCode PR comment failed (non-blocking)" in caplog.text


def test_maybe_post_unscoped_python_comment_logs_logic_errors(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = GitCodeCommentConfig(owner="Ascend", repo="msmodeling", pr_number=7, pat_token="pat")
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.comments.load_gitcode_comment_config",
        lambda _cfg=None: config,
    )

    def _raise(*_args: object, **_kwargs: object) -> None:
        raise ValueError("unexpected marker state")

    monkeypatch.setattr("scripts.helpers.ci_gate.comments.post_unscoped_python_comment", _raise)
    with caplog.at_level(logging.ERROR, logger="scripts.helpers.ci_gate.comments"):
        maybe_post_unscoped_python_comment(("misc/foo.py",))
    assert "GitCode PR comment logic error (non-blocking)" in caplog.text
    assert "unexpected marker state" in caplog.text
