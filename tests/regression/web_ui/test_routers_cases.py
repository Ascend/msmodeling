"""Real unit tests for api/routers/cases.py.

Single test file for the cases router. The async ``get_case_log`` handler is
awaited via ``asyncio.run()`` (``pytest-asyncio`` is declared but not installed).
Real imports + fixture-scoped mocks only, per tests/SKILL.md.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest
from api.routers.cases import get_case_log, router
from fastapi import HTTPException
from fastapi.responses import PlainTextResponse


def _run(coro):
    """Await an async coroutine synchronously (no pytest-asyncio needed)."""
    return asyncio.run(coro)


# The router validates case_hash as a sha256 hex digest (64 hex chars). Use
# a 64-char lowercase hex string for tests that should pass validation.
_VALID_HASH = "a" * 64
_ANOTHER_HASH = "b" * 64


class TestRouterConfiguration:
    """Tests for router setup."""

    def test_router_prefix(self):
        """Router is mounted under /api/cases."""
        assert router.prefix == "/api/cases"

    def test_router_tag(self):
        """Router carries the cases tag."""
        assert router.tags == ["cases"]


class TestGetCaseLog:
    """Tests for the get_case_log endpoint (DB primary, file fallback)."""

    def test_returns_db_content_when_present(self):
        """DB hit returns the stored content; file fallback is not consulted."""
        with (
            patch("api.routers.cases.CaseLogRepository") as mock_repo,
            patch("api.routers.cases.read_case_log_file") as mock_read,
        ):
            mock_repo.return_value.get.return_value = "db log body"
            response = _run(get_case_log(_VALID_HASH, MagicMock(), 0))
        assert isinstance(response, PlainTextResponse)
        assert response.body == b"db log body"
        mock_read.assert_not_called()

    def test_falls_back_to_file_when_db_empty(self):
        """An empty DB hit falls back to the file reader."""
        with (
            patch("api.routers.cases.CaseLogRepository") as mock_repo,
            patch("api.routers.cases.read_case_log_file", return_value="file body") as mock_read,
        ):
            mock_repo.return_value.get.return_value = ""
            response = _run(get_case_log(_VALID_HASH, MagicMock(), 0))
        assert response.body == b"file body"
        mock_read.assert_called_once_with(_VALID_HASH)

    def test_falls_back_to_file_when_db_none(self):
        """A None DB hit falls back to the file reader."""
        with (
            patch("api.routers.cases.CaseLogRepository") as mock_repo,
            patch("api.routers.cases.read_case_log_file", return_value="file body"),
        ):
            mock_repo.return_value.get.return_value = None
            response = _run(get_case_log(_VALID_HASH, MagicMock(), 0))
        assert response.body == b"file body"

    def test_raises_404_when_both_db_and_file_empty(self):
        """404 when neither DB nor file has content."""
        with (
            patch("api.routers.cases.CaseLogRepository") as mock_repo,
            patch("api.routers.cases.read_case_log_file", return_value=""),
        ):
            mock_repo.return_value.get.return_value = None
            with pytest.raises(HTTPException) as exc:
                _run(get_case_log(_VALID_HASH, MagicMock(), 0))
        assert exc.value.status_code == 404
        assert _VALID_HASH in exc.value.detail

    def test_raises_404_includes_case_hash(self):
        """The 404 detail includes the case_hash."""
        with (
            patch("api.routers.cases.CaseLogRepository") as mock_repo,
            patch("api.routers.cases.read_case_log_file", return_value=None),
        ):
            mock_repo.return_value.get.return_value = ""
            with pytest.raises(HTTPException) as exc:
                _run(get_case_log(_ANOTHER_HASH, MagicMock(), 0))
        assert _ANOTHER_HASH in exc.value.detail

    def test_raises_400_for_invalid_case_hash(self):
        """A non-sha256 case_hash is rejected with 400 (path traversal defense)."""
        with patch("api.routers.cases.CaseLogRepository") as mock_repo:
            with pytest.raises(HTTPException) as exc:
                _run(get_case_log("not-a-valid-hash", MagicMock(), 0))
        assert exc.value.status_code == 400
        mock_repo.return_value.get.assert_not_called()

    def test_tail_zero_returns_full_log(self):
        """tail=0 (default) returns the full log."""
        body = "line1\nline2\nline3\nline4\nline5"
        with patch("api.routers.cases.CaseLogRepository") as mock_repo, patch("api.routers.cases.read_case_log_file"):
            mock_repo.return_value.get.return_value = body
            response = _run(get_case_log(_VALID_HASH, MagicMock(), 0))
        assert response.body.decode() == body

    def test_tail_positive_returns_last_n_lines(self):
        """tail=N returns the last N lines."""
        with patch("api.routers.cases.CaseLogRepository") as mock_repo, patch("api.routers.cases.read_case_log_file"):
            mock_repo.return_value.get.return_value = "l1\nl2\nl3\nl4\nl5"
            response = _run(get_case_log(_VALID_HASH, MagicMock(), 2))
        assert response.body.decode() == "l4\nl5"

    def test_tail_one_returns_last_line(self):
        """tail=1 returns only the last line."""
        with patch("api.routers.cases.CaseLogRepository") as mock_repo, patch("api.routers.cases.read_case_log_file"):
            mock_repo.return_value.get.return_value = "l1\nl2\nl3"
            response = _run(get_case_log(_VALID_HASH, MagicMock(), 1))
        assert response.body.decode() == "l3"

    def test_tail_larger_than_line_count_returns_all(self):
        """tail > line count returns the whole log unchanged."""
        with patch("api.routers.cases.CaseLogRepository") as mock_repo, patch("api.routers.cases.read_case_log_file"):
            mock_repo.return_value.get.return_value = "l1\nl2"
            response = _run(get_case_log(_VALID_HASH, MagicMock(), 10))
        assert response.body.decode() == "l1\nl2"

    def test_tail_applied_to_file_fallback_content(self):
        """tail is applied to whichever content (DB or file) was resolved."""
        with (
            patch("api.routers.cases.CaseLogRepository") as mock_repo,
            patch("api.routers.cases.read_case_log_file", return_value="f1\nf2\nf3\nf4"),
        ):
            mock_repo.return_value.get.return_value = ""
            response = _run(get_case_log(_VALID_HASH, MagicMock(), 3))
        assert response.body.decode() == "f2\nf3\nf4"

    def test_returns_plain_text_media_type(self):
        """Response is a PlainTextResponse with text/plain media type."""
        with patch("api.routers.cases.CaseLogRepository") as mock_repo, patch("api.routers.cases.read_case_log_file"):
            mock_repo.return_value.get.return_value = "log"
            response = _run(get_case_log(_VALID_HASH, MagicMock(), 0))
        assert isinstance(response, PlainTextResponse)
        assert response.media_type == "text/plain"
