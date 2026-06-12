"""Regression tests for cli.logo rendering, color gating, and stderr emission."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from cli.logo import (
    _BRAND_INNER,
    _LOGO_WIDTH,
    _SLOGAN_TEXT,
    _ensure_windows_console,
    _supports_color,
    print_logo,
    render_logo,
)

if TYPE_CHECKING:
    import pytest

_ANSI_ESCAPE_RE = re.compile(r"\033\[[0-9;]*m")


def test_render_logo_color_false_four_lines_border_width_65() -> None:
    output = render_logo(color=False)
    lines = output.splitlines()
    assert len(lines) == 4
    assert len(lines[0]) == 65
    assert lines[0] == "=" * 65
    assert lines[3] == "=" * 65


def test_render_logo_color_true_has_ansi() -> None:
    output = render_logo(color=True)
    assert "\033[" in output


def test_brand_line_centered_in_block() -> None:
    lines = render_logo(color=False).splitlines()
    assert lines[1] == _BRAND_INNER.center(_LOGO_WIDTH)


def test_slogan_line_centered_in_block() -> None:
    lines = render_logo(color=False).splitlines()
    assert lines[2] == _SLOGAN_TEXT.center(_LOGO_WIDTH)


def test_terminal_centering_at_80_cols() -> None:
    lines = render_logo(color=False, terminal_cols=80).splitlines()
    assert all(len(line) == 80 for line in lines)


def test_color_strip_equals_plain() -> None:
    plain_lines = render_logo(color=False).splitlines()
    colored_lines = render_logo(color=True).splitlines()
    for plain, colored in zip(plain_lines, colored_lines, strict=True):
        assert _ANSI_ESCAPE_RE.sub("", colored) == plain


def test_render_logo_no_leading_newline() -> None:
    output = render_logo(color=False)
    assert not output.startswith("\n")


def test_supports_color_not_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.stderr.isatty", lambda: False)
    assert _supports_color() is False


def test_supports_color_term_dumb(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.stderr.isatty", lambda: True)
    monkeypatch.setenv("TERM", "dumb")
    assert _supports_color() is False


def test_supports_color_term_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.stderr.isatty", lambda: True)
    monkeypatch.setenv("TERM", "unknown")
    assert _supports_color() is False


def test_supports_color_posix_missing_term(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.platform", "linux")
    monkeypatch.setattr("sys.stderr.isatty", lambda: True)
    monkeypatch.delenv("TERM", raising=False)
    assert _supports_color() is False


def test_supports_color_win32_without_term(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.platform", "win32")
    monkeypatch.setattr("sys.stderr.isatty", lambda: True)
    monkeypatch.delenv("TERM", raising=False)
    assert _supports_color() is True


def test_print_logo_trailing_blank(monkeypatch: pytest.MonkeyPatch) -> None:
    writes: list[str] = []

    def capture_write(data: str) -> None:
        writes.append(data)

    monkeypatch.setattr("sys.stderr.write", capture_write)
    monkeypatch.setattr("cli.logo._supports_color", lambda: False)
    monkeypatch.setattr("cli.logo._terminal_cols", lambda: 80)

    print_logo()

    assert len(writes) == 2
    assert writes[1] == "\n\n"


def test_windows_console_init_once(monkeypatch: pytest.MonkeyPatch) -> None:
    _ensure_windows_console.cache_clear()
    call_count = 0

    def mock_fix() -> None:
        nonlocal call_count
        call_count += 1

    import colorama

    monkeypatch.setattr("sys.platform", "win32")
    monkeypatch.setattr(colorama, "just_fix_windows_console", mock_fix)
    monkeypatch.setattr("cli.logo._supports_color", lambda: False)
    monkeypatch.setattr("cli.logo._terminal_cols", lambda: 80)
    monkeypatch.setattr("sys.stderr.write", lambda _: None)

    print_logo()
    print_logo()

    assert call_count == 1
    _ensure_windows_console.cache_clear()
