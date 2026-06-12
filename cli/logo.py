from __future__ import annotations

import functools
import os
import shutil
import sys
from typing import Final

_COLOR_BORDER: Final = "\033[38;5;240m"
_COLOR_TEXT: Final = "\033[1;97m"
_COLOR_BRAND: Final = "\033[48;5;21;38;5;46m"
_COLOR_RESET: Final = "\033[0m"

_LOGO_WIDTH: Final = 65
_DEFAULT_COLS: Final = 80
_BRAND_INNER: Final = ">>>>>   MindStudio   <<<<<"
_SLOGAN_TEXT: Final = "THE END-TO-END TOOLCHAIN TO UNLEASH HUAWEI ASCEND COMPUTE"
_NO_COLOR_TERMS: Final = frozenset({"dumb", "unknown"})

LogoBlock = tuple[str, str, str, str]


@functools.cache
def _ensure_windows_console() -> None:
    if sys.platform == "win32":
        import colorama

        colorama.just_fix_windows_console()


def _build_logo_block(block_width: int = _LOGO_WIDTH) -> LogoBlock:
    border = "=" * block_width
    return border, _BRAND_INNER.center(block_width), _SLOGAN_TEXT.center(block_width), border


def _center_block_in_terminal(lines: LogoBlock, terminal_cols: int | None) -> LogoBlock:
    block_width = len(lines[0])
    if terminal_cols is None or terminal_cols <= block_width:
        return lines
    top, brand, slogan, bottom = lines
    return (
        top.center(terminal_cols),
        brand.center(terminal_cols),
        slogan.center(terminal_cols),
        bottom.center(terminal_cols),
    )


def _colorize_logo_block(lines: LogoBlock) -> LogoBlock:
    top, brand_line, slogan_line, bottom = lines
    before, after = brand_line.split("MindStudio", 1)
    return (
        f"{_COLOR_BORDER}{top}{_COLOR_RESET}",
        f"{_COLOR_TEXT}{before}{_COLOR_BRAND}MindStudio{_COLOR_RESET}{_COLOR_TEXT}{after}{_COLOR_RESET}",
        f"{_COLOR_TEXT}{slogan_line}{_COLOR_RESET}",
        f"{_COLOR_BORDER}{bottom}{_COLOR_RESET}",
    )


def _terminal_cols() -> int:
    return shutil.get_terminal_size(fallback=(_DEFAULT_COLS, 24)).columns


def _supports_color() -> bool:
    if not sys.stderr.isatty():
        return False
    term = os.environ.get("TERM")
    if sys.platform == "win32":
        return term is None or term not in _NO_COLOR_TERMS
    if term is None:
        return False
    return term not in _NO_COLOR_TERMS


def render_logo(*, color: bool, width: int = _LOGO_WIDTH, terminal_cols: int | None = None) -> str:
    lines = _center_block_in_terminal(_build_logo_block(width), terminal_cols)
    if color:
        lines = _colorize_logo_block(lines)
    return "\n".join(lines)


def print_logo() -> None:
    _ensure_windows_console()
    sys.stderr.write(
        render_logo(
            color=_supports_color(),
            width=_LOGO_WIDTH,
            terminal_cols=_terminal_cols(),
        )
    )
    sys.stderr.write("\n\n")
