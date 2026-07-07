# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
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
"""OptiX loguru configuration and structured logging constants."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from collections import deque
from contextlib import suppress
from enum import Enum
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

from loguru import logger

LOG_FORMAT_INFO = (
    "<green>{time:HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{extra[run_id]}</cyan> | "
    "<yellow>{extra[stage]}</yellow> | "
    "{message}"
)

LOG_FORMAT_DEBUG = (
    "<green>{time:HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{extra[run_id]}</cyan> | "
    "<yellow>{extra[stage]}</yellow> | "
    "<dim>{file.name}:{line}</dim> | "
    "{message}"
)

LOG_FORMAT = LOG_FORMAT_INFO

DEFAULT_LOG_EXTRA = {"run_id": "-", "stage": "-", "engine": "-"}


class LogStage(Enum):
    """High-level optimizer run stages for structured logging."""

    INIT = "init"
    BASELINE = "baseline"
    SEARCH = "search"
    FINE_TUNE = "fine_tune"
    EVALUATE = "evaluate"
    SIM_START = "sim_start"
    SIM_STOP = "sim_stop"
    BENCH_RUN = "bench_run"
    PERSIST = "persist"
    CLEANUP = "cleanup"
    DONE = "done"


def resolve_log_level() -> str:
    """Resolve log level from OPTIX_LOG_LEVEL or legacy MODELEVALSTATE_LEVEL."""
    raw = os.getenv("OPTIX_LOG_LEVEL")
    if raw:
        return raw.upper()
    legacy = os.getenv("MODELEVALSTATE_LEVEL")
    if legacy:
        return legacy.upper()
    return "INFO"


def resolve_log_format(level: str) -> str:
    """Return log format with file:line for DEBUG/TRACE, plain format otherwise."""
    if level.upper() in {"DEBUG", "TRACE"}:
        return LOG_FORMAT_DEBUG
    return LOG_FORMAT_INFO


def format_command(command: Sequence[str]) -> str:
    """Format argv as a single readable shell line."""
    return shlex.join(command)


def read_log_tail(log_path: str | Path | None, *, lines: int = 10) -> str | None:
    """Return the last ``lines`` of a log file, or None if unavailable."""
    if not log_path:
        return None
    from optix.io_utils import open_file

    path = Path(log_path) if isinstance(log_path, str) else log_path
    if not path.is_file():
        return None
    encodings = ("utf-8", "latin-1", "gbk", "cp1252")
    tail_lines: deque[str] | None = None
    for encoding in encodings:
        try:
            with open_file(path, "r", encoding=encoding, errors="replace") as handle:
                tail_lines = deque(handle, maxlen=lines)
            break
        except (UnicodeError, OSError):
            if encoding == encodings[-1]:
                return None
            continue
    if tail_lines is None or not tail_lines:
        return None
    return "".join(tail_lines).rstrip("\n")


def format_subprocess_start(
    command: Sequence[str],
    log_path: str | Path,
    *,
    pid: int | None = None,
) -> str:
    """Multiline INFO message for subprocess startup."""
    lines = [
        "Starting service subprocess",
        f"  command: {format_command(command)}",
        f"  log: {log_path}",
    ]
    if pid is not None:
        lines.append(f"  pid: {pid}")
    return "\n".join(lines)


def format_subprocess_failure(
    command: Sequence[str],
    return_code: int | None,
    log_path: str | Path | None,
    *,
    log_tail: str | None = None,
) -> str:
    """Single user-facing subprocess failure message without nested wrappers."""
    exit_label = return_code if return_code is not None else "?"
    lines = [
        f"Service subprocess failed (exit={exit_label})",
        f"  command: {format_command(command)}",
    ]
    if log_path:
        lines.append(f"  log: {log_path}")
    if log_tail is None and log_path:
        log_tail = read_log_tail(log_path)
    if log_tail:
        lines.append("  log tail:")
        lines.extend(f"    {line.strip()}" for line in log_tail.splitlines() if line.strip())
    elif log_path:
        path = Path(log_path)
        if path.is_file() and path.stat().st_size == 0:
            lines.append("  hint: log file is empty; service may still be starting or exited before writing output")
    return "\n".join(lines)


def format_evaluation_failure(scheduler: object, error: object) -> str:
    """Build subprocess failure text while simulator logs are still readable."""
    if isinstance(error, subprocess.SubprocessError) and error.args:
        message = str(error.args[0])
        if "Service subprocess failed" in message:
            return message

    simulator = getattr(scheduler, "simulator", None)
    if simulator is None:
        return str(error)

    command = list(getattr(simulator, "command", None) or [])
    log_path = getattr(simulator, "run_log", None)
    return_code = None
    process = getattr(simulator, "process", None)
    if process is not None:
        return_code = process.returncode

    log_tail = None
    get_last_log = getattr(simulator, "get_last_log", None)
    if callable(get_last_log):
        raw_tail = get_last_log(10)
        if isinstance(raw_tail, str) and raw_tail.strip():
            log_tail = raw_tail

    if isinstance(error, subprocess.SubprocessError):
        if log_tail or log_path or command:
            return format_subprocess_failure(command, return_code, log_path, log_tail=log_tail)
        return str(error)

    if log_tail:
        return format_subprocess_failure(command, return_code, log_path, log_tail=log_tail)

    if isinstance(error, TimeoutError) and (log_path or command):
        return format_subprocess_failure(command, return_code, log_path, log_tail=None)

    return str(error)


@cache
def configure_logger() -> None:
    """Configure optix loguru handler without clearing unrelated handlers."""
    level = resolve_log_level()
    with suppress(ValueError):
        logger.remove(0)
    logger.configure(extra=DEFAULT_LOG_EXTRA.copy())
    logger.add(
        sys.stderr,
        level=level,
        format=resolve_log_format(level),
        enqueue=True,
        diagnose=level == "DEBUG",
        backtrace=level == "DEBUG",
    )
