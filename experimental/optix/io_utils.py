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

import logging
import os
from pathlib import Path
from typing import Iterator, Optional, Union

logger = logging.getLogger(__name__)

DEFAULT_MAX_READ_FILE_SIZE = 10 * 1024 * 1024 * 1024
FORMULA_PREFIXES = ("=", "+", "-", "@")
PATH_MAX_LENGTH = 4096


def ensure_existing_file(path: Union[os.PathLike, str], max_size: Optional[int] = DEFAULT_MAX_READ_FILE_SIZE) -> Path:
    file_path = Path(path)

    # Reject paths exceeding PATH_MAX
    if len(str(file_path)) > PATH_MAX_LENGTH:
        raise ValueError(f"File path exceeds PATH_MAX ({PATH_MAX_LENGTH}): {file_path!r}")

    # Symlink: log at debug level, do not block execution
    if file_path.is_symlink():
        logger.debug(f"Path is a symbolic link, resolved: {path!r} -> {file_path.resolve()!r}")
    file_path = file_path.resolve()

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path!r}")
    if not file_path.is_file():
        raise ValueError(f"Expect a file, not a directory: {file_path!r}")
    if max_size is not None and file_path.stat().st_size > max_size:
        raise ValueError(f"File is too large: {file_path!r}")
    return file_path


def open_file(path: Union[os.PathLike, str], mode: str = "r", *args, **kwargs):
    if "r" in mode and "+" not in mode:
        ensure_existing_file(path)
    file_path = Path(path).resolve()
    if "b" in mode:
        return open(file_path, mode, *args, **kwargs)  # pylint: disable=unspecified-encoding
    return open(file_path, mode, *args, encoding=kwargs.pop("encoding", "utf-8"), **kwargs)


def walk_files(path: Union[os.PathLike, str]) -> Iterator[Path]:
    root = Path(path).resolve()
    if not root.exists():
        return
    if root.is_file():
        yield root
        return
    for p in root.rglob("*"):
        if p.is_file():
            # Skip symbolic links with debug-level warning
            if p.is_symlink():
                logger.debug(f"Skipping symbolic link during walk: {p!r}")
                continue
            yield p


def sanitize_csv_value(value):
    """Sanitize CSV value to prevent formula injection."""
    if isinstance(value, str) and value.startswith(FORMULA_PREFIXES):
        return "'" + value
    return value
