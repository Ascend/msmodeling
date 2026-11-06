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

"""Hardcoded defaults for unpublished ``scripts/`` tooling.

Process env defaults (UV / HF / base branch / cache) live in
``scripts/defaults.env``. Test-map path layout and the default download URL are
Python constants here. The URL template and its host allowlist may be overridden
at runtime; see :func:`build_test_map_url`.

``scripts/defaults.env`` grammar (must match ``scripts/lib/common.sh``):

- Empty lines are ignored.
- ``#`` starts a comment that runs to end of line (including mid-line).
- ``KEY=VALUE`` assignments; surrounding whitespace on key/value is stripped.
- Values must not contain ``#`` (it always begins a comment).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final
from urllib.parse import quote, urlsplit

_DEFAULTS_ENV: Final = Path(__file__).resolve().parent.parent / "defaults.env"
_PYTEST_XDIST_ARGS: Final = ("-n", "auto", "--dist", "worksteal")

# Not env vars — internal layout. ``{branch}`` is the source branch name.
_TEST_MAP_FILENAME: Final = "test_map.json"
_TEST_MAP_URL_TEMPLATE: Final = (
    "https://gitcode.com/api/v5/repos/eveyin1/msmodeling_pr/raw/test_map.json?ref=test-map/{branch}"
)
_TEST_MAP_ALLOWED_HOSTS: Final = frozenset({"gitcode.com"})

# Keys applied as process env defaults (setdefault). Keep in sync with common.sh.
_DEFAULT_ENV_KEYS: Final = frozenset(
    {
        "UV_INDEX_URL",
        "HF_ENDPOINT",
        "MSMODELING_TEST_BASE_BRANCH",
        "MSMODELING_CACHE",
    },
)


def parse_defaults_env_line(raw_line: str) -> tuple[str, str] | None:
    """Parse one defaults.env line; same rules as ``scripts/lib/common.sh``.

    Returns ``(key, value)`` or ``None`` when the line is blank/comment-only
    or not an assignment.
    """
    # Match bash: _line="${_line%%#*}" then trim.
    line = raw_line.split("#", 1)[0].strip()
    if not line or "=" not in line:
        return None
    key, value = line.split("=", 1)
    key = key.strip()
    value = value.strip()
    if not key:
        return None
    return key, value


def _parse_defaults_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        parsed = parse_defaults_env_line(raw_line)
        if parsed is None:
            continue
        key, value = parsed
        if key in _DEFAULT_ENV_KEYS:
            values[key] = value
    return values


_DEFAULTS: Final = _parse_defaults_env(_DEFAULTS_ENV)

DEFAULT_UV_INDEX_URL: Final = _DEFAULTS["UV_INDEX_URL"]
DEFAULT_HF_ENDPOINT: Final = _DEFAULTS["HF_ENDPOINT"]
DEFAULT_BASE_BRANCH: Final = _DEFAULTS["MSMODELING_TEST_BASE_BRANCH"]
DEFAULT_MSMODELING_CACHE: Final = _DEFAULTS["MSMODELING_CACHE"]
PYTEST_XDIST_ARGS: Final = _PYTEST_XDIST_ARGS


def _validate_test_map_template(template: str) -> None:
    """Validate the URL template independently from a concrete branch name."""
    if template.count("{branch}") != 1:
        raise ValueError("MSMODELING_TEST_MAP_URL_TEMPLATE must contain one {branch} placeholder")
    try:
        url = template.format(branch="branch")
    except (KeyError, ValueError) as exc:
        raise ValueError("MSMODELING_TEST_MAP_URL_TEMPLATE contains an invalid placeholder") from exc
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("MSMODELING_TEST_MAP_URL_TEMPLATE must produce an HTTP(S) URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("Credentials are not allowed in MSMODELING_TEST_MAP_URL_TEMPLATE")


def _allowed_test_map_hosts() -> frozenset[str]:
    raw = os.environ.get("MSMODELING_TEST_MAP_ALLOWED_HOSTS")
    if raw is None:
        return _TEST_MAP_ALLOWED_HOSTS
    hosts = frozenset(host.strip().lower() for host in raw.split(",") if host.strip())
    if not hosts:
        raise ValueError("MSMODELING_TEST_MAP_ALLOWED_HOSTS must contain at least one hostname")
    return hosts


def build_test_map_url(base_branch: str) -> str:
    """Return the validated test_map download URL for *base_branch*."""
    branch = base_branch.strip().lstrip("/")
    if not branch:
        raise ValueError("base_branch must be non-empty")
    template = os.environ.get("MSMODELING_TEST_MAP_URL_TEMPLATE", _TEST_MAP_URL_TEMPLATE).strip()
    _validate_test_map_template(template)
    url = template.format(branch=quote(branch, safe="/"))
    hostname = urlsplit(url).hostname
    if hostname is None or hostname.lower() not in _allowed_test_map_hosts():
        raise ValueError(f"test_map URL host {hostname!r} is not allowed")
    return url


def default_test_map_path(*, cache_dir: str | Path | None = None, base_branch: str) -> Path:
    """Return ``{cache}/test_map/{base_branch}/test_map.json``.

    Branch is kept as path segments (e.g. ``poc/AiClusterHub``) so different
    base branches never share one cached file.
    """
    root = Path(cache_dir) if cache_dir is not None else Path(DEFAULT_MSMODELING_CACHE)
    branch = base_branch.strip().lstrip("/")
    if not branch:
        raise ValueError("base_branch must be non-empty")
    return root / "test_map" / branch / _TEST_MAP_FILENAME
