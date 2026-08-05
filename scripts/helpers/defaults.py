"""Hardcoded defaults for unpublished ``scripts/`` tooling.

Process env defaults (UV / HF / base branch / cache) live in
``scripts/defaults.env``. Test-map path layout and OBS URL template are
Python constants here — not environment variables.

``scripts/defaults.env`` grammar (must match ``scripts/lib/common.sh``):

- Empty lines are ignored.
- ``#`` starts a comment that runs to end of line (including mid-line).
- ``KEY=VALUE`` assignments; surrounding whitespace on key/value is stripped.
- Values must not contain ``#`` (it always begins a comment).
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

_DEFAULTS_ENV: Final = Path(__file__).resolve().parent.parent / "defaults.env"
_PYTEST_XDIST_ARGS: Final = ("-n", "auto", "--dist", "worksteal")

# Not env vars — internal layout / OBS contract only.
_TEST_MAP_FILENAME: Final = "test_map.json"
_TEST_MAP_URL_TEMPLATE: Final = (
    "https://mindstudio-pr.obs.cn-north-4.myhuaweicloud.com/msmodeling/sync/{branch}/test_map.json"
)

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


def build_test_map_url(base_branch: str) -> str:
    """Return the OBS test_map URL for *base_branch* (may contain ``/``)."""
    branch = base_branch.strip().lstrip("/")
    if not branch:
        raise ValueError("base_branch must be non-empty")
    return _TEST_MAP_URL_TEMPLATE.format(branch=branch)


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
