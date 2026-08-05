"""Apply unpublished scripts defaults into an environment dict (no global mutate)."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from scripts.helpers.defaults import (
    DEFAULT_BASE_BRANCH,
    DEFAULT_HF_ENDPOINT,
    DEFAULT_MSMODELING_CACHE,
    DEFAULT_UV_INDEX_URL,
)

if TYPE_CHECKING:
    from collections.abc import Mapping


_TEST_DEFAULTS: Mapping[str, str] = {
    "UV_INDEX_URL": DEFAULT_UV_INDEX_URL,
    "HF_ENDPOINT": DEFAULT_HF_ENDPOINT,
    "MSMODELING_TEST_BASE_BRANCH": DEFAULT_BASE_BRANCH,
    "MSMODELING_CACHE": DEFAULT_MSMODELING_CACHE,
}


def apply_test_defaults(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return a copy of *env* (or ``os.environ``) with script defaults filled.

    Never mutates ``os.environ``. Empty string is treated as unset.
    """
    target = dict(os.environ if env is None else env)
    for key, value in _TEST_DEFAULTS.items():
        current = target.get(key)
        if current is None or current.strip() == "":
            target[key] = value
    return target
