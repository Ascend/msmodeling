"""Stable params-hash for result+log caching (Phase C).

A deterministic sha256 of (module_id, form_schema_version, params) so that two
submissions with the same module + schema version + form values hit the cache.
The JSON is canonicalized (sorted keys, ``default=str``) so field order /
non-JSON values don't fragment the cache. The schema version is part of the key
so a form-schema upgrade invalidates stale results (same params, new semantics
→ distinct hash).
"""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Any


def _normalize(obj: Any) -> Any:
    """Recursively normalize params for deterministic hashing.

    Handles types ``json.dumps(default=str)`` would stringify inconsistently:
    ``Enum`` → ``.value`` (so a submission sending the enum vs its raw value hash
    the same). Pure-JSON params (the normal frontend case) pass through unchanged,
    so existing cache keys are NOT affected.
    """
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        return {k: _normalize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_normalize(v) for v in obj]
    return obj


def compute_params_hash(module_id: str, form_schema_version: str, params: dict[str, Any]) -> str:
    """Return the sha256 hex of the canonical (module_id, version, params) blob."""
    canonical = json.dumps(
        {"module": module_id, "version": form_schema_version, "params": _normalize(params)},
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
