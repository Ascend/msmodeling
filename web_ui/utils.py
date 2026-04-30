from __future__ import annotations

import ast
import hashlib
import json
from typing import Any


def parse_scalar_or_list(raw: Any, cast=str) -> list[Any]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [cast(v) for v in raw]
    if isinstance(raw, tuple):
        return [cast(v) for v in raw]
    if isinstance(raw, bool):
        return [raw]
    text = str(raw).strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        if not inner:
            return []
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return [cast(v.strip().strip("'\"")) for v in inner.split(",") if v.strip()]
        if not isinstance(parsed, list):
            raise ValueError(f"Expected list format, got: {raw}")
        return [cast(v) for v in parsed]
    if "," in text:
        return [cast(v.strip()) for v in text.split(",") if v.strip()]
    return [cast(text)]


def parse_optional_number(raw: Any, cast=float) -> Any:
    if raw in (None, "", "None", "none", "auto"):
        return None
    return cast(raw)


def stable_hash(data: dict[str, Any]) -> str:
    norm = json.dumps(data, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]


def bool_from_ui(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "on"}


def normalize_value(value: Any) -> Any:
    if isinstance(value, list):
        return [normalize_value(v) for v in value]
    if isinstance(value, tuple):
        return [normalize_value(v) for v in value]
    return value
