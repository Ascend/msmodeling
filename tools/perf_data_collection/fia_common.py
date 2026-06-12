from __future__ import annotations

from typing import Iterable


def split_metadata_field(raw_value: str) -> list[str]:
    cleaned = (raw_value or "").strip().strip('"')
    return [item.strip() for item in cleaned.split(";")]


def parse_shape_or_none(raw_shape: str | None) -> tuple[int, ...] | None:
    cleaned = (raw_shape or "").strip()
    if not cleaned:
        return None
    return tuple(int(part.strip()) for part in cleaned.split(",") if part.strip())


def parse_runtime_int(raw_value: str | None) -> int | None:
    cleaned = (raw_value or "").strip()
    if not cleaned:
        return None
    return int(cleaned)


def parse_runtime_int_list(raw_value: str | None) -> list[int] | None:
    cleaned = (raw_value or "").strip()
    if not cleaned:
        return None
    normalized = cleaned.replace(";", ",")
    values = [item.strip() for item in normalized.split(",") if item.strip()]
    if not values:
        return None
    return [int(item) for item in values]


def shape_numel(shape: tuple[int, ...] | None) -> int:
    if not shape:
        return 0
    total = 1
    for dim in shape:
        total *= dim
    return total


def shape_to_text(shape: Iterable[int] | None) -> str:
    if not shape:
        return ""
    return ",".join(str(int(dim)) for dim in shape)
