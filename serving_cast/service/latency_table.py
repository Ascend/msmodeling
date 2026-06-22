# Copyright (c) 2026-2026 Huawei Technologies Co., Ltd.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from .base_throughput_optimizer import BaseThroughputOptimizer
    from .utils import OptimizerData


@dataclass(frozen=True)
class ForwardShapeKey:
    is_decode: bool
    model_concurrency: int
    query_len: int
    seq_len: int
    image_batch_size: int | None = None
    image_height: int | None = None
    image_width: int | None = None


@dataclass(frozen=True)
class ForwardLatencyRecord:
    latency_ms: float
    memory_left_gb: float
    breakdowns: str
    raw_breakdowns: dict[str, dict[str, float]] = field(default_factory=dict)


def _unique_keys(keys: Iterable[ForwardShapeKey]) -> list[ForwardShapeKey]:
    return list(dict.fromkeys(keys))


def _validate_forward_shape_key(key: ForwardShapeKey) -> None:
    if key.model_concurrency <= 0 or key.query_len <= 0 or key.seq_len < key.query_len:
        raise ValueError(f"Invalid forward shape key: {key}")


class ForwardLatencyTable:
    def __init__(
        self,
        optimizer: "BaseThroughputOptimizer",
        optimizer_data: "OptimizerData",
    ) -> None:
        self.optimizer = optimizer
        self.optimizer_data = optimizer_data
        self.memory_exceeded_key: ForwardShapeKey | None = None
        self.records: dict[ForwardShapeKey, ForwardLatencyRecord] = {}

    def prefetch(self, keys: Iterable[ForwardShapeKey]) -> None:
        unique_keys = _unique_keys(keys)
        for key in unique_keys:
            _validate_forward_shape_key(key)

        if self.memory_exceeded_key is not None:
            return

        for key in unique_keys:
            if key in self.records:
                if self._should_stop_on_record(key, self.records[key]):
                    break
                continue
            cached_record = self.optimizer._get_cached_forward_latency_record(key)
            if cached_record is not None:
                record = cached_record
            else:
                record = self._compute(key)
            self._store_record(key, record)
            if self._should_stop_on_record(key, record):
                break

    def get(self, key: ForwardShapeKey) -> ForwardLatencyRecord:
        _validate_forward_shape_key(key)
        if key not in self.records:
            self.prefetch([key])
        if key not in self.records:
            raise RuntimeError(f"Latency table stopped at {self.memory_exceeded_key}; missing record for {key}.")
        return self.records[key]

    def _compute(self, key: ForwardShapeKey) -> ForwardLatencyRecord:
        _validate_forward_shape_key(key)
        return self.optimizer._compute_forward_latency_record(key, self.optimizer_data)

    def _store_record(self, key: ForwardShapeKey, record: ForwardLatencyRecord) -> None:
        self.records[key] = record
        self.optimizer._cache_forward_latency_record(key, record)

    def _should_stop_on_record(self, key: ForwardShapeKey, record: ForwardLatencyRecord) -> bool:
        if record.memory_left_gb < 0:
            self.memory_exceeded_key = key
            return True
        return False
