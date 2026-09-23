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

"""Serializable kernel-query demands captured from profiling lookups."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import threading
from typing import Any, Iterable


QUERY_DEMAND_SCHEMA_VERSION = 1
QUERY_TRACE_DIR_ENV = "MSMODELING_SHAPE_QUERY_TRACE_DIR"
QUERY_TRACE_MODEL_ENV = "MSMODELING_SHAPE_QUERY_MODEL"
QUERY_TRACE_WORKLOAD_ENV = "MSMODELING_SHAPE_QUERY_WORKLOAD"


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return str(value)


def _shape_tuple(value: Iterable[Iterable[int]]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(dim) for dim in shape) for shape in value)


@dataclass(frozen=True)
class KernelQueryDemand:
    """One normalized query attempted against a kernel CSV."""

    projector_version: str
    op_name: str
    kernel_type: str
    query_mode: str
    input_shapes: tuple[tuple[int, ...], ...] = ()
    output_shapes: tuple[tuple[int, ...], ...] = ()
    input_dtypes: tuple[str, ...] = ()
    output_dtypes: tuple[str, ...] = ()
    attributes: dict[str, Any] = field(default_factory=dict, compare=False, hash=False)
    tensor_parallel_size: int | None = None
    expert_parallel_size: int | None = None
    candidate_rank: int = 0
    model_id: str = ""
    workload_id: str = ""

    @property
    def signature(self) -> str:
        """Return a stable identity excluding trace provenance."""
        payload = self.to_dict()
        payload.pop("model_id", None)
        payload.pop("workload_id", None)
        return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": QUERY_DEMAND_SCHEMA_VERSION,
            "projector_version": self.projector_version,
            "op_name": self.op_name,
            "kernel_type": self.kernel_type,
            "query_mode": self.query_mode,
            "input_shapes": [list(shape) for shape in self.input_shapes],
            "output_shapes": [list(shape) for shape in self.output_shapes],
            "input_dtypes": list(self.input_dtypes),
            "output_dtypes": list(self.output_dtypes),
            "attributes": _json_value(self.attributes),
            "tensor_parallel_size": self.tensor_parallel_size,
            "expert_parallel_size": self.expert_parallel_size,
            "candidate_rank": self.candidate_rank,
            "model_id": self.model_id,
            "workload_id": self.workload_id,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "KernelQueryDemand":
        schema_version = payload.get("schema_version")
        if schema_version != QUERY_DEMAND_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported kernel-query demand schema version: {schema_version!r}; "
                f"expected {QUERY_DEMAND_SCHEMA_VERSION}"
            )
        return cls(
            projector_version=str(payload["projector_version"]),
            op_name=str(payload["op_name"]),
            kernel_type=str(payload["kernel_type"]),
            query_mode=str(payload["query_mode"]),
            input_shapes=_shape_tuple(payload.get("input_shapes", ())),
            output_shapes=_shape_tuple(payload.get("output_shapes", ())),
            input_dtypes=tuple(str(value) for value in payload.get("input_dtypes", ())),
            output_dtypes=tuple(str(value) for value in payload.get("output_dtypes", ())),
            attributes=dict(payload.get("attributes", {})),
            tensor_parallel_size=payload.get("tensor_parallel_size"),
            expert_parallel_size=payload.get("expert_parallel_size"),
            candidate_rank=int(payload.get("candidate_rank", 0)),
            model_id=str(payload.get("model_id", "")),
            workload_id=str(payload.get("workload_id", "")),
        )


def projector_version_from_mapping(op_mapping: dict[str, Any]) -> str:
    """Build a stable software-stack identity from ``op_mapping.yaml``."""
    fields = (
        ("device", op_mapping.get("device", "unknown")),
        ("vllm", op_mapping.get("version", "unknown")),
        ("torch", op_mapping.get("pytorch_version", "unknown")),
        ("cann", op_mapping.get("cann_version", "unknown")),
        ("plugin", op_mapping.get("op_plugin_version", "unknown")),
    )
    suffix = ";".join(f"{name}={value}" for name, value in fields)
    return f"cann-backend-projector/v1;{suffix}"


class QueryDemandTraceWriter:
    """Process-local, de-duplicating JSONL trace writer."""

    def __init__(self, trace_dir: Path):
        self.trace_dir = trace_dir
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.trace_path = trace_dir / f"query-demands-{os.getpid()}.jsonl"
        self._seen: set[str] = set()
        self._lock = threading.Lock()

    def record(self, demand: KernelQueryDemand) -> None:
        signature = demand.signature
        with self._lock:
            if signature in self._seen:
                return
            self._seen.add(signature)
            line = json.dumps(demand.to_dict(), ensure_ascii=True, sort_keys=True)
            with self.trace_path.open("a", encoding="utf-8", newline="\n") as trace_file:
                trace_file.write(line)
                trace_file.write("\n")


_WRITERS: dict[Path, QueryDemandTraceWriter] = {}
_WRITERS_LOCK = threading.Lock()


def trace_writer_from_environment() -> QueryDemandTraceWriter | None:
    raw_trace_dir = os.environ.get(QUERY_TRACE_DIR_ENV)
    if not raw_trace_dir:
        return None
    trace_dir = Path(raw_trace_dir).resolve()
    with _WRITERS_LOCK:
        writer = _WRITERS.get(trace_dir)
        if writer is None:
            writer = QueryDemandTraceWriter(trace_dir)
            _WRITERS[trace_dir] = writer
        return writer


def load_query_demand_traces(trace_dir: Path) -> list[KernelQueryDemand]:
    """Load and globally de-duplicate all worker traces."""
    demands: list[KernelQueryDemand] = []
    seen: set[str] = set()
    for trace_path in sorted(trace_dir.glob("query-demands-*.jsonl")):
        with trace_path.open("r", encoding="utf-8") as trace_file:
            for line_number, raw_line in enumerate(trace_file, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    demand = KernelQueryDemand.from_dict(json.loads(line))
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                    raise ValueError(f"Invalid query demand at {trace_path}:{line_number}: {error}") from error
                if demand.signature in seen:
                    continue
                seen.add(demand.signature)
                demands.append(demand)
    return demands
