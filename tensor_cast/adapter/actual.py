import collections
import dataclasses
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

from tensor_cast.runtime import Runtime, RuntimeEvent


@dataclasses.dataclass(frozen=True)
class ActualOpSummary:
    name: str
    count: int
    total_time_s: float
    avg_time_s: float
    shape_variants: Tuple[str, ...] = ()
    coverage: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "count": self.count,
            "total_time_s": self.total_time_s,
            "avg_time_s": self.avg_time_s,
            "shape_variants": list(self.shape_variants),
            "coverage": dict(self.coverage),
        }


@dataclasses.dataclass(frozen=True)
class ActualSummary:
    case_name: str
    total_forward_time_s: float
    ops: Dict[str, ActualOpSummary]
    perf_model_name: Optional[str] = None
    coverage: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def get_op(self, name: str) -> Optional[ActualOpSummary]:
        return self.ops.get(name)

    def high_time_ops(self, min_total_time_s: float) -> List[ActualOpSummary]:
        return [op for op in self.ops.values() if op.total_time_s >= min_total_time_s]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_name": self.case_name,
            "total_forward_time_s": self.total_forward_time_s,
            "perf_model_name": self.perf_model_name,
            "coverage": dict(self.coverage),
            "ops": {name: op.to_dict() for name, op in sorted(self.ops.items())},
        }


def _iter_tensors(value: Any) -> Iterable[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_tensors(item)


def _shape_signature(event: RuntimeEvent) -> str:
    shapes = [list(tensor.shape) for tensor in _iter_tensors(event.op_invoke_info.args)]
    kwarg_shapes = [list(tensor.shape) for tensor in _iter_tensors(event.op_invoke_info.kwargs)]
    if kwarg_shapes:
        shapes.extend(kwarg_shapes)
    return str(shapes)


def build_actual_summary_from_events(
    events: Iterable[RuntimeEvent],
    case_name: str = "default",
    perf_model_name: Optional[str] = None,
    total_forward_time_s: Optional[float] = None,
    coverage: Optional[Dict[str, Any]] = None,
) -> ActualSummary:
    aggregated: Dict[str, Dict[str, Any]] = collections.defaultdict(
        lambda: {"count": 0, "total_time_s": 0.0, "shape_variants": set()}
    )
    inferred_perf_model_name = perf_model_name
    total_time_sum = 0.0

    for event in events:
        if inferred_perf_model_name is None and event.perf_results:
            inferred_perf_model_name = next(iter(event.perf_results))
        model_name = inferred_perf_model_name
        if model_name is None or model_name not in event.perf_results:
            duration_s = 0.0
        else:
            duration_s = event.perf_results[model_name].execution_time_s
        op_name = str(event.op_invoke_info.func)
        entry = aggregated[op_name]
        entry["count"] += 1
        entry["total_time_s"] += duration_s
        entry["shape_variants"].add(_shape_signature(event))
        total_time_sum += duration_s

    ops = {}
    for name, data in aggregated.items():
        count = int(data["count"])
        total_time = float(data["total_time_s"])
        ops[name] = ActualOpSummary(
            name=name,
            count=count,
            total_time_s=total_time,
            avg_time_s=total_time / count if count else 0.0,
            shape_variants=tuple(sorted(data["shape_variants"])),
        )

    return ActualSummary(
        case_name=case_name,
        total_forward_time_s=total_time_sum if total_forward_time_s is None else total_forward_time_s,
        ops=ops,
        perf_model_name=inferred_perf_model_name,
        coverage={} if coverage is None else coverage,
    )


def build_actual_summary_from_runtime(
    runtime: Runtime,
    case_name: str = "default",
    perf_model_name: Optional[str] = None,
    coverage: Optional[Dict[str, Any]] = None,
) -> ActualSummary:
    model_name = perf_model_name
    if model_name is None and runtime.perf_models:
        model_name = runtime.perf_models[0].name
    total_forward_time_s = None
    if model_name is not None:
        total_forward_time_s = runtime.total_execution_time_s().get(model_name)
    return build_actual_summary_from_events(
        runtime.event_list,
        case_name=case_name,
        perf_model_name=model_name,
        total_forward_time_s=total_forward_time_s,
        coverage=coverage,
    )
