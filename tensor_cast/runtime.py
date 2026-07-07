import collections
import contextlib
import dataclasses
import json
import logging
import threading
import time
from typing import Any, Dict, List, Optional, Union

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from .device import DeviceProfile
from .performance_model.bound_analyzer import (
    COMMUNICATION_BOUND,
    COMPUTE_BOUND_GP,
    COMPUTE_BOUND_MMA,
    MEMORY_BOUND,
    BoundAnalyzer,
)
from .patch_torch import patch_torch
from .performance_model.base import CachingPerformanceModel, PerformanceModel
from .performance_model.memory_tracker import MemoryTracker
from .performance_model.op_invoke_info import OpInvokeInfo, Region

logger = logging.getLogger(__name__)

_current_runtime = threading.local()


def current_runtime():
    return getattr(_current_runtime, "value", None)


BoundComponentTotals = Dict[str, float]
BoundComponentsByModel = Dict[str, BoundComponentTotals]
_BOUND_COMPONENT_KEYS = (MEMORY_BOUND, COMMUNICATION_BOUND, COMPUTE_BOUND_MMA, COMPUTE_BOUND_GP)


def _default_bound_component_totals() -> BoundComponentTotals:
    return {key: 0.0 for key in _BOUND_COMPONENT_KEYS}


def _default_bound_components_by_model() -> BoundComponentsByModel:
    return collections.defaultdict(_default_bound_component_totals)


@dataclasses.dataclass
class RuntimeEvent:
    op_invoke_info: OpInvokeInfo
    perf_results: Dict[str, PerformanceModel.Result] = dataclasses.field(default_factory=dict)
    stream_id: int = 0
    dependency_token_ids: tuple[int, ...] = ()
    produced_token_ids: List[int] = dataclasses.field(default_factory=list)
    memory_aliases: List[tuple[torch.Tensor, torch.Tensor]] = dataclasses.field(default_factory=list)


@dataclasses.dataclass(frozen=True)
class OpAverageGroupKey:
    op_name: str
    bound: str = ""
    input_shapes: str = ""


@dataclasses.dataclass
class OpAverageGroupData:
    count: int = 0
    total_runtimes: Dict[str, float] = dataclasses.field(default_factory=lambda: collections.defaultdict(float))
    bound_components: BoundComponentsByModel = dataclasses.field(default_factory=_default_bound_components_by_model)


class Runtime(TorchDispatchMode):
    """
    Runtime of TensorCast that simulates the execution of a PyTorch program.
    """

    _INTERNAL_WAIT_AND_BIND = torch.ops.tensor_cast._internal_wait_and_bind.default
    _INTERNAL_RECORD = torch.ops.tensor_cast._internal_record.default

    def __deepcopy__(self, memo):
        return self

    def __init__(
        self,
        perf_models: Union[PerformanceModel, List[PerformanceModel]],
        device_profile: DeviceProfile,
        memory_tracker: Optional[MemoryTracker] = None,
    ):
        super().__init__()
        self.perf_models = perf_models if isinstance(perf_models, (list, tuple)) else [perf_models]
        self.perf_models = [
            perf_model if isinstance(perf_model, CachingPerformanceModel) else CachingPerformanceModel(perf_model)
            for perf_model in self.perf_models
        ]
        self.device_profile = device_profile
        self.memory_tracker: Optional[MemoryTracker] = memory_tracker
        self.op_invoke_infos: List[OpInvokeInfo] = []
        self.op_info_group: List[Union[OpInvokeInfo, Region]] = []
        self.event_list: List[RuntimeEvent] = []
        self._event_reference_ids: List[int] = []
        self._pending_wait_stream_id: Optional[int] = None
        self._pending_wait_dependency_token_ids: List[int] = []
        self._pending_wait_memory_aliases: List[tuple[torch.Tensor, torch.Tensor]] = []

        self.exit_stack = contextlib.ExitStack()

    @classmethod
    def is_infra_mode(cls):
        return True

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs

        if not torch.compiler.is_compiling():
            func_name = func.__qualname__ if hasattr(func, "__qualname__") else str(func)
            start = time.perf_counter() if logger.isEnabledFor(logging.DEBUG) else None
            out = func(*args, **kwargs)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "Operation '%s' executed in %.6f",
                    func_name,
                    time.perf_counter() - start,
                )

            op_invoke_info = OpInvokeInfo(func, args, kwargs, out)
            self.op_invoke_infos.append(op_invoke_info)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Recorded '%s': %s", func_name, op_invoke_info)

            return out
        else:
            return func(*args, **kwargs)

    def repeat_op_invoke_infos(self):
        region_id_to_op_invoke_infos = {}
        current_id = None
        for op_invoke_info in self.op_invoke_infos:
            if op_invoke_info.func == torch.ops.tensor_cast._internal_mark_region_begin.default:
                assert current_id is None, f"Already in region {current_id}, we do not support nested regions"
                current_id = op_invoke_info.args[1]
                assert current_id not in region_id_to_op_invoke_infos, f"Duplicated region id {current_id} found"
                region_id_to_op_invoke_infos[current_id] = Region(op_invoke_info)
            elif op_invoke_info.func == torch.ops.tensor_cast._internal_mark_region_end.default:
                current_id = op_invoke_info.args[1]
                assert current_id in region_id_to_op_invoke_infos, (
                    f"Region end with id {current_id} not paired with a region begin"
                )
                region_id_to_op_invoke_infos[current_id].finalize(op_invoke_info)
                self.op_info_group.append(region_id_to_op_invoke_infos[current_id])
                current_id = None
            elif op_invoke_info.func == torch.ops.tensor_cast._internal_copy_region.default:
                assert current_id is None, f"Already in region {current_id}, we do not support nested regions"
                copy_id = op_invoke_info.args[1]
                assert copy_id in region_id_to_op_invoke_infos, f"Regioin {copy_id} not marked before copy"
                self.op_info_group.append(
                    region_id_to_op_invoke_infos[copy_id].shallow_copy(op_invoke_info.args[0], op_invoke_info.out)
                )
            else:
                if current_id is not None:
                    region_id_to_op_invoke_infos[current_id].op_invoke_infos.append(op_invoke_info)
                else:
                    self.op_info_group.append(op_invoke_info)

    @staticmethod
    def _dedup_token_ids(token_ids: List[int]) -> tuple[int, ...]:
        return tuple(dict.fromkeys(token_ids))

    @staticmethod
    def _extract_tensor_token_ids(value: Any) -> List[int]:
        if isinstance(value, torch.Tensor):
            return [id(value)]
        if isinstance(value, (list, tuple)):
            token_ids: List[int] = []
            for item in value:
                token_ids.extend(Runtime._extract_tensor_token_ids(item))
            return token_ids
        if isinstance(value, dict):
            token_ids: List[int] = []
            for item in value.values():
                token_ids.extend(Runtime._extract_tensor_token_ids(item))
            return token_ids
        return []

    def _consume_pending_wait_context(
        self,
    ) -> tuple[int, tuple[int, ...], List[tuple[torch.Tensor, torch.Tensor]]]:
        if self._pending_wait_stream_id is None:
            return 0, (), []
        stream_id = self._pending_wait_stream_id
        dependency_token_ids = self._dedup_token_ids(self._pending_wait_dependency_token_ids)
        memory_aliases = self._pending_wait_memory_aliases
        self._pending_wait_stream_id = None
        self._pending_wait_dependency_token_ids.clear()
        self._pending_wait_memory_aliases = []
        return stream_id, dependency_token_ids, memory_aliases

    def _handle_wait_and_bind(self, op_invoke_info: OpInvokeInfo) -> None:
        stream_id = 0
        if len(op_invoke_info.args) > 1:
            stream_id = int(op_invoke_info.args[1])
        deps = op_invoke_info.args[2] if len(op_invoke_info.args) > 2 else []
        dep_token_ids = self._extract_tensor_token_ids(deps)
        if self._pending_wait_stream_id is not None and stream_id != self._pending_wait_stream_id:
            raise RuntimeError(f"Conflicting wait_and_bind stream ids ({self._pending_wait_stream_id} vs. {stream_id})")
        self._pending_wait_stream_id = stream_id
        self._pending_wait_dependency_token_ids.extend(dep_token_ids)
        if (
            len(op_invoke_info.args) > 0
            and isinstance(op_invoke_info.args[0], torch.Tensor)
            and isinstance(op_invoke_info.out, torch.Tensor)
        ):
            self._pending_wait_memory_aliases.append((op_invoke_info.args[0], op_invoke_info.out))

    def _handle_record(self, op_invoke_info: OpInvokeInfo) -> None:
        if not self.event_list:
            logger.warning("Ignoring _internal_record because no preceding runtime event exists.")
            return
        event = self.event_list[-1]
        if len(op_invoke_info.args) > 1:
            event.stream_id = int(op_invoke_info.args[1])
        token_ids = self._extract_tensor_token_ids(op_invoke_info.out)
        if not token_ids:
            return
        event.produced_token_ids = list(self._dedup_token_ids(event.produced_token_ids + token_ids))

    def _replay_single_op(self, op_invoke_info):
        if op_invoke_info.func == self._INTERNAL_WAIT_AND_BIND:
            self._handle_wait_and_bind(op_invoke_info)
            return
        if op_invoke_info.func == self._INTERNAL_RECORD:
            self._handle_record(op_invoke_info)
            return

        stream_id, dependency_token_ids, memory_aliases = self._consume_pending_wait_context()
        perf_results = {}
        for perf_model in self.perf_models:
            result = perf_model.process_op(op_invoke_info)
            perf_results[perf_model.name] = result
        self.event_list.append(
            RuntimeEvent(
                op_invoke_info=op_invoke_info,
                perf_results=perf_results,
                stream_id=stream_id,
                dependency_token_ids=dependency_token_ids,
                memory_aliases=memory_aliases,
            )
        )

    @classmethod
    def _is_multistream_anchor_op(cls, func) -> bool:
        return func in (cls._INTERNAL_WAIT_AND_BIND, cls._INTERNAL_RECORD)

    def _record_single_memory_invocation(self, op_invoke_info: OpInvokeInfo, reference_id: int) -> None:
        if self._is_multistream_anchor_op(op_invoke_info.func):
            return
        self.memory_tracker.record_single_op_invocation(op_invoke_info, reference_id)

    def _iter_flat_invocations(self) -> List[tuple[OpInvokeInfo, int]]:
        invocations: List[tuple[OpInvokeInfo, int]] = []
        for op_info_or_region in self.op_info_group:
            if not isinstance(op_info_or_region, Region):
                invocations.append((op_info_or_region, 0))
                continue
            reference_id = getattr(op_info_or_region, "reference_id")
            op_invoke_infos = getattr(op_info_or_region, "op_invoke_infos")
            invocations.extend((op_invoke_info, reference_id) for op_invoke_info in op_invoke_infos)
        return invocations

    @staticmethod
    def _event_duration_s(event: RuntimeEvent) -> float:
        if not event.perf_results:
            return 0.0
        return max(perf_result.execution_time_s for perf_result in event.perf_results.values())

    def _record_memory_invocations(self) -> None:
        if self.memory_tracker is None:
            return
        memory_events = self.event_list
        if len(self._event_reference_ids) != len(self.event_list):
            logger.warning(
                "Runtime event/reference mismatch for memory tracking: events=%d, references=%d.",
                len(self.event_list),
                len(self._event_reference_ids),
            )
        event_reference_id = {
            id(event): (self._event_reference_ids[index] if index < len(self._event_reference_ids) else 0)
            for index, event in enumerate(self.event_list)
        }
        # Multistream anchors are runtime control ops rather than model-semantic ops.
        # In particular, _internal_record publishes control tokens, not activations.
        # Skip anchors so activation-memory accounting tracks model tensors only.
        # MemoryTracker models tensor liveness from def-use order. Reordering by
        # simulated completion time can place a consumer before its producer and
        # incorrectly turn intermediate tensors into model inputs.
        for event in memory_events:
            reference_id = event_reference_id.get(id(event), 0)
            consumed_tensor_ids = set(
                self._extract_tensor_token_ids((event.op_invoke_info.args, event.op_invoke_info.kwargs))
            )
            for source_tensor, alias_tensor in event.memory_aliases:
                if id(alias_tensor) in consumed_tensor_ids:
                    self.memory_tracker.record_tensor_alias(source_tensor, alias_tensor, reference_id)
            self._record_single_memory_invocation(event.op_invoke_info, reference_id)

    def replay_op_invoke_infos(self):
        self._pending_wait_stream_id = None
        self._pending_wait_dependency_token_ids.clear()
        self.event_list.clear()
        self._event_reference_ids.clear()
        invocations = self._iter_flat_invocations()
        for op_invoke_info, reference_id in invocations:
            num_events_before_replay = len(self.event_list)
            self._replay_single_op(op_invoke_info)
            if len(self.event_list) > num_events_before_replay:
                self._event_reference_ids.append(reference_id)
        if self._pending_wait_stream_id is not None:
            logger.warning(
                "Dropping dangling _internal_wait_and_bind context on stream %s.",
                self._pending_wait_stream_id,
            )
            self._pending_wait_stream_id = None
            self._pending_wait_dependency_token_ids.clear()
            self._pending_wait_memory_aliases.clear()
        self._record_memory_invocations()

    def __enter__(self):
        super().__enter__()
        self.exit_stack.enter_context(patch_torch())
        _current_runtime.value = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            super().__exit__(exc_type, exc_val, exc_tb)
            self.repeat_op_invoke_infos()
            self.replay_op_invoke_infos()
            if self.memory_tracker:
                self.memory_tracker.analyze()
        finally:
            _current_runtime.value = None
            self.exit_stack.close()

    @classmethod
    def _bound_components(cls, result: PerformanceModel.Result) -> Dict[str, float]:
        return BoundAnalyzer.components(result).as_dict()

    @classmethod
    def _dominant_bound(cls, result: PerformanceModel.Result) -> str:
        return BoundAnalyzer.dominant(result)

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Formats time in seconds to a human-readable string (ms, us, ns)."""
        if seconds >= 1.0:
            return f"{seconds:.3f}s"
        if seconds >= 1e-3:
            return f"{seconds * 1e3:.3f}ms"
        if seconds >= 1e-6:
            return f"{seconds * 1e6:.3f}us"
        return f"{seconds * 1e9:.3f}ns"

    @staticmethod
    def _get_input_shapes_str(op_info: "OpInvokeInfo") -> str:
        """Extracts tensor shapes from operator arguments for display."""
        shapes = []
        for arg in op_info.args:
            if isinstance(arg, torch.Tensor):
                shapes.append(str(list(arg.shape)))
        return ", ".join(shapes)

    def _aggregate_average_table_data(
        self,
        first_model: Optional[str],
        group_by_input_shapes: bool,
        dump_op_bound_results: bool,
    ) -> Dict[OpAverageGroupKey, OpAverageGroupData]:
        aggregated_data: Dict[OpAverageGroupKey, OpAverageGroupData] = collections.defaultdict(OpAverageGroupData)
        for event in self.event_list:
            op_name = str(event.op_invoke_info.func)
            first_result = event.perf_results.get(first_model) if first_model else None
            key = OpAverageGroupKey(
                op_name=op_name,
                bound=self._dominant_bound(first_result) if dump_op_bound_results and first_result else "",
                input_shapes=self._get_input_shapes_str(event.op_invoke_info) if group_by_input_shapes else "",
            )

            entry = aggregated_data[key]
            entry.count += 1
            for model_name, result in event.perf_results.items():
                entry.total_runtimes[model_name] += result.execution_time_s
                if dump_op_bound_results:
                    components = self._bound_components(result)
                    # Ratios are rendered from this model-specific total, not across models.
                    for bound_name, value in components.items():
                        entry.bound_components[model_name][bound_name] += value
        return dict(aggregated_data)

    @staticmethod
    def _sort_average_table_items(
        aggregated_data: Dict[OpAverageGroupKey, OpAverageGroupData],
        first_model: Optional[str],
    ) -> List[tuple[OpAverageGroupKey, OpAverageGroupData]]:
        def sort_key(item):
            if first_model:
                return item[1].total_runtimes.get(first_model, 0)
            return 0

        return sorted(aggregated_data.items(), key=sort_key, reverse=True)

    @staticmethod
    def _average_table_headers(
        model_names: List[str],
        bound_header: str,
        group_by_input_shapes: bool,
        dump_op_bound_results: bool,
    ) -> List[str]:
        headers = ["Name"]
        if dump_op_bound_results:
            headers.append(bound_header)
        if group_by_input_shapes:
            headers.append("Input Shapes")
        for name in model_names:
            headers.extend([f"{name} total", f"{name} avg"])
            if dump_op_bound_results:
                headers.extend([f"{name} memory %", f"{name} comm %", f"{name} mma %", f"{name} gp %"])
        headers.append("# of Calls")
        return headers

    @classmethod
    def _average_table_col_widths(
        cls,
        sorted_items: List[tuple[OpAverageGroupKey, OpAverageGroupData]],
        headers: List[str],
        model_names: List[str],
        bound_header: str,
        group_by_input_shapes: bool,
        dump_op_bound_results: bool,
    ) -> Dict[str, int]:
        col_widths = {h: len(h) for h in headers}
        for key, data in sorted_items:
            col_widths["Name"] = max(col_widths["Name"], len(key.op_name))

            if dump_op_bound_results:
                col_widths[bound_header] = max(col_widths[bound_header], len(key.bound))

            if group_by_input_shapes:
                col_widths["Input Shapes"] = max(col_widths["Input Shapes"], len(key.input_shapes))
            col_widths["# of Calls"] = max(col_widths["# of Calls"], len(str(data.count)))
            for model_name in model_names:
                total_time = data.total_runtimes[model_name]
                avg_time = total_time / data.count
                col_widths[f"{model_name} total"] = max(
                    col_widths[f"{model_name} total"], len(cls._format_time(total_time))
                )
                col_widths[f"{model_name} avg"] = max(col_widths[f"{model_name} avg"], len(cls._format_time(avg_time)))
                if dump_op_bound_results:
                    for header in (
                        f"{model_name} memory %",
                        f"{model_name} comm %",
                        f"{model_name} mma %",
                        f"{model_name} gp %",
                    ):
                        col_widths[header] = max(col_widths[header], len("100.00%"))
        return col_widths

    @staticmethod
    def _format_bound_ratio(components: Dict[str, float], bound_name: str) -> str:
        component_total = sum(components.get(key, 0.0) for key in _BOUND_COMPONENT_KEYS)
        if component_total <= 0:
            return "0.00%"
        return f"{components.get(bound_name, 0.0) * 100 / component_total:.2f}%"

    @classmethod
    def _render_average_table(
        cls,
        sorted_items: List[tuple[OpAverageGroupKey, OpAverageGroupData]],
        model_names: List[str],
        headers: List[str],
        bound_header: str,
        group_by_input_shapes: bool,
        dump_op_bound_results: bool,
    ) -> str:
        col_widths = cls._average_table_col_widths(
            sorted_items,
            headers,
            model_names,
            bound_header,
            group_by_input_shapes,
            dump_op_bound_results,
        )
        output_lines = []
        header_line = "  ".join(h.center(col_widths[h]) for h in headers)
        separator_line = "  ".join("-" * col_widths[h] for h in headers)

        output_lines.append(separator_line)
        output_lines.append(header_line)
        output_lines.append(separator_line)

        for key, data in sorted_items:
            row = []
            row.append(key.op_name.ljust(col_widths["Name"]))

            if dump_op_bound_results:
                row.append(key.bound.ljust(col_widths[bound_header]))

            if group_by_input_shapes:
                row.append(key.input_shapes.ljust(col_widths["Input Shapes"]))
            for model_name in model_names:
                total_time = data.total_runtimes[model_name]
                avg_time = total_time / data.count
                row.append(cls._format_time(total_time).rjust(col_widths[f"{model_name} total"]))
                row.append(cls._format_time(avg_time).rjust(col_widths[f"{model_name} avg"]))
                if dump_op_bound_results:
                    components = data.bound_components[model_name]
                    row.append(
                        cls._format_bound_ratio(components, MEMORY_BOUND).rjust(col_widths[f"{model_name} memory %"])
                    )
                    row.append(
                        cls._format_bound_ratio(components, COMMUNICATION_BOUND).rjust(
                            col_widths[f"{model_name} comm %"]
                        )
                    )
                    row.append(
                        cls._format_bound_ratio(components, COMPUTE_BOUND_MMA).rjust(col_widths[f"{model_name} mma %"])
                    )
                    row.append(
                        cls._format_bound_ratio(components, COMPUTE_BOUND_GP).rjust(col_widths[f"{model_name} gp %"])
                    )
            row.append(str(data.count).rjust(col_widths["# of Calls"]))

            output_lines.append("  ".join(row))

        output_lines.append(separator_line)

        summary_totals = collections.defaultdict(float)
        for _, data in sorted_items:
            for model_name, total_time in data.total_runtimes.items():
                summary_totals[model_name] += total_time

        for model_name in model_names:
            total_str = cls._format_time(summary_totals[model_name])
            output_lines.append(f"Total time for {model_name}: {total_str}")

        return "\n".join(output_lines)

    def table_averages(self, group_by_input_shapes=False, dump_op_bound_results=False) -> str:
        """
        Dump pretty-print table, grouped by ops by default.

        Args:
            group_by_input_shapes: group the events by input shapes when turned on.
            dump_op_bound_results: dump memory/communication/MMA/GP time ratios for each grouped row.
        """
        if not self.event_list:
            return "No events recorded."

        model_names = [model.name for model in self.perf_models]
        first_model = model_names[0] if model_names else None
        aggregated_data = self._aggregate_average_table_data(
            first_model=first_model,
            group_by_input_shapes=group_by_input_shapes,
            dump_op_bound_results=dump_op_bound_results,
        )
        if not aggregated_data:
            return "No performance results to display."

        sorted_items = self._sort_average_table_items(aggregated_data, first_model)
        bound_header = f"Bound ({first_model})" if first_model else "Bound"
        headers = self._average_table_headers(
            model_names=model_names,
            bound_header=bound_header,
            group_by_input_shapes=group_by_input_shapes,
            dump_op_bound_results=dump_op_bound_results,
        )
        return self._render_average_table(
            sorted_items=sorted_items,
            model_names=model_names,
            headers=headers,
            bound_header=bound_header,
            group_by_input_shapes=group_by_input_shapes,
            dump_op_bound_results=dump_op_bound_results,
        )

    def get_trace_events(self):
        """
        Transform self.event_list to trace_events. Results from different performance models are
        arranged in different processes. Multiple streams are organized as threads in each process.
        """
        trace_events = []

        # Map performance model names to Process IDs (pid)
        perf_model_pids = {model.name: i for i, model in enumerate(self.perf_models)}
        model_timelines = self._build_model_timelines()

        # 1. Add Metadata Events to name the processes for readability in the trace viewer
        for model_name, pid in perf_model_pids.items():
            trace_events.append(
                {
                    "name": "process_name",
                    "ph": "M",  # Metadata event type
                    "pid": pid,
                    "args": {"name": f"{model_name} (PID: {pid})"},
                }
            )
            stream_ids = sorted(model_timelines[model_name]["stream_end_s"].keys())
            if not stream_ids:
                stream_ids = [0]
            for stream_id in stream_ids:
                trace_events.append(
                    {
                        "name": "thread_name",
                        "ph": "M",
                        "pid": pid,
                        "tid": stream_id,
                        "args": {"name": f"Stream {stream_id}"},
                    }
                )

        # 2. Iterate through events and create trace entries
        for event_idx, event in enumerate(self.event_list):
            op_name = str(event.op_invoke_info.func)

            # Create a trace event for each performance model's result
            for model_name, result in event.perf_results.items():
                pid = perf_model_pids[model_name]
                timeline = model_timelines[model_name]
                start_time_us = max(0, int(round(timeline["event_start_s"][event_idx] * 1e6)))
                duration_us = max(0, int(round(result.execution_time_s * 1e6)))

                trace_event = {
                    "name": op_name,
                    "cat": model_name,  # Category can be the model name
                    "ph": "X",  # 'X' denotes a "complete" event (start and end time)
                    "ts": start_time_us,
                    "dur": duration_us,
                    "pid": pid,
                    "tid": event.stream_id,
                    "args": {  # Add any extra useful info here
                        "Inputs": str(event.op_invoke_info.args) + " kwargs: " + str(event.op_invoke_info.kwargs),
                        "Output": str(event.op_invoke_info.out),
                        # Structured input tensor shapes for per-shape analysis tools.
                        # Only captures top-level Tensor args; ops taking List[Tensor]
                        # (e.g. aten.cat, grouped_matmul) will show [] here.
                        "simulation_shapes": str(
                            [list(a.shape) for a in event.op_invoke_info.args if isinstance(a, torch.Tensor)]
                        ),
                        **{name: str(value) for name, value in result.statistics.items()},
                    },
                }
                trace_events.append(trace_event)

        return trace_events

    def export_chrome_trace(self, trace_file):
        """
        Dump trace_events as the chrome trace file.
        """
        trace_events = self.get_trace_events()
        # Write the final JSON object to the specified file

        if isinstance(trace_file, str):
            f = open(trace_file, "w", encoding="utf-8")  # noqa: SIM115
            file_context = f
        else:
            f = trace_file
            file_context = contextlib.nullcontext()
        with file_context:
            # The top-level object should contain the 'traceEvents' key
            json.dump({"traceEvents": trace_events}, f)

    def get_breakdowns(self) -> Dict[str, Dict[str, float]]:
        """
        A breakdown of op categories according to the classification of each performance model in the runtime.
        The classification is decided by the performance models.

        Return:
            Dict: name of breakdown -> [category name, value for this category]
            The semantics of the values are defined by the performance models. See [NOTE: Breakdown from Op Classifier]
            for details.
            The runtime combines all the breakdowns from the classifiers of perf models.
        """
        breakdowns = {}
        for perf_model in self.perf_models:
            if classifiers := perf_model.get_classifiers():
                event_list_for_this = [
                    (event.op_invoke_info, event.perf_results[perf_model.name]) for event in self.event_list
                ]
                for classifier in classifiers:
                    breakdown = classifier.classify(event_list_for_this)
                    breakdowns[f"{perf_model.name}_{classifier.name}"] = breakdown
        return breakdowns

    def total_execution_time_s(self) -> Dict[str, float]:
        timelines = self._build_model_timelines()
        return {perf_model.name: timelines[perf_model.name]["total_time_s"] for perf_model in self.perf_models}

    def _build_model_timelines(self) -> Dict[str, Dict[str, object]]:
        timelines: Dict[str, Dict[str, object]] = {}
        for perf_model in self.perf_models:
            model_name = perf_model.name
            stream_end_s: Dict[int, float] = collections.defaultdict(float)
            token_ready_s: Dict[int, float] = {}
            event_start_s: List[float] = []

            for event in self.event_list:
                dep_ready_s = 0.0
                for token_id in event.dependency_token_ids:
                    dep_ready_s = max(dep_ready_s, token_ready_s.get(token_id, 0.0))
                start_time_s = max(stream_end_s[event.stream_id], dep_ready_s)
                duration_s = max(0.0, event.perf_results[model_name].execution_time_s)
                end_time_s = start_time_s + duration_s
                stream_end_s[event.stream_id] = end_time_s
                for token_id in event.produced_token_ids:
                    token_ready_s[token_id] = end_time_s
                event_start_s.append(start_time_s)

            timelines[model_name] = {
                "event_start_s": event_start_s,
                "stream_end_s": dict(stream_end_s),
                "total_time_s": max(stream_end_s.values(), default=0.0),
            }

        return timelines
