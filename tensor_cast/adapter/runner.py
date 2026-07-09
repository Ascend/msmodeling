import copy
import dataclasses
from typing import Any, Dict

from tensor_cast.core.input_generator import generate_inputs
from tensor_cast.core.model_runner import ModelRunner
from tensor_cast.performance_model.metrics_collector import MetricsCollector

from .actual import ActualSummary, build_actual_summary_from_events
from .evidence import EvidenceCase


@dataclasses.dataclass(frozen=True)
class ActualRunResult:
    metrics: Any
    summary: ActualSummary


def _collect_empirical_coverage(perf_models) -> Dict[str, Any]:
    coverage = {}
    for perf_model in perf_models:
        base_model = getattr(perf_model, "_base_model", perf_model)
        op_records = getattr(base_model, "op_records", None)
        if op_records is None:
            continue
        collector = MetricsCollector()
        collector.collect_from_records(op_records)
        coverage[base_model.name] = collector.export_hit_miss_report()
    return coverage


def run_actual_case(evidence_case: EvidenceCase, user_input: Any) -> ActualRunResult:
    case_input = copy.copy(user_input)
    for key, value in evidence_case.input.items():
        if key == "performance_model" and isinstance(value, str):
            value = [value]
        if hasattr(case_input, key):
            setattr(case_input, key, value)
    runner = ModelRunner(case_input)
    runtime_events = []
    coverage: Dict[str, Any] = {}
    observed_runtime = False
    perf_model_name = None
    total_forward_time_s = None

    def collect_summary(runtime):
        nonlocal observed_runtime, perf_model_name, total_forward_time_s
        observed_runtime = True
        runtime_events.extend(runtime.event_list)
        if perf_model_name is None and runtime.perf_models:
            perf_model_name = runtime.perf_models[0].name
        if perf_model_name is not None:
            runtime_total = runtime.total_execution_time_s().get(perf_model_name)
            if runtime_total is not None:
                total_forward_time_s = (0.0 if total_forward_time_s is None else total_forward_time_s) + runtime_total
        coverage.update(_collect_empirical_coverage(runtime.perf_models))

    metrics = runner.run_inference(generate_inputs_func=generate_inputs, runtime_observer=collect_summary)
    if not observed_runtime:
        raise RuntimeError("ModelRunner did not provide runtime events for actual summary collection.")
    summary = build_actual_summary_from_events(
        runtime_events,
        case_name=evidence_case.name,
        perf_model_name=perf_model_name,
        total_forward_time_s=total_forward_time_s,
        coverage=coverage,
    )
    return ActualRunResult(metrics=metrics, summary=summary)
