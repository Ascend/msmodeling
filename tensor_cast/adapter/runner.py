import copy
import dataclasses
from typing import Any, Dict

from tensor_cast.core.input_generator import generate_inputs
from tensor_cast.core.model_runner import ModelRunner
from tensor_cast.performance_model.metrics_collector import MetricsCollector

from .actual import ActualSummary, build_actual_summary_from_runtime
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
    metrics = runner.run_inference(generate_inputs_func=generate_inputs)
    runtime = getattr(metrics, "runtime", None)
    if runtime is None:
        raise RuntimeError(
            "ModelRunnerMetrics does not expose runtime events. Use build_actual_summary_from_runtime "
            "directly for now or enable runtime capture in ModelRunner."
        )
    summary = build_actual_summary_from_runtime(
        runtime,
        case_name=evidence_case.name,
        coverage=_collect_empirical_coverage(runtime.perf_models),
    )
    return ActualRunResult(metrics=metrics, summary=summary)
