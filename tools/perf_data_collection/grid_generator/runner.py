"""Orchestration for query-driven shape-grid generation."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Collection, Iterable

from tensor_cast.performance_model.profiling_database.query_demand import (
    KernelQueryDemand,
    load_query_demand_traces,
)

try:
    from ..signature_utils import normalize_op_name
except ImportError:
    from signature_utils import normalize_op_name

from .optimizer_spec import OptimizerSpec, load_optimizer_spec
from .preflight import preflight_generated_rows
from .query_coverage import build_query_generated_rows
from .query_model import resolve_query_model_architecture
from .query_workloads import (
    CommandRunner,
    QueryWorkloadRunResult,
    QUERY_WORKLOAD_POLICY_VERSION,
    WorkloadScenario,
    _load_database_identity,
    build_workload_scenarios,
    run_query_workloads,
)
from .config import load_op_mapping_metadata, load_shape_grid_config
from .theory_fallback import build_theory_fallback_rows, theory_generation_is_skipped
from .utils import load_csv_template_rows, replace_csv_with_generated_rows


QUERY_REPORT_FILE_NAME = "shape-generation-report.json"
REPORT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class OperatorGenerationResult:
    csv_path: Path
    appended_rows: int
    demand_count: int
    projected_exact: int
    rejected: int
    duplicates: int
    reason: str = "generated"
    exact_existing: int = 0
    exact_generated: int = 0
    exact_budget_truncated: int = 0
    coverage_appended: int = 0
    fallback_appended: int = 0
    preflight_rejected: int = 0
    preflight_failures: tuple[str, ...] = ()
    demand_statuses: tuple[str, ...] = ()
    csv_rows_before: int = 0

    def to_report_dict(self) -> dict[str, Any]:
        return {
            "kernel_type": self.csv_path.stem,
            "csv": str(self.csv_path),
            "demands": self.demand_count,
            "projected_exact": self.projected_exact,
            "exact_existing": self.exact_existing,
            "exact_generated": self.exact_generated,
            "exact_budget_truncated": self.exact_budget_truncated,
            "exact_rejected": self.rejected,
            "coverage_appended": self.coverage_appended,
            "fallback_appended": self.fallback_appended,
            "duplicates": self.duplicates,
            "preflight_rejected": self.preflight_rejected,
            "preflight_failures": list(self.preflight_failures),
            "appended_rows": self.appended_rows,
            "csv_rows_before": self.csv_rows_before,
            "csv_rows_after": self.csv_rows_before + self.appended_rows,
            "pending_microbench_rows": self.appended_rows,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class QueryGenerationResult:
    workloads: QueryWorkloadRunResult
    operators: tuple[OperatorGenerationResult, ...]
    captured_demands: int
    report: dict[str, Any] = field(default_factory=dict)
    report_path: Path | None = None

    @property
    def total_appended_rows(self) -> int:
        return sum(item.appended_rows for item in self.operators)

    @property
    def generated_files(self) -> tuple[Path, ...]:
        return tuple(item.csv_path for item in self.operators if item.appended_rows)

    @property
    def skipped_files(self) -> tuple[Path, ...]:
        return tuple(item.csv_path for item in self.operators if not item.appended_rows)


@dataclass(frozen=True)
class _OperatorWritePlan:
    csv_path: Path
    headers: list[str]
    source_rows: list[dict[str, str]]
    generated_rows: list[dict[str, str]]


def discover_replay_supported_ops(op_replay_dir: Path) -> tuple[str, ...]:
    """Discover the support boundary from the real replay entry points."""
    return tuple(
        sorted(
            normalize_op_name(path.name)
            for path in op_replay_dir.glob("*_run.py")
            if path.name != "run_all_op.py"
        )
    )


def iter_csv_files(data_dir: Path) -> Iterable[Path]:
    return sorted(path for path in data_dir.rglob("*.csv") if f".tmp{path.suffix}" not in path.name)


def load_csv_files(data_dir: Path) -> list[Path]:
    if not data_dir.is_dir():
        raise ValueError(f"Database directory does not exist: {data_dir}")
    csv_files = list(iter_csv_files(data_dir))
    if not csv_files:
        raise ValueError(f"No CSV files found under database directory: {data_dir}")
    return csv_files


def _update_digest_from_file(digest, path: Path, *, label: str) -> None:
    digest.update(label.encode("utf-8"))
    with path.open("rb") as input_file:
        while chunk := input_file.read(1024 * 1024):
            digest.update(chunk)


def _update_digest_from_stat(digest, path: Path, *, label: str) -> None:
    """Update *digest* using O(1) stat fingerprint (size + mtime_ns).

    Avoids reading full file contents for large databases (35+ CSVs) on every
    run. Content changes still trigger cache invalidation via mtime_ns/size.
    """
    digest.update(label.encode("utf-8"))
    stat = path.stat()
    digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode("utf-8"))


def _query_cache_directory(
    data_dir: Path,
    model_ids: list[str],
    repo_root: Path,
    extra_digest: str | None = None,
) -> Path:
    """Return an automatic cache isolated by model, database, and query semantics."""
    digest = hashlib.sha256()
    digest.update(QUERY_WORKLOAD_POLICY_VERSION.encode("utf-8"))
    digest.update("\0".join(sorted(model_ids)).encode("utf-8"))
    if extra_digest:
        digest.update(f"\0spec:{extra_digest}".encode("utf-8"))
    for path in [data_dir / "op_mapping.yaml", *iter_csv_files(data_dir)]:
        _update_digest_from_file(digest, path, label=path.relative_to(data_dir).as_posix())
    semantic_sources = (
        "tools/perf_data_collection/grid_generator/query_workloads.py",
        "tools/perf_data_collection/grid_generator/query_model.py",
        "tensor_cast/performance_model/profiling_database/query_demand.py",
        "tensor_cast/performance_model/profiling_database/backend_projector.py",
        "tensor_cast/performance_model/profiling_database/profiling_data_source.py",
        "tensor_cast/performance_model/profiling_database/interpolating_data_source.py",
    )
    for relative_path in semantic_sources:
        source_path = repo_root / relative_path
        if source_path.is_file():
            _update_digest_from_file(digest, source_path, label=relative_path)
    return Path(tempfile.gettempdir()) / "msmodeling-shape-query-cache" / digest.hexdigest()[:32]


def _load_completed_query_demands(
    workload_result: QueryWorkloadRunResult,
    trace_dir: Path,
    *,
    preferred_workload_ids: Collection[str] = (),
) -> list[KernelQueryDemand]:
    demands_by_signature: dict[str, KernelQueryDemand] = {}
    completed_dirs = workload_result.trace_directories or (trace_dir,)
    for completed_dir in completed_dirs:
        for demand in load_query_demand_traces(completed_dir):
            existing = demands_by_signature.get(demand.signature)
            if existing is None or (
                demand.workload_id in preferred_workload_ids
                and existing.workload_id not in preferred_workload_ids
            ):
                demands_by_signature[demand.signature] = demand
    return list(demands_by_signature.values())


def normalize_target_models(raw_models: str | Iterable[str]) -> list[str]:
    values = [raw_models] if isinstance(raw_models, str) else list(raw_models)
    models = [part.strip() for value in values for part in str(value).split(",") if part.strip()]
    if not models:
        raise ValueError("--target-models requires at least one HuggingFace model ID")
    return list(dict.fromkeys(models))


def normalize_selected_ops(raw_ops: Iterable[str] | None, supported_ops: Iterable[str]) -> set[str] | None:
    if raw_ops is None:
        return None
    supported = set(supported_ops)
    selected = {normalize_op_name(value) for value in raw_ops}
    unknown = sorted(selected - supported)
    if unknown:
        raise ValueError(
            "--ops contains operators without an op_replay entry point: "
            f"{', '.join(unknown)}. Supported operators: {', '.join(sorted(supported))}"
        )
    return selected


def _assemble_scenarios(
    *,
    spec: OptimizerSpec | None,
    model_ids: list[str],
    database_path: Path,
) -> list[WorkloadScenario]:
    """Merge optimizer-args scenarios with internal sampling scenarios."""
    scenarios: list[WorkloadScenario] = []
    if spec is not None:
        scenarios.extend(spec.scenarios)
    for model_id in model_ids:
        config = resolve_query_model_architecture(model_id)
        scenarios.extend(build_workload_scenarios(model_id, config, database_path))
    unique = {scenario.workload_id: scenario for scenario in scenarios}
    return [unique[key] for key in sorted(unique)]


def _collect_query_demands(
    *,
    scenarios: list[WorkloadScenario],
    database_path: Path,
    trace_dir: Path,
    repo_root: Path,
    command_runner: CommandRunner,
    allow_empty: bool = False,
    preferred_workload_ids: Collection[str] = (),
) -> tuple[list[KernelQueryDemand], QueryWorkloadRunResult]:
    workload_result = run_query_workloads(
        scenarios,
        database_path=database_path,
        trace_dir=trace_dir,
        repo_root=repo_root,
        command_runner=command_runner,
    )
    demands = _load_completed_query_demands(
        workload_result,
        trace_dir,
        preferred_workload_ids=preferred_workload_ids,
    )
    if workload_result.succeeded == 0:
        raise RuntimeError(
            "All internal throughput_optimizer workloads failed; no trustworthy shape demand was produced. "
            "Inspect the [QUERY] failure output above."
        )
    if not demands and not allow_empty:
        raise RuntimeError(
            "Optimizer workloads completed but captured no profiling-database query. "
            "Verify that the profiling performance model is active for this database."
        )
    return demands, workload_result


def run_query_mode(
    args: argparse.Namespace,
    *,
    data_dir: Path,
    op_replay_dir: Path,
    repo_root: Path,
    command_runner: CommandRunner = subprocess.run,
) -> QueryGenerationResult:
    """Capture optimizer queries, plan local coverage, and append replay rows."""
    if args.rows <= 0:
        raise ValueError("--rows must be greater than 0")
    mapping_path = data_dir / "op_mapping.yaml"
    if not mapping_path.is_file():
        raise ValueError(f"op_mapping.yaml does not exist under database path: {data_dir}")
    spec_path = getattr(args, "optimizer_args_file", None)
    spec = load_optimizer_spec(spec_path) if spec_path else None
    raw_models = getattr(args, "target_models", None)
    model_ids = normalize_target_models(raw_models) if raw_models else []
    if spec is None and not model_ids:
        raise ValueError(
            "Either --target-models or --optimizer-args-file must provide the workload source"
        )
    supported_ops = discover_replay_supported_ops(op_replay_dir)
    selected_ops = normalize_selected_ops(args.ops, supported_ops)
    csv_files = load_csv_files(data_dir)
    csv_by_op: dict[str, Path] = {}
    duplicate_stems: set[str] = set()
    for path in csv_files:
        if path.stem in csv_by_op:
            duplicate_stems.add(path.stem)
        csv_by_op[path.stem] = path
    if duplicate_stems:
        raise ValueError(
            "Database path contains duplicate operator CSV names; pass one versioned database directory: "
            + ", ".join(sorted(duplicate_stems))
        )
    available_ops = set(csv_by_op) & set(supported_ops)
    if selected_ops is not None:
        missing_csv = sorted(selected_ops - set(csv_by_op))
        if missing_csv:
            raise ValueError(
                "Selected replay-supported operators have no CSV in the target database: "
                + ", ".join(missing_csv)
            )
    elif not available_ops:
        raise ValueError("Target database contains no CSV with a matching op_replay entry point")

    database_device, _mapping = _load_database_identity(data_dir)
    if spec is not None:
        mismatched_devices = sorted({scenario.device for scenario in spec.scenarios} - {database_device})
        if mismatched_devices:
            raise ValueError(
                "Workload spec targets device(s) that do not match the database device "
                f"{database_device!r}: {mismatched_devices}. "
                "Point --database-path at the database for the spec device, or fix the spec 'device' fields."
            )

    scenarios = _assemble_scenarios(spec=spec, model_ids=model_ids, database_path=data_dir)
    spec_model_ids = sorted({scenario.model_id for scenario in scenarios})
    exact_unlimited_workload_ids = frozenset(
        scenario.workload_id for scenario in spec.scenarios
    ) if spec is not None else frozenset()
    print(
        "Query-driven shape generation: "
        f"models={spec_model_ids}, "
        f"optimizer_args_workloads={len(spec.scenarios) if spec else 0}, "
        f"requested_ops={sorted(selected_ops) if selected_ops else 'model queries'}, "
        f"rows/csv={args.rows}, seed={args.seed}, "
        f"exact_unlimited_workloads={len(exact_unlimited_workload_ids)}"
    )
    spec_digest = spec.digest if spec is not None else None
    trace_dir = _query_cache_directory(data_dir, spec_model_ids, repo_root, extra_digest=spec_digest)
    print(f"[QUERY] automatic checkpoint cache: {trace_dir}")
    demands, workload_result = _collect_query_demands(
        scenarios=scenarios,
        database_path=data_dir,
        trace_dir=trace_dir,
        repo_root=repo_root,
        command_runner=command_runner,
        allow_empty=selected_ops is not None,
        preferred_workload_ids=exact_unlimited_workload_ids,
    )

    demands_by_kernel: dict[str, list[KernelQueryDemand]] = {}
    unsupported_kernel_demands: Counter[str] = Counter()
    for demand in demands:
        if demand.kernel_type in available_ops:
            demands_by_kernel.setdefault(demand.kernel_type, []).append(demand)
        else:
            unsupported_kernel_demands[demand.kernel_type] += 1

    target_ops = selected_ops if selected_ops is not None else set(demands_by_kernel)
    if not target_ops:
        raise ValueError(
            "Target model workloads queried no operator with both a database CSV and op_replay entry point"
        )

    operator_results: list[OperatorGenerationResult] = []
    write_plans: list[_OperatorWritePlan] = []
    theory_config: dict | None = None
    op_meta: dict[str, dict] | None = None
    for kernel_type in sorted(target_ops):
        csv_path = csv_by_op[kernel_type]
        kernel_demands = demands_by_kernel.get(kernel_type, [])
        loaded = load_csv_template_rows(csv_path, require_rows=True)
        if loaded is None:
            raise ValueError(f"{csv_path} is missing a usable Shape schema")
        headers, source_rows = loaded
        generated_rows: list[dict[str, str]] = []
        summary: dict[str, Any] = {}
        if kernel_demands:
            generated_rows, summary = build_query_generated_rows(
                csv_path=csv_path,
                headers=headers,
                source_rows=source_rows,
                demands=kernel_demands,
                row_limit=args.rows,
                seed=args.seed,
                exact_unlimited_workload_ids=exact_unlimited_workload_ids,
            )
            result = OperatorGenerationResult(
                csv_path=csv_path,
                appended_rows=len(generated_rows),
                demand_count=summary["demands"],
                projected_exact=summary["projected_exact"],
                rejected=summary["rejected"],
                duplicates=summary["duplicates"],
                reason="generated" if generated_rows else "no_new_candidate",
                exact_existing=summary["exact_existing"],
                exact_generated=summary["exact_generated"],
                exact_budget_truncated=summary["exact_budget_truncated"],
                coverage_appended=summary["coverage_appended"],
                fallback_appended=summary["fallback_appended"],
                demand_statuses=summary["demand_statuses"],
                csv_rows_before=len(source_rows),
            )
        else:
            if theory_config is None:
                theory_config = load_shape_grid_config(Path(__file__).with_name("config.yaml"))
                op_meta = load_op_mapping_metadata(data_dir)
            assert op_meta is not None
            if theory_generation_is_skipped(kernel_type, theory_config, op_meta):
                operator_results.append(
                    OperatorGenerationResult(
                        csv_path,
                        0,
                        0,
                        0,
                        0,
                        0,
                        reason="theory_skipped",
                    )
                )
                print(f"[GRID] {kernel_type}: skipped by generic Shape policy")
                continue
            fallback = build_theory_fallback_rows(
                kernel_type=kernel_type,
                model_names=spec_model_ids,
                config=theory_config,
                op_meta=op_meta,
                csv_path=csv_path,
                headers=headers,
                source_rows=source_rows,
                row_limit=args.rows,
            )
            if fallback is None:
                operator_results.append(
                    OperatorGenerationResult(
                        csv_path,
                        0,
                        0,
                        0,
                        0,
                        0,
                        reason="theory_skipped",
                    )
                )
                print(f"[GRID] {kernel_type}: skipped because no generic Shape generator exists")
                continue
            generated_rows, summary = fallback
            result = OperatorGenerationResult(
                csv_path=csv_path,
                appended_rows=len(generated_rows),
                demand_count=0,
                projected_exact=0,
                rejected=0,
                duplicates=summary["duplicates"],
                reason="generated" if generated_rows else "no_new_candidate",
                fallback_appended=summary.get("fallback_appended", 0),
                csv_rows_before=len(source_rows),
            )
        # Replay preflight: run every candidate row through the kernel's real
        # build_case contract in pure Python and drop rows that would be
        # rejected (and silently deleted) by start_microbench.
        preflight_results = preflight_generated_rows(kernel_type, generated_rows, op_replay_dir)
        preflight_failures = tuple(
            f"row {item.row_index} ({generated_rows[item.row_index].get('Input Shapes', '')}): {item.reason}"
            for item in preflight_results
            if not item.passed
        )
        if preflight_failures:
            generated_rows = [
                row for row, item in zip(generated_rows, preflight_results) if item.passed
            ]
        failed_row_indexes = {
            item.row_index for item in preflight_results if not item.passed
        }
        exact_end = result.exact_generated
        coverage_end = exact_end + result.coverage_appended
        exact_preflight_rejected = sum(index < exact_end for index in failed_row_indexes)
        coverage_preflight_rejected = sum(
            exact_end <= index < coverage_end for index in failed_row_indexes
        )
        fallback_preflight_rejected = sum(
            coverage_end <= index < coverage_end + result.fallback_appended
            for index in failed_row_indexes
        )
        demand_statuses = list(result.demand_statuses)
        generated_exact_demand_indexes = tuple(summary.get("generated_exact_demand_indexes", ()))
        for row_index in sorted(index for index in failed_row_indexes if index < exact_end):
            if row_index < len(generated_exact_demand_indexes):
                demand_index = generated_exact_demand_indexes[row_index]
                demand_statuses[demand_index] = "preflight_rejected"
        result = OperatorGenerationResult(
            csv_path=result.csv_path,
            appended_rows=len(generated_rows),
            demand_count=result.demand_count,
            projected_exact=result.projected_exact,
            rejected=result.rejected,
            duplicates=result.duplicates,
            reason=(
                result.reason
                if generated_rows
                else "preflight_rejected" if preflight_failures else "no_new_candidate"
            ),
            exact_existing=result.exact_existing,
            exact_generated=result.exact_generated - exact_preflight_rejected,
            exact_budget_truncated=result.exact_budget_truncated,
            coverage_appended=result.coverage_appended - coverage_preflight_rejected,
            fallback_appended=result.fallback_appended - fallback_preflight_rejected,
            preflight_rejected=len(preflight_failures),
            preflight_failures=preflight_failures,
            demand_statuses=tuple(demand_statuses),
            csv_rows_before=result.csv_rows_before,
        )
        operator_results.append(result)
        if generated_rows:
            write_plans.append(
                _OperatorWritePlan(
                    csv_path=csv_path,
                    headers=headers,
                    source_rows=source_rows,
                    generated_rows=generated_rows,
                )
            )
        print(
            f"[GRID] {kernel_type}: demand={result.demand_count}, exact={result.projected_exact}, "
            f"rejected={result.rejected}, duplicate={result.duplicates}, appended={result.appended_rows}",
            end="",
        )
        if result.preflight_rejected:
            print(f", preflight_rejected={result.preflight_rejected}", end="")
        fallback_appended = result.fallback_appended
        if fallback_appended:
            fallback_attempted = summary.get("fallback_attempted", 0)
            fallback_duplicates = summary.get("fallback_duplicates", 0)
            fallback_rejected_safety = summary.get("fallback_rejected_safety", 0)
            print(
                f" | constraint_fallback: attempted={fallback_attempted}, "
                f"safety_rejected={fallback_rejected_safety}, "
                f"duplicate={fallback_duplicates}, appended={fallback_appended}",
                end="",
            )
        print()

    for plan in write_plans:
        replace_csv_with_generated_rows(
            plan.csv_path,
            plan.headers,
            plan.source_rows,
            plan.generated_rows,
        )

    mode = "optimizer-args" if spec is not None else "target-models"
    if spec is not None and model_ids:
        mode = "optimizer-args+target-models"
    demand_statuses_by_kernel: dict[str, tuple[str, ...]] = {}
    for item in operator_results:
        demand_statuses_by_kernel[item.csv_path.stem] = tuple(item.demand_statuses)
    ledger_cursor: dict[str, int] = {kernel: 0 for kernel in demand_statuses_by_kernel}
    demand_ledger = []
    for demand in demands:
        kernel = demand.kernel_type
        if kernel in demand_statuses_by_kernel and ledger_cursor[kernel] < len(demand_statuses_by_kernel[kernel]):
            status = demand_statuses_by_kernel[kernel][ledger_cursor[kernel]]
            ledger_cursor[kernel] += 1
        elif kernel in unsupported_kernel_demands:
            status = "unsupported"
        else:
            status = "not_selected"
        demand_ledger.append(
            {
                "demand_id": hashlib.sha256(demand.signature.encode("ascii")).hexdigest()[:16],
                "kernel_type": kernel,
                "workload_id": demand.workload_id,
                "status": status,
            }
        )
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "mode": mode,
        "policy_version": QUERY_WORKLOAD_POLICY_VERSION,
        "optimizer_args": spec.to_report_dict() if spec is not None else None,
        "target_models": model_ids or None,
        "rows_budget": args.rows,
        "seed": args.seed,
        "rows_budget_semantics": (
            "optimizer-args exact demand rows are never budget-limited; --rows bounds coverage/fallback candidates"
            if mode == "optimizer-args"
            else (
                "optimizer-args exact demand rows are exempt; target-model exact and coverage/fallback candidates "
                "share the --rows budget"
                if mode == "optimizer-args+target-models"
                else "--rows bounds all appended rows per CSV; existing/duplicate/rejected rows are free"
            )
        ),
        "workloads": {
            "attempted": workload_result.attempted,
            "succeeded": workload_result.succeeded,
            "cached": workload_result.cached,
            "failed_workloads": list(workload_result.failed_workloads),
            "elapsed_seconds": round(workload_result.elapsed_seconds, 3),
        },
        "captured_demands": len(demands),
        "unsupported_kernel_demands": dict(sorted(unsupported_kernel_demands.items())),
        "demand_ledger": demand_ledger,
        "operators": [item.to_report_dict() for item in operator_results],
        "totals": {
            "appended_rows": sum(item.appended_rows for item in operator_results),
            "preflight_rejected": sum(item.preflight_rejected for item in operator_results),
            "updated_csvs": len({item.csv_path for item in operator_results if item.appended_rows}),
            "pending_microbench_rows": sum(item.appended_rows for item in operator_results),
        },
    }
    configured_report_path = getattr(args, "report_path", None)
    report_path = configured_report_path or trace_dir / QUERY_REPORT_FILE_NAME
    write_generation_report(report_path, report)
    return QueryGenerationResult(
        workloads=workload_result,
        operators=tuple(operator_results),
        captured_demands=len(demands),
        report=report,
        report_path=report_path,
    )


def write_generation_report(report_path: Path, report: dict[str, Any]) -> Path:
    """Persist the machine-readable generation report at its requested path."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report_path
