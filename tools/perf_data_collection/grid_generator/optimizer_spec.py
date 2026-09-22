"""Load user-provided throughput-optimizer workload specs for shape generation.

The spec file (YAML or JSON) describes one or more *actual* optimizer scenarios.
Field names mirror the ``throughput_optimizer`` CLI flags in snake_case and are
validated against the same enum, token, and candidate-resolution helpers the CLI
uses, so a spec scenario produces exactly the query demand of the equivalent
command line. Unknown, conflicting, or currently unsupported fields fail closed
with targeted errors instead of being silently ignored.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
from pathlib import Path
from typing import Any

import yaml

from tensor_cast import device_profiles  # noqa: F401 - register device profiles
from tensor_cast.core.compilation_config import COMPILATION_CONFIG_OPTIONS
from tensor_cast.core.quantization.datatypes import (
    QuantizeAttentionAction,
    QuantizeLinearAction,
)
from tensor_cast.device import DeviceProfile
from tensor_cast.model_config import WordEmbeddingTPMode
from tensor_cast.transformers.utils import AutoModelConfigLoader

from .query_workloads import WorkloadScenario

try:
    from serving_cast.service.utils import (
        build_pp_search_candidates,
        resolve_parallel_search_candidates,
        resolve_pp_layer_partitions,
        resolve_pp_sizes,
        resolve_search_sizes,
    )
except ImportError:  # pragma: no cover - direct-script execution path
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from serving_cast.service.utils import (
        build_pp_search_candidates,
        resolve_parallel_search_candidates,
        resolve_pp_layer_partitions,
        resolve_pp_sizes,
        resolve_search_sizes,
    )

try:
    from cli.spec_cli import make_token_type
except ImportError:  # pragma: no cover - direct-script execution path
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from cli.spec_cli import make_token_type


DEFAULT_MTP_ACCEPTANCE_RATES = (0.9, 0.6, 0.4, 0.2)

SPEC_FIELDS = {
    "name",
    "model",
    "device",
    "num_devices",
    "input_length",
    "output_length",
    "max_batched_tokens",
    "batch_range",
    "tp_sizes",
    "pp_sizes",
    "pp_layer_partitions",
    "ep_sizes",
    "moe_dp_sizes",
    "dcp_sizes",
    "num_mtp_tokens",
    "mtp_acceptance_rates",
    "quantize_linear_action",
    "quantize_non_expert_linear_action",
    "quantize_attention_action",
    "compile",
    "compilation_config",
    "enable_shared_expert_tp",
    "disagg",
    "ttft_limit",
    "tpot_limit",
    "reserved_memory_gb",
    "prefix_cache_hit_rate",
    "word_embedding_tp",
    "mxfp4_group_size",
}

# Fields users may copy from historical optimizer command lines or optimizer
# dumps that this entry point cannot honor. They get a targeted message instead
# of a generic unknown-field error so the spec never silently drops them.
KNOWN_UNSUPPORTED_FIELDS = {
    "dp_sizes": "There is no independent --dp-sizes; plain DP is derived as num-devices / (TP x PP). Remove 'dp_sizes'.",
    "speculative_method": "The speculative (dflash/dspark/spec-mtp) entry is not supported by workload specs yet; use 'num_mtp_tokens'.",
    "num_speculative_tokens": "The speculative (dflash/dspark/spec-mtp) entry is not supported by workload specs yet; use 'num_mtp_tokens'.",
    "acceptance_length": "The speculative (dflash/dspark/spec-mtp) entry is not supported by workload specs yet; use 'mtp_acceptance_rates'.",
    "draft_model_config_path": "Draft-model specs are not supported by workload specs yet; use 'num_mtp_tokens'.",
    "image_batch_size": "Multimodal inputs are not supported by workload specs.",
    "image_height": "Multimodal inputs are not supported by workload specs.",
    "image_width": "Multimodal inputs are not supported by workload specs.",
    "enable_optimize_prefill_decode_ratio": "PD-ratio optimization is not supported by workload specs.",
    "prefill_devices_per_instance": "PD-ratio optimization is not supported by workload specs.",
    "decode_devices_per_instance": "PD-ratio optimization is not supported by workload specs.",
    "performance_model": "'performance_model' is managed by generate_shape_grid (always 'profiling'); remove it from the spec.",
    "profiling_database_path": "'profiling_database_path' is managed by generate_shape_grid --database-path; remove it from the spec.",
    "profiling_database": "'profiling_database' is managed by generate_shape_grid --database-path; remove it from the spec.",
    "dump_original_results": "'dump_original_results' only affects optimizer output files and is not needed for shape generation.",
    "chrome_trace": "'chrome_trace' only affects optimizer output files and is not needed for shape generation.",
    "chrome_trace_file": "'chrome_trace_file' only affects optimizer output files and is not needed for shape generation.",
    "jobs": "'jobs' is managed by the internal workload runner; remove it from the spec.",
    "log_level": "'log_level' is managed by the internal workload runner; remove it from the spec.",
    "max_search_combinations": "'max_search_combinations' only controls a CLI warning and does not change query demand.",
    "concurrency_search_strategy": "'concurrency_search_strategy' does not change profiling query demand.",
    "serving_cost": "'serving_cost' does not change profiling query demand.",
}

_LINEAR_ACTIONS = {action.value for action in QuantizeLinearAction}
_ATTENTION_ACTIONS = {action.value for action in QuantizeAttentionAction}
_WORD_EMBEDDING_TP_MODES = {mode.value for mode in WordEmbeddingTPMode}
_parse_compilation_token, _ = make_token_type(
    COMPILATION_CONFIG_OPTIONS,
    "--compilation-config",
    store_canonical="snake",
)


class OptimizerSpecError(ValueError):
    """Raised for any invalid or unsupported workload spec content."""


@dataclass(frozen=True)
class OptimizerWorkloadSummary:
    """One source scenario before parallel-combination expansion."""

    name: str
    model: str
    num_devices: int
    input_length: int
    output_length: int
    requested_combinations: int
    expanded_workloads: int
    skipped_combinations: int
    skipped_duplicate_workloads: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model": self.model,
            "num_devices": self.num_devices,
            "input_length": self.input_length,
            "output_length": self.output_length,
            "requested_combinations": self.requested_combinations,
            "expanded_workloads": self.expanded_workloads,
            "skipped_combinations": self.skipped_combinations,
            "skipped_duplicate_workloads": self.skipped_duplicate_workloads,
        }


@dataclass(frozen=True)
class OptimizerSpec:
    """A validated workload spec ready for query-driven shape generation."""

    path: Path
    digest: str
    scenarios: tuple[WorkloadScenario, ...]
    workload_summaries: tuple[OptimizerWorkloadSummary, ...]

    def to_report_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "digest": self.digest,
            "workloads": [summary.to_dict() for summary in self.workload_summaries],
        }


def _fail(scenario_label: str, field: str, message: str) -> None:
    raise OptimizerSpecError(f"workload '{scenario_label}' field '{field}': {message}")


def _require_int(scenario_label: str, field: str, value: Any, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(scenario_label, field, f"must be an integer, got {value!r}")
    if value < minimum:
        _fail(scenario_label, field, f"must be >= {minimum}, got {value}")
    return value


def _optional_int(scenario_label: str, field: str, value: Any) -> int | None:
    if value is None:
        return None
    return _require_int(scenario_label, field, value)


def _optional_float(scenario_label: str, field: str, value: Any, *, minimum: float, allow_equal: bool) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(scenario_label, field, f"must be a number, got {value!r}")
    number = float(value)
    if number < minimum or (number == minimum and not allow_equal):
        bound = ">=" if allow_equal else ">"
        _fail(scenario_label, field, f"must be {bound} {minimum}, got {number}")
    return number


def _require_bool(scenario_label: str, field: str, value: Any) -> bool:
    if not isinstance(value, bool):
        _fail(scenario_label, field, f"must be a boolean, got {value!r}")
    return value


def _size_list(scenario_label: str, field: str, value: Any, num_devices: int) -> list[int]:
    """Mirror CLI size-list semantics; an empty list means the default search range."""
    if not isinstance(value, list):
        _fail(scenario_label, field, "must be a list of positive integers (empty list = default range)")
    normalized: list[int] = []
    for item in value:
        size = _require_int(scenario_label, field, item)
        if size > num_devices:
            _fail(
                scenario_label,
                field,
                f"contains value {size}, which is larger than 'num_devices' ({num_devices})",
            )
        if size not in normalized:
            normalized.append(size)
    return normalized


def _require_enum(scenario_label: str, field: str, value: Any, allowed: set[str]) -> str:
    if not isinstance(value, str) or value not in allowed:
        _fail(scenario_label, field, f"must be one of {sorted(allowed)}, got {value!r}")
    return value


def _resolve_batch_range(scenario_label: str, value: Any) -> tuple[int, int]:
    label = "batch_range"
    if isinstance(value, int) and not isinstance(value, bool):
        values = [value]
    elif isinstance(value, list):
        values = value
    else:
        _fail(scenario_label, label, "must be an integer max or a [min, max] list")
    if len(values) not in (1, 2):
        _fail(scenario_label, label, "expects [min, max] or a single [max]")
    numbers = [_require_int(scenario_label, label, item) for item in values]
    if len(numbers) == 2 and numbers[0] > numbers[1]:
        _fail(scenario_label, label, f"min must be <= max, got {numbers}")
    minimum = numbers[0] if len(numbers) == 2 else 1
    return minimum, numbers[-1]


def _bounded_prefix_cache_rate(scenario_label: str, value: Any) -> Any:
    """Enforce the CLI's ``[0, 1)`` prefix-cache-hit-rate domain."""
    if isinstance(value, (int, float)) and not isinstance(value, bool) and float(value) >= 1.0:
        _fail(scenario_label, "prefix_cache_hit_rate", f"must be in [0, 1), got {value}")
    return value


def _validate_legacy_mtp_rates(scenario_label: str, mtp_tokens: list[int], rates: tuple[float, ...]) -> None:
    maximum = min(9, len(rates) + 1)
    invalid = [value for value in mtp_tokens if value > maximum]
    if invalid:
        _fail(
            scenario_label,
            "num_mtp_tokens",
            f"candidates {invalid} must be in 0-{maximum} for mtp_acceptance_rates length {len(rates)}",
        )


def _resolve_num_hidden_layers(model_id: str) -> int:
    """Load the HF config's num_hidden_layers (pp validation needs it)."""
    try:
        config = AutoModelConfigLoader().load_config(model_id)
    except Exception as error:  # noqa: BLE001 - surface any loader failure as spec error
        raise OptimizerSpecError(
            f"unable to load HuggingFace config for model {model_id!r} to validate pp_sizes: "
            f"{type(error).__name__}: {error}"
        ) from error
    value = getattr(config, "num_hidden_layers", None)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise OptimizerSpecError(
            f"model {model_id!r} does not expose a positive num_hidden_layers; "
            "pp_sizes cannot be validated for this model"
        )
    return value


def _resolve_pp_sizes(scenario_label: str, value: Any, num_devices: int) -> tuple[int, ...]:
    """Use the optimizer's shared --pp-sizes resolution."""
    if value is None:
        return ()
    try:
        return tuple(resolve_pp_sizes(value, num_devices))
    except ValueError as error:
        _fail(scenario_label, "pp_sizes", str(error))


def _resolve_pp_layer_partitions(
    scenario_label: str,
    value: Any,
    pp_sizes: tuple[int, ...],
    num_hidden_layers: int,
) -> tuple[tuple[int, ...], ...]:
    """Use the optimizer's shared PP partition validation."""
    if value is None:
        return ()
    try:
        grouped = resolve_pp_layer_partitions(value, list(pp_sizes), num_hidden_layers)
    except ValueError as error:
        _fail(scenario_label, "pp_layer_partitions", str(error))
    return tuple(partition for partitions in grouped.values() for partition in partitions)


def _build_source_scenario(scenario_label: str, config: dict[str, Any]) -> WorkloadScenario:
    model = config.get("model")
    if not isinstance(model, str) or not model.strip():
        _fail(scenario_label, "model", "must be a non-empty HuggingFace model ID")
    device = config.get("device")
    if not isinstance(device, str) or not device:
        _fail(scenario_label, "device", f"must be a registered DeviceProfile name, got {device!r}")
    if device not in DeviceProfile.all_device_profiles:
        _fail(
            scenario_label,
            "device",
            f"{device!r} is not a registered DeviceProfile; available: {sorted(DeviceProfile.all_device_profiles)}",
        )
    num_devices = _require_int(scenario_label, "num_devices", config.get("num_devices"))
    input_length = config.get("input_length")
    if isinstance(input_length, str):
        _fail(
            scenario_label,
            "input_length",
            "must be an integer; variable-length distribution files are not supported by workload specs",
        )
    input_length = _require_int(scenario_label, "input_length", input_length)
    output_length = _require_int(scenario_label, "output_length", config.get("output_length"))

    tp_raw = config.get("tp_sizes")
    ep_raw = config.get("ep_sizes")
    moe_dp_raw = config.get("moe_dp_sizes")
    tp_sizes = None if tp_raw is None else _size_list(scenario_label, "tp_sizes", tp_raw, num_devices)
    ep_sizes = None if ep_raw is None else _size_list(scenario_label, "ep_sizes", ep_raw, num_devices)
    moe_dp_sizes = None if moe_dp_raw is None else _size_list(
        scenario_label, "moe_dp_sizes", moe_dp_raw, num_devices
    )
    dcp_raw = config.get("dcp_sizes")
    dcp_sizes = None if dcp_raw is None else _size_list(scenario_label, "dcp_sizes", dcp_raw, num_devices)
    if tp_raw is None and ep_raw is None and moe_dp_raw is None:
        # Backward-compatible CLI default: search TP with the default range.
        tp_sizes = []
    num_mtp_tokens = config.get("num_mtp_tokens")
    mtp_candidates: list[int] = []
    if num_mtp_tokens is not None:
        if not isinstance(num_mtp_tokens, list) or not num_mtp_tokens:
            _fail(scenario_label, "num_mtp_tokens", "expects at least one candidate when provided")
        normalized_tokens: list[int] = []
        for item in num_mtp_tokens:
            token = _require_int(scenario_label, "num_mtp_tokens", item, minimum=0)
            if token not in normalized_tokens:
                normalized_tokens.append(token)
        mtp_candidates = normalized_tokens
    mtp_acceptance_rates = config.get("mtp_acceptance_rates")
    if mtp_acceptance_rates is None:
        rates = DEFAULT_MTP_ACCEPTANCE_RATES
    else:
        if not isinstance(mtp_acceptance_rates, list) or not mtp_acceptance_rates:
            _fail(scenario_label, "mtp_acceptance_rates", "must be a non-empty list of numbers in (0, 1]")
        normalized_rates: list[float] = []
        for item in mtp_acceptance_rates:
            rate = _optional_float(scenario_label, "mtp_acceptance_rates", item, minimum=0.0, allow_equal=False)
            if rate > 1.0:
                _fail(scenario_label, "mtp_acceptance_rates", f"must be in (0, 1], got {rate}")
            normalized_rates.append(rate)
        rates = tuple(normalized_rates)
    _validate_legacy_mtp_rates(scenario_label, mtp_candidates or [0], rates)

    compilation_config = config.get("compilation_config")
    canonical_compilation: list[str] = []
    if compilation_config is not None:
        if not isinstance(compilation_config, list) or not compilation_config:
            _fail(scenario_label, "compilation_config", "must be a non-empty list of option tokens when provided")
        for token in compilation_config:
            if not isinstance(token, str):
                _fail(scenario_label, "compilation_config", f"tokens must be strings, got {token!r}")
            try:
                canonical = _parse_compilation_token(token)
            except argparse.ArgumentTypeError as error:
                _fail(scenario_label, "compilation_config", str(error))
            if canonical not in canonical_compilation:
                canonical_compilation.append(canonical)
    compile_requested = _require_bool(scenario_label, "compile", config.get("compile", False))

    pp_sizes = _resolve_pp_sizes(scenario_label, config.get("pp_sizes"), num_devices)
    if config.get("pp_layer_partitions") is not None and not pp_sizes:
        _fail(scenario_label, "pp_layer_partitions", "requires pp_sizes to be provided")
    pp_layer_partitions: tuple[tuple[int, ...], ...] = ()
    if pp_sizes:
        num_hidden_layers = _resolve_num_hidden_layers(model.strip())
        if any(pp > 1 for pp in pp_sizes) and 0 not in (mtp_candidates or [0]):
            _fail(
                scenario_label,
                "pp_sizes",
                "PP>1 currently requires num_mtp_tokens to include 0; "
                "MTP is unsupported with pipeline parallelism",
            )
        pp_layer_partitions = _resolve_pp_layer_partitions(
            scenario_label,
            config.get("pp_layer_partitions"),
            pp_sizes,
            num_hidden_layers,
        )

    # Mirror the optimizer's own candidate resolution so omitted spec keys keep
    # the exact semantics of omitted CLI flags (including the TP-only
    # backward-compatibility default).
    tp_candidates, ep_candidates, moe_dp_candidates, mtp_resolved = resolve_parallel_search_candidates(
        tp_sizes,
        ep_sizes,
        moe_dp_sizes,
        list(mtp_candidates),
        mtp_candidates[0] if mtp_candidates else 0,
        num_devices,
    )
    dcp_candidates = resolve_search_sizes(dcp_sizes, num_devices, 1)
    if mtp_candidates:
        mtp_resolved = list(mtp_candidates)
    if tp_candidates is None or not tp_candidates:
        # resolve_search_sizes never returns an empty list; keep a typed guard.
        raise OptimizerSpecError(f"workload '{scenario_label}': no TP candidate could be resolved")
    return WorkloadScenario(
        model_id=model.strip(),
        device=device,
        num_devices=num_devices,
        input_length=input_length,
        output_length=output_length,
        max_batched_tokens=_optional_int(scenario_label, "max_batched_tokens", config.get("max_batched_tokens")),
        tp_sizes=tuple(tp_candidates),
        ep_sizes=tuple(ep_candidates),
        moe_dp_sizes=tuple(moe_dp_candidates),
        dcp_sizes=tuple(dcp_candidates),
        mtp_tokens=tuple(mtp_resolved),
        pp_sizes=pp_sizes,
        pp_layer_partitions=pp_layer_partitions,
        sweep_name=scenario_label,
        compilation_config=tuple(canonical_compilation),
        compile=compile_requested or bool(canonical_compilation),
        quantize_linear_action=_require_enum(
            scenario_label,
            "quantize_linear_action",
            config.get("quantize_linear_action", "W8A8_DYNAMIC"),
            _LINEAR_ACTIONS,
        ),
        quantize_non_expert_linear_action=_require_enum(
            scenario_label,
            "quantize_non_expert_linear_action",
            config.get("quantize_non_expert_linear_action", "DISABLED"),
            _LINEAR_ACTIONS,
        ),
        quantize_attention_action=_require_enum(
            scenario_label,
            "quantize_attention_action",
            config.get("quantize_attention_action", "DISABLED"),
            _ATTENTION_ACTIONS,
        ),
        batch_range=_resolve_batch_range(scenario_label, config.get("batch_range", [1, 512])),
        disagg=_require_bool(scenario_label, "disagg", config.get("disagg", False)),
        ttft_limit=_optional_float(scenario_label, "ttft_limit", config.get("ttft_limit"), minimum=0.0, allow_equal=False),
        tpot_limit=_optional_float(scenario_label, "tpot_limit", config.get("tpot_limit"), minimum=0.0, allow_equal=False),
        reserved_memory_gb=_optional_float(
            scenario_label,
            "reserved_memory_gb",
            config.get("reserved_memory_gb"),
            minimum=0.0,
            allow_equal=True,
        ),
        prefix_cache_hit_rate=_optional_float(
            scenario_label,
            "prefix_cache_hit_rate",
            _bounded_prefix_cache_rate(scenario_label, config.get("prefix_cache_hit_rate", 0.0)),
            minimum=0.0,
            allow_equal=True,
        ),
        enable_shared_expert_tp=_require_bool(
            scenario_label,
            "enable_shared_expert_tp",
            config.get("enable_shared_expert_tp", False),
        ),
        word_embedding_tp=(
            _require_enum(
                scenario_label,
                "word_embedding_tp",
                config.get("word_embedding_tp"),
                _WORD_EMBEDDING_TP_MODES,
            )
            if config.get("word_embedding_tp")
            else ""
        ),
        mtp_acceptance_rates=rates,
        mxfp4_group_size=_optional_int(scenario_label, "mxfp4_group_size", config.get("mxfp4_group_size")),
    )


def _configured_size_values(config: dict[str, Any], field: str) -> list[int] | None:
    """Return a validated spec size list while preserving omitted vs empty."""
    value = config.get(field)
    return None if value is None else list(value)


def _requested_parallel_combinations(
    source_scenario: WorkloadScenario,
    config: dict[str, Any],
) -> int:
    """Count the optimizer candidate Cartesian space before legality filters."""
    if not source_scenario.pp_sizes:
        return source_scenario.parallel_combinations

    tp_values = _configured_size_values(config, "tp_sizes")
    ep_values = _configured_size_values(config, "ep_sizes")
    dcp_values = _configured_size_values(config, "dcp_sizes")
    moe_dp_values = resolve_search_sizes(
        _configured_size_values(config, "moe_dp_sizes"),
        source_scenario.num_devices,
        1,
    )
    mtp_values = source_scenario.mtp_tokens or (0,)
    total = 0
    for pp_size in source_scenario.pp_sizes:
        # Non-divisible PP values are still part of the requested space and
        # will be counted as skipped. The floor is used only to give omitted
        # or empty search axes a deterministic pre-filter cardinality.
        stage_devices = max(1, source_scenario.num_devices // pp_size)
        tp_candidates = resolve_search_sizes(tp_values, stage_devices, stage_devices)
        ep_candidates = resolve_search_sizes(ep_values, stage_devices, stage_devices)
        dcp_candidates = resolve_search_sizes(dcp_values, stage_devices, 1)
        if pp_size == 1 or not source_scenario.pp_layer_partitions:
            partition_count = 1
        else:
            partition_count = sum(
                len(partition) == pp_size for partition in source_scenario.pp_layer_partitions
            )
        total += (
            len(tp_candidates)
            * len(ep_candidates)
            * len(moe_dp_values)
            * len(dcp_candidates)
            * len(mtp_values)
            * partition_count
        )
    return total


def _expand_with_optimizer_candidates(
    source_scenario: WorkloadScenario,
    config: dict[str, Any],
) -> list[WorkloadScenario]:
    """Expand a spec scenario with the optimizer's canonical PP candidate builder."""
    num_hidden_layers = (
        _resolve_num_hidden_layers(source_scenario.model_id)
        if source_scenario.pp_sizes
        else 1
    )
    try:
        candidates = build_pp_search_candidates(
            num_devices=source_scenario.num_devices,
            # For explicit PP search, omitted TP/EP/DCP values must reach the
            # optimizer as None so their defaults are resolved per stage. The
            # source scenario has legacy globally-resolved values and cannot
            # preserve that distinction by itself.
            tp_sizes=_configured_size_values(config, "tp_sizes") if source_scenario.pp_sizes else list(source_scenario.tp_sizes),
            pp_sizes=list(source_scenario.pp_sizes) if source_scenario.pp_sizes else None,
            num_hidden_layers=num_hidden_layers,
            ep_sizes=_configured_size_values(config, "ep_sizes") if source_scenario.pp_sizes else list(source_scenario.ep_sizes),
            moe_dp_sizes=(
                _configured_size_values(config, "moe_dp_sizes")
                if source_scenario.pp_sizes
                else list(source_scenario.moe_dp_sizes)
            ),
            num_mtp_token_sizes=list(source_scenario.mtp_tokens),
            pp_layer_partitions=[list(partition) for partition in source_scenario.pp_layer_partitions] or None,
            dcp_sizes=(
                _configured_size_values(config, "dcp_sizes")
                if source_scenario.pp_sizes
                else list(source_scenario.dcp_sizes)
            ),
            # EP token-domain enforcement (issue #456) applies only when the
            # scenario runs the SP / dispatch_ffn_combine model path; moe_tp > 1
            # with EP stays a formal contract elsewhere.
            enforce_ep_domain=(
                "enable_sequence_parallel" in source_scenario.compilation_config
                or "enable_dispatch_ffn_combine" in source_scenario.compilation_config
            ),
        )
    except ValueError as error:
        raise OptimizerSpecError(f"workload '{source_scenario.sweep_name}': {error}") from error

    return [
        replace(
            source_scenario,
            tp_sizes=(candidate.tp_size,),
            pp_sizes=(candidate.pp_size,) if source_scenario.pp_sizes else (),
            ep_sizes=(candidate.ep_size,),
            moe_dp_sizes=(candidate.moe_dp_size,),
            dcp_sizes=(candidate.dcp_size,),
            mtp_tokens=(candidate.num_mtp_tokens,),
            pp_layer_partitions=((candidate.layer_partition,) if candidate.layer_partition is not None else ()),
        )
        for candidate in candidates
    ]


def _reject_unknown_fields(scenario_label: str, config: dict[str, Any]) -> None:
    for field in sorted(config):
        if field in SPEC_FIELDS:
            continue
        if field in KNOWN_UNSUPPORTED_FIELDS:
            raise OptimizerSpecError(f"workload '{scenario_label}' field '{field}': {KNOWN_UNSUPPORTED_FIELDS[field]}")
        raise OptimizerSpecError(
            f"workload '{scenario_label}' field '{field}': unknown field. "
            f"Supported fields: {sorted(SPEC_FIELDS)}. "
            "Fields from the optimizer CLI that are absent here are intentionally unsupported and fail closed."
        )


def _extract_scenarios(payload: Any) -> list[tuple[str, dict[str, Any]]]:
    if isinstance(payload, list):
        entries = payload
    elif isinstance(payload, dict) and "workloads" in payload:
        workloads = payload["workloads"]
        if not isinstance(workloads, list) or not workloads:
            raise OptimizerSpecError("'workloads' must be a non-empty list of workload objects")
        entries = workloads
        extra = set(payload) - {"workloads"}
        if extra:
            raise OptimizerSpecError(
                f"Unknown top-level spec keys: {sorted(extra)}; only 'workloads' is supported"
            )
    elif isinstance(payload, dict):
        entries = [payload]
    else:
        raise OptimizerSpecError(
            "Spec root must be a workload object, a list of workload objects, or {'workloads': [...]}"
        )

    scenarios: list[tuple[str, dict[str, Any]]] = []
    for index, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            raise OptimizerSpecError(f"workload #{index} must be an object, got {type(entry).__name__}")
        name = entry.get("name")
        if name is None:
            name = f"actual_{index}"
        if not isinstance(name, str) or not name.strip():
            raise OptimizerSpecError(f"workload #{index} field 'name': must be a non-empty string when provided")
        scenarios.append((name.strip(), entry))
    return scenarios


def load_optimizer_spec(spec_path: Path) -> OptimizerSpec:
    """Load, validate, and expand one workload spec file into scenarios."""
    spec_path = Path(spec_path)
    if not spec_path.is_file():
        raise OptimizerSpecError(f"Optimizer workload spec file does not exist: {spec_path}")
    raw_bytes = spec_path.read_bytes()
    digest = hashlib.sha256(raw_bytes).hexdigest()
    try:
        payload = yaml.safe_load(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise OptimizerSpecError(f"Failed to parse workload spec {spec_path} as YAML/JSON: {error}") from error

    entries = _extract_scenarios(payload)
    scenarios: list[WorkloadScenario] = []
    summaries: list[OptimizerWorkloadSummary] = []
    for scenario_label, entry in entries:
        _reject_unknown_fields(scenario_label, entry)
        source_scenario = _build_source_scenario(scenario_label, entry)
        expanded = _expand_with_optimizer_candidates(source_scenario, entry)
        requested = _requested_parallel_combinations(source_scenario, entry)
        skipped = requested - len(expanded)
        if not expanded:
            raise OptimizerSpecError(
                f"workload '{scenario_label}': no valid parallel combination exists under the requested settings; "
                "verify pp against num_hidden_layers and tp/ep/moe_dp/dcp divisibility under num_devices"
            )
        duplicate_workloads = 0
        appended_workloads = 0
        for scenario in expanded:
            workload_id = scenario.workload_id
            if any(existing.workload_id == workload_id for existing in scenarios):
                duplicate_workloads += 1
                continue
            scenarios.append(scenario)
            appended_workloads += 1
        summaries.append(
            OptimizerWorkloadSummary(
                name=scenario_label,
                model=source_scenario.model_id,
                num_devices=source_scenario.num_devices,
                input_length=source_scenario.input_length,
                output_length=source_scenario.output_length,
                requested_combinations=requested,
                expanded_workloads=appended_workloads,
                skipped_combinations=skipped,
                skipped_duplicate_workloads=duplicate_workloads,
            )
        )
    return OptimizerSpec(
        path=spec_path,
        digest=digest,
        scenarios=tuple(scenarios),
        workload_summaries=tuple(summaries),
    )
