# Copyright (c) 2026-2026 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""User-facing diagnostics run profile (only required run inputs)."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from tools.model_diagnostics.domain import (
    DiagnosticsRequest,
    ExecutionPhase,
    ModelDiagnosticsSpec,
    ModelRunContext,
    ParallelContext,
)
from tools.model_diagnostics.errors import InvalidDiagnosticsRequest
from tools.model_diagnostics.specification.errors import SpecificationLoadError

_ALLOWED_KEYS = frozenset(
    {
        "schema_version",
        "model_name",
        "entrypoint",
        "phase",
        "batch_size",
        "query_length",
        "context_length",
        "num_mtp_tokens",
        "parallel",
        "selected_language_layers",
        "selected_stage_regions",
        "num_hidden_layers_override",
        "do_compile",
        "device",
        "quantize_linear_action",
        "word_embedding_tp",
        "enable_redundant_experts",
        "enable_external_shared_experts",
    }
)
_FORBIDDEN_KEYS = frozenset(
    {
        "model_config",
        "quantization_config",
        "capture",
    }
)
_QUANTIZE_LINEAR_ACTIONS = frozenset(
    {
        "DISABLED",
        "W8A16_STATIC",
        "W8A8_STATIC",
        "W4A8_STATIC",
        "W8A16_DYNAMIC",
        "W8A8_DYNAMIC",
        "W4A8_DYNAMIC",
        "FP8",
        "MXFP4",
    }
)


class DiagnosticsSelectionWarning(UserWarning):
    """A requested validation layer is unavailable and will be skipped."""


@dataclass(frozen=True)
class DiagnosticsRunProfile:
    """One user-editable YAML profile for end-to-end diagnostics."""

    schema_version: str
    model_name: str
    entrypoint: str
    phase: ExecutionPhase
    batch_size: int
    query_length: int
    context_length: int | None
    num_mtp_tokens: int
    parallel: ParallelContext
    selected_stage_regions: tuple[str, ...]
    num_hidden_layers_override: int
    do_compile: bool
    device: str
    quantize_linear_action: str
    word_embedding_tp: str | None
    enable_redundant_experts: bool = False
    enable_external_shared_experts: bool = False
    selected_language_layers: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.selected_language_layers is not None:
            object.__setattr__(
                self,
                "selected_language_layers",
                _normalize_layer_indices(
                    self.selected_language_layers,
                    field_name="selected_language_layers",
                ),
            )

    def to_request(
        self,
        *,
        context: ModelRunContext,
        spec: ModelDiagnosticsSpec,
    ) -> DiagnosticsRequest:
        """Build a request from a materialized Spec and the post-capture Context.

        Region selection is derived from the Spec (never a hardcoded region id):
        ``selected_language_layers`` samples only physical language decoder layers.
        When omitted, every materialized language layer is selected. MTP proposal
        selection is independent and always uses the built-in representative-layer
        policy.
        ``num_hidden_layers_override`` is a capture/Context concern and does not
        truncate selection here; callers must materialize a layout that already
        reflects the intended layer count.
        Every non-layered region declaring region-level ``stages`` is selected
        unless the profile explicitly overrides ``selected_stage_regions``.
        """

        selected_layers = _derive_selected_layers(self, spec)
        selected_stage_regions = _derive_selected_stage_regions(self, spec)
        if not selected_layers and not selected_stage_regions:
            raise InvalidDiagnosticsRequest("num_hidden_layers_override or model layer count is required")
        return DiagnosticsRequest(
            context=context,
            selected_layers=selected_layers,
            selected_stage_regions=selected_stage_regions,
        )


def _derive_selected_layers(
    profile: DiagnosticsRunProfile,
    spec: ModelDiagnosticsSpec,
) -> dict[str, tuple[int, ...]]:
    layered_regions = {region.region_id: region for region in spec.regions if region.layer_specs}
    layers: dict[str, tuple[int, ...]] = {}
    language = layered_regions.get("language")
    if language is not None:
        language_count = len(language.layer_layout)
        requested = profile.selected_language_layers
        if requested is None:
            selected_language = tuple(range(language_count))
        else:
            selected_language = tuple(index for index in requested if index < language_count)
            unavailable = tuple(index for index in requested if index >= language_count)
            if unavailable:
                available = "none" if language_count == 0 else f"0..{language_count - 1}"
                if not selected_language:
                    raise InvalidDiagnosticsRequest(
                        "selected_language_layers has no indices within captured "
                        f"language layers; requested {requested}, available {available}"
                    )
                warnings.warn(
                    "selected_language_layers contains unavailable indices "
                    f"{unavailable}; captured language layers are {available}; "
                    "unavailable layers are skipped",
                    DiagnosticsSelectionWarning,
                    stacklevel=2,
                )
        if selected_language:
            layers["language"] = selected_language

    mtp_region = layered_regions.get("mtp")
    if mtp_region is not None:
        selected_mtp = _select_representative_mtp_layers(
            len(mtp_region.layer_layout)
        )
        if selected_mtp:
            layers["mtp"] = selected_mtp

    for region_id, region in layered_regions.items():
        if region_id in {"language", "mtp"}:
            continue
        layout_count = len(region.layer_layout)
        if layout_count <= 0:
            continue
        layers[region_id] = tuple(range(layout_count))
    return layers


def _select_representative_mtp_layers(layer_count: int) -> tuple[int, ...]:
    """Select the first MTP layer and one subsequent-layer representative."""

    if layer_count <= 0:
        return ()
    if layer_count == 1:
        return (0,)
    return (0, 1)


def _derive_selected_stage_regions(
    profile: DiagnosticsRunProfile,
    spec: ModelDiagnosticsSpec,
) -> tuple[str, ...]:
    if profile.selected_stage_regions:
        return profile.selected_stage_regions
    return tuple(region.region_id for region in spec.regions if region.stages)


def load_diagnostics_run_profile(path: str | Path) -> DiagnosticsRunProfile:
    """Load and validate one diagnostics run profile YAML."""

    source = Path(path)
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except OSError as error:
        raise SpecificationLoadError(f"cannot read run profile {source}") from error
    except yaml.YAMLError as error:
        raise SpecificationLoadError(f"invalid run profile YAML {source}: {error}") from error
    if not isinstance(raw, dict):
        raise SpecificationLoadError(f"run profile {source} must be a mapping")
    try:
        return _parse_profile(raw)
    except (InvalidDiagnosticsRequest, TypeError, ValueError) as error:
        raise SpecificationLoadError(f"invalid run profile {source}: {error}") from error


def _parse_profile(raw: Mapping[str, Any]) -> DiagnosticsRunProfile:
    unknown = sorted(set(raw) - _ALLOWED_KEYS)
    if unknown:
        raise SpecificationLoadError("unsupported run-profile field(s): " + ", ".join(unknown))
    for forbidden in _FORBIDDEN_KEYS:
        if forbidden in raw:
            raise SpecificationLoadError(f"{forbidden} is not a run-profile field")
    schema_version_raw = raw.get("schema_version", "1")
    if isinstance(schema_version_raw, bool) or not isinstance(schema_version_raw, (str, int)):
        raise SpecificationLoadError("run profile schema_version must be a string or integer")
    schema_version = str(schema_version_raw).strip()
    if schema_version != "1":
        raise SpecificationLoadError(f"unsupported run profile schema_version {schema_version!r}")
    parallel_raw = raw.get("parallel", {})
    if parallel_raw is None:
        parallel_raw = {}
    if not isinstance(parallel_raw, Mapping):
        raise SpecificationLoadError("parallel must be a mapping")
    unsupported_parallel = sorted(
        set(parallel_raw) & {"moe_tensor_parallel_size", "moe_tp_size"}
    )
    if unsupported_parallel:
        raise SpecificationLoadError(
            "parallel field(s) not supported: "
            + ", ".join(unsupported_parallel)
            + ". MoE tensor parallel is fixed at 1 by this module; sizes greater than 1 are unsupported."
        )
    stage_regions = raw.get("selected_stage_regions")
    if stage_regions is None:
        selected_stage_regions: tuple[str, ...] = ()
    else:
        if not isinstance(stage_regions, (list, tuple)):
            raise SpecificationLoadError("selected_stage_regions must be a sequence of non-empty strings")
        if any(not isinstance(region, str) or not region.strip() for region in stage_regions):
            raise SpecificationLoadError("selected_stage_regions must contain non-empty strings")
        selected_stage_regions = tuple(region.strip() for region in stage_regions)
    phase_value = _require_str(raw, "phase").lower()
    try:
        phase = ExecutionPhase(phase_value)
    except ValueError as error:
        raise SpecificationLoadError(f"unsupported phase {phase_value!r}") from error
    entrypoint = _optional_non_empty_str(
        raw.get("entrypoint"),
        default="text_generate",
        field_name="entrypoint",
    )
    override = raw.get("num_hidden_layers_override", 0)
    if isinstance(override, bool) or not isinstance(override, int) or override < 0:
        raise SpecificationLoadError("num_hidden_layers_override must be a non-negative integer")
    return DiagnosticsRunProfile(
        schema_version=schema_version,
        model_name=_require_str(raw, "model_name"),
        entrypoint=entrypoint,
        phase=phase,
        batch_size=_require_positive_int(raw, "batch_size"),
        query_length=_require_positive_int(raw, "query_length"),
        context_length=_optional_non_negative_int(raw.get("context_length"), field_name="context_length"),
        num_mtp_tokens=(
            _optional_non_negative_int(raw.get("num_mtp_tokens"), field_name="num_mtp_tokens") or 0
        ),
        parallel=ParallelContext(
            tensor_parallel_size=_positive_int_value(
                parallel_raw.get("tensor_parallel_size", 1), "parallel.tensor_parallel_size"
            ),
            pipeline_parallel_size=_positive_int_value(
                parallel_raw.get("pipeline_parallel_size", 1), "parallel.pipeline_parallel_size"
            ),
            data_parallel_size=_positive_int_value(
                parallel_raw.get("data_parallel_size", 1), "parallel.data_parallel_size"
            ),
            expert_parallel_size=_positive_int_value(
                parallel_raw.get("expert_parallel_size", 1), "parallel.expert_parallel_size"
            ),
            moe_data_parallel_size=_positive_int_value(
                parallel_raw.get("moe_data_parallel_size", parallel_raw.get("moe_dp_size", 1)),
                "parallel.moe_data_parallel_size",
            ),
        ),
        selected_language_layers=_optional_layer_indices(
            raw.get("selected_language_layers"),
            field_name="selected_language_layers",
        ),
        selected_stage_regions=selected_stage_regions,
        num_hidden_layers_override=override,
        do_compile=_optional_bool(raw.get("do_compile"), default=True, field_name="do_compile"),
        device=_optional_non_empty_str(raw.get("device"), default="TEST_DEVICE", field_name="device"),
        quantize_linear_action=_choice_value(
            raw.get("quantize_linear_action", "DISABLED"),
            choices=_QUANTIZE_LINEAR_ACTIONS,
            field_name="quantize_linear_action",
        ),
        word_embedding_tp=_optional_choice(
            raw.get("word_embedding_tp"),
            choices=frozenset({"col", "row"}),
            field_name="word_embedding_tp",
        ),
        enable_redundant_experts=_optional_bool(
            raw.get("enable_redundant_experts"),
            default=False,
            field_name="enable_redundant_experts",
        ),
        enable_external_shared_experts=_optional_bool(
            raw.get("enable_external_shared_experts"),
            default=False,
            field_name="enable_external_shared_experts",
        ),
    )


def _require_str(raw: Mapping[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise SpecificationLoadError(f"{key} must be a non-empty string")
    return value.strip()


def _require_positive_int(raw: Mapping[str, Any], key: str) -> int:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SpecificationLoadError(f"{key} must be a positive integer")
    return value


def _positive_int_value(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SpecificationLoadError(f"{field_name} must be a positive integer")
    return value


def _optional_non_negative_int(value: object, *, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SpecificationLoadError(f"{field_name} must be a non-negative integer")
    return value


def _optional_non_empty_str(value: object, *, default: str, field_name: str) -> str:
    if value is None:
        return default
    if not isinstance(value, str) or not value.strip():
        raise SpecificationLoadError(f"{field_name} must be a non-empty string")
    return value.strip()


def _optional_bool(value: object, *, default: bool, field_name: str) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise SpecificationLoadError(f"{field_name} must be a boolean")
    return value


def _optional_layer_indices(
    value: object,
    *,
    field_name: str,
) -> tuple[int, ...] | None:
    if value is None:
        return None
    return _normalize_layer_indices(value, field_name=field_name)


def _normalize_layer_indices(
    indices: object,
    *,
    field_name: str,
) -> tuple[int, ...]:
    if not isinstance(indices, (list, tuple)) or not indices:
        raise SpecificationLoadError(f"{field_name} must be a non-empty sequence of integers")
    if any(isinstance(index, bool) or not isinstance(index, int) for index in indices):
        raise SpecificationLoadError(f"{field_name} indices must be integers")
    if any(index < 0 for index in indices):
        raise SpecificationLoadError(f"{field_name} indices must be non-negative")
    return tuple(sorted(set(indices)))


def _optional_choice(
    value: object,
    *,
    choices: frozenset[str],
    field_name: str,
) -> str | None:
    if value is None or value == "":
        return None
    if not isinstance(value, str) or value not in choices:
        rendered = ", ".join(repr(choice) for choice in sorted(choices))
        raise SpecificationLoadError(f"{field_name} must be one of {{{rendered}}} or null")
    return value


def _choice_value(value: object, *, choices: frozenset[str], field_name: str) -> str:
    if not isinstance(value, str) or value not in choices:
        rendered = ", ".join(repr(choice) for choice in sorted(choices))
        raise SpecificationLoadError(f"{field_name} must be one of {{{rendered}}}")
    return value
