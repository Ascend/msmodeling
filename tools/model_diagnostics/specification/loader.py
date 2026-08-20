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
"""Strict YAML Spec loading and typed option conversion."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Protocol

import yaml

from tools.model_diagnostics.domain.models import ModelRunContext, SourceKind
from tools.model_diagnostics.domain.specification import (
    ComparisonOptions,
    ComparisonSpec,
    LayerSpec,
    ModelDiagnosticsSpec,
    RegionSpec,
    SourceStageOptions,
    SpecMatchCriteria,
    StageSpec,
    TheoryStageOptions,
)
from tools.model_diagnostics.comparison import (
    ComparisonOptionParseError,
    StrategyResolutionError,
)
from tools.model_diagnostics.schema_utils import SchemaGuard
from tools.model_diagnostics.specification.errors import SpecificationLoadError
from tools.model_diagnostics.specification.mtp_window import (
    effective_num_mtp_layers,
    validate_mtp_decode_window,
)
from tools.model_diagnostics.specification.operator_activation import (
    OperatorActivationRegistry,
    OperatorActivationRequest,
)
from tools.model_diagnostics.specification.source_options import SourceOptionsParser
from tools.model_diagnostics.specification.theory_fragments import (
    TheoryFragmentStage,
    TheoryFragmentRegistry,
    compose_mtp_layer_stages,
    load_builtin_theory_fragment_registry,
)

_DEFAULT_SPECS_DIR = Path(__file__).resolve().parents[1] / "specs"
_SCHEMA = SchemaGuard(
    error=SpecificationLoadError,
    accept_any_mapping=True,
    kind_mapping="mapping",
    kind_list="list",
    text_non_empty=True,
    require_string_keys=True,
)
_exact_keys = _SCHEMA.exact_keys
_require_mapping = _SCHEMA.mapping
_require_list = _SCHEMA.sequence
_as_str = _SCHEMA.text


class _ComparisonRegistry(Protocol):
    def parse_options(
        self,
        strategy_id: str,
        raw: Mapping[str, object],
    ) -> ComparisonOptions: ...


@dataclass(frozen=True)
class _LayerLayoutRule:
    strategy: str
    count_from: str
    layer_kind: str | None = None
    prefix_layer_kind: str | None = None
    repeated_layer_kind: str | None = None
    prefix_count_from: str | None = None


@dataclass(frozen=True)
class LoadedSpecDocument:
    """A parsed Spec paired with its Context-pending ``layer_layout_rule`` values.

    ``load``/``load_mapping`` return this instead of a bare ``ModelDiagnosticsSpec`` so
    that ``materialize`` is a pure function of its arguments: it never depends on
    loader-instance state populated by an earlier ``load`` call.
    """

    spec: ModelDiagnosticsSpec
    layout_rules: Mapping[str, _LayerLayoutRule]
    region_activations: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "layout_rules", MappingProxyType(dict(self.layout_rules)))
        object.__setattr__(
            self,
            "region_activations",
            MappingProxyType(dict(self.region_activations)),
        )


def _string_list(value: object, label: str) -> tuple[str, ...]:
    return tuple(_as_str(item, f"{label} item") for item in _require_list(value, label))


def _string_mapping(value: object, label: str) -> dict[str, str]:
    mapping = _require_mapping(value, label)
    return {_as_str(key, f"{label} key"): _as_str(mapped, f"{label} value") for key, mapped in mapping.items()}


def _schema_version(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise SpecificationLoadError("schema_version must be a string or integer")
    return _as_str(str(value), "schema_version")


def _materialize_stage(
    stage: StageSpec,
    *,
    spec: ModelDiagnosticsSpec,
    region_id: str,
    context: ModelRunContext,
    activation_registry: OperatorActivationRegistry,
) -> StageSpec | None:
    source_options = dict(stage.source_options)
    theory = source_options.get(SourceKind.THEORY)
    if isinstance(theory, TheoryStageOptions):
        operators = []
        for operator in theory.operators:
            if operator.activation is not None:
                policy = activation_registry.resolve(operator.activation)
                if not policy.is_active(
                    OperatorActivationRequest(
                        spec_id=spec.spec_id,
                        model_category=spec.model_category,
                        region_id=region_id,
                        stage_id=stage.stage_id,
                        operator=operator,
                        context=context,
                    )
                ):
                    continue
            operators.append(operator)
        if not operators:
            return None
        source_options[SourceKind.THEORY] = TheoryStageOptions(operators=tuple(operators))
    if not source_options:
        return None
    return StageSpec(
        stage_id=stage.stage_id,
        source_options=source_options,
        comparisons=stage.comparisons,
    )


def _resolve_count_from(context: ModelRunContext, count_from: str) -> int:
    parts = count_from.split(".")
    if not parts or any(not part for part in parts):
        raise SpecificationLoadError(f"invalid count_from path: {count_from!r}")
    root = parts[0]
    if root == "model_config":
        current: object = dict(context.model_config)
    elif root == "quantization_config":
        current = dict(context.quantization_config)
    else:
        raise SpecificationLoadError(f"count_from root must be model_config or quantization_config: {count_from!r}")
    for part in parts[1:]:
        if not isinstance(current, Mapping) or part not in current:
            raise SpecificationLoadError(f"count_from path not found in context: {count_from!r}")
        current = current[part]
    if isinstance(current, bool) or not isinstance(current, int) or current < 0:
        raise SpecificationLoadError(f"count_from must resolve to a non-negative int: {count_from!r}")
    return current


class YamlModelDiagnosticsSpecLoader:
    """Load builtin Spec YAML files into immutable domain values.

    ``load(spec_id)`` matches the design Protocol and does not take Context. It
    returns a ``LoadedSpecDocument`` carrying both the Spec and its pending
    ``layer_layout_rule`` values. ``materialize(loaded, context)`` is a pure
    function over that document and a ``ModelRunContext``; the loader instance
    holds no per-spec mutable state between calls.
    """

    def __init__(
        self,
        *,
        comparison_registry: _ComparisonRegistry,
        activation_registry: OperatorActivationRegistry,
        source_options_parsers: Mapping[str, SourceOptionsParser],
        specs_dir: Path | None = None,
        fragment_registry: TheoryFragmentRegistry | None = None,
    ) -> None:
        self._specs_dir = specs_dir or _DEFAULT_SPECS_DIR
        self._comparison_registry = comparison_registry
        self._activation_registry = activation_registry
        self._source_options_parsers = dict(source_options_parsers)
        self._fragment_registry = fragment_registry or load_builtin_theory_fragment_registry()
        for yaml_key, parser in self._source_options_parsers.items():
            if yaml_key != parser.yaml_key:
                raise ValueError(
                    f"source option parser key {yaml_key!r} does not match parser yaml_key {parser.yaml_key!r}"
                )
        source_kinds = tuple(parser.source_kind for parser in self._source_options_parsers.values())
        if len(source_kinds) != len(set(source_kinds)):
            raise ValueError("source option parsers must have unique source kinds")

    @property
    def fragment_registry(self) -> TheoryFragmentRegistry:
        return self._fragment_registry

    def load(self, spec_id: str) -> LoadedSpecDocument:
        path = self._resolve_spec_path(spec_id)
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise SpecificationLoadError(f"cannot read spec file for {spec_id!r}: {path}") from exc
        except UnicodeDecodeError as exc:
            raise SpecificationLoadError(f"cannot decode spec file for {spec_id!r}: {path}") from exc
        except yaml.YAMLError as exc:
            raise SpecificationLoadError(f"invalid YAML for {spec_id!r}") from exc
        return self.load_mapping(raw)

    def _resolve_spec_path(self, spec_id: str) -> Path:
        if not isinstance(spec_id, str) or not spec_id.strip():
            raise SpecificationLoadError(f"invalid spec_id {spec_id!r}")
        # Reject absolute paths and any value that is not a plain file name.
        if Path(spec_id).name != spec_id or spec_id in {".", ".."}:
            raise SpecificationLoadError(
                f"spec_id must be a file name without path separators: {spec_id!r}"
            )
        root = self._specs_dir.resolve()
        candidates = (root / f"{spec_id}.yaml", root / spec_id)
        for candidate in candidates:
            resolved = candidate.resolve()
            try:
                resolved.relative_to(root)
            except ValueError as error:
                raise SpecificationLoadError(
                    f"spec_id escapes specs directory: {spec_id!r}"
                ) from error
            if resolved.is_file():
                return resolved
        raise SpecificationLoadError(f"spec file not found for {spec_id!r}: {candidates[0]}")


    def load_mapping(self, raw: object) -> LoadedSpecDocument:
        try:
            return self._load_mapping(raw)
        except SpecificationLoadError:
            raise
        except (TypeError, ValueError) as exc:
            raise SpecificationLoadError(f"invalid model diagnostics Spec: {exc}") from exc

    def _load_mapping(self, raw: object) -> LoadedSpecDocument:
        payload = _require_mapping(raw, "spec document")
        _exact_keys(
            payload,
            required={
                "schema_version",
                "spec_id",
                "spec_version",
                "model_category",
                "matches",
                "regions",
            },
            optional={"operator_aliases"},
            label="spec document",
        )

        matches_raw = _require_mapping(payload.get("matches"), "matches")
        _exact_keys(
            matches_raw,
            required=set(),
            optional={"entrypoints", "model_types", "required_features"},
            label="matches",
        )
        matches = SpecMatchCriteria(
            entrypoints=_string_list(matches_raw.get("entrypoints", []), "matches.entrypoints"),
            model_types=_string_list(matches_raw.get("model_types", []), "matches.model_types"),
            required_features=_string_list(
                matches_raw.get("required_features", []),
                "matches.required_features",
            ),
        )
        regions_raw = _require_mapping(payload.get("regions"), "regions")
        if not regions_raw:
            raise SpecificationLoadError("regions must not be empty")
        regions: list[RegionSpec] = []
        layout_rules: dict[str, _LayerLayoutRule] = {}
        region_activations: dict[str, str] = {}
        for region_id, region_body in regions_raw.items():
            region_name = _as_str(region_id, "region id")
            region, rule, activation = self._parse_region(region_name, region_body)
            regions.append(region)
            if rule is not None:
                layout_rules[region.region_id] = rule
            if activation is not None:
                region_activations[region.region_id] = activation

        operator_aliases = (
            _string_mapping(payload["operator_aliases"], "operator_aliases") if "operator_aliases" in payload else {}
        )
        spec = ModelDiagnosticsSpec(
            schema_version=_schema_version(payload["schema_version"]),
            spec_id=_as_str(payload.get("spec_id"), "spec_id"),
            spec_version=_as_str(payload.get("spec_version"), "spec_version"),
            model_category=_as_str(payload.get("model_category"), "model_category"),
            matches=matches,
            regions=tuple(regions),
            operator_aliases=operator_aliases,
        )
        return LoadedSpecDocument(
            spec=spec,
            layout_rules=layout_rules,
            region_activations=region_activations,
        )

    def materialize(
        self,
        loaded: LoadedSpecDocument,
        context: ModelRunContext,
    ) -> ModelDiagnosticsSpec:
        """Expand retained ``layer_layout_rule`` values into domain ``layer_layout``.

        ``loaded`` must be the ``LoadedSpecDocument`` produced by this loader's
        ``load``/``load_mapping``; this method reads only its arguments (no
        instance state), so any loader instance can materialize any document.
        """

        if not isinstance(loaded, LoadedSpecDocument):
            raise TypeError(
                f"materialize requires a LoadedSpecDocument from load()/load_mapping(), got {type(loaded).__name__}"
            )
        validate_mtp_decode_window(context)
        enriched_config = dict(context.model_config)
        enriched_config["effective_num_mtp_layers"] = effective_num_mtp_layers(context)
        materialize_context = replace(context, model_config=enriched_config)
        spec = loaded.spec
        rules = loaded.layout_rules
        regions: list[RegionSpec] = []
        for region in spec.regions:
            region_activation = loaded.region_activations.get(region.region_id)
            if region_activation is not None:
                policy = self._activation_registry.resolve(region_activation)
                if not policy.is_active(
                    OperatorActivationRequest(
                        spec_id=spec.spec_id,
                        model_category=spec.model_category,
                        region_id=region.region_id,
                        stage_id="",
                        operator=None,
                        context=materialize_context,
                    )
                ):
                    regions.append(RegionSpec(region_id=region.region_id))
                    continue
            rule = rules.get(region.region_id)
            layer_layout = region.layer_layout
            if rule is not None:
                count = _resolve_count_from(materialize_context, rule.count_from)
                if rule.strategy == "repeat":
                    if rule.layer_kind is None:
                        raise SpecificationLoadError(
                            "layer_layout_rule.layer_kind is required for repeat strategy"
                        )
                    layer_layout = tuple(rule.layer_kind for _ in range(count))
                elif rule.strategy == "prefix_then_repeat":
                    missing_fields = [
                        field_name
                        for field_name, value in (
                            ("prefix_layer_kind", rule.prefix_layer_kind),
                            ("repeated_layer_kind", rule.repeated_layer_kind),
                            ("prefix_count_from", rule.prefix_count_from),
                        )
                        if value is None
                    ]
                    if missing_fields:
                        raise SpecificationLoadError(
                            "layer_layout_rule field(s) required for prefix_then_repeat strategy: "
                            + ", ".join(missing_fields)
                        )
                    prefix_count = _resolve_count_from(materialize_context, rule.prefix_count_from)
                    layer_layout = tuple(
                        rule.prefix_layer_kind if index < prefix_count else rule.repeated_layer_kind
                        for index in range(count)
                    )
                else:
                    raise SpecificationLoadError(f"unsupported layer_layout_rule.strategy: {rule.strategy}")
            materialized_stages = tuple(
                stage
                for stage in (
                    _materialize_stage(
                        stage,
                        spec=spec,
                        region_id=region.region_id,
                        context=materialize_context,
                        activation_registry=self._activation_registry,
                    )
                    for stage in region.stages
                )
                if stage is not None
            )
            materialized_layer_specs: dict[str, LayerSpec] = {}
            for layer_kind, layer_spec in region.layer_specs.items():
                layer_stages = tuple(
                    stage
                    for stage in (
                        _materialize_stage(
                            stage,
                            spec=spec,
                            region_id=region.region_id,
                            context=materialize_context,
                            activation_registry=self._activation_registry,
                        )
                        for stage in layer_spec.stages
                    )
                    if stage is not None
                )
                if not layer_stages:
                    raise SpecificationLoadError(
                        f"layer_specs.{layer_kind}.stages must retain at least one stage after activation"
                    )
                materialized_layer_specs[layer_kind] = LayerSpec(
                    layer_kind=layer_kind,
                    stages=layer_stages,
                )
            regions.append(
                RegionSpec(
                    region_id=region.region_id,
                    stages=materialized_stages,
                    layer_layout=layer_layout,
                    layer_specs=materialized_layer_specs,
                )
            )
        return ModelDiagnosticsSpec(
            schema_version=spec.schema_version,
            spec_id=spec.spec_id,
            spec_version=spec.spec_version,
            model_category=spec.model_category,
            matches=spec.matches,
            regions=tuple(regions),
            operator_aliases=spec.operator_aliases,
        )

    def _parse_region(
        self,
        region_id: str,
        body: object,
    ) -> tuple[RegionSpec, _LayerLayoutRule | None, str | None]:
        region = _require_mapping(body, f"region {region_id}")
        _exact_keys(
            region,
            required=set(),
            optional={
                "stages",
                "include_fragment",
                "compose",
                "runtime_options",
                "comparisons",
                "activation",
                "layer_layout_rule",
                "layer_specs",
            },
            label=f"region {region_id}",
        )
        if "compose" in region:
            conflicting = {"stages", "include_fragment", "layer_specs"}.intersection(region)
            if conflicting:
                raise SpecificationLoadError(
                    f"region {region_id} compose cannot be combined with "
                    f"{sorted(conflicting)[0]}"
                )
            if "layer_layout_rule" not in region:
                raise SpecificationLoadError(
                    f"region {region_id} compose requires layer_layout_rule"
                )
            return self._parse_composed_mtp_region(region_id, region)
        if "stages" in region and "include_fragment" in region:
            raise SpecificationLoadError(
                f"region {region_id} must declare either stages or include_fragment, not both"
            )
        if "include_fragment" in region:
            include = _require_mapping(
                region.get("include_fragment"),
                f"region {region_id}.include_fragment",
            )
            _exact_keys(
                include,
                required={"fragment", "stage_group"},
                label=f"region {region_id}.include_fragment",
            )
            fragment_id = _as_str(
                include.get("fragment"),
                f"region {region_id}.include_fragment.fragment",
            )
            group_id = _as_str(
                include.get("stage_group"),
                f"region {region_id}.include_fragment.stage_group",
            )
            fragment = self._fragment_registry.get(fragment_id)
            stages = tuple(
                self._fragment_stage_to_stage_spec(stage)
                for stage in fragment.stage_group(group_id)
            )
            stages = self._apply_fragment_stage_overrides(
                stages,
                runtime_raw=region.get("runtime_options"),
                comparisons_raw=region.get("comparisons"),
                label=f"region {region_id}",
            )
        else:
            if "runtime_options" in region or "comparisons" in region:
                raise SpecificationLoadError(
                    f"region {region_id} runtime_options/comparisons require include_fragment"
                )
            stages_raw = region.get("stages", [])
            stages = tuple(
                self._parse_stage(item)
                for item in _require_list(stages_raw, "region.stages")
            )

        layer_specs: dict[str, LayerSpec] = {}
        layer_specs_raw = region.get("layer_specs", {})
        for layer_kind, layer_body in _require_mapping(layer_specs_raw, "layer_specs").items():
            layer_name = _as_str(layer_kind, "layer kind")
            layer_map = _require_mapping(layer_body, f"layer_specs.{layer_name}")
            _exact_keys(
                layer_map,
                required=set(),
                optional={
                    "stages",
                    "include_fragment",
                    "compose",
                    "runtime_options",
                    "comparisons",
                },
                label=f"layer_specs.{layer_name}",
            )
            declarations = {
                key
                for key in ("stages", "include_fragment", "compose")
                if key in layer_map
            }
            if len(declarations) != 1:
                raise SpecificationLoadError(
                    f"layer_specs.{layer_name} must declare exactly one of "
                    "stages, include_fragment, or compose"
                )
            if "compose" in layer_map:
                layer_stages = self._parse_composed_mtp_layer(
                    layer_map.get("compose"),
                    label=f"layer_specs.{layer_name}.compose",
                )
                layer_stages = self._apply_fragment_stage_overrides(
                    layer_stages,
                    runtime_raw=layer_map.get("runtime_options"),
                    comparisons_raw=layer_map.get("comparisons"),
                    label=f"layer_specs.{layer_name}",
                )
            elif "include_fragment" in layer_map:
                fragment_id = _as_str(
                    layer_map.get("include_fragment"),
                    f"layer_specs.{layer_name}.include_fragment",
                )
                fragment = self._fragment_registry.get(fragment_id)
                if not fragment.stages:
                    raise SpecificationLoadError(
                        f"included fragment {fragment_id!r} must declare stages"
                    )
                layer_stages = tuple(
                    self._fragment_stage_to_stage_spec(stage)
                    for stage in fragment.stages
                )
                layer_stages = self._apply_fragment_stage_overrides(
                    layer_stages,
                    runtime_raw=layer_map.get("runtime_options"),
                    comparisons_raw=layer_map.get("comparisons"),
                    label=f"layer_specs.{layer_name}",
                )
            elif "stages" in layer_map:
                if "runtime_options" in layer_map or "comparisons" in layer_map:
                    raise SpecificationLoadError(
                        f"layer_specs.{layer_name} runtime_options/comparisons require compose"
                    )
                layer_stages = tuple(
                    self._parse_stage(item)
                    for item in _require_list(
                        layer_map.get("stages"),
                        f"layer_specs.{layer_name}.stages",
                    )
                )
            else:
                raise SpecificationLoadError(
                    f"layer_specs.{layer_name} must declare stages, include_fragment, or compose"
                )
            if not layer_stages:
                raise SpecificationLoadError(f"layer_specs.{layer_name}.stages must not be empty")
            layer_specs[layer_name] = LayerSpec(
                layer_kind=layer_name,
                stages=layer_stages,
            )

        rule: _LayerLayoutRule | None = None
        if "layer_layout_rule" in region:
            raw_rule = _require_mapping(region.get("layer_layout_rule"), "layer_layout_rule")
            strategy = _as_str(raw_rule.get("strategy"), "layer_layout_rule.strategy")
            if strategy == "repeat":
                _exact_keys(raw_rule, required={"strategy", "layer_kind", "count_from"}, label="layer_layout_rule")
                rule = _LayerLayoutRule(
                    strategy=strategy,
                    count_from=_as_str(raw_rule.get("count_from"), "layer_layout_rule.count_from"),
                    layer_kind=_as_str(raw_rule.get("layer_kind"), "layer_layout_rule.layer_kind"),
                )
                required_kinds = (rule.layer_kind,)
            elif strategy == "prefix_then_repeat":
                _exact_keys(
                    raw_rule,
                    required={
                        "strategy", "count_from", "prefix_layer_kind", "repeated_layer_kind",
                        "prefix_count_from",
                    },
                    label="layer_layout_rule",
                )
                rule = _LayerLayoutRule(
                    strategy=strategy,
                    count_from=_as_str(raw_rule.get("count_from"), "layer_layout_rule.count_from"),
                    prefix_layer_kind=_as_str(raw_rule.get("prefix_layer_kind"), "layer_layout_rule.prefix_layer_kind"),
                    repeated_layer_kind=_as_str(raw_rule.get("repeated_layer_kind"), "layer_layout_rule.repeated_layer_kind"),
                    prefix_count_from=_as_str(raw_rule.get("prefix_count_from"), "layer_layout_rule.prefix_count_from"),
                )
                required_kinds = (rule.prefix_layer_kind, rule.repeated_layer_kind)
            else:
                raise SpecificationLoadError(f"unsupported layer_layout_rule.strategy: {strategy}")
            missing_kinds = set(required_kinds).difference(layer_specs)
            if missing_kinds:
                raise SpecificationLoadError(
                    f"layer_layout_rule layer kind(s) missing from layer_specs: {sorted(missing_kinds)}"
                )

        activation = None
        if "activation" in region:
            activation = _as_str(region.get("activation"), f"region {region_id}.activation")
            try:
                self._activation_registry.resolve(activation)
            except KeyError as error:
                raise SpecificationLoadError(str(error)) from error

        return (
            RegionSpec(
                region_id=region_id,
                stages=stages,
                layer_layout=(),
                layer_specs=layer_specs,
            ),
            rule,
            activation,
        )

    def _parse_composed_mtp_region(
        self,
        region_id: str,
        region: Mapping[str, object],
    ) -> tuple[RegionSpec, _LayerLayoutRule, str | None]:
        """Compose one complete MTP Region from framework request and proposal roles."""

        compose = _require_mapping(
            region.get("compose"),
            f"region {region_id}.compose",
        )
        _exact_keys(
            compose,
            required={"framework", "predictor"},
            optional={"predictor_adapter"},
            label=f"region {region_id}.compose",
        )
        framework_id = _as_str(
            compose.get("framework"),
            f"region {region_id}.compose.framework",
        )
        predictor_id = _as_str(
            compose.get("predictor"),
            f"region {region_id}.compose.predictor",
        )
        adapter_id = (
            None
            if "predictor_adapter" not in compose
            else _as_str(
                compose.get("predictor_adapter"),
                f"region {region_id}.compose.predictor_adapter",
            )
        )
        framework = self._fragment_registry.require_kind(
            framework_id,
            kind="mtp_framework",
        )
        request_stages = tuple(
            self._fragment_stage_to_stage_spec(stage)
            for stage in framework.stage_group("request")
        )
        proposal_stages = tuple(
            self._fragment_stage_to_stage_spec(stage)
            for stage in compose_mtp_layer_stages(
                self._fragment_registry,
                framework_id=framework_id,
                predictor_id=predictor_id,
                predictor_adapter_id=adapter_id,
            )
        )
        request_count = len(request_stages)
        combined = self._apply_fragment_stage_overrides(
            request_stages + proposal_stages,
            runtime_raw=region.get("runtime_options"),
            comparisons_raw=region.get("comparisons"),
            label=f"region {region_id}",
        )
        request_stages = combined[:request_count]
        proposal_stages = combined[request_count:]

        raw_rule = _require_mapping(
            region.get("layer_layout_rule"),
            f"region {region_id}.layer_layout_rule",
        )
        _exact_keys(
            raw_rule,
            required={"strategy", "layer_kind", "count_from"},
            label=f"region {region_id}.layer_layout_rule",
        )
        rule = _LayerLayoutRule(
            strategy=_as_str(
                raw_rule.get("strategy"),
                f"region {region_id}.layer_layout_rule.strategy",
            ),
            layer_kind=_as_str(
                raw_rule.get("layer_kind"),
                f"region {region_id}.layer_layout_rule.layer_kind",
            ),
            count_from=_as_str(
                raw_rule.get("count_from"),
                f"region {region_id}.layer_layout_rule.count_from",
            ),
        )
        activation = None
        if "activation" in region:
            activation = _as_str(
                region.get("activation"),
                f"region {region_id}.activation",
            )
            try:
                self._activation_registry.resolve(activation)
            except KeyError as error:
                raise SpecificationLoadError(str(error)) from error
        return (
            RegionSpec(
                region_id=region_id,
                stages=request_stages,
                layer_specs={
                    rule.layer_kind: LayerSpec(
                        layer_kind=rule.layer_kind,
                        stages=proposal_stages,
                    )
                },
            ),
            rule,
            activation,
        )

    def _parse_composed_mtp_layer(self, raw: object, *, label: str) -> tuple[StageSpec, ...]:
        compose = _require_mapping(raw, label)
        _exact_keys(
            compose,
            required={"framework", "predictor"},
            optional={"predictor_adapter"},
            label=label,
        )
        composed = compose_mtp_layer_stages(
            self._fragment_registry,
            framework_id=_as_str(compose.get("framework"), f"{label}.framework"),
            predictor_id=_as_str(compose.get("predictor"), f"{label}.predictor"),
            predictor_adapter_id=(
                None
                if "predictor_adapter" not in compose
                else _as_str(
                    compose.get("predictor_adapter"),
                    f"{label}.predictor_adapter",
                )
            ),
        )
        return tuple(
            self._fragment_stage_to_stage_spec(fragment_stage)
            for fragment_stage in composed
        )

    def _fragment_stage_to_stage_spec(
        self,
        fragment_stage: TheoryFragmentStage,
    ) -> StageSpec:
        operators = fragment_stage.operators
        for operator in operators:
            if operator.activation is None:
                continue
            try:
                self._activation_registry.resolve(operator.activation)
            except KeyError as error:
                raise SpecificationLoadError(str(error)) from error
        source_options: dict[SourceKind, SourceStageOptions] = {
            SourceKind.THEORY: TheoryStageOptions(operators=operators)
        }
        if fragment_stage.runtime_options is not None:
            source_options[SourceKind.RUNTIME] = fragment_stage.runtime_options
        comparisons = (
            {}
            if fragment_stage.comparisons is None
            else self._parse_comparisons_mapping(
                fragment_stage.comparisons,
                label=f"fragment stage {fragment_stage.stage_id}.comparisons",
            )
        )
        return StageSpec(
            stage_id=fragment_stage.stage_id,
            source_options=source_options,
            comparisons=comparisons,
        )

    def _apply_fragment_stage_overrides(
        self,
        stages: tuple[StageSpec, ...],
        *,
        runtime_raw: object,
        comparisons_raw: object,
        label: str,
    ) -> tuple[StageSpec, ...]:
        stage_ids = {stage.stage_id for stage in stages}
        runtime_by_stage = (
            {}
            if runtime_raw is None
            else _require_mapping(runtime_raw, f"{label}.runtime_options")
        )
        comparisons_by_stage = (
            {}
            if comparisons_raw is None
            else _require_mapping(comparisons_raw, f"{label}.comparisons")
        )
        configured_ids = set(runtime_by_stage).union(comparisons_by_stage)
        unknown_ids = configured_ids.difference(stage_ids)
        if unknown_ids:
            raise SpecificationLoadError(
                f"{label} configures unknown included stage {sorted(unknown_ids)[0]!r}"
            )

        runtime_parser = self._source_options_parsers.get("runtime")
        if runtime_by_stage and runtime_parser is None:
            raise SpecificationLoadError("unregistered source options parser: 'runtime'")

        configured: list[StageSpec] = []
        for stage in stages:
            source_options = dict(stage.source_options)
            raw_runtime = runtime_by_stage.get(stage.stage_id)
            if raw_runtime is not None:
                source_options[SourceKind.RUNTIME] = runtime_parser.parse(
                    _require_mapping(
                        raw_runtime,
                        f"{label}.runtime_options.{stage.stage_id}",
                    )
                )

            comparisons = dict(stage.comparisons)
            raw_comparisons = comparisons_by_stage.get(stage.stage_id)
            if raw_comparisons is not None:
                comparisons.update(
                    self._parse_comparisons_mapping(
                        raw_comparisons,
                        label=f"{label}.comparisons.{stage.stage_id}",
                    )
                )

            configured.append(
                StageSpec(
                    stage_id=stage.stage_id,
                    source_options=source_options,
                    comparisons=comparisons,
                )
            )
        return tuple(configured)

    def _parse_stage(self, raw: object) -> StageSpec:
        stage = _require_mapping(raw, "stage")
        _exact_keys(
            stage,
            required={"id"},
            optional={"source_options", "include_stage", "runtime_options", "comparisons"},
            label="stage",
        )
        stage_id = _as_str(stage.get("id"), "stage.id")
        source_options: dict[SourceKind, SourceStageOptions] = {}
        comparisons: dict[tuple[SourceKind, SourceKind], ComparisonSpec] = {}
        if ("source_options" in stage) == ("include_stage" in stage):
            raise SpecificationLoadError(
                "stage must declare exactly one of source_options or include_stage"
            )
        if "include_stage" in stage:
            ref = _require_mapping(stage.get("include_stage"), "stage.include_stage")
            _exact_keys(
                ref,
                required={"fragment", "stage"},
                label="stage.include_stage",
            )
            fragment_id = _as_str(ref.get("fragment"), "stage.include_stage.fragment")
            fragment_stage_id = _as_str(ref.get("stage"), "stage.include_stage.stage")
            fragment_stage = self._fragment_registry.get(fragment_id).stage(fragment_stage_id)
            source_options[SourceKind.THEORY] = TheoryStageOptions(
                operators=fragment_stage.operators
            )
            if fragment_stage.runtime_options is not None:
                source_options[SourceKind.RUNTIME] = fragment_stage.runtime_options
            if "runtime_options" in stage:
                runtime_parser = self._source_options_parsers.get("runtime")
                if runtime_parser is None:
                    raise SpecificationLoadError("unregistered source options parser: 'runtime'")
                source_options[SourceKind.RUNTIME] = runtime_parser.parse(
                    _require_mapping(stage.get("runtime_options"), "stage.runtime_options")
                )
            if fragment_stage.comparisons is not None:
                comparisons.update(
                    self._parse_comparisons_mapping(
                        fragment_stage.comparisons,
                        label=f"fragment stage {fragment_stage.stage_id}.comparisons",
                    )
                )
        else:
            if "runtime_options" in stage:
                raise SpecificationLoadError(
                    "stage.runtime_options requires include_stage"
                )
            source_raw = _require_mapping(stage.get("source_options"), "source_options")
            for yaml_key, raw_options in source_raw.items():
                parser = self._source_options_parsers.get(str(yaml_key))
                if parser is None:
                    raise SpecificationLoadError(f"unregistered source options parser: {yaml_key!r}")
                source_options[parser.source_kind] = parser.parse(
                    _require_mapping(raw_options, f"source_options.{yaml_key}")
                )
        if not source_options:
            raise SpecificationLoadError(f"stage {stage_id!r} must declare source_options")
        theory = source_options.get(SourceKind.THEORY)
        if isinstance(theory, TheoryStageOptions):
            for operator in theory.operators:
                if operator.activation is None:
                    continue
                try:
                    self._activation_registry.resolve(operator.activation)
                except KeyError as error:
                    raise SpecificationLoadError(str(error)) from error

        if "comparisons" in stage:
            comparisons.update(
                self._parse_comparisons_mapping(
                    stage.get("comparisons"),
                    label="comparisons",
                )
            )

        return StageSpec(
            stage_id=stage_id,
            source_options=source_options,
            comparisons=comparisons,
        )

    def _parse_comparisons_mapping(
        self,
        raw: object,
        *,
        label: str,
    ) -> dict[tuple[SourceKind, SourceKind], ComparisonSpec]:
        comparisons_raw = _require_mapping(raw, label)
        if not comparisons_raw:
            raise SpecificationLoadError(
                f"{label} must be omitted when empty; do not write an empty mapping"
            )
        comparisons: dict[tuple[SourceKind, SourceKind], ComparisonSpec] = {}
        for key, body in comparisons_raw.items():
            comparison_key = _as_str(key, "comparison key")
            parts = comparison_key.split("-")
            if len(parts) != 2:
                raise SpecificationLoadError(
                    f"comparison key must be '<left>-<right>': {comparison_key!r}"
                )
            try:
                source_pair = (SourceKind(parts[0]), SourceKind(parts[1]))
            except ValueError as error:
                raise SpecificationLoadError(
                    f"comparison key has unknown source kind: {comparison_key!r}"
                ) from error
            comparisons[source_pair] = self._parse_comparison(body)
        return comparisons

    def _parse_comparison(self, raw: object) -> ComparisonSpec:
        body = _require_mapping(raw, "comparison")
        _exact_keys(
            body,
            required={"strategy"},
            optional={"options"},
            label="comparison",
        )
        strategy_id = _as_str(body.get("strategy"), "comparison.strategy")
        options_raw = body.get("options")
        options_map = {} if options_raw is None else _require_mapping(options_raw, "comparison.options")
        try:
            options = self._comparison_registry.parse_options(strategy_id, options_map)
        except (ComparisonOptionParseError, StrategyResolutionError) as exc:
            raise SpecificationLoadError(f"invalid comparison strategy {strategy_id!r}: {exc}") from exc
        return ComparisonSpec(strategy_id=strategy_id, options=options)
