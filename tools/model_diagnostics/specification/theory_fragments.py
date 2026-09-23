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

"""Static reusable stage-fragment registry for model diagnostics composition."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import yaml

from tools.model_diagnostics.domain.specification import RuntimeStageOptions, TheoryOperatorSpec
from tools.model_diagnostics.schema_utils import SchemaGuard, load_yaml_strict
from tools.model_diagnostics.specification.errors import SpecificationLoadError
from tools.model_diagnostics.specification.source_options import (
    RuntimeSourceOptionsParser,
    parse_theory_operator,
)

_DEFAULT_FRAGMENTS_DIR = Path(__file__).resolve().parents[1] / "specs" / "theory_fragments"
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

_FRAGMENT_KINDS = frozenset(
    {
        "mtp_framework",
        "mtp_predictor_adapter",
        "model_decoder",
        "vision_encoder",
    }
)


@dataclass(frozen=True)
class TheoryFragmentStage:
    stage_id: str
    operators: tuple[TheoryOperatorSpec, ...]
    repeat: str | None = None
    runtime_options: RuntimeStageOptions | None = None
    comparisons: Mapping[str, object] | None = None
    activation: str | None = None


@dataclass(frozen=True)
class TheoryFragment:
    fragment_id: str
    fragment_kind: str
    module_groups: Mapping[str, tuple[TheoryOperatorSpec, ...]]
    stages: tuple[TheoryFragmentStage, ...]
    stage_groups: Mapping[str, tuple[str, ...]]

    def group(self, group_id: str) -> tuple[TheoryOperatorSpec, ...]:
        try:
            return self.module_groups[group_id]
        except KeyError as error:
            raise SpecificationLoadError(
                f"theory fragment {self.fragment_id!r} has no module group {group_id!r}"
            ) from error

    def stage(self, stage_id: str) -> TheoryFragmentStage:
        for stage in self.stages:
            if stage.stage_id == stage_id:
                return stage
        raise SpecificationLoadError(
            f"theory fragment {self.fragment_id!r} has no stage {stage_id!r}"
        )

    def stage_group(self, group_id: str) -> tuple[TheoryFragmentStage, ...]:
        try:
            stage_ids = self.stage_groups[group_id]
        except KeyError as error:
            raise SpecificationLoadError(
                f"theory fragment {self.fragment_id!r} has no stage group {group_id!r}"
            ) from error
        return tuple(self.stage(stage_id) for stage_id in stage_ids)


class TheoryFragmentRegistry:
    """Load and resolve statically registered reusable fragments by stable ID."""

    def __init__(self, fragments: Mapping[str, TheoryFragment]) -> None:
        self._fragments = dict(fragments)

    def get(self, fragment_id: str) -> TheoryFragment:
        try:
            return self._fragments[fragment_id]
        except KeyError as error:
            raise SpecificationLoadError(
                f"unregistered theory fragment id: {fragment_id!r}"
            ) from error

    def require_kind(self, fragment_id: str, *, kind: str) -> TheoryFragment:
        fragment = self.get(fragment_id)
        if fragment.fragment_kind != kind:
            raise SpecificationLoadError(
                f"theory fragment {fragment_id!r} has kind {fragment.fragment_kind!r}, "
                f"expected {kind!r}"
            )
        return fragment


@dataclass(frozen=True)
class _ParsedFragmentFile:
    fragment_id: str
    fragment_kind: str
    source: str
    include_fragments: tuple[tuple[str, str | None], ...]
    runtime_options: Mapping[str, Mapping[str, object]]
    module_groups: Mapping[str, tuple[TheoryOperatorSpec, ...]]
    stages: tuple[TheoryFragmentStage, ...]
    stage_groups: Mapping[str, tuple[str, ...]]


def load_builtin_theory_fragment_registry(
    fragments_dir: Path | None = None,
) -> TheoryFragmentRegistry:
    root = fragments_dir or _DEFAULT_FRAGMENTS_DIR
    if not root.is_dir():
        raise SpecificationLoadError(f"theory fragments directory not found: {root}")
    parsed: dict[str, _ParsedFragmentFile] = {}
    for path in sorted(root.glob("*.yaml")):
        try:
            raw = load_yaml_strict(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as error:
            raise SpecificationLoadError(f"invalid theory fragment YAML {path}: {error}") from error
        document = _parse_fragment_file(raw, source=path.name)
        if document.fragment_id in parsed:
            raise SpecificationLoadError(f"duplicate theory fragment id: {document.fragment_id!r}")
        parsed[document.fragment_id] = document
    return TheoryFragmentRegistry(_resolve_fragment_includes(parsed))


def _parse_fragment_file(raw: object, *, source: str) -> _ParsedFragmentFile:
    payload = _require_mapping(raw, f"theory fragment {source}")
    _exact_keys(
        payload,
        required={"fragment_id", "fragment_kind"},
        optional={"module_groups", "stages", "stage_groups", "include_fragments", "runtime_options"},
        label=f"theory fragment {source}",
    )
    fragment_id = _as_str(payload.get("fragment_id"), "fragment_id")
    fragment_kind = _as_str(payload.get("fragment_kind"), "fragment_kind")
    include_refs: tuple[tuple[str, str | None], ...] = ()
    if "include_fragments" in payload:
        include_refs = tuple(
            _parse_fragment_include_ref(
                item,
                label=f"{fragment_id}.include_fragments[{index}]",
            )
            for index, item in enumerate(
                _require_list(payload.get("include_fragments"), "include_fragments")
            )
        )
        if not include_refs:
            raise SpecificationLoadError(
                f"theory fragment {fragment_id!r} include_fragments must not be empty"
            )
        child_ids = tuple(child_id for child_id, _activation in include_refs)
        if len(child_ids) != len(set(child_ids)):
            raise SpecificationLoadError(
                f"theory fragment {fragment_id!r} include_fragments contains duplicates"
            )
    fragment_runtime_options: Mapping[str, Mapping[str, object]] = {}
    if "runtime_options" in payload:
        runtime_options_raw = _require_mapping(
            payload.get("runtime_options"),
            "runtime_options",
        )
        fragment_runtime_options = {
            _as_str(stage_id, "runtime_options key"): _require_mapping(
                options,
                f"runtime_options.{stage_id}",
            )
            for stage_id, options in runtime_options_raw.items()
        }
    if fragment_kind not in _FRAGMENT_KINDS:
        raise SpecificationLoadError(
            f"unsupported theory fragment_kind {fragment_kind!r}; "
            f"expected one of {sorted(_FRAGMENT_KINDS)}"
        )

    module_groups: dict[str, tuple[TheoryOperatorSpec, ...]] = {}
    if "module_groups" in payload:
        groups_raw = _require_mapping(payload.get("module_groups"), "module_groups")
        for group_id, modules_raw in groups_raw.items():
            group_name = _as_str(group_id, "module_groups key")
            operators = tuple(
                parse_theory_operator(item, index)
                for index, item in enumerate(_require_list(modules_raw, f"module_groups.{group_name}"))
            )
            if not operators:
                raise SpecificationLoadError(f"module_groups.{group_name} must not be empty")
            module_groups[group_name] = operators

    stages: list[TheoryFragmentStage] = []
    if "stages" in payload:
        seen_stage_ids: set[str] = set()
        for index, stage_raw in enumerate(_require_list(payload.get("stages"), "stages")):
            stage_map = _require_mapping(stage_raw, f"stages[{index}]")
            _exact_keys(
                stage_map,
                required={"id", "modules"},
                optional={"runtime", "comparisons", "activation", "repeat"},
                label=f"stages[{index}]",
            )
            stage_id = _as_str(stage_map.get("id"), f"stages[{index}].id")
            if stage_id in seen_stage_ids:
                raise SpecificationLoadError(
                    f"theory fragment {fragment_id!r} ({source}) contains duplicate "
                    f"stage id {stage_id!r}; stage ids must be unique"
                )
            seen_stage_ids.add(stage_id)
            operators = tuple(
                parse_theory_operator(item, op_index)
                for op_index, item in enumerate(
                    _require_list(stage_map.get("modules"), f"stages[{index}].modules")
                )
            )
            if not operators:
                raise SpecificationLoadError(f"stages[{index}].modules must not be empty")
            activation = None
            if "activation" in stage_map:
                activation = _as_str(
                    stage_map.get("activation"),
                    f"stages[{index}].activation",
                )
            runtime_options = None
            if "runtime" in stage_map:
                try:
                    runtime_options = RuntimeSourceOptionsParser().parse(
                        _require_mapping(
                            stage_map.get("runtime"),
                            f"stages[{index}].runtime",
                        )
                    )
                except SpecificationLoadError as error:
                    raise SpecificationLoadError(
                        f"invalid runtime options in fragment {fragment_id!r} "
                        f"stage {stage_id!r}: {error}"
                    ) from error
            stages.append(
                TheoryFragmentStage(
                    stage_id=stage_id,
                    operators=operators,
                    repeat=(
                        _as_str(stage_map.get("repeat"), f"stages[{index}].repeat")
                        if "repeat" in stage_map
                        else None
                    ),
                    runtime_options=runtime_options,
                    comparisons=(
                        None
                        if "comparisons" not in stage_map
                        else _require_mapping(
                            stage_map.get("comparisons"),
                            f"stages[{index}].comparisons",
                        )
                    ),
                    activation=activation,
                )
            )

    stage_groups: dict[str, tuple[str, ...]] = {}
    if "stage_groups" in payload:
        groups_raw = _require_mapping(payload.get("stage_groups"), "stage_groups")
        known_stage_ids = {stage.stage_id for stage in stages}
        for group_id, stage_ids_raw in groups_raw.items():
            group_name = _as_str(group_id, "stage_groups key")
            stage_ids = tuple(
                _as_str(stage_id, f"stage_groups.{group_name} item")
                for stage_id in _require_list(
                    stage_ids_raw,
                    f"stage_groups.{group_name}",
                )
            )
            if len(stage_ids) != len(set(stage_ids)):
                raise SpecificationLoadError(
                    f"stage_groups.{group_name} contains duplicate stage ids"
                )
            unknown_ids = set(stage_ids).difference(known_stage_ids)
            if unknown_ids:
                raise SpecificationLoadError(
                    f"stage_groups.{group_name} references unknown stage "
                    f"{sorted(unknown_ids)[0]!r}"
                )
            stage_groups[group_name] = stage_ids

    if include_refs and stage_groups:
        raise SpecificationLoadError(
            f"theory fragment {fragment_id!r} stage_groups require locally declared stages"
        )
    if not module_groups and not stages and not include_refs:
        raise SpecificationLoadError(
            f"theory fragment {fragment_id!r} must declare module_groups, stages, or include_fragments"
        )
    return _ParsedFragmentFile(
        fragment_id=fragment_id,
        fragment_kind=fragment_kind,
        source=source,
        include_fragments=include_refs,
        runtime_options=fragment_runtime_options,
        module_groups=module_groups,
        stages=tuple(stages),
        stage_groups=stage_groups,
    )


def _parse_fragment_include_ref(raw: object, *, label: str) -> tuple[str, str | None]:
    if isinstance(raw, str):
        return _as_str(raw, label), None
    payload = _require_mapping(raw, label)
    _exact_keys(payload, required={"fragment"}, optional={"activation"}, label=label)
    fragment_id = _as_str(payload.get("fragment"), f"{label}.fragment")
    activation = (
        None
        if "activation" not in payload
        else _as_str(payload.get("activation"), f"{label}.activation")
    )
    return fragment_id, activation


def _resolve_fragment_includes(
    parsed: Mapping[str, _ParsedFragmentFile],
) -> dict[str, TheoryFragment]:
    resolved: dict[str, TheoryFragment] = {}
    visiting: set[str] = set()

    def resolve(fragment_id: str) -> TheoryFragment:
        cached = resolved.get(fragment_id)
        if cached is not None:
            return cached
        document = parsed.get(fragment_id)
        if document is None:
            raise SpecificationLoadError(f"unregistered theory fragment id: {fragment_id!r}")
        if fragment_id in visiting:
            raise SpecificationLoadError(
                f"theory fragment include_fragments cycle involving {fragment_id!r}"
            )
        if not document.include_fragments:
            stages = tuple(
                _apply_fragment_runtime_override(stage, document.runtime_options)
                for stage in document.stages
            )
            fragment = TheoryFragment(
                fragment_id=document.fragment_id,
                fragment_kind=document.fragment_kind,
                module_groups=document.module_groups,
                stages=stages,
                stage_groups=document.stage_groups,
            )
            resolved[fragment_id] = fragment
            return fragment
        visiting.add(fragment_id)
        stages: list[TheoryFragmentStage] = []
        seen_ids: set[str] = set()
        for child_id, fragment_activation in document.include_fragments:
            child = resolve(child_id)
            if not child.stages:
                raise SpecificationLoadError(
                    f"included fragment {child_id!r} must declare stages"
                )
            for stage in child.stages:
                if stage.stage_id in seen_ids:
                    raise SpecificationLoadError(
                        f"theory fragment {fragment_id!r} include_fragments repeats "
                        f"stage id {stage.stage_id!r}"
                    )
                seen_ids.add(stage.stage_id)
                if fragment_activation is not None and stage.activation is None:
                    stage = TheoryFragmentStage(
                        stage_id=stage.stage_id,
                        operators=stage.operators,
                        repeat=stage.repeat,
                        runtime_options=stage.runtime_options,
                        comparisons=stage.comparisons,
                        activation=fragment_activation,
                    )
                stages.append(_apply_fragment_runtime_override(stage, document.runtime_options))
        for stage in document.stages:
            if stage.stage_id in seen_ids:
                raise SpecificationLoadError(
                    f"theory fragment {fragment_id!r} local stage repeats included "
                    f"stage id {stage.stage_id!r}"
                )
            seen_ids.add(stage.stage_id)
            stages.append(_apply_fragment_runtime_override(stage, document.runtime_options))
        visiting.remove(fragment_id)
        fragment = TheoryFragment(
            fragment_id=document.fragment_id,
            fragment_kind=document.fragment_kind,
            module_groups=document.module_groups,
            stages=tuple(stages),
            stage_groups=document.stage_groups,
        )
        resolved[fragment_id] = fragment
        return fragment

    for fragment_id in parsed:
        resolve(fragment_id)
    return resolved


def _apply_fragment_runtime_override(
    stage: TheoryFragmentStage,
    runtime_options: Mapping[str, Mapping[str, object]],
) -> TheoryFragmentStage:
    """Apply a fragment-level per-stage Runtime override onto an included stage.

    Boundary operators replace the stage defaults; ignored operators are
    appended after the inherited set (deduplicated), mirroring the layer-level
    runtime override semantics used by the loader.
    """

    raw_override = runtime_options.get(stage.stage_id)
    if raw_override is None:
        return stage
    parser = RuntimeSourceOptionsParser()
    boundaries, extra_ignored = parser.parse_override(raw_override, label=stage.stage_id)
    inherited = stage.runtime_options
    if boundaries is None:
        if inherited is None:
            raise SpecificationLoadError(
                f"fragment runtime_options for stage {stage.stage_id!r} require an "
                "included Runtime stage"
            )
        boundaries = inherited.boundary_operators
    ignored = list(inherited.ignored_operators if inherited is not None else ())
    seen = set(ignored)
    for operator in extra_ignored:
        if operator in seen:
            continue
        ignored.append(operator)
        seen.add(operator)
    return TheoryFragmentStage(
        stage_id=stage.stage_id,
        operators=stage.operators,
        repeat=stage.repeat,
        runtime_options=RuntimeStageOptions(
            boundary_operators=boundaries,
            ignored_operators=tuple(ignored),
        ),
        comparisons=stage.comparisons,
        activation=stage.activation,
    )


def compose_mtp_layer_stages(
    registry: TheoryFragmentRegistry,
    *,
    framework_id: str,
    predictor_id: str,
    predictor_adapter_id: str | None = None,
    predictor_stage_group: str | None = None,
) -> tuple[TheoryFragmentStage, ...]:
    """Return ordered reusable stages for one MTP predictor layer."""

    framework = registry.require_kind(framework_id, kind="mtp_framework")
    adapter = (
        None
        if predictor_adapter_id is None
        else registry.require_kind(
            predictor_adapter_id,
            kind="mtp_predictor_adapter",
        )
    )
    predictor = registry.require_kind(predictor_id, kind="model_decoder")
    if not predictor.stages:
        raise SpecificationLoadError(
            f"model decoder fragment {predictor_id!r} must declare stages"
        )

    composed = list(framework.stage_group("proposal_prefix"))
    if adapter is not None:
        composed.extend(adapter.stage_group("before_predictor"))
    predictor_stages = (
        predictor.stages
        if predictor_stage_group is None
        else predictor.stage_group(predictor_stage_group)
    )
    composed.extend(predictor_stages)
    if adapter is not None:
        composed.extend(adapter.stage_group("after_predictor"))
    composed.extend(framework.stage_group("proposal_suffix"))
    stage_ids = [stage.stage_id for stage in composed]
    duplicate_ids = sorted(
        stage_id for stage_id in set(stage_ids) if stage_ids.count(stage_id) > 1
    )
    if duplicate_ids:
        raise SpecificationLoadError(
            f"MTP composition contains duplicate stage id {duplicate_ids[0]!r}; "
            "framework, predictor adapter, and predictor stage ids must be unique"
        )
    return tuple(composed)
