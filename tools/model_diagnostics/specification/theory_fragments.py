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
"""Static reusable stage-fragment registry for model diagnostics composition."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import yaml

from tools.model_diagnostics.domain.specification import RuntimeStageOptions, TheoryOperatorSpec
from tools.model_diagnostics.schema_utils import SchemaGuard
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
    }
)


@dataclass(frozen=True)
class TheoryFragmentStage:
    stage_id: str
    operators: tuple[TheoryOperatorSpec, ...]
    runtime_options: RuntimeStageOptions | None = None
    comparisons: Mapping[str, object] | None = None


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


def load_builtin_theory_fragment_registry(
    fragments_dir: Path | None = None,
) -> TheoryFragmentRegistry:
    root = fragments_dir or _DEFAULT_FRAGMENTS_DIR
    if not root.is_dir():
        raise SpecificationLoadError(f"theory fragments directory not found: {root}")
    fragments: dict[str, TheoryFragment] = {}
    for path in sorted(root.glob("*.yaml")):
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as error:
            raise SpecificationLoadError(f"invalid theory fragment YAML {path}") from error
        fragment = _parse_fragment(raw, source=path.name)
        if fragment.fragment_id in fragments:
            raise SpecificationLoadError(f"duplicate theory fragment id: {fragment.fragment_id!r}")
        fragments[fragment.fragment_id] = fragment
    return TheoryFragmentRegistry(fragments)


def _parse_fragment(raw: object, *, source: str) -> TheoryFragment:
    payload = _require_mapping(raw, f"theory fragment {source}")
    _exact_keys(
        payload,
        required={"fragment_id", "fragment_kind"},
        optional={"module_groups", "stages", "stage_groups"},
        label=f"theory fragment {source}",
    )
    fragment_id = _as_str(payload.get("fragment_id"), "fragment_id")
    fragment_kind = _as_str(payload.get("fragment_kind"), "fragment_kind")
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
                optional={"runtime", "comparisons"},
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
                    runtime_options=runtime_options,
                    comparisons=(
                        None
                        if "comparisons" not in stage_map
                        else _require_mapping(
                            stage_map.get("comparisons"),
                            f"stages[{index}].comparisons",
                        )
                    ),
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

    if not module_groups and not stages:
        raise SpecificationLoadError(
            f"theory fragment {fragment_id!r} must declare module_groups and/or stages"
        )
    return TheoryFragment(
        fragment_id=fragment_id,
        fragment_kind=fragment_kind,
        module_groups=module_groups,
        stages=tuple(stages),
        stage_groups=stage_groups,
    )


def compose_mtp_layer_stages(
    registry: TheoryFragmentRegistry,
    *,
    framework_id: str,
    predictor_id: str,
    predictor_adapter_id: str | None = None,
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
    composed.extend(predictor.stages)
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
