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
"""Semantic Runtime ignore-group contract tests."""

from collections import Counter
from pathlib import Path

import pytest

from tools.model_diagnostics.schema_utils import load_yaml_strict
from tools.model_diagnostics.specification.errors import SpecificationLoadError
from tools.model_diagnostics.specification.ignore_groups import (
    load_builtin_ignore_groups,
)
from tools.model_diagnostics.specification.source_options import (
    RuntimeSourceOptionsParser,
)


def test_runtime_options_expand_groups_then_stage_local_operators() -> None:
    parser = RuntimeSourceOptionsParser(
        ignore_groups={
            "layout": ("view", "reshape"),
            "transport": ("reshape", "all_gather"),
        }
    )

    options = parser.parse(
        {
            "boundary_operators": ["attention"],
            "ignored_operator_groups": ["layout", "transport"],
            "ignored_operators": ["view", "slice"],
        }
    )

    assert options.ignored_operators == (
        "view",
        "reshape",
        "all_gather",
        "slice",
    )


def test_runtime_options_reject_unknown_ignore_group() -> None:
    parser = RuntimeSourceOptionsParser(ignore_groups={"layout": ("view",)})

    with pytest.raises(SpecificationLoadError, match="unknown ignore group 'missing'"):
        parser.parse(
            {
                "boundary_operators": ["attention"],
                "ignored_operator_groups": ["missing"],
            }
        )


@pytest.mark.parametrize(
    ("groups", "selected", "local", "message"),
    (
        ({"layout": ("view",)}, ["layout", "layout"], [], "duplicate ignore group"),
        (
            {"layout": ("view",)},
            ["layout"],
            ["slice", "slice"],
            "duplicate operators",
        ),
    ),
)
def test_runtime_options_reject_ambiguous_ignore_composition(
    groups,
    selected,
    local,
    message,
) -> None:
    parser = RuntimeSourceOptionsParser(ignore_groups=groups)

    with pytest.raises(SpecificationLoadError, match=message):
        parser.parse(
            {
                "boundary_operators": ["attention"],
                "ignored_operator_groups": selected,
                "ignored_operators": local,
            }
        )


def test_builtin_ignore_groups_are_non_empty_and_semantic() -> None:
    groups = load_builtin_ignore_groups()

    assert groups["shape_views"]
    assert groups["quantization"]
    assert "mul" not in groups["shape_views"]


def test_every_builtin_ignore_group_is_used_by_a_spec_stage() -> None:
    specs_dir = Path(__file__).resolve().parents[4] / "tools" / "model_diagnostics" / "specs"
    references: Counter[str] = Counter()

    def collect(value: object) -> None:
        if isinstance(value, dict):
            references.update(value.get("ignored_operator_groups", ()))
            for nested in value.values():
                collect(nested)
        elif isinstance(value, list):
            for nested in value:
                collect(nested)

    for path in specs_dir.rglob("*.yaml"):
        if path.name != "ignore_groups.yaml":
            collect(load_yaml_strict(path.read_text(encoding="utf-8")))

    assert set(references) == set(load_builtin_ignore_groups())
    assert all(count >= 2 for count in references.values())
