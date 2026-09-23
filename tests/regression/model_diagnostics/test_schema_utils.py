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

"""SchemaGuard boundary helper tests."""

from __future__ import annotations

import pytest
import yaml

from tools.model_diagnostics.schema_utils import SchemaGuard, load_yaml_strict
from tools.model_diagnostics.specification.errors import SpecificationLoadError


def test_exact_keys_reports_mixed_type_unknown_keys_without_typeerror() -> None:
    guard = SchemaGuard(error=SpecificationLoadError, require_string_keys=False)

    with pytest.raises(SpecificationLoadError, match="fields mismatch") as caught:
        guard.exact_keys({1: True, "extra": True}, required={"name"}, label="item")

    assert "unknown=" in str(caught.value)
    assert not isinstance(caught.value.__cause__, TypeError)


def test_load_yaml_strict_rejects_duplicate_keys_at_any_depth() -> None:
    with pytest.raises(yaml.constructor.ConstructorError, match="duplicate key 'size'"):
        load_yaml_strict(
            """
parallel:
  size: 2
  size: 4
"""
        )


@pytest.mark.parametrize(
    "text",
    [
        "parallel: {<<: {size: 1}, size: 2}",
        "parallel: {size: 2, <<: {size: 1}}",
        "parallel: {<<: [{size: 2}, {size: 1}]}",
        "defaults: &defaults {<<: {size: 1}, size: 2}\nparallel: {<<: *defaults}",
    ],
)
def test_load_yaml_strict_preserves_merge_precedence(text: str) -> None:
    assert load_yaml_strict(text)["parallel"] == {"size": 2}


@pytest.mark.parametrize(
    "text",
    [
        "parallel: {<<: {size: 1}, size: 2, size: 3}",
        "parallel: {<<: {size: 1, size: 2}}",
        "parallel: {<<: [{size: 1, size: 2}, {size: 3}]}",
    ],
)
def test_load_yaml_strict_rejects_explicit_duplicates_with_merges(text: str) -> None:
    with pytest.raises(yaml.constructor.ConstructorError, match="duplicate key 'size'"):
        load_yaml_strict(text)
