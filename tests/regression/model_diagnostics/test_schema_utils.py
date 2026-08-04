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
"""SchemaGuard boundary helper tests."""

from __future__ import annotations

import pytest

from tools.model_diagnostics.schema_utils import SchemaGuard
from tools.model_diagnostics.specification.errors import SpecificationLoadError


def test_exact_keys_reports_mixed_type_unknown_keys_without_typeerror() -> None:
    guard = SchemaGuard(error=SpecificationLoadError, require_string_keys=False)

    with pytest.raises(SpecificationLoadError, match="fields mismatch") as caught:
        guard.exact_keys({1: True, "extra": True}, required={"name"}, label="item")

    assert "unknown=" in str(caught.value)
    assert not isinstance(caught.value.__cause__, TypeError)
