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
"""Tests for comparison-time operator naming defaults."""

from tools.model_diagnostics.comparison.operator_policy import (
    DEFAULT_OPERATOR_ALIASES,
    resolve_operator_aliases,
)


def test_defaults_align_theory_linears_to_runtime_canonical_names() -> None:
    assert DEFAULT_OPERATOR_ALIASES["o_projection"] == "mm"
    assert DEFAULT_OPERATOR_ALIASES["tensor_cast.fp8_linear.default"] == "mm"
    assert DEFAULT_OPERATOR_ALIASES["lm_head_select"] == "index"
    assert DEFAULT_OPERATOR_ALIASES["grouped_matmul_quant_swiglu"] == "grouped_matmul_swiglu"
    assert DEFAULT_OPERATOR_ALIASES["grouped_matmul_quant"] == "grouped_matmul"


def test_resolve_merges_spec_overrides_onto_defaults() -> None:
    aliases = resolve_operator_aliases({"custom_op": "mm", "o_projection": "addmm"})
    assert aliases["custom_op"] == "mm"
    assert aliases["o_projection"] == "addmm"
    assert aliases["lm_head"] == "mm"
