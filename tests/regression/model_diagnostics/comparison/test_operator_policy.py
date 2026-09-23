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
