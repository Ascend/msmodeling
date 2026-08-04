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
"""Theory expression evaluator tests (no Runtime fixtures)."""

from __future__ import annotations

import pytest

from tools.model_diagnostics.specification.errors import SpecificationLoadError
from tools.model_diagnostics.specification.expressions import evaluate_dtype, evaluate_shape


def test_shape_arithmetic_and_functions() -> None:
    env = {
        "T": 8,
        "H": 4096,
        "Lh": 32,
        "Dh": 128,
        "Nkv": 8,
        "TP": 1,
        "MTP": 0,
        "Rtgt": 1,
        "Rprop": 1,
        "unknown": None,
    }
    assert evaluate_shape("[T, H]", env) == (8, 4096)
    assert evaluate_shape("[T, Lh * Dh]", env) == (8, 4096)
    assert evaluate_shape("[T, max(Nkv // TP, 1) * Dh]", env) == (8, 1024)
    assert evaluate_shape("[B, Q + MTP]", {"B": 2, "Q": 4, "MTP": 1, "unknown": None}) == (2, 5)
    assert evaluate_shape("[Rtgt, H]", {"Rtgt": 6, "H": 4096, "unknown": None}) == (6, 4096)
    assert evaluate_shape("[Rprop, Vtp]", {"Rprop": 2, "Vtp": 1000, "unknown": None}) == (2, 1000)
    assert evaluate_shape("[B, MTP + 1]", {"B": 2, "MTP": 2, "unknown": None}) == (2, 3)
    assert evaluate_shape("[ceil(2.2)]", {"unknown": None}) == (3,)


def test_dtype_variable_binding() -> None:
    assert evaluate_dtype("D", {"D": "bfloat16", "unknown": None}) == "bfloat16"
    assert evaluate_dtype("int64", {"int64": "int64", "unknown": None}) == "int64"


def test_unknown_shape_and_dtype() -> None:
    env = {"T": 8, "H": 4096, "unknown": None}
    assert evaluate_shape("?", env) is None
    assert evaluate_shape("unknown", env) is None
    assert evaluate_shape("[T, unknown]", env) is None
    assert evaluate_dtype("?", env) is None


def test_rejects_python_eval_and_unknown_names() -> None:
    with pytest.raises(SpecificationLoadError):
        evaluate_shape("__import__('os').name", {})
    with pytest.raises(SpecificationLoadError):
        evaluate_shape("[missing]", {"unknown": None})
    with pytest.raises(SpecificationLoadError):
        evaluate_shape("[T / 3]", {"T": 8, "unknown": None})


@pytest.mark.parametrize("expression", ("[H / 0]", "[H // 0]", "[H % 0]"))
def test_rejects_division_by_zero(expression: str) -> None:
    with pytest.raises(SpecificationLoadError, match="division by zero"):
        evaluate_shape(expression, {"H": 8, "unknown": None})
