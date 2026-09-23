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

"""Nightly marks that cannot be attached on the test method itself.

``parameterized.expand`` on unittest ``TestCase`` cannot take per-case
``pytest.param(marks=...)`` (pytest forwards mark node ids as extra positional
args; see ``tests/regression/tensor_cast/test_parameterized_pytest_param_compat.py``).
Method-level ``@pytest.mark.nightly`` is also not copied onto the generated
``test_*_N_...`` node ids.

Cases that can use ``@pytest.mark.nightly`` or native ``pytest.param(..., marks=)``
are marked at the test site and are intentionally absent here.

Sibling expand parameters that stayed under 4m30s are also absent.
"""

from __future__ import annotations

# Node ids as collected from the repo root (pytest default rootdir).
EXPAND_NIGHTLY_NODE_IDS: frozenset[str] = frozenset(
    {
        # Mixed expand: only this parameter exceeded 4m30s.
        "tests/regression/tensor_cast/test_text_generate.py::TestTextGenerate::test_ling_basic_0_inclusionAI_Ling_1T",
        "tests/regression/tensor_cast/test_text_generate.py::TestTextGenerate::test_gate_returns_precomputed_topk_3_Qwen_Qwen3_5_397B_A17B",
        "tests/regression/tensor_cast/test_quant_linear.py::TestQuantLinear::test_model_quant_tensorcast_dynamic_w4a8_2_zai_org_GLM_4_5",
        # Whole-method expand: every remaining parameter is slow, but the
        # method decorator does not land on the generated node id.
        "tests/regression/tensor_cast/test_text_generate.py::TestTextGenerate::test_mla_int8_with_linear_quant_0_deepseek_ai_DeepSeek_V3_1",
        "tests/regression/tensor_cast/test_text_generate.py::TestTextGenerate::test_mlapo_linear_quant_0_deepseek_ai_DeepSeek_V3_1",
        "tests/regression/tensor_cast/test_text_generate.py::TestTextGenerate::test_mlapo_quant_disabled_0_deepseek_ai_DeepSeek_V3_1",
        "tests/regression/tensor_cast/test_parallel_linear.py::ParallelLinearTestCase::test_deepseek_with_tp_and_dp_0_deepseek_ai_DeepSeek_V3_1",
        "tests/regression/tensor_cast/test_parallel_linear.py::ParallelLinearTestCase::test_deepseek_with_tp_and_dp_1_deepseek_ai_DeepSeek_V3_1",
        "tests/regression/tensor_cast/test_runtime.py::PerfAnalysisTestCase::test_deepseek_0_deepseek_ai_DeepSeek_V3_1",
    }
)
