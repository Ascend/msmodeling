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

"""Shared assertions for model-diagnostics tests."""

from __future__ import annotations

import math

from tools.model_diagnostics.domain import ParallelContext


def assert_parallel_contract(env: dict[str, object], parallel: ParallelContext, *, raw_logits_gate: bool) -> None:
    """Validate rank-local heads and MoE token domains for a parallel layout."""

    assert env["Lh"] == env["Nh"] // parallel.tensor_parallel_size
    if parallel.expert_parallel_size > 1:
        assert env["Tmoe"] == math.ceil(env["T"] / parallel.tensor_parallel_size)
    else:
        assert env["Tmoe"] == env["T"] * parallel.data_parallel_size
    if raw_logits_gate and parallel.expert_parallel_size > 1:
        assert env["MOE_GATE_TOKENS"] == env["T"]
    else:
        assert env["MOE_GATE_TOKENS"] == env["Tmoe"]
