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

"""Generate config combinations for parameterized tests."""

from collections.abc import Iterable


def build_case_matrix(**dimensions: Iterable[object]) -> list[dict[str, object]]:
    """Build cartesian product matrix from named dimensions."""
    cases: list[dict[str, object]] = [{}]
    for key, values in dimensions.items():
        value_list = list(values)
        next_cases: list[dict[str, object]] = []
        for case in cases:
            for value in value_list:
                item = dict(case)
                item[key] = value
                next_cases.append(item)
        cases = next_cases
    return cases


def build_latency_thresholds(*, ttft_ms: float, tpot_ms: float, tolerance_ms: float = 0.1) -> dict[str, float]:
    """Create threshold config shared by serving latency tests."""
    return {
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "tolerance_ms": tolerance_ms,
    }
