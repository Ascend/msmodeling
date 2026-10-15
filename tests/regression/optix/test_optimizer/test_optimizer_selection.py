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
"""Regression coverage for final OptiX candidate selection."""

from types import SimpleNamespace

import pytest

from optix.config.config import PdDisaggConfig, PerformanceIndex
from optix.optimizer.optimizer import PSOOptimizer


def _select_candidate(metrics, ttft_penalty=1, tpot_penalty=0, ttft_threshold=0.5):
    optimizer = PSOOptimizer(
        scheduler=SimpleNamespace(),
        fine_tune=SimpleNamespace(
            fine_tune_mode="pd_mixed",
            ttft_upper_bound=ttft_threshold,
            tpot_upper_bound=0.05,
        ),
        ttft_penalty=ttft_penalty,
        tpot_penalty=tpot_penalty,
    )
    performances = [
        PerformanceIndex(time_to_first_token=ttft, time_per_output_token=tpot, generate_speed=speed)
        for ttft, tpot, speed in metrics
    ]
    fitness = [float(i) for i in range(len(metrics))]
    params = [{"choice": i} for i in range(len(metrics))]
    result = optimizer.best_params(fitness, params, performances)
    return result, performances


@pytest.mark.parametrize(
    "metrics,expected_index",
    [
        pytest.param([(1.0, 0.01, 100), (0.2, 0.01, 1000)], 1, id="issue-498"),
        pytest.param([(0.2, 0.01, 100), (0.3, 0.01, 200)], 1, id="fastest-feasible"),
        pytest.param([(0.2, 0.01, 100), (0.8, 0.01, 1000)], 0, id="reject-faster-infeasible"),
        pytest.param([(1.0, 0.01, 1000), (0.6, 0.01, 100)], 1, id="all-infeasible"),
        pytest.param([(0.2, 0.01, 1000), (0.3, 0.01, 100)], 0, id="baseline-already-best"),
        pytest.param([(0.2, 0.01, 100), (0.3, 1.0, 1000)], 1, id="disabled-tpot-ignored"),
        pytest.param([(None, 0.01, 1000), (0.2, 0.01, 100)], 1, id="missing-ttft"),
        pytest.param([(1.0, 0.01, 1000), (0.5, 0.01, 100)], 1, id="at-slo-fallback"),
        pytest.param([(0.3, 0.01, 100), (0.5, 0.01, 1000)], 0, id="strict-slo-matches-other-modes"),
    ],
)
def test_ttft_only_selects_candidate(metrics, expected_index):
    result, performances = _select_candidate(metrics)

    assert result[0] == float(expected_index)
    assert result[1] == {"choice": expected_index}
    assert result[2] is performances[expected_index]


@pytest.mark.parametrize("ttft_penalty,tpot_penalty", [(0, 0), (0, 1), (1, 1)])
def test_other_constraint_modes_keep_best_candidate(ttft_penalty, tpot_penalty):
    result, performances = _select_candidate(
        [(1.0, 0.01, 100), (0.2, 0.01, 1000)],
        ttft_penalty,
        tpot_penalty,
    )

    assert result[1] == {"choice": 1}
    assert result[2] is performances[1]


def test_ttft_only_zero_threshold_preserves_baseline():
    result, _ = _select_candidate([(1.0, 0.01, 100), (0.2, 0.01, 1000)], ttft_threshold=0)

    assert result[1] == {"choice": 0}


def test_default_prefill_phase_selects_better_candidate():
    phase = PdDisaggConfig().model_dump()["prefill"]
    result, _ = _select_candidate(
        [(1.0, 0.01, 100), (0.2, 0.01, 1000)],
        phase["ttft_penalty"],
        phase["tpot_penalty"],
        phase["ttft_slo"],
    )

    assert result[1] == {"choice": 1}
