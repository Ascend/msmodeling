# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
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
import math
import numpy as np
import pytest

from optix.config.config import OptimizerConfigField, PerformanceIndex
from optix.optimizer.experience_fine_tunning import FineTune, StopFineTune


def test_update_field():
    fine_tune = FineTune(ttft_penalty=1, tpot_penalty=1)
    my_support_field = [
        OptimizerConfigField(name="CONCURRENCY", config_position="env", min=10, max=1001, dtype="int", value=200),
        OptimizerConfigField(name="REQUESTRATE", config_position="env", min=0, max=1001, dtype="int", value=20),
    ]
    proportion = 0.05
    _flag = fine_tune.update_field(my_support_field, proportion)
    assert _flag
    assert my_support_field[-1].value == 21
    proportion = -0.05
    _flag = fine_tune.update_field(my_support_field, proportion)
    assert _flag
    assert math.isclose(my_support_field[-1].value, 19.95, rel_tol=1e-9)


def test_fine_tune_with_concurrency_and_request_rate():
    """Test decreasing prefill size when both ttft and tpot exceed SLO and the ttft gap is large."""
    my_support_field = [
        OptimizerConfigField(name="CONCURRENCY", config_position="env", min=10, max=1001, dtype="int", value=200),
        OptimizerConfigField(name="REQUESTRATE", config_position="env", min=0, max=1001, dtype="int", value=20),
    ]
    params = np.array([100, 20])
    mindie_fine_tune = FineTune(tpot_penalty=1, ttft_penalty=1, target_field=tuple(my_support_field))
    performance_index = PerformanceIndex()
    performance_index.time_to_first_token = 0.5
    performance_index.time_per_output_token = 0.05
    # SLOs are satisfied
    with pytest.raises(StopFineTune):
        mindie_fine_tune.fine_tune_with_concurrency_and_request_rate(params, performance_index)
    performance_index.time_to_first_token = 0.46
    performance_index.time_per_output_token = 0.046
    with pytest.raises(StopFineTune):
        mindie_fine_tune.fine_tune_with_concurrency_and_request_rate(params, performance_index)
    mindie_fine_tune.ttft_penalty = 0
    mindie_fine_tune.tpot_penalty = 0
    with pytest.raises(StopFineTune):
        mindie_fine_tune.fine_tune_with_concurrency_and_request_rate(params, performance_index)
    # Case: only tpot is penalized
    mindie_fine_tune.reset_history()
    mindie_fine_tune.tpot_penalty = 3.0
    with pytest.raises(StopFineTune):
        mindie_fine_tune.fine_tune_with_concurrency_and_request_rate(params, performance_index)
    mindie_fine_tune.reset_history()
    performance_index.time_per_output_token = 0.06
    result = mindie_fine_tune.fine_tune_with_concurrency_and_request_rate(params, performance_index)
    assert result[0].value == 100 * (1 - 0.01 / 0.05 * 0.5)
    performance_index.time_per_output_token = 0.04
    result = mindie_fine_tune.fine_tune_with_concurrency_and_request_rate(np.array([50, 20]), performance_index)
    assert result[0].value == 50 + 50 * (0.01 / 0.05 * 0.5)
    # Case: both tpot and ttft are penalized
    mindie_fine_tune.reset_history()
    mindie_fine_tune.ttft_penalty = 3.0
    performance_index.time_per_output_token = 0.06
    performance_index.time_to_first_token = 0.58
    result = mindie_fine_tune.fine_tune_with_concurrency_and_request_rate(params, performance_index)
    assert result[0].value == 100 * (1 - 0.01 / 0.05 * 0.5)
    performance_index.time_per_output_token = 0.04
    performance_index.time_to_first_token = 0.6
    result = mindie_fine_tune.fine_tune_with_concurrency_and_request_rate(np.array([90, 20]), performance_index)
    assert result[0].value == 90 + 10 * ((0.05 - 0.04) / 0.05 * 0.5)
    performance_index.time_per_output_token = 0.05
    performance_index.time_to_first_token = 0.3
    result = mindie_fine_tune.fine_tune_with_concurrency_and_request_rate(np.array([91, 20]), performance_index)
    assert result[-1].value == 30
    performance_index.time_per_output_token = 0.05
    performance_index.time_to_first_token = 0.4
    result = mindie_fine_tune.fine_tune_with_concurrency_and_request_rate(np.array([91, 80]), performance_index)
    assert result[-1].value == 91
    performance_index.time_per_output_token = 0.051
    performance_index.time_to_first_token = 0.6
    result = mindie_fine_tune.fine_tune_with_concurrency_and_request_rate(np.array([91, 91]), performance_index)
    assert result[-1].value < 91
    performance_index.time_per_output_token = 0.068
    performance_index.time_to_first_token = 0.6
    result = mindie_fine_tune.fine_tune_with_concurrency_and_request_rate(np.array([91, 91]), performance_index)
    assert result[0].value < 91
    assert result[-1].value == 20
