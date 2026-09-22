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
from math import inf

from optix.config.config import PerformanceIndex
from optix.optimizer.performance_tunner import PerformanceTuner


def test_minimum_algorithm():
    tuner = PerformanceTuner()

    # Case: generate_speed is None
    index = PerformanceIndex(generate_speed=None)
    assert tuner.minimum_algorithm(index) == inf

    # Case: generate_speed is 0
    index = PerformanceIndex(generate_speed=0)
    assert tuner.minimum_algorithm(index) == inf

    # Case: time_to_first_token is None
    index = PerformanceIndex(generate_speed=1, time_to_first_token=None)
    assert tuner.minimum_algorithm(index) == inf

    # Case: time_to_first_token causes OverflowError
    index = PerformanceIndex(generate_speed=1, time_to_first_token=1e10)
    assert tuner.minimum_algorithm(index) == inf

    # Case: time_per_output_token is None
    index = PerformanceIndex(generate_speed=1, time_to_first_token=1, time_per_output_token=None)
    assert tuner.minimum_algorithm(index) == inf

    # Case: time_per_output_token causes OverflowError
    index = PerformanceIndex(generate_speed=1, time_to_first_token=1, time_per_output_token=1e10)
    assert tuner.minimum_algorithm(index) == inf

    # Case: success_rate is None
    index = PerformanceIndex(generate_speed=1, time_to_first_token=1, time_per_output_token=1, success_rate=None)
    assert tuner.minimum_algorithm(index) == inf

    # Case: success_rate is 0
    index = PerformanceIndex(generate_speed=1, time_to_first_token=1, time_per_output_token=1, success_rate=0)
    assert tuner.minimum_algorithm(index) == inf

    # Case: success_rate causes OverflowError
    index = PerformanceIndex(generate_speed=1, time_to_first_token=1, time_per_output_token=1, success_rate=1e-10)
    assert tuner.minimum_algorithm(index) == inf

    # Case: all parameters are normal
    index = PerformanceIndex(generate_speed=1, time_to_first_token=1, time_per_output_token=1, success_rate=1)
    assert tuner.minimum_algorithm(index) != inf
    # A better parameter combination yields a smaller value
    index = PerformanceIndex(generate_speed=1000, time_to_first_token=0.49, time_per_output_token=0.049, success_rate=1)
    index2 = PerformanceIndex(
        generate_speed=1000, time_to_first_token=0.29, time_per_output_token=0.014, success_rate=1
    )
    assert tuner.minimum_algorithm(index) > tuner.minimum_algorithm(index2)
    index = PerformanceIndex(generate_speed=1000, time_to_first_token=0.89, time_per_output_token=0.049, success_rate=1)
    index2 = PerformanceIndex(
        generate_speed=1000, time_to_first_token=0.59, time_per_output_token=0.014, success_rate=1
    )
    assert tuner.minimum_algorithm(index) > tuner.minimum_algorithm(index2)
    index = PerformanceIndex(generate_speed=1000, time_to_first_token=0.89, time_per_output_token=0.099, success_rate=1)
    index2 = PerformanceIndex(
        generate_speed=1000, time_to_first_token=0.59, time_per_output_token=0.054, success_rate=1
    )
    assert tuner.minimum_algorithm(index) > tuner.minimum_algorithm(index2)
    index = PerformanceIndex(generate_speed=1000, time_to_first_token=0.49, time_per_output_token=0.049, success_rate=1)
    index2 = PerformanceIndex(
        generate_speed=2000, time_to_first_token=0.49, time_per_output_token=0.049, success_rate=1
    )
    assert tuner.minimum_algorithm(index) > tuner.minimum_algorithm(index2)
