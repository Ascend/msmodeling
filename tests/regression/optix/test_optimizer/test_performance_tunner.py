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
from unittest.mock import MagicMock


from optix.optimizer.performance_tunner import PerformanceTuner


class TestPerformanceTuner:
    def setup_method(self):
        self.tuner = PerformanceTuner(
            ttft_penalty=3.0,
            tpot_penalty=3.0,
            success_rate_penalty=5.0,
            ttft_slo=0.5,
            tpot_slo=0.05,
            success_rate_slo=1.0,
            generate_speed_target=5300,
        )

    def _make_perf(self, gen_speed=2000, ttft=0.3, tpot=0.04, success_rate=1.0):
        perf = MagicMock()
        perf.generate_speed = gen_speed
        perf.time_to_first_token = ttft
        perf.time_per_output_token = tpot
        perf.success_rate = success_rate
        return perf

    def test_normal_calculation(self):
        perf = self._make_perf(gen_speed=5300, ttft=0.5, tpot=0.05, success_rate=1.0)
        result = self.tuner.minimum_algorithm(perf)
        assert result > 0
        assert result < inf

    def test_zero_generate_speed(self):
        perf = self._make_perf(gen_speed=0)
        assert self.tuner.minimum_algorithm(perf) == inf

    def test_none_generate_speed(self):
        perf = self._make_perf()
        perf.generate_speed = None
        assert self.tuner.minimum_algorithm(perf) == inf

    def test_negative_generate_speed(self):
        perf = self._make_perf(gen_speed=-100)
        assert self.tuner.minimum_algorithm(perf) == inf

    def test_zero_success_rate(self):
        perf = self._make_perf(success_rate=0)
        assert self.tuner.minimum_algorithm(perf) == inf

    def test_none_success_rate(self):
        perf = self._make_perf()
        perf.success_rate = None
        assert self.tuner.minimum_algorithm(perf) == inf

    def test_ttft_overflow(self):
        perf = self._make_perf(ttft=99999)
        assert self.tuner.minimum_algorithm(perf) == inf

    def test_tpot_overflow(self):
        perf = self._make_perf(tpot=99999)
        assert self.tuner.minimum_algorithm(perf) == inf

    def test_success_rate_overflow(self):
        perf = self._make_perf(success_rate=0.0001)
        assert self.tuner.minimum_algorithm(perf) == inf

    def test_no_ttft_penalty(self):
        tuner = PerformanceTuner(ttft_penalty=0, tpot_penalty=3.0)
        perf = self._make_perf()
        perf.time_to_first_token = None
        result = tuner.minimum_algorithm(perf)
        assert result > 0
        assert result < inf

    def test_no_tpot_penalty(self):
        tuner = PerformanceTuner(ttft_penalty=3.0, tpot_penalty=0)
        perf = self._make_perf()
        perf.time_per_output_token = None
        result = tuner.minimum_algorithm(perf)
        assert result > 0
        assert result < inf

    def test_custom_weights(self):
        tuner = PerformanceTuner()
        assert tuner.w_gen == 0.4
        assert tuner.w_ft == 0.2
        assert tuner.w_pot == 0.3
        assert tuner.w_succ == 0.1
