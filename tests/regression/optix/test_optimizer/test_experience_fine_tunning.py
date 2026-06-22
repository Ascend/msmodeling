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

import numpy as np
import pytest

from optix.optimizer.experience_fine_tunning import FineTune, StopFineTune
from optix.config.config import OptimizerConfigField, PerformanceIndex


def _make_field(name, value, min_val=0.0, max_val=100.0, constant=None):
    return OptimizerConfigField(
        name=name,
        value=value,
        min=min_val,
        max=max_val,
        dtype="float",
        constant=constant,
    )


def _make_perf(ttft=0.3, tpot=0.04, gen_speed=2000, success_rate=1.0):
    return PerformanceIndex(
        time_to_first_token=ttft,
        time_per_output_token=tpot,
        generate_speed=gen_speed,
        success_rate=success_rate,
    )


class TestFineTuneInit:
    def test_normal_init(self):
        ft = FineTune(ttft_penalty=3.0, tpot_penalty=3.0)
        assert ft.ttft_penalty == 3.0
        assert ft.tpot_penalty == 3.0

    def test_zero_penalties_raises(self):
        with pytest.raises(StopFineTune):
            FineTune(ttft_penalty=0, tpot_penalty=0)

    def test_penalty_with_zero_slo_raises(self):
        with pytest.raises(ValueError, match="Penalty is set but SLO is zero"):
            FineTune(ttft_penalty=3.0, tpot_penalty=0, ttft_slo=0)


class TestUpdateField:
    def test_update_normal(self):
        field = _make_field("REQUESTRATE", 50.0, 10.0, 100.0)
        result = FineTune.update_field((field,), 0.5, field_names=("REQUESTRATE",))
        assert result is True
        assert field.value > 50.0

    def test_update_zero_factor(self):
        field = _make_field("REQUESTRATE", 50.0)
        result = FineTune.update_field((field,), 0.0, field_names=("REQUESTRATE",))
        assert result is False

    def test_update_inf_factor(self):
        field = _make_field("REQUESTRATE", 50.0)
        result = FineTune.update_field((field,), inf, field_names=("REQUESTRATE",))
        assert result is False

    def test_update_constant_field(self):
        field = _make_field("REQUESTRATE", 50.0, 50.0, 50.0, constant=50.0)
        result = FineTune.update_field((field,), 0.5, field_names=("REQUESTRATE",))
        assert result is False

    def test_update_with_last(self):
        field = _make_field("REQUESTRATE", 50.0, 10.0, 100.0)
        result = FineTune.update_field((field,), 0.5, field_names=("REQUESTRATE",), last=40.0)
        assert result is True

    def test_update_clamped_to_max(self):
        field = _make_field("REQUESTRATE", 95.0, 10.0, 100.0)
        result = FineTune.update_field((field,), 0.5, field_names=("REQUESTRATE",))
        assert result is True
        assert field.value <= 100.0

    def test_update_no_matching_field(self):
        field = _make_field("OTHER", 50.0)
        result = FineTune.update_field((field,), 0.5, field_names=("REQUESTRATE",))
        assert result is False


class TestAddHistory:
    def test_add_new_key(self):
        target = {}
        FineTune.add_history(target, "key1", 1.0)
        assert target == {"key1": [1.0]}

    def test_add_existing_key(self):
        target = {"key1": [1.0]}
        FineTune.add_history(target, "key1", 2.0)
        assert target == {"key1": [1.0, 2.0]}


class TestCheckConfigAndPerformance:
    def test_valid(self):
        ft = FineTune(ttft_penalty=3.0, tpot_penalty=3.0)
        perf = _make_perf()
        ft.check_config_and_performance(perf)

    def test_missing_tpot(self):
        ft = FineTune(ttft_penalty=3.0, tpot_penalty=3.0)
        perf = _make_perf()
        perf.time_per_output_token = None
        with pytest.raises(ValueError, match="Missing performance data for TPOT"):
            ft.check_config_and_performance(perf)

    def test_missing_ttft_with_penalty(self):
        ft = FineTune(ttft_penalty=3.0, tpot_penalty=3.0)
        perf = _make_perf()
        perf.time_to_first_token = None
        with pytest.raises(ValueError, match="Missing performance data for TTFT"):
            ft.check_config_and_performance(perf)


class TestDirectionOfFieldUpdate:
    def test_tpot_over_slo(self):
        ft = FineTune(ttft_penalty=3.0, tpot_penalty=3.0, tpot_slo=0.05, ttft_slo=0.5)
        perf = _make_perf(tpot=0.1, ttft=0.3)
        ft.direction_of_field_update(perf)
        assert ft.tpot_over_slo is True
        assert ft.tpot_under_lower_bound is False

    def test_tpot_under_lower(self):
        ft = FineTune(
            ttft_penalty=3.0,
            tpot_penalty=3.0,
            tpot_slo=0.05,
            ttft_slo=0.5,
            slo_coefficient=0.1,
        )
        perf = _make_perf(tpot=0.01, ttft=0.3)
        ft.direction_of_field_update(perf)
        assert ft.tpot_under_lower_bound is True

    def test_ttft_over_slo(self):
        ft = FineTune(ttft_penalty=3.0, tpot_penalty=3.0, tpot_slo=0.05, ttft_slo=0.5)
        perf = _make_perf(tpot=0.04, ttft=0.8)
        ft.direction_of_field_update(perf)
        assert ft.ttft_over_slo is True


class TestFineTuneWithConcurrencyAndRequestRate:
    def test_stop_when_no_change(self):
        ft = FineTune(ttft_penalty=3.0, tpot_penalty=3.0, tpot_slo=0.05, ttft_slo=0.5)
        fields = (
            _make_field("REQUESTRATE", 50.0, 50.0, 50.0, constant=50.0),
            _make_field("CONCURRENCY", 10.0, 10.0, 10.0, constant=10.0),
        )
        perf = _make_perf(tpot=0.04, ttft=0.4)
        params = np.array([50.0, 10.0])

        with pytest.raises(StopFineTune):
            with pytest.MonkeyPatch.context() as m:
                m.setattr(
                    "optix.optimizer.experience_fine_tunning.map_param_with_value",
                    lambda p, f: fields,
                )
                ft.fine_tune_with_concurrency_and_request_rate(params, perf)


class TestResetHistory:
    def test_reset(self):
        ft = FineTune(ttft_penalty=3.0, tpot_penalty=3.0)
        ft.last_signed_factor = {"key": [1.0]}
        ft.last_value = {"key": [2.0]}
        ft.reset_history()
        assert ft.last_signed_factor == {}
        assert ft.last_value == {}


class TestHandleConcurrency:
    def test_tpot_over_slo_adjusts_down(self):
        ft = FineTune(
            ttft_penalty=3.0,
            tpot_penalty=3.0,
            tpot_slo=0.05,
            ttft_slo=0.5,
            step_size=0.5,
        )
        perf = _make_perf(tpot=0.10, ttft=0.3)
        ft.direction_of_field_update(perf)
        field = _make_field("CONCURRENCY", 50.0, 1.0, 100.0)
        result = ft.handle_concurrency((field,), perf)
        assert result is True
        assert field.value < 50.0

    def test_tpot_under_lower_bound_adjusts_up(self):
        ft = FineTune(
            ttft_penalty=3.0,
            tpot_penalty=3.0,
            tpot_slo=0.05,
            ttft_slo=0.5,
            slo_coefficient=0.2,
        )
        perf = _make_perf(tpot=0.01, ttft=0.3)
        ft.direction_of_field_update(perf)
        field = _make_field("CONCURRENCY", 50.0, 1.0, 100.0)
        result = ft.handle_concurrency((field,), perf)
        assert result is True
        assert field.value > 50.0

    def test_no_tpot_issue_no_change(self):
        ft = FineTune(
            ttft_penalty=3.0,
            tpot_penalty=3.0,
            tpot_slo=0.05,
            ttft_slo=0.5,
            slo_coefficient=0.1,
        )
        perf = _make_perf(tpot=0.048, ttft=0.3)
        ft.direction_of_field_update(perf)
        field = _make_field("CONCURRENCY", 50.0, 1.0, 100.0)
        result = ft.handle_concurrency((field,), perf)
        assert result is False

    def test_oscillation_uses_midpoint(self):
        ft = FineTune(
            ttft_penalty=3.0,
            tpot_penalty=3.0,
            tpot_slo=0.05,
            ttft_slo=0.5,
            step_size=0.5,
        )
        from optix.config.base_config import CONCURRENCYS

        ft.last_signed_factor[CONCURRENCYS] = [0.3]
        ft.last_value[CONCURRENCYS] = [40.0, 50.0]
        perf = _make_perf(tpot=0.10, ttft=0.3)
        ft.direction_of_field_update(perf)
        field = _make_field("CONCURRENCY", 50.0, 1.0, 100.0)
        result = ft.handle_concurrency((field,), perf)
        assert result is True


class TestHandleRequestRate:
    def test_ttft_over_slo_adjusts_down(self):
        ft = FineTune(
            ttft_penalty=3.0,
            tpot_penalty=3.0,
            tpot_slo=0.05,
            ttft_slo=0.5,
            step_size=0.5,
        )
        perf = _make_perf(tpot=0.04, ttft=0.8)
        ft.direction_of_field_update(perf)
        field = _make_field("REQUESTRATE", 50.0, 1.0, 100.0)
        result = ft.handle_request_rate((field,), perf)
        assert result is True
        assert field.value < 50.0

    def test_ttft_under_lower_bound_adjusts_up(self):
        ft = FineTune(
            ttft_penalty=3.0,
            tpot_penalty=3.0,
            tpot_slo=0.05,
            ttft_slo=0.5,
            slo_coefficient=0.2,
        )
        perf = _make_perf(tpot=0.04, ttft=0.1)
        ft.direction_of_field_update(perf)
        field = _make_field("REQUESTRATE", 50.0, 1.0, 100.0)
        result = ft.handle_request_rate((field,), perf)
        assert result is True
        assert field.value > 50.0

    def test_no_ttft_issue_no_change(self):
        ft = FineTune(
            ttft_penalty=3.0,
            tpot_penalty=3.0,
            tpot_slo=0.05,
            ttft_slo=0.5,
            slo_coefficient=0.1,
        )
        perf = _make_perf(tpot=0.04, ttft=0.48)
        ft.direction_of_field_update(perf)
        field = _make_field("REQUESTRATE", 50.0, 1.0, 100.0)
        result = ft.handle_request_rate((field,), perf)
        assert result is False

    def test_oscillation_uses_midpoint(self):
        ft = FineTune(
            ttft_penalty=3.0,
            tpot_penalty=3.0,
            tpot_slo=0.05,
            ttft_slo=0.5,
            step_size=0.5,
        )
        from optix.config.base_config import REQUESTRATES

        ft.last_signed_factor[REQUESTRATES] = [0.3]
        ft.last_value[REQUESTRATES] = [40.0, 50.0]
        perf = _make_perf(tpot=0.04, ttft=0.8)
        ft.direction_of_field_update(perf)
        field = _make_field("REQUESTRATE", 50.0, 1.0, 100.0)
        result = ft.handle_request_rate((field,), perf)
        assert result is True


class TestFineTuneFullFlow:
    def test_concurrency_updated_resets_request_rate(self):
        ft = FineTune(
            ttft_penalty=3.0,
            tpot_penalty=3.0,
            tpot_slo=0.05,
            ttft_slo=0.5,
            step_size=0.5,
        )
        perf = _make_perf(tpot=0.10, ttft=0.3)
        fields = (
            _make_field("REQUESTRATE", 20.0, 1.0, 100.0),
            _make_field("CONCURRENCY", 50.0, 1.0, 100.0),
        )
        params = np.array([20.0, 50.0])
        with pytest.MonkeyPatch.context() as m:
            m.setattr(
                "optix.optimizer.experience_fine_tunning.map_param_with_value",
                lambda p, f: fields,
            )
            result = ft.fine_tune_with_concurrency_and_request_rate(params, perf)
        assert result is not None

    def test_request_rate_updated_when_concurrency_unchanged(self):
        ft = FineTune(
            ttft_penalty=3.0,
            tpot_penalty=3.0,
            tpot_slo=0.05,
            ttft_slo=0.5,
            step_size=0.5,
        )
        perf = _make_perf(tpot=0.04, ttft=0.8)
        fields = (
            _make_field("REQUESTRATE", 50.0, 1.0, 100.0),
            _make_field("CONCURRENCY", 50.0, 50.0, 50.0, constant=50.0),
        )
        params = np.array([50.0, 50.0])
        with pytest.MonkeyPatch.context() as m:
            m.setattr(
                "optix.optimizer.experience_fine_tunning.map_param_with_value",
                lambda p, f: fields,
            )
            result = ft.fine_tune_with_concurrency_and_request_rate(params, perf)
        assert result is not None

    def test_raises_stop_when_both_unchanged(self):
        ft = FineTune(
            ttft_penalty=3.0,
            tpot_penalty=3.0,
            tpot_slo=0.05,
            ttft_slo=0.5,
            slo_coefficient=0.1,
        )
        perf = _make_perf(tpot=0.048, ttft=0.48)
        fields = (
            _make_field("REQUESTRATE", 50.0, 50.0, 50.0, constant=50.0),
            _make_field("CONCURRENCY", 50.0, 50.0, 50.0, constant=50.0),
        )
        params = np.array([50.0, 50.0])
        with pytest.raises(StopFineTune):
            with pytest.MonkeyPatch.context() as m:
                m.setattr(
                    "optix.optimizer.experience_fine_tunning.map_param_with_value",
                    lambda p, f: fields,
                )
                ft.fine_tune_with_concurrency_and_request_rate(params, perf)

    def test_nan_factor_update_returns_false(self):
        field = _make_field("REQUESTRATE", 50.0, 1.0, 100.0)
        result = FineTune.update_field((field,), float("nan"), field_names=("REQUESTRATE",))
        assert result is False

    def test_min_equals_max_returns_false(self):
        field = _make_field("REQUESTRATE", 50.0, 50.0, 50.0)
        result = FineTune.update_field((field,), 0.5, field_names=("REQUESTRATE",))
        assert result is False
