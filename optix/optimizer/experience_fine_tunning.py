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
from itertools import cycle
from math import floor, isinf, isnan
from typing import Optional, Tuple

import numpy as np

from ..config.config import (
    default_support_field,
    PerformanceIndex,
    map_param_with_value,
    OptimizerConfigField,
)
from ..config.base_config import REQUESTRATES, CONCURRENCYS

FINE_TUNE_MODE_PD_MIXED = "pd_mixed"
FINE_TUNE_MODE_PD_DISAGGREGATION = "pd_disaggregation"


class StopFineTune(Exception):
    pass


class FineTune:
    def __init__(
        self,
        ttft_penalty: float = 0,
        tpot_penalty: float = 0,
        target_field: Optional[Tuple] = None,
        ttft_slo: float = 0.5,
        tpot_slo: float = 0.05,
        slo_coefficient: float = 0.1,
        step_size: float = 0.5,
        fine_tune_mode: str = FINE_TUNE_MODE_PD_MIXED,
    ):
        self.ttft_penalty = ttft_penalty  # Penalty coefficient in optimization algorithm
        self.tpot_penalty = tpot_penalty
        self.ttft_slo = ttft_slo
        self.tpot_slo = tpot_slo
        self.slo_coefficient = slo_coefficient
        self.target_field = target_field if target_field else default_support_field
        self.fine_tune_target = ["REQUESTRATE"]
        self.fine_tune_type = cycle(self.fine_tune_target)
        self.step_size = step_size
        self.fine_tune_mode = fine_tune_mode
        self.ttft_lower_bound = self.ttft_slo * (1 - self.slo_coefficient)
        self.ttft_upper_bound = self.ttft_slo
        self.tpot_lower_bound = self.tpot_slo * (1 - self.slo_coefficient)
        self.tpot_upper_bound = self.tpot_slo
        if self.ttft_penalty == 0 and self.tpot_penalty == 0:
            raise StopFineTune("No penalties, no need to fine-tune.")
        ttft_flag = self.ttft_penalty != 0 and self.ttft_slo == 0
        tpot_flag = self.tpot_penalty != 0 and self.tpot_slo == 0
        if ttft_flag or tpot_flag:
            raise ValueError("Penalty is set but SLO is zero.")
        self.ttft_over_slo = False
        self.ttft_under_lower_bound = False
        self.tpot_over_slo = False
        self.tpot_under_lower_bound = False
        self.last_signed_factor = {}
        self.last_value = {}
        self.pd_fixed_request_rate = None
        self.pd_lower_concurrency = None
        self.pd_upper_concurrency = None

    @staticmethod
    def update_field(
        simulate_run_info,
        signed_factor,
        field_names: tuple = REQUESTRATES,
        last: Optional[float] = None,
        force_min_change: bool = False,
    ) -> bool:
        if signed_factor == 0 or isinf(signed_factor) or isnan(signed_factor):
            return False
        for _field in simulate_run_info:
            if _field.name.upper().strip() in field_names:
                if _field.constant is not None or _field.min == _field.max:
                    return False
                original_value = _field.value
                if _field.name.upper().strip() in REQUESTRATES and _field.value == 0:
                    if signed_factor > 0:
                        return False
                    # For benchmark tools, REQUESTRATE=0 means unlimited QPS, not numeric zero.
                    # When reducing pressure from unlimited, start from the finite upper bound.
                    finite_base = _field.max
                    if isinf(finite_base) or isnan(finite_base) or finite_base <= 0:
                        finite_base = last if last and last > 0 else _field.min
                    min_change_base = finite_base
                    _new_value = finite_base * (1 + signed_factor)
                elif last:
                    min_change_base = _field.value
                    _new_value = _field.value + signed_factor * abs(_field.value - last)
                else:
                    min_change_base = _field.value
                    _new_value = _field.value * (1 + signed_factor)
                if isinf(_new_value) or isnan(_new_value):
                    return False
                _field.value = _new_value
                _new_value = max(_field.min, min(_field.max, _field.value))
                if isinf(_new_value) or isnan(_new_value):
                    _field.value = original_value
                    return False
                _field.value = _new_value
                if abs(_field.value - original_value) >= 0.1:
                    return True
                if not force_min_change:
                    return False
                min_delta = 1 if _field.dtype == "int" else 0.1
                direction = 1 if signed_factor > 0 else -1
                _field.value = _field.find_available_value(min_change_base + direction * min_delta)
                return abs(_field.value - original_value) >= 0.1
        return False

    @staticmethod
    def add_history(target, key_name, value):
        if key_name in target:
            target[key_name].append(value)
        else:
            target[key_name] = [value]

    def reset_history(self):
        self.last_signed_factor = {}
        self.last_value = {}
        self.pd_fixed_request_rate = None
        self.pd_lower_concurrency = None
        self.pd_upper_concurrency = None

    @staticmethod
    def is_field_name(field: OptimizerConfigField, names: tuple) -> bool:
        return field.name.upper().strip() in names

    @staticmethod
    def floor_to_one_decimal(value: float) -> float:
        return floor(value * 10) / 10

    @staticmethod
    def get_qps(performance_index: PerformanceIndex) -> float:
        if performance_index.throughput is not None and performance_index.throughput > 0:
            return performance_index.throughput
        raise ValueError("Missing positive throughput/QPS metric for PD disaggregation fine-tune.")

    @staticmethod
    def find_field(simulate_run_info: Tuple[OptimizerConfigField, ...], names: tuple):
        for field in simulate_run_info:
            if FineTune.is_field_name(field, names):
                return field
        return None

    @staticmethod
    def set_field_value(field: OptimizerConfigField, value, allow_below_min: bool = False) -> bool:
        if field is None:
            return False
        original_value = field.value
        if allow_below_min:
            field.value = field.convert_dtype(value)
        else:
            field.value = field.find_available_value(value)
        return abs(float(field.value) - float(original_value)) >= 0.1

    def prepare_pd_disaggregation_probe(self, params: np.ndarray):
        simulate_run_info = map_param_with_value(params, self.target_field)
        request_rate_field = self.find_field(simulate_run_info, REQUESTRATES)
        if request_rate_field is None:
            raise ValueError("PD disaggregation fine-tune requires REQUESTRATE.")
        self.set_field_value(request_rate_field, 0, allow_below_min=True)
        return simulate_run_info

    def init_pd_disaggregation_request_rate(self, performance_index: PerformanceIndex) -> float:
        request_rate = self.floor_to_one_decimal(self.get_qps(performance_index))
        if request_rate <= 0:
            raise ValueError("PD disaggregation fine-tune got non-positive request rate.")
        self.pd_fixed_request_rate = request_rate
        return request_rate

    def handle_pd_disaggregation_concurrency(
        self,
        simulate_run_info: Tuple[OptimizerConfigField, ...],
        performance_index: PerformanceIndex,
    ) -> bool:
        concurrency_field = self.find_field(simulate_run_info, CONCURRENCYS)
        request_rate_field = self.find_field(simulate_run_info, REQUESTRATES)
        if concurrency_field is None or request_rate_field is None:
            raise ValueError("PD disaggregation fine-tune requires CONCURRENCY and REQUESTRATE.")
        if self.pd_fixed_request_rate is None:
            raise ValueError("PD disaggregation fixed request rate is not initialized.")
        if performance_index.time_per_output_token is None:
            raise ValueError("Missing performance data for TPOT.")

        current_concurrency = float(concurrency_field.value)
        if current_concurrency <= 0:
            raise StopFineTune("Current concurrency must be positive.")

        if performance_index.time_per_output_token > self.tpot_upper_bound:
            self.pd_upper_concurrency = current_concurrency
            if self.pd_lower_concurrency is None:
                next_concurrency = current_concurrency / 2
            else:
                next_concurrency = (self.pd_lower_concurrency + current_concurrency) / 2
        else:
            self.pd_lower_concurrency = current_concurrency
            if self.pd_upper_concurrency is None:
                next_concurrency = current_concurrency * 2
            else:
                next_concurrency = (current_concurrency + self.pd_upper_concurrency) / 2

        self.set_field_value(request_rate_field, self.pd_fixed_request_rate)
        was_updated = self.set_field_value(concurrency_field, next_concurrency)
        if not was_updated:
            raise StopFineTune("Parameter value reached its boundary or did not change.")
        return True

    def fine_tune_pd_disaggregation(self, params: np.ndarray, performance_index: PerformanceIndex):
        simulate_run_info = map_param_with_value(params, self.target_field)
        self.handle_pd_disaggregation_concurrency(simulate_run_info, performance_index)
        return simulate_run_info

    def check_config_and_performance(self, performance_index: PerformanceIndex):
        if self.ttft_penalty == 0 and self.tpot_penalty == 0:
            raise StopFineTune("No penalties, no need to fine-tune.")
        ttft_flag = self.ttft_penalty != 0 and self.ttft_slo == 0
        tpot_flag = self.tpot_penalty != 0 and self.tpot_slo == 0
        if ttft_flag or tpot_flag:
            raise ValueError("Penalty is set but SLO is zero.")
        if performance_index.time_per_output_token is None:
            raise ValueError("Missing performance data for TPOT.")
        if self.ttft_penalty != 0 and performance_index.time_to_first_token is None:
            raise ValueError("Missing performance data for TTFT.")

    def direction_of_field_update(self, performance_index: PerformanceIndex):
        actual_tpot = performance_index.time_per_output_token
        actual_ttft = performance_index.time_to_first_token
        self.ttft_over_slo = False
        self.ttft_under_lower_bound = False
        self.tpot_over_slo = actual_tpot > self.tpot_upper_bound
        self.tpot_under_lower_bound = actual_tpot < self.tpot_lower_bound
        # Also constrain ttft
        if self.ttft_penalty != 0:
            self.ttft_over_slo = actual_ttft > self.ttft_upper_bound
            self.ttft_under_lower_bound = actual_ttft < self.ttft_lower_bound

    def handle_concurrency(
        self,
        simulate_run_info: Tuple[OptimizerConfigField, ...],
        performance_index: PerformanceIndex,
    ):
        was_updated_c = False
        signed_factor_c = None
        if self.tpot_over_slo:
            deviation_ratio = (performance_index.time_per_output_token - self.tpot_slo) / self.tpot_slo
            signed_factor_c = -deviation_ratio * self.step_size
        elif self.tpot_under_lower_bound:
            deviation_ratio = (self.tpot_slo - performance_index.time_per_output_token) / self.tpot_slo
            signed_factor_c = deviation_ratio * self.step_size
        # TPOT not met, adjust concurrency
        if signed_factor_c:
            self.add_history(self.last_signed_factor, CONCURRENCYS, signed_factor_c)
            signed_factor_c = max(-self.step_size, min(self.step_size, signed_factor_c))
            _concurrency_signed_factor = self.last_signed_factor.get(CONCURRENCYS)
            last_concurrency = None
            if len(_concurrency_signed_factor or []) >= 2:
                # Indicates the direction of the last metric differs from the current one, can adjust to middle
                if (
                    _concurrency_signed_factor[-2] * _concurrency_signed_factor[-1] < 0
                    and self.last_value.get(CONCURRENCYS)[-2] != self.last_value.get(CONCURRENCYS)[-1]
                ):
                    last_concurrency = self.last_value.get(CONCURRENCYS)[-2]
            was_updated_c = self.update_field(
                simulate_run_info,
                signed_factor_c,
                field_names=CONCURRENCYS,
                last=last_concurrency,
                force_min_change=self.tpot_over_slo,
            )
        return was_updated_c

    def handle_request_rate(
        self,
        simulate_run_info: Tuple[OptimizerConfigField, ...],
        performance_index: PerformanceIndex,
    ):
        # Check if ttft is met, adjust request rate
        was_updated_r = False
        signed_factor_r = None
        if self.ttft_over_slo:
            deviation_ratio = (performance_index.time_to_first_token - self.ttft_slo) / self.ttft_slo
            signed_factor_r = -deviation_ratio * self.step_size
            signed_factor_r = max(-self.step_size, min(self.step_size, signed_factor_r))
        elif self.ttft_under_lower_bound:
            signed_factor_r = self.step_size
        if signed_factor_r:
            self.add_history(self.last_signed_factor, REQUESTRATES, signed_factor_r)
            # Limit maximum update magnitude
            _request_rate_factor = self.last_signed_factor.get(REQUESTRATES)
            last_req_rate = None
            if len(_request_rate_factor or []) >= 2:
                # Indicates that the direction of the previous and current metrics is inconsistent, can adjust toward the middle
                if _request_rate_factor[-2] * _request_rate_factor[-1] < 0:
                    last_req_rate = self.last_value.get(REQUESTRATES)[-2]
            was_updated_r = self.update_field(
                simulate_run_info,
                signed_factor_r,
                field_names=REQUESTRATES,
                last=last_req_rate,
                force_min_change=self.ttft_over_slo,
            )
        return was_updated_r

    def fine_tune_with_concurrency_and_request_rate(self, params: np.ndarray, performance_index: PerformanceIndex):
        # First time gets concurrency, request rate is minimum
        self.check_config_and_performance(performance_index)
        self.direction_of_field_update(performance_index)
        simulate_run_info = map_param_with_value(params, self.target_field)
        _concurrency_flag = _request_rate_flag = False
        _concurrency_field = _request_field = None
        for _field in simulate_run_info:
            if _field.name in REQUESTRATES:
                self.add_history(self.last_value, REQUESTRATES, _field.value)
                _request_rate_flag = True
                _request_field = _field
            if _field.name in CONCURRENCYS:
                self.add_history(self.last_value, CONCURRENCYS, _field.value)
                _concurrency_flag = True
                _concurrency_field = _field
        if _concurrency_flag:
            was_updated_c = self.handle_concurrency(simulate_run_info, performance_index)
            if was_updated_c:
                # When concurrency is updated, request rate resets to the original request rate value.
                if _concurrency_field.value != self.last_value[CONCURRENCYS][-1] and _request_field:
                    _request_field.value = self.last_value[REQUESTRATES][0]
                return simulate_run_info
        if _request_rate_flag:
            # Update request rate upper and lower bounds
            for _field in simulate_run_info:
                if _field.name in REQUESTRATES:
                    if _concurrency_flag and self.last_value.get(CONCURRENCYS)[-1] < _field.max:
                        _field.max = self.last_value.get(CONCURRENCYS)[-1]
            # Check if ttft is met, adjust request rate
            was_updated_r = self.handle_request_rate(simulate_run_info, performance_index)
            if was_updated_r:
                return simulate_run_info
        raise StopFineTune("Parameter value reached its boundary or did not change.")
