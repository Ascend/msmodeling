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
# pylint: disable=too-many-lines,too-many-nested-blocks
import bisect
import json
import os
from collections.abc import Callable
from copy import deepcopy
from enum import Enum
from inspect import isfunction  # pylint: disable=no-name-in-module
from math import isclose, isinf, isnan
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
from loguru import logger
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

from ..common import is_mindie, is_vllm
from ..config.custom_command import (
    AisBenchCommandConfig,
    MindieCommandConfig,
    VllmBenchmarkCommandConfig,
    VllmCommandConfig,
)
from ..io_utils import open_file
from . import base_config
from .base_config import (
    INSTALL_PATH,
    RUN_PATH,
    ServiceType,
    ms_serviceparam_optimizer_config_path,
)

CUSTOM_OUTPUT = base_config.CUSTOM_OUTPUT
MODEL_EVAL_STATE_CONFIG_PATH = base_config.MODEL_EVAL_STATE_CONFIG_PATH


class MetricAlgorithm(BaseModel):
    metric: str = "TTFT"
    algorithm: str = "average"


class PerformanceConfig(BaseModel):
    time_to_first_token: MetricAlgorithm = MetricAlgorithm(metric="TTFT", algorithm="average")
    time_per_output_token: MetricAlgorithm = MetricAlgorithm(metric="TPOT", algorithm="average")


dtype_func = {"int": int, "float": float, "str": str}


class ErrorSeverity(Enum):
    """Error severity level"""

    FATAL = "fatal"
    RETRYABLE = "retryable"


class ErrorType(Enum):
    """Error type classification"""

    OUT_OF_MEMORY = "out_of_memory"
    DEVICE_ERROR = "device_error"
    NETWORK_ERROR = "network_error"
    IO_ERROR = "io_error"
    UNKNOWN = "unknown"


class OptimizerConfigField(BaseModel):
    name: str = "max_batch_size"
    config_position: str = "BackendConfig.ScheduleConfig.maxBatchSize"
    min: float = 0.0
    max: float = 100.0
    dtype: str = "float"
    value: Union[int, float, bool, str] = 0.0
    dtype_param: Any = None
    constant: Optional[float] = None  # Identify if the field is a constant

    @model_validator(mode="after")
    def update_constant(self):
        if self.min > self.max:
            raise ValueError(f"min({self.min}) > max({self.max}). please check")
        # If min equals max but constant is not set, auto-set constant to max value.
        if self.constant and not isclose(self.min, self.max):
            self.min = self.max = self.constant
        elif self.constant is None and isclose(self.min, self.max, rel_tol=1e-5) and self.dtype in dtype_func:
            self.constant = dtype_func.get(self.dtype, float)(self.max)

        return self

    def convert_dtype(self, value):
        if self.dtype == "str":
            return str(value)
        return dtype_func.get(self.dtype, float)(value)

    def find_available_value(self, value):
        if self.dtype == "str":
            # For string type, just return the string value
            return str(value)
        _new_value = dtype_func.get(self.dtype, float)(value)
        if self.dtype == "enum":
            enum_values = list(self.dtype_param) if isinstance(self.dtype_param, (list, tuple)) else []
            if not enum_values:
                return _new_value
            # Check if dtype_param contains string values
            if isinstance(enum_values[0], str):
                # String enum: check if value is in the enum list
                if value in enum_values:
                    return value
                # For string enum, return the first value as default
                return enum_values[0]
            # Numeric enum: use bisect
            if value in enum_values:
                return value
            _index = bisect.bisect_left(enum_values, value)
            if _index == len(enum_values):
                _new_value = enum_values[-1]
            else:
                _new_value = enum_values[_index]
            return _new_value
        if self.min <= _new_value <= self.max:
            return _new_value
        if _new_value < self.min:
            return dtype_func.get(self.dtype, float)(self.min)
        return dtype_func.get(self.dtype, float)(self.max)


default_support_field = [
    # The minimum value of max batch size must be greater than the maximum value of max_prefill_batch_size.
    OptimizerConfigField(
        name="max_batch_size",
        config_position="BackendConfig.ScheduleConfig.maxBatchSize",
        min=10,
        max=1000,
        dtype="int",
    ),
    OptimizerConfigField(
        name="max_prefill_batch_size",
        config_position="BackendConfig.ScheduleConfig.maxPrefillBatchSize",
        min=0.1,
        max=0.7,
        dtype="ratio",
        dtype_param="max_batch_size",
    ),
    OptimizerConfigField(
        name="prefill_time_ms_per_req",
        config_position="BackendConfig.ScheduleConfig.prefillTimeMsPerReq",
        max=1000,
        dtype="int",
    ),
    OptimizerConfigField(
        name="decode_time_ms_per_req",
        config_position="BackendConfig.ScheduleConfig.decodeTimeMsPerReq",
        max=1000,
        dtype="int",
    ),
    OptimizerConfigField(
        name="support_select_batch",
        config_position="BackendConfig.ScheduleConfig.supportSelectBatch",
        max=1,
        dtype="bool",
    ),
    OptimizerConfigField(
        name="max_prefill_token",
        config_position="BackendConfig.ScheduleConfig.maxPrefillTokens",
        min=4096,
        max=409600,
        dtype="int",
    ),
    OptimizerConfigField(
        name="max_queue_delay_microseconds",
        config_position="BackendConfig.ScheduleConfig.maxQueueDelayMicroseconds",
        min=500,
        max=1000000,
        dtype="int",
    ),
    OptimizerConfigField(
        name="prefill_policy_type",
        config_position="BackendConfig.ScheduleConfig.prefillPolicyType",
        min=0,
        max=1,
        dtype="enum",
        dtype_param=[0, 1, 3],
    ),
    OptimizerConfigField(
        name="decode_policy_type",
        config_position="BackendConfig.ScheduleConfig.decodePolicyType",
        min=0,
        max=1,
        dtype="enum",
        dtype_param=[0, 1, 3],
    ),
    OptimizerConfigField(
        name="max_preempt_count",
        config_position="BackendConfig.ScheduleConfig.maxPreemptCount",
        min=0,
        max=1,
        dtype="ratio",
        dtype_param="max_batch_size",
    ),
    OptimizerConfigField(
        name="tp",
        config_position="BackendConfig.ModelDeployConfig.ModelConfig.0.tp",
        min=0,
        max=1,
        dtype="enum",
        dtype_param=[1, 2, 4, 8, 16],
    ),
    OptimizerConfigField(
        name="dp",
        config_position="BackendConfig.ModelDeployConfig.ModelConfig.0.dp",
        min=0,
        max=0,
        dtype="factories",
        dtype_param={"target_name": "tp", "product": 16, "dtype": "int"},
    ),
    OptimizerConfigField(
        name="moe_ep",
        config_position="BackendConfig.ModelDeployConfig.ModelConfig.0.moe_ep",
        min=0,
        max=1,
        dtype="enum",
        dtype_param=[1, 2, 4, 8, 16],
    ),
    OptimizerConfigField(
        name="moe_tp",
        config_position="BackendConfig.ModelDeployConfig.ModelConfig.0.moe_tp",
        min=0,
        max=0,
        dtype="factories",
        dtype_param={"target_name": "moe_ep", "product": 16, "dtype": "int"},
    ),
]


def range_to_enum(params_field: tuple[OptimizerConfigField, ...]):
    for v in params_field:
        if v.dtype != "range":
            continue
        if not v.dtype_param:
            continue
        try:
            _start = int(v.min)
            _end = int(v.max)
            _step = int(v.dtype_param)
        except (ValueError, TypeError):
            logger.error(f"Failed convert to int data, data: {v.min, v.max, v.dtype_param}")
            continue
        _enums = list(range(_start, _end + _step, _step))
        v.min = 0
        v.max = 1
        v.dtype_param = _enums
        v.dtype = "enum"


class DecodeContext(BaseModel):
    """
    Particle decode context, used by the balanced strategy to evenly distribute repair priority
    across different particles and iteration rounds.

    Attributes:
        particle_index: Current particle index (0-based)
        n_particles:    Total number of particles in the population
        iteration:      Current iteration round (0-based), used by balanced strategy to alternate
                        direction between rounds, preventing the same particle from being locked
                        to the same repair order throughout the optimization process.
                        When None, falls back to pure particle index splitting.
    """

    particle_index: Optional[int] = None
    n_particles: Optional[int] = None
    iteration: Optional[int] = None


def resolve_priority(dtype_param: dict, context=None) -> list:
    """
    Resolve field priority order for repair based on priority_policy and particle context.

    Strategies:
    - fixed:    Uses the explicit order specified by dtype_param["priority"];
                falls back to target_names order if not specified
    - balanced: Splits two directions evenly by particle index, reducing structural bias
                introduced by a single decode order;
                falls back to target_names order when no context is available

    Args:
        dtype_param: The dtype_param dict from ternary_factories
        context:     DecodeContext instance, or None (non-PSO path)

    Returns:
        [High priority field name, Low priority field name]
    """
    target_names = dtype_param.get("target_names", [])
    if len(target_names) < 2:
        return list(target_names)

    policy = dtype_param.get("priority_policy", "balanced")

    if policy == "fixed":
        priority = list(dtype_param.get("priority", target_names))
        if len(priority) != len(target_names) or set(priority) != set(target_names):
            logger.warning(f"Invalid fixed priority {priority}; fallback to target_names {target_names}.")
            return list(target_names)
        return priority

    # balanced (default): first half of particles use forward order, second half use reverse order,
    # reducing decode bias. Also alternates direction between iterations to prevent the same particle
    # from being locked to the same repair order throughout the optimization process.
    if policy == "balanced":
        if context is None or context.particle_index is None or context.n_particles is None:
            return list(target_names)
        reverse = context.particle_index >= context.n_particles / 2
        # Flip direction on odd iteration rounds so each particle gets a different priority order in adjacent iterations
        if context.iteration is not None and context.iteration % 2 == 1:
            reverse = not reverse
        return list(reversed(target_names)) if reverse else list(target_names)

    return list(target_names)


def _repair_ternary_factories_with_priority(
    v, simulate_run_info, params_field, product, min_val, max_val, conv, context=None
):
    """
    Priority-aware constraint repair (new version): replaces the global nearest-distance strategy
    of _repair_ternary_factories.
    Repair is performed in two stages:
    - Stage 1 (keep high priority field): fix the current value of the keep field, search candidate values for the adjust field
    - Stage 2 (both fields adjustable): joint search using candidate values sorted by their respective distances
    Fallback behavior is compatible with the old version: if repair fails, the caller falls back to clamping.

    Args:
        v:                The OptimizerConfigField definition for the current derived field
        simulate_run_info:Mutable list of field copies (will be modified in-place)
        params_field:     Original field definition tuple (used to obtain candidate value ranges)
        product:          The product value from dtype_param
        min_val:          Result lower bound (None means unbounded)
        max_val:          Result upper bound (None means unbounded)
        conv:             Type conversion function (int / float)
        context:          DecodeContext instance, determines balanced strategy direction (degenerates when None)

    Returns:
        True  Repair succeeded, simulate_run_info has been updated in-place
        False Unable to repair, caller should fall back to clamping
    """
    target_names = v.dtype_param.get("target_names", [])
    if len(target_names) < 2:
        return False

    priority = resolve_priority(v.dtype_param, context)
    keep_name = priority[0]  # High priority: try to keep unchanged
    adjust_name = priority[1]  # Low priority: adjust first

    def_by_name = {f.name: f for f in params_field}
    sim_by_name = {f.name: f for f in simulate_run_info}

    def_keep = def_by_name.get(keep_name)
    def_adjust = def_by_name.get(adjust_name)
    if def_keep is None or def_adjust is None:
        return False

    cands_keep = _get_field_candidates(def_keep)
    cands_adjust = _get_field_candidates(def_adjust)
    if not cands_keep or not cands_adjust:
        return False

    cur_keep = sim_by_name[keep_name].value if keep_name in sim_by_name else 0
    cur_adjust = sim_by_name[adjust_name].value if adjust_name in sim_by_name else 0

    is_int_dtype = v.dtype_param.get("dtype", "int") == "int"
    cands_keep_sorted = sorted(cands_keep, key=lambda c: abs(c - (cur_keep or 0)))
    cands_adjust_sorted = sorted(cands_adjust, key=lambda c: abs(c - (cur_adjust or 0)))

    def is_valid_combination(keep_val, adjust_val):
        """Check if (keep_val, adjust_val) combination is valid, returns (ok, result)"""
        if not keep_val or not adjust_val:
            return False, None
        divisor = keep_val * adjust_val
        if divisor == 0:
            return False, None
        if is_int_dtype and product % divisor != 0:
            return False, None
        result = conv(product / divisor)
        if min_val is not None and result < min_val:
            return False, None
        if max_val is not None and result > max_val:
            return False, None
        return True, result

    def apply_result(keep_val, adjust_val, result, stage):
        old_derived = sim_by_name[v.name].value if v.name in sim_by_name else None
        sim_by_name[keep_name].value = keep_val
        sim_by_name[adjust_name].value = adjust_val
        sim_by_name[v.name].value = result
        keep_part = f"{keep_name}={keep_val}(kept)" if keep_val == cur_keep else f"{keep_name}: {cur_keep}→{keep_val}"
        adjust_part = (
            f"{adjust_name}={adjust_val}(kept)"
            if adjust_val == cur_adjust
            else f"{adjust_name}: {cur_adjust}→{adjust_val}"
        )
        derived_part = f"{v.name}: {old_derived}→{result}"
        logger.info(
            f"ternary_factories repair [{stage}] '{v.name}' "
            f"(policy={v.dtype_param.get('priority_policy', 'balanced')}): "
            f"{keep_part}, {adjust_part}, {derived_part} "
            f"(product={product})"
        )

    # Stage 1: fix the current value of the high priority field, only adjust the low priority field
    for adjust_val in cands_adjust_sorted:
        ok, result = is_valid_combination(cur_keep, adjust_val)
        if ok:
            apply_result(cur_keep, adjust_val, result, "stage1-fix-keep")
            return True

    # Stage 2: both fields adjustable, joint search using candidate values sorted by their respective distances
    for keep_val in cands_keep_sorted:
        for adjust_val in cands_adjust_sorted:
            ok, result = is_valid_combination(keep_val, adjust_val)
            if ok:
                apply_result(keep_val, adjust_val, result, "stage2-both-adjust")
                return True

    return False


# Old repair function (global normalized Manhattan distance strategy), retained for fallback.
# The main path has been replaced by _repair_ternary_factories_with_priority.
def _get_field_candidates(field_def):
    """
    Get the list of candidate discrete values for a field, used for ternary_factories constraint repair search.

    - enum type: returns the numeric candidate list from dtype_param
    - int type (range <= 256): returns the integer interval [min, max]
    - Other types or range too large: returns None, indicating enumeration is not possible (fall back to clamping)

    Args:
        field_def: OptimizerConfigField definition object
    Returns:
        List of candidate values, or None (when enumeration is not possible)
    """
    if field_def.dtype == "enum":
        params = field_def.dtype_param
        return [p for p in params if isinstance(p, (int, float))] if params else []
    if field_def.dtype == "int":
        lo, hi = int(field_def.min), int(field_def.max)
        if 0 <= hi - lo <= 256:
            return list(range(lo, hi + 1))
    return None


def _update_ratio_field(field, i, params_field, simulate_run_info, decode_context=None):
    """Ratio type handler: value = int(self_ratio × target.value)"""
    _field = simulate_run_info[i]
    _t_op = [_op for _op in simulate_run_info if _op.name == field.dtype_param][0]
    _field.value = int(_field.value * _t_op.value)


def _update_factories_field(field, i, params_field, simulate_run_info, decode_context=None):
    """Factories type handler: value = product / target.value"""
    _field = simulate_run_info[i]
    _t_op = [_op for _op in simulate_run_info if _op.name == field.dtype_param["target_name"]][0]
    if _t_op.value != 0:
        _field.value = dtype_func.get(field.dtype_param["dtype"], int)(field.dtype_param["product"] / _t_op.value)


def _update_times_field(field, i, params_field, simulate_run_info, decode_context=None):
    """Times type handler: value = product × target.value"""
    _field = simulate_run_info[i]
    _t_op = [_op for _op in simulate_run_info if _op.name == field.dtype_param["target_name"]][0]
    if _t_op.value is not None and not (isnan(_t_op.value) if isinstance(_t_op.value, float) else False):
        _field.value = dtype_func.get(field.dtype_param["dtype"], int)(field.dtype_param["product"] * _t_op.value)
    else:
        logger.warning(f"Target value for {field.name} is invalid, skipping times calculation")


def _update_ternary_factories_field(field, i, params_field, simulate_run_info, decode_context=None):
    """
    ternary_factories type handler: value = product / (field_a × field_b)

    dtype_param structure: {"target_names": ["field_a", "field_b"], "product": 16, "dtype": "int",
                          "min_value": 1,   # optional, result lower bound, int type defaults to 1
                          "max_value": 16}  # optional, result upper bound
    When result is out of bounds: first try constraint repair (adjust source fields to find nearest valid combo),
                                 fall back to clamping if repair fails.
    """
    _field = simulate_run_info[i]
    target_names = field.dtype_param.get("target_names", [])
    target_ops = [_op for _op in simulate_run_info if _op.name in target_names]
    found_names = {op.name for op in target_ops}
    missing = [n for n in target_names if n not in found_names]
    if missing:
        logger.warning(
            f"ternary_factories '{field.name}': target_names {missing} not found in fields. "
            f"Check for typos or case mismatch. Available fields: "
            f"{[op.name for op in simulate_run_info]}"
        )
        return

    divisor = 1
    for _t_op in target_ops:
        if _t_op.value != 0:
            divisor *= _t_op.value
        else:
            logger.warning(f"Target value {_t_op.name} is 0, skipping ternary_factories calculation for {field.name}")
            return

    product = field.dtype_param.get("product", 1)
    conv = dtype_func.get(field.dtype_param.get("dtype", "int"), int)
    result_value = conv(product / divisor)
    min_value = field.dtype_param.get("min_value", 1 if field.dtype_param.get("dtype", "int") == "int" else None)
    max_value = field.dtype_param.get("max_value", None)
    is_int_dtype = field.dtype_param.get("dtype", "int") == "int"
    needs_repair = (
        (min_value is not None and result_value < min_value)
        or (max_value is not None and result_value > max_value)
        or (is_int_dtype and product % divisor != 0)
    )
    if needs_repair:
        if not _repair_ternary_factories_with_priority(
            field,
            simulate_run_info,
            params_field,
            product,
            min_value,
            max_value,
            conv,
            context=decode_context,
        ):
            repaired = False
            if min_value is not None and result_value < min_value:
                logger.warning(
                    f"ternary_factories priority repair failed for '{field.name}'; "
                    f"fallback to clamp: {result_value} → min_value {min_value}."
                )
                _field.value = conv(min_value)
                repaired = True
            if max_value is not None and result_value > max_value:
                logger.warning(
                    f"ternary_factories priority repair failed for '{field.name}'; "
                    f"fallback to clamp: {result_value} → max_value {max_value}."
                )
                _field.value = conv(max_value)
                repaired = True
            if not repaired:
                if is_int_dtype and product % divisor != 0:
                    raise ValueError(
                        f"ternary_factories constraint violated for '{field.name}': "
                        f"product={product} not divisible by divisor={divisor} "
                        f"(targets={target_names}), and repair could not find valid source values."
                    )
                _field.value = result_value
    else:
        _field.value = result_value


def _update_ternary_times_field(field, i, params_field, simulate_run_info, decode_context=None):
    """
    ternary_times type handler: value = product × field_a × field_b

    dtype_param structure: {"target_names": ["field_a", "field_b"], "product": 1, "dtype": "int"}
    """
    _field = simulate_run_info[i]
    target_names = field.dtype_param.get("target_names", [])
    target_ops = [_op for _op in simulate_run_info if _op.name in target_names]
    found_names = {op.name for op in target_ops}
    missing = [n for n in target_names if n not in found_names]
    if missing:
        logger.warning(
            f"ternary_times '{field.name}': target_names {missing} not found in fields. "
            f"Check for typos or case mismatch. Available fields: "
            f"{[op.name for op in simulate_run_info]}"
        )
        return
    result = field.dtype_param.get("product", 1)
    for _t_op in target_ops:
        if _t_op.value is not None and not (isnan(_t_op.value) if isinstance(_t_op.value, float) else False):
            result *= _t_op.value
        else:
            logger.warning(f"Target value {_t_op.name} for {field.name} is invalid, skipping ternary_times calculation")
            return
    _field.value = dtype_func.get(field.dtype_param.get("dtype", "int"), int)(result)


def _update_share_field(field, i, params_field, simulate_run_info, decode_context=None):
    """Share type handler: value = int(target.min + target.max - target.value)"""
    _field = simulate_run_info[i]
    for _op in simulate_run_info:
        if _op.name == field.dtype_param:
            _field.value = int(_op.min + _op.max - _op.value)
            break


DERIVED_FIELD_HANDLERS = {
    "ratio": _update_ratio_field,
    "share": _update_share_field,
    "factories": _update_factories_field,
    "times": _update_times_field,
    "ternary_factories": _update_ternary_factories_field,
    "ternary_times": _update_ternary_times_field,
}


def update_optimizer_value(
    params_field: tuple[OptimizerConfigField, ...],
    simulate_run_info: tuple[OptimizerConfigField, ...],
    support_select_is_false,
    decode_context: Optional["DecodeContext"] = None,
):
    """
    Post-process and assign derived field values in simulate_run_info based on inter-field dependencies.

    This function handles fields with the following derived dtypes (these fields typically have min=max and are marked
    as constants, with their values derived by this function):

    Binary relations (depend on a single field)
    -------------------------------------------
    - ``ratio``            : value = int(self_ratio × target.value)
    - ``factories``        : value = product / target.value  (skipped when target.value is 0)
    - ``times``            : value = product × target.value  (skipped when target.value is None/NaN)

    Ternary relations (depend on two fields)
    -----------------------------------------
    - ``ternary_factories``: value = product / (field_a.value × field_b.value)
                             Skipped with warning when any dependent field value is 0.
                             dtype_param format: {"target_names": [str, str], "product": number, "dtype": str}
    - ``ternary_times``    : value = product × field_a.value × field_b.value
                             Skipped with warning when any dependent field value is None or NaN.
                             dtype_param format: {"target_names": [str, str], "product": number, "dtype": str}

    Also handles the following business constraints:
    - maxPrefillBatchSize field value is forced to 1 when it is 0.
    - When support_select_is_false is True, prefillTimeMsPerReq / decodeTimeMsPerReq are forced to 0.

    Args:
        params_field:          Original field definition tuple, used to determine each field's dtype and dtype_param.
        simulate_run_info:     Deep-copied list of the same length as params_field; values will be modified in-place.
        support_select_is_false: Pass True when supportSelectBatch field value is False,
                                 triggering prefill/decode time field zeroing logic.
    """
    for i, v in enumerate(params_field):
        handler = DERIVED_FIELD_HANDLERS.get(v.dtype)
        if handler:
            handler(v, i, params_field, simulate_run_info, decode_context)

        # cross-dtype post-processing
        if "maxPrefillBatchSize" in v.config_position:
            _field = simulate_run_info[i]
            if _field.value == 0:
                _field.value = 1
        if support_select_is_false:
            _field = simulate_run_info[i]
            if "prefillTimeMsPerReq" in _field.config_position:
                _field.value = 0
            if "decodeTimeMsPerReq" in _field.config_position:
                _field.value = 0


def map_param_with_value(
    params: np.ndarray,
    params_field: tuple[OptimizerConfigField, ...],
    decode_context: Optional["DecodeContext"] = None,
):
    _simulate_run_info = []
    _support_select_is_false = False
    i = 0
    for v in params_field:
        _field = deepcopy(v)
        if _field.constant is not None or isclose(_field.min, _field.max, rel_tol=1e-5):
            if _field.value and not isinf(_field.value):
                try:
                    _field.value = dtype_func.get(v.dtype, int)(_field.value)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Failed in func {params[i]} for {v}, error: {e}")
            _simulate_run_info.append(_field)
            continue
        if v.dtype == "int":
            try:
                _field.value = int(params[i])
            except (ValueError, TypeError):
                logger.warning(f"Failed convert to int data, data: {params[i]}")
                _field.value = params[i]
        elif v.dtype == "bool":
            if params[i] > 0.5:
                _field.value = True
                if "supportSelectBatch" in _field.name:
                    _support_select_is_false = True
            else:
                _field.value = False
        elif v.dtype == "enum":
            # Check if dtype_param contains string values
            if v.dtype_param and len(v.dtype_param) > 0 and isinstance(v.dtype_param[0], str):
                # String enum: use simple indexing based on value position
                num_options = len(v.dtype_param)
                # Map param value to enum index
                if num_options == 1:
                    _field.value = v.dtype_param[0]
                else:
                    # Normalize param to [0, 1] range then scale to enum index
                    normalized = (params[i] - v.min) / (v.max - v.min) if v.max > v.min else 0
                    _enum_index = int(normalized * (num_options - 1) + 0.5)
                    _enum_index = max(0, min(_enum_index, num_options - 1))
                    _field.value = v.dtype_param[_enum_index]
            else:
                # Numeric enum: use existing logic with linspace
                segment = np.linspace(v.min, v.max, len(v.dtype_param) + 1)
                if params[i] <= v.min:
                    _field.value = v.dtype_param[0]
                elif params[i] >= v.max:
                    _field.value = v.dtype_param[-1]
                else:
                    _enum_index = np.searchsorted(segment, params[i]) - 1
                    _field.value = v.dtype_param[_enum_index]
        else:
            try:
                _field.value = float(params[i])
            except (ValueError, TypeError):
                logger.warning(f"Failed convert to float data, data: {params[i]}")
                _field.value = params[i]
        i += 1
        _simulate_run_info.append(_field)
    update_optimizer_value(
        params_field,
        tuple(_simulate_run_info),
        _support_select_is_false,
        decode_context,
    )
    return _simulate_run_info


def reverse_special_field(params_field: tuple[OptimizerConfigField, ...], params: np.ndarray, concurrency: int):
    _params = params
    i = 0
    for v in params_field:
        if v.constant is not None or isclose(v.min, v.max, rel_tol=1e-5):
            continue
        if v.dtype == "ratio":
            for _op in params_field:
                if _op.name == v.dtype_param and _op.value != 0:
                    _t_op = _op
                    _params[i] = float(v.value / _t_op.value)
        if v.name in ["CONCURRENCY", "MAXCONCURRENCY"]:
            if v.value == 0 and v.dtype == "ratio":
                _params[i] = 1
            elif v.value is not None and v.dtype == "ratio" and concurrency > 0:
                _params[i] = v.value / concurrency
            elif v.value is not None:
                _params[i] = v.value
            else:
                _params[i] = concurrency
        i += 1
    return _params


def field_to_param(params_field: tuple[OptimizerConfigField, ...]):
    concurrency = None
    _params = []
    for _, v in enumerate(params_field):
        if v.constant is not None or isclose(v.min, v.max, rel_tol=1e-5):
            continue
        if v.dtype == "int":
            try:
                _params.append(int(v.value))
            except Exception as e:
                logger.warning(f"Failed in field to param, error: {e}")
                _params.append(v.value)
        elif v.dtype == "bool":
            if v.value:
                _params.append(1)
            else:
                _params.append(0)
        elif v.dtype == "enum":
            if v.value not in v.dtype_param and isinstance(v.value, str):
                v.dtype_param.append(v.value)
            if v.value not in v.dtype_param and isinstance(v.value, (int, float)):
                v.dtype_param.sort()
                bisect.insort_left(v.dtype_param, v.value)
            _index = v.dtype_param.index(v.value)
            segment = np.linspace(v.min, v.max, len(v.dtype_param) + 1)
            _params.append((segment[_index] + segment[_index + 1]) / 2)
        else:
            _params.append(v.value)
        if v.config_position == "BackendConfig.ScheduleConfig.maxBatchSize" or v.name in [
            "MAX_NUM_SEQS",
            "max_batch_size",
        ]:
            concurrency = v.value
    _params = np.array(_params, dtype=float)
    return reverse_special_field(params_field, _params, concurrency)


class PerformanceIndex(BaseModel):
    generate_speed: Optional[float] = None
    time_to_first_token: Optional[float] = None
    time_per_output_token: Optional[float] = None
    success_rate: Optional[float] = None
    throughput: Optional[float] = None


class DataStorageConfig(BaseModel):
    store_dir: Path = Path("store")
    pso_top_k: int = 3

    @field_validator("store_dir")
    @classmethod
    def create_path(cls, path: Path) -> Path:
        path.mkdir(parents=True, exist_ok=True, mode=0o750)
        return path


class LatencyModel(BaseModel):
    base_path: Path = Path("latency_model")
    model_path: Optional[Path] = Field(
        default_factory=lambda data: data["base_path"].joinpath("bak/base/xgb_model.ubj").resolve()
    )
    static_file_dir: Optional[Path] = Field(
        default_factory=lambda data: data["base_path"].joinpath("model_static_file").resolve(),
        validate_default=True,
    )
    req_and_decode_file: Optional[Path] = Field(
        default_factory=lambda data: data["base_path"].joinpath("req_id_and_decode_num.json").resolve()
    )
    cache_data: Optional[Path] = Field(default_factory=lambda data: data["base_path"].joinpath("cache").resolve())

    @field_validator("base_path", "cache_data", "static_file_dir")
    @classmethod
    def create_path(cls, path: Path) -> Path:
        path.mkdir(parents=True, exist_ok=True)
        return path


def _get_mindie_config_paths():
    """Get mindie configuration file paths"""
    default_config_path = Path("/usr/local/Ascend/mindie/latest/mindie-service/conf/config.json")
    default_config_bak_path = Path("/usr/local/Ascend/mindie/latest/mindie-service/conf/config_bak.json")

    if not default_config_path.is_file():
        mies_install_path = os.getenv("MIES_INSTALL_PATH")
        if mies_install_path:
            new_config_path_parent = Path(mies_install_path).parent
            return (
                new_config_path_parent / "mindie_llm/conf/config.json",
                new_config_path_parent / "mindie_llm/conf/config_bak.json",
            )
    return default_config_path, default_config_bak_path


class MindieConfig(BaseModel):
    process_name: str = "mindie, mindie-llm, mindieservice_daemon, mindie_llm"
    output: Path = Path("mindie")
    work_path: Path = Field(default_factory=lambda: Path(os.getcwd()).resolve())
    config_path: Path = Field(default_factory=lambda: _get_mindie_config_paths()[0])
    config_bak_path: Path = Field(default_factory=lambda: _get_mindie_config_paths()[1])
    command: MindieCommandConfig = MindieCommandConfig()
    target_field: list[OptimizerConfigField] = default_support_field


class AisBenchConfig(BaseModel):
    process_name: str = "ais_bench"
    output_path: Path = Path("ais_bench")
    work_path: Path = Field(default_factory=lambda: Path(os.getcwd()).resolve())
    command: AisBenchCommandConfig = AisBenchCommandConfig()
    performance_config: PerformanceConfig = PerformanceConfig()
    target_field: list[OptimizerConfigField] = Field(default_factory=list)
    model: str = ""
    path: str = ""
    host_ip: str = ""
    host_port: int = 0
    max_out_len: int = 0
    best_concurrency_coefficient: int = 3
    best_concurrency_threshold: int = 200


class VllmBenchmarkConfig(BaseModel):
    output_path: Path = Path("vllm")
    process_name: str = ""
    command: VllmBenchmarkCommandConfig = VllmBenchmarkCommandConfig()
    performance_config: PerformanceConfig = PerformanceConfig()
    target_field: list[OptimizerConfigField] = Field(default_factory=list)


class VllmConfig(BaseModel):
    output: Path = Path("vllm")
    process_name: str = "vllm"
    work_path: Path = Field(default_factory=lambda: Path(os.getcwd()).resolve())
    command: VllmCommandConfig = VllmCommandConfig()
    target_field: list[OptimizerConfigField] = Field(default_factory=list)


class PsoOptions(BaseModel):
    c1: float = 2.0
    c2: float = 2.0
    w: float = 1.8


class PsoStrategy(BaseModel):
    w: str = "exp_decay"
    c1: str = "exp_decay"
    c2: str = "exp_decay"


class ErrorPatternConfig(BaseModel):
    """Error pattern configuration - 3-tier design: ErrorType -> patterns -> severity"""

    fatal_patterns: dict[ErrorType, list[str]] = Field(
        default_factory=lambda: {
            ErrorType.OUT_OF_MEMORY: [],
            ErrorType.DEVICE_ERROR: [],
        }
    )
    retryable_patterns: dict[ErrorType, list[str]] = Field(
        default_factory=lambda: {ErrorType.NETWORK_ERROR: [], ErrorType.IO_ERROR: []}
    )


class HealthCheckConfig(BaseModel):
    """Health check configuration"""

    service_errors: ErrorPatternConfig = Field(default_factory=ErrorPatternConfig)
    benchmark_errors: ErrorPatternConfig = Field(
        default_factory=lambda: ErrorPatternConfig(
            fatal_patterns={},
            retryable_patterns={ErrorType.NETWORK_ERROR: [], ErrorType.IO_ERROR: []},
        )
    )
    log_snippet_length: int = 50


class Settings(BaseSettings):
    """
    Settings class definition, initialized by reading configuration files
    """

    model_config = SettingsConfigDict(
        toml_file=[
            INSTALL_PATH.joinpath("model_eval_state.toml"),
            Path("~/model_eval_state.toml").expanduser(),
            RUN_PATH.joinpath("model_eval_state.toml"),
            INSTALL_PATH.joinpath("config.toml"),
            INSTALL_PATH.joinpath("optix/config.toml"),
            Path("~/config.toml").expanduser(),
            RUN_PATH.joinpath("config.toml"),
            ms_serviceparam_optimizer_config_path,
        ],
        env_prefix="model_eval_state_",
    )

    output: Path = Field(
        default_factory=lambda: Path(os.getcwd()).joinpath("result").resolve(),
        validate_default=True,
    )
    simulator_output: Path = Field(default_factory=lambda data: data["output"].joinpath("simulator").resolve())
    pso_options: PsoOptions = PsoOptions()
    pso_strategy: PsoStrategy = PsoStrategy()
    particles_time_out: int = 1 * 60 * 60
    wait_start_time: int = 1800
    n_particles: int = Field(default=5, gt=0, lt=1000)
    iters: int = Field(default=10, gt=0, lt=1000)
    ftol: float = -np.inf
    ftol_iter: int = 1
    ttft_penalty: float = 3.0
    tpot_penalty: float = 3.0
    success_rate_penalty: float = 5.0
    ttft_slo: float = Field(default=0.5, gt=0)
    tpot_slo: float = Field(default=0.05, gt=0)
    success_rate_slo: float = Field(default=1.0, gt=0)
    slo_coefficient: float = 0.1
    generate_speed_target: float = 5000.0
    mem_coefficient: float = 0.8
    max_fine_tune: int = 10
    use_request_rate_calibration: bool = True
    scaling_coefficient: float = 1.3
    step_size: float = 0.6
    theory_guided_enable: bool = True
    service: str = ServiceType.master.value
    latency_model: LatencyModel = Field(
        default_factory=lambda data: LatencyModel(base_path=data["output"].joinpath("latency_model")),
        validate_default=True,
    )
    vllm: VllmConfig = Field(
        default_factory=lambda data: VllmConfig(output=data["output"].joinpath("vllm")),
        validate_default=True,
    )
    mindie: MindieConfig = Field(
        default_factory=lambda data: MindieConfig(output=data["output"].joinpath("mindie")),
        validate_default=True,
    )
    ais_bench: AisBenchConfig = AisBenchConfig()

    vllm_benchmark: VllmBenchmarkConfig = VllmBenchmarkConfig()

    data_storage: DataStorageConfig = Field(
        default_factory=lambda data: DataStorageConfig(store_dir=data["output"].joinpath("store")),
        validate_default=True,
    )

    health_check: HealthCheckConfig = Field(default_factory=HealthCheckConfig)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            TomlConfigSettingsSource(settings_cls),
            file_secret_settings,
        )

    @field_validator("output", "simulator_output")
    @classmethod
    def create_path(cls, path: Path) -> Path:
        path.mkdir(parents=True, exist_ok=True, mode=0o750)
        return path

    @model_validator(mode="after")
    def normalize_benchmark_paths(self):
        ais_output = AisBenchConfig.model_fields["output_path"].default
        if self.ais_bench.output_path == ais_output:
            self.ais_bench.output_path = self.output.joinpath(ais_output)
        if not self.ais_bench.command.work_dir:
            self.ais_bench.command.work_dir = str(self.ais_bench.output_path)

        vllm_bench_output = VllmBenchmarkConfig.model_fields["output_path"].default
        if self.vllm_benchmark.output_path == vllm_bench_output:
            self.vllm_benchmark.output_path = self.output.joinpath(vllm_bench_output)
        result_dir = self.vllm_benchmark.command.result_dir.strip()
        if not result_dir or Path(result_dir).resolve() == Path.cwd().resolve():
            self.vllm_benchmark.command.result_dir = str(self.vllm_benchmark.output_path.joinpath("result"))
        Path(self.vllm_benchmark.command.result_dir).mkdir(parents=True, exist_ok=True, mode=0o750)
        return self

    @model_validator(mode="after")
    def partial_update_vllm(self):
        if not is_vllm():
            return self
        output = VllmConfig.model_fields["output"].default
        if self.vllm.output == output:
            self.vllm.output = self.output.joinpath(output)
        self.vllm_benchmark.command.host = self.vllm.command.host
        self.vllm_benchmark.command.port = self.vllm.command.port
        self.vllm_benchmark.command.model = self.vllm.command.model
        self.vllm_benchmark.command.served_model_name = self.vllm.command.served_model_name
        if self.vllm.target_field:
            range_to_enum(self.vllm.target_field)
        if self.vllm_benchmark.target_field:
            range_to_enum(self.vllm_benchmark.target_field)
        return self

    @model_validator(mode="after")
    def partial_update_aisbench(self):
        if self.ais_bench.target_field:
            range_to_enum(self.ais_bench.target_field)
        return self

    @model_validator(mode="after")
    def partial_update_mindie(self):
        if self.data_storage.store_dir == DataStorageConfig.model_fields["store_dir"].default:
            self.data_storage.store_dir = self.output.joinpath("store")
        range_to_enum(self.mindie.target_field)
        if not is_mindie():
            return self
        if not self.mindie.config_path.exists():
            logger.error(f"File Not Found. file: {self.mindie.config_path!r}")
            return self
        with open_file(self.mindie.config_path, "r") as f:
            try:
                json.load(f)
            except json.decoder.JSONDecodeError as e:
                logger.error(f"Failed in load {self.mindie.config_path!r}. error: {e}")
                raise e
        output = MindieConfig.model_fields["output"].default
        if self.mindie.output == output:
            self.mindie.output = self.output.joinpath(output)
        return self


custom_settings_func: Optional[Callable] = None

settings = None


def get_settings() -> Settings:
    """
    Get the settings object
    Return: Settings() instance
    """
    global settings
    if not settings:
        if custom_settings_func and isfunction(custom_settings_func):
            settings = custom_settings_func()
        else:
            settings = Settings()
    return settings


def register_settings(func: Optional[Callable] = None) -> None:
    """
    Register custom settings - can provide a function to generate or provide new settings
    """
    global custom_settings_func
    custom_settings_func = func
