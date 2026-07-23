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
# pylint: disable=too-many-lines,duplicate-code,protected-access
import os
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from optix.config.config import (
    DecodeContext,
    ErrorPatternConfig,
    ErrorType,
    HealthCheckConfig,
    MindieConfig,
    OptimizerConfigField,
    _get_mindie_config_paths,
    _repair_ternary_factories_with_priority,
    _update_factories_field,
    _update_times_field,
    field_to_param,
    get_settings,
    map_param_with_value,
    register_settings,
    resolve_priority,
    reverse_special_field,
    Settings,
    update_optimizer_value,
)

DEFAULT_MINDIE_CONFIG = Path("/usr/local/Ascend/mindie/latest/mindie-service/conf/config.json")
DEFAULT_MINDIE_BACKUP = Path("/usr/local/Ascend/mindie/latest/mindie-service/conf/config_bak.json")


def field(
    name,
    dtype,
    min_=0,
    max_=0,
    value=0,
    dtype_param=None,
    config_position=None,
    constant=None,
):
    return OptimizerConfigField(
        name=name,
        config_position=config_position or f"Test.{name}",
        min=min_,
        max=max_,
        dtype=dtype,
        value=value,
        dtype_param=dtype_param,
        constant=constant,
    )


def schedule_fields():
    return [
        field(
            "max_batch_size",
            "int",
            25,
            300,
            config_position="BackendConfig.ScheduleConfig.maxBatchSize",
        ),
        field(
            "max_prefill_batch_size",
            "int",
            1,
            25,
            config_position="BackendConfig.ScheduleConfig.maxPrefillBatchSize",
        ),
        field(
            "prefill_time_ms_per_req",
            "int",
            0,
            1000,
            config_position="BackendConfig.ScheduleConfig.prefillTimeMsPerReq",
        ),
        field(
            "decode_time_ms_per_req",
            "int",
            0,
            1000,
            config_position="BackendConfig.ScheduleConfig.decodeTimeMsPerReq",
        ),
        field(
            "support_select_batch",
            "bool",
            0,
            1,
            config_position="BackendConfig.ScheduleConfig.supportSelectBatch",
        ),
        field(
            "max_prefill_token",
            "int",
            4096,
            409600,
            config_position="BackendConfig.ScheduleConfig.maxPrefillTokens",
        ),
        field(
            "max_queue_delay_microseconds",
            "int",
            500,
            1000000,
            config_position="BackendConfig.ScheduleConfig.maxQueueDelayMicroseconds",
        ),
        field(
            "prefill_policy_type",
            "enum",
            0,
            1,
            dtype_param=[0, 1, 3],
            config_position="BackendConfig.ScheduleConfig.prefillPolicyType",
        ),
        field(
            "decode_policy_type",
            "enum",
            0,
            1,
            dtype_param=[0, 1, 3],
            config_position="BackendConfig.ScheduleConfig.decodePolicyType",
        ),
        field(
            "max_preempt_count",
            "ratio",
            0,
            1,
            dtype_param="max_batch_size",
            config_position="BackendConfig.ScheduleConfig.maxPreemptCount",
        ),
    ]


def pd_share_fields():
    return [
        field("default_p_rate", "int", 1, 3, 1, config_position="default_p_rate"),
        field(
            "default_d_rate",
            "share",
            1,
            3,
            dtype_param="default_p_rate",
            config_position="default_d_rate",
        ),
    ]


def clone_with_values(fields, values):
    cloned = [deepcopy(item) for item in fields]
    for item, value in zip(cloned, values):
        item.value = value
    return cloned


def derive(fields, values, support_select_is_false=False, context=None):
    runtime_fields = clone_with_values(fields, values)
    update_optimizer_value(tuple(fields), tuple(runtime_fields), support_select_is_false, context)
    return runtime_fields


def pair_fields(product=32, policy="balanced", priority=None, tp_candidates=None, pp_candidates=None):
    dtype_param = {
        "target_names": ["tp", "pp"],
        "product": product,
        "dtype": "int",
        "priority_policy": policy,
    }
    if priority is not None:
        dtype_param["priority"] = priority
    return (
        field("tp", "enum", 0, 1, dtype_param=tp_candidates or [1, 2, 4, 8]),
        field("pp", "enum", 0, 1, dtype_param=pp_candidates or [1, 2, 4]),
        field("dp", "ternary_factories", 0, 0, dtype_param=dtype_param),
    )


@pytest.mark.parametrize(
    "params,fields,expected",
    [
        (
            np.array([26.7, 12.3, 999.9, 500.0, 0.6, 40960.0, 750000.0]),
            schedule_fields()[:7],
            [26, 12, 999, 500, True, 40960, 750000],
        ),
        (
            np.array([24.9, 0.0, 0.0, 0.0, 0.4, 4095.9, 499.9, -1.0, 2.0, 1.1]),
            schedule_fields(),
            [24, 1, 0, 0, False, 4095, 499, 0],
        ),
    ],
)
def test_map_param_converts_schedule_fields(params, fields, expected):
    result = map_param_with_value(params, fields)

    assert [item.value for item in result[: len(expected)]] == expected


def test_map_param_selects_numeric_enum_segments():
    result = map_param_with_value(np.array([0.0, 0.3, 0.6, 1.0]), schedule_fields()[7:9])

    assert [item.value for item in result] == [0, 0]


def test_ratio_field_uses_resolved_target_value():
    max_batch_size = field(
        "max_batch_size",
        "int",
        value=100,
        constant=100,
        config_position="BackendConfig.ScheduleConfig.maxBatchSize",
    )
    ratio = schedule_fields()[9]

    result = map_param_with_value(np.array([0.5]), [max_batch_size, ratio])

    assert result[1].value == 50


def test_share_field_keeps_complementary_rate():
    assert map_param_with_value(np.array([1, 2]), pd_share_fields())[1].value == 3


def test_error_pattern_config_accepts_custom_and_empty_sets():
    custom = ErrorPatternConfig(
        fatal_patterns={ErrorType.OUT_OF_MEMORY: ["custom OOM pattern"]},
        retryable_patterns={ErrorType.NETWORK_ERROR: ["custom network pattern"]},
    )
    empty = ErrorPatternConfig(fatal_patterns={}, retryable_patterns={})

    assert custom.fatal_patterns[ErrorType.OUT_OF_MEMORY] == ["custom OOM pattern"]
    assert custom.retryable_patterns[ErrorType.NETWORK_ERROR] == ["custom network pattern"]
    assert empty.fatal_patterns == {}
    assert empty.retryable_patterns == {}


def test_health_check_config_defaults_and_overrides():
    default = HealthCheckConfig()
    custom = HealthCheckConfig(
        service_errors=ErrorPatternConfig(
            fatal_patterns={ErrorType.DEVICE_ERROR: ["device fault"]},
            retryable_patterns={},
        ),
        benchmark_errors=ErrorPatternConfig(fatal_patterns={}, retryable_patterns={ErrorType.IO_ERROR: ["disk full"]}),
        log_snippet_length=300,
    )

    assert isinstance(default.service_errors, ErrorPatternConfig)
    assert ErrorType.OUT_OF_MEMORY in default.service_errors.fatal_patterns
    assert ErrorType.NETWORK_ERROR in default.service_errors.retryable_patterns
    assert default.benchmark_errors.fatal_patterns == {}
    assert ErrorType.IO_ERROR in default.benchmark_errors.retryable_patterns
    assert HealthCheckConfig(log_snippet_length=500).log_snippet_length == 500
    assert custom.service_errors.fatal_patterns[ErrorType.DEVICE_ERROR] == ["device fault"]
    assert custom.benchmark_errors.retryable_patterns[ErrorType.IO_ERROR] == ["disk full"]
    assert custom.log_snippet_length == 300


@patch.object(Path, "is_file")
def test_mindie_config_paths_use_default_when_available(mock_is_file):
    mock_is_file.return_value = True

    assert _get_mindie_config_paths() == (DEFAULT_MINDIE_CONFIG, DEFAULT_MINDIE_BACKUP)


@patch.object(Path, "is_file")
def test_mindie_config_paths_fallback_to_default_without_env(mock_is_file, monkeypatch):
    mock_is_file.return_value = False
    monkeypatch.delenv("MIES_INSTALL_PATH", raising=False)

    assert _get_mindie_config_paths() == (DEFAULT_MINDIE_CONFIG, DEFAULT_MINDIE_BACKUP)


@patch("optix.config.config._get_mindie_config_paths")
def test_mindie_config_defaults_are_bound_from_path_resolver(mock_get_paths):
    mock_get_paths.return_value = (
        Path("/test/config.json"),
        Path("/test/config_bak.json"),
    )

    config = MindieConfig()

    assert config.process_name == "mindie, mindie-llm, mindieservice_daemon, mindie_llm"
    assert config.output == Path("mindie")
    assert config.config_path == Path("/test/config.json")
    assert config.config_bak_path == Path("/test/config_bak.json")
    assert isinstance(config.target_field, list)
    assert config.target_field


@patch("optix.config.config._get_mindie_config_paths")
def test_mindie_config_allows_custom_output(mock_get_paths):
    mock_get_paths.return_value = (
        Path("/test/config.json"),
        Path("/test/config_bak.json"),
    )

    assert MindieConfig(output=Path("/custom/output")).output == Path("/custom/output")


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        (
            {"min_": 100, "max_": 100, "dtype": "int"},
            {"constant": 100, "min": 100, "max": 100},
        ),
        (
            {"min_": 0, "max_": 100, "dtype": "int", "constant": 50},
            {"constant": 50, "min": 50, "max": 50},
        ),
    ],
)
def test_optimizer_field_constant_normalization(kwargs, expected):
    item = field("test_field", config_position="test.position", **kwargs)

    assert {"constant": item.constant, "min": item.min, "max": item.max} == expected


def test_optimizer_field_rejects_invalid_bounds():
    with pytest.raises(ValueError, match="min.*max"):
        field("test_field", "int", 100, 0, config_position="test.position")


@pytest.mark.parametrize(
    "item,value,expected",
    [
        (field("bounded", "int", 0, 100), 50, 50),
        (field("lower", "int", 0, 100), -10, 0),
        (field("upper", "int", 0, 100), 150, 100),
        (field("enum_exact", "enum", 0, 1, dtype_param=[1, 2, 4, 8]), 2, 2),
        (field("enum_next", "enum", 0, 1, dtype_param=[1, 2, 4, 8]), 3, 4),
        (field("enum_floor", "enum", 0, 1, dtype_param=[1, 2, 4, 8]), 0, 1),
    ],
)
def test_find_available_value_uses_bounds_or_enum_candidates(item, value, expected):
    assert item.find_available_value(value) == expected


def test_convert_dtype_uses_field_dtype():
    assert field("int_field", "int", config_position="test.position").convert_dtype("42") == 42
    assert field("float_field", "float", config_position="test.position").convert_dtype("3.14") == pytest.approx(3.14)


@pytest.mark.parametrize(
    "fields,values,index,expected",
    [
        (
            [
                field("tp", "int", 1, 8),
                field("pp", "int", 1, 4),
                field(
                    "dp",
                    "ternary_factories",
                    dtype_param={
                        "target_names": ["tp", "pp"],
                        "product": 16,
                        "dtype": "int",
                    },
                ),
            ],
            [2, 4, 0],
            2,
            2,
        ),
        (
            [
                field("tp", "int", 1, 8),
                field("pp", "int", 1, 4),
                field(
                    "dp_f",
                    "ternary_factories",
                    dtype_param={
                        "target_names": ["tp", "pp"],
                        "product": 10.0,
                        "dtype": "float",
                    },
                ),
            ],
            [2, 2, 0.0],
            2,
            2.5,
        ),
        (
            [
                field("tp", "int", 1, 8),
                field("pp", "int", 1, 4),
                field(
                    "dp",
                    "ternary_factories",
                    dtype_param={"target_names": ["tp", "pp"], "dtype": "int"},
                ),
            ],
            [2, 1, 0],
            2,
            1,
        ),
        (
            [
                field("tp", "int", 1, 8),
                field("pp", "int", 1, 4),
                field(
                    "dp",
                    "ternary_factories",
                    dtype_param={
                        "target_names": ["tp", "pp"],
                        "product": 16,
                        "dtype": "int",
                        "min_value": 1,
                    },
                ),
            ],
            [8, 4, 99],
            2,
            1,
        ),
        (
            [
                field("tp", "int", 1, 8),
                field("pp", "int", 1, 4),
                field(
                    "dp",
                    "ternary_factories",
                    value=99,
                    dtype_param={
                        "target_names": ["tp", "pp"],
                        "product": 16,
                        "dtype": "int",
                    },
                ),
            ],
            [8, 4, 99],
            2,
            1,
        ),
        (
            [
                field("tp", "int", 1, 8),
                field("pp", "int", 1, 4),
                field(
                    "dp",
                    "ternary_factories",
                    dtype_param={
                        "target_names": ["tp", "pp"],
                        "product": 64,
                        "dtype": "int",
                        "max_value": 8,
                    },
                ),
            ],
            [1, 1, 0],
            2,
            8,
        ),
        (
            [
                field("tp", "int", 1, 8),
                field("pp", "int", 1, 4),
                field(
                    "dp",
                    "ternary_factories",
                    dtype_param={
                        "target_names": ["tp", "pp"],
                        "product": 8,
                        "dtype": "int",
                        "min_value": 1,
                        "max_value": 3,
                    },
                ),
            ],
            [2, 2, 0],
            2,
            2,
        ),
        (
            [
                field("seq_len", "int", 128, 4096),
                field("batch_size", "int", 1, 64),
                field(
                    "total_tokens",
                    "ternary_times",
                    dtype_param={
                        "target_names": ["seq_len", "batch_size"],
                        "product": 2,
                        "dtype": "int",
                    },
                ),
            ],
            [512, 4, 0],
            2,
            4096,
        ),
        (
            [
                field("a", "int", 1, 10),
                field("b", "int", 1, 10),
                field(
                    "c",
                    "ternary_times",
                    dtype_param={
                        "target_names": ["a", "b"],
                        "product": 1,
                        "dtype": "int",
                    },
                ),
            ],
            [3, 7, 0],
            2,
            21,
        ),
        (
            [
                field("a", "int", 1, 10),
                field("b", "int", 1, 10),
                field(
                    "c",
                    "ternary_times",
                    dtype_param={"target_names": ["a", "b"], "dtype": "int"},
                ),
            ],
            [3, 5, 0],
            2,
            15,
        ),
    ],
)
def test_ternary_derived_fields_update_value(fields, values, index, expected):
    assert derive(fields, values)[index].value == expected


@pytest.mark.parametrize(
    "fields,values,index,original",
    [
        (
            [
                field("tp", "int", 1, 8),
                field("pp", "int", 1, 4),
                field(
                    "dp",
                    "ternary_factories",
                    value=99,
                    dtype_param={
                        "target_names": ["tp", "pp"],
                        "product": 16,
                        "dtype": "int",
                    },
                ),
            ],
            [0, 4, 99],
            2,
            99,
        ),
        (
            [
                field("tp", "int", 1, 8),
                field("pp", "int", 1, 4),
                field(
                    "dp",
                    "ternary_factories",
                    value=88,
                    dtype_param={
                        "target_names": ["tp", "pp"],
                        "product": 16,
                        "dtype": "int",
                    },
                ),
            ],
            [4, 0, 88],
            2,
            88,
        ),
        (
            [
                field("a", "float", 1.0, 10.0, value=float("nan")),
                field("b", "int", 1, 10),
                field(
                    "c",
                    "ternary_times",
                    value=999,
                    dtype_param={
                        "target_names": ["a", "b"],
                        "product": 2,
                        "dtype": "int",
                    },
                ),
            ],
            [float("nan"), 5, 999],
            2,
            999,
        ),
        (
            [
                field("a", "int", 1, 10),
                field("b", "float", 1.0, 10.0, value=float("nan")),
                field(
                    "c",
                    "ternary_times",
                    value=777,
                    dtype_param={
                        "target_names": ["a", "b"],
                        "product": 3,
                        "dtype": "int",
                    },
                ),
            ],
            [5, float("nan"), 777],
            2,
            777,
        ),
        (
            [
                field("a", "int", 1, 10),
                field(
                    "c",
                    "ternary_times",
                    value=777,
                    dtype_param={
                        "target_names": ["a", "missing_b"],
                        "product": 2,
                        "dtype": "int",
                    },
                ),
            ],
            [3, 777],
            1,
            777,
        ),
    ],
)
def test_ternary_derived_fields_keep_value_when_source_invalid(fields, values, index, original):
    assert derive(fields, values)[index].value == original


@pytest.mark.parametrize(
    "fields,values,product",
    [
        (
            [
                field("tp", "int", 1, 8),
                field("pp", "int", 1, 4),
                field(
                    "dp",
                    "ternary_factories",
                    dtype_param={
                        "target_names": ["tp", "pp"],
                        "product": 16,
                        "dtype": "int",
                    },
                ),
            ],
            [8, 4, 0],
            16,
        ),
        (
            [
                field("tp", "enum", 0, 1, dtype_param=[1, 2, 4, 8]),
                field("pp", "enum", 0, 1, dtype_param=[1, 2, 4]),
                field(
                    "dp",
                    "ternary_factories",
                    dtype_param={
                        "target_names": ["tp", "pp"],
                        "product": 16,
                        "dtype": "int",
                    },
                ),
            ],
            [8, 4, 0],
            16,
        ),
        (
            [
                field("tp", "int", 2, 3),
                field("pp", "int", 2, 3),
                field(
                    "dp",
                    "ternary_factories",
                    dtype_param={
                        "target_names": ["tp", "pp"],
                        "product": 12,
                        "dtype": "int",
                    },
                ),
            ],
            [3, 3, 0],
            12,
        ),
    ],
)
def test_ternary_factories_repair_keeps_product_consistent(fields, values, product):
    result = derive(fields, values)
    tp_value, pp_value, dp_value = [item.value for item in result]

    assert dp_value > 0
    assert tp_value * pp_value * dp_value == product


def test_ternary_factories_repair_falls_back_to_clamp_for_non_discrete_sources():
    fields = [
        field("tp", "float", 0.5, 8.0),
        field("pp", "float", 0.5, 4.0),
        field(
            "dp",
            "ternary_factories",
            dtype_param={"target_names": ["tp", "pp"], "product": 16, "dtype": "int"},
        ),
    ]

    assert derive(fields, [8.0, 4.0, 0])[2].value == 1


def test_ternary_factories_non_divisible_and_unrepairable_combination_raises():
    fields = [
        field("tp", "int", 1, 1000),
        field("pp", "enum", 0, 1, value=3, dtype_param=[3]),
        field(
            "dp",
            "ternary_factories",
            value=99,
            dtype_param={"target_names": ["tp", "pp"], "product": 32, "dtype": "int"},
        ),
    ]

    with pytest.raises(ValueError, match="product=32 not divisible by divisor=24"):
        derive(fields, [8, 3, 99])


def test_ternary_factories_map_param_integration():
    fields = [
        field("tp", "int", 1, 8),
        field("pp", "int", 1, 4),
        field(
            "dp",
            "ternary_factories",
            0,
            0,
            dtype_param={"target_names": ["tp", "pp"], "product": 16, "dtype": "int"},
        ),
    ]

    result = map_param_with_value(np.array([2.0, 4.0]), fields)

    assert [item.value for item in result] == [2, 4, 2]


def test_ternary_times_map_param_integration():
    fields = [
        field("seq_len", "int", 128, 4096),
        field("batch_size", "int", 1, 64),
        field(
            "total_tokens",
            "ternary_times",
            0,
            0,
            dtype_param={
                "target_names": ["seq_len", "batch_size"],
                "product": 1,
                "dtype": "int",
            },
        ),
    ]

    result = map_param_with_value(np.array([512.0, 4.0]), fields)

    assert [item.value for item in result] == [512, 4, 2048]


@pytest.mark.parametrize(
    "dtype_param,context,expected",
    [
        (
            {
                "target_names": ["tp", "pp"],
                "priority_policy": "fixed",
                "priority": ["pp", "tp"],
            },
            None,
            ["pp", "tp"],
        ),
        (
            {"target_names": ["tp", "pp"], "priority_policy": "fixed"},
            None,
            ["tp", "pp"],
        ),
        (
            {
                "target_names": ["tp", "pp"],
                "priority_policy": "fixed",
                "priority": ["tp"],
            },
            None,
            ["tp", "pp"],
        ),
        (
            {"target_names": ["tp", "pp"], "priority_policy": "balanced"},
            None,
            ["tp", "pp"],
        ),
        (
            {"target_names": ["tp", "pp"], "priority_policy": "balanced"},
            DecodeContext(),
            ["tp", "pp"],
        ),
        (
            {"target_names": ["tp", "pp"], "priority_policy": "balanced"},
            DecodeContext(particle_index=0, n_particles=10),
            ["tp", "pp"],
        ),
        (
            {"target_names": ["tp", "pp"], "priority_policy": "balanced"},
            DecodeContext(particle_index=9, n_particles=10),
            ["pp", "tp"],
        ),
        (
            {"target_names": ["tp", "pp"], "priority_policy": "balanced"},
            DecodeContext(particle_index=0, n_particles=10, iteration=1),
            ["pp", "tp"],
        ),
        (
            {"target_names": ["tp", "pp"], "priority_policy": "balanced"},
            DecodeContext(particle_index=9, n_particles=10, iteration=1),
            ["tp", "pp"],
        ),
        (
            {"target_names": ["tp", "pp"]},
            DecodeContext(particle_index=0, n_particles=4),
            ["tp", "pp"],
        ),
        (
            {"target_names": ["tp"], "priority_policy": "balanced"},
            DecodeContext(particle_index=0, n_particles=10),
            ["tp"],
        ),
    ],
)
def test_resolve_priority_strategies(dtype_param, context, expected):
    assert resolve_priority(dtype_param, context) == expected


@pytest.mark.parametrize(
    "total,forward_indexes,reverse_indexes",
    [(10, range(0, 5), range(5, 10)), (11, range(0, 6), range(6, 11))],
)
def test_balanced_priority_splits_particle_population(total, forward_indexes, reverse_indexes):
    dtype_param = {"target_names": ["tp", "pp"], "priority_policy": "balanced"}

    assert [
        resolve_priority(dtype_param, DecodeContext(particle_index=i, n_particles=total)) for i in forward_indexes
    ] == [["tp", "pp"]] * len(list(forward_indexes))
    assert [
        resolve_priority(dtype_param, DecodeContext(particle_index=i, n_particles=total)) for i in reverse_indexes
    ] == [["pp", "tp"]] * len(list(reverse_indexes))


def test_balanced_priority_alternates_by_iteration():
    dtype_param = {"target_names": ["tp", "pp"], "priority_policy": "balanced"}

    assert [
        resolve_priority(
            dtype_param,
            DecodeContext(particle_index=0, n_particles=10, iteration=iteration),
        )
        for iteration in (0, 2, 4)
    ] == [["tp", "pp"]] * 3
    assert [
        resolve_priority(
            dtype_param,
            DecodeContext(particle_index=0, n_particles=10, iteration=iteration),
        )
        for iteration in (1, 3, 5)
    ] == [["pp", "tp"]] * 3


@pytest.mark.parametrize(
    "params_fields,values,context,expected_name,expected_value",
    [
        (
            pair_fields(policy="fixed", priority=["tp", "pp"]),
            [8, 5, 0],
            None,
            "tp",
            8,
        ),
        (
            pair_fields(policy="fixed", priority=["pp", "tp"]),
            [3, 4, 0],
            None,
            "pp",
            4,
        ),
        (
            pair_fields(policy="balanced"),
            [8, 3, 0],
            DecodeContext(particle_index=2, n_particles=10),
            "tp",
            8,
        ),
        (
            pair_fields(policy="balanced"),
            [3, 4, 0],
            DecodeContext(particle_index=7, n_particles=10),
            "pp",
            4,
        ),
    ],
)
def test_priority_repair_preserves_expected_source(params_fields, values, context, expected_name, expected_value):
    runtime_fields = clone_with_values(params_fields, values)

    ok = _repair_ternary_factories_with_priority(
        params_fields[2],
        runtime_fields,
        params_fields,
        product=32,
        min_val=1,
        max_val=None,
        conv=int,
        context=context,
    )

    by_name = {item.name: item.value for item in runtime_fields}
    assert ok
    assert by_name[expected_name] == expected_value
    assert by_name["tp"] * by_name["pp"] * by_name["dp"] == 32


def test_priority_repair_returns_false_when_no_candidate_combination_is_valid():
    params_fields = pair_fields(policy="fixed", priority=["tp", "pp"], tp_candidates=[4, 8], pp_candidates=[3])
    runtime_fields = clone_with_values(params_fields, [8, 3, 0])

    ok = _repair_ternary_factories_with_priority(
        params_fields[2],
        runtime_fields,
        params_fields,
        product=32,
        min_val=1,
        max_val=None,
        conv=int,
    )

    assert ok is False


def test_map_param_forwards_decode_context_to_priority_repair():
    fields = list(pair_fields(policy="balanced"))

    result = map_param_with_value(
        np.array([0.375, 0.375]),
        fields,
        decode_context=DecodeContext(particle_index=0, n_particles=10),
    )
    tp_value, pp_value, dp_value = [item.value for item in result]

    assert tp_value > 0
    assert pp_value > 0
    assert dp_value == int(32 / (tp_value * pp_value))


def test_map_param_without_decode_context_still_repairs_to_consistent_values():
    result = map_param_with_value(np.array([0.875, 0.375]), list(pair_fields(policy="balanced")))
    tp_value, pp_value, dp_value = [item.value for item in result]

    assert 32 % (tp_value * pp_value) == 0
    assert dp_value == int(32 / (tp_value * pp_value))


def test_env_backup_is_restored_for_manual_mindie_path_check(monkeypatch):
    monkeypatch.setenv("MIES_INSTALL_PATH", "/opt/mindie/latest/bin")
    before = os.environ.get("MIES_INSTALL_PATH")

    with patch.object(Path, "is_file", return_value=False):
        config_path, backup_path = _get_mindie_config_paths()

    assert os.environ.get("MIES_INSTALL_PATH") == before
    assert config_path == Path("/opt/mindie/latest/mindie_llm/conf/config.json")
    assert backup_path == Path("/opt/mindie/latest/mindie_llm/conf/config_bak.json")


# ==========================
# Supplementary tests for uncovered lines
# ==========================


class TestOptimizerConfigFieldConvertDtype:
    def test_convert_dtype_str(self):
        f = OptimizerConfigField(name="f", dtype="str", min=0, max=1, value="hello")
        assert f.convert_dtype(123) == "123"

    def test_convert_dtype_int(self):
        f = OptimizerConfigField(name="f", dtype="int", min=0, max=100, value=0)
        assert f.convert_dtype(3.7) == 3

    def test_convert_dtype_float(self):
        f = OptimizerConfigField(name="f", dtype="float", min=0, max=100, value=0)
        assert f.convert_dtype("3.7") == 3.7

    def test_convert_dtype_unknown_defaults_to_float(self):
        f = OptimizerConfigField(name="f", dtype="unknown", min=0, max=100, value=0)
        assert f.convert_dtype("3.7") == 3.7


class TestFindAvailableValue:
    def test_str_type(self):
        f = OptimizerConfigField(name="f", dtype="str", min=0, max=1, value="x")
        assert f.find_available_value(42) == "42"

    def test_enum_empty_dtype_param(self):
        f = OptimizerConfigField(name="f", dtype="enum", min=0, max=1, value=0, dtype_param=[])
        result = f.find_available_value(5)
        assert result == 5.0

    def test_enum_string_values_found(self):
        """For enum with string dtype_param, find_available_value raises ValueError on string input"""
        f = OptimizerConfigField(name="f", dtype="enum", min=0, max=1, value="a", dtype_param=["a", "b", "c"])
        # dtype_func.get("enum", float)("b") → float("b") raises ValueError
        with pytest.raises(ValueError):
            f.find_available_value("b")

    def test_enum_string_values_not_found(self):
        """For enum with string dtype_param, find_available_value raises ValueError on string input"""
        f = OptimizerConfigField(name="f", dtype="enum", min=0, max=1, value="a", dtype_param=["a", "b", "c"])
        with pytest.raises(ValueError):
            f.find_available_value("z")

    def test_enum_numeric_exact_match(self):
        f = OptimizerConfigField(name="f", dtype="enum", min=0, max=1, value=0, dtype_param=[1, 2, 4, 8])
        assert f.find_available_value(4) == 4

    def test_enum_numeric_bisect_middle(self):
        f = OptimizerConfigField(name="f", dtype="enum", min=0, max=1, value=0, dtype_param=[1, 2, 4, 8])
        # Value 3 -> bisect_left finds index 2 (value=4)
        assert f.find_available_value(3) == 4

    def test_enum_numeric_bisect_past_end(self):
        f = OptimizerConfigField(name="f", dtype="enum", min=0, max=1, value=0, dtype_param=[1, 2, 4, 8])
        # Value 10 -> bisect_left returns len(list), so returns last element
        assert f.find_available_value(10) == 8

    def test_value_within_range(self):
        f = OptimizerConfigField(name="f", dtype="int", min=0, max=100, value=0)
        assert f.find_available_value(50) == 50

    def test_value_below_min(self):
        f = OptimizerConfigField(name="f", dtype="int", min=10, max=100, value=0)
        assert f.find_available_value(5) == 10

    def test_value_above_max(self):
        f = OptimizerConfigField(name="f", dtype="int", min=10, max=100, value=0)
        assert f.find_available_value(200) == 100


class TestResolvePriorityExtended:
    def test_single_target(self):
        dtype_param = {"target_names": ["field_a"]}
        assert resolve_priority(dtype_param) == ["field_a"]

    def test_fixed_policy_valid(self):
        dtype_param = {
            "target_names": ["a", "b"],
            "priority_policy": "fixed",
            "priority": ["b", "a"],
        }
        assert resolve_priority(dtype_param) == ["b", "a"]

    def test_fixed_policy_invalid_falls_back(self):
        dtype_param = {
            "target_names": ["a", "b"],
            "priority_policy": "fixed",
            "priority": ["x", "y"],
        }
        assert resolve_priority(dtype_param) == ["a", "b"]

    def test_balanced_no_context(self):
        dtype_param = {"target_names": ["a", "b"], "priority_policy": "balanced"}
        assert resolve_priority(dtype_param, context=None) == ["a", "b"]

    def test_balanced_first_half(self):
        ctx = DecodeContext(particle_index=0, n_particles=10, iteration=0)
        dtype_param = {"target_names": ["a", "b"], "priority_policy": "balanced"}
        assert resolve_priority(dtype_param, context=ctx) == ["a", "b"]

    def test_balanced_second_half(self):
        ctx = DecodeContext(particle_index=7, n_particles=10, iteration=0)
        dtype_param = {"target_names": ["a", "b"], "priority_policy": "balanced"}
        assert resolve_priority(dtype_param, context=ctx) == ["b", "a"]

    def test_balanced_odd_iteration_flips(self):
        ctx = DecodeContext(particle_index=0, n_particles=10, iteration=1)
        dtype_param = {"target_names": ["a", "b"], "priority_policy": "balanced"}
        # First half + odd iteration -> flipped
        assert resolve_priority(dtype_param, context=ctx) == ["b", "a"]

    def test_unknown_policy_returns_target_names(self):
        dtype_param = {"target_names": ["a", "b"], "priority_policy": "unknown_policy"}
        assert resolve_priority(dtype_param) == ["a", "b"]


class TestMapParamWithValueBranches:
    def test_bool_dtype_true(self):
        fields = [OptimizerConfigField(name="flag", dtype="bool", min=0, max=1, value=0)]
        result = map_param_with_value(np.array([0.8]), fields)
        assert result[0].value is True

    def test_bool_dtype_false(self):
        fields = [OptimizerConfigField(name="flag", dtype="bool", min=0, max=1, value=0)]
        result = map_param_with_value(np.array([0.3]), fields)
        assert result[0].value is False

    def test_string_enum(self):
        fields = [
            OptimizerConfigField(
                name="mode",
                dtype="enum",
                min=0,
                max=1,
                value="auto",
                dtype_param=["auto", "manual"],
            )
        ]
        result = map_param_with_value(np.array([0.9]), fields)
        assert result[0].value in ("auto", "manual")

    def test_single_string_enum(self):
        fields = [
            OptimizerConfigField(
                name="mode",
                dtype="enum",
                min=0,
                max=1,
                value="only",
                dtype_param=["only"],
            )
        ]
        result = map_param_with_value(np.array([0.5]), fields)
        assert result[0].value == "only"

    def test_numeric_enum_at_min(self):
        fields = [OptimizerConfigField(name="bs", dtype="enum", min=0, max=1, value=0, dtype_param=[8, 16, 32])]
        result = map_param_with_value(np.array([0.0]), fields)
        assert result[0].value == 8

    def test_numeric_enum_at_max(self):
        fields = [OptimizerConfigField(name="bs", dtype="enum", min=0, max=1, value=0, dtype_param=[8, 16, 32])]
        result = map_param_with_value(np.array([1.0]), fields)
        assert result[0].value == 32

    def test_constant_field_skipped(self):
        fields = [
            OptimizerConfigField(name="c", dtype="int", min=5, max=5, value=5, constant=5),
            OptimizerConfigField(name="x", dtype="int", min=1, max=100, value=0),
        ]
        result = map_param_with_value(np.array([50]), fields)
        assert result[0].value == 5
        assert result[1].value == 50

    def test_float_dtype(self):
        fields = [OptimizerConfigField(name="lr", dtype="float", min=0, max=1, value=0)]
        result = map_param_with_value(np.array([0.75]), fields)
        assert result[0].value == 0.75

    def test_int_conversion_error(self):
        """Test map_param_with_value handles ValueError for int conversion"""
        fields = [OptimizerConfigField(name="bs", dtype="int", min=0, max=100, value=0)]
        # Use NaN which cannot be converted to int
        result = map_param_with_value(np.array([float("nan")]), fields)
        # NaN cannot be converted to int, falls back to raw value
        import math

        assert math.isnan(result[0].value)

    def test_float_conversion_error(self):
        """Test map_param_with_value handles TypeError for float conversion"""
        fields = [OptimizerConfigField(name="lr", dtype="float", min=0, max=1, value=0)]
        # Use object array with non-numeric value
        params = np.array([None], dtype=object)
        result = map_param_with_value(params, fields)
        assert result[0].value is None

    def test_support_select_batch_zeroes_time_fields(self):
        """Test that supportSelectBatch=True zeroes out prefill/decode time fields"""
        fields = [
            OptimizerConfigField(
                name="supportSelectBatch",
                dtype="bool",
                min=0,
                max=1,
                value=0,
                config_position="BackendConfig.ScheduleConfig.supportSelectBatch",
            ),
            OptimizerConfigField(
                name="prefillTime",
                dtype="int",
                min=0,
                max=1000,
                value=500,
                config_position="BackendConfig.ScheduleConfig.prefillTimeMsPerReq",
            ),
            OptimizerConfigField(
                name="decodeTime",
                dtype="int",
                min=0,
                max=1000,
                value=300,
                config_position="BackendConfig.ScheduleConfig.decodeTimeMsPerReq",
            ),
        ]
        # 0.8 > 0.5 → True for supportSelectBatch
        result = map_param_with_value(np.array([0.8, 500, 300]), fields)
        assert result[1].value == 0
        assert result[2].value == 0


class TestUpdateFactoriesField:
    def test_basic_division(self):
        """Test factories type: value = product / target.value"""
        target = OptimizerConfigField(name="batch_size", dtype="int", min=1, max=100, value=4)
        dependent = OptimizerConfigField(
            name="dp",
            dtype="int",
            min=1,
            max=16,
            value=0,
            dtype_param={"target_name": "batch_size", "product": 16, "dtype": "int"},
        )
        simulate_run_info = [deepcopy(target), deepcopy(dependent)]
        _update_factories_field(dependent, 1, (target, dependent), simulate_run_info)
        assert simulate_run_info[1].value == 4  # 16 / 4

    def test_zero_target_no_update(self):
        """Test factories type: zero target doesn't update (division by zero guard)"""
        target = OptimizerConfigField(name="batch_size", dtype="int", min=0, max=100, value=0)
        dependent = OptimizerConfigField(
            name="dp",
            dtype="int",
            min=1,
            max=16,
            value=99,
            dtype_param={"target_name": "batch_size", "product": 16, "dtype": "int"},
        )
        simulate_run_info = [deepcopy(target), deepcopy(dependent)]
        _update_factories_field(dependent, 1, (target, dependent), simulate_run_info)
        # value not updated because target is 0
        assert simulate_run_info[1].value == 99


class TestUpdateTimesField:
    def test_basic_multiplication(self):
        """Test times type: value = product × target.value"""
        target = OptimizerConfigField(name="tp", dtype="int", min=1, max=8, value=4)
        dependent = OptimizerConfigField(
            name="world_size",
            dtype="int",
            min=1,
            max=64,
            value=0,
            dtype_param={"target_name": "tp", "product": 2, "dtype": "int"},
        )
        simulate_run_info = [deepcopy(target), deepcopy(dependent)]
        _update_times_field(dependent, 1, (target, dependent), simulate_run_info)
        assert simulate_run_info[1].value == 8  # 2 * 4

    def test_none_target_skips_with_warning(self):
        """Test times type: None target value triggers warning and skips"""
        target = OptimizerConfigField(name="tp", dtype="int", min=0, max=8, value=0)
        dependent = OptimizerConfigField(
            name="world_size",
            dtype="int",
            min=1,
            max=64,
            value=99,
            dtype_param={"target_name": "tp", "product": 2, "dtype": "int"},
        )
        simulate_run_info = [deepcopy(target), deepcopy(dependent)]
        # Manually set value to None after creation to bypass pydantic validation
        object.__setattr__(simulate_run_info[0], "value", None)
        _update_times_field(dependent, 1, (target, dependent), simulate_run_info)
        # value not updated because target is None
        assert simulate_run_info[1].value == 99

    def test_nan_target_skips_with_warning(self):
        """Test times type: NaN target value triggers warning and skips"""
        target = OptimizerConfigField(name="tp", dtype="float", min=0, max=8, value=float("nan"))
        dependent = OptimizerConfigField(
            name="world_size",
            dtype="int",
            min=1,
            max=64,
            value=77,
            dtype_param={"target_name": "tp", "product": 2, "dtype": "int"},
        )
        simulate_run_info = [deepcopy(target), deepcopy(dependent)]
        _update_times_field(dependent, 1, (target, dependent), simulate_run_info)
        assert simulate_run_info[1].value == 77


class TestReverseSpecialField:
    def test_ratio_field_reverse(self):
        """Test reverse_special_field reverses ratio field"""
        ratio_field = OptimizerConfigField(
            name="max_preempt",
            dtype="ratio",
            min=0,
            max=1,
            value=50,
            dtype_param="batch_size",
        )
        target_field = OptimizerConfigField(
            name="batch_size",
            dtype="int",
            min=10,
            max=100,
            value=100,
            config_position="BackendConfig.ScheduleConfig.maxBatchSize",
        )
        params = np.array([0.0, 100.0])
        result = reverse_special_field((ratio_field, target_field), params, concurrency=100)
        assert result[0] == 0.5  # 50 / 100

    def test_concurrency_field_with_ratio_zero_value(self):
        """Test CONCURRENCY field with ratio dtype and zero value sets to 1"""
        conc_field = OptimizerConfigField(name="CONCURRENCY", dtype="ratio", min=0, max=1, value=0)
        params = np.array([0.0])
        result = reverse_special_field((conc_field,), params, concurrency=50)
        assert result[0] == 1

    def test_concurrency_field_with_ratio_nonzero(self):
        """Test CONCURRENCY field with ratio dtype and non-zero value"""
        conc_field = OptimizerConfigField(name="CONCURRENCY", dtype="ratio", min=0, max=1, value=25)
        params = np.array([0.0])
        result = reverse_special_field((conc_field,), params, concurrency=50)
        assert result[0] == 0.5  # 25 / 50

    def test_concurrency_field_non_ratio_with_value(self):
        """Test CONCURRENCY field with non-ratio dtype uses value directly"""
        conc_field = OptimizerConfigField(name="CONCURRENCY", dtype="int", min=1, max=100, value=30)
        params = np.array([0.0])
        result = reverse_special_field((conc_field,), params, concurrency=50)
        assert result[0] == 30

    def test_concurrency_field_none_value_uses_concurrency(self):
        """Test CONCURRENCY field with None value uses concurrency arg"""
        conc_field = OptimizerConfigField(name="CONCURRENCY", dtype="int", min=1, max=100, value=0)
        # Bypass pydantic to set None
        object.__setattr__(conc_field, "value", None)
        params = np.array([0.0])
        result = reverse_special_field((conc_field,), params, concurrency=42)
        assert result[0] == 42


class TestFieldToParam:
    def test_int_field(self):
        """Test field_to_param converts int field"""
        fields = (OptimizerConfigField(name="bs", dtype="int", min=1, max=100, value=50),)
        result = field_to_param(fields)
        assert result[0] == 50.0

    def test_bool_true_field(self):
        """Test field_to_param converts bool True to 1"""
        fields = (OptimizerConfigField(name="flag", dtype="bool", min=0, max=1, value=True),)
        result = field_to_param(fields)
        assert result[0] == 1

    def test_bool_false_field(self):
        """Test field_to_param converts bool False to 0"""
        fields = (OptimizerConfigField(name="flag", dtype="bool", min=0, max=1, value=False),)
        result = field_to_param(fields)
        assert result[0] == 0

    def test_enum_field_existing_value(self):
        """Test field_to_param with enum value in dtype_param"""
        fields = (
            OptimizerConfigField(
                name="policy",
                dtype="enum",
                min=0,
                max=1,
                value=1,
                dtype_param=[0, 1, 3],
            ),
        )
        result = field_to_param(fields)
        # index=1, segment midpoint
        assert result[0] is not None

    def test_enum_field_string_value_not_in_list(self):
        """Test field_to_param with string enum value not in dtype_param appends it"""
        fields = (
            OptimizerConfigField(
                name="mode",
                dtype="enum",
                min=0,
                max=1,
                value="new_mode",
                dtype_param=["auto", "manual"],
            ),
        )
        _ = field_to_param(fields)
        assert "new_mode" in fields[0].dtype_param

    def test_enum_field_numeric_value_not_in_list(self):
        """Test field_to_param with numeric enum value not in dtype_param inserts it"""
        fields = (OptimizerConfigField(name="bs", dtype="enum", min=0, max=1, value=12, dtype_param=[8, 16, 32]),)
        _ = field_to_param(fields)
        assert 12 in fields[0].dtype_param

    def test_float_field(self):
        """Test field_to_param converts float field"""
        fields = (OptimizerConfigField(name="lr", dtype="float", min=0, max=1, value=0.5),)
        result = field_to_param(fields)
        assert result[0] == 0.5

    def test_constant_field_skipped(self):
        """Test field_to_param skips constant fields"""
        fields = (
            OptimizerConfigField(name="c", dtype="int", min=5, max=5, value=5, constant=5),
            OptimizerConfigField(name="x", dtype="int", min=1, max=100, value=42),
        )
        result = field_to_param(fields)
        assert len(result) == 1
        assert result[0] == 42.0

    def test_concurrency_field_sets_concurrency(self):
        """Test field_to_param detects max_batch_size and sets concurrency"""
        fields = (
            OptimizerConfigField(
                name="max_batch_size",
                dtype="int",
                min=10,
                max=200,
                value=64,
                config_position="BackendConfig.ScheduleConfig.maxBatchSize",
            ),
        )
        result = field_to_param(fields)
        assert result[0] == 64.0

    def test_int_conversion_error(self):
        """Test field_to_param handles int conversion error"""
        # float('inf') cannot be converted to int (OverflowError)
        fields = (OptimizerConfigField(name="bs", dtype="int", min=0, max=100, value=float("inf")),)
        result = field_to_param(fields)
        # Falls back to appending raw value (inf) which works in float array
        assert result[0] == float("inf")


class TestSettingsValidators(unittest.TestCase):
    def test_settings_customise_sources_includes_toml(self):
        init_settings = MagicMock()
        env_settings = MagicMock()
        dotenv_settings = MagicMock()
        file_secret_settings = MagicMock()

        sources = Settings.settings_customise_sources(
            Settings,
            init_settings,
            env_settings,
            dotenv_settings,
            file_secret_settings,
        )

        assert sources[0] is init_settings
        assert sources[1] is env_settings
        assert sources[2].__class__.__name__ == "TomlConfigSettingsSource"
        assert sources[3] is file_secret_settings

    def test_partial_update_vllm_syncs_benchmark_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            settings = Settings(output=base / "out", simulator_output=base / "sim")
            settings.vllm.command.host = "10.0.0.1"
            settings.vllm.command.port = "8123"
            settings.vllm.command.model = "test-model"
            settings.vllm.command.served_model_name = "served-name"

            updated = Settings.partial_update_vllm(settings)

            assert updated.vllm_benchmark.command.host == "10.0.0.1"
            assert updated.vllm_benchmark.command.port == "8123"
            assert updated.vllm_benchmark.command.model == "test-model"
            assert updated.vllm_benchmark.command.served_model_name == "served-name"


class TestGetSettingsAndRegister:
    def test_register_settings_custom_func(self):
        """Test register_settings with custom function"""
        import optix.config.config as config_mod

        # Save original state
        original_settings = config_mod.settings
        original_func = config_mod.custom_settings_func
        try:
            config_mod.settings = None
            sentinel = object()

            def custom():
                return sentinel

            register_settings(custom)
            result = get_settings()
            assert result is sentinel
        finally:
            # Restore
            config_mod.settings = original_settings
            config_mod.custom_settings_func = original_func

    def test_get_settings_default(self):
        """Test get_settings returns default Settings when no custom func"""
        import optix.config.config as config_mod

        original_settings = config_mod.settings
        original_func = config_mod.custom_settings_func
        try:
            config_mod.settings = None
            config_mod.custom_settings_func = None
            result = get_settings()
            assert result is not None
        finally:
            config_mod.settings = original_settings
            config_mod.custom_settings_func = original_func
