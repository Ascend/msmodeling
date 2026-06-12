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
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from experimental.optix.config.config import (
    DecodeContext,
    ErrorPatternConfig,
    ErrorType,
    HealthCheckConfig,
    MindieConfig,
    OptimizerConfigField,
    _get_mindie_config_paths,
    _repair_ternary_factories_with_priority,
    map_param_with_value,
    resolve_priority,
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
            "max_queue_deloy_microseconds",
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


@patch("experimental.optix.config.config._get_mindie_config_paths")
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


@patch("experimental.optix.config.config._get_mindie_config_paths")
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
