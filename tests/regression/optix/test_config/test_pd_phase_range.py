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
"""Regression coverage for integer range fields in P/D launch commands."""

import pytest
from pydantic import ValidationError

from optix.config.config import PdDisaggConfig, field_to_param, map_param_with_value
from optix.serve_plan import render_serve_flags


@pytest.mark.parametrize("phase_name", ["prefill", "decode"])
@pytest.mark.parametrize(
    "name,minimum,maximum,step,initial,flag",
    [
        ("MAX_NUM_BATCHED_TOKENS", 8192, 32768, 1024, 16384, "--max-num-batched-tokens"),
        ("MAX_NUM_SEQS", 32, 128, 32, 64, "--max-num-seqs"),
        ("MAX_NUM_SEQS", 32, 36, 1, 34, "--max-num-seqs"),
    ],
)
def test_pd_range_baseline_renders_integer(phase_name, name, minimum, maximum, step, initial, flag):
    config = PdDisaggConfig.model_validate(
        {
            phase_name: {
                "target_field": [
                    {
                        "name": name,
                        "config_position": "run",
                        "dtype": "range",
                        "min": minimum,
                        "max": maximum,
                        "dtype_param": step,
                        "value": initial,
                    }
                ]
            }
        }
    )
    # Re-reading a normalized configuration must preserve its integer candidates.
    config = PdDisaggConfig.model_validate(config.model_dump())
    fields = tuple(getattr(config, phase_name).target_field)
    assert fields[0].dtype == "enum"
    assert fields[0].dtype_param == list(range(minimum, maximum + 1, step))

    baseline = map_param_with_value(field_to_param(fields), fields)
    assert type(baseline[0].value) is int
    assert render_serve_flags(baseline) == [flag, str(initial)]

    for position in (0.0, 0.5, 1.0):
        candidate = map_param_with_value([position], fields)
        assert type(candidate[0].value) is int
        assert candidate[0].value in fields[0].dtype_param


@pytest.mark.parametrize("phase_name", ["prefill", "decode"])
@pytest.mark.parametrize(
    "step_config",
    [
        pytest.param({}, id="missing"),
        pytest.param({"dtype_param": None}, id="none"),
        pytest.param({"dtype_param": 0}, id="zero"),
        pytest.param({"dtype_param": -1024}, id="negative"),
        pytest.param({"dtype_param": 1.5}, id="fractional"),
        pytest.param({"dtype_param": 1.0}, id="float"),
        pytest.param({"dtype_param": "1024"}, id="string"),
        pytest.param({"dtype_param": True}, id="true"),
        pytest.param({"dtype_param": False}, id="false"),
        pytest.param({"dtype_param": []}, id="list"),
        pytest.param({"dtype_param": {}}, id="dict"),
    ],
)
def test_pd_range_rejects_invalid_step(phase_name, step_config):
    with pytest.raises(ValidationError, match="MAX_NUM_BATCHED_TOKENS.*dtype_param.*positive integer") as exc_info:
        PdDisaggConfig.model_validate(
            {
                phase_name: {
                    "target_field": [
                        {
                            "name": "MAX_NUM_BATCHED_TOKENS",
                            "dtype": "range",
                            "min": 8192,
                            "max": 32768,
                            **step_config,
                        }
                    ]
                }
            }
        )
    assert exc_info.value.errors()[0]["loc"] == (phase_name, "target_field")


@pytest.mark.parametrize("phase_name", ["prefill", "decode"])
@pytest.mark.parametrize(
    "bounds",
    [
        pytest.param({"min": float("nan"), "max": 128}, id="nan-min"),
        pytest.param({"min": 32, "max": float("nan")}, id="nan-max"),
        pytest.param({"min": float("-inf"), "max": 128}, id="infinite-min"),
        pytest.param({"min": 32, "max": float("inf")}, id="infinite-max"),
    ],
)
def test_pd_range_rejects_nonfinite_bounds(phase_name, bounds):
    with pytest.raises(ValidationError, match="MAX_NUM_SEQS.*min.*max.*finite"):
        PdDisaggConfig.model_validate(
            {
                phase_name: {
                    "target_field": [
                        {"name": "MAX_NUM_SEQS", "dtype": "range", "dtype_param": 32, "value": 64, **bounds}
                    ]
                }
            }
        )


@pytest.mark.parametrize("phase_name", ["prefill", "decode"])
@pytest.mark.parametrize("bound", ["min", "max"])
def test_pd_range_rejects_nonnumeric_bounds(phase_name, bound):
    field = {"name": "MAX_NUM_SEQS", "dtype": "range", "min": 32, "max": 128, "dtype_param": 32}
    field[bound] = "not-a-number"
    with pytest.raises(ValidationError) as exc_info:
        PdDisaggConfig.model_validate({phase_name: {"target_field": [field]}})
    assert exc_info.value.errors()[0]["loc"] == (phase_name, "target_field", 0, bound)
