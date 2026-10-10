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
import numpy as np
import pytest
import tomlkit

from optix.config.config import OptimizerConfigField, Settings, map_param_with_value, range_to_enum
from optix.config.custom_command import VllmCommand


@pytest.mark.parametrize(
    "start, end, step, expected",
    [
        (1, 9, 3, [1, 4, 7]),
        (1, 9, 2, [1, 3, 5, 7, 9]),
        (1, 9, 10, [1]),
        (5, 5, 3, [5]),
        (0, 3, 1, [0, 1, 2, 3]),
        (-5, 2, 3, [-5, -2, 1]),
    ],
)
def test_range_to_enum_respects_inclusive_bounds(start, end, step, expected):
    field = OptimizerConfigField(min=start, max=end, dtype="range", dtype_param=step)

    range_to_enum((field,))

    assert field.dtype == "enum"
    assert (field.min, field.max) == (0, 1)
    assert field.dtype_param == expected


@pytest.mark.parametrize("step", [0, "0", -1, "-1"])
def test_range_to_enum_rejects_non_positive_step(step):
    field = OptimizerConfigField(
        name="MAX_NUM_SEQS",
        min=1,
        max=9,
        dtype="range",
        dtype_param=step,
    )

    with pytest.raises(ValueError, match="MAX_NUM_SEQS.*positive step"):
        range_to_enum((field,))


@pytest.mark.parametrize("step, expected", [(3, [1, 4, 7]), (2, [1, 3, 5, 7, 9])])
def test_range_settings_decode_and_vllm_command_stay_within_bounds(tmp_path, monkeypatch, step, expected):
    monkeypatch.chdir(tmp_path)
    config = tomlkit.parse(
        f'''
output = "output"
use_request_rate_calibration = false
n_particles = 1
iters = 1

[vllm.command]
host = "127.0.0.1"
port = "8000"
model = "example-model"
served_model_name = "example-model"
others = ""

[[vllm.target_field]]
name = "MAX_NUM_SEQS"
config_position = "run"
dtype = "range"
min = 1
max = 9
dtype_param = {step}
value = 1
'''
    )
    settings = Settings(**config)
    fields = tuple(settings.vllm.target_field)
    assert fields[0].dtype_param == expected

    # Exercise both endpoints and every enum bucket through the real command builder.
    positions = [0.0, *((i + 0.5) / len(expected) for i in range(len(expected))), 1.0]
    decoded_values = []
    for position in positions:
        decoded = map_param_with_value(np.array([position]), fields)
        value = decoded[0].value
        command = VllmCommand(settings.vllm.command, decoded).command
        assert value in expected
        assert command[command.index("--max-num-seqs") + 1] == str(value)
        decoded_values.append(value)
    assert decoded_values == [expected[0], *expected, expected[-1]]
