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

from types import SimpleNamespace

import pytest

from tensor_cast.layers.moe_layer import (
    FusedMoETensorCast,
    _expected_max_load,
    _expected_max_z,
    _soft_critical_load,
)


@pytest.mark.parametrize(
    ("total", "parts", "expected"),
    [
        (0, 32, 0),
        (48, 32, 4),
        (192, 32, 11),
        (48, 1, 48),
    ],
)
def test_expected_max_load(total, parts, expected):
    assert _expected_max_load(total, parts) == expected


@pytest.mark.parametrize(
    ("num_tokens", "expected_critical_load"),
    [
        (48, 10),
        (192, 12),
    ],
)
def test_moe_split_preserves_tokens_and_models_critical_rank(num_tokens, expected_critical_load):
    layer = SimpleNamespace(
        ep_group=SimpleNamespace(world_size=32, rank_in_group=0),
        num_external_shared_experts=0,
        num_global_experts=384,
        expert_idx_start=0,
        num_local_experts=12,
        _expected_max_z=_expected_max_z(32),
    )

    input_by_device, output_by_device, input_by_expert, output_by_expert = FusedMoETensorCast.get_split_sizes(
        layer, num_tokens, top_k=6
    )

    assert sum(input_by_device) == num_tokens
    assert input_by_device[0] == expected_critical_load
    assert max(input_by_device) == expected_critical_load
    assert output_by_device == [expected_critical_load] * 32
    assert sum(output_by_device) == 32 * expected_critical_load

    assert sum(input_by_expert) == num_tokens
    assert len(output_by_expert) == 32
    assert all(split == output_by_expert[0] for split in output_by_expert)
    assert sum(output_by_expert[0]) == expected_critical_load


def test_moe_split_is_workload_sensitive_without_full_balance():
    assert _soft_critical_load(192, 32, 384, 12) > _soft_critical_load(48, 32, 384, 12)
    assert 32 * _soft_critical_load(48, 32, 384, 12) < 384
    assert 32 * _soft_critical_load(192, 32, 384, 12) <= 384


def test_small_moe_layout_keeps_balanced_critical_load():
    assert _soft_critical_load(10, 3, 6, 2) == 4
    assert _soft_critical_load(7, 3, 5, 2) == 3
