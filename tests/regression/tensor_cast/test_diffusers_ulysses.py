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

import pytest
import torch

from tensor_cast.device import TEST_DEVICE
from tensor_cast.diffusers.diffusers_attention import _attention, get_sp_group, set_sp_group
from tensor_cast.parallel_group import ParallelGroup
from tensor_cast.performance_model.analytic import AnalyticPerformanceModel
from tensor_cast.runtime import Runtime


@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("last_rank", [False, True])
@pytest.mark.parametrize(
    "batch_size,dtype,seq_len,kv_seq_len",
    [(1, torch.bfloat16, 8, 8), (1, torch.bfloat16, 4096, 4096), (2, torch.float32, 8, 16)],
)
def test_ulysses_all_to_all_conserves_remote_payload(world_size, last_rank, batch_size, dtype, seq_len, kv_seq_len):
    query = torch.empty((batch_size, seq_len // world_size, 24, 128), device="meta", dtype=dtype)
    key = torch.empty((batch_size, kv_seq_len // world_size, 24, 128), device="meta", dtype=dtype)
    group = ParallelGroup(world_size - 1 if last_rank else 0, [list(range(world_size))], world_size)
    previous_group = get_sp_group()
    set_sp_group(group)
    try:
        with Runtime(AnalyticPerformanceModel(TEST_DEVICE), TEST_DEVICE) as runtime, torch.no_grad():
            output = _attention(query, key, key)
    finally:
        set_sp_group(previous_group)

    assert output.shape == query.shape
    assert output.dtype == query.dtype
    exchanges = [
        event for event in runtime.event_list if event.op_invoke_info.func == torch.ops.tensor_cast.all_to_all.default
    ]
    assert len(exchanges) == 4
    # Each rank keeps one head shard and sends every other shard to a peer.
    for exchange, source in zip(exchanges, (query, key, key, query)):
        expected_bytes = source.numel() * source.element_size() * (world_size - 1) // world_size
        statistics = exchange.perf_results["analytic"].statistics
        assert statistics["total_bytes_sent"] == expected_bytes
        assert statistics["total_bytes_received"] == expected_bytes
        assert statistics["message_size_bytes"] == expected_bytes

    # Packing communication payload must not model synthetic initialization or copies.
    helper_events = [
        event
        for event in runtime.event_list
        if event.op_invoke_info.func
        not in (torch.ops.tensor_cast.attention.default, torch.ops.tensor_cast.all_to_all.default)
    ]
    assert sum(event.op_invoke_info.get_perf_properties().memory_write_bytes for event in helper_events) == 0
    assert sum(event.perf_results["analytic"].execution_time_s for event in helper_events) == 0
