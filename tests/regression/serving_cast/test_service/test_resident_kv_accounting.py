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
from unittest.mock import patch

import torch

from tensor_cast.core.input_generator import RequestInfo
from tensor_cast.core.model_runner import ModelRunner
from serving_cast.service.agg_throughput_optimizer import _resident_kv_cache_size_gb


def test_resident_kv_uses_actual_compressed_tensors_and_excludes_linear_layers():
    runner = ModelRunner.__new__(ModelRunner)
    runner.model = object()
    runner.user_input = SimpleNamespace(block_size=128)
    requests = [RequestInfo(1, 32768, is_decode=False)]
    # A compressed/sliding cache must keep its actual physical shape rather
    # than being expanded to the full 32768-token logical context.
    inputs = {
        "kv_cache_by_layers": {
            0: torch.empty(2, 128, 64, dtype=torch.bfloat16, device="meta"),
            1: torch.empty(4096, dtype=torch.float32, device="meta"),
        },
        "indexer_cache_by_layers": {0: torch.empty(128, 32, dtype=torch.int8, device="meta")},
    }
    with (
        patch("serving_cast.service.agg_throughput_optimizer.generate_inputs_varlen", return_value=inputs) as generate,
        patch("serving_cast.service.agg_throughput_optimizer.kv_cache_excluded_layer_indices", return_value={1}),
    ):
        actual = _resident_kv_cache_size_gb(runner, requests)
    assert actual == (2 * 128 * 64 * 2 + 128 * 32) / 1024**3
    generate.assert_called_once_with(runner.model, requests, 128)
