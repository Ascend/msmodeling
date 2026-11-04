# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of the License at
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

import torch

from tensor_cast.layers.rotary_embedding import CachingMRotaryEmb
from tensor_cast.performance_model.op_invoke_info import OpInvokeInfo


class _FakeMRotaryEmb(torch.nn.Module):
    def forward(self, x, position_ids):
        return (
            torch.empty((*position_ids.shape, 8), device=x.device, dtype=x.dtype),
            torch.empty((*position_ids.shape, 8), device=x.device, dtype=x.dtype),
        )


def test_mrope_cache_lookup_accepts_three_axis_positions():
    cache = torch.empty((32, 24), device="meta", dtype=torch.bfloat16)
    positions = torch.empty((3, 2, 5), device="meta", dtype=torch.long)

    cos_sin = torch.ops.tensor_cast.mrope_cache_lookup(cache, positions, 12)

    assert cos_sin.shape == (2, 5, 24)
    assert cos_sin.dtype == torch.bfloat16


def test_caching_mrope_builds_cache_and_selects_runtime_positions():
    cached = CachingMRotaryEmb(_FakeMRotaryEmb(), torch.bfloat16, max_position_embeddings=32)
    x = torch.empty((2, 5), device="meta", dtype=torch.bfloat16)
    positions = torch.empty((3, 2, 5), device="meta", dtype=torch.long)

    cos, sin = cached(x, positions)

    assert cached.cos_sin_cache.shape == (32, 16)
    assert cached.rotary_dim == 8
    assert cos.shape == (2, 5, 8)
    assert sin.shape == (2, 5, 8)


def test_caching_mrope_falls_back_for_a_different_activation_dtype():
    cached = CachingMRotaryEmb(_FakeMRotaryEmb(), torch.bfloat16, max_position_embeddings=32)
    x = torch.empty((1, 5), device="meta", dtype=torch.float32)
    positions = torch.empty((1, 5), device="meta", dtype=torch.long)

    cos, sin = cached(x, positions)

    assert cos.shape == (1, 5, 8)
    assert sin.shape == (1, 5, 8)
    assert cos.dtype == torch.float32
    assert sin.dtype == torch.float32


def test_mrope_cache_lookup_properties_charge_selected_rows_only():
    cache = torch.empty((32, 24), device="meta", dtype=torch.bfloat16)
    positions = torch.empty((3, 2, 5), device="meta", dtype=torch.long)
    op = torch.ops.tensor_cast.mrope_cache_lookup.default
    output = op(cache, positions, 12)

    properties = OpInvokeInfo(op, (cache, positions, 12), {}, output).get_perf_properties()

    assert properties.memory_read_bytes == positions.numel() * positions.element_size() + output.numel() * 2
    assert properties.memory_write_bytes == output.numel() * output.element_size()


def test_vision_rotary_uses_fp32_and_restores_input_dtypes():
    from tensor_cast.transformers.builtin_model.bailing_v3_vl import (
        _bailing_vl_apply_vision_rotary_pos_emb,
        _bailing_vl_rotate_half,
    )

    query = torch.randn((3, 2, 8), dtype=torch.bfloat16)
    key = torch.randn((3, 2, 8), dtype=torch.float16)
    cos = torch.randn((3, 8), dtype=torch.bfloat16)
    sin = torch.randn((3, 8), dtype=torch.bfloat16)

    actual_query, actual_key = _bailing_vl_apply_vision_rotary_pos_emb(query, key, cos, sin)
    cos_fp32 = cos.unsqueeze(1).float()
    sin_fp32 = sin.unsqueeze(1).float()
    expected_query = (query.float() * cos_fp32 + _bailing_vl_rotate_half(query.float()) * sin_fp32).to(query.dtype)
    expected_key = (key.float() * cos_fp32 + _bailing_vl_rotate_half(key.float()) * sin_fp32).to(key.dtype)

    assert actual_query.dtype == query.dtype
    assert actual_key.dtype == key.dtype
    torch.testing.assert_close(actual_query, expected_query, rtol=0, atol=0)
    torch.testing.assert_close(actual_key, expected_key, rtol=0, atol=0)
