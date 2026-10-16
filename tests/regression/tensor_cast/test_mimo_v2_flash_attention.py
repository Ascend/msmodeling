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
from unittest.mock import MagicMock

import pytest
import torch

from tensor_cast.core.input_generator import get_kv_cache_info
from tensor_cast.layers.attention import AttentionMetadataTensorCast, flash_attention_forward
from tensor_cast.performance_model.op_invoke_info import OpInvokeInfo
from tensor_cast.transformers.builtin_model.mimo_v2_flash import MiMoV2AttentionTensorCast
from tensor_cast.transformers.custom_model_registry import get_model_profile


def _mimo_attention_args(sinks):
    query = torch.empty((8192, 64 * 192), device="meta", dtype=torch.bfloat16)
    key = torch.empty((128, 128, 4, 192), device="meta", dtype=torch.bfloat16)
    value = torch.empty((128, 128, 4, 128), device="meta", dtype=torch.bfloat16)
    block_table = torch.empty((4, 16), device="meta", dtype=torch.int32)
    query_start_loc = torch.empty((5,), device="meta", dtype=torch.int32)
    seq_lens = torch.full((4,), 2048, dtype=torch.int32)
    query_lens = torch.full((4,), 2048, dtype=torch.int32)
    return (
        query,
        key,
        value,
        None,
        block_table,
        query_start_loc,
        seq_lens,
        query_lens,
        192,
        128,
        sinks,
        torch.bfloat16,
    )


def test_mimo_v2_attention_shape_and_asymmetric_cost():
    sinks = torch.empty((64,), device="meta", dtype=torch.bfloat16)
    args = _mimo_attention_args(sinks)
    output = torch.ops.tensor_cast.mimo_v2_attention(*args)
    properties = OpInvokeInfo(torch.ops.tensor_cast.mimo_v2_attention.default, args, {}, output).get_perf_properties()

    interactions = 4 * 2048 * 2048
    expected_mma = 2 * interactions * 64 * (192 + 128)
    expected_softmax = (interactions + 4 * 2048) * 64 * 4

    assert output.shape == (8192, 64 * 128)
    assert output.dtype == torch.bfloat16
    assert properties.compute_ops[torch.bfloat16].mma_ops == expected_mma
    assert properties.compute_ops[torch.bfloat16].gp_ops == expected_softmax


def test_mimo_v2_sink_only_adds_one_softmax_column():
    with_sink_args = _mimo_attention_args(torch.empty((64,), device="meta", dtype=torch.bfloat16))
    without_sink_args = _mimo_attention_args(None)
    with_sink_out = torch.ops.tensor_cast.mimo_v2_attention(*with_sink_args)
    without_sink_out = torch.ops.tensor_cast.mimo_v2_attention(*without_sink_args)
    with_sink = OpInvokeInfo(
        torch.ops.tensor_cast.mimo_v2_attention.default, with_sink_args, {}, with_sink_out
    ).get_perf_properties()
    without_sink = OpInvokeInfo(
        torch.ops.tensor_cast.mimo_v2_attention.default, without_sink_args, {}, without_sink_out
    ).get_perf_properties()

    assert with_sink.compute_ops[torch.bfloat16].mma_ops == without_sink.compute_ops[torch.bfloat16].mma_ops
    assert (
        with_sink.compute_ops[torch.bfloat16].gp_ops - without_sink.compute_ops[torch.bfloat16].gp_ops
        == 4 * 2048 * 64 * 4
    )


def test_mimo_v2_attention_supports_four_dimensional_prefill_without_cache():
    query = torch.empty((2, 3, 4, 192), device="meta", dtype=torch.bfloat16)
    key = torch.empty((2, 5, 2, 192), device="meta", dtype=torch.bfloat16)
    value = torch.empty((2, 5, 2, 128), device="meta", dtype=torch.bfloat16)
    sinks = torch.empty((4,), device="meta", dtype=torch.bfloat16)
    args = (query, key, value, None, None, None, None, None, 192, 128, sinks, torch.bfloat16)

    output = torch.ops.tensor_cast.mimo_v2_attention(*args)
    properties = OpInvokeInfo(torch.ops.tensor_cast.mimo_v2_attention.default, args, {}, output).get_perf_properties()

    interactions = 2 * 3 * 5
    assert output.shape == (2, 3, 4, 128)
    assert properties.compute_ops[torch.bfloat16].mma_ops == 2 * interactions * 4 * (192 + 128)
    assert properties.compute_ops[torch.bfloat16].gp_ops == (interactions + 2 * 3) * 4 * 4


@pytest.mark.parametrize(
    ("query", "qk_head_dim", "sinks", "message"),
    [
        (
            torch.empty((2, 4 * 192), device="meta", dtype=torch.bfloat16),
            0,
            None,
            "head dimensions must be positive",
        ),
        (
            torch.empty((2, 769), device="meta", dtype=torch.bfloat16),
            192,
            None,
            "query width must be divisible",
        ),
        (
            torch.empty((2, 4 * 192), device="meta", dtype=torch.bfloat16),
            192,
            torch.empty((5,), device="meta", dtype=torch.bfloat16),
            "sink count must match local query heads",
        ),
    ],
)
def test_mimo_v2_attention_rejects_invalid_shapes(query, qk_head_dim, sinks, message):
    with pytest.raises(ValueError, match=message):
        torch.ops.tensor_cast.mimo_v2_attention(
            query,
            torch.empty((2, 2 * 192), device="meta", dtype=torch.bfloat16),
            torch.empty((2, 2 * 128), device="meta", dtype=torch.bfloat16),
            None,
            None,
            None,
            None,
            None,
            qk_head_dim,
            128,
            sinks,
            torch.bfloat16,
        )


def test_flash_attention_forward_passes_mimo_v2_asymmetric_head_dimensions():
    attention = MagicMock()
    attention.forward.return_value = torch.empty((2, 4 * 128), device="meta", dtype=torch.bfloat16)
    module = SimpleNamespace(layer_idx=0, head_dim=192, v_head_dim=128)

    output, weights = flash_attention_forward(
        module,
        torch.empty((1, 4, 2, 192), device="meta", dtype=torch.bfloat16),
        torch.empty((1, 2, 2, 192), device="meta", dtype=torch.bfloat16),
        torch.empty((1, 2, 2, 128), device="meta", dtype=torch.bfloat16),
        None,
        attention_by_layers={0: attention},
    )

    assert output.shape == (2, 4 * 128)
    assert weights is None
    assert attention.forward.call_args.kwargs["_tensor_cast_qk_head_dim"] == 192
    assert attention.forward.call_args.kwargs["_tensor_cast_v_head_dim"] == 128


def test_mimo_v2_reshape_and_cache_accounts_valid_kv_payload():
    key = torch.empty((2, 2 * 192), device="meta", dtype=torch.bfloat16)
    value = torch.empty((2, 2 * 128), device="meta", dtype=torch.bfloat16)
    kv_cache = torch.empty((1, 1, 128, 2, 320), device="meta", dtype=torch.bfloat16)
    slot_mapping = torch.empty((2,), device="meta", dtype=torch.int32)
    args = (key, value, kv_cache, slot_mapping, 192, 128)

    properties = OpInvokeInfo(
        torch.ops.tensor_cast.mimo_v2_reshape_and_cache.default,
        args,
        {},
        None,
    ).get_perf_properties()

    expected_write_bytes = (key.numel() + value.numel()) * kv_cache.element_size()
    assert properties.memory_write_bytes == expected_write_bytes


def test_mimo_v2_attention_adapter_localizes_sink_and_uses_packed_cache():
    attention = MiMoV2AttentionTensorCast()
    metadata = AttentionMetadataTensorCast(
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        seq_lens=torch.tensor([2], dtype=torch.int32),
        query_lens=torch.tensor([2], dtype=torch.int32),
        block_table_tensor=torch.empty((1, 1), device="meta", dtype=torch.int32),
        slot_mapping=torch.empty((2,), device="meta", dtype=torch.int32),
    )

    output = attention(
        torch.empty((2, 4 * 192), device="meta", dtype=torch.bfloat16),
        torch.empty((2, 2 * 192), device="meta", dtype=torch.bfloat16),
        torch.empty((2, 2 * 128), device="meta", dtype=torch.bfloat16),
        None,
        kv_cache=torch.empty((1, 1, 128, 2, 320), device="meta", dtype=torch.bfloat16),
        attention_meta=metadata,
        _tensor_cast_qk_head_dim=192,
        _tensor_cast_v_head_dim=128,
        # Simulate TP=2: the parameter retains 8 global heads while the local
        # query has 4 heads. The wrapper must slice the sink before fusion.
        sinks=torch.empty((8,), device="meta", dtype=torch.bfloat16),
    )

    assert output.shape == (2, 4 * 128)


def test_mimo_v2_cache_uses_layer_specific_kv_heads_and_exact_packed_width():
    text_config = SimpleNamespace(
        model_type="mimo_v2_flash",
        hybrid_layer_pattern=[0, 1],
        num_key_value_heads=4,
        head_dim=192,
        v_head_dim=128,
        swa_num_key_value_heads=8,
        swa_head_dim=192,
        swa_v_head_dim=128,
    )
    model_config = SimpleNamespace(
        hf_config=text_config,
        parallel_config=SimpleNamespace(tensor_parallel_size=1, decode_context_parallel_size=1),
        dtype=torch.bfloat16,
        mla_config=None,
        mtp_config=None,
        dflash_config=None,
        dspark_config=None,
        has_draft_spec=lambda: False,
        draft_num_layers=lambda: 0,
        quant_config=SimpleNamespace(attention_configs={}),
    )
    model = SimpleNamespace(model_config=model_config, text_config=text_config, num_hidden_layers=2)

    caches, bytes_per_token = get_kv_cache_info(model, num_blocks=16, block_size=128)

    assert caches[0].shape == (1, 16, 128, 4, 320)
    assert caches[1].shape == (1, 16, 128, 8, 320)
    assert bytes_per_token == (4 + 8) * (192 + 128) * 2


def test_mimo_v2_profile_selects_asymmetric_attention_adapter():
    model = SimpleNamespace(model_config=SimpleNamespace(attention_cls=None))
    profile = get_model_profile("mimo_v2_flash")

    profile.patch_method(model)

    assert model.model_config.attention_cls is MiMoV2AttentionTensorCast
