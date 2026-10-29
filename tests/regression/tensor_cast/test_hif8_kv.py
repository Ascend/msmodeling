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

import copy

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from tensor_cast.core.quantization.config import create_attention_quant_config, create_linear_quant_config
from tensor_cast.core.quantization.datatypes import QuantizeAttentionAction
from tensor_cast.device import ATLAS_800, TEST_DEVICE
from tensor_cast.layers.attention import AttentionMetadataTensorCast, AttentionTensorCast
from tensor_cast.layers.parallel_linear import ColumnParallelLinear, RowParallelLinear
from tensor_cast.layers.quant_linear import TensorCastQuantLinear
from tensor_cast.model_config import AttentionQuantConfig
from tensor_cast.utils import SimFormat
from tensor_cast.parallel_group import ParallelGroup
from tensor_cast.performance_model.analytic import AnalyticPerformanceModel
from tensor_cast.performance_model.calibration.signature import build_calibration_signature
from tensor_cast.performance_model.op_invoke_info import OpInvokeInfo
from tensor_cast.performance_model.profiling_database.profiling_data_source import ProfilingDataSource
from tensor_cast.quantize_utils import AttentionQuantType
from tensor_cast.runtime import Runtime


def _tensor(*shape, dtype=torch.bfloat16):
    return torch.empty(shape, dtype=dtype, device="meta")


def _attention_info(dtype=torch.uint8, fake_lengths=False):
    q = _tensor(4, 64)
    k, v = _tensor(128, 16, 2, 16, dtype=dtype), _tensor(128, 16, 2, 16, dtype=dtype)
    table = torch.zeros((2, 2), dtype=torch.int32)
    start = torch.tensor([0, 3, 4], dtype=torch.int32)
    seq, qlen = torch.tensor([8, 24], dtype=torch.int32), torch.tensor([3, 1], dtype=torch.int32)
    if fake_lengths:
        seq, qlen = seq.to("meta"), qlen.to("meta")
    if dtype == torch.uint8:
        op = torch.ops.tensor_cast.attention_hif8_kv.default
        args = (q, k, v, None, table, start, seq, qlen, _tensor(dtype=torch.float32), 16, 16, None)
    else:
        op = torch.ops.tensor_cast.attention.default
        args = (q, k, v, None, table, start, seq, qlen)
    return OpInvokeInfo(op, args, {}, op(*args))


def test_storage_config_keeps_query_and_probabilities_unquantized():
    cfg = create_attention_quant_config(QuantizeAttentionAction.HIF8)
    assert cfg.is_hif8_kv
    assert cfg.get_quant_dtype() == torch.uint8
    assert cfg.kv_scale.dtype == torch.float32 and cfg.kv_scale.numel() == 1
    assert cfg.query_scale is cfg.attention_prob_scale is cfg.kv_projected_scale is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kv_scale": torch.ones(2)},
        {"kv_scale": torch.ones((), dtype=torch.bfloat16)},
        {"kv_offset": torch.tensor(0)},
        {"kv_scale": torch.tensor(0.0)},
        {"kv_scale": torch.tensor(float("nan"))},
    ],
)
def test_invalid_scale_or_offset_is_rejected(kwargs):
    with pytest.raises(ValueError, match="HIF8 KV cache"):
        AttentionQuantConfig(quant_type=AttentionQuantType.HIF8, **kwargs)


@pytest.mark.parametrize("scale_device", ["cpu", "meta", "fake"])
def test_scale_metadata_accepts_real_meta_and_fake_scales_at_both_entry_points(scale_device):
    if scale_device == "fake":
        mode = FakeTensorMode(allow_non_fake_inputs=True)
        scale = mode.from_tensor(torch.tensor(1.0, dtype=torch.float32))
    else:
        scale = torch.ones((), dtype=torch.float32, device=scale_device)
    config = AttentionQuantConfig(quant_type=AttentionQuantType.HIF8, kv_scale=scale)
    assert config.kv_scale is scale
    key, value = _tensor(3, 1, 16), _tensor(3, 1, 16)
    cache = _tensor(2, 4, 1, 16, dtype=torch.uint8)
    slots = _tensor(3, dtype=torch.int64)
    torch.ops.tensor_cast.hif8_kv_cache_write(key, value, cache, slots, scale)


@pytest.mark.parametrize("shape,dtype", [((2,), torch.float32), ((0,), torch.float32), ((), torch.bfloat16)])
def test_invalid_scale_metadata_is_rejected_at_config_and_direct_operator_entry(shape, dtype):
    scale = torch.ones(shape, dtype=dtype)
    message = "HIF8 KV cache requires a scalar float32 static scale"
    with pytest.raises(ValueError, match=message):
        AttentionQuantConfig(quant_type=AttentionQuantType.HIF8, kv_scale=scale)
    key, value = _tensor(3, 1, 16), _tensor(3, 1, 16)
    cache = _tensor(2, 4, 1, 16, dtype=torch.uint8)
    slots = _tensor(3, dtype=torch.int64)
    with pytest.raises(ValueError, match=message):
        torch.ops.tensor_cast.hif8_kv_cache_write(key, value, cache, slots, scale)


@pytest.mark.parametrize("fake_lengths", [False, True])
def test_attention_uses_bf16_mma_and_only_reads_addressed_hif8_cache(fake_lengths):
    info, baseline = _attention_info(fake_lengths=fake_lengths), _attention_info(torch.bfloat16, fake_lengths)
    p, bp = info.get_perf_properties(), baseline.get_perf_properties()
    assert p.compute_ops[torch.bfloat16].mma_ops == bp.compute_ops[torch.bfloat16].mma_ops
    assert SimFormat.HIF8 not in p.compute_ops and torch.uint8 not in p.compute_ops
    active_tokens = 2 * 2 * 16 if fake_lengths else 8 + 24
    payload_elements = active_tokens * 2 * (16 + 16)
    assert bp.memory_read_bytes - p.memory_read_bytes == payload_elements - 4  # one FP32 scale
    assert p.compute_ops[torch.float32].gp_ops == 3 * payload_elements
    assert p.memory_write_bytes == bp.memory_write_bytes == 4 * 64 * 2


def test_cache_recipe_does_not_use_hif8_mma_peak():
    info = _attention_info()
    device = copy.copy(ATLAS_800.A2_376T_64G)
    assert SimFormat.HIF8 not in device.mma_ops
    result = AnalyticPerformanceModel(device).process_op(info)
    expected = info.get_perf_properties().compute_ops[torch.bfloat16].mma_ops / (
        device.mma_ops[torch.bfloat16] * device.compute_efficiency
    )
    assert result.statistics["mma_ops_time_s"] == pytest.approx(expected)
    device.mma_ops = {**device.mma_ops, SimFormat.HIF8: 1}
    assert AnalyticPerformanceModel(device).process_op(info).execution_time_s == result.execution_time_s


@pytest.mark.parametrize("key_dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("value_dtype", [torch.float16, torch.bfloat16, torch.float32, None])
def test_write_counts_appended_rows_instead_of_allocated_capacity(key_dtype, value_dtype):
    key = _tensor(4, 2, 16, dtype=key_dtype)
    value = _tensor(4, 2, 16, dtype=value_dtype) if value_dtype is not None else None
    args = (
        key,
        value,
        _tensor(2 if value is not None else 1, 128, 16, 2, 16, dtype=torch.uint8),
        torch.arange(4),
        _tensor(dtype=torch.float32),
    )
    op = torch.ops.tensor_cast.hif8_kv_cache_write.default
    assert op(*args) is None
    p = OpInvokeInfo(op, args, {}, None).get_perf_properties()
    inputs = [x for x in (key, value) if x is not None]
    assert p.memory_read_bytes == sum(x.numel() * x.element_size() for x in inputs) + 4 * 8 + 4
    assert p.memory_write_bytes == sum(x.numel() for x in inputs)
    assert p.memory_readwrite_bytes == 0
    # The FP32 stream skips promotion, including when K and V have different dtypes.
    expected = sum(x.numel() * (4 if x.dtype == torch.float32 else 5) for x in inputs)
    assert p.compute_ops[torch.float32].gp_ops == expected
    quant = torch.ops.tensor_cast.quantize_hif8.default
    standalone = [OpInvokeInfo(quant, (x, args[4]), {}, quant(x, args[4])).get_perf_properties() for x in inputs]
    assert sum(props.compute_ops[torch.float32].gp_ops for props in standalone) == expected


def test_layer_emits_storage_write_and_bf16_attention_without_query_quantization():
    layer = AttentionTensorCast()
    layer.quant_config = create_attention_quant_config(QuantizeAttentionAction.HIF8)
    meta = AttentionMetadataTensorCast(
        torch.tensor([0, 1, 2]),
        torch.tensor([8, 24]),
        torch.tensor([1, 1]),
        torch.zeros(2, 2, dtype=torch.int32),
        torch.arange(2),
    )
    q, k, v = _tensor(2, 64, dtype=torch.float16), _tensor(2, 2, 16), _tensor(2, 2, 16)
    cache = _tensor(2, 4, 16, 2, 16, dtype=torch.uint8)
    with Runtime(AnalyticPerformanceModel(TEST_DEVICE), TEST_DEVICE) as runtime:
        out = layer(q, k, v, None, cache, meta)
    assert out.shape == q.shape and out.dtype == q.dtype
    names = [str(e.op_invoke_info.func) for e in runtime.event_list]
    assert "tensor_cast.hif8_kv_cache_write.default" in names
    assert "tensor_cast.attention_hif8_kv.default" in names
    assert "tensor_cast.quantize.default" not in names


def test_calibration_and_profiling_keep_storage_recipe_distinct():
    info = _attention_info()
    raw = AnalyticPerformanceModel(TEST_DEVICE).process_op(info)
    sig = build_calibration_signature(info, raw, device_name=TEST_DEVICE.name)
    assert sig.op_family == "attention"
    assert sig.dtype == "hif8"
    assert sig.features["compute_dtype"] == "bfloat16"
    assert sig.features["kv_cache_dtype"] == "hif8"
    assert sig.features["heads"] == 4 and sig.features["head_dim"] == 16
    source = ProfilingDataSource.__new__(ProfilingDataSource)
    source._op_mapping = {"operator_mappings": {str(info.func): {"zero_cost": True}}}
    assert source.lookup(info) is None
    assert source.last_miss_reason == "hif8_requires_verified_profile"


def test_payload_ops_reject_real_numerical_execution():
    with pytest.raises(ValueError, match="numerical execution"):
        torch.ops.tensor_cast.hif8_kv_cache_write(
            torch.ones(1, 8), None, torch.empty(1, 16, 8, dtype=torch.uint8), torch.tensor([0]), torch.tensor(1.0)
        )


def test_empty_cache_write_keeps_layout_and_has_no_conversion_work():
    op = torch.ops.tensor_cast.hif8_kv_cache_write.default
    args = (
        _tensor(0, 2, 16),
        _tensor(0, 2, 16),
        _tensor(2, 8, 16, 2, 16, dtype=torch.uint8),
        torch.empty(0, dtype=torch.int64),
        _tensor(dtype=torch.float32),
    )
    assert op(*args) is None
    props = OpInvokeInfo(op, args, {}, None).get_perf_properties()
    assert props.memory_write_bytes == 0
    assert all(work.gp_ops == work.mma_ops == 0 for work in props.compute_ops.values())


@pytest.mark.parametrize("action", ["HIF8", "hif8"])
def test_text_cli_accepts_hif8_kv_action(action):
    from cli.inference.text_generate import arg_parse

    args = arg_parse(
        [
            "tests/assets/model_config/qwen3_dense_0_6b",
            "--num-queries",
            "1",
            "--query-length",
            "8",
            "--quantize-attention-action",
            action,
        ]
    )
    assert args.quantize_attention_action == QuantizeAttentionAction.HIF8


@pytest.mark.parametrize(
    "parallel_cls,scale_channels,input_width,output_width",
    [
        (RowParallelLinear, 32, 8, 32),
        (ColumnParallelLinear, 16, 16, 16),
    ],
)
def test_hif8_linear_tp_preserves_output_channel_scale_axis(parallel_cls, scale_channels, input_width, output_width):
    group = ParallelGroup(rank=0, rank_groups=[[0, 1]], global_world_size=2)
    inner = TensorCastQuantLinear(
        torch.nn.Linear(16, 32, bias=False, device="meta", dtype=torch.bfloat16), create_linear_quant_config("HIF8")
    )
    layer = parallel_cls(inner, group, group)
    assert inner.weight_scale.shape == (scale_channels,)
    out = layer(_tensor(3, input_width))
    assert out.shape == (3, output_width)


@pytest.mark.parametrize("query_lens,seq_lens", [([8, 8], [8, 8]), ([1, 1], [24, 32]), ([8, 1], [8, 32])])
@pytest.mark.parametrize("fake_lengths", [False, True])
def test_mla_cache_bytes_and_bf16_compute(query_lens, seq_lens, fake_lengths):
    total = sum(query_lens)
    prefill = sum(x for x in query_lens if x >= 8)
    q = _tensor(total, 4, 32)
    projected, absorbed = _tensor(prefill, 4, 48), _tensor(total - prefill, 4, 48)
    table = torch.zeros(2, 4, dtype=torch.int32)
    starts = torch.tensor([0, query_lens[0], total], dtype=torch.int32)
    qlens, slens = torch.tensor(query_lens), torch.tensor(seq_lens)
    if fake_lengths:
        qlens, slens = qlens.to("meta"), slens.to("meta")
    prefix = (q, projected, absorbed)
    suffix = (table, starts, slens, qlens, 32, 32, None, None)
    cache_shape = (128, 16, 48)
    op = torch.ops.tensor_cast.multihead_latent_attention_hif8_kv.default
    args = (*prefix, _tensor(*cache_shape, dtype=torch.uint8), *suffix, _tensor(dtype=torch.float32))
    info = OpInvokeInfo(op, args, {}, op(*args))
    base_op = torch.ops.tensor_cast.multihead_latent_attention.default
    base_args = (*prefix, _tensor(*cache_shape), *suffix)
    base = OpInvokeInfo(base_op, base_args, {}, base_op(*base_args))
    p, bp = info.get_perf_properties(), base.get_perf_properties()
    elements = (2 * 4 * 16 if fake_lengths else sum(seq_lens)) * 48
    assert bp.memory_read_bytes - p.memory_read_bytes == elements - 4
    assert p.compute_ops[torch.float32].gp_ops == 3 * elements
    assert p.compute_ops[torch.bfloat16].mma_ops == bp.compute_ops[torch.bfloat16].mma_ops
    assert all(x.dtype == torch.bfloat16 for x in info.out)


def test_diffusion_without_kv_cache_rejects_recipe(monkeypatch):
    from tensor_cast.diffusers import diffusers_attention

    monkeypatch.setattr(
        diffusers_attention,
        "get_attention_quant_config",
        lambda: create_attention_quant_config(QuantizeAttentionAction.HIF8),
    )
    with pytest.raises(ValueError, match="diffusion attention has no KV cache"):
        diffusers_attention._run_attention(_tensor(1, 4, 2, 16), _tensor(1, 4, 2, 16), _tensor(1, 4, 2, 16))


def test_builtin_bf16_calibration_cannot_match_hif8_cache():
    from pathlib import Path
    from tensor_cast.performance_model.calibrated_analytic import CalibratedAnalyticPerformanceModel
    from tensor_cast.performance_model.calibration.profile import ProfileCalibrationDataSource

    profile = (
        Path(__file__).parents[3]
        / "tensor_cast/performance_model/calibration/profiles/a3_vllm018_cann85_profile.sqlite"
    )
    device = ATLAS_800.A3_752T_128G_DIE
    source = ProfileCalibrationDataSource(
        profile, device_name=device.name, device_compute_efficiency=device.compute_efficiency
    )
    result = CalibratedAnalyticPerformanceModel(AnalyticPerformanceModel(device), source).process_op(_attention_info())
    assert result.statistics["source"] == "ANALYTIC_RAW"
    assert result.statistics["calibration"]["profile_id"] is None


@pytest.mark.parametrize("mode", ["dense_prefill", "dense_decode", "sparse_limit", "sparse_indices"])
def test_mla_calibration_signature_preserves_runtime_and_hif8_cache_identity(mode):
    query_lens = (1, 1) if mode == "dense_decode" else (3, 4)
    tokens = sum(query_lens)
    prefill = 0 if mode == "dense_decode" else tokens
    sparse = mode.startswith("sparse")
    topk = 4 if mode == "sparse_limit" else None
    indices = torch.empty(tokens, 4, dtype=torch.int32, device="meta") if mode == "sparse_indices" else None
    args = (
        _tensor(tokens, 4, 32),
        _tensor(prefill, 4, 48),
        _tensor(tokens - prefill, 4, 32),
        _tensor(8, 16, 48, dtype=torch.uint8),
        torch.zeros(2, 4, dtype=torch.int32),
        torch.tensor([0, query_lens[0], tokens], dtype=torch.int32),
        torch.tensor([8, 24], dtype=torch.int32),
        torch.tensor(query_lens, dtype=torch.int32),
        32,
        32,
        topk,
        indices,
        _tensor(dtype=torch.float32),
    )
    op = torch.ops.tensor_cast.multihead_latent_attention_hif8_kv.default
    info = OpInvokeInfo(op, args, {}, op(*args))
    raw = AnalyticPerformanceModel(TEST_DEVICE).process_op(info)
    signature = build_calibration_signature(info, raw, device_name=TEST_DEVICE.name)
    features = signature.features
    assert signature.op_family == "attention"
    assert signature.dtype == "hif8"
    assert features["compute_dtype"] == "bfloat16"
    assert features["kv_cache_dtype"] == "hif8"
    assert features["quantization"] == "hif8_kv_bf16_compute"
    assert features["q_tokens"] == tokens
    assert features["heads"] == 4
    assert features["kv_heads"] == (1 if sparse or mode == "dense_decode" else 4)
    assert features["head_dim"] == (32 if sparse or mode == "dense_decode" else 16)
    assert features["phase"] == ("decode" if mode == "dense_decode" else "prefill")
    assert features["is_sparse"] == sparse
    assert features["kv_lengths"] == (8, 24)
    assert features["effective_kv_len"] == 16
    assert features["block_size"] == 16
    assert features["valid_blocks"] == (1, 2)
    if sparse:
        assert features["topk"] == 4
    else:
        assert "topk" not in features
