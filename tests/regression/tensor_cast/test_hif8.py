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

"""HIF8 storage, graph, performance and format isolation contracts."""

import copy
import math

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from tensor_cast.compilation import get_backend
from tensor_cast.compilation.passes.merge_linear_pass import MergeLinearPass
from tensor_cast.core.quantization.config import create_linear_quant_config, create_quant_config
from tensor_cast.core.model_runner import ModelRunner
from tensor_cast.device import ATLAS_800, TEST_DEVICE
from tensor_cast.layers.quant_linear import QuantLinearBase, TensorCastQuantLinear
from tensor_cast.model_config import LinearQuantConfig
from tensor_cast.utils import SimFormat
from tensor_cast.performance_model.analytic import AnalyticPerformanceModel
from tensor_cast.performance_model.calibration.signature import build_calibration_signature
from tensor_cast.performance_model.op_invoke_info import OpInvokeInfo
from tensor_cast.performance_model.profiling_database.data_source import QuerySource
from tensor_cast.performance_model.profiling_database.profiling_data_source import ProfilingDataSource
from tensor_cast.performance_model.profiling_database.interpolating_data_source import InterpolatingDataSource
from tensor_cast.quantize_utils import get_quant_config, LinearQuantType, QuantGranularity, QuantScheme
from tensor_cast.runtime import Runtime
from tensor_cast.utils import DTYPE_FP8, performance_dtype


def _layer(**kwargs):
    return TensorCastQuantLinear(
        torch.nn.Linear(16, 32, bias=False, dtype=torch.bfloat16, device="meta"),
        create_linear_quant_config("HIF8", **kwargs),
    )


def _info(m=3, k=16, n=32):
    x = torch.empty(m, k, dtype=torch.uint8, device="meta")
    w = torch.empty(k, n, dtype=torch.uint8, device="meta")
    xs = torch.empty(m, 1, device="meta")
    ws = torch.empty(n, device="meta")
    args = (x, w, xs, ws, None, torch.bfloat16)
    out = torch.ops.tensor_cast.hif8_linear(*args)
    return OpInvokeInfo(torch.ops.tensor_cast.hif8_linear.default, args, {}, out)


def test_hif8_identity_is_independent_of_payload_and_fp8():
    assert performance_dtype(SimFormat.HIF8) == SimFormat.HIF8
    assert performance_dtype(torch.uint8) == torch.uint8
    assert performance_dtype(torch.float8_e4m3fn) == DTYPE_FP8
    assert TEST_DEVICE.mma_ops[SimFormat.HIF8] > 0
    assert SimFormat.HIF8 not in ATLAS_800.A3_752T_128G_DIE.mma_ops


@pytest.mark.parametrize("shape", [(3, 16), (2, 3, 16), (0, 16)])
@pytest.mark.parametrize("static", [False, True])
def test_payload_scale_and_output_storage(shape, static):
    layer = _layer(**({"activation_scale": torch.tensor(1.0)} if static else {}))
    y = layer(torch.empty(shape, dtype=torch.bfloat16, device="meta"))
    assert y.shape == (*shape[:-1], 32)
    assert y.dtype == torch.bfloat16
    assert layer.qweight.dtype == torch.uint8
    assert layer.qweight.numel() * layer.qweight.element_size() == 16 * 32
    assert layer.weight_scale.shape == (32,)
    assert layer.weight_scale.dtype == torch.float32
    assert list(layer.parameters()) == []  # original BF16 weight is not retained


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"dynamic_quant_scheme": QuantScheme.ASYMMETRIC}, "symmetric"),
        ({"weight_offset": torch.tensor(1.0)}, "offset"),
        ({"weight_group_size": 32}, "block format"),
        ({"dynamic_quant_granularity": QuantGranularity.PER_GROUP}, "activations"),
        ({"out_dtype": torch.uint8}, "output"),
        ({"weight_scale": torch.ones(32, dtype=torch.bfloat16)}, "float32"),
    ],
)
def test_invalid_recipes_are_rejected(kwargs, match):
    with pytest.raises(ValueError, match=match):
        LinearQuantConfig(quant_type=LinearQuantType.HIF8, **kwargs)


@pytest.mark.parametrize(
    "granularity,scale_shape",
    [
        (QuantGranularity.PER_CHANNEL, ()),
        (QuantGranularity.PER_CHANNEL, (1,)),
        (QuantGranularity.PER_CHANNEL, (31,)),
        (QuantGranularity.PER_CHANNEL, (1, 32)),
        (QuantGranularity.PER_TENSOR, (32,)),
        (QuantGranularity.PER_TENSOR, (1,)),
    ],
)
def test_weight_scale_must_match_configured_granularity(granularity, scale_shape):
    with pytest.raises(ValueError, match="HIF8 weight_scale"):
        _layer(
            weight_quant_granularity=granularity,
            weight_scale=torch.empty(scale_shape, dtype=torch.float32),
        )


@pytest.mark.parametrize("granularity", [QuantGranularity.PER_CHANNEL, QuantGranularity.PER_TENSOR])
@pytest.mark.parametrize("out_features", [1, 32])
def test_explicit_weight_scale_preserves_granularity(granularity, out_features):
    scale_shape = (out_features,) if granularity == QuantGranularity.PER_CHANNEL else ()
    layer = TensorCastQuantLinear(
        torch.nn.Linear(16, out_features, bias=False, dtype=torch.bfloat16, device="meta"),
        create_linear_quant_config(
            "HIF8",
            weight_quant_granularity=granularity,
            weight_scale=torch.ones(scale_shape, dtype=torch.float32),
        ),
    )
    assert layer.weight_scale.shape == scale_shape
    assert layer.weight_scale.device.type == "meta"
    assert layer(torch.empty(3, 16, dtype=torch.bfloat16, device="meta")).shape == (3, out_features)


@pytest.mark.parametrize("weight_dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("out_dtype", [None, torch.float16, torch.bfloat16, torch.float32])
def test_external_hif8_payload_uses_floating_output_dtype(weight_dtype, out_dtype):
    layer = TensorCastQuantLinear(
        torch.nn.Linear(16, 32, bias=True, dtype=weight_dtype, device="meta"),
        create_linear_quant_config("HIF8", out_dtype=out_dtype),
    )
    payload, scale = torch.ops.tensor_cast.dynamic_quantize_hif8(torch.empty(3, 16, dtype=weight_dtype, device="meta"))
    assert layer.weight_dtype == weight_dtype
    assert layer.qweight.dtype == payload.dtype == torch.uint8
    result = layer(payload, scale)
    assert result.shape == (3, 32)
    assert result.dtype == (out_dtype if out_dtype is not None else weight_dtype)


@pytest.mark.parametrize("scale_mode", ["distinct_scalars", "shared_scalar", "mixed", "vectors"])
def test_hif8_merge_preserves_weight_scale_identity(scale_mode):
    def projections(x, w1, w2, xs, ws1, ws2):
        first = torch.ops.tensor_cast.hif8_linear(x, w1, xs, ws1, None, torch.bfloat16)
        second_scale = ws1 if scale_mode == "shared_scalar" else ws2
        second = torch.ops.tensor_cast.hif8_linear(x, w2, xs, second_scale, None, torch.bfloat16)
        return first, second

    args = (
        torch.empty(3, 16, dtype=torch.uint8, device="meta"),
        torch.empty(16, 32, dtype=torch.uint8, device="meta"),
        torch.empty(16, 64, dtype=torch.uint8, device="meta"),
        torch.empty(3, 1, device="meta"),
        torch.empty((32,) if scale_mode == "vectors" else (), device="meta"),
        torch.empty((64,) if scale_mode in ("mixed", "vectors") else (), device="meta"),
    )
    gm = make_fx(projections)(*args)
    original_scale_names = [
        node.args[3].name for node in gm.graph.nodes if node.target == torch.ops.tensor_cast.hif8_linear.default
    ]
    gm = MergeLinearPass()(gm)
    linear_nodes = [node for node in gm.graph.nodes if node.target == torch.ops.tensor_cast.hif8_linear.default]
    can_merge = scale_mode in ("shared_scalar", "vectors")
    assert len(linear_nodes) == (1 if can_merge else 2)
    if scale_mode == "shared_scalar":
        assert linear_nodes[0].args[3].name == original_scale_names[0]
    elif scale_mode == "vectors":
        scale_cat = linear_nodes[0].args[3]
        assert scale_cat.target == torch.ops.tensor_cast.cat.default
        assert [node.name for node in scale_cat.args[0]] == original_scale_names
    else:
        assert [node.args[3].name for node in linear_nodes] == original_scale_names
    assert [out.shape for out in gm(*args)] == [(3, 32), (3, 64)]


def test_real_weights_are_not_silently_integer_quantized():
    with pytest.raises(ValueError, match="numerical execution"):
        QuantLinearBase(torch.nn.Linear(16, 32), create_linear_quant_config("HIF8"))
    with pytest.raises(ValueError, match="numerical execution"):
        torch.ops.tensor_cast.dynamic_quantize_hif8(torch.randn(3, 16))


def test_nonexpert_override_and_mxfp4_kwargs_do_not_leak():
    cfg = create_quant_config(
        "MXFP4", "HIF8", weight_group_size=32, weight_quant_granularity=QuantGranularity.PER_GROUP
    )
    nonexpert = get_quant_config("model.layers.0.self_attn.q_proj", cfg, "default")
    expert = get_quant_config("model.layers.0.mlp.experts.0.gate_proj", cfg, "default")
    assert nonexpert.quant_type == LinearQuantType.HIF8
    assert nonexpert.weight_group_size is None
    assert expert.quant_type == LinearQuantType.MXFP4


def test_matmul_cost_and_rate_selection():
    info = _info()
    props = info.get_perf_properties()
    assert props.compute_ops[SimFormat.HIF8].mma_ops == 2 * 3 * 16 * 32
    assert torch.uint8 not in props.compute_ops
    assert props.memory_read_bytes == 3 * 16 + 16 * 32 + 4 * 3 + 4 * 32
    assert props.memory_write_bytes == 2 * 3 * 32
    assert props.memory_readwrite_bytes == 0
    result = AnalyticPerformanceModel(TEST_DEVICE).process_op(info)
    expected = 2 * 3 * 16 * 32 / (TEST_DEVICE.mma_ops[SimFormat.HIF8] * TEST_DEVICE.compute_efficiency)
    assert result.statistics["mma_ops_time_s"] == pytest.approx(expected)
    assert result.statistics["logical_format"] == "hif8"
    assert result.statistics["performance_source"] == "analytic"


@pytest.mark.parametrize("peak", [0, -1, float("nan"), float("inf")])
def test_missing_or_invalid_hif8_peak_fails_instead_of_dropping_compute(peak):
    device = copy.copy(TEST_DEVICE)
    device.mma_ops = {**TEST_DEVICE.mma_ops, SimFormat.HIF8: peak}
    with pytest.raises(ValueError, match="HIF8 MMA rate"):
        AnalyticPerformanceModel(device).process_op(_info())


@pytest.mark.parametrize("dtype,static_work", [(torch.float32, 48), (torch.float16, 60), (torch.bfloat16, 60)])
@pytest.mark.parametrize("mode", ["static", "per_token", "per_tensor"])
@pytest.mark.parametrize("shape", [(3, 4), (3, 1), (0, 4), (3, 0)])
def test_quantization_cost_respects_dtype_and_scale_groups(dtype, static_work, mode, shape):
    x = torch.empty(shape, dtype=dtype, device="meta")
    dynamic = mode != "static"
    if dynamic:
        op = torch.ops.tensor_cast.dynamic_quantize_hif8.default
        args = (x,)
        kwargs = {} if mode == "per_token" else {"per_token": False}
    else:
        op = torch.ops.tensor_cast.quantize_hif8.default
        args = (x, torch.empty((), dtype=torch.float32, device="meta"))
        kwargs = {}
    out = op(*args, **kwargs)
    info = OpInvokeInfo(op, args, kwargs, out)
    props = info.get_perf_properties()
    # For 12 elements: FP32 costs 12 scales + 24 clamps + 12 HIF8
    # conversions. FP16/BF16 also need 12 conversions to FP32. No integer
    # rounding/offset is charged for the uint8 storage carrier.
    expected = static_work * x.numel() // 12
    if dynamic:
        groups = out[1].numel()
        # An abs per element, exactly E-S max comparisons and S scale divisions.
        expected += x.numel() + (x.numel() - groups) + groups if x.numel() else 0
    assert sum(work.gp_ops for work in props.compute_ops.values()) == expected
    assert all(key == torch.float32 and work.mma_ops == 0 for key, work in props.compute_ops.items())
    assert props.memory_read_bytes == x.numel() * x.element_size() + (0 if dynamic else 4)
    assert props.memory_write_bytes == x.numel() + (4 * out[1].numel() if dynamic else 0)
    assert props.memory_readwrite_bytes == 0
    result = AnalyticPerformanceModel(TEST_DEVICE).process_op(info)
    expected_time = expected / (TEST_DEVICE.gp_ops[torch.float32] * TEST_DEVICE.compute_efficiency)
    assert result.statistics["gp_ops_time_s"] == pytest.approx(expected_time)


def test_calibration_uses_logical_hif8_and_scale_modes():
    info = _info()
    raw = AnalyticPerformanceModel(TEST_DEVICE).process_op(info)
    sig = build_calibration_signature(info, raw, device_name=TEST_DEVICE.name)
    assert sig.dtype == "hif8"
    assert sig.features["compute_dtype"] == "hif8"
    assert sig.features["activation_scale_mode"] == "per_token"
    assert sig.features["weight_scale_mode"] == "per_channel"


def test_unverified_profile_cannot_match_int8_or_fp8_even_via_interpolation():
    base = ProfilingDataSource.__new__(ProfilingDataSource)
    base._op_mapping = {"operator_mappings": {"tensor_cast.hif8_linear.default": {"zero_cost": True}}}
    assert base.lookup(_info()) is None
    assert base.last_miss_reason == "hif8_requires_verified_profile"
    wrapper = InterpolatingDataSource.__new__(InterpolatingDataSource)
    wrapper.base = base
    assert wrapper.lookup(_info()) is None


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("next_op", ["zero_cost", "accepted_miss", "communication"])
def test_hif8_profile_miss_does_not_leak_into_next_hit(tmp_path, wrapped, next_op):
    (tmp_path / "op_mapping.yaml").write_text(
        'version: "test"\ndevice: TEST_DEVICE\noperator_mappings:\n'
        '  "aten.view.default":\n    zero_cost: true\n'
        '  "aten.clone.default":\n    accepted_miss: "absorbed by fused kernel"\n'
        '  "tensor_cast.all_reduce.default":\n    category: communication\n    kernel_type: hcom_allReduce_\n'
    )
    (tmp_path / "hcom_allReduce_.csv").write_text(
        "message_bytes,num_devices,dtype,topology_tier,Duration(us)\n96,2,DT_BF16,0,12.5\n"
    )
    base = ProfilingDataSource(tmp_path)
    source = InterpolatingDataSource(base) if wrapped else base
    x = torch.empty(3, 16, dtype=torch.bfloat16, device="meta")
    if next_op == "communication":
        info = OpInvokeInfo(torch.ops.tensor_cast.all_reduce.default, (x, 0, [0, 1]), {}, x)
        expected_latency = 12.5
    elif next_op == "zero_cost":
        info = OpInvokeInfo(torch.ops.aten.view.default, (x, [3, 16]), {}, x)
        expected_latency = 0.0
    else:
        info = OpInvokeInfo(torch.ops.aten.clone.default, (x,), {}, x)
        expected_latency = 0.0

    assert source.lookup(_info()) is None
    assert source.last_miss_reason == "hif8_requires_verified_profile"
    result = source.lookup(info)
    assert result is not None
    assert result.source == QuerySource.MEASURED
    assert result.latency_us == pytest.approx(expected_latency)
    assert source.last_miss_reason == ""
    assert base.last_miss_reason == ""
    # The guard must still reject the next HIF8 lookup after a successful hit.
    assert source.lookup(_info()) is None
    assert source.last_miss_reason == "hif8_requires_verified_profile"


@pytest.mark.parametrize("per_token", [False, True])
@pytest.mark.parametrize("split_dim", [0, -2])
def test_compiled_experts_keep_token_and_channel_scales_in_order(per_token, split_dim):
    class Experts(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = torch.nn.ModuleList(
                [
                    _layer(
                        dynamic_quant_granularity=QuantGranularity.PER_SAMPLE
                        if per_token
                        else QuantGranularity.PER_TENSOR
                    )
                    for _ in range(3)
                ]
            )

        def forward(self, x):
            parts = x.split([3, 0, 2], dim=split_dim)
            return torch.cat([layer(part) for layer, part in zip(self.experts, parts)])

    torch.compiler.reset()
    model = torch.compile(Experts(), backend=get_backend(), fullgraph=True)
    with Runtime(AnalyticPerformanceModel(TEST_DEVICE), TEST_DEVICE) as runtime:
        out = model(torch.empty(5, 16, dtype=torch.bfloat16, device="meta"))
    assert out.shape == (5, 32)
    events = [e.op_invoke_info for e in runtime.event_list]
    gmms = [info for info in events if info.func == torch.ops.tensor_cast.grouped_matmul_hif8.default]
    # Per-tensor reductions stay inside each expert and cannot be sunk across
    # the token split. Per-token scales can follow the payload into GMM.
    assert bool(gmms) == per_token
    matmuls = [
        info
        for info in events
        if info.func in (torch.ops.tensor_cast.hif8_linear.default, torch.ops.tensor_cast.grouped_matmul_hif8.default)
    ]
    assert sum(info.get_perf_properties().compute_ops[SimFormat.HIF8].mma_ops for info in matmuls) == 2 * 5 * 16 * 32
    assert all(math.isfinite(e.perf_results["analytic"].execution_time_s) for e in runtime.event_list)


def test_empty_expert_and_keyword_arguments():
    info = _info(m=0)
    args = info.args
    out = torch.ops.tensor_cast.grouped_matmul_hif8([args[0]], [args[1]], [args[3]], [args[2]], [None], args[5])
    assert out.shape == (0, 32)
    keyword_info = OpInvokeInfo(
        info.func, (), dict(zip(("x", "w", "x_scale", "w_scale", "bias", "out_dtype"), args)), info.out
    )
    assert SimFormat.HIF8 not in keyword_info.get_perf_properties().compute_ops


def test_runner_rejects_unsupported_hardware_before_building_model():
    from types import SimpleNamespace

    user = SimpleNamespace(quantize_linear_action="HIF8")
    with pytest.raises(ValueError, match="HIF8 MMA rate"):
        ModelRunner.create_performance_models(user, ATLAS_800.A3_752T_128G_DIE)


def test_static_per_tensor_compilation_and_external_payload():
    torch.compiler.reset()
    layer = _layer(activation_scale=torch.tensor(1.0), weight_quant_granularity=QuantGranularity.PER_TENSOR)
    compiled = torch.compile(layer, backend=get_backend(), fullgraph=True)
    with Runtime(AnalyticPerformanceModel(TEST_DEVICE), TEST_DEVICE) as runtime:
        out = compiled(torch.empty(3, 16, dtype=torch.bfloat16, device="meta"))
    assert out.dtype == torch.bfloat16
    targets = {event.op_invoke_info.func for event in runtime.event_list}
    assert torch.ops.tensor_cast.quantize_hif8.default in targets
    assert torch.ops.tensor_cast.dynamic_quantize_hif8.default not in targets
    payload = torch.empty(3, 16, dtype=torch.uint8, device="meta")
    assert layer(payload, torch.empty(3, 1, device="meta")).dtype == torch.bfloat16


def test_calibration_preserves_one_token_scale_mode():
    info = _info(m=1)
    raw = AnalyticPerformanceModel(TEST_DEVICE).process_op(info)
    signature = build_calibration_signature(info, raw, device_name=TEST_DEVICE.name)
    assert signature.features["activation_scale_mode"] == "per_token"
