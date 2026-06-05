import types

import pytest
import torch
import torch.fx as fx
from tensor_cast.config import performance_model as perf_config
from tensor_cast.compilation.shape_prop import shape_propagation
from tensor_cast.device import TEST_DEVICE
from tensor_cast.model_config import AttentionQuantConfig, QuantConfig
from tensor_cast.performance_model.op_benchmark import (
    OpBenchmark,
    get_op_impl,
    register_op_impl,
)
from tensor_cast.performance_model.op_estimator_registry import (
    _op_estimator_table,
    get_op_estimator,
    register_op_estimator,
)
from tensor_cast.performance_model.op_invoke_info import OpInvokeInfo
from tensor_cast.quantize_utils import AttentionQuantType


class _NonDefaultEpsRMSNormModule(torch.nn.Module):
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(4, dtype=torch.float32))

    def _rms_norm(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states.to(input_dtype)

    def forward(self, hidden_states, residual):
        rms = self._rms_norm(hidden_states)
        add_rms = self._rms_norm(hidden_states + residual)
        added = hidden_states + residual
        add_rms2 = self._rms_norm(added)
        return rms, add_rms, add_rms2, added


def test_register_and_get_op_estimator():
    op_key = object()
    original = _op_estimator_table.get(None, {}).get(op_key)

    @register_op_estimator(op_key, None, True)
    def _estimator(op_invoke_info, device_profile):
        return "ok"

    assert get_op_estimator(op_key, None) is _estimator

    if original is None:
        _op_estimator_table[None].pop(op_key, None)
    else:
        _op_estimator_table[None][op_key] = original


def test_rms_norm_non_default_eps_path_consistency():
    module = _NonDefaultEpsRMSNormModule(eps=1e-6)
    hidden_states = torch.randn(2, 4, dtype=torch.float32)
    residual = torch.randn(2, 4, dtype=torch.float32)
    _, add_rms, add_rms2, added = module(hidden_states, residual)
    torch.testing.assert_close(add_rms, add_rms2, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(added, hidden_states + residual, rtol=0.0, atol=0.0)


def test_quant_attention_config_can_target_single_layer():
    quant_config = QuantConfig()
    attn_config = AttentionQuantConfig(
        quant_type=AttentionQuantType.INT8,
        query_scale=torch.tensor(1.0),
        kv_scale=torch.tensor(1.0),
        attention_prob_scale=torch.tensor(1.0),
    )
    quant_config.attention_configs[0] = attn_config
    assert 0 in quant_config.attention_configs
    assert quant_config.attention_configs[0].quant_type == AttentionQuantType.INT8


def test_multistream_count_nodes_helper_behavior():
    graph = fx.Graph()
    x = graph.placeholder("x")
    y = graph.call_function(torch.ops.aten.neg.default, args=(x,))
    graph.output(y)
    gm = fx.GraphModule({}, graph)
    count = sum(1 for node in gm.graph.nodes if node.target == torch.ops.aten.neg.default)
    assert count == 1


def test_grouped_matmul_meta_ops_preserve_shapes_and_dtype():
    x = [torch.empty((2, 3), device="meta"), torch.empty((1, 3), device="meta")]
    w = [torch.empty((3, 4), device="meta"), torch.empty((3, 4), device="meta")]
    bias = [None, torch.empty((4,), device="meta")]
    scales = [torch.empty((1,), device="meta"), torch.empty((1,), device="meta")]

    assert torch.ops.tensor_cast.grouped_matmul.default(x, w, bias).shape == (3, 4)
    quant_out = torch.ops.tensor_cast.grouped_matmul_quant.default(
        x,
        w,
        scales,
        [None, None],
        scales,
        [None, None],
        bias,
        None,
    )
    assert quant_out.shape == (3, 4)
    int4_out = torch.ops.tensor_cast.grouped_matmul_quant_int4.default(
        x,
        w,
        scales,
        [None, None],
        scales,
        [None, None],
        bias,
        torch.float16,
    )
    assert int4_out.dtype == torch.float16
    assert torch.ops.tensor_cast.grouped_matmul_fp8.default(x, w, scales, scales, bias, torch.bfloat16).dtype == (
        torch.bfloat16
    )
    assert torch.ops.tensor_cast.grouped_matmul_mxfp4.default(x, w, scales, scales, bias, None).dtype == torch.float32
    assert torch.ops.tensor_cast.grouped_matmul_swiglu.default([], [], []).shape == (
        0,
        0,
    )
    assert torch.ops.tensor_cast.grouped_matmul_quant_swiglu.default([], [], [], [], [], [], [], None).dtype == (
        torch.float32
    )
    assert torch.ops.tensor_cast.grouped_matmul_fp8_swiglu.default([], [], [], [], [], None).shape == (0, 0)


def test_communication_meta_ops_compute_collective_shapes(monkeypatch):
    x = torch.empty((4, 3), device="meta")

    assert torch.ops.tensor_cast.all_to_all.default(x, [1, 3], [2, 2], 0, [0, 1]).shape == (4, 3)
    assert torch.ops.tensor_cast.all_reduce.default(x, 0, [0, 1]).shape == x.shape
    assert torch.ops.tensor_cast.reduce_scatter.default(x, 0, 0, [0, 1]).shape == (2, 3)
    assert torch.ops.tensor_cast.all_gather.default(x, 1, 0, [0, 1]).shape == (4, 6)
    matmul_out = torch.ops.tensor_cast.matmul_all_reduce.default(x, torch.empty((3, 5), device="meta"), None, 0, [0])
    assert matmul_out.shape == (4, 5)

    linear_out = torch.empty((4, 5), device="meta", dtype=torch.float16)
    monkeypatch.setattr(torch.ops.tensor_cast.static_quant_linear, "default", lambda *args: linear_out)
    monkeypatch.setattr(
        torch.ops.tensor_cast.static_quant_linear_int4,
        "default",
        lambda *args: linear_out,
    )
    monkeypatch.setattr(torch.ops.tensor_cast.fp8_linear, "default", lambda *args: linear_out)
    monkeypatch.setattr(torch.ops.tensor_cast.mxfp4_linear, "default", lambda *args: linear_out)

    quant_args = (
        x,
        torch.empty((3, 5), device="meta"),
        torch.empty((1,), device="meta"),
        None,
        None,
        None,
        None,
        None,
        0,
        [0],
    )
    assert torch.ops.tensor_cast.static_quant_linear_all_reduce.default(*quant_args).shape == (4, 5)
    assert torch.ops.tensor_cast.static_quant_linear_int4_all_reduce.default(*quant_args).dtype == torch.float16
    fp_args = (
        x,
        torch.empty((3, 5), device="meta"),
        torch.empty((1,), device="meta"),
        torch.empty((1,), device="meta"),
        None,
        None,
        0,
        [0],
    )
    assert torch.ops.tensor_cast.fp8_linear_all_reduce.default(*fp_args).shape == (4, 5)
    assert torch.ops.tensor_cast.mxfp4_linear_all_reduce.default(*fp_args).shape == (
        4,
        5,
    )


def test_shape_propagation_records_tensor_metadata():
    class Tiny(torch.nn.Module):
        def forward(self, x):
            return x + 1

    gm = fx.symbolic_trace(Tiny())
    result = shape_propagation(gm, [torch.empty((2, 3), device="meta")])

    output_node = next(node for node in result.graph.nodes if node.op == "output")
    produced_node = output_node.args[0]
    assert tuple(produced_node.meta["tensor_meta"].shape) == (2, 3)


def test_op_benchmark_registry_runtime_and_quantize(monkeypatch):
    quantize_impl = get_op_impl(torch.ops.tensor_cast.quantize.default, torch.device("cpu"))
    x = torch.tensor([1.1, 2.1])
    scale = torch.tensor([1.0, 1.0])
    assert torch.equal(
        quantize_impl(x, scale, torch.tensor([1.0, -1.0])),
        torch.tensor([2, 1], dtype=torch.int8),
    )

    op_name = "unit_test_op"
    register_op_impl(op_name, "cpu")(lambda tensor: tensor)
    with pytest.raises(ValueError, match="already registered"):
        register_op_impl(op_name, "cpu")(lambda tensor: tensor)

    benchmark = OpBenchmark(TEST_DEVICE)
    assert benchmark.runtime_device == torch.device("cpu")
    monkeypatch.setattr(perf_config.empirical, "warmup_runs", 0)
    monkeypatch.setattr(perf_config.empirical, "benchmark_runs", 1)
    result = benchmark.do_bench(lambda tensor: tensor + 1, (torch.empty((2, 2), device="meta"),), {})
    assert result.execution_time_s >= 0

    monkeypatch.setattr(perf_config.empirical, "runtime_device_override", torch.device("cpu"))
    try:
        assert OpBenchmark(TEST_DEVICE).infer_runtime_device() == torch.device("cpu")
    finally:
        monkeypatch.setattr(perf_config.empirical, "runtime_device_override", None)

    class FakeTensorCastOp:
        namespace = "tensor_cast"
        is_view = False

    fake_func = FakeTensorCastOp()
    info = OpInvokeInfo(fake_func, (), {}, None, cache_key="unit")
    with pytest.raises(ValueError, match="No implementation registered"):
        benchmark.benchmark(info)


def test_op_benchmark_handles_non_tensor_cast_ops(monkeypatch):
    benchmark = OpBenchmark(TEST_DEVICE)
    monkeypatch.setattr(benchmark, "do_bench", lambda op_impl, args, kwargs: op_impl(*args, **kwargs))
    info = types.SimpleNamespace(
        func=torch.ops.aten.neg.default,
        args=(torch.tensor([1.0]),),
        kwargs={},
    )

    assert torch.equal(benchmark.benchmark(info), torch.tensor([-1.0]))
