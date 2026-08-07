import types

import pytest
import torch
import torch.fx as fx
from tensor_cast.config import performance_model as perf_config
from tensor_cast.compilation.shape_prop import shape_propagation
from tensor_cast.device import TEST_DEVICE
from tensor_cast.model_config import (
    AttentionBackend,
    AttentionQuantConfig,
    AttentionRoutePlan,
    DEFAULT_BLOCK_SPARSE_ATTENTION_BLOCK_SIZE,
    QuantConfig,
)
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


def _semantic_op_properties(op, *args):
    out = op(*args)
    return out, OpInvokeInfo(op, args, {}, out).get_perf_properties()


@pytest.mark.parametrize(
    ("op", "extra_args"),
    (
        (torch.ops.tensor_cast.rms_norm.default, (None, 1e-6)),
        (torch.ops.tensor_cast.layer_norm.default, (None, None, 1e-6)),
        (
            torch.ops.tensor_cast.modulated_layer_norm.default,
            (
                None,
                None,
                torch.empty((2, 1, 0), device="meta", dtype=torch.bfloat16),
                torch.empty((2, 1, 0), device="meta", dtype=torch.bfloat16),
                1e-6,
            ),
        ),
    ),
)
def test_norm_properties_support_zero_width(op, extra_args):
    x = torch.empty((2, 3, 0), device="meta", dtype=torch.bfloat16)

    out, properties = _semantic_op_properties(op, x, *extra_args)

    assert out.shape == x.shape
    assert properties.compute_ops == {}
    assert properties.memory_read_bytes == 0
    assert properties.memory_write_bytes == 0


@pytest.mark.parametrize(
    ("op", "extra_args"),
    (
        (torch.ops.tensor_cast.rms_norm.default, (None, 1e-6)),
        (torch.ops.tensor_cast.layer_norm.default, (None, None, 1e-6)),
        (
            torch.ops.tensor_cast.modulated_layer_norm.default,
            (
                None,
                None,
                torch.empty((), device="meta", dtype=torch.bfloat16),
                torch.empty((), device="meta", dtype=torch.bfloat16),
                1e-6,
            ),
        ),
    ),
)
def test_norm_properties_reject_scalar_inputs(op, extra_args):
    x = torch.empty((), device="meta", dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="at least one dimension"):
        _semantic_op_properties(op, x, *extra_args)


def test_semantic_fusion_ops_preserve_meta_contracts_and_roofline_properties():
    x = torch.empty((2, 3, 4), device="meta", dtype=torch.bfloat16)
    weight = torch.empty((4,), device="meta", dtype=torch.float32)
    bias = torch.empty((4,), device="meta", dtype=torch.float32)

    expected_layer_norm_cases = [
        (None, None, 144, 48),
        (weight, None, 168, 64),
        (None, bias, 168, 64),
        (weight, bias, 192, 80),
    ]
    for layer_weight, layer_bias, expected_gp_ops, expected_read_bytes in expected_layer_norm_cases:
        out, properties = _semantic_op_properties(
            torch.ops.tensor_cast.layer_norm.default,
            x,
            layer_weight,
            layer_bias,
            1e-6,
        )
        assert out.shape == x.shape
        assert out.dtype == x.dtype
        assert properties.compute_ops[torch.float32].gp_ops == expected_gp_ops
        assert properties.memory_read_bytes == expected_read_bytes
        assert properties.memory_write_bytes == 48
        assert properties.memory_readwrite_bytes == 0

    scale = torch.empty((2, 1, 4), device="meta", dtype=torch.bfloat16)
    shift = torch.empty((2, 1, 4), device="meta", dtype=torch.bfloat16)
    out, properties = _semantic_op_properties(
        torch.ops.tensor_cast.modulated_layer_norm.default,
        x,
        weight,
        bias,
        scale,
        shift,
        1e-6,
    )
    assert out.shape == x.shape
    assert properties.compute_ops[torch.float32].gp_ops == 192
    assert properties.compute_ops[torch.bfloat16].gp_ops == 72
    assert properties.memory_read_bytes == 112
    assert properties.memory_write_bytes == 48

    for approximate, expected_gp_ops in (("none", 384), ("tanh", 312)):
        out, properties = _semantic_op_properties(torch.ops.tensor_cast.gelu.default, x, approximate)
        assert out.shape == x.shape
        assert out.dtype == x.dtype
        assert properties.compute_ops[torch.float32].gp_ops == expected_gp_ops
        assert properties.memory_read_bytes == 48
        assert properties.memory_write_bytes == 48

    residual = torch.empty((1, 3, 1), device="meta", dtype=torch.float16)
    update = torch.empty((2, 1, 4), device="meta", dtype=torch.float32)
    gate = torch.empty((1, 3, 4), device="meta", dtype=torch.bfloat16)
    out, properties = _semantic_op_properties(torch.ops.tensor_cast.gated_residual_add.default, residual, update, gate)
    assert out.shape == (2, 3, 4)
    assert out.dtype == torch.float32
    assert properties.compute_ops[torch.float32].gp_ops == 48
    assert properties.memory_read_bytes == 62
    assert properties.memory_write_bytes == 96

    query = torch.empty((2, 3, 4), device="meta", dtype=torch.bfloat16)
    key = torch.empty((2, 3, 4), device="meta", dtype=torch.float32)
    cos = torch.empty((3, 4), device="meta", dtype=torch.float32)
    sin = torch.empty((3, 4), device="meta", dtype=torch.float32)
    (query_out, key_out), properties = _semantic_op_properties(
        torch.ops.tensor_cast.apply_rope.default,
        query,
        key,
        cos,
        sin,
        False,
    )
    assert query_out.dtype == torch.bfloat16
    assert key_out.dtype == torch.float32
    assert torch.bfloat16 not in properties.compute_ops
    assert properties.compute_ops[torch.float32].gp_ops == 144
    assert properties.memory_read_bytes == 240
    assert properties.memory_write_bytes == 144

    out, properties = _semantic_op_properties(
        torch.ops.tensor_cast.apply_rope_single.default,
        query,
        cos,
        sin,
        False,
        False,
    )
    assert out.shape == query.shape
    assert out.dtype == query.dtype
    assert torch.bfloat16 not in properties.compute_ops
    assert properties.compute_ops[torch.float32].gp_ops == 72
    assert properties.memory_read_bytes == 144
    assert properties.memory_write_bytes == 48

    out, properties = _semantic_op_properties(
        torch.ops.tensor_cast.apply_rope_single.default,
        query,
        cos.to(torch.bfloat16),
        sin.to(torch.bfloat16),
        False,
        False,
    )
    assert out.dtype == query.dtype
    assert torch.bfloat16 not in properties.compute_ops
    assert properties.compute_ops[torch.float32].gp_ops == 72
    assert properties.memory_read_bytes == 96
    assert properties.memory_write_bytes == 48


def test_block_sparse_attention_route_plan_and_semantic_ops():
    route_plan = AttentionRoutePlan(backend="block_sparse_attention")
    assert route_plan.backend is AttentionBackend.block_sparse_attention
    assert route_plan.block_size == DEFAULT_BLOCK_SPARSE_ATTENTION_BLOCK_SIZE
    with pytest.raises(ValueError, match="block size"):
        AttentionRoutePlan(block_size=0)
    with pytest.raises(ValueError, match="sparsity"):
        AttentionRoutePlan(sparsity=1.0)
    with pytest.raises(ValueError, match="dense attention.*sparsity"):
        AttentionRoutePlan(sparsity=0.5)
    with pytest.raises(ValueError, match="dense attention.*block size"):
        AttentionRoutePlan(block_size=64)

    query = torch.empty((1, 5, 2, 4), device="meta", dtype=torch.float16)
    attention_mask = torch.empty((1, 1, 5, 5), device="meta", dtype=torch.bool)
    route_metadata, route_properties = _semantic_op_properties(
        torch.ops.tensor_cast.attention_route_generate.default,
        query,
        query,
        4,
        0.5,
    )
    assert route_metadata.shape == (1, 2, 2, 2)
    assert route_metadata.dtype == torch.int32
    assert route_properties.compute_ops[torch.float16].gp_ops == 72
    assert route_properties.memory_write_bytes == 32

    dense_out, dense_properties = _semantic_op_properties(
        torch.ops.tensor_cast.block_sparse_attention.default,
        query,
        query,
        query,
        None,
        route_metadata,
        4,
        0.0,
    )
    sparse_out, sparse_properties = _semantic_op_properties(
        torch.ops.tensor_cast.block_sparse_attention.default,
        query,
        query,
        query,
        None,
        route_metadata,
        4,
        0.5,
    )
    masked_out, masked_properties = _semantic_op_properties(
        torch.ops.tensor_cast.block_sparse_attention.default,
        query,
        query,
        query,
        attention_mask,
        route_metadata,
        4,
        0.5,
    )
    assert dense_out.shape == sparse_out.shape == masked_out.shape == query.shape
    assert dense_out.dtype == sparse_out.dtype == masked_out.dtype == query.dtype
    assert (
        sparse_properties.compute_ops[torch.float16].mma_ops * 2 == dense_properties.compute_ops[torch.float16].mma_ops
    )
    assert sparse_properties.compute_ops[torch.float16].gp_ops * 2 == dense_properties.compute_ops[torch.float16].gp_ops
    assert sparse_properties.memory_read_bytes == 272
    assert sparse_properties.memory_write_bytes == 80
    assert masked_properties.memory_read_bytes == 297
    assert masked_properties.memory_write_bytes == 80


@pytest.mark.parametrize(
    ("block_size", "query_len", "key_len", "sparsity", "expected_mma", "expected_gp"),
    (
        (4, 9, 17, 0.5, 4608, 1152),
        (8, 9, 17, 0.5, 8192, 2048),
    ),
)
def test_block_sparse_attention_charges_padded_block_interactions(
    block_size, query_len, key_len, sparsity, expected_mma, expected_gp
):
    query = torch.empty((1, query_len, 2, 4), device="meta", dtype=torch.float16)
    route_metadata = torch.empty(
        (1, 2, (query_len + block_size - 1) // block_size, (key_len + block_size - 1) // block_size),
        device="meta",
        dtype=torch.int32,
    )

    _, properties = _semantic_op_properties(
        torch.ops.tensor_cast.block_sparse_attention.default,
        query,
        torch.empty((1, key_len, 2, 4), device="meta", dtype=torch.float16),
        torch.empty((1, key_len, 2, 4), device="meta", dtype=torch.float16),
        None,
        route_metadata,
        block_size,
        sparsity,
    )

    assert properties.compute_ops[torch.float16].mma_ops == expected_mma
    assert properties.compute_ops[torch.float16].gp_ops == expected_gp


def test_block_sparse_attention_rounds_retained_blocks_at_decimal_boundary():
    block_size = 4
    query = torch.empty((1, 5, 1, 2), device="meta", dtype=torch.float16)
    route_metadata = torch.empty((1, 1, 2, 50), device="meta", dtype=torch.int32)

    _, properties = _semantic_op_properties(
        torch.ops.tensor_cast.block_sparse_attention.default,
        query,
        torch.empty((1, 200, 1, 2), device="meta", dtype=torch.float16),
        torch.empty((1, 200, 1, 2), device="meta", dtype=torch.float16),
        None,
        route_metadata,
        block_size,
        0.58,
    )

    # ceil(50 * (1 - 0.58)) is mathematically 21, despite binary float artifacts.
    assert properties.compute_ops[torch.float16].mma_ops == 5376
    assert properties.compute_ops[torch.float16].gp_ops == 2688


def test_block_sparse_attention_memory_is_independent_of_sparsity():
    query = torch.empty((1, 5, 2, 4), device="meta", dtype=torch.float16)
    key = torch.empty((1, 9, 2, 4), device="meta", dtype=torch.float16)
    route_metadata = torch.empty((1, 2, 2, 3), device="meta", dtype=torch.int32)
    op = torch.ops.tensor_cast.block_sparse_attention.default

    properties = [
        _semantic_op_properties(op, query, key, key, None, route_metadata, 4, sparsity)[1]
        for sparsity in (0.0, 0.5, 0.9)
    ]

    assert [(p.memory_read_bytes, p.memory_write_bytes) for p in properties] == [(416, 80)] * 3


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float32))
def test_block_sparse_attention_softmax_uses_query_dtype(dtype):
    query = torch.empty((1, 5, 2, 4), device="meta", dtype=dtype)
    route_metadata = torch.empty((1, 2, 2, 2), device="meta", dtype=torch.int32)

    _, properties = _semantic_op_properties(
        torch.ops.tensor_cast.block_sparse_attention.default,
        query,
        query,
        query,
        None,
        route_metadata,
        4,
        0.5,
    )

    assert properties.compute_ops[dtype].gp_ops > 0
    assert torch.half not in properties.compute_ops


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
