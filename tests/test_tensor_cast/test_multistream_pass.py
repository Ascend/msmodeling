import contextlib
import unittest

import torch
import torch.fx as fx
from torch._subclasses.fake_tensor import DynamicOutputShapeException, FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv

from tensor_cast import config, ops  # noqa: F401
from tensor_cast.compilation.compile_backend import CompilerBackend
from tensor_cast.compilation.passes.multistream_pass import MultiStreamSchedulePass
from tensor_cast.device import TEST_DEVICE


def _count_nodes(gm: torch.fx.GraphModule, target) -> int:
    return sum(1 for node in gm.graph.nodes if node.target == target)


@contextlib.contextmanager
def _override_multistream_config(**overrides):
    multistream_config = config.compilation.multistream
    original_values = {field: getattr(multistream_config, field) for field in overrides}
    try:
        for field, value in overrides.items():
            setattr(multistream_config, field, value)
        yield
    finally:
        for field, value in original_values.items():
            setattr(multistream_config, field, value)


def _apply_multistream_pass(
    model,
    inputs,
    *,
    role_to_stream_ids=...,
    compute_stream_id=...,
    comm_stream_id=...,
    cross_stream_sync_overhead_s=...,
):
    gm = fx.symbolic_trace(model)
    backend = CompilerBackend(device_name=TEST_DEVICE.name)
    overrides = {"enable": True}
    if role_to_stream_ids is not ...:
        overrides["role_to_stream_ids"] = role_to_stream_ids
    if compute_stream_id is not ...:
        overrides["compute_stream_id"] = compute_stream_id
    if comm_stream_id is not ...:
        overrides["comm_stream_id"] = comm_stream_id
    if cross_stream_sync_overhead_s is not ...:
        overrides["cross_stream_sync_overhead_s"] = cross_stream_sync_overhead_s

    with _override_multistream_config(**overrides):
        backend.apply_multistream_pass(gm, inputs)

    return gm


def _build_mla_graph():
    graph = fx.Graph()
    q = graph.placeholder("q")
    kv_cache = graph.placeholder("kv_cache")
    block_table = graph.placeholder("block_table")
    query_start_loc = graph.placeholder("query_start_loc")
    seq_lens = graph.placeholder("seq_lens")
    query_lens = graph.placeholder("query_lens")
    w_uk_t = graph.placeholder("w_uk_t")
    w_uv = graph.placeholder("w_uv")
    kv_b_proj = graph.placeholder("kv_b_proj")
    out = graph.call_function(
        torch.ops.tensor_cast.multihead_latent_attention.default,
        args=(
            q,
            kv_cache,
            block_table,
            query_start_loc,
            seq_lens,
            query_lens,
            w_uk_t,
            w_uv,
            kv_b_proj,
            64,
        ),
    )
    graph.output(out)
    return fx.GraphModule({}, graph), out


def _build_unary_graph(target):
    graph = fx.Graph()
    x = graph.placeholder("x")
    out = graph.call_function(target, args=(x,))
    graph.output(out)
    return fx.GraphModule({}, graph), out


class MultiStreamPassTestCase(unittest.TestCase):
    def setUp(self):
        torch.compiler.reset()

    def test_multistream_pass_injects_anchor_ops(self):
        class ToyGraph(torch.nn.Module):
            def forward(self, x):
                compute = torch.ops.aten.neg.default(x)
                comm = torch.ops.tensor_cast.all_reduce.default(x, 0, [0, 1])
                return torch.ops.aten.add.Tensor(compute, comm)

        inputs = (torch.empty((8, 8), dtype=torch.float16, device="meta"),)
        gm = _apply_multistream_pass(ToyGraph(), inputs)

        self.assertEqual(_count_nodes(gm, torch.ops.tensor_cast._internal_wait_and_bind.default), 3)
        self.assertEqual(_count_nodes(gm, torch.ops.tensor_cast._internal_record.default), 1)
        dep_wait_count = sum(
            1
            for node in gm.graph.nodes
            if node.target == torch.ops.tensor_cast._internal_wait_and_bind.default
            and len(node.args) >= 3
            and len(node.args[2]) > 0
        )
        self.assertEqual(dep_wait_count, 2)

        record_nodes = [
            node for node in gm.graph.nodes if node.target == torch.ops.tensor_cast._internal_record.default
        ]
        self.assertEqual(len(record_nodes), 1)
        self.assertEqual(record_nodes[0].args[0].target, torch.ops.tensor_cast.all_reduce.default)
        self.assertEqual(record_nodes[0].args[1], 1)

    def test_multistream_pass_lowers_hybrid_and_comm_with_graph_nodes(self):
        class ToyGraph(torch.nn.Module):
            def forward(self, x, w):
                compute = torch.ops.aten.neg.default(x)
                hybrid = torch.ops.tensor_cast.matmul_all_reduce.default(x, w, None, 0, [0, 1])
                comm = torch.ops.tensor_cast.all_reduce.default(x, 0, [0, 1])
                mixed = torch.ops.aten.add.Tensor(compute, hybrid)
                return torch.ops.aten.add.Tensor(mixed, comm)

        x = torch.empty((8, 8), dtype=torch.float16, device="meta")
        w = torch.empty((8, 8), dtype=torch.float16, device="meta")
        gm = _apply_multistream_pass(ToyGraph(), (x, w))

        record_nodes = [
            node for node in gm.graph.nodes if node.target == torch.ops.tensor_cast._internal_record.default
        ]
        self.assertEqual(len(record_nodes), 1)

        hybrid_record_nodes = [
            node for node in record_nodes if node.args[0].target == torch.ops.tensor_cast.matmul_all_reduce.default
        ]
        self.assertEqual(len(hybrid_record_nodes), 0)

        comm_record_nodes = [
            node for node in record_nodes if node.args[0].target == torch.ops.tensor_cast.all_reduce.default
        ]
        self.assertEqual(len(comm_record_nodes), 1)
        self.assertEqual(comm_record_nodes[0].args[1], 1)

    def test_multistream_guard_skips_when_no_gain(self):
        class ChainGraph(torch.nn.Module):
            def forward(self, x):
                y = torch.ops.aten.neg.default(x)
                z = torch.ops.tensor_cast.all_reduce.default(y, 0, [0, 1])
                return torch.ops.aten.relu.default(z)

        inputs = (torch.empty((8, 8), dtype=torch.float16, device="meta"),)
        gm = _apply_multistream_pass(ChainGraph(), inputs)
        self.assertEqual(_count_nodes(gm, torch.ops.tensor_cast._internal_wait_and_bind.default), 0)
        self.assertEqual(_count_nodes(gm, torch.ops.tensor_cast._internal_record.default), 0)

    def test_wait_anchor_preserves_value_without_aliasing_input(self):
        x = torch.randn((2, 3), dtype=torch.float32)

        y = torch.ops.tensor_cast._internal_wait_and_bind.default(x, 0, [])

        self.assertTrue(torch.equal(y, x))
        self.assertNotEqual(y.data_ptr(), x.data_ptr())

    def test_analytic_cost_fallback_does_not_leak_unbacked_symbols(self):
        gm, mla_node = _build_mla_graph()
        shape_env = ShapeEnv()
        pass_ = MultiStreamSchedulePass(device_name=TEST_DEVICE.name)

        with FakeTensorMode(shape_env=shape_env, allow_non_fake_inputs=True):
            values = {
                "q": torch.empty((100, 128, 576), dtype=torch.float16),
                "kv_cache": torch.empty((10000, 128, 576), dtype=torch.float16),
                "block_table": torch.empty((1, 10), dtype=torch.int64),
                "query_start_loc": torch.empty((2,), dtype=torch.int64),
                "seq_lens": torch.empty((1,), dtype=torch.int64),
                "query_lens": torch.empty((1,), dtype=torch.int64),
                "w_uk_t": torch.empty((128, 512, 512), dtype=torch.float16),
                "w_uv": torch.empty((128, 512, 64), dtype=torch.float16),
                "kv_b_proj": torch.empty((512, 128 * (64 + 64)), dtype=torch.float16),
            }
            for node in gm.graph.nodes:
                if node.op == "placeholder":
                    node.meta["val"] = values[node.name]
            mla_node.meta["val"] = torch.empty((100, 128, 64), dtype=torch.float16)

            cost = pass_._estimate_node_cost_with_analytic(mla_node)

        self.assertIsNone(cost)
        self.assertEqual(len(shape_env.pending_fresh_unbacked_symbols), 0)

    def test_analytic_cost_falls_back_for_optional_none_args(self):
        graph = fx.Graph()
        q = graph.placeholder("q")
        query_start_loc = graph.placeholder("query_start_loc")
        seq_lens = graph.placeholder("seq_lens")
        query_lens = graph.placeholder("query_lens")
        w_uk_t = graph.placeholder("w_uk_t")
        w_uv = graph.placeholder("w_uv")
        out = graph.call_function(
            torch.ops.tensor_cast.multihead_latent_attention.default,
            args=(
                q,
                None,
                None,
                query_start_loc,
                seq_lens,
                query_lens,
                w_uk_t,
                w_uv,
                None,
                64,
            ),
        )
        graph.output(out)
        gm = fx.GraphModule({}, graph)
        pass_ = MultiStreamSchedulePass(device_name=TEST_DEVICE.name)

        for node in gm.graph.nodes:
            if node.op == "placeholder":
                node.meta["val"] = torch.empty((1,), dtype=torch.float16)
        q.meta["val"] = torch.empty((1, 128, 576), dtype=torch.float16)
        w_uk_t.meta["val"] = torch.empty((128, 512, 512), dtype=torch.float16)
        w_uv.meta["val"] = torch.empty((128, 512, 64), dtype=torch.float16)
        out.meta["val"] = torch.empty((1, 128, 64), dtype=torch.float16)

        cost = pass_._estimate_node_cost_with_analytic(out)

        self.assertIsNone(cost)

    def test_dynamic_output_shape_falls_back_to_heuristic(self):
        gm, node = _build_unary_graph(torch.ops.aten.neg.default)
        shape_env = ShapeEnv()
        pass_ = MultiStreamSchedulePass(device_name=TEST_DEVICE.name)

        def dynamic_output_estimator(op_invoke_info):
            raise DynamicOutputShapeException(torch.ops.aten.nonzero.default)

        pass_._analytic_model.process_op = dynamic_output_estimator

        with FakeTensorMode(shape_env=shape_env, allow_non_fake_inputs=True):
            for graph_node in gm.graph.nodes:
                if graph_node.op == "placeholder":
                    graph_node.meta["val"] = torch.empty((4,), dtype=torch.int64)
            node.meta["val"] = torch.empty((4,), dtype=torch.int64)

            cost = pass_._estimate_node_cost_with_analytic(node)

        self.assertIsNone(cost)
        self.assertEqual(len(shape_env.pending_fresh_unbacked_symbols), 0)


if __name__ == "__main__":
    unittest.main()
