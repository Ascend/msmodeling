import contextlib
import unittest

import torch
import torch.fx as fx
from tensor_cast import config
from tensor_cast.compilation.compile_backend import CompilerBackend
from tensor_cast.compilation.passes.multistream_pass import MultiStreamSchedulePass
from tensor_cast.device import TEST_DEVICE
import tensor_cast.performance_model.builtin_model  # noqa: F401 - register builtin op cost handlers
from tensor_cast.performance_model import _mla_metadata_attn_len
from tensor_cast.performance_model.op_invoke_info import OpInvokeInfo
from torch._subclasses.fake_tensor import DynamicOutputShapeException, FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv

# Core multistream node-count assertions were moved to the unified entry in test_ops.py.


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


def _build_attention_graph():
    graph = fx.Graph()
    query = graph.placeholder("query")
    key = graph.placeholder("key")
    value = graph.placeholder("value")
    attention_mask = graph.placeholder("attention_mask")
    block_table = graph.placeholder("block_table")
    query_start_loc = graph.placeholder("query_start_loc")
    seq_lens = graph.placeholder("seq_lens")
    query_lens = graph.placeholder("query_lens")
    out = graph.call_function(
        torch.ops.tensor_cast.attention.default,
        args=(
            query,
            key,
            value,
            attention_mask,
            block_table,
            query_start_loc,
            seq_lens,
            query_lens,
        ),
    )
    graph.output(out)
    return fx.GraphModule({}, graph), out


def _build_sparse_attention_graph():
    graph = fx.Graph()
    q = graph.placeholder("q")
    kv = graph.placeholder("kv")
    attn_sink = graph.placeholder("attn_sink")
    topk_indices = graph.placeholder("topk_indices")
    out = graph.call_function(
        torch.ops.tensor_cast.sparse_attn_sharedkv.default,
        args=(q, kv, attn_sink, topk_indices, 0.125, 128),
    )
    graph.output(out)
    return fx.GraphModule({}, graph), out


def _build_dsa_indexer_graph():
    graph = fx.Graph()
    hidden_states = graph.placeholder("hidden_states")
    qa_normed = graph.placeholder("qa_normed")
    cos = graph.placeholder("cos")
    sin = graph.placeholder("sin")
    indexer_cache = graph.placeholder("indexer_cache")
    slot_mapping = graph.placeholder("slot_mapping")
    block_tables = graph.placeholder("block_tables")
    seq_lens = graph.placeholder("seq_lens")
    wq_b_weight = graph.placeholder("wq_b_weight")
    wk_weight = graph.placeholder("wk_weight")
    weights_proj_weight = graph.placeholder("weights_proj_weight")
    k_norm_weight = graph.placeholder("k_norm_weight")
    out = graph.call_function(
        torch.ops.tensor_cast.dsa_indexer.default,
        args=(
            hidden_states,
            qa_normed,
            cos,
            sin,
            indexer_cache,
            slot_mapping,
            block_tables,
            seq_lens,
            wq_b_weight,
            wk_weight,
            weights_proj_weight,
            k_norm_weight,
            8,
            64,
            32,
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


def _set_placeholder_meta(gm, values):
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            node.meta["val"] = values[node.name]


def _estimate_fake_analytic_cost(gm, cost_node, placeholder_specs, output_spec):
    shape_env = ShapeEnv()
    pass_ = MultiStreamSchedulePass(device_name=TEST_DEVICE.name)

    with FakeTensorMode(shape_env=shape_env, allow_non_fake_inputs=True):
        _set_placeholder_meta(
            gm,
            {name: torch.empty(shape, dtype=dtype) for name, (shape, dtype) in placeholder_specs.items()},
        )
        output_shape, output_dtype = output_spec
        cost_node.meta["val"] = torch.empty(output_shape, dtype=output_dtype)

        cost = pass_._estimate_node_cost_with_analytic(cost_node)

    return cost, shape_env


class MultiStreamPassTestCase(unittest.TestCase):
    def setUp(self):
        torch.compiler.reset()

    def test_unsafe_schema_allows_readonly_view_aliases(self):
        self.assertFalse(MultiStreamSchedulePass._target_has_unsafe_schema(torch.ops.aten.view.default))
        self.assertFalse(MultiStreamSchedulePass._target_has_unsafe_schema(torch.ops.aten.transpose.int))
        self.assertTrue(MultiStreamSchedulePass._target_has_unsafe_schema(torch.ops.aten.copy_.default))

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

    def test_internal_control_ops_are_unschedulable(self):
        graph = fx.Graph()
        x = graph.placeholder("x")
        begin = graph.call_function(torch.ops.tensor_cast._internal_mark_region_begin.default, args=(x,))
        copy = graph.call_function(torch.ops.tensor_cast._internal_copy_region.default, args=(begin,))
        end = graph.call_function(torch.ops.tensor_cast._internal_mark_region_end.default, args=(copy, begin))
        wait = graph.call_function(torch.ops.tensor_cast._internal_wait_and_bind.default, args=(x, 1, []))
        record = graph.call_function(torch.ops.tensor_cast._internal_record.default, args=(x, 1))
        graph.output((end, wait, record))
        for node in (begin, copy, end, wait):
            node.meta["val"] = torch.empty((4,), dtype=torch.float16, device="meta")
        record.meta["val"] = torch.empty((), dtype=torch.int64, device="meta")
        pass_ = MultiStreamSchedulePass(device_name=TEST_DEVICE.name)

        self.assertFalse(pass_._is_schedulable_node(wait))
        self.assertFalse(pass_._is_schedulable_node(record))
        self.assertFalse(pass_._is_schedulable_node(begin))
        self.assertFalse(pass_._is_schedulable_node(copy))
        self.assertFalse(pass_._is_schedulable_node(end))

    def test_wait_anchor_preserves_value_without_aliasing_input(self):
        x = torch.randn((2, 3), dtype=torch.float32)

        y = torch.ops.tensor_cast._internal_wait_and_bind.default(x, 0, [])

        self.assertTrue(torch.equal(y, x))
        self.assertNotEqual(y.data_ptr(), x.data_ptr())

    def test_metadata_analytic_cost_does_not_leak_unbacked_symbols(self):
        cases = {
            "mla": (
                _build_mla_graph,
                {
                    "q": ((100, 128, 576), torch.float16),
                    "kv_cache": ((10000, 128, 576), torch.float16),
                    "block_table": ((1, 10), torch.int64),
                    "query_start_loc": ((2,), torch.int64),
                    "seq_lens": ((1,), torch.int64),
                    "query_lens": ((1,), torch.int64),
                    "w_uk_t": ((128, 512, 512), torch.float16),
                    "w_uv": ((128, 512, 64), torch.float16),
                    "kv_b_proj": ((512, 128 * (64 + 64)), torch.float16),
                },
                ((100, 128, 64), torch.float16),
            ),
            "attention": (
                _build_attention_graph,
                {
                    "query": ((32, 512), torch.float16),
                    "key": ((100, 16, 4, 128), torch.float16),
                    "value": ((100, 16, 4, 128), torch.float16),
                    "attention_mask": ((2, 4, 16, 160), torch.float16),
                    "block_table": ((2, 10), torch.int64),
                    "query_start_loc": ((3,), torch.int64),
                    "seq_lens": ((2,), torch.int64),
                    "query_lens": ((2,), torch.int64),
                },
                ((32, 512), torch.float16),
            ),
            "sparse_attention": (
                _build_sparse_attention_graph,
                {
                    "q": ((2, 8, 4, 192), torch.float16),
                    "kv": ((2, 128, 640), torch.float16),
                    "attn_sink": ((4,), torch.float32),
                    "topk_indices": ((2, 8, 64), torch.int64),
                },
                ((2, 8, 4, 128), torch.float16),
            ),
            "dsa_indexer": (
                _build_dsa_indexer_graph,
                {
                    "hidden_states": ((2, 16, 512), torch.float16),
                    "qa_normed": ((2, 16, 256), torch.float16),
                    "cos": ((16, 32), torch.float16),
                    "sin": ((16, 32), torch.float16),
                    "indexer_cache": ((2, 512, 64), torch.float16),
                    "slot_mapping": ((32,), torch.int64),
                    "block_tables": ((2, 4), torch.int64),
                    "seq_lens": ((2,), torch.int64),
                    "wq_b_weight": ((256, 8 * 64), torch.float16),
                    "wk_weight": ((512, 64), torch.float16),
                    "weights_proj_weight": ((512, 8), torch.float16),
                    "k_norm_weight": ((64,), torch.float16),
                },
                ((2, 16, 64), torch.int64),
            ),
        }
        for name, (build_graph, placeholder_specs, output_spec) in cases.items():
            with self.subTest(name=name):
                gm, node = build_graph()
                cost, shape_env = _estimate_fake_analytic_cost(gm, node, placeholder_specs, output_spec)
                self.assertIsNotNone(cost)
                self.assertGreater(cost, 0)
                self.assertEqual(len(shape_env.pending_fresh_unbacked_symbols), 0)

    def test_sparse_mla_metadata_cache_read_uses_prefill_upper_bound(self):
        shape_env = ShapeEnv()
        with FakeTensorMode(shape_env=shape_env, allow_non_fake_inputs=True):
            q = torch.empty((100, 128, 576), dtype=torch.float16)
            kv_cache = torch.empty((10000, 128, 576), dtype=torch.float16)
            block_table = torch.empty((1, 10), dtype=torch.int64)
            query_start_loc = torch.empty((2,), dtype=torch.int64)
            seq_lens = torch.empty((1,), dtype=torch.int64)
            query_lens = torch.empty((1,), dtype=torch.int64)
            w_uk_t = torch.empty((128, 512, 512), dtype=torch.float16)
            w_uv = torch.empty((128, 512, 64), dtype=torch.float16)
            kv_b_proj = torch.empty((512, 128 * (64 + 64)), dtype=torch.float16)
            out = torch.empty((100, 128, 64), dtype=torch.float16)
            dense_op_info = OpInvokeInfo(
                torch.ops.tensor_cast.multihead_latent_attention.default,
                (
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
                {},
                out,
            )
            sparse_op_info = OpInvokeInfo(
                torch.ops.tensor_cast.multihead_latent_attention.default,
                dense_op_info.args + (64,),
                {},
                out,
            )

            dense_properties = dense_op_info.get_perf_properties()
            sparse_properties = sparse_op_info.get_perf_properties()

        self.assertGreaterEqual(sparse_properties.memory_read_bytes, dense_properties.memory_read_bytes)
        self.assertEqual(len(shape_env.pending_fresh_unbacked_symbols), 0)

    def test_mla_metadata_attention_len_uses_cache_token_extent(self):
        shape_env = ShapeEnv()
        with FakeTensorMode(shape_env=shape_env, allow_non_fake_inputs=True):
            kv_cache = torch.empty((10000, 128, 576), dtype=torch.float16)
            block_table = torch.empty((1, 10), dtype=torch.int64)

            self.assertEqual(_mla_metadata_attn_len(kv_cache, block_table), 1280)
            self.assertEqual(_mla_metadata_attn_len(kv_cache, None), 10000)

        self.assertEqual(len(shape_env.pending_fresh_unbacked_symbols), 0)

    def test_mla_metadata_cache_read_uses_request_batch_without_block_table(self):
        def _properties_for_batch(batch_size):
            q = torch.empty((100, 128, 576), dtype=torch.float16)
            kv_cache = torch.empty((10000, 128, 576), dtype=torch.float16)
            query_start_loc = torch.empty((1,), dtype=torch.int64)
            seq_lens = torch.empty((batch_size,), dtype=torch.int64)
            query_lens = torch.empty((batch_size,), dtype=torch.int64)
            w_uk_t = torch.empty((128, 512, 512), dtype=torch.float16)
            w_uv = torch.empty((128, 512, 64), dtype=torch.float16)
            kv_b_proj = torch.empty((512, 128 * (64 + 64)), dtype=torch.float16)
            out = torch.empty((100, 128, 64), dtype=torch.float16)
            return OpInvokeInfo(
                torch.ops.tensor_cast.multihead_latent_attention.default,
                (q, kv_cache, None, query_start_loc, seq_lens, query_lens, w_uk_t, w_uv, kv_b_proj, 64),
                {},
                out,
            ).get_perf_properties()

        shape_env = ShapeEnv()
        with FakeTensorMode(shape_env=shape_env, allow_non_fake_inputs=True):
            single_batch_properties = _properties_for_batch(1)
            multi_batch_properties = _properties_for_batch(3)

        cache_entry_size = 576 * torch.empty((), dtype=torch.float16).element_size()
        expected_extra_cache_read = 2 * 10000 * cache_entry_size
        self.assertGreaterEqual(
            multi_batch_properties.memory_read_bytes - single_batch_properties.memory_read_bytes,
            expected_extra_cache_read,
        )
        self.assertEqual(len(shape_env.pending_fresh_unbacked_symbols), 0)

    def test_quant_mla_metadata_topk_bounds_quant_ops(self):
        shape_env = ShapeEnv()
        with FakeTensorMode(shape_env=shape_env, allow_non_fake_inputs=True):
            q = torch.empty((100, 128, 576), dtype=torch.float16)
            kv_cache = torch.empty((10000, 128, 576), dtype=torch.float16)
            block_table = torch.empty((1, 10), dtype=torch.int64)
            query_start_loc = torch.empty((2,), dtype=torch.int64)
            seq_lens = torch.empty((1,), dtype=torch.int64)
            query_lens = torch.empty((1,), dtype=torch.int64)
            w_uk_t = torch.empty((128, 512, 512), dtype=torch.float16)
            w_uv = torch.empty((128, 512, 64), dtype=torch.float16)
            kv_b_proj = torch.empty((512, 128 * (64 + 64)), dtype=torch.float16)
            scale = torch.empty((), dtype=torch.float32)
            out = torch.empty((100, 128, 64), dtype=torch.float16)
            args = (
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
                None,
                None,
                scale,
                None,
                scale,
                None,
                scale,
                None,
                scale,
                None,
                scale,
                None,
                scale,
                None,
                scale,
                None,
                scale,
                None,
                None,
            )
            dense_properties = OpInvokeInfo(
                torch.ops.tensor_cast.multihead_latent_attention_quant.default,
                args,
                {},
                out,
            ).get_perf_properties()
            sparse_properties = OpInvokeInfo(
                torch.ops.tensor_cast.multihead_latent_attention_quant.default,
                args[:10] + (64, None) + args[12:],
                {},
                out,
            ).get_perf_properties()

        dense_gp_ops = sum(ops.gp_ops for ops in dense_properties.compute_ops.values())
        sparse_gp_ops = sum(ops.gp_ops for ops in sparse_properties.compute_ops.values())
        self.assertLess(sparse_gp_ops, dense_gp_ops)
        self.assertEqual(len(shape_env.pending_fresh_unbacked_symbols), 0)

    def test_quant_mla_metadata_output_ops_follow_dtype_conversion(self):
        def _gp_ops_for_out_dtype(out_dtype):
            q = torch.empty((100, 128, 576), dtype=torch.float16)
            kv_cache = torch.empty((10000, 128, 576), dtype=torch.float16)
            block_table = torch.empty((1, 10), dtype=torch.int64)
            query_start_loc = torch.empty((2,), dtype=torch.int64)
            seq_lens = torch.empty((1,), dtype=torch.int64)
            query_lens = torch.empty((1,), dtype=torch.int64)
            w_uk_t = torch.empty((128, 512, 512), dtype=torch.float16)
            w_uv = torch.empty((128, 512, 64), dtype=torch.float16)
            kv_b_proj = torch.empty((512, 128 * (64 + 64)), dtype=torch.float16)
            scale = torch.empty((), dtype=torch.float32)
            out = torch.empty((100, 128, 64), dtype=out_dtype or q.dtype)
            args = (
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
                None,
                None,
                scale,
                None,
                scale,
                None,
                scale,
                None,
                scale,
                None,
                scale,
                None,
                scale,
                None,
                scale,
                None,
                scale,
                None,
                out_dtype,
            )
            properties = OpInvokeInfo(
                torch.ops.tensor_cast.multihead_latent_attention_quant.default,
                args,
                {},
                out,
            ).get_perf_properties()
            return sum(ops.gp_ops for ops in properties.compute_ops.values())

        shape_env = ShapeEnv()
        with FakeTensorMode(shape_env=shape_env, allow_non_fake_inputs=True):
            default_gp_ops = _gp_ops_for_out_dtype(None)
            same_dtype_gp_ops = _gp_ops_for_out_dtype(torch.float16)
            converted_dtype_gp_ops = _gp_ops_for_out_dtype(torch.float32)

        expected_output_ops = 100 * 128 * 64 * 2
        self.assertEqual(same_dtype_gp_ops, default_gp_ops)
        self.assertEqual(converted_dtype_gp_ops - same_dtype_gp_ops, expected_output_ops)
        self.assertEqual(len(shape_env.pending_fresh_unbacked_symbols), 0)

    def test_quant_mla_optional_none_args_still_use_analytic_cost(self):
        graph = fx.Graph()

        def placeholder(name, value):
            node = graph.placeholder(name)
            node.meta["val"] = value
            return node

        q = placeholder("q", torch.empty((100, 128, 576), device="meta", dtype=torch.float16))
        kv_cache = placeholder("kv_cache", torch.empty((10000, 128, 576), device="meta", dtype=torch.float16))
        block_table = placeholder("block_table", torch.empty((1, 10), device="meta", dtype=torch.int64))
        query_start_loc = placeholder("query_start_loc", torch.empty((2,), device="meta", dtype=torch.int64))
        seq_lens = placeholder("seq_lens", torch.empty((1,), device="meta", dtype=torch.int64))
        query_lens = placeholder("query_lens", torch.empty((1,), device="meta", dtype=torch.int64))
        w_uk_t = placeholder("w_uk_t", torch.empty((128, 512, 512), device="meta", dtype=torch.float16))
        w_uv = placeholder("w_uv", torch.empty((128, 512, 64), device="meta", dtype=torch.float16))
        kv_b_proj = placeholder("kv_b_proj", torch.empty((512, 128 * (64 + 64)), device="meta", dtype=torch.float16))
        scale = placeholder("scale", torch.empty((), device="meta", dtype=torch.float32))
        out = graph.call_function(
            torch.ops.tensor_cast.multihead_latent_attention_quant.default,
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
                None,
                None,
                scale,
                None,
                scale,
                None,
                scale,
                None,
                scale,
                None,
                scale,
                None,
                scale,
                None,
                scale,
                None,
                scale,
                None,
                None,
            ),
        )
        out.meta["val"] = torch.empty((100, 128, 64), device="meta", dtype=torch.float16)
        graph.output(out)

        pass_ = MultiStreamSchedulePass(device_name=TEST_DEVICE.name)
        self.assertIsNotNone(pass_._estimate_node_cost_with_analytic(out))

    def test_quant_mla_concrete_output_ops_follow_dtype_conversion(self):
        def _gp_ops_for_out_dtype(out_dtype):
            q = torch.empty((6, 2, 8), dtype=torch.float16)
            kv_cache = torch.empty((20, 2, 6), dtype=torch.float16)
            query_start_loc = torch.tensor([0, 5, 6], dtype=torch.int64)
            seq_lens = torch.tensor([6, 1], dtype=torch.int64)
            query_lens = torch.tensor([5, 1], dtype=torch.int64)
            w_uk_t = torch.empty((2, 6, 4), dtype=torch.float16)
            w_uv = torch.empty((2, 4, 3), dtype=torch.float16)
            kv_b_proj = torch.empty((4, 2 * (6 + 3)), dtype=torch.float16)
            scale = torch.empty((), dtype=torch.float32)
            out = torch.empty((6, 2, 3), dtype=out_dtype or q.dtype)
            args = (
                q,
                kv_cache,
                None,
                query_start_loc,
                seq_lens,
                query_lens,
                w_uk_t,
                w_uv,
                kv_b_proj,
                3,
                None,
                None,
                scale,
                None,
                scale,
                None,
                scale,
                None,
                scale,
                None,
                scale,
                None,
                scale,
                None,
                scale,
                None,
                scale,
                None,
                out_dtype,
            )
            properties = OpInvokeInfo(
                torch.ops.tensor_cast.multihead_latent_attention_quant.default,
                args,
                {},
                out,
            ).get_perf_properties()
            return sum(ops.gp_ops for ops in properties.compute_ops.values())

        default_gp_ops = _gp_ops_for_out_dtype(None)
        same_dtype_gp_ops = _gp_ops_for_out_dtype(torch.float16)
        converted_dtype_gp_ops = _gp_ops_for_out_dtype(torch.float32)

        expected_output_ops = 6 * 2 * 3 * 2
        self.assertEqual(same_dtype_gp_ops, default_gp_ops)
        self.assertEqual(converted_dtype_gp_ops - same_dtype_gp_ops, expected_output_ops)

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

    def test_analytic_cost_does_not_swallow_regular_exceptions(self):
        gm, node = _build_unary_graph(torch.ops.aten.neg.default)
        pass_ = MultiStreamSchedulePass(device_name=TEST_DEVICE.name)

        def broken_estimator(op_invoke_info):
            raise ValueError("regular estimator bug")

        pass_._analytic_model.process_op = broken_estimator
        for graph_node in gm.graph.nodes:
            if graph_node.op == "placeholder":
                graph_node.meta["val"] = torch.empty((4,), dtype=torch.float16, device="meta")
        node.meta["val"] = torch.empty((4,), dtype=torch.float16, device="meta")

        with self.assertRaisesRegex(ValueError, "regular estimator bug"):
            pass_._estimate_node_cost_with_analytic(node)

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

        heuristic_cost = pass_._estimate_node_cost_s(node, config.compilation.multistream.compute_stream_id)

        self.assertGreater(heuristic_cost, 0)


if __name__ == "__main__":
    unittest.main()
