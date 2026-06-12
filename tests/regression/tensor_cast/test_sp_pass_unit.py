"""Unit tests for SequenceParallelPass ordered rewrites.

P1: all_reduce -> [begin?] -> rms_norm | add_rms_norm
P2: all_reduce -> add_rms_norm2, selective all_gather on getitems
P3: getitem[1] + all_reduce -> add -> [end? -> copy*] -> norm
Edges: world_size=1, non-shardable candidates, and mixed-shape graphs
"""

import operator
import unittest

import torch
import torch.fx as fx
from torch.fx.passes.shape_prop import ShapeProp
from tensor_cast import config
import tensor_cast.ops  # noqa: F401
from tensor_cast.compilation.passes.sequence_parallel_pass import (
    Pattern1Rewriter,
    Pattern2Rewriter,
    Pattern3Rewriter,
    SequenceParallelPass,
)

INPUT_SHAPE = (1, 128, 4096)
LOCAL_INPUT_SHAPE = (1, 64, 4096)
WEIGHT_SHAPE = (4096,)
RANK_GROUP = [0, 1]  # world_size = 2
BAD_RANK_GROUP = [0, 1, 2]
EPS = 1e-5


def _meta_tensor(shape):
    return torch.empty(shape, device="meta")


def _make_meta_value(tensor_meta):
    if hasattr(tensor_meta, "shape"):
        dtype = getattr(tensor_meta, "dtype", torch.float32)
        return torch.empty(tuple(tensor_meta.shape), dtype=dtype, device="meta")
    if isinstance(tensor_meta, (tuple, list)):
        if not tensor_meta:
            return None
        tensor_meta = tensor_meta[0]
        return _make_meta_value(tensor_meta)
    return None


def _populate_val_metadata(graph_module):
    for node in graph_module.graph.nodes:
        if "val" in node.meta:
            continue
        meta_value = _make_meta_value(node.meta.get("tensor_meta"))
        if meta_value is not None:
            node.meta["val"] = meta_value
    return graph_module


def _trace_graph_module(module, *meta_inputs):
    graph_module = fx.symbolic_trace(module)
    ShapeProp(graph_module).propagate(*meta_inputs)
    return _populate_val_metadata(graph_module)


def _trace_program(module, *input_shapes):
    return _trace_graph_module(module, *(_meta_tensor(shape) for shape in input_shapes))


def _run_pass(graph_module):
    saved = config.compilation.passes.enable_sequence_parallel
    config.compilation.passes.enable_sequence_parallel = True
    try:
        graph_module = SequenceParallelPass()(graph_module)
    finally:
        config.compilation.passes.enable_sequence_parallel = saved
    return graph_module


def _count_calls(graph_module, target):
    return sum(1 for node in graph_module.graph.nodes if node.op == "call_function" and node.target is target)


def _find_calls(graph_module, target):
    return [node for node in graph_module.graph.nodes if node.op == "call_function" and node.target is target]


def _has_user(node, target):
    return any(user.op == "call_function" and user.target is target for user in node.users)


def _find_getitems(graph_module, index):
    return [
        node
        for node in graph_module.graph.nodes
        if node.op == "call_function" and node.target is operator.getitem and node.args[1] == index
    ]


def _mark_add_rms_norm2_sp_local(graph_module):
    [norm2] = _find_calls(graph_module, torch.ops.tensor_cast.add_rms_norm2.default)
    norm2.meta["tensor_cast_sp_local"] = True
    return norm2


def _get_node_index(graph_module, node):
    for index, current_node in enumerate(graph_module.graph.nodes):
        if current_node is node:
            return index
    raise AssertionError("node not found in graph")


def _trace_p1_graph(
    *,
    marker=False,
    add_rms_norm=False,
    rank_group=RANK_GROUP,
    input_shape=INPUT_SHAPE,
    residual_shape=INPUT_SHAPE,
):
    class Program(torch.nn.Module):
        def forward(self, x, residual, w):
            all_reduce = torch.ops.tensor_cast.all_reduce.default(x, 0, rank_group)
            current = torch.ops.tensor_cast._internal_mark_region_begin.default(all_reduce, 0) if marker else all_reduce
            if add_rms_norm:
                return torch.ops.tensor_cast.add_rms_norm.default(current, residual, w, EPS)
            return torch.ops.tensor_cast.rms_norm.default(current, w, EPS)

    return _trace_program(Program(), input_shape, residual_shape, WEIGHT_SHAPE)


def _trace_p2_p3_graph(
    *,
    begin_on_res=False,
    with_end=True,
    with_view=False,
    num_copies=0,
    tail_rank_group=RANK_GROUP,
    tail_input_shape=INPUT_SHAPE,
):
    class Program(torch.nn.Module):
        def forward(self, x, res, x2, w, w2):
            all_reduce_1 = torch.ops.tensor_cast.all_reduce.default(x, 0, RANK_GROUP)
            res_input = torch.ops.tensor_cast._internal_mark_region_begin.default(res, 0) if begin_on_res else res
            getitem0, getitem1 = torch.ops.tensor_cast.add_rms_norm2.default(
                all_reduce_1,
                res_input,
                w,
                EPS,
            )

            all_reduce_2 = torch.ops.tensor_cast.all_reduce.default(x2, 0, tail_rank_group)
            comm_output = torch.ops.aten.reshape.default(all_reduce_2, list(INPUT_SHAPE)) if with_view else all_reduce_2
            current = torch.ops.aten.add.Tensor(getitem1, comm_output)
            if with_end:
                current = torch.ops.tensor_cast._internal_mark_region_end.default(current, 0)
            for index in range(num_copies):
                current = torch.ops.tensor_cast._internal_copy_region.default(current, index)
            tail = torch.ops.tensor_cast.rms_norm.default(current, w2, EPS)
            return getitem0, tail

    return _trace_program(
        Program(),
        INPUT_SHAPE,
        LOCAL_INPUT_SHAPE,
        tail_input_shape,
        WEIGHT_SHAPE,
        WEIGHT_SHAPE,
    )


class Pattern1RewriterTestCase(unittest.TestCase):
    def test_pattern1_rewriter_apply(self):
        graph_module = _trace_p1_graph()

        rewritten = Pattern1Rewriter().apply(graph_module.graph)

        self.assertEqual(rewritten, 1)
        self.assertEqual(_count_calls(graph_module, torch.ops.tensor_cast.all_reduce.default), 1)
        self.assertEqual(_count_calls(graph_module, torch.ops.tensor_cast.reduce_scatter.default), 1)
        self.assertEqual(_count_calls(graph_module, torch.ops.tensor_cast.all_gather.default), 1)

    def test_p1_with_marker(self):
        graph_module = _run_pass(_trace_p1_graph(marker=True))

        self.assertEqual(_count_calls(graph_module, torch.ops.tensor_cast.all_reduce.default), 0)
        self.assertEqual(_count_calls(graph_module, torch.ops.tensor_cast.reduce_scatter.default), 1)
        self.assertEqual(_count_calls(graph_module, torch.ops.tensor_cast.all_gather.default), 1)
        [reduce_scatter] = _find_calls(graph_module, torch.ops.tensor_cast.reduce_scatter.default)
        [begin_node] = _find_calls(graph_module, torch.ops.tensor_cast._internal_mark_region_begin.default)
        [norm_node] = _find_calls(graph_module, torch.ops.tensor_cast.rms_norm.default)
        [all_gather] = _find_calls(graph_module, torch.ops.tensor_cast.all_gather.default)
        self.assertIs(begin_node.args[0], reduce_scatter)
        self.assertIs(norm_node.args[0], begin_node)
        self.assertIs(all_gather.args[0], norm_node)

    def test_p1_without_marker(self):
        graph_module = _run_pass(_trace_p1_graph())

        self.assertEqual(_count_calls(graph_module, torch.ops.tensor_cast.all_reduce.default), 0)
        [reduce_scatter] = _find_calls(graph_module, torch.ops.tensor_cast.reduce_scatter.default)
        [norm_node] = _find_calls(graph_module, torch.ops.tensor_cast.rms_norm.default)
        [all_gather] = _find_calls(graph_module, torch.ops.tensor_cast.all_gather.default)
        self.assertIs(norm_node.args[0], reduce_scatter)
        self.assertIs(all_gather.args[0], norm_node)

    def test_p1_add_rms_norm(self):
        graph_module = _run_pass(_trace_p1_graph(marker=True, add_rms_norm=True))

        self.assertEqual(_count_calls(graph_module, torch.ops.tensor_cast.all_reduce.default), 0)
        [norm_node] = _find_calls(graph_module, torch.ops.tensor_cast.add_rms_norm.default)
        self.assertTrue(_has_user(norm_node, torch.ops.tensor_cast.all_gather.default))

    def test_p1_dim_mismatch_view_uses_comm_output_shard_dim(self):
        graph_module = _trace_p1_graph(input_shape=(1, 256, 4096))

        # This regression covers compiler IR where all_reduce has a rank-reduced
        # output metadata shape. Build the graph from a PyTorch program, then
        # adjust only the metadata needed to exercise that IR-only shape repair.
        [all_reduce] = _find_calls(graph_module, torch.ops.tensor_cast.all_reduce.default)
        [norm] = _find_calls(graph_module, torch.ops.tensor_cast.rms_norm.default)
        all_reduce.meta["val"] = _meta_tensor((256, 4096))
        norm.meta["val"] = _meta_tensor((256, 4096))

        graph_module = _run_pass(graph_module)

        [view] = _find_calls(graph_module, torch.ops.aten.view.default)
        self.assertEqual(view.args[1], [128, 4096])


class Pattern2RewriterTestCase(unittest.TestCase):
    def test_pattern2_rewriter_apply(self):
        class Program(torch.nn.Module):
            def forward(self, x, res, w):
                all_reduce = torch.ops.tensor_cast.all_reduce.default(x, 0, RANK_GROUP)
                return torch.ops.tensor_cast.add_rms_norm2.default(all_reduce, res, w, EPS)[0]

        graph_module = _trace_program(Program(), INPUT_SHAPE, LOCAL_INPUT_SHAPE, WEIGHT_SHAPE)

        rewritten = Pattern2Rewriter().apply(graph_module.graph)

        self.assertEqual(rewritten, 1)
        self.assertEqual(_count_calls(graph_module, torch.ops.tensor_cast.all_reduce.default), 1)
        self.assertEqual(_count_calls(graph_module, torch.ops.tensor_cast.reduce_scatter.default), 1)
        self.assertEqual(_count_calls(graph_module, torch.ops.tensor_cast.all_gather.default), 1)

    def test_p2_skips_when_other_input_is_full_shape(self):
        class Program(torch.nn.Module):
            def forward(self, x, full_residual, w):
                all_reduce = torch.ops.tensor_cast.all_reduce.default(x, 0, RANK_GROUP)
                out, _ = torch.ops.tensor_cast.add_rms_norm2.default(all_reduce, full_residual, w, EPS)
                return out

        graph_module = _run_pass(_trace_program(Program(), INPUT_SHAPE, INPUT_SHAPE, WEIGHT_SHAPE))

        self.assertEqual(_count_calls(graph_module, torch.ops.tensor_cast.all_reduce.default), 1)
        self.assertEqual(_count_calls(graph_module, torch.ops.tensor_cast.reduce_scatter.default), 0)
        self.assertEqual(_count_calls(graph_module, torch.ops.tensor_cast.all_gather.default), 0)
        [norm2_new] = _find_calls(graph_module, torch.ops.tensor_cast.add_rms_norm2.default)
        [all_reduce] = _find_calls(graph_module, torch.ops.tensor_cast.all_reduce.default)
        full_residual = next(node for node in graph_module.graph.nodes if node.name == "full_residual")
        self.assertIs(norm2_new.args[0], all_reduce)
        self.assertIs(norm2_new.args[1], full_residual)

    def test_p2_skips_when_other_input_is_all_gathered_full_shape(self):
        class Program(torch.nn.Module):
            def forward(self, x, local_src, w):
                full_residual = torch.ops.tensor_cast.all_gather.default(local_src, 1, 0, RANK_GROUP)
                all_reduce = torch.ops.tensor_cast.all_reduce.default(x, 0, RANK_GROUP)
                out, _ = torch.ops.tensor_cast.add_rms_norm2.default(all_reduce, full_residual, w, EPS)
                return out

        graph_module = _run_pass(_trace_program(Program(), INPUT_SHAPE, LOCAL_INPUT_SHAPE, WEIGHT_SHAPE))

        self.assertEqual(_count_calls(graph_module, torch.ops.tensor_cast.all_reduce.default), 1)
        self.assertEqual(_count_calls(graph_module, torch.ops.tensor_cast.reduce_scatter.default), 0)
        self.assertEqual(_count_calls(graph_module, torch.ops.tensor_cast.all_gather.default), 1)
        [norm2_new] = _find_calls(graph_module, torch.ops.tensor_cast.add_rms_norm2.default)
        [all_reduce] = _find_calls(graph_module, torch.ops.tensor_cast.all_reduce.default)
        [full_residual] = _find_calls(graph_module, torch.ops.tensor_cast.all_gather.default)
        self.assertIs(norm2_new.args[0], all_reduce)
        self.assertIs(norm2_new.args[1], full_residual)

    def test_p2_skips_bias_like_residual_shape(self):
        class Program(torch.nn.Module):
            def forward(self, x, bias, w):
                all_reduce = torch.ops.tensor_cast.all_reduce.default(x, 0, RANK_GROUP)
                out, _ = torch.ops.tensor_cast.add_rms_norm2.default(all_reduce, bias, w, EPS)
                return out

        graph_module = _run_pass(_trace_program(Program(), INPUT_SHAPE, WEIGHT_SHAPE, WEIGHT_SHAPE))

        self.assertEqual(_count_calls(graph_module, torch.ops.tensor_cast.all_reduce.default), 1)
        self.assertEqual(_count_calls(graph_module, torch.ops.tensor_cast.reduce_scatter.default), 0)
        self.assertEqual(_count_calls(graph_module, torch.ops.tensor_cast.all_gather.default), 0)

    def test_p3_skips_tail_when_source_p2_was_not_localized(self):
        class Program(torch.nn.Module):
            def forward(self, x, full_residual, x2, w, w2):
                all_reduce_1 = torch.ops.tensor_cast.all_reduce.default(x, 0, RANK_GROUP)
                _, residual = torch.ops.tensor_cast.add_rms_norm2.default(all_reduce_1, full_residual, w, EPS)
                all_reduce_2 = torch.ops.tensor_cast.all_reduce.default(x2, 0, RANK_GROUP)
                add_node = torch.ops.aten.add.Tensor(residual, all_reduce_2)
                return torch.ops.tensor_cast.rms_norm.default(add_node, w2, EPS)

        graph_module = _run_pass(
            _trace_program(
                Program(),
                INPUT_SHAPE,
                INPUT_SHAPE,
                INPUT_SHAPE,
                WEIGHT_SHAPE,
                WEIGHT_SHAPE,
            )
        )

        self.assertEqual(_count_calls(graph_module, torch.ops.tensor_cast.all_reduce.default), 2)
        self.assertEqual(_count_calls(graph_module, torch.ops.tensor_cast.reduce_scatter.default), 0)
        self.assertEqual(_count_calls(graph_module, torch.ops.tensor_cast.all_gather.default), 0)

    def _assert_p2_p3_middle(self, graph_module):
        self.assertEqual(_count_calls(graph_module, torch.ops.tensor_cast.all_reduce.default), 0)
        self.assertGreaterEqual(_count_calls(graph_module, torch.ops.tensor_cast.reduce_scatter.default), 2)
        for getitem1 in _find_getitems(graph_module, 1):
            self.assertFalse(
                _has_user(getitem1, torch.ops.tensor_cast.all_gather.default),
                "gi[1] must NOT be gathered",
            )
            self.assertTrue(
                _has_user(getitem1, torch.ops.aten.add.Tensor),
                "gi[1] must directly feed an add node",
            )
        norms = _find_calls(graph_module, torch.ops.tensor_cast.rms_norm.default)
        self.assertTrue(
            any(_has_user(norm_node, torch.ops.tensor_cast.all_gather.default) for norm_node in norms),
            "tail rms_norm must feed all_gather",
        )

    def test_p2_middle_no_marker(self):
        graph_module = _run_pass(_trace_p2_p3_graph(with_end=True))
        self._assert_p2_p3_middle(graph_module)

    def test_p2_middle_with_marker(self):
        graph_module = _run_pass(_trace_p2_p3_graph(begin_on_res=True, with_end=True))
        self._assert_p2_p3_middle(graph_module)

    def test_p2_residual_fanout(self):
        class Program(torch.nn.Module):
            def forward(self, x, res, other, w):
                all_reduce = torch.ops.tensor_cast.all_reduce.default(x, 0, RANK_GROUP)
                out, residual = torch.ops.tensor_cast.add_rms_norm2.default(all_reduce, res, w, EPS)
                add_node = torch.ops.aten.add.Tensor(residual, other)
                return out, add_node

        graph_module = _run_pass(_trace_program(Program(), INPUT_SHAPE, LOCAL_INPUT_SHAPE, INPUT_SHAPE, WEIGHT_SHAPE))

        for node in _find_getitems(graph_module, 1):
            self.assertTrue(_has_user(node, torch.ops.tensor_cast.all_gather.default))

    def test_p2_dual_all_reduce(self):
        class Program(torch.nn.Module):
            def forward(self, x1, x2, w):
                all_reduce_1 = torch.ops.tensor_cast.all_reduce.default(x1, 0, RANK_GROUP)
                all_reduce_2 = torch.ops.tensor_cast.all_reduce.default(x2, 0, RANK_GROUP)
                out, _ = torch.ops.tensor_cast.add_rms_norm2.default(all_reduce_1, all_reduce_2, w, EPS)
                return out

        graph_module = _run_pass(_trace_program(Program(), INPUT_SHAPE, INPUT_SHAPE, WEIGHT_SHAPE))

        self.assertEqual(_count_calls(graph_module, torch.ops.tensor_cast.all_reduce.default), 2)
        self.assertEqual(_count_calls(graph_module, torch.ops.tensor_cast.reduce_scatter.default), 0)
        self.assertEqual(_count_calls(graph_module, torch.ops.tensor_cast.all_gather.default), 0)
        [norm2_new] = _find_calls(graph_module, torch.ops.tensor_cast.add_rms_norm2.default)
        all_reduce_nodes = _find_calls(graph_module, torch.ops.tensor_cast.all_reduce.default)
        self.assertIn(norm2_new.args[0], all_reduce_nodes)
        self.assertIn(norm2_new.args[1], all_reduce_nodes)
        self.assertIsNot(norm2_new.args[0], norm2_new.args[1])

    def test_p2_marker_wrapped_comm(self):
        class Program(torch.nn.Module):
            def forward(self, x, res, w):
                all_reduce = torch.ops.tensor_cast.all_reduce.default(x, 0, RANK_GROUP)
                begin = torch.ops.tensor_cast._internal_mark_region_begin.default(all_reduce, 0)
                out, _ = torch.ops.tensor_cast.add_rms_norm2.default(begin, res, w, EPS)
                return out

        graph_module = _run_pass(_trace_program(Program(), INPUT_SHAPE, INPUT_SHAPE, WEIGHT_SHAPE))

        self.assertEqual(_count_calls(graph_module, torch.ops.tensor_cast.all_reduce.default), 1)
        self.assertEqual(_count_calls(graph_module, torch.ops.tensor_cast.reduce_scatter.default), 0)
        self.assertEqual(_count_calls(graph_module, torch.ops.tensor_cast.add_rms_norm2.default), 1)
        [norm2_new] = _find_calls(graph_module, torch.ops.tensor_cast.add_rms_norm2.default)
        [begin_new] = _find_calls(graph_module, torch.ops.tensor_cast._internal_mark_region_begin.default)
        [all_reduce_new] = _find_calls(graph_module, torch.ops.tensor_cast.all_reduce.default)
        self.assertIs(norm2_new.args[0], begin_new)
        self.assertIs(begin_new.args[0], all_reduce_new)

    def test_p2_2d_shard_dim(self):
        shape_2d = (256, 4096)
        local_shape_2d = (128, 4096)

        class Program(torch.nn.Module):
            def forward(self, x, res, w):
                all_reduce = torch.ops.tensor_cast.all_reduce.default(x, 0, RANK_GROUP)
                out, residual = torch.ops.tensor_cast.add_rms_norm2.default(all_reduce, res, w, EPS)
                return out, residual

        graph_module = _run_pass(_trace_program(Program(), shape_2d, local_shape_2d, WEIGHT_SHAPE))

        self.assertEqual(_count_calls(graph_module, torch.ops.tensor_cast.all_reduce.default), 0)
        [reduce_scatter] = _find_calls(graph_module, torch.ops.tensor_cast.reduce_scatter.default)
        self.assertEqual(reduce_scatter.args[1], 0)

        for node in _find_getitems(graph_module, 1):
            self.assertTrue(_has_user(node, torch.ops.tensor_cast.all_gather.default))
        for node in _find_getitems(graph_module, 0):
            self.assertTrue(_has_user(node, torch.ops.tensor_cast.all_gather.default))

        all_gathers = _find_calls(graph_module, torch.ops.tensor_cast.all_gather.default)
        self.assertGreaterEqual(len(all_gathers), 2)
        for all_gather in all_gathers:
            self.assertEqual(all_gather.args[1], 0)

    def test_p2_last(self):
        class Program(torch.nn.Module):
            def forward(self, x, res, w):
                all_reduce = torch.ops.tensor_cast.all_reduce.default(x, 0, RANK_GROUP)
                out, _ = torch.ops.tensor_cast.add_rms_norm2.default(all_reduce, res, w, EPS)
                return out

        graph_module = _run_pass(_trace_program(Program(), INPUT_SHAPE, LOCAL_INPUT_SHAPE, WEIGHT_SHAPE))

        self.assertEqual(_count_calls(graph_module, torch.ops.tensor_cast.all_reduce.default), 0)
        for node in _find_getitems(graph_module, 0):
            self.assertTrue(_has_user(node, torch.ops.tensor_cast.all_gather.default))

    def test_p2_reduce_scatter_inserted_after_all_reduce(self):
        class Program(torch.nn.Module):
            def forward(self, x, res, w):
                all_reduce = torch.ops.tensor_cast.all_reduce.default(x, 0, RANK_GROUP)
                out, _ = torch.ops.tensor_cast.add_rms_norm2.default(all_reduce, res, w, EPS)
                return out

        graph_module = _trace_program(Program(), INPUT_SHAPE, LOCAL_INPUT_SHAPE, WEIGHT_SHAPE)
        [all_reduce] = _find_calls(graph_module, torch.ops.tensor_cast.all_reduce.default)

        Pattern2Rewriter().apply(graph_module.graph)

        [reduce_scatter] = _find_calls(graph_module, torch.ops.tensor_cast.reduce_scatter.default)
        self.assertLess(
            _get_node_index(graph_module, all_reduce),
            _get_node_index(graph_module, reduce_scatter),
        )

    def test_markerless_shared_entry_all_reduce_and_fused_tail(self):
        class Program(torch.nn.Module):
            def forward(self, x, x2, x3, w, w2, w3):
                all_reduce_0 = torch.ops.tensor_cast.all_reduce.default(x, 0, RANK_GROUP)
                torch.ops.tensor_cast.rms_norm.default(all_reduce_0, w, EPS)

                all_reduce_1 = torch.ops.tensor_cast.all_reduce.default(x2, 0, RANK_GROUP)
                getitem0, getitem1 = torch.ops.tensor_cast.add_rms_norm2.default(
                    all_reduce_0,
                    all_reduce_1,
                    w2,
                    EPS,
                )

                all_reduce_2 = torch.ops.tensor_cast.all_reduce.default(x3, 0, RANK_GROUP)
                tail = torch.ops.tensor_cast.add_rms_norm.default(getitem1, all_reduce_2, w3, EPS)
                return getitem0, tail

        graph_module = _run_pass(
            _trace_program(
                Program(),
                INPUT_SHAPE,
                INPUT_SHAPE,
                INPUT_SHAPE,
                WEIGHT_SHAPE,
                WEIGHT_SHAPE,
                WEIGHT_SHAPE,
            )
        )

        self.assertEqual(_count_calls(graph_module, torch.ops.tensor_cast.all_reduce.default), 0)
        self.assertEqual(_count_calls(graph_module, torch.ops.tensor_cast.reduce_scatter.default), 3)
        self.assertGreaterEqual(_count_calls(graph_module, torch.ops.tensor_cast.all_gather.default), 2)
        for node in _find_getitems(graph_module, 0):
            self.assertTrue(_has_user(node, torch.ops.tensor_cast.all_gather.default))
        for node in _find_getitems(graph_module, 1):
            self.assertFalse(_has_user(node, torch.ops.tensor_cast.all_gather.default))
        [tail_new] = _find_calls(graph_module, torch.ops.tensor_cast.add_rms_norm.default)
        self.assertTrue(_has_user(tail_new, torch.ops.tensor_cast.all_gather.default))

    def test_p2_residual_chained_into_next_p2_stays_local(self):
        class Program(torch.nn.Module):
            def forward(self, res0, x1, x2, w1, w2):
                all_reduce_1 = torch.ops.tensor_cast.all_reduce.default(x1, 0, RANK_GROUP)
                _, residual = torch.ops.tensor_cast.add_rms_norm2.default(res0, all_reduce_1, w1, EPS)

                all_reduce_2 = torch.ops.tensor_cast.all_reduce.default(x2, 0, RANK_GROUP)
                out, _ = torch.ops.tensor_cast.add_rms_norm2.default(residual, all_reduce_2, w2, EPS)
                return out

        graph_module = _run_pass(
            _trace_program(
                Program(),
                LOCAL_INPUT_SHAPE,
                INPUT_SHAPE,
                INPUT_SHAPE,
                WEIGHT_SHAPE,
                WEIGHT_SHAPE,
            )
        )

        chained_getitem1 = _find_getitems(graph_module, 1)[0]
        self.assertFalse(
            _has_user(chained_getitem1, torch.ops.tensor_cast.all_gather.default),
            "intermediate residual feeding next P2 must stay local",
        )
        norm2_2_new = _find_calls(graph_module, torch.ops.tensor_cast.add_rms_norm2.default)[1]
        self.assertIs(norm2_2_new.args[0], chained_getitem1)

    def test_p2_gathers_residual_when_downstream_p2_is_not_shardable(self):
        class Program(torch.nn.Module):
            def forward(self, x, res, bad_x, w1, w2):
                all_reduce_1 = torch.ops.tensor_cast.all_reduce.default(x, 0, RANK_GROUP)
                out1, residual = torch.ops.tensor_cast.add_rms_norm2.default(all_reduce_1, res, w1, EPS)
                all_reduce_2 = torch.ops.tensor_cast.all_reduce.default(bad_x, 0, RANK_GROUP)
                out2, _ = torch.ops.tensor_cast.add_rms_norm2.default(residual, all_reduce_2, w2, EPS)
                return out1, out2

        graph_module = _run_pass(
            _trace_program(
                Program(),
                INPUT_SHAPE,
                LOCAL_INPUT_SHAPE,
                (1, 127, 4096),
                WEIGHT_SHAPE,
                WEIGHT_SHAPE,
            )
        )

        self.assertEqual(_count_calls(graph_module, torch.ops.tensor_cast.all_reduce.default), 1)
        self.assertEqual(_count_calls(graph_module, torch.ops.tensor_cast.reduce_scatter.default), 1)
        first_residual = _find_getitems(graph_module, 1)[0]
        self.assertTrue(_has_user(first_residual, torch.ops.tensor_cast.all_gather.default))

    def test_p2_gathers_residual_when_downstream_p3_is_not_shardable(self):
        graph_module = _run_pass(_trace_p2_p3_graph(tail_rank_group=BAD_RANK_GROUP))

        self.assertEqual(_count_calls(graph_module, torch.ops.tensor_cast.all_reduce.default), 1)
        self.assertEqual(_count_calls(graph_module, torch.ops.tensor_cast.reduce_scatter.default), 1)
        [residual] = _find_getitems(graph_module, 1)
        self.assertTrue(_has_user(residual, torch.ops.tensor_cast.all_gather.default))


class Pattern3RewriterTestCase(unittest.TestCase):
    def _assert_p3_base(self, graph_module):
        self.assertEqual(
            _count_calls(graph_module, torch.ops.tensor_cast.all_reduce.default),
            0,
            "no all_reduce should remain",
        )
        self.assertGreaterEqual(_count_calls(graph_module, torch.ops.tensor_cast.reduce_scatter.default), 2)
        for getitem1 in _find_getitems(graph_module, 1):
            self.assertFalse(
                _has_user(getitem1, torch.ops.tensor_cast.all_gather.default),
                "gi[1] must NOT be gathered",
            )
            self.assertTrue(_has_user(getitem1, torch.ops.aten.add.Tensor), "gi[1] should feed add")
        norms = _find_calls(graph_module, torch.ops.tensor_cast.rms_norm.default)
        self.assertTrue(
            any(_has_user(node, torch.ops.tensor_cast.all_gather.default) for node in norms),
            "tail norm must feed all_gather",
        )

    def _get_add_comm_input(self, graph_module):
        [add_node] = _find_calls(graph_module, torch.ops.aten.add.Tensor)
        getitem_ids = {id(node) for node in _find_getitems(graph_module, 1)}
        for arg in add_node.args:
            if id(arg) not in getitem_ids:
                return arg
        self.fail("add node has no non-gi1 input")

    def test_p3_with_marker(self):
        graph_module = _run_pass(_trace_p2_p3_graph(with_end=True))
        self._assert_p3_base(graph_module)

        comm = self._get_add_comm_input(graph_module)
        self.assertIs(comm.target, torch.ops.tensor_cast.reduce_scatter.default)

        [add_node] = _find_calls(graph_module, torch.ops.aten.add.Tensor)
        end_users = [
            user
            for user in add_node.users
            if hasattr(user, "target") and user.target is torch.ops.tensor_cast._internal_mark_region_end.default
        ]
        self.assertEqual(len(end_users), 1, "add must feed exactly one region_end")
        end_node = end_users[0]
        self.assertIs(end_node.args[0], add_node)

        [norm] = _find_calls(graph_module, torch.ops.tensor_cast.rms_norm.default)
        self.assertIs(norm.args[0], end_node, "rms_norm must consume region_end")

    def test_p3_without_marker(self):
        graph_module = _run_pass(_trace_p2_p3_graph(with_end=False))
        self._assert_p3_base(graph_module)

        comm = self._get_add_comm_input(graph_module)
        self.assertIs(comm.target, torch.ops.tensor_cast.reduce_scatter.default)

        [add_node] = _find_calls(graph_module, torch.ops.aten.add.Tensor)
        [norm] = _find_calls(graph_module, torch.ops.tensor_cast.rms_norm.default)
        self.assertIs(norm.args[0], add_node, "norm must consume add directly")

    def test_p3_with_view(self):
        graph_module = _run_pass(_trace_p2_p3_graph(with_view=True))
        self._assert_p3_base(graph_module)

        comm = self._get_add_comm_input(graph_module)
        self.assertIs(comm.target, torch.ops.aten.reshape.default)
        self.assertIs(comm.args[0].target, torch.ops.tensor_cast.reduce_scatter.default)

    def test_p3_no_marker_copy_region(self):
        graph_module = _run_pass(_trace_p2_p3_graph(with_end=False, num_copies=2))
        self._assert_p3_base(graph_module)

        self.assertEqual(
            _count_calls(graph_module, torch.ops.tensor_cast._internal_mark_region_end.default),
            0,
        )
        [norm] = _find_calls(graph_module, torch.ops.tensor_cast.rms_norm.default)
        copy1 = norm.args[0]
        self.assertIs(copy1.target, torch.ops.tensor_cast._internal_copy_region.default)
        copy0 = copy1.args[0]
        self.assertIs(copy0.target, torch.ops.tensor_cast._internal_copy_region.default)
        [add_node] = _find_calls(graph_module, torch.ops.aten.add.Tensor)
        self.assertIs(copy0.args[0], add_node)

    def test_p3_with_copy_region_chain(self):
        graph_module = _run_pass(_trace_p2_p3_graph(with_end=True, num_copies=2))
        self._assert_p3_base(graph_module)

        [norm] = _find_calls(graph_module, torch.ops.tensor_cast.rms_norm.default)
        copy1 = norm.args[0]
        self.assertIs(copy1.target, torch.ops.tensor_cast._internal_copy_region.default)
        copy0 = copy1.args[0]
        self.assertIs(copy0.target, torch.ops.tensor_cast._internal_copy_region.default)
        end_node = copy0.args[0]
        self.assertIs(end_node.target, torch.ops.tensor_cast._internal_mark_region_end.default)
        [add_node] = _find_calls(graph_module, torch.ops.aten.add.Tensor)
        self.assertIs(end_node.args[0], add_node)

    def test_p3_reduce_scatter_inserted_after_all_reduce(self):
        graph_module = _run_pass(_trace_p2_p3_graph(with_end=True))

        reduce_scatter_nodes = _find_calls(graph_module, torch.ops.tensor_cast.reduce_scatter.default)
        add_comm = self._get_add_comm_input(graph_module)
        p3_reduce_scatter = (
            add_comm if add_comm.target is torch.ops.tensor_cast.reduce_scatter.default else add_comm.args[0]
        )

        reduce_scatter_positions = [_get_node_index(graph_module, node) for node in reduce_scatter_nodes]
        p3_position = _get_node_index(graph_module, p3_reduce_scatter)
        entry_position = min(pos for pos in reduce_scatter_positions if pos != p3_position)
        self.assertGreater(p3_position, entry_position)


class SequenceParallelPassEdgeTestCase(unittest.TestCase):
    def test_sequence_parallel_pass_owns_rewriters(self):
        sequence_parallel_pass = SequenceParallelPass()
        self.assertIsInstance(sequence_parallel_pass._p1_rewriter, Pattern1Rewriter)
        self.assertIsInstance(sequence_parallel_pass._p2_rewriter, Pattern2Rewriter)
        self.assertIsInstance(sequence_parallel_pass._p3_rewriter, Pattern3Rewriter)

    def test_no_tp(self):
        graph_module = _run_pass(_trace_p1_graph(rank_group=[0]))

        self.assertEqual(
            _count_calls(graph_module, torch.ops.tensor_cast.all_reduce.default),
            1,
            "should not rewrite when world_size=1",
        )
        self.assertEqual(_count_calls(graph_module, torch.ops.tensor_cast.reduce_scatter.default), 0)
        self.assertEqual(_count_calls(graph_module, torch.ops.tensor_cast.all_gather.default), 0)

    def test_skip_when_shard_dim_not_divisible(self):
        class Program(torch.nn.Module):
            def forward(self, x, res, w):
                all_reduce = torch.ops.tensor_cast.all_reduce.default(x, 0, RANK_GROUP)
                out, _ = torch.ops.tensor_cast.add_rms_norm2.default(all_reduce, res, w, EPS)
                return out

        graph_module = _run_pass(_trace_program(Program(), (1, 127, 4096), (1, 127, 4096), WEIGHT_SHAPE))

        self.assertEqual(_count_calls(graph_module, torch.ops.tensor_cast.all_reduce.default), 1)
        self.assertEqual(_count_calls(graph_module, torch.ops.tensor_cast.reduce_scatter.default), 0)
        self.assertEqual(_count_calls(graph_module, torch.ops.tensor_cast.all_gather.default), 0)

    def test_mixed_shardability_rewrites_valid_p1_candidate(self):
        class Program(torch.nn.Module):
            def forward(self, x, bad_x, w):
                all_reduce = torch.ops.tensor_cast.all_reduce.default(x, 0, RANK_GROUP)
                norm = torch.ops.tensor_cast.rms_norm.default(all_reduce, w, EPS)
                bad_all_reduce = torch.ops.tensor_cast.all_reduce.default(bad_x, 0, RANK_GROUP)
                return norm, bad_all_reduce

        graph_module = _run_pass(_trace_program(Program(), INPUT_SHAPE, (1, 127, 4096), WEIGHT_SHAPE))

        self.assertEqual(_count_calls(graph_module, torch.ops.tensor_cast.all_reduce.default), 1)
        self.assertEqual(_count_calls(graph_module, torch.ops.tensor_cast.reduce_scatter.default), 1)
        self.assertEqual(_count_calls(graph_module, torch.ops.tensor_cast.all_gather.default), 1)


class SequenceParallelPassNegativeTestCase(unittest.TestCase):
    def test_p1_all_reduce_without_norm_consumer(self):
        class Program(torch.nn.Module):
            def forward(self, x, y):
                all_reduce = torch.ops.tensor_cast.all_reduce.default(x, 0, RANK_GROUP)
                return torch.ops.aten.add.Tensor(all_reduce, y)

        graph_module = _trace_program(Program(), INPUT_SHAPE, INPUT_SHAPE)

        matched = Pattern1Rewriter().apply(graph_module.graph)
        self.assertEqual(matched, 0)
        self.assertEqual(_count_calls(graph_module, torch.ops.tensor_cast.reduce_scatter.default), 0)
        self.assertEqual(_count_calls(graph_module, torch.ops.tensor_cast.all_gather.default), 0)

    def test_p1_rms_norm_without_all_reduce_source(self):
        class Program(torch.nn.Module):
            def forward(self, x, w):
                return torch.ops.tensor_cast.rms_norm.default(x, w, EPS)

        graph_module = _trace_program(Program(), INPUT_SHAPE, WEIGHT_SHAPE)

        matched = Pattern1Rewriter().apply(graph_module.graph)
        self.assertEqual(matched, 0)
        self.assertEqual(_count_calls(graph_module, torch.ops.tensor_cast.reduce_scatter.default), 0)
        self.assertEqual(_count_calls(graph_module, torch.ops.tensor_cast.all_gather.default), 0)

    def test_p2_add_rms_norm2_without_all_reduce(self):
        class Program(torch.nn.Module):
            def forward(self, x, res, w):
                out, residual = torch.ops.tensor_cast.add_rms_norm2.default(x, res, w, EPS)
                return out, residual

        graph_module = _trace_program(Program(), INPUT_SHAPE, INPUT_SHAPE, WEIGHT_SHAPE)

        matched = Pattern2Rewriter().apply(graph_module.graph)
        self.assertEqual(matched, 0)
        self.assertEqual(_count_calls(graph_module, torch.ops.tensor_cast.reduce_scatter.default), 0)
        self.assertEqual(_count_calls(graph_module, torch.ops.tensor_cast.add_rms_norm2.default), 1)

    def test_p3_add_sibling_is_plain_placeholder(self):
        class Program(torch.nn.Module):
            def forward(self, x, res, plain, w, w2):
                all_reduce = torch.ops.tensor_cast.all_reduce.default(x, 0, RANK_GROUP)
                _, residual = torch.ops.tensor_cast.add_rms_norm2.default(all_reduce, res, w, EPS)
                add_node = torch.ops.aten.add.Tensor(residual, plain)
                return torch.ops.tensor_cast.rms_norm.default(add_node, w2, EPS)

        graph_module = _trace_program(Program(), INPUT_SHAPE, INPUT_SHAPE, INPUT_SHAPE, WEIGHT_SHAPE, WEIGHT_SHAPE)
        _mark_add_rms_norm2_sp_local(graph_module)

        matched = Pattern3Rewriter().apply(graph_module.graph)
        self.assertEqual(matched, 0)

    def test_p3_add_residual_not_from_add_rms_norm2(self):
        class Program(torch.nn.Module):
            def forward(self, x, y, w):
                parts = torch.ops.aten.split.Tensor(y, 64, 1)
                residual = parts[1]
                all_reduce = torch.ops.tensor_cast.all_reduce.default(x, 0, RANK_GROUP)
                add_node = torch.ops.aten.add.Tensor(residual, all_reduce)
                return torch.ops.tensor_cast.rms_norm.default(add_node, w, EPS)

        graph_module = _trace_program(Program(), LOCAL_INPUT_SHAPE, INPUT_SHAPE, WEIGHT_SHAPE)
        [residual] = _find_getitems(graph_module, 1)
        self.assertIsNot(residual.args[0].target, torch.ops.tensor_cast.add_rms_norm2.default)

        matched = Pattern3Rewriter().apply(graph_module.graph)
        self.assertEqual(matched, 0)


if __name__ == "__main__":
    unittest.main()
