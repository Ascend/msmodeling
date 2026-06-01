"""Unit tests for SequenceParallelPass ordered rewrites.

P1 (3): all_reduce -> [begin?] -> rms_norm | add_rms_norm
P2 (4): all_reduce -> add_rms_norm2, selective all_gather on getitems
P3 (4): getitem[1] + all_reduce -> add -> [end? -> copy*] -> norm
Edge (1): world_size=1 -> no rewrite
"""

import operator
import unittest

import torch
import torch.fx as fx
from tensor_cast import config
from tensor_cast.compilation.passes.sequence_parallel_pass import (
    Pattern1Rewriter,
    Pattern2Rewriter,
    Pattern3Rewriter,
    SequenceParallelPass,
)

INPUT_SHAPE = (1, 128, 4096)
WEIGHT_SHAPE = (4096,)
RANK_GROUP = [0, 1]  # world_size = 2


def _make_placeholder(graph, name, shape=INPUT_SHAPE):
    node = graph.placeholder(name)
    node.meta["val"] = torch.empty(shape, device="meta")
    return node


def _make_call(graph, target, args, shape=INPUT_SHAPE):
    node = graph.call_function(target, args=args)
    node.meta["val"] = torch.empty(shape, device="meta")
    return node


def _make_getitem(graph, node, index, shape=INPUT_SHAPE):
    getitem_node = graph.call_function(operator.getitem, args=(node, index))
    getitem_node.meta["val"] = torch.empty(shape, device="meta")
    return getitem_node


def _make_graph_module(graph):
    return fx.GraphModule(torch.nn.Module(), graph)


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


def _get_node_index(graph_module, node):
    for index, current_node in enumerate(graph_module.graph.nodes):
        if current_node is node:
            return index
    raise AssertionError("node not found in graph")


def _build_p2_p3_graph(begin_on_res=False, with_end=True, with_view=False, num_copies=0):
    """Build a graph with P2 (add_rms_norm2) + P3 (gi1 + ar -> add -> norm)."""
    graph = fx.Graph()
    x = _make_placeholder(graph, "x")
    res = _make_placeholder(graph, "res")
    w = _make_placeholder(graph, "w", WEIGHT_SHAPE)
    x2 = _make_placeholder(graph, "x2")
    w2 = _make_placeholder(graph, "w2", WEIGHT_SHAPE)

    all_reduce_1 = _make_call(graph, torch.ops.tensor_cast.all_reduce.default, (x, 0, RANK_GROUP))
    res_input = (
        _make_call(graph, torch.ops.tensor_cast._internal_mark_region_begin.default, (res, 0)) if begin_on_res else res
    )
    add_rms_norm2 = _make_call(
        graph,
        torch.ops.tensor_cast.add_rms_norm2.default,
        (all_reduce_1, res_input, w, 1e-5),
    )
    getitem0 = _make_getitem(graph, add_rms_norm2, 0)
    getitem1 = _make_getitem(graph, add_rms_norm2, 1)

    all_reduce_2 = _make_call(graph, torch.ops.tensor_cast.all_reduce.default, (x2, 0, RANK_GROUP))
    comm_output = (
        _make_call(graph, torch.ops.aten.reshape.default, (all_reduce_2, list(INPUT_SHAPE)))
        if with_view
        else all_reduce_2
    )
    add_node = _make_call(graph, torch.ops.aten.add.Tensor, (getitem1, comm_output))

    current = add_node
    if with_end:
        current = _make_call(graph, torch.ops.tensor_cast._internal_mark_region_end.default, (current, 0))
    for index in range(num_copies):
        current = _make_call(graph, torch.ops.tensor_cast._internal_copy_region.default, (current, index))
    tail = _make_call(graph, torch.ops.tensor_cast.rms_norm.default, (current, w2, 1e-5))
    graph.output((getitem0, tail))
    return _make_graph_module(graph)


class Pattern1RewriterTestCase(unittest.TestCase):
    def test_pattern1_rewriter_apply(self):
        graph = fx.Graph()
        x = _make_placeholder(graph, "x")
        w = _make_placeholder(graph, "w", WEIGHT_SHAPE)
        all_reduce = _make_call(graph, torch.ops.tensor_cast.all_reduce.default, (x, 0, RANK_GROUP))
        norm = _make_call(graph, torch.ops.tensor_cast.rms_norm.default, (all_reduce, w, 1e-5))
        graph.output(norm)
        graph_module = _make_graph_module(graph)

        rewritten = Pattern1Rewriter().apply(graph_module.graph)

        self.assertEqual(rewritten, 1)
        self.assertEqual(_count_calls(graph_module, torch.ops.tensor_cast.all_reduce.default), 1)
        self.assertEqual(_count_calls(graph_module, torch.ops.tensor_cast.reduce_scatter.default), 1)
        self.assertEqual(_count_calls(graph_module, torch.ops.tensor_cast.all_gather.default), 1)

    def test_p1_with_marker(self):
        graph = fx.Graph()
        x = _make_placeholder(graph, "x")
        w = _make_placeholder(graph, "w", WEIGHT_SHAPE)
        all_reduce = _make_call(graph, torch.ops.tensor_cast.all_reduce.default, (x, 0, RANK_GROUP))
        begin = _make_call(
            graph,
            torch.ops.tensor_cast._internal_mark_region_begin.default,
            (all_reduce, 0),
        )
        norm = _make_call(graph, torch.ops.tensor_cast.rms_norm.default, (begin, w, 1e-5))
        graph.output(norm)
        graph_module = _run_pass(_make_graph_module(graph))

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
        graph = fx.Graph()
        x = _make_placeholder(graph, "x")
        w = _make_placeholder(graph, "w", WEIGHT_SHAPE)
        all_reduce = _make_call(graph, torch.ops.tensor_cast.all_reduce.default, (x, 0, RANK_GROUP))
        norm = _make_call(graph, torch.ops.tensor_cast.rms_norm.default, (all_reduce, w, 1e-5))
        graph.output(norm)
        graph_module = _run_pass(_make_graph_module(graph))

        self.assertEqual(_count_calls(graph_module, torch.ops.tensor_cast.all_reduce.default), 0)
        [reduce_scatter] = _find_calls(graph_module, torch.ops.tensor_cast.reduce_scatter.default)
        [norm_node] = _find_calls(graph_module, torch.ops.tensor_cast.rms_norm.default)
        [all_gather] = _find_calls(graph_module, torch.ops.tensor_cast.all_gather.default)
        self.assertIs(norm_node.args[0], reduce_scatter)
        self.assertIs(all_gather.args[0], norm_node)

    def test_p1_add_rms_norm(self):
        graph = fx.Graph()
        x = _make_placeholder(graph, "x")
        res = _make_placeholder(graph, "res")
        w = _make_placeholder(graph, "w", WEIGHT_SHAPE)
        all_reduce = _make_call(graph, torch.ops.tensor_cast.all_reduce.default, (x, 0, RANK_GROUP))
        begin = _make_call(
            graph,
            torch.ops.tensor_cast._internal_mark_region_begin.default,
            (all_reduce, 0),
        )
        norm = _make_call(graph, torch.ops.tensor_cast.add_rms_norm.default, (begin, res, w, 1e-5))
        graph.output(norm)
        graph_module = _run_pass(_make_graph_module(graph))

        self.assertEqual(_count_calls(graph_module, torch.ops.tensor_cast.all_reduce.default), 0)
        [norm_node] = _find_calls(graph_module, torch.ops.tensor_cast.add_rms_norm.default)
        self.assertTrue(_has_user(norm_node, torch.ops.tensor_cast.all_gather.default))


class Pattern2RewriterTestCase(unittest.TestCase):
    def test_pattern2_rewriter_apply(self):
        graph = fx.Graph()
        x = _make_placeholder(graph, "x")
        res = _make_placeholder(graph, "res")
        w = _make_placeholder(graph, "w", WEIGHT_SHAPE)
        all_reduce = _make_call(graph, torch.ops.tensor_cast.all_reduce.default, (x, 0, RANK_GROUP))
        norm2 = _make_call(
            graph,
            torch.ops.tensor_cast.add_rms_norm2.default,
            (all_reduce, res, w, 1e-5),
        )
        getitem0 = _make_getitem(graph, norm2, 0)
        graph.output(getitem0)
        graph_module = _make_graph_module(graph)

        rewritten = Pattern2Rewriter().apply(graph_module.graph)

        self.assertEqual(rewritten, 1)
        self.assertEqual(_count_calls(graph_module, torch.ops.tensor_cast.all_reduce.default), 1)
        self.assertEqual(_count_calls(graph_module, torch.ops.tensor_cast.reduce_scatter.default), 1)
        self.assertEqual(_count_calls(graph_module, torch.ops.tensor_cast.all_gather.default), 1)

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
        graph_module = _run_pass(_build_p2_p3_graph(with_end=True))
        self._assert_p2_p3_middle(graph_module)

    def test_p2_middle_with_marker(self):
        graph_module = _run_pass(_build_p2_p3_graph(begin_on_res=True, with_end=True))
        self._assert_p2_p3_middle(graph_module)

    def test_p2_residual_fanout(self):
        graph = fx.Graph()
        x = _make_placeholder(graph, "x")
        res = _make_placeholder(graph, "res")
        w = _make_placeholder(graph, "w", WEIGHT_SHAPE)
        all_reduce = _make_call(graph, torch.ops.tensor_cast.all_reduce.default, (x, 0, RANK_GROUP))
        norm2 = _make_call(
            graph,
            torch.ops.tensor_cast.add_rms_norm2.default,
            (all_reduce, res, w, 1e-5),
        )
        getitem0 = _make_getitem(graph, norm2, 0)
        getitem1 = _make_getitem(graph, norm2, 1)
        other = _make_placeholder(graph, "other")
        add_node = _make_call(graph, torch.ops.aten.add.Tensor, (getitem1, other))
        graph.output((getitem0, add_node))
        graph_module = _run_pass(_make_graph_module(graph))

        for node in _find_getitems(graph_module, 1):
            self.assertTrue(_has_user(node, torch.ops.tensor_cast.all_gather.default))

    def test_p2_dual_all_reduce(self):
        graph = fx.Graph()
        x1 = _make_placeholder(graph, "x1")
        x2 = _make_placeholder(graph, "x2")
        w = _make_placeholder(graph, "w", WEIGHT_SHAPE)
        all_reduce_1 = _make_call(graph, torch.ops.tensor_cast.all_reduce.default, (x1, 0, RANK_GROUP))
        all_reduce_2 = _make_call(graph, torch.ops.tensor_cast.all_reduce.default, (x2, 0, RANK_GROUP))
        norm2 = _make_call(
            graph,
            torch.ops.tensor_cast.add_rms_norm2.default,
            (all_reduce_1, all_reduce_2, w, 1e-5),
        )
        getitem0 = _make_getitem(graph, norm2, 0)
        graph.output(getitem0)
        graph_module = _run_pass(_make_graph_module(graph))

        self.assertEqual(_count_calls(graph_module, torch.ops.tensor_cast.all_reduce.default), 2)
        self.assertEqual(_count_calls(graph_module, torch.ops.tensor_cast.reduce_scatter.default), 0)
        self.assertEqual(_count_calls(graph_module, torch.ops.tensor_cast.all_gather.default), 0)
        [norm2_new] = _find_calls(graph_module, torch.ops.tensor_cast.add_rms_norm2.default)
        all_reduce_nodes = _find_calls(graph_module, torch.ops.tensor_cast.all_reduce.default)
        self.assertIn(norm2_new.args[0], all_reduce_nodes)
        self.assertIn(norm2_new.args[1], all_reduce_nodes)
        self.assertIsNot(norm2_new.args[0], norm2_new.args[1])

    def test_p2_marker_wrapped_comm(self):
        graph = fx.Graph()
        x = _make_placeholder(graph, "x")
        res = _make_placeholder(graph, "res")
        w = _make_placeholder(graph, "w", WEIGHT_SHAPE)
        all_reduce = _make_call(graph, torch.ops.tensor_cast.all_reduce.default, (x, 0, RANK_GROUP))
        begin = _make_call(
            graph,
            torch.ops.tensor_cast._internal_mark_region_begin.default,
            (all_reduce, 0),
        )
        norm2 = _make_call(graph, torch.ops.tensor_cast.add_rms_norm2.default, (begin, res, w, 1e-5))
        getitem0 = _make_getitem(graph, norm2, 0)
        graph.output(getitem0)
        graph_module = _run_pass(_make_graph_module(graph))

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
        graph = fx.Graph()
        x = _make_placeholder(graph, "x", shape_2d)
        res = _make_placeholder(graph, "res", shape_2d)
        w = _make_placeholder(graph, "w", WEIGHT_SHAPE)
        all_reduce = _make_call(
            graph,
            torch.ops.tensor_cast.all_reduce.default,
            (x, 0, RANK_GROUP),
            shape=shape_2d,
        )
        norm2 = _make_call(
            graph,
            torch.ops.tensor_cast.add_rms_norm2.default,
            (all_reduce, res, w, 1e-5),
            shape=shape_2d,
        )
        getitem0 = _make_getitem(graph, norm2, 0, shape=shape_2d)
        getitem1 = _make_getitem(graph, norm2, 1, shape=shape_2d)
        graph.output((getitem0, getitem1))
        graph_module = _run_pass(_make_graph_module(graph))

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
        graph = fx.Graph()
        x = _make_placeholder(graph, "x")
        res = _make_placeholder(graph, "res")
        w = _make_placeholder(graph, "w", WEIGHT_SHAPE)
        all_reduce = _make_call(graph, torch.ops.tensor_cast.all_reduce.default, (x, 0, RANK_GROUP))
        norm2 = _make_call(
            graph,
            torch.ops.tensor_cast.add_rms_norm2.default,
            (all_reduce, res, w, 1e-5),
        )
        getitem0 = _make_getitem(graph, norm2, 0)
        graph.output(getitem0)
        graph_module = _run_pass(_make_graph_module(graph))

        self.assertEqual(_count_calls(graph_module, torch.ops.tensor_cast.all_reduce.default), 0)
        for node in _find_getitems(graph_module, 0):
            self.assertTrue(_has_user(node, torch.ops.tensor_cast.all_gather.default))

    def test_p2_reduce_scatter_inserted_after_all_reduce(self):
        graph = fx.Graph()
        x = _make_placeholder(graph, "x")
        res = _make_placeholder(graph, "res")
        w = _make_placeholder(graph, "w", WEIGHT_SHAPE)
        all_reduce = _make_call(graph, torch.ops.tensor_cast.all_reduce.default, (x, 0, RANK_GROUP))
        norm2 = _make_call(
            graph,
            torch.ops.tensor_cast.add_rms_norm2.default,
            (all_reduce, res, w, 1e-5),
        )
        getitem0 = _make_getitem(graph, norm2, 0)
        graph.output(getitem0)

        graph_module = _make_graph_module(graph)
        Pattern2Rewriter().apply(graph_module.graph)

        [reduce_scatter] = _find_calls(graph_module, torch.ops.tensor_cast.reduce_scatter.default)
        self.assertLess(
            _get_node_index(graph_module, all_reduce),
            _get_node_index(graph_module, reduce_scatter),
        )

    def test_markerless_shared_entry_all_reduce_and_fused_tail(self):
        graph = fx.Graph()
        x = _make_placeholder(graph, "x")
        x2 = _make_placeholder(graph, "x2")
        x3 = _make_placeholder(graph, "x3")
        w = _make_placeholder(graph, "w", WEIGHT_SHAPE)
        w2 = _make_placeholder(graph, "w2", WEIGHT_SHAPE)
        w3 = _make_placeholder(graph, "w3", WEIGHT_SHAPE)

        all_reduce_0 = _make_call(graph, torch.ops.tensor_cast.all_reduce.default, (x, 0, RANK_GROUP))
        _make_call(graph, torch.ops.tensor_cast.rms_norm.default, (all_reduce_0, w, 1e-5))

        all_reduce_1 = _make_call(graph, torch.ops.tensor_cast.all_reduce.default, (x2, 0, RANK_GROUP))
        norm2 = _make_call(
            graph,
            torch.ops.tensor_cast.add_rms_norm2.default,
            (all_reduce_0, all_reduce_1, w2, 1e-5),
        )
        getitem0 = _make_getitem(graph, norm2, 0)
        getitem1 = _make_getitem(graph, norm2, 1)

        all_reduce_2 = _make_call(graph, torch.ops.tensor_cast.all_reduce.default, (x3, 0, RANK_GROUP))
        tail = _make_call(
            graph,
            torch.ops.tensor_cast.add_rms_norm.default,
            (getitem1, all_reduce_2, w3, 1e-5),
        )
        graph.output((getitem0, tail))
        graph_module = _run_pass(_make_graph_module(graph))

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
        graph = fx.Graph()
        res0 = _make_placeholder(graph, "res0")
        x1 = _make_placeholder(graph, "x1")
        x2 = _make_placeholder(graph, "x2")
        w1 = _make_placeholder(graph, "w1", WEIGHT_SHAPE)
        w2 = _make_placeholder(graph, "w2", WEIGHT_SHAPE)

        all_reduce_1 = _make_call(graph, torch.ops.tensor_cast.all_reduce.default, (x1, 0, RANK_GROUP))
        norm2_1 = _make_call(
            graph,
            torch.ops.tensor_cast.add_rms_norm2.default,
            (res0, all_reduce_1, w1, 1e-5),
        )
        getitem1 = _make_getitem(graph, norm2_1, 1)

        all_reduce_2 = _make_call(graph, torch.ops.tensor_cast.all_reduce.default, (x2, 0, RANK_GROUP))
        norm2_2 = _make_call(
            graph,
            torch.ops.tensor_cast.add_rms_norm2.default,
            (getitem1, all_reduce_2, w2, 1e-5),
        )
        getitem0 = _make_getitem(graph, norm2_2, 0)
        graph.output(getitem0)
        graph_module = _run_pass(_make_graph_module(graph))

        chained_getitem1 = _find_getitems(graph_module, 1)[0]
        self.assertFalse(
            _has_user(chained_getitem1, torch.ops.tensor_cast.all_gather.default),
            "intermediate residual feeding next P2 must stay local",
        )
        [norm2_2_new] = _find_calls(graph_module, torch.ops.tensor_cast.add_rms_norm2.default)[1:]
        self.assertIs(norm2_2_new.args[0], chained_getitem1)


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
        graph_module = _run_pass(_build_p2_p3_graph(with_end=True))
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
        graph_module = _run_pass(_build_p2_p3_graph(with_end=False))
        self._assert_p3_base(graph_module)

        comm = self._get_add_comm_input(graph_module)
        self.assertIs(comm.target, torch.ops.tensor_cast.reduce_scatter.default)

        [add_node] = _find_calls(graph_module, torch.ops.aten.add.Tensor)
        [norm] = _find_calls(graph_module, torch.ops.tensor_cast.rms_norm.default)
        self.assertIs(norm.args[0], add_node, "norm must consume add directly")

    def test_p3_with_view(self):
        graph_module = _run_pass(_build_p2_p3_graph(with_view=True))
        self._assert_p3_base(graph_module)

        comm = self._get_add_comm_input(graph_module)
        self.assertIs(comm.target, torch.ops.aten.reshape.default)
        self.assertIs(comm.args[0].target, torch.ops.tensor_cast.reduce_scatter.default)

    def test_p3_no_marker_copy_region(self):
        graph_module = _run_pass(_build_p2_p3_graph(with_end=False, num_copies=2))
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
        graph_module = _run_pass(_build_p2_p3_graph(with_end=True, num_copies=2))
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
        graph_module = _run_pass(_build_p2_p3_graph(with_end=True))

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
        graph = fx.Graph()
        x = _make_placeholder(graph, "x")
        w = _make_placeholder(graph, "w", WEIGHT_SHAPE)
        all_reduce = _make_call(graph, torch.ops.tensor_cast.all_reduce.default, (x, 0, [0]))
        norm = _make_call(graph, torch.ops.tensor_cast.rms_norm.default, (all_reduce, w, 1e-5))
        graph.output(norm)
        graph_module = _run_pass(_make_graph_module(graph))

        self.assertEqual(
            _count_calls(graph_module, torch.ops.tensor_cast.all_reduce.default),
            1,
            "should not rewrite when world_size=1",
        )
        self.assertEqual(_count_calls(graph_module, torch.ops.tensor_cast.reduce_scatter.default), 0)
        self.assertEqual(_count_calls(graph_module, torch.ops.tensor_cast.all_gather.default), 0)


class SequenceParallelPassNegativeTestCase(unittest.TestCase):
    def test_p1_all_reduce_without_norm_consumer(self):
        graph = fx.Graph()
        x = _make_placeholder(graph, "x")
        y = _make_placeholder(graph, "y")
        all_reduce = _make_call(graph, torch.ops.tensor_cast.all_reduce.default, (x, 0, RANK_GROUP))
        add_node = _make_call(graph, torch.ops.aten.add.Tensor, (all_reduce, y))
        graph.output(add_node)
        graph_module = _make_graph_module(graph)

        matched = Pattern1Rewriter().apply(graph_module.graph)
        self.assertEqual(matched, 0)
        self.assertEqual(_count_calls(graph_module, torch.ops.tensor_cast.reduce_scatter.default), 0)
        self.assertEqual(_count_calls(graph_module, torch.ops.tensor_cast.all_gather.default), 0)

    def test_p1_rms_norm_without_all_reduce_source(self):
        graph = fx.Graph()
        x = _make_placeholder(graph, "x")
        w = _make_placeholder(graph, "w", WEIGHT_SHAPE)
        norm = _make_call(graph, torch.ops.tensor_cast.rms_norm.default, (x, w, 1e-5))
        graph.output(norm)
        graph_module = _make_graph_module(graph)

        matched = Pattern1Rewriter().apply(graph_module.graph)
        self.assertEqual(matched, 0)
        self.assertEqual(_count_calls(graph_module, torch.ops.tensor_cast.reduce_scatter.default), 0)
        self.assertEqual(_count_calls(graph_module, torch.ops.tensor_cast.all_gather.default), 0)

    def test_p2_add_rms_norm2_without_all_reduce(self):
        graph = fx.Graph()
        x = _make_placeholder(graph, "x")
        res = _make_placeholder(graph, "res")
        w = _make_placeholder(graph, "w", WEIGHT_SHAPE)
        norm2 = _make_call(graph, torch.ops.tensor_cast.add_rms_norm2.default, (x, res, w, 1e-5))
        _make_getitem(graph, norm2, 0)
        graph.output(norm2)
        graph_module = _make_graph_module(graph)

        matched = Pattern2Rewriter().apply(graph_module.graph)
        self.assertEqual(matched, 0)
        self.assertEqual(_count_calls(graph_module, torch.ops.tensor_cast.reduce_scatter.default), 0)
        self.assertEqual(_count_calls(graph_module, torch.ops.tensor_cast.add_rms_norm2.default), 1)

    def test_p3_add_sibling_is_plain_placeholder(self):
        graph = fx.Graph()
        x = _make_placeholder(graph, "x")
        res = _make_placeholder(graph, "res")
        plain = _make_placeholder(graph, "plain")
        w = _make_placeholder(graph, "w", WEIGHT_SHAPE)
        w2 = _make_placeholder(graph, "w2", WEIGHT_SHAPE)
        all_reduce = _make_call(graph, torch.ops.tensor_cast.all_reduce.default, (x, 0, RANK_GROUP))
        norm2 = _make_call(
            graph,
            torch.ops.tensor_cast.add_rms_norm2.default,
            (all_reduce, res, w, 1e-5),
        )
        _make_getitem(graph, norm2, 0)
        getitem1 = _make_getitem(graph, norm2, 1)
        add_node = _make_call(graph, torch.ops.aten.add.Tensor, (getitem1, plain))
        tail = _make_call(graph, torch.ops.tensor_cast.rms_norm.default, (add_node, w2, 1e-5))
        graph.output(tail)
        graph_module = _make_graph_module(graph)

        matched = Pattern3Rewriter().apply(graph_module.graph)
        self.assertEqual(matched, 0)

    def test_p3_add_residual_not_from_add_rms_norm2(self):
        graph = fx.Graph()
        x = _make_placeholder(graph, "x")
        y = _make_placeholder(graph, "y")
        w = _make_placeholder(graph, "w", WEIGHT_SHAPE)
        all_reduce = _make_call(graph, torch.ops.tensor_cast.all_reduce.default, (x, 0, RANK_GROUP))
        add_node = _make_call(graph, torch.ops.aten.add.Tensor, (y, all_reduce))
        tail = _make_call(graph, torch.ops.tensor_cast.rms_norm.default, (add_node, w, 1e-5))
        graph.output(tail)
        graph_module = _make_graph_module(graph)

        matched = Pattern3Rewriter().apply(graph_module.graph)
        self.assertEqual(matched, 0)


if __name__ == "__main__":
    unittest.main()
