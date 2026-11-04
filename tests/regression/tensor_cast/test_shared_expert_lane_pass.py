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

import contextlib
import operator
import unittest
import unittest.mock

import torch
import torch.fx as fx
from tensor_cast import config
from tensor_cast.compilation.compile_backend import CompilerBackend
from tensor_cast.compilation.passes.shared_expert_lane_pass import SharedExpertLanePass
import tensor_cast.performance_model.builtin_model  # noqa: F401 - register builtin op cost handlers
from tensor_cast.performance_model.op_invoke_info import OpInvokeInfo

_LANE_BEGIN = torch.ops.tensor_cast._internal_mark_lane_begin.default
_LANE_END = torch.ops.tensor_cast._internal_mark_lane_end.default
_SPILL = torch.ops.tensor_cast._internal_spill.default
_WAIT = torch.ops.tensor_cast._internal_wait_and_bind.default
_RECORD = torch.ops.tensor_cast._internal_record.default

_FEATURE = "multistream_overlap_shared_expert"
_FEATURE_FIELDS = ("enable", "role", "stream_id", "force_mode", "allow_with_dfc")


@contextlib.contextmanager
def _override_lane_config(**overrides):
    lane_config = getattr(config.compilation, _FEATURE)
    original = {field: getattr(lane_config, field) for field in _FEATURE_FIELDS}
    try:
        for field, value in overrides.items():
            setattr(lane_config, field, value)
        yield
    finally:
        for field, value in original.items():
            setattr(lane_config, field, value)


def _value(*shape, dtype=torch.float32):
    return torch.empty(*shape, dtype=dtype)


def _set(node, value):
    node.meta["val"] = value
    return node


def _count(gm, target):
    return sum(1 for node in gm.graph.nodes if node.target == target)


def _nodes(gm, target):
    return [node for node in gm.graph.nodes if node.target == target]


def _stage_pairs(gm, stream_id=2):
    """Pair each lane stage wait with the tagged record that closes it."""
    pairs = {}
    pending = None
    for node in gm.graph.nodes:
        if node.target == _WAIT and node.args[1] == stream_id:
            pending = node
            continue
        if node.target == _RECORD and node.args[1] == stream_id and len(node.args) == 3 and pending is not None:
            pairs[node.args[2].split(":")[1]] = (pending, node)
            pending = None
    return pairs


def _stage_records(gm):
    """Tagged stage-completion records (window-token records carry ``:window:``)."""
    return [
        node
        for node in _nodes(gm, _RECORD)
        if len(node.args) == 3 and isinstance(node.args[2], str) and ":window:" not in node.args[2]
    ]


def _routed_deps(wait):
    return [dep for dep in wait.args[2] if dep.target == _RECORD and dep.args[1] == 0]


def _apply(gm, **overrides):
    settings = {"enable": True}
    settings.update(overrides)
    with _override_lane_config(**settings):
        SharedExpertLanePass()(gm)
    return gm


def _build_moe_graph(
    *,
    quantized=True,
    comm_in_lane=False,
    router=True,
    ep=True,
    nested_markers=False,
    activation="swiglu",
    join=True,
    join_op=None,
    second_region_output=False,
):
    """One MoE layer: routed branch + shared-expert lane region between markers."""
    graph = fx.Graph()
    x = graph.placeholder("x")
    w_router = graph.placeholder("w_router")
    w_up = graph.placeholder("w_up")
    w_down = graph.placeholder("w_down")
    w_scale = graph.placeholder("w_scale")
    topk_indices = graph.placeholder("topk_indices")

    # producer outside the region (stands for the post-attention norm)
    hidden = _set(graph.call_function(torch.ops.aten.mul.Tensor, args=(x, x)), _value(16, 64))

    # routed branch -------------------------------------------------------
    routed = hidden
    if router:
        routed = _set(graph.call_function(torch.ops.aten.mm.default, args=(routed, w_router)), _value(8, 16))
        routed = _set(
            graph.call_function(torch.ops.tensor_cast.init_routing_v2.default, args=(routed, topk_indices)),
            _value(8, 32),
        )
    if ep:
        routed = _set(
            graph.call_function(
                torch.ops.tensor_cast.all_to_all.default,
                args=(routed, [1, 1, 1, 1], [1, 1, 1, 1], 0, [0, 1, 2, 3]),
            ),
            _value(8, 32),
        )
    routed = _set(
        graph.call_function(
            torch.ops.tensor_cast.static_quant_linear.default,
            args=(routed, w_up, w_scale, None, None, None, None, torch.bfloat16),
        ),
        _value(8, 64),
    )
    combine_input = _set(graph.call_function(torch.ops.aten.cat.default, args=([routed, routed],)), _value(16, 64))
    combined = combine_input
    if ep:
        combined = _set(
            graph.call_function(
                torch.ops.tensor_cast.all_to_all.default,
                args=(combine_input, [1, 1, 1, 1], [1, 1, 1, 1], 0, [0, 1, 2, 3]),
            ),
            _value(16, 64),
        )
    unpermuted = _set(
        graph.call_function(torch.ops.tensor_cast.unpermute_tokens.default, args=(combined, topk_indices)),
        _value(16, 64),
    )
    routed_out = _set(graph.call_function(torch.ops.aten.sum.dim_IntList, args=(unpermuted, [-2])), _value(16, 64))

    # shared-expert lane --------------------------------------------------
    begin = graph.call_function(_LANE_BEGIN, args=(hidden, "shared_expert"))
    lane = begin
    if nested_markers:
        inner_begin = graph.call_function(_LANE_BEGIN, args=(lane, "shared_expert"))
        inner_end = graph.call_function(_LANE_END, args=(inner_begin, "shared_expert"))
        lane = inner_end
    if quantized:
        quantized_out = graph.call_function(torch.ops.tensor_cast.dynamic_quantize_symmetric.default, args=(lane, [1]))
        quantized_out.meta["val"] = (_value(16, 32, dtype=torch.int8), _value(16, 1))
        lane = quantized_out
    gate_up = _set(
        graph.call_function(
            torch.ops.tensor_cast.static_quant_linear.default,
            args=(lane, w_up, w_scale, None, None, None, None, torch.bfloat16),
        ),
        _value(16, 64),
    )
    split = graph.call_function(torch.ops.aten.split_with_sizes.default, args=(gate_up, [32, 32]))
    split.meta["val"] = (_value(16, 32), _value(16, 32))
    gate = _set(graph.call_function(operator.getitem, args=(split, 0)), _value(16, 32))
    up = _set(graph.call_function(operator.getitem, args=(split, 1)), _value(16, 32))
    if activation == "clamped_swiglu":
        activated = _set(
            graph.call_function(torch.ops.tensor_cast.clamped_swiglu.default, args=(gate, up, 7.0)), _value(16, 32)
        )
    else:
        activated = _set(graph.call_function(torch.ops.tensor_cast.swiglu.default, args=(gate, up)), _value(16, 32))
    lane = activated
    if quantized:
        quantized_act = graph.call_function(
            torch.ops.tensor_cast.dynamic_quantize_symmetric.default, args=(activated, [1])
        )
        quantized_act.meta["val"] = (_value(16, 32, dtype=torch.int8), _value(16, 1))
        lane = quantized_act
    down = _set(
        graph.call_function(
            torch.ops.tensor_cast.static_quant_linear.default,
            args=(lane, w_down, w_scale, None, None, None, None, torch.bfloat16),
        ),
        _value(16, 64),
    )
    lane = down
    if comm_in_lane:
        lane = _set(
            graph.call_function(torch.ops.tensor_cast.all_reduce.default, args=(lane, 0, [0, 1, 2, 3])), _value(16, 64)
        )
    end = graph.call_function(_LANE_END, args=(lane, "shared_expert"))
    end.meta["val"] = _value(16, 64)

    if join:
        consumer = join_op or torch.ops.aten.add.Tensor
        if consumer is torch.ops.tensor_cast.add_rms_norm2.default:
            out = _set(
                graph.call_function(consumer, args=(routed_out, end, torch.ones(64), 1e-6)),
                (_value(16, 64), _value(16, 64)),
            )
            graph.output(out[0] if not isinstance(out, fx.Node) else out)
        elif second_region_output:
            # Two distinct activations leave the region (`down` is also consumed
            # outside); a single shared wait would silently substitute the first
            # output's tensor for the second one.
            out = _set(graph.call_function(consumer, args=(routed_out, end, down)), _value(16, 64))
            graph.output(out)
        else:
            out = _set(graph.call_function(consumer, args=(routed_out, end)), _value(16, 64))
            graph.output(out)
    else:
        graph.output((routed_out, end))
    return fx.GraphModule({}, graph)


def _build_two_region_graph():
    """Two stacked MoE-like blocks, each with its own router, lane region and join."""
    graph = fx.Graph()
    x = graph.placeholder("x")
    w_router = graph.placeholder("w_router")
    w_up = graph.placeholder("w_up")
    w_down = graph.placeholder("w_down")
    w_scale = graph.placeholder("w_scale")
    topk_indices = graph.placeholder("topk_indices")
    hidden = _set(graph.call_function(torch.ops.aten.mul.Tensor, args=(x, x)), _value(16, 64))

    for _ in range(2):
        routed = _set(graph.call_function(torch.ops.aten.mm.default, args=(hidden, w_router)), _value(8, 16))
        routed = _set(
            graph.call_function(torch.ops.tensor_cast.init_routing_v2.default, args=(routed, topk_indices)),
            _value(8, 32),
        )
        routed = _set(
            graph.call_function(
                torch.ops.tensor_cast.all_to_all.default,
                args=(routed, [1, 1, 1, 1], [1, 1, 1, 1], 0, [0, 1, 2, 3]),
            ),
            _value(8, 32),
        )
        routed = _set(
            graph.call_function(
                torch.ops.tensor_cast.static_quant_linear.default,
                args=(routed, w_up, w_scale, None, None, None, None, torch.bfloat16),
            ),
            _value(8, 64),
        )
        combine = _set(graph.call_function(torch.ops.aten.cat.default, args=([routed, routed],)), _value(16, 64))
        combined = _set(
            graph.call_function(
                torch.ops.tensor_cast.all_to_all.default,
                args=(combine, [1, 1, 1, 1], [1, 1, 1, 1], 0, [0, 1, 2, 3]),
            ),
            _value(16, 64),
        )
        unpermuted = _set(
            graph.call_function(torch.ops.tensor_cast.unpermute_tokens.default, args=(combined, topk_indices)),
            _value(16, 64),
        )
        routed_out = _set(graph.call_function(torch.ops.aten.sum.dim_IntList, args=(unpermuted, [-2])), _value(16, 64))

        begin = graph.call_function(_LANE_BEGIN, args=(hidden, "shared_expert"))
        quantized = graph.call_function(torch.ops.tensor_cast.dynamic_quantize_symmetric.default, args=(begin, [1]))
        quantized.meta["val"] = (_value(16, 32, dtype=torch.int8), _value(16, 1))
        gate_up = _set(
            graph.call_function(
                torch.ops.tensor_cast.static_quant_linear.default,
                args=(quantized, w_up, w_scale, None, None, None, None, torch.bfloat16),
            ),
            _value(16, 64),
        )
        activated = _set(
            graph.call_function(torch.ops.tensor_cast.swiglu.default, args=(gate_up, gate_up)), _value(16, 64)
        )
        down = _set(
            graph.call_function(
                torch.ops.tensor_cast.static_quant_linear.default,
                args=(activated, w_down, w_scale, None, None, None, None, torch.bfloat16),
            ),
            _value(16, 64),
        )
        end = graph.call_function(_LANE_END, args=(down, "shared_expert"))
        end.meta["val"] = _value(16, 64)

        hidden = _set(graph.call_function(torch.ops.aten.add.Tensor, args=(routed_out, end)), _value(16, 64))

    graph.output(hidden)
    return fx.GraphModule({}, graph)


def _all_stage_pairs(gm, stream_id=2):
    """Every lane stage wait paired with the tagged record that closes it."""
    pairs = []
    pending = None
    for node in gm.graph.nodes:
        if node.target == _WAIT and node.args[1] == stream_id:
            pending = node
            continue
        if node.target == _RECORD and node.args[1] == stream_id and len(node.args) == 3 and pending is not None:
            pairs.append((node.args[2].split(":")[1], (pending, node)))
            pending = None
    return pairs


def _build_dfc_graph():
    gm = _build_moe_graph()
    graph = gm.graph
    routed = [n for n in graph.nodes if n.target == torch.ops.tensor_cast.all_to_all.default][0]
    x = [n for n in graph.nodes if n.op == "placeholder"][0]
    with graph.inserting_before(routed):
        fused = graph.call_function(
            torch.ops.tensor_cast.dispatch_ffn_combine_quant.default, args=(x, x, x, x, x, x, x, x)
        )
    fused.meta["val"] = _value(16, 64)
    routed.replace_all_uses_with(fused)
    return gm


class SharedExpertLanePassTest(unittest.TestCase):
    def test_disabled_by_default(self):
        gm = _build_moe_graph()
        before = len(list(gm.graph.nodes))
        with _override_lane_config(enable=False):
            SharedExpertLanePass()(gm)
        self.assertEqual(len(list(gm.graph.nodes)), before)
        self.assertEqual(_count(gm, _WAIT), 0)

    def test_region_ops_bound_to_stream(self):
        gm = _apply(_build_moe_graph())
        waits = _nodes(gm, _WAIT)
        lane_waits = [node for node in waits if node.args[1] == 2]
        self.assertGreaterEqual(len(lane_waits), 2)
        # every stage wait sits inside the marker pair
        stage_waits = [node for node in lane_waits if not any(a.target == _SPILL for a in node.users)]
        self.assertEqual(len(stage_waits), 4)
        for node in _nodes(gm, _RECORD):
            if node.args[1] == 2:
                self.assertEqual(len(node.args), 3)

    def test_markers_contain_only_stage_anchors(self):
        gm = _apply(_build_moe_graph())
        nodes = list(gm.graph.nodes)
        begin = next(node for node in nodes if node.target == _LANE_BEGIN)
        end = next(node for node in nodes if node.target == _LANE_END)
        inside = nodes[nodes.index(begin) + 1 : nodes.index(end)]
        anchors = [node for node in inside if node.target in (_WAIT, _RECORD)]
        self.assertTrue(anchors)
        for node in anchors:
            self.assertEqual(node.args[1], 2)

    def test_compute_ops_unchanged(self):
        gm = _build_moe_graph()
        before = [
            (node.target, len(node.args)) for node in gm.graph.nodes if node.target not in (_LANE_BEGIN, _LANE_END)
        ]
        _apply(gm)
        after = [
            (node.target, len(node.args))
            for node in gm.graph.nodes
            if node.target not in (_LANE_BEGIN, _LANE_END, _WAIT, _RECORD, _SPILL)
        ]
        self.assertEqual(before, after)

    def test_weights_not_gated(self):
        gm = _apply(_build_moe_graph())
        for node in gm.graph.nodes:
            if node.target == _WAIT:
                self.assertNotEqual(node.args[0].op, "get_attr")

    def test_nested_markers_merged(self):
        gm = _apply(_build_moe_graph(nested_markers=True))
        self.assertEqual(_count(gm, _LANE_BEGIN), 2)
        stage_waits = [node for node in _nodes(gm, _WAIT) if node.args[1] == 2]
        self.assertEqual(len(stage_waits), 4)

    def test_quantized_split_four_stages(self):
        gm = _apply(_build_moe_graph(quantized=True))
        records = [node for node in _stage_records(gm) if node.args[1] == 2]
        kinds = [node.args[2].split(":")[1] for node in records]
        self.assertEqual(kinds[:4], ["prep", "gate_up", "act", "down"])

    def test_non_quantized_three_stages(self):
        gm = _apply(_build_moe_graph(quantized=False))
        records = [node for node in _stage_records(gm) if node.args[1] == 2]
        kinds = [node.args[2].split(":")[1] for node in records]
        self.assertEqual(kinds[:3], ["gate_up", "act", "down"])

    def test_multi_output_ops_inside_region(self):
        gm = _apply(_build_moe_graph(quantized=True))
        kinds = [node.args[2].split(":")[1] for node in _stage_records(gm)]
        self.assertIn("prep", kinds)  # dynamic_quantize_symmetric (tuple output)
        self.assertIn("gate_up", kinds)  # split_with_sizes (tuple output)

    def test_control_ops_excluded_from_region(self):
        gm = _apply(_build_moe_graph(quantized=True))
        for record in _stage_records(gm):
            self.assertNotIn(record.args[0].target, (_LANE_BEGIN, _LANE_END, _WAIT, _RECORD, _SPILL))

    def test_prep_has_no_cross_stream_dependency(self):
        gm = _apply(_build_moe_graph())
        entry_wait, _ = _stage_pairs(gm)["prep"]
        deps = list(entry_wait.args[2])
        self.assertEqual(len(deps), 1)  # only the producer outside the region
        self.assertEqual(deps[0].args[0].target, torch.ops.aten.mul.Tensor)

    def test_comm_becomes_last_stage(self):
        gm = _apply(_build_moe_graph(comm_in_lane=True))
        kinds = [node.args[2].split(":")[1] for node in _stage_records(gm)]
        self.assertEqual(kinds[-1], "comm")
        comm_record = [
            node for node in _nodes(gm, _RECORD) if len(node.args) == 3 and node.args[2].split(":")[1] == "comm"
        ][0]
        self.assertEqual(comm_record.args[0].target, torch.ops.tensor_cast.all_reduce.default)

    def test_comm_stage_has_no_spill(self):
        gm = _apply(_build_moe_graph(comm_in_lane=True))
        roles = [node.args[1] for node in _nodes(gm, _SPILL)]
        self.assertNotIn("shared_expert:comm", roles)

    def test_join_waits_for_last_stage(self):
        gm = _apply(_build_moe_graph())
        add = next(node for node in gm.graph.nodes if node.target == torch.ops.aten.add.Tensor)
        wait = next(node for node in add.all_input_nodes if node.target == _WAIT)
        self.assertEqual(wait.args[1], 0)
        last_stage_record = [node for node in _stage_records(gm) if node.args[2].split(":")[1] == "down"][0]
        self.assertIn(last_stage_record, wait.args[2])

    def test_anchor_A_up_is_router_gemm(self):
        gm = _apply(_build_moe_graph())
        wait, _ = _stage_pairs(gm)["gate_up"]
        routed = _routed_deps(wait)
        self.assertTrue(routed)
        self.assertEqual(routed[0].args[0].target, torch.ops.aten.mm.default)

    def test_combine_start_used_when_present(self):
        gm = _apply(_build_moe_graph())
        wait, _ = _stage_pairs(gm)["down"]
        routed = _routed_deps(wait)
        self.assertTrue(routed)
        self.assertEqual(routed[0].args[0].target, torch.ops.aten.cat.default)

    def test_combine_start_missing_falls_back_to_tail(self):
        gm = _apply(_build_moe_graph(ep=False))
        wait, _ = _stage_pairs(gm)["down"]
        routed = _routed_deps(wait)
        self.assertTrue(routed)
        # no all_to_all -> down binds the last routed op before the region instead
        self.assertEqual(routed[0].args[0].target, torch.ops.aten.sum.dim_IntList)

    def test_whole_mode_when_router_missing(self):
        gm = _apply(_build_moe_graph(router=False))
        # one concurrent block + the join wait, and no stage anchors
        self.assertEqual(_count(gm, _WAIT), 2)
        self.assertEqual(_count(gm, _SPILL), 0)
        self.assertEqual(len(_stage_records(gm)), 1)

    def test_force_mode_staged_and_whole(self):
        gm_whole = _apply(_build_moe_graph(), force_mode="whole")
        gm_staged = _apply(_build_moe_graph(), force_mode="staged")
        self.assertEqual(len(_stage_records(gm_whole)), 1)
        self.assertEqual(len(_stage_records(gm_staged)), 4)

    def test_spill_targets_same_unit_successor(self):
        gm = _apply(_build_moe_graph())
        spills = _nodes(gm, _SPILL)
        roles = {node.args[1].split(":")[1] for node in spills}
        self.assertEqual(roles, {"prep", "gate_up", "act", "down"})
        targets = {}
        for spill in spills:
            roles_key = spill.args[1].split(":")[1]
            consumers = [user for user in spill.users if user.target != _RECORD]
            targets[roles_key] = consumers[0].target if consumers else None
        self.assertEqual(targets["prep"], torch.ops.tensor_cast.init_routing_v2.default)
        self.assertEqual(targets["gate_up"], torch.ops.tensor_cast.static_quant_linear.default)
        self.assertEqual(targets["act"], torch.ops.tensor_cast.unpermute_tokens.default)
        self.assertEqual(targets["down"], torch.ops.aten.add.Tensor)

    def test_spill_carries_wait_tag(self):
        gm = _apply(_build_moe_graph())
        records = {node.args[2] for node in _stage_records(gm)}
        window_records = {
            node.args[2]
            for node in _nodes(gm, _RECORD)
            if len(node.args) == 3 and isinstance(node.args[2], str) and ":window:" in node.args[2]
        }
        for spill in _nodes(gm, _SPILL):
            self.assertEqual(len(spill.args), 4)
            self.assertIn(spill.args[2], records)
            self.assertIn(spill.args[3], window_records)

    def test_spill_tokens_keep_spills_alive(self):
        gm = _apply(_build_moe_graph())
        for spill in _nodes(gm, _SPILL):
            self.assertTrue(any(user.target == _RECORD for user in spill.users))
        add = next(node for node in gm.graph.nodes if node.target == torch.ops.aten.add.Tensor)
        join_wait = next(node for node in add.all_input_nodes if node.target == _WAIT)
        spill_records = [user for node in _nodes(gm, _SPILL) for user in node.users if user.target == _RECORD]
        for record in spill_records:
            self.assertIn(record, join_wait.args[2])

    def test_dfc_yields_by_default(self):
        gm = _build_dfc_graph()
        before = len(list(gm.graph.nodes))
        _apply(gm)
        self.assertEqual(len(list(gm.graph.nodes)), before)
        self.assertEqual(_count(gm, _WAIT), 0)

    def test_dfc_allowed_with_flag(self):
        gm = _build_dfc_graph()
        _apply(gm, allow_with_dfc=True)
        self.assertGreater(_count(gm, _RECORD), 0)

    def test_unpaired_marker_ignored(self):
        gm = _build_moe_graph()
        graph = gm.graph
        begin = next(node for node in graph.nodes if node.target == _LANE_BEGIN)
        begin.args = (begin.args[0], "other_role")
        _apply(gm)
        self.assertEqual(_count(gm, _WAIT), 0)

    def test_idempotent(self):
        gm = _apply(_build_moe_graph())
        nodes_after_first = len(list(gm.graph.nodes))
        _apply(gm)
        self.assertEqual(len(list(gm.graph.nodes)), nodes_after_first)

    def test_idempotent_in_whole_mode(self):
        gm = _apply(_build_moe_graph(), force_mode="whole")
        nodes_after_first = len(list(gm.graph.nodes))
        waits_after_first = _count(gm, _WAIT)
        self.assertEqual(waits_after_first, 2)  # one concurrent block + the join wait
        _apply(gm, force_mode="whole")
        self.assertEqual(len(list(gm.graph.nodes)), nodes_after_first)
        self.assertEqual(_count(gm, _WAIT), waits_after_first)

    def test_idempotent_with_single_mixed_stage(self):
        # Without a router the region collapses to one concurrent block, whose only
        # wait sits in front of the begin marker instead of a compute op.
        gm = _apply(_build_moe_graph(router=False))
        nodes_after_first = len(list(gm.graph.nodes))
        self.assertEqual(len([n for n in _nodes(gm, _RECORD) if len(n.args) == 3]), 1)
        _apply(gm)
        self.assertEqual(len(list(gm.graph.nodes)), nodes_after_first)

    def test_window_tokens_are_kept_alive(self):
        """Window records only feed the overrun metric; the join wait must consume them."""
        gm = _apply(_build_moe_graph())
        window_records = [
            node
            for node in _nodes(gm, _RECORD)
            if len(node.args) == 3 and isinstance(node.args[2], str) and ":window:" in node.args[2]
        ]
        self.assertTrue(window_records)
        add = next(node for node in gm.graph.nodes if node.target == torch.ops.aten.add.Tensor)
        join_wait = next(node for node in add.all_input_nodes if node.target == _WAIT)
        for record in window_records:
            self.assertIn(record, join_wait.args[2])

    def test_stream_id_respected(self):
        gm = _apply(_build_moe_graph(), stream_id=3)
        self.assertTrue(_nodes(gm, _WAIT))
        for wait in _nodes(gm, _WAIT):
            if wait.args[1] not in (0,):
                self.assertEqual(wait.args[1], 3)

    def test_no_performance_model_invocation(self):
        class _Exploding:
            def process_op(self, op_invoke_info):  # pragma: no cover - must not be called
                raise AssertionError("performance model must not run at compile time")

            def get_classifiers(self):
                return None

        gm = _build_moe_graph()
        original = CompilerBackend._instance_models if hasattr(CompilerBackend, "_instance_models") else None
        with unittest.mock.patch.object(OpInvokeInfo, "get_perf_properties", side_effect=AssertionError("no")):
            _apply(gm)
        self.assertGreater(_count(gm, _WAIT), 0)
        del original

    def test_regions_are_lowered_independently(self):
        gm = _apply(_build_two_region_graph())
        pairs = _all_stage_pairs(gm)
        # Four stages per region, and no stage tag is shared between the two regions.
        self.assertEqual(len(pairs), 8)
        tags = [record.args[2] for _, (_, record) in pairs]
        self.assertEqual(len(set(tags)), 8)
        self.assertEqual(len({tag.split(":")[-1] for tag in tags}), 2)  # two begin markers

        # Every spill references a tag published by its own region.
        spills = _nodes(gm, _SPILL)
        self.assertTrue(spills)
        for spill in spills:
            self.assertIn(spill.args[2], tags)

        # Each join consumes its own region's completion ticket.
        joins = [node for node in gm.graph.nodes if node.target == torch.ops.aten.add.Tensor]
        self.assertEqual(len(joins), 2)
        for join in joins:
            wait = next(node for node in join.all_input_nodes if node.target == _WAIT)
            deps = [dep for dep in wait.args[2] if dep.target == _RECORD and dep.args[1] == 2]
            self.assertTrue(deps)
            self.assertTrue(all(dep.args[2] in tags for dep in deps))

    def test_second_region_anchors_stay_in_its_own_layer(self):
        gm = _apply(_build_two_region_graph())
        gate_up = [pair for kind, pair in _all_stage_pairs(gm) if kind == "gate_up"]
        self.assertEqual(len(gate_up), 2)
        anchors = [_routed_deps(wait)[0].args[0] for wait, _ in gate_up]
        self.assertEqual(len(set(anchors)), 2, "both regions resolved A_up to the same GEMM")
        for anchor in anchors:
            self.assertEqual(anchor.target, torch.ops.aten.mm.default)

        # Each anchor feeds one of the two routers (looked through the records the pass
        # inserted), so each region resolved its own router projection.
        from tensor_cast.compilation.passes.control_flow import collect_transparent

        order = {node: index for index, node in enumerate(gm.graph.nodes)}
        routers = [node for node in gm.graph.nodes if node.target == torch.ops.tensor_cast.init_routing_v2.default]
        self.assertEqual(len(routers), 2)
        fed = [
            next(
                node
                for node in collect_transparent([anchor], forward=True, program_order=order)
                if node.target == torch.ops.tensor_cast.init_routing_v2.default
            )
            for anchor in anchors
        ]
        self.assertEqual(len(set(fed)), 2)

        # The second region's router GEMM sits after the first region's join.
        joins = [node for node in gm.graph.nodes if node.target == torch.ops.aten.add.Tensor]
        self.assertGreater(order[anchors[1]], order[joins[0]])


class LaneUnknownOpTest(unittest.TestCase):
    """Ops outside the unit table must not be used as spill landing points."""

    def _graph_with_unknown_tail_op(self):
        gm = _build_moe_graph()
        graph = gm.graph
        # Insert an op the unit table does not know about between the routed chain
        # and the join: it must be skipped, not gated.
        add = next(node for node in graph.nodes if node.target == torch.ops.aten.add.Tensor)
        routed = next(node for node in add.all_input_nodes if node.target != _LANE_END)
        with graph.inserting_before(add):
            unknown = graph.call_function(torch.ops.aten.hardsigmoid.default, args=(routed,))
        unknown.meta["val"] = _value(16, 64)
        add.args = tuple(unknown if arg is routed else arg for arg in add.args)
        return gm, unknown

    def test_unknown_op_is_not_a_spill_landing_point(self):
        gm, unknown = self._graph_with_unknown_tail_op()
        _apply(gm)
        down_spill = next(spill for spill in _nodes(gm, _SPILL) if spill.args[1].endswith(":down"))
        self.assertNotIn(unknown, down_spill.users.keys())
        self.assertIn(torch.ops.aten.add.Tensor, {user.target for user in down_spill.users})

    def test_unknown_op_still_gets_join_wait(self):
        gm, unknown = self._graph_with_unknown_tail_op()
        _apply(gm)
        waits = [node for node in _nodes(gm, _WAIT) if node.args[1] == 0]
        consumers = {user.target for wait in waits for user in wait.users}
        self.assertIn(torch.ops.aten.add.Tensor, consumers)


class LaneJoinVariantsTest(unittest.TestCase):
    """The join must not be hard-coded to ``aten.add``: models differ here."""

    def _consumer_of_join_wait(self, gm):
        waits = [node for node in _nodes(gm, _WAIT) if node.args[1] == 0]
        self.assertTrue(waits)
        return {user for wait in waits for user in wait.users}

    def test_join_plain_add(self):
        gm = _apply(_build_moe_graph())
        self.assertIn(torch.ops.aten.add.Tensor, {node.target for node in self._consumer_of_join_wait(gm)})

    def test_join_fused_add_rms_norm(self):
        gm = _apply(_build_moe_graph(join_op=torch.ops.tensor_cast.add_rms_norm2.default))
        consumers = self._consumer_of_join_wait(gm)
        self.assertIn(torch.ops.tensor_cast.add_rms_norm2.default, {node.target for node in consumers})

    def test_join_attention_like_consumer(self):
        gm = _apply(_build_moe_graph(join_op=torch.ops.tensor_cast.attention.default))
        consumers = self._consumer_of_join_wait(gm)
        self.assertIn(torch.ops.tensor_cast.attention.default, {node.target for node in consumers})

    def test_join_hc_post_like_consumer(self):
        gm = _apply(_build_moe_graph(join_op=torch.ops.tensor_cast.unpermute_tokens.default))
        consumers = self._consumer_of_join_wait(gm)
        self.assertIn(torch.ops.tensor_cast.unpermute_tokens.default, {node.target for node in consumers})

    def test_each_region_output_gets_its_own_wait(self):
        gm = _apply(_build_moe_graph(second_region_output=True))
        add = next(node for node in gm.graph.nodes if node.target == torch.ops.aten.add.Tensor)
        waits = [node for node in add.all_input_nodes if node.target == _WAIT]
        self.assertEqual(len(waits), 2)
        self.assertNotEqual(waits[0], waits[1])
        # Each wait carries its own argument, so both activations keep their value.
        self.assertEqual(len({wait.args[0] for wait in waits}), 2)

    def test_single_region_output_uses_one_wait(self):
        gm = _apply(_build_moe_graph())
        add = next(node for node in gm.graph.nodes if node.target == torch.ops.aten.add.Tensor)
        waits = [node for node in add.all_input_nodes if node.target == _WAIT]
        self.assertEqual(len(waits), 1)

    def test_spill_targets_join_for_any_consumer(self):
        for consumer in (
            torch.ops.aten.add.Tensor,
            torch.ops.tensor_cast.add_rms_norm2.default,
            torch.ops.tensor_cast.attention.default,
        ):
            gm = _apply(_build_moe_graph(join_op=consumer))
            down_spill = next(spill for spill in _nodes(gm, _SPILL) if spill.args[1].endswith(":down"))
            self.assertIn(consumer, {user.target for user in down_spill.users})


class LaneMarkingHelperTest(unittest.TestCase):
    """Model-side helper: markers only appear when the option is enabled."""

    def test_helper_is_identity_when_disabled(self):
        from tensor_cast.layers.internal import lane_marking_enabled, mark_lane_begin, mark_lane_end

        x = torch.empty(4, 8)
        with _override_lane_config(enable=False):
            self.assertFalse(lane_marking_enabled("shared_expert"))
            self.assertIs(mark_lane_begin(x, "shared_expert"), x)
            self.assertIs(mark_lane_end(x, "shared_expert"), x)

    def test_helper_emits_markers_when_enabled(self):
        from tensor_cast.layers.internal import lane_marking_enabled, mark_lane_begin

        x = torch.empty(4, 8)
        with _override_lane_config(enable=True):
            self.assertTrue(lane_marking_enabled("shared_expert"))
            self.assertFalse(lane_marking_enabled("other_role"))
            self.assertIs(mark_lane_begin(x, "other_role"), x)
        with _override_lane_config(enable=True):
            marked = mark_lane_begin(x, "shared_expert")
            self.assertEqual(marked.op if hasattr(marked, "op") else None, None)
            # eager execution keeps the tensor contract
            self.assertEqual(marked.shape, x.shape)

    def test_record_and_spill_accept_tags(self):
        x = torch.empty(4, 8)
        token = torch.ops.tensor_cast._internal_record.default(x, 2, "shared_expert:gate_up:r0")
        self.assertEqual(token.dtype, torch.int64)
        spill = torch.ops.tensor_cast._internal_spill.default(x, "shared_expert:gate_up", "shared_expert:gate_up:r0")
        self.assertEqual(spill.shape, x.shape)


class LaneModuleHelpersTest(unittest.TestCase):
    """Module-level helpers are executed at import time; cover them explicitly."""

    def test_op_resolves_registered_overload(self):
        from tensor_cast.compilation.passes.shared_expert_lane_pass import _op

        self.assertEqual(_op("tensor_cast", "_internal_spill"), _SPILL)
        self.assertIsNone(_op("tensor_cast", "not_a_real_op_name"))
        self.assertIsNone(_op("no_such_namespace", "_internal_spill"))

    def test_targets_skips_unknown_ops(self):
        from tensor_cast.compilation.passes.shared_expert_lane_pass import _targets

        resolved = _targets(("aten", "mm"), ("tensor_cast", "not_a_real_op_name"), ("tensor_cast", "all_to_all"))
        self.assertEqual(resolved, {torch.ops.aten.mm.default, torch.ops.tensor_cast.all_to_all.default})

    def test_internal_spill_properties_are_empty(self):
        from tensor_cast.performance_model import _internal_spill_properties
        from tensor_cast.performance_model.op_invoke_info import OpInvokeInfo

        x = torch.empty(4, 4)
        info = OpInvokeInfo(_SPILL, (x, "shared_expert:prep", "tag"), {}, x)
        properties = _internal_spill_properties(info)
        self.assertEqual(properties.memory_read_bytes, 0)
        self.assertEqual(properties.memory_write_bytes, 0)
        self.assertEqual(properties.compute_ops, {})

    def test_internal_anchor_estimator_is_free(self):
        from tensor_cast.device import TEST_DEVICE
        from tensor_cast.performance_model import _estimate_internal_multistream_anchor
        from tensor_cast.performance_model.op_invoke_info import OpInvokeInfo

        x = torch.empty(4, 4)
        for func in (_WAIT, _RECORD, _SPILL, _LANE_BEGIN, _LANE_END):
            info = OpInvokeInfo(func, (x, 0), {}, x)
            self.assertEqual(_estimate_internal_multistream_anchor(info, TEST_DEVICE).execution_time_s, 0.0)


if __name__ == "__main__":
    unittest.main()
