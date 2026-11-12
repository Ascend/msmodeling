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

import operator

import pytest
import torch
from torch.fx import Graph, GraphModule
from torch.fx.experimental.proxy_tensor import make_fx

from tensor_cast.compilation.passes.lift_quant_pass import LiftCombineQuantPass
from tensor_cast.compilation.topo_sort import stable_topo_sort


def _make_out_of_order_view_graph():
    graph = Graph()
    values = graph.placeholder("values")
    dimension = graph.placeholder("dimension")
    view = graph.call_function(torch.ops.aten.view.default, (values, [-1, None]))
    power = graph.call_function(operator.pow, (dimension, 2))
    view.args = (values, [-1, power])
    graph.output(view)
    return GraphModule(torch.nn.Module(), graph)


def test_stable_topo_sort_moves_dynamic_shape_dependencies_before_view():
    gm = _make_out_of_order_view_graph()

    stable_topo_sort(gm)
    gm.graph.lint()

    nodes = list(gm.graph.nodes)
    assert nodes.index(next(node for node in nodes if node.target is operator.pow)) < nodes.index(
        next(node for node in nodes if node.target is torch.ops.aten.view.default)
    )


def test_lift_combine_quant_pass_repairs_dynamic_shape_order_before_dce():
    graph = Graph()
    values = graph.placeholder("values")
    dimension = graph.placeholder("dimension")
    view = graph.call_function(torch.ops.aten.view.default, (values, [-1, None]))
    power = graph.call_function(operator.pow, (dimension, 2))
    view.args = (values, [-1, power])
    quantized = graph.call_function(torch.ops.tensor_cast.dynamic_quantize_symmetric.default, (view, []))
    output = graph.call_function(operator.getitem, (quantized, 0))
    graph.output(output)
    gm = GraphModule(torch.nn.Module(), graph)

    LiftCombineQuantPass()(gm)
    gm.graph.lint()


@pytest.mark.parametrize("asymmetric", [False, True])
@pytest.mark.parametrize("keyword_dims", [False, True])
@pytest.mark.parametrize("dims", [[-1], [1]])
def test_lift_quant_preserves_reduction_groups_across_reshape(asymmetric, keyword_dims, dims):
    graph = Graph()
    values = graph.placeholder("values")
    view = graph.call_function(torch.ops.aten.view.default, (values, [4, 8]))
    target = (
        torch.ops.tensor_cast.dynamic_quantize_asymmetric.default
        if asymmetric
        else torch.ops.tensor_cast.dynamic_quantize_symmetric.default
    )
    quantized = graph.call_function(
        target, (view,) if keyword_dims else (view, dims), {"dims": dims} if keyword_dims else {}
    )
    outputs = tuple(graph.call_function(operator.getitem, (quantized, i)) for i in range(3 if asymmetric else 2))
    graph.output(outputs)
    gm = GraphModule(torch.nn.Module(), graph)
    x = torch.empty(1, 4, 1, 8, 1, device="meta")
    values.meta["val"] = x
    view.meta["val"] = x.view(4, 8)
    expected = [tuple(t.shape) for t in gm(x)]

    LiftCombineQuantPass()(gm)

    assert [tuple(t.shape) for t in gm(x)] == expected
    quantized = gm.graph.find_nodes(op="call_function", target=target)[0]
    assert quantized.args[0].target is torch.ops.aten.view.default


@pytest.mark.parametrize("asymmetric", [False, True])
@pytest.mark.parametrize("keyword_dims", [False, True])
@pytest.mark.parametrize("tracing_mode", ["real", "symbolic"])
def test_lift_row_quant_preserves_scale_and_offset_layout(asymmetric, keyword_dims, tracing_mode):
    target = (
        torch.ops.tensor_cast.dynamic_quantize_asymmetric.default
        if asymmetric
        else torch.ops.tensor_cast.dynamic_quantize_symmetric.default
    )

    def quantize(x):
        viewed = x.reshape(4, 8).reshape(2, 2, 8)
        return target(viewed, dims=[-1]) if keyword_dims else target(viewed, [-1])

    x = torch.empty(1, 4, 8, device="meta")
    gm = make_fx(quantize, tracing_mode=tracing_mode)(x)
    expected = [(t.shape, t.dtype) for t in gm(x)]

    LiftCombineQuantPass()(gm)

    assert [(t.shape, t.dtype) for t in gm(x)] == expected
    quantized = gm.graph.find_nodes(op="call_function", target=target)[0]
    assert quantized.args[0].op == "placeholder"


def test_lift_quant_keeps_scalar_scale_lifting():
    graph = Graph()
    values = graph.placeholder("values")
    view = graph.call_function(torch.ops.aten.view.default, (values, [4, 8]))
    quantized = graph.call_function(torch.ops.tensor_cast.dynamic_quantize_symmetric.default, (view, []))
    payload = graph.call_function(operator.getitem, (quantized, 0))
    scale = graph.call_function(operator.getitem, (quantized, 1))
    graph.output((payload, scale))
    gm = GraphModule(torch.nn.Module(), graph)

    LiftCombineQuantPass()(gm)

    quantized = gm.graph.find_nodes(
        op="call_function", target=torch.ops.tensor_cast.dynamic_quantize_symmetric.default
    )[0]
    assert quantized.args[0].op == "placeholder"
    output, scale = gm(torch.empty(1, 4, 1, 8, 1, device="meta"))
    assert output.shape == (4, 8)
    assert scale.shape == ()
