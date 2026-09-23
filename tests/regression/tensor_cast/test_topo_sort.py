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

import torch
from torch.fx import Graph, GraphModule

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
