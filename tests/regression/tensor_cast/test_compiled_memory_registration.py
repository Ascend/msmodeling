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
from torch._inductor.freezing_utils import maybe_set_is_frozen_param

from tensor_cast.compilation import get_backend
from tensor_cast.compilation.constant_folding import fold_meta_constants
from tensor_cast.compilation.memory_registration import (
    mark_frozen_weight_origins,
    register_graph_weight_aliases,
)
from tensor_cast.device import TEST_DEVICE
from tensor_cast.performance_model.memory_tracker import MemoryTracker
from tensor_cast.runtime import Runtime
from tensor_cast.runtime_workload import RuntimeWorkloadTrace


def test_frozen_view_registered_on_every_invocation_and_replay():
    weight = torch.empty(8, 8, device="meta")
    module = torch.nn.Module()
    module.register_buffer("transposed", weight.t())
    module.register_buffer("copied", weight.clone())
    graph = torch.fx.Graph()
    inputs = graph.placeholder("inputs")
    view = graph.get_attr("transposed")
    copied = graph.get_attr("copied")
    hidden = graph.call_function(torch.ops.aten.mm.default, (inputs, view))
    output = graph.call_function(torch.ops.aten.mm.default, (hidden, copied))
    graph.output(output)
    execute = register_graph_weight_aliases(torch.fx.GraphModule(module, graph))
    for _ in range(2):
        tracker = MemoryTracker(TEST_DEVICE)
        tracker.register_static_weight(weight)
        inputs = torch.empty(1, 8, device="meta")
        with Runtime([], TEST_DEVICE, tracker) as runtime:
            out = execute(inputs)
            tracker.register_model_outputs(out)
        summary = tracker.get_forward_memory_summary()
        # Only the input and the independent copy remain chargeable.
        assert summary.request_input_bytes == 32 + 256
        assert summary.activation_peak_bytes == 64
        frozen = RuntimeWorkloadTrace.from_runtime(runtime)
        replayed = MemoryTracker(TEST_DEVICE)
        frozen.replay(Runtime([], TEST_DEVICE, replayed))
        replayed.analyze()
        assert replayed.get_forward_memory_summary() == summary


def test_compile_cache_hit_with_new_tracker():
    torch._dynamo.reset()
    model = torch.nn.Linear(8, 8, bias=False, device="meta").eval()
    compiled = torch.compile(model, backend=get_backend(), fullgraph=True)
    for _ in range(2):
        tracker = MemoryTracker(TEST_DEVICE)
        tracker.register_static_weight(model.weight)
        inputs = torch.empty(1, 8, device="meta")
        with Runtime([], TEST_DEVICE, tracker), torch.no_grad():
            out = compiled(inputs)
            tracker.register_model_outputs(out)
        assert tracker.get_forward_memory_summary().request_input_bytes == 32
    torch._dynamo.reset()


def test_storage_proof_handles_views_without_exempting_same_shape_or_copy():
    weight = torch.empty(8, 8, device="meta")
    aliases = (weight.t(), weight.detach(), weight[2:5], weight.view(torch.uint8))
    unrelated = (weight.clone(), weight.t().contiguous(), torch.empty_like(weight))
    tracker = MemoryTracker(TEST_DEVICE)
    tracker.register_static_weight(weight)
    tracker.register_compiled_weight_aliases((*aliases, *unrelated))
    assert all(id(t) in tracker._resident_tensors for t in aliases)
    assert all(id(t) not in tracker._resident_tensors for t in unrelated)


def test_registered_cache_is_not_reclassified_as_weight():
    cache = torch.empty(8, 8, device="meta")
    view = cache.t()
    tracker = MemoryTracker(TEST_DEVICE)
    tracker.register_kv_cache(cache)
    tracker.register_compiled_weight_aliases((view,))
    assert id(view) not in tracker._resident_tensors


def test_mixed_get_attr_registers_only_frozen_tensor_leaf_as_weight():
    weight = torch.empty(8, 8, device="meta")
    maybe_set_is_frozen_param(weight)
    literal = torch.empty(8, 8, device="meta")
    module = torch.nn.Module()
    module.mixed = (weight, literal)
    graph = torch.fx.Graph()
    inputs = graph.placeholder("inputs")
    mixed = graph.get_attr("mixed")
    frozen_leaf = graph.call_function(operator.getitem, (mixed, 0))
    literal_leaf = graph.call_function(operator.getitem, (mixed, 1))
    hidden = graph.call_function(torch.ops.aten.mm.default, (inputs, frozen_leaf))
    output = graph.call_function(torch.ops.aten.mm.default, (hidden, literal_leaf))
    graph.output(output)
    graph_module = torch.fx.GraphModule(module, graph)
    mark_frozen_weight_origins(graph_module)
    execute = register_graph_weight_aliases(graph_module)

    tracker = MemoryTracker(TEST_DEVICE)
    tracker.register_static_weight(weight)
    inputs = torch.empty(1, 8, device="meta")
    with Runtime([], TEST_DEVICE, tracker):
        out = execute(inputs)
        tracker.register_model_outputs(out)

    assert tracker.get_forward_memory_summary().request_input_bytes == 32 + 256


def _folded_constant_graph(attribute_name):
    weight = torch.empty(8, 8, device="meta")
    module = torch.nn.Module()
    module.register_buffer(attribute_name, weight)
    graph = torch.fx.Graph()
    inputs = graph.placeholder("inputs")
    source = graph.get_attr(attribute_name)
    copied = graph.call_function(torch.ops.aten.clone.default, (source,))
    output = graph.call_function(torch.ops.aten.mm.default, (inputs, copied))
    graph.output(output)
    gm = torch.fx.GraphModule(module, graph)
    mark_frozen_weight_origins(gm)
    return weight, register_graph_weight_aliases(fold_meta_constants(gm))


def test_folded_weight_copy_inherits_compiler_provenance():
    weight, execute = _folded_constant_graph("_frozen_param0")
    tracker = MemoryTracker(TEST_DEVICE)
    tracker.register_static_weight(weight)
    inputs = torch.empty(1, 8, device="meta")
    with Runtime([], TEST_DEVICE, tracker):
        out = execute(inputs)
        tracker.register_model_outputs(out)
    assert tracker.get_forward_memory_summary().request_input_bytes == 32


def test_unrelated_folded_constant_remains_chargeable():
    weight, execute = _folded_constant_graph("literal")
    tracker = MemoryTracker(TEST_DEVICE)
    tracker.register_static_weight(torch.empty_like(weight))
    inputs = torch.empty(1, 8, device="meta")
    with Runtime([], TEST_DEVICE, tracker):
        out = execute(inputs)
        tracker.register_model_outputs(out)
    assert tracker.get_forward_memory_summary().request_input_bytes == 32 + 256


def test_nested_graph_module_frozen_weight_is_registered():
    weight = torch.empty(8, 8, device="meta")
    child_module = torch.nn.Module()
    child_module.register_buffer("_frozen_param0", weight.clone())
    child_graph = torch.fx.Graph()
    child_input = child_graph.placeholder("inputs")
    child_weight = child_graph.get_attr("_frozen_param0")
    child_graph.output(child_graph.call_function(torch.ops.aten.mm.default, (child_input, child_weight)))
    child = torch.fx.GraphModule(child_module, child_graph)

    parent_module = torch.nn.Module()
    parent_module.add_module("child", child)
    parent_graph = torch.fx.Graph()
    parent_input = parent_graph.placeholder("inputs")
    parent_graph.output(parent_graph.call_module("child", (parent_input,)))
    parent = torch.fx.GraphModule(parent_module, parent_graph)

    mark_frozen_weight_origins(parent)
    execute = register_graph_weight_aliases(parent)
    tracker = MemoryTracker(TEST_DEVICE)
    tracker.register_static_weight(weight)
    inputs = torch.empty(1, 8, device="meta")
    with Runtime([], TEST_DEVICE, tracker):
        out = execute(inputs)
        tracker.register_model_outputs(out)
    assert tracker.get_forward_memory_summary().request_input_bytes == 32
