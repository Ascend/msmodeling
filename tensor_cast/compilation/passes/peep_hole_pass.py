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

import logging

import torch
import torch.fx as fx

from ... import ops  # noqa: F401
from ..pass_base import TensorCastGraphModulePass


logger = logging.getLogger(__name__)


class PeepHolePass(TensorCastGraphModulePass):
    def __call__(self, gm: fx.GraphModule) -> fx.GraphModule:
        """
        Rules:
        1. Sink torch.ops.aten.view.default before tensor_cast.all_reduce.
        """
        graph = gm.graph
        modified = False

        for node in list(graph.nodes):
            if node.target == torch.ops.aten.view.default and self._sink_view_before_all_reduce(node):
                modified = True
                continue

        if modified:
            graph.eliminate_dead_code()
            gm.recompile()
        return gm

    def _sink_view_before_all_reduce(self, view_node: fx.Node) -> bool:
        """
        Move view nodes that only add a singleton dimension ahead of
        tensor_cast.all_reduce so existing mm->all_reduce patterns can match.
        """
        users = list(view_node.users.keys())
        if len(users) != 1:
            return False
        all_reduce_node = users[0]
        if all_reduce_node.target != torch.ops.tensor_cast.all_reduce.default:
            return False

        # Only the packed-token singleton expansion [T,H] -> [1,T,H]
        # preserves the shard axis used by SequenceParallelPass (0 -> 1).
        # Do not guess layouts such as [T,1,H] or [T,heads,dim].
        input_meta = view_node.args[0].meta.get("val")
        output_meta = view_node.meta.get("val")
        if input_meta is None or output_meta is None:
            return False
        input_shape = tuple(input_meta.shape)
        output_shape = tuple(output_meta.shape)
        if (
            len(input_shape) != 2
            or output_shape != (1, *input_shape)
            or any(not isinstance(dim, int) or dim <= 0 for dim in input_shape)
        ):
            return False

        # Remember downstream users so we can reconnect them to the new view.
        downstream_users = list(all_reduce_node.users.keys())
        original_view_meta = dict(view_node.meta)

        # Feed the pre-view tensor into all_reduce.
        all_reduce_node.replace_input_with(view_node, view_node.args[0])
        # The collective now preserves the pre-view shape. Keeping its old
        # metadata makes redundant-view elimination erase the required restore
        # to [batch, sequence, hidden] (notably the DSpark draft residual).
        for key in ("val", "tensor_meta"):
            if key in view_node.args[0].meta:
                all_reduce_node.meta[key] = view_node.args[0].meta[key]
            else:
                all_reduce_node.meta.pop(key, None)

        graph = view_node.graph
        view_args = tuple(view_node.args[1:])
        if view_args and isinstance(view_args[0], (list, tuple)):
            shape = list(view_args[0])
            # SP can subsequently replace all_reduce with reduce_scatter.
            # The validated leading singleton maps the packed token axis 0
            # to axis 1. Positive dimensions above avoid empty-view ambiguity.
            if len(shape) == 3 and -1 not in shape:
                shape[1] = -1
                view_args = (shape,) + view_args[1:]
        new_args = (all_reduce_node,) + view_args
        with graph.inserting_after(all_reduce_node):
            new_view = graph.call_function(torch.ops.aten.view.default, new_args, dict(view_node.kwargs))
            new_view.meta = original_view_meta

        # Redirect previous consumers of all_reduce to the new view.
        for user in downstream_users:
            user.replace_input_with(all_reduce_node, new_view)

        graph.erase_node(view_node)
        return True
