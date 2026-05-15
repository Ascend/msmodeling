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

        # Remember downstream users so we can reconnect them to the new view.
        downstream_users = list(all_reduce_node.users.keys())

        # Feed the pre-view tensor into all_reduce.
        all_reduce_node.replace_input_with(view_node, view_node.args[0])

        graph = view_node.graph
        new_args = (all_reduce_node,) + tuple(view_node.args[1:])
        with graph.inserting_after(all_reduce_node):
            new_view = graph.call_function(torch.ops.aten.view.default, new_args, dict(view_node.kwargs))
            new_view.meta = dict(view_node.meta)

        # Redirect previous consumers of all_reduce to the new view.
        for user in downstream_users:
            user.replace_input_with(all_reduce_node, new_view)

        graph.erase_node(view_node)
        return True
