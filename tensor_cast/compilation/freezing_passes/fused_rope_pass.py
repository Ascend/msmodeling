import logging
import operator

import torch
import torch.fx as fx

from ..pass_base import TensorCastGraphModulePass
from ..topo_sort import stable_topo_sort

logger = logging.getLogger(__name__)


class FusedRopePass(TensorCastGraphModulePass):
    """Replace apply_rope nodes with fused_rope nodes for NPU profiling alignment.

    On NPU, the InterleaveRope/fused_rope_qk_mqa kernel fuses cos/sin lookup
    and partial RoPE into a single operator. This pass replaces apply_rope
    with fused_rope so the simulation graph matches the NPU profiling
    operator count.

    apply_rope(q, k, cos, sin, is_neox) -> (q_out, k_out)
    fused_rope(q_3d, k_3d, cos_sin, rotary_dim, is_neox) -> (q_3d_out, k_3d_out)

    apply_rope takes BHSD input and returns BSHD output. fused_rope models
    the NPU kernel signature, which takes flattened 3D Q/K tensors plus a
    concatenated cos_sin tensor. This pass inserts the required reshape/cat
    nodes so the replacement keeps apply_rope's public shape contract.
    """

    @staticmethod
    def _node_shape(node: fx.Node) -> tuple | None:
        if not hasattr(node, "meta"):
            return None
        val = node.meta.get("val")
        if val is not None and hasattr(val, "shape"):
            return tuple(val.shape)
        tensor_meta = node.meta.get("tensor_meta")
        if tensor_meta is not None and hasattr(tensor_meta, "shape"):
            return tuple(tensor_meta.shape)
        return None

    def __call__(self, gm: fx.GraphModule) -> fx.GraphModule:
        graph = gm.graph
        modified = False

        for node in list(graph.nodes):
            if node.op != "call_function":
                continue
            if node.target != torch.ops.tensor_cast.apply_rope.default:
                continue

            args = list(node.args)
            # apply_rope(q, k, cos, sin, is_neox)
            q_node = args[0]
            k_node = args[1]
            cos_node = args[2]
            sin_node = args[3]
            is_neox = args[4] if len(args) > 4 else True

            q_shape = self._node_shape(q_node)
            k_shape = self._node_shape(k_node)
            cos_shape = self._node_shape(cos_node)
            if q_shape is None or k_shape is None or cos_shape is None or len(q_shape) != 4 or len(k_shape) != 4:
                logger.debug("Skip fused_rope replacement because RoPE tensor shapes are unavailable.")
                continue

            batch, num_q_heads, seq_len, q_head_dim = q_shape
            _, num_kv_heads, _, k_head_dim = k_shape
            rotary_dim = cos_shape[-1]
            num_tokens = batch * seq_len
            if rotary_dim >= q_head_dim and rotary_dim >= k_head_dim:
                logger.debug("Skip fused_rope replacement for full-dimension RoPE.")
                continue

            with graph.inserting_before(node):
                q_bshd_node = graph.create_node(
                    "call_function",
                    torch.ops.aten.transpose.int,
                    args=(q_node, 1, 2),
                )
                k_bshd_node = graph.create_node(
                    "call_function",
                    torch.ops.aten.transpose.int,
                    args=(k_node, 1, 2),
                )
                q_3d_node = graph.create_node(
                    "call_function",
                    torch.ops.aten.reshape.default,
                    args=(q_bshd_node, (num_tokens, num_q_heads, q_head_dim)),
                )
                k_3d_node = graph.create_node(
                    "call_function",
                    torch.ops.aten.reshape.default,
                    args=(k_bshd_node, (num_tokens, num_kv_heads, k_head_dim)),
                )
                cos_sin_node = graph.create_node(
                    "call_function",
                    torch.ops.aten.cat.default,
                    args=([cos_node, sin_node], -1),
                )
                if len(cos_shape) != 2:
                    cos_sin_node = graph.create_node(
                        "call_function",
                        torch.ops.aten.reshape.default,
                        args=(cos_sin_node, (num_tokens, rotary_dim * 2)),
                    )
                fused_node = graph.create_node(
                    "call_function",
                    torch.ops.tensor_cast.fused_rope.default,
                    args=(q_3d_node, k_3d_node, cos_sin_node, rotary_dim, is_neox),
                )
                q_fused_node = graph.create_node(
                    "call_function",
                    operator.getitem,
                    args=(fused_node, 0),
                )
                k_fused_node = graph.create_node(
                    "call_function",
                    operator.getitem,
                    args=(fused_node, 1),
                )
                q_out_node = graph.create_node(
                    "call_function",
                    torch.ops.aten.reshape.default,
                    args=(q_fused_node, (batch, seq_len, num_q_heads, q_head_dim)),
                )
                k_out_node = graph.create_node(
                    "call_function",
                    torch.ops.aten.reshape.default,
                    args=(k_fused_node, (batch, seq_len, num_kv_heads, k_head_dim)),
                )
                output_node = graph.create_node(
                    "call_function",
                    tuple,
                    args=([q_out_node, k_out_node],),
                )

            node.replace_all_uses_with(output_node)
            modified = True

        if modified:
            stable_topo_sort(gm)
            gm.graph.eliminate_dead_code()
            gm.graph.lint()
            gm.recompile()

        return gm
