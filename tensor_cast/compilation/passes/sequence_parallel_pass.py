"""Sequence parallel pass with ordered pattern rewrites.

P1 + P2 run first:
  P1: all_reduce -> [region_begin?] -> rms_norm / add_rms_norm
      => reduce_scatter -> [...] -> norm -> all_gather
  P2: [region_begin?(residual) +] all_reduce -> add_rms_norm2
      => reduce_scatter -> add_rms_norm2 (selective all_gather)

P3 runs after P2 because it depends on the residual left local by P2.
  P3: getitem[1] + all_reduce[/view] -> add -> [region_end -> copy*] -> norm
      => reduce_scatter + residual -> add -> ... -> norm -> all_gather

M1 runs after P2 and folds the explicit TP full->local->full wrapper around
prefill MoE back into the local-token region:
  all_gather -> view -> gate -> TP slice -> topk
             -> TP slice(hidden) -> routed/shared experts -> all_gather
      => local view -> gate -> topk
                    -> routed/shared experts (local output)
"""

import logging
import operator

import torch
from torch.fx import Node

from ... import config
from ..pass_base import TensorCastGraphModulePass

logger = logging.getLogger(__name__)

# ── Op constants ──────────────────────────────────────────────────

_SINGLE_OUTPUT_NORMS = {
    torch.ops.tensor_cast.rms_norm.default,
    torch.ops.tensor_cast.add_rms_norm.default,
}
_ALL_REDUCE = torch.ops.tensor_cast.all_reduce.default
_REDUCE_SCATTER = torch.ops.tensor_cast.reduce_scatter.default
_ALL_GATHER = torch.ops.tensor_cast.all_gather.default
_REGION_BEGIN = torch.ops.tensor_cast._internal_mark_region_begin.default
_REGION_END = torch.ops.tensor_cast._internal_mark_region_end.default
_COPY_REGION = torch.ops.tensor_cast._internal_copy_region.default
_ADD_RMS_NORM2 = torch.ops.tensor_cast.add_rms_norm2.default
_ADD_RMS_NORM = torch.ops.tensor_cast.add_rms_norm.default
_ADD_OPS = {torch.ops.aten.add.Tensor}
_VIEW_OPS = {torch.ops.aten.view.default, torch.ops.aten.reshape.default}
_SLICE = torch.ops.aten.slice.Tensor
_MOE_TOPK = torch.ops.tensor_cast.moe_gating_top_k_softmax.default
_INIT_ROUTING = torch.ops.tensor_cast.init_routing_v2.default
_UNPERMUTE_TOKENS = torch.ops.tensor_cast.unpermute_tokens.default
_DSA_ATTENTION_OPS = {
    torch.ops.tensor_cast.mla_sparse_attention.default,
    torch.ops.tensor_cast.mla_sparse_attention_quant.default,
}
_STATIC_QUANT_LINEAR = torch.ops.tensor_cast.static_quant_linear.default
_OPROJ_LINEAR_OPS = {_STATIC_QUANT_LINEAR, torch.ops.aten.mm.default}
_TRANSPARENT_OPS = _VIEW_OPS | {_REGION_BEGIN, _REGION_END, _COPY_REGION}


# ── Helpers ────────────────────────────────────────────────────────


def _shard_dim(node: Node) -> int:
    """Return 0 for 2-D tensors, else 1 (seq dim)."""
    meta = node.meta.get("val")
    if meta is not None and hasattr(meta, "dim") and meta.dim() == 2:
        return 0
    return 1


def _world_size(rank_group) -> int:
    return len(rank_group) if isinstance(rank_group, (list, tuple)) else 1


def _meta_shape(node):
    meta = node.meta.get("val") if isinstance(node, Node) else None
    if meta is None or not hasattr(meta, "shape"):
        return None
    return tuple(meta.shape)


def _is_sp_local_shape(shape, expected_shape) -> bool:
    """Return True only when *shape* proves local-sequence shard layout.

    The rank-reduced case is for compiler IR that drops a leading batch
    dimension, so it is accepted only when the expected batch size is one.
    """
    if shape is None or expected_shape is None:
        return False
    shape = tuple(shape)
    expected_shape = tuple(expected_shape)
    if shape == expected_shape:
        return True
    if len(shape) == len(expected_shape) - 1 and expected_shape[0] == 1:
        return shape == expected_shape[1:]
    return False


def _infer_rs_shape(comm_node: Node, world_size: int):
    """Return the reduce_scatter output shape after any required view repair."""
    if not comm_node.args:
        return None
    inp = comm_node.args[0]
    inp_meta = inp.meta.get("val") if isinstance(inp, Node) else None
    if inp_meta is None or not hasattr(inp_meta, "dim"):
        return None

    comm_meta = comm_node.meta.get("val")
    if comm_meta is not None and hasattr(comm_meta, "dim") and inp_meta.dim() != comm_meta.dim():
        if abs(inp_meta.dim() - comm_meta.dim()) != 1:
            return None
        comm_shape = list(comm_meta.shape)
        # This path repairs compiler IR that squeezes the leading batch dim:
        # the input RS dim and comm-output RS dim name the same sequence axis.
        dim = _shard_dim(comm_node)
        if dim >= len(comm_shape) or comm_shape[dim] % world_size != 0:
            return None
        comm_shape[dim] = comm_shape[dim] // world_size
        return tuple(comm_shape)

    dim = _shard_dim(inp)
    if dim >= inp_meta.dim() or inp_meta.shape[dim] % world_size != 0:
        return None
    shape = list(inp_meta.shape)
    shape[dim] = shape[dim] // world_size
    return tuple(shape)


def _as_concrete_dim(dim):
    if isinstance(dim, int):
        return dim
    try:
        return int(dim)
    except (TypeError, ValueError, RuntimeError):
        return None


def _repair_view_shape_for_codegen(target_shape, inferred_dim: int):
    """Return a view shape that avoids unbound SymInt expressions in FX codegen."""
    shape = []
    for idx, dim in enumerate(target_shape):
        concrete_dim = _as_concrete_dim(dim)
        if concrete_dim is not None:
            shape.append(concrete_dim)
        elif idx == inferred_dim:
            shape.append(-1)
        else:
            shape.append(dim)
    return shape


def _insert_reduce_scatter(graph, comm_node, rank, rank_group):
    """Insert reduce_scatter and repair 2-D/3-D shape mismatches with a view."""
    inp = comm_node.args[0]
    rs_dim = _shard_dim(comm_node.args[0])
    ws = _world_size(rank_group)
    target_shape = _infer_rs_shape(comm_node, ws)
    with graph.inserting_after(comm_node):
        rs = graph.call_function(_REDUCE_SCATTER, (inp, rs_dim, rank, rank_group))
    if target_shape is None:
        return rs

    inp_meta = inp.meta.get("val") if isinstance(inp, Node) else None
    comm_meta = comm_node.meta.get("val")
    if (
        inp_meta is None
        or comm_meta is None
        or not hasattr(inp_meta, "dim")
        or not hasattr(comm_meta, "dim")
        or inp_meta.dim() == comm_meta.dim()
    ):
        return rs

    repair_dim = _shard_dim(comm_node)
    repair_shape = _repair_view_shape_for_codegen(target_shape, repair_dim)
    with graph.inserting_after(rs):
        return graph.call_function(torch.ops.aten.view.default, (rs, repair_shape))


def _infer_comm_rs_shape(comm_node):
    if not isinstance(comm_node, Node) or len(comm_node.args) < 3:
        return None
    return _infer_rs_shape(comm_node, _world_size(comm_node.args[2]))


def _is_comm_shardable(comm_node) -> bool:
    return _infer_comm_rs_shape(comm_node) is not None


def _is_sp_local_value(node, expected_shape=None, visited=None) -> bool:
    """Return True when node is proven to be on the local sequence shard."""
    if not isinstance(node, Node):
        return False
    if visited is None:
        visited = set()
    if node in visited:
        return False
    visited.add(node)

    if node.op == "call_function" and node.target is _ALL_GATHER:
        return False
    if node.op == "call_function" and node.target in _TRANSPARENT_OPS:
        return bool(node.args) and _is_sp_local_value(node.args[0], expected_shape, visited)
    if _is_sp_local_shape(_meta_shape(node), expected_shape):
        return True
    if node.op != "call_function":
        return False
    if node.target is _REDUCE_SCATTER:
        return True
    if node.target is operator.getitem and len(node.args) >= 2 and node.args[1] == 1 and isinstance(node.args[0], Node):
        return bool(node.args[0].meta.get("tensor_cast_sp_local"))
    return False


def _is_moe_sp_local_value(node, expected_shape=None, visited=None) -> bool:
    """MoE variant that also follows either output of an SP-local fused norm."""
    if not isinstance(node, Node):
        return False
    if node.meta.get("tensor_cast_sp_local"):
        return True
    if (
        node.op == "call_function"
        and node.target is operator.getitem
        and len(node.args) >= 2
        and node.args[1] in {0, 1}
        and isinstance(node.args[0], Node)
        and node.args[0].meta.get("tensor_cast_sp_local")
    ):
        return True
    return _is_sp_local_value(node, expected_shape, visited)


def _p2_match(node):
    if node.op != "call_function" or node.target is not _ADD_RMS_NORM2:
        return None
    ar_inputs = [
        arg
        for arg in node.args[:2]
        if isinstance(arg, Node) and arg.op == "call_function" and arg.target is _ALL_REDUCE
    ]
    if len(ar_inputs) != 1:
        return None
    comm = ar_inputs[0]
    other = node.args[1] if node.args[0] is comm else node.args[0]
    expected_shape = _infer_comm_rs_shape(comm)
    if expected_shape is None:
        return None
    if not _is_sp_local_value(other, expected_shape):
        return None
    return comm, node


def _p2_moe_match(node):
    """Match P2 structurally for MoE graphs before earlier layers are rewritten."""
    if node.op != "call_function" or node.target is not _ADD_RMS_NORM2:
        return None
    ar_inputs = [
        arg
        for arg in node.args[:2]
        if isinstance(arg, Node) and arg.op == "call_function" and arg.target is _ALL_REDUCE
    ]
    if len(ar_inputs) != 1:
        return None
    comm = ar_inputs[0]
    comm_input_local = _is_moe_sp_local_value(comm.args[0])
    if not comm_input_local and _infer_comm_rs_shape(comm) is None:
        return None
    return comm, node, comm_input_local


def _insert_all_gather(graph, node, dim, rank, rank_group):
    """Insert all_gather after *node* and redirect all downstream users."""
    if any(u.op == "call_function" and u.target is _ALL_GATHER for u in node.users):
        return
    with graph.inserting_after(node):
        ag = graph.call_function(_ALL_GATHER, (node, dim, rank, rank_group))
    for u in list(node.users):
        if u is not ag:
            u.replace_input_with(node, ag)


def _unwrap_comm(node):
    """Return (all_reduce_node, output_node) or (None, None)."""
    if isinstance(node, Node) and node.op == "call_function":
        if node.target is _ALL_REDUCE:
            return node, node
        if node.target in _VIEW_OPS:
            src = node.args[0] if node.args else None
            if isinstance(src, Node) and src.target is _ALL_REDUCE:
                return src, node
    return None, None


def _find_norm_after_add(add_node):
    """Walk add -> [region_end?] -> [copy_region*] -> norm."""
    users = list(add_node.users)
    if len(users) != 1:
        return None
    cur = users[0]
    if cur.op == "call_function" and cur.target is _REGION_END:
        users = list(cur.users)
        if len(users) != 1:
            return None
        cur = users[0]
    visited = set()
    while cur.op == "call_function" and cur.target is _COPY_REGION and id(cur) not in visited:
        visited.add(id(cur))
        users = list(cur.users)
        if len(users) != 1:
            return None
        cur = users[0]
    if cur.op == "call_function" and cur.target in _SINGLE_OUTPUT_NORMS:
        return cur
    return None


def _find_moe_norm_after_add(add_node):
    """MoE/repetition variant that accepts a region-begin before the norm."""
    users = list(add_node.users)
    if len(users) != 1:
        return None
    cur = users[0]
    if cur.op == "call_function" and cur.target is _REGION_END:
        users = list(cur.users)
        if len(users) != 1:
            return None
        cur = users[0]
    visited = set()
    while cur.op == "call_function" and cur.target is _COPY_REGION and id(cur) not in visited:
        visited.add(id(cur))
        users = list(cur.users)
        if len(users) != 1:
            return None
        cur = users[0]
    if cur.op == "call_function" and cur.target is _REGION_BEGIN:
        norm_users = [user for user in cur.users if user.op == "call_function" and user.target in _SINGLE_OUTPUT_NORMS]
        if len(norm_users) != 1:
            return None
        cur = norm_users[0]
    return cur if cur.op == "call_function" and cur.target in _SINGLE_OUTPUT_NORMS else None


def _is_p3_tail(getitem_node):
    """True if *getitem_node* is consumed by a full P3 pattern.

    A P3 tail is: getitem[1] -> add(getitem, comm_or_view) ->
    [region_end?] -> [copy_region*] -> norm, or a fused
    add_rms_norm(getitem, comm_or_view). The comm side must be an all_reduce
    (possibly through a view/reshape). If any part of this chain is missing,
    we must NOT skip the all_gather.
    """
    users = list(getitem_node.users)
    if len(users) != 1:
        return False
    tail_node = users[0]
    if tail_node.op != "call_function":
        return False
    if tail_node.target is _ADD_RMS_NORM and len(tail_node.args) >= 2:
        comm, _ = _unwrap_comm(tail_node.args[1])
        return comm is not None and _is_comm_shardable(comm)
    if tail_node.target not in _ADD_OPS:
        return False
    other = None
    for a in tail_node.args:
        if isinstance(a, Node) and a is not getitem_node:
            other = a
            break
    if other is None:
        return False
    comm, _ = _unwrap_comm(other)
    if comm is None:
        return False
    return _is_comm_shardable(comm) and _find_norm_after_add(tail_node) is not None


def _is_moe_p3_tail(getitem_node):
    """MoE/repetition variant of _is_p3_tail."""
    users = list(getitem_node.users)
    if len(users) != 1:
        return False
    tail_node = users[0]
    if tail_node.op != "call_function":
        return False
    if tail_node.target is _ADD_RMS_NORM and len(tail_node.args) >= 2:
        comm, _ = _unwrap_comm(tail_node.args[1])
        return comm is not None and _is_comm_shardable(comm)
    if tail_node.target not in _ADD_OPS:
        return False
    other = next((arg for arg in tail_node.args if isinstance(arg, Node) and arg is not getitem_node), None)
    comm, _ = _unwrap_comm(other)
    return comm is not None and _is_comm_shardable(comm) and _find_moe_norm_after_add(tail_node) is not None


def _is_moe_residual_add(getitem_node):
    """True when getitem[1] feeds a residual add whose other operand derives
    from the SP-local MoE output.

    ``MoeLocalTokenRewriter`` keeps the routed result on the local token shard,
    so the residual operand of ``residual + moe_output`` must stay local too;
    otherwise P2 gathers it back to the full-token domain and the add mixes
    1000/2000-token tensors (issue #322).
    """
    users = list(getitem_node.users)
    if len(users) != 1:
        return False
    tail_node = users[0]
    if tail_node.op != "call_function" or tail_node.target not in _ADD_OPS:
        return False
    other = next(
        (arg for arg in tail_node.args if isinstance(arg, Node) and arg is not getitem_node),
        None,
    )
    if not isinstance(other, Node):
        return False
    return (
        _first_ancestor(
            other,
            lambda n: n.op == "call_function" and n.target in {_UNPERMUTE_TOKENS, _MOE_TOPK},
        )
        is not None
    )


def _is_p2_chain_tail(getitem_node):
    """True if *getitem_node* feeds the residual input of a downstream P2 node."""
    users = list(getitem_node.users)
    if len(users) != 1:
        return False
    user = users[0]
    if user.op != "call_function" or user.target is not _ADD_RMS_NORM2:
        return False

    if len(user.args) < 2:
        return False
    if user.args[0] is not getitem_node and user.args[1] is not getitem_node:
        return False
    if user.meta.get("tensor_cast_sp_local"):
        return True

    for arg in user.args[:2]:
        if not isinstance(arg, Node) or arg is getitem_node:
            continue
        if arg.op != "call_function":
            continue
        if arg.target is _ALL_REDUCE:
            return _is_comm_shardable(arg)
        if arg.target is _REDUCE_SCATTER:
            return True
    return False


# ===================================================================
# Pattern3Rewriter
# ===================================================================


class _P3Match:
    """Data class for a matched P3 pattern."""

    __slots__ = ("comm_node", "comm_output", "add_node", "norm_node")

    def __init__(self, comm_node, comm_output, add_node, norm_node):
        self.comm_node = comm_node
        self.comm_output = comm_output
        self.add_node = add_node
        self.norm_node = norm_node


class Pattern3Rewriter:
    """P3: residual + all_reduce[/view] -> add -> [...] -> norm.

    Extracted as standalone class per spec requirement.
    """

    def apply(self, graph):
        if any(node.op == "call_function" and node.target is _MOE_TOPK for node in graph.nodes):
            return self._apply_moe(graph)

        matches = self._find(graph)
        for m in matches:
            self._rewrite(graph, m)
        for m in matches:
            if m.comm_node in graph.nodes and not m.comm_node.users:
                graph.erase_node(m.comm_node)
        return len(matches)

    __call__ = apply

    def _find(self, graph):
        out, seen = [], set()
        for node in graph.nodes:
            if not (
                node.op == "call_function"
                and node.target is operator.getitem
                and len(node.args) >= 2
                and node.args[1] == 1
                and isinstance(node.args[0], Node)
                and node.args[0].target is _ADD_RMS_NORM2
            ):
                continue
            if not node.args[0].meta.get("tensor_cast_sp_local"):
                continue
            fused_users = [u for u in node.users if u.op == "call_function" and u.target is _ADD_RMS_NORM]
            if len(fused_users) == 1:
                norm = fused_users[0]
                if id(norm) in seen:
                    continue
                other = norm.args[1] if len(norm.args) >= 2 else None
                comm, comm_out = _unwrap_comm(other)
                if comm is None:
                    continue
                if not _is_comm_shardable(comm):
                    continue
                seen.add(id(norm))
                out.append(_P3Match(comm, comm_out, norm, norm))
                continue
            add_users = [u for u in node.users if u.op == "call_function" and u.target in _ADD_OPS]
            if len(add_users) != 1:
                continue
            add_node = add_users[0]
            if id(add_node) in seen:
                continue
            other = None
            for a in add_node.args:
                if isinstance(a, Node) and a is not node:
                    other = a
                    break
            if other is None:
                continue
            comm, comm_out = _unwrap_comm(other)
            if comm is None:
                continue
            if not _is_comm_shardable(comm):
                continue
            norm = _find_norm_after_add(add_node)
            if norm is None:
                continue
            seen.add(id(add_node))
            out.append(_P3Match(comm, comm_out, add_node, norm))
        return out

    def _rewrite(self, graph, m):
        rank, rg = m.comm_node.args[1], m.comm_node.args[2]
        rs = _insert_reduce_scatter(graph, m.comm_node, rank, rg)
        if m.comm_output is m.comm_node:
            m.add_node.replace_input_with(m.comm_node, rs)
        else:
            m.comm_output.replace_input_with(m.comm_node, rs)
        ag_dim = _shard_dim(m.norm_node)
        _insert_all_gather(graph, m.norm_node, ag_dim, rank, rg)

    def _apply_moe(self, graph):
        matches = self._find_moe(graph)
        for match in matches:
            self._rewrite_moe(graph, match)
        for match in matches:
            if match.comm_node in graph.nodes and not match.comm_node.users:
                graph.erase_node(match.comm_node)
        return len(matches)

    def _find_moe(self, graph):
        matches = self._find(graph)
        seen = {id(match.add_node) for match in matches}
        for node in graph.nodes:
            if not (
                node.op == "call_function"
                and node.target is operator.getitem
                and len(node.args) >= 2
                and node.args[1] == 1
                and isinstance(node.args[0], Node)
                and node.args[0].target is _ADD_RMS_NORM2
                and node.args[0].meta.get("tensor_cast_sp_local")
            ):
                continue
            add_users = [user for user in node.users if user.op == "call_function" and user.target in _ADD_OPS]
            if len(add_users) != 1 or id(add_users[0]) in seen:
                continue
            add_node = add_users[0]
            other = next((arg for arg in add_node.args if isinstance(arg, Node) and arg is not node), None)
            comm, comm_out = _unwrap_comm(other)
            norm = _find_moe_norm_after_add(add_node)
            if comm is None or not _is_comm_shardable(comm) or norm is None:
                continue
            seen.add(id(add_node))
            matches.append(_P3Match(comm, comm_out, add_node, norm))
        return matches

    def _rewrite_moe(self, graph, match):
        self._rewrite(graph, match)
        match.add_node.meta["tensor_cast_sp_local"] = True


class Pattern1Rewriter:
    """P1: all_reduce -> [region_begin?] -> norm."""

    def apply(self, graph):
        matches = self._find(graph)
        for comm, marker, norm in matches:
            self._rewrite(graph, comm, marker, norm)
        return len(matches)

    @staticmethod
    def _find(graph):
        out = []
        for node in graph.nodes:
            if node.op != "call_function" or node.target not in _SINGLE_OUTPUT_NORMS:
                continue
            inp = node.args[0]
            if not isinstance(inp, Node):
                continue
            if inp.target is _REGION_BEGIN and isinstance(inp.args[0], Node) and inp.args[0].target is _ALL_REDUCE:
                if _is_comm_shardable(inp.args[0]):
                    out.append((inp.args[0], inp, node))
            elif inp.target is _ALL_REDUCE and _is_comm_shardable(inp):
                out.append((inp, None, node))
        return out

    @staticmethod
    def _rewrite(graph, comm, marker, norm):
        if not comm.args:
            return
        rank, rg = comm.args[1], comm.args[2]
        rs = _insert_reduce_scatter(graph, comm, rank, rg)
        if marker is not None:
            marker.replace_input_with(comm, rs)
        else:
            # Markerless path: the same all_reduce can feed both the entry
            # norm and add_rms_norm2(arg0). Markers normally provide a shared
            # region_begin wrapper for both consumers; without that wrapper,
            # we need to redirect the add_rms_norm2 edge explicitly.
            norm.replace_input_with(comm, rs)
            for user in list(comm.users):
                if (
                    user is not rs
                    and user is not norm
                    and user.op == "call_function"
                    and user.target is _ADD_RMS_NORM2
                    and len(user.args) >= 1
                    and user.args[0] is comm
                ):
                    user.replace_input_with(comm, rs)
        ag_dim = _shard_dim(norm)
        _insert_all_gather(graph, norm, ag_dim, rank, rg)


class Pattern2Rewriter:
    """P2: all_reduce -> add_rms_norm2 with selective gather on outputs."""

    def apply(self, graph):
        if any(node.op == "call_function" and node.target is _MOE_TOPK for node in graph.nodes):
            return self._apply_moe(graph)

        count = 0
        for node in list(graph.nodes):
            match = _p2_match(node)
            if match is None:
                continue
            comm, norm2 = match
            # Keep find+rewrite inline: downstream P2/P3 candidates may rely on
            # tensor_cast_sp_local metadata set by an earlier P2 in this walk.
            self._rewrite(graph, comm, norm2)
            count += 1
        return count

    @staticmethod
    def _rewrite(graph, comm, norm2):
        rank, rg = comm.args[1], comm.args[2]
        rs = _insert_reduce_scatter(graph, comm, rank, rg)
        norm2.replace_input_with(comm, rs)
        norm2.meta["tensor_cast_sp_local"] = True
        ag_dim = _shard_dim(norm2)
        for u in list(norm2.users):
            if u.op != "call_function" or u.target is not operator.getitem:
                continue
            if u.args[1] == 1 and (_is_p3_tail(u) or _is_p2_chain_tail(u)):
                continue  # residual stays local for P3
            _insert_all_gather(graph, u, ag_dim, rank, rg)

    def _apply_moe(self, graph):
        """Batch-rewrite all P2 patterns in a MoE graph from one snapshot."""
        matches = [match for node in graph.nodes if (match := _p2_moe_match(node)) is not None]
        for comm, norm2, comm_input_local in matches:
            self._rewrite_moe(graph, comm, norm2, comm_input_local)
        return len(matches)

    @staticmethod
    def _rewrite_moe(graph, comm, norm2, comm_input_local):
        rank, rg = comm.args[1], comm.args[2]
        if not comm_input_local:
            rs = _insert_reduce_scatter(graph, comm, rank, rg)
            norm2.replace_input_with(comm, rs)
            # ``add_rms_norm2`` derives its output shapes from its first input
            # (``torch.empty_like(x)``).  When the reduce-scattered attention
            # output lands in the residual slot, promote it to the
            # shape-defining ``x`` slot so the norm output truly becomes local;
            # the previous ``x`` moves to the residual slot and is restored to
            # the full-token domain by the selective all_gathers below.  This
            # keeps gate/topk and the MoE dispatch/combine token counts
            # consistent (issue #322).
            if norm2.args[0] is not rs:
                prev_x = norm2.args[0]
                norm2.args = (rs, prev_x, *norm2.args[2:])
        norm2.meta["tensor_cast_sp_local"] = True
        ag_dim = _shard_dim(norm2)
        for user in list(norm2.users):
            if user.op != "call_function" or user.target is not operator.getitem:
                continue
            if user.args[1] == 1 and (_is_moe_p3_tail(user) or _is_p2_chain_tail(user) or _is_moe_residual_add(user)):
                continue
            _insert_all_gather(graph, user, ag_dim, rank, rg)


def _first_ancestor(node, predicate, visited=None):
    if not isinstance(node, Node):
        return None
    if predicate(node):
        return node
    if visited is None:
        visited = set()
    if node in visited:
        return None
    visited.add(node)
    for arg in node.args:
        found = _first_ancestor(arg, predicate, visited) if isinstance(arg, Node) else None
        if found is not None:
            return found
    return None


def _first_descendant(node, predicate, stop_predicate=None):
    """Find a downstream FX node iteratively, optionally pruning boundaries."""
    queue, visited, cursor = list(node.users), set(), 0
    while cursor < len(queue):
        current = queue[cursor]
        cursor += 1
        if current in visited:
            continue
        visited.add(current)
        if predicate(current):
            return current
        if stop_predicate is None or not stop_predicate(current):
            queue.extend(current.users)
    return None


class MoeLocalTokenRewriter:
    """Keep prefill MoE gate, routed experts, and shared experts SP-local.

    ``ParallelMoELayer`` emits a generic DP-domain wrapper before compilation:
    the norm result is gathered, gate logits and hidden states are sliced back
    to the TP-local token range, and the routed result is gathered again.  Once
    P2 has made the norm/residual genuinely sequence-local, those operations
    are redundant and, more importantly, break propagation into the next
    layer.  This rewriter is deliberately anchored on the TensorCast MoE top-k
    and routing ops so ordinary tensor slices/all-gathers are never folded.
    """

    def apply(self, graph):
        matches = self._find(graph)
        for match in matches:
            self._rewrite(graph, *match)
        return len(matches)

    def _find(self, graph):
        matches = []
        for topk in graph.nodes:
            match = self._find_one(topk)
            if match is not None:
                matches.append(match)
        return matches

    @staticmethod
    def _find_one(topk):
        if topk.op != "call_function" or topk.target is not _MOE_TOPK or not topk.args:
            return None
        logits_slice = topk.args[0]
        if not isinstance(logits_slice, Node) or logits_slice.target is not _SLICE or not logits_slice.args:
            return None
        gate_logits = logits_slice.args[0]
        if not isinstance(gate_logits, Node):
            return None

        full_view = _first_ancestor(
            gate_logits,
            lambda n: n.op == "call_function"
            and n.target in _VIEW_OPS
            and n.args
            and isinstance(n.args[0], Node)
            and n.args[0].target is _ALL_GATHER,
        )
        if full_view is None:
            return None
        view_shape = full_view.args[1] if len(full_view.args) > 1 else None
        if not isinstance(view_shape, (list, tuple)) or -1 not in view_shape:
            return None
        gather_in = full_view.args[0]
        local_value = gather_in.args[0]
        if not isinstance(local_value, Node) or not _is_moe_sp_local_value(local_value):
            return None

        hidden_slices = [
            user
            for user in full_view.users
            if user.op == "call_function"
            and user.target is _SLICE
            and any(u.op == "call_function" and u.target is _INIT_ROUTING for u in user.users)
        ]
        if len(hidden_slices) != 1:
            return None

        topk_indices = [
            user
            for user in topk.users
            if user.op == "call_function"
            and user.target is operator.getitem
            and len(user.args) > 1
            and user.args[1] == 1
        ]
        if len(topk_indices) != 1:
            return None
        unpermute = _first_descendant(
            topk_indices[0],
            lambda node: node.op == "call_function" and node.target is _UNPERMUTE_TOKENS,
            lambda node: node.op == "output",
        )
        if unpermute is None:
            return None
        exit_gather = _first_descendant(
            unpermute,
            lambda node: node.op == "call_function" and node.target is _ALL_GATHER,
            lambda node: node.op == "call_function" and node.target in {_ALL_REDUCE, _REDUCE_SCATTER},
        )
        if exit_gather is None:
            return None
        return full_view, local_value, logits_slice, gate_logits, hidden_slices[0], exit_gather

    @staticmethod
    def _rewrite(graph, full_view, local_value, logits_slice, gate_logits, hidden_slice, exit_gather):
        # The view uses -1 for the token dimension in traced MoE graphs, so it
        # remains valid after replacing the full-token gather with a local value.
        full_view.replace_input_with(full_view.args[0], local_value)
        topk = next(iter(logits_slice.users))
        topk.replace_input_with(logits_slice, gate_logits)

        # The hidden slice now selects the entire local tensor.  Bypass it at
        # the routing boundary instead of relying on stale FX shape metadata.
        for user in list(hidden_slice.users):
            if user.op == "call_function" and user.target is _INIT_ROUTING:
                user.replace_input_with(hidden_slice, full_view)

        # Routed and shared branches meet downstream of this gather.  Keeping
        # the value local lets a following P2 match consume the local residual.
        full_shape = _meta_shape(exit_gather)
        local_shape = _meta_shape(exit_gather.args[0])
        full_tokens = full_shape[0] if full_shape else None
        local_tokens = local_shape[0] if local_shape else None
        for user in list(exit_gather.users):
            user.replace_input_with(exit_gather, exit_gather.args[0])
            MoeLocalTokenRewriter._mark_local_descendants(user, full_tokens, local_tokens)

        # In an expanded graph, the local MoE output is fused with the next
        # layer's input norm.  Only the normalized output must return to the
        # full-token attention domain; the residual output stays local for the
        # following o_proj reduce-scatter/P2 rewrite.
        boundary_norm = _first_descendant(
            exit_gather.args[0],
            lambda node: node.op == "call_function" and node.target is _ADD_RMS_NORM2,
            lambda node: node.op == "call_function" and node.target in {_ALL_GATHER, _REDUCE_SCATTER},
        )
        if boundary_norm is not None:
            boundary_norm.meta["tensor_cast_sp_local"] = True
            rank, rank_group = exit_gather.args[2], exit_gather.args[3]
            for user in list(boundary_norm.users):
                if user.op != "call_function" or user.target is not operator.getitem:
                    continue
                user.meta["tensor_cast_sp_local"] = True
                if user.args[1] == 0:
                    _insert_all_gather(graph, user, _shard_dim(boundary_norm), rank, rank_group)

    @staticmethod
    def _mark_local_descendants(start, full_tokens=None, local_tokens=None):
        """Propagate local-layout evidence through the post-MoE value chain."""
        queue, visited = [start], set()
        stop_ops = {_ALL_GATHER, _ALL_REDUCE, _REDUCE_SCATTER}
        while queue:
            node = queue.pop()
            if node in visited or node.op != "call_function" or node.target in stop_ops:
                continue
            visited.add(node)
            node.meta["tensor_cast_sp_local"] = True
            if (
                node.target in _VIEW_OPS
                and len(node.args) > 1
                and isinstance(node.args[1], (list, tuple))
                and full_tokens is not None
                and local_tokens is not None
            ):
                shape = list(node.args[1])
                # Do not rewrite flattened 2-D views when hidden_size happens to equal full_tokens.
                seq_dim = _shard_dim(node)
                if len(shape) == 3 and seq_dim < len(shape) and shape[seq_dim] == full_tokens:
                    shape[seq_dim] = local_tokens
                node.args = (node.args[0], shape, *node.args[2:])
            if node.target is _ADD_RMS_NORM2:
                continue
            queue.extend(node.users)


class DsaFullOProjRewriter:
    """Slice full-token DSA attention output before a replicated o_proj."""

    def apply(self, graph):
        matches = self._find(graph)
        for match in matches:
            self._rewrite(graph, *match)
        return len(matches)

    @staticmethod
    def _find(graph):
        matches = []
        for norm2 in graph.nodes:
            if norm2.op != "call_function" or norm2.target is not _ADD_RMS_NORM2:
                continue
            o_proj_out = next(
                (
                    arg
                    for arg in norm2.args[:2]
                    if isinstance(arg, Node)
                    and arg.op == "call_function"
                    and arg.target in _VIEW_OPS
                    and arg.args
                    and isinstance(arg.args[0], Node)
                    and arg.args[0].target in _OPROJ_LINEAR_OPS
                ),
                None,
            )
            if o_proj_out is None:
                continue
            linear = o_proj_out.args[0]
            attention = _first_ancestor(linear.args[0], lambda node: node.target in _DSA_ATTENTION_OPS)
            gather = _first_ancestor(linear.args[0], lambda node: node.target is _ALL_GATHER)
            if attention is None or gather is None or len(gather.args) < 4:
                continue
            full_shape = _meta_shape(attention)
            world_size = _world_size(gather.args[3])
            if not full_shape or world_size <= 1 or full_shape[0] % world_size != 0:
                continue
            full_tokens = full_shape[0]
            local_tokens = full_tokens // world_size
            matches.append((attention, norm2, gather.args[2], gather.args[3], full_tokens, local_tokens))
        return matches

    @staticmethod
    def _rewrite(graph, attention, norm2, rank, rank_group, full_tokens, local_tokens):
        with graph.inserting_after(attention):
            offset = rank_group.index(rank) * local_tokens
            local_attention = graph.call_function(_SLICE, (attention, 0, offset, offset + local_tokens))
        for user in list(attention.users):
            if user is not local_attention:
                user.replace_input_with(attention, local_attention)
        MoeLocalTokenRewriter._mark_local_descendants(local_attention, full_tokens, local_tokens)
        norm2.meta["tensor_cast_sp_local"] = True
        ag_dim = _shard_dim(norm2)
        for user in list(norm2.users):
            if user.op != "call_function" or user.target is not operator.getitem:
                continue
            # The normalized output returns to the full-token attention domain.
            # The residual output remains local across decoder layers and feeds
            # the next local o_proj/norm boundary directly.
            if user.args[1] == 1:
                continue
            _insert_all_gather(graph, user, ag_dim, rank, rank_group)


# ===================================================================
# SequenceParallelPass
# ===================================================================


class SequenceParallelPass(TensorCastGraphModulePass):
    """Sequence-parallel pass with ordered P1/P2/P3 rewrites."""

    def __init__(self):
        self._p1_rewriter = Pattern1Rewriter()
        self._p2_rewriter = Pattern2Rewriter()
        self._p3_rewriter = Pattern3Rewriter()
        self._moe_rewriter = MoeLocalTokenRewriter()
        self._dsa_o_proj_rewriter = DsaFullOProjRewriter()

    def __call__(self, gm):
        if not config.compilation.passes.enable_sequence_parallel:
            return gm
        graph = gm.graph
        ws = self._get_world_size(graph)
        if ws <= 1:
            return gm

        logger.debug("SP pass: world_size=%d", ws)

        # Discover and rewrite each independent pattern family in one batch.
        # P2 matching is structural, so all decoder layers are selected from
        # the original graph without waiting for local-layout metadata from a
        # preceding layer.  P3 and MoE then consume the graph state established
        # by the completed P2 batch.
        p1 = self._p1_rewriter.apply(graph)
        dsa_o_proj = self._dsa_o_proj_rewriter.apply(graph)
        p2 = self._p2_rewriter.apply(graph)
        p3 = self._p3_rewriter.apply(graph)
        moe = self._moe_rewriter.apply(graph)
        logger.debug("SP ordered rewrites: %d P1, %d P2, %d MoE matches", p1, p2, moe)
        logger.debug("SP ordered rewrites: %d P3 matches", p3)
        logger.debug("SP ordered rewrites: %d DSA full-o_proj matches", dsa_o_proj)

        if p1 == 0 and p2 == 0 and p3 == 0 and moe == 0 and dsa_o_proj == 0:
            return gm

        gm.graph.eliminate_dead_code()
        gm.graph.lint()
        gm.recompile()
        return gm

    @staticmethod
    def _get_world_size(graph):
        for n in graph.nodes:
            if (
                n.op == "call_function"
                and n.target is _ALL_REDUCE
                and len(n.args) >= 3
                and isinstance(n.args[2], (list, tuple))
            ):
                return len(n.args[2])
        return 0
