"""Sequence parallel pass with ordered pattern rewrites.

P1 + P2 run first:
  P1: all_reduce -> [region_begin?] -> rms_norm / add_rms_norm
      => reduce_scatter -> [...] -> norm -> all_gather
  P2: [region_begin?(residual) +] all_reduce -> add_rms_norm2
      => reduce_scatter -> add_rms_norm2 (selective all_gather)

P3 runs after P2 because it depends on the residual left local by P2.
  P3: getitem[1] + all_reduce[/view] -> add -> [region_end -> copy*] -> norm
      => reduce_scatter + residual -> add -> ... -> norm -> all_gather
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


# ── Helpers ────────────────────────────────────────────────────────


def _shard_dim(node: Node) -> int:
    """Return 0 for 2-D tensors, else 1 (seq dim)."""
    meta = node.meta.get("val")
    if meta is not None and hasattr(meta, "dim") and meta.dim() == 2:
        return 0
    return 1


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
    if isinstance(node, torch.fx.Node) and node.op == "call_function":
        if node.target is _ALL_REDUCE:
            return node, node
        if node.target in _VIEW_OPS:
            src = node.args[0] if node.args else None
            if isinstance(src, torch.fx.Node) and src.target is _ALL_REDUCE:
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
    while (
        cur.op == "call_function"
        and cur.target is _COPY_REGION
        and id(cur) not in visited
    ):
        visited.add(id(cur))
        users = list(cur.users)
        if len(users) != 1:
            return None
        cur = users[0]
    if cur.op == "call_function" and cur.target in _SINGLE_OUTPUT_NORMS:
        return cur
    return None


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
        return comm is not None
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
    return _find_norm_after_add(tail_node) is not None


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

    for arg in user.args[:2]:
        if not isinstance(arg, Node) or arg is getitem_node:
            continue
        if arg.op != "call_function":
            continue
        if arg.target in {_ALL_REDUCE, _REDUCE_SCATTER}:
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
            fused_users = [
                u
                for u in node.users
                if u.op == "call_function" and u.target is _ADD_RMS_NORM
            ]
            if len(fused_users) == 1:
                norm = fused_users[0]
                if id(norm) in seen:
                    continue
                other = norm.args[1] if len(norm.args) >= 2 else None
                comm, comm_out = _unwrap_comm(other)
                if comm is None:
                    continue
                seen.add(id(norm))
                out.append(_P3Match(comm, comm_out, norm, norm))
                continue
            add_users = [
                u
                for u in node.users
                if u.op == "call_function" and u.target in _ADD_OPS
            ]
            if len(add_users) != 1:
                continue
            add_node = add_users[0]
            if id(add_node) in seen:
                continue
            other = None
            for a in add_node.args:
                if isinstance(a, torch.fx.Node) and a is not node:
                    other = a
                    break
            if other is None:
                continue
            comm, comm_out = _unwrap_comm(other)
            if comm is None:
                continue
            norm = _find_norm_after_add(add_node)
            if norm is None:
                continue
            seen.add(id(add_node))
            out.append(_P3Match(comm, comm_out, add_node, norm))
        return out

    def _rewrite(self, graph, m):
        rank, rg = m.comm_node.args[1], m.comm_node.args[2]
        rs_dim = _shard_dim(m.comm_node.args[0])
        with graph.inserting_after(m.comm_node):
            rs = graph.call_function(
                _REDUCE_SCATTER, (m.comm_node.args[0], rs_dim, rank, rg)
            )
        if m.comm_output is m.comm_node:
            m.add_node.replace_input_with(m.comm_node, rs)
        else:
            m.comm_output.replace_input_with(m.comm_node, rs)
        ag_dim = _shard_dim(m.norm_node)
        _insert_all_gather(graph, m.norm_node, ag_dim, rank, rg)


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
            if (
                inp.target is _REGION_BEGIN
                and isinstance(inp.args[0], Node)
                and inp.args[0].target is _ALL_REDUCE
            ):
                out.append((inp.args[0], inp, node))
            elif inp.target is _ALL_REDUCE:
                out.append((inp, None, node))
        return out

    @staticmethod
    def _rewrite(graph, comm, marker, norm):
        if not comm.args:
            return
        rank, rg = comm.args[1], comm.args[2]
        rs_dim = _shard_dim(comm.args[0])
        with graph.inserting_after(comm):
            rs = graph.call_function(_REDUCE_SCATTER, (comm.args[0], rs_dim, rank, rg))
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
        matches = self._find(graph)
        for comm, norm2 in matches:
            self._rewrite(graph, comm, norm2)
        return len(matches)

    @staticmethod
    def _find(graph):
        matches = []
        for node in graph.nodes:
            if node.op != "call_function" or node.target is not _ADD_RMS_NORM2:
                continue
            ar_inputs = [
                arg
                for arg in node.args[:2]
                if isinstance(arg, torch.fx.Node)
                and arg.op == "call_function"
                and arg.target is _ALL_REDUCE
            ]
            if len(ar_inputs) != 1:
                continue
            matches.append((ar_inputs[0], node))
        return matches

    @staticmethod
    def _rewrite(graph, comm, norm2):
        rank, rg = comm.args[1], comm.args[2]
        rs_dim = _shard_dim(comm.args[0])
        with graph.inserting_after(comm):
            rs = graph.call_function(_REDUCE_SCATTER, (comm.args[0], rs_dim, rank, rg))
        norm2.replace_input_with(comm, rs)
        ag_dim = _shard_dim(norm2)
        for u in list(norm2.users):
            if u.op != "call_function" or u.target is not operator.getitem:
                continue
            if u.args[1] == 1 and (_is_p3_tail(u) or _is_p2_chain_tail(u)):
                continue  # residual stays local for P3
            _insert_all_gather(graph, u, ag_dim, rank, rg)


# ===================================================================
# SequenceParallelPass
# ===================================================================


class SequenceParallelPass(TensorCastGraphModulePass):
    """Sequence-parallel pass with ordered P1/P2/P3 rewrites."""

    def __init__(self):
        self._p1_rewriter = Pattern1Rewriter()
        self._p2_rewriter = Pattern2Rewriter()
        self._p3_rewriter = Pattern3Rewriter()

    def __call__(self, gm):
        if not config.compilation.passes.enable_sequence_parallel:
            return gm
        graph = gm.graph
        ws = self._get_world_size(graph)
        if ws <= 1:
            return gm

        logger.debug("SP pass: world_size=%d", ws)

        # Apply P1 + P2 first so P2 can leave gi[1] local for downstream P3.
        p1 = self._p1_rewriter.apply(graph)
        p2 = self._p2_rewriter.apply(graph)
        logger.debug("SP ordered rewrites: %d P1, %d P2 matches", p1, p2)

        # Run P3 after P2 because it consumes the local residual path.
        cnt = self._p3_rewriter.apply(graph)
        logger.debug("SP ordered rewrites: %d P3 matches", cnt)

        if p1 == 0 and p2 == 0 and cnt == 0:
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
