"""Extract a fusable subgraph from a pre-rewrite fx GraphModule.

Motivation: instead of asking the LLM to guess the AOT-decomposed graph
structure (wrong op overloads, wrong constants), we capture it from the real
compiled graph and give the LLM a structured, annotated section to transcribe.

Key design choice: boundary detection is TOPOLOGICAL, not whitelist-based.
A node is a boundary input when its fx node is a placeholder (model input) or
when none of its producers are in the connected subgraph. This handles any op,
including ops absent from a fixed ELEMENTWISE_OPS list.
"""

import torch.fx
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ExtractedNode:
    """One node in the captured subgraph (real graph spelling)."""

    target: str  # full overload, e.g. "aten.mul.Tensor"
    var_name: str  # local var name assigned in the graph
    arg_vars: List[str]  # var names of positional tensor args (in order)
    users_in_region: int  # how many other region nodes consume this node
    _fx_node: object = field(repr=False, compare=False)  # back-ref to fx Node


@dataclass
class SubgraphInfo:
    """A fusable subgraph captured from a real fx GraphModule."""

    seed: str  # the seed op target string
    nodes: List[ExtractedNode]  # topologically ordered region nodes
    boundary_inputs: List[str]  # var names of external tensor inputs
    output_idx: int  # index into nodes of the region output

    def to_prompt_str(self) -> str:
        """Format the subgraph as a human/LLM-readable annotated section.

        Each line: ``var = op(args)`` with BOUNDARY markers and fan-out notes.
        Designed so the LLM can transcribe this directly into a ``_pattern()``
        function without guessing op overloads or boundary structure.
        """
        lines = [f"# captured subgraph — seed: {self.seed}"]
        lines.append("#")

        for b in self.boundary_inputs:
            lines.append(f"# {b}  [BOUNDARY — external tensor input]")
        lines.append("#")

        for i, n in enumerate(self.nodes):
            out_mark = "  ← OUTPUT" if i == self.output_idx else ""
            fanout_note = f"  # _users={n.users_in_region}" if n.users_in_region >= 2 else ""
            args_str = ", ".join(n.arg_vars)
            lines.append(f"{n.var_name} = {n.target}({args_str}){out_mark}{fanout_note}")

        return "\n".join(lines)


# Namespaces considered "light" enough to expand upstream into.
# tensor_cast.*, auto_functionalized, getitem, view, permute etc. are
# heavy or layout-only and should remain as boundary inputs.
_LIGHT_NAMESPACES = {"aten", "prims"}

# aten ops that are shape-changing or IO-heavy — treat as boundaries.
# Prefix-match: op_name.startswith(p)  e.g. "view" catches view.default
_LAYOUT_OP_PREFIXES = {
    "view",
    "permute",
    "reshape",
    "transpose",
    "unsqueeze",
    "squeeze",
    "contiguous",
    "alias",
    "expand",
    "select",
    "slice",
    "split",
    "cat",
    "stack",
    "unbind",
    "index",
    "embedding",
}

# Heavy compute ops — treated as region boundaries (not fused into epilogue).
# mm/bmm/addmm are intentionally here (not in _LAYOUT_OP_PREFIXES) so that
# upstream BFS stops at matmul outputs: relu(mm_out) correctly captures the
# full mm+relu epilogue with mm as a boundary input, matching the intended
# fusion scope.  Exact op_name match: parts[1] in _HEAVY_OP_EXACT.
_HEAVY_OP_EXACT = {
    "mm",
    "bmm",
    "addmm",
    "matmul",
    "linear",
    "convolution",
    "conv1d",
    "conv2d",
    "conv3d",
    "scaled_dot_product_attention",
    "native_batch_norm",
    "layer_norm",
    "group_norm",
}


def _is_fusable(node) -> bool:
    """True when node may be part of an elementwise fusion region.

    Only aten.* and prims.* ops that are not heavy/layout ops qualify.
    tensor_cast ops, auto_functionalized, getitem, view, permute, embedding,
    mm, matmul, linear, attention etc. are treated as boundary-forming nodes.
    """
    target = str(node.target)
    parts = target.split(".")
    if not parts or parts[0] not in _LIGHT_NAMESPACES:
        return False
    op_name = parts[1] if len(parts) >= 2 else ""
    # Exact-name heavy ops (not caught by prefix matching)
    if op_name in _HEAVY_OP_EXACT:
        return False
    # Prefix-match for layout / IO-heavy ops
    return not any(op_name.startswith(p) for p in _LAYOUT_OP_PREFIXES)


def _collect_region(gm, seed: str):
    """BFS from seed: expand to connected call_function nodes via tensor edges.

    Upstream expansion (producers): unconditional — follow all call_function
    inputs up to placeholder boundaries.

    Downstream expansion (consumers): conditional — a user is added to the
    region only when ALL of its call_function inputs are already in-region or
    are placeholder nodes. This blocks fan-in nodes (e.g. a residual ``add``
    that merges outputs from multiple independent chains) from pulling unrelated
    chains into the region.
    """
    seed_nodes = [n for n in gm.graph.nodes if n.op == "call_function" and str(n.target) == seed]
    if not seed_nodes:
        return None

    in_region: set = set()
    region = []

    def _add(n):
        if id(n) in in_region or n.op != "call_function":
            return
        in_region.add(id(n))
        region.append(n)

    def _can_add_downstream(u) -> bool:
        """True when u is fusable AND all its fx.Node inputs are in-region or boundary."""
        if u.op != "call_function":
            return False
        if not _is_fusable(u):
            return False
        for a in u.args:
            if not isinstance(a, torch.fx.Node):
                continue
            # placeholder (model input) and get_attr (learned weight) are both
            # legitimate region-external boundaries; treat them as resolved.
            if a.op in ("placeholder", "get_attr"):
                continue
            if a.op == "call_function" and id(a) not in in_region:
                return False
        for a in u.kwargs.values():
            if not isinstance(a, torch.fx.Node):
                continue
            if a.op in ("placeholder", "get_attr"):
                continue
            if a.op == "call_function" and id(a) not in in_region:
                return False
        return True

    # seed
    _add(seed_nodes[0])

    changed = True
    while changed:
        changed = False
        for n in list(region):
            # expand upstream — only into fusable (light elementwise) ops
            for a in n.args:
                if (
                    isinstance(a, torch.fx.Node)
                    and a.op == "call_function"
                    and id(a) not in in_region
                    and _is_fusable(a)
                ):
                    _add(a)
                    changed = True
            # expand downstream (conditional: all inputs already resolved)
            for u in n.users:
                if id(u) not in in_region and _can_add_downstream(u):
                    _add(u)
                    changed = True

    return region


def _topo_sort(region: list) -> list:
    region_ids = {id(n) for n in region}
    ordered, placed = [], set()

    remaining = list(region)
    while remaining:
        progressed = False
        for n in list(remaining):
            deps_ready = all(
                not (isinstance(a, torch.fx.Node) and id(a) in region_ids) or id(a) in placed for a in n.args
            )
            if deps_ready:
                ordered.append(n)
                placed.add(id(n))
                remaining.remove(n)
                progressed = True
        if not progressed:
            ordered.extend(remaining)
            break
    return ordered


def _build_subgraph(seed: str, ordered: list) -> SubgraphInfo:
    region_ids = {id(n): i for i, n in enumerate(ordered)}
    boundary_inputs: List[str] = []
    boundary_of_node = {}  # id(fx_node) -> var_name

    def _boundary_var(node) -> str:
        if id(node) not in boundary_of_node:
            nm = node.name  # keep the graph's own name for clarity
            boundary_of_node[id(node)] = nm
            boundary_inputs.append(nm)
        return boundary_of_node[id(node)]

    nodes: List[ExtractedNode] = []
    for n in ordered:
        arg_vars = []
        for a in n.args:
            if isinstance(a, torch.fx.Node):
                if id(a) in region_ids:
                    arg_vars.append(ordered[region_ids[id(a)]].name)
                else:
                    arg_vars.append(_boundary_var(a))
            else:
                # Non-Node positional args: scalars (eps, exponent, dim), dtype, list.
                arg_vars.append(f"# LITERAL: {repr(a)}")

        # Non-Node kwargs (e.g. alpha=2 in aten.add.Tensor): include as annotated
        # entries so to_prompt_str() produces a complete call signature.
        # Node kwargs follow the same boundary/region logic as positional args.
        for k, v in n.kwargs.items():
            if isinstance(v, torch.fx.Node):
                if id(v) in region_ids:
                    arg_vars.append(f"# KWARG({k}): {ordered[region_ids[id(v)]].name}")
                else:
                    arg_vars.append(f"# KWARG({k}): {_boundary_var(v)}")
            else:
                arg_vars.append(f"# LITERAL(kwarg): {k}={repr(v)}")

        # count how many OTHER region nodes use this node's output
        users_in_region = sum(1 for u in n.users if id(u) in region_ids)
        nodes.append(
            ExtractedNode(
                target=str(n.target),
                var_name=n.name,
                arg_vars=arg_vars,
                users_in_region=users_in_region,
                _fx_node=n,
            )
        )

    # For single-output chains: the OUTPUT node is the last (in topological
    # order) node that has at least one consumer outside the region.  Using
    # the LAST (not the first) external-consumer node correctly handles
    # multi-step elementwise chains where every intermediate node fans out to
    # a downstream consumer that happens to be outside the region — the final
    # node in the chain is the true fusion output.
    region_ids_set = set(region_ids)
    external_output_nodes = [i for i, n in enumerate(ordered) if any(id(u) not in region_ids_set for u in n.users)]
    if len(external_output_nodes) > 1:
        # v1 scope: multi-output fusions are not supported by the Plugin
        # protocol (SKILL.md §v1 limits).  Return None so the caller knows
        # to skip rather than generate an incorrect single-output pattern.
        return None

    output_idx = external_output_nodes[0] if external_output_nodes else len(ordered) - 1

    return SubgraphInfo(
        seed=seed,
        nodes=nodes,
        boundary_inputs=boundary_inputs,
        output_idx=output_idx,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_subgraph(gm, seed_op: str) -> Optional[SubgraphInfo]:
    """Extract the connected call_function subgraph anchored at ``seed_op``.

    Args:
        gm: a ``torch.fx.GraphModule`` (pre-rewrite; use make_fx or spy on
            CompilerBackend.apply_pattern_match_passes to obtain one).
        seed_op: full op overload string, e.g. ``"aten.sigmoid.default"``.

    Returns:
        ``SubgraphInfo`` on success.  ``None`` for two distinct reasons:

        - *seed absent*: ``seed_op`` has no ``call_function`` node in ``gm``.
          Caller interpretation: "unsupported — op not in this model's graph."
        - *v1 unsupported topology*: the extracted region has multiple external
          output consumers (multi-output fusion).  v1 Plugin protocol supports
          single-output only (SKILL.md §v1 limits).
          Caller interpretation: "unsupported for now — skip generation."

        Callers should distinguish these two cases if diagnostics matter; the
        current implementation returns None for both.
    """
    region = _collect_region(gm, seed_op)
    if region is None:
        return None
    ordered = _topo_sort(region)
    return _build_subgraph(seed_op, ordered)


__all__ = ["extract_subgraph", "SubgraphInfo", "ExtractedNode"]
