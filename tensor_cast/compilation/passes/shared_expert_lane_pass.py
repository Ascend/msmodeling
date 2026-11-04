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

"""Bind the shared-expert branch of each MoE layer to a dedicated stream.

The pass is the whole of the ``multistream_overlap_shared_expert`` feature after
model-side marking (see ``layers/internal.py``):

1. collect lane regions from the ``_internal_mark_lane_begin/end`` pair, merging
   nested markers into the outermost region;
2. split the region into 4~5 stages by operator *class* and position;
3. locate the anchors that decide how early each stage may start;
4. pick a mode (``staged`` / ``whole``) and lower each stage with one
   ``_internal_wait_and_bind`` + one ``_internal_record``;
5. emit one ``_internal_spill`` per stage that has an overlap window, so a stage
   longer than its window serializes the routed branch instead of silently
   over-claiming parallelism.

The pass never estimates time and never removes, relocates or replaces a compute
op. Design: RFC Ascend/msmodeling#584.
"""

import dataclasses
import logging
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.fx as fx

from ... import config, ops  # noqa: F401
from ..pass_base import TensorCastGraphModulePass
from .control_flow import CONTROL_TARGETS, collect_transparent, reachable
from ..topo_sort import stable_topo_sort

logger = logging.getLogger(__name__)


def _op(namespace: str, name: str):
    """Resolve ``torch.ops.<namespace>.<name>.default`` when this build defines it."""
    packet = getattr(getattr(torch.ops, namespace, None), name, None)
    if packet is None:
        return None
    try:
        return packet.default
    except AttributeError:
        return None


def _targets(*specs: Tuple[str, str]) -> set:
    resolved = set()
    for namespace, name in specs:
        op = _op(namespace, name)
        if op is not None:
            resolved.add(op)
    return resolved


_LANE_BEGIN = _op("tensor_cast", "_internal_mark_lane_begin")
_LANE_END = _op("tensor_cast", "_internal_mark_lane_end")
_SPILL = _op("tensor_cast", "_internal_spill")
_WAIT_AND_BIND = _op("tensor_cast", "_internal_wait_and_bind")
_RECORD = _op("tensor_cast", "_internal_record")

_CONTROL_TARGETS = CONTROL_TARGETS

_GEMM_TARGETS = _targets(
    ("aten", "mm"),
    ("aten", "matmul"),
    ("aten", "addmm"),
    ("aten", "bmm"),
    ("aten", "linear"),
    ("tensor_cast", "static_quant_linear"),
    ("tensor_cast", "static_quant_linear_int4"),
    ("tensor_cast", "fp8_linear"),
    ("tensor_cast", "mxfp4_linear"),
    ("tensor_cast", "grouped_matmul"),
    ("tensor_cast", "grouped_matmul_quant"),
    ("tensor_cast", "grouped_matmul_quant_swiglu"),
)

_ACTIVATION_TARGETS = _targets(
    ("tensor_cast", "swiglu"),
    ("tensor_cast", "clamped_swiglu"),
    ("aten", "silu"),
    ("aten", "gelu"),
    ("aten", "relu"),
    ("aten", "sigmoid"),
    ("aten", "mul"),
)

_QUANT_TARGETS = _targets(
    ("tensor_cast", "dynamic_quantize_symmetric"),
    ("prims", "convert_element_type"),
    ("aten", "_to_copy"),
)

# Ops dominated by Vector work even though they are not elementwise math. Without
# this, stage 0's spill would look for a "vector successor" and skip the top-k
# block entirely, losing the overflow wait.
_VECTOR_LIKE_TARGETS = _targets(
    ("tensor_cast", "init_routing_v2"),
    ("tensor_cast", "unpermute_tokens"),
    ("tensor_cast", "moe_gating_top_k"),
    ("tensor_cast", "moe_gating_top_k_softmax"),
    ("tensor_cast", "moe_gating_top_k_hash"),
    ("aten", "topk"),
    ("aten", "sort"),
    ("aten", "gather"),
    ("aten", "where"),
    ("aten", "index.Tensor"),
    ("aten", "sum"),
    ("aten", "cat"),
)

_COMM_TARGETS = _targets(
    ("tensor_cast", "all_reduce"),
    ("tensor_cast", "all_gather"),
    ("tensor_cast", "reduce_scatter"),
    ("tensor_cast", "all_to_all"),
)

_ALL_TO_ALL_TARGETS = _targets(("tensor_cast", "all_to_all"))

_ROUTER_TARGET_TAGS = ("moe_gating_top_k", "moe_gating", "init_routing")

#: Per-stage anchor preference: the first locatable anchor wins, ``()`` means
#: "no cross-stream dependency beyond the region's own inputs".
_STAGED_DEPENDENCIES: Dict[str, Tuple[str, ...]] = {
    "prep": (),
    "gate_up": ("A_up",),
    "act": ("A_up",),
    "down": ("combine_start", "A_tail"),
    "comm": ("A_tail",),
    "mixed": ("A_tail",),
}

_KIND_UNIT = {
    "prep": "vector",
    "act": "vector",
    "gate_up": "cube",
    "down": "cube",
    "comm": "comm",
}


@dataclasses.dataclass(frozen=True)
class _Stage:
    kind: str
    nodes: Tuple[fx.Node, ...]
    unit: str


@dataclasses.dataclass(frozen=True)
class _Anchors:
    A_up: Optional[fx.Node] = None
    A_tail: Optional[fx.Node] = None
    combine_start: Optional[fx.Node] = None
    join: Optional[fx.Node] = None

    def get(self, name: str) -> Optional[fx.Node]:
        return getattr(self, name, None)


@dataclasses.dataclass(frozen=True)
class _LaneRegion:
    role: str
    begin: fx.Node
    end: fx.Node
    #: Schedulable ops inside the region (OpOverload targets only).
    nodes: Tuple[fx.Node, ...]
    #: ``nodes`` plus the zero-cost unpacking ops they own (``getitem`` of a
    #: multi-output lane op): those belong to the lane as well, otherwise they leak
    #: into the routed side and their tickets invert program order.
    owned: Tuple[fx.Node, ...] = ()
    external_inputs: Tuple[fx.Node, ...] = ()


class SharedExpertLanePass(TensorCastGraphModulePass):
    """Bind shared-expert lane regions to a dedicated stream (see module docstring)."""

    def __call__(self, gm: fx.GraphModule) -> fx.GraphModule:
        cfg = config.compilation.multistream_overlap_shared_expert
        if not cfg.enable:
            return gm

        graph = gm.graph
        nodes_in_order = list(graph.nodes)
        if self._has_dispatch_ffn_combine(nodes_in_order) and not cfg.allow_with_dfc:
            logger.warning(
                "multistream_overlap_shared_expert disabled: dispatch_ffn_combine fusion detected. "
                "Set allow_with_dfc=True to keep this feature (stage 3 then falls back to A_tail)."
            )
            return gm

        self._nodes_in_order = nodes_in_order
        self._order = {node: index for index, node in enumerate(nodes_in_order)}
        regions = self._collect_regions(nodes_in_order)
        if not regions:
            logger.debug("No lane region found; shared-expert lane pass is a no-op.")
            return gm

        self._ticket_cache: Dict[fx.Node, fx.Node] = {}
        # Built once per graph so each region looks its anchors up instead of
        # re-scanning the whole node list (keeps the pass linear in graph size).
        self._prefix_tail: List[Optional[fx.Node]] = []
        self._prefix_router: List[Optional[fx.Node]] = []
        last_lane: Optional[fx.Node] = None
        last_router: Optional[fx.Node] = None
        for node in nodes_in_order:
            # Two independent checks: router ops are lane nodes as well.
            if self._is_lane_node(node):
                last_lane = node
            if self._is_router_candidate(node):
                last_router = node
            self._prefix_tail.append(last_lane)
            self._prefix_router.append(last_router)
        self._reach_cache: Dict[Tuple[fx.Node, bool], set] = {}
        self._has_foreign_anchors = any(
            node.op == "call_function" and node.target in {_WAIT_AND_BIND, _RECORD} for node in nodes_in_order
        )
        applied = 0
        boundary = 0
        for region in regions:
            stages = self._split_stages(region)
            anchors = self._locate_anchors(region, graph, boundary)
            # The next region may live in a later layer: never look before this
            # region when resolving its anchors.
            boundary = self._order[region.end]
            mode = self._select_mode(anchors, cfg.force_mode)
            if mode is None:
                continue
            if mode == "whole":
                # Conservative fallback: one concurrent block, no per-stage anchors.
                stages = (self._stage("mixed", region.owned),)
            if not self._lower_region(graph, region, stages, anchors, mode, cfg.stream_id):
                continue
            applied += 1
            logger.info(
                "Shared-expert lane: stream=%d mode=%s stages=%s anchors=%s",
                cfg.stream_id,
                mode,
                [stage.kind for stage in stages],
                {stage.kind: self._anchor_name_for(stage.kind, anchors, mode) for stage in stages},
            )
        if not applied:
            return gm

        stable_topo_sort(gm)
        graph.lint()
        gm.recompile()
        return gm

    # ------------------------------------------------------------------ detect
    @staticmethod
    def _has_dispatch_ffn_combine(nodes: Sequence[fx.Node]) -> bool:
        return any("dispatch_ffn_combine" in str(node.target) for node in nodes)

    # ------------------------------------------------------------------ region
    def _collect_regions(self, nodes_in_order: Sequence[fx.Node]) -> List[_LaneRegion]:
        regions: List[_LaneRegion] = []
        stack: List[Tuple[str, int, fx.Node]] = []
        for idx, node in enumerate(nodes_in_order):
            if node.target == _LANE_BEGIN:
                stack.append((str(node.args[1]), idx, node))
                continue
            if node.target != _LANE_END:
                continue
            role = str(node.args[1])
            if not stack or stack[-1][0] != role:
                logger.warning("Unpaired lane marker %s ignored.", node.name)
                continue
            _, begin_idx, begin_node = stack.pop()
            if stack and stack[-1][0] == role:
                # Inner marker of a nested same-role region: merged into the outer one.
                continue
            span = nodes_in_order[begin_idx + 1 : idx]
            lane_ops = tuple(n for n in span if self._is_lane_node(n))
            if not lane_ops:
                logger.warning("Lane region %s..%s holds no schedulable op; ignored.", begin_node.name, node.name)
                continue
            owned = self._owned_nodes(lane_ops, span)
            regions.append(
                _LaneRegion(
                    role=role,
                    begin=begin_node,
                    end=node,
                    nodes=lane_ops,
                    owned=owned,
                    external_inputs=self._external_inputs(owned),
                )
            )
        if stack:
            logger.warning("%d unclosed lane marker(s) ignored.", len(stack))
        return regions

    @staticmethod
    def _owned_nodes(lane_ops: Sequence[fx.Node], span: Sequence[fx.Node]) -> Tuple[fx.Node, ...]:
        """Lane ops plus the zero-cost unpack nodes they own, in program order.

        ``getitem`` (and friends) are not ``OpOverload`` targets, but when they unpack
        a lane op's output they run inside the lane: keeping them in the owned set stops
        the region from publishing tickets for its own intermediates.
        """
        owned = set(lane_ops)
        span_set = set(span)
        changed = True
        while changed:
            changed = False
            for node in span:
                if node in owned or node not in span_set:
                    continue
                if node.op != "call_function" or isinstance(node.target, torch._ops.OpOverload):
                    continue
                if any(parent in owned for parent in node.all_input_nodes):
                    owned.add(node)
                    changed = True
        return tuple(node for node in span if node in owned)

    def _external_inputs(self, owned: Sequence[fx.Node]) -> Tuple[fx.Node, ...]:
        """Schedulable producers outside the region, resolved through control ops.

        The region's first op consumes the marker itself, so the marker (and any other
        control op) must be looked through: publishing a ticket for it would create a
        cycle with the entry wait.
        """
        inner = set(owned)
        return tuple(
            producer
            for producer in collect_transparent(owned, forward=False, program_order=self._order)
            if producer not in inner and producer.op in ("call_function", "call_module")
        )

    def _is_lane_node(self, node: fx.Node) -> bool:
        """Whether ``node`` is a schedulable op inside a lane region.

        Three conditions, each backed by a real graph:

        * ``call_function`` with an ``OpOverload`` target;
        * not a control op (markers / anchors / spill / region helpers);
        * produces a tensor **or a tuple/list of tensors** -- ``dynamic_quantize_symmetric``
          returns ``(quantized, scale)`` and ``split_with_sizes`` returns a tuple, so
          filtering on "single tensor value" would silently drop the entry quant
          (stage 0) and the split from the region.
        """
        if node.op != "call_function" or not isinstance(node.target, torch._ops.OpOverload):
            return False
        if node.target in _CONTROL_TARGETS:
            return False
        value = node.meta.get("val")
        if isinstance(value, torch.Tensor):
            return True
        return isinstance(value, (tuple, list)) and any(isinstance(item, torch.Tensor) for item in value)

    # ------------------------------------------------------------------ stages
    def _split_stages(self, region: _LaneRegion) -> Tuple[_Stage, ...]:
        """Split by operator class and position; unpack ops follow their producer."""
        nodes = list(region.owned)
        comm_idx = [i for i, node in enumerate(nodes) if node.target in _COMM_TARGETS]
        head = nodes[: comm_idx[0]] if comm_idx else nodes
        tail = nodes[comm_idx[0] :] if comm_idx else []

        gemm = [i for i, node in enumerate(head) if node.target in _GEMM_TARGETS]
        if len(gemm) < 2:
            return (self._stage("mixed", head),)

        act = [i for i, node in enumerate(head) if node.target in _ACTIVATION_TARGETS]
        last_gemm = gemm[-1]
        buckets: List[Tuple[str, int, int]] = []  # (kind, start, end) over ``head``
        if gemm[0] > 0:
            buckets.append(("prep", 0, gemm[0]))
        if not act or act[0] >= last_gemm:
            buckets.append(("gate_up", gemm[0], last_gemm))
            buckets.append(("down", last_gemm, len(head)))
        else:
            buckets.append(("gate_up", gemm[0], act[0]))
            buckets.append(("act", act[0], last_gemm))
            buckets.append(("down", last_gemm, len(head)))

        stage_nodes: Dict[str, List[fx.Node]] = {kind: [] for kind, _, _ in buckets}
        for index, node in enumerate(head):
            kind = self._bucket_of(index, buckets, stage_nodes)
            if kind is not None:
                stage_nodes[kind].append(node)
        stages = [self._stage(kind, stage_nodes[kind]) for kind, _, _ in buckets if stage_nodes[kind]]
        if tail:
            stages.append(self._stage("comm", tail))
        return tuple(stages)

    @staticmethod
    def _bucket_of(
        index: int, buckets: Sequence[Tuple[str, int, int]], assigned: Dict[str, List[fx.Node]]
    ) -> Optional[str]:
        """Bucket an index, attaching unpack ops to the stage that produced them."""
        for kind, start, end in buckets:
            if start <= index < end:
                return kind
        # Outside every bucket (cannot happen for the head slice) -> last non-empty.
        for kind, _, _ in reversed(buckets):
            return kind
        return None

    @staticmethod
    def _stage(kind: str, nodes: Sequence[fx.Node]) -> _Stage:
        return _Stage(kind=kind, nodes=tuple(nodes), unit=_KIND_UNIT.get(kind, "mixed"))

    # ----------------------------------------------------------------- anchors
    def _locate_anchors(self, region: _LaneRegion, graph: fx.Graph, boundary: int = 0) -> _Anchors:
        begin_idx = self._order[region.begin]
        a_tail = self._prefix_tail[begin_idx - 1] if begin_idx > 0 else None
        router = self._prefix_router[begin_idx - 1] if begin_idx > 0 else None
        if router is not None and self._order[router] < boundary:
            # The last router before this region belongs to an earlier layer.
            router = None
        a_up = self._last_gemm_before(router, boundary) if router is not None else None
        join = self._join_point(region)
        sliced = self._routed_slice(region, router, join, boundary) if router and join else []
        self._current_routed_slice = sliced
        self._current_anchors = _Anchors(A_up=a_up, A_tail=a_tail, combine_start=None, join=join)
        combine_start = self._combine_start(sliced)
        return _Anchors(A_up=a_up, A_tail=a_tail, combine_start=combine_start, join=join)

    @staticmethod
    def _is_router_candidate(node: fx.Node) -> bool:
        if node.op != "call_function":
            return False
        name = str(node.target)
        return any(tag in name for tag in _ROUTER_TARGET_TAGS)

    def _last_gemm_before(self, router: fx.Node, boundary: int = 0) -> Optional[fx.Node]:
        """Reverse-DFS from the router and keep the last GEMM in program order.

        The walk is bounded twice so one layer does not rescan every earlier layer:

        * nodes at or before ``boundary`` (the previous lane region) belong to an
          earlier layer and are skipped;
        * a branch whose node already precedes the best candidate cannot contain a
          later GEMM (FX guarantees inputs precede the node), so it is pruned.
        """
        best: Optional[fx.Node] = None
        seen: set = set()
        stack = [router]
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            order = self._order.get(node, -1)
            if order < boundary:
                continue
            if node.target in _GEMM_TARGETS:
                if best is None or order > self._order.get(best, -1):
                    best = node
                continue  # a GEMM's own inputs cannot hold a later GEMM
            if best is not None and order <= self._order.get(best, -1):
                continue
            stack.extend(
                parent for parent in node.all_input_nodes if parent.op in ("call_function", "call_module", "get_attr")
            )
        return best

    @staticmethod
    def _join_point(region: _LaneRegion) -> Optional[fx.Node]:
        inner = set(region.owned)
        for user in region.end.users:
            if user not in inner and user.op == "call_function":
                return user
        return None

    @staticmethod
    def _combine_start(sliced: Sequence[fx.Node]) -> Optional[fx.Node]:
        all_to_all = [node for node in sliced if node.target in _ALL_TO_ALL_TARGETS]
        if not all_to_all:
            return None
        producer = None
        for parent in all_to_all[-1].all_input_nodes:
            if parent.target in _COMM_TARGETS:
                continue
            producer = parent
        return producer

    def _routed_slice(
        self,
        region: _LaneRegion,
        router: fx.Node,
        join: fx.Node,
        boundary: int = 0,
    ) -> List[fx.Node]:
        """Routed-side ops between the router output and the join point.

        Both walks stop at the other anchor and ignore everything before ``boundary``,
        so the query costs one layer instead of a full-graph traversal per MoE layer.
        """
        forward = self._reachable_cached(router, forward=True, stop=join, boundary=boundary)
        backward = self._reachable_cached(join, forward=False, stop=router, boundary=boundary)
        inner = set(region.owned)
        return [
            node
            for node in self._nodes_in_order[self._order[router] :]
            if node in forward and node in backward and node not in inner and self._is_lane_node(node)
        ]

    def _reachable_cached(
        self,
        node: fx.Node,
        forward: bool,
        stop: Optional[fx.Node] = None,
        boundary: int = 0,
    ) -> set:
        """Reachability per graph, bounded by ``boundary`` and optionally by ``stop``."""
        key = (node, forward, boundary, stop)
        cached = self._reach_cache.get(key)
        if cached is None:
            cached = reachable(
                node,
                forward=forward,
                stop=(stop,) if stop is not None else (),
                accept=lambda current: self._order.get(current, 0) >= boundary,
            )
            self._reach_cache[key] = cached
        return cached

    # ------------------------------------------------------------------- modes
    @staticmethod
    def _select_mode(anchors: _Anchors, force_mode: Optional[str]) -> Optional[str]:
        if force_mode is not None:
            if force_mode in ("staged", "whole"):
                logger.info("shared-expert lane: force_mode=%s", force_mode)
                return force_mode
            logger.warning("Unknown force_mode=%r ignored; falling back to auto.", force_mode)
        if anchors.A_up is None:
            logger.info("shared-expert lane: router-output anchor missing; using whole-region overlap.")
            return "whole"
        return "staged"

    @staticmethod
    def _anchor_name_for(kind: str, anchors: _Anchors, mode: str) -> str:
        if mode != "staged":
            return "none"
        for name in _STAGED_DEPENDENCIES.get(kind, ()):
            if anchors.get(name) is not None:
                return name
        return "none"

    # ---------------------------------------------------------------- lowering
    def _lower_region(
        self,
        graph: fx.Graph,
        region: _LaneRegion,
        stages: Sequence[_Stage],
        anchors: _Anchors,
        mode: str,
        stream_id: int,
    ) -> bool:
        if self._already_lowered(region):
            logger.info("shared-expert lane: region already lowered; skipped.")
            return False

        upstream: List[fx.Node] = []
        for producer in region.external_inputs:
            ticket = self._ticket_for(graph, producer)
            if ticket is not None:
                upstream.append(ticket)

        stage_tickets: List[Optional[fx.Node]] = []
        previous_ticket: Optional[fx.Node] = None
        for index, stage in enumerate(stages):
            deps: List[fx.Node] = []
            if index == 0:
                deps.extend(upstream)
            elif previous_ticket is not None:
                deps.append(previous_ticket)
            dep_name = self._anchor_name_for(stage.kind, anchors, mode)
            if dep_name != "none":
                anchor_ticket = self._ticket_for(graph, anchors.get(dep_name))
                if anchor_ticket is not None:
                    deps.append(anchor_ticket)

            if index == 0:
                anchor_node = region.begin
                entry_arg = region.begin.args[0]
            else:
                anchor_node = stage.nodes[0]
                entry_arg = self._first_tensor_input(anchor_node)
                if entry_arg is None:
                    entry_arg = region.begin.args[0]

            deps = self._ordered_deps(deps, anchor_node)
            with graph.inserting_before(anchor_node):
                entry = graph.call_function(_WAIT_AND_BIND, args=(entry_arg, stream_id, deps))
            self._rewire_first_tensor_arg(anchor_node, entry)

            tail_node = stage.nodes[-1]
            is_last = index == len(stages) - 1
            with graph.inserting_before(region.end) if is_last else graph.inserting_after(tail_node):
                previous_ticket = graph.call_function(
                    _RECORD, args=(tail_node, stream_id, self._stage_tag(region, stage))
                )
            stage_tickets.append(previous_ticket)

        if previous_ticket is None:
            return False

        spill_roles: Dict[str, str] = {}
        spill_tokens = self._insert_spills(graph, region, stages, anchors, stage_tickets, spill_roles)
        self._lower_join(graph, region, previous_ticket, spill_tokens)
        if spill_roles:
            logger.info("shared-expert lane spills: %s", spill_roles)
        return True

    def _window_tag(
        self, region: _LaneRegion, stage: _Stage, graph: fx.Graph, target: fx.Node
    ) -> Tuple[str, Optional[fx.Node]]:
        """Publish a tagged token on the routed node that ends this stage's window.

        The window is the routed-side segment the stage is supposed to hide behind
        (router GEMM / dispatch / grouped matmul / combine), so ``overrun = stage_ready -
        window_ready`` answers "how much of this stage did not fit". The blocking wait is
        a different quantity (see ``Runtime._resolve_spill_durations``).
        """
        tag = f"{region.role}:{stage.kind}:window:{region.begin.name}"
        window_node = self._window_node(stage, region)
        tensor_node = self._tensor_value_node(window_node)
        if tensor_node is None:
            return "", None
        with graph.inserting_after(tensor_node):
            ticket = graph.call_function(_RECORD, args=(tensor_node, 0, tag))
        return tag, ticket

    def _window_node(self, stage: _Stage, region: _LaneRegion) -> Optional[fx.Node]:
        """Routed node that closes this stage's overlap window."""
        anchors = getattr(self, "_current_anchors", None) or _Anchors()
        sliced = getattr(self, "_current_routed_slice", [])
        all_to_all = [node for node in sliced if node.target in _ALL_TO_ALL_TARGETS]
        cubes = [node for node in sliced if node.target in _GEMM_TARGETS]
        if stage.kind == "prep":
            return anchors.A_up
        if stage.kind == "gate_up":
            return all_to_all[0] if all_to_all else anchors.A_up
        if stage.kind == "act":
            if all_to_all:
                first_combine = all_to_all[-1]
                before = [node for node in cubes if self._order.get(node, 0) < self._order.get(first_combine, 0)]
                return before[-1] if before else first_combine
            return cubes[-1] if cubes else anchors.A_up
        if stage.kind == "down":
            return all_to_all[-1] if all_to_all else anchors.A_tail
        return anchors.A_tail

    def _tensor_value_node(self, node: Optional[fx.Node]) -> Optional[fx.Node]:
        """A single-tensor node standing for ``node``'s completion time.

        Multi-output lane ops (``split_with_sizes``, ``dynamic_quantize_symmetric``)
        are unpacked by ``getitem`` users; the record must carry a tensor, so use the
        first unpacking user when the node itself values a tuple/list.
        """
        if node is None:
            return None
        value = node.meta.get("val")
        if isinstance(value, torch.Tensor):
            return node
        for user in node.users:
            if isinstance(user.meta.get("val"), torch.Tensor):
                return user
        return None

    @staticmethod
    def _stage_tag(region: _LaneRegion, stage: _Stage) -> str:
        """Token tag shared by a stage's record and the spill that waits on it."""
        return f"{region.role}:{stage.kind}:{region.begin.name}"

    @staticmethod
    def _first_tensor_input(node: fx.Node) -> Optional[fx.Node]:
        for parent in node.all_input_nodes:
            if isinstance(parent.meta.get("val"), torch.Tensor):
                return parent
        return None

    @staticmethod
    def _rewire_first_tensor_arg(node: fx.Node, replacement: fx.Node) -> None:
        """Point the first tensor input of ``node`` at ``replacement``.

        Keeps the inserted anchor alive through DCE (same technique as the global
        multistream pass) while leaving every other edge untouched.
        """
        args = list(node.args)
        for index, arg in enumerate(args):
            if isinstance(arg, fx.Node) and isinstance(arg.meta.get("val"), torch.Tensor):
                args[index] = replacement
                node.args = tuple(args)
                return

    def _ticket_for(self, graph: fx.Graph, node: Optional[fx.Node]) -> Optional[fx.Node]:
        """Publish (or reuse) a sync ticket for ``node``; never retags another stream's event.

        ``Runtime._handle_record`` rewrites the *previous* event's ``stream_id`` to the
        record's argument, so inserting ``record(node, 0)`` for a node that already runs
        on another stream would silently move that event. Hence: reuse an existing
        record, else only publish when no foreign anchors exist, else fall back to the
        nearest producer ticket.
        """
        if node is None:
            return None
        cached = self._ticket_cache.get(node)
        if cached is not None:
            return cached
        for user in node.users:
            if user.target == _RECORD:
                self._ticket_cache[node] = user
                return user
        if not self._has_foreign_anchors:
            with graph.inserting_after(node):
                ticket = graph.call_function(_RECORD, args=(node, 0))
            self._ticket_cache[node] = ticket
            return ticket
        for parent in node.all_input_nodes:
            for user in parent.users:
                if user.target == _RECORD:
                    self._ticket_cache[node] = user
                    return user
        logger.debug("shared-expert lane: no ticket available for %s; dependency dropped.", node.name)
        return None

    # ------------------------------------------------------------------- spills
    def _insert_spills(
        self,
        graph: fx.Graph,
        region: _LaneRegion,
        stages: Sequence[_Stage],
        anchors: _Anchors,
        stage_tickets: Sequence[Optional[fx.Node]],
        spill_roles: Optional[Dict[str, str]] = None,
    ) -> List[fx.Node]:
        """Emit one overflow wait per stage that owns an overlap window.

        The wait sits immediately before the routed-side node that competes for the same
        unit and depends on the stage's ticket, so the runtime derives its length as the
        measured residual. The routed node then consumes the spill output, which keeps
        the node alive without publishing a record next to a foreign stream event.
        """
        spills: List[fx.Node] = []
        window_tickets: List[fx.Node] = []
        for stage, ticket in zip(stages, stage_tickets):
            if ticket is None or stage.kind in ("comm", "mixed"):
                continue
            target = self._spill_target(stage, anchors)
            if target is None:
                continue
            spill_input = self._first_tensor_input(target)
            if spill_input is None:
                continue
            window_tag, window_ticket = self._window_tag(region, stage, graph, target)
            if window_ticket is not None:
                window_tickets.append(window_ticket)
            with graph.inserting_before(target):
                # Forward reference: the stage token is produced later in program order,
                # so it travels as a tag instead of an FX arg (see _internal_spill).
                spill = graph.call_function(
                    _SPILL,
                    args=(spill_input, f"{region.role}:{stage.kind}", self._stage_tag(region, stage), window_tag),
                )
                # Close the binding domain immediately: the routed node after the spill
                # inherits the delay through the stream, not through an extra dependency.
                spill_token = graph.call_function(_RECORD, args=(spill, 0))
            self._rewire_first_tensor_arg(target, spill)
            if spill_roles is not None:
                spill_roles[stage.kind] = target.name
            spills.append(spill_token)
        self._window_tickets = window_tickets
        unclassified = sorted(
            {
                str(node.target)
                for node in getattr(self, "_current_routed_slice", [])
                if self._unit_of_node(node) == "mixed"
            }
        )
        if unclassified:
            logger.debug(
                "shared-expert lane: routed ops without a known unit (never used as spill landing "
                "points, so an unknown Cube op would under-estimate the wait): %s",
                unclassified,
            )
        return spills

    def _spill_target(self, stage: _Stage, anchors: _Anchors) -> Optional[fx.Node]:
        """Routed-side node that must wait for a stage competing for the same unit."""
        sliced = getattr(self, "_current_routed_slice", [])
        if stage.kind in ("down", "mixed") or anchors.join is None:
            return anchors.join
        if stage.kind == "gate_up":
            candidates = self._after_all_to_all(sliced, last=False)
        elif stage.kind == "act":
            candidates = self._after_all_to_all(sliced, last=True)
        else:
            candidates = sliced
        for node in candidates:
            if self._unit_of_node(node) == stage.unit:
                return node
        for node in sliced:
            if self._unit_of_node(node) == stage.unit:
                return node
        logger.debug(
            "shared-expert lane: no %s successor found for stage %s; spill falls back to the join point.",
            stage.unit,
            stage.kind,
        )
        return anchors.join

    @staticmethod
    def _after_all_to_all(candidates: Sequence[fx.Node], last: bool) -> Sequence[fx.Node]:
        indices = [i for i, node in enumerate(candidates) if node.target in _ALL_TO_ALL_TARGETS]
        if not indices:
            return candidates
        pivot = indices[-1] if last else indices[0]
        return candidates[pivot + 1 :]

    @staticmethod
    def _unit_of_node(node: fx.Node) -> str:
        if node.target in _GEMM_TARGETS:
            return "cube"
        if node.target in _ACTIVATION_TARGETS or node.target in _QUANT_TARGETS:
            return "vector"
        if node.target in _VECTOR_LIKE_TARGETS:
            return "vector"
        if node.target in _COMM_TARGETS:
            return "comm"
        return "mixed"

    # --------------------------------------------------------------------- join
    def _ordered_deps(self, deps: Sequence[fx.Node], insertion_point: fx.Node) -> List[fx.Node]:
        """Drop dependencies defined after ``insertion_point`` (FX needs program-order args).

        Anchors and stage tickets are normally defined earlier in the graph, but a
        graph with replayed layers or an unusual program order can invert that. Dropping
        such a dependency only makes the wait later (conservative); keeping it would make
        the graph invalid.
        """
        limit = self._order.get(insertion_point, len(self._order))
        ordered: List[fx.Node] = []
        for dep in deps:
            if dep is None or dep in ordered:
                continue
            index = self._order.get(dep)
            if index is not None and index > limit:
                logger.debug(
                    "shared-expert lane: dropping forward-referenced dependency %s before %s.",
                    dep.name,
                    insertion_point.name,
                )
                continue
            ordered.append(dep)
        return ordered

    def _lower_join(
        self,
        graph: fx.Graph,
        region: _LaneRegion,
        last_ticket: fx.Node,
        spill_tokens: Sequence[fx.Node] = (),
    ) -> None:
        """Make every external consumer of the region wait for the last stage.

        The spill tokens are appended so those control events keep a real consumer
        (they must survive DCE for the trace to explain the overflow).
        """
        # Window tokens only feed the overrun metric; the join wait keeps them alive so
        # DCE cannot silently turn the metric into zero.
        deps = [last_ticket, *spill_tokens, *getattr(self, "_window_tickets", [])]
        inner = set(region.owned)
        for consumer in self._external_consumers(region):
            region_args = [
                arg for arg in consumer.args if isinstance(arg, fx.Node) and self._is_region_output(arg, inner)
            ]
            if not region_args:
                continue
            # ``wait_and_bind`` carries the value of its own argument, so each distinct
            # region output needs its own wait: reusing one wait for several outputs
            # would substitute the first output's tensor for all of them.
            replacements: Dict[fx.Node, fx.Node] = {}
            with graph.inserting_before(consumer):
                ordered_deps = self._ordered_deps(deps, consumer)
                for arg in region_args:
                    if arg in replacements:
                        continue
                    replacements[arg] = graph.call_function(_WAIT_AND_BIND, args=(arg, 0, ordered_deps))
            consumer.args = tuple(
                replacements.get(arg, arg) if isinstance(arg, fx.Node) else arg for arg in consumer.args
            )

    def _external_consumers(self, region: _LaneRegion) -> List[fx.Node]:
        """Consumers of the region resolved through our own control nodes."""
        owned = set(region.owned)
        return [
            consumer
            for consumer in collect_transparent(region.owned, forward=True, program_order=self._order)
            if consumer not in owned
        ]

    @staticmethod
    def _is_region_output(node: fx.Node, inner: set) -> bool:
        """Whether ``node`` carries an activation produced inside the region."""
        value = node.meta.get("val")
        if not isinstance(value, torch.Tensor) or value.dtype == torch.int64:
            return False  # sync tickets are scalars, not activations
        seen: set = set()
        stack = [node]
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            if current in inner:
                return True
            if current.target in _CONTROL_TARGETS:
                stack.extend(current.all_input_nodes)
        return False

    @staticmethod
    def _already_lowered(region: _LaneRegion) -> bool:
        """Whether this region already carries lane anchors.

        ``whole`` mode (and a single ``mixed`` stage) has no second stage, so its only
        wait sits in front of the begin marker and rewires the marker's own input; the
        owned compute ops are not direct consumers of a wait. Both places must be
        checked, otherwise applying the pass twice would insert the anchors again.
        """
        if any(parent.target == _WAIT_AND_BIND for parent in region.begin.all_input_nodes):
            return True
        return any(parent.target == _WAIT_AND_BIND for node in region.owned for parent in node.all_input_nodes)
