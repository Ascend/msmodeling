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

"""Control-op-aware dataflow traversal shared by the graph scheduling passes.

Both passes need to look *through* the internal control ops they insert
(``_internal_mark_*``, ``_internal_wait_and_bind``, ``_internal_record``,
``_internal_spill``): those nodes have no compute cost and no stream of their own,
so a dependency or a consumer must be resolved against the real schedulable op
behind them. Keeping the traversal in one place avoids two subtly different
implementations of the same rule.
"""

from typing import Callable, Iterable, Iterator, Optional, Set

import torch
import torch.fx as fx

#: Internal ops that only exist to steer lowering and carry no dataflow of their own.
CONTROL_TARGETS = frozenset(
    op
    for op in (
        torch.ops.tensor_cast._internal_wait_and_bind.default,
        torch.ops.tensor_cast._internal_record.default,
        torch.ops.tensor_cast._internal_order_barrier.default,
        torch.ops.tensor_cast._internal_mark_region_begin.default,
        torch.ops.tensor_cast._internal_mark_region_end.default,
        torch.ops.tensor_cast._internal_copy_region.default,
        torch.ops.tensor_cast._internal_copy_region_v2.default,
        torch.ops.tensor_cast._internal_mark_lane_begin.default,
        torch.ops.tensor_cast._internal_mark_lane_end.default,
        torch.ops.tensor_cast._internal_spill.default,
    )
)


def is_control_op(node: fx.Node) -> bool:
    """Whether ``node`` is one of the internal lowering ops."""
    return node.target in CONTROL_TARGETS


def iter_transparent(node: fx.Node, *, forward: bool) -> Iterator[fx.Node]:
    """Yield the non-control neighbours of ``node`` in one direction.

    Control ops are walked through instead of returned, so callers see the real
    producers (``forward=False``) or consumers (``forward=True``) behind them.
    """
    pending: list[fx.Node] = [node]
    seen: Set[fx.Node] = set()
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        neighbours = current.users.keys() if forward else current.all_input_nodes
        for neighbour in neighbours:
            if is_control_op(neighbour):
                pending.append(neighbour)
                continue
            yield neighbour


def collect_transparent(
    nodes: Iterable[fx.Node],
    *,
    forward: bool,
    program_order: dict,
) -> list[fx.Node]:
    """Deduplicated, program-order-sorted transparent neighbours of ``nodes``.

    Sorting by ``program_order`` keeps scheduling deterministic: ``fx.Node.users``
    is a dict whose iteration order depends on insertion history, which would
    otherwise leak into rank/ticket ordering.
    """
    found: dict[fx.Node, None] = {}
    for node in nodes:
        for neighbour in iter_transparent(node, forward=forward):
            found.setdefault(neighbour, None)
    return sorted(found, key=lambda node: program_order.get(node, len(program_order)))


def reachable(
    start: fx.Node,
    *,
    forward: bool,
    stop: Iterable[fx.Node] = (),
    accept: Optional[Callable[[fx.Node], bool]] = None,
) -> Set[fx.Node]:
    """Nodes reachable from ``start`` through dataflow (control ops excluded).

    ``stop`` bounds the walk: those nodes are recorded but not expanded, so a query
    such as "everything between the router and the join" does not spill over into the
    next layer. ``accept`` additionally prunes nodes that carry no useful dataflow
    (e.g. everything before the current layer's window).
    """
    stop_set = set(stop)
    seen: Set[fx.Node] = set()
    pending: list[fx.Node] = [start]
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        if accept is not None and not accept(current):
            continue
        seen.add(current)
        if current in stop_set:
            continue
        neighbours = current.users.keys() if forward else current.all_input_nodes
        pending.extend(
            neighbour for neighbour in neighbours if neighbour.op in ("call_function", "call_module", "get_attr")
        )
    return seen
