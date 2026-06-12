import heapq
import operator

import torch
import torch.fx


def stable_topo_sort(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """
    Similar to torch.fx.passes.tools_common.legalize_graph but preserve
    the original order in the graph as much as possible.
    """

    # These operators are used for making runtime assertions before any
    # data-dependent operators occur. We want to prioritize sorting these to
    # ensure that these assertions appear before any data-dependent operations
    # in the graph.
    PRIORITIZED_OPS = [
        operator.add,
        operator.mul,
        operator.sub,
        operator.floordiv,
        operator.truediv,
        operator.mod,
        operator.le,
        operator.lt,
        operator.ge,
        operator.gt,
        operator.eq,
        operator.ne,
        torch.ops.aten.sym_constrain_range.default,
        torch.ops.aten.sym_constrain_range_for_size.default,
        torch.ops.aten._assert_async.msg,
        torch.ops.aten.scalar_tensor.default,
        torch.ops.aten._assert_scalar.default,
    ]
    # Use a set for efficient O(1) lookups
    PRIORITIZED_OPS_SET = set(PRIORITIZED_OPS)

    # Store original graph order to break ties in the topological sort
    original_order = {node: i for i, node in enumerate(gm.graph.nodes)}

    indeg = dict.fromkeys(gm.graph.nodes, 0)
    new_graph = torch.fx.Graph()
    # Track how many unfulfilled dependencies each node has
    for node in gm.graph.nodes:
        for user in node.users:
            indeg[user] += 1

    # Use a priority queue (min-heap) instead of a deque
    # Items are (op_priority, original_index, node)
    queue: list[tuple[int, int, torch.fx.Node]] = []

    # Add all nodes with no dependencies to the queue
    for node in gm.graph.nodes:
        if indeg[node] == 0:
            op_priority = 0 if node.op == "call_function" and node.target in PRIORITIZED_OPS_SET else 1
            original_index = original_order[node]
            heapq.heappush(queue, (op_priority, original_index, node))

    env: dict[torch.fx.Node, torch.fx.Node] = {}
    # Pop nodes from the queue, and add nodes that have had all their
    # dependencies fulfilled
    while len(queue) > 0:
        # Pop the node with the highest priority (lowest number)
        op_prio, orig_idx, cur = heapq.heappop(queue)

        # Avoid processing a node more than once if the graph is weird
        if cur in env:
            continue

        env[cur] = new_graph.node_copy(cur, lambda x: env[x])
        for user in cur.users:
            indeg[user] -= 1
            if indeg[user] == 0:
                # Add newly ready nodes to the heap with their priority
                op_priority = 0 if user.op == "call_function" and user.target in PRIORITIZED_OPS_SET else 1
                original_index = original_order[user]
                heapq.heappush(queue, (op_priority, original_index, user))

    # If the new graph's size is not as large as the old one, then there must be
    # a cycle (i.e. some node's dependencies were not satisfied.)
    if len(new_graph.nodes) < len(gm.graph.nodes):
        raise RuntimeError(f"Input graph has cycles, unable to add {[node for node in indeg if indeg[node] != 0]}")

    new_graph._codegen = gm.graph._codegen
    gm.graph = new_graph
    return gm
