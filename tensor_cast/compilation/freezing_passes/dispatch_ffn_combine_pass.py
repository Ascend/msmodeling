import logging
from collections import deque

import torch
import torch.fx as fx

from ... import ops  # noqa: F401
from ..pass_base import TensorCastGraphModulePass
from ..topo_sort import stable_topo_sort

logger = logging.getLogger(__name__)


class DispatchFFNCombinePass(TensorCastGraphModulePass):
    _GROUPED_MATMUL_OPS = {
        torch.ops.tensor_cast.grouped_matmul.default,
        torch.ops.tensor_cast.grouped_matmul_quant.default,
        torch.ops.tensor_cast.grouped_matmul_quant_int4.default,
        torch.ops.tensor_cast.grouped_matmul_fp8.default,
        torch.ops.tensor_cast.grouped_matmul_mxfp4.default,
    }

    _GROUPED_MATMUL_SWIGLU_OPS = {
        torch.ops.tensor_cast.grouped_matmul_swiglu.default,
        torch.ops.tensor_cast.grouped_matmul_quant_swiglu.default,
        torch.ops.tensor_cast.grouped_matmul_quant_int4_swiglu.default,
        torch.ops.tensor_cast.grouped_matmul_fp8_swiglu.default,
        torch.ops.tensor_cast.grouped_matmul_mxfp4_swiglu.default,
    }

    _LINEAR_FFN_OPS = {
        torch.ops.tensor_cast.static_quant_linear.default,
        torch.ops.tensor_cast.static_quant_linear_int4.default,
        torch.ops.tensor_cast.fp8_linear.default,
        torch.ops.tensor_cast.mxfp4_linear.default,
    }

    _SWIGLU_OPS = {
        torch.ops.tensor_cast.swiglu.default,
        torch.ops.tensor_cast.grouped_matmul_swiglu.default,
        torch.ops.tensor_cast.grouped_matmul_quant_swiglu.default,
        torch.ops.tensor_cast.grouped_matmul_quant_int4_swiglu.default,
        torch.ops.tensor_cast.grouped_matmul_fp8_swiglu.default,
        torch.ops.tensor_cast.grouped_matmul_mxfp4_swiglu.default,
    }

    # Map from grouped_matmul_*_swiglu target → DFC fused op target
    _DFC_OP_MAP_GMM = {
        torch.ops.tensor_cast.grouped_matmul_swiglu.default: (
            torch.ops.tensor_cast.dispatch_ffn_combine.default
        ),
        torch.ops.tensor_cast.grouped_matmul_quant_swiglu.default: (
            torch.ops.tensor_cast.dispatch_ffn_combine_quant.default
        ),
        torch.ops.tensor_cast.grouped_matmul_quant_int4_swiglu.default: (
            torch.ops.tensor_cast.dispatch_ffn_combine_quant_int4.default
        ),
        torch.ops.tensor_cast.grouped_matmul_fp8_swiglu.default: (
            torch.ops.tensor_cast.dispatch_ffn_combine_fp8.default
        ),
        torch.ops.tensor_cast.grouped_matmul_mxfp4_swiglu.default: (
            torch.ops.tensor_cast.dispatch_ffn_combine_mxfp4.default
        ),
    }

    # Map from unfused linear_ffn target → DFC fused op target.
    # Used when SinkSplit doesn't group the MoE linear ops (the common case).
    # BF16 (aten.mm) is NOT included — it always takes the grouped path (Case 1)
    # because GroupedMatmulSwigluPass runs before DFC and successfully groups BF16.
    _DFC_OP_MAP_LINEAR = {
        torch.ops.tensor_cast.static_quant_linear.default: (
            torch.ops.tensor_cast.dispatch_ffn_combine_quant.default
        ),
        torch.ops.tensor_cast.static_quant_linear_int4.default: (
            torch.ops.tensor_cast.dispatch_ffn_combine_quant_int4.default
        ),
        torch.ops.tensor_cast.fp8_linear.default: (
            torch.ops.tensor_cast.dispatch_ffn_combine_fp8.default
        ),
        torch.ops.tensor_cast.mxfp4_linear.default: (
            torch.ops.tensor_cast.dispatch_ffn_combine_mxfp4.default
        ),
    }

    # Upper bound on BFS depth when traversing from init_routing_v2 to
    # unpermute_tokens.  Derived from the largest observed MoE subgraph
    # (DeepSeek-V3 with 5 quant variants × ~100 nodes each ≈ 500 nodes,
    # plus 20% headroom → 600).
    _MAX_TRAVERSE_DEPTH = 600

    def __call__(self, gm: fx.GraphModule) -> fx.GraphModule:
        graph = gm.graph
        modified = False

        # Pre-scan: find all init_routing_v2 (start) and unpermute_tokens (end) nodes
        all_permute_starts = []
        all_unpermute_ends = []

        for node in graph.nodes:
            if self._is_permute_token(node):
                all_permute_starts.append(node)
            if self._is_unpermute_token(node):
                all_unpermute_ends.append(node)

        # Traverse from each init_routing_v2 to find corresponding unpermute_tokens
        processed_nodes = set()
        for start_node in all_permute_starts:
            if start_node in processed_nodes:
                continue

            # Collect region nodes with forward BFS
            region_nodes, end_node = self._collect_region_nodes_forward(
                start_node, processed_nodes, self._MAX_TRAVERSE_DEPTH
            )

            if not region_nodes or end_node is None:
                continue

            # Check required operators in region
            has_required_ops, reason = self._check_region_features(region_nodes)
            if not has_required_ops:
                logger.debug(
                    "DispatchFFNCombinePass skip region start=%s end=%s reason=%s",
                    start_node.name,
                    end_node.name,
                    reason,
                )
                continue

            # Determine DFC variant and extract weight args
            result = self._resolve_dfc_variant(region_nodes)
            if result is None:
                logger.debug(
                    "DispatchFFNCombinePass skip: cannot resolve DFC variant "
                    "(start=%s end=%s)",
                    start_node.name,
                    end_node.name,
                )
                continue

            dfc_target, gmm1_w_args, gmm2_w_args, rank_node, rank_group_node = result

            # Build fused args
            fused_args = (
                start_node.args[0],  # x
                start_node.args[1],  # expert_indices
                *gmm1_w_args,  # GMM1 weights/scales/bias
                *gmm2_w_args,  # GMM2 weights/scales/bias
                rank_node,  # rank
                rank_group_node,  # rank_group
            )

            # Replace region with fused operator
            with graph.inserting_before(end_node):
                fused_node = graph.create_node(
                    "call_function",
                    dfc_target,
                    args=fused_args,
                    kwargs={},
                    name="dispatch_ffn_combine_fused",
                )

            # Redirect all uses of unpermute_tokens to fused node
            end_node.replace_all_uses_with(fused_node)
            processed_nodes.update(region_nodes)
            modified = True

        # Clean up graph and recompile
        if modified:
            stable_topo_sort(gm)
            gm.graph.eliminate_dead_code()
            gm.graph.lint()
            gm.recompile()

        return gm

    def _resolve_dfc_variant(self, region_nodes):
        """Determine DFC variant and extract weight args from region.

        Handles two graph shapes:
        1. Grouped: region has grouped_matmul_*_swiglu + grouped_matmul_*
           (after SinkSplit + GroupedMatmulSwigluPass)
        2. Unfused: region has individual static_quant_linear/etc + swiglu
           (before SinkSplit, or SinkSplit didn't group MoE path)

        Returns (dfc_target, gmm1_w_args, gmm2_w_args, rank, rank_group)
        or None if cannot resolve.
        """
        # Find all_to_all (present in EP mode, absent in non-EP MoE)
        all_to_all_node = None
        for node in region_nodes:
            if self._is_all_to_all(node):
                all_to_all_node = node
                break

        if all_to_all_node is not None:
            rank_node = all_to_all_node.args[3]
            rank_group_node = all_to_all_node.args[4]
        else:
            # Non-EP MoE (e.g., Qwen3): no all_to_all, comm cost = 0
            rank_node = 0
            rank_group_node = [0]

        # Case 1: Grouped ops (after SinkSplit + GroupedMatmulSwigluPass)
        gmm_swiglu_node = None
        gmm_plain_node = None
        for node in region_nodes:
            if self._is_grouped_matmul_swiglu(node):
                gmm_swiglu_node = node
            elif self._is_grouped_matmul(node):
                gmm_plain_node = node

        if gmm_swiglu_node is not None and gmm_plain_node is not None:
            dfc_target = self._DFC_OP_MAP_GMM.get(gmm_swiglu_node.target)
            if dfc_target is None:
                return None
            # Skip args[0] (activation) from each GMM node
            return (
                dfc_target,
                gmm_swiglu_node.args[1:],
                gmm_plain_node.args[1:],
                rank_node,
                rank_group_node,
            )

        # Case 1.5: Half-match — gmm_swiglu exists but gmm_plain is missing
        # (GroupedMatmulSwigluPass fused gate_up but GroupedMatmulPass didn't
        # fuse down_proj, leaving per-expert static_quant_linear nodes).
        if gmm_swiglu_node is not None and gmm_plain_node is None:
            logger.debug(
                "DFC: half-match — grouped_matmul_swiglu found but no "
                "grouped_matmul for down_proj. Skipping (SinkSplitPass "
                "did not fully group the MoE region)."
            )
            return None

        # Case 2: Unfused linear ops (common case — SinkSplit doesn't group MoE)
        # Identify gate_up and down_proj linears by tracing swiglu's inputs.
        swiglu_nodes = []
        linear_nodes = []
        for node in region_nodes:
            if (
                node.op == "call_function"
                and node.target == torch.ops.tensor_cast.swiglu.default
            ):
                swiglu_nodes.append(node)
            if self._is_linear_ffn(node):
                linear_nodes.append(node)

        if not swiglu_nodes or not linear_nodes:
            logger.debug("DFC: no swiglu or linear_ffn nodes in region")
            return None

        # Use the first linear node's target to determine quant variant
        linear_target = linear_nodes[0].target
        dfc_target = self._DFC_OP_MAP_LINEAR.get(linear_target)
        if dfc_target is None:
            logger.debug("DFC: unmapped linear target=%s", linear_target)
            return None

        # Case 2 regions contain one unfused branch per local expert:
        #   gate_up_linear -> split/getitem -> swiglu -> down_proj_linear
        # Collect every swiglu branch in graph order. The previous logic kept
        # only one swiglu node, so only one expert's gate_up linear landed in
        # GMM1 while all remaining gate_up linears were mis-bucketed into GMM2.
        graph_order = {
            node: idx for idx, node in enumerate(linear_nodes[0].graph.nodes)
        }
        swiglu_nodes.sort(key=graph_order.get)

        # Collect gate_up linears: trace back from ALL swiglu nodes
        gate_up_seen = set()
        gate_up_linears = []
        for swiglu_node in swiglu_nodes:
            for gate_up_node in self._collect_linear_predecessors(swiglu_node):
                if gate_up_node not in gate_up_seen:
                    gate_up_seen.add(gate_up_node)
                    gate_up_linears.append(gate_up_node)

        # Down linears = all linears NOT in gate_up set
        # (down_proj may not be a direct user of swiglu — there can be
        #  intermediate nodes like slice, quantize, copy between them)
        down_linears = [n for n in linear_nodes if n not in gate_up_seen]

        if not gate_up_linears:
            logger.debug("DFC: could not identify gate_up linear")
            return None

        if not down_linears:
            logger.debug("DFC: could not identify down_proj linear, skipping fusion")
            return None

        gate_up_linears.sort(key=graph_order.get)
        down_linears.sort(key=graph_order.get)

        # Collect ALL experts' weight args into List[Tensor] format.
        # Each expert has one gate_up linear and one down_proj linear.
        gmm1_w_args = self._collect_linear_args_as_lists(gate_up_linears)
        gmm2_w_args = self._collect_linear_args_as_lists(down_linears)

        return (dfc_target, gmm1_w_args, gmm2_w_args, rank_node, rank_group_node)

    @staticmethod
    def _collect_linear_args_as_lists(linear_nodes: list) -> tuple:
        """Collect weight args from multiple linear nodes into List[Tensor] format.

        Each linear op has signature (x, w, ...). We skip x (args[0]) and collect
        the remaining args across all nodes. For tensor args, we build a list
        containing one element per node (per expert). Non-tensor args (dtype, etc)
        are taken from the first node and passed through as-is.

        Example: 16 static_quant_linear nodes with args (x, w, ws, wo, xs, xo, bias, dt)
        → ([w0..w15], [ws0..ws15], [wo0..wo15], [xs0..xs15], [xo0..xo15], [b0..b15], dt)
        """
        if not linear_nodes:
            return ()
        template = linear_nodes[0]
        num_args = len(template.args)
        result = []
        for i in range(1, num_args):  # skip args[0] (activation)
            first_val = template.args[i]
            if isinstance(first_val, fx.Node) or first_val is None:
                # Tensor arg or optional tensor → collect from all nodes into a list.
                # Optional tensor (e.g., w_offset=None) → collect into List[None].
                # Downstream _swiglu_fusion_properties_helper handles List[None]
                # correctly via `w_offset and i < len(w_offset)` guard.
                result.append([node.args[i] for node in linear_nodes])
            else:
                # Non-tensor (dtype, etc) → take from first node
                result.append(first_val)
        return tuple(result)

    def _collect_linear_predecessors(self, node: fx.Node) -> list[fx.Node]:
        """Trace backwards from a node until the first linear_ffn ancestors."""
        predecessors = []
        seen = set()
        queue = [arg for arg in node.args if isinstance(arg, fx.Node)]

        while queue:
            current = queue.pop()
            if current in seen:
                continue
            seen.add(current)

            if self._is_linear_ffn(current):
                predecessors.append(current)
                continue

            for arg in getattr(current, "args", ()):
                if isinstance(arg, fx.Node):
                    queue.append(arg)

        return predecessors

    # Forward BFS to collect region nodes with depth limit
    def _collect_region_nodes_forward(
        self, start_node: fx.Node, processed: set, max_depth: int
    ) -> tuple[set, fx.Node]:
        region = set()
        q = deque([(start_node, 0)])
        end_node = None

        while q and end_node is None:
            n, depth = q.popleft()

            if depth > max_depth or n in region or n in processed:
                continue

            region.add(n)

            # Stop traversal if unpermute_tokens is found
            if self._is_unpermute_token(n):
                end_node = n
                continue

            # Skip placeholders/constants
            if n.op in ["placeholder", "get_attr"]:
                continue

            # Traverse to next nodes
            for user in n.users:
                if isinstance(user, fx.Node) and user not in region:
                    q.append((user, depth + 1))

        return (region, end_node) if end_node else (set(), None)

    # Node type check helpers
    def _is_permute_token(self, node: fx.Node) -> bool:
        return (
            node.op == "call_function"
            and node.target == torch.ops.tensor_cast.init_routing_v2.default
        )

    def _is_unpermute_token(self, node: fx.Node) -> bool:
        return (
            node.op == "call_function"
            and node.target == torch.ops.tensor_cast.unpermute_tokens.default
        )

    def _is_grouped_matmul(self, node: fx.Node) -> bool:
        return node.op == "call_function" and node.target in self._GROUPED_MATMUL_OPS

    def _is_grouped_matmul_swiglu(self, node: fx.Node) -> bool:
        return (
            node.op == "call_function"
            and node.target in self._GROUPED_MATMUL_SWIGLU_OPS
        )

    def _is_linear_ffn(self, node: fx.Node) -> bool:
        return node.op == "call_function" and node.target in self._LINEAR_FFN_OPS

    def _is_swiglu(self, node: fx.Node) -> bool:
        return node.op == "call_function" and node.target in self._SWIGLU_OPS

    def _is_all_to_all(self, node: fx.Node) -> bool:
        return (
            node.op == "call_function"
            and node.target == torch.ops.tensor_cast.all_to_all.default
        )

    # Check if region contains required MoE FFN operators
    def _check_region_features(self, region: set) -> tuple[bool, str]:
        has_ffn_compute = False
        has_permute = False
        has_unpermute = False
        has_swiglu = False

        for node in region:
            if self._is_permute_token(node):
                has_permute = True
            if self._is_unpermute_token(node):
                has_unpermute = True
            if (
                self._is_grouped_matmul(node)
                or self._is_grouped_matmul_swiglu(node)
                or self._is_linear_ffn(node)
            ):
                has_ffn_compute = True
            if self._is_swiglu(node):
                has_swiglu = True

        # Validate required operator counts
        if not has_permute:
            return False, "missing_init_routing_v2"
        if not has_unpermute:
            return False, "missing_unpermute_tokens"
        if not has_ffn_compute:
            return False, "missing_ffn_compute_ops"
        if not has_swiglu:
            return False, "missing_swiglu"
        return True, "matched"
