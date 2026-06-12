import collections
import functools
import logging
import operator
from typing import Any, Callable, List, Optional, Tuple

import torch
from torch.fx import Node

from ..pass_base import TensorCastGraphModulePass
from ..topo_sort import stable_topo_sort
from ..utils import get_node_shape

logger = logging.getLogger(__name__)

# --- Type Alias for Merge Specification ---
# (grouping_key_args, grouping_key_kwargs, w_idx, bias_idx)
OpMergeSpec = Tuple[Callable[[Node], Tuple[Any, ...]], int, Optional[int], Optional[int], Optional[int]]


class MergeLinearPass(TensorCastGraphModulePass):
    """
    FX Pass to find and merge linear-like ops (quant_linear, mm).

    It inserts 'aten.cat' nodes to merge the 'w' and 'bias' inputs, and
    'w_scale/w_offset' if applicable, and creates a single merged op.
    The output of the merged op is then split and routed to the original
    consumers.

    It relies on 'node.meta['val']' containing shape information for
    the 'w' tensors to correctly create the output split.

    **Requirements:**
    1. Merges ops that share the same non-weight/non-bias arguments.
    2. For ops with an optional bias, a group is only merged if ALL
       ops in the group have a tensor bias OR all have a None bias.
    3. NOTE: For w_scale and w_offset, we use loose conditions that only check
       their shapes (ndim or full shape) to be the same across the group with
       the assumption that the values of them are set consistently so that
       we can merge them safely. This is only valid for performance simulation
       purpose. In the real scenario, we should make sure their values match
       in some scenarios, e.g., when it is per-tensor or per-group quantization.
    """

    def _get_merge_spec(self, node: Node) -> Optional[OpMergeSpec]:
        """
        Returns the merge specification for a given node, if supported.
        """

        def default_grouping_key_func(grouping_key_args, n):
            return (n.target, tuple(n.args[i] for i in grouping_key_args))

        def ndim_key(n, arg_idx):
            assert isinstance(n.args[arg_idx], Node)
            shape = get_node_shape(n.args[arg_idx])
            assert shape is not None
            return len(shape)

        def shape_key(n, arg_idx):
            assert isinstance(n.args[arg_idx], Node)
            shape = get_node_shape(n.args[arg_idx])
            assert shape is not None
            return shape

        # static_quant_linear(x, w, w_scale, w_offset, x_scale, x_offset, bias, out_dtype)
        if node.target in (
            torch.ops.tensor_cast.static_quant_linear.default,
            torch.ops.tensor_cast.static_quant_linear_int4.default,
        ):
            # Group by: x, w_scale, w_offset, x_scale, x_offset, out_dtype
            grouping_key_args = (0, 4, 5, 7)
            # Concat: w (idx 1) and bias (idx 6)
            w_idx, bias_idx, w_scale, w_offset = 1, 6, 2, 3

            def grouping_key_func_quant_linear(n):
                key = default_grouping_key_func(grouping_key_args, n)
                w_scale_key = ndim_key(n, w_scale)
                if n.args[w_offset] is not None:
                    w_offset_key = ndim_key(n, w_offset)
                else:
                    w_offset_key = None
                return key + (w_scale_key, w_offset_key)

            return grouping_key_func_quant_linear, w_idx, bias_idx, w_scale, w_offset

        # aten.mm.default(x, w)
        if node.target == torch.ops.aten.mm.default:
            # Group by: x
            grouping_key_args = (0,)
            # Concat: w (idx 1)
            w_idx, bias_idx = 1, None
            grouping_key_func = functools.partial(default_grouping_key_func, grouping_key_args)
            return grouping_key_func, w_idx, bias_idx, None, None

        # aten.addmm.default(bias, x, w)
        if node.target == torch.ops.aten.addmm.default:
            # Group by: x
            grouping_key_args = (1,)
            # Concat: w (idx 2)
            w_idx, bias_idx = 2, 0
            grouping_key_func = functools.partial(default_grouping_key_func, grouping_key_args)
            return grouping_key_func, w_idx, bias_idx, None, None

        if node.target in (
            torch.ops.tensor_cast.fp8_linear.default,
            torch.ops.tensor_cast.mxfp4_linear.default,
        ):
            # Group by: x, x_scale, w_scale, out_dtype
            grouping_key_args = (0, 2, 5)
            # Concat: w (idx 1) and bias (idx 4)
            w_idx, bias_idx, w_scale = 1, 4, 3

            def grouping_key_func_fp8_mxfp4(n):
                key = default_grouping_key_func(grouping_key_args, n)
                if n.target == torch.ops.tensor_cast.fp8_linear.default:
                    w_scale_key = ndim_key(n, w_scale)
                else:
                    assert n.target == torch.ops.tensor_cast.mxfp4_linear.default
                    w_scale_key = shape_key(n, w_scale)
                return key + (w_scale_key,)

            return grouping_key_func_fp8_mxfp4, w_idx, bias_idx, w_scale, None

        return None

    def __call__(self, gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
        logger.debug("Running MergeLinearPass.........")

        # {grouping_key: [list_of_nodes]}
        groups = collections.defaultdict(list)

        # 1. Group all target nodes
        for node in gm.graph.nodes:
            if node.op != "call_function":
                continue

            spec = self._get_merge_spec(node)
            if spec is None:
                continue

            grouping_key_func, *_ = spec
            # key_args = tuple(node.args[i] for i in grouping_key_args_idx)
            key = grouping_key_func(node)
            # key = (node.target, key_args)
            groups[key].append(node)

        logger.debug(
            "Found %d groups to process in MergeLinearPass",
            len([group for group in groups.values() if len(group) > 1]),
        )

        # 2. Process each group
        for group in groups.values():
            if len(group) <= 1:
                continue

            ref_node = group[0]
            spec = self._get_merge_spec(ref_node)
            assert spec is not None
            _, w_arg_idx, bias_arg_idx, w_scale_idx, w_offset_idx = spec

            w_nodes: List[Node] = []
            bias_nodes: List[Node] = []
            w_scale_nodes: List[Node] = []
            w_offset_nodes: List[Node] = []
            split_sizes: List[int] = []
            can_merge = True
            has_bias = bias_arg_idx is not None
            has_w_scale = w_scale_idx is not None
            has_w_offset = w_offset_idx is not None

            # 3. Enforce bias consistency and collect inputs
            all_bias_none = True
            if has_bias:
                is_first_bias_none = ref_node.args[bias_arg_idx] is None
                all_bias_none = is_first_bias_none

                for node in group:
                    current_bias_arg = node.args[bias_arg_idx]
                    current_bias_is_none = current_bias_arg is None

                    if current_bias_is_none != is_first_bias_none:
                        can_merge = False
                        break

                    if not current_bias_is_none:
                        assert isinstance(current_bias_arg, Node)
                        bias_nodes.append(current_bias_arg)

                if not can_merge:
                    continue  # Skip group, mixed bias

            # 4. Collect 'w' nodes and check for shape metadata
            for node in group:
                w_node = node.args[w_arg_idx]
                assert isinstance(w_node, Node), "Expected w_node to be a Node"

                # Need shape meta to create the split
                shape = get_node_shape(node)
                if shape is None:
                    can_merge = False
                    break

                split_sizes.append(shape[-1])
                w_nodes.append(w_node)
                if has_w_scale:
                    w_scale_nodes.append(node.args[w_scale_idx])
                if has_w_offset:
                    w_offset_nodes.append(node.args[w_offset_idx])

            if not can_merge:
                continue

            # 5. Insert concatenation and merged op nodes
            insertion_point = group[-1]

            def insert_concat_node(insertion_point, nodes: List[Node], dim) -> Node:
                with gm.graph.inserting_before(insertion_point):
                    cat_node = gm.graph.create_node(
                        "call_function",
                        torch.ops.tensor_cast.cat.default,
                        args=(nodes, dim),
                    )
                return cat_node

            # Insert 'w' concatenation
            w_cat_node = insert_concat_node(insertion_point, w_nodes, 1)

            # Insert 'bias' concatenation
            bias_cat_node = None
            if has_bias and not all_bias_none:
                bias_cat_node = insert_concat_node(insertion_point, bias_nodes, 0)

            # Insert 'w_scale/w_offset' concatenation for per-channel quantization
            w_scale_cat_node = None
            w_offset_cat_node = None
            if ref_node.target in (
                torch.ops.tensor_cast.static_quant_linear.default,
                torch.ops.tensor_cast.static_quant_linear_int4.default,
                torch.ops.tensor_cast.fp8_linear.default,
            ):
                # static_quant_linear case
                if has_w_scale:
                    w_scale_ref_node = ref_node.args[w_scale_idx]
                    shape = get_node_shape(w_scale_ref_node)
                    assert shape is not None
                    if len(shape) == 1:
                        w_scale_cat_node = insert_concat_node(insertion_point, w_scale_nodes, len(shape) - 1)

                if has_w_offset:
                    w_offset_ref_node = ref_node.args[w_offset_idx]
                    if w_offset_ref_node is not None:
                        shape = get_node_shape(w_offset_ref_node)
                        assert shape is not None
                        if len(shape) == 1:
                            w_offset_cat_node = insert_concat_node(insertion_point, w_offset_nodes, len(shape) - 1)

            # Create the new merged op
            new_args = list(ref_node.args)
            new_kwargs = dict(ref_node.kwargs)

            new_args[w_arg_idx] = w_cat_node
            if bias_cat_node is not None:
                assert bias_arg_idx is not None
                new_args[bias_arg_idx] = bias_cat_node
            if w_scale_cat_node is not None:
                assert w_scale_idx is not None
                new_args[w_scale_idx] = w_scale_cat_node
            if w_offset_cat_node is not None:
                assert w_offset_idx is not None
                new_args[w_offset_idx] = w_offset_cat_node

            with gm.graph.inserting_before(insertion_point):
                merged_node = gm.graph.create_node(
                    "call_function",
                    ref_node.target,
                    args=tuple(new_args),
                    kwargs=new_kwargs,
                )

            # 6. Split the output of the merged op and replace uses
            with gm.graph.inserting_after(merged_node):
                split_node = gm.graph.create_node(
                    "call_function",
                    torch.ops.aten.split_with_sizes.default,
                    args=(merged_node, split_sizes, 1),
                )

            for i, old_node in enumerate(group):
                with gm.graph.inserting_after(old_node):
                    getitem_node = gm.graph.create_node("call_function", operator.getitem, args=(split_node, i))
                    old_node.replace_all_uses_with(getitem_node)

        # 7. Clean up the graph
        # Make sure the graph is topologically ordered first before DCE
        stable_topo_sort(gm)
        gm.graph.eliminate_dead_code()
        gm.graph.lint()
        gm.recompile()
        return gm
