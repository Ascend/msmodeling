import logging
import operator
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple

import torch
import torch.fx as fx
from torch.fx.node import Argument, Node, Target

from ... import ops  # noqa: F401
from ..pass_base import TensorCastGraphModulePass
from ..topo_sort import stable_topo_sort
from ..utils import get_node_shape, is_non_scalar_tensor_node, maybe_copy_meta

logger = logging.getLogger(__name__)


def _is_split_with_sizes_node(node: Node) -> bool:
    return node.target == torch.ops.aten.split_with_sizes.default


def _get_num_split_users(split_node: Node) -> int:
    return len(split_node.users)


def _get_getitem_sizes(split_node: Node) -> Dict[int, Argument]:
    getitem_sizes = {}
    split_sizes = split_node.args[1]
    assert isinstance(split_sizes, (list, tuple))
    for user in split_node.users:
        if user.target == operator.getitem:
            index = user.args[1]
            assert isinstance(index, int) and index < len(split_sizes), index
            getitem_sizes[index] = split_sizes[index]
    return getitem_sizes


def _is_cat_node(node: Node) -> bool:
    return node.target in (
        torch.ops.aten.cat.default,
        torch.ops.tensor_cast.cat.default,
    )


@dataclass
class SinkConfig:
    """
    Configuration for a sinkable operation.

    Attributes:
        source_op: The source op type that we support to be rewritten into
                   a single op and allows the split op to sink.
        split_input_indices: Which input args (by index) of the Source Op
                             are allowed to be split.
        split_output_indices: Which output args (by index) of the Source Op
                              should be split after sinking.
        rewrite_op: The target op (Rewrite Op) to rewrite to.
                    If None, Source Op == Rewrite Op.
        custom_arg_builder: Custom function to build args for the Rewrite Op.
        extra_check: Extra check function to validate a match.
    """

    source_op: Target
    split_input_indices: Set[int] = field(default_factory=set)
    split_output_indices: Set[int] = field(default_factory=set)
    rewrite_op: Optional[Target] = None
    custom_arg_builder: Optional[Callable] = None
    extra_check: Optional[Callable] = None

    # These fields are populated automatically from the op schema
    rewrite_input_types: List[type] = field(init=False, default_factory=list)
    rewrite_output_types: List[type] = field(init=False, default_factory=list)

    @staticmethod
    def _get_py_type_from_schema_type(schema_type: torch._C.Type) -> type:
        type_str = str(schema_type)

        if type_str.startswith("List["):
            return list
        if "Tensor" in type_str:
            return torch.Tensor
        if type_str == "int" or "SymInt" in type_str:
            return int
        if type_str == "float":
            return float
        if type_str == "bool":
            return bool
        if type_str == "str":
            return str
        if type_str == "ScalarType" or "dtype" in type_str:
            return torch.dtype
        if type_str == "None":
            return type(None)
        if "Scalar" in type_str:
            return float

        return object

    def __post_init__(self):
        schema_op_target = self.rewrite_op or self.source_op
        assert hasattr(schema_op_target, "_schema")
        schema = schema_op_target._schema

        # Populate rewrite_input_types
        self.rewrite_input_types = [self._get_py_type_from_schema_type(arg.type) for arg in schema.arguments]

        # Populate rewrite_output_types
        self.rewrite_output_types = [self._get_py_type_from_schema_type(ret.type) for ret in schema.returns]


@dataclass
class Match:
    """
    A match of sinkable split pattern, i.e. a group of source ops connected
    to split nodes and can be rewritten to a single op.
    """

    op_config: SinkConfig
    source_op_group: List[Node]
    """A group of nodes that are connected to a single split node via getitems"""
    split_args: Dict[int, Node]
    """split args: mapping from arg index to the split node"""
    uniform_args: Dict[int, Argument]
    """uniform args: mapping from arg index to the uniform arg node"""
    uniform_kwargs: Dict[str, Argument]


class SinkSplitPass(TensorCastGraphModulePass):
    _sink_config_registry: Dict[Target, SinkConfig] = {}
    _sink_config_registry_populated: bool = False

    @classmethod
    def _populate_registry(cls):
        if cls._sink_config_registry_populated:
            return
        cls._sink_config_registry_populated = True

        # The reason why we need these custom arg builders is that the args
        # do not match between the grouped ops and the original ops.
        def mm_arg_builder(graph, match):
            source_op_group = match.source_op_group
            x_list = [source_op.args[0] for source_op in source_op_group]
            w_list = [source_op.args[1] for source_op in source_op_group]
            bias_list = [None] * len(source_op_group)
            return (x_list, w_list, bias_list), {}

        def addmm_arg_builder(graph, match):
            source_op_group = match.source_op_group
            x_list = [source_op.args[1] for source_op in source_op_group]
            w_list = [source_op.args[2] for source_op in source_op_group]
            bias_list = [source_op.args[0] for source_op in source_op_group]
            return (x_list, w_list, bias_list), {}

        def split_with_sizes_extra_check(split_node, source_op_group, split_args, uniform_args, template_kwargs):
            """Extra check for split_with_sizes op to be sunk. Only allow
            different split dim from the split node.
            """
            source_op = source_op_group[0]
            assert _is_split_with_sizes_node(source_op) or source_op.target == torch.ops.aten.split.Tensor, (
                f"Assertion failed: expected operator is 'split_with_sizes' or 'split'."
                f"The operator currently executed is: {source_op.target}. Please check if the correct operator is used."
            )
            split_dim = split_node.args[2] if len(split_node.args) > 2 else 0
            source_op_split_dim = source_op.args[2] if len(source_op.args) > 2 else 0
            return split_dim != source_op_split_dim

        # Helper to safely add ops to the config map
        def add_config(source_op, *config_args):
            cls._sink_config_registry[source_op] = SinkConfig(source_op, *config_args)

        # TODO(jgong5): for most of the ops below, we should add an extra check to make sure
        #  the values match in their uniform args.
        # Unary ops
        unary_ops = [
            torch.ops.prims.convert_element_type.default,
            torch.ops.aten.sigmoid.default,
            torch.ops.tensor_cast.all_reduce.default,
        ]
        for op in unary_ops:
            add_config(op, {0}, {0})
        add_config(
            torch.ops.aten.split_with_sizes.default,
            {0},
            {0},
            None,
            None,
            split_with_sizes_extra_check,
        )
        add_config(
            torch.ops.aten.split.Tensor,
            {0},
            {0},
            None,
            None,
            split_with_sizes_extra_check,
        )

        # Binary ops
        binary_ops = [
            torch.ops.aten.mul.Tensor,
            torch.ops.tensor_cast.swiglu.default,
        ]
        for op in binary_ops:
            add_config(op, {0, 1}, {0})

        # Quantization ops
        add_config(torch.ops.tensor_cast.quantize.default, {0}, {0})
        add_config(
            torch.ops.tensor_cast.dynamic_quantize_asymmetric.default,
            {0},
            {0, 1, 2},
        )
        add_config(
            torch.ops.tensor_cast.dynamic_quantize_symmetric.default,
            {0},
            {0, 1},
        )
        add_config(torch.ops.tensor_cast.dynamic_quantize_mxfp4.default, {0}, {0})

        # Matmul Ops
        add_config(
            torch.ops.aten.mm.default,
            {0},
            {0},
            torch.ops.tensor_cast.grouped_matmul.default,
            mm_arg_builder,
        )
        add_config(
            torch.ops.aten.addmm.default,
            {1},
            {0},
            torch.ops.tensor_cast.grouped_matmul.default,
            addmm_arg_builder,
        )
        add_config(
            torch.ops.tensor_cast.static_quant_linear.default,
            {0, 4, 5},
            {0},
            torch.ops.tensor_cast.grouped_matmul_quant.default,
        )
        add_config(
            torch.ops.tensor_cast.static_quant_linear_int4.default,
            {0, 4, 5},
            {0},
            torch.ops.tensor_cast.grouped_matmul_quant_int4.default,
        )
        add_config(
            torch.ops.tensor_cast.fp8_linear.default,
            {0, 2},
            {0},
            torch.ops.tensor_cast.grouped_matmul_fp8.default,
        )
        add_config(
            torch.ops.tensor_cast.mxfp4_linear.default,
            {0, 2},
            {0},
            torch.ops.tensor_cast.grouped_matmul_mxfp4.default,
        )

    @staticmethod
    def _collapse_split_tree(graph: fx.Graph) -> bool:
        """
        Collapses trees of torch.ops.aten.split_with_sizes.default into a single split.

        Criteria for collapsing:
        1. Operations must be split_with_sizes.default.
        2. Operations must act on the same dimension.
        3. Intermediate 'getitem' nodes must be consumed EXCLUSIVELY by the child split.
        (If an intermediate tensor is used elsewhere, we cannot flatten it without
            introducing a complex 'cat' or breaking the other usage).
        """

        # --- Inner Helper Functions ---
        def _find_getitem_user(node: Node, index: int) -> Optional[torch.fx.Node]:
            """Find the getitem node that extracts 'index' from 'node'."""
            for user in node.users:
                if user.target == operator.getitem and user.args[1] == index:
                    return user
            return None

        def _is_input_from_another_split(split_node: Node) -> bool:
            """Check if the input tensor to this split comes from a getitem of another split."""
            inp = split_node.args[0]
            if not isinstance(inp, Node):
                return False

            # Is input a getitem?
            if inp.target == operator.getitem:
                parent_of_inp = inp.args[0]
                assert isinstance(parent_of_inp, Node)
                return _is_split_with_sizes_node(parent_of_inp)
            return False

        def _expand_branch(
            getitem_node: Node,
            current_size: int,
            root_dim: int,
            nodes_to_purge: List[Node],
        ) -> Tuple[List[Tuple[int, Optional[Node]]], bool]:
            """
            Recursively checks if a getitem node feeds into another compatible split.

            Args:
                nodes_to_purge: A list to collect nodes that will be removed if the rewrite happens.

            Returns:
                leaves: List of (size, node_to_replace)
                expanded: Boolean, true if we successfully descended into a child split.
            """

            nodes_to_purge.append(getitem_node)  # Will be replaced, so mark for deletion

            # Constraint 1: If the intermediate tensor is used by anything OTHER than
            # a single split node, we cannot collapse it (without adding 'cat' ops).
            # We strictly check if it has exactly one user.
            if len(getitem_node.users) != 1:
                return [(current_size, getitem_node)], False

            user = next(iter(getitem_node.users))

            # Constraint 2: User must be a split_with_sizes
            if not _is_split_with_sizes_node(user):
                return [(current_size, getitem_node)], False

            # Constraint 3: Dimensions must match
            child_dim = user.args[2] if len(user.args) > 2 else 0
            child_sizes = user.args[1]

            if child_dim != root_dim:
                return [(current_size, getitem_node)], False

            # We found a collapsible child split!
            # Mark the child split for deletion
            nodes_to_purge.append(user)

            leaves = []

            for i, size in enumerate(child_sizes):
                child_getitem = _find_getitem_user(user, i)
                if child_getitem is None:
                    leaves.append((size, None))
                else:
                    # Recurse further
                    sub_leaves, _ = _expand_branch(child_getitem, size, root_dim, nodes_to_purge)
                    leaves.extend(sub_leaves)

            return leaves, True

        changed = False
        for node in list(graph.nodes):
            if not _is_split_with_sizes_node(node):
                continue

            # Check if this node is a child of another split-tree we would have processed.
            if _is_input_from_another_split(node):
                continue

            # We have a potential root. Let's try to flatten the tree starting here.
            root_split = node

            # Get arguments: input_tensor, split_sizes, dim
            root_dim = root_split.args[2] if len(root_split.args) > 2 else 0
            current_sizes = root_split.args[1]

            # Recursive collection
            leaves = []
            nodes_to_purge = [root_split]  # We aim to delete the root split as well
            has_nested_splits = False

            # Iterate through the current split's logical outputs
            for i, size in enumerate(current_sizes):
                # Find the getitem node corresponding to this index
                getitem_node = _find_getitem_user(root_split, i)

                if getitem_node is None:
                    leaves.append((size, None))
                    continue

                # Check if this branch can be expanded
                branch_leaves, branch_expanded = _expand_branch(getitem_node, size, root_dim, nodes_to_purge)
                leaves.extend(branch_leaves)
                if branch_expanded:
                    has_nested_splits = True

            # If no nested splits were found, do nothing for this root
            if not has_nested_splits:
                continue

            changed = True
            # --- REWRITING THE GRAPH ---
            with graph.inserting_after(root_split):
                # Calculate new sizes
                new_sizes = [leaf[0] for leaf in leaves]

                # Create the new split node
                new_split = graph.call_function(
                    torch.ops.aten.split_with_sizes.default,
                    args=(root_split.args[0], new_sizes, root_dim),
                )
                new_split.meta = root_split.meta  # Copy metadata if exists

                # Rewire users
                for idx, (_, old_getitem) in enumerate(leaves):
                    if old_getitem is None:
                        continue

                    # Create a new getitem for this leaf
                    with graph.inserting_after(new_split):
                        new_getitem = graph.call_function(operator.getitem, args=(new_split, idx))

                    # Replace all usages of the old leaf node with the new one
                    # This makes 'old_getitem' have 0 users.
                    old_getitem.replace_all_uses_with(new_getitem)

            # Erase the old nodes. We iterate in reverse order (Bottom-Up).
            # This ensures that by the time we reach a parent node (like a split),
            # its children (getitems) have already been erased, so the parent has 0 users.
            for node_to_erase in reversed(nodes_to_purge):
                assert len(node_to_erase.users) == 0
                graph.erase_node(node_to_erase)

        return changed

    @staticmethod
    def _collapse_cat_tree(graph: fx.Graph) -> bool:
        """
        Collapses trees of torch.ops.aten.cat.default into a single cat.

        Criteria:
        1. Operations must be cat.default.
        2. Same dimension.
        3. Intermediate cat nodes must be consumed EXCLUSIVELY by the parent cat.
        """

        def _flatten_inputs(input_node: Node, root_dim: int, nodes_to_purge: List[Node]) -> Tuple[List[Node], bool]:
            """
            Recursively checks if an input node is a compatible cat node.
            Returns (flattened_input_list, was_expanded).
            """
            # Check basic compatibility
            if not isinstance(input_node, Node):
                return [input_node], False
            if not _is_cat_node(input_node):
                return [input_node], False
            if len(input_node.users) != 1:
                return [input_node], False

            # Check dimension
            # cat args: (tensors, dim=0)
            child_dim = input_node.args[1] if len(input_node.args) > 1 else 0

            if child_dim != root_dim:
                return [input_node], False

            # It is collapsible!
            nodes_to_purge.append(input_node)
            child_inputs_list = input_node.args[0]

            flattened_inputs = []
            # We don't need to track if children expanded, just that THIS node expanded

            for child_input in child_inputs_list:
                # Recurse upwards
                sub_inputs, _ = _flatten_inputs(child_input, root_dim, nodes_to_purge)
                flattened_inputs.extend(sub_inputs)

            return flattened_inputs, True

        # We process nodes. Because we collapse inputs (upwards), standard iteration works fine.
        # If we process a downstream CAT, it will recursively pull in upstream CATs.
        changed = False
        for node in reversed(list(graph.nodes)):
            if not _is_cat_node(node):
                continue

            root_cat = node
            root_dim = root_cat.args[1] if len(root_cat.args) > 1 else 0
            current_inputs = root_cat.args[0]

            nodes_to_purge = []
            new_inputs = []
            has_nested_cats = False

            for inp in current_inputs:
                flattened, expanded = _flatten_inputs(inp, root_dim, nodes_to_purge)
                new_inputs.extend(flattened)
                if expanded:
                    has_nested_cats = True

            if not has_nested_cats:
                continue

            changed = True
            # Rewrite
            with graph.inserting_after(root_cat):
                new_cat = graph.call_function(torch.ops.tensor_cast.cat.default, args=(new_inputs, root_dim))
                new_cat.meta = root_cat.meta

                root_cat.replace_all_uses_with(new_cat)
                # The root cat is now unused, mark for deletion
                nodes_to_purge.append(root_cat)

            # Cleanup
            for node_to_erase in reversed(nodes_to_purge):
                assert len(node_to_erase.users) == 0
                graph.erase_node(node_to_erase)

        return changed

    @staticmethod
    def _cancel_split_concat(graph: fx.Graph):
        """Remove the paired split-concat patterns in the graph."""
        nodes_to_clean = set()
        for cat_node in graph.nodes:
            if not _is_cat_node(cat_node):
                continue

            tensors_arg = cat_node.args[0]
            assert isinstance(tensors_arg, (list, tuple))
            assert len(tensors_arg) > 0

            if not all(isinstance(n, Node) for n in tensors_arg):
                continue

            first_input = tensors_arg[0]
            if first_input.target != operator.getitem:
                continue

            split_node = first_input.args[0]
            if not _is_split_with_sizes_node(split_node):
                continue

            if len(tensors_arg) != len(split_node.users):
                continue

            # cat is the only user of split-getitems
            is_valid_pattern = True
            for node in tensors_arg:
                if not (node.target == operator.getitem and node.args[0] == split_node and len(node.users) == 1):
                    is_valid_pattern = False
                    break

            if not is_valid_pattern:
                continue

            # getitem nodes used by cat cover all items of split outputs
            sorted_tensors_arg = sorted(tensors_arg, key=lambda n: n.args[1])
            for i, node in enumerate(sorted_tensors_arg):
                if node.args[1] != i:
                    is_valid_pattern = False
                    break

            if not is_valid_pattern:
                continue

            split_dim = split_node.args[2] if len(split_node.args) > 2 else 0
            cat_dim = cat_node.args[1] if len(cat_node.args) > 1 else 0

            if split_dim != cat_dim:
                continue

            input_tensor = split_node.args[0]
            cat_node.replace_all_uses_with(input_tensor)

            nodes_to_clean.add(cat_node)
            nodes_to_clean.add(split_node)
            nodes_to_clean.update(tensors_arg)

        for node in reversed(graph.nodes):
            if node in nodes_to_clean:
                graph.erase_node(node)

        return len(nodes_to_clean) > 0

    @staticmethod
    def _check_pattern(split_node: Node, op_registry: Dict[Target, SinkConfig]) -> List[Match]:
        """
        Checks if the users of a split_node match the sinking criteria.
        """

        def uniform_arg_match(left, right) -> bool:
            """Matching rule for non-split args:
            1. Shape matches for Tensor args.
            2. Value matches for non-Tensor args.
            """
            if isinstance(left, Node) and isinstance(right, Node):
                shape_left = get_node_shape(left)
                shape_right = get_node_shape(right)
                return shape_left is not None and shape_right is not None and shape_left == shape_right
            else:
                return left == right

        def get_source_op_groups() -> List[List[Node]]:
            """Get groups of source ops from the users of a split node. Ops in each group have the same target.
            Returns:
                List[List[Node]]: A list of source op groups, each group is a list of source op nodes.
            """
            source_op_groups: List[List[Node]] = []
            assert isinstance(split_node.args[1], (list, tuple))
            if len(split_node.args[1]) == 1:
                return source_op_groups
            num_split_users = _get_num_split_users(split_node)
            if num_split_users == 0:
                return source_op_groups
            getitem_nodes = sorted(split_node.users, key=lambda n: n.args[1])  # sort by index

            # Group source ops by their target
            target_to_group: Dict[Target, List[Node]] = {}
            for getitem_node in getitem_nodes:
                for user in getitem_node.users:
                    group = target_to_group.setdefault(user.target, [])
                    group.append(user)

            # Check that every user uses its getitem at the *same argument position*.
            # This pattern is required for "grouped" fusion (e.g., grouped_matmul).
            #
            # Valid example (multiple ops, same arg position):
            #   a, b = split(x)  # split produces 2 getitem nodes
            #   y1 = linear(a)   # a is at args[0] of linear1
            #   y2 = linear(b)   # b is at args[0] of linear2 → same position → OK
            #
            # Invalid (SwiGLU pattern):
            #   a, b = split(x)  # split produces 2 getitem nodes
            #   y = swiglu(a, b) # 1 op uses 2 getitems → filtered out
            source_op_groups = []
            for group in target_to_group.values():
                if len(group) != num_split_users:
                    continue

                arg_pos_set = set()
                for user_node in group:
                    if user_node.op != "call_function":
                        break

                    current_pos = []
                    for arg_idx, arg in enumerate(user_node.args):
                        if (
                            isinstance(arg, Node)
                            and arg.target == operator.getitem
                            and len(arg.args) > 0
                            and arg.args[0] == split_node
                        ):
                            current_pos.append(arg_idx)

                    if len(current_pos) != 1:
                        break

                    arg_pos_set.add(current_pos[0])
                    if len(arg_pos_set) > 1:
                        break
                if len(arg_pos_set) == 1:
                    source_op_groups.append(group)
            return source_op_groups

        matches = []
        source_op_groups = get_source_op_groups()
        if not source_op_groups:
            return matches
        for source_op_group in source_op_groups:
            source_op_target = source_op_group[0].target
            if source_op_target not in op_registry:
                continue

            op_config = op_registry[source_op_target]

            # Check Args
            template_source_op = source_op_group[0]
            template_args = template_source_op.args
            template_kwargs = template_source_op.kwargs

            split_args = {}
            uniform_args = {}
            matched = True
            # The default checking rules:
            # 1. For inputs that allow splitting, we check if they come from some split node.
            #    If all of the inputs come from the split node, we need all of them come from
            #    a single split node and use the getitem nodes in an ordered manner. Otherwise,
            #    we require all of them use the same node and treat them as uniform args.
            # 2. For inputs that don't allow splitting, we make sure their shapes match for
            #    tensors and values match for non-tensors.
            for i, arg_node in enumerate(template_args):
                if i in op_config.split_input_indices and is_non_scalar_tensor_node(arg_node):
                    if all(
                        isinstance(source_op.args[i], Node)
                        and source_op.args[i].target == operator.getitem
                        and len(source_op.args[i].args) > 0
                        and _is_split_with_sizes_node(source_op.args[i].args[0])
                        for source_op in source_op_group
                    ):
                        _split_node = arg_node.args[0]
                        if _get_num_split_users(_split_node) != len(source_op_group):
                            matched = False
                            break
                        for source_op in source_op_group:
                            _arg_node = source_op.args[i]
                            if _arg_node.args[0] != _split_node:
                                matched = False
                                break
                        if not matched:
                            break
                        split_args[i] = _split_node
                        continue
                    elif any(source_op.args[i] != arg_node for source_op in source_op_group):
                        matched = False
                        break
                uniform_args[i] = arg_node

            if not matched:
                continue

            for user in source_op_group[1:]:
                for i, val in uniform_args.items():
                    if not uniform_arg_match(user.args[i], val):
                        matched = False
                        break
                if not matched:
                    break
                if len(user.kwargs) != len(template_kwargs):
                    matched = False
                    break
                # NOTE: we assume all kwargs are uniform args here
                if not all(
                    uniform_arg_match(kwarg, template_kwarg)
                    for kwarg, template_kwarg in zip(user.kwargs.values(), template_kwargs.values())
                ):
                    matched = False
                    break

            if not matched:
                continue

            if op_config.extra_check and not op_config.extra_check(
                split_node, source_op_group, split_args, uniform_args, template_kwargs
            ):
                continue

            matches.append(
                Match(
                    op_config,
                    source_op_group,
                    split_args,
                    uniform_args,
                    dict(template_kwargs),
                )
            )

        return matches

    @staticmethod
    def _build_new_op(
        graph: fx.Graph,
        match: Match,
    ) -> Node:
        """
        Creates the new Rewrite Op node
        """
        source_op_group = match.source_op_group
        op_config = match.op_config
        split_args = match.split_args
        uniform_args = match.uniform_args
        uniform_kwargs = match.uniform_kwargs

        template_source_op = source_op_group[0]
        rewrite_op_target = op_config.rewrite_op or template_source_op.target

        if op_config.custom_arg_builder:
            new_op_args, new_op_kwargs = op_config.custom_arg_builder(graph, match)
        else:
            # Default arg building rule:
            # 1. For split args, use the split node's first output if the rewrite op
            #    expects a single tensor, else build a list of all split outputs.
            # 2. For uniform args, use the uniform node directly since we assume they are the same
            #    across all source ops.
            new_op_args = []
            for i, arg_type in enumerate(op_config.rewrite_input_types):
                if i >= len(template_source_op.args):
                    continue

                if arg_type == list and rewrite_op_target != template_source_op.target:  # noqa: E721
                    # when we use a different rewrite op which groups the input args
                    # into a list, such as grouped matmul, we need to build a list of split outputs.
                    arg_list = [source_op.args[i] for source_op in source_op_group]
                    new_op_args.append(arg_list)
                else:
                    if i in split_args:
                        new_op_args.append(split_args[i].args[0])
                    else:
                        assert i in uniform_args
                        uniform_node = uniform_args[i]
                        new_op_args.append(uniform_node)

            new_op_args = tuple(new_op_args)
            new_op_kwargs = uniform_kwargs

        with graph.inserting_before(template_source_op):
            new_op_node = graph.call_function(rewrite_op_target, args=new_op_args, kwargs=new_op_kwargs)
        return new_op_node

    def _rewrite_outputs(
        self,
        graph: fx.Graph,
        match: Match,
        rewrite_op: Node,
    ):
        """
        Rewrites the outputs of the new Rewrite Op node.
        We build new split nodes on outputs that are marked as split outputs in the config.
        For other outputs, we directly replace the old uses to use the new single output.
        """
        source_op_group = match.source_op_group

        # we assume the output splits share the same split dim as the input ones
        assert match.split_args
        split_node = next(iter(match.split_args.values()))
        split_sizes = split_node.args[1]
        split_dim = split_node.args[2] if len(split_node.args) > 2 else 0
        getitem_sizes = _get_getitem_sizes(split_node)

        op_config = match.op_config
        num_outputs = len(op_config.rewrite_output_types)

        for i in range(num_outputs):
            output_type = op_config.rewrite_output_types[i]
            old_output_nodes = source_op_group
            new_output_node = rewrite_op
            if num_outputs > 1:
                # here, we are handling ops like dynamic quant node which
                # has multiple outputs, so we need to get the user of getitem node
                old_output_nodes = []
                for source_op in source_op_group:
                    for getitem in source_op.users:
                        assert getitem.target == operator.getitem
                        if getitem.args[1] == i:
                            old_output_nodes.append(getitem)
                            break
                assert len(old_output_nodes) == len(source_op_group)
                with graph.inserting_after(new_output_node):
                    new_output_node = graph.call_function(operator.getitem, args=(new_output_node, i))
                    maybe_copy_meta(new_output_node, old_output_nodes[0])

            old_node_shape = get_node_shape(old_output_nodes[0])
            assert output_type == list or old_node_shape is not None  # noqa: E721
            if i in op_config.split_output_indices and (output_type == list or len(old_node_shape) > 0):  # noqa: E721
                if output_type == list:  # noqa: E721
                    # we have to split on the getitem outputs so
                    # we insert the getitems after the split first
                    new_splits = {}
                    with graph.inserting_after(new_output_node):
                        old_template_node = old_output_nodes[0]
                        for user in old_template_node.users:
                            assert user.target == operator.getitem
                            index = user.args[1]
                            new_getitem = graph.call_function(operator.getitem, args=(new_output_node, index))
                            new_splits[index] = graph.call_function(
                                torch.ops.aten.split_with_sizes.default,
                                args=(new_getitem, split_sizes, split_dim),
                            )
                    old_node_idx = 0
                    for j, split_size in getitem_sizes.items():
                        if split_size == 0:
                            continue
                        assert old_node_idx < len(old_output_nodes)
                        old_node = old_output_nodes[old_node_idx]
                        for user in old_node.users:
                            assert user.target == operator.getitem
                            index = user.args[1]
                            with graph.inserting_after(new_splits[index]):
                                new_getitem = graph.call_function(operator.getitem, args=(new_splits[index], j))
                                maybe_copy_meta(new_getitem, user)
                            user.replace_all_uses_with(new_getitem)
                        old_node_idx += 1
                else:
                    with graph.inserting_after(new_output_node):
                        new_split = graph.call_function(
                            torch.ops.aten.split_with_sizes.default,
                            args=(new_output_node, split_sizes, split_dim),
                        )

                    old_node_idx = 0
                    for j, split_size in getitem_sizes.items():
                        if split_size == 0:
                            continue
                        assert old_node_idx < len(old_output_nodes)
                        old_node = old_output_nodes[old_node_idx]
                        with graph.inserting_after(new_split):
                            new_getitem = graph.call_function(operator.getitem, args=(new_split, j))
                            maybe_copy_meta(new_getitem, old_node)
                        old_node.replace_all_uses_with(new_getitem)
                        old_node_idx += 1
            else:
                for old_node in old_output_nodes:
                    old_node.replace_all_uses_with(new_output_node)

    def _cleanup_nodes(self, graph: fx.Graph, match: Match):
        """
        Cleans up the old nodes in the matched pattern.
        """
        source_op_group = match.source_op_group

        for source_op in source_op_group:
            users = list(source_op.users)
            if users and users[0].target == operator.getitem:
                getitem_nodes = users
                for getitem in getitem_nodes:
                    graph.erase_node(getitem)
            graph.erase_node(source_op)

    def _run_sinking_pass(self, graph: fx.Graph, op_registry: Dict[Target, SinkConfig]):
        pass_changed = False

        for node in reversed(graph.nodes):
            if not _is_split_with_sizes_node(node):
                continue

            split_node = node

            matches = self._check_pattern(split_node, op_registry)
            if not matches:
                continue

            for match in matches:
                new_op_node = self._build_new_op(graph, match)
                self._rewrite_outputs(graph, match, new_op_node)
                self._cleanup_nodes(graph, match)

            pass_changed = True

        return pass_changed

    def __call__(self, gm: fx.GraphModule) -> fx.GraphModule:
        """
        Applies the Split Sinking optimization pass to a GraphModule.

        The pass performs two main transformations:
        1. Cancels redundant `split -> cat` patterns.
        2. "Sinks" `split_with_sizes` ops past their users,
        grouping the user ops into more efficient "grouped"
        kernels (such as grouped matmul) where possible.
        """

        # Ensure all ops are registered
        self._populate_registry()

        while True:
            changed_collapse_split = self._collapse_split_tree(gm.graph)
            changed_collapse_cat = self._collapse_cat_tree(gm.graph)
            changed_concat = self._cancel_split_concat(gm.graph)
            changed_sinking = self._run_sinking_pass(gm.graph, self._sink_config_registry)
            if not (changed_collapse_split or changed_collapse_cat or changed_concat or changed_sinking):
                break
        stable_topo_sort(gm)
        gm.graph.eliminate_dead_code()
        gm.recompile()
        return gm
