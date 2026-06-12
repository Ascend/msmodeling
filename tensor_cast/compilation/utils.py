import torch
import torch.fx
from torch.fx.node import Node


def get_node_shape(node: Node) -> torch.Size | None:
    """Retrieve the shape of the tensor represented by the node, if available."""
    if not hasattr(node, "meta") or "val" not in node.meta:
        return None
    if isinstance(node.meta["val"], torch.Tensor):
        return node.meta["val"].shape
    else:
        return None


def is_non_scalar_tensor_node(node: Node) -> bool:
    """Check if a node represents a non-scalar tensor."""
    if not hasattr(node, "meta") or "val" not in node.meta:
        return False
    val = node.meta["val"]
    return isinstance(val, torch.Tensor) and len(val.shape) > 0


def maybe_copy_meta(target_node: Node, source_node: Node):
    """Copy meta information from source_node to target_node if available."""
    if hasattr(source_node, "meta"):
        target_node.meta = dict(source_node.meta)
