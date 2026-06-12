import dataclasses
import hashlib
import itertools
import logging
from typing import Dict, List, Optional

import torch
from torch.utils._cxx_pytree import tree_map

from .. import ops  # noqa: F401
from ..utils import EquivalentKeyManager
from .utils import bytes_of_tensor, is_noop_self_copy_op, is_view_op, run_once

logger = logging.getLogger(__name__)


class OpInvokeInfo:
    _op_properties_functors = {}

    @dataclasses.dataclass
    class ComputeOps:
        mma_ops: int = 0
        """Number of Matrix-Multiply-Accumulate ops"""
        gp_ops: int = 0
        """Number of General-Purpose ops"""

    @dataclasses.dataclass
    class PerformanceProperties:
        compute_ops: Dict[torch.dtype, "OpInvokeInfo.ComputeOps"] = dataclasses.field(default_factory=dict)
        memory_read_bytes: int = 0
        """Read-only bytes"""
        memory_write_bytes: int = 0
        """Write-only bytes"""
        memory_readwrite_bytes: int = 0
        """Read-write bytes"""

        def combine(self, other: "OpInvokeInfo.PerformanceProperties", compute_only=False):
            for dtype, compute_ops in other.compute_ops.items():
                if dtype not in self.compute_ops:
                    self.compute_ops[dtype] = OpInvokeInfo.ComputeOps()
                self.compute_ops[dtype].mma_ops += compute_ops.mma_ops
                self.compute_ops[dtype].gp_ops += compute_ops.gp_ops
            if not compute_only:
                self.memory_read_bytes += other.memory_read_bytes
                self.memory_write_bytes += other.memory_write_bytes
                self.memory_readwrite_bytes += other.memory_readwrite_bytes

    def __init__(self, func, args, kwargs, out, cache_key=None):
        self.func = func
        self.args = args
        self.kwargs = {} if kwargs is None else kwargs
        self.out = out
        self.cache_key = cache_key or self.compute_cache_key()

    @classmethod
    def get_op_properties_functor(cls, op):
        def default_functor(self: OpInvokeInfo) -> OpInvokeInfo.PerformanceProperties:
            """Default functor only counts in the memory accesses"""
            if is_view_op(self.func) or is_noop_self_copy_op(self.func, self.args):
                return OpInvokeInfo.PerformanceProperties()
            run_once(
                self.func,
                logger.warning,
                f"No op properties function defined for {self.func}, assuming it is memory-bandwidth bound.",
            )
            return self.get_memory_access_properties()

        if op not in OpInvokeInfo._op_properties_functors:
            return default_functor
        return OpInvokeInfo._op_properties_functors[op]

    @classmethod
    def register_op_properties(cls, op, override=False):
        def decorator(functor):
            if op in OpInvokeInfo._op_properties_functors:
                if override:
                    logger.warning("Overwriting existing properties functor for op: %s", op)
                else:
                    raise ValueError(f"Op {op} already registered")
            OpInvokeInfo._op_properties_functors[op] = functor
            return functor

        return decorator

    def get_memory_access_properties(
        self,
        exclude_input_ids: Optional[set] = None,
        exclude_output_ids: Optional[set] = None,
    ) -> "OpInvokeInfo.PerformanceProperties":
        """Get memory read/write properties"""
        exclude_input_ids = set() if exclude_input_ids is None else exclude_input_ids
        exclude_output_ids = set() if exclude_output_ids is None else exclude_output_ids
        memory_read_bytes = 0
        memory_write_bytes = 0
        memory_readwrite_bytes = 0
        args_schema = self.func._schema.arguments
        for i, arg in enumerate(itertools.chain(self.args, self.kwargs.values())):
            if i not in exclude_input_ids:
                inputs = arg if isinstance(arg, (list, tuple)) else [arg]
                if inputs and isinstance(inputs[0], torch.Tensor):
                    for tensor in inputs:
                        access_bytes = bytes_of_tensor(tensor)
                        if args_schema[i].is_out:
                            memory_write_bytes += access_bytes
                        elif args_schema[i].is_write:
                            memory_readwrite_bytes += access_bytes
                        else:
                            memory_read_bytes += access_bytes
        out = self.out if isinstance(self.out, (list, tuple)) else [self.out]
        for i, arg in enumerate(out):
            if isinstance(arg, torch.Tensor) and i not in exclude_output_ids:
                access_bytes = bytes_of_tensor(arg)
                memory_write_bytes += access_bytes
        return OpInvokeInfo.PerformanceProperties(
            memory_read_bytes=memory_read_bytes,
            memory_write_bytes=memory_write_bytes,
            memory_readwrite_bytes=memory_readwrite_bytes,
        )

    def get_perf_properties(self) -> "OpInvokeInfo.PerformanceProperties":
        functor = self.get_op_properties_functor(self.func)
        return functor(self)

    def compute_cache_key(self) -> str:
        """
        Compute an efficient cache key based on operation signature and tensor properties.
        This key represents the computational characteristics of the operation.

        Returns:
            A string hash that can be used as a cache key
        """
        key_components = []

        key_components.append(str(self.func))

        def add_tensor_info(t, components):
            if isinstance(t, torch.Tensor):
                components.extend(
                    [
                        str(t.shape),
                        str(t.dtype),
                        str(t.device),
                        str(t.stride()) if not t.is_contiguous() else "contiguous",
                        str(t.requires_grad),
                    ]
                )
            elif isinstance(t, (list, tuple)):
                components.append(type(t).__name__)
                components.append(str(len(t)))
                for item in t:
                    add_tensor_info(item, components)
            elif isinstance(t, dict):
                components.append("dict")
                components.append(str(len(t)))
                for key in sorted(t):
                    components.append(str(key))
                    add_tensor_info(t[key], components)
            else:
                components.append(type(t).__name__)
                components.append(str(t))

        for arg in self.args:
            add_tensor_info(arg, key_components)

        if self.kwargs:
            for k, v in sorted(self.kwargs.items()):
                key_components.append(k)
                add_tensor_info(v, key_components)

        # Create hash
        key_string = "|".join(key_components)
        return hashlib.sha256(key_string.encode()).hexdigest()

    def __repr__(self):
        return f"OpInvokeInfo({self.func}, {self.args}, {self.kwargs}, {self.out})"


class Region:
    # if A region is a refernce of B region, then B is root region
    root_region_id_to_reference_count = {}
    region_id_to_root_region_id = {}
    equivalent_tensor_id_manager = EquivalentKeyManager()

    def __init__(self, mark_begin: Optional[OpInvokeInfo]):
        # Region contains a sequence of op invocations excluding the region markers
        self.mark_begin = mark_begin
        self.mark_end: Optional[OpInvokeInfo] = None
        self.op_invoke_infos: List[OpInvokeInfo] = []
        self.reference_id = 0
        self.real_input_tensor = None
        self.real_output_tensor = None

    def _add_equivalent_info(self):
        Region.equivalent_tensor_id_manager.add_equivalent_keys(
            [
                (id(self.real_input_tensor), 0),
                (id(self.input_tensor), self.reference_id),
            ]
        )
        Region.equivalent_tensor_id_manager.add_equivalent_keys(
            [
                (id(self.real_output_tensor), 0),
                (id(self.output_tensor), self.reference_id),
            ]
        )

    @classmethod
    def get_tensor_id(cls, tensor, region_reference_id=0):
        raw_tensor_id = (id(tensor), region_reference_id)
        equivalent_tensor_id = cls.equivalent_tensor_id_manager.get_group_root_key((id(tensor), region_reference_id))
        return equivalent_tensor_id if equivalent_tensor_id is not None else raw_tensor_id

    def shallow_copy(self, real_input_tensor, real_output_tensor) -> "Region":
        copied_region = Region(None)
        copied_region.mark_begin = self.mark_begin
        copied_region.mark_end = self.mark_end
        copied_region.op_invoke_infos = self.op_invoke_infos
        copied_region.real_input_tensor = real_input_tensor
        copied_region.real_output_tensor = real_output_tensor
        root_id = Region.region_id_to_root_region_id.get(id(self), id(self))

        if root_id not in Region.root_region_id_to_reference_count:
            Region.root_region_id_to_reference_count[root_id] = 0
        Region.root_region_id_to_reference_count[root_id] += 1
        copied_region.reference_id = Region.root_region_id_to_reference_count[root_id]
        Region.region_id_to_root_region_id[id(copied_region)] = root_id
        copied_region._add_equivalent_info()
        return copied_region

    def finalize(self, mark_end: OpInvokeInfo):
        # Patch op_invoke_infos' in/out tensors so that they use the input of mark_begin
        # and output of mark_end so that the region is connected to the full model after
        # removing the markers.
        if self.reference_id != 0:
            raise ValueError("this region is a copied region, cannot finalize")

        def patch_inout(t):
            if not isinstance(t, torch.Tensor):
                return t
            if id(t) == id(self.mark_begin.out):
                return self.mark_begin.args[0]
            if id(t) == id(mark_end.args[0]):
                return mark_end.out
            return t

        self.mark_end = mark_end
        inouts = []
        for op_invoke_info in self.op_invoke_infos:
            inouts.append((op_invoke_info.args, op_invoke_info.kwargs, op_invoke_info.out))
        new_inouts = tree_map(patch_inout, inouts)
        for op_invoke_info, (new_args, new_kwargs, new_out) in zip(self.op_invoke_infos, new_inouts):
            op_invoke_info.args = new_args
            op_invoke_info.kwargs = new_kwargs
            op_invoke_info.out = new_out

        self.real_input_tensor = self.mark_begin.args[0]
        self.real_output_tensor = self.mark_end.out

        Region.region_id_to_root_region_id[id(self)] = id(self)
        Region.root_region_id_to_reference_count[id(self)] = 0
        self._add_equivalent_info()

    @property
    def input_tensor(self):
        return self.mark_begin.args[0]

    @property
    def output_tensor(self):
        assert self.mark_end is not None, "Region end not finalized"
        return self.mark_end.out
