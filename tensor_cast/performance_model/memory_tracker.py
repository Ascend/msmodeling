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

import dataclasses
import logging
from enum import auto, Enum
from typing import Any, cast, Dict, List, NamedTuple, Optional, Set

import torch

from ..device import DeviceProfile
from ..memory_accounting import ForwardMemorySummary
from ..performance_model import OpInvokeInfo
from ..performance_model.op_invoke_info import Region, RegionIdentityRegistry
from .utils import bytes_of_tensor

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class OpMemoryProfile:
    op_invoke_info: Optional[OpInvokeInfo]
    usage_before_call_bytes: float
    usage_after_call_bytes: float


@dataclasses.dataclass
class _TensorInfo:
    size_bytes: int
    def_op_idx: int = -1
    use_op_indices: List[int] = dataclasses.field(default_factory=list)
    use_op_indices_by_alias: List[int] = dataclasses.field(default_factory=list)
    """Indirect op use via its alias"""
    last_use_op_idx: int = -1
    # Storage-level identity for view/repack-robust param matching.
    storage_ptr: int = 0
    base_id: int = 0
    # Object identity at first record, for matching compile-time folded
    # constants (``_frozen_param*`` buffers) that carry weight bytes but were
    # never registered as params.
    obj_id: int = 0


def _base_root_id(tensor) -> int:
    """id() of the view-chain root tensor (the tensor itself when not a view)."""
    base = tensor
    seen = 0
    while getattr(base, "_base", None) is not None and seen < 32:
        base = base._base
        seen += 1
    return id(base)


def _storage_ptr(tensor) -> int:
    """data_ptr of the untyped storage; 0 when unavailable (e.g. meta tensors)."""
    try:
        return int(tensor.untyped_storage().data_ptr())
    except Exception:  # noqa: BLE001 - meta/fake tensors may not expose storage
        return 0


class TensorKey(NamedTuple):
    tensor_id: int
    repeat_id: int


# Alias plan format:
# - arg_names: schema argument names indexed by argument position.
# - output_plans: per-output alias mapping instructions as
#   (output_kind, output_index, candidate_input_indices), where:
#   * output_kind: OutputKind.TENSOR or OutputKind.LIST
#   * output_index: index in schema.returns / runtime outputs
#   * candidate_input_indices: schema argument indices that may alias this output
class OutputKind(Enum):
    TENSOR = auto()
    LIST = auto()


AliasPlan = tuple[List[str], List[tuple[OutputKind, int, List[int]]]]
ZeroStorageInputSpec = int | str


_MISSING = object()


class MemoryTracker:
    """
    Tracks the memory allocation during the execution of a PyTorch program
    and gives its memory profiles.

    The tracker works in two stages:
    Stage 1 - Recording stage
        Record the input and output tensors of each op execution via multiple calls to `record_op_invocation`.
        We also record the memory consumption during each op execution where op might allocate temporary memory
        as the workspace.
    Stage 2 - Analysis stage
        After the execution of the model, we analyze the def-use and liveness of the input/output tensors,
        identifying what are the model inputs (no def) and outputs (no use). Together with the temporary memory
        usage, we can plan the memory allocations as follows:
        1. All model inputs are allocated before all op execution.
        2. Model inputs and outputs are never freed.
        3. Intermediate tensors (def>=1 and use>=1) are allocated from a memory pool so that buffers are reused.
        4. Intermediate tensors are freed immediately after last use.
        5. Workspace buffers are allocated and freed in the same memory pool as intermediate tensors.
        6. Tensors have aliases. We don't allocate aliased outputs and only free those with all aliases unused later.

    After these two stages, the caller can get a memory profile of the program by calling `get_profile` which gives
    the memory consumption before and after each op invocation.
    """

    _zero_storage_input_specs: Dict[Any, Set[ZeroStorageInputSpec]] = {}
    _default_zero_storage_input_specs: Dict[Any, Set[ZeroStorageInputSpec]] = {}

    @staticmethod
    def _validate_zero_storage_input_specs(op: Any, input_specs: tuple[ZeroStorageInputSpec, ...]) -> None:
        schema = getattr(op, "_schema", None)
        valid_names = {arg.name for arg in schema.arguments} if schema is not None else set()

        for input_spec in input_specs:
            if isinstance(input_spec, int):
                if input_spec < 0:
                    raise ValueError(f"Zero-storage input spec index must be non-negative, got {input_spec}")
            elif isinstance(input_spec, str) and schema is not None and input_spec not in valid_names:
                raise ValueError(f"Zero-storage input spec name {input_spec!r} not found in op schema")

    @classmethod
    def register_zero_storage_input(cls, op: Any, *input_specs: ZeroStorageInputSpec) -> None:
        """Register op inputs whose tensor shape is semantic metadata only.

        ``input_specs`` entries may be schema argument indices or names. During
        liveness analysis the matching tensors keep their def/use edges, but
        their modeled storage size is forced to zero.

        This is a process-global tensor storage override: once a registered
        consumer marks a tensor as zero-storage, every reference to that tensor
        across the liveness window is modeled as zero bytes. Only register
        tensors that are metadata for their full lifetime.
        """
        if not input_specs:
            raise ValueError("At least one zero-storage input spec is required.")
        cls._validate_zero_storage_input_specs(op, input_specs)
        cls._zero_storage_input_specs.setdefault(op, set()).update(input_specs)

    @classmethod
    def register_default_zero_storage_input(cls, op: Any, *input_specs: ZeroStorageInputSpec) -> None:
        """Register process-default zero-storage inputs.

        Test fixtures can call ``restore_default_zero_storage_inputs`` after
        tests that mutate the active registry, so import-time model defaults do
        not disappear after an unrelated ``clear_zero_storage_inputs`` call.
        """
        cls.register_zero_storage_input(op, *input_specs)
        cls._default_zero_storage_input_specs.setdefault(op, set()).update(input_specs)

    @classmethod
    def clear_zero_storage_inputs(cls) -> None:
        """Clear active zero-storage registrations.

        Defaults registered with ``register_default_zero_storage_input`` are not
        deleted from the default registry; call
        ``restore_default_zero_storage_inputs`` to reapply them.
        """
        cls._zero_storage_input_specs.clear()

    @classmethod
    def restore_default_zero_storage_inputs(cls) -> None:
        cls._zero_storage_input_specs.clear()
        for op, input_specs in cls._default_zero_storage_input_specs.items():
            cls._zero_storage_input_specs[op] = set(input_specs)

    def __init__(self, device_profile: DeviceProfile):
        self.device_profile = device_profile
        self.region_identity_registry = RegionIdentityRegistry()
        # A list to store invocation details for each operation.
        self.op_invoke_infos_with_repeat_id: List[tuple[OpInvokeInfo, int]] = []
        # A dictionary to store metadata for each tensor encountered.
        # Key: TensorKey(tensor_id, repeat_id)
        self.tensor_infos: Dict[TensorKey, _TensorInfo] = {}
        # Key: TensorKey, Value: id of tensor owning the buffer
        self.alias_info: Dict[TensorKey, TensorKey] = {}
        # Sets to store the IDs of model input and output tensors.
        self.model_input_tensors: Set[TensorKey] = set()
        self.model_output_tensors: Set[TensorKey] = set()
        # Internal list to store the calculated memory profiles after analysis.
        self.memory_profiles: List[OpMemoryProfile] = []
        # A flag to ensure analysis is done before retrieving the profile.
        self.is_analyzed: bool = False
        self._resident_tensors = {}
        self._weight_storages = set()
        self._resolved_boundaries = None
        self._weight_templates = set()
        self._returned_tensors = []
        self._forward_summary = ForwardMemorySummary()
        self.op_input_tensor_ids: List[Set[TensorKey]] = []
        self.op_output_tensor_ids: List[Set[TensorKey]] = []
        self._alias_plan_cache: Dict[Any, Optional[AliasPlan]] = {}

    def _register_resident(self, tensor, owner):
        if self.is_analyzed:
            raise RuntimeError("Register resident tensors before analysis")
        existing = self._resident_tensors.get(id(tensor))
        if existing is not None and existing[1] != owner:
            raise ValueError("A storage cannot be charged as both weight and cache")
        self._resolved_boundaries = None
        self._resident_tensors[id(tensor)] = (tensor, owner)

    def register_static_weight(self, tensor, *, repeat_template=False):
        """Register an already charged weight; never read its values or add bytes."""
        self._register_resident(tensor, "weight")
        if tensor.layout == torch.strided:
            self._weight_storages.add(tensor.untyped_storage())
        if repeat_template:
            self._weight_templates.add(id(tensor))

    def register_model_memory(self, model, inventory):
        for tensor, repeat_template in model.iter_weight_tensor_registrations():
            self.register_static_weight(tensor, repeat_template=repeat_template)
        for entry in inventory.entries:
            for tensor in entry.tensors:
                self.register_kv_cache(tensor)

    def register_kv_cache(self, tensor):
        self._register_resident(tensor, "cache")

    def register_compiled_weight_aliases(self, tensors):
        """Recognize frozen views of already charged weights, without reading data.

        Storage identity remains meaningful for meta tensors even though their
        data pointers are zero. Match the storage objects themselves so copies
        and unrelated constants remain chargeable. This runs once per graph
        invocation, not per op.
        """
        for tensor in tensors:
            if id(tensor) in self._resident_tensors or tensor.layout != torch.strided:
                continue
            if tensor.untyped_storage() in self._weight_storages:
                self.register_static_weight(tensor)

    def register_compiled_weight_constants(self, tensors):
        """Register graph constants with compiler-proven static-weight provenance."""
        for tensor in tensors:
            registration = self._resident_tensors.get(id(tensor))
            if registration is None:
                self.register_static_weight(tensor)
            elif registration[1] != "weight":
                raise ValueError("A storage cannot be charged as both weight and cache")

    def register_model_outputs(self, outputs):
        if self.is_analyzed:
            raise RuntimeError("Register outputs before analysis")
        self._resolved_boundaries = None
        self._returned_tensors.extend(self._extract_tensors(outputs))

    def get_forward_memory_summary(self):
        if not self.is_analyzed:
            raise RuntimeError("Analyze the trace before reading activation memory")
        return self._forward_summary

    def bind_region_identity_registry(self, identity_registry: RegionIdentityRegistry) -> None:
        """Use the owning Runtime's registry to resolve Region tensor aliases."""
        if identity_registry is None:
            raise ValueError("identity_registry is required")
        self.region_identity_registry = identity_registry

    def _extract_tensors(self, data: Any) -> List[torch.Tensor]:
        """A helper function to recursively find all torch.Tensor objects
        within a nested data structure (e.g., list, tuple, dict).
        """
        tensors = []
        if isinstance(data, torch.Tensor):
            return [data]
        if isinstance(data, (list, tuple, set)):
            for item in data:
                tensors.extend(self._extract_tensors(item))
        elif isinstance(data, dict):
            for value in data.values():
                tensors.extend(self._extract_tensors(value))
        return tensors

    def _get_real_tensor_id(self, tensor, repeat_id) -> TensorKey:
        tensor_id, repeat_id = Region.get_tensor_id(
            tensor,
            repeat_id,
            identity_registry=self.region_identity_registry,
        )
        return TensorKey(tensor_id=tensor_id, repeat_id=repeat_id)

    def _schema_arg_names(self, op_invoke_info: OpInvokeInfo) -> List[str]:
        schema = getattr(op_invoke_info.func, "_schema", None)
        if schema is None:
            return []
        return [arg.name for arg in schema.arguments]

    def _get_schema_input(self, op_invoke_info: OpInvokeInfo, input_spec: ZeroStorageInputSpec) -> Any:
        arg_names = self._schema_arg_names(op_invoke_info)
        if isinstance(input_spec, int):
            if input_spec < 0:
                raise ValueError(f"Zero-storage input spec index must be non-negative, got {input_spec}")
            if input_spec < len(op_invoke_info.args):
                return op_invoke_info.args[input_spec]
            if input_spec < len(arg_names):
                return op_invoke_info.kwargs.get(arg_names[input_spec])
            return None

        if input_spec in op_invoke_info.kwargs:
            return op_invoke_info.kwargs[input_spec]
        if input_spec in arg_names:
            input_index = arg_names.index(input_spec)
            if input_index < len(op_invoke_info.args):
                return op_invoke_info.args[input_index]
        return None

    def _zero_storage_input_ids(self, op_invoke_info: OpInvokeInfo, repeat_id: int) -> Set[TensorKey]:
        input_specs = self._zero_storage_input_specs.get(op_invoke_info.func)
        if not input_specs:
            return set()

        tensor_ids: Set[TensorKey] = set()
        for input_spec in input_specs:
            input_value = self._get_schema_input(op_invoke_info, input_spec)
            for tensor in self._extract_tensors(input_value):
                tensor_ids.add(self._get_real_tensor_id(tensor, repeat_id))
        return tensor_ids

    def record_tensor_alias(
        self,
        source_tensor: torch.Tensor,
        alias_tensor: torch.Tensor,
        repeat_id: int = 0,
    ) -> None:
        """Record that ``alias_tensor`` reuses ``source_tensor``'s storage."""
        source_id = self._get_real_tensor_id(source_tensor, repeat_id)
        alias_id = self._get_real_tensor_id(alias_tensor, repeat_id)
        if source_id == alias_id:
            return

        if source_id not in self.tensor_infos:
            self.tensor_infos[source_id] = _TensorInfo(size_bytes=int(bytes_of_tensor(source_tensor)))

        self.alias_info[alias_id] = self.alias_info.get(source_id, source_id)

    def _handle_aliasing(self, op_invoke_info: OpInvokeInfo, repeat_id: int):
        cached = self._alias_plan_cache.get(op_invoke_info.func, _MISSING)
        if cached is _MISSING:
            schema = op_invoke_info.func._schema
            arg_names = [arg.name for arg in schema.arguments]
            output_plans: List[tuple[OutputKind, int, List[int]]] = []
            for output_idx, output_schema in enumerate(schema.returns):
                if output_schema.alias_info is None:
                    continue

                if isinstance(output_schema.real_type, torch.TensorType):
                    output_alias_set = output_schema.alias_info.before_set
                    input_indices = []
                    for input_idx, input_schema in enumerate(schema.arguments):
                        input_alias = input_schema.alias_info
                        if input_alias is not None and output_alias_set & input_alias.before_set:
                            input_indices.append(input_idx)
                    if input_indices:
                        output_plans.append((OutputKind.TENSOR, output_idx, input_indices))
                elif isinstance(output_schema.real_type, torch.ListType):
                    list_alias_input_idx = None
                    for input_idx, input_schema in enumerate(schema.arguments):
                        input_alias = input_schema.alias_info
                        if input_alias is not None and input_alias.after_set == {"*"}:
                            list_alias_input_idx = input_idx
                            break
                    if list_alias_input_idx is not None:
                        output_plans.append((OutputKind.LIST, output_idx, [list_alias_input_idx]))
                else:
                    logger.warning(
                        "MemoryTracker: unsupported alias output type %s for op %s "
                        "(output index %d); alias tracking is skipped for this output",
                        type(output_schema.real_type).__name__,
                        op_invoke_info.func,
                        output_idx,
                    )

            plan = (arg_names, output_plans) if output_plans else None
            self._alias_plan_cache[op_invoke_info.func] = plan
        else:
            plan = cast(Optional[AliasPlan], cached)

        if not plan:
            return

        arg_names, output_plans = plan

        def get_input_id(index):
            if index < len(op_invoke_info.args):
                input_tensor = op_invoke_info.args[index]
            else:
                input_name = arg_names[index]
                input_tensor = op_invoke_info.kwargs.get(input_name)
            if not isinstance(input_tensor, torch.Tensor):
                return None
            return self._get_real_tensor_id(input_tensor, repeat_id)

        def set_alias(input_id, output_id):
            if input_id == output_id:
                return
            aliased_tensor_id = self.alias_info.get(input_id)
            if aliased_tensor_id is not None:
                self.alias_info[output_id] = aliased_tensor_id
            else:
                self.alias_info[output_id] = input_id

        outputs = op_invoke_info.out if isinstance(op_invoke_info.out, tuple) else [op_invoke_info.out]
        for output_kind, output_idx, input_indices in output_plans:
            output = outputs[output_idx]
            if output_kind is OutputKind.TENSOR:
                if not isinstance(output, torch.Tensor):
                    continue
                output_id = self._get_real_tensor_id(output, repeat_id)
                for input_idx in input_indices:
                    input_id = get_input_id(input_idx)
                    if input_id is None:
                        continue
                    set_alias(input_id, output_id)
                    break
            elif output_kind is OutputKind.LIST:
                if not isinstance(output, list):
                    continue
                input_id = get_input_id(input_indices[0])
                if input_id is None:
                    continue
                for output_tensor in output:
                    if not isinstance(output_tensor, torch.Tensor):
                        continue
                    output_id = self._get_real_tensor_id(output_tensor, repeat_id)
                    set_alias(input_id, output_id)
            else:
                raise RuntimeError(f"Unsupported output kind: {output_kind!r}")

    def record_op_invocation(self, op_info_or_region):
        """
        Record the memory usage of an op invocation. Client code calls this method
        multiple times for ops executed by the PyTorch program.
        """
        if isinstance(op_info_or_region, Region):
            region = op_info_or_region
            for op_invoke_info in region.op_invoke_infos:
                self.record_single_op_invocation(op_invoke_info, region.reference_id)
        elif isinstance(op_info_or_region, OpInvokeInfo):
            op_invoke_info = op_info_or_region
            self.record_single_op_invocation(op_invoke_info, 0)
        else:
            raise ValueError(f"record_op_invocation failed: Unsupported type: {type(op_info_or_region)}")

    def record_single_op_invocation(self, op_invoke_info: OpInvokeInfo, repeat_id: int = 0):
        """
        Record the memory usage of an op invocation. Client code calls this method
        multiple times for ops executed by the PyTorch program.
        """
        op_idx = len(self.op_invoke_infos_with_repeat_id)
        self.op_invoke_infos_with_repeat_id.append((op_invoke_info, repeat_id))
        zero_storage_tensor_ids = self._zero_storage_input_ids(op_invoke_info, repeat_id)

        # Identify all input tensors and record their usage.
        input_tensors = self._extract_tensors(op_invoke_info.args)
        if op_invoke_info.kwargs:
            input_tensors.extend(self._extract_tensors(op_invoke_info.kwargs))
        input_tensor_ids: Set[TensorKey] = set()
        for tensor in input_tensors:
            tensor_id = self._get_real_tensor_id(tensor, repeat_id)
            input_tensor_ids.add(tensor_id)
            if tensor_id not in self.tensor_infos:
                # If a tensor is used before it's defined, it's a model input.
                # We initialize its info here.
                self.tensor_infos[tensor_id] = _TensorInfo(
                    size_bytes=int(bytes_of_tensor(tensor)),
                    storage_ptr=_storage_ptr(tensor),
                    base_id=_base_root_id(tensor),
                    obj_id=id(tensor),
                )
            if tensor_id in zero_storage_tensor_ids:
                self.tensor_infos[tensor_id].size_bytes = 0
            self.tensor_infos[tensor_id].use_op_indices.append(op_idx)
            aliased_tensor_id = self.alias_info.get(tensor_id)
            if aliased_tensor_id is not None:
                self.tensor_infos[aliased_tensor_id].use_op_indices_by_alias.append(op_idx)

        # Identify all output tensors and record their definition site.
        output_tensors = self._extract_tensors(op_invoke_info.out)
        output_tensor_ids: Set[TensorKey] = set()
        for tensor in output_tensors:
            tensor_id = self._get_real_tensor_id(tensor, repeat_id)
            output_tensor_ids.add(tensor_id)
            if tensor_id not in self.tensor_infos:
                self.tensor_infos[tensor_id] = _TensorInfo(size_bytes=int(bytes_of_tensor(tensor)))
                # This op is the one that defines (creates) the tensor.
                self.tensor_infos[tensor_id].def_op_idx = op_idx

        self._handle_aliasing(op_invoke_info, repeat_id)
        self.op_input_tensor_ids.append(input_tensor_ids)
        self.op_output_tensor_ids.append(output_tensor_ids)

    def resolve_memory_boundaries(self):
        """Resolve identities while Region mappings are available (also for frozen replay)."""
        # Region copies virtualize internal outputs, but an external tensor
        # without a definition in that copy still refers to its original storage.
        for key, info in self.tensor_infos.items():
            original = TensorKey(key.tensor_id, 0)
            if key.repeat_id and info.def_op_idx == -1 and original in self.tensor_infos:
                self.alias_info.setdefault(key, original)
        for key in self.alias_info:
            root = self.alias_info[key]
            while root in self.alias_info:
                root = self.alias_info[root]
            self.alias_info[key] = root
        owners = {}
        for tensor, owner in self._resident_tensors.values():
            key = self._get_real_tensor_id(tensor, 0)
            root = self.alias_info.get(key, key)
            if owners.setdefault(root, owner) != owner:
                raise ValueError("A storage cannot be charged as both weight and cache")
        resident = set(owners)
        for key in self.tensor_infos:
            registration = self._resident_tensors.get(key.tensor_id)
            # A shared static tensor can also be read inside a repeated region
            # without belonging to its representative module subtree.
            if registration is not None:
                resident.add(self.alias_info.get(key, key))
        returned_keys = (self._get_real_tensor_id(t, 0) for t in self._returned_tensors)
        returned = {self.alias_info.get(key, key) for key in returned_keys}
        self._resolved_boundaries = (resident, returned)
        return self._resolved_boundaries

    def analyze(self):
        """
        Analyze the memory usage of the executed PyTorch program by simulating
        the allocation and deallocation of tensors based on their lifecycles.
        """
        if self.is_analyzed:
            return
        if not self.op_invoke_infos_with_repeat_id:
            self.is_analyzed = True
            return

        resident, returned = self._resolved_boundaries or self.resolve_memory_boundaries()
        # Include uses of external aliases introduced by Region normalization.
        for alias, root in self.alias_info.items():
            info = self.tensor_infos.get(alias)
            if info is not None:
                self.tensor_infos[root].use_op_indices_by_alias.extend(info.use_op_indices)

        # Step 1: Finalize tensor lifecycle info (inputs, outputs, last use).
        for tensor_id, info in self.tensor_infos.items():
            all_use_op_indices = info.use_op_indices + info.use_op_indices_by_alias
            if not all_use_op_indices:
                if tensor_id in self.alias_info:
                    # An output alias keeps its root buffer alive until model completion.
                    self.model_output_tensors.add(self.alias_info[tensor_id])
                else:
                    self.model_output_tensors.add(tensor_id)
            else:
                info.last_use_op_idx = max(all_use_op_indices)

            if info.def_op_idx == -1:
                if tensor_id in self.alias_info:
                    source_id = self.alias_info[tensor_id]
                    source_info = self.tensor_infos.get(source_id)
                    if source_info is None or source_info.def_op_idx == -1:
                        self.model_input_tensors.add(source_id)
                else:
                    self.model_input_tensors.add(tensor_id)

        # Step 2: Simulate memory usage over the sequence of operations.
        # Start with memory consumed by model inputs, which are pre-allocated.
        # Region replay can expose the same external resident storage through
        # several repeat keys. Count each physical tensor identity once in the
        # legacy graph working-set profile.
        counted_resident_tensor_ids: Set[int] = set()
        initial_usage = 0
        for t_id in self.model_input_tensors:
            size = self.tensor_infos[t_id].size_bytes
            if t_id in resident:
                if t_id.tensor_id in counted_resident_tensor_ids:
                    continue
                counted_resident_tensor_ids.add(t_id.tensor_id)
            initial_usage += size
        current_memory_usage = initial_usage
        freed_roots: Set[TensorKey] = set()

        # Region replay gives the same external request tensor multiple keys.
        # It remains one allocation; only graph-created repeat outputs are distinct.
        request_sizes = {}
        for key in self.model_input_tensors - resident:
            request_sizes[key.tensor_id] = self.tensor_infos[key].size_bytes
        request_bytes = sum(request_sizes.values())
        activation = peak = 0
        for op_idx, (op_info, _) in enumerate(self.op_invoke_infos_with_repeat_id):
            usage_before_call = current_memory_usage

            # Allocate storage only where a TensorKey is first defined. In-place and
            # custom mutating ops can return an existing input TensorKey.
            output_tensor_ids = self.op_output_tensor_ids[op_idx]

            mem_allocated = sum(
                self.tensor_infos[t_id].size_bytes
                for t_id in output_tensor_ids
                if t_id not in self.alias_info and self.tensor_infos[t_id].def_op_idx == op_idx
            )

            # The memory usage after the call includes the newly allocated tensors.
            # TODO(jgong5): This model does not account for temporary workspace memory, which would
            #               require a more sophisticated model to estimate the workspace usage.
            usage_after_call = usage_before_call + mem_allocated

            self.memory_profiles.append(OpMemoryProfile(op_info, usage_before_call, usage_after_call))

            new_roots = {
                k
                for k in output_tensor_ids
                if k not in self.alias_info and k not in resident and self.tensor_infos[k].def_op_idx == op_idx
            }
            activation += sum(self.tensor_infos[k].size_bytes for k in new_roots)
            if activation > peak:
                peak = activation

            # Update memory state for the beginning of the next operation.
            current_memory_usage = usage_after_call

            # Free tensors whose lifecycle ends after this operation.
            input_tensor_ids = self.op_input_tensor_ids[op_idx]
            mem_freed = 0
            input_root_ids = {self.alias_info.get(t_id, t_id) for t_id in input_tensor_ids}
            for key in (input_root_ids | new_roots) - resident - returned:
                info = self.tensor_infos[key]
                if info.def_op_idx >= 0 and max(info.last_use_op_idx, info.def_op_idx) == op_idx:
                    activation -= info.size_bytes
            for root_id in input_root_ids:
                if root_id in freed_roots:
                    continue
                root_info = self.tensor_infos[root_id]
                # A root buffer is freed once, after the last use of the root or any of its aliases.
                if (
                    root_info.last_use_op_idx == op_idx
                    and root_id not in self.model_input_tensors
                    and root_id not in self.model_output_tensors
                ):
                    mem_freed += root_info.size_bytes
                    freed_roots.add(root_id)

            # Note: we do not count in the memory fragmentation here.
            current_memory_usage -= mem_freed

        # memory usage after model execution, all intermediate tensors should be freed
        self.memory_profiles.append(OpMemoryProfile(None, current_memory_usage, current_memory_usage))
        self._forward_summary = ForwardMemorySummary(request_bytes, peak)

        self.is_analyzed = True

    def _ensure_analyzed(self, caller: str) -> None:
        """Shared pre-condition guard for all post-``analyze()`` accessors."""
        if not self.is_analyzed:
            raise RuntimeError(f"`analyze()` must be called before `{caller}`.")

    def get_profile(self, initial_mem_usage_bytes: float = 0.0) -> List[OpMemoryProfile]:
        """
        Return the memory profile of the recorded PyTorch program.

        Args:
            initial_mem_usage_bytes: memory usage before executing the PyTorch program.
        """
        self._ensure_analyzed("get_profile()")

        # Adjust the calculated profile with the initial memory offset.
        if initial_mem_usage_bytes == 0:
            return self.memory_profiles
        else:
            return [
                OpMemoryProfile(
                    op_invoke_info=profile.op_invoke_info,
                    usage_before_call_bytes=profile.usage_before_call_bytes + initial_mem_usage_bytes,
                    usage_after_call_bytes=profile.usage_after_call_bytes + initial_mem_usage_bytes,
                )
                for profile in self.memory_profiles
            ]

    def peak_mem_usage(self, initial_mem_usage_bytes: float = 0.0):
        """Return the high-water mark of device memory during execution.

        ``analyze()`` records ``usage_after_call`` after allocating this op's
        outputs and before freeing last-use inputs. The peak is therefore
        ``max_i max(U_i, P_i)``, not ``max_i U_i``.
        """
        profiles = self.get_profile(initial_mem_usage_bytes)
        return max(
            (max(p.usage_before_call_bytes, p.usage_after_call_bytes) for p in profiles),
            default=initial_mem_usage_bytes,
        )
