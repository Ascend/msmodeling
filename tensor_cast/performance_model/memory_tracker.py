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
        # Cached input/output tensor ids per op index, used by analyze() to avoid
        # re-extracting tensors from op arguments.
        # Key: op index in self.op_invoke_infos_with_repeat_id
        # Value: set of TensorKey seen in that op's inputs/outputs.
        self.op_input_tensor_ids: List[Set[TensorKey]] = []
        self.op_output_tensor_ids: List[Set[TensorKey]] = []
        # Cached alias matching plan per op schema for faster alias processing.
        # Key: torch op handle (op_invoke_info.func)
        # Value: AliasPlan if op has alias-bearing outputs; otherwise None.
        self._alias_plan_cache: Dict[Any, Optional[AliasPlan]] = {}
        # --- Three-account classification (Issue #463) ---
        # Registered model param/buffer tensor_ids, populated by
        # ``register_model_params`` before ``model.forward()``.
        self._registered_param_tensor_ids: Set = set()
        # Storage-level keys of registered params/buffers: repacked or viewed
        # weight copies consumed by ops share storage (or a ``_base`` root)
        # with a registered tensor even when their tensor_id differs.
        self._registered_param_storage_ptrs: Set[int] = set()
        self._registered_param_base_ids: Set[int] = set()
        # Object-identity sets handed over from constant folding (see
        # ``register_folded_constants``): folded ``_frozen_param*`` constants
        # are weight bytes created at compile time (never registered as
        # params), and folded sources are originals fully replaced by them.
        self._folded_constant_obj_ids: Set[int] = set()
        self._folded_source_obj_ids: Set[int] = set()
        # Category map: TensorKey (root) -> "param" | "derived".
        # Populated during ``analyze()`` after step 1.  Tensors not in this
        # map are classified as "intermediate".
        self._category_map: Dict[TensorKey, str] = {}
        # Peak per-category bytes, computed during ``analyze()`` step 2.
        self._peak_param_bytes: int = 0
        self._peak_derived_resident_bytes: int = 0
        self._peak_intermediate_bytes: int = 0

    def register_model_params(self, model) -> None:
        """Register model parameters/buffers for three-account classification.

        Must be called **before** ``model.forward()`` so that tensor IDs are
        captured before the ``EquivalentKeyManager`` remaps them during
        representative-layer (``RegionMarkerWrapper``) expansion.

        After this call, ``analyze()`` will classify every tensor in the trace
        into one of three accounts:

        * **param_matched** — model inputs whose ``tensor_id`` matches a
          registered param/buffer.  These correspond to weights/buffers that
          are actually consumed by ops in the trace.
        * **derived_resident** — model inputs that do *not* match any
          registered param/buffer (e.g. KV-cache tensors, runtime-materialised
          look-up tables).
        * **intermediate** — tensors defined by an op (``def_op_idx >= 0``).
          These are the true activation tensors.

        The split fixes the set-mismatch root cause of Issue #463 where
        ``peak`` (traced tensors) and ``weight_size`` (all registered params)
        used different collection scopes.
        """
        for _, param in model.named_parameters():
            tensor_id = self._get_real_tensor_id(param, 0).tensor_id
            self._registered_param_tensor_ids.add(tensor_id)
            ptr = _storage_ptr(param)
            if ptr:
                self._registered_param_storage_ptrs.add(ptr)
            self._registered_param_base_ids.add(id(param))
            self._registered_param_base_ids.add(_base_root_id(param))
        for _, buffer in model.named_buffers():
            tensor_id = self._get_real_tensor_id(buffer, 0).tensor_id
            self._registered_param_tensor_ids.add(tensor_id)
            ptr = _storage_ptr(buffer)
            if ptr:
                self._registered_param_storage_ptrs.add(ptr)
            self._registered_param_base_ids.add(id(buffer))
            self._registered_param_base_ids.add(_base_root_id(buffer))

    def register_folded_constants(
        self,
        constant_obj_ids: Set[int],
        source_obj_ids: Set[int],
    ) -> None:
        """Hand over constant-folding identity sets for three-account matching.

        Called before executing each compiled graph. See
        ``compilation.constant_folding`` for how the sets are populated.

        * ``constant_obj_ids``: folded ``_frozen_param*`` constant tensors.
          They carry real weight bytes consumed by ops but were created at
          compile time, so neither tensor_id nor storage/base matching can
          link them to registered params; match by object identity instead.
        * ``source_obj_ids``: original params/buffers whose graph references
          were entirely replaced by folded constants.  They never appear in
          the trace, so ``unexecuted_param_bytes`` must skip them — the folded
          replacement is the resident copy the ops actually consume.
        """
        self._folded_constant_obj_ids.update(constant_obj_ids)
        self._folded_source_obj_ids.update(source_obj_ids)

    def _matches_registered_param(self, t_key: TensorKey) -> bool:
        """Four-way match of a traced tensor against registered params/buffers.

        Quantizers and model wrappers often feed ops a repacked or viewed copy
        of a weight whose tensor_id differs from the registered param, which a
        plain tensor_id match misses (the copy then lands in derived_resident
        and the weight is double counted).  A copy shares the untyped-storage
        ``data_ptr`` or the ``_base`` view-chain root with its registered
        original, so match on those keys as well.  Storage matching no-ops for
        meta tensors (``data_ptr`` == 0).  Compile-time folded constants match
        by object identity: they are new weight tensors with no registered
        counterpart and no shared storage.
        """
        if t_key.tensor_id in self._registered_param_tensor_ids:
            return True
        info = self.tensor_infos.get(t_key)
        if info is None:
            return False
        if info.obj_id and info.obj_id in self._folded_constant_obj_ids:
            return True
        if info.storage_ptr and info.storage_ptr in self._registered_param_storage_ptrs:
            return True
        return bool(info.base_id) and info.base_id in self._registered_param_base_ids

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

    def analyze(self):
        """
        Analyze the memory usage of the executed PyTorch program by simulating
        the allocation and deallocation of tensors based on their lifecycles.
        """
        if not self.op_invoke_infos_with_repeat_id:
            self.is_analyzed = True
            return

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

        # Step 1.5: Classify model input tensors into three accounts.
        #
        # ``model_input_tensors`` are external inputs (no def_op_idx).  Split
        # them into *param_matched* (matches a registered model param/buffer)
        # and *derived_resident* (KV-cache, runtime lookup tables, etc.).
        # Everything else (tensors with def_op_idx >= 0) is *intermediate*.
        self._category_map = {}
        for t_key in self.model_input_tensors:
            if self._matches_registered_param(t_key):
                self._category_map[t_key] = "param"
            else:
                self._category_map[t_key] = "derived"

        # Step 2: Simulate memory usage over the sequence of operations.
        # Start with memory consumed by model inputs, which are pre-allocated.
        # Region replay reuses the SAME representative weight storage across
        # repeats, so each physical weight tensor (distinct tensor_id) is
        # counted once here; the remaining layers are covered by the
        # represented-weight account in ModelRunner.  Derived-resident tensors
        # (KV cache etc.) keep per-repeat multiplicity because each repeat
        # simulates a distinct layer's resident state.
        counted_param_tensor_ids: Set[int] = set()
        initial_usage = 0
        current_param = 0
        current_derived = 0
        for t_id in self.model_input_tensors:
            size = self.tensor_infos[t_id].size_bytes
            if self._category_map.get(t_id) == "param":
                if t_id.tensor_id in counted_param_tensor_ids:
                    continue
                counted_param_tensor_ids.add(t_id.tensor_id)
                current_param += size
            else:
                current_derived += size
            initial_usage += size
        current_memory_usage = initial_usage
        freed_roots: Set[TensorKey] = set()

        # Per-category live bytes tracking (Issue #463).
        current_intermediate = 0
        peak_param = current_param
        peak_derived = current_derived
        peak_intermediate = current_intermediate

        for op_idx, (op_info, _) in enumerate(self.op_invoke_infos_with_repeat_id):
            usage_before_call = current_memory_usage

            # Peak check at the "before" point (matches peak_mem_usage semantics).
            peak_param = max(peak_param, current_param)
            peak_derived = max(peak_derived, current_derived)
            peak_intermediate = max(peak_intermediate, current_intermediate)

            # Allocate storage only where a TensorKey is first defined. In-place and
            # custom mutating ops can return an existing input TensorKey.
            output_tensor_ids = self.op_output_tensor_ids[op_idx]

            mem_allocated = sum(
                self.tensor_infos[t_id].size_bytes
                for t_id in output_tensor_ids
                if t_id not in self.alias_info and self.tensor_infos[t_id].def_op_idx == op_idx
            )

            # Track per-category allocation for newly defined tensors.
            for t_id in output_tensor_ids:
                if t_id in self.alias_info or self.tensor_infos[t_id].def_op_idx != op_idx:
                    continue
                size = self.tensor_infos[t_id].size_bytes
                root_id = self.alias_info.get(t_id, t_id)
                cat = self._category_map.get(root_id, "intermediate")
                if cat == "param":
                    current_param += size
                elif cat == "derived":
                    current_derived += size
                else:
                    current_intermediate += size

            # The memory usage after the call includes the newly allocated tensors.
            # TODO(jgong5): This model does not account for temporary workspace memory, which would
            #               require a more sophisticated model to estimate the workspace usage.
            usage_after_call = usage_before_call + mem_allocated

            # Peak check at the "after" point (after allocation, before freeing).
            peak_param = max(peak_param, current_param)
            peak_derived = max(peak_derived, current_derived)
            peak_intermediate = max(peak_intermediate, current_intermediate)

            self.memory_profiles.append(OpMemoryProfile(op_info, usage_before_call, usage_after_call))

            # Update memory state for the beginning of the next operation.
            current_memory_usage = usage_after_call

            # Free tensors whose lifecycle ends after this operation.
            input_tensor_ids = self.op_input_tensor_ids[op_idx]
            mem_freed = 0
            input_root_ids = {self.alias_info.get(t_id, t_id) for t_id in input_tensor_ids}
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
                    # Track per-category freeing (only intermediates can be freed;
                    # model inputs and outputs are never freed).
                    cat = self._category_map.get(root_id, "intermediate")
                    if cat == "param":
                        current_param -= root_info.size_bytes
                    elif cat == "derived":
                        current_derived -= root_info.size_bytes
                    else:
                        current_intermediate -= root_info.size_bytes

            # Note: we do not count in the memory fragmentation here.
            current_memory_usage -= mem_freed

        # memory usage after model execution, all intermediate tensors should be freed
        self.memory_profiles.append(OpMemoryProfile(None, current_memory_usage, current_memory_usage))

        # Final peak check (post-execution state).
        peak_param = max(peak_param, current_param)
        peak_derived = max(peak_derived, current_derived)
        peak_intermediate = max(peak_intermediate, current_intermediate)

        self._peak_param_bytes = peak_param
        self._peak_derived_resident_bytes = peak_derived
        self._peak_intermediate_bytes = peak_intermediate

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
        return max(max(p.usage_before_call_bytes, p.usage_after_call_bytes) for p in profiles)

    # ------------------------------------------------------------------
    # Three-account peak accessors (Issue #463)
    # ------------------------------------------------------------------
    #
    # ``peak_mem_usage()`` returns the *total* high-water mark.  The three
    # accessors below decompose that total into three disjoint accounts:
    #
    #   peak = peak_param_bytes + peak_derived_resident_bytes + peak_intermediate_bytes
    #
    # (KV-cache tensors that are *model inputs* fall into ``derived_resident``;
    # KV-cache tensors allocated *inside* the trace fall into ``intermediate``.)

    def peak_param_bytes(self) -> int:
        """Peak bytes of registered params/buffers actually consumed in the trace."""
        self._ensure_analyzed("peak_param_bytes()")
        return self._peak_param_bytes

    def peak_derived_resident_bytes(self) -> int:
        """Peak bytes of non-registered model inputs (KV cache, look-up tables)."""
        self._ensure_analyzed("peak_derived_resident_bytes()")
        return self._peak_derived_resident_bytes

    def peak_intermediate_bytes(self) -> int:
        """Peak bytes of execution-graph intermediates — the true activation."""
        self._ensure_analyzed("peak_intermediate_bytes()")
        return self._peak_intermediate_bytes

    # ------------------------------------------------------------------
    # Fourth account (Issue #463 follow-up)
    # ------------------------------------------------------------------
    def unexecuted_param_bytes(self, named_tensors) -> int:
        """Bytes of registered params/buffers never touched by any traced op.

        Fourth account complementing the three-account peak: tensors that
        hang on the module tree (so ``weight_size`` counts them) but that no
        traced op ever consumed (so the traced ``peak`` misses them).  In a
        real deployment they stay resident in device memory for the whole
        run — e.g. an ``lm_head`` a decode simulation never samples with,
        shared experts whose forward is bypassed, draft-model heads under
        speculative decoding, rotary caches — so capacity planning must count
        them as resident instead of treating ``total - traced_peak`` as free.

        Weights consumed by a traced op (including originals read while the
        quantizer materializes its replacement copies) appear in
        ``tensor_infos`` and are excluded here.  Tensors that never appeared
        in the trace (``tensor_id`` absent from ``tensor_infos``) are the
        unexecuted-resident set.

        Args:
            named_tensors: iterable of ``(name, tensor)`` pairs, typically
                ``list(model.named_parameters()) + list(model.named_buffers())``.
                Callers may pre-filter (e.g. skip the vision tower for
                text-only runs to stay consistent with the weight account).

        Returns:
            Total bytes of the unexecuted set.
        """
        self._ensure_analyzed("unexecuted_param_bytes()")
        traced_tensor_ids = {key.tensor_id for key in self.tensor_infos}
        # Symmetric with ``_matches_registered_param``: a param consumed through
        # a repacked/viewed copy leaves its storage (or ``_base`` root) in the
        # trace even when its own tensor_id never appears.
        traced_storage_ptrs = {info.storage_ptr for info in self.tensor_infos.values() if info.storage_ptr}
        traced_base_ids = {info.base_id for info in self.tensor_infos.values() if info.base_id}
        total = 0
        for _, tensor in named_tensors:
            tensor_id = self._get_real_tensor_id(tensor, 0).tensor_id
            if tensor_id in traced_tensor_ids:
                continue
            if id(tensor) in self._folded_source_obj_ids:
                # Fully replaced by a folded constant that is consumed (and
                # counted as param) in its place.
                continue
            ptr = _storage_ptr(tensor)
            if ptr and ptr in traced_storage_ptrs:
                continue
            if id(tensor) in traced_base_ids or _base_root_id(tensor) in traced_base_ids:
                continue
            total += int(bytes_of_tensor(tensor))
        return total
