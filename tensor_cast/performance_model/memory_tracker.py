import dataclasses
import logging
from enum import auto, Enum
from typing import Any, cast, Dict, List, NamedTuple, Optional, Set

import torch

from ..device import DeviceProfile
from ..performance_model import OpInvokeInfo
from ..performance_model.op_invoke_info import Region
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

    def __init__(self, device_profile: DeviceProfile):
        self.device_profile = device_profile
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

    def _extract_tensors(self, data: Any) -> List[torch.Tensor]:
        """A helper function to recursively find all torch.Tensor objects
        within a nested data structure (e.g., list, tuple, dict)."""
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
        tensor_id, repeat_id = Region.get_tensor_id(tensor, repeat_id)
        return TensorKey(tensor_id=tensor_id, repeat_id=repeat_id)

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
            self.tensor_infos[source_id] = _TensorInfo(
                size_bytes=int(bytes_of_tensor(source_tensor))
            )

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
                        if (
                            input_alias is not None
                            and output_alias_set & input_alias.before_set
                        ):
                            input_indices.append(input_idx)
                    if input_indices:
                        output_plans.append(
                            (OutputKind.TENSOR, output_idx, input_indices)
                        )
                elif isinstance(output_schema.real_type, torch.ListType):
                    list_alias_input_idx = None
                    for input_idx, input_schema in enumerate(schema.arguments):
                        input_alias = input_schema.alias_info
                        if input_alias is not None and input_alias.after_set == {"*"}:
                            list_alias_input_idx = input_idx
                            break
                    if list_alias_input_idx is not None:
                        output_plans.append(
                            (OutputKind.LIST, output_idx, [list_alias_input_idx])
                        )
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

        outputs = (
            op_invoke_info.out
            if isinstance(op_invoke_info.out, tuple)
            else [op_invoke_info.out]
        )
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
                    aliased_tensor_id = self.alias_info.get(input_id)
                    if aliased_tensor_id is not None:
                        self.alias_info[output_id] = aliased_tensor_id
                    else:
                        self.alias_info[output_id] = input_id
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
            raise ValueError(
                f"record_op_invocation failed: Unsupported type: {type(op_info_or_region)}"
            )

    def record_single_op_invocation(
        self, op_invoke_info: OpInvokeInfo, repeat_id: int = 0
    ):
        """
        Record the memory usage of an op invocation. Client code calls this method
        multiple times for ops executed by the PyTorch program.
        """
        op_idx = len(self.op_invoke_infos_with_repeat_id)
        self.op_invoke_infos_with_repeat_id.append((op_invoke_info, repeat_id))

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
                    size_bytes=int(bytes_of_tensor(tensor))
                )
            self.tensor_infos[tensor_id].use_op_indices.append(op_idx)
            aliased_tensor_id = self.alias_info.get(tensor_id)
            if aliased_tensor_id is not None:
                self.tensor_infos[aliased_tensor_id].use_op_indices_by_alias.append(
                    op_idx
                )

        # Identify all output tensors and record their definition site.
        output_tensors = self._extract_tensors(op_invoke_info.out)
        output_tensor_ids: Set[TensorKey] = set()
        for tensor in output_tensors:
            tensor_id = self._get_real_tensor_id(tensor, repeat_id)
            output_tensor_ids.add(tensor_id)
            if tensor_id not in self.tensor_infos:
                self.tensor_infos[tensor_id] = _TensorInfo(
                    size_bytes=int(bytes_of_tensor(tensor))
                )
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
                    # we treat the aliased tensor as output, not the aliasing ones
                    source_id = self.alias_info[tensor_id]
                    source_info = self.tensor_infos.get(source_id)
                    source_use_op_indices = (
                        []
                        if source_info is None
                        else source_info.use_op_indices
                        + source_info.use_op_indices_by_alias
                    )
                    if not source_use_op_indices or info.def_op_idx >= max(
                        source_use_op_indices
                    ):
                        self.model_output_tensors.add(source_id)
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
        current_memory_usage = sum(
            self.tensor_infos[t_id].size_bytes for t_id in self.model_input_tensors
        )

        for op_idx, (op_info, _) in enumerate(self.op_invoke_infos_with_repeat_id):
            usage_before_call = current_memory_usage

            # Calculate memory allocated for new output tensors. We don't allocate for aliases.
            output_tensor_ids = self.op_output_tensor_ids[op_idx]

            mem_allocated = sum(
                self.tensor_infos[t_id].size_bytes
                for t_id in output_tensor_ids
                if t_id not in self.alias_info
            )

            # The memory usage after the call includes the newly allocated tensors.
            # TODO(jgong5): This model does not account for temporary workspace memory, which would
            #               require a more sophisticated model to estimate the workspace usage.
            usage_after_call = usage_before_call + mem_allocated

            self.memory_profiles.append(
                OpMemoryProfile(op_info, usage_before_call, usage_after_call)
            )

            # Update memory state for the beginning of the next operation.
            current_memory_usage = usage_after_call

            # Free tensors whose lifecycle ends after this operation.
            input_tensor_ids = self.op_input_tensor_ids[op_idx]
            mem_freed = 0
            for t_id in input_tensor_ids:
                info = self.tensor_infos[t_id]
                # A tensor is freed if this op is its last use and it's not a model input/output.
                t_id = self.alias_info[t_id] if t_id in self.alias_info else t_id
                if (
                    info.last_use_op_idx == op_idx
                    and t_id not in self.model_input_tensors
                    and t_id not in self.model_output_tensors
                ):
                    mem_freed += info.size_bytes

            # Note: we do not count in the memory fragmentation here.
            current_memory_usage -= mem_freed

        # memory usage after model execution, all intermediate tensors should be freed
        self.memory_profiles.append(
            OpMemoryProfile(None, current_memory_usage, current_memory_usage)
        )

        self.is_analyzed = True

    def get_profile(
        self, initial_mem_usage_bytes: float = 0.0
    ) -> List[OpMemoryProfile]:
        """
        Return the memory profile of the recorded PyTorch program.

        Args:
            initial_mem_usage_bytes: memory usage before executing the PyTorch program.
        """
        if not self.is_analyzed:
            raise RuntimeError("`analyze()` must be called before `get_profile()`.")

        # Adjust the calculated profile with the initial memory offset.
        if initial_mem_usage_bytes == 0:
            return self.memory_profiles
        else:
            return [
                OpMemoryProfile(
                    op_invoke_info=profile.op_invoke_info,
                    usage_before_call_bytes=profile.usage_before_call_bytes
                    + initial_mem_usage_bytes,
                    usage_after_call_bytes=profile.usage_after_call_bytes
                    + initial_mem_usage_bytes,
                )
                for profile in self.memory_profiles
            ]

    def peak_mem_usage(self, initial_mem_usage_bytes: float = 0.0):
        return max(
            [mem_profile.usage_before_call_bytes for mem_profile in self.get_profile()]
        )
