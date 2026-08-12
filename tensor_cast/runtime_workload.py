# Copyright (c) 2026 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""Portable, device-independent Runtime workload traces.

The trace intentionally contains tensor *metadata* only.  It can therefore be
sent to another process without retaining FakeTensor storage, a Runtime, or a
DeviceProfile.  Unsupported values fail closed through ``WorkloadFreezeError``
so callers can use the existing direct simulation path instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from tensor_cast.performance_model.op_invoke_info import OpInvokeInfo, Region


class WorkloadFreezeError(ValueError):
    """Raised when a recorded Runtime cannot be safely represented as an IR."""


@dataclass(frozen=True)
class FrozenTensor:
    """Metadata required to recreate a meta tensor for performance estimation."""

    tensor_id: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: str
    requires_grad: bool


@dataclass(frozen=True)
class FrozenOpInvoke:
    """One flattened op invocation and its logical region reference id."""

    schema_name: str
    overload_name: str
    args: Any
    kwargs: Any
    out: Any
    cache_key: str
    reference_id: int


@dataclass(frozen=True)
class FrozenRegionAlias:
    """Logical-to-physical tensor mapping used by a repeated Runtime region."""

    reference_id: int
    logical_input: Any
    logical_output: Any
    real_input: Any
    real_output: Any


@dataclass(frozen=True)
class RuntimeWorkloadTrace:
    """A pickle-safe runtime trace that can be replayed on another profile."""

    tensors: tuple[FrozenTensor, ...]
    invocations: tuple[FrozenOpInvoke, ...]
    region_aliases: tuple[FrozenRegionAlias, ...] = ()

    @classmethod
    def from_runtime(cls, runtime) -> "RuntimeWorkloadTrace":
        """Freeze a completed Runtime without retaining any live tensor objects."""
        freezer = _ValueFreezer()
        region_aliases = []
        for group in runtime.op_info_group:
            if not isinstance(group, Region):
                continue
            if group.real_input_tensor is None or group.real_output_tensor is None:
                raise WorkloadFreezeError("unfinalized runtime region")
            region_aliases.append(
                FrozenRegionAlias(
                    reference_id=group.reference_id,
                    logical_input=freezer.freeze(group.input_tensor),
                    logical_output=freezer.freeze(group.output_tensor),
                    real_input=freezer.freeze(group.real_input_tensor),
                    real_output=freezer.freeze(group.real_output_tensor),
                )
            )
        frozen_ops = []
        for op_invoke_info, reference_id in runtime._iter_flat_invocations():
            schema = getattr(op_invoke_info.func, "_schema", None)
            if schema is None:
                raise WorkloadFreezeError(f"op has no dispatcher schema: {op_invoke_info.func!r}")
            frozen_ops.append(
                FrozenOpInvoke(
                    schema_name=str(schema.name),
                    overload_name=str(schema.overload_name),
                    args=freezer.freeze(op_invoke_info.args),
                    kwargs=freezer.freeze(op_invoke_info.kwargs),
                    out=freezer.freeze(op_invoke_info.out),
                    cache_key=op_invoke_info.cache_key,
                    reference_id=reference_id,
                )
            )
        return cls(
            tensors=tuple(freezer.tensors.values()),
            invocations=tuple(frozen_ops),
            region_aliases=tuple(region_aliases),
        )

    def replay(self, runtime) -> None:
        """Rebuild the op inputs and replay them through an existing Runtime."""
        thawer = _ValueThawer(self.tensors)
        # Runtime records tensor ids while replaying. Scope the aliases to this
        # operation so transient meta-tensor ids cannot affect a later workload
        # handled by the same worker process.
        with Region.equivalent_tensor_id_manager.scoped_aliases():
            for alias in self.region_aliases:
                logical_input = thawer.thaw(alias.logical_input)
                logical_output = thawer.thaw(alias.logical_output)
                real_input = thawer.thaw(alias.real_input)
                real_output = thawer.thaw(alias.real_output)
                if not all(
                    isinstance(value, torch.Tensor)
                    for value in (logical_input, logical_output, real_input, real_output)
                ):
                    raise WorkloadFreezeError("region alias does not describe tensors")
                Region.equivalent_tensor_id_manager.add_equivalent_keys(
                    [(id(real_input), 0), (id(logical_input), alias.reference_id)]
                )
                Region.equivalent_tensor_id_manager.add_equivalent_keys(
                    [(id(real_output), 0), (id(logical_output), alias.reference_id)]
                )
            invocations = []
            for frozen_op in self.invocations:
                invocations.append(
                    (
                        OpInvokeInfo(
                            _resolve_op(frozen_op.schema_name, frozen_op.overload_name),
                            thawer.thaw(frozen_op.args),
                            thawer.thaw(frozen_op.kwargs),
                            thawer.thaw(frozen_op.out),
                            cache_key=frozen_op.cache_key,
                        ),
                        frozen_op.reference_id,
                    )
                )
            runtime.replay_flat_op_invoke_infos(invocations)


class _ValueFreezer:
    def __init__(self) -> None:
        self.tensors: dict[int, FrozenTensor] = {}

    def freeze(self, value: Any) -> Any:
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, torch.Tensor):
            tensor_id = id(value)
            if tensor_id not in self.tensors:
                try:
                    self.tensors[tensor_id] = FrozenTensor(
                        tensor_id=tensor_id,
                        shape=tuple(int(dim) for dim in value.shape),
                        stride=tuple(int(dim) for dim in value.stride()),
                        dtype=str(value.dtype),
                        requires_grad=bool(value.requires_grad),
                    )
                except (TypeError, ValueError) as error:
                    raise WorkloadFreezeError(f"unsupported tensor metadata: {value!r}") from error
            return ("tensor", tensor_id)
        if isinstance(value, torch.dtype):
            return ("dtype", str(value))
        if isinstance(value, torch.device):
            return ("device", str(value))
        if isinstance(value, torch.Size):
            return ("size", tuple(int(dim) for dim in value))
        if isinstance(value, tuple):
            return ("tuple", tuple(self.freeze(item) for item in value))
        if isinstance(value, list):
            return ("list", tuple(self.freeze(item) for item in value))
        if isinstance(value, dict):
            return ("dict", tuple((self.freeze(key), self.freeze(item)) for key, item in value.items()))
        if isinstance(value, slice):
            return ("slice", self.freeze(value.start), self.freeze(value.stop), self.freeze(value.step))
        if type(value).__module__ == "torch" and type(value).__name__ in {"layout", "memory_format"}:
            return (type(value).__name__, str(value))
        raise WorkloadFreezeError(f"unsupported runtime value: {type(value).__module__}.{type(value).__name__}")


class _ValueThawer:
    def __init__(self, tensors: tuple[FrozenTensor, ...]) -> None:
        self._tensor_metadata = {tensor.tensor_id: tensor for tensor in tensors}
        self._tensors: dict[int, torch.Tensor] = {}

    def _tensor(self, tensor_id: int) -> torch.Tensor:
        if tensor_id not in self._tensors:
            try:
                frozen = self._tensor_metadata[tensor_id]
                dtype = getattr(torch, frozen.dtype.removeprefix("torch."))
                self._tensors[tensor_id] = torch.empty_strided(
                    frozen.shape,
                    frozen.stride,
                    dtype=dtype,
                    device="meta",
                    requires_grad=frozen.requires_grad,
                )
            except (AttributeError, KeyError, RuntimeError) as error:
                raise WorkloadFreezeError(f"unable to recreate tensor {tensor_id}") from error
        return self._tensors[tensor_id]

    def thaw(self, value: Any) -> Any:
        if not isinstance(value, tuple) or not value:
            return value
        tag = value[0]
        if tag == "tensor":
            return self._tensor(value[1])
        if tag == "dtype":
            return getattr(torch, value[1].removeprefix("torch."))
        if tag == "device":
            return torch.device(value[1])
        if tag == "size":
            return torch.Size(value[1])
        if tag == "tuple":
            return tuple(self.thaw(item) for item in value[1])
        if tag == "list":
            return [self.thaw(item) for item in value[1]]
        if tag == "dict":
            return {self.thaw(key): self.thaw(item) for key, item in value[1]}
        if tag == "slice":
            return slice(self.thaw(value[1]), self.thaw(value[2]), self.thaw(value[3]))
        if tag == "layout":
            return getattr(torch, value[1].removeprefix("torch."))
        if tag == "memory_format":
            return getattr(torch, value[1].removeprefix("torch."))
        raise WorkloadFreezeError(f"unsupported frozen value tag: {tag!r}")


def _resolve_op(schema_name: str, overload_name: str):
    try:
        namespace, operator = schema_name.split("::", maxsplit=1)
        packet = getattr(getattr(torch.ops, namespace), operator)
        return getattr(packet, overload_name or "default")
    except (AttributeError, ValueError) as error:
        raise WorkloadFreezeError(f"unable to resolve op {schema_name}.{overload_name}") from error
