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

"""Pure-Python replay preflight for generated CSV rows.

Every row written by shape-grid generation is executed against the *real*
``op_replay/*_run.py`` ``build_case`` entry point with the NPU tensor plumbing
stubbed out. The shared parsing, dtype, format, and structural auxiliary-tensor
constraints inside each kernel's ``build_case`` therefore run without allocating
device tensors. This is not an NPU execution check and cannot validate
data-dependent constraints: ``item()`` uses a neutral placeholder value. Rows
whose ``build_case`` raises are not written to the CSV; the failure reason is
reported instead of silently dropping the row during NPU replay.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import importlib.util
import math
from pathlib import Path
import re
import sys
from types import ModuleType
from typing import Any, Iterator

try:
    from ..signature_utils import normalize_op_name
except ImportError:  # pragma: no cover - direct-script execution path
    from signature_utils import normalize_op_name


OP_REPLAY_PACKAGE_DIR = Path(__file__).resolve().parents[1] / "op_replay"


class _StubRuntime(ModuleType):
    """Duck-typed stand-in for ``torch``/``torch_npu`` during preflight."""

    _SHAPE_FACTORIES = ("arange", "full", "zeros", "ones", "empty", "randint", "randn")

    def __getattr__(self, name: str) -> Any:
        if name in _StubRuntime._SHAPE_FACTORIES:
            return _StubFactory(name)
        return _StubAny(f"{self.__name__}.{name}")


class _StubFactory:
    """Factory placeholder (torch.arange/zeros/...) returning shaped stubs."""

    def __init__(self, name: str) -> None:
        self._name = name

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        dtype = kwargs.get("dtype", "DT_UNDEFINED")
        for arg in args:
            if isinstance(arg, (tuple, list)) and all(isinstance(dim, int) for dim in arg):
                return _StubTensor(tuple(arg), dtype, self._name)
            if isinstance(arg, int) and not isinstance(arg, bool):
                return _StubTensor((arg,), dtype, self._name)
        return _StubAny(self._name)


class _StubAny:
    """Permissive placeholder returned by any stubbed attribute or call."""

    def __init__(self, label: str = "stub") -> None:
        self._label = label

    def __call__(self, *args: Any, **kwargs: Any) -> "_StubAny":
        return self

    def __getattr__(self, name: str) -> "_StubAny":
        return _StubAny(f"{self._label}.{name}")

    def __iter__(self):
        return iter(())

    def __len__(self) -> int:
        return 0

    def __getitem__(self, item: Any) -> "_StubAny":
        return self

    def __repr__(self) -> str:
        return f"<stub {self._label}>"


class _StubTensor:
    """Tensor placeholder that remembers the requested shape and dtype."""

    def __init__(self, shape: tuple[int, ...], dtype: Any, label: str) -> None:
        self.shape = shape
        self.dtype = dtype
        self._label = label

    def _same(self) -> "_StubTensor":
        return _StubTensor(self.shape, self.dtype, self._label)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return _StubMethod(self, name)

    def __call__(self, *args: Any, **kwargs: Any) -> "_StubTensor":
        return self

    def __getitem__(self, item: Any) -> "_StubTensor":
        return self._same()

    def __setitem__(self, item: Any, value: Any) -> None:
        return None

    def __mod__(self, other: Any) -> "_StubTensor":
        return self._same()

    def __floordiv__(self, other: Any) -> "_StubTensor":
        return self._same()

    def __add__(self, other: Any) -> "_StubTensor":
        return self._same()

    def __sub__(self, other: Any) -> "_StubTensor":
        return self._same()

    def __mul__(self, other: Any) -> "_StubTensor":
        return self._same()

    def __iter__(self):
        return iter(())

    def __len__(self) -> int:
        return self.shape[0] if self.shape else 0

    def __repr__(self) -> str:
        return f"<stub-tensor shape={self.shape}>"


class _StubMethod:
    """Attribute-access placeholder on a stub tensor.

    Method calls return shape-preserving stub tensors so downstream
    ``.shape`` arithmetic keeps working; known scalar-returning methods
    produce real values instead.
    """

    _SHAPE_METHODS = ("reshape", "view", "resize")

    def __init__(self, tensor: _StubTensor, name: str) -> None:
        self._tensor = tensor
        self._name = name

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self._name == "item":
            return 0
        if self._name in ("numel", "nelement"):
            return math.prod(self._tensor.shape)
        if self._name == "dim":
            return len(self._tensor.shape)
        if self._name in _StubMethod._SHAPE_METHODS:
            for arg in args:
                if isinstance(arg, (tuple, list)) and all(isinstance(dim, int) for dim in arg):
                    return _StubTensor(tuple(arg), self._tensor.dtype, self._tensor._label)
            ints = [arg for arg in args if isinstance(arg, int)]
            if ints:
                return _StubTensor(tuple(ints), self._tensor.dtype, self._tensor._label)
        return self._tensor


def _stub_build_input_tensor(
    shape: tuple[int, ...] | None,
    input_format: str,
    dtype_name: str,
    transpose: bool = False,
) -> _StubTensor:
    if shape is None:
        raise ValueError(
            "build_input_tensor received None shape "
            f"(dtype={dtype_name}, format={input_format})"
        )
    if any(dim is None for dim in shape):
        raise ValueError(f"build_input_tensor received shape with None elements: {shape}")
    # Route the dtype through the real runtime resolution so unsupported dtype
    # names fail the preflight exactly like a microbench replay would.
    common = sys.modules.get("common")
    if common is not None and hasattr(common, "resolve_runtime_dtype"):
        common.resolve_runtime_dtype(dtype_name)
    return _StubTensor(tuple(shape), dtype_name, f"input[{input_format}]")


def _stub_build_host_tensor(shape: tuple[int, ...], dtype: Any) -> _StubTensor:
    return _StubTensor(tuple(shape), dtype, "host")


def _identity_maybe_cast_internal_format(tensor: Any, input_format: str) -> Any:
    return tensor


def _noop_init_runtime() -> None:
    return None


def _stub_get_runtime_modules() -> tuple[Any, Any]:
    return _StubRuntime("torch"), _StubRuntime("torch_npu")


def _ensure_op_replay_on_path() -> None:
    package_dir = str(OP_REPLAY_PACKAGE_DIR)
    if package_dir not in sys.path:
        sys.path.insert(0, package_dir)


def _patched_namespaces(run_module: ModuleType) -> list[ModuleType]:
    namespaces = [run_module]
    for module_name in ("common", "replay_framework"):
        module = sys.modules.get(module_name)
        if module is not None:
            namespaces.append(module)
    return namespaces


@contextmanager
def _patch_runtime_stub(run_module: ModuleType) -> Iterator[None]:
    """Temporarily replace NPU tensor plumbing with pure-Python stubs."""
    # Distributed replay scripts set up HCCL/torchrun groups from their case
    # builders; preflight must neutralize those entry points like the NPU
    # tensor plumbing.
    replacements = {
        "init_runtime": _noop_init_runtime,
        "get_runtime_modules": _stub_get_runtime_modules,
        "build_input_tensor": _stub_build_input_tensor,
        "build_host_tensor": _stub_build_host_tensor,
        "maybe_cast_internal_format": _identity_maybe_cast_internal_format,
        "ensure_npu_available": lambda: None,
        "get_default_hccl_group_name": lambda *args, **kwargs: "preflight_hccl_group",
        "init_ep_process_group": lambda *args, **kwargs: "preflight_ep_group",
        "launch_torchrun_and_wait": lambda *args, **kwargs: None,
    }
    patched_attributes: list[tuple[ModuleType, str, Any]] = []
    for namespace in _patched_namespaces(run_module):
        for name, replacement in replacements.items():
            if hasattr(namespace, name):
                patched_attributes.append((namespace, name, getattr(namespace, name)))
                setattr(namespace, name, replacement)
    common = sys.modules.get("common")
    common_runtime_attributes: list[tuple[ModuleType, str, Any]] = []
    if common is not None:
        common_runtime_attributes.extend(
            (common, name, getattr(common, name)) for name in ("torch", "torch_npu", "DTYPE_MAP")
        )
        common.torch = _StubRuntime("torch")
        common.torch_npu = _StubRuntime("torch_npu")
        if not getattr(common, "DTYPE_MAP", None):
            common.DTYPE_MAP = {
                "DT_FLOAT": "float32",
                "DT_FLOAT16": "float16",
                "DT_BF16": "bfloat16",
                "DT_DOUBLE": "float64",
                "DT_INT8": "int8",
                "DT_UINT8": "uint8",
                "DT_INT16": "int16",
                "DT_INT32": "int32",
                "DT_INT64": "int64",
                "DT_BOOL": "bool",
            }
    try:
        yield
    finally:
        for namespace, name, original in reversed(common_runtime_attributes):
            setattr(namespace, name, original)
        for namespace, name, original in reversed(patched_attributes):
            setattr(namespace, name, original)


def _resolve_run_module(kernel_type: str, op_replay_dir: Path) -> ModuleType | None:
    module_name = "op_replay_preflight_" + re.sub(r"\W", "_", kernel_type).strip("_")
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    run_path = op_replay_dir / f"{normalize_op_name(kernel_type)}_run.py"
    if not run_path.is_file():
        # The kernel name may itself differ from the file stem (for example
        # _triton_rope_siso); fall back to a suffix-insensitive scan.
        candidates = [
            path
            for path in op_replay_dir.glob("*_run.py")
            if normalize_op_name(path.name) == normalize_op_name(kernel_type)
        ]
        if not candidates:
            return None
        run_path = candidates[0]
    _ensure_op_replay_on_path()
    spec = importlib.util.spec_from_file_location(module_name, run_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


def _resolve_build_case(run_module: ModuleType) -> Any:
    build_case = getattr(run_module, "build_case", None)
    if callable(build_case):
        return build_case
    op = getattr(run_module, "op", None)
    if op is not None and callable(getattr(op, "build_case", None)):
        return op.build_case
    # Distributed replay scripts (for example DispatchFFNCombine) expose a
    # row-level case builder instead of the standard build_case entry point.
    build_row_case = getattr(run_module, "build_row_case", None)
    if callable(build_row_case):
        return build_row_case
    return None


@dataclass(frozen=True)
class RowPreflightResult:
    row_index: int
    passed: bool
    reason: str = ""


def preflight_generated_rows(
    kernel_type: str,
    rows: list[dict[str, str]],
    op_replay_dir: Path,
) -> list[RowPreflightResult]:
    """Dry-run every row through the kernel's real ``build_case`` entry point.

    Returns one result per row in input order. Rows whose ``build_case`` raises
    are marked failed with the exception summary; callers must not write them.
    Kernels without an ``op_replay`` entry point fail closed for every row.
    """
    results: list[RowPreflightResult] = []
    if not rows:
        return results
    try:
        run_module = _resolve_run_module(kernel_type, op_replay_dir)
    except Exception as error:
        run_module = None
        import_error = f"import failed for {kernel_type} run module: {type(error).__name__}: {error}"
    else:
        import_error = ""
    if run_module is None:
        reason = import_error or f"no op_replay entry point found for kernel {kernel_type}"
        return [RowPreflightResult(index, False, reason) for index in range(len(rows))]
    build_case = _resolve_build_case(run_module)
    if build_case is None:
        reason = f"kernel {kernel_type} exposes no build_case entry point"
        return [RowPreflightResult(index, False, reason) for index in range(len(rows))]
    with _patch_runtime_stub(run_module):
        for index, row in enumerate(rows):
            try:
                build_case(row)
            except Exception as error:
                summary = f"{type(error).__name__}: {error}"
                results.append(RowPreflightResult(index, False, summary[:400]))
            else:
                results.append(RowPreflightResult(index, True))
    return results
