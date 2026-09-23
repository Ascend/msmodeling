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

"""Profiled child process for one standalone operator request."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import re
import traceback
from pathlib import Path
from typing import Any

try:
    from .adapters import build_runtime_case
except ImportError:
    from adapters import build_runtime_case


TORCH_DTYPE_NAMES: dict[str, str] = {
    "torch.bool": "DT_BOOL",
    "torch.int8": "DT_INT8",
    "torch.uint8": "DT_UINT8",
    "torch.int16": "DT_INT16",
    "torch.int32": "DT_INT32",
    "torch.int64": "DT_INT64",
    "torch.float16": "DT_FLOAT16",
    "torch.bfloat16": "DT_BF16",
    "torch.float32": "DT_FLOAT",
    "torch.float64": "DT_DOUBLE",
}
DTYPE_SIZE_BYTES = {
    "DT_BOOL": 1,
    "DT_INT8": 1,
    "DT_UINT8": 1,
    "DT_INT16": 2,
    "DT_FLOAT16": 2,
    "DT_BF16": 2,
    "DT_INT32": 4,
    "DT_FLOAT": 4,
    "DT_FLOAT32": 4,
    "DT_INT64": 8,
    "DT_DOUBLE": 8,
    "DT_FLOAT64": 8,
}


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Execute one allow-listed operator on Ascend NPU.")
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--metadata-out", type=Path, required=True)
    return parser


def _distribution_version(*names: str) -> str | None:
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def _format_name(torch_npu: Any, tensor: Any) -> str | None:
    try:
        format_id = int(torch_npu.get_npu_format(tensor))
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None
    for name in dir(torch_npu.Format):
        if not name.isupper():
            continue
        try:
            if int(getattr(torch_npu.Format, name)) == format_id:
                return name
        except (TypeError, ValueError):
            continue
    return str(format_id)


def _tensor_metadata(torch_npu: Any, tensor: Any, *, name: str | None, path: str | None, origin: str) -> dict[str, Any]:
    dtype_name = TORCH_DTYPE_NAMES.get(str(tensor.dtype), str(tensor.dtype))
    logical_shape = list(tensor.shape)
    return {
        "name": name,
        "path": path,
        "origin": origin,
        "logical_shape": logical_shape,
        "device_shape": logical_shape,
        "dtype": dtype_name,
        "format": _format_name(torch_npu, tensor),
        "size_bytes": tensor.numel() * tensor.element_size(),
    }


def _replay_input_metadata(tensor: dict[str, Any]) -> dict[str, Any]:
    elements = 1
    for dim in tensor["shape"]:
        elements *= dim
    element_size = DTYPE_SIZE_BYTES.get(tensor["dtype"])
    return {
        "name": tensor["name"],
        "path": None,
        "origin": "replay_metadata",
        "logical_shape": tensor["shape"],
        "device_shape": None,
        "dtype": tensor["dtype"],
        "format": tensor["format"],
        "size_bytes": None if element_size is None else elements * element_size,
    }


def _flatten_outputs(value: Any, path: str = "$") -> list[tuple[str, Any]]:
    if isinstance(value, (tuple, list)):
        flattened: list[tuple[str, Any]] = []
        for index, item in enumerate(value):
            flattened.extend(_flatten_outputs(item, f"{path}[{index}]"))
        return flattened
    if isinstance(value, dict):
        flattened = []
        for key in sorted(value):
            flattened.extend(_flatten_outputs(value[key], f"{path}.{key}"))
        return flattened
    return [(path, value)]


def _environment(torch: Any, torch_npu: Any, device_id: int) -> dict[str, Any]:
    try:
        device_name = torch.npu.get_device_name(device_id)
    except (AttributeError, RuntimeError):
        device_name = None
    cann_version = _cann_version(torch_npu)
    return {
        "device_id": device_id,
        "device_name": device_name,
        "cann_version": cann_version,
        "torch_version": torch.__version__,
        "torch_npu_version": torch_npu.__version__,
        "vllm_ascend_version": _distribution_version("vllm-ascend", "vllm_ascend"),
        "msprof_version": None,
        "simulated": False,
    }


def _cann_version(torch_npu: Any) -> str | None:
    version_module = getattr(torch_npu, "version", None)
    for candidate in (
        getattr(version_module, "cann", None),
        getattr(version_module, "cann_version", None),
    ):
        if candidate:
            return str(candidate)
    roots = []
    for env_name in (
        "ASCEND_HOME_PATH",
        "ASCEND_TOOLKIT_HOME",
        "ASCEND_TOOLKIT_HOME_PATH",
        "ASCEND_INSTALL_PATH",
    ):
        raw_value = (os.environ.get(env_name, "") or "").strip()
        if raw_value:
            roots.append(Path(raw_value))
    roots.extend(
        [
            Path("/usr/local/Ascend/ascend-toolkit/latest"),
            Path("/usr/local/Ascend/cann"),
            Path("/usr/local/Ascend/latest"),
        ]
    )
    for root in roots:
        name_match = re.fullmatch(r"cann-([0-9A-Za-z._+-]+)", root.name)
        if name_match:
            return name_match.group(1)
        for filename in ("version.info", "ascend_toolkit_install.info"):
            path = root / filename
            try:
                raw_text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            match = re.search(
                r"(?im)^(?:version|package_version)\s*[=:]\s*([^\s]+)\s*$",
                raw_text,
            )
            if match:
                return match.group(1).strip()
    return None


def execute(request: dict[str, Any]) -> dict[str, Any]:
    """Build and execute exactly warmup_count + repeat_count invocations."""
    import torch
    import torch_npu

    device_id = request["profiling"]["device_id"]
    torch.npu.set_device(device_id)
    torch_npu.npu.config.allow_internal_format = True
    case = build_runtime_case(request)
    torch.npu.synchronize()

    output = None
    total_count = request["profiling"]["warmup_count"] + request["profiling"]["repeat_count"]
    for _ in range(total_count):
        output = case["invoke"]()
        torch.npu.synchronize()
    if output is None:
        raise RuntimeError("The operator did not produce an output.")

    inputs = [_replay_input_metadata(tensor) for tensor in request["inputs"]]
    outputs = []
    for path, tensor in _flatten_outputs(output):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Unsupported non-Tensor operator output at {path}: {type(tensor).__name__}")
        outputs.append(_tensor_metadata(torch_npu, tensor, name=None, path=path, origin="operator"))

    spec = case["spec"]
    return {
        "status": "succeeded",
        "operator": {
            "requested_kernel_type": request["kernel_type"],
            "resolved_adapter": spec.name,
            "api_path": spec.api_path,
            "target_op_types": list(spec.target_op_types),
            "is_composite": spec.is_composite,
        },
        "environment": _environment(torch, torch_npu, device_id),
        "inputs": inputs,
        "outputs": outputs,
        "execution": {
            "warmup_count": request["profiling"]["warmup_count"],
            "repeat_count": request["profiling"]["repeat_count"],
            "total_invocations": total_count,
        },
        "error": None,
    }


def _write_metadata(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    try:
        request = json.loads(args.request.read_text(encoding="utf-8"))
        result = execute(request)
    except Exception as exc:
        result = {
            "status": "failed",
            "operator": None,
            "environment": None,
            "inputs": [],
            "outputs": [],
            "execution": None,
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        }
        _write_metadata(args.metadata_out, result)
        return 1
    _write_metadata(args.metadata_out, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
