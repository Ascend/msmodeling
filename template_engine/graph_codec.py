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

"""JSON+xz graph codec for template-engine artifacts.

Replaces pickle as the on-disk format for captured operation traces and
template graphs (review issue #32 security follow-up: pickle is executable
serialization; JSON cannot instantiate arbitrary objects).

The codec covers exactly the value surface the corpus census found inside
``CapturedOp`` trees and template payloads: ``None``/``bool``/``int``/
``float``/``str``/``list``/``dict`` (string keys) travel as native JSON;
tensors, tuples, ``torch.dtype``, ``torch.layout`` and ``torch.memory_format`` values travel
through explicit sentinel objects so their semantics survive the round
trip.  Tensors are Meta placeholders - only shape/stride/dtype identity is
meaningful, never values - and the alias identity the capture pool creates
is preserved through an explicit per-document tensor-id table.

Sentinels (deliberately terse for the multi-million-node graphs):

- ``{"__t__": {"id": N, "s": [...], "st": [...], "d": "torch.bfloat16"}}``
  one entry per distinct tensor; decoders must return THE SAME tensor
  object for repeated ids (aliasing is load-bearing in template building).
- ``{"__u__": [...]}`` - tuple.
- ``{"__dt__": "torch.bfloat16"}`` - ``torch.dtype``.
- ``{"__layout__": "torch.strided"}`` - ``torch.layout``.
- ``{"__mf__": "torch.contiguous_format"}`` - ``torch.memory_format``.
- ``{"__b64__": "..."}`` - ``bytes`` (base64).
- ``{"__enum__": "tensor_cast.core.quantization.datatypes.QuantScheme.SYMMETRIC"}``
  - ``enum.Enum``; restored through attribute lookup validated to BE an
  Enum instance (getattr cannot execute code beyond importing the module).
- ``{"__dc__": {"cls": "...", "fields": {...}}}`` - a ``dataclass`` from
  the project's own modules; restored as ``cls(**fields)`` after checking
  the class really is a dataclass.
- ``{"__op__": [func, args, kwargs, out]}`` - ``CapturedOp``.

Non-string dictionary keys travel as explicit key/value pairs. String keys,
including sentinel-looking names, round-trip unchanged. Legacy group-key
restoration belongs to the lazy template reader, not to this generic codec.
HF configurations are inert data snapshots; only a closed set of project
records and exports may be restored. Artifact data never selects an import.
"""

from __future__ import annotations

import json
import lzma
import os
import tempfile
from pathlib import Path
from typing import Any

_XZ_MAGIC = b"\xfd7zXZ\x00"
_PRESET = 6
#: 9 | lzma.PRESET_EXTREME - used by the low-frequency template writers.
PRESET_MAX = 9 | (1 << 31)


class GraphCodecError(ValueError):
    """Raised when a JSON+xz artifact does not follow the graph codec."""


def _short(name: str) -> str:
    return name.split(".", 1)[-1] if name.startswith("torch.") else name


class ArtifactConfig:
    """Data-only config snapshot; deserialization never invokes HF constructors."""

    def __init__(self, data):
        self._data = dict(data)

    def __getattr__(self, name):
        if name not in self._data:
            raise AttributeError(name)
        value = self._data[name]
        return ArtifactConfig(value) if isinstance(value, dict) else value

    def to_dict(self):
        return dict(self._data)

    @classmethod
    def from_dict(cls, data):
        return cls(data)

    def get_text_config(self):
        nested = self._data.get("text_config")
        return ArtifactConfig(nested) if isinstance(nested, dict) else self


# Exact exports, checked BEFORE importing. No prefix-wide module trust and no
# artifact-selected constructors/functions outside this closed set.
_TRUSTED_EXPORTS = {
    "tensor_cast.quantize_utils": {"LinearQuantType", "AttentionQuantType", "QuantScheme", "QuantGranularity"},
    "template_engine.capture_ops": {"CapturedOp"},
    "tensor_cast.model_config": {
        "ModelConfig",
        "ParallelConfig",
        "QuantConfig",
        "LinearQuantConfig",
        "AttentionQuantConfig",
        "MultiheadLatentAttentionQuantConfig",
        "MoEFieldNames",
        "MoEConfig",
        "MlaFieldNames",
        "MlaConfig",
        "MtpConfig",
        "DflashConfig",
        "DsparkConfig",
        "RemoteSource",
        "WordEmbeddingTPMode",
        "AttentionBackend",
        "AttentionRoutePlan",
        "DiffusersPipelineMetadata",
        "DiffusersConfig",
        "DiffusersTextConfig",
        "DiffusersTransformerConfig",
        "DiffusersVaeConfig",
    },
    "tensor_cast.core.quantization.datatypes": {
        "QuantScheme",
        "QuantGranularity",
        "QuantizeLinearAction",
        "QuantizeAttentionAction",
        "QuantDtype",
    },
    "tensor_cast.layers.attention": {"AttentionTensorCast"},
    "tensor_cast.layers.glm5": {"Glm5SparseAttention"},
    "tensor_cast.layers.quant_linear": {"TensorCastQuantLinear"},
    "tensor_cast.layers.deepseek_v4": {"DeepseekV4SparseAttention", "route_deepseek_v4_gate"},
    "torch.nn.functional": {"silu", "gelu", "relu"},
}


def _resolve_torch_path(name: str) -> Any:
    import importlib

    for module_name, exports in _TRUSTED_EXPORTS.items():
        prefix = module_name + "."
        if not name.startswith(prefix):
            continue
        attributes = name[len(prefix) :].split(".")
        if attributes[0] not in exports or any(part.startswith("_") for part in attributes):
            break
        obj = importlib.import_module(module_name)
        for attribute in attributes:
            obj = getattr(obj, attribute, None)
        return obj
    raise GraphCodecError(f"artifact reference is not an approved export: {name!r}")


def encode_value(value: Any, tensors: list[dict[str, Any]], tensor_ids: dict[int, int]) -> Any:
    """Convert one node of a capture/template tree to JSON-native form."""

    import base64
    import dataclasses
    import enum
    import types

    import torch

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise GraphCodecError(f"non-finite float cannot round-trip through JSON: {value!r}")
        return value
    if isinstance(value, torch.Tensor):
        source_id = id(value)
        index = tensor_ids.get(source_id)
        if index is None:
            index = len(tensors)
            tensor_ids[source_id] = index
            tensors.append(
                {
                    "id": index,
                    "s": [int(d) for d in value.shape],
                    "st": [int(s) for s in value.stride()],
                    "d": str(value.dtype),
                }
            )
        return {"__t__": index}
    if isinstance(value, torch.dtype):
        return {"__dt__": str(value)}
    if isinstance(value, torch.layout):
        return {"__layout__": str(value)}
    if isinstance(value, torch.memory_format):
        return {"__mf__": str(value)}
    if isinstance(value, torch.device):
        return {"__dev__": str(value)}
    if isinstance(value, enum.Enum):
        return {"__enum__": f"{type(value).__module__}.{type(value).__name__}.{value.name}"}
    if isinstance(value, types.FunctionType) and getattr(value, "__module__", None):
        return {"__fn__": f"{value.__module__}.{value.__qualname__}"}
    if isinstance(value, type) and getattr(value, "__module__", None):
        return {"__cls__": f"{value.__module__}.{value.__qualname__}"}
    # PretrainedConfig-shaped objects (duck-typed: to_dict + from_dict, no
    # transformers import here); restored through cls.from_dict
    if hasattr(value, "to_dict") and hasattr(type(value), "from_dict") and not isinstance(value, type):
        cls_name = f"{type(value).__module__}.{type(value).__qualname__}"
        try:
            dumped = value.to_dict()
        except Exception as error:  # noqa: BLE001 - not a serialisable config-shaped object
            raise GraphCodecError(f"config-shaped object {cls_name} failed to_dict: {error}") from error
        return {"__cfg__": {"cls": cls_name, "dict": encode_value(dumped, tensors, tensor_ids)}}
    if isinstance(value, (bytes, bytearray)):
        return {"__b64__": base64.b64encode(bytes(value)).decode("ascii")}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        fields = {
            field.name: encode_value(getattr(value, field.name), tensors, tensor_ids)
            for field in dataclasses.fields(value)
        }
        return {"__dc__": {"cls": f"{type(value).__module__}.{type(value).__qualname__}", "fields": fields}}
    if isinstance(value, dict):
        if all(isinstance(key, str) and not key.startswith("__") for key in value):
            return {key: encode_value(item, tensors, tensor_ids) for key, item in value.items()}
        # mixed / non-string keys: lossless pair list (JSON object keys are
        # strings, so every non-str key travels through encode_value and is
        # rebuilt with its original type on decode)
        return {
            "__d__": [
                [encode_value(key, tensors, tensor_ids), encode_value(item, tensors, tensor_ids)]
                for key, item in value.items()
            ]
        }
    if isinstance(value, tuple):
        return {"__u__": [encode_value(item, tensors, tensor_ids) for item in value]}
    if isinstance(value, list):
        return [encode_value(item, tensors, tensor_ids) for item in value]
    if hasattr(value, "func") and hasattr(value, "args") and hasattr(value, "kwargs"):
        return {
            "__op__": [
                str(value.func),
                encode_value(value.args, tensors, tensor_ids),
                encode_value(value.kwargs, tensors, tensor_ids),
                encode_value(value.out, tensors, tensor_ids),
            ]
        }
    raise GraphCodecError(f"value type is not representable: {type(value).__module__}.{type(value).__qualname__}")


def decode_value(payload: Any, tensors: list[Any]) -> Any:
    """Rebuild one JSON-native node, materialising tensors through ``tensors``."""

    import base64
    import dataclasses
    import enum

    import torch

    if payload is None or isinstance(payload, (bool, int, float, str)):
        return payload
    if isinstance(payload, list):
        return [decode_value(item, tensors) for item in payload]
    if isinstance(payload, dict):
        if len(payload) != 1:
            return {key: decode_value(item, tensors) for key, item in payload.items()}
        if "__d__" in payload:
            pairs = payload["__d__"]
            if not isinstance(pairs, list):
                raise GraphCodecError("__d__ payload must be a list of [key, value] pairs")
            return {decode_value(key, tensors): decode_value(item, tensors) for key, item in pairs}
        if "__t__" in payload:
            index = payload["__t__"]
            if type(index) is not int or not 0 <= index < len(tensors):
                raise GraphCodecError("invalid tensor reference")
            return tensors[index]
        if "__u__" in payload:
            return tuple(decode_value(item, tensors) for item in payload["__u__"])
        if "__dt__" in payload:
            name = str(payload["__dt__"])
            resolved = getattr(torch, _short(name), None)
            if not isinstance(resolved, torch.dtype):
                raise GraphCodecError(f"unknown torch dtype name: {name!r}")
            return resolved
        if "__layout__" in payload:
            name = payload["__layout__"]
            if not isinstance(name, str):
                raise GraphCodecError("torch layout name must be a string")
            resolved = getattr(torch, _short(name), None)
            if not isinstance(resolved, torch.layout):
                raise GraphCodecError(f"unknown torch layout name: {name!r}")
            return resolved
        if "__mf__" in payload:
            name = str(payload["__mf__"])
            resolved = getattr(torch, _short(name), None)
            if not isinstance(resolved, torch.memory_format):
                raise GraphCodecError(f"unknown torch memory format: {name!r}")
            return resolved
        if "__dev__" in payload:
            return torch.device(str(payload["__dev__"]))
        if "__enum__" in payload:
            name = str(payload["__enum__"])
            resolved = _resolve_torch_path(name)
            if not isinstance(resolved, enum.Enum):
                raise GraphCodecError(f"unknown enum name: {name!r}")
            return resolved
        if "__fn__" in payload:
            import types

            name = str(payload["__fn__"])
            resolved = _resolve_torch_path(name)
            if not isinstance(resolved, types.FunctionType):
                raise GraphCodecError(f"unknown function reference: {name!r}")
            return resolved
        if "__cls__" in payload:
            name = str(payload["__cls__"])
            resolved = _resolve_torch_path(name)
            if not isinstance(resolved, type):
                raise GraphCodecError(f"unknown class reference: {name!r}")
            return resolved
        if "__cfg__" in payload:
            spec = payload["__cfg__"]
            name = str(spec.get("cls", ""))
            if not isinstance(spec.get("cls"), str) or not name:
                raise GraphCodecError("config snapshot class label must be a nonempty string")
            # The original class name is provenance only. Custom model configs
            # are as inert as HF configs: no import and no from_dict invocation.
            dumped = decode_value(spec.get("dict"), tensors)
            if not isinstance(dumped, dict):
                raise GraphCodecError("config payload must be an object")
            return ArtifactConfig(dumped)
        if "__b64__" in payload:
            return base64.b64decode(str(payload["__b64__"]))
        if "__dc__" in payload:
            spec = payload["__dc__"]
            cls = _resolve_torch_path(str(spec.get("cls", "")))
            if not isinstance(cls, type) or not dataclasses.is_dataclass(cls):
                raise GraphCodecError(f"unknown dataclass: {spec.get('cls')!r}")
            fields = spec.get("fields")
            if not isinstance(fields, dict):
                raise GraphCodecError("dataclass fields must be an object")
            return cls(**{key: decode_value(item, tensors) for key, item in fields.items()})
        if "__op__" in payload:
            from .capture_ops import CapturedOp

            fields = payload["__op__"]
            if not isinstance(fields, list) or len(fields) != 4:
                raise GraphCodecError("__op__ payload must be [func, args, kwargs, out]")
            return CapturedOp(
                func=str(fields[0]),
                args=decode_value(fields[1], tensors),
                kwargs=decode_value(fields[2], tensors),
                out=decode_value(fields[3], tensors),
            )
        return {key: decode_value(item, tensors) for key, item in payload.items()}
    raise GraphCodecError(f"unexpected JSON payload type: {type(payload).__name__}")


def encode_document(payload: Any, kind: str, *, preset: int = _PRESET) -> bytes:
    """JSON-encode a whole artifact (payload + tensor table) and xz it."""

    tensors: list[dict[str, Any]] = []
    tensor_ids: dict[int, int] = {}
    doc = {
        "format": "graph_codec_json/1",
        "kind": kind,
        "data": encode_value(payload, tensors, tensor_ids),
        "tensors": tensors,
    }
    return dumps_bytes(doc, preset=preset)


def decode_document(data: bytes, kind: str) -> Any:
    """Decode artifact bytes written by :func:`encode_document`."""

    doc = loads_bytes(data)
    if not isinstance(doc, dict) or doc.get("format") != "graph_codec_json/1":
        raise GraphCodecError("artifact root must be a graph_codec_json/1 object")
    if doc.get("kind") != kind:
        raise GraphCodecError(f"artifact kind mismatch: expected {kind!r}, got {doc.get('kind')!r}")
    tensors = [_materialise_tensor(entry) for entry in doc.get("tensors", [])]
    return decode_value(doc.get("data"), tensors)


def encode_capture(ops: Any, exec_time: float) -> bytes:
    """Encode a ``(ops, exec_time)`` capture payload as artifact bytes."""

    return encode_document({"ops": list(ops), "exec_time": float(exec_time)}, "capture/1")


def decode_capture(data: bytes) -> tuple[list[Any], float]:
    """Decode capture artifact bytes back to ``(ops, exec_time)``."""

    payload = decode_document(data, "capture/1")
    if not isinstance(payload, dict):
        raise GraphCodecError("capture artifact payload must be an object")
    ops = payload.get("ops")
    if not isinstance(ops, list):
        raise GraphCodecError(f"capture ops must decode to a list, got {type(ops).__name__}")
    return ops, float(payload["exec_time"])


def _materialise_tensor(entry: Any) -> Any:
    """Rebuild one Meta placeholder tensor from its codec entry."""

    import torch

    if not isinstance(entry, dict):
        raise GraphCodecError(f"tensor entry must be an object, got {type(entry).__name__}")
    name = str(entry.get("d"))
    dtype = getattr(torch, _short(name), None)
    if not isinstance(dtype, torch.dtype):
        raise GraphCodecError(f"unknown torch dtype name: {name!r}")
    shape = [int(d) for d in entry.get("s", [])]
    stride = [int(s) for s in entry.get("st", [])]
    return torch.empty_strided(tuple(shape), tuple(stride), dtype=dtype, device="meta")


def dumps_bytes(payload: Any, *, preset: int = _PRESET) -> bytes:
    """JSON-encode ``payload`` and xz-compress it into artifact bytes."""

    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return lzma.compress(text.encode("utf-8"), preset=preset)


#: Safety caps for untrusted artifact files (review issue #32 security
#: follow-up on the pickle replace): a malicious xz stream can expand
#: quintillion-fold, so both the compressed input and the decompressed
#: output are bounded before anything is parsed.  The largest legitimate
#: artifact today is ~1 MB compressed / ~20 MB expanded - the caps are
#: orders of magnitude above honest use and far below OOM territory.
MAX_DECODER_MEMORY_BYTES = 256 * 1024 * 1024
MAX_COMPRESSED_BYTES = 256 * 1024 * 1024
MAX_UNCOMPRESSED_BYTES = 1 * 1024 * 1024 * 1024
_DECOMPRESS_CHUNK = 1 << 20


def _decompress_limited(data: bytes) -> bytes:
    """xz-decompress with a hard output cap (decompression-bomb guard).

    Streams through ``LZMADecompressor`` with ``max_length`` so a hostile
    stream cannot balloon past :data:`MAX_UNCOMPRESSED_BYTES` before the
    check fires; truncated and trailing-garbage streams refuse as well.
    """

    if len(data) > MAX_COMPRESSED_BYTES:
        raise GraphCodecError(f"artifact exceeds the compressed-size cap ({len(data)} > {MAX_COMPRESSED_BYTES} bytes)")
    decompressor = lzma.LZMADecompressor(memlimit=MAX_DECODER_MEMORY_BYTES)
    out = bytearray()
    view = memoryview(data)
    position = 0
    while not decompressor.eof:
        if decompressor.needs_input:
            if position >= len(view):
                raise GraphCodecError("artifact is a truncated xz stream")
            chunk = bytes(view[position : position + _DECOMPRESS_CHUNK])
            position += _DECOMPRESS_CHUNK
        else:
            chunk = b""
        out += decompressor.decompress(chunk, max_length=_DECOMPRESS_CHUNK)
        if len(out) > MAX_UNCOMPRESSED_BYTES:
            raise GraphCodecError(
                f"artifact exceeds the uncompressed-size cap (>{MAX_UNCOMPRESSED_BYTES} bytes) "
                "- refusing a possible decompression bomb"
            )
    return bytes(out)


def loads_bytes(data: bytes) -> Any:
    """Decode artifact bytes (xz-wrapped or bare JSON) back to Python."""

    if data[:6] == _XZ_MAGIC:
        try:
            data = _decompress_limited(data)
        except lzma.LZMAError as error:
            raise GraphCodecError(f"invalid xz stream or decoder memory limit exceeded: {error}") from error
    elif len(data) > MAX_UNCOMPRESSED_BYTES:
        raise GraphCodecError(f"artifact exceeds the size cap ({len(data)} > {MAX_UNCOMPRESSED_BYTES} bytes)")
    try:
        return json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise GraphCodecError(f"artifact is not valid JSON: {error}") from error


def atomic_write_bytes(target: Path, blob: bytes) -> Path:
    """Write ``blob`` to ``target`` atomically: mkstemp + fsync + replace.

    Shared by every template/capture writer (review issue #32 security
    follow-up on the fixed ``<target>.tmp`` path): the temporary file is
    created with :func:`tempfile.mkstemp` in the destination directory
    (unique, never following a pre-planted symlink), fsynced, then moved
    onto the target with :func:`os.replace`; failures clean up.
    """

    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        fd, temporary_name = tempfile.mkstemp(prefix=f".{target.stem}.", suffix=".tmp", dir=target.parent)
        temporary_path = Path(temporary_name)
        with os.fdopen(fd, "wb") as stream:
            stream.write(blob)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, target)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
    return target


def save_json_xz(payload: Any, path: str | Path, *, preset: int = _PRESET) -> Path:
    """Atomically write ``payload`` as a json+xz artifact.

    ``payload`` may be a JSON-encodable object or already-encoded artifact
    bytes (as returned by :func:`encode_document`), which are written as-is.
    """

    target = Path(path)
    blob = bytes(payload) if isinstance(payload, (bytes, bytearray)) else dumps_bytes(payload, preset=preset)
    return atomic_write_bytes(target, blob)


def read_artifact_bytes(path: str | Path) -> bytes:
    """Bound disk input before allocating the artifact body."""
    with Path(path).open("rb") as stream:
        prefix = stream.read(6)
        limit = MAX_COMPRESSED_BYTES if prefix == _XZ_MAGIC else MAX_UNCOMPRESSED_BYTES
        if os.fstat(stream.fileno()).st_size > limit:
            raise GraphCodecError(f"artifact exceeds input-size cap: {path}")
        data = prefix + stream.read(max(0, limit - len(prefix)) + 1)
    if len(data) > limit:
        raise GraphCodecError(f"artifact exceeds input-size cap: {path}")
    return data


def load_json_xz(path: str | Path) -> Any:
    """Read a json+xz artifact."""

    data = read_artifact_bytes(path)
    if not data:
        raise GraphCodecError(f"artifact is empty: {path}")
    return loads_bytes(data)
