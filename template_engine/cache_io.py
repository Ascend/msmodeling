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

"""Atomic json+xz cache for captured template-engine operation traces.

The cache deliberately stores the pair returned by :func:`capture` as one
json+xz artifact.  The operation records have already been converted to
``CapturedOp`` objects before reaching this module, so the file never
retains live dispatcher handles or real tensors - Meta placeholders only,
re-encoded through :mod:`template_engine.graph_codec` (shape/stride/dtype
identity plus alias ids; JSON cannot instantiate arbitrary objects, which
is exactly why the format replaced pickle).

A zero-byte file is treated as a cache miss, which prevents an interrupted
write from making a configuration look complete.
"""

from __future__ import annotations

from .file_lock import destination_lock

from .graph_codec import read_artifact_bytes

import os
import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .graph_codec import atomic_write_bytes, decode_capture, encode_capture


class CacheFormatError(ValueError):
    """Raised when a non-empty cache file does not contain a capture payload."""


def capture_config(key: str, cache_dir: str | os.PathLike[str], *, model_id: str | None = None) -> dict[str, Any]:
    """Read explicit key identity, supplemented by audited legacy provenance."""
    from .template_builder import parse_cache_key

    config = parse_cache_key(key)
    # Model identity lives in the capture's own sidecar (the key deliberately
    # carries no model: layouts are model-local, so the corpus is per-model).
    # The sidecar is read through the model-scoped lookup so a capture can only
    # ever be attributed to the model that owns it.
    sidecar = locate_capture(key, cache_dir, model_id=model_id)
    if sidecar is not None:
        meta_path = Path(str(sidecar) + ".meta")
        if meta_path.is_file():
            from .graph_codec import GraphCodecError, decode_document

            try:
                sidecar_meta = decode_document(read_artifact_bytes(meta_path), "model_meta/1")
            except GraphCodecError as error:
                raise CacheFormatError(
                    f"model sidecar {meta_path} is not a readable model_meta artifact: {error}"
                ) from error
            if isinstance(sidecar_meta, Mapping) and sidecar_meta.get('model_parameters') is not None:
                from .model_parameters import formula_parameters

                declared = dict(sidecar_meta['model_parameters'])
                resolved = dict(formula_parameters(sidecar_meta.get('model_config')) or ())
                if any(name in resolved and resolved[name] != value for name, value in declared.items()):
                    raise CacheFormatError('Declared model parameters disagree with the captured model config')
                config['model_parameters'] = tuple(sorted({**resolved, **declared}.items()))
            if isinstance(sidecar_meta, Mapping):
                config["decode_query_semantics"] = sidecar_meta.get("decode_query_semantics")
            if isinstance(sidecar_meta, Mapping) and sidecar_meta.get("model_id"):
                named = str(sidecar_meta["model_id"])
                if model_id is not None and named != str(model_id):
                    raise CacheFormatError(
                        f"model provenance conflict for {key}: lookup names {model_id!r} "
                        f"but the capture names {named!r}"
                    )
                config["model_id"] = named
    elif model_id is not None:
        config["model_id"] = str(model_id)
    manifest = Path(cache_dir) / "capture_manifest.json"
    if manifest.is_file():
        record = json.loads(manifest.read_text(encoding="utf-8")).get("captures", {}).get(key, {})
        mode = record.get("do_compile")
        if mode is not None:
            if not isinstance(mode, bool):
                raise CacheFormatError("manifest compile mode must be bool")
            if config["do_compile"] is not None and config["do_compile"] is not mode:
                raise CacheFormatError(f"compile provenance conflict for {key}")
            config["do_compile"] = mode
    return config


def build_cache_key(
    tp: int,
    ep: int,
    md: int,
    B: int,
    S: int,
    *,
    num_mtp_tokens: int = 0,
    dcp: int = 1,
    enable_dispatch_ffn_combine: bool = False,
    quantize_linear_action: str = "DISABLED",
    quantize_attention_action: str = "DISABLED",
    is_decode: bool = False,
    do_compile: bool | None = None,
) -> str:
    """Build the text key specified by Card 1.

    ``md`` is the cache-key spelling for ``moe_dp``.  With both quantization
    actions disabled the suffix is the fixed ``_DISABLED_L1`` recorded by the
    card (byte-identical to the original keys).  When either action is enabled
    the suffix becomes ``q{LINEAR}_{ATTENTION}_L1`` so different quantization
    families cannot collide; ``template_engine.template_builder.parse_cache_key``
    understands both spellings.

    Decode captures insert a ``_dec`` segment so they cannot collide with
    prefill keys of the same configuration; prefill keys stay byte-identical
    to the card.  ``parse_cache_key`` understands both spellings and the
    expander/coverage group keys carry the decode marker.
    """

    values = {"tp": tp, "ep": ep, "md": md, "B": B, "S": S, "mtp": num_mtp_tokens, "dcp": dcp}
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    if min(tp, ep, md, dcp) < 1:
        raise ValueError("tp, ep, md, and dcp must be positive")
    linear = str(quantize_linear_action)
    attention = str(quantize_attention_action)
    # Attention action names must not contain "_" so the q{LIN}_{ATT}_L1 split
    # is unambiguous even when the linear name does (e.g. W8A8_STATIC).
    if not linear or not attention or "/" in linear or "/" in attention or "_" in attention:
        raise ValueError(f"invalid quantization action names: {linear!r}, {attention!r}")
    if linear == "DISABLED" and attention == "DISABLED":
        quant_segment = "DISABLED"
    else:
        quant_segment = f"q{linear}_{attention}"
    parts = [f"tp{tp}", f"ep{ep}", f"md{md}", f"B{B}", f"S{S}"]
    if num_mtp_tokens > 0:
        parts.append(f"mtp{num_mtp_tokens}")
    if dcp > 1:
        parts.append(f"dcp{dcp}")
    if is_decode:
        parts.append("dec")
    if do_compile is not None:
        if not isinstance(do_compile, bool):
            raise ValueError("do_compile must be bool or None (legacy provenance unknown)")
        parts.append(f"cmp_{'on' if do_compile else 'off'}")
    parts.extend([f"dfc_{'on' if enable_dispatch_ffn_combine else 'off'}", f"{quant_segment}_L1"])
    return "_".join(parts)


def normalize_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Canonical config dict for one capture/template request — no string
    round-trip.

    Field-for-field equal to ``parse_cache_key(build_cache_key(...))`` for
    every config the key builder accepts: same coercions, same defaults
    (DCP 1, mtp 0, DISABLED actions), same compile tri-state (``None`` =
    legacy provenance unknown, no ``cmp_`` key segment).  Callers that
    need both the key and the config should use :func:`build_key_and_config`
    so both derive from this single normalization.
    """
    do_compile = config.get("do_compile")
    if do_compile is not None and not isinstance(do_compile, bool):
        raise ValueError("do_compile must be bool or None (legacy provenance unknown)")
    linear = str(config.get("quantize_linear_action", "DISABLED"))
    attention = str(config.get("quantize_attention_action", "DISABLED"))
    # Attention action names must not contain "_" so the q{LIN}_{ATT}_L1 split
    # is unambiguous even when the linear name does (e.g. W8A8_STATIC).
    if not linear or not attention or "/" in linear or "/" in attention or "_" in attention:
        raise ValueError(f"invalid quantization action names: {linear!r}, {attention!r}")
    values = {
        name: config.get(name, 0 if name == 'num_mtp_tokens' else 1)
        for name in ('TP', 'EP', 'MOE_DP', 'DCP', 'B', 'S', 'num_mtp_tokens')
    }
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    if min(values[name] for name in ("TP", "EP", "MOE_DP", "DCP", "B", "S")) < 1:
        raise ValueError("B, S, TP, EP, MOE_DP and DCP must be positive")
    for name in ('enable_dispatch_ffn_combine', 'is_decode'):
        if not isinstance(config.get(name, False), bool):
            raise ValueError(f'{name} must be bool')
    return {
        **values,
        "enable_dispatch_ffn_combine": bool(config.get("enable_dispatch_ffn_combine", False)),
        "is_decode": bool(config.get("is_decode", False)),
        "do_compile": do_compile,
        "quantize_linear_action": linear,
        "quantize_attention_action": attention,
    }


def build_key_and_config(config: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    """Normalize ``config`` once and build its cache key — no round-trip.

    Equivalent to ``parse_cache_key(build_cache_key(...))`` by construction:
    the key and the returned config both derive from the same
    :func:`normalize_config` output.
    """
    key_config = normalize_config(config)
    key = build_cache_key(
        key_config["TP"],
        key_config["EP"],
        key_config["MOE_DP"],
        key_config["B"],
        key_config["S"],
        num_mtp_tokens=key_config["num_mtp_tokens"],
        dcp=key_config["DCP"],
        enable_dispatch_ffn_combine=key_config["enable_dispatch_ffn_combine"],
        quantize_linear_action=key_config["quantize_linear_action"],
        quantize_attention_action=key_config["quantize_attention_action"],
        is_decode=key_config["is_decode"],
        do_compile=key_config["do_compile"],
    )
    return key, key_config


def cache_variant_dir(cache_dir, layers=0, sampler=False):
    """Keep structural capture options outside the legacy configuration key."""
    if cache_dir is None:
        return None
    if type(layers) is not int or layers < 0:
        raise ValueError("capture layer override must be a nonnegative integer")
    root = Path(cache_dir)
    return root / f"layers-{layers}-sampler-{int(bool(sampler))}" if layers or sampler else cache_dir


def model_slug(model_id: str | None) -> str | None:
    """Filesystem-safe single-component spelling of a model id.

    ``org/name`` becomes ``org__name`` so one directory holds one model
    without nesting, and a missing id keeps the legacy flat layout.
    """

    if not model_id:
        return None
    slug = str(model_id).strip().replace("\\", "/").strip("/").replace("/", "__")
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", slug).strip("-")
    if not slug or slug in {".", ".."}:
        raise ValueError(f"model id does not yield a usable directory name: {model_id!r}")
    digest = hashlib.sha256(str(model_id).encode("utf-8")).hexdigest()[:20]
    return slug[:32] + "--" + digest


def cache_path(key: str, cache_dir: str | os.PathLike[str], *, model_id: str | None = None) -> Path:
    """Return the ``.json.xz`` path for ``key`` without creating directories.

    ``cache_dir`` is deliberately required: there is no default corpus, and
    the historical silent fallback pointed at the legacy v3 directory.
    When ``model_id`` is given the capture belongs to that model's own
    sub-directory, so two models whose layout spellings collide can never
    overwrite each other.
    """

    if not key or Path(key).name != key or key in {".", ".."}:
        raise ValueError(f"cache key must be a single non-empty path component, got {key!r}")
    if cache_dir is None:
        raise ValueError("cache_dir is required: pass the capture corpus directory explicitly")
    root = Path(cache_dir)
    slug = model_slug(model_id)
    if slug is not None:
        root = root / slug
    return root / f"{key}.json.xz"


def _sidecar_model_id(path: Path) -> str | None:
    """Return the model a capture's sidecar names, or ``None`` when unreadable."""

    meta_path = Path(str(path) + ".meta")
    if not meta_path.is_file():
        return None
    from .graph_codec import GraphCodecError, decode_document

    try:
        meta = decode_document(read_artifact_bytes(meta_path), "model_meta/1")
    except (OSError, GraphCodecError):
        return None
    if isinstance(meta, Mapping) and meta.get("model_id"):
        return str(meta["model_id"])
    return None


def locate_capture(
    key: str,
    cache_dir: str | os.PathLike[str],
    *,
    model_id: str | None = None,
) -> Path | None:
    """Find ``key`` inside the corpus of ``model_id``.

    The cache key deliberately carries no model, so the model is part of the
    *lookup* rather than implied by the key.  ``model_id`` is therefore a
    required argument and must be passed explicitly:

    * a model id searches that model's own sub-directory, plus a legacy flat
      file that is either unprovenanced or whose sidecar names the same model;
    * ``None`` searches the legacy flat root only.

    A flat file whose sidecar names a *different* model is refused, because it
    proves the corpus is shared and the key alone cannot disambiguate it.

    Consequence, and the reason for the rule: one model can never answer
    another model's key.  Before this, ``locate_capture`` scanned *every* model
    sub-directory, so a request for GLM-4.5 at ``B16/EP8`` silently loaded the
    DeepSeek-V4-Pro capture that happened to share the key.
    """

    if model_id:
        scoped = cache_path(key, cache_dir, model_id=model_id)
        if scoped.is_file():
            sidecar = Path(str(scoped) + ".meta")
            if not sidecar.exists() or _sidecar_model_id(scoped) == str(model_id):
                return scoped
        # Old lossy directory names are readable only with exact provenance.
        legacy_slug = str(model_id).strip().replace("\\", "/").strip("/").replace("/", "__")
        legacy_slug = re.sub(r"[^A-Za-z0-9._-]+", "-", legacy_slug).strip("-")
        if legacy_slug not in ("", ".", ".."):
            legacy = Path(cache_dir) / legacy_slug / f"{key}.json.xz"
            if legacy.is_file() and _sidecar_model_id(legacy) == str(model_id):
                return legacy
    flat = cache_path(key, cache_dir)
    if not flat.is_file():
        return None
    if model_id is None:
        return flat
    named = _sidecar_model_id(flat)
    # ``named is None`` is the legacy single-model corpus: it has no provenance
    # to contradict the caller, and refusing it would strand every pre-scoping
    # capture.  A *named* mismatch is the case that must never pass.
    return flat if named is None or named == str(model_id) else None


def cache_is_valid(
    key: str,
    cache_dir: str | os.PathLike[str],
    *,
    model_id: str | None = None,
) -> bool:
    """Return whether a non-empty capture exists for ``key`` and ``model_id``."""

    path = locate_capture(key, cache_dir, model_id=model_id)
    if path is None:
        return False
    try:
        return path.stat().st_size > 0
    except OSError:
        return False


def _normalise_payload(payload: Any) -> tuple[list[Any], float]:
    if isinstance(payload, tuple) and len(payload) == 2:
        ops, exec_time = payload
    elif isinstance(payload, dict) and "ops" in payload and "exec_time" in payload:
        ops, exec_time = payload["ops"], payload["exec_time"]
    else:
        raise CacheFormatError("capture cache must contain (ops, exec_time) or {'ops', 'exec_time'}")
    if not isinstance(ops, (list, tuple)):
        raise CacheFormatError(f"capture cache ops must be a list/tuple, got {type(ops).__name__}")
    try:
        exec_time = float(exec_time)
    except (TypeError, ValueError) as error:
        raise CacheFormatError("capture cache exec_time must be numeric") from error
    return list(ops), exec_time


def load_capture(
    key: str,
    cache_dir: str | os.PathLike[str],
    *,
    model_id: str | None = None,
) -> tuple[list[Any], float] | None:
    """Load a capture; return ``None`` for missing or zero-byte files."""

    path = locate_capture(key, cache_dir, model_id=model_id)
    if path is None or path.stat().st_size <= 0:
        return None
    from .graph_codec import GraphCodecError

    try:
        payload = decode_capture(read_artifact_bytes(path))
    except (GraphCodecError, OSError, ValueError) as error:
        raise CacheFormatError(f"capture cache {path} is not a readable json+xz artifact: {error}") from error
    return _normalise_payload(payload)


def save_capture(
    key: str,
    ops: Iterable[Any],
    exec_time: float,
    cache_dir: str | os.PathLike[str],
    *,
    model_id: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Atomically save one capture and return its final path.

    Existing non-empty files are left untouched, making the operation
    idempotent and preventing a later process from replacing a completed
    capture accidentally.
    """

    path = cache_path(key, cache_dir, model_id=model_id)
    with destination_lock(path):
        if cache_is_valid(key, cache_dir, model_id=model_id) and not overwrite:
            existing = locate_capture(key, cache_dir, model_id=model_id)
            return existing if existing is not None else path
        blob = encode_capture(list(ops), float(exec_time))
        target = atomic_write_bytes(path, blob)
        if not cache_is_valid(key, cache_dir, model_id=model_id):
            raise IOError(f"cache write did not produce a non-empty file: {target}")
        return target
