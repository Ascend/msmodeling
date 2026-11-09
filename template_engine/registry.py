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

"""Automatic template-graph discovery: model identity -> template path.

Users never specify a template path.  Given the model (and optionally its
layer count), :func:`template_path_for` locates the matching template graph
by loading candidates and validating the stored ``meta`` block.

Lookup order:

1. ``template_engine/templates/<slug>_cmp_{on,off}.json.xz`` - the per-model
   compile-provenance variants; when the caller does not constrain the
   compile marker, BOTH variants are probed (existence semantics - the
   delivered artifact answers regardless of its marker, review issue #1);
2. ``template_engine/templates/<slug>.json.xz`` - the compile-agnostic
   per-model registry convention (one file per model, slug = last path
   segment of the model id, lower-cased, non-alphanumeric folded to ``_``);
3. the legacy repository-root ``template_graph.json.xz`` (current deployment
   location) - validated against ``meta`` exactly like the registry file.

A candidate that fails the identity check is skipped, never served - a
template built for another model must not silently answer lookups.  No
match returns ``None`` and callers proceed without template acceleration
(feature-invisible when absent).

Compile-marker semantics are defined ONCE in :func:`_marker_matches` and
shared by the locate and the register paths: an unspecified request
(``None``) is an existence probe, while a boolean request matches only an
artifact carrying exactly that marker (an unknown-stored marker matches no
boolean request - honest, never guessed).
"""

from __future__ import annotations

from .file_lock import destination_lock

from .graph_codec import read_artifact_bytes

import logging
import re
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
_LEGACY_PATH = Path(__file__).resolve().parent.parent / "template_graph.json.xz"
_MODEL_META_CACHE: dict[str, tuple[tuple[int, int], dict[str, Any] | None]] = {}
_GRAPH_CACHE = {}
_GRAPH_CACHE_LOCK = threading.Lock()


def load_registered_template(path):
    """Share one registered graph per file version within each worker process."""
    from .template_graph import load_template_graph

    path = Path(path).resolve()
    stamp = path.stat()
    version = (stamp.st_mtime_ns, stamp.st_size)
    with _GRAPH_CACHE_LOCK:
        cached = _GRAPH_CACHE.get(path)
        if cached is None or cached[0] != version:
            cached = (version, load_template_graph(path))
            _GRAPH_CACHE[path] = cached
        return cached[1]


def slugify_model_id(model_id: str) -> str:
    """``deepseek-ai/DeepSeek-V4-Pro`` -> ``deepseek_v4_pro``."""

    tail = str(model_id).replace("\\", "/").rstrip("/").split("/")[-1]
    return re.sub(r"[^0-9a-zA-Z]+", "_", tail).lower().strip("_")


def _marker_matches(stored: Any, requested: bool | None) -> bool:
    """Compile-marker comparison, defined once (review issue #1).

    - ``requested is None``: existence probe - any stored marker matches
      (the caller did not constrain the compile provenance);
    - ``requested`` is a bool: the stored marker must BE that bool; an
      unknown (``None``) stored marker matches no boolean request.
    """

    if requested is None:
        return True
    return stored is requested


def _model_matches(meta, model_id):
    identities = meta.get('model_ids')
    if identities is not None:
        if (
            not isinstance(identities, list)
            or not identities
            or any(not isinstance(value, str) for value in identities)
        ):
            return False
        return {value.casefold() for value in identities} == {str(model_id).casefold()}
    stored = str(meta.get('model', '')).casefold()
    if '/' in stored:
        return stored == str(model_id).casefold()
    return bool(stored) and stored == slugify_model_id(model_id)


def _load_meta(path: Path) -> dict[str, Any] | None:
    """Read only a candidate's identity metadata (review issue #31).

    Service discovery needs the lightweight identity fields (model, layer
    count, compile provenance) - NOT the derived engine.  Decoding the
    full template document materialises the group-body model objects and
    imports Torch/Transformers just to answer "which model is this", so
    this loader extracts the ``meta`` mapping straight from the JSON
    layer of the artifact (xz-decompressed, then plain ``json.loads``).
    After the #31 slim the meta mapping carries no encoded model objects,
    so no codec value restoration - and no heavy import - happens here.
    """

    key = str(path)
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    stamp = (stat.st_mtime_ns, stat.st_size)
    cached = _MODEL_META_CACHE.get(key)
    if cached is not None and cached[0] == stamp:
        return cached[1]
    from .graph_codec import loads_bytes

    payload = loads_bytes(read_artifact_bytes(path))
    container = payload.get('data') if isinstance(payload, dict) else None
    candidate = container.get('meta') if isinstance(container, dict) else None
    if not isinstance(candidate, dict):
        raise ValueError(f'Template metadata is missing or invalid: {path}')
    meta = dict(candidate)
    _MODEL_META_CACHE[key] = (stamp, meta)
    return meta


def template_path_for(
    model_id: str | None,
    num_hidden_layers: int | None = None,
    *,
    do_compile: bool | None = None,
) -> str | None:
    """Locate the template graph for ``model_id``; ``None`` when absent."""

    if not model_id:
        return None
    slug = slugify_model_id(model_id)
    # Preserve discovery of artifacts previously named from a Windows path.
    legacy_slug = re.sub(r'[^0-9a-zA-Z]+', '_', str(model_id).rstrip('/').split('/')[-1]).lower().strip('_')
    slugs = dict.fromkeys((slug, legacy_slug))
    suffixes = ('_cmp_on', '_cmp_off', '') if do_compile is None else (f"_cmp_{'on' if do_compile else 'off'}", '')
    candidates = [_TEMPLATES_DIR / f'{name}{suffix}.json.xz' for name in slugs for suffix in suffixes]
    candidates.append(_LEGACY_PATH)
    for candidate in candidates:
        if not candidate.is_file():
            continue
        meta = _load_meta(candidate)
        if meta is None:
            continue
        if not _marker_matches(meta.get("do_compile"), do_compile):
            # Distinguishable states (review issue #1): a candidate that
            # exists but carries a different compile marker is a SKIP, not
            # an absence - log it so operators can tell "no template" from
            # "template refused by provenance"; the execution layer turns a
            # provenance mismatch that reaches it into a hard refusal.
            logger.debug(
                "template candidate %s skipped: compile marker %r does not match requested %r",
                candidate.name,
                meta.get("do_compile"),
                do_compile,
            )
            continue
        if not _model_matches(meta, model_id):
            continue
        if num_hidden_layers is not None and meta.get("num_hidden_layers") != num_hidden_layers:
            continue
        return str(candidate)
    return None


def registration_path(model_id, do_compile):
    suffix = '' if do_compile is None else f"_cmp_{'on' if do_compile else 'off'}"
    return _TEMPLATES_DIR / f'{slugify_model_id(model_id)}{suffix}.json.xz'


def register_template(
    source_path: str | Path,
    model_id: str,
    num_hidden_layers: int,
    *,
    do_compile: bool | None = None,
) -> str:
    """Install ``source_path`` into the per-model registry; returns the path."""

    _TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
    _MODEL_META_CACHE.pop(str(Path(source_path)), None)
    meta = _load_meta(Path(source_path)) or {}
    if not _model_matches(meta, model_id):
        raise ValueError('template model identity does not match registration')
    if meta.get('num_hidden_layers') != num_hidden_layers:
        raise ValueError('template layer count does not match registration')
    mode = meta.get("do_compile")
    if do_compile is not None and not _marker_matches(mode, do_compile):
        raise ValueError("template compile provenance does not match registration")
    if mode is not None and not isinstance(mode, bool):
        raise ValueError("invalid template compile provenance")
    destination = registration_path(model_id, mode)
    from .graph_codec import atomic_write_bytes

    with destination_lock(destination):
        atomic_write_bytes(destination, read_artifact_bytes(Path(source_path)))
        _MODEL_META_CACHE.pop(str(destination), None)
    return str(destination)
