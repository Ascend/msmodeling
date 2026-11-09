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

"""Lazy template storage and compatibility graph/timing queries.

``resolve_config`` delegates graph supply to ``template_lookup.lookup_ops``
and adds performance evaluation. Storage keeps group bodies lazy while
preserving captured tensor/reference identity. Legacy CLI parsing delegates
to the unified CLI entrypoints.
"""

from __future__ import annotations

from .graph_codec import read_artifact_bytes

import json
import sys
import time
import threading

from pathlib import Path
from typing import Any, Callable, Mapping

_BODY_LOCK = threading.RLock()

CaptureFn = Callable[..., tuple[list[Any], float]]


def resolve_config(
    config: Mapping[str, Any],
    *,
    template: Mapping[str, Any] | None = None,
    template_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    allow_capture: bool = True,
    capture_fn: CaptureFn | None = None,
    num_hidden_layers: int | None = None,
) -> dict[str, Any]:
    """Resolve one config to a graph through the three tiers.

    Returns graph provenance, op count and acquisition wall time. Native
    capture timing is retained only for captured/cache-sourced graphs. With
    ``allow_capture=False`` an uncovered config returns
    ``covered=False`` + ``blocker`` instead of capturing (planning mode).
    """

    from .template_lookup import lookup_ops

    started = time.perf_counter()
    supplied = lookup_ops(
        config,
        template=template,
        template_path=template_path,
        cache_dir=cache_dir,
        allow_capture=allow_capture,
        layers=num_hidden_layers or 0,
        capture_fn=capture_fn,
        persist=True,
    )
    result = {name: supplied[name] for name in ('key', 'source')}
    result['covered'] = supplied['source'] != 'uncovered'
    if result['covered']:
        if supplied['source'] == 'template':
            result['coverage'] = supplied['coverage']['verdict']
        else:
            timing = {'capture': float(supplied['exec_time_s'])}
        result['op_count'] = len(supplied['ops'])
        if supplied['source'] != 'template':
            result['execution_time_s'] = timing
    else:
        result['blocker'] = supplied['blocker']
    result['wall_time_s'] = round(time.perf_counter() - started, 3)
    return result


def _load_template_any(path: str | Path):
    """Load a template artifact (json+xz, transparently handling bare JSON).

    The delivery artifact ships xz-compressed JSON (review issue #32:
    JSON cannot instantiate arbitrary objects, which is exactly why the
    format replaced pickle).

    Lazy-shard format (``lazy_shards_json_v1``): group bodies travel as
    JSON text blobs (bundled files may use UTF-8 bytes) (``body_blobs``) while the outer document
    carries only the metadata manifest; the single outer xz stream is the
    only compression layer, so the heavy cross-body repetition actually
    compresses.  Group keys may be tuples - they travel as ``repr`` strings
    and are restored on load.  A blob store that lost entries or bytes
    fails LOUDLY here (``body_blob_count`` / ``body_blob_digest`` stamps)
    - never as a silent 0-op expansion.
    """

    from .graph_codec import decode_document

    template = decode_document(read_artifact_bytes(Path(path)), "template_lazy/1")
    if isinstance(template, dict) and template.get("format") == _LAZY_SHARD_FORMAT:
        engine = template.get("engine")
        if isinstance(engine, dict):
            blobs = template.pop("body_blobs", {})
            blobs = {_restore_group_key(key): value for key, value in blobs.items()}
            engine["_body_blobs"] = blobs
            engine["_body_blob_codec"] = template.pop("body_blob_codec", _BODY_CODEC_JSON)
            groups = engine.get("groups")
            if isinstance(groups, dict):
                engine["groups"] = {_restore_group_key(key): value for key, value in groups.items()}
            _validate_body_blob_store(blobs, template)
    return template


def load_template_engine(path: str | Path) -> dict:
    """Load a template artifact and return its ENGINE mapping.

    The single loading boundary for the supply entries (review issue #22):
    handles JSON+xz lazy group bodies and the
    template_graph wrapper shape (``{"engine": ...}``) - every entry sees the
    same engine mapping regardless of the delivered file shape, so a legal
    artifact can no longer be misread as empty or lose its tensor references.
    """

    template = _load_template_any(path)
    if isinstance(template, dict) and isinstance(template.get("engine"), dict):
        template = template["engine"]
    if not isinstance(template, dict):
        raise TypeError(f"Template artifact root must be a dict, got {type(template).__name__}")
    return template


_LAZY_SHARD_FORMAT = "lazy_shards_json_v1"
_LAZY_BODY_KEYS = ("entries",)
# Body-blob codec: "json" = UTF-8 JSON bytes; the outer xz stream is the
# only compression layer, so the heavy cross-body repetition compresses.
_BODY_CODEC_JSON = "json"


def _json_group_key(key: Any) -> str:
    """JSON dict keys must be strings; tuple group keys travel as repr."""

    return key if isinstance(key, str) else repr(key)


def _restore_group_key(key: Any) -> Any:
    """Restore a tuple group key from its repr form; strings pass through."""

    import ast

    if isinstance(key, str) and key.startswith("(") and key.endswith(")"):
        try:
            restored = ast.literal_eval(key)
        except (ValueError, SyntaxError):
            return key
        return restored
    return key


def _body_blob_digest(blobs: Mapping[Any, bytes | str]) -> str:
    """Stable content digest of the body-blob store (canonical key order).

    Keys are normalised through :func:`_json_group_key` so the artifact
    writer's repr-string keys and the loader's restored tuple keys hash
    identically.
    """

    import hashlib

    hasher = hashlib.sha256()
    for key in sorted(blobs, key=_json_group_key):
        blob = blobs[key]
        if isinstance(blob, str):
            blob = blob.encode("utf-8")
        hasher.update(_json_group_key(key).encode("utf-8"))
        hasher.update(len(blob).to_bytes(8, "little"))
        hasher.update(blob)
    return hasher.hexdigest()


def _validate_body_blob_store(blobs: Mapping[Any, bytes | str], container: Mapping[str, Any]) -> None:
    """Fail loudly on a blob store that lost entries or bytes.

    xz's embedded CRC already catches bit flips and truncation of the FILE;
    these stamps additionally catch a re-pickled container whose blobs were
    dropped or altered after the build - the failure mode that would
    otherwise surface as a silent 0-op expansion.
    """

    expected_count = container.get("body_blob_count")
    if expected_count is not None and int(expected_count) != len(blobs):
        raise ValueError(
            "template body blob store corrupt: "
            f"expected {expected_count} body blobs, found {len(blobs)} - "
            "the artifact is incomplete; rebuild it via build_template_graph"
        )
    expected_digest = container.get("body_blob_digest")
    if expected_digest is not None and _body_blob_digest(blobs) != expected_digest:
        raise ValueError(
            "template body blob store corrupt: content digest mismatch - "
            "the artifact's bodies were altered after the build; "
            "rebuild it via build_template_graph"
        )


def _slim_template_meta(template_meta: Any) -> tuple[dict[str, Any], dict[str, int]]:
    """Copy ``template_meta`` minus the derived diagnostic bulk.

    ``coverage.failed_formulas`` stays (the runtime interpolated gate reads
    it as the per-mode fallback, coverage.py); families keep their identity
    fields (configs/members) and lose only the per-op ``op_count`` records -
    derived data with no runtime reader (the builder's in-memory report
    keeps everything).
    """

    meta = dict(template_meta) if isinstance(template_meta, Mapping) else {}
    coverage = dict(meta.get("coverage")) if isinstance(meta.get("coverage"), Mapping) else {}
    counts: dict[str, int] = {}
    notes = coverage.pop("notes", None)
    if isinstance(notes, list):
        counts["coverage_notes"] = len(notes)
    source_keys = coverage.pop("source_keys", None)
    if isinstance(source_keys, list):
        counts["coverage_source_keys"] = len(source_keys)
    families = coverage.get("families")
    if isinstance(families, list):
        slimmed_families = []
        op_count_entries = 0
        for family in families:
            if isinstance(family, Mapping) and "op_count" in family:
                slimmed_families.append({k: v for k, v in family.items() if k != "op_count"})
                op_count_entries += 1
            else:
                slimmed_families.append(family)
        coverage["families"] = slimmed_families
        counts["families_op_count_entries"] = op_count_entries
    meta["coverage"] = coverage
    return meta, counts


def _slim_meta(meta: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, int]]:
    """Copy the graph ``meta`` minus the runtime-dead bulk.

    Service discovery only reads the lightweight identity fields (model,
    layer count, compile provenance, decode request semantics), so the
    service file carries those plus counts:

    - ``model_meta`` is DROPPED: the virtual-runner consumer that used to
      read it via ``load_graph_model_meta`` is deleted, no current runtime
      path reads the mapping, and keeping full model objects in the service
      artifact forced every registry metadata load to import
      Torch/Transformers just to decode it.  The audit-provenance copy
      lives in the capture sidecar (``<cache-key>.json.xz.meta``), which
      keeps its full ``model_meta`` (review issue #31).
    - the corpus key list is derivable by scanning the capture cache and
      build notes are diagnostic, so both travel as counts only.
    """

    out = dict(meta)
    counts: dict[str, int] = {}
    model_meta = out.pop("model_meta", None)
    if model_meta is not None:
        counts["meta_model_meta"] = 1
    corpus = out.pop("corpus", None)
    if isinstance(corpus, list):
        out["corpus_count"] = len(corpus)
        counts["meta_corpus_keys"] = len(corpus)
    build_notes = out.pop("build_notes", None)
    if isinstance(build_notes, list):
        out["build_notes_count"] = len(build_notes)
        counts["meta_build_notes"] = len(build_notes)
    return out, counts


_FORMULA_FIELDS = ("args_f", "kwargs_f", "out_f")


def _pack_column(values):
    """Factor a JSON column without changing any scalar or container type."""
    first = values[0]
    if not isinstance(first, (list, dict)) and all(type(value) is type(first) for value in values):
        if all(value == first for value in values):
            return ['constant', first]
        if type(first) is int:
            return ['delta', [first, *[b - a for a, b in zip(values, values[1:])]]]
    if isinstance(first, list) and all(isinstance(value, list) and len(value) == len(first) for value in values):
        return ['list', [_pack_column([value[i] for value in values]) for i in range(len(first))]]
    if isinstance(first, dict) and all(isinstance(value, dict) and tuple(value) == tuple(first) for value in values):
        return ['dict', {key: _pack_column([value[key] for value in values]) for key in first}]
    return ['values', values]


def _unpack_column(node, count):
    kind, value = node
    if kind == 'constant':
        if isinstance(value, (list, dict)):
            raise ValueError('invalid constant template column')
        return [value] * count
    if kind == 'delta':
        from itertools import accumulate

        if not all(type(item) is int for item in value):
            raise ValueError('invalid integer template column')
        result = list(accumulate(value))
    elif kind == 'list':
        columns = [_unpack_column(item, count) for item in value]
        return [[column[i] for column in columns] for i in range(count)]
    elif kind == 'dict':
        columns = {key: _unpack_column(item, count) for key, item in value.items()}
        return [{key: column[i] for key, column in columns.items()} for i in range(count)]
    elif kind == 'values':
        result = value
    else:
        raise ValueError('unsupported template column encoding')
    if len(result) != count:
        raise ValueError('template column length does not match operator order')
    return result


def _pack_entry_columns(rows, field_sets):
    """Group storage by schema/op; order retains every original operator position."""
    groups, group_ids, order = [], {}, []
    for row in rows:
        fields = field_sets[row[0]]
        key = (row[0], row[1 + fields.index('func')])
        if key not in group_ids:
            group_ids[key] = len(groups)
            groups.append({'fields': row[0], 'columns': [[] for _ in fields]})
        index = group_ids[key]
        order.append(index)
        for column, value in zip(groups[index]['columns'], row[1:], strict=True):
            column.append(value)
    for group in groups:
        group['columns'] = [_pack_column(column) for column in group['columns']]
    return {'order': order, 'groups': groups}


def _unpack_entry_columns(packed):
    groups, order = packed['groups'], packed['order']
    counts = [0] * len(groups)
    for index in order:
        if type(index) is not int or not 0 <= index < len(groups):
            raise ValueError('invalid template column group reference')
        counts[index] += 1
    # Counts come from explicit operator positions, never a claimed repeat count.
    columns = [
        [_unpack_column(column, count) for column in group['columns']]
        for group, count in zip(groups, counts, strict=True)
    ]
    offsets, rows = [0] * len(groups), []
    for index in order:
        rows.append([groups[index]['fields'], *[column[offsets[index]] for column in columns[index]]])
        offsets[index] += 1
    return rows


def _encode_body_blob(body: dict[str, Any]) -> str:
    """Store JSON text with shared formula/field tables and positional tensor rows.

    Only serialization changes: each operator retains its own position and
    tensor references. The graph codec reconstructs independent formula
    containers on decode, while tensor identity stays in its tensor table.
    """
    from .graph_codec import encode_value

    tensors: list[dict[str, Any]] = []
    encoded = encode_value(body, tensors, {})
    formulas: list[Any] = []
    formula_ids: dict[str, int] = {}
    for entry in encoded.get("entries", []):
        for field in _FORMULA_FIELDS:
            if field not in entry:
                continue
            value = entry[field]
            signature = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            if signature not in formula_ids:
                formula_ids[signature] = len(formulas)
                formulas.append(value)
            entry[field] = formula_ids[signature]
    entry_fields: list[tuple[str, ...]] = []
    field_ids: dict[tuple[str, ...], int] = {}
    rows = []
    for entry in encoded.get("entries", []):
        fields = tuple(entry)
        if fields not in field_ids:
            field_ids[fields] = len(entry_fields)
            entry_fields.append(fields)
        rows.append([field_ids[fields], *entry.values()])
    encoded["entries"] = _pack_entry_columns(rows, entry_fields)
    payload = {
        "format": "group_body_json/3",
        "body": encoded,
        "tensors": [[tensor["s"], tensor["st"], tensor["d"]] for tensor in tensors],
        "formulas": formulas,
        "entry_fields": entry_fields,
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _decode_body_blob(blob: bytes | str) -> dict[str, Any]:
    """Decode bundled JSON bodies and the formula-table representation."""
    from .graph_codec import _materialise_tensor, decode_value

    payload = json.loads(blob)
    if not isinstance(payload, dict) or payload.get("format") not in ("group_body_json/2", "group_body_json/3"):
        raise ValueError("unsupported template group body JSON format")
    encoded = payload["body"]
    field_sets = payload["entry_fields"]
    entries = []
    rows = _unpack_entry_columns(encoded["entries"]) if payload["format"] == "group_body_json/3" else encoded["entries"]
    for row in rows:
        index = row[0]
        if type(index) is not int or not 0 <= index < len(field_sets):
            raise ValueError("invalid template entry field reference")
        entries.append(dict(zip(field_sets[index], row[1:], strict=True)))
    encoded["entries"] = entries
    payload["tensors"] = [dict(zip(("s", "st", "d"), row, strict=True)) for row in payload["tensors"]]
    formulas = payload["formulas"]
    for entry in entries:
        for field in _FORMULA_FIELDS:
            if field in entry:
                index = entry[field]
                if type(index) is not int or not 0 <= index < len(formulas):
                    raise ValueError("invalid template formula table reference")
                entry[field] = formulas[index]
    tensors = [_materialise_tensor(entry) for entry in payload.get("tensors", [])]
    body = decode_value(encoded, tensors)
    if not isinstance(body, dict):
        raise ValueError("group body blob did not decode to a dict")
    return body


def _encode_template_lazy(template: dict) -> bytes:
    """Slim single-file save: metadata and compact lazy JSON bodies, one xz stream.

    Coverage and expansion decisions read only a group's metadata
    (``served_keys`` / ``mode_meta`` / ``trained_mtp``) and the
    ``template_meta.coverage`` disclosure minus its diagnostic bulk; the
    structure bodies are needed only for the few groups an optimizer run
    actually expands.  The writer therefore:

    1. strips the derived diagnostic bulk from the SAVED artifact
       (``coverage.families[].op_count`` per-op records, ``coverage.notes``,
       ``coverage.source_keys``, ``meta.corpus`` list, ``meta.build_notes``
       - each remembered as a count in ``template_meta.slim_stamp``); the
       in-memory build report keeps everything;
    2. stores each group body as JSON text with deduplicated formulas (codec
       ``body_blob_codec="json"``) and xz-compresses the WHOLE document once
       - the single stream exploits the heavy cross-body repetition;
    3. stamps the blob store (count + sha256 digest) so a corrupted or
       incomplete artifact fails LOUDLY at load, never as a silent 0-op.

    The bodies stay lazy: the outer document materialises only the manifest
    plus atomic text strings; :func:`ensure_group_body` attaches a group's
    body on first structure access.  Tuple group keys travel as repr
    strings (JSON dict keys must be strings) and are restored on load.
    """

    engine = template["engine"]
    manifest: dict[str, Any] = {}
    blobs: dict[str, str] = {}
    for key, group in engine["groups"].items():
        ensure_group_body(engine, key)
        body = {k: group[k] for k in _LAZY_BODY_KEYS if k in group}
        if body:
            blobs[_json_group_key(key)] = _encode_body_blob(body)
            manifest[_json_group_key(key)] = {k: v for k, v in group.items() if k not in body}
        else:
            manifest[_json_group_key(key)] = dict(group)
    container = dict(template)
    container["engine"] = {k: v for k, v in engine.items() if k not in ("_body_blobs", "_body_blob_codec")}
    container["engine"]["groups"] = manifest
    container["engine"]["template_meta"], tm_counts = _slim_template_meta(engine.get("template_meta"))
    meta_counts: dict[str, int] = {}
    if isinstance(template.get("meta"), Mapping):
        container["meta"], meta_counts = _slim_meta(template["meta"])
    container["format"] = _LAZY_SHARD_FORMAT
    container["body_blobs"] = blobs
    container["body_blob_codec"] = _BODY_CODEC_JSON
    container["body_blob_count"] = len(blobs)
    container["body_blob_digest"] = _body_blob_digest(blobs)
    container["engine"]["template_meta"]["slim_stamp"] = {
        "format": "slim_single_file_json_v1",
        "body_blob_codec": _BODY_CODEC_JSON,
        "dropped": [
            "engine.template_meta.coverage.families[].op_count",
            "engine.template_meta.coverage.notes",
            "engine.template_meta.coverage.source_keys",
            "meta.corpus (kept as corpus_count)",
            "meta.build_notes (kept as build_notes_count)",
            "meta.model_meta (audit copy lives in the capture sidecar - review issue #31)",
        ],
        **tm_counts,
        **meta_counts,
    }
    from .graph_codec import PRESET_MAX, encode_document

    return encode_document(container, "template_lazy/1", preset=PRESET_MAX)


def _save_template_lazy(template: dict, path: str | Path) -> None:
    """Publish an encoded lazy artifact only after preparation succeeds."""
    from .graph_codec import atomic_write_bytes

    atomic_write_bytes(Path(path), _encode_template_lazy(template))


def ensure_group_body(template: Any, group_key: Any) -> None:
    """Load a fixed entry body and reject retired structural formats."""
    if not isinstance(template, Mapping):
        return
    with _BODY_LOCK:
        groups = template.get('groups')
        group = groups.get(group_key) if isinstance(groups, dict) else None
        if not isinstance(group, dict):
            return
        blobs = template.get('_body_blobs')
        blob = blobs.get(group_key) if isinstance(blobs, dict) else None
        if blob is not None:
            group.update(_decode_body_blob(blob))
        if any(
            name in group
            for name in (
                'program',
                'fold',
                'prefix',
                'prefix_segments',
                'prefix_pre',
                'chunk',
                'slice',
                'suffix_mtp0',
                'mtp_structure',
            )
        ) or any(
            entry.get('type') in ('structured', 'moe_fold') or entry.get('condition_formula') is not None
            for entry in group.get('entries', [])
        ):
            raise ValueError('Unsupported dynamic template format; rebuild with the official template CLI or Skill')
        if blob is not None:
            blobs.pop(group_key, None)


def _save_template_compressed(template: Any, path: str | Path) -> None:
    """JSON + xz-compress the template atomically (single stream).

    Same outer ``template_lazy/1`` document kind as the slim writer - a
    fully materialised engine simply carries no ``body_blobs``.
    """

    from .graph_codec import PRESET_MAX, atomic_write_bytes, encode_document

    atomic_write_bytes(Path(path), encode_document(template, "template_lazy/1", preset=PRESET_MAX))


def load_or_build_template(
    template_path: str | Path | None,
    cache_dir: str | Path | None = None,
    *,
    rebuild: bool = False,
) -> tuple[Any, str | Path | None]:
    """Load the template pickle, building it once from the corpus if absent."""

    if template_path is not None and Path(template_path).is_file() and not rebuild:
        # The bootstrap entry goes through the same unified loading boundary
        # as every other supply entry (review issue #22 supplement): a
        # delivered wrapper artifact must unwrap to its engine here, or the
        # CLI reads the outer mapping, finds no groups and refuses every
        # config with blocker=empty.
        return load_template_engine(template_path), template_path

    from .template_builder import build_fixed_template, discover_cache_keys, load_captures_from_cache

    keys = discover_cache_keys(str(cache_dir) if cache_dir else None)
    captures = load_captures_from_cache(keys, cache_dir)
    template, _report = build_fixed_template(captures)
    if template_path is not None:
        _save_template_compressed(template, template_path)
    return template, template_path


_CONFIG_KEY_ALIASES = {
    "b": "B",
    "s": "S",
    "tp": "TP",
    "ep": "EP",
    "moe-dp": "MOE_DP",
    "moe_dp": "MOE_DP",
    "dcp": "DCP",
    "decode": "is_decode",
    "compile": "do_compile",
    "mtp": "num_mtp_tokens",
    "dfc": "enable_dispatch_ffn_combine",
    "quant-linear": "quantize_linear_action",
    "quant_linear": "quantize_linear_action",
    "quant-attention": "quantize_attention_action",
    "quant_attention": "quantize_attention_action",
}


def parse_config_spec(spec: str) -> dict[str, Any]:
    """Parse ``B=2,S=512,tp=8`` into a config dict."""

    config: dict[str, Any] = {}
    for part in spec.split(","):
        if not part.strip():
            continue
        raw_key, _, raw_value = part.partition("=")
        canonical = _CONFIG_KEY_ALIASES.get(raw_key.strip().lower())
        if canonical is None:
            raise ValueError(f"unknown config key {raw_key!r} in {spec!r}")
        value: Any = raw_value.strip()
        if canonical in ("B", "S", "TP", "EP", "MOE_DP", "DCP", "num_mtp_tokens"):
            value = int(value)
        elif canonical in ("is_decode", "enable_dispatch_ffn_combine", "do_compile"):
            value = value.lower() in ("1", "true", "yes", "on")
        config[canonical] = value
    return config


def main(argv=None):
    from cli.inference.template_graph_diagnostics import resolve_main

    return resolve_main(argv)


if __name__ == "__main__":
    sys.exit(main())
