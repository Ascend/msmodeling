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

"""Build measured attention-latency calibration models from kernel CSVs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml

from ...device import DeviceProfile
from ..profiling_database.profiling_data_source import _parse_shape_str, _parse_str_list
from .profile import validate_mm_profile_document
from .sqlite_profile import load_sqlite_document, write_sqlite_profile

GENERATOR_NAME = "tensor_cast.analytic_calibration.attention_build"


def _is_lfs_pointer(path: Path) -> bool:
    return path.read_bytes()[:64].startswith(b"version https://git-lfs.github.com/spec/v1")


def _database_digest(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.name):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _ints(value: Any) -> list[int]:
    if value is None:
        return []
    text = str(value).strip().strip('"')
    try:
        return [int(item) for item in text.replace(";", ",").split(",") if item.strip()]
    except ValueError:
        return []


def _scalar(row: Mapping[str, Any], name: str) -> Optional[int]:
    value = row.get(name)
    try:
        return int(float(value)) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _runtime_values(row: Mapping[str, Any], name: str) -> list[int]:
    value = row.get(name)
    if value in (None, ""):
        return []
    text = str(value).strip().strip('"')
    values: list[int] = []
    for item in text.replace(";", ",").split(","):
        try:
            if item.strip():
                values.append(int(float(item.strip())))
        except ValueError:
            return []
    return values


def _q_shape(row: Mapping[str, Any]) -> tuple[int, ...]:
    shapes = _parse_shape_str(str(row.get("Input Shapes", "")), preserve_empty_slots=False)
    return shapes[0] if shapes else ()


def _row_features(row: Mapping[str, Any]) -> dict[str, Any]:
    q_shape = _q_shape(row)
    runtime_avg = _scalar(row, "Runtime avg_seq_len")
    runtime_kv = _runtime_values(row, "Runtime actual_seq_lengths_kv_values")
    if runtime_avg is None and runtime_kv:
        runtime_avg = sum(runtime_kv) // len(runtime_kv)
    heads = _scalar(row, "Runtime num_heads")
    kv_heads = _scalar(row, "Runtime num_key_value_heads")
    sparse_mode = _scalar(row, "Runtime sparse_mode")
    block_size = _scalar(row, "Runtime block_size")
    q_tokens = (q_shape[0] * q_shape[-2]) if len(q_shape) >= 4 else (q_shape[0] if q_shape else None)
    head_dim = q_shape[-1] if q_shape else None
    features: dict[str, Any] = {}
    phase = None
    if runtime_avg is not None:
        # CSVs do not consistently carry a phase column.  Decode rows have a
        # one-token query shape; prefill rows have a larger query token axis.
        phase = "decode" if len(q_shape) >= 4 and q_shape[-2] == 1 else "prefill"
    for key, value in (
        ("q_tokens", q_tokens),
        ("effective_kv_len", runtime_avg),
        ("heads", heads),
        ("kv_heads", kv_heads),
        ("head_dim", head_dim),
        ("sparse_mode", sparse_mode),
        ("block_size", block_size),
        ("phase", phase),
    ):
        if value is not None and (not isinstance(value, (int, float)) or value > 0):
            features[key] = value
    layout = str(row.get("Runtime input_layout", "")).strip()
    cache_mode = str(row.get("Runtime kv_cache_mode", "")).strip()
    if layout:
        features["input_layout"] = layout
    if cache_mode:
        features["kv_cache_mode"] = cache_mode
    return features


def _load_rows(path: Path, kernel_type: str) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    report: dict[str, Any] = {"kernel": kernel_type, "path": path.name}
    if not path.is_file():
        return [], {**report, "status": "missing", "accepted_rows": 0}, "bfloat16"
    if _is_lfs_pointer(path):
        return [], {**report, "status": "lfs_pointer", "accepted_rows": 0}, "bfloat16"
    report["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    rows: list[dict[str, Any]] = []
    latency_columns: set[str] = set()
    dtype_name = "bfloat16"
    rejected = 0
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row_index, row in enumerate(csv.DictReader(handle), start=1):
            try:
                latency_column = "Average Duration(us)"
                raw_latency = row.get(latency_column, "")
                if raw_latency in (None, ""):
                    latency_column = "Profiling Average Duration(us)"
                    raw_latency = row.get(latency_column, "")
                latency = float(raw_latency)
                features = _row_features(row)
                dtypes = _parse_str_list(str(row.get("Input Data Types", "")))
                if dtypes:
                    dtype_name = {
                        "DT_BF16": "bfloat16",
                        "DT_FLOAT16": "float16",
                        "DT_FLOAT": "float32",
                    }.get(dtypes[0].strip().upper(), dtype_name)
                if not math.isfinite(latency) or latency <= 0 or not features:
                    raise ValueError("missing positive latency or runtime shape features")
                latency_columns.add(latency_column)
                rows.append(
                    {"features": features, "latency_us": latency, "latency_column": latency_column, "row": row_index}
                )
            except (TypeError, ValueError):
                rejected += 1
    return (
        rows,
        {
            **report,
            "status": "loaded" if rows else "no_compatible_rows",
            "accepted_rows": len(rows),
            "rejected_rows": rejected,
            "latency_columns": sorted(latency_columns),
        },
        dtype_name,
    )


def _generic_fia_model(model: Mapping[str, Any]) -> dict[str, Any]:
    """Project FIA measurements onto features available to generic attention.

    FIA/MLA rows may contain runtime-only fields such as block size and input
    layout.  The generic ``tensor_cast.attention[_quant]`` signature cannot
    provide those fields, so it receives a separate exact-point model using
    only the common semantic dimensions.
    """
    curves = []
    for curve in model["curves"]:
        features = curve["features"]
        required = ("q_tokens", "effective_kv_len", "head_dim", "heads", "kv_heads", "phase")
        if not all(key in features for key in required):
            continue
        curves.append(
            {
                **curve,
                "features": {
                    "q_tokens": features["q_tokens"],
                    "kv_len": features["effective_kv_len"],
                    "head_dim": features["head_dim"],
                    "heads": features["heads"],
                    "kv_heads": features["kv_heads"],
                    "phase": features["phase"],
                },
            }
        )
    return {
        **model,
        "id": f"{model['id']}-generic",
        "tc_ops": ["tensor_cast.attention.default", "tensor_cast.attention_quant.default"],
        "curves": curves,
    }


def build_attention_profile(
    database_path: str | Path,
    *,
    targets: list[str],
    device_name: str,
    software_stack: str,
    base_profile: str | Path | None = None,
    profile_id: str | None = None,
    audit_path: str | Path,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    database = Path(database_path)
    if device_name not in DeviceProfile.all_device_profiles:
        raise ValueError(f"unknown device profile {device_name!r}")
    rows: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    source_paths: list[Path] = []
    models: list[dict[str, Any]] = []
    for target in targets:
        try:
            model_id, kernel_type = target.split("=", 1)
        except ValueError as error:
            raise ValueError("--target must use id=KernelType") from error
        path = database / f"{kernel_type}.csv"
        loaded, report, dtype_name = _load_rows(path, kernel_type)
        reports.append(report)
        if path.is_file():
            source_paths.append(path)
        if not loaded:
            raise ValueError(f"kernel {kernel_type} is unavailable ({report['status']})")
        rows.extend(loaded)
        tc_ops = (
            ["tensor_cast.mla_sparse_attention.default", "tensor_cast.mla_sparse_attention_quant.default"]
            if kernel_type == "SparseFlashAttention"
            else [
                "tensor_cast.multihead_latent_attention.default",
                "tensor_cast.multihead_latent_attention_quant.default",
            ]
            if kernel_type == "FusedInferAttentionScore"
            else []
        )
        if kernel_type == "FusedInferAttentionScore":
            # FIA is also the physical kernel behind the generic attention
            # TensorCast ops.  Keep those semantic ops in the same measured
            # model so the target catalog and runtime agree.
            tc_ops.extend(["tensor_cast.attention.default", "tensor_cast.attention_quant.default"])
        models.append(
            {
                "id": model_id,
                "kernel_type": kernel_type,
                "tc_ops": tc_ops,
                "dtype": dtype_name,
                "curves": [
                    {
                        "features": item["features"],
                        "latency_us": item["latency_us"],
                        "confidence": "measured",
                        "source_row": item["row"],
                        "latency_column": item["latency_column"],
                    }
                    for item in loaded
                ],
            }
        )
        if kernel_type == "FusedInferAttentionScore":
            generic_model = _generic_fia_model(models[-1])
            if generic_model["curves"]:
                models.append(generic_model)
    output_id = profile_id or f"{device_name.lower()}-{software_stack}-attention"
    audit = {
        "version": 1,
        "generator": GENERATOR_NAME,
        "device": device_name,
        "software_stack": software_stack,
        "database": database.name,
        "database_digest": _database_digest(source_paths),
        "kernels": reports,
    }
    rendered = yaml.safe_dump(audit, sort_keys=False, allow_unicode=True)
    profile: dict[str, Any]
    database_digest = _database_digest(source_paths)
    audit_sha256 = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
    attention_source = {
        "database": database.name,
        "database_digest": database_digest,
        "audit": Path(audit_path).name,
        "audit_sha256": audit_sha256,
    }
    if base_profile is not None:
        profile = load_sqlite_document(base_profile)
        if profile.get("version") != 3 or profile.get("device") != device_name:
            raise ValueError("base profile must be a SQLite profile for the same device")
        if profile.get("software_stack") != software_stack:
            raise ValueError("base profile software stack does not match attention database")
    else:
        profile = {
            "version": 3,
            "id": output_id,
            "device": device_name,
            "software_stack": software_stack,
            "source": attention_source,
            "mm_models": [],
        }
    profile["attention_source"] = attention_source
    profile["calibration_sources"] = dict(profile.get("calibration_sources", {}))
    profile["calibration_sources"]["attention"] = attention_source
    # Rebuilding an existing target must refresh its measured curves and
    # semantic op declarations, while unrelated attention families remain
    # intact.  This is also what makes a profile reproducibly upgradeable
    # after the runtime semantic mapping changes.
    rebuilt_ids = {model["id"] for model in models}
    existing_attention = [
        item
        for item in profile.get("attention_models", [])
        if isinstance(item, Mapping) and item.get("id") not in rebuilt_ids
    ]
    profile["attention_models"] = existing_attention + models
    profile.setdefault("mm_models", [])
    profile.setdefault("gmm_models", [])
    profile.setdefault("communication_models", [])
    validate_mm_profile_document(profile)
    audit["audit_sha256"] = audit_sha256
    return profile, audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build measured attention-latency calibration models.")
    parser.add_argument("database", type=Path)
    parser.add_argument("--device", dest="device_name", required=True)
    parser.add_argument("--software-stack", required=True)
    parser.add_argument("--target", action="append", required=True, help="id=KernelType; repeatable")
    parser.add_argument("--base-profile", type=Path)
    parser.add_argument("--profile-id")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    args = parser.parse_args(argv)
    profile, audit = build_attention_profile(
        args.database,
        targets=args.target,
        device_name=args.device_name,
        software_stack=args.software_stack,
        base_profile=args.base_profile,
        profile_id=args.profile_id,
        audit_path=args.audit_output,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(yaml.safe_dump(audit, sort_keys=False, allow_unicode=True), encoding="utf-8")
    write_sqlite_profile(profile, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
