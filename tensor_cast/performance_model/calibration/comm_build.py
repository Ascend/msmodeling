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

"""Build collective latency curves from HCCL microbenchmark CSVs."""

from __future__ import annotations

import argparse
import csv
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import yaml

from .profile import validate_mm_profile_document
from .sqlite_profile import load_sqlite_document, write_sqlite_profile

GENERATOR_NAME = "tensor_cast.analytic_calibration.comm_build"
GENERATOR_VERSION = "1"
_COLLECTIVES = {
    "all_reduce": ("hcom_allReduce_.csv", "tensor_cast.all_reduce.default"),
    "all_gather": ("hcom_allGather_.csv", "tensor_cast.all_gather.default"),
    "reduce_scatter": ("hcom_reduceScatter_.csv", "tensor_cast.reduce_scatter.default"),
    "all_to_all": ("hcom_alltoallv_.csv", "tensor_cast.all_to_all.default"),
}
_DTYPES = {"DT_BF16": "bfloat16", "DT_FP16": "float16", "DT_FLOAT": "float32"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _directory_digest(paths: Iterable[Path], directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(directory).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _as_positive_int(value: Any, location: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{location} must be an integer") from error
    if parsed <= 0:
        raise ValueError(f"{location} must be positive")
    return parsed


def _load_collective(path: Path, collective: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"communication CSV does not exist: {path}")
    required = {"message_bytes", "num_devices", "dtype", "topology_tier", "Duration(us)"}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            missing = sorted(required - set(reader.fieldnames or []))
            raise ValueError(f"communication CSV {path.name} lacks required columns {missing}")
        rows = list(reader)

    grouped: dict[tuple[str, int, int], list[list[float]]] = defaultdict(list)
    rejected: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int, int]] = set()
    for row_index, row in enumerate(rows):
        try:
            dtype = _DTYPES[row["dtype"]]
            message_bytes = _as_positive_int(row["message_bytes"], "message_bytes")
            group_size = _as_positive_int(row["num_devices"], "num_devices")
            topology_tier = int(row["topology_tier"])
            latency_us = float(row["Duration(us)"])
            if group_size <= 1 or topology_tier < 0 or latency_us < 0:
                raise ValueError("invalid collective dimensions or latency")
            key = dtype, group_size, topology_tier, message_bytes
            if key in seen:
                raise ValueError("duplicate dtype/group_size/topology_tier/message_bytes row")
            seen.add(key)
            grouped[dtype, group_size, topology_tier].append([message_bytes, latency_us])
        except (KeyError, TypeError, ValueError) as error:
            rejected.append({"row": row_index, "reason": str(error)})

    if rejected:
        raise ValueError(f"communication CSV {path.name} contains invalid rows: {rejected[:3]}")
    models = []
    for (dtype, group_size, topology_tier), points in sorted(grouped.items()):
        points.sort(key=lambda point: point[0])
        models.append(
            {
                "collective": collective,
                "tc_op": _COLLECTIVES[collective][1],
                "dtype": dtype,
                "group_size": group_size,
                "topology_tier": topology_tier,
                "latency_points_us": points,
                "confidence": "measured",
            }
        )
    if not models:
        raise ValueError(f"communication CSV {path.name} contains no valid rows")
    return models, {
        "collective": collective,
        "path": path.name,
        "sha256": _sha256(path),
        "accepted_rows": len(rows),
        "rejected_rows": 0,
        "curve_count": len(models),
    }


def build_comm_profile(
    hccl_directory: str | Path,
    *,
    audit_path: str | Path,
    base_profile: str | Path | None = None,
    device_name: Optional[str] = None,
    software_stack: Optional[str] = None,
    profile_id: Optional[str] = None,
    collectives: Iterable[str] = _COLLECTIVES,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    directory = Path(hccl_directory)
    selected = list(collectives)
    if not selected or any(collective not in _COLLECTIVES for collective in selected):
        raise ValueError("collectives must be one or more supported collective names")
    paths = [directory / _COLLECTIVES[collective][0] for collective in selected]
    all_models: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    for collective, path in zip(selected, paths):
        models, report = _load_collective(path, collective)
        all_models.extend(models)
        reports.append(report)

    digest = _directory_digest(paths, directory)
    audit = {
        "version": 1,
        "generator": GENERATOR_NAME,
        "generator_version": GENERATOR_VERSION,
        "hccl_directory": directory.as_posix(),
        "hccl_directory_digest": digest,
        "byte_semantics": {
            "all_reduce": "per-rank input buffer",
            "all_gather": "per-rank local input buffer",
            "reduce_scatter": "per-rank output buffer (TC input bytes / group_size)",
            "all_to_all": "full per-rank input list",
        },
        "kernels": reports,
        "models": [
            {
                "collective": model["collective"],
                "dtype": model["dtype"],
                "group_size": model["group_size"],
                "topology_tier": model["topology_tier"],
                "message_bytes_range": [model["latency_points_us"][0][0], model["latency_points_us"][-1][0]],
                "point_count": len(model["latency_points_us"]),
            }
            for model in all_models
        ],
    }
    rendered_audit = yaml.safe_dump(audit, sort_keys=False, allow_unicode=True).encode("utf-8")
    comm_source = {
        "directory": directory.as_posix(),
        "directory_digest": digest,
        "audit": Path(audit_path).name,
        "audit_sha256": hashlib.sha256(rendered_audit).hexdigest(),
    }
    if base_profile is not None:
        profile = load_sqlite_document(base_profile)
        validate_mm_profile_document(profile)
        if device_name is not None and profile.get("device") != device_name:
            raise ValueError("base profile device does not match device_name")
        if software_stack is not None and profile.get("software_stack") != software_stack:
            raise ValueError("base profile software stack does not match software_stack")
        profile = dict(profile)
    else:
        if not device_name or not software_stack:
            raise ValueError("device_name and software_stack are required without a base profile")
        profile = {
            "version": 3,
            "id": profile_id or f"{device_name.lower()}-{software_stack}-comm",
            "device": device_name,
            "software_stack": software_stack,
            "source": {
                "database": directory.as_posix(),
                "database_digest": digest,
                "audit": Path(audit_path).name,
                "audit_sha256": hashlib.sha256(rendered_audit).hexdigest(),
            },
        }
    profile["communication_source"] = comm_source
    profile["calibration_sources"] = dict(profile.get("calibration_sources", {}))
    profile["calibration_sources"]["communication"] = comm_source
    profile["communication_models"] = all_models
    validate_mm_profile_document(profile)
    return profile, audit


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build HCCL collective calibration curves.")
    parser.add_argument("hccl_directory", type=Path)
    parser.add_argument("--base-profile", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--software-stack")
    parser.add_argument("--profile-id")
    parser.add_argument("--collective", action="append", choices=sorted(_COLLECTIVES))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    args = parser.parse_args(argv)
    profile, audit = build_comm_profile(
        args.hccl_directory,
        audit_path=args.audit_output,
        base_profile=args.base_profile,
        device_name=args.device,
        software_stack=args.software_stack,
        profile_id=args.profile_id,
        collectives=args.collective or _COLLECTIVES,
    )
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(yaml.safe_dump(audit, sort_keys=False, allow_unicode=True), encoding="utf-8")
    write_sqlite_profile(profile, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
