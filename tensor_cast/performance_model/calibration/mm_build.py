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

"""Build compact MM-utilization calibration profiles from profiling CSVs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import torch
import yaml

from ...device import DeviceProfile
from ...utils import performance_dtype
from ..profiling_database.profiling_data_source import _parse_shape_str, _parse_str_list, fractal_nz_to_nd
from .profile import validate_mm_profile_document
from .sqlite_profile import SQLiteCalibrationProfile, load_sqlite_document, write_sqlite_profile

GENERATOR_NAME = "tensor_cast.analytic_calibration.mm_build"
GENERATOR_VERSION = "1"
DEFAULT_TARGETS = (
    "dense=MatMulV2,MatMulV3,MatMulCommon",
    "weight_quant:w8a8=QuantBatchMatmulV3",
)


@dataclass(frozen=True)
class _Target:
    kind: str
    quantization: str
    kernels: tuple[str, ...]


@dataclass(frozen=True)
class _SourceRef:
    kernel: str
    row: int
    latency_column: str
    latency_us: float
    profiling_backed: bool


@dataclass(frozen=True)
class _Sample:
    kind: str
    quantization: str
    compute_dtype: str
    m: int
    k: int
    n: int
    utilization: float
    measured_latency_us: float
    ideal_mma_latency_us: float
    sources: tuple[_SourceRef, ...]

    @property
    def identity(self) -> tuple[str, str, str]:
        return self.kind, self.quantization, self.compute_dtype

    @property
    def signature(self) -> tuple[str, str, str, int, int, int]:
        return (*self.identity, self.m, self.k, self.n)


def _as_mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{location} must be a mapping")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _database_digest(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.name):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _is_lfs_pointer(path: Path) -> bool:
    try:
        return path.read_bytes()[:64].startswith(b"version https://git-lfs.github.com/spec/v1")
    except OSError:
        return False


def _select_average_latency(row: Mapping[str, Any]) -> float:
    """Use only the standalone single-op measurement for MM fitting."""
    try:
        latency_us = float(row.get("Average Duration(us)", ""))
    except (TypeError, ValueError) as error:
        raise ValueError("Average Duration(us) is missing or non-numeric") from error
    if not math.isfinite(latency_us) or latency_us <= 0:
        raise ValueError("Average Duration(us) must be a finite positive value")
    return latency_us


def _parse_target(value: str) -> _Target:
    try:
        selector, kernel_text = value.split("=", 1)
    except ValueError as error:
        raise ValueError("--target must use kind[:quantization]=KernelA,KernelB") from error
    kind, separator, quantization = selector.partition(":")
    kernels = tuple(kernel.strip() for kernel in kernel_text.split(",") if kernel.strip())
    if not kind or not kernels:
        raise ValueError("--target needs a non-empty kind and at least one kernel")
    return _Target(kind=kind, quantization=quantization if separator else "none", kernels=kernels)


def _canonical_csv_dtype(value: str) -> str:
    normalized = value.strip().upper()
    aliases = {
        "BF16": "DT_BF16",
        "FLOAT16": "DT_FLOAT16",
        "FLOAT": "DT_FLOAT",
        "INT8": "DT_INT8",
        "INT4": "DT_INT4",
        "UINT4": "DT_UINT4",
        "INT32": "DT_INT32",
    }
    return aliases.get(normalized, normalized)


def _dtype_from_csv(value: str) -> tuple[str, torch.dtype]:
    normalized = _canonical_csv_dtype(value)
    mapping = {
        "DT_BF16": ("bfloat16", torch.bfloat16),
        "DT_FLOAT16": ("float16", torch.float16),
        "DT_FLOAT": ("float32", torch.float32),
        "DT_INT8": ("int8", torch.int8),
        "DT_FLOAT8_E5M2": ("float8_e5m2", torch.float8_e5m2),
        "DT_FLOAT8_E4M3FN": ("float8_e4m3fn", torch.float8_e4m3fn),
    }
    if normalized not in mapping:
        raise ValueError(f"unsupported compute dtype {normalized!r}")
    return mapping[normalized]


def _restore_shape(shape: tuple[int, ...], fmt: str) -> tuple[int, ...]:
    return fractal_nz_to_nd(shape) if fmt == "FRACTAL_NZ" else shape


def _quantization_matches(target: _Target, dtypes: list[str]) -> bool:
    if target.quantization == "none":
        return True
    if len(dtypes) < 2:
        return False
    activation_dtype, weight_dtype = map(_canonical_csv_dtype, dtypes[:2])
    if target.quantization == "w8a8":
        return activation_dtype == "DT_INT8" and weight_dtype == "DT_INT8"
    if target.quantization == "w4a8":
        return activation_dtype == "DT_INT8" and weight_dtype in {"DT_INT4", "DT_UINT4", "DT_INT32"}
    if target.quantization == "fp8":
        return activation_dtype.startswith("DT_FLOAT8") and weight_dtype.startswith("DT_FLOAT8")
    raise ValueError(f"unsupported target quantization {target.quantization!r}")


def _extract_mkn(row: Mapping[str, Any], target: _Target) -> tuple[int, int, int, str, torch.dtype]:
    shapes = _parse_shape_str(str(row.get("Input Shapes", "")), preserve_empty_slots=False)
    dtypes = _parse_str_list(str(row.get("Input Data Types", "")))
    formats = _parse_str_list(str(row.get("Input Formats", "")))
    if len(shapes) < 2 or len(dtypes) < 2:
        raise ValueError("row needs at least two input shapes and dtypes")
    if not _quantization_matches(target, dtypes):
        raise ValueError(f"input dtypes {dtypes[:2]!r} do not match {target.quantization}")
    lhs = _restore_shape(shapes[0], formats[0] if formats else "ND")
    rhs = _restore_shape(shapes[1], formats[1] if len(formats) > 1 else "ND")
    if len(lhs) < 2 or len(rhs) < 2:
        raise ValueError("MM builder requires rank-2-or-higher activation and weight")
    m = math.prod(lhs[:-1])
    k = lhs[-1]
    if rhs[-1] == k:
        n = rhs[-2]
    elif rhs[-2] == k:
        n = rhs[-1]
    else:
        raise ValueError("weight shape does not share the activation K dimension")
    dtype_name, torch_dtype = _dtype_from_csv(dtypes[0])
    if min(m, k, n) <= 0:
        raise ValueError("M, K, and N must be positive")
    return int(m), int(k), int(n), dtype_name, torch_dtype


def _load_kernel_samples(
    csv_path: Path,
    target: _Target,
    device: DeviceProfile,
) -> tuple[list[_Sample], dict[str, Any]]:
    kernel = csv_path.stem
    report: dict[str, Any] = {"kernel": kernel, "path": csv_path.name}
    if not csv_path.is_file():
        report.update(status="missing", accepted_rows=0, rejected_rows=0)
        return [], report
    report["sha256"] = _sha256(csv_path)
    if _is_lfs_pointer(csv_path):
        report.update(status="lfs_pointer", accepted_rows=0, rejected_rows=0)
        return [], report

    accepted: list[_Sample] = []
    rejected: list[dict[str, Any]] = []
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        for row_index, row in enumerate(csv.DictReader(handle)):
            try:
                m, k, n, dtype_name, torch_dtype = _extract_mkn(row, target)
                peak_ops = device.mma_ops.get(performance_dtype(torch_dtype))
                if peak_ops is None or peak_ops <= 0:
                    raise ValueError(f"device has no MMA peak for {dtype_name}")
                latency_us = _select_average_latency(row)
                ideal_mma_latency_us = 2 * m * k * n / peak_ops * 1e6
                utilization = ideal_mma_latency_us / latency_us
                if not math.isfinite(utilization) or not 0 < utilization <= 1:
                    raise ValueError(f"derived utilization {utilization:.6f} is outside (0, 1]")
                accepted.append(
                    _Sample(
                        kind=target.kind,
                        quantization=target.quantization,
                        compute_dtype=dtype_name,
                        m=m,
                        k=k,
                        n=n,
                        utilization=utilization,
                        measured_latency_us=latency_us,
                        ideal_mma_latency_us=ideal_mma_latency_us,
                        sources=(
                            _SourceRef(
                                kernel=kernel,
                                row=row_index,
                                latency_column="Average Duration(us)",
                                latency_us=latency_us,
                                profiling_backed=float(row.get("Profiling Average Duration(us)", 0) or 0) > 0,
                            ),
                        ),
                    )
                )
            except ValueError as error:
                rejected.append({"row": row_index, "reason": str(error)})
    report.update(
        status="loaded" if accepted else "no_compatible_rows",
        accepted_rows=len(accepted),
        rejected_rows=len(rejected),
        rejected=rejected,
    )
    return accepted, report


def _deduplicate(samples: Iterable[_Sample]) -> list[_Sample]:
    grouped: dict[tuple[str, str, str, int, int, int], list[_Sample]] = {}
    for sample in samples:
        grouped.setdefault(sample.signature, []).append(sample)
    result = []
    for group in grouped.values():
        representative = group[0]
        measured_latency_us = float(statistics.median(sample.measured_latency_us for sample in group))
        result.append(
            _Sample(
                kind=representative.kind,
                quantization=representative.quantization,
                compute_dtype=representative.compute_dtype,
                m=representative.m,
                k=representative.k,
                n=representative.n,
                utilization=representative.ideal_mma_latency_us / measured_latency_us,
                measured_latency_us=measured_latency_us,
                ideal_mma_latency_us=representative.ideal_mma_latency_us,
                sources=tuple(source for sample in group for source in sample.sources),
            )
        )
    return sorted(result, key=lambda sample: sample.signature)


def _predict_latency(sample: _Sample, fixed_overhead_us: float, utilization: float) -> float:
    return fixed_overhead_us + sample.ideal_mma_latency_us / utilization


def _mean_relative_error(samples: Iterable[_Sample], fixed_overhead_us: float, utilization: float) -> float:
    values = list(samples)
    if not values:
        return 0.0
    return sum(
        abs(_predict_latency(sample, fixed_overhead_us, utilization) / sample.measured_latency_us - 1.0)
        for sample in values
    ) / len(values)


def _quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round(fraction * (len(ordered) - 1))]


def _domain(samples: Iterable[_Sample]) -> dict[str, list[int]]:
    values = list(samples)
    return {
        feature: [
            min(getattr(sample, feature) for sample in values),
            max(getattr(sample, feature) for sample in values) + 1,
        ]
        for feature in ("m", "k", "n")
    }


def _error_statistics(samples: Iterable[_Sample], predict) -> dict[str, float]:
    errors = [abs(predict(sample) / sample.measured_latency_us - 1.0) * 100 for sample in samples]
    if not errors:
        return {"mape_percent": 0.0, "p90_percent": 0.0, "max_percent": 0.0}
    errors.sort()
    p90_index = math.ceil(0.9 * len(errors)) - 1
    return {
        "mape_percent": sum(errors) / len(errors),
        "p90_percent": errors[p90_index],
        "max_percent": errors[-1],
    }


def _curve_utilization_from_samples(knots: list[_Sample], m: int) -> float:
    ordered = sorted(knots, key=lambda sample: sample.m)
    if len(ordered) == 1:
        return ordered[0].utilization
    for left, right in zip(ordered, ordered[1:]):
        if left.m <= m <= right.m:
            if m == left.m:
                return left.utilization
            if m == right.m:
                return right.utilization
            ratio = (math.log(m) - math.log(left.m)) / (math.log(right.m) - math.log(left.m))
            normalized_left = left.m / left.utilization
            normalized_right = right.m / right.utilization
            normalized = math.exp(
                math.log(normalized_left) + ratio * (math.log(normalized_right) - math.log(normalized_left))
            )
            return m / normalized
    raise ValueError("curve query is outside its measured M range")


def _curve_prediction(sample: _Sample, knots: list[_Sample]) -> float:
    return sample.ideal_mma_latency_us / _curve_utilization_from_samples(knots, sample.m)


def _compress_curve(samples: list[_Sample], tolerance: float) -> list[_Sample]:
    ordered = sorted(samples, key=lambda sample: sample.m)
    if len(ordered) <= 2:
        return ordered
    kept = {0, len(ordered) - 1}
    while True:
        knots = [ordered[index] for index in sorted(kept)]
        candidates = [
            (abs(_curve_prediction(sample, knots) / sample.measured_latency_us - 1.0), index)
            for index, sample in enumerate(ordered)
            if index not in kept
        ]
        if not candidates:
            return knots
        error, index = max(candidates)
        if error <= tolerance:
            return knots
        kept.add(index)


def _curve_holdout_statistics(groups: Iterable[list[_Sample]], tolerance: float) -> dict[str, float]:
    measured: list[_Sample] = []
    predictions: dict[tuple[str, str, str, int, int, int], float] = {}
    for samples in groups:
        ordered = sorted(samples, key=lambda sample: sample.m)
        if len(ordered) < 3:
            continue
        holdout = [
            sample
            for sample in ordered[1:-1]
            if hashlib.sha256(":".join(map(str, sample.signature)).encode("utf-8")).digest()[0] % 5 == 0
        ]
        fit = [sample for sample in ordered if sample not in holdout]
        if not holdout or len(fit) < 2:
            continue
        knots = _compress_curve(fit, tolerance)
        for sample in holdout:
            measured.append(sample)
            predictions[sample.signature] = _curve_prediction(sample, knots)
    return _error_statistics(measured, lambda sample: predictions[sample.signature])


def _fit_curve_model(samples: list[_Sample], *, curve_tolerance: float) -> tuple[dict[str, Any], dict[str, Any]]:
    kind, quantization, compute_dtype = samples[0].identity
    grouped: dict[tuple[int, int], list[_Sample]] = {}
    for sample in samples:
        grouped.setdefault((sample.k, sample.n), []).append(sample)
    curve_groups = {key: group for key, group in grouped.items() if group}
    curves = []
    curve_audits = []
    curve_samples: list[_Sample] = []
    common_knots: dict[tuple[int, int], list[_Sample]] = {}
    for (k, n), group in sorted(curve_groups.items()):
        knots = _compress_curve(group, curve_tolerance)
        common_knots[(k, n)] = knots
        curve_samples.extend(group)
        curve_id = f"k{k}-n{n}"
        curves.append(
            {
                "k": k,
                "n": n,
                "m_points": [[sample.m, round(sample.utilization, 9)] for sample in knots],
                "confidence": "measured",
            }
        )
        curve_audits.append(
            {
                "id": curve_id,
                "source_shape_count": len(group),
                "knot_count": len(knots),
                "m_range": [min(sample.m for sample in group), max(sample.m for sample in group)],
                "compression_error": _error_statistics(
                    group, lambda sample, fitted=knots: _curve_prediction(sample, fitted)
                ),
                "profiling_anchor_count": sum(
                    any(source.profiling_backed for source in sample.sources) for sample in group
                ),
            }
        )
    model = {
        "kind": kind,
        "quantization": quantization,
        "compute_dtype": compute_dtype,
        "calibration_mode": "total_latency",
        "domain": _domain(samples),
        "curves": curves,
    }
    audit = {
        "kind": kind,
        "quantization": quantization,
        "compute_dtype": compute_dtype,
        "unique_shape_count": len(samples),
        "curve_count": len(curves),
        "curve_shape_count": len(curve_samples),
        "curve_knot_count": sum(len(curve["m_points"]) for curve in curves),
        "curve_compression_error": _error_statistics(
            curve_samples,
            lambda sample: _curve_prediction(sample, common_knots[(sample.k, sample.n)]),
        ),
        "curve_holdout_error": _curve_holdout_statistics(curve_groups.values(), curve_tolerance),
        "curves": curve_audits,
        "samples": [
            {
                "shape": {"m": sample.m, "k": sample.k, "n": sample.n},
                "utilization": sample.utilization,
                "measured_latency_us": sample.measured_latency_us,
                "ideal_mma_latency_us": sample.ideal_mma_latency_us,
                "route": "curve",
                "sources": [source.__dict__ for source in sample.sources],
            }
            for sample in samples
        ],
    }
    return model, audit


def build_mm_profile(
    database_path: str | Path,
    *,
    targets: Iterable[str] = DEFAULT_TARGETS,
    profile_id: Optional[str] = None,
    curve_tolerance: float = 0.10,
    allow_missing_kernels: bool = False,
    device_name: Optional[str] = None,
    software_stack: Optional[str] = None,
    base_profile: str | Path | None = None,
    audit_path: str | Path,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    database = Path(database_path)
    mapping_path = database / "op_mapping.yaml"
    mapping = {}
    if mapping_path.is_file():
        mapping = _as_mapping(yaml.safe_load(mapping_path.read_text(encoding="utf-8")) or {}, "op_mapping.yaml")
    device_name = device_name or mapping.get("device")
    if not isinstance(device_name, str) or device_name not in DeviceProfile.all_device_profiles:
        raise ValueError("a known --device is required when op_mapping.yaml is absent")
    device = DeviceProfile.all_device_profiles[device_name]
    software_stack = software_stack or database.name
    parsed_targets = [_parse_target(target) for target in targets]

    all_samples: list[_Sample] = []
    kernel_reports: list[dict[str, Any]] = []
    digest_paths = {mapping_path} if mapping_path.is_file() else set()
    for target in parsed_targets:
        for kernel in target.kernels:
            csv_path = database / f"{kernel}.csv"
            if csv_path.is_file():
                digest_paths.add(csv_path)
            samples, report = _load_kernel_samples(csv_path, target, device)
            report.update(kind=target.kind, quantization=target.quantization)
            kernel_reports.append(report)
            all_samples.extend(samples)
            if report["status"] != "loaded" and not allow_missing_kernels:
                raise ValueError(
                    f"kernel {kernel} is unavailable ({report['status']}); materialize its CSV or allow missing"
                )

    samples = _deduplicate(all_samples)
    if not samples:
        raise ValueError("no usable MM samples were loaded")
    grouped: dict[tuple[str, str, str], list[_Sample]] = {}
    for sample in samples:
        grouped.setdefault(sample.identity, []).append(sample)

    models, model_audits = [], []
    for identity_samples in grouped.values():
        models_and_audit = _fit_curve_model(identity_samples, curve_tolerance=curve_tolerance)
        models.append(models_and_audit[0])
        model_audits.append(models_and_audit[1])

    database_digest = _database_digest(digest_paths)
    stack = software_stack
    resolved_profile_id = profile_id or f"{device_name.lower()}-{stack}-mm"
    audit = {
        "version": 1,
        "generator": GENERATOR_NAME,
        "generator_version": GENERATOR_VERSION,
        "profile_id": resolved_profile_id,
        "device": device_name,
        "software_stack": stack,
        "database": database.name,
        "database_digest": database_digest,
        "fit_config": {
            "strategy": "average_duration_kn_curves",
            "curve_tolerance": curve_tolerance,
            "holdout": "interior M points selected by sha256(shape) modulo 5",
        },
        "kernels": kernel_reports,
        "models": model_audits,
    }
    audit_rendered = yaml.safe_dump(audit, sort_keys=False, allow_unicode=True)
    audit_sha256 = hashlib.sha256(audit_rendered.encode("utf-8")).hexdigest()
    audit_file = Path(audit_path)
    mm_source = {
        "database": database.name,
        "database_digest": database_digest,
        "audit": audit_file.name,
        "audit_sha256": audit_sha256,
    }
    if base_profile is not None:
        profile = load_sqlite_document(base_profile)
        if profile.get("version") != 3 or profile.get("device") != device_name:
            raise ValueError("base profile must be a SQLite profile for the same device")
        if profile.get("software_stack") != stack:
            raise ValueError("base profile software stack does not match MM database")
        profile["source"] = dict(profile.get("source", {}))
        profile["source"].update(mm_source)
        identities = {(model["kind"], model.get("quantization", "none"), model["compute_dtype"]) for model in models}
        existing_models = [
            model
            for model in profile.get("mm_models", [])
            if (model["kind"], model.get("quantization", "none"), model["compute_dtype"]) not in identities
        ]
        profile["mm_models"] = [*existing_models, *models]
    else:
        profile = {
            "version": 3,
            "id": resolved_profile_id,
            "device": device_name,
            "software_stack": stack,
            "source": mm_source,
            "mm_models": models,
        }
    profile["mm_source"] = mm_source
    profile["calibration_sources"] = dict(profile.get("calibration_sources", {}))
    profile["calibration_sources"]["mm"] = mm_source
    validate_mm_profile_document(profile)
    return profile, audit


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build a compact MM-utilization calibration profile.")
    parser.add_argument(
        "database", type=Path, nargs="?", help="Profiling database directory containing kernel CSV files."
    )
    parser.add_argument("--device", dest="device_name")
    parser.add_argument("--software-stack")
    parser.add_argument("--target", action="append", help="kind[:quantization]=KernelA,KernelB; repeatable.")
    parser.add_argument("--profile-id")
    parser.add_argument("--base-profile", type=Path)
    parser.add_argument("--curve-tolerance", type=float, default=0.10)
    parser.add_argument("--allow-missing-kernels", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--audit-output", type=Path)
    parser.add_argument("--validate-profile", type=Path)
    args = parser.parse_args(argv)
    if args.validate_profile:
        if args.database or args.output or args.audit_output or args.target:
            parser.error("--validate-profile cannot be combined with build inputs")
        SQLiteCalibrationProfile(args.validate_profile)
        return 0
    if args.database is None or args.output is None or args.audit_output is None:
        parser.error("database, --output, and --audit-output are required")
    profile, audit = build_mm_profile(
        args.database,
        targets=args.target or DEFAULT_TARGETS,
        profile_id=args.profile_id,
        base_profile=args.base_profile,
        curve_tolerance=args.curve_tolerance,
        allow_missing_kernels=args.allow_missing_kernels,
        device_name=args.device_name,
        software_stack=args.software_stack,
        audit_path=args.audit_output,
    )
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_bytes(yaml.safe_dump(audit, sort_keys=False, allow_unicode=True).encode("utf-8"))
    write_sqlite_profile(profile, args.output, base_profile=args.base_profile)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
