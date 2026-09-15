"""Build compact INT8 GMM calibration models from profiling CSVs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import torch
import yaml

from ...device import DeviceProfile
from ...utils import performance_dtype
from ..profiling_database.profiling_data_source import _parse_shape_str, _parse_str_list, fractal_nz_to_nd
from .mm_build import (
    _compress_curve,
    _curve_holdout_statistics,
    _curve_prediction,
    _database_digest,
    _error_statistics,
    _is_lfs_pointer,
    _sha256,
)
from .profile import validate_mm_profile_document
from .sqlite_profile import load_sqlite_document, write_sqlite_profile

GENERATOR_NAME = "tensor_cast.analytic_calibration.gmm_build"
GENERATOR_VERSION = "1"
DEFAULT_TARGETS = (
    "plain=GroupedMatmul",
    "swiglu_quant=GroupedMatmulSwigluQuant",
)

_TC_OPS = {
    "plain": ["tensor_cast.grouped_matmul_quant.default"],
    "swiglu_quant": ["tensor_cast.grouped_matmul_quant_swiglu.default"],
}


@dataclass(frozen=True)
class _Target:
    variant: str
    kernel: str


@dataclass(frozen=True)
class _SourceRef:
    kernel: str
    row: int
    latency_column: str
    latency_us: float
    profiling_backed: bool


@dataclass(frozen=True)
class _Sample:
    variant: str
    m: int
    k: int
    n: int
    gemm_n: int
    num_experts: int
    utilization: float
    measured_latency_us: float
    ideal_mma_latency_us: float
    sources: tuple[_SourceRef, ...]

    @property
    def identity(self) -> tuple[str]:
        return (self.variant,)

    @property
    def signature(self) -> tuple[str, int, int, int, int, int]:
        return self.variant, self.m, self.k, self.n, self.gemm_n, self.num_experts


def _parse_target(value: str) -> _Target:
    try:
        variant, kernel = (part.strip() for part in value.split("=", 1))
    except ValueError as error:
        raise ValueError("--target must use variant=Kernel") from error
    if variant not in _TC_OPS or not kernel:
        raise ValueError(f"unsupported GMM target {value!r}")
    return _Target(variant=variant, kernel=kernel)


def _restore_weight(shape: tuple[int, ...], input_format: str) -> tuple[int, ...]:
    return fractal_nz_to_nd(shape) if input_format == "FRACTAL_NZ" else shape


def _select_average_latency(row: Mapping[str, Any]) -> float:
    """Use only the standalone single-op measurement for GMM fitting."""
    try:
        latency_us = float(row.get("Average Duration(us)", ""))
    except (TypeError, ValueError) as error:
        raise ValueError("Average Duration(us) is missing or non-numeric") from error
    if not math.isfinite(latency_us) or latency_us <= 0:
        raise ValueError("Average Duration(us) must be a finite positive value")
    return latency_us


def _extract_shape(row: Mapping[str, Any], target: _Target) -> tuple[int, int, int, int, int]:
    shapes = _parse_shape_str(str(row.get("Input Shapes", "")), preserve_empty_slots=True)
    dtypes = _parse_str_list(str(row.get("Input Data Types", "")))
    formats = _parse_str_list(str(row.get("Input Formats", "")))
    outputs = _parse_shape_str(str(row.get("Output Shapes", "")), preserve_empty_slots=True)
    output_dtypes = _parse_str_list(str(row.get("Output Data Types", "")))
    if len(shapes) < 2 or len(dtypes) < 2 or not outputs:
        raise ValueError("row needs activation, weight, and output shapes")
    if dtypes[0].strip().upper() not in {"INT8", "DT_INT8"} or dtypes[1].strip().upper() not in {
        "INT8",
        "DT_INT8",
    }:
        raise ValueError("P0 GMM calibration accepts INT8 activation and weight only")
    activation = shapes[0]
    weight = _restore_weight(shapes[1], formats[1] if len(formats) > 1 else "ND")
    output = outputs[0]
    if len(activation) != 2 or len(weight) != 3 or len(output) != 2:
        raise ValueError("GMM activation/output must be rank 2 and weight rank 3")
    m, k = activation
    num_experts = weight[0]
    if output[0] != m:
        raise ValueError("output M does not match activation M_total")
    n = output[1]
    gemm_n = n * 2 if target.variant == "swiglu_quant" else n
    if {weight[1], weight[2]} != {k, gemm_n}:
        raise ValueError("weight dimensions do not match activation K and output N")
    if k == gemm_n:
        raise ValueError("weight orientation is ambiguous when K equals logical N")
    group_slot = 4 if target.variant == "swiglu_quant" else 7
    if len(shapes) <= group_slot or shapes[group_slot] != (num_experts,):
        raise ValueError("group-list shape does not match the weight expert dimension")
    if target.variant == "swiglu_quant":
        if len(output_dtypes) < 2 or output_dtypes[0].strip().upper() not in {"INT8", "DT_INT8"}:
            raise ValueError("GMM+SwiGLU+Quant row must expose an INT8 primary output and scale output")
        if len(outputs) < 2 or outputs[1] != (m,):
            raise ValueError("GMM+SwiGLU+Quant scale output must be [M_total]")
    if min(m, k, n, gemm_n, num_experts) <= 0:
        raise ValueError("GMM dimensions must be positive")
    return int(m), int(k), int(n), int(gemm_n), int(num_experts)


def _load_samples(csv_path: Path, target: _Target, device: DeviceProfile) -> tuple[list[_Sample], dict[str, Any]]:
    report: dict[str, Any] = {"kernel": target.kernel, "path": csv_path.name, "variant": target.variant}
    if not csv_path.is_file():
        return [], {**report, "status": "missing", "accepted_rows": 0, "rejected_rows": 0}
    report["sha256"] = _sha256(csv_path)
    if _is_lfs_pointer(csv_path):
        return [], {**report, "status": "lfs_pointer", "accepted_rows": 0, "rejected_rows": 0}
    peak_ops = device.mma_ops.get(performance_dtype(torch.int8))
    if peak_ops is None or peak_ops <= 0:
        raise ValueError(f"device {device.name} has no INT8 MMA peak")
    accepted: list[_Sample] = []
    rejected: list[dict[str, Any]] = []
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        for row_index, row in enumerate(csv.DictReader(handle)):
            try:
                m, k, n, gemm_n, num_experts = _extract_shape(row, target)
                latency_us = _select_average_latency(row)
                ideal_us = 2 * m * k * gemm_n / peak_ops * 1e6
                utilization = ideal_us / latency_us
                if not math.isfinite(utilization) or not 0 < utilization <= 1:
                    raise ValueError(f"derived utilization {utilization:.6f} is outside (0, 1]")
                accepted.append(
                    _Sample(
                        variant=target.variant,
                        m=m,
                        k=k,
                        n=n,
                        gemm_n=gemm_n,
                        num_experts=num_experts,
                        utilization=utilization,
                        measured_latency_us=latency_us,
                        ideal_mma_latency_us=ideal_us,
                        sources=(
                            _SourceRef(
                                kernel=target.kernel,
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
    grouped: dict[tuple[Any, ...], list[_Sample]] = {}
    for sample in samples:
        grouped.setdefault(sample.signature, []).append(sample)
    result = []
    for group in grouped.values():
        representative = group[0]
        measured = sorted(item.measured_latency_us for item in group)[len(group) // 2]
        result.append(
            _Sample(
                variant=representative.variant,
                m=representative.m,
                k=representative.k,
                n=representative.n,
                gemm_n=representative.gemm_n,
                num_experts=representative.num_experts,
                utilization=representative.ideal_mma_latency_us / measured,
                measured_latency_us=measured,
                ideal_mma_latency_us=representative.ideal_mma_latency_us,
                sources=tuple(source for item in group for source in item.sources),
            )
        )
    return sorted(result, key=lambda item: item.signature)


def _domain(samples: list[_Sample]) -> dict[str, list[int]]:
    return {
        feature: [min(getattr(item, feature) for item in samples), max(getattr(item, feature) for item in samples) + 1]
        for feature in ("m", "k", "n", "gemm_n", "num_experts")
    } | {"m_total": [min(item.m for item in samples), max(item.m for item in samples) + 1]}


def _fit_model(samples: list[_Sample], tolerance: float) -> tuple[dict[str, Any], dict[str, Any]]:
    variant = samples[0].variant
    grouped: dict[tuple[int, int, int, int], list[_Sample]] = {}
    for sample in samples:
        grouped.setdefault((sample.num_experts, sample.k, sample.n, sample.gemm_n), []).append(sample)
    curve_groups = {key: group for key, group in grouped.items() if group}
    curves = []
    audits = []
    curve_samples: list[_Sample] = []
    fitted: dict[tuple[int, int, int, int], list[_Sample]] = {}
    for key, group in sorted(curve_groups.items()):
        num_experts, k, n, gemm_n = key
        knots = _compress_curve(group, tolerance)
        fitted[key] = knots
        curve_samples.extend(group)
        curves.append(
            {
                "num_experts": num_experts,
                "k": k,
                "n": n,
                "gemm_n": gemm_n,
                "m_points": [[sample.m, round(sample.utilization, 9)] for sample in knots],
                "confidence": "measured",
            }
        )
        audits.append(
            {
                "key": {"num_experts": num_experts, "k": k, "n": n, "gemm_n": gemm_n},
                "source_shape_count": len(group),
                "knot_count": len(knots),
                "m_range": [min(item.m for item in group), max(item.m for item in group)],
                "profiling_anchor_count": sum(
                    any(source.profiling_backed for source in item.sources) for item in group
                ),
                "compression_error": _error_statistics(group, lambda item, value=knots: _curve_prediction(item, value)),
            }
        )
    output_dtype = "int8" if variant == "swiglu_quant" else "bfloat16"
    domain = _domain(samples)
    domain.pop("m")
    model = {
        "variant": variant,
        "tc_ops": _TC_OPS[variant],
        "quantization": "w8a8",
        "input_dtype": "int8",
        "weight_dtype": "int8",
        "output_dtype": output_dtype,
        "distribution": "balanced",
        "calibration_mode": "total_latency",
        "domain": domain,
        "curves": curves,
    }
    audit = {
        "variant": variant,
        "unique_shape_count": len(samples),
        "curve_count": len(curves),
        "curve_shape_count": len(curve_samples),
        "curve_knot_count": sum(len(curve["m_points"]) for curve in curves),
        "curve_compression_error": _error_statistics(
            curve_samples,
            lambda item: _curve_prediction(item, fitted[(item.num_experts, item.k, item.n, item.gemm_n)]),
        ),
        "curve_holdout_error": _curve_holdout_statistics(curve_groups.values(), tolerance),
        "curves": audits,
        "samples": [
            {
                "shape": {
                    "m_total": item.m,
                    "k": item.k,
                    "n": item.n,
                    "gemm_n": item.gemm_n,
                    "num_experts": item.num_experts,
                },
                "utilization": item.utilization,
                "measured_latency_us": item.measured_latency_us,
                "ideal_mma_latency_us": item.ideal_mma_latency_us,
                "route": "curve",
                "sources": [source.__dict__ for source in item.sources],
            }
            for item in samples
        ],
    }
    return model, audit


def build_gmm_profile(
    database_path: str | Path,
    *,
    targets: Iterable[str] = DEFAULT_TARGETS,
    base_profile: str | Path | None = None,
    profile_id: Optional[str] = None,
    curve_tolerance: float = 0.10,
    device_name: Optional[str] = None,
    software_stack: Optional[str] = None,
    audit_path: str | Path,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    database = Path(database_path)
    mapping_path = database / "op_mapping.yaml"
    mapping = yaml.safe_load(mapping_path.read_text(encoding="utf-8")) if mapping_path.is_file() else {}
    device_name = device_name or (mapping or {}).get("device")
    if not isinstance(device_name, str) or device_name not in DeviceProfile.all_device_profiles:
        raise ValueError("a known --device is required when op_mapping.yaml is absent")
    device = DeviceProfile.all_device_profiles[device_name]
    software_stack = software_stack or database.name
    samples: list[_Sample] = []
    reports = []
    digest_paths = {mapping_path} if mapping_path.is_file() else set()
    for target_text in targets:
        target = _parse_target(target_text)
        csv_path = database / f"{target.kernel}.csv"
        if csv_path.is_file():
            digest_paths.add(csv_path)
        loaded, report = _load_samples(csv_path, target, device)
        reports.append(report)
        samples.extend(loaded)
        if report["status"] != "loaded":
            raise ValueError(f"kernel {target.kernel} is unavailable ({report['status']})")
    samples = _deduplicate(samples)
    grouped: dict[str, list[_Sample]] = {}
    for sample in samples:
        grouped.setdefault(sample.variant, []).append(sample)
    models, model_audits = [], []
    for group in grouped.values():
        model, model_audit = _fit_model(group, curve_tolerance)
        models.append(model)
        model_audits.append(model_audit)
    stack = software_stack
    resolved_profile_id = profile_id or f"{device_name.lower()}-{stack}-mm-gmm"
    database_digest = _database_digest(digest_paths)
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
            "strategy": "average_duration_e_kn_curves",
            "distribution": "balanced replay group-list",
            "curve_tolerance": curve_tolerance,
        },
        "kernels": reports,
        "models": model_audits,
    }
    rendered_audit = yaml.safe_dump(audit, sort_keys=False, allow_unicode=True)
    gmm_source = {
        "database": database.name,
        "database_digest": database_digest,
        "audit": Path(audit_path).name,
        "audit_sha256": hashlib.sha256(rendered_audit.encode("utf-8")).hexdigest(),
    }
    if base_profile is not None:
        profile = load_sqlite_document(base_profile)
        if profile.get("version") != 3 or profile.get("device") != device_name:
            raise ValueError("base profile must be a SQLite profile for the same device")
        if profile.get("software_stack") != stack:
            raise ValueError("base profile software stack does not match the GMM database")
        profile = dict(profile)
    else:
        profile = {
            "version": 3,
            "id": resolved_profile_id,
            "device": device_name,
            "software_stack": stack,
            "source": gmm_source,
        }
    profile["gmm_source"] = gmm_source
    profile["calibration_sources"] = dict(profile.get("calibration_sources", {}))
    profile["calibration_sources"]["gmm"] = gmm_source
    profile["gmm_models"] = models
    validate_mm_profile_document(profile)
    return profile, audit


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build compact INT8 GMM calibration models.")
    parser.add_argument("database", type=Path)
    parser.add_argument("--device", dest="device_name")
    parser.add_argument("--software-stack")
    parser.add_argument("--target", action="append", help="plain=GroupedMatmul or swiglu_quant=Kernel")
    parser.add_argument("--base-profile", type=Path)
    parser.add_argument("--profile-id")
    parser.add_argument("--curve-tolerance", type=float, default=0.10)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    args = parser.parse_args(argv)
    profile, audit = build_gmm_profile(
        args.database,
        targets=args.target or DEFAULT_TARGETS,
        base_profile=args.base_profile,
        profile_id=args.profile_id,
        curve_tolerance=args.curve_tolerance,
        device_name=args.device_name,
        software_stack=args.software_stack,
        audit_path=args.audit_output,
    )
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(yaml.safe_dump(audit, sort_keys=False, allow_unicode=True), encoding="utf-8")
    write_sqlite_profile(profile, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
