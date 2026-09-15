"""SQLite calibration profiles.

Runtime calibration consumes only the SQLite artifact produced by the
family builders.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import sqlite3
from typing import Any, Mapping, Optional

from ...device import DeviceProfile
from .base import CalibrationDataSource, CalibrationRule
from .rules import (
    AttentionLatencyRule,
    CommunicationLatencyCurveRule,
    GmmUtilizationRule,
    MmaUtilizationRule,
)
from .signature import CalibrationSignature
from .sqlite_profile import SQLiteCalibrationProfile


def _as_mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{location} must be a mapping")
    return value


def _range_matches(value: Any, selector: Any) -> bool:
    if isinstance(selector, (str, int, float, bool)):
        return value == selector
    selector = _as_mapping(selector, "shape selector")
    if "exact" in selector:
        return value == selector["exact"]
    if "buckets" in selector:
        buckets = selector["buckets"]
        if not isinstance(buckets, list) or not buckets:
            raise ValueError("shape selector buckets must be a non-empty list")
        return any(_range_matches(value, bucket) for bucket in buckets)
    if not isinstance(value, (int, float)):
        return False
    minimum = selector.get("min")
    maximum = selector.get("max")
    maximum_exclusive = selector.get("max_exclusive")
    if minimum is not None and value < minimum:
        return False
    if maximum is not None and value > maximum:
        return False
    if maximum_exclusive is not None and value >= maximum_exclusive:
        return False
    return minimum is not None or maximum is not None or maximum_exclusive is not None


def _half_open_selector(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{location} must be a [min, max] pair")
    minimum, maximum = value
    if minimum is not None and not isinstance(minimum, (int, float)):
        raise ValueError(f"{location} min must be numeric or null")
    if maximum is not None and not isinstance(maximum, (int, float)):
        raise ValueError(f"{location} max must be numeric or null")
    if minimum is None and maximum is None:
        raise ValueError(f"{location} cannot be unbounded on both sides")
    if minimum is not None and maximum is not None and minimum >= maximum:
        raise ValueError(f"{location} min must be less than max")
    selector: dict[str, Any] = {}
    if minimum is not None:
        selector["min"] = minimum
    if maximum is not None:
        selector["max_exclusive"] = maximum
    return selector


def _intersect_half_open_selectors(
    left: Mapping[str, Any], right: Mapping[str, Any], location: str
) -> Mapping[str, Any]:
    minimums = [value for value in (left.get("min"), right.get("min")) if value is not None]
    maximums = [value for value in (left.get("max_exclusive"), right.get("max_exclusive")) if value is not None]
    minimum = max(minimums) if minimums else None
    maximum = min(maximums) if maximums else None
    if minimum is not None and maximum is not None and minimum >= maximum:
        raise ValueError(f"{location} does not intersect the model domain")
    result: dict[str, Any] = {}
    if minimum is not None:
        result["min"] = minimum
    if maximum is not None:
        result["max_exclusive"] = maximum
    return result


def _selector_bounds(selector: Mapping[str, Any]) -> tuple[Optional[float], Optional[float]]:
    return selector.get("min"), selector.get("max_exclusive")


def _boxes_overlap(left: Mapping[str, Mapping[str, Any]], right: Mapping[str, Mapping[str, Any]]) -> bool:
    for feature in set(left) | set(right):
        left_min, left_max = _selector_bounds(left.get(feature, {}))
        right_min, right_max = _selector_bounds(right.get(feature, {}))
        if left_max is not None and right_min is not None and left_max <= right_min:
            return False
        if right_max is not None and left_min is not None and right_max <= left_min:
            return False
    return True


def validate_mm_profile_document(document: Mapping[str, Any]) -> None:
    """Validate builder model documents used by the SQLite profile."""
    if document.get("version") != 3:
        raise ValueError("calibration model document version must be 3")
    for key in ("id", "device", "software_stack"):
        if not isinstance(document.get(key), str) or not document[key]:
            raise ValueError(f"MM calibration profile {key} must be a non-empty string")
    source = _as_mapping(document.get("source"), "MM calibration profile source")
    for key in ("database", "database_digest", "audit", "audit_sha256"):
        if not isinstance(source.get(key), str) or not source[key]:
            raise ValueError(f"MM calibration profile source.{key} must be a non-empty string")
    models = document.get("mm_models", [])
    if not isinstance(models, list):
        raise ValueError("MM calibration profile mm_models must be a list")
    gmm_models = document.get("gmm_models", [])
    if not isinstance(gmm_models, list):
        raise ValueError("MM calibration profile gmm_models must be a list")
    communication_models = document.get("communication_models", [])
    if not isinstance(communication_models, list):
        raise ValueError("MM calibration profile communication_models must be a list")
    attention_models = document.get("attention_models", [])
    if not isinstance(attention_models, list):
        raise ValueError("MM calibration profile attention_models must be a list")
    if not models and not gmm_models and not communication_models and not attention_models:
        raise ValueError(
            "MM calibration profile needs mm_models, gmm_models, communication_models, or attention_models"
        )
    if communication_models:
        communication_source = _as_mapping(document.get("communication_source"), "communication calibration source")
        for key in ("directory", "directory_digest", "audit", "audit_sha256"):
            if not isinstance(communication_source.get(key), str) or not communication_source[key]:
                raise ValueError(f"communication calibration source.{key} must be a non-empty string")
    identities: set[tuple[str, str, str]] = set()
    for index, model in enumerate(models):
        model = _as_mapping(model, f"MM model {index}")
        kind = model.get("kind")
        quantization = model.get("quantization", "none")
        compute_dtype = model.get("compute_dtype")
        if not all(isinstance(value, str) and value for value in (kind, quantization, compute_dtype)):
            raise ValueError(f"MM model {index} needs kind, quantization, and compute_dtype")
        identity = (kind, quantization, compute_dtype)
        if identity in identities:
            raise ValueError(f"duplicate MM model identity {identity!r}")
        identities.add(identity)
        calibration_mode = model.get("calibration_mode", "component_roofline")
        if calibration_mode not in {"component_roofline", "total_latency"}:
            raise ValueError(f"MM model {index} calibration_mode is unsupported")
        if "default" in model:
            raise ValueError(f"MM model {index} no longer supports a global default utilization")
        domain = _as_mapping(model.get("domain"), f"MM model {index} domain")
        if not domain:
            raise ValueError(f"MM model {index} domain must not be empty")
        for feature, bounds in domain.items():
            _half_open_selector(bounds, f"MM model {index} domain.{feature}")
        regions = model.get("regions", [])
        curves = model.get("curves", [])
        if not isinstance(regions, list) or not isinstance(curves, list):
            raise ValueError(f"MM model {index} regions and curves must be lists")
        region_ids: set[str] = set()
        region_boxes: list[tuple[str, dict[str, Mapping[str, Any]]]] = []
        for region_index, region in enumerate(regions):
            region = _as_mapping(region, f"MM model {index} region {region_index}")
            region_id = region.get("id")
            if not isinstance(region_id, str) or not region_id or region_id in region_ids:
                raise ValueError(f"MM model {index} region ids must be unique non-empty strings")
            region_ids.add(region_id)
            utilization = region.get("utilization")
            if not isinstance(utilization, (int, float)) or not 0 < utilization <= 1:
                raise ValueError(f"MM model {index} region {region_id} utilization must be in (0, 1]")
            if (
                not isinstance(region.get("fixed_overhead_us", 0), (int, float))
                or region.get("fixed_overhead_us", 0) < 0
            ):
                raise ValueError(f"MM model {index} region {region_id} fixed_overhead_us must be non-negative")
            when = _as_mapping(region.get("when", {}), f"MM model {index} region {region_id} when")
            box = {
                feature: _half_open_selector(bounds, f"MM model {index} domain.{feature}")
                for feature, bounds in domain.items()
            }
            for feature, bounds in when.items():
                selector = _half_open_selector(bounds, f"MM model {index} region {region_id} when.{feature}")
                box[feature] = (
                    _intersect_half_open_selectors(
                        box[feature], selector, f"MM model {index} region {region_id} when.{feature}"
                    )
                    if feature in box
                    else selector
                )
            region_boxes.append((region_id, box))
        for left_index, (left_id, left_box) in enumerate(region_boxes):
            for right_id, right_box in region_boxes[left_index + 1 :]:
                if _boxes_overlap(left_box, right_box):
                    raise ValueError(f"MM model {index} regions {left_id!r} and {right_id!r} overlap")
        curve_keys: set[tuple[int, int]] = set()
        for curve_index, curve in enumerate(curves):
            curve = _as_mapping(curve, f"MM model {index} curve {curve_index}")
            try:
                key = int(curve["k"]), int(curve["n"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"MM model {index} curve needs integer k and n") from error
            if min(key) <= 0 or key in curve_keys:
                raise ValueError(f"MM model {index} curve k/n must be positive and unique")
            curve_keys.add(key)
            _curve_points(curve.get("m_points"), f"MM model {index} curve {key}")
    _validate_gmm_models(gmm_models)
    _validate_communication_models(communication_models)
    _validate_attention_models(attention_models)


def _validate_attention_models(models: list[Any]) -> None:
    for index, value in enumerate(models):
        model = _as_mapping(value, f"attention model {index}")
        for key in ("id", "kernel_type", "dtype"):
            if not isinstance(model.get(key), str) or not model[key]:
                raise ValueError(f"attention model {index} needs {key}")
        tc_ops = model.get("tc_ops", [])
        if not isinstance(tc_ops, list) or not tc_ops or not all(isinstance(op, str) and op for op in tc_ops):
            raise ValueError(f"attention model {index} tc_ops must be a non-empty string list")
        curves = model.get("curves")
        if not isinstance(curves, list) or not curves:
            raise ValueError(f"attention model {index} curves must be a non-empty list")
        for curve_index, value in enumerate(curves):
            curve = _as_mapping(value, f"attention model {index} curve {curve_index}")
            if not isinstance(curve.get("features"), Mapping):
                raise ValueError(f"attention model {index} curve {curve_index} needs features")
            latency = curve.get("latency_us")
            if not isinstance(latency, (int, float)) or latency <= 0:
                raise ValueError(f"attention model {index} curve {curve_index} latency_us must be positive")


def _validate_communication_models(models: list[Any]) -> None:
    identities: set[tuple[str, str, int, int]] = set()
    for index, model_value in enumerate(models):
        model = _as_mapping(model_value, f"communication model {index}")
        try:
            collective = str(model["collective"])
            tc_op = str(model["tc_op"])
            dtype = str(model["dtype"])
            group_size = int(model["group_size"])
            topology_tier = int(model["topology_tier"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"communication model {index} needs collective/tc_op/dtype/group_size/topology_tier"
            ) from error
        if collective not in {"all_reduce", "all_gather", "reduce_scatter", "all_to_all"}:
            raise ValueError(f"communication model {index} has unsupported collective {collective!r}")
        if tc_op != f"tensor_cast.{collective}.default" or not dtype or group_size <= 1 or topology_tier < 0:
            raise ValueError(f"communication model {index} has an invalid identity")
        identity = collective, dtype, group_size, topology_tier
        if identity in identities:
            raise ValueError(f"duplicate communication model identity {identity!r}")
        identities.add(identity)
        points = model.get("latency_points_us")
        if not isinstance(points, list) or not points:
            raise ValueError(f"communication model {index} latency_points_us must be non-empty")
        normalized: list[tuple[int, float]] = []
        for point in points:
            if not isinstance(point, list) or len(point) != 2:
                raise ValueError(f"communication model {index} points must use [message_bytes, latency_us]")
            message_bytes, latency_us = point
            if not isinstance(message_bytes, int) or message_bytes <= 0:
                raise ValueError(f"communication model {index} message_bytes must be positive integers")
            if not isinstance(latency_us, (int, float)) or latency_us < 0:
                raise ValueError(f"communication model {index} latency_us must be non-negative")
            normalized.append((message_bytes, float(latency_us)))
        if any(left[0] >= right[0] for left, right in zip(normalized, normalized[1:])):
            raise ValueError(f"communication model {index} message_bytes must be strictly increasing")


def _validate_gmm_models(models: list[Any]) -> None:
    identities: set[tuple[str, str, str, str, str, str]] = set()
    for index, model_value in enumerate(models):
        model = _as_mapping(model_value, f"GMM model {index}")
        required = ("variant", "quantization", "input_dtype", "weight_dtype", "output_dtype", "distribution")
        if not all(isinstance(model.get(key), str) and model[key] for key in required):
            raise ValueError(f"GMM model {index} needs {', '.join(required)}")
        identity = tuple(str(model[key]) for key in required)
        if identity in identities:
            raise ValueError(f"duplicate GMM model identity {identity!r}")
        identities.add(identity)
        tc_ops = model.get("tc_ops")
        if not isinstance(tc_ops, list) or not tc_ops or not all(isinstance(op, str) and op for op in tc_ops):
            raise ValueError(f"GMM model {index} tc_ops must be a non-empty string list")
        if model.get("calibration_mode") != "total_latency":
            raise ValueError(f"GMM model {index} must use total_latency")
        if "default" in model:
            raise ValueError(f"GMM model {index} no longer supports a global default utilization")
        domain = _as_mapping(model.get("domain"), f"GMM model {index} domain")
        for feature in ("m_total", "k", "n", "gemm_n", "num_experts"):
            if feature not in domain:
                raise ValueError(f"GMM model {index} domain needs {feature}")
            _half_open_selector(domain[feature], f"GMM model {index} domain.{feature}")
        curves = model.get("curves", [])
        if not isinstance(curves, list) or not curves:
            raise ValueError(f"GMM model {index} curves must be a non-empty list")
        curve_keys: set[tuple[int, int, int, int]] = set()
        for curve_index, curve_value in enumerate(curves):
            curve = _as_mapping(curve_value, f"GMM model {index} curve {curve_index}")
            try:
                key = tuple(int(curve[field]) for field in ("num_experts", "k", "n", "gemm_n"))
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"GMM model {index} curve needs integer num_experts/k/n/gemm_n") from error
            if min(key) <= 0 or key in curve_keys:
                raise ValueError(f"GMM model {index} curve key must be positive and unique")
            curve_keys.add(key)
            _curve_points(curve.get("m_points"), f"GMM model {index} curve {key}")


def _curve_points(value: Any, location: str) -> list[tuple[float, float]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{location} m_points must be a non-empty list")
    points = []
    for point in value:
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError(f"{location} points must use [m, utilization]")
        m, utilization = point
        if not isinstance(m, (int, float)) or m <= 0:
            raise ValueError(f"{location} m must be positive")
        if not isinstance(utilization, (int, float)) or not 0 < utilization <= 1:
            raise ValueError(f"{location} utilization must be in (0, 1]")
        points.append((float(m), float(utilization)))
    points.sort()
    if any(left[0] == right[0] for left, right in zip(points, points[1:])):
        raise ValueError(f"{location} m values must be unique")
    return points


def _curve_utilization(points: list[tuple[float, float]], m: float) -> float:
    if len(points) == 1:
        return points[0][1]
    for left, right in zip(points, points[1:]):
        if left[0] <= m <= right[0]:
            if m == left[0]:
                return left[1]
            if m == right[0]:
                return right[1]
            ratio = (math.log(m) - math.log(left[0])) / (math.log(right[0]) - math.log(left[0]))
            normalized_left = left[0] / left[1]
            normalized_right = right[0] / right[1]
            normalized = math.exp(
                math.log(normalized_left) + ratio * (math.log(normalized_right) - math.log(normalized_left))
            )
            return m / normalized
    raise ValueError("curve query is outside its measured M range")


_UTILIZATION_EPSILON = 1e-6

# Shape-aware K/N fallback is only meaningful within a local log-shape
# neighbourhood.  A distance of 2.0 in log2 space bounds the combined
# deviation to roughly two binary scales before falling back to raw analytic.
_MAX_SHAPE_FALLBACK_DISTANCE = 2.0


def _logit(utilization: float) -> float:
    bounded = min(max(utilization, _UTILIZATION_EPSILON), 1 - _UTILIZATION_EPSILON)
    return math.log(bounded / (1 - bounded))


def _sigmoid(value: float) -> float:
    return 1 / (1 + math.exp(-value))


def _shape_coordinates(features: Mapping[str, Any], fields: tuple[str, ...]) -> tuple[float, ...]:
    values = tuple(float(features[field]) for field in fields)
    if any(value <= 0 for value in values):
        raise ValueError("shape-aware calibration requires positive dimensions")
    return tuple(math.log2(value) for value in values)


def _shape_distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return math.sqrt(sum((left_value - right_value) ** 2 for left_value, right_value in zip(left, right)))


def _triangle_interpolation(
    candidates: list[Mapping[str, Any]], target: tuple[float, float]
) -> tuple[float, list[Mapping[str, Any]]] | None:
    """Interpolate a utilization inside a measured K/N triangle in log-shape space."""
    best: tuple[float, float, list[Mapping[str, Any]]] | None = None
    for first_index, first in enumerate(candidates):
        for second_index in range(first_index + 1, len(candidates)):
            second = candidates[second_index]
            for third in candidates[second_index + 1 :]:
                ax, ay = first["coordinates"]
                bx, by = second["coordinates"]
                cx, cy = third["coordinates"]
                denominator = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
                if math.isclose(denominator, 0.0, abs_tol=1e-12):
                    continue
                tx, ty = target
                first_weight = ((by - cy) * (tx - cx) + (cx - bx) * (ty - cy)) / denominator
                second_weight = ((cy - ay) * (tx - cx) + (ax - cx) * (ty - cy)) / denominator
                third_weight = 1 - first_weight - second_weight
                weights = (first_weight, second_weight, third_weight)
                if min(weights) < -1e-9:
                    continue
                score = min(weights)
                selected = [first, second, third]
                utilization = _sigmoid(
                    sum(weight * _logit(item["utilization"]) for weight, item in zip(weights, selected))
                )
                if best is None or score > best[0]:
                    best = score, utilization, selected
    return None if best is None else (best[1], best[2])


def _shape_aware_utilization(
    data: Mapping[str, Any], signature: CalibrationSignature
) -> tuple[float, str, str, dict[str, Any]] | None:
    """Resolve an MM/GMM curve with shape-aware interpolation and safe fallback.

    M is always interpolated within measured bounds first.  K/N operations are
    then performed in log-shape space; curves outside the local neighbourhood
    are deliberately ignored so the caller can retain the raw analytic result.
    """
    features = signature.features
    is_gmm = data.get("rule_type") == "gmm_utilization"
    m_feature = "m_total" if is_gmm else "m"
    coordinate_fields = ("k", "n", "gemm_n") if is_gmm else ("k", "n")
    required = (m_feature, *coordinate_fields)
    if any(feature not in features or not isinstance(features[feature], (int, float)) for feature in required):
        return None
    target_m = float(features[m_feature])
    target_coordinates = _shape_coordinates(features, coordinate_fields)
    candidates: list[dict[str, Any]] = []
    for curve_value in data.get("curves", []):
        curve = _as_mapping(curve_value, "shape-aware curve")
        if is_gmm and curve.get("num_experts") != features.get("num_experts"):
            continue
        try:
            points = _curve_points(curve["m_points"], "shape-aware curve")
            curve_features = {field: int(curve[field]) for field in coordinate_fields}
        except (KeyError, TypeError, ValueError):
            continue
        if not points[0][0] <= target_m <= points[-1][0]:
            continue
        candidates.append(
            {
                "curve": curve,
                "features": curve_features,
                "points": points,
                "coordinates": _shape_coordinates(curve_features, coordinate_fields),
                "utilization": _curve_utilization(points, target_m),
            }
        )
    if not candidates:
        return None

    def source_details(items: list[Mapping[str, Any]]) -> list[dict[str, int]]:
        result = []
        for item in items:
            source = dict(item["features"])
            if is_gmm:
                source["num_experts"] = int(features["num_experts"])
            result.append(source)
        return result

    exact = [item for item in candidates if item["coordinates"] == target_coordinates]
    if exact:
        target_is_knot = any(target_m == point_m for point_m, _ in exact[0]["points"])
        return (
            exact[0]["utilization"],
            "curve",
            str(exact[0]["curve"].get("confidence", "measured")) if target_is_knot else "interpolated",
            {"interpolated_feature": m_feature, "source_curves": source_details(exact[:1])},
        )

    # Prefer a measured K/N simplex.  GMM keeps gemm_n fixed while performing
    # the K/N interpolation, after the required exact num_experts match.
    interpolation_candidates = candidates
    interpolation_target = target_coordinates
    if is_gmm:
        interpolation_candidates = [
            {**item, "coordinates": item["coordinates"][:2]}
            for item in candidates
            if item["features"]["gemm_n"] == features["gemm_n"]
        ]
        interpolation_target = target_coordinates[:2]
    interpolation = _triangle_interpolation(interpolation_candidates, interpolation_target)
    if interpolation is not None:
        utilization, sources = interpolation
        return (
            utilization,
            "kn_interpolation",
            "interpolated",
            {"interpolated_feature": m_feature, "source_curves": source_details(sources)},
        )

    nearby = []
    for item in candidates:
        distance = _shape_distance(item["coordinates"], target_coordinates)
        if distance <= _MAX_SHAPE_FALLBACK_DISTANCE:
            nearby.append((distance, item))
    nearby.sort(key=lambda item: item[0])
    if len(nearby) >= 2:
        selected = nearby[:4]
        weights = [1 / max(distance, 1e-12) for distance, _ in selected]
        utilization = _sigmoid(
            sum(weight * _logit(item["utilization"]) for weight, (_, item) in zip(weights, selected)) / sum(weights)
        )
        return (
            utilization,
            "knn_weighted",
            "extrapolated",
            {
                "interpolated_feature": m_feature,
                "source_curves": source_details([item for _, item in selected]),
                "shape_distance": selected[0][0],
            },
        )
    if nearby:
        distance, nearest = nearby[0]
        return (
            nearest["utilization"],
            "nearest_neighbor",
            "extrapolated",
            {
                "interpolated_feature": m_feature,
                "source_curves": source_details([nearest]),
                "shape_distance": distance,
            },
        )
    return None


@dataclass(frozen=True)
class _ConfiguredRule:
    profile_id: str
    data: Mapping[str, Any]
    priority: int
    specificity: int = 0
    device_compute_efficiency: float = 1.0

    def matches(self, signature: CalibrationSignature, software_stack: Optional[str]) -> tuple[bool, Optional[str]]:
        data = self.data
        if data.get("op_family") != signature.op_family:
            return False, "op_family_mismatch"
        tc_op = data.get("tc_op")
        tc_ops = data.get("tc_ops")
        if tc_op is not None and tc_op != signature.tc_op:
            return False, "tc_op_mismatch"
        if tc_ops is not None and signature.tc_op not in tc_ops:
            return False, "tc_op_mismatch"
        dtype = data.get("dtype")
        if dtype is not None and dtype != signature.dtype:
            return False, "dtype_mismatch"
        phase = data.get("phase")
        if phase is not None and phase != signature.features.get("phase"):
            return False, "phase_mismatch"
        stack = data.get("software_stack")
        if stack not in (None, "*") and stack != software_stack:
            return False, "stack_mismatch"
        for data_key, feature_key in (
            ("mm_kind", "mm_kind"),
            ("quantization", "quantization"),
            ("compute_dtype", "compute_dtype"),
            ("gmm_variant", "gmm_variant"),
            ("weight_dtype", "weight_dtype"),
            ("output_dtype", "output_dtype"),
            ("distribution", "distribution"),
        ):
            expected = data.get(data_key)
            if expected is not None and expected != signature.features.get(feature_key):
                return False, f"{feature_key}_mismatch"
        if data.get("type") == "shape_aware_utilization":
            if _shape_aware_utilization(data, signature) is None:
                return False, "no_shape_aware_match"
            return True, None
        shape = data.get("shape", {})
        shape = _as_mapping(shape, f"rule {data.get('id', '<unnamed>')} shape")
        for feature, selector in shape.items():
            if feature not in signature.features:
                return False, f"shape_feature_missing:{feature}"
            if not _range_matches(signature.features[feature], selector):
                return False, f"shape_out_of_range:{feature}"
        interpolation = data.get("interpolate")
        if interpolation is not None:
            interpolation = _as_mapping(interpolation, f"rule {data.get('id', '<unnamed>')} interpolate")
            feature = interpolation.get("feature")
            if not isinstance(feature, str) or feature not in signature.features:
                return False, f"shape_feature_missing:{feature}"
            value = signature.features[feature]
            points = _interpolation_points(interpolation)
            if not isinstance(value, (int, float)) or value < points[0][0] or value > points[-1][0]:
                return False, f"shape_out_of_range:{feature}"
        return True, None

    def build_rule(self, signature: CalibrationSignature) -> CalibrationRule:
        data = self.data
        rule_type = data.get("type")
        rule_id = str(data.get("id", rule_type or "unnamed"))
        confidence = str(data.get("confidence", "measured"))
        details = {"match_rule": rule_id}
        if data.get("match_level") is not None:
            details["match_level"] = data["match_level"]
        if rule_type == "attention_latency":
            return AttentionLatencyRule(
                latency_us=float(data["latency_us"]),
                kernel_type=str(data["kernel_type"]),
                profile_id=self.profile_id,
                rule_id=rule_id,
                confidence=confidence,
                details={"match_level": "attention_runtime", **details},
            )
        if rule_type == "mm_utilization":
            utilization = data.get("utilization")
            curve = data.get("curve")
            if curve is not None:
                points = _curve_points(curve, f"rule {rule_id} curve")
                curve_feature = str(data.get("curve_feature", "m"))
                utilization = _curve_utilization(points, float(signature.features[curve_feature]))
                details["interpolated_feature"] = curve_feature
            if utilization is None:
                raise ValueError(f"mm_utilization rule {rule_id} needs utilization or curve")
            rule_class = GmmUtilizationRule if data.get("rule_type") == "gmm_utilization" else MmaUtilizationRule
            return rule_class(
                utilization=float(utilization),
                device_compute_efficiency=self.device_compute_efficiency,
                profile_id=self.profile_id,
                rule_id=rule_id,
                fixed_overhead_us=float(data.get("fixed_overhead_us", 0.0)),
                calibration_mode=str(data.get("calibration_mode", "component_roofline")),
                rule_type=str(data.get("rule_type", "mm_utilization")),
                confidence=confidence,
                details=details,
            )
        if rule_type == "shape_aware_utilization":
            resolved = _shape_aware_utilization(data, signature)
            if resolved is None:  # Guarded by matches(); keep this failure explicit if the source changes.
                raise ValueError(f"shape-aware utilization rule {rule_id} has no eligible curve")
            utilization, match_level, confidence, shape_details = resolved
            details.update(shape_details)
            details["match_level"] = match_level
            rule_class = GmmUtilizationRule if data.get("rule_type") == "gmm_utilization" else MmaUtilizationRule
            return rule_class(
                utilization=utilization,
                device_compute_efficiency=self.device_compute_efficiency,
                profile_id=self.profile_id,
                rule_id=rule_id,
                fixed_overhead_us=0.0,
                calibration_mode=str(data.get("calibration_mode", "component_roofline")),
                rule_type=str(data.get("rule_type", "mm_utilization")),
                confidence=confidence,
                details=details,
            )
        if rule_type == "communication_latency_curve":
            points = data.get("latency_points_us")
            if not isinstance(points, list):
                raise ValueError(f"communication_latency_curve rule {rule_id} needs latency_points_us")
            return CommunicationLatencyCurveRule(
                points_us=tuple((int(point[0]), float(point[1])) for point in points),
                profile_id=self.profile_id,
                rule_id=rule_id,
                confidence=confidence,
                details=details,
            )
        raise ValueError(f"Unsupported calibration rule type: {rule_type!r}")


def _interpolation_points(interpolation: Mapping[str, Any]) -> list[tuple[float, float]]:
    """Normalize and validate interpolation points declared by a profile rule."""
    points = interpolation.get("points")
    if not isinstance(points, list) or len(points) < 2:
        raise ValueError("interpolate points must contain at least two entries")
    normalized: list[tuple[float, float]] = []
    for point in points:
        point = _as_mapping(point, "interpolate point")
        try:
            value = float(point["value"])
            factor = float(point["slowdown_factor"])
        except (KeyError, TypeError, ValueError) as err:
            raise ValueError("interpolate points need numeric value and slowdown_factor") from err
        if factor <= 0:
            raise ValueError("interpolate slowdown_factor must be positive")
        normalized.append((value, factor))
    normalized.sort()
    if any(left[0] == right[0] for left, right in zip(normalized, normalized[1:])):
        raise ValueError("interpolate point values must be unique")
    return normalized


def _interpolate(points: list[tuple[float, float]], value: float) -> float:
    """Linearly interpolate a value within the declared profile range."""
    for left, right in zip(points, points[1:]):
        if left[0] <= value <= right[0]:
            if value == left[0]:
                return left[1]
            if value == right[0]:
                return right[1]
            ratio = (value - left[0]) / (right[0] - left[0])
            return left[1] + ratio * (right[1] - left[1])
    raise ValueError("interpolation value is outside the declared range")


class ProfileCalibrationDataSource(CalibrationDataSource):
    """Resolve the device profile, then hardware-family profile."""

    def __init__(
        self,
        profile_path: Optional[str] = None,
        *,
        device_name: str,
        software_stack: Optional[str] = None,
        device_compute_efficiency: Optional[float] = None,
    ):
        self.device_name = device_name
        self.software_stack = software_stack
        if device_compute_efficiency is None:
            device = DeviceProfile.all_device_profiles.get(device_name)
            device_compute_efficiency = device.compute_efficiency if device is not None else 1.0
        self.device_compute_efficiency = float(device_compute_efficiency)
        self._last_lookup_diagnostic: dict[str, Any] = {"fallback_reason": "no_profile"}
        self._rules: list[_ConfiguredRule] = []
        self._sqlite_profile: SQLiteCalibrationProfile | None = None
        if profile_path:
            self._rules.extend(self._load(Path(profile_path)))

    @property
    def last_lookup_diagnostic(self) -> dict:
        return dict(self._last_lookup_diagnostic)

    def _load(self, path: Path) -> list[_ConfiguredRule]:
        if not path.is_file():
            raise ValueError(f"Analytic calibration profile does not exist: {path}")
        try:
            self._sqlite_profile = SQLiteCalibrationProfile(path)
        except (ValueError, OSError, UnicodeDecodeError, sqlite3.DatabaseError) as error:
            raise ValueError("analytic calibration profile must be a SQLite file") from error
        if self._sqlite_profile.device != self.device_name:
            self._sqlite_profile.close()
            self._sqlite_profile = None
            return []
        if self.software_stack is None:
            self.software_stack = self._sqlite_profile.software_stack
        elif self.software_stack != self._sqlite_profile.software_stack:
            self._sqlite_profile.close()
            self._sqlite_profile = None
            return []
        return self._load_sqlite_models(self._sqlite_profile.document())

    def _load_sqlite_models(self, document: Mapping[str, Any]) -> list[_ConfiguredRule]:
        """Build MM/GMM/communication rules from validated SQLite data."""
        if any("default" in _as_mapping(model, "MM model") for model in document.get("mm_models", [])):
            raise ValueError("MM model no longer supports a global default utilization")
        if document["device"] != self.device_name:
            return []
        profile_stack = str(document["software_stack"])
        if self.software_stack is None and profile_stack != "*":
            self.software_stack = profile_stack

        configured: list[_ConfiguredRule] = []
        for model_index, model_value in enumerate(document.get("mm_models", [])):
            model = _as_mapping(model_value, f"MM model {model_index}")
            common = {
                "type": "mm_utilization",
                "op_family": "matmul",
                "mm_kind": model["kind"],
                "quantization": model.get("quantization", "none"),
                "compute_dtype": model["compute_dtype"],
                "software_stack": profile_stack,
                "calibration_mode": model.get("calibration_mode", "component_roofline"),
            }
            domain = {
                feature: _half_open_selector(bounds, f"MM model {model_index} domain.{feature}")
                for feature, bounds in _as_mapping(model["domain"], f"MM model {model_index} domain").items()
            }
            for region_value in model.get("regions", []):
                region = _as_mapping(region_value, f"MM model {model_index} region")
                shape = dict(domain)
                for feature, bounds in _as_mapping(region.get("when", {}), "MM region when").items():
                    region_selector = _half_open_selector(bounds, f"MM region {region['id']} when.{feature}")
                    shape[feature] = (
                        _intersect_half_open_selectors(
                            shape[feature], region_selector, f"MM region {region['id']} when.{feature}"
                        )
                        if feature in shape
                        else region_selector
                    )
                configured.append(
                    _ConfiguredRule(
                        profile_id=str(document["id"]),
                        data={
                            **common,
                            "id": region["id"],
                            "shape": shape,
                            "utilization": region["utilization"],
                            "fixed_overhead_us": region.get("fixed_overhead_us", 0.0),
                            "confidence": region.get("confidence", "measured"),
                            "match_level": "region",
                        },
                        priority=20,
                        specificity=10,
                        device_compute_efficiency=self.device_compute_efficiency,
                    )
                )
            if model.get("curves"):
                configured.append(
                    _ConfiguredRule(
                        profile_id=str(document["id"]),
                        data={
                            **common,
                            "id": f"{model['kind']}-{model.get('quantization', 'none')}-{model['compute_dtype']}-shape-aware",
                            "type": "shape_aware_utilization",
                            "rule_type": "mm_utilization",
                            "curves": model["curves"],
                        },
                        priority=20,
                        specificity=20,
                        device_compute_efficiency=self.device_compute_efficiency,
                    )
                )
        configured.extend(self._load_gmm_models(document, profile_stack))
        configured.extend(self._load_communication_models(document, profile_stack))
        return configured

    def _load_communication_models(self, document: Mapping[str, Any], profile_stack: str) -> list[_ConfiguredRule]:
        configured: list[_ConfiguredRule] = []
        for index, model_value in enumerate(document.get("communication_models", [])):
            model = _as_mapping(model_value, f"communication model {index}")
            points = model["latency_points_us"]
            configured.append(
                _ConfiguredRule(
                    profile_id=str(document["id"]),
                    data={
                        "id": (
                            f"{model['collective']}-{model['dtype']}-g{model['group_size']}"
                            f"-tier{model['topology_tier']}"
                        ),
                        "type": "communication_latency_curve",
                        "op_family": "communication",
                        "tc_op": model["tc_op"],
                        "dtype": model["dtype"],
                        "software_stack": profile_stack,
                        "shape": {
                            "collective": model["collective"],
                            "group_size": int(model["group_size"]),
                            "topology_tier": int(model["topology_tier"]),
                            "message_bytes": {"min": points[0][0], "max": points[-1][0]},
                        },
                        "latency_points_us": points,
                        "confidence": model.get("confidence", "measured"),
                        "match_level": "communication_curve",
                    },
                    priority=20,
                    specificity=30,
                )
            )
        return configured

    def _load_gmm_models(self, document: Mapping[str, Any], profile_stack: str) -> list[_ConfiguredRule]:
        configured: list[_ConfiguredRule] = []
        for model_index, model_value in enumerate(document.get("gmm_models", [])):
            model = _as_mapping(model_value, f"GMM model {model_index}")
            common = {
                "type": "mm_utilization",
                "rule_type": "gmm_utilization",
                "op_family": "matmul",
                "mm_kind": "grouped",
                "gmm_variant": model["variant"],
                "quantization": model["quantization"],
                "compute_dtype": model["input_dtype"],
                "weight_dtype": model["weight_dtype"],
                "distribution": model["distribution"],
                "software_stack": profile_stack,
                "calibration_mode": "total_latency",
                "curve_feature": "m_total",
            }
            for tc_op in model["tc_ops"]:
                configured.append(
                    _ConfiguredRule(
                        profile_id=str(document["id"]),
                        data={
                            **common,
                            "tc_op": tc_op,
                            "id": f"{model['variant']}-{model['quantization']}-shape-aware",
                            "type": "shape_aware_utilization",
                            "rule_type": "gmm_utilization",
                            "curves": model["curves"],
                        },
                        priority=20,
                        specificity=30,
                        device_compute_efficiency=self.device_compute_efficiency,
                    )
                )
        return configured

    def lookup(self, signature: CalibrationSignature) -> Optional[CalibrationRule]:
        best: Optional[_ConfiguredRule] = None
        reasons: list[str] = []
        for configured in self._rules:
            matched, reason = configured.matches(signature, self.software_stack)
            if matched and (
                best is None or (configured.priority, configured.specificity) > (best.priority, best.specificity)
            ):
                best = configured
            elif reason:
                reasons.append(reason)
        attention_match = None
        if self._sqlite_profile is not None and signature.op_family == "attention":
            attention_match = self._sqlite_profile.lookup_attention(
                tc_op=signature.tc_op,
                dtype=signature.dtype,
                features=signature.features,
            )
        if best is None and attention_match is not None:
            self._last_lookup_diagnostic = {
                "profile_id": attention_match.profile_id,
                "match_rule": attention_match.rule_id,
                "confidence": attention_match.confidence,
            }
            return AttentionLatencyRule(
                latency_us=attention_match.latency_us,
                kernel_type=attention_match.kernel_type,
                profile_id=attention_match.profile_id,
                rule_id=attention_match.rule_id,
                confidence=attention_match.confidence,
                details={
                    "match_rule": attention_match.rule_id,
                    "match_level": "attention_runtime",
                    "sample_count": attention_match.sample_count,
                    "feature_names": list(attention_match.feature_names),
                },
            )
        if best is None:
            self._last_lookup_diagnostic = {
                "fallback_reason": next((reason for reason in reasons if reason.startswith("shape_out_of_range")), None)
                or next((reason for reason in reasons if reason in {"dtype_mismatch", "stack_mismatch"}), None)
                or next((reason for reason in reasons if reason == "no_shape_aware_match"), None)
                or "no_matching_rule",
            }
            return None
        self._last_lookup_diagnostic = {
            "profile_id": best.profile_id,
            "match_rule": best.data.get("id"),
            "confidence": best.data.get("confidence", "measured"),
        }
        return best.build_rule(signature)
