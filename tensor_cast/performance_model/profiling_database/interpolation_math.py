"""Stateless math helpers for profiling interpolation."""

from __future__ import annotations

import math
import logging
from typing import Any, Mapping, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


def validate_positive_latency(value: Any, *, min_latency_us: float = 0.0) -> bool:
    """Return True when value is finite and greater than a lower bound."""
    try:
        latency = float(value)
    except (TypeError, ValueError, OverflowError):
        return False
    return math.isfinite(latency) and latency > float(min_latency_us)


def validate_interpolated_latency(
    value: Any,
    candidate_latencies: Optional[Sequence[float]] = None,
) -> bool:
    """Return True when an interpolated latency is finite and positive."""
    if not validate_positive_latency(value):
        return False

    if candidate_latencies is None:
        return True

    return any(validate_positive_latency(latency) for latency in candidate_latencies)


def find_boundary(values: Sequence[float], target: float) -> Optional[tuple[float, float]]:
    """Find the closest lower/upper boundary values containing target.

    Returns None when target is outside the measured range. Phase 1 does not
    extrapolate.
    """
    finite_values = sorted({float(v) for v in values if math.isfinite(float(v))})
    if not finite_values or not math.isfinite(float(target)):
        return None

    below = [v for v in finite_values if v <= target]
    above = [v for v in finite_values if v >= target]
    if not below or not above:
        return None
    return max(below), min(above)


def linear_interp(x: float, x0: float, y0: float, x1: float, y1: float) -> float:
    """One-dimensional linear interpolation."""
    if x1 == x0:
        return float(y0)
    t = (x - x0) / (x1 - x0)
    return float(y0 + t * (y1 - y0))


def griddata_linear_interp(
    points: Sequence[Sequence[float]],
    values: Sequence[float],
    target: Sequence[float],
    *,
    return_details: bool = False,
) -> Optional[float] | tuple[Optional[float], dict[str, Any]]:
    """Linear 2D/3D interpolation through SciPy Delaunay simplices.

    Returns None for invalid data, degenerate point clouds, target outside the
    convex hull, scipy numerical failures, NaN, Inf, or negative latency.
    """
    details: dict[str, Any] = {}
    try:
        from scipy.interpolate import LinearNDInterpolator
    except ImportError as exc:
        logger.debug("griddata_linear_interp scipy unavailable; falling back to analytic path", exc_info=True)
        details["failure_reason"] = "scipy_unavailable"
        details["exception_type"] = type(exc).__name__
        return (None, details) if return_details else None

    point_array = np.asarray(points, dtype=float)
    value_array = np.asarray(values, dtype=float)
    target_array = np.asarray(target, dtype=float)

    if point_array.ndim != 2 or value_array.ndim != 1 or target_array.ndim != 1:
        details["failure_reason"] = "invalid_array_rank"
        return (None, details) if return_details else None
    if len(point_array) != len(value_array) or point_array.shape[1] != target_array.shape[0]:
        details["failure_reason"] = "shape_mismatch"
        return (None, details) if return_details else None
    dim = point_array.shape[1]
    if dim not in (2, 3) or len(point_array) < dim + 1:
        details["failure_reason"] = "insufficient_points"
        return (None, details) if return_details else None
    if not np.isfinite(point_array).all() or not np.isfinite(value_array).all() or not np.isfinite(target_array).all():
        details["failure_reason"] = "non_finite_input"
        return (None, details) if return_details else None
    if np.linalg.matrix_rank(point_array - point_array[0]) < dim:
        details["failure_reason"] = "degenerate_point_cloud"
        return (None, details) if return_details else None

    try:
        interpolator = LinearNDInterpolator(point_array, value_array)
        result = interpolator(target_array.reshape(1, -1))
        simplex_index = int(interpolator.tri.find_simplex(target_array))
    except Exception as exc:
        if isinstance(exc, MemoryError):
            logger.warning(
                "griddata_linear_interp encountered MemoryError with %d points; falling back to analytic path",
                len(point_array),
            )
            logger.debug("griddata_linear_interp MemoryError details", exc_info=True)
        else:
            logger.debug("griddata_linear_interp scipy exception; falling back to analytic path", exc_info=True)
        details["failure_reason"] = "scipy_exception"
        details["exception_type"] = type(exc).__name__
        return (None, details) if return_details else None

    if result is None or len(result) == 0:
        details["failure_reason"] = "empty_result"
        return (None, details) if return_details else None
    latency = float(result[0])
    if simplex_index < 0 or not math.isfinite(latency):
        details["failure_reason"] = "outside_convex_hull"
        return (None, details) if return_details else None

    simplex_indices = interpolator.tri.simplices[simplex_index]
    transform = interpolator.tri.transform[simplex_index]
    barycentric = np.dot(transform[:dim], target_array - transform[dim])
    barycentric = np.append(barycentric, 1.0 - barycentric.sum())

    details.update(
        {
            "simplex_vertex_indices": simplex_indices.tolist(),
            "barycentric_weights": barycentric.tolist(),
        }
    )
    if not validate_interpolated_latency(latency, value_array.tolist()):
        details["failure_reason"] = "latency_guard_rejected"
        return (None, details) if return_details else None
    return (latency, details) if return_details else latency


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def build_interpolation_details(
    *,
    method: str,
    interpolation_dim: int,
    axes: Sequence[str],
    target: Mapping[str, float],
    axis_boundary: Mapping[str, tuple[float, float]],
    candidate_points: Sequence[Mapping[str, float]],
    candidate_count: int,
    confidence: float,
    fallback_from: str = "exact_miss",
    exact_fields: Optional[Mapping[str, Any]] = None,
    axis_transform: Optional[str] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> dict:
    """Build a serializable details dict for QueryResult."""
    details = {
        "method": method,
        "interpolation_dim": interpolation_dim,
        "axes": list(axes),
        "target": dict(target),
        "axis_boundary": {axis: list(bounds) for axis, bounds in axis_boundary.items()},
        "candidate_points": list(candidate_points),
        "candidate_count": int(candidate_count),
        "confidence": float(confidence),
        "fallback_from": fallback_from,
    }
    if exact_fields:
        details["exact_fields"] = dict(exact_fields)
    if axis_transform:
        details["axis_transform"] = axis_transform
    if extra:
        details.update(extra)
    return _jsonable(details)
