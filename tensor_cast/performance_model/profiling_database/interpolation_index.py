"""Candidate point indexes for profiling interpolation."""

from __future__ import annotations

import copy
import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

from .interpolation_math import (
    build_interpolation_details,
    find_boundary,
    griddata_linear_interp,
    linear_interp,
    validate_interpolated_latency,
    validate_positive_latency,
)


RegimeKey = tuple[tuple[str, Any], ...]


@dataclass
class CandidatePoint:
    kernel_type: str
    axes: dict[str, float]
    latency_us: float
    regime_key: RegimeKey
    input_shapes: list[tuple[int, ...]] = field(default_factory=list)
    input_dtypes: list[str] = field(default_factory=list)
    input_formats: list[str] = field(default_factory=list)
    row_index: int = -1
    row_meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class InterpolationTarget:
    func_name: str
    kernel_type: str
    axes: dict[str, float]
    regime_key: RegimeKey
    tc_shapes: list[tuple[int, ...]] = field(default_factory=list)
    input_dtypes: list[str] = field(default_factory=list)
    query_mode: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class InterpolationResult:
    latency_us: float
    confidence: float
    method: str
    interpolation_dim: int
    axes: tuple[str, ...]
    details: dict[str, Any]
    shape_match_rule: str
    matched_points: list[CandidatePoint]


def make_regime_key(fields: Mapping[str, Any] | Iterable[tuple[str, Any]]) -> RegimeKey:
    """Build a stable tuple key from ordered fields."""
    if isinstance(fields, Mapping):
        items = fields.items()
    else:
        items = fields
    return tuple((str(k), _freeze_value(v)) for k, v in items)


def _freeze_value(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple((str(k), _freeze_value(v)) for k, v in sorted(value.items()))
    if isinstance(value, list):
        return tuple(_freeze_value(v) for v in value)
    if isinstance(value, tuple):
        return tuple(_freeze_value(v) for v in value)
    return value


def _key_dict(key: RegimeKey) -> dict[str, Any]:
    return {name: value for name, value in key}


class CandidateGroup:
    """Candidate points in one exact/regime group."""

    _CONFIDENCE_BY_DIM = {1: 0.70, 2: 0.65, 3: 0.60}

    def __init__(
        self,
        regime_key: RegimeKey,
        points: Sequence[CandidatePoint],
    ):
        self.regime_key = regime_key
        self.points = list(points)
        self.last_diagnostics: dict[str, Any] = {}

    def interpolate(
        self,
        target_axes: Mapping[str, float],
        axis_groups: Sequence[Sequence[str]],
        *,
        fallback_from: str = "exact_miss",
        axis_transform: Optional[str] = None,
        max_interpolation_dim: Optional[int] = None,
        extra_details: Optional[Mapping[str, Any]] = None,
        axis_matchers: Optional[Mapping[str, Callable[[float, float], bool]]] = None,
    ) -> Optional[InterpolationResult]:
        attempts: list[dict[str, Any]] = []
        for axis_group in axis_groups:
            axes = tuple(axis_group)
            if len(axes) == 0 or len(axes) > 3:
                continue
            if max_interpolation_dim is not None and len(axes) > int(max_interpolation_dim):
                attempts.append({"axes": list(axes), "status": "interpolation_dim_disabled"})
                continue
            if any(axis not in target_axes for axis in axes):
                attempts.append({"axes": list(axes), "status": "missing_target_axis"})
                continue

            candidates = self._filter_candidates(target_axes, axes, axis_matchers=axis_matchers)
            if len(candidates) < 2:
                attempts.append(
                    {
                        "axes": list(axes),
                        "status": "insufficient_filtered_candidates",
                        "candidate_count": len(candidates),
                    }
                )
                continue

            bounds = self._bounds_for_axes(candidates, target_axes, axes)
            if bounds is None:
                attempts.append(
                    {
                        "axes": list(axes),
                        "status": "outside_axis_boundary",
                        "candidate_count": len(candidates),
                    }
                )
                continue

            if len(axes) == 1:
                linear = self._try_1d(candidates, target_axes, axes, bounds)
                if linear is None:
                    attempts.append(
                        {
                            "axes": list(axes),
                            "status": "boundary_point_missing",
                            "candidate_count": len(candidates),
                        }
                    )
                    continue
                latency, matched_points = linear
                result = self._build_result(
                    latency,
                    "linear_1d",
                    axes,
                    target_axes,
                    bounds,
                    candidates,
                    matched_points,
                    fallback_from,
                    axis_transform,
                    extra_details,
                )
                if result is not None:
                    self.last_diagnostics = {
                        "attempts": attempts,
                        "selected_axes": list(axes),
                        "status": "success",
                    }
                    return result
                attempts.append(
                    {
                        "axes": list(axes),
                        "status": "latency_guard_rejected",
                        "candidate_count": len(candidates),
                    }
                )

            if len(axes) >= 2:
                grid_quality = self._grid_quality(candidates, axes)
                if grid_quality["rejected"]:
                    attempts.append(
                        {
                            "axes": list(axes),
                            "status": "grid_structure_rejected",
                            "candidate_count": len(candidates),
                            "quality": grid_quality,
                        }
                    )
                    continue
                griddata, grid_details = self._try_griddata(candidates, target_axes, axes)
                if griddata is not None:
                    latency, matched_points = griddata
                    grid_extra = {
                        **(extra_details or {}),
                        "grid_quality": grid_quality,
                        "simplex": grid_details,
                    }
                    result = self._build_result(
                        latency,
                        "griddata_linear",
                        axes,
                        target_axes,
                        bounds,
                        candidates,
                        matched_points,
                        fallback_from,
                        axis_transform,
                        grid_extra,
                    )
                    if result is not None:
                        self.last_diagnostics = {
                            "attempts": attempts,
                            "selected_axes": list(axes),
                            "status": "success",
                        }
                        return result
                    attempts.append(
                        {
                            "axes": list(axes),
                            "status": "latency_guard_rejected",
                            "candidate_count": len(candidates),
                        }
                    )
                else:
                    attempts.append(
                        {
                            "axes": list(axes),
                            "status": grid_details.get("failure_reason", "griddata_failed"),
                            "candidate_count": len(candidates),
                            "griddata": grid_details,
                        }
                    )

        self.last_diagnostics = {"attempts": attempts, "status": "failed"}
        return None

    def _filter_candidates(
        self,
        target_axes: Mapping[str, float],
        selected_axes: Sequence[str],
        *,
        axis_matchers: Optional[Mapping[str, Callable[[float, float], bool]]] = None,
    ) -> list[CandidatePoint]:
        selected = set(selected_axes)
        candidates = []
        for point in self.points:
            if any(axis not in point.axes for axis in selected_axes):
                continue
            matched = True
            for axis, target_value in target_axes.items():
                if axis in selected:
                    continue
                if axis not in point.axes:
                    matched = False
                    break
                point_value = point.axes[axis]
                matcher = axis_matchers.get(axis) if axis_matchers else None
                if matcher is not None:
                    if not matcher(float(point_value), float(target_value)):
                        matched = False
                        break
                    continue
                if point_value != target_value:
                    matched = False
                    break
            if matched and validate_positive_latency(point.latency_us):
                candidates.append(point)
        return candidates

    @staticmethod
    def _bounds_for_axes(
        candidates: Sequence[CandidatePoint],
        target_axes: Mapping[str, float],
        selected_axes: Sequence[str],
    ) -> Optional[dict[str, tuple[float, float]]]:
        bounds = {}
        for axis in selected_axes:
            values = [point.axes[axis] for point in candidates if axis in point.axes]
            boundary = find_boundary(values, float(target_axes[axis]))
            if boundary is None:
                return None
            lo, hi = boundary
            bounds[axis] = (lo, hi)
        return bounds

    @staticmethod
    def _point_map(
        candidates: Sequence[CandidatePoint],
        selected_axes: Sequence[str],
    ) -> dict[tuple[float, ...], CandidatePoint]:
        grouped: dict[tuple[float, ...], list[CandidatePoint]] = {}
        for point in candidates:
            coord = tuple(float(point.axes[axis]) for axis in selected_axes)
            grouped.setdefault(coord, []).append(point)

        by_coord: dict[tuple[float, ...], CandidatePoint] = {}
        for coord, points in grouped.items():
            if len(points) == 1:
                by_coord[coord] = points[0]
                continue
            merged = copy.copy(points[0])
            latencies = [float(point.latency_us) for point in points]
            merged.latency_us = float(statistics.median(latencies))
            merged.row_meta = {
                **points[0].row_meta,
                "duplicate_count": len(points),
                "duplicate_row_indices": [point.row_index for point in points],
                "duplicate_row_meta": [dict(point.row_meta) for point in points],
                "duplicate_latency_min_us": min(latencies),
                "duplicate_latency_max_us": max(latencies),
                "duplicate_latency_std_us": statistics.pstdev(latencies),
                "aggregation": "median",
            }
            by_coord[coord] = merged
        return by_coord

    def _try_1d(
        self,
        candidates: Sequence[CandidatePoint],
        target_axes: Mapping[str, float],
        selected_axes: Sequence[str],
        bounds: Mapping[str, tuple[float, float]],
    ) -> Optional[tuple[float, list[CandidatePoint]]]:
        by_coord = self._point_map(candidates, selected_axes)
        axis = selected_axes[0]
        lo, hi = bounds[axis]
        lower = by_coord.get((lo,))
        upper = by_coord.get((hi,))
        if lower is None or upper is None:
            return None
        latency = linear_interp(float(target_axes[axis]), lo, lower.latency_us, hi, upper.latency_us)
        matched = [lower] if lo == hi else [lower, upper]
        return latency, matched

    def _try_griddata(
        self,
        candidates: Sequence[CandidatePoint],
        target_axes: Mapping[str, float],
        selected_axes: Sequence[str],
    ) -> tuple[Optional[tuple[float, list[CandidatePoint]]], dict[str, Any]]:
        by_coord = CandidateGroup._point_map(candidates, selected_axes)
        point_coords = list(by_coord)
        unique_points = list(by_coord.values())
        latencies = [point.latency_us for point in unique_points]

        target = tuple(float(target_axes[axis]) for axis in selected_axes)
        latency, details = griddata_linear_interp(point_coords, latencies, target, return_details=True)
        if latency is None:
            return None, details
        simplex_indices = details.get("simplex_vertex_indices", [])
        matched_points = [unique_points[int(index)] for index in simplex_indices]
        return (latency, matched_points), details

    @staticmethod
    def _grid_quality(
        candidates: Sequence[CandidatePoint],
        selected_axes: Sequence[str],
    ) -> dict[str, Any]:
        by_coord = CandidateGroup._point_map(candidates, selected_axes)
        dim = len(selected_axes)
        min_unique_points = dim + 1
        axis_unique_counts = {
            axis: len({float(point.axes[axis]) for point in candidates if axis in point.axes}) for axis in selected_axes
        }
        degenerate_axes = [axis for axis, count in axis_unique_counts.items() if count < 2]
        rejected_reason = None
        if len(by_coord) < min_unique_points:
            rejected_reason = "unique_point_count_lt_min"
        elif degenerate_axes:
            rejected_reason = "degenerate_axes"
        return {
            "unique_point_count": len(by_coord),
            "min_unique_point_count": min_unique_points,
            "axis_unique_counts": axis_unique_counts,
            "degenerate_axes": degenerate_axes,
            "rejected": rejected_reason is not None,
            "rejected_reason": rejected_reason,
        }

    @staticmethod
    def _shape_rule(method: str, dim: int) -> str:
        if method == "linear_1d":
            return "interpolated_1d_linear"
        return f"interpolated_{dim}d_griddata_linear"

    def _build_result(
        self,
        latency: float,
        method: str,
        selected_axes: tuple[str, ...],
        target_axes: Mapping[str, float],
        bounds: Mapping[str, tuple[float, float]],
        candidates: Sequence[CandidatePoint],
        matched_points: Sequence[CandidatePoint],
        fallback_from: str,
        axis_transform: Optional[str],
        extra_details: Optional[Mapping[str, Any]],
    ) -> Optional[InterpolationResult]:
        candidate_latencies = [point.latency_us for point in matched_points] or [
            point.latency_us for point in candidates
        ]
        if not validate_interpolated_latency(latency, candidate_latencies):
            return None

        dim = len(selected_axes)
        confidence = self._CONFIDENCE_BY_DIM.get(dim, 0.60)
        target = {axis: float(target_axes[axis]) for axis in selected_axes}
        candidate_points = [
            {axis: float(point.axes[axis]) for axis in selected_axes if axis in point.axes} for point in matched_points
        ]
        filter_info = self._axis_filter_info(candidates, target_axes, selected_axes)
        details_extra = {**(extra_details or {}), **filter_info}
        matched_row_meta = [dict(point.row_meta) for point in matched_points if point.row_meta]
        if matched_row_meta:
            details_extra["matched_row_meta"] = matched_row_meta
        exact_axis_values = (
            {axis: bounds[axis][0] for axis in selected_axes if bounds[axis][0] == bounds[axis][1]}
            if method == "linear_1d"
            else {}
        )
        if exact_axis_values:
            details_extra["exact_axis_value"] = exact_axis_values
        exact_coordinate_match = any(
            all(float(point.axes[axis]) == float(target_axes[axis]) for axis in selected_axes)
            for point in matched_points
        )
        details_extra["exact_coordinate_match"] = exact_coordinate_match
        if axis_transform is not None:
            pre_transform = self._pre_transform_info(matched_points, target_axes, selected_axes, bounds, extra_details)
            details_extra.update(pre_transform)
        effective_bounds = {
            axis: (
                min(float(point.axes[axis]) for point in matched_points),
                max(float(point.axes[axis]) for point in matched_points),
            )
            for axis in selected_axes
        }
        details = build_interpolation_details(
            method=method,
            interpolation_dim=dim,
            axes=selected_axes,
            target=target,
            axis_boundary=effective_bounds,
            candidate_points=candidate_points,
            candidate_count=len(candidates),
            confidence=confidence,
            fallback_from=fallback_from,
            exact_fields=dict(self.regime_key),
            axis_transform=axis_transform,
            extra=details_extra,
        )
        return InterpolationResult(
            latency_us=float(latency),
            confidence=confidence,
            method=method,
            interpolation_dim=dim,
            axes=selected_axes,
            details=details,
            shape_match_rule=self._shape_rule(method, dim),
            matched_points=list(matched_points),
        )

    @staticmethod
    def _axis_filter_info(
        candidates: Sequence[CandidatePoint],
        target_axes: Mapping[str, float],
        selected_axes: Sequence[str],
    ) -> dict[str, Any]:
        selected = set(selected_axes)
        non_selected_axes = [axis for axis in target_axes if axis not in selected]
        effective_filters = []
        unfiltered_due_to_missing_axis = []
        for axis in non_selected_axes:
            present_count = sum(1 for point in candidates if axis in point.axes)
            if present_count:
                effective_filters.append(axis)
            if present_count < len(candidates):
                unfiltered_due_to_missing_axis.append(axis)
        return {
            "effective_filters": effective_filters,
            "unfiltered_due_to_missing_axis": unfiltered_due_to_missing_axis,
        }

    @staticmethod
    def _pre_transform_info(
        matched_points: Sequence[CandidatePoint],
        target_axes: Mapping[str, float],
        selected_axes: Sequence[str],
        bounds: Mapping[str, tuple[float, float]],
        extra_details: Optional[Mapping[str, Any]],
    ) -> dict[str, Any]:
        if extra_details is None:
            raise ValueError("axis_transform interpolation requires pre-transform axes in extra_details")
        if "axes_pre_transform" in extra_details:
            original_target_axes = dict(extra_details["axes_pre_transform"])
        elif "attention_axes" in extra_details:
            original_target_axes = dict(extra_details["attention_axes"])
        else:
            raise ValueError("axis_transform interpolation requires axes_pre_transform or attention_axes")
        missing_axes = [axis for axis in selected_axes if axis not in original_target_axes]
        if missing_axes:
            raise ValueError(f"axis_transform interpolation missing pre-transform axes: {missing_axes}")

        target_pre_transform = {}
        axis_boundary_pre_transform = {}
        for axis in selected_axes:
            target_pre_transform[axis] = float(original_target_axes[axis])
            values = []
            for point in matched_points:
                original_axes = point.row_meta.get("pre_transform_axes", point.axes)
                if axis in original_axes:
                    values.append(float(original_axes[axis]))
            if values:
                axis_boundary_pre_transform[axis] = [min(values), max(values)]
            else:
                axis_boundary_pre_transform[axis] = [float(bounds[axis][0]), float(bounds[axis][1])]

        return {
            "target_pre_transform": target_pre_transform,
            "axis_boundary_pre_transform": axis_boundary_pre_transform,
        }


class CandidateIndex:
    """Interpolation candidate index grouped by exact/regime key."""

    def __init__(self, points: Sequence[CandidatePoint]):
        self.points = list(points)
        self._candidate_groups: dict[RegimeKey, CandidateGroup] = {}
        grouped: dict[RegimeKey, list[CandidatePoint]] = {}
        for point in self.points:
            grouped.setdefault(point.regime_key, []).append(point)
        for key, group_points in grouped.items():
            self._candidate_groups[key] = CandidateGroup(key, group_points)

    def candidate_groups_matching(
        self,
        target_key: RegimeKey,
        *,
        allow_extra_fields: Optional[set[str]] = None,
        required_target_fields: Optional[set[str]] = None,
    ) -> list[CandidateGroup]:
        target = _key_dict(target_key)
        allow_extra_fields = allow_extra_fields or set()
        required_target_fields = required_target_fields or set()
        if any(field not in target for field in required_target_fields):
            return []
        matches = []
        for key, candidate_group in self._candidate_groups.items():
            candidate = _key_dict(key)
            mismatch = False
            for required_field in required_target_fields:
                if required_field in target and required_field not in candidate:
                    mismatch = True
                    break
            if mismatch:
                continue
            for key_field, value in candidate.items():
                if key_field in target:
                    if target[key_field] != value:
                        mismatch = True
                        break
                elif key_field not in allow_extra_fields:
                    mismatch = True
                    break
            if mismatch:
                continue
            matches.append(candidate_group)
        return matches
