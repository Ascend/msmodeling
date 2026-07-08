"""Profiling datasource wrapper for measured-data interpolation.

The wrapper preserves ProfilingDataSource exact hits. When the base datasource
returns PARTIAL or None, it attempts Phase 1 interpolation for supported compute
and attention_special operators. Communication interpolation remains owned by
ProfilingDataSource.
"""

import hashlib
import json
import logging
import math
import weakref
from dataclasses import replace
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import pandas as pd
import torch

from .data_source import DataSourcePerformanceModel, QueryResult, QuerySource, ShapeMatchInfo
from .interpolation_index import (
    CandidateGroup,
    CandidateIndex,
    CandidatePoint,
    InterpolationResult,
    InterpolationTarget,
    make_regime_key,
)
from .interpolation_math import validate_positive_latency
from .profiling_data_source import (
    _DTYPE_COMPAT,
    _DTYPE_RELAXED_KERNELS,
    _dtype_byte_size,
    _is_block_padded,
    _MATMUL_KERNELS,
    _normalize_fia_q_shape,
    _normalize_func_name,
    _parse_shape_str,
    _parse_str_list,
    _strip_batch_dim,
    COMPOSITE_DECOMPOSERS,
    DTYPE_MAP,
    fractal_nz_to_nd,
    ProfilingDataSource,
)

if TYPE_CHECKING:
    from ..op_invoke_info import OpInvokeInfo

logger = logging.getLogger(__name__)

_BATCHED_MATMUL_KERNELS = frozenset({"BatchMatMulV2", "BatchMatMulNd", "TransposeBatchMatMul", "QuantBatchMatmulV3"})
_INTERPOLATION_MATMUL_KERNELS = _MATMUL_KERNELS | frozenset({"BatchMatMulNd"})
_UNKNOWN_SPARSE_MODE = -1
_UNKNOWN_KV_HEADS = -1


def _to_int_cell(value: Any) -> Optional[int]:
    try:
        if value is None or pd.isna(value):
            return None
    except TypeError:
        pass
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(numeric):
        return None
    return int(numeric)


def _optional_str_cell(value: Any) -> Optional[str]:
    try:
        if value is None or pd.isna(value):
            return None
    except TypeError:
        pass
    text = str(value).strip()
    return text or None


def _finite_positive_latency(value: Any) -> Optional[float]:
    try:
        latency = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if math.isfinite(latency) and latency > 0:
        return latency
    return None


def _candidate_latency_cols(preferred_col: str) -> tuple[str, ...]:
    cols = (
        preferred_col,
        "Profiling Average Duration(us)",
        "Profiling Median Duration(us)",
        "Average Duration(us)",
        "Median Duration(us)",
        "Duration(us)",
    )
    return tuple(dict.fromkeys(cols))


def _infer_attention_input_layout(q_shape: Tuple[int, ...], head_dim: int = 0) -> Optional[str]:
    if len(q_shape) == 4:
        return "BNSD_NBSD"
    if len(q_shape) == 3:
        return "TND"
    if len(q_shape) == 2 and _normalize_fia_q_shape(q_shape, head_dim) is not None:
        return "TND"
    return None


def _infer_attention_sparse_mode(query_lens: Any, input_layout: Optional[str]) -> Optional[int]:
    if input_layout == "BNSD_NBSD":
        return 0
    if input_layout == "TND" or query_lens is not None:
        return 3
    return None


def _attention_kv_heads_from_key(key: Any, input_layout: Optional[str]) -> Optional[int]:
    if not isinstance(key, torch.Tensor) or key.ndim < 2:
        return None
    if input_layout == "BNSD_NBSD" and key.ndim >= 4:
        return int(key.shape[1])
    if input_layout == "TND" and key.ndim >= 3:
        return int(key.shape[-2])
    return int(key.shape[-2]) if key.ndim >= 2 else None


def _explicit_attention_quant_mode(
    op_invoke_info: "OpInvokeInfo",
    mapping: dict,
    kernel_override_quant_mode: Optional[str],
) -> Optional[str]:
    if kernel_override_quant_mode is not None:
        return str(kernel_override_quant_mode)
    if mapping.get("quant_mode") is not None:
        return str(mapping["quant_mode"])
    kwargs = getattr(op_invoke_info, "kwargs", {}) or {}
    if kwargs.get("quant_mode") is not None:
        return str(kwargs["quant_mode"])
    return None


_COMPUTE_AXIS_GROUPS = (
    ("M",),
    ("K",),
    ("N",),
    ("M", "K"),
    ("M", "N"),
    ("K", "N"),
    ("M", "K", "N"),
)
_GENERIC_COMPUTE_AXIS_GROUPS = (("axis_0",),)
_GENERIC_COMPUTE_AXIS_0 = "axis_0"
_GENERIC_COMPUTE_OUTPUT_NUMEL_AXIS = "output_numel"
_LATENCY_SOURCE_SELECTED = "selected_column"
_LATENCY_SOURCE_FALLBACK = "fallback_column"

_ATTENTION_AXES = (
    "seq",
    "batch",
    "heads",
    "head_dim",
)
_ATTENTION_AXIS_GROUPS = tuple(axes for dim in range(1, 4) for axes in combinations(_ATTENTION_AXES, dim))


class InterpolatingDataSource(DataSourcePerformanceModel):
    """Datasource decorator that adds interpolation fallback.

    When the base ProfilingDataSource returns PARTIAL or None, this layer tries
    supported interpolation paths and returns QuerySource.INTERPOLATED only
    after a complete interpolation result is available.
    """

    def __init__(self, base: ProfilingDataSource):
        self.base = base
        ip = self.base._op_mapping.get("interpolation_policy", {})
        self._policy_hash = self._stable_digest(ip)
        self._kernel_overrides = ip.get("kernel_overrides", {})
        self._compute_index_cache: Dict[tuple[Any, ...], CandidateIndex] = {}
        self._attention_index_cache: Dict[tuple[Any, ...], CandidateIndex] = {}
        self._elementwise_index_cache: Dict[tuple[Any, ...], CandidateIndex] = {}
        self._dataframe_fingerprint_cache: Dict[int, tuple[weakref.ReferenceType, str]] = {}
        self._compute_index_diagnostics: Dict[str, dict[str, Any]] = {}
        self._attention_index_diagnostics: Dict[str, dict[str, Any]] = {}
        self._last_miss_reason = ""
        self._last_miss_details: dict[str, Any] = {}

    @property
    def last_miss_reason(self) -> str:
        return self._last_miss_reason or self.base.last_miss_reason

    @property
    def last_miss_details(self) -> dict[str, Any]:
        return dict(self._last_miss_details)

    def _record_miss(self, reason: str, **details: Any) -> None:
        prior_attempts = list(self._last_miss_details.get("miss_history", []))
        attempt = {"reason": reason, **details}
        self._last_miss_reason = reason
        self._last_miss_details = {**details, "miss_reason": reason, "miss_history": [*prior_attempts, attempt]}

    def lookup(self, op_invoke_info: "OpInvokeInfo") -> Optional[QueryResult]:
        self._last_miss_reason = ""
        self._last_miss_details = {}
        result = self.base.lookup(op_invoke_info)
        if result is not None and result.source != QuerySource.PARTIAL:
            return result
        # PARTIAL or None: try interpolation.
        fallback_from = "partial" if result is not None and result.source == QuerySource.PARTIAL else "exact_miss"
        interp_result = self._interpolate(op_invoke_info, fallback_from=fallback_from)
        if interp_result is not None:
            # Mark as interpolated so runtime writes shape_match_rule="interpolated"
            if interp_result.shape_match_info is None:
                return replace(
                    interp_result,
                    shape_match_info=ShapeMatchInfo(
                        simulation_shapes=[],
                        kernel_shapes=[],
                        shape_match_rule="interpolated",
                    ),
                )
            return interp_result
        # Interpolation failed; return None so empirical falls back to analytic.
        if not self._last_miss_reason:
            self._record_miss(
                "wrapper_interpolation_failed",
                base_miss_reason=self.base.last_miss_reason,
                fallback_from=fallback_from,
            )
        return None

    def _interpolate(
        self, op_invoke_info: "OpInvokeInfo", *, fallback_from: str = "exact_miss"
    ) -> Optional[QueryResult]:
        """Determine which query path to use and dispatch to the right interpolator."""
        func_str = _normalize_func_name(op_invoke_info.func)
        mappings = self.base._op_mapping.get("operator_mappings", {})
        mapping = mappings.get(func_str)
        if mapping is None:
            self._record_miss(
                "wrapper_unmapped",
                op_name=func_str,
                base_miss_reason=self.base.last_miss_reason,
            )
            return None

        # Don't interpolate zero_cost or accepted_miss ops
        if mapping.get("zero_cost") or mapping.get("accepted_miss"):
            self._record_miss(
                "wrapper_not_applicable",
                op_name=func_str,
                base_miss_reason=self.base.last_miss_reason,
            )
            return None

        # Composite ops: decompose into sub-kernels, interpolate each
        if mapping.get("composite"):
            return self._interpolate_composite(op_invoke_info, mapping, func_str)

        if mapping.get("category") == "communication":
            # Comm interpolation handled by base's _query_comm_csv alpha-beta model
            self._record_miss(
                "wrapper_communication_owned_by_base",
                op_name=func_str,
                base_miss_reason=self.base.last_miss_reason,
            )
            return None
        if mapping.get("query_mode") == "moe_fused":
            # Phase 1 compute interpolation does not model EP Size regimes yet.
            self._record_miss(
                "wrapper_moe_fused_disabled",
                op_name=func_str,
                base_miss_reason=self.base.last_miss_reason,
            )
            return None
        if mapping.get("query_mode") == "attention_special":
            return self._interpolate_attention(op_invoke_info, mapping, fallback_from=fallback_from)
        if mapping.get("query_mode") == "elementwise":
            return self._interpolate_elementwise(op_invoke_info, mapping, fallback_from=fallback_from)
        return self._interpolate_compute(op_invoke_info, mapping, fallback_from=fallback_from)

    # ---- Multidimensional index helpers ----

    @staticmethod
    def _dtype_key(kernel_type: str, dtype_str: str) -> str:
        if kernel_type in _DTYPE_RELAXED_KERNELS:
            return _DTYPE_COMPAT.get(dtype_str, dtype_str)
        return dtype_str

    @staticmethod
    def _stable_digest(value: Any) -> str:
        payload = json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _compute_dataframe_fingerprint(df: pd.DataFrame) -> str:
        digest = hashlib.sha256()
        digest.update(str(df.shape).encode("utf-8"))
        digest.update(json.dumps([str(column) for column in df.columns], separators=(",", ":")).encode("utf-8"))
        try:
            row_hashes = pd.util.hash_pandas_object(df, index=True).to_numpy(dtype="uint64", copy=False)
        except (TypeError, ValueError):
            row_hashes = pd.util.hash_pandas_object(df.astype(str), index=True).to_numpy(dtype="uint64", copy=False)
        digest.update(row_hashes.tobytes())
        return digest.hexdigest()

    def _dataframe_fingerprint(self, df: pd.DataFrame) -> str:
        cached = self._dataframe_fingerprint_cache.get(id(df))
        if cached is not None and cached[0]() is df:
            return cached[1]
        fingerprint = self._compute_dataframe_fingerprint(df)
        self._dataframe_fingerprint_cache[id(df)] = (weakref.ref(df), fingerprint)
        return fingerprint

    @staticmethod
    def _logical_csv_shape(shape: Tuple[int, ...], fmt: str) -> Tuple[int, ...]:
        if fmt == "FRACTAL_NZ":
            return fractal_nz_to_nd(shape)
        return tuple(shape)

    @staticmethod
    def _normalize_matmul_shape(shape: Tuple[int, ...], *, batched: bool) -> Tuple[int, ...]:
        """Normalize explicit batch=1 without destroying a valid 2D M=1 matrix."""
        shape = tuple(shape)
        if batched:
            return shape
        if len(shape) >= 3 and shape[0] == 1:
            return shape[1:]
        return shape

    def _candidate_latency(self, row: Any, latency_col: str) -> tuple[Optional[float], dict[str, Any]]:
        first_rejection: Optional[dict[str, Any]] = None
        for column in _candidate_latency_cols(latency_col):
            try:
                raw_value = row[column]
            except KeyError:
                continue
            latency = _finite_positive_latency(raw_value)
            if latency is not None:
                return latency, {
                    "latency_column": column,
                    "latency_selection": "selected_column" if column == latency_col else "fallback_column",
                    "raw_latency_us": latency,
                }
            try:
                numeric_value = float(raw_value)
            except (TypeError, ValueError, OverflowError):
                numeric_value = None
            if numeric_value is not None and math.isfinite(numeric_value) and numeric_value == 0.0:
                if first_rejection is None:
                    first_rejection = {
                        "latency_rejected_reason": "latency_zero",
                        "latency_column": column,
                        "raw_latency_us": numeric_value,
                    }
                continue
            if first_rejection is None:
                first_rejection = {
                    "latency_rejected_reason": "latency_invalid",
                    "latency_column": column,
                    "raw_latency_us": numeric_value if numeric_value is not None else raw_value,
                }
        return None, first_rejection or {"latency_rejected_reason": "latency_invalid"}

    @staticmethod
    def _extract_matmul_axes_from_shapes(
        kernel_type: str,
        input_shapes: List[Tuple[int, ...]],
    ) -> Optional[tuple[dict[str, float], tuple[tuple[int, ...], tuple[int, ...]], str]]:
        if len(input_shapes) < 2:
            return None
        batched = kernel_type in _BATCHED_MATMUL_KERNELS
        lhs = InterpolatingDataSource._normalize_matmul_shape(tuple(input_shapes[0]), batched=batched)
        rhs = InterpolatingDataSource._normalize_matmul_shape(tuple(input_shapes[1]), batched=batched)
        if len(lhs) < 2 or len(rhs) < 2:
            return None

        m_dim = lhs[-2]
        k_dim = lhs[-1]
        if rhs[-2] == k_dim:
            n_dim = rhs[-1]
            source_layout = "rhs_k_n"
        elif rhs[-1] == k_dim:
            # Profiling rows sometimes store matmul weights as (N, K) while
            # TensorCast sees (K, N).
            n_dim = rhs[-2]
            source_layout = "rhs_n_k"
        else:
            return None

        axes = {"M": float(m_dim), "K": float(k_dim), "N": float(n_dim)}
        batch_dims = (tuple(lhs[:-2]), tuple(rhs[:-2]))
        return axes, batch_dims, source_layout

    def _candidate_from_compute_row_with_reason(
        self,
        row: Any,
        kernel_type: str,
        latency_col: str,
        row_index: int,
        tc_input_count: Optional[int],
    ) -> tuple[Optional[CandidatePoint], Optional[str]]:
        csv_shapes = _parse_shape_str(str(row.get("Input Shapes", "")))
        if tc_input_count is not None:
            csv_shapes = csv_shapes[:tc_input_count]
        if len(csv_shapes) < 2:
            return None, "input_shapes_lt_2"

        csv_dtypes = _parse_str_list(str(row.get("Input Data Types", "")))
        csv_formats = _parse_str_list(str(row.get("Input Formats", "")))
        if tc_input_count is not None:
            csv_dtypes = csv_dtypes[:tc_input_count]
            csv_formats = csv_formats[:tc_input_count]
        if len(csv_dtypes) < len(csv_shapes):
            return None, "input_dtypes_missing"
        if len(csv_formats) < len(csv_shapes):
            return None, "input_formats_missing"

        logical_shapes = [self._logical_csv_shape(tuple(shape), csv_formats[i]) for i, shape in enumerate(csv_shapes)]
        axes_and_batch = self._extract_matmul_axes_from_shapes(kernel_type, logical_shapes)
        if axes_and_batch is None:
            return None, "matmul_axes_unextractable"
        axes, batch_dims, source_layout = axes_and_batch

        latency, latency_meta = self._candidate_latency(row, latency_col)
        if latency is None:
            return None, str(latency_meta["latency_rejected_reason"])

        input_count = tc_input_count if tc_input_count is not None else len(csv_shapes)
        dtype_key = tuple(self._dtype_key(kernel_type, dtype) for dtype in csv_dtypes[:input_count])
        regime_key = make_regime_key(
            [
                ("kernel_type", kernel_type),
                ("input_count", input_count),
                ("input_dtypes", dtype_key),
                ("batch_dims", batch_dims),
                ("input_formats", tuple(csv_formats[:input_count])),
            ]
        )
        return CandidatePoint(
            kernel_type=kernel_type,
            axes=axes,
            latency_us=latency,
            regime_key=regime_key,
            input_shapes=logical_shapes,
            input_dtypes=csv_dtypes[:input_count],
            input_formats=csv_formats[:input_count],
            row_index=row_index,
            row_meta={"batch_dims": batch_dims, "source_layout": source_layout, **latency_meta},
        ), None

    def _get_compute_index(
        self,
        kernel_type: str,
        tc_input_count: Optional[int],
    ) -> Optional[CandidateIndex]:
        df = self.base._load_csv(kernel_type)
        if df is None:
            return None
        cache_key = (kernel_type, tc_input_count, self._dataframe_fingerprint(df), self._policy_hash)
        if cache_key in self._compute_index_cache:
            return self._compute_index_cache[cache_key]
        latency_col = self.base._latency_col(df)
        points = []
        rejected_reasons: dict[str, int] = {}
        for row_index, (_, row) in enumerate(df.iterrows()):
            point, reason = self._candidate_from_compute_row_with_reason(
                row, kernel_type, latency_col, row_index, tc_input_count
            )
            if point is not None:
                points.append(point)
            elif reason:
                rejected_reasons[reason] = rejected_reasons.get(reason, 0) + 1
        self._compute_index_diagnostics[kernel_type] = {
            "csv_rows": len(df),
            "usable_points": len(points),
            "rejected_reasons": rejected_reasons,
        }
        index = CandidateIndex(points)
        self._compute_index_cache[cache_key] = index
        return index

    def _build_compute_target(
        self,
        op_invoke_info: "OpInvokeInfo",
        mapping: dict,
        kernel_type: str,
    ) -> Optional[InterpolationTarget]:
        if kernel_type not in _INTERPOLATION_MATMUL_KERNELS:
            return None

        tc_inputs = self.base._extract_tensor_inputs(op_invoke_info)
        tc_input_count = mapping.get("tc_input_count")
        if tc_input_count is not None:
            tc_inputs = tc_inputs[:tc_input_count]
        if len(tc_inputs) < 2:
            return None

        input_shapes = [tuple(shape) for shape, _ in tc_inputs]
        axes_and_batch = self._extract_matmul_axes_from_shapes(kernel_type, input_shapes)
        if axes_and_batch is None:
            return None
        axes, batch_dims, _source_layout = axes_and_batch

        dtype_values = []
        for _, dtype in tc_inputs:
            dtype_str = DTYPE_MAP.get(dtype)
            if dtype_str is None:
                return None
            dtype_values.append(self._dtype_key(kernel_type, dtype_str))

        input_count = tc_input_count if tc_input_count is not None else len(tc_inputs)
        regime_key = make_regime_key(
            [
                ("kernel_type", kernel_type),
                ("input_count", input_count),
                ("input_dtypes", tuple(dtype_values[:input_count])),
                ("batch_dims", batch_dims),
            ]
        )
        return InterpolationTarget(
            func_name=_normalize_func_name(op_invoke_info.func),
            kernel_type=kernel_type,
            axes=axes,
            regime_key=regime_key,
            tc_shapes=input_shapes,
            input_dtypes=dtype_values[:input_count],
            query_mode="compute",
        )

    def _build_compute_target_from_shapes(
        self,
        kernel_type: str,
        input_shapes: List[Tuple[int, ...]],
        dtype_str: str,
        *,
        tc_input_count: Optional[int] = None,
        func_name: Optional[str] = None,
        query_mode: str = "compute",
    ) -> Optional[InterpolationTarget]:
        if kernel_type not in _INTERPOLATION_MATMUL_KERNELS:
            return None
        input_shapes = [tuple(shape) for shape in input_shapes]
        if tc_input_count is not None:
            input_shapes = input_shapes[:tc_input_count]
        if len(input_shapes) < 2:
            return None

        axes_and_batch = self._extract_matmul_axes_from_shapes(kernel_type, input_shapes)
        if axes_and_batch is None:
            return None
        axes, batch_dims, _source_layout = axes_and_batch

        input_count = len(input_shapes)
        dtype_key = tuple(self._dtype_key(kernel_type, dtype_str) for _ in range(input_count))
        regime_key = make_regime_key(
            [
                ("kernel_type", kernel_type),
                ("input_count", input_count),
                ("input_dtypes", dtype_key),
                ("batch_dims", batch_dims),
            ]
        )
        return InterpolationTarget(
            func_name=func_name or kernel_type,
            kernel_type=kernel_type,
            axes=axes,
            regime_key=regime_key,
            tc_shapes=input_shapes,
            input_dtypes=list(dtype_key),
            query_mode=query_mode,
        )

    @staticmethod
    def _matched_axis_shapes(result: InterpolationResult) -> List[List[float]]:
        return [
            [float(point.axes[axis]) for axis in result.axes if axis in point.axes] for point in result.matched_points
        ]

    def _query_result_from_interpolation(
        self,
        target: InterpolationTarget,
        result: InterpolationResult,
    ) -> QueryResult:
        details = {
            **result.details,
            "kernel_type": target.kernel_type,
            "query_mode": target.query_mode,
        }
        return QueryResult(
            latency_us=result.latency_us,
            confidence=result.confidence,
            source=QuerySource.INTERPOLATED,
            details=details,
            shape_match_info=ShapeMatchInfo(
                simulation_shapes=[list(shape) for shape in target.tc_shapes],
                kernel_shapes=self._matched_axis_shapes(result),
                shape_match_rule=result.shape_match_rule,
            ),
        )

    @staticmethod
    def _candidate_failure_reason(default_reason: str, diagnostics: Optional[dict[str, Any]]) -> str:
        attempts = (diagnostics or {}).get("attempts") or []
        fallback_status = None
        if attempts:
            for attempt in attempts:
                status = attempt.get("status")
                if not status:
                    continue
                if fallback_status is None:
                    fallback_status = str(status)
                if status != "missing_target_axis":
                    return str(status)
        if fallback_status is not None:
            return fallback_status
        return default_reason

    @staticmethod
    def _generic_compute_shape_signature(input_shapes: List[Tuple[int, ...]]) -> tuple[Any, ...]:
        if not input_shapes:
            return ()
        first = tuple(input_shapes[0])
        return (first[1:], tuple(tuple(shape) for shape in input_shapes[1:]))

    def _generic_compute_policy(self, kernel_type: str, policy_kernel_type: Optional[str] = None) -> dict[str, Any]:
        kernel_types = [kernel_type]
        if policy_kernel_type is not None and policy_kernel_type not in kernel_types:
            kernel_types.append(policy_kernel_type)
        for kt in kernel_types:
            override = self._kernel_overrides.get(kt, {})
            policy = override.get("generic_compute", {})
            if isinstance(policy, dict) and policy:
                return policy
        return {}

    def _generic_compute_axis_name(self, kernel_type: str, policy_kernel_type: Optional[str] = None) -> str:
        axis = self._generic_compute_policy(kernel_type, policy_kernel_type).get("axis", _GENERIC_COMPUTE_AXIS_0)
        if axis == _GENERIC_COMPUTE_OUTPUT_NUMEL_AXIS:
            return _GENERIC_COMPUTE_OUTPUT_NUMEL_AXIS
        return _GENERIC_COMPUTE_AXIS_0

    def _generic_compute_axis_groups(
        self,
        kernel_type: str,
        policy_kernel_type: Optional[str] = None,
    ) -> tuple[tuple[str, ...], ...]:
        return ((self._generic_compute_axis_name(kernel_type, policy_kernel_type),),)

    @staticmethod
    def _shape_numel(shape: Tuple[int, ...]) -> Optional[int]:
        numel = 1
        for dim in shape:
            if dim < 0:
                return None
            numel *= int(dim)
        return numel

    @staticmethod
    def _extract_output_shapes(output: Any) -> List[Tuple[int, ...]]:
        if isinstance(output, torch.Tensor):
            return [tuple(output.shape)]
        if isinstance(output, (list, tuple)):
            shapes = []
            for item in output:
                if isinstance(item, torch.Tensor):
                    shapes.append(tuple(item.shape))
            return shapes
        return []

    @staticmethod
    def _canonical_output_numel_shape(shape: Tuple[int, ...]) -> Tuple[int, ...]:
        shape = tuple(shape)
        if len(shape) >= 3 and shape[0] == 1:
            return shape[1:]
        return shape

    def _generic_compute_axes_and_regime_with_reason(
        self,
        kernel_type: str,
        logical_shapes: List[Tuple[int, ...]],
        output_shapes: Optional[List[Tuple[int, ...]]] = None,
        policy_kernel_type: Optional[str] = None,
    ) -> tuple[Optional[tuple[dict[str, float], list[tuple[str, Any]]]], Optional[str]]:
        axis_name = self._generic_compute_axis_name(kernel_type, policy_kernel_type)
        if axis_name == _GENERIC_COMPUTE_OUTPUT_NUMEL_AXIS:
            if not output_shapes:
                return None, "generic_compute_output_shape_unavailable"
            if len(output_shapes) != 1:
                return None, "generic_compute_output_numel_multi_output_unsupported"
            output_shape = self._canonical_output_numel_shape(tuple(output_shapes[0]))
            numel = self._shape_numel(output_shape)
            if numel is None:
                return None, "generic_compute_output_shape_invalid"
            return (
                {_GENERIC_COMPUTE_OUTPUT_NUMEL_AXIS: float(numel)},
                [("output_tail_shape", output_shape[1:])],
            ), None
        if not logical_shapes or not logical_shapes[0]:
            return None, "generic_compute_input_shape_unavailable"
        return (
            {_GENERIC_COMPUTE_AXIS_0: float(logical_shapes[0][0])},
            [("shape_signature", self._generic_compute_shape_signature(logical_shapes))],
        ), None

    def _generic_compute_axes_and_regime(
        self,
        kernel_type: str,
        logical_shapes: List[Tuple[int, ...]],
        output_shapes: Optional[List[Tuple[int, ...]]] = None,
        policy_kernel_type: Optional[str] = None,
    ) -> Optional[tuple[dict[str, float], list[tuple[str, Any]]]]:
        axes_and_regime, _reason = self._generic_compute_axes_and_regime_with_reason(
            kernel_type,
            logical_shapes,
            output_shapes,
            policy_kernel_type,
        )
        return axes_and_regime

    def _build_generic_compute_target_from_shapes_with_reason(
        self,
        kernel_type: str,
        input_shapes: List[Tuple[int, ...]],
        dtype_str: str,
        *,
        dtype_values: Optional[List[str]] = None,
        output_shapes: Optional[List[Tuple[int, ...]]] = None,
        tc_input_count: Optional[int] = None,
        func_name: Optional[str] = None,
        query_mode: str = "compute",
        policy_kernel_type: Optional[str] = None,
    ) -> tuple[Optional[InterpolationTarget], Optional[str]]:
        input_shapes = [tuple(shape) for shape in input_shapes]
        if tc_input_count is not None:
            input_shapes = input_shapes[:tc_input_count]
        if not input_shapes:
            return None, "generic_compute_input_shape_unavailable"
        logical_shapes = [tuple(_strip_batch_dim(shape)) for shape in input_shapes]
        if not logical_shapes[0]:
            return None, "generic_compute_input_shape_unavailable"
        input_count = len(logical_shapes)
        if dtype_values is None:
            dtype_key = tuple(self._dtype_key(kernel_type, dtype_str) for _ in range(input_count))
        else:
            if len(dtype_values) < input_count:
                return None, "generic_compute_dtype_unavailable"
            dtype_key = tuple(self._dtype_key(kernel_type, dtype) for dtype in dtype_values[:input_count])
        axes_and_extra_regime, reason = self._generic_compute_axes_and_regime_with_reason(
            kernel_type,
            logical_shapes,
            output_shapes,
            policy_kernel_type,
        )
        if axes_and_extra_regime is None:
            return None, reason
        axes, extra_regime = axes_and_extra_regime
        regime_key = make_regime_key(
            [
                ("kernel_type", kernel_type),
                ("input_count", input_count),
                ("input_dtypes", dtype_key),
                *extra_regime,
            ]
        )
        return InterpolationTarget(
            func_name=func_name or kernel_type,
            kernel_type=kernel_type,
            axes=axes,
            regime_key=regime_key,
            tc_shapes=input_shapes,
            input_dtypes=list(dtype_key),
            query_mode=query_mode,
        ), None

    def _build_generic_compute_target_from_shapes(
        self,
        kernel_type: str,
        input_shapes: List[Tuple[int, ...]],
        dtype_str: str,
        *,
        dtype_values: Optional[List[str]] = None,
        output_shapes: Optional[List[Tuple[int, ...]]] = None,
        tc_input_count: Optional[int] = None,
        func_name: Optional[str] = None,
        query_mode: str = "compute",
        policy_kernel_type: Optional[str] = None,
    ) -> Optional[InterpolationTarget]:
        target, _reason = self._build_generic_compute_target_from_shapes_with_reason(
            kernel_type,
            input_shapes,
            dtype_str,
            dtype_values=dtype_values,
            output_shapes=output_shapes,
            tc_input_count=tc_input_count,
            func_name=func_name,
            query_mode=query_mode,
            policy_kernel_type=policy_kernel_type,
        )
        return target

    def _build_generic_compute_target(
        self,
        op_invoke_info: "OpInvokeInfo",
        mapping: dict,
        kernel_type: str,
        policy_kernel_type: Optional[str] = None,
    ) -> Optional[InterpolationTarget]:
        tc_inputs = self.base._extract_tensor_inputs(op_invoke_info)
        tc_input_count = mapping.get("tc_input_count")
        if tc_input_count is not None:
            tc_inputs = tc_inputs[:tc_input_count]
        if not tc_inputs:
            return None
        input_shapes = [tuple(shape) for shape, _ in tc_inputs]
        dtype_values = []
        for _, dtype in tc_inputs:
            dtype_str = DTYPE_MAP.get(dtype)
            if dtype_str is None:
                return None
            dtype_values.append(dtype_str)
        return self._build_generic_compute_target_from_shapes(
            kernel_type,
            input_shapes,
            dtype_values[0] if dtype_values else "",
            dtype_values=dtype_values,
            output_shapes=self._extract_output_shapes(getattr(op_invoke_info, "out", None)),
            tc_input_count=None,
            func_name=_normalize_func_name(op_invoke_info.func),
            query_mode="compute",
            policy_kernel_type=policy_kernel_type,
        )

    def _candidate_from_generic_compute_row_with_reason(
        self,
        row: Any,
        kernel_type: str,
        latency_col: str,
        row_index: int,
        tc_input_count: Optional[int],
        policy_kernel_type: Optional[str] = None,
    ) -> tuple[Optional[CandidatePoint], Optional[str]]:
        csv_shapes = _parse_shape_str(str(row.get("Input Shapes", "")))
        csv_dtypes = _parse_str_list(str(row.get("Input Data Types", "")))
        csv_formats = _parse_str_list(str(row.get("Input Formats", "")))
        if tc_input_count is not None:
            csv_shapes = csv_shapes[:tc_input_count]
            csv_dtypes = csv_dtypes[:tc_input_count]
            csv_formats = csv_formats[:tc_input_count]
        if not csv_shapes:
            return None, "input_shapes_missing"
        if len(csv_dtypes) < len(csv_shapes):
            return None, "input_dtypes_missing"
        if len(csv_formats) < len(csv_shapes):
            return None, "input_formats_missing"

        logical_shapes = [self._logical_csv_shape(tuple(shape), csv_formats[i]) for i, shape in enumerate(csv_shapes)]
        logical_shapes = [tuple(_strip_batch_dim(shape)) for shape in logical_shapes]
        if not logical_shapes[0]:
            return None, "input_shape_empty"
        output_shapes = _parse_shape_str(str(row.get("Output Shapes", "")))
        output_shapes = [tuple(shape) for shape in output_shapes]
        axes_and_extra_regime, reason = self._generic_compute_axes_and_regime_with_reason(
            kernel_type,
            logical_shapes,
            output_shapes,
            policy_kernel_type,
        )
        if axes_and_extra_regime is None:
            return None, reason or "generic_compute_axis_unextractable"
        axes, extra_regime = axes_and_extra_regime
        latency, latency_meta = self._candidate_latency(row, latency_col)
        if latency is None:
            return None, str(latency_meta["latency_rejected_reason"])

        input_count = len(logical_shapes)
        dtype_key = tuple(self._dtype_key(kernel_type, dtype) for dtype in csv_dtypes[:input_count])
        regime_key = make_regime_key(
            [
                ("kernel_type", kernel_type),
                ("input_count", input_count),
                ("input_dtypes", dtype_key),
                *extra_regime,
            ]
        )
        return CandidatePoint(
            kernel_type=kernel_type,
            axes=axes,
            latency_us=latency,
            regime_key=regime_key,
            input_shapes=logical_shapes,
            input_dtypes=csv_dtypes[:input_count],
            input_formats=csv_formats[:input_count],
            row_index=row_index,
            row_meta={**latency_meta},
        ), None

    def _get_generic_compute_index(
        self,
        kernel_type: str,
        tc_input_count: Optional[int],
        policy_kernel_type: Optional[str] = None,
    ) -> Optional[CandidateIndex]:
        df = self.base._load_csv(kernel_type)
        if df is None:
            return None
        effective_policy_kernel_type = policy_kernel_type or kernel_type
        cache_key = (
            "generic_compute",
            kernel_type,
            effective_policy_kernel_type,
            tc_input_count,
            self._dataframe_fingerprint(df),
            self._policy_hash,
        )
        if cache_key in self._compute_index_cache:
            return self._compute_index_cache[cache_key]
        latency_col = self.base._latency_col(df)
        points = []
        rejected_reasons: dict[str, int] = {}
        for row_index, (_, row) in enumerate(df.iterrows()):
            point, reason = self._candidate_from_generic_compute_row_with_reason(
                row,
                kernel_type,
                latency_col,
                row_index,
                tc_input_count,
                policy_kernel_type,
            )
            if point is not None:
                points.append(point)
            elif reason:
                rejected_reasons[reason] = rejected_reasons.get(reason, 0) + 1
        self._compute_index_diagnostics[kernel_type] = {
            "csv_rows": len(df),
            "usable_points": len(points),
            "rejected_reasons": rejected_reasons,
        }
        index = CandidateIndex(points)
        self._compute_index_cache[cache_key] = index
        return index

    def _interpolate_generic_compute_target(
        self,
        target: InterpolationTarget,
        tc_input_count: Optional[int],
        *,
        fallback_from: str,
        interpolation_path: str,
        policy_kernel_type: Optional[str] = None,
    ) -> Optional[QueryResult]:
        index = self._get_generic_compute_index(target.kernel_type, tc_input_count, policy_kernel_type)
        if index is None:
            self._record_miss(
                "compute_csv_not_found",
                kernel_type=target.kernel_type,
                interpolation_path=interpolation_path,
            )
            return None
        candidate_groups = index.candidate_groups_matching(target.regime_key)
        if not candidate_groups:
            self._record_miss(
                "regime_key_unmatched",
                kernel_type=target.kernel_type,
                interpolation_path=interpolation_path,
                target_axes=target.axes,
                target_regime_key=dict(target.regime_key),
                index_diagnostics=self._compute_index_diagnostics.get(target.kernel_type, {}),
            )
            return None
        attempts: list[dict[str, Any]] = []
        override = self._kernel_overrides.get(target.kernel_type, {})
        for candidate_group in candidate_groups:
            for source_attempt, source_group in self._source_pure_candidate_group_attempts(candidate_group):
                result = source_group.interpolate(
                    target.axes,
                    self._generic_compute_axis_groups(target.kernel_type, policy_kernel_type),
                    fallback_from=fallback_from,
                    max_interpolation_dim=override.get("max_interpolation_dim"),
                    extra_details={
                        "kernel_type": target.kernel_type,
                        "query_mode": target.query_mode,
                        "interpolation_path": interpolation_path,
                        "latency_source_attempt": source_attempt,
                    },
                )
                if result is None:
                    attempts.append(
                        {
                            "regime_key": dict(source_group.regime_key),
                            "latency_source_attempt": source_attempt,
                            "diagnostics": source_group.last_diagnostics,
                        }
                    )
                    continue
                return self._query_result_from_interpolation(target, result)
        self._record_miss(
            self._candidate_failure_reason(
                "candidate_group_failed", attempts[-1].get("diagnostics") if attempts else {}
            ),
            kernel_type=target.kernel_type,
            interpolation_path=interpolation_path,
            attempts=attempts,
            target_axes=target.axes,
        )
        return None

    @staticmethod
    def _compute_candidate_group_rank(candidate_group: CandidateGroup) -> tuple[int, int]:
        fields = dict(candidate_group.regime_key)
        input_formats = fields.get("input_formats")
        if isinstance(input_formats, tuple) and input_formats and all(fmt == "ND" for fmt in input_formats):
            return 0, -len(candidate_group.points)
        return 1, -len(candidate_group.points)

    @staticmethod
    def _candidate_latency_source(point: CandidatePoint) -> Optional[str]:
        value = point.row_meta.get("latency_selection")
        if value in {_LATENCY_SOURCE_SELECTED, _LATENCY_SOURCE_FALLBACK}:
            return str(value)
        return None

    @staticmethod
    def _attention_q_tokens_match(candidate_value: float, target_value: float) -> bool:
        candidate = int(candidate_value)
        target = int(target_value)
        return candidate == target or _is_block_padded(candidate, target) or _is_block_padded(target, candidate)

    @classmethod
    def _source_pure_candidate_group_attempts(cls, candidate_group: CandidateGroup) -> list[tuple[str, CandidateGroup]]:
        selected_points: list[CandidatePoint] = []
        fallback_points: list[CandidatePoint] = []
        unknown_points: list[CandidatePoint] = []
        for point in candidate_group.points:
            source = cls._candidate_latency_source(point)
            if source == _LATENCY_SOURCE_SELECTED:
                selected_points.append(point)
            elif source == _LATENCY_SOURCE_FALLBACK:
                fallback_points.append(point)
            else:
                unknown_points.append(point)

        if not selected_points and not fallback_points:
            return [("all", candidate_group)]

        attempts: list[tuple[str, CandidateGroup]] = []
        for label, points in (
            ("selected_only", selected_points),
            ("fallback_only", fallback_points),
            ("unknown_only", unknown_points),
        ):
            if points:
                attempts.append((label, CandidateGroup(candidate_group.regime_key, points)))
        return attempts

    def _interpolate_compute_target(
        self,
        target: InterpolationTarget,
        tc_input_count: Optional[int],
        *,
        fallback_from: str,
        interpolation_path: str,
    ) -> Optional[QueryResult]:
        index = self._get_compute_index(target.kernel_type, tc_input_count)
        if index is None:
            self._record_miss(
                "compute_csv_not_found",
                kernel_type=target.kernel_type,
                interpolation_path=interpolation_path,
            )
            return None
        candidate_groups = index.candidate_groups_matching(target.regime_key, allow_extra_fields={"input_formats"})
        if not candidate_groups:
            self._record_miss(
                "regime_key_unmatched",
                kernel_type=target.kernel_type,
                interpolation_path=interpolation_path,
                target_axes=target.axes,
                target_regime_key=dict(target.regime_key),
                index_diagnostics=self._compute_index_diagnostics.get(target.kernel_type, {}),
            )
            return None

        attempts: list[dict[str, Any]] = []
        override = self._kernel_overrides.get(target.kernel_type, {})
        for candidate_group in sorted(candidate_groups, key=self._compute_candidate_group_rank):
            for source_attempt, source_group in self._source_pure_candidate_group_attempts(candidate_group):
                result = source_group.interpolate(
                    target.axes,
                    _COMPUTE_AXIS_GROUPS,
                    fallback_from=fallback_from,
                    max_interpolation_dim=override.get("max_interpolation_dim"),
                    extra_details={
                        "kernel_type": target.kernel_type,
                        "query_mode": target.query_mode,
                        "interpolation_path": interpolation_path,
                        "latency_source_attempt": source_attempt,
                    },
                )
                if result is None:
                    attempts.append(
                        {
                            "regime_key": dict(source_group.regime_key),
                            "latency_source_attempt": source_attempt,
                            "diagnostics": source_group.last_diagnostics,
                        }
                    )
                    continue
                logger.debug(
                    "INTERPOLATED compute op=%s kernel=%s dim=%d axes=%s method=%s confidence=%.2f",
                    target.func_name,
                    target.kernel_type,
                    result.interpolation_dim,
                    ",".join(result.axes),
                    result.method,
                    result.confidence,
                )
                return self._query_result_from_interpolation(target, result)

        self._record_miss(
            self._candidate_failure_reason(
                "candidate_group_failed", attempts[-1].get("diagnostics") if attempts else {}
            ),
            kernel_type=target.kernel_type,
            interpolation_path=interpolation_path,
            attempts=attempts,
            target_axes=target.axes,
        )
        return None

    def _interpolate_compute_multidim(
        self,
        op_invoke_info: "OpInvokeInfo",
        mapping: dict,
        *,
        fallback_from: str = "exact_miss",
    ) -> Optional[QueryResult]:
        kernel_type = mapping.get("kernel_type")
        if not kernel_type:
            return None
        kernel_types = [kernel_type]
        for alt in mapping.get("alternate_kernel_types", []):
            if alt not in kernel_types:
                kernel_types.append(alt)

        tc_input_count = mapping.get("tc_input_count")
        attempts: list[dict[str, Any]] = []
        for kt in kernel_types:
            target = self._build_compute_target(op_invoke_info, mapping, kt)
            if target is None:
                attempts.append({"kernel_type": kt, "status": "target_unavailable"})
                continue
            result = self._interpolate_compute_target(
                target,
                tc_input_count,
                fallback_from=fallback_from,
                interpolation_path="multidim",
            )
            if result is not None:
                return result
            attempts.append(
                {
                    "kernel_type": kt,
                    "status": self.last_miss_reason or "candidate_group_failed",
                    "miss_details": self.last_miss_details,
                }
            )
        self._record_miss(
            "compute_multidim_interpolation_failed",
            attempted_kernel_types=kernel_types,
            attempts=attempts,
        )
        return None

    def _candidate_from_attention_row(
        self,
        row: Any,
        kernel_type: str,
        latency_col: str,
        row_index: int,
        avg_seq_col: str,
        *,
        has_sparse_col: bool,
        has_kv_heads_col: bool,
        has_layout_col: bool,
        has_quant_col: bool,
    ) -> Optional[CandidatePoint]:
        seq_value = _to_int_cell(row.get(avg_seq_col))
        if seq_value is None or seq_value < 0:
            return None

        input_shapes = _parse_shape_str(str(row.get("Input Shapes", "")))
        q_raw = input_shapes[0] if input_shapes else None
        if q_raw is None:
            return None
        csv_head_dim = input_shapes[1][-1] if len(input_shapes) > 1 and input_shapes[1] else q_raw[-1]
        q_3d = _normalize_fia_q_shape(q_raw, csv_head_dim)
        if q_3d is None:
            return None

        csv_dtypes = _parse_str_list(str(row.get("Input Data Types", "")))
        if not csv_dtypes:
            return None

        latency, latency_meta = self._candidate_latency(row, latency_col)
        if latency is None:
            return None

        axes = {
            "q_tokens": float(q_3d[0]),
            "seq": float(seq_value),
            "heads": float(q_3d[1]),
            "head_dim": float(q_3d[2]),
        }
        if "Runtime batch_size" in row.index:
            batch_value = _to_int_cell(row.get("Runtime batch_size"))
        elif len(q_raw) == 4 and q_raw[2] == 1:
            batch_value = int(q_raw[0])
        else:
            batch_value = None
        if batch_value is not None:
            axes["batch"] = float(batch_value)

        key_fields: list[tuple[str, Any]] = [
            ("kernel_type", kernel_type),
            ("dtype", csv_dtypes[0]),
        ]
        sparse_value = _to_int_cell(row.get("Runtime sparse_mode")) if has_sparse_col else None
        kv_heads_value = _to_int_cell(row.get("Runtime num_key_value_heads")) if has_kv_heads_col else None
        if has_sparse_col:
            key_fields.append(("sparse_mode", sparse_value if sparse_value is not None else _UNKNOWN_SPARSE_MODE))
        if has_kv_heads_col:
            key_fields.append(("kv_heads", kv_heads_value if kv_heads_value is not None else _UNKNOWN_KV_HEADS))
        if has_layout_col:
            layout_value = _optional_str_cell(row.get("Runtime input_layout"))
            if layout_value is not None:
                key_fields.append(("input_layout", layout_value))
        if has_quant_col:
            quant_value = _optional_str_cell(row.get("Runtime quant_mode"))
            if quant_value is not None:
                key_fields.append(("quant_mode", quant_value))

        return CandidatePoint(
            kernel_type=kernel_type,
            axes=axes,
            latency_us=latency,
            regime_key=make_regime_key(key_fields),
            input_shapes=[q_3d],
            input_dtypes=[csv_dtypes[0]],
            row_index=row_index,
            row_meta={
                "q_shape_3d": q_3d,
                "sparse_mode": sparse_value,
                "kv_heads": kv_heads_value,
                **latency_meta,
            },
        )

    def _get_attention_index(self, kernel_type: str) -> Optional[CandidateIndex]:
        df = self.base._load_csv(kernel_type)
        if df is None:
            return None
        cache_key = (kernel_type, self._dataframe_fingerprint(df), self._policy_hash)
        if cache_key in self._attention_index_cache:
            return self._attention_index_cache[cache_key]
        if "Runtime avg_seq_len" in df.columns:
            avg_seq_col = "Runtime avg_seq_len"
        elif "avg_seq_len" in df.columns:
            avg_seq_col = "avg_seq_len"
        else:
            return None
        if "Input Shapes" not in df.columns:
            return None

        latency_col = self.base._latency_col(df)
        has_sparse_col = "Runtime sparse_mode" in df.columns
        has_kv_heads_col = "Runtime num_key_value_heads" in df.columns
        has_layout_col = "Runtime input_layout" in df.columns
        has_quant_col = "Runtime quant_mode" in df.columns
        self._attention_index_diagnostics[kernel_type] = {
            "csv_rows": len(df),
            "has_sparse_col": has_sparse_col,
            "has_kv_heads_col": has_kv_heads_col,
            "has_layout_col": has_layout_col,
            "has_quant_col": has_quant_col,
        }
        points = []
        for row_index, (_, row) in enumerate(df.iterrows()):
            point = self._candidate_from_attention_row(
                row,
                kernel_type,
                latency_col,
                row_index,
                avg_seq_col,
                has_sparse_col=has_sparse_col,
                has_kv_heads_col=has_kv_heads_col,
                has_layout_col=has_layout_col,
                has_quant_col=has_quant_col,
            )
            if point is not None:
                points.append(point)
        self._attention_index_diagnostics[kernel_type]["usable_points"] = len(points)
        index = CandidateIndex(points)
        self._attention_index_cache[cache_key] = index
        return index

    def _build_attention_target(
        self,
        op_invoke_info: "OpInvokeInfo",
        mapping: dict,
        kernel_type: str,
        *,
        include_batch_axis: bool = False,
    ) -> Optional[InterpolationTarget]:
        args = op_invoke_info.args
        if len(args) < 7:
            return None

        query = args[0]
        key = args[1]
        seq_lens = args[6]
        query_lens = args[7] if len(args) > 7 else None
        if not isinstance(query, torch.Tensor) or not isinstance(seq_lens, torch.Tensor):
            return None

        head_dim = key.shape[-1] if isinstance(key, torch.Tensor) and key.ndim >= 1 else 0
        q_3d = _normalize_fia_q_shape(tuple(query.shape), head_dim)
        if q_3d is None:
            return None
        try:
            avg_seq_len = int(seq_lens.float().mean().item())
        except Exception:
            return None

        dtype_str = DTYPE_MAP.get(query.dtype)
        if dtype_str is None:
            return None

        axes = {
            "q_tokens": float(q_3d[0]),
            "seq": float(avg_seq_len),
            "heads": float(q_3d[1]),
            "head_dim": float(q_3d[2]),
        }
        batch_axis = None
        if query.ndim == 4:
            batch_axis = float(query.shape[0])
        elif seq_lens.ndim > 0:
            batch_axis = float(seq_lens.numel())
        if include_batch_axis and batch_axis is not None:
            axes["batch"] = batch_axis

        input_layout = _infer_attention_input_layout(tuple(query.shape), head_dim)
        sparse_mode = _infer_attention_sparse_mode(query_lens, input_layout)
        if sparse_mode is None:
            self._record_miss(
                "attention_sparse_mode_unknown",
                kernel_type=kernel_type,
                query_shape=tuple(query.shape),
                input_layout=input_layout,
            )
            return None
        kv_heads = _attention_kv_heads_from_key(key, input_layout)
        quant_mode = _explicit_attention_quant_mode(
            op_invoke_info,
            mapping,
            self._kernel_overrides.get(kernel_type, {}).get("quant_mode"),
        )

        key_fields: list[tuple[str, Any]] = [
            ("kernel_type", kernel_type),
            ("dtype", dtype_str),
            ("sparse_mode", sparse_mode),
            ("kv_heads", int(kv_heads) if kv_heads is not None else _UNKNOWN_KV_HEADS),
        ]
        if input_layout is not None:
            key_fields.append(("input_layout", input_layout))
        if quant_mode is not None:
            key_fields.append(("quant_mode", quant_mode))

        return InterpolationTarget(
            func_name=_normalize_func_name(op_invoke_info.func),
            kernel_type=kernel_type,
            axes=axes,
            regime_key=make_regime_key(key_fields),
            tc_shapes=[tuple(arg.shape) for arg in args if isinstance(arg, torch.Tensor)],
            input_dtypes=[dtype_str],
            query_mode="attention_special",
            metadata={"batch_axis": batch_axis} if batch_axis is not None else {},
        )

    def _build_attention_target_from_params(
        self,
        kernel_type: str,
        params: Dict[str, Any],
        dtype_str: str,
        *,
        func_name: Optional[str] = None,
    ) -> Optional[InterpolationTarget]:
        q_shape_3d = params.get("q_shape_3d")
        avg_seq_len = params.get("avg_seq_len")
        if q_shape_3d is None or avg_seq_len is None:
            return None
        q_shape_3d = tuple(q_shape_3d)
        if len(q_shape_3d) < 3:
            return None

        axes = {
            "q_tokens": float(q_shape_3d[0]),
            "seq": float(avg_seq_len),
            "heads": float(q_shape_3d[1]),
            "head_dim": float(q_shape_3d[2]),
        }
        if params.get("batch_size") is not None:
            axes["batch"] = float(params["batch_size"])

        key_fields: list[tuple[str, Any]] = [
            ("kernel_type", kernel_type),
            ("dtype", dtype_str),
        ]
        if params.get("sparse_mode") is not None:
            key_fields.append(("sparse_mode", int(params["sparse_mode"])))
        if params.get("num_kv_heads") is not None:
            key_fields.append(("kv_heads", int(params["num_kv_heads"])))
        input_layout = self._attention_input_layout_from_params(params)
        if input_layout is not None:
            key_fields.append(("input_layout", input_layout))
        if params.get("quant_mode") is not None:
            key_fields.append(("quant_mode", str(params["quant_mode"])))

        return InterpolationTarget(
            func_name=func_name or kernel_type,
            kernel_type=kernel_type,
            axes=axes,
            regime_key=make_regime_key(key_fields),
            tc_shapes=[q_shape_3d],
            input_dtypes=[dtype_str],
            query_mode="attention_special",
            metadata={"batch_axis": float(params["batch_size"])} if params.get("batch_size") is not None else {},
        )

    def _attention_quant_unknown_details(self, kernel_type: str, target: InterpolationTarget) -> dict[str, Any]:
        target_quant_mode = dict(target.regime_key).get("quant_mode")
        if target_quant_mode is None:
            return {}
        df = self.base._load_csv(kernel_type)
        if df is not None and "Runtime quant_mode" not in df.columns:
            return {
                "quant_mode_unknown_in_csv": True,
                "target_quant_mode": target_quant_mode,
            }
        return {}

    def _attention_csv_has_quant_column(self, kernel_type: str) -> bool:
        df = self.base._load_csv(kernel_type)
        return df is not None and "Runtime quant_mode" in df.columns

    def _attention_matching_fields(self, kernel_type: str, target: InterpolationTarget) -> tuple[set[str], set[str]]:
        target_fields = dict(target.regime_key)
        required_fields = {"sparse_mode", "kv_heads"}
        allow_extra_fields: set[str] = set()
        if "input_layout" in target_fields:
            required_fields.add("input_layout")
        if "quant_mode" in target_fields and self._attention_csv_has_quant_column(kernel_type):
            required_fields.add("quant_mode")
        return required_fields, allow_extra_fields

    @staticmethod
    def _attention_target_batch_axis(target: InterpolationTarget) -> Optional[float]:
        batch_axis = target.axes.get("batch")
        if batch_axis is not None:
            return float(batch_axis)
        metadata_batch = target.metadata.get("batch_axis")
        if metadata_batch is not None:
            return float(metadata_batch)
        return None

    @staticmethod
    def _attention_input_layout_from_params(params: Dict[str, Any]) -> Optional[str]:
        explicit_layout = params.get("input_layout")
        if explicit_layout is not None:
            return str(explicit_layout)
        sparse_mode = params.get("sparse_mode")
        if sparse_mode is None:
            return None
        try:
            sparse_mode_int = int(sparse_mode)
        except (TypeError, ValueError):
            return None
        if sparse_mode_int == 0:
            return "BNSD_NBSD"
        if sparse_mode_int == 3:
            return "TND"
        return None

    @staticmethod
    def _sqrt_seq_group(candidate_group: CandidateGroup) -> CandidateGroup:
        transformed_points = []
        for point in candidate_group.points:
            if "seq" not in point.axes or point.axes["seq"] < 0:
                continue
            axes = dict(point.axes)
            row_meta = dict(point.row_meta)
            row_meta["pre_transform_axes"] = dict(point.axes)
            axes["seq"] = math.sqrt(axes["seq"])
            transformed_points.append(replace(point, axes=axes, row_meta=row_meta))
        return CandidateGroup(candidate_group.regime_key, transformed_points)

    def _interpolate_attention_target(
        self,
        target: InterpolationTarget,
        *,
        fallback_from: str,
        interpolation_path: str,
    ) -> Optional[QueryResult]:
        index = self._get_attention_index(target.kernel_type)
        if index is None:
            self._record_miss(
                "attention_index_unavailable",
                kernel_type=target.kernel_type,
                interpolation_path=interpolation_path,
            )
            return None

        override = self._kernel_overrides.get(target.kernel_type, {})
        transform_config = override.get("axis_transform", override.get("shape_transform"))
        use_sqrt = transform_config in {"sqrt", "sqrt_seq"}
        axis_transform = None
        if use_sqrt:
            axis_transform = "sqrt(seq)"

        extra_details = {
            "kernel_type": target.kernel_type,
            "query_mode": target.query_mode,
            "interpolation_path": interpolation_path,
            "attention_axes": dict(target.axes),
            "target_regime_key": dict(target.regime_key),
            **self._attention_quant_unknown_details(target.kernel_type, target),
        }
        target_fields = dict(target.regime_key)
        index_diagnostics = self._attention_index_diagnostics.get(target.kernel_type, {})
        if index_diagnostics.get("has_layout_col") and "input_layout" not in target_fields:
            self._record_miss(
                "attention_input_layout_unavailable",
                kernel_type=target.kernel_type,
                interpolation_path=interpolation_path,
                target_axes=target.axes,
                target_regime_key=target_fields,
                index_diagnostics=index_diagnostics,
            )
            return None
        required_fields, allow_extra_fields = self._attention_matching_fields(target.kernel_type, target)
        candidate_groups = index.candidate_groups_matching(
            target.regime_key,
            required_target_fields=required_fields,
            allow_extra_fields=allow_extra_fields,
        )
        if not candidate_groups:
            self._record_miss(
                "regime_key_unmatched",
                kernel_type=target.kernel_type,
                interpolation_path=interpolation_path,
                target_axes=target.axes,
                target_regime_key=dict(target.regime_key),
            )
            return None

        attempts: list[dict[str, Any]] = []
        target_batch_axis = self._attention_target_batch_axis(target)
        for candidate_group in candidate_groups:
            target_axes = dict(target.axes)
            batch_axis_status: Optional[str] = None
            batch_axis_details: dict[str, Any] = {}
            active_candidate_group = candidate_group
            if any("batch" in point.axes for point in candidate_group.points) and "batch" not in target_axes:
                if target_batch_axis is not None:
                    target_axes["batch"] = target_batch_axis
                else:
                    batchless_points = [point for point in candidate_group.points if "batch" not in point.axes]
                    if batchless_points:
                        batch_axis_status = "batch_axis_filtered"
                        batch_axis_details = {
                            "dropped_batched_candidates": len(candidate_group.points) - len(batchless_points)
                        }
                        active_candidate_group = CandidateGroup(candidate_group.regime_key, batchless_points)
                    else:
                        batch_values = {float(point.axes["batch"]) for point in candidate_group.points}
                        if len(batch_values) == 1:
                            batch_axis_status = "batch_axis_constant"
                            batch_value = next(iter(batch_values))
                            target_axes["batch"] = batch_value
                            batch_axis_details = {"batch": batch_value}
                        else:
                            attempts.append(
                                {
                                    "regime_key": dict(candidate_group.regime_key),
                                    "diagnostics": {
                                        "attempts": [
                                            {
                                                "status": "batch_axis_unconstrained",
                                                "axes": ["batch"],
                                                "batch_values": sorted(batch_values),
                                            }
                                        ]
                                    },
                                }
                            )
                            continue
            axes_pre_transform = dict(target_axes)
            if use_sqrt:
                target_axes["seq"] = math.sqrt(target_axes["seq"])
            active_group = self._sqrt_seq_group(active_candidate_group) if use_sqrt else active_candidate_group
            interpolation_extra_details = {
                **extra_details,
                "attention_axes": axes_pre_transform,
            }
            if use_sqrt:
                interpolation_extra_details["axes_pre_transform"] = axes_pre_transform
            if batch_axis_status is not None:
                interpolation_extra_details["batch_axis_status"] = batch_axis_status
                interpolation_extra_details.update(batch_axis_details)
            result = active_group.interpolate(
                target_axes,
                _ATTENTION_AXIS_GROUPS,
                fallback_from=fallback_from,
                axis_transform=axis_transform,
                extra_details=interpolation_extra_details,
                axis_matchers={"q_tokens": self._attention_q_tokens_match},
            )
            if result is None:
                attempts.append(
                    {
                        "regime_key": dict(candidate_group.regime_key),
                        "diagnostics": active_group.last_diagnostics,
                    }
                )
                continue
            if use_sqrt:
                result = self._mark_sqrt_interpolation(result)
            logger.debug(
                "INTERPOLATED attention op=%s kernel=%s dim=%d axes=%s method=%s confidence=%.2f",
                target.func_name,
                target.kernel_type,
                result.interpolation_dim,
                ",".join(result.axes),
                result.method,
                result.confidence,
            )
            return self._query_result_from_interpolation(target, result)

        self._record_miss(
            self._candidate_failure_reason(
                "attention_candidate_group_failed", attempts[-1].get("diagnostics") if attempts else {}
            ),
            kernel_type=target.kernel_type,
            interpolation_path=interpolation_path,
            attempts=attempts,
            target_axes=target.axes,
        )
        return None

    def _interpolate_attention_multidim(
        self,
        op_invoke_info: "OpInvokeInfo",
        mapping: dict,
        *,
        fallback_from: str = "exact_miss",
    ) -> Optional[QueryResult]:
        kernel_type = mapping.get("kernel_type")
        if not kernel_type:
            return None
        kernel_types = [kernel_type]
        for alt in mapping.get("alternate_kernel_types", []):
            if alt not in kernel_types:
                kernel_types.append(alt)

        for kt in kernel_types:
            index = self._get_attention_index(kt)
            if index is None:
                self._record_miss("attention_index_unavailable", kernel_type=kt)
                continue
            target = self._build_attention_target(
                op_invoke_info,
                mapping,
                kt,
            )
            if target is None:
                self._record_miss("attention_target_unavailable", kernel_type=kt)
                continue

            result = self._interpolate_attention_target(
                target,
                fallback_from=fallback_from,
                interpolation_path="multidim",
            )
            if result is not None:
                return result
        return None

    @staticmethod
    def _mark_sqrt_interpolation(result: InterpolationResult) -> InterpolationResult:
        method = result.method
        details = dict(result.details)
        if not method.endswith("_sqrt"):
            method = f"{method}_sqrt"
            details["method"] = method
        return replace(
            result,
            method=method,
            details=details,
            shape_match_rule=f"{result.shape_match_rule}_sqrt",
        )

    # ---- Compute interpolation ----

    def _interpolate_compute(
        self,
        op_invoke_info: "OpInvokeInfo",
        mapping: dict,
        *,
        fallback_from: str = "exact_miss",
    ) -> Optional[QueryResult]:
        kernel_type = mapping.get("kernel_type")
        if not kernel_type:
            self._record_miss("compute_kernel_type_missing")
            return None
        if kernel_type in _INTERPOLATION_MATMUL_KERNELS:
            return self._interpolate_compute_multidim(op_invoke_info, mapping, fallback_from=fallback_from)

        kernel_types = [kernel_type]
        for alt in mapping.get("alternate_kernel_types", []):
            if alt not in kernel_types:
                kernel_types.append(alt)
        for kt in kernel_types:
            target = self._build_generic_compute_target(
                op_invoke_info,
                mapping,
                kt,
                policy_kernel_type=kernel_type,
            )
            if target is None:
                self._record_miss("compute_target_unavailable", kernel_type=kt)
                continue
            result = self._interpolate_generic_compute_target(
                target,
                mapping.get("tc_input_count"),
                fallback_from=fallback_from,
                interpolation_path="compute_1d",
                policy_kernel_type=kernel_type,
            )
            if result is not None:
                return result
        return None

    # ---- Communication interpolation ----

    # Communication interpolation is handled by ProfilingDataSource._query_comm_csv
    # which has built-in alpha-beta least-squares interpolation. If base.lookup()
    # returns None for a comm op, there's no data to interpolate against.

    # ---- Attention interpolation ----

    def _interpolate_attention(
        self,
        op_invoke_info: "OpInvokeInfo",
        mapping: dict,
        *,
        fallback_from: str = "exact_miss",
    ) -> Optional[QueryResult]:
        return self._interpolate_attention_multidim(op_invoke_info, mapping, fallback_from=fallback_from)

    # ---- Composite interpolation ----

    def _interpolate_composite(
        self, op_invoke_info: "OpInvokeInfo", mapping: dict, func_str: str
    ) -> Optional[QueryResult]:
        """Interpolate composite ops by decomposing into sub-kernels.

        Uses registered decomposers to get sub-kernel specs, then interpolates
        each sub-kernel individually and sums the results.
        """
        decomposer = COMPOSITE_DECOMPOSERS.get(func_str)
        if decomposer is None:
            return None

        specs = decomposer(op_invoke_info, mapping)
        if not specs:
            return None

        total_latency = 0.0
        hit_kernels = []
        sub_kernel_details = []

        for spec_index, spec in enumerate(specs):
            lat = None
            sub_detail: dict[str, Any] = {
                "kernel_type": spec.kernel_type,
                "query_mode": spec.query_mode,
                "fallback_from": "composite",
            }

            # First try exact match via base ProfilingDataSource
            kernel_types = [spec.kernel_type] + (spec.alternate_kernel_types or [])
            sub_detail["candidate_kernel_types"] = kernel_types
            matched_kernel_type = spec.kernel_type
            if spec.query_mode == "attention" and spec.attention_params:
                result_exact = self.base._query_by_attn_params(kernel_types, spec.attention_params, spec.dtype)
                lat = result_exact[0] if result_exact else None
                if lat is not None:
                    matched_kernel_type = result_exact[1]
                    sub_detail.update(
                        {
                            "source": QuerySource.MEASURED.name,
                            "method": "exact_attention_params",
                            "matched_kernel_type": matched_kernel_type,
                        }
                    )
            else:
                torch_dtype = None
                for k, v in DTYPE_MAP.items():
                    if v == spec.dtype:
                        torch_dtype = k
                        break
                if torch_dtype is not None:
                    tc_inputs = [(shape, torch_dtype) for shape in spec.input_shapes]
                    hit = self.base._find_compute_match(kernel_types, tc_inputs, spec.tc_input_count)
                    lat = hit.latency_us if hit else None
                    if hit is not None:
                        matched_kernel_type = hit.kernel_type
                        sub_detail.update(
                            {
                                "source": QuerySource.MEASURED.name,
                                "method": "exact_compute_match",
                                "matched_kernel_type": matched_kernel_type,
                            }
                        )
                else:
                    lat = None

            # If exact miss, try interpolation
            if lat is None:
                result_interp = None
                if spec.query_mode == "attention" and spec.attention_params:
                    result_interp = self._interpolate_attention_by_params(
                        kernel_types, spec.attention_params, spec.dtype
                    )
                else:
                    result_interp = self._interpolate_compute_by_shapes(
                        kernel_types,
                        spec.input_shapes,
                        spec.dtype,
                        spec.tc_input_count,
                    )
                if result_interp is not None:
                    lat = result_interp.latency_us
                    sub_detail.update(
                        {
                            "source": result_interp.source.name,
                            "method": result_interp.details.get("method"),
                            "matched_kernel_type": result_interp.details.get("kernel_type", spec.kernel_type),
                            "axes": result_interp.details.get("axes"),
                            "candidate_count": result_interp.details.get("candidate_count"),
                            "fallback_from": result_interp.details.get("fallback_from", "composite"),
                            "details": result_interp.details,
                        }
                    )

            if lat is None:
                self._record_miss(
                    "composite_sub_kernel_failed",
                    kernel_type=spec.kernel_type,
                    query_mode=spec.query_mode,
                    fallback_from="composite",
                    completed_sub_kernels=sub_kernel_details,
                    failed_sub_kernel_index=spec_index,
                    sub_kernel_count=len(specs),
                    completed_latency_us=total_latency,
                    failed_sub_kernel={
                        "kernel_type": spec.kernel_type,
                        "query_mode": spec.query_mode,
                        "alternate_kernel_types": spec.alternate_kernel_types or [],
                    },
                    sub_kernel_miss_reason=self.last_miss_reason,
                    sub_kernel_miss_details=self.last_miss_details,
                )
                return None

            total_latency += lat
            hit_kernels.append(sub_detail.get("matched_kernel_type", matched_kernel_type))
            sub_detail["latency_us"] = lat
            sub_kernel_details.append(sub_detail)

        logger.debug(
            "INTERPOLATED (composite) %s: sub_kernels=%s, total=%.1f us",
            func_str,
            hit_kernels,
            total_latency,
        )
        all_sub_kernels_measured = all(
            detail.get("source") == QuerySource.MEASURED.name for detail in sub_kernel_details
        )
        source = QuerySource.MEASURED if all_sub_kernels_measured else QuerySource.INTERPOLATED
        shape_match_rule = "composite_measured" if all_sub_kernels_measured else "interpolated_composite"
        return QueryResult(
            latency_us=total_latency,
            confidence=0.5,
            source=source,
            details={
                "kernel_type": ",".join(hit_kernels),
                "composite": True,
                "method": "decomposed_interpolation",
                "sub_kernels": sub_kernel_details,
            },
            shape_match_info=ShapeMatchInfo(
                simulation_shapes=[],
                kernel_shapes=[],
                shape_match_rule=shape_match_rule,
            ),
        )

    def _interpolate_compute_by_shapes(
        self,
        kernel_type: str | list[str],
        input_shapes: List[Tuple[int, ...]],
        dtype_str: str,
        tc_input_count: Optional[int] = None,
    ) -> Optional[QueryResult]:
        """Interpolate a compute sub-kernel by explicit shapes.

        Same logic as _interpolate_compute but takes shapes directly
        instead of extracting from OpInvokeInfo.
        """
        if not input_shapes:
            return None

        kernel_types = [kernel_type] if isinstance(kernel_type, str) else list(kernel_type)
        policy_kernel_type = kernel_types[0] if kernel_types else None
        for kt in kernel_types:
            result = self._interpolate_compute_by_shapes_one(
                kt,
                input_shapes,
                dtype_str,
                tc_input_count,
                policy_kernel_type=policy_kernel_type,
            )
            if result is not None:
                return result
        return None

    def _interpolate_compute_by_shapes_one(
        self,
        kernel_type: str,
        input_shapes: List[Tuple[int, ...]],
        dtype_str: str,
        tc_input_count: Optional[int] = None,
        policy_kernel_type: Optional[str] = None,
    ) -> Optional[QueryResult]:
        effective_tc_input_count = tc_input_count if tc_input_count is not None else len(input_shapes)
        if kernel_type in _INTERPOLATION_MATMUL_KERNELS:
            target = self._build_compute_target_from_shapes(
                kernel_type,
                input_shapes,
                dtype_str,
                tc_input_count=effective_tc_input_count,
                query_mode="compute",
            )
            if target is None:
                self._record_miss(
                    "compute_target_unavailable",
                    kernel_type=kernel_type,
                    interpolation_path="composite_compute",
                )
                return None
            return self._interpolate_compute_target(
                target,
                effective_tc_input_count,
                fallback_from="composite",
                interpolation_path="composite_compute",
            )

        target, reason = self._build_generic_compute_target_from_shapes_with_reason(
            kernel_type,
            input_shapes,
            dtype_str,
            tc_input_count=effective_tc_input_count,
            query_mode="compute",
            policy_kernel_type=policy_kernel_type,
        )
        if target is None:
            self._record_miss(
                reason or "compute_target_unavailable",
                kernel_type=kernel_type,
                interpolation_path="composite_compute_1d",
            )
            return None
        return self._interpolate_generic_compute_target(
            target,
            effective_tc_input_count,
            fallback_from="composite",
            interpolation_path="composite_compute_1d",
            policy_kernel_type=policy_kernel_type,
        )

    def _interpolate_attention_by_params(
        self,
        kernel_type: str | list[str],
        params: Dict,
        dtype_str: str,
    ) -> Optional[QueryResult]:
        """Interpolate attention sub-kernel using enriched CSV by explicit params.

        params: {q_shape_3d, avg_seq_len, sparse_mode->, num_kv_heads->}
        """
        kernel_types = [kernel_type] if isinstance(kernel_type, str) else list(kernel_type)
        for kt in kernel_types:
            result = self._interpolate_attention_by_params_one(kt, params, dtype_str)
            if result is not None:
                return result
        return None

    def _interpolate_attention_by_params_one(
        self,
        kernel_type: str,
        params: Dict,
        dtype_str: str,
    ) -> Optional[QueryResult]:
        index = self._get_attention_index(kernel_type)
        if index is None:
            self._record_miss(
                "attention_index_unavailable",
                kernel_type=kernel_type,
                interpolation_path="composite_attention",
            )
            return None
        target = self._build_attention_target_from_params(kernel_type, params, dtype_str)
        if target is None:
            self._record_miss(
                "attention_target_unavailable",
                kernel_type=kernel_type,
                interpolation_path="composite_attention",
            )
            return None
        return self._interpolate_attention_target(
            target,
            fallback_from="composite",
            interpolation_path="composite_attention",
        )

    # ---- Elementwise interpolation ----

    @staticmethod
    def _elementwise_input_role(input_shape: tuple[int, ...], output_shape: tuple[int, ...]) -> str:
        if not input_shape:
            return "scalar"
        if input_shape == output_shape:
            return "full"
        if len(input_shape) < len(output_shape) and input_shape == output_shape[-len(input_shape) :]:
            return "broadcast"
        if all(dim == 1 for dim in input_shape):
            return "scalar"
        return "unknown"

    @classmethod
    def _elementwise_input_signature(
        cls,
        input_shapes: list[tuple[int, ...]],
        output_shape: tuple[int, ...],
    ) -> Optional[tuple[tuple[str, tuple[int, ...]], ...]]:
        if not input_shapes:
            return None
        signature = []
        for shape in input_shapes:
            logical_shape = tuple(_strip_batch_dim(tuple(shape)))
            role = cls._elementwise_input_role(logical_shape, output_shape)
            if role == "full":
                signature_shape = logical_shape[1:]
            elif role == "scalar":
                signature_shape = ()
            elif role == "unknown" and logical_shape and output_shape and logical_shape[0] == output_shape[0]:
                signature_shape = logical_shape[1:]
            else:
                signature_shape = logical_shape
            signature.append((role, signature_shape))
        return tuple(signature)

    def _candidate_from_elementwise_row(
        self,
        row: Any,
        kernel_type: str,
        latency_col: str,
        row_index: int,
        tc_dtype_str: Optional[str],
    ) -> Optional[CandidatePoint]:
        csv_out_shapes = _parse_shape_str(str(row.get("Output Shapes", "")))
        csv_out_dtypes = _parse_str_list(str(row.get("Output Data Types", "")))
        if not csv_out_shapes:
            return None

        csv_shape = _strip_batch_dim(tuple(csv_out_shapes[0]))
        if not csv_shape:
            return None
        csv_input_shapes = [tuple(shape) for shape in _parse_shape_str(str(row.get("Input Shapes", "")))]
        input_signature = self._elementwise_input_signature(csv_input_shapes, csv_shape)
        if input_signature is None:
            return None

        latency, latency_meta = self._candidate_latency(row, latency_col)
        if latency is None:
            return None

        csv_dtype_str = csv_out_dtypes[0] if csv_out_dtypes else None
        scale = 1.0
        if csv_dtype_str and tc_dtype_str and csv_dtype_str != tc_dtype_str:
            tc_bytes = _dtype_byte_size(tc_dtype_str)
            csv_bytes = _dtype_byte_size(csv_dtype_str)
            if tc_bytes > 0 and csv_bytes > 0:
                scale = tc_bytes / csv_bytes
                latency *= scale
        if not validate_positive_latency(latency):
            return None

        regime_key = make_regime_key(
            [
                ("kernel_type", kernel_type),
                ("query_mode", "elementwise"),
                ("output_shape_tail", tuple(csv_shape[1:])),
                ("input_signature", input_signature),
                ("csv_output_dtype", csv_dtype_str or ""),
            ]
        )
        return CandidatePoint(
            kernel_type=kernel_type,
            axes={"axis_0": float(csv_shape[0])},
            latency_us=latency,
            regime_key=regime_key,
            input_shapes=csv_input_shapes,
            input_dtypes=[csv_dtype_str] if csv_dtype_str else [],
            row_index=row_index,
            row_meta={
                **latency_meta,
                "dtype_scale": scale,
                "csv_output_dtype": csv_dtype_str,
                "input_signature": input_signature,
            },
        )

    def _get_elementwise_index(self, kernel_type: str, tc_dtype_str: Optional[str]) -> Optional[CandidateIndex]:
        df = self.base._load_csv(kernel_type)
        if df is None:
            return None
        cache_key = (
            "elementwise",
            kernel_type,
            tc_dtype_str or "",
            self._dataframe_fingerprint(df),
            self._policy_hash,
        )
        if cache_key in self._elementwise_index_cache:
            return self._elementwise_index_cache[cache_key]

        latency_col = self.base._latency_col(df)
        points = []
        for row_index, (_, row) in enumerate(df.iterrows()):
            point = self._candidate_from_elementwise_row(row, kernel_type, latency_col, row_index, tc_dtype_str)
            if point is not None:
                points.append(point)
        index = CandidateIndex(points)
        self._elementwise_index_cache[cache_key] = index
        return index

    @staticmethod
    def _elementwise_candidate_group_attempts(
        index: CandidateIndex,
        kernel_type: str,
        output_shape_tail: tuple[int, ...],
        input_signature: tuple[tuple[str, tuple[int, ...]], ...],
        tc_dtype_str: Optional[str],
    ) -> list[tuple[str, CandidateGroup]]:
        base_fields = [
            ("kernel_type", kernel_type),
            ("query_mode", "elementwise"),
            ("output_shape_tail", output_shape_tail),
            ("input_signature", input_signature),
        ]
        attempts: list[tuple[str, CandidateGroup]] = []
        if tc_dtype_str:
            same_dtype_key = make_regime_key([*base_fields, ("csv_output_dtype", tc_dtype_str)])
            attempts.extend(("same_dtype", group) for group in index.candidate_groups_matching(same_dtype_key))

        fallback_key = make_regime_key(base_fields)
        fallback_groups = []
        target_bytes = _dtype_byte_size(tc_dtype_str) if tc_dtype_str else 0
        for group in index.candidate_groups_matching(fallback_key, allow_extra_fields={"csv_output_dtype"}):
            csv_dtype = dict(group.regime_key).get("csv_output_dtype")
            if tc_dtype_str and csv_dtype == tc_dtype_str:
                continue
            csv_bytes = _dtype_byte_size(csv_dtype) if csv_dtype else 0
            byte_distance = abs(target_bytes - csv_bytes) if target_bytes > 0 and csv_bytes > 0 else 999
            fallback_groups.append((byte_distance, group))
        for _byte_distance, group in sorted(fallback_groups, key=lambda item: (-len(item[1].points), item[0])):
            attempts.append(("scaled_dtype", group))
        return attempts

    def _interpolate_elementwise(
        self,
        op_invoke_info: "OpInvokeInfo",
        mapping: dict,
        *,
        fallback_from: str = "exact_miss",
    ) -> Optional[QueryResult]:
        """Interpolate elementwise ops on first dim of output shape, dtype-relaxed.

        Groups CSV rows by output_shape[1:] (hidden dims must match exactly).
        Collects (output_shape[0], latency_scaled) candidates and interpolates
        on the first dim (num_tokens). Byte-ratio scaling applied per-candidate
        before interpolation.
        """
        kernel_type = mapping.get("kernel_type")
        if not kernel_type:
            return None

        df = self.base._load_csv(kernel_type)
        if df is None:
            return None

        # NOTE: OpInvokeInfo uses .out (not .output); aten ops may return tuple.
        out = op_invoke_info.out
        if isinstance(out, (list, tuple)):
            out = out[0] if out else None
        if out is None or not isinstance(out, torch.Tensor) or out.ndim == 0:
            return None

        output_shape = _strip_batch_dim(tuple(out.shape))
        if len(output_shape) < 1:
            return None
        target_dim = float(output_shape[0])
        tc_dtype_str = DTYPE_MAP.get(out.dtype)
        input_shapes = [
            tuple(arg.shape) for arg in getattr(op_invoke_info, "args", ()) if isinstance(arg, torch.Tensor)
        ]
        input_signature = self._elementwise_input_signature(input_shapes, output_shape)
        if input_signature is None:
            self._record_miss(
                "elementwise_input_signature_unavailable",
                kernel_type=kernel_type,
                interpolation_path="elementwise_1d",
            )
            return None

        target = InterpolationTarget(
            func_name=_normalize_func_name(op_invoke_info.func),
            kernel_type=kernel_type,
            axes={"axis_0": target_dim},
            regime_key=make_regime_key(
                [
                    ("kernel_type", kernel_type),
                    ("query_mode", "elementwise"),
                    ("output_shape_tail", tuple(output_shape[1:])),
                    ("input_signature", input_signature),
                ]
            ),
            tc_shapes=[tuple(output_shape)],
            input_dtypes=[tc_dtype_str] if tc_dtype_str else [],
            query_mode="elementwise",
        )
        index = self._get_elementwise_index(kernel_type, tc_dtype_str)
        if index is None:
            self._record_miss(
                "elementwise_csv_not_found",
                kernel_type=kernel_type,
                interpolation_path="elementwise_1d",
            )
            return None

        candidate_attempts = self._elementwise_candidate_group_attempts(
            index,
            kernel_type,
            tuple(output_shape[1:]),
            input_signature,
            tc_dtype_str,
        )
        candidate_count = sum(len(group.points) for _label, group in candidate_attempts)
        if (
            not candidate_attempts
            or max(
                (len({point.axes["axis_0"] for point in group.points}) for _label, group in candidate_attempts),
                default=0,
            )
            < 2
        ):
            self._record_miss(
                "insufficient_filtered_candidates",
                kernel_type=kernel_type,
                interpolation_path="elementwise_1d",
                target=float(target_dim),
                candidate_count=candidate_count,
            )
            return None

        attempts: list[dict[str, Any]] = []
        for dtype_attempt, candidate_group in candidate_attempts:
            result = candidate_group.interpolate(
                target.axes,
                _GENERIC_COMPUTE_AXIS_GROUPS,
                fallback_from=fallback_from,
                extra_details={
                    "kernel_type": kernel_type,
                    "query_mode": "elementwise",
                    "interpolation_path": "elementwise_1d",
                    "dtype_attempt": dtype_attempt,
                    "csv_output_dtype": dict(candidate_group.regime_key).get("csv_output_dtype"),
                },
            )
            if result is None:
                attempts.append(
                    {
                        "regime_key": dict(candidate_group.regime_key),
                        "dtype_attempt": dtype_attempt,
                        "diagnostics": candidate_group.last_diagnostics,
                    }
                )
                continue
            selected_scales = [float(point.row_meta.get("dtype_scale", 1.0)) for point in result.matched_points]
            has_dtype_scaling = any(scale != 1.0 for scale in selected_scales)
            details = dict(result.details)
            details["dtype_scaled"] = has_dtype_scaling
            details["dtype_attempt"] = dtype_attempt
            if has_dtype_scaling:
                details["dtype_scales"] = sorted(set(selected_scales))
            result = replace(result, details=details)
            return self._query_result_from_interpolation(target, result)

        self._record_miss(
            self._candidate_failure_reason(
                "candidate_group_failed", attempts[-1].get("diagnostics") if attempts else {}
            ),
            kernel_type=kernel_type,
            interpolation_path="elementwise_1d",
            target=float(target_dim),
            candidate_count=candidate_count,
            attempts=attempts,
        )
        return None
