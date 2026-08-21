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
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

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
from .profiling_data_source import (
    _DTYPE_COMPAT,
    _DTYPE_RELAXED_KERNELS,
    _is_block_padded,
    _MATMUL_KERNELS,
    _normalize_fia_q_shape,
    _normalize_func_name,
    _parse_shape_str,
    _parse_str_list,
    _project_dispatch_ffn_combine_inputs,
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
    if not numeric.is_integer():
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
_COMPUTE_SCALE_AXIS_GROUPS = (("M",), ("K",), ("M", "K"))
_COMPUTE_SCALE_SUBCATEGORY = "compute_scale"
_QUANTIZED_MATMUL_SUBCATEGORY = "quantized_matmul"
_GENERIC_COMPUTE_AXIS_GROUPS = (("axis_0",),)
_GENERIC_COMPUTE_AXIS_0 = "axis_0"
_GENERIC_COMPUTE_OUTPUT_NUMEL_AXIS = "output_numel"
_LATENCY_COLUMN_PREFERRED = "preferred_latency_column"
_LATENCY_COLUMN_ALTERNATE = "alternate_latency_column"

_ATTENTION_AXES = (
    "seq",
    "batch",
    "heads",
    "head_dim",
)
_ATTENTION_AXIS_GROUPS = tuple(axes for dim in range(1, 4) for axes in combinations(_ATTENTION_AXES, dim))
_MOE_FUSED_AXIS_GROUPS = (("tokens",),)
_ELEMENTWISE_AXIS_GROUPS = (("io_numel",),)
_LIGHTNING_INDEXER_AXIS_GROUPS = (
    ("q_tokens",),
    ("effective_kv_len",),
    ("q_tokens", "effective_kv_len"),
)
_SPARSE_ATTENTION_AXIS_GROUPS = _LIGHTNING_INDEXER_AXIS_GROUPS
_SCATTER_ND_UPDATE_AXIS_GROUPS = (("tokens",),)
_RUNTIME_ATTENTION_KERNELS = frozenset({"LightningIndexer", "LightningIndexerVllm"})
_SPARSE_RUNTIME_ATTENTION_KERNELS = frozenset({"SparseFlashAttention"})
_MOE_FUSED_ROUTE_INPUT_INDEX = 3
_MOE_FUSED_ROUTE_DTYPE = "INT32"


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
        self._moe_fused_index_cache: Dict[tuple[Any, ...], tuple[CandidateIndex, dict[str, int]]] = {}
        self._elementwise_index_cache: Dict[tuple[Any, ...], CandidateIndex] = {}
        self._lightning_indexer_index_cache: Dict[tuple[Any, ...], tuple[CandidateIndex, dict[str, int]]] = {}
        self._sparse_attention_index_cache: Dict[tuple[Any, ...], tuple[CandidateIndex, dict[str, int]]] = {}
        self._scatter_nd_update_index_cache: Dict[tuple[Any, ...], tuple[CandidateIndex, dict[str, int]]] = {}
        self._dataframe_fingerprint_cache: Dict[int, tuple[weakref.ReferenceType, str]] = {}
        self._compute_index_diagnostics: Dict[Any, dict[str, Any]] = {}
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
            return self._interpolate_moe_fused(op_invoke_info, mapping, fallback_from=fallback_from)
        if mapping.get("query_mode") == "attention_special":
            return self._interpolate_attention(op_invoke_info, mapping, fallback_from=fallback_from)
        if mapping.get("query_mode") == "attention_lightning_indexer":
            self._record_miss(
                "runtime_attention_leaf_required",
                kernel_type=mapping.get("kernel_type"),
                query_mode="attention_lightning_indexer",
            )
            return None
        if mapping.get("query_mode") == "scatter_nd_update_mla":
            return self._interpolate_scatter_nd_update(op_invoke_info, mapping, fallback_from=fallback_from)
        if mapping.get("query_mode") == "elementwise":
            return self._interpolate_elementwise(op_invoke_info, mapping, fallback_from=fallback_from)
        # query_mode values such as mtp_projection are owned by the base exact
        # path; compute_subcategory explicitly selects the wrapper fallback.
        compute_subcategory = self._compute_subcategory(mapping)
        if compute_subcategory == _COMPUTE_SCALE_SUBCATEGORY:
            return self._interpolate_compute_scale(op_invoke_info, mapping, fallback_from=fallback_from)
        if compute_subcategory == _QUANTIZED_MATMUL_SUBCATEGORY:
            return self._interpolate_compute_multidim(op_invoke_info, mapping, fallback_from=fallback_from)
        if compute_subcategory is not None:
            self._record_miss(
                "compute_subcategory_unknown",
                op_name=func_str,
                compute_subcategory=compute_subcategory,
            )
            return None
        return self._interpolate_compute(op_invoke_info, mapping, fallback_from=fallback_from)

    # ---- Multidimensional index helpers ----

    @staticmethod
    def _dtype_key(kernel_type: str, dtype_str: str) -> str:
        if kernel_type in _DTYPE_RELAXED_KERNELS:
            return _DTYPE_COMPAT.get(dtype_str, dtype_str)
        return dtype_str

    @staticmethod
    def _compute_subcategory(mapping: dict) -> Optional[str]:
        value = mapping.get("compute_subcategory")
        return str(value) if value is not None else None

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
    def _build_candidate_index(
        df: pd.DataFrame,
        candidate_builder: Callable[
            [Any, int],
            tuple[Optional[CandidatePoint], Optional[str]],
        ],
    ) -> tuple[CandidateIndex, dict[str, int]]:
        points: list[CandidatePoint] = []
        rejected_reasons: dict[str, int] = {}
        for row_index, (_, row) in enumerate(df.iterrows()):
            point, reason = candidate_builder(row, row_index)
            if point is not None:
                points.append(point)
            elif reason:
                rejected_reasons[reason] = rejected_reasons.get(reason, 0) + 1
        return CandidateIndex(points), rejected_reasons

    @staticmethod
    def _runtime_int_values(value: Any) -> Optional[tuple[int, ...]]:
        if isinstance(value, torch.Tensor):
            if value.device.type == "meta":
                return None
            try:
                value = value.detach().cpu().reshape(-1).tolist()
            except (RuntimeError, TypeError, ValueError):
                return None
        elif isinstance(value, str):
            value = [item.strip() for item in value.split(",") if item.strip()]
        elif not isinstance(value, (list, tuple)):
            value = [value]

        result: list[int] = []
        for item in value:
            parsed = _to_int_cell(item)
            if parsed is None:
                return None
            result.append(parsed)
        return tuple(result) if result else None

    @staticmethod
    def _query_lengths_from_cumulative(
        q_tokens: int,
        cumulative_offsets: Sequence[int],
    ) -> Optional[tuple[int, ...]]:
        if q_tokens <= 0 or not cumulative_offsets:
            return None
        previous = 0
        query_lengths: list[int] = []
        for offset in cumulative_offsets:
            current = int(offset)
            if current < previous or current > q_tokens:
                return None
            query_lengths.append(current - previous)
            previous = current
        if previous != q_tokens:
            return None
        return tuple(query_lengths)

    @staticmethod
    def _attention_runtime_workload(
        *,
        q_tokens: int,
        query_lengths: Sequence[int],
        kv_lengths: Sequence[int],
    ) -> Optional[dict[str, Any]]:
        if q_tokens <= 0 or not query_lengths or len(query_lengths) != len(kv_lengths):
            return None
        query_values = tuple(int(value) for value in query_lengths)
        kv_values = tuple(int(value) for value in kv_lengths)
        if any(value < 0 for value in query_values) or any(value < 0 for value in kv_values):
            return None
        if sum(query_values) != q_tokens:
            return None
        active = [(query_len, kv_len) for query_len, kv_len in zip(query_values, kv_values) if query_len > 0]
        if not active or any(kv_len <= 0 for _, kv_len in active):
            return None
        if all(query_len == kv_len for query_len, kv_len in active):
            phase = "prefill"
        elif all(query_len == 1 for query_len, _ in active):
            phase = "decode"
        else:
            phase = "mixed"
        effective_kv_len = sum(query_len * kv_len for query_len, kv_len in active) / q_tokens
        if not math.isfinite(effective_kv_len) or effective_kv_len <= 0:
            return None
        return {
            "q_tokens": float(q_tokens),
            "effective_kv_len": float(effective_kv_len),
            "phase": phase,
            "batch_size": len(query_values),
        }

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
                    "latency_column_selection": (
                        _LATENCY_COLUMN_PREFERRED if column == latency_col else _LATENCY_COLUMN_ALTERNATE
                    ),
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
        *,
        include_output_signature: bool = False,
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
        regime_fields: list[tuple[str, Any]] = [
            ("kernel_type", kernel_type),
            ("input_count", input_count),
            ("input_dtypes", dtype_key),
            ("batch_dims", batch_dims),
            ("input_formats", tuple(csv_formats[:input_count])),
        ]
        if include_output_signature:
            output_shapes = _parse_shape_str(str(row.get("Output Shapes", "")))
            output_dtypes = _parse_str_list(str(row.get("Output Data Types", "")))
            output_formats = _parse_str_list(str(row.get("Output Formats", "")))
            if (
                not output_shapes
                or len(output_dtypes) != len(output_shapes)
                or len(output_formats) != len(output_shapes)
            ):
                return None, "output_signature_missing"
            regime_fields.extend(
                [
                    ("output_count", len(output_shapes)),
                    ("output_dtypes", tuple(output_dtypes)),
                    ("output_formats", tuple(output_formats)),
                ]
            )
        return CandidatePoint(
            kernel_type=kernel_type,
            axes=axes,
            latency_us=latency,
            regime_key=make_regime_key(regime_fields),
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
        *,
        include_output_signature: bool = False,
    ) -> Optional[CandidateIndex]:
        df = self.base._load_csv(kernel_type)
        if df is None:
            return None
        cache_key = (
            kernel_type,
            tc_input_count,
            include_output_signature,
            self._dataframe_fingerprint(df),
            self._policy_hash,
        )
        if cache_key in self._compute_index_cache:
            return self._compute_index_cache[cache_key]
        latency_col = self.base._latency_col(df)
        points = []
        rejected_reasons: dict[str, int] = {}
        for row_index, (_, row) in enumerate(df.iterrows()):
            point, reason = self._candidate_from_compute_row_with_reason(
                row,
                kernel_type,
                latency_col,
                row_index,
                tc_input_count,
                include_output_signature=include_output_signature,
            )
            if point is not None:
                points.append(point)
            elif reason:
                rejected_reasons[reason] = rejected_reasons.get(reason, 0) + 1
        self._compute_index_diagnostics[(kernel_type, include_output_signature)] = {
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
        compute_subcategory = self._compute_subcategory(mapping)
        if compute_subcategory == _QUANTIZED_MATMUL_SUBCATEGORY:
            output_tensors = self._output_tensors(getattr(op_invoke_info, "out", None))
            activation_shape = input_shapes[0]
            if output_tensors is None or len(activation_shape) != 2 or output_tensors[0].ndim != 2:
                return None
            output_shape = tuple(output_tensors[0].shape)
            if any(int(dim) <= 0 for dim in (*activation_shape, *output_shape)):
                return None
            if int(activation_shape[0]) != int(output_shape[0]):
                return None
            axes = {
                "M": float(activation_shape[0]),
                "K": float(activation_shape[1]),
                "N": float(output_shape[1]),
            }
            batch_dims = ((), ())
        else:
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
        regime_fields: list[tuple[str, Any]] = [
            ("kernel_type", kernel_type),
            ("input_count", input_count),
            ("input_dtypes", tuple(dtype_values[:input_count])),
            ("batch_dims", batch_dims),
        ]
        if compute_subcategory == _QUANTIZED_MATMUL_SUBCATEGORY:
            expected_input_formats = mapping.get("expected_input_formats")
            if (
                not isinstance(expected_input_formats, list)
                or len(expected_input_formats) != input_count
                or any(not isinstance(fmt, str) or not fmt.strip() for fmt in expected_input_formats)
            ):
                return None
            regime_fields.append(("input_formats", tuple(fmt.strip() for fmt in expected_input_formats)))
            output_dtypes = []
            for tensor in output_tensors:
                dtype_str = DTYPE_MAP.get(tensor.dtype)
                if dtype_str is None:
                    return None
                output_dtypes.append(dtype_str)
            regime_fields.extend(
                [
                    ("output_count", len(output_tensors)),
                    ("output_dtypes", tuple(output_dtypes)),
                    ("output_formats", tuple("ND" for _ in output_tensors)),
                ]
            )
        regime_key = make_regime_key(regime_fields)
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
        matched_shapes = []
        for point in result.matched_points:
            reported_axes = point.row_meta.get("pre_transform_axes", point.axes)
            matched_shapes.append([float(reported_axes[axis]) for axis in result.axes if axis in reported_axes])
        return matched_shapes

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
    def _moe_fused_activation_shape_info(shape: Tuple[int, ...]) -> Optional[tuple[float, int, Tuple[int, ...]]]:
        if len(shape) < 2:
            return None
        token_count = math.prod(int(dim) for dim in shape[:-1])
        hidden = int(shape[-1])
        return float(token_count), hidden, tuple(shape[:-1])

    @staticmethod
    def _first_output_tensor(op_invoke_info: "OpInvokeInfo") -> Optional[torch.Tensor]:
        out = getattr(op_invoke_info, "out", None)
        if isinstance(out, (list, tuple)):
            out = out[0] if out else None
        return out if isinstance(out, torch.Tensor) and out.ndim > 0 else None

    @staticmethod
    def _moe_fused_topk(
        input_shapes: List[Tuple[int, ...]],
        *,
        tokens: float,
        input_dtypes: Optional[List[str]] = None,
    ) -> Optional[int]:
        dtype_list = input_dtypes or []
        if len(input_shapes) <= _MOE_FUSED_ROUTE_INPUT_INDEX or len(dtype_list) <= _MOE_FUSED_ROUTE_INPUT_INDEX:
            return None
        route_shape = input_shapes[_MOE_FUSED_ROUTE_INPUT_INDEX]
        route_dtype = str(dtype_list[_MOE_FUSED_ROUTE_INPUT_INDEX]).upper()
        if len(route_shape) < 2 or route_dtype != _MOE_FUSED_ROUTE_DTYPE:
            return None
        topk = int(route_shape[-1])
        route_tokens = math.prod(int(dim) for dim in route_shape[:-1])
        if float(route_tokens) != float(tokens) or topk <= 0:
            return None
        return topk

    @classmethod
    def _elementwise_axes_from_shapes(
        cls,
        input_shapes: List[Tuple[int, ...]],
        output_shape: Tuple[int, ...],
    ) -> Optional[dict[str, float]]:
        input_numels = [cls._shape_numel(shape) for shape in input_shapes]
        write_numel = cls._shape_numel(output_shape)
        if any(numel is None for numel in input_numels) or write_numel is None:
            return None
        read_numel = sum(numel for numel in input_numels if numel is not None)
        return {"io_numel": float(read_numel + write_numel)}

    @staticmethod
    def _elementwise_broadcast_pattern(
        output_shape: Tuple[int, ...],
        input_shapes: Optional[List[Tuple[int, ...]]] = None,
    ) -> tuple[Any, ...]:
        if not output_shape or not input_shapes:
            return ()
        output_shape = tuple(output_shape)
        patterns = []
        for input_shape in input_shapes:
            logical_shape = tuple(input_shape)
            padded = (None,) * max(0, len(output_shape) - len(logical_shape)) + logical_shape[-len(output_shape) :]
            relation = []
            for input_dim, output_dim in zip(padded, output_shape):
                if input_dim is None:
                    relation.append("missing")
                elif int(input_dim) == int(output_dim):
                    relation.append("same")
                elif int(input_dim) == 1 and int(output_dim) != 1:
                    relation.append("broadcast")
                else:
                    relation.append("other")
            patterns.append((len(logical_shape), tuple(relation)))
        return tuple(patterns)

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
            kernel_type, logical_shapes, output_shapes, policy_kernel_type
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
        index, rejected_reasons = self._build_candidate_index(
            df,
            lambda row, row_index: self._candidate_from_generic_compute_row_with_reason(
                row,
                kernel_type,
                latency_col,
                row_index,
                tc_input_count,
                policy_kernel_type,
            ),
        )
        self._compute_index_diagnostics[kernel_type] = {
            "csv_rows": len(df),
            "usable_points": len(index.points),
            "rejected_reasons": rejected_reasons,
        }
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
            for latency_column_group, candidate_subset in self._latency_column_pure_candidate_group_attempts(
                candidate_group
            ):
                result = candidate_subset.interpolate(
                    target.axes,
                    self._generic_compute_axis_groups(target.kernel_type, policy_kernel_type),
                    fallback_from=fallback_from,
                    max_interpolation_dim=override.get("max_interpolation_dim"),
                    extra_details={
                        "kernel_type": target.kernel_type,
                        "query_mode": target.query_mode,
                        "interpolation_path": interpolation_path,
                        "latency_column_group": latency_column_group,
                    },
                )
                if result is None:
                    attempts.append(
                        {
                            "regime_key": dict(candidate_subset.regime_key),
                            "latency_column_group": latency_column_group,
                            "diagnostics": candidate_subset.last_diagnostics,
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
    def _candidate_latency_column_selection(point: CandidatePoint) -> Optional[str]:
        value = point.row_meta.get("latency_column_selection")
        if value in {_LATENCY_COLUMN_PREFERRED, _LATENCY_COLUMN_ALTERNATE}:
            return str(value)
        return None

    @classmethod
    def _latency_column_pure_candidate_group_attempts(
        cls, candidate_group: CandidateGroup
    ) -> list[tuple[str, CandidateGroup]]:
        preferred_column_points: list[CandidatePoint] = []
        alternate_column_points: list[CandidatePoint] = []
        unlabeled_points: list[CandidatePoint] = []
        for point in candidate_group.points:
            selection = cls._candidate_latency_column_selection(point)
            if selection == _LATENCY_COLUMN_PREFERRED:
                preferred_column_points.append(point)
            elif selection == _LATENCY_COLUMN_ALTERNATE:
                alternate_column_points.append(point)
            else:
                unlabeled_points.append(point)

        if not preferred_column_points and not alternate_column_points:
            return [("all", candidate_group)]

        attempts: list[tuple[str, CandidateGroup]] = []
        for label, points in (
            ("preferred_column_only", preferred_column_points),
            ("alternate_column_only", alternate_column_points),
            ("unlabeled_only", unlabeled_points),
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
        compute_subcategory: Optional[str] = None,
    ) -> Optional[QueryResult]:
        include_output_signature = compute_subcategory == _QUANTIZED_MATMUL_SUBCATEGORY
        if include_output_signature:
            index = self._get_compute_index(
                target.kernel_type,
                tc_input_count,
                include_output_signature=True,
            )
        else:
            index = self._get_compute_index(target.kernel_type, tc_input_count)
        if index is None:
            self._record_miss(
                "compute_csv_not_found",
                kernel_type=target.kernel_type,
                interpolation_path=interpolation_path,
            )
            return None
        allow_extra_fields = {"output_count", "output_dtypes", "output_formats"}
        if compute_subcategory != _QUANTIZED_MATMUL_SUBCATEGORY:
            allow_extra_fields.add("input_formats")
        candidate_groups = index.candidate_groups_matching(target.regime_key, allow_extra_fields=allow_extra_fields)
        if not candidate_groups:
            self._record_miss(
                "regime_key_unmatched",
                kernel_type=target.kernel_type,
                interpolation_path=interpolation_path,
                target_axes=target.axes,
                target_regime_key=dict(target.regime_key),
                index_diagnostics=self._compute_index_diagnostics.get(
                    (target.kernel_type, include_output_signature),
                    {},
                ),
            )
            return None
        attempts: list[dict[str, Any]] = []
        override = self._kernel_overrides.get(target.kernel_type, {})
        for candidate_group in sorted(candidate_groups, key=self._compute_candidate_group_rank):
            for latency_column_group, candidate_subset in self._latency_column_pure_candidate_group_attempts(
                candidate_group
            ):
                result = candidate_subset.interpolate(
                    target.axes,
                    _COMPUTE_AXIS_GROUPS,
                    fallback_from=fallback_from,
                    max_interpolation_dim=override.get("max_interpolation_dim"),
                    extra_details={
                        "kernel_type": target.kernel_type,
                        "query_mode": target.query_mode,
                        "interpolation_path": interpolation_path,
                        "latency_column_group": latency_column_group,
                        **({"compute_subcategory": compute_subcategory} if compute_subcategory is not None else {}),
                    },
                )
                if result is None:
                    attempts.append(
                        {
                            "regime_key": dict(candidate_subset.regime_key),
                            "latency_column_group": latency_column_group,
                            "diagnostics": candidate_subset.last_diagnostics,
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
                if compute_subcategory is not None:
                    result.details["interpolation_path"] = f"{compute_subcategory}_{result.interpolation_dim}d"
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
        compute_subcategory = self._compute_subcategory(mapping)
        if compute_subcategory != _QUANTIZED_MATMUL_SUBCATEGORY:
            compute_subcategory = None
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
                interpolation_path=compute_subcategory or "multidim",
                compute_subcategory=compute_subcategory,
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

    # ---- MoE / DispatchFFNCombine interpolation ----

    def _candidate_from_moe_fused_row(
        self,
        row: Any,
        kernel_type: str,
        latency_col: str,
        row_index: int,
    ) -> tuple[Optional[CandidatePoint], Optional[str]]:
        csv_shapes = _parse_shape_str(str(row.get("Input Shapes", "")))
        csv_dtypes = _parse_str_list(str(row.get("Input Data Types", "")))
        csv_formats = _parse_str_list(str(row.get("Input Formats", "")))
        if not csv_shapes:
            return None, "input_shapes_missing"
        if not csv_dtypes:
            return None, "input_dtypes_missing"
        if not csv_formats:
            return None, "input_formats_missing"
        if len(csv_shapes) != len(csv_dtypes) or len(csv_shapes) != len(csv_formats):
            return None, "moe_physical_signature_incomplete"
        if len(csv_shapes) != 7:
            return None, "moe_physical_signature_incomplete"
        input_dtype_signature = tuple(csv_dtypes)

        logical_shapes = [tuple(shape) for shape in csv_shapes]
        gmm1_weight_shape = logical_shapes[1]
        gmm2_weight_shape = logical_shapes[2]
        if len(gmm1_weight_shape) < 3 or len(gmm2_weight_shape) < 3:
            return None, "moe_weight_shapes_unextractable"
        first_format = csv_formats[0]
        first_shape = self._logical_csv_shape(tuple(csv_shapes[0]), first_format)
        logical_shapes[0] = first_shape
        shape_info = self._moe_fused_activation_shape_info(first_shape)
        if shape_info is None:
            return None, "token_axis_unextractable"
        tokens, hidden, leading_dims = shape_info

        latency, latency_meta = self._candidate_latency(row, latency_col)
        if latency is None:
            return None, str(latency_meta["latency_rejected_reason"])

        has_ep_size = hasattr(row, "index") and "EP Size" in row.index
        ep_size = _to_int_cell(row.get("EP Size")) if has_ep_size and hasattr(row, "get") else None
        if not has_ep_size or ep_size is None:
            return None, "ep_size_missing"
        if ep_size <= 0:
            return None, "ep_size_invalid"
        topk = self._moe_fused_topk(
            logical_shapes,
            tokens=tokens,
            input_dtypes=csv_dtypes,
        )
        if topk is None:
            return None, "topk_unextractable"
        axes = {"tokens": float(tokens)}
        regime_fields: list[tuple[str, Any]] = [
            ("kernel_type", kernel_type),
            ("activation_dtype", self._dtype_key(kernel_type, csv_dtypes[0])),
            ("activation_format", first_format),
            ("input_dtype_signature", input_dtype_signature),
            ("gmm1_weight_shape", gmm1_weight_shape),
            ("gmm2_weight_shape", gmm2_weight_shape),
            ("hidden", hidden),
            ("topk", topk),
            ("ep_size", ep_size),
        ]

        return CandidatePoint(
            kernel_type=kernel_type,
            axes=axes,
            latency_us=latency,
            regime_key=make_regime_key(regime_fields),
            input_shapes=logical_shapes,
            input_dtypes=csv_dtypes,
            input_formats=csv_formats,
            row_index=row_index,
            row_meta={
                "tokens": tokens,
                "hidden": hidden,
                "leading_dims": leading_dims,
                "ep_size": ep_size,
                "input_dtype_signature": input_dtype_signature,
                "input_format_signature": tuple(csv_formats),
                "gmm1_weight_shape": gmm1_weight_shape,
                "gmm2_weight_shape": gmm2_weight_shape,
                "moe_axes": dict(axes),
                **latency_meta,
            },
        ), None

    def _get_moe_fused_index(self, kernel_type: str) -> tuple[Optional[CandidateIndex], dict[str, int]]:
        df = self.base._load_csv(kernel_type)
        if df is None:
            return None, {}
        cache_key = (
            "moe_fused",
            kernel_type,
            self._dataframe_fingerprint(df),
            self._policy_hash,
        )
        if cache_key in self._moe_fused_index_cache:
            return self._moe_fused_index_cache[cache_key]

        latency_col = self.base._latency_col(df)
        rejected_reasons: dict[str, int] = {}
        if "EP Size" not in df.columns:
            rejected_reasons["ep_size_missing"] = len(df)
            index = CandidateIndex([])
            self._moe_fused_index_cache[cache_key] = (index, rejected_reasons)
            return index, rejected_reasons
        index, rejected_reasons = self._build_candidate_index(
            df,
            lambda row, row_index: self._candidate_from_moe_fused_row(row, kernel_type, latency_col, row_index),
        )
        self._moe_fused_index_cache[cache_key] = (index, rejected_reasons)
        return index, rejected_reasons

    def _build_moe_fused_target(self, op_invoke_info: "OpInvokeInfo", mapping: dict) -> Optional[InterpolationTarget]:
        kernel_type = mapping.get("kernel_type")
        if not kernel_type:
            return None
        projected_inputs = _project_dispatch_ffn_combine_inputs(op_invoke_info)
        if projected_inputs is None:
            return None

        projected_shapes = [tuple(int(dim) for dim in shape) for shape, _dtype in projected_inputs]
        first_shape, first_dtype = projected_inputs[0]
        shape_info = self._moe_fused_activation_shape_info(tuple(first_shape))
        if shape_info is None:
            return None
        tokens, hidden, _leading_dims = shape_info
        dtype_str = DTYPE_MAP.get(first_dtype, str(first_dtype))
        input_dtype_signature = tuple(DTYPE_MAP.get(dtype, str(dtype)) for _shape, dtype in projected_inputs)
        gmm1_weight_shape = projected_shapes[1]
        gmm2_weight_shape = projected_shapes[2]

        regime_fields: list[tuple[str, Any]] = [
            ("kernel_type", kernel_type),
            ("activation_dtype", self._dtype_key(kernel_type, dtype_str)),
            ("activation_format", "ND"),
            ("input_dtype_signature", input_dtype_signature),
            ("gmm1_weight_shape", gmm1_weight_shape),
            ("gmm2_weight_shape", gmm2_weight_shape),
            ("hidden", hidden),
        ]
        if self.base.ep_size is None:
            return None
        regime_fields.append(("ep_size", int(self.base.ep_size)))

        topk = self._moe_fused_topk(
            projected_shapes,
            tokens=tokens,
            input_dtypes=list(input_dtype_signature),
        )
        if topk is None:
            return None
        regime_fields.append(("topk", topk))
        return InterpolationTarget(
            func_name=_normalize_func_name(op_invoke_info.func),
            kernel_type=kernel_type,
            axes={"tokens": float(tokens)},
            regime_key=make_regime_key(regime_fields),
            tc_shapes=projected_shapes,
            input_dtypes=[dtype_str],
            query_mode="moe_fused",
        )

    def _interpolate_moe_fused(
        self,
        op_invoke_info: "OpInvokeInfo",
        mapping: dict,
        *,
        fallback_from: str = "exact_miss",
    ) -> Optional[QueryResult]:
        kernel_type = mapping.get("kernel_type")
        if not kernel_type:
            return None
        override = self._kernel_overrides.get(kernel_type, {})
        if self.base.ep_size is None:
            self._record_miss("ep_size_not_configured", kernel_type=kernel_type, query_mode="moe_fused")
            return None

        index, rejected_reasons = self._get_moe_fused_index(kernel_type)
        if index is None:
            self._record_miss("csv_not_found", kernel_type=kernel_type, query_mode="moe_fused")
            return None
        target = self._build_moe_fused_target(op_invoke_info, mapping)
        if target is None:
            self._record_miss("moe_fused_target_unextractable", kernel_type=kernel_type)
            return None

        target_regime = dict(target.regime_key)
        candidate_groups = index.candidate_groups_matching(target.regime_key)
        attempts: list[dict[str, Any]] = []
        if not candidate_groups:
            attempts.append({"status": "regime_key_unmatched", "target_regime": dict(target.regime_key)})
        for candidate_group in candidate_groups:
            for latency_column_group, candidate_subset in self._latency_column_pure_candidate_group_attempts(
                candidate_group
            ):
                result = candidate_subset.interpolate(
                    target.axes,
                    _MOE_FUSED_AXIS_GROUPS,
                    max_interpolation_dim=override.get("max_interpolation_dim"),
                    fallback_from=fallback_from,
                    extra_details={
                        "kernel_type": target.kernel_type,
                        "query_mode": "moe_fused",
                        "interpolation_path": "moe_fused_1d",
                        "latency_column_group": latency_column_group,
                        "ep_size": self.base.ep_size,
                        "input_dtype_signature": target_regime.get("input_dtype_signature"),
                        "target_moe_axes": dict(target.axes),
                        "rejected_reasons": rejected_reasons,
                    },
                )
                if result is None:
                    attempts.append(
                        {
                            "status": "candidate_group_failed",
                            "regime_key": dict(candidate_subset.regime_key),
                            "latency_column_group": latency_column_group,
                            "diagnostics": candidate_subset.last_diagnostics,
                        }
                    )
                    continue
                return self._query_result_from_interpolation(target, result)

        self._record_miss(
            self._candidate_failure_reason(
                "moe_fused_interpolation_failed", attempts[-1].get("diagnostics") if attempts else {}
            ),
            kernel_type=kernel_type,
            query_mode="moe_fused",
            candidate_count=len(index.points),
            rejected_reasons=rejected_reasons,
            attempts=attempts,
            target_axes=target.axes,
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
    def _attention_q_tokens_match(candidate_value: float, target_value: float) -> bool:
        candidate = int(candidate_value)
        target = int(target_value)
        return candidate == target or _is_block_padded(candidate, target) or _is_block_padded(target, candidate)

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

    # ---- Attention auxiliary: LightningIndexer ----

    @staticmethod
    def _runtime_attention_regime_fields(
        kernel_type: str,
        dtype_str: str,
        q_shape: tuple[int, ...],
        workload: dict[str, Any],
        params: dict[str, Any],
        *,
        include_sparse_fields: bool,
    ) -> Optional[list[tuple[str, Any]]]:
        required = (
            "sparse_mode",
            "num_kv_heads",
            "input_layout",
            "topk",
            "block_size",
            "num_heads",
            "cache_layout",
            "kv_cache_mode",
        )
        if any(params.get(field) is None for field in required):
            return None
        sparse_mode = _to_int_cell(params["sparse_mode"])
        kv_heads = _to_int_cell(params["num_kv_heads"])
        topk = _to_int_cell(params["topk"])
        block_size = _to_int_cell(params["block_size"])
        num_heads = _to_int_cell(params["num_heads"])
        if (
            sparse_mode is None
            or kv_heads is None
            or kv_heads <= 0
            or topk is None
            or topk <= 0
            or block_size is None
            or block_size <= 0
            or num_heads is None
            or num_heads <= 0
        ):
            return None
        fields: list[tuple[str, Any]] = [
            ("kernel_type", kernel_type),
            ("dtype", dtype_str),
            ("phase", workload["phase"]),
            ("query_rank", len(q_shape)),
            ("head_dim", int(q_shape[-1])),
            ("sparse_mode", sparse_mode),
            ("kv_heads", kv_heads),
            ("input_layout", str(params["input_layout"])),
            ("topk", topk),
            ("block_size", block_size),
            ("num_heads", num_heads),
            ("cache_layout", str(params["cache_layout"])),
            ("kv_cache_mode", str(params["kv_cache_mode"])),
        ]
        if include_sparse_fields:
            sparse_required = ("sparse_block_size", "sparse_indices_pattern", "sparse_indices_valid_count")
            if any(params.get(field) is None for field in sparse_required):
                return None
            sparse_block_size = _to_int_cell(params["sparse_block_size"])
            sparse_indices_valid_count = _to_int_cell(params["sparse_indices_valid_count"])
            if (
                sparse_block_size is None
                or sparse_block_size <= 0
                or sparse_indices_valid_count is None
                or sparse_indices_valid_count <= 0
            ):
                return None
            fields.extend(
                [
                    ("sparse_block_size", sparse_block_size),
                    ("sparse_indices_pattern", str(params["sparse_indices_pattern"])),
                    ("sparse_indices_valid_count", sparse_indices_valid_count),
                ]
            )
        return fields

    def _candidate_from_runtime_attention_row(
        self,
        row: Any,
        kernel_type: str,
        latency_col: str,
        row_index: int,
        *,
        include_sparse_fields: bool,
    ) -> tuple[Optional[CandidatePoint], Optional[str]]:
        if "Runtime metadata_completeness" in row.index:
            completeness = _optional_str_cell(row.get("Runtime metadata_completeness"))
            if completeness != "complete":
                return None, "runtime_metadata_incomplete"
        input_shapes = _parse_shape_str(str(row.get("Input Shapes", "")))
        input_dtypes = _parse_str_list(str(row.get("Input Data Types", "")))
        if not input_shapes or not input_dtypes:
            return None, "input_signature_missing"
        q_shape = tuple(input_shapes[0])
        if len(q_shape) != 3 or any(int(dim) <= 0 for dim in q_shape):
            return None, "query_shape_unextractable"
        q_tokens = int(q_shape[0])
        cumulative = self._runtime_int_values(row.get("Runtime actual_seq_lengths_values"))
        kv_lengths = self._runtime_int_values(row.get("Runtime actual_seq_lengths_kv_values"))
        if cumulative is None or kv_lengths is None:
            return None, "runtime_sequence_values_missing"
        query_lengths = self._query_lengths_from_cumulative(q_tokens, cumulative)
        if query_lengths is None:
            return None, "runtime_query_offsets_invalid"
        workload = self._attention_runtime_workload(
            q_tokens=q_tokens,
            query_lengths=query_lengths,
            kv_lengths=kv_lengths,
        )
        if workload is None:
            return None, "runtime_workload_unextractable"
        params = {
            "sparse_mode": _to_int_cell(row.get("Runtime sparse_mode")),
            "num_kv_heads": _to_int_cell(row.get("Runtime num_key_value_heads")),
            "input_layout": _optional_str_cell(row.get("Runtime input_layout")),
            "topk": _to_int_cell(row.get("Runtime topk")),
            "block_size": _to_int_cell(row.get("Runtime block_size")),
            "num_heads": _to_int_cell(row.get("Runtime num_heads")),
            "cache_layout": _optional_str_cell(row.get("Runtime cache_layout")),
            "kv_cache_mode": _optional_str_cell(row.get("Runtime kv_cache_mode")),
            "sparse_block_size": _to_int_cell(row.get("Runtime sparse_block_size")),
            "sparse_indices_pattern": _optional_str_cell(row.get("Runtime sparse_indices_pattern")),
            "sparse_indices_valid_count": _to_int_cell(row.get("Runtime sparse_indices_valid_count")),
        }
        regime_fields = self._runtime_attention_regime_fields(
            kernel_type,
            input_dtypes[0],
            q_shape,
            workload,
            params,
            include_sparse_fields=include_sparse_fields,
        )
        if regime_fields is None:
            return None, "runtime_regime_incomplete"
        latency, latency_meta = self._candidate_latency(row, latency_col)
        if latency is None:
            return None, str(latency_meta["latency_rejected_reason"])
        return CandidatePoint(
            kernel_type=kernel_type,
            axes={
                "q_tokens": workload["q_tokens"],
                "effective_kv_len": workload["effective_kv_len"],
            },
            latency_us=latency,
            regime_key=make_regime_key(regime_fields),
            input_shapes=[q_shape],
            input_dtypes=[input_dtypes[0]],
            row_index=row_index,
            row_meta={
                "phase": workload["phase"],
                "batch_size": workload["batch_size"],
                "query_lengths": query_lengths,
                "kv_lengths": kv_lengths,
                **latency_meta,
            },
        ), None

    def _candidate_from_lightning_indexer_row(
        self,
        row: Any,
        kernel_type: str,
        latency_col: str,
        row_index: int,
    ) -> tuple[Optional[CandidatePoint], Optional[str]]:
        return self._candidate_from_runtime_attention_row(
            row,
            kernel_type,
            latency_col,
            row_index,
            include_sparse_fields=False,
        )

    def _candidate_from_sparse_attention_row(
        self,
        row: Any,
        kernel_type: str,
        latency_col: str,
        row_index: int,
    ) -> tuple[Optional[CandidatePoint], Optional[str]]:
        return self._candidate_from_runtime_attention_row(
            row,
            kernel_type,
            latency_col,
            row_index,
            include_sparse_fields=True,
        )

    def _build_runtime_attention_target_from_params(
        self,
        kernel_type: str,
        params: dict[str, Any],
        dtype_str: str,
        *,
        include_sparse_fields: bool,
    ) -> Optional[InterpolationTarget]:
        q_shape_value = params.get("q_shape_3d")
        if not isinstance(q_shape_value, (list, tuple)):
            return None
        try:
            q_shape = tuple(int(dim) for dim in q_shape_value)
        except (TypeError, ValueError):
            return None
        if len(q_shape) != 3 or any(dim <= 0 for dim in q_shape):
            return None
        q_tokens = q_shape[0]
        cumulative = self._runtime_int_values(params.get("actual_seq_lengths_values"))
        kv_lengths = self._runtime_int_values(params.get("actual_seq_lengths_kv_values"))
        if cumulative is None or kv_lengths is None:
            return None
        query_lengths = self._query_lengths_from_cumulative(q_tokens, cumulative)
        if query_lengths is None:
            return None
        workload = self._attention_runtime_workload(
            q_tokens=q_tokens,
            query_lengths=query_lengths,
            kv_lengths=kv_lengths,
        )
        if workload is None:
            return None
        regime_fields = self._runtime_attention_regime_fields(
            kernel_type,
            dtype_str,
            q_shape,
            workload,
            params,
            include_sparse_fields=include_sparse_fields,
        )
        if regime_fields is None:
            return None
        path = "sparse_attention" if include_sparse_fields else "lightning_indexer"
        return InterpolationTarget(
            func_name=kernel_type,
            kernel_type=kernel_type,
            axes={
                "q_tokens": workload["q_tokens"],
                "effective_kv_len": workload["effective_kv_len"],
            },
            regime_key=make_regime_key(regime_fields),
            tc_shapes=[q_shape],
            input_dtypes=[dtype_str],
            query_mode=path,
            metadata={
                "phase": workload["phase"],
                "batch_size": workload["batch_size"],
            },
        )

    def _get_lightning_indexer_index(self, kernel_type: str) -> tuple[Optional[CandidateIndex], dict[str, int]]:
        df = self.base._load_csv(kernel_type)
        if df is None:
            return None, {}
        cache_key = ("lightning_indexer", kernel_type, self._dataframe_fingerprint(df), self._policy_hash)
        if cache_key in self._lightning_indexer_index_cache:
            return self._lightning_indexer_index_cache[cache_key]
        latency_col = self.base._latency_col(df)
        index, rejected_reasons = self._build_candidate_index(
            df,
            lambda row, row_index: self._candidate_from_lightning_indexer_row(
                row,
                kernel_type,
                latency_col,
                row_index,
            ),
        )
        self._lightning_indexer_index_cache[cache_key] = (index, rejected_reasons)
        return index, rejected_reasons

    def _get_sparse_attention_index(self, kernel_type: str) -> tuple[Optional[CandidateIndex], dict[str, int]]:
        df = self.base._load_csv(kernel_type)
        if df is None:
            return None, {}
        cache_key = ("sparse_attention", kernel_type, self._dataframe_fingerprint(df), self._policy_hash)
        if cache_key in self._sparse_attention_index_cache:
            return self._sparse_attention_index_cache[cache_key]
        latency_col = self.base._latency_col(df)
        index, rejected_reasons = self._build_candidate_index(
            df,
            lambda row, row_index: self._candidate_from_sparse_attention_row(
                row,
                kernel_type,
                latency_col,
                row_index,
            ),
        )
        self._sparse_attention_index_cache[cache_key] = (index, rejected_reasons)
        return index, rejected_reasons

    # ---- V4 MLA cache update: ScatterNdUpdate ----

    @staticmethod
    def _scatter_update_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
        if len(shape) <= 2:
            return shape
        return (int(math.prod(shape[:-1])), int(shape[-1]))

    def _candidate_from_scatter_nd_update_row(
        self,
        row: Any,
        kernel_type: str,
        latency_col: str,
        row_index: int,
    ) -> tuple[Optional[CandidatePoint], Optional[str]]:
        input_shapes = _parse_shape_str(str(row.get("Input Shapes", "")))
        input_dtypes = _parse_str_list(str(row.get("Input Data Types", "")))
        input_formats = _parse_str_list(str(row.get("Input Formats", "")))
        if len(input_shapes) < 3:
            return None, "input_shapes_missing"
        if len(input_dtypes) < 3:
            return None, "input_dtypes_missing"
        if len(input_formats) < 3:
            return None, "input_formats_missing"
        cache_shape = tuple(input_shapes[0])
        update_shape = self._scatter_update_shape(tuple(input_shapes[2]))
        if not cache_shape or not update_shape:
            return None, "scatter_shape_unextractable"
        latency, latency_meta = self._candidate_latency(row, latency_col)
        if latency is None:
            return None, str(latency_meta["latency_rejected_reason"])
        key_fields: list[tuple[str, Any]] = [
            ("kernel_type", kernel_type),
            ("cache_dtype", input_dtypes[0]),
            ("index_dtype", input_dtypes[1]),
            ("update_dtype", input_dtypes[2]),
            ("full_cache_shape", cache_shape),
            ("update_tail", update_shape[1:]),
            ("input_formats", tuple(input_formats[:3])),
        ]
        return CandidatePoint(
            kernel_type=kernel_type,
            axes={"tokens": float(update_shape[0])},
            latency_us=latency,
            regime_key=make_regime_key(key_fields),
            input_shapes=[cache_shape, tuple(input_shapes[1]), update_shape],
            input_dtypes=input_dtypes[:3],
            input_formats=input_formats[:3],
            row_index=row_index,
            row_meta=latency_meta,
        ), None

    def _build_scatter_nd_update_target(
        self,
        op_invoke_info: "OpInvokeInfo",
        kernel_type: str,
    ) -> Optional[InterpolationTarget]:
        args = getattr(op_invoke_info, "args", ())
        if len(args) < 3:
            return None
        # TensorCast calls (update, cache, index), while the profiling CSV stores
        # physical inputs as (cache, index, update).
        update, cache, index = args[:3]
        if not all(isinstance(arg, torch.Tensor) for arg in (update, cache, index)):
            return None
        update_shape = self._scatter_update_shape(tuple(update.shape))
        cache_shape = tuple(cache.shape)
        if not update_shape or not cache_shape:
            return None
        cache_dtype = DTYPE_MAP.get(cache.dtype)
        update_dtype = DTYPE_MAP.get(update.dtype)
        index_dtype = DTYPE_MAP.get(index.dtype)
        if cache_dtype is None or update_dtype is None or index_dtype is None:
            return None
        return InterpolationTarget(
            func_name=_normalize_func_name(op_invoke_info.func),
            kernel_type=kernel_type,
            axes={"tokens": float(update_shape[0])},
            regime_key=make_regime_key(
                [
                    ("kernel_type", kernel_type),
                    ("cache_dtype", cache_dtype),
                    ("index_dtype", index_dtype),
                    ("update_dtype", update_dtype),
                    ("full_cache_shape", cache_shape),
                    ("update_tail", update_shape[1:]),
                    # TensorCast tensors do not carry physical format metadata;
                    # the current cache-update mapping is intentionally ND-only.
                    ("input_formats", ("ND", "ND", "ND")),
                ]
            ),
            tc_shapes=[cache_shape, tuple(index.shape), update_shape],
            input_dtypes=[cache_dtype, index_dtype, update_dtype],
            query_mode="scatter_nd_update_mla",
        )

    @staticmethod
    def _build_scatter_nd_update_target_from_shapes(
        kernel_type: str,
        input_shapes: Sequence[Tuple[int, ...]],
        input_dtypes: Sequence[str],
    ) -> Optional[InterpolationTarget]:
        if len(input_shapes) < 3 or len(input_dtypes) < 3:
            return None
        cache_shape = tuple(input_shapes[0])
        index_shape = tuple(input_shapes[1])
        update_shape = InterpolatingDataSource._scatter_update_shape(tuple(input_shapes[2]))
        if not cache_shape or not index_shape or not update_shape:
            return None
        return InterpolationTarget(
            func_name=kernel_type,
            kernel_type=kernel_type,
            axes={"tokens": float(update_shape[0])},
            regime_key=make_regime_key(
                [
                    ("kernel_type", kernel_type),
                    ("cache_dtype", input_dtypes[0]),
                    ("index_dtype", input_dtypes[1]),
                    ("update_dtype", input_dtypes[2]),
                    ("full_cache_shape", cache_shape),
                    ("update_tail", update_shape[1:]),
                    ("input_formats", ("ND", "ND", "ND")),
                ]
            ),
            tc_shapes=[cache_shape, index_shape, update_shape],
            input_dtypes=list(input_dtypes[:3]),
            query_mode="scatter_cache_write",
        )

    def _get_scatter_nd_update_index(self, kernel_type: str) -> tuple[Optional[CandidateIndex], dict[str, int]]:
        df = self.base._load_csv(kernel_type)
        if df is None:
            return None, {}
        cache_key = ("scatter_nd_update", kernel_type, self._dataframe_fingerprint(df), self._policy_hash)
        if cache_key in self._scatter_nd_update_index_cache:
            return self._scatter_nd_update_index_cache[cache_key]
        latency_col = self.base._latency_col(df)
        index, rejected_reasons = self._build_candidate_index(
            df,
            lambda row, row_index: self._candidate_from_scatter_nd_update_row(
                row,
                kernel_type,
                latency_col,
                row_index,
            ),
        )
        self._scatter_nd_update_index_cache[cache_key] = (index, rejected_reasons)
        return index, rejected_reasons

    def _interpolate_scatter_nd_update_by_shapes(
        self,
        kernel_types: Sequence[str],
        input_shapes: Sequence[Tuple[int, ...]],
        input_dtypes: Sequence[str],
    ) -> Optional[QueryResult]:
        attempts: list[dict[str, Any]] = []
        for kernel_type in kernel_types:
            index, rejected_reasons = self._get_scatter_nd_update_index(kernel_type)
            if index is None:
                attempts.append({"kernel_type": kernel_type, "status": "csv_not_found"})
                continue
            target = self._build_scatter_nd_update_target_from_shapes(
                kernel_type,
                input_shapes,
                input_dtypes,
            )
            if target is None:
                attempts.append({"kernel_type": kernel_type, "status": "scatter_target_unextractable"})
                continue
            override = self._kernel_overrides.get(kernel_type, {})
            candidate_groups = index.candidate_groups_matching(target.regime_key)
            if not candidate_groups:
                attempts.append(
                    {
                        "kernel_type": kernel_type,
                        "status": "regime_key_unmatched",
                        "target_regime": dict(target.regime_key),
                        "rejected_reasons": rejected_reasons,
                    }
                )
                continue
            for candidate_group in candidate_groups:
                for latency_column_group, candidate_subset in self._latency_column_pure_candidate_group_attempts(
                    candidate_group
                ):
                    result = candidate_subset.interpolate(
                        target.axes,
                        _SCATTER_ND_UPDATE_AXIS_GROUPS,
                        max_interpolation_dim=override.get("max_interpolation_dim"),
                        fallback_from="composite",
                        extra_details={
                            "kernel_type": kernel_type,
                            "query_mode": "scatter_cache_write",
                            "interpolation_path": "scatter_cache_write_1d",
                            "latency_column_group": latency_column_group,
                            "rejected_reasons": rejected_reasons,
                        },
                    )
                    if result is not None:
                        return self._query_result_from_interpolation(target, result)
                    attempts.append(
                        {
                            "kernel_type": kernel_type,
                            "status": "candidate_group_failed",
                            "latency_column_group": latency_column_group,
                            "diagnostics": candidate_subset.last_diagnostics,
                        }
                    )
        final_diagnostics = attempts[-1].get("diagnostics") if attempts else None
        self._record_miss(
            self._candidate_failure_reason("scatter_cache_write_interpolation_failed", final_diagnostics),
            attempted_kernel_types=list(kernel_types),
            attempts=attempts,
        )
        return None

    def _interpolate_scatter_nd_update(
        self,
        op_invoke_info: "OpInvokeInfo",
        mapping: dict,
        *,
        fallback_from: str = "exact_miss",
    ) -> Optional[QueryResult]:
        kernel_types = [mapping.get("kernel_type")]
        kernel_types.extend(mapping.get("alternate_kernel_types") or [])
        kernel_types = [str(kernel_type) for kernel_type in kernel_types if kernel_type]
        attempts: list[dict[str, Any]] = []
        for kernel_type in kernel_types:
            index, rejected_reasons = self._get_scatter_nd_update_index(kernel_type)
            if index is None:
                attempts.append({"kernel_type": kernel_type, "status": "csv_not_found"})
                continue
            target = self._build_scatter_nd_update_target(op_invoke_info, kernel_type)
            if target is None:
                attempts.append({"kernel_type": kernel_type, "status": "scatter_target_unextractable"})
                continue
            candidate_groups = index.candidate_groups_matching(target.regime_key)
            if not candidate_groups:
                attempts.append(
                    {
                        "kernel_type": kernel_type,
                        "status": "regime_key_unmatched",
                        "target_regime": dict(target.regime_key),
                    }
                )
                continue
            override = self._kernel_overrides.get(kernel_type, {})
            for candidate_group in candidate_groups:
                for latency_column_group, candidate_subset in self._latency_column_pure_candidate_group_attempts(
                    candidate_group
                ):
                    result = candidate_subset.interpolate(
                        target.axes,
                        _SCATTER_ND_UPDATE_AXIS_GROUPS,
                        max_interpolation_dim=override.get("max_interpolation_dim"),
                        fallback_from=fallback_from,
                        extra_details={
                            "kernel_type": kernel_type,
                            "query_mode": "scatter_nd_update_mla",
                            "interpolation_path": "scatter_nd_update_mla_1d",
                            "latency_column_group": latency_column_group,
                            "rejected_reasons": rejected_reasons,
                        },
                    )
                    if result is not None:
                        return self._query_result_from_interpolation(target, result)
                    attempts.append(
                        {
                            "kernel_type": kernel_type,
                            "status": "candidate_group_failed",
                            "latency_column_group": latency_column_group,
                            "diagnostics": candidate_subset.last_diagnostics,
                        }
                    )

        self._record_miss(
            self._candidate_failure_reason(
                "scatter_nd_update_interpolation_failed",
                attempts[-1].get("diagnostics") if attempts else {},
            ),
            query_mode="scatter_nd_update_mla",
            attempted_kernel_types=kernel_types,
            attempts=attempts,
        )
        return None

    # ---- Compute subcategory: quantization scale ----

    @staticmethod
    def _compute_scale_axes(shape: Tuple[int, ...]) -> Optional[dict[str, float]]:
        if not shape or any(int(dim) <= 0 for dim in shape):
            return None
        tokens = math.prod(int(dim) for dim in shape[:-1])
        return {"M": float(tokens), "K": float(shape[-1])}

    @staticmethod
    def _compute_scale_profiling_dtype(dtype: torch.dtype) -> Optional[str]:
        if dtype == torch.float16:
            return "DT_FLOAT16"
        return DTYPE_MAP.get(dtype)

    @staticmethod
    def _scalar_aware_numel(shape: Tuple[int, ...]) -> Optional[int]:
        numel = 1
        for dim in shape:
            if int(dim) < 0:
                return None
            numel *= int(dim)
        return numel

    @classmethod
    def _compute_scale_mode(
        cls,
        input_shape: Tuple[int, ...],
        scale_shape: Tuple[int, ...],
        kernel_type: str,
    ) -> Optional[tuple[str, Optional[int]]]:
        axes = cls._compute_scale_axes(input_shape)
        scale_numel = cls._scalar_aware_numel(scale_shape)
        if axes is None or scale_numel is None or scale_numel <= 0:
            return None
        if not scale_shape:
            return "per_tensor", None
        tokens = int(axes["M"])
        channels = int(axes["K"])
        if kernel_type == "DynamicBlockQuant":
            if len(scale_shape) != 1 or scale_numel > channels:
                return None
            block_size = (channels + scale_numel - 1) // scale_numel
            return "per_block", block_size
        if scale_numel == tokens:
            return "per_token", None
        if scale_numel == channels:
            return "per_channel", None
        return None

    @staticmethod
    def _output_tensors(output: Any) -> Optional[list[torch.Tensor]]:
        if isinstance(output, torch.Tensor):
            return [output]
        if isinstance(output, (list, tuple)) and output and all(isinstance(item, torch.Tensor) for item in output):
            return list(output)
        return None

    def _candidate_from_compute_scale_row(
        self,
        row: Any,
        kernel_type: str,
        latency_col: str,
        row_index: int,
    ) -> tuple[Optional[CandidatePoint], Optional[str]]:
        input_shapes = _parse_shape_str(str(row.get("Input Shapes", "")))
        input_dtypes = _parse_str_list(str(row.get("Input Data Types", "")))
        input_formats = _parse_str_list(str(row.get("Input Formats", "")))
        output_shapes = _parse_shape_str(str(row.get("Output Shapes", "")))
        output_dtypes = _parse_str_list(str(row.get("Output Data Types", "")))
        output_formats = _parse_str_list(str(row.get("Output Formats", "")))
        if not input_shapes:
            return None, "input_shapes_missing"
        if not input_dtypes:
            return None, "input_dtypes_missing"
        if not input_formats:
            return None, "input_formats_missing"
        if len(output_shapes) < 2:
            return None, "compute_scale_outputs_lt_2"
        if len(output_dtypes) != len(output_shapes):
            return None, "output_dtypes_mismatch"
        if len(output_formats) != len(output_shapes):
            return None, "output_formats_mismatch"

        input_shape = self._logical_csv_shape(tuple(input_shapes[0]), input_formats[0])
        quant_shape = self._logical_csv_shape(tuple(output_shapes[0]), output_formats[0])
        if quant_shape != input_shape:
            return None, "quant_output_shape_mismatch"
        axes = self._compute_scale_axes(input_shape)
        extracted_modes = [
            self._compute_scale_mode(input_shape, tuple(shape), kernel_type) for shape in output_shapes[1:]
        ]
        if axes is None or any(mode is None for mode in extracted_modes):
            return None, "compute_scale_mode_unextractable"
        auxiliary_modes = tuple(mode for mode in extracted_modes if mode is not None)
        scale_mode, block_size = auxiliary_modes[0]
        latency, latency_meta = self._candidate_latency(row, latency_col)
        if latency is None:
            return None, str(latency_meta["latency_rejected_reason"])

        regime_fields: list[tuple[str, Any]] = [
            ("kernel_type", kernel_type),
            ("input_dtype", self._dtype_key(kernel_type, input_dtypes[0])),
            ("input_format", input_formats[0]),
            ("output_count", len(output_shapes)),
            ("output_dtypes", tuple(output_dtypes)),
            ("output_formats", tuple(output_formats)),
            ("scale_mode", scale_mode),
            ("auxiliary_modes", auxiliary_modes),
        ]
        if block_size is not None:
            regime_fields.append(("block_size", block_size))
        return CandidatePoint(
            kernel_type=kernel_type,
            axes=axes,
            latency_us=latency,
            regime_key=make_regime_key(regime_fields),
            input_shapes=[input_shape],
            input_dtypes=[input_dtypes[0]],
            input_formats=[input_formats[0]],
            row_index=row_index,
            row_meta={
                "compute_subcategory": _COMPUTE_SCALE_SUBCATEGORY,
                "scale_mode": scale_mode,
                "block_size": block_size,
                "auxiliary_modes": auxiliary_modes,
                **latency_meta,
            },
        ), None

    def _get_compute_scale_index(self, kernel_type: str) -> Optional[CandidateIndex]:
        df = self.base._load_csv(kernel_type)
        if df is None:
            return None
        cache_key = (
            _COMPUTE_SCALE_SUBCATEGORY,
            kernel_type,
            self._dataframe_fingerprint(df),
            self._policy_hash,
        )
        if cache_key in self._compute_index_cache:
            return self._compute_index_cache[cache_key]
        latency_col = self.base._latency_col(df)
        index, rejected_reasons = self._build_candidate_index(
            df,
            lambda row, row_index: self._candidate_from_compute_scale_row(
                row,
                kernel_type,
                latency_col,
                row_index,
            ),
        )
        diagnostics_key = (_COMPUTE_SCALE_SUBCATEGORY, kernel_type)
        self._compute_index_diagnostics[diagnostics_key] = {
            "csv_rows": len(df),
            "usable_points": len(index.points),
            "rejected_reasons": rejected_reasons,
        }
        self._compute_index_cache[cache_key] = index
        return index

    def _build_compute_scale_target(
        self,
        op_invoke_info: "OpInvokeInfo",
        kernel_type: str,
    ) -> Optional[InterpolationTarget]:
        tc_inputs = self.base._extract_tensor_inputs(op_invoke_info)
        if not tc_inputs:
            return None
        input_shape, input_dtype = tc_inputs[0]
        input_shape = tuple(input_shape)
        axes = self._compute_scale_axes(input_shape)
        input_dtype_str = self._compute_scale_profiling_dtype(input_dtype)
        output_tensors = self._output_tensors(getattr(op_invoke_info, "out", None))
        if axes is None or input_dtype_str is None or output_tensors is None or len(output_tensors) < 2:
            return None
        output_shapes = [tuple(tensor.shape) for tensor in output_tensors]
        if output_shapes[0] != input_shape:
            return None
        output_dtypes = []
        for tensor in output_tensors:
            dtype_str = self._compute_scale_profiling_dtype(tensor.dtype)
            if dtype_str is None:
                return None
            output_dtypes.append(dtype_str)
        extracted_modes = [self._compute_scale_mode(input_shape, shape, kernel_type) for shape in output_shapes[1:]]
        if any(mode is None for mode in extracted_modes):
            return None
        auxiliary_modes = tuple(mode for mode in extracted_modes if mode is not None)
        scale_mode, block_size = auxiliary_modes[0]
        regime_fields: list[tuple[str, Any]] = [
            ("kernel_type", kernel_type),
            ("input_dtype", self._dtype_key(kernel_type, input_dtype_str)),
            ("input_format", "ND"),
            ("output_count", len(output_tensors)),
            ("output_dtypes", tuple(output_dtypes)),
            ("output_formats", tuple("ND" for _ in output_tensors)),
            ("scale_mode", scale_mode),
            ("auxiliary_modes", auxiliary_modes),
        ]
        if block_size is not None:
            regime_fields.append(("block_size", block_size))
        return InterpolationTarget(
            func_name=_normalize_func_name(op_invoke_info.func),
            kernel_type=kernel_type,
            axes=axes,
            regime_key=make_regime_key(regime_fields),
            tc_shapes=[input_shape],
            input_dtypes=[input_dtype_str],
            query_mode="compute_scale",
            metadata={
                "scale_mode": scale_mode,
                "block_size": block_size,
                "auxiliary_modes": auxiliary_modes,
            },
        )

    def _interpolate_compute_scale(
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
        kernel_types = [kernel_type]
        for alternate in mapping.get("alternate_kernel_types", []):
            if alternate not in kernel_types:
                kernel_types.append(alternate)

        attempts: list[dict[str, Any]] = []
        for candidate_kernel in kernel_types:
            target = self._build_compute_scale_target(op_invoke_info, candidate_kernel)
            if target is None:
                attempts.append({"kernel_type": candidate_kernel, "status": "target_unavailable"})
                continue
            index = self._get_compute_scale_index(candidate_kernel)
            diagnostics_key = (_COMPUTE_SCALE_SUBCATEGORY, candidate_kernel)
            if index is None:
                attempts.append({"kernel_type": candidate_kernel, "status": "csv_not_found"})
                continue
            candidate_groups = index.candidate_groups_matching(target.regime_key)
            if not candidate_groups:
                attempts.append(
                    {
                        "kernel_type": candidate_kernel,
                        "status": "regime_key_unmatched",
                        "target_regime": dict(target.regime_key),
                        "index_diagnostics": self._compute_index_diagnostics.get(diagnostics_key, {}),
                    }
                )
                continue
            override = {
                **self._kernel_overrides.get(kernel_type, {}),
                **self._kernel_overrides.get(candidate_kernel, {}),
            }
            for candidate_group in candidate_groups:
                for latency_column_group, candidate_subset in self._latency_column_pure_candidate_group_attempts(
                    candidate_group
                ):
                    result = candidate_subset.interpolate(
                        target.axes,
                        _COMPUTE_SCALE_AXIS_GROUPS,
                        fallback_from=fallback_from,
                        max_interpolation_dim=override.get("max_interpolation_dim"),
                        extra_details={
                            "kernel_type": candidate_kernel,
                            "query_mode": target.query_mode,
                            "compute_subcategory": _COMPUTE_SCALE_SUBCATEGORY,
                            "scale_mode": target.metadata["scale_mode"],
                            "block_size": target.metadata["block_size"],
                            "auxiliary_modes": target.metadata["auxiliary_modes"],
                            "latency_column_group": latency_column_group,
                        },
                    )
                    if result is None:
                        attempts.append(
                            {
                                "kernel_type": candidate_kernel,
                                "status": "candidate_group_failed",
                                "regime_key": dict(candidate_subset.regime_key),
                                "latency_column_group": latency_column_group,
                                "diagnostics": candidate_subset.last_diagnostics,
                            }
                        )
                        continue
                    result.details["interpolation_path"] = f"{_COMPUTE_SCALE_SUBCATEGORY}_{result.interpolation_dim}d"
                    return self._query_result_from_interpolation(target, result)

        final_diagnostics = attempts[-1].get("diagnostics") if attempts else None
        self._record_miss(
            self._candidate_failure_reason("compute_scale_interpolation_failed", final_diagnostics),
            compute_subcategory=_COMPUTE_SCALE_SUBCATEGORY,
            attempted_kernel_types=kernel_types,
            attempts=attempts,
        )
        return None

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

        runtime_mapping = self.base._build_composite_runtime_mapping(mapping)
        specs = decomposer(op_invoke_info, runtime_mapping)
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
            elif spec.query_mode == "mlapo_preprocess" and spec.runtime_params:
                hit = self.base._query_mlapo_preprocess(kernel_types, spec.runtime_params, spec.dtype)
                lat = hit.latency_us if hit is not None else None
                if hit is not None:
                    matched_kernel_type = hit.kernel_type
                    sub_detail.update(
                        {
                            "source": QuerySource.MEASURED.name,
                            "method": "exact_mlapo_preprocess",
                            "matched_kernel_type": matched_kernel_type,
                        }
                    )
            elif spec.query_mode == "scatter_cache_write" and spec.cache_params:
                hit = self.base._query_scatter_cache_write(kernel_types, spec.cache_params, spec.dtype)
                lat = hit.latency_us if hit is not None else None
                if hit is not None:
                    matched_kernel_type = hit.kernel_type
                    sub_detail.update(
                        {
                            "source": QuerySource.MEASURED.name,
                            "method": "exact_scatter_cache_write",
                            "matched_kernel_type": matched_kernel_type,
                        }
                    )
            elif spec.query_mode == "cache_postprocess" and spec.cache_params:
                hit = self.base._query_cache_postprocess(kernel_types, spec.cache_params, spec.dtype)
                lat = hit.latency_us if hit is not None else None
                if hit is not None:
                    matched_kernel_type = hit.kernel_type
                    sub_detail.update(
                        {
                            "source": QuerySource.MEASURED.name,
                            "method": "exact_cache_postprocess",
                            "matched_kernel_type": matched_kernel_type,
                        }
                    )
            else:
                profile_dtypes = spec.input_dtypes or [spec.dtype] * len(spec.input_shapes)
                torch_dtypes = [next((k for k, v in DTYPE_MAP.items() if v == dtype), None) for dtype in profile_dtypes]
                if len(torch_dtypes) == len(spec.input_shapes) and all(dtype is not None for dtype in torch_dtypes):
                    tc_inputs = list(zip(spec.input_shapes, torch_dtypes))
                    hit = self.base._find_compute_match(
                        kernel_types,
                        tc_inputs,
                        spec.tc_input_count,
                        auto_truncate=True,
                    )
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
                elif spec.query_mode == "scatter_cache_write":
                    input_dtypes = spec.input_dtypes or [spec.dtype for _ in spec.input_shapes]
                    result_interp = self._interpolate_scatter_nd_update_by_shapes(
                        kernel_types,
                        spec.input_shapes,
                        input_dtypes,
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
        if kernel_type in _RUNTIME_ATTENTION_KERNELS:
            return self._interpolate_runtime_attention_by_params_one(
                kernel_type,
                params,
                dtype_str,
                include_sparse_fields=False,
            )
        if kernel_type in _SPARSE_RUNTIME_ATTENTION_KERNELS:
            return self._interpolate_runtime_attention_by_params_one(
                kernel_type,
                params,
                dtype_str,
                include_sparse_fields=True,
            )
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

    def _interpolate_runtime_attention_by_params_one(
        self,
        kernel_type: str,
        params: Dict[str, Any],
        dtype_str: str,
        *,
        include_sparse_fields: bool,
    ) -> Optional[QueryResult]:
        if include_sparse_fields:
            index, rejected_reasons = self._get_sparse_attention_index(kernel_type)
            axis_groups = _SPARSE_ATTENTION_AXIS_GROUPS
            interpolation_path = "sparse_attention"
        else:
            index, rejected_reasons = self._get_lightning_indexer_index(kernel_type)
            axis_groups = _LIGHTNING_INDEXER_AXIS_GROUPS
            interpolation_path = "lightning_indexer"
        if index is None:
            self._record_miss(
                "csv_not_found",
                kernel_type=kernel_type,
                interpolation_path=interpolation_path,
            )
            return None
        target = self._build_runtime_attention_target_from_params(
            kernel_type,
            params,
            dtype_str,
            include_sparse_fields=include_sparse_fields,
        )
        if target is None:
            self._record_miss(
                f"{interpolation_path}_target_unextractable",
                kernel_type=kernel_type,
                rejected_reasons=rejected_reasons,
            )
            return None

        attempts: list[dict[str, Any]] = []
        candidate_groups = index.candidate_groups_matching(target.regime_key)
        if not candidate_groups:
            attempts.append({"status": "regime_key_unmatched", "target_regime": dict(target.regime_key)})
        override = self._kernel_overrides.get(kernel_type, {})
        for candidate_group in candidate_groups:
            for latency_column_group, candidate_subset in self._latency_column_pure_candidate_group_attempts(
                candidate_group
            ):
                result = candidate_subset.interpolate(
                    target.axes,
                    axis_groups,
                    max_interpolation_dim=override.get("max_interpolation_dim"),
                    fallback_from="composite",
                    extra_details={
                        "kernel_type": kernel_type,
                        "query_mode": target.query_mode,
                        "interpolation_path": interpolation_path,
                        "phase": target.metadata["phase"],
                        "latency_column_group": latency_column_group,
                        "rejected_reasons": rejected_reasons,
                    },
                )
                if result is not None:
                    return self._query_result_from_interpolation(target, result)
                attempts.append(
                    {
                        "status": "candidate_group_failed",
                        "regime_key": dict(candidate_subset.regime_key),
                        "latency_column_group": latency_column_group,
                        "diagnostics": candidate_subset.last_diagnostics,
                    }
                )
        self._record_miss(
            self._candidate_failure_reason(
                f"{interpolation_path}_interpolation_failed",
                attempts[-1].get("diagnostics") if attempts else {},
            ),
            kernel_type=kernel_type,
            query_mode=target.query_mode,
            candidate_count=len(index.points),
            rejected_reasons=rejected_reasons,
            attempts=attempts,
            target_axes=target.axes,
        )
        return None

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
        tc_dtype_str: str,
    ) -> Optional[CandidatePoint]:
        csv_input_shapes = _parse_shape_str(str(row.get("Input Shapes", "")))
        csv_out_shapes = _parse_shape_str(str(row.get("Output Shapes", "")))
        csv_out_dtypes = _parse_str_list(str(row.get("Output Data Types", "")))
        if not csv_out_shapes:
            return None

        csv_shape = _strip_batch_dim(tuple(csv_out_shapes[0]))
        if not csv_shape:
            return None

        latency, latency_meta = self._candidate_latency(row, latency_col)
        if latency is None:
            return None

        csv_dtype_str = csv_out_dtypes[0] if csv_out_dtypes else None
        if not csv_dtype_str or not tc_dtype_str or csv_dtype_str != tc_dtype_str:
            return None

        input_shapes = [_strip_batch_dim(tuple(shape)) for shape in csv_input_shapes]
        axes = self._elementwise_axes_from_shapes(input_shapes, csv_shape)
        if axes is None:
            return None
        broadcast_pattern = self._elementwise_broadcast_pattern(csv_shape, input_shapes)
        regime_key = make_regime_key(
            [
                ("kernel_type", kernel_type),
                ("query_mode", "elementwise"),
                ("output_rank", len(csv_shape)),
                ("input_count", len(input_shapes)),
                ("output_dtype", csv_dtype_str),
                ("broadcast_pattern", broadcast_pattern),
            ]
        )
        return CandidatePoint(
            kernel_type=kernel_type,
            axes=axes,
            latency_us=latency,
            regime_key=regime_key,
            input_shapes=input_shapes or [csv_shape],
            input_dtypes=[csv_dtype_str],
            row_index=row_index,
            row_meta={
                **latency_meta,
                "csv_output_dtype": csv_dtype_str,
                "target_output_dtype": tc_dtype_str,
                "output_shape": csv_shape,
                "broadcast_pattern": broadcast_pattern,
                "elementwise_axes": dict(axes),
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
        index, _rejected_reasons = self._build_candidate_index(
            df,
            lambda row, row_index: (
                self._candidate_from_elementwise_row(row, kernel_type, latency_col, row_index, tc_dtype_str),
                None,
            ),
        )
        self._elementwise_index_cache[cache_key] = index
        return index

    def _interpolate_elementwise(
        self,
        op_invoke_info: "OpInvokeInfo",
        mapping: dict,
        *,
        fallback_from: str = "exact_miss",
    ) -> Optional[QueryResult]:
        """Interpolate elementwise ops using guarded 1D total-I/O work."""
        kernel_type = mapping.get("kernel_type")
        if not kernel_type:
            return None
        kernel_types = [kernel_type]
        for alt in mapping.get("alternate_kernel_types", []):
            if alt not in kernel_types:
                kernel_types.append(alt)

        attempts: list[dict[str, Any]] = []
        for kt in kernel_types:
            result = self._interpolate_elementwise_kernel(
                op_invoke_info,
                mapping,
                kt,
                fallback_from=fallback_from,
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

        if len(attempts) > 1:
            self._record_miss(
                "elementwise_interpolation_failed",
                attempted_kernel_types=kernel_types,
                attempts=attempts,
            )
        return None

    def _interpolate_elementwise_kernel(
        self,
        op_invoke_info: "OpInvokeInfo",
        mapping: dict,
        kernel_type: str,
        *,
        fallback_from: str,
    ) -> Optional[QueryResult]:
        """Interpolate one concrete elementwise kernel type."""
        override = self._kernel_overrides.get(kernel_type, {})

        out = self._first_output_tensor(op_invoke_info)
        if out is None:
            self._record_miss("elementwise_output_unavailable", kernel_type=kernel_type)
            return None

        output_shape = _strip_batch_dim(tuple(out.shape))
        if len(output_shape) < 1:
            self._record_miss("elementwise_output_shape_unavailable", kernel_type=kernel_type)
            return None
        tc_dtype_str = DTYPE_MAP.get(out.dtype)
        tc_inputs = self.base._extract_tensor_inputs(op_invoke_info)
        input_shapes = [_strip_batch_dim(tuple(shape)) for shape, _ in tc_inputs]
        target_axes = self._elementwise_axes_from_shapes(input_shapes, output_shape)
        if target_axes is None:
            self._record_miss("elementwise_shape_invalid", kernel_type=kernel_type)
            return None
        broadcast_pattern = self._elementwise_broadcast_pattern(output_shape, input_shapes)

        target = InterpolationTarget(
            func_name=_normalize_func_name(op_invoke_info.func),
            kernel_type=kernel_type,
            axes=target_axes,
            regime_key=make_regime_key(
                [
                    ("kernel_type", kernel_type),
                    ("query_mode", "elementwise"),
                    ("output_rank", len(output_shape)),
                    ("input_count", len(input_shapes)),
                    ("output_dtype", tc_dtype_str or ""),
                    ("broadcast_pattern", broadcast_pattern),
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

        candidate_groups = index.candidate_groups_matching(target.regime_key)
        candidate_count = sum(len(group.points) for group in candidate_groups)
        if not candidate_groups:
            self._record_miss(
                "insufficient_filtered_candidates",
                kernel_type=kernel_type,
                query_mode="elementwise",
                interpolation_path="elementwise_1d",
                target=float(target_axes.get("io_numel", 0.0)),
                target_axes=target_axes,
                target_regime_key=dict(target.regime_key),
                candidate_count=candidate_count,
            )
            return None

        attempts: list[dict[str, Any]] = []
        for candidate_group in candidate_groups:
            for latency_column_group, candidate_subset in self._latency_column_pure_candidate_group_attempts(
                candidate_group
            ):
                result = candidate_subset.interpolate(
                    target.axes,
                    _ELEMENTWISE_AXIS_GROUPS,
                    max_interpolation_dim=override.get("max_interpolation_dim"),
                    fallback_from=fallback_from,
                    extra_details={
                        "kernel_type": kernel_type,
                        "query_mode": "elementwise",
                        "interpolation_path": "elementwise_1d",
                        "latency_column_group": latency_column_group,
                        "csv_output_dtype": dict(candidate_subset.regime_key).get("output_dtype"),
                        "target_elementwise_axes": dict(target.axes),
                    },
                )
                if result is None:
                    attempts.append(
                        {
                            "regime_key": dict(candidate_subset.regime_key),
                            "latency_column_group": latency_column_group,
                            "diagnostics": candidate_subset.last_diagnostics,
                        }
                    )
                    continue
                details = dict(result.details)
                details["dtype_scaled"] = False
                details["latency_column_group"] = latency_column_group
                result = replace(result, details=details)
                return self._query_result_from_interpolation(target, result)

        self._record_miss(
            self._candidate_failure_reason(
                "candidate_group_failed", attempts[-1].get("diagnostics") if attempts else {}
            ),
            kernel_type=kernel_type,
            query_mode="elementwise",
            interpolation_path="elementwise_1d",
            target=float(target_axes.get("io_numel", 0.0)),
            target_axes=target_axes,
            target_regime_key=dict(target.regime_key),
            candidate_count=candidate_count,
            attempts=attempts,
        )
        return None
