"""InterpolatingDataSource: wraps ProfilingDataSource with 1D linear interpolation.

When exact lookup misses, tries interpolation on the varying dimension:
- Compute ops: interpolate on first dim of first input (seq_len/num_tokens)
- Comm ops: interpolate on message_bytes
- Attention ops: interpolate on avg_seq_len (with optional sqrt transform)
- Composite ops: decompose into sub-kernels, interpolate each, sum
"""

import logging
import math
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import torch

from .data_source import DataSourcePerformanceModel, QueryResult, QuerySource
from .profiling_data_source import (
    _dtype_byte_size,
    _FLATTEN_BATCH_KERNELS,
    _infer_sparse_mode,
    _is_block_padded,
    _MATMUL_KERNELS,
    _normalize_fia_q_shape,
    _normalize_func_name,
    _parse_fia_q_shape,
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


def _interp_1d(x0: float, y0: float, x1: float, y1: float, target_x: float) -> float:
    """Linear interpolation between two points. Clamps to non-negative."""
    if x1 == x0:
        return y0
    t = (target_x - x0) / (x1 - x0)
    return max(0.0, y0 + t * (y1 - y0))


def _find_bracket(values: List[float], target: float) -> Optional[Tuple[float, float]]:
    """Find (left, right) values that bracket target. Returns None if can't bracket."""
    below = [v for v in values if v <= target]
    above = [v for v in values if v >= target]
    if not below or not above:
        return None  # Can't interpolate, would need extrapolation
    return (max(below), min(above))


class InterpolatingDataSource(DataSourcePerformanceModel):
    """Wrapper datasource that adds 1D linear interpolation fallback.

    When the base ProfilingDataSource returns None (exact miss), this layer
    attempts interpolation by finding bracketing data points and linearly
    interpolating on the varying dimension.
    """

    def __init__(self, base: ProfilingDataSource):
        self.base = base
        # Read kernel_overrides from op_mapping for sqrt transform etc.
        ip = self.base._op_mapping.get("interpolation_policy", {})
        self._kernel_overrides = ip.get("kernel_overrides", {})

    @property
    def last_miss_reason(self) -> str:
        return self.base.last_miss_reason

    def lookup(self, op_invoke_info: "OpInvokeInfo") -> Optional[QueryResult]:
        result = self.base.lookup(op_invoke_info)
        if result is not None and result.source != QuerySource.PARTIAL:
            return result
        # PARTIAL or None — try interpolation
        interp_result = self._interpolate(op_invoke_info)
        if interp_result is not None:
            # Mark as interpolated so runtime writes shape_match_rule="interpolated"
            if interp_result.shape_match_info is None:
                from .data_source import ShapeMatchInfo

                interp_result.shape_match_info = ShapeMatchInfo(
                    simulation_shapes=[],
                    kernel_shapes=[],
                    shape_match_rule="interpolated",
                )
            return interp_result
        # Fall back to PARTIAL (if available) or None
        return result

    def _interpolate(self, op_invoke_info: "OpInvokeInfo") -> Optional[QueryResult]:
        """Determine which query path to use and dispatch to the right interpolator."""
        func_str = _normalize_func_name(op_invoke_info.func)
        mappings = self.base._op_mapping.get("operator_mappings", {})
        mapping = mappings.get(func_str)
        if mapping is None:
            return None

        # Don't interpolate zero_cost or accepted_miss ops
        if mapping.get("zero_cost") or mapping.get("accepted_miss"):
            return None

        # Composite ops: decompose into sub-kernels, interpolate each
        if mapping.get("composite"):
            return self._interpolate_composite(op_invoke_info, mapping, func_str)

        if mapping.get("category") == "communication":
            # Comm interpolation handled by base's _query_comm_csv alpha-beta model
            return None
        if mapping.get("query_mode") == "attention_special":
            return self._interpolate_attention(op_invoke_info, mapping)
        if mapping.get("query_mode") == "elementwise":
            return self._interpolate_elementwise(op_invoke_info, mapping)
        return self._interpolate_compute(op_invoke_info, mapping)

    # ---- Compute interpolation ----

    def _interpolate_compute(
        self, op_invoke_info: "OpInvokeInfo", mapping: dict
    ) -> Optional[QueryResult]:
        """Interpolate compute ops on the first dimension of the first input.

        Strategy: find all CSV rows where dtype matches and all dimensions
        match EXCEPT the first dim of the first input. Collect
        (first_dim_value, duration) pairs, bracket the target, and interpolate.
        """
        kernel_type = mapping.get("kernel_type")
        if not kernel_type:
            return None

        df = self.base._load_csv(kernel_type)
        if df is None:
            return None

        tc_inputs = self.base._extract_tensor_inputs(op_invoke_info)
        if not tc_inputs:
            return None

        # Respect tc_input_count truncation (Issue #8: match base lookup behavior)
        tc_input_count = mapping.get("tc_input_count")
        if tc_input_count is not None:
            tc_inputs = tc_inputs[:tc_input_count]

        # Target: first dim of first input after batch strip (typically seq_len)
        target_dim = _strip_batch_dim(tc_inputs[0][0])[0]

        latency_col = self.base._latency_col(df)

        # Find CSV rows where all dims/dtypes match except first dim of first input
        candidates: List[Tuple[float, float]] = []  # (first_dim_value, duration)

        for _, row in df.iterrows():
            csv_shapes = _parse_shape_str(str(row.get("Input Shapes", "")))
            csv_dtypes = _parse_str_list(str(row.get("Input Data Types", "")))
            csv_formats = _parse_str_list(str(row.get("Input Formats", "")))

            if len(csv_shapes) != len(tc_inputs):
                continue

            # Check all dtypes match
            dtype_match = True
            for i, (_, tc_dtype) in enumerate(tc_inputs):
                expected = DTYPE_MAP.get(tc_dtype)
                if i >= len(csv_dtypes) or expected != csv_dtypes[i]:
                    dtype_match = False
                    break
            if not dtype_match:
                continue

            # Check all dims match except first dim of first input
            # Apply shape transforms matching ProfilingDataSource logic:
            # FRACTAL_NZ→ND, batch dim strip, ND transpose, flatten batch
            all_match = True
            for i, (tc_shape, _) in enumerate(tc_inputs):
                csv_shape = csv_shapes[i]
                fmt = csv_formats[i] if i < len(csv_formats) else "ND"
                if fmt == "FRACTAL_NZ":
                    csv_shape = fractal_nz_to_nd(csv_shape)

                tc_s = _strip_batch_dim(tc_shape)
                csv_s = _strip_batch_dim(csv_shape)

                if i == 0:
                    # First input: skip first dim, check rest match
                    if len(tc_s) != len(csv_s):
                        all_match = False
                        break
                    if tc_s[1:] != csv_s[1:]:
                        all_match = False
                        break
                else:
                    # Other inputs: all dims must match exactly
                    if tc_s == csv_s:
                        continue
                    # ND transpose for matmul weights
                    if (
                        kernel_type in _MATMUL_KERNELS
                        and len(tc_s) == 2
                        and len(csv_s) == 2
                        and tc_s == (csv_s[1], csv_s[0])
                    ):
                        continue
                    # Flatten batch: TC (B,M,D) → CSV (B*M,D)
                    if (
                        kernel_type in _FLATTEN_BATCH_KERNELS
                        and len(csv_s) == 2
                        and len(tc_s) == 3
                        and (tc_s[0] * tc_s[1], tc_s[2]) == csv_s
                    ):
                        continue
                    all_match = False
                    break
            if all_match:
                # Use stripped first dim for candidate value
                first_csv_shape = csv_shapes[0]
                fmt0 = csv_formats[0] if csv_formats else "ND"
                if fmt0 == "FRACTAL_NZ":
                    first_csv_shape = fractal_nz_to_nd(first_csv_shape)
                first_csv_shape = _strip_batch_dim(first_csv_shape)
                candidates.append((float(first_csv_shape[0]), float(row[latency_col])))

        if len(candidates) < 2:
            return None

        candidates.sort(key=lambda x: x[0])
        return self._interpolate_from_candidates(
            candidates, float(target_dim), kernel_type
        )

    # ---- Communication interpolation ----

    # Communication interpolation is handled by ProfilingDataSource._query_comm_csv
    # which has built-in alpha-beta least-squares interpolation. If base.lookup()
    # returns None for a comm op, there's no data to interpolate against.

    # ---- Attention interpolation ----

    def _interpolate_attention(
        self, op_invoke_info: "OpInvokeInfo", mapping: dict
    ) -> Optional[QueryResult]:
        """Interpolate attention ops on avg_seq_len using enriched CSV.

        Filters by (N, D, dtype, sparse_mode, num_kv_heads) exact match on
        normalized Q shape from slot 0, collects (avg_seq_len, latency)
        candidates, applies sqrt transform if kernel_overrides specifies it.
        """
        kernel_type = mapping.get("kernel_type")
        if not kernel_type:
            return None

        df = self.base._load_csv(kernel_type)
        if df is None:
            return None

        # Require enriched CSV format — unified column naming
        avg_seq_col = None
        if "Runtime avg_seq_len" in df.columns:
            avg_seq_col = "Runtime avg_seq_len"
        elif "avg_seq_len" in df.columns:
            avg_seq_col = "avg_seq_len"
        else:
            return None
        if "Input Shapes" not in df.columns:
            return None

        args = op_invoke_info.args
        if len(args) < 7:
            return None

        query = args[0]
        key = args[1]
        seq_lens = args[6]
        query_lens = args[7] if len(args) > 7 else None
        if not isinstance(query, torch.Tensor) or not isinstance(
            seq_lens, torch.Tensor
        ):
            return None

        head_dim = (
            key.shape[-1] if isinstance(key, torch.Tensor) and key.ndim >= 1 else 0
        )
        tc_q_3d = _normalize_fia_q_shape(tuple(query.shape), head_dim)
        if tc_q_3d is None:
            return None
        tc_N, tc_D = tc_q_3d[1], tc_q_3d[2]

        try:
            avg_seq_len = int(seq_lens.float().mean().item())
        except Exception:
            return None

        dtype_str = DTYPE_MAP.get(query.dtype)
        if dtype_str is None:
            return None

        # Infer sparse_mode and num_kv_heads from TC args
        tc_sparse_mode = _infer_sparse_mode(query_lens)
        tc_num_kv_heads = (
            key.shape[-2] if isinstance(key, torch.Tensor) and key.ndim >= 2 else None
        )

        has_sparse_col = "Runtime sparse_mode" in df.columns
        has_kv_heads_col = "Runtime num_key_value_heads" in df.columns

        latency_col = self.base._latency_col(df)

        # Collect candidates: filter by (N, D, dtype, sparse_mode, kv_heads),
        # vary avg_seq_len
        candidates: List[Tuple[float, float]] = []
        for _, row in df.iterrows():
            csv_avg_seq = int(row[avg_seq_col])
            if csv_avg_seq < 0:
                continue

            shapes_str = str(row.get("Input Shapes", "")).strip('"')
            csv_q_raw = _parse_fia_q_shape(shapes_str)
            if csv_q_raw is None:
                continue
            csv_q_3d = _normalize_fia_q_shape(csv_q_raw, head_dim)
            if csv_q_3d is None:
                continue

            csv_dtypes_str = str(row.get("Input Data Types", ""))
            csv_first_dtype = (
                csv_dtypes_str.split(";")[0].strip() if csv_dtypes_str else ""
            )
            if dtype_str != csv_first_dtype:
                continue

            if tc_N != csv_q_3d[1] or tc_D != csv_q_3d[2]:
                continue

            # T (token count) filter: must match exactly or within block-padding
            csv_T = csv_q_3d[0]
            tc_T = tc_q_3d[0]
            if (
                tc_T != csv_T
                and not _is_block_padded(tc_T, csv_T)
                and not _is_block_padded(csv_T, tc_T)
            ):
                continue

            # sparse_mode filter (skip if CSV lacks column)
            if has_sparse_col and tc_sparse_mode != int(row["Runtime sparse_mode"]):
                continue

            # num_kv_heads filter (skip if CSV lacks column)
            if (
                has_kv_heads_col
                and tc_num_kv_heads is not None
                and tc_num_kv_heads != int(row["Runtime num_key_value_heads"])
            ):
                continue

            candidates.append((float(csv_avg_seq), float(row[latency_col])))

        if len(candidates) < 2:
            return None

        candidates.sort(key=lambda x: x[0])

        override = self._kernel_overrides.get(kernel_type, {})
        transform = override.get("shape_transform")

        if transform == "sqrt":
            return self._interpolate_from_candidates_sqrt(
                candidates, float(avg_seq_len), kernel_type
            )
        return self._interpolate_from_candidates(
            candidates, float(avg_seq_len), kernel_type
        )

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

        for spec in specs:
            lat = None

            # First try exact match via base ProfilingDataSource
            kernel_types = [spec.kernel_type] + (spec.alternate_kernel_types or [])
            if spec.query_mode == "attention" and spec.attention_params:
                result_exact = self.base._query_by_attn_params(
                    kernel_types, spec.attention_params, spec.dtype
                )
                lat = result_exact[0] if result_exact else None
            else:
                torch_dtype = None
                for k, v in DTYPE_MAP.items():
                    if v == spec.dtype:
                        torch_dtype = k
                        break
                if torch_dtype is not None:
                    tc_inputs = [(shape, torch_dtype) for shape in spec.input_shapes]
                    hit = self.base._find_compute_match(
                        kernel_types, tc_inputs, spec.tc_input_count
                    )
                    lat = hit.latency_us if hit else None
                else:
                    lat = None

            # If exact miss, try interpolation
            if lat is None:
                if spec.query_mode == "attention" and spec.attention_params:
                    lat = self._interpolate_attention_by_params(
                        spec.kernel_type, spec.attention_params, spec.dtype
                    )
                else:
                    lat = self._interpolate_compute_by_shapes(
                        spec.kernel_type, spec.input_shapes, spec.dtype
                    )

            if lat is None:
                return None

            total_latency += lat
            hit_kernels.append(spec.kernel_type)

        logger.debug(
            "INTERPOLATED (composite) %s: sub_kernels=%s, total=%.1f us",
            func_str,
            hit_kernels,
            total_latency,
        )
        return QueryResult(
            latency_us=total_latency,
            confidence=0.5,
            source=QuerySource.INTERPOLATED,
            details={
                "kernel_type": ",".join(hit_kernels),
                "composite": True,
                "method": "decomposed_interpolation",
            },
        )

    def _interpolate_compute_by_shapes(
        self,
        kernel_type: str,
        input_shapes: List[Tuple[int, ...]],
        dtype_str: str,
    ) -> Optional[float]:
        """Interpolate a compute sub-kernel by explicit shapes.

        Same logic as _interpolate_compute but takes shapes directly
        instead of extracting from OpInvokeInfo.
        """
        df = self.base._load_csv(kernel_type)
        if df is None:
            return None

        if not input_shapes:
            return None

        target_dim = float(input_shapes[0][0])

        latency_col = self.base._latency_col(df)

        # Find CSV rows where all dims/dtypes match except first dim of first input
        candidates: List[Tuple[float, float]] = []

        for _, row in df.iterrows():
            csv_shapes = _parse_shape_str(str(row.get("Input Shapes", "")))
            csv_dtypes = _parse_str_list(str(row.get("Input Data Types", "")))

            if len(csv_shapes) != len(input_shapes):
                continue

            # Check dtype of first input
            if not csv_dtypes or csv_dtypes[0] != dtype_str:
                continue

            # Check all dims match except first dim of first input
            all_match = True
            for i, shape in enumerate(input_shapes):
                csv_shape = csv_shapes[i]
                if i == 0:
                    if len(shape) != len(csv_shape):
                        all_match = False
                        break
                    if shape[1:] != csv_shape[1:]:
                        all_match = False
                        break
                else:
                    if shape != csv_shape:
                        all_match = False
                        break
            if all_match:
                candidates.append((float(csv_shapes[0][0]), float(row[latency_col])))

        if len(candidates) < 2:
            return None

        candidates.sort(key=lambda x: x[0])
        result = self._interpolate_from_candidates(candidates, target_dim, kernel_type)
        return result.latency_us if result else None

    def _interpolate_attention_by_params(
        self,
        kernel_type: str,
        params: Dict,
        dtype_str: str,
    ) -> Optional[float]:
        """Interpolate attention sub-kernel using enriched CSV by explicit params.

        params: {q_shape_3d, avg_seq_len, sparse_mode?, num_kv_heads?}
        """
        df = self.base._load_csv(kernel_type)
        if df is None:
            return None

        # Unified column naming
        avg_seq_col = None
        if "Runtime avg_seq_len" in df.columns:
            avg_seq_col = "Runtime avg_seq_len"
        elif "avg_seq_len" in df.columns:
            avg_seq_col = "avg_seq_len"
        else:
            return None
        if "Input Shapes" not in df.columns:
            return None

        q_shape_3d = params.get("q_shape_3d")
        target_avg_seq = params.get("avg_seq_len")
        if q_shape_3d is None or target_avg_seq is None:
            return None

        target_sparse = params.get("sparse_mode")
        target_kv_heads = params.get("num_kv_heads")
        has_sparse_col = "Runtime sparse_mode" in df.columns
        has_kv_heads_col = "Runtime num_key_value_heads" in df.columns

        tc_N, tc_D = q_shape_3d[1], q_shape_3d[2]
        head_dim = tc_D
        latency_col = self.base._latency_col(df)

        candidates: List[Tuple[float, float]] = []
        for _, row in df.iterrows():
            csv_avg_seq = int(row[avg_seq_col])
            if csv_avg_seq < 0:
                continue

            shapes_str = str(row.get("Input Shapes", "")).strip('"')
            csv_q_raw = _parse_fia_q_shape(shapes_str)
            if csv_q_raw is None:
                continue
            csv_q_3d = _normalize_fia_q_shape(csv_q_raw, head_dim)
            if csv_q_3d is None:
                continue

            csv_dtypes_str = str(row.get("Input Data Types", ""))
            csv_first_dtype = (
                csv_dtypes_str.split(";")[0].strip() if csv_dtypes_str else ""
            )
            if dtype_str != csv_first_dtype:
                continue

            if tc_N != csv_q_3d[1] or tc_D != csv_q_3d[2]:
                continue

            # T (token count) filter: must match exactly or within block-padding
            csv_T = csv_q_3d[0]
            tc_T = q_shape_3d[0]
            if (
                tc_T != csv_T
                and not _is_block_padded(tc_T, csv_T)
                and not _is_block_padded(csv_T, tc_T)
            ):
                continue

            # sparse_mode filter
            if (
                has_sparse_col
                and target_sparse is not None
                and int(row["Runtime sparse_mode"]) != target_sparse
            ):
                continue

            # num_kv_heads filter
            if (
                has_kv_heads_col
                and target_kv_heads is not None
                and int(row["Runtime num_key_value_heads"]) != target_kv_heads
            ):
                continue

            candidates.append((float(csv_avg_seq), float(row[latency_col])))

        if len(candidates) < 2:
            return None

        candidates.sort(key=lambda x: x[0])
        target = float(target_avg_seq)

        override = self._kernel_overrides.get(kernel_type, {})
        transform = override.get("shape_transform")

        if transform == "sqrt":
            result = self._interpolate_from_candidates_sqrt(
                candidates, target, kernel_type
            )
        else:
            result = self._interpolate_from_candidates(candidates, target, kernel_type)
        return result.latency_us if result else None

    # ---- Elementwise interpolation ----

    def _interpolate_elementwise(
        self, op_invoke_info: "OpInvokeInfo", mapping: dict
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

        latency_col = self.base._latency_col(df)
        has_dtype_scaling = False

        candidates: List[Tuple[float, float]] = []
        for _, row in df.iterrows():
            csv_out_shapes = _parse_shape_str(str(row.get("Output Shapes", "")))
            csv_out_dtypes = _parse_str_list(str(row.get("Output Data Types", "")))
            if not csv_out_shapes:
                continue

            csv_shape = _strip_batch_dim(tuple(csv_out_shapes[0]))

            # Hidden dims must match (everything except first dim)
            if len(csv_shape) != len(output_shape) or csv_shape[1:] != output_shape[1:]:
                continue

            # Compute byte-ratio scaled latency
            latency = float(row[latency_col])
            csv_dtype_str = csv_out_dtypes[0] if csv_out_dtypes else None
            if csv_dtype_str and tc_dtype_str and csv_dtype_str != tc_dtype_str:
                tc_bytes = _dtype_byte_size(tc_dtype_str)
                csv_bytes = _dtype_byte_size(csv_dtype_str)
                if tc_bytes > 0 and csv_bytes > 0:
                    latency *= tc_bytes / csv_bytes
                    has_dtype_scaling = True

            candidates.append((float(csv_shape[0]), latency))

        if len(candidates) < 2:
            return None

        candidates.sort(key=lambda x: x[0])
        # Confidence: 0.6 if dtype-scaled (combining dtype approximation + interpolation),
        # 0.7 if same dtype (standard interpolation confidence).
        result = self._interpolate_from_candidates(candidates, target_dim, kernel_type)
        if result is not None and has_dtype_scaling:
            result = QueryResult(
                latency_us=result.latency_us,
                confidence=0.6,
                source=result.source,
                details={**result.details, "dtype_scaled": True},
            )
        return result

    # ---- Shared interpolation helpers ----

    def _interpolate_from_candidates(
        self,
        candidates: List[Tuple[float, float]],
        target: float,
        kernel_type: str,
    ) -> Optional[QueryResult]:
        """Find bracket and linearly interpolate from sorted candidates."""
        xs = [c[0] for c in candidates]
        bracket = _find_bracket(xs, target)
        if bracket is None:
            return None

        x_lo, x_hi = bracket

        # Find the duration values for the bracket bounds
        y_lo = next(y for x, y in candidates if x == x_lo)
        y_hi = next(y for x, y in candidates if x == x_hi)

        # If exact match was found by bracket (x_lo == x_hi == target),
        # that should have been caught by base lookup. But handle gracefully.
        latency = _interp_1d(x_lo, y_lo, x_hi, y_hi, target)

        logger.debug(
            "INTERPOLATED %s: target=%.1f, bracket=(%.1f, %.1f), "
            "durations=(%.1f, %.1f) -> %.1f us",
            kernel_type,
            target,
            x_lo,
            x_hi,
            y_lo,
            y_hi,
            latency,
        )
        return QueryResult(
            latency_us=latency,
            confidence=0.7,
            source=QuerySource.INTERPOLATED,
            details={
                "kernel_type": kernel_type,
                "method": "linear_1d",
                "bracket": (x_lo, x_hi),
            },
        )

    def _interpolate_from_candidates_sqrt(
        self,
        candidates: List[Tuple[float, float]],
        target: float,
        kernel_type: str,
    ) -> Optional[QueryResult]:
        """Interpolate with sqrt transform on the x-axis.

        For O(n^2) attention ops, sqrt-transform the interpolation dimension
        before linear interpolation. This better captures the quadratic
        relationship between seq_len and latency.
        """
        xs = [c[0] for c in candidates]
        bracket = _find_bracket(xs, target)
        if bracket is None:
            return None

        x_lo, x_hi = bracket
        y_lo = next(y for x, y in candidates if x == x_lo)
        y_hi = next(y for x, y in candidates if x == x_hi)

        # Transform to sqrt space for interpolation
        sqrt_lo = math.sqrt(x_lo)
        sqrt_hi = math.sqrt(x_hi)
        sqrt_target = math.sqrt(target)

        latency = _interp_1d(sqrt_lo, y_lo, sqrt_hi, y_hi, sqrt_target)

        logger.debug(
            "INTERPOLATED (sqrt) %s: target=%.1f (sqrt=%.2f), "
            "bracket=(%.1f, %.1f), durations=(%.1f, %.1f) -> %.1f us",
            kernel_type,
            target,
            sqrt_target,
            x_lo,
            x_hi,
            y_lo,
            y_hi,
            latency,
        )
        return QueryResult(
            latency_us=latency,
            confidence=0.6,  # Lower confidence for transformed interpolation
            source=QuerySource.INTERPOLATED,
            details={
                "kernel_type": kernel_type,
                "method": "linear_1d_sqrt",
                "bracket": (x_lo, x_hi),
            },
        )
