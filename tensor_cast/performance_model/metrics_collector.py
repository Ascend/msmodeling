"""MetricsCollector: M1-M5 metrics collection for EmpiricalPerformanceModel.

Decoupled from EmpiricalPerformanceModel — reads op_records exposed by the
perf model rather than being called from inside process_op().

Usage::

    with Runtime(...) as runtime:
        model.forward(...)

    for pm in perf_models:
        if isinstance(pm, EmpiricalPerformanceModel):
            collector = MetricsCollector()
            collector.collect_from_records(pm.op_records)
            collector.log_stats()
            collector.export_hit_miss_report(output_path)
"""

import json
import logging
from collections import Counter
from pathlib import Path
from typing import List, Optional

from .empirical import EmpiricalOpRecord
from .profiling_database.data_source import QueryResult, QuerySource

# Human-readable descriptions for miss reason codes
_MISS_REASON_LABELS = {
    "unmapped": "not in op_mapping.yaml",
    "shape_mismatch": "kernel found, no matching shape in CSV",
    "input_count_mismatch": "TC input count differs from CSV",
    "csv_format_raw": "CSV has raw profiling format (needs microbenchmark)",
    "csv_not_found": "kernel CSV file missing",
    "no_sub_kernels": "composite op has no sub_kernels defined",
    "invalid_args": "op args could not be parsed",
}


# Default fused op groups — maps NPU fusion name to constituent TC op prefixes
DEFAULT_FUSED_GROUPS = {
    "DispatchFFNCombine": [
        "tensor_cast.init_routing_v2",
        "tensor_cast.grouped_matmul",  # prefix covers all variants
        "tensor_cast.unpermute_tokens",
        "tensor_cast.all_to_all",
    ],
    "MLAPO": [
        "tensor_cast.mlapo",
        "tensor_cast.mlapo_quant",
    ],
    "MLA": [
        "tensor_cast.multihead_latent_attention",
    ],
    "MC2": [
        "tensor_cast.matmul_all_reduce",
        "tensor_cast.static_quant_linear_all_reduce",
        "tensor_cast.fp8_linear_all_reduce",
    ],
}


def compute_fused_op_stats(
    hit_details: list[tuple[str, str, tuple, float]],
    miss_details: list[tuple[str, str, list, ...]],
    fused_groups: dict[str, list[str]] | None = None,
) -> dict:
    """Compute Fused Op Match Rate with pessimistic grouping.

    Phase 1 metrics (M1-M3):
    - M1 (Raw Op-Count HR): reported separately by EmpiricalPerformanceModel
    - M2 (Fused Op HR): per unique func_name, pessimistic rule, with fused grouping
    - M3 (Fused Op HR w/o zc): same as M2 excluding zero_cost ops

    Pessimistic rule: if an op appears in BOTH hits and misses (different
    shapes), it counts as MISS. An op is HIT only if ALL its invocations HIT.

    Fused grouping: DFC/MLAPO/MLA/MC2 constituent ops collapse to 1 fused op.
    A fused group is HIT only if ALL members are HIT and NONE MISS.

    Args:
        hit_details: list of (func_name, kernel_type, shape_sig, latency_s) tuples
        miss_details: list of (func_name, reason, shapes) tuples
        fused_groups: map of group_name -> list of TC op prefixes to group

    Returns:
        dict with fused_hit, fused_miss, fused_total, fused_hr,
        _no_zc variants, and per_shape stats.
    """
    if fused_groups is None:
        fused_groups = DEFAULT_FUSED_GROUPS

    # Build reverse map: tc_op_prefix -> group_name
    op_to_group: dict[str, str] = {}
    for group_name, prefixes in fused_groups.items():
        for prefix in prefixes:
            op_to_group[prefix] = group_name

    def _get_group(func_name: str) -> str | None:
        for prefix, group in op_to_group.items():
            if func_name.startswith(prefix):
                return group
        return None

    # --- Phase 1: Pessimistic per-func_name counting ---
    # Collect all unique func_names and which ones ever missed
    all_func_names: set[str] = set()
    miss_func_names: set[str] = set()
    zero_cost_funcs: set[str] = set()

    for func_name, kernel_type, _shape_sig, _latency_s in hit_details:
        all_func_names.add(func_name)
        if kernel_type in ("zero_cost", "accepted_miss"):
            zero_cost_funcs.add(func_name)

    for func_name, _reason, _shapes, *_ in miss_details:
        all_func_names.add(func_name)
        miss_func_names.add(func_name)

    # Pessimistic: HIT only if NEVER missed
    hit_func_names = all_func_names - miss_func_names

    # Group hits
    ungrouped_hits: set[str] = set()
    hit_groups_seen: dict[str, set[str]] = {}
    for func_name in hit_func_names:
        group = _get_group(func_name)
        if group:
            hit_groups_seen.setdefault(group, set()).add(func_name)
        else:
            ungrouped_hits.add(func_name)

    # Group misses
    miss_groups_seen: set[str] = set()
    ungrouped_misses: set[str] = set()
    for func_name in miss_func_names:
        group = _get_group(func_name)
        if group:
            miss_groups_seen.add(group)
        else:
            ungrouped_misses.add(func_name)

    # A fused group is HIT only if ALL members HIT and NONE MISS
    grouped_hits: set[str] = set()
    for group in hit_groups_seen:
        if group not in miss_groups_seen:
            grouped_hits.add(group)

    all_groups = set(hit_groups_seen.keys()) | miss_groups_seen

    fused_hit = len(ungrouped_hits) + len(grouped_hits)
    fused_miss = len(ungrouped_misses) + len(all_groups - grouped_hits)
    fused_total = fused_hit + fused_miss

    # No zero_cost view
    fused_hit_no_zc = len(ungrouped_hits - zero_cost_funcs) + len(grouped_hits)
    fused_total_no_zc = fused_total - len(zero_cost_funcs & hit_func_names)

    # Per-shape stats computed separately by compute_per_shape_stats()

    return {
        "m2_fused_hit": fused_hit,
        "m2_fused_miss": fused_miss,
        "m2_fused_total": fused_total,
        "m2_fused_op_hr": fused_hit / fused_total if fused_total > 0 else 0,
        "m3_fused_hit_no_zc": fused_hit_no_zc,
        "m3_fused_total_no_zc": fused_total_no_zc,
        "m3_fused_op_hr_no_zc": (
            fused_hit_no_zc / fused_total_no_zc if fused_total_no_zc > 0 else 0
        ),
    }


def compute_per_shape_stats(
    hit_details: list[tuple[str, str, tuple, float]],
    miss_details: list[tuple[str, str, list, ...]],
) -> dict:
    """M4: Per-Shape Match HR (unique shape variants, excl zero_cost).

    Each unique (func_name, shape_sig) pair is counted independently.
    No pessimistic rule, no fused grouping.

    Returns:
        dict with hit_shapes, total_shapes, m4, miss_shape_list.
    """
    hit_shapes: set[tuple[str, tuple]] = set()
    for func_name, kernel_type, shape_sig, _latency_s in hit_details:
        if kernel_type in ("zero_cost", "accepted_miss"):
            continue
        hit_shapes.add((func_name, shape_sig))

    all_shapes: set[tuple[str, tuple]] = set(hit_shapes)
    for func_name, _reason, tc_shapes, *_ in miss_details:
        shape_sig = tuple(tuple(s) for s in tc_shapes) if tc_shapes else ()
        all_shapes.add((func_name, shape_sig))

    m4 = len(hit_shapes) / len(all_shapes) if all_shapes else 0.0
    miss_shape_list = sorted(all_shapes - hit_shapes)
    return {
        "m4_hit_shapes": len(hit_shapes),
        "m4_total_shapes": len(all_shapes),
        "m4_per_shape_hr": m4,
        "m4_miss_shape_list": miss_shape_list,
    }


logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collects M1-M5 metrics by reading EmpiricalPerformanceModel.op_records.

    Decoupled from EmpiricalPerformanceModel: the perf model stores raw
    EmpiricalOpRecord entries; this class processes them into metrics.

    Usage::

        collector = MetricsCollector()
        collector.collect_from_records(pm.op_records)
        collector.log_stats()
        collector.export_hit_miss_report(output_path)
    """

    def __init__(self):
        self._stats = {"hit": 0, "miss": 0}
        self._hit_details: list[tuple[str, str, tuple, float]] = []
        # Each miss: (func_name, reason, tc_shapes, analytic_latency_s)
        self._miss_details: list[tuple[str, str, list[tuple], float]] = []
        # M5: Simulated Latency Coverage accumulators (Roofline-weighted)
        # Uses analytic latency as importance weight — Roofline is inaccurate
        # in absolute terms but reliable for relative importance ranking.
        self._hit_latency_sum = 0.0
        self._total_latency_sum = 0.0

    def collect_from_records(self, records: List[EmpiricalOpRecord]) -> None:
        """Process a list of EmpiricalOpRecord entries into M1-M5 metrics.

        Reads attributes of EmpiricalOpRecord (which are exposed by
        EmpiricalPerformanceModel.op_records) to complete its work.

        Args:
            records: List of EmpiricalOpRecord from EmpiricalPerformanceModel.op_records
        """
        for record in records:
            self._collect_one(
                record.func_name,
                record.lookup_result,
                record.analytic_latency_s,
                record.tc_shapes,
                record.miss_reason,
            )

    def _collect_one(
        self,
        func_name: str,
        result: Optional[QueryResult],
        analytic_latency_s: float,
        tc_shapes: list[tuple],
        miss_reason: Optional[str] = None,
    ) -> None:
        self._total_latency_sum += analytic_latency_s

        if result is not None and result.source != QuerySource.PARTIAL:
            # Full HIT — count as HIT in metrics
            self._stats["hit"] += 1
            self._hit_latency_sum += analytic_latency_s
            kernel_type = result.details.get("kernel_type", "?")
            # For M3/M4 metrics, zero_cost and accepted_miss ops need a
            # sentinel kernel_type so compute_fused_op_stats can exclude them.
            # The real kernel_type is preserved in statistics (→ chrome trace).
            metric_kernel_type = kernel_type
            if result.details.get("zero_cost"):
                metric_kernel_type = "zero_cost"
            elif kernel_type == "accepted_miss":
                metric_kernel_type = "accepted_miss"
            shape_sig = tuple(tc_shapes)
            empirical_s = result.latency_us * 1e-6
            self._hit_details.append(
                (func_name, metric_kernel_type, shape_sig, empirical_s)
            )

        elif result is not None and result.source == QuerySource.PARTIAL:
            # PARTIAL: use empirical latency in E2E sum, but count as MISS
            # in match rate (M1). Do NOT update _hit_latency_sum (M5) —
            # PARTIAL is still conceptually a MISS for accuracy metrics.
            self._stats["miss"] += 1
            missed_kernels = result.details.get("missed_kernels", [])
            reason = f"partial:{','.join(missed_kernels)}"
            self._miss_details.append(
                (func_name, reason, tc_shapes, analytic_latency_s)
            )

        else:
            # Full MISS
            self._stats["miss"] += 1
            reason = miss_reason or "unknown"
            self._miss_details.append(
                (func_name, reason, tc_shapes, analytic_latency_s)
            )

    def get_stats(self) -> dict:
        """Return M1: Raw Op-Count Match Rate."""
        total = self._stats["hit"] + self._stats["miss"]
        return {
            **self._stats,
            "total": total,
            "m1_raw_op_count_hr": self._stats["hit"] / total if total > 0 else 0,
        }

    def log_stats(self) -> None:
        """Log M1-M5 metrics to logger."""
        stats = self.get_stats()
        logger.info(
            "EmpiricalPerformanceModel: %d/%d ops matched (%.1f%%)",
            stats["hit"],
            stats["total"],
            stats["m1_raw_op_count_hr"] * 100,
        )

        partial_details = [
            (fn, reason, shapes, lat)
            for fn, reason, shapes, lat in self._miss_details
            if reason.startswith("partial:")
        ]
        full_miss_details = [
            (fn, reason, shapes, lat)
            for fn, reason, shapes, lat in self._miss_details
            if not reason.startswith("partial:")
        ]

        if partial_details:
            total = stats["total"]
            partial_count = len(partial_details)
            partial_op_counts = Counter(
                fn.removeprefix("torch.ops.").split(".")[-1] if "." in fn else fn
                for fn, _r, _s, _l in partial_details
            )
            op_strs = [
                f"{name}\u00d7{count}" if count > 1 else name
                for name, count in partial_op_counts.most_common()
            ]
            logger.info(
                "  PARTIAL: %d/%d (%s)",
                partial_count,
                total,
                ", ".join(op_strs),
            )

        if self._hit_details:
            display_keys = [f"{fn}->{kt}" for fn, kt, _, _ in self._hit_details]
            hit_counts = Counter(display_keys)
            hit_lines = [
                f"  {mapping} (x{count})" if count > 1 else f"  {mapping}"
                for mapping, count in hit_counts.most_common()
            ]
            logger.info(
                "  HITs (%d unique):\n%s", len(hit_counts), "\n".join(hit_lines)
            )

        if full_miss_details:
            by_reason: dict[str, list[tuple[str, list[tuple]]]] = {}
            for func_name, reason, tc_shapes, _lat in full_miss_details:
                by_reason.setdefault(reason, []).append((func_name, tc_shapes))

            miss_lines = []
            for reason, ops in sorted(by_reason.items()):
                label = _MISS_REASON_LABELS.get(reason, reason)
                op_counts = Counter(func_name for func_name, _ in ops)
                op_strs = [
                    f"{name} (x{count})" if count > 1 else name
                    for name, count in op_counts.most_common()
                ]
                miss_lines.append(f"  [{reason}] {label}: {', '.join(op_strs)}")
                for func_name, tc_shapes in ops:
                    logger.debug("    %s shapes: %s", func_name, tc_shapes)

            logger.info(
                "  MISSes (%d unique reasons):\n%s",
                len(by_reason),
                "\n".join(miss_lines),
            )

        fused = compute_fused_op_stats(self._hit_details, self._miss_details)
        logger.info(
            "Fused Op Match Rate: %d/%d (%.1f%%) [GO/NO-GO]",
            fused["m2_fused_hit"],
            fused["m2_fused_total"],
            fused["m2_fused_op_hr"] * 100,
        )
        logger.info(
            "Fused Op Match Rate (excl zero_cost): %d/%d (%.1f%%) [Reference]",
            fused["m3_fused_hit_no_zc"],
            fused["m3_fused_total_no_zc"],
            fused["m3_fused_op_hr_no_zc"] * 100,
        )

        shape_stats = compute_per_shape_stats(self._hit_details, self._miss_details)
        logger.info(
            "Per-Shape Match Rate: %d/%d (%.1f%%)",
            shape_stats["m4_hit_shapes"],
            shape_stats["m4_total_shapes"],
            shape_stats["m4_per_shape_hr"] * 100,
        )
        if shape_stats["m4_miss_shape_list"]:
            miss_lines = [
                f"  {fn} {ss}" for fn, ss in shape_stats["m4_miss_shape_list"][:20]
            ]
            remaining = len(shape_stats["m4_miss_shape_list"]) - 20
            if remaining > 0:
                miss_lines.append(f"  ... and {remaining} more")
            logger.info(
                "  MISS shapes (%d):\n%s",
                len(shape_stats["m4_miss_shape_list"]),
                "\n".join(miss_lines),
            )

        if self._total_latency_sum > 0:
            m5 = self._hit_latency_sum / self._total_latency_sum
            logger.info(
                "Simulated Latency Coverage: %.1f%% (%.3fms / %.3fms)",
                m5 * 100,
                self._hit_latency_sum * 1000,
                self._total_latency_sum * 1000,
            )

    def export_hit_miss_report(
        self,
        output_path: Path | None = None,
    ) -> dict:
        """Export M1-M5 metrics and per-op MISS details.

        Returns dict with M1-M5 metric summaries and misses list.
        Per-op HIT details are available in the chrome trace (use
        --chrome-trace for per-op analysis with simulation_shapes, kernel_type,
        sub_kernel_durations, etc.).

        If output_path provided, writes JSON to file.

        Note: M6 is computed separately by compute_m6.py using
        --chrome-trace (TC trace) vs --prof-trace (clean forward pass CSV).
        """
        fused = compute_fused_op_stats(self._hit_details, self._miss_details)
        shape = compute_per_shape_stats(self._hit_details, self._miss_details)

        report = {
            "m1": {
                "m1_hit": self._stats["hit"],
                "m1_miss": self._stats["miss"],
                "m1_total": self._stats["hit"] + self._stats["miss"],
                "m1_raw_op_count_hr": self.get_stats()["m1_raw_op_count_hr"],
            },
            "m2": {
                "m2_fused_hit": fused["m2_fused_hit"],
                "m2_fused_total": fused["m2_fused_total"],
                "m2_fused_op_hr": fused["m2_fused_op_hr"],
            },
            "m3": {
                "m3_fused_hit_no_zc": fused["m3_fused_hit_no_zc"],
                "m3_fused_total_no_zc": fused["m3_fused_total_no_zc"],
                "m3_fused_op_hr_no_zc": fused["m3_fused_op_hr_no_zc"],
            },
            "m4": {
                "m4_hit_shapes": shape["m4_hit_shapes"],
                "m4_total_shapes": shape["m4_total_shapes"],
                "m4_per_shape_hr": shape["m4_per_shape_hr"],
                "m4_miss_shape_list": [
                    {"func_name": fn, "shape": [list(s) for s in ss]}
                    for fn, ss in shape["m4_miss_shape_list"]
                ],
            },
            "m5": {
                "m5_hit_latency_sum_s": self._hit_latency_sum,
                "m5_total_latency_sum_s": self._total_latency_sum,
                "m5_simulated_latency_coverage": (
                    self._hit_latency_sum / self._total_latency_sum
                    if self._total_latency_sum > 0
                    else 0.0
                ),
            },
            "misses": [
                {
                    "func_name": fn,
                    "reason": r,
                    "tc_shapes": [list(s) for s in shapes],
                    "analytic_latency_s": lat,
                }
                for fn, r, shapes, lat in self._miss_details
            ],
        }

        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
            logger.info("Metrics report exported to %s", output_path)

        return report
