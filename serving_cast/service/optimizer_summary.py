# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from typing import Optional

import pandas as pd
from prettytable import PrettyTable

from serving_cast.utils import best_pd_row_per_group, rank_pd_ratio_rows, sort_pd_ratio_dict_rows

logger = logging.getLogger(__name__)


def _positive_float(value) -> Optional[float]:
    num = pd.to_numeric(value, errors="coerce")
    if num is None or pd.isna(num) or float(num) <= 0:
        return None
    return float(num)


def _compute_disagg_request_qps(row: pd.Series, output_length: Optional[int]) -> Optional[float]:
    """Per-phase request QPS for disaggregation summary rows.

    Same formulas as ``pd_ratio_throughput_optimizer``: prefill uses
    ``concurrency / ttft * 1000``, decode uses
    ``concurrency / (tpot * output_length) * 1000``. Rows with both TTFT and
    TPOT valid (aggregation-style) return ``None``.
    """
    conc = _positive_float(row.get("concurrency"))
    if conc is None:
        return None
    ttft = _positive_float(row.get("ttft"))
    tpot = _positive_float(row.get("tpot"))

    if ttft is not None and tpot is None:
        return conc / ttft * 1000.0
    if tpot is not None and ttft is None:
        if output_length is None or int(output_length) <= 0:
            return None
        return conc / (tpot * float(output_length)) * 1000.0
    return None


TTFT_COLUMN = "TTFT (ms)"
TPOT_COLUMN = "TPOT (ms)"
SHOW_COLUMNS = [
    "Top",
    "\033[1mThroughput\033[0m (token/s)",
    TTFT_COLUMN,
    TPOT_COLUMN,
    "concurrency",
    "num_devices",
    "parallel",
    "batch_size",
]


def _fmt_optional(value, fmt: str = "{:.2f}") -> str:
    return fmt.format(value) if value is not None else "-"


def _sorted_rows(rows: list[dict], metric: str) -> list[dict]:
    return sorted(rows, key=lambda row: row.get(metric, 0.0), reverse=True)


class OptimizerSummary:
    def __init__(self, data_config):
        self._early_stop_flag = None
        self._summary_df = None
        self.data_config = data_config

    def set_summary_df(self, summary_df):
        self._summary_df = summary_df

    def get_summary_df(self):
        return self._summary_df

    def set_early_stop_flag(self, memory_left, tpot, ttft):
        def check(value, limit):
            return value is not None and limit is not None and value > limit

        self._early_stop_flag = (
            (memory_left < 0) or check(tpot, self.data_config.tpot_limits) or check(ttft, self.data_config.ttft_limits)
        )

    def check_early_stop_flag(self):
        return self._early_stop_flag

    def _is_pd_ratio_mode(self):
        """Return whether this is PD ratio optimization mode."""
        return (
            hasattr(self.data_config, "prefill_devices_per_instance")
            and self.data_config.prefill_devices_per_instance is not None
            and hasattr(self.data_config, "decode_devices_per_instance")
            and self.data_config.decode_devices_per_instance is not None
        )

    def report_final_result(self, args, silent: bool = False):
        if silent:
            return
        if self._summary_df is None or self._summary_df.empty:
            logger.warning("Summary DataFrame is empty or unset. Please set it first.")
            return

        if self._is_pd_ratio_mode():
            # Apply PD ratio filtering for both dump and normal output
            filtered_df = self._prepare_pd_ratio_results()
            if args.dump_original_results:
                if filtered_df.empty:
                    logger.info("No results after PD ratio filtering.")
                else:
                    print("\n" + filtered_df.to_string(index=False) + "\n")
            else:
                final_out = self._get_pd_ratio_final_out(args, filtered_df)
                print("\n" + "\n".join(final_out))
        elif args.dump_original_results:
            print("\n" + self._summary_df.to_string(index=False) + "\n")
        else:
            final_out = self._get_agg_disagg_final_out(args)
            print("\n" + "\n".join(final_out))

    def _prepare_agg_disagg_results(self):
        """Prepare and filter results for aggregation/disaggregation mode."""
        tpot_limit = self.data_config.tpot_limits or float("inf")
        ttft_limit = self.data_config.ttft_limits or float("inf")

        mask = (pd.to_numeric(self._summary_df["tpot"], errors="coerce").fillna(float("inf")) <= tpot_limit) & (
            pd.to_numeric(self._summary_df["ttft"], errors="coerce").fillna(float("inf")) <= ttft_limit
        )

        return (
            self._summary_df[mask]
            .sort_values(by="token/s", ascending=False)
            .groupby("parallel")
            .first()
            .reset_index()
            .sort_values(by="token/s", ascending=False)
            .reset_index(drop=True)
        )

    def _get_agg_disagg_final_out(self, args):
        sorted_summary_df = self._prepare_agg_disagg_results()
        if sorted_summary_df.empty:
            logger.warning("No optimizer rows passed TTFT/TPOT filters; cannot pick best configuration.")
            return ["*" * 80, "No configurations satisfy the current TTFT/TPOT filters.", "*" * 80]

        best_result = sorted_summary_df.loc[0]

        final_out = []
        final_out.append("*" * 80)

        final_out.append("  " + "-" * 76)
        final_out.append("  Input Configuration: ")
        final_out.append(f"    Model: {args.model_id}")
        final_out.append(f"    Quantize Linear action: {args.quantize_linear_action}")
        final_out.append(f"    Quantize Attention action: {args.quantize_attention_action}")
        final_out.append(f"    Devices: {args.num_devices} {args.device}")
        final_out.append(f"    TTFT Limits: {self.data_config.ttft_limits} ms")
        final_out.append(f"    TPOT Limits: {self.data_config.tpot_limits} ms")
        final_out.append("  " + "-" * 76)

        final_out.append("  Overall Best Configuration: ")
        final_out.append(f"    Best Throughput: {best_result['token/s']:.2f} tokens/s")
        if best_result["ttft"] is not None:
            final_out.append(f"    TTFT: {best_result['ttft']:.2f} ms")
        if best_result["tpot"] is not None:
            final_out.append(f"    TPOT: {best_result['tpot']:.2f} ms")
        final_out.append("  " + "-" * 76)

        table_buf = (
            _get_disagg_table_buf(sorted_summary_df, self.data_config.output_length)
            if args.disagg
            else _get_agg_table_buf(sorted_summary_df)
        )
        final_out.append(table_buf)
        final_out.append("*" * 80)

        return final_out

    def collect_comparison_row(self, device_label: str) -> Optional[dict]:
        """Pick the best aggregation/disaggregation row for cross-hardware comparison."""
        return self._best_agg_disagg_row(device_label)

    def collect_disagg_prefill_row(self, device_label: str) -> Optional[dict]:
        """Best Prefill row from a disaggregation Prefill-phase summary (cross-hardware)."""
        if self.data_config.ttft_limits is None or self.data_config.tpot_limits is not None:
            return None
        return self._best_agg_disagg_row(device_label)

    def collect_disagg_decode_row(self, device_label: str) -> Optional[dict]:
        """Best Decode row from a disaggregation Decode-phase summary (cross-hardware)."""
        if self.data_config.tpot_limits is None or self.data_config.ttft_limits is not None:
            return None
        return self._best_agg_disagg_row(device_label)

    def _best_agg_disagg_row(self, device_label: str) -> Optional[dict]:
        if self._summary_df is None or self._summary_df.empty or self._is_pd_ratio_mode():
            return None
        filtered = self._prepare_agg_disagg_results()
        if filtered.empty:
            return None
        return self._row_dict_from_filtered_best(device_label, filtered.iloc[0])

    def collect_pd_ratio_comparison_row(self, device_label: str) -> Optional[dict]:
        """Pick the best PD-ratio row (max ``balanced_qps`` after filtering) for cross-hardware."""
        if self._summary_df is None or self._summary_df.empty:
            return None
        if not self._is_pd_ratio_mode():
            return None
        filtered = self._prepare_pd_ratio_results()
        if filtered.empty:
            return None
        r = filtered.iloc[0]
        p_inst = None
        d_inst = None
        nd = self.data_config.num_devices
        if nd is not None:
            p_calc, d_calc = self._calculate_instance_distribution(
                float(r["pd_ratio"]),
                int(nd),
                int(r["num_devices_p"]),
                int(r["num_devices_d"]),
            )
            if p_calc > 0 and d_calc > 0:
                p_inst, d_inst = p_calc, d_calc

        return {
            "device": device_label,
            "balanced_qps": float(r["balanced_qps"]),
            "pd_ratio": float(r["pd_ratio"]),
            "p_qps": float(r["p_qps"]),
            "d_qps": float(r["d_qps"]),
            "ttft_p": float(r["ttft_p"]),
            "tpot_d": float(r["tpot_d"]),
            "parallel_p": r["parallel_p"],
            "parallel_d": r["parallel_d"],
            "p_instances": p_inst,
            "d_instances": d_inst,
            "total_devices": int(nd) if nd is not None else None,
        }

    def _row_dict_from_filtered_best(self, device_label: str, r: pd.Series) -> dict:
        def _fnum(key: str):
            v = r.get(key)
            if v is None or pd.isna(v):
                return None
            return float(v)

        return {
            "device": device_label,
            "throughput_tps": float(r["token/s"]),
            "ttft_ms": _fnum("ttft"),
            "tpot_ms": _fnum("tpot"),
            "concurrency": r["concurrency"],
            "num_devices": r["num_devices"],
            "parallel": r["parallel"],
            "batch_size": r["batch_size"],
            "qps_req_s": _compute_disagg_request_qps(r, getattr(self.data_config, "output_length", None)),
        }

    def _prepare_pd_ratio_results(self):
        """Prepare and filter results for PD ratio mode.

        Filters applied:
        1. Keep only the best result for each unique (p_parallel, d_parallel) combination
        2. Keep only one result for each unique balanced_qps value

        Results are sorted by PD_RATIO_RANK_KEYS (see serving_cast.utils).
        """
        tpot_limit = self.data_config.tpot_limits or float("inf")
        ttft_limit = self.data_config.ttft_limits or float("inf")

        # Apply limits filter
        mask = (pd.to_numeric(self._summary_df["ttft_p"], errors="coerce").fillna(float("inf")) <= ttft_limit) & (
            pd.to_numeric(self._summary_df["tpot_d"], errors="coerce").fillna(float("inf")) <= tpot_limit
        )

        filtered_df = best_pd_row_per_group(self._summary_df[mask], ["parallel_p", "parallel_d"])
        filtered_df["_balanced_qps_rounded"] = filtered_df["balanced_qps"].round(2)
        result_df = best_pd_row_per_group(filtered_df, ["_balanced_qps_rounded"]).drop(
            columns=["_balanced_qps_rounded"]
        )
        return rank_pd_ratio_rows(result_df).reset_index(drop=True)

    def _get_pd_ratio_final_out(self, args, sorted_summary_df):
        """Generate the final output string for PD ratio mode.

        Args:
            args: Command line arguments.
            sorted_summary_df: Pre-filtered and sorted DataFrame.
        """
        best_result = sorted_summary_df.loc[0]

        final_out = []
        final_out.append("*" * 120)

        # Input Configuration section
        final_out.append("  " + "-" * 116)
        final_out.append("  Input Configuration:")
        final_out.append(f"    Model: {args.model_id}")
        # Only show Devices when user specifies --num-devices
        if (
            self.data_config.num_devices
            >= self.data_config.prefill_devices_per_instance + self.data_config.decode_devices_per_instance
        ):
            final_out.append(f"    Devices: {self.data_config.num_devices} {args.device}")
        final_out.append(f"    Prefill Devices Per Instance: {self.data_config.prefill_devices_per_instance}")
        final_out.append(f"    Decode Devices Per Instance: {self.data_config.decode_devices_per_instance}")
        final_out.append(f"    TTFT Limits: {self.data_config.ttft_limits} ms")
        final_out.append(f"    TPOT Limits: {self.data_config.tpot_limits} ms")
        final_out.append("  " + "-" * 116)

        # Overall Best Configuration section
        final_out.append("  Overall Best Configuration:")
        final_out.append(f"      PD Ratio: {best_result['pd_ratio']:.2f} (P Instance:D Instance)")
        final_out.append(
            f"      Prefill QPS: {best_result['p_qps']:.2f} req/s  "
            f"(TTFT: {best_result['ttft_p']:.2f} ms, Parallel: {best_result['parallel_p']}, "
            f"Batch: {best_result['batch_size_p']}, Concurrency: {best_result['concurrency_p']})"
        )
        final_out.append(
            f"      Decode QPS:  {best_result['d_qps']:.2f} req/s  "
            f"(TPOT: {best_result['tpot_d']:.2f} ms, Parallel: {best_result['parallel_d']}, "
            f"Batch: {best_result['batch_size_d']}, Concurrency: {best_result['concurrency_d']})"
        )

        # Calculate instance distribution when num_devices is specified
        if self.data_config.num_devices is not None:
            p_inst, d_inst = self._calculate_instance_distribution(
                best_result["pd_ratio"],
                self.data_config.num_devices,
                best_result["num_devices_p"],
                best_result["num_devices_d"],
            )
            if p_inst > 0 and d_inst > 0:
                final_out.append(f"      P Instances: {p_inst} ({p_inst * best_result['num_devices_p']} devices)")
                final_out.append(f"      D Instances: {d_inst} ({d_inst * best_result['num_devices_d']} devices)")

        final_out.append("  " + "-" * 116)

        # Top N table (using filtered results)
        table_buf = _get_pd_ratio_table_buf(sorted_summary_df)
        final_out.append(table_buf)
        final_out.append("*" * 120)

        return final_out

    def _calculate_instance_distribution(
        self,
        pd_ratio: float,
        total_devices: int,
        p_devices_per_inst: int,
        d_devices_per_inst: int,
    ) -> tuple[int, int]:
        """Calculate the number of P and D instances.

        Args:
            pd_ratio: PD ratio (P:D ratio).
            total_devices: Total number of devices available.
            p_devices_per_inst: Devices per P instance.
            d_devices_per_inst: Devices per D instance.

        Returns:
            Tuple of (p_instances, d_instances).
        """
        # PD ratio = D_QPS / P_QPS
        # For supply-demand balance: P_instances * P_QPS = D_instances * D_QPS
        # So: P_instances / D_instances = D_QPS / P_QPS = pd_ratio
        # Therefore: P_instances = D_instances * pd_ratio

        best_p_inst = 0
        best_d_inst = 0
        best_diff = float("inf")

        max_d_inst = total_devices // d_devices_per_inst
        for d_inst in range(1, max_d_inst + 1):
            ideal_p_inst = d_inst * pd_ratio
            p_inst = round(ideal_p_inst)

            if p_inst < 1:
                p_inst = 1

            total_used = p_inst * p_devices_per_inst + d_inst * d_devices_per_inst
            if total_used <= total_devices:
                diff = abs(p_inst - ideal_p_inst)
                if diff < best_diff:
                    best_diff = diff
                    best_p_inst = p_inst
                    best_d_inst = d_inst

        return best_p_inst, best_d_inst


def _get_agg_table_buf(df: pd.DataFrame):
    show_len = len(df)
    table_buf = []
    table_buf.append(f"Top {show_len} PD Aggregated Configurations: ")
    table = PrettyTable()
    table.field_names = SHOW_COLUMNS
    for i in range(show_len):
        row = df.loc[i]
        table.add_row(
            [
                i + 1,
                f"\033[1m{row['token/s']:.2f}\033[0m",
                f"{row['ttft']:.2f}",
                f"{row['tpot']:.2f}",
                row["concurrency"],
                row["num_devices"],
                row["parallel"],
                row["batch_size"],
            ]
        )
    table_buf.append(table.get_string())
    return "\n".join(table_buf)


def _get_disagg_table_buf(df: pd.DataFrame, output_length: Optional[int] = None):
    local_column = SHOW_COLUMNS.copy()
    ttft0 = df.iloc[0]["ttft"] if len(df) and "ttft" in df.columns else None
    is_decode = ttft0 is None or pd.isna(ttft0)
    show_len = len(df)
    table_buf = []
    table = PrettyTable()
    if is_decode:
        table_buf.append(f"Top {show_len} PD Disaggregated Decode Configurations: ")
        local_column.insert(2, "QPS (req/s)")
        local_column.remove(TTFT_COLUMN)
    else:
        table_buf.append(f"Top {show_len} PD Disaggregated Prefill Configurations: ")
        local_column.insert(2, "QPS (req/s)")
        local_column.remove(TPOT_COLUMN)

    table.field_names = local_column
    for i in range(show_len):
        row = df.loc[i]
        qps = _compute_disagg_request_qps(row, output_length)
        qps_cell = f"{qps:.2f}" if qps is not None else "-"
        table.add_row(
            [
                i + 1,
                f"\033[1m{row['token/s']:.2f}\033[0m",
                qps_cell,
                f"{row['tpot']:.2f}" if is_decode else f"{row['ttft']:.2f}",
                row["concurrency"],
                row["num_devices"],
                row["parallel"],
                row["batch_size"],
            ]
        )
    table_buf.append(table.get_string())
    return "\n".join(table_buf)


def _get_pd_ratio_table_buf(df: pd.DataFrame):
    """Generate the PD ratio table buffer.

    Args:
        df: DataFrame containing PD ratio results.

    Returns:
        String representation of the PD ratio table.
    """
    show_len = len(df)
    table_buf = []
    table_buf.append(f"  Top {show_len} PD Ratio Configurations:")

    table = PrettyTable()

    table.field_names = [
        "Top",
        "PD Ratio",
        "Balanced QPS (req/s)",
        "P QPS (req/s)",
        "D QPS (req/s)",
        "TTFT (ms)",
        "TPOT (ms)",
        "P Parallel",
        "D Parallel",
        "P Devices/Instance",
        "D Devices/Instance",
        "P Batch Size",
        "D Batch Size",
        "P Concurrency",
        "D Concurrency",
    ]

    for i in range(show_len):
        row = df.loc[i]
        row_data = [
            i + 1,
            f"{row['pd_ratio']:.2f}",
            f"{row['balanced_qps']:.2f}",
            f"{row['p_qps']:.2f}",
            f"{row['d_qps']:.2f}",
            f"{row['ttft_p']:.2f}",
            f"{row['tpot_d']:.2f}",
            row["parallel_p"],
            row["parallel_d"],
            row["num_devices_p"],
            row["num_devices_d"],
            row["batch_size_p"],
            row["batch_size_d"],
            row["concurrency_p"],
            row["concurrency_d"],
        ]
        table.add_row(row_data)

    table_buf.append(table.get_string())
    return "\n".join(table_buf)


def render_cross_device_comparison(rows: list[dict]) -> str:
    """Pretty-print a ranked table of best configs across hardware profiles."""
    if not rows:
        return ""
    sorted_rows = _sorted_rows(rows, "throughput_tps")
    lines = [
        "",
        "*" * 100,
        "  Cross-hardware - PD Aggregated (best throughput config per device under TTFT/TPOT limits)",
        "  " + "-" * 96,
    ]
    table = PrettyTable()
    table.field_names = [
        "Top",
        "Device",
        "Throughput (token/s)",
        "TTFT (ms)",
        "TPOT (ms)",
        "Concurrency",
        "Parallel",
        "Batch",
        "num_devices",
    ]
    for i, row in enumerate(sorted_rows):
        table.add_row(
            [
                i + 1,
                row.get("device", ""),
                f"{row['throughput_tps']:.2f}",
                _fmt_optional(row.get("ttft_ms")),
                _fmt_optional(row.get("tpot_ms")),
                row.get("concurrency", ""),
                row.get("parallel", ""),
                row.get("batch_size", ""),
                row.get("num_devices", ""),
            ]
        )
    lines.append(table.get_string())
    lines.append("*" * 100)
    lines.append("")
    return "\n".join(lines)


def render_cross_hardware_pd_ratio(rows: list[dict]) -> str:
    """Cross-device PD ratio: one row per hardware (best balanced QPS after PD filtering)."""
    if not rows:
        return ""
    sorted_rows = sort_pd_ratio_dict_rows(rows)
    banner_w = 120
    lines = [
        "",
        "*" * banner_w,
        "  Cross-hardware - PD Ratio (best balanced QPS per device under TTFT/TPOT limits)",
        "  " + "-" * (banner_w - 4),
    ]
    table = PrettyTable()
    table.field_names = [
        "Top",
        "Device",
        "Balanced QPS (req/s)",
        "PD Ratio (P:D inst)",
        "P QPS (req/s)",
        "D QPS (req/s)",
        "TTFT (ms)",
        "TPOT (ms)",
        "P inst",
        "D inst",
    ]
    for i, row in enumerate(sorted_rows):
        p_inst = row.get("p_instances")
        d_inst = row.get("d_instances")
        table.add_row(
            [
                i + 1,
                row.get("device", ""),
                f"{row['balanced_qps']:.2f}",
                f"{row['pd_ratio']:.2f}",
                f"{row['p_qps']:.2f}",
                f"{row['d_qps']:.2f}",
                f"{row['ttft_p']:.2f}",
                f"{row['tpot_d']:.2f}",
                str(p_inst) if p_inst is not None else "-",
                str(d_inst) if d_inst is not None else "-",
            ]
        )
    lines.append(table.get_string())
    if any(r.get("p_instances") is not None for r in sorted_rows):
        td = sorted_rows[0].get("total_devices")
        if td is not None:
            lines.append(
                "  P/D instance counts: heuristic integer split under "
                f"--num-devices={td} (same rule as per-device Overall Best)."
            )
    lines.append("*" * banner_w)
    lines.append("")
    return "\n".join(lines)


def render_cross_hardware_disagg_prefill(rows: list[dict]) -> str:
    """Cross-device table for disaggregation Prefill phase (TTFT-constrained)."""
    if not rows:
        return ""
    sorted_rows = _sorted_rows(rows, "throughput_tps")
    lines = [
        "",
        "*" * 108,
        "  Cross-hardware - PD Disaggregated Prefill (best token/s per device under TTFT limits)",
        "  " + "-" * 104,
    ]
    table = PrettyTable()
    table.field_names = [
        "Top",
        "Device",
        "Prefill throughput (token/s)",
        "QPS (req/s)",
        "TTFT (ms)",
        "TPOT (ms)",
        "Concurrency",
        "Parallel",
        "Batch",
        "num_devices",
    ]
    for i, row in enumerate(sorted_rows):
        table.add_row(
            [
                i + 1,
                row.get("device", ""),
                f"{row['throughput_tps']:.2f}",
                _fmt_optional(row.get("qps_req_s")),
                _fmt_optional(row.get("ttft_ms")),
                "-",
                row.get("concurrency", ""),
                row.get("parallel", ""),
                row.get("batch_size", ""),
                row.get("num_devices", ""),
            ]
        )
    lines.append(table.get_string())
    lines.append("*" * 108)
    lines.append("")
    return "\n".join(lines)


def render_cross_hardware_disagg_decode(rows: list[dict]) -> str:
    """Cross-device table for disaggregation Decode phase (TPOT-constrained)."""
    if not rows:
        return ""
    sorted_rows = _sorted_rows(rows, "throughput_tps")
    lines = [
        "",
        "*" * 108,
        "  Cross-hardware - PD Disaggregated Decode (best token/s per device under TPOT limits)",
        "  " + "-" * 104,
    ]
    table = PrettyTable()
    table.field_names = [
        "Top",
        "Device",
        "Decode throughput (token/s)",
        "QPS (req/s)",
        "TTFT (ms)",
        "TPOT (ms)",
        "Concurrency",
        "Parallel",
        "Batch",
        "num_devices",
    ]
    for i, row in enumerate(sorted_rows):
        table.add_row(
            [
                i + 1,
                row.get("device", ""),
                f"{row['throughput_tps']:.2f}",
                _fmt_optional(row.get("qps_req_s")),
                "-",
                _fmt_optional(row.get("tpot_ms")),
                row.get("concurrency", ""),
                row.get("parallel", ""),
                row.get("batch_size", ""),
                row.get("num_devices", ""),
            ]
        )
    lines.append(table.get_string())
    lines.append("*" * 108)
    lines.append("")
    return "\n".join(lines)


def render_hardware_profile_comparison(device_names: list[str]) -> str:
    """Pretty-print core modeling parameters for multiple ``--device`` profiles.

    Compact ASCII-oriented labels: effective cube/vector compute, memory bandwidth,
    communication bandwidth, capacity, and logical comm-grid shape.
    """
    if not device_names:
        return ""
    try:
        import torch

        from tensor_cast import device_profiles  # noqa: F401 - register profiles
        from tensor_cast.device import DeviceProfile
    except ImportError as exc:
        logger.warning("Hardware profile comparison skipped: %s", exc)
        return ""

    ordered = list(dict.fromkeys(device_names))
    banner_w = 108
    lines = [
        "",
        "*" * banner_w,
        "  Cross-hardware - device profile summary (modeling abstraction vs performance merge tables)",
        "  Device profile parameter comparison (effective compute / memory BW / comm BW)",
        "  " + "-" * (banner_w - 4),
    ]
    table = PrettyTable()
    table.field_names = [
        "Device",
        "Cube Compute (TFLOPS)",
        "Vector Compute (TFLOPS)",
        "HBM BW (TB/s)",
        "Memory (GB)",
        "Comm Grid",
        "Comm BW (GB/s)",
    ]

    def _effective_tflops(ops: dict, profile: DeviceProfile) -> Optional[float]:
        peak = ops.get(torch.bfloat16)
        if peak is None:
            peak = ops.get(torch.half)
        if peak is None and ops:
            peak = max(ops.values())
        if peak is None:
            return None
        return (peak / 1e12) * profile.compute_efficiency

    def _fmt_compact_num(value: float) -> str:
        return f"{value:g}"

    def _comm_bw_expr(profile: DeviceProfile) -> str:
        parts = []
        for idx in sorted(profile.comm_grid.topologies):
            topology = profile.comm_grid.topologies[idx]
            eff_bw_gbs = topology.bandwidth_bytes_ps * topology.comm_efficiency / 1e9
            parts.append(_fmt_compact_num(eff_bw_gbs))
        return " | ".join(parts) if parts else "-"

    def _shape_str(profile: DeviceProfile) -> str:
        g = profile.comm_grid.grid
        return " x ".join(str(int(x)) for x in g.shape)

    for name in ordered:
        prof = DeviceProfile.all_device_profiles.get(name)
        if prof is None:
            table.add_row([name, "-", "-", "-", "-", "-", "-"])
            continue
        cube = _effective_tflops(prof.mma_ops, prof)
        vector = _effective_tflops(prof.gp_ops, prof)
        nom_bw_TBs = prof.memory_bandwidth_bytes_ps / (1024**4)
        eff_bw_TBs = nom_bw_TBs * prof.memory_efficiency
        mem_gb = prof.memory_size_bytes / (1024**3)
        table.add_row(
            [
                prof.name,
                f"{cube:.2f}" if cube is not None else "-",
                f"{vector:.2f}" if vector is not None else "-",
                f"{eff_bw_TBs:.3f}",
                f"{mem_gb:.1f}",
                _shape_str(prof),
                _comm_bw_expr(prof),
            ]
        )

    lines.append(table.get_string())
    lines.extend(
        [
            "  Notes:",
            "  - Cube/Vector Compute: nominal BF16 peak x compute_efficiency (FP16 peak if BF16 unset).",
            "  - HBM BW: nominal HBM bandwidth x memory_efficiency.",
            "  - Comm BW: effective GB/s per topology (bandwidth_bytes_ps x comm_efficiency / 1e9), in topology order.",
            "    Example: 50 x 0.7 = 35 GB/s.",
        ]
    )
    lines.append("*" * banner_w)
    lines.append("")
    return "\n".join(lines)
