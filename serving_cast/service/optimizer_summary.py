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

import pandas as pd
from prettytable import PrettyTable


logger = logging.getLogger(__name__)


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
        """Check if this is PD ratio optimization mode."""
        return (
            hasattr(self.data_config, "prefill_devices_per_instance")
            and self.data_config.prefill_devices_per_instance is not None
            and hasattr(self.data_config, "decode_devices_per_instance")
            and self.data_config.decode_devices_per_instance is not None
        )

    def report_final_result(self, args):
        if self._summary_df is None or self._summary_df.empty:
            logger.warning("Summary DataFrame is None. Please set it first.")
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
            logger.warning(
                "No optimizer rows passed TTFT/TPOT filters; cannot pick best configuration."
            )
            return [
                "*" * 80,
                "No configurations satisfy the current TTFT/TPOT filters.",
                "*" * 80,
            ]

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

        table_buf = _get_disagg_table_buf(sorted_summary_df) if args.disagg else _get_agg_table_buf(sorted_summary_df)
        final_out.append(table_buf)
        final_out.append("*" * 80)

        return final_out

    def _prepare_pd_ratio_results(self):
        """Prepare and filter results for PD ratio mode.

        Filters applied:
        1. Keep only the best result for each unique (p_parallel, d_parallel) combination
        2. Keep only one result for each unique balanced_qps value

        Results are sorted by balanced_qps in descending order.
        """
        tpot_limit = self.data_config.tpot_limits or float("inf")
        ttft_limit = self.data_config.ttft_limits or float("inf")

        # Apply limits filter
        mask = (pd.to_numeric(self._summary_df["ttft_p"], errors="coerce").fillna(float("inf")) <= ttft_limit) & (
            pd.to_numeric(self._summary_df["tpot_d"], errors="coerce").fillna(float("inf")) <= tpot_limit
        )

        filtered_df = self._summary_df[mask]

        # Step 1: Keep best result for each (parallel_p, parallel_d) combination
        filtered_df = (
            filtered_df.sort_values(by="balanced_qps", ascending=False)
            .groupby(["parallel_p", "parallel_d"], as_index=False)
            .first()
        )

        # Step 2: Keep one result per balanced_qps (round to 2 decimal places to group similar values)
        filtered_df["_balanced_qps_rounded"] = filtered_df["balanced_qps"].round(2)
        result_df = (
            filtered_df.sort_values(by="balanced_qps", ascending=False)
            .groupby("_balanced_qps_rounded", as_index=False)
            .first()
            .drop(columns=["_balanced_qps_rounded"])
            .sort_values(by="balanced_qps", ascending=False)
            .reset_index(drop=True)
        )

        return result_df

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
    table_buf.append(f"Top {show_len} Aggregation Configurations: ")
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


def _get_disagg_table_buf(df: pd.DataFrame):
    is_decode = df.get("ttft").loc[0] is None
    local_column = SHOW_COLUMNS.copy()
    show_len = len(df)
    table_buf = []
    table = PrettyTable()
    if is_decode:
        table_buf.append(f"Top {show_len} Disaggregation (Decode) Configurations: ")
        local_column.remove(TTFT_COLUMN)
    else:
        table_buf.append(f"Top {show_len} Disaggregation (Prefill) Configurations: ")
        local_column.remove(TPOT_COLUMN)

    table.field_names = local_column
    for i in range(show_len):
        row = df.loc[i]
        table.add_row(
            [
                i + 1,
                f"\033[1m{row['token/s']:.2f}\033[0m",
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
