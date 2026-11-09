# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

import logging

import numpy as np
import pandas as pd

from serving_cast.service.optimizer_summary import (
    PP_RESULT_COLUMNS,
    _compute_disagg_request_qps,
)
from serving_cast.utils import rank_pd_ratio_rows


logger = logging.getLogger(__name__)


class PDRatioThroughputOptimizer:
    """Optimizer for Prefill-Decode ratio throughput optimization.

    This optimizer combines independent P and D optimization results,
    calculates QPS and PD ratio, and outputs Top N configurations.

    QPS Formulas:
        PP=1 P QPS = p_concurrency / ttft * 1000 (req/s)
        PP>1 P QPS = token/s / input_length (steady-state full-prompt basis)
        (shared with the disaggregation summary and capacity planner)
        D QPS = d_concurrency / (tpot * max(output_length - 1, 1)) * 1000 (req/s)

    PD Ratio Calculation:
        PD ratio = D_QPS / P_QPS
    """

    def __init__(self, output_length: int):
        """Initialize the PD ratio optimizer.

        Args:
            output_length: The expected output length for D QPS calculation.
        """
        self.output_length = output_length
        self._p_df: pd.DataFrame = None
        self._d_df: pd.DataFrame = None
        self._result_df: pd.DataFrame = None

    def set_p_results(self, df: pd.DataFrame):
        self._p_df = df

    def set_d_results(self, df: pd.DataFrame):
        self._d_df = df

    def optimize(self) -> pd.DataFrame:
        """Run PD ratio optimization.

        Combines all P and D results, calculates QPS and PD ratio for each
        combination, and returns sorted Top N results.

        Returns:
            DataFrame with PD ratio results sorted by PD_RATIO_RANK_KEYS.
        """
        if self._p_df is None or self._p_df.empty:
            self._result_df = pd.DataFrame()
            return self._result_df

        if self._d_df is None or self._d_df.empty:
            self._result_df = pd.DataFrame()
            return self._result_df

        # PP=1 is the common case, so keep its TTFT-based QPS calculation
        # vectorized. Only PP>1 rows need the per-row steady-state formula and
        # its phase-makespan fallback.
        p_df = self._p_df.copy()
        for column in PP_RESULT_COLUMNS:
            if column not in p_df.columns:
                p_df[column] = None
        concurrency = pd.to_numeric(
            p_df.get("concurrency", pd.Series(index=p_df.index, dtype="float64")), errors="coerce"
        )
        ttft = pd.to_numeric(p_df.get("ttft", pd.Series(index=p_df.index, dtype="float64")), errors="coerce")
        p_df["p_qps"] = concurrency / ttft.where(ttft > 0) * 1000.0
        pp_size = pd.to_numeric(p_df.get("pp_size"), errors="coerce")
        pp_gt1 = pp_size > 1
        if pp_gt1.any():
            p_df.loc[pp_gt1, "p_qps"] = p_df.loc[pp_gt1].apply(
                lambda row: _compute_disagg_request_qps(row, output_length=None),
                axis=1,
            )
        p_df["p_qps"] = pd.to_numeric(p_df["p_qps"], errors="coerce")
        p_df = p_df[np.isfinite(p_df["p_qps"]) & (p_df["p_qps"] > 0)]

        # D QPS = d_concurrency / (tpot * max(output_length - 1, 1)) * 1000 (req/s)
        # Filter out zero tpot to avoid ZeroDivisionError
        d_df = self._d_df.copy()
        for column in PP_RESULT_COLUMNS:
            if column not in d_df.columns:
                d_df[column] = None
        d_df = d_df[d_df["tpot"] > 0]
        d_df["d_qps"] = d_df["concurrency"] / (d_df["tpot"] * max(self.output_length - 1, 1)) * 1000
        d_df = d_df[d_df["d_qps"] > 0]

        if p_df.empty or d_df.empty:
            self._result_df = pd.DataFrame()
            return self._result_df

        # Create cross join for all P and D combinations
        merged = p_df.merge(d_df, how="cross", suffixes=("_p", "_d"))

        # Calculate PD ratio and balanced QPS, p_qps already filtered to be greater than 0
        merged["pd_ratio"] = merged["d_qps"] / merged["p_qps"]
        merged["balanced_qps"] = merged[["p_qps", "d_qps"]].min(axis=1)

        # Select and order columns with consistent suffix naming
        # After cross join: ttft_p from prefill, tpot_d from decode
        result_cols = [
            "pd_ratio",
            "p_qps",
            "d_qps",
            "balanced_qps",
            "ttft_p",
            "tpot_d",
            "parallel_p",
            "parallel_d",
            "num_devices_p",
            "num_devices_d",
            "batch_size_p",
            "batch_size_d",
            "concurrency_p",
            "concurrency_d",
            # P1-42: vision-encoder accounting per phase. ``_d`` is always 0
            # because decode requests carry no images — that inequality is the
            # direct evidence that vision belongs to the P instances only.
            *(
                [f"{column}_p" for column in ("encoder_waves", "vision_time_ms")]
                + [f"{column}_d" for column in ("encoder_waves", "vision_time_ms")]
                if {"encoder_waves_p", "vision_time_ms_p", "encoder_waves_d", "vision_time_ms_d"} <= set(merged.columns)
                else []
            ),
            *[f"{column}_p" for column in PP_RESULT_COLUMNS],
            *[f"{column}_d" for column in PP_RESULT_COLUMNS],
        ]
        self._result_df = rank_pd_ratio_rows(merged[result_cols]).reset_index(drop=True)

        return self._result_df
