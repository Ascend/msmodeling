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

import pandas as pd

from serving_cast.service.optimizer_summary import PP_RESULT_COLUMNS, _steady_pp_prefill_request_qps
from serving_cast.service.utils import PREFILL_PHASE_MAKESPAN_COLUMN
from serving_cast.utils import rank_pd_ratio_rows


logger = logging.getLogger(__name__)


class PDRatioThroughputOptimizer:
    """Optimizer for Prefill-Decode ratio throughput optimization.

    This optimizer combines independent P and D optimization results,
    calculates QPS and PD ratio, and outputs Top N configurations.

    QPS Formulas:
        P QPS = p_concurrency / prefill_phase_makespan_ms * 1000 (req/s)
                (legacy rows fall back to ttft; PP>1 rows use the steady-state
                token/s / input_length — full-prompt token basis of the
                throughput numerator — via the shared per-row helper
                ``_steady_pp_prefill_request_qps``; Issues #543 and review P1/P8)
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

        # Calculate QPS using vectorized operations. Chunked prefill's TTFT is
        # a request-average completion time, whereas its QPS is defined by the
        # whole phase makespan. Older result frames did not retain that timing,
        # so keep a TTFT fallback for backward compatibility.
        p_df = self._p_df.copy()
        for column in PP_RESULT_COLUMNS:
            if column not in p_df.columns:
                p_df[column] = None
        if PREFILL_PHASE_MAKESPAN_COLUMN in p_df.columns:
            p_df["_p_qps_denominator"] = pd.to_numeric(p_df[PREFILL_PHASE_MAKESPAN_COLUMN], errors="coerce")
        else:
            p_df["_p_qps_denominator"] = pd.to_numeric(p_df["ttft"], errors="coerce")
        p_df = p_df[p_df["_p_qps_denominator"] > 0]
        p_df["p_qps"] = p_df["concurrency"] / p_df["_p_qps_denominator"] * 1000
        # PP>1 prefill rows report steady-state throughput (anchored on the
        # steady wave interval, PR !821); override their P QPS with the
        # shared per-row steady-state helper so summary QPS, p_qps and
        # capacity planning cannot drift apart (Issue #543, review P1/P8).
        # The helper guards PP=1 and missing operands itself; rows it cannot
        # resolve keep the phase-makespan formula above.
        steady_qps = p_df.apply(_steady_pp_prefill_request_qps, axis=1)
        take_steady = steady_qps.notna() & (steady_qps > 0)
        # Guard the empty case: assigning an all-False object Series slice
        # into the float p_qps column raises LossySetitemError on pandas.
        if take_steady.any():
            p_df.loc[take_steady, "p_qps"] = steady_qps[take_steady].astype(float)
        p_df = p_df.drop(columns="_p_qps_denominator")
        p_df = p_df[p_df["p_qps"] > 0]

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
            *[f"{column}_p" for column in PP_RESULT_COLUMNS],
            *[f"{column}_d" for column in PP_RESULT_COLUMNS],
        ]
        self._result_df = rank_pd_ratio_rows(merged[result_cols]).reset_index(drop=True)

        return self._result_df
