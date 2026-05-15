# Copyright (c) 2026-2026 Huawei Technologies Co., Ltd.
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


logger = logging.getLogger(__name__)


class PDRatioThroughputOptimizer:
    """Optimizer for Prefill-Decode ratio throughput optimization.

    This optimizer combines independent P and D optimization results,
    calculates QPS and PD ratio, and outputs Top N configurations.

    QPS Formulas:
        P QPS = p_concurrency / ttft * 1000 (req/s)
        D QPS = d_concurrency / (tpot * output_length) * 1000 (req/s)

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
            DataFrame with PD ratio results sorted by balanced_qps in descending order.
        """
        if self._p_df is None or self._p_df.empty:
            self._result_df = pd.DataFrame()
            return self._result_df

        if self._d_df is None or self._d_df.empty:
            self._result_df = pd.DataFrame()
            return self._result_df

        # Calculate QPS using vectorized operations
        # P QPS = p_concurrency / ttft * 1000 (req/s)
        # Filter out zero ttft to avoid ZeroDivisionError
        p_df = self._p_df.copy()
        p_df = p_df[p_df["ttft"] > 0]
        p_df["p_qps"] = p_df["concurrency"] / p_df["ttft"] * 1000
        p_df = p_df[p_df["p_qps"] > 0]

        # D QPS = d_concurrency / (tpot * output_length) * 1000 (req/s)
        # Filter out zero tpot to avoid ZeroDivisionError
        d_df = self._d_df.copy()
        d_df = d_df[d_df["tpot"] > 0]
        d_df["d_qps"] = d_df["concurrency"] / (d_df["tpot"] * self.output_length) * 1000
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
        ]
        self._result_df = merged[result_cols].sort_values(by="balanced_qps", ascending=False).reset_index(drop=True)

        return self._result_df
