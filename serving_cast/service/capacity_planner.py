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

"""Capacity planning: given target QPS + SLO, find minimum device count.

Two-phase approach:
  Phase 0: determine search range — the user-specified device count, or the full
           power-of-2 budget sequence derived from the device comm_grid
  Phase 1: run the existing optimizer at every budget in that range and merge
           the summaries, reporting budgets that cannot run the model
  Phase 2: extrapolate per-instance QPS to compute total devices needed

The whole budget range is searched rather than stopping at the first feasible
budget: per-die QPS is not monotonic in device count, so a larger budget can
still produce the global minimum total device count.

Core assumption: DP replicas are independent (no inter-replica communication),
so per-instance QPS = total_qps / DP, and throughput scales linearly with
replica count.
"""

from __future__ import annotations

import copy
import logging
import math
import sys
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from serving_cast.service.optimizer_summary import (
    _compute_disagg_request_qps,
    compute_request_qps,
)

logger = logging.getLogger(__name__)


def _resource_failure_types() -> tuple[type[BaseException], ...]:
    """Exception types meaning "this device count cannot run the model".

    A worker pool crash (typically the OOM killer) or an out-of-memory error is
    an expected outcome of a budget that is too small, so capacity planning
    escalates instead of aborting. Every other exception is a defect or a
    deliberate fail-loud validation error and must reach the caller.

    torch is looked up in ``sys.modules`` rather than imported: it is already
    loaded by ``ParallelRunner`` whenever these failures can occur, so this stays
    cheap for importers that never run the optimizer.

    Returns:
        Tuple of exception classes to treat as resource failures.
    """
    from concurrent.futures.process import BrokenProcessPool

    types: tuple[type[BaseException], ...] = (BrokenProcessPool, MemoryError)
    oom = getattr(sys.modules.get("torch"), "OutOfMemoryError", None)
    if isinstance(oom, type) and issubclass(oom, BaseException):
        types += (oom,)
    return types


@dataclass
class CapacityPlanConfig:
    target_qps: float
    ttft_limit: Optional[float] = None
    tpot_limit: Optional[float] = None
    mode: str = "aggregation"
    output_length: int = 512
    num_devices: Optional[int] = None
    device: str = "TEST_DEVICE"
    #: PD-ratio auto mode per-instance device budgets. ``None`` derives them from
    #: the device comm_grid; ``--tp-sizes`` is deliberately not used here because a
    #: TP candidate describes the structure *inside* an instance, not its size.
    pd_instance_budgets: Optional[list] = None
    prefill_devices_per_instance: Optional[int] = None
    decode_devices_per_instance: Optional[int] = None


@dataclass
class CapacityPlanResult:
    minimum_devices: Optional[int] = None
    achieved_qps: float = 0.0
    per_die_qps: float = 0.0
    replicas_needed: Optional[int] = None
    best_config: dict = field(default_factory=dict)
    scaling_table: list[dict] = field(default_factory=list)
    pd_ratio_info: Optional[dict] = None


class CapacityPlanner:
    """Capacity planning meta-optimizer.

    Wraps existing ParallelRunner: runs the forward optimizer once at each
    candidate device count (a single user-specified count, or the auto-derived
    budget sequence), then extrapolates per-instance QPS to compute the minimum
    total devices that meet the target QPS under SLO.
    """

    def __init__(self, config: CapacityPlanConfig):
        self.config = config

    def run(self, args) -> CapacityPlanResult:
        if self.config.mode == "pd_ratio":
            return self._run_pd_ratio(args)
        return self._run_single_mode(args)

    # ------------------------------------------------------------------
    # Phase 0: determine search range
    # ------------------------------------------------------------------

    def _comm_grid_capacity(self) -> int:
        """Largest device count the configured device profile can wire up.

        ``ParallelRunner`` raises ``ValueError`` above this bound, so capacity
        planning treats it as a hard feasibility limit rather than a budget to
        try. This mirrors ``ParallelRunner.__init__`` exactly.

        Returns:
            ``comm_grid.grid.nelement()`` for ``config.device``.
        """
        from tensor_cast.device import DeviceProfile

        return DeviceProfile.all_device_profiles[self.config.device].comm_grid.grid.nelement()

    def _auto_budget_sequence(self) -> list[int]:
        """Power-of-2 budget sequence from the comm_grid seed up to its capacity.

        The seed is the second-innermost grid dimension (multi-dimensional
        grids) or the only dimension (1-D grids), floored at 2. The whole
        sequence is searched rather than stopping at the first feasible budget:
        per-die QPS is not monotonic in device count, so a larger budget can
        still hold the global minimum. The sequence is finite and bounded by the
        grid capacity.

        Returns:
            Ascending budget list; empty when the seed exceeds the capacity.
        """
        from tensor_cast.device import DeviceProfile

        device_profile = DeviceProfile.all_device_profiles[self.config.device]
        comm_grid_max = self._comm_grid_capacity()

        grid_shape = device_profile.comm_grid.grid.shape
        if len(grid_shape) >= 2:
            start = int(grid_shape[-2])
        else:
            start = int(grid_shape[0])
        if start < 2:
            start = 2

        budgets = []
        budget = start
        while budget <= comm_grid_max:
            budgets.append(budget)
            budget *= 2
        if budgets and budgets[-1] != comm_grid_max:
            budgets.append(comm_grid_max)
        if not budgets:
            logger.error(
                "comm_grid capacity %d < start %d for %s, cannot search.",
                comm_grid_max,
                start,
                self.config.device,
            )
        return budgets

    def _pd_instance_budgets(self) -> list[int]:
        """Per-instance device budgets to search in PD-ratio auto mode.

        A PD instance budget is the number of devices in one prefill *or* one
        decode instance, so it cannot be derived from ``--tp-sizes``: a TP
        candidate is a structure *inside* an instance, and reusing it as the
        instance size caps every instance at the largest TP (8 by default),
        making a model that needs 16/32-device instances look infeasible even
        though the comm_grid supports those budgets. It also hides PP>1
        topologies, where the instance footprint is ``TP x PP`` rather than TP.

        Budgets therefore come from the device comm_grid. ``--tp-sizes`` /
        ``--pp-sizes`` / ``--ep-sizes`` keep their meaning: they select the
        parallel structure searched *within* each budget. When ``--num-devices``
        is given it also bounds the derived budgets, since an instance larger
        than the user's stated device budget is never intended. Explicit
        ``config.pd_instance_budgets`` are taken as-is, bounded only by the grid.

        Returns:
            Ascending per-instance budgets; empty when nothing is runnable.
        """
        if self.config.pd_instance_budgets is not None:
            capacity = self._comm_grid_capacity()
            requested = sorted({int(budget) for budget in self.config.pd_instance_budgets if int(budget) > 0})
            runnable = [budget for budget in requested if budget <= capacity]
            if not runnable:
                logger.error(
                    "PD per-instance budgets %s all exceed comm_grid capacity %d for %s, cannot search.",
                    requested,
                    capacity,
                    self.config.device,
                )
            return runnable

        budgets = self._auto_budget_sequence()
        if self.config.num_devices is not None:
            cap = int(self.config.num_devices)
            budgets = [budget for budget in budgets if budget <= cap] or [cap]
        return budgets

    def _determine_search_range(self) -> list[int]:
        """Return the total-device-count budgets to search, ascending.

        If ``num_devices`` is specified, returns ``[N]`` (single run); otherwise
        the comm_grid-derived sequence (see :meth:`_auto_budget_sequence`).

        Returns:
            Ascending budget list; empty when the requested device count cannot
            be satisfied by the device profile (no runnable budget).
        """
        comm_grid_max = self._comm_grid_capacity()

        if self.config.num_devices is not None:
            n = int(self.config.num_devices)
            if n > comm_grid_max:
                logger.error(
                    "num_devices=%d exceeds comm_grid capacity %d for %s, cannot run.",
                    n,
                    comm_grid_max,
                    self.config.device,
                )
                return []
            return [n]

        return self._auto_budget_sequence()

    def _run_optimizer_at_n(self, args, n: int) -> Optional[pd.DataFrame]:
        """Run existing optimizer at device count n, return merged summary df.

        Returns:
            Merged summary rows, or None when this device count cannot run the
            model. Only resource failures and unwireable device counts are
            reported as "cannot run": other exceptions are defects or deliberate
            fail-loud validation errors and propagate to the caller.
        """
        from serving_cast.parallel_runner import ParallelRunner

        capacity = self._comm_grid_capacity()
        if n > capacity:
            # Doomed: ParallelRunner would reject this outright. Report it as an
            # infeasible budget instead of spending a run to discover that.
            logger.warning(
                "Skipping %d devices: exceeds %s comm_grid capacity %d.",
                n,
                self.config.device,
                capacity,
            )
            return None

        run_args = copy.copy(args)
        run_args.num_devices = n
        if isinstance(run_args.device, list):
            run_args.device = run_args.device[0]

        try:
            runner = ParallelRunner(run_args)
            if self.config.mode == "aggregation":
                summaries = runner.run_agg()
            else:
                summaries = runner.run_disagg()
        except _resource_failure_types() as exc:
            # A worker killed for memory, or an out-of-memory error, means this
            # device count is infeasible — escalate to the next budget. Anything
            # else (KeyError, ValueError from validation, ...) is a real defect
            # and must not be disguised as OOM.
            logger.warning(
                "Optimizer could not run at %d devices (%s: %s), escalating.",
                n,
                type(exc).__name__,
                exc,
            )
            logger.debug("Optimizer failure detail at %d devices", n, exc_info=True)
            return None

        dfs = []
        for s in summaries:
            df = s.get_summary_df()
            if df is not None and not df.empty:
                dfs.append(df)

        if not dfs:
            return None
        return pd.concat(dfs, axis=0, ignore_index=True)

    def _filter_slo(
        self,
        df: pd.DataFrame,
        ttft_col: str,
        tpot_col: str,
        fail_closed: bool = True,
    ) -> pd.DataFrame:
        """Generic SLO filter for any ttft/tpot column names.

        When fail_closed=True, missing column → no rows pass.
        When fail_closed=False, missing column → condition skipped.
        """
        mask = pd.Series(True, index=df.index)
        if self.config.ttft_limit is not None:
            if ttft_col not in df.columns:
                if fail_closed:
                    return df.iloc[0:0].copy()
            else:
                ttft = pd.to_numeric(df[ttft_col], errors="coerce")
                mask &= ttft.fillna(float("inf")) <= self.config.ttft_limit
        if self.config.tpot_limit is not None:
            if tpot_col not in df.columns:
                if fail_closed:
                    return df.iloc[0:0].copy()
            else:
                tpot = pd.to_numeric(df[tpot_col], errors="coerce")
                mask &= tpot.fillna(float("inf")) <= self.config.tpot_limit
        return df[mask].copy()

    def _filter_slo_compliant(self, df: pd.DataFrame) -> pd.DataFrame:
        """SLO filter for aggregation/disagg mode (ttft/tpot columns)."""
        return self._filter_slo(df, "ttft", "tpot", fail_closed=True)

    def _filter_slo_compliant_pd(self, df: pd.DataFrame) -> pd.DataFrame:
        """SLO filter for PD ratio mode (ttft_p/tpot_d columns)."""
        return self._filter_slo(df, "ttft_p", "tpot_d", fail_closed=True)

    # ------------------------------------------------------------------
    # Phase 2: extrapolate per-instance QPS
    # ------------------------------------------------------------------

    def _extrapolate_agg_disagg(
        self,
        df: pd.DataFrame,
        output_length: int,
    ) -> list[dict]:
        """Extract per-instance QPS and compute total devices for each row.

        Replication unit depends on EP topology:
        - EP within a single DP replica (ep_size <= num_devices // dp):
          DP replicas are independent → per-DP-replica extrapolation.
        - EP spans across DP replicas (num_devices // dp < ep_size <= num_devices):
          entire instance is the replication unit → instance-level extrapolation.
        - EP exceeds total devices (ep_size > num_devices): invalid, skip.
        """
        rows = []
        for _, row in df.iterrows():
            dp = int(row.get("dp_size", 1) or 1)
            num_devices = int(row.get("num_devices", 0) or 0)
            if dp <= 0 or num_devices <= 0:
                continue

            ep_size = int(row.get("ep_size", 1) or 1)
            per_dp_gpus = num_devices // dp

            if ep_size > num_devices:
                logger.warning(
                    "Skipping config %s: EP=%d exceeds total devices=%d.",
                    row.get("parallel", ""),
                    ep_size,
                    num_devices,
                )
                continue

            if ep_size > per_dp_gpus:
                per_replica_gpus = num_devices
                replication_unit = "instance"
            else:
                per_replica_gpus = per_dp_gpus
                replication_unit = "dp-replica"

            total_req_qps = _compute_disagg_request_qps(
                row,
                output_length,
            )
            if total_req_qps is None:
                token_s = pd.to_numeric(row.get("token/s"), errors="coerce")
                if pd.isna(token_s) or token_s <= 0 or output_length <= 0:
                    continue
                total_req_qps = compute_request_qps(float(token_s), output_length)

            if replication_unit == "instance":
                per_replica_req_qps = total_req_qps
            else:
                per_replica_req_qps = total_req_qps / dp

            if per_replica_req_qps <= 0:
                continue

            replicas_needed = math.ceil(
                self.config.target_qps / per_replica_req_qps,
            )
            total_gpus = replicas_needed * per_replica_gpus
            per_die_qps = per_replica_req_qps / per_replica_gpus

            rows.append(
                {
                    "parallel": row.get("parallel", ""),
                    # Full precision on purpose: these feed ceil() for the replica
                    # count and are multiplied back up for achieved QPS, so rounding
                    # here would scale the error by the replica count (a 0.014 QPS
                    # instance would report 0.72 instead of 1.008 at 72 replicas).
                    # Rounding happens only when rendering.
                    "per_replica_qps": per_replica_req_qps,
                    "per_replica_gpus": per_replica_gpus,
                    "replicas_needed": replicas_needed,
                    "total_gpus": total_gpus,
                    "per_die_qps": per_die_qps,
                    "ttft": float(row.get("ttft", 0) or 0),
                    "tpot": float(row.get("tpot", 0) or 0),
                    "batch_size": int(row.get("batch_size", 0) or 0),
                    "concurrency": int(row.get("concurrency", 0) or 0),
                }
            )
        return rows

    def _extrapolate_pd_ratio(
        self,
        df: pd.DataFrame,
    ) -> list[dict]:
        """Extrapolate PD ratio results: P/D independent scaling.

        p_inst = ceil(target_qps / p_qps)
        d_inst = ceil(target_qps / d_qps)
        total_gpus = p_inst * p_dev + d_inst * d_dev
        """
        rows = []
        for _, row in df.iterrows():
            p_qps = pd.to_numeric(row.get("p_qps"), errors="coerce")
            d_qps = pd.to_numeric(row.get("d_qps"), errors="coerce")
            if pd.isna(p_qps) or pd.isna(d_qps) or p_qps <= 0 or d_qps <= 0:
                continue

            p_dev = int(row.get("num_devices_p", 0) or 0)
            d_dev = int(row.get("num_devices_d", 0) or 0)
            if p_dev <= 0 or d_dev <= 0:
                continue

            p_inst = math.ceil(self.config.target_qps / float(p_qps))
            d_inst = math.ceil(self.config.target_qps / float(d_qps))
            total_gpus = p_inst * p_dev + d_inst * d_dev
            achieved_qps = min(p_inst * float(p_qps), d_inst * float(d_qps))
            per_die_qps = achieved_qps / total_gpus if total_gpus > 0 else 0

            rows.append(
                {
                    "parallel_p": row.get("parallel_p", ""),
                    "parallel_d": row.get("parallel_d", ""),
                    # Full precision: p_inst/d_inst and achieved_qps are derived from
                    # these, and rounding before the multiplication re-introduces the
                    # scaled error described in _extrapolate_agg_disagg. Rendering
                    # formats to two decimals.
                    "p_qps": float(p_qps),
                    "d_qps": float(d_qps),
                    "p_inst": p_inst,
                    "d_inst": d_inst,
                    "total_gpus": total_gpus,
                    "achieved_qps": achieved_qps,
                    "per_die_qps": per_die_qps,
                    "ttft_p": float(row.get("ttft_p", 0) or 0),
                    "tpot_d": float(row.get("tpot_d", 0) or 0),
                    "pd_ratio": float(row.get("pd_ratio", 0) or 0),
                }
            )
        return rows

    def _rank_and_select(
        self,
        scaling_rows: list[dict],
    ) -> tuple[Optional[dict], list[dict]]:
        """Sort by total_gpus ascending, tiebreak per_die_qps descending."""
        if not scaling_rows:
            return None, []
        ranked = sorted(
            scaling_rows,
            key=lambda r: (r["total_gpus"], -r["per_die_qps"]),
        )
        return ranked[0], ranked

    # ------------------------------------------------------------------
    # Main entry: single-mode (aggregation / disagg)
    # ------------------------------------------------------------------

    def _run_single_mode(self, args) -> CapacityPlanResult:
        output_length = self.config.output_length
        budgets = self._determine_search_range()

        # Every budget is searched and the rows merged, rather than stopping at
        # the first feasible one: per-die QPS is not monotonic in device count,
        # so a larger budget can still yield the global minimum total devices.
        if not budgets:
            logger.error(
                "No device-count budget to search for %s (see the preceding error for the cause).",
                self.config.device,
            )
            return CapacityPlanResult(minimum_devices=None, scaling_table=[])

        all_rows = []
        tried_budgets = []
        had_any_rows = False
        had_slo_compliant = False

        for budget in budgets:
            logger.info("Capacity planning: trying %d devices.", budget)
            tried_budgets.append(budget)
            df = self._run_optimizer_at_n(args, budget)
            if df is None or df.empty:
                logger.info(
                    "No feasible results at %d devices (model does not fit, or the run failed).",
                    budget,
                )
                continue

            had_any_rows = True

            slo_df = self._filter_slo_compliant(df)
            if slo_df.empty:
                logger.info(
                    "No SLO-compliant configs at %d devices.",
                    budget,
                )
                continue

            had_slo_compliant = True
            rows = self._extrapolate_agg_disagg(slo_df, output_length)
            all_rows.extend(rows)

        best, ranked = self._rank_and_select(all_rows)
        if best is None:
            if not had_any_rows:
                logger.error(
                    "Model does not fit at any searched budget. Tried: %s",
                    tried_budgets,
                )
            elif not had_slo_compliant:
                logger.error(
                    "Rows found but none SLO-compliant. Tried: %s. "
                    "Consider relaxing SLO limits or increasing --num-devices.",
                    tried_budgets,
                )
            else:
                logger.error(
                    "No feasible configuration after extrapolation. Tried: %s",
                    tried_budgets,
                )
            return CapacityPlanResult(
                minimum_devices=None,
                scaling_table=[],
            )

        return CapacityPlanResult(
            minimum_devices=best["total_gpus"],
            achieved_qps=best["per_replica_qps"] * best["replicas_needed"],
            per_die_qps=best["per_die_qps"],
            replicas_needed=best["replicas_needed"],
            best_config=best,
            scaling_table=ranked,
        )

    # ------------------------------------------------------------------
    # PD ratio mode
    # ------------------------------------------------------------------

    def _run_pd_ratio(self, args) -> CapacityPlanResult:
        p_dev = self.config.prefill_devices_per_instance
        d_dev = self.config.decode_devices_per_instance

        if p_dev is not None and d_dev is not None:
            return self._run_pd_ratio_fixed(args, p_dev, d_dev)
        return self._run_pd_ratio_auto(args)

    def _run_pd_ratio_fixed(
        self,
        args,
        p_dev: int,
        d_dev: int,
    ) -> CapacityPlanResult:
        run_args = copy.copy(args)
        run_args.num_devices = p_dev + d_dev

        df = self._run_optimizer_at_n(run_args, p_dev + d_dev)
        if df is None or df.empty:
            logger.error(
                "PD ratio: no results at P=%d D=%d (does not fit, exceeds comm_grid capacity, or the run failed).",
                p_dev,
                d_dev,
            )
            return CapacityPlanResult(minimum_devices=None)

        slo_df = self._filter_slo_compliant_pd(df)
        if slo_df.empty:
            logger.error(
                "PD ratio: rows found but none SLO-compliant at P=%d D=%d.",
                p_dev,
                d_dev,
            )
            return CapacityPlanResult(minimum_devices=None)

        rows = self._extrapolate_pd_ratio(slo_df)
        best, ranked = self._rank_and_select(rows)
        if best is None:
            logger.error(
                "PD ratio: no feasible configuration after extrapolation at P=%d D=%d.",
                p_dev,
                d_dev,
            )
            return CapacityPlanResult(minimum_devices=None)

        return CapacityPlanResult(
            minimum_devices=best["total_gpus"],
            achieved_qps=best["achieved_qps"],
            per_die_qps=best["per_die_qps"],
            replicas_needed=None,
            best_config=best,
            scaling_table=ranked,
            pd_ratio_info=best,
        )

    def _run_pd_ratio_auto(self, args) -> CapacityPlanResult:
        budgets = self._pd_instance_budgets()
        if not budgets:
            logger.error(
                "PD auto-search: no per-instance device budget to search for %s "
                "(see the preceding error for the cause).",
                self.config.device,
            )
            return CapacityPlanResult(minimum_devices=None)

        # P and D instance sizes are enumerated independently; each pair is one
        # optimizer run, which is why the budget list is derived from the device
        # comm_grid rather than from --tp-sizes (see _pd_instance_budgets).
        candidates = [(p, d) for p in budgets for d in budgets]
        logger.info(
            "PD auto-search: %d per-instance budget(s) %s -> %d (P, D) pairs.",
            len(budgets),
            budgets,
            len(candidates),
        )
        all_rows = []
        had_infeasible = False
        had_rows_no_slo = False
        for p_dev, d_dev in candidates:
            logger.info("PD auto-search: P=%d, D=%d", p_dev, d_dev)
            run_args = copy.copy(args)
            run_args.prefill_devices_per_instance = p_dev
            run_args.decode_devices_per_instance = d_dev
            run_args.num_devices = p_dev + d_dev

            df = self._run_optimizer_at_n(run_args, p_dev + d_dev)
            if df is None or df.empty:
                had_infeasible = True
                logger.info(
                    "PD auto-search P=%d D=%d: no results "
                    "(does not fit, exceeds comm_grid capacity, or the run failed).",
                    p_dev,
                    d_dev,
                )
                continue

            slo_df = self._filter_slo_compliant_pd(df)
            if slo_df.empty:
                had_rows_no_slo = True
                logger.info(
                    "PD auto-search P=%d D=%d: rows found but none SLO-compliant.",
                    p_dev,
                    d_dev,
                )
                continue
            rows = self._extrapolate_pd_ratio(slo_df)
            all_rows.extend(rows)

        best, ranked = self._rank_and_select(all_rows)
        if best is None:
            if had_infeasible and not had_rows_no_slo:
                logger.error(
                    "PD auto-search: no (P, D) pair produced results. Each either "
                    "exceeded the device comm_grid capacity, did not fit, or failed "
                    "to run. Consider pinning --prefill-devices-per-instance / "
                    "--decode-devices-per-instance.",
                )
            elif had_rows_no_slo and not had_infeasible:
                logger.error(
                    "PD auto-search: rows found but none SLO-compliant. "
                    "Consider relaxing SLO limits (--ttft-limit / --tpot-limit).",
                )
            elif had_infeasible and had_rows_no_slo:
                logger.error(
                    "PD auto-search: some (P, D) pairs produced no results, others had "
                    "rows but none SLO-compliant. Consider pinning the per-instance "
                    "device counts and/or relaxing SLO limits.",
                )
            else:
                logger.error(
                    "PD auto-search: no feasible configuration after extrapolation.",
                )
            return CapacityPlanResult(minimum_devices=None)

        return CapacityPlanResult(
            minimum_devices=best["total_gpus"],
            achieved_qps=best["achieved_qps"],
            per_die_qps=best["per_die_qps"],
            replicas_needed=None,
            best_config=best,
            scaling_table=ranked,
            pd_ratio_info=best,
        )


def render_capacity_planning_result(
    result: CapacityPlanResult,
    config: CapacityPlanConfig,
    *,
    logger: logging.Logger,
) -> None:
    """Print capacity planning summary and scaling comparison table."""

    from prettytable import PrettyTable

    width = 100
    print()
    print("*" * width)
    print("  Capacity Planning Result")
    print("  " + "-" * (width - 4))
    print(f"  Target QPS:  {config.target_qps:.2f} req/s")
    slo_parts = []
    if config.ttft_limit is not None:
        slo_parts.append(f"TTFT <= {config.ttft_limit:.0f} ms")
    if config.tpot_limit is not None:
        slo_parts.append(f"TPOT <= {config.tpot_limit:.0f} ms")
    print(f"  SLO:         {', '.join(slo_parts) if slo_parts else 'N/A'}")
    print("  " + "-" * (width - 4))

    if result.minimum_devices is None:
        print("  No feasible configuration found under the given constraints.")
        print("*" * width)
        return

    best = result.best_config
    print(f"  Minimum Devices: {result.minimum_devices}")
    print(f"  Achieved QPS:     {result.achieved_qps:.2f} req/s")
    print(f"  Per-Die QPS:      {result.per_die_qps:.2f} req/s")

    if result.replicas_needed is not None:
        print(f"  Replicas Needed:  {result.replicas_needed}")
        print(f"  Per-Replica Dies: {best.get('per_replica_gpus', 'N/A')}")
        print(f"  Parallel:         {best.get('parallel', 'N/A')}")
        print(f"  Batch Size:       {best.get('batch_size', 'N/A')}")
        if best.get("ttft"):
            print(f"  TTFT:             {best['ttft']:.2f} ms")
        if best.get("tpot"):
            print(f"  TPOT:             {best['tpot']:.2f} ms")
    elif result.pd_ratio_info is not None:
        pd_info = result.pd_ratio_info
        print(
            f"  Topology:         {pd_info['p_inst']}P{pd_info['d_inst']}D "
            f"(P{pd_info.get('parallel_p', '').split('|')[0].strip().split('=')[-1] if pd_info.get('parallel_p') else 'N/A'} "
            f"D{pd_info.get('parallel_d', '').split('|')[0].strip().split('=')[-1] if pd_info.get('parallel_d') else 'N/A'})"
        )
        print(f"  P QPS:            {pd_info['p_qps']:.2f} req/s")
        print(f"  D QPS:            {pd_info['d_qps']:.2f} req/s")
        print(f"  PD Ratio:         {pd_info['pd_ratio']:.2f}")

    print("  " + "-" * (width - 4))

    if result.scaling_table:
        print("  推算对比表 (按总卡数升序, 同卡数取单卡QPS最高)")
        print("  " + "-" * (width - 4))
        table = PrettyTable()

        if result.replicas_needed is not None:
            table.field_names = [
                "并行配置",
                "每实例QPS",
                "每实例卡数",
                "所需实例数",
                "总卡数",
                "单卡QPS",
            ]
            for row in result.scaling_table:
                table.add_row(
                    [
                        row.get("parallel", ""),
                        f"{row['per_replica_qps']:.2f}",
                        row["per_replica_gpus"],
                        row["replicas_needed"],
                        row["total_gpus"],
                        f"{row['per_die_qps']:.2f}",
                    ]
                )
        elif result.pd_ratio_info is not None:
            table.field_names = [
                "P并行",
                "D并行",
                "P QPS",
                "D QPS",
                "P实例",
                "D实例",
                "总卡数",
                "单卡QPS",
            ]
            for row in result.scaling_table:
                table.add_row(
                    [
                        row.get("parallel_p", ""),
                        row.get("parallel_d", ""),
                        f"{row['p_qps']:.2f}",
                        f"{row['d_qps']:.2f}",
                        row["p_inst"],
                        row["d_inst"],
                        row["total_gpus"],
                        f"{row['per_die_qps']:.2f}",
                    ]
                )
        print(table.get_string())

    print("*" * width)
    print()
