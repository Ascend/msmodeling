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

"""Restore one fused Roofline across split analytic MLA runtime events."""

from typing import List, Optional, Sequence

from .base import PerformanceModel
from .bound_analyzer import BoundAnalyzer
from .op_invoke_info import OpInvokeInfo


_CORE_MARKERS = (
    "multihead_latent_attention",
    "mla_sparse_attention",
)
_BUNDLE_END_MARKER = "mla_merge_phase_outputs"
_STATIC_COST_KEY = "static_cost_time_s"


def mla_family_role(op_invoke_info: OpInvokeInfo) -> Optional[str]:
    name = str(op_invoke_info.func)
    if _BUNDLE_END_MARKER in name:
        return "bundle_end"
    if "mla_kv_projection" in name:
        return "prefill_projection"
    if "mla_q_absorb_projection" in name or "mla_v_up_projection" in name:
        return "decode_projection"
    if any(marker in name for marker in _CORE_MARKERS):
        return "core"
    return None


def _copy_stats(stats):
    if not isinstance(stats, dict):
        return stats
    return {key: _copy_stats(value) for key, value in stats.items()}


def _scale_resource_times(stats, factor: float) -> None:
    """Scale resource times but not the one-off static launch cost."""
    if not isinstance(stats, dict):
        return
    for key, value in stats.items():
        if isinstance(value, dict):
            _scale_resource_times(value, factor)
        elif key != _STATIC_COST_KEY and isinstance(value, (int, float)) and not isinstance(value, bool):
            if str(key).endswith("_time_s"):
                stats[key] = value * factor


def rescale_analytic_mla_bundle(events: Sequence) -> None:
    """Rewrite one MLA family so its members sum to the fused Roofline."""
    # A calibrated component already contains its measured/interpolated final
    # latency.  Preserve independent FIA/SFA and projection timings; the
    # runtime scheduler sums these events in execution order.
    for model_name in ("analytic", "calibrated"):
        if model_name == "calibrated" and any(
            isinstance(event.perf_results.get("calibrated"), PerformanceModel.Result)
            and event.perf_results["calibrated"].statistics.get("source") == "ANALYTIC_CALIBRATED"
            for event in events
        ):
            continue
        members = []
        for event in events:
            result = event.perf_results.get(model_name)
            if result is None:
                continue
            result = PerformanceModel.Result(
                execution_time_s=result.execution_time_s,
                statistics=_copy_stats(result.statistics),
            )
            event.perf_results[model_name] = result
            components = BoundAnalyzer.components(result)
            members.append(
                (
                    mla_family_role(event.op_invoke_info),
                    result,
                    components.mma_ops_time_s + components.gp_ops_time_s,
                    components.memory_time_s,
                )
            )
        if len(members) < 2:
            continue
        prefill_m = sum(memory for role, _r, _c, memory in members if role == "prefill_projection")
        decode_m = sum(memory for role, _r, _c, memory in members if role == "decode_projection")
        other_m = sum(
            memory for role, _r, _c, memory in members if role not in {"prefill_projection", "decode_projection"}
        )
        work_s = max(sum(item[2] for item in members), other_m + max(prefill_m, decode_m))
        shares = [max(compute, memory) for _role, _result, compute, memory in members]
        total_share = sum(shares)
        if total_share <= 0:
            continue
        core_indices = [index for index, item in enumerate(members) if item[0] == "core"]
        static_index = core_indices[0] if core_indices else len(members) - 1
        static_cost_s = members[static_index][1].statistics.get(_STATIC_COST_KEY, 0.0)
        for index, ((_role, result, _compute, _memory), share) in enumerate(zip(members, shares)):
            member_work_s = work_s * share / total_share
            _scale_resource_times(result.statistics, member_work_s / share if share > 0 else 0.0)
            result.execution_time_s = member_work_s + (static_cost_s if index == static_index else 0.0)


class AnalyticMlaFusionRescaler:
    """Collect MLA family events during replay and rescale each finished family."""

    def __init__(self):
        self._pending: List = []

    def observe(self, event) -> None:
        role = mla_family_role(event.op_invoke_info)
        if role is None:
            return
        if role == "bundle_end":
            self.flush()
            return
        self._pending.append(event)

    def flush(self) -> None:
        rescale_analytic_mla_bundle(self._pending)
        self._pending.clear()
