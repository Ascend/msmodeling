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

"""Serving integration for adaptive speculative verification."""

import logging
import time
from dataclasses import replace

import pandas as pd

from tensor_cast.core.speculative_verification import (
    AcceptanceProfile,
    InfeasibleVerificationPlanError,
    PlanCost,
    select_verification_plan,
)

from .latency_table import ForwardShapeKey
from .optimizer_summary import OptimizerSummary


logger = logging.getLogger(__name__)


class AdaptiveVerificationStrategy:
    """Evaluate verification candidates and collect per-batch serving statistics."""

    def __init__(self, model_runner, forward_record_cache, evaluate_forward):
        if model_runner.model.is_vl_model:
            raise ValueError("Adaptive DSpark verification supports text-only models")
        self.profile = AcceptanceProfile.from_rates(
            model_runner.user_input.dspark_acceptance_rates, model_runner.model.model_config.draft_block_size() - 1
        )
        self.dp = model_runner.model.model_config.parallel_config.data_parallel_size
        self.steps = []
        self.early_stop_reason = None
        self._cache_hits = self._cache_misses = 0
        self._forward_record_cache = forward_record_cache
        self._evaluate_forward = evaluate_forward

    def run_range(self, data, left, right, get_inference_info):
        self.early_stop_reason = None
        rows = []
        result = OptimizerSummary(data)
        for batch in range(left, right + 1):
            data.batch_size = batch
            self.steps.clear()
            started = time.perf_counter()
            summary = get_inference_info(data)
            if summary.check_early_stop_flag():
                self.early_stop_reason = summary.get_early_stop_reason()
                break
            frame = summary.get_summary_df()
            if self.steps:
                for name, value in self.report().items():
                    frame[name] = [value] * len(frame)
                frame["simulation_wall_time_s"] = time.perf_counter() - started
            rows.append(frame)
            if summary.get_memory_info():
                result.set_memory_info(summary.get_memory_info())
        if not rows:
            return None
        result.set_summary_df(pd.concat(rows, ignore_index=True).sort_values("token/s", ignore_index=True))
        return result

    def forward_record(self, shapes, data):
        key = ForwardShapeKey(
            is_decode=True,
            model_concurrency=len(shapes) * self.dp,
            query_len=max(s.query_len for s in shapes),
            seq_len=max(s.seq_len for s in shapes),
            request_shapes=tuple((s.query_len, s.seq_len) for s in shapes),
        )
        hit = key in self._forward_record_cache
        self._cache_hits += hit
        self._cache_misses += not hit
        return self._evaluate_forward(key, data)

    def select_step(self, shapes, data, evaluate, remaining_outputs=None):
        profile = self.profile
        decode_indices = [i for i, shape in enumerate(shapes) if shape.is_decode]
        if not decode_indices:
            raise ValueError("adaptive verification was asked to plan a batch with no decode request")
        records = {}
        self._cache_hits = self._cache_misses = 0

        def cost(plan):
            actual = list(shapes)
            for i, q in zip(decode_indices, plan.query_lens):
                shape = shapes[i]
                actual[i] = replace(shape, query_len=q, seq_len=shape.seq_len - shape.query_len + q)
            actual = tuple(actual)
            record = evaluate(actual)
            records[plan.prefixes] = (actual, record)
            return PlanCost(record.latency_ms, record.memory_left_gb)

        try:
            selection = select_verification_plan(
                (profile.survival,) * len(decode_indices),
                cost,
                max_evaluations=data.dspark_adaptive_max_evaluations,
                tpot_limit_ms=data.tpot_limits,
                remaining_outputs=remaining_outputs,
            )
        except InfeasibleVerificationPlanError as exc:
            selection = exc.selection
            actual, record = records[selection.plan.prefixes]
            # Multi-step aggregation can have a P+D mixed step (dominated by prefill)
            # or a fractional tail step (remaining < 1) whose single-step latency/output
            # exceeds tpot_limits while end-to-end request TPOT remains feasible.
            can_amortize_step_tpot = (
                remaining_outputs is not None
                and selection.cost.memory_left_gb >= 0
                and selection.cost.latency_ms > 0
                and all(e > 0 for e in selection.plan.expected_outputs)
                and (any(not shape.is_decode for shape in shapes) or any(r < 1.0 for r in remaining_outputs))
            )
            if not can_amortize_step_tpot:
                return selection, actual, record
            selection = replace(selection, feasible=True)
        actual, record = records[selection.plan.prefixes]
        self.steps.append((selection, self._cache_hits, self._cache_misses))
        logger.info(
            "Adaptive DSpark: q=%s expected=%s latency_ms=%.5f candidates=%d cache_hits=%d",
            selection.plan.query_lens,
            selection.plan.expected_outputs,
            record.latency_ms,
            selection.evaluated_candidates,
            self._cache_hits,
        )
        return selection, actual, record

    def report(self):
        selections = [step[0] for step in self.steps]
        prefixes = [k for selected in selections for k in selected.plan.prefixes]
        outputs = [e for selected in selections for e in selected.plan.expected_outputs]
        n = len(self.profile.survival)
        return {
            "max_speculative_tokens": n,
            "target_total_rows": sum(prefixes) + len(prefixes),
            "expected_output_tokens": sum(outputs),
            "full_window_step_latency_mean_ms": sum(s.full_window_cost.latency_ms for s in selections)
            / len(selections),
            "selected_step_latency_mean_ms": sum(s.cost.latency_ms for s in selections) / len(selections),
            "verification_prefix_min": min(prefixes),
            "verification_prefix_mean": sum(prefixes) / len(prefixes),
            "verification_prefix_max": max(prefixes),
            "expected_output_per_request": sum(outputs) / len(outputs),
            "draft_verification_pruned_ratio": 1 - sum(prefixes) / (len(prefixes) * n),
            "verification_candidates": sum(s.evaluated_candidates for s in selections),
            "forward_cache_hits": sum(step[1] for step in self.steps),
            "additional_performance_evaluations": sum(step[2] for step in self.steps),
        }
