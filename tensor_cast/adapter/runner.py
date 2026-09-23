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

import dataclasses
from typing import Any

from tensor_cast.core.input_generator import generate_inputs
from tensor_cast.core.model_runner import ModelRunner

from .actual import ActualSummary, build_actual_summary_from_events


@dataclasses.dataclass(frozen=True)
class ActualRunResult:
    metrics: Any
    summary: ActualSummary


def run_simulation_case(user_input: Any, case_name: str = "default") -> ActualRunResult:
    """Run one simulation case and summarize runtime op invocations.

    The ModelRunner may invoke the observer once per runtime (for example one
    per pipeline stage); events and total time are accumulated across calls.
    """
    runner = ModelRunner(user_input)
    runtime_events = []
    observed_runtime = False
    perf_model_name = None
    total_forward_time_s = None

    def collect_summary(runtime):
        nonlocal observed_runtime, perf_model_name, total_forward_time_s
        observed_runtime = True
        runtime_events.extend(runtime.event_list)
        if perf_model_name is None and runtime.perf_models:
            perf_model_name = runtime.perf_models[0].name
        if perf_model_name is not None:
            runtime_total = runtime.total_execution_time_s().get(perf_model_name)
            if runtime_total is not None:
                total_forward_time_s = (0.0 if total_forward_time_s is None else total_forward_time_s) + runtime_total

    metrics = runner.run_inference(generate_inputs_func=generate_inputs, runtime_observer=collect_summary)
    if not observed_runtime:
        raise RuntimeError("ModelRunner did not provide runtime events for actual summary collection.")
    summary = build_actual_summary_from_events(
        runtime_events,
        case_name=case_name,
        perf_model_name=perf_model_name,
        total_forward_time_s=total_forward_time_s,
    )
    return ActualRunResult(metrics=metrics, summary=summary)
