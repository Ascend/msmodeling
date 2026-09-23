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

"""Compile shape-mode decisions for the throughput optimizer.

The cache deliberately holds only scalar decision metadata.  Compiled runners
and runtime workloads remain process-local and are never persisted or shared
through this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any


COMPILE_MODE_RATIO_THRESHOLD = 1.5

_USER_INPUT_KEY_FIELDS = (
    "model_id",
    "remote_source",
    "performance_model",
    "world_size",
    "tp_size",
    "dp_size",
    "pp_size",
    "ep_size",
    "moe_tp_size",
    "moe_dp_size",
    "dcp_size",
    "o_proj_tp_size",
    "o_proj_dp_size",
    "mlp_tp_size",
    "mlp_dp_size",
    "lmhead_tp_size",
    "lmhead_dp_size",
    "vision_tp_size",
    "word_embedding_tp",
    "quantize_linear_action",
    "quantize_non_expert_linear_action",
    "quantize_attention_action",
    "quantize_lmhead",
    "mxfp4_group_size",
    "num_mtp_tokens",
    "mtp_acceptance_rate",
    "block_size",
    "allow_graph_break",
    "enable_multistream",
    "enable_sequence_parallel",
    "enable_matmul_allreduce",
    "enable_dispatch_ffn_combine",
    "enable_shared_expert_tp",
    "enable_external_shared_experts",
    "host_external_shared_experts",
    "enable_redundant_experts",
    "num_hidden_layers_override",
    "disable_repetition",
)


@dataclass(frozen=True)
class CompileDecisionKey:
    """Immutable identity for a scalar compile-mode decision."""

    digest: str

    @property
    def short_hash(self) -> str:
        return self.digest[:12]

    @classmethod
    def from_inputs(
        cls,
        user_input: Any,
        optimizer_data: Any,
        *,
        phase: str,
        probe_batch_size: int,
        is_decode: bool,
    ) -> "CompileDecisionKey":
        """Build a key from graph-affecting inputs and the representative shape.

        SLO values, optimizer outcomes, and device profile are intentionally
        excluded. They do not affect the compiled graph, and one CLI invocation
        deliberately reuses the first device's calibration across profiles.
        """
        if is_decode:
            query_len = user_input.num_mtp_tokens + 1
            seq_len = optimizer_data.output_length // 2 + optimizer_data.get_decode_context_length() + query_len
        else:
            query_len = optimizer_data.get_effective_input_length()
            seq_len = query_len
        dp_size = user_input.dp_size or 1
        pp_size = user_input.pp_size or 1
        values = {
            "phase": phase,
            "probe_batch_size": probe_batch_size,
            "probe_concurrency": probe_batch_size * dp_size * pp_size,
            "query_len": query_len,
            "seq_len": seq_len,
            "input_length": optimizer_data.input_length,
            "output_length": optimizer_data.output_length,
            "prefix_cache_hit_rate": optimizer_data.prefix_cache_hit_rate,
            "image_batch_size": optimizer_data.image_batch_size,
            "image_height": optimizer_data.image_height,
            "image_width": optimizer_data.image_width,
        }
        values.update({field: getattr(user_input, field, None) for field in _USER_INPUT_KEY_FIELDS})
        canonical = json.dumps(values, default=str, sort_keys=True, separators=(",", ":"))
        return cls(hashlib.sha256(canonical.encode("utf-8")).hexdigest())


@dataclass(frozen=True)
class CompileModeDecision:
    """Selected shape mode and the host-wall evidence used to select it."""

    dynamic_shapes: bool
    reason: str
    static_run_time_s: float | None = None
    dynamic_run_time_s: float | None = None
    ratio: float | None = None
    threshold: float = COMPILE_MODE_RATIO_THRESHOLD


class CompileModeDecisionCache:
    """Process-local cache of scalar decisions for one CLI run.

    This object intentionally contains only pickleable values because
    ``ParallelRunner`` is submitted to ``ProcessPoolExecutor`` workers on
    Windows. Concurrent cache misses may duplicate a calibration forward, but
    cannot change the selected mode or share live runner state.
    """

    def __init__(self) -> None:
        self._decisions: dict[CompileDecisionKey, CompileModeDecision] = {}

    def get(self, key: CompileDecisionKey) -> CompileModeDecision | None:
        return self._decisions.get(key)

    def set(self, key: CompileDecisionKey, decision: CompileModeDecision) -> None:
        self._decisions[key] = decision


def decide_compile_shape_mode(static_run_time_s: float, dynamic_run_time_s: float) -> CompileModeDecision:
    """Choose static only when dynamic host wall exceeds the fixed threshold."""
    if not math.isfinite(static_run_time_s) or static_run_time_s <= 0:
        return CompileModeDecision(
            dynamic_shapes=True,
            reason="invalid_static_probe_time",
            static_run_time_s=static_run_time_s,
            dynamic_run_time_s=dynamic_run_time_s,
        )
    if not math.isfinite(dynamic_run_time_s) or dynamic_run_time_s <= 0:
        return CompileModeDecision(
            dynamic_shapes=True,
            reason="invalid_dynamic_probe_time",
            static_run_time_s=static_run_time_s,
            dynamic_run_time_s=dynamic_run_time_s,
        )

    ratio = dynamic_run_time_s / static_run_time_s
    if ratio > COMPILE_MODE_RATIO_THRESHOLD:
        return CompileModeDecision(
            dynamic_shapes=False,
            reason="dynamic_static_ratio_exceeds_threshold",
            static_run_time_s=static_run_time_s,
            dynamic_run_time_s=dynamic_run_time_s,
            ratio=ratio,
        )
    return CompileModeDecision(
        dynamic_shapes=True,
        reason="dynamic_static_ratio_within_threshold",
        static_run_time_s=static_run_time_s,
        dynamic_run_time_s=dynamic_run_time_s,
        ratio=ratio,
    )
