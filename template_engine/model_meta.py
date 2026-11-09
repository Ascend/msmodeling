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

"""Capture-time model metadata for template-backed serving.

The template graph decides WHICH structure serves a config; serving a
config WITHOUT modeling or compilation additionally needs the model's
scalar metadata - weight size, kv per-token cost, kv token capacity
factor and the ``model_config`` object the strategy reads its parallel
axes from.  ``capture_ops`` records one sidecar ``<capture>.json.xz.meta``
per capture (the runner is built anyway during capture, so the metadata
is free there); ``build_template_graph`` aggregates the sidecars into
``meta["model_meta"]`` keyed by the weight key.

A weight key pins every axis that changes the weights or the kv layout:
TP/EP/MOE-DP/DCP, the MTP level and both quant actions.  DFC, compile
state, decode/prefill and (B, S) never change them, so one capture
covers every structure variant of its weight key.
"""

from __future__ import annotations

from typing import Any


def weight_key(
    tp: int,
    ep: int,
    moe_dp: int,
    dcp: int,
    num_mtp_tokens: int,
    quantize_linear_action: Any,
    quantize_attention_action: Any,
) -> str:
    """Stable identity of everything that changes weights/kv layout."""

    return (
        f"tp{int(tp)}_ep{int(ep)}_md{int(moe_dp)}_dcp{int(dcp)}"
        f"_mtp{int(num_mtp_tokens)}_{quantize_linear_action}_{quantize_attention_action}"
    )


def user_weight_key(user_input: Any) -> str:
    """Weight key from a ``UserInputConfig`` (defensive field access)."""

    return weight_key(
        getattr(user_input, "tp_size", 1),
        getattr(user_input, "ep_size", 1),
        getattr(user_input, "moe_dp_size", 1),
        getattr(user_input, "dcp_size", 1),
        getattr(user_input, "num_mtp_tokens", 0) or 0,
        getattr(user_input, "quantize_linear_action", "DISABLED"),
        getattr(user_input, "quantize_attention_action", "DISABLED"),
    )
