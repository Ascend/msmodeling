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


"""Startup candidate preparation; measured outcomes, not estimates, establish a baseline."""

from copy import deepcopy

from .pd import positive_integer, resource_nodes
from .pd_layouts import enumerate_layouts


def initial_layouts(config, params, *, tp_options=None, max_candidates=16):
    """Preserve supplied specs; otherwise try host-local TP sizes in increasing order.

    These are launch trials, not promises of model fit. Model/backend-specific
    restrictions still pass through the normal preflight and actual execution.
    """
    positive_integer(max_candidates, "startup candidates")
    nodes = resource_nodes(config)
    maximum = max((len(n.get("gpu_ids", [])) for n in nodes), default=0)
    options = tp_options or [2**n for n in range(maximum.bit_length()) if 2**n <= maximum]
    for option in options:
        positive_integer(option, "startup TP")
    choices = {}
    for role in ("prefill", "decode"):
        name = "TENSOR_PARALLEL_SIZE_" + role
        choices[role] = [positive_integer(params[name], name)] if name in params else options
    result = []
    for p in choices["prefill"]:
        for d in choices["decode"]:
            selected = {
                **params,
                "TENSOR_PARALLEL_SIZE_prefill": p,
                "TENSOR_PARALLEL_SIZE_decode": d,
            }
            layouts = enumerate_layouts(
                config,
                selected,
                use_full_device=False,
                max_candidates=max_candidates,
                max_combinations=10000,
            )
            for item in layouts["layouts"]:
                roles = item["layout"]["roles"]
                if all(roles[r]["instances"] == 1 for r in roles):
                    result.append({**item, "params": deepcopy(selected), "purpose": "bootstrap"})
                    # During startup, prioritize different resource specs over
                    # many placements of the same spec. Placement search follows.
                    break
    result.sort(key=lambda x: sum(r["devices"] for r in x["layout"]["roles"].values()))
    return {
        "candidates": result[:max_candidates],
        "coverage": "bounded startup choices, not exhaustive",
        "note": "TP feasibility and memory fit require preflight and real launch; retain backend defaults",
    }


def next_action(failure):
    """Return a bounded decision hint; never silently mutate a frozen parameter."""
    category = (failure or {}).get("category", "unknown")
    actions = {
        "oom": "try_larger_instance_or_lower_memory_pressure",
        "constraint": "repair_candidate_within_user_constraints",
        "timeout": "inspect_startup_and_network_evidence",
    }
    return {
        "action": actions.get(category, "inspect_evidence_before_retry"),
        "requires": "remaining bootstrap budget and completed cleanup",
    }
