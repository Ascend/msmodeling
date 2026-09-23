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

"""Human checkpoints for the streamlined adaptation flow.

Questions are derived only from the deterministic structure scan and the
candidate profile (public model information). There is no measured profiling
data to reconcile against.
"""

import dataclasses
from typing import Any, Dict, List, Optional

from .inspect import ModelStructureFacts, ProfileCandidate


def build_human_questions(
    structure: Optional[ModelStructureFacts] = None,
    candidate: Optional[ProfileCandidate] = None,
    limit: int = 8,
) -> List[Dict[str, Any]]:
    questions: List[Dict[str, Any]] = []

    if candidate is not None:
        for field in dataclasses.fields(candidate):
            value = getattr(candidate, field.name)
            if value is None or value.confidence != "low":
                continue
            questions.append(
                {
                    "kind": "confirm_candidate_field",
                    "priority": "medium",
                    "question": (
                        f"Candidate profile field {field.name!r} = {value.value!r} was derived with "
                        f"low confidence from {value.source!r}. Confirm it against the installed "
                        "model source before registering the profile, or override it explicitly."
                    ),
                    "evidence": {
                        "field": field.name,
                        "value": value.value,
                        "source": value.source,
                    },
                }
            )

    if structure is not None:
        if structure.moe_like_modules and (candidate is None or candidate.moe_num_experts_key is None):
            questions.append(
                {
                    "kind": "confirm_expert_key",
                    "priority": "high",
                    "question": (
                        f"{len(structure.moe_like_modules)} MoE-like modules were found but no expert-count "
                        "config key was located. Confirm the expert count key from the installed model "
                        "config and set moe_num_experts_key explicitly."
                    ),
                    "evidence": {
                        "moe_module_count": len(structure.moe_like_modules),
                        "expert_fields": structure.expert_fields,
                    },
                }
            )
        if structure.visual_module_paths and (candidate is None or candidate.visual_module_path is None):
            questions.append(
                {
                    "kind": "confirm_visual_fields",
                    "priority": "high",
                    "question": (
                        "Visual modules were found but no visual profile fields were derived. Confirm "
                        "the visual/language module paths and linear mappings from the installed "
                        "model source."
                    ),
                    "evidence": {"visual_module_paths": list(structure.visual_module_paths)},
                }
            )
        if "deepseek_like_mla" in structure.known_recipe_matches and (
            candidate is None or candidate.mla_module_name is None
        ):
            questions.append(
                {
                    "kind": "confirm_mla_fields",
                    "priority": "medium",
                    "question": (
                        "MLA-like attention modules were detected but no MLA candidate was derived. "
                        "Confirm the attention module name and field mapping from the installed "
                        "model source."
                    ),
                    "evidence": {"known_recipe_matches": list(structure.known_recipe_matches)},
                }
            )

    return questions[:limit]
