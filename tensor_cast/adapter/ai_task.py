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
from typing import Any, Dict, List, Optional


@dataclasses.dataclass(frozen=True)
class AiAssistanceTask:
    task_type: str
    title: str
    summary: str
    model_type: Optional[str]
    evidence: Dict[str, Any]
    suspected_locations: List[Dict[str, Any]]
    constraints: List[str]
    required_output: List[str]
    verification_commands: List[str]
    prompt_text: str

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)
