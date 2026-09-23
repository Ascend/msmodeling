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

"""Typed, code-owned activation policies for conditional Theory operators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from tools.model_diagnostics.domain import ModelRunContext, TheoryOperatorSpec


@dataclass(frozen=True)
class OperatorActivationRequest:
    """Stable context available to an operator or enclosing-region policy."""

    spec_id: str
    model_category: str
    region_id: str
    stage_id: str
    operator: TheoryOperatorSpec | None
    context: ModelRunContext


class OperatorActivationPolicy(Protocol):
    policy_id: str

    def is_active(self, request: OperatorActivationRequest) -> bool: ...


class OperatorActivationRegistry:
    """Resolve stable YAML policy ids without dynamic imports or eval."""

    def __init__(self) -> None:
        self._policies: dict[str, OperatorActivationPolicy] = {}

    def register(self, policy: OperatorActivationPolicy) -> None:
        policy_id = policy.policy_id.strip()
        if not policy_id:
            raise ValueError("operator activation policy_id must not be empty")
        if policy_id in self._policies:
            raise ValueError(f"duplicate operator activation policy_id: {policy_id}")
        self._policies[policy_id] = policy

    def resolve(self, policy_id: str) -> OperatorActivationPolicy:
        try:
            return self._policies[policy_id]
        except KeyError as error:
            raise KeyError(f"unregistered operator activation policy_id: {policy_id}") from error
