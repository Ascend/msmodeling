# Copyright (c) 2026-2026 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
