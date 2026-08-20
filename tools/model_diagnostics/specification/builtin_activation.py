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
"""Builtin semantic activation policies shared across model Specs."""

from __future__ import annotations

from tools.model_diagnostics.domain import ExecutionPhase
from tools.model_diagnostics.specification.mtp_window import is_mtp_enabled
from tools.model_diagnostics.specification.operator_activation import (
    OperatorActivationRegistry,
    OperatorActivationRequest,
)


class LmHeadTokenSelectionActivation:
    """Enable prefill-only semantic selection performed immediately before ``lm_head``.

    MTP target selection uses the separate ``mtp_target_select`` operator under
    ``mtp_enabled``; the two contracts must not share one ambiguous operator.
    """

    policy_id = "lm_head_token_selection"

    def is_active(self, request: OperatorActivationRequest) -> bool:
        return request.context.phase is ExecutionPhase.PREFILL


class MtpEnabledActivation:
    """Enable operators that exist only for a legal fixed-length MTP decode window."""

    policy_id = "mtp_enabled"

    def is_active(self, request: OperatorActivationRequest) -> bool:
        return is_mtp_enabled(request.context)


class NonMtpLmHeadActivation:
    """Enable the ordinary ``lm_head`` path when MTP target verification is inactive."""

    policy_id = "non_mtp_lm_head"

    def is_active(self, request: OperatorActivationRequest) -> bool:
        return not is_mtp_enabled(request.context)


class DsaEnabledActivation:
    """Enable DSA semantics when the loaded model config exposes DSA top-k."""

    policy_id = "dsa_enabled"

    def is_active(self, request: OperatorActivationRequest) -> bool:
        return "index_topk" in request.context.model_config


class ExplicitMoeGateActivation:
    """Activate the standalone gate only when Runtime emits it as a call.

    Most DeepSeek-family runtimes execute the router as a dedicated gate mm.
    Kimi K2.5/K2.6 (``kimi_k2``) patch the MoE inference so routing is computed
    inside the fused kernels without a standalone gate call, so their gate stage
    is omitted. Keep this allowlist in sync with the Runtime patch in
    ``tensor_cast.transformers.builtin_model.kimi_k25``.
    """

    policy_id = "explicit_moe_gate"

    def is_active(self, request: OperatorActivationRequest) -> bool:
        return request.context.model_config.get("model_type") != "kimi_k2"


def create_builtin_operator_activation_registry() -> OperatorActivationRegistry:
    registry = OperatorActivationRegistry()
    registry.register(LmHeadTokenSelectionActivation())
    registry.register(MtpEnabledActivation())
    registry.register(NonMtpLmHeadActivation())
    registry.register(DsaEnabledActivation())
    registry.register(ExplicitMoeGateActivation())
    return registry
