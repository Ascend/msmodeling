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


class VisionPrefillActivation:
    """Enable vision-front-end regions only for image prefill captures."""

    policy_id = "vision_prefill"

    def is_active(self, request: OperatorActivationRequest) -> bool:
        image_dimensions = tuple(
            request.context.model_config.get(key, 0)
            for key in ("image_batch_size", "image_height", "image_width")
        )
        return (
            request.context.phase is ExecutionPhase.PREFILL
            and all(
                isinstance(value, int) and not isinstance(value, bool) and value > 0
                for value in image_dimensions
            )
        )


class DsaEnabledActivation:
    """Enable DSA semantics when the loaded model config exposes DSA top-k."""

    policy_id = "dsa_enabled"

    def is_active(self, request: OperatorActivationRequest) -> bool:
        return "index_topk" in request.context.model_config


class MlaPrefillKvProjectionActivation:
    """Enable the Prefill-only compressed-KV decompress projection.

    Decode leaves ``num_prefill_tokens == 0`` and does not emit
    ``mla_kv_projection``; the unused-phase placeholder stays on the attention core.
    """

    policy_id = "mla_prefill_kv_projection"

    def is_active(self, request: OperatorActivationRequest) -> bool:
        return request.context.phase is ExecutionPhase.PREFILL


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
        return request.context.model_config.get("model_type") not in {"kimi_k2", "kimi_k25"}


class MoEFusedTopkActivation:
    """Fused ``moe_gating_top_k_softmax`` exists only on the raw-logits gate path."""

    policy_id = "moe_fused_topk"

    def is_active(self, request: OperatorActivationRequest) -> bool:
        return request.context.model_config.get("model_type") in {"deepseek_v3", "glm_moe_dsa"}


class Qwen35DenseFfnActivation:
    """Qwen3.5 Dense uses the category-1 dense FFN for every layer."""

    policy_id = "qwen3_5_dense_ffn"

    def is_active(self, request: OperatorActivationRequest) -> bool:
        return request.context.model_config.get("model_type") == "qwen3_5_text"


class Qwen35MoeFfnActivation:
    """Qwen3.5-MoE and Qwen3-Next route every layer through the category-2 MoE FFN."""

    policy_id = "qwen3_5_moe_ffn"

    def is_active(self, request: OperatorActivationRequest) -> bool:
        return request.context.model_config.get("model_type") in {"qwen3_5_moe_text", "qwen3_next"}


class Qwen35LinearGdnActivation:
    """Qwen3.5 and Qwen3-Next expand GatedDeltaNet into projection/rule/output stages."""

    policy_id = "qwen3_5_linear_gdn"

    def is_active(self, request: OperatorActivationRequest) -> bool:
        return request.context.model_config.get("model_type") in {
            "qwen3_5_text",
            "qwen3_5_moe_text",
            "qwen3_next",
        }


def create_builtin_operator_activation_registry() -> OperatorActivationRegistry:
    registry = OperatorActivationRegistry()
    registry.register(LmHeadTokenSelectionActivation())
    registry.register(MtpEnabledActivation())
    registry.register(NonMtpLmHeadActivation())
    registry.register(VisionPrefillActivation())
    registry.register(DsaEnabledActivation())
    registry.register(MlaPrefillKvProjectionActivation())
    registry.register(ExplicitMoeGateActivation())
    registry.register(MoEFusedTopkActivation())
    registry.register(Qwen35DenseFfnActivation())
    registry.register(Qwen35MoeFfnActivation())
    registry.register(Qwen35LinearGdnActivation())
    return registry
