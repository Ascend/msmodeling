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


def _has_complete_image_dimensions(request: OperatorActivationRequest) -> bool:
    image_dimensions = tuple(
        request.context.model_config.get(key, 0)
        for key in ("image_batch_size", "image_height", "image_width")
    )
    return all(
        isinstance(value, int) and not isinstance(value, bool) and value > 0
        for value in image_dimensions
    )


class VisionPrefillActivation:
    """Enable vision-front-end regions only for image prefill captures."""

    policy_id = "vision_prefill"

    def is_active(self, request: OperatorActivationRequest) -> bool:
        return request.context.phase is ExecutionPhase.PREFILL and _has_complete_image_dimensions(request)


class KimiTextPrefillActivation:
    """Select hidden states before lm_head for Kimi text-only prefill."""

    policy_id = "kimi_text_prefill"

    def is_active(self, request: OperatorActivationRequest) -> bool:
        return (
            request.context.phase is ExecutionPhase.PREFILL
            and request.context.model_config.get("model_type") == "kimi_k25"
            and not _has_complete_image_dimensions(request)
        )


class KimiNonTextPrefillActivation:
    """Use Kimi's full-logits path for vision prefill and decode."""

    policy_id = "kimi_non_text_prefill"

    def is_active(self, request: OperatorActivationRequest) -> bool:
        return not KimiTextPrefillActivation().is_active(request)


class DsaEnabledActivation:
    """Enable DSA semantics when the loaded model config exposes DSA top-k."""

    policy_id = "dsa_enabled"

    def is_active(self, request: OperatorActivationRequest) -> bool:
        return "index_topk" in request.context.model_config


class DsaExpandedActivation:
    policy_id = "dsa_expanded"

    def is_active(self, request):
        return (
            "index_topk" in request.context.model_config
            and request.context.model_config.get("model_type") in {"glm_moe_dsa", "deepseek_v32"}
        )


class DsaExpandedCompiledActivation(DsaExpandedActivation):
    """The compiled indexer merges K and head-weight projections."""

    policy_id = "dsa_expanded_compiled"

    def is_active(self, request):
        return super().is_active(request) and "compiled" in request.context.model_config.get("features", ())


class DsaExpandedEagerActivation(DsaExpandedActivation):
    """The eager indexer exposes separate K and head-weight projections."""

    policy_id = "dsa_expanded_eager"

    def is_active(self, request):
        return super().is_active(request) and "compiled" not in request.context.model_config.get("features", ())


class DsaOpaqueActivation:
    policy_id = "dsa_opaque"

    def is_active(self, request):
        return "index_topk" in request.context.model_config and not DsaExpandedActivation().is_active(request)


class MlapoEnabledActivation:
    """Select the explicitly enabled fused MLA prolog contract."""

    policy_id = "mlapo_enabled"

    def is_active(self, request: OperatorActivationRequest) -> bool:
        return bool(request.context.model_config.get("enable_mlapo", False))


class MlapoDisabledActivation:
    """Select the expanded MLA prolog, permitting compiler-local linear merges."""

    policy_id = "mlapo_disabled"

    def is_active(self, request: OperatorActivationRequest) -> bool:
        return not bool(request.context.model_config.get("enable_mlapo", False))


class MlapoDisabledCompiledActivation(MlapoDisabledActivation):
    """Select the compiler's locally merged q_a/kv_a projection."""

    policy_id = "mlapo_disabled_compiled"

    def is_active(self, request: OperatorActivationRequest) -> bool:
        return super().is_active(request) and "compiled" in request.context.model_config.get("features", ())


class MlapoDisabledEagerActivation(MlapoDisabledActivation):
    """Select separate q_a and kv_a projections in eager mode."""

    policy_id = "mlapo_disabled_eager"

    def is_active(self, request: OperatorActivationRequest) -> bool:
        return super().is_active(request) and "compiled" not in request.context.model_config.get("features", ())

class MlaPrefillKvProjectionActivation:
    """Prefill KV projection is standalone only when MLAPO is disabled."""

    policy_id = "mla_prefill_kv_projection"

    def is_active(self, request: OperatorActivationRequest) -> bool:
        return request.context.phase is ExecutionPhase.PREFILL and not request.context.model_config.get(
            "enable_mlapo", False
        )


class KimiDynamicQaNormActivation:
    policy_id = "kimi_dynamic_qa_norm"

    def is_active(self, request: OperatorActivationRequest) -> bool:
        return MlapoDisabledActivation().is_active(request) and request.context.quantization_config.get("action") in {
            "W8A8_DYNAMIC", "W4A8_DYNAMIC"
        }


class KimiRegularQaNormActivation(KimiDynamicQaNormActivation):
    policy_id = "kimi_regular_qa_norm"

    def is_active(self, request: OperatorActivationRequest) -> bool:
        return MlapoDisabledActivation().is_active(request) and not super().is_active(request)


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
    registry.register(KimiTextPrefillActivation())
    registry.register(KimiNonTextPrefillActivation())
    registry.register(DsaEnabledActivation())
    registry.register(DsaExpandedActivation())
    registry.register(DsaExpandedCompiledActivation())
    registry.register(DsaExpandedEagerActivation())
    registry.register(DsaOpaqueActivation())
    registry.register(MlaPrefillKvProjectionActivation())
    registry.register(KimiDynamicQaNormActivation())
    registry.register(KimiRegularQaNormActivation())
    registry.register(MlapoEnabledActivation())
    registry.register(MlapoDisabledActivation())
    registry.register(MlapoDisabledCompiledActivation())
    registry.register(MlapoDisabledEagerActivation())
    registry.register(ExplicitMoeGateActivation())
    registry.register(MoEFusedTopkActivation())
    registry.register(Qwen35DenseFfnActivation())
    registry.register(Qwen35MoeFfnActivation())
    registry.register(Qwen35LinearGdnActivation())
    return registry
