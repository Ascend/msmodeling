import torch

from ...model_config import MoEFieldNames

from ...utils import exact_division

from ..custom_model_registry import (
    ModelProfile,
    register_model_profile,
    resolve_visual_config,
)


QWEN3_5_VISUAL_CONFIG = resolve_visual_config({})


def _set_qwen3_5_linear_attn_tp_size(model):
    tp_size = model.parallel_group_manager.tp_group.world_size
    if tp_size <= 1:
        return

    for module in model._inner.modules():
        if hasattr(module, "num_k_heads") and hasattr(module, "num_v_heads"):
            if module.num_k_heads % tp_size != 0:
                raise ValueError(
                    "Qwen3.5 linear attention requires tp_size to divide "
                    f"num_k_heads exactly, but got num_k_heads={module.num_k_heads} "
                    f"and tp_size={tp_size}."
                )
            if module.num_v_heads % tp_size != 0:
                raise ValueError(
                    "Qwen3.5 linear attention requires tp_size to divide "
                    f"num_v_heads exactly, but got num_v_heads={module.num_v_heads} "
                    f"and tp_size={tp_size}."
                )
            module.tensor_cast_tp_size = tp_size


def patch_method_for_qwen3_5(model):
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5GatedDeltaNet,
        Qwen3_5Model,
        Qwen3_5TextModel,
    )
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeGatedDeltaNet,
        Qwen3_5MoeModel,
        Qwen3_5MoeTextModel,
    )

    def _get_local_linear_attn_heads(self):
        tp_size = getattr(self, "tensor_cast_tp_size", 1)
        return (
            exact_division(self.num_k_heads, tp_size),
            exact_division(self.num_v_heads, tp_size),
        )

    def _patched_update_linear_attn_mask(self, attention_mask, cache_position):
        # Qwen3.5 linear-attention mask path has tensor-value-based branches that
        # are compile-unfriendly under TensorCast tracing; return None in compile
        # mode to keep a stable graph and align with decode behavior.
        if torch.compiler.is_compiling():
            return None

        if cache_position is not None and cache_position.device.type == "meta":
            return attention_mask

        linear_attn_mask = attention_mask

        is_meta_tensor = (
            hasattr(cache_position, "is_meta") and cache_position.is_meta
        ) or (
            attention_mask is not None
            and hasattr(attention_mask, "is_meta")
            and attention_mask.is_meta
        )

        if is_meta_tensor:
            return None

        try:
            cache_condition = (
                cache_position[0] > 0 if cache_position.numel() > 0 else False
            )
            mask_condition = (
                torch.all(attention_mask == 1).item()
                if attention_mask is not None and attention_mask.numel() > 0
                else False
            )

            if cache_condition or mask_condition:
                linear_attn_mask = None
        except RuntimeError:
            return None

        return linear_attn_mask

    def _patched_linear_attn_forward(
        self,
        hidden_states: torch.Tensor,
        cache_params=None,
        cache_position=None,
        attention_mask=None,
    ):
        # Route Qwen3.5 GatedDeltaNet through tensor_cast.linear_attention so
        # TensorCast can model mixed full/linear attention explicitly.
        del cache_params
        local_num_k_heads, local_num_v_heads = _get_local_linear_attn_heads(self)
        return torch.ops.tensor_cast.linear_attention(
            hidden_states,
            attention_mask,
            cache_position,
            local_num_k_heads,
            local_num_v_heads,
            self.head_k_dim,
            self.head_v_dim,
            self.conv_kernel_size,
        )

    target_classes = [Qwen3_5Model, Qwen3_5MoeModel]
    original_methods = {cls: cls.get_placeholder_mask for cls in target_classes}

    def _patched_get_placeholder_mask(self, *args, **kwargs):
        # In meta/simulation runs, HF strict image_features-vs-token-count checks
        # can fail even when we only need shape simulation; skip this check.
        kwargs["image_features"] = None
        return original_methods[type(self)](self, *args, **kwargs)

    # Model-specific monkey patches required for TensorCast simulation:
    # - linear attention op routing and mask behavior for compile stability
    # - VL placeholder-mask relaxation for meta execution
    Qwen3_5TextModel._update_linear_attn_mask = _patched_update_linear_attn_mask
    Qwen3_5MoeTextModel._update_linear_attn_mask = _patched_update_linear_attn_mask
    Qwen3_5GatedDeltaNet.forward = _patched_linear_attn_forward
    Qwen3_5MoeGatedDeltaNet.forward = _patched_linear_attn_forward
    for cls in target_classes:
        cls.get_placeholder_mask = _patched_get_placeholder_mask

    _set_qwen3_5_linear_attn_tp_size(model)


register_model_profile(
    ModelProfile(
        model_type="qwen3_5_moe",
        moe_module_name="Qwen3_5MoeSparseMoeBlock",
        moe_gate_returns_raw_logits=False,
        moe_num_experts_key=["text_config", "num_experts"],
        moe_field_names_override=MoEFieldNames(
            shared_experts="shared_expert",
            shared_experts_gate="shared_expert_gate",
        ),
        model_family="qwen3_5",
        patch_method=patch_method_for_qwen3_5,
        **QWEN3_5_VISUAL_CONFIG,
    )
)
