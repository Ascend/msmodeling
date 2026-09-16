"""TensorCast profile for the language path of GLM-5.3-Flash."""

import torch

from ...layers import COLWISE_LINEAR, ROWWISE_LINEAR
from ...layers.glm5_next import (
    Glm5NextSparseAttention,
    glm5_next_decoder_forward,
    glm5_next_kda_forward,
    glm5_next_mhc_head_forward,
    glm5_next_mhc_forward,
)
from ..custom_model_registry import ModelProfile, register_model_profile, resolve_visual_config


class Glm5NextMoeExpertMLP(torch.nn.Module):
    """Expose a packed GLM5Next expert as TensorCast's standard SwiGLU MLP."""

    def __init__(self, original_experts_module: torch.nn.Module, expert_idx: int):
        super().__init__()
        hidden_size = original_experts_module.hidden_dim
        intermediate_size = original_experts_module.intermediate_dim
        self.swiglu_limit = original_experts_module.swiglu_limit
        self.gate_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = torch.nn.Linear(intermediate_size, hidden_size, bias=False)

        with torch.no_grad():
            gate_weight, up_weight = original_experts_module.gate_up_proj[expert_idx].chunk(2, dim=0)
            self.gate_proj.weight.copy_(gate_weight)
            self.up_proj.weight.copy_(up_weight)
            self.down_proj.weight.copy_(original_experts_module.down_proj[expert_idx])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            torch.ops.tensor_cast.clamped_swiglu(
                self.gate_proj(hidden_states), self.up_proj(hidden_states), self.swiglu_limit
            )
        )


def patch_method_for_glm5_next(_model) -> None:
    """Bridge GLM5Next's NoPE MLA and KDA calls to TensorCast wrappers."""
    from transformers.models.glm5_next.modeling_glm5_next import (
        Glm5NextModel,
        Glm5NextTextDecoderLayer,
        Glm5NextTextHyperHead,
        Glm5NextTextLinearAttention,
        Glm5NextTextHyperConnection,
        Glm5NextTextMLP,
    )

    if not getattr(Glm5NextTextDecoderLayer, "_tensor_cast_glm5_next_nope_patch", False):
        Glm5NextTextDecoderLayer.forward = glm5_next_decoder_forward
        Glm5NextTextDecoderLayer._tensor_cast_glm5_next_nope_patch = True
        Glm5NextTextLinearAttention.forward = glm5_next_kda_forward
        Glm5NextTextHyperConnection.forward = glm5_next_mhc_forward
        Glm5NextTextHyperHead.forward = glm5_next_mhc_head_forward

        def patched_mlp_forward(self, hidden_states):
            return self.down_proj(
                torch.ops.tensor_cast.clamped_swiglu(
                    self.gate_proj(hidden_states), self.up_proj(hidden_states), self.swiglu_limit
                )
            )

        Glm5NextTextMLP.forward = patched_mlp_forward
        original_get_placeholder_mask = Glm5NextModel.get_placeholder_mask

        def patched_get_placeholder_mask(self, *args, **kwargs):
            # TensorCast uses meta image features. The upstream token-count
            # check materializes a scalar with ``.item()`` and is therefore
            # not evaluable in a shape-only simulation. Keep the masks and
            # skip only that value-dependent consistency check.
            kwargs["image_features"] = None
            kwargs["video_features"] = None
            return original_get_placeholder_mask(self, *args, **kwargs)

        Glm5NextModel.get_placeholder_mask = patched_get_placeholder_mask

    tp_size = _model.parallel_group_manager.tp_group.world_size
    for module in _model._inner.modules():
        if isinstance(module, Glm5NextTextLinearAttention):
            module.tensor_cast_tp_size = tp_size
            module.tensor_cast_tp_rank = _model.parallel_group_manager.tp_group.rank_in_group


GLM5_NEXT_VISUAL_CONFIG = resolve_visual_config(
    {
        "visual_merger_linear_mapping": {
            "visual.merger.gate_proj": COLWISE_LINEAR,
            "visual.merger.up_proj": COLWISE_LINEAR,
            "visual.merger.down_proj": ROWWISE_LINEAR,
        },
        "visual_mlp_linear_mapping": {
            "visual.blocks.*.mlp.gate_proj": COLWISE_LINEAR,
            "visual.blocks.*.mlp.up_proj": COLWISE_LINEAR,
            "visual.blocks.*.mlp.down_proj": ROWWISE_LINEAR,
        },
    }
)


register_model_profile(
    ModelProfile(
        model_type="glm5_next",
        moe_module_name="Glm5NextTextMoE",
        moe_num_experts_key=["text_config", "n_routed_experts"],
        moe_gate_returns_raw_logits=False,
        mla_module_name="Glm5NextTextAttention",
        mla_module_class_type=Glm5NextSparseAttention,
        custom_expert_module_type=Glm5NextMoeExpertMLP,
        model_family="glm5_next",
        patch_method=patch_method_for_glm5_next,
        **GLM5_NEXT_VISUAL_CONFIG,
    )
)
