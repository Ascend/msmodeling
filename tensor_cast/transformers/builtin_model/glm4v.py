import torch

from tensor_cast.layers import COLWISE_LINEAR, ROWWISE_LINEAR
from ..custom_model_registry import (
    ModelProfile,
    register_model_profile,
    resolve_visual_config,
)


GLM4V_VISUAL_CONFIG = resolve_visual_config(
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


def patch_method_for_glm4_vl(_model):
    """
    Patch the GLM4V-MoE model to fix simulation issues in meta mode.

    Problem background:
    1. VisionEmbeddings.forward converts lengths in list form to a meta tensor,
        while subsequent computations require actual values (implicitly calling item), which causes errors;
    2. get_placeholder_mask uses boolean-mask-based tensor indexing operations,
        which fail or cause dimension mismatch in meta mode.

    Solution:
    * Convert list-based lengths to a tensor before entering forward, avoiding the creation of a meta tensor.
    * Force image_features=None to skip image-related checks in get_placeholder_mask.
    """

    from transformers.models.glm4v_moe import Glm4vMoeModel

    original_get_placeholder_mask = Glm4vMoeModel.get_placeholder_mask

    def patched_get_placeholder_mask(self, *args, **kwargs):
        # Forcibly skip image_features
        kwargs["image_features"] = None
        return original_get_placeholder_mask(self, *args, **kwargs)

    Glm4vMoeModel.get_placeholder_mask = patched_get_placeholder_mask

    from transformers.models.glm4v_moe.modeling_glm4v_moe import (
        Glm4vMoeVisionEmbeddings,
    )

    original_forward = Glm4vMoeVisionEmbeddings.forward

    def patched_forward(self, *args, **kwargs):
        if len(args) > 1 and isinstance(args[1], list):
            lengths_tensor = torch.tensor(args[1], dtype=torch.long)
            args = (args[0], lengths_tensor) + args[2:]
        return original_forward(self, *args, **kwargs)

    Glm4vMoeVisionEmbeddings.forward = patched_forward


register_model_profile(
    ModelProfile(
        model_type="glm4v_moe",
        moe_module_name="Glm4vMoeTextMoE",
        moe_gate_returns_raw_logits=True,
        moe_num_experts_key=["text_config", "n_routed_experts"],
        model_family="glm4v",
        patch_method=patch_method_for_glm4_vl,
        **GLM4V_VISUAL_CONFIG,
    )
)
