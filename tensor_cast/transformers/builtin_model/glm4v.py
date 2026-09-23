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


def _patch_mtp_position_ids(rotary_cls):
    """Accept the 2-D text positions supplied by TensorCast's MTP wrapper."""
    if getattr(rotary_cls, "_tensor_cast_mtp_position_ids_patched", False):
        return

    original_forward = rotary_cls.forward

    def patched_forward(self, x, position_ids, *args, **kwargs):
        # HF GLM-4V normally receives three MRoPE position streams from the
        # multimodal model. TensorCast MTP starts from flattened text positions;
        # for text-only speculative decode all three streams are identical.
        if position_ids.ndim == 2:
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        return original_forward(self, x, position_ids, *args, **kwargs)

    rotary_cls.forward = patched_forward
    rotary_cls._tensor_cast_mtp_position_ids_patched = True


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

    if not getattr(Glm4vMoeModel, "_tensor_cast_placeholder_mask_patched", False):
        original_get_placeholder_mask = Glm4vMoeModel.get_placeholder_mask

        def patched_get_placeholder_mask(self, *args, **kwargs):
            # Forcibly skip image_features
            kwargs["image_features"] = None
            return original_get_placeholder_mask(self, *args, **kwargs)

        Glm4vMoeModel.get_placeholder_mask = patched_get_placeholder_mask
        Glm4vMoeModel._tensor_cast_placeholder_mask_patched = True

    from transformers.models.glm4v_moe.modeling_glm4v_moe import (
        Glm4vMoeTextRotaryEmbedding,
        Glm4vMoeVisionEmbeddings,
    )

    _patch_mtp_position_ids(Glm4vMoeTextRotaryEmbedding)

    if not getattr(Glm4vMoeVisionEmbeddings, "_tensor_cast_lengths_patched", False):
        original_forward = Glm4vMoeVisionEmbeddings.forward

        def patched_forward(self, *args, **kwargs):
            if len(args) > 1 and isinstance(args[1], list):
                lengths_tensor = torch.tensor(args[1], dtype=torch.long)
                args = (args[0], lengths_tensor) + args[2:]
            return original_forward(self, *args, **kwargs)

        Glm4vMoeVisionEmbeddings.forward = patched_forward
        Glm4vMoeVisionEmbeddings._tensor_cast_lengths_patched = True


def patch_method_for_glm4v_dense(_model):
    """Patch dense GLM-4V placeholder checks that call ``item()`` in meta mode."""

    from transformers.models.glm4v import Glm4vModel
    from transformers.models.glm4v.modeling_glm4v import Glm4vTextRotaryEmbedding

    _patch_mtp_position_ids(Glm4vTextRotaryEmbedding)

    if not getattr(Glm4vModel, "_tensor_cast_placeholder_mask_patched", False):
        original_get_placeholder_mask = Glm4vModel.get_placeholder_mask

        def patched_get_placeholder_mask(self, *args, **kwargs):
            kwargs["image_features"] = None
            return original_get_placeholder_mask(self, *args, **kwargs)

        Glm4vModel.get_placeholder_mask = patched_get_placeholder_mask
        Glm4vModel._tensor_cast_placeholder_mask_patched = True


register_model_profile(
    ModelProfile(
        model_type="glm4v",
        model_family="glm4v",
        mtp_block_module_name="Glm4vTextDecoderLayer",
        patch_method=patch_method_for_glm4v_dense,
        **GLM4V_VISUAL_CONFIG,
    )
)


register_model_profile(
    ModelProfile(
        model_type="glm4v_moe",
        moe_module_name="Glm4vMoeTextMoE",
        moe_gate_returns_raw_logits=True,
        moe_num_experts_key=["text_config", "n_routed_experts"],
        model_family="glm4v",
        mtp_block_module_name="Glm4vMoeTextDecoderLayer",
        patch_method=patch_method_for_glm4_vl,
        **GLM4V_VISUAL_CONFIG,
    )
)
