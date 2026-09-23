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

"""GLM-4V TensorCast patch regression tests."""

from tensor_cast.transformers.builtin_model.glm4v import (
    patch_method_for_glm4_vl,
    patch_method_for_glm4v_dense,
)


def test_glm4v_dense_patch_is_idempotent() -> None:
    from transformers.models.glm4v import Glm4vModel
    from transformers.models.glm4v.modeling_glm4v import Glm4vTextRotaryEmbedding

    patch_method_for_glm4v_dense(None)
    methods = (Glm4vModel.get_placeholder_mask, Glm4vTextRotaryEmbedding.forward)

    patch_method_for_glm4v_dense(None)

    assert (Glm4vModel.get_placeholder_mask, Glm4vTextRotaryEmbedding.forward) == methods


def test_glm4v_moe_patch_is_idempotent() -> None:
    from transformers.models.glm4v_moe import Glm4vMoeModel
    from transformers.models.glm4v_moe.modeling_glm4v_moe import (
        Glm4vMoeTextRotaryEmbedding,
        Glm4vMoeVisionEmbeddings,
    )

    patch_method_for_glm4_vl(None)
    methods = (
        Glm4vMoeModel.get_placeholder_mask,
        Glm4vMoeTextRotaryEmbedding.forward,
        Glm4vMoeVisionEmbeddings.forward,
    )

    patch_method_for_glm4_vl(None)

    assert (
        Glm4vMoeModel.get_placeholder_mask,
        Glm4vMoeTextRotaryEmbedding.forward,
        Glm4vMoeVisionEmbeddings.forward,
    ) == methods
