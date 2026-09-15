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
