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

from ...model_config import MoEFieldNames
from ..custom_model_registry import ModelProfile, register_model_profile
from .qwen3_5_moe import patch_method_for_qwen3_5


register_model_profile(
    ModelProfile(
        model_type="qwen3_5_moe_text",
        moe_module_name="Qwen3_5MoeSparseMoeBlock",
        moe_gate_returns_raw_logits=False,
        moe_num_experts_key="num_experts",
        moe_field_names_override=MoEFieldNames(
            shared_experts="shared_expert",
            shared_experts_gate="shared_expert_gate",
        ),
        mtp_block_module_name="Qwen3_5MoeDecoderLayer",
        # Qwen3.8 open-source weights are Qwen3_5MoeForCausalLM + model_type=qwen3_5_moe_text,
        # a text-only MoE variant of the Qwen3.5 family (reusing Qwen3_5 module names and
        # patch_method_for_qwen3_5). model_family must be "qwen3_5" so the Gated DeltaNet
        # linear_attn TP plan gate in transformations.py (model_family == "qwen3_5") matches;
        # otherwise TP>1 sharding diverges from the patch and breaks.
        model_family="qwen3_5",
        patch_method=patch_method_for_qwen3_5,
    )
)
