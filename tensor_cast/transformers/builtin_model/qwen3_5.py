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

from ..custom_model_registry import ModelProfile, register_model_profile
from .qwen3_5_moe import patch_method_for_qwen3_5, QWEN3_5_VISUAL_CONFIG

register_model_profile(
    ModelProfile(
        model_type="qwen3_5",
        model_family="qwen3_5",
        patch_method=patch_method_for_qwen3_5,
        **QWEN3_5_VISUAL_CONFIG,
    )
)
