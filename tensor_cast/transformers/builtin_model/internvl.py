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

from tensor_cast.layers import COLWISE_LINEAR, ROWWISE_LINEAR
from tensor_cast.transformers.custom_model_registry import (
    ModelProfile,
    register_model_profile,
)

INTERNVL_VISUAL_CONFIG = {
    "visual_module_path": "visual",
    "language_module_path": "language_model",
    "visual_layers_module_path": "vision_tower.encoder.layer",
    "visual_layers_path_str": "vision_tower.encoder.layer",
    "language_layers_path_str": "language_model.layers",
    "visual_merger_linear_mapping": {},
    "visual_mlp_linear_mapping": {
        "vision_tower.encoder.layer.*.mlp.fc1": COLWISE_LINEAR,
        "vision_tower.encoder.layer.*.mlp.fc2": ROWWISE_LINEAR,
    },
}

register_model_profile(
    ModelProfile(
        model_type="internvl",
        model_family="internvl",
        **INTERNVL_VISUAL_CONFIG,
    )
)
