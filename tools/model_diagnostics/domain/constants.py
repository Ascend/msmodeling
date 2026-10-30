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

"""Constants shared by model-diagnostics sources and Theory evaluation."""

# P0-02 single source of truth: the boundary-token constant is owned by
# tensor_cast.core.vision_geometry and re-exported here so model
# diagnostics and Theory evaluation stay aligned with input generation.
from tensor_cast.core.vision_geometry import (  # noqa: E402,F401
    VISION_IMAGE_BOUNDARY_TOKEN_COUNT,
)
