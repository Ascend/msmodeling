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

"""Semantic calibration primitives for the analytic performance model.

The calibrated analytic route intentionally does not import or consult the
profiling database's OP mapping.  Calibration rules are selected from the
TensorCast operation semantic signature instead.
"""

from .base import CalibrationDataSource, CalibrationRule, EmptyCalibrationDataSource
from .profile import ProfileCalibrationDataSource
from .rules import (
    AttentionLatencyRule,
    CommunicationLatencyCurveRule,
    GmmUtilizationRule,
    MmaUtilizationRule,
)
from .signature import CalibrationSignature, build_calibration_signature

__all__ = [
    "CalibrationDataSource",
    "CalibrationRule",
    "CalibrationSignature",
    "AttentionLatencyRule",
    "CommunicationLatencyCurveRule",
    "EmptyCalibrationDataSource",
    "MmaUtilizationRule",
    "GmmUtilizationRule",
    "ProfileCalibrationDataSource",
    "build_calibration_signature",
]
