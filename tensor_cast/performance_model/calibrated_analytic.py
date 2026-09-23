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

"""Calibrated wrapper for the raw analytic performance model."""

import logging
import sqlite3
from typing import List

from overrides import override

from .base import PerformanceModel
from .calibration import CalibrationDataSource, build_calibration_signature
from .op_invoke_info import OpInvokeInfo

logger = logging.getLogger(__name__)


class CalibratedAnalyticPerformanceModel(PerformanceModel):
    """Apply semantic calibration rules to an existing raw analytic model.

    A missing rule returns the original raw result unchanged.  This preserves
    roofline statistics and makes the calibrated route safe to enable with a
    partial calibration profile.
    """

    def __init__(
        self,
        analytic_model: PerformanceModel,
        calibration_source: CalibrationDataSource,
    ):
        if analytic_model.name != "analytic":
            raise ValueError("CalibratedAnalyticPerformanceModel requires an analytic base model")
        super().__init__("calibrated", analytic_model.device_profile)
        self.analytic_model = analytic_model
        self.calibration_source = calibration_source

    @override
    def process_op(self, op_invoke_info: OpInvokeInfo) -> PerformanceModel.Result:
        raw_result = self.analytic_model.process_op(op_invoke_info)
        rule = None
        fallback_reason = None
        try:
            signature = build_calibration_signature(
                op_invoke_info,
                raw_result,
                device_name=self.device_profile.name,
            )
            rule = self.calibration_source.lookup(signature)
            if rule is not None:
                result = rule.apply(raw_result, signature)
        except (KeyError, TypeError, ValueError, OSError, sqlite3.DatabaseError) as error:
            fallback_reason = "calibration_error"
            rule = None
            logger.warning("calibration failed; using raw analytic result for %s: %s", op_invoke_info.func, error)
        if rule is None:
            statistics = dict(raw_result.statistics)
            diagnostic = self.calibration_source.last_lookup_diagnostic
            fallback_reason = fallback_reason or diagnostic.get("fallback_reason", "no_matching_rule")
            statistics.update(
                {
                    "source": "ANALYTIC_RAW",
                    "calibration": {
                        "raw_latency_s": raw_result.execution_time_s,
                        "calibrated_latency_s": raw_result.execution_time_s,
                        "profile_id": None,
                        "match_rule": None,
                        "confidence": None,
                        "fallback_reason": fallback_reason,
                    },
                }
            )
            logger.debug(
                "calibration_hit=false op=%s fallback_reason=%s",
                op_invoke_info.func,
                fallback_reason,
            )
            return PerformanceModel.Result(execution_time_s=raw_result.execution_time_s, statistics=statistics)
        calibration = result.statistics.get("calibration", {})
        logger.debug(
            "calibration_hit=true op=%s source=%s confidence=%s rule=%s",
            op_invoke_info.func,
            calibration.get("source", "ANALYTIC_CALIBRATED"),
            calibration.get("confidence"),
            calibration.get("rule_id"),
        )
        return result

    @override
    def get_classifiers(self) -> List[PerformanceModel.OpClassifier]:
        return self.analytic_model.get_classifiers()
