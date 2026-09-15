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
