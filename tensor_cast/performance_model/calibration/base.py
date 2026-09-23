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

"""Base contracts for analytic calibration data and rules.

These contracts deliberately differ from ``DataSourcePerformanceModel``:
profiling data sources return a measured latency for a mapped NPU kernel,
whereas calibration data sources return a rule to apply to an already computed
TensorCast analytic result.
"""

from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING

from ..base import PerformanceModel

if TYPE_CHECKING:
    from .signature import CalibrationSignature


class CalibrationRule(ABC):
    """A transformation from a raw analytic result to a calibrated result."""

    @abstractmethod
    def apply(
        self,
        raw_result: PerformanceModel.Result,
        signature: "CalibrationSignature",
    ) -> PerformanceModel.Result:
        """Return a calibrated result for one semantic operation invocation."""
        ...


class CalibrationDataSource(ABC):
    """Find an analytic calibration rule by TensorCast semantic signature."""

    @abstractmethod
    def lookup(self, signature: "CalibrationSignature") -> Optional[CalibrationRule]:
        """Return a matching calibration rule, or ``None`` for raw fallback."""
        ...

    @property
    def last_lookup_diagnostic(self) -> dict:
        """Explain the most recent lookup when the source supports diagnostics.

        This deliberately has a harmless default so existing custom data sources
        only need to implement :meth:`lookup`.
        """
        return {}


class EmptyCalibrationDataSource(CalibrationDataSource):
    """A source with no rules, useful as an explicit raw-analytic fallback."""

    def lookup(self, signature: "CalibrationSignature") -> Optional[CalibrationRule]:
        del signature
        return None
