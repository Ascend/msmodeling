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
"""Minimal pytest-compatible assertion adapter."""

from tools.model_diagnostics.domain import DiagnosticsResult, FindingStatus


def assert_diagnostics_passed(result: DiagnosticsResult) -> None:
    if result.summary.overall_status is FindingStatus.PASS:
        return
    failing = tuple(finding for finding in result.findings if finding.status is not FindingStatus.PASS)
    preview = "; ".join(
        f"{item.status.value}:{item.region_id}/{item.stage_id}:{item.message_code}" for item in failing[:5]
    )
    suffix = "" if len(failing) <= 5 else f"; ... {len(failing) - 5} more"
    raise AssertionError(
        f"model diagnostics {result.summary.overall_status.value}: "
        f"{len(failing)} non-pass finding(s): {preview}{suffix}"
    )
