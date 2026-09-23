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

"""Re-export CI gate policy APIs."""

from __future__ import annotations

from scripts.helpers.ci_gate import policy as _policy_mod
from scripts.helpers.ci_gate.models import (
    CiGatePolicy,
    ExpiredExemptionReport,
    GatePolicy,
    PathPatterns,
    SourceExemption,
    TestDiscovery,
    TestExemption,
)
from scripts.helpers.ci_gate.policy import (
    APPROVERS_REL,
    CI_POLICY_REL,
    GATE_POLICY_REL,
    find_expired_test_exemptions,
    find_expired_unmapped,
    format_expired_exemptions_section,
    format_expired_test_exemptions_section,
    gate_policy_changed_in_diff,
    is_config_path,
    is_exempt,
    is_gate_test_path,
    is_policy_config_path,
    is_source_path,
    is_test_exempt,
    is_test_path,
    load_gate_policy,
    matches_path_patterns,
    validate_gate_policy_if_changed,
)

_load_gate_policy_cached = _policy_mod._load_gate_policy_cached

__all__ = [
    "APPROVERS_REL",
    "CI_POLICY_REL",
    "GATE_POLICY_REL",
    "CiGatePolicy",
    "ExpiredExemptionReport",
    "GatePolicy",
    "PathPatterns",
    "SourceExemption",
    "TestDiscovery",
    "TestExemption",
    "find_expired_test_exemptions",
    "find_expired_unmapped",
    "format_expired_exemptions_section",
    "format_expired_test_exemptions_section",
    "gate_policy_changed_in_diff",
    "is_config_path",
    "is_exempt",
    "is_gate_test_path",
    "is_policy_config_path",
    "is_source_path",
    "is_test_exempt",
    "is_test_path",
    "load_gate_policy",
    "matches_path_patterns",
    "validate_gate_policy_if_changed",
]
