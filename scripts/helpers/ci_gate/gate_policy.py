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
