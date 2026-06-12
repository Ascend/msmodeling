"""Structured gate error formatting for categorized output."""

from __future__ import annotations

from scripts.helpers.ci_gate.models import GateError

_CATEGORY_LABEL: dict[str, str] = {
    "new_source": "[A]",
    "modified_source": "[M]",
    "deleted_source": "[D]",
    "deleted_test": "[D]",
}

_CATEGORY_HEADER: dict[str, str] = {
    "new_source": "new source file(s) have no test_map entry and are not exempt",
    "modified_source": "modified symbol(s) have no test_map entry and are not exempt",
    "deleted_source": "deleted source file(s) have no test_map entry",
    "deleted_test": "deleted test(s) are sole coverage for source symbols",
}

_CATEGORY_SUGGESTION: dict[str, str] = {
    "new_source": (
        "Add test cases or register an exemption in tests/.ci/gate_policy.yaml.\n"
        "  → If already exempted, ensure symbols matches path::symbol above exactly."
    ),
    "modified_source": (
        "Add test cases or register an exemption in tests/.ci/gate_policy.yaml.\n"
        "  → If already exempted, ensure symbols matches path::symbol above exactly."
    ),
    "deleted_source": "Remove orphaned test_map entries or restore source file.",
    "deleted_test": "Add replacement test cases or delete the corresponding source symbols.",
}


def format_blocking_errors(errors: tuple[GateError, ...]) -> str:
    by_category: dict[str, list[GateError]] = {}
    for e in errors:
        by_category.setdefault(e.category, []).append(e)

    lines: list[str] = []
    lines.append(
        "CI gate failed: policy violation — incremental phases (Phase 1/2) were not run.\n"
        f"Blocking items: {len(errors)}. "
        "Phase 0 may still have run when this PR modified test files.\n"
    )

    for category in ("new_source", "modified_source", "deleted_source", "deleted_test"):
        group = by_category.get(category)
        if not group:
            continue
        tag = _CATEGORY_LABEL.get(category, "")
        header = _CATEGORY_HEADER.get(category, category)
        suggestion = _CATEGORY_SUGGESTION.get(category, "")
        lines.append(f"The following {len(group)} {header}:")
        for e in group:
            label = f"{e.path}::{e.symbol}" if e.symbol else e.path
            lines.append(f"  - {label} {tag}")
            if e.detail:
                for dline in e.detail.splitlines():
                    lines.append(f"    {dline}")
        lines.append(f"  → {suggestion}")
        lines.append("")

    return "\n".join(lines)


def format_phase0_failure_hint(node_ids: tuple[str, ...]) -> str:
    """Format Phase 0 pytest failure guidance with optional test exemption YAML."""
    lines = [
        "CI gate failed: new test(s) failed. Fix test failures before gate check.",
        "",
        "Executed node(s):",
    ]
    for node_id in sorted(node_ids):
        lines.append(f"  - {node_id}")
    lines.extend(
        [
            "",
            "To exempt failing test(s) from Phase 0/2, add entries under exemptions.tests",
            "in tests/.ci/gate_policy.yaml:",
            "  exemptions:",
            "    tests:",
            "      - symbols:",
            "          - tests/path/to/test_file.py::test_name",
            '        reason: "<why this test is exempt from PR gate>"',
            "        applicant: <your-id>",
            "        approver: <approver-from-tests/.ci/approvers.yaml>",
            "        deadline: YYYY-MM-DD",
        ]
    )
    return "\n".join(lines)
