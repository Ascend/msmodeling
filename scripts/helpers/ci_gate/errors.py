"""Structured gate error formatting for categorized output."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scripts.helpers.ci_gate.models import GateError

_CATEGORY_LABEL: dict[str, str] = {
    "new_source": "[A]",
    "modified_source": "[M]",
    "deleted_test": "[D]",
    "deleted_source": "[DS]",
    "exemption_drift": "[ED]",
}

_CATEGORY_HEADER: dict[str, str] = {
    "new_source": "new source file(s) have no coverage mapping entry and are not exempt",
    "modified_source": "modified symbol(s) have no coverage mapping entry and are not exempt",
    "deleted_test": "deleted test(s) are sole coverage for source symbols",
    "deleted_source": "deleted source file(s) still have sole-coverage test mapping entries",
    "exemption_drift": "gate_policy exemption(s) reference deleted or renamed paths",
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
    "deleted_test": "Add replacement test cases or delete the corresponding source symbols.",
    "deleted_source": ("Delete the sole-coverage test node(s) in the same PR, or refresh test_map after nightly/sync."),
    "exemption_drift": (
        "Update or remove stale entries in tests/.ci/gate_policy.yaml to match the renamed or deleted paths in this PR."
    ),
}


def format_blocking_errors(errors: tuple[GateError, ...], *, pytest_ran: bool = False) -> str:
    by_category: dict[str, list[GateError]] = {}
    for err in errors:
        by_category.setdefault(err.category, []).append(err)

    lines: list[str] = []
    if pytest_ran:
        lines.append(
            f"CI gate failed: coverage mapping policy not satisfied after pytest.\nBlocking items: {len(errors)}."
        )
    else:
        lines.append(f"CI gate failed: policy violation — pytest was not run.\nBlocking items: {len(errors)}.")

    for category in (
        "exemption_drift",
        "new_source",
        "modified_source",
        "deleted_test",
        "deleted_source",
    ):
        group = by_category.get(category)
        if not group:
            continue
        tag = _CATEGORY_LABEL.get(category, "")
        header = _CATEGORY_HEADER.get(category, category)
        suggestion = _CATEGORY_SUGGESTION.get(category, "")
        lines.append(f"The following {len(group)} {header}:")
        for err in group:
            label = f"{err.path}::{err.symbol}" if err.symbol else err.path
            lines.append(f"  - {label} {tag}")
            if err.detail:
                lines.extend(f"    {dline}" for dline in err.detail.splitlines())
        lines.append(f"  → {suggestion}")
        lines.append("")

    return "\n".join(lines)


def format_pytest_failure_hint(node_ids: tuple[str, ...]) -> str:
    lines = [
        "CI gate failed: selected test(s) failed. Fix test failures before gate check.",
        "",
        "Executed node(s):",
        *[f"  - {node_id}" for node_id in sorted(node_ids)],
        "",
        "To exempt failing test(s), add entries under exemptions.tests",
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
    return "\n".join(lines)
