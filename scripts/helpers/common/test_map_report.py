"""Nightly reporting helpers for test_node -> source_file -> symbols test_map."""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scripts.helpers.ci_gate.models import ExpiredExemptionReport, GatePolicy
    from scripts.helpers.nightly.report_models import MapCoverageSummary

from scripts.helpers.common.ast_utils import MODULE_SYMBOL

UNCLASSIFIED_SYMBOL = MODULE_SYMBOL


def iter_unique_symbol_refs(
    mapping: dict[str, dict[str, list[str]]],
) -> set[tuple[str, str]]:
    """Return unique ``(source_file, symbol)`` pairs covered by *mapping*."""
    refs: set[tuple[str, str]] = set()
    for sources in mapping.values():
        for src_file, symbols in sources.items():
            for symbol in symbols:
                refs.add((src_file, symbol))
    return refs


def summarize_test_map(mapping: dict[str, dict[str, list[str]]]) -> MapCoverageSummary:
    """Build nightly summary counts for the node-oriented test_map."""
    from scripts.helpers.nightly.report_models import MapCoverageSummary

    return MapCoverageSummary(
        test_nodes=len(mapping),
        symbol_refs=len(iter_unique_symbol_refs(mapping)),
    )


def is_source_symbol_mapped(
    mapping: dict[str, dict[str, list[str]]],
    source_file: str,
    symbol: str,
) -> bool:
    """Return True when any test node covers ``source_file::symbol``."""
    for sources in mapping.values():
        for mapped_symbol in sources.get(source_file, ()):
            if mapped_symbol == symbol:
                return True
    return False


def find_expired_unmapped_in_map(
    policy: GatePolicy,
    mapping: dict[str, dict[str, list[str]]],
    *,
    today: date | None = None,
) -> tuple[ExpiredExemptionReport, ...]:
    """Return expired source exemptions still missing from a node-oriented test_map."""
    from scripts.helpers.ci_gate.models import ExpiredExemptionReport

    check_date = today or date.today()
    reports: list[ExpiredExemptionReport] = []
    for entry in policy.source_exemptions:
        if entry.deadline >= check_date:
            continue
        if is_source_symbol_mapped(mapping, entry.file, entry.symbol):
            continue
        reports.append(
            ExpiredExemptionReport(
                symbol_key=entry.symbol_key,
                deadline=entry.deadline,
                reason=entry.reason,
                applicant=entry.applicant,
                approver=entry.approver,
                ticket=entry.ticket,
            )
        )
    return tuple(reports)
