#!/usr/bin/env python3
"""Validate the /next comment protocol contract.

Checks that ``spec/governance/next-comment-protocol.md`` is internally
consistent:
- the file exists
- the verb table is present and parseable
- all 7 expected verbs are present (review, ack, approve, return, forward,
  block, reject)
- no unexpected verbs
- Issue-only verb (reject) is marked PR — / Issue ✓

Mirrors the validate_skills.py pattern: argparse + --json + dataclass findings
+ exit code.  Added to the AGENTS.md §9 AI-native gate list.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

PROTOCOL_PATH = Path("spec/governance/next-comment-protocol.md")

# Verbs expected in the protocol doc, with their entity applicability.
# (verb, applies_to_pr, applies_to_issue)
EXPECTED_VERBS: list[tuple[str, bool, bool]] = [
    ("review", True, True),
    ("ack", True, True),
    ("approve", True, True),
    ("return", True, True),
    ("forward", True, True),
    ("block", True, True),
    ("reject", False, True),
]

EXPECTED_VERB_NAMES = {v[0] for v in EXPECTED_VERBS}


@dataclass(frozen=True)
class Finding:
    path: str
    message: str
    severity: str = "error"


@dataclass(frozen=True)
class VerbRow:
    verb: str
    pr: bool
    issue: bool


def _parse_yes_dash(cell: str) -> bool:
    """Parse a table cell: ✓/yes/✅ → True, —/-/no/❌ → False."""
    c = cell.strip().lower()
    return c in ("✓", "yes", "✅", "y", "true")


def _extract_verb_table(text: str) -> list[VerbRow]:
    """Extract verb rows from the markdown verb table under '## verb 表'."""
    rows: list[VerbRow] = []
    in_table = False
    for line in text.splitlines():
        if line.startswith("## "):
            in_table = "verb" in line.lower()
            continue
        if not in_table:
            continue
        # Skip separator row (|---|---|...) and header row (| verb | PR | ...).
        if re.match(r"^\|[-\s|]+$", line):
            continue
        cells = [c.strip() for c in line.split("|")]
        # cells[0] is empty (before first |), cells[-1] is empty (after last |).
        cells = [c for c in cells if c != "" or True]  # keep alignment
        # Filter out empty leading/trailing from split.
        parts = [c for c in line.split("|")]
        # parts: ['', ' verb ', ' PR ', ' Issue ', ' 含义 ', ' 阈值 ', '']
        parts = [p.strip() for p in parts]
        # Remove empty strings from start/end.
        while parts and parts[0] == "":
            parts.pop(0)
        while parts and parts[-1] == "":
            parts.pop()
        if len(parts) < 3:
            continue
        verb_cell = parts[0].strip("`").strip()
        # Skip header row.
        if verb_cell.lower() == "verb":
            continue
        pr_cell = parts[1] if len(parts) > 1 else ""
        issue_cell = parts[2] if len(parts) > 2 else ""
        rows.append(VerbRow(verb=verb_cell, pr=_parse_yes_dash(pr_cell), issue=_parse_yes_dash(issue_cell)))
    return rows


def validate_next_protocol(repo_root: Path) -> list[Finding]:
    findings: list[Finding] = []
    protocol = repo_root / PROTOCOL_PATH
    if not protocol.is_file():
        findings.append(Finding(str(PROTOCOL_PATH), "protocol doc missing"))
        return findings
    try:
        text = protocol.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        findings.append(Finding(str(PROTOCOL_PATH), f"protocol doc unreadable: {exc}"))
        return findings
    rows = _extract_verb_table(text)

    if not rows:
        findings.append(Finding(str(PROTOCOL_PATH), "verb table not found or empty"))
        return findings

    found_verbs = {r.verb for r in rows}

    # Missing verbs.
    missing = EXPECTED_VERB_NAMES - found_verbs
    if missing:
        findings.append(Finding(str(PROTOCOL_PATH), f"verbs missing from table: {sorted(missing)}"))

    # Unexpected verbs.
    extra = found_verbs - EXPECTED_VERB_NAMES
    if extra:
        findings.append(Finding(str(PROTOCOL_PATH), f"unexpected verbs in table: {sorted(extra)}"))

    # Check PR/Issue applicability for each expected verb.
    row_map = {r.verb: r for r in rows}
    for verb, exp_pr, exp_issue in EXPECTED_VERBS:
        r = row_map.get(verb)
        if not r:
            continue  # already reported as missing
        if r.pr != exp_pr:
            findings.append(
                Finding(
                    str(PROTOCOL_PATH),
                    f"verb '{verb}' PR column should be {'✓' if exp_pr else '—'} but is {'✓' if r.pr else '—'}",
                )
            )
        if r.issue != exp_issue:
            findings.append(
                Finding(
                    str(PROTOCOL_PATH),
                    f"verb '{verb}' Issue column should be {'✓' if exp_issue else '—'} but is "
                    f"{'✓' if r.issue else '—'}",
                )
            )

    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    findings = validate_next_protocol(args.repo_root.resolve())
    has_error = any(f.severity == "error" for f in findings)
    if args.json:
        print(
            json.dumps(
                {"ok": not has_error, "findings": [asdict(item) for item in findings]},
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        for finding in findings:
            tag = "WARN " if finding.severity == "warning" else "ERROR"
            print(f"{tag} {finding.path}: {finding.message}")
        print(f"next protocol validation: {'passed' if not has_error else 'failed'}")
    return 1 if has_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
