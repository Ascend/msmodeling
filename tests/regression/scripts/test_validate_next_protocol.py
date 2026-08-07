"""Tests for validate_next_protocol.py (/next comment protocol validator)."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest

from scripts.ai.validate_next_protocol import (
    EXPECTED_VERB_NAMES,
    _extract_verb_table,
    _parse_yes_dash,
    validate_next_protocol,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_FILE = REPO_ROOT / "spec" / "governance" / "next-comment-protocol.md"


# ---------------------------------------------------------------------------
# _parse_yes_dash
# ---------------------------------------------------------------------------


class TestParseYesDash:
    """Verify ✓/— parsing for PR/Issue applicability cells."""

    @pytest.mark.parametrize(
        "cell,expected",
        [
            ("✓", True),
            ("✅", True),
            ("yes", True),
            ("—", False),
            ("-", False),
            ("no", False),
            ("❌", False),
        ],
    )
    def test_parse(self, cell: str, expected: bool) -> None:
        assert _parse_yes_dash(cell) is expected


# ---------------------------------------------------------------------------
# _extract_verb_table
# ---------------------------------------------------------------------------


SAMPLE_DOC = """\
# /next 协议

## verb 表（7）

| verb | PR | Issue | 含义 | 阈值 |
|---|---|---|---|---|
| `review` | ✓ | ✓ | 请检视 | 24 |
| `ack` | ✓ | ✓ | 确认接手 | 8 |
| `approve` | ✓ | ✓ | 请审批 | 48 |
| `return` | ✓ | ✓ | 退回 | 24 |
| `forward` | ✓ | ✓ | 转交 | 24 |
| `block` | ✓ | ✓ | 阻塞 | 0 |
| `reject` | — | ✓ | wontfix | 0 |

## 范围
"""


class TestExtractVerbTable:
    """Verify verb table extraction from markdown."""

    def test_extracts_all_verbs(self) -> None:
        rows = _extract_verb_table(SAMPLE_DOC)
        verbs = {r.verb for r in rows}
        assert verbs == EXPECTED_VERB_NAMES

    def test_pr_issue_flags(self) -> None:
        rows = _extract_verb_table(SAMPLE_DOC)
        row_map = {r.verb: r for r in rows}
        # reject is Issue-only.
        assert row_map["reject"].pr is False
        assert row_map["reject"].issue is True
        # review is shared.
        assert row_map["review"].pr is True
        assert row_map["review"].issue is True

    def test_empty_doc(self) -> None:
        assert _extract_verb_table("no table here") == []

    def test_skips_header_and_separator(self) -> None:
        rows = _extract_verb_table(SAMPLE_DOC)
        # header row "verb" should not appear as a verb.
        assert "verb" not in {r.verb for r in rows}


# ---------------------------------------------------------------------------
# validate_next_protocol (integration with the real protocol doc)
# ---------------------------------------------------------------------------


class TestValidateNextProtocol:
    """Verify the validator against the real protocol doc + edge cases."""

    def test_real_doc_passes(self) -> None:
        """The committed protocol doc must pass validation."""
        findings = validate_next_protocol(REPO_ROOT)
        errors = [f for f in findings if f.severity == "error"]
        assert not errors, f"protocol doc has errors: {[asdict(f) for f in errors]}"

    def test_missing_file(self, tmp_path: Path) -> None:
        findings = validate_next_protocol(tmp_path)
        assert len(findings) == 1
        assert "missing" in findings[0].message

    def test_unreadable_file(self, tmp_path: Path) -> None:
        """File exists but unreadable → structured Finding, not a crash.

        Uses mock instead of chmod(0o000) because CI containers often run as
        root, where chmod does not prevent reads.
        """
        from unittest.mock import patch

        protocol_dir = tmp_path / "spec" / "governance"
        protocol_dir.mkdir(parents=True)
        protocol_file = protocol_dir / "next-comment-protocol.md"
        protocol_file.write_text(SAMPLE_DOC, encoding="utf-8")
        # Mock read_text to raise PermissionError regardless of user/permissions.
        with patch.object(Path, "read_text", side_effect=PermissionError("mock: denied")):
            findings = validate_next_protocol(tmp_path)
        assert any("unreadable" in f.message for f in findings)

    def test_missing_verb_detected(self, tmp_path: Path) -> None:
        """Remove the 'reject' row → validator should report it missing."""
        protocol_dir = tmp_path / "spec" / "governance"
        protocol_dir.mkdir(parents=True)
        doc = SAMPLE_DOC.replace("| `reject` | — | ✓ | wontfix | 0 |\n", "")
        (protocol_dir / "next-comment-protocol.md").write_text(doc, encoding="utf-8")
        findings = validate_next_protocol(tmp_path)
        messages = " ".join(f.message for f in findings)
        assert "reject" in messages and "missing" in messages

    def test_unexpected_verb_detected(self, tmp_path: Path) -> None:
        """Add a bogus verb → validator should report it as unexpected."""
        protocol_dir = tmp_path / "spec" / "governance"
        protocol_dir.mkdir(parents=True)
        # Insert the bogus row INSIDE the verb table (before the "## 范围" section).
        doc = SAMPLE_DOC.replace(
            "| `reject` | — | ✓ | wontfix | 0 |\n",
            "| `reject` | — | ✓ | wontfix | 0 |\n| `bogus` | ✓ | ✓ | fake | 99 |\n",
        )
        (protocol_dir / "next-comment-protocol.md").write_text(doc, encoding="utf-8")
        findings = validate_next_protocol(tmp_path)
        messages = " ".join(f.message for f in findings)
        assert "bogus" in messages and "unexpected" in messages

    def test_wrong_applicability(self, tmp_path: Path) -> None:
        """Flip reject's PR column to ✓ → validator should report mismatch."""
        protocol_dir = tmp_path / "spec" / "governance"
        protocol_dir.mkdir(parents=True)
        doc = SAMPLE_DOC.replace(
            "| `reject` | — | ✓ | wontfix | 0 |",
            "| `reject` | ✓ | ✓ | wontfix | 0 |",
        )
        (protocol_dir / "next-comment-protocol.md").write_text(doc, encoding="utf-8")
        findings = validate_next_protocol(tmp_path)
        messages = " ".join(f.message for f in findings)
        assert "reject" in messages and "PR" in messages
