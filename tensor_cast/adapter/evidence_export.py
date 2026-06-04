import json
from pathlib import Path
from typing import Optional

import yaml


def load_evidence_draft_from_doctor_report(report_path: str) -> dict:
    path = Path(report_path)
    report = json.loads(path.read_text(encoding="utf-8"))
    evidence_draft = report.get("evidence_draft")
    if not evidence_draft:
        raise ValueError(f"No evidence_draft found in doctor report: {path}")
    if not isinstance(evidence_draft, dict):
        raise ValueError(f"doctor report evidence_draft must be an object: {path}")
    return evidence_draft


def export_evidence_from_doctor_report(report_path: str, output_path: Optional[str] = None) -> str:
    evidence_draft = load_evidence_draft_from_doctor_report(report_path)
    content = yaml.safe_dump(evidence_draft, allow_unicode=True, sort_keys=False)
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return content
