# Copyright (c) 2026-2026 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Self-contained human-readable HTML reports."""

from __future__ import annotations

import html
import os
import tempfile
from pathlib import Path

from tools.model_diagnostics.domain import DiagnosticsResult, FindingStatus, SimulationExecutionArtifact

from .formatting import context_line, display_actual, display_expected, finding_location

_STYLE = """
body{font:14px/1.45 system-ui,sans-serif;margin:2rem;color:#202124}h1{margin-bottom:.25rem}
.meta{color:#5f6368}.toolbar{position:sticky;top:0;background:white;padding:.75rem 0;border-bottom:1px solid #ddd}
input{width:min(36rem,80%);padding:.5rem}button{margin-left:.4rem;padding:.5rem}.card{border:1px solid #ddd;border-radius:8px;margin:.7rem 0;padding:.75rem}
summary{cursor:pointer;font-weight:600}.pass{border-left:5px solid #188038}.fail,.incomplete,.unsupported,.skip{border-left:5px solid #d93025}
table{border-collapse:collapse;width:100%;margin-top:.5rem}th,td{text-align:left;border-bottom:1px solid #eee;padding:.35rem;vertical-align:top}code{white-space:pre-wrap}
"""
_SCRIPT = """
const q=document.querySelector('#search');q?.addEventListener('input',()=>{const s=q.value.toLowerCase();document.querySelectorAll('.card').forEach(x=>x.hidden=!x.textContent.toLowerCase().includes(s));});
function toggle(open){document.querySelectorAll('details').forEach(x=>x.open=open)}
"""


def _page(title: str, body: str) -> str:
    return f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title><style>{_STYLE}</style></head><body>{body}<script>{_SCRIPT}</script></body></html>\n"


def _context(context) -> str:
    phase = context.phase.value if context.phase is not None else "?"
    tp = context.parallel.tensor_parallel_size
    ctx = 0 if context.context_length is None else context.context_length
    return f"{context.model_name} | {phase} | batch={context.batch_size} query={context.query_length} context={ctx} | TP={tp}"


class RuntimeHtmlRenderer:
    """Render every captured Runtime call and tensor slot."""

    def render(self, artifact: SimulationExecutionArtifact) -> str:
        cards = []
        for call in artifact.operator_calls:
            rows = "".join(
                f"<tr><td>{html.escape(str(tensor.slot))}</td><td><code>{html.escape(str(tensor.shape))}</code></td><td>{html.escape(tensor.dtype)}</td></tr>"
                for tensor in call.tensors
            )
            cards.append(
                f"<details class='card'><summary>#{call.call_index} {html.escape(call.operator_name)}</summary>"
                f"<div class='meta'>{html.escape(call.source_reference or '')}</div><table><thead><tr><th>Slot</th><th>Shape</th><th>Dtype</th></tr></thead><tbody>{rows}</tbody></table></details>"
            )
        body = (
            "<h1>Runtime execution report</h1>"
            f"<p class='meta'>{html.escape(_context(artifact.run_context))}</p>"
            f"<p>{len(artifact.operator_calls)} operator calls · backend {html.escape(artifact.producer.capture_backend)}</p>"
            "<div class='toolbar'><input id='search' placeholder='Filter operators, shapes or dtypes'>"
            "<button onclick='toggle(true)'>Expand all</button><button onclick='toggle(false)'>Collapse all</button></div>"
            + "".join(cards)
        )
        return _page("Runtime execution report", body)


class ComparisonHtmlRenderer:
    """Render a complete Theory-to-Runtime comparison result."""

    def render(self, result: DiagnosticsResult) -> str:
        cards = []
        for finding in result.findings:
            evidence = (*finding.left_evidence, *finding.right_evidence)
            refs = "<br>".join(
                html.escape(f"{item.source_kind.value} call[{item.call_index}] {item.operator_name} {item.tensor_slot or ''}")
                for item in evidence
            ) or "-"
            cards.append(
                f"<section class='card {finding.status.value}'><strong>{finding.status.value.upper()} · {html.escape(finding_location(finding))}</strong>"
                f"<table><tr><th>Expected</th><td><code>{html.escape(display_expected(finding) or '-')}</code></td></tr>"
                f"<tr><th>Actual</th><td><code>{html.escape(display_actual(finding) or '-')}</code></td></tr>"
                f"<tr><th>Message</th><td>{html.escape(finding.message)}</td></tr><tr><th>Evidence</th><td>{refs}</td></tr></table></section>"
            )
        counts = result.summary.counts_by_status
        failed = sum(counts[s] for s in FindingStatus if s is not FindingStatus.PASS)
        limitations = "".join(f"<li><strong>{html.escape(x.code)}</strong>: {html.escape(x.message)}</li>" for x in result.limitations) or "<li>none</li>"
        outcome = "PASS" if result.summary.overall_status is FindingStatus.PASS else "FAIL"
        body = (
            f"<h1>Model diagnostics: {outcome}</h1>"
            f"<p class='meta'>{html.escape(context_line(result))}</p><p>Summary: {counts[FindingStatus.PASS]} pass, {failed} fail</p>"
            "<div class='toolbar'><input id='search' placeholder='Filter findings'></div>" + "".join(cards) + f"<h2>Limitations</h2><ul>{limitations}</ul>"
        )
        return _page("Theory-to-Runtime comparison", body)


def write_html_report(path: Path, content: str) -> None:
    """Atomically write one HTML report, creating only its parent directories."""

    if path.exists() and path.is_dir():
        raise ValueError(f"report path is a directory: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
