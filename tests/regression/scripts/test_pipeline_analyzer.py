"""Regression tests for the repository-bundled openLiBing analyzer."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / ".agents" / "skills" / "gitcode-pipeline-analyzer" / "scripts" / "fetch_pipeline_logs.py"


def load_analyzer() -> ModuleType:
    spec = importlib.util.spec_from_file_location("fetch_pipeline_logs", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_redact_text_removes_credentials() -> None:
    analyzer = load_analyzer()
    source = "Authorization: Bearer abc token=def password=ghi"
    redacted = analyzer.redact_text(source)
    assert "abc" not in redacted
    assert "def" not in redacted
    assert "ghi" not in redacted
    assert "[REDACTED" in redacted


def test_redact_url_preserves_non_sensitive_query() -> None:
    analyzer = load_analyzer()
    redacted = analyzer.redact_url("https://example.test/log?pipelineId=123&signature=secret")
    assert "pipelineId=123" in redacted
    assert "secret" not in redacted
    assert "%5BREDACTED%5D" in redacted


def test_parse_current_pipeline_table() -> None:
    analyzer = load_analyzer()
    table = """
    <table>
      <tr><th>任务名称</th><th>状态</th><th>日志</th></tr>
      <tr>
        <td>unit-test</td><td>FAILED ❌</td>
        <td><a href="https://example.test/task/1">详情</a></td>
      </tr>
    </table>
    """
    assert analyzer.parse_table_rows(table) == [
        {
            "stage": "",
            "task": "unit-test",
            "status": "failed",
            "link": "https://example.test/task/1",
        }
    ]
