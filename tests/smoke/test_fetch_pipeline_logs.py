"""Regression tests for fetch_pipeline_logs pipeline-comment selection.

Covers the review finding: invalidation notices ("source code change are
detected, tasks labels is removed") are independent comments that never enter
``blocks`` (parse_pipeline_comments only keeps trigger comments), so
``choose_block`` must scan the full comment stream to detect them.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / ".agents" / "skills" / "gitcode-pipeline-analyzer" / "scripts" / "fetch_pipeline_logs.py"

_spec = importlib.util.spec_from_file_location("fetch_pipeline_logs", _SCRIPT)
assert _spec is not None and _spec.loader is not None
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def _trigger_comment(cid: int, when: str, pid: str = "p1", task: str = "UT_x", status: str = "FAILED") -> str:
    return (
        f"#{cid}) ID: {cid}\n"
        "流水线任务触发成功\n"
        f"Author: bot at {when}\n"
        f"https://www.openlibing.com/apps/pipelineDetail?pipelineId={pid}&pipelineRunId=r{cid}\n"
        "<table><tr><th>任务名称</th><th>状态</th></tr>"
        f"<tr><td>{task}</td><td>{status}</td></tr></table>"
    )


def _invalidation_comment(cid: int, when: str) -> str:
    return f"#{cid}) ID: {cid}\nsource code change are detected, tasks labels is removed\nAuthor: bot at {when}"


def test_invalidation_notice_stays_out_of_blocks():
    comments = "\n".join(
        [
            _trigger_comment(1, "2024-01-01 10:00", pid="p1"),
            _invalidation_comment(2, "2024-01-01 11:00"),
        ]
    )
    blocks = mod.parse_pipeline_comments(comments)
    assert len(blocks) == 1
    assert blocks[0]["comment_time"] == "2024-01-01 10:00"


def test_choose_block_skips_invalidated_pipeline_via_independent_notice():
    comments = "\n".join(
        [
            _trigger_comment(1, "2024-01-01 10:00", pid="p1"),
            _invalidation_comment(2, "2024-01-01 11:00"),
            _trigger_comment(3, "2024-01-01 12:00", pid="p2"),
        ]
    )
    blocks = mod.parse_pipeline_comments(comments)
    assert len(blocks) == 2
    block = mod.choose_block(blocks, comments)
    assert block is not None
    assert block["comment_time"] == "2024-01-01 12:00"
    assert "p2" in block["link"]


def test_choose_block_returns_latest_when_no_invalidation():
    comments = "\n".join(
        [
            _trigger_comment(1, "2024-01-01 10:00", pid="p1"),
            _trigger_comment(2, "2024-01-01 12:00", pid="p2"),
        ]
    )
    blocks = mod.parse_pipeline_comments(comments)
    block = mod.choose_block(blocks, comments)
    assert block is not None
    assert block["comment_time"] == "2024-01-01 12:00"


def test_choose_block_without_comments_text_is_legacy_and_cannot_invalidate():
    comments = "\n".join(
        [
            _trigger_comment(1, "2024-01-01 10:00", pid="p1"),
            _invalidation_comment(2, "2024-01-01 11:00"),
        ]
    )
    blocks = mod.parse_pipeline_comments(comments)
    assert len(blocks) == 1
    block = mod.choose_block(blocks)
    assert block is not None
    assert block["comment_time"] == "2024-01-01 10:00"
