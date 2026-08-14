"""Nightly summary helpers: git env collection."""

from __future__ import annotations

from datetime import datetime

try:
    from datetime import UTC
except ImportError:
    from datetime import timezone

    UTC = timezone.utc

from scripts.helpers._paths import REPO_ROOT
from scripts.helpers.ci_gate.diff import git_stdout
from scripts.helpers.nightly.report_models import EnvInfo


def fetch_env_info() -> EnvInfo:
    commit = git_stdout(REPO_ROOT, "rev-parse", "--short", "HEAD") or "unknown"
    branch = git_stdout(REPO_ROOT, "branch", "--show-current") or "unknown"
    timestamp = datetime.now(UTC).isoformat()
    return EnvInfo(commit=commit, branch=branch, timestamp=timestamp)
