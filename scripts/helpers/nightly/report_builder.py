# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

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
