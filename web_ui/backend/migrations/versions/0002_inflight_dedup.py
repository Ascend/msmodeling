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

"""Add partial unique index for in-flight job dedup.

Prevents race condition where two concurrent requests with identical params
both pass the application-level duplicate check before either job is persisted.

The partial index only applies to PENDING/RUNNING jobs, allowing:
- Re-running the same params after SUCCEEDED/FAILED/CANCELLED/INTERRUPTED
- Multiple jobs with different params
- Multiple jobs with same params but different modules

SQLite supports partial indexes via WHERE clause.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op  # pylint: disable=no-name-in-module

revision = "0002_inflight_dedup"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Partial unique index: only one PENDING/RUNNING job per (module_id, params_hash).
    # This prevents the TOCTOU race where two concurrent requests both pass the
    # application-level duplicate check before either job is persisted.
    #
    # The WHERE clause ensures:
    # - Only inflight statuses are constrained (PENDING, RUNNING)
    # - NULL params_hash is excluded (legacy jobs without hash)
    # - Terminal jobs (SUCCEEDED, FAILED, etc.) can have duplicate hashes
    op.create_index(
        "uq_jobs_inflight_params_hash",
        "jobs",
        ["module_id", "params_hash"],
        unique=True,
        sqlite_where=sa.text("status IN ('pending', 'running') AND params_hash IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index("uq_jobs_inflight_params_hash", table_name="jobs")
