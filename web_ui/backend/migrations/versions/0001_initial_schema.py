"""Initial schema for the web_ui: all tables, indexes, and module seed.

Tables: modules, form_schemas, jobs, result_records, feedbacks,
telemetry_events, case_logs.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op  # pylint: disable=no-name-in-module

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None

_MODULES = [
    ("text_generate", "Text Generation", "ModelRunner", "Single-pass text generation profiling."),
    ("video_generate", "Video Generation", "VideoGenerateRunner", "Diffusers video/diffusion inference profiling."),
    (
        "throughput_optimizer",
        "Throughput Optimizer",
        "ParallelRunner",
        "Serving throughput optimization across parallel/search configs.",
    ),
]


def upgrade() -> None:
    # 1. modules
    op.create_table(
        "modules",
        sa.Column("id", sa.String, primary_key=True),
        sa.Column("display_name", sa.String, nullable=False),
        sa.Column("runner_class", sa.String, nullable=False),
        sa.Column("description", sa.String, nullable=True),
        sa.Column(
            "created_at", sa.String, nullable=False, server_default=sa.text("(strftime('%Y-%m-%dT%H:%M:%SZ','now'))")
        ),
    )
    op.bulk_insert(
        sa.table(
            "modules",
            sa.column("id", sa.String),
            sa.column("display_name", sa.String),
            sa.column("runner_class", sa.String),
            sa.column("description", sa.String),
        ),
        [{"id": i, "display_name": d, "runner_class": r, "description": desc} for i, d, r, desc in _MODULES],
    )

    # 2. form_schemas
    op.create_table(
        "form_schemas",
        sa.Column("module_id", sa.String, sa.ForeignKey("modules.id"), primary_key=True),
        sa.Column("version", sa.String, primary_key=True),
        sa.Column("schema_hash", sa.String, nullable=False),
        sa.Column("fields", sa.String, nullable=False),
        sa.Column(
            "created_at", sa.String, nullable=False, server_default=sa.text("(strftime('%Y-%m-%dT%H:%M:%SZ','now'))")
        ),
    )

    # 3. jobs
    op.create_table(
        "jobs",
        sa.Column("id", sa.String, primary_key=True),
        sa.Column("module_id", sa.String, sa.ForeignKey("modules.id"), nullable=False),
        sa.Column("status", sa.String, nullable=False),
        sa.Column("progress", sa.Integer, nullable=True),
        sa.Column("progress_text", sa.String, nullable=True),
        sa.Column("params", sa.String, nullable=False),
        sa.Column("form_schema_version", sa.String, nullable=False),
        sa.Column("label", sa.String, nullable=True),
        sa.Column("error", sa.String, nullable=True),
        sa.Column("error_detail", sa.String, nullable=True),
        sa.Column("params_hash", sa.Text, nullable=True),
        sa.Column("log_text", sa.Text, nullable=True),
        sa.Column(
            "created_at", sa.String, nullable=False, server_default=sa.text("(strftime('%Y-%m-%dT%H:%M:%SZ','now'))")
        ),
        sa.Column("started_at", sa.String, nullable=True),
        sa.Column("completed_at", sa.String, nullable=True),
        sa.CheckConstraint(
            "status IN ('pending','running','succeeded','failed','cancelled','interrupted')",
            name="ck_jobs_status",
        ),
        sa.CheckConstraint(
            "progress IS NULL OR progress BETWEEN 0 AND 100",
            name="ck_jobs_progress",
        ),
    )
    op.create_index("idx_jobs_module_status", "jobs", ["module_id", "status"])
    op.create_index("idx_jobs_created_at", "jobs", [sa.text("created_at DESC")])
    op.create_index("ix_jobs_params_hash", "jobs", ["params_hash"])

    # 4. result_records
    op.create_table(
        "result_records",
        sa.Column("id", sa.String, primary_key=True),
        sa.Column("job_id", sa.String, sa.ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("seq", sa.Integer, nullable=False),
        sa.Column("rank", sa.Integer, nullable=True),
        sa.Column("config", sa.String, nullable=False),
        sa.Column("summary", sa.String, nullable=False),
        sa.Column("tables", sa.String, nullable=False, server_default="{}"),
        sa.Column("case_hash", sa.Text, nullable=True),
        sa.Column(
            "created_at", sa.String, nullable=False, server_default=sa.text("(strftime('%Y-%m-%dT%H:%M:%SZ','now'))")
        ),
        sa.UniqueConstraint("job_id", "seq", name="uq_result_records_job_seq"),
    )
    op.create_index("idx_result_records_job", "result_records", ["job_id", "seq"])
    op.create_index("idx_result_records_rank", "result_records", ["job_id", "rank"])
    op.create_index("ix_result_records_case_hash", "result_records", ["case_hash"])

    # 5. feedbacks
    op.create_table(
        "feedbacks",
        sa.Column("id", sa.String, primary_key=True),
        sa.Column("job_id", sa.String, sa.ForeignKey("jobs.id"), nullable=True),
        sa.Column("module_id", sa.String, nullable=True),
        sa.Column("kind", sa.String, nullable=False),
        sa.Column("content_text", sa.Text, nullable=True),
        sa.Column("screenshot_path", sa.Text, nullable=True),
        sa.Column("fingerprint", sa.String, nullable=True),
        sa.Column(
            "created_at",
            sa.String,
            nullable=False,
            server_default=sa.text("(strftime('%Y-%m-%dT%H:%M:%SZ','now'))"),
        ),
    )
    op.create_index("ix_feedbacks_job_id", "feedbacks", ["job_id"])
    op.create_index("ix_feedbacks_module_id", "feedbacks", ["module_id"])
    op.create_index("ix_feedbacks_fingerprint", "feedbacks", ["fingerprint"])
    op.create_index("ix_feedbacks_created_at", "feedbacks", ["created_at"])

    # 6. telemetry_events
    op.create_table(
        "telemetry_events",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("module_id", sa.Text, nullable=False),
        sa.Column("target", sa.Text, nullable=False),
        sa.Column("event_type", sa.Text, nullable=False, server_default="change"),
        sa.Column("fingerprint", sa.Text, nullable=True),
        sa.Column(
            "created_at",
            sa.Text,
            nullable=False,
            server_default=sa.text("(strftime('%Y-%m-%dT%H:%M:%SZ','now'))"),
        ),
    )
    op.create_index("ix_telemetry_events_module_id", "telemetry_events", ["module_id"])
    op.create_index("ix_telemetry_events_target", "telemetry_events", ["target"])
    op.create_index("ix_telemetry_events_created_at", "telemetry_events", ["created_at"])
    op.create_index("ix_telemetry_events_fingerprint", "telemetry_events", ["fingerprint"])

    # 7. case_logs
    op.create_table(
        "case_logs",
        sa.Column("case_hash", sa.Text, primary_key=True),
        sa.Column("content", sa.Text, nullable=False, server_default=""),
        sa.Column(
            "created_at",
            sa.Text,
            nullable=False,
            server_default=sa.text("(strftime('%Y-%m-%dT%H:%M:%SZ','now'))"),
        ),
    )


def downgrade() -> None:
    op.drop_table("case_logs")
    op.drop_index("ix_telemetry_events_fingerprint", table_name="telemetry_events")
    op.drop_index("ix_telemetry_events_created_at", table_name="telemetry_events")
    op.drop_index("ix_telemetry_events_target", table_name="telemetry_events")
    op.drop_index("ix_telemetry_events_module_id", table_name="telemetry_events")
    op.drop_table("telemetry_events")
    op.drop_index("ix_feedbacks_created_at", table_name="feedbacks")
    op.drop_index("ix_feedbacks_fingerprint", table_name="feedbacks")
    op.drop_index("ix_feedbacks_module_id", table_name="feedbacks")
    op.drop_index("ix_feedbacks_job_id", table_name="feedbacks")
    op.drop_table("feedbacks")
    op.drop_index("ix_result_records_case_hash", table_name="result_records")
    op.drop_index("idx_result_records_rank", table_name="result_records")
    op.drop_index("idx_result_records_job", table_name="result_records")
    op.drop_table("result_records")
    op.drop_index("ix_jobs_params_hash", table_name="jobs")
    op.drop_index("idx_jobs_created_at", table_name="jobs")
    op.drop_index("idx_jobs_module_status", table_name="jobs")
    op.drop_table("jobs")
    op.drop_table("form_schemas")
    op.drop_table("modules")
