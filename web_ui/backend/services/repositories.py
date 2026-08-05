"""Repository Protocol interfaces + concrete SQLModel-backed repositories.

Flattened from the former DDD ``application/ports/`` (Protocol interfaces) and
``infrastructure/persistence/repositories/`` (concrete classes). SQLModel is
imported lazily inside each method so the module (and the app) imports without
the heavy simulation stack.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Iterable, Protocol, runtime_checkable
from uuid import uuid4

from models.entities import CapabilityModule, Job, ResultRecord
from models.enums import JobStatus

_logger = logging.getLogger(__name__)


def _new_record_id() -> str:
    """Fresh id for a cloned result record (ResultRecordRow.id has no default)."""
    return uuid4().hex


@runtime_checkable
class SchemaRegistryPort(Protocol):  # pragma: no cover - Protocol class; body is type signatures only, no runtime code
    """Snapshot registry for form schemas."""

    def upsert_form_schema(self, module_id: str, version: str, envelope: dict, schema_hash: str) -> None:
        """Insert a snapshot (full envelope) or refuse on hash mismatch for the same version."""

    def get_form_schema(self, module_id: str, version: str | None = None) -> dict | None: ...


class SchemaMismatchError(RuntimeError):
    """Raised when a bundled config's hash differs from the stored snapshot
    for the SAME version (a changed file must bump version).
    """


@runtime_checkable
class RunnerPort(Protocol):  # pragma: no cover - Protocol class; body is type signatures only, no runtime code
    """The interface every runner adapter implements."""

    def run(self, params: dict[str, Any], *, on_progress=None, cancel_flag=None) -> list[ResultRecord]:
        """Run the simulation and return normalized result records.

        ``on_progress(progress: int | None, text: str | None)`` is called with
        optimizer percent or text/video milestone updates. ``cancel_flag`` is a
        callable returning ``True`` when cancel was requested (cooperative).
        """


# === Concrete repositories ============================================================


def _imports():
    """Lazy accessor for orm + session_scope (imported on first DB use)."""
    from models import orm
    from db import session_scope

    return orm, session_scope


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _row_to_job(row) -> Job:
    """Map a ``JobRow`` (persistence) onto a domain ``Job`` (params JSON decoded)."""
    return Job(
        id=row.id,
        module_id=row.module_id,
        params=json.loads(row.params) if row.params else {},
        form_schema_version=row.form_schema_version,
        status=JobStatus(row.status),
        label=row.label,
        progress=row.progress,
        progress_text=row.progress_text,
        error=row.error,
        error_detail=row.error_detail,
        created_at=row.created_at,
        started_at=row.started_at,
        completed_at=row.completed_at,
        params_hash=getattr(row, "params_hash", None),
        log_text=getattr(row, "log_text", None),
    )


def _row_to_module(row) -> CapabilityModule:
    """Map a ``ModuleRow`` (persistence) onto a domain ``CapabilityModule``."""
    return CapabilityModule(
        id=row.id,
        display_name=row.display_name,
        runner_class=row.runner_class,
        description=row.description,
    )


class JobRepository:
    """Concrete repository over the SQLite ``modules``/``jobs`` tables."""

    def get_module(self, module_id: str) -> CapabilityModule | None:
        """Return a capability module by id, or ``None`` if not seeded."""

        orm, session_scope = _imports()
        with session_scope() as session:
            row = session.get(orm.ModuleRow, module_id)
            return _row_to_module(row) if row else None

    def list_modules(self) -> list[CapabilityModule]:
        """Return all seeded capability modules."""
        from sqlmodel import select

        orm, session_scope = _imports()
        with session_scope() as session:
            rows = session.exec(select(orm.ModuleRow)).all()
            return [_row_to_module(r) for r in rows]

    def seed_modules(self) -> int:
        """Idempotently seed the 3 capability modules if the table is empty.

        Returns the number of modules inserted (0 if already seeded). The
        ``form_schemas`` table has a FK to ``modules.id``, so modules MUST be
        seeded before schema snapshots are upserted at startup.
        """
        from sqlmodel import select

        orm, session_scope = _imports()
        with session_scope() as session:
            existing = session.exec(select(orm.ModuleRow)).all()
            if existing:
                return 0
            seeds = [
                orm.ModuleRow(
                    id="text_generate",
                    display_name="Text Generation",
                    runner_class="ModelRunner",
                    description="Estimate per-device TPS, memory, and operator breakdowns for text models.",
                ),
                orm.ModuleRow(
                    id="video_generate",
                    display_name="Video Generation",
                    runner_class="VideoGenerateRunner",
                    description="Profile execution time and per-operator cost for video generation models.",
                ),
                orm.ModuleRow(
                    id="throughput_optimizer",
                    display_name="Throughput Optimizer",
                    runner_class="ParallelRunner",
                    description="Sweep parallel/concurrency configs to find the best throughput per device.",
                ),
            ]
            for row in seeds:
                session.add(row)
            session.commit()
            return len(seeds)

    def get(self, job_id: str) -> Job | None:
        """Return a job by id, or ``None`` if no such row exists."""
        orm, session_scope = _imports()
        with session_scope() as session:
            row = session.get(orm.JobRow, job_id)
            return _row_to_job(row) if row else None

    def find_succeeded_by_params_hash(self, module_id: str, params_hash: str) -> Job | None:
        """Return the most recent SUCCEEDED job with the same (module_id,
        params_hash), or ``None``. Used by Phase C result+log caching: a hit
        means an identical run already succeeded and can be reused as-is.
        """
        from sqlmodel import select

        orm, session_scope = _imports()
        with session_scope() as session:
            stmt = (
                select(orm.JobRow)
                .where(orm.JobRow.module_id == module_id)
                .where(orm.JobRow.params_hash == params_hash)
                .where(orm.JobRow.status == JobStatus.SUCCEEDED.value)
                .order_by(orm.JobRow.completed_at.desc())
            )
            row = session.exec(stmt).first()
            return _row_to_job(row) if row else None

    def add(self, job: Job) -> Job:
        """Insert a new job row and return the persisted ``Job`` (with row id)."""
        orm, session_scope = _imports()
        with session_scope() as session:
            row = orm.JobRow(
                id=job.id,
                module_id=job.module_id,
                status=job.status.value,
                progress=job.progress,
                progress_text=job.progress_text,
                params=json.dumps(job.params, ensure_ascii=False),
                form_schema_version=job.form_schema_version,
                label=job.label,
                error=job.error,
                created_at=job.created_at or _utcnow_iso(),
                started_at=job.started_at,
                completed_at=job.completed_at,
            )
            session.add(row)
            session.commit()
            session.refresh(row)
            return _row_to_job(row)

    def add_many(self, jobs: Iterable[Job]) -> list[Job]:
        """Insert multiple jobs in a single transaction (executemany semantics).

        Each job MUST already have its id assigned. All rows share one commit
        so the whole batch lands atomically.
        """
        orm, session_scope = _imports()
        with session_scope() as session:
            rows = []
            for job in jobs:
                row = orm.JobRow(
                    id=job.id,
                    module_id=job.module_id,
                    status=job.status.value,
                    progress=job.progress,
                    progress_text=job.progress_text,
                    params=json.dumps(job.params, ensure_ascii=False),
                    form_schema_version=job.form_schema_version,
                    label=job.label,
                    error=job.error,
                    created_at=job.created_at or _utcnow_iso(),
                    started_at=job.started_at,
                    completed_at=job.completed_at,
                )
                session.add(row)
                rows.append(row)
            session.commit()
            for row in rows:
                session.refresh(row)
            return [_row_to_job(r) for r in rows]

    def update(self, job_id: str, **changes: Any) -> Job | None:
        """Patch selected job fields; return the updated ``Job`` or ``None``.

        Only non-None ``changes`` are written (callers pass explicit values),
        so an omitted field is never clobbered. A ``JobStatus`` value is
        serialized to its string column form. Status transitions are checked
        against the state machine (``models.enums.can_transition``); an illegal
        move is logged as a warning (and still applied) so a latent bug surfaces
        loudly without bricking the run loop.
        """
        # Only non-None changes are written (callers pass explicit values).
        patch = {k: v for k, v in changes.items() if v is not None}
        if "status" in patch and isinstance(patch["status"], JobStatus):
            patch["status"] = patch["status"].value
        orm, session_scope = _imports()
        with session_scope() as session:
            row = session.get(orm.JobRow, job_id)
            if row is None:
                return None
            # State-machine guard: surface illegal moves.
            if "status" in patch and patch["status"] != row.status:
                from models.enums import can_transition

                try:
                    _from = JobStatus(row.status)
                    _to = JobStatus(patch["status"])
                    if not can_transition(_from, _to):
                        _logger.warning(
                            "Illegal job transition %s -> %s for job %s",
                            row.status,
                            patch["status"],
                            job_id,
                        )
                except ValueError:
                    # Unknown status string — let the DB CHECK constraint catch it.
                    pass
            for key, value in patch.items():
                setattr(row, key, value)
            session.add(row)
            session.commit()
            session.refresh(row)
            return _row_to_job(row)

    def list_jobs(
        self,
        *,
        module_id: str | None = None,
        status: JobStatus | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Job]:
        """Page through jobs, newest first, optionally filtered by module/status."""
        from sqlmodel import select

        orm, session_scope = _imports()
        stmt = select(orm.JobRow)
        if module_id:
            stmt = stmt.where(orm.JobRow.module_id == module_id)
        if status:
            stmt = stmt.where(orm.JobRow.status == status.value)
        stmt = stmt.order_by(orm.JobRow.created_at.desc()).offset(offset).limit(limit)
        with session_scope() as session:
            rows = session.exec(stmt).all()
            return [_row_to_job(r) for r in rows]

    def count_jobs(
        self,
        *,
        module_id: str | None = None,
        status: JobStatus | None = None,
    ) -> int:
        """Count jobs matching the same filter used by ``list_jobs`` (no paging)."""
        from sqlmodel import func, select

        orm, session_scope = _imports()
        stmt = select(func.count()).select_from(orm.JobRow)
        if module_id:
            stmt = stmt.where(orm.JobRow.module_id == module_id)
        if status:
            stmt = stmt.where(orm.JobRow.status == status.value)
        with session_scope() as session:
            return int(session.exec(stmt).one())

    def sweep_interrupted(self) -> int:
        """Startup sweep: pending/running -> interrupted (server died mid-run)."""
        from sqlalchemy import update

        orm, session_scope = _imports()
        with session_scope() as session:
            result = session.exec(
                update(orm.JobRow.__table__)  # type: ignore[arg-type]
                .where(orm.JobRow.status.in_(["pending", "running"]))
                .values(status=JobStatus.INTERRUPTED.value)
            )
            session.commit()
            return getattr(result, "rowcount", 0) or 0


def _row_to_record(row) -> ResultRecord:
    """Map a ``ResultRecordRow`` onto a domain ``ResultRecord`` (JSON fields decoded)."""
    return ResultRecord(
        id=row.id,
        job_id=row.job_id,
        seq=row.seq,
        rank=row.rank,
        config=json.loads(row.config) if row.config else {},
        summary=json.loads(row.summary) if row.summary else {},
        tables=json.loads(row.tables) if row.tables else {},
        case_hash=getattr(row, "case_hash", None),
        created_at=row.created_at,
    )


class ResultRepository:
    """Concrete repository over the SQLite ``result_records`` table."""

    def add(self, record: ResultRecord) -> ResultRecord:
        """Insert one result record; return the persisted ``ResultRecord``."""
        orm, session_scope = _imports()
        with session_scope() as session:
            row = orm.ResultRecordRow(
                id=record.id,
                job_id=record.job_id,
                seq=record.seq,
                rank=record.rank,
                config=json.dumps(record.config, ensure_ascii=False),
                summary=json.dumps(record.summary, ensure_ascii=False),
                tables=json.dumps(record.tables, ensure_ascii=False),
                case_hash=record.case_hash,
                created_at=record.created_at,
            )
            session.add(row)
            session.commit()
            session.refresh(row)
            return _row_to_record(row)

    def add_many(self, records: Iterable[ResultRecord]) -> list[ResultRecord]:
        """Insert several result records in one transaction; return all persisted rows."""
        orm, session_scope = _imports()
        rows = []
        with session_scope() as session:
            for rec in records:
                row = orm.ResultRecordRow(
                    id=rec.id,
                    job_id=rec.job_id,
                    seq=rec.seq,
                    rank=rec.rank,
                    config=json.dumps(rec.config, ensure_ascii=False),
                    summary=json.dumps(rec.summary, ensure_ascii=False),
                    tables=json.dumps(rec.tables, ensure_ascii=False),
                    case_hash=rec.case_hash,
                )
                session.add(row)
                rows.append(row)
            session.commit()
            for row in rows:
                session.refresh(row)
            return [_row_to_record(r) for r in rows]

    def list_for_job(self, job_id: str) -> list[ResultRecord]:
        """Return a job's result records ordered by stable ``seq`` (capture order)."""
        from sqlmodel import select

        orm, session_scope = _imports()
        with session_scope() as session:
            stmt = (
                select(orm.ResultRecordRow)
                .where(orm.ResultRecordRow.job_id == job_id)
                .order_by(orm.ResultRecordRow.seq)
            )
            return [_row_to_record(r) for r in session.exec(stmt).all()]

    def clone_records(self, src_job_id: str, dst_job_id: str) -> int:
        """Copy all result records from ``src_job_id`` to ``dst_job_id`` (new row
        ids, dst job_id, preserved seq/rank). Used by Phase C cache hits to reuse
        a prior succeeded job's results without re-running. Returns the count.
        """
        from sqlmodel import select

        orm, session_scope = _imports()
        with session_scope() as session:
            stmt = (
                select(orm.ResultRecordRow)
                .where(orm.ResultRecordRow.job_id == src_job_id)
                .order_by(orm.ResultRecordRow.seq)
            )
            src_rows = session.exec(stmt).all()
            for r in src_rows:
                session.add(
                    orm.ResultRecordRow(
                        id=_new_record_id(),
                        job_id=dst_job_id,
                        seq=r.seq,
                        rank=r.rank,
                        config=r.config,
                        summary=r.summary,
                        tables=r.tables,
                        case_hash=r.case_hash,
                    )
                )
            session.commit()
            return len(src_rows)

    def succeeded_case_hashes_for_module(self, module_id: str) -> set[str]:
        """Distinct ``case_hash`` values among SUCCEEDED records of ``module_id``
        jobs. Seeds the worker's case-dedup skip set. ``case_hash`` already encodes
        the form-schema version, so no extra version filter is needed.
        """
        from sqlmodel import text

        _, session_scope = _imports()
        sql = text(
            "SELECT DISTINCT rr.case_hash FROM result_records rr "
            "JOIN jobs j ON j.id = rr.job_id "
            "WHERE j.module_id = :module_id AND j.status = 'succeeded' "
            "AND rr.case_hash IS NOT NULL"
        )
        with session_scope() as session:
            rows = session.execute(sql, {"module_id": module_id}).all()
            return {r[0] for r in rows if r[0]}

    def find_succeeded_record_ids_by_case_hash(self, case_hash: str) -> list[str]:
        """Return the record ids of the most recent SUCCEEDED job's records sharing
        ``case_hash`` (ordered by seq). Used to clone a cached case's records.
        """
        from sqlmodel import text

        _, session_scope = _imports()
        with session_scope() as session:
            job_row = session.execute(
                text(
                    "SELECT j.id FROM jobs j JOIN result_records rr ON rr.job_id = j.id "
                    "WHERE rr.case_hash = :ch AND j.status = 'succeeded' "
                    "ORDER BY j.completed_at DESC LIMIT 1"
                ),
                {"ch": case_hash},
            ).first()
            if not job_row:
                return []
            rows = session.execute(
                text("SELECT id FROM result_records WHERE job_id = :jid AND case_hash = :ch ORDER BY seq"),
                {"jid": job_row[0], "ch": case_hash},
            ).all()
            return [r[0] for r in rows]

    def get_succeeded_records_by_case_hash(self, case_hash: str) -> list[ResultRecord]:
        """Return the records (as domain entities) of the most recent SUCCEEDED
        job sharing ``case_hash`` (ordered by seq). Case-level dedup uses this to
        reuse a cached case's data — the caller re-assigns id/job_id/seq and
        re-ranks before persisting, so no PK collision.
        """
        from sqlmodel import select, text

        orm, session_scope = _imports()
        with session_scope() as session:
            job_row = session.execute(
                text(
                    "SELECT j.id FROM jobs j JOIN result_records rr ON rr.job_id = j.id "
                    "WHERE rr.case_hash = :ch AND j.status = 'succeeded' "
                    "ORDER BY j.completed_at DESC LIMIT 1"
                ),
                {"ch": case_hash},
            ).first()
            if not job_row:
                return []
            stmt = (
                select(orm.ResultRecordRow)
                .where(orm.ResultRecordRow.job_id == job_row[0])
                .where(orm.ResultRecordRow.case_hash == case_hash)
                .order_by(orm.ResultRecordRow.seq)
            )
            return [_row_to_record(r) for r in session.exec(stmt).all()]

    def clone_records_by_ids(self, src_record_ids: list[str], dst_job_id: str) -> int:
        """Clone specific result records (by id) into ``dst_job_id`` (new ids,
        preserved seq/rank/case_hash). Case-level dedup uses this to reuse a cached
        case's records. Returns the count cloned.
        """
        from sqlmodel import select

        orm, session_scope = _imports()
        with session_scope() as session:
            if not src_record_ids:
                return 0
            rows = session.exec(
                select(orm.ResultRecordRow)
                .where(orm.ResultRecordRow.id.in_(tuple(src_record_ids)))
                .order_by(orm.ResultRecordRow.seq)
            ).all()
            for r in rows:
                session.add(
                    orm.ResultRecordRow(
                        id=_new_record_id(),
                        job_id=dst_job_id,
                        seq=r.seq,
                        rank=r.rank,
                        config=r.config,
                        summary=r.summary,
                        tables=r.tables,
                        case_hash=r.case_hash,
                    )
                )
            session.commit()
            return len(rows)


class CaseLogRepository:
    """Persistence for per-case CLI logs (``case_logs`` table).

    Replaces the regex-splitting of ``{job_id}.log``: each case's output is
    captured at run time and upserted here keyed by ``case_hash``. Case-level
    dedup reuses a prior run's log via ``get(case_hash)`` — no re-extraction.
    """

    def upsert(self, case_hash: str, content: str) -> None:
        """Insert or replace the log for ``case_hash``."""
        orm, session_scope = _imports()
        with session_scope() as session:
            row = session.get(orm.CaseLogRow, case_hash)
            if row is None:
                session.add(orm.CaseLogRow(case_hash=case_hash, content=content))
            else:
                row.content = content
                session.add(row)
            session.commit()

    def upsert_many(self, items: dict[str, str]) -> int:
        """Bulk upsert {case_hash: content}; return the number written."""
        if not items:
            return 0
        orm, session_scope = _imports()
        with session_scope() as session:
            for case_hash, content in items.items():
                row = session.get(orm.CaseLogRow, case_hash)
                if row is None:
                    session.add(orm.CaseLogRow(case_hash=case_hash, content=content))
                else:
                    row.content = content
                    session.add(row)
            session.commit()
        return len(items)

    def get(self, case_hash: str) -> str | None:
        """Return the stored log for ``case_hash`` (None if absent)."""
        orm, session_scope = _imports()
        with session_scope() as session:
            row = session.get(orm.CaseLogRow, case_hash)
            return row.content if row is not None else None
