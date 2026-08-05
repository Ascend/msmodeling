"""Real unit tests for services/repositories.py.

Uses a throwaway in-memory SQLite DB (same pattern as the backend integration
conftest) so every repository method hits a real SQLModel session — no mocking
of the persistence layer. This is the meaningful way to unit-test repos: the
DB is the SUT. Per tests/SKILL.md (real project deps, fixture-scoped setup,
no sys.modules mocking).
"""

from __future__ import annotations

import db
import pytest
from models.entities import CapabilityModule, Job, ResultRecord
from models.enums import JobStatus
from services.repositories import (
    CaseLogRepository,
    JobRepository,
    ResultRepository,
    SchemaMismatchError,
    _new_record_id,
    _row_to_job,
    _row_to_module,
    _row_to_record,
    _utcnow_iso,
)


@pytest.fixture
def repo_db(tmp_path):
    """A fresh seeded in-memory SQLite DB for each test."""
    db.reset_engine()
    db.init_db(str(tmp_path / "test.db"))
    JobRepository().seed_modules()
    yield
    db.reset_engine()


# ---------------------------------------------------------------------------
# Module-level helpers + protocols
# ---------------------------------------------------------------------------


class TestHelpers:
    """Tests for the module-level helper functions."""

    def test_new_record_id_is_32_hex(self):
        rid = _new_record_id()
        assert len(rid) == 32
        int(rid, 16)

    def test_new_record_id_unique(self):
        ids = {_new_record_id() for _ in range(100)}
        assert len(ids) == 100

    def test_utcnow_iso_format(self):
        ts = _utcnow_iso()
        assert ts.endswith("Z")
        assert "T" in ts

    def test_row_to_job_maps_all_fields(self):
        row = type("R", (), {})()
        row.id = "j1"
        row.module_id = "text_generate"
        row.params = '{"a": 1}'
        row.form_schema_version = "1.0.0"
        row.status = "running"
        row.label = "L"
        row.progress = 50
        row.progress_text = "half"
        row.error = None
        row.error_detail = None
        row.created_at = "2026-01-01T00:00:00Z"
        row.started_at = None
        row.completed_at = None
        row.params_hash = "ph"
        row.log_text = "log"
        job = _row_to_job(row)
        assert job.id == "j1"
        assert job.params == {"a": 1}
        assert job.status == JobStatus.RUNNING

    def test_row_to_job_empty_params(self):
        row = type("R", (), {})()
        row.id = "j1"
        row.module_id = "m"
        row.params = None
        row.form_schema_version = "1.0.0"
        row.status = "pending"
        row.label = None
        row.progress = 0
        row.progress_text = ""
        row.error = None
        row.error_detail = None
        row.created_at = None
        row.started_at = None
        row.completed_at = None
        job = _row_to_job(row)
        assert job.params == {}

    def test_row_to_module(self):
        row = type("R", (), {})()
        row.id = "m1"
        row.display_name = "M"
        row.runner_class = "Runner"
        row.description = "desc"
        mod = _row_to_module(row)
        assert isinstance(mod, CapabilityModule)
        assert mod.id == "m1"

    def test_row_to_record_decodes_json(self):
        row = type("R", (), {})()
        row.id = "r1"
        row.job_id = "j1"
        row.seq = 0
        row.rank = 1
        row.config = '{"k": "v"}'
        row.summary = '{"s": 1}'
        row.tables = '[{"t": 1}]'
        row.case_hash = "ch"
        row.created_at = "2026-01-01"
        rec = _row_to_record(row)
        assert rec.config == {"k": "v"}
        assert rec.summary == {"s": 1}
        assert rec.tables == [{"t": 1}]
        assert rec.case_hash == "ch"

    def test_row_to_record_empty_json(self):
        row = type("R", (), {})()
        row.id = "r1"
        row.job_id = "j1"
        row.seq = 0
        row.rank = 0
        row.config = None
        row.summary = None
        row.tables = None
        row.case_hash = None
        row.created_at = None
        rec = _row_to_record(row)
        assert rec.config == {}

    def test_schema_mismatch_error_is_runtime_error(self):
        assert issubclass(SchemaMismatchError, RuntimeError)


# ---------------------------------------------------------------------------
# JobRepository
# ---------------------------------------------------------------------------


class TestJobRepositoryModules:
    """Tests for module seeding + lookup."""

    def test_seed_modules_idempotent(self, repo_db):
        assert JobRepository().seed_modules() == 0

    def test_seed_modules_inserts_when_empty(self, tmp_path):
        """seed_modules seeds 3 modules when the table is empty.

        Alembic's 0001 migration already seeds the modules, so to exercise the
        empty-table insert path we init a fresh DB (creates tables + alembic
        seed), wipe the module rows, then call seed_modules — it must re-insert 3.
        """
        db.reset_engine()
        db.init_db(str(tmp_path / "empty.db"))
        repo = JobRepository()
        # Wipe the alembic-seeded rows so the table is genuinely empty.
        from models import orm

        orm_module = orm.ModuleRow
        with db.session_scope() as session:
            for row in session.exec(__import__("sqlmodel").select(orm_module)).all():
                session.delete(row)
            session.commit()
        n = repo.seed_modules()
        db.reset_engine()
        assert n == 3

    def test_get_module_returns_entity(self, repo_db):
        mod = JobRepository().get_module("text_generate")
        assert mod is not None
        assert mod.id == "text_generate"
        assert mod.runner_class == "ModelRunner"

    def test_get_module_none_for_unknown(self, repo_db):
        assert JobRepository().get_module("nope") is None

    def test_list_modules_returns_all(self, repo_db):
        mods = JobRepository().list_modules()
        assert len(mods) == 3
        assert {m.id for m in mods} == {"text_generate", "video_generate", "throughput_optimizer"}


class TestJobRepositoryJobs:
    """Tests for job CRUD + queries."""

    def _job(self, **kw):
        defaults = {"module_id": "text_generate", "params": {"x": 1}, "form_schema_version": "1.0.0"}
        defaults.update(kw)
        return Job(**defaults)

    def test_add_and_get(self, repo_db):
        repo = JobRepository()
        job = self._job()
        added = repo.add(job)
        assert added.id == job.id
        fetched = repo.get(job.id)
        assert fetched is not None
        assert fetched.params == {"x": 1}

    def test_get_returns_none_for_missing(self, repo_db):
        assert JobRepository().get("missing") is None

    def test_add_many(self, repo_db):
        repo = JobRepository()
        jobs = [self._job(), self._job(), self._job()]
        for i, j in enumerate(jobs):
            j.id = f"job-{i}"
        assert len(repo.add_many(jobs)) == 3

    def test_add_many_empty(self, repo_db):
        assert JobRepository().add_many([]) == []

    def test_update_changes_fields(self, repo_db):
        repo = JobRepository()
        job = self._job()
        repo.add(job)
        updated = repo.update(job.id, status=JobStatus.RUNNING, progress=42, progress_text="running")
        assert updated.status == JobStatus.RUNNING
        assert updated.progress == 42

    def test_update_serializes_status_enum(self, repo_db):
        repo = JobRepository()
        job = self._job()
        repo.add(job)
        repo.update(job.id, status=JobStatus.RUNNING)
        assert repo.get(job.id).status == JobStatus.RUNNING

    def test_update_returns_none_for_missing(self, repo_db):
        assert JobRepository().update("missing", status=JobStatus.RUNNING) is None

    def test_update_skips_none_values(self, repo_db):
        """Passing progress=None must NOT overwrite an existing progress value.

        The update() method filters out None-valued kwargs so callers can pass
        a partial patch without clobbering fields they didn't intend to change.
        We first set progress=50, then call update(progress=None) and assert
        the stored progress is still 50 (not reset to None).
        """
        repo = JobRepository()
        job = self._job()
        repo.add(job)
        # First set progress to a non-None value.
        repo.update(job.id, progress=50, progress_text="halfway")
        assert repo.get(job.id).progress == 50
        # Now pass progress=None explicitly — it must be skipped, not applied.
        updated = repo.update(job.id, progress=None, progress_text="txt")
        assert updated is not None
        assert updated.progress == 50  # still 50, NOT None
        assert updated.progress_text == "txt"  # this non-None field DID update
        # Confirm persistence too.
        assert repo.get(job.id).progress == 50

    def test_update_warns_on_illegal_transition(self, repo_db, caplog):
        repo = JobRepository()
        job = self._job(status=JobStatus.SUCCEEDED)
        repo.add(job)
        import logging

        with caplog.at_level(logging.WARNING):
            repo.update(job.id, status=JobStatus.RUNNING)
        assert any("Illegal job transition" in r.message for r in caplog.records)

    def test_update_unknown_status_string(self, repo_db):
        """A bogus status string bypasses the state-machine guard but hits the
        DB CHECK constraint on commit — the exception must propagate out of
        update() (not be silently swallowed).
        """
        repo = JobRepository()
        job = self._job()
        repo.add(job)
        from sqlalchemy.exc import IntegrityError

        with pytest.raises(IntegrityError):
            repo.update(job.id, status="bogus_status")

    def test_list_jobs_pagination_and_filter(self, repo_db):
        repo = JobRepository()
        for i in range(5):
            j = self._job(module_id="text_generate" if i < 3 else "video_generate")
            j.id = f"j{i}"
            repo.add(j)
        assert len(repo.list_jobs(module_id="text_generate")) == 3
        assert len(repo.list_jobs(limit=2, offset=1)) == 2
        assert repo.count_jobs(module_id="video_generate") == 2

    def test_count_jobs_total(self, repo_db):
        repo = JobRepository()
        for i in range(4):
            j = self._job()
            j.id = f"c{i}"
            repo.add(j)
        assert repo.count_jobs() == 4

    def test_list_jobs_and_count_with_status_filter(self, repo_db):
        """The status filter branches (.value) in list_jobs + count_jobs."""
        repo = JobRepository()
        j1 = self._job(status=JobStatus.PENDING)
        j1.id = "p1"
        j2 = self._job(status=JobStatus.RUNNING)
        j2.id = "r1"
        repo.add(j1)
        repo.add(j2)
        assert len(repo.list_jobs(status=JobStatus.RUNNING)) == 1
        assert repo.count_jobs(status=JobStatus.PENDING) == 1

    def test_find_succeeded_by_params_hash(self, repo_db):
        repo = JobRepository()
        j = self._job(status=JobStatus.SUCCEEDED)
        j.id = "succ-1"
        j.completed_at = "2026-01-01"
        repo.add(j)
        # add() doesn't persist params_hash; set it via update (the cache-key field).
        repo.update(j.id, status=JobStatus.SUCCEEDED, params_hash="hash-abc")
        found = repo.find_succeeded_by_params_hash("text_generate", "hash-abc")
        assert found is not None
        assert found.id == "succ-1"
        assert repo.find_succeeded_by_params_hash("text_generate", "nope") is None

    def test_sweep_interrupted(self, repo_db):
        repo = JobRepository()
        j1 = self._job(status=JobStatus.PENDING)
        j1.id = "p1"
        j2 = self._job(status=JobStatus.RUNNING)
        j2.id = "r1"
        j3 = self._job(status=JobStatus.SUCCEEDED)
        j3.id = "s1"
        for j in (j1, j2, j3):
            repo.add(j)
        assert repo.sweep_interrupted() == 2
        assert repo.get("p1").status == JobStatus.INTERRUPTED
        assert repo.get("s1").status == JobStatus.SUCCEEDED


# ---------------------------------------------------------------------------
# ResultRepository
# ---------------------------------------------------------------------------


class TestResultRepository:
    """Tests for result record persistence."""

    def _seed_job(self, repo, job_id="j1", status=JobStatus.SUCCEEDED):
        job = Job(id=job_id, module_id="text_generate", params={}, form_schema_version="1.0.0", status=status)
        repo.add(job)
        return job

    def _record(self, job_id="j1", seq=0, case_hash="ch1"):
        return ResultRecord(
            id=f"r{seq}", job_id=job_id, seq=seq, config={"c": 1}, summary={"s": 1}, tables=[], case_hash=case_hash
        )

    def test_add_and_list_for_job(self, repo_db):
        repo = JobRepository()
        res = ResultRepository()
        self._seed_job(repo)
        res.add(self._record(seq=0))
        res.add(self._record(seq=1, case_hash="ch2"))
        recs = res.list_for_job("j1")
        assert len(recs) == 2
        assert recs[0].seq == 0

    def test_add_many(self, repo_db):
        repo = JobRepository()
        res = ResultRepository()
        self._seed_job(repo)
        added = res.add_many([self._record(seq=0), self._record(seq=1, case_hash="ch2")])
        assert len(added) == 2

    def test_add_many_empty(self, repo_db):
        assert ResultRepository().add_many([]) == []

    def test_clone_records(self, repo_db):
        repo = JobRepository()
        res = ResultRepository()
        self._seed_job(repo, "src")
        self._seed_job(repo, "dst", status=JobStatus.PENDING)
        res.add(self._record(job_id="src", seq=0))
        res.add(self._record(job_id="src", seq=1, case_hash="ch2"))
        assert res.clone_records("src", "dst") == 2
        assert len(res.list_for_job("dst")) == 2

    def test_succeeded_case_hashes_for_module(self, repo_db):
        repo = JobRepository()
        res = ResultRepository()
        self._seed_job(repo, "j1", JobStatus.SUCCEEDED)
        self._seed_job(repo, "j2", JobStatus.FAILED)
        res.add(self._record(job_id="j1", case_hash="good"))
        rec2 = self._record(job_id="j2", seq=1, case_hash="bad")
        res.add(rec2)
        hashes = res.succeeded_case_hashes_for_module("text_generate")
        assert "good" in hashes
        assert "bad" not in hashes

    def test_find_succeeded_record_ids_by_case_hash(self, repo_db):
        repo = JobRepository()
        res = ResultRepository()
        self._seed_job(repo, "j1", JobStatus.SUCCEEDED)
        res.add(self._record(job_id="j1", case_hash="ch"))
        ids = res.find_succeeded_record_ids_by_case_hash("ch")
        assert len(ids) == 1
        assert res.find_succeeded_record_ids_by_case_hash("nope") == []

    def test_get_succeeded_records_by_case_hash(self, repo_db):
        repo = JobRepository()
        res = ResultRepository()
        self._seed_job(repo, "j1", JobStatus.SUCCEEDED)
        res.add(self._record(job_id="j1", case_hash="ch"))
        recs = res.get_succeeded_records_by_case_hash("ch")
        assert len(recs) == 1
        assert recs[0].case_hash == "ch"
        assert res.get_succeeded_records_by_case_hash("nope") == []

    def test_clone_records_by_ids(self, repo_db):
        repo = JobRepository()
        res = ResultRepository()
        self._seed_job(repo, "src")
        self._seed_job(repo, "dst", status=JobStatus.PENDING)
        rec = self._record(job_id="src")
        res.add(rec)
        assert res.clone_records_by_ids([rec.id], "dst") == 1
        assert res.clone_records_by_ids([], "dst") == 0


# ---------------------------------------------------------------------------
# CaseLogRepository
# ---------------------------------------------------------------------------


class TestCaseLogRepository:
    """Tests for per-case CLI log persistence."""

    def test_upsert_insert_then_update(self, repo_db):
        repo = CaseLogRepository()
        assert repo.get("ch") is None
        repo.upsert("ch", "first")
        assert repo.get("ch") == "first"
        repo.upsert("ch", "second")
        assert repo.get("ch") == "second"

    def test_upsert_many(self, repo_db):
        repo = CaseLogRepository()
        assert repo.upsert_many({}) == 0
        assert repo.upsert_many({"a": "1", "b": "2"}) == 2
        assert repo.get("a") == "1"
        assert repo.upsert_many({"a": "updated"}) == 1
        assert repo.get("a") == "updated"

    def test_get_missing(self, repo_db):
        assert CaseLogRepository().get("nope") is None
