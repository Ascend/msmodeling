"""Migration correctness tests for the alembic chain — IN-PROCESS.

Scope: the migration FILES themselves (DDL correctness, idempotency, seed
data, downgrade symmetry) — NOT the ``db.py`` call-orchestration layer, which
is covered by ``tests/regression/web_ui/test_db.py``.

In-process, not subprocess: we drive the REAL pip ``alembic`` library via
``alembic.command.upgrade/downgrade`` inside the test process, so coverage
records execution of ``migrations/env.py`` + ``migrations/versions/*.py``.

No shadowing workaround needed: the migrations directory is named ``migrations``
(NOT ``alembic``), so it does not shadow the pip ``alembic`` package —
``import alembic`` resolves to the pip package directly.

Each test uses its own ``MSMODELING_DB_URL`` pointing at a throwaway file.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest
import alembic.command  # pylint: disable=no-name-in-module
import alembic.config  # pylint: disable=no-name-in-module

# Repo root is parents[3] from tests/regression/web_ui/; web_ui/backend lives there.
_BACKEND_DIR = Path(__file__).resolve().parents[3] / "web_ui" / "backend"


@pytest.fixture(autouse=True)
def _isolate_alembic_side_effects():
    """Isolate two global side effects of running alembic in-process:

    1. env.py calls ``logging.config.fileConfig`` on import, which RESETS global
       logging AND (with the default ``disable_existing_loggers``) DISABLES every
       named logger not listed in alembic.ini — e.g. the ``telemetry_sink``
       logger other tests assert against via ``caplog``.

    2. ``_make_config`` sets ``os.environ['MSMODELING_DB_URL']``; db.py inherits
       ``os.environ`` wholesale into its alembic subprocess, so a leftover value
       would break tests that assert init_db passes an EMPTY env.

    Snapshot both before each test and restore after so nothing leaks.
    """
    import logging

    root = logging.getLogger()
    saved_root_level = root.level
    saved_root_handlers = list(root.handlers)
    manager = logging.Logger.manager
    saved_names = dict(manager.loggerDict)
    saved_disabled = {name: logging.getLogger(name).disabled for name in saved_names}
    saved_levels = {name: logging.getLogger(name).level for name in saved_names}
    saved_disable_mask = logging.root.manager.disable  # save global mask
    saved_db_url = os.environ.get("MSMODELING_DB_URL")
    logging.disable(logging.NOTSET)  # clear any global mask for the test
    yield
    # Restore logging state.
    root.handlers = saved_root_handlers
    root.setLevel(saved_root_level)
    for name in saved_names:
        lg = logging.getLogger(name)
        lg.disabled = saved_disabled.get(name, False)
        lg.setLevel(saved_levels.get(name, logging.WARNING))
    logging.disable(saved_disable_mask)  # restore original global mask
    # Restore / clear the DB URL env var.
    if saved_db_url is None:
        os.environ.pop("MSMODELING_DB_URL", None)
    else:
        os.environ["MSMODELING_DB_URL"] = saved_db_url


def _make_config(db_path: Path) -> "alembic.config.Config":
    """Build an alembic Config whose engine points at an isolated tmp DB.

    ``script_location`` is set to an ABSOLUTE path — ``alembic.ini`` declares it
    relative (``migrations``), which only resolves when cwd is the backend dir
    (true for the app's subprocess, not necessarily for pytest).
    """
    os.environ["MSMODELING_DB_URL"] = f"sqlite:///{db_path.as_posix()}"
    cfg = alembic.config.Config(str(_BACKEND_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(_BACKEND_DIR / "migrations"))
    return cfg


def _columns(conn: sqlite3.Connection, table: str) -> dict[str, tuple[str, int]]:
    """Return {column_name: (type, notnull)} for ``table``."""
    return {row[1]: (row[2], row[3]) for row in conn.execute(f"PRAGMA table_info({table})")}


def _index_names(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA index_list({table})")}


def _table_names(conn: sqlite3.Connection) -> set[str]:
    return {
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' AND name != 'alembic_version'"
        )
    }


# Tables the ORM defines — the migration MUST build exactly these.
_EXPECTED_TABLES = {
    "case_logs",
    "feedbacks",
    "form_schemas",
    "jobs",
    "modules",
    "result_records",
    "telemetry_events",
}


class TestMigrationUpgrade:
    """`alembic upgrade head` builds the complete, correct schema from zero."""

    def test_creates_all_tables(self, tmp_path: Path):
        db_file = tmp_path / "schema.db"
        alembic.command.upgrade(_make_config(db_file), "head")
        conn = sqlite3.connect(db_file)
        try:
            assert _table_names(conn) == _EXPECTED_TABLES
        finally:
            conn.close()

    def test_builds_expected_columns_and_types(self, tmp_path: Path):
        """Spot-check key columns: PKs, the cache/dedup columns (jobs.params_hash/
        log_text, result_records.case_hash, telemetry_events.fingerprint), and
        FK-backed columns exist.
        """
        db_file = tmp_path / "cols.db"
        alembic.command.upgrade(_make_config(db_file), "head")
        conn = sqlite3.connect(db_file)
        try:
            jobs = _columns(conn, "jobs")
            assert "params_hash" in jobs  # cache column
            assert "log_text" in jobs
            assert jobs["status"][1] == 1  # NOT NULL

            rr = _columns(conn, "result_records")
            assert "case_hash" in rr  # dedup column

            te = _columns(conn, "telemetry_events")
            assert "fingerprint" in te
            assert te["id"][0] == "INTEGER"  # autoincrement PK

            assert "module_id" in _columns(conn, "form_schemas")
            assert "module_id" in _columns(conn, "jobs")
        finally:
            conn.close()

    def test_builds_expected_indexes(self, tmp_path: Path):
        """The composite/dedup indexes (not derivable from ORM) are created."""
        db_file = tmp_path / "idx.db"
        alembic.command.upgrade(_make_config(db_file), "head")
        conn = sqlite3.connect(db_file)
        try:
            assert "idx_jobs_module_status" in _index_names(conn, "jobs")
            assert "ix_jobs_params_hash" in _index_names(conn, "jobs")
            assert "ix_result_records_case_hash" in _index_names(conn, "result_records")
            assert "ix_telemetry_events_fingerprint" in _index_names(conn, "telemetry_events")
            assert {"ix_feedbacks_job_id", "ix_feedbacks_created_at"} <= _index_names(conn, "feedbacks")
        finally:
            conn.close()

    def test_seeds_three_capability_modules(self, tmp_path: Path):
        db_file = tmp_path / "seed.db"
        alembic.command.upgrade(_make_config(db_file), "head")
        conn = sqlite3.connect(db_file)
        try:
            ids = {r[0] for r in conn.execute("SELECT id FROM modules")}
            assert ids == {"text_generate", "video_generate", "throughput_optimizer"}
        finally:
            conn.close()

    def test_jobs_status_check_constraint_enforced(self, tmp_path: Path):
        """The ck_jobs_status CHECK rejects out-of-range statuses."""
        db_file = tmp_path / "ck.db"
        alembic.command.upgrade(_make_config(db_file), "head")
        conn = sqlite3.connect(db_file)
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            conn.execute("INSERT INTO modules(id,display_name,runner_class) VALUES('m','M','R')")
            conn.execute(
                "INSERT INTO jobs(id,module_id,status,params,form_schema_version) "
                "VALUES('j','m','pending','{}','1.0.0')"
            )
            conn.commit()
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO jobs(id,module_id,status,params,form_schema_version) "
                    "VALUES('j2','m','bogus','{}','1.0.0')"
                )
        finally:
            conn.close()


class TestMigrationIdempotency:
    """`upgrade head` is idempotent — re-running against an already-current DB
    is a no-op (does not error). Critical for safe redeploys.
    """

    def test_second_upgrade_is_noop(self, tmp_path: Path):
        db_file = tmp_path / "idem.db"
        cfg = _make_config(db_file)
        alembic.command.upgrade(cfg, "head")
        alembic.command.upgrade(cfg, "head")  # must not raise

        conn = sqlite3.connect(db_file)
        try:
            ver = conn.execute("SELECT version_num FROM alembic_version").fetchone()[0]
            assert ver == "0001_initial"
            assert _table_names(conn) == _EXPECTED_TABLES
        finally:
            conn.close()


class TestMigrationDowngrade:
    """`downgrade base` cleanly reverses the initial migration (drops everything
    the upgrade created). Guards symmetric migration authoring.
    """

    def test_downgrade_to_base_drops_all_tables(self, tmp_path: Path):
        db_file = tmp_path / "down.db"
        cfg = _make_config(db_file)
        alembic.command.upgrade(cfg, "head")
        alembic.command.downgrade(cfg, "base")

        conn = sqlite3.connect(db_file)
        try:
            assert _table_names(conn) == set()
            remaining = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            assert not (remaining & _EXPECTED_TABLES)
        finally:
            conn.close()

    def test_upgrade_after_downgrade_rebuilds(self, tmp_path: Path):
        """Round-trip: upgrade -> downgrade base -> upgrade yields a working DB."""
        db_file = tmp_path / "roundtrip.db"
        cfg = _make_config(db_file)
        alembic.command.upgrade(cfg, "head")
        alembic.command.downgrade(cfg, "base")
        alembic.command.upgrade(cfg, "head")

        conn = sqlite3.connect(db_file)
        try:
            assert _table_names(conn) == _EXPECTED_TABLES
            ids = {r[0] for r in conn.execute("SELECT id FROM modules")}
            assert ids == {"text_generate", "video_generate", "throughput_optimizer"}
        finally:
            conn.close()


class TestMigrationOffline:
    """``alembic upgrade --sql`` exercises env.py's OFFLINE migration path
    (run_migrations_offline + the top-level ``is_offline_mode`` dispatch), which
    the online tests above do not reach.
    """

    def test_offline_upgrade_emits_sql(self, tmp_path: Path, capsys):
        """Offline mode renders DDL as SQL instead of connecting to a DB."""
        db_file = tmp_path / "offline.db"
        cfg = _make_config(db_file)
        # sql=True forces offline mode -> env.run_migrations_offline().
        alembic.command.upgrade(cfg, "head", sql=True)
        out = capsys.readouterr().out
        # Offline output names the target revision + a CREATE TABLE from 0001.
        assert "0001_initial" in out
        assert "CREATE TABLE" in out

    def test_file_config_failure_is_swallowed(self, tmp_path: Path, monkeypatch):
        """env.py loads the alembic.ini logging config via ``fileConfig``; if that
        raises, the migration must STILL proceed (the failure is swallowed).
        Covers env.py's ``except Exception: pass`` around fileConfig.
        """
        import logging.config

        def _boom(*args, **kwargs):
            raise RuntimeError("intentional fileConfig failure")

        # env.py does ``from logging.config import fileConfig`` at load time, and
        # alembic re-executes env.py on each command, so patching the source
        # attribute makes env.py's import pick up the raising version.
        monkeypatch.setattr(logging.config, "fileConfig", _boom)
        db_file = tmp_path / "fc.db"
        cfg = _make_config(db_file)
        # Must not raise despite fileConfig blowing up.
        alembic.command.upgrade(cfg, "head")
        conn = sqlite3.connect(db_file)
        try:
            assert _table_names(conn) == _EXPECTED_TABLES
        finally:
            conn.close()
