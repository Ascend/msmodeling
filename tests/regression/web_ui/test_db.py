"""Real unit tests for db.py module.

Tests database initialization and session management. Uses real SQLModel imports
via fixture-scoped patches. Per tests/SKILL.md — no conftest mocking.
"""

from __future__ import annotations

from pathlib import Path

import pytest


class TestGetEngine:
    """Tests for get_engine function."""

    def test_returns_engine_object(self, tmp_path):
        """get_engine returns an engine with connect/execute attrs."""
        test_db = tmp_path / "test.db"
        from db import get_engine, reset_engine

        reset_engine()
        engine = get_engine(test_db)
        assert engine is not None
        assert hasattr(engine, "connect")

    def test_caches_engine(self, tmp_path):
        """Subsequent calls return the same engine."""
        from db import get_engine, reset_engine

        reset_engine()
        test_db = tmp_path / "test.db"
        e1 = get_engine(test_db)
        e2 = get_engine(test_db)
        assert e1 is e2

    def test_creates_parent_directory(self, tmp_path):
        """get_engine creates the parent directory if missing."""
        from db import get_engine, reset_engine

        reset_engine()
        nested = tmp_path / "sub" / "test.db"
        get_engine(nested)
        assert nested.parent.exists()


class TestGetSession:
    """Tests for get_session generator."""

    def test_is_generator(self, tmp_path):
        """get_session is a generator function."""
        from db import get_engine, get_session, reset_engine

        reset_engine()
        get_engine(tmp_path / "test.db")
        import inspect

        assert inspect.isgeneratorfunction(get_session)


class TestSessionScope:
    """Tests for session_scope context manager."""

    def test_yields_session_and_commits(self, tmp_path):
        """session_scope commits: a record written inside is readable in a new session."""
        from db import get_engine, reset_engine, session_scope
        from sqlmodel import SQLModel, Session
        from models import orm  # noqa: F401  register table models with SQLModel.metadata

        reset_engine()
        test_db = tmp_path / "test.db"
        get_engine(test_db)
        SQLModel.metadata.create_all(get_engine(test_db))

        with session_scope(test_db) as s:
            s.add(orm.ModuleRow(id="test_mod", display_name="Test", runner_class="TestRunner"))

        # Open a new session and read back — commit must have persisted.
        with Session(get_engine(test_db)) as s:
            row = s.get(orm.ModuleRow, "test_mod")
            assert row is not None
            assert row.display_name == "Test"
        reset_engine()

    def test_rollbacks_on_exception(self, tmp_path):
        """session_scope rolls back uncommitted data when an exception occurs."""
        from db import get_engine, reset_engine, session_scope
        from sqlmodel import SQLModel, Session
        from models import orm  # noqa: F401  register table models with SQLModel.metadata

        reset_engine()
        test_db = tmp_path / "test.db"
        get_engine(test_db)
        SQLModel.metadata.create_all(get_engine(test_db))

        # Write a record but raise before the context exits → rollback.
        with pytest.raises(ValueError):
            with session_scope(test_db) as s:
                s.add(orm.ModuleRow(id="ghost_mod", display_name="Ghost", runner_class="GhostRunner"))
                raise ValueError("boom")

        # Record must NOT exist in a new session.
        with Session(get_engine(test_db)) as s:
            row = s.get(orm.ModuleRow, "ghost_mod")
            assert row is None
        reset_engine()


class TestResetEngine:
    """Tests for reset_engine."""

    def test_sets_engine_to_none(self):
        """reset_engine clears _engine."""
        import db

        db.reset_engine()
        assert db._engine is None

    def test_allows_new_path_after_reset(self, tmp_path):
        """After reset, a different DB path can be used."""
        import db

        db.reset_engine()
        db.get_engine(tmp_path / "a.db")
        assert db._engine is not None
        db.reset_engine()
        db.get_engine(tmp_path / "b.db")


class TestConstants:
    """Tests for module-level constants."""

    def test_default_dir_is_path(self):
        """_DEFAULT_DIR is a Path."""
        from db import _DEFAULT_DIR

        assert isinstance(_DEFAULT_DIR, Path)

    def test_default_db_path(self):
        """DEFAULT_DB_PATH ends with msmodeling.db."""
        from db import DEFAULT_DB_PATH

        assert DEFAULT_DB_PATH.name == "msmodeling.db"


class TestInitDbAndAlembic:
    """Tests for init_db + _run_alembic (in-process alembic Python API)."""

    def test_init_db_runs_alembic_upgrade(self, tmp_path):
        """init_db runs `alembic upgrade head` against a fresh DB."""
        import db

        db.reset_engine()
        db.init_db(str(tmp_path / "init.db"))
        from services.repositories import JobRepository

        assert len(JobRepository().list_modules()) == 3
        db.reset_engine()

    def test_init_db_sets_db_url_env_for_path(self, tmp_path, monkeypatch):
        """init_db communicates the path to alembic via MSMODELING_DB_URL env var.

        _run_alembic sets MSMODELING_DB_URL in os.environ around the alembic
        call so migrations/env.py can read it. We capture the env during the
        call via a patched alembic.command.upgrade.
        """
        import os

        import db

        captured_env: dict[str, str] = {}

        def fake_upgrade(cfg, *args):
            captured_env["MSMODELING_DB_URL"] = os.environ.get("MSMODELING_DB_URL", "")

        monkeypatch.setattr("alembic.command.upgrade", fake_upgrade)
        db.reset_engine()
        db.init_db(str(tmp_path / "with_path.db"))
        db.reset_engine()
        assert captured_env["MSMODELING_DB_URL"].endswith("with_path.db")

    def test_run_alembic_unsupported_command_raises(self):
        """An unsupported alembic command raises RuntimeError."""
        import db

        with pytest.raises(RuntimeError, match="unsupported alembic command"):
            db._run_alembic("frobnicate", "head")

    def test_run_alembic_restores_env_on_success(self, monkeypatch):
        """_run_alembic restores env vars after the call (no leak)."""
        import os

        import db

        sentinel = "db.py-test-not-leaked"
        assert os.environ.get("MSMODELING_DB_URL") is None

        def fake_upgrade(cfg, *args):
            # Inside the call, the env var is set.
            assert os.environ.get("MSMODELING_DB_URL") == "sqlite:///tmp/x.db"

        monkeypatch.setattr("alembic.command.upgrade", fake_upgrade)
        db._run_alembic("upgrade", "head", extra_env={"MSMODELING_DB_URL": "sqlite:///tmp/x.db"})
        # After the call, the env var is restored to absent.
        assert os.environ.get("MSMODELING_DB_URL") is None
        # Unrelated env is unaffected.
        assert sentinel not in os.environ

    def test_run_alembic_restores_env_on_error(self, monkeypatch):
        """_run_alembic restores env vars even when the command raises."""
        import os

        import db

        def failing_upgrade(cfg, *args):
            raise RuntimeError("alembic exploded")

        monkeypatch.setattr("alembic.command.upgrade", failing_upgrade)
        original = os.environ.get("MSMODELING_DB_URL")
        with pytest.raises(RuntimeError, match="alembic exploded"):
            db._run_alembic("upgrade", "head", extra_env={"MSMODELING_DB_URL": "sqlite:///tmp/x.db"})
        # Env restored even after error.
        assert os.environ.get("MSMODELING_DB_URL") == original

    def test_run_alembic_stamp_command(self, monkeypatch):
        """_run_alembic dispatches 'stamp' to alembic.command.stamp."""
        import db

        stamped = []
        monkeypatch.setattr("alembic.command.stamp", lambda cfg, *args: stamped.append(args))
        db._run_alembic("stamp", "head")
        assert stamped == [("head",)]

    def test_run_alembic_other_commands(self, monkeypatch):
        """_run_alembic dispatches downgrade/current/heads/history to alembic.command."""
        import db

        for cmd in ("downgrade", "current", "heads", "history"):
            called = []
            monkeypatch.setattr(f"alembic.command.{cmd}", lambda cfg, *a, _c=called: _c.append(True))
            db._run_alembic(cmd)
            assert called, f"{cmd} should have been called"

    def test_run_alembic_env_restore_when_var_already_set(self, monkeypatch):
        """_run_alembic restores env vars even when they were set before the call."""
        import os

        import db

        # Pre-set the env var; _run_alembic should restore it to the pre-call value.
        monkeypatch.setenv("MSMODELING_DB_URL", "sqlite:///original.db")

        captured_inside = []

        def fake_upgrade(cfg, *args):
            captured_inside.append(os.environ.get("MSMODELING_DB_URL"))

        monkeypatch.setattr("alembic.command.upgrade", fake_upgrade)
        db._run_alembic("upgrade", "head", extra_env={"MSMODELING_DB_URL": "sqlite:///overridden.db"})
        # Inside the call: overridden.
        assert captured_inside == ["sqlite:///overridden.db"]
        # After the call: restored to original.
        assert os.environ.get("MSMODELING_DB_URL") == "sqlite:///original.db"

    def test_init_db_without_path_skips_db_url(self, monkeypatch):
        """init_db with no db_path does NOT set MSMODELING_DB_URL."""
        import os

        import db

        saw_url = []

        def fake_upgrade(cfg, *args):
            saw_url.append(os.environ.get("MSMODELING_DB_URL"))

        monkeypatch.setattr("alembic.command.upgrade", fake_upgrade)
        db.reset_engine()
        db.init_db()  # db_path=None → no MSMODELING_DB_URL set
        db.reset_engine()
        assert saw_url == [None]


class TestGetSessionDependency:
    """Tests for the get_session FastAPI dependency."""

    def test_get_session_yields_session(self, tmp_path):
        """get_session yields a SQLModel Session bound to the engine."""
        import db

        db.reset_engine()
        db.get_engine(tmp_path / "gs.db")
        gen = db.get_session()
        session = next(gen)
        assert hasattr(session, "execute")
        with pytest.raises(StopIteration):
            next(gen)
        db.reset_engine()


class TestAdoptLegacyDb:
    """Tests for _adopt_legacy_db_if_needed (the legacy-DB auto-stamp)."""

    def test_legacy_db_with_tables_but_no_stamp_is_adopted(self, tmp_path):
        """A DB with app tables but no alembic stamp gets stamped head on init."""
        import db
        import sqlalchemy
        from models import orm  # noqa: F401  register tables
        from sqlmodel import SQLModel

        legacy = tmp_path / "legacy.db"
        db.reset_engine()
        eng = db.get_engine(legacy)
        # Build the schema the OLD way (create_all) — tables present, no stamp.
        SQLModel.metadata.create_all(eng)
        tables_before = set(sqlalchemy.inspect(eng).get_table_names())
        assert "modules" in tables_before
        assert "alembic_version" not in tables_before

        db.reset_engine()
        # init_db must NOT raise "table already exists" — it stamps head first.
        db.init_db(legacy)

        db.reset_engine()
        eng = db.get_engine(legacy)
        tables_after = set(sqlalchemy.inspect(eng).get_table_names())
        assert "alembic_version" in tables_after
        db.reset_engine()

    def test_fresh_db_is_not_stamped(self, tmp_path):
        """A fresh DB (no tables) is left for upgrade to create — not stamped."""
        import db
        import sqlalchemy

        fresh = tmp_path / "fresh.db"
        db.reset_engine()
        db.init_db(fresh)  # creates everything via upgrade
        eng = db.get_engine(fresh)
        # upgrade creates alembic_version itself (not via stamp).
        assert "modules" in sqlalchemy.inspect(eng).get_table_names()
        db.reset_engine()

    def test_already_stamped_db_not_re_stamped(self, tmp_path):
        """A DB already stamped at head is not re-stamped (idempotent)."""
        import db

        stamped = tmp_path / "stamped.db"
        db.reset_engine()
        db.init_db(stamped)  # first init stamps + upgrades to head
        db.reset_engine()
        db.init_db(stamped)  # second init: already stamped → no re-stamp, no error
        db.reset_engine()

    def test_empty_alembic_version_table_triggers_stamp(self, tmp_path):
        """A DB whose alembic_version table exists but is empty (stamp lost) is
        re-stamped at head.
        """
        import db
        from models import orm  # noqa: F401
        from sqlalchemy import text
        from sqlmodel import Session, SQLModel

        legacy = tmp_path / "empty_stamp.db"
        db.reset_engine()
        eng = db.get_engine(legacy)
        SQLModel.metadata.create_all(eng)  # app tables, no alembic_version
        # Manually add an EMPTY alembic_version table (stamp row missing).
        with Session(eng) as s:
            s.exec(text("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)"))
            s.commit()
        db.reset_engine()
        db.init_db(legacy)  # empty stamp → adopts → stamps head, no crash
        db.reset_engine()
        eng = db.get_engine(legacy)
        with Session(eng) as s:
            row = s.exec(text("SELECT version_num FROM alembic_version")).first()
        assert row is not None  # now populated
        db.reset_engine()

    def test_db_with_only_non_app_tables_not_stamped(self, tmp_path):
        """A DB with no application tables (e.g. only a stray table) is left for
        upgrade to create the schema — not stamped.
        """
        import db
        from sqlalchemy import inspect as sa_inspect
        from sqlalchemy import text
        from sqlmodel import Session

        other = tmp_path / "other.db"
        db.reset_engine()
        eng = db.get_engine(other)
        with Session(eng) as s:
            s.exec(text("CREATE TABLE random_leftover (x INTEGER)"))
            s.commit()
        db.reset_engine()
        db.init_db(other)  # no app tables → not stamped; upgrade creates schema
        db.reset_engine()
        eng = db.get_engine(other)
        assert "modules" in sa_inspect(eng).get_table_names()  # upgrade ran
        db.reset_engine()

    def test_orphaned_stamp_is_re_stamped(self, tmp_path):
        """A DB stamped at a revision no longer in the chain (orphaned, e.g. from
        a pre-consolidation build) is re-stamped head, so ``alembic upgrade head``
        succeeds instead of failing with "Can't locate revision".
        """
        import db
        from sqlalchemy import text
        from sqlmodel import Session

        orphan = tmp_path / "orphan.db"
        db.reset_engine()
        db.init_db(orphan)  # build a valid DB at head
        db.reset_engine()
        eng = db.get_engine(orphan)
        # Corrupt the stamp to a revision that doesn't exist in the current chain.
        with Session(eng) as s:
            s.exec(text("UPDATE alembic_version SET version_num = '0004_phase_c_cache'"))
            s.commit()
        db.reset_engine()
        # init_db must not raise "Can't locate revision" — it re-stamps head first.
        db.init_db(orphan)
        db.reset_engine()
        eng = db.get_engine(orphan)
        with Session(eng) as s:
            row = s.exec(text("SELECT version_num FROM alembic_version")).first()
        assert row is not None and row[0] == "0001_initial"  # re-stamped to valid head
        db.reset_engine()
