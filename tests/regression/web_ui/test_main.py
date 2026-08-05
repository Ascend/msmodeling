"""Real unit tests for main.py module.

Tests FastAPI application factory and lifespan using real imports and fixture-scoped mocks.
Per tests/SKILL.md — real imports + fixture-scoped mocks only.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Import main module
from main import create_app


class TestCreateApp:
    """Tests for create_app function."""

    def test_returns_fastapi_instance(self):
        """create_app returns a FastAPI instance."""
        result = create_app()
        assert isinstance(result, FastAPI)

    def test_has_health_endpoint(self):
        """App has /api/health endpoint."""
        test_app = create_app()
        routes = [route for route in test_app.routes if route.path == "/api/health"]
        assert len(routes) == 1
        assert routes[0].methods == {"GET"}

    def test_includes_modules_router(self):
        """App includes modules router (prefix /api/modules)."""
        test_app = create_app()
        routes = [route.path for route in test_app.routes]
        assert any("/api/modules" in str(route) for route in routes)

    def test_includes_jobs_router(self):
        """App includes jobs router (prefix /api/jobs)."""
        test_app = create_app()
        routes = [route.path for route in test_app.routes]
        assert any("/api/jobs" in str(route) for route in routes)

    def test_includes_cases_router(self):
        """App includes cases router (prefix /api/cases)."""
        test_app = create_app()
        routes = [route.path for route in test_app.routes]
        assert any("/api/cases" in str(route) for route in routes)

    def test_includes_options_router(self):
        """App includes options router (prefix /api/options)."""
        test_app = create_app()
        routes = [route.path for route in test_app.routes]
        assert any("/api/options" in str(route) for route in routes)


class TestInitStorage:
    """Tests for _init_storage function."""

    @patch("db.init_db")
    def test_calls_init_db(self, mock_init_db):
        """_init_storage calls db.init_db()."""
        from main import _init_storage

        _init_storage()

        mock_init_db.assert_called_once()


class TestUpsertSchemaSnapshots:
    """Tests for _upsert_schema_snapshots function."""

    @patch("services.schema_registry.copy_bundled_configs")
    @patch("main.SchemaRegistry")
    def test_copies_bundled_configs(self, mock_registry_class, mock_copy):
        """_upsert_schema_snapshots copies bundled configs and upserts."""
        mock_registry = MagicMock()
        mock_registry.upsert_all_from_bundle.return_value = [("kind", "mod", "ver")]
        mock_registry_class.return_value = mock_registry

        from main import _upsert_schema_snapshots

        result = _upsert_schema_snapshots()

        mock_copy.assert_called_once()
        mock_registry.upsert_all_from_bundle.assert_called_once()
        assert len(result) == 1

    @patch("services.schema_registry.copy_bundled_configs", side_effect=OSError("Copy failed"))
    @patch("main.SchemaRegistry")
    def test_handles_copy_exception_gracefully(self, mock_registry_class, mock_copy):
        """Handles exception when copying bundled configs."""
        mock_registry = MagicMock()
        mock_registry.upsert_all_from_bundle.return_value = []
        mock_registry_class.return_value = mock_registry

        from main import _upsert_schema_snapshots

        # Should not raise despite copy exception
        result = _upsert_schema_snapshots()

        mock_registry.upsert_all_from_bundle.assert_called_once()
        assert result == []


class TestSweepInterruptedJobs:
    """Tests for _sweep_interrupted_jobs function."""

    @patch("main.JobRepository")
    def test_calls_sweep_interrupted(self, mock_repo_class):
        """_sweep_interrupted_jobs calls repository sweep_interrupted."""
        mock_repo = MagicMock()
        mock_repo.sweep_interrupted.return_value = 5
        mock_repo_class.return_value = mock_repo

        from main import _sweep_interrupted_jobs

        result = _sweep_interrupted_jobs()

        mock_repo_class.assert_called_once()
        mock_repo.sweep_interrupted.assert_called_once()
        assert result == 5


class TestHealthEndpoint:
    """Tests for health endpoint."""

    def test_health_returns_ok_status(self):
        """Health endpoint returns ok status."""
        from fastapi.testclient import TestClient

        client = TestClient(create_app())
        response = client.get("/api/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestErrorHandling:
    """Tests for error handling in main.py."""

    @patch("db.init_db", side_effect=Exception("DB error"))
    def test_init_storage_raises_on_db_error(self, mock_init_db):
        """_init_storage propagates database errors."""
        from main import _init_storage

        with pytest.raises(Exception, match="DB error"):
            _init_storage()


class TestRouterRegistration:
    """Tests that each router module is importable."""

    def test_modules_router_importable(self):
        from api.routers.modules import router as r

        assert r is not None

    def test_options_router_importable(self):
        from api.routers.options import router as r

        assert r is not None

    def test_jobs_router_importable(self):
        from api.routers.jobs import router as r

        assert r is not None

    def test_cases_router_importable(self):
        from api.routers.cases import router as r

        assert r is not None


class TestModuleComposition:
    """Tests for module structure and composition."""

    def test_imports_main_module(self):
        """main.py can be imported without errors."""
        import main

        assert main is not None

    def test_exports_create_app(self):
        """main module exports create_app function."""
        from main import create_app

        assert callable(create_app)

    def test_no_module_level_app(self):
        """main module intentionally does NOT export a module-level ``app``.

        This blocks ``uvicorn main:app`` — only ``python main.py`` is supported.
        """
        import main

        assert not hasattr(main, "app"), "main.app must not exist (use create_app())"


class TestLifespan:
    """Tests for the lifespan startup/shutdown (driven via TestClient)."""

    @pytest.fixture(autouse=True)
    def _isolate_default_db_dir(self, tmp_path, monkeypatch):
        """Redirect the default DB dir to tmp_path so the lifespan's alembic
        subprocess (env.py reads ``MSMODELING_UI_DIR`` fresh) never touches the
        real ``.msmodeling_ui`` — each test gets a fresh DB, avoiding
        stale/orphan-stamp failures regardless of the developer's local DB state.
        """
        monkeypatch.setenv("MSMODELING_UI_DIR", str(tmp_path))

    def test_lifespan_startup_wires_state_and_shutdown(self, tmp_path, monkeypatch):
        """Entering the app's lifespan runs init/seed/schema/sweep + wires state;
        exiting calls manager.shutdown.
        """
        import db as db_mod

        db_mod.reset_engine()
        # Use the DEFAULT DB path (under the fixture-isolated MSMODELING_UI_DIR)
        # so the lifespan's _init_storage() → db.init_db() operates on the same
        # DB we set up here.
        db_mod.init_db()
        from services import sim_warmup

        monkeypatch.setattr(sim_warmup, "ensure_sim_stack_warmed", lambda: None)
        # copy_bundled_configs raises → the except path (no frontend build).
        monkeypatch.setattr(
            "services.schema_registry.copy_bundled_configs",
            lambda *a, **kw: (_ for _ in ()).throw(OSError("no bundle")),
        )

        test_app = create_app()
        with TestClient(test_app) as client:
            assert hasattr(test_app.state, "job_manager")
            assert hasattr(test_app.state, "job_repository")
            assert hasattr(test_app.state, "result_repository")
            assert hasattr(test_app.state, "schema_registry")
            assert client.get("/api/health").json() == {"status": "ok"}
        db_mod.reset_engine()

    def test_lifespan_logs_when_modules_seeded(self, tmp_path, monkeypatch, caplog):
        """When seed_modules inserts (fresh DB), the 'Seeded N' info log fires."""
        import logging

        import db as db_mod

        db_mod.reset_engine()
        # Use the DEFAULT DB path (under the fixture-isolated MSMODELING_UI_DIR)
        # so the lifespan's _init_storage() → db.init_db() operates on the same
        # DB we set up here.
        db_mod.init_db()
        # Wipe modules so seed_modules re-inserts (alembic 0001 seeds by default).
        from models import orm
        from sqlmodel import Session, select

        engine = db_mod.get_engine()
        with Session(engine) as s:
            for row in s.exec(select(orm.ModuleRow)).all():
                s.delete(row)
            s.commit()
        from services import sim_warmup

        monkeypatch.setattr(sim_warmup, "ensure_sim_stack_warmed", lambda: None)
        with caplog.at_level(logging.INFO, logger="msmodeling.web"), TestClient(create_app()):
            pass
        db_mod.reset_engine()
        assert any("Seeded" in r.message for r in caplog.records)

    def test_lifespan_logs_registered_and_swept(self, tmp_path, monkeypatch, caplog):
        """A registered schema snapshot + a stale interrupted job trigger both
        'Registered N' and 'Swept N' info logs.
        """
        import json as _json
        import logging

        import db as db_mod

        db_mod.reset_engine()
        # Use the DEFAULT DB path (under the fixture-isolated MSMODELING_UI_DIR).
        db_mod.init_db()
        # Insert a stale pending job (left behind by a crashed server).
        from models.entities import Job
        from models.enums import JobStatus
        from services.repositories import JobRepository

        JobRepository().add(
            Job(module_id="text_generate", params={}, form_schema_version="1.0.0", status=JobStatus.PENDING)
        )
        # Stage a forms bundle so upsert_all_from_bundle registers a snapshot.
        config_dir = tmp_path / "config"
        forms_dir = config_dir / "forms"
        forms_dir.mkdir(parents=True)
        (forms_dir / "text_generate.json").write_text(
            _json.dumps({"moduleId": "text_generate", "version": "1.0.0", "fields": []}),
            encoding="utf-8",
        )
        # Point the SchemaRegistry at this bundle dir (copy_bundled_configs is a no-op).
        monkeypatch.setattr("services.schema_registry._DEFAULT_VAR_CONFIG", config_dir)
        monkeypatch.setattr("services.schema_registry.copy_bundled_configs", lambda *a, **kw: config_dir)
        from services import sim_warmup

        monkeypatch.setattr(sim_warmup, "ensure_sim_stack_warmed", lambda: None)
        with caplog.at_level(logging.INFO, logger="msmodeling.web"), TestClient(create_app()):
            pass
        db_mod.reset_engine()
        assert any("Registered" in r.message for r in caplog.records)
        assert any("Swept" in r.message for r in caplog.records)
