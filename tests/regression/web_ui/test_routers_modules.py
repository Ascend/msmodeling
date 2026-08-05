"""Unit tests for the modules router (web_ui/backend/api/routers/modules.py)."""

from __future__ import annotations

from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routers.modules import router as modules_router


def _app(*, repo=None, registry=None):
    """Build a minimal app with the modules router + injected state."""
    app = FastAPI()
    app.include_router(modules_router)
    if repo is not None:
        app.state.job_repository = repo
    if registry is not None:
        app.state.schema_registry = registry
    return app


class TestListModules:
    def test_lists_modules(self):
        repo = MagicMock()
        m = MagicMock(id="text_generate", display_name="Text", runner_class="R", description="d")
        repo.list_modules.return_value = [m]
        client = TestClient(_app(repo=repo))
        resp = client.get("/api/modules")
        assert resp.status_code == 200
        body = resp.json()
        assert body[0]["id"] == "text_generate"
        assert body[0]["display_name"] == "Text"

    def test_500_when_repo_uninitialized(self):
        client = TestClient(_app())  # no job_repository on state
        resp = client.get("/api/modules")
        assert resp.status_code == 500


class TestGetFormSchema:
    def test_returns_schema(self):
        registry = MagicMock()
        registry.get_form_schema.return_value = {"fields": []}
        client = TestClient(_app(registry=registry))
        resp = client.get("/api/modules/text_generate/form-schema")
        assert resp.status_code == 200
        assert resp.json() == {"fields": []}
        registry.get_form_schema.assert_called_once_with("text_generate", None)

    def test_passes_version(self):
        registry = MagicMock()
        registry.get_form_schema.return_value = {"fields": []}
        client = TestClient(_app(registry=registry))
        client.get("/api/modules/text_generate/form-schema?version=1.2.3")
        registry.get_form_schema.assert_called_once_with("text_generate", "1.2.3")

    def test_404_when_absent(self):
        registry = MagicMock()
        registry.get_form_schema.return_value = None
        client = TestClient(_app(registry=registry))
        resp = client.get("/api/modules/unknown/form-schema")
        assert resp.status_code == 404

    def test_500_when_registry_uninitialized(self):
        client = TestClient(_app())  # no schema_registry on state
        resp = client.get("/api/modules/text_generate/form-schema")
        assert resp.status_code == 500
