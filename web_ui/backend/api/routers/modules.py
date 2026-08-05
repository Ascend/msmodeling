"""Modules router.

* ``GET /api/modules`` — list the seeded capability modules.
* ``GET /api/modules/{id}/form-schema?version=`` — form snapshot.

Schema snapshots are read from the SQLite registry table (pinned versions for
reopening historical jobs); ``version`` omitted = the current bundled version.
The schema registry is imported from the app's dependency-injected state
(stashed on ``request.app.state`` by ``main.create_app``).
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request

from api.schemas import ModuleOut

router = APIRouter(prefix="/api/modules", tags=["modules"])


def _schema_registry(request: Request):
    """Resolve the app-wide ``SchemaRegistry`` from injected app state.

    The registry is stashed on ``request.app.state`` by ``main.create_app``;
    raises 500 if it was never wired (misconfigured lifespan).
    """
    registry = getattr(request.app.state, "schema_registry", None)
    if registry is None:
        raise HTTPException(status_code=500, detail="Schema registry not initialized")
    return registry


def _job_repository(request: Request):
    """Resolve the ``JobRepository`` from injected app state (500 if unset)."""
    repo = getattr(request.app.state, "job_repository", None)
    if repo is None:
        raise HTTPException(status_code=500, detail="Job repository not initialized")
    return repo


@router.get("", response_model=list[ModuleOut])
def list_modules(request: Request) -> list[ModuleOut]:
    """List the seeded capability modules."""
    repo = _job_repository(request)
    return [
        ModuleOut(
            id=m.id,
            display_name=m.display_name,
            runner_class=m.runner_class,
            description=m.description,
        )
        for m in repo.list_modules()
    ]


@router.get("/{module_id}/form-schema")
def get_form_schema(
    module_id: str,
    request: Request,
    version: str | None = Query(default=None),
) -> dict:
    """Return the pinned form-schema snapshot for ``module_id`` (404 if absent).

    ``version`` omitted -> the current bundled version; otherwise the
    pinned snapshot used to reopen a historical job.
    """
    registry = _schema_registry(request)
    schema = registry.get_form_schema(module_id, version)
    if schema is None:
        raise HTTPException(status_code=404, detail=f"No form-schema for module {module_id}")
    return schema
