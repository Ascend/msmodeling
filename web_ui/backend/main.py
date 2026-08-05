"""msmodeling Web Console — backend composition root.

FastAPI application factory + router registration + lifespan wiring.

Conventional flat layout (Constitution v2.2.0 Principle I — thin orchestration,
no DDD mandate): ``api/`` (routers/schemas/errors), ``models/`` (orm + entities
+ enums), ``services/`` (orchestration + logic + repositories), ``runners/``
(sim adapters), ``db.py`` + ``migrations/`` for persistence.

Lifespan startup:
  1. init the SQLite engine (WAL/FK/busy_timeout) + create tables;
  2. upsert the bundled form-schema snapshots (REFUSE on hash/version
     mismatch — refuse-on-mismatch guardrail);
  3. interrupted-sweep: pending/running jobs -> ``interrupted`` (server died);
  4. wire the JobManager (inject run_job).

The app imports WITHOUT torch/sqlmodel: every heavy dep is lazy-imported inside
functions/handlers. The only top-level imports here are FastAPI + the router
modules (which themselves only import pydantic).
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import Response

from services.job_manager import JobManager
from services.repositories import JobRepository
from services.repositories import ResultRepository
from services.schema_registry import SchemaRegistry
from api.errors import register_error_handlers
from api.routers import modules_router, options_router
from api.routers import jobs_router
from api.routers import cases_router
from plugins import load_plugins, PluginManager

logger = logging.getLogger("msmodeling.web")


def _init_storage() -> None:
    """Create the SQLite engine + tables (idempotent)."""
    import db

    db.init_db()


def _upsert_schema_snapshots() -> list[tuple[str, str, str]]:
    """Copy the bundled config (if present) and upsert all snapshots.

    Returns the registered (kind, module, version) tuples. Raises on hash
    mismatch for a version (refuse-on-mismatch).
    """
    from services.schema_registry import copy_bundled_configs

    try:
        copy_bundled_configs()
    except OSError:
        # No bundled config yet (e.g. before the first frontend build) — skip.
        logger.warning("Bundled config copy skipped (frontend build not run yet?)")
    registry = SchemaRegistry()
    return registry.upsert_all_from_bundle()


def _sweep_interrupted_jobs() -> int:
    """Startup sweep: mark jobs left pending/running by a crashed server as
    ``interrupted``. Returns the number of rows swept.
    """
    repo = JobRepository()
    return repo.sweep_interrupted()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- startup ---
    _init_storage()
    # Seed the 3 capability modules BEFORE upserting form-schema snapshots
    # (form_schemas.module_id has a FK to modules.id — ordering).
    seeded = JobRepository().seed_modules()
    if seeded:
        logger.info("Seeded %d capability modules", seeded)
    registered = _upsert_schema_snapshots()
    if registered:
        logger.info("Registered %d schema snapshots", len(registered))
    swept = _sweep_interrupted_jobs()
    if swept:
        logger.info("Swept %d interrupted jobs", swept)

    job_repo = JobRepository()
    result_repo = ResultRepository()
    # Concurrent execution: one worker slot per module by default. Each
    # throughput job spawns its own ProcessPoolExecutor, so raise this only on
    # boxes that can take the load. Tune via MSMODELING_MAX_WORKERS.
    max_workers = int(os.environ.get("MSMODELING_MAX_WORKERS", "8"))
    manager = JobManager(job_repo, max_workers=max_workers)
    app.state.job_repository = job_repo
    app.state.result_repository = result_repo
    app.state.job_manager = manager
    app.state.schema_registry = SchemaRegistry()

    # Wire job execution flow
    import services.job_runner

    services.job_runner.build_run_job(manager)

    # Plugin system: apply contributions and bootstrap startup hooks.
    # Plugins are loaded and already validated/api_version-checked.
    # This runs before yield so plugins are ready before the app serves requests.
    if hasattr(app.state, "plugins") and app.state.plugins:
        plugin_manager = PluginManager(app.state.plugins, app)
        app.state.plugin_manager = plugin_manager
        plugin_manager.register()  # declare extension points
        plugin_manager.apply()  # mount routers, register schemas
        await plugin_manager.bootstrap()  # call startup hooks (sync or async)

    yield
    # --- shutdown ---
    # Plugin system: destroy phase (call shutdown hooks in reverse order)
    if hasattr(app.state, "plugin_manager"):
        await app.state.plugin_manager.destroy()

    # Bounded-wait shutdown (#36): wait up to 30s for in-flight workers to
    # finish (so their final state writes reach the WriteQueue), then drain
    # the write queue. A stuck worker past the timeout is left RUNNING; the
    # next boot's startup sweep marks it ``interrupted``. The previous
    # ``wait=False`` call would stop the manager immediately, orphaning writes
    # from workers that were mid-flight at shutdown time.
    manager.shutdown(wait=True, worker_timeout=30.0)


def create_app() -> FastAPI:
    """Application factory."""
    app = FastAPI(
        title="msmodeling Web Console",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(modules_router)
    app.include_router(options_router)
    app.include_router(jobs_router)
    app.include_router(cases_router)
    register_error_handlers(app)

    @app.middleware("http")
    async def add_security_headers(request: Request, call_next):
        response: Response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    # Plugin system: load whitelisted plugins via entry_points.
    # Plugins are mounted per their declared mount_path (or /plugins/{id}) and
    # participate in lifespan.
    try:
        plugins = load_plugins(app)
        if plugins:
            app.state.plugins = plugins
            logger.info("Loaded %d plugin(s): %s", len(plugins), list(plugins.keys()))
    except Exception:
        logger.exception("Plugin loading failed; continuing without plugins")
        app.state.plugins = {}

    @app.get("/api/health", tags=["meta"])
    def health() -> dict:
        return {"status": "ok"}

    return app


def get_bind_address() -> str:
    """Auto-detect localhost address.

    Prefers IPv4 (``127.0.0.1``), falls back to IPv6 (``::1``) if the
    machine has no IPv4 stack.  This mirrors the pattern in the upstream
    ``web_ui.web_ui_start.get_bind_address`` — no ``--host`` CLI parameter
    is exposed, so the server can never be accidentally bound to
    ``0.0.0.0`` or an external IP.

    .. warning:: **Security disclaimer.** Loopback binding only blocks
       *remote* network access. TCP loopback is visible to **all users on
       the same host**, and this service has **no authentication** — any
       local user can call every API, submit jobs, and read results. If
       the server process runs with elevated privileges (root / admin),
       this constitutes a local privilege-escalation risk. **This service
       is designed for single-user, single-machine use only.** Do NOT run
       it on shared / multi-user hosts.

    Returns a bare IP literal (no URL brackets). The caller is responsible
    for adding ``[...]`` when embedding in a URL.
    """
    import socket as _socket

    # IPv4
    try:
        s = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
        s.bind(("127.0.0.1", 0))
        s.close()
        return "127.0.0.1"
    except OSError:
        pass

    # IPv6
    try:
        s = _socket.socket(_socket.AF_INET6, _socket.SOCK_STREAM)
        s.bind(("::1", 0))
        s.close()
        return "::1"
    except OSError:
        return "127.0.0.1"


def main() -> None:
    """CLI entry: bind to localhost and run the app via uvicorn.

    Internal deployment entry — hard-binds to localhost (do not expose --host).
    Only ``python main.py`` is supported. ``uvicorn main:app`` is intentionally
    blocked (no module-level ``app`` object exists) to enforce a single startup
    path with correct working directory and environment.
    """
    import uvicorn

    host = get_bind_address()
    port = int(os.environ.get("MSMODELING_PORT", "8000"))
    uvicorn.run(create_app(), host=host, port=port, log_level="info")


if __name__ == "__main__":  # pragma: no cover (module is imported by tests, never run as a script)
    main()
