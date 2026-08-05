"""SQLite engine + session (WAL, FK, busy_timeout).

SQLModel is imported lazily inside ``get_engine``/``init_db`` so that the
FastAPI app can be imported and booted in environments without the heavy
simulation stack — the engine is only created on first DB use (startup lifespan
wiring / first request). ``check_same_thread=False`` because FastAPI serves
across threads and the single ThreadPoolExecutor worker writes too.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)


# Default DB location: <repo>/.msmodeling_ui/msmodeling.db (gitignored).
# Computed dynamically so tests that monkeypatch MSMODELING_UI_DIR after import
# see the updated path (the env.py reload in _run_alembic handles alembic's
# module-level reads; this handles db.py's own reads).
def _default_dir() -> Path:
    return Path(
        os.environ.get(
            "MSMODELING_UI_DIR",
            Path(__file__).resolve().parents[2] / ".msmodeling_ui",
        )
    )


def _default_db_path() -> Path:
    return _default_dir() / "msmodeling.db"


# Back-compat aliases (module-level values are still available for tests that
# import them directly, but get_engine/init_db use the dynamic helpers).
_DEFAULT_DIR = _default_dir()
DEFAULT_DB_PATH = _default_db_path()

_engine = None
_SessionLocal = None


def _apply_pragmas_on_connect(dbapi_conn, connection_record) -> None:
    """Apply PRAGMAs on every new DB-API connection.

    ``check_same_thread=False`` causes SQLAlchemy to use ``SingletonThreadPool``,
    where each worker thread gets its own connection. The old ``_apply_pragmas``
    ran once at engine creation, so worker-thread connections lacked FK / WAL /
    busy_timeout. This event listener ensures EVERY connection (main thread +
    workers) gets the PRAGMAs applied at connect time.
    """
    # dbapi_conn is the raw sqlite3.Connection
    dbapi_conn.execute("PRAGMA journal_mode=WAL")
    dbapi_conn.execute("PRAGMA foreign_keys=ON")
    dbapi_conn.execute("PRAGMA busy_timeout=5000")


def get_engine(db_path: str | os.PathLike[str] | None = None):
    """Create (once) and return the SQLModel engine for ``db_path``."""
    global _engine, _SessionLocal
    if _engine is not None:
        return _engine

    from sqlmodel import Session, SQLModel, create_engine  # noqa: F401
    from sqlalchemy import event

    path = Path(db_path) if db_path is not None else _default_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    url = f"sqlite:///{path.as_posix()}"
    _engine = create_engine(
        url,
        echo=False,
        connect_args={"check_same_thread": False},
    )
    # Register the connect event listener so PRAGMAs are applied on EVERY new
    # connection (including worker threads). This replaces the old one-time
    # ``_apply_pragmas(engine)`` call which didn't cover worker-thread connections.
    event.listen(_engine, "connect", _apply_pragmas_on_connect)
    _SessionLocal = Session
    return _engine


def init_db(db_path: str | os.PathLike[str] | None = None) -> None:
    """Apply the alembic migration chain — single source of truth for schema.

    ``alembic upgrade head`` handles both new databases (creates all tables
    from 0001→head) and existing ones (applies only missing migrations).
    SQLModel metadata is imported to register table models (needed by
    alembic's target_metadata), but ``create_all`` is NOT called — alembic
    owns schema creation.

    When ``db_path`` is given, it is communicated to alembic via
    ``MSMODELING_DB_URL`` (which ``migrations/env.py`` reads). When not given,
    alembic uses its own default (same ``.msmodeling_ui/msmodeling.db`` as
    ``DEFAULT_DB_PATH``).

    Legacy DBs created before alembic became the schema SSOT (e.g. via
    ``SQLModel.create_all`` in an older release) have the tables but no
    alembic version stamp — ``upgrade head`` would re-run 0001 and fail with
    "table already exists". ``_adopt_legacy_db_if_needed`` detects that state
    and stamps head first so upgrade is a no-op for them.
    """
    from models import orm as _orm  # noqa: F401

    engine = get_engine(db_path)  # creates engine + pragmas, caches it
    env = {}
    if db_path is not None:
        path = Path(db_path)
        env["MSMODELING_DB_URL"] = f"sqlite:///{path.as_posix()}"
    _adopt_legacy_db_if_needed(engine, env)
    _run_alembic("upgrade", "head", extra_env=env)


def _alembic_script():
    """The alembic ``ScriptDirectory`` for the current migration chain.

    Built in-process with an absolute ``script_location`` so it resolves
    regardless of the caller's cwd (alembic.ini's ``migrations`` is relative to
    the backend dir).
    """
    import alembic.config as _alembic_config  # pylint: disable=no-name-in-module
    import alembic.script as _alembic_script  # pylint: disable=no-name-in-module

    _backend_dir = Path(__file__).resolve().parent
    cfg = _alembic_config.Config(str(_backend_dir / "alembic.ini"))
    cfg.set_main_option("script_location", str(_backend_dir / "migrations"))
    return _alembic_script.ScriptDirectory.from_config(cfg)


def _adopt_legacy_db_if_needed(engine, env: dict[str, str]) -> None:
    """Adopt a DB alembic can't upgrade directly.

    Two such states both make ``alembic upgrade head`` fail and are fixed by
    running ``alembic stamp head`` first (adopting the existing schema so the
    subsequent upgrade is a no-op):

    * **Untracked** — built before alembic was the schema SSOT (e.g. via
      ``SQLModel.create_all``): tables present, no ``alembic_version`` row →
      upgrade re-runs the initial migration and crashes on "table already
      exists".
    * **Orphaned stamp** — stamped at a revision that no longer exists in the
      chain (e.g. a revision dropped during migration consolidation): upgrade
      fails with "Can't locate revision".

    Logged loudly: stamping assumes the existing schema matches head. If it is
    actually stale (missing newer columns), stamp head would hide the gap —
    the warning surfaces it so an operator can stamp a base + upgrade instead.
    """
    from sqlalchemy import inspect as sa_inspect
    from sqlmodel import Session, text

    table_names = set(sa_inspect(engine).get_table_names())
    if not table_names:
        return  # fresh DB — let upgrade create everything

    stamped: str | None = None
    if "alembic_version" in table_names:
        with Session(engine) as session:
            row = session.exec(text("SELECT version_num FROM alembic_version")).first()
        if row:
            stamped = row[0]

    if stamped is None:
        # Tables present but alembic has no record → legacy DB. Stamp at head.
        _APP_TABLES = {"modules", "jobs", "result_records", "form_schemas"}
        if table_names & _APP_TABLES:
            logger.warning(
                "Adopting legacy DB (application tables present but no alembic stamp) — "
                "running `alembic stamp head`. If the schema is stale, stamp the matching "
                "base revision and run `alembic upgrade head` manually."
            )
            _run_alembic("stamp", "head", extra_env=env)
        return

    # Tracked at a revision the current chain still knows? Normal path — let
    # upgrade advance it.
    script = _alembic_script()
    if stamped in {rev.revision for rev in script.walk_revisions()}:
        return

    # Orphaned stamp (revision removed from the chain, e.g. a pre-consolidation
    # build): `alembic upgrade head` would fail with "Can't locate revision", and
    # `alembic stamp head` ALSO refuses (it tries to resolve the unresolvable
    # current revision first). So re-stamp the version table directly to head.
    head = script.get_current_head()
    logger.warning(
        "DB stamped at %r, which is not in the current migration chain (orphaned "
        "revision, likely from a pre-consolidation build) — re-stamping the version "
        "table to head %r directly. If the schema is actually stale, stamp the matching "
        "base revision and run `alembic upgrade head` manually.",
        stamped,
        head,
    )
    with Session(engine) as session:
        session.execute(text("UPDATE alembic_version SET version_num = :head"), {"head": head})
        session.commit()


def _run_alembic(command: str, *args: str, extra_env: dict[str, str] | None = None) -> None:
    """Run ``alembic <command> <args>`` via alembic's Python API.

    Uses the in-process ``alembic.command`` API (no subprocess) so this works
    on environments that block subprocess execution (e.g. Windows Application
    Control policies that reject the ``alembic.exe`` console script).
    ``extra_env`` entries are applied to ``os.environ`` for the duration of the
    call (alembic's ``env.py`` reads ``MSMODELING_DB_URL`` from the process env).

    Note: ``migrations/env.py`` is loaded by alembic via ``runpy.run_path``
    (script semantics), not as an imported module, so its module-level env-var
    reads (``MSMODELING_UI_DIR`` / ``MSMODELING_DB_URL``) see the current env
    on every call — no manual reload needed.
    """
    import os as _os
    from alembic import command as _alembic_command  # pylint: disable=no-name-in-module

    _BACKEND_DIR = Path(__file__).resolve().parent
    cfg_file = _BACKEND_DIR / "alembic.ini"
    import alembic.config as _alembic_config  # pylint: disable=no-name-in-module

    cfg = _alembic_config.Config(str(cfg_file))
    cfg.set_main_option("script_location", str(_BACKEND_DIR / "migrations"))

    # Apply extra_env entries for the duration of the call so migrations/env.py
    # can read them (e.g. MSMODELING_DB_URL). Restored on exit.
    saved: dict[str, str | None] = {}
    if extra_env:
        for key, value in extra_env.items():
            saved[key] = _os.environ.get(key)
            _os.environ[key] = value
    try:
        if command == "upgrade":
            _alembic_command.upgrade(cfg, *args)
        elif command == "stamp":
            _alembic_command.stamp(cfg, *args)
        elif command == "downgrade":
            _alembic_command.downgrade(cfg, *args)
        elif command == "current":
            _alembic_command.current(cfg)
        elif command == "heads":
            _alembic_command.heads(cfg)
        elif command == "history":
            _alembic_command.history(cfg)
        else:
            raise RuntimeError(f"unsupported alembic command: {command}")
    finally:
        # Restore env
        for key, original in saved.items():
            if original is None:
                _os.environ.pop(key, None)
            else:
                _os.environ[key] = original


def get_session() -> "Iterator":  # type: ignore[type-arg]
    """FastAPI dependency yielding a SQLModel ``Session``."""
    from sqlmodel import Session

    engine = get_engine()
    with Session(engine) as session:
        yield session


@contextmanager
def session_scope(db_path: str | os.PathLike[str] | None = None) -> Iterator:
    """Context-manager session for use outside the request cycle (startup)."""
    from sqlmodel import Session

    engine = get_engine(db_path)
    with Session(engine) as session:
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise


def reset_engine() -> None:
    """Test helper: drop the cached engine/session so a new path can be used."""
    global _engine, _SessionLocal
    _engine = None
    _SessionLocal = None
