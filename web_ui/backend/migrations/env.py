"""Alembic env: runs migrations against the SQLModel metadata (lazy imports)."""

from __future__ import annotations

import logging as _logging
import os
from logging.config import fileConfig
from pathlib import Path

from alembic import context  # pylint: disable=no-name-in-module

from plugin_discovery import collect_plugin_migration_paths

config = context.config

# fileConfig swallows errors: a bad or missing logging config must not block
# migrations. Passing None is handled by the Exception catch below (TypeError /
# FileNotFoundError are Exception subclasses), so no ``is not None`` guard is
# needed.
# ``disable_existing_loggers=False`` preserves loggers configured by the host
# application (e.g. ``msmodeling.web`` in main.py). The default ``True`` would
# silence them for the rest of the process after the first alembic run.
# Additionally, fileConfig reconfigures the ROOT logger's handlers from
# alembic.ini's ``[logger_root]`` section, which clobbers pytest's
# ``LogCaptureHandler`` when running in-process. Save + restore the root
# logger's handlers around the call so capsys/caplog continue to work.

_root = _logging.getLogger()
_saved_root_handlers = list(_root.handlers)
try:
    fileConfig(config.config_file_name, disable_existing_loggers=False)
except Exception:
    pass
finally:
    _root.handlers = _saved_root_handlers

_DB_DIR = Path(
    os.environ.get(
        "MSMODELING_UI_DIR",
        Path(__file__).resolve().parents[3] / ".msmodeling_ui",
    )
)
_DB_DIR.mkdir(parents=True, exist_ok=True)
_DEFAULT_URL = f"sqlite:///{(_DB_DIR / 'msmodeling.db').as_posix()}"
config.set_main_option("sqlalchemy.url", os.environ.get("MSMODELING_DB_URL", _DEFAULT_URL))


def _get_metadata():
    # Python caches module imports in sys.modules, so re-importing SQLModel /
    # models.orm is a no-op after the first call. The explicit target_metadata
    # cache added a branch for no measurable gain — just import + return.
    from sqlmodel import SQLModel
    from models import orm as _orm  # noqa: F401

    return SQLModel.metadata


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")

    # Set version_locations to include core + plugin migrations directories
    # Core migrations: web_ui/backend/migrations/versions
    core_versions = Path(__file__).parent / "versions"
    plugin_versions = collect_plugin_migration_paths()
    version_locations = [str(core_versions)] + [str(p) for p in plugin_versions]
    config.set_main_option("version_locations", " ".join(version_locations))

    context.configure(
        url=url,
        target_metadata=_get_metadata(),
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,  # SQLite ALTER support
        version_table="alembic_version",  # Single shared table for core + plugins
        # Note: multi-base branching (version_table_schema) is rejected as overkill for SQLite
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    from sqlalchemy import engine_from_config, pool

    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    # Set version_locations to include core + plugin migrations directories
    core_versions = Path(__file__).parent / "versions"
    plugin_versions = collect_plugin_migration_paths()
    version_locations = [str(core_versions)] + [str(p) for p in plugin_versions]
    config.set_main_option("version_locations", " ".join(version_locations))

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=_get_metadata(),
            render_as_batch=True,
            version_table="alembic_version",  # Single shared table
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
