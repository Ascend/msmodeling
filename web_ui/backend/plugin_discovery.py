"""Plugin migration-path discovery — extracted from ``migrations/env.py``.

Pure, alembic-runtime-independent logic: discover enabled plugins (same
allow-list as ``plugins.loader``) and collect each one's ``migrations_path``.

Lives outside ``env.py`` so it can be unit-tested directly — ``env.py`` is an
alembic script whose top-level ``config = context.config`` only exists while
alembic is running a migration, making it un-importable in isolation.
``env.py`` imports and calls :func:`collect_plugin_migration_paths`.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def collect_plugin_migration_paths() -> list[Path]:
    """Collect ``migrations_path`` from enabled plugins.

    Plugins are discovered via entry_points(group="msmodeling.plugins") and
    filtered by the ``MSMODELING_PLUGINS`` env var (same logic as
    ``plugins.loader``).

    Returns:
        List of plugin migrations/versions directories to append to
        ``version_locations``.
    """
    import os

    paths: list[Path] = []

    # Get allowed plugins from environment (same logic as loader.get_allowed_plugins)
    env_value = os.environ.get("MSMODELING_PLUGINS", "")
    if not env_value:
        return []

    allowed = {part.strip() for part in env_value.split(",") if part.strip()}
    if not allowed:
        return []

    import importlib.metadata

    # entry_points(group=...) is supported on Python >=3.10 (the project minimum).
    entry_points = importlib.metadata.entry_points(group="msmodeling.plugins")

    for ep in entry_points:
        plugin_id = ep.name
        if plugin_id not in allowed:
            continue

        try:
            plugin_factory = ep.load()
            plugin = plugin_factory()
            if hasattr(plugin, "migrations_path") and plugin.migrations_path:
                path = Path(plugin.migrations_path)
                if path.is_dir():
                    paths.append(path)
        except Exception:
            # Silently skip plugins that fail to load during migration discovery
            # (alembic env is run in CLI context, not server context)
            logger.exception("Failed to load plugin %r during migration discovery; skipping", plugin_id)

    return paths
