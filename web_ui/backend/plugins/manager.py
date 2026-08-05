"""Plugin manager: orchestrates plugin registration, application, and lifecycle.

Called from main.py lifespan() to:
- register (declare extension points as metadata)
- apply (include routers, upsert form schemas, append migrations)
- bootstrap (call startup hooks in topological order)
- destroy (call shutdown hooks in reverse order)

Plugin routers are mounted via include_router (NOT app.mount) so they
participate in the host's lifespan events.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, TYPE_CHECKING

from fastapi import FastAPI

from .contract import MsmdPlugin

if TYPE_CHECKING:  # pragma: no cover - type-checker-only import; TYPE_CHECKING is False at runtime
    from services.schema_registry import SchemaRegistry

logger = logging.getLogger(__name__)


class PluginManager:
    """Orchestrates plugin lifecycle: register -> apply -> bootstrap -> destroy."""

    def __init__(self, plugins: dict[str, MsmdPlugin], app: FastAPI) -> None:
        """Initialize with the already-loaded plugins (sorted topologically).

        Args:
            plugins: Dict of plugin ID to MsmdPlugin instance (already validated and sorted).
            app: The FastAPI application instance.
        """
        self.plugins = plugins
        self.app = app
        self.schema_registry: SchemaRegistry | None = None

    def register(self) -> None:
        """Register phase: declare extension-point contributions as metadata.

        This phase does not perform I/O or modify app state; it's pure metadata
        registration (e.g., declaring that a plugin provides a telemetry sink
        or device profile extensions).
        """
        # Aggregate extension points across ALL plugins. Assigning per-iteration
        # overwrote the whole map each time, so only the last plugin's points
        # survived. Merge by extension-point name (last-wins on conflict) and
        # publish the aggregate to app.state once.
        aggregate: dict[str, Any] = {}
        for pid, plugin in self.plugins.items():
            if plugin.extension_points:
                logger.debug("Plugin %r declares extension points: %s", pid, list(plugin.extension_points.keys()))
                for ep_name, ep_value in plugin.extension_points.items():
                    if ep_name in aggregate:
                        logger.debug("Extension point %r overridden by plugin %r (last-wins)", ep_name, pid)
                    aggregate[ep_name] = ep_value
        self.app.state.plugins_extension_points = aggregate

    def apply(self) -> None:
        """Apply phase: mount routers, register form schemas, append migrations.

        This phase modifies app state:
        - include_router for each plugin with a router (prefix /plugins/{id})
        - SchemaRegistry.upsert_form_schema for each plugin with form configs
        - Append migrations_path to alembic version_locations (done in alembic/env.py)
        """
        from services.schema_registry import SchemaRegistry

        self.schema_registry = SchemaRegistry()

        for pid, plugin in self.plugins.items():
            # Mount router if present. mount_path (declared by the plugin) decides:
            # set => bare-mount at the plugin's baked prefix (legacy/compat URL);
            # None => namespaced under /plugins/{id}.
            if plugin.router is not None:
                if plugin.mount_path is not None:
                    self.app.include_router(plugin.router, tags=[f"plugin-{pid}"])
                    logger.info("Mounted plugin %r router at %s (compat)", pid, plugin.mount_path)
                else:
                    prefix = f"/plugins/{pid}"
                    self.app.include_router(plugin.router, prefix=prefix, tags=[f"plugin-{pid}"])
                    logger.info("Mounted plugin %r router at %s", pid, prefix)

    async def bootstrap(self) -> None:
        """Bootstrap phase: call startup hooks in topological order.

        Plugins are already sorted topologically, so we iterate in order.
        Startup hooks run after the host's initialization (storage seeded, job manager wired)
        but before the app starts serving requests. Hooks may be sync or async;
        async hooks (coroutines) are awaited.
        """
        for pid, plugin in self.plugins.items():
            if plugin.startup is not None:
                logger.info("Calling startup hook for plugin %r", pid)
                try:
                    result = plugin.startup(self.app)
                    if inspect.isawaitable(result):
                        await result
                except Exception:
                    logger.exception("Startup hook for plugin %r failed; continuing", pid)

    async def destroy(self) -> None:
        """Destroy phase: call shutdown hooks in reverse topological order.

        Called before the host's manager.shutdown(wait=True) during lifespan teardown.
        Hooks may be sync or async; async hooks (coroutines) are awaited.
        """
        for pid in reversed(list(self.plugins.keys())):
            plugin = self.plugins[pid]
            if plugin.shutdown is not None:
                logger.info("Calling shutdown hook for plugin %r", pid)
                try:
                    result = plugin.shutdown(self.app)
                    if inspect.isawaitable(result):
                        await result
                except Exception:
                    logger.exception("Shutdown hook for plugin %r failed; continuing", pid)
