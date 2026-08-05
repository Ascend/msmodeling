"""Plugin discovery and loading via entry_points.

Discovers plugins via importlib.metadata.entry_points(group="msmodeling.plugins"),
filters by the MSMODELING_PLUGINS environment variable (comma-separated whitelist),
and resolves dependencies topologically.
"""

from __future__ import annotations

import logging
import os
from collections import deque
from typing import Any

import importlib.metadata
from fastapi import FastAPI

from .contract import API_VERSION, MsmdPlugin, check_api_version

logger = logging.getLogger(__name__)

# Environment variable that holds the comma-separated whitelist of plugin IDs.
# Only whitelisted plugins are loaded. Empty string or unset means no plugins.
MSMODELING_PLUGINS_ENV = "MSMODELING_PLUGINS"


def get_allowed_plugins() -> set[str]:
    """Parse the MSMODELING_PLUGINS environment variable into a set of plugin IDs.

    Returns:
        Set of allowed plugin IDs. Empty set if unset or empty string.

    Examples:
        MSMODELING_PLUGINS="device-upload,telemetry" -> {"device-upload", "telemetry"}
        MSMODELING_PLUGINS="" -> set()
        MSMODELING_PLUGINS unset -> set()
    """
    env_value = os.environ.get(MSMODELING_PLUGINS_ENV, "")
    if not env_value:
        return set()
    # Split by comma, strip whitespace, filter empty strings
    return {part.strip() for part in env_value.split(",") if part.strip()}


def discover_plugins() -> dict[str, Any]:
    """Discover all plugins via entry_points.

    Returns:
        Dict mapping plugin ID to the entry point callable.
        Plugins with duplicate IDs raise RuntimeError (ambiguous discovery).
    """
    plugins: dict[str, Any] = {}

    # entry_points(group=...) is supported on Python >=3.10 (the project minimum),
    # so no <3.10 fallback is needed here.
    entry_points = importlib.metadata.entry_points(group="msmodeling.plugins")

    for ep in entry_points:
        plugin_id = ep.name
        if plugin_id in plugins:
            raise RuntimeError(f"Duplicate plugin ID {plugin_id!r} from entry points")
        plugins[plugin_id] = ep

    if plugins:
        logger.info("Discovered %d plugin(s) via entry_points: %s", len(plugins), list(plugins.keys()))
    else:
        logger.info("No plugins discovered via entry_points(group='msmodeling.plugins')")

    return plugins


def topological_sort(plugins: dict[str, MsmdPlugin]) -> list[str]:
    """Topologically sort plugins by their 'depends' field.

    Args:
        plugins: Dict of plugin ID to MsmdPlugin instance.

    Returns:
        List of plugin IDs in topological order (dependencies before dependents).

    Raises:
        ValueError if a cycle is detected or a dependency is missing.
    """
    # Build adjacency list and in-degree count
    graph: dict[str, set[str]] = {pid: set() for pid in plugins}
    in_degree: dict[str, int] = {pid: 0 for pid in plugins}

    for pid, plugin in plugins.items():
        for dep in plugin.depends:
            if dep not in plugins:
                raise ValueError(f"Plugin {pid!r} depends on unknown plugin {dep!r}")
            graph[dep].add(pid)
            in_degree[pid] += 1

    # Kahn's algorithm
    queue: deque[str] = deque(pid for pid, degree in in_degree.items() if degree == 0)
    sorted_ids: list[str] = []

    while queue:
        pid = queue.popleft()
        sorted_ids.append(pid)
        for dependent in graph[pid]:
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                queue.append(dependent)

    if len(sorted_ids) != len(plugins):
        # Cycle detected
        remaining = {pid for pid in plugins if pid not in sorted_ids}
        raise ValueError(f"Cyclic dependency detected among plugins: {remaining}")

    return sorted_ids


def load_plugins(app: FastAPI) -> dict[str, MsmdPlugin]:
    """Load plugins: discover, filter, validate, resolve dependencies, and instantiate.

    This is called from main.py create_app() after the core routers are included.

    Args:
        app: The FastAPI application instance.

    Returns:
        Dict of plugin ID to MsmdPlugin instance (only whitelisted plugins).
    """
    allowed = get_allowed_plugins()
    if not allowed:
        logger.info("MSMODELING_PLUGINS is empty/unset; no plugins will be loaded")
        return {}

    discovered = discover_plugins()
    if not discovered:
        logger.info("No plugins discovered; MSMODELING_PLUGINS is ignored")
        return {}

    # Filter by allow-list
    to_load: dict[str, Any] = {}
    for plugin_id, ep in discovered.items():
        if plugin_id in allowed:
            to_load[plugin_id] = ep
        else:
            logger.info("Plugin %r discovered but not in MSMODELING_PLUGINS whitelist; skipping", plugin_id)

    if not to_load:
        logger.info("No whitelisted plugins to load")
        return {}

    # Instantiate plugins
    instances: dict[str, MsmdPlugin] = {}
    for plugin_id, ep in to_load.items():
        try:
            plugin_factory = ep.load()
            plugin = plugin_factory()
            if not isinstance(plugin, MsmdPlugin):
                logger.error(
                    "Plugin %r entry point did not return MsmdPlugin instance; got %r", plugin_id, type(plugin)
                )
                continue
            instances[plugin_id] = plugin
        except Exception:
            logger.exception("Failed to load plugin %r; skipping", plugin_id)
            continue

    # Validate api_version
    valid_plugins: dict[str, MsmdPlugin] = {}
    for pid, plugin in instances.items():
        if check_api_version(plugin.api_version):
            valid_plugins[pid] = plugin
        else:
            logger.warning(
                "Plugin %r api_version=%r incompatible with host %r; skipping", pid, plugin.api_version, API_VERSION
            )

    if not valid_plugins:
        logger.warning("No valid plugins after api_version check")
        return {}

    # Topological sort by dependencies
    try:
        sorted_ids = topological_sort(valid_plugins)
    except ValueError as exc:
        logger.error("Plugin dependency resolution failed: %s", exc)
        return {}

    logger.info("Loading %d plugin(s) in dependency order: %s", len(sorted_ids), sorted_ids)

    # Return in dependency order
    return {pid: valid_plugins[pid] for pid in sorted_ids}
