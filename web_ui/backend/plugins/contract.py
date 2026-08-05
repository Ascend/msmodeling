"""Plugin contract for msmodeling.

Defines the MsmdPlugin dataclass/Protocol that plugins must implement,
and the API_VERSION constant that plugins must target.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI

logger = logging.getLogger(__name__)

# The plugin API contract version that this host supports.
# Plugins declare an api_version field; hosts refuse plugins targeting
# incompatible versions (mirrors the existing schema_registry refuse-on-hash-mismatch guardrail).
API_VERSION = "1"


def check_api_version(plugin_api_version: str) -> bool:
    """Check if a plugin's api_version is compatible with this host.

    Args:
        plugin_api_version: The api_version declared by the plugin.

    Returns:
        True if compatible, False otherwise.

    For v1, we require exact match. Future versions may support semver ranges.
    """
    if plugin_api_version == API_VERSION:
        return True
    logger.info(
        "Plugin API version %r does not match host API version %r",
        plugin_api_version,
        API_VERSION,
    )
    return False


@dataclass(frozen=True)
class MsmdPlugin:
    """Plugin contract for msmodeling.

    Plugins are discovered via Python entry_points(group="msmodeling.plugins").
    Each entry point callable must return an instance of this dataclass.

    Fields:
        id: Plugin identifier (^[a-z][a-z0-9_-]*$). Must be unique. Used as the
            allow-list key in MSMODELING_PLUGINS, and as the default URL namespace
            (/plugins/{id}) when mount_path is not set.
        version: Plugin semver (e.g., "1.0.0"). Independent of core version.
        api_version: The host plugin-API contract version this plugin targets
            (e.g., "1"). Host refuses hard on mismatch via check_api_version().
        router: Optional FastAPI router. Mounted per mount_path (see below).
        mount_path: Optional absolute URL path to mount the router at. When set,
            the router is mounted BARE (the router is expected to bake this prefix
            itself, e.g. APIRouter(prefix="/api/feedback")) — used for legacy/compat
            URLs. When None, the router is mounted under /plugins/{id}.
        migrations_path: Optional path to the plugin's alembic/versions directory.
            If provided, appended to alembic's version_locations at runtime.
        startup: Optional async callback called during host lifespan startup.
            Receives the FastAPI app instance. Use for initialization that requires
            the app to be fully constructed (e.g., registering dependency providers).
        shutdown: Optional async callback called during host lifespan shutdown.
            Receives the FastAPI app instance. Use for cleanup (e.g., closing connections).
        depends: Tuple of plugin ids this plugin depends on. Host resolves these
            dependencies topologically before calling startup(). Empty tuple if no deps.
        extension_points: Optional dict of registry/sink contributions.
            For example, a telemetry plugin may declare {"telemetry_sink": TelemetrySinkImpl},
            or a device-upload plugin may declare {"device_profiles": [...]}. Host does not
            interpret this field; it's metadata for host or other plugins to consume.
    """

    id: str
    version: str
    api_version: str
    router: Any | None = None  # FastAPI APIRouter (avoid circular import)
    mount_path: str | None = None
    migrations_path: Path | None = None
    startup: Callable[[FastAPI], Any] | None = None
    shutdown: Callable[[FastAPI], Any] | None = None
    depends: tuple[str, ...] = field(default_factory=tuple)
    extension_points: dict[str, Any] = field(default_factory=dict)
