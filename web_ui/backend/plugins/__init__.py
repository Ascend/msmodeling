"""msmodeling plugin system.

Provides the plugin contract (MsmdPlugin) and loader for discovering,
validating, and loading plugins via Python entry points.

Plugins are discovered via importlib.metadata.entry_points(group="msmodeling.plugins")
and filtered by the MSMODELING_PLUGINS environment variable (comma-separated whitelist).
Only whitelisted plugins are registered into the FastAPI app.
"""

from __future__ import annotations

import sys

from .contract import MsmdPlugin, API_VERSION
from .loader import discover_plugins, load_plugins
from .manager import PluginManager

# The backend runs from its own directory (cwd = web_ui/backend), so its modules
# are imported as TOP-LEVEL packages (``plugins``, ``plugins.contract``, ...).
# Installed plugins, however, depend on the ``msmodeling`` distribution and import
# the same files via their ABSOLUTE path (``web_ui.backend.plugins.contract``).
# Without bridging the two, Python loads contract.py TWICE → two distinct
# MsmdPlugin classes → the loader's isinstance() check rejects every plugin
# ("entry point did not return MsmdPlugin instance"). Alias the absolute-path
# entries to the already-loaded top-level modules so both import paths resolve
# to identical objects.
for _name, _mod in list(sys.modules.items()):
    if _name == "plugins" or _name.startswith("plugins."):
        _abs = "web_ui.backend." + _name
        sys.modules.setdefault(_abs, _mod)

__all__ = ["MsmdPlugin", "API_VERSION", "discover_plugins", "load_plugins", "PluginManager"]
