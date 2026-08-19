"""Plugin loader: dynamically import a fusion plugin .py and trigger its
self-registration via ``register_all_patterns()``.

Design constraints (see RFC §4.2 / §6):
- The loader does NOT modify the main flow; it only invokes the plugin's own
  ``register_all_patterns()``, which calls the existing
  ``register_tensor_cast_op`` / ``register_pattern`` /
  ``register_op_properties`` APIs.
- A faulty plugin (syntax error / missing dependency / missing entry) is
  warned about and skipped, never crashing the host process.
- Idempotency guard: the same plugin file is imported at most once per
  process. Re-loading would re-run ``register_tensor_cast_op`` (torch
  ``custom_op`` raises on duplicate definition) and
  ``register_op_properties`` (raises ``already registered`` by default),
  so dedup is required for pytest multi-case / Python API loops.
"""

import importlib.util
import logging
import uuid
from pathlib import Path
from typing import Optional, Set

from tensor_cast.plugins._utils import iter_plugin_files

logger = logging.getLogger(__name__)

# Absolute paths of plugins already imported in this process.
_loaded_plugins: Set[str] = set()

REGISTER_ENTRY = "register_all_patterns"


def _already_loaded(abs_path: str) -> bool:
    return abs_path in _loaded_plugins


def load_plugin(plugin_path: Optional[str]) -> bool:
    """Load a single plugin file and call its ``register_all_patterns()``.

    Returns True if the plugin was loaded (and registered) on this call,
    False if it was skipped (``plugin_path is None`` no-plugin baseline /
    already loaded) or failed (warned, not raised).

    ``plugin_path=None`` is the explicit no-plugin baseline (RFC §4.2 / §5.3):
    the same ``evaluate_fusion_plugin()`` entry drives both the baseline and
    the fused run, so a ``None`` path must early-return rather than crash on
    ``Path(None)``.

    ⚠️  Trust model: the plugin file is executed in the current Python process
    with no sandboxing.  The caller is responsible for ensuring the path comes
    from a trusted source (validated user config, pre-reviewed file).  Do NOT
    pass paths derived from untrusted input without prior review.
    """
    if plugin_path is None:
        logger.debug("No plugin path given (no-plugin baseline), skipping")
        return False

    abs_path = str(Path(plugin_path).resolve())

    # Idempotency guard: import each file at most once per process.
    if _already_loaded(abs_path):
        logger.debug("Plugin already loaded, skipping: %s", abs_path)
        return False

    if not Path(abs_path).is_file():
        logger.warning("Plugin path is not a file, skipping: %s", abs_path)
        return False

    # Use a unique module name so two plugins never collide in sys.modules.
    module_name = f"tensor_cast_plugin_{uuid.uuid4().hex}"
    try:
        spec = importlib.util.spec_from_file_location(module_name, abs_path)
        if spec is None or spec.loader is None:
            logger.warning("Cannot create import spec for plugin: %s", abs_path)
            return False
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as exc:
        logger.warning("Plugin import failed, skipping: %s (%s)", abs_path, exc)
        return False

    entry = getattr(module, REGISTER_ENTRY, None)
    if not callable(entry):
        logger.warning("Plugin %s missing callable %s(), skipping", abs_path, REGISTER_ENTRY)
        return False

    try:
        entry()
    except Exception as exc:
        logger.warning("Plugin %s %s() failed, skipping: %s", abs_path, REGISTER_ENTRY, exc)
        return False

    # Mark loaded only after successful registration.
    _loaded_plugins.add(abs_path)
    logger.info("Loaded fusion plugin: %s", abs_path)
    return True


def load_plugin_dir(plugin_dir: str) -> int:
    """Scan a directory (non-recursive) and load every ``*.py`` file.

    Returns the number of plugins newly loaded on this call.
    """
    directory = Path(plugin_dir)
    if not directory.is_dir():
        logger.warning("Plugin dir not found, skipping: %s", plugin_dir)
        return 0

    loaded = 0
    for py_file in iter_plugin_files(plugin_dir):
        if load_plugin(str(py_file)):
            loaded += 1
    return loaded
