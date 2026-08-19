"""Fusion plugin subpackage.

Provides the plugin loader that imports user fusion plugins and lets them
self-register fx patterns / virtual ops / performance properties into the
existing tensor_cast global tables. The main inference flow stays unchanged.

See docs/RFC/rfc_manual_fusion_eval_zh.md (Plugin Mode) for the design.
"""

from typing import List

from . import loader as _loader
from .loader import load_plugin, load_plugin_dir

__all__ = ["load_plugin", "load_plugin_dir", "list_loaded_plugins"]


def list_loaded_plugins() -> List[str]:
    """Return the absolute paths of all plugins loaded in this process."""
    return sorted(_loader._loaded_plugins)
