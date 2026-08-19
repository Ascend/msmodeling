"""Shared utilities for the fusion plugin subsystem."""

import json
from pathlib import Path
from typing import Iterator

from tensor_cast import config

_FUSION_ENABLE_PREFIX = "enable_"


def disable_fusion_patterns() -> None:
    """Turn off every built-in fusion pattern switch (all ``enable_*`` flags).

    Used by both the Python API (_disable_default_patterns) and the L3-real
    subprocess child code to ensure the plugin under test is the only active
    fusion source.

    NOTE: only effective before the first ``torch.compile`` in the process;
    see ``_disable_default_patterns()`` in plugin_framework.__init__ for the
    lru_cache guard that detects and warns about the stale-cache case.
    """
    fusion_patterns = config.compilation.fusion_patterns
    for name in dir(fusion_patterns):
        if name.startswith(_FUSION_ENABLE_PREFIX):
            setattr(fusion_patterns, name, False)


def parse_marker_line(stdout: str, marker: str, error_cls, context: str) -> dict:
    """Scan stdout (last line first) for a JSON payload prefixed by *marker*.

    Public helper shared by plugin_framework and l3_real so neither imports
    the other's private symbols.  Raises *error_cls* when the marker is absent.
    """
    for line in reversed(stdout.splitlines()):
        if line.startswith(marker):
            return json.loads(line[len(marker) :])
    raise error_cls(f"subprocess produced no result marker ({context})")


def iter_plugin_files(plugin_dir: str) -> Iterator[Path]:
    """Yield non-private ``*.py`` files in *plugin_dir* (sorted, non-recursive).

    Shared by ``loader.load_plugin_dir`` and ``plugin_framework.evaluate_fusion_plugins``
    to avoid the glob + underscore-filter predicate drifting between callers.
    """
    directory = Path(plugin_dir)
    for py_file in sorted(directory.glob("*.py")):
        if not py_file.name.startswith("_"):
            yield py_file
