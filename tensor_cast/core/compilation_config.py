"""Unified entry point for ``--compilation-config`` option.

This module is the single source of truth for the names accepted by the
``--compilation-config`` CLI flag and the runtime config attributes they map to.

Both ``cli/inference/text_generate.py`` and ``serving_cast/parallel_runner.py``
go through :func:`apply_compilation_config` so the two CLI entry points stay in
sync — adding a new compilation feature only requires extending
``COMPILATION_CONFIG_MAP`` and ``COMPILATION_CONFIG_OPTIONS`` here.
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional

from .. import config


logger = logging.getLogger(__name__)


# Single source of truth for the option names exposed via ``--compilation-config``.
# argparse choices, the Gradio CheckboxGroup, and UserInputConfig parsing all
# derive from this list.
COMPILATION_CONFIG_OPTIONS: list[str] = [
    "enable_multistream",
    "enable_sequence_parallel",
    "enable_matmul_allreduce",
    "enable_dispatch_ffn_combine",
]


# Map from option name → dotted path under the root config object.
# Each path is split on "." and walked via getattr before setattr.
_COMPILATION_CONFIG_MAP: dict[str, str] = {
    "enable_multistream": "compilation.multistream.enable",
    "enable_sequence_parallel": "compilation.passes.enable_sequence_parallel",
    "enable_matmul_allreduce": "compilation.fusion_patterns.enable_matmul_allreduce",
    "enable_dispatch_ffn_combine": "compilation.fusion_patterns.enable_dispatch_ffn_combine",
}


def apply_compilation_config(
    compilation_config: Optional[Iterable[str]],
    root: object = config,
) -> None:
    """Apply ``--compilation-config`` options to the given root config object.

    Options not present in ``compilation_config`` are explicitly reset to
    ``False``. This guarantees that running multiple tasks in the same process
    does not leak compilation state from a previous task — a previous bug
    where ``enable_multistream`` / ``enable_matmul_allreduce`` were only
    set to ``True`` but never reset.

    Args:
        compilation_config: Iterable of option names (typically the list from
            ``argparse`` with ``nargs="*"``). ``None`` and empty iterables are
            treated identically: all options are reset to ``False``.
        root: Config object to mutate. Defaults to the global ``config``.
            Tests may pass a stub.
    """
    selected = set(compilation_config) if compilation_config else set()
    unknown = selected - set(_COMPILATION_CONFIG_MAP.keys())
    if unknown:
        raise ValueError(
            f"Unknown --compilation-config option(s): {sorted(unknown)}. Valid options: {COMPILATION_CONFIG_OPTIONS}"
        )
    for opt, path in _COMPILATION_CONFIG_MAP.items():
        segments = path.split(".")
        obj = root
        for seg in segments[:-1]:
            obj = getattr(obj, seg)
        setattr(obj, segments[-1], opt in selected)
    logger.debug("apply_compilation_config: applied=%s", sorted(selected))
