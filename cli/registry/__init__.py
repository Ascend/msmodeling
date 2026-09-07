"""CLI/WebUI parameter registry.

This package is the single source of truth for CLI-side parameter definitions:
- ``datatypes``: Param, ValidatorRef, ModuleSpec
- ``shared``: cross-module shared Param definitions (MODEL_ID, DEVICE, etc.)
- ``argparse_adapter``: build CLI ArgumentParser from ModuleSpec
- ``validators``: cross-field Python validator functions (L2)
- ``modules``: per-module ModuleSpec definitions (one per CLI command)

UI types (I18nText, UIFieldProps) live in web_ui/backend/services/ui_props/.
"""

from .datatypes import (
    ModuleSpec,
    Param,
    ValidatorRef,
)

__all__ = [
    "ModuleSpec",
    "Param",
    "ValidatorRef",
]
