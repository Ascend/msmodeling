# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Capture and cache primitives for the template-graph engine."""

from importlib import import_module

from .cache_io import (
    build_cache_key,
    cache_path,
    cache_is_valid,
    load_capture,
    save_capture,
)

# Star-import surface: only the eagerly imported cache_io names.  The lazy
# capture/template exports below resolve through the module ``__getattr__``
# (attribute access, e.g. ``template_engine.capture``), which keeps the
# engine's import cost flat without listing unresolved names in ``__all__``.
__all__ = [
    "build_cache_key",
    "cache_is_valid",
    "cache_path",
    "load_capture",
    "save_capture",
]


_CAPTURE_EXPORTS = {
    "CapturedOp",
    "CaptureSerializationError",
    "capture",
    "serialize_op_invoke_infos",
    "serialize_runtime",
}

_MODULE_EXPORTS = {
    "build_template": "template_builder",
    "discover_cache_keys": "template_builder",
    "load_captures_from_cache": "template_builder",
    "parse_cache_key": "template_builder",
    "save_template": "template_builder",
    "BuildReport": "template_builder",
    "TemplateBuildError": "template_builder",
    "derive_value_formula": "formulas",
    "evaluate_formula_safe": "formulas",
    "count_failed_formulas": "formulas",
    "FAILED_FORMULA": "formulas",
    "compare_op_sequences": "compare_ops",
    "CompareReport": "compare_ops",
    "region_marker_counts": "compare_ops",
}


def __getattr__(name: str):
    if name in _CAPTURE_EXPORTS:
        module = import_module(".capture_ops", __name__)
        return getattr(module, name)
    if name in _MODULE_EXPORTS:
        module = import_module(f".{_MODULE_EXPORTS[name]}", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
