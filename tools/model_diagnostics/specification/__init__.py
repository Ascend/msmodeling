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

"""Theory-owned specification package exports."""

from tools.model_diagnostics.specification.builtin_activation import (
    create_builtin_operator_activation_registry,
)
from tools.model_diagnostics.specification.errors import (
    AmbiguousModelSpec,
    SourceLoadError,
    SpecificationError,
    SpecificationLoadError,
    UnsupportedModelSpec,
)
from tools.model_diagnostics.specification.expressions import evaluate_dtype, evaluate_shape
from tools.model_diagnostics.specification.loader import (
    LoadedSpecDocument,
    YamlModelDiagnosticsSpecLoader,
)
from tools.model_diagnostics.specification.operator_activation import (
    OperatorActivationPolicy,
    OperatorActivationRegistry,
    OperatorActivationRequest,
)
from tools.model_diagnostics.specification.provider import ResolvingSpecProvider
from tools.model_diagnostics.specification.resolver import (
    LoadedSpecCatalogResolver,
    matches_context,
)
from tools.model_diagnostics.specification.run_profile import (
    DiagnosticsRunProfile,
    DiagnosticsSelectionWarning,
    load_diagnostics_run_profile,
)
from tools.model_diagnostics.specification.source_options import (
    RuntimeSourceOptionsParser,
    SourceOptionsParser,
    TheorySourceOptionsParser,
    create_builtin_source_options_parsers,
)

__all__ = [
    "AmbiguousModelSpec",
    "DiagnosticsRunProfile",
    "DiagnosticsSelectionWarning",
    "LoadedSpecCatalogResolver",
    "LoadedSpecDocument",
    "OperatorActivationPolicy",
    "OperatorActivationRegistry",
    "OperatorActivationRequest",
    "ResolvingSpecProvider",
    "SourceLoadError",
    "SpecificationError",
    "SpecificationLoadError",
    "SourceOptionsParser",
    "RuntimeSourceOptionsParser",
    "TheorySourceOptionsParser",
    "UnsupportedModelSpec",
    "YamlModelDiagnosticsSpecLoader",
    "evaluate_dtype",
    "evaluate_shape",
    "create_builtin_source_options_parsers",
    "create_builtin_operator_activation_registry",
    "load_diagnostics_run_profile",
    "matches_context",
]
