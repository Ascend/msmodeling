# Copyright (c) 2026-2026 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
