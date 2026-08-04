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
"""Exact SpecMatchCriteria resolution without fuzzy fallback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from tools.model_diagnostics.domain.models import ModelRunContext
from tools.model_diagnostics.domain.specification import ModelDiagnosticsSpec, SpecMatchCriteria
from tools.model_diagnostics.specification.errors import AmbiguousModelSpec, UnsupportedModelSpec


def _context_features(context: ModelRunContext) -> set[str]:
    raw = context.model_config.get("features", ())
    if isinstance(raw, str):
        return {raw}
    if isinstance(raw, (list, tuple, set)):
        return {str(item) for item in raw}
    return set()


def matches_context(criteria: SpecMatchCriteria, context: ModelRunContext) -> bool:
    if criteria.entrypoints:
        if context.entrypoint is None or context.entrypoint not in criteria.entrypoints:
            return False
    if criteria.model_types:
        model_type = context.model_config.get("model_type")
        if model_type is None or str(model_type) not in criteria.model_types:
            return False
    if criteria.required_features:
        features = _context_features(context)
        if any(feature not in features for feature in criteria.required_features):
            return False
    return True


@dataclass(frozen=True)
class LoadedSpecCatalogResolver:
    """Resolve against already-loaded Spec documents."""

    specs: Sequence[ModelDiagnosticsSpec]

    def resolve(self, context: ModelRunContext) -> str:
        matched = tuple(spec.spec_id for spec in self.specs if matches_context(spec.matches, context))
        if not matched:
            raise UnsupportedModelSpec("no Spec matched the provided ModelRunContext")
        if len(matched) > 1:
            names = ", ".join(sorted(matched))
            raise AmbiguousModelSpec(f"multiple Specs matched ModelRunContext: {names}")
        return matched[0]
