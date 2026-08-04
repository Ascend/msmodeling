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
"""Compose Spec resolution and context-aware materialization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

from tools.model_diagnostics.domain.models import ModelRunContext
from tools.model_diagnostics.domain.specification import ModelDiagnosticsSpec
from tools.model_diagnostics.specification.errors import SpecificationLoadError
from tools.model_diagnostics.specification.loader import LoadedSpecDocument, YamlModelDiagnosticsSpecLoader


class _Resolver(Protocol):
    def resolve(self, context: ModelRunContext) -> str: ...


@dataclass(frozen=True)
class ResolvingSpecProvider:
    """resolve(spec_id) then materialize its preloaded Spec document with Context.

    Unlike a naive provider, ``get`` never re-reads YAML: every Spec document the
    catalog can resolve to is loaded once at composition time and held in
    ``documents``; only ``materialize`` (pure, Context-driven) runs per request.
    """

    resolver: _Resolver
    loader: YamlModelDiagnosticsSpecLoader
    documents: Mapping[str, LoadedSpecDocument]

    def get(self, context: ModelRunContext) -> ModelDiagnosticsSpec:
        spec_id = self.resolver.resolve(context)
        try:
            loaded = self.documents[spec_id]
        except KeyError as error:
            raise SpecificationLoadError(f"resolver returned spec_id {spec_id!r} with no preloaded document") from error
        return self.loader.materialize(loaded, context)
