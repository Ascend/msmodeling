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
