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
"""Shared region/layer selection normalization for diagnostics requests."""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

from tools.model_diagnostics.errors import InvalidDiagnosticsRequest


def normalize_selected_layers(
    selected_layers: Mapping[str, tuple[int, ...]],
) -> Mapping[str, tuple[int, ...]]:
    """Validate and canonicalize a region-id -> layer-index selection mapping."""

    normalized: dict[str, tuple[int, ...]] = {}
    for region_id, layer_indices in selected_layers.items():
        if not region_id.strip():
            raise InvalidDiagnosticsRequest("selected layer region id must not be empty")
        if not layer_indices:
            raise InvalidDiagnosticsRequest(f"selected layers for region {region_id!r} must not be empty")
        if any(isinstance(index, bool) or not isinstance(index, int) for index in layer_indices):
            raise InvalidDiagnosticsRequest("selected layer indices must be integers")
        if any(index < 0 for index in layer_indices):
            raise InvalidDiagnosticsRequest("selected layer indices must be non-negative")
        normalized[region_id] = tuple(sorted(set(layer_indices)))
    return MappingProxyType(normalized)


def normalize_selected_stage_regions(
    selected_stage_regions: tuple[str, ...],
) -> tuple[str, ...]:
    """Deduplicate (preserving order) and validate a stage-region selection."""

    normalized = tuple(dict.fromkeys(selected_stage_regions))
    if any(not region_id.strip() for region_id in normalized):
        raise InvalidDiagnosticsRequest("selected stage region id must not be empty")
    return normalized
