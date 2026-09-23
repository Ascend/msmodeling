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
