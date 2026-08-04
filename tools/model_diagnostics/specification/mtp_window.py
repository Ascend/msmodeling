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
"""Shared MTP window predicates for activation and Spec materialization."""

from __future__ import annotations

from tools.model_diagnostics.domain.models import ExecutionPhase, ModelRunContext
from tools.model_diagnostics.specification.errors import SpecificationLoadError


def parse_num_mtp_tokens(context: ModelRunContext) -> int:
    """Return ``num_mtp_tokens`` / ``MTP`` as a non-negative integer."""

    config = context.model_config
    mtp_raw = config.get("num_mtp_tokens", config.get("MTP", 0))
    if mtp_raw is None:
        return 0
    if isinstance(mtp_raw, bool) or not isinstance(mtp_raw, int) or mtp_raw < 0:
        raise SpecificationLoadError("model_config.num_mtp_tokens must be a non-negative integer")
    return mtp_raw


def is_mtp_enabled(context: ModelRunContext) -> bool:
    """True for fixed-length MTP decode: phase=decode, MTP>0, Q >= MTP+1."""

    mtp = parse_num_mtp_tokens(context)
    return (
        context.phase is ExecutionPhase.DECODE
        and mtp > 0
        and context.query_length is not None
        and context.query_length >= mtp + 1
    )


def effective_num_mtp_layers(context: ModelRunContext) -> int:
    """Layer count for the MTP region: ``MTP`` when enabled, otherwise 0."""

    return parse_num_mtp_tokens(context) if is_mtp_enabled(context) else 0


def validate_mtp_decode_window(context: ModelRunContext) -> None:
    """Reject illegal MTP decode windows with an explicit Q >= MTP+1 error."""

    mtp = parse_num_mtp_tokens(context)
    if context.phase is not ExecutionPhase.DECODE or mtp <= 0:
        return
    query = context.query_length
    if query is None or query < mtp + 1:
        raise SpecificationLoadError(
            "MTP decode requires query_length >= num_mtp_tokens + 1 "
            f"(Q={query}, MTP={mtp})"
        )
