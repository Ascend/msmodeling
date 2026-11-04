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

from typing import List

import torch

from ..utils import register_tensor_cast_op


@register_tensor_cast_op("_internal_mark_region_begin")
def _(
    x: torch.Tensor,
    id: int,
) -> torch.Tensor:
    """Mark the beginning of a region of execution.

    Must not return an input alias: under ``torch.compile``, identity aliases can be
    DCE'd so ``mark_region_begin`` disappears while ``mark_region_end`` remains,
    which breaks ``Runtime.repeat_op_invoke_infos`` pairing.
    """
    return x.clone()


@register_tensor_cast_op("_internal_mark_region_end")
def _(
    x: torch.Tensor,
    id: int,
) -> torch.Tensor:
    """Mark the end of a region of execution.

    Non-aliasing return keeps the end marker in compiled graphs (same rationale as
    ``_internal_mark_region_begin``).
    """
    return x.clone()


@register_tensor_cast_op("_internal_copy_region")
def _(
    x: torch.Tensor,
    id: int,
) -> torch.Tensor:
    """Copy a region of execution marked previously."""
    return x


@register_tensor_cast_op("_internal_copy_region_v2")
def _(
    current_input: torch.Tensor,
    output_template: torch.Tensor,
    id: int,
) -> torch.Tensor:
    """Copy a region whose input and output tensor specifications may differ.

    ``current_input`` keeps the replay connected to the current layer and is
    used by Runtime as the copied Region's real input. ``output_template`` is
    the representative Region's marked output after graph passes have run; it
    determines the replay output shape and dtype. Keeping these roles separate
    is required when a pass changes a Region from full-token input to
    sequence-local output (for example all-reduce -> reduce-scatter).
    """
    del current_input, id
    return output_template.clone()


@register_tensor_cast_op("_internal_wait_and_bind")
def _(
    x: torch.Tensor,
    stream_id: int,
    deps: List[torch.Tensor],
) -> torch.Tensor:
    """Bind real ops through the paired record to ``stream_id`` after waiting on ``deps``.

    This is a control-flow anchor used by multistream lowering. It does not modify
    the data carried by ``x``. Instead, the runtime interprets it as:

    1. real ops through the paired ``_internal_record`` execute on ``stream_id``;
    2. the first real op waits until all dependency tokens in ``deps`` are ready.

    Example:
        y = _internal_wait_and_bind(x, 1, [token0])
        z = real_op(y)
        token1 = _internal_record(z, 1)

    Here ``real_op`` runs on stream 1 only after ``token0`` is ready.
    """
    # torch.library custom ops are not allowed to return an input alias.
    return x.clone()


@register_tensor_cast_op("_internal_order_barrier")
def _(
    x: torch.Tensor,
    dep: torch.Tensor,
) -> torch.Tensor:
    """Force ``dep`` to be sequenced before consumers of the result under ``torch.compile``.

    Values of ``x`` are unchanged (clone only to satisfy custom-op non-aliasing rules).
    Used when two subgraphs are logically ordered in Python but share no tensor edge,
    so AOT/Inductor would otherwise sink the earlier work (e.g. draft embedding) after
    a later independent path (e.g. context KV ``reshape_and_cache``).
    """
    del dep  # FX keeps the arg as a real dependency; value is unused.
    return x.clone()


@register_tensor_cast_op("_internal_record")
def _(
    x: torch.Tensor,
    stream_id: int,
    tag: str = "",
) -> torch.Tensor:
    """Create a control token marking completion of the paired stream region.

    This op is paired with ``_internal_wait_and_bind`` during multistream lowering.
    The returned scalar tensor is a runtime control token, not a model activation.

    Example:
        y = real_op(x)
        token = _internal_record(y, 0)
        z = _internal_wait_and_bind(other, 1, [token])

    The wait op can use ``token`` to express that a later op must not start until
    ``real_op`` on stream 0 has completed.

    ``tag`` optionally names the token so a *forward-referenced* consumer can wait on
    it without holding the FX node in its args (see ``_internal_spill``): the shared
    expert lane publishes one tagged token per stage.
    """
    del tag
    return torch.empty((), dtype=torch.int64, device=x.device)


@register_tensor_cast_op("_internal_mark_lane_begin")
def _(
    x: torch.Tensor,
    role: str,
) -> torch.Tensor:
    """Mark the beginning of a multistream lane region (shared-expert branch).

    The paired ``_internal_mark_lane_end`` closes the region. ``SharedExpertLanePass``
    binds every op between the pair to a dedicated stream. ``clone`` keeps the marker
    alive under DCE (custom ops may not return an input alias).
    """
    del role
    return x.clone()


@register_tensor_cast_op("_internal_mark_lane_end")
def _(
    x: torch.Tensor,
    role: str,
) -> torch.Tensor:
    """Mark the end of a multistream lane region.

    Non-aliasing return keeps the end marker in compiled graphs, same rationale as
    ``_internal_mark_lane_begin``.
    """
    del role
    return x.clone()


@register_tensor_cast_op("_internal_spill")
def _(
    x: torch.Tensor,
    role: str,
    wait_tag: str = "",
    window_tag: str = "",
) -> torch.Tensor:
    """Occupy ``stream 0`` until a lane stage finishes (overflow wait).

    Emitted by ``SharedExpertLanePass`` before the routed-side node that competes for
    the same hardware unit as a shared-expert stage. The op itself costs nothing: the
    runtime derives its duration as ``max(0, dep_ready - stream_end[stream 0])``, i.e.
    the measured residual of the stage that could not be hidden. A zero-length spill
    is dropped from the chrome trace and never enters op statistics.

    ``wait_tag`` names the tagged token published by the stage's ``_internal_record``.
    The reference is *forward*: the spill sits at the routed window boundary, while the
    stage finishes later in program order, so the FX args cannot carry the token node.
    The runtime resolves the tag when it evaluates the timeline.

    ``window_tag`` names the token of the routed node that ends the overlap window. The
    runtime reports ``overrun = stage_ready - window_ready`` next to the blocking
    ``delay`` so a stage that missed its window is measurable even when the routed tail
    absorbs part of the overrun.

    The return value is an activation clone consumed by the routed node that competes
    for the same unit, which keeps the node alive through DCE.
    """
    del role, wait_tag, window_tag
    return x.clone()
