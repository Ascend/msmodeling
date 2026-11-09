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

"""Shared vision-token geometry constants and the per-image spec type."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

# Each image contributes one vision-start token and one vision-end token to
# the language sequence in addition to its projected visual tokens.
VISION_IMAGE_BOUNDARY_TOKEN_COUNT = 2


@dataclass(frozen=True)
class ImageSpec:
    """One image attached to a request.

    Lives in the TensorCast layer so both the input generators (which need the
    geometry) and ServingCast's length distribution (which declares it) can
    agree on one representation without an inverted dependency.
    """

    height: int
    width: int
    image_id: str = ""

    @property
    def label(self) -> str:
        return f"{self.height}x{self.width}"


def specs_from_scalars(count: int | None, height: int | None, width: int | None) -> "tuple[ImageSpec, ...]":
    """Expand the historical scalar image config into per-image specs.

    ``None`` count (or a non-positive one) means "no images"; a partially
    specified geometry is a caller bug, not a text-only request.
    """
    if count == 0:
        return ()
    if count is None and height is None and width is None:
        return ()
    if count is None or height is None or width is None:
        raise ValueError(
            "partial image geometry: set all of count/height/width or none of them "
            f"(count={count}, height={height}, width={width})"
        )
    if count < 0:
        raise ValueError(f"image count must be >= 0, got {count}")
    return tuple(ImageSpec(height=height, width=width) for _ in range(count))


def flatten_image_specs(groups: Sequence[Sequence["ImageSpec"]]) -> "tuple[ImageSpec, ...]":
    """Flatten per-request spec groups into the batch image order."""
    return tuple(spec for group in groups for spec in group)
