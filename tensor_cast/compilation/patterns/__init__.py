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

from functools import lru_cache
from typing import Any, Callable, List

from ..passes.pattern_match_pass import PatternMatchPass
from . import gated_residual, gelu, rms_norm, rotary_embedding, silu, situ, swiglu

# three levels of graph passes, apply them in order
all_passes = [
    PatternMatchPass(),
    PatternMatchPass(),
    PatternMatchPass(),
]


def register_pattern(
    name: str,
    pattern: Callable[..., Any],
    replacement: Callable[..., Any],
    example_inputs: List[Any],
    level=0,
    scalar_workaround: dict[str, Any] | None = None,
):
    if level >= len(all_passes):
        raise ValueError(f"Invalid level {level}, must be less than {len(all_passes)}")
    all_passes[level].register_pattern(
        name,
        pattern,
        replacement,
        example_inputs,
        scalar_workaround=scalar_workaround,
    )


@lru_cache(None)
def lazy_init():
    # register all patterns of a certain dtype below
    rms_norm.register_all_patterns()
    rotary_embedding.register_all_patterns()
    swiglu.register_all_patterns()
    gelu.register_all_patterns()
    silu.register_all_patterns()
    gated_residual.register_all_patterns()
    situ.register_all_patterns()
