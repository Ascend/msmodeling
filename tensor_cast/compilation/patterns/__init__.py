from functools import lru_cache
from typing import Any, Callable, List

from ..passes.pattern_match_pass import PatternMatchPass
from . import rms_norm, rotary_embedding, swiglu

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
