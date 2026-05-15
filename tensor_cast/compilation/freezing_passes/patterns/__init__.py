from functools import lru_cache
from typing import Any, Callable, List

from ..freezing_pattern_pass import FreezingPatternPass
from . import matmul_allreduce

all_passes: List[FreezingPatternPass] = [
    FreezingPatternPass(pass_name="freezing_pattern_pass_0"),  # nosec B106
]


def register_pattern(
    name: str,
    pattern: Any,
    handler: Callable[..., Any],
    level: int = 0,
    extra_check: Callable[[Any], bool] | None = None,
) -> None:
    if level >= len(all_passes):
        raise ValueError(f"Invalid level {level}, must be less than {len(all_passes)}")
    all_passes[level].register_pattern(
        name=name,
        pattern=pattern,
        handler=handler,
        extra_check=extra_check,
    )


@lru_cache(None)
def lazy_init() -> None:
    matmul_allreduce.register_all_patterns()
