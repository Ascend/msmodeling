from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class CacheConfig:
    """Block replacement range for DiT cache wrappers."""

    block_start: int = 0
    block_end: int = 10000


@dataclass
class CacheState:
    """Shared runtime state for DiT cache simulation wrappers."""

    reuse: bool = False
    range_hidden: Optional[Any] = None
    range_encoder: Optional[Any] = None
    delta_hidden: Optional[Any] = None
    delta_encoder: Optional[Any] = None
