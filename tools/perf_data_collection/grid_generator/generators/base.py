from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TheoryShapeRow:
    input_shapes: list[tuple[int, ...]]
    output_shapes: list[tuple[int, ...]]
    extra_values: dict[str, str] = field(default_factory=dict)
