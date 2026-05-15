import logging
from typing import Any, Callable, Dict, Tuple

import torch
import torch._inductor.pattern_matcher as pm
from torch._inductor.pattern_matcher import (
    Match,
    PatternMatcherPass,
    PatternPrettyPrinter,
)

from ..pass_base import TensorCastGraphModulePass

logger = logging.getLogger(__name__)


def _always_true(_match: Match) -> bool:
    return True


class FreezingPatternPass(TensorCastGraphModulePass):
    """A generic graph-pattern pass used only in the after-freezing stage."""

    def __init__(self, pass_name: str = "freezing_pattern_pass"):  # nosec B107
        self.pattern_handlers: Dict[str, Tuple[Any, Callable[..., Any]]] = {}
        self.pattern_pass: PatternMatcherPass = PatternMatcherPass(pass_name=pass_name)

    def __call__(self, gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
        matched_cnt = 0
        while True:
            cnt = self.pattern_pass.apply(gm)
            if cnt == 0:
                break
            matched_cnt += cnt

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("FreezingPatternPass replace %d patterns.", matched_cnt)
            pattern_idx = 0
            logger.debug("Patterns registered for replacement:")
            for pattern_entry in self.pattern_pass.patterns.values():
                for p in pattern_entry:
                    p_str = PatternPrettyPrinter.run(p.pattern)
                    logger.debug("Pattern %d: %s", pattern_idx, p_str)
                    pattern_idx += 1

        return gm

    def register_pattern(
        self,
        name: str,
        pattern: Any,
        handler: Callable[..., Any],
        extra_check: Callable[[Match], bool] | None = None,
    ) -> None:
        if name in self.pattern_handlers:
            raise ValueError(f"Pattern '{name}' is already registered.")

        self.pattern_handlers[name] = (pattern, handler)
        logger.debug("Register freezing pattern: %s", name)

        try:
            pm.register_graph_pattern(
                pattern,
                extra_check=extra_check or _always_true,
                pass_dict=self.pattern_pass,
            )(handler)
            logger.debug("Successfully register freezing pattern: %s", name)
        except RuntimeError as e:
            if "Duplicate pattern" in str(e):
                logger.warning(
                    "Pattern '%s' is already registered. Skipping duplicate registration.",
                    name,
                )
            else:
                raise e

    def has_pattern(self, pattern_name: str) -> bool:
        return pattern_name in self.pattern_handlers
