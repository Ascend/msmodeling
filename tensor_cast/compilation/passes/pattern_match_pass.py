import logging
from typing import Any, Callable, Dict, List, Tuple

import torch
import torch._inductor.pattern_matcher as pm
from torch._inductor.pattern_matcher import PatternMatcherPass, PatternPrettyPrinter

from ..pass_base import TensorCastGraphModulePass

logger = logging.getLogger(__name__)


class PatternMatchPass(TensorCastGraphModulePass):
    def __init__(self):
        self.pattern_replacements: Dict[str, Tuple[Callable[..., Any], Callable[..., Any]]] = {}
        self.pattern_pass: PatternMatcherPass = PatternMatcherPass(
            pass_name="pattern_match_pass"  # nosec B106
        )

    def __call__(self, gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
        matched_cnt = 0
        while True:
            cnt = self.pattern_pass.apply(gm)
            if cnt == 0:
                break
            matched_cnt += cnt
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("PatternMatchPass replace %d patterns.", matched_cnt)
            pattern_idx = 0
            logger.debug("Patterns registered for replacement:")
            for pattern_entry in self.pattern_pass.patterns.values():
                for p in pattern_entry:
                    p_str = PatternPrettyPrinter.run(p.pattern)
                    logger.debug("Pattern %d: %s", pattern_idx, p_str)
                    pattern_idx += 1
        return gm

    def uuid(self) -> Any:
        # TODO: hash all registered patterns
        return super().uuid()

    def register_pattern(
        self,
        name: str,
        pattern: Callable[..., Any],
        replacement: Callable[..., Any],
        example_inputs: List[Any],
        scalar_workaround: Dict[str, Any] | None = None,
    ):
        if name in self.pattern_replacements:
            raise ValueError(f"Pattern '{name}' is already registered.")

        self.pattern_replacements[name] = (pattern, replacement)
        logger.debug("Registering pattern: %s", name)
        try:
            pm.register_replacement(
                pattern,
                replacement,
                example_inputs,
                pm.fwd_only,
                self.pattern_pass.patterns,
                scalar_workaround=scalar_workaround,
            )
            logger.debug("Successfully register pattern: %s", name)
        except RuntimeError as e:
            if "Duplicate pattern" in str(e):
                logger.warning(
                    "Pattern '%s' is already registered. Skipping duplicate registration.",
                    name,
                )
            else:
                raise e

    def has_pattern(self, pattern_name: str) -> bool:
        return pattern_name in self.pattern_replacements
