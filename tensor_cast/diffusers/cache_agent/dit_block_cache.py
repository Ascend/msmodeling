import types
from typing import List, Tuple

import torch

from .cache import CacheState


class DiTBlockCache(torch.nn.Module):
    """Cache-aware block wrapper used only on the configured cache range."""

    def __init__(
        self,
        block,
        state: CacheState,
        block_index,
        block_start: int,
        block_end: int,
        make_wrapped_forward,
    ):
        super().__init__()
        self._inner = block
        self._state = state
        self._block_index = block_index
        self._block_start = block_start
        self._block_end = block_end
        self.forward = types.MethodType(
            make_wrapped_forward(self)(block.forward),
            self,
        )

    def __getattr__(self, item):
        try:
            return super().__getattr__(item)
        except AttributeError:
            if hasattr(self._inner, item):
                return getattr(self._inner, item)
            raise

    def apply(self, func: callable, *args, **kwargs):
        hidden_states = kwargs.pop("hidden_states", None)
        if hidden_states is None:
            raise ValueError("[DiTBlockCache] Input 'hidden_states' is None.")

        encoder_hidden_states = kwargs.pop("encoder_hidden_states", None)
        if self._state.reuse:
            return self._reuse(hidden_states, encoder_hidden_states)

        if encoder_hidden_states is None:
            res = func(hidden_states, *args, **kwargs)
        else:
            res = func(hidden_states, encoder_hidden_states, *args, **kwargs)
        self._update_cache(res, hidden_states, encoder_hidden_states)
        return res

    def _reuse(self, hidden_states, encoder_hidden_states):
        state = self._state
        if state.delta_hidden is None:
            raise RuntimeError("[DiTBlockCache] Cache delta is empty before reuse.")

        is_range_start = self._block_index == self._block_start
        if state.delta_encoder is not None:
            if encoder_hidden_states is None:
                raise ValueError("[DiTBlockCache] 'encoder_hidden_states' is required for two-output cache reuse.")
            if is_range_start:
                return (
                    hidden_states + state.delta_hidden,
                    encoder_hidden_states + state.delta_encoder,
                )
            return hidden_states, encoder_hidden_states

        return hidden_states + state.delta_hidden if is_range_start else hidden_states

    def _update_cache(self, res, ori_hidden_states, ori_encoder_hidden_states):
        state = self._state
        output_count = len(res) if isinstance(res, (List, Tuple)) else 1
        if output_count not in (1, 2):
            raise RuntimeError(f"[DiTBlockCache] The output count must be 1 or 2, but got {output_count}.")

        is_range_start = self._block_index == self._block_start
        is_range_end = self._block_index == (self._block_end - 1)

        if is_range_start:
            state.range_hidden = ori_hidden_states
            state.range_encoder = ori_encoder_hidden_states

        if not is_range_end:
            return

        if state.range_hidden is None:
            raise RuntimeError("[DiTBlockCache] Missing cache range input for hidden_states.")

        if output_count == 2:
            hidden_states, encoder_hidden_states = res
            if hidden_states is None or encoder_hidden_states is None:
                raise RuntimeError("[DiTBlockCache] Cache function output is None.")
            if state.range_encoder is None:
                raise ValueError("[DiTBlockCache] 'encoder_hidden_states' is required when output count is 2.")
            state.delta_hidden = hidden_states - state.range_hidden
            state.delta_encoder = encoder_hidden_states - state.range_encoder
            return

        if res is None:
            raise RuntimeError("[DiTBlockCache] Cache function output is None.")
        state.delta_hidden = res - state.range_hidden
        state.delta_encoder = None
