import torch

from ... import config

_SILU_DTYPE_LIST = [torch.float16, torch.bfloat16, torch.float32]


class DirectSiLUPattern:
    @staticmethod
    def create(dtype):
        def get_inputs():
            hidden_states = torch.empty(2, 3, 4, dtype=dtype, device="meta")
            return [hidden_states]

        def pattern(hidden_states):
            return torch.ops.aten.silu.default(hidden_states)

        def replacement(hidden_states):
            return torch.ops.tensor_cast.silu(hidden_states)

        return pattern, replacement, get_inputs()


class DecomposedSiLUPattern:
    @staticmethod
    def create(dtype):
        def get_inputs():
            hidden_states = torch.empty(2, 3, 4, dtype=dtype, device="meta")
            return [hidden_states]

        def pattern(hidden_states):
            return torch.ops.aten.mul.Tensor(hidden_states, torch.ops.aten.sigmoid.default(hidden_states))

        def replacement(hidden_states):
            return torch.ops.tensor_cast.silu(hidden_states)

        return pattern, replacement, get_inputs()


def register_all_patterns():
    from . import register_pattern

    if not config.compilation.fusion_patterns.enable_silu:
        return

    for dtype in _SILU_DTYPE_LIST:
        pattern, replacement, example_inputs = DirectSiLUPattern.create(dtype)
        register_pattern(
            f"silu_direct_{dtype}",
            pattern,
            replacement,
            example_inputs,
            level=0,
        )
        pattern, replacement, example_inputs = DecomposedSiLUPattern.create(dtype)
        register_pattern(
            f"silu_decomposed_{dtype}",
            pattern,
            replacement,
            example_inputs,
            level=0,
        )
