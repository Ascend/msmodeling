"""GELU fusion patterns."""

import math

import torch
import torch._prims as prims

from ... import config

_GELU_DTYPE_LIST = [torch.float16, torch.bfloat16, torch.float32]


def _create_pattern_result(pattern, replacement, example_inputs):
    return pattern, replacement, example_inputs


class DirectGELUPattern:
    """Match direct aten GELU calls."""

    @staticmethod
    def create(dtype, approximate: str, shape=(2, 4)):
        def get_inputs():
            hidden_states = torch.empty(*shape, dtype=dtype, device="meta")
            return [hidden_states]

        def pattern(hidden_states):
            return torch.ops.aten.gelu.default(hidden_states, approximate=approximate)

        def replacement(hidden_states):
            return torch.ops.tensor_cast.gelu(hidden_states, approximate)

        return _create_pattern_result(pattern, replacement, get_inputs())


class TanhGELUDecomposedPattern:
    """Match decomposed tanh-approximate GELU graphs."""

    @staticmethod
    def create(dtype, shape=(2, 4), input_cast: bool = True, output_cast: bool = True):
        sqrt_2_over_pi = math.sqrt(2 / math.pi)

        def get_inputs():
            hidden_states = torch.empty(*shape, dtype=dtype, device="meta")
            return [hidden_states]

        def pattern(hidden_states):
            if input_cast:
                gelu_input = prims.convert_element_type(hidden_states, torch.float32)
            else:
                gelu_input = hidden_states
            x_cubed = torch.ops.aten.pow.Tensor_Scalar(gelu_input, 3)
            inner = torch.ops.aten.add.Tensor(gelu_input, torch.ops.aten.mul.Tensor(x_cubed, 0.044715))
            tanh_arg = torch.ops.aten.mul.Tensor(inner, sqrt_2_over_pi)
            tanh_out = torch.ops.aten.tanh.default(tanh_arg)
            scale = torch.ops.aten.mul.Tensor(gelu_input, 0.5)
            out = torch.ops.aten.mul.Tensor(scale, torch.ops.aten.add.Tensor(tanh_out, 1.0))
            if output_cast:
                return prims.convert_element_type(out, dtype)
            return out

        def replacement(hidden_states):
            return torch.ops.tensor_cast.gelu(hidden_states, "tanh")

        return _create_pattern_result(pattern, replacement, get_inputs())


class ErfGELUDecomposedPattern:
    """Match decomposed exact GELU graphs using erf."""

    @staticmethod
    def create(dtype, shape=(2, 4), input_cast: bool = True, output_cast: bool = True):
        sqrt_2 = math.sqrt(2)

        def get_inputs():
            hidden_states = torch.empty(*shape, dtype=dtype, device="meta")
            return [hidden_states]

        def pattern(hidden_states):
            if input_cast:
                gelu_input = prims.convert_element_type(hidden_states, torch.float32)
            else:
                gelu_input = hidden_states
            erf_arg = torch.ops.aten.div.Tensor(gelu_input, sqrt_2)
            erf_out = torch.ops.aten.erf.default(erf_arg)
            scale = torch.ops.aten.mul.Tensor(gelu_input, 0.5)
            out = torch.ops.aten.mul.Tensor(scale, torch.ops.aten.add.Tensor(erf_out, 1.0))
            if output_cast:
                return prims.convert_element_type(out, dtype)
            return out

        def replacement(hidden_states):
            return torch.ops.tensor_cast.gelu(hidden_states, "none")

        return _create_pattern_result(pattern, replacement, get_inputs())


def register_all_patterns():
    """Register enabled GELU fusion patterns."""
    from . import register_pattern

    if not config.compilation.fusion_patterns.enable_gelu:
        return

    pattern_shapes = [(2, 4), (2, 3, 4)]
    for dtype in _GELU_DTYPE_LIST:
        for shape in pattern_shapes:
            for approximate in ("none", "tanh"):
                pattern, replacement, example_inputs = DirectGELUPattern.create(dtype, approximate, shape)
                register_pattern(
                    f"gelu_direct_{approximate}_{dtype}_rank_{len(shape)}",
                    pattern,
                    replacement,
                    example_inputs,
                    level=0,
                )

            cast_variants = ((False, False),) if dtype is torch.float32 else ((True, True), (False, False))
            for input_cast, output_cast in cast_variants:
                pattern, replacement, example_inputs = TanhGELUDecomposedPattern.create(
                    dtype, shape, input_cast, output_cast
                )
                register_pattern(
                    f"gelu_tanh_decomposed_{dtype}_rank_{len(shape)}_cast_{input_cast}_{output_cast}",
                    pattern,
                    replacement,
                    example_inputs,
                    level=0,
                )
                pattern, replacement, example_inputs = ErfGELUDecomposedPattern.create(
                    dtype, shape, input_cast, output_cast
                )
                register_pattern(
                    f"gelu_erf_decomposed_{dtype}_rank_{len(shape)}_cast_{input_cast}_{output_cast}",
                    pattern,
                    replacement,
                    example_inputs,
                    level=0,
                )
