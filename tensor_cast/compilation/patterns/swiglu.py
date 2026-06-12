import logging

import torch
import torch._prims as prims

from ... import config

_SWIGLU_DTYPE_LIST = [torch.float16, torch.bfloat16]

logger = logging.getLogger(__name__)


class SwiGLUPattern:
    @staticmethod
    def create(dtype):
        def get_inputs():
            minimal_shape = (1, 1)
            gate = torch.empty(*minimal_shape, dtype=dtype, device="meta")
            up = torch.empty(*minimal_shape, dtype=dtype, device="meta")
            return [gate, up]

        def _build_core(gate, up):
            """
            Internal helper to build the common SwiGLU computation graph.
            Returns the processed silu_gate and the original up tensor.
            """
            gate_fp32 = prims.convert_element_type(gate, torch.float32)
            sigmoid_gate = torch.ops.aten.sigmoid.default(gate_fp32)
            silu_gate_fp32 = torch.ops.aten.mul.Tensor(gate_fp32, sigmoid_gate)
            silu_gate = prims.convert_element_type(silu_gate_fp32, dtype)
            return silu_gate, up

        def _make_pattern(reverse: bool):
            def pattern(gate, up):
                """
                Pattern function for SwiGLU activation fusion (exclude matmul)
                Matches only the activation computation segment:
                gate → fp32 conversion → sigmoid → mul → fp16 conversion → mul with up
                """
                silu_gate, up_tensor = _build_core(gate, up)

                if reverse:
                    return torch.ops.aten.mul.Tensor(up_tensor, silu_gate)
                else:
                    return torch.ops.aten.mul.Tensor(silu_gate, up_tensor)

            return pattern

        def replacement(gate, up):
            return torch.ops.tensor_cast.swiglu(gate, up)

        example_inputs = get_inputs()
        base_replacement = replacement

        # Generate both pattern variants
        pattern_up_first = _make_pattern(True)
        pattern_silu_first = _make_pattern(False)

        return [
            {
                "name": f"swiglu_mul_up_first_{dtype}",
                "pattern": pattern_up_first,
                "replacement": base_replacement,
                "inputs": example_inputs,
            },
            {
                "name": f"swiglu_mul_silu_first_{dtype}",
                "pattern": pattern_silu_first,
                "replacement": base_replacement,
                "inputs": example_inputs,
            },
        ]


def register_all_patterns():
    from . import register_pattern

    if config.compilation.fusion_patterns.enable_swiglu:
        for dtype in _SWIGLU_DTYPE_LIST:
            patterns_config = SwiGLUPattern.create(dtype)
            for pattern in patterns_config:
                register_pattern(
                    pattern["name"],
                    pattern["pattern"],
                    pattern["replacement"],
                    pattern["inputs"],
                )
