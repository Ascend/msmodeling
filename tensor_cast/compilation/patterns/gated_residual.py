import torch

from ... import config


class GatedResidualAddPattern:
    @staticmethod
    def create(reverse_mul: bool, reverse_add: bool):
        def get_inputs():
            residual = torch.empty(2, 3, 4, dtype=torch.bfloat16, device="meta")
            update = torch.empty(2, 3, 4, dtype=torch.bfloat16, device="meta")
            gate = torch.empty(2, 1, 4, dtype=torch.bfloat16, device="meta")
            return [residual, update, gate]

        def pattern(residual, update, gate):
            if reverse_mul:
                gated = torch.ops.aten.mul.Tensor(gate, update)
            else:
                gated = torch.ops.aten.mul.Tensor(update, gate)
            if reverse_add:
                return torch.ops.aten.add.Tensor(gated, residual)
            return torch.ops.aten.add.Tensor(residual, gated)

        def replacement(residual, update, gate):
            return torch.ops.tensor_cast.gated_residual_add(residual, update, gate)

        return pattern, replacement, get_inputs()


def register_all_patterns():
    from . import register_pattern

    if not config.compilation.fusion_patterns.enable_gated_residual_add:
        return

    for reverse_mul in (False, True):
        for reverse_add in (False, True):
            pattern, replacement, example_inputs = GatedResidualAddPattern.create(reverse_mul, reverse_add)
            register_pattern(
                f"gated_residual_add_pattern_mul_{reverse_mul}_add_{reverse_add}",
                pattern,
                replacement,
                example_inputs,
                level=2,
            )
