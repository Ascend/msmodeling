from typing import Tuple

import torch

from ... import config


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class NormalRopePattern:
    """
    Pattern for Applying rotary embedding.
    This pattern applies rotary embedding to the query and key tensors.
    """

    @staticmethod
    def create(is_neox, unsqueeze_dim=1) -> Tuple:
        def get_inputs():
            q = torch.empty(4, 4, 4, 4, device="meta")
            k = torch.empty(4, 4, 4, 4, device="meta")
            cos = torch.empty(4, 4, 4, device="meta")
            sin = torch.empty(4, 4, 4, device="meta")
            return [q, k, cos, sin]

        def pattern_interleave(q, k, cos, sin):
            cos = cos.unsqueeze(unsqueeze_dim)
            sin = sin.unsqueeze(unsqueeze_dim)

            b, h, s, d = q.shape
            q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

            b, h, s, d = k.shape
            k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

            q_embed = (q * cos) + (rotate_half(q) * sin)
            k_embed = (k * cos) + (rotate_half(k) * sin)
            # BHSD -> BSHD
            q_embed = q_embed.transpose(1, 2)
            k_embed = k_embed.transpose(1, 2)
            return q_embed, k_embed

        def pattern_neox(q, k, cos, sin):
            cos = cos.unsqueeze(unsqueeze_dim)
            sin = sin.unsqueeze(unsqueeze_dim)
            q_embed = (q * cos) + (rotate_half(q) * sin)
            k_embed = (k * cos) + (rotate_half(k) * sin)
            # BHSD -> BSHD
            q_embed = q_embed.transpose(1, 2)
            k_embed = k_embed.transpose(1, 2)
            return q_embed, k_embed

        def replacement(q, k, cos, sin):
            q_embed, k_embed = torch.ops.tensor_cast.apply_rope(q, k, cos, sin, is_neox)
            return q_embed, k_embed

        if is_neox:
            return (pattern_neox, replacement, get_inputs())
        else:
            return (pattern_interleave, replacement, get_inputs())


# TODO(hw-whx): add support for special rope type of GLM4.5.
def register_all_patterns():
    from . import register_pattern

    if config.compilation.fusion_patterns.enable_rope:
        for is_neox in [False, True]:
            pattern, replacement, example_inputs = NormalRopePattern.create(is_neox)
            # Register the pattern with the PatternManager
            register_pattern(
                f"apply_rope_pattern_is_neox({is_neox})",
                pattern,
                replacement,
                example_inputs,
                level=0,
            )
