from typing import Tuple

import torch

from ... import config


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class NormalRopePattern:
    """Match paired query/key RoPE graphs with batch-aware frequency tensors.

    `q` and `k` use `[B, H, S, D]`; `cos` and `sin` use `[B, S, D]` and
    broadcast to `[B, 1, S, D]` across heads. `is_neox=True` rotates the
    contiguous dimension halves. `False` first rearranges interleaved pairs into
    that split-half form. Both results transpose from `BHSD` to `BSHD`.

    Unlike SingleRopePattern, this pattern lowers a query/key pair to
    `apply_rope`; unlike PairwiseSingleRopePattern, its input graph is
    batch-aware BHSD rather than BSHD with sequence-only frequencies.
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
            # `[B, S, D]` frequencies broadcast across the head dimension.
            cos = cos.unsqueeze(unsqueeze_dim)
            sin = sin.unsqueeze(unsqueeze_dim)

            # Convert adjacent-pair inputs into the split-half layout used by rotate_half.
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
            # NeoX stores rotation halves contiguously; only head broadcasting is needed.
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


class SingleRopePattern:
    """Match conventional single-RoPE graphs with batch-aware frequencies.

    Input `x` is BHSD. The output is BSHD when `transpose_output=True` and
    remains BHSD otherwise. `is_neox=True` rotates contiguous halves; `False`
    rearranges interleaved pairs into that form. Unlike PairwiseSingleRopePattern,
    this path uses `[B, S, D]` cos/sin tensors and does not encode explicit fp32
    accumulation.
    """

    @staticmethod
    def create(is_neox, unsqueeze_dim=1, transpose_output=True) -> Tuple:
        def get_inputs():
            x = torch.empty(2, 3, 5, 4, dtype=torch.bfloat16, device="meta")
            cos = torch.empty(2, 5, 4, dtype=torch.bfloat16, device="meta")
            sin = torch.empty(2, 5, 4, dtype=torch.bfloat16, device="meta")
            return [x, cos, sin]

        def pattern_interleave(x, cos, sin):
            # Convert interleaved pairs to the split-half layout used by rotate_half.
            cos = cos.unsqueeze(unsqueeze_dim)
            sin = sin.unsqueeze(unsqueeze_dim)
            b, h, s, d = x.shape
            x = x.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
            x_embed = (x * cos) + (rotate_half(x) * sin)
            if transpose_output:
                x_embed = torch.ops.aten.permute.default(x_embed, [0, 2, 1, 3])
            return x_embed

        def pattern_neox(x, cos, sin):
            # NeoX already stores the rotation halves contiguously.
            cos = torch.ops.aten.unsqueeze.default(cos, unsqueeze_dim)
            sin = torch.ops.aten.unsqueeze.default(sin, unsqueeze_dim)
            rotated = rotate_half(x)
            x_embed = torch.ops.aten.add.Tensor(
                torch.ops.aten.mul.Tensor(x, cos), torch.ops.aten.mul.Tensor(rotated, sin)
            )
            if transpose_output:
                # Convert the input BHSD layout to BSHD.
                x_embed = torch.ops.aten.permute.default(x_embed, [0, 2, 1, 3])
            return x_embed

        def replacement(x, cos, sin):
            return torch.ops.tensor_cast.apply_rope_single(x, cos, sin, is_neox, transpose_output)

        if is_neox:
            return (pattern_neox, replacement, get_inputs())
        else:
            return (pattern_interleave, replacement, get_inputs())


class PairwiseSingleRopePattern:
    """Match BSHD single-RoPE with sequence-only frequencies and adjacent-pair rotation.

    `x` is `[B, S, H, D]`; `cos` and `sin` are `[S, D]` and broadcast to
    `[1, S, 1, D]`. For every adjacent pair `(x[..., 2i], x[..., 2i + 1])`,
    rotation builds `(-x[..., 2i + 1], x[..., 2i])` and computes
    `y = x * cos + rotate_pair(x) * sin`.

    The source graph performs the multiply/add in fp32, then casts to `x.dtype`.
    The replacement records pairwise rotation (`is_neox=False`) without a layout
    transpose. Unlike SingleRopePattern, this path is BSHD and has no batch axis
    in its frequency tensors.
    """

    @staticmethod
    def create() -> Tuple:
        def get_inputs():
            x = torch.empty(2, 3, 5, 4, dtype=torch.bfloat16, device="meta")
            cos = torch.empty(3, 4, dtype=torch.float32, device="meta")
            sin = torch.empty(3, 4, dtype=torch.float32, device="meta")
            return [x, cos, sin]

        def pattern(x, cos, sin):
            # `[S, D]` frequencies broadcast across the batch and head dimensions.
            cos = cos[None, :, None, :]
            sin = sin[None, :, None, :]
            # Rotate adjacent real/imag coordinates: [x0, x1, ...] -> [-x1, x0, ...].
            x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
            x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
            # Preserve the source graph's fp32 arithmetic and cast-back contract.
            return (x.float() * cos + x_rotated.float() * sin).to(x.dtype)

        def replacement(x, cos, sin):
            return torch.ops.tensor_cast.apply_rope_single(x, cos, sin, False, False)

        return pattern, replacement, get_inputs()


# TODO(hw-whx): add support for special rope type of GLM4.5.
def register_all_patterns():
    from . import register_pattern

    if config.compilation.fusion_patterns.enable_rope:
        pattern, replacement, example_inputs = PairwiseSingleRopePattern.create()
        register_pattern(
            "apply_rope_single_pairwise_bshd_fp32",
            pattern,
            replacement,
            example_inputs,
            level=0,
        )
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
            for transpose_output in (False, True):
                pattern, replacement, example_inputs = SingleRopePattern.create(
                    is_neox, transpose_output=transpose_output
                )
                register_pattern(
                    f"apply_rope_single_neox_{is_neox}_transpose_{transpose_output}",
                    pattern,
                    replacement,
                    example_inputs,
                    level=0,
                )
