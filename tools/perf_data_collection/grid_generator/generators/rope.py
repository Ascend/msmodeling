from __future__ import annotations

from collections.abc import Generator

try:
    from ..model_configs import resolve_configs
    from ..shape_grids import ELEM_TOKENS_GRID
except ImportError:
    from model_configs import resolve_configs
    from shape_grids import ELEM_TOKENS_GRID

from .base import TheoryShapeRow


def generate_split_qkv_rmsnorm_rope_rows(
    model_names: list[str] | None,
) -> Generator[TheoryShapeRow, None, None]:
    """Generate replayable vLLM-Ascend fused QKV+RMSNorm+RoPE rows.

    The custom op consumes a fused QKV input whose last dimension is
    q_hidden + 2 * kv_hidden, then returns Q, K, and V tensors. The profiled
    CSV only records the fused input plus the rope cache width; replay creates
    positions and a full cos/sin cache from that metadata.
    """

    seen: set[tuple[int, int, int, int]] = set()
    for cfg in resolve_configs(model_names):
        if cfg.is_mla():
            continue

        for tp in cfg.tp_sizes:
            if cfg.num_attention_heads % tp != 0:
                continue

            local_q_heads = cfg.num_attention_heads // tp
            if cfg.num_kv_heads >= tp:
                local_kv_heads = cfg.num_kv_heads // tp
            elif tp % cfg.num_kv_heads == 0:
                local_kv_heads = 1
            else:
                continue

            q_hidden = local_q_heads * cfg.head_dim
            kv_hidden = local_kv_heads * cfg.head_dim
            input_hidden = q_hidden + 2 * kv_hidden
            rope_dim = cfg.head_dim
            for tokens in ELEM_TOKENS_GRID:
                key = (tokens, q_hidden, kv_hidden, rope_dim)
                if key in seen:
                    continue
                seen.add(key)
                yield TheoryShapeRow(
                    [(tokens, input_hidden), (rope_dim,)],
                    [(tokens, q_hidden), (tokens, kv_hidden), (tokens, kv_hidden)],
                    extra_values={
                        "Input Data Types": "DT_BF16;DT_BF16",
                        "Input Formats": "ND;ND",
                        "Output Data Types": "DT_BF16;DT_BF16;DT_BF16",
                        "Output Formats": "ND;ND;ND",
                    },
                )
