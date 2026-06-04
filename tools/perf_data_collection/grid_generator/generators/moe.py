from __future__ import annotations

from typing import Generator

try:
    from ..model_configs import get_moe_configs
    from ..shape_grids import MOE_TOKENS_GRID
except ImportError:
    from model_configs import get_moe_configs
    from shape_grids import MOE_TOKENS_GRID

from .base import TheoryShapeRow


def generate_grouped_matmul_rows(
    model_names: list[str] | None,
) -> Generator[TheoryShapeRow, None, None]:
    for cfg in get_moe_configs(model_names):
        for ep in cfg.ep_sizes:
            for tp in cfg.tp_sizes:
                hidden = max(1, cfg.hidden_size // tp)
                expert_intermediate = max(1, cfg.expert_intermediate_size // tp)
                experts = max(1, cfg.num_experts_per_card // ep)
                expert_rows = max(1, expert_intermediate // 16)
                for tokens in MOE_TOKENS_GRID:
                    yield TheoryShapeRow(
                        [
                            (tokens, hidden),
                            (experts, hidden, expert_rows, 16),
                            (),
                            (experts, expert_intermediate),
                            (),
                            (),
                            (),
                            (experts,),
                            (tokens,),
                        ],
                        [(tokens, expert_intermediate)],
                    )


def generate_dispatch_ffn_combine_rows(
    model_names: list[str] | None,
) -> Generator[TheoryShapeRow, None, None]:
    """DispatchFFNCombine: super-fusion for MoE.

    Product spec shapes:
      x:         (M, H)
      weight1:   (expertPerRank, H, N)
      weight2:   (expertPerRank, N/2, H)
      expertIdx: (M, topK)
      scale1:    (expertPerRank * N,)
      scale2:    (expertPerRank * H,)
      probs:     (M, topK)
      out:       (M, H), (expertPerRank,)

    Product spec constraints:
      expertPerRank in [1, 17]

    Multi-node deployments:
      DFC is a per-card operator. Different deployment scales change
      num_experts_per_card = num_experts / total_cards.
      E.g. DSv3 (256 experts): 8 cards -> 32/card, 16 cards -> 16/card,
      32 cards -> 8/card, 64 cards -> 4/card.
    """
    # Deployment scales (total cards) to enumerate.
    DFC_DEPLOYMENT_CARDS = (8, 16, 32, 64)

    for cfg in get_moe_configs(model_names):
        # Collect all valid expertPerRank values across deployment scales and EP.
        # Different (total_cards, ep) combos can yield the same expertPerRank,
        # so we deduplicate upfront to avoid redundant inner iterations.
        expert_per_rank_set: set[int] = set()
        for total_cards in DFC_DEPLOYMENT_CARDS:
            if total_cards > cfg.num_experts:
                continue
            experts_per_card = cfg.num_experts // total_cards
            for ep in cfg.ep_sizes:
                if ep > experts_per_card:
                    continue
                experts = max(1, experts_per_card // ep)
                if 1 <= experts <= 17:
                    expert_per_rank_set.add(experts)

        seen: set[tuple] = set()  # deduplicate identical shape rows
        h = cfg.hidden_size
        topk = cfg.topk
        for experts in sorted(expert_per_rank_set):
            for tp in cfg.tp_sizes:
                n = max(1, cfg.expert_intermediate_size // tp) * 2  # N
                for m in MOE_TOKENS_GRID:
                    if m * h * topk > 58 * 1024 * 1024:
                        continue
                    key = (m, h, n, experts, topk)
                    if key in seen:
                        continue
                    seen.add(key)
                    yield TheoryShapeRow(
                        [
                            (m, h),                  # x: (M, H)
                            (experts, h, n),         # weight1: (expertPerRank, H, N)
                            (experts, n // 2, h),    # weight2: (expertPerRank, N/2, H)
                            (m, topk),               # expertIdx: (M, topK)
                            (experts * n,),          # scale1: (expertPerRank * N,)
                            (experts * h,),          # scale2: (expertPerRank * H,)
                            (m, topk),               # probs: (M, topK)
                        ],
                        [(m, h), (experts,)],        # out: (M, H), (expertPerRank,)
                        extra_values={"EP Size": str(cfg.num_experts // experts)},
                    )
