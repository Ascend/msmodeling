# Bottleneck Rules

Use these as heuristics, not absolute laws.

## Cube

High Cube share usually indicates GEMM or tensor-core-like compute dominates. Throughput differences should correlate more strongly with low-precision compute capability when shapes and parallel strategy are comparable.

## Vec

High Vec share often points to elementwise, normalization, routing, sampling, or non-GEMM work. Faster matrix units may not improve this part much.

## Comm

High Comm share points to TP collectives, EP all-to-all, reductions, or topology limits. Larger TP or EP can reduce per-device compute while increasing communication.

## Mem

High Mem share points to HBM pressure, KV cache access, activation movement, or memory capacity constraints. Decode is often more sensitive to memory and KV cache behavior than long Prefill.

## Prefill

Long Prefill is usually more compute intensive because it processes many prompt tokens in larger matrix operations. Attention cost grows with sequence length, and prefix cache reduces the effective Prefill length.

## Decode

Decode usually processes one or a few new tokens per step with large context/KV reads. It is more likely to be memory, communication, or small-batch efficiency limited.

## MoE

MoE results depend on expert compute, routing, EP communication, shared experts, redundant experts, and MOE-DP. A better EP/MOE-DP strategy on one hardware may reflect different communication versus expert-compute tradeoffs.

## Op Bound

Use `text_generate --dump-op-bound-results` when phase-level metrics are not enough to localize a bottleneck. Sort by operator total time first, then inspect the dominant bound and memory/comm/mma/gp percentages for the top operators.

Map op bounds to phase hypotheses:

- `memory_bound` -> Mem: HBM, KV cache, activation, or weight movement pressure.
- `communication_bound` -> Comm: TP collectives, EP all-to-all, reductions, or topology limits.
- `compute_bound_mma` -> Cube: GEMM or tensor-core-like matrix compute.
- `compute_bound_gp` -> Vec: scalar/vector/general-purpose compute, elementwise, routing, or non-MMA work.

Treat op-bound output as simulated operator attribution. It can explain TensorCast model behavior and guide validation, but it is not real profiler or kernel evidence.

## Strategy Checks

Suspicious cases:

- faster hardware has much lower throughput with identical phase breakdown and no SLO/memory explanation
- throughput ratio is much higher than all relevant hardware ratios without a parallel strategy explanation
- EP improves a dense model result
- a Decode-heavy case scales like pure compute without breakdown evidence
- aggregation explanation ignores separate Prefill and Decode breakdowns
