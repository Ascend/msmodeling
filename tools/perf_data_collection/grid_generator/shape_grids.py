"""Theory-guided shape grid constants for NPU operator performance database.

Defines fixed-dimension grids (model-architecture-determined) and variable-dimension
sampling points (runtime-determined) used by `generate_shape_grid.py --mode theory`.

Reference: OPERATOR_PERF_DATABASE_DESIGN_zh_v1.5.md, Appendix I.
"""

# ═══════════════════════════════════════════════════════════════
#  Fixed dimensions — enumerated from LLM architecture space
# ═══════════════════════════════════════════════════════════════

# GEMM N/K: covers hidden_size / intermediate_size across TP=[1,2,4,8,16,32].
# Includes exact values for DSv3, Qwen3-32B, LLaMA-70B, Kimi-K2.
NK_GRID: list[int] = [
    128, 192, 256, 320, 384, 448, 512, 640, 768, 896,
    1024, 1280, 1536, 1728, 1792, 2048, 2304, 2560, 3072, 3456, 3584,
    4096, 4608, 5120, 6144, 6912, 7168, 8192,
    10240, 12288, 13824, 14336, 16384, 18432, 27648, 28672, 55296,
]

# Attention num_heads (after TP split)
HEADS_GRID: list[int] = [1, 2, 3, 4, 5, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64, 128]

# Attention head_dim — nearly all LLMs use one of these three
HEAD_DIM_GRID: list[int] = [64, 128, 256]

# GQA num_kv_heads (after TP split)
KV_HEADS_GRID: list[int] = [1, 2, 4, 8, 16, 32]


# ═══════════════════════════════════════════════════════════════
#  Variable dimensions — sampled based on performance characteristics
# ═══════════════════════════════════════════════════════════════

# GEMM M (num_tokens): Dense in small values (decode phase, memory-bound),
# sparse in large values (prefill phase, compute-bound).
# Matches AIConfigurator's exact M sampling strategy to improve cubic interpolation accuracy.
M_GRID: list[int] = [
    1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32,
    36, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 256,
    384, 512, 768, 1024, 2048, 4096, 8192, 16384, 32768,
]

# Elementwise num_tokens: sparse (memory-bound ≈ linear, few points suffice).
ELEM_TOKENS_GRID: list[int] = [
    1, 4, 16, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768,
]

# Pad num_tokens: specifically targeting unaligned sizes that trigger padding logic.
PAD_TOKENS_GRID: list[int] = [
    1, 2, 3, 4, 5, 6, 7, 9, 15, 31, 63, 127, 129, 255, 257, 511, 513,
    1023, 1025, 2047, 2049, 4095, 4097, 8191, 8193,
]

# Attention seq_len / avg_seq_len: sqrt-space uniform (O(seq^2) complexity).
ATTN_SEQ_GRID: list[int] = [
    1, 4, 16, 64, 128, 256, 512, 1024, 2048, 4096,
    8192, 16384, 32768, 65536, 131072,
]

# Attention batch_size
ATTN_BATCH_GRID: list[int] = [1, 2, 4, 8, 16, 32, 48, 64, 128]

# MOE tokens_per_expert: similar to GEMM M (GroupedMatmul = batched GEMM).
MOE_TOKENS_GRID: list[int] = [
    1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048,
    4096, 8192, 16384, 32768, 65536,
]


# ═══════════════════════════════════════════════════════════════
#  Elementwise hidden-dim grid (subset of NK_GRID for common D values)
# ═══════════════════════════════════════════════════════════════

# For elementwise / norm / quantize ops, the "fixed" dimension is D (hidden_size/tp).
# We reuse NK_GRID but can filter to commonly used values if needed.
ELEM_HIDDEN_GRID: list[int] = list(NK_GRID)
