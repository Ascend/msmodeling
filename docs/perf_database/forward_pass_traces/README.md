# Forward Pass Kernel Traces

Single-forward-pass kernel traces extracted from Ascend Profiler `kernel_details.csv`.
Used as ground truth for M6 computation and TC-NPU alignment analysis.

## Committed Files

| File | Model | Scenario | Tokens | Source Profiling |
|------|-------|----------|--------|-----------------|
| `qwen3-32b_pf_4112tok.csv` | Qwen3-32B | Prefill | 4112 | `profiler-qwen3-input4096-output1-concurrency1-rank0` fwd #3 |
| `qwen3-32b_dc_16tok.csv` | Qwen3-32B | Decode (batch=16) | 16 | `profiler-qwen3-input4096-output1536-concurrency1-rrate1-rank0` fwd #178 |
| `dsv3_pf_4099tok.csv` | DeepSeek-V3 | Prefill | 4099 | `profiler-dsv3-input4096-output1-concurrency1-rank0` fwd #27 |
| `dsv3_dc_1tok.csv` | DeepSeek-V3 | Decode | 1 | `profiler-dsv3-input4096-output1-concurrency1-rank0` fwd #28 |
| `glm5-5.1_dc_1tok_ctx2500.csv` | GLM-5.1 | Decode | 1 at ctx 2500 | `dp0_pp0_tp0_dcp0_ep0_rank0_decode-2500` fwd #35 |

## GLM-5.1 Local Traces

This directory keeps one representative GLM-5.1 decode single-forward-step
trace, matching the Qwen3 and DeepSeek-V3 examples above. Additional GLM-5.1
trace CSVs and extraction metadata remain local generated artifacts and are not
committed in this PR.

Generated GLM-5.1 trace CSVs use the same columns as the original
`kernel_details.csv`. Local metadata JSON files record the source profiling
paths and selected windows.

## Extraction Method

Qwen3 and DeepSeek-V3 forward passes are detected by grouping consecutive
`FusedInferAttentionScore` (FIA) anchors:

- Qwen3-32B: 64 FIA per forward pass (64 layers)
- DeepSeek-V3: 61 FIA per forward pass (61 layers)

For Qwen3 and DeepSeek-V3, the time window is `[first_FIA_start,
last_FIA_end]` per forward pass. This covers layer 0 attention through layer
N-1 attention, but **excludes**:

- Pre-first-FIA: embedding, layer 0 pre-attention (RmsNorm, QKV proj, RoPE, KV cache)
- Post-last-FIA: last layer FFN, output projection, sampling

Excluded portions: ~1% for prefill, ~10-20% for decode.

GLM-5.1 decode uses 80 `SparseFlashAttention` anchors per forward pass (80
layers). The committed decode trace contains the selected monotonic
`Start Time(us)` block for one forward step, anchored by those
`SparseFlashAttention` kernels; it does not use FIA boundaries.

GLM-5.1 prefill uses the parsed single-capture 2.5K profiling window directly
instead of truncating to attention anchors, because the attention-only window
drops a large MoE/communication tail that is needed for M6 alignment.

## Known Issues

- **hcom double-counting**: Each `hcom_allReduce_` appears on both Stream N/A
  and a hardware stream with identical `(start_time, duration)`. Deduplicate by
  `(int(start_time), kernel_type)` before summing.
- **FIA window boundary**: Qwen3 and DeepSeek-V3 FIA windows do not include
  pre-attention or post-FFN kernels. For precise ground truth, manually extend
  the window using the kernel sequence patterns documented in the design spec.

## Profiling Data Source

Pass the local Ascend Profiler archive root with `--profiler-root`.
