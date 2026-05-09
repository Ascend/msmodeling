# Forward Pass Kernel Traces

Single-forward-pass kernel traces extracted from Ascend Profiler `kernel_details.csv`.
Used as ground truth for M6 computation and TC-NPU alignment analysis.

## Files

| File | Model | Scenario | Tokens | Source Profiling |
|------|-------|----------|--------|-----------------|
| `qwen3-32b_pf_4112tok.csv` | Qwen3-32B | Prefill | 4112 | `profiler-qwen3-input4096-output1-concurrency1-rank0` fwd #3 |
| `qwen3-32b_dc_16tok.csv` | Qwen3-32B | Decode (batch=16) | 16 | `profiler-qwen3-input4096-output1536-concurrency1-rrate1-rank0` fwd #178 |
| `dsv3_pf_4099tok.csv` | DeepSeek-V3 | Prefill | 4099 | `profiler-dsv3-input4096-output1-concurrency1-rank0` fwd #27 |
| `dsv3_dc_1tok.csv` | DeepSeek-V3 | Decode | 1 | `profiler-dsv3-input4096-output1-concurrency1-rank0` fwd #28 |

Each CSV has the same columns as the original `kernel_details.csv`.

## Extraction Method

Forward passes detected by grouping consecutive `FusedInferAttentionScore` (FIA) anchors:

- Qwen3-32B: 64 FIA per forward pass (64 layers)
- DeepSeek-V3: 61 FIA per forward pass (61 layers)

**Time window**: `[first_FIA_start, last_FIA_end]` per forward pass.
This covers layer 0 attention through layer N-1 attention, but **excludes**:

- Pre-first-FIA: embedding, layer 0 pre-attention (RmsNorm, QKV proj, RoPE, KV cache)
- Post-last-FIA: last layer FFN, output projection, sampling

Excluded portions: ~1% for prefill, ~10-20% for decode.

## Known Issues

- **hcom double-counting**: Each `hcom_allReduce_` appears on both Stream N/A and a hardware stream with identical `(start_time, duration)`. Deduplicate by `(int(start_time), kernel_type)` before summing.
- **FIA window boundary**: Does not include pre-attention or post-FFN kernels. For precise ground truth, manually extend the window using the kernel sequence patterns documented in the design spec.

## Profiling Data Base Path

`/Users/horacehxw/Data/Profiling/Profiling-0325-final-vllm-new/`
