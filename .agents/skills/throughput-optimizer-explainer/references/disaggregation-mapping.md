# Disaggregation Mapping

Disaggregation mode maps directly to a single phase.
When the user needs bottleneck attribution, append `--dump-op-bound-results` to the mapped validation command, or run `scripts/build_text_generate_commands.py --include-op-bound`.

## Prefill

When the row has TTFT and no TPOT, treat it as Prefill capability.

```bash
python -m cli.inference.text_generate <model> \
  --device <device> \
  --num-devices <num_devices> \
  --num-queries <concurrency> \
  --query-length <input_length> \
  --context-length 0 \
  --prefix-cache-hit-rate <prefix_cache_hit_rate> \
  --tp-size <TP> \
  --dp-size <DP> \
  --ep-size <EP> \
  --moe-dp-size <MOE-DP>
```

## Decode

When the row has TPOT and no TTFT, treat it as Decode capability.

```bash
python -m cli.inference.text_generate <model> \
  --device <device> \
  --num-devices <num_devices> \
  --num-queries <concurrency> \
  --query-length <num_mtp_tokens + 1> \
  --context-length <decode_context_length + output_length // 2> \
  --decode \
  --tp-size <TP> \
  --dp-size <DP> \
  --ep-size <EP> \
  --moe-dp-size <MOE-DP>
```

## PD Ratio

PD ratio mode combines independently optimized Prefill and Decode rows. Explain Prefill QPS and Decode QPS separately, then explain the selected ratio as balancing the two service rates under the available device budget.
For op-bound analysis, generate separate Prefill and Decode text_generate commands from the selected rows and compare them phase by phase. Keep the conclusion at simulated operator-attribution level unless real profiler data is provided.

For Decode, `decode_context_length` is the full KV-cache prompt context: use the original `input_length` for fixed-length input, or the weighted raw `num_input_tokens` representative when a length distribution is used. Do not reduce it for prefix-cache hits.
