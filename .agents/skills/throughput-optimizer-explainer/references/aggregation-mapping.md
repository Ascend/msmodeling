# Aggregation Mapping

Aggregation mode optimizes a combined Prefill-Decode serving instance. Its best throughput is a service-level metric, not the TPS of one forward.

## Internal Model

For an aggregation row:

```text
effective_input_length = max(1, input_length - floor(input_length * prefix_cache_hit_rate))
prefill_batch_size = max_prefill_tokens // effective_input_length
concurrency = batch_size * DP * PP
```

The optimizer runs or caches two phase simulations:

- Prefill: `is_decode=False`, `query_len=seq_len=effective_input_length`, `concurrency=prefill_batch_size`.
- Decode: `is_decode=True`, `query_len=num_mtp_tokens + 1`, `seq_len=input_length + output_length // 2 + query_len`, `concurrency=batch_size * DP * PP`.

Then it combines these phase latencies into TTFT, TPOT, and output throughput.

## Text Generate Mapping

Use the same model, device, num devices, quantization, compile flags, and parallel strategy as the best row.
When the user needs bottleneck attribution, append `--dump-op-bound-results` to both validation commands, or run `scripts/build_text_generate_commands.py --include-op-bound`.

Prefill validation:

```bash
python -m cli.inference.text_generate <model> \
  --device <device> \
  --num-devices <num_devices> \
  --num-queries <prefill_batch_size> \
  --query-length <effective_input_length> \
  --context-length 0 \
  --tp-size <TP> \
  --dp-size <DP> \
  --ep-size <EP> \
  --moe-dp-size <MOE-DP>
```

Decode validation:

```bash
python -m cli.inference.text_generate <model> \
  --device <device> \
  --num-devices <num_devices> \
  --num-queries <concurrency> \
  --query-length <num_mtp_tokens + 1> \
  --context-length <input_length + output_length // 2> \
  --decode \
  --tp-size <TP> \
  --dp-size <DP> \
  --ep-size <EP> \
  --moe-dp-size <MOE-DP>
```

If `concurrency % prefill_batch_size != 0`, mention that the optimizer also accounts for a partial Prefill wave. Do not generate a partial-wave command unless it is needed for a precise TTFT explanation.

## Op Bound Attribution

For aggregation results, collect op-bound output separately for Prefill and Decode. Compare each phase against its own mapped command; never use one combined aggregation row as if it were one forward pass. Label conclusions as simulated operator attribution unless real profiler data is provided.

## Prefix Cache

For aggregation validation, prefer passing the already reduced `effective_input_length` to `text_generate` instead of passing the original input length plus `--prefix-cache-hit-rate`. This matches the optimizer's internal Prefill request construction more directly.
