# Aggregation Mapping

Aggregation mode optimizes a combined Prefill-Decode serving instance. Its best throughput is a service-level metric, not the TPS of one forward.

## Internal Model

For an aggregation row:

```text
effective_input_length = max(1, input_length - floor(input_length * prefix_cache_hit_rate))
decode_context_length = get_decode_context_length()
prefill_batch_size = max_batched_tokens // effective_input_length
concurrency = batch_size * DP * PP
```

The optimizer runs or caches two phase simulations:

- Prefill: `is_decode=False`, `query_len=effective_input_length`, `seq_len=None` (resolved downstream to `cached_prefix + query_len == input_length`), `concurrency=prefill_batch_size`.
- Decode: `is_decode=True`, `query_len=num_mtp_tokens + 1`, `seq_len=decode_context_length + output_length // 2 + query_len`, `concurrency=batch_size * DP * PP`.

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
  --query-length <input_length> \
  --context-length 0 \
  --prefix-cache-hit-rate <prefix_cache_hit_rate> \
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
  --context-length <decode_context_length + output_length // 2> \
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

For aggregation validation, pass the original `input_length` together with `--prefix-cache-hit-rate`. This produces `query_len=effective_input_length` while retaining the cached prefix in the full Prefill KV `seq_len`, matching the optimizer's request construction.

`decode_context_length` is the full KV-cache prompt context: it is the original `input_length` for fixed-length input, or the weighted raw `num_input_tokens` representative for a length distribution. Prefix-cache hits reduce Prefill work only; they do not shorten Decode context.
