# RFC: Support Prefix Cache Hit Rate in CLI `text_generate` / `throughput_optimizer`

## Metadata

| Item | Content |
|:---|:---|
| **Status** | Approved |
| **Author** | yaohan404, Codex |
| **Created Date** | 2026-3-23 |
| **Related Links** | None |

---

## 1. Summary

This RFC adds `--prefix-cache-hit-rate` to the following CLIs so they can quickly estimate performance with prefix cache enabled:

- `cli.inference.text_generate`
- `cli.inference.throughput_optimizer`

The first version is a token-level approximation only. It does not model block-level hits, prefix-cache management overhead, or standalone resident prefix-cache memory.

## 2. Goals and Non-goals

Goals:

- allow users to configure prefix cache hit rate from CLI
- let `text_generate` estimate latency under prefix cache
- let `throughput_optimizer` include prefix cache impact in prefill modeling

Non-goals:

- do not modify the underlying performance model
- no block-level hit modeling
- no hash / lookup / eviction / replacement modeling
- no serving-system-level prefix-cache simulation

## 3. Core Semantics

- prefix cache only affects `prefill`
- hit rate is approximated at `token` granularity
- requests in the same batch are assumed to share prompt length and hit rate

### 3.1 `text_generate`

Original input:

- `context_length = C`
- `query_length = Q`

If `H = floor(Q * hit_rate)` query tokens are hit, rewrite the request as:

- `effective_context_length = C + H`
- `effective_query_length = Q - H`

At the modeling layer:

- `RequestInfo.query_len = effective_query_length`
- `RequestInfo.seq_len = effective_context_length + effective_query_length`

So total `seq_len` stays unchanged, while the number of tokens that still need prefill is reduced.

Example:

- original: `context_length = 1000`, `query_length = 200`
- `hit_rate = 0.5`
- result: `effective_context_length = 1100`, `effective_query_length = 100`, `seq_len = 1200`

### 3.2 `throughput_optimizer`

Internally introduce:

- `cached_prefix_tokens = floor(input_length * prefix_cache_hit_rate)`
- `effective_input_length = input_length - cached_prefix_tokens`

Policy:

- all prefill-related paths use `effective_input_length`
- all decode-related paths keep the original logic

## 4. Design

### 4.1 CLI and config

Add this argument to both entrypoints:

- `--prefix-cache-hit-rate`

Constraints:

- type: `float`
- default: `0.0`
- range: `[0, 1)`
- examples use `0.5`, not `50%`

Add this field to `UserInputConfig`:

- `prefix_cache_hit_rate: float = 0.0`

### 4.2 `text_generate` rewrite point

Compute effective lengths in `UserInputConfig.get_request_info()` so downstream code does not need to rewrite lengths again.

### 4.3 `throughput_optimizer` integration

The effective-length semantics should be introduced in the shared forward-shape construction path rather than only inside one optimizer class.

For aggregation mode:

- prefill wave capacity uses `effective_input_length`
- prefill latency uses the prefix-cache-adjusted input length
- decode latency keeps the original logic
- `TTFT` decreases with prefill latency

Here, `prefill_batch_size = max_prefill_tokens // effective_input_length` means the number of requests that fit in one prefill wave under the prefill token budget. It does not change the user-visible `batch_size`.

For disaggregation mode:

- `disaggregation-prefill` uses `effective_input_length`
- `disaggregation-decode` ignores prefix cache

### 4.4 `max_prefill_tokens`

After prefix cache is introduced, the following logic must use `effective_input_length`:

- validation against `max_prefill_tokens`
- `prefill_batch_size = max_prefill_tokens // effective_input_length` in aggregation mode

## 5. Metrics and Boundaries

### 5.1 Metric semantics

- `prefill latency`: affected by prefix cache
- `decode latency`: not directly affected by prefix cache
- `TTFT`: reduced when prefill latency is reduced
- `TPOT`: may change only if its displayed definition includes `TTFT`; that does not mean decode is optimized

### 5.2 Boundary conditions

In `text_generate`, if both are specified:

- `--decode`
- `--prefix-cache-hit-rate > 0`

the tool still runs, emits a warning, and ignores prefix cache hit rate.

The current implementation also requires:

- `effective_query_len >= 1`
- `effective_input_length >= 1`

Otherwise, the scenario is unsupported in this version.

### 5.3 Memory semantics

This scheme only approximates reduced compute for the current request:

- `text_generate` keeps total `seq_len` unchanged
- no standalone resident prefix-cache memory is modeled

So reported memory numbers should not be interpreted as total cache residency of a real serving system.

### 5.4 Out of scope for v1

- block-level hits and partial block reuse
- non-uniform hit distribution
- cache management overhead
- extra decode-stage optimizations
- high-fidelity serving-system simulation

## 6. Testing and Acceptance

Argument tests:

- default is `0.0`
- `0.5` is valid
- `-0.1` and `1.0` are invalid
- inputs leading to effective length `0` are invalid
- `text_generate --decode` with `--prefix-cache-hit-rate > 0` should emit a warning

`text_generate` tests:

- `context_length = 1000`, `query_length = 200`, `hit_rate = 0.5`
- verify effective lengths are `1100` and `100`
- verify `seq_len = 1200`
- verify `hit_rate = 0` matches original behavior

`throughput_optimizer` tests:

- `input_length = 200`, `hit_rate = 0.5`
- verify `effective_input_length = 100`
- verify aggregation prefill uses effective length
- verify disaggregation prefill uses effective length
- verify disaggregation decode keeps the original logic
- verify `max_prefill_tokens` validation and prefill-wave capacity both use effective length
- verify invalid inputs return non-zero exit code

## 7. Future Work

If higher fidelity is needed later:

- add block-level prefix-cache modeling
- add hit-distribution modeling
- add prefix-cache management and memory modeling
- extend `serving_cast` integration for prefix-cache scenarios
