"""KV cache offload performance modeling helpers.

Implements the cost/benefit math for the KV cache offload strategies described
in ``docs/RFC/rfc_kv_cache_offloading_*`` (aligned with upstream vLLM RFC #48203
and the vllm-ascend layerwise + sparse KV cache offloading design):

* **Decode sparse offload**: the device main KV is moved to host; only an
  indexer + a small per-layer top-k hot buffer stay on device. Each decode step
  on-loads only the miss rows (top-k entries not already resident) from host.
* **Prefill layerwise offload**: N cache-bearing layers share B physical NPU
  buffers, compressing the main-KV NPU footprint to ``(I + min(B, R)) / N``.

These are pure functions; Phase 1 integrates them in ``ModelRunner.run_inference``
(peak post-processing + transfer time) and
``base_throughput_optimizer._build_forward_latency_record`` (TPOT fold-in).

``main_kv_per_token_bytes`` is the variant-agnostic per-token main-KV size
already accumulated by ``_get_kv_cache_info`` as ``input_kwargs["kv_cache_per_token"]``
(= ``num_layers * row_bytes`` where ``row_bytes`` is one token's K+V for one
layer on one rank). Using it avoids re-deriving the per-rank KV width from the
per-variant cache tensor shapes.
"""

import math


class HostKVCapacityExceeded(ValueError):
    """The offloaded host KV pool exceeds ``--host-memory-available-gb``.

    This is a search-domain signal, not a configuration error: candidate
    evaluations in the throughput optimizer legitimately probe batches whose
    offloaded pool outgrows the configured host budget. The optimizer catches
    it at the candidate-evaluation boundary (``_get_inference_metrics`` /
    ``_get_batched_forward_info``) and folds it into the existing OOM
    early-stop path (``device_memory_available_gb < 0``), so the search
    shrinks its upper bound and keeps already-feasible candidates instead of
    aborting the whole run. Configuration errors (non-positive bandwidth, no
    indexer, ...) keep raising plain ``ValueError``.
    """

    def __init__(self, message: str, *, host_kv_pool_gb: float = 0.0, phase: str = "none"):
        super().__init__(message)
        self.host_kv_pool_gb = host_kv_pool_gb
        self.phase = phase


def decode_hot_buffer_bytes(
    main_kv_per_token_bytes: int,
    *,
    topk: int,
    buffer_multiplier: float,
    num_seqs: int,
) -> int:
    """Device-resident top-k hot buffer size (bytes) for sparse decode offload.

    The hot buffer holds ``buffer_multiplier * topk`` selected token rows per
    active sequence in each cache-bearing layer; the returned size is the
    total across all layers, not a per-layer value. Since
    ``main_kv_per_token_bytes`` already spans all layers
    (``num_layers * row_bytes``), the buffer is::

        num_seqs * topk * buffer_multiplier * main_kv_per_token_bytes
    """
    return int(num_seqs * topk * buffer_multiplier * main_kv_per_token_bytes)


def decode_offload_miss_transfer_time_ms(
    main_kv_per_token_bytes: int,
    *,
    topk: int,
    hit_rate: float,
    num_seqs: int = 1,
    host_bandwidth_bytes_ps: float,
    host_latency_s: float = 0.0,
    host_efficiency: float = 1.0,
    swap_kernel_ms: float = 0.0,
) -> float:
    """Per-decode-step host transfer time (ms) for sparse decode offload.

    Two components, both proportional to the number of active sequences:

    - **Miss H2D**: each active sequence independently misses
      ``ceil(topk * (1 - hit_rate))`` rows per layer, so the whole batch
      fetches ``num_seqs * miss_rows_per_layer * main_kv_per_token_bytes``.
    - **New-KV D2H write-back**: the main KV lives on host, so every newly
      generated token's KV must be persisted back per sequence per step
      (``num_seqs * main_kv_per_token_bytes``). This keeps the cost
      non-zero even at ``hit_rate == 1`` -- there is no free offload.

    Because ``main_kv_per_token_bytes`` is the per-token cost summed over all
    layers, the payload is
    ``num_seqs * (miss_rows_per_layer + 1) * main_kv_per_token_bytes``.
    ``topk`` (hence ``miss_rows_per_layer = ceil(topk * (1 - hit_rate))``) is
    a per-layer row count. For the uniform-layer models covered by the
    aggregate accounting, per-layer ``row_bytes`` is uniform, so
    ``num_layers * (miss_rows_per_layer + 1) * row_bytes`` collapses exactly
    to ``(miss_rows_per_layer + 1) * main_kv_per_token_bytes`` -- the layer
    count cancels, which is why this function takes no ``num_layers``.
    Non-uniform (hybrid attention) layer models are an RFC open item.
    Upstream measured 80%-90% hit rate at ``buffer = 2 * topk``
    (DeepSeek-V3.2); the default hit rate is 0.85.

    ``time = host_latency + payload / (host_bandwidth * efficiency) + swap_kernel``.
    Returns ``swap_kernel_ms`` only when host bandwidth is not configured
    (``<= 0``) -- the fail-fast guard for an unconfigured host link lives in
    the callers (``ModelRunner._apply_decode_kv_offload_accounting`` raises
    when offload is enabled without a usable host link).
    """
    bandwidth = host_bandwidth_bytes_ps * host_efficiency
    if bandwidth <= 0:
        return swap_kernel_ms
    miss_rate = max(0.0, 1.0 - hit_rate)
    miss_rows_per_layer = math.ceil(topk * miss_rate)
    seqs = max(num_seqs, 1)
    payload = seqs * (miss_rows_per_layer + 1) * main_kv_per_token_bytes
    if payload <= 0:
        return swap_kernel_ms
    transfer_s = host_latency_s + payload / bandwidth
    return transfer_s * 1000.0 + swap_kernel_ms


def prefill_layerwise_footprint_ratio(
    num_layers: int,
    independent_layers: int,
    shared_buffers: int,
) -> float:
    """Main-KV NPU footprint ratio for prefill layerwise offload.

    ``physical_buffers = I + min(B, R)`` where ``R = N - I`` (independent layers
    get dedicated buffers; the remaining compatible layers round-robin onto the
    ``B`` shared buffers). Returns ``1.0`` when ``num_layers <= 0``.
    """
    if num_layers <= 0:
        return 1.0
    if independent_layers < 0 or shared_buffers < 0:
        raise ValueError(
            f"independent_layers ({independent_layers}) and shared_buffers ({shared_buffers}) must be non-negative"
        )
    independent = min(independent_layers, num_layers)
    remaining = num_layers - independent
    physical = independent + min(shared_buffers, remaining)
    return physical / num_layers


def fold_offload_transfer_time_ms(
    base_latency_ms: float,
    offload_mode: object,
    offload_transfer_time_ms: object,
) -> float:
    """Fold a KV offload transfer cost into a phase latency (ms).

    Single source of truth for the phase-aware fold-in shared by
    ``base_throughput_optimizer._build_forward_latency_record``, the disagg
    single-shape branch, the disagg variable-chunk loop and the agg chunked
    prefill loop (RFC ``docs/RFC/rfc_kv_cache_offloading_*``):

    - prefill layerwise offload: transfer overlaps compute ->
      ``max(base_latency_ms, transfer_ms)``;
    - decode sparse offload: miss-row H2D is serial ->
      ``base_latency_ms + transfer_ms``;
    - any other mode (``"none"``/unknown): base latency unchanged.

    ``offload_mode`` and ``offload_transfer_time_ms`` are accepted as
    ``object`` and guarded so lightweight metrics mocks (``Mock``/
    ``SimpleNamespace`` auto-create unknown attributes) degrade to a no-op
    instead of raising.
    """
    if offload_mode not in ("decode", "prefill"):
        return base_latency_ms
    transfer_ms = float(offload_transfer_time_ms) if isinstance(offload_transfer_time_ms, (int, float)) else 0.0
    if offload_mode == "prefill":
        return max(base_latency_ms, transfer_ms)
    return base_latency_ms + transfer_ms
