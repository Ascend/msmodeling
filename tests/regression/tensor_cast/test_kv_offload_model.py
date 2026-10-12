from types import SimpleNamespace

import pytest

from tensor_cast.core.kv_offload import (
    HostKVCapacityExceeded,
    decode_hot_buffer_bytes,
    decode_offload_miss_transfer_time_ms,
    prefill_layerwise_footprint_ratio,
)
from tensor_cast.core.model_runner import ModelRunner

# main_kv_per_token = num_layers * row_bytes; row_bytes (one token K+V, one
# layer, per rank) = kv_width_per_rank * 2 * dtype_bytes.
# For num_layers=4, kv_width=128, dtype=2 (bf16): main_kv_per_token = 4 * 512 = 2048.
_MKPT = 2048  # 4 layers * (128 * 2 * 2)


def test_decode_hot_buffer_bytes():
    # num_seqs=2, topk=8, mult=2 -> 2 * 8 * 2 * 2048 = 65536
    assert decode_hot_buffer_bytes(_MKPT, topk=8, buffer_multiplier=2.0, num_seqs=2) == 65536


def test_decode_hot_buffer_scales_linearly_with_seqs_and_topk():
    base = decode_hot_buffer_bytes(61 * 2048, topk=2048, buffer_multiplier=2.0, num_seqs=1)
    assert decode_hot_buffer_bytes(61 * 2048, topk=2048, buffer_multiplier=2.0, num_seqs=4) == 4 * base
    assert decode_hot_buffer_bytes(61 * 2048, topk=4096, buffer_multiplier=2.0, num_seqs=1) == 2 * base


def test_decode_miss_transfer_formula():
    # main_kv_per_token = 61 * 2048 = 124928; miss_rows_per_layer = ceil(2048*0.15) = 308
    # payload = num_seqs * (miss_rows_per_layer + 1 new-KV token) * mkpt
    #        = 1 * 309 * 124928 = 38_602_752; bw = 32e9*0.8 = 25.6e9
    mkpt = 61 * 2048
    ms = decode_offload_miss_transfer_time_ms(
        mkpt,
        topk=2048,
        hit_rate=0.85,
        num_seqs=1,
        host_bandwidth_bytes_ps=32e9,
        host_latency_s=5e-6,
        host_efficiency=0.8,
    )
    expected = (5e-6 + 38_602_752 / 25.6e9) * 1000.0
    assert ms == pytest.approx(expected, rel=1e-9)


def test_decode_miss_transfer_scales_with_num_seqs():
    # Each active sequence fetches its own miss rows and writes back its own
    # new KV: the payload must scale linearly with the local batch.
    common = dict(
        main_kv_per_token_bytes=61 * 2048,
        topk=2048,
        hit_rate=0.85,
        host_bandwidth_bytes_ps=32e9,
        host_latency_s=0.0,
        host_efficiency=1.0,
    )
    one = decode_offload_miss_transfer_time_ms(num_seqs=1, **common)
    four = decode_offload_miss_transfer_time_ms(num_seqs=4, **common)
    assert four == pytest.approx(4 * one, rel=1e-9)


def test_decode_miss_transfer_full_hit_still_writes_back_new_kv():
    # hit_rate=1.0 -> zero miss rows, but the main KV lives on host: every new
    # token's KV must be persisted back each step. There is no free offload.
    mkpt = 61 * 2048
    ms = decode_offload_miss_transfer_time_ms(
        mkpt,
        topk=2048,
        hit_rate=1.0,
        num_seqs=1,
        host_bandwidth_bytes_ps=32e9,
        host_latency_s=0.0,
        host_efficiency=1.0,
    )
    assert ms == pytest.approx(mkpt / 32e9 * 1000.0, rel=1e-9)
    assert ms > 0


def test_decode_miss_transfer_monotonic_in_hit_rate():
    common = dict(
        main_kv_per_token_bytes=61 * 2048,
        topk=2048,
        host_bandwidth_bytes_ps=32e9,
        host_latency_s=5e-6,
        host_efficiency=0.8,
    )
    t_low = decode_offload_miss_transfer_time_ms(hit_rate=0.5, **common)
    t_mid = decode_offload_miss_transfer_time_ms(hit_rate=0.85, **common)
    t_high = decode_offload_miss_transfer_time_ms(hit_rate=0.99, **common)
    assert t_low > t_mid > t_high


def test_decode_miss_transfer_zero_bandwidth_returns_swap_only():
    # Unconfigured host link (0 bandwidth) -> not modeled; only the swap kernel
    # constant is returned (fail-fast is the estimator's job, not this helper's).
    assert (
        decode_offload_miss_transfer_time_ms(
            61 * 2048, topk=2048, hit_rate=0.85, host_bandwidth_bytes_ps=0.0, swap_kernel_ms=0.3
        )
        == 0.3
    )
    assert decode_offload_miss_transfer_time_ms(61 * 2048, topk=2048, hit_rate=0.85, host_bandwidth_bytes_ps=0.0) == 0.0


def test_decode_miss_transfer_full_hit_never_free():
    # hit_rate=1.0 -> zero miss rows, but the per-step new-KV D2H write-back
    # still applies: swapping must never make the transfer free.
    assert (
        decode_offload_miss_transfer_time_ms(
            61 * 2048,
            topk=2048,
            hit_rate=1.0,
            num_seqs=1,
            host_bandwidth_bytes_ps=32e9,
            swap_kernel_ms=0.2,
        )
        > 0.2
    )


def test_prefill_layerwise_footprint_ratio():
    assert prefill_layerwise_footprint_ratio(80, 0, 4) == pytest.approx(4 / 80)
    assert prefill_layerwise_footprint_ratio(80, 2, 4) == pytest.approx(6 / 80)
    # B clamped to R = N - I
    assert prefill_layerwise_footprint_ratio(80, 70, 100) == pytest.approx(1.0)
    assert prefill_layerwise_footprint_ratio(0, 0, 4) == 1.0


def test_prefill_layerwise_footprint_ratio_rejects_negative_args():
    with pytest.raises(ValueError):
        prefill_layerwise_footprint_ratio(80, -1, 4)
    with pytest.raises(ValueError):
        prefill_layerwise_footprint_ratio(80, 0, -2)
    # I > N must clamp, never exceed ratio 1
    assert prefill_layerwise_footprint_ratio(80, 200, 4) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Accounting-level fail-fast and exact-value tests (review P1 fixes).
# ---------------------------------------------------------------------------


def _stub_runner(**user_overrides):
    user = SimpleNamespace(
        enable_decode_kv_offload=True,
        enable_prefill_kv_offload=False,
        decode_offload_buffer_multiplier=2.0,
        decode_offload_hit_rate=0.85,
        reserved_memory_gb=5.0,
    )
    for k, v in user_overrides.items():
        setattr(user, k, v)
    profile = SimpleNamespace(
        name="TEST_DEVICE",
        host_bandwidth_bytes_ps=32e9,
        host_latency_s=5e-6,
        host_transfer_efficiency=0.8,
    )
    # The sparse window (top-k) is resolved from the model checkpoint config.
    model = SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace(index_topk=2048)))
    return SimpleNamespace(user_input=user, device_profile=profile, model=model)


def _apply_decode(runner, kv_bytes=1024**3, indexer_bytes=1024**2, mkpt=90_000, batch=1):
    return ModelRunner._apply_decode_kv_offload_accounting(runner, 100.0, kv_bytes, indexer_bytes, mkpt, batch, True)


def _apply_prefill(runner, kv_bytes=1024**3, indexer_bytes=1024**2, mkpt=90_000, batch=1, layers=78):
    return ModelRunner._apply_prefill_kv_offload_accounting(
        runner, 100.0, kv_bytes, indexer_bytes, mkpt, batch, False, layers
    )


def test_decode_offload_rejects_model_without_sparse_window():
    # The sparse window is a checkpoint-config property: a model whose config
    # declares no index_topk/topk_limit cannot offload safely, because the
    # hot buffer would be sized against an unknown window.
    runner = _stub_runner()
    runner.model = SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="qwen3")))
    with pytest.raises(ValueError, match="sparse attention window"):
        _apply_decode(runner)


def test_decode_offload_rejects_dense_model_without_indexer():
    # Dense/GQA models have no indexer cache: the hot buffer cannot restore a
    # decode step, so removing the main KV must be refused.
    runner = _stub_runner()
    with pytest.raises(ValueError, match="indexer cache"):
        _apply_decode(runner, indexer_bytes=0)


def test_decode_offload_rejects_unconfigured_host_link():
    runner = _stub_runner()
    runner.device_profile.host_bandwidth_bytes_ps = 0.0
    with pytest.raises(ValueError, match="host bandwidth"):
        _apply_decode(runner)


def test_decode_offload_rejects_zero_transfer_efficiency():
    # The gate must check the effective bandwidth (bw x efficiency), matching
    # the prefill accounting and the estimator: efficiency = 0 would otherwise
    # keep the capacity benefit while the miss transfer silently costs 0.
    runner = _stub_runner()
    runner.device_profile.host_transfer_efficiency = 0.0
    with pytest.raises(ValueError, match="host_transfer_efficiency"):
        _apply_decode(runner)


def test_decode_offload_rejects_host_pool_over_capacity():
    # Host capacity gate: an offloaded KV pool beyond --host-memory-available-gb
    # must fail fast instead of returning an undeployable result.
    runner = _stub_runner(host_memory_available_gb=1.0)
    with pytest.raises(ValueError, match="host memory budget"):
        _apply_decode(runner, kv_bytes=2 * 1024**3)


def test_decode_offload_allows_host_pool_within_capacity():
    runner = _stub_runner(host_memory_available_gb=4.0)
    peak, kv_gb, mode, pool, transfer = _apply_decode(runner, kv_bytes=2 * 1024**3)
    assert mode == "decode"
    assert pool == pytest.approx(2.0)
    assert peak > 0


def test_decode_offload_happy_path_exact_values():
    runner = _stub_runner()
    kv_bytes, indexer_bytes, mkpt, batch = 2**30, 1024**2, 90_000, 4
    peak, kv_gb, mode, pool, transfer = _apply_decode(
        runner, kv_bytes=kv_bytes, indexer_bytes=indexer_bytes, mkpt=mkpt, batch=batch
    )
    assert mode == "decode"
    hot = batch * 2048 * 2.0 * mkpt
    assert pool == kv_bytes / 1024**3
    assert kv_gb == (indexer_bytes + hot) / 1024**3
    assert peak == 100.0 - kv_bytes / 1024**3 + hot / 1024**3
    # miss rows = ceil(2048*0.15) = 308; payload = 4 * (308+1) * 90_000
    payload = batch * (308 + 1) * mkpt
    assert transfer == pytest.approx((5e-6 + payload / (32e9 * 0.8)) * 1000.0, rel=1e-9)


def test_prefill_offload_ratio_one_is_noop():
    # B covers every layer -> nothing to offload: no host pool, no transfer.
    runner = _stub_runner(enable_prefill_kv_offload=True)
    peak, kv_gb, mode, pool, transfer = _apply_prefill(runner, layers=2, kv_bytes=1024**3)
    assert mode == "prefill"
    assert pool == 0.0
    assert transfer == 0.0
    assert peak == 100.0
    assert kv_gb == (1024**3 + 1024**2) / 1024**3


def test_prefill_offload_host_pool_and_transfer_cover_offloaded_share_only():
    # 78 layers, B=2 (code default) -> ratio 2/78; only (1-ratio) offloaded.
    runner = _stub_runner(enable_prefill_kv_offload=True)
    kv_bytes = 1024**3
    peak, kv_gb, mode, pool, transfer = _apply_prefill(runner, kv_bytes=kv_bytes, layers=78)
    ratio = 2 / 78
    assert mode == "prefill"
    assert pool == kv_bytes / 1024**3 * (1 - ratio)
    expected = (5e-6 + (2 * kv_bytes * (1 - ratio)) / (32e9 * 0.8)) * 1000.0
    assert transfer == pytest.approx(expected, rel=1e-9)
    assert peak == 100.0 - kv_bytes / 1024**3 * (1 - ratio)


def test_prefill_offload_rejects_unconfigured_host_link():
    runner = _stub_runner(enable_prefill_kv_offload=True)
    runner.device_profile.host_bandwidth_bytes_ps = 0.0
    with pytest.raises(ValueError, match="host bandwidth"):
        _apply_prefill(runner)


def test_prefill_offload_rejects_host_pool_over_capacity():
    # Host capacity gate: the offloaded KV pool beyond --host-memory-available-gb
    # must fail fast instead of returning an undeployable result.
    runner = _stub_runner(enable_prefill_kv_offload=True, host_memory_available_gb=0.5)
    with pytest.raises(ValueError, match="host memory budget"):
        _apply_prefill(runner, kv_bytes=1024**3)


def test_prefill_offload_transfer_includes_host_latency():
    # The round-trip transfer carries one host-link latency term, matching the
    # decode path estimator (review: prefill transfer omitted host_latency_s).
    runner = _stub_runner(enable_prefill_kv_offload=True)
    kv_bytes = 1024**3
    peak, kv_gb, mode, pool, transfer = _apply_prefill(runner, kv_bytes=kv_bytes, layers=78)
    ratio = 2 / 78
    expected = (5e-6 + (2 * kv_bytes * (1 - ratio)) / (32e9 * 0.8)) * 1000.0
    assert transfer == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# Review fixes (ChenHuiwen 10-09): hot-buffer floor, hybrid-layer denominator,
# host-capacity as a search-domain signal.
# ---------------------------------------------------------------------------


def test_decode_offload_rejects_hot_buffer_multiplier_below_one():
    # A sub-1x hot buffer cannot hold one full top-k window: the configured
    # hit rate (default 0.85) would have no physical carrier -- the device
    # keeps (almost) no resident KV while still on-loading miss rows.
    for multiplier in (0.0, 0.5):
        runner = _stub_runner(decode_offload_buffer_multiplier=multiplier)
        with pytest.raises(ValueError, match="multiplier must be >= 1"):
            _apply_decode(runner)


def test_decode_offload_host_capacity_raises_dedicated_exception():
    # The host-pool capacity gate raises the dedicated search-domain exception
    # (a ValueError subclass, so plain ValueError matches keep working) carrying
    # the pool size and phase for the optimizer's candidate boundary.
    runner = _stub_runner(host_memory_available_gb=1.0)
    with pytest.raises(HostKVCapacityExceeded, match="host memory budget") as exc_info:
        _apply_decode(runner, kv_bytes=2 * 1024**3)
    assert exc_info.value.host_kv_pool_gb == pytest.approx(2.0)
    assert exc_info.value.phase == "decode"
    assert isinstance(exc_info.value, ValueError)


def test_prefill_offload_host_capacity_raises_dedicated_exception():
    runner = _stub_runner(enable_prefill_kv_offload=True, host_memory_available_gb=0.5)
    with pytest.raises(HostKVCapacityExceeded, match="host memory budget") as exc_info:
        _apply_prefill(runner, kv_bytes=1024**3)
    assert exc_info.value.phase == "prefill"
    assert exc_info.value.host_kv_pool_gb > 0


def test_resolve_offload_cache_layers_counts_only_cache_bearing_layers():
    # Hybrid model (e.g. Qwen3.5-style: 8 hidden layers, only 2 hold paged KV):
    # the prefill shared-buffer ratio must spread buffers across the 2 KV
    # layers, not 8 -- otherwise ratio = 2/8 manufactures a 4x phantom gain.
    per_layer = {0: 1024, 1: 1024, 2: 1024, 3: 1024}
    count = ModelRunner._resolve_offload_cache_layers(per_layer, enable_prefill_offload=True)
    assert count == 4
    assert ModelRunner._resolve_offload_cache_layers({}, enable_prefill_offload=True) == 0


def test_resolve_offload_cache_layers_rejects_non_uniform_prefill():
    # Non-uniform cache-bearing layers defeat the uniform shared-buffer ratio:
    # refuse under prefill offload instead of silently mis-scaling.
    per_layer = {0: 1024, 3: 2048}
    with pytest.raises(ValueError, match="uniform cache-bearing"):
        ModelRunner._resolve_offload_cache_layers(per_layer, enable_prefill_offload=True)
    # With offload off the check is inert (no behavior change on the OFF path).
    assert ModelRunner._resolve_offload_cache_layers(per_layer, enable_prefill_offload=False) == 2
