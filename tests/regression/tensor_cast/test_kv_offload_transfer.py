import types

import pytest
import torch

from tensor_cast.device import TEST_DEVICE, DeviceProfile
from tensor_cast.performance_model.bound_analyzer import StatsKey
from tensor_cast.performance_model.op_estimator_registry import get_op_estimator

# Ensure the kv_offload_transfer custom op is registered before the estimator
# table is queried.
import tensor_cast.ops.communication  # noqa: F401


KV_OFFLOAD_OP = torch.ops.tensor_cast.kv_offload_transfer.default


def _estimator():
    est = get_op_estimator(KV_OFFLOAD_OP, None)
    assert est is not None, "kv_offload_transfer estimator not registered"
    return est


def test_device_profile_host_link_fields():
    import dataclasses

    names = {f.name for f in dataclasses.fields(DeviceProfile)}
    assert {"host_bandwidth_bytes_ps", "host_latency_s", "host_transfer_efficiency"} <= names
    assert TEST_DEVICE.host_bandwidth_bytes_ps > 0
    assert TEST_DEVICE.host_latency_s > 0
    assert 0 < TEST_DEVICE.host_transfer_efficiency <= 1.0


def test_kv_offload_transfer_op_passthrough():
    kv = torch.randn(8, 16, dtype=torch.bfloat16)
    out = KV_OFFLOAD_OP(kv, 0)
    assert out.shape == kv.shape
    assert out.dtype == kv.dtype


def test_estimator_time_is_latency_plus_payload_over_bandwidth():
    est = _estimator()
    kv = torch.randn(4, 128, dtype=torch.float32)  # 2048 bytes
    info = types.SimpleNamespace(args=(kv, 0))
    result = est(info, TEST_DEVICE)

    payload = kv.numel() * kv.element_size()
    bw = TEST_DEVICE.host_bandwidth_bytes_ps * TEST_DEVICE.host_transfer_efficiency
    expected = TEST_DEVICE.host_latency_s + payload / bw
    assert result.execution_time_s == pytest.approx(expected, rel=1e-9, abs=1e-12)
    assert result.statistics[StatsKey.COMMUNICATION] == pytest.approx(expected, rel=1e-9, abs=1e-12)
    assert result.statistics["message_size_bytes"] == payload
    assert result.statistics["direction"] == 0


def test_estimator_zero_bandwidth_raises():
    est = _estimator()
    no_host = types.SimpleNamespace(
        name="NO_HOST",
        host_bandwidth_bytes_ps=0.0,
        host_latency_s=0.0,
        host_transfer_efficiency=1.0,
    )
    kv = torch.randn(4, 128, dtype=torch.float32)
    info = types.SimpleNamespace(args=(kv, 0))
    with pytest.raises(ValueError):
        est(info, no_host)


def test_apply_decode_kv_offload_accounting_off_is_noop():
    from tensor_cast.core.model_runner import ModelRunner as TCMRunner
    from tensor_cast.core.user_config import UserInputConfig

    runner = TCMRunner.__new__(TCMRunner)
    runner.user_input = UserInputConfig(device="TEST_DEVICE", model_id="x")  # offload OFF
    runner.device_profile = TEST_DEVICE

    kv = int(2 * 1024**3)
    idx = int(0.1 * 1024**3)
    peak, kv_size, mode, host_pool, transfer = runner._apply_decode_kv_offload_accounting(
        10.0, kv, idx, 124928, 4, True
    )
    assert mode == "none"
    assert transfer == 0.0 and host_pool == 0.0
    assert peak == 10.0
    assert kv_size == pytest.approx((kv + idx) / 1024**3)


def test_apply_decode_kv_offload_accounting_on_decode():
    from types import SimpleNamespace

    from tensor_cast.core.kv_offload import (
        decode_hot_buffer_bytes,
        decode_offload_miss_transfer_time_ms,
    )
    from tensor_cast.core.model_runner import ModelRunner as TCMRunner
    from tensor_cast.core.user_config import UserInputConfig

    ui = UserInputConfig(
        device="TEST_DEVICE",
        model_id="x",
        enable_decode_kv_offload=True,
        decode_offload_buffer_multiplier=2.0,
        decode_offload_hit_rate=0.85,
    )
    runner = TCMRunner.__new__(TCMRunner)
    runner.user_input = ui
    runner.device_profile = TEST_DEVICE
    runner.model = SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace(index_topk=2048)))

    kv = int(2 * 1024**3)
    idx = int(0.1 * 1024**3)
    mkpt = 124928
    peak_in = 10.0
    batch = 4
    peak, kv_size, mode, host_pool, transfer = runner._apply_decode_kv_offload_accounting(
        peak_in, kv, idx, mkpt, batch, True
    )
    assert mode == "decode"
    hot = decode_hot_buffer_bytes(mkpt, topk=2048, buffer_multiplier=2.0, num_seqs=batch)
    assert peak == pytest.approx(peak_in - kv / 1024**3 + hot / 1024**3)
    assert kv_size == pytest.approx((idx + hot) / 1024**3)
    assert host_pool == pytest.approx(kv / 1024**3)
    expected_transfer = decode_offload_miss_transfer_time_ms(
        mkpt,
        topk=2048,
        hit_rate=0.85,
        num_seqs=batch,
        host_bandwidth_bytes_ps=TEST_DEVICE.host_bandwidth_bytes_ps,
        host_latency_s=TEST_DEVICE.host_latency_s,
        host_efficiency=TEST_DEVICE.host_transfer_efficiency,
    )
    assert transfer == pytest.approx(expected_transfer, rel=1e-9)
    assert transfer > 0


def test_apply_decode_kv_offload_accounting_on_but_prefill_skipped():
    from types import SimpleNamespace

    from tensor_cast.core.model_runner import ModelRunner as TCMRunner
    from tensor_cast.core.user_config import UserInputConfig

    ui = UserInputConfig(
        device="TEST_DEVICE",
        model_id="x",
        enable_decode_kv_offload=True,
    )
    runner = TCMRunner.__new__(TCMRunner)
    runner.user_input = ui
    runner.device_profile = TEST_DEVICE
    runner.model = SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace(index_topk=2048)))

    peak, kv_size, mode, host_pool, transfer = runner._apply_decode_kv_offload_accounting(
        10.0, int(2 * 1024**3), 0, 124928, 4, False
    )
    assert mode == "none" and transfer == 0.0 and host_pool == 0.0
    assert peak == 10.0


def test_apply_prefill_kv_offload_accounting_off_is_noop():
    from tensor_cast.core.model_runner import ModelRunner as TCMRunner
    from tensor_cast.core.user_config import UserInputConfig

    runner = TCMRunner.__new__(TCMRunner)
    runner.user_input = UserInputConfig(device="TEST_DEVICE", model_id="x")  # prefill offload OFF
    runner.device_profile = TEST_DEVICE

    kv = int(2 * 1024**3)
    idx = int(0.1 * 1024**3)
    peak, kv_size, mode, host_pool, transfer = runner._apply_prefill_kv_offload_accounting(
        10.0, kv, idx, 124928, 4, False, 80
    )
    assert mode == "none" and transfer == 0.0 and host_pool == 0.0
    assert peak == 10.0
    assert kv_size == pytest.approx((kv + idx) / 1024**3)


def test_apply_prefill_kv_offload_accounting_on_prefill():
    from tensor_cast.core.kv_offload import prefill_layerwise_footprint_ratio
    from tensor_cast.core.model_runner import ModelRunner as TCMRunner
    from tensor_cast.core.user_config import UserInputConfig

    ui = UserInputConfig(
        device="TEST_DEVICE",
        model_id="x",
        enable_prefill_kv_offload=True,
    )
    runner = TCMRunner.__new__(TCMRunner)
    runner.user_input = ui
    runner.device_profile = TEST_DEVICE

    kv = int(2 * 1024**3)
    idx = int(0.1 * 1024**3)
    n_layers = 80
    peak_in = 10.0
    peak, kv_size, mode, host_pool, transfer = runner._apply_prefill_kv_offload_accounting(
        peak_in, kv, idx, 124928, 4, False, n_layers
    )
    assert mode == "prefill"
    ratio = prefill_layerwise_footprint_ratio(n_layers, 0, 2)
    kv_gb = kv / 1024**3
    idx_gb = idx / 1024**3
    assert peak == pytest.approx(peak_in - kv_gb * (1 - ratio))
    assert kv_size == pytest.approx(ratio * kv_gb + idx_gb)
    assert host_pool == pytest.approx(kv_gb * (1 - ratio))
    bw = TEST_DEVICE.host_bandwidth_bytes_ps * TEST_DEVICE.host_transfer_efficiency
    expected_transfer = (TEST_DEVICE.host_latency_s + (2 * kv * (1 - ratio)) / bw) * 1000.0
    assert transfer == pytest.approx(expected_transfer, rel=1e-9)
    assert transfer > 0


def test_apply_prefill_kv_offload_accounting_on_but_decode_skipped():
    from tensor_cast.core.model_runner import ModelRunner as TCMRunner
    from tensor_cast.core.user_config import UserInputConfig

    ui = UserInputConfig(
        device="TEST_DEVICE",
        model_id="x",
        enable_prefill_kv_offload=True,
    )
    runner = TCMRunner.__new__(TCMRunner)
    runner.user_input = ui
    runner.device_profile = TEST_DEVICE

    peak, kv_size, mode, host_pool, transfer = runner._apply_prefill_kv_offload_accounting(
        10.0,
        int(2 * 1024**3),
        0,
        124928,
        4,
        True,
        80,  # is_decode=True -> skip
    )
    assert mode == "none" and transfer == 0.0 and host_pool == 0.0
    assert peak == 10.0


def test_atlas_a_series_host_link_configuration():
    from tensor_cast.device import ATLAS_800

    profiles = [
        ATLAS_800.A2_376T_64G,
        ATLAS_800.A2_313T_64G,
        ATLAS_800.A2_280T_64G,
        ATLAS_800.A2_280T_64G_PCIE,
        ATLAS_800.A2_280T_32G_PCIE,
        ATLAS_800.A3_752T_128G_DIE,
        ATLAS_800.A3_560T_128G_DIE,
        ATLAS_800.A3_560T_128G_DIE_ROCE,
    ]
    for profile in profiles:
        if "A3" in profile.name:
            # A3 host link is calibrated to the confirmed die-level spec.
            assert profile.host_bandwidth_bytes_ps == 70e9, f"{profile.name} wrong A3 host bandwidth"
        else:
            assert profile.host_bandwidth_bytes_ps == 32e9, f"{profile.name} wrong A2 host bandwidth"
        assert profile.host_latency_s == 5e-6, f"{profile.name} wrong latency"
        assert profile.host_transfer_efficiency == 0.8, f"{profile.name} wrong efficiency"
    # A3_560T_128G_DIE_ROCE is built via dataclasses.replace and must inherit the
    # host link from A3_560T_128G_DIE.
    assert ATLAS_800.A3_560T_128G_DIE_ROCE.host_bandwidth_bytes_ps == ATLAS_800.A3_560T_128G_DIE.host_bandwidth_bytes_ps
    # The confirmed A3 specification is 140 GB/s per direction for a dual-die
    # card, so the die-level theoretical profile is 70 GB/s per direction.
    assert ATLAS_800.A3_752T_128G_DIE.host_bandwidth_bytes_ps == 70e9
    assert ATLAS_800.A3_560T_128G_DIE.host_bandwidth_bytes_ps == 70e9


def test_a5_host_link_configuration():
    from tensor_cast.device import A5, DeviceProfile

    profiles = [v for v in vars(A5).values() if isinstance(v, DeviceProfile)]
    assert profiles, "no A5 DeviceProfile instances found"
    for profile in profiles:
        assert profile.host_bandwidth_bytes_ps == 112e9, f"{profile.name} wrong A5 host bandwidth"
        assert profile.host_latency_s == 5e-6, f"{profile.name} wrong latency"
        assert profile.host_transfer_efficiency == 0.8, f"{profile.name} wrong efficiency"


def test_model_runner_metrics_offload_fields_default_inactive():
    import dataclasses

    from tensor_cast.core.model_runner import ModelRunnerMetrics

    names = {f.name for f in dataclasses.fields(ModelRunnerMetrics)}
    assert {"offload_mode", "host_kv_pool_gb", "offload_transfer_time_ms"} <= names

    # Construct with only the pre-existing required fields; the offload fields
    # must default to the inactive state so non-offload runs are unaffected.
    metrics = ModelRunnerMetrics(
        total_device_memory_gb=0.0,
        model_weight_size_gb=0.0,
        peak_memory_usage_gb=0.0,
        kv_cache_size_gb=0.0,
        kv_cache_per_token_gb=0.0,
        model_activation_size_gb=0.0,
        reserved_memory_gb=0.0,
        device_memory_available_gb=0.0,
        execution_time_s={},
        tps_per_model={},
        run_time_s=0.0,
        batch_size=0,
    )
    assert metrics.offload_mode == "none"
    assert metrics.host_kv_pool_gb == 0.0
    assert metrics.offload_transfer_time_ms == 0.0
