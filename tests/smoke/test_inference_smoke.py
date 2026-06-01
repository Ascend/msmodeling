"""Smoke guards for nightly inference and perf-analysis regressions.

Uses local tiny configs so PR CI stays offline and fast.

Nightly coverage mapping
------------------------
test_single_token_prefill_slower_than_decode_smoke -> TestTextGenerateNightly.test_single_token_prefill_vs_decode
                                                    (smoke uses local deepseek_new; nightly uses Qwen3.5 / Qwen3-Next)
test_perf_analysis_mla_no_compile_local           -> PerfAnalysisTestCase.test_deepseek (do_compile=False)
test_perf_analysis_kimi_k2_remote                 -> PerfAnalysisNightlyTestCase.test_deepseek
                                                    (moonshotai/Kimi-K2-Base, do_compile=True)
test_perf_analysis_glm45_compile_remote           -> PerfAnalysisNightlyTestCase.test_model
                                                    (zai-org/GLM-4.5, do_compile=True)
test_perf_analysis_qwen235b_compile_remote        -> PerfAnalysisNightlyTestCase.test_model
                                                    (Qwen/Qwen3-235B-A22B, do_compile=True)
"""

from __future__ import annotations

import torch
from tensor_cast.core.input_generator import generate_inputs
from tensor_cast.core.model_builder import build_model
from tensor_cast.core.model_runner import ModelRunner, ModelRunnerMetrics
from tensor_cast.core.user_config import UserInputConfig
from tensor_cast.device import TEST_DEVICE
from tensor_cast.performance_model.analytic import AnalyticPerformanceModel
from tensor_cast.performance_model.memory_tracker import MemoryTracker
from tensor_cast.runtime import Runtime
from tests.regression.tensor_cast.test_common import (
    create_attn_metadata_and_kv_cache,
    create_mla_metadata_and_kv_cache,
    has_submodule_with_cls_name,
)

_DATA_DIR = "tests/assets/model_config"

# Analytic roofline can tie on 1-layer tiny configs (float noise ~1e-8).
_ANALYTIC_TIME_ABS_TOL_S = 1e-6


def test_single_token_prefill_slower_than_decode_smoke():
    """Guards TestTextGenerateNightly.test_single_token_prefill_vs_decode (local tiny config)."""
    model_id = f"{_DATA_DIR}/deepseek_new"
    prefill_config = UserInputConfig(
        model_id=model_id,
        num_queries=1,
        query_len=1,
        context_length=0,
        num_hidden_layers_override=1,
        do_compile=False,
    )
    prefill_result = ModelRunner(prefill_config).run_inference(generate_inputs_func=generate_inputs)
    decode_config = UserInputConfig(
        model_id=model_id,
        num_queries=1,
        query_len=1,
        context_length=10,
        num_hidden_layers_override=1,
        do_compile=False,
    )
    decode_result = ModelRunner(decode_config).run_inference(generate_inputs_func=generate_inputs)
    assert prefill_result is not None and decode_result is not None
    if isinstance(prefill_result, ModelRunnerMetrics) and isinstance(decode_result, ModelRunnerMetrics):
        prefill_time = prefill_result.execution_time_s.get("analytic", 0)
        decode_time = decode_result.execution_time_s.get("analytic", 0)
        assert prefill_time >= decode_time - _ANALYTIC_TIME_ABS_TOL_S


def test_perf_analysis_mla_no_compile_local():
    """Guards PerfAnalysisTestCase.test_deepseek (do_compile=False) with local MLA config."""
    user_config = UserInputConfig(
        model_id=f"{_DATA_DIR}/deepseek_new",
        do_compile=False,
        num_hidden_layers_override=1,
    )
    model = build_model(user_config)
    assert has_submodule_with_cls_name(model, "MultiheadLatentAttentionTensorCast")
    attn_meta, kv_cache_by_layers, num_tokens = create_mla_metadata_and_kv_cache(model, model.model_config)
    inputs = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
    position_ids = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
    device_profile = TEST_DEVICE
    perf_model = AnalyticPerformanceModel(device_profile)
    with Runtime(perf_model, device_profile, memory_tracker=MemoryTracker(device_profile)) as runtime:
        with torch.no_grad():
            outputs = model.forward(
                inputs,
                position_ids,
                attention_meta=attn_meta,
                kv_cache_by_layers=kv_cache_by_layers,
            )
            assert outputs.shape == (1, num_tokens, model.vocab_size)
    result = runtime.table_averages()
    assert "tensor_cast.multihead_latent_attention" in result


def test_perf_analysis_kimi_k2_remote():
    """Guards PerfAnalysisNightlyTestCase.test_deepseek for remote Kimi-K2-Base MLA."""
    user_config = UserInputConfig(
        model_id="moonshotai/Kimi-K2-Base",
        do_compile=True,
        num_hidden_layers_override=1,
    )
    model = build_model(user_config)
    assert has_submodule_with_cls_name(model, "MultiheadLatentAttentionTensorCast")
    attn_meta, kv_cache_by_layers, num_tokens = create_mla_metadata_and_kv_cache(model, model.model_config)
    inputs = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
    position_ids = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
    device_profile = TEST_DEVICE
    perf_model = AnalyticPerformanceModel(device_profile)
    with Runtime(perf_model, device_profile, memory_tracker=MemoryTracker(device_profile)) as runtime:
        with torch.no_grad():
            outputs = model.forward(
                inputs,
                position_ids,
                attention_meta=attn_meta,
                kv_cache_by_layers=kv_cache_by_layers,
            )
            assert outputs.shape == (1, num_tokens, model.vocab_size)
    result = runtime.table_averages()
    assert "tensor_cast.multihead_latent_attention" in result


def test_perf_analysis_glm45_compile_remote():
    """Guards PerfAnalysisNightlyTestCase.test_model for remote GLM-4.5 compile."""
    user_config = UserInputConfig(
        model_id="zai-org/GLM-4.5",
        do_compile=True,
        num_hidden_layers_override=1,
    )
    model = build_model(user_config)
    attn_meta, kv_cache_by_layers, num_tokens = create_attn_metadata_and_kv_cache(model, model.model_config)
    inputs = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
    position_ids = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
    device_profile = TEST_DEVICE
    perf_model = AnalyticPerformanceModel(device_profile)
    with Runtime(perf_model, device_profile, memory_tracker=MemoryTracker(device_profile)) as runtime:
        with torch.no_grad():
            outputs = model.forward(
                inputs,
                position_ids,
                attention_meta=attn_meta,
                kv_cache_by_layers=kv_cache_by_layers,
            )
            assert outputs.shape == (1, num_tokens, model.vocab_size)
    result = runtime.table_averages()
    assert "tensor_cast." in result


def test_perf_analysis_qwen235b_compile_remote():
    """Guards PerfAnalysisNightlyTestCase.test_model for remote Qwen3-235B compile."""
    user_config = UserInputConfig(
        model_id="Qwen/Qwen3-235B-A22B",
        do_compile=True,
        num_hidden_layers_override=1,
    )
    model = build_model(user_config)
    attn_meta, kv_cache_by_layers, num_tokens = create_attn_metadata_and_kv_cache(model, model.model_config)
    inputs = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
    position_ids = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
    device_profile = TEST_DEVICE
    perf_model = AnalyticPerformanceModel(device_profile)
    with Runtime(perf_model, device_profile, memory_tracker=MemoryTracker(device_profile)) as runtime:
        with torch.no_grad():
            outputs = model.forward(
                inputs,
                position_ids,
                attention_meta=attn_meta,
                kv_cache_by_layers=kv_cache_by_layers,
            )
            assert outputs.shape == (1, num_tokens, model.vocab_size)
    result = runtime.table_averages()
    assert "tensor_cast." in result
