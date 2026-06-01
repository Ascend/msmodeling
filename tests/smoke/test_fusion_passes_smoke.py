"""Smoke guards for GMM / DFC / GMM-fusion nightly regressions.

Uses remote config.json only (meta tensors). Set ``MSMODELING_OFFLINE=1`` to skip offline.

Nightly coverage mapping
------------------------
test_gmm_pass_grouped_matmul_smoke   -> GmmPassTestCase.test_qwen3_fp (Qwen/Qwen3-235B-A22B)
test_gmm_pass_vl_moe_smoke           -> GmmPassTestCase.test_qwen3_fp (Qwen/Qwen3-VL-30B-A3B-Instruct)
test_gmm_fusion_ep_smoke             -> TestTextGenerateNightly.test_gmm_fusion
test_dfc_dispatch_ffn_combine_smoke  -> DfcPassNightlyTestCase.test_dfc_dsv3_ep
test_vl_moe_tp_ep_compile_smoke      -> TestTextGenerateNightly.test_vl_moe_tp_ep_different_parallel
"""

from __future__ import annotations

from dataclasses import asdict

import tensor_cast.config as tc_config
from tensor_cast.core.input_generator import generate_inputs
from tensor_cast.core.model_runner import ModelRunner, ModelRunnerMetrics
from tensor_cast.core.quantization.datatypes import QuantizeLinearAction
from tensor_cast.core.user_config import UserInputConfig


def test_gmm_pass_grouped_matmul_smoke():
    """MoE compile path with grouped_matmul; guards GmmPassTestCase.test_qwen3_fp on 235B."""
    user_input = UserInputConfig(
        model_id="Qwen/Qwen3-235B-A22B",
        num_queries=1,
        query_len=32,
        context_length=0,
        do_compile=True,
        num_hidden_layers_override=1,
        quantize_linear_action=QuantizeLinearAction.DISABLED,
    )
    result = ModelRunner(user_input).run_inference(generate_inputs_func=generate_inputs)
    assert result is not None
    if isinstance(result, ModelRunnerMetrics):
        result = asdict(result)
    assert "tensor_cast.grouped_matmul" in result["table_result"]


def test_gmm_pass_vl_moe_smoke():
    """VL MoE compile path with grouped_matmul; guards GmmPassTestCase.test_qwen3_fp on VL-30B."""
    user_input = UserInputConfig(
        model_id="Qwen/Qwen3-VL-30B-A3B-Instruct",
        num_queries=1,
        query_len=8,
        context_length=0,
        do_compile=True,
        num_hidden_layers_override=1,
        image_batch_size=1,
        image_height=224,
        image_width=224,
        quantize_linear_action=QuantizeLinearAction.DISABLED,
    )
    runner = ModelRunner(user_input)
    assert runner.model.is_vl_model
    result = runner.run_inference(generate_inputs_func=generate_inputs)
    assert result is not None
    if isinstance(result, ModelRunnerMetrics):
        result = asdict(result)
    assert "tensor_cast.grouped_matmul" in result["table_result"]


def test_gmm_fusion_ep_smoke():
    """EP+compile GMM fusion; guards TestTextGenerateNightly.test_gmm_fusion."""
    user_input = UserInputConfig(
        model_id="Qwen/Qwen3-235B-A22B",
        num_queries=1,
        query_len=1,
        context_length=32,
        do_compile=True,
        num_hidden_layers_override=1,
        quantize_linear_action=QuantizeLinearAction.DISABLED,
        world_size=2,
        ep_size=2,
        moe_dp_size=1,
        moe_tp_size=1,
        tp_size=1,
    )
    result = ModelRunner(user_input).run_inference(generate_inputs_func=generate_inputs)
    assert result is not None
    if isinstance(result, ModelRunnerMetrics):
        result = asdict(result)
    assert "tensor_cast.grouped_matmul" in result["table_result"]


def test_dfc_dispatch_ffn_combine_smoke():
    """Single DFC prefill scenario; guards DfcPassNightlyTestCase.test_dfc_dsv3_ep."""
    orig = tc_config.compilation.fusion_patterns.enable_dispatch_ffn_combine
    tc_config.compilation.fusion_patterns.enable_dispatch_ffn_combine = True
    try:
        user_input = UserInputConfig(
            model_id="deepseek-ai/DeepSeek-V3",
            num_queries=1,
            query_len=32,
            context_length=0,
            do_compile=True,
            allow_graph_break=True,
            num_hidden_layers_override=4,
            world_size=2,
            tp_size=2,
            ep_size=2,
            quantize_linear_action=QuantizeLinearAction.W8A8_STATIC,
        )
        result = ModelRunner(user_input).run_inference(generate_inputs_func=generate_inputs)
        assert result is not None
        if isinstance(result, ModelRunnerMetrics):
            result = asdict(result)
        assert "tensor_cast.dispatch_ffn_combine" in result["table_result"]
    finally:
        tc_config.compilation.fusion_patterns.enable_dispatch_ffn_combine = orig


def test_vl_moe_tp_ep_compile_smoke():
    """VL MoE TP+EP compile; guards TestTextGenerateNightly.test_vl_moe_tp_ep_different_parallel."""
    user_input = UserInputConfig(
        model_id="Qwen/Qwen3-VL-30B-A3B-Instruct",
        num_queries=1,
        query_len=8,
        image_batch_size=1,
        image_height=224,
        image_width=224,
        do_compile=True,
        num_hidden_layers_override=1,
        quantize_linear_action=QuantizeLinearAction.W8A8_DYNAMIC,
        world_size=2,
        tp_size=2,
        ep_size=2,
        moe_dp_size=1,
        moe_tp_size=1,
    )
    runner = ModelRunner(user_input)
    assert runner.model.is_vl_model
    input_kwargs = generate_inputs(
        runner.model,
        runner.request_info_default,
        block_size=runner.user_input.block_size,
    )
    assert "pixel_values" in input_kwargs
    result = runner.run_inference(generate_inputs_func=generate_inputs)
    assert result is not None
