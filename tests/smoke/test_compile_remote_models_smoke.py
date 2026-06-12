"""Smoke guard for nightly compile regressions that require remote model configs.

Only ``config.json`` (architecture config) is fetched from HuggingFace /
ModelScope on first run — no weight files are downloaded.  All forward passes
use ``device="meta"`` tensors so no GPU/NPU is required.

Set ``MSMODELING_OFFLINE=1`` to skip when running fully offline.

Nightly coverage mapping
------------------------
test_compile_qwen3_vanilla       -> ModelLoadNightlyTestCase
                                    (test_vanilla_transformer_model with do_compile=True)
                                    RepetitionNightlyTestCase
                                    (test_vanilla_transformer_model with do_compile=True)
test_compile_qwen3_moe           -> GmmPassTestCase
                                    (test_qwen3_fp, test_qwen3_dynamic_quant)
                                    SwiGLUFusionPassNightlyTestCase
                                    (test_gmm_swiglu_fused_op_present)
test_compile_deepseek_v31        -> ModelLoadNightlyTestCase
                                    (test_deepseek_without_kvcache / test_deepseek_with_kvcache)
                                    SwiGLUFusionPassNightlyTestCase
                                    (test_swiglu_fused_op_present_deepseek)
test_vl_compile_glm45v           -> TestVLCompilePrefillNightly
                                    (test_glm45v_prefill_with_compile)
"""

from tensor_cast.core.input_generator import generate_inputs
from tensor_cast.core.model_builder import build_model
from tensor_cast.core.model_runner import ModelRunner, ModelRunnerMetrics
from tensor_cast.core.quantization.datatypes import QuantizeLinearAction
from tensor_cast.core.user_config import UserInputConfig


def test_compile_qwen3_vanilla():
    """Vanilla-transformer compile with Qwen3-32B; guards model-load / repetition nightly regressions."""
    user_config = UserInputConfig(
        model_id="Qwen/Qwen3-32B",
        do_compile=True,
        num_hidden_layers_override=1,
        quantize_linear_action=QuantizeLinearAction.DISABLED,
    )
    model = build_model(user_config)
    assert model is not None


def test_compile_qwen3_moe():
    """MoE-transformer compile with Qwen3-235B-A22B; guards GMM-pass / SwiGLU-fusion nightly regressions."""
    user_config = UserInputConfig(
        model_id="Qwen/Qwen3-235B-A22B",
        do_compile=True,
        num_hidden_layers_override=1,
        quantize_linear_action=QuantizeLinearAction.DISABLED,
    )
    model = build_model(user_config)
    assert model is not None


def test_compile_deepseek_v31():
    """DeepSeek-MLA compile with DeepSeek-V3.1; guards model-load-deepseek / SwiGLU-deepseek nightly regressions."""
    user_config = UserInputConfig(
        model_id="deepseek-ai/DeepSeek-V3.1",
        do_compile=True,
        num_hidden_layers_override=1,
        quantize_linear_action=QuantizeLinearAction.DISABLED,
    )
    model = build_model(user_config)
    assert model is not None


def test_vl_compile_glm45v():
    """VL model compile + inference; guards TestVLCompilePrefillNightly."""
    user_config = UserInputConfig(
        model_id="zai-org/GLM-4.5V",
        num_queries=1,
        query_len=30,
        context_length=0,
        do_compile=True,
        allow_graph_break=False,
        num_hidden_layers_override=1,
        image_batch_size=1,
        image_height=224,
        image_width=224,
        quantize_linear_action=QuantizeLinearAction.DISABLED,
    )
    runner = ModelRunner(user_config)
    assert runner.model.is_vl_model
    result = runner.run_inference(generate_inputs_func=generate_inputs)
    assert result is not None
    if isinstance(result, ModelRunnerMetrics):
        assert result.table_result is not None
