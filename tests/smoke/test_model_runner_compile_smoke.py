"""Smoke guard for ModelRunner compile + inference nightly regressions.

Uses local tiny model configs so PR CI stays fast. Nightly tests add full
model IDs, specific table-result op assertions, and multi-shape sweeps.

Nightly coverage mapping
------------------------
test_model_runner_compile_deepseek       -> TestTextGenerateNightly (test_with_compilation,
                                            test_with_compilation_and_graph_break)
                                            PerfAnalysisNightlyTestCase (test_model / test_deepseek)
test_model_runner_compile_quant_deepseek -> TestTextGenerateNightly (test_qwen2_5_with_compile)
                                            MatmulAllReducePassTestCase (proxy)
                                            TestDeepseekV32ModelNightly (proxy)
                                            TestQuantLinearNightly (proxy)
"""

from tensor_cast.core.input_generator import generate_inputs
from tensor_cast.core.model_runner import ModelRunner, ModelRunnerMetrics
from tensor_cast.core.quantization.datatypes import QuantizeLinearAction
from tensor_cast.core.user_config import UserInputConfig

_DATA_DIR = "tests/assets/model_config"


def test_model_runner_compile_deepseek():
    """ModelRunner compile + inference round-trip; guards text-generate / perf-analysis nightly."""
    user_config = UserInputConfig(
        model_id=f"{_DATA_DIR}/deepseek_new",
        num_queries=1,
        query_len=32,
        context_length=0,
        do_compile=True,
        num_hidden_layers_override=1,
        quantize_linear_action=QuantizeLinearAction.DISABLED,
    )
    runner = ModelRunner(user_config)
    result = runner.run_inference(generate_inputs_func=generate_inputs)
    assert result is not None
    if isinstance(result, ModelRunnerMetrics):
        assert result.table_result is not None


def test_model_runner_compile_quant_deepseek():
    """ModelRunner W8A8-dynamic compile + inference; guards compile+quant nightly regressions."""
    user_config = UserInputConfig(
        model_id=f"{_DATA_DIR}/deepseek_new",
        num_queries=1,
        query_len=32,
        context_length=0,
        do_compile=True,
        num_hidden_layers_override=1,
        quantize_linear_action=QuantizeLinearAction.W8A8_DYNAMIC,
    )
    runner = ModelRunner(user_config)
    result = runner.run_inference(generate_inputs_func=generate_inputs)
    assert result is not None
    if isinstance(result, ModelRunnerMetrics):
        assert result.table_result is not None
