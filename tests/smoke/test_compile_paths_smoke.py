"""Smoke guard for nightly compile-path regressions.

Each case exercises one compile variant with a local tiny model config so PR CI
stays fast. Nightly tests add full model IDs, op-level assertions, and
multi-shape sweeps.

Nightly coverage mapping
------------------------
test_compile_w8a8_dynamic_quant_deepseek  -> TestQuantLinearNightly
                                             ParallelLinearNightlyTestCase (quant paths)
test_compile_with_mtp_tokens_deepseek     -> MtpNightlyTestCase
                                             MtpEpNightlyTestCase
test_compile_with_tp_parallel_deepseek    -> ParallelLinearNightlyTestCase
                                             MatmulAllReducePassTestCase
                                             SequenceParallelPassTestCase
"""

from tensor_cast.core.model_builder import build_model
from tensor_cast.core.quantization.datatypes import QuantizeLinearAction
from tensor_cast.core.user_config import UserInputConfig


_DATA_DIR = "tests/assets/model_config"


def test_compile_w8a8_dynamic_quant_deepseek():
    """W8A8-dynamic quant compile path; guards quantized-linear nightly regressions."""
    user_config = UserInputConfig(
        model_id=f"{_DATA_DIR}/deepseek_new",
        do_compile=True,
        num_hidden_layers_override=1,
        quantize_linear_action=QuantizeLinearAction.W8A8_DYNAMIC,
    )
    model = build_model(user_config)
    assert model is not None


def test_compile_with_mtp_tokens_deepseek():
    """MTP-token compile path; guards MTP / MTP-EP nightly regressions."""
    user_config = UserInputConfig(
        model_id=f"{_DATA_DIR}/deepseek_new",
        do_compile=True,
        num_hidden_layers_override=1,
        num_mtp_tokens=1,
    )
    model = build_model(user_config)
    assert model is not None


def test_compile_with_tp_parallel_deepseek():
    """TP-parallel compile path; guards parallel-linear / matmul-allreduce / SP nightly regressions."""
    user_config = UserInputConfig(
        model_id=f"{_DATA_DIR}/deepseek_new",
        do_compile=True,
        num_hidden_layers_override=1,
        world_size=2,
        tp_size=2,
    )
    model = build_model(user_config)
    assert model is not None
