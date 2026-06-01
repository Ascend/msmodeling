"""Test for sequence parallel pass.

Transforms: all_reduce → rms_norm  =>  reduce_scatter → rms_norm(local) → all_gather
This matches the expected sequence parallel communication rewrite on NPU.
"""

import unittest
from dataclasses import asdict

import pytest
import torch
from parameterized import parameterized
from tensor_cast import config
from tensor_cast.core.input_generator import generate_inputs
from tensor_cast.core.model_runner import ModelRunner, ModelRunnerMetrics
from tensor_cast.core.quantization.datatypes import QuantizeLinearAction
from tensor_cast.core.user_config import UserInputConfig
from tensor_cast.model_config import WordEmbeddingTPMode


@pytest.mark.nightly
class SequenceParallelPassTestCase(unittest.TestCase):
    """Test sequence parallel pass transforms all_reduce+norm patterns."""

    def setUp(self):
        torch.compiler.reset()
        self._orig_enable_sequence_parallel = config.compilation.passes.enable_sequence_parallel

    def tearDown(self):
        config.compilation.passes.enable_sequence_parallel = self._orig_enable_sequence_parallel

    @parameterized.expand(
        [
            # (tp_size, expected_local_seq, disable_repetition)
            # disable_repetition=False: layers carry `_internal_mark_region_*`
            # markers (marker-aware SP path).
            # disable_repetition=True: each layer is instantiated separately and
            # the markers are absent, so the SP pass must match the markerless
            # pattern — this was the failing case in PR #175.
            (2, 64, False),
            (2, 64, True),
        ]
    )
    def test_sp_reduces_rms_norm_seq_dim(self, tp_size: int, expected_local_seq: int, disable_repetition: bool):
        """Verify rms_norm operates on reduced seq length with sequence parallel enabled."""
        config.compilation.passes.enable_sequence_parallel = True
        user_input = UserInputConfig(
            model_id="Qwen/Qwen3-32B",
            num_queries=1,
            query_len=128,
            context_length=0,
            do_compile=True,
            dump_input_shapes=True,
            enable_sequence_parallel=True,
            disable_repetition=disable_repetition,
            num_mtp_tokens=0,
            num_hidden_layers_override=1,
            world_size=tp_size,
            tp_size=tp_size,
            word_embedding_tp=True,
            word_embedding_tp_mode=WordEmbeddingTPMode.row.value,
            quantize_linear_action=QuantizeLinearAction.DISABLED,
        )

        model_runner = ModelRunner(user_input)
        result = model_runner.run_inference(generate_inputs_func=generate_inputs)
        if isinstance(result, ModelRunnerMetrics):
            result = asdict(result)

        table = result["table_result"]

        # Verify rms_norm is present
        self.assertIn("tensor_cast.rms_norm.default", table)
        self.assertIn(
            f"[1, {expected_local_seq}, 5120], [5120]",
            table,
            "Sequence parallel should shard the entry rms_norm sequence dimension",
        )

        # Verify sequence parallel pattern presence
        if tp_size > 1:
            # With sequence parallel: should have reduce_scatter and all_gather
            self.assertIn(
                "tensor_cast.reduce_scatter.default",
                table,
                "Sequence parallel mode should have reduce_scatter",
            )
            self.assertIn(
                "tensor_cast.all_gather.default",
                table,
                "Sequence parallel mode should have all_gather",
            )
            # Should NOT have all_reduce (replaced by sequence parallel pattern)
            self.assertNotIn(
                "tensor_cast.all_reduce.default",
                table,
                "Sequence parallel mode should replace all_reduce",
            )
        else:
            # Without sequence parallel: should have all_reduce
            self.assertIn(
                "tensor_cast.all_reduce.default",
                table,
                "Non-sequence-parallel mode should have all_reduce",
            )


if __name__ == "__main__":
    # PYTHONPATH=/pathto/msmodeling:$PYTHONPATH pytest -v \
    #   tests/regression/tensor_cast/test_sequence_parallel_pass.py \
    #   --log-cli-level=DEBUG > test.log
    unittest.main()
