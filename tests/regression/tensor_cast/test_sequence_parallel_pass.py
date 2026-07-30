"""Test for sequence parallel pass.

Transforms: all_reduce → rms_norm  =>  reduce_scatter → rms_norm(local) → all_gather
This matches the expected sequence parallel communication rewrite on NPU.
"""

import operator
import unittest
from dataclasses import asdict

import pytest
import torch
from parameterized import parameterized
from torch.fx import Graph
from tensor_cast import config
from tensor_cast.compilation.passes.sequence_parallel_pass import (
    MoeLocalTokenRewriter,
    Pattern2Rewriter,
    Pattern3Rewriter,
    _is_moe_p3_tail,
)
from tensor_cast.core.input_generator import generate_inputs
from tensor_cast.core.model_runner import ModelRunner, ModelRunnerMetrics
from tensor_cast.core.quantization.datatypes import QuantizeLinearAction
from tensor_cast.core.user_config import UserInputConfig
from tensor_cast.model_config import WordEmbeddingTPMode
from tests.helpers.cli_runner import run_module_main


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
            word_embedding_tp=WordEmbeddingTPMode.row,
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


class SequenceParallelPassRegressionTestCase(unittest.TestCase):
    """Regression tests for sequence parallel pass that must run in CI gate."""

    def test_local_descendants_rewrites_only_sequence_dimension(self):
        graph = Graph()
        local = graph.placeholder("local")
        sequence_view = graph.call_function(torch.ops.aten.view.default, (local, [1, 16384, -1]))
        sequence_view.meta["val"] = torch.empty((1, 16384, 16384), device="meta")
        flattened_view = graph.call_function(torch.ops.aten.view.default, (sequence_view, [-1, 16384]))
        # Preserve the stale rank-3 metadata seen during the compiler rewrite:
        # it must not make the 2-D target shape's hidden dimension look like
        # the sequence dimension.
        flattened_view.meta["val"] = torch.empty((1, 16384, 16384), device="meta")
        graph.output(flattened_view)

        MoeLocalTokenRewriter._mark_local_descendants(
            sequence_view,
            full_tokens=16384,
            local_tokens=1024,
        )

        self.assertEqual(sequence_view.args[1], [1, 1024, -1])
        self.assertEqual(flattened_view.args[1], [-1, 16384])
        self.assertTrue(flattened_view.meta["tensor_cast_sp_local"])

    @parameterized.expand(
        [
            # Regression case from run_sc.sh: Qwen3-32B + sequence parallel +
            # row word-embedding TP with large input length and batch size.
            # Batch size must be divisible by TP size to avoid
            # ``AssertionError: X is not divisible by Y`` in reduce_scatter.
            (8, 16),
            (16, 16),
        ]
    )
    def test_sp_throughput_optimizer_row_embedding_tp(self, tp_size: int, batch_size: int):
        """throughput_optimizer entry point works for SP + row embedding TP.

        Mirrors the ``run_sc.sh`` aggregation case via
        ``cli.inference.throughput_optimizer``:
        - model: Qwen/Qwen3-32B
        - input length: 4096
        - output length: 1
        - batch size: 16 (aligned to TP size)
        - TP size: 8 or 16
        - row word-embedding TP
        - quantize linear: DISABLED
        - compile enabled
        """
        args = [
            "Qwen/Qwen3-32B",
            "--device=ATLAS_800_A3_752T_128G_DIE",
            "--num-devices=16",
            "--input-length=4096",
            "--output-length=1",
            "--compile",
            "--tp-sizes",
            str(tp_size),
            "--batch-range",
            str(batch_size),
            str(batch_size),
            "--compilation-config",
            "enable_sequence_parallel",
            "enable_multistream",
            "--word-embedding-tp=row",
            "--quantize-linear-action=DISABLED",
            "--ttft-limits=2000",
        ]

        result = run_module_main("cli.inference.throughput_optimizer", args)
        full_output = result.stdout + result.stderr

        self.assertEqual(
            result.returncode,
            0,
            f"throughput_optimizer failed for TP={tp_size}, batch={batch_size}: {result.stderr}",
        )
        self.assertIn(
            "Overall Best Configuration:",
            full_output,
            "Optimizer should produce an overall best configuration",
        )

    def test_moe_sp_rewrite_keeps_gate_experts_and_output_local(self):
        """Fold the generic TP full/local wrapper around an expanded MoE graph."""
        graph = Graph()
        local = graph.placeholder("local")
        local.meta["tensor_cast_sp_local"] = True
        gate_weight = graph.placeholder("gate_weight")
        shared_weight = graph.placeholder("shared_weight")
        rank_group = [0, 1]

        gathered = graph.call_function(torch.ops.tensor_cast.all_gather.default, (local, 0, 0, rank_group))
        full_view = graph.call_function(torch.ops.aten.view.default, (gathered, [-1, 16]))
        gate_logits = graph.call_function(torch.ops.aten.mm.default, (full_view, gate_weight))
        logits_slice = graph.call_function(torch.ops.aten.slice.Tensor, (gate_logits, 0, 0, 4))
        topk = graph.call_function(torch.ops.tensor_cast.moe_gating_top_k_softmax.default, (logits_slice, 2))
        topk_indices = graph.call_function(operator.getitem, (topk, 1))

        hidden_slice = graph.call_function(torch.ops.aten.slice.Tensor, (full_view, 0, 0, 4))
        routed = graph.call_function(torch.ops.tensor_cast.init_routing_v2.default, (hidden_slice, topk_indices))
        unpermute = graph.call_function(torch.ops.tensor_cast.unpermute_tokens.default, (routed, topk_indices))
        routed_local = graph.call_function(torch.ops.aten.sum.dim_IntList, (unpermute, [-2]))
        exit_gather = graph.call_function(torch.ops.tensor_cast.all_gather.default, (routed_local, 0, 0, rank_group))
        shared = graph.call_function(torch.ops.aten.mm.default, (full_view, shared_weight))
        output = graph.call_function(torch.ops.aten.add.Tensor, (exit_gather, shared))
        graph.output(output)

        self.assertEqual(MoeLocalTokenRewriter().apply(graph), 1)
        self.assertIs(full_view.args[0], local)
        self.assertIs(topk.args[0], gate_logits)
        self.assertIs(routed.args[0], full_view)
        self.assertIs(output.args[0], routed_local)
        self.assertTrue(output.meta["tensor_cast_sp_local"])

    def test_moe_p2_preserves_shared_expert_all_reduce_on_local_tokens(self):
        """MoE P2 keeps a feature all-reduce when its input is already token-local."""
        graph = Graph()
        shared_local = graph.placeholder("shared_local")
        shared_local.meta["tensor_cast_sp_local"] = True
        residual_local = graph.placeholder("residual_local")
        residual_local.meta["tensor_cast_sp_local"] = True
        weight = graph.placeholder("weight")
        router_logits = graph.placeholder("router_logits")
        graph.call_function(torch.ops.tensor_cast.moe_gating_top_k_softmax.default, (router_logits, 2))
        shared_reduce = graph.call_function(torch.ops.tensor_cast.all_reduce.default, (shared_local, 0, [0, 1]))
        norm2 = graph.call_function(
            torch.ops.tensor_cast.add_rms_norm2.default, (shared_reduce, residual_local, weight, 1e-5)
        )
        graph.output(norm2)

        self.assertEqual(Pattern2Rewriter().apply(graph), 1)
        self.assertIs(norm2.args[0], shared_reduce)
        self.assertFalse(any(node.target is torch.ops.tensor_cast.reduce_scatter.default for node in graph.nodes))
        self.assertTrue(norm2.meta["tensor_cast_sp_local"])

    def test_moe_p3_crosses_adjacent_repetition_region_boundary(self):
        """MoE P3 keeps residual local across region_end/copies/region_begin."""
        graph = Graph()
        local_hidden = graph.placeholder("local_hidden")
        local_residual = graph.placeholder("local_residual")
        projection = graph.placeholder("projection")
        projection.meta["val"] = torch.empty(1, 4096, 16)
        weight = graph.placeholder("weight")
        router_logits = graph.placeholder("router_logits")
        graph.call_function(torch.ops.tensor_cast.moe_gating_top_k_softmax.default, (router_logits, 2))
        rank_group = [0, 1]
        norm2 = graph.call_function(
            torch.ops.tensor_cast.add_rms_norm2.default,
            (local_hidden, local_residual, weight, 1e-5),
        )
        norm2.meta["tensor_cast_sp_local"] = True
        residual = graph.call_function(operator.getitem, (norm2, 1))
        reduced = graph.call_function(torch.ops.tensor_cast.all_reduce.default, (projection, 0, rank_group))
        reduced.meta["val"] = torch.empty(1, 4096, 16)
        added = graph.call_function(torch.ops.aten.add.Tensor, (residual, reduced))
        region_end = graph.call_function(torch.ops.tensor_cast._internal_mark_region_end.default, (added, 1))
        copied = graph.call_function(torch.ops.tensor_cast._internal_copy_region.default, (region_end, 1))
        region_begin = graph.call_function(torch.ops.tensor_cast._internal_mark_region_begin.default, (copied, 2))
        graph.call_function(torch.ops.aten.clone.default, (region_begin,))
        norm = graph.call_function(torch.ops.tensor_cast.rms_norm.default, (region_begin, weight, 1e-5))
        graph.output(norm)

        self.assertTrue(_is_moe_p3_tail(residual))
        self.assertEqual(Pattern3Rewriter().apply(graph), 1)
        self.assertTrue(added.meta["tensor_cast_sp_local"])
        self.assertTrue(any(node.target is torch.ops.tensor_cast.reduce_scatter.default for node in graph.nodes))
        self.assertTrue(any(node.target is torch.ops.tensor_cast.all_gather.default for node in graph.nodes))


if __name__ == "__main__":
    # PYTHONPATH=/pathto/msmodeling:$PYTHONPATH pytest -v \
    #   tests/regression/tensor_cast/test_sequence_parallel_pass.py \
    #   --log-cli-level=DEBUG > test.log
    unittest.main()
