"""Regression: stacked confidence must survive torch.compile (anti-DCE)."""

import unittest
from collections import Counter

import torch
from transformers import Qwen3Config

from tensor_cast.compilation import get_backend
from tensor_cast.device import TEST_DEVICE
from tensor_cast.layers.dspark import (
    DsparkDraftModel,
    DsparkWrapper,
    apply_cli_overrides_to_dspark_config,
)
from tensor_cast.model_config import DsparkConfig
from tensor_cast.performance_model.analytic import AnalyticPerformanceModel
from tensor_cast.runtime import Runtime


class TestDsparkConfidenceResidency(unittest.TestCase):
    def test_stacked_confidence_returned_and_recorded(self):
        scfg = DsparkConfig(
            dspark_block_size=4,
            num_draft_layers=1,
            markov_rank=8,
            enable_confidence_head=True,
            confidence_head_with_markov=True,
        )
        apply_cli_overrides_to_dspark_config(scfg, cli_block_size=4, cli_num_draft_layers=1)
        dcfg = scfg.to_dflash_config()
        draft_hf = Qwen3Config(
            vocab_size=32,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            max_position_embeddings=128,
        )
        draft_hf._attn_implementation = "tensor_cast"
        draft = DsparkDraftModel(draft_hf, dcfg, scfg, layer_idx_offset=0)

        class _DummyInner(torch.nn.Module):
            def forward(self, *args, **kwargs):
                return torch.zeros(1, 4, 32)

        wrapper = DsparkWrapper(scfg, dcfg, draft_hf, _DummyInner(), draft, draft_hf)
        batch, block = 2, 4
        draft_hidden = torch.zeros(batch, block, 64)
        base_logits = torch.zeros(batch, block, 32)
        next_tokens = torch.zeros(batch, block, dtype=torch.long)

        draft_tokens, verify_tokens, confidence = wrapper._propose_draft_tokens(
            draft_hidden, base_logits, batch, block, next_tokens
        )
        self.assertEqual(tuple(draft_tokens.shape), (batch, block))
        self.assertIs(verify_tokens, next_tokens)
        self.assertEqual(tuple(confidence.shape), (batch, block))

        class _SeqOnly(torch.nn.Module):
            def __init__(self, w):
                super().__init__()
                self.w = w

            def forward(self, hidden, logits, tokens):
                # Mirror Decode side-output shape used by the wrapper.
                return self.w._propose_draft_tokens(hidden, logits, 2, 4, tokens)

        torch._dynamo.reset()
        mod = torch.compile(_SeqOnly(wrapper).half(), backend=get_backend(), fullgraph=False)
        with Runtime(AnalyticPerformanceModel(TEST_DEVICE), TEST_DEVICE) as runtime:
            with torch.no_grad():
                out = mod(
                    torch.zeros(2, 4, 64, dtype=torch.half),
                    torch.zeros(2, 4, 32, dtype=torch.half),
                    torch.zeros(2, 4, dtype=torch.long),
                )
        self.assertEqual(len(out), 3)
        self.assertEqual(tuple(out[2].shape), (2, 4))
        counts = Counter(str(e.op_invoke_info.func) for e in runtime.event_list)
        # N embeddings (Markov) + N markov_bias mm + N confidence mm
        self.assertEqual(counts.get("aten.embedding.default", 0), 4)
        self.assertGreaterEqual(counts.get("aten.mm.default", 0), 8)

    def test_markov_bias_tp_local_vocab_matches_lm_head(self):
        """markov_bias col-shards to V/TP with gather_output like lm_head."""
        from tensor_cast.layers.parallel_linear import ColumnParallelLinear
        from tensor_cast.parallel_group import ParallelGroup

        scfg = DsparkConfig(
            dspark_block_size=4,
            num_draft_layers=1,
            markov_rank=8,
            enable_confidence_head=False,
        )
        apply_cli_overrides_to_dspark_config(scfg, cli_block_size=4, cli_num_draft_layers=1)
        dcfg = scfg.to_dflash_config()
        vocab, hidden, tp_size = 64, 32, 4
        draft_hf = Qwen3Config(
            vocab_size=vocab,
            hidden_size=hidden,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=128,
        )
        draft_hf._attn_implementation = "tensor_cast"
        draft = DsparkDraftModel(draft_hf, dcfg, scfg, layer_idx_offset=0)
        self.assertIsNotNone(draft.markov_head)
        self.assertEqual(draft.markov_head.markov_bias.out_features, vocab)

        tp = ParallelGroup(rank=0, rank_groups=[list(range(tp_size))], global_world_size=tp_size)
        params = {"tp_group": tp, "global_tp_group": tp, "gather_output": True}
        lm_head = ColumnParallelLinear(torch.nn.Linear(hidden, vocab, bias=False), **params)
        markov_bias = ColumnParallelLinear(draft.markov_head.markov_bias, **params)
        self.assertEqual(markov_bias.out_features_per_partition, vocab // tp_size)
        self.assertEqual(markov_bias.out_features_per_partition, lm_head.out_features_per_partition)
        self.assertTrue(markov_bias.gather_output)

    def test_align_markov_bias_slices_full_vocab_to_local_logits(self):
        """When lm_head emits V/TP, full-V Markov bias is narrowed to the same shard."""
        from tensor_cast.layers.dspark import DsparkWrapper

        logits = torch.zeros(2, 16)
        bias = torch.arange(64, dtype=torch.float32).view(1, 64).expand(2, -1)
        lm_head = torch.nn.Linear(8, 16, bias=False)
        lm_head.tp_rank = 1  # type: ignore[attr-defined]
        aligned = DsparkWrapper._align_markov_bias_vocab(logits, bias, lm_head)
        self.assertEqual(tuple(aligned.shape), (2, 16))
        self.assertEqual(aligned[0].tolist(), list(range(16, 32)))

    def test_unify_markov_tp_disables_gather_when_lm_head_is_local(self):
        """Inner/local lm_head ⇒ markov_bias.gather_output=False (unified local V/TP)."""
        from tensor_cast.layers.parallel_linear import ColumnParallelLinear
        from tensor_cast.parallel_group import ParallelGroup
        from tensor_cast.transformers.transformations import _unify_dspark_markov_tp_with_lm_head

        scfg = DsparkConfig(
            dspark_block_size=4,
            num_draft_layers=1,
            markov_rank=8,
            enable_confidence_head=False,
        )
        apply_cli_overrides_to_dspark_config(scfg, cli_block_size=4, cli_num_draft_layers=1)
        dcfg = scfg.to_dflash_config()
        vocab, hidden, tp_size = 64, 32, 4
        draft_hf = Qwen3Config(
            vocab_size=vocab,
            hidden_size=hidden,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=128,
        )
        draft_hf._attn_implementation = "tensor_cast"
        draft = DsparkDraftModel(draft_hf, dcfg, scfg, layer_idx_offset=0)
        local_lm = torch.nn.Linear(hidden, vocab // tp_size, bias=False)
        draft.set_shared(torch.nn.Embedding(vocab, hidden), local_lm)
        tp = ParallelGroup(rank=0, rank_groups=[list(range(tp_size))], global_world_size=tp_size)
        draft.markov_head.markov_bias = ColumnParallelLinear(
            draft.markov_head.markov_bias,
            tp_group=tp,
            global_tp_group=tp,
            gather_output=True,
        )

        class _Target(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_tokens = draft.embed_tokens
                self.lm_head = local_lm

        wrapper = DsparkWrapper(scfg, dcfg, draft_hf, _Target(), draft, draft_hf)

        class _Model:
            _inner = wrapper
            parallel_group_manager = type("PGM", (), {"tp_group": tp, "lmhead_tp_group": tp})()

        _unify_dspark_markov_tp_with_lm_head(_Model())
        self.assertFalse(draft.markov_head.markov_bias.gather_output)


class TestDsparkAcceptanceClamp(unittest.TestCase):
    def test_apply_cli_overrides_clamps_acceptance_to_unit_interval(self):
        scfg = DsparkConfig(dspark_block_size=8, num_draft_layers=1, dspark_acceptance_length=5.0)
        # Bypass __post_init__: negative acceptance must be clamped at re-resolve.
        scfg.dspark_acceptance_length = -1.0
        apply_cli_overrides_to_dspark_config(scfg, cli_block_size=8, cli_num_draft_layers=1)
        self.assertEqual(scfg.dspark_acceptance_length, 0.0)

        scfg.dspark_acceptance_length = 99.0
        apply_cli_overrides_to_dspark_config(scfg, cli_block_size=8, cli_num_draft_layers=1)
        self.assertEqual(scfg.dspark_acceptance_length, 8.0)


if __name__ == "__main__":
    unittest.main()
