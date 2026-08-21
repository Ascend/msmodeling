"""Regression: DSpark MarkovHead vanilla / gated / rnn sequential modeling."""

import unittest
from collections import Counter

import torch
from transformers import Qwen3Config

from tensor_cast.compilation import get_backend
from tensor_cast.device import TEST_DEVICE
from tensor_cast.layers.dspark import (
    DsparkDraftModel,
    DsparkWrapper,
    MarkovHead,
    apply_cli_overrides_to_dspark_config,
)
from tensor_cast.model_config import DsparkConfig
from tensor_cast.performance_model.analytic import AnalyticPerformanceModel
from tensor_cast.runtime import Runtime


def _tiny_draft_hf(vocab: int = 32, hidden: int = 64) -> Qwen3Config:
    cfg = Qwen3Config(
        vocab_size=vocab,
        hidden_size=hidden,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
    )
    cfg._attn_implementation = "tensor_cast"
    return cfg


def _build_wrapper(head_type: str, *, confidence: bool = False) -> DsparkWrapper:
    scfg = DsparkConfig(
        dspark_block_size=4,
        num_draft_layers=1,
        markov_rank=8,
        markov_head_type=head_type,
        enable_confidence_head=confidence,
        confidence_head_with_markov=confidence,
    )
    apply_cli_overrides_to_dspark_config(scfg, cli_block_size=4, cli_num_draft_layers=1)
    dcfg = scfg.to_dflash_config()
    draft_hf = _tiny_draft_hf()
    draft = DsparkDraftModel(draft_hf, dcfg, scfg, layer_idx_offset=0)

    class _DummyInner(torch.nn.Module):
        def forward(self, *args, **kwargs):
            return torch.zeros(1, 4, 32)

    return DsparkWrapper(scfg, dcfg, draft_hf, _DummyInner(), draft, draft_hf)


class TestDsparkMarkovHeadVariants(unittest.TestCase):
    def test_vanilla_forward_returns_no_state(self):
        head = MarkovHead(32, 32, 8, head_type="vanilla")
        emb, bias, state = head(torch.zeros(2, dtype=torch.long))
        self.assertEqual(tuple(emb.shape), (2, 8))
        self.assertEqual(tuple(bias.shape), (2, 32))
        self.assertIsNone(state)
        self.assertIsNone(head.gate_proj)
        self.assertIsNone(head.joint_proj)

    def test_gated_constructs_gate_proj_and_needs_hidden(self):
        head = MarkovHead(32, 32, 8, hidden_size=64, head_type="gated")
        self.assertIsNotNone(head.gate_proj)
        self.assertEqual(head.gate_proj.in_features, 64 + 8)
        self.assertEqual(head.gate_proj.out_features, 8)
        with self.assertRaises(ValueError):
            head(torch.zeros(2, dtype=torch.long))
        emb, bias, state = head(torch.zeros(2, dtype=torch.long), torch.zeros(2, 64))
        self.assertEqual(tuple(bias.shape), (2, 32))
        self.assertIsNone(state)
        self.assertEqual(tuple(emb.shape), (2, 8))

    def test_rnn_updates_state_across_steps(self):
        head = MarkovHead(32, 32, 8, hidden_size=64, head_type="rnn")
        self.assertIsNotNone(head.joint_proj)
        self.assertEqual(head.joint_proj.in_features, 2 * 8 + 64)
        self.assertEqual(head.joint_proj.out_features, 3 * 8)
        prev = torch.zeros(2, dtype=torch.long)
        hidden = torch.zeros(2, 64)
        emb0, bias0, state0 = head(prev, hidden, None)
        self.assertIsNotNone(state0)
        self.assertEqual(tuple(state0.shape), (2, 8))
        emb1, bias1, state1 = head(prev, hidden, state0)
        self.assertEqual(tuple(bias0.shape), (2, 32))
        self.assertEqual(tuple(bias1.shape), (2, 32))
        self.assertEqual(tuple(emb0.shape), (2, 8))
        self.assertEqual(tuple(emb1.shape), (2, 8))
        # Second step consumes previous state (outputs may differ after init).
        self.assertEqual(tuple(state1.shape), (2, 8))

    def test_gated_sequential_op_counts(self):
        wrapper = _build_wrapper("gated")
        self.assertEqual(wrapper.draft.markov_head.head_type, "gated")
        batch, block = 2, 4

        class _SeqOnly(torch.nn.Module):
            def __init__(self, w):
                super().__init__()
                self.w = w

            def forward(self, hidden, logits, tokens):
                return self.w._propose_draft_tokens(hidden, logits, batch, block, tokens)

        torch._dynamo.reset()
        mod = torch.compile(_SeqOnly(wrapper).half(), backend=get_backend(), fullgraph=False)
        with Runtime(AnalyticPerformanceModel(TEST_DEVICE), TEST_DEVICE) as runtime:
            with torch.no_grad():
                out = mod(
                    torch.zeros(batch, block, 64, dtype=torch.half),
                    torch.zeros(batch, block, 32, dtype=torch.half),
                    torch.zeros(batch, block, dtype=torch.long),
                )
        self.assertEqual(tuple(out[0].shape), (batch, block))
        counts = Counter(str(e.op_invoke_info.func) for e in runtime.event_list)
        # N embed + N gate_proj (addmm, bias=True) + N markov_bias (mm, bias=False)
        self.assertEqual(counts.get("aten.embedding.default", 0), block)
        self.assertEqual(counts.get("aten.mm.default", 0), block)
        self.assertEqual(counts.get("aten.addmm.default", 0), block)

    def test_rnn_sequential_op_counts(self):
        wrapper = _build_wrapper("rnn")
        self.assertEqual(wrapper.draft.markov_head.head_type, "rnn")
        batch, block = 2, 4

        class _SeqOnly(torch.nn.Module):
            def __init__(self, w):
                super().__init__()
                self.w = w

            def forward(self, hidden, logits, tokens):
                return self.w._propose_draft_tokens(hidden, logits, batch, block, tokens)

        torch._dynamo.reset()
        mod = torch.compile(_SeqOnly(wrapper).half(), backend=get_backend(), fullgraph=False)
        with Runtime(AnalyticPerformanceModel(TEST_DEVICE), TEST_DEVICE) as runtime:
            with torch.no_grad():
                out = mod(
                    torch.zeros(batch, block, 64, dtype=torch.half),
                    torch.zeros(batch, block, 32, dtype=torch.half),
                    torch.zeros(batch, block, dtype=torch.long),
                )
        self.assertEqual(tuple(out[0].shape), (batch, block))
        counts = Counter(str(e.op_invoke_info.func) for e in runtime.event_list)
        self.assertEqual(counts.get("aten.embedding.default", 0), block)
        # N joint_proj (addmm) + N markov_bias (mm)
        self.assertEqual(counts.get("aten.mm.default", 0), block)
        self.assertEqual(counts.get("aten.addmm.default", 0), block)

    def test_draft_model_builds_gated_and_rnn(self):
        for head_type in ("gated", "rnn"):
            scfg = DsparkConfig(
                dspark_block_size=4,
                num_draft_layers=1,
                markov_rank=8,
                markov_head_type=head_type,
                enable_confidence_head=False,
            )
            apply_cli_overrides_to_dspark_config(scfg, cli_block_size=4, cli_num_draft_layers=1)
            draft = DsparkDraftModel(_tiny_draft_hf(), scfg.to_dflash_config(), scfg)
            self.assertEqual(draft.markov_head.head_type, head_type)
            if head_type == "gated":
                self.assertIsNotNone(draft.markov_head.gate_proj)
            else:
                self.assertIsNotNone(draft.markov_head.joint_proj)


if __name__ == "__main__":
    unittest.main()
