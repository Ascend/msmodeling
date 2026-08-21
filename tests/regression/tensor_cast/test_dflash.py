# Copyright (c) Huawei Technologies Co., Ltd. All rights reserved.
"""Unit tests for unified Dflash modeling."""

import unittest

import torch

from tensor_cast.layers.attention import AttentionMetadataTensorCast, AttentionTensorCast
from tensor_cast.layers.dflash import (
    DflashDraftModel,
    DflashWrapper,
    apply_cli_overrides_to_source_and_dcfg,
    build_draft_attention_metadata,
    build_draft_hf_config,
    resolve_l_ctx,
    resolve_target_embed_and_lm_head,
)
from tensor_cast.layers.dflash_qwen3 import Qwen3DFlashAttention, Qwen3DFlashDecoderLayer
from tensor_cast.model_config import DflashConfig


class TestDflashConfig(unittest.TestCase):
    def test_acceptance_clamp(self):
        cfg = DflashConfig(dflash_block_size=8, dflash_acceptance_length=10)
        self.assertEqual(cfg.dflash_acceptance_length, 7.0)

    def test_requires_block_size(self):
        with self.assertRaises(ValueError):
            DflashConfig(dflash_block_size=1)


class TestResolveLCtx(unittest.TestCase):
    def test_full_attention_uses_context(self):
        self.assertEqual(resolve_l_ctx(1024, "full_attention", 2048), 1024)

    def test_sliding_uses_min(self):
        self.assertEqual(resolve_l_ctx(4096, "sliding_attention", 2048), 2048)
        self.assertEqual(resolve_l_ctx(512, "sliding_attention", 2048), 512)


class TestBuildDraftAttentionMetadata(unittest.TestCase):
    def test_meta_device_aligns_with_input_generator_fields(self):
        """Draft metadata must construct on meta without .item() and disable DCP."""
        batch, ctx_len, block = 4, 16, 3
        context_meta, noise_meta = build_draft_attention_metadata(
            batch,
            ctx_len,
            block,
            device=torch.device("meta"),
        )
        for meta in (context_meta, noise_meta):
            self.assertIsInstance(meta, AttentionMetadataTensorCast)
            self.assertEqual(meta.seq_lens_values, [ctx_len + block] * batch)
            self.assertEqual(meta.query_lens_values, [block] * batch)
            self.assertEqual(meta.is_decode_values, [True] * batch)
            self.assertEqual(meta.max_total_seq_len, ctx_len + block)
            self.assertIs(meta.is_dcp_decode, False)


class TestBuiltinDraftConfig(unittest.TestCase):
    def test_cli_override_block_and_layers(self):
        dcfg = DflashConfig(dflash_block_size=8, num_draft_layers=6, dflash_acceptance_length=5)
        apply_cli_overrides_to_source_and_dcfg(dcfg, cli_block_size=4, cli_num_draft_layers=2)
        self.assertEqual(dcfg.dflash_block_size, 4)
        self.assertEqual(dcfg.num_draft_layers, 2)
        self.assertEqual(dcfg.dflash_acceptance_length, 3.0)
        self.assertTrue(dcfg.aux_hidden_state_layer_ids)

    def test_prefer_existing_does_not_clobber_cli_block_size(self):
        """build_dflash_draft_and_wrapper re-applies config; must keep CLI block_size."""
        dcfg = DflashConfig(dflash_block_size=8, num_draft_layers=6, dflash_acceptance_length=5)
        apply_cli_overrides_to_source_and_dcfg(dcfg, cli_block_size=16, cli_num_draft_layers=2)
        self.assertEqual(dcfg.dflash_block_size, 16)
        self.assertEqual(dcfg.num_draft_layers, 2)
        # Second apply without CLI (as in build_*) must not reload builtin block_size=8.
        apply_cli_overrides_to_source_and_dcfg(dcfg, prefer_existing=True)
        self.assertEqual(dcfg.dflash_block_size, 16)
        self.assertEqual(dcfg.num_draft_layers, 2)

    def test_build_draft_hf_merges_target_dims(self):
        dcfg = DflashConfig(dflash_block_size=8, num_draft_layers=2, context_length=128)
        apply_cli_overrides_to_source_and_dcfg(dcfg, cli_num_draft_layers=2)
        draft_hf = build_draft_hf_config(
            dcfg, target_hidden_size=7168, target_vocab_size=163840, target_max_position_embeddings=4096
        )
        self.assertEqual(draft_hf.hidden_size, 7168)
        self.assertEqual(draft_hf.vocab_size, 163840)
        self.assertEqual(draft_hf.num_hidden_layers, 2)
        self.assertEqual(len(draft_hf.layer_types), 2)
        self.assertEqual(draft_hf._attn_implementation, "tensor_cast")


class TestDflashSlidingWindow(unittest.TestCase):
    def test_layer_types_drive_sliding_window_with_global_offset(self):
        """layer_types use draft-local index; attention_by_layers uses global offset."""
        dcfg = DflashConfig(
            dflash_block_size=4,
            num_draft_layers=2,
            context_length=4096,
            sliding_window=2048,
            layer_types=["sliding_attention", "full_attention"],
        )
        dcfg.aux_hidden_state_layer_ids = [0, 1]
        draft_hf = build_draft_hf_config(dcfg, target_hidden_size=64, target_vocab_size=128)
        draft_hf.num_attention_heads = 4
        draft_hf.num_key_value_heads = 2
        draft_hf.head_dim = 16
        draft_hf.hidden_size = 64
        draft_hf.intermediate_size = 128
        draft_hf.vocab_size = 128
        draft_hf.layer_types = ["sliding_attention", "full_attention"]
        draft_hf.sliding_window = 2048
        dcfg.layer_types = list(draft_hf.layer_types)
        dcfg.sliding_window = 2048

        offset = 61  # typical target depth; must not use offset % N for layer_types
        draft = DflashDraftModel(draft_hf, dcfg, layer_idx_offset=offset)
        self.assertEqual(draft.sliding_window_indices, [0])
        sw_attn = draft.layers[0].dflash_block.self_attn
        full_attn = draft.layers[1].dflash_block.self_attn
        self.assertEqual(sw_attn.layer_idx, offset)
        self.assertEqual(full_attn.layer_idx, offset + 1)
        self.assertEqual(sw_attn.sliding_window, 2048)
        self.assertIsNone(full_attn.sliding_window)
        self.assertEqual(sw_attn.layer_type, "sliding_attention")
        self.assertEqual(full_attn.layer_type, "full_attention")
        self.assertEqual(resolve_l_ctx(4096, draft.layer_types[0], draft.sliding_window), 2048)
        self.assertEqual(resolve_l_ctx(4096, draft.layer_types[1], draft.sliding_window), 4096)

    def test_sw_layer_uses_capped_kv_len(self):
        from tensor_cast.device import TEST_DEVICE
        from tensor_cast.performance_model.analytic import AnalyticPerformanceModel
        from tensor_cast.runtime import Runtime

        dcfg = DflashConfig(
            dflash_block_size=4,
            num_draft_layers=1,
            context_length=4096,
            sliding_window=32,
            layer_types=["sliding_attention"],
        )
        dcfg.aux_hidden_state_layer_ids = [0]
        draft_hf = build_draft_hf_config(dcfg, target_hidden_size=64, target_vocab_size=128)
        draft_hf.num_attention_heads = 4
        draft_hf.num_key_value_heads = 2
        draft_hf.head_dim = 16
        draft_hf.hidden_size = 64
        draft_hf.intermediate_size = 128
        draft_hf.vocab_size = 128
        draft_hf.layer_types = ["sliding_attention"]
        draft_hf.sliding_window = 32
        dcfg.layer_types = ["sliding_attention"]
        dcfg.sliding_window = 32

        draft = DflashDraftModel(draft_hf, dcfg, layer_idx_offset=0)
        draft.set_shared(torch.nn.Embedding(128, 64), torch.nn.Linear(64, 128, bias=False))
        attention_by_layers = {0: AttentionTensorCast()}
        noise = torch.randn(1, 4, 64)
        position_ids = torch.arange(4).view(1, 4)
        # fc length = max L_ctx = 32; SW layer should slice to 32 (already max).
        target_context = torch.randn(1, draft.max_l_ctx(), 64)
        device = TEST_DEVICE
        with Runtime(AnalyticPerformanceModel(device), device) as runtime:
            with torch.no_grad():
                _ = draft(
                    noise,
                    target_context,
                    position_ids,
                    attention_by_layers=attention_by_layers,
                )
        self.assertEqual(draft.max_l_ctx(), 32)
        rope_events = [e for e in runtime.event_list if "tensor_cast.apply_rope" in str(e.op_invoke_info.func)]
        cache_events = [e for e in runtime.event_list if "tensor_cast.reshape_and_cache" in str(e.op_invoke_info.func)]
        attn_events = [e for e in runtime.event_list if "tensor_cast.attention" in str(e.op_invoke_info.func)]
        self.assertGreaterEqual(len(rope_events), 2)  # 1 fused ctx rope + 1 noise rope
        self.assertGreaterEqual(len(cache_events), 2)  # ctx write + noise write
        self.assertEqual(len(attn_events), 1)
        # Fused context rope seq = L_ctx; noise rope seq = block.
        rope_seqs = []
        for e in rope_events:
            q = e.op_invoke_info.args[0]
            if isinstance(q, torch.Tensor) and q.ndim == 4:
                rope_seqs.append(int(q.shape[2]))  # BHSD
        self.assertIn(32, rope_seqs)  # context L_ctx
        self.assertIn(4, rope_seqs)  # noise block
        query = attn_events[0].op_invoke_info.args[0]
        self.assertEqual(query.shape[0], 4)  # block tokens
        # Cached K is paged: [num_blocks, page_size, kv_heads, head_dim]
        key = attn_events[0].op_invoke_info.args[1]
        self.assertEqual(key.shape[-2:], (2, 16))


class TestDflashKvInjectTcAttention(unittest.TestCase):
    def test_attention_emits_tensor_cast_op(self):
        dcfg = DflashConfig(dflash_block_size=4, num_draft_layers=1, context_length=8)
        apply_cli_overrides_to_source_and_dcfg(dcfg, cli_num_draft_layers=1)
        draft_hf = build_draft_hf_config(dcfg, target_hidden_size=64, target_vocab_size=128)
        # Shrink heads for a tiny unit test.
        draft_hf.num_attention_heads = 4
        draft_hf.num_key_value_heads = 2
        draft_hf.head_dim = 16
        draft_hf.hidden_size = 64
        draft_hf.intermediate_size = 128
        draft_hf.vocab_size = 128

        attn = Qwen3DFlashAttention(draft_hf, layer_idx=0)
        attention_by_layers = {0: AttentionTensorCast()}
        noise = torch.randn(1, 4, 64)
        target_hidden = torch.randn(1, 8, 64)
        # Pre-RoPE'd context K/V required for no-cache fallback (BSHD).
        num_kv_heads, head_dim = draft_hf.num_key_value_heads, draft_hf.head_dim
        context_kv = (
            torch.randn(1, 8, num_kv_heads, head_dim),
            torch.randn(1, 8, num_kv_heads, head_dim),
        )
        # Dummy cos/sin covering noise window.
        cos = torch.randn(1, 4, 16)
        sin = torch.randn(1, 4, 16)
        out, _ = attn(
            hidden_states=noise,
            target_hidden=target_hidden,
            position_embeddings=(cos, sin),
            context_kv=context_kv,
            attention_by_layers=attention_by_layers,
        )
        self.assertEqual(out.shape, (1, 4, 64))
        with self.assertRaises(ValueError):
            attn(
                hidden_states=noise,
                target_hidden=target_hidden,
                position_embeddings=(cos, sin),
                attention_by_layers=attention_by_layers,
            )

    def test_one_shot_context_kv_proj_then_short_noise(self):
        """Draft model: one fused context_kv_proj; layers only short noise; attn KV = L_ctx+block."""
        from tensor_cast.device import TEST_DEVICE
        from tensor_cast.performance_model.analytic import AnalyticPerformanceModel
        from tensor_cast.runtime import Runtime

        dcfg = DflashConfig(dflash_block_size=4, num_draft_layers=2, context_length=32)
        apply_cli_overrides_to_source_and_dcfg(dcfg, cli_num_draft_layers=2)
        dcfg.aux_hidden_state_layer_ids = [0, 1]
        draft_hf = build_draft_hf_config(dcfg, target_hidden_size=64, target_vocab_size=128)
        draft_hf.num_attention_heads = 4
        draft_hf.num_key_value_heads = 2
        draft_hf.head_dim = 16
        draft_hf.hidden_size = 64
        draft_hf.intermediate_size = 128
        draft_hf.vocab_size = 128
        draft = DflashDraftModel(draft_hf, dcfg, layer_idx_offset=0)
        draft.set_shared(torch.nn.Embedding(128, 64), torch.nn.Linear(64, 128, bias=False))
        # head-major: num_kv_heads(2) × N(2) × 2 × head_dim(16) = 128
        self.assertEqual(draft.context_kv_proj.in_features, 64)
        self.assertEqual(draft.context_kv_proj.out_features, 2 * 2 * 2 * 16)

        attention_by_layers = {0: AttentionTensorCast(), 1: AttentionTensorCast()}
        block, ctx_len = 4, draft.max_l_ctx()
        noise = torch.randn(1, block, 64)
        position_ids = torch.arange(block).view(1, block)
        target_context = torch.randn(1, ctx_len, 2 * 64)
        device = TEST_DEVICE
        with Runtime(AnalyticPerformanceModel(device), device) as runtime:
            with torch.no_grad():
                out = draft(
                    noise,
                    target_context,
                    position_ids,
                    attention_by_layers=attention_by_layers,
                )
        self.assertEqual(out.shape, (1, block, 64))

        # One fused context_kv_proj GEMM with leading dim L_ctx (not per-layer full proj × N).
        kv_in, kv_out = 64, 2 * 2 * 2 * 16
        found_ctx_proj = False
        for event in runtime.event_list:
            shapes = [tuple(a.shape) for a in event.op_invoke_info.args if isinstance(a, torch.Tensor)]
            if any(len(s) == 2 and s[0] == ctx_len and s[-1] == kv_in for s in shapes) and any(
                len(s) == 2 and kv_in in s and kv_out in s for s in shapes
            ):
                found_ctx_proj = True
        self.assertTrue(found_ctx_proj, "expected one-shot context_kv_proj over full L_ctx")

        rope_events = [e for e in runtime.event_list if "tensor_cast.apply_rope" in str(e.op_invoke_info.func)]
        cache_events = [e for e in runtime.event_list if "tensor_cast.reshape_and_cache" in str(e.op_invoke_info.func)]
        attn_events = [e for e in runtime.event_list if "tensor_cast.attention" in str(e.op_invoke_info.func)]
        self.assertEqual(len(rope_events), 3)  # 1 fused ctx (2×L_ctx) + 2 noise
        self.assertGreaterEqual(len(cache_events), 4)  # 2 layers × (ctx + noise)
        self.assertEqual(len(attn_events), 2)
        rope_seqs = []
        for e in rope_events:
            q = e.op_invoke_info.args[0]
            if isinstance(q, torch.Tensor) and q.ndim == 4:
                rope_seqs.append(int(q.shape[2]))  # BHSD
        self.assertEqual(rope_seqs.count(ctx_len * 2), 1)  # fused Tile(L_ctx×N)
        self.assertEqual(rope_seqs.count(block), 2)  # per-layer short noise only
        query = attn_events[0].op_invoke_info.args[0]
        self.assertEqual(query.shape[0], block)
        key = attn_events[0].op_invoke_info.args[1]
        self.assertEqual(key.shape[-2:], (2, 16))

    def test_context_kv_proj_matches_k_proj_under_tp(self):
        """TP ColumnParallel on context_kv_proj must match local k_proj width (cat-safe)."""
        from tensor_cast.layers.parallel_linear import ColumnParallelLinear
        from tensor_cast.parallel_group import ParallelGroup

        dcfg = DflashConfig(dflash_block_size=4, num_draft_layers=2, context_length=16)
        apply_cli_overrides_to_source_and_dcfg(dcfg, cli_num_draft_layers=2)
        dcfg.aux_hidden_state_layer_ids = [0, 1]
        draft_hf = build_draft_hf_config(dcfg, target_hidden_size=64, target_vocab_size=128)
        draft_hf.num_attention_heads = 4
        draft_hf.num_key_value_heads = 2
        draft_hf.head_dim = 16
        draft_hf.hidden_size = 64
        draft_hf.intermediate_size = 128
        draft_hf.vocab_size = 128
        draft = DflashDraftModel(draft_hf, dcfg, layer_idx_offset=0)

        tp = ParallelGroup(rank=0, rank_groups=[[0, 1]], global_world_size=2)
        kv_params = {"tp_group": tp, "global_tp_group": tp, "head_num": 2, "is_replicable": True}
        draft.context_kv_proj = ColumnParallelLinear(draft.context_kv_proj, **kv_params)
        layer0 = draft.layers[0].dflash_block.self_attn
        layer0.k_proj = ColumnParallelLinear(layer0.k_proj, **kv_params)
        layer0.v_proj = ColumnParallelLinear(layer0.v_proj, **kv_params)

        target_hidden = torch.randn(1, 16, 64)
        noise = torch.randn(1, 4, 64)
        _, (k_ctx, v_ctx) = draft._split_context_kv(target_hidden)[0]
        k_noise = layer0.k_proj(noise)
        v_noise = layer0.v_proj(noise)
        self.assertEqual(k_ctx.shape[-1], k_noise.shape[-1])
        self.assertEqual(v_ctx.shape[-1], v_noise.shape[-1])
        # Local after TP=2: 1 kv head × head_dim
        self.assertEqual(k_ctx.shape[-1], 16)
        torch.cat([k_ctx, k_noise], dim=1)
        torch.cat([v_ctx, v_noise], dim=1)

    def test_decoder_layer_forward(self):
        dcfg = DflashConfig(dflash_block_size=4, num_draft_layers=1, context_length=8)
        apply_cli_overrides_to_source_and_dcfg(dcfg, cli_num_draft_layers=1)
        draft_hf = build_draft_hf_config(dcfg, target_hidden_size=64, target_vocab_size=128)
        draft_hf.num_attention_heads = 4
        draft_hf.num_key_value_heads = 2
        draft_hf.head_dim = 16
        draft_hf.hidden_size = 64
        draft_hf.intermediate_size = 128
        layer = Qwen3DFlashDecoderLayer(draft_hf, layer_idx=0)
        attention_by_layers = {0: AttentionTensorCast()}
        noise = torch.randn(1, 4, 64)
        target_hidden = torch.randn(1, 8, 64)
        num_kv_heads, head_dim = draft_hf.num_key_value_heads, draft_hf.head_dim
        context_kv = (
            torch.randn(1, 8, num_kv_heads, head_dim),
            torch.randn(1, 8, num_kv_heads, head_dim),
        )
        cos = torch.randn(1, 4, 16)
        sin = torch.randn(1, 4, 16)
        out = layer(
            hidden_states=noise,
            target_hidden=target_hidden,
            position_embeddings=(cos, sin),
            context_kv=context_kv,
            attention_by_layers=attention_by_layers,
        )
        self.assertEqual(out.shape, (1, 4, 64))


class TestDflashFcOnDecodePath(unittest.TestCase):
    def test_fc_laux_hidden_is_recorded(self):
        """draft.fc (L_aux·H → H) must appear in Runtime events for decode draft forward."""
        from tensor_cast.device import TEST_DEVICE
        from tensor_cast.layers.attention import AttentionTensorCast
        from tensor_cast.performance_model.analytic import AnalyticPerformanceModel
        from tensor_cast.runtime import Runtime

        dcfg = DflashConfig(dflash_block_size=4, num_draft_layers=1, context_length=16)
        apply_cli_overrides_to_source_and_dcfg(dcfg, cli_num_draft_layers=1)
        dcfg.aux_hidden_state_layer_ids = [0, 1, 2]
        draft_hf = build_draft_hf_config(dcfg, target_hidden_size=64, target_vocab_size=128)
        draft_hf.num_attention_heads = 4
        draft_hf.num_key_value_heads = 2
        draft_hf.head_dim = 16
        draft_hf.hidden_size = 64
        draft_hf.intermediate_size = 128
        draft_hf.vocab_size = 128
        draft = DflashDraftModel(draft_hf, dcfg, layer_idx_offset=0)
        draft.set_shared(torch.nn.Embedding(128, 64), torch.nn.Linear(64, 128, bias=False))
        self.assertEqual(draft.fc.in_features, 3 * 64)
        self.assertEqual(draft.fc.out_features, 64)

        attention_by_layers = {0: AttentionTensorCast()}
        noise = torch.randn(1, 4, 64)
        position_ids = torch.arange(4).view(1, 4)
        l_ctx = draft.max_l_ctx()
        aux = [torch.randn(1, l_ctx, 64) for _ in draft.target_layer_ids]
        target_context = draft.build_context_features(aux, l_ctx)
        # Match fc in_features: override with explicit L_aux·H features.
        target_context = torch.randn(1, l_ctx, dcfg.num_selected_layers * 64)
        device = TEST_DEVICE
        with Runtime(AnalyticPerformanceModel(device), device) as runtime:
            with torch.no_grad():
                _ = draft(
                    noise,
                    target_context,
                    position_ids,
                    attention_by_layers=attention_by_layers,
                )

        found_fc = False
        in_f, out_f = 3 * 64, 64
        for event in runtime.event_list:
            name = str(event.op_invoke_info.func)
            shapes = [tuple(a.shape) for a in event.op_invoke_info.args if isinstance(a, torch.Tensor)]
            for shape in shapes:
                if (
                    len(shape) == 2
                    and shape[-1] == in_f
                    and any(len(w) == 2 and in_f in w and out_f in w for w in shapes)
                ):
                    found_fc = True
                if len(shape) == 2 and set(shape) == {in_f, out_f}:
                    found_fc = True
            if "mkl_linear" in name or "mm" in name:
                if any(len(s) == 2 and s[-1] == in_f for s in shapes) and any(
                    len(s) == 2 and in_f in s and out_f in s for s in shapes
                ):
                    found_fc = True
        self.assertTrue(found_fc, "expected draft.fc GEMM (L_aux·H→H) in runtime event list")

    def test_build_context_features_emits_cat_not_clone(self):
        """Aux concat from target-layer hiddens must emit tensor_cast.cat, not aten.clone."""
        from tensor_cast.device import TEST_DEVICE
        from tensor_cast.performance_model.analytic import AnalyticPerformanceModel
        from tensor_cast.runtime import Runtime

        dcfg = DflashConfig(dflash_block_size=8, num_draft_layers=6, context_length=16)
        apply_cli_overrides_to_source_and_dcfg(dcfg, cli_num_draft_layers=6)
        dcfg.aux_hidden_state_layer_ids = [0, 1, 2, 3, 4, 5]
        draft_hf = build_draft_hf_config(dcfg, target_hidden_size=64, target_vocab_size=128)
        draft_hf.num_attention_heads = 4
        draft_hf.num_key_value_heads = 2
        draft_hf.head_dim = 16
        draft_hf.hidden_size = 64
        draft_hf.intermediate_size = 128
        draft_hf.vocab_size = 128
        draft = DflashDraftModel(draft_hf, dcfg, layer_idx_offset=0)
        l_ctx = draft.max_l_ctx()
        self.assertEqual(len(draft.target_layer_ids), 6)

        aux = [torch.randn(1, l_ctx, 64) for _ in draft.target_layer_ids]
        device = TEST_DEVICE
        with Runtime(AnalyticPerformanceModel(device), device) as runtime:
            with torch.no_grad():
                ctx = draft.build_context_features(aux, l_ctx)

        self.assertEqual(ctx.shape, (1, l_ctx, 6 * 64))
        names = [str(e.op_invoke_info.func) for e in runtime.event_list]
        self.assertTrue(
            any("tensor_cast.cat" in n for n in names),
            f"expected tensor_cast.cat in events, got: {names}",
        )
        self.assertFalse(
            any("aten.clone" in n for n in names),
            f"build_context_features must not emit aten.clone, got: {names}",
        )

    def test_fc_under_torch_compile_records_events(self):
        """TC CompilerBackend: fc + context_kv_proj must survive fold_meta_constants."""
        from tensor_cast.compilation import get_backend
        from tensor_cast.device import TEST_DEVICE
        from tensor_cast.layers.attention import AttentionTensorCast
        from tensor_cast.performance_model.analytic import AnalyticPerformanceModel
        from tensor_cast.runtime import Runtime

        dcfg = DflashConfig(dflash_block_size=4, num_draft_layers=1, context_length=16)
        apply_cli_overrides_to_source_and_dcfg(dcfg, cli_num_draft_layers=1)
        dcfg.aux_hidden_state_layer_ids = [0, 1, 2]
        draft_hf = build_draft_hf_config(dcfg, target_hidden_size=64, target_vocab_size=128)
        draft_hf.num_attention_heads = 4
        draft_hf.num_key_value_heads = 2
        draft_hf.head_dim = 16
        draft_hf.hidden_size = 64
        draft_hf.intermediate_size = 128
        draft_hf.vocab_size = 128
        draft = DflashDraftModel(draft_hf, dcfg, layer_idx_offset=0)
        draft.set_shared(torch.nn.Embedding(128, 64), torch.nn.Linear(64, 128, bias=False))
        l_ctx = draft.max_l_ctx()

        class _DecodeLike(torch.nn.Module):
            """Graph-connected aux (formal target outs) → build_context_features → draft."""

            def __init__(self, inner: DflashDraftModel):
                super().__init__()
                self.inner = inner
                # Fake target-layer taps on the FX graph (replaces hook side-channel).
                self.taps = torch.nn.ModuleList([torch.nn.Linear(64, 64, bias=False) for _ in inner.target_layer_ids])

            def forward(self, noise, position_ids, seed, **kwargs):
                aux_hiddens = [tap(seed) for tap in self.taps]
                ctx = self.inner.build_context_features(aux_hiddens, self.inner.max_l_ctx())
                return self.inner(noise, ctx, position_ids, **kwargs)

        model = torch.compile(_DecodeLike(draft), backend=get_backend(), fullgraph=False)

        attention_by_layers = {0: AttentionTensorCast()}
        noise = torch.randn(1, 4, 64)
        position_ids = torch.arange(4).view(1, 4)
        seed = torch.randn(1, l_ctx, 64)
        device = TEST_DEVICE
        with Runtime(AnalyticPerformanceModel(device), device) as runtime:
            with torch.no_grad():
                out = model(
                    noise,
                    position_ids,
                    seed,
                    attention_by_layers=attention_by_layers,
                )
        self.assertEqual(out.shape[-1], 64)
        in_f, out_f = 3 * 64, 64
        kv_in, kv_out = 64, 1 * 2 * 2 * 16
        found_full_fc = False
        found_ctx_proj = False
        for event in runtime.event_list:
            shapes = [tuple(a.shape) for a in event.op_invoke_info.args if isinstance(a, torch.Tensor)]
            if any(len(s) == 2 and s[0] == l_ctx and s[-1] == in_f for s in shapes) and any(
                len(s) == 2 and in_f in s and out_f in s for s in shapes
            ):
                found_full_fc = True
            if any(len(s) == 2 and s[0] == l_ctx and s[-1] == kv_in for s in shapes) and any(
                len(s) == 2 and kv_in in s and kv_out in s for s in shapes
            ):
                found_ctx_proj = True
        self.assertTrue(found_full_fc, f"expected full-L_ctx={l_ctx} draft.fc under TC compile")
        self.assertTrue(found_ctx_proj, f"expected full-L_ctx={l_ctx} context_kv_proj under TC compile")

    def test_fc_survives_compile_with_graph_connected_aux(self):
        """Formal aux (in-graph taps) keeps cat/fc without live_anchor *0."""
        from tensor_cast.compilation import get_backend
        from tensor_cast.device import TEST_DEVICE
        from tensor_cast.layers.attention import AttentionTensorCast
        from tensor_cast.performance_model.analytic import AnalyticPerformanceModel
        from tensor_cast.runtime import Runtime

        dcfg = DflashConfig(dflash_block_size=4, num_draft_layers=1, context_length=16)
        apply_cli_overrides_to_source_and_dcfg(dcfg, cli_num_draft_layers=1)
        dcfg.aux_hidden_state_layer_ids = [0, 1, 2]
        draft_hf = build_draft_hf_config(dcfg, target_hidden_size=64, target_vocab_size=128)
        draft_hf.num_attention_heads = 4
        draft_hf.num_key_value_heads = 2
        draft_hf.head_dim = 16
        draft_hf.hidden_size = 64
        draft_hf.intermediate_size = 128
        draft_hf.vocab_size = 128
        draft = DflashDraftModel(draft_hf, dcfg, layer_idx_offset=0)
        draft.set_shared(torch.nn.Embedding(128, 64), torch.nn.Linear(64, 128, bias=False))
        l_ctx = draft.max_l_ctx()

        class _DecodeLike(torch.nn.Module):
            def __init__(self, inner: DflashDraftModel):
                super().__init__()
                self.inner = inner
                self.taps = torch.nn.ModuleList([torch.nn.Linear(64, 64, bias=False) for _ in inner.target_layer_ids])

            def forward(self, noise, position_ids, seed, **kwargs):
                aux_hiddens = [tap(seed) for tap in self.taps]
                ctx = self.inner.build_context_features(aux_hiddens, self.inner.max_l_ctx())
                return self.inner(noise, ctx, position_ids, **kwargs)

        model = torch.compile(_DecodeLike(draft), backend=get_backend(), fullgraph=False)
        attention_by_layers = {0: AttentionTensorCast()}
        noise = torch.randn(1, 4, 64)
        position_ids = torch.arange(4).view(1, 4)
        seed = torch.randn(1, l_ctx, 64)
        device = TEST_DEVICE
        with Runtime(AnalyticPerformanceModel(device), device) as runtime:
            with torch.no_grad():
                _ = model(
                    noise,
                    position_ids,
                    seed,
                    attention_by_layers=attention_by_layers,
                )
        names = [str(e.op_invoke_info.func) for e in runtime.event_list]
        self.assertTrue(
            any("tensor_cast.cat" in n for n in names),
            f"expected tensor_cast.cat under compile with graph aux, got: {names}",
        )
        in_f, out_f = 3 * 64, 64
        found_fc = False
        for event in runtime.event_list:
            shapes = [tuple(a.shape) for a in event.op_invoke_info.args if isinstance(a, torch.Tensor)]
            if any(len(s) == 2 and s[0] == l_ctx and s[-1] == in_f for s in shapes) and any(
                len(s) == 2 and in_f in s and out_f in s for s in shapes
            ):
                found_fc = True
        self.assertTrue(found_fc, "expected draft.fc under compile with graph-connected aux")

    def test_hidden_norm_fuses_under_torch_compile(self):
        """hidden_norm must use tensor_cast.rms_norm, not decomposed aten ops."""
        from tensor_cast.compilation import get_backend
        from tensor_cast.device import TEST_DEVICE
        from tensor_cast.performance_model.analytic import AnalyticPerformanceModel
        from tensor_cast.runtime import Runtime

        dcfg = DflashConfig(dflash_block_size=4, num_draft_layers=1, context_length=8)
        apply_cli_overrides_to_source_and_dcfg(dcfg, cli_num_draft_layers=1)
        dcfg.aux_hidden_state_layer_ids = [0]
        draft_hf = build_draft_hf_config(dcfg, target_hidden_size=64, target_vocab_size=128)
        draft = DflashDraftModel(draft_hf, dcfg, layer_idx_offset=0)

        class _HiddenNormOnly(torch.nn.Module):
            def __init__(self, inner: DflashDraftModel):
                super().__init__()
                self.inner = inner

            def forward(self, target_context):
                return self.inner.hidden_norm(self.inner.fc(target_context))

        model = torch.compile(_HiddenNormOnly(draft).half(), backend=get_backend())
        target_context = torch.randn(1, 8, 64, dtype=torch.float16)
        with Runtime(AnalyticPerformanceModel(TEST_DEVICE), TEST_DEVICE) as runtime:
            with torch.no_grad():
                model(target_context)
        op_names = {str(e.op_invoke_info.func) for e in runtime.event_list}
        self.assertIn("tensor_cast.rms_norm.default", op_names)
        decomposed = {n for n in op_names if "aten.mean" in n or "aten.rsqrt" in n}
        self.assertFalse(decomposed, f"hidden_norm should not decompose to {decomposed}")


class TestDflashDraftInputUsesBlockSize(unittest.TestCase):
    def test_block_ids_are_batch_by_block(self):
        """Draft position/token ids are constructed as [B, block], not padded packed."""
        batch, block = 2, 16
        position_ids = DflashWrapper._block_position_ids(batch, block, device=torch.device("cpu"), dtype=torch.long)
        tokens = DflashWrapper._draft_token_ids(batch, block, device=torch.device("cpu"), dtype=torch.long)
        self.assertEqual(tuple(position_ids.shape), (batch, block))
        self.assertEqual(tuple(tokens.shape), (batch, block))

    def test_draft_token_ids_use_live_anchor_and_mask(self):
        """Live sampler anchors keep embed→Layer0 qkv out of fold_meta_constants."""
        batch, block, mask_id = 2, 4, 99
        anchor = torch.tensor([3, 5], dtype=torch.long)
        tokens = DflashWrapper._draft_token_ids(
            batch,
            block,
            device=torch.device("cpu"),
            dtype=torch.long,
            anchor_tokens=anchor,
            mask_token_id=mask_id,
        )
        self.assertEqual(tuple(tokens.shape), (batch, block))
        self.assertEqual(tokens[:, 0].tolist(), [3, 5])
        self.assertTrue(torch.all(tokens[:, 1:] == mask_id))

    def test_as_bsh_normalizes_packed_target_once(self):
        batch, block, hidden = 3, 16, 64
        packed = torch.randn(1, batch * block, hidden)
        out = DflashWrapper.as_bsh(packed, batch, block)
        self.assertEqual(tuple(out.shape), (batch, block, hidden))

    def test_draft_forward_q_len_equals_block_not_query_len(self):
        """End-to-end draft stack: attention Q length follows block_size."""
        from tensor_cast.device import TEST_DEVICE
        from tensor_cast.performance_model.analytic import AnalyticPerformanceModel
        from tensor_cast.runtime import Runtime

        block = 16
        dcfg = DflashConfig(dflash_block_size=block, num_draft_layers=1, context_length=32)
        apply_cli_overrides_to_source_and_dcfg(dcfg, cli_num_draft_layers=1, cli_block_size=block)
        dcfg.aux_hidden_state_layer_ids = [0, 1]
        draft_hf = build_draft_hf_config(dcfg, target_hidden_size=64, target_vocab_size=128)
        draft_hf.num_attention_heads = 4
        draft_hf.num_key_value_heads = 2
        draft_hf.head_dim = 16
        draft_hf.hidden_size = 64
        draft_hf.intermediate_size = 128
        draft_hf.vocab_size = 128
        draft = DflashDraftModel(draft_hf, dcfg, layer_idx_offset=0)
        draft.set_shared(torch.nn.Embedding(128, 64), torch.nn.Linear(64, 128, bias=False))

        position_ids = DflashWrapper._block_position_ids(1, block, device=torch.device("cpu"), dtype=torch.long)
        tokens = DflashWrapper._draft_token_ids(1, block, device=torch.device("cpu"), dtype=torch.long)
        noise = draft.embed_tokens(tokens)
        self.assertEqual(noise.shape[1], block)

        attention_by_layers = {0: AttentionTensorCast()}
        l_ctx = draft.max_l_ctx()
        aux = [torch.randn(1, l_ctx, 64) for _ in draft.target_layer_ids]
        target_context = draft.build_context_features(aux, l_ctx)
        device = TEST_DEVICE
        with Runtime(AnalyticPerformanceModel(device), device) as runtime:
            with torch.no_grad():
                out = draft(
                    noise,
                    target_context,
                    position_ids,
                    attention_by_layers=attention_by_layers,
                )
        self.assertEqual(out.shape, (1, block, 64))
        attn_events = [e for e in runtime.event_list if "tensor_cast.attention" in str(e.op_invoke_info.func)]
        self.assertEqual(len(attn_events), 1)
        query = attn_events[0].op_invoke_info.args[0]
        # query packed as [batch*block, ...]
        self.assertEqual(query.shape[0], block)

    def test_apply_rope_and_reshape_and_cache_emitted(self):
        """Draft path must emit tensor_cast.apply_rope and reshape_and_cache."""
        from tensor_cast.device import TEST_DEVICE
        from tensor_cast.performance_model.analytic import AnalyticPerformanceModel
        from tensor_cast.runtime import Runtime

        dcfg = DflashConfig(dflash_block_size=4, num_draft_layers=1, context_length=16)
        apply_cli_overrides_to_source_and_dcfg(dcfg, cli_num_draft_layers=1)
        dcfg.aux_hidden_state_layer_ids = [0]
        draft_hf = build_draft_hf_config(dcfg, target_hidden_size=64, target_vocab_size=128)
        draft_hf.num_attention_heads = 4
        draft_hf.num_key_value_heads = 2
        draft_hf.head_dim = 16
        draft_hf.hidden_size = 64
        draft_hf.intermediate_size = 128
        draft_hf.vocab_size = 128
        draft = DflashDraftModel(draft_hf, dcfg, layer_idx_offset=0)
        draft.set_shared(torch.nn.Embedding(128, 64), torch.nn.Linear(64, 128, bias=False))
        noise = torch.randn(1, 4, 64)
        position_ids = torch.arange(4).view(1, 4)
        target_context = torch.randn(1, draft.max_l_ctx(), 64)
        with Runtime(AnalyticPerformanceModel(TEST_DEVICE), TEST_DEVICE) as runtime:
            with torch.no_grad():
                draft(
                    noise,
                    target_context,
                    position_ids,
                    attention_by_layers={0: AttentionTensorCast()},
                )
        names = [str(e.op_invoke_info.func) for e in runtime.event_list]
        self.assertTrue(any("tensor_cast.apply_rope" in n for n in names))
        self.assertTrue(any("tensor_cast.reshape_and_cache" in n for n in names))
        # One fused context rope (L_ctx, K-only) + one short noise rope (block, Q+K).
        rope = [e for e in runtime.event_list if "tensor_cast.apply_rope" in str(e.op_invoke_info.func)]
        self.assertEqual(len(rope), 2)
        self.assertEqual(sum("apply_rope_single" in str(e.op_invoke_info.func) for e in rope), 1)
        rope_seqs = []
        for e in rope:
            q = e.op_invoke_info.args[0]
            if isinstance(q, torch.Tensor) and q.ndim == 4:
                rope_seqs.append(int(q.shape[2]))
        self.assertIn(draft.max_l_ctx(), rope_seqs)
        self.assertIn(4, rope_seqs)
        cache = [e for e in runtime.event_list if "tensor_cast.reshape_and_cache" in str(e.op_invoke_info.func)]
        self.assertGreaterEqual(len(cache), 2)


class TestDflashDraftSharedVocab(unittest.TestCase):
    def test_set_shared(self):
        dcfg = DflashConfig(dflash_block_size=8, num_draft_layers=1, context_length=16)
        apply_cli_overrides_to_source_and_dcfg(dcfg, cli_num_draft_layers=1)
        draft_hf = build_draft_hf_config(dcfg, target_hidden_size=64, target_vocab_size=32)
        draft_hf.num_attention_heads = 4
        draft_hf.num_key_value_heads = 2
        draft_hf.head_dim = 16
        draft_hf.hidden_size = 64
        draft_hf.intermediate_size = 128
        draft_hf.vocab_size = 32
        dcfg.aux_hidden_state_layer_ids = [0]
        draft = DflashDraftModel(draft_hf, dcfg, layer_idx_offset=0)
        embed = torch.nn.Embedding(32, 64)
        lm_head = torch.nn.Linear(64, 32, bias=False)
        draft.set_shared(embed, lm_head)
        self.assertIs(draft.embed_tokens, embed)
        self.assertIs(draft.lm_head, lm_head)

    def test_resolve_embed_lm_head_from_causal_lm_wrapper(self):
        """CausalLmWrapper keeps lm_head on wrapper; backbone only has embed_tokens."""
        from transformers import Qwen3Config

        from tensor_cast.layers.utils import ModelWrapperBase
        from tensor_cast.transformers.model import CausalLmWrapper

        hf_config = Qwen3Config(vocab_size=64, hidden_size=32, num_hidden_layers=2)
        backbone = torch.nn.Module()
        backbone.embed_tokens = torch.nn.Embedding(64, 32)
        wrapper = CausalLmWrapper(hf_config, backbone)

        class _Root(ModelWrapperBase):
            def __init__(self, inner):
                super().__init__(inner)

        root = _Root(wrapper)
        embed, lm_head = resolve_target_embed_and_lm_head(root)
        self.assertIs(embed, backbone.embed_tokens)
        self.assertIs(lm_head, wrapper.lm_head)

    def test_dflash_lm_head_excludes_anchor(self):
        """lm_head token dim must be block_size-1 (NPU Index then MatMul)."""
        dcfg = DflashConfig(dflash_block_size=8, num_draft_layers=1, context_length=16)
        apply_cli_overrides_to_source_and_dcfg(dcfg, cli_num_draft_layers=1)
        dcfg.aux_hidden_state_layer_ids = [0]
        draft_hf = build_draft_hf_config(dcfg, target_hidden_size=64, target_vocab_size=32)
        draft_hf.num_attention_heads = 4
        draft_hf.num_key_value_heads = 2
        draft_hf.head_dim = 16
        draft_hf.hidden_size = 64
        draft_hf.intermediate_size = 128
        draft_hf.vocab_size = 32
        draft = DflashDraftModel(draft_hf, dcfg, layer_idx_offset=0)
        draft.set_shared(torch.nn.Embedding(32, 64), torch.nn.Linear(64, 32, bias=False))
        wrapper = DflashWrapper(dcfg, draft_hf, torch.nn.Identity(), draft, draft_hf)
        block = dcfg.dflash_block_size
        hidden = torch.randn(1, block, 64)
        logits, lm_seq = wrapper._apply_draft_lm_head(hidden, block=block)
        self.assertEqual(lm_seq, block - 1)
        self.assertEqual(tuple(logits.shape), (1, block - 1, 32))

    def test_propose_draft_tokens_skips_bonus_assembly(self):
        """Decode draft output is argmax only [B, block-1]; no copy/scatter assembly."""
        dcfg = DflashConfig(dflash_block_size=8, num_draft_layers=1, context_length=16)
        apply_cli_overrides_to_source_and_dcfg(dcfg, cli_num_draft_layers=1)
        dcfg.aux_hidden_state_layer_ids = [0]
        draft_hf = build_draft_hf_config(dcfg, target_hidden_size=64, target_vocab_size=32)
        draft_hf.num_attention_heads = 4
        draft_hf.num_key_value_heads = 2
        draft_hf.head_dim = 16
        draft_hf.hidden_size = 64
        draft_hf.intermediate_size = 128
        draft_hf.vocab_size = 32
        draft = DflashDraftModel(draft_hf, dcfg, layer_idx_offset=0)
        draft.set_shared(torch.nn.Embedding(32, 64), torch.nn.Linear(64, 32, bias=False))
        wrapper = DflashWrapper(dcfg, draft_hf, torch.nn.Identity(), draft, draft_hf)
        block = dcfg.dflash_block_size
        batch = 2
        hidden = torch.randn(batch, block, 64)
        logits = torch.randn(batch, block - 1, 32)
        next_tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8], [8, 7, 6, 5, 4, 3, 2, 1]])
        draft_tokens, verify_tokens = wrapper._propose_draft_tokens(hidden, logits, batch, block, next_tokens)
        self.assertEqual(tuple(draft_tokens.shape), (batch, block - 1))
        self.assertIs(verify_tokens, next_tokens)
        self.assertTrue(torch.equal(draft_tokens, torch.argmax(logits, dim=-1)))


class TestDflashLatencyFolding(unittest.TestCase):
    def test_base_optimizer_folds_by_acceptance(self):
        from serving_cast.service.base_throughput_optimizer import BaseThroughputOptimizer
        from serving_cast.service.latency_table import ForwardLatencyRecord, ForwardShapeKey
        from serving_cast.service.utils import OptimizerData

        class _Probe(BaseThroughputOptimizer):
            def initialize(self, model_runner):
                return None

            def get_inference_info(self, optimizer_data):
                return None

        opt = _Probe()
        key = ForwardShapeKey(is_decode=True, model_concurrency=1, query_len=8, seq_len=128)
        record = ForwardLatencyRecord(latency_ms=12.0, memory_left_gb=1.0, breakdowns="")
        data = OptimizerData(
            input_length=128,
            output_length=128,
            dflash_block_size=8,
            dflash_acceptance_length=5.0,
        )
        self.assertAlmostEqual(opt._get_forward_latency_ms(key, record, data), 2.0)

    def test_fold_helper_prefers_dflash_over_mtp_fields(self):
        from serving_cast.service.base_throughput_optimizer import BaseThroughputOptimizer
        from serving_cast.service.utils import OptimizerData

        data = OptimizerData(
            dflash_block_size=8,
            dflash_acceptance_length=3.0,
            num_mtp_tokens=2,
            mtp_acceptance_rate=[0.9, 0.6],
        )
        # Dflash branch wins when block_size >= 2 (G2 should prevent both being set in CLI).
        self.assertAlmostEqual(BaseThroughputOptimizer._fold_decode_latency_ms(16.0, data), 4.0)

    def test_format_parallel_label_appends_dflash(self):
        from tensor_cast.model_config import ParallelConfig
        from serving_cast.service.utils import format_parallel_label

        label = format_parallel_label(
            ParallelConfig(world_size=8, tensor_parallel_size=8, data_parallel_size=1),
            is_moe_model=False,
            num_mtp_tokens=0,
            dflash_block_size=16,
            dflash_acceptance_length=5.0,
        )
        self.assertIn("DFlash=16/acc=5", label)
        self.assertNotIn("MTP=", label)

    def test_format_parallel_label_without_dflash_unchanged(self):
        from tensor_cast.model_config import ParallelConfig
        from serving_cast.service.utils import format_parallel_label

        label = format_parallel_label(
            ParallelConfig(world_size=8, tensor_parallel_size=8, data_parallel_size=1),
            is_moe_model=False,
            num_mtp_tokens=2,
        )
        self.assertEqual(label, "TP=8 | PP=1 | DP=1 | MTP=2")


class TestDflashCliDependentArgs(unittest.TestCase):
    def test_dependent_options_require_method(self):
        from argparse import ArgumentParser

        from cli.utils import validate_draft_spec_cli_args

        parser = ArgumentParser()
        args = parser.parse_args([])
        args.speculative_method = None
        args.num_mtp_tokens = 0
        with self.assertRaises(SystemExit):
            validate_draft_spec_cli_args(parser, args, argv=["--num-speculative-tokens", "15"])
        with self.assertRaises(SystemExit):
            validate_draft_spec_cli_args(parser, args, argv=["--num-draft-layers", "4"])
        with self.assertRaises(SystemExit):
            validate_draft_spec_cli_args(parser, args, argv=["--draft-model-config-path", "x.json"])

    def test_dependent_options_allowed_with_method(self):
        from argparse import ArgumentParser

        from cli.utils import validate_draft_spec_cli_args

        parser = ArgumentParser()
        args = parser.parse_args([])
        args.speculative_method = "dflash"
        args.num_mtp_tokens = 0
        validate_draft_spec_cli_args(
            parser,
            args,
            argv=["--speculative-method", "dflash", "--num-speculative-tokens", "15", "--num-draft-layers", "4"],
        )

    def test_no_draft_options_ok_without_method(self):
        from argparse import ArgumentParser

        from cli.utils import validate_draft_spec_cli_args

        parser = ArgumentParser()
        args = parser.parse_args([])
        args.speculative_method = None
        args.num_mtp_tokens = 0
        validate_draft_spec_cli_args(parser, args, argv=["--query-length", "8"])


class TestDflashEnableSwitch(unittest.TestCase):
    def test_dflash_flag_enables_with_builtin_block_size(self):
        from unittest.mock import MagicMock

        from tensor_cast.core.config_resolver import ConfigResolver

        resolver = ConfigResolver.__new__(ConfigResolver)
        resolver.model_config = MagicMock()
        resolver.model_config.mtp_config = None
        resolver.model_config.dflash_config = None

        resolver.update_dflash_config(dflash=True, dflash_block_size=0)
        self.assertIsNotNone(resolver.model_config.dflash_config)
        self.assertEqual(resolver.model_config.dflash_config.dflash_block_size, 8)

    def test_block_size_alone_does_not_enable(self):
        from unittest.mock import MagicMock

        from tensor_cast.core.config_resolver import ConfigResolver

        resolver = ConfigResolver.__new__(ConfigResolver)
        resolver.model_config = MagicMock()
        resolver.model_config.mtp_config = None
        resolver.model_config.dflash_config = None

        resolver.update_dflash_config(dflash=False, dflash_block_size=4)
        self.assertIsNone(resolver.model_config.dflash_config)

    def test_neither_flag_nor_block_size_keeps_disabled(self):
        from unittest.mock import MagicMock

        from tensor_cast.core.config_resolver import ConfigResolver

        resolver = ConfigResolver.__new__(ConfigResolver)
        resolver.model_config = MagicMock()
        resolver.model_config.mtp_config = None
        resolver.model_config.dflash_config = None

        resolver.update_dflash_config(dflash=False, dflash_block_size=0)
        self.assertIsNone(resolver.model_config.dflash_config)


class TestDflashAuxFromTargetLayers(unittest.TestCase):
    def test_build_context_features_prefill_pads_to_l_ctx(self):
        dcfg = DflashConfig(dflash_block_size=4, num_draft_layers=1, context_length=16)
        apply_cli_overrides_to_source_and_dcfg(dcfg, cli_num_draft_layers=1)
        dcfg.aux_hidden_state_layer_ids = [0, 1, 2]
        draft_hf = build_draft_hf_config(dcfg, target_hidden_size=64, target_vocab_size=128)
        draft_hf.num_attention_heads = 4
        draft_hf.num_key_value_heads = 2
        draft_hf.head_dim = 16
        draft_hf.hidden_size = 64
        draft_hf.intermediate_size = 128
        draft_hf.vocab_size = 128
        draft = DflashDraftModel(draft_hf, dcfg, layer_idx_offset=0)
        l_ctx = draft.max_l_ctx()
        # Prefill: short aux pads to context_length-derived L_ctx.
        short = [torch.randn(1, 4, 64) for _ in draft.target_layer_ids]
        ctx = draft.build_context_features(short, l_ctx)
        self.assertEqual(ctx.shape, (1, l_ctx, 3 * 64))

    def test_build_context_features_decode_uses_block(self):
        dcfg = DflashConfig(dflash_block_size=4, num_draft_layers=1, context_length=16)
        apply_cli_overrides_to_source_and_dcfg(dcfg, cli_num_draft_layers=1)
        dcfg.aux_hidden_state_layer_ids = [0, 1, 2]
        draft_hf = build_draft_hf_config(dcfg, target_hidden_size=64, target_vocab_size=128)
        draft_hf.num_attention_heads = 4
        draft_hf.num_key_value_heads = 2
        draft_hf.head_dim = 16
        draft_hf.hidden_size = 64
        draft_hf.intermediate_size = 128
        draft_hf.vocab_size = 128
        draft = DflashDraftModel(draft_hf, dcfg, layer_idx_offset=0)
        block = dcfg.dflash_block_size
        # Decode: already [B, block, H]; align_seq=False skips pad→(B,2*block) cat noise.
        short = [torch.randn(1, block, 64) for _ in draft.target_layer_ids]
        from tensor_cast.device import TEST_DEVICE
        from tensor_cast.performance_model.analytic import AnalyticPerformanceModel
        from tensor_cast.runtime import Runtime

        with Runtime(AnalyticPerformanceModel(TEST_DEVICE), TEST_DEVICE) as runtime:
            ctx = draft.build_context_features(short, block, align_seq=False)
        self.assertEqual(ctx.shape, (1, block, 3 * 64))
        for e in runtime.event_list:
            if "tensor_cast.cat" not in str(e.op_invoke_info.func):
                continue
            out = e.op_invoke_info.out
            if isinstance(out, torch.Tensor) and out.dim() >= 2:
                self.assertNotEqual(int(out.shape[1]), 2 * block)

    def test_decode_attn_uses_configured_l_ctx_with_short_fc(self):
        """Decode: fc/context_kv at block; attention seq_lens still L_ctx+block."""
        from tensor_cast.device import TEST_DEVICE
        from tensor_cast.performance_model.analytic import AnalyticPerformanceModel
        from tensor_cast.runtime import Runtime

        dcfg = DflashConfig(dflash_block_size=4, num_draft_layers=1, context_length=16)
        apply_cli_overrides_to_source_and_dcfg(dcfg, cli_num_draft_layers=1)
        dcfg.aux_hidden_state_layer_ids = [0]
        draft_hf = build_draft_hf_config(dcfg, target_hidden_size=64, target_vocab_size=128)
        draft_hf.num_attention_heads = 4
        draft_hf.num_key_value_heads = 2
        draft_hf.head_dim = 16
        draft_hf.hidden_size = 64
        draft_hf.intermediate_size = 128
        draft_hf.vocab_size = 128
        draft = DflashDraftModel(draft_hf, dcfg, layer_idx_offset=0)
        draft.set_shared(torch.nn.Embedding(128, 64), torch.nn.Linear(64, 128, bias=False))
        block = dcfg.dflash_block_size
        l_ctx = draft.max_l_ctx()
        noise = torch.randn(1, block, 64)
        position_ids = torch.arange(block).view(1, block)
        target_context = torch.randn(1, block, 64)
        with Runtime(AnalyticPerformanceModel(TEST_DEVICE), TEST_DEVICE) as runtime:
            with torch.no_grad():
                out = draft(
                    noise,
                    target_context,
                    position_ids,
                    attention_by_layers={0: AttentionTensorCast()},
                    attn_use_configured_context=True,
                )
        self.assertEqual(out.shape, (1, block, 64))
        # Context write rope stays at block (short fc), not L_ctx.
        rope_events = [e for e in runtime.event_list if "tensor_cast.apply_rope" in str(e.op_invoke_info.func)]
        rope_seqs = []
        for e in rope_events:
            q = e.op_invoke_info.args[0]
            if isinstance(q, torch.Tensor) and q.ndim == 4:
                rope_seqs.append(int(q.shape[2]))
        self.assertIn(block, rope_seqs)
        self.assertNotIn(l_ctx, rope_seqs)
        # attention(..., seq_lens, ...) — arg index 6 is seq_lens = L_ctx+block.
        attn_events = [
            e
            for e in runtime.event_list
            if str(e.op_invoke_info.func).endswith("attention")
            or "tensor_cast.attention." in str(e.op_invoke_info.func)
        ]
        attn_events = [
            e
            for e in runtime.event_list
            if "tensor_cast.attention" in str(e.op_invoke_info.func) and "quant" not in str(e.op_invoke_info.func)
        ]
        self.assertEqual(len(attn_events), 1)
        seq_lens = attn_events[0].op_invoke_info.args[6]
        self.assertIsInstance(seq_lens, torch.Tensor)
        self.assertEqual(tuple(seq_lens.shape), (1,))
        # Meta-safe: seq_lens is a full() constant vector of L_ctx+block.
        self.assertEqual(int(seq_lens.tolist()[0]), l_ctx + block)

    def test_draft_module_names_skipped_by_quant_exclude(self):
        from tensor_cast.utils import pattern_match

        exclude = ["lm_head", "draft"]
        self.assertTrue(pattern_match("draft.fc", exclude))
        self.assertTrue(pattern_match("draft.context_kv_proj", exclude))
        self.assertTrue(pattern_match("draft.layers.0.dflash_block.self_attn.q_proj", exclude))
        self.assertTrue(pattern_match("draft.layers.0.dflash_block.mlp.gate_proj", exclude))
        self.assertFalse(pattern_match("_inner.model.layers.0.self_attn.q_proj", exclude))

    def test_quantize_model_keeps_draft_linears_unquantized(self):
        """Draft FFN/Attn/fc must stay nn.Linear when target uses --quantize-linear-action."""
        from tensor_cast.core.quantization.config import create_quant_config
        from tensor_cast.core.quantization.datatypes import QuantizeLinearAction
        from tensor_cast.layers.quant_linear import QuantLinearBase, TensorCastQuantLinear
        from tensor_cast.model_config import ModelConfig, ParallelConfig
        from tensor_cast.transformers.transformations import (
            _ensure_draft_excluded_from_linear_quant,
            quantize_linear,
        )

        dcfg = DflashConfig(
            dflash_block_size=4,
            num_draft_layers=1,
            context_length=8,
            aux_hidden_state_layer_ids=[0],
        )
        apply_cli_overrides_to_source_and_dcfg(dcfg, cli_num_draft_layers=1)
        draft_hf = build_draft_hf_config(dcfg, target_hidden_size=64, target_vocab_size=32)
        draft_hf.num_attention_heads = 4
        draft_hf.num_key_value_heads = 2
        draft_hf.head_dim = 16
        draft_hf.hidden_size = 64
        draft_hf.intermediate_size = 128
        draft_hf.vocab_size = 32
        draft = DflashDraftModel(draft_hf, dcfg, layer_idx_offset=0)
        draft.set_shared(torch.nn.Embedding(32, 64), torch.nn.Linear(64, 32, bias=False))

        class _Target(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList([torch.nn.Linear(64, 64, bias=False) for _ in range(2)])

        target = _Target()
        wrapper = DflashWrapper(dcfg, draft_hf, target, draft, draft_hf)

        class _Host(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self._inner = inner
                self.model_config = ModelConfig(
                    parallel_config=ParallelConfig(),
                    quant_config=create_quant_config(QuantizeLinearAction.W8A8_DYNAMIC),
                    quant_linear_cls=TensorCastQuantLinear,
                    dflash_config=dcfg,
                )

        host = _Host(wrapper)
        _ensure_draft_excluded_from_linear_quant(host)
        self.assertIn("draft", host.model_config.quant_config.modules_to_not_convert)

        quantize_linear(host)

        # Target linears are quantized.
        self.assertIsInstance(host._inner._inner.layers[0], QuantLinearBase)
        # Draft-owned linears (FFN / Attn / fc / context_kv) stay dense.
        draft_mod = host._inner.draft
        self.assertIsInstance(draft_mod.fc, torch.nn.Linear)
        self.assertNotIsInstance(draft_mod.fc, QuantLinearBase)
        self.assertIsInstance(draft_mod.context_kv_proj, torch.nn.Linear)
        block = draft_mod.layers[0].dflash_block
        self.assertIsInstance(block.mlp.gate_proj, torch.nn.Linear)
        self.assertIsInstance(block.mlp.up_proj, torch.nn.Linear)
        self.assertIsInstance(block.mlp.down_proj, torch.nn.Linear)
        self.assertIsInstance(block.self_attn.q_proj, torch.nn.Linear)
        self.assertNotIsInstance(block.mlp.gate_proj, QuantLinearBase)
        self.assertNotIsInstance(block.self_attn.q_proj, QuantLinearBase)

    def test_select_aux_hidden_states_skips_embedding(self):
        from tensor_cast.transformers.model import select_aux_hidden_states

        # HF layout: embed + layers
        all_h = [torch.randn(1, 4, 8) for _ in range(5)]  # embed + 4 layers
        aux = select_aux_hidden_states(all_h, [0, 2], num_layers=4)
        self.assertIs(aux[0], all_h[1])
        self.assertIs(aux[1], all_h[3])

    def test_select_aux_hidden_states_layers_only_non_last(self):
        from tensor_cast.transformers.model import select_aux_hidden_states

        # Layers-only layout: requesting a non-last layer must not use offset=1.
        all_h = [torch.randn(1, 4, 8) for _ in range(6)]
        aux = select_aux_hidden_states(all_h, [1], num_layers=6)
        self.assertIs(aux[0], all_h[1])
        with self.assertRaises(ValueError):
            select_aux_hidden_states(all_h, [1], num_layers=4)

    def test_run_target_collect_aux_synthesizes_from_last_hidden(self):
        """Modeling path: no output_hidden_states; L_aux from intermediate hidden."""
        dcfg = DflashConfig(
            dflash_block_size=4,
            num_draft_layers=1,
            context_length=8,
            aux_hidden_state_layer_ids=[1, 12, 24],
        )
        draft_hf = build_draft_hf_config(dcfg, target_hidden_size=16, target_vocab_size=32)
        draft_hf.num_attention_heads = 4
        draft_hf.num_key_value_heads = 2
        draft_hf.head_dim = 4
        draft_hf.hidden_size = 16
        draft_hf.intermediate_size = 32
        draft_hf.vocab_size = 32
        draft = DflashDraftModel(draft_hf, dcfg, layer_idx_offset=0)
        logits = torch.zeros(1, 4, 32)
        last_hidden = torch.randn(1, 4, 16)
        captured = {}

        class _Inner(torch.nn.Module):
            def forward(self, *args, **kwargs):
                captured.update(kwargs)
                return logits, last_hidden

        wrapper = DflashWrapper(dcfg, draft_hf, _Inner(), draft, draft_hf)
        out, aux = wrapper._run_target_collect_aux(None, torch.zeros(1, 4), None, batch_size=1, tokens_per_req=4)
        self.assertTrue(captured.get("output_intermediate_hidden_states"))
        self.assertNotIn("output_aux_hidden_state_layer_ids", captured)
        self.assertIs(out, logits)
        self.assertEqual(len(aux), 3)
        self.assertTrue(all(a.shape == last_hidden.shape for a in aux))
        # Independent storages so MemoryTracker counts L_aux residency.
        ptrs = {a.data_ptr() for a in aux}
        self.assertEqual(len(ptrs), 3)
        self.assertNotIn(last_hidden.data_ptr(), ptrs)

    def test_unpack_packed_target_hidden_for_draft(self):
        """Packed [1, B*S, H] must become [B, S, H] once via as_bsh."""
        dcfg = DflashConfig(
            dflash_block_size=4,
            num_draft_layers=1,
            context_length=8,
            aux_hidden_state_layer_ids=[0, 1],
        )
        draft_hf = build_draft_hf_config(dcfg, target_hidden_size=16, target_vocab_size=32)
        draft_hf.num_attention_heads = 4
        draft_hf.num_key_value_heads = 2
        draft_hf.head_dim = 4
        draft = DflashDraftModel(draft_hf, dcfg, layer_idx_offset=0)
        wrapper = DflashWrapper(dcfg, draft_hf, torch.nn.Identity(), draft, draft_hf)

        batch, block, hidden = 3, 4, 16
        packed = torch.randn(1, batch * block, hidden)
        unpacked = wrapper.as_bsh(packed, batch, block)
        self.assertEqual(tuple(unpacked.shape), (batch, block, hidden))

        aux = wrapper._synthesize_modeling_aux_hiddens(packed, batch_size=batch, tokens_per_req=block)
        self.assertEqual(len(aux), 2)
        self.assertTrue(all(tuple(a.shape) == (batch, block, hidden) for a in aux))
        self.assertEqual(len({a.data_ptr() for a in aux}), 2)

    def test_resolve_batch_size_from_query_start_loc(self):
        """Packed [1, B*Q] must not be read as B=1; use sampling query_start_loc."""
        from tensor_cast.layers.sampler import SamplingMetadata

        packed = torch.zeros(1, 12, dtype=torch.long)  # B=3, Q=4
        meta = SamplingMetadata(query_start_loc=torch.tensor([0, 4, 8, 12]))
        b = DflashWrapper._resolve_batch_size(packed, None, {"sampling_metadata": meta})
        self.assertEqual(b, 3)

    def test_resync_shared_vocab_rebinds_parallel_embedding(self):
        """After TP replace on target, draft must use ParallelEmbedding (vocab/TP)."""
        from unittest.mock import MagicMock

        from tensor_cast.layers.parallel_embedding import ParallelEmbedding
        from tensor_cast.model_config import WordEmbeddingTPMode
        from tensor_cast.transformers.transformations import _resync_dflash_shared_vocab

        dcfg = DflashConfig(dflash_block_size=4, num_draft_layers=1, context_length=8)
        apply_cli_overrides_to_source_and_dcfg(dcfg, cli_num_draft_layers=1)
        dcfg.aux_hidden_state_layer_ids = [0]
        draft_hf = build_draft_hf_config(dcfg, target_hidden_size=32, target_vocab_size=64)
        draft_hf.num_attention_heads = 4
        draft_hf.num_key_value_heads = 2
        draft_hf.head_dim = 8
        draft_hf.hidden_size = 32
        draft_hf.intermediate_size = 64
        draft_hf.vocab_size = 64
        draft = DflashDraftModel(draft_hf, dcfg, layer_idx_offset=0)

        raw_embed = torch.nn.Embedding(64, 32)
        raw_lm = torch.nn.Linear(32, 64, bias=False)
        draft.set_shared(raw_embed, raw_lm)

        class _Target(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_tokens = raw_embed
                self.lm_head = raw_lm

            def get_input_embeddings(self):
                return self.embed_tokens

            def get_output_embeddings(self):
                return self.lm_head

        target = _Target()
        wrapper = DflashWrapper(dcfg, draft_hf, target, draft, draft_hf)
        # Simulate TP replace on target only (draft still holds raw aliases).
        tp_group = MagicMock()
        tp_group.world_size = 4
        tp_group.rank_in_group = 0
        tp_group.all_reduce = lambda x: x
        parallel_embed = ParallelEmbedding(raw_embed, tp_group, shard_mode=WordEmbeddingTPMode.row)
        target.embed_tokens = parallel_embed
        self.assertIs(wrapper.draft.embed_tokens, raw_embed)
        self.assertEqual(wrapper.draft.embed_tokens.weight.shape[0], 16)  # mutated in place

        model = MagicMock()
        model._inner = wrapper
        model.unwrap = lambda: target
        model.is_vl_model = False
        model.hf_config = draft_hf
        _resync_dflash_shared_vocab(model)
        self.assertIs(wrapper.draft.embed_tokens, parallel_embed)
        self.assertEqual(wrapper.draft.embed_tokens._inner.weight.shape[0], 16)


class TestSpecDecodeSkipsRunnerSampler(unittest.TestCase):
    def _make_wrapper(self):
        dcfg = DflashConfig(dflash_block_size=4, num_draft_layers=1, context_length=8)
        apply_cli_overrides_to_source_and_dcfg(dcfg, cli_num_draft_layers=1)
        draft_hf = build_draft_hf_config(dcfg, target_hidden_size=64, target_vocab_size=128)
        draft = DflashDraftModel(draft_hf, dcfg, layer_idx_offset=0)
        return DflashWrapper(dcfg, draft_hf, torch.nn.Identity(), draft, draft_hf)

    def test_decode_skips_prefill_does_not(self):
        from tensor_cast.core.model_runner import _spec_decode_skips_runner_sampler
        from tensor_cast.layers.sampler import SamplingMetadata

        wrapper = self._make_wrapper()
        model = type("M", (), {"_inner": wrapper})()

        decode_kw = {"sampling_metadata": SamplingMetadata(selected_token_indices=None)}
        prefill_kw = {
            "sampling_metadata": SamplingMetadata(
                selected_token_indices=torch.tensor([0], dtype=torch.long),
            )
        }
        self.assertTrue(_spec_decode_skips_runner_sampler(model, decode_kw))
        self.assertFalse(_spec_decode_skips_runner_sampler(model, prefill_kw))
        self.assertFalse(_spec_decode_skips_runner_sampler(torch.nn.Identity(), decode_kw))

    def test_bare_dflash_wrapper_is_detected(self):
        """Top-level DflashWrapper must not be skipped via its ``_inner`` target."""
        from tensor_cast.core.model_runner import _spec_decode_skips_runner_sampler
        from tensor_cast.layers.sampler import SamplingMetadata

        wrapper = self._make_wrapper()
        decode_kw = {"sampling_metadata": SamplingMetadata(selected_token_indices=None)}
        self.assertTrue(_spec_decode_skips_runner_sampler(wrapper, decode_kw))


if __name__ == "__main__":
    unittest.main()
