"""Tests for grid_generator/model_configs.py — ModelConfig and related utilities."""

import unittest

from tools.perf_data_collection.grid_generator.model_configs import (
    ModelConfig,
    _normalize_name,
    get_matmul_nk_pairs,
    get_expert_nk_pairs,
    get_moe_configs,
)


class TestNormalizeName(unittest.TestCase):
    def test_lowercase(self):
        self.assertEqual(_normalize_name("Qwen3-32B"), "qwen332b")

    def test_remove_dashes(self):
        self.assertEqual(_normalize_name("DeepSeek-V3"), "deepseekv3")

    def test_remove_underscores(self):
        self.assertEqual(_normalize_name("llama_70b"), "llama70b")

    def test_remove_spaces(self):
        self.assertEqual(_normalize_name("Qwen3 32B"), "qwen332b")

    def test_remove_dots_and_slashes(self):
        self.assertEqual(_normalize_name("meta-llama/Meta-Llama-3-70B"), "metallamametallama370b")


class TestModelConfig(unittest.TestCase):
    def test_dense_is_not_mla(self):
        cfg = ModelConfig(
            name="Qwen3-32B",
            hidden_size=5120,
            intermediate_size=25600,
            num_attention_heads=64,
            num_kv_heads=8,
        )
        self.assertFalse(cfg.is_mla())

    def test_mla_with_q_lora(self):
        cfg = ModelConfig(
            name="DSv3",
            hidden_size=7168,
            intermediate_size=18432,
            num_attention_heads=128,
            num_kv_heads=1,
            q_lora_rank=1536,
        )
        self.assertTrue(cfg.is_mla())

    def test_mla_with_kv_lora(self):
        cfg = ModelConfig(
            name="GLM-5.1",
            hidden_size=6144,
            intermediate_size=12288,
            num_attention_heads=64,
            num_kv_heads=64,
            kv_lora_rank=512,
        )
        self.assertTrue(cfg.is_mla())

    def test_dense_matmul_nk_pairs(self):
        cfg = ModelConfig(
            name="Test",
            hidden_size=5120,
            intermediate_size=25600,
            num_attention_heads=64,
            num_kv_heads=8,
            tp_sizes=(1, 4),
        )
        pairs = cfg.matmul_nk_pairs()
        self.assertIn((5120, 5120), pairs)
        self.assertIn((1280, 5120), pairs)
        self.assertIn((25600, 5120), pairs)
        self.assertIn((6400, 5120), pairs)

    def test_mla_matmul_nk_pairs(self):
        cfg = ModelConfig(
            name="DSv3",
            hidden_size=7168,
            intermediate_size=18432,
            num_attention_heads=128,
            num_kv_heads=1,
            q_lora_rank=1536,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            head_dim=128,
            tp_sizes=(1,),
        )
        pairs = cfg.matmul_nk_pairs()
        self.assertIn((1536, 7168), pairs)
        q_up = 128 * 128
        self.assertIn((q_up, 1536), pairs)
        self.assertIn((512 + 64, 7168), pairs)
        self.assertIn((q_up, 512), pairs)

    def test_dense_no_expert_nk_pairs(self):
        cfg = ModelConfig(
            name="Qwen3-32B",
            hidden_size=5120,
            intermediate_size=25600,
            num_attention_heads=64,
            num_kv_heads=8,
        )
        self.assertEqual(cfg.expert_nk_pairs(), set())

    def test_moe_expert_nk_pairs(self):
        cfg = ModelConfig(
            name="DSv3",
            hidden_size=7168,
            intermediate_size=18432,
            num_attention_heads=128,
            num_kv_heads=1,
            num_experts=256,
            expert_intermediate_size=2048,
            tp_sizes=(1, 8),
        )
        pairs = cfg.expert_nk_pairs()
        self.assertIn((2048, 7168), pairs)
        self.assertIn((256, 896), pairs)


class TestGetMatmulNkPairs(unittest.TestCase):
    def test_default_returns_all_models(self):
        pairs = get_matmul_nk_pairs(None)
        self.assertIsInstance(pairs, set)
        self.assertTrue(len(pairs) > 0)

    def test_qwen3_32b(self):
        pairs = get_matmul_nk_pairs(["Qwen3-32B"])
        self.assertTrue(len(pairs) > 0)
        for n, k in pairs:
            self.assertIsInstance(n, int)
            self.assertIsInstance(k, int)


class TestGetExpertNkPairs(unittest.TestCase):
    def test_dense_model_returns_empty(self):
        pairs = get_expert_nk_pairs(["Qwen3-32B"])
        self.assertEqual(pairs, set())

    def test_moe_model_returns_pairs(self):
        pairs = get_expert_nk_pairs(["DSv3"])
        self.assertTrue(len(pairs) > 0)


class TestGetMoeConfigs(unittest.TestCase):
    def test_dense_model_filtered_out(self):
        configs = get_moe_configs(["Qwen3-32B"])
        self.assertEqual(len(configs), 0)

    def test_moe_model_returned(self):
        configs = get_moe_configs(["DSv3"])
        self.assertEqual(len(configs), 1)
        self.assertGreater(configs[0].num_experts, 0)
