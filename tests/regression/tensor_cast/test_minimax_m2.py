# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from parameterized import parameterized
from tensor_cast.model_config import ModelConfig, ParallelConfig, QuantConfig
from tensor_cast.transformers.builtin_model.minimax_m2 import shard_qk_norm
from tensor_cast.transformers.custom_model_registry import get_model_profile, get_mtp_block_module_name
from tensor_cast.transformers.model import TransformerModel
from torch import nn


class MiniMaxM2ShardQkNormTestCase(unittest.TestCase):
    def setUp(self):
        self.model_id = str(Path(__file__).resolve().parents[2] / "assets" / "model_config" / "minimax_m2")

    def test_model_profile_declares_mtp_decoder(self):
        self.assertEqual(get_mtp_block_module_name("minimax_m2"), "MiniMaxM2DecoderLayer")

    def test_model_profile_declares_model_structure(self):
        profile = get_model_profile("minimax_m2")

        self.assertIsNotNone(profile)
        self.assertEqual(profile.language_layers_path_str, "layers")
        self.assertEqual(profile.moe_module_name, "MiniMaxM2SparseMoeBlock")
        self.assertFalse(profile.moe_gate_returns_raw_logits)
        self.assertEqual(profile.moe_num_experts_key, "num_experts")
        self.assertEqual(profile.mtp_block_module_name, "MiniMaxM2DecoderLayer")

    def _build_model(self):
        model_config = ModelConfig(
            ParallelConfig(),
            QuantConfig(),
            num_hidden_layers_override=1,
        )
        return TransformerModel(self.model_id, model_config)

    def _get_self_attn(self, model: TransformerModel):
        layer = model.unwrap().layers[0]
        self_attn = layer
        while hasattr(self_attn, "_inner"):
            self_attn = self_attn._inner
        if hasattr(self_attn, "self_attn"):
            self_attn = self_attn.self_attn
        return self_attn

    def _set_tp_group(self, model: TransformerModel, tp_size: int, tp_rank: int):
        tp_group = model.parallel_group_manager.tp_group
        tp_group.world_size = tp_size
        tp_group.rank_in_group = tp_rank

    def _set_qk_norm_weights(
        self,
        self_attn: nn.Module,
        q_requires_grad: bool = True,
        k_requires_grad: bool = True,
    ):
        self_attn.q_norm.weight = nn.Parameter(
            torch.arange(6144, dtype=torch.float32),
            requires_grad=q_requires_grad,
        )
        self_attn.k_norm.weight = nn.Parameter(
            torch.arange(1024, dtype=torch.float32),
            requires_grad=k_requires_grad,
        )

    @parameterized.expand(
        [
            (
                "tp_enabled",
                8,
                3,
                True,
                False,
                True,
                torch.Size([768]),
                torch.Size([128]),
                torch.arange(2304, 3072, dtype=torch.float32),
                torch.arange(384, 512, dtype=torch.float32),
                False,
                True,
            ),
            (
                "gqa_k_norm_rank7",
                8,
                7,
                True,
                True,
                True,
                None,
                torch.Size([128]),
                None,
                torch.arange(896, 1024, dtype=torch.float32),
                True,
                True,
            ),
        ]
    )
    def test_shard_qk_norm_shards_weights(
        self,
        _name,
        tp_size,
        tp_rank,
        use_qk_norm,
        q_requires_grad,
        k_requires_grad,
        expected_q_shape,
        expected_k_shape,
        expected_q_values,
        expected_k_values,
        expected_q_requires_grad,
        expected_k_requires_grad,
    ):
        model = self._build_model()
        self_attn = self._get_self_attn(model)
        self._set_qk_norm_weights(
            self_attn,
            q_requires_grad=q_requires_grad,
            k_requires_grad=k_requires_grad,
        )
        self._set_tp_group(model, tp_size=tp_size, tp_rank=tp_rank)
        model.hf_config.use_qk_norm = use_qk_norm

        result = shard_qk_norm(model)

        self.assertIs(result, model)
        if expected_q_shape is not None:
            self.assertEqual(self_attn.q_norm.weight.shape, expected_q_shape)
        if expected_k_shape is not None:
            self.assertEqual(self_attn.k_norm.weight.shape, expected_k_shape)
        if expected_q_values is not None:
            self.assertTrue(torch.equal(self_attn.q_norm.weight.detach(), expected_q_values))
        if expected_k_values is not None:
            self.assertTrue(torch.equal(self_attn.k_norm.weight.detach(), expected_k_values))
        self.assertEqual(
            self_attn.q_norm.weight.requires_grad,
            expected_q_requires_grad,
        )
        self.assertEqual(
            self_attn.k_norm.weight.requires_grad,
            expected_k_requires_grad,
        )

    @parameterized.expand(
        [
            ("tp_size_one", 1, 0, True),
            ("qk_norm_disabled", 8, 3, False),
        ]
    )
    def test_shard_qk_norm_returns_early(
        self,
        _name,
        tp_size,
        tp_rank,
        use_qk_norm,
    ):
        model = self._build_model()
        self_attn = self._get_self_attn(model)
        self._set_qk_norm_weights(self_attn)
        self._set_tp_group(model, tp_size=tp_size, tp_rank=tp_rank)
        model.hf_config.use_qk_norm = use_qk_norm
        original_q_weight = self_attn.q_norm.weight
        original_k_weight = self_attn.k_norm.weight

        result = shard_qk_norm(model)

        self.assertIs(result, model)
        self.assertIs(self_attn.q_norm.weight, original_q_weight)
        self.assertIs(self_attn.k_norm.weight, original_k_weight)
        self.assertEqual(self_attn.q_norm.weight.shape, torch.Size([6144]))
        self.assertEqual(self_attn.k_norm.weight.shape, torch.Size([1024]))

    def test_shard_qk_norm_deduplicates_each_norm_independently(self):
        q_norm_1 = nn.LayerNorm(8)
        q_norm_2 = nn.LayerNorm(8)
        shared_k_norm = nn.LayerNorm(8)
        modules = (
            SimpleNamespace(q_norm=q_norm_1, k_norm=shared_k_norm),
            SimpleNamespace(q_norm=q_norm_2, k_norm=shared_k_norm),
        )
        model = SimpleNamespace(
            parallel_group_manager=SimpleNamespace(
                tp_group=SimpleNamespace(world_size=2, rank_in_group=0),
            ),
            hf_config=SimpleNamespace(
                use_qk_norm=True,
                num_attention_heads=4,
                num_key_value_heads=2,
            ),
            modules=lambda: iter(modules),
        )

        with patch("tensor_cast.transformers.builtin_model.minimax_m2.replace_with_sharded_tensor") as replace:
            shard_qk_norm(model)

        self.assertEqual(replace.call_count, 3)
        self.assertEqual([call.args[0] for call in replace.call_args_list], [q_norm_1, shared_k_norm, q_norm_2])


if __name__ == "__main__":
    unittest.main()
