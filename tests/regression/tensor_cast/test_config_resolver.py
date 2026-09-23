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
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from tensor_cast.core.config_resolver import ConfigResolver, _resolve_model_dtype
from tensor_cast.model_config import ParallelConfig


def _make_resolver(
    ep_size: int = 1, tp_size: int = 2, dp_size: int = 2, moe_dp_size: int = 1, sp_or_dfc: bool = False
) -> ConfigResolver:
    """Build a ConfigResolver with mocked internals, no network needed.

    Defaults keep the EP token domain consistent (ep * moe_dp == tp * dp) so
    callers that only care about other constraints stay unaffected; pass
    explicit values to exercise the EP-domain validation. ``sp_or_dfc`` turns
    on enable_sequence_parallel (the model path the EP-domain check guards).
    """
    resolver = object.__new__(ConfigResolver)
    parallel_config = MagicMock(spec=ParallelConfig)
    parallel_config.expert_parallel_size = ep_size
    parallel_config.tensor_parallel_size = tp_size
    parallel_config.data_parallel_size = dp_size
    parallel_config.moe_data_parallel_size = moe_dp_size
    model_config = MagicMock()
    model_config.parallel_config = parallel_config
    resolver.model_config = model_config
    resolver.user_input = SimpleNamespace(
        enable_sequence_parallel=sp_or_dfc,
        enable_dispatch_ffn_combine=False,
    )
    return resolver


class ValidateMoeParallelConfigTestCase(unittest.TestCase):
    def test_no_moe_config_passes(self):
        resolver = _make_resolver()
        resolver.model_config.moe_config = None
        resolver.validate_moe_parallel_config()  # should not raise

    def test_shared_expert_tp_requires_ep_greater_than_1(self):
        resolver = _make_resolver(ep_size=1)
        moe_config = MagicMock()
        moe_config.enable_shared_expert_tp = True
        moe_config.host_external_shared_experts = False
        resolver.model_config.moe_config = moe_config
        with self.assertRaises(ValueError) as ctx:
            resolver.validate_moe_parallel_config()
        self.assertIn("expert_parallel_size must be greater than 1", str(ctx.exception))

    def test_shared_expert_tp_and_host_external_mutually_exclusive(self):
        resolver = _make_resolver(ep_size=4)
        moe_config = MagicMock()
        moe_config.enable_shared_expert_tp = True
        moe_config.host_external_shared_experts = True
        resolver.model_config.moe_config = moe_config
        with self.assertRaises(ValueError) as ctx:
            resolver.validate_moe_parallel_config()
        self.assertIn("mutually exclusive", str(ctx.exception))

    def test_valid_shared_expert_tp_with_ep(self):
        resolver = _make_resolver(ep_size=4)
        moe_config = MagicMock()
        moe_config.enable_shared_expert_tp = True
        moe_config.host_external_shared_experts = False
        resolver.model_config.moe_config = moe_config
        resolver.validate_moe_parallel_config()  # should not raise

    def test_ep_domain_mismatch_raises(self):
        """EP x MOE-DP != TP x DP is rejected under SP (issue #456: TP=2, DP=4, EP=4)."""
        resolver = _make_resolver(ep_size=4, tp_size=2, dp_size=4, moe_dp_size=1, sp_or_dfc=True)
        moe_config = MagicMock()
        moe_config.enable_shared_expert_tp = False
        moe_config.host_external_shared_experts = False
        resolver.model_config.moe_config = moe_config
        with self.assertRaises(ValueError) as ctx:
            resolver.validate_moe_parallel_config()
        self.assertIn("Invalid expert-parallel domain", str(ctx.exception))
        self.assertIn("EP (4) x MOE-DP (1)", str(ctx.exception))
        self.assertIn("TP (2) x DP (4)", str(ctx.exception))
        self.assertIn("--moe-tp-size", str(ctx.exception))

    def test_ep_domain_mismatch_allowed_without_sp_or_dfc(self):
        """moe_tp > 1 with EP stays a formal contract without SP / dispatch_ffn_combine.

        The EP token-domain check (issue #456) must not fire on other model
        paths: cli/registry/validators.py accepts moe_tp x moe_dp x ep ==
        num_devices regardless of TP x DP.
        """
        resolver = _make_resolver(ep_size=4, tp_size=2, dp_size=4, moe_dp_size=1, sp_or_dfc=False)
        moe_config = MagicMock()
        moe_config.enable_shared_expert_tp = False
        moe_config.host_external_shared_experts = False
        resolver.model_config.moe_config = moe_config
        resolver.validate_moe_parallel_config()  # should not raise

    def test_ep_domain_conservation_with_moe_dp_passes(self):
        """EP x MOE-DP == TP x DP with MOE-DP > 1 is a valid expert-replicated domain."""
        resolver = _make_resolver(ep_size=4, tp_size=2, dp_size=4, moe_dp_size=2, sp_or_dfc=True)
        moe_config = MagicMock()
        moe_config.enable_shared_expert_tp = False
        moe_config.host_external_shared_experts = False
        resolver.model_config.moe_config = moe_config
        resolver.validate_moe_parallel_config()  # should not raise

    def test_ep_domain_vllm_style_passes(self):
        """vLLM-style EP = TP x DP with MOE-DP = 1 passes."""
        resolver = _make_resolver(ep_size=8, tp_size=2, dp_size=4, moe_dp_size=1, sp_or_dfc=True)
        moe_config = MagicMock()
        moe_config.enable_shared_expert_tp = False
        moe_config.host_external_shared_experts = False
        resolver.model_config.moe_config = moe_config
        resolver.validate_moe_parallel_config()  # should not raise

    def test_ep_domain_check_skipped_without_ep(self):
        """EP=1 disables expert parallelism; the domain check must not apply."""
        resolver = _make_resolver(ep_size=1, tp_size=2, dp_size=4, moe_dp_size=2)
        moe_config = MagicMock()
        moe_config.enable_shared_expert_tp = False
        moe_config.host_external_shared_experts = False
        resolver.model_config.moe_config = moe_config
        resolver.validate_moe_parallel_config()  # should not raise


class ModelDtypeResolutionTestCase(unittest.TestCase):
    def test_prefers_outer_torch_dtype(self):
        config = SimpleNamespace(torch_dtype="bfloat16", dtype="float16")

        self.assertIs(_resolve_model_dtype(config), torch.bfloat16)

    def test_uses_dtype_when_torch_dtype_is_absent(self):
        config = SimpleNamespace(dtype="bfloat16")

        self.assertIs(_resolve_model_dtype(config), torch.bfloat16)

    def test_uses_text_config_dtype_when_outer_config_has_no_declaration(self):
        config = SimpleNamespace(text_config=SimpleNamespace(dtype=torch.bfloat16))

        self.assertIs(_resolve_model_dtype(config), torch.bfloat16)

    def test_falls_back_to_fp16_for_missing_or_unsupported_dtype(self):
        self.assertIs(_resolve_model_dtype(SimpleNamespace()), torch.float16)
        self.assertIs(_resolve_model_dtype(SimpleNamespace(torch_dtype="float8_e4m3fn")), torch.float16)

    @patch("tensor_cast.core.config_resolver.AutoModelConfigLoader")
    def test_config_resolver_passes_declared_dtype_to_model_config(self, loader_cls):
        loader = loader_cls.return_value
        loader.load_config.return_value = SimpleNamespace(torch_dtype="bfloat16")
        loader.is_transformers_natively_supported = True
        user_input = MagicMock()
        user_input.model_id = "test-model"
        user_input.remote_source = "huggingface"
        user_input.get_quant_config.return_value = MagicMock()
        user_input.get_parallel_config.return_value = MagicMock()

        resolver = ConfigResolver(user_input=user_input)

        self.assertIs(resolver.model_config.dtype, torch.bfloat16)


class DsaCpStructureTestCase(unittest.TestCase):
    def test_has_dsa_structure_matches_index_topk(self):
        resolver = _make_resolver()
        resolver.hf_config = SimpleNamespace(hf_text_config=SimpleNamespace(index_topk=8))

        self.assertTrue(resolver._has_dsa_structure())

    def test_has_dsa_structure_matches_normalized_topk_limit_and_index_heads(self):
        resolver = _make_resolver()
        resolver.hf_config = SimpleNamespace(topk_limit=8, index_n_heads=64)

        self.assertTrue(resolver._has_dsa_structure())

    @patch("tensor_cast.core.config_resolver.get_mla_module")
    @patch("tensor_cast.core.config_resolver.get_model_profile", return_value=None)
    @patch("tensor_cast.core.config_resolver.get_mla_module_name", return_value="FakeMla")
    def test_update_mla_config_owns_dsa_cp_layout(self, _module_name, _profile, mla_cls):
        resolver = _make_resolver()
        resolver.hf_config = SimpleNamespace(model_type="fake_mla")

        resolver.update_mla_config(enable_dsa_cp=True)

        self.assertTrue(resolver.model_config.mla_config.enable_dsa_cp)
        self.assertIs(resolver.model_config.mla_config.mla_cls, mla_cls.return_value)


class DraftSpecRepetitionTestCase(unittest.TestCase):
    def _resolve_with(self, user_input):
        from tensor_cast.model_config import ModelConfig, ParallelConfig, QuantConfig

        resolver = object.__new__(ConfigResolver)
        resolver.user_input = user_input
        resolver.model_config = ModelConfig(ParallelConfig(), QuantConfig())
        resolver.hf_config = MagicMock()
        resolver.hf_config.model_type = "qwen3"
        captured = {}

        def capture(**kwargs):
            captured.update(kwargs)

        resolver.update_hf_config = capture
        resolver.update_moe_config = MagicMock()
        resolver.update_mla_config = MagicMock()
        resolver.update_mtp_config = MagicMock()
        resolver.update_dspark_config = MagicMock()
        resolver.update_dflash_config = MagicMock()
        resolver.update_parallel_config = MagicMock()
        resolver.validate_moe_parallel_config = MagicMock()
        resolver.resolve()
        return captured

    def test_dspark_keeps_representative_layer_reuse(self):
        from tensor_cast.core.user_config import UserInputConfig

        captured = self._resolve_with(
            UserInputConfig(speculative_method="dspark", num_speculative_tokens=7, disable_repetition=False)
        )
        self.assertTrue(captured["enable_repetition"])

    def test_dflash_keeps_representative_layer_reuse(self):
        from tensor_cast.core.user_config import UserInputConfig

        captured = self._resolve_with(
            UserInputConfig(speculative_method="dflash", num_speculative_tokens=7, disable_repetition=False)
        )
        self.assertTrue(captured["enable_repetition"])

    def test_baseline_keeps_repetition_when_not_disabled(self):
        from tensor_cast.core.user_config import UserInputConfig

        captured = self._resolve_with(UserInputConfig(disable_repetition=False))
        self.assertTrue(captured["enable_repetition"])


if __name__ == "__main__":
    unittest.main()
