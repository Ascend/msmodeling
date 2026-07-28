import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from tensor_cast.core.config_resolver import ConfigResolver
from tensor_cast.model_config import ParallelConfig


def _make_resolver(ep_size: int = 1) -> ConfigResolver:
    """Build a ConfigResolver with mocked internals, no network needed."""
    resolver = object.__new__(ConfigResolver)
    parallel_config = MagicMock(spec=ParallelConfig)
    parallel_config.expert_parallel_size = ep_size
    model_config = MagicMock()
    model_config.parallel_config = parallel_config
    resolver.model_config = model_config
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


if __name__ == "__main__":
    unittest.main()
