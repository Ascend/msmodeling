import unittest
from unittest.mock import MagicMock

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


if __name__ == "__main__":
    unittest.main()
