# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import unittest

from serving_cast.service.agg_throughput_optimizer import AggThroughputOptimizer
from serving_cast.service.disagg_throughput_optimizer import DisaggThroughputOptimizer
from serving_cast.service.optimizer_factory import OptimizerFactory
from tensor_cast.core.model_runner import ModelRunner
from tensor_cast.core.user_config import UserInputConfig

from .test_common import SimpleArgs


class TestStrategyFactory(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.args = SimpleArgs()

    def test_create_aggregation_optimizer(self):
        """Test creating aggregation strategy"""
        user_input = UserInputConfig.from_args(self.args)
        model_runner = ModelRunner(user_input)
        strategy = OptimizerFactory.create_strategy(model_runner)

        self.assertIsInstance(strategy, AggThroughputOptimizer)
        self.assertEqual(strategy.name, "aggregation")

    def test_create_disaggregation_optimizer(self):
        """Test creating disaggregation strategy"""
        self.args.disagg = True
        user_input = UserInputConfig.from_args(self.args)
        model_runner = ModelRunner(user_input)
        strategy = OptimizerFactory.create_strategy(model_runner, True)

        self.assertIsInstance(strategy, DisaggThroughputOptimizer)
        self.assertEqual(strategy.name, "disaggregation")


if __name__ == "__main__":
    unittest.main()
