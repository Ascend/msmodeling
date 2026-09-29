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
from unittest.mock import Mock, patch

from serving_cast.parallel_runner import ParallelRunner
from serving_cast.service.utils import LengthBin, LengthDistribution, OptimizerData
from tensor_cast.core.user_config import UserInputConfig

from .test_common import SimpleArgs


class TestStaticCompileMode(unittest.TestCase):
    def setUp(self):
        self.args = SimpleArgs()
        self.runner = ParallelRunner(self.args)

    def test_generated_optimizer_configs_use_static_shapes(self):
        configs = list(self.runner._get_user_config())

        self.assertTrue(configs)
        self.assertTrue(all(not config.dynamic_shapes for config in configs))

    def test_runner_forces_static_without_calibration_forward(self):
        user_input = UserInputConfig.from_args(self.args)
        user_input.dynamic_shapes = True
        model_runner = Mock()
        strategy = Mock()

        with (
            patch.object(self.runner, "_build_model_runner", return_value=model_runner) as build_runner,
            patch.object(self.runner, "_create_strategy", return_value=strategy),
        ):
            resolved_runner, resolved_strategy = self.runner._build_static_runner(user_input, disagg_mode=True)

        self.assertIs(resolved_runner, model_runner)
        self.assertIs(resolved_strategy, strategy)
        self.assertFalse(build_runner.call_args.args[0].dynamic_shapes)
        self.assertTrue(user_input.dynamic_shapes)
        model_runner.run_inference.assert_not_called()

    def test_variable_length_requests_run_with_static_runner(self):
        user_input = UserInputConfig.from_args(self.args)
        user_input.dynamic_shapes = True
        model_runner = Mock()
        model_runner.model.model_config = SimpleNamespace(
            mtp_config=None,
            moe_config=None,
            parallel_config=SimpleNamespace(
                data_parallel_size=1,
                tensor_parallel_size=1,
                pipeline_parallel_size=1,
                expert_parallel_size=1,
                moe_tensor_parallel_size=1,
                moe_data_parallel_size=1,
            ),
        )
        optimizer_data = OptimizerData(
            length_distribution=LengthDistribution(
                bins=[
                    LengthBin(min_tokens=100, max_tokens=200, weight=0.5),
                    LengthBin(min_tokens=500, max_tokens=600, weight=0.5),
                ]
            ),
            output_length=16,
        )

        with patch.object(self.runner, "_build_model_runner", return_value=model_runner) as build_runner:
            resolved_runner, strategy = self.runner._build_static_runner(user_input, disagg_mode=False)
            strategy._get_batched_forward_info(2, optimizer_data)

        self.assertIs(resolved_runner, model_runner)
        self.assertFalse(build_runner.call_args.args[0].dynamic_shapes)
        requests = model_runner.run_inference.call_args.args[0]
        self.assertEqual([request.query_len for request in requests], [150, 550])
        self.assertEqual([request.seq_len for request in requests], [150, 550])


if __name__ == "__main__":
    unittest.main()
