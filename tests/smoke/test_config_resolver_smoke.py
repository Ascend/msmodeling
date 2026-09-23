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
from unittest.mock import MagicMock

from tensor_cast.core.config_resolver import ConfigResolver
from tensor_cast.model_config import ParallelConfig


def _make_resolver(ep_size: int = 1, tp_size: int = 2, dp_size: int = 2, moe_dp_size: int = 1) -> ConfigResolver:
    # Defaults keep the EP token domain consistent (ep * moe_dp == tp * dp, issue #456).
    resolver = object.__new__(ConfigResolver)
    parallel_config = MagicMock(spec=ParallelConfig)
    parallel_config.expert_parallel_size = ep_size
    parallel_config.tensor_parallel_size = tp_size
    parallel_config.data_parallel_size = dp_size
    parallel_config.moe_data_parallel_size = moe_dp_size
    model_config = MagicMock()
    model_config.parallel_config = parallel_config
    resolver.model_config = model_config
    return resolver


class ValidateMoeParallelConfigSmokeTestCase(unittest.TestCase):
    def test_no_moe_config_passes(self):
        resolver = _make_resolver()
        resolver.model_config.moe_config = None
        resolver.validate_moe_parallel_config()

    def test_valid_shared_expert_tp_with_ep(self):
        resolver = _make_resolver(ep_size=4)
        moe_config = MagicMock()
        moe_config.enable_shared_expert_tp = True
        moe_config.host_external_shared_experts = False
        resolver.model_config.moe_config = moe_config
        resolver.validate_moe_parallel_config()


if __name__ == "__main__":
    unittest.main()
