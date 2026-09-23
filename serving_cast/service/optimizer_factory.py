# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2025-2026 Huawei Technologies Co.,Ltd.
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

from .agg_throughput_optimizer import AggThroughputOptimizer
from .disagg_throughput_optimizer import DisaggThroughputOptimizer


class OptimizerFactory:
    _frameworks_cls = {
        AggThroughputOptimizer.name: AggThroughputOptimizer,
        DisaggThroughputOptimizer.name: DisaggThroughputOptimizer,
    }

    @staticmethod
    def create_strategy(model_runner, disagg_mode: bool = False):
        strategy_name = DisaggThroughputOptimizer.name if disagg_mode else AggThroughputOptimizer.name
        if strategy_name not in OptimizerFactory._frameworks_cls.keys():
            raise ValueError(f"Unsupported strategy: {strategy_name}")

        framework = OptimizerFactory._frameworks_cls[strategy_name]()
        framework.initialize(model_runner)
        return framework
