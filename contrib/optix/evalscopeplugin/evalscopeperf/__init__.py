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

"""evalscopeperf 插件：封装 evalscope 的 perf 命令作为 optix 寻优器的benchmark。"""


def register():
    """向 optix 注册本插件的 benchmark 与配置。"""
    # 导入当前包内的模块
    from evalscopeperf.benchmark import EvalscopePerfBenchMark
    from evalscopeperf.settings import CusSettings
    from optix.optimizer.register import register_benchmarks
    from optix.config.config import register_settings

    # 注册插件
    register_benchmarks("evalscopeperf", EvalscopePerfBenchMark)
    register_settings(lambda: CusSettings())
