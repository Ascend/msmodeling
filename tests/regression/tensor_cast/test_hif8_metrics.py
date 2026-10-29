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

"""User-visible diagnostics for unverified HIF8 profiling data."""

import logging


def test_hif8_profile_miss_reports_analytic_fallback(caplog):
    from tensor_cast.performance_model.empirical import EmpiricalOpRecord
    from tensor_cast.performance_model.metrics_collector import MetricsCollector

    collector = MetricsCollector()
    collector.collect_from_records(
        [
            EmpiricalOpRecord(
                func_name="tensor_cast.hif8_linear.default",
                lookup_result=None,
                analytic_latency_s=50e-6,
                tc_shapes=[(3, 16), (16, 32)],
                miss_reason="hif8_requires_verified_profile",
            )
        ]
    )
    with caplog.at_level(logging.INFO):
        collector.log_stats()
    assert collector.get_stats()["miss"] == 1
    assert collector.get_stats()["hit"] == 0
    assert "hif8_requires_verified_profile" in caplog.text
    assert "kernel/scale/layout profile unavailable" in caplog.text
    assert "using hif8 analytic model" in caplog.text.casefold()
    assert "tensor_cast.hif8_linear.default" in caplog.text
