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

"""
profiling_database: operator performance data source system for TensorCast.

Public API:
    DataSourcePerformanceModel  - abstract base class for all data sources
    QueryResult                 - dataclass returned by lookup()
    QuerySource                 - enum indicating how a result was obtained
    ProfilingDataSource         - CSV-backed data source with shape matching
    InterpolatingDataSource     - wrapper that adds interpolation capability
"""

from .backend_projector import CANNBackendProjector
from .data_source import DataSourcePerformanceModel, QueryResult, QuerySource
from .interpolating_data_source import InterpolatingDataSource
from .profiling_data_source import ProfilingDataSource
from .query_demand import KernelQueryDemand, load_query_demand_traces

__all__ = [
    "CANNBackendProjector",
    "DataSourcePerformanceModel",
    "InterpolatingDataSource",
    "KernelQueryDemand",
    "ProfilingDataSource",
    "QueryResult",
    "QuerySource",
    "load_query_demand_traces",
]
