# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
profiling_database: operator performance data source system for TensorCast.

Public API:
    DataSourcePerformanceModel  - abstract base class for all data sources
    QueryResult                 - dataclass returned by lookup()
    QuerySource                 - enum indicating how a result was obtained
    ProfilingDataSource         - CSV-backed data source with shape matching
    InterpolatingDataSource     - wrapper that adds interpolation capability
"""

from .data_source import DataSourcePerformanceModel, QueryResult, QuerySource
from .interpolating_data_source import InterpolatingDataSource
from .profiling_data_source import ProfilingDataSource

__all__ = [
    "DataSourcePerformanceModel",
    "InterpolatingDataSource",
    "ProfilingDataSource",
    "QueryResult",
    "QuerySource",
]
