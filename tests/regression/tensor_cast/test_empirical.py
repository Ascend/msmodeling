# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import unittest
from unittest.mock import Mock

import torch
from tensor_cast.device import TEST_DEVICE
from tensor_cast.performance_model.analytic import AnalyticPerformanceModel
from tensor_cast.performance_model.base import PerformanceModel
from tensor_cast.performance_model.empirical import EmpiricalPerformanceModel
from tensor_cast.performance_model.profiling_database.data_source import (
    DataSourcePerformanceModel,
    QueryResult,
    QuerySource,
)
from tensor_cast.runtime import Runtime


class TestEmpiricalPerformanceModel(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.device_profile = TEST_DEVICE

        # Create a mock data source
        self.data_source = Mock(spec=DataSourcePerformanceModel)

        # Create a mock fallback model
        self.fallback_model = Mock(spec=PerformanceModel)

        # Create the empirical performance model
        self.empirical_model = EmpiricalPerformanceModel(
            device_profile=self.device_profile,
            data_source=self.data_source,
            fallback_model=self.fallback_model,
        )

    def test_init_with_fallback_model(self):
        """Test initialization with a provided fallback model."""
        self.assertEqual(self.empirical_model.name, "empirical")
        self.assertEqual(self.empirical_model.device_profile, self.device_profile)
        self.assertEqual(self.empirical_model.data_source, self.data_source)
        self.assertEqual(self.empirical_model.fallback_model, self.fallback_model)

    def test_init_without_fallback_model(self):
        """Test initialization without a provided fallback model (should use AnalyticPerformanceModel)."""
        empirical_model = EmpiricalPerformanceModel(device_profile=self.device_profile, data_source=self.data_source)

        self.assertEqual(empirical_model.name, "empirical")
        self.assertEqual(empirical_model.device_profile, self.device_profile)
        self.assertEqual(empirical_model.data_source, self.data_source)
        self.assertIsInstance(empirical_model.fallback_model, AnalyticPerformanceModel)
        self.assertEqual(empirical_model.fallback_model.device_profile, self.device_profile)

    def test_get_classifiers_with_runtime(self):
        """Test get_classifiers returns classifiers from fallback model via Runtime."""
        # Configure mock data source to return None (trigger fallback)
        self.data_source.lookup.return_value = None

        # Create empirical model with mock data source and analytic fallback
        perf_model = EmpiricalPerformanceModel(self.device_profile, self.data_source)

        # Run a simple operation via Runtime to trigger classification
        def func(x, y):
            return torch.matmul(x, y)

        x = torch.randn([100, 100], device="meta")
        y = torch.randn([100, 100], device="meta")

        with (
            Runtime(perf_model, self.device_profile) as runtime,
            torch.no_grad(),
        ):
            func(x, y)

        # Get classifiers from empirical model (should delegate to fallback)
        classifiers = perf_model.get_classifiers()

        # Verify classifiers are returned (from analytic fallback)
        self.assertIsNotNone(classifiers)
        self.assertIsInstance(classifiers, list)

        # Verify table_averages output contains expected keys
        result = runtime.table_averages()
        self.assertIn("empirical", result)

        # Verify breakdowns can be retrieved
        breakdowns = runtime.get_breakdowns()
        self.assertIsInstance(breakdowns, dict)

    def test_empirical_model_with_runtime_matmul(self):
        """Test EmpiricalPerformanceModel with real PyTorch matmul op via Runtime."""
        # Configure mock data source to return a result for matmul
        query_result = Mock(spec=QueryResult)
        query_result.latency_us = 100.0
        query_result.confidence = 0.95
        query_result.source = QuerySource.MEASURED
        query_result.details = {"kernel_type": "MatMulV2"}
        query_result.shape_debug_statistics.return_value = {}
        self.data_source.lookup.return_value = query_result

        # Create empirical model with mock data source
        perf_model = EmpiricalPerformanceModel(self.device_profile, self.data_source)

        # Run a simple matmul operation via Runtime
        def func(x, y):
            return torch.matmul(x, y)

        x = torch.randn([100, 100], device="meta")
        y = torch.randn([100, 100], device="meta")

        with (
            Runtime(perf_model, self.device_profile) as runtime,
            torch.no_grad(),
        ):
            func(x, y)

        # Verify the operation was recorded
        total_time_s = runtime.total_execution_time_s()[perf_model.name]
        self.assertGreater(total_time_s, 0)

        # Verify data source was queried
        self.assertTrue(self.data_source.lookup.called)

    def test_empirical_model_with_runtime_add(self):
        """Test EmpiricalPerformanceModel with real PyTorch add op via Runtime."""
        # Configure mock data source to return None (trigger fallback)
        self.data_source.lookup.return_value = None

        # Create empirical model with mock data source and analytic fallback
        perf_model = EmpiricalPerformanceModel(self.device_profile, self.data_source)

        # Run a simple add operation via Runtime
        def func(x, y):
            return torch.add(x, y)

        x = torch.randn([1000, 1000], device="meta")
        y = torch.randn([1000, 1000], device="meta")

        with (
            Runtime(perf_model, self.device_profile) as runtime,
            torch.no_grad(),
        ):
            func(x, y)

        # Verify the operation was recorded
        total_time_s = runtime.total_execution_time_s()[perf_model.name]
        self.assertGreater(total_time_s, 0)

        # Verify data source was queried
        self.assertTrue(self.data_source.lookup.called)


if __name__ == "__main__":
    unittest.main()
