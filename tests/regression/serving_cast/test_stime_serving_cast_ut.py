# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import unittest

import serving_cast.stime as stime


class TestCallableTask(unittest.TestCase):
    def setUp(self):
        stime.init_simulation()

    def test_callable_task_basic(self):
        """Test CallableTask with basic function."""
        result = []

        def test_func(value):
            result.append(value)

        task = stime.CallableTask(test_func, 42)
        task.process()
        self.assertEqual(result, [42])

    def test_callable_task_with_kwargs(self):
        """Test CallableTask with kwargs."""
        result = []

        def test_func(a, b, c=10):
            result.append(a + b + c)

        task = stime.CallableTask(test_func, 1, 2, c=3)
        task.process()
        self.assertEqual(result, [6])

    def test_callable_task_simulation(self):
        """Test CallableTask in simulation."""
        execution_order = []

        def func1():
            execution_order.append("func1_start")
            stime.elapse(1.0)
            execution_order.append("func1_end")

        def func2():
            execution_order.append("func2_start")
            stime.elapse(1.0)
            execution_order.append("func2_end")
            stime.stop_simulation()

        _ = stime.CallableTask(func1)
        _ = stime.CallableTask(func2)
        stime.start_simulation()
        # func1 should complete before func2 starts due to sequential execution
        self.assertIn("func1_start", execution_order)
