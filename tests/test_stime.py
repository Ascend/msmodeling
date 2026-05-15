# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import unittest

import salabim as sim
from serving_cast.stime import (
    elapse,
    init_simulation,
    now,
    SimulationEnv,
    start_simulation,
    stop_simulation,
    Task,
)


class TestSimulationEnv(unittest.TestCase):
    """Test SimulationEnv singleton pattern and related functions"""

    def test_singleton_pattern(self):
        """Test the singleton pattern implementation of SimulationEnv"""
        env1 = SimulationEnv()
        env2 = SimulationEnv()
        self.assertIs(env1, env2, "SimulationEnv should implement singleton pattern")


class TestTimeFunctions(unittest.TestCase):
    """Test time-related functions"""

    def setUp(self):
        """Clear the simulation environment before each test"""
        init_simulation()

    def test_now_function(self):
        """Test that the now() function returns the correct logical time"""
        env = SimulationEnv()
        env.run(till=10.0)  # pylint: disable=no-member
        self.assertEqual(now(), 10.0, "now() should return the current simulation time")

    def test_start_simulation(self):
        """Test that start_simulation() runs the simulation correctly"""

        class TestComponent(sim.Component):
            def process(self):
                elapse(5.0)

        TestComponent()
        start_simulation()
        self.assertEqual(now(), 5.0, "Simulation should run until all components complete")


class TestTask(unittest.TestCase):
    """Test Task class functionality"""

    def setUp(self):
        init_simulation()

    def test_task_basic_functions(self):
        """Test Task's time advancement, waiting and awakening functions"""

        class WaitTask(Task):
            def process(self):
                elapse(2.0)  # Test time advancement
                self.wait()  # Test waiting
                elapse(3.0)  # Continue execution after awakening

        coro = WaitTask()
        SimulationEnv()

        # Run simulation
        start_simulation()
        self.assertEqual(now(), 2.0, "Should stop at the waiting point")

        # Awaken the task
        coro.notify()
        start_simulation()
        self.assertEqual(now(), 5.0, "Should continue execution after awakening")

    def test_task_stop(self):
        """Test Task's stop() method"""

        class LongRunningTask(Task):
            def process(self):
                elapse(100.0)  # Long-running task

        class StoppingTask(Task):
            def process(self):
                elapse(10.0)
                stop_simulation()  # Stop simulation at 10 seconds

        LongRunningTask()
        StoppingTask()

        start_simulation()
        self.assertEqual(now(), 10.0, "stop() should terminate the entire simulation")


if __name__ == "__main__":
    unittest.main()
