# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import unittest

from serving_cast.communication import (
    CommunicationManager,
    get_estimated_communication_time,
)
from serving_cast.config import CommunicationConfig
from serving_cast.stime import (
    CallableTask,
    elapse,
    get_logger,
    init_simulation,
    now,
    start_simulation,
    stop_simulation,
)

logger = get_logger(__name__)


class TestGetEstimatedCommunicationTime(unittest.TestCase):
    def test_basic_calculation(self):
        """Test basic communication time calculation."""
        # num_bytes / (bandwidth * rate)
        # 100 / (100 * 0.5) = 2.0
        result = get_estimated_communication_time(100, 100, 0.5)
        self.assertEqual(result, 2.0)

    def test_large_values(self):
        """Test with large values."""
        # 1e9 / (1e10 * 0.8) = 0.125
        result = get_estimated_communication_time(1e9, 1e10, 0.8)
        self.assertEqual(result, 0.125)


class TestCommunicationManager(unittest.TestCase):
    def setUp(self) -> None:
        # Create a new manager with 10 blocks before each test case
        self.host2device_bandwidth = 100
        self.host2device_rate = 0.5
        init_simulation()
        commun_args = CommunicationConfig(
            host2device_bandwidth=self.host2device_bandwidth,
            host2device_rate=self.host2device_rate,
        )
        self.mgr = CommunicationManager(commun_args)

    def test_host2device_async(self):
        send_bytes = 100
        target_bytes_commun_time = 2
        interval_time = 10

        def func():
            for i in range(5):
                elapse(interval_time)

                def check_callback(index):
                    _ = now()
                    target_now_time = (index + 1) * interval_time + target_bytes_commun_time
                    self.assertEqual(now(), target_now_time)

                self.mgr.host2device_async(send_bytes, check_callback, i)
            elapse(2)
            stop_simulation()

        _ = CallableTask(func)
        start_simulation()

    def test_host2device_sync(self):
        send_bytes = 100
        target_bytes_commun_time = 2
        interval_time = 10

        def func():
            for i in range(5):
                elapse(interval_time)
                self.mgr.host2device_sync(send_bytes)
                _ = now()
                target_now_time = (i + 1) * (interval_time + target_bytes_commun_time)
                self.assertEqual(now(), target_now_time)
            stop_simulation()

        _ = CallableTask(func)
        start_simulation()

    def test_host2device_async_workload_stack(self):
        send_bytes = 1000
        target_bytes_commun_time = 20
        interval_time = 10

        def func():
            for i in range(5):
                elapse(interval_time)

                def check_callback(index):
                    _ = now()
                    target_now_time = (index + 1) * target_bytes_commun_time + interval_time
                    self.assertEqual(now(), target_now_time)

                self.mgr.host2device_async(send_bytes, check_callback, i)
            elapse(2)
            stop_simulation()

        _ = CallableTask(func)
        start_simulation()

    def test_host2device_async_fifo(self):
        unit_send_bytes = 50
        unit_target_bytes_commun_time = 1
        interval_time = 0

        def func():
            for i in range(5):
                elapse(interval_time)

                def check_callback(index):
                    _ = now()
                    target_now_time = unit_target_bytes_commun_time * ((index + 2) * (index + 1) // 2)
                    self.assertEqual(now(), target_now_time)

                self.mgr.host2device_async(unit_send_bytes * (i + 1), check_callback, i)
            elapse(2)
            stop_simulation()

        _ = CallableTask(func)
        start_simulation()


class TestCommunicationManagerValidation(unittest.TestCase):
    def setUp(self) -> None:
        init_simulation()

    def test_host2device_bandwidth_zero(self):
        """Test that zero host2device_bandwidth raises error."""
        config = CommunicationConfig(
            host2device_bandwidth=0,
            host2device_rate=0.5,
        )
        with self.assertRaises(ValueError):
            CommunicationManager(config)

    def test_host2device_bandwidth_negative(self):
        """Test that negative host2device_bandwidth raises error."""
        config = CommunicationConfig(
            host2device_bandwidth=-1,
            host2device_rate=0.5,
        )
        with self.assertRaises(ValueError):
            CommunicationManager(config)

    def test_host2device_rate_zero(self):
        """Test that zero host2device_rate raises error."""
        config = CommunicationConfig(
            host2device_bandwidth=100,
            host2device_rate=0,
        )
        with self.assertRaises(ValueError):
            CommunicationManager(config)

    def test_host2device_rate_gt_one(self):
        """Test that host2device_rate > 1 raises error."""
        config = CommunicationConfig(
            host2device_bandwidth=100,
            host2device_rate=1.5,
        )
        with self.assertRaises(ValueError):
            CommunicationManager(config)

    def test_device2device_bandwidth_zero(self):
        """Test that zero device2device_bandwidth raises error."""
        config = CommunicationConfig(
            host2device_bandwidth=100,
            host2device_rate=0.5,
            device2device_bandwidth=0,
        )
        with self.assertRaises(ValueError):
            CommunicationManager(config)

    def test_device2device_rate_zero(self):
        """Test that zero device2device_rate raises error."""
        config = CommunicationConfig(
            host2device_bandwidth=100,
            host2device_rate=0.5,
            device2device_rate=0,
        )
        with self.assertRaises(ValueError):
            CommunicationManager(config)

    def test_device2device_rate_gt_one(self):
        """Test that device2device_rate > 1 raises error."""
        config = CommunicationConfig(
            host2device_bandwidth=100,
            host2device_rate=0.5,
            device2device_rate=1.5,
        )
        with self.assertRaises(ValueError):
            CommunicationManager(config)


class TestDevice2DeviceCommunication(unittest.TestCase):
    def setUp(self) -> None:
        self.device2device_bandwidth = 200
        self.device2device_rate = 0.5
        init_simulation()
        commun_args = CommunicationConfig(
            host2device_bandwidth=100,
            host2device_rate=0.5,
            device2device_bandwidth=self.device2device_bandwidth,
            device2device_rate=self.device2device_rate,
        )
        self.mgr = CommunicationManager(commun_args)

    def test_device2device_sync(self):
        """Test device2device_sync method."""
        send_bytes = 100
        target_bytes_commun_time = 1  # 100 / (200 * 0.5) = 1

        def func():
            self.mgr.device2device_sync(send_bytes)
            self.assertEqual(now(), target_bytes_commun_time)
            stop_simulation()

        _ = CallableTask(func)
        start_simulation()

    def test_device2device_async(self):
        """Test device2device_async method."""
        send_bytes = 100
        target_bytes_commun_time = 1  # 100 / (200 * 0.5) = 1

        def func():
            def check_callback():
                self.assertEqual(now(), target_bytes_commun_time)

            self.mgr.device2device_async(send_bytes, check_callback)
            elapse(2)
            stop_simulation()

        _ = CallableTask(func)
        start_simulation()


if __name__ == "__main__":
    unittest.main()
