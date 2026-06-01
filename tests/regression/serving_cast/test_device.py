# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import unittest

from serving_cast.device import Device, DeviceConfig, DummyDeviceConfig, MachineConfig, MachineManager


class TestDeviceConfig(unittest.TestCase):
    def test_device_config_creation(self):
        """Test DeviceConfig creation."""
        config = DeviceConfig()
        self.assertIsNotNone(config)

    def test_device_config_is_empty_class(self):
        """Test that DeviceConfig is a placeholder class."""
        config = DeviceConfig()
        # DeviceConfig is currently a placeholder with no attributes
        self.assertIsInstance(config, DeviceConfig)


class TestDummyDeviceConfig(unittest.TestCase):
    def test_dummy_device_config_creation(self):
        """Test DummyDeviceConfig creation."""
        config = DummyDeviceConfig()
        self.assertIsNotNone(config)

    def test_dummy_device_config_inheritance(self):
        """Test that DummyDeviceConfig inherits from DeviceConfig."""
        config = DummyDeviceConfig()
        self.assertIsInstance(config, DeviceConfig)
        self.assertIsInstance(config, DummyDeviceConfig)


class TestMachineConfig(unittest.TestCase):
    def test_machine_config_default_devices(self):
        """Test MachineConfig with default number of devices."""
        device_config = DummyDeviceConfig()
        machine_config = MachineConfig(device_config)
        self.assertEqual(machine_config.num_devices, 1)

    def test_machine_config_custom_devices(self):
        """Test MachineConfig with custom number of devices."""
        device_config = DummyDeviceConfig()
        machine_config = MachineConfig(device_config, num_devices=8)
        self.assertEqual(machine_config.num_devices, 8)

    def test_machine_config_single_device(self):
        """Test MachineConfig with single device."""
        device_config = DummyDeviceConfig()
        machine_config = MachineConfig(device_config, num_devices=1)
        self.assertEqual(machine_config.num_devices, 1)


class TestDevice(unittest.TestCase):
    def test_device_creation(self):
        """Test Device creation with valid parameters."""
        device_config = DummyDeviceConfig()
        machine_config = MachineConfig(device_config, num_devices=4)
        device = Device(machine_config, device_id=0)
        self.assertEqual(device.id, 0)
        self.assertEqual(device.machine_config, machine_config)

    def test_device_multiple_devices(self):
        """Test creating multiple devices with different IDs."""
        device_config = DummyDeviceConfig()
        machine_config = MachineConfig(device_config, num_devices=4)
        devices = [Device(machine_config, device_id=i) for i in range(4)]
        for i, device in enumerate(devices):
            self.assertEqual(device.id, i)
            self.assertEqual(device.machine_config, machine_config)


class TestMachineManager(unittest.TestCase):
    def test_machine_manager_creation(self):
        """Test MachineManager creation."""
        device_config = DummyDeviceConfig()
        machine_config = MachineConfig(device_config, num_devices=4)
        manager = MachineManager(machine_config)
        self.assertEqual(manager.machine_config, machine_config)
        self.assertEqual(len(manager.devices), 4)

    def test_machine_manager_get_devices(self):
        """Test MachineManager.get_devices method."""
        device_config = DummyDeviceConfig()
        machine_config = MachineConfig(device_config, num_devices=4)
        manager = MachineManager(machine_config)
        devices = manager.get_devices()
        self.assertEqual(len(devices), 4)
        for i, device in enumerate(devices):
            self.assertEqual(device.id, i)
            self.assertEqual(device.machine_config, machine_config)

    def test_machine_manager_single_device(self):
        """Test MachineManager with single device."""
        device_config = DummyDeviceConfig()
        machine_config = MachineConfig(device_config, num_devices=1)
        manager = MachineManager(machine_config)
        devices = manager.get_devices()
        self.assertEqual(len(devices), 1)
        self.assertEqual(devices[0].id, 0)


if __name__ == "__main__":
    unittest.main()
