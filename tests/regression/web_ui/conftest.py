"""web_ui regression fixtures.

The repo root (which exposes the top-level ``web_ui`` package) is made
importable by ``pythonpath = ["."]`` in pyproject's pytest config, so no
sys.path mutation is needed here.

Note: Mock tensor_cast modules before pytest collection to avoid torch dependency.
"""

import sys
from unittest.mock import MagicMock, Mock

# Mock tensor_cast modules to avoid torch dependency
sys.modules["tensor_cast"] = MagicMock()
sys.modules["tensor_cast.device"] = MagicMock()
sys.modules["tensor_cast.ops"] = MagicMock()

# Create a mock DeviceProfile class
mock_device_profile = Mock()
mock_device_profile.all_device_profiles = {}
mock_device_profile.vendor = "MockVendor"
mock_device_profile.name = "MockDevice"
sys.modules["tensor_cast.device"].DeviceProfile = mock_device_profile
