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

from typing import List


class DeviceConfig:
    # TOBEDONE add more device spec
    pass


class DummyDeviceConfig(DeviceConfig):
    pass


class MachineConfig:
    """
    A "machine" has a list of homogeneous devices residing in the same node
    or across nodes. They are connected with some interconnect topology and used
    as a server instance for some computation tasks.
    """

    def __init__(self, device_config: DeviceConfig, num_devices: int = 1):
        self.num_devices = num_devices
        # TOBEDONE add topology info


class Device:
    def __init__(self, machine_config: MachineConfig, device_id: int):
        self.machine_config = machine_config
        self.id = device_id


class MachineManager:
    def __init__(self, machine_config: MachineConfig):
        self.machine_config = machine_config
        self.devices = [Device(machine_config, i) for i in range(machine_config.num_devices)]

    def get_devices(self) -> List[Device]:
        return self.devices
