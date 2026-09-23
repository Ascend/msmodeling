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

# _*_coding:utf-8_*_
"""
utils
"""


def get_available_memory_gb(device_profile, runtime, reserved_memory_size_gb=0):
    """
    Get available memory on the device during executing models under the runtime. It is the minimum
    available memory, not the one after model execution.

    :param device_profile: The device configuration
    :param runtime: The runtime under which the models have been executed
    :param reserved_memory_size_gb: The reserved memory size on top of the consumption of the models.
    :return: The minimum available memory during execution.
    """
    total_device_memory_gb = device_profile.memory_size_bytes / 1024**3
    peak_memory_usage_gb = runtime.memory_tracker.peak_mem_usage() / 1024**3
    device_memory_available_gb = total_device_memory_gb - peak_memory_usage_gb - reserved_memory_size_gb
    return device_memory_available_gb
