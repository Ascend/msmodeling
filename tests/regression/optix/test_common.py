# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
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

import subprocess
import shutil

from optix.common import get_npu_total_memory


def test_get_npu_total_memory_success_ascend910(monkeypatch):
    def _npu_info_usages(*args):
        key_word = "H" + "B" + "M"
        if args and args[0][2] == "-m":
            return "0 0 0 0 Ascend910".encode()
        return f"""
    NPU ID                         : 0
    Chip Count                     : 1

    DDR Capacity(MB)               : 0
    DDR Usage Rate(%)              : 0
    DDR Hugepages Total(page)      : 0
    DDR Hugepages Usage Rate(%)    : 0
    {key_word} Capacity(MB)               : 65536
    {key_word} Usage Rate(%)              : 3
    Aicore Usage Rate(%)           : 0
    Aivector Usage Rate(%)         : 0
    Aicpu Usage Rate(%)            : 0
    Ctrlcpu Usage Rate(%)          : 2
    DDR Bandwidth Usage Rate(%)    : 0
    {key_word} Bandwidth Usage Rate(%)    : 0
    Chip ID                        : 0
    """.encode()

    monkeypatch.setattr(subprocess, "check_output", _npu_info_usages)
    monkeypatch.setattr(shutil, "which", lambda _: "/usr/bin/npu-smi")

    # Call the function and check the result
    total_memory, usage_rate = get_npu_total_memory()
    assert total_memory == 65536
    assert usage_rate == 3


def test_get_npu_total_memory_success_ascend950pr(monkeypatch):
    def _npu_info_usages(*args):
        key_word = "H" + "B" + "M"
        if args and args[0][2] == "-m":
            return "0 0 0 0 Ascend950PR".encode()
        return f"""
    NPU ID                         : 0
    Chip Count                     : 1

    DDR Capacity(MB)               : 0
    DDR Usage Rate(%)              : 0
    DDR Hugepages Total(page)      : 0
    DDR Hugepages Usage Rate(%)    : 0
    {key_word} Capacity(MB)               : 114688
    {key_word} Usage Rate(%)              : 4
    Aicore Usage Rate(%)           : 0
    Aivector Usage Rate(%)         : 0
    Aicpu Usage Rate(%)            : 0
    Ctrlcpu Usage Rate(%)          : 2
    DDR Bandwidth Usage Rate(%)    : 0
    {key_word} Bandwidth Usage Rate(%)    : 0
    Chip ID                        : 0
    """.encode()

    monkeypatch.setattr(subprocess, "check_output", _npu_info_usages)
    monkeypatch.setattr(shutil, "which", lambda _: "/usr/bin/npu-smi")

    # Call the function and check the result
    total_memory, usage_rate = get_npu_total_memory()
    assert total_memory == 114688
    assert usage_rate == 4
