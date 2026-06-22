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
import importlib.util

import pytest
import pandas as pd

from optix.common import (
    get_npu_total_memory,
    get_train_sub_path,
    read_csv_s,
    is_mindie,
    is_vllm,
    ais_bench_exists,
)


def test_get_npu_total_memory_success_ascend910(monkeypatch):
    def _npu_info_usages(*args):
        key_word = "HBM"
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
        key_word = "HBM"
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


class TestGetTrainSubPath:
    def test_creates_subdirectory(self, tmp_path):
        sub = get_train_sub_path(tmp_path)
        assert sub.exists()
        assert sub.is_dir()
        assert sub.parent == tmp_path

    def test_incrementing_names(self, tmp_path):
        sub1 = get_train_sub_path(tmp_path)
        sub2 = get_train_sub_path(tmp_path)
        assert sub1 != sub2

    def test_creates_base_path_if_missing(self, tmp_path):
        new_base = tmp_path / "nonexistent" / "deep"
        sub = get_train_sub_path(new_base)
        assert new_base.exists()
        assert sub.exists()


class TestReadCsvS:
    def test_read_valid_csv(self, tmp_path):
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("col1,col2\n1,2\n3,4\n")
        df = read_csv_s(csv_file)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert list(df.columns) == ["col1", "col2"]

    def test_read_nonexistent_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Failed to read csv"):
            read_csv_s(tmp_path / "nonexistent.csv")


class TestIsMindie:
    def test_returns_bool(self, monkeypatch):
        monkeypatch.setattr(importlib.util, "find_spec", lambda x: None)
        assert is_mindie() is False

    def test_returns_true_when_found(self, monkeypatch):
        mock_spec = object()
        monkeypatch.setattr(
            importlib.util,
            "find_spec",
            lambda x: mock_spec if x == "mindie_llm" else None,
        )
        assert is_mindie() is True


class TestIsVllm:
    def test_returns_false_when_not_installed(self, monkeypatch):
        monkeypatch.setattr(importlib.util, "find_spec", lambda x: None)
        assert is_vllm() is False


class TestAisBenchExists:
    def test_returns_false_when_not_installed(self, monkeypatch):
        monkeypatch.setattr(importlib.util, "find_spec", lambda x: None)
        assert ais_bench_exists() is False


class TestGetNpuTotalMemoryNpuSmiNotFound:
    def test_raises_when_npu_smi_not_found(self, monkeypatch):
        monkeypatch.setattr(shutil, "which", lambda _: None)
        with pytest.raises(ValueError, match="Not Found npu-smi"):
            get_npu_total_memory()


class TestGetNpuTotalMemoryA2Device:
    """Test A2 device branch (ValueError on tuple unpacking → flag=True)"""

    def test_a2_device_uses_direct_query(self, monkeypatch):
        """A2 device has fewer columns, triggers ValueError in unpack → uses -i device_id"""
        key_word = "HBM"

        def _npu_info_usages(*args):
            if args and args[0][2] == "-m":
                # First line succeeds unpack (5 items, phy_id=99 doesn't match device_id=0)
                # Second line has only 3 items → triggers ValueError → flag=True
                return "0 0 0 99 Ascend910B\n0 0 0\n".encode()
            return f"""
    NPU ID                         : 0
    Chip Count                     : 1

    {key_word} Capacity(MB)               : 32768
    {key_word} Usage Rate(%)              : 5
    """.encode()

        monkeypatch.setattr(subprocess, "check_output", _npu_info_usages)
        monkeypatch.setattr(shutil, "which", lambda _: "/usr/bin/npu-smi")

        total_memory, usage_rate = get_npu_total_memory(device_id=0)
        assert total_memory == 32768
        assert usage_rate == 5

    def test_ddr_fallback_when_hbm_not_found(self, monkeypatch):
        """When HBM keywords are not found, falls back to DDR keywords"""

        def _npu_info_usages(*args):
            if args and args[0][2] == "-m":
                return "0 0 0 0 Ascend310\n".encode()
            return """
    NPU ID                         : 0
    Chip Count                     : 1

    DDR Capacity(MB)               : 16384
    DDR Usage Rate(%)              : 5
    DDR Hugepages Total(page)      : 0
    DDR Hugepages Usage Rate(%)    : 10
    """.encode()

        monkeypatch.setattr(subprocess, "check_output", _npu_info_usages)
        monkeypatch.setattr(shutil, "which", lambda _: "/usr/bin/npu-smi")

        total_memory, usage_rate = get_npu_total_memory(device_id=0)
        assert total_memory == 16384
        assert usage_rate == 10

    def test_exception_raised_on_parse_failure(self, monkeypatch):
        """When output cannot be parsed at all"""

        def _npu_info_usages(*args):
            if args and args[0][2] == "-m":
                return "0 0 0 0 Ascend910\n".encode()
            return "no useful info here\n".encode()

        monkeypatch.setattr(subprocess, "check_output", _npu_info_usages)
        monkeypatch.setattr(shutil, "which", lambda _: "/usr/bin/npu-smi")

        with pytest.raises(Exception):
            get_npu_total_memory(device_id=0)

    def test_npu_id_not_digit_raises(self, monkeypatch):
        """When _npu_id is not a digit"""

        def _npu_info_usages(*args):
            if args and args[0][2] == "-m":
                return "NPU CHIP LOGIC PHY NAME\n".encode()
            return "".encode()

        monkeypatch.setattr(subprocess, "check_output", _npu_info_usages)
        monkeypatch.setattr(shutil, "which", lambda _: "/usr/bin/npu-smi")

        with pytest.raises(Exception):
            get_npu_total_memory(device_id=0)

    def test_chip_id_not_digit_raises(self, monkeypatch):
        """When _npu_id is digit but _chip_id is not"""

        def _npu_info_usages(*args):
            if args and args[0][2] == "-m":
                # npu_id=0, chip_id=X (not digit), logic_id=0, phy_id=0, name=Ascend910
                return "0 X 0 0 Ascend910\n".encode()
            return "".encode()

        monkeypatch.setattr(subprocess, "check_output", _npu_info_usages)
        monkeypatch.setattr(shutil, "which", lambda _: "/usr/bin/npu-smi")

        with pytest.raises(ValueError, match="_chip_id"):
            get_npu_total_memory(device_id=0)
