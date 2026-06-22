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
from pathlib import Path
from typing import Tuple
import importlib.util
import shutil
import pandas as pd
from loguru import logger
from .io_utils import ensure_existing_file


_KEY_WORD = "HBM"


def get_train_sub_path(base_path: Path):
    # Generate a new subdirectory under the training output directory
    if not base_path.exists():
        base_path.mkdir(parents=True, exist_ok=True, mode=0o750)
    _sub_len = len([0 for _ in base_path.iterdir()])
    _sub_dir = base_path.joinpath(f"{_sub_len + 1}")
    _sub_dir.mkdir(parents=True, exist_ok=True, mode=0o750)
    return _sub_dir


def read_csv_s(path, **kwargs):
    try:
        ensure_existing_file(path)
        return pd.read_csv(path, **kwargs)
    except Exception as e:
        raise ValueError("Failed to read csv %r." % path) from e


def is_mindie():
    return importlib.util.find_spec("mindie_llm") is not None


def is_vllm():
    return importlib.util.find_spec("vllm") is not None


def ais_bench_exists():
    return importlib.util.find_spec("ais_bench") is not None


def get_npu_total_memory(device_id: int = 0) -> Tuple[int, int]:
    _npu_smi_path = shutil.which("npu-smi")
    if not _npu_smi_path:
        raise ValueError("Not Found npu-smi command path. ")
    _id_map_cmd = ["npu-smi", "info", "-m"]
    cmd = ["npu-smi", "info", "-t", "usages"]
    flag = False
    try:
        _map_out = subprocess.check_output(_id_map_cmd).decode("utf-8")
        _npu_id = _chip_id = 0
        for _line in _map_out.split("\n"):
            if not _line.strip():
                continue
            _result = _line.split()
            try:
                (
                    _npu_id,
                    _chip_id,
                    _chip_logic_id,
                    _chip_phy_id,
                    _chip_name,
                    *_chip_other,
                ) = _result
            except ValueError:
                # A2 does not have phy_id
                flag = True
                break
            if _chip_phy_id.strip() == str(device_id):
                if _chip_name.strip() == "Ascend950PR":
                    # Identify A5
                    flag = True
                break
        if not _npu_id.isdigit():
            raise ValueError(f"_npu_id {_npu_id} is not a digit.")
        if not _chip_id.isdigit():
            raise ValueError(f"_chip_id {_chip_id} is not a digit.")
        if flag:
            # For A2 and A5, query memory info directly using -i
            cmd.extend(["-i", str(device_id)])
        else:
            cmd.extend(["-i", _npu_id, "-c", _chip_id])
        output = subprocess.check_output(cmd).decode("utf-8")
        memory_key_word = _KEY_WORD + " Capacity(MB)"
        usage_rate_key_word = _KEY_WORD + " Usage Rate(%)"
        try:
            total_memory_line = [line for line in output.splitlines() if memory_key_word in line][0]
            memory_usage_rate = [line for line in output.splitlines() if usage_rate_key_word in line][0]
        except IndexError:
            total_memory_line = [line for line in output.splitlines() if "DDR Capacity(MB)" in line][0]
            memory_usage_rate = [line for line in output.splitlines() if "DDR Hugepages Usage Rate(%)" in line][0]
        total_memory_line = total_memory_line.split(":")[1].strip()
        memory_usage_rate = memory_usage_rate.split(":")[1].strip()

        logger.debug(f"cmd: {cmd}, result: {int(total_memory_line), int(memory_usage_rate)}")
        return int(total_memory_line), int(memory_usage_rate)
    except Exception as e:
        logger.error(
            f"Failed to retrieve total video memory. Please check if the video memory query command {cmd} "
            f"matches the current parsing code. "
        )
        raise e
