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
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Tuple
import importlib.util
import shutil
import pandas as pd
from loguru import logger
from msguard import validate_params, Rule


_PREFILL = "prefill"
_DECODE = "decode"
_KEY_WORD = "H" + "B" + "M"


class StateType(Enum):
    DEFAULT = 0
    LINE = 1


@dataclass
class State:
    prefill: int = 0
    decode: int = 0
    batch_prefill: int = 0
    batch_decode: int = 0

    def __repr__(self):
        return f"TT_{self.prefill}_{self.decode}_{self.batch_prefill}_{self.batch_decode}"

    def __hash__(self):
        return hash(self.__repr__())

    def __eq__(self, other):
        return self.__repr__() == other.__repr__()

    def __ne__(self, other):
        return self.__repr__() != other.__repr__()

    def sum(self):
        return self.prefill + self.decode + self.batch_prefill + self.batch_decode


def computer_speed(line_node, field):
    if getattr(line_node, field) == 0:
        return 1
    return 1 / (getattr(line_node, field) * 10**-3)


def computer_speed_with_second(line_node, field):
    if getattr(line_node, field) == 0:
        return 1
    return 1 / (getattr(line_node, field) * 10**-6)


def get_train_sub_path(base_path: Path):
    # Generate a new subdirectory under the training output directory
    if not base_path.exists():
        base_path.mkdir(parents=True, exist_ok=True, mode=0o750)
    _sub_len = len([0 for _ in base_path.iterdir()])
    _sub_dir = base_path.joinpath(f"{_sub_len + 1}")
    _sub_dir.mkdir(parents=True, exist_ok=True, mode=0o750)
    return _sub_dir


def update_global_coefficient(global_coefficient: Dict, key: State, value: float) -> None:
    if key not in global_coefficient:
        global_coefficient[key] = [value]
    else:
        global_coefficient[key].append(value)


def get_module_version(module_name):
    try:
        # Method 1: direct import
        import importlib

        module = importlib.import_module(module_name)
        if hasattr(module, "__version__"):
            return module.__version__
        elif hasattr(module, "version"):
            return module.version
    except ImportError:
        pass

    try:
        # Method 2: use pkg_resources
        import pkg_resources

        return pkg_resources.get_distribution(module_name).version
    except (ImportError, pkg_resources.DistributionNotFound):
        pass

    try:
        # Method 3: use importlib.metadata (Python 3.8+)
        import importlib.metadata

        return importlib.metadata.version(module_name)
    except ImportError:
        pass

    raise ValueError("Module is not installed or version cannot be obtained")


@validate_params({"path": Rule.input_file_read})
def read_csv_s(path, **kwargs):
    try:
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
                _npu_id, _chip_id, _chip_logic_id, _chip_phy_id, _chip_name, *_chip_other = _result
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
