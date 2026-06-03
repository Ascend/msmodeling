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
import os
import shutil
from pathlib import Path

import psutil
from loguru import logger

from msguard import Rule
from msguard.security import walk_s


def remove_file(output_path: Path):
    if not output_path:
        return
    if not isinstance(output_path, Path):
        output_path = Path(output_path)
    if not output_path.exists():
        return
    if output_path.is_file():
        output_path.unlink()
        return
    for file in output_path.iterdir():
        if file.is_file():
            file.unlink()
        else:
            try:
                shutil.rmtree(file)
            except OSError:
                logger.error(f"remove file failed, file_path :{output_path!r}")
                return


def kill_children(children):
    for child in children:
        if not child.is_running():
            continue
        try:
            child.send_signal(9)
            child.wait(10)
        except Exception as e:
            logger.error(f"Failed to kill the {child.pid} process. detail: {e}")
            continue

        if child.is_running():
            logger.error(f"Failed to kill the {child.pid} process.")


def kill_process(process_name):
    for proc in psutil.process_iter(["pid", "name"]):
        if not hasattr(proc, "info"):
            continue
        if process_name not in proc.info["name"]:
            continue
        children = psutil.Process(proc.pid).children(recursive=True)
        kill_children([proc])
        kill_children(children)


def backup(target, bak, class_name="", max_depth=10, current_depth=0):
    if not target or not bak:
        return
    if not isinstance(target, Path):
        target = Path(target)
    if not isinstance(bak, Path):
        bak = Path(bak)
    if not target.exists() or not bak.exists():
        return
    if current_depth >= max_depth:
        logger.warning(f"Reached maximum backup depth {max_depth} for {target}")
        return

    new_file = bak.joinpath(class_name).joinpath(target.name)
    if target.is_file():
        if not Rule.input_file_read.is_satisfied_by(target):
            return
        new_file.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
        if not new_file.exists():
            shutil.copy(target, new_file)
    else:
        if not Rule.input_dir_traverse.is_satisfied_by(target):
            return
        if new_file.exists():
            for child in new_file.iterdir():
                backup(child, new_file, class_name, max_depth, current_depth + 1)
        else:
            shutil.copytree(target, new_file)


def close_file_fp(file_fp):
    if not file_fp:
        return
    try:
        if hasattr(file_fp, 'close'):
            file_fp.close()
        else:
            os.close(file_fp)
    except (AttributeError, OSError):
        return


def get_folder_size(folder_path: Path) -> int:
    folder = Path(folder_path)
    if not folder.exists():
        return 0
    total_size = 0
    for file_path in walk_s(folder):
        if os.path.isfile(file_path):
            total_size += os.path.getsize(file_path)

    return total_size


def get_required_field_from_json(data, key, max_depth=20, current_depth=0):
    """
    Safely retrieve value from json object using dot-separated key path.

    Args:
        data: json object with multi-level nested structure
        key: field name to retrieve, nested levels separated by '.'
    Returns:
        Value if found, None if not found
    """
    if current_depth > max_depth:
        raise ValueError(f"Recursive depth exceeded maximum allowed depth of {max_depth}")
    _cur_key = key
    _next_key = None
    if "." in key:
        _index = key.find(".")
        _cur_key = key[:_index]
        _next_key = key[_index + 1 :]
    _value = None
    if isinstance(data, dict):
        _value = data[_cur_key]
    elif isinstance(data, list):
        _value = data[int(_cur_key)]
    else:
        raise ValueError(f"Unsupported data type: {data}, please confirm. ")
    if _next_key:
        return get_required_field_from_json(_value, _next_key, max_depth, current_depth + 1)
    else:
        return _value


def is_root():
    return os.name != 'nt' and os.getuid() == 0  # pylint: disable=E1101
