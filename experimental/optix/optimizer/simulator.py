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
import json
import os
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from copy import deepcopy
from pathlib import Path
from typing import Any, Tuple, Optional, Union
import shutil
import tempfile
import time
from loguru import logger
import yaml
from msguard.security import open_s
from ..config.config import MindieConfig, VllmConfig, OptimizerConfigField, KubectlConfig
from ..config.base_config import simulate_flag, SIMULATE
from ..config.custom_command import MindieCommand, VllmCommand
from ..optimizer.custom_process import CustomProcess
from ..optimizer.utils import backup, remove_file, close_file_fp

from ..optimizer.plugins.simulate import Simulator


@contextmanager
def enable_simulate_old(simulate):
    if simulate_flag and isinstance(simulate, Simulator):
        origin_data = simulate.default_config
        data = deepcopy(origin_data)
        simulate.default_config = data
        model_config = data["BackendConfig"]["ModelDeployConfig"]["ModelConfig"][0]
        if "plugin_params" in model_config:
            _plugin_params = json.loads(model_config["plugin_params"])
            if SIMULATE not in _plugin_params["plugin_type"]:
                _plugin_params["plugin_type"] += "," + SIMULATE
                model_config["plugin_params"] = json.dumps(_plugin_params)
        else:
            model_config["plugin_params"] = json.dumps({"plugin_type": SIMULATE})
        with open_s(simulate.config.config_path, 'w') as f:
            json.dump(data, f, indent=4)
        yield simulate_flag
        if simulate.config.config_path.exists():
            simulate.config.config_path.unlink()
        with open_s(simulate.config.config_path, 'w') as f:
            json.dump(origin_data, f, indent=4)
    else:
        yield simulate_flag
    return