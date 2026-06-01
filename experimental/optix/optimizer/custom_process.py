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
import subprocess
import tempfile
import time
from math import isnan, isinf
from pathlib import Path
from typing import Tuple, Optional, List

import psutil
from loguru import logger
from msguard.security import open_s
from ..config.base_config import CUSTOM_OUTPUT, MODEL_EVAL_STATE_CONFIG_PATH, \
    ms_serviceparam_optimizer_config_path
from ..config.config import OptimizerConfigField, get_settings
from ..optimizer.utils import close_file_fp, remove_file, kill_children, \
    backup, kill_process


class CustomProcess:
    def __init__(self, bak_path: Optional[Path] = None, command: Optional[List[str]] = None,
                 work_path: Optional[Path] = None, print_log: bool = False,
                 process_name: str = ""):
        self.command = command
        self.bak_path = bak_path
        self.work_path = work_path if work_path else os.getcwd()
        self.run_log = None
        self.run_log_offset = None
        self.run_log_fp = None
        self.process = None
        self.print_log = print_log
        self.process_name = process_name
        self.env = os.environ.copy()
        self.command = None

    @staticmethod
    def kill_residual_process(process_name):
        """
        Check environment, see if there are residual tasks and clean them up
        """
        logger.debug("check env")
        _residual_process = []
        _all_process_name = process_name.split(",")
        for proc in psutil.process_iter(["pid", "name"]):
            if not hasattr(proc, "info"):
                continue
            _proc_flag = []
            for p in _all_process_name:
                if p not in proc.info["name"]:
                    _proc_flag.append(True)
                else:
                    _proc_flag.append(False)
            if all(_proc_flag):
                continue
            _residual_process.append(proc)
        if _residual_process:
            logger.debug("kill residual_process")
            for _p_name in _all_process_name:
                try:
                    kill_process(_p_name)
                except Exception as e:
                    logger.error(f"Failed to kill process. {e}")
        time.sleep(1)


    def backup(self):
        # Backup operation, default to backing up log
        backup(self.run_log, self.bak_path, self.__class__.__name__)

    def before_run(self, run_params: Optional[Tuple[OptimizerConfigField, ...]] = None):
        self.run_log_fp, self.run_log = tempfile.mkstemp(prefix="ms_serviceparam_optimizer_")
        self.run_log_offset = 0
        if not run_params:
            return
        for k in run_params:
            if k.config_position == "env":
                # env type data, set environment variables and update variable references in command, all uppercase when setting
                self.env[k.name.upper().strip()] = str(k.value)
                _var_name = f"${k.name.upper().strip()}"
                if _var_name not in self.command:
                    continue
                _i = self.command.index(_var_name)
                value_flag = isnan(k.value) or isinf(k.value)
                if k.value is None or value_flag or not str(k).strip():
                    self.command.pop(_i)
                    if _i > 0:
                        self.command.pop(_i - 1)
                else:
                    self.command[_i] = str(k.value)
        if CUSTOM_OUTPUT not in self.env:
            # Set output directory
            self.env[CUSTOM_OUTPUT] = str(get_settings().output)
        # Set the json file to read
        if MODEL_EVAL_STATE_CONFIG_PATH not in self.env:
            self.env[MODEL_EVAL_STATE_CONFIG_PATH] = str(ms_serviceparam_optimizer_config_path)
                

    def run(self, run_params: Optional[Tuple[OptimizerConfigField, ...]] = None, **kwargs):
        # Start the test
        if self.process_name:
            try:
                self.kill_residual_process(self.process_name)
            except Exception as e:
                logger.error(f"Failed to kill residual process. {e}")
        self.before_run(run_params)
        for i, v in enumerate(self.command):
            if not v.strip():
                continue
            if '-' or '--' not in v:
                continue
            if v in self.command[:i]:
                logger.warning("{} field appears multiple times in the command. please confirm.", v)
        for k, v in self.env.items():
            if isinstance(k, str) and isinstance(v, str):
                continue
            else:
                logger.error(f"Possible Problem with Environment Variable Type. "
                             f"env: {k}={v}, k type: {type(k)}, v type: {type(v)}")
        try:
            self.process = subprocess.Popen(self.command, env=self.env, stdout=self.run_log_fp,
                                            stderr=subprocess.STDOUT, cwd=self.work_path)
        except OSError as e:
            logger.error(f"Failed to run {self.command}. error {e}")
            raise e
        logger.info(f"Start running the command: {' '.join(self.command)}, log file: {self.run_log}")

    def get_log(self):
        output = None
        if not self.run_log:
            return output
        run_log_path = Path(self.run_log)
        if run_log_path.exists():
            try:
                with open_s(run_log_path, "r", encoding="utf-8") as f:
                    f.seek(self.run_log_offset)
                    output = f.read()
                    self.run_log_offset = f.tell()
            except (UnicodeError, OSError) as e:
                logger.error(f"Failed read {self.command} log. error {e}")
        return output

    def check_success(self):
        if self.print_log:
            output = self.get_log()
            logger.debug(output)
        if self.process.poll() is None:
            return False
        elif self.process.poll() == 0:
            return True
        else:
            raise subprocess.SubprocessError(
                f"Failed in run {self.command}. return code: {self.process.returncode}. log: {self.run_log}")

    def stop(self, del_log: bool = True):
        self.run_log_offset = 0
        close_file_fp(self.run_log_fp)
        if del_log and self.run_log:
            remove_file(Path(self.run_log))
        if not self.process:
            return
        _process_state = self.process.poll()
        if _process_state is not None:
            logger.info(f"The program has exited. exit_code: {_process_state}")
            return
        try:
            children = psutil.Process(self.process.pid).children(recursive=True)
            self.process.kill()
            try:
                self.process.wait(10)
            except subprocess.TimeoutExpired:
                self.process.send_signal(9)
            if self.process.poll() is not None:
                logger.debug(f"The {self.process.pid} process has been shut down.")
            else:
                logger.error(f"The {self.process.pid} process shutdown failed.")
            kill_children(children)
        except Exception as e:
            logger.error(f"Failed to stop simulator process. {e}")

    def get_last_log(self, number: int = 5):
        output = None
        if not self.run_log:
            return output
        run_log_path = Path(self.run_log)
        if run_log_path.exists():
            try:
                with open_s(run_log_path, "r", encoding="utf-8") as f:
                    file_lines = f.readlines()
            except (UnicodeError, OSError) as e:
                logger.error(f"Failed read {self.command} log. error {e}")
            number = min(number, len(file_lines))
            output = '\n'.join(file_lines[-number:])
        return output
