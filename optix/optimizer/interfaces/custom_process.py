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
import re
import subprocess
import tempfile
import time
from math import isinf, isnan
from pathlib import Path
from typing import Any, Optional

import psutil
from loguru import logger

from ...config.base_config import (
    CUSTOM_OUTPUT,
    MODEL_EVAL_STATE_CONFIG_PATH,
    ms_serviceparam_optimizer_config_path,
)
from ...deploy_env import materialize_command, resolve_deploy_context
from ...io_utils import open_file
from ..utils import backup, close_file_fp, kill_children, kill_process, remove_file

# Mapping from field name to CLI flag name, used to remove CLI flags when removing invalid values
FIELD_TO_CLI_FLAG = {
    "REQUESTRATE": "--request-rate",
}

# Fields whose values are non-positive (<=0) should be considered invalid and CLI params removed
# Note: non-positive filtering is a semantic constraint for specific fields, not a universal behavior
NON_POSITIVE_INVALID_FIELDS = frozenset(FIELD_TO_CLI_FLAG.keys())


class CustomProcess:
    from ...config.config import OptimizerConfigField

    def __init__(
        self,
        bak_path: Optional[Path] = None,
        command: Optional[list[str]] = None,
        work_path: Optional[Path] = None,
        print_log: bool = False,
        process_name: str = "",
    ):
        self.command = command
        self.bak_path = bak_path
        self.work_path = work_path or os.getcwd()
        self.run_log = None
        self.run_log_offset = None
        self.run_log_fp = None
        self.process = None
        self.print_log = print_log
        self.process_name = process_name
        self._runtime_ctx, self.env = resolve_deploy_context()
        from ...config.constant import ProcessState, Stage

        self._process_stage = ProcessState(stage=Stage.stop)

    @property
    def process_stage(self):
        return self._process_stage

    @process_stage.setter
    def process_stage(self, value):
        if value.stage == self._process_stage.stage:
            return
        self._process_stage = value

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

    def _split_merged_args(self):
        """
        Split merged args into independent parts.
        For example: '--compilation-config \'{"cudagraph_mode": "FULL_DECODE_ONLY"}\''
        Splits into: '--compilation-config' and '{"cudagraph_mode": "FULL_DECODE_ONLY"}'

        This resolves the issue where vllm's argument parser converts underscores in JSON keys to hyphens.
        Compatible with all JSON-like parameter input forms: bare JSON/quoted JSON/escaped JSON/fullwidth symbol JSON.
        Does not rely on hardcoded JSON parameter lists; auto-detects whether to split based on value format.
        """
        import json
        import re

        def clean_json_string(json_str):
            """
            Generic JSON string cleaning: based on syntax only, not coupled to any parameter names
            Handles: escape chars, outer quotes (single/double/fullwidth), fullwidth symbols, extra spaces
            """
            # 1. Restore escaped characters (\\" -> ", \\\\ -> \)
            json_str = json_str.replace('\\"', '"').replace("\\\\", "\\")
            # 2. Remove leading/trailing quotes of all types and extra spaces
            json_str = (
                json_str.strip().strip("'").strip('"').strip("\u2018").strip("\u2019").strip("\u201c").strip("\u201d")
            )
            # 3. Convert fullwidth symbols to halfwidth
            json_str = (
                json_str.replace("\uff0c", ",").replace("\uff1a", ":").replace("\uff08", "(").replace("\uff09", ")")
            )
            return json_str

        def is_json_like(value):
            """
            Determine if string is JSON format (based on syntax features only, no parameter coupling)
            Feature: contains {} and can be parsed as JSON (or can be parsed after cleaning)
            """
            cleaned = clean_json_string(value)
            try:
                parsed = json.loads(cleaned)
                return isinstance(parsed, (dict, list))
            except (json.JSONDecodeError, ValueError, TypeError):
                return False

        new_command = []
        i = 0
        while i < len(self.command):
            cmd_element = self.command[i]
            if not isinstance(cmd_element, str):
                new_command.append(cmd_element)
                i += 1
                continue

            # Match pattern: --param_name space quote...quote
            # Use \S+ to match param name (including dots and other chars)
            match = re.match(r"^(-\S+)\s+", cmd_element)
            if not match:
                new_command.append(cmd_element)
                i += 1
                continue

            param_name = match.group(1)
            rest = cmd_element[match.end() :]

            if not rest:
                new_command.append(cmd_element)
                i += 1
                continue

            # Check if it's JSON format (doesn't depend on hardcoded parameter list)
            if not is_json_like(rest):
                # Non-JSON format, keep as-is
                new_command.append(cmd_element)
                i += 1
                continue

            # Find the first quote
            first_char = rest[0]
            if first_char not in ('"', "'"):
                # No quotes, try to split directly
                cleaned_value = clean_json_string(rest)
                if is_json_like(rest):
                    new_command.append(param_name)
                    new_command.append(cleaned_value)
                    logger.debug(f"[FIX] Split merged arg (no quotes, valid JSON): {param_name} + {cleaned_value}")
                else:
                    new_command.append(cmd_element)
                i += 1
                continue

            # Find the last matching quote
            last_idx = rest.rfind(first_char)
            if last_idx <= 0:
                new_command.append(cmd_element)
                i += 1
                continue

            json_value = rest[1:last_idx]

            # Clean the JSON string
            cleaned_value = clean_json_string(json_value)

            # Try to split, let vllm handle it even if it's not standard JSON
            new_command.append(param_name)
            new_command.append(cleaned_value)
            if is_json_like(json_value):
                logger.debug(f"[FIX] Split merged arg (valid JSON): {param_name} + {cleaned_value}")
            else:
                logger.warning(f"[FIX] Non-standard JSON param (vllm may parse it): {param_name} = {cleaned_value}")
            i += 1

        self.command = new_command

    def backup(self):
        # Backup operation, default to backing up log
        backup(self.run_log, self.bak_path, self.__class__.__name__)

    def before_run(self, run_params: Optional[tuple[OptimizerConfigField, ...]] = None):
        from ...config.config import get_settings

        """
        Preparation work before running command
        Args:
            run_params: tuning parameter list, a tuple, each element defined by value and config_position
        """
        self.run_log_fp, self.run_log = tempfile.mkstemp(prefix="ms_serviceparam_optimizer_")
        self.run_log_offset = 0
        if not run_params:
            if CUSTOM_OUTPUT not in self.env:
                self.env[CUSTOM_OUTPUT] = str(get_settings().output)
            if MODEL_EVAL_STATE_CONFIG_PATH not in self.env:
                self.env[MODEL_EVAL_STATE_CONFIG_PATH] = str(ms_serviceparam_optimizer_config_path)
            if self.command:
                self.command = materialize_command(self.command, self.env, self._runtime_ctx)
            return
        for k in run_params:
            if k.config_position == "env":
                # env type data, set environment variables and update variable references in command, all uppercase when setting
                _env_name = k.name.upper().strip()
                _var_name = f"${_env_name}"

                # Check if value is empty/invalid
                if isinstance(k.value, str):
                    value_flag = k.value is None or not k.value.strip()
                else:
                    value_flag = k.value is None or isnan(k.value) or isinf(k.value)

                if value_flag:
                    # When value is empty, delete from environment, do not set empty value
                    if _env_name in self.env:
                        del self.env[_env_name]
                        logger.debug(f"Removed empty env var: {_env_name}")
                else:
                    # When value is valid, set environment variable
                    self.env[_env_name] = str(k.value)

                # Handle variable references in the command line
                if _var_name not in self.command:
                    continue
                _i = self.command.index(_var_name)
                _cli_flag = FIELD_TO_CLI_FLAG.get(_env_name)
                # Specific fields (e.g. REQUESTRATE) with non-positive values are considered invalid, to avoid causing assertion errors in benchmark
                if not value_flag and isinstance(k.value, (int, float)) and k.value <= 0:
                    if _env_name in NON_POSITIVE_INVALID_FIELDS:
                        value_flag = True
                if value_flag:
                    self.command.pop(_i)
                    if _cli_flag and _i > 0 and self.command[_i - 1] == _cli_flag:
                        self.command.pop(_i - 1)
                else:
                    self.command[_i] = str(k.value)

        # Replace custom variables in the others fields
        # Supports using $VAR_NAME format custom variables in others parameters
        # For example: --speculative-config '{"num_speculative_tokens": $NUM_VAR,"method":"deepseek_mtp"}'
        # Note: this handles all parameters (including config_position="env" ones), because the original code's exact match replacement
        # cannot handle variables nested inside strings (like variables inside JSON format parameters)
        for k in run_params:
            _var_name = f"${k.name.upper().strip()}"
            # Handle string values, don't call isnan/isinf on strings
            if isinstance(k.value, str):
                value_flag = k.value is None or not k.value.strip()
            else:
                value_flag = k.value is None or isnan(k.value) or isinf(k.value)
            if value_flag:
                continue
            # Replace variables in each element of the command (including variables in others fields)
            # Use while loop to ensure all occurrences are replaced (a variable may appear multiple times in one element)
            pattern = re.compile(rf"(?<![A-Z0-9_]){re.escape(_var_name)}(?![A-Z0-9_])")
            for i, cmd_element in enumerate(self.command):
                if isinstance(cmd_element, str):
                    self.command[i] = pattern.sub(str(k.value), cmd_element)

        # Fix: split merged args into independent parts
        # For example: '--compilation-config \'{"cudagraph_mode": "FULL_DECODE_ONLY"}\''
        # Splits into: '--compilation-config' and '{"cudagraph_mode": "FULL_DECODE_ONLY"}'
        self._split_merged_args()

        if CUSTOM_OUTPUT not in self.env:
            # Set output directory
            self.env[CUSTOM_OUTPUT] = str(get_settings().output)
        # Set the json file to read
        if MODEL_EVAL_STATE_CONFIG_PATH not in self.env:
            self.env[MODEL_EVAL_STATE_CONFIG_PATH] = str(ms_serviceparam_optimizer_config_path)

        if self.command:
            self.command = materialize_command(self.command, self.env, self._runtime_ctx)

    def run(self, run_params: Optional[tuple[OptimizerConfigField, ...]] = None, **kwargs):
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
            if "-" not in v and "--" not in v:
                continue
            if v in self.command[:i]:
                logger.warning("{} field appears multiple times in the command. please confirm.", v)
        for k, v in self.env.items():
            if isinstance(k, str) and isinstance(v, str):
                continue
            else:
                logger.error(
                    f"Possible Problem with Environment Variable Type. "
                    f"env: {k}={v}, k type: {type(k)}, v type: {type(v)}"
                )
        from ...config.constant import ProcessState, Stage

        try:
            self.process = subprocess.Popen(
                self.command,
                env=self.env,
                stdout=self.run_log_fp,
                stderr=subprocess.STDOUT,
                cwd=self.work_path,
            )
            self.process_stage = ProcessState(stage=Stage.start)
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
                with open_file(run_log_path, "r", encoding="utf-8", errors="ignore") as f:
                    f.seek(self.run_log_offset)
                    output = f.read()
                    self.run_log_offset = f.tell()
            except (UnicodeError, OSError) as e:
                logger.error(f"Failed read {self.command} log. error {e}")
        return output

    def health(self):
        from ...config.constant import ProcessState, Stage

        """
        Check if the task ran successfully
        Returns: returns bool value, check if the program started successfully
        """
        if self.print_log:
            output = self.get_log()
            logger.debug(output)
        if self.process.poll() is None:
            return ProcessState(stage=Stage.running)
        elif self.process.poll() == 0:
            return ProcessState(stage=Stage.stop)
        else:
            return ProcessState(
                stage=Stage.error,
                info=f"Failed in run {self.command!r}. \
                                        return code: {self.process.returncode}. log: {self.run_log}",
            )

    def stop(self, del_log: bool = True):
        from ...config.constant import ProcessState, Stage

        self.run_log_offset = 0
        close_file_fp(self.run_log_fp)
        if del_log and self.run_log:
            remove_file(Path(self.run_log))
        if not self.process:
            return
        _process_state = self.process.poll()
        if _process_state is not None:
            self.process_stage = ProcessState(stage=Stage.stop)
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
            self.process_stage = ProcessState(stage=Stage.stop)
        except Exception as e:
            logger.error(f"Failed to stop simulator process. {e}")
            self.process_stage = ProcessState(stage=Stage.error, info=f"Failed to stop simulator process. {e}")

    def get_last_log(self, number: int = 5):
        output = None
        if not self.run_log:
            return output
        run_log_path = Path(self.run_log)
        if run_log_path.exists():
            file_lines = []
            encodings_to_try = ["utf-8", "latin-1", "gbk", "cp1252"]

            for encoding in encodings_to_try:
                try:
                    with open_file(run_log_path, "r", encoding=encoding, errors="replace") as f:
                        file_lines = f.readlines()
                    break
                except (UnicodeError, OSError) as e:
                    if encoding == encodings_to_try[-1]:
                        logger.error(f"Failed read {self.command} log after trying all encodings. error {e}")
                    continue
            number = min(number, len(file_lines))
            output = "\n".join(file_lines[-number:])
        return output


class BaseDataField:
    from ...config.config import OptimizerConfigField

    def __init__(self, config: Optional[Any] = None):
        from ...config.config import get_settings

        if config:
            self.config = config
        else:
            settings = get_settings()
            self.config = settings.ais_bench

    @property
    def data_field(self) -> tuple[OptimizerConfigField, ...]:
        """
        Get data field property
        """
        if hasattr(self.config, "target_field") and self.config.target_field:
            return tuple(self.config.target_field)
        return ()

    @data_field.setter
    def data_field(self, value: tuple[OptimizerConfigField] = ()) -> None:
        """
        Provide new data, update and replace data field properties.
        """
        _default_name = []
        if hasattr(self.config, "target_field") and self.config.target_field:
            _default_name = [_f.name for _f in self.config.target_field]
        for _field in value:
            if _field.name not in _default_name:
                continue
            _index = _default_name.index(_field.name)
            self.config.target_field[_index] = _field
