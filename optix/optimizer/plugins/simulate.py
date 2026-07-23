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
import shutil
import subprocess
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Optional

from loguru import logger

from ...config.config import (
    MindieConfig,
    OptimizerConfigField,
    VllmConfig,
    get_settings,
)
from ...config.constant import Stage
from ...config.custom_command import VllmCommand
from ...deploy_env import materialize_command, resolve_mindie_argv
from ...io_utils import open_file
from ..interfaces.simulator import SimulatorInterface
from ..utils import backup, remove_file

"""
Mindie simulation engine - provides interfaces for starting/stopping mindie simulation services.

The LocalSimulate class provides comprehensive simulation capabilities for running mindie models
locally, including configuration management, port resolution, and multi-instance support.
"""


def _find_executable(name: str, env: dict[str, str]) -> str | None:
    return shutil.which(name, path=env.get("PATH") or None)


@dataclass
class ConfigContextdict:
    origin_config: dict
    cur_key: str
    next_key: str
    next_level: str
    value: Any
    current_depth: int


@dataclass
class ConfigContextlist:
    origin_config: list
    cur_key: str
    next_key: str
    next_level: str
    value: Any
    current_depth: int


class Simulator(SimulatorInterface):
    def __init__(self, *args, config: Optional[MindieConfig] = None, **kwargs):
        if config:
            self.config = config
        else:
            settings = get_settings()
            self.config = settings.mindie
        super().__init__(*args, process_name=self.config.process_name, **kwargs)
        logger.debug(
            f"config path {self.config.config_path}",
        )
        if not self.config.config_path.exists():
            raise FileNotFoundError(self.config.config_path)
        with open_file(self.config.config_path, "r") as f:
            data = json.load(f)
        self.default_config = data
        logger.debug(
            f"config bak path {self.config.config_bak_path}",
        )
        if self.config.config_bak_path.exists():
            self.config.config_bak_path.unlink()
        with open_file(self.config.config_bak_path, "w") as fout:
            json.dump(self.default_config, fout, indent=4)
        self.update_command()

    @property
    def base_url(self) -> str:
        """
        Get the base url property of the service
        Returns:

        """

    @staticmethod
    def is_int(x):
        try:
            int(x)
            return True
        except ValueError:
            return False

    @staticmethod
    def set_config_for_dict(context: ConfigContextdict):
        if context.cur_key in context.origin_config:
            Simulator.set_config(
                context.origin_config[context.cur_key],
                context.next_level,
                context.value,
                context.current_depth,
            )
        elif Simulator.is_int(context.cur_key):
            raise KeyError(f"data: {context.origin_config}, key: {context.cur_key}")
        elif Simulator.is_int(context.next_key):
            context.origin_config[context.cur_key] = []
            Simulator.set_config(
                context.origin_config[context.cur_key],
                context.next_level,
                context.value,
                context.current_depth,
            )
        else:
            context.origin_config[context.cur_key] = {}
            Simulator.set_config(
                context.origin_config[context.cur_key],
                context.next_level,
                context.value,
                context.current_depth,
            )

    @staticmethod
    def set_config_for_list(context: ConfigContextlist):
        if len(context.origin_config) > int(context.cur_key):
            Simulator.set_config(
                context.origin_config[int(context.cur_key)],
                context.next_level,
                context.value,
                context.current_depth,
            )
        elif len(context.origin_config) == int(context.cur_key) and Simulator.is_int(context.next_key):
            context.origin_config.append([])
            Simulator.set_config(
                context.origin_config[int(context.cur_key)],
                context.next_level,
                context.value,
                context.current_depth,
            )
        elif len(context.origin_config) == int(context.cur_key) and not Simulator.is_int(context.next_key):
            context.origin_config.append({})
            Simulator.set_config(
                context.origin_config[int(context.cur_key)],
                context.next_level,
                context.value,
                context.current_depth,
            )
        else:
            raise IndexError(f"data: {context.origin_config}, index: {context.cur_key}")

    @staticmethod
    def set_config(origin_config, key: str, value: Any, current_depth=0):
        if current_depth > 10:
            raise RecursionError("Exceeded maximum recursion depth")
        next_level = None
        try:
            if "." in key:
                _f_index = key.index(".")
                _cur_key, next_level = key[:_f_index], key[_f_index + 1 :]
            else:
                _cur_key = key
            if next_level is None:
                if isinstance(origin_config, dict):
                    origin_config[_cur_key] = value
                elif isinstance(origin_config, list):
                    if len(origin_config) > int(_cur_key):
                        origin_config[int(_cur_key)] = value
                    elif len(origin_config) == int(_cur_key):
                        origin_config.append(value)
                    else:
                        raise IndexError(
                            f"Cannot set index {_cur_key} on list of length "
                            f"{len(origin_config)}. Index must be <= current "
                            f"length (append) or < length (overwrite)."
                        )
                return
            if "." in next_level:
                _next_index = next_level.index(".")
                _next_key = next_level[:_next_index]
            elif next_level:
                _next_key = next_level
            else:
                _next_key = None
        except Exception as e:
            logger.error(f"Unexpected error occurred at {key}")
            raise e
        if isinstance(origin_config, dict):
            context = ConfigContextdict(
                origin_config=origin_config,
                cur_key=_cur_key,
                next_key=_next_key,
                next_level=next_level,
                value=value,
                current_depth=current_depth + 1,
            )
            Simulator.set_config_for_dict(context)
        elif isinstance(origin_config, list):
            context = ConfigContextlist(
                origin_config=origin_config,
                cur_key=_cur_key,
                next_key=_next_key,
                next_level=next_level,
                value=value,
                current_depth=current_depth + 1,
            )
            Simulator.set_config_for_list(context)
        else:
            raise ValueError(f"Not Support type {type(origin_config)}")

    def update_command(self) -> None:
        raw_command = resolve_mindie_argv(self.env)
        self.command = materialize_command(raw_command, self.env, self._runtime_ctx, cwd=self.work_path)

    def before_run(self, run_params: Optional[tuple[OptimizerConfigField]] = None):
        self.update_config(run_params)
        super().before_run(run_params)

        pkill_path = _find_executable("pkill", self.env)
        if not pkill_path:
            logger.warning("pkill command not found in PATH")
            return
        subprocess.run(
            [pkill_path, "-9", "mindie"],
            env=self.env,
            stdout=self.run_log_fp,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=self.work_path,
        )

        npu_smi_path = _find_executable("npu-smi", self.env)
        if not npu_smi_path:
            logger.warning("npu-smi command not found in PATH")
            return
        subprocess.run(
            [npu_smi_path, "info"],
            env=self.env,
            stdout=self.run_log_fp,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=self.work_path,
        )

    def backup(self):
        super().backup()
        backup(self.config.config_path, self.bak_path, self.__class__.__name__)

    def health(self):
        """
        Get the current service status.
        Current implementation based on vllm url
        Returns: None

        """
        process_res = super().health()
        if process_res.stage != Stage.running:
            proxy_status = super(SimulatorInterface, self).health()
            self.run_log_offset = 0
            output = self.get_log()
            if output and "Daemon start success!" in output and proxy_status.stage == Stage.running:
                return proxy_status
        return process_res

    def update_config(self, params: Optional[tuple[OptimizerConfigField]] = None):
        if not params:
            return
        new_config = deepcopy(self.default_config)
        for p in params:
            if not p.config_position.startswith("BackendConfig"):
                continue
            Simulator.set_config(new_config, p.config_position, p.value)

        logger.debug(f"new config {new_config}")
        if self.config.config_path.exists():
            self.config.config_path.unlink()
        with open_file(self.config.config_path, "w") as fout:
            json.dump(new_config, fout, indent=4, ensure_ascii=False)

    def stop(self, del_log: bool = True):
        remove_file(self.config.config_path)
        with open_file(self.config.config_path, "w") as fout:
            json.dump(self.default_config, fout, indent=4, ensure_ascii=False)
        super().stop(del_log)


class VllmSimulator(SimulatorInterface):
    required_executable = "vllm"

    def __init__(self, config: Optional[VllmConfig] = None, *args, **kwargs):
        if config:
            self.config = config
        else:
            settings = get_settings()
            self.config = settings.vllm
        super().__init__(*args, process_name=self.config.process_name, **kwargs)

        self.update_command()

    @property
    def base_url(self) -> str:
        """
        Get the base url property of the service
        Returns:

        """
        return f"http://localhost:{self.config.command.port}/health"

    def stop(self, del_log: bool = True):
        """
        Stop vllm service, with retry and progressive kill mechanism

        Args:
            del_log: whether to delete log files
        """
        self._stop_vllm_process()
        super().stop(del_log)

    def _stop_vllm_process(self, max_attempts: int = 3, timeout: int = 5) -> bool:
        """
        Stop vllm process, prefer PID-based targeted recycling, fallback to pkill

        Args:
            max_attempts: max attempts (only for pkill fallback strategy)
            timeout: wait timeout in seconds after each attempt

        Returns:
            True if all processes stopped, False if residual remains
        """
        if self.process and self.process.poll() is None:
            try:
                import psutil

                parent = psutil.Process(self.process.pid)
                children = parent.children(recursive=True)

                for child in children:
                    try:
                        child.terminate()
                    except psutil.NoSuchProcess:
                        pass

                parent.terminate()

                gone, alive = psutil.wait_procs([parent] + children, timeout=timeout)

                for p in alive:
                    try:
                        p.kill()
                    except psutil.NoSuchProcess:
                        pass

                psutil.wait_procs(alive, timeout=timeout)

                if not self._is_vllm_running():
                    logger.info(f"vllm process (pid={self.process.pid}) terminated via targeted stop")
                    return True

            except (psutil.NoSuchProcess, psutil.AccessDenied, OSError) as e:
                logger.warning(f"Targeted stop failed, fallback to pkill: {e}")
            except Exception as e:
                logger.warning(f"Unexpected error in targeted stop, fallback to pkill: {e}")

        pkill_path = _find_executable("pkill", self.env)
        if not pkill_path:
            logger.error("pkill command not found in PATH")
            return False

        signals = ["-15", "-9"]  # SIGTERM, SIGKILL

        for _attempt in range(1, max_attempts + 1):
            for signal in signals:
                if not self._is_vllm_running():
                    logger.info("vllm process has been terminated")
                    return True

                try:
                    subprocess.run(
                        [pkill_path, signal, "-f", "vllm serve"],
                        stderr=subprocess.STDOUT,
                        stdout=subprocess.PIPE,
                        text=True,
                        timeout=10,
                        check=False,
                        env=self.env,
                    )
                except subprocess.SubprocessError as e:
                    logger.warning(f"Failed to send signal {signal} to vllm: {e}")

                if self._wait_for_process_exit(timeout):
                    logger.info(f"vllm process terminated after signal {signal}")
                    return True

        if self._is_vllm_running():
            logger.error(f"Failed to stop vllm process after {max_attempts} attempts")
            self._log_residual_processes()
            return False

        return True

    def _is_vllm_running(self) -> bool:
        """Check if vllm process is running"""
        pgrep_path = _find_executable("pgrep", self.env)
        if not pgrep_path:
            logger.warning("pgrep command not found in PATH")
            return False
        try:
            result = subprocess.run(
                [pgrep_path, "-c", "-f", "vllm serve"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=5,
                env=self.env,
            )
            count = int((result.stdout or "0").strip() or "0")
            return count > 0
        except (OSError, subprocess.SubprocessError, ValueError):
            return False

    def _wait_for_process_exit(self, timeout: int) -> bool:
        """Wait for process to exit, returns whether exit was successful"""
        start = time.time()
        while time.time() - start < timeout:
            if not self._is_vllm_running():
                return True
            time.sleep(0.5)
        return False

    def _log_residual_processes(self):
        """Log residual process info for diagnostics"""
        pgrep_path = _find_executable("pgrep", self.env)
        if not pgrep_path:
            logger.warning("pgrep command not found in PATH")
            return

        try:
            result = subprocess.run(
                [pgrep_path, "-a", "-f", "vllm serve"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=5,
                env=self.env,
            )
            if result.stdout:
                logger.warning(f"Residual vllm processes:\n{result.stdout}")
        except subprocess.SubprocessError:
            pass

    def update_command(self) -> None:
        raw_command = VllmCommand(self.config.command).command
        self.command = materialize_command(raw_command, self.env, self._runtime_ctx, cwd=self.work_path)
