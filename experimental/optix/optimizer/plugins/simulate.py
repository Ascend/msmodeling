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
import tempfile
import time
from copy import deepcopy
from typing import Any, Optional, Tuple
import shutil
from dataclasses import dataclass

import yaml
from loguru import logger
from msguard.security import open_s

from ..interfaces.simulator import SimulatorInterface
from ..utils import remove_file, close_file_fp, backup
from ...config.custom_command import VllmCommand, MindieCommand
from ...config.config import get_settings, OptimizerConfigField, VllmConfig, MindieConfig, KubectlConfig
from ...config.constant import ProcessState, Stage

"""
Mindie simulation engine - provides interfaces for starting/stopping mindie simulation services.

The LocalSimulate class provides comprehensive simulation capabilities for running mindie models
locally, including configuration management, port resolution, and multi-instance support.
"""


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
        with open_s(self.config.config_path, "r") as f:
            data = json.load(f)
        self.default_config = data
        logger.debug(
            f"config bak path {self.config.config_bak_path}",
        )
        if self.config.config_bak_path.exists():
            self.config.config_bak_path.unlink()
        with open_s(self.config.config_bak_path, 'w') as fout:
            json.dump(self.default_config, fout, indent=4)
        self.command = MindieCommand(self.config.command).command

    @property
    def base_url(self) -> str:
        """
        Get the base url property of the service
        Returns:

        """
        pass

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
                context.origin_config[context.cur_key], context.next_level, context.value, context.current_depth
            )
        elif Simulator.is_int(context.cur_key):
            raise KeyError(f"data: {context.origin_config}, key: {context.cur_key}")
        elif Simulator.is_int(context.next_key):
            context.origin_config[context.cur_key] = []
            Simulator.set_config(
                context.origin_config[context.cur_key], context.next_level, context.value, context.current_depth
            )
        else:
            context.origin_config[context.cur_key] = {}
            Simulator.set_config(
                context.origin_config[context.cur_key], context.next_level, context.value, context.current_depth
            )

    @staticmethod
    def set_config_for_list(context: ConfigContextlist):
        if len(context.origin_config) > int(context.cur_key):
            Simulator.set_config(
                context.origin_config[int(context.cur_key)], context.next_level, context.value, context.current_depth
            )
        elif len(context.origin_config) == int(context.cur_key) and Simulator.is_int(context.next_key):
            context.origin_config.append([])
            Simulator.set_config(
                context.origin_config[int(context.cur_key)], context.next_level, context.value, context.current_depth
            )
        elif len(context.origin_config) == int(context.cur_key) and not Simulator.is_int(context.next_key):
            context.origin_config.append({})
            Simulator.set_config(
                context.origin_config[int(context.cur_key)], context.next_level, context.value, context.current_depth
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
                    else:
                        origin_config.append(value)
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

    def update_command(self):
        self.command = MindieCommand(self.config.command).command

    def before_run(self, run_params: Optional[Tuple[OptimizerConfigField]] = None):
        self.update_config(run_params)
        super().before_run(run_params)

        pkill_path = shutil.which("pkill")
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

        npu_smi_path = shutil.which("npu-smi")
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

    def update_config(self, params: Optional[Tuple[OptimizerConfigField]] = None):
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
        with open_s(self.config.config_path, "w") as fout:
            json.dump(new_config, fout, indent=4, ensure_ascii=False)

    def stop(self, del_log: bool = True):
        remove_file(self.config.config_path)
        with open_s(self.config.config_path, "w") as fout:
            json.dump(self.default_config, fout, indent=4, ensure_ascii=False)
        super().stop(del_log)


class VllmSimulator(SimulatorInterface):
    def __init__(self, config: Optional[VllmConfig] = None, *args, **kwargs):
        if config:
            self.config = config
        else:
            settings = get_settings()
            self.config = settings.vllm
        super().__init__(*args, process_name=self.config.process_name, **kwargs)

        self.command = VllmCommand(self.config.command).command

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

        pkill_path = shutil.which("pkill")
        if not pkill_path:
            logger.error("pkill command not found in PATH")
            return False

        signals = ["-15", "-9"]  # SIGTERM, SIGKILL

        for attempt in range(1, max_attempts + 1):
            for signal in signals:
                if not self._is_vllm_running():
                    logger.info("vllm process has been terminated")
                    return True

                logger.debug(f"Attempt {attempt}/{max_attempts}: sending signal {signal} to vllm")
                try:
                    result = subprocess.run(
                        [pkill_path, signal, "-f", "vllm serve"],
                        stderr=subprocess.STDOUT,
                        stdout=subprocess.PIPE,
                        text=True,
                        timeout=10,
                    )
                    if result.returncode == 0 or result.returncode == 1:
                        logger.debug(f"Signal {signal} sent successfully (rc={result.returncode})")
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
        pgrep_path = shutil.which("pgrep")
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
            )
            count = int((result.stdout or "0").strip() or "0")
            return count > 0
        except (OSError, subprocess.SubprocessError, ValueError):
            return False

    def _wait_for_process_exit(self, timeout: int) -> bool:
        """Wait for process to exit, returns whether exit was successful"""
        import time

        start = time.time()
        while time.time() - start < timeout:
            if not self._is_vllm_running():
                return True
            time.sleep(0.5)
        return False

    def _log_residual_processes(self):
        """Log residual process info for diagnostics"""
        pgrep_path = shutil.which("pgrep")
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
            )
            if result.stdout:
                logger.warning(f"Residual vllm processes:\n{result.stdout}")
        except subprocess.SubprocessError:
            pass

    def update_command(self):
        self.command = VllmCommand(self.config.command).command


class DisaggregationSimulator(SimulatorInterface):
    from ...config.custom_command import KubectlCommand

    def __init__(self, *args, config: Optional[KubectlConfig] = None, **kwargs):
        if config:
            self.config = config
        else:
            settings = get_settings()
            self.config = settings.kubectl
            self.config.target_field = settings.mindie.target_field
        super().__init__(*args, process_name=self.config.process_name, **kwargs)
        if not self.config.config_single_path.exists():
            raise FileNotFoundError(self.config.config_single_path)
        with open_s(self.config.config_single_path, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                logger.error(f"Failed in read config.json. file: {self.config.config_single_path}")
        with open_s(self.config.config_single_pd_path, "r") as f:
            try:
                pd_data = json.load(f)
            except json.JSONDecodeError:
                logger.error(f"Failed in read ms_controller.json. file: {self.config.config_single_pd_path}")
        self.default_pd_config = pd_data
        self.default_config = data
        logger.debug(
            f"config bak path {self.config.config_single_bak_path!r}",
        )
        if self.config.config_single_bak_path.exists():
            self.config.config_single_bak_path.unlink()
        with open_s(self.config.config_single_bak_path, "w") as fout:
            json.dump(self.default_config, fout, indent=4)
        self.run_log = None
        self.mindie_log_offset = 0
        self.mindie_log_fp = None
        self.process = None
        self.command = self.KubectlCommand(self.config.command).command
        self.log_command = self.KubectlCommand(self.config.command).log_command
        self.monitor_command = self.KubectlCommand(self.config.command).monitor_command

    @property
    def base_url(self) -> str:
        """
        Get the base url property of the service
        Returns:

        """
        pass

    @staticmethod
    def set_config_for_dict(context: ConfigContextdict):
        if context.cur_key in context.origin_config:
            DisaggregationSimulator.set_config(
                context.origin_config[context.cur_key], context.next_level, context.value, context.current_depth
            )
        elif DisaggregationSimulator.is_int(context.cur_key):
            raise KeyError(f"data: {context.origin_config}, key: {context.cur_key}")
        elif DisaggregationSimulator.is_int(context.next_key):
            context.origin_config[context.cur_key] = []
            DisaggregationSimulator.set_config(
                context.origin_config[context.cur_key], context.next_level, context.value, context.current_depth
            )
        else:
            context.origin_config[context.cur_key] = {}
            DisaggregationSimulator.set_config(
                context.origin_config[context.cur_key], context.next_level, context.value, context.current_depth
            )

    @staticmethod
    def set_config_for_list(context: ConfigContextlist):
        if len(context.origin_config) > int(context.cur_key):
            DisaggregationSimulator.set_config(
                context.origin_config[int(context.cur_key)], context.next_level, context.value, context.current_depth
            )
        elif len(context.origin_config) == int(context.cur_key) and DisaggregationSimulator.is_int(context.next_key):
            context.origin_config.append([])
            DisaggregationSimulator.set_config(
                context.origin_config[int(context.cur_key)], context.next_level, context.value, context.current_depth
            )
        elif len(context.origin_config) == int(context.cur_key) and not DisaggregationSimulator.is_int(
            context.next_key
        ):
            context.origin_config.append({})
            DisaggregationSimulator.set_config(
                context.origin_config[int(context.cur_key)], context.next_level, context.value, context.current_depth
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
                    else:
                        origin_config.append(value)
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
            DisaggregationSimulator.set_config_for_dict(context)
        elif isinstance(origin_config, list):
            context = ConfigContextlist(
                origin_config=origin_config,
                cur_key=_cur_key,
                next_key=_next_key,
                next_level=next_level,
                value=value,
                current_depth=current_depth + 1,
            )
            DisaggregationSimulator.set_config_for_list(context)
        else:
            raise ValueError(f"Not Support type {type(origin_config)}")

    @staticmethod
    def is_int(x):
        try:
            int(x)
            return True
        except ValueError:
            return False

    def prepare_before_start_server(self):
        bash_path = shutil.which("bash")
        if bash_path is not None:
            subprocess.run([bash_path, self.config.delete_path, "mindie", "."], cwd=self.config.kubectl_default_path)
            while True:
                signal = True
                proc = subprocess.run(
                    self.log_command, stdout=subprocess.PIPE, text=True, cwd=self.config.kubectl_default_path
                )
                lines = proc.stdout.splitlines()
                for line in lines:
                    if line.startswith('mindie'):
                        signal = False
                if signal is True:
                    break
                time.sleep(1)
        else:
            logger.error("bash not found in path")

    def backup(self):
        super().backup()
        backup(self.config.config_path, self.bak_path, self.__class__.__name__)

    def update_config(self, params: Tuple[OptimizerConfigField]):
        new_config = deepcopy(self.default_config)
        pd_config = deepcopy(self.default_pd_config)
        for p in params:
            if p.config_position.startswith("default"):
                DisaggregationSimulator.set_config(pd_config, p.config_position, p.value)
            if not p.config_position.startswith("BackendConfig"):
                continue
            DisaggregationSimulator.set_config(new_config, p.config_position, p.value)

        logger.debug(f"new config {new_config}")
        if self.config.config_single_path.exists():
            self.config.config_single_path.unlink()
        with open_s(self.config.config_single_path, "w") as fout:
            json.dump(new_config, fout, indent=4)
        if self.config.config_single_pd_path.exists():
            self.config.config_single_pd_path.unlink()
        with open_s(self.config.config_single_pd_path, "w") as fout:
            json.dump(pd_config, fout, indent=4)

    def update_command(self):
        self.command = self.KubectlCommand(self.config.command).command
        self.log_command = self.KubectlCommand(self.config.command).log_command
        self.monitor_command = self.KubectlCommand(self.config.command).monitor_command

    def test_curl(self):
        import requests

        logger.info(f"kubectl_single_path: {self.config.kubectl_single_path}")
        curl_port = None
        yaml_dir = self.config.kubectl_single_path.parent
        yaml_path = os.path.join(yaml_dir, "deployment/mindie_service_single_container.yaml")
        logger.info(f"yaml_path: {yaml_path}")
        with open_s(yaml_path, 'r') as file:
            all_documents = yaml.safe_load_all(file)
            for doc in all_documents:
                if 'spec' in doc and 'ports' in doc['spec']:
                    ports = doc['spec']['ports']
                    for port in ports:
                        if 'nodePort' in port:
                            curl_port = port['nodePort']
        if curl_port:
            url = f"http://localhost:{curl_port}"
        else:
            raise ("cannot find port from mindie_service_single_container.yaml, please check")

        payload = {
            "inputs": "Please introduce yourself.",
            "parameters": {
                "max_new_tokens": 20,
                "temperature": 0.3,
                "top_p": 0.3,
                "top_k": 5,
                "do_sample": True,
                "repetition_penalty": 1.05,
                "seed": 128,
            },
        }

        headers = {"Content-Type": "application/json"}

        try:
            response = requests.post(url, json=payload, headers=headers, timeout=10)

            if response.status_code == 200:
                return True
            else:
                return False
        except requests.exceptions.RequestException:
            return False

    def health(self):
        process_res = ProcessState()
        process_res.stage = Stage.error
        with open_s(self.run_log, "r") as f:
            try:
                f.seek(self.mindie_log_offset)
                output = f.read()
                self.mindie_log_offset = f.tell()
            except Exception as e:
                logger.warning(f"Failed in read mindie log. error: {e}")
        if output:
            logger.debug(f"simulate out: \n{output}")
            if "MindIE-MS coordinator is ready!!!" in output:
                while True:
                    if self.test_curl() is True:
                        process_res.stage = Stage.running
                        return process_res
                    time.sleep(1)
        return process_res

    def start_server(self, run_params: Tuple[OptimizerConfigField]):
        super().before_run(run_params)
        self.prepare_before_start_server()
        self.mindie_log_fp, self.run_log = tempfile.mkstemp(prefix="ms_serviceparam_optimizer_mindie")
        self.mindie_log_offset = 0
        logger.info(f"start running the command: {self.command}")
        self.process = subprocess.run(self.command, env=self.env, text=True, cwd=self.config.kubectl_default_path)
        logger.info(f"self.log_command: {self.log_command}")
        while True:
            proc = subprocess.run(
                self.log_command, stdout=subprocess.PIPE, text=True, cwd=self.config.kubectl_default_path
            )

            lines = proc.stdout.splitlines()
            for line in lines:
                if line.startswith('mindie'):
                    mindie_name = line.split()
                    break
            if mindie_name[3] == 'Running':
                break
            time.sleep(1)
        kubectl_monitor_command = self.KubectlCommand(self.config.command).monitor_command
        kubectl_monitor_command.append(mindie_name[1])
        logger.debug(f"mindie_name: {mindie_name[1]}")
        self.log_process = subprocess.Popen(
            kubectl_monitor_command,
            stdout=self.mindie_log_fp,
            stderr=subprocess.STDOUT,
            env=self.env,
            text=True,
            cwd=self.config.kubectl_default_path,
        )
        logger.info(f"Start running the command: {' '.join(kubectl_monitor_command)}, log file: {self.run_log}")

    def run(self, run_params: Tuple[OptimizerConfigField]):
        logger.info(f'Start running in simulator. params are: {run_params}')
        self.update_config(run_params)
        self.start_server(run_params)

    def stop(self, del_log=True):
        logger.debug("Stop simulator process")
        close_file_fp(self.mindie_log_fp)
        if del_log:
            remove_file(self.run_log)
        self.mindie_log_offset = 0
        try:
            bash_path = shutil.which("bash")
            if bash_path is not None:
                subprocess.run([bash_path, self.config.delete_path, "mindie", "./"])
            else:
                logger.error("bash not found in path")
        except Exception as e:
            logger.error(f"Failed to stop simulator process. {e}")
