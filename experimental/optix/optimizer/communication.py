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
import stat
import time
from pathlib import Path
from loguru import logger
from filelock import FileLock
from msguard.security import open_s
 
 
class CustomCommand:
    cmd_eof = "eof"
 
    def __init__(self):
        self._start = "start"
        self._check_success = "check_success"
        self._process_poll = "process_poll"
        self._stop = "stop"
        self._history = []
        self._backup = "backup"
        self._init = "init"
 
    @property
    def backup(self):
        return f"{self._backup} {time.time_ns()}"
 
    @property
    def init(self):
        return f"{self._init} {time.time_ns()}"
 
    @property
    def start(self):
        return f"{self._start} {time.time_ns()}"
 
    @property
    def check_success(self):
        return f"{self._check_success} {time.time_ns()}"
 
    @property
    def process_poll(self):
        return f"{self._process_poll} {time.time_ns()}"
 
    @property
    def stop(self):
        return f"{self._stop} {time.time_ns()}"
 
    @property
    def history(self):
        return tuple(self._history)
 
    @history.setter
    def history(self, value):
        self._history.append(value)
 
    @history.deleter
    def history(self):
        self._history.clear()
 
 
class CommunicationForFile:
    def __init__(self, cmd_file: Path, res_file: Path, timeout=120):
        if not cmd_file.parent.exists():
            cmd_file.parent.mkdir(parents=True, mode=0o750)
        if not res_file.parent.exists():
            res_file.parent.mkdir(parents=True, mode=0o750)
        self.cmd_file = cmd_file
        self.cmd_file_lock = cmd_file.parent.joinpath(f"{cmd_file.name}.lock")
        if not self.cmd_file_lock.exists():
            with open_s(self.cmd_file_lock, "w") as f:
                pass
        self.res_file = res_file
        self.res_file_lock = res_file.parent.joinpath(f"{res_file.name}.lock")
        if not self.res_file_lock.exists():
            with open_s(self.res_file_lock, "w") as f:
                pass
        self.timeout = timeout
 
    def send_command(self, cmd):
        with FileLock(self.cmd_file_lock):
            if self.cmd_file.exists():
                with open_s(self.cmd_file, "w") as fcmd:
                    fcmd.write(cmd)
            else:
                with open_s(self.cmd_file, "w", buffering=1024) as fcmd:
                    fcmd.write(cmd)
 
    def recv_command(self):
        with FileLock(self.res_file_lock):
            if not self.res_file.exists():
                return ''
            with open_s(self.res_file, 'r', encoding="utf-8") as f:
                data = f.read()
        return data
 
    def clear_command(self, command):
        # Confirm whether the command is completed and clean up related commands.
        st = time.perf_counter()
        while True:
            if time.perf_counter() - st > self.timeout:
                raise TimeoutError("Timeout while getting command result. command {}", command)
            time.sleep(1)
            cmd_res = self.recv_command()
            if not cmd_res:
                continue
            if command not in cmd_res:
                continue
            res = cmd_res[len(command) + 1:].strip().lower()
            if res == "done":
                status = "done"
                break
            elif res == "true":
                status = True
                break
            elif res == "false":
                status = False
                break
            elif res == "none":
                status = None
                break
            elif 'error' in res:
                raise ValueError(f"Failed to start the program on another server. info: {cmd_res}")
            else:
                status = res
                break
        self.send_command(CustomCommand.cmd_eof)
        self.clear_res()
        return status
 
    def clear_res(self):
        start_time = time.time()
        timeout = 10  # Set timeout to 10 seconds
        while True:
            time.sleep(1)
            data = self.recv_command()
            if data.strip().lower() == CustomCommand.cmd_eof:
                self.send_command(CustomCommand.cmd_eof)
                break
            # Check if timed out
            if time.time() - start_time > timeout:
                # Timeout handling, e.g. send error message or exit loop
                logger.error("EOF response not received, timeout exit")
                break