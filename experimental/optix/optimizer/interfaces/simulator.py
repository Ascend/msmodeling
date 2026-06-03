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

from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Tuple, Optional

import requests

from .custom_process import BaseDataField, CustomProcess
from ...config.config import OptimizerConfigField
from ...config.constant import ProcessState, Stage


class SimulatorInterface(CustomProcess, BaseDataField, ABC):
    """
    Operate service framework. Used to operate service-related functions.
    """

    @property
    @abstractmethod
    def base_url(self) -> str:
        """
        Get the base url property of the service
        Returns:

        """
        pass

    @abstractmethod
    def update_command(self) -> None:
        """
        Update service startup command. Update self.command property.
        Returns: None

        """
        pass

    def update_config(self, params: Optional[Tuple[OptimizerConfigField]] = None) -> bool:
        """
        Update service config file or other config based on params. Modify config file before service
        startup based on passed parameter values so new config takes effect.
        Args:
            params: tuning parameter list, a tuple, each element defined by value and config_position.

        Returns: None

        """
        return True

    def stop(self, del_log: bool = True):
        """
        Runtime, other preparation work.
        Returns:

        """
        super().stop(del_log)

    def health(self) -> ProcessState:
        """
        Get the current service status.
        Current implementation based on vllm url
        Returns: None

        """
        last_process_stage = self.process_stage
        process_res = super().health()
        if process_res.stage == Stage.error:
            return process_res
        try:
            res = requests.get(self.base_url, timeout=10)
        except requests.exceptions.RequestException as e:
            if last_process_stage.stage == Stage.start:
                return ProcessState(stage=Stage.start, info=str(e))
            return ProcessState(stage=Stage.error, info=str(e))
        else:
            if res.status_code == 200:
                return ProcessState(stage=Stage.running)
            else:
                return ProcessState(stage=Stage.error, info=f"return code {res.status_code}. text {res.text}")

    @contextmanager
    def enable_simulation_model(self):
        """
        Start using simulation model for inference instead of real model.
        Returns: None

        """
        # Enable simulation model instead of real model
        yield True
        # Disable simulation model instead of real model
