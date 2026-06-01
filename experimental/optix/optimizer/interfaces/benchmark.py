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
from .custom_process import BaseDataField, CustomProcess
from ...config.config import PerformanceIndex

MS_TO_S = 10 ** 3
US_TO_S = 10 ** 6


class BenchmarkInterface(CustomProcess, BaseDataField, ABC):
    """
    Operate benchmark program, test performance.
    """
    @property
    def num_prompts(self) -> int:
        """
        Get the process name property of the service
        Returns:""

        """
        return 0

    @num_prompts.setter
    def num_prompts(self, value):
        """
        Get the process name property of the service
        Returns:""

        """
        pass

    @property
    def model_name(self) -> str:
        """
        Get the current running model name
        Returns:

        """
        return ""

    @property
    def dataset_path(self) -> str:
        """
        Get the current dataset being used
        Returns:

        """
        return ""

    @property
    def max_output_len(self) -> 0:
        """
        Get the current max output length setting
        Returns:
        """
        return 0

    @abstractmethod
    def update_command(self) -> None:
        """
        Update service startup command. Update self.command property.
        Returns: None

        """
        pass

    @abstractmethod
    def get_performance_index(self) -> PerformanceIndex:
        """
        Get performance index
        Returns: index data class

        """
        pass