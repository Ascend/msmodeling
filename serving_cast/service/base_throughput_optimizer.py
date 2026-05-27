# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod

from typing import Optional

import pandas as pd

from tensor_cast.core.input_generator import generate_inputs, RequestInfo
from tensor_cast.core.model_runner import ModelRunner, ModelRunnerMetrics
from .optimizer_summary import OptimizerSummary
from .utils import AGG_COLUMNS, MAX_ITER_NUMS, OptimizerData


class BaseThroughputOptimizer(ABC):
    """
    Abstract base class for throughput optimization strategies.
    This class provides a framework for optimizing model inference throughput by
    finding the optimal batch size through binary search. Subclasses must implement
    the initialize and get_inference_info methods to support specific optimization
    strategies.
    Attributes:
        name: Identifier for the optimizer strategy, defaults to "base".
    """

    name = "base"

    def __init__(self) -> None:
        self.model_runner: Optional[ModelRunner] = None
        self.num_mtp_tokens: int = 0

    @abstractmethod
    def initialize(self, model_runner: ModelRunner):
        """
        Initialize the optimizer with a model runner instance.
        Args:
            model_runner: The ModelRunner instance used for model inference.
        Note:
            This method should be implemented to set up any required resources
            or configurations for the optimization process.
        """

    @abstractmethod
    def get_inference_info(self, optimizer_data: OptimizerData) -> OptimizerSummary:
        """
        Execute inference and return optimization summary.
        Args:
            optimizer_data: Contains optimization parameters including batch size,
                input length, output length, etc.
        Returns:
            OptimizerSummary containing inference metrics and results.
        Note:
            This method should be implemented to perform model inference with
            the specified batch size and return performance metrics.
        """

    def run(self, optimizer_data: OptimizerData, batch_range: list[int]) -> OptimizerSummary:
        left, right = 1, 512
        result = []
        result_df = pd.DataFrame(columns=AGG_COLUMNS)

        if batch_range:
            if len(batch_range) == 2:
                left, right = batch_range
            elif len(batch_range) == 1:
                right = batch_range[0]
        else:
            for _ in range(MAX_ITER_NUMS):
                optimizer_data.batch_size = right
                summary = self.get_inference_info(optimizer_data)
                if summary.check_early_stop_flag():
                    break
                else:
                    left, right = right, right * 2

        # early_stop
        optimizer_data.batch_size = left
        summary = self.get_inference_info(optimizer_data)
        if summary.check_early_stop_flag():
            return None

        while left <= right:
            mid = (left + right) // 2
            optimizer_data.batch_size = mid
            summary = self.get_inference_info(optimizer_data)
            if summary.check_early_stop_flag():
                right = mid - 1
            else:
                left = mid + 1
                result.append(summary.get_summary_df())

        if result:
            result_df = pd.concat(result, axis=0, ignore_index=True)

        sorted_df = result_df.sort_values(by=["token/s"], ascending=[True]).round(3)

        ret_summary = OptimizerSummary(optimizer_data)
        ret_summary.set_summary_df(sorted_df)

        return ret_summary

    def _get_forward_info(
        self,
        concurrency: int,
        optimizer_data: OptimizerData,
        is_decode: bool,
    ) -> ModelRunnerMetrics:
        if is_decode:
            query_len = self.num_mtp_tokens + 1
            seq_len = optimizer_data.output_length // 2 + optimizer_data.input_length + query_len
        else:
            seq_len = query_len = optimizer_data.get_effective_input_length()

        # avoid print duplicate image input log
        _image_batch_size = None
        if optimizer_data.image_height is not None:
            _image_batch_size = (
                optimizer_data.image_batch_size
                if optimizer_data.image_batch_size is not None
                else optimizer_data.batch_size
            )
        requests = [
            RequestInfo(
                query_len=query_len,
                seq_len=seq_len,
                image_batch_size=_image_batch_size,
                image_height=optimizer_data.image_height,
                image_width=optimizer_data.image_width,
                concurrency=concurrency,
                is_decode=is_decode,
            )
        ]

        runner = self.model_runner
        assert runner is not None, "initialize() must set model_runner"
        metrics = runner.run_inference(requests, generate_inputs_func=generate_inputs)

        return metrics
