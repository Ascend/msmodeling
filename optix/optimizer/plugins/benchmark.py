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
import re
from pathlib import Path
import subprocess
from typing import Optional, Tuple
import glob
from loguru import logger
import pandas as pd
from ...config.base_config import MINDIE_BENCHMARK_PERF_COLUMNS
from ...config.config import (
    AisBenchConfig,
    VllmBenchmarkConfig,
    get_settings,
    PerformanceIndex,
    OptimizerConfigField,
)
from ...config.custom_command import AisBenchCommand, VllmBenchmarkCommand
from ...io_utils import open_file, walk_files
from ...optimizer.interfaces.benchmark import BenchmarkInterface

from ...optimizer.utils import backup, remove_file


MS_TO_S = 10**3
US_TO_S = 10**6


def parse_result(res):
    if isinstance(res, str):
        _res = res.strip().split()
        if len(_res) > 1:
            if _res[1].strip().lower() == "ms":
                return float(_res[0]) / MS_TO_S
            elif _res[1].strip().lower() == "us":
                return float(_res[0]) / US_TO_S
            else:
                return float(_res[0])
        return float(res)
    return res


class AisBench(BenchmarkInterface):
    def __init__(self, *args, config: Optional[AisBenchConfig] = None, **kwargs):
        if config:
            self.config = config
        else:
            settings = get_settings()
            self.config = settings.ais_bench
        super().__init__(*args, **kwargs)
        self.work_path = self.config.work_path
        self.update_command()
        self.model_config_path = self.get_models_config_path()
        with open_file(self.model_config_path, "r", encoding="utf-8") as f:
            self.default_data = f.read()
        self.mindie_benchmark_perf_columns = [k.lower().strip() for k in MINDIE_BENCHMARK_PERF_COLUMNS]

    @property
    def num_prompts(self) -> int:
        """
        Get the process name property of the service
        Returns:""

        """
        return self.config.command.num_prompts

    @num_prompts.setter
    def num_prompts(self, value):
        """
        Get the process name property of the service
        Returns:""

        """
        self.config.command.num_prompts = value

    def update_command(self):
        self.command = AisBenchCommand(self.config.command).command

    def get_models_config_path(self):
        cmd = [self.command[0], "--models", self.config.command.models, "--search"]
        res = subprocess.run(cmd, text=True, capture_output=True)
        if res.returncode != 0:
            raise ValueError(f"The command {cmd} execution failed, with an exit code of {res.returncode}")
        _output = res.stdout
        if not _output:
            _output = res.stderr
        for _line in _output.split("\n"):
            if "--models" not in _line:
                continue
            _lines = _line.strip().split()
            if len(_lines) != 7:
                raise ValueError(
                    f"The expected data format is Task Type │ Task Name │ Config File Path. But get data is {_lines}"
                )
            config_path = Path(_lines[-2].strip())
            return config_path
        raise ValueError(
            f"The expected data format is Task Type │ Task Name │ Config File Path. But get data is {_output}"
        )

    def backup(self, del_log=True):
        backup(self.config.output_path, self.bak_path, self.__class__.__name__)
        if not del_log:
            backup(self.run_log, self.bak_path, self.__class__.__name__)

    def get_performance_metric(self, metric_name: str, algorithm: str = "average"):
        output_path = Path(self.config.output_path)
        result_files = glob.glob(f"{output_path}/*/performances/*/*.csv")
        if len(result_files) != 1:
            logger.error(
                f"The ais bench result for csv files are not unique, result files {result_files}; "
                f"output path: {output_path}. please check"
            )
        metric_name = metric_name.lower().strip()
        algorithm = algorithm.strip().lower()
        if algorithm not in self.mindie_benchmark_perf_columns:
            raise ValueError(
                f"The {algorithm} does not support it; only {self.mindie_benchmark_perf_columns} are supported."
            )
        for file in result_files:
            df = pd.read_csv(file)
            _all_metrics = [k.strip().lower() for k in df["Performance Parameters"].tolist()]
            if metric_name not in _all_metrics:
                continue
            _i = _all_metrics.index(metric_name)
            _columns = [k.lower().strip() for k in df.columns]
            _col_index = _columns.index(algorithm)
            _res = df.iloc[_i, _col_index]
            if not _res:
                continue
            return parse_result(_res)
        raise ValueError(f"Not Found value.  metric_name {metric_name}, algorithm: {algorithm}")

    def get_best_concurrency(self):
        output_path = Path(self.config.output_path)
        csv_files = glob.glob(f"{output_path}/*/performances/*/*.csv")
        if len(csv_files) != 1:
            logger.error(
                f"The ais bench result for csv files are not unique, result files {csv_files}; "
                f"output path: {output_path}. please check"
            )
        dir_path = os.path.dirname(csv_files[0])
        file_name = os.path.splitext(os.path.basename(csv_files[0]))[0]
        # Generate the JSON file path
        json_file = os.path.join(dir_path, f"{file_name}.json")

        with open_file(json_file, "r") as f:
            try:
                data = json.load(f)
            except json.decoder.JSONDecodeError as e:
                raise ValueError(
                    f"JSON file format error, cannot find concurrency value. File path: {json_file}"
                ) from e
        _concurrency = float(data["Concurrency"]["total"])
        _concurrency *= self.config.best_concurrency_coefficient
        _max_concurrency = float(data["Max Concurrency"]["total"])
        if _concurrency < self.config.best_concurrency_threshold:
            best_concurrency = self.config.best_concurrency_threshold
        else:
            best_concurrency = int(min(_concurrency, _max_concurrency))
        return best_concurrency

    def get_performance_index(self):
        output_path = Path(self.config.output_path)
        performance_index = PerformanceIndex()
        if not output_path.exists():
            logger.error(f"the output of aisbench is not find: {output_path}")
        performance_index.time_to_first_token = self.get_performance_metric(
            self.config.performance_config.time_to_first_token.metric,
            self.config.performance_config.time_to_first_token.algorithm,
        )
        performance_index.time_per_output_token = self.get_performance_metric(
            self.config.performance_config.time_per_output_token.metric,
            self.config.performance_config.time_per_output_token.algorithm,
        )
        csv_files = glob.glob(f"{output_path}/*/performances/*/*.csv")
        if len(csv_files) != 1:
            logger.error(
                f"The ais bench result for csv files are not unique, result files {csv_files}; "
                f"output path: {output_path}. please check"
            )
        dir_path = os.path.dirname(csv_files[0])
        file_name = os.path.splitext(os.path.basename(csv_files[0]))[0]

        # Generate JSON file path
        json_file = os.path.join(dir_path, f"{file_name}.json")

        with open_file(json_file, "r") as f:
            try:
                data = json.load(f)
            except json.decoder.JSONDecodeError as e:
                raise ValueError(
                    f"JSON file format error, cannot find total number of requests. File path: {json_file}"
                ) from e
        total_requests = data["Total Requests"]["total"]
        success_req = data["Success Requests"]["total"]
        performance_index.throughput = float(data["Request Throughput"]["total"].split()[0])
        if total_requests != 0:
            performance_index.success_rate = success_req / total_requests
            output_average = data["Output Token Throughput"]["total"]
            performance_index.generate_speed = float(output_average.split()[0])
        return performance_index

    def before_run(self, run_params: Optional[Tuple[OptimizerConfigField]] = None):
        remove_file(Path(self.config.output_path))
        super().before_run(run_params)
        # Start the test
        logger.debug("Start the aisbench test.")
        concurrency = rate = None
        for k in run_params:
            try:
                if k.name == "CONCURRENCY" and k.value:
                    concurrency = int(k.value)
                if k.name == "REQUESTRATE" and k.value:
                    rate = k.value
            except ValueError:
                logger.warning(f"the {k.name} is not number; please check: {k.value}")
                concurrency = rate = None
        with open_file(self.model_config_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        _request_rate_pattern = re.compile(r"(request_rate\s*=\s*)\d{1,10}(?:\.\d{1,30})?\s*,")
        _batch_size_pattern = re.compile(r"(batch_size\s*=\s*)\d{1,10}(?:\.\d{1,30})?\s*,")
        # Modify request_rate and batch_size
        for i, line in enumerate(lines):
            if "request_rate" in line:
                _res = _request_rate_pattern.search(lines[i])
                if _res:
                    if rate is None:
                        rate = 0
                    lines[i] = lines[i].replace(_res.group(), f"request_rate = {rate},")
            if "batch_size" in line:
                _res = _batch_size_pattern.search(lines[i])
                if _res:
                    if concurrency is None:
                        concurrency = 1000
                    lines[i] = lines[i].replace(_res.group(), f"batch_size = {concurrency},")

        # Write the modified content back to the file
        with open_file(self.model_config_path, "w", encoding="utf-8") as f:
            f.writelines(lines)


class VllmBenchMark(BenchmarkInterface):
    def __init__(self, config: Optional[VllmBenchmarkConfig] = None, *args, **kwargs):
        if config:
            self.config = config
        else:
            settings = get_settings()
            self.config = settings.vllm_benchmark
        super().__init__(*args, **kwargs)
        self.command = VllmBenchmarkCommand(self.config.command).command

    @property
    def num_prompts(self) -> int:
        """
        Get the process name property of the service
        Returns:""

        """
        return self.config.command.num_prompts

    @num_prompts.setter
    def num_prompts(self, value):
        """
        Get the process name property of the service
        Returns:""

        """
        self.config.command.num_prompts = value

    def update_command(self):
        self.command = VllmBenchmarkCommand(self.config.command).command

    def stop(self, del_log: bool = True):
        # Delete output files
        output_path = Path(self.config.command.result_dir)
        remove_file(output_path)
        super().stop(del_log)

    def before_run(self, run_params: Optional[Tuple[OptimizerConfigField, ...]] = None):  # Delete output files
        # Clean output directory before start because get_performance_index only retrieves one record,
        # to avoid getting wrong data
        output_path = Path(self.config.command.result_dir)
        remove_file(output_path)
        super().before_run(run_params)

    def get_performance_index(self):
        output_path = Path(self.config.command.result_dir)
        performance_index = PerformanceIndex()
        for file in walk_files(output_path):
            file = Path(file)
            if not file.name.endswith(".json"):
                continue
            with open_file(file, mode="r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    logger.error(f"Failed in parse vllm benchmark result. file: {file}")
                    continue

            performance_index.generate_speed = data.get("output_throughput", 0)
            performance_index.time_to_first_token = data.get("mean_ttft_ms", 0) / MS_TO_S
            performance_index.time_per_output_token = data.get("mean_tpot_ms", 0) / MS_TO_S
            num_prompts = data.get("num_prompts", 1)
            completed = data.get("completed", 0)
            performance_index.success_rate = 0
            if num_prompts > 0:
                performance_index.success_rate = completed / num_prompts
            performance_index.throughput = float(data.get("request_throughput", 3.0))
        return performance_index
