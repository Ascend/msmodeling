import json
from pathlib import Path
from typing import Optional, Tuple

from ms_serviceparam_optimizer.config.config import VllmBenchmarkConfig, get_settings, PerformanceIndex, OptimizerConfigField
from ms_serviceparam_optimizer.config.custom_command import VllmBenchmarkCommand
from ms_serviceparam_optimizer.optimizer.interfaces.benchmark import BenchmarkInterface
from ms_serviceparam_optimizer.optimizer.utils import remove_file


class VllmBenchMark(BenchmarkInterface):
    def __init__(self, config: Optional[VllmBenchmarkConfig] = None, *args, **kwargs):
        if config:
            self.config = config
        else:
            settings = get_settings()
            if settings.name != "vllm-inference-optimization":
                raise ValueError("Settings is invalidator.")
            self.config = settings.vllm_benchmark
        super().__init__(*args, **kwargs)
        self.command = VllmBenchmarkCommand(self.config.command).command

    def update_command(self):
        self.command = VllmBenchmarkCommand(self.config.command).command

    @property
    def num_prompts(self) -> int:
        """
        获取数据的请求数
        Returns:""

        """
        return self.config.command.num_prompts

    @num_prompts.setter
    def num_prompts(self, value):
        """
        设置数据的请求数
        Returns:""

        """
        self.config.command.num_prompts = value

    @property
    def model_name(self) -> str:
        """
        获取当前运行运行模型的名字
        Returns:

        """
        return ""

    @property
    def dataset_path(self) -> str:
        """
        获取当前使用的数据集
        Returns:

        """
        return ""

    @property
    def max_output_len(self) -> 0:
        """
        获取当前设置的最大输出长度。
        Returns:

        """
        return 0

    def stop(self, del_log: bool = True):
        # 删除输出的文件
        output_path = Path(self.config.command.result_dir)
        remove_file(output_path)
        super().stop(del_log)

    def before_run(self, run_params: Optional[Tuple[OptimizerConfigField, ...]] = None):
        # 启动前清理输出目录 因为get_performance_index是从里面获取其中一条数据，防止获取到错误数据
        output_path = Path(self.config.command.result_dir)
        remove_file(output_path)
        super().before_run(run_params)

    def get_performance_index(self) -> PerformanceIndex:
        output_path = Path(self.config.command.result_dir)
        performance_index = PerformanceIndex()
        for file in output_path.iterdir():
            if not file.name.endswith(".json"):
                continue
            with open(file, mode='r', encoding="utf-8") as f:
                data = json.load(f)
            performance_index.generate_speed = data.get("output_throughput", 0)
            performance_index.time_to_first_token = data.get(
                self.config.performance_config.time_to_first_token.metric,
                0) / 10 ** 3
            performance_index.time_per_output_token = data.get(
                self.config.performance_config.time_per_output_token.metric,
                0) / 10 ** 3
            num_prompts = data.get("num_prompts", 1)
            completed = data.get("completed", 0)
            performance_index.success_rate = completed / num_prompts
            performance_index.throughput = float(data.get("request_throughput", 3.0))
        return performance_index
