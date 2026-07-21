import json
import logging
import shutil
import shlex
from pathlib import Path
from typing import Optional, Tuple, List
from datetime import datetime

from optix.config.config import get_settings, PerformanceIndex, OptimizerConfigField
from optix.optimizer.interfaces.benchmark import BenchmarkInterface
from optix.optimizer.utils import remove_file
from pydantic import BaseModel, Field


class MetricAlgorithm(BaseModel):
    metric: str = "TTFT"
    algorithm: str = "average"


class PerformanceConfig(BaseModel):
    time_to_first_token: MetricAlgorithm = MetricAlgorithm(metric="TTFT", algorithm="average")
    time_per_output_token: MetricAlgorithm = MetricAlgorithm(metric="TPOT", algorithm="average")


def backup_file(output_path: Path, suffix: Optional[str] = None) -> None:
    """
    备份指定目录（或文件）。

    suffix: 备份名后缀。传入时形如 pso_3_5（阶段_迭代_种子），用于按迭代次数和种子序号
            命名备份目录；为空时回退到时间戳，保持原有行为。
    """
    if not output_path.exists():
        return

    if not suffix:
        suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"{output_path.name}_backup"
    backup_path = output_path.parent / backup_name / suffix

    try:
        if output_path.is_dir():
            shutil.copytree(output_path, backup_path, dirs_exist_ok=True)
        else:
            shutil.copy2(output_path, backup_path)
        logging.info(f"success backup '{output_path}' -->: '{backup_path}'")
    except Exception as e:
        logging.error(f"failed to backup '{output_path}': {e}")


class EvalscopeCommandConfig(BaseModel):
    url: str = ""
    model: str = ""
    tokenizer_path: str = ""
    dataset: str = ""
    outputs_dir: str = ""
    others: str = ""


class EvalscopePerfCommand:
    def __init__(self, benchmark_command_config: EvalscopeCommandConfig):
        # TODO: adjust to the name in pyproject.toml
        self.process = shutil.which("evalscope")
        if self.process is None:
            # TODO: adjust to the name in pyproject.toml
            raise ValueError("Error: The 'evalscope' executable was not found in the system PATH.")
        self.benchmark_command_config = benchmark_command_config

    @property
    def command(self):
        # TODO:  adjust to your benchmark cli, reference its --help info
        cmd = [
            self.process,
            "perf",
            "--url",
            self.benchmark_command_config.url,
            "--model",
            self.benchmark_command_config.model,
            "--tokenizer-path",
            self.benchmark_command_config.tokenizer_path,
            "--dataset",
            self.benchmark_command_config.dataset,
            "--outputs-dir",
            self.benchmark_command_config.outputs_dir,
        ]
        if self.benchmark_command_config.others:
            cmd.extend(shlex.split(self.benchmark_command_config.others))
        return cmd


# TODO: rename as your benchmark's name.
class EvalscopePerfConfig(BaseModel):
    # TODO: set output path for result files
    output_path: Path = Path("evalscopeperf")
    process_name: str = ""
    # TODO: rename for your own benchmark
    command: EvalscopeCommandConfig = EvalscopeCommandConfig()
    performance_config: PerformanceConfig = PerformanceConfig()
    target_field: List[OptimizerConfigField] = Field(default_factory=list)


# TODO: rename as your benchmark's name
class EvalscopePerfBenchMark(BenchmarkInterface):
    def __init__(self, config: Optional[EvalscopePerfConfig] = None, *args, **kwargs):
        if config:
            self.config = config
        else:
            settings = get_settings()
            evalscopeperf_data = settings.evalscopeperf
            if isinstance(evalscopeperf_data, dict):
                self.config = EvalscopePerfConfig(**evalscopeperf_data)
            else:
                self.config = evalscopeperf_data
        super().__init__(*args, **kwargs)
        self.command = EvalscopePerfCommand(self.config.command).command

    def update_command(self):
        self.command = EvalscopePerfCommand(self.config.command).command

    def _backup_suffix(self) -> Optional[str]:
        """根据 scheduler 设置的 bak_path 生成备份名后缀。

        bak_path 形如 bak/pso_003/5，其中 pso_003 为 {阶段}_{迭代次数}，5 为种子（粒子）序号，
        组合成 pso_003/5 作为后缀，备份目录形如 <输出目录>_backup/pso_003/5，
        即按阶段_迭代为一级、种子号为二级目录组织。
        bak_path 为空（如超过目录大小限制被置空）时返回 None，由调用方回退到时间戳。
        """
        bak_path = getattr(self, "bak_path", None)
        if not bak_path:
            return None
        bak_path = Path(bak_path)
        seed = bak_path.name
        phase_iter = bak_path.parent.name
        return f"{phase_iter}/{seed}"

    def stop(self, del_log: bool = True):
        # 删除输出的文件
        output_path = Path(self.config.command.outputs_dir)
        if output_path.resolve() == Path.cwd().resolve():
            raise ValueError(f"Output path '{output_path}' cannot be the same as current path {Path.cwd()}.")
        backup_file(output_path, self._backup_suffix())
        remove_file(output_path)
        super().stop(del_log)

    def before_run(self, run_params: Optional[Tuple[OptimizerConfigField, ...]] = None):
        # 启动前清理输出目录 因为get_performance_index是从里面获取其中一条数据，防止获取到错误数据
        output_path = Path(self.config.command.outputs_dir)
        if output_path.resolve() == Path.cwd().resolve():
            raise ValueError(f"Output path '{output_path}' cannot be the same as current path {Path.cwd()}.")
        remove_file(output_path)
        super().before_run(run_params)

    # TODO: find all the needed data elements in your benchmark's performance result file
    def get_performance_index(self) -> PerformanceIndex:
        output_path = Path(self.config.command.outputs_dir)
        performance_index = PerformanceIndex()
        benchmark_file = None
        for file in output_path.rglob("benchmark_summary.json"):
            benchmark_file = file
            with open(benchmark_file, mode='r', encoding="utf-8") as f:
                data = json.load(f)
            performance_index.generate_speed = data.get("Output Throughput (tok/s)") or data.get(
                "Output token throughput (tok/s)", 0
            )
            # TTFT 首包时延（输出：s）
            ttft_ms = data.get("TTFT (ms)")
            if ttft_ms is not None:
                performance_index.time_to_first_token = ttft_ms / 1000
            else:
                performance_index.time_to_first_token = data.get("Average time to first token (s)", 0)
            # TPOT 每输出token时延（输出：s）
            tpot_ms = data.get("TPOT (ms)")
            if tpot_ms is not None:
                performance_index.time_per_output_token = tpot_ms / 1000
            else:
                performance_index.time_per_output_token = data.get("Average time per output token (s)", 0)
            # 防止值为0或键缺失导致除零，回退到1
            num_prompts = data.get("Total Requests") or data.get("Total requests") or 1
            completed = data.get("Success Requests") or data.get("Succeed requests", 0)
            performance_index.success_rate = completed / num_prompts
            performance_index.throughput = data.get("Req Throughput (req/s)") or data.get(
                "Request throughput (req/s)", 0.0
            )
        # 未找到任何结果文件说明本轮测试失败（测试未完成、输出目录配置错误等），
        # 抛异常让寻优器感知本轮失败，避免默认值被当成真实数据污染寻优
        if benchmark_file is None:
            raise FileNotFoundError(
                f"No 'benchmark_summary.json' found under '{output_path}', benchmark may have failed."
            )
        return performance_index
