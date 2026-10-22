# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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
"""OptiX benchmark interface for an AISBench-compatible command."""

import re
import shlex
import shutil
from pathlib import Path

from optix.config.base_config import MINDIE_BENCHMARK_PERF_COLUMNS
from optix.config.config import get_settings
from optix.optimizer.errors import BenchmarkResultError
from optix.optimizer.interfaces.benchmark import BenchmarkInterface
from optix.optimizer.plugins.benchmark import AisBench
from optix.optimizer.utils import remove_file

from .config import AisBenchAutoConfig
from .results import AisBenchResultReader, select_result_output


class AisBenchAuto(BenchmarkInterface):
    """Run the native tool and read its final formal performance result."""

    backup = AisBench.backup

    def __init__(self, config: AisBenchAutoConfig | dict | None = None, *args, **kwargs):
        if config is None:
            config = getattr(get_settings(), "aisbench_auto", None)
        if config is None:
            raise ValueError("Configure aisbench_auto.command.script before using the aisbench_auto benchmark")
        self.config = AisBenchAutoConfig.model_validate(config).model_copy(deep=True)
        super().__init__(*args, process_name=self.config.process_name, **kwargs)
        self.work_path = self.config.work_path.expanduser().resolve()
        self.config.output_path = (self.work_path / self.config.output_path.expanduser()).resolve()
        self.mindie_benchmark_perf_columns = [name.lower().strip() for name in MINDIE_BENCHMARK_PERF_COLUMNS]
        self._result_reader = None
        self.update_command()

    def update_command(self):
        config = self.config.command
        executable = shutil.which(config.executable, path=self.env.get("PATH"))
        if executable is None:
            raise ValueError(f"Benchmark executable not found: {config.executable}")
        script = Path(config.script).expanduser()
        if not script.is_absolute():
            script = self.work_path / script
        if not script.is_file():
            raise ValueError(f"Benchmark script not found: {script}")
        self.command = [
            str(Path(executable).absolute()),
            str(script),
            *shlex.split(config.others),
            "--concurrency",
            "$CONCURRENCY",
            "--request_rate",
            "$REQUESTRATE",
        ]

    def before_run(self, run_params=None):
        self._result_reader = None
        output = (self.work_path / self.config.output_path.expanduser()).resolve()
        if output == self.work_path or output in self.work_path.parents:
            raise ValueError("Benchmark output cannot contain its working directory")
        protected = Path(get_settings().output).expanduser().resolve()
        if output == protected or output in protected.parents or protected in output.parents:
            raise ValueError(f"Benchmark output conflicts with a protected directory: {protected}")
        if output in (Path.cwd().resolve(), Path.home().resolve()):
            raise ValueError(f"Benchmark output conflicts with a protected directory: {output}")
        if output.exists() and not output.is_dir():
            raise ValueError(f"Benchmark output is not a directory: {output}")
        self.update_command()
        values = {field.name.upper(): str(field.value) for field in self.config.target_field}
        values.update({field.name.upper(): str(field.value) for field in run_params or ()})
        command = [
            re.sub(r"\$([A-Z][A-Z0-9_]*)", lambda match: values.get(match[1], match[0]), arg) for arg in self.command
        ]
        missing = sorted({match[1] for arg in command for match in re.finditer(r"\$([A-Z][A-Z0-9_]*)", arg)})
        if missing:
            raise ValueError(f"Undefined benchmark parameters: {', '.join(missing)}")
        remove_file(output)
        try:
            if any(output.iterdir()):
                raise ValueError(f"Benchmark output was not cleaned: {output}")
        except FileNotFoundError:
            pass
        self.config.output_path = output
        # Preserve quoted arguments, zero request rate and a Python venv's
        # executable symlink while retaining the common process/env lifecycle.
        self.command = command.copy()
        super().before_run(run_params)
        self.command = command

    def get_performance_index(self):
        return self._get_result_reader().get_performance_index()

    def get_performance_metric(self, metric_name, algorithm="average"):
        return self._get_result_reader().get_performance_metric(metric_name, algorithm)

    def get_best_concurrency(self):
        return self._get_result_reader().get_best_concurrency()

    def _get_result_reader(self):
        if self.process is not None and self.process.poll() != 0:
            raise BenchmarkResultError("AISBench process has not completed successfully")
        if self._result_reader is None:
            output = select_result_output(self.config.output_path, self.work_path, self.run_log, self.command)
            self._result_reader = AisBenchResultReader(
                self.config.model_copy(update={"output_path": output}), self.mindie_benchmark_perf_columns
            )
        return self._result_reader
