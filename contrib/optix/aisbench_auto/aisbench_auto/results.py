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
"""Select the final native AISBench experiment without dropping raw rounds."""

import argparse
import json
import re
import shutil
from pathlib import Path

from optix.optimizer.errors import BenchmarkResultError
from optix.optimizer.plugins.benchmark import AisBench, _require_unique_csv


class AisBenchResultReader:
    """Reuse the native metric readers against an isolated result view."""

    get_performance_metric = AisBench.get_performance_metric
    get_performance_index = AisBench.get_performance_index
    get_best_concurrency = AisBench.get_best_concurrency

    def __init__(self, config, columns):
        self.config = config
        self.mindie_benchmark_perf_columns = columns


def _expected_experiments(command):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False, exit_on_error=False)
    parser.add_argument("--dataset_type", default="normal")
    parser.add_argument("--prefix_test", action="store_true")
    parser.add_argument("--repeat", type=int, default=1)
    try:
        args, _ = parser.parse_known_args(command[2:])
    except argparse.ArgumentError as exc:
        raise BenchmarkResultError(f"Invalid AISBench round parameters: {exc}") from exc
    if args.repeat < 1:
        raise BenchmarkResultError("AISBench --repeat must be positive")
    # The native tool repeats normal datasets; prefix datasets have one formal pass.
    return 1 + int(args.prefix_test) if args.dataset_type == "prefix_cache" else args.repeat


def _experiment_folders(run_log, work_path):
    folders = []
    if run_log is None or not Path(run_log).is_file():
        return folders
    with Path(run_log).open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            line = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", line)
            if "Current exp folder:" in line:
                path = line.split("Current exp folder:", 1)[1].strip()
                if not path:
                    raise BenchmarkResultError("Empty AISBench experiment path in current run log")
                folders.append((work_path / path).resolve())
    return folders


def _validate_pair(csv_file, experiment):
    json_file = csv_file.with_suffix(".json")
    for path in (csv_file, json_file):
        if not path.resolve().is_relative_to(experiment) or not path.is_file():
            raise BenchmarkResultError(f"Missing or out-of-directory AISBench result: {path}")
    try:
        with json_file.open(encoding="utf-8") as stream:
            json.load(stream)
    except (OSError, ValueError) as exc:
        raise BenchmarkResultError(f"Invalid final AISBench result: {json_file}") from exc
    return json_file


def select_result_output(output, work_path, run_log, command):
    """Publish the last logged experiment; never fall back to an earlier pass."""
    expected = _expected_experiments(command)
    folders = _experiment_folders(run_log, work_path)
    if not folders:
        if expected > 1:
            raise BenchmarkResultError("Missing experiment markers for a warmup or repeated AISBench run")
        csv_file = Path(_require_unique_csv(output)[0])
        _validate_pair(csv_file, output)
        return output
    if len(folders) < expected:
        raise BenchmarkResultError(f"Incomplete AISBench run: expected {expected} experiments, found {len(folders)}")
    experiment = folders[-1]
    if experiment.parent != output:
        raise BenchmarkResultError(f"Final AISBench experiment is outside output_path: {experiment}")
    if experiment in folders[:-1]:
        raise BenchmarkResultError(f"Final AISBench experiment reuses an earlier round directory: {experiment}")
    csv_files = list(experiment.glob("performances/*/*.csv"))
    if len(csv_files) != 1:
        raise BenchmarkResultError(f"Final AISBench experiment must contain one performance CSV: {experiment}")
    csv_file = csv_files[0]
    json_file = _validate_pair(csv_file, experiment)
    selected = output / ".optix_selected"
    if selected.exists() or selected.is_symlink():
        raise BenchmarkResultError(f"Reserved AISBench selection directory already exists: {selected}")
    selected.mkdir()
    destination = selected / experiment.name / "performances" / csv_file.parent.name
    destination.mkdir(parents=True, exist_ok=False)
    for path in (csv_file, json_file):
        shutil.copy2(path, destination / path.name)
    (selected / "selection.json").write_text(
        json.dumps(
            {"policy": "last", "experiment": str(experiment), "csv": str(csv_file), "rounds": len(folders)}, indent=2
        ),
        encoding="utf-8",
    )
    return selected
