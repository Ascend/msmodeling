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
import csv
import json
import os
from dataclasses import asdict, is_dataclass
from itertools import chain
from pathlib import Path
from shutil import copymode
from tempfile import NamedTemporaryFile
from typing import Any, Optional

import numpy as np
from loguru import logger

from ..common import CSV_COLUMN_HEADERS, CSV_HEADER_FIELDS, read_csv_s
from ..config.base_config import RUN_TIME
from ..config.config import (
    DataStorageConfig,
    OptimizerConfigField,
    PerformanceIndex,
    get_settings,
)
from ..io_utils import open_file, sanitize_csv_value

LLM_MODEL = "llm_model"
DATASET_PATH = "dataset_path"
SIMULATOR = "simulator"
NUM_PROMPTS = "num_prompts"
MAX_OUTPUT_LEN = "max_output_len"


class DataStorage:
    def __init__(
        self,
        config: DataStorageConfig,
        simulator=None,
        benchmark=None,
    ):
        self.config = config
        if not self.config.store_dir.exists():
            self.config.store_dir.mkdir(parents=True, mode=0o750)
        self.save_file = self.config.store_dir.joinpath(f"data_storage_{RUN_TIME}.csv")
        self.metrics_save_file = self.config.store_dir.joinpath(f"metrics_samples_{RUN_TIME}.csv")
        self._case_counter = 0
        self.simulator = simulator
        self.benchmark = benchmark

    def next_case_id(self) -> str:
        self._case_counter += 1
        return f"case_{self._case_counter:06d}"

    @staticmethod
    def _safe_sanitize_csv_value(value):
        """Sanitize structured and command-line values before writing CSV."""
        if isinstance(value, (dict, list, tuple)):
            value = json.dumps(value, ensure_ascii=False)
        if isinstance(value, str):
            if value.startswith("--"):
                value = value[2:]
            elif "--" in value:
                value = value.replace("--", "")
        return sanitize_csv_value(value)

    def save_metrics_sample(self, sample, **metadata) -> None:
        """Append one metrics sample immediately so partial case traces survive failures."""
        if is_dataclass(sample):
            sample = asdict(sample)
        elif not isinstance(sample, dict):
            raise TypeError("metrics sample must be a dataclass or dict")
        row = {**metadata, **sample}
        exists = self.metrics_save_file.exists()
        with open_file(self.metrics_save_file, "a+" if exists else "w") as file:
            writer = csv.DictWriter(file, fieldnames=list(row))
            if not exists:
                writer.writeheader()
            writer.writerow({key: self._safe_sanitize_csv_value(value) for key, value in row.items()})

    @staticmethod
    def load_history_position(load_dir: Path, filter_field: Optional[dict] = None) -> Optional[list]:
        if not load_dir.exists():
            raise FileNotFoundError(f"file: {load_dir}")
        if not load_dir.is_dir():
            raise ValueError("Expect a directory, not a file.")
        history_data = []
        for file in sorted(
            [f for f in load_dir.iterdir() if f.is_file()],
            key=lambda x: x.stat().st_ctime,
        ):
            if file.name.startswith("data_storage") and file.suffix == ".csv":
                data = read_csv_s(file).to_dict(orient="records")
                history_data.extend(data)
        if not history_data:
            return None
        return DataStorage.filter_data(history_data, filter_field)

    @staticmethod
    def filter_data(data: list[dict], filter_field: Optional[dict] = None):
        if not filter_field:
            return data
        filtered_data = []
        for d in data:
            flag = False
            for k, v in filter_field.items():
                if k not in d:
                    flag = True
                    continue
                _d_value = d[k]
                if isinstance(_d_value, int) and _d_value != int(v):
                    flag = True
                    break
                if isinstance(_d_value, float) and _d_value != float(v):
                    flag = True
                    break
                if isinstance(_d_value, bool) and _d_value != bool(v):
                    flag = True
                    break
                if str(_d_value).strip().lower() != str(v).strip().lower():
                    flag = True
                    break
            if flag:
                continue
            filtered_data.append(d)
        return filtered_data

    def save(
        self,
        performance_index: PerformanceIndex,
        params: tuple[OptimizerConfigField],
        **kwargs,
    ):
        logger.info("Save result with DataStorage. File path: {!r}", self.save_file.resolve())

        row = {}
        values = chain(
            performance_index.model_dump().items(),
            ((field.name, field.value) for field in params),
            kwargs.items(),
        )
        for name, value in values:
            column = CSV_HEADER_FIELDS.get(name, name)
            if column in row:
                raise ValueError(
                    f"Colliding CSV column {column!r} in {str(self.save_file.resolve())!r}: "
                    "metrics, parameters and metadata must have distinct names"
                )
            row[column] = value
        with open_file(self.save_file, "a+", newline="") as file:
            file.seek(0)
            reader = csv.reader(file)
            header = next(reader, None)
            columns = [CSV_HEADER_FIELDS.get(column, column) for column in header] if header else list(row)
            if len(columns) != len(set(columns)):
                raise ValueError(f"Ambiguous CSV header in {str(self.save_file.resolve())!r}: {header}")
            new_columns = [column for column in row if column not in columns]
            if new_columns:
                old_rows = list(reader)
            else:
                # Append mode preserves prior rows; make the read-to-write transition explicit.
                file.seek(0, os.SEEK_END)
                writer = csv.writer(file)
                if not header:
                    writer.writerow([CSV_COLUMN_HEADERS.get(column, column) for column in columns])
                writer.writerow([self._safe_sanitize_csv_value(row.get(column)) for column in columns])
                return
        self._extend_csv_header(header, old_rows, new_columns, row)

    def _extend_csv_header(self, header, old_rows, new_columns, row):
        """Extend a serial writer's schema without exposing a partly rewritten CSV."""
        for line, old_row in enumerate(old_rows, start=2):
            if len(old_row) != len(header):
                raise ValueError(
                    f"Cannot extend CSV {str(self.save_file.resolve())!r}: row {line} has "
                    f"{len(old_row)} columns, expected {len(header)}"
                )
        columns = [CSV_HEADER_FIELDS.get(column, column) for column in header] + new_columns
        temporary = None
        try:
            with NamedTemporaryFile(
                mode="w", encoding="utf-8", newline="", dir=self.save_file.parent, delete=False
            ) as stream:
                temporary = Path(stream.name)
                writer = csv.writer(stream)
                writer.writerow([*header, *(CSV_COLUMN_HEADERS.get(column, column) for column in new_columns)])
                padding = [""] * len(new_columns)
                for old_row in old_rows:
                    writer.writerow([*old_row, *padding])
                writer.writerow([self._safe_sanitize_csv_value(row.get(column)) for column in columns])
                stream.flush()
                os.fsync(stream.fileno())
            copymode(self.save_file, temporary)
            temporary.replace(self.save_file)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        logger.info("Extended CSV header in {} with columns {}", self.save_file.resolve(), new_columns)

    @staticmethod
    def _to_bool(value: Any, default: bool = False) -> bool:
        if value is None:
            return default
        try:
            if np.isnan(value):
                return default
        except TypeError:
            pass
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        value_str = str(value).strip().lower()
        if value_str in {"true", "1", "yes", "y"}:
            return True
        if value_str in {"false", "0", "no", "n"}:
            return False
        return default

    def _get_eligible_results(self):
        settings = get_settings()
        optimizer_result = read_csv_s(self.save_file)
        optimizer_result = optimizer_result.replace([np.inf, -np.inf], np.nan)
        pso_result = optimizer_result
        if self.benchmark:
            command = self.benchmark.config.command
            request_nums = getattr(command, "num_prompts", None)
            if request_nums not in (None, "") and NUM_PROMPTS in optimizer_result.columns:
                request_nums = str(request_nums).strip()
                pso_result = optimizer_result[
                    optimizer_result[NUM_PROMPTS].apply(lambda value: str(value).strip()) == request_nums
                ]
        pso_result = pso_result.dropna(subset="fitness")
        if "early_exit" in pso_result.columns:
            pso_result = pso_result[~pso_result["early_exit"].apply(DataStorage._to_bool)]
        if "usable_as_best" in pso_result.columns:
            pso_result = pso_result[pso_result["usable_as_best"].apply(lambda value: DataStorage._to_bool(value, True))]
        if settings.ttft_penalty > 0:
            pso_result = pso_result[pso_result["time_to_first_token"] > 0]
        if settings.tpot_penalty > 0:
            pso_result = pso_result[pso_result["time_per_output_token"] > 0]
        pso_result = pso_result[pso_result["generate_speed"] > 0]
        return pso_result.reset_index()

    def get_best_result(self):
        settings = get_settings()
        pso_result = self._get_eligible_results()
        # pso_top_k=0 disables candidate FineTune, but the optimizer still needs
        # at least one PSO result for final validation and best-result selection.
        selection_top_k = max(1, self.config.pso_top_k)
        _fitness_index = pso_result.nsmallest(selection_top_k, "fitness").index
        if settings.ttft_penalty and settings.tpot_penalty:
            _generate_speed_index = (
                pso_result[
                    (pso_result["time_to_first_token"] <= settings.ttft_slo * (1 + settings.slo_coefficient))
                    & (pso_result["time_per_output_token"] <= settings.tpot_slo * (1 + settings.slo_coefficient))
                ]
                .nlargest(selection_top_k, "generate_speed")
                .index
            )
        elif settings.tpot_penalty:
            _generate_speed_index = (
                pso_result[pso_result["time_per_output_token"] <= settings.tpot_slo * (1 + settings.slo_coefficient)]
                .nlargest(selection_top_k, "generate_speed")
                .index
            )
        else:
            _generate_speed_index = pso_result.nlargest(selection_top_k, "generate_speed").index
        _fine_tune_index = _fitness_index.union(_generate_speed_index)
        return pso_result.iloc[_fine_tune_index]

    def get_reference_performance_index(self) -> Optional[PerformanceIndex]:
        if not self.save_file.exists():
            return None
        try:
            eligible_results = self._get_eligible_results()
        except Exception:
            return None
        if eligible_results.empty:
            return None
        row = eligible_results.nsmallest(1, "fitness").iloc[0]
        generate_speed = self._optional_float(row.get("metrics_window_generate_speed"))
        if generate_speed is None or generate_speed <= 0:
            return None
        return PerformanceIndex(
            generate_speed=generate_speed,
            time_to_first_token=self._optional_float(row.get("metrics_window_time_to_first_token")),
            time_per_output_token=self._optional_float(row.get("metrics_window_time_per_output_token")),
            success_rate=self._optional_float(row.get("metrics_window_success_rate")),
        )

    @staticmethod
    def _optional_float(value: Any) -> Optional[float]:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if np.isnan(numeric):
            return None
        return numeric
