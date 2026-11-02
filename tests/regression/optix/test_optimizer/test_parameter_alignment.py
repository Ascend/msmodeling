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
"""Keep parameter names attached to CSV values and selected measurements."""

import csv
from itertools import permutations
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from optix.common import CSV_COLUMN_HEADERS, CSV_HEADER_FIELDS, read_csv_s
from optix.config.config import (
    DataStorageConfig,
    OptimizerConfigField,
    PerformanceIndex,
    field_to_param,
    map_param_with_value,
)
from optix.optimizer.experience_fine_tunning import FineTune
from optix.optimizer.optimizer import PSOOptimizer
from optix.optimizer.store import DataStorage

LOAD_NAMES = ("CONCURRENCY", "REQUESTRATE", "NUM_PROMPTS")


def _load_fields(order, *, concurrency, num_prompts):
    values = {"CONCURRENCY": concurrency, "REQUESTRATE": 0.0, "NUM_PROMPTS": num_prompts}
    return tuple(
        OptimizerConfigField(
            name=name,
            config_position="env",
            min=0,
            max=4096,
            dtype="float" if name == "REQUESTRATE" else "int",
            value=values[name],
        )
        for name in order
    )


@pytest.mark.parametrize("order", list(permutations(LOAD_NAMES)))
@pytest.mark.parametrize("legacy_header", [False, True], ids=["unit_headers", "legacy_headers"])
def test_csv_order_does_not_change_candidate_load(tmp_path, order, legacy_header):
    config = DataStorageConfig(store_dir=tmp_path, pso_top_k=1)
    storage = DataStorage(config)
    baseline = PerformanceIndex(case_id="baseline", generate_speed=2, time_to_first_token=0.2, success_rate=1)
    storage.save(baseline, _load_fields(LOAD_NAMES, concurrency=32, num_prompts=128), duration=1, fitness=2)
    if legacy_header:
        with storage.save_file.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.reader(stream))
        rows[0] = [CSV_HEADER_FIELDS.get(name, name) for name in rows[0]]
        with storage.save_file.open("w", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerows(rows)

    # Reopening the writer must also preserve the existing file's column order.
    storage = DataStorage(config)
    candidate = PerformanceIndex(case_id="candidate", generate_speed=4, time_to_first_token=0.15, success_rate=1)
    storage.save(candidate, _load_fields(order, concurrency=308, num_prompts=1232), fitness=1, duration=2)
    loaded = read_csv_s(storage.save_file)
    assert list(loaded["CONCURRENCY"]) == [32, 308]
    assert list(loaded["NUM_PROMPTS"]) == [128, 1232]
    assert list(loaded["REQUESTRATE"]) == [0.0, 0.0]
    assert list(loaded["duration"]) == [1, 2]
    assert list(loaded["fitness"]) == [2, 1]
    assert list(loaded["case_id"]) == ["baseline", "candidate"]
    assert list(loaded["generate_speed"]) == [2, 4]
    assert list(loaded["time_to_first_token"]) == [0.2, 0.15]
    assert loaded["time_per_output_token"].isna().all()

    with patch("optix.optimizer.store.get_settings") as settings:
        settings.return_value.ttft_penalty = 1
        settings.return_value.tpot_penalty = 0
        selected = storage.get_best_result().iloc[0]
    optimizer = PSOOptimizer(
        scheduler=MagicMock(), target_field=_load_fields(LOAD_NAMES, concurrency=32, num_prompts=128)
    )
    candidate_fields = optimizer.get_target_field_from_case_data(selected)
    replay = {
        field.name: field.value for field in map_param_with_value(field_to_param(candidate_fields), candidate_fields)
    }
    assert replay == {"CONCURRENCY": 308, "REQUESTRATE": 0.0, "NUM_PROMPTS": 1232}


def test_csv_missing_metadata_does_not_shift_fitness(tmp_path):
    storage = DataStorage(DataStorageConfig(store_dir=tmp_path))
    fields = _load_fields(LOAD_NAMES, concurrency=32, num_prompts=128)
    storage.save(PerformanceIndex(), fields, duration=10, fitness=2)
    storage.save(PerformanceIndex(), tuple(reversed(fields)), fitness=1)
    loaded = read_csv_s(storage.save_file)
    assert loaded.iloc[1]["fitness"] == 1
    assert loaded.iloc[1]["CONCURRENCY"] == 32
    assert loaded.iloc[1]["NUM_PROMPTS"] == 128
    assert loaded["duration"].isna().iloc[1]


def test_csv_new_column_preserves_old_rows(tmp_path):
    storage = DataStorage(DataStorageConfig(store_dir=tmp_path))
    fields = _load_fields(LOAD_NAMES, concurrency=32, num_prompts=128)
    storage.save(PerformanceIndex(case_id="baseline"), fields, fitness=2)
    with storage.save_file.open(newline="", encoding="utf-8") as stream:
        original = list(csv.reader(stream))
    storage.save(PerformanceIndex(case_id="candidate"), fields, fitness=1, new_metric=3)
    with storage.save_file.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    assert rows[0] == [*original[0], "new_metric"]
    assert rows[1] == [*original[1], ""]
    loaded = read_csv_s(storage.save_file)
    assert list(loaded["fitness"]) == [2, 1]
    assert pd.isna(loaded.iloc[0]["new_metric"])
    assert loaded.iloc[1]["new_metric"] == 3


@pytest.mark.parametrize("new_column", [False, True], ids=["append", "extend_header"])
def test_append_to_preexisting_unit_header_csv(tmp_path, new_column):
    config = DataStorageConfig(store_dir=tmp_path)
    save_file = DataStorage(config).save_file
    columns = [
        "legacy_note",
        *reversed(PerformanceIndex.model_fields),
        "NUM_PROMPTS",
        "REQUESTRATE",
        "CONCURRENCY",
        "fitness",
        "duration",
    ]
    header = [CSV_COLUMN_HEADERS.get(column, column) for column in columns]
    original_row = {
        "legacy_note": "from an earlier run, with a comma\nand a second line",
        "case_id": "baseline",
        "generate_speed": 2,
        "time_to_first_token": 0.2,
        "NUM_PROMPTS": 128,
        "REQUESTRATE": 0,
        "CONCURRENCY": 32,
        "fitness": 2,
        "duration": 1,
    }
    with save_file.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerow([original_row.get(column) for column in columns])
    original = save_file.read_bytes()
    assert "generate_speed(tokens/s)" in header

    storage = DataStorage(config)
    candidate = PerformanceIndex(case_id="candidate", generate_speed=4, time_to_first_token=0.15, success_rate=1)
    fields = _load_fields(LOAD_NAMES, concurrency=308, num_prompts=1232)
    storage.save(candidate, fields, duration=2, fitness=1, **({"new_metric": 3} if new_column else {}))
    if not new_column:
        assert save_file.read_bytes().startswith(original)
    with save_file.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    assert len(rows) == 3
    expected_header = [*header, "new_metric"] if new_column else header
    assert rows[0] == expected_header
    assert all(len(row) == len(expected_header) for row in rows)
    if new_column:
        assert rows[1][-1] == ""
        assert rows[-1][-1] == "3"
    appended = dict(zip([CSV_HEADER_FIELDS.get(name, name) for name in expected_header], rows[-1]))
    assert appended["case_id"] == "candidate"
    assert float(appended["generate_speed"]) == 4
    assert float(appended["time_to_first_token"]) == 0.15
    assert int(appended["CONCURRENCY"]) == 308
    assert int(appended["NUM_PROMPTS"]) == 1232
    assert float(appended["REQUESTRATE"]) == 0
    assert float(appended["fitness"]) == 1
    assert float(appended["duration"]) == 2
    assert appended["legacy_note"] == ""
    assert appended["time_per_output_token"] == ""


@pytest.mark.parametrize("extend_schema", [False, True])
def test_same_storage_appends_fifty_results(tmp_path, extend_schema):
    storage = DataStorage(DataStorageConfig(store_dir=tmp_path))
    for i in range(50):
        order = LOAD_NAMES if i % 2 else tuple(reversed(LOAD_NAMES))
        storage.save(
            PerformanceIndex(case_id=f"case_{i}", generate_speed=i + 1),
            _load_fields(order, concurrency=i + 1, num_prompts=(i + 1) * 4),
            fitness=50 - i,
            **({"new_metric": i * 2} if extend_schema and i >= 25 else {}),
        )
    with storage.save_file.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    assert len(rows) == 51
    assert rows[0].count("generate_speed(tokens/s)") == 1
    assert rows[0].count("CONCURRENCY") == 1
    assert all(len(row) == len(rows[0]) for row in rows)
    loaded = read_csv_s(storage.save_file)
    assert list(loaded["case_id"]) == [f"case_{i}" for i in range(50)]
    assert list(loaded["generate_speed"]) == list(range(1, 51))
    assert list(loaded["CONCURRENCY"]) == list(range(1, 51))
    assert list(loaded["NUM_PROMPTS"]) == [i * 4 for i in range(1, 51)]
    assert list(loaded["fitness"]) == list(range(50, 0, -1))
    if extend_schema:
        assert loaded["new_metric"].iloc[:25].isna().all()
        assert list(loaded["new_metric"].iloc[25:]) == [i * 2 for i in range(25, 50)]


@pytest.mark.parametrize("name", ["generate_speed", "generate_speed(tokens/s)"])
@pytest.mark.parametrize("source", ["parameter", "metadata"])
def test_csv_rejects_metric_name_collision_without_changing_file(tmp_path, name, source):
    storage = DataStorage(DataStorageConfig(store_dir=tmp_path))
    storage.save(PerformanceIndex(generate_speed=4), (), fitness=2)
    original = storage.save_file.read_bytes()
    params = (OptimizerConfigField(name=name, value=999),) if source == "parameter" else ()
    metadata = {name: 999} if source == "metadata" else {}
    with pytest.raises(ValueError, match="[Cc]olliding.*generate_speed"):
        storage.save(PerformanceIndex(generate_speed=5), params, fitness=1, **metadata)
    assert storage.save_file.read_bytes() == original


def test_csv_migrates_missing_performance_fields_from_old_header(tmp_path):
    storage = DataStorage(DataStorageConfig(store_dir=tmp_path))
    header = ["case_id", "generate_speed(tokens/s)", "CONCURRENCY", "fitness"]
    with storage.save_file.open("w", newline="", encoding="utf-8") as stream:
        csv.writer(stream).writerows([header, ["old", 2, 32, 2]])
    storage.save(
        PerformanceIndex(case_id="new", generate_speed=4, time_to_first_token=0.15),
        _load_fields(LOAD_NAMES, concurrency=308, num_prompts=1232),
        fitness=1,
    )
    with storage.save_file.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    assert rows[0][: len(header)] == header
    assert rows[1][: len(header)] == ["old", "2", "32", "2"]
    assert all(value == "" for value in rows[1][len(header) :])
    assert all(len(row) == len(rows[0]) for row in rows)
    loaded = read_csv_s(storage.save_file)
    assert list(loaded["case_id"]) == ["old", "new"]
    assert list(loaded["generate_speed"]) == [2, 4]
    assert pd.isna(loaded.iloc[0]["time_to_first_token"])
    assert loaded.iloc[1]["time_to_first_token"] == 0.15
    assert loaded.iloc[1]["CONCURRENCY"] == 308
    assert loaded.iloc[1]["NUM_PROMPTS"] == 1232


@pytest.mark.parametrize("failure", ["flush", "replace"])
def test_csv_migration_failure_preserves_original_file(tmp_path, failure):
    storage = DataStorage(DataStorageConfig(store_dir=tmp_path))
    storage.save(PerformanceIndex(case_id="old"), (), fitness=2)
    original = storage.save_file.read_bytes()
    target = "optix.optimizer.store.os.fsync" if failure == "flush" else "pathlib.Path.replace"
    with (
        patch(target, side_effect=OSError("simulated migration failure")),
        pytest.raises(OSError, match="simulated migration failure"),
    ):
        storage.save(PerformanceIndex(case_id="new"), (), fitness=1, new_metric=3)
    assert storage.save_file.read_bytes() == original
    assert list(tmp_path.iterdir()) == [storage.save_file]


@pytest.mark.parametrize("malformed", ["duplicate_header", "wrong_row_width"])
def test_csv_rejects_ambiguous_or_misaligned_legacy_data(tmp_path, malformed):
    storage = DataStorage(DataStorageConfig(store_dir=tmp_path))
    header = ["generate_speed(tokens/s)", "generate_speed"] if malformed == "duplicate_header" else ["case_id"]
    with storage.save_file.open("w", newline="", encoding="utf-8") as stream:
        csv.writer(stream).writerows([header, ["old", "unexpected"]])
    original = storage.save_file.read_bytes()
    with pytest.raises(ValueError) as exc_info:
        storage.save(PerformanceIndex(case_id="new"), (), fitness=1)
    assert str(storage.save_file.resolve()) in str(exc_info.value)
    assert storage.save_file.read_bytes() == original


def test_empty_csv_receives_header(tmp_path):
    storage = DataStorage(DataStorageConfig(store_dir=tmp_path))
    storage.save_file.touch()
    storage.save(PerformanceIndex(), _load_fields(LOAD_NAMES, concurrency=32, num_prompts=128), fitness=2)
    assert read_csv_s(storage.save_file).iloc[0]["CONCURRENCY"] == 32


def _baseline_optimizer(*, manage_service, enum_baseline):
    seq = OptimizerConfigField(
        name="MAX_NUM_SEQS",
        config_position="run",
        min=0 if enum_baseline else 1,
        max=1 if enum_baseline else 128,
        dtype="enum" if enum_baseline else "int",
        dtype_param=[16, 48] if enum_baseline else None,
        value=64,
    )
    concurrency, rate, prompts = _load_fields(LOAD_NAMES, concurrency=32, num_prompts=128)
    prompts.dtype = "times"
    prompts.dtype_param = {"target_name": "CONCURRENCY", "product": 4, "dtype": "int"}
    fields = (seq, concurrency, rate, prompts)
    scheduler = MagicMock()
    scheduler.error_info = None
    scheduler.last_outcome = None
    scheduler.data_storage.config.pso_top_k = 0
    scheduler.simulator.data_field = (seq, prompts)
    scheduler.benchmark.data_field = (concurrency, rate)
    result = PerformanceIndex(case_id="baseline", generate_speed=5.203, time_to_first_token=0.2, success_rate=1)

    def measure(params, target_fields, **kwargs):
        scheduler.simulate_run_info = map_param_with_value(params, target_fields)
        return result

    scheduler.run.side_effect = measure
    scheduler.run_benchmark_only.side_effect = measure
    optimizer = PSOOptimizer(
        scheduler=scheduler,
        target_field=fields,
        ttft_penalty=1,
        tpot_penalty=0,
        skip_pso=True,
        max_fine_tune=0,
        use_request_rate_calibration=False,
        manage_simulator_lifecycle=manage_service,
        fine_tune=FineTune(ttft_penalty=1, tpot_penalty=0, target_field=fields),
    )
    return optimizer, fields


@pytest.mark.parametrize("manage_service", [False, True])
@pytest.mark.parametrize("enum_baseline", [False, True])
@pytest.mark.parametrize("skip_pso", [False, True])
def test_selected_baseline_keeps_measured_parameters(manage_service, enum_baseline, skip_pso):
    optimizer, fields = _baseline_optimizer(manage_service=manage_service, enum_baseline=enum_baseline)
    optimizer.skip_pso = skip_pso
    optimizer.scheduler.data_storage.get_best_result.return_value = pd.DataFrame()
    with patch("optix.optimizer.global_best_custom.CustomGlobalBestPSO"):
        optimizer._run_pso_optimize = MagicMock(return_value=(0, None))
        result = optimizer.run_plugin()
    assert [field.name for field in optimizer.target_field] == [field.name for field in fields]
    assert result.params == {"MAX_NUM_SEQS": 64, "CONCURRENCY": 32, "REQUESTRATE": 0.0, "NUM_PROMPTS": 128}
    assert result.performance_index.case_id == "baseline"
    if enum_baseline:
        assert optimizer.target_field[0].dtype_param == [16, 48]


def test_selected_baseline_uses_actual_calibrated_load():
    optimizer, _ = _baseline_optimizer(manage_service=True, enum_baseline=False)
    result = PerformanceIndex(case_id="baseline", generate_speed=5.203, time_to_first_token=0.2, success_rate=1)

    def measure(params, fields):
        measured = map_param_with_value(params, fields)
        for field in measured:
            if field.name == "CONCURRENCY":
                field.value = 16
            elif field.name == "NUM_PROMPTS":
                field.value = 64
            elif field.name == "REQUESTRATE":
                field.value = 2.5
        optimizer.scheduler.simulate_run_info = measured
        return result

    optimizer.scheduler.run.side_effect = measure
    selected = optimizer.run_plugin()
    assert selected.params == {"MAX_NUM_SEQS": 64, "CONCURRENCY": 16, "REQUESTRATE": 2.5, "NUM_PROMPTS": 64}


@pytest.mark.parametrize("skip_pso", [False, True])
@pytest.mark.parametrize("same_performance", [False, True], ids=["better_candidate", "equal_performance"])
def test_selection_keeps_parameters_of_the_selected_measurement(skip_pso, same_performance):
    optimizer, _ = _baseline_optimizer(manage_service=True, enum_baseline=False)
    optimizer.skip_pso = skip_pso
    candidates = []

    def refine(*args):
        if same_performance:
            candidate = optimizer.default_res.model_copy(deep=True)
            assert candidate == optimizer.default_res
            assert candidate is not optimizer.default_res
        else:
            candidate = PerformanceIndex(
                case_id="candidate", generate_speed=6, time_to_first_token=0.15, success_rate=1
            )
        candidates.append(candidate)
        fields = optimizer.get_target_field_from_case_data(
            {"MAX_NUM_SEQS": 96, "CONCURRENCY": 77, "REQUESTRATE": 7.5, "NUM_PROMPTS": 308}
        )
        return (
            [optimizer.default_fitness, optimizer.minimum_algorithm(candidate)],
            [optimizer.default_run_param, field_to_param(fields)],
            [optimizer.default_res, candidate],
        )

    optimizer.refine_default_candidate = MagicMock(side_effect=refine)
    optimizer.refine_optimization_candidates = MagicMock(side_effect=refine)
    optimizer._run_pso_optimize = MagicMock(return_value=(0, None))
    with patch("optix.optimizer.global_best_custom.CustomGlobalBestPSO"):
        result = optimizer.run_plugin()
    if skip_pso:
        optimizer._run_pso_optimize.assert_not_called()
        optimizer.refine_default_candidate.assert_called_once_with()
        optimizer.refine_optimization_candidates.assert_not_called()
    else:
        optimizer._run_pso_optimize.assert_called_once()
        optimizer.refine_default_candidate.assert_not_called()
        optimizer.refine_optimization_candidates.assert_called_once()
    if same_performance:
        # Equal measurements retain the first entry, which is the baseline.
        assert result.params == {"MAX_NUM_SEQS": 64, "CONCURRENCY": 32, "REQUESTRATE": 0.0, "NUM_PROMPTS": 128}
        assert result.performance_index is optimizer.default_res
    else:
        assert result.params == {"MAX_NUM_SEQS": 96, "CONCURRENCY": 77, "REQUESTRATE": 7.5, "NUM_PROMPTS": 308}
        assert result.performance_index is candidates[0]


@pytest.mark.parametrize("manage_service", [False, True])
@pytest.mark.parametrize("skip_pso", [False, True])
@pytest.mark.parametrize("same_performance", [False, True], ids=["better_candidate", "equal_performance"])
def test_recorded_candidate_keeps_actual_calibrated_load(manage_service, skip_pso, same_performance):
    optimizer, _ = _baseline_optimizer(manage_service=manage_service, enum_baseline=True)
    optimizer.skip_pso = skip_pso
    optimizer.max_fine_tune = 1 if skip_pso else 0
    candidate_values = {"MAX_NUM_SEQS": 48, "CONCURRENCY": 77, "REQUESTRATE": 3.75, "NUM_PROMPTS": 308}
    candidate_fields = optimizer.get_target_field_from_case_data(candidate_values)
    optimizer.fine_tune.fine_tune_with_concurrency_and_request_rate = MagicMock(return_value=candidate_fields)
    optimizer.scheduler.data_storage.get_best_result.return_value = pd.DataFrame([candidate_values])
    baseline_measure = optimizer.scheduler.run.side_effect
    measurements = []

    def measure(params, fields, **kwargs):
        if not measurements:
            result = baseline_measure(params, fields, **kwargs)
        else:
            measured = map_param_with_value(params, fields)
            for field in measured:
                if field.name == "CONCURRENCY":
                    field.value = 16
                elif field.name == "NUM_PROMPTS":
                    field.value = 64
                elif field.name == "REQUESTRATE":
                    field.value = 2.5
            optimizer.scheduler.simulate_run_info = tuple(reversed(measured))
            result = optimizer.default_res.model_copy(deep=True)
            if not same_performance:
                result.case_id = "candidate"
                result.generate_speed = 6
                result.time_to_first_token = 0.15
        measurements.append(result)
        return result

    optimizer.scheduler.run.side_effect = measure
    optimizer.scheduler.run_benchmark_only.side_effect = measure
    optimizer.scheduler.rerun_benchmark_only.side_effect = measure
    optimizer._run_pso_optimize = MagicMock(return_value=(0, None))
    with patch("optix.optimizer.global_best_custom.CustomGlobalBestPSO"):
        result = optimizer.run_plugin()
    assert len(measurements) == 2
    assert measurements[0] is not measurements[1]
    if same_performance:
        assert measurements[0] == measurements[1]
        assert result.performance_index is measurements[0]
        assert result.params == {"MAX_NUM_SEQS": 64, "CONCURRENCY": 32, "REQUESTRATE": 0.0, "NUM_PROMPTS": 128}
    else:
        assert result.performance_index is measurements[1]
        assert result.params == {"MAX_NUM_SEQS": 48, "CONCURRENCY": 16, "REQUESTRATE": 2.5, "NUM_PROMPTS": 64}
