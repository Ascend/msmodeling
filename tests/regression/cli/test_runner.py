"""Tests for query-driven shape-grid orchestration."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from unittest import mock

import pytest

from tensor_cast.performance_model.profiling_database.query_demand import KernelQueryDemand
from tools.perf_data_collection.grid_generator.query_model import QueryModelArchitecture
from tools.perf_data_collection.grid_generator.query_workloads import QueryWorkloadRunResult
from tools.perf_data_collection.grid_generator.runner import (
    _query_cache_directory,
    discover_replay_supported_ops,
    iter_csv_files,
    load_csv_files,
    normalize_selected_ops,
    normalize_target_models,
    run_query_mode,
)


def _write_add_csv(path: Path) -> None:
    headers = [
        "OP State",
        "Input Shapes",
        "Input Data Types",
        "Input Formats",
        "Output Shapes",
        "Output Data Types",
        "Output Formats",
        "Average Duration(us)",
    ]
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=headers)
        writer.writeheader()
        writer.writerow(
            {
                "OP State": "dynamic",
                "Input Shapes": "1,4;1,4",
                "Input Data Types": "DT_BF16;DT_BF16",
                "Input Formats": "ND;ND",
                "Output Shapes": "1,4",
                "Output Data Types": "DT_BF16",
                "Output Formats": "ND",
                "Average Duration(us)": "1.0",
            }
        )


def _add_demand(tokens: int) -> KernelQueryDemand:
    return KernelQueryDemand(
        projector_version="test/v1",
        op_name="aten.add.Tensor",
        kernel_type="Add",
        query_mode="elementwise",
        input_shapes=((tokens, 4), (tokens, 4)),
        output_shapes=((tokens, 4),),
        input_dtypes=("DT_BF16", "DT_BF16"),
        output_dtypes=("DT_BF16",),
    )


class TestCsvDiscovery:
    def test_sorts_and_excludes_tmp_files(self, tmp_path: Path) -> None:
        (tmp_path / "PadV3.csv").write_text("a", encoding="utf-8")
        (tmp_path / "MatMulV2.csv").write_text("b", encoding="utf-8")
        (tmp_path / "MatMulV2.tmp.csv").write_text("c", encoding="utf-8")
        assert [path.name for path in iter_csv_files(tmp_path)] == ["MatMulV2.csv", "PadV3.csv"]

    def test_load_rejects_empty_or_missing_directory(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="No CSV files"):
            load_csv_files(tmp_path)
        with pytest.raises(ValueError, match="does not exist"):
            load_csv_files(tmp_path / "missing")

    def test_query_cache_is_invalidated_by_database_or_model_change(self, tmp_path: Path) -> None:
        database = tmp_path / "database"
        database.mkdir()
        (database / "op_mapping.yaml").write_text("device: TEST\n", encoding="utf-8")
        csv_path = database / "Add.csv"
        csv_path.write_text("Input Shapes\n1,4\n", encoding="utf-8")

        original = _query_cache_directory(database, ["org/model"], tmp_path)
        other_model = _query_cache_directory(database, ["org/other"], tmp_path)
        csv_path.write_text("Input Shapes\n2,4\n", encoding="utf-8")
        changed_database = _query_cache_directory(database, ["org/model"], tmp_path)

        assert original != other_model
        assert original != changed_database


class TestPublicInputNormalization:
    def test_target_models_accept_spaces_and_commas(self) -> None:
        assert normalize_target_models(["org/a,org/b", "org/a"]) == ["org/a", "org/b"]

    def test_replay_support_is_discovered_from_scripts(self, tmp_path: Path) -> None:
        (tmp_path / "Add_run.py").write_text("", encoding="utf-8")
        (tmp_path / "RmsNorm_run.py").write_text("", encoding="utf-8")
        (tmp_path / "helper.py").write_text("", encoding="utf-8")
        assert discover_replay_supported_ops(tmp_path) == ("Add", "RmsNorm")

    def test_selected_ops_fail_closed_outside_replay_support(self) -> None:
        with pytest.raises(ValueError, match="without an op_replay"):
            normalize_selected_ops(["Unknown"], ["Add"])


class TestRunQueryMode:
    @pytest.fixture(autouse=True)
    def _stub_query_model_architecture(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "tools.perf_data_collection.grid_generator.runner.resolve_query_model_architecture",
            lambda _model_id: QueryModelArchitecture(
                max_context_length=1024,
                num_experts=0,
                num_mtp_layers=0,
                tp_sizes=(1,),
                ep_sizes=(1,),
            ),
        )
        monkeypatch.setattr(
            "tools.perf_data_collection.grid_generator.runner.preflight_generated_rows",
            lambda _kernel_type, rows, _op_replay_dir: [mock.Mock(passed=True) for _row in rows],
        )

    def test_rows_are_incremental_per_csv(self, tmp_path: Path) -> None:
        database = tmp_path / "database"
        replay = tmp_path / "op_replay"
        database.mkdir()
        replay.mkdir()
        (database / "op_mapping.yaml").write_text("device: TEST_DEVICE\n", encoding="utf-8")
        _write_add_csv(database / "Add.csv")
        (replay / "Add_run.py").write_text("", encoding="utf-8")
        args = argparse.Namespace(
            database_path=database,
            rows=1,
            target_models=["org/model"],
            ops=["Add"],
            seed=0,
        )
        collection = (
            [_add_demand(2), _add_demand(3)],
            QueryWorkloadRunResult(attempted=1, succeeded=1, failed_workloads=()),
        )
        with mock.patch(
            "tools.perf_data_collection.grid_generator.runner._collect_query_demands",
            return_value=collection,
        ):
            first = run_query_mode(
                args,
                data_dir=database,
                op_replay_dir=replay,
                repo_root=tmp_path,
            )
            second = run_query_mode(
                args,
                data_dir=database,
                op_replay_dir=replay,
                repo_root=tmp_path,
            )

        assert first.total_appended_rows == 1
        assert second.total_appended_rows == 1
        with (database / "Add.csv").open("r", encoding="utf-8-sig", newline="") as input_file:
            rows = list(csv.DictReader(input_file))
        assert len(rows) == 3
        assert {row["Input Shapes"].strip('"') for row in rows} == {
            "1,4;1,4",
            "2,4;2,4",
            "3,4;3,4",
        }

    def test_explicit_unqueried_operator_uses_theory_fallback(self, tmp_path: Path) -> None:
        database = tmp_path / "database"
        replay = tmp_path / "op_replay"
        database.mkdir()
        replay.mkdir()
        (database / "op_mapping.yaml").write_text("device: TEST_DEVICE\n", encoding="utf-8")
        _write_add_csv(database / "Add.csv")
        (replay / "Add_run.py").write_text("", encoding="utf-8")
        args = argparse.Namespace(rows=1, target_models=["org/model"], ops=["Add"], seed=0)
        collection = (
            [],
            QueryWorkloadRunResult(attempted=1, succeeded=1, failed_workloads=()),
        )
        fallback_row = {
            "OP State": "dynamic",
            "Input Shapes": "2,4;2,4",
            "Input Data Types": "DT_BF16;DT_BF16",
            "Input Formats": "ND;ND",
            "Output Shapes": "2,4",
            "Output Data Types": "DT_BF16",
            "Output Formats": "ND",
            "Average Duration(us)": "0",
        }
        with (
            mock.patch(
                "tools.perf_data_collection.grid_generator.runner._collect_query_demands",
                return_value=collection,
            ),
            mock.patch(
                "tools.perf_data_collection.grid_generator.runner.build_theory_fallback_rows",
                return_value=([fallback_row], {"duplicates": 0}),
            ) as fallback,
        ):
            result = run_query_mode(
                args,
                data_dir=database,
                op_replay_dir=replay,
                repo_root=tmp_path,
            )

        assert result.total_appended_rows == 1
        fallback.assert_called_once()
        with (database / "Add.csv").open("r", encoding="utf-8-sig", newline="") as input_file:
            rows = list(csv.DictReader(input_file))
        assert [row["Input Shapes"] for row in rows] == ["1,4;1,4", "2,4;2,4"]

    def test_explicit_ops_override_other_queried_operators(self, tmp_path: Path) -> None:
        database = tmp_path / "database"
        replay = tmp_path / "op_replay"
        database.mkdir()
        replay.mkdir()
        (database / "op_mapping.yaml").write_text("device: TEST_DEVICE\n", encoding="utf-8")
        for operator in ("Add", "MaskedFill"):
            _write_add_csv(database / f"{operator}.csv")
            (replay / f"{operator}_run.py").write_text("", encoding="utf-8")
        args = argparse.Namespace(
            rows=1,
            target_models=["org/model"],
            ops=["MaskedFill"],
            seed=0,
        )
        collection = (
            [_add_demand(2)],
            QueryWorkloadRunResult(attempted=1, succeeded=1, failed_workloads=()),
        )
        fallback_row = {
            "OP State": "dynamic",
            "Input Shapes": "3,4;3,4",
            "Input Data Types": "DT_BF16;DT_BF16",
            "Input Formats": "ND;ND",
            "Output Shapes": "3,4",
            "Output Data Types": "DT_BF16",
            "Output Formats": "ND",
            "Average Duration(us)": "0",
        }
        with (
            mock.patch(
                "tools.perf_data_collection.grid_generator.runner._collect_query_demands",
                return_value=collection,
            ),
            mock.patch(
                "tools.perf_data_collection.grid_generator.runner.build_theory_fallback_rows",
                return_value=([fallback_row], {"duplicates": 0}),
            ),
        ):
            result = run_query_mode(
                args,
                data_dir=database,
                op_replay_dir=replay,
                repo_root=tmp_path,
            )

        assert [item.csv_path.stem for item in result.operators] == ["MaskedFill"]
        with (database / "Add.csv").open("r", encoding="utf-8", newline="") as input_file:
            assert len(list(csv.DictReader(input_file))) == 1
        with (database / "MaskedFill.csv").open("r", encoding="utf-8-sig", newline="") as input_file:
            assert len(list(csv.DictReader(input_file))) == 2

    def test_unqueried_skip_operator_is_reported_without_write(self, tmp_path: Path) -> None:
        database = tmp_path / "database"
        replay = tmp_path / "op_replay"
        database.mkdir()
        replay.mkdir()
        (database / "op_mapping.yaml").write_text("device: TEST_DEVICE\n", encoding="utf-8")
        _write_add_csv(database / "BatchMatMulV2.csv")
        (replay / "BatchMatMulV2_run.py").write_text("", encoding="utf-8")
        args = argparse.Namespace(
            rows=1,
            target_models=["org/model"],
            ops=["BatchMatMulV2"],
            seed=0,
        )
        collection = (
            [],
            QueryWorkloadRunResult(attempted=1, succeeded=1, failed_workloads=()),
        )
        with mock.patch(
            "tools.perf_data_collection.grid_generator.runner._collect_query_demands",
            return_value=collection,
        ):
            result = run_query_mode(
                args,
                data_dir=database,
                op_replay_dir=replay,
                repo_root=tmp_path,
            )

        assert result.total_appended_rows == 0
        assert result.operators[0].reason == "theory_skipped"
        with (database / "BatchMatMulV2.csv").open("r", encoding="utf-8", newline="") as input_file:
            assert len(list(csv.DictReader(input_file))) == 1

    def test_planning_error_does_not_partially_write_other_operators(self, tmp_path: Path) -> None:
        database = tmp_path / "database"
        replay = tmp_path / "op_replay"
        database.mkdir()
        replay.mkdir()
        (database / "op_mapping.yaml").write_text("device: TEST_DEVICE\n", encoding="utf-8")
        for operator in ("Add", "MoeTokenPermute"):
            _write_add_csv(database / f"{operator}.csv")
            (replay / f"{operator}_run.py").write_text("", encoding="utf-8")
        args = argparse.Namespace(
            rows=1,
            target_models=["org/model"],
            ops=["Add", "MoeTokenPermute"],
            seed=0,
        )
        collection = (
            [_add_demand(2)],
            QueryWorkloadRunResult(attempted=1, succeeded=1, failed_workloads=()),
        )
        with (
            mock.patch(
                "tools.perf_data_collection.grid_generator.runner._collect_query_demands",
                return_value=collection,
            ),
            mock.patch(
                "tools.perf_data_collection.grid_generator.runner.build_theory_fallback_rows",
                side_effect=ValueError("invalid fallback Shape"),
            ),
            pytest.raises(ValueError, match="invalid fallback Shape"),
        ):
            run_query_mode(
                args,
                data_dir=database,
                op_replay_dir=replay,
                repo_root=tmp_path,
            )

        with (database / "Add.csv").open("r", encoding="utf-8", newline="") as input_file:
            assert len(list(csv.DictReader(input_file))) == 1

    def test_unqueried_operator_without_generic_generator_is_skipped(self, tmp_path: Path) -> None:
        database = tmp_path / "database"
        replay = tmp_path / "op_replay"
        database.mkdir()
        replay.mkdir()
        (database / "op_mapping.yaml").write_text("device: TEST_DEVICE\n", encoding="utf-8")
        _write_add_csv(database / "MoeTokenPermute.csv")
        (replay / "MoeTokenPermute_run.py").write_text("", encoding="utf-8")
        args = argparse.Namespace(
            rows=1,
            target_models=["org/model"],
            ops=["MoeTokenPermute"],
            seed=0,
        )
        collection = (
            [],
            QueryWorkloadRunResult(attempted=1, succeeded=1, failed_workloads=()),
        )
        with mock.patch(
            "tools.perf_data_collection.grid_generator.runner._collect_query_demands",
            return_value=collection,
        ):
            result = run_query_mode(
                args,
                data_dir=database,
                op_replay_dir=replay,
                repo_root=tmp_path,
            )

        assert result.operators[0].reason == "theory_skipped"
        assert result.total_appended_rows == 0

    def test_without_explicit_ops_only_queried_replay_operators_are_generated(
        self,
        tmp_path: Path,
    ) -> None:
        database = tmp_path / "database"
        replay = tmp_path / "op_replay"
        database.mkdir()
        replay.mkdir()
        (database / "op_mapping.yaml").write_text("device: TEST_DEVICE\n", encoding="utf-8")
        for operator in ("Add", "MaskedFill"):
            _write_add_csv(database / f"{operator}.csv")
            (replay / f"{operator}_run.py").write_text("", encoding="utf-8")
        args = argparse.Namespace(rows=1, target_models=["org/model"], ops=None, seed=0)
        collection = (
            [_add_demand(2)],
            QueryWorkloadRunResult(attempted=1, succeeded=1, failed_workloads=()),
        )
        with mock.patch(
            "tools.perf_data_collection.grid_generator.runner._collect_query_demands",
            return_value=collection,
        ):
            result = run_query_mode(
                args,
                data_dir=database,
                op_replay_dir=replay,
                repo_root=tmp_path,
            )

        assert [item.csv_path.stem for item in result.operators] == ["Add"]
        with (database / "MaskedFill.csv").open("r", encoding="utf-8", newline="") as input_file:
            assert len(list(csv.DictReader(input_file))) == 1

    def test_rows_must_be_positive(self, tmp_path: Path) -> None:
        args = argparse.Namespace(rows=0, target_models=["org/model"], ops=None, seed=0)
        with pytest.raises(ValueError, match="greater than 0"):
            run_query_mode(args, data_dir=tmp_path, op_replay_dir=tmp_path, repo_root=tmp_path)

    def test_database_mapping_is_validated_before_workload_collection(self, tmp_path: Path) -> None:
        args = argparse.Namespace(rows=1, target_models=["org/model"], ops=None, seed=0)
        with pytest.raises(ValueError, match="op_mapping.yaml does not exist"):
            run_query_mode(args, data_dir=tmp_path, op_replay_dir=tmp_path, repo_root=tmp_path)
