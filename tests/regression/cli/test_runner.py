"""Tests for grid_generator/runner.py — pure functions and CLI path."""

import argparse
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from tools.perf_data_collection.grid_generator.runner import (
    iter_csv_files,
    load_csv_files,
    process_theory_csv,
    run_theory_mode,
)


class TestIterCsvFiles:
    def test_sorts_files(self):
        with tempfile.TemporaryDirectory() as td:
            datadir = Path(td)
            (datadir / "MatMulV2.csv").write_text("a")
            (datadir / "PadV3.csv").write_text("b")
            (datadir / "RmsNorm.csv").write_text("c")
            result = list(iter_csv_files(datadir))
            names = [p.name for p in result]
            assert "MatMulV2.csv" in names
            assert "PadV3.csv" in names
            assert "RmsNorm.csv" in names

    def test_excludes_tmp_files(self):
        with tempfile.TemporaryDirectory() as td:
            datadir = Path(td)
            (datadir / "MatMulV2.csv").write_text("a")
            (datadir / "MatMulV2.tmp.csv").write_text("b")
            result = list(iter_csv_files(datadir))
            names = [p.name for p in result]
            assert "MatMulV2.csv" in names
            assert "MatMulV2.tmp.csv" not in names

    def test_subdirs(self):
        with tempfile.TemporaryDirectory() as td:
            datadir = Path(td)
            sub = datadir / "sub"
            sub.mkdir()
            (sub / "Op.csv").write_text("x")
            result = list(iter_csv_files(datadir))
            assert len(result) == 1


class TestLoadCsvFiles:
    def test_empty_dir_raises(self):
        with tempfile.TemporaryDirectory() as td:
            datadir = Path(td)
            (datadir / "config.yaml").write_text("{}")
            with pytest.raises(ValueError):
                load_csv_files(datadir)

    def test_non_existent_dir_raises(self):
        with pytest.raises(ValueError):
            load_csv_files(Path("/nonexistent/path"))


class TestProcessTheoryCsv:
    def test_no_generator_returns_none(self):
        with tempfile.TemporaryDirectory() as td:
            csv_path = Path(td) / "UnknownKernel.csv"
            csv_path.write_text('Input Shapes,Average Duration(us)\n"128,5120",10.0\n')

            result = process_theory_csv(
                csv_path=csv_path,
                model_names=None,
                config={"assignments": {}, "patterns": {}},
                op_meta={},
                file_index=1,
                total_files=1,
            )
            assert result is None


class TestRunTheoryMode:
    def test_parses_target_models(self):
        with tempfile.TemporaryDirectory() as td:
            datadir = Path(td)
            (datadir / "config.yaml").write_text("assignments: {}\npatterns: {}\n")
            (datadir / "op_mapping.yaml").write_text("operator_mappings: {}\n")

            with (
                mock.patch.object(Path, "resolve", return_value=datadir),
                mock.patch(
                    "tools.perf_data_collection.grid_generator.runner.load_shape_grid_config",
                    return_value={"assignments": {}, "patterns": {}},
                ),
                mock.patch(
                    "tools.perf_data_collection.grid_generator.runner.load_op_mapping_metadata",
                    return_value={},
                ),
                mock.patch(
                    "tools.perf_data_collection.grid_generator.runner.iter_csv_files",
                    return_value=[],
                ),
            ):
                args = argparse.Namespace(
                    target_models="deepseek-ai/DeepSeek-V3,Qwen/Qwen3-32B",
                    rows=0,
                    seed=0,
                    max_hbm_gb=32.0,
                )
                total, skipped = run_theory_mode(args, datadir, [])
                assert total == 0

    def test_no_target_models(self):
        with tempfile.TemporaryDirectory() as td:
            datadir = Path(td)
            (datadir / "config.yaml").write_text("assignments: {}\npatterns: {}\n")

            with (
                mock.patch.object(Path, "resolve", return_value=datadir),
                mock.patch(
                    "tools.perf_data_collection.grid_generator.runner.load_shape_grid_config",
                    return_value={"assignments": {}, "patterns": {}},
                ),
                mock.patch(
                    "tools.perf_data_collection.grid_generator.runner.load_op_mapping_metadata",
                    return_value={},
                ),
                mock.patch(
                    "tools.perf_data_collection.grid_generator.runner.iter_csv_files",
                    return_value=[],
                ),
            ):
                args = argparse.Namespace(
                    target_models=None,
                    rows=50,
                    seed=42,
                    max_hbm_gb=0,
                )
                total, skipped = run_theory_mode(args, datadir, [])
                assert total == 0

    def test_with_csv_files_skipped(self):
        with tempfile.TemporaryDirectory() as td:
            datadir = Path(td)
            (datadir / "config.yaml").write_text("assignments: {}\npatterns: {}\n")
            csv_file = Path(td) / "UnknownKernel.csv"
            csv_file.write_text('Input Shapes,Average Duration(us)\n"128,5120",10.0\n')

            with (
                mock.patch.object(Path, "resolve", return_value=datadir),
                mock.patch(
                    "tools.perf_data_collection.grid_generator.runner.load_shape_grid_config",
                    return_value={"assignments": {}, "patterns": {}},
                ),
                mock.patch(
                    "tools.perf_data_collection.grid_generator.runner.load_op_mapping_metadata",
                    return_value={},
                ),
            ):
                args = argparse.Namespace(
                    target_models=None,
                    rows=0,
                    seed=0,
                    max_hbm_gb=32.0,
                )
                total, skipped = run_theory_mode(args, datadir, [csv_file])
                assert total == 0
                assert len(skipped) == 1
