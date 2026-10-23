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

"""Unit tests for profiling_data_source._check_profiling_database.

A missing ``op_mapping.yaml`` (or CSV data) makes every profiling lookup miss
and the simulation silently degrades to the analytic (roofline) baseline. The
default CLI log level is ``error``, so the check prints an explicit
``[WARNING]`` notice to stderr (bypassing logging) to keep the degradation
visible. The check runs inside ``ProfilingDataSource.__init__`` so every
construction path is covered.
"""

from tensor_cast.performance_model.profiling_database.profiling_data_source import (
    _check_profiling_database,
    ProfilingDataSource,
)


class TestCheckProfilingDatabase:
    def test_missing_directory_warns_to_stderr(self, tmp_path, capsys):
        missing = tmp_path / "no_such_db"
        _check_profiling_database(missing)
        captured = capsys.readouterr()
        assert "does not exist" in captured.err
        assert "[WARNING]" in captured.err
        assert str(missing.resolve()) in captured.err
        assert "analytic (roofline) baseline" in captured.err

    def test_filesystem_error_is_swallowed(self, tmp_path, capsys, caplog, monkeypatch):
        """The check must never raise; it logs the error detail and skips instead."""
        db_dir = tmp_path / "unlistable_db"
        db_dir.mkdir()
        monkeypatch.setattr(db_dir.__class__, "resolve", lambda self: (_ for _ in ()).throw(OSError("unresolvable")))
        with caplog.at_level(
            "WARNING", logger="tensor_cast.performance_model.profiling_database.profiling_data_source"
        ):
            _check_profiling_database(db_dir)
        captured = capsys.readouterr()
        assert captured.err == ""
        assert "check skipped due to filesystem error" in caplog.text
        assert "unresolvable" in caplog.text

    def test_init_wires_the_check(self, tmp_path, capsys, monkeypatch):
        """ProfilingDataSource.__init__ must run the preflight check first."""
        from tensor_cast.performance_model.profiling_database import profiling_data_source as pds

        called = []
        real_check = pds._check_profiling_database
        monkeypatch.setattr(
            pds, "_check_profiling_database", lambda data_dir: (called.append(data_dir), real_check(data_dir))[1]
        )
        db_dir = tmp_path / "wiring_db"
        db_dir.mkdir()
        source = ProfilingDataSource(db_dir)
        assert called == [db_dir]
        captured = capsys.readouterr()
        assert "op_mapping.yaml not found" in captured.err
        assert source._op_mapping == {}

    def test_directory_without_op_mapping_warns_to_stderr(self, tmp_path, capsys):
        db_dir = tmp_path / "empty_db"
        db_dir.mkdir()
        _check_profiling_database(db_dir)
        captured = capsys.readouterr()
        assert "op_mapping.yaml not found" in captured.err
        assert "[WARNING]" in captured.err
        assert str(db_dir.resolve()) in captured.err

    def test_directory_without_csv_data_warns_to_stderr(self, tmp_path, capsys):
        db_dir = tmp_path / "csv_less_db"
        db_dir.mkdir()
        (db_dir / "op_mapping.yaml").write_text("operator_mappings: {}\n", encoding="utf-8")
        _check_profiling_database(db_dir)
        captured = capsys.readouterr()
        assert "no kernel CSV data found" in captured.err
        assert "[WARNING]" in captured.err
        assert str(db_dir.resolve()) in captured.err

    def test_valid_database_is_silent(self, tmp_path, capsys):
        db_dir = tmp_path / "valid_db"
        db_dir.mkdir()
        (db_dir / "op_mapping.yaml").write_text("ops: {}\n", encoding="utf-8")
        (db_dir / "MatMul.csv").write_text("col\n", encoding="utf-8")
        _check_profiling_database(db_dir)
        captured = capsys.readouterr()
        assert captured.err == ""
        assert captured.out == ""

    def test_accepts_string_path(self, tmp_path, capsys):
        _check_profiling_database(str(tmp_path / "no_such_db"))
        captured = capsys.readouterr()
        assert "does not exist" in captured.err

    def test_missing_path_is_listed_before_op_mapping_check(self, tmp_path, capsys):
        """A nonexistent path must report the directory error, not the file one."""
        _check_profiling_database(tmp_path / "ghost" / "db")
        captured = capsys.readouterr()
        assert "does not exist" in captured.err
        assert "op_mapping.yaml not found" not in captured.err

    def test_missing_csv_is_listed_after_op_mapping_check(self, tmp_path, capsys):
        """A dir with op_mapping.yaml but no CSV reports the CSV error, not reversed."""
        db_dir = tmp_path / "mapping_only"
        db_dir.mkdir()
        (db_dir / "op_mapping.yaml").write_text("ops: {}\n", encoding="utf-8")
        _check_profiling_database(db_dir)
        captured = capsys.readouterr()
        assert "no kernel CSV data found" in captured.err
        assert "op_mapping.yaml not found" not in captured.err
