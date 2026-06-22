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
from pathlib import Path

import pytest

from optix.io_utils import (
    ensure_existing_file,
    open_file,
    walk_files,
    sanitize_csv_value,
    PATH_MAX_LENGTH,
)


class TestEnsureExistingFile:
    def test_normal_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello")
        result = ensure_existing_file(f)
        assert result.exists()
        assert result.is_file()

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ensure_existing_file(tmp_path / "nonexistent.txt")

    def test_path_is_directory(self, tmp_path):
        with pytest.raises(ValueError, match="Expect a file"):
            ensure_existing_file(tmp_path)

    def test_file_too_large(self, tmp_path):
        f = tmp_path / "big.txt"
        f.write_text("x")
        with pytest.raises(ValueError, match="File is too large"):
            ensure_existing_file(f, max_size=0)

    def test_max_size_none_skips_check(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello")
        result = ensure_existing_file(f, max_size=None)
        assert result.exists()

    def test_path_exceeds_path_max(self, tmp_path):
        long_name = "a" * (PATH_MAX_LENGTH + 1)
        with pytest.raises(ValueError, match="File path exceeds PATH_MAX"):
            ensure_existing_file(Path(long_name))

    def test_symlink_resolved(self, tmp_path):
        real_file = tmp_path / "real.txt"
        real_file.write_text("content")
        link = tmp_path / "link.txt"
        try:
            link.symlink_to(real_file)
        except OSError:
            pytest.skip("Symlink not supported on this platform")
        result = ensure_existing_file(link)
        assert result == real_file.resolve()


class TestOpenFile:
    def test_read_existing_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello", encoding="utf-8")
        with open_file(f, "r") as fh:
            assert fh.read() == "hello"

    def test_read_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            open_file(tmp_path / "nope.txt", "r")

    def test_write_mode(self, tmp_path):
        f = tmp_path / "out.txt"
        with open_file(f, "w") as fh:
            fh.write("data")
        assert f.read_text(encoding="utf-8") == "data"

    def test_binary_mode(self, tmp_path):
        f = tmp_path / "bin.dat"
        f.write_bytes(b"\x00\x01\x02")
        with open_file(f, "rb") as fh:
            assert fh.read() == b"\x00\x01\x02"

    def test_read_plus_mode_no_check(self, tmp_path):
        f = tmp_path / "rw.txt"
        f.write_text("init")
        with open_file(f, "r+") as fh:
            content = fh.read()
            assert content == "init"


class TestWalkFiles:
    def test_walk_directory(self, tmp_path):
        (tmp_path / "a.txt").write_text("a")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "b.txt").write_text("b")
        files = list(walk_files(tmp_path))
        assert len(files) == 2

    def test_walk_single_file(self, tmp_path):
        f = tmp_path / "single.txt"
        f.write_text("x")
        files = list(walk_files(f))
        assert len(files) == 1
        assert files[0] == f.resolve()

    def test_walk_nonexistent(self, tmp_path):
        files = list(walk_files(tmp_path / "nope"))
        assert files == []

    def test_walk_skips_symlinks(self, tmp_path):
        real = tmp_path / "real.txt"
        real.write_text("content")
        link = tmp_path / "link.txt"
        try:
            link.symlink_to(real)
        except OSError:
            pytest.skip("Symlink not supported")
        files = list(walk_files(tmp_path))
        # Only the real file should be yielded
        assert len(files) == 1
        assert files[0] == real.resolve()


class TestSanitizeCsvValue:
    def test_formula_prefix_equals(self):
        assert sanitize_csv_value("=cmd") == "'=cmd"

    def test_formula_prefix_plus(self):
        assert sanitize_csv_value("+1") == "'+1"

    def test_formula_prefix_minus(self):
        assert sanitize_csv_value("-1") == "'-1"

    def test_formula_prefix_at(self):
        assert sanitize_csv_value("@sum") == "'@sum"

    def test_normal_string(self):
        assert sanitize_csv_value("hello") == "hello"

    def test_non_string_value(self):
        assert sanitize_csv_value(123) == 123
        assert sanitize_csv_value(None) is None


class TestOpenFileAdditional:
    def test_write_binary_mode(self, tmp_path):
        f = tmp_path / "bin_out.dat"
        with open_file(f, "wb") as fh:
            fh.write(b"\x00\x01\x02")
        assert f.read_bytes() == b"\x00\x01\x02"

    def test_append_mode(self, tmp_path):
        f = tmp_path / "append.txt"
        f.write_text("start", encoding="utf-8")
        with open_file(f, "a") as fh:
            fh.write("_end")
        assert f.read_text(encoding="utf-8") == "start_end"

    def test_open_with_encoding(self, tmp_path):
        f = tmp_path / "enc.txt"
        f.write_text("test", encoding="utf-8")
        with open_file(f, "r", encoding="utf-8") as fh:
            assert fh.read() == "test"


class TestWalkFilesAdditional:
    def test_walk_empty_directory(self, tmp_path):
        sub = tmp_path / "empty"
        sub.mkdir()
        files = list(walk_files(sub))
        assert files == []

    def test_walk_nested_directories(self, tmp_path):
        (tmp_path / "a" / "b").mkdir(parents=True)
        (tmp_path / "a" / "b" / "deep.txt").write_text("deep")
        (tmp_path / "top.txt").write_text("top")
        files = list(walk_files(tmp_path))
        assert len(files) == 2

    def test_walk_skips_symlink_in_subdir(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        real = sub / "real.txt"
        real.write_text("content")
        link = sub / "link.txt"
        try:
            link.symlink_to(real)
        except OSError:
            pytest.skip("Symlink not supported")
        files = list(walk_files(tmp_path))
        resolved_files = [f.name for f in files]
        assert "real.txt" in resolved_files
        # link should be skipped
        assert sum(1 for f in files if f.name == "real.txt") == 1


class TestEnsureExistingFileAdditional:
    def test_resolve_removes_dots(self, tmp_path):
        f = tmp_path / "sub" / ".." / "test.txt"
        (tmp_path / "test.txt").write_text("hello")
        result = ensure_existing_file(f)
        assert result == (tmp_path / "test.txt").resolve()
