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
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from optix.optimizer.utils import (
    backup,
    close_file_fp,
    get_folder_size,
    get_required_field_from_json,
    is_root,
    kill_children,
    kill_process,
    remove_file,
)


class TestRemoveFile:
    def test_none_path(self):
        remove_file(None)

    def test_nonexistent_path(self, tmp_path):
        remove_file(tmp_path / "nope")

    def test_remove_single_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("data")
        remove_file(f)
        assert not f.exists()

    def test_remove_directory_files(self, tmp_path):
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.txt").write_text("b")
        remove_file(tmp_path)
        assert len(list(tmp_path.iterdir())) == 0

    def test_remove_string_path(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("data")
        remove_file(str(f))
        assert not f.exists()

    def test_remove_directory_with_subdirectory(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "file.txt").write_text("x")
        remove_file(tmp_path)
        assert not sub.exists()

    def test_remove_directory_rmtree_error(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        with patch("shutil.rmtree", side_effect=OSError("perm denied")):
            with patch("optix.optimizer.utils.logger.error") as mock_log:
                remove_file(tmp_path)
        mock_log.assert_called_once()
        # The log should reference the specific sub-path that failed, not
        # the top-level output_path.
        assert str(sub) in mock_log.call_args[0][0]

    def test_remove_file_refuses_current_working_directory(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        sentinel = tmp_path / "keep.txt"
        sentinel.write_text("stay", encoding="utf-8")

        with patch("optix.optimizer.utils.logger.error") as mock_log:
            remove_file(Path(""))

        mock_log.assert_called_once()
        assert sentinel.exists()
        assert list(tmp_path.iterdir()) == [sentinel]

    def test_remove_file_refuses_home_directory(self, tmp_path):
        home_file = tmp_path / "home_secret.txt"
        home_file.write_text("keep", encoding="utf-8")

        with (
            patch("optix.optimizer.utils.Path.home", return_value=tmp_path),
            patch("optix.optimizer.utils.logger.error") as mock_log,
        ):
            remove_file(tmp_path)

        mock_log.assert_called_once()
        assert home_file.exists()

    def test_remove_file_refuses_filesystem_root(self, tmp_path):
        with patch("optix.optimizer.utils.logger.error") as mock_log:
            remove_file(Path("/"))

        mock_log.assert_called_once()
        assert mock_log.call_args[0][0] == "Refusing to clean {}: {}"
        assert mock_log.call_args[0][1] == "filesystem root"


class TestKillChildren:
    def test_kill_running_child(self):
        # kill_children calls is_running() twice per child:
        #   (1) before sending SIGKILL — to decide whether to enter the block;
        #   (2) after wait(10)       — to check if the process is still alive.
        # The side_effect [True, False] models a process that is running
        # initially but terminates after the kill.
        child = MagicMock()
        child.is_running.side_effect = [True, False]
        child.pid = 1234
        kill_children([child])
        assert child.is_running.call_count == 2
        child.send_signal.assert_called_once_with(9)
        child.wait.assert_called_once_with(10)

    def test_skip_not_running(self):
        child = MagicMock()
        child.is_running.return_value = False
        kill_children([child])
        child.send_signal.assert_not_called()

    def test_kill_exception(self):
        # When send_signal raises an exception, the except block triggers
        # a continue — so is_running() is called only once (before kill),
        # and wait() is never invoked.
        child = MagicMock()
        child.is_running.return_value = True
        child.send_signal.side_effect = Exception("fail")
        child.pid = 999
        kill_children([child])
        assert child.is_running.call_count == 1
        child.send_signal.assert_called_once_with(9)
        child.wait.assert_not_called()

    def test_kill_still_running(self):
        # When the child process remains alive even after send_signal(9) +
        # wait(10), kill_children logs an error but does NOT retry — the
        # loop simply moves on to the next child.  Both is_running() calls
        # (before and after kill) return True.
        child = MagicMock()
        child.is_running.side_effect = [True, True]
        child.pid = 111
        kill_children([child])
        assert child.is_running.call_count == 2
        child.send_signal.assert_called_once_with(9)
        child.wait.assert_called_once_with(10)


class TestKillProcess:
    @patch("optix.optimizer.utils.psutil.process_iter")
    @patch("optix.optimizer.utils.psutil.Process")
    def test_kill_matching_process(self, mock_process_cls, mock_iter):
        proc = MagicMock()
        proc.info = {"pid": 100, "name": "vllm_worker"}
        proc.pid = 100
        proc.is_running.side_effect = [True, False]
        mock_iter.return_value = [proc]
        mock_process_cls.return_value.children.return_value = []
        kill_process("vllm")

    @patch("optix.optimizer.utils.psutil.process_iter")
    def test_skip_no_info(self, mock_iter):
        proc = MagicMock(spec=[])
        mock_iter.return_value = [proc]
        kill_process("vllm")


class TestBackup:
    def test_backup_none(self):
        backup(None, None)

    def test_backup_nonexistent_target(self, tmp_path):
        backup(tmp_path / "nope", tmp_path)

    def test_backup_file(self, tmp_path):
        src = tmp_path / "src.txt"
        src.write_text("content")
        bak_dir = tmp_path / "bak"
        bak_dir.mkdir()
        backup(src, bak_dir, class_name="cls")
        assert (bak_dir / "cls" / "src.txt").exists()

    def test_backup_file_already_exists(self, tmp_path):
        src = tmp_path / "src.txt"
        src.write_text("content")
        bak_dir = tmp_path / "bak"
        bak_dir.mkdir()
        dest = bak_dir / "cls" / "src.txt"
        dest.parent.mkdir(parents=True)
        dest.write_text("old")
        backup(src, bak_dir, class_name="cls")
        # Should not overwrite
        assert dest.read_text() == "old"

    def test_backup_directory(self, tmp_path):
        src = tmp_path / "srcdir"
        src.mkdir()
        (src / "a.txt").write_text("a")
        bak_dir = tmp_path / "bak"
        bak_dir.mkdir()
        backup(src, bak_dir, class_name="cls")
        assert (bak_dir / "cls" / "srcdir" / "a.txt").exists()

    def test_backup_max_depth(self, tmp_path):
        src = tmp_path / "src.txt"
        src.write_text("x")
        bak_dir = tmp_path / "bak"
        bak_dir.mkdir()
        backup(src, bak_dir, max_depth=10, current_depth=10)


class TestCloseFileFp:
    def test_close_none(self):
        close_file_fp(None)

    def test_close_with_close_method(self):
        fp = MagicMock()
        close_file_fp(fp)
        fp.close.assert_called_once()

    def test_close_fd(self):
        import tempfile

        fd, path = tempfile.mkstemp()
        close_file_fp(fd)
        os.unlink(path)

    def test_close_error(self):
        fp = MagicMock()
        fp.close.side_effect = OSError("fail")
        close_file_fp(fp)


class TestGetFolderSize:
    def test_nonexistent(self, tmp_path):
        assert get_folder_size(tmp_path / "nope") == 0

    def test_folder_with_files(self, tmp_path):
        (tmp_path / "a.txt").write_text("hello")
        (tmp_path / "b.txt").write_text("world!")
        size = get_folder_size(tmp_path)
        assert size > 0


class TestGetRequiredFieldFromJson:
    def test_simple_dict(self):
        data = {"key": "value"}
        assert get_required_field_from_json(data, "key") == "value"

    def test_nested_dict(self):
        data = {"a": {"b": {"c": 42}}}
        assert get_required_field_from_json(data, "a.b.c") == 42

    def test_list_access(self):
        data = {"items": [10, 20, 30]}
        assert get_required_field_from_json(data, "items.1") == 20

    def test_max_depth_exceeded(self):
        data = {"a": {"b": "val"}}
        with pytest.raises(ValueError, match="Recursive depth exceeded"):
            get_required_field_from_json(data, "a.b", max_depth=0)

    def test_unsupported_type(self):
        with pytest.raises(ValueError, match="Unsupported data type"):
            get_required_field_from_json("string", "key")

    def test_missing_key_returns_none(self):
        data = {"a": {"b": 1}}
        assert get_required_field_from_json(data, "a.missing") is None
        assert get_required_field_from_json(data, "missing") is None

    def test_missing_list_index_returns_none(self):
        data = {"items": [10]}
        assert get_required_field_from_json(data, "items.5") is None


class TestIsRoot:
    @patch("os.name", "nt")
    def test_not_root_on_windows(self):
        assert is_root() is False

    @patch("os.name", "posix")
    @patch("os.getuid", return_value=0)
    def test_root_on_posix(self, mock_uid):
        assert is_root() is True

    @patch("os.name", "posix")
    @patch("os.getuid", return_value=1000)
    def test_non_root_on_posix(self, mock_uid):
        assert is_root() is False
