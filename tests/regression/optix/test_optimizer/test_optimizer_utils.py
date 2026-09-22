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
from unittest.mock import patch, Mock

import pytest
from loguru import logger

from optix.optimizer.utils import (
    remove_file,
    kill_children,
    kill_process,
    backup,
    close_file_fp,
    get_folder_size,
    get_required_field_from_json,
)


@pytest.fixture
def loguru_capture():
    """Capture loguru messages; pytest caplog only sees stdlib logging."""
    records = []
    handler_id = logger.add(records.append, level="WARNING")
    yield records
    logger.remove(handler_id)


# --------------------------
# Test remove_file
# --------------------------
def test_remove_file_none():
    remove_file(None)


def test_remove_file_nonexistent(tmp_path):
    file_path = tmp_path / "nonexistent.txt"
    remove_file(file_path)
    assert not file_path.exists()


def test_remove_file_regular_file(tmp_path):
    file_path = tmp_path / "test.txt"
    file_path.write_text("hello")
    assert file_path.exists()
    remove_file(file_path)
    assert not file_path.exists()


def test_remove_file_directory(tmp_path):
    dir_path = tmp_path / "subdir"
    dir_path.mkdir()
    (dir_path / "file.txt").write_text("content")
    remove_file(dir_path)
    assert dir_path.exists()


def test_remove_file_directory_with_unremovable_subdir(tmp_path, loguru_capture):
    dir_path = tmp_path / "subdir"
    dir_path.mkdir()
    protected = dir_path / "protected"
    protected.mkdir()
    # Make it non-removable to trigger exception
    with patch("shutil.rmtree", side_effect=OSError("Cannot remove")):
        remove_file(dir_path)
    assert dir_path.exists()
    assert protected.exists()
    assert any("remove file failed" in msg for msg in loguru_capture)


# --------------------------
# Test kill_children
# --------------------------
@patch("psutil.Process")
def test_kill_children(mock_process):
    mock_child = Mock()
    mock_child.is_running.return_value = True
    mock_child.pid = 1234
    mock_child.send_signal = Mock()
    mock_child.wait = Mock(return_value=None)

    kill_children([mock_child])

    mock_child.send_signal.assert_called_with(9)
    mock_child.wait.assert_called_with(10)


@patch("psutil.Process")
def test_kill_children_not_running(mock_process):
    mock_child = Mock()
    mock_child.is_running.return_value = False
    kill_children([mock_child])
    mock_child.send_signal.assert_not_called()


@patch("psutil.Process")
def test_kill_children_exception_on_signal(mock_process, loguru_capture):
    mock_child = Mock()
    mock_child.is_running.return_value = True
    mock_child.pid = 1234
    mock_child.send_signal.side_effect = Exception("Permission denied")
    kill_children([mock_child])
    mock_child.wait.assert_not_called()
    assert any("Failed to kill the 1234 process" in msg for msg in loguru_capture)


@patch("psutil.Process")
def test_kill_children_still_running_after_wait(mock_process, loguru_capture):
    mock_child = Mock()
    mock_child.is_running.side_effect = [True, True]  # still running after wait
    mock_child.pid = 1234
    mock_child.send_signal = Mock()
    mock_child.wait = Mock(return_value=None)
    kill_children([mock_child])
    mock_child.send_signal.assert_called_once_with(9)
    mock_child.wait.assert_called_once_with(10)
    assert any("Failed to kill the 1234 process" in msg for msg in loguru_capture)


# --------------------------
# Test kill_process
# --------------------------
@patch("psutil.process_iter")
@patch("psutil.Process")
def test_kill_process(mock_psutil_process, mock_process_iter):
    # Mock process list
    mock_proc_info = Mock()
    mock_proc_info.info = {"name": "target_process.exe"}
    mock_proc_info.pid = 1001
    mock_process_iter.return_value = [mock_proc_info]

    # Mock children
    child_proc = Mock()
    child_proc.pid = 2001
    mock_psutil_process.return_value.children.return_value = [child_proc]

    kill_process("target_process")

    # Check signals sent
    mock_proc_info.send_signal.assert_called_with(9)
    child_proc.send_signal.assert_called_with(9)


@patch("psutil.process_iter")
def test_kill_process_no_match(mock_process_iter):
    mock_proc_info = Mock()
    mock_proc_info.info = {"name": "other_process.exe"}
    mock_process_iter.return_value = [mock_proc_info]

    with patch("psutil.Process") as mock_psutil_process:
        kill_process("target_process")
        mock_psutil_process.assert_not_called()


# --------------------------
# Test backup
# --------------------------
def test_backup_file_success(tmp_path):
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    src_file = src_dir / "test.txt"
    src_file.write_text("data")

    bak_dir = tmp_path / "bak"
    bak_dir.mkdir()
    class_name = "TestClass"

    backup(src_file, bak_dir, class_name)

    dest_file = bak_dir / class_name / "test.txt"
    assert dest_file.exists()
    assert dest_file.read_text() == "data"


def test_backup_file_permission_denied(tmp_path):
    src_file = tmp_path / "src.txt"
    src_file.write_text("data")
    bak_dir = tmp_path / "bak"
    bak_dir.mkdir()
    # backup() does not swallow PermissionError raised while creating the destination
    with patch("pathlib.Path.mkdir", side_effect=PermissionError("Denied")):
        with pytest.raises(PermissionError):
            backup(src_file, bak_dir, "TestClass")
    assert not (bak_dir / "TestClass" / "src.txt").exists()


def test_backup_file_rule_not_satisfied(tmp_path):
    src_file = tmp_path / "src.txt"
    src_file.write_text("data")
    bak_dir = tmp_path / "bak"  # Do not create bak_dir to trigger the backup early return
    backup(src_file, bak_dir, "TestClass")
    dest_file = bak_dir / "TestClass" / "src.txt"
    assert not dest_file.exists()


def test_backup_directory_success(tmp_path):
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "file1.txt").write_text("data1")
    sub = src_dir / "sub"
    sub.mkdir()
    (sub / "file2.txt").write_text("data2")

    bak_dir = tmp_path / "bak"
    bak_dir.mkdir()

    backup(src_dir, bak_dir, "TestClass")

    dest_dir = bak_dir / "TestClass" / "src"
    assert (dest_dir / "file1.txt").read_text() == "data1"
    assert (dest_dir / "sub" / "file2.txt").read_text() == "data2"


def test_backup_directory_rule_not_satisfied(tmp_path):
    src_dir = tmp_path / "src"  # Do not create src_dir to trigger the backup early return
    bak_dir = tmp_path / "bak"
    bak_dir.mkdir()
    backup(src_dir, bak_dir, "TestClass")
    dest_dir = bak_dir / "TestClass" / "src"
    assert not dest_dir.exists()


def test_backup_max_depth_reached(loguru_capture, tmp_path):
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    bak_dir = tmp_path / "bak"
    bak_dir.mkdir()
    backup(src_dir, bak_dir, "TestClass", max_depth=0, current_depth=0)
    assert not (bak_dir / "TestClass").exists()
    assert any("Reached maximum backup depth 0" in msg for msg in loguru_capture)


# --------------------------
# Test close_file_fp
# --------------------------
def test_close_file_fp_file_object():
    mock_file = Mock()
    close_file_fp(mock_file)
    mock_file.close.assert_called_once()


def test_close_file_fp_file_descriptor():
    with patch("os.close") as mock_os_close:
        close_file_fp(3)
        mock_os_close.assert_called_with(3)


def test_close_file_fp_none():
    close_file_fp(None)  # Should not raise


# --------------------------
# Test get_folder_size
# --------------------------
def test_get_folder_size_empty_dir(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    size = get_folder_size(empty_dir)
    assert size == 0


def test_get_folder_size_with_files(tmp_path):
    dir_path = tmp_path / "data"
    dir_path.mkdir()
    (dir_path / "file1.txt").write_bytes(b"12345")  # 5 bytes
    (dir_path / "file2.txt").write_bytes(b"1234567890")  # 10 bytes

    size = get_folder_size(dir_path)

    assert size == 15


def test_get_folder_size_nonexistent_path():
    size = get_folder_size(Path("/nonexistent/path"))
    assert size == 0


def test_get_required_field_from_json():
    # Case 1: get a field from a dict
    data = {"name": "John", "age": 30, "city": "New York"}
    assert get_required_field_from_json(data, "name") == "John"

    # Case 2: get a field from a nested dict
    data = {"person": {"name": "John", "age": 30, "city": "New York"}}
    assert get_required_field_from_json(data, "person.name") == "John"

    # Case 3: get a field from a list
    data = ["John", 30, "New York"]
    assert get_required_field_from_json(data, "0") == "John"

    # Case 4: get a field from a nested list
    data = [["John", 30, "New York"], ["Jane", 25, "Los Angeles"]]
    assert get_required_field_from_json(data, "1.0") == "Jane"

    # Case 5: get a field from mixed dict/list structures
    data = {"person": {"name": "John", "age": 30, "city": ["New York", "Los Angeles"]}}
    assert get_required_field_from_json(data, "person.city.1") == "Los Angeles"

    # Case 6: get a field from an unsupported data type
    data = "John"
    with pytest.raises(ValueError):
        get_required_field_from_json(data, "name")

    # Case 7: get a field from empty data (missing key returns None)
    data = {}
    assert get_required_field_from_json(data, "name") is None

    # Case 8: get a field from an empty list (out-of-range index returns None)
    data = []
    assert get_required_field_from_json(data, "0") is None

    # Case 9: get a field from an empty nested dict
    data = {"person": {}}
    assert get_required_field_from_json(data, "person.name") is None

    # Case 10: get a field from an empty nested list
    data = {"person": []}
    assert get_required_field_from_json(data, "person.0") is None

    # Case 11: get a field from empty nested dict/list structures
    data = {"person": {"name": "John", "age": 30, "city": []}}
    assert get_required_field_from_json(data, "person.city.0") is None

    # Case 12: get a field from nested dict/list structures
    data = {"person": {"name": "John", "age": 30, "city": ["New York", "Los Angeles"]}}
    assert get_required_field_from_json(data, "person.city.1") == "Los Angeles"
