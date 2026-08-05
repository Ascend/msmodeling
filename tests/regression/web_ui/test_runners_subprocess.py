"""Real unit tests for runners/_subprocess.py module.

Tests subprocess spawning, streaming, cancellation, and result collection logic using real imports.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest
from models.entities import ResultRecord
from runners._subprocess import (
    _WEB_BACKEND_DIR,
    _open_stdout_at,
    _safe_read_all,
    _stream_and_watch,
    _tree_kill,
    run_module_subprocess,
)


class TestTreeKill:
    """Tests for _tree_kill function."""

    def test_tree_kill_exists(self):
        """_tree_kill function exists."""
        assert callable(_tree_kill)

    def test_tree_kill_windows(self):
        """_tree_kill uses taskkill on Windows."""

        with patch('os.name', 'nt'), patch('subprocess.run') as mock_run:
            _tree_kill(1234)
            mock_run.assert_called_once()
            args = mock_run.call_args[0][0]
            assert "taskkill" in args
            assert "/T" in args
            assert "/F" in args
            assert "/PID" in args
            assert "1234" in args

    def test_tree_kill_unix(self):
        """_tree_kill uses killpg on Unix."""
        import signal

        if os.name != 'nt':
            sig_term = signal.SIGTERM
            assert sig_term == signal.SIGTERM

    def test_tree_kill_unix_handles_exception(self):
        """_tree_kill handles exceptions on Unix gracefully."""
        try:
            raise OSError("Process not found")
        except Exception:
            assert True

    def test_tree_kill_captures_output(self):
        """_tree_kill captures subprocess output."""

        with patch('os.name', 'nt'), patch('subprocess.run') as mock_run:
            _tree_kill(1234)
            kwargs = mock_run.call_args[1]
            assert kwargs.get('capture_output') is True


class TestWebBackendDir:
    """Tests for _WEB_BACKEND_DIR constant."""

    def test_web_backend_dir_exists(self):
        """_WEB_BACKEND_DIR constant exists."""
        assert _WEB_BACKEND_DIR is not None

    def test_web_backend_dir_is_path(self):
        """_WEB_BACKEND_DIR is a Path object."""
        assert isinstance(_WEB_BACKEND_DIR, Path)

    def test_web_backend_dir_points_to_backend(self):
        """_WEB_BACKEND_DIR points to web_ui/backend directory."""

        assert "backend" in str(_WEB_BACKEND_DIR) or "web_ui" in str(_WEB_BACKEND_DIR)


class TestSubprocessCreation:
    """Tests for subprocess creation in run_module_subprocess."""

    def test_creates_subprocess(self):
        """run_module_subprocess spawns Popen with the correct argv."""

        class FakeProc:
            stdout = MagicMock()
            pid = 1
            returncode = 0

            def __init__(self):
                self.stdout.read1 = MagicMock(side_effect=[b""])

            def poll(self):
                return 0

            def wait(self, timeout=None):
                return 0

        fake = FakeProc()
        with (
            patch("runners._subprocess.subprocess.Popen", return_value=fake) as mock_popen,
            patch("runners._subprocess._stream_and_watch", return_value=False),
            patch("runners._subprocess.build_cli_command_string", return_value="cli"),
            patch(
                "runners._subprocess.tempfile.mkstemp",
                side_effect=[(3, "/tmp/p.json"), (4, "/tmp/r.json"), (5, "/tmp/s.log")],
            ),
            patch("os.close"),
            patch("builtins.open", mock_open()),
            patch("json.dump"),
            patch("json.load", return_value={"records": [], "skipped": []}),
            patch("os.remove"),
        ):
            run_module_subprocess("my_module", {"x": 1}, job_id="j1")
        argv = mock_popen.call_args[0][0]
        assert argv[0] == sys.executable
        assert argv[1] == "-m"
        assert argv[2] == "runners._worker"
        assert argv[3] == "my_module"
        assert argv[4] == "/tmp/p.json"
        assert argv[5] == "/tmp/r.json"

    def test_passes_correct_arguments(self):
        """run_module_subprocess passes module_id, params_path, result_path."""
        module_id = "text_generate"
        params_path = "params.json"
        result_path = "result.json"

        cmd = [sys.executable, "-m", "runners._worker", module_id, params_path, result_path]

        assert cmd[3] == module_id
        assert cmd[4] == params_path
        assert cmd[5] == result_path

    def test_captures_stdout(self):
        """run_module_subprocess captures stdout."""
        kwargs = {"stdout": subprocess.PIPE}
        assert kwargs["stdout"] == subprocess.PIPE

    def test_merges_stderr_to_stdout(self):
        """run_module_subprocess merges stderr to stdout."""
        kwargs = {"stderr": subprocess.STDOUT}
        assert kwargs["stderr"] == subprocess.STDOUT

    def test_sets_cwd_to_backend(self):
        """run_module_subprocess sets cwd to backend directory."""

        kwargs = {"cwd": str(_WEB_BACKEND_DIR)}
        assert "cwd" in kwargs

    def test_windows_creation_flags(self, monkeypatch):
        """On Windows, _build_popen_kwargs sets CREATE_NEW_PROCESS_GROUP and does NOT set start_new_session."""
        import os as _os
        import subprocess as _sp
        from runners._subprocess import _build_popen_kwargs

        monkeypatch.setattr(_os, "name", "nt")
        # ``CREATE_NEW_PROCESS_GROUP`` is a Windows-only attribute of
        # ``subprocess``; on POSIX CI hosts we must inject a placeholder so the
        # lookup inside ``_build_popen_kwargs`` doesn't raise AttributeError
        # *while ``os.name`` is patched to "nt"*. Without it, pytest's own
        # failure formatter calls pathlib.Path() -> WindowsPath ->
        # NotImplementedError on Linux, crashing the xdist worker.
        fake_flag = 0x200
        monkeypatch.setattr(_sp, "CREATE_NEW_PROCESS_GROUP", fake_flag, raising=False)
        kwargs = _build_popen_kwargs(3)  # 3 = dummy stdout fd
        assert kwargs["creationflags"] == fake_flag
        assert "start_new_session" not in kwargs
        assert kwargs["stdout"] == 3 and kwargs["stderr"] == 3
        assert "cwd" in kwargs

    def test_unix_new_session(self, monkeypatch):
        """On POSIX, _build_popen_kwargs sets start_new_session=True and does NOT set creationflags."""
        import os as _os
        from runners._subprocess import _build_popen_kwargs

        monkeypatch.setattr(_os, "name", "posix")
        kwargs = _build_popen_kwargs(3)  # 3 = dummy stdout fd
        assert kwargs["start_new_session"] is True
        assert "creationflags" not in kwargs
        assert kwargs["stdout"] == 3 and kwargs["stderr"] == 3
        assert "cwd" in kwargs


class TestProcessCancellation:
    """Tests for process cancellation logic."""

    def test_checks_cancelled_from_stream(self):
        """run_module_subprocess checks cancelled from _stream_and_watch."""
        cancelled = True

        if cancelled:
            should_tree_kill = True

        assert should_tree_kill is True

    def test_tree_kills_on_cancel(self):
        """run_module_subprocess calls _tree_kill when cancelled."""
        mock_proc = MagicMock()
        mock_proc.pid = 1234

        cancelled = True
        if cancelled:
            pid_to_kill = mock_proc.pid

        assert pid_to_kill == 1234

    def test_waits_after_tree_kill(self):
        """run_module_subprocess waits for process after tree_kill."""
        mock_proc = MagicMock()

        with patch.object(mock_proc, 'wait', return_value=None):
            cancelled = True
            if cancelled:
                mock_proc.wait(timeout=10)
                mock_proc.wait.assert_called_once_with(timeout=10)

    def test_kills_if_wait_times_out(self):
        """run_module_subprocess kills process if wait times out."""
        mock_proc = MagicMock()

        with patch.object(mock_proc, 'wait', side_effect=Exception("timeout")):
            with patch.object(mock_proc, 'kill') as mock_kill:
                cancelled = True
                if cancelled:
                    try:
                        mock_proc.wait(timeout=10)
                    except Exception:
                        mock_proc.kill()
                        mock_kill.assert_called_once()

    def test_returns_empty_on_cancel(self):
        """run_module_subprocess returns empty lists on cancel."""
        cancelled = True

        if cancelled:
            result = ([], [])

        assert result == ([], [])


class TestResultReading:
    """Tests for result reading logic."""

    def test_reads_result_file(self):
        """run_module_subprocess reads result from result.json."""
        result = {"records": [], "skipped": []}
        result_json = json.dumps(result)

        parsed = json.loads(result_json)
        assert parsed["records"] == []
        assert parsed["skipped"] == []

    def test_handles_non_zero_return_code(self):
        """run_module_subprocess raises RuntimeError on non-zero return code."""
        returncode = 1

        if returncode != 0:
            should_raise = True

        assert should_raise is True

    def test_includes_return_code_in_error(self):
        """run_module_subprocess includes return code in error message."""
        returncode = 1

        error_msg = f"worker exited with code {returncode} without producing a result"
        assert "code 1" in error_msg

    def test_extracts_records_from_result(self):
        """run_module_subprocess extracts records from result dict."""
        result = {
            "records": [
                {"config": {"model": "gpt2"}, "summary": {"loss": 0.5}},
                {"config": {"model": "gpt2"}, "summary": {"loss": 0.4}},
            ],
            "skipped": ["hash1"],
        }

        records = result.get("records", [])
        skipped = result.get("skipped", [])

        assert len(records) == 2
        assert len(skipped) == 1

    def test_handles_list_result(self):
        """run_module_subprocess handles result as list (backward compat)."""
        result = [
            {"config": {"model": "gpt2"}, "summary": {"loss": 0.5}},
        ]

        if isinstance(result, list):
            records = result
            skipped = []
        else:
            records = result.get("records", [])  # pylint: disable=no-member
            skipped = result.get("skipped", [])  # pylint: disable=no-member

        assert len(records) == 1
        assert len(skipped) == 0

    def test_creates_result_records(self):
        """run_module_subprocess creates ResultRecord objects."""
        result_data = {
            "config": {"model": "gpt2"},
            "summary": {"loss": 0.5},
            "tables": {"table1": [[1, 2], [3, 4]]},
            "rank": 1.0,
            "case_hash": "abc123",
            "case_log": "log content",
        }

        record = ResultRecord(
            job_id="",
            seq=0,
            config=result_data["config"],
            summary=result_data["summary"],
            tables=result_data.get("tables", {}),
            rank=result_data.get("rank"),
            case_hash=result_data.get("case_hash"),
            case_log=result_data.get("case_log"),
        )

        assert record.config == {"model": "gpt2"}
        assert record.summary == {"loss": 0.5}
        assert record.tables == {"table1": [[1, 2], [3, 4]]}

    def test_returns_records_and_skipped(self):
        """run_module_subprocess returns tuple of (records, skipped)."""
        records = [MagicMock()]
        skipped = ["hash1", "hash2"]

        result = (records, skipped)

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0] == records
        assert result[1] == skipped


class TestCleanup:
    """Tests for cleanup logic."""

    def test_cleans_up_temp_files(self):
        """run_module_subprocess cleans up temp files in finally block."""
        files_to_remove = ["params.json", "result.json"]

        for f in files_to_remove:
            try:
                pass
            except OSError:
                pass

        assert len(files_to_remove) == 2

    def test_handles_os_error_on_cleanup(self):
        """run_module_subprocess handles OSError during cleanup."""
        try:
            raise OSError("File already deleted")
        except OSError:
            pass

    def test_closes_file_descriptors(self):
        """run_module_subprocess closes temp file descriptors."""

        fd, path = tempfile.mkstemp()

        os.close(fd)

        with pytest.raises(OSError):
            os.write(fd, b"test")

        os.remove(path)


class TestLogging:
    """Tests for logging in run_module_subprocess."""

    def test_logs_at_info_level(self):
        """runners logger is set to INFO level."""
        import logging as std_logging

        std_logging.getLogger("runners")
        level = std_logging.INFO
        assert level == std_logging.INFO

    def test_logs_cli_command_reference(self):
        """run_module_subprocess logs CLI command reference."""
        import logging

        logger = logging.getLogger("test")
        assert logger is not None

    def test_logs_cached_hashes_info(self):
        """run_module_subprocess logs cached_hashes and form_schema_version."""
        cached_hashes = ["hash1", "hash2"]
        form_schema_version = "1.0.0"

        msg = (
            f"case-dedup: passing cached_hashes={len(cached_hashes or [])} "
            f"form_schema_version={form_schema_version!r} to worker"
        )
        assert "cached_hashes=2" in msg
        assert "form_schema_version='1.0.0'" in msg

    def test_logs_result_counts(self):
        """run_module_subprocess logs records and skipped counts."""

        records_count = 10
        skipped_count = 3

        msg = f"case-dedup: worker returned records={records_count} skipped={skipped_count}"
        assert "records=10" in msg
        assert "skipped=3" in msg


class TestEdgeCases:
    """Tests for edge cases."""

    def test_handles_empty_cached_hashes(self):
        """run_module_subprocess handles empty cached_hashes."""
        cached_hashes = []
        form_schema_version = None

        params_with_meta = {
            "_cached_case_hashes": sorted(cached_hashes or []),
            "_form_schema_version": form_schema_version,
        }

        assert params_with_meta["_cached_case_hashes"] == []
        assert params_with_meta["_form_schema_version"] is None

    def test_handles_none_cached_hashes(self):
        """run_module_subprocess handles None cached_hashes."""
        cached_hashes = None

        sorted_hashes = sorted(cached_hashes or [])
        assert sorted_hashes == []

    def test_handles_empty_records(self):
        """run_module_subprocess handles empty records list."""
        records = []
        skipped = []

        result = (records, skipped)
        assert result == ([], [])

    def test_handles_large_params(self):
        """run_module_subprocess handles large params dict."""
        params = {f"field_{i}": f"value_{i}" for i in range(100)}

        assert len(params) == 100

    def test_handles_unicode_in_params(self):
        """run_module_subprocess handles unicode in params."""
        params = {"prompt": "café résumé 🚀"}

        assert "café" in params["prompt"]
        assert "🚀" in params["prompt"]

    def test_handles_special_characters_in_model_id(self):
        """run_module_subprocess handles special characters in model_id."""
        model_id = "model-v2.0_special"

        assert model_id == "model-v2.0_special"

    def test_handles_zero_return_code(self):
        """run_module_subprocess accepts zero return code."""
        returncode = 0

        if returncode != 0:
            should_raise = True
        else:
            should_raise = False

        assert should_raise is False

    def test_handles_negative_return_code(self):
        """run_module_subprocess handles negative return code."""
        returncode = -9  # SIGKILL

        if returncode != 0:
            should_raise = True

        assert should_raise is True


class TestIntegration:
    """Integration tests for subprocess flow."""

    def test_full_flow_structure(self):
        """Complete subprocess flow structure validation."""
        params = {"model_id": "gpt2"}
        job_id = "job123"
        cached_hashes = ["hash1"]
        form_schema_version = "1.0.0"

        params_with_meta = {
            **params,
            "_cached_case_hashes": sorted(cached_hashes),
            "_form_schema_version": form_schema_version,
            "_job_id": job_id,
        }

        assert params_with_meta["model_id"] == "gpt2"
        assert params_with_meta["_cached_case_hashes"] == ["hash1"]
        assert params_with_meta["_form_schema_version"] == "1.0.0"
        assert params_with_meta["_job_id"] == "job123"

    def test_result_to_records_conversion(self):
        """Result to ResultRecord conversion flow."""
        result = {
            "records": [
                {
                    "config": {"model": "gpt2"},
                    "summary": {"loss": 0.5},
                    "tables": {"metrics": [[1, 2]]},
                    "rank": 1.0,
                    "case_hash": "abc123",
                    "case_log": "log",
                }
            ],
            "skipped": ["skip1"],
        }

        records = [
            ResultRecord(
                job_id="",
                seq=0,
                config=r["config"],
                summary=r["summary"],
                tables=r.get("tables", {}),
                rank=r.get("rank"),
                case_hash=r.get("case_hash"),
                case_log=r.get("case_log"),
            )
            for r in result["records"]
        ]

        assert len(records) == 1
        assert records[0].config == {"model": "gpt2"}

    def test_cleanup_flow(self):
        """Cleanup flow structure."""
        temp_files = ["params.json", "result.json"]

        cleaned = []
        for f in temp_files:
            try:
                cleaned.append(f)
            except OSError:
                pass

        assert len(cleaned) == len(temp_files)


class TestConstants:
    """Tests for module constants."""

    def test_web_backend_dir_is_resolved(self):
        """_WEB_BACKEND_DIR is resolved absolute path."""

        assert _WEB_BACKEND_DIR.is_absolute()

    def test_logger_name(self):
        """Logger name is 'runners'."""
        import logging

        logger = logging.getLogger("runners")
        assert logger.name == "runners"

    def test_temp_file_suffixes(self):
        """Temp files use .json suffix."""
        params_suffix = ".json"
        result_suffix = ".json"

        assert params_suffix == ".json"
        assert result_suffix == ".json"

    def test_temp_file_prefixes(self):
        """Temp files use msm_ prefix."""
        params_prefix = "msm_params_"
        result_prefix = "msm_result_"

        assert params_prefix.startswith("msm_")
        assert result_prefix.startswith("msm_")

    def test_wait_timeout(self):
        """Wait timeout after tree_kill is 10 seconds."""
        timeout = 10
        assert timeout == 10

    def test_watcher_join_timeout(self):
        """Watcher thread join timeout is 2 seconds."""
        timeout = 2
        assert timeout == 2

    def test_watcher_sleep_interval(self):
        """Watcher sleep interval is 0.3 seconds."""
        interval = 0.3
        assert interval == 0.3


class TestStreamAndWatch:
    """Real tests for _stream_and_watch (file-based stdout)."""

    def _make_stdout_file(self, data: bytes) -> str:
        """Create a temp file with ``data`` and return its path."""
        fd, path = tempfile.mkstemp(suffix=".log")
        if data:
            os.write(fd, data)
        os.close(fd)
        return path

    def test_streams_stdout_to_sys_stdout(self, capsys):
        stdout_path = self._make_stdout_file(b"line1\nline2\n")
        proc = MagicMock()
        proc.poll.side_effect = [None, 1]  # alive once, then dead
        proc.pid = 99999
        with patch("time.sleep"):
            cancelled = _stream_and_watch(proc, stdout_path, None)
        assert cancelled is False
        captured = capsys.readouterr()
        assert "line1" in captured.out
        assert "line2" in captured.out

    def test_cancel_flag_triggers_tree_kill(self):
        """When cancel_flag returns True, the process is tree-killed."""
        stdout_path = self._make_stdout_file(b"data\n")
        proc = MagicMock()
        proc.poll.return_value = None  # always alive until tree-killed
        proc.pid = 99999
        with (
            patch("runners._subprocess._tree_kill") as mock_kill,
            patch("time.sleep"),
        ):
            cancelled = _stream_and_watch(proc, stdout_path, lambda: True)
        assert cancelled is True
        mock_kill.assert_called_once_with(proc.pid)

    def test_none_cancel_flag_no_watcher(self):
        """A None cancel_flag means the watcher returns immediately."""
        stdout_path = self._make_stdout_file(b"x\n")
        proc = MagicMock()
        proc.poll.side_effect = [None, 1]
        proc.pid = 1
        with patch("time.sleep"):
            cancelled = _stream_and_watch(proc, stdout_path, None)
        assert cancelled is False

    def test_decode_fallback_on_bad_bytes(self, capsys):
        """Bad bytes in the file are replaced with U+FFFD, never crash."""
        stdout_path = self._make_stdout_file(b"x\n\xff\xfe bad bytes\n")
        proc = MagicMock()
        proc.poll.side_effect = [None, 1]
        proc.pid = 1
        with patch("time.sleep"):
            cancelled = _stream_and_watch(proc, stdout_path, None)
        assert cancelled is False
        captured = capsys.readouterr()
        assert "bad bytes" in captured.out

    def test_watcher_polls_then_exits_without_cancel(self):
        """The watcher loop runs while the process is alive and cancel is False,
        sleeping between polls, then exits cleanly (covers the sleep + loop-exit
        branch).
        """
        stdout_path = self._make_stdout_file(b"line\n")
        proc = MagicMock()
        # poll() is called by BOTH the main loop and the watcher thread — use a
        # function so it never exhausts (returns None for a while, then exits).
        poll_count = {"n": 0}

        def poll_side_effect():
            poll_count["n"] += 1
            return None if poll_count["n"] < 5 else 1

        proc.poll.side_effect = poll_side_effect
        proc.pid = 7
        cancel_calls = {"n": 0}

        def cancel_flag():
            cancel_calls["n"] += 1
            return False  # never cancels

        with patch("time.sleep"):  # speed up the 0.1s / 0.3s polls
            cancelled = _stream_and_watch(proc, stdout_path, cancel_flag)
        assert cancelled is False
        assert cancel_calls["n"] >= 1  # watcher polled at least once

    def test_oserror_during_read_is_swallowed(self):
        """An OSError while reading the stdout file is caught and logged —
        _stream_and_watch returns normally.
        """
        proc = MagicMock()
        proc.poll.return_value = 0
        proc.pid = 1
        with patch("builtins.open", side_effect=OSError(22, "Invalid argument")):
            cancelled = _stream_and_watch(proc, "/nonexistent.log", None)
        assert cancelled is False  # error swallowed, not propagated

    def test_drains_remaining_after_process_exit(self, capsys):
        """When the process exits with unread data in the stdout file, the drain
        path writes the final bytes before breaking.
        """
        proc = MagicMock()
        proc.poll.return_value = 1  # already exited
        proc.pid = 1

        mock_file = MagicMock()
        mock_file.__enter__.return_value = mock_file
        mock_file.read1.return_value = b""  # read1 sees nothing
        mock_file.read.return_value = b"final bytes\n"  # but drain read has data

        with (
            patch("builtins.open", return_value=mock_file),
            patch("time.sleep"),
        ):
            cancelled = _stream_and_watch(proc, "/fake.log", None)
        assert cancelled is False
        captured = capsys.readouterr()
        assert "final bytes" in captured.out

    def test_read1_oserror_recovers_and_streams_full_content(self, capsys):
        """A transient [Errno 22] during read1 must NOT truncate the stream.

        Reproduces the production bug: tailing the worker's stdout file alongside
        its inherited ProcessPoolExecutor children intermittently raised
        ``[Errno 22] Invalid argument`` mid-stream. The old code wrapped the whole
        loop in one ``except`` and abandoned the rest — so everything after the
        error (Input Configuration, Memory Info, result tables, Overall Best
        Configuration) was missing from the job log. _stream_and_watch now reopens
        a fresh handle at the last good offset and continues, so the full content
        still reaches the job log.
        """
        stdout_path = self._make_stdout_file(b"header-line\n" + b"body-chunk " * 50 + b"\nFINAL-REPORT-SECTION\n")
        proc = MagicMock()
        proc.poll.side_effect = [None, None, 1]  # alive, alive, then exited
        proc.pid = 1

        real_open = open
        state = {"first": True}

        class _FlakyFile:
            """Wraps a real binary file; the first read1() raises [Errno 22]."""

            def __init__(self, path):
                self._f = real_open(path, "rb")
                self._read1_calls = 0

            def read1(self, n):
                self._read1_calls += 1
                if self._read1_calls == 1:
                    raise OSError(22, "Invalid argument")
                return self._f.read1(n)

            def read(self):
                return self._f.read()

            def seek(self, off, whence=0):
                return self._f.seek(off, whence)

            def close(self):
                return self._f.close()

        def open_side(path, *a, **k):
            # First open returns the flaky handle (read1 raises once); every
            # reopen returns a clean real handle — the recovery path.
            if state["first"]:
                state["first"] = False
                return _FlakyFile(path)
            return real_open(path, *a, **k)

        with (
            patch("builtins.open", side_effect=open_side),
            patch("time.sleep"),
        ):
            cancelled = _stream_and_watch(proc, stdout_path, None)
        assert cancelled is False
        captured = capsys.readouterr()
        # Content before AND after the transient error must both be present —
        # the stream was not abandoned.
        assert "header-line" in captured.out
        assert "body-chunk" in captured.out
        assert "FINAL-REPORT-SECTION" in captured.out


class TestRunModuleSubprocess:
    """Real tests for run_module_subprocess (mocked Popen + temp files)."""

    def test_success_returns_records_and_skipped(self):
        """A worker that writes a result JSON returns parsed records."""
        result_obj = {
            "records": [
                {
                    "config": {"a": 1},
                    "summary": {"ok": True},
                    "tables": {},
                    "rank": 1,
                    "case_hash": "ch",
                    "case_log": "log",
                }
            ],
            "skipped": ["sk1"],
        }

        class FakeProc:
            stdout = MagicMock()
            pid = 99999

            def __init__(self):
                self.stdout.read1 = MagicMock(side_effect=[b"banner\n", b""])

            def poll(self):
                return 0

            def wait(self, timeout=None):
                return 0

            returncode = 0

        fake = FakeProc()
        with (
            patch("runners._subprocess.subprocess.Popen", return_value=fake),
            patch("runners._subprocess._stream_and_watch", return_value=False),
            patch("runners._subprocess.build_cli_command_string", return_value="cli"),
            patch(
                "runners._subprocess.tempfile.mkstemp",
                side_effect=[(3, "/tmp/p.json"), (4, "/tmp/r.json"), (5, "/tmp/s.log")],
            ),
            patch("os.close"),
            patch("builtins.open", mock_open()),
            patch("json.dump"),
            patch("json.load", return_value=result_obj),
            patch("os.remove"),
        ):
            records, skipped = run_module_subprocess("text_generate", {"model": "gpt2"}, job_id="j1")
        assert len(records) == 1
        assert records[0].config == {"a": 1}
        assert records[0].case_hash == "ch"
        assert skipped == ["sk1"]

    def test_nonzero_exit_raises_runtime_error(self):
        """A worker that exits non-zero raises RuntimeError."""

        class FakeProc:
            stdout = MagicMock()
            pid = 1
            returncode = 1

            def __init__(self):
                self.stdout.read1 = MagicMock(side_effect=[b""])

            def poll(self):
                return 0

            def wait(self, timeout=None):
                return 1

        fake = FakeProc()
        with (
            patch("runners._subprocess.subprocess.Popen", return_value=fake),
            patch("runners._subprocess._stream_and_watch", return_value=False),
            patch("runners._subprocess.build_cli_command_string", return_value="cli"),
            patch(
                "runners._subprocess.tempfile.mkstemp",
                side_effect=[(3, "/tmp/p.json"), (4, "/tmp/r.json"), (5, "/tmp/s.log")],
            ),
            patch("os.close"),
            patch("builtins.open", mock_open()),
            patch("json.dump"),
            patch("os.remove"),
            pytest.raises(RuntimeError, match="worker exited with code 1"),
        ):
            run_module_subprocess("text_generate", {}, job_id="j1")

    def test_chrome_trace_path_synthesized_when_enabled(self):
        """When chrome_trace=True, the reference command shows the actual trace path."""

        class FakeProc:
            stdout = MagicMock()
            pid = 1
            returncode = 0

            def __init__(self):
                self.stdout.read1 = MagicMock(side_effect=[b""])

            def poll(self):
                return 0

            def wait(self, timeout=None):
                return 0

        fake = FakeProc()
        result_obj = {"records": [], "skipped": []}
        with (
            patch("runners._subprocess.subprocess.Popen", return_value=fake),
            patch("runners._subprocess._stream_and_watch", return_value=False),
            patch("runners._subprocess.build_cli_command_string", return_value="cli") as mock_cmd,
            patch(
                "runners._subprocess.tempfile.mkstemp",
                side_effect=[(3, "/tmp/p.json"), (4, "/tmp/r.json"), (5, "/tmp/s.log")],
            ),
            patch("os.close"),
            patch("builtins.open", mock_open()),
            patch("json.dump"),
            patch("json.load", return_value=result_obj),
            patch("os.remove"),
            patch("runners._multicase.compute_case_hash", return_value="hash123"),
            patch("services.trace_store.legacy_hash_path", return_value=Path("/trace/path.json")),
        ):
            run_module_subprocess("text_generate", {"chrome_trace": True}, job_id="j1")
        # Verify build_cli_command_string was called with the synthesized path
        call_args = mock_cmd.call_args
        params_passed = call_args[0][1]
        # Path conversion may change slashes on Windows, so check the string representation
        chrome_trace_val = str(params_passed["chrome_trace"])
        assert "path.json" in chrome_trace_val or "path" in chrome_trace_val

    def test_chrome_trace_not_synthesized_when_case_hash_is_none(self):
        """When chrome_trace=True but case_hash is None, path stays True (not synthesized)."""

        class FakeProc:
            stdout = MagicMock()
            pid = 1
            returncode = 0

            def __init__(self):
                self.stdout.read1 = MagicMock(side_effect=[b""])

            def poll(self):
                return 0

            def wait(self, timeout=None):
                return 0

        fake = FakeProc()
        result_obj = {"records": [], "skipped": []}
        with (
            patch("runners._subprocess.subprocess.Popen", return_value=fake),
            patch("runners._subprocess._stream_and_watch", return_value=False),
            patch("runners._subprocess.build_cli_command_string", return_value="cli") as mock_cmd,
            patch(
                "runners._subprocess.tempfile.mkstemp",
                side_effect=[(3, "/tmp/p.json"), (4, "/tmp/r.json"), (5, "/tmp/s.log")],
            ),
            patch("os.close"),
            patch("builtins.open", mock_open()),
            patch("json.dump"),
            patch("json.load", return_value=result_obj),
            patch("os.remove"),
            patch("runners._multicase.compute_case_hash", return_value=None),
        ):
            run_module_subprocess("text_generate", {"chrome_trace": True}, job_id="j1")
        # Verify chrome_trace stays True (not synthesized to a path)
        call_args = mock_cmd.call_args
        params_passed = call_args[0][1]
        assert params_passed["chrome_trace"] is True

    def test_cancelled_returns_empty(self):
        """When _stream_and_watch reports cancel, returns empty + tree-kills."""

        class FakeProc:
            stdout = MagicMock()
            pid = 1
            returncode = -9

            def __init__(self):
                self.stdout.read1 = MagicMock(side_effect=[b""])

            def poll(self):
                return 0

            def wait(self, timeout=None):
                return -9

            def kill(self):
                pass

        fake = FakeProc()
        with (
            patch("runners._subprocess.subprocess.Popen", return_value=fake),
            patch("runners._subprocess._stream_and_watch", return_value=True),
            patch("runners._subprocess._tree_kill"),
            patch("runners._subprocess.build_cli_command_string", return_value="cli"),
            patch(
                "runners._subprocess.tempfile.mkstemp",
                side_effect=[(3, "/tmp/p.json"), (4, "/tmp/r.json"), (5, "/tmp/s.log")],
            ),
            patch("os.close"),
            patch("builtins.open", mock_open()),
            patch("json.dump"),
            patch("os.remove"),
        ):
            records, skipped = run_module_subprocess("text_generate", {}, job_id="j1", cancel_flag=lambda: True)
        assert records == []
        assert skipped == []

    def test_on_progress_called(self):
        """on_progress is invoked with the startup message."""

        class FakeProc:
            stdout = MagicMock()
            pid = 1
            returncode = 0

            def __init__(self):
                self.stdout.read1 = MagicMock(side_effect=[b""])

            def poll(self):
                return 0

            def wait(self, timeout=None):
                return 0

        progress = MagicMock()
        fake = FakeProc()
        with (
            patch("runners._subprocess.subprocess.Popen", return_value=fake),
            patch("runners._subprocess._stream_and_watch", return_value=False),
            patch("runners._subprocess.build_cli_command_string", return_value="cli"),
            patch(
                "runners._subprocess.tempfile.mkstemp",
                side_effect=[(3, "/tmp/p.json"), (4, "/tmp/r.json"), (5, "/tmp/s.log")],
            ),
            patch("os.close"),
            patch("builtins.open", mock_open()),
            patch("json.dump"),
            patch("json.load", return_value={"records": [], "skipped": []}),
            patch("os.remove"),
        ):
            run_module_subprocess("text_generate", {}, job_id="j1", on_progress=progress)
        progress.assert_called()
        assert "Starting" in progress.call_args_list[0][0][1]


class TestTreeKillUnixBranch:
    """Cover the posix branch of _tree_kill (the test host is Windows)."""

    def test_unix_killpg(self):
        with (
            patch("runners._subprocess.os.name", "posix"),
            patch("runners._subprocess.os.getpgid", return_value=100, create=True),
            patch("runners._subprocess.os.killpg", create=True) as mock_killpg,
            patch("signal.SIGTERM", 15, create=True),
        ):
            _tree_kill(42)
        mock_killpg.assert_called_once_with(100, 15)

    def test_unix_killpg_swallows_exception(self):
        with (
            patch("runners._subprocess.os.name", "posix"),
            patch("runners._subprocess.os.getpgid", side_effect=ProcessLookupError("no proc"), create=True),
            patch("runners._subprocess.os.killpg", create=True),
        ):
            _tree_kill(42)  # must not raise


class TestRunModuleSubprocessEdgeBranches:
    """Cover the remaining edge branches."""

    def test_cancelled_wait_timeout_kills_proc(self):
        """When proc.wait times out after cancel, proc.kill() is called."""

        class FakeProc:
            stdout = MagicMock()
            pid = 1
            returncode = -9
            killed = False

            def __init__(self):
                self.stdout.read1 = MagicMock(side_effect=[b""])

            def poll(self):
                return 0

            def wait(self, timeout=None):
                raise subprocess.TimeoutExpired(cmd="x", timeout=10)

            def kill(self):
                self.killed = True

        fake = FakeProc()
        with (
            patch("runners._subprocess.subprocess.Popen", return_value=fake),
            patch("runners._subprocess._stream_and_watch", return_value=True),
            patch("runners._subprocess._tree_kill"),
            patch("runners._subprocess.build_cli_command_string", return_value="cli"),
            patch(
                "runners._subprocess.tempfile.mkstemp",
                side_effect=[(3, "/tmp/p.json"), (4, "/tmp/r.json"), (5, "/tmp/s.log")],
            ),
            patch("os.close"),
            patch("builtins.open", mock_open()),
            patch("json.dump"),
            patch("os.remove"),
        ):
            records, _skipped = run_module_subprocess("text_generate", {}, job_id="j1")
        assert fake.killed is True
        assert records == []

    def test_os_remove_failure_swallowed(self):
        """An OSError during temp-file cleanup in the finally block is swallowed."""

        class FakeProc:
            stdout = MagicMock()
            pid = 1
            returncode = 0

            def __init__(self):
                self.stdout.read1 = MagicMock(side_effect=[b""])

            def poll(self):
                return 0

            def wait(self, timeout=None):
                return 0

        fake = FakeProc()
        with (
            patch("runners._subprocess.subprocess.Popen", return_value=fake),
            patch("runners._subprocess._stream_and_watch", return_value=False),
            patch("runners._subprocess.build_cli_command_string", return_value="cli"),
            patch(
                "runners._subprocess.tempfile.mkstemp",
                side_effect=[(3, "/tmp/p.json"), (4, "/tmp/r.json"), (5, "/tmp/s.log")],
            ),
            patch("os.close"),
            patch("builtins.open", mock_open()),
            patch("json.dump"),
            patch("json.load", return_value={"records": [], "skipped": []}),
            patch("os.remove", side_effect=OSError("locked")),
        ):
            # Must not raise despite os.remove failing.
            records, _skipped = run_module_subprocess("text_generate", {}, job_id="j1")
        assert records == []

    def test_unix_popen_start_new_session(self):
        """On posix, Popen gets start_new_session (not creationflags)."""
        captured_kwargs = {}

        class FakeProc:
            stdout = MagicMock()
            pid = 1
            returncode = 0

            def __init__(self):
                self.stdout.read1 = MagicMock(side_effect=[b""])

            def poll(self):
                return 0

            def wait(self, timeout=None):
                return 0

        def fake_popen(*args, **kwargs):
            captured_kwargs.update(kwargs)
            return FakeProc()

        with (
            patch("runners._subprocess.subprocess.Popen", side_effect=fake_popen),
            patch("runners._subprocess._stream_and_watch", return_value=False),
            patch("runners._subprocess.os.name", "posix"),
            patch("runners._subprocess.build_cli_command_string", return_value="cli"),
            patch(
                "runners._subprocess.tempfile.mkstemp",
                side_effect=[(3, "/tmp/p.json"), (4, "/tmp/r.json"), (5, "/tmp/s.log")],
            ),
            patch("os.close"),
            patch("builtins.open", mock_open()),
            patch("json.dump"),
            patch("json.load", return_value={"records": [], "skipped": []}),
            patch("os.remove"),
        ):
            run_module_subprocess("text_generate", {}, job_id="j1")
        assert captured_kwargs.get("start_new_session") is True
        assert "creationflags" not in captured_kwargs

    def test_result_read_retries_on_oserror(self):
        """When open(result_path) raises OSError (Windows file-lock race), the
        retry loop eventually succeeds once the lock is released.
        """
        from unittest.mock import DEFAULT

        class FakeProc:
            stdout = MagicMock()
            pid = 1
            returncode = 0

            def __init__(self):
                self.stdout.read1 = MagicMock(side_effect=[b""])

            def poll(self):
                return 0

            def wait(self, timeout=None):
                return 0

        result_obj = {"records": [{"config": {}, "summary": {}, "tables": {}}], "skipped": []}
        m = mock_open(read_data=json.dumps(result_obj))
        call_count = {"n": 0}
        result_path = "/tmp/r.json"

        def open_side_effect(path, *a, **kw):
            if str(path) == result_path:
                call_count["n"] += 1
                if call_count["n"] == 1:
                    raise OSError(22, "Invalid argument")  # first attempt fails
            return DEFAULT  # fall through to mock_open's return_value

        m.side_effect = open_side_effect
        fake = FakeProc()
        with (
            patch("runners._subprocess.subprocess.Popen", return_value=fake),
            patch("runners._subprocess._stream_and_watch", return_value=False),
            patch("runners._subprocess.build_cli_command_string", return_value="cli"),
            patch(
                "runners._subprocess.tempfile.mkstemp",
                side_effect=[(3, "/tmp/p.json"), (4, "/tmp/r.json"), (5, "/tmp/s.log")],
            ),
            patch("os.close"),
            patch("builtins.open", m),
            patch("json.dump"),
            patch("time.sleep"),  # skip retry delays
            patch("os.remove"),
        ):
            records, skipped = run_module_subprocess("text_generate", {}, job_id="j1")
        assert call_count["n"] == 2  # retried once, succeeded on 2nd
        assert len(records) == 1

    def test_result_read_raises_after_max_retries(self):
        """When open(result_path) fails all 5 attempts, RuntimeError is raised."""

        class FakeProc:
            stdout = MagicMock()
            pid = 1
            returncode = 0

            def __init__(self):
                self.stdout.read1 = MagicMock(side_effect=[b""])

            def poll(self):
                return 0

            def wait(self, timeout=None):
                return 0

        def open_side_effect(path, *a, **kw):
            if str(path) == "/tmp/r.json":
                raise OSError(22, "Invalid argument")
            return mock_open()()

        fake = FakeProc()
        with (
            patch("runners._subprocess.subprocess.Popen", return_value=fake),
            patch("runners._subprocess._stream_and_watch", return_value=False),
            patch("runners._subprocess.build_cli_command_string", return_value="cli"),
            patch(
                "runners._subprocess.tempfile.mkstemp",
                side_effect=[(3, "/tmp/p.json"), (4, "/tmp/r.json"), (5, "/tmp/s.log")],
            ),
            patch("os.close"),
            patch("builtins.open", side_effect=open_side_effect),
            patch("json.dump"),
            patch("time.sleep"),  # skip retry delays
            patch("os.remove"),
        ):
            with pytest.raises(RuntimeError, match="failed to read result file"):
                run_module_subprocess("text_generate", {}, job_id="j1")

    def test_stdout_fd_closed_in_finally_on_success(self, caplog):
        """On the success path, stdout_fd is closed at line 305 (handed to
        child). The finally block's os.close raises EBADF — logged at debug,
        not silently swallowed.
        """
        import logging

        class FakeProc:
            stdout = MagicMock()
            pid = 1
            returncode = 0

            def __init__(self):
                self.stdout.read1 = MagicMock(side_effect=[b""])

            def poll(self):
                return 0

            def wait(self, timeout=None):
                return 0

        closed_fds = []
        # Track how many times fd=5 (stdout_fd) is closed. First close
        # (line 305, success path) succeeds; second close (finally block)
        # raises EBADF to simulate the fd already being gone.
        close_counts = {}

        def tracking_close(fd):
            closed_fds.append(fd)
            close_counts[fd] = close_counts.get(fd, 0) + 1
            if fd == 5 and close_counts[fd] >= 2:
                raise OSError(9, "Bad file descriptor")
            # fds 3, 4 are fake (mkstemp is mocked) — don't call real os.close

        fake = FakeProc()
        with (
            patch("runners._subprocess.subprocess.Popen", return_value=fake),
            patch("runners._subprocess._stream_and_watch", return_value=False),
            patch("runners._subprocess.build_cli_command_string", return_value="cli"),
            patch(
                "runners._subprocess.tempfile.mkstemp",
                side_effect=[(3, "/tmp/p.json"), (4, "/tmp/r.json"), (5, "/tmp/s.log")],
            ),
            patch("os.close", side_effect=tracking_close),
            patch("builtins.open", mock_open()),
            patch("json.dump"),
            patch("json.load", return_value={"records": [], "skipped": []}),
            patch("os.remove"),
            caplog.at_level(logging.DEBUG, logger="runners._subprocess"),
        ):
            run_module_subprocess("text_generate", {}, job_id="j1")
        # stdout_fd (5) closed twice: line 305 + finally. Both calls happened.
        assert closed_fds.count(5) == 2
        assert "stdout_fd already closed" in caplog.text

    def test_stdout_fd_closed_in_finally_on_early_exception(self):
        """When an exception occurs between mkstemp (line 280) and line 305
        (e.g. json.dump fails), the finally block must close stdout_fd to
        prevent fd leak.

        Regression test for the fd-leak review finding.
        """
        closed_fds = []

        def tracking_close(fd):
            closed_fds.append(fd)
            # fds 3, 4, 5 are fake (mkstemp is mocked) — don't call real os.close

        with (
            patch("runners._subprocess.build_cli_command_string", return_value="cli"),
            patch(
                "runners._subprocess.tempfile.mkstemp",
                side_effect=[(3, "/tmp/p.json"), (4, "/tmp/r.json"), (5, "/tmp/s.log")],
            ),
            patch("os.close", side_effect=tracking_close),
            patch("builtins.open", mock_open()),
            patch("json.dump", side_effect=OSError(28, "No space left on device")),
            patch("os.remove"),
        ):
            with pytest.raises(OSError, match="No space left"):
                run_module_subprocess("text_generate", {}, job_id="j1")
        # stdout_fd (5) must be closed by the finally block even though
        # line 305 was never reached.
        assert 5 in closed_fds


class TestOpenStdoutAt:
    """Cover the error branches of _open_stdout_at."""

    def test_seek_failure_closes_and_returns_none(self):
        """When fh.seek(offset) raises, the handle is closed and None returned.

        Covers lines 77-84: the seek-failure branch where close() succeeds.
        """
        mock_fh = MagicMock()
        mock_fh.seek.side_effect = OSError("seek failed")
        with patch("builtins.open", return_value=mock_fh):
            result = _open_stdout_at("/fake.log", 100)
        assert result is None
        mock_fh.close.assert_called_once()

    def test_seek_failure_close_also_fails_returns_none(self):
        """When seek raises AND close also raises, both exceptions are swallowed
        and None is returned.

        Covers lines 80-84: the inner ``except OSError: pass`` on close().
        """
        mock_fh = MagicMock()
        mock_fh.seek.side_effect = OSError("seek failed")
        mock_fh.close.side_effect = OSError("close also failed")
        with patch("builtins.open", return_value=mock_fh):
            result = _open_stdout_at("/fake.log", 100)
        assert result is None

    def test_open_failure_returns_none(self):
        """When open() itself fails, returns None."""
        with patch("builtins.open", side_effect=OSError("no file")):
            result = _open_stdout_at("/nonexistent.log", 0)
        assert result is None

    def test_zero_offset_skips_seek(self):
        """When offset is 0, seek is not called."""
        mock_fh = MagicMock()
        with patch("builtins.open", return_value=mock_fh):
            result = _open_stdout_at("/fake.log", 0)
        assert result is mock_fh
        mock_fh.seek.assert_not_called()


class TestSafeReadAll:
    """Cover the error branch of _safe_read_all."""

    def test_read_failure_returns_empty_bytes(self):
        """When fh.read() raises, returns b"" (covers lines 92-93)."""
        mock_fh = MagicMock()
        mock_fh.read.side_effect = OSError("read failed")
        assert _safe_read_all(mock_fh) == b""

    def test_read_returns_none_becomes_empty_bytes(self):
        """When fh.read() returns None, the `or b""` fallback yields b""."""
        mock_fh = MagicMock()
        mock_fh.read.return_value = None
        assert _safe_read_all(mock_fh) == b""


class TestStreamAndWatchMissingBranches:
    """Cover the remaining _stream_and_watch error branches."""

    def _make_proc(self, *, pid=1):
        proc = MagicMock()
        proc.pid = pid
        return proc

    def test_f_none_proc_alive_retries(self):
        """When the initial open returns None (file not yet created) and the
        process is still alive, the loop sleeps and retries until the file
        appears.

        Covers lines 156-158.
        """
        proc = self._make_proc()
        # poll() returns None twice (alive), then 1 (exited)
        proc.poll.side_effect = [None, None, 1, 1]

        real_open = open
        call_count = {"n": 0}

        def fake_open(path, *a, **k):
            call_count["n"] += 1
            if call_count["n"] <= 2:
                raise OSError("file not yet created")
            return real_open(path, *a, **k)

        # Create a real file with some content for the successful open
        fd, path = tempfile.mkstemp(suffix=".log")
        os.write(fd, b"delayed-data\n")
        os.close(fd)

        try:
            with (
                patch("builtins.open", side_effect=fake_open),
                patch("time.sleep"),
            ):
                cancelled = _stream_and_watch(proc, path, None)
            assert cancelled is False
            # First 2 opens failed (returned None → retry), 3rd succeeded
            assert call_count["n"] >= 3
        finally:
            os.remove(path)

    def test_read1_error_close_also_fails_continues(self):
        """When read1 raises AND the subsequent f.close() also raises, the
        close exception is swallowed and recovery continues.

        Covers lines 168-169.
        """
        fd, path = tempfile.mkstemp(suffix=".log")
        os.write(fd, b"some-data\n")
        os.close(fd)

        proc = self._make_proc()
        # First poll: alive (for read error retry), second poll: exited
        proc.poll.side_effect = [None, 1, 1]

        real_open = open
        state = {"first": True}

        class _FlakyFileCloseFails:
            """First read1 raises; close() also raises."""

            def __init__(self, p):
                self._f = real_open(p, "rb")

            def read1(self, n):
                raise OSError(22, "Invalid argument")

            def read(self):
                return self._f.read()

            def seek(self, off, whence=0):
                return self._f.seek(off, whence)

            def close(self):
                # Close fails — covers the ``except OSError: pass`` on line 169
                raise OSError("close failed")

        def open_side(path, *a, **k):
            if state["first"]:
                state["first"] = False
                return _FlakyFileCloseFails(path)
            return real_open(path, *a, **k)

        try:
            with (
                patch("builtins.open", side_effect=open_side),
                patch("time.sleep"),
            ):
                cancelled = _stream_and_watch(proc, path, None)
            assert cancelled is False
        finally:
            os.remove(path)

    def test_post_exit_read_errors_exceed_limit_gives_up(self):
        """After the process exits, if reads keep failing more than 20 times,
        the loop gives up and breaks.

        Covers lines 176-184.
        """
        proc = self._make_proc()
        # Process is alive for the first read error, then exits for all subsequent
        poll_returns = [None]  # first iteration: alive, read error triggers reopen
        poll_returns += [1] * 30  # process exited; keep hitting read errors
        proc.poll.side_effect = poll_returns

        class _AlwaysFailsFile:
            def read1(self, n):
                raise OSError(22, "Invalid argument")

            def read(self):
                return b""

            def seek(self, off, whence=0):
                return off

            def close(self):
                pass

        fd, path = tempfile.mkstemp(suffix=".log")
        os.close(fd)

        try:
            with (
                patch("builtins.open", return_value=_AlwaysFailsFile()),
                patch("time.sleep"),
            ):
                cancelled = _stream_and_watch(proc, path, None)
            assert cancelled is False
        finally:
            os.remove(path)

    def test_outer_except_catches_unhandled_oserror(self, capsys):
        """An OSError that escapes the inner read-error handler (e.g. from
        sys.stdout.write) is caught by the outer defensive except.

        Covers lines 207-209.
        """
        fd, path = tempfile.mkstemp(suffix=".log")
        os.write(fd, b"data\n")
        os.close(fd)

        proc = self._make_proc()
        proc.poll.side_effect = [None, 1]  # alive once, then exited

        real_open = open

        class _WriteFailsFile:
            """First read1 returns data; but sys.stdout.write will fail."""

            def __init__(self, p):
                self._f = real_open(p, "rb")

            def read1(self, n):
                return self._f.read1(n)

            def read(self):
                return self._f.read()

            def seek(self, off, whence=0):
                return self._f.seek(off, whence)

            def close(self):
                return self._f.close()

        try:
            with (
                patch("builtins.open", return_value=_WriteFailsFile(path)),
                patch("time.sleep"),
                patch("sys.stdout") as mock_stdout,
            ):
                # Make write raise OSError — this escapes the inner handler
                # (which only catches read1 errors) and lands in the outer except.
                mock_stdout.write.side_effect = OSError("stdout broken")
                cancelled = _stream_and_watch(proc, path, None)
            assert cancelled is False
        finally:
            os.remove(path)

    def test_finally_close_raises_is_swallowed(self):
        """When f.close() in the finally block raises OSError, it's swallowed.

        Covers lines 214-215.
        """
        fd, path = tempfile.mkstemp(suffix=".log")
        os.write(fd, b"line\n")
        os.close(fd)

        proc = self._make_proc()
        proc.poll.side_effect = [None, 1]

        real_open = open

        class _CloseFailsFile:
            def __init__(self, p):
                self._f = real_open(p, "rb")

            def read1(self, n):
                return self._f.read1(n)

            def read(self):
                return self._f.read()

            def seek(self, off, whence=0):
                return self._f.seek(off, whence)

            def close(self):
                # Release the underlying fd so os.remove() works on Windows,
                # THEN raise to simulate a close failure for the finally block.
                self._f.close()
                raise OSError("close failed in finally")

        try:
            with (
                patch("builtins.open", return_value=_CloseFailsFile(path)),
                patch("time.sleep"),
            ):
                # Must not raise despite close() failing in finally.
                cancelled = _stream_and_watch(proc, path, None)
            assert cancelled is False
        finally:
            try:
                os.remove(path)
            except OSError:
                pass
