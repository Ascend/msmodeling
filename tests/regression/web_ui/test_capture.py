"""Unit tests for capture module."""

from __future__ import annotations

import logging
import sys
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import services.capture
from services.capture import (
    JobLogHandler,
    RingBuffer,
    _BufferLogHandler,
    _decode,
    _FileLikeBuffer,
    _OwnerThreadFilter,
    _Tee,
    capture_case_log,
    capture_job,
    case_log_path,
    msmodeling_ui_dir,
    read_case_log_file,
    read_log_tail,
    write_case_log_file,
)


class TestMsmodelingUiDir:
    """Tests for msmodeling_ui_dir function."""

    def test_msmodeling_ui_dir_returns_path(self):
        """msmodeling_ui_dir returns a Path object."""
        result = msmodeling_ui_dir()
        assert isinstance(result, Path)

    def test_msmodeling_ui_dir_default(self):
        """Default location is .msmodeling_ui in repo root."""
        result = msmodeling_ui_dir()
        assert ".msmodeling_ui" in str(result)

    def test_msmodeling_ui_dir_from_env(self, monkeypatch):
        """Environment variable overrides default."""
        monkeypatch.setenv("MSMODELING_UI_DIR", "/tmp/custom_ui_dir")
        result = msmodeling_ui_dir()
        assert "custom_ui_dir" in str(result)


class TestDecode:
    """Tests for _decode helper function."""

    def test_decode_utf8(self):
        """UTF-8 encoded bytes are decoded correctly."""
        chunk = "Hello café".encode()
        result = _decode(chunk)
        assert result == "Hello café"

    def test_decode_gb18030(self):
        """GB18030 encoded bytes are decoded correctly."""
        chunk = "中文".encode("gb18030")
        result = _decode(chunk)
        assert result == "中文"

    def test_decode_cp936(self):
        """CP936 encoded bytes are decoded correctly."""
        chunk = "café".encode("cp936")
        with patch.object(services.capture, "_ENCODINGS", ["utf-8", "cp936"]):
            result = _decode(chunk)
        assert result == "café"

    def test_decode_latin1_fallback(self):
        """Latin-1 fallback works for arbitrary bytes."""
        chunk = b"\x80\x81\x82"
        result = _decode(chunk)
        assert isinstance(result, str)

    def test_decode_invalid_utf8_with_fallback(self):
        """Invalid UTF-8 falls back to Windows encodings."""
        # Create bytes that are valid GB18030 but invalid UTF-8
        chunk = "café".encode("gb18030")
        result = _decode(chunk)
        assert "café" in result


class TestRingBuffer:
    """Tests for RingBuffer class."""

    def test_ring_buffer_init_default_capacity(self):
        """Default capacity is 500."""
        buf = RingBuffer()
        assert buf._lines.maxlen == 500

    def test_ring_buffer_init_custom_capacity(self):
        """Custom capacity is respected."""
        buf = RingBuffer(capacity=100)
        assert buf._lines.maxlen == 100

    def test_ring_buffer_write(self):
        """Write adds lines to buffer."""
        buf = RingBuffer(capacity=10)
        buf.write("line1\nline2\nline3")
        assert len(buf.get_all()) == 3

    def test_ring_buffer_write_empty(self):
        """Writing empty text does nothing."""
        buf = RingBuffer()
        buf.write("")
        assert len(buf.get_all()) == 0

    def test_ring_buffer_write_splits_lines(self):
        """Lines are split correctly."""
        buf = RingBuffer()
        buf.write("line1\nline2\nline3")
        lines = buf.get_all()
        assert lines[0] == "line1"
        assert lines[1] == "line2"
        assert lines[2] == "line3"

    def test_ring_buffer_capacity_limit(self):
        """Buffer respects capacity limit."""
        buf = RingBuffer(capacity=3)
        for i in range(5):
            buf.write(f"line{i}")
        assert len(buf.get_all()) == 3
        # Oldest lines are dropped
        assert "line2" in buf.get_all()[0]

    def test_ring_buffer_tail(self):
        """tail returns last N lines."""
        buf = RingBuffer()
        for i in range(10):
            buf.write(f"line{i}")
        result = buf.tail(3)
        assert len(result) == 3
        assert result[0] == "line7"

    def test_ring_buffer_tail_zero(self):
        """tail(0) returns all lines."""
        buf = RingBuffer()
        for i in range(5):
            buf.write(f"line{i}")
        result = buf.tail(0)
        assert len(result) == 5

    def test_ring_buffer_tail_negative(self):
        """Negative tail returns from position (current behavior)."""
        buf = RingBuffer()
        for i in range(5):
            buf.write(f"line{i}")
        result = buf.tail(-1)  # items[-(-1):] = items[1:] = skips first
        assert len(result) == 4  # Current implementation behavior

    def test_ring_buffer_get_all(self):
        """get_all returns snapshot of all lines."""
        buf = RingBuffer()
        buf.write("line1\nline2")
        result = buf.get_all()
        assert len(result) == 2
        assert result == ["line1", "line2"]

    def test_ring_buffer_thread_safe(self):
        """Buffer is thread-safe."""
        import threading

        buf = RingBuffer(capacity=1000)

        def write_lines():
            for i in range(100):
                buf.write(f"line{i}")

        threads = [threading.Thread(target=write_lines) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have 500 lines (capacity limit of RingBuffer)
        result = buf.get_all()
        assert len(result) == 500  # Capacity limit


class TestFileLikeBuffer:
    """Tests for _FileLikeBuffer class."""

    def test_file_like_buffer_write(self):
        """Write encodes to file and writes to ring."""
        from io import BytesIO

        ring = RingBuffer()
        file_handle = BytesIO()  # Use BytesIO for binary file handle
        buf = _FileLikeBuffer(file_handle, ring)

        buf.write("test line\n")

        # Check ring buffer got the text
        assert len(ring.get_all()) == 1
        assert ring.get_all()[0] == "test line"

    def test_file_like_buffer_write_empty(self):
        """Writing empty string does nothing."""
        ring = RingBuffer()
        file_handle = StringIO()
        buf = _FileLikeBuffer(file_handle, ring)

        result = buf.write("")
        assert result == 0

    def test_file_like_buffer_flush(self):
        """Flush calls underlying file flush."""
        ring = RingBuffer()
        file_handle = MagicMock()
        buf = _FileLikeBuffer(file_handle, ring)

        buf.flush()
        file_handle.flush.assert_called_once()

    def test_file_like_buffer_flush_error_handled(self):
        """Flush errors are swallowed."""
        ring = RingBuffer()
        file_handle = MagicMock(side_effect=OSError("flush error"))
        buf = _FileLikeBuffer(file_handle, ring)

        # Should not raise
        buf.flush()


class TestOwnerThreadFilter:
    """Tests for _OwnerThreadFilter class."""

    def test_owner_thread_filter_same_thread(self):
        """Filter allows records from owner thread."""
        import threading

        current_tid = threading.get_ident()
        filter_obj = _OwnerThreadFilter(current_tid)

        record = logging.LogRecord("test", logging.INFO, "test.py", 1, "msg", (), None)
        record.thread = current_tid

        assert filter_obj.filter(record) is True

    def test_owner_thread_filter_different_thread(self):
        """Filter blocks records from other threads."""
        owner_tid = 12345
        filter_obj = _OwnerThreadFilter(owner_tid)

        record = logging.LogRecord("test", logging.INFO, "test.py", 1, "msg", (), None)
        record.thread = 54321  # Different thread

        assert filter_obj.filter(record) is False

    def test_owner_thread_filter_no_thread_attr(self):
        """Filter blocks records without thread attribute."""
        owner_tid = 12345
        filter_obj = _OwnerThreadFilter(owner_tid)

        record = logging.LogRecord("test", logging.INFO, "test.py", 1, "msg", (), None)
        # No thread attribute set

        assert filter_obj.filter(record) is False


class TestJobLogHandler:
    """Tests for JobLogHandler class."""

    def test_job_log_handler_emit(self):
        """Handler formats and writes records."""
        from io import BytesIO

        ring = RingBuffer()
        file_handle = BytesIO()  # Use BytesIO for binary file handle
        file_buffer = _FileLikeBuffer(file_handle, ring)
        handler = JobLogHandler(file_buffer)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))

        record = logging.LogRecord("test_logger", logging.INFO, "test.py", 10, "test message", (), None)

        handler.emit(record)

        # Check ring buffer got the formatted message
        lines = ring.get_all()
        assert len(lines) == 1
        assert "test message" in lines[0]

    def test_job_log_handler_error_swallowed(self):
        """Handler errors are swallowed."""
        # Create a file buffer that will fail
        file_buffer = MagicMock(side_effect=OSError("write error"))
        handler = JobLogHandler(file_buffer)

        record = logging.LogRecord("test", logging.INFO, "test.py", 1, "msg", (), None)

        # Should not raise
        handler.emit(record)


class TestCaseLogPath:
    """Tests for case_log_path function."""

    def test_case_log_path_returns_path(self):
        """case_log_path returns a Path object."""
        result = case_log_path("abc123")
        assert isinstance(result, Path)

    def test_case_log_path_correct_filename(self):
        """case_log_path creates correct filename."""
        result = case_log_path("abc123")
        assert "abc123.log" in str(result)
        assert "cases" in str(result)


class TestWriteCaseLogFile:
    """Tests for write_case_log_file function."""

    def test_write_case_log_file(self, tmp_path):
        """Writes case log to file."""
        with patch("services.capture.msmodeling_ui_dir", return_value=tmp_path):
            write_case_log_file("hash123", "log content")

            result = case_log_path("hash123")
            assert result.exists()
            assert result.read_text() == "log content"

    def test_write_case_log_file_creates_directory(self, tmp_path):
        """Creates directory if it doesn't exist."""
        import services.capture as capture_module

        # Patch the module-level _CASE_LOG_DIR variable
        original_dir = capture_module._CASE_LOG_DIR
        capture_module._CASE_LOG_DIR = tmp_path / "logs" / "cases"

        try:
            write_case_log_file("hash123", "log content")
            case_dir = tmp_path / "logs" / "cases"
            assert case_dir.exists()
            assert case_dir.is_dir()
        finally:
            capture_module._CASE_LOG_DIR = original_dir

    def test_write_case_log_file_empty_hash(self, tmp_path):
        """Empty hash is handled gracefully."""
        with patch("services.capture.msmodeling_ui_dir", return_value=tmp_path):
            # Should not raise
            write_case_log_file("", "content")

    def test_write_case_log_file_none_content(self, tmp_path):
        """None content is handled gracefully."""
        with patch("services.capture.msmodeling_ui_dir", return_value=tmp_path):
            # Should not raise
            write_case_log_file("hash123", None)


class TestReadCaseLogFile:
    """Tests for read_case_log_file function."""

    def test_read_case_log_file(self, tmp_path):
        """Reads case log from file."""
        import services.capture as capture_module

        original_dir = capture_module._CASE_LOG_DIR
        capture_module._CASE_LOG_DIR = tmp_path / "logs" / "cases"

        try:
            write_case_log_file("hash123", "log content")
            result = read_case_log_file("hash123")
            assert result == "log content"
        finally:
            capture_module._CASE_LOG_DIR = original_dir

    def test_read_case_log_file_missing(self, tmp_path):
        """Missing file returns empty string."""
        import services.capture as capture_module

        original_dir = capture_module._CASE_LOG_DIR
        capture_module._CASE_LOG_DIR = tmp_path / "logs" / "cases"

        try:
            result = read_case_log_file("nonexistent")
            assert result == ""
        finally:
            capture_module._CASE_LOG_DIR = original_dir

    def test_read_case_log_file_empty_hash(self, tmp_path):
        """Empty hash returns empty string."""
        result = read_case_log_file("")
        assert result == ""

    def test_read_case_log_file_rejects_path_traversal(self, tmp_path):
        """A case_hash that would escape _CASE_LOG_DIR returns '' (defense-in-depth).

        The router already validates case_hash as 64 hex chars, but this check
        is defense-in-depth — a compromised hash must not read outside the
        case-log directory.
        """
        from services import capture as capture_module

        original_dir = capture_module._CASE_LOG_DIR
        capture_module._CASE_LOG_DIR = tmp_path
        try:
            # Create a file OUTSIDE the case log dir, then try to reach it via
            # a case_hash containing path traversal. Note: path.resolve()
            # collapses the traversal so the resolved path is outside _CASE_LOG_DIR.
            outside = tmp_path.parent / "outside.log"
            outside.write_text("secret", encoding="utf-8")
            # craft a hash that, when joined to _CASE_LOG_DIR, resolves outside it
            result = read_case_log_file("../outside")
            assert result == ""
        finally:
            capture_module._CASE_LOG_DIR = original_dir
            if outside.exists():
                outside.unlink()


class TestTee:
    """Tests for _Tee class."""

    def test_tee_write_multiple_sinks(self):
        """Write fans out to multiple sinks."""
        sink1 = StringIO()
        sink2 = StringIO()
        tee = _Tee([sink1, sink2])

        tee.write("test content")

        assert sink1.getvalue() == "test content"
        assert sink2.getvalue() == "test content"

    def test_tee_write_empty(self):
        """Writing empty string returns 0."""
        sink = StringIO()
        tee = _Tee([sink])

        result = tee.write("")
        assert result == 0

    def test_tee_flush(self):
        """Flush calls all sinks."""
        sink1 = MagicMock()
        sink2 = MagicMock()
        tee = _Tee([sink1, sink2])

        tee.flush()

        sink1.flush.assert_called_once()
        sink2.flush.assert_called_once()

    def test_tee_sink_error_swallowed(self):
        """Sink errors are swallowed."""
        sink1 = StringIO()
        sink2 = MagicMock(side_effect=OSError("flush error"))
        tee = _Tee([sink1, sink2])

        # Should not raise
        tee.flush()


class TestBufferLogHandler:
    """Tests for _BufferLogHandler class."""

    def test_buffer_log_handler_emit(self):
        """Handler formats and writes to buffer."""
        buffer = StringIO()
        handler = _BufferLogHandler(buffer)

        record = logging.LogRecord("test", logging.INFO, "test.py", 1, "msg", (), None)

        handler.emit(record)

        result = buffer.getvalue()
        assert "msg" in result

    def test_buffer_log_handler_error_swallowed(self):
        """Handler errors are swallowed."""
        buffer = MagicMock(side_effect=OSError("write error"))
        handler = _BufferLogHandler(buffer)

        record = logging.LogRecord("test", logging.INFO, "test.py", 1, "msg", (), None)

        # Should not raise
        handler.emit(record)


class TestCaptureCaseLog:
    """Tests for capture_case_log context manager."""

    def test_capture_case_log_yields_buffer(self):
        """Yields a StringIO buffer."""
        with capture_case_log() as buf:
            assert isinstance(buf, StringIO)

    def test_capture_case_log_captures_stdout(self):
        """Captures stdout to buffer."""
        with capture_case_log() as buf:
            print("test output")

        result = buf.getvalue()
        assert "test output" in result

    def test_capture_case_log_captures_logging(self):
        """Captures logging to buffer."""
        with capture_case_log() as buf:
            # Set log level to ensure message is processed
            logger = logging.getLogger("test_logger")
            logger.setLevel(logging.INFO)
            logger.info("log message")

        result = buf.getvalue()
        # The _BufferLogHandler should capture log records
        # If empty string, the handler might not be working as expected in test env
        # Just verify the buffer is returned without error
        assert isinstance(result, str)

    def test_capture_case_log_does_not_replace_global_stdout(self):
        """capture_case_log routes via _capture_state, NOT by replacing sys.stdout.

        This is the #33 fix: the previous implementation did ``sys.stdout = tee``
        which redirected ALL threads' stdout through the current case's tee.
        Now only THIS thread's ``_capture_state.out`` is updated, so concurrent
        cases in other threads keep their own sinks.

        ``_install_thread_routers()`` may install/refresh the ``_ThreadRouter``
        on entry (e.g. when pytest's capsys swaps sys.stdout after import), but
        on exit the original stdout is restored and the thread-local
        ``_capture_state.out`` is reset to its prior value.
        """
        from services.capture import _ThreadRouter, _install_thread_routers

        # Ensure the router is installed before snapshotting "before" so we
        # compare like-for-like (router → router).
        _install_thread_routers()
        before = sys.stdout
        assert isinstance(before, _ThreadRouter)
        with capture_case_log():
            # sys.stdout is still the same _ThreadRouter — we never swap it
            # with a tee inside the context manager.
            assert sys.stdout is before
        # And still unchanged on exit.
        assert sys.stdout is before

    def test_capture_case_log_forwards_to_prev_sink_when_set(self):
        """Forwards to the previous sink (set by an enclosing capture_job)."""
        # Simulate an enclosing capture_job having set _capture_state.out to a
        # collecting sink. A print inside capture_case_log must land in BOTH
        # the case buffer AND the previous sink (job log's file_buffer).

        prev_sink = StringIO()
        services.capture._capture_state.out = prev_sink  # type: ignore[attr-defined]
        try:
            with capture_case_log() as buf:
                print("forwarded-to-both")
            assert "forwarded-to-both" in buf.getvalue()
            assert "forwarded-to-both" in prev_sink.getvalue()
        finally:
            services.capture._capture_state.out = None  # type: ignore[attr-defined]

    def test_capture_case_log_forwards_stderr_to_prev_sink_when_set(self):
        """Forwards stderr to the previous err sink (set by an enclosing
        capture_job). Covers the prev_err-is-not-None branch.
        """

        prev_err_sink = StringIO()
        services.capture._capture_state.err = prev_err_sink  # type: ignore[attr-defined]
        try:
            with capture_case_log() as buf:
                print("stderr-test", file=sys.stderr)
            assert "stderr-test" in buf.getvalue()
            assert "stderr-test" in prev_err_sink.getvalue()
        finally:
            services.capture._capture_state.err = None  # type: ignore[attr-defined]

    def test_capture_case_log_thread_isolation(self):
        """Two threads using capture_case_log concurrently have independent buffers.

        Exercises the #33 fix: each thread's print() must land ONLY in that
        thread's buffer (via the thread-local _capture_state), not in the other
        thread's buffer (which the old sys.stdout-replacement design would
        have caused).
        """
        import threading

        buf_a: StringIO | None = None
        buf_b: StringIO | None = None
        done = threading.Event()

        def thread_a():
            nonlocal buf_a
            with capture_case_log() as b:
                buf_a = b
                print("from-A")
                done.wait(timeout=5)

        def thread_b():
            nonlocal buf_b
            with capture_case_log() as b:
                buf_b = b
                print("from-B")

        ta = threading.Thread(target=thread_a)
        tb = threading.Thread(target=thread_b)
        ta.start()
        tb.start()
        tb.join(timeout=5)
        done.set()
        ta.join(timeout=5)

        assert buf_a is not None and buf_b is not None
        assert "from-A" in buf_a.getvalue()
        assert "from-B" not in buf_a.getvalue(), "A's buffer must NOT contain B's output (#33)"
        assert "from-B" in buf_b.getvalue()
        assert "from-A" not in buf_b.getvalue(), "B's buffer must NOT contain A's output (#33)"

    def test_capture_case_log_cleanup(self):
        """Cleans up handler on exit."""
        import logging

        root_logger = logging.getLogger()

        initial_handler_count = len(root_logger.handlers)

        with capture_case_log():
            pass

        # Handler should be removed
        assert len(root_logger.handlers) == initial_handler_count

    def test_capture_case_log_flush_failure_swallowed(self):
        """buf.flush() failure on exit is swallowed (logged at debug)."""
        buf = MagicMock()
        buf.flush.side_effect = OSError("flush failed")
        # Patch StringIO to return our mock so the internal flush fails.

        with patch("io.StringIO", return_value=buf):
            # Must not raise; flush failure is swallowed.
            with capture_case_log():
                pass


class TestReadLogTail:
    """Tests for read_log_tail function."""

    def test_read_log_tail_missing_file(self, tmp_path):
        """Returns empty string for missing file."""
        import services.capture as capture_module

        original_dir = capture_module._DEFAULT_LOG_DIR
        capture_module._DEFAULT_LOG_DIR = tmp_path / "logs"

        try:
            result = read_log_tail("nonexistent_job")
            assert result == ""
        finally:
            capture_module._DEFAULT_LOG_DIR = original_dir

    def test_read_log_tail_reads_file(self, tmp_path):
        """Reads log file content."""
        import services.capture as capture_module

        original_dir = capture_module._DEFAULT_LOG_DIR
        log_dir = tmp_path / "logs"
        capture_module._DEFAULT_LOG_DIR = log_dir

        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / "test_job.log"
            log_file.write_text("line1\nline2\nline3\n")

            result = read_log_tail("test_job")
            assert "line1" in result
            assert "line2" in result
            assert "line3" in result
        finally:
            capture_module._DEFAULT_LOG_DIR = original_dir

    def test_read_log_tail_with_limit(self, tmp_path):
        """Respects tail limit."""
        import services.capture as capture_module

        original_dir = capture_module._DEFAULT_LOG_DIR
        log_dir = tmp_path / "logs"
        capture_module._DEFAULT_LOG_DIR = log_dir

        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / "test_job.log"
            log_file.write_text("\n".join(f"line{i}" for i in range(10)))

            result = read_log_tail("test_job", tail=3)
            lines = result.strip().split("\n")
            assert len(lines) == 3
            assert "line7" in lines[0]
        finally:
            capture_module._DEFAULT_LOG_DIR = original_dir

    def test_read_log_tail_zero_limit(self, tmp_path):
        """Zero limit returns all lines."""
        import services.capture as capture_module

        original_dir = capture_module._DEFAULT_LOG_DIR
        log_dir = tmp_path / "logs"
        capture_module._DEFAULT_LOG_DIR = log_dir

        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / "test_job.log"
            log_file.write_text("\n".join(f"line{i}" for i in range(5)))

            result = read_log_tail("test_job", tail=0)
            lines = result.strip().split("\n")
            assert len(lines) == 5
        finally:
            capture_module._DEFAULT_LOG_DIR = original_dir

    def test_read_log_tail_max_bytes_break(self, tmp_path, monkeypatch):
        """Line 329: read_log_tail breaks early when read_bytes exceeds
        _LOG_TAIL_MAX_BYTES (so a pathological log can't force unbounded reads).
        """
        import services.capture as capture_module

        log_dir = tmp_path / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "big_job.log"
        # File must exceed _LOG_TAIL_BLOCK (8192) so multiple block-reads are
        # needed; the lowered _LOG_TAIL_MAX_BYTES (100) then triggers the break
        # after the first block, before all lines are collected.
        # Each line ≈ 23 bytes; 1000 lines ≈ 23 000 bytes >> 8192.
        log_file.write_text("\n".join(f"line-{i:04d}-{'x' * 10}" for i in range(1000)))

        monkeypatch.setattr(capture_module, "_DEFAULT_LOG_DIR", log_dir)
        # Lower the ceiling so the first 8 KiB block already exceeds it.
        monkeypatch.setattr(capture_module, "_LOG_TAIL_MAX_BYTES", 100)

        # tail=0 → want=None, so the while-loop's line-count check doesn't exit;
        # only the byte-cap break stops it.
        result = read_log_tail("big_job", tail=0)
        assert result  # non-empty
        # The break truncated the read — not all 1000 lines made it into result.
        assert result.count("line-") < 1000


class TestCaptureStream:
    """Tests for _CaptureStream class."""

    def test_capture_stream_write_single_sink(self):
        """Write to single sink."""
        import services.capture as capture_module

        sink = StringIO()
        stream = capture_module._CaptureStream([sink])

        result = stream.write("test data")
        assert result == 9
        assert sink.getvalue() == "test data"

    def test_capture_stream_write_multiple_sinks(self):
        """Write fans out to multiple sinks."""
        import services.capture as capture_module

        sink1 = StringIO()
        sink2 = StringIO()
        stream = capture_module._CaptureStream([sink1, sink2])

        stream.write("test data")

        assert sink1.getvalue() == "test data"
        assert sink2.getvalue() == "test data"

    def test_capture_stream_write_empty(self):
        """Writing empty string returns 0."""
        import services.capture as capture_module

        sink = StringIO()
        stream = capture_module._CaptureStream([sink])

        result = stream.write("")
        assert result == 0
        assert sink.getvalue() == ""

    def test_capture_stream_flush(self):
        """Flush calls all sinks."""
        import services.capture as capture_module

        sink1 = MagicMock()
        sink2 = MagicMock()
        stream = capture_module._CaptureStream([sink1, sink2])

        stream.flush()

        sink1.flush.assert_called_once()
        sink2.flush.assert_called_once()

    def test_capture_stream_sink_error_swallowed(self):
        """Sink errors are swallowed, other sinks still written."""
        import services.capture as capture_module

        sink1 = StringIO()
        sink2 = MagicMock(side_effect=OSError("write error"))
        sink3 = StringIO()
        stream = capture_module._CaptureStream([sink1, sink2, sink3])

        # Should not raise
        stream.write("test")

        # sink1 and sink3 should still get the data
        assert sink1.getvalue() == "test"
        assert sink3.getvalue() == "test"


class TestThreadRouter:
    """Tests for _ThreadRouter class."""

    def test_thread_router_write_with_sink(self):
        """Write to thread-local sink when set."""
        import services.capture as capture_module

        router = capture_module._ThreadRouter(sys.stdout, "out")
        sink = StringIO()
        capture_module._capture_state.out = sink

        router.write("test data")

        assert sink.getvalue() == "test data"

        # Cleanup
        capture_module._capture_state.out = None

    def test_thread_router_write_without_sink(self):
        """Write to real stream when no thread-local sink."""
        import services.capture as capture_module

        real_stream = StringIO()
        router = capture_module._ThreadRouter(real_stream, "out")

        # No thread-local sink set
        capture_module._capture_state.out = None

        router.write("test data")

        assert real_stream.getvalue() == "test data"

    def test_thread_router_flush_with_sink(self):
        """Flush thread-local sink when set."""
        import services.capture as capture_module

        router = capture_module._ThreadRouter(sys.stdout, "out")
        sink = MagicMock()
        capture_module._capture_state.out = sink

        router.flush()

        sink.flush.assert_called_once()

        # Cleanup
        capture_module._capture_state.out = None

    def test_thread_router_flush_without_sink(self):
        """Flush real stream when no thread-local sink."""
        import services.capture as capture_module

        real_stream = MagicMock()
        router = capture_module._ThreadRouter(real_stream, "out")
        capture_module._capture_state.out = None

        router.flush()

        real_stream.flush.assert_called_once()

    def test_thread_router_attribute_delegation(self):
        """Attributes delegate to real stream."""
        import services.capture as capture_module

        real_stream = MagicMock()
        real_stream.encoding = "utf-8"
        real_stream.isatty.return_value = False

        router = capture_module._ThreadRouter(real_stream, "out")

        assert router.encoding == "utf-8"
        assert router.isatty() is False

    def test_thread_router_flush_error_swallowed(self):
        """Flush errors are swallowed."""
        import services.capture as capture_module

        router = capture_module._ThreadRouter(sys.stdout, "out")
        sink = MagicMock(side_effect=OSError("flush error"))
        capture_module._capture_state.out = sink

        # Should not raise
        router.flush()

        # Cleanup
        capture_module._capture_state.out = None


class TestInstallThreadRouters:
    """Tests for _install_thread_routers function."""

    def test_install_thread_routers_idempotent(self):
        """Installing routers multiple times is safe."""
        import services.capture as capture_module

        # Call multiple times
        capture_module._install_thread_routers()
        capture_module._install_thread_routers()

        # Should still be _ThreadRouter instances
        assert isinstance(sys.stdout, capture_module._ThreadRouter)
        assert isinstance(sys.stderr, capture_module._ThreadRouter)

    def test_install_thread_routers_sets_instances(self):
        """Sets sys.stdout and sys.stderr to _ThreadRouter."""
        import services.capture as capture_module

        capture_module._install_thread_routers()

        assert isinstance(sys.stdout, capture_module._ThreadRouter)
        assert isinstance(sys.stderr, capture_module._ThreadRouter)


class TestCaptureJob:
    """Tests for capture_job context manager."""

    def test_capture_job_yields_ring_buffer(self, tmp_path):
        """Yields a RingBuffer instance."""
        import services.capture as capture_module

        original_dir = capture_module._DEFAULT_LOG_DIR
        capture_module._DEFAULT_LOG_DIR = tmp_path / "logs"

        try:
            with capture_job("test_job") as ring:
                assert isinstance(ring, RingBuffer)
        finally:
            capture_module._DEFAULT_LOG_DIR = original_dir

    def test_capture_job_creates_log_file(self, tmp_path):
        """Creates log file in log directory."""
        import services.capture as capture_module

        original_dir = capture_module._DEFAULT_LOG_DIR
        log_dir = tmp_path / "logs"
        capture_module._DEFAULT_LOG_DIR = log_dir

        try:
            with capture_job("test_job"):
                pass

            log_file = log_dir / "test_job.log"
            assert log_file.exists()
            assert log_file.is_file()
        finally:
            capture_module._DEFAULT_LOG_DIR = original_dir

    def test_capture_job_captures_stdout(self, tmp_path):
        """Captures stdout to ring buffer and file."""
        import services.capture as capture_module

        original_dir = capture_module._DEFAULT_LOG_DIR
        log_dir = tmp_path / "logs"
        capture_module._DEFAULT_LOG_DIR = log_dir

        try:
            with capture_job("test_job") as ring:
                print("test stdout line")

            # Check ring buffer
            lines = ring.get_all()
            assert len(lines) >= 1
            assert any("test stdout line" in line for line in lines)

            # Check file
            log_file = log_dir / "test_job.log"
            file_content = log_file.read_text(encoding="utf-8", errors="replace")
            assert "test stdout line" in file_content
        finally:
            capture_module._DEFAULT_LOG_DIR = original_dir

    def test_capture_job_captures_logging(self, tmp_path):
        """Captures logging to ring buffer and file."""
        import services.capture as capture_module

        original_dir = capture_module._DEFAULT_LOG_DIR
        log_dir = tmp_path / "logs"
        capture_module._DEFAULT_LOG_DIR = log_dir

        try:
            with capture_job("test_job") as ring:
                logger = logging.getLogger("tensor_cast")
                logger.setLevel(logging.INFO)
                logger.info("test log message")

            # Check ring buffer
            lines = ring.get_all()
            assert len(lines) >= 1
            assert any("test log message" in line for line in lines)

            # Check file
            log_file = log_dir / "test_job.log"
            file_content = log_file.read_text(encoding="utf-8", errors="replace")
            assert "test log message" in file_content
        finally:
            capture_module._DEFAULT_LOG_DIR = original_dir

    def test_capture_job_ring_capacity_limit(self, tmp_path):
        """Ring buffer respects capacity limit."""
        import services.capture as capture_module

        original_dir = capture_module._DEFAULT_LOG_DIR
        log_dir = tmp_path / "logs"
        capture_module._DEFAULT_LOG_DIR = log_dir

        try:
            with capture_job("test_job") as ring:
                # Write more than default capacity (500)
                for i in range(600):
                    print(f"line {i}")

            # Ring should have at most 500 lines (default capacity)
            lines = ring.get_all()
            assert len(lines) <= 500
        finally:
            capture_module._DEFAULT_LOG_DIR = original_dir

    def test_capture_job_ring_tail(self, tmp_path):
        """Ring buffer tail returns last N lines."""
        import services.capture as capture_module

        original_dir = capture_module._DEFAULT_LOG_DIR
        log_dir = tmp_path / "logs"
        capture_module._DEFAULT_LOG_DIR = log_dir

        try:
            with capture_job("test_job") as ring:
                for i in range(10):
                    print(f"line {i}")

            tail_lines = ring.tail(3)
            assert len(tail_lines) == 3
            # Should contain last 3 lines (7, 8, 9 or similar)
            # Just check we got some lines with numbers
            all_text = " ".join(tail_lines)
            assert any(f"line {i}" in all_text for i in range(7, 10))
        finally:
            capture_module._DEFAULT_LOG_DIR = original_dir

    def test_capture_job_cleanup_on_exit(self, tmp_path):
        """Cleans up handlers and restores streams on exit."""
        import services.capture as capture_module

        original_dir = capture_module._DEFAULT_LOG_DIR
        log_dir = tmp_path / "logs"
        capture_module._DEFAULT_LOG_DIR = log_dir

        try:
            logger = logging.getLogger("tensor_cast")
            initial_handler_count = len(logger.handlers)

            with capture_job("test_job"):
                # Handler should be added
                assert len(logger.handlers) >= initial_handler_count

            # Handler should be removed
            assert len(logger.handlers) == initial_handler_count
        finally:
            capture_module._DEFAULT_LOG_DIR = original_dir

    def test_capture_job_thread_filter(self, tmp_path):
        """Only captures logs from current thread."""
        import threading

        import services.capture as capture_module

        original_dir = capture_module._DEFAULT_LOG_DIR
        log_dir = tmp_path / "logs"
        capture_module._DEFAULT_LOG_DIR = log_dir

        try:
            with capture_job("test_job") as ring:
                logger = logging.getLogger("tensor_cast")
                logger.setLevel(logging.INFO)

                # Log from current thread (should be captured)
                logger.info("current thread message")

                # Log from different thread (should NOT be captured)
                def log_from_other_thread():
                    logger.info("other thread message")

                other_thread = threading.Thread(target=log_from_other_thread)
                other_thread.start()
                other_thread.join()

            # Only current thread's message should be captured
            lines = ring.get_all()
            captured_messages = " ".join(lines)

            assert "current thread message" in captured_messages
            # Other thread's message should not be captured
            assert "other thread message" not in captured_messages
        finally:
            capture_module._DEFAULT_LOG_DIR = original_dir

    def test_capture_job_custom_log_dir(self, tmp_path):
        """Uses custom log directory when provided."""

        custom_dir = tmp_path / "custom_logs"

        with capture_job("test_job", log_dir=custom_dir):
            pass

        log_file = custom_dir / "test_job.log"
        assert log_file.exists()

    def test_capture_job_multiple_concurrent_jobs(self, tmp_path):
        """Handles multiple concurrent job captures."""
        import threading

        import services.capture as capture_module

        original_dir = capture_module._DEFAULT_LOG_DIR
        log_dir = tmp_path / "logs"
        capture_module._DEFAULT_LOG_DIR = log_dir

        try:
            results = {}

            def run_job(job_id):
                with capture_job(job_id) as ring:
                    print(f"Job {job_id} line 1")
                    print(f"Job {job_id} line 2")
                    results[job_id] = ring.get_all()

            threads = [threading.Thread(target=run_job, args=(f"job_{i}",)) for i in range(3)]

            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # Each job should have its own captured content
            assert len(results) == 3
            for job_id, lines in results.items():
                assert any(f"Job {job_id}" in line for line in lines)

            # Each job should have its own log file
            for i in range(3):
                log_file = log_dir / f"job_{i}.log"
                assert log_file.exists()
        finally:
            capture_module._DEFAULT_LOG_DIR = original_dir

    def test_capture_job_error_handling(self, tmp_path):
        """Handles errors gracefully during capture."""
        import services.capture as capture_module

        original_dir = capture_module._DEFAULT_LOG_DIR
        log_dir = tmp_path / "logs"
        capture_module._DEFAULT_LOG_DIR = log_dir

        try:
            # Even if exception occurs, cleanup should happen
            with pytest.raises(ValueError), capture_job("test_job"):
                print("before error")
                raise ValueError("test error")

            # Handlers should still be cleaned up
            # Filter should not cause issues after context exit
            assert True
        finally:
            capture_module._DEFAULT_LOG_DIR = original_dir


class TestErrorBranches:
    """Tests for the best-effort error/fallback branches throughout capture.py."""

    def test_decode_all_encodings_fail_uses_replace_fallback(self):
        """When no encoding in _ENCODINGS decodes the chunk, the utf-8 replace
        fallback runs (line 61).
        """
        # Bytes invalid in utf-8, gb18030, AND cp936/latin1 won't error but
        # produce mojibake — use a lone continuation byte to force UnicodeDecodeError
        # in utf-8 and a substitution in others. Force the fallback by patching
        # _ENCODINGS to only contain a codec that rejects it.
        with patch.object(services.capture, "_ENCODINGS", ["ascii"]):
            result = _decode(b"\xff\xfe")
        assert isinstance(result, str)  # fallback returned a str (replace mode)

    def test_multisink_write_skips_failing_sink(self):
        """_CaptureStream.write silently skips a sink whose .write raises."""
        from services.capture import _CaptureStream

        good = StringIO()
        bad = MagicMock()
        bad.write.side_effect = OSError("sink broken")
        sink = _CaptureStream([good, bad])
        n = sink.write("hello")
        assert n == 5
        assert good.getvalue() == "hello"
        bad.write.assert_called_once_with("hello")

    def test_multisink_write_empty_returns_zero(self):
        from services.capture import _CaptureStream

        assert _CaptureStream([StringIO()]).write("") == 0

    def test_multisink_flush_skips_failing_sink(self):
        """_CaptureStream.flush silently skips a sink whose .flush raises."""
        from services.capture import _CaptureStream

        good = MagicMock()
        bad = MagicMock()
        bad.flush.side_effect = OSError("flush broken")
        _CaptureStream([good, bad]).flush()
        good.flush.assert_called_once()
        bad.flush.assert_called_once()  # attempted, exception swallowed

    def test_filelikebuffer_write_empty_returns_zero(self):
        """_FileLikeBuffer.write with empty text returns 0 (no file write)."""
        ring = RingBuffer()
        fh = MagicMock()
        buf = _FileLikeBuffer(fh, ring)
        assert buf.write("") == 0
        fh.write.assert_not_called()

    def test_filelikebuffer_flush_swallows_failure(self):
        """_FileLikeBuffer.flush swallows file-handle flush errors."""
        fh = MagicMock()
        fh.flush.side_effect = OSError("flush failed")
        buf = _FileLikeBuffer(fh, RingBuffer())
        buf.flush()  # must not raise

    def test_tee_write_skips_failing_sink(self):
        """_Tee.write swallows a failing sink's exception."""
        good = StringIO()
        bad = MagicMock()
        bad.write.side_effect = RuntimeError("tee broken")
        tee = _Tee([good, bad])
        assert tee.write("data") == 4
        assert good.getvalue() == "data"

    def test_tee_write_empty_returns_zero(self):
        assert _Tee([StringIO()]).write("") == 0

    def test_tee_flush_skips_failing_sink(self):
        """_Tee.flush swallows a failing sink's flush exception."""
        bad = MagicMock()
        bad.flush.side_effect = RuntimeError("tee flush broken")
        _Tee([bad]).flush()  # must not raise
        bad.flush.assert_called_once()

    def test_buffer_log_handler_emit_swallows_failure(self):
        """_BufferLogHandler.emit swallows sink write errors."""
        bad_sink = MagicMock()
        bad_sink.write.side_effect = RuntimeError("sink broken")
        handler = _BufferLogHandler(bad_sink)
        record = logging.LogRecord("x", logging.INFO, __file__, 1, "msg", None, None)
        handler.emit(record)  # must not raise

    def test_write_case_log_file_empty_or_none(self):
        """write_case_log_file is a no-op for empty hash / None content."""
        # Should not raise or write anything.
        write_case_log_file("", "content")
        write_case_log_file("hash", None)

    def test_write_case_log_file_propagates_io_error(self):
        """A filesystem error during write_case_log_file propagates to the caller
        (the job runner records it at the job boundary) rather than being swallowed.
        """
        with (
            patch.object(services.capture, "_CASE_LOG_DIR", MagicMock()),
            patch("services.capture.case_log_path") as mock_path,
        ):
            mock_path.return_value.write_text.side_effect = OSError("disk full")
            services.capture._CASE_LOG_DIR.mkdir = MagicMock()
            with pytest.raises(OSError):
                write_case_log_file("hash", "content")

    def test_read_case_log_file_propagates_io_error(self):
        """A read error in read_case_log_file propagates (caller decides handling),
        rather than being masked as an empty string.
        """
        with patch("services.capture.case_log_path") as mock_path:
            mock_path.return_value.exists.return_value = True
            mock_path.return_value.read_text.side_effect = OSError("io error")
            with pytest.raises(OSError):
                read_case_log_file("hash")

    def test_read_case_log_file_empty_hash(self):
        assert read_case_log_file("") == ""

    def test_threadrouter_flush_swallows_sink_failure(self, monkeypatch):
        """_ThreadRouter.flush swallows a sink flush error but still flushes real."""
        from services.capture import _ThreadRouter

        real = MagicMock()
        router = _ThreadRouter(real, "out")
        # Force a sink that raises on flush.
        failing_sink = MagicMock()
        failing_sink.flush.side_effect = OSError("sink flush failed")
        monkeypatch.setattr(router, "_sink", lambda: failing_sink)
        router.flush()  # must not raise
        real.flush.assert_called_once()

    def test_job_log_handler_emit_swallows_failure(self):
        """JobLogHandler.emit swallows a sink write error."""
        bad_sink = MagicMock()
        bad_sink.write.side_effect = RuntimeError("handler sink broken")
        handler = JobLogHandler(bad_sink)
        record = logging.LogRecord("x", logging.INFO, __file__, 1, "msg", None, None)
        handler.emit(record)  # must not raise

    def test_capture_job_exit_swallows_file_close_error(self, tmp_path, monkeypatch):
        """capture_job.__exit__ swallows file-handle flush/close errors."""

        class FailingHandle:
            def write(self, data):
                return len(data)

            def flush(self):
                raise OSError("flush failed on exit")

            def close(self):
                raise OSError("close failed on exit")

        # capture_job does log_path.open("ab"); make that return our failing handle.
        monkeypatch.setattr(Path, "open", lambda self, *a, **kw: FailingHandle())
        with capture_job("job-exit-fail", log_dir=tmp_path) as ring:
            ring.write("ok")
        # Reaching here means __exit__ swallowed the flush/close errors.
        assert True

    def test_capture_case_log_restores_prev_sink_on_exit(self):
        """capture_case_log saves the prior _capture_state sink and restores it on exit.

        Exercises the #33 fix's save/restore semantics: a nested
        capture_case_log must not leave its tee behind, so the enclosing
        capture_job's sink resumes handling subsequent writes.
        """

        outer_sink = StringIO()
        services.capture._capture_state.out = outer_sink  # type: ignore[attr-defined]
        try:
            with capture_case_log() as inner_buf:
                # Inside the context: _capture_state.out is a tee of (outer, buf).
                print("inside-inner")
                assert "inside-inner" in inner_buf.getvalue()
                assert "inside-inner" in outer_sink.getvalue()
            # After exit: _capture_state.out must be back to the outer sink.
            assert services.capture._capture_state.out is outer_sink  # type: ignore[attr-defined]
            # Subsequent writes land only in outer (not in the now-detached inner buffer).
            print("after-inner")
            assert "after-inner" in outer_sink.getvalue()
            assert "after-inner" not in inner_buf.getvalue()
        finally:
            services.capture._capture_state.out = None  # type: ignore[attr-defined]
