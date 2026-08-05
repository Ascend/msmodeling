"""Unit tests for trace_store module."""

from __future__ import annotations

from pathlib import Path

from services.trace_store import (
    copy_all_traces,
    legacy_hash_path,
    materialize_traces,
    trace_dir,
    trace_path,
)


class MockRecord:
    """Mock record for testing."""

    def __init__(self, job_id, seq, case_hash):
        self.job_id = job_id
        self.seq = seq
        self.case_hash = case_hash


class TestTraceDir:
    """Tests for trace_dir function."""

    def test_trace_dir_returns_path(self):
        """trace_dir returns a Path object."""
        result = trace_dir("job123")
        assert isinstance(result, Path)

    def test_trace_dir_correct_path(self):
        """trace_dir creates correct path."""
        result = trace_dir("job123")
        assert "job123" in str(result)
        assert "chrome_traces" in str(result)

    def test_trace_dir_creates_directory(self, tmp_path):
        """trace_dir creates directory if it doesn't exist."""
        from unittest.mock import patch

        with patch("services.trace_store.msmodeling_ui_dir", return_value=tmp_path):
            job_id = "test_job"
            result = trace_dir(job_id)
            assert result.exists()
            assert result.is_dir()


class TestTracePath:
    """Tests for trace_path function."""

    def test_trace_path_returns_path(self):
        """trace_path returns a Path object."""
        result = trace_path("job123", 1)
        assert isinstance(result, Path)

    def test_trace_path_correct_filename(self):
        """trace_path creates correct filename with case seq."""
        result = trace_path("job123", 5)
        assert "case_5.json" in str(result) or "case_5" in str(result)
        assert "job123" in str(result)

    def test_trace_path_zero_seq(self):
        """trace_path handles seq=0."""
        result = trace_path("job123", 0)
        assert "case_0" in str(result)


class TestLegacyHashPath:
    """Tests for legacy_hash_path function."""

    def test_legacy_hash_path_returns_path(self):
        """legacy_hash_path returns a Path object."""
        result = legacy_hash_path("job123", "abc123def")
        assert isinstance(result, Path)

    def test_legacy_hash_path_correct_filename(self):
        """legacy_hash_path creates correct filename with case hash."""
        result = legacy_hash_path("job123", "abc123def")
        assert "abc123def.json" in str(result)
        assert "job123" in str(result)

    def test_legacy_hash_path_empty_hash(self):
        """legacy_hash_path handles empty hash."""
        result = legacy_hash_path("job123", "")
        assert ".json" in str(result)


class TestCopyAllTraces:
    """Tests for copy_all_traces function."""

    def test_copy_all_traces_no_source_dir(self, tmp_path):
        """Returns 0 when source directory doesn't exist."""
        from unittest.mock import patch

        with patch("services.trace_store.msmodeling_ui_dir", return_value=tmp_path):
            result = copy_all_traces("nonexistent_job", "dest_job")
            assert result == 0

    def test_copy_all_traces_copies_files(self, tmp_path):
        """Copies trace files from source to destination."""
        from unittest.mock import patch

        with patch("services.trace_store.msmodeling_ui_dir", return_value=tmp_path):
            # Create source files
            src_dir = trace_dir("src_job")
            src_dir.mkdir(parents=True, exist_ok=True)
            (src_dir / "case_1.json").write_text("{}")
            (src_dir / "case_2.json").write_text("{}")

            result = copy_all_traces("src_job", "dest_job")

            assert result == 2
            dst_dir = trace_dir("dest_job")
            assert (dst_dir / "case_1.json").exists()
            assert (dst_dir / "case_2.json").exists()

    def test_copy_all_traces_skips_existing(self, tmp_path):
        """Does not overwrite existing destination files."""
        from unittest.mock import patch

        with patch("services.trace_store.msmodeling_ui_dir", return_value=tmp_path):
            # Create source and dest files
            src_dir = trace_dir("src_job")
            src_dir.mkdir(parents=True, exist_ok=True)
            (src_dir / "case_1.json").write_text('{"new": "data"}')

            dst_dir = trace_dir("dest_job")
            dst_dir.mkdir(parents=True, exist_ok=True)
            (dst_dir / "case_1.json").write_text('{"old": "data"}')

            result = copy_all_traces("src_job", "dest_job")

            assert result == 0  # No files copied
            content = (dst_dir / "case_1.json").read_text()
            assert content == '{"old": "data"}'

    def test_copy_all_traces_skips_directories(self, tmp_path):
        """Skips subdirectories in source."""
        from unittest.mock import patch

        with patch("services.trace_store.msmodeling_ui_dir", return_value=tmp_path):
            src_dir = trace_dir("src_job")
            src_dir.mkdir(parents=True, exist_ok=True)
            (src_dir / "case_1.json").write_text("{}")
            (src_dir / "subdir").mkdir()

            result = copy_all_traces("src_job", "dest_job")

            assert result == 1  # Only the file

    def test_copy_all_traces_missing_source_returns_zero(self, tmp_path):
        """When the source trace dir is absent, returns 0.

        trace_dir() auto-creates dirs, so we patch it to NOT create (return a
        fixed nonexistent path) to exercise the not-exists guard in copy_all_traces.
        """
        from unittest.mock import patch

        nonexistent = tmp_path / "ghost_src"
        with (
            patch("services.trace_store.msmodeling_ui_dir", return_value=tmp_path),
            patch("services.trace_store.trace_dir", return_value=nonexistent),
        ):
            assert copy_all_traces("src_job", "dest_job") == 0


class TestMaterializeTraces:
    """Tests for materialize_traces function."""

    def test_materialize_traces_fresh_case(self, tmp_path):
        """Renames worker-written hash file to case seq."""
        from unittest.mock import patch

        with patch("services.trace_store.msmodeling_ui_dir", return_value=tmp_path):
            job_id = "test_job"
            case_hash = "abc123"

            # Create legacy hash path file
            legacy = legacy_hash_path(job_id, case_hash)
            legacy.parent.mkdir(parents=True, exist_ok=True)
            legacy.write_text('{"trace": "data"}')

            records = [MockRecord(job_id, 0, case_hash)]
            materialize_traces(job_id, records, set())

            # Check renamed file exists
            result = trace_path(job_id, 0)
            assert result.exists()
            assert result.read_text() == '{"trace": "data"}'
            assert not legacy.exists()

    def test_materialize_traces_cached_case(self, tmp_path):
        """Copies trace from source job for cached cases."""
        from unittest.mock import patch

        with patch("services.trace_store.msmodeling_ui_dir", return_value=tmp_path):
            src_job = "source_job"
            dst_job = "dest_job"
            case_hash = "cached_hash"

            # Create source trace
            src_trace = trace_path(src_job, 5)
            src_trace.parent.mkdir(parents=True, exist_ok=True)
            src_trace.write_text('{"cached": "trace"}')

            records = [MockRecord(src_job, 5, case_hash)]
            materialize_traces(dst_job, records, {case_hash})

            # Check copied file exists
            result = trace_path(dst_job, 0)
            assert result.exists()
            assert result.read_text() == '{"cached": "trace"}'

    def test_materialize_traces_cached_case_missing_source(self, tmp_path):
        """A cached case whose source trace is absent is skipped (no copy, no
        crash) — covers the src.exists()==False branch.
        """
        from unittest.mock import patch

        with patch("services.trace_store.msmodeling_ui_dir", return_value=tmp_path):
            # Source trace intentionally NOT created.
            records = [MockRecord("ghost_src", 0, "missing_hash")]
            materialize_traces("dst_job", records, {"missing_hash"})
            # No destination file written.
            assert not trace_path("dst_job", 0).exists()

    def test_materialize_traces_idempotent(self, tmp_path):
        """When the destination already exists, the orphaned legacy temp is
        removed (not renamed). No error is raised and the existing content
        is preserved.
        """
        from unittest.mock import patch

        with patch("services.trace_store.msmodeling_ui_dir", return_value=tmp_path):
            job_id = "test_job"
            case_hash = "abc123"

            # Create legacy file (the worker temp).
            legacy = legacy_hash_path(job_id, case_hash)
            legacy.parent.mkdir(parents=True, exist_ok=True)
            legacy.write_text('{"new": "data"}')

            # Create the destination with different content (simulates a prior run).
            dest = trace_path(job_id, 0)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text('{"existing": "data"}')

            records = [MockRecord(job_id, 0, case_hash)]
            # Must NOT raise — the legacy temp is unlinked, dest preserved.
            materialize_traces(job_id, records, set())

            # Destination still exists with its original content (never overwritten).
            assert dest.exists()
            assert dest.read_text() == '{"existing": "data"}'
            # Legacy temp file has been cleaned up (orphaned worker file removed).
            assert not legacy.exists()

    def test_materialize_traces_mixed_cases(self, tmp_path):
        """Handles mix of fresh and cached cases."""
        from unittest.mock import patch

        with patch("services.trace_store.msmodeling_ui_dir", return_value=tmp_path):
            src_job = "source_job"
            dst_job = "dest_job"

            # Setup fresh case
            fresh_hash = "fresh_hash"
            fresh_legacy = legacy_hash_path(dst_job, fresh_hash)
            fresh_legacy.parent.mkdir(parents=True, exist_ok=True)
            fresh_legacy.write_text('{"fresh": "data"}')

            # Setup cached case
            cached_hash = "cached_hash"
            src_trace = trace_path(src_job, 3)
            src_trace.parent.mkdir(parents=True, exist_ok=True)
            src_trace.write_text('{"cached": "data"}')

            records = [
                MockRecord(dst_job, 0, fresh_hash),
                MockRecord(src_job, 3, cached_hash),
            ]
            materialize_traces(dst_job, records, {cached_hash})

            # Check both files exist
            assert trace_path(dst_job, 0).exists()
            assert trace_path(dst_job, 1).exists()

    def test_materialize_traces_missing_source(self, tmp_path):
        """Handles missing source files gracefully."""
        from unittest.mock import patch

        with patch("services.trace_store.msmodeling_ui_dir", return_value=tmp_path):
            job_id = "test_job"
            case_hash = "abc123"

            # Don't create any files
            records = [MockRecord(job_id, 0, case_hash)]
            materialize_traces(job_id, records, set())

            # Should not raise, just no file created
            assert not trace_path(job_id, 0).exists()

    def test_materialize_traces_throughput_optimizer_glob_match(self, tmp_path):
        """For throughput_optimizer: when legacy hash path doesn't exist,
        glob for {case_hash}_*.json files and copy the first match (alphabetically).
        """
        from unittest.mock import patch

        with patch("services.trace_store.msmodeling_ui_dir", return_value=tmp_path):
            job_id = "test_job"
            case_hash = "abc123"

            # Create throughput_optimizer-style trace files (with suffixes)
            legacy_dir = legacy_hash_path(job_id, case_hash).parent
            legacy_dir.mkdir(parents=True, exist_ok=True)
            # Create files matching the pattern {case_hash}_*.json
            content_tp1 = '{"trace": "data1"}'
            content_tp2 = '{"trace": "data2"}'
            (legacy_dir / f"{case_hash}_tp1dp4mtp0.json").write_text(content_tp1)
            (legacy_dir / f"{case_hash}_tp2dp2mtp1.json").write_text(content_tp2)

            records = [MockRecord(job_id, 0, case_hash)]
            materialize_traces(job_id, records, set())

            # Check that the first matching file (alphabetically) was copied to case_0.json
            result = trace_path(job_id, 0)
            assert result.exists()
            # The production code sorts matches then picks [0] — tp1 < tp2 alphabetically.
            assert result.read_text() == content_tp1
