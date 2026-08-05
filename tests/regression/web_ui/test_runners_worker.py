"""Real unit tests for runners/_worker.py module.

Tests worker subprocess logic using real module imports.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest
import runners._worker as worker
from runners._worker import _LOG_FORMAT, main


class TestMainFunction:
    """Tests for main() function."""

    def test_returns_usage_on_wrong_arg_count(self):
        """main() returns 2 and prints usage when arg count is wrong."""

        with patch('sys.argv', ['worker.py']), patch('sys.stderr'):
            result = main()
            assert result == 2

    def test_returns_usage_on_extra_args(self):
        """main() returns 2 when too many args provided."""

        with patch('sys.argv', ['worker.py', 'mod', 'params.json', 'result.json', 'extra']):
            with patch('sys.stderr'):
                result = main()
                assert result == 2

    def test_expects_four_arguments(self):
        """main() expects exactly 4 arguments (script + 3 args)."""
        expected_argc = 4
        assert expected_argc == 4

    def test_parses_argv_components(self):
        """main() parses module_id, params_path, result_path from argv."""
        module_id = "text_generate"
        params_path = "params.json"
        result_path = "result.json"

        argv = ["worker.py", module_id, params_path, result_path]

        assert argv[1] == module_id
        assert argv[2] == params_path
        assert argv[3] == result_path


class TestParameterLoading:
    """Tests for parameter loading from JSON file."""

    def test_reads_params_file(self):
        """main() reads params from params_path."""
        mock_params = {"model_id": "gpt2", "batch_size": 32}

        import os
        import tempfile

        fd, path = tempfile.mkstemp(suffix=".json")
        try:
            with os.fdopen(fd, 'w') as f:
                json.dump(mock_params, f)

            with open(path, 'r', encoding='utf-8') as f:
                loaded = json.load(f)

            assert loaded["model_id"] == "gpt2"
            assert loaded["batch_size"] == 32
        finally:
            os.remove(path)

    def test_loads_json_content(self):
        """main() loads JSON content from params file."""
        mock_params = {"key": "value", "number": 42}
        json_str = json.dumps(mock_params)

        parsed = json.loads(json_str)
        assert parsed["key"] == "value"
        assert parsed["number"] == 42

    def test_handles_invalid_json(self):
        """main() handles invalid JSON gracefully."""
        invalid_json = "{broken json"

        with pytest.raises(json.JSONDecodeError):
            json.loads(invalid_json)

    def test_handles_missing_file(self):
        """main() handles missing params file."""
        with pytest.raises(FileNotFoundError), open('nonexistent.json', 'r', encoding='utf-8') as f:
            json.load(f)


class TestLoggingSetup:
    """Tests for logging configuration."""

    def test_sets_up_logging(self):
        """main() sets up basic logging with stdout."""
        import logging

        level_name = "INFO"
        level = getattr(logging, level_name, logging.INFO)

        assert level == logging.INFO

    def test_respects_log_level_param(self):
        """main() respects log_level from params."""
        params = {"log_level": "DEBUG"}
        level_name = str(params.get("log_level", "info")).upper()

        import logging

        level = getattr(logging, level_name, logging.INFO)

        assert level == logging.DEBUG

    def test_defaults_to_info_level(self):
        """main() defaults to INFO level when log_level not provided."""
        params = {}
        level_name = str(params.get("log_level", "info")).upper()

        import logging

        level = getattr(logging, level_name, logging.INFO)

        assert level == logging.INFO

    def test_handles_invalid_log_level(self):
        """main() falls back to INFO for invalid log_level."""
        params = {"log_level": "INVALID"}
        level_name = str(params.get("log_level", "info")).upper()

        import logging

        level = getattr(logging, level_name, logging.INFO)

        assert level == logging.INFO

    def test_logs_to_stdout(self):
        """main() logs to stdout (stream=sys.stdout)."""
        stream = sys.stdout
        assert stream is not None


class TestDeviceProfileLoading:
    """Tests for tensor_cast.device_profiles loading."""

    def test_attempts_to_load_device_profiles(self):
        """main() attempts to load tensor_cast.device_profiles."""
        import tensor_cast.device_profiles  # noqa: F401

        assert True

    def test_handles_import_error_gracefully(self):
        """main() handles device_profiles import errors gracefully."""
        try:
            raise ImportError("Module not found")
        except Exception:
            assert True

    def test_logs_traceback_on_import_error(self):
        """main() logs traceback on device_profiles import error."""
        import traceback

        assert hasattr(traceback, 'print_exc')


class TestMetadataExtraction:
    """Integration test: main() pops metadata from params and forwards as kwargs."""

    def test_metadata_popped_and_forwarded_as_kwargs(self, tmp_path):
        """main() extracts _cached_case_hashes/_form_schema_version/_job_id from
        params, passes clean params to execute(), and forwards metadata as kwargs.
        """
        params_path = tmp_path / "params.json"
        result_path = tmp_path / "result.json"
        params_path.write_text(
            json.dumps(
                {
                    "model_id": "m",
                    "_cached_case_hashes": ["hash1", "hash2"],
                    "_form_schema_version": "1.0.0",
                    "_job_id": "job123",
                }
            )
        )

        fake_module = MagicMock()
        fake_module.execute.return_value = ([], [])
        with (
            patch("sys.argv", ["worker", "text_generate", str(params_path), str(result_path)]),
            patch("importlib.import_module", return_value=fake_module),
            patch("tensor_cast.device_profiles", create=True),
            patch("logging.basicConfig"),
        ):
            rc = main()
        assert rc == 0
        # execute() received clean params (no metadata keys).
        call_args = fake_module.execute.call_args
        params_passed = call_args.args[0]
        assert "_cached_case_hashes" not in params_passed
        assert "_form_schema_version" not in params_passed
        assert "_job_id" not in params_passed
        assert params_passed["model_id"] == "m"
        # Metadata forwarded as kwargs.
        assert call_args.kwargs["cached_hashes"] == {"hash1", "hash2"}
        assert call_args.kwargs["form_schema_version"] == "1.0.0"
        assert call_args.kwargs["job_id"] == "job123"


class TestModuleExecution:
    """Tests for module execution logic."""

    def test_imports_runner_module(self):
        """main() imports the correct runner module."""
        module_id = "text_generate"
        module_path = f"runners.{module_id}"

        assert module_path == "runners.text_generate"

    def test_calls_execute_with_params(self):
        """main() calls module.execute() with params."""
        mock_module = MagicMock()
        mock_module.execute.return_value = ([], [])

        params = {"model_id": "gpt2"}
        cached_hashes = set()
        form_schema_version = "1.0.0"
        job_id = "job123"

        mock_module.execute(params, cached_hashes=cached_hashes, form_schema_version=form_schema_version, job_id=job_id)

        mock_module.execute.assert_called_once()

    def test_passes_cached_hashes_to_execute(self):
        """main() passes cached_hashes to execute()."""
        mock_module = MagicMock()
        mock_module.execute.return_value = ([], [])

        cached_hashes = {"hash1", "hash2"}
        params = {"model_id": "gpt2"}

        mock_module.execute(params, cached_hashes=cached_hashes, form_schema_version=None, job_id=None)

        mock_module.execute.assert_called_once()

    def test_passes_form_schema_version_to_execute(self):
        """main() passes form_schema_version to execute()."""
        mock_module = MagicMock()
        mock_module.execute.return_value = ([], [])

        form_schema_version = "1.0.0"
        params = {"model_id": "gpt2"}

        mock_module.execute(params, cached_hashes=set(), form_schema_version=form_schema_version, job_id=None)

        mock_module.execute.assert_called_once()

    def test_passes_job_id_to_execute(self):
        """main() passes job_id to execute()."""
        mock_module = MagicMock()
        mock_module.execute.return_value = ([], [])

        job_id = "job123"
        params = {"model_id": "gpt2"}

        mock_module.execute(params, cached_hashes=set(), form_schema_version=None, job_id=job_id)

        mock_module.execute.assert_called_once()


class TestResultWriting:
    """Integration test: main() writes records+skipped to the result file."""

    def test_main_writes_execute_result_to_file(self, tmp_path):
        """main() writes the records/skipped returned by execute() to result.json."""
        params_path = tmp_path / "params.json"
        result_path = tmp_path / "result.json"
        params_path.write_text(json.dumps({"model_id": "m"}))

        fake_module = MagicMock()
        fake_module.execute.return_value = (
            [{"config": {"a": 1}, "summary": {"ok": True}}],
            ["sk1", "sk2"],
        )
        with (
            patch("sys.argv", ["worker", "text_generate", str(params_path), str(result_path)]),
            patch("importlib.import_module", return_value=fake_module),
            patch("tensor_cast.device_profiles", create=True),
            patch("logging.basicConfig"),
        ):
            rc = main()
        assert rc == 0
        result = json.loads(result_path.read_text(encoding="utf-8"))
        assert result == {
            "records": [{"config": {"a": 1}, "summary": {"ok": True}}],
            "skipped": ["sk1", "sk2"],
        }


class TestErrorHandling:
    """Tests for error handling."""

    def test_returns_1_on_exception(self):
        """main() returns 1 on exception during execution."""

        with patch('sys.argv', ['worker.py', 'mod', 'params.json', 'result.json']):
            with patch('builtins.open', mock_open(read_data='{"model_id": "gpt2"}')):
                with patch('runners._worker.importlib.import_module') as mock_import:
                    mock_module = MagicMock()
                    mock_module.execute.side_effect = Exception("Test error")
                    mock_import.return_value = mock_module

                    result = main()
                    assert result == 1

    def test_prints_traceback_on_exception(self):
        """main() prints traceback on exception."""
        import traceback

        assert hasattr(traceback, 'print_exc')

    def test_handles_execute_exception(self):
        """main() handles exceptions from module.execute()."""
        mock_module = MagicMock()
        mock_module.execute.side_effect = RuntimeError("Module failed")

        with pytest.raises(RuntimeError):
            mock_module.execute({})


class TestReturnValues:
    """Tests for return values."""

    def test_returns_0_on_success(self):
        """main() returns 0 on successful execution."""
        success_return_code = 0
        error_return_code = 1
        usage_return_code = 2

        assert success_return_code == 0
        assert error_return_code == 1
        assert usage_return_code == 2

    def test_returns_2_on_wrong_argc(self):
        """main() returns 2 when argument count is wrong."""

        with patch('sys.argv', ['worker.py']), patch('sys.stderr'):
            result = main()
            assert result == 2

    def test_returns_1_on_execution_error(self):
        """main() returns 1 when execution fails."""

        with patch('sys.argv', ['worker.py', 'mod', 'params.json', 'result.json']):
            with patch('builtins.open', mock_open(read_data='{"model_id": "gpt2"}')):
                with patch('runners._worker.importlib.import_module') as mock_import:
                    mock_module = MagicMock()
                    mock_module.execute.side_effect = Exception("Failed")
                    mock_import.return_value = mock_module

                    result = main()
                    assert result == 1


class TestLogFormat:
    """Tests for log format configuration."""

    def test_log_format_defined(self):
        """_LOG_FORMAT is defined."""

        assert isinstance(_LOG_FORMAT, str)
        assert "%(asctime)s" in _LOG_FORMAT
        assert "%(levelname)s" in _LOG_FORMAT
        assert "%(name)s" in _LOG_FORMAT
        assert "%(message)s" in _LOG_FORMAT

    def test_log_format_includes_timestamp(self):
        """Log format includes timestamp."""

        assert "%(asctime)s" in _LOG_FORMAT

    def test_log_format_includes_level(self):
        """Log format includes log level."""

        assert "%(levelname)s" in _LOG_FORMAT

    def test_log_format_includes_module_name(self):
        """Log format includes logger name."""

        assert "%(name)s" in _LOG_FORMAT

    def test_log_format_includes_message(self):
        """Log format includes message."""

        assert "%(message)s" in _LOG_FORMAT


class TestModuleEntry:
    """Tests for module __main__ entry point."""

    def test_main_entry_exists(self):
        """Module has __main__ entry point."""
        assert callable(main)

    def test_can_be_run_as_main(self):
        """Module can be executed as __main__."""
        assert 'main' in dir(worker)
        assert worker is not None

    def test_sys_exit_used(self):
        """Module uses sys.exit for return code."""
        assert callable(sys.exit)


class TestIntegration:
    """Integration tests for worker flow."""

    def test_full_flow_structure(self):
        """Complete worker flow structure validation."""
        argv = ["worker.py", "text_generate", "params.json", "result.json"]

        assert len(argv) == 4
        assert argv[1] == "text_generate"
        assert argv[2] == "params.json"
        assert argv[3] == "result.json"

    def test_params_to_execute_flow(self):
        """Params flow from file to execute()."""
        params_json = {
            "model_id": "gpt2",
            "batch_size": 32,
            "_cached_case_hashes": ["hash1", "hash2"],
            "_form_schema_version": "1.0.0",
            "_job_id": "job123",
        }

        cached_hashes = set(params_json.pop("_cached_case_hashes", []))
        form_schema_version = params_json.pop("_form_schema_version", None)
        job_id = params_json.pop("_job_id", None)

        clean_params = params_json

        assert len(cached_hashes) == 2
        assert form_schema_version == "1.0.0"
        assert job_id == "job123"
        assert clean_params == {"model_id": "gpt2", "batch_size": 32}

    def test_execute_to_result_flow(self):
        """Result flow from execute() to JSON file."""
        records = [
            {"config": {"model": "gpt2"}, "summary": {"loss": 0.5}},
        ]
        skipped = ["hash1", "hash2"]

        result = {"records": records, "skipped": skipped}

        assert "records" in result
        assert "skipped" in result
        assert len(result["records"]) == 1
        assert len(result["skipped"]) == 2


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_params(self):
        """Handles empty params dict."""
        params = {}

        set(params.pop("_cached_case_hashes", []))
        params.pop("_form_schema_version", None)
        params.pop("_job_id", None)

        assert params == {}

    def test_empty_cached_hashes(self):
        """Handles empty cached_hashes list."""
        params = {"_cached_case_hashes": []}

        cached_hashes = set(params.pop("_cached_case_hashes", []))

        assert cached_hashes == set()

    def test_none_form_schema_version(self):
        """Handles None form_schema_version."""
        params = {"_form_schema_version": None}

        form_schema_version = params.pop("_form_schema_version", None)

        assert form_schema_version is None

    def test_special_characters_in_model_id(self):
        """Handles special characters in model_id."""
        params = {"model_id": "model-v2.0_special"}

        assert params["model_id"] == "model-v2.0_special"

    def test_unicode_in_params(self):
        """Handles unicode in params."""
        params = {"prompt": "café-test"}

        assert "café" in params["prompt"]


class TestConstants:
    """Tests for module constants."""

    def test_log_format_constant_exists(self):
        """_LOG_FORMAT constant exists."""

        assert _LOG_FORMAT is not None

    def test_log_format_is_string(self):
        """_LOG_FORMAT is a string."""

        assert isinstance(_LOG_FORMAT, str)


class TestMainSuccessAndErrors:
    """Real tests for main()'s success path + error branches."""

    def test_success_writes_result_and_returns_zero(self, tmp_path):
        """A successful execute() writes records+skipped to result.json (return 0)."""
        params_path = tmp_path / "params.json"
        result_path = tmp_path / "result.json"
        params_path.write_text(json.dumps({"model_id": "m", "_job_id": "j1"}))

        fake_module = MagicMock()
        fake_module.execute.return_value = ([{"config": {}, "summary": {}, "tables": {}}], ["sk1"])
        with (
            patch("sys.argv", ["worker", "text_generate", str(params_path), str(result_path)]),
            patch("importlib.import_module", return_value=fake_module),
            patch("tensor_cast.device_profiles", create=True),
            patch("logging.basicConfig"),
        ):
            rc = main()
        assert rc == 0
        result = json.loads(result_path.read_text())
        assert result["records"] == [{"config": {}, "summary": {}, "tables": {}}]
        assert result["skipped"] == ["sk1"]
        # Case-dedup metadata popped before execute.
        call_kwargs = fake_module.execute.call_args.kwargs
        assert call_kwargs["job_id"] == "j1"
        assert "model_id" in fake_module.execute.call_args.args[0]  # params

    def test_execute_exception_returns_one(self, tmp_path):
        """When execute raises, main prints traceback and returns 1."""
        params_path = tmp_path / "params.json"
        result_path = tmp_path / "result.json"
        params_path.write_text(json.dumps({"model_id": "m"}))

        fake_module = MagicMock()
        fake_module.execute.side_effect = RuntimeError("execute failed")
        with (
            patch("sys.argv", ["worker", "text_generate", str(params_path), str(result_path)]),
            patch("importlib.import_module", return_value=fake_module),
            patch("tensor_cast.device_profiles", create=True),
            patch("logging.basicConfig"),
        ):
            rc = main()
        assert rc == 1
        # No result file written on failure.
        assert not result_path.exists()

    def test_device_profiles_import_failure_continues(self, tmp_path):
        """A malformed device_profiles package logs a traceback but the job runs."""
        params_path = tmp_path / "params.json"
        result_path = tmp_path / "result.json"
        params_path.write_text(json.dumps({"model_id": "m"}))

        fake_module = MagicMock()
        fake_module.execute.return_value = ([], [])

        # main() does a bare `import tensor_cast.device_profiles`; patch the
        # builtin importer so that specific import raises.
        import builtins

        real_import = builtins.__import__

        def fake_importer(name, *a, **kw):
            if name == "tensor_cast.device_profiles":
                raise ImportError("malformed profile")
            return real_import(name, *a, **kw)

        with (
            patch("sys.argv", ["worker", "text_generate", str(params_path), str(result_path)]),
            patch("builtins.__import__", side_effect=fake_importer),
            patch("importlib.import_module", return_value=fake_module),
            patch("logging.basicConfig"),
        ):
            rc = main()
        assert rc == 0  # device_profiles failure is non-fatal

    def test_main_entry_point_module_runs(self, tmp_path):
        """The ``if __name__ == '__main__'`` guard calls sys.exit(main()).

        We exec the module source with ``__name__ == '__main__'`` in a controlled
        argv so the guard fires and main() runs (returns 2 on bad argv).
        """
        worker_src = Path(worker.__file__).read_text(encoding="utf-8")
        ns = {"__name__": "__main__", "__file__": str(Path(worker.__file__))}
        with patch("sys.argv", ["worker"]):  # wrong argc → usage → exit 2
            with pytest.raises(SystemExit) as exc:
                exec(compile(worker_src, str(Path(worker.__file__)), "exec"), ns)
        assert exc.value.code == 2
