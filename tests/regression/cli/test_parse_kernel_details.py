"""Tests for parse_kernel_details.py — build_argparser and CLI argument parsing."""

from pathlib import Path

import pytest

from tools.perf_data_collection.parsers.parse_kernel_details import build_argparser


class TestBuildArgparser:
    """Verify CLI argument parsing for parse_kernel_details.py."""

    def test_returns_argument_parser(self):
        parser = build_argparser()
        assert isinstance(parser, pytest.importorskip("argparse").ArgumentParser)

    def test_profiling_path_is_required(self):
        parser = build_argparser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_profiling_path_accepts_file(self):
        parser = build_argparser()
        args = parser.parse_args(["--profiling-path", "kernel_details.csv"])
        assert args.profiling_path == "kernel_details.csv"

    def test_database_path_default_none(self):
        parser = build_argparser()
        args = parser.parse_args(["--profiling-path", "kernel_details.csv"])
        assert args.database_path is None

    def test_database_path_accepts_path(self):
        parser = build_argparser()
        args = parser.parse_args(["--profiling-path", "kernel_details.csv", "--database-path", "/custom/db"])
        assert args.database_path == Path("/custom/db")

    def test_device_default(self):
        from tools.perf_data_collection.parsers.parse_kernel_details import (
            DEFAULT_DEVICE,
        )

        parser = build_argparser()
        args = parser.parse_args(["--profiling-path", "kernel_details.csv"])
        assert args.device == DEFAULT_DEVICE

    def test_device_accepts_supported(self):
        from tools.perf_data_collection.parsers.parse_kernel_details import (
            SUPPORTED_DEVICES,
        )

        parser = build_argparser()
        for device in SUPPORTED_DEVICES:
            args = parser.parse_args(["--profiling-path", "kernel_details.csv", "--device", device])
            assert args.device == device

    def test_device_rejects_unsupported(self):
        parser = build_argparser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--profiling-path", "kernel_details.csv", "--device", "UNKNOWN_DEVICE"])

    def test_vllm_version_default_none(self):
        parser = build_argparser()
        args = parser.parse_args(["--profiling-path", "kernel_details.csv"])
        assert args.vllm_version is None

    def test_vllm_version_accepts_valid(self):
        parser = build_argparser()
        args = parser.parse_args(["--profiling-path", "kernel_details.csv", "--vllm-version", "0.18.0"])
        assert args.vllm_version == "0.18.0"

    def test_vllm_version_accepts_full_dir_name(self):
        parser = build_argparser()
        args = parser.parse_args(
            [
                "--profiling-path",
                "kernel_details.csv",
                "--vllm-version",
                "vllm0.18.0_torch2.9.0_cann8.5",
            ]
        )
        assert args.vllm_version == "vllm0.18.0_torch2.9.0_cann8.5"

    def test_vllm_version_rejects_invalid(self):
        parser = build_argparser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--profiling-path", "kernel_details.csv", "--vllm-version", ""])

    def test_torch_version_default_none(self):
        parser = build_argparser()
        args = parser.parse_args(["--profiling-path", "kernel_details.csv"])
        assert args.torch_version is None

    def test_torch_version_accepts_valid(self):
        parser = build_argparser()
        args = parser.parse_args(["--profiling-path", "kernel_details.csv", "--torch-version", "2.9.0"])
        assert args.torch_version == "2.9.0"

    def test_cann_version_default_none(self):
        parser = build_argparser()
        args = parser.parse_args(["--profiling-path", "kernel_details.csv"])
        assert args.cann_version is None

    def test_cann_version_accepts_valid(self):
        parser = build_argparser()
        args = parser.parse_args(["--profiling-path", "kernel_details.csv", "--cann-version", "8.5"])
        assert args.cann_version == "8.5"

    def test_non_existent_flags_rejected(self):
        parser = build_argparser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--profiling-path", "kernel_details.csv", "--unknown-flag"])
