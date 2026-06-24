"""Regression tests for MindStudio Logo hooks on Python CLI entry points."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.helpers.cli_runner import CliResult, run_module_main

_LOGO_BRAND = "MindStudio"
_LOGO_SLOGAN = "THE END-TO-END TOOLCHAIN TO UNLEASH HUAWEI ASCEND COMPUTE"

_HELP_MODULES = (
    "cli.inference.text_generate",
    "cli.inference.video_generate",
    "cli.inference.throughput_optimizer",
    "cli.inference.model_adapter",
    "optix",
    "serving_cast.main",
    "tools.perf_data_collection.comm_bench.validate_comm_alignment",
    "tools.perf_data_collection.generate_shape_grid",
)


def _streams_contain_logo(result: CliResult) -> bool:
    combined = f"{result.stdout}\n{result.stderr}"
    return _LOGO_BRAND in combined and _LOGO_SLOGAN in combined


def _assert_logo_on_stderr(stderr: str) -> None:
    assert _LOGO_BRAND in stderr
    assert _LOGO_SLOGAN in stderr
    assert stderr.count("=") >= 2


def test_help_suppresses_logo_on_cli_modules() -> None:
    for module_name in _HELP_MODULES:
        result = run_module_main(module_name, ["--help"])
        assert result.returncode == 0, module_name
        assert not _streams_contain_logo(result), module_name


def test_generate_shape_grid_repo_root_resolves_cli_logo() -> None:
    """REPO_ROOT must be msmodeling root (parents[1]), not gitcode (parents[2])."""
    script = Path("tools/perf_data_collection/generate_shape_grid.py").resolve()
    repo_root = script.parent.parents[1]
    assert (repo_root / "cli" / "logo.py").is_file()


def test_generate_shape_grid_main_cli(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Exercise generate_shape_grid.main in-process for CI test_map coverage."""
    monkeypatch.setattr(
        "tools.perf_data_collection.generate_shape_grid.load_csv_files",
        lambda _data_dir: [],
    )
    monkeypatch.setattr(
        "tools.perf_data_collection.generate_shape_grid.run_theory_mode",
        lambda _args, _data_dir, _csv_files: (0, []),
    )
    monkeypatch.setattr(
        "tools.perf_data_collection.generate_shape_grid.clear_progress",
        lambda: None,
    )

    result = run_module_main(
        "tools.perf_data_collection.generate_shape_grid",
        ["--database-path", str(tmp_path), "--rows", "0"],
    )

    assert result.returncode == 0
    assert "Appended 0 rows" in result.stdout
    _assert_logo_on_stderr(result.stderr)


def test_validate_comm_alignment_prints_logo_after_parse() -> None:
    result = run_module_main(
        "tools.perf_data_collection.comm_bench.validate_comm_alignment",
        ["--csv-dir", "/nonexistent-logo-hook-dir"],
    )
    assert result.returncode != 0
    _assert_logo_on_stderr(result.stderr)


def test_model_adapter_main_cli(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Exercise model_adapter.main in-process for CI test_map coverage."""
    doctor_report = tmp_path / "doctor.json"
    doctor_report.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "tensor_cast.adapter.evidence_export.export_evidence_from_doctor_report",
        lambda _report, _output: "version: 1\n",
    )

    result = run_module_main(
        "cli.inference.model_adapter",
        [
            "export-evidence",
            "--doctor-report",
            str(doctor_report),
        ],
    )

    assert result.returncode == 0
    assert "version: 1" in result.stdout
    _assert_logo_on_stderr(result.stderr)


def test_model_adapter_help_entrypoints_do_not_require_check_dependencies() -> None:
    for argv in (["--help"], ["doctor", "--help"], ["verify", "--help"], ["export-evidence", "--help"]):
        result = run_module_main("cli.inference.model_adapter", argv)
        assert result.returncode == 0, argv
        assert "usage:" in (result.stdout + result.stderr).lower()
