"""Smoke tests for tools/perf_data_collection/op_replay/ scripts."""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

OP_REPLAY_DIR = (
    Path(__file__).resolve().parents[2] / "tools" / "perf_data_collection" / "op_replay"
)


class TestOpReplayScriptsExist:
    EXPECTED_SCRIPTS = [
        "common.py",
        "replay_framework.py",
        "run_all_op.py",
        "MatMulV2_run.py",
        "MatMulV3_run.py",
        "RmsNorm_run.py",
        "SwiGlu_run.py",
        "QuantBatchMatmulV3_run.py",
    ]

    @pytest.mark.parametrize("script", EXPECTED_SCRIPTS)
    def test_script_exists(self, script):
        assert (OP_REPLAY_DIR / script).is_file()


class TestOpReplayArgparse:
    """Verify scripts accept --help without crashing (no NPU required)."""

    SCRIPTS_WITH_HELP = [
        "run_all_op.py",
        "MatMulV2_run.py",
    ]

    @pytest.mark.parametrize("script", SCRIPTS_WITH_HELP)
    def test_help_flag(self, script):
        result = subprocess.run(
            [sys.executable, str(OP_REPLAY_DIR / script), "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0, f"--help failed for {script}: {result.stderr}"
        assert "--device" in result.stdout


class TestCommonModule:
    def test_syntax_valid(self):
        """Verify common.py compiles without import errors (torch is lazy-loaded)."""
        source = (OP_REPLAY_DIR / "common.py").read_text(
            encoding="utf-8",
            errors="ignore",
        )
        ast.parse(source)

    def test_data_dir_points_to_profiling_database(self):
        """Verify DATA_DIR resolves to the correct profiling_database/data/ path."""
        source = (OP_REPLAY_DIR / "common.py").read_text(
            encoding="utf-8",
            errors="ignore",
        )
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "DATA_DIR":
                        source_line = ast.get_source_segment(source, node)
                        assert "profiling_database" in source_line
                        assert "data" in source_line
                        return
        pytest.fail("DATA_DIR assignment not found in common.py")
