"""Smoke tests for tools/perf_data_collection/op_replay/ scripts."""

import subprocess
import sys
from pathlib import Path

import pytest

# pylint: disable=no-name-in-module
from tools.perf_data_collection.op_replay import common

OP_REPLAY_DIR = Path(__file__).resolve().parents[3] / "tools" / "perf_data_collection" / "op_replay"


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
    def test_module_imports_without_npu(self):
        """common.py imports without NPU; torch is lazy-loaded (stays None until init_runtime)."""
        assert common.torch is None
        assert common.torch_npu is None

    def test_data_dir_points_to_profiling_database(self):
        """DATA_DIR resolves to the profiling_database/data/ tree."""
        assert common.DATA_DIR.parts[-2:] == ("profiling_database", "data")
