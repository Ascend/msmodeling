"""Smoke tests for tools/perf_data_collection/op_replay/ scripts."""

import importlib
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

# pylint: disable=no-name-in-module
from tools.perf_data_collection.op_replay import common

OP_REPLAY_DIR = Path(__file__).resolve().parents[3] / "tools" / "perf_data_collection" / "op_replay"
if str(OP_REPLAY_DIR) not in sys.path:
    sys.path.insert(0, str(OP_REPLAY_DIR))

dispatch_ffn = importlib.import_module("DispatchFFNCombine_run")
split_qkv = importlib.import_module("split_qkv_rmsnorm_rope_kernel_run")
op_common = importlib.import_module("common")
run_all_op = importlib.import_module("run_all_op")


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

    def test_build_host_tensor_uses_empty_for_float_dtypes(self, monkeypatch):
        class FakeTorch:
            bool = object()
            int32 = object()
            int64 = object()
            float16 = object()
            bfloat16 = object()
            float32 = object()
            float64 = object()

            def __init__(self):
                self.empty_calls = []
                self.randint_calls = []

            def empty(self, shape, dtype):
                self.empty_calls.append((shape, dtype))
                return ("empty", shape, dtype)

            def randint(self, *args, **kwargs):
                self.randint_calls.append((args, kwargs))
                return ("randint", args, kwargs)

        fake_torch = FakeTorch()
        monkeypatch.setattr(op_common, "get_runtime_modules", lambda: (fake_torch, None))

        tensor = op_common.build_host_tensor((2, 3), fake_torch.bfloat16)

        assert tensor == ("empty", (2, 3), fake_torch.bfloat16)
        assert fake_torch.empty_calls == [((2, 3), fake_torch.bfloat16)]
        assert fake_torch.randint_calls == []


class TestSplitQkvReplay:
    def test_build_case_accepts_legacy_two_output_rows(self, monkeypatch):
        monkeypatch.setattr(split_qkv.op, "resolve_api", lambda: "fake_api")
        monkeypatch.setattr(
            split_qkv,
            "build_input_tensor",
            lambda shape, tensor_format, dtype_name: {
                "shape": shape,
                "format": tensor_format,
                "dtype": dtype_name,
            },
        )
        monkeypatch.setattr(
            split_qkv,
            "build_positions_tensor",
            lambda shape, max_position_embeddings: {
                "shape": shape,
                "max_position_embeddings": max_position_embeddings,
            },
        )
        monkeypatch.setattr(
            split_qkv,
            "build_weight_tensor",
            lambda length, dtype_name: (length, dtype_name),
        )

        case = split_qkv.build_case(
            {
                "Input Shapes": "128,1152;64",
                "Input Formats": "ND;ND",
                "Input Data Types": "DT_BF16;DT_FLOAT",
                "Output Shapes": "128,1024;128,64",
            }
        )

        assert case["kwargs"]["q_hidden_size"] == 1024
        assert case["kwargs"]["kv_hidden_size"] == 64
        assert case["kwargs"]["cos_sin_cache"]["shape"] == (2048, 64)
        assert case["kwargs"]["positions"]["shape"] == (128,)


class TestDispatchFfnReplay:
    def test_multinode_requires_explicit_master_port(self):
        with pytest.raises(ValueError, match="--master-port"):
            dispatch_ffn.launch_torchrun_and_wait(
                32,
                [],
                nproc_per_node=16,
                nnodes=2,
                node_rank=0,
                master_addr="127.0.0.1",
                master_port=None,
            )

    def test_single_node_auto_port_still_launches(self, monkeypatch):
        calls = []
        monkeypatch.setattr(dispatch_ffn, "find_free_port", lambda: 12345)
        monkeypatch.setattr(
            dispatch_ffn.subprocess,
            "run",
            lambda cmd, env, check: (calls.append((cmd, env, check)) or SimpleNamespace(returncode=0)),
        )

        dispatch_ffn.launch_torchrun_and_wait(
            16,
            ["--repeat-count", "1"],
            nproc_per_node=16,
            nnodes=1,
            node_rank=0,
            master_addr="127.0.0.1",
            master_port=None,
        )

        cmd, env, check = calls[0]
        assert "--master_port=12345" in cmd
        assert env["_DFC_AUTO_TORCHRUN"] == "1"
        assert check is True

    def test_extension_load_success_is_cached(self, monkeypatch):
        calls = []
        utils_mod = types.ModuleType("vllm_ascend.utils")
        utils_mod.enable_custom_op = lambda: calls.append("enable")
        package_mod = types.ModuleType("vllm_ascend")

        monkeypatch.setattr(dispatch_ffn, "_EXTENSION_LOAD_STATE", [None])
        monkeypatch.setitem(sys.modules, "vllm_ascend", package_mod)
        monkeypatch.setitem(sys.modules, "vllm_ascend.utils", utils_mod)

        dispatch_ffn.ensure_vllm_ascend_extension_loaded()
        dispatch_ffn.ensure_vllm_ascend_extension_loaded()

        assert calls == ["enable"]
        assert dispatch_ffn._EXTENSION_LOAD_STATE[0] is True

    def test_extension_load_failure_is_cached(self, monkeypatch):
        warnings = []
        imports = []

        utils_mod = types.ModuleType("vllm_ascend.utils")

        def fail_enable_custom_op():
            raise RuntimeError("missing extension")

        def fail_import_module(name):
            imports.append(name)
            raise ImportError(name)

        utils_mod.enable_custom_op = fail_enable_custom_op
        package_mod = types.ModuleType("vllm_ascend")
        package_mod.__file__ = __file__

        monkeypatch.setattr(dispatch_ffn, "_EXTENSION_LOAD_STATE", [None])
        monkeypatch.setattr(
            dispatch_ffn,
            "warn_vllm_ascend_extension_load_failure",
            lambda context, exc: warnings.append((context, type(exc).__name__)),
        )
        monkeypatch.setattr(dispatch_ffn.importlib, "import_module", fail_import_module)
        monkeypatch.setattr(dispatch_ffn.importlib.util, "find_spec", lambda name: None)
        monkeypatch.setitem(sys.modules, "vllm_ascend", package_mod)
        monkeypatch.setitem(sys.modules, "vllm_ascend.utils", utils_mod)

        dispatch_ffn.ensure_vllm_ascend_extension_loaded()
        dispatch_ffn.ensure_vllm_ascend_extension_loaded()

        assert warnings == [("enable_custom_op", "RuntimeError")]
        assert imports == ["vllm_ascend.vllm_ascend_C"]
        assert dispatch_ffn._EXTENSION_LOAD_STATE[0] is False

    def test_extension_load_warning_mentions_context(self, capsys):
        dispatch_ffn.warn_vllm_ascend_extension_load_failure("unit-test", RuntimeError("missing"))

        captured = capsys.readouterr()
        assert "unit-test" in captured.err
        assert "RuntimeError" in captured.err


class TestRunAllOp:
    def test_argparser_parses_replay_options(self, tmp_path):
        parser = run_all_op.build_argparser()

        args = parser.parse_args(
            [
                "--database-path",
                str(tmp_path),
                "--device",
                "TEST_DEVICE",
                "--update-mode",
                "missing-only",
                "--execution-mode",
                "subprocess",
                "--op",
                "MatMulV2",
                "PadV3",
                "--continue-on-error",
            ]
        )

        assert args.database_path == tmp_path
        assert args.device == "TEST_DEVICE"
        assert args.update_mode == "missing-only"
        assert args.execution_mode == "subprocess"
        assert args.op == ["MatMulV2", "PadV3"]
        assert args.continue_on_error is True
