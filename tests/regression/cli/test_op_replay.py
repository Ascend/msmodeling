"""Smoke tests for tools/perf_data_collection/op_replay/ scripts."""

import importlib
import importlib.util
import subprocess
import sys
from contextlib import contextmanager
from types import SimpleNamespace
import types
from pathlib import Path

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


def test_get_replay_repeat_count_direct_coverage():
    from tools.perf_data_collection.op_replay.common import (
        DEFAULT_REPLAY_REPEAT_COUNT,
        get_replay_repeat_count,
    )

    assert get_replay_repeat_count(5) == 5
    assert get_replay_repeat_count(None) == DEFAULT_REPLAY_REPEAT_COUNT
    with pytest.raises(ValueError, match="--repeat-count must be positive"):
        get_replay_repeat_count(0)


@contextmanager
def op_replay_import_path():
    path = str(OP_REPLAY_DIR)
    inserted = path not in sys.path
    if inserted:
        sys.path.insert(0, path)
    try:
        yield
    finally:
        if inserted:
            sys.path.remove(path)


def import_op_replay_script(script: str):
    module_name = f"_test_op_replay_{Path(script).stem}"
    spec = importlib.util.spec_from_file_location(module_name, OP_REPLAY_DIR / script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    with op_replay_import_path():
        spec.loader.exec_module(module)
    return module


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
        "BatchMatMulV2_run.py",
        "GroupedMatmul_run.py",
        "GroupedMatmulSwigluQuant_run.py",
        "LightningIndexer_run.py",
        "MoeTokenPermute_run.py",
        "MoeTokenUnpermute_run.py",
        "ScatterNdUpdate_run.py",
        "SparseFlashAttention_run.py",
        "TransposeBatchMatMul_run.py",
        "DispatchFFNCombine_run.py",
    ]

    @pytest.mark.parametrize("script", EXPECTED_SCRIPTS)
    def test_script_exists(self, script):
        assert (OP_REPLAY_DIR / script).is_file()


class TestOpReplayImportMap:
    NEW_REPLAY_SCRIPTS = [
        "BatchMatMulV2_run.py",
        "GroupedMatmul_run.py",
        "GroupedMatmulSwigluQuant_run.py",
        "LightningIndexer_run.py",
        "MoeTokenPermute_run.py",
        "MoeTokenUnpermute_run.py",
        "ScatterNdUpdate_run.py",
        "SparseFlashAttention_run.py",
        "TransposeBatchMatMul_run.py",
    ]

    def test_new_replay_script_mains_are_coverage_visible(self, monkeypatch):
        calls = []
        for script in self.NEW_REPLAY_SCRIPTS:
            module = import_op_replay_script(script)
            monkeypatch.setattr(module.op, "main", lambda name=script: calls.append(name))

            module.main()

        assert calls == self.NEW_REPLAY_SCRIPTS


class TestOpReplayArgparse:
    """Verify scripts accept --help without crashing (no NPU required)."""

    SCRIPTS_WITH_HELP = [
        "run_all_op.py",
        "MatMulV2_run.py",
        "BatchMatMulV2_run.py",
        "GroupedMatmul_run.py",
        "GroupedMatmulSwigluQuant_run.py",
        "LightningIndexer_run.py",
        "MoeTokenPermute_run.py",
        "MoeTokenUnpermute_run.py",
        "ScatterNdUpdate_run.py",
        "SparseFlashAttention_run.py",
        "TransposeBatchMatMul_run.py",
        "DispatchFFNCombine_run.py",
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


class FakeTensor:
    def __init__(self, shape=(1,), dtype="float32", device="npu"):
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self.ndim = len(shape)

    def npu(self):
        return self

    def to(self, dtype):
        return FakeTensor(self.shape, dtype=dtype, device=self.device)

    def unsqueeze(self, _dim):
        return FakeTensor((1, *self.shape), dtype=self.dtype, device=self.device)


class FakeTorch:
    int32 = "int32"
    float32 = "float32"

    class Npu:
        @staticmethod
        def synchronize():
            return None

    npu = Npu()

    class Ops:
        class Ascend:
            @staticmethod
            def dispatch_ffn_combine(**kwargs):
                return kwargs["out"], kwargs["expert_token_nums"]

        _C_ascend = Ascend()

    ops = Ops()

    @staticmethod
    def arange(*_args, **_kwargs):
        return FakeTensor((4,), dtype="int32")

    @staticmethod
    def full(shape, _fill_value, dtype=None):
        return FakeTensor(tuple(shape), dtype=dtype)


class TestDispatchFFNCombineReplayHelpers:
    def test_argparser_and_simple_helpers(self, monkeypatch, capsys):
        module = import_op_replay_script("DispatchFFNCombine_run.py")

        parser = module.build_argparser()
        args = parser.parse_args(["--ep-size", "8", "--no-balanced", "--max-output-size", "123"])
        assert args.ep_size == 8
        assert args.balanced is False
        assert args.max_output_size == 123

        monkeypatch.setattr(module, "MAX_OUTPUT_SIZE", None)
        assert module.infer_max_output_size((2, 4), 2) == module.DEFAULT_DFC_MAX_OUTPUT_SIZE
        monkeypatch.setattr(module, "MAX_OUTPUT_SIZE", 256)
        assert module.infer_max_output_size((2, 4), 2) == 256

        monkeypatch.setattr(module, "EP_SIZE", 16)
        assert module.should_skip_row_for_ep_size(Path("DispatchFFNCombine.csv"), 1, {"EP Size": "8"})
        assert not module.should_skip_row_for_ep_size(Path("DispatchFFNCombine.csv"), 1, {"EP Size": "16"})
        assert not module.should_skip_row_for_ep_size(Path("DispatchFFNCombine.csv"), 1, {"EP Size": ""})
        assert "does not match replay" in capsys.readouterr().out

    def test_shape_builders_validate_without_npu(self, monkeypatch):
        module = import_op_replay_script("DispatchFFNCombine_run.py")
        monkeypatch.setattr(module, "get_runtime_modules", lambda: (FakeTorch, object()))
        monkeypatch.setattr(module, "resolve_runtime_dtype", lambda name: name)

        with pytest.raises(ValueError, match="num_experts must be positive"):
            module.build_balanced_expert_idx_tensor((2, 2), 0)
        with pytest.raises(ValueError, match="scale shape mismatch"):
            module.build_scale_tensor((2, 3), (2, 4), "FLOAT")

    def test_debug_and_extension_paths_are_non_fatal(self, monkeypatch, capsys):
        module = import_op_replay_script("DispatchFFNCombine_run.py")
        monkeypatch.setenv("DFC_DEBUG_DEVICES", "1")
        monkeypatch.setattr(module, "_PRINTED_DFC_DEVICE_DEBUG", False)

        case = {
            "x": FakeTensor((2, 4)),
            "weight1_list": [FakeTensor((1, 4, 8))],
            "weight2_list": [FakeTensor((1, 8, 4))],
            "expert_idx": FakeTensor((2, 1), dtype="int32"),
            "scale1_list": [FakeTensor((1, 8))],
            "scale2_list": [FakeTensor((1, 4))],
            "probs": FakeTensor((2, 1)),
            "out": FakeTensor((2, 4)),
            "expert_token_nums": FakeTensor((1,), dtype="int32"),
        }
        module.debug_dfc_tensor_devices(case)
        assert "[DFC debug]" in capsys.readouterr().out

        monkeypatch.setattr(module.importlib.util, "find_spec", lambda _name: None)
        module.ensure_vllm_ascend_extension_loaded()
        assert "DispatchFFNCombine replay may fail" in capsys.readouterr().err

    def test_launch_torchrun_builds_command(self, monkeypatch):
        module = import_op_replay_script("DispatchFFNCombine_run.py")
        calls = []
        monkeypatch.setattr(module, "find_free_port", lambda: 23456)
        monkeypatch.setattr(module.subprocess, "run", lambda command, **kwargs: calls.append((command, kwargs)))

        module.launch_torchrun_and_wait(
            2,
            ["--database-path", "db"],
            nproc_per_node=2,
            nnodes=1,
            node_rank=0,
            master_addr="127.0.0.1",
            master_port=None,
        )

        command, kwargs = calls[0]
        assert "torch.distributed.run" in command
        assert "--master_port=23456" in command
        assert kwargs["env"]["_DFC_AUTO_TORCHRUN"] == "1"

    def test_row_and_operator_paths_can_be_stubbed(self, monkeypatch, tmp_path):
        module = import_op_replay_script("DispatchFFNCombine_run.py")
        monkeypatch.setattr(module, "get_runtime_modules", lambda: (FakeTorch, object()))
        monkeypatch.setattr(module, "ensure_vllm_ascend_extension_loaded", lambda: None)

        case = {
            "x": FakeTensor((2, 4)),
            "weight1_list": [FakeTensor((1, 4, 8))],
            "weight2_list": [FakeTensor((1, 8, 4))],
            "expert_idx": FakeTensor((2, 1), dtype="int32"),
            "scale1_list": [FakeTensor((1, 8))],
            "scale2_list": [FakeTensor((1, 4))],
            "probs": FakeTensor((2, 1)),
            "group": "hccl",
            "max_output_size": 64,
            "out": FakeTensor((2, 4)),
            "expert_token_nums": FakeTensor((1,), dtype="int32"),
            "expected_output_shapes": [(2, 4), (1,)],
            "weight_kind": "BF16",
            "num_experts": 1,
            "global_num_experts": 1,
            "topk": 1,
        }
        out, expert_token_nums, used_fallback = module.execute_dfc_op(case)
        assert (out, expert_token_nums, used_fallback) == (case["out"], case["expert_token_nums"], False)

        monkeypatch.setattr(module, "build_row_case", lambda row, balanced: case)
        module.run_row(tmp_path / "DispatchFFNCombine.csv", 1, {}, balanced=True)

    def test_build_row_case_rejects_bad_metadata(self, monkeypatch):
        module = import_op_replay_script("DispatchFFNCombine_run.py")
        monkeypatch.setattr(module, "init_runtime", lambda: None)

        row = {
            "Input Shapes": "2,4;1,4,8",
            "Input Data Types": "BF16;BF16",
            "Input Formats": "ND;ND",
            "Output Shapes": "2,4;1",
            "Output Data Types": "BF16;INT32",
            "Output Formats": "ND;ND",
        }
        with pytest.raises(ValueError, match="seven input metadata slots"):
            module.build_row_case(row)

    def test_main_reports_missing_csv_before_npu_setup(self, monkeypatch, tmp_path):
        module = import_op_replay_script("DispatchFFNCombine_run.py")
        args = SimpleNamespace(
            repeat_count=1,
            ep_size=1,
            balanced=True,
            max_output_size=None,
            device="ATLAS_800_A3_752T_128G_DIE",
            vllm_version="0.18.0",
            database_path=tmp_path,
            torch_version=None,
            cann_version=None,
            update_mode="all",
            nproc_per_node=None,
            nnodes=1,
            node_rank=0,
            master_addr="127.0.0.1",
            master_port=None,
        )
        monkeypatch.setattr(module, "build_argparser", lambda: SimpleNamespace(parse_args=lambda: args))
        monkeypatch.setattr(module, "get_replay_repeat_count", lambda value: value)
        monkeypatch.setattr(module, "get_target_data_dir", lambda **_kwargs: tmp_path)

        with pytest.raises(FileNotFoundError, match="No DispatchFFNCombine.csv"):
            module.main()


class TestRunAllOpHelpers:
    def test_argparser_and_dispatch_args(self):
        module = import_op_replay_script("run_all_op.py")

        args = module.build_argparser().parse_args(
            [
                "--execution-mode",
                "subprocess",
                "--dispatch-ffn-combine-ep-size",
                "32",
            ]
        )
        assert args.execution_mode == "subprocess"
        assert args.dispatch_ffn_combine_ep_size == 32

        command = ["python", "DispatchFFNCombine_run.py"]
        module.append_dispatch_ffn_combine_args(
            command,
            Path("DispatchFFNCombine_run.py"),
            dispatch_ffn_combine_ep_size=32,
            dispatch_ffn_combine_nproc_per_node=16,
            dispatch_ffn_combine_nnodes=2,
            dispatch_ffn_combine_node_rank=1,
            dispatch_ffn_combine_master_addr="host0",
            dispatch_ffn_combine_master_port=29501,
        )
        assert command[-12:] == [
            "--ep-size",
            "32",
            "--nproc-per-node",
            "16",
            "--nnodes",
            "2",
            "--node-rank",
            "1",
            "--master-addr",
            "host0",
            "--master-port",
            "29501",
        ]

    def test_run_script_modes_build_expected_invocations(self, monkeypatch, tmp_path):
        module = import_op_replay_script("run_all_op.py")
        script_path = tmp_path / "Add_run.py"
        script_path.write_text("print('ok')\n", encoding="utf-8")

        monkeypatch.setattr(module, "SCRIPT_DIR", tmp_path)
        monkeypatch.setattr(module, "build_database_cli_args", lambda **_kwargs: ["--database-path", "db"])
        calls = []
        monkeypatch.setattr(module.subprocess, "run", lambda command, **kwargs: calls.append((command, kwargs)))

        module.run_script_subprocess(
            script_path,
            database_path=Path("db"),
            device="ATLAS_800_A3_752T_128G_DIE",
            vllm_ascend_version=None,
            torch_version=None,
            cann_version=None,
            repeat_count=2,
            update_mode="all",
            dispatch_ffn_combine_ep_size=None,
            dispatch_ffn_combine_nproc_per_node=None,
            dispatch_ffn_combine_nnodes=1,
            dispatch_ffn_combine_node_rank=0,
            dispatch_ffn_combine_master_addr="127.0.0.1",
            dispatch_ffn_combine_master_port=None,
        )
        assert calls[0][0][1] == str(script_path)
        assert "--repeat-count" in calls[0][0]

        runpy_calls = []
        monkeypatch.setattr(module.runpy, "run_path", lambda path, **kwargs: runpy_calls.append((path, kwargs)))
        module.run_script_inprocess(
            script_path,
            database_path=Path("db"),
            device="ATLAS_800_A3_752T_128G_DIE",
            vllm_ascend_version=None,
            torch_version=None,
            cann_version=None,
            repeat_count=None,
            update_mode="missing-only",
            dispatch_ffn_combine_ep_size=None,
            dispatch_ffn_combine_nproc_per_node=None,
            dispatch_ffn_combine_nnodes=1,
            dispatch_ffn_combine_node_rank=0,
            dispatch_ffn_combine_master_addr="127.0.0.1",
            dispatch_ffn_combine_master_port=None,
        )
        assert runpy_calls == [(str(script_path), {"run_name": "__main__"})]

    def test_run_script_dispatches_and_main_summarizes(self, monkeypatch, tmp_path):
        module = import_op_replay_script("run_all_op.py")
        script_path = tmp_path / "Add_run.py"
        script_path.write_text("print('ok')\n", encoding="utf-8")

        mode_calls = []
        monkeypatch.setattr(module, "run_script_subprocess", lambda *args, **kwargs: mode_calls.append("subprocess"))
        monkeypatch.setattr(module, "run_script_inprocess", lambda *args, **kwargs: mode_calls.append("inprocess"))
        module.run_script(
            script_path,
            database_path=Path("db"),
            device="ATLAS_800_A3_752T_128G_DIE",
            vllm_ascend_version=None,
            torch_version=None,
            cann_version=None,
            repeat_count=None,
            update_mode="all",
            dispatch_ffn_combine_ep_size=None,
            dispatch_ffn_combine_nproc_per_node=None,
            dispatch_ffn_combine_nnodes=1,
            dispatch_ffn_combine_node_rank=0,
            dispatch_ffn_combine_master_addr="127.0.0.1",
            dispatch_ffn_combine_master_port=None,
            execution_mode="subprocess",
        )
        assert mode_calls == ["subprocess"]

        args = SimpleNamespace(
            execution_mode="inprocess",
            ops=None,
            device="ATLAS_800_A3_752T_128G_DIE",
            vllm_version=None,
            database_path=tmp_path,
            torch_version=None,
            cann_version=None,
            repeat_count=None,
            update_mode="all",
            dispatch_ffn_combine_ep_size=None,
            dispatch_ffn_combine_nproc_per_node=None,
            dispatch_ffn_combine_nnodes=1,
            dispatch_ffn_combine_node_rank=0,
            dispatch_ffn_combine_master_addr="127.0.0.1",
            dispatch_ffn_combine_master_port=None,
            continue_on_error=False,
        )
        monkeypatch.setattr(module, "build_argparser", lambda: SimpleNamespace(parse_args=lambda: args))
        monkeypatch.setattr(module, "reset_invalid_replay_rows", lambda: None)
        monkeypatch.setattr(module, "discover_run_scripts", lambda: [script_path])
        monkeypatch.setattr(module, "get_target_data_dir", lambda **_kwargs: tmp_path)
        monkeypatch.setattr(module, "has_operator_csv", lambda *_args: True)
        monkeypatch.setattr(module, "run_script", lambda **_kwargs: mode_calls.append("main"))
        monkeypatch.setattr(module, "get_invalid_replay_rows", lambda: [])
        monkeypatch.setattr(module, "print_invalid_replay_summary", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(module, "SCRIPT_DIR", tmp_path)

        module.main()
        assert mode_calls[-1] == "main"
        assert (tmp_path / "run_all_op_status.json").is_file()


class TestCommonModule:
    def test_module_imports_without_npu(self):
        """common.py imports without NPU; torch is lazy-loaded (stays None until init_runtime)."""
        assert common.torch is None
        assert common.torch_npu is None

    def test_data_dir_points_to_profiling_database(self):
        """DATA_DIR resolves to the profiling_database/data/ tree."""
        assert common.DATA_DIR.parts[-2:] == ("profiling_database", "data")

    def test_get_replay_repeat_count_uses_cli_value_or_default(self):
        assert op_common.get_replay_repeat_count(7) == 7
        assert op_common.get_replay_repeat_count(None) == op_common.DEFAULT_REPLAY_REPEAT_COUNT

    @pytest.mark.parametrize("repeat_count", [0, -1])
    def test_get_replay_repeat_count_rejects_non_positive_values(self, repeat_count):
        with pytest.raises(ValueError, match="--repeat-count must be positive"):
            op_common.get_replay_repeat_count(repeat_count)

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
                "--ops",
                "MatMulV2",
                "PadV3",
                "--continue-on-error",
            ]
        )

        assert args.database_path == tmp_path
        assert args.device == "TEST_DEVICE"
        assert args.update_mode == "missing-only"
        assert args.execution_mode == "subprocess"
        assert args.ops == ["MatMulV2", "PadV3"]
        assert args.continue_on_error is True

    def test_discover_run_scripts(self):
        scripts = run_all_op.discover_run_scripts()
        assert len(scripts) > 0
        assert run_all_op.SELF_NAME not in [s.name for s in scripts]

    def test_filter_run_scripts_exact_match(self):
        scripts = [
            Path("MatMulV2_run.py"),
            Path("PadV3_run.py"),
            Path("RmsNorm_run.py"),
        ]
        filtered = run_all_op.filter_run_scripts(scripts, {"MatMulV2"})
        names = [s.name for s in filtered]
        assert names == ["MatMulV2_run.py"]

    def test_filter_run_scripts_none_returns_all(self):
        scripts = [Path("MatMulV2_run.py"), Path("PadV3_run.py")]
        filtered = run_all_op.filter_run_scripts(scripts, None)
        assert len(filtered) == 2

    def test_get_csv_name(self):
        assert run_all_op.get_csv_name(Path("MatMulV2_run.py")) == "MatMulV2.csv"
        assert run_all_op.get_csv_name(Path("PadV3_run.py")) == "PadV3.csv"

    def test_has_operator_csv(self, tmp_path):
        datadir = Path(tmp_path)
        sub = datadir / "sub"
        sub.mkdir(parents=True)
        (sub / "MatMulV2.csv").write_text("x")
        assert run_all_op.has_operator_csv(datadir, "MatMulV2.csv")
        assert not run_all_op.has_operator_csv(datadir, "Nonexistent.csv")


class TestDispatchFfnConstants:
    def test_default_ep_size(self):
        from DispatchFFNCombine_run import DEFAULT_EP_SIZE, DEFAULT_DFC_REPEAT_COUNT

        assert DEFAULT_EP_SIZE == 16
        assert DEFAULT_DFC_REPEAT_COUNT > 0

    def test_default_max_output_size(self):
        from DispatchFFNCombine_run import DEFAULT_DFC_MAX_OUTPUT_SIZE

        assert DEFAULT_DFC_MAX_OUTPUT_SIZE > 0

    def test_build_argparser(self):
        parser = dispatch_ffn.build_standard_argparser(
            description="test",
            usage_examples=["python test.py"],
            version_help="test",
        )
        args = parser.parse_args(["--database-path", "test_dir"])
        assert args.database_path == Path("test_dir")
