"""Smoke tests for tools/perf_data_collection/op_replay/ scripts."""

import importlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from contextlib import contextmanager
from types import SimpleNamespace
import types
from pathlib import Path

import pytest
import torch

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


def test_worker_device_selection_fails_closed(monkeypatch):
    class FakeNpu:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def set_device(_device_id):
            raise RuntimeError("device is unavailable")

    monkeypatch.setattr(
        op_common,
        "get_runtime_modules",
        lambda: (SimpleNamespace(npu=FakeNpu()), None),
    )
    monkeypatch.setenv("MB_DEVICE_ID", "1")

    with pytest.raises(RuntimeError, match="failed to select Ascend NPU device 1"):
        op_common.ensure_npu_available()


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


def test_batch_matmul_3d_rows_use_torch_bmm(monkeypatch):
    module = import_op_replay_script("BatchMatMulV2_run.py")
    bmm_api = object()
    monkeypatch.setattr(
        module,
        "_build_batched_case",
        lambda _row: {"inputs": [object(), object()], "kwargs": {}},
    )
    monkeypatch.setattr(
        op_common,
        "get_runtime_modules",
        lambda: (SimpleNamespace(bmm=bmm_api), None),
    )

    case = module.build_case(
        {
            "Input Shapes": "16,3,192;16,192,512",
            "Output Shapes": "16,3,512",
        }
    )

    assert case["api"] is bmm_api


def test_transpose_batch_matmul_accepts_pr567_lower_endpoint():
    module = import_op_replay_script("TransposeBatchMatMul_run.py")

    module._validate_input_metadata(
        [(1, 8, 64), (1, 64, 64)],
        ["ND", "ND"],
        ["DT_BF16", "DT_BF16"],
    )
    with pytest.raises(ValueError, match="divisible by 64"):
        module._validate_input_metadata(
            [(1, 8, 32), (1, 32, 64)],
            ["ND", "ND"],
            ["DT_BF16", "DT_BF16"],
        )


def test_add_omitted_scalar_keeps_integral_literal():
    module = import_op_replay_script("Add_run.py")
    calls = []
    case = {
        "inputs": ["int32-input"],
        "api": lambda *args: calls.append(args) or "output",
    }

    assert module.run_case(case) == "output"
    assert calls == [("int32-input", 1)]
    assert isinstance(calls[0][1], int)


def test_cast_replay_builds_scalar_input_from_bracket_shape(monkeypatch):
    module = import_op_replay_script("Cast_run.py")
    replay_framework = importlib.import_module("replay_framework")
    built_shapes = []
    scalar_tensor = object()

    monkeypatch.setattr(replay_framework, "init_runtime", lambda: None)
    monkeypatch.setattr(
        replay_framework,
        "build_input_tensor",
        lambda **kwargs: built_shapes.append(kwargs["shape"]) or scalar_tensor,
    )
    monkeypatch.setattr(module, "resolve_runtime_dtype", lambda _dtype: torch.int64)

    case = module.op.build_case(
        {
            "Input Shapes": "[]",
            "Input Data Types": "DT_FLOAT",
            "Input Formats": "ND",
            "Output Data Types": "DT_INT64",
        }
    )

    assert built_shapes == [()]
    assert case["inputs"] == [scalar_tensor]


@pytest.mark.parametrize("script", ["ArgMaxV2_run.py", "Sort_run.py"])
def test_replay_output_is_not_cast_inside_timed_run_case(script):
    module = import_op_replay_script(script)

    class Output:
        def int(self):
            raise AssertionError("run_case must not insert an output Cast")

    output = Output()
    api_result = (output, output) if script == "Sort_run.py" else output
    case = {"api": lambda *args, **kwargs: api_result, "inputs": [object()], "kwargs": {}}

    assert module.op.run_case(case) is api_result


def test_mul_replay_dispatches_scalar_broadcast_and_degenerate_bmm():
    module = import_op_replay_script("Mul_run.py")

    assert module.classify_mul_replay({"Input Shapes": "1,24,64;", "Output Shapes": "1,24,64"}) == "scalar"
    assert (
        module.classify_mul_replay(
            {
                "Input Shapes": "7,8,6144;7,8,1",
                "Output Shapes": "7,8,6144",
            }
        )
        == "elementwise"
    )
    assert module.classify_mul_replay({"Input Shapes": "1,32,1;1,1,27", "Output Shapes": "1,32,27"}) == "bmm"
    assert module.infer_scalar_value({"Output Data Types": "INT64"}) == 1
    assert module.infer_scalar_value({"Output Data Types": "DT_BF16"}) == 1.0


def test_mul_scalar_replay_builds_zero_dim_tensor_for_active_scalar(monkeypatch):
    module = import_op_replay_script("Mul_run.py")
    mul_api = object()
    monkeypatch.setattr(module.op, "build_inputs", lambda _row: ["lhs"])
    monkeypatch.setattr(module, "build_input_tensor", lambda **kwargs: ("scalar", kwargs["shape"]))
    monkeypatch.setattr(
        module,
        "get_runtime_modules",
        lambda: (SimpleNamespace(mul=mul_api, bmm=object()), None),
    )

    case = module.build_case(
        {
            "Input Shapes": "1,24,64;",
            "Input Data Types": "DT_BF16;DT_BF16",
            "Input Formats": "NCL;ND",
            "Output Shapes": "1,24,64",
            "Output Data Types": "DT_BF16",
        }
    )

    assert case["api"] is mul_api
    assert case["inputs"] == ["lhs", ("scalar", ())]


def test_index_replay_accepts_profiler_column_shape(monkeypatch):
    module = import_op_replay_script("Index_run.py")

    class FakeIndex:
        def npu(self):
            return self

    monkeypatch.setattr(module, "build_input_tensor", lambda **_kwargs: object())
    monkeypatch.setattr(
        module,
        "get_runtime_modules",
        lambda: (
            SimpleNamespace(
                int32=object(),
                int64=object(),
                arange=lambda *_args, **_kwargs: FakeIndex(),
            ),
            None,
        ),
    )

    case = module.build_case(
        {
            "Input Shapes": "27,1;1;2;24",
            "Input Data Types": "INT64;INT64;INT64;INT64",
            "Input Formats": "ND;ND;ND;ND",
            "Output Shapes": "24,1",
        }
    )

    assert case["index_axis"] == 0
    assert case["index_len"] == 24


def test_cast_aicore_reuses_cast_contract(monkeypatch):
    module = import_op_replay_script("CastAiCore_run.py")
    cast_module = importlib.import_module("Cast_run")
    monkeypatch.setattr(module.op, "build_inputs", lambda _row: ["input"])
    monkeypatch.setattr(cast_module, "resolve_runtime_dtype", lambda _dtype: torch.int64)
    case = module.build_case({"Output Data Types": "INT64"})
    assert case["inputs"] == ["input"]
    assert case["output_dtype"] is torch.int64


def test_fill_replays_shape_metadata_with_torch_full(monkeypatch):
    module = import_op_replay_script("Fill_run.py")
    calls = []

    class FakeNpu:
        @staticmethod
        def is_available():
            return True

    fake_torch = SimpleNamespace(
        npu=FakeNpu(),
        full=lambda *args, **kwargs: calls.append((args, kwargs)) or "filled",
    )
    monkeypatch.setattr(module, "get_runtime_modules", lambda: (fake_torch, None))
    monkeypatch.setattr(module, "resolve_runtime_dtype", lambda _dtype: "bf16")
    case = module.build_case(
        {
            "Input Shapes": "2;",
            "Input Data Types": "INT64;DT_BF16",
            "Input Formats": "ND;ND",
            "Output Shapes": "3,128",
            "Output Data Types": "DT_BF16",
        }
    )
    assert module.run_case(case) == "filled"
    assert calls == [(((3, 128), 1.0), {"dtype": "bf16", "device": "npu"})]


def test_triton_rope_siso_replays_opaque_profiler_dtype(monkeypatch):
    module = import_op_replay_script("_triton_rope_siso_run.py")
    built = []

    def api(*_args, **_kwargs):
        return "rope-output"

    monkeypatch.setattr(
        module,
        "build_input_tensor",
        lambda shape, tensor_format, dtype_name: (
            built.append((shape, tensor_format, dtype_name)) or f"tensor-{len(built)}"
        ),
    )
    monkeypatch.setattr(module, "resolve_rope_api", lambda: api)
    case = module.build_case(
        {
            "Input Shapes": "3,32,128;3,64;3,64",
            "Input Data Types": "DT_BF16;65535;DT_BF16",
            "Input Formats": "ND;ND;ND",
            "Output Shapes": "3,32,128",
        }
    )
    assert case["rope_dim"] == 128
    assert [dtype for _, _, dtype in built] == ["DT_BF16", "DT_BF16", "DT_BF16"]
    assert module.run_case(case) == "rope-output"


def test_scatter_indices_are_deterministic_and_bounded(monkeypatch):
    module = import_op_replay_script("ScatterNdUpdate_run.py")

    def arange(num_tokens, *, dtype, device):
        assert dtype == torch.int32
        assert device == "npu"
        return torch.arange(num_tokens, dtype=torch.int32)

    monkeypatch.setattr(
        module,
        "get_runtime_modules",
        lambda: (SimpleNamespace(int32=torch.int32, arange=arange), None),
    )

    indices = module.build_indices_tensor((6, 1), (226048, 128))

    assert indices.shape == (6, 1)
    assert indices.flatten().tolist() == list(range(6))


def test_scatter_aicore_reuses_scatter_contract(monkeypatch):
    module = import_op_replay_script("ScatterNdUpdateAiCore_run.py")
    source = importlib.import_module("ScatterNdUpdate_run")
    monkeypatch.setattr(module.op, "build_inputs", lambda _row: ["data", "indices", "updates"])
    monkeypatch.setattr(module.op, "resolve_api", lambda: "scatter-api")
    monkeypatch.setattr(source, "build_indices_tensor", lambda *_args, **_kwargs: "legal-indices")
    case = module.build_case({"Input Shapes": "8,4;3,1;3,4"})
    assert case == {
        "inputs": ["data", "legal-indices", "updates"],
        "kwargs": {},
        "api": "scatter-api",
    }


def test_layer_norm_v3_builds_native_three_output_contract(monkeypatch):
    module = import_op_replay_script("LayerNormV3_run.py")
    tensors = [
        SimpleNamespace(shape=(15, 128)),
        SimpleNamespace(shape=(128,)),
        SimpleNamespace(shape=(128,)),
    ]
    calls = []

    def native_layer_norm(*args):
        calls.append(args)
        return "output", "mean", "rstd"

    monkeypatch.setattr(module.op, "build_inputs", lambda _row: tensors)
    monkeypatch.setattr(module.op, "resolve_api", lambda: native_layer_norm)

    case = module.build_case({})
    result = module.run_case(case)

    assert case["kwargs"] == {"normalized_shape": (128,), "eps": 1e-5}
    assert result == ("output", "mean", "rstd")
    assert calls[0] == (tensors[0], (128,), tensors[1], tensors[2], 1e-5)


def test_add_rms_norm_bias_projects_mixed_rank_query_to_physical_pair(monkeypatch):
    module = import_op_replay_script("AddRmsNormBias_run.py")
    built = []

    monkeypatch.setattr(module, "init_runtime", lambda: None)
    monkeypatch.setattr(
        module,
        "build_input_tensor",
        lambda shape, tensor_format, dtype_name: (
            built.append((shape, tensor_format, dtype_name)) or SimpleNamespace(shape=shape)
        ),
    )

    case = module.build_case(
        {
            "Input Shapes": "1,5,6144;5,6144;6144;",
            "Input Data Types": "DT_BF16;DT_BF16;DT_BF16;DT_UNDEFINED",
            "Input Formats": "NCL;ND;ND;NULL",
            "Output Shapes": "1,5,6144;1,5,1;1,5,6144",
        }
    )

    assert [shape for shape, _, _ in built] == [(1, 5, 6144), (1, 5, 6144), (6144,)]
    assert case["query_x_shapes"] == ((1, 5, 6144), (5, 6144))
    assert case["physical_x_shapes"] == ((1, 5, 6144), (1, 5, 6144))
    assert case["mixed_rank_projection"] is True
    assert case["expected_output_shapes"] == [(1, 5, 6144), (1, 5, 1), (1, 5, 6144)]
    assert case["beta_tensor"] is None
    assert module.op.exact_runtime_match is True


def test_moe_gating_topk_builds_supported_runtime_contract(monkeypatch):
    module = import_op_replay_script("MoeGatingTopK_run.py")
    logits = SimpleNamespace(shape=(1250, 256))
    bias = SimpleNamespace(shape=(256,))
    calls = []

    def gating_api(*args, **kwargs):
        calls.append((args, kwargs))
        return "weights", "indices", "normalized"

    monkeypatch.setattr(module.op, "build_inputs", lambda _row: [logits, bias])
    monkeypatch.setattr(module.op, "resolve_api", lambda: gating_api)
    case = module.build_case({"Output Shapes": "1250,8;1250,8;1250,256"})

    assert module.run_case(case) == ("weights", "indices", "normalized")
    assert calls == [
        (
            (logits,),
            {
                "k": 8,
                "bias": bias,
                "k_group": 1,
                "group_count": 1,
                "group_select_mode": 0,
                "renorm": 0,
                "norm_type": 1,
                "out_flag": True,
                "routed_scaling_factor": 2.5,
                "eps": 1e-20,
            },
        )
    ]


class TestOpReplayScriptsExist:
    EXPECTED_SCRIPTS = [
        "common.py",
        "replay_framework.py",
        "run_all_op.py",
        "MatMulV2_run.py",
        "MatMulV3_run.py",
        "RmsNorm_run.py",
        "LayerNormV3_run.py",
        "SwiGlu_run.py",
        "QuantBatchMatmulV3_run.py",
        "BatchMatMulV2_run.py",
        "GroupedMatmul_run.py",
        "GroupedMatmulSwigluQuant_run.py",
        "LightningIndexer_run.py",
        "MoeTokenPermute_run.py",
        "MoeTokenUnpermute_run.py",
        "MoeGatingTopK_run.py",
        "ScatterNdUpdate_run.py",
        "SparseFlashAttention_run.py",
        "TransposeBatchMatMul_run.py",
        "DispatchFFNCombine_run.py",
        "CastAiCore_run.py",
        "Fill_run.py",
        "_triton_rope_siso_run.py",
        "ScatterNdUpdateAiCore_run.py",
        "SliceAiCore_run.py",
    ]

    @pytest.mark.parametrize("script", EXPECTED_SCRIPTS)
    def test_script_exists(self, script):
        assert (OP_REPLAY_DIR / script).is_file()


class TestOpReplayImportMap:
    NEW_REPLAY_SCRIPTS = [
        "BatchMatMulV2_run.py",
        "GroupedMatmul_run.py",
        "GroupedMatmulSwigluQuant_run.py",
        "LayerNormV3_run.py",
        "LightningIndexer_run.py",
        "MoeTokenPermute_run.py",
        "MoeTokenUnpermute_run.py",
        "MoeGatingTopK_run.py",
        "ScatterNdUpdate_run.py",
        "SparseFlashAttention_run.py",
        "TransposeBatchMatMul_run.py",
        "CastAiCore_run.py",
        "Fill_run.py",
        "_triton_rope_siso_run.py",
        "ScatterNdUpdateAiCore_run.py",
        "SliceAiCore_run.py",
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
        "MoeGatingTopK_run.py",
        "ScatterNdUpdate_run.py",
        "SparseFlashAttention_run.py",
        "TransposeBatchMatMul_run.py",
        "DispatchFFNCombine_run.py",
        "CastAiCore_run.py",
        "Fill_run.py",
        "_triton_rope_siso_run.py",
        "ScatterNdUpdateAiCore_run.py",
        "SliceAiCore_run.py",
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


class TestSparseRuntimeReplayMetadata:
    SFA_SHAPES = [
        (256, 64, 512),
        (1695, 128, 1, 512),
        (1695, 128, 1, 512),
        (256, 1, 2048),
        (1, 1584),
        (1,),
        (1,),
        (256, 64, 64),
        (1695, 128, 1, 64),
    ]
    SFA_DTYPES = [
        "DT_BF16",
        "DT_BF16",
        "DT_BF16",
        "DT_INT32",
        "DT_INT32",
        "DT_INT32",
        "DT_INT32",
        "DT_BF16",
        "DT_BF16",
    ]
    SFA_FORMATS = ["ND"] * 9
    LIGHTNING_SHAPES = [
        (256, 32, 128),
        (1695, 128, 1, 128),
        (256, 32),
        (1,),
        (1,),
        (1, 1584),
    ]
    LIGHTNING_DTYPES = [
        "DT_BF16",
        "DT_BF16",
        "DT_BF16",
        "DT_INT32",
        "DT_INT32",
        "DT_INT32",
    ]
    LIGHTNING_FORMATS = ["ND"] * 6
    RUNTIME_ROW = {
        "Runtime avg_seq_len": "4096",
        "Runtime sparse_mode": "3",
        "Runtime num_key_value_heads": "1",
        "Runtime input_layout": "TND",
        "Runtime topk": "2048",
        "Runtime block_size": "128",
    }

    COMPLETE_RUNTIME_ROW = {
        **RUNTIME_ROW,
        "Runtime case_id": "runtime_case",
        "Runtime actual_seq_lengths_shape": "1",
        "Runtime actual_seq_lengths_values": "256",
        "Runtime actual_seq_lengths_kv_shape": "1",
        "Runtime actual_seq_lengths_kv_values": "4096",
        "Runtime block_table_shape": "1,1584",
        "Runtime block_table_valid_blocks": "32",
        "Runtime num_heads": "64",
        "Runtime cache_layout": "PA_BSND",
        "Runtime kv_cache_mode": "paged",
        "Runtime metadata_completeness": "complete",
        "Runtime sparse_block_size": "1",
        "Runtime sparse_indices_pattern": "uniform",
        "Runtime sparse_indices_valid_count": "2048",
        "Runtime sparse_indices_seed": "7",
    }

    def test_sfa_runtime_metadata_drives_sequence_state(self):
        module = import_op_replay_script("SparseFlashAttention_run.py")

        metadata = module.resolve_case_metadata(
            self.RUNTIME_ROW,
            self.SFA_SHAPES,
            self.SFA_FORMATS,
            self.SFA_DTYPES,
        )

        assert metadata == {
            "avg_seq_len": 4096,
            "block_size": 128,
            "cache_layout": "PA_BSND",
            "input_layout": "TND",
            "kv_cache_mode": "paged",
            "kv_lengths": [4096],
            "kv_heads": 1,
            "query_lengths": [256],
            "sparse_block_size": 1,
            "sparse_indices_pattern": "uniform",
            "sparse_indices_seed": 0,
            "sparse_indices_valid_count": 2048,
            "sparse_mode": 3,
            "topk": 2048,
            "valid_blocks": [32],
        }

    def test_lightning_runtime_metadata_drives_sequence_state(self):
        module = import_op_replay_script("LightningIndexer_run.py")

        metadata = module.resolve_case_metadata(
            self.RUNTIME_ROW,
            self.LIGHTNING_SHAPES,
            self.LIGHTNING_FORMATS,
            self.LIGHTNING_DTYPES,
            [(256, 1, 2048), (256, 1, 2048)],
        )

        assert metadata == {
            "avg_seq_len": 4096,
            "block_size": 128,
            "cache_layout": "PA_BSND",
            "input_layout": "TND",
            "kv_cache_mode": "paged",
            "kv_lengths": [4096],
            "kv_heads": 1,
            "query_lengths": [256],
            "sparse_mode": 3,
            "topk": 2048,
            "valid_blocks": [32],
        }

    def test_complete_sfa_runtime_metadata_is_consumed_exactly(self):
        module = import_op_replay_script("SparseFlashAttention_run.py")

        metadata = module.resolve_case_metadata(
            self.COMPLETE_RUNTIME_ROW,
            self.SFA_SHAPES,
            self.SFA_FORMATS,
            self.SFA_DTYPES,
        )

        assert metadata["query_lengths"] == [256]
        assert metadata["kv_lengths"] == [4096]
        assert metadata["valid_blocks"] == [32]
        assert metadata["sparse_indices_seed"] == 7

    def test_complete_runtime_row_rejects_missing_causal_field(self):
        module = import_op_replay_script("SparseFlashAttention_run.py")
        row = {**self.COMPLETE_RUNTIME_ROW, "Runtime actual_seq_lengths_kv_values": ""}

        with pytest.raises(ValueError, match="missing Runtime actual_seq_lengths_kv_values"):
            module.resolve_case_metadata(
                row,
                self.SFA_SHAPES,
                self.SFA_FORMATS,
                self.SFA_DTYPES,
            )

    @pytest.mark.parametrize(
        ("script", "shapes", "formats", "dtypes", "outputs"),
        [
            (
                "SparseFlashAttention_run.py",
                [
                    (256, 64, 512),
                    (1695, 128, 1, 512),
                    (1695, 128, 1, 512),
                    (256, 1, 2048),
                    (4, 1584),
                    (4,),
                    (4,),
                    (256, 64, 64),
                    (1695, 128, 1, 64),
                ],
                SFA_FORMATS,
                SFA_DTYPES,
                None,
            ),
            (
                "LightningIndexer_run.py",
                [
                    (256, 32, 128),
                    (1695, 128, 1, 128),
                    (256, 32),
                    (4,),
                    (4,),
                    (4, 1584),
                ],
                LIGHTNING_FORMATS,
                LIGHTNING_DTYPES,
                [(256, 1, 2048), (256, 1, 2048)],
            ),
        ],
    )
    def test_rank_zero_runtime_accepts_inactive_packed_requests(self, script, shapes, formats, dtypes, outputs):
        module = import_op_replay_script(script)
        row = {
            **self.COMPLETE_RUNTIME_ROW,
            "Runtime avg_seq_len": "64",
            "Runtime actual_seq_lengths_shape": "4",
            "Runtime actual_seq_lengths_values": "256,256,256,256",
            "Runtime actual_seq_lengths_kv_shape": "4",
            "Runtime actual_seq_lengths_kv_values": "256,0,0,0",
            "Runtime block_table_shape": "4,1584",
            "Runtime block_table_valid_blocks": "2,0,0,0",
            "Runtime num_heads": str(shapes[0][-2]),
            "Runtime sparse_indices_valid_count": "256",
        }

        args = (row, shapes, formats, dtypes)
        if outputs is not None:
            args += (outputs,)
        metadata = module.resolve_case_metadata(*args)

        assert metadata["query_lengths"] == [256, 256, 256, 256]
        assert metadata["kv_lengths"] == [256, 0, 0, 0]
        assert metadata["valid_blocks"] == [2, 0, 0, 0]

    @pytest.mark.parametrize(
        ("script", "shapes", "formats", "dtypes", "outputs"),
        [
            (
                "SparseFlashAttention_run.py",
                [
                    (6, 64, 512),
                    (313, 128, 1, 512),
                    (313, 128, 1, 512),
                    (6, 1, 2048),
                    (2, 157),
                    (2,),
                    (2,),
                    (6, 64, 64),
                    (313, 128, 1, 64),
                ],
                SFA_FORMATS,
                SFA_DTYPES,
                None,
            ),
            (
                "LightningIndexer_run.py",
                [
                    (6, 32, 128),
                    (313, 128, 1, 128),
                    (6, 32),
                    (2,),
                    (2,),
                    (2, 157),
                ],
                LIGHTNING_FORMATS,
                LIGHTNING_DTYPES,
                [(6, 1, 2048), (6, 1, 2048)],
            ),
        ],
    )
    def test_runtime_accepts_reused_kv_pool(self, script, shapes, formats, dtypes, outputs):
        module = import_op_replay_script(script)
        row = {
            **self.COMPLETE_RUNTIME_ROW,
            "Runtime avg_seq_len": "20003",
            "Runtime actual_seq_lengths_shape": "2",
            "Runtime actual_seq_lengths_values": "3,6",
            "Runtime actual_seq_lengths_kv_shape": "2",
            "Runtime actual_seq_lengths_kv_values": "20003,20003",
            "Runtime block_table_shape": "2,157",
            "Runtime block_table_valid_blocks": "157,157",
            "Runtime num_heads": str(shapes[0][-2]),
        }

        args = (row, shapes, formats, dtypes)
        if outputs is not None:
            args += (outputs,)
        metadata = module.resolve_case_metadata(*args)

        assert metadata["valid_blocks"] == [157, 157]

    @pytest.mark.parametrize(
        ("script", "shapes", "formats", "dtypes", "outputs"),
        [
            (
                "SparseFlashAttention_run.py",
                SFA_SHAPES,
                SFA_FORMATS,
                SFA_DTYPES,
                None,
            ),
            (
                "LightningIndexer_run.py",
                LIGHTNING_SHAPES,
                LIGHTNING_FORMATS,
                LIGHTNING_DTYPES,
                [(256, 1, 2048), (256, 1, 2048)],
            ),
        ],
    )
    def test_runtime_shape_conflicts_are_rejected(self, script, shapes, formats, dtypes, outputs):
        module = import_op_replay_script(script)
        row = {**self.RUNTIME_ROW, "Runtime block_size": "64"}

        args = (row, shapes, formats, dtypes)
        if outputs is not None:
            args += (outputs,)
        with pytest.raises(ValueError, match="Runtime block_size=64 conflicts"):
            module.resolve_case_metadata(*args)

    @pytest.mark.parametrize(
        ("script", "shapes", "formats", "dtypes", "outputs"),
        [
            ("SparseFlashAttention_run.py", SFA_SHAPES, SFA_FORMATS, SFA_DTYPES, None),
            (
                "LightningIndexer_run.py",
                LIGHTNING_SHAPES,
                LIGHTNING_FORMATS,
                LIGHTNING_DTYPES,
                [(256, 1, 2048), (256, 1, 2048)],
            ),
        ],
    )
    def test_explicit_zero_runtime_length_is_rejected(self, script, shapes, formats, dtypes, outputs):
        module = import_op_replay_script(script)
        row = {**self.RUNTIME_ROW, "Runtime avg_seq_len": "0"}

        args = (row, shapes, formats, dtypes)
        if outputs is not None:
            args += (outputs,)
        with pytest.raises(ValueError, match="Runtime avg_seq_len=0"):
            module.resolve_case_metadata(*args)


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
        monkeypatch.setattr(
            module.subprocess,
            "run",
            lambda command, **kwargs: calls.append((command, kwargs)),
        )

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
        assert (out, expert_token_nums, used_fallback) == (
            case["out"],
            case["expert_token_nums"],
            False,
        )

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
    def test_case_shards_are_forwarded_only_to_capable_adapters(self):
        module = import_op_replay_script("run_all_op.py")

        for script_name in (
            "Add_run.py",
            "LightningIndexer_run.py",
            "SparseFlashAttention_run.py",
            "mla_preprocess_0_mix_aic_run.py",
        ):
            command: list[str] = []
            module.append_case_shard_args(
                command,
                Path(script_name),
                case_shard_count=4,
                case_shard_index=2,
            )
            assert command == ["--case-shard-count", "4", "--case-shard-index", "2"]

        for script_name in (
            "DispatchFFNCombine_run.py",
            "FusedInferAttentionScore_run.py",
            "QuantBatchMatmulV3_run.py",
            "RINGMLAPrefillBF16Kernel_run.py",
        ):
            command = []
            module.append_case_shard_args(
                command,
                Path(script_name),
                case_shard_count=4,
                case_shard_index=2,
            )
            assert command == []

    def test_argparser_and_dispatch_args(self):
        module = import_op_replay_script("run_all_op.py")

        args = module.build_argparser().parse_args(
            [
                "--execution-mode",
                "subprocess",
                "--status-path",
                "custom-status.json",
                "--dispatch-ffn-combine-ep-size",
                "32",
            ]
        )
        assert args.execution_mode == "subprocess"
        assert args.status_path == Path("custom-status.json")
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

    def test_status_json_records_execution_mode(self, monkeypatch, tmp_path):
        """Status JSON must always carry execution_mode even in subprocess mode."""
        module = import_op_replay_script("run_all_op.py")
        script_path = tmp_path / "Add_run.py"
        script_path.write_text("# test operator entry\n", encoding="utf-8")
        status_path = tmp_path / "status.json"
        monkeypatch.setattr(module, "SCRIPT_DIR", tmp_path)
        monkeypatch.setattr(module, "print_logo", lambda: None)
        monkeypatch.setattr(module, "discover_run_scripts", lambda: [script_path])
        monkeypatch.setattr(module, "get_target_data_dir", lambda **_kwargs: tmp_path)
        monkeypatch.setattr(module, "has_operator_csv", lambda *_args: True)
        monkeypatch.setattr(module, "run_script", lambda **_kwargs: None)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_all_op.py",
                "--execution-mode",
                "subprocess",
                "--status-path",
                str(status_path),
            ],
        )
        module.main()

        assert json.loads(status_path.read_text(encoding="utf-8"))["execution_mode"] == "subprocess"

    def test_parallel_shell_wrapper_strips_optional_separator(self, tmp_path, monkeypatch):
        """The compatibility wrapper must not forward argparse's ``--`` delimiter."""
        if os.name == "nt":
            pytest.skip("the wrapper requires a POSIX shell and path semantics")
        bash = shutil.which("bash")
        if bash is None:
            pytest.skip("bash is not available in this environment")

        capture_path = tmp_path / "argv.txt"
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        python3 = bin_dir / "python3"
        python3.write_text(
            '#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$ARG_CAPTURE"\n',
            encoding="utf-8",
        )
        python3.chmod(0o755)
        monkeypatch.setenv("ARG_CAPTURE", str(capture_path))
        monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")

        wrapper = OP_REPLAY_DIR / "start_microbench_parallel.sh"
        subprocess.run(
            [bash, str(wrapper), "2", "/tmp/db", "--", "--ops", "Add"],
            check=True,
        )

        forwarded = capture_path.read_text(encoding="utf-8").splitlines()
        assert "--" not in forwarded
        assert forwarded[-2:] == ["--ops", "Add"]

    def test_run_script_modes_build_expected_invocations(self, monkeypatch, tmp_path):
        module = import_op_replay_script("run_all_op.py")
        script_path = tmp_path / "Add_run.py"
        script_path.write_text("print('ok')\n", encoding="utf-8")

        monkeypatch.setattr(module, "SCRIPT_DIR", tmp_path)
        monkeypatch.setattr(
            module,
            "build_database_cli_args",
            lambda **_kwargs: ["--database-path", "db"],
        )
        calls = []
        monkeypatch.setattr(
            module.subprocess,
            "run",
            lambda command, **kwargs: calls.append((command, kwargs)),
        )

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
        monkeypatch.setattr(
            module.runpy,
            "run_path",
            lambda path, **kwargs: runpy_calls.append((path, kwargs)),
        )
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
        monkeypatch.setattr(
            module,
            "run_script_subprocess",
            lambda *args, **kwargs: mode_calls.append("subprocess"),
        )
        monkeypatch.setattr(
            module,
            "run_script_inprocess",
            lambda *args, **kwargs: mode_calls.append("inprocess"),
        )
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

    def test_fail_fast_persists_runtime_status_before_reraising(self, monkeypatch, tmp_path):
        module = import_op_replay_script("run_all_op.py")
        script_path = tmp_path / "SparseFlashAttention_run.py"
        script_path.touch()
        status_path = tmp_path / "status.json"
        args = SimpleNamespace(
            execution_mode="inprocess",
            ops=None,
            device="ATLAS_800_A3_752T_128G_DIE",
            vllm_version=None,
            database_path=tmp_path,
            torch_version=None,
            cann_version=None,
            repeat_count=2,
            update_mode="all",
            case_shard_count=1,
            case_shard_index=0,
            dispatch_ffn_combine_ep_size=None,
            dispatch_ffn_combine_nproc_per_node=None,
            dispatch_ffn_combine_nnodes=1,
            dispatch_ffn_combine_node_rank=0,
            dispatch_ffn_combine_master_addr="127.0.0.1",
            dispatch_ffn_combine_master_port=None,
            continue_on_error=False,
            status_path=status_path,
        )
        monkeypatch.setattr(module, "build_argparser", lambda: SimpleNamespace(parse_args=lambda: args))
        monkeypatch.setattr(module, "reset_invalid_replay_rows", lambda: None)
        monkeypatch.setattr(module, "reset_runtime_replay_cases", lambda: None)
        monkeypatch.setattr(module, "discover_run_scripts", lambda: [script_path])
        monkeypatch.setattr(module, "get_target_data_dir", lambda **_kwargs: tmp_path)
        monkeypatch.setattr(module, "has_operator_csv", lambda *_args: True)
        monkeypatch.setattr(
            module,
            "run_script",
            lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        monkeypatch.setattr(
            module,
            "get_runtime_replay_cases",
            lambda: [{"kernel_type": "SparseFlashAttention", "case_id": "case-1"}],
        )
        monkeypatch.setattr(module, "get_invalid_replay_rows", lambda: [{"row_index": 2}])

        with pytest.raises(RuntimeError, match="boom"):
            module.main()

        status = json.loads(status_path.read_text(encoding="utf-8"))
        assert status["failed"] == [{"op": script_path.name, "reason": "boom"}]
        assert status["cases"][0]["case_id"] == "case-1"
        assert status["invalid_rows"] == [{"row_index": 2}]


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
            lambda cmd, env, check: calls.append((cmd, env, check)) or SimpleNamespace(returncode=0),
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
