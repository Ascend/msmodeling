"""Non-NPU coverage anchors for perf collection CI test_map.

These tests exercise pure-Python tooling paths so the PR gate can build
symbol-level test_map entries without requiring Ascend hardware.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.perf_data_collection import signature_utils
from tools.perf_data_collection.grid_generator.generators import fused_attention
from tools.perf_data_collection.grid_generator.generators.moe import (
    generate_dispatch_ffn_combine_rows,
)
from tools.perf_data_collection.grid_generator.theory_router import (
    generate_from_template,
)
from tools.perf_data_collection.grid_generator.utils import (
    align_shape_slot_count,
    dedupe_generated_rows,
    process_csv_with_generated_rows,
)
from tools.perf_data_collection.comm_bench.generate_comm_microbench import (
    build_argparser as build_comm_microbench_argparser,
)
from tools.perf_data_collection.op_replay.common import (
    DEFAULT_REPLAY_REPEAT_COUNT,
    get_replay_repeat_count,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
PERF_DATA_COLLECTION_DIR = REPO_ROOT / "tools" / "perf_data_collection"
OP_REPLAY_DIR = PERF_DATA_COLLECTION_DIR / "op_replay"

for path in (PERF_DATA_COLLECTION_DIR, OP_REPLAY_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

dispatch_ffn = importlib.import_module("DispatchFFNCombine_run")
run_all_op = importlib.import_module("run_all_op")
start_microbench = importlib.import_module("start_microbench")


def test_modified_perf_cli_policy_symbols_are_covered():
    parser = build_comm_microbench_argparser()
    args = parser.parse_args(["--database-path", "db", "--bench-mode", "event"])

    assert args.database_path == "db"
    assert args.bench_mode == "event"
    assert not hasattr(args, "run")
    with pytest.raises(SystemExit):
        parser.parse_args(["--do-run"])

    assert get_replay_repeat_count(3) == 3
    assert get_replay_repeat_count(None) == DEFAULT_REPLAY_REPEAT_COUNT
    with pytest.raises(ValueError, match="--repeat-count must be positive"):
        get_replay_repeat_count(0)


def test_signature_utils_canonicalizes_tooling_profile_signatures():
    matmul_row = {
        "OP Type": "MatMulCommon",
        "Input Shapes": "2,3;4,3",
        "Input Data Types": "DT_BF16;DT_BF16",
        "Input Formats": "ND;ND",
        "Output Shapes": "2,4",
        "Output Data Types": "DT_BF16",
    }
    transposed_row = dict(matmul_row, **{"Input Shapes": "2,3;3,4"})

    assert signature_utils.normalize_op_name("MatMulV2_run.py") == "MatMulV2"
    assert signature_utils.is_matmul_family("MatMulV3.csv")
    assert signature_utils.get_sig(matmul_row, op_name="MatMulV2") == signature_utils.get_sig(
        transposed_row,
        op_name="MatMulV3",
    )

    index_row = {
        "OP State": "Index",
        "Input Shapes": "8,16;1,2,3",
        "Input Data Types": "DT_BF16;INT64",
        "Input Formats": "ND;ND",
        "Output Shapes": "4,16",
        "Output Data Types": "DT_BF16",
    }
    assert signature_utils.get_sig(index_row, op_name="Index")[0] == "8,16;4"

    dispatch_row = {
        "OP Type": "DispatchFFNCombine",
        "Input Shapes": "1,4",
        "Input Data Types": "DT_BF16",
        "Input Formats": "ND",
        "Output Shapes": "1,4",
        "Output Data Types": "DT_BF16",
        "EP Size": "32",
    }
    assert signature_utils.get_sig(dispatch_row)[-1] == "32"


def test_shape_grid_generators_cover_fia_and_dfc_rows():
    dense_prefill = fused_attention._build_dense_prefill_row(
        scene_name="dense_prefill",
        batch=2,
        seq=128,
        num_heads=8,
        num_kv_heads=2,
        head_dim=64,
    )
    dense_decode = fused_attention._build_dense_decode_row(
        scene_name="dense_decode",
        batch=2,
        avg_seq_len=256,
        num_heads=8,
        num_kv_heads=2,
        head_dim=64,
    )
    mla_prefill = fused_attention._build_mla_prefill_row(
        scene_name="mla_prefill",
        batch=1,
        seq=128,
        num_heads=8,
        kv_lora_rank=32,
        qk_nope_head_dim=16,
        qk_rope_head_dim=8,
    )
    mla_decode = fused_attention._build_mla_decode_row(
        scene_name="mla_decode",
        batch=1,
        avg_seq_len=256,
        num_heads=8,
        kv_lora_rank=32,
        qk_rope_head_dim=8,
    )

    seen: set[tuple] = set()
    assert fused_attention._should_emit(dense_prefill, seen)
    assert not fused_attention._should_emit(dense_prefill, seen)
    assert "DT_BF16" in fused_attention._fia_input_metadata(mla_prefill.input_shapes)["Input Data Types"]
    assert fused_attention._slot_text(["A"] * 40).count(";") == 30
    assert dense_decode.output_shapes[0][0] == 2
    assert mla_decode.extra_values[fused_attention.RUNTIME_INPUT_LAYOUT] == "BNSD_NBSD"

    generated_rows = list(fused_attention.generate_fused_attention_rows(["zai-org/GLM-5.1"]))
    assert generated_rows

    dfc_row = next(generate_dispatch_ffn_combine_rows(["zai-org/GLM-5.1"]))
    assert dfc_row.extra_values["EP Size"]


def test_theory_template_and_csv_row_processing(tmp_path: Path):
    rows = list(
        generate_from_template(
            {
                "iterators": {"M": [1, 2]},
                "constants": {"K": 4},
                "constraints": ["M <= 2"],
                "inputs": ["M,K"],
                "outputs": ["M,K"],
                "input_dtypes": ["DT_BF16"],
                "input_formats": ["ND"],
                "output_dtypes": ["DT_BF16"],
            },
            model_names=None,
        )
    )
    assert [row.input_shapes for row in rows] == [[(1, 4)], [(2, 4)]]
    assert align_shape_slot_count([(1, 4), ()], [(2, 4)]) == [(2, 4), ()]

    headers = [
        "OP State",
        "Input Shapes",
        "Input Data Types",
        "Input Formats",
        "Output Shapes",
        "Output Data Types",
        "Average Duration(us)",
    ]
    csv_path = tmp_path / "MatMulV2.csv"
    csv_path.write_text(
        ",".join(headers) + "\n" + 'MatMulV2,"1,4;4,4",DT_BF16;DT_BF16,ND;ND,"1,4",DT_BF16,0\n',
        encoding="utf-8",
    )

    generated = {
        "OP State": "MatMulV3",
        "Input Shapes": "1,4;4,4",
        "Input Data Types": "DT_BF16;DT_BF16",
        "Input Formats": "ND;ND",
        "Output Shapes": "1,4",
        "Output Data Types": "DT_BF16",
        "Average Duration(us)": "0",
    }
    assert dedupe_generated_rows(headers, [], [generated], csv_path=csv_path) == [generated]

    appended = process_csv_with_generated_rows(
        csv_path,
        require_rows=True,
        generated_rows_builder=lambda _headers, _source_rows: [
            dict(generated, **{"Input Shapes": "2,4;4,4", "Output Shapes": "2,4"})
        ],
    )
    assert appended == 1
    assert "2,4;4,4" in csv_path.read_text(encoding="utf-8")


def test_non_npu_replay_and_msprof_command_paths(monkeypatch, tmp_path: Path):
    parser = dispatch_ffn.build_argparser()
    args = parser.parse_args(["--ep-size", "32", "--nnodes", "2", "--master-port", "29501"])
    assert args.ep_size == 32
    assert dispatch_ffn.infer_max_output_size((2, 4), topk=8) == dispatch_ffn.DEFAULT_DFC_MAX_OUTPUT_SIZE
    assert not dispatch_ffn.try_load_shared_object(str(tmp_path / "missing.so"))

    monkeypatch.setattr(dispatch_ffn, "EP_SIZE", 32)
    assert dispatch_ffn.should_skip_row_for_ep_size(tmp_path / "DispatchFFNCombine.csv", 1, {"EP Size": "16"})

    command: list[str] = []
    run_all_op.append_dispatch_ffn_combine_args(
        command,
        Path("DispatchFFNCombine_run.py"),
        dispatch_ffn_combine_ep_size=32,
        dispatch_ffn_combine_nproc_per_node=16,
        dispatch_ffn_combine_nnodes=2,
        dispatch_ffn_combine_node_rank=0,
        dispatch_ffn_combine_master_addr="10.0.0.1",
        dispatch_ffn_combine_master_port=29501,
    )
    assert "--node-rank" not in command
    assert "--master-port" in command

    microbench_args = SimpleNamespace(
        fail_fast=False,
        database_path=None,
        device="TEST_DEVICE",
        vllm_version=None,
        torch_version=None,
        cann_version=None,
        repeat_count=1,
        update_mode="missing-only",
        dispatch_ffn_combine_ep_size=32,
        dispatch_ffn_combine_nproc_per_node=16,
        dispatch_ffn_combine_nnodes=2,
        dispatch_ffn_combine_node_rank=0,
        dispatch_ffn_combine_master_addr="10.0.0.1",
        dispatch_ffn_combine_master_port=29501,
    )
    msprof_cmd = start_microbench.build_msprof_cmd(tmp_path, microbench_args, ["DispatchFFNCombine"])
    assert "--dispatch-ffn-combine-node-rank" in msprof_cmd
    assert start_microbench.should_skip_dispatch_ffn_msprof(
        ["DispatchFFNCombine"],
        ep_size=32,
        nproc_per_node=16,
        visible_devices=2,
        update_mode="missing-only",
        has_prof_path=False,
    )

    monkeypatch.setattr(start_microbench, "import_module", lambda _name: SimpleNamespace(npu=None))
    assert start_microbench.get_visible_npu_count() == 0


def test_msprof_execution_paths_are_mockable(monkeypatch, tmp_path: Path):
    prof_dir = tmp_path / "PROF_001"
    prof_dir.mkdir()
    monkeypatch.setattr(
        start_microbench.subprocess,
        "run",
        lambda cmd, check, cwd: SimpleNamespace(returncode=0),
    )
    returncode, prof_dirs = start_microbench.run_msprof_cmd(tmp_path, ["msprof", "python"])
    assert returncode == 0
    assert prof_dir in prof_dirs

    args = SimpleNamespace(
        fail_fast=False,
        database_path=None,
        device="TEST_DEVICE",
        vllm_version=None,
        torch_version=None,
        cann_version=None,
        repeat_count=1,
        update_mode="missing-only",
        dispatch_ffn_combine_ep_size=1,
        dispatch_ffn_combine_nproc_per_node=None,
        dispatch_ffn_combine_nnodes=1,
        dispatch_ffn_combine_node_rank=0,
        dispatch_ffn_combine_master_addr="127.0.0.1",
        dispatch_ffn_combine_master_port=None,
    )
    monkeypatch.setattr(start_microbench, "run_msprof_cmd", lambda root, cmd: (0, {prof_dir}))
    monkeypatch.setattr(
        start_microbench,
        "find_summary_files",
        lambda profs, raise_if_missing=True: [tmp_path / "x.csv"],
    )
    profiler_root, profs = start_microbench.run_msprof(tmp_path, args, ["MatMulV2"])
    assert profiler_root.exists()
    assert profs == {prof_dir}

    calls = []
    monkeypatch.setattr(
        run_all_op.subprocess,
        "run",
        lambda command, check, cwd: calls.append((command, check, cwd)) or SimpleNamespace(returncode=0),
    )
    run_all_op.run_script_subprocess(
        Path("MatMulV2_run.py"),
        database_path=tmp_path,
        device="TEST_DEVICE",
        vllm_ascend_version=None,
        torch_version=None,
        cann_version=None,
        repeat_count=1,
        update_mode="missing-only",
        dispatch_ffn_combine_ep_size=None,
        dispatch_ffn_combine_nproc_per_node=None,
        dispatch_ffn_combine_nnodes=1,
        dispatch_ffn_combine_node_rank=0,
        dispatch_ffn_combine_master_addr="127.0.0.1",
        dispatch_ffn_combine_master_port=None,
    )
    assert calls and "MatMulV2_run.py" in str(calls[0][0])


def test_dfc_tensor_helpers_validate_shapes(monkeypatch):
    class FakeTensor:
        def __init__(self, shape):
            self.shape = tuple(shape)

        def __mod__(self, _other):
            return self

        def __floordiv__(self, _other):
            return self

        def __mul__(self, _other):
            return self

        def __add__(self, _other):
            return self

        def reshape(self, *shape):
            if shape == (-1,):
                size = 1
                for dim in self.shape:
                    size *= dim
                return FakeTensor((size,))
            return FakeTensor(shape)

        def npu(self):
            return self

    class FakeTorch:
        int32 = "int32"
        float32 = "float32"

        @staticmethod
        def arange(size, dtype=None):
            return FakeTensor((size,))

        @staticmethod
        def full(shape, fill_value, dtype=None):
            return FakeTensor(shape)

    monkeypatch.setattr(dispatch_ffn, "get_runtime_modules", lambda: (FakeTorch, None))
    monkeypatch.setattr(dispatch_ffn, "resolve_runtime_dtype", lambda dtype_name: dtype_name)
    monkeypatch.setattr(dispatch_ffn, "EP_SIZE", 2)

    assert dispatch_ffn.build_balanced_expert_idx_tensor((4, 2), 4).shape == (4, 2)
    assert dispatch_ffn.build_scale_tensor((2, 4), (2, 4), "FLOAT").shape == (2, 4)
    assert dispatch_ffn.build_scale_tensor((8,), (2, 4), "FLOAT").shape == (8,)
    with pytest.raises(ValueError, match="flattened scale size mismatch"):
        dispatch_ffn.build_scale_tensor((7,), (2, 4), "FLOAT")
