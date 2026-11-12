# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Exercise DSV4 replay ABIs on CPU with reduced profiling shapes."""

import importlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools.perf_data_collection.op_replay import replay_framework
from tools.perf_data_collection.op_replay.operator_metadata import (
    supports_case_sharding,
)

KERNELS = (
    "Compressor",
    "HcPost",
    "HcPre",
    "InplacePartialRotaryMul",
    "MoeGatingTopKHash",
    "ScatterNdUpdateV2",
    "SparseAttnSharedkv",
    "VllmQuantLightningIndexer",
)


def row(shapes, dtypes, outputs):
    return {
        "Input Shapes": shapes,
        "Input Data Types": dtypes,
        "Input Formats": ";".join("ND" if slot else "NULL" for slot in shapes.split(";")),
        "Output Shapes": outputs,
    }


@pytest.fixture
def replay_on_cpu(monkeypatch):
    """Replace device allocation and custom calls, retaining real tensor operations."""
    metadata_calls = []

    def metadata_api(**kwargs):
        metadata_calls.append(kwargs)
        return "metadata"

    class CpuRuntime:
        ops = SimpleNamespace(
            _C_ascend=SimpleNamespace(
                npu_sparse_attn_sharedkv_metadata=metadata_api,
                npu_vllm_quant_lightning_indexer_metadata=metadata_api,
            )
        )

        def __getattr__(self, name):
            api = getattr(torch, name)
            if name in {"arange", "tensor", "zeros"}:

                def allocate(*args, **kwargs):
                    kwargs.pop("device", None)
                    return api(*args, **kwargs)

                return allocate
            return api

    runtime = CpuRuntime()
    monkeypatch.setattr(torch.Tensor, "npu", lambda self: self, raising=False)

    def tensor(shape, input_format, dtype_name):
        del input_format
        dtype = {
            "DT_BF16": torch.bfloat16,
            "DT_FLOAT": torch.float32,
            "DT_FLOAT16": torch.float16,
            "DT_INT8": torch.int8,
            "DT_INT32": torch.int32,
            "DT_INT64": torch.int64,
        }[dtype_name]
        return torch.zeros(shape, dtype=dtype)

    monkeypatch.setattr(replay_framework, "init_runtime", lambda: None)
    monkeypatch.setattr(replay_framework, "build_input_tensor", tensor)

    def load(name):
        module = importlib.import_module(f"tools.perf_data_collection.op_replay.{name}_run")
        monkeypatch.setattr(module, "get_runtime_modules", lambda: (runtime, None), raising=False)
        monkeypatch.setattr(module, "build_input_tensor", tensor, raising=False)
        monkeypatch.setattr(module.op, "resolve_api", lambda: lambda *args, **kwargs: (args, kwargs))
        return module

    return load, metadata_calls


@pytest.mark.parametrize("kernel", KERNELS)
def test_dsv4_help_and_sharding(kernel):
    script = Path(replay_framework.__file__).with_name(f"{kernel}_run.py")
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "--database-path" in result.stdout
    assert supports_case_sharding(kernel)


def test_hc_pre_post_physical_abi(replay_on_cpu):
    load, _ = replay_on_cpu
    pre = load("HcPre")
    case = pre.op.build_case(row("2,4,8;24,32;3;24", "BF16;FLOAT;FLOAT;FLOAT", "2,8;2,4;2,4,4"))
    args, kwargs = pre.op.run_case(case)
    assert [tuple(t.shape) for t in args] == [(2, 4, 8), (24, 32), (3,), (24,)]
    assert kwargs["hc_mult"] == 4
    assert kwargs["hc_sinkhorn_iters"] == 20
    with pytest.raises(ValueError, match="hc_fn leading dim"):
        pre.op.build_case(row("2,4,8;23,32;3;24", "BF16;FLOAT;FLOAT;FLOAT", ""))

    post = load("HcPost")
    case = post.op.build_case(row("1,2,8;1,2,4,8;1,2,4;1,2,4,4", "BF16;BF16;FLOAT;FLOAT", "1,2,4,8"))
    args, kwargs = post.op.run_case(case)
    assert len(args) == 4
    assert kwargs == {}


def test_partial_rope_returns_mutated_input(replay_on_cpu):
    load, _ = replay_on_cpu
    module = load("InplacePartialRotaryMul")
    case = module.op.build_case(row("8,1,2,128;8,1,1,64;8,1,1,64", "BF16;FLOAT;FLOAT", "8,1,2,128"))
    assert case["kwargs"] == {"rotary_mode": "interleave", "partial_slice": [64, 128]}
    assert module.op.run_case(case) is case["inputs"][0]


@pytest.mark.parametrize("hashed", [False, True])
def test_hash_routing_preserves_optional_slots(replay_on_cpu, hashed):
    load, _ = replay_on_cpu
    module = load("MoeGatingTopKHash")
    data = (
        row("8,16;;8;32,6", "FLOAT;;INT64;INT32", "8,6;8,6;8,16")
        if hashed
        else row(
            "8,16;16;;",
            "FLOAT;FLOAT;;",
            "8,6;8,6;8,16",
        )
    )
    case = module.op.build_case(data)
    args, kwargs = module.op.run_case(case)
    assert args[1] == 6
    assert (kwargs["bias"] is None) == hashed
    if hashed:
        assert kwargs["input_ids"].tolist() == list(range(8))
        assert kwargs["tid2eid"].min() == 0
        assert kwargs["tid2eid"].max() < 16
    else:
        assert kwargs["input_ids"] is None
        assert kwargs["tid2eid"] is None


def test_scatter_indices_cover_legal_pages(replay_on_cpu):
    load, _ = replay_on_cpu
    module = load("ScatterNdUpdateV2")
    case = module.op.build_case(row("3,4,1,8;9,2;9,8", "BF16;INT32;BF16", "3,4,1,8"))
    assert case["inputs"][1].tolist() == [[i // 4, i % 4] for i in range(9)]
    assert module.op.run_case(case) is case["inputs"][0]
    with pytest.raises(ValueError, match="exceeds cache capacity"):
        module._legal_cache_indices((13, 2), (3, 4, 1, 8))


@pytest.mark.parametrize("ratio,head_dim,coff", [(4, 128, 2), (4, 512, 2), (128, 512, 1)])
def test_compressor_ratios_and_physical_cache(replay_on_cpu, ratio, head_dim, coff):
    load, _ = replay_on_cpu
    module = load("Compressor")
    tokens = 256
    compressed = tokens // ratio + 1
    width = head_dim * coff
    case = module.op.build_case(
        row(
            f"{tokens},16;{width},16;{width},16;4,8,{2 * width};{ratio},{width};{head_dim};"
            f"{compressed},64;{compressed},64;1,4;2;;1",
            "BF16;BF16;BF16;FLOAT;FLOAT;BF16;FLOAT;FLOAT;INT32;INT32;;INT32",
            f"{compressed},{head_dim};4,8,{2 * width}",
        )
    )
    args, kwargs = module.op.run_case(case)
    assert len(args) == 8
    assert kwargs["cmp_ratio"] == ratio
    assert kwargs["coff"] == coff
    assert kwargs["seqused"] is None
    assert kwargs["cu_seqlens"].tolist() == [0, tokens]
    assert 0 < kwargs["state_block_table"].min() <= kwargs["state_block_table"].max() < 4


@pytest.mark.parametrize("ratio", [1, 4, 128])
def test_sparse_attention_regenerates_metadata(replay_on_cpu, ratio):
    load, metadata_calls = replay_on_cpu
    module = load("SparseAttnSharedkv")
    shapes = ["256,8,512", "4,128,1,512", "", "", "", "1,4", "", "2", "2", "", "", "1", "8", "1024"]
    dtypes = ["BF16", "BF16", "", "", "", "INT32", "", "INT32", "INT32", "", "", "INT32", "FLOAT", "INT32"]
    if ratio > 1:
        shapes[2], shapes[6] = "4,128,1,512", "1,2"
        dtypes[2], dtypes[6] = "BF16", "INT32"
    if ratio == 4:
        shapes[4], dtypes[4] = "256,1,512", "INT32"
    case = module.op.build_case(row(";".join(shapes), ";".join(dtypes), "256,8,512;"))
    _, kwargs = module.op.run_case(case)
    assert kwargs["cmp_ratio"] == ratio
    assert kwargs["metadata"] == "metadata"
    assert metadata_calls[0]["cmp_ratio"] == ratio
    assert metadata_calls[0]["has_cmp_kv"] == (ratio > 1)
    assert kwargs["ori_sparse_indices"] is None
    if ratio == 4:
        assert kwargs["cmp_sparse_indices"].max() < 256 // 4 + 1


@pytest.mark.parametrize("kernel", ["Compressor", "SparseAttnSharedkv"])
def test_projected_physical_rows_use_existing_replay(replay_on_cpu, kernel):
    from tensor_cast.performance_model.profiling_database.backend_projector import (
        _V4_CONTRACT,
        _v4_project_workload,
    )

    load, _ = replay_on_cpu
    module = load(kernel)
    if kernel == "Compressor":
        template = row(
            "256,16;256,16;256,16;4,8,512;4,256;128;65,64;65,64;1,40;2;;1",
            "BF16;BF16;BF16;FLOAT;FLOAT;BF16;FLOAT;FLOAT;INT32;INT32;;INT32",
            "65,128;4,8,512",
        )
        workload = {
            "contract": _V4_CONTRACT,
            "tokens": 257,
            "hidden_size": 16,
            "head_dim": 128,
            "rope_head_dim": 64,
            "compress_ratio": 4,
        }
    else:
        template = row(
            "256,8,512;4,128,1,512;4,128,1,512;;256,1,64;1,4;1,2;2;2;;;1;8;1024",
            "BF16;BF16;BF16;;INT32;INT32;INT32;INT32;INT32;;;INT32;FLOAT;INT32",
            "256,8,512;",
        )
        workload = {
            "contract": _V4_CONTRACT,
            "tokens": 257,
            "head_dim": 512,
            "num_heads": 8,
            "compress_ratio": 4,
            "window_size": 128,
            "compress_topk": 64,
        }
    physical = _v4_project_workload(kernel, workload, template)
    projected = row(
        ";".join(",".join(map(str, shape)) for shape in physical.inputs),
        ";".join(physical.input_dtypes),
        ";".join(",".join(map(str, shape)) for shape in physical.outputs),
    )
    case = module.op.build_case(projected)
    args, kwargs = module.op.run_case(case)
    assert args[0].shape[0] == 257
    assert kwargs["cmp_ratio"] == 4
    if kernel == "Compressor":
        assert kwargs["cu_seqlens"].tolist() == [0, 257]
    else:
        assert kwargs["cu_seqlens_q"].tolist() == [0, 257]
        assert kwargs["ori_sparse_indices"] is None


def test_compressor_short_input_uses_explicit_ape_ratio(replay_on_cpu):
    load, _ = replay_on_cpu
    module = load("Compressor")
    data = row(
        "2,16;128,16;128,16;4,8,256;128,128;128;1,64;1,64;1,4;2;;1",
        "BF16;BF16;BF16;FLOAT;FLOAT;BF16;FLOAT;FLOAT;INT32;INT32;;INT32",
        "1,128;4,8,256",
    )
    assert module.op.build_case(data)["kwargs"]["cmp_ratio"] == 128
    data["Output Shapes"] = "2,128;4,8,256"
    with pytest.raises(ValueError, match="APE ratio conflicts"):
        module.op.build_case(data)


@pytest.mark.parametrize("kernel", ["Compressor", "SparseAttnSharedkv"])
@pytest.mark.parametrize("query,kv", [([1, 1], [1024, 2048]), ([16, 32], [16, 32])])
def test_request_runtime_replay_uses_real_length_vectors(replay_on_cpu, monkeypatch, tmp_path, kernel, query, kv):
    from tensor_cast.performance_model.profiling_database.backend_projector import (
        _V4_CONTRACT,
        _V4_RUNTIME_WORKLOAD,
        normalize_v4_workload,
        _v4_project_workload,
    )

    load, metadata_calls = replay_on_cpu
    module = load(kernel)
    template = row(
        "256,16;256,16;256,16;4,8,512;4,256;128;65,64;65,64;1,40;2;;1"
        if kernel == "Compressor"
        else "256,8,512;4,128,1,512;4,128,1,512;;256,1,64;1,4;1,2;2;2;;;1;8;1024",
        "",
        "",
    )
    workload = {
        "contract": _V4_CONTRACT,
        "tokens": sum(query),
        "query_lengths": query,
        "kv_lengths": kv,
        "compress_ratio": 4,
        "head_dim": 128 if kernel == "Compressor" else 512,
    }
    workload.update(
        {"hidden_size": 16, "rope_head_dim": 64}
        if kernel == "Compressor"
        else {"num_heads": 8, "window_size": 128, "compress_topk": min(512, max(kv) // 4)},
    )
    workload = normalize_v4_workload(workload)
    physical = _v4_project_workload(kernel, workload, template)
    data = row(
        ";".join(",".join(map(str, shape)) for shape in physical.inputs),
        ";".join(physical.input_dtypes),
        ";".join(",".join(map(str, shape)) for shape in physical.outputs),
    )
    data[_V4_RUNTIME_WORKLOAD] = json.dumps(workload, sort_keys=True, separators=(",", ":"))
    data.update(
        {"Output Data Types": ";".join(physical.output_dtypes), "Output Formats": ";".join(physical.output_formats)}
    )
    case = module.op.build_case(data)
    args, kwargs = module.op.run_case(case)
    assert args[0].shape[0] == sum(query)
    if kernel == "Compressor":
        assert kwargs["cu_seqlens"].tolist() == [0, query[0], sum(query)]
        assert kwargs["start_pos"].tolist() == [s - q for s, q in zip(kv, query)]
        assert args[6].shape[0] == sum(q // 4 + 1 for q in query)
        assert kwargs["state_block_table"][0, 0] != kwargs["state_block_table"][1, 0]
    else:
        assert kwargs["cu_seqlens_q"].tolist() == [0, query[0], sum(query)]
        assert kwargs["seqused_kv"].tolist() == kv
        assert kwargs["cu_seqlens_cmp_kv"].tolist() == [0, kv[0] // 4, sum(s // 4 for s in kv)]
        assert metadata_calls[-1]["batch_size"] == 2
        assert metadata_calls[-1]["max_seqlen_q"] == max(query)
        assert metadata_calls[-1]["max_seqlen_kv"] == max(kv)
        first_pages = set(kwargs["ori_block_table"][0, : (kv[0] + 127) // 128].tolist())
        second_pages = set(kwargs["ori_block_table"][1, : (kv[1] + 127) // 128].tolist())
        assert first_pages.isdisjoint(second_pages)
    recorded = []
    monkeypatch.setattr(replay_framework, "record_runtime_replay_case", lambda **values: recorded.append(values))
    module.op._record_runtime_case(tmp_path / f"{kernel}.csv", 0, data, warmup_count=2, repeat_count=3)
    assert recorded[0]["aggregation"] == "mean"
    assert recorded[0]["signature_context"][_V4_RUNTIME_WORKLOAD] == data[_V4_RUNTIME_WORKLOAD]


@pytest.mark.parametrize(
    "kernel,ratio",
    [
        ("Compressor", 4),
        ("Compressor", 128),
        ("SparseAttnSharedkv", 0),
        ("SparseAttnSharedkv", 4),
        ("SparseAttnSharedkv", 128),
    ],
)
@pytest.mark.parametrize(
    "query,kv", [([1], [4096]), ([1, 1], [1024, 2048]), ([16, 32], [16, 32]), ([16, 32], [64, 128]), ([1, 1], [1, 1])]
)
def test_runtime_csv_contract_agrees_across_generation_query_and_replay(replay_on_cpu, kernel, ratio, query, kv):
    """Generation, lookup and replay must agree on the recorded runtime contract."""
    from tensor_cast.performance_model.profiling_database.backend_projector import (
        _V4_RUNTIME_WORKLOAD,
        normalize_v4_workload,
        _v4_project_workload,
        _v4_row_workload,
    )

    load, _ = replay_on_cpu
    module = load(kernel)
    template = row(
        "256,16;256,16;256,16;4,8,512;4,256;128;65,64;65,64;1,40;2;;1"
        if kernel == "Compressor"
        else "256,8,512;4,128,1,512;4,128,1,512;;256,1,64;1,4;1,2;2;2;;;1;8;1024",
        "",
        "",
    )
    workload = {
        "contract": "v4_requests_v2",
        "tokens": sum(query),
        "query_lengths": query,
        "kv_lengths": kv,
        "compress_ratio": ratio,
        "head_dim": 128 if kernel == "Compressor" else 512,
    }
    workload.update(
        {"hidden_size": 16, "rope_head_dim": 64}
        if kernel == "Compressor"
        else {
            "num_heads": 8,
            "window_size": 128,
            "compress_topk": min(512, max(1, max(kv) // ratio))
            if ratio == 4
            else max(1, max(kv) // ratio)
            if ratio
            else 0,
        },
    )
    workload = normalize_v4_workload(workload)
    physical = _v4_project_workload(kernel, workload, template)
    data = row(
        ";".join(",".join(map(str, shape)) for shape in physical.inputs),
        ";".join(physical.input_dtypes),
        ";".join(",".join(map(str, shape)) for shape in physical.outputs),
    )
    data[_V4_RUNTIME_WORKLOAD] = json.dumps(workload, sort_keys=True, separators=(",", ":"))
    data.update(
        {"Output Data Types": ";".join(physical.output_dtypes), "Output Formats": ";".join(physical.output_formats)}
    )
    assert _v4_row_workload(kernel, data) == workload
    assert module.infer_runtime_workload(data, *module._slots(data)) == workload

    # A row can retain identical shapes while its request metadata is corrupted.
    # Both query and replay must reject it before consuming the measurement or allocating inputs.
    for key, invalid_value in (
        ("phase", "invalid"),
        ("tokens", sum(query) + 1),
        ("kv_lengths", [0] * len(query)),
        ("contract", "unknown"),
        ("rotate", True),
        ("kv_quant_dtype", "int8"),
    ):
        damaged = dict(data)
        damaged["Runtime v4_workload"] = json.dumps({**workload, key: invalid_value})
        with pytest.raises(ValueError):
            _v4_row_workload(kernel, damaged)
        with pytest.raises(ValueError):
            module.infer_runtime_workload(damaged, *module._slots(damaged))

    for column, replacement in (
        ("Input Data Types", "DT_INT8"),
        ("Input Formats", "FRACTAL_NZ"),
        ("Output Data Types", "DT_INT8"),
        ("Output Formats", "FRACTAL_NZ"),
    ):
        parts = data[column].split(";")
        parts[0] = replacement
        damaged = {**data, column: ";".join(parts)}
        with pytest.raises(ValueError):
            _v4_row_workload(kernel, damaged)
        with pytest.raises(ValueError):
            module.infer_runtime_workload(damaged, *module._slots(damaged))


def test_quant_indexer_uses_recorded_topk_and_new_metadata(replay_on_cpu):
    load, metadata_calls = replay_on_cpu
    module = load("VllmQuantLightningIndexer")
    case = module.op.build_case(
        row(
            "256,64,128;4,128,1,128;256,64;256,64;4,128,1;1;1;1,2;1024",
            "INT8;INT8;FLOAT16;FLOAT16;FLOAT16;INT32;INT32;INT32;INT32",
            "256,1,512;",
        )
    )
    args, kwargs = module.op.run_case(case)
    assert len(args) == 5
    assert kwargs["sparse_count"] == 512
    assert kwargs["cmp_ratio"] == 4
    assert kwargs["metadata"] == "metadata"
    assert kwargs["actual_seq_lengths_query"].tolist() == [256]
    assert kwargs["block_table"].tolist() == [[0, 1]]
    assert metadata_calls[0]["sparse_count"] == 512


@pytest.mark.parametrize(
    "metadata",
    [
        {"Runtime phase": "decode"},
        {"Runtime query_lengths": "[128,128]"},
        {"Runtime kv_lengths": "[6144]"},
        {"Runtime actual_seq_lengths_values": "[1]"},
    ],
)
def test_quant_indexer_refuses_to_replace_request_metadata(replay_on_cpu, metadata):
    load, calls = replay_on_cpu
    module = load("VllmQuantLightningIndexer")
    data = row(
        "256,64,128;4,128,1,128;256,64;256,64;4,128,1;1;1;1,2;1024",
        "INT8;INT8;FLOAT16;FLOAT16;FLOAT16;INT32;INT32;INT32;INT32",
        "256,1,512;",
    )
    data.update(metadata)
    with pytest.raises(ValueError, match="single-request full prefill"):
        module.op.build_case(data)
    assert calls == []


def test_quant_indexer_signatures_keep_request_evidence():
    from tools.perf_data_collection.signature_utils import get_sig

    data = row("256,64,128", "INT8", "256,1,512;")
    before = get_sig(data, op_name="VllmQuantLightningIndexer")
    data["Runtime phase"] = "decode"
    assert get_sig(data, op_name="VllmQuantLightningIndexer") != before


@pytest.mark.parametrize("missing", ["", "NULL"])
def test_sparse_attention_requires_sinks_without_runtime_metadata(replay_on_cpu, missing):
    load, metadata_calls = replay_on_cpu
    module = load("SparseAttnSharedkv")
    shapes = ["256,8,512", "4,128,1,512", "", "", "", "1,4", "", "2", "2", "", "", "1", missing, "1024"]
    dtypes = ["BF16", "BF16", "", "", "", "INT32", "", "INT32", "INT32", "", "", "INT32", missing, "INT32"]
    with pytest.raises(ValueError, match="SparseAttnSharedkv requires the sinks slot for replay"):
        module.op.build_case(row(";".join(shapes), ";".join(dtypes), "256,8,512;"))
    assert not metadata_calls


def _shared_stack_mapping():
    import yaml

    root = Path(__file__).resolve().parents[3]
    database = root / "tensor_cast/performance_model/profiling_database/data/ATLAS_800_A3_752T_128G_DIE/vllm_ascend"
    return yaml.safe_load(
        (database / "vllm0.23.0rc1_torch2.10.0_cann9.0.1/op_mapping.yaml").read_text(encoding="utf-8")
    )


@pytest.mark.parametrize(
    "name,kernel,alternates",
    [
        ("aten.mm.default", "MatMulV2", ["MatMulV3", "MatMulCommon"]),
        (
            "aten.bmm.default",
            "BatchMatMulV2",
            ["batch_matmul_transpose_0", "batch_matmul_transpose", "TransposeBatchMatMul", "BatchMatMulNd"],
        ),
        ("aten.embedding.default", "GatherV2", ["GatherV3", "GatherV2AiCore"]),
        ("aten.index_select.default", "GatherV3", ["GatherV2"]),
        ("aten.add.Tensor", "Add", ["AddAiCore"]),
        ("aten.to.dtype", "Cast", ["TensorMove", "CastAiCore", "TensorMoveAiCore"]),
        ("tensor_cast.grouped_matmul_quant.default", "GroupedMatmul", []),
        ("tensor_cast.grouped_matmul_quant_swiglu.default", "GroupedMatmulSwigluQuantV2", ["GroupedMatmulSwigluQuant"]),
        ("tensor_cast.init_routing_v2.default", "MoeDistributeDispatchV2", ["MoeTokenPermute"]),
        ("tensor_cast.unpermute_tokens.default", "MoeDistributeCombineV2", ["MoeTokenUnpermute"]),
        ("tensor_cast.dynamic_quantize_symmetric.default", "DynamicQuant", ["AscendQuantV2", "AscendQuantV2Aicore"]),
        ("aten.mul.Tensor", "Mul", ["MulAiCore"]),
    ],
)
def test_v4_database_preserves_shared_kernel_mappings(name, kernel, alternates):
    mapping = _shared_stack_mapping()["operator_mappings"][name]
    assert mapping["kernel_type"] == kernel
    assert mapping.get("alternate_kernel_types", []) == alternates
    assert "physical_projection" not in mapping
    assert mapping.get("query_mode") not in {"v4_grouped_moe", "grouped_moe", "moe_distribute"}


def test_v4_packed_mapping_is_separate_from_default():
    config = _shared_stack_mapping()
    assert "repo_digests" not in config
    mappings = config["operator_mappings"]
    assert (
        mappings["tensor_cast.grouped_matmul_quant.packed"]["physical_projection"]["contract"] == "ascend_v4_grouped_v1"
    )
    assert "physical_projection" not in mappings["tensor_cast.grouped_matmul_quant.default"]


@pytest.mark.parametrize(
    "name,kernel,shapes",
    [
        ("aten.bmm.default", "BatchMatMulV2", [(2, 4, 8), (2, 8, 16)]),
        ("tensor_cast.init_routing_v2.default", "MoeDistributeDispatchV2", [(4, 8), (4, 2)]),
        ("tensor_cast.unpermute_tokens.default", "MoeDistributeCombineV2", [(4, 8), (4, 2)]),
    ],
)
def test_v4_database_keeps_ordinary_shared_operator_lookup(tmp_path, name, kernel, shapes):
    import csv
    import yaml

    from tensor_cast.performance_model.op_invoke_info import OpInvokeInfo
    from tensor_cast.performance_model.profiling_database import ProfilingDataSource

    config = _shared_stack_mapping()
    (tmp_path / "op_mapping.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    types = [torch.bfloat16, torch.bfloat16] if name.startswith("aten") else [torch.bfloat16, torch.int64]
    labels = ["DT_BF16", "DT_BF16"] if name.startswith("aten") else ["DT_BF16", "INT64"]
    recorded_count = config["operator_mappings"][name].get("tc_input_count", 2)
    sample = {
        "Input Shapes": ";".join(",".join(map(str, shape)) for shape in shapes[:recorded_count]),
        "Input Data Types": ";".join(labels[:recorded_count]),
        "Input Formats": ";".join(["ND"] * recorded_count),
        "Average Duration(us)": "10",
    }
    with (tmp_path / f"{kernel}.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(sample))
        writer.writeheader()
        writer.writerow(sample)
    namespace, opname, overload = name.split(".")
    func = getattr(getattr(getattr(torch.ops, namespace), opname), overload)
    args = tuple(torch.empty(shape, dtype=dtype, device="meta") for shape, dtype in zip(shapes, types))
    info = OpInvokeInfo(func, args, {}, torch.empty((4, 8), dtype=torch.bfloat16, device="meta"))
    result = ProfilingDataSource(tmp_path).lookup(info)
    assert result is not None
    assert result.latency_us == 10
    assert result.details["kernel_type"] == kernel
