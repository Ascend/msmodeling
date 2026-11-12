# -------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Licensed under Mulan PSL v2. See LICENSE for details.
# -------------------------------------------------------------------------
"""V4 physical lookup, interpolation and runtime metadata regression tests."""

import csv
import json
import sys
from pathlib import Path

import pytest
import torch
import yaml

from tensor_cast import ops  # noqa: F401
from tensor_cast.performance_model.op_invoke_info import OpInvokeInfo
from tensor_cast.performance_model.profiling_database import (
    InterpolatingDataSource,
    ProfilingDataSource,
)
from tensor_cast.performance_model.profiling_database.backend_projector import (
    _V4_RUNTIME_WORKLOAD,
    _v4_project_workload,
    _v4_row_workload,
    _v4_shapes,
)
from tensor_cast.performance_model.profiling_database.profiling_data_source import (
    _aligned_input_metadata,
)
from tensor_cast.performance_model.profiling_database.query_demand import (
    KernelQueryDemand,
    load_query_demand_traces,
)

ROOT = Path(__file__).resolve().parents[3]
DATABASE: Path | None = None


@pytest.fixture(autouse=True)
def captured_physical_database(tmp_path, monkeypatch):
    """Small snapshots of recorded rows; CI need not download the LFS database."""
    mapping = yaml.safe_load(
        (ROOT / "tests/assets/profiling/v4_paged_w8a8/op_mapping.yaml").read_text(encoding="utf-8")
    )
    target = tmp_path / "captured"
    target.mkdir()
    (target / "op_mapping.yaml").write_text(yaml.safe_dump(mapping), encoding="utf-8")
    samples = {
        "Compressor": [
            (
                "18432,4096;256,4096;256,4096;4500,8,512;4,256;128;4609,64;4609,64;1,2560;2;;1",
                "DT_BF16;DT_BF16;DT_BF16;FLOAT;FLOAT;DT_BF16;FLOAT;FLOAT;INT32;INT32;DT_UNDEFINED;INT32",
                "4609,128;4500,8,512",
                "DT_BF16;FLOAT",
                686.22,
            )
        ],
        "HcPost": [
            (
                "1,2304,4096;1,2304,4,4096;1,2304,4;1,2304,4,4",
                "DT_BF16;DT_BF16;FLOAT;FLOAT",
                "1,2304,4,4096",
                "DT_BF16",
                168.58,
            )
        ],
        "SparseAttnSharedkv": [
            (
                "18432,8,512;4500,128,1,512;4500,128,1,512;;18432,1,512;1,160;1,40;2;2;;;1;8;1024",
                "DT_BF16;DT_BF16;DT_BF16;DT_UNDEFINED;INT32;INT32;INT32;INT32;INT32;DT_UNDEFINED;DT_UNDEFINED;INT32;FLOAT;INT32",
                "18432,8,512;",
                "DT_BF16;FLOAT",
                15655.44,
            ),
            (
                "18432,8,512;4500,128,1,512;4500,128,1,512;;;1,160;1,2;2;2;;;1;8;1024",
                "DT_BF16;DT_BF16;DT_BF16;DT_UNDEFINED;DT_UNDEFINED;INT32;INT32;INT32;INT32;DT_UNDEFINED;DT_UNDEFINED;INT32;FLOAT;INT32",
                "18432,8,512;",
                "DT_BF16;FLOAT",
                4074.74,
            ),
            (
                "18432,8,512;4500,128,1,512;;;;1,160;;2;2;;;1;8;1024",
                "DT_BF16;DT_BF16;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;INT32;DT_UNDEFINED;INT32;INT32;DT_UNDEFINED;DT_UNDEFINED;INT32;FLOAT;INT32",
                "18432,8,512;",
                "DT_BF16;FLOAT",
                1855.08,
            ),
        ],
    }
    headers = [
        "Input Shapes",
        "Input Data Types",
        "Input Formats",
        "Output Shapes",
        "Output Data Types",
        "Output Formats",
        "Average Duration(us)",
        "Profiling Average Duration(us)",
    ]
    for kernel, records in samples.items():
        with (target / f"{kernel}.csv").open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=headers)
            writer.writeheader()
            for inputs, dtypes, outputs, output_dtypes, latency in records:
                writer.writerow(
                    dict(
                        zip(
                            headers,
                            (
                                inputs,
                                dtypes,
                                ";".join("ND" if part else "NULL" for part in inputs.split(";")),
                                outputs,
                                output_dtypes,
                                ";".join("ND" for _ in outputs.split(";")),
                                latency,
                                0,
                            ),
                        )
                    )
                )
    monkeypatch.setattr(sys.modules[__name__], "DATABASE", target)


def tensor(*shape, dtype=torch.bfloat16):
    return torch.empty(shape, dtype=dtype, device="meta")


def rows(kernel):
    with (DATABASE / f"{kernel}.csv").open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def compressor(tokens=18432, head=128, ratio=4, **changes):
    kwargs = {
        "compress_ratio": ratio,
        "head_dim": head,
        "rope_head_dim": 64,
        "rotate": False,
        "cache_layout": "PA_BSND",
        "seq_lens": torch.tensor([tokens]),
        "query_lens": torch.tensor([tokens]),
    }
    kwargs.update(changes)
    return OpInvokeInfo(
        torch.ops.tensor_cast.compressor.default,
        (tensor(1, tokens, 4096), tensor(1, tokens // ratio, head)),
        kwargs,
        (tensor(1, tokens // ratio, head), tensor(1, tokens // ratio, head)),
    )


def sparse(tokens=18432, ratio=4, **changes):
    topk = min(512, tokens // ratio) if ratio == 4 else tokens // ratio if ratio else 0
    kwargs = {
        "softmax_scale": 512**-0.5,
        "head_dim": 512,
        "compress_ratio": ratio,
        "window_size": 128,
        "cache_layout": "PA_BSND",
        "compress_topk": topk,
        "seq_lens": torch.tensor([tokens]),
        "query_lens": torch.tensor([tokens]),
    }
    kwargs.update(changes)
    return OpInvokeInfo(
        torch.ops.tensor_cast.sparse_attn_sharedkv.default,
        (
            tensor(1, tokens, 8, 512),
            tensor(1, tokens + (tokens // ratio if ratio else 0), 512),
            tensor(8, dtype=torch.float32),
            tensor(1, tokens, min(tokens, 128) + topk, dtype=torch.int32),
        ),
        kwargs,
        tensor(1, tokens, 8, 512),
    )


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("hif8", [False, True])
def test_master_kv_arguments_coexist_with_shape_query_metadata(tmp_path, monkeypatch, wrapped, hif8):
    """Keep master's positional KV arguments and the branch's keyword metadata."""
    from tensor_cast.performance_model.op_invoke_info import get_named_args, is_hif8_op_from_info

    trace = tmp_path / "trace"
    monkeypatch.setenv("MSMODELING_SHAPE_QUERY_TRACE_DIR", str(trace))
    base = ProfilingDataSource(DATABASE)
    source = InterpolatingDataSource(base) if wrapped else base
    info = sparse()
    operands = list(info.args)
    scale = tensor(dtype=torch.float32) if hif8 else None
    if hif8:
        operands[1] = torch.empty_like(operands[1], dtype=torch.uint8)
    # These first eleven arguments retain master's positions. Shape metadata
    # follows them as optional keyword arguments rather than shifting kv_scale.
    args = (*operands, info.kwargs["softmax_scale"], 512, None, None, 0, scale, "hif8" if hif8 else None)
    metadata = {key: value for key, value in info.kwargs.items() if key not in {"softmax_scale", "head_dim"}}
    out = info.func(*args, **metadata)
    call = OpInvokeInfo(info.func, args, metadata, out)
    named = get_named_args(call)
    assert named["kv_scale"] is scale
    assert named["kv_format"] == ("hif8" if hif8 else None)
    assert named["compress_ratio"] == 4
    assert named["query_lens"].tolist() == [18432]
    assert is_hif8_op_from_info(call) == hif8

    if hif8:

        def unexpected_projection(*args, **kwargs):
            pytest.fail("HIF8 must not reuse unverified BF16 physical projection")

        monkeypatch.setattr(base, "_lookup_v4_physical", unexpected_projection)
        if wrapped:
            monkeypatch.setattr(source, "_interpolate_v4_physical", unexpected_projection)
        assert source.lookup(call) is None
        assert source.last_miss_reason == "hif8_requires_verified_profile"
        assert load_query_demand_traces(trace) == []
    else:
        result = source.lookup(call)
        assert result is not None
        assert result.latency_us == pytest.approx(15655.44)
        demands = load_query_demand_traces(trace)
        assert len(demands) == 1
        assert demands[0].attributes["query_lengths"] == [18432]
        assert demands[0].attributes["compress_ratio"] == 4


def write_database(directory, kernel, data):
    mapping = yaml.safe_load(
        (ROOT / "tests/assets/profiling/v4_paged_w8a8/op_mapping.yaml").read_text(encoding="utf-8")
    )
    (directory / "op_mapping.yaml").write_text(yaml.safe_dump(mapping), encoding="utf-8")
    with (directory / f"{kernel}.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)
    return InterpolatingDataSource(ProfilingDataSource(directory))


@pytest.mark.parametrize("ratio,latency", [(4, 15655.44), (128, 4074.74), (0, 1855.08)])
def test_sparse_queries_real_physical_rows(ratio, latency):
    source = InterpolatingDataSource(ProfilingDataSource(DATABASE))
    result = source.lookup(sparse(ratio=ratio))
    assert result is not None
    assert result.latency_us == pytest.approx(latency)
    assert result.shape_match_info.shape_match_rule == "v4_physical_projection"
    assert len(result.shape_match_info.kernel_shapes) == 14


def test_compressor_queries_real_physical_row():
    source = ProfilingDataSource(DATABASE)
    result = source.lookup(compressor())
    assert result is not None
    assert result.latency_us == pytest.approx(686.22)
    # A main-attention compressor cannot reuse the index-width measurement.
    assert source.lookup(compressor(head=512)) is None


@pytest.mark.parametrize("flattened", [False, True])
def test_hc_post_accepts_local_tokens_without_matching_full_sequence(tmp_path, flattened):
    mapping_path = ROOT / "tests/assets/profiling/v4_paged_w8a8/op_mapping.yaml"
    (tmp_path / "op_mapping.yaml").write_text(mapping_path.read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "HcPost.csv").write_text((DATABASE / "HcPost.csv").read_text(encoding="utf-8"), encoding="utf-8")
    source = InterpolatingDataSource(ProfilingDataSource(tmp_path))

    def call(tokens):
        return OpInvokeInfo(
            torch.ops.tensor_cast.hc_post.default,
            (
                tensor(tokens, 4096) if flattened else tensor(1, tokens, 4096),
                tensor(1, tokens, 4, 4096),
                tensor(1, tokens, 4, dtype=torch.float32),
                tensor(1, tokens, 4, 4, dtype=torch.float32),
                4,
            ),
            {},
            tensor(1, tokens, 4, 4096),
        )

    assert source.lookup(call(2304)).latency_us == pytest.approx(168.58)
    assert source.lookup(call(18432)) is None


@pytest.mark.parametrize(
    "factory,changes,reason",
    [
        (compressor, {"cache_layout": "BND"}, "v4_paged_runtime_required"),
        (sparse, {"cache_layout": "BND"}, "v4_paged_runtime_required"),
        (compressor, {"query_lens": None}, "v4_sequence_metadata_missing"),
        (compressor, {"seq_lens": torch.tensor([100])}, "v4_request_lengths_invalid"),
        (sparse, {"compress_ratio": -1}, "v4_sparse_parameters_missing_or_unsupported"),
        (sparse, {"window_size": 256}, "v4_sparse_parameters_missing_or_unsupported"),
        (sparse, {"kv_quant_dtype": torch.float8_e4m3fn}, "v4_quantized_kv_unverified"),
        (sparse, {"seq_lens": torch.tensor([9216, 9216])}, "v4_request_lengths_invalid"),
        (sparse, {"softmax_scale": 1.0}, "v4_softmax_scale_unverified"),
    ],
)
def test_incomplete_or_different_scenes_do_not_hit(factory, changes, reason):
    source = InterpolatingDataSource(ProfilingDataSource(DATABASE))
    assert source.lookup(factory(**changes)) is None
    assert source.last_miss_reason == reason


@pytest.mark.parametrize("kernel,factory", [("Compressor", compressor), ("SparseAttnSharedkv", sparse)])
def test_interpolates_only_same_workload_and_inside_sample_range(tmp_path, kernel, factory):
    template = rows(kernel)[0]
    base = ProfilingDataSource(DATABASE)
    data = []
    for tokens, latency in [(4096, 10), (8192, 30)]:
        info = factory(tokens=tokens)
        mapping = base._op_mapping["operator_mappings"][str(info.func)]
        workload = base._v4_workload(info, mapping)
        physical = _v4_project_workload(kernel, workload, template)
        row = dict(template)
        row["Input Shapes"] = ";".join(",".join(map(str, shape)) for shape in physical.inputs)
        row["Output Shapes"] = ";".join(",".join(map(str, shape)) for shape in physical.outputs)
        row["Input Data Types"] = ";".join(physical.input_dtypes)
        row["Input Formats"] = ";".join(physical.input_formats)
        row["Average Duration(us)"] = str(latency)
        data.append(row)
    source = write_database(tmp_path, kernel, data)
    assert source.lookup(factory(tokens=6144)).latency_us == pytest.approx(20)
    assert source.lookup(factory(tokens=16384)) is None
    assert source.lookup(factory(tokens=6144, ratio=128)) is None


def test_hc_post_declares_all_three_linked_inputs(tmp_path):
    template = rows("HcPost")[0]
    data = []
    for tokens, latency in [(100, 10), (200, 30)]:
        row = dict(template)
        row["Input Shapes"] = row["Input Shapes"].replace("2304", str(tokens))
        row["Output Shapes"] = row["Output Shapes"].replace("2304", str(tokens))
        row["Average Duration(us)"] = str(latency)
        data.append(row)
    source = write_database(tmp_path, "HcPost", data)
    info = OpInvokeInfo(
        torch.ops.tensor_cast.hc_post.default,
        (
            tensor(1, 150, 4096),
            tensor(1, 150, 4, 4096),
            tensor(1, 150, 4, dtype=torch.float32),
            tensor(1, 150, 4, 4, dtype=torch.float32),
            4,
        ),
        {},
        tensor(1, 150, 4, 4096),
    )
    assert source.lookup(info).latency_us == pytest.approx(20)


def test_optional_slots_keep_dtype_alignment():
    row = {"Input Shapes": "2,4;;1", "Input Data Types": "DT_BF16;DT_UNDEFINED;INT32", "Input Formats": "ND;NULL;ND"}
    assert _aligned_input_metadata(row) == ([(2, 4), (1,)], ["DT_BF16", "INT32"], ["ND", "ND"])
    assert (
        ProfilingDataSource(DATABASE)._inputs_match([((2, 4), torch.bfloat16), ((1,), torch.int32)], row) == "identity"
    )


def request_op(kernel, query, kv, *, packed=True):
    batch, width = (1, sum(query)) if packed else (len(query), max(query))
    common = {
        "compress_ratio": 4,
        "cache_layout": "PA_BSND",
        "seq_lens": torch.tensor(kv),
        "query_lens": torch.tensor(query),
    }
    if kernel == "Compressor":
        return OpInvokeInfo(
            torch.ops.tensor_cast.compressor.default,
            (tensor(batch, width, 4096), tensor(1, max(kv), 128)),
            {**common, "head_dim": 128, "rope_head_dim": 64, "rotate": False},
            (tensor(batch, max(1, width // 4), 128), tensor(1, max(kv), 128)),
        )
    topk = min(512, max(1, max(kv) // 4))
    window = 128 if all(q == 1 for q in query) else min(width, 128)
    kv_width = width + width // 4 if query == kv else max(kv)
    return OpInvokeInfo(
        torch.ops.tensor_cast.sparse_attn_sharedkv.default,
        (
            tensor(batch, width, 8, 512),
            tensor(batch, kv_width, 512),
            tensor(8, dtype=torch.float32),
            tensor(batch, width, window + topk, dtype=torch.int32),
        ),
        {**common, "softmax_scale": 512**-0.5, "head_dim": 512, "window_size": 128, "compress_topk": topk},
        tensor(batch, width, 8, 512),
    )


def request_row(kernel, info, latency):
    source = ProfilingDataSource(DATABASE)
    mapping = source._op_mapping["operator_mappings"][str(info.func)]
    workload = source._v4_workload(info, mapping)
    template = rows(kernel)[0]
    demand = KernelQueryDemand(
        projector_version="test",
        op_name=str(info.func),
        kernel_type=kernel,
        query_mode="v4_physical",
        attributes=workload,
    )
    physical = _v4_project_workload(kernel, workload, template)
    row = dict(template)
    for name, shapes in (("Input Shapes", physical.inputs), ("Output Shapes", physical.outputs)):
        row[name] = ";".join(",".join(map(str, shape)) for shape in shapes)
    row["Input Data Types"] = ";".join(physical.input_dtypes)
    row["Output Data Types"] = ";".join(physical.output_dtypes)
    row["Input Formats"] = ";".join(physical.input_formats)
    row["Output Formats"] = ";".join(physical.output_formats)
    row[_V4_RUNTIME_WORKLOAD] = json.dumps(workload)
    row["Average Duration(us)"] = str(latency)
    return row, demand


@pytest.mark.parametrize("kernel", ["Compressor", "SparseAttnSharedkv"])
@pytest.mark.parametrize(
    "query,kv,packed",
    [
        ([1], [4096], True),
        ([1, 1], [1024, 2048], True),
        ([1, 1], [1024, 2048], False),
        ([16, 32], [16, 32], True),
        ([16, 32], [16, 32], False),
        ([16, 32], [64, 128], True),
        ([16, 32], [64, 128], False),
    ],
)
def test_request_scenes_round_trip_lookup(tmp_path, kernel, query, kv, packed):
    info = request_op(kernel, query, kv, packed=packed)
    row, demand = request_row(kernel, info, 31)
    assert _v4_row_workload(kernel, row) == demand.attributes
    assert demand.attributes["tokens"] == sum(query)
    source = write_database(tmp_path, kernel, [row])
    assert source.lookup(info).latency_us == 31
    physical = _v4_shapes(row["Input Shapes"])
    assert physical[8 if kernel == "Compressor" else 5][0] == len(query)


@pytest.mark.parametrize("kernel", ["Compressor", "SparseAttnSharedkv"])
def test_same_shapes_with_different_contexts_keep_distinct_measurements(tmp_path, kernel):
    first = request_op(kernel, [1, 1], [4096, 8192])
    second = request_op(kernel, [1, 1], [8192, 4096])
    left, _ = request_row(kernel, first, 10)
    right, _ = request_row(kernel, second, 30)
    assert left["Input Shapes"] == right["Input Shapes"]
    assert left[_V4_RUNTIME_WORKLOAD] != right[_V4_RUNTIME_WORKLOAD]
    assert first.cache_key != second.cache_key
    source = write_database(tmp_path, kernel, [left, right])
    assert source.lookup(first).latency_us == 10
    assert source.lookup(second).latency_us == 30
    assert source.lookup(request_op(kernel, [1, 1], [6000, 6000])) is None


@pytest.mark.parametrize("kernel", ["Compressor", "SparseAttnSharedkv"])
@pytest.mark.parametrize("query", [[1], [1, 1], [16, 32]])
def test_request_interpolation_uses_existing_attention_axes(tmp_path, kernel, query):
    left, _ = request_row(kernel, request_op(kernel, query, [4096] * len(query)), 10)
    right, _ = request_row(kernel, request_op(kernel, query, [8192] * len(query)), 30)
    source = write_database(tmp_path, kernel, [left, right])
    target = request_op(kernel, query, [6144] * len(query))
    result = source.lookup(target)
    assert result.source.name == "INTERPOLATED"
    assert result.latency_us == pytest.approx(20)
    assert source.lookup(request_op(kernel, query, [10240] * len(query))) is None


@pytest.mark.parametrize("kernel", ["Compressor", "SparseAttnSharedkv"])
def test_multi_request_prefill_uses_existing_effective_kv_axis(tmp_path, kernel):
    data = []
    for lengths, latency in [([2048, 2048], 10), ([1024, 3072], 30)]:
        row, _ = request_row(kernel, request_op(kernel, lengths, lengths), latency)
        data.append(row)
    source = write_database(tmp_path, kernel, data)
    lengths = [1536, 2560]
    assert source.lookup(request_op(kernel, lengths, lengths)).latency_us == pytest.approx(15)


@pytest.mark.parametrize("kernel,factory", [("Compressor", compressor), ("SparseAttnSharedkv", sparse)])
def test_prefill_interpolation_does_not_group_by_pool_allocation(tmp_path, kernel, factory):
    template = rows(kernel)[0]
    source = ProfilingDataSource(DATABASE)
    data = []
    for tokens, latency, blocks in [(4096, 10, 4500), (8192, 30, 9000)]:
        info = factory(tokens=tokens)
        mapping = source._op_mapping["operator_mappings"][str(info.func)]
        workload = source._v4_workload(info, mapping)
        physical = _v4_project_workload(kernel, workload, template)
        row = dict(template)
        inputs = list(physical.inputs)
        pool_index = 3 if kernel == "Compressor" else 1
        inputs[pool_index] = (blocks, *inputs[pool_index][1:])
        outputs = list(physical.outputs)
        if kernel == "Compressor":
            outputs[1] = inputs[3]
        else:
            inputs[2] = inputs[1]
        row["Input Shapes"] = ";".join(",".join(map(str, shape)) for shape in inputs)
        row["Output Shapes"] = ";".join(",".join(map(str, shape)) for shape in outputs)
        row["Average Duration(us)"] = str(latency)
        data.append(row)
    source = write_database(tmp_path, kernel, data)
    assert source.lookup(factory(tokens=6144)).latency_us == pytest.approx(20)


def test_v4_interpolation_honors_existing_dimension_override(tmp_path):
    left, _ = request_row("Compressor", request_op("Compressor", [1], [4096]), 10)
    right, _ = request_row("Compressor", request_op("Compressor", [1], [8192]), 30)
    source = write_database(tmp_path, "Compressor", [left, right])
    source._kernel_overrides["Compressor"] = {"max_interpolation_dim": 0}
    assert source.lookup(request_op("Compressor", [1], [6144])) is None


@pytest.mark.parametrize("kernel", ["Compressor", "SparseAttnSharedkv"])
def test_runtime_metadata_is_required_for_nonlegacy_rows(tmp_path, kernel):
    info = request_op(kernel, [1, 1], [1024, 2048])
    row, _ = request_row(kernel, info, 10)
    row[_V4_RUNTIME_WORKLOAD] = ""
    source = write_database(tmp_path, kernel, [row])
    assert source.lookup(info) is None
