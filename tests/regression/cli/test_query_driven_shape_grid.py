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

"""Regression tests for demand tracing and backend-shape projection."""

from __future__ import annotations

import json
from itertools import islice
from pathlib import Path

import pytest

from tensor_cast.performance_model.profiling_database.query_demand import (
    KernelQueryDemand,
    QueryDemandTraceWriter,
    load_query_demand_traces,
)
from tools.perf_data_collection.generate_shape_grid import build_argparser
from tools.perf_data_collection.grid_generator.query_coverage import _permuted_product, project_query_demand


def _headers() -> list[str]:
    return [
        "OP State",
        "Input Shapes",
        "Input Data Types",
        "Input Formats",
        "Output Shapes",
        "Output Data Types",
        "Output Formats",
        "Average Duration(us)",
    ]


def test_shape_grid_cli_exposes_stable_public_inputs() -> None:
    parser = build_argparser()
    public_options = {
        option
        for action in parser._actions  # noqa: SLF001 - argparse is the contract under test
        for option in action.option_strings
        if option != "--help"
    }
    assert public_options == {
        "-h",
        "--database-path",
        "--rows",
        "--target-models",
        "--optimizer-args-file",
        "--report-path",
        "--ops",
        "--seed",
    }
    help_text = parser.format_help()
    assert "--shape-profile" not in help_text
    assert "--device" not in help_text
    assert "--max-hbm-gb" not in help_text


def test_query_trace_roundtrip_and_global_deduplication(tmp_path: Path) -> None:
    demand = KernelQueryDemand(
        projector_version="projector/v1",
        op_name="aten.mm.default",
        kernel_type="MatMulV3",
        query_mode="compute",
        input_shapes=((2, 4), (4, 8)),
        output_shapes=((2, 8),),
        input_dtypes=("DT_BF16", "DT_BF16"),
        output_dtypes=("DT_BF16",),
        model_id="org/model",
        workload_id="first",
    )
    writer = QueryDemandTraceWriter(tmp_path)
    writer.record(demand)
    writer.record(demand)
    duplicate_path = tmp_path / "query-demands-duplicate.jsonl"
    duplicate_path.write_text(json.dumps(demand.to_dict()) + "\n", encoding="utf-8")

    loaded = load_query_demand_traces(tmp_path)

    assert loaded == [demand]
    assert loaded[0].signature == demand.signature


def test_query_trace_rejects_unknown_schema(tmp_path: Path) -> None:
    trace = tmp_path / "query-demands-bad.jsonl"
    trace.write_text('{"schema_version": 999}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported kernel-query demand schema"):
        load_query_demand_traces(tmp_path)


def test_coverage_product_is_seeded_unique_and_lazy() -> None:
    small = list(_permuted_product([[1, 2], [10, 20, 30]], seed=7, schema_key="schema"))
    assert len(small) == len(set(small)) == 6
    assert set(small) == {(left, right) for left in (1, 2) for right in (10, 20, 30)}

    # A materialized/sorted product here would allocate 100 million tuples.
    prefix = list(
        islice(
            _permuted_product([range(100), range(100), range(100), range(100)], seed=0, schema_key="large"),
            5,
        )
    )
    assert len(prefix) == 5


def test_matmul_projection_uses_database_weight_orientation() -> None:
    headers = _headers()
    template = {
        "OP State": "dynamic",
        "Input Shapes": "1,4;8,4",
        "Input Data Types": "DT_BF16;DT_BF16",
        "Input Formats": "ND;ND",
        "Output Shapes": "1,8",
        "Output Data Types": "DT_BF16",
        "Output Formats": "ND",
        "Average Duration(us)": "1",
    }
    demand = KernelQueryDemand(
        projector_version="projector/v1",
        op_name="aten.mm.default",
        kernel_type="MatMulV3",
        query_mode="compute",
        input_shapes=((2, 4), (4, 8)),
        output_shapes=((2, 8),),
        input_dtypes=("DT_BF16", "DT_BF16"),
        output_dtypes=("DT_BF16",),
    )

    projected, _ = project_query_demand(demand, headers, [template])

    assert projected.input_shapes == ((2, 4), (8, 4))
    assert projected.output_shapes == ((2, 8),)


def test_quant_matmul_projection_builds_fractal_nz_weight() -> None:
    headers = _headers()
    template = {
        "OP State": "dynamic",
        "Input Shapes": "1,64;4,4,16,32;128;128",
        "Input Data Types": "INT8;INT8;FLOAT;INT32",
        "Input Formats": "ND;FRACTAL_NZ;ND;ND",
        "Output Shapes": "1,128",
        "Output Data Types": "DT_BF16",
        "Output Formats": "ND",
        "Average Duration(us)": "1",
    }
    demand = KernelQueryDemand(
        projector_version="projector/v1",
        op_name="tensor_cast.quant_linear.default",
        kernel_type="QuantBatchMatmulV3",
        query_mode="compute",
        input_shapes=((2, 64), (64, 128), (128,), (128,)),
        output_shapes=((2, 128),),
        input_dtypes=("INT8", "INT8", "FLOAT", "INT32"),
        output_dtypes=("DT_BF16",),
    )

    projected, _ = project_query_demand(demand, headers, [template])

    assert projected.input_shapes[1] == (4, 4, 16, 32)
    assert projected.output_shapes == ((2, 128),)


def test_grouped_matmul_projection_keeps_expert_prefix_and_weight_orientation() -> None:
    headers = _headers()
    template = {
        "OP State": "dynamic",
        "Input Shapes": "1,4;2,8,4;;2;1",
        "Input Data Types": "DT_BF16;INT8;DT_UNDEFINED;INT64;FLOAT",
        "Input Formats": "ND;ND;NULL;ND;ND",
        "Output Shapes": "1,8",
        "Output Data Types": "DT_BF16",
        "Output Formats": "ND",
        "Average Duration(us)": "1",
    }
    demand = KernelQueryDemand(
        projector_version="projector/v1",
        op_name="tensor_cast.grouped_matmul.default",
        kernel_type="GroupedMatmul",
        query_mode="compute",
        input_shapes=((5, 4), (2, 4, 8)),
        output_shapes=((5, 8),),
        input_dtypes=("DT_BF16", "INT8"),
        output_dtypes=("DT_BF16",),
    )

    projected, _ = project_query_demand(demand, headers, [template])

    assert projected.input_shapes[:2] == ((5, 4), (2, 8, 4))
    assert projected.output_shapes == ((5, 8),)


def test_dynamic_quant_projection_preserves_scalar_scale_buffer() -> None:
    headers = _headers()
    template = {
        "OP State": "dynamic",
        "Input Shapes": "1,1",
        "Input Data Types": "DT_BF16",
        "Input Formats": "ND",
        "Output Shapes": "1,1;1",
        "Output Data Types": "INT8;FLOAT",
        "Output Formats": "ND;ND",
        "Average Duration(us)": "1",
    }
    demand = KernelQueryDemand(
        projector_version="projector/v1",
        op_name="tensor_cast.dynamic_quantize_symmetric.default",
        kernel_type="DynamicQuant",
        query_mode="compute",
        input_shapes=((128, 1024),),
        output_shapes=((128, 1024), ()),
        input_dtypes=("DT_BF16",),
        output_dtypes=("INT8", "FLOAT"),
    )

    projected, _ = project_query_demand(demand, headers, [template])

    assert projected.output_shapes == ((128, 1024), (1,))


def test_compute_scale_projection_keeps_fp16_and_builds_replayable_row(monkeypatch) -> None:
    from tools.perf_data_collection.op_replay import replay_framework
    from tools.perf_data_collection.op_replay.DynamicQuant_run import op

    headers = _headers()
    bf16_template = {
        "OP State": "dynamic",
        "Input Shapes": "1,1",
        "Input Data Types": "DT_BF16",
        "Input Formats": "ND",
        "Output Shapes": "1,1;1",
        "Output Data Types": "INT8;FLOAT",
        "Output Formats": "ND;ND",
        "Average Duration(us)": "1",
    }
    fp16_template = {
        **bf16_template,
        "Input Data Types": "DT_FLOAT16",
    }
    demand = KernelQueryDemand(
        projector_version="projector/v1",
        op_name="tensor_cast.dynamic_quantize_symmetric.default",
        kernel_type="DynamicQuant",
        query_mode="compute_scale",
        input_shapes=((1, 127, 1024),),
        output_shapes=((1, 127, 1024), ()),
        input_dtypes=("DT_FLOAT16",),
        output_dtypes=("INT8", "FLOAT"),
        attributes={
            "compute_subcategory": "compute_scale",
            "input_format": "NCL",
            "output_formats": ["NCL", "ND"],
            "scale_mode": "per_tensor",
            "output_count": 2,
        },
    )

    projected, template = project_query_demand(
        demand,
        headers,
        [bf16_template, fp16_template],
    )
    row = projected.to_row(headers, template)

    assert template is fp16_template
    assert projected.input_shapes == ((1, 127, 1024),)
    # The per-token scale is the input without its last dim (1,2,256 -> 1,2),
    # which is what the profiler records for DynamicQuant replay.
    assert projected.output_shapes == ((1, 127, 1024), (1, 127))
    assert projected.input_dtypes == ("DT_FLOAT16",)
    assert projected.input_formats == ("NCL",)
    assert projected.output_formats == ("NCL", "ND")

    monkeypatch.setattr(replay_framework, "init_runtime", lambda: None)
    monkeypatch.setattr(
        replay_framework,
        "build_input_tensor",
        lambda *, shape, input_format, dtype_name: (shape, input_format, dtype_name),
    )
    assert op.build_inputs(row) == [((1, 127, 1024), "NCL", "DT_FLOAT16")]


def test_ascend_quant_projection_moves_auxiliary_values_to_inputs() -> None:
    headers = _headers()
    template = {
        "OP State": "dynamic",
        "Input Shapes": "1,8;8;8",
        "Input Data Types": "DT_BF16;DT_BF16;DT_BF16",
        "Input Formats": "ND;ND;ND",
        "Output Shapes": "1,8",
        "Output Data Types": "INT8",
        "Output Formats": "ND",
        "Average Duration(us)": "1",
    }
    demand = KernelQueryDemand(
        projector_version="projector/v1",
        op_name="tensor_cast.dynamic_quantize_symmetric.default",
        kernel_type="AscendQuantV2",
        query_mode="compute",
        input_shapes=((128, 1024),),
        output_shapes=((128, 1024), ()),
        input_dtypes=("DT_BF16",),
        output_dtypes=("INT8", "FLOAT"),
    )

    projected, _ = project_query_demand(demand, headers, [template])

    assert projected.input_shapes == ((128, 1024), (8,), (8,))
    assert projected.output_shapes == ((128, 1024),)


def test_reshape_and_cache_projection_splits_cache_and_restores_heads() -> None:
    headers = _headers()
    template = {
        "OP State": "dynamic",
        "Input Shapes": "1,1,128;1,1,128;16,128,1,128;16,128,1,128;1",
        "Input Data Types": "DT_BF16;DT_BF16;DT_BF16;DT_BF16;INT32",
        "Input Formats": "ND;ND;ND;ND;ND",
        "Output Shapes": "16,128,1,128;16,128,1,128",
        "Output Data Types": "DT_BF16;DT_BF16",
        "Output Formats": "ND;ND",
        "Average Duration(us)": "1",
    }
    demand = KernelQueryDemand(
        projector_version="projector/v1",
        op_name="tensor_cast.reshape_and_cache.default",
        kernel_type="ReshapeAndCacheNdKernel",
        query_mode="compute",
        input_shapes=((128, 1024), (128, 1024), (2, 2, 128, 8, 128), (128,)),
        input_dtypes=("DT_BF16", "DT_BF16", "DT_BF16", "INT64"),
    )

    projected, _ = project_query_demand(demand, headers, [template])

    assert projected.input_shapes == (
        (128, 8, 128),
        (128, 8, 128),
        (2, 128, 8, 128),
        (2, 128, 8, 128),
        (128,),
    )
    assert projected.output_shapes == ((2, 128, 8, 128), (2, 128, 8, 128))
    assert projected.input_dtypes[-1] == "INT32"


def test_moe_token_permute_projection_flattens_tokens_and_adds_sorted_indices() -> None:
    headers = _headers()
    template = {
        "OP State": "dynamic",
        "Input Shapes": "256,6144;256,8",
        "Input Data Types": "DT_BF16;INT32",
        "Input Formats": "ND;ND",
        "Output Shapes": "2048,6144;2048",
        "Output Data Types": "DT_BF16;INT32",
        "Output Formats": "ND;ND",
        "Average Duration(us)": "1",
    }
    demand = KernelQueryDemand(
        projector_version="projector/v1",
        op_name="tensor_cast.init_routing_v2.default",
        kernel_type="MoeTokenPermute",
        query_mode="compute",
        input_shapes=((1, 128, 6144), (128, 8)),
        output_shapes=((1024, 6144),),
        input_dtypes=("DT_BF16", "INT64"),
        output_dtypes=("DT_BF16",),
    )

    projected, _ = project_query_demand(demand, headers, [template])

    assert projected.input_shapes == ((128, 6144), (128, 8))
    assert projected.input_dtypes == ("DT_BF16", "INT32")
    assert projected.output_shapes == ((1024, 6144), (1024,))


def test_moe_token_unpermute_projection_adds_index_input_and_reduces_topk() -> None:
    headers = _headers()
    template = {
        "OP State": "dynamic",
        "Input Shapes": "2048,6144;2048;",
        "Input Data Types": "DT_BF16;INT32;DT_UNDEFINED",
        "Input Formats": "ND;ND;NULL",
        "Output Shapes": "256,6144",
        "Output Data Types": "DT_BF16",
        "Output Formats": "ND",
        "Average Duration(us)": "1",
    }
    demand = KernelQueryDemand(
        projector_version="projector/v1",
        op_name="tensor_cast.unpermute_tokens.default",
        kernel_type="MoeTokenUnpermute",
        query_mode="compute",
        input_shapes=((1024, 6144),),
        output_shapes=((128, 8, 6144),),
        input_dtypes=("DT_BF16",),
        output_dtypes=("DT_BF16",),
    )

    projected, _ = project_query_demand(demand, headers, [template])

    assert projected.input_shapes[:2] == ((1024, 6144), (1024,))
    assert projected.input_dtypes[:2] == ("DT_BF16", "INT32")
    assert projected.output_shapes == ((128, 6144),)


def test_scatter_cache_projection_keeps_pool_capacity_and_updates_query_axes() -> None:
    headers = _headers()
    template = {
        "OP State": "dynamic",
        "Input Shapes": "214784,128;4096,1;4096,128",
        "Input Data Types": "DT_BF16;INT32;DT_BF16",
        "Input Formats": "ND;ND;ND",
        "Output Shapes": "214784,128",
        "Output Data Types": "DT_BF16",
        "Output Formats": "ND",
        "Average Duration(us)": "1",
    }
    demand = KernelQueryDemand(
        projector_version="projector/v1",
        op_name="tensor_cast.dsa_indexer.default",
        kernel_type="ScatterNdUpdate",
        query_mode="composite_scatter_cache_write",
        input_shapes=((2, 128, 128), (128, 1), (128, 128)),
        input_dtypes=("DT_BF16", "INT32", "DT_BF16"),
    )

    projected, _ = project_query_demand(demand, headers, [template])

    assert projected.input_shapes == ((214784, 128), (128, 1), (128, 128))
    assert projected.output_shapes == ((214784, 128),)


def test_compiled_rms_norm_projection_synthesizes_kernel_stat_output() -> None:
    headers = _headers()
    template = {
        "OP State": "dynamic",
        "Input Shapes": "1,1,6144;6144",
        "Input Data Types": "DT_BF16;DT_BF16",
        "Input Formats": "NCL;ND",
        "Output Shapes": "1,1,6144;1,1,1",
        "Output Data Types": "DT_BF16;FLOAT",
        "Output Formats": "NCL;ND",
        "Average Duration(us)": "1",
    }
    demand = KernelQueryDemand(
        projector_version="projector/v1",
        op_name="tensor_cast.rms_norm.default",
        kernel_type="RmsNorm",
        query_mode="compute",
        input_shapes=((1, 128, 6144), (6144,)),
        output_shapes=((1, 128, 6144),),
        input_dtypes=("DT_BF16", "DT_BF16"),
        output_dtypes=("DT_BF16",),
    )

    projected, _ = project_query_demand(demand, headers, [template])

    assert projected.output_shapes == ((1, 128, 6144), (1, 128, 1))
    assert projected.output_dtypes == ("DT_BF16", "FLOAT")


def test_dispatch_ffn_combine_projection_uses_runtime_expert_count() -> None:
    headers = [*_headers(), "EP Size"]
    template = {
        "OP State": "dynamic",
        "Input Shapes": "1,6144;8,6144,4096;8,2048,6144;1,8;32768;49152;1,8",
        "Input Data Types": "DT_BF16;INT8;INT8;INT32;INT64;INT64;FLOAT",
        "Input Formats": "ND;FRACTAL_NZ;FRACTAL_NZ;ND;ND;ND;ND",
        "Output Shapes": "1,6144;8",
        "Output Data Types": "DT_BF16;INT32",
        "Output Formats": "ND;ND",
        "EP Size": "32",
        "Average Duration(us)": "1",
    }
    demand = KernelQueryDemand(
        projector_version="projector/v1",
        op_name="tensor_cast.dispatch_ffn_combine_quant.default",
        kernel_type="DispatchFFNCombine",
        query_mode="moe_fused",
        input_shapes=(
            (128, 6144),
            (256, 6144, 4096),
            (256, 2048, 6144),
            (128, 8),
            (1048576,),
            (1572864,),
            (128, 8),
        ),
        output_shapes=((128, 8, 6144),),
        input_dtypes=("DT_BF16", "INT8", "INT8", "INT32", "INT64", "INT64", "FLOAT"),
        output_dtypes=("DT_BF16",),
        attributes={"ep_size": 1},
    )

    projected, _ = project_query_demand(demand, headers, [template])

    assert projected.input_shapes[1:3] == ((256, 6144, 4096), (256, 2048, 6144))
    # expert_token_nums must mirror the projected weight's expert count — the
    # replay contract (build_row_case) and existing rows record (num_experts,).
    assert projected.output_shapes == ((128, 6144), (256,))
    assert dict(projected.extra_values)["EP Size"] == "1"


def test_matmul_projection_preserves_float_query_dtype() -> None:
    headers = _headers()
    template = {
        "OP State": "dynamic",
        "Input Shapes": "1,32,1;1,1,8",
        "Input Data Types": "DT_BF16;DT_BF16",
        "Input Formats": "ND;ND",
        "Output Shapes": "1,32,8",
        "Output Data Types": "DT_BF16",
        "Output Formats": "ND",
        "Average Duration(us)": "1",
    }
    demand = KernelQueryDemand(
        projector_version="projector/v1",
        op_name="aten.bmm.default",
        kernel_type="BatchMatMulV2",
        query_mode="compute",
        input_shapes=((1, 32, 1), (1, 1, 128)),
        output_shapes=((1, 32, 128),),
        input_dtypes=("FLOAT", "FLOAT"),
        output_dtypes=("FLOAT",),
    )

    projected, _ = project_query_demand(demand, headers, [template])

    assert projected.input_dtypes == ("FLOAT", "FLOAT")
    assert projected.output_dtypes == ("FLOAT",)


def test_sparse_attention_shape_is_derived_from_runtime_demand() -> None:
    headers = [
        *_headers(),
        "Runtime avg_seq_len",
        "Runtime block_table_shape",
        "Runtime block_table_valid_blocks",
        "Runtime topk",
        "Runtime case_id",
        "Runtime metadata_completeness",
        "Runtime source_profile",
    ]
    template = {
        "OP State": "dynamic",
        "Input Shapes": "1,32,128;16,128,1,128;1,32;1;1;1,1",
        "Input Data Types": "DT_BF16;DT_BF16;DT_BF16;INT32;INT32;INT32",
        "Input Formats": "ND;ND;ND;ND;ND;ND",
        "Output Shapes": "1,1,2048;1,1,2048",
        "Output Data Types": "INT32;DT_BF16",
        "Output Formats": "ND;ND",
        "Runtime metadata_completeness": "complete",
        "Runtime source_profile": "profile_a",
    }
    demand = KernelQueryDemand(
        projector_version="projector/v1",
        op_name="tensor_cast.mla_sparse_attention.default",
        kernel_type="LightningIndexer",
        query_mode="composite_attention",
        attributes={
            "q_shape_3d": [8, 32, 128],
            "avg_seq_len": 4096,
            "batch_size": 4,
            "actual_seq_lengths_values": [2, 4, 6, 8],
            "actual_seq_lengths_kv_values": [4096, 4096, 4096, 4096],
            "block_table_valid_blocks": [32, 32, 32, 32],
            "block_size": 128,
            "topk": 2048,
        },
    )

    projected, _ = project_query_demand(demand, headers, [template])

    assert projected.input_shapes[0] == (8, 32, 128)
    assert projected.input_shapes[1] == (128, 128, 1, 128)
    assert projected.input_shapes[5] == (4, 32)
    assert projected.output_shapes == ((8, 1, 2048), (8, 1, 2048))
    assert dict(projected.extra_values)["Runtime block_table_shape"] == "4,32"
    row = projected.to_row(headers, template)
    assert row["Runtime source_profile"] == ""


def test_validate_candidate_rejects_unaligned_transpose_batch_matmul() -> None:
    from tools.perf_data_collection.grid_generator.query_coverage import (
        ProjectedCandidate,
        _validate_candidate,
    )

    candidate = ProjectedCandidate(
        kernel_type='TransposeBatchMatMul',
        input_shapes=((1, 32, 1), (1, 1, 1)),
        output_shapes=((32, 1, 1),),
        input_dtypes=('DT_BF16', 'DT_BF16'),
        output_dtypes=('DT_BF16',),
        input_formats=('ND', 'ND'),
        output_formats=('ND',),
        extra_values=(),
        schema_key='test',
    )
    assert not _validate_candidate(candidate)


def test_validate_candidate_accepts_aligned_transpose_batch_matmul() -> None:
    from tools.perf_data_collection.grid_generator.query_coverage import (
        ProjectedCandidate,
        _validate_candidate,
    )

    candidate = ProjectedCandidate(
        kernel_type='TransposeBatchMatMul',
        input_shapes=((16, 4, 512), (16, 512, 128)),
        output_shapes=((4, 16, 128),),
        input_dtypes=('DT_BF16', 'DT_BF16'),
        output_dtypes=('DT_BF16',),
        input_formats=('ND', 'ND'),
        output_formats=('ND',),
        extra_values=(),
        schema_key='test',
    )
    assert _validate_candidate(candidate)


def test_validate_candidate_rejects_index_output_exceeding_source() -> None:
    from tools.perf_data_collection.grid_generator.query_coverage import (
        ProjectedCandidate,
        _validate_candidate,
    )

    candidate = ProjectedCandidate(
        kernel_type='Index',
        input_shapes=((1, 1, 6144), (2,), (3,), (1,)),
        output_shapes=((1, 2, 6144),),
        input_dtypes=('DT_BF16', 'INT32', 'INT32', 'INT32'),
        output_dtypes=('DT_BF16',),
        input_formats=('ND', 'ND', 'ND', 'ND'),
        output_formats=('ND',),
        extra_values=(),
        schema_key='test',
    )
    assert not _validate_candidate(candidate)


def test_validate_candidate_rejects_oversized_transpose() -> None:
    from tools.perf_data_collection.grid_generator.query_coverage import (
        ProjectedCandidate,
        _validate_candidate,
    )

    candidate = ProjectedCandidate(
        kernel_type='Transpose',
        input_shapes=((107009, 154880), (2,)),
        output_shapes=((154880, 107009),),
        input_dtypes=('DT_BF16', 'INT64'),
        output_dtypes=('DT_BF16',),
        input_formats=('NCL', 'ND'),
        output_formats=('NCL',),
        extra_values=(),
        schema_key='test',
    )
    assert not _validate_candidate(candidate)


def test_validate_candidate_rejects_matmul_wrong_weight_orientation() -> None:
    from tools.perf_data_collection.grid_generator.query_coverage import (
        ProjectedCandidate,
        _validate_candidate,
    )

    # Weight stored as (K, N) instead of (N, K) -- should be rejected.
    candidate = ProjectedCandidate(
        kernel_type="MatMulV2",
        input_shapes=((1, 6144), (6144, 256)),
        output_shapes=((1, 256),),
        input_dtypes=("DT_BF16", "DT_BF16"),
        output_dtypes=("DT_BF16",),
        input_formats=("ND", "ND"),
        output_formats=("ND",),
        extra_values=(),
        schema_key="test",
    )
    assert not _validate_candidate(candidate)


def test_validate_candidate_accepts_matmul_correct_weight_orientation() -> None:
    from tools.perf_data_collection.grid_generator.query_coverage import (
        ProjectedCandidate,
        _validate_candidate,
    )

    # Weight stored as (N, K) -- correct orientation for replay.
    candidate = ProjectedCandidate(
        kernel_type="MatMulV2",
        input_shapes=((1, 6144), (256, 6144)),
        output_shapes=((1, 256),),
        input_dtypes=("DT_BF16", "DT_BF16"),
        output_dtypes=("DT_BF16",),
        input_formats=("ND", "ND"),
        output_formats=("ND",),
        extra_values=(),
        schema_key="test",
    )
    assert _validate_candidate(candidate)


def test_validate_candidate_accepts_matmul_square_matrix() -> None:
    from tools.perf_data_collection.grid_generator.query_coverage import (
        ProjectedCandidate,
        _validate_candidate,
    )

    # Square matrix (K == N) -- orientation is ambiguous but valid.
    candidate = ProjectedCandidate(
        kernel_type="MatMulV3",
        input_shapes=((1, 6144), (6144, 6144)),
        output_shapes=((1, 6144),),
        input_dtypes=("DT_BF16", "DT_BF16"),
        output_dtypes=("DT_BF16",),
        input_formats=("ND", "ND"),
        output_formats=("ND",),
        extra_values=(),
        schema_key="test",
    )
    assert _validate_candidate(candidate)


def _candidate(
    kernel_type: str,
    input_shapes: list[tuple[int, ...]],
    output_shapes: list[tuple[int, ...]],
    *,
    input_dtypes: tuple[str, ...] = ("DT_BF16", "DT_BF16", "DT_BF16"),
    output_dtypes: tuple[str, ...] = ("DT_BF16",),
) -> "object":
    from tools.perf_data_collection.grid_generator.query_coverage import ProjectedCandidate

    return ProjectedCandidate(
        kernel_type=kernel_type,
        input_shapes=tuple(tuple(x) for x in input_shapes),
        output_shapes=tuple(tuple(x) for x in output_shapes),
        input_dtypes=input_dtypes[: len(input_shapes)],
        output_dtypes=output_dtypes,
        input_formats=("ND",) * len(input_shapes),
        output_formats=("ND",) * len(output_shapes),
        extra_values=(),
        schema_key="test",
        exact=False,
    )


def test_validate_candidate_rejects_non_broadcastable_add() -> None:
    from tools.perf_data_collection.grid_generator.query_coverage import _validate_candidate

    # Coverage interpolation mixed independent axis domains across operands.
    assert not _validate_candidate(_candidate("Add", [(1, 1, 256), (1, 32, 6144)], [(1, 1, 256)]))
    assert not _validate_candidate(_candidate("Add", [(1, 2, 6144), (1, 11, 6144)], [(1, 2, 6144)]))
    # A broadcastable pair with a consistent output remains valid.
    assert _validate_candidate(_candidate("Add", [(1, 256), (256,)], [(1, 256)]))
    # Output must equal the broadcast result.
    assert not _validate_candidate(_candidate("Add", [(1, 1, 6144), (1, 1, 6144)], [(1, 2, 6144)]))


def test_validate_candidate_requires_ascend_quant_scale_matches_last_dim() -> None:
    from tools.perf_data_collection.grid_generator.query_coverage import _validate_candidate

    # scale/offset must equal the input last dim (or be scalar), and the
    # quantized output must keep the input shape.
    assert _validate_candidate(_candidate("AscendQuantV2", [(1, 2048), (2048,), (2048,)], [(1, 2048)]))
    assert _validate_candidate(_candidate("AscendQuantV2", [(1, 2048), (1,), (1,)], [(1, 2048)]))
    assert not _validate_candidate(_candidate("AscendQuantV2", [(1, 1, 6144), (384,), (384,)], [(1, 1, 6144)]))
    assert not _validate_candidate(_candidate("AscendQuantV2", [(1, 2048), (1,), (1,)], [(1, 7168)]))


def test_validate_candidate_requires_ascend_quant_scale_dtype_matches_input() -> None:
    from tools.perf_data_collection.grid_generator.query_coverage import _validate_candidate

    # aclnnAscendQuantV3 rejects x vs scale/offset dtype mismatches (161002).
    assert _validate_candidate(
        _candidate(
            "AscendQuantV2",
            [(4, 7168), (7168,), (7168,)],
            [(4, 7168)],
            input_dtypes=("DT_BF16", "DT_BF16", "DT_BF16"),
        )
    )
    assert _validate_candidate(
        _candidate(
            "AscendQuantV2",
            [(4, 7168), (7168,), (7168,)],
            [(4, 7168)],
            input_dtypes=("DT_FLOAT16", "DT_FLOAT16", "DT_FLOAT16"),
        )
    )
    assert not _validate_candidate(
        _candidate(
            "AscendQuantV2",
            [(4, 7168), (7168,), (7168,)],
            [(4, 7168)],
            input_dtypes=("DT_FLOAT16", "DT_BF16", "DT_BF16"),
        )
    )


def test_validate_candidate_requires_dynamic_quant_scale_output() -> None:
    from tools.perf_data_collection.grid_generator.query_coverage import _validate_candidate

    # DynamicQuant's per-token scale is a real second output equal to the
    # input without its last dim; empty or wrong scale slots never match the
    # profiler signature.
    assert _validate_candidate(_candidate("DynamicQuant", [(1, 2, 256)], [(1, 2, 256), (1, 2)]))
    assert _validate_candidate(_candidate("DynamicQuant", [(1, 2048)], [(1, 2048), (1,)]))
    assert not _validate_candidate(_candidate("DynamicQuant", [(1, 2, 256)], [(1, 2, 256)]))
    assert not _validate_candidate(_candidate("DynamicQuant", [(1, 2, 256)], [(1, 2, 256), ()]))
    assert not _validate_candidate(_candidate("DynamicQuant", [(1, 2, 256)], [(1, 2, 256), (1,)]))


def test_validate_candidate_requires_transpose_batch_matmul_output_layout() -> None:
    from tools.perf_data_collection.grid_generator.query_coverage import _validate_candidate

    assert _validate_candidate(_candidate("TransposeBatchMatMul", [(16, 1, 512), (16, 512, 128)], [(1, 16, 128)]))
    # The recorded output must equal the fused (M, B, N) layout.
    assert not _validate_candidate(_candidate("TransposeBatchMatMul", [(1, 512, 512), (1, 512, 256)], [(1, 16, 128)]))
    assert _validate_candidate(_candidate("TransposeBatchMatMul", [(1, 512, 512), (1, 512, 256)], [(512, 1, 256)]))


def test_validate_candidate_rejects_degenerate_batch_matmul_contraction() -> None:
    from tools.perf_data_collection.grid_generator.query_coverage import _validate_candidate

    # K=1 batched rows replay OK but never dispatch a measurable GEMM kernel.
    assert not _validate_candidate(_candidate("BatchMatMulV2", [(1, 32, 1), (1, 4096, 1)], [(1, 32, 4096)]))
    assert not _validate_candidate(_candidate("BatchMatMulV2", [(1, 32, 1), (1, 1, 4096)], [(1, 32, 4096)]))
    assert _validate_candidate(_candidate("BatchMatMulV2", [(1, 32, 128), (1, 128, 64)], [(1, 32, 64)]))
    # (B, N, K) second input is also accepted by the adapter (transpose_b=True).
    assert _validate_candidate(_candidate("BatchMatMulV2", [(1, 32, 128), (1, 64, 128)], [(1, 32, 64)]))
    # N may be smaller than the dispatch threshold; only the contraction K is
    # constrained, for both supported RHS layouts.
    assert _validate_candidate(_candidate("BatchMatMulV2", [(1, 32, 128), (1, 128, 1)], [(1, 32, 1)]))
    assert _validate_candidate(_candidate("BatchMatMulV2", [(1, 32, 128), (1, 1, 128)], [(1, 32, 1)]))
    assert not _validate_candidate(_candidate("BatchMatMulV2", [(1, 32, 128), (1, 256, 64)], [(1, 32, 64)]))


# ── Budget semantics: exact demand vs the --rows budget ─────────────────────


def _budget_csv(tmp_path: Path) -> Path:
    csv_path = tmp_path / "Add.csv"
    header = ",".join(_headers())
    csv_path.write_text(f'{header}\nenabled,"1,4;1,4",BF16;BF16,ND;ND,"1,4",BF16,ND,0\n', encoding="utf-8")
    return csv_path


def _elementwise_demand(shape: tuple[int, int], workload_id: str = "target") -> KernelQueryDemand:
    return KernelQueryDemand(
        projector_version="test/v1",
        op_name="tensor_cast.add.default",
        kernel_type="Add",
        query_mode="elementwise",
        input_shapes=(shape, shape),
        output_shapes=(shape,),
        input_dtypes=("BF16", "BF16"),
        output_dtypes=("BF16",),
        tensor_parallel_size=1,
        expert_parallel_size=1,
        workload_id=workload_id,
    )


def _budget_rows(tmp_path: Path, row_limit: int, exact_unlimited: bool):
    from tools.perf_data_collection.grid_generator.query_coverage import build_query_generated_rows
    from tools.perf_data_collection.grid_generator.utils import load_csv_template_rows

    csv_path = _budget_csv(tmp_path)
    headers, source_rows = load_csv_template_rows(csv_path, require_rows=True)
    workload_id = "spec" if exact_unlimited else "target"
    demands = [
        _elementwise_demand((2, 4), workload_id),
        _elementwise_demand((3, 4), workload_id),
        _elementwise_demand((4, 4), workload_id),
    ]
    return build_query_generated_rows(
        csv_path=csv_path,
        headers=headers,
        source_rows=source_rows,
        demands=demands,
        row_limit=row_limit,
        seed=0,
        exact_unlimited_workload_ids={"spec"} if exact_unlimited else set(),
    )


def test_legacy_budget_still_caps_exact_rows(tmp_path: Path) -> None:
    rows, summary = _budget_rows(tmp_path, row_limit=2, exact_unlimited=False)

    assert len(rows) == 2
    assert summary["exact_generated"] == 2
    assert summary["exact_budget_truncated"] == 1
    assert summary["coverage_appended"] == 0
    statuses = sorted(summary["demand_statuses"])
    assert statuses.count("generated") == 2
    assert statuses.count("budget_truncated") == 1


def test_exact_unlimited_mode_never_truncates_exact_demand(tmp_path: Path) -> None:
    rows, summary = _budget_rows(tmp_path, row_limit=1, exact_unlimited=True)

    # All three exact demand rows are written; the budget bounds exactly one
    # interpolated coverage row.
    assert summary["exact_generated"] == 3
    assert summary["exact_budget_truncated"] == 0
    assert summary["coverage_appended"] == 1
    assert len(rows) == 4
    statuses = sorted(summary["demand_statuses"])
    assert statuses == ["generated", "generated", "generated"]


def test_existing_and_duplicate_demand_statuses(tmp_path: Path) -> None:
    from tools.perf_data_collection.grid_generator.query_coverage import build_query_generated_rows
    from tools.perf_data_collection.grid_generator.utils import load_csv_template_rows

    csv_path = _budget_csv(tmp_path)
    headers, source_rows = load_csv_template_rows(csv_path, require_rows=True)
    # One demand already covered by the template row, two identical new demands.
    demands = [
        _elementwise_demand((1, 4)),
        _elementwise_demand((2, 4)),
        _elementwise_demand((2, 4)),
    ]
    _rows, summary = build_query_generated_rows(
        csv_path=csv_path,
        headers=headers,
        source_rows=source_rows,
        demands=demands,
        row_limit=10,
        seed=0,
        exact_unlimited_workload_ids={"target"},
    )

    statuses = list(summary["demand_statuses"])
    assert statuses.count("existing") == 1
    assert statuses.count("generated") == 1
    assert statuses.count("duplicate") == 1


def test_mixed_sources_only_exempt_optimizer_args_exact_demands(tmp_path: Path) -> None:
    from tools.perf_data_collection.grid_generator.query_coverage import build_query_generated_rows
    from tools.perf_data_collection.grid_generator.utils import load_csv_template_rows

    csv_path = _budget_csv(tmp_path)
    headers, source_rows = load_csv_template_rows(csv_path, require_rows=True)
    demands = [
        _elementwise_demand((2, 4), "spec"),
        _elementwise_demand((3, 4), "spec"),
        _elementwise_demand((4, 4), "target"),
        _elementwise_demand((5, 4), "target"),
    ]

    rows, summary = build_query_generated_rows(
        csv_path=csv_path,
        headers=headers,
        source_rows=source_rows,
        demands=demands,
        row_limit=1,
        seed=0,
        exact_unlimited_workload_ids={"spec"},
    )

    assert len(rows) == 3
    assert summary["exact_generated"] == 3
    assert summary["exact_budget_truncated"] == 1
    statuses = list(summary["demand_statuses"])
    assert statuses[:2] == ["generated", "generated"]
    assert statuses[2:].count("generated") == 1
    assert statuses[2:].count("budget_truncated") == 1


def test_dfc_coverage_keeps_structural_expert_axis_at_anchor_values(tmp_path: Path) -> None:
    from tools.perf_data_collection.grid_generator.query_coverage import build_query_generated_rows
    from tools.perf_data_collection.grid_generator.utils import load_csv_template_rows

    headers = [
        "OP State",
        "Input Shapes",
        "Input Data Types",
        "Input Formats",
        "Output Shapes",
        "Output Data Types",
        "Output Formats",
        "EP Size",
        "Average Duration(us)",
    ]
    template = ",".join(
        [
            "dynamic",
            '"20000,6144;16,6144,4096;16,2048,6144;20000,8;65536;98304;20000,8"',
            "DT_BF16;INT8;INT8;INT32;INT64;INT64;FLOAT",
            "ND;FRACTAL_NZ;FRACTAL_NZ;ND;ND;ND;ND",
            '"20000,6144;16"',
            "DT_BF16;INT32",
            "ND;ND",
            "16",
            "1",
        ]
    )
    csv_path = tmp_path / "DispatchFFNCombine.csv"
    csv_path.write_text(",".join(headers) + "\n" + template + "\n", encoding="utf-8")
    csv_headers, source_rows = load_csv_template_rows(csv_path, require_rows=True)

    def dfc_demand(tokens: int, local_experts: int, ep_size: int) -> KernelQueryDemand:
        scales = local_experts * 4096, local_experts * 6144
        return KernelQueryDemand(
            projector_version="projector/v1",
            op_name="tensor_cast.dispatch_ffn_combine_quant.default",
            kernel_type="DispatchFFNCombine",
            query_mode="moe_fused",
            input_shapes=(
                (tokens, 6144),
                (local_experts, 6144, 4096),
                (local_experts, 2048, 6144),
                (tokens, 8),
                (scales[0],),
                (scales[1],),
                (tokens, 8),
            ),
            output_shapes=((tokens, 8, 6144),),
            input_dtypes=("DT_BF16", "INT8", "INT8", "INT32", "INT64", "INT64", "FLOAT"),
            output_dtypes=("DT_BF16",),
            attributes={"ep_size": ep_size},
        )

    demands = [
        dfc_demand(20000, 16, 16),
        dfc_demand(1, 16, 16),
        dfc_demand(20000, 8, 32),
    ]
    rows, summary = build_query_generated_rows(
        csv_path=csv_path,
        headers=csv_headers,
        source_rows=source_rows,
        demands=demands,
        row_limit=100,
        seed=0,
        exact_unlimited_workload_ids={""},
    )

    assert rows, "coverage should still interpolate the token axis"
    for row in rows:
        shapes = row["Input Shapes"].replace('"', "").split(";")
        experts = int(shapes[1].split(",")[0])
        scale1 = int(shapes[4])
        assert experts in (16, 8), f"expert axis must stay at anchor values, got {experts}"
        assert scale1 == experts * 4096, "scale slot must stay coupled to the expert count"
