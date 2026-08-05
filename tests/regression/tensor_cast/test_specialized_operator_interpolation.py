from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from tensor_cast.performance_model.profiling_database.data_source import QuerySource
from tensor_cast.performance_model.profiling_database.interpolating_data_source import InterpolatingDataSource
from tensor_cast.performance_model.profiling_database.profiling_data_source import ProfilingDataSource


class _FuncName:
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return self.value


class _ParallelConfig:
    expert_parallel_size = 16
    tensor_parallel_size = 1


_REAL_V018_DATA_DIR = (
    Path(__file__).resolve().parents[3]
    / "tensor_cast/performance_model/profiling_database/data/ATLAS_800_A3_752T_128G_DIE"
    / "vllm_ascend/vllm0.18.0_torch2.9.0_cann8.5"
)


def _make_op_info(func_name, args, out=None):
    if (
        func_name.startswith("tensor_cast.dispatch_ffn_combine")
        and len(args) == 4
        and isinstance(args[1], torch.Tensor)
        and args[1].ndim >= 3
    ):
        x, gmm1_w, gmm2_w, expert_indices = args
        if func_name == "tensor_cast.dispatch_ffn_combine.default":
            args = [x, expert_indices, gmm1_w, [], gmm2_w, [], 0, []]
        else:
            args = [x, expert_indices, gmm1_w, [], [], [], None, gmm2_w, [], [], [], None, 0, []]
    elif (
        func_name.startswith("tensor_cast.dispatch_ffn_combine")
        and len(args) == 2
        and isinstance(args[1], torch.Tensor)
    ):
        x, expert_indices = args
        weight_dtype = torch.bfloat16 if func_name == "tensor_cast.dispatch_ffn_combine.default" else torch.int8
        gmm1_w = torch.empty(16, 7168, 4096, device="meta", dtype=weight_dtype)
        gmm2_w = torch.empty(16, 2048, 7168, device="meta", dtype=weight_dtype)
        if func_name == "tensor_cast.dispatch_ffn_combine.default":
            args = [x, expert_indices, gmm1_w, [], gmm2_w, [], 0, []]
        else:
            args = [x, expert_indices, gmm1_w, [], [], [], None, gmm2_w, [], [], [], None, 0, []]
    if func_name.startswith("tensor_cast.dispatch_ffn_combine") and len(args) >= 8:
        args = list(args)
        gmm2_index = 4 if func_name == "tensor_cast.dispatch_ffn_combine.default" else 7
        for weight_index in (2, gmm2_index):
            weight = args[weight_index]
            if isinstance(weight, torch.Tensor) and weight.ndim == 3:
                per_expert = torch.empty(weight.shape[1:], device=weight.device, dtype=weight.dtype)
                args[weight_index] = [per_expert] * int(weight.shape[0])
    mock = MagicMock()
    mock.func = _FuncName(f"torch.ops.{func_name}")
    mock.args = tuple(args)
    mock.kwargs = {}
    mock.out = out
    return mock


def _write_text(path, content):
    path.write_text(content.strip(), encoding="utf-8")


def test_elementwise_axes_use_total_input_output_numel():
    axes = InterpolatingDataSource._elementwise_axes_from_shapes(
        [(2, 3), (1, 3)],
        (2, 3),
    )

    assert axes == {"io_numel": 15.0}


def test_elementwise_base_miss_does_not_recover_local_measured_exact(tmp_path):
    data_dir = tmp_path / "elementwise_fallback_only"
    data_dir.mkdir()
    _write_text(
        data_dir / "op_mapping.yaml",
        """
version: "test"
operator_mappings:
  "aten.mul.Tensor":
    kernel_type: Mul
    query_mode: elementwise
""",
    )
    _write_text(
        data_dir / "Mul.csv",
        """
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Duration(us)
"8,8","DT_BF16","ND","8,8","DT_BF16","ND",10.0
""",
    )
    base = ProfilingDataSource(data_dir)
    base.lookup = MagicMock(return_value=None)
    ds = InterpolatingDataSource(base)
    out = torch.empty(8, 8, device="meta", dtype=torch.bfloat16)

    result = ds.lookup(_make_op_info("aten.mul.Tensor", [out], out))

    base.lookup.assert_called_once()
    assert result is None
    assert ds.last_miss_reason != ""


def test_moe_fused_real_csv_keeps_full_weight_shapes_in_candidate_regime():
    ds = InterpolatingDataSource(ProfilingDataSource(_REAL_V018_DATA_DIR, parallel_config=_ParallelConfig()))
    tokens = 1
    op = _make_op_info(
        "tensor_cast.dispatch_ffn_combine_quant.default",
        [
            torch.empty(tokens, 7168, device="meta", dtype=torch.bfloat16),
            torch.empty(tokens, 8, device="meta", dtype=torch.int32),
            torch.empty(16, 7168, 4096, device="meta", dtype=torch.int8),
            [],
            [],
            [],
            None,
            torch.empty(16, 2048, 7168, device="meta", dtype=torch.int8),
            [],
            [],
            [],
            None,
            0,
            [],
        ],
        torch.empty(tokens, 8, 7168, device="meta", dtype=torch.bfloat16),
    )

    result = ds.lookup(op)

    assert result is not None
    assert result.source == QuerySource.INTERPOLATED
    assert tuple(result.details["exact_fields"]["gmm1_weight_shape"]) == (16, 7168, 4096)
    assert tuple(result.details["exact_fields"]["gmm2_weight_shape"]) == (16, 2048, 7168)
    assert "duplicate_count" not in result.details["matched_row_meta"][0]


def test_moe_fused_target_uses_projected_physical_dtype_signature():
    ds = InterpolatingDataSource(ProfilingDataSource(_REAL_V018_DATA_DIR, parallel_config=_ParallelConfig()))
    tokens = 180
    op = _make_op_info(
        "tensor_cast.dispatch_ffn_combine_quant.default",
        [
            torch.empty(1, tokens, 7168, device="meta", dtype=torch.bfloat16),
            torch.empty(16, 7168, 4096, device="meta", dtype=torch.int8),
            torch.empty(16, 2048, 7168, device="meta", dtype=torch.int8),
            torch.empty(tokens, 4, device="meta", dtype=torch.int32),
        ],
    )
    mapping = ds.base._op_mapping["operator_mappings"]["tensor_cast.dispatch_ffn_combine_quant.default"]

    target = ds._build_moe_fused_target(op, mapping)

    assert target is not None
    regime = dict(target.regime_key)
    assert regime["input_dtype_signature"] == (
        "DT_BF16",
        "INT8",
        "INT8",
        "INT32",
        "INT64",
        "INT64",
        "FLOAT",
    )
    assert "quant_subtype" not in regime


def test_moe_fused_missing_weight_shape_fails_closed():
    ds = InterpolatingDataSource(ProfilingDataSource(_REAL_V018_DATA_DIR, parallel_config=_ParallelConfig()))
    op = _make_op_info(
        "tensor_cast.dispatch_ffn_combine_quant.default",
        [
            torch.empty(1, 7168, device="meta", dtype=torch.bfloat16),
            torch.empty(1, 8, device="meta", dtype=torch.int32),
            [],
            [],
            [],
            [],
            None,
            torch.empty(16, 2048, 7168, device="meta", dtype=torch.int8),
            [],
            [],
            [],
            None,
            0,
            [],
        ],
    )

    mapping = ds.base._op_mapping["operator_mappings"]["tensor_cast.dispatch_ffn_combine_quant.default"]
    assert ds._interpolate_moe_fused(op, mapping) is None
    assert ds.last_miss_reason == "moe_fused_target_unextractable"


def test_moe_fused_rank1_activation_fails_closed():
    ds = InterpolatingDataSource(ProfilingDataSource(_REAL_V018_DATA_DIR, parallel_config=_ParallelConfig()))
    op = _make_op_info(
        "tensor_cast.dispatch_ffn_combine_quant.default",
        [
            torch.empty(7168, device="meta", dtype=torch.bfloat16),
            torch.empty(7168, 8, device="meta", dtype=torch.int32),
            torch.empty(16, 7168, 4096, device="meta", dtype=torch.int8),
            [],
            [],
            [],
            None,
            torch.empty(16, 2048, 7168, device="meta", dtype=torch.int8),
            [],
            [],
            [],
            None,
            0,
            [],
        ],
    )

    mapping = ds.base._op_mapping["operator_mappings"]["tensor_cast.dispatch_ffn_combine_quant.default"]
    assert ds._interpolate_moe_fused(op, mapping) is None
    assert ds.last_miss_reason == "moe_fused_target_unextractable"


def test_moe_fused_dispatch_ffn_combine_interpolates_tokens_only(tmp_path):
    data_dir = tmp_path / "moe_fused_tokens"
    data_dir.mkdir()
    _write_text(
        data_dir / "op_mapping.yaml",
        """
version: "test"
operator_mappings:
  "tensor_cast.dispatch_ffn_combine_quant.default":
    kernel_type: DispatchFFNCombine
    query_mode: moe_fused
    tc_input_count: 1
""",
    )
    rows = [
        '"1,120,7168;16,7168,4096;16,2048,7168;120,4;65536;114688;120,4","DT_BF16;INT8;INT8;INT32;INT64;INT64;FLOAT","ND;FRACTAL_NZ;FRACTAL_NZ;ND;ND;ND;ND","1,120,7168","DT_BF16","ND",16,10.0',
        '"1,240,7168;16,7168,4096;16,2048,7168;240,4;65536;114688;240,4","DT_BF16;INT8;INT8;INT32;INT64;INT64;FLOAT","ND;FRACTAL_NZ;FRACTAL_NZ;ND;ND;ND;ND","1,240,7168","DT_BF16","ND",16,20.0',
    ]
    _write_text(
        data_dir / "DispatchFFNCombine.csv",
        "\n".join(
            [
                "Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,EP Size,Duration(us)",
                *rows,
            ]
        ),
    )
    ds = InterpolatingDataSource(ProfilingDataSource(data_dir, parallel_config=_ParallelConfig()))
    op = _make_op_info(
        "tensor_cast.dispatch_ffn_combine_quant.default",
        [
            torch.empty(1, 180, 7168, device="meta", dtype=torch.bfloat16),
            torch.empty(16, 7168, 4096, device="meta", dtype=torch.int8),
            torch.empty(16, 2048, 7168, device="meta", dtype=torch.int8),
            torch.empty(180, 4, device="meta", dtype=torch.int32),
        ],
        torch.empty(1, 180, 7168, device="meta", dtype=torch.bfloat16),
    )

    result = ds.lookup(op)

    assert result is not None
    assert result.source == QuerySource.INTERPOLATED
    assert result.details["query_mode"] == "moe_fused"
    assert result.details["interpolation_path"] == "moe_fused_1d"
    assert result.details["interpolation_dim"] == 1
    assert result.details["axes"] == ["tokens"]
    assert "local_tokens" not in result.details["target_moe_axes"]
    assert "expert_tokens" not in result.details["target_moe_axes"]
    assert result.details["ep_size"] == 16


def test_moe_fused_does_not_mix_latency_columns(tmp_path):
    data_dir = tmp_path / "moe_fused_latency_source_mixed"
    data_dir.mkdir()
    _write_text(
        data_dir / "op_mapping.yaml",
        """
version: "test"
operator_mappings:
  "tensor_cast.dispatch_ffn_combine_quant.default":
    kernel_type: DispatchFFNCombine
    query_mode: moe_fused
    tc_input_count: 1
""",
    )
    _write_text(
        data_dir / "DispatchFFNCombine.csv",
        """
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,EP Size,Average Duration(us),Duration(us)
"1,120,7168;16,7168,4096;16,2048,7168;120,4;65536;114688;120,4","DT_BF16;INT8;INT8;INT32;INT64;INT64;FLOAT","ND;FRACTAL_NZ;FRACTAL_NZ;ND;ND;ND;ND","1,120,7168","DT_BF16","ND",16,10.0,10.0
"1,240,7168;16,7168,4096;16,2048,7168;240,4;65536;114688;240,4","DT_BF16;INT8;INT8;INT32;INT64;INT64;FLOAT","ND;FRACTAL_NZ;FRACTAL_NZ;ND;ND;ND;ND","1,240,7168","DT_BF16","ND",16,0.0,20.0
""",
    )
    ds = InterpolatingDataSource(ProfilingDataSource(data_dir, parallel_config=_ParallelConfig()))
    op = _make_op_info(
        "tensor_cast.dispatch_ffn_combine_quant.default",
        [
            torch.empty(1, 180, 7168, device="meta", dtype=torch.bfloat16),
            torch.empty(16, 7168, 4096, device="meta", dtype=torch.int8),
            torch.empty(16, 2048, 7168, device="meta", dtype=torch.int8),
            torch.empty(180, 4, device="meta", dtype=torch.int32),
        ],
        torch.empty(1, 180, 7168, device="meta", dtype=torch.bfloat16),
    )

    result = ds.lookup(op)

    assert result is None
    assert ds.last_miss_reason == "insufficient_filtered_candidates"


@pytest.mark.parametrize(
    "func_name,tokens,weight_dtype,expected_signature",
    [
        (
            "tensor_cast.dispatch_ffn_combine.default",
            180,
            torch.bfloat16,
            ("DT_BF16", "DT_BF16", "DT_BF16", "INT32", "INT64", "INT64", "FLOAT"),
        ),
        (
            "tensor_cast.dispatch_ffn_combine_quant_int4.default",
            180,
            torch.uint8,
            ("DT_BF16", "torch.uint8", "torch.uint8", "INT32", "INT64", "INT64", "FLOAT"),
        ),
        (
            "tensor_cast.dispatch_ffn_combine_fp8.default",
            180,
            torch.float8_e5m2,
            ("DT_BF16", "torch.float8_e5m2", "torch.float8_e5m2", "INT32", "INT64", "INT64", "FLOAT"),
        ),
        (
            "tensor_cast.dispatch_ffn_combine_mxfp4.default",
            180,
            torch.int4,
            ("DT_BF16", "torch.int4", "torch.int4", "INT32", "INT64", "INT64", "FLOAT"),
        ),
    ],
)
def test_moe_fused_does_not_reuse_w8a8_rows_for_other_physical_signatures(
    tmp_path, func_name, tokens, weight_dtype, expected_signature
):
    data_dir = tmp_path / func_name.split(".")[-2]
    data_dir.mkdir()
    _write_text(
        data_dir / "op_mapping.yaml",
        f"""
version: "test"
operator_mappings:
  "{func_name}":
    kernel_type: DispatchFFNCombine
    query_mode: moe_fused
    tc_input_count: 1
""",
    )
    _write_text(
        data_dir / "DispatchFFNCombine.csv",
        """
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,EP Size,Duration(us)
        "1,120,7168;16,7168,4096;16,2048,7168;120,4;65536;114688;120,4","DT_BF16;INT8;INT8;INT32;INT64;INT64;FLOAT","ND;FRACTAL_NZ;FRACTAL_NZ;ND;ND;ND;ND","1,120,7168","DT_BF16","ND",16,10.0
        "1,240,7168;16,7168,4096;16,2048,7168;240,4;65536;114688;240,4","DT_BF16;INT8;INT8;INT32;INT64;INT64;FLOAT","ND;FRACTAL_NZ;FRACTAL_NZ;ND;ND;ND;ND","1,240,7168","DT_BF16","ND",16,20.0
""",
    )
    ds = InterpolatingDataSource(ProfilingDataSource(data_dir, parallel_config=_ParallelConfig()))
    op = _make_op_info(
        func_name,
        [
            torch.empty(1, tokens, 7168, device="meta", dtype=torch.bfloat16),
            torch.empty(16, 7168, 4096, device="meta", dtype=weight_dtype),
            torch.empty(16, 2048, 7168, device="meta", dtype=weight_dtype),
            torch.empty(tokens, 4, device="meta", dtype=torch.int32),
        ],
        torch.empty(1, tokens, 7168, device="meta", dtype=torch.bfloat16),
    )

    mapping = ds.base._op_mapping["operator_mappings"][func_name]
    assert ds._interpolate_moe_fused(op, mapping) is None
    assert ds.last_miss_reason == "moe_fused_interpolation_failed"
    target_regime = ds.last_miss_details["attempts"][0]["target_regime"]
    assert target_regime["input_dtype_signature"] == expected_signature
    assert "quant_subtype" not in target_regime


def test_moe_fused_topk_is_discrete_and_not_part_of_interpolation_axes():
    topk = InterpolatingDataSource._moe_fused_topk(
        [(2, 4, 72, 7168), (2, 4, 72, 4)],
        tokens=576.0,
        input_dtypes=["DT_BF16", "INT32"],
    )

    assert topk == 4


def test_moe_fused_topk_requires_integer_dtype():
    topk = InterpolatingDataSource._moe_fused_topk(
        [(1, 180, 7168), (180, 4)],
        tokens=180.0,
        input_dtypes=["DT_BF16", ""],
    )

    assert topk is None


def test_moe_fused_topk_rejects_ambiguous_integer_tensors():
    topk = InterpolatingDataSource._moe_fused_topk(
        [(1, 180, 7168), (180, 4), (180, 8)],
        tokens=180.0,
        input_dtypes=["DT_BF16", "INT32", "INT64"],
    )

    assert topk is None


def test_moe_fused_max_dim_1_allows_token_interpolation(tmp_path):
    data_dir = tmp_path / "moe_fused_max_dim_1"
    data_dir.mkdir()
    _write_text(
        data_dir / "op_mapping.yaml",
        """
version: "test"
interpolation_policy:
  kernel_overrides:
    DispatchFFNCombine:
      max_interpolation_dim: 1
operator_mappings:
  "tensor_cast.dispatch_ffn_combine_quant.default":
    kernel_type: DispatchFFNCombine
    query_mode: moe_fused
    tc_input_count: 1
""",
    )
    rows = [
        '"1,120,7168;16,7168,4096;16,2048,7168;120,4;65536;114688;120,4","DT_BF16;INT8;INT8;INT32;INT64;INT64;FLOAT","ND;FRACTAL_NZ;FRACTAL_NZ;ND;ND;ND;ND","1,120,7168;16","DT_BF16;INT32","ND;ND",16,10.0',
        '"1,240,7168;16,7168,4096;16,2048,7168;240,4;65536;114688;240,4","DT_BF16;INT8;INT8;INT32;INT64;INT64;FLOAT","ND;FRACTAL_NZ;FRACTAL_NZ;ND;ND;ND;ND","1,240,7168;16","DT_BF16;INT32","ND;ND",16,20.0',
        '"3,40,7168;16,7168,4096;16,2048,7168;120,4;65536;114688;120,4","DT_BF16;INT8;INT8;INT32;INT64;INT64;FLOAT","ND;FRACTAL_NZ;FRACTAL_NZ;ND;ND;ND;ND","3,40,7168;16","DT_BF16;INT32","ND;ND",16,30.0',
        '"3,80,7168;16,7168,4096;16,2048,7168;240,4;65536;114688;240,4","DT_BF16;INT8;INT8;INT32;INT64;INT64;FLOAT","ND;FRACTAL_NZ;FRACTAL_NZ;ND;ND;ND;ND","3,80,7168;16","DT_BF16;INT32","ND;ND",16,40.0',
    ]
    _write_text(
        data_dir / "DispatchFFNCombine.csv",
        "\n".join(
            [
                "Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,EP Size,Duration(us)",
                *rows,
            ]
        ),
    )
    ds = InterpolatingDataSource(ProfilingDataSource(data_dir, parallel_config=_ParallelConfig()))
    op = _make_op_info(
        "tensor_cast.dispatch_ffn_combine_quant.default",
        [
            torch.empty(2, 72, 7168, device="meta", dtype=torch.bfloat16),
            torch.empty(144, 4, device="meta", dtype=torch.int32),
        ],
        torch.empty(2, 72, 7168, device="meta", dtype=torch.bfloat16),
    )

    result = ds.lookup(op)

    assert result is not None
    assert result.source == QuerySource.INTERPOLATED
    assert result.details["axes"] == ["tokens"]
    assert result.details["interpolation_dim"] == 1


def test_moe_fused_rejects_local_expert_mismatch(tmp_path):
    data_dir = tmp_path / "moe_fused_local_experts"
    data_dir.mkdir()
    _write_text(
        data_dir / "op_mapping.yaml",
        """
version: "test"
operator_mappings:
  "tensor_cast.dispatch_ffn_combine_quant.default":
    kernel_type: DispatchFFNCombine
    query_mode: moe_fused
    tc_input_count: 1
""",
    )
    rows = [
        '"1,120,7168;16,7168,4096;16,2048,7168;120,4","DT_BF16;INT8;INT8;INT32","ND;FRACTAL_NZ;FRACTAL_NZ;ND","1,120,7168","DT_BF16","ND",16,10.0',
        '"1,240,7168;16,7168,4096;16,2048,7168;240,4","DT_BF16;INT8;INT8;INT32","ND;FRACTAL_NZ;FRACTAL_NZ;ND","1,240,7168","DT_BF16","ND",16,20.0',
    ]
    _write_text(
        data_dir / "DispatchFFNCombine.csv",
        "\n".join(
            [
                "Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,EP Size,Duration(us)",
                *rows,
            ]
        ),
    )
    ds = InterpolatingDataSource(ProfilingDataSource(data_dir, parallel_config=_ParallelConfig()))
    op = _make_op_info(
        "tensor_cast.dispatch_ffn_combine_quant.default",
        [
            torch.empty(1, 180, 7168, device="meta", dtype=torch.bfloat16),
            torch.empty(8, 7168, 4096, device="meta", dtype=torch.int8),
            torch.empty(8, 2048, 7168, device="meta", dtype=torch.int8),
            torch.empty(180, 4, device="meta", dtype=torch.int32),
        ],
        torch.empty(1, 180, 7168, device="meta", dtype=torch.bfloat16),
    )

    result = ds.lookup(op)

    assert result is None
    assert ds.last_miss_reason == "moe_fused_interpolation_failed"
    assert ds.last_miss_details["attempts"][0]["status"] == "regime_key_unmatched"
    assert ds.last_miss_details["attempts"][0]["target_regime"]["gmm1_weight_shape"] == (8, 7168, 4096)


def test_moe_fused_rejects_rows_with_blank_ep_size_when_ep_column_exists(tmp_path):
    data_dir = tmp_path / "moe_fused_blank_ep"
    data_dir.mkdir()
    _write_text(
        data_dir / "op_mapping.yaml",
        """
version: "test"
operator_mappings:
  "tensor_cast.dispatch_ffn_combine_quant.default":
    kernel_type: DispatchFFNCombine
    query_mode: moe_fused
    tc_input_count: 1
""",
    )
    rows = [
        '"1,120,7168;16,7168,4096;16,2048,7168;120,4;65536;114688;120,4","DT_BF16;INT8;INT8;INT32;INT64;INT64;FLOAT","ND;FRACTAL_NZ;FRACTAL_NZ;ND;ND;ND;ND","1,120,7168","DT_BF16","ND",,10.0',
        '"1,240,7168;16,7168,4096;16,2048,7168;240,4;65536;114688;240,4","DT_BF16;INT8;INT8;INT32;INT64;INT64;FLOAT","ND;FRACTAL_NZ;FRACTAL_NZ;ND;ND;ND;ND","1,240,7168","DT_BF16","ND",,20.0',
    ]
    _write_text(
        data_dir / "DispatchFFNCombine.csv",
        "\n".join(
            [
                "Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,EP Size,Duration(us)",
                *rows,
            ]
        ),
    )
    ds = InterpolatingDataSource(ProfilingDataSource(data_dir, parallel_config=_ParallelConfig()))
    op = _make_op_info(
        "tensor_cast.dispatch_ffn_combine_quant.default",
        [
            torch.empty(1, 180, 7168, device="meta", dtype=torch.bfloat16),
            torch.empty(180, 4, device="meta", dtype=torch.int32),
        ],
        torch.empty(1, 180, 7168, device="meta", dtype=torch.bfloat16),
    )

    result = ds.lookup(op)

    assert result is None
    assert ds.last_miss_reason == "moe_fused_interpolation_failed"
    assert ds.last_miss_details["candidate_count"] == 0
    assert ds.last_miss_details["rejected_reasons"] == {"ep_size_missing": 2}


def test_moe_fused_requires_runtime_ep_size_when_csv_declares_ep_size(tmp_path):
    data_dir = tmp_path / "moe_fused_ep_not_configured"
    data_dir.mkdir()
    _write_text(
        data_dir / "op_mapping.yaml",
        """
version: "test"
operator_mappings:
  "tensor_cast.dispatch_ffn_combine_quant.default":
    kernel_type: DispatchFFNCombine
    query_mode: moe_fused
    tc_input_count: 1
""",
    )
    _write_text(
        data_dir / "DispatchFFNCombine.csv",
        """
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,EP Size,Duration(us)
"1,120,7168;120,4","DT_BF16;INT32","ND;ND","1,120,7168","DT_BF16","ND",16,10.0
"1,240,7168;240,4","DT_BF16;INT32","ND;ND","1,240,7168","DT_BF16","ND",16,20.0
""",
    )
    ds = InterpolatingDataSource(ProfilingDataSource(data_dir))
    op = _make_op_info(
        "tensor_cast.dispatch_ffn_combine_quant.default",
        [
            torch.empty(1, 180, 7168, device="meta", dtype=torch.bfloat16),
            torch.empty(180, 4, device="meta", dtype=torch.int32),
        ],
        torch.empty(1, 180, 7168, device="meta", dtype=torch.bfloat16),
    )

    assert ds.lookup(op) is None
    assert ds.last_miss_reason == "ep_size_not_configured"
    assert ds.last_miss_details["query_mode"] == "moe_fused"


def test_moe_fused_rejects_rows_when_ep_size_column_is_missing(tmp_path):
    data_dir = tmp_path / "moe_fused_missing_ep_column"
    data_dir.mkdir()
    _write_text(
        data_dir / "op_mapping.yaml",
        """
version: "test"
operator_mappings:
  "tensor_cast.dispatch_ffn_combine.default":
    kernel_type: DispatchFFNCombine
    query_mode: moe_fused
    tc_input_count: 1
""",
    )
    rows = [
        '"1,120,7168;16,7168,4096;16,2048,7168;120,4","DT_BF16;INT8;INT8;INT32","ND;FRACTAL_NZ;FRACTAL_NZ;ND","1,120,7168","DT_BF16","ND",10.0',
        '"1,240,7168;16,7168,4096;16,2048,7168;240,4","DT_BF16;INT8;INT8;INT32","ND;FRACTAL_NZ;FRACTAL_NZ;ND","1,240,7168","DT_BF16","ND",20.0',
    ]
    _write_text(
        data_dir / "DispatchFFNCombine.csv",
        "\n".join(
            [
                "Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Duration(us)",
                *rows,
            ]
        ),
    )
    ds = InterpolatingDataSource(ProfilingDataSource(data_dir, parallel_config=_ParallelConfig()))
    op = _make_op_info(
        "tensor_cast.dispatch_ffn_combine.default",
        [
            torch.empty(1, 180, 7168, device="meta", dtype=torch.bfloat16),
            torch.empty(180, 4, device="meta", dtype=torch.int32),
        ],
        torch.empty(1, 180, 7168, device="meta", dtype=torch.bfloat16),
    )

    result = ds.lookup(op)

    assert result is None
    assert ds.last_miss_reason == "moe_fused_interpolation_failed"
    assert ds.last_miss_details["candidate_count"] == 0
    assert ds.last_miss_details["rejected_reasons"] == {"ep_size_missing": 2}


def test_moe_fused_rejects_topk_mismatch(tmp_path):
    data_dir = tmp_path / "moe_fused_topk"
    data_dir.mkdir()
    _write_text(
        data_dir / "op_mapping.yaml",
        """
version: "test"
operator_mappings:
  "tensor_cast.dispatch_ffn_combine_quant.default":
    kernel_type: DispatchFFNCombine
    query_mode: moe_fused
    tc_input_count: 1
""",
    )
    rows = [
        '"1,120,7168;16,7168,4096;16,2048,7168;120,2","DT_BF16;INT8;INT8;INT32","ND;FRACTAL_NZ;FRACTAL_NZ;ND","1,120,7168","DT_BF16","ND",16,10.0',
        '"1,240,7168;16,7168,4096;16,2048,7168;240,8","DT_BF16;INT8;INT8;INT32","ND;FRACTAL_NZ;FRACTAL_NZ;ND","1,240,7168","DT_BF16","ND",16,20.0',
    ]
    _write_text(
        data_dir / "DispatchFFNCombine.csv",
        "\n".join(
            [
                "Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,EP Size,Duration(us)",
                *rows,
            ]
        ),
    )
    ds = InterpolatingDataSource(ProfilingDataSource(data_dir, parallel_config=_ParallelConfig()))
    op = _make_op_info(
        "tensor_cast.dispatch_ffn_combine_quant.default",
        [
            torch.empty(1, 180, 7168, device="meta", dtype=torch.bfloat16),
            torch.empty(180, 4, device="meta", dtype=torch.int32),
        ],
        torch.empty(1, 180, 7168, device="meta", dtype=torch.bfloat16),
    )

    result = ds.lookup(op)

    assert result is None
    assert ds.last_miss_reason == "moe_fused_interpolation_failed"
    assert ds.last_miss_details["attempts"][0]["status"] == "regime_key_unmatched"
    assert ds.last_miss_details["attempts"][0]["target_regime"]["topk"] == 4


def test_elementwise_interpolates_guarded_1d_with_total_io_axis(tmp_path):
    data_dir = tmp_path / "elementwise_2d"
    data_dir.mkdir()
    _write_text(
        data_dir / "op_mapping.yaml",
        """
version: "test"
operator_mappings:
  "aten.add.Tensor":
    kernel_type: Add
    query_mode: elementwise
""",
    )
    rows = [
        '"8,10","DT_BF16","ND","8,10","DT_BF16","ND",16.0',
        '"8,20","DT_BF16","ND","8,20","DT_BF16","ND",24.0',
        '"12,10","DT_BF16","ND","12,10","DT_BF16","ND",24.0',
        '"12,20","DT_BF16","ND","12,20","DT_BF16","ND",36.0',
    ]
    _write_text(
        data_dir / "Add.csv",
        "\n".join(
            [
                "Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Duration(us)",
                *rows,
            ]
        ),
    )
    ds = InterpolatingDataSource(ProfilingDataSource(data_dir))
    out = torch.empty(10, 15, device="meta", dtype=torch.bfloat16)
    op = _make_op_info(
        "aten.add.Tensor",
        [torch.empty(10, 15, device="meta", dtype=torch.bfloat16)],
        out,
    )

    result = ds.lookup(op)

    assert result is not None
    assert result.source == QuerySource.INTERPOLATED
    assert result.details["query_mode"] == "elementwise"
    assert result.details["interpolation_path"] == "elementwise_1d"
    assert result.details["interpolation_dim"] == 1
    assert result.details["axes"] == ["io_numel"]
    assert result.latency_us == pytest.approx(24.0)
    assert result.details["target_elementwise_axes"] == {"io_numel": 300.0}


def test_elementwise_skipped_axis_groups_return_clean_miss(tmp_path, monkeypatch):
    data_dir = tmp_path / "elementwise_skipped_axes"
    data_dir.mkdir()
    _write_text(
        data_dir / "op_mapping.yaml",
        """
version: "test"
operator_mappings:
  "aten.add.Tensor":
    kernel_type: Add
    query_mode: elementwise
""",
    )
    _write_text(
        data_dir / "Add.csv",
        """
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Duration(us)
"8,10","DT_BF16","ND","8,10","DT_BF16","ND",16.0
"12,20","DT_BF16","ND","12,20","DT_BF16","ND",36.0
""",
    )
    ds = InterpolatingDataSource(ProfilingDataSource(data_dir))
    monkeypatch.setattr(ds, "_elementwise_axis_groups_for", lambda _group: (("missing_axis",),))
    op = _make_op_info(
        "aten.add.Tensor",
        [torch.empty(10, 15, device="meta", dtype=torch.bfloat16)],
        torch.empty(10, 15, device="meta", dtype=torch.bfloat16),
    )

    assert ds.lookup(op) is None
    assert ds.last_miss_reason == "candidate_group_failed"


def test_elementwise_interpolates_total_io_for_fixed_first_dimension(tmp_path):
    data_dir = tmp_path / "elementwise_axis_1"
    data_dir.mkdir()
    _write_text(
        data_dir / "op_mapping.yaml",
        """
version: "test"
operator_mappings:
  "aten.add.Tensor":
    kernel_type: Add
    query_mode: elementwise
""",
    )
    _write_text(
        data_dir / "Add.csv",
        """
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Duration(us)
"8,10","DT_BF16","ND","8,10","DT_BF16","ND",10.0
"8,20","DT_BF16","ND","8,20","DT_BF16","ND",20.0
""",
    )
    ds = InterpolatingDataSource(ProfilingDataSource(data_dir))
    out = torch.empty(8, 15, device="meta", dtype=torch.bfloat16)

    result = ds.lookup(_make_op_info("aten.add.Tensor", [out], out))

    assert result is not None
    assert result.source == QuerySource.INTERPOLATED
    assert result.latency_us == pytest.approx(15.0)
    assert result.details["axes"] == ["io_numel"]
    assert result.details["target_elementwise_axes"] == {"io_numel": 240.0}


def test_elementwise_same_total_io_coordinate_returns_interpolated_result(tmp_path):
    data_dir = tmp_path / "elementwise_axis_1_boundary"
    data_dir.mkdir()
    _write_text(
        data_dir / "op_mapping.yaml",
        """
version: "test"
operator_mappings:
  "aten.add.Tensor":
    kernel_type: Add
    query_mode: elementwise
""",
    )
    _write_text(
        data_dir / "Add.csv",
        """
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Duration(us)
"8,10","DT_BF16","ND","8,10","DT_BF16","ND",10.0
"8,20","DT_BF16","ND","8,20","DT_BF16","ND",20.0
"12,10","DT_BF16","ND","12,10","DT_BF16","ND",100.0
"12,20","DT_BF16","ND","12,20","DT_BF16","ND",200.0
""",
    )
    ds = InterpolatingDataSource(ProfilingDataSource(data_dir))
    out = torch.empty(8, 15, device="meta", dtype=torch.bfloat16)

    result = ds.lookup(_make_op_info("aten.add.Tensor", [out], out))

    assert result is not None
    assert result.source == QuerySource.INTERPOLATED
    assert result.latency_us == pytest.approx(100.0)
    assert result.details["axes"] == ["io_numel"]


def test_elementwise_rank3_uses_total_io_axis(tmp_path):
    data_dir = tmp_path / "elementwise_rank3_tail"
    data_dir.mkdir()
    _write_text(
        data_dir / "op_mapping.yaml",
        """
version: "test"
operator_mappings:
  "aten.add.Tensor":
    kernel_type: Add
    query_mode: elementwise
""",
    )
    rows = [
        '"8,2,6","DT_BF16","ND","8,2,6","DT_BF16","ND",16.0',
        '"8,3,8","DT_BF16","ND","8,3,8","DT_BF16","ND",24.0',
        '"12,2,6","DT_BF16","ND","12,2,6","DT_BF16","ND",24.0',
        '"12,3,8","DT_BF16","ND","12,3,8","DT_BF16","ND",36.0',
    ]
    _write_text(
        data_dir / "Add.csv",
        "\n".join(
            [
                "Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Duration(us)",
                *rows,
            ]
        ),
    )
    ds = InterpolatingDataSource(ProfilingDataSource(data_dir))
    out = torch.empty(10, 2, 9, device="meta", dtype=torch.bfloat16)

    result = ds.lookup(
        _make_op_info(
            "aten.add.Tensor",
            [torch.empty(10, 2, 9, device="meta", dtype=torch.bfloat16)],
            out,
        )
    )

    assert result is not None
    assert result.source == QuerySource.INTERPOLATED
    assert result.latency_us == pytest.approx(24.0)
    assert result.details["axes"] == ["io_numel"]


def test_elementwise_input_signature_separates_broadcast_and_full_tensor_inputs(tmp_path):
    data_dir = tmp_path / "elementwise_input_signature"
    data_dir.mkdir()
    _write_text(
        data_dir / "op_mapping.yaml",
        """
version: "test"
operator_mappings:
  "aten.add.Tensor":
    kernel_type: Add
    query_mode: elementwise
""",
    )
    rows = [
        '"128,7168;7168","DT_BF16;DT_BF16","ND;ND","128,7168","DT_BF16","ND",10.0',
        '"256,7168;7168","DT_BF16;DT_BF16","ND;ND","256,7168","DT_BF16","ND",20.0',
        '"128,7168;128,7168","DT_BF16;DT_BF16","ND;ND","128,7168","DT_BF16","ND",100.0',
        '"256,7168;256,7168","DT_BF16;DT_BF16","ND;ND","256,7168","DT_BF16","ND",200.0',
    ]
    _write_text(
        data_dir / "Add.csv",
        "\n".join(
            [
                "Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Duration(us)",
                *rows,
            ]
        ),
    )
    ds = InterpolatingDataSource(ProfilingDataSource(data_dir))
    out = torch.empty(192, 7168, device="meta", dtype=torch.bfloat16)

    broadcast_result = ds.lookup(
        _make_op_info(
            "aten.add.Tensor",
            [
                torch.empty(192, 7168, device="meta", dtype=torch.bfloat16),
                torch.empty(7168, device="meta", dtype=torch.bfloat16),
            ],
            out,
        )
    )
    full_tensor_result = ds.lookup(
        _make_op_info(
            "aten.add.Tensor",
            [
                torch.empty(192, 7168, device="meta", dtype=torch.bfloat16),
                torch.empty(192, 7168, device="meta", dtype=torch.bfloat16),
            ],
            out,
        )
    )

    assert broadcast_result is not None
    assert broadcast_result.source == QuerySource.INTERPOLATED
    assert broadcast_result.latency_us == pytest.approx(15.0)
    assert broadcast_result.details["exact_fields"]["broadcast_pattern"] == [
        [2, ["same", "same"]],
        [1, ["missing", "same"]],
    ]
    assert full_tensor_result is not None
    assert full_tensor_result.source == QuerySource.INTERPOLATED
    assert full_tensor_result.latency_us == pytest.approx(150.0)
    assert full_tensor_result.details["exact_fields"]["broadcast_pattern"] == [
        [2, ["same", "same"]],
        [2, ["same", "same"]],
    ]


def test_elementwise_max_dim_1_allows_total_io_interpolation(tmp_path):
    data_dir = tmp_path / "elementwise_max_dim_1"
    data_dir.mkdir()
    _write_text(
        data_dir / "op_mapping.yaml",
        """
version: "test"
interpolation_policy:
  kernel_overrides:
    Add:
      max_interpolation_dim: 1
operator_mappings:
  "aten.add.Tensor":
    kernel_type: Add
    query_mode: elementwise
""",
    )
    rows = [
        '"8,10","DT_BF16","ND","8,10","DT_BF16","ND",16.0',
        '"8,20","DT_BF16","ND","8,20","DT_BF16","ND",24.0',
        '"12,10","DT_BF16","ND","12,10","DT_BF16","ND",24.0',
        '"12,20","DT_BF16","ND","12,20","DT_BF16","ND",36.0',
    ]
    _write_text(
        data_dir / "Add.csv",
        "\n".join(
            [
                "Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Duration(us)",
                *rows,
            ]
        ),
    )
    ds = InterpolatingDataSource(ProfilingDataSource(data_dir))
    out = torch.empty(10, 15, device="meta", dtype=torch.bfloat16)
    op = _make_op_info(
        "aten.add.Tensor",
        [torch.empty(10, 15, device="meta", dtype=torch.bfloat16)],
        out,
    )

    result = ds.lookup(op)

    assert result is not None
    assert result.source == QuerySource.INTERPOLATED
    assert result.details["axes"] == ["io_numel"]


def test_elementwise_interpolates_1d_total_io(tmp_path):
    data_dir = tmp_path / "elementwise_1d"
    data_dir.mkdir()
    _write_text(
        data_dir / "op_mapping.yaml",
        """
version: "test"
operator_mappings:
  "aten.mul.Tensor":
    kernel_type: Mul
    query_mode: elementwise
""",
    )
    rows = [
        '"8,8","DT_BF16","ND","8,8","DT_BF16","ND",10.0',
        '"16,8","DT_BF16","ND","16,8","DT_BF16","ND",20.0',
    ]
    _write_text(
        data_dir / "Mul.csv",
        "\n".join(
            [
                "Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Duration(us)",
                *rows,
            ]
        ),
    )
    ds = InterpolatingDataSource(ProfilingDataSource(data_dir))
    out = torch.empty(12, 8, device="meta", dtype=torch.bfloat16)
    op = _make_op_info(
        "aten.mul.Tensor",
        [torch.empty(12, 8, device="meta", dtype=torch.bfloat16)],
        out,
    )

    result = ds.lookup(op)

    assert result is not None
    assert result.source == QuerySource.INTERPOLATED
    assert result.latency_us == pytest.approx(15.0)
    assert result.details["interpolation_path"] == "elementwise_1d"
    assert result.details["axes"] == ["io_numel"]


def test_elementwise_interpolation_tries_alternate_kernel_types(tmp_path):
    data_dir = tmp_path / "elementwise_alternate_kernel"
    data_dir.mkdir()
    _write_text(
        data_dir / "op_mapping.yaml",
        """
version: "test"
operator_mappings:
  "aten.add.Tensor":
    kernel_type: Add
    alternate_kernel_types: [AddAiCore]
    query_mode: elementwise
""",
    )
    rows = [
        '"8,8","DT_BF16","ND","8,8","DT_BF16","ND",10.0',
        '"16,8","DT_BF16","ND","16,8","DT_BF16","ND",20.0',
    ]
    _write_text(
        data_dir / "AddAiCore.csv",
        "\n".join(
            [
                "Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Duration(us)",
                *rows,
            ]
        ),
    )
    ds = InterpolatingDataSource(ProfilingDataSource(data_dir))
    out = torch.empty(12, 8, device="meta", dtype=torch.bfloat16)
    op = _make_op_info(
        "aten.add.Tensor",
        [torch.empty(12, 8, device="meta", dtype=torch.bfloat16)],
        out,
    )

    result = ds.lookup(op)

    assert result is not None
    assert result.source == QuerySource.INTERPOLATED
    assert result.latency_us == pytest.approx(15.0)
    assert result.details["kernel_type"] == "AddAiCore"
    assert result.details["interpolation_path"] == "elementwise_1d"


def test_elementwise_interpolation_rejects_cross_dtype_candidates(tmp_path):
    data_dir = tmp_path / "elementwise_dtype_scale"
    data_dir.mkdir()
    _write_text(
        data_dir / "op_mapping.yaml",
        """
version: "test"
operator_mappings:
  "aten.mul.Tensor":
    kernel_type: Mul
    query_mode: elementwise
""",
    )
    rows = [
        '"8,8","DT_BF16","ND","8,8","DT_BF16","ND",10.0',
        '"16,8","DT_BF16","ND","16,8","DT_BF16","ND",20.0',
    ]
    _write_text(
        data_dir / "Mul.csv",
        "\n".join(
            [
                "Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Duration(us)",
                *rows,
            ]
        ),
    )
    ds = InterpolatingDataSource(ProfilingDataSource(data_dir))
    out = torch.empty(12, 8, device="meta", dtype=torch.float32)
    op = _make_op_info(
        "aten.mul.Tensor",
        [torch.empty(12, 8, device="meta", dtype=torch.float32)],
        out,
    )

    result = ds.lookup(op)

    assert result is None
    assert ds.last_miss_reason == "insufficient_filtered_candidates"
    assert ds.last_miss_details["candidate_count"] == 0


def test_elementwise_exact_coordinate_rejects_cross_dtype_candidate(tmp_path):
    data_dir = tmp_path / "elementwise_exact_cross_dtype"
    data_dir.mkdir()
    _write_text(
        data_dir / "op_mapping.yaml",
        """
version: "test"
operator_mappings:
  "aten.mul.Tensor":
    kernel_type: Mul
    query_mode: elementwise
""",
    )
    _write_text(
        data_dir / "Mul.csv",
        """
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Duration(us)
"8,8","DT_BF16","ND","8,8","DT_BF16","ND",10.0
""",
    )
    ds = InterpolatingDataSource(ProfilingDataSource(data_dir))
    out = torch.empty(8, 8, device="meta", dtype=torch.float32)
    op = _make_op_info("aten.mul.Tensor", [torch.empty_like(out)], out)

    mapping = ds.base._op_mapping["operator_mappings"]["aten.mul.Tensor"]
    assert ds._interpolate_elementwise(op, mapping) is None
    assert ds.last_miss_reason == "insufficient_filtered_candidates"


def test_elementwise_base_exact_hit_is_returned_unchanged(tmp_path):
    data_dir = tmp_path / "elementwise_exact_same_dtype"
    data_dir.mkdir()
    _write_text(
        data_dir / "op_mapping.yaml",
        """
version: "test"
operator_mappings:
  "aten.mul.Tensor":
    kernel_type: Mul
    query_mode: elementwise
""",
    )
    _write_text(
        data_dir / "Mul.csv",
        """
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Duration(us)
"8,8","DT_BF16","ND","8,8","DT_BF16","ND",10.0
""",
    )
    ds = InterpolatingDataSource(ProfilingDataSource(data_dir))
    out = torch.empty(8, 8, device="meta", dtype=torch.bfloat16)
    op = _make_op_info("aten.mul.Tensor", [torch.empty_like(out)], out)

    result = ds.lookup(op)

    assert result is not None
    assert result.source == QuerySource.MEASURED
    assert "interpolation_path" not in result.details


def test_elementwise_exact_coordinate_requires_full_output_shape(tmp_path):
    data_dir = tmp_path / "elementwise_exact_full_output_shape"
    data_dir.mkdir()
    _write_text(
        data_dir / "op_mapping.yaml",
        """
version: "test"
operator_mappings:
  "aten.mul.Tensor":
    kernel_type: Mul
    query_mode: elementwise
""",
    )
    _write_text(
        data_dir / "Mul.csv",
        """
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Duration(us)
"8,4,16","DT_BF16","ND","8,4,16","DT_BF16","ND",10.0
""",
    )
    ds = InterpolatingDataSource(ProfilingDataSource(data_dir))
    out = torch.empty(8, 8, 8, device="meta", dtype=torch.bfloat16)
    op = _make_op_info("aten.mul.Tensor", [torch.empty_like(out)], out)

    assert ds.lookup(op) is None


def test_elementwise_does_not_mix_latency_columns(tmp_path):
    data_dir = tmp_path / "elementwise_latency_source_mixed"
    data_dir.mkdir()
    _write_text(
        data_dir / "op_mapping.yaml",
        """
version: "test"
operator_mappings:
  "aten.add.Tensor":
    kernel_type: Add
    query_mode: elementwise
""",
    )
    _write_text(
        data_dir / "Add.csv",
        """
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Average Duration(us),Duration(us)
"10,10","DT_BF16","ND","10,10","DT_BF16","ND",10.0,10.0
"20,10","DT_BF16","ND","20,10","DT_BF16","ND",0.0,20.0
""",
    )
    ds = InterpolatingDataSource(ProfilingDataSource(data_dir))
    out = torch.empty(15, 10, device="meta", dtype=torch.bfloat16)
    op = _make_op_info("aten.add.Tensor", [torch.empty(15, 10, device="meta", dtype=torch.bfloat16)], out)

    result = ds.lookup(op)

    assert result is None
    assert ds.last_miss_reason == "insufficient_filtered_candidates"
