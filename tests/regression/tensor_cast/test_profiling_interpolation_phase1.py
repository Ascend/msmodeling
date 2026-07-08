import importlib
import sys
from unittest.mock import MagicMock

import pytest
import torch

from tensor_cast.performance_model.profiling_database import interpolation_index as interpolation_index_module
from tensor_cast.performance_model.profiling_database import (
    interpolating_data_source as interpolating_data_source_module,
)
from tensor_cast.performance_model.profiling_database.interpolation_index import (
    CandidateIndex,
    CandidateGroup,
    CandidatePoint,
    InterpolationResult,
    InterpolationTarget,
    make_regime_key,
)
from tensor_cast.performance_model.profiling_database.data_source import QueryResult, QuerySource
from tensor_cast.performance_model.profiling_database.interpolating_data_source import (
    InterpolatingDataSource,
    _ATTENTION_AXIS_GROUPS,
    _attention_kv_heads_from_key,
    _to_int_cell,
)
from tensor_cast.performance_model.profiling_database.profiling_data_source import ProfilingDataSource, SubKernelSpec


def _make_op_info(func, args):
    mock = MagicMock()
    mock.func = func
    mock.args = tuple(args)
    mock.kwargs = {}
    mock.out = None
    return mock


def _write_text(path, content):
    path.write_text(content.strip(), encoding="utf-8")


def _matmul_row(m_dim, k_dim, n_dim):
    latency = float(m_dim + k_dim + n_dim)
    return f'"{m_dim},{k_dim};{k_dim},{n_dim}","DT_BF16;DT_BF16","ND;ND","{m_dim},{n_dim}","DT_BF16","ND",{latency}'


def _matmul_row_with_dtype(m_dim, k_dim, n_dim, dtype):
    latency = float(m_dim + k_dim + n_dim)
    return f'"{m_dim},{k_dim};{k_dim},{n_dim}","{dtype};{dtype}","ND;ND","{m_dim},{n_dim}","{dtype}","ND",{latency}'


def _matmul_row_transposed_weight(m_dim, k_dim, n_dim):
    latency = float(m_dim + k_dim + n_dim)
    return f'"{m_dim},{k_dim};{n_dim},{k_dim}","DT_BF16;DT_BF16","ND;ND","{m_dim},{n_dim}","DT_BF16","ND",{latency}'


def test_interpolation_index_class_definitions_are_covered_in_test_context():
    module_name = "tensor_cast.performance_model.profiling_database._interpolation_index_coverage"
    spec = importlib.util.spec_from_file_location(module_name, interpolation_index_module.__file__)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)

    assert module.CandidatePoint.__name__ == "CandidatePoint"
    assert module.InterpolationTarget.__name__ == "InterpolationTarget"
    assert module.InterpolationResult.__name__ == "InterpolationResult"
    assert module.CandidateGroup.__name__ == "CandidateGroup"
    assert module.CandidateIndex.__name__ == "CandidateIndex"


def test_interpolation_index_dataclasses_and_regime_matching():
    key = make_regime_key({"kernel_type": "MatMulV2", "dtype": ["DT_BF16"]})
    target = InterpolationTarget(
        func_name="aten.mm.default",
        kernel_type="MatMulV2",
        axes={"M": 150.0},
        regime_key=key,
        tc_shapes=[(150, 64), (64, 256)],
        input_dtypes=["DT_BF16", "DT_BF16"],
        query_mode="compute",
    )
    points = [
        CandidatePoint(
            kernel_type="MatMulV2",
            axes={"M": 100.0},
            latency_us=10.0,
            regime_key=key,
            input_shapes=[(100, 64), (64, 256)],
            input_dtypes=["DT_BF16", "DT_BF16"],
            row_index=1,
            row_meta={"row": 1},
        ),
        CandidatePoint(
            kernel_type="MatMulV2",
            axes={"M": 200.0},
            latency_us=20.0,
            regime_key=key,
            input_shapes=[(200, 64), (64, 256)],
            input_dtypes=["DT_BF16", "DT_BF16"],
            row_index=2,
            row_meta={"row": 2},
        ),
    ]

    group = CandidateGroup(key, points)
    result = group.interpolate(target.axes, [["M"]])

    assert result is not None
    assert isinstance(result, InterpolationResult)
    assert result.latency_us == pytest.approx(15.0)
    assert result.matched_points == points
    assert result.details["exact_fields"]["dtype"] == ["DT_BF16"]

    candidate_index = CandidateIndex(points)
    compute_groups = candidate_index.candidate_groups_matching(key)
    assert len(compute_groups) == 1
    assert compute_groups[0].regime_key == key
    assert compute_groups[0].points == points
    assert candidate_index.candidate_groups_matching(key)[0].regime_key == key
    assert candidate_index.candidate_groups_matching(make_regime_key({"kernel_type": "Other"})) == []


def test_candidate_group_axis_matcher_allows_equivalent_non_selected_axis():
    key = make_regime_key({"kernel_type": "FusedInferAttentionScore"})
    group = CandidateGroup(
        key,
        [
            CandidatePoint("FusedInferAttentionScore", {"seq": 1000.0, "q_tokens": 144.0}, 100.0, key),
            CandidatePoint("FusedInferAttentionScore", {"seq": 2000.0, "q_tokens": 144.0}, 200.0, key),
        ],
    )

    without_matcher = group.interpolate({"seq": 1500.0, "q_tokens": 136.0}, [("seq",)])
    with_matcher = group.interpolate(
        {"seq": 1500.0, "q_tokens": 136.0},
        [("seq",)],
        axis_matchers={"q_tokens": lambda candidate, target: int(candidate) == 144 and int(target) == 136},
    )

    assert without_matcher is None
    assert with_matcher is not None
    assert with_matcher.latency_us == pytest.approx(150.0)


def test_interpolation_index_requires_strong_target_fields():
    candidate_key = make_regime_key({"kernel_type": "FusedInferAttentionScore", "dtype": "DT_BF16"})
    target_key = make_regime_key(
        {
            "kernel_type": "FusedInferAttentionScore",
            "dtype": "DT_BF16",
            "quant_mode": "quantized",
        }
    )
    point = CandidatePoint(
        kernel_type="FusedInferAttentionScore",
        axes={"seq": 1000.0},
        latency_us=10.0,
        regime_key=candidate_key,
    )

    index = CandidateIndex([point])

    assert index.candidate_groups_matching(target_key)
    assert index.candidate_groups_matching(candidate_key, required_target_fields={"quant_mode"}) == []
    assert index.candidate_groups_matching(target_key, required_target_fields={"quant_mode"}) == []


def test_interpolation_index_group_matching():
    nd_key = make_regime_key({"kernel_type": "MatMulV2", "dtype": "DT_BF16", "layout": "ND"})
    nz_key = make_regime_key({"kernel_type": "MatMulV2", "dtype": "DT_BF16", "layout": "FRACTAL_NZ"})
    quant_key = make_regime_key({"kernel_type": "FusedInferAttentionScore", "dtype": "DT_BF16", "quant_mode": "W8A8"})
    points = [
        CandidatePoint("MatMulV2", {"M": 100.0}, 10.0, nd_key, row_index=1),
        CandidatePoint("MatMulV2", {"M": 200.0}, 20.0, nd_key, row_index=2),
        CandidatePoint("MatMulV2", {"M": 100.0}, 30.0, nz_key, row_index=3),
        CandidatePoint("FusedInferAttentionScore", {"seq": 128.0}, 40.0, quant_key, row_index=4),
    ]

    base_index = CandidateIndex(points)
    target_key_without_layout = make_regime_key({"kernel_type": "MatMulV2", "dtype": "DT_BF16"})
    exact_groups = base_index.candidate_groups_matching(target_key_without_layout)
    layout_pooled_groups = base_index.candidate_groups_matching(
        target_key_without_layout,
        allow_extra_fields={"layout"},
    )

    assert exact_groups == []
    assert len(layout_pooled_groups) == 2
    assert {group.regime_key for group in layout_pooled_groups} == {nd_key, nz_key}
    compute_groups = CandidateIndex(points).candidate_groups_matching(nd_key)
    attention_groups = CandidateIndex(points).candidate_groups_matching(
        quant_key, required_target_fields={"quant_mode"}
    )

    assert len(compute_groups) == 1
    assert compute_groups[0].points[:2] == points[:2]
    assert len(attention_groups) == 1
    assert attention_groups[0].points == [points[3]]
    assert (
        CandidateIndex(points).candidate_groups_matching(
            make_regime_key({"kernel_type": "FusedInferAttentionScore", "dtype": "DT_BF16", "quant_mode": "W4A8"}),
            required_target_fields={"quant_mode"},
        )
        == []
    )


@pytest.fixture
def matmul_grid_dir(tmp_path):
    data_dir = tmp_path / "matmul_grid"
    data_dir.mkdir()
    _write_text(
        data_dir / "op_mapping.yaml",
        """
version: "test"
operator_mappings:
  "aten.mm.default":
    kernel_type: MatMulV2
""",
    )
    rows = [_matmul_row(m, k, n) for m in (100, 200) for k in (64, 128) for n in (256, 512)]
    _write_text(
        data_dir / "MatMulV2.csv",
        "\n".join(
            [
                "Input Shapes,Input Data Types,Input Formats,Output Shapes,"
                "Output Data Types,Output Formats,Duration(us)",
                *rows,
            ]
        ),
    )
    return data_dir


@pytest.fixture
def matmul_invalid_latency_dir(tmp_path):
    data_dir = tmp_path / "matmul_invalid_latency"
    data_dir.mkdir()
    _write_text(
        data_dir / "op_mapping.yaml",
        """
version: "test"
operator_mappings:
  "aten.mm.default":
    kernel_type: MatMulV2
""",
    )
    rows = [
        '"150,96;96,300","DT_BF16;DT_BF16","ND;ND","150,300","DT_BF16","ND",bad_latency',
        *[_matmul_row(m, k, n) for m in (100, 200) for k in (64, 128) for n in (256, 512)],
    ]
    _write_text(
        data_dir / "MatMulV2.csv",
        "\n".join(
            [
                "Input Shapes,Input Data Types,Input Formats,Output Shapes,"
                "Output Data Types,Output Formats,Duration(us)",
                *rows,
            ]
        ),
    )
    return data_dir


@pytest.fixture
def matmul_dtype_mismatch_dir(tmp_path):
    data_dir = tmp_path / "matmul_dtype_mismatch"
    data_dir.mkdir()
    _write_text(
        data_dir / "op_mapping.yaml",
        """
version: "test"
operator_mappings:
  "aten.mm.default":
    kernel_type: MatMulV3
""",
    )
    rows = [_matmul_row_with_dtype(m, k, n, "FLOAT") for m in (100, 200) for k in (64, 128) for n in (256, 512)]
    _write_text(
        data_dir / "MatMulV3.csv",
        "\n".join(
            [
                "Input Shapes,Input Data Types,Input Formats,Output Shapes,"
                "Output Data Types,Output Formats,Duration(us)",
                *rows,
            ]
        ),
    )
    return data_dir


@pytest.fixture
def matmul_missing_input_format_dir(tmp_path):
    data_dir = tmp_path / "matmul_missing_input_format"
    data_dir.mkdir()
    _write_text(
        data_dir / "op_mapping.yaml",
        """
version: "test"
operator_mappings:
  "aten.mm.default":
    kernel_type: MatMulV2
""",
    )
    rows = [
        f'"{m},{k};{k},{n}","DT_BF16;DT_BF16","ND","{m},{n}","DT_BF16","ND",{float(m + k + n)}'
        for m in (100, 200)
        for k in (64, 128)
        for n in (256, 512)
    ]
    _write_text(
        data_dir / "MatMulV2.csv",
        "\n".join(
            [
                "Input Shapes,Input Data Types,Input Formats,Output Shapes,"
                "Output Data Types,Output Formats,Duration(us)",
                *rows,
            ]
        ),
    )
    return data_dir


@pytest.fixture
def matmul_transposed_weight_dir(tmp_path):
    data_dir = tmp_path / "matmul_transposed_weight"
    data_dir.mkdir()
    _write_text(
        data_dir / "op_mapping.yaml",
        """
version: "test"
operator_mappings:
  "aten.mm.default":
    kernel_type: MatMulV2
""",
    )
    rows = [_matmul_row_transposed_weight(m, k, n) for m in (100, 200) for k in (64, 128) for n in (256, 512)]
    _write_text(
        data_dir / "MatMulV2.csv",
        "\n".join(
            [
                "Input Shapes,Input Data Types,Input Formats,Output Shapes,"
                "Output Data Types,Output Formats,Duration(us)",
                *rows,
            ]
        ),
    )
    return data_dir


def test_compute_prefers_lowest_valid_axes_when_n_is_exact(matmul_grid_dir):
    ds = InterpolatingDataSource(ProfilingDataSource(matmul_grid_dir))
    op = _make_op_info(
        torch.ops.aten.mm.default,
        [
            torch.empty(150, 96, device="meta", dtype=torch.bfloat16),
            torch.empty(96, 256, device="meta", dtype=torch.bfloat16),
        ],
    )

    result = ds.lookup(op)

    assert result is not None
    assert result.source == QuerySource.INTERPOLATED
    assert result.latency_us == pytest.approx(150 + 96 + 256)
    assert result.details["interpolation_dim"] == 2
    assert result.details["axes"] == ["M", "K"]
    assert result.details["method"] == "griddata_linear"
    assert result.details["interpolation_path"] == "multidim"
    assert result.shape_match_info.shape_match_rule == "interpolated_2d_griddata_linear"


def test_compute_interpolates_3d_m_k_n(matmul_grid_dir):
    ds = InterpolatingDataSource(ProfilingDataSource(matmul_grid_dir))
    op = _make_op_info(
        torch.ops.aten.mm.default,
        [
            torch.empty(150, 96, device="meta", dtype=torch.bfloat16),
            torch.empty(96, 384, device="meta", dtype=torch.bfloat16),
        ],
    )

    result = ds.lookup(op)

    assert result is not None
    assert result.source == QuerySource.INTERPOLATED
    assert result.latency_us == pytest.approx(150 + 96 + 384)
    assert result.details["interpolation_dim"] == 3
    assert result.details["axes"] == ["M", "K", "N"]
    assert result.details["method"] == "griddata_linear"
    assert result.details["interpolation_path"] == "multidim"
    assert result.shape_match_info.shape_match_rule == "interpolated_3d_griddata_linear"


def test_matmul_sparse_gemm_defaults_limit_multidim_interpolation(tmp_path):
    rows = [
        _matmul_row(100, 64, 256),
        _matmul_row(200, 64, 256),
        _matmul_row(100, 128, 256),
        _matmul_row(200, 128, 256),
    ]

    def write_case(data_dir, *, kernel_type, max_interpolation_dim=None):
        data_dir.mkdir()
        policy = ""
        if max_interpolation_dim is not None:
            policy = f"""
interpolation_policy:
  kernel_overrides:
    {kernel_type}:
      max_interpolation_dim: {max_interpolation_dim}
"""
        _write_text(
            data_dir / "op_mapping.yaml",
            f"""
version: "test"
operator_mappings:
  "aten.mm.default":
    kernel_type: {kernel_type}
{policy}
""",
        )
        _write_text(
            data_dir / f"{kernel_type}.csv",
            "\n".join(
                [
                    "Input Shapes,Input Data Types,Input Formats,Output Shapes,"
                    "Output Data Types,Output Formats,Duration(us)",
                    *rows,
                ]
            ),
        )

    op = _make_op_info(
        torch.ops.aten.mm.default,
        [
            torch.empty(150, 96, device="meta", dtype=torch.bfloat16),
            torch.empty(96, 256, device="meta", dtype=torch.bfloat16),
        ],
    )

    opt_in_dir = tmp_path / "matmul_v2_opt_in_2d"
    write_case(opt_in_dir, kernel_type="MatMulV2", max_interpolation_dim=2)
    opt_in_result = InterpolatingDataSource(ProfilingDataSource(opt_in_dir)).lookup(op)

    assert opt_in_result is not None
    assert opt_in_result.details["interpolation_dim"] == 2
    assert opt_in_result.details["method"] == "griddata_linear"

    for kernel_type in ("MatMulV2", "MatMulV3", "MatMulCommon"):
        limited_dir = tmp_path / f"{kernel_type}_1d_only"
        write_case(limited_dir, kernel_type=kernel_type, max_interpolation_dim=1)
        limited_result = InterpolatingDataSource(ProfilingDataSource(limited_dir)).lookup(op)
        assert limited_result is None


def test_compute_index_skips_invalid_latency_rows(matmul_invalid_latency_dir):
    ds = InterpolatingDataSource(ProfilingDataSource(matmul_invalid_latency_dir))
    op = _make_op_info(
        torch.ops.aten.mm.default,
        [
            torch.empty(150, 96, device="meta", dtype=torch.bfloat16),
            torch.empty(96, 384, device="meta", dtype=torch.bfloat16),
        ],
    )

    result = ds.lookup(op)

    assert result is not None
    assert result.source == QuerySource.INTERPOLATED
    assert result.latency_us == pytest.approx(150 + 96 + 384)


def test_wrapper_interpolation_records_partial_fallback_source(matmul_grid_dir, monkeypatch):
    ds = InterpolatingDataSource(ProfilingDataSource(matmul_grid_dir))
    monkeypatch.setattr(
        ds.base,
        "lookup",
        lambda _op: QueryResult(latency_us=1.0, confidence=0.1, source=QuerySource.PARTIAL),
    )
    op = _make_op_info(
        torch.ops.aten.mm.default,
        [
            torch.empty(150, 96, device="meta", dtype=torch.bfloat16),
            torch.empty(96, 384, device="meta", dtype=torch.bfloat16),
        ],
    )

    result = ds.lookup(op)

    assert result is not None
    assert result.source == QuerySource.INTERPOLATED
    assert result.details["fallback_from"] == "partial"


def test_compute_interpolates_transposed_weight_candidate(matmul_transposed_weight_dir):
    ds = InterpolatingDataSource(ProfilingDataSource(matmul_transposed_weight_dir))
    op = _make_op_info(
        torch.ops.aten.mm.default,
        [
            torch.empty(150, 96, device="meta", dtype=torch.bfloat16),
            torch.empty(96, 384, device="meta", dtype=torch.bfloat16),
        ],
    )

    result = ds.lookup(op)

    assert result is not None
    assert result.source == QuerySource.INTERPOLATED
    assert result.latency_us == pytest.approx(150 + 96 + 384)
    assert result.details["interpolation_dim"] == 3
    assert result.details["matched_row_meta"][0]["source_layout"] == "rhs_n_k"


def test_compute_does_not_cross_dtype_regime(matmul_dtype_mismatch_dir):
    ds = InterpolatingDataSource(ProfilingDataSource(matmul_dtype_mismatch_dir))
    op = _make_op_info(
        torch.ops.aten.mm.default,
        [
            torch.empty(150, 96, device="meta", dtype=torch.bfloat16),
            torch.empty(96, 384, device="meta", dtype=torch.bfloat16),
        ],
    )

    assert ds.lookup(op) is None


def test_compute_skips_rows_with_missing_input_formats(matmul_missing_input_format_dir):
    ds = InterpolatingDataSource(ProfilingDataSource(matmul_missing_input_format_dir))
    op = _make_op_info(
        torch.ops.aten.mm.default,
        [
            torch.empty(150, 96, device="meta", dtype=torch.bfloat16),
            torch.empty(96, 384, device="meta", dtype=torch.bfloat16),
        ],
    )

    assert ds.lookup(op) is None


def _batch_matmul_row(batch, m_dim, k_dim, n_dim):
    latency = float(batch * 1000 + m_dim + k_dim + n_dim)
    return (
        f'"{batch},{m_dim},{k_dim};{batch},{k_dim},{n_dim}",'
        '"DT_BF16;DT_BF16","ND;ND",'
        f'"{batch},{m_dim},{n_dim}","DT_BF16","ND",{latency}'
    )


@pytest.fixture
def batch_matmul_grid_dir(tmp_path):
    data_dir = tmp_path / "batch_matmul_grid"
    data_dir.mkdir()
    _write_text(
        data_dir / "op_mapping.yaml",
        """
version: "test"
operator_mappings:
  "aten.bmm.default":
    kernel_type: BatchMatMulV2
""",
    )
    rows = [_batch_matmul_row(2, m, k, n) for m in (100, 200) for k in (64, 128) for n in (256, 512)]
    _write_text(
        data_dir / "BatchMatMulV2.csv",
        "\n".join(
            [
                "Input Shapes,Input Data Types,Input Formats,Output Shapes,"
                "Output Data Types,Output Formats,Duration(us)",
                *rows,
            ]
        ),
    )
    return data_dir


@pytest.fixture
def batch_matmul_nd_grid_dir(tmp_path):
    data_dir = tmp_path / "batch_matmul_nd_grid"
    data_dir.mkdir()
    _write_text(
        data_dir / "op_mapping.yaml",
        """
version: "test"
operator_mappings:
  "aten.bmm.default":
    kernel_type: BatchMatMulNd
""",
    )
    rows = [_batch_matmul_row(2, m, k, n) for m in (100, 200) for k in (64, 128) for n in (256, 512)]
    _write_text(
        data_dir / "BatchMatMulNd.csv",
        "\n".join(
            [
                "Input Shapes,Input Data Types,Input Formats,Output Shapes,"
                "Output Data Types,Output Formats,Duration(us)",
                *rows,
            ]
        ),
    )
    return data_dir


def test_batch_matmul_interpolates_m_k_n_without_flattening_batch(batch_matmul_grid_dir):
    ds = InterpolatingDataSource(ProfilingDataSource(batch_matmul_grid_dir))
    op = _make_op_info(
        torch.ops.aten.bmm.default,
        [
            torch.empty(2, 150, 96, device="meta", dtype=torch.bfloat16),
            torch.empty(2, 96, 384, device="meta", dtype=torch.bfloat16),
        ],
    )

    result = ds.lookup(op)

    assert result is not None
    assert result.source == QuerySource.INTERPOLATED
    assert result.latency_us == pytest.approx(2 * 1000 + 150 + 96 + 384)
    assert result.details["interpolation_dim"] == 3
    assert result.details["axes"] == ["M", "K", "N"]


def test_batch_matmul_nd_interpolates_m_k_n_without_flattening_batch(batch_matmul_nd_grid_dir):
    ds = InterpolatingDataSource(ProfilingDataSource(batch_matmul_nd_grid_dir))
    op = _make_op_info(
        torch.ops.aten.bmm.default,
        [
            torch.empty(2, 150, 96, device="meta", dtype=torch.bfloat16),
            torch.empty(2, 96, 384, device="meta", dtype=torch.bfloat16),
        ],
    )

    result = ds.lookup(op)

    assert result is not None
    assert result.source == QuerySource.INTERPOLATED
    assert result.latency_us == pytest.approx(2 * 1000 + 150 + 96 + 384)
    assert result.details["interpolation_dim"] == 3
    assert result.details["axes"] == ["M", "K", "N"]


def test_batch_matmul_batch_one_keeps_batch_regime(tmp_path):
    data_dir = tmp_path / "batch_matmul_one"
    data_dir.mkdir()
    _write_text(
        data_dir / "op_mapping.yaml",
        """
version: "test"
operator_mappings:
  "aten.bmm.default":
    kernel_type: BatchMatMulV2
""",
    )
    rows = [_batch_matmul_row(1, m, k, n) for m in (100, 200) for k in (64, 128) for n in (256, 512)]
    _write_text(
        data_dir / "BatchMatMulV2.csv",
        "\n".join(
            [
                "Input Shapes,Input Data Types,Input Formats,Output Shapes,"
                "Output Data Types,Output Formats,Duration(us)",
                *rows,
            ]
        ),
    )
    ds = InterpolatingDataSource(ProfilingDataSource(data_dir))
    op = _make_op_info(
        torch.ops.aten.bmm.default,
        [
            torch.empty(1, 150, 96, device="meta", dtype=torch.bfloat16),
            torch.empty(1, 96, 384, device="meta", dtype=torch.bfloat16),
        ],
    )

    result = ds.lookup(op)

    assert result is not None
    assert result.source == QuerySource.INTERPOLATED
    assert result.latency_us == pytest.approx(1 * 1000 + 150 + 96 + 384)
    assert dict(result.details["exact_fields"])["batch_dims"] == [[1], [1]]


def test_compute_does_not_extrapolate(matmul_grid_dir):
    ds = InterpolatingDataSource(ProfilingDataSource(matmul_grid_dir))
    op = _make_op_info(
        torch.ops.aten.mm.default,
        [
            torch.empty(50, 96, device="meta", dtype=torch.bfloat16),
            torch.empty(96, 384, device="meta", dtype=torch.bfloat16),
        ],
    )

    assert ds.lookup(op) is None


def test_compute_target_does_not_cross_input_format_regime(tmp_path):
    data_dir = tmp_path / "matmul_mixed_formats"
    data_dir.mkdir()
    _write_text(
        data_dir / "op_mapping.yaml",
        """
version: "test"
operator_mappings:
  "aten.mm.default":
    kernel_type: MatMulV2
""",
    )
    rows = [
        '"8,4,16,16;4,16,16,16","DT_BF16;DT_BF16","FRACTAL_NZ;FRACTAL_NZ","128,256","DT_BF16","ND",900.0',
        '"10,4,16,16;4,16,16,16","DT_BF16;DT_BF16","FRACTAL_NZ;FRACTAL_NZ","160,256","DT_BF16","ND",950.0',
        '"128,64;64,256","DT_BF16;DT_BF16","ND;ND","128,256","DT_BF16","ND",100.0',
        '"160,64;64,256","DT_BF16;DT_BF16","ND;ND","160,256","DT_BF16","ND",200.0',
    ]
    _write_text(
        data_dir / "MatMulV2.csv",
        "\n".join(
            [
                "Input Shapes,Input Data Types,Input Formats,Output Shapes,"
                "Output Data Types,Output Formats,Duration(us)",
                *rows,
            ]
        ),
    )
    ds = InterpolatingDataSource(ProfilingDataSource(data_dir))
    op = _make_op_info(
        torch.ops.aten.mm.default,
        [
            torch.empty(144, 64, device="meta", dtype=torch.bfloat16),
            torch.empty(64, 256, device="meta", dtype=torch.bfloat16),
        ],
    )

    result = ds.lookup(op)

    assert result is not None
    assert result.source == QuerySource.INTERPOLATED
    assert result.latency_us == pytest.approx(150.0)
    assert dict(result.details["exact_fields"])["input_formats"] == ["ND", "ND"]


def test_compute_3d_does_not_extrapolate(matmul_grid_dir):
    ds = InterpolatingDataSource(ProfilingDataSource(matmul_grid_dir))
    op = _make_op_info(
        torch.ops.aten.mm.default,
        [
            torch.empty(150, 160, device="meta", dtype=torch.bfloat16),
            torch.empty(160, 384, device="meta", dtype=torch.bfloat16),
        ],
    )

    assert ds.lookup(op) is None


def _attention_row(seq, batch, heads, head_dim, latency):
    q_shape = f"2,{heads},{head_dim}"
    key_shape = f"16,128,{heads},{head_dim}"
    return (
        f'"{q_shape};{key_shape};{key_shape}",'
        '"DT_BF16;DT_BF16;DT_BF16","ND;ND;ND",'
        f'"{q_shape}","DT_BF16","ND",{latency},'
        f"{seq},{batch},3,{heads},TND"
    )


def _attention_row_quant(seq, batch, heads, head_dim, latency, quant_mode):
    return f"{_attention_row(seq, batch, heads, head_dim, latency)},{quant_mode}"


def _attention_row_with_layout(seq, batch, heads, head_dim, latency, input_layout):
    return f"{_attention_row(seq, batch, heads, head_dim, latency).removesuffix(',TND')},{input_layout}"


def _attention_row_bnsd(seq, batch, heads, head_dim, latency):
    q_shape = f"{batch},{heads},1,{head_dim}"
    key_shape = f"{batch},{heads},16,{head_dim}"
    return (
        f'"{q_shape};{key_shape};{key_shape}",'
        '"DT_BF16;DT_BF16;DT_BF16","ND;ND;ND",'
        f'"{q_shape}","DT_BF16","ND",{latency},'
        f"{seq},{batch},0,{heads},BNSD_NBSD"
    )


def _attention_row_with_q_tokens(q_tokens, seq, batch, heads, head_dim, latency, sparse_mode=3, kv_heads=None):
    kv_heads_value = heads if kv_heads is None else kv_heads
    q_shape = f"{q_tokens},{heads},{head_dim}"
    key_shape = f"16,128,{heads},{head_dim}"
    return (
        f'"{q_shape};{key_shape};{key_shape}",'
        '"DT_BF16;DT_BF16;DT_BF16","ND;ND;ND",'
        f'"{q_shape}","DT_BF16","ND",{latency},'
        f"{seq},{batch},{sparse_mode},{kv_heads_value},TND"
    )


def _attention_row_without_batch(seq, heads, head_dim, latency, sparse_mode=3, kv_heads=None):
    kv_heads = heads if kv_heads is None else kv_heads
    q_shape = f"2,{heads},{head_dim}"
    key_shape = f"16,128,{heads},{head_dim}"
    return (
        f'"{q_shape};{key_shape};{key_shape}",'
        '"DT_BF16;DT_BF16;DT_BF16","ND;ND;ND",'
        f'"{q_shape}","DT_BF16","ND",{latency},'
        f"{seq},{sparse_mode},{kv_heads},TND"
    )


@pytest.fixture
def attention_2d_dir(tmp_path):
    data_dir = tmp_path / "attention_2d"
    data_dir.mkdir()
    _write_text(
        data_dir / "op_mapping.yaml",
        """
version: "test"
operator_mappings:
  "tensor_cast.attention.default":
    kernel_type: FusedInferAttentionScore
    query_mode: attention_special
""",
    )
    rows = [_attention_row(seq, batch, 4, 128, seq / 10 + batch * 10) for seq in (1000, 2000) for batch in (1, 3)]
    _write_text(
        data_dir / "FusedInferAttentionScore.csv",
        "\n".join(
            [
                "Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,"
                "Duration(us),Runtime avg_seq_len,Runtime batch_size,Runtime sparse_mode,"
                "Runtime num_key_value_heads,Runtime input_layout",
                *rows,
            ]
        ),
    )
    return data_dir


@pytest.fixture
def attention_invalid_latency_dir(tmp_path):
    data_dir = tmp_path / "attention_invalid_latency"
    data_dir.mkdir()
    _write_text(
        data_dir / "op_mapping.yaml",
        """
version: "test"
operator_mappings:
  "tensor_cast.attention.default":
    kernel_type: FusedInferAttentionScore
    query_mode: attention_special
""",
    )
    rows = [
        _attention_row(1250, 2, 4, 128, "bad_latency"),
        *[_attention_row(seq, batch, 4, 128, seq / 10 + batch * 10) for seq in (1000, 2000) for batch in (1, 3)],
    ]
    _write_text(
        data_dir / "FusedInferAttentionScore.csv",
        "\n".join(
            [
                "Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,"
                "Duration(us),Runtime avg_seq_len,Runtime batch_size,Runtime sparse_mode,"
                "Runtime num_key_value_heads,Runtime input_layout",
                *rows,
            ]
        ),
    )
    return data_dir


@pytest.fixture
def attention_quant_mode_dir(tmp_path):
    data_dir = tmp_path / "attention_quant_mode"
    data_dir.mkdir()
    _write_text(
        data_dir / "op_mapping.yaml",
        """
version: "test"
operator_mappings:
  "tensor_cast.attention.default":
    kernel_type: FusedInferAttentionScore
    query_mode: attention_special
    quant_mode: W8A8
""",
    )
    rows = [
        _attention_row_quant(seq, batch, 4, 128, seq / 10 + batch * 10, "W8A8")
        for seq in (1000, 2000)
        for batch in (1, 3)
    ]
    _write_text(
        data_dir / "FusedInferAttentionScore.csv",
        "\n".join(
            [
                "Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,"
                "Duration(us),Runtime avg_seq_len,Runtime batch_size,Runtime sparse_mode,"
                "Runtime num_key_value_heads,Runtime input_layout,Runtime quant_mode",
                *rows,
            ]
        ),
    )
    return data_dir


def test_attention_quant_without_csv_quant_column_can_interpolate(tmp_path):
    data_dir = tmp_path / "attention_quant_plain_fia"
    data_dir.mkdir()
    _write_text(
        data_dir / "op_mapping.yaml",
        """
version: "test"
operator_mappings:
  "tensor_cast.attention_quant.default":
    kernel_type: FusedInferAttentionScore
    query_mode: attention_special
""",
    )
    rows = [_attention_row(seq, 2, 4, 128, seq / 10 + 20) for seq in (1000, 2000)]
    _write_text(
        data_dir / "FusedInferAttentionScore.csv",
        "\n".join(
            [
                "Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,"
                "Duration(us),Runtime avg_seq_len,Runtime batch_size,Runtime sparse_mode,"
                "Runtime num_key_value_heads,Runtime input_layout",
                *rows,
            ]
        ),
    )
    ds = InterpolatingDataSource(ProfilingDataSource(data_dir))
    op = _make_op_info(
        torch.ops.tensor_cast.attention_quant.default,
        [
            torch.empty(2, 4, 128, device="meta", dtype=torch.bfloat16),
            torch.empty(16, 128, 4, 128, device="meta", dtype=torch.bfloat16),
            torch.empty(16, 128, 4, 128, device="meta", dtype=torch.bfloat16),
            None,
            None,
            None,
            torch.tensor([1500, 1500], dtype=torch.int64),
            torch.tensor([1, 1], dtype=torch.int64),
            1.0,
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            torch.bfloat16,
        ],
    )

    result = ds.lookup(op)

    assert result is not None
    assert result.source == QuerySource.INTERPOLATED
    assert result.latency_us == pytest.approx(170.0)


def test_attention_quant_target_rejects_blank_quant_candidates_when_csv_has_quant_column(tmp_path):
    data_dir = tmp_path / "attention_quant_required_field"
    data_dir.mkdir()
    _write_text(
        data_dir / "op_mapping.yaml",
        """
version: "test"
operator_mappings:
  "tensor_cast.attention.default":
    kernel_type: FusedInferAttentionScore
    query_mode: attention_special
    quant_mode: W8A8
""",
    )
    rows = [
        f"{_attention_row(1000, 2, 4, 128, 100.0)},W8A8",
        f"{_attention_row(2000, 2, 4, 128, 200.0)},W8A8",
        f"{_attention_row(1000, 2, 4, 128, 900.0)},",
        f"{_attention_row(2000, 2, 4, 128, 1200.0)},",
    ]
    _write_text(
        data_dir / "FusedInferAttentionScore.csv",
        "\n".join(
            [
                "Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,"
                "Duration(us),Runtime avg_seq_len,Runtime batch_size,Runtime sparse_mode,"
                "Runtime num_key_value_heads,Runtime input_layout,Runtime quant_mode",
                *rows,
            ]
        ),
    )
    ds = InterpolatingDataSource(ProfilingDataSource(data_dir))

    result = ds.lookup(_attention_op())

    assert result is not None
    assert result.latency_us == pytest.approx(150.0)
    assert result.details["exact_fields"]["quant_mode"] == "W8A8"


def test_attention_empty_quant_mode_cell_is_not_string_nan(tmp_path):
    data_dir = tmp_path / "attention_empty_quant_mode"
    data_dir.mkdir()
    _write_text(
        data_dir / "op_mapping.yaml",
        """
version: "test"
operator_mappings:
  "tensor_cast.attention.default":
    kernel_type: FusedInferAttentionScore
    query_mode: attention_special
""",
    )
    _write_text(
        data_dir / "FusedInferAttentionScore.csv",
        "\n".join(
            [
                "Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,"
                "Duration(us),Runtime avg_seq_len,Runtime batch_size,Runtime sparse_mode,"
                "Runtime num_key_value_heads,Runtime input_layout,Runtime quant_mode",
                f"{_attention_row(1500, 2, 4, 128, 123.0)},",
            ]
        ),
    )
    op = _make_op_info(
        torch.ops.tensor_cast.attention.default,
        [
            torch.empty(2, 4, 128, device="meta", dtype=torch.bfloat16),
            torch.empty(16, 128, 4, 128, device="meta", dtype=torch.bfloat16),
            torch.empty(16, 128, 4, 128, device="meta", dtype=torch.bfloat16),
            None,
            None,
            None,
            torch.tensor([1500, 1500], dtype=torch.int64),
            torch.tensor([1, 1], dtype=torch.int64),
        ],
    )

    result = ProfilingDataSource(data_dir).lookup(op)
    assert result is not None
    assert result.source == QuerySource.MEASURED
    assert result.latency_us == 123.0


def test_attention_by_params_filters_runtime_input_layout(tmp_path):
    data_dir = tmp_path / "attention_layout_filter"
    data_dir.mkdir()
    _write_text(
        data_dir / "op_mapping.yaml",
        """
version: "test"
operator_mappings: {}
""",
    )
    rows = [
        _attention_row(1000, 2, 4, 128, 10.0),
        _attention_row(2000, 2, 4, 128, 20.0),
        _attention_row_bnsd(1000, 2, 4, 128, 100.0),
        _attention_row_bnsd(2000, 2, 4, 128, 300.0),
    ]
    _write_text(
        data_dir / "FusedInferAttentionScore.csv",
        "\n".join(
            [
                "Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,"
                "Duration(us),Runtime avg_seq_len,Runtime batch_size,Runtime sparse_mode,"
                "Runtime num_key_value_heads,Runtime input_layout",
                *rows,
            ]
        ),
    )
    ds = InterpolatingDataSource(ProfilingDataSource(data_dir))

    result = ds._interpolate_attention_by_params(
        "FusedInferAttentionScore",
        {
            "q_shape_3d": (2, 4, 128),
            "avg_seq_len": 1500,
            "batch_size": 2,
            "sparse_mode": 0,
            "num_kv_heads": 4,
            "input_layout": "BNSD_NBSD",
        },
        "DT_BF16",
    )

    assert result is not None
    assert result.latency_us == pytest.approx(200.0)


def test_attention_by_params_without_layout_rejects_layout_csv(tmp_path):
    data_dir = tmp_path / "attention_layout_required"
    data_dir.mkdir()
    _write_text(
        data_dir / "op_mapping.yaml",
        """
version: "test"
operator_mappings: {}
""",
    )
    _write_text(
        data_dir / "FusedInferAttentionScore.csv",
        "\n".join(
            [
                "Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,"
                "Duration(us),Runtime avg_seq_len,Runtime batch_size,Runtime sparse_mode,"
                "Runtime num_key_value_heads,Runtime input_layout",
                _attention_row(1000, 2, 4, 128, 10.0),
                _attention_row(2000, 2, 4, 128, 20.0),
            ]
        ),
    )
    ds = InterpolatingDataSource(ProfilingDataSource(data_dir))

    result = ds._interpolate_attention_by_params(
        "FusedInferAttentionScore",
        {
            "q_shape_3d": (2, 4, 128),
            "avg_seq_len": 1500,
            "num_kv_heads": 4,
        },
        "DT_BF16",
    )

    assert result is None
    assert ds.last_miss_reason == "attention_input_layout_unavailable"


def test_attention_matching_fields_keep_layout_boundary_when_target_has_layout(tmp_path):
    data_dir = tmp_path / "attention_matching_fields"
    data_dir.mkdir()
    _write_text(data_dir / "op_mapping.yaml", 'version: "test"')
    ds = InterpolatingDataSource(ProfilingDataSource(data_dir))
    base_fields = [
        ("kernel_type", "FusedInferAttentionScore"),
        ("dtype", "DT_BF16"),
        ("sparse_mode", 3),
        ("kv_heads", 4),
    ]
    target_with_layout = InterpolationTarget(
        func_name="attention",
        kernel_type="FusedInferAttentionScore",
        axes={"seq": 1500.0},
        regime_key=make_regime_key([*base_fields, ("input_layout", "TND")]),
    )
    target_without_layout = InterpolationTarget(
        func_name="attention",
        kernel_type="FusedInferAttentionScore",
        axes={"seq": 1500.0},
        regime_key=make_regime_key(base_fields),
    )

    required_with_layout, allow_extra_with_layout = ds._attention_matching_fields(
        "FusedInferAttentionScore",
        target_with_layout,
    )
    required_without_layout, allow_extra_without_layout = ds._attention_matching_fields(
        "FusedInferAttentionScore",
        target_without_layout,
    )

    assert {"sparse_mode", "kv_heads", "input_layout"}.issubset(required_with_layout)
    assert "input_layout" not in allow_extra_with_layout
    assert {"sparse_mode", "kv_heads"}.issubset(required_without_layout)
    assert "input_layout" not in allow_extra_without_layout


def test_compute_by_shapes_fallback_tries_alternate_kernel_types(tmp_path):
    data_dir = tmp_path / "compute_alternate_fallback"
    data_dir.mkdir()
    _write_text(
        data_dir / "op_mapping.yaml",
        """
version: "test"
operator_mappings: {}
""",
    )
    _write_text(
        data_dir / "BatchMatMulNd.csv",
        "\n".join(
            [
                "Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,"
                "Duration(us)",
                '"8,128;128,512","DT_BF16;DT_BF16","ND;ND","8,512","DT_BF16","ND",8.0',
                '"24,128;128,512","DT_BF16;DT_BF16","ND;ND","24,512","DT_BF16","ND",24.0',
            ]
        ),
    )
    ds = InterpolatingDataSource(ProfilingDataSource(data_dir))

    result = ds._interpolate_compute_by_shapes(
        ["BatchMatMulV2", "BatchMatMulNd"],
        [(16, 128), (128, 512)],
        "DT_BF16",
    )

    assert result is not None
    assert result.latency_us == pytest.approx(16.0)


def test_generic_compute_interpolation_tries_alternate_kernel_types(tmp_path):
    data_dir = tmp_path / "generic_compute_alternate"
    data_dir.mkdir()
    _write_text(
        data_dir / "op_mapping.yaml",
        """
version: "test"
operator_mappings:
  "tensor_cast.fake_generic.default":
    kernel_type: GatherV3
    alternate_kernel_types:
      - GatherV2
""",
    )
    _write_text(
        data_dir / "GatherV2.csv",
        """
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Duration(us)
"100,64","DT_BF16","ND","100,64","DT_BF16","ND",10.0
"200,64","DT_BF16","ND","200,64","DT_BF16","ND",20.0
""",
    )
    ds = InterpolatingDataSource(ProfilingDataSource(data_dir))
    op = _make_op_info(
        "tensor_cast.fake_generic.default",
        [torch.empty(150, 64, device="meta", dtype=torch.bfloat16)],
    )
    mapping = ds.base._op_mapping["operator_mappings"]["tensor_cast.fake_generic.default"]

    result = ds._interpolate_compute(op, mapping)

    assert result is not None
    assert result.latency_us == pytest.approx(15.0)
    assert result.details["kernel_type"] == "GatherV2"


def test_generic_compute_target_index_and_interpolation_use_candidate_group(tmp_path):
    data_dir = tmp_path / "generic_compute_1d"
    data_dir.mkdir()
    _write_text(
        data_dir / "op_mapping.yaml",
        """
version: "test"
operator_mappings:
  "tensor_cast.rms_norm.default":
    kernel_type: RmsNorm
""",
    )
    _write_text(
        data_dir / "RmsNorm.csv",
        """
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Duration(us)
"100,64","DT_BF16","ND","100,64","DT_BF16","ND",10.0
"200,64","DT_BF16","ND","200,64","DT_BF16","ND",20.0
""",
    )
    ds = InterpolatingDataSource(ProfilingDataSource(data_dir))
    op = _make_op_info(
        torch.ops.tensor_cast.rms_norm.default,
        [torch.empty(150, 64, device="meta", dtype=torch.bfloat16)],
    )
    mapping = {"kernel_type": "RmsNorm"}

    assert ds._generic_compute_shape_signature([]) == ()
    assert ds._generic_compute_shape_signature([(150, 64), (64,)]) == ((64,), ((64,),))

    target_from_shapes = ds._build_generic_compute_target_from_shapes(
        "RmsNorm",
        [(150, 64)],
        "DT_BF16",
        func_name="tensor_cast.rms_norm.default",
    )
    target_from_op = ds._build_generic_compute_target(op, mapping, "RmsNorm")
    assert target_from_shapes is not None
    assert target_from_op is not None
    assert target_from_op.axes == {"axis_0": 150.0}
    assert target_from_op.regime_key == target_from_shapes.regime_key
    direct_axes_and_regime = ds._generic_compute_axes_and_regime("RmsNorm", [(150, 64)])
    assert direct_axes_and_regime is not None
    direct_axes, direct_extra_regime = direct_axes_and_regime
    assert direct_axes == {"axis_0": 150.0}
    assert direct_extra_regime == [("shape_signature", ((64,), ()))]

    df = ds.base._load_csv("RmsNorm")
    assert df is not None
    point, reason = ds._candidate_from_generic_compute_row_with_reason(
        df.iloc[0],
        "RmsNorm",
        ds.base._latency_col(df),
        0,
        None,
    )
    assert reason is None
    assert point is not None
    assert point.axes == {"axis_0": 100.0}

    index = ds._get_generic_compute_index("RmsNorm", None)
    assert index is not None
    assert ds._get_generic_compute_index("RmsNorm", None) is index

    result = ds._interpolate_generic_compute_target(
        target_from_op,
        None,
        fallback_from="exact_miss",
        interpolation_path="compute_1d",
    )
    assert result is not None
    assert result.source == QuerySource.INTERPOLATED
    assert result.latency_us == pytest.approx(15.0)
    assert result.details["interpolation_path"] == "compute_1d"


def test_generic_compute_output_numel_axis_policy_uses_output_tail_regime(tmp_path):
    data_dir = tmp_path / "generic_compute_output_numel"
    data_dir.mkdir()
    _write_text(
        data_dir / "op_mapping.yaml",
        """
version: "test"
interpolation_policy:
  kernel_overrides:
    GatherV2:
      generic_compute:
        axis: output_numel
operator_mappings:
  "aten.embedding.default":
    kernel_type: GatherV2
    tc_input_count: 2
""",
    )
    _write_text(
        data_dir / "GatherV2.csv",
        """
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Duration(us)
"1000,64;10","DT_BF16;INT64","ND;ND","10,64","DT_BF16","ND",10.0
"1000,64;30","DT_BF16;INT64","ND;ND","30,64","DT_BF16","ND",30.0
"1000,64;20","DT_BF16;INT64","ND;ND","20,32","DT_BF16","ND",999.0
""",
    )
    ds = InterpolatingDataSource(ProfilingDataSource(data_dir))
    op = _make_op_info(
        torch.ops.aten.embedding.default,
        [
            torch.empty(1000, 64, device="meta", dtype=torch.bfloat16),
            torch.empty(20, device="meta", dtype=torch.int64),
        ],
    )
    op.out = torch.empty(20, 64, device="meta", dtype=torch.bfloat16)
    mapping = {"kernel_type": "GatherV2", "tc_input_count": 2}

    target = ds._build_generic_compute_target(op, mapping, "GatherV2")
    assert target is not None
    assert target.axes == {"output_numel": 1280.0}
    target_regime = dict(target.regime_key)
    assert target_regime["output_tail_shape"] == (64,)
    assert "shape_signature" not in target_regime

    index = ds._get_generic_compute_index("GatherV2", 2)
    assert index is not None
    groups = index.candidate_groups_matching(target.regime_key)
    assert len(groups) == 1
    assert [point.latency_us for point in groups[0].points] == [10.0, 30.0]

    result = ds._interpolate_generic_compute_target(
        target,
        2,
        fallback_from="exact_miss",
        interpolation_path="compute_1d",
    )
    assert result is not None
    assert result.latency_us == pytest.approx(20.0)
    assert result.details["axes"] == ["output_numel"]


def test_generic_compute_output_numel_keeps_singleton_output_tail_shape(tmp_path):
    data_dir = tmp_path / "generic_compute_output_numel_singleton"
    data_dir.mkdir()
    _write_text(
        data_dir / "op_mapping.yaml",
        """
version: "test"
interpolation_policy:
  kernel_overrides:
    GatherV2:
      generic_compute:
        axis: output_numel
operator_mappings:
  "aten.embedding.default":
    kernel_type: GatherV2
    tc_input_count: 2
""",
    )
    _write_text(
        data_dir / "GatherV2.csv",
        """
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Duration(us)
"1000,64;1","DT_BF16;INT64","ND;ND","1,64","DT_BF16","ND",10.0
"1000,64;4","DT_BF16;INT64","ND;ND","4,64","DT_BF16","ND",40.0
"1000,32;4","DT_BF16;INT64","ND;ND","4,32","DT_BF16","ND",999.0
""",
    )
    ds = InterpolatingDataSource(ProfilingDataSource(data_dir))
    op = _make_op_info(
        torch.ops.aten.embedding.default,
        [
            torch.empty(1000, 64, device="meta", dtype=torch.bfloat16),
            torch.empty(2, device="meta", dtype=torch.int64),
        ],
    )
    op.out = torch.empty(2, 64, device="meta", dtype=torch.bfloat16)
    mapping = {"kernel_type": "GatherV2", "tc_input_count": 2}

    target = ds._build_generic_compute_target(op, mapping, "GatherV2")
    assert target is not None
    assert target.axes == {"output_numel": 128.0}
    assert dict(target.regime_key)["output_tail_shape"] == (64,)

    index = ds._get_generic_compute_index("GatherV2", 2)
    assert index is not None
    groups = index.candidate_groups_matching(target.regime_key)
    assert len(groups) == 1
    assert [point.axes["output_numel"] for point in groups[0].points] == [64.0, 256.0]

    result = ds._interpolate_generic_compute_target(
        target,
        2,
        fallback_from="exact_miss",
        interpolation_path="compute_1d",
    )

    assert result is not None
    assert result.latency_us == pytest.approx(20.0)


def test_generic_compute_output_numel_rejects_multi_output_shapes(tmp_path):
    data_dir = tmp_path / "generic_compute_output_numel_multi_output"
    data_dir.mkdir()
    _write_text(
        data_dir / "op_mapping.yaml",
        """
version: "test"
interpolation_policy:
  kernel_overrides:
    GatherV2:
      generic_compute:
        axis: output_numel
operator_mappings:
  "aten.embedding.default":
    kernel_type: GatherV2
    tc_input_count: 2
""",
    )
    _write_text(
        data_dir / "GatherV2.csv",
        """
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Duration(us)
"1000,64;10","DT_BF16;INT64","ND;ND","10,64;10,1","DT_BF16;DT_BF16","ND;ND",10.0
"1000,64;10","DT_BF16;INT64","ND;ND","10,64;10,4","DT_BF16;DT_BF16","ND;ND",40.0
""",
    )
    ds = InterpolatingDataSource(ProfilingDataSource(data_dir))

    target, reason = ds._build_generic_compute_target_from_shapes_with_reason(
        "GatherV2",
        [(1000, 64), (10,)],
        "DT_BF16",
        dtype_values=["DT_BF16", "INT64"],
        output_shapes=[(10, 64), (10, 2)],
        tc_input_count=2,
    )
    assert target is None
    assert reason == "generic_compute_output_numel_multi_output_unsupported"

    index = ds._get_generic_compute_index("GatherV2", 2)
    assert index is not None
    assert index.candidate_groups_matching(make_regime_key([("kernel_type", "GatherV2")])) == []
    assert ds._compute_index_diagnostics["GatherV2"]["rejected_reasons"] == {
        "generic_compute_output_numel_multi_output_unsupported": 2
    }


def test_generic_compute_output_numel_accepts_scalar_output_shape(tmp_path):
    data_dir = tmp_path / "generic_compute_output_numel_scalar"
    data_dir.mkdir()
    _write_text(
        data_dir / "op_mapping.yaml",
        """
version: "test"
interpolation_policy:
  kernel_overrides:
    GatherV2:
      generic_compute:
        axis: output_numel
operator_mappings:
  "aten.embedding.default":
    kernel_type: GatherV2
    tc_input_count: 1
""",
    )
    ds = InterpolatingDataSource(ProfilingDataSource(data_dir))

    axes_and_regime, reason = ds._generic_compute_axes_and_regime_with_reason(
        "GatherV2",
        logical_shapes=[(1,)],
        output_shapes=[()],
    )

    assert reason is None
    assert axes_and_regime is not None
    axes, extra_regime = axes_and_regime
    assert axes == {"output_numel": 1.0}
    assert extra_regime == [("output_tail_shape", ())]


def test_generic_compute_alternate_kernel_inherits_primary_output_numel_policy(tmp_path):
    data_dir = tmp_path / "generic_compute_alternate_output_numel"
    data_dir.mkdir()
    _write_text(
        data_dir / "op_mapping.yaml",
        """
version: "test"
interpolation_policy:
  kernel_overrides:
    GatherV2:
      generic_compute:
        axis: output_numel
operator_mappings:
  "aten.embedding.default":
    kernel_type: GatherV2
    alternate_kernel_types:
      - GatherV2AiCore
    tc_input_count: 2
""",
    )
    _write_text(
        data_dir / "GatherV2AiCore.csv",
        """
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Duration(us)
"1000,64;10","DT_BF16;INT64","ND;ND","10,64","DT_BF16","ND",10.0
"1000,64;30","DT_BF16;INT64","ND;ND","30,64","DT_BF16","ND",30.0
""",
    )
    ds = InterpolatingDataSource(ProfilingDataSource(data_dir))
    op = _make_op_info(
        torch.ops.aten.embedding.default,
        [
            torch.empty(1000, 64, device="meta", dtype=torch.bfloat16),
            torch.empty(20, device="meta", dtype=torch.int64),
        ],
    )
    op.out = torch.empty(20, 64, device="meta", dtype=torch.bfloat16)
    mapping = ds.base._op_mapping["operator_mappings"]["aten.embedding.default"]

    result = ds._interpolate_compute(op, mapping)

    assert result is not None
    assert result.details["kernel_type"] == "GatherV2AiCore"
    assert result.details["axes"] == ["output_numel"]
    assert result.latency_us == pytest.approx(20.0)


def test_generic_compute_max_interpolation_dim_zero_disables_interpolation(tmp_path):
    data_dir = tmp_path / "generic_compute_disabled"
    data_dir.mkdir()
    _write_text(
        data_dir / "op_mapping.yaml",
        """
version: "test"
interpolation_policy:
  kernel_overrides:
    Neg:
      max_interpolation_dim: 0
operator_mappings:
  "profiling.Neg":
    kernel_type: Neg
""",
    )
    _write_text(
        data_dir / "Neg.csv",
        """
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Duration(us)
"100,64","FLOAT","ND","100,64","FLOAT","ND",10.0
"200,64","FLOAT","ND","200,64","FLOAT","ND",20.0
""",
    )
    ds = InterpolatingDataSource(ProfilingDataSource(data_dir))
    op = _make_op_info(
        torch.ops.aten.neg.default,
        [torch.empty(150, 64, device="meta", dtype=torch.float32)],
    )
    mapping = {"kernel_type": "Neg"}

    target = ds._build_generic_compute_target(op, mapping, "Neg")
    assert target is not None
    result = ds._interpolate_generic_compute_target(
        target,
        None,
        fallback_from="exact_miss",
        interpolation_path="compute_1d",
    )
    assert result is None
    assert ds.last_miss_reason == "interpolation_dim_disabled"


def test_compute_interpolation_uses_source_pure_selected_candidates(tmp_path, monkeypatch):
    key = make_regime_key(
        {
            "kernel_type": "MatMulV2",
            "input_count": 2,
            "input_dtypes": ("DT_BF16", "DT_BF16"),
        }
    )
    points = [
        CandidatePoint(
            "MatMulV2",
            {"M": 100.0, "K": 512.0, "N": 1024.0},
            10.0,
            key,
            row_meta={"latency_selection": "selected_column"},
        ),
        CandidatePoint(
            "MatMulV2",
            {"M": 200.0, "K": 512.0, "N": 1024.0},
            20.0,
            key,
            row_meta={"latency_selection": "selected_column"},
        ),
        CandidatePoint(
            "MatMulV2",
            {"M": 100.0, "K": 512.0, "N": 1024.0},
            1000.0,
            key,
            row_meta={"latency_selection": "fallback_column"},
        ),
        CandidatePoint(
            "MatMulV2",
            {"M": 200.0, "K": 512.0, "N": 1024.0},
            2000.0,
            key,
            row_meta={"latency_selection": "fallback_column"},
        ),
    ]
    ds = InterpolatingDataSource(ProfilingDataSource(tmp_path))
    monkeypatch.setattr(ds, "_get_compute_index", lambda *_args: CandidateIndex(points))
    target = InterpolationTarget(
        func_name="aten.mm.default",
        kernel_type="MatMulV2",
        axes={"M": 150.0, "K": 512.0, "N": 1024.0},
        regime_key=key,
        query_mode="compute",
    )

    result = ds._interpolate_compute_target(
        target,
        None,
        fallback_from="exact_miss",
        interpolation_path="multidim",
    )

    assert result is not None
    assert result.latency_us == pytest.approx(15.0)
    assert result.details["latency_source_attempt"] == "selected_only"


def test_compute_interpolation_falls_back_to_source_pure_fallback_candidates(tmp_path, monkeypatch):
    key = make_regime_key(
        {
            "kernel_type": "MatMulV2",
            "input_count": 2,
            "input_dtypes": ("DT_BF16", "DT_BF16"),
        }
    )
    points = [
        CandidatePoint(
            "MatMulV2",
            {"M": 100.0, "K": 512.0, "N": 1024.0},
            10.0,
            key,
            row_meta={"latency_selection": "selected_column"},
        ),
        CandidatePoint(
            "MatMulV2",
            {"M": 100.0, "K": 512.0, "N": 1024.0},
            100.0,
            key,
            row_meta={"latency_selection": "fallback_column"},
        ),
        CandidatePoint(
            "MatMulV2",
            {"M": 200.0, "K": 512.0, "N": 1024.0},
            200.0,
            key,
            row_meta={"latency_selection": "fallback_column"},
        ),
    ]
    ds = InterpolatingDataSource(ProfilingDataSource(tmp_path))
    monkeypatch.setattr(ds, "_get_compute_index", lambda *_args: CandidateIndex(points))
    target = InterpolationTarget(
        func_name="aten.mm.default",
        kernel_type="MatMulV2",
        axes={"M": 150.0, "K": 512.0, "N": 1024.0},
        regime_key=key,
        query_mode="compute",
    )

    result = ds._interpolate_compute_target(
        target,
        None,
        fallback_from="exact_miss",
        interpolation_path="multidim",
    )

    assert result is not None
    assert result.latency_us == pytest.approx(150.0)
    assert result.details["latency_source_attempt"] == "fallback_only"


@pytest.fixture
def attention_no_batch_dir(tmp_path):
    data_dir = tmp_path / "attention_no_batch"
    data_dir.mkdir()
    _write_text(
        data_dir / "op_mapping.yaml",
        """
version: "test"
operator_mappings:
  "tensor_cast.attention.default":
    kernel_type: FusedInferAttentionScore
    query_mode: attention_special
""",
    )
    rows = [
        _attention_row_without_batch(seq, 4, head_dim, seq / 10 + head_dim)
        for seq in (1000, 2000)
        for head_dim in (64, 128)
    ]
    _write_text(
        data_dir / "FusedInferAttentionScore.csv",
        "\n".join(
            [
                "Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,"
                "Duration(us),Runtime avg_seq_len,Runtime sparse_mode,Runtime num_key_value_heads,"
                "Runtime input_layout",
                *rows,
            ]
        ),
    )
    return data_dir


@pytest.fixture
def attention_bnsd_dir(tmp_path):
    data_dir = tmp_path / "attention_bnsd"
    data_dir.mkdir()
    _write_text(
        data_dir / "op_mapping.yaml",
        """
version: "test"
operator_mappings:
  "tensor_cast.attention.default":
    kernel_type: FusedInferAttentionScore
    query_mode: attention_special
""",
    )
    rows = [_attention_row_bnsd(seq, 2, 4, 128, seq / 10 + 20) for seq in (1000, 2000)]
    _write_text(
        data_dir / "FusedInferAttentionScore.csv",
        "\n".join(
            [
                "Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,"
                "Duration(us),Runtime avg_seq_len,Runtime batch_size,Runtime sparse_mode,"
                "Runtime num_key_value_heads,Runtime input_layout",
                *rows,
            ]
        ),
    )
    return data_dir


@pytest.fixture
def attention_sparse_mismatch_dir(tmp_path):
    data_dir = tmp_path / "attention_sparse_mismatch"
    data_dir.mkdir()
    _write_text(
        data_dir / "op_mapping.yaml",
        """
version: "test"
operator_mappings:
  "tensor_cast.attention.default":
    kernel_type: FusedInferAttentionScore
    query_mode: attention_special
""",
    )
    rows = [_attention_row_without_batch(seq, 4, 128, seq / 10, sparse_mode=0) for seq in (1000, 2000)]
    _write_text(
        data_dir / "FusedInferAttentionScore.csv",
        "\n".join(
            [
                "Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,"
                "Duration(us),Runtime avg_seq_len,Runtime sparse_mode,Runtime num_key_value_heads,"
                "Runtime input_layout",
                *rows,
            ]
        ),
    )
    return data_dir


@pytest.fixture
def attention_3d_dir(tmp_path):
    data_dir = tmp_path / "attention_3d"
    data_dir.mkdir()
    _write_text(
        data_dir / "op_mapping.yaml",
        """
version: "test"
operator_mappings:
  "tensor_cast.attention.default":
    kernel_type: FusedInferAttentionScore
    query_mode: attention_special
""",
    )
    rows = [
        _attention_row(seq, batch, 4, head_dim, seq / 10 + batch * 10 + head_dim)
        for seq in (1000, 2000)
        for batch in (1, 3)
        for head_dim in (64, 128)
    ]
    _write_text(
        data_dir / "FusedInferAttentionScore.csv",
        "\n".join(
            [
                "Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,"
                "Duration(us),Runtime avg_seq_len,Runtime batch_size,Runtime sparse_mode,"
                "Runtime num_key_value_heads,Runtime input_layout",
                *rows,
            ]
        ),
    )
    return data_dir


def _attention_op(hidden_dim=512, head_dim=128):
    return _make_op_info(
        torch.ops.tensor_cast.attention.default,
        [
            torch.empty(2, hidden_dim, device="meta", dtype=torch.bfloat16),
            torch.empty(16, 128, 4, head_dim, device="meta", dtype=torch.bfloat16),
            torch.empty(16, 128, 4, head_dim, device="meta", dtype=torch.bfloat16),
            None,
            None,
            None,
            torch.tensor([1500, 1500], dtype=torch.int64),
            torch.tensor([1, 1], dtype=torch.int64),
        ],
    )


def _attention_bnsd_op():
    return _make_op_info(
        torch.ops.tensor_cast.attention.default,
        [
            torch.empty(2, 4, 1, 128, device="meta", dtype=torch.bfloat16),
            torch.empty(2, 4, 16, 128, device="meta", dtype=torch.bfloat16),
            torch.empty(2, 4, 16, 128, device="meta", dtype=torch.bfloat16),
            None,
            None,
            None,
            torch.tensor([1500, 1500], dtype=torch.int64),
            torch.tensor([1, 1], dtype=torch.int64),
        ],
    )


def test_attention_special_interpolates_2d_seq_batch(attention_2d_dir):
    ds = InterpolatingDataSource(ProfilingDataSource(attention_2d_dir))

    result = ds.lookup(_attention_op())

    assert result is not None
    assert result.source == QuerySource.INTERPOLATED
    assert result.latency_us == pytest.approx(1500 / 10 + 2 * 10)
    assert result.details["interpolation_dim"] == 2
    assert result.details["axes"] == ["seq", "batch"]
    assert result.details["method"] == "griddata_linear"
    assert result.shape_match_info.shape_match_rule == "interpolated_2d_griddata_linear"


def test_attention_index_skips_invalid_latency_rows(attention_invalid_latency_dir):
    ds = InterpolatingDataSource(ProfilingDataSource(attention_invalid_latency_dir))

    result = ds.lookup(_attention_op())

    assert result is not None
    assert result.source == QuerySource.INTERPOLATED
    assert result.latency_us == pytest.approx(1500 / 10 + 2 * 10)


def test_attention_unknown_sparse_mode_records_miss_without_exception(attention_2d_dir, monkeypatch):
    monkeypatch.setattr(interpolating_data_source_module, "_infer_attention_input_layout", lambda *_args: None)
    ds = InterpolatingDataSource(ProfilingDataSource(attention_2d_dir))
    op = _make_op_info(
        torch.ops.tensor_cast.attention.default,
        [
            torch.empty(2, 512, device="meta", dtype=torch.bfloat16),
            torch.empty(16, 128, 4, 128, device="meta", dtype=torch.bfloat16),
            torch.empty(16, 128, 4, 128, device="meta", dtype=torch.bfloat16),
            None,
            None,
            None,
            torch.tensor([1500, 1500], dtype=torch.int64),
            None,
        ],
    )

    result = ds.lookup(op)

    assert result is None
    assert ds.last_miss_reason == "attention_target_unavailable"
    assert {
        "reason": "attention_sparse_mode_unknown",
        "kernel_type": "FusedInferAttentionScore",
        "query_shape": (2, 512),
        "input_layout": None,
    } in ds.last_miss_details["miss_history"]


def test_attention_target_includes_quant_mode_for_quantized_candidate_groups(attention_quant_mode_dir):
    ds = InterpolatingDataSource(ProfilingDataSource(attention_quant_mode_dir))
    op = _attention_op()
    mapping = ds.base._op_mapping["operator_mappings"]["tensor_cast.attention.default"]

    target = ds._build_attention_target(op, mapping, "FusedInferAttentionScore")
    result = ds._interpolate_attention_multidim(op, mapping)

    assert target is not None
    assert dict(target.regime_key)["quant_mode"] == "W8A8"
    assert result is not None
    assert result.source == QuerySource.INTERPOLATED


def test_attention_q_tokens_filters_candidates_without_exact_field(tmp_path):
    data_dir = tmp_path / "attention_q_tokens"
    data_dir.mkdir()
    _write_text(
        data_dir / "op_mapping.yaml",
        """
version: "test"
operator_mappings:
  "tensor_cast.attention.default":
    kernel_type: FusedInferAttentionScore
    query_mode: attention_special
""",
    )
    rows = [_attention_row_with_q_tokens(2, seq, 2, 4, 128, latency) for seq, latency in ((1000, 100.0), (2000, 200.0))]
    rows.extend(_attention_row_with_q_tokens(4, seq, 2, 4, 128, 900.0) for seq in (1000, 2000))
    _write_text(
        data_dir / "FusedInferAttentionScore.csv",
        "\n".join(
            [
                "Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,"
                "Duration(us),Runtime avg_seq_len,Runtime batch_size,Runtime sparse_mode,"
                "Runtime num_key_value_heads,Runtime input_layout",
                *rows,
            ]
        ),
    )
    ds = InterpolatingDataSource(ProfilingDataSource(data_dir))

    result = ds.lookup(_attention_op())

    assert result is not None
    assert result.source == QuerySource.INTERPOLATED
    assert result.latency_us == pytest.approx(150.0)
    assert "q_tokens" not in dict(result.details["exact_fields"])
    assert result.details["attention_axes"]["q_tokens"] == 2.0
    assert "q_tokens" in result.details["effective_filters"]


def test_attention_q_tokens_accepts_block_padded_candidates(tmp_path):
    data_dir = tmp_path / "attention_q_tokens_block_padded"
    data_dir.mkdir()
    _write_text(
        data_dir / "op_mapping.yaml",
        """
version: "test"
operator_mappings:
  "tensor_cast.attention.default":
    kernel_type: FusedInferAttentionScore
    query_mode: attention_special
""",
    )
    rows = [
        _attention_row_with_q_tokens(144, seq, 2, 4, 128, latency) for seq, latency in ((1000, 100.0), (2000, 200.0))
    ]
    _write_text(
        data_dir / "FusedInferAttentionScore.csv",
        "\n".join(
            [
                "Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,"
                "Duration(us),Runtime avg_seq_len,Runtime batch_size,Runtime sparse_mode,"
                "Runtime num_key_value_heads,Runtime input_layout",
                *rows,
            ]
        ),
    )
    ds = InterpolatingDataSource(ProfilingDataSource(data_dir))
    op = _make_op_info(
        torch.ops.tensor_cast.attention.default,
        [
            torch.empty(136, 512, device="meta", dtype=torch.bfloat16),
            torch.empty(16, 128, 4, 128, device="meta", dtype=torch.bfloat16),
            torch.empty(16, 128, 4, 128, device="meta", dtype=torch.bfloat16),
            None,
            None,
            None,
            torch.tensor([1500, 1500], dtype=torch.int64),
            torch.tensor([1, 1], dtype=torch.int64),
        ],
    )

    result = ds.lookup(op)

    assert result is not None
    assert result.source == QuerySource.INTERPOLATED
    assert result.latency_us == pytest.approx(150.0)
    assert result.details["attention_axes"]["q_tokens"] == pytest.approx(136.0)
    assert "q_tokens" in result.details["effective_filters"]


def test_attention_q_tokens_rejects_non_padded_candidates(tmp_path):
    data_dir = tmp_path / "attention_q_tokens_non_padded"
    data_dir.mkdir()
    _write_text(
        data_dir / "op_mapping.yaml",
        """
version: "test"
operator_mappings:
  "tensor_cast.attention.default":
    kernel_type: FusedInferAttentionScore
    query_mode: attention_special
""",
    )
    rows = [
        _attention_row_with_q_tokens(150, seq, 2, 4, 128, latency) for seq, latency in ((1000, 100.0), (2000, 200.0))
    ]
    _write_text(
        data_dir / "FusedInferAttentionScore.csv",
        "\n".join(
            [
                "Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,"
                "Duration(us),Runtime avg_seq_len,Runtime batch_size,Runtime sparse_mode,"
                "Runtime num_key_value_heads,Runtime input_layout",
                *rows,
            ]
        ),
    )
    ds = InterpolatingDataSource(ProfilingDataSource(data_dir))
    op = _make_op_info(
        torch.ops.tensor_cast.attention.default,
        [
            torch.empty(136, 512, device="meta", dtype=torch.bfloat16),
            torch.empty(16, 128, 4, 128, device="meta", dtype=torch.bfloat16),
            torch.empty(16, 128, 4, 128, device="meta", dtype=torch.bfloat16),
            None,
            None,
            None,
            torch.tensor([1500, 1500], dtype=torch.int64),
            torch.tensor([1, 1], dtype=torch.int64),
        ],
    )

    result = ds.lookup(op)

    assert result is None
    assert ds.last_miss_reason == "insufficient_filtered_candidates"


def test_attention_index_cache_is_scoped_to_dataframe_content(attention_2d_dir):
    base = ProfilingDataSource(attention_2d_dir)
    ds = InterpolatingDataSource(base)

    first_index = ds._get_attention_index("FusedInferAttentionScore")
    same_content_df = base._load_csv("FusedInferAttentionScore").copy()
    base._csv_cache["FusedInferAttentionScore"] = same_content_df
    second_index = ds._get_attention_index("FusedInferAttentionScore")
    changed_content_df = same_content_df.copy()
    changed_content_df.loc[0, "Duration(us)"] = float(changed_content_df.loc[0, "Duration(us)"]) + 1.0
    base._csv_cache["FusedInferAttentionScore"] = changed_content_df
    third_index = ds._get_attention_index("FusedInferAttentionScore")

    assert first_index is not None
    assert second_index is not None
    assert third_index is not None
    assert second_index is first_index
    assert third_index is not first_index
    assert len(ds._attention_index_cache) == 2


def test_dataframe_fingerprint_is_computed_once_per_dataframe_object(attention_2d_dir, monkeypatch):
    base = ProfilingDataSource(attention_2d_dir)
    ds = InterpolatingDataSource(base)
    df = base._load_csv("FusedInferAttentionScore")
    original = ds._compute_dataframe_fingerprint
    calls = []

    def counted(frame):
        calls.append(id(frame))
        return original(frame)

    monkeypatch.setattr(ds, "_compute_dataframe_fingerprint", counted)

    copied_df = df.copy()
    first = ds._dataframe_fingerprint(df)
    second = ds._dataframe_fingerprint(df)
    copied = ds._dataframe_fingerprint(copied_df)

    assert first == second == copied
    assert calls == [id(df), id(copied_df)]


def test_attention_special_interpolates_3d_seq_batch_head_dim(attention_3d_dir):
    ds = InterpolatingDataSource(ProfilingDataSource(attention_3d_dir))

    result = ds.lookup(_attention_op(hidden_dim=4 * 96, head_dim=96))

    assert result is not None
    assert result.source == QuerySource.INTERPOLATED
    assert result.latency_us == pytest.approx(1500 / 10 + 2 * 10 + 96)
    assert result.details["interpolation_dim"] == 3
    assert result.details["axes"] == ["seq", "batch", "head_dim"]
    assert result.details["method"] == "griddata_linear"
    assert result.shape_match_info.shape_match_rule == "interpolated_3d_griddata_linear"


def test_attention_bnsd_layout_reads_kv_heads_from_second_dimension(attention_bnsd_dir):
    ds = InterpolatingDataSource(ProfilingDataSource(attention_bnsd_dir))

    result = ds.lookup(_attention_bnsd_op())

    assert result is not None
    assert result.source == QuerySource.INTERPOLATED
    assert dict(result.details["exact_fields"])["input_layout"] == "BNSD_NBSD"


def test_attention_kv_heads_respects_runtime_layout_axis_order():
    assert _attention_kv_heads_from_key(torch.empty(2, 4, 16, 128), "BNSD_NBSD") == 4
    assert _attention_kv_heads_from_key(torch.empty(16, 128, 4, 128), "TND") == 4


def test_attention_by_params_infers_input_layout_from_sparse_mode(tmp_path):
    data_dir = tmp_path / "attention_params_layout_inference"
    data_dir.mkdir()
    _write_text(
        data_dir / "op_mapping.yaml",
        """
version: "test"
operator_mappings: {}
""",
    )
    _write_text(
        data_dir / "FusedInferAttentionScore.csv",
        "\n".join(
            [
                "Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,"
                "Duration(us),Runtime avg_seq_len,Runtime batch_size,Runtime sparse_mode,"
                "Runtime num_key_value_heads,Runtime input_layout",
                _attention_row(1000, 2, 4, 128, 10.0),
                _attention_row(2000, 2, 4, 128, 20.0),
                _attention_row_bnsd(1000, 2, 4, 128, 100.0),
                _attention_row_bnsd(2000, 2, 4, 128, 300.0),
            ]
        ),
    )
    ds = InterpolatingDataSource(ProfilingDataSource(data_dir))

    result = ds._interpolate_attention_by_params(
        "FusedInferAttentionScore",
        {
            "q_shape_3d": (2, 4, 128),
            "avg_seq_len": 1500,
            "batch_size": 2,
            "sparse_mode": 0,
            "num_kv_heads": 4,
        },
        "DT_BF16",
    )

    assert result is not None
    assert result.latency_us == pytest.approx(200.0)
    assert dict(result.details["exact_fields"])["input_layout"] == "BNSD_NBSD"


def test_attention_2d_exact_lookup_does_not_force_tnd_layout(tmp_path):
    data_dir = tmp_path / "attention_2d_exact_layout"
    data_dir.mkdir()
    _write_text(
        data_dir / "op_mapping.yaml",
        """
version: "test"
operator_mappings:
  "tensor_cast.attention.default":
    kernel_type: FusedInferAttentionScore
    query_mode: attention_special
""",
    )
    _write_text(
        data_dir / "FusedInferAttentionScore.csv",
        "\n".join(
            [
                "Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,"
                "Duration(us),Runtime avg_seq_len,Runtime batch_size,Runtime sparse_mode,"
                "Runtime num_key_value_heads,Runtime input_layout",
                _attention_row_with_layout(1500, 2, 4, 128, 170.0, "TND"),
            ]
        ),
    )
    ds = ProfilingDataSource(data_dir)

    result = ds.lookup(_attention_op())

    assert result is not None
    assert result.source == QuerySource.MEASURED
    assert result.latency_us == pytest.approx(170.0)


def test_attention_2d_interpolation_target_does_not_force_tnd_layout(attention_2d_dir):
    ds = InterpolatingDataSource(ProfilingDataSource(attention_2d_dir))
    mapping = ds.base._op_mapping["operator_mappings"]["tensor_cast.attention.default"]

    target = ds._build_attention_target(_attention_op(), mapping, "FusedInferAttentionScore")

    assert target is not None
    target_fields = dict(target.regime_key)
    assert target_fields["input_layout"] == "TND"
    assert target_fields["sparse_mode"] == 3


def test_attention_without_runtime_batch_column_uses_schema_aware_axes(attention_no_batch_dir):
    ds = InterpolatingDataSource(ProfilingDataSource(attention_no_batch_dir))

    result = ds.lookup(_attention_op(hidden_dim=4 * 96, head_dim=96))

    assert result is not None
    assert result.source == QuerySource.INTERPOLATED
    assert result.latency_us == pytest.approx(1500 / 10 + 96)
    assert "batch" not in result.details["attention_axes"]
    assert result.details["axes"] == ["seq", "head_dim"]


def test_attention_mixed_layout_does_not_pollute_tnd_group_with_global_batch_axis(tmp_path):
    data_dir = tmp_path / "attention_mixed_batch_axis"
    data_dir.mkdir()
    _write_text(
        data_dir / "op_mapping.yaml",
        """
version: "test"
operator_mappings:
  "tensor_cast.attention.default":
    kernel_type: FusedInferAttentionScore
    query_mode: attention_special
""",
    )

    def tnd_without_batch(seq, head_dim, latency):
        q_shape = f"2,4,{head_dim}"
        key_shape = f"16,128,4,{head_dim}"
        return (
            f'"{q_shape};{key_shape};{key_shape}",'
            '"DT_BF16;DT_BF16;DT_BF16","ND;ND;ND",'
            f'"{q_shape}","DT_BF16","ND",{latency},'
            f"{seq},,3,4,TND"
        )

    rows = [
        tnd_without_batch(1000, 96, 196.0),
        tnd_without_batch(2000, 96, 296.0),
        _attention_row_bnsd(1000, 2, 4, 128, 900.0),
        _attention_row_bnsd(2000, 2, 4, 128, 1200.0),
    ]
    _write_text(
        data_dir / "FusedInferAttentionScore.csv",
        "\n".join(
            [
                "Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,"
                "Duration(us),Runtime avg_seq_len,Runtime batch_size,Runtime sparse_mode,"
                "Runtime num_key_value_heads,Runtime input_layout",
                *rows,
            ]
        ),
    )
    ds = InterpolatingDataSource(ProfilingDataSource(data_dir))

    result = ds.lookup(_attention_op(hidden_dim=4 * 96, head_dim=96))

    assert result is not None
    assert result.latency_us == pytest.approx(246.0)
    assert "batch" not in result.details["attention_axes"]
    assert result.details["axes"] == ["seq"]


def test_attention_filters_batched_points_when_batchless_candidates_exist(tmp_path):
    data_dir = tmp_path / "attention_batchless_subset"
    data_dir.mkdir()
    _write_text(
        data_dir / "op_mapping.yaml",
        """
version: "test"
operator_mappings: {}
""",
    )

    def tnd_row(seq, batch_value, latency):
        q_shape = "2,4,128"
        key_shape = "16,128,4,128"
        return (
            f'"{q_shape};{key_shape};{key_shape}",'
            '"DT_BF16;DT_BF16;DT_BF16","ND;ND;ND",'
            f'"{q_shape}","DT_BF16","ND",{latency},'
            f"{seq},{batch_value},3,4,TND"
        )

    _write_text(
        data_dir / "FusedInferAttentionScore.csv",
        "\n".join(
            [
                "Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,"
                "Duration(us),Runtime avg_seq_len,Runtime batch_size,Runtime sparse_mode,"
                "Runtime num_key_value_heads,Runtime input_layout",
                tnd_row(1000, "", 10.0),
                tnd_row(2000, "", 20.0),
                tnd_row(1000, "8", 1000.0),
                tnd_row(2000, "8", 2000.0),
            ]
        ),
    )
    ds = InterpolatingDataSource(ProfilingDataSource(data_dir))

    result = ds._interpolate_attention_by_params(
        "FusedInferAttentionScore",
        {
            "q_shape_3d": (2, 4, 128),
            "avg_seq_len": 1500,
            "sparse_mode": 3,
            "num_kv_heads": 4,
        },
        "DT_BF16",
    )

    assert result is not None
    assert result.latency_us == pytest.approx(15.0)
    assert result.details["batch_axis_status"] == "batch_axis_filtered"
    assert result.details["dropped_batched_candidates"] == 2


def test_attention_batch_axis_in_candidates_requires_known_target_batch(tmp_path):
    data_dir = tmp_path / "attention_batch_guard"
    data_dir.mkdir()
    _write_text(
        data_dir / "op_mapping.yaml",
        """
version: "test"
operator_mappings: {}
""",
    )
    _write_text(
        data_dir / "FusedInferAttentionScore.csv",
        "\n".join(
            [
                "Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,"
                "Duration(us),Runtime avg_seq_len,Runtime batch_size,Runtime sparse_mode,"
                "Runtime num_key_value_heads,Runtime input_layout",
                _attention_row(1000, 2, 4, 128, 10.0),
                _attention_row(2000, 2, 4, 128, 20.0),
            ]
        ),
    )
    ds = InterpolatingDataSource(ProfilingDataSource(data_dir))
    params = {
        "q_shape_3d": (2, 4, 128),
        "avg_seq_len": 1500,
        "sparse_mode": 3,
        "num_kv_heads": 4,
    }

    result_without_batch = ds._interpolate_attention_by_params("FusedInferAttentionScore", params, "DT_BF16")

    assert result_without_batch is not None
    assert result_without_batch.latency_us == pytest.approx(15.0)
    assert result_without_batch.details["batch_axis_status"] == "batch_axis_constant"
    assert result_without_batch.details["batch"] == pytest.approx(2.0)

    result_with_batch = ds._interpolate_attention_by_params(
        "FusedInferAttentionScore",
        {**params, "batch_size": 2},
        "DT_BF16",
    )

    assert result_with_batch is not None
    assert result_with_batch.latency_us == pytest.approx(15.0)
    assert result_with_batch.details["attention_axes"]["batch"] == pytest.approx(2.0)


def test_attention_rejects_mixed_batched_points_without_target_batch(tmp_path):
    data_dir = tmp_path / "attention_batch_guard_mixed"
    data_dir.mkdir()
    _write_text(
        data_dir / "op_mapping.yaml",
        """
version: "test"
operator_mappings: {}
""",
    )
    _write_text(
        data_dir / "FusedInferAttentionScore.csv",
        "\n".join(
            [
                "Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,"
                "Duration(us),Runtime avg_seq_len,Runtime batch_size,Runtime sparse_mode,"
                "Runtime num_key_value_heads,Runtime input_layout",
                _attention_row(1000, 2, 4, 128, 10.0),
                _attention_row(2000, 2, 4, 128, 20.0),
                _attention_row(1000, 3, 4, 128, 30.0),
                _attention_row(2000, 3, 4, 128, 40.0),
            ]
        ),
    )
    ds = InterpolatingDataSource(ProfilingDataSource(data_dir))

    result = ds._interpolate_attention_by_params(
        "FusedInferAttentionScore",
        {
            "q_shape_3d": (2, 4, 128),
            "avg_seq_len": 1500,
            "sparse_mode": 3,
            "num_kv_heads": 4,
        },
        "DT_BF16",
    )

    assert result is None
    assert ds.last_miss_reason == "batch_axis_unconstrained"
    assert ds.last_miss_details["interpolation_path"] == "composite_attention"


def test_attention_does_not_cross_sparse_mode_regime(attention_sparse_mismatch_dir):
    ds = InterpolatingDataSource(ProfilingDataSource(attention_sparse_mismatch_dir))

    assert ds.lookup(_attention_op()) is None


def test_attention_multidim_marks_2d_method(attention_2d_dir):
    ds = InterpolatingDataSource(ProfilingDataSource(attention_2d_dir))

    result = ds.lookup(_attention_op())

    assert result is not None
    assert result.source == QuerySource.INTERPOLATED
    assert result.details["method"] == "griddata_linear"
    assert result.details["axis_boundary"]["seq"] == [1000.0, 2000.0]
    assert result.shape_match_info.shape_match_rule == "interpolated_2d_griddata_linear"


def test_attention_multidim_marks_3d_method(attention_3d_dir):
    ds = InterpolatingDataSource(ProfilingDataSource(attention_3d_dir))

    result = ds.lookup(_attention_op(hidden_dim=4 * 96, head_dim=96))

    assert result is not None
    assert result.source == QuerySource.INTERPOLATED
    assert result.details["method"] == "griddata_linear"
    assert result.details["axis_boundary"]["seq"] == [1000.0, 2000.0]
    assert result.shape_match_info.shape_match_rule == "interpolated_3d_griddata_linear"


def test_attention_sqrt_seq_policy_is_applied_through_lookup(tmp_path):
    data_dir = tmp_path / "attention_sqrt_lookup"
    data_dir.mkdir()
    _write_text(
        data_dir / "op_mapping.yaml",
        """
version: "test"
interpolation_policy:
  kernel_overrides:
    FusedInferAttentionScore:
      axis_transform: sqrt_seq
operator_mappings:
  "tensor_cast.attention.default":
    kernel_type: FusedInferAttentionScore
    query_mode: attention_special
""",
    )
    rows = [_attention_row_without_batch(100, 4, 128, 10.0), _attention_row_without_batch(400, 4, 128, 20.0)]
    _write_text(
        data_dir / "FusedInferAttentionScore.csv",
        "\n".join(
            [
                "Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,"
                "Duration(us),Runtime avg_seq_len,Runtime sparse_mode,Runtime num_key_value_heads,"
                "Runtime input_layout",
                *rows,
            ]
        ),
    )
    op = _make_op_info(
        torch.ops.tensor_cast.attention.default,
        [
            torch.empty(2, 4 * 128, device="meta", dtype=torch.bfloat16),
            torch.empty(16, 128, 4, 128, device="meta", dtype=torch.bfloat16),
            torch.empty(16, 128, 4, 128, device="meta", dtype=torch.bfloat16),
            None,
            None,
            None,
            torch.tensor([225, 225], dtype=torch.int64),
            torch.tensor([1, 1], dtype=torch.int64),
        ],
    )
    ds = InterpolatingDataSource(ProfilingDataSource(data_dir))

    result = ds.lookup(op)

    assert result is not None
    assert result.latency_us == pytest.approx(15.0)
    assert result.details["method"] == "linear_1d_sqrt"
    assert result.details["axis_transform"] == "sqrt(seq)"
    assert result.details["target"]["seq"] == pytest.approx(15.0)
    assert result.details["target_pre_transform"]["seq"] == pytest.approx(225.0)
    assert result.details["axis_boundary_pre_transform"]["seq"] == [100.0, 400.0]
    assert result.shape_match_info.shape_match_rule == "interpolated_1d_linear_sqrt"


def test_axis_transform_requires_pre_transform_axes():
    key = make_regime_key([("kernel_type", "FusedInferAttentionScore")])
    candidate_group = CandidateGroup(
        key,
        [
            CandidatePoint(
                "FusedInferAttentionScore",
                {"seq": 1000.0},
                10.0,
                key,
                row_meta={"pre_transform_axes": {"seq": 1000.0}},
            ),
            CandidatePoint(
                "FusedInferAttentionScore",
                {"seq": 2000.0},
                20.0,
                key,
                row_meta={"pre_transform_axes": {"seq": 2000.0}},
            ),
        ],
    )

    with pytest.raises(ValueError, match="pre-transform|attention_axes"):
        candidate_group.interpolate({"seq": 1500.0}, [("seq",)], axis_transform="sqrt(seq)")


def test_sqrt_seq_group_marks_pre_transform_axes_and_result_method():
    key = make_regime_key([("kernel_type", "FusedInferAttentionScore")])
    candidate_group = CandidateGroup(
        key,
        [
            CandidatePoint("FusedInferAttentionScore", {"seq": 100.0}, 10.0, key),
            CandidatePoint("FusedInferAttentionScore", {"seq": 400.0}, 20.0, key),
            CandidatePoint("FusedInferAttentionScore", {"heads": 4.0}, 30.0, key),
        ],
    )

    transformed = InterpolatingDataSource._sqrt_seq_group(candidate_group)

    assert [point.axes["seq"] for point in transformed.points] == [10.0, 20.0]
    assert transformed.points[0].row_meta["pre_transform_axes"] == {"seq": 100.0}
    assert "pre_transform_axes" not in candidate_group.points[0].row_meta

    result = transformed.interpolate(
        {"seq": 15.0},
        [("seq",)],
        axis_transform="sqrt(seq)",
        extra_details={"attention_axes": {"seq": 225.0}},
    )
    assert result is not None

    marked = InterpolatingDataSource._mark_sqrt_interpolation(result)
    marked_again = InterpolatingDataSource._mark_sqrt_interpolation(marked)

    assert marked.method == "linear_1d_sqrt"
    assert marked.details["method"] == "linear_1d_sqrt"
    assert marked.shape_match_rule == "interpolated_1d_linear_sqrt"
    assert marked_again.method == marked.method
    assert marked_again.details["method"] == marked.details["method"]


def test_duplicate_coordinate_candidates_use_median_latency():
    key = make_regime_key([("kernel_type", "MatMulV2")])
    candidate_group = CandidateGroup(
        key,
        [
            CandidatePoint("MatMulV2", {"M": 100.0}, 10.0, key, row_index=0),
            CandidatePoint("MatMulV2", {"M": 100.0}, 14.0, key, row_index=1),
            CandidatePoint("MatMulV2", {"M": 100.0}, 100.0, key, row_index=2),
            CandidatePoint("MatMulV2", {"M": 200.0}, 20.0, key, row_index=3),
        ],
    )

    result = candidate_group.interpolate({"M": 150.0}, [("M",)], extra_details={"interpolation_path": "multidim"})

    assert result is not None
    assert result.latency_us == pytest.approx(17.0)
    assert result.matched_points[0].row_meta["duplicate_count"] == 3
    assert result.matched_points[0].row_meta["duplicate_row_indices"] == [0, 1, 2]
    assert result.matched_points[0].row_meta["aggregation"] == "median"
    assert result.matched_points[0].row_meta["duplicate_latency_max_us"] == 100.0


def test_candidate_filter_rejects_missing_non_selected_axis():
    key = make_regime_key([("kernel_type", "MatMulV2")])
    candidate_group = CandidateGroup(
        key,
        [
            CandidatePoint("MatMulV2", {"M": 100.0}, 10.0, key),
            CandidatePoint("MatMulV2", {"M": 200.0}, 20.0, key),
        ],
    )

    result = candidate_group.interpolate({"M": 150.0, "K": 64.0}, [("M",)])

    assert result is None


def test_irregular_2d_candidates_use_griddata_path():
    key = make_regime_key([("kernel_type", "MatMulV2")])
    candidate_group = CandidateGroup(
        key,
        [
            CandidatePoint("MatMulV2", {"M": 0.0, "K": 0.0}, 1.0, key, row_index=0),
            CandidatePoint("MatMulV2", {"M": 2.0, "K": 0.0}, 3.0, key, row_index=1),
            CandidatePoint("MatMulV2", {"M": 0.0, "K": 2.0}, 4.0, key, row_index=2),
        ],
    )

    result = candidate_group.interpolate({"M": 0.5, "K": 0.5}, [("M", "K")])

    assert result is not None
    assert result.details["method"] == "griddata_linear"
    assert result.latency_us == pytest.approx(2.25)
    assert sorted(result.details["simplex"]["barycentric_weights"]) == pytest.approx([0.25, 0.25, 0.5])
    assert len(result.matched_points) == 3


def test_multidim_projection_matches_do_not_count_as_exact_coordinate():
    key = make_regime_key([("kernel_type", "MatMulV2")])
    candidate_group = CandidateGroup(
        key,
        [
            CandidatePoint("MatMulV2", {"M": 0.0, "K": 1.0}, 10.0, key, row_index=0),
            CandidatePoint("MatMulV2", {"M": 1.0, "K": 0.0}, 20.0, key, row_index=1),
            CandidatePoint("MatMulV2", {"M": 2.0, "K": 2.0}, 30.0, key, row_index=2),
        ],
    )

    result = candidate_group.interpolate({"M": 1.0, "K": 1.0}, [("M", "K")])

    assert result is not None
    assert result.confidence == pytest.approx(0.65)
    assert result.details["exact_coordinate_match"] is False
    assert "exact_axis_value" not in result.details
    assert result.details["axis_boundary"] == {"M": [0.0, 2.0], "K": [0.0, 2.0]}


def test_griddata_rejects_degenerate_axes():
    key = make_regime_key([("kernel_type", "MatMulV2")])
    candidate_group = CandidateGroup(
        key,
        [
            CandidatePoint("MatMulV2", {"K": 7168.0, "N": 256.0}, 23.36, key),
            CandidatePoint("MatMulV2", {"K": 7168.0, "N": 512.0}, 6.32, key),
            CandidatePoint("MatMulV2", {"K": 7168.0, "N": 1024.0}, 317.186, key),
        ],
    )

    result = candidate_group.interpolate({"K": 7168.0, "N": 576.0}, [("K", "N")])

    assert result is None
    assert candidate_group.last_diagnostics["attempts"][0]["status"] == "grid_structure_rejected"
    assert candidate_group.last_diagnostics["attempts"][0]["quality"]["rejected_reason"] == "degenerate_axes"


def test_linear_1d_accepts_large_latency_jump_when_boundary_is_valid():
    key = make_regime_key([("kernel_type", "DynamicQuant")])
    candidate_group = CandidateGroup(
        key,
        [
            CandidatePoint("DynamicQuant", {"axis_0": 1024.0}, 82.702, key),
            CandidatePoint("DynamicQuant", {"axis_0": 4096.0}, 632.572, key),
        ],
    )

    result = candidate_group.interpolate({"axis_0": 2048.0}, [("axis_0",)])

    assert result is not None
    assert result.latency_us == pytest.approx(265.992)


@pytest.mark.parametrize(
    ("rhs_shape", "expected_layout"),
    [
        ((7168, 256), "rhs_k_n"),
        ((256, 7168), "rhs_n_k"),
    ],
)
def test_matmul_m1_shape_is_not_treated_as_batch_dimension(rhs_shape, expected_layout):
    extracted = InterpolatingDataSource._extract_matmul_axes_from_shapes(
        "MatMulV2",
        [(1, 7168), rhs_shape],
    )

    assert extracted is not None
    axes, batch_dims, source_layout = extracted
    assert axes == {"M": 1.0, "K": 7168.0, "N": 256.0}
    assert batch_dims == ((), ())
    assert source_layout == expected_layout


def test_compute_candidate_reports_latency_quality_reason(matmul_grid_dir):
    ds = InterpolatingDataSource(ProfilingDataSource(matmul_grid_dir))
    common = {
        "Input Shapes": "1,7168;7168,256",
        "Input Data Types": "DT_BF16;DT_BF16",
        "Input Formats": "ND;ND",
    }

    point, reason = ds._candidate_from_compute_row_with_reason(
        {**common, "Duration(us)": 13.32},
        "MatMulV2",
        "Duration(us)",
        0,
        None,
    )
    zero_point, zero_reason = ds._candidate_from_compute_row_with_reason(
        {**common, "Duration(us)": 0.0},
        "MatMulV2",
        "Duration(us)",
        1,
        None,
    )
    nan_point, nan_reason = ds._candidate_from_compute_row_with_reason(
        {**common, "Duration(us)": float("nan")},
        "MatMulV2",
        "Duration(us)",
        2,
        None,
    )

    assert point is not None
    assert point.axes == {"M": 1.0, "K": 7168.0, "N": 256.0}
    assert reason is None
    assert zero_point is None
    assert zero_reason == "latency_zero"
    assert nan_point is None
    assert nan_reason == "latency_invalid"


def test_attention_q_tokens_is_not_a_continuous_interpolation_axis():
    key = make_regime_key([("kernel_type", "FusedInferAttentionScore")])
    group = CandidateGroup(
        key,
        [
            CandidatePoint(
                "FusedInferAttentionScore",
                {"q_tokens": 2.0, "seq": 1000.0},
                20.0,
                key,
            ),
            CandidatePoint(
                "FusedInferAttentionScore",
                {"q_tokens": 4.0, "seq": 1000.0},
                40.0,
                key,
            ),
        ],
    )

    assert all("q_tokens" not in axes for axes in _ATTENTION_AXIS_GROUPS)

    result = group.interpolate(
        {"q_tokens": 3.0, "seq": 1000.0},
        _ATTENTION_AXIS_GROUPS,
    )

    assert result is None


def test_degenerate_2d_candidates_can_fall_back_to_1d():
    key = make_regime_key([("kernel_type", "MatMulV2")])
    candidate_group = CandidateGroup(
        key,
        [
            CandidatePoint("MatMulV2", {"M": 100.0, "K": 64.0}, 164.0, key),
            CandidatePoint("MatMulV2", {"M": 200.0, "K": 64.0}, 264.0, key),
        ],
    )

    result = candidate_group.interpolate({"M": 150.0, "K": 64.0}, [("M", "K"), ("M",)])

    assert result is not None
    assert result.latency_us == pytest.approx(214.0)
    assert result.details["method"] == "linear_1d"


def test_degenerate_3d_candidates_can_fall_back_to_1d():
    key = make_regime_key([("kernel_type", "MatMulV2")])
    candidate_group = CandidateGroup(
        key,
        [
            CandidatePoint("MatMulV2", {"M": 100.0, "K": 64.0, "N": 256.0}, 420.0, key),
            CandidatePoint("MatMulV2", {"M": 200.0, "K": 64.0, "N": 256.0}, 520.0, key),
        ],
    )

    result = candidate_group.interpolate({"M": 150.0, "K": 64.0, "N": 256.0}, [("M", "K", "N"), ("M",)])

    assert result is not None
    assert result.latency_us == pytest.approx(470.0)
    assert result.confidence == pytest.approx(0.7)
    assert result.details["method"] == "linear_1d"


def test_to_int_cell_rejects_infinity():
    assert _to_int_cell(float("inf")) is None


def test_compute_index_keeps_input_format_regimes_in_separate_candidate_groups():
    nd_key = make_regime_key([("kernel_type", "MatMulV2"), ("input_formats", ("ND", "ND"))])
    nz_key = make_regime_key([("kernel_type", "MatMulV2"), ("input_formats", ("ND", "FRACTAL_NZ"))])
    target_key = make_regime_key([("kernel_type", "MatMulV2")])
    index = CandidateIndex(
        [
            CandidatePoint("MatMulV2", {"M": 100.0}, 10.0, nd_key),
            CandidatePoint("MatMulV2", {"M": 200.0}, 20.0, nz_key),
        ]
    )

    candidate_groups = index.candidate_groups_matching(target_key, allow_extra_fields={"input_formats"})

    assert len(candidate_groups) == 2
    assert all(len(candidate_group.points) == 1 for candidate_group in candidate_groups)
    assert {dict(candidate_group.regime_key)["input_formats"] for candidate_group in candidate_groups} == {
        ("ND", "ND"),
        ("ND", "FRACTAL_NZ"),
    }


def test_candidate_group_1d_result_marks_interpolation_path():
    key = make_regime_key([("kernel_type", "MatMulV2")])
    candidate_group = CandidateGroup(
        key,
        [
            CandidatePoint("MatMulV2", {"axis_0": 100.0}, 10.0, key),
            CandidatePoint("MatMulV2", {"axis_0": 200.0}, 20.0, key),
        ],
    )
    result = candidate_group.interpolate(
        {"axis_0": 150.0},
        [("axis_0",)],
        extra_details={"interpolation_path": "compute_1d"},
    )

    assert result is not None
    assert result.details["interpolation_path"] == "compute_1d"
    assert result.details["method"] == "linear_1d"


def test_candidate_group_1d_exact_axis_value_is_marked():
    key = make_regime_key([("kernel_type", "MatMulV2")])
    candidate_group = CandidateGroup(
        key,
        [
            CandidatePoint("MatMulV2", {"axis_0": 100.0}, 10.0, key),
            CandidatePoint("MatMulV2", {"axis_0": 200.0}, 20.0, key),
        ],
    )
    result = candidate_group.interpolate({"axis_0": 100.0}, [("axis_0",)])

    assert result is not None
    assert result.latency_us == pytest.approx(10.0)
    assert result.details["exact_axis_value"] == {"axis_0": 100.0}
    assert result.details["axis_boundary"] == {"axis_0": [100.0, 100.0]}


def test_candidate_group_1d_rejects_zero_latency_interpolation():
    key = make_regime_key([("kernel_type", "RmsNorm")])
    candidate_group = CandidateGroup(
        key,
        [
            CandidatePoint("RmsNorm", {"axis_0": 100.0}, 0.0, key),
            CandidatePoint("RmsNorm", {"axis_0": 200.0}, 0.0, key),
        ],
    )
    result = candidate_group.interpolate({"axis_0": 150.0}, [("axis_0",)])

    assert result is None


def test_candidate_group_1d_accepts_small_latency_large_ratio():
    key = make_regime_key([("kernel_type", "RmsNorm")])
    candidate_group = CandidateGroup(
        key,
        [
            CandidatePoint("RmsNorm", {"axis_0": 100.0}, 3.0, key),
            CandidatePoint("RmsNorm", {"axis_0": 200.0}, 8.0, key),
        ],
    )
    result = candidate_group.interpolate({"axis_0": 150.0}, [("axis_0",)])

    assert result is not None
    assert result.latency_us == pytest.approx(5.5)


def test_candidate_group_dynamic_quant_uses_same_math_invariant_as_default():
    default_key = make_regime_key([("kernel_type", "MatMulV2")])
    dynamic_key = make_regime_key([("kernel_type", "DynamicQuant")])
    default_group = CandidateGroup(
        default_key,
        [
            CandidatePoint("MatMulV2", {"axis_0": 100.0}, 20.0, default_key),
            CandidatePoint("MatMulV2", {"axis_0": 200.0}, 70.0, default_key),
        ],
    )
    dynamic_group = CandidateGroup(
        dynamic_key,
        [
            CandidatePoint("DynamicQuant", {"axis_0": 100.0}, 20.0, dynamic_key),
            CandidatePoint("DynamicQuant", {"axis_0": 200.0}, 70.0, dynamic_key),
        ],
    )

    default_result = default_group.interpolate({"axis_0": 150.0}, [("axis_0",)])
    dynamic_result = dynamic_group.interpolate({"axis_0": 150.0}, [("axis_0",)])

    assert default_result is not None
    assert dynamic_result is not None
    assert dynamic_result.latency_us == pytest.approx(default_result.latency_us)


def test_candidate_group_1d_ignores_removed_quality_guard_override():
    key = make_regime_key([("kernel_type", "MatMulV2")])
    candidate_group = CandidateGroup(
        key,
        [
            CandidatePoint("MatMulV2", {"axis_0": 100.0}, 10.0, key),
            CandidatePoint("MatMulV2", {"axis_0": 200.0}, 20.0, key),
        ],
    )
    result = candidate_group.interpolate({"axis_0": 150.0}, [("axis_0",)])

    assert result is not None
    assert result.latency_us == pytest.approx(15.0)


def test_candidate_group_1d_skips_invalid_duplicate_candidate():
    key = make_regime_key([("kernel_type", "RmsNorm")])
    candidate_group = CandidateGroup(
        key,
        [
            CandidatePoint("RmsNorm", {"axis_0": 100.0}, float("nan"), key),
            CandidatePoint("RmsNorm", {"axis_0": 100.0}, 10.0, key),
            CandidatePoint("RmsNorm", {"axis_0": 200.0}, 20.0, key),
        ],
    )
    result = candidate_group.interpolate({"axis_0": 150.0}, [("axis_0",)])

    assert result is not None
    assert result.latency_us == pytest.approx(15.0)


def test_composite_compute_candidate_shortage_records_miss(tmp_path):
    data_dir = tmp_path / "composite_compute_shortage"
    data_dir.mkdir()
    _write_text(data_dir / "op_mapping.yaml", 'version: "test"')
    _write_text(
        data_dir / "MatMulV2.csv",
        """
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Duration(us)
"100,64;64,256","DT_BF16;DT_BF16","ND;ND","100,256","DT_BF16","ND",10.0
""",
    )
    ds = InterpolatingDataSource(ProfilingDataSource(data_dir))

    result = ds._interpolate_compute_by_shapes("MatMulV2", [(150, 64), (64, 256)], "DT_BF16")

    assert result is None
    assert ds.last_miss_reason == "insufficient_filtered_candidates"
    assert ds.last_miss_details["interpolation_path"] == "composite_compute"
    assert ds.last_miss_details["target_axes"]["M"] == pytest.approx(150.0)


def test_composite_generic_compute_uses_effective_input_count_for_index(tmp_path):
    data_dir = tmp_path / "composite_generic_effective_input_count"
    data_dir.mkdir()
    _write_text(data_dir / "op_mapping.yaml", 'version: "test"')
    _write_text(
        data_dir / "RmsNorm.csv",
        """
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Duration(us)
"100,64;999","DT_BF16;DT_INT64","ND;ND","100,64","DT_BF16","ND",10.0
"200,64;999","DT_BF16;DT_INT64","ND;ND","200,64","DT_BF16","ND",20.0
""",
    )
    ds = InterpolatingDataSource(ProfilingDataSource(data_dir))

    result = ds._interpolate_compute_by_shapes("RmsNorm", [(150, 64)], "DT_BF16")

    assert result is not None
    assert result.source == QuerySource.INTERPOLATED
    assert result.latency_us == pytest.approx(15.0)
    assert result.details["interpolation_path"] == "composite_compute_1d"
    assert result.details["candidate_count"] == 2
    assert dict(result.details["exact_fields"])["input_count"] == 1


def test_composite_generic_output_numel_missing_output_shape_records_specific_reason(tmp_path, monkeypatch):
    data_dir = tmp_path / "composite_output_numel_missing_output"
    data_dir.mkdir()
    func_str = "tensor_cast.fake_output_numel_composite.default"
    _write_text(
        data_dir / "op_mapping.yaml",
        f"""
version: "test"
interpolation_policy:
  kernel_overrides:
    GatherV2:
      generic_compute:
        axis: output_numel
operator_mappings:
  "{func_str}":
    composite: true
""",
    )
    _write_text(
        data_dir / "GatherV2.csv",
        """
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Duration(us)
"100,64","DT_BF16","ND","100,64","DT_BF16","ND",10.0
"200,64","DT_BF16","ND","200,64","DT_BF16","ND",20.0
""",
    )

    def fake_decomposer(_op_invoke_info, _mapping):
        return [SubKernelSpec("GatherV2", [(150, 64)], "DT_BF16")]

    monkeypatch.setitem(interpolating_data_source_module.COMPOSITE_DECOMPOSERS, func_str, fake_decomposer)
    ds = InterpolatingDataSource(ProfilingDataSource(data_dir))

    result = ds._interpolate_composite(_make_op_info(func_str, []), {"composite": True}, func_str)

    assert result is None
    assert ds.last_miss_reason == "composite_sub_kernel_failed"
    assert ds.last_miss_details["sub_kernel_miss_reason"] == "generic_compute_output_shape_unavailable"


def test_composite_interpolation_records_exact_and_interpolated_sub_kernel_details(tmp_path, monkeypatch):
    data_dir = tmp_path / "composite_sub_kernel_details"
    data_dir.mkdir()
    _write_text(data_dir / "op_mapping.yaml", 'version: "test"')
    _write_text(
        data_dir / "MatMulV2.csv",
        """
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Duration(us)
"100,64;64,256","DT_BF16;DT_BF16","ND;ND","100,256","DT_BF16","ND",10.0
""",
    )
    _write_text(
        data_dir / "RmsNorm.csv",
        """
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Duration(us)
"100,64","DT_BF16","ND","100,64","DT_BF16","ND",20.0
"200,64","DT_BF16","ND","200,64","DT_BF16","ND",40.0
""",
    )
    func_str = "tensor_cast.fake_composite.default"

    def fake_decomposer(_op_invoke_info, _mapping):
        return [
            SubKernelSpec("MatMulV2", [(100, 64), (64, 256)], "DT_BF16"),
            SubKernelSpec("RmsNorm", [(150, 64)], "DT_BF16"),
        ]

    monkeypatch.setitem(interpolating_data_source_module.COMPOSITE_DECOMPOSERS, func_str, fake_decomposer)
    ds = InterpolatingDataSource(ProfilingDataSource(data_dir))

    result = ds._interpolate_composite(_make_op_info(func_str, []), {"composite": True}, func_str)

    assert result is not None
    assert result.source == QuerySource.INTERPOLATED
    assert result.latency_us == pytest.approx(40.0)
    assert result.shape_match_info is not None
    assert result.shape_match_info.shape_match_rule == "interpolated_composite"
    assert result.details["method"] == "decomposed_interpolation"
    assert result.details["sub_kernels"][0]["source"] == QuerySource.MEASURED.name
    assert result.details["sub_kernels"][0]["candidate_kernel_types"] == ["MatMulV2"]
    assert result.details["sub_kernels"][0]["matched_kernel_type"] == "MatMulV2"
    assert result.details["sub_kernels"][0]["latency_us"] == pytest.approx(10.0)
    assert result.details["sub_kernels"][1]["source"] == QuerySource.INTERPOLATED.name
    assert result.details["sub_kernels"][1]["matched_kernel_type"] == "RmsNorm"
    assert result.details["sub_kernels"][1]["method"] == "linear_1d"
    assert result.details["sub_kernels"][1]["latency_us"] == pytest.approx(30.0)


def test_composite_all_measured_sub_kernels_rolls_up_measured_source(tmp_path, monkeypatch):
    data_dir = tmp_path / "composite_all_measured_rollup"
    data_dir.mkdir()
    func_str = "tensor_cast.fake_all_measured_composite.default"
    _write_text(
        data_dir / "op_mapping.yaml",
        f"""
version: "test"
operator_mappings:
  "{func_str}":
    composite: true
""",
    )
    _write_text(
        data_dir / "MatMulV2.csv",
        """
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Duration(us)
"100,64;64,256","DT_BF16;DT_BF16","ND;ND","100,256","DT_BF16","ND",10.0
"200,64;64,128","DT_BF16;DT_BF16","ND;ND","200,128","DT_BF16","ND",20.0
""",
    )

    def fake_decomposer(_op_invoke_info, _mapping):
        return [
            SubKernelSpec("MatMulV2", [(100, 64), (64, 256)], "DT_BF16"),
            SubKernelSpec("MatMulV2", [(200, 64), (64, 128)], "DT_BF16"),
        ]

    monkeypatch.setitem(interpolating_data_source_module.COMPOSITE_DECOMPOSERS, func_str, fake_decomposer)
    ds = InterpolatingDataSource(ProfilingDataSource(data_dir))

    result = ds._interpolate_composite(_make_op_info(func_str, []), {"composite": True}, func_str)

    assert result is not None
    assert result.source == QuerySource.MEASURED
    assert result.confidence == pytest.approx(0.5)
    assert result.latency_us == pytest.approx(30.0)
    assert result.shape_match_info is not None
    assert result.shape_match_info.shape_match_rule == "composite_measured"
    assert result.details["composite"] is True
    assert len(result.details["sub_kernels"]) == 2
    assert all(detail["source"] == QuerySource.MEASURED.name for detail in result.details["sub_kernels"])
    assert [detail["matched_kernel_type"] for detail in result.details["sub_kernels"]] == ["MatMulV2", "MatMulV2"]

    monkeypatch.setattr(ds.base, "lookup", lambda _op_invoke_info: None)
    lookup_result = ds.lookup(_make_op_info(func_str, []))

    assert lookup_result is not None
    assert lookup_result.source == QuerySource.MEASURED
    assert lookup_result.shape_match_info is not None
    assert lookup_result.shape_match_info.shape_match_rule == "composite_measured"


def test_composite_attention_candidate_shortage_records_miss(tmp_path):
    data_dir = tmp_path / "composite_attention_shortage"
    data_dir.mkdir()
    _write_text(data_dir / "op_mapping.yaml", 'version: "test"')
    _write_text(
        data_dir / "FusedInferAttentionScore.csv",
        """
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Duration(us),Runtime avg_seq_len
"16,4,128","DT_BF16","ND","16,4,128","DT_BF16","ND",10.0,1000
""",
    )
    ds = InterpolatingDataSource(ProfilingDataSource(data_dir))

    result = ds._interpolate_attention_by_params(
        "FusedInferAttentionScore",
        {"q_shape_3d": (16, 4, 128), "avg_seq_len": 1500},
        "DT_BF16",
    )

    assert result is None
    assert ds.last_miss_reason in {"insufficient_filtered_candidates", "regime_key_unmatched"}
    assert ds.last_miss_details["interpolation_path"] == "composite_attention"
    if "target_axes" in ds.last_miss_details:
        assert ds.last_miss_details["target_axes"]["seq"] == pytest.approx(1500.0)


def test_elementwise_candidate_shortage_records_miss(tmp_path):
    data_dir = tmp_path / "elementwise_shortage"
    data_dir.mkdir()
    _write_text(data_dir / "op_mapping.yaml", 'version: "test"')
    _write_text(
        data_dir / "RmsNorm.csv",
        """
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Duration(us)
"100,64","DT_BF16","ND","100,64","DT_BF16","ND",10.0
""",
    )
    ds = InterpolatingDataSource(ProfilingDataSource(data_dir))
    op = _make_op_info("tensor_cast.rms_norm.default", [torch.empty(150, 64, device="meta")])
    op.out = torch.empty(150, 64, device="meta", dtype=torch.bfloat16)

    result = ds._interpolate_elementwise(op, {"kernel_type": "RmsNorm"})

    assert result is None
    assert ds.last_miss_reason == "insufficient_filtered_candidates"
    assert ds.last_miss_details["interpolation_path"] == "elementwise_1d"
    assert ds.last_miss_details["target"] == pytest.approx(150.0)


def test_elementwise_decode_batch_shape_interpolates_positive_rms_norm_latency(tmp_path):
    data_dir = tmp_path / "rms_norm_decode_batch"
    data_dir.mkdir()
    _write_text(
        data_dir / "op_mapping.yaml",
        """
version: "test"
operator_mappings:
  "tensor_cast.rms_norm.default":
    kernel_type: RmsNorm
    query_mode: elementwise
""",
    )
    _write_text(
        data_dir / "RmsNorm.csv",
        """
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Duration(us)
"1,6,6144;6144","DT_BF16;FLOAT","ND;ND","1,6,6144","DT_BF16","ND",5.702
"1,8,6144;6144","DT_BF16;FLOAT","ND;ND","1,8,6144","DT_BF16","ND",5.36
""",
    )
    ds = InterpolatingDataSource(ProfilingDataSource(data_dir))
    op = _make_op_info(
        "tensor_cast.rms_norm.default",
        [
            torch.empty(1, 7, 6144, device="meta", dtype=torch.bfloat16),
            torch.empty(6144, device="meta", dtype=torch.float32),
        ],
    )
    op.out = torch.empty(1, 7, 6144, device="meta", dtype=torch.bfloat16)

    result = ds.lookup(op)

    assert result is not None
    assert result.source == QuerySource.INTERPOLATED
    assert result.latency_us == pytest.approx(5.531)
    assert result.latency_us > 0.0
    assert result.details["axis_boundary"] == {"axis_0": [6.0, 8.0]}


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
    _write_text(
        data_dir / "Add.csv",
        """
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Duration(us)
"128,7168;7168","DT_BF16;DT_BF16","ND;ND","128,7168","DT_BF16","ND",10.0
"256,7168;7168","DT_BF16;DT_BF16","ND;ND","256,7168","DT_BF16","ND",20.0
"128,7168;128,7168","DT_BF16;DT_BF16","ND;ND","128,7168","DT_BF16","ND",100.0
"256,7168;256,7168","DT_BF16;DT_BF16","ND;ND","256,7168","DT_BF16","ND",200.0
""",
    )
    ds = InterpolatingDataSource(ProfilingDataSource(data_dir))
    broadcast_op = _make_op_info(
        torch.ops.aten.add.Tensor,
        [
            torch.empty(192, 7168, device="meta", dtype=torch.bfloat16),
            torch.empty(7168, device="meta", dtype=torch.bfloat16),
        ],
    )
    broadcast_op.out = torch.empty(192, 7168, device="meta", dtype=torch.bfloat16)
    full_op = _make_op_info(
        torch.ops.aten.add.Tensor,
        [
            torch.empty(192, 7168, device="meta", dtype=torch.bfloat16),
            torch.empty(192, 7168, device="meta", dtype=torch.bfloat16),
        ],
    )
    full_op.out = torch.empty(192, 7168, device="meta", dtype=torch.bfloat16)

    broadcast_result = ds.lookup(broadcast_op)
    full_result = ds.lookup(full_op)

    assert broadcast_result is not None
    assert broadcast_result.latency_us == pytest.approx(15.0)
    assert full_result is not None
    assert full_result.latency_us == pytest.approx(150.0)


def test_elementwise_unknown_signature_strips_token_axis_when_aligned(tmp_path):
    data_dir = tmp_path / "elementwise_unknown_signature"
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
"128,7168;128,1","DT_BF16;DT_BF16","ND;ND","128,7168","DT_BF16","ND",10.0
"256,7168;256,1","DT_BF16;DT_BF16","ND;ND","256,7168","DT_BF16","ND",20.0
"128,7168;64,1","DT_BF16;DT_BF16","ND;ND","128,7168","DT_BF16","ND",1000.0
"256,7168;64,1","DT_BF16;DT_BF16","ND;ND","256,7168","DT_BF16","ND",2000.0
""",
    )
    ds = InterpolatingDataSource(ProfilingDataSource(data_dir))
    op = _make_op_info(
        torch.ops.aten.add.Tensor,
        [
            torch.empty(192, 7168, device="meta", dtype=torch.bfloat16),
            torch.empty(192, 1, device="meta", dtype=torch.bfloat16),
        ],
    )
    op.out = torch.empty(192, 7168, device="meta", dtype=torch.bfloat16)

    result = ds.lookup(op)

    assert result is not None
    assert result.latency_us == pytest.approx(15.0)
    assert InterpolatingDataSource._elementwise_input_signature(
        [(192, 7168), (192, 1)],
        (192, 7168),
    ) == (("full", (7168,)), ("unknown", (1,)))


def test_elementwise_dtype_scaled_fallback_keeps_input_signature_boundary(tmp_path):
    data_dir = tmp_path / "elementwise_scaled_signature"
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
"128,7168;7168","DT_BF16;DT_BF16","ND;ND","128,7168","DT_BF16","ND",10.0
"256,7168;7168","DT_BF16;DT_BF16","ND;ND","256,7168","DT_BF16","ND",20.0
"128,7168;128,7168","DT_BF16;DT_BF16","ND;ND","128,7168","DT_BF16","ND",100.0
"256,7168;256,7168","DT_BF16;DT_BF16","ND;ND","256,7168","DT_BF16","ND",200.0
""",
    )
    ds = InterpolatingDataSource(ProfilingDataSource(data_dir))
    broadcast_op = _make_op_info(
        torch.ops.aten.add.Tensor,
        [
            torch.empty(192, 7168, device="meta", dtype=torch.float32),
            torch.empty(7168, device="meta", dtype=torch.float32),
        ],
    )
    broadcast_op.out = torch.empty(192, 7168, device="meta", dtype=torch.float32)

    result = ds.lookup(broadcast_op)

    assert result is not None
    assert result.latency_us == pytest.approx(30.0)
    assert result.details["dtype_attempt"] == "scaled_dtype"
    assert result.details["dtype_scaled"] is True


def test_model_runner_wraps_profiling_datasource_by_default(monkeypatch, tmp_path):
    from tensor_cast.core import model_runner as model_runner_module
    from tensor_cast.core.user_config import UserInputConfig

    captured = {}

    class FakeProfilingDataSource:
        def __init__(self, data_dir, device_profile, parallel_config=None):
            self.data_dir = data_dir

    class FakeInterpolatingDataSource:
        def __init__(self, base):
            self.base = base

    class FakeAnalyticPerformanceModel:
        def __init__(self, device_profile):
            self.device_profile = device_profile

    class FakeEmpiricalPerformanceModel:
        def __init__(self, device_profile, data_source, fallback_model=None):
            captured["data_source"] = data_source
            self.device_profile = device_profile
            self.data_source = data_source
            self.fallback_model = fallback_model

    fake_model = MagicMock()
    fake_model.eval.return_value = fake_model
    fake_model.weight_size = 0

    monkeypatch.setattr(model_runner_module, "ProfilingDataSource", FakeProfilingDataSource)
    monkeypatch.setattr(model_runner_module, "InterpolatingDataSource", FakeInterpolatingDataSource)
    monkeypatch.setattr(model_runner_module, "AnalyticPerformanceModel", FakeAnalyticPerformanceModel)
    monkeypatch.setattr(model_runner_module, "EmpiricalPerformanceModel", FakeEmpiricalPerformanceModel)
    monkeypatch.setattr(model_runner_module, "build_model", lambda user_input: fake_model)

    config = UserInputConfig(
        device="TEST_DEVICE",
        model_id="Qwen/Qwen3-32B",
        performance_model=["profiling"],
        profiling_database=str(tmp_path),
    )

    model_runner_module.ModelRunner(config)

    assert isinstance(captured["data_source"], FakeInterpolatingDataSource)
    assert isinstance(captured["data_source"].base, FakeProfilingDataSource)


def test_model_runner_disable_profiling_interpolation_uses_base_datasource(monkeypatch, tmp_path):
    from tensor_cast.core import model_runner as model_runner_module
    from tensor_cast.core.user_config import UserInputConfig

    captured = {}

    class FakeProfilingDataSource:
        def __init__(self, data_dir, device_profile, parallel_config=None):
            self.data_dir = data_dir

    class FakeInterpolatingDataSource:
        def __init__(self, base):
            self.base = base

    class FakeAnalyticPerformanceModel:
        def __init__(self, device_profile):
            self.device_profile = device_profile

    class FakeEmpiricalPerformanceModel:
        def __init__(self, device_profile, data_source, fallback_model=None):
            captured["data_source"] = data_source
            self.device_profile = device_profile
            self.data_source = data_source
            self.fallback_model = fallback_model

    fake_model = MagicMock()
    fake_model.eval.return_value = fake_model
    fake_model.weight_size = 0

    monkeypatch.setattr(model_runner_module, "ProfilingDataSource", FakeProfilingDataSource)
    monkeypatch.setattr(model_runner_module, "InterpolatingDataSource", FakeInterpolatingDataSource)
    monkeypatch.setattr(model_runner_module, "AnalyticPerformanceModel", FakeAnalyticPerformanceModel)
    monkeypatch.setattr(model_runner_module, "EmpiricalPerformanceModel", FakeEmpiricalPerformanceModel)
    monkeypatch.setattr(model_runner_module, "build_model", lambda user_input: fake_model)

    config = UserInputConfig(
        device="TEST_DEVICE",
        model_id="Qwen/Qwen3-32B",
        performance_model=["profiling"],
        profiling_database=str(tmp_path),
        disable_profiling_interpolation=True,
    )

    model_runner_module.ModelRunner(config)

    assert isinstance(captured["data_source"], FakeProfilingDataSource)


def test_user_config_print_info_reports_profiling_interpolation(capsys):
    from tensor_cast.core.user_config import UserInputConfig

    config = UserInputConfig(
        performance_model=["profiling"],
        disable_profiling_interpolation=True,
    )

    config._print_info()

    assert "Profiling interpolation: Disabled" in capsys.readouterr().out


def test_tensor_cast_text_generate_main_propagates_disable_interpolation(monkeypatch):
    from cli.inference import text_generate
    from tensor_cast.core import input_generator as input_generator_module
    from tensor_cast.core import model_runner as model_runner_module
    from tensor_cast.core import user_config as user_config_module

    captured = {}

    class FakeUserInputConfig:
        @classmethod
        def from_args(cls, args):
            captured["disable_profiling_interpolation"] = args.disable_profiling_interpolation
            return object()

    class FakeMetrics:
        def print_info(self):
            captured["printed"] = True

    class FakeModelRunner:
        def __init__(self, user_input):
            captured["user_input"] = user_input

        def run_inference(self, generate_inputs_func):
            captured["generate_inputs_func"] = generate_inputs_func
            return FakeMetrics()

    def fake_generate_inputs(*_args, **_kwargs):
        return None

    monkeypatch.setattr(user_config_module, "UserInputConfig", FakeUserInputConfig)
    monkeypatch.setattr(model_runner_module, "ModelRunner", FakeModelRunner)
    monkeypatch.setattr(input_generator_module, "generate_inputs", fake_generate_inputs)
    monkeypatch.setattr(
        "sys.argv",
        [
            "text_generate",
            "Qwen/Qwen3-32B",
            "--num-queries",
            "1",
            "--query-length",
            "8",
            "--disable-profiling-interpolation",
        ],
    )

    text_generate.main()

    assert captured["disable_profiling_interpolation"] is True
    assert captured["printed"] is True
