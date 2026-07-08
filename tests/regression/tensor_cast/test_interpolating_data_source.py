"""Tests for InterpolatingDataSource."""

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
import torch
from tensor_cast.performance_model.profiling_database.data_source import (
    QueryResult,
    QuerySource,
)
from tensor_cast.performance_model.profiling_database.interpolating_data_source import (
    InterpolatingDataSource,
)
from tensor_cast.performance_model.profiling_database.profiling_data_source import (
    ProfilingDataSource,
)


def _make_op_info(func, input_tensors):
    mock = MagicMock()
    mock.func = func
    mock.args = tuple(input_tensors)
    mock.kwargs = {}
    mock.out = None
    return mock


# --- Fixtures ---

INTERP_COMPUTE_MAPPING = """\
version: "test"
device: TEST_DEVICE
operator_mappings:
  "aten.mm.default":
    kernel_type: MatMulV2
  "tensor_cast.all_reduce.default":
    kernel_type: hcom_allReduce_
    category: communication
  "tensor_cast.attention.default":
    kernel_type: FusedInferAttentionScore
    query_mode: attention_special
  "aten.add.Tensor":
    kernel_type: Add
    query_mode: elementwise
"""

# Add CSV with multiple M values for elementwise interpolation
INTERP_ADD_CSV = """\
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Duration(us)
"128,7168;128,7168","DT_BF16;DT_BF16","ND;ND","128,7168","DT_BF16","ND",6.0
"256,7168;256,7168","DT_BF16;DT_BF16","ND;ND","256,7168","DT_BF16","ND",12.0
"128,1536;128,1536","DT_BF16;DT_BF16","ND;ND","128,1536","DT_BF16","ND",2.0
"256,1536;256,1536","DT_BF16;DT_BF16","ND;ND","256,1536","DT_BF16","ND",4.0\
"""

# MatMulV2 CSV with multiple seq lengths for interpolation
INTERP_MATMUL_CSV = """\
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Duration(us)
"100,512;512,1024","DT_BF16;DT_BF16","ND;ND","100,1024","DT_BF16","ND",10.0
"200,512;512,1024","DT_BF16;DT_BF16","ND;ND","200,1024","DT_BF16","ND",20.0
"400,512;512,1024","DT_BF16;DT_BF16","ND;ND","400,1024","DT_BF16","ND",40.0"""

INTERP_COMM_CSV = """\
message_bytes,num_devices,dtype,topology_tier,Duration(us)
100000,16,DT_BF16,0,100.0
200000,16,DT_BF16,0,200.0
400000,16,DT_BF16,0,400.0"""

_INTERP_FIA_ROW_COMMON = (
    '"1,4,128;16,128,4,128;16,128,4,128;;;;1;;;;;;;;1,16;;;;;;;;;;;;;;"'
    ',"DT_BF16;DT_BF16;DT_BF16;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;'
    "INT64;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;"
    "DT_UNDEFINED;DT_UNDEFINED;INT32;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;"
    "DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;"
    "DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;"
    'DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED"'
    ',"ND;ND;ND;NULL;NULL;NULL;ND;NULL;NULL;NULL;NULL;NULL;NULL;NULL;ND;'
    'NULL;NULL;NULL;NULL;NULL;NULL;NULL;NULL;NULL;NULL;NULL;NULL;NULL;NULL;NULL;NULL"'
    ',"""1,4,128;""","DT_BF16;FLOAT","ND;ND"'
)
INTERP_FIA_CSV = (
    "Input Shapes,Input Data Types,Input Formats,Output Shapes,"
    "Output Data Types,Output Formats,Duration(us),Runtime avg_seq_len,Runtime batch_size,"
    "Runtime sparse_mode,Runtime num_key_value_heads,Runtime input_layout\n"
    + _INTERP_FIA_ROW_COMMON
    + ",100.0,1000,1,3,4,TND\n"
    + _INTERP_FIA_ROW_COMMON
    + ",1600.0,4000,1,3,4,TND"
)


@pytest.fixture
def interp_data_dir(tmp_path):
    data_dir = tmp_path / "interp"
    data_dir.mkdir()
    (data_dir / "op_mapping.yaml").write_text(INTERP_COMPUTE_MAPPING)
    (data_dir / "MatMulV2.csv").write_text(INTERP_MATMUL_CSV.strip())
    (data_dir / "hcom_allReduce_.csv").write_text(INTERP_COMM_CSV.strip())
    (data_dir / "FusedInferAttentionScore.csv").write_text(INTERP_FIA_CSV.strip())
    (data_dir / "Add.csv").write_text(INTERP_ADD_CSV.strip())
    return data_dir


# --- Tests ---


def test_exact_match_passthrough(interp_data_dir):
    """Exact match should pass through from base."""
    base = ProfilingDataSource(interp_data_dir)
    ds = InterpolatingDataSource(base)
    op = _make_op_info(
        torch.ops.aten.mm.default,
        [
            torch.empty(100, 512, device="meta", dtype=torch.bfloat16),
            torch.empty(512, 1024, device="meta", dtype=torch.bfloat16),
        ],
    )
    result = ds.lookup(op)
    assert result is not None
    assert abs(result.latency_us - 10.0) < 0.01
    assert result.source == QuerySource.MEASURED


def test_compute_interpolation_midpoint(interp_data_dir):
    """seq=150 between 100 and 200 should interpolate to ~15.0 us."""
    base = ProfilingDataSource(interp_data_dir)
    ds = InterpolatingDataSource(base)
    op = _make_op_info(
        torch.ops.aten.mm.default,
        [
            torch.empty(150, 512, device="meta", dtype=torch.bfloat16),
            torch.empty(512, 1024, device="meta", dtype=torch.bfloat16),
        ],
    )
    result = ds.lookup(op)
    assert result is not None, "Should interpolate between seq=100 and seq=200"
    assert abs(result.latency_us - 15.0) < 0.5
    assert result.source == QuerySource.INTERPOLATED


def test_moe_fused_query_mode_skips_phase1_interpolation(tmp_path, monkeypatch):
    data_dir = tmp_path / "moe_fused"
    data_dir.mkdir()
    (data_dir / "op_mapping.yaml").write_text(
        """
version: "test"
operator_mappings:
  "aten.mm.default":
    kernel_type: DispatchFFNCombine
    query_mode: moe_fused
""".strip(),
        encoding="utf-8",
    )
    ds = InterpolatingDataSource(ProfilingDataSource(data_dir))

    monkeypatch.setattr(
        ds, "_interpolate_compute", lambda *_args, **_kwargs: pytest.fail("moe_fused used compute interpolation")
    )
    op = _make_op_info(
        torch.ops.aten.mm.default,
        [
            torch.empty(150, 512, device="meta", dtype=torch.bfloat16),
            torch.empty(512, 1024, device="meta", dtype=torch.bfloat16),
        ],
    )

    ds.base.last_miss_reason = "latency_invalid"
    assert ds._interpolate(op) is None
    assert ds.last_miss_reason == "wrapper_moe_fused_disabled"
    assert ds.last_miss_details["base_miss_reason"] == "latency_invalid"


def test_compute_interpolation_quarter(interp_data_dir):
    """seq=300 between 200 and 400 should interpolate to ~30.0 us."""
    base = ProfilingDataSource(interp_data_dir)
    ds = InterpolatingDataSource(base)
    op = _make_op_info(
        torch.ops.aten.mm.default,
        [
            torch.empty(300, 512, device="meta", dtype=torch.bfloat16),
            torch.empty(512, 1024, device="meta", dtype=torch.bfloat16),
        ],
    )
    result = ds.lookup(op)
    assert result is not None
    assert abs(result.latency_us - 30.0) < 0.5
    assert result.source == QuerySource.INTERPOLATED


def test_compute_no_interpolation_wrong_weight(interp_data_dir):
    """Different weight shape (not just seq dim) should not interpolate."""
    base = ProfilingDataSource(interp_data_dir)
    ds = InterpolatingDataSource(base)
    op = _make_op_info(
        torch.ops.aten.mm.default,
        [
            torch.empty(150, 512, device="meta", dtype=torch.bfloat16),
            torch.empty(512, 2048, device="meta", dtype=torch.bfloat16),
        ],
    )
    result = ds.lookup(op)
    assert result is None, "Can't interpolate when non-seq dims differ"


def test_comm_interpolation(interp_data_dir):
    """Communication: interpolate by message_bytes."""
    base = ProfilingDataSource(interp_data_dir)
    ds = InterpolatingDataSource(base)
    # 150000 bytes between 100000 and 200000 -> 150.0 us
    # Need tensor with 150000 / 2 = 75000 elements (BF16 = 2 bytes)
    op = _make_op_info(
        torch.ops.tensor_cast.all_reduce.default,
        [
            torch.empty(75000, device="meta", dtype=torch.bfloat16),
            0,
            list(range(16)),
        ],
    )
    result = ds.lookup(op)
    assert result is not None, "Should interpolate comm by message_bytes"
    assert abs(result.latency_us - 150.0) < 1.0
    assert result.source == QuerySource.INTERPOLATED


def test_comm_interpolation_comes_from_base_not_wrapper(interp_data_dir):
    """Communication ops should not enter InterpolatingDataSource's wrapper fallback."""
    base = ProfilingDataSource(interp_data_dir)
    ds = InterpolatingDataSource(base)
    called = False

    def fail_if_called(_op):
        nonlocal called
        called = True
        return QueryResult(
            latency_us=1.0,
            confidence=0.1,
            source=QuerySource.INTERPOLATED,
            details={"unexpected": True},
        )

    ds._interpolate = fail_if_called
    op = _make_op_info(
        torch.ops.tensor_cast.all_reduce.default,
        [
            torch.empty(75000, device="meta", dtype=torch.bfloat16),
            0,
            list(range(16)),
        ],
    )

    result = ds.lookup(op)

    assert not called
    assert result is not None
    assert result.source == QuerySource.INTERPOLATED
    assert result.details.get("unexpected") is None


def test_comm_base_miss_does_not_enter_wrapper_multidim(interp_data_dir, monkeypatch):
    """Communication base miss should not fall into compute/attention multidim paths."""
    base = ProfilingDataSource(interp_data_dir)
    ds = InterpolatingDataSource(base)

    monkeypatch.setattr(base, "lookup", lambda _op: None)
    monkeypatch.setattr(
        ds,
        "_interpolate_compute_multidim",
        lambda *_args, **_kwargs: pytest.fail("comm op entered compute multidim interpolation"),
    )
    monkeypatch.setattr(
        ds,
        "_interpolate_attention_multidim",
        lambda *_args, **_kwargs: pytest.fail("comm op entered attention multidim interpolation"),
    )

    op = _make_op_info(
        torch.ops.tensor_cast.all_reduce.default,
        [
            torch.empty(75000, device="meta", dtype=torch.bfloat16),
            0,
            list(range(16)),
        ],
    )

    assert ds.lookup(op) is None


def test_attention_interpolation_linear(interp_data_dir):
    """Attention: interpolate avg_seq_len linearly by default.

    CSV has: seq=1000 -> 100us, seq=4000 -> 1600us
    For seq=2000: linear t = (2000 - 1000) / (4000 - 1000) = 1/3
    linear_interp = 100 + 1/3 * (1600 - 100) = 600
    """
    base = ProfilingDataSource(interp_data_dir)
    ds = InterpolatingDataSource(base)
    op = _make_op_info(
        torch.ops.tensor_cast.attention.default,
        [
            torch.empty(1, 512, device="meta", dtype=torch.bfloat16),  # query
            torch.empty(16, 128, 4, 128, device="meta", dtype=torch.bfloat16),  # key
            torch.empty(16, 128, 4, 128, device="meta", dtype=torch.bfloat16),  # value
            None,
            None,
            None,
            torch.tensor([2000], dtype=torch.int64),  # seq_lens (CPU)
            torch.tensor([1], dtype=torch.int64),  # query_lens
        ],
    )
    result = ds.lookup(op)
    assert result is not None, "Should interpolate attention linearly by default"
    assert result.source == QuerySource.INTERPOLATED
    assert result.latency_us == pytest.approx(600.0)
    assert list(result.details["axis_boundary"]) == ["seq"]


def test_attention_param_interpolation_uses_linear_default(interp_data_dir):
    ds = InterpolatingDataSource(ProfilingDataSource(interp_data_dir))

    result = ds._interpolate_attention_by_params(
        "FusedInferAttentionScore",
        {
            "q_shape_3d": (1, 4, 128),
            "avg_seq_len": 2000,
            "batch_size": 1,
            "sparse_mode": 3,
            "num_kv_heads": 4,
            "input_layout": "TND",
        },
        "DT_BF16",
    )

    assert result is not None
    assert result.source == QuerySource.INTERPOLATED
    assert result.latency_us == pytest.approx(600.0)


def test_unmapped_op_no_interpolation(interp_data_dir):
    """Unmapped ops should still return None."""
    base = ProfilingDataSource(interp_data_dir)
    ds = InterpolatingDataSource(base)
    op = _make_op_info(
        torch.ops.aten.mul.Tensor,
        [
            torch.empty(100, 512, device="meta", dtype=torch.bfloat16),
            torch.empty(100, 512, device="meta", dtype=torch.bfloat16),
        ],
    )
    result = ds.lookup(op)
    assert result is None


def _make_elementwise_op_info(func, input_tensors, out_tensor):
    """Create mock OpInvokeInfo with .out set to output tensor."""
    mock = MagicMock()
    mock.func = func
    mock.args = tuple(input_tensors)
    mock.kwargs = {}
    mock.out = out_tensor
    return mock


def test_interpolate_elementwise_basic(interp_data_dir):
    """Interpolate (192,7168) BF16 between (128,7168)→6.0us and (256,7168)→12.0us → ~9.0us.

    Linear interp: t = (192-128)/(256-128) = 64/128 = 0.5
    latency = 6.0 + 0.5 * (12.0 - 6.0) = 9.0 us
    """
    base = ProfilingDataSource(interp_data_dir)
    ds = InterpolatingDataSource(base)
    out = torch.empty(192, 7168, device="meta", dtype=torch.bfloat16)
    op = _make_elementwise_op_info(
        torch.ops.aten.add.Tensor,
        [
            torch.empty(192, 7168, device="meta", dtype=torch.bfloat16),
            torch.empty(192, 7168, device="meta", dtype=torch.bfloat16),
        ],
        out,
    )
    result = ds.lookup(op)
    assert result is not None, "Should interpolate elementwise (192,7168) between boundary rows"
    assert abs(result.latency_us - 9.0) < 0.5, f"Expected ~9.0 us, got {result.latency_us}"
    assert result.source == QuerySource.INTERPOLATED
    assert result.confidence == 0.7


def test_interpolate_elementwise_dtype_scaled(interp_data_dir):
    """FP32 target, BF16 CSV: candidates are dtype-scaled before 1D interpolation.

    CSV rows: (128,7168) BF16 → 6.0us, (256,7168) BF16 → 12.0us
    FP32 is 4 bytes, BF16 is 2 bytes → scale factor 2.0
    Scaled candidates: (128,7168) → 12.0us, (256,7168) → 24.0us
    Interpolate at 192: t=0.5, latency = 12.0 + 0.5*(24.0-12.0) = 18.0us
    """
    base = ProfilingDataSource(interp_data_dir)
    ds = InterpolatingDataSource(base)
    out = torch.empty(192, 7168, device="meta", dtype=torch.float32)
    op = _make_elementwise_op_info(
        torch.ops.aten.add.Tensor,
        [
            torch.empty(192, 7168, device="meta", dtype=torch.float32),
            torch.empty(192, 7168, device="meta", dtype=torch.float32),
        ],
        out,
    )
    result = ds.lookup(op)
    assert result is not None, "Should interpolate FP32 target with dtype-scaled BF16 candidates"
    assert abs(result.latency_us - 18.0) < 1.0, f"Expected ~18.0 us, got {result.latency_us}"
    assert result.source == QuerySource.INTERPOLATED
    assert result.confidence == 0.7, f"1D interpolation should have fixed confidence=0.7, got {result.confidence}"


def test_interpolate_elementwise_hidden_dim_filter(interp_data_dir):
    """(M,7168) rows don't mix with (M,1536) rows during interpolation.

    Target: (192,7168) — hidden dim 7168
    CSV has rows for both (M,7168) and (M,1536).
    Only the (M,7168) rows should be candidates; (M,1536) must be filtered out.
    Interpolation should still succeed using only (128,7168) and (256,7168).
    """
    base = ProfilingDataSource(interp_data_dir)
    ds = InterpolatingDataSource(base)
    out = torch.empty(192, 7168, device="meta", dtype=torch.bfloat16)
    op = _make_elementwise_op_info(
        torch.ops.aten.add.Tensor,
        [
            torch.empty(192, 7168, device="meta", dtype=torch.bfloat16),
            torch.empty(192, 7168, device="meta", dtype=torch.bfloat16),
        ],
        out,
    )
    result = ds.lookup(op)
    assert result is not None
    # Result should be ~9.0 us (from 7168 rows only), not ~3.0 us (mixed 7168+1536 rows)
    assert abs(result.latency_us - 9.0) < 0.5, (
        f"Expected ~9.0 us (7168 rows only), got {result.latency_us} — hidden dim rows may have been mixed"
    )


def test_elementwise_same_dtype_group_takes_priority(tmp_path):
    data_dir = tmp_path / "elementwise_same_dtype"
    data_dir.mkdir()
    (data_dir / "op_mapping.yaml").write_text(
        """
version: "test"
operator_mappings:
  "aten.add.Tensor":
    kernel_type: Add
    query_mode: elementwise
""".strip()
    )
    (data_dir / "Add.csv").write_text(
        """
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Duration(us)
"128,7168;128,7168","DT_BF16;DT_BF16","ND;ND","128,7168","DT_BF16","ND",6.0
"256,7168;256,7168","DT_BF16;DT_BF16","ND;ND","256,7168","DT_BF16","ND",12.0
"128,7168;128,7168","FLOAT;FLOAT","ND;ND","128,7168","FLOAT","ND",30.0
"256,7168;256,7168","FLOAT;FLOAT","ND;ND","256,7168","FLOAT","ND",60.0
""".strip()
    )
    ds = InterpolatingDataSource(ProfilingDataSource(data_dir))
    out = torch.empty(192, 7168, device="meta", dtype=torch.float32)
    op = _make_elementwise_op_info(
        torch.ops.aten.add.Tensor,
        [
            torch.empty(192, 7168, device="meta", dtype=torch.float32),
            torch.empty(192, 7168, device="meta", dtype=torch.float32),
        ],
        out,
    )

    result = ds.lookup(op)

    assert result is not None
    assert result.latency_us == pytest.approx(45.0)
    assert result.details["dtype_attempt"] == "same_dtype"
    assert result.details["exact_fields"]["csv_output_dtype"] == "FLOAT"
    assert not result.details["dtype_scaled"]


def test_elementwise_mixed_csv_dtypes_use_separate_candidate_groups(tmp_path):
    data_dir = tmp_path / "elementwise_dtype_groups"
    data_dir.mkdir()
    (data_dir / "op_mapping.yaml").write_text('version: "test"')
    (data_dir / "Add.csv").write_text(
        """
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Duration(us)
"128,7168","DT_BF16","ND","128,7168","DT_BF16","ND",6.0
"256,7168","DT_BF16","ND","256,7168","DT_BF16","ND",12.0
"128,7168","FLOAT","ND","128,7168","FLOAT","ND",30.0
"256,7168","FLOAT","ND","256,7168","FLOAT","ND",60.0
""".strip()
    )
    ds = InterpolatingDataSource(ProfilingDataSource(data_dir))
    index = ds._get_elementwise_index("Add", "FLOAT")
    assert index is not None

    input_signature = index.points[0].row_meta["input_signature"]
    groups = InterpolatingDataSource._elementwise_candidate_group_attempts(
        index, "Add", (7168,), input_signature, "FLOAT"
    )

    assert [label for label, _group in groups] == ["same_dtype", "scaled_dtype"]
    assert [dict(group.regime_key)["csv_output_dtype"] for _label, group in groups] == ["FLOAT", "DT_BF16"]
    assert all(len({point.input_dtypes[0] for point in group.points}) == 1 for _label, group in groups)


def test_elementwise_dtype_scaled_cache_is_scoped_to_target_dtype(interp_data_dir):
    ds = InterpolatingDataSource(ProfilingDataSource(interp_data_dir))
    bf16_out = torch.empty(192, 7168, device="meta", dtype=torch.bfloat16)
    fp32_out = torch.empty(192, 7168, device="meta", dtype=torch.float32)
    bf16_op = _make_elementwise_op_info(
        torch.ops.aten.add.Tensor,
        [
            torch.empty(192, 7168, device="meta", dtype=torch.bfloat16),
            torch.empty(192, 7168, device="meta", dtype=torch.bfloat16),
        ],
        bf16_out,
    )
    fp32_op = _make_elementwise_op_info(
        torch.ops.aten.add.Tensor,
        [
            torch.empty(192, 7168, device="meta", dtype=torch.float32),
            torch.empty(192, 7168, device="meta", dtype=torch.float32),
        ],
        fp32_out,
    )

    bf16_result = ds.lookup(bf16_op)
    fp32_result = ds.lookup(fp32_op)

    assert bf16_result is not None
    assert fp32_result is not None
    assert bf16_result.latency_us == pytest.approx(9.0)
    assert fp32_result.latency_us == pytest.approx(18.0)
    assert fp32_result.details["dtype_scaled"]
    assert len(ds._elementwise_index_cache) == 2

    bf16_index = ds._get_elementwise_index("Add", "DT_BF16")
    ds._policy_hash = "changed-policy"
    changed_policy_bf16_index = ds._get_elementwise_index("Add", "DT_BF16")

    assert bf16_index is not None
    assert changed_policy_bf16_index is not None
    assert changed_policy_bf16_index is not bf16_index
    assert len(ds._elementwise_index_cache) == 3


class TestFiaRawNoInterpolation:
    """Spec §6.2 F3: InterpolatingDataSource skips raw FIA CSV interpolation."""

    def test_f3_raw_csv_no_interpolation(self):
        """FIA raw CSV MISS → InterpolatingDataSource returns None (no interpolation)."""
        fixture_dir = Path(__file__).parents[2] / "benchmark" / "ops" / "perf_database" / "fixtures" / "fia_raw_test"
        assert fixture_dir.exists()
        base = ProfilingDataSource(fixture_dir)
        interp = InterpolatingDataSource(base)

        # Build attention op_info with a shape that won't match any CSV row.
        # Raw CSV rows have query shapes like (128,4,128), (336,4,128), etc.
        # Use num_tokens=999 which doesn't appear in the fixture.
        op = _make_op_info(
            torch.ops.tensor_cast.attention.default,
            [
                torch.empty(999, 512, device="meta", dtype=torch.bfloat16),  # query
                torch.empty(12307, 128, 128, device="meta", dtype=torch.bfloat16),  # key
                torch.empty(12307, 128, 128, device="meta", dtype=torch.bfloat16),  # value
                None,
                None,
                None,
                torch.tensor([100] * 999, dtype=torch.int64),  # seq_lens
                None,
            ],
        )

        result = interp.lookup(op)
        # Raw CSV cannot be interpolated on structured attention dims — must return None
        assert result is None, f"InterpolatingDataSource must not interpolate raw FIA CSV, got {result}"


def test_partial_falls_through_to_interpolation(interp_data_dir):
    """PARTIAL from base should not block interpolation attempt.

    When base returns PARTIAL (e.g., composite with some sub-kernel misses),
    InterpolatingDataSource should try interpolation first. If interpolation
    succeeds, it should return the interpolated result instead of PARTIAL.
    """
    base = ProfilingDataSource(interp_data_dir)
    ds = InterpolatingDataSource(base)

    # Mock base.lookup to return PARTIAL
    partial_result = QueryResult(
        latency_us=50.0,
        confidence=0.5,
        source=QuerySource.PARTIAL,
        details={"hit_kernels": ["MatMulV2"], "missed_kernels": ["SomeKernel"]},
    )
    base.lookup = lambda op: partial_result

    # Also mock _interpolate to return INTERPOLATED with better result
    interp_result = QueryResult(
        latency_us=15.0,
        confidence=0.7,
        source=QuerySource.INTERPOLATED,
        details={"kernel_type": "MatMulV2", "method": "linear_1d"},
    )
    ds._interpolate = lambda op, **_kwargs: interp_result

    op = _make_op_info(
        torch.ops.aten.mm.default,
        [
            torch.empty(150, 512, device="meta", dtype=torch.bfloat16),
            torch.empty(512, 1024, device="meta", dtype=torch.bfloat16),
        ],
    )
    result = ds.lookup(op)
    assert result is not None
    assert result.source == QuerySource.INTERPOLATED, (
        f"Expected INTERPOLATED to take priority over PARTIAL, got {result.source}"
    )


def test_lookup_adds_shape_match_info_without_mutating_interpolation_result(interp_data_dir):
    base = ProfilingDataSource(interp_data_dir)
    ds = InterpolatingDataSource(base)
    base.lookup = lambda op: None
    interp_result = QueryResult(
        latency_us=15.0,
        confidence=0.7,
        source=QuerySource.INTERPOLATED,
        details={"kernel_type": "MatMulV2", "method": "linear_1d"},
        shape_match_info=None,
    )
    ds._interpolate = lambda op, **_kwargs: interp_result
    op = _make_op_info(
        torch.ops.aten.mm.default,
        [
            torch.empty(150, 512, device="meta", dtype=torch.bfloat16),
            torch.empty(512, 1024, device="meta", dtype=torch.bfloat16),
        ],
    )

    result = ds.lookup(op)

    assert result is not None
    assert result is not interp_result
    assert result.shape_match_info is not None
    assert interp_result.shape_match_info is None


def test_partial_returns_none_when_interpolation_fails(interp_data_dir):
    """When base returns PARTIAL and interpolation fails, fall back to analytic."""
    base = ProfilingDataSource(interp_data_dir)
    ds = InterpolatingDataSource(base)

    partial_result = QueryResult(
        latency_us=50.0,
        confidence=0.5,
        source=QuerySource.PARTIAL,
        details={"hit_kernels": ["MatMulV2"], "missed_kernels": ["SomeKernel"]},
    )
    base.lookup = lambda op: partial_result
    ds._interpolate = lambda op, **_kwargs: None  # interpolation fails

    op = _make_op_info(
        torch.ops.aten.mm.default,
        [
            torch.empty(150, 512, device="meta", dtype=torch.bfloat16),
            torch.empty(512, 1024, device="meta", dtype=torch.bfloat16),
        ],
    )
    result = ds.lookup(op)
    assert result is None


def test_candidate_latency_scans_fallback_after_zero_primary_column():
    ds = object.__new__(InterpolatingDataSource)
    row = pd.Series(
        {
            "Average Duration(us)": 0.0,
            "Profiling Average Duration(us)": 12.5,
        }
    )

    latency, meta = ds._candidate_latency(row, "Average Duration(us)")

    assert latency == pytest.approx(12.5)
    assert meta["latency_column"] == "Profiling Average Duration(us)"
    assert meta["latency_selection"] == "fallback_column"


def test_candidate_latency_uses_wrapper_median_fallback_column():
    ds = object.__new__(InterpolatingDataSource)
    row = pd.Series(
        {
            "Average Duration(us)": 0.0,
            "Profiling Average Duration(us)": "bad",
            "Profiling Median Duration(us)": 12.0,
            "Duration(us)": 99.0,
        }
    )

    latency, meta = ds._candidate_latency(row, "Average Duration(us)")

    assert latency == pytest.approx(12.0)
    assert meta["latency_column"] == "Profiling Median Duration(us)"
    assert meta["latency_selection"] == "fallback_column"


def test_candidate_latency_rejects_non_finite_and_non_positive_cells():
    ds = object.__new__(InterpolatingDataSource)
    row = pd.Series(
        {
            "Average Duration(us)": float("nan"),
            "Profiling Average Duration(us)": float("inf"),
            "Profiling Median Duration(us)": -1.0,
            "Median Duration(us)": 0.0,
        }
    )

    latency, meta = ds._candidate_latency(row, "Average Duration(us)")

    assert latency is None
    assert meta["latency_rejected_reason"] == "latency_invalid"


def test_base_row_latency_pair_keeps_priority_and_skips_zero_primary_column():
    row = pd.Series(
        {
            "Average Duration(us)": 0.0,
            "Profiling Average Duration(us)": 12.5,
            "Duration(us)": 99.0,
        }
    )

    result = ProfilingDataSource._row_latency_pair(row, "Average Duration(us)")

    assert result == ("Profiling Average Duration(us)", 12.5)


def _write_minimal_matmul_dir(data_dir, csv_body):
    data_dir.mkdir()
    (data_dir / "op_mapping.yaml").write_text(
        """
version: "test"
operator_mappings:
  "aten.mm.default":
    kernel_type: MatMulV2
""".strip()
    )
    (data_dir / "MatMulV2.csv").write_text(csv_body.strip())


def _matmul_probe_op():
    return _make_op_info(
        torch.ops.aten.mm.default,
        [
            torch.empty(150, 64, device="meta", dtype=torch.bfloat16),
            torch.empty(64, 256, device="meta", dtype=torch.bfloat16),
        ],
    )


def test_empty_candidate_csv_returns_none_without_error(tmp_path):
    data_dir = tmp_path / "empty_candidate_csv"
    _write_minimal_matmul_dir(
        data_dir,
        "Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Duration(us)",
    )
    ds = InterpolatingDataSource(ProfilingDataSource(data_dir))

    result = ds.lookup(_matmul_probe_op())

    assert result is None


def test_single_candidate_csv_returns_candidate_shortage(tmp_path):
    data_dir = tmp_path / "single_candidate_csv"
    _write_minimal_matmul_dir(
        data_dir,
        """
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Duration(us)
"100,64;64,256","DT_BF16;DT_BF16","ND;ND","100,256","DT_BF16","ND",10.0
""",
    )
    ds = InterpolatingDataSource(ProfilingDataSource(data_dir))

    result = ds.lookup(_matmul_probe_op())

    assert result is None
    assert ds.last_miss_reason in {"insufficient_filtered_candidates", "compute_multidim_interpolation_failed"}


def test_all_invalid_latency_csv_returns_none_without_error(tmp_path):
    data_dir = tmp_path / "invalid_latency_csv"
    _write_minimal_matmul_dir(
        data_dir,
        """
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Duration(us)
"100,64;64,256","DT_BF16;DT_BF16","ND;ND","100,256","DT_BF16","ND",0.0
"200,64;64,256","DT_BF16;DT_BF16","ND;ND","200,256","DT_BF16","ND",bad
""",
    )
    ds = InterpolatingDataSource(ProfilingDataSource(data_dir))

    result = ds.lookup(_matmul_probe_op())

    assert result is None
    assert ds.last_miss_reason in {
        "regime_key_unmatched",
        "insufficient_filtered_candidates",
        "compute_multidim_interpolation_failed",
    }
