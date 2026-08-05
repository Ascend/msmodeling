import shutil
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import pandas as pd
import torch
from tensor_cast.performance_model.profiling_database.data_source import QueryResult, QuerySource
from tensor_cast.performance_model.profiling_database.interpolating_data_source import InterpolatingDataSource
from tensor_cast.performance_model.profiling_database.profiling_data_source import (
    COMPOSITE_DECOMPOSERS,
    ProfilingDataSource,
)


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


def test_elementwise_axes_reject_negative_dimensions():
    assert InterpolatingDataSource._elementwise_axes_from_shapes([(-1, 3)], (2, 3)) is None
    assert InterpolatingDataSource._elementwise_axes_from_shapes([(2, 3)], (-1, 3)) is None


def test_concat_uses_output_numel_for_generic_compute_interpolation(tmp_path, monkeypatch):
    data_dir = tmp_path / "concat_output_numel"
    data_dir.mkdir()
    _write_text(
        data_dir / "op_mapping.yaml",
        """
version: "test"
interpolation_policy:
  kernel_overrides:
    ConcatD:
      generic_compute:
        axis: output_numel
operator_mappings:
  "aten.cat.default":
    kernel_type: ConcatD
""",
    )
    _write_text(
        data_dir / "ConcatD.csv",
        """
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Duration(us)
"50,64;50,64","DT_BF16;DT_BF16","ND;ND","100,64","DT_BF16","ND",10.0
"100,64;100,64","DT_BF16;DT_BF16","ND;ND","200,64","DT_BF16","ND",20.0
""",
    )
    ds = InterpolatingDataSource(ProfilingDataSource(data_dir))
    monkeypatch.setattr(ds.base, "lookup", lambda _op: None)
    lhs = torch.empty(75, 64, device="meta", dtype=torch.bfloat16)
    rhs = torch.empty(75, 64, device="meta", dtype=torch.bfloat16)
    out = torch.empty(150, 64, device="meta", dtype=torch.bfloat16)

    result = ds.lookup(_make_op_info("aten.cat.default", [[lhs, rhs], 0], out))

    assert result is not None
    assert result.source == QuerySource.INTERPOLATED
    assert result.latency_us == pytest.approx(15.0)
    assert result.details["axes"] == ["output_numel"]


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


def test_attention_runtime_workload_uses_query_weighted_kv_length():
    workload = InterpolatingDataSource._attention_runtime_workload(
        q_tokens=6,
        query_lengths=(2, 4),
        kv_lengths=(2, 8),
    )

    assert workload == {
        "q_tokens": 6.0,
        "effective_kv_len": 6.0,
        "phase": "mixed",
        "batch_size": 2,
    }


def test_attention_runtime_workload_rejects_negative_kv_length():
    assert (
        InterpolatingDataSource._attention_runtime_workload(
            q_tokens=2,
            query_lengths=(2, 0),
            kv_lengths=(4, -1),
        )
        is None
    )


def test_runtime_attention_regime_rejects_malformed_numeric_fields():
    params = {
        "sparse_mode": 3,
        "num_kv_heads": 1,
        "input_layout": "TND",
        "topk": "invalid",
        "block_size": 128,
        "num_heads": 32,
        "cache_layout": "PA_BSND",
        "kv_cache_mode": "paged",
    }

    fields = InterpolatingDataSource._runtime_attention_regime_fields(
        "LightningIndexer",
        "DT_BF16",
        (1, 32, 128),
        {"phase": "decode"},
        params,
        include_sparse_fields=False,
    )

    assert fields is None


def test_query_lengths_from_cumulative_offsets_fails_closed():
    assert InterpolatingDataSource._query_lengths_from_cumulative(6, (2, 6)) == (2, 4)
    assert InterpolatingDataSource._query_lengths_from_cumulative(6, (2, 5)) is None
    assert InterpolatingDataSource._query_lengths_from_cumulative(6, (3, 2, 6)) is None


def test_real_lightning_indexer_candidate_uses_runtime_workload():
    ds = InterpolatingDataSource(ProfilingDataSource(_REAL_V018_DATA_DIR))
    df = pd.read_csv(_REAL_V018_DATA_DIR / "LightningIndexer.csv")
    matching_rows = df.loc[df["Runtime case_id"] == "li_8c110488d58d621d"]
    assert len(matching_rows) == 1
    complete_row = matching_rows.iloc[0]

    point, reason = ds._candidate_from_lightning_indexer_row(
        complete_row,
        "LightningIndexer",
        "Average Duration(us)",
        0,
    )

    assert reason is None
    assert point is not None
    assert point.axes == {"q_tokens": 1.0, "effective_kv_len": 256.0}
    assert dict(point.regime_key)["phase"] == "decode"


def test_real_sparse_attention_candidate_uses_runtime_workload():
    ds = InterpolatingDataSource(ProfilingDataSource(_REAL_V018_DATA_DIR))
    df = pd.read_csv(_REAL_V018_DATA_DIR / "SparseFlashAttention.csv")
    matching_rows = df.loc[df["Runtime case_id"] == "sfa_d48f8c82b4169093"]
    assert len(matching_rows) == 1
    complete_row = matching_rows.iloc[0]

    point, reason = ds._candidate_from_sparse_attention_row(
        complete_row,
        "SparseFlashAttention",
        "Average Duration(us)",
        0,
    )

    assert reason is None
    assert point is not None
    assert point.axes == {"q_tokens": 1.0, "effective_kv_len": 5769.0}
    assert dict(point.regime_key)["phase"] == "decode"


def test_real_lightning_indexer_interpolates_effective_kv_length():
    ds = InterpolatingDataSource(ProfilingDataSource(_REAL_V018_DATA_DIR))
    params = {
        "q_shape_3d": (1, 32, 128),
        "sparse_mode": 3,
        "num_kv_heads": 1,
        "input_layout": "TND",
        "topk": 2048,
        "block_size": 128,
        "num_heads": 32,
        "cache_layout": "PA_BSND",
        "kv_cache_mode": "paged",
        "actual_seq_lengths_values": [1],
        "actual_seq_lengths_kv_values": [3000],
    }

    result = ds._interpolate_attention_by_params_one("LightningIndexer", params, "DT_BF16")

    assert result is not None
    assert result.source == QuerySource.INTERPOLATED
    assert result.details["axes"] == ["effective_kv_len"]
    assert result.details["phase"] == "decode"


def test_real_lightning_indexer_interpolates_runtime_workload_in_2d():
    ds = InterpolatingDataSource(ProfilingDataSource(_REAL_V018_DATA_DIR))
    params = {
        "q_shape_3d": (5, 32, 128),
        "sparse_mode": 3,
        "num_kv_heads": 1,
        "input_layout": "TND",
        "topk": 2048,
        "block_size": 128,
        "num_heads": 32,
        "cache_layout": "PA_BSND",
        "kv_cache_mode": "paged",
        "actual_seq_lengths_values": [2, 5],
        "actual_seq_lengths_kv_values": [3000, 3000],
    }

    result = ds._interpolate_attention_by_params_one("LightningIndexer", params, "DT_BF16")

    assert result is not None
    assert result.source == QuerySource.INTERPOLATED
    assert result.details["interpolation_dim"] == 2
    assert result.details["axes"] == ["q_tokens", "effective_kv_len"]
    assert result.details["phase"] == "mixed"


def test_real_lightning_indexer_rejects_runtime_topk_regime_mismatch():
    ds = InterpolatingDataSource(ProfilingDataSource(_REAL_V018_DATA_DIR))
    params = {
        "q_shape_3d": (1, 32, 128),
        "sparse_mode": 3,
        "num_kv_heads": 1,
        "input_layout": "TND",
        "topk": 1024,
        "block_size": 128,
        "num_heads": 32,
        "cache_layout": "PA_BSND",
        "kv_cache_mode": "paged",
        "actual_seq_lengths_values": [1],
        "actual_seq_lengths_kv_values": [3000],
    }

    result = ds._interpolate_attention_by_params_one("LightningIndexer", params, "DT_BF16")

    assert result is None
    assert ds.last_miss_reason == "lightning_indexer_interpolation_failed"
    assert ds.last_miss_details["attempts"][0]["status"] == "regime_key_unmatched"
    assert ds.last_miss_details["attempts"][0]["target_regime"]["topk"] == 1024


@pytest.mark.parametrize(
    ("field", "value", "expected_regime_field", "expected_regime_value"),
    [
        ("sparse_mode", 99, "sparse_mode", 99),
        ("num_kv_heads", 99, "kv_heads", 99),
        ("input_layout", "INVALID", "input_layout", "INVALID"),
        ("block_size", 64, "block_size", 64),
        ("num_heads", 31, "num_heads", 31),
        ("cache_layout", "INVALID", "cache_layout", "INVALID"),
        ("kv_cache_mode", "INVALID", "kv_cache_mode", "INVALID"),
        ("q_shape_3d", (1, 32, 127), "head_dim", 127),
    ],
)
def test_real_lightning_indexer_rejects_runtime_regime_mismatch(
    field,
    value,
    expected_regime_field,
    expected_regime_value,
):
    ds = InterpolatingDataSource(ProfilingDataSource(_REAL_V018_DATA_DIR))
    params = {
        "q_shape_3d": (1, 32, 128),
        "sparse_mode": 3,
        "num_kv_heads": 1,
        "input_layout": "TND",
        "topk": 2048,
        "block_size": 128,
        "num_heads": 32,
        "cache_layout": "PA_BSND",
        "kv_cache_mode": "paged",
        "actual_seq_lengths_values": [1],
        "actual_seq_lengths_kv_values": [3000],
    }
    params[field] = value

    result = ds._interpolate_attention_by_params_one("LightningIndexer", params, "DT_BF16")

    assert result is None
    assert ds.last_miss_reason == "lightning_indexer_interpolation_failed"
    attempt = ds.last_miss_details["attempts"][0]
    assert attempt["status"] == "regime_key_unmatched"
    assert attempt["target_regime"][expected_regime_field] == expected_regime_value


def test_real_lightning_indexer_rejects_runtime_dtype_mismatch():
    ds = InterpolatingDataSource(ProfilingDataSource(_REAL_V018_DATA_DIR))
    params = {
        "q_shape_3d": (1, 32, 128),
        "sparse_mode": 3,
        "num_kv_heads": 1,
        "input_layout": "TND",
        "topk": 2048,
        "block_size": 128,
        "num_heads": 32,
        "cache_layout": "PA_BSND",
        "kv_cache_mode": "paged",
        "actual_seq_lengths_values": [1],
        "actual_seq_lengths_kv_values": [3000],
    }

    result = ds._interpolate_attention_by_params_one("LightningIndexer", params, "DT_FLOAT16")

    assert result is None
    assert ds.last_miss_reason == "lightning_indexer_interpolation_failed"
    attempt = ds.last_miss_details["attempts"][0]
    assert attempt["status"] == "regime_key_unmatched"
    assert attempt["target_regime"]["dtype"] == "DT_FLOAT16"


def test_real_sparse_attention_interpolates_effective_kv_length():
    ds = InterpolatingDataSource(ProfilingDataSource(_REAL_V018_DATA_DIR))
    params = {
        "q_shape_3d": (1, 4, 512),
        "sparse_mode": 3,
        "num_kv_heads": 1,
        "input_layout": "TND",
        "topk": 2048,
        "block_size": 128,
        "num_heads": 4,
        "cache_layout": "PA_BSND",
        "kv_cache_mode": "paged",
        "actual_seq_lengths_values": [1],
        "actual_seq_lengths_kv_values": [3000],
        "sparse_block_size": 1,
        "sparse_indices_pattern": "uniform",
        "sparse_indices_valid_count": 2048,
    }

    result = ds._interpolate_attention_by_params_one("SparseFlashAttention", params, "DT_BF16")

    assert result is not None
    assert result.source == QuerySource.INTERPOLATED
    assert result.details["axes"] == ["effective_kv_len"]
    assert result.details["phase"] == "decode"


def test_v018_dsa_indexer_mapping_has_registered_decomposer():
    base = ProfilingDataSource(_REAL_V018_DATA_DIR)
    mapping = base._op_mapping["operator_mappings"]["tensor_cast.dsa_indexer.default"]

    assert mapping["composite"] is True
    assert mapping["decomposer"] is True
    assert "tensor_cast.dsa_indexer.default" in COMPOSITE_DECOMPOSERS


def test_quantized_matmul_target_uses_activation_and_output_shapes(tmp_path):
    data_dir = tmp_path / "quantized_matmul_target"
    data_dir.mkdir()
    _write_text(
        data_dir / "op_mapping.yaml",
        """
version: "test"
operator_mappings:
  "tensor_cast.static_quant_linear.default":
    kernel_type: QuantBatchMatmulV3
    tc_input_count: 2
    compute_subcategory: quantized_matmul
    expected_input_formats: [ND, FRACTAL_NZ]
""",
    )
    ds = InterpolatingDataSource(ProfilingDataSource(data_dir))
    mapping = ds.base._op_mapping["operator_mappings"]["tensor_cast.static_quant_linear.default"]
    op = _make_op_info(
        "tensor_cast.static_quant_linear.default",
        [
            torch.empty((15, 32), device="meta", dtype=torch.int8),
            torch.empty((2, 2, 16, 32), device="meta", dtype=torch.int8),
        ],
        torch.empty((15, 64), device="meta", dtype=torch.bfloat16),
    )

    target = ds._build_compute_target(op, mapping, "QuantBatchMatmulV3")

    assert target is not None
    assert target.axes == {"M": 15.0, "K": 32.0, "N": 64.0}


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


def test_scatter_base_miss_does_not_recover_local_measured_exact(tmp_path):
    data_dir = tmp_path / "scatter_real_csv"
    data_dir.mkdir()
    _write_text(
        data_dir / "op_mapping.yaml",
        """
version: "test"
operator_mappings:
  "tensor_cast.scatter_nd_update_mla.default":
    kernel_type: ScatterNdUpdate
    query_mode: scatter_nd_update_mla
""",
    )
    shutil.copyfile(_REAL_V018_DATA_DIR / "ScatterNdUpdate.csv", data_dir / "ScatterNdUpdate.csv")
    ds = InterpolatingDataSource(ProfilingDataSource(data_dir))
    op = _make_op_info(
        "tensor_cast.scatter_nd_update_mla.default",
        [
            torch.empty(4096, 128, device="meta", dtype=torch.bfloat16),
            torch.empty(214784, 128, device="meta", dtype=torch.bfloat16),
            torch.empty(4096, 1, device="meta", dtype=torch.int32),
        ],
        torch.empty(214784, 128, device="meta", dtype=torch.bfloat16),
    )

    result = ds.lookup(op)

    assert result is None
    assert ds.last_miss_reason == "insufficient_filtered_candidates"


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


def test_real_v018_dynamic_quant_preserves_base_exact_result():
    ds = InterpolatingDataSource(ProfilingDataSource(_REAL_V018_DATA_DIR))
    x = torch.empty(1, 2304, device="meta", dtype=torch.bfloat16)
    op = _make_op_info(
        "tensor_cast.dynamic_quantize_symmetric.default",
        [x, [-1]],
        (
            torch.empty_like(x, dtype=torch.int8),
            torch.empty(1, 1, device="meta", dtype=torch.float32),
        ),
    )

    result = ds.lookup(op)

    assert result is not None
    assert result.source == QuerySource.MEASURED
    assert result.details["kernel_type"] == "DynamicQuant"


def test_real_v018_quantized_matmul_preserves_base_exact_result():
    ds = InterpolatingDataSource(ProfilingDataSource(_REAL_V018_DATA_DIR))
    x = torch.empty(1, 7168, device="meta", dtype=torch.int8)
    weight = torch.empty(2112, 7168, device="meta", dtype=torch.int8)
    op = _make_op_info(
        "tensor_cast.static_quant_linear.default",
        [x, weight],
        torch.empty(1, 2112, device="meta", dtype=torch.bfloat16),
    )

    result = ds.lookup(op)

    assert result is not None
    assert result.source == QuerySource.MEASURED
    assert result.details["kernel_type"] == "QuantBatchMatmulV3"


def test_real_v018_dynamic_quant_interpolates_per_token_m(monkeypatch):
    ds = InterpolatingDataSource(ProfilingDataSource(_REAL_V018_DATA_DIR))
    monkeypatch.setattr(ds.base, "lookup", lambda _op: None)
    x = torch.empty(2528, 6144, device="meta", dtype=torch.bfloat16)
    op = _make_op_info(
        "tensor_cast.dynamic_quantize_symmetric.default",
        [x, [-1]],
        (
            torch.empty_like(x, dtype=torch.int8),
            torch.empty(2528, 1, device="meta", dtype=torch.float32),
        ),
    )

    result = ds.lookup(op)

    assert result is not None
    assert result.source == QuerySource.INTERPOLATED
    assert result.details["axes"] == ["M"]
    assert result.details["exact_fields"]["scale_mode"] == "per_token"


def test_real_v018_quantized_matmul_interpolates_m_bracket(monkeypatch):
    ds = InterpolatingDataSource(ProfilingDataSource(_REAL_V018_DATA_DIR))
    monkeypatch.setattr(ds.base, "lookup", lambda _op: None)
    op = _make_op_info(
        "tensor_cast.static_quant_linear.default",
        [
            torch.empty(2528, 6144, device="meta", dtype=torch.int8),
            torch.empty(2624, 6144, device="meta", dtype=torch.int8),
        ],
        torch.empty(2528, 2624, device="meta", dtype=torch.bfloat16),
    )

    result = ds.lookup(op)

    assert result is not None
    assert result.source == QuerySource.INTERPOLATED
    assert result.details["axes"] == ["M"]
    assert result.details["target"] == {"M": 2528.0}
    assert result.details["effective_filters"] == ["K", "N"]


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
        [(2, 4, 72, 7168), (16, 7168, 4096), (16, 2048, 7168), (2, 4, 72, 4)],
        tokens=576.0,
        input_dtypes=["DT_BF16", "INT8", "INT8", "INT32"],
    )

    assert topk == 4


def test_moe_fused_topk_uses_the_physical_route_input():
    topk = InterpolatingDataSource._moe_fused_topk(
        [(1, 180, 7168), (16, 7168, 4096), (16, 2048, 7168), (180, 4), (180, 8)],
        tokens=180.0,
        input_dtypes=["DT_BF16", "INT8", "INT8", "INT32", "INT64"],
    )

    assert topk == 4


def test_moe_fused_topk_requires_integer_dtype():
    topk = InterpolatingDataSource._moe_fused_topk(
        [(1, 180, 7168), (16, 7168, 4096), (16, 2048, 7168), (180, 4)],
        tokens=180.0,
        input_dtypes=["DT_BF16", "INT8", "INT8", "UINT32"],
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


def test_direct_lightning_indexer_legacy_cache_dtype_contract_requires_runtime_leaf(tmp_path):
    data_dir = tmp_path / "lightning_indexer_cache_dtype"
    data_dir.mkdir()
    _write_text(
        data_dir / "op_mapping.yaml",
        """
version: "test"
operator_mappings:
  "tensor_cast.quant_lightning_indexer.default":
    kernel_type: LightningIndexer
    query_mode: attention_lightning_indexer
""",
    )
    _write_text(
        data_dir / "LightningIndexer.csv",
        """
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Duration(us)
"100,8,128;64,16;100,8;1","DT_BF16;DT_BF16;DT_BF16;INT32","ND;ND;ND;ND","100,5","INT32","ND",10.0
"200,8,128;64,16;200,8;1","DT_BF16;DT_BF16;DT_BF16;INT32","ND;ND;ND;ND","200,5","INT32","ND",20.0
""",
    )
    ds = InterpolatingDataSource(ProfilingDataSource(data_dir))
    op = _make_op_info(
        "tensor_cast.quant_lightning_indexer.default",
        [
            torch.empty(1, 150, 8, 128, device="meta", dtype=torch.bfloat16),
            torch.empty(1, 150, 8, device="meta", dtype=torch.bfloat16),
            torch.empty(64, 16, device="meta", dtype=torch.float32),
            5,
            1,
            torch.empty(1, device="meta", dtype=torch.int32),
            torch.empty(1, device="meta", dtype=torch.int32),
        ],
        torch.empty(1, 150, 5, device="meta", dtype=torch.int64),
    )

    assert ds.lookup(op) is None
    assert ds.last_miss_reason == "runtime_attention_leaf_required"


def test_direct_lightning_indexer_legacy_cache_shape_contract_requires_runtime_leaf(tmp_path):
    data_dir = tmp_path / "lightning_indexer_cache_tail"
    data_dir.mkdir()
    _write_text(
        data_dir / "op_mapping.yaml",
        """
version: "test"
operator_mappings:
  "tensor_cast.quant_lightning_indexer.default":
    kernel_type: LightningIndexer
    query_mode: attention_lightning_indexer
""",
    )
    _write_text(
        data_dir / "LightningIndexer.csv",
        """
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Duration(us)
"100,8,128;64,16;100,8;1","DT_BF16;DT_BF16;DT_BF16;INT32","ND;ND;ND;ND","100,5","INT32","ND",10.0
"200,8,128;64,16;200,8;1","DT_BF16;DT_BF16;DT_BF16;INT32","ND;ND;ND;ND","200,5","INT32","ND",20.0
""",
    )
    ds = InterpolatingDataSource(ProfilingDataSource(data_dir))
    op = _make_op_info(
        "tensor_cast.quant_lightning_indexer.default",
        [
            torch.empty(1, 150, 8, 128, device="meta", dtype=torch.bfloat16),
            torch.empty(1, 150, 8, device="meta", dtype=torch.bfloat16),
            torch.empty(64, 32, device="meta", dtype=torch.bfloat16),
            5,
            1,
            torch.empty(1, device="meta", dtype=torch.int32),
            torch.empty(1, device="meta", dtype=torch.int32),
        ],
        torch.empty(1, 150, 5, device="meta", dtype=torch.int64),
    )

    assert ds.lookup(op) is None
    assert ds.last_miss_reason == "runtime_attention_leaf_required"


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


def test_direct_lightning_indexer_legacy_sequence_contract_requires_runtime_leaf(tmp_path):
    data_dir = tmp_path / "lightning_indexer_2d"
    data_dir.mkdir()
    _write_text(
        data_dir / "op_mapping.yaml",
        """
version: "test"
operator_mappings:
  "tensor_cast.lightning_indexer.default":
    kernel_type: LightningIndexer
    query_mode: attention_lightning_indexer
""",
    )
    rows = [
        '"100,8,128;64,16;1;2","DT_BF16;INT32;INT32;INT32","ND;ND;ND;ND","100,5","INT32","ND",10.0',
        '"200,8,128;64,16;1;2","DT_BF16;INT32;INT32;INT32","ND;ND;ND;ND","200,5","INT32","ND",20.0',
        '"100,8,128;128,16;1;2","DT_BF16;INT32;INT32;INT32","ND;ND;ND;ND","100,5","INT32","ND",30.0',
        '"200,8,128;128,16;1;2","DT_BF16;INT32;INT32;INT32","ND;ND;ND;ND","200,5","INT32","ND",40.0',
    ]
    _write_text(
        data_dir / "LightningIndexer.csv",
        "\n".join(
            [
                "Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Duration(us)",
                *rows,
            ]
        ),
    )
    ds = InterpolatingDataSource(ProfilingDataSource(data_dir))
    out = torch.empty(125, 5, device="meta", dtype=torch.int32)
    op = _make_op_info(
        "tensor_cast.lightning_indexer.default",
        [
            torch.empty(125, 8, 128, device="meta", dtype=torch.bfloat16),
            torch.empty(80, 16, device="meta", dtype=torch.int32),
            torch.empty(1, device="meta", dtype=torch.int32),
            torch.empty(2, device="meta", dtype=torch.int32),
        ],
        out,
    )

    result = ds.lookup(op)

    assert result is None
    assert ds.last_miss_reason == "runtime_attention_leaf_required"


def test_direct_lightning_indexer_legacy_latency_contract_requires_runtime_leaf(tmp_path):
    data_dir = tmp_path / "lightning_indexer_latency_source_mixed"
    data_dir.mkdir()
    _write_text(
        data_dir / "op_mapping.yaml",
        """
version: "test"
operator_mappings:
  "tensor_cast.lightning_indexer.default":
    kernel_type: LightningIndexer
    query_mode: attention_lightning_indexer
""",
    )
    _write_text(
        data_dir / "LightningIndexer.csv",
        """
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Average Duration(us),Duration(us)
"100,8,128;64,16;1;2","DT_BF16;INT32;INT32;INT32","ND;ND;ND;ND","100,5","INT32","ND",10.0,10.0
"200,8,128;64,16;1;2","DT_BF16;INT32;INT32;INT32","ND;ND;ND;ND","200,5","INT32","ND",0.0,20.0
""",
    )
    ds = InterpolatingDataSource(ProfilingDataSource(data_dir))
    op = _make_op_info(
        "tensor_cast.lightning_indexer.default",
        [
            torch.empty(150, 8, 128, device="meta", dtype=torch.bfloat16),
            torch.empty(64, 16, device="meta", dtype=torch.int32),
            torch.empty(1, device="meta", dtype=torch.int32),
            torch.empty(2, device="meta", dtype=torch.int32),
        ],
        torch.empty(150, 5, device="meta", dtype=torch.int32),
    )

    result = ds.lookup(op)

    assert result is None
    assert ds.last_miss_reason == "runtime_attention_leaf_required"


def test_direct_lightning_indexer_without_canonical_runtime_contract_fails_closed(tmp_path):
    data_dir = tmp_path / "lightning_indexer_quant_op"
    data_dir.mkdir()
    _write_text(
        data_dir / "op_mapping.yaml",
        """
version: "test"
operator_mappings:
  "tensor_cast.quant_lightning_indexer.default":
    kernel_type: LightningIndexer
    query_mode: attention_lightning_indexer
""",
    )
    rows = [
        '"100,8,128;64,16;1;1","DT_BF16;DT_BF16;INT32;INT32","ND;ND;ND;ND","100,5","INT32","ND",10.0',
        '"200,8,128;64,16;1;1","DT_BF16;DT_BF16;INT32;INT32","ND;ND;ND;ND","200,5","INT32","ND",20.0',
        '"100,8,128;128,16;1;1","DT_BF16;DT_BF16;INT32;INT32","ND;ND;ND;ND","100,5","INT32","ND",30.0',
        '"200,8,128;128,16;1;1","DT_BF16;DT_BF16;INT32;INT32","ND;ND;ND;ND","200,5","INT32","ND",40.0',
    ]
    _write_text(
        data_dir / "LightningIndexer.csv",
        "\n".join(
            [
                "Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Duration(us)",
                *rows,
            ]
        ),
    )
    ds = InterpolatingDataSource(ProfilingDataSource(data_dir))
    op = _make_op_info(
        "tensor_cast.quant_lightning_indexer.default",
        [
            torch.empty(1, 125, 8, 128, device="meta", dtype=torch.bfloat16),
            torch.empty(1, 125, 8, device="meta", dtype=torch.bfloat16),
            torch.empty(80, 16, device="meta", dtype=torch.bfloat16),
            5,
            1,
            torch.empty(1, device="meta", dtype=torch.int32),
            torch.empty(1, device="meta", dtype=torch.int32),
        ],
        torch.empty(1, 125, 5, device="meta", dtype=torch.int64),
    )

    result = ds.lookup(op)

    assert result is None
    assert ds.last_miss_reason == "runtime_attention_leaf_required"


def test_direct_quant_lightning_indexer_legacy_request_contract_requires_runtime_leaf(tmp_path):
    data_dir = tmp_path / "lightning_indexer_missing_seq_lens"
    data_dir.mkdir()
    _write_text(
        data_dir / "op_mapping.yaml",
        """
version: "test"
operator_mappings:
  "tensor_cast.quant_lightning_indexer.default":
    kernel_type: LightningIndexer
    query_mode: attention_lightning_indexer
""",
    )
    rows = [
        '"100,8,128;64,16;1;1","DT_BF16;DT_BF16;INT32;INT32","ND;ND;ND;ND","100,5","INT32","ND",10.0',
        '"200,8,128;64,16;1;1","DT_BF16;DT_BF16;INT32;INT32","ND;ND;ND;ND","200,5","INT32","ND",20.0',
    ]
    _write_text(
        data_dir / "LightningIndexer.csv",
        "\n".join(
            [
                "Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Duration(us)",
                *rows,
            ]
        ),
    )
    ds = InterpolatingDataSource(ProfilingDataSource(data_dir))
    op = _make_op_info(
        "tensor_cast.quant_lightning_indexer.default",
        [
            torch.empty(1, 150, 8, 128, device="meta", dtype=torch.bfloat16),
            torch.empty(1, 150, 8, device="meta", dtype=torch.bfloat16),
            torch.empty(64, 16, device="meta", dtype=torch.bfloat16),
            5,
            1,
            None,
            torch.empty(1, device="meta", dtype=torch.int32),
        ],
        torch.empty(1, 150, 5, device="meta", dtype=torch.int64),
    )

    result = ds.lookup(op)

    assert result is None
    assert ds.last_miss_reason == "runtime_attention_leaf_required"


def test_direct_lightning_indexer_max_dim_does_not_bypass_runtime_leaf_contract(tmp_path):
    data_dir = tmp_path / "lightning_indexer_max_dim_2"
    data_dir.mkdir()
    _write_text(
        data_dir / "op_mapping.yaml",
        """
version: "test"
interpolation_policy:
  kernel_overrides:
    LightningIndexer:
      max_interpolation_dim: 2
operator_mappings:
  "tensor_cast.lightning_indexer.default":
    kernel_type: LightningIndexer
    query_mode: attention_lightning_indexer
""",
    )
    rows = [
        '"100,8,128;64,16;1;2","DT_BF16;INT32;INT32;INT32","ND;ND;ND;ND","100,5","INT32","ND",10.0',
        '"200,8,128;64,16;1;2","DT_BF16;INT32;INT32;INT32","ND;ND;ND;ND","200,5","INT32","ND",20.0',
        '"100,8,128;128,16;1;2","DT_BF16;INT32;INT32;INT32","ND;ND;ND;ND","100,5","INT32","ND",30.0',
        '"200,8,128;128,16;1;2","DT_BF16;INT32;INT32;INT32","ND;ND;ND;ND","200,5","INT32","ND",40.0',
    ]
    _write_text(
        data_dir / "LightningIndexer.csv",
        "\n".join(
            [
                "Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Duration(us)",
                *rows,
            ]
        ),
    )
    ds = InterpolatingDataSource(ProfilingDataSource(data_dir))
    out = torch.empty(125, 5, device="meta", dtype=torch.int32)
    op = _make_op_info(
        "tensor_cast.lightning_indexer.default",
        [
            torch.empty(125, 8, 128, device="meta", dtype=torch.bfloat16),
            torch.empty(80, 16, device="meta", dtype=torch.int32),
            torch.empty(1, device="meta", dtype=torch.int32),
            torch.empty(2, device="meta", dtype=torch.int32),
        ],
        out,
    )

    result = ds.lookup(op)

    assert result is None
    assert ds.last_miss_reason == "runtime_attention_leaf_required"


def test_direct_lightning_indexer_legacy_topk_contract_requires_runtime_leaf(tmp_path):
    data_dir = tmp_path / "lightning_indexer_topk_regime"
    data_dir.mkdir()
    _write_text(
        data_dir / "op_mapping.yaml",
        """
version: "test"
operator_mappings:
  "tensor_cast.lightning_indexer.default":
    kernel_type: LightningIndexer
    query_mode: attention_lightning_indexer
""",
    )
    _write_text(
        data_dir / "LightningIndexer.csv",
        """
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Duration(us)
"100,8,128;64,16;1;2","DT_BF16;INT32;INT32;INT32","ND;ND;ND;ND","100,4","INT32","ND",10.0
"200,8,128;64,16;1;2","DT_BF16;INT32;INT32;INT32","ND;ND;ND;ND","200,4","INT32","ND",20.0
"100,8,128;64,16;1;2","DT_BF16;INT32;INT32;INT32","ND;ND;ND;ND","100,8","INT32","ND",30.0
"200,8,128;64,16;1;2","DT_BF16;INT32;INT32;INT32","ND;ND;ND;ND","200,8","INT32","ND",40.0
""",
    )
    ds = InterpolatingDataSource(ProfilingDataSource(data_dir))
    op = _make_op_info(
        "tensor_cast.lightning_indexer.default",
        [
            torch.empty(150, 8, 128, device="meta", dtype=torch.bfloat16),
            torch.empty(64, 16, device="meta", dtype=torch.int32),
            torch.empty(1, device="meta", dtype=torch.int32),
            torch.empty(2, device="meta", dtype=torch.int32),
        ],
        torch.empty(150, 6, device="meta", dtype=torch.int32),
    )

    assert ds.lookup(op) is None
    assert ds.last_miss_reason == "runtime_attention_leaf_required"


def test_direct_lightning_indexer_legacy_request_group_requires_runtime_leaf(tmp_path):
    data_dir = tmp_path / "lightning_indexer_request_group"
    data_dir.mkdir()
    _write_text(
        data_dir / "op_mapping.yaml",
        """
version: "test"
operator_mappings:
  "tensor_cast.lightning_indexer.default":
    kernel_type: LightningIndexer
    query_mode: attention_lightning_indexer
""",
    )
    _write_text(
        data_dir / "LightningIndexer.csv",
        """
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Duration(us)
"100,8,128;64,16;1;2","DT_BF16;INT32;INT32;INT32","ND;ND;ND;ND","100,4","INT32","ND",10.0
"200,8,128;64,16;1;2","DT_BF16;INT32;INT32;INT32","ND;ND;ND;ND","200,4","INT32","ND",20.0
""",
    )
    ds = InterpolatingDataSource(ProfilingDataSource(data_dir))
    op = _make_op_info(
        "tensor_cast.lightning_indexer.default",
        [
            torch.empty(150, 8, 128, device="meta", dtype=torch.bfloat16),
            torch.empty(64, 16, device="meta", dtype=torch.int32),
            torch.empty(1, device="meta", dtype=torch.int32),
            torch.empty(3, device="meta", dtype=torch.int32),
        ],
        torch.empty(150, 4, device="meta", dtype=torch.int32),
    )

    assert ds.lookup(op) is None
    assert ds.last_miss_reason == "runtime_attention_leaf_required"


def test_direct_lightning_indexer_rows_without_runtime_metadata_require_runtime_leaf(tmp_path):
    data_dir = tmp_path / "lightning_indexer_missing_request_group"
    data_dir.mkdir()
    _write_text(
        data_dir / "op_mapping.yaml",
        """
version: "test"
operator_mappings:
  "tensor_cast.lightning_indexer.default":
    kernel_type: LightningIndexer
    query_mode: attention_lightning_indexer
""",
    )
    _write_text(
        data_dir / "LightningIndexer.csv",
        """
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Duration(us)
"100,8,128;64,16","DT_BF16;INT32","ND;ND","100,4","INT32","ND",10.0
"200,8,128;64,16","DT_BF16;INT32","ND;ND","200,4","INT32","ND",20.0
""",
    )
    ds = InterpolatingDataSource(ProfilingDataSource(data_dir))
    op = _make_op_info(
        "tensor_cast.lightning_indexer.default",
        [
            torch.empty(150, 8, 128, device="meta", dtype=torch.bfloat16),
            torch.empty(64, 16, device="meta", dtype=torch.int32),
            torch.empty(1, device="meta", dtype=torch.int32),
            torch.empty(2, device="meta", dtype=torch.int32),
        ],
        torch.empty(150, 4, device="meta", dtype=torch.int32),
    )

    assert ds.lookup(op) is None
    assert ds.last_miss_reason == "runtime_attention_leaf_required"


def test_direct_lightning_indexer_missing_csv_still_requires_runtime_leaf(tmp_path):
    data_dir = tmp_path / "lightning_indexer_missing_csv"
    data_dir.mkdir()
    _write_text(
        data_dir / "op_mapping.yaml",
        """
version: "test"
operator_mappings:
  "tensor_cast.lightning_indexer.default":
    kernel_type: LightningIndexer
    query_mode: attention_lightning_indexer
""",
    )
    ds = InterpolatingDataSource(ProfilingDataSource(data_dir))
    op = _make_op_info(
        "tensor_cast.lightning_indexer.default",
        [torch.empty(125, 8, 128, device="meta", dtype=torch.bfloat16)],
        torch.empty(125, 5, device="meta", dtype=torch.int32),
    )

    result = ds.lookup(op)

    assert result is None
    assert ds.last_miss_reason == "runtime_attention_leaf_required"
    assert ds.last_miss_details["query_mode"] == "attention_lightning_indexer"
    assert ds.last_miss_details["kernel_type"] == "LightningIndexer"


def test_lightning_indexer_preserves_base_exact_result(tmp_path):
    data_dir = tmp_path / "lightning_indexer_base_compute_exact"
    data_dir.mkdir()
    _write_text(
        data_dir / "op_mapping.yaml",
        """
version: "test"
operator_mappings:
  "tensor_cast.lightning_indexer.variant.default":
    kernel_type: LightningIndexer
    query_mode: attention_lightning_indexer
""",
    )
    rows = [
        '"150,8,128;64,16;1;2","DT_BF16;INT32;INT32;INT32","ND;ND;ND;ND","150,4","INT32","ND",99.0',
        '"100,8,128;64,16;1;2","DT_BF16;INT32;INT32;INT32","ND;ND;ND;ND","100,5","INT32","ND",20.0',
        '"200,8,128;64,16;1;2","DT_BF16;INT32;INT32;INT32","ND;ND;ND;ND","200,5","INT32","ND",40.0',
    ]
    _write_text(
        data_dir / "LightningIndexer.csv",
        "\n".join(
            [
                "Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Duration(us)",
                *rows,
            ]
        ),
    )
    ds = InterpolatingDataSource(ProfilingDataSource(data_dir))
    op = _make_op_info(
        "tensor_cast.lightning_indexer.variant.default",
        [
            torch.empty(150, 8, 128, device="meta", dtype=torch.bfloat16),
            torch.empty(64, 16, device="meta", dtype=torch.int32),
            torch.empty(1, device="meta", dtype=torch.int32),
            torch.empty(2, device="meta", dtype=torch.int32),
        ],
        torch.empty(150, 5, device="meta", dtype=torch.int32),
    )

    result = ds.lookup(op)

    assert result is not None
    assert result.source == QuerySource.MEASURED
    assert result.latency_us == pytest.approx(99.0)


def test_scatter_nd_update_mla_interpolates_reordered_cache_update(tmp_path):
    data_dir = tmp_path / "scatter_nd_update_mla"
    data_dir.mkdir()
    _write_text(
        data_dir / "op_mapping.yaml",
        """
version: "test"
operator_mappings:
  "tensor_cast.scatter_nd_update_mla.default":
    kernel_type: ScatterNdUpdate
    alternate_kernel_types: [ScatterNdUpdateAiCore]
    query_mode: scatter_nd_update_mla
""",
    )
    _write_text(
        data_dir / "ScatterNdUpdate.csv",
        "\n".join(
            [
                "Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Duration(us)",
                '"1000,128;100,1;100,128","DT_BF16;INT32;DT_BF16","ND;ND;ND","1000,128","DT_BF16","ND",10.0',
                '"1000,128;200,1;200,128","DT_BF16;INT32;DT_BF16","ND;ND;ND","1000,128","DT_BF16","ND",20.0',
            ]
        ),
    )
    ds = InterpolatingDataSource(ProfilingDataSource(data_dir))
    op = _make_op_info(
        "tensor_cast.scatter_nd_update_mla.default",
        [
            torch.empty(150, 128, device="meta", dtype=torch.bfloat16),
            torch.empty(1000, 128, device="meta", dtype=torch.bfloat16),
            torch.empty(150, device="meta", dtype=torch.int32),
        ],
        torch.empty(1000, 128, device="meta", dtype=torch.bfloat16),
    )

    result = ds.lookup(op)

    assert result is not None
    assert result.source == QuerySource.INTERPOLATED
    assert result.latency_us == pytest.approx(15.0)
    assert result.details["query_mode"] == "scatter_nd_update_mla"
    assert result.details["interpolation_path"] == "scatter_nd_update_mla_1d"
    assert result.details["axes"] == ["tokens"]


def test_scatter_nd_update_mla_does_not_mix_latency_columns(tmp_path):
    data_dir = tmp_path / "scatter_nd_update_mla_latency_source_mixed"
    data_dir.mkdir()
    _write_text(
        data_dir / "op_mapping.yaml",
        """
version: "test"
operator_mappings:
  "tensor_cast.scatter_nd_update_mla.default":
    kernel_type: ScatterNdUpdate
    query_mode: scatter_nd_update_mla
""",
    )
    _write_text(
        data_dir / "ScatterNdUpdate.csv",
        """
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Average Duration(us),Duration(us)
"1000,128;100,1;100,128","DT_BF16;INT32;DT_BF16","ND;ND;ND","1000,128","DT_BF16","ND",10.0,10.0
"1000,128;200,1;200,128","DT_BF16;INT32;DT_BF16","ND;ND;ND","1000,128","DT_BF16","ND",0.0,20.0
""",
    )
    ds = InterpolatingDataSource(ProfilingDataSource(data_dir))
    op = _make_op_info(
        "tensor_cast.scatter_nd_update_mla.default",
        [
            torch.empty(150, 128, device="meta", dtype=torch.bfloat16),
            torch.empty(1000, 128, device="meta", dtype=torch.bfloat16),
            torch.empty(150, device="meta", dtype=torch.int32),
        ],
        torch.empty(1000, 128, device="meta", dtype=torch.bfloat16),
    )

    result = ds.lookup(op)

    assert result is None
    assert ds.last_miss_reason == "insufficient_filtered_candidates"


def test_scatter_nd_update_mla_scalar_update_fails_closed(tmp_path):
    data_dir = tmp_path / "scatter_nd_update_mla_scalar_update"
    data_dir.mkdir()
    _write_text(
        data_dir / "op_mapping.yaml",
        """
version: "test"
operator_mappings:
  "tensor_cast.scatter_nd_update_mla.default":
    kernel_type: ScatterNdUpdate
    query_mode: scatter_nd_update_mla
""",
    )
    _write_text(
        data_dir / "ScatterNdUpdate.csv",
        """
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Duration(us)
"1000,128;100,1;100,128","DT_BF16;INT32;DT_BF16","ND;ND;ND","1000,128","DT_BF16","ND",10.0
""",
    )
    ds = InterpolatingDataSource(ProfilingDataSource(data_dir))
    op = _make_op_info(
        "tensor_cast.scatter_nd_update_mla.default",
        [
            torch.empty((), device="meta", dtype=torch.bfloat16),
            torch.empty(1000, 128, device="meta", dtype=torch.bfloat16),
            torch.empty(1, device="meta", dtype=torch.int32),
        ],
        torch.empty(1000, 128, device="meta", dtype=torch.bfloat16),
    )

    result = ds.lookup(op)

    assert result is None
    assert ds.last_miss_reason == "scatter_nd_update_interpolation_failed"
    assert ds.last_miss_details["attempts"][0]["status"] == "scatter_target_unextractable"


def test_scatter_nd_update_mla_single_exact_coordinate_is_not_interpolation(tmp_path):
    data_dir = tmp_path / "scatter_nd_update_mla_exact_coordinate"
    data_dir.mkdir()
    _write_text(
        data_dir / "op_mapping.yaml",
        """
version: "test"
operator_mappings:
  "tensor_cast.scatter_nd_update_mla.default":
    kernel_type: ScatterNdUpdate
    query_mode: scatter_nd_update_mla
""",
    )
    _write_text(
        data_dir / "ScatterNdUpdate.csv",
        """
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Duration(us)
"1000,128;150,1;150,128","DT_BF16;INT32;DT_BF16","ND;ND;ND","1000,128","DT_BF16","ND",12.0
""",
    )
    ds = InterpolatingDataSource(ProfilingDataSource(data_dir))
    op = _make_op_info(
        "tensor_cast.scatter_nd_update_mla.default",
        [
            torch.empty(150, 128, device="meta", dtype=torch.bfloat16),
            torch.empty(1000, 128, device="meta", dtype=torch.bfloat16),
            torch.empty(150, device="meta", dtype=torch.int32),
        ],
        torch.empty(1000, 128, device="meta", dtype=torch.bfloat16),
    )

    result = ds.lookup(op)

    assert result is None
    assert ds.last_miss_reason == "insufficient_filtered_candidates"


def test_scatter_cache_write_sub_kernel_interpolates_with_full_cache_regime(tmp_path):
    data_dir = tmp_path / "scatter_cache_write_sub_kernel"
    data_dir.mkdir()
    _write_text(data_dir / "op_mapping.yaml", 'version: "test"\noperator_mappings: {}')
    _write_text(
        data_dir / "ScatterNdUpdate.csv",
        """
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Duration(us)
"1000,128;100,1;100,128","DT_BF16;INT32;DT_BF16","ND;ND;ND","1000,128","DT_BF16","ND",10.0
"1000,128;200,1;200,128","DT_BF16;INT32;DT_BF16","ND;ND;ND","1000,128","DT_BF16","ND",20.0
""",
    )
    ds = InterpolatingDataSource(ProfilingDataSource(data_dir))

    result = ds._interpolate_scatter_nd_update_by_shapes(
        ["ScatterNdUpdate"],
        [(1000, 128), (150, 1), (150, 128)],
        ["DT_BF16", "INT32", "DT_BF16"],
    )

    assert result is not None
    assert result.source == QuerySource.INTERPOLATED
    assert result.latency_us == pytest.approx(15.0)


def test_scatter_cache_write_sub_kernel_records_failure_details(tmp_path):
    data_dir = tmp_path / "scatter_cache_write_sub_kernel_miss"
    data_dir.mkdir()
    _write_text(data_dir / "op_mapping.yaml", 'version: "test"\noperator_mappings: {}')
    _write_text(
        data_dir / "ScatterNdUpdate.csv",
        """
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Duration(us)
"1000,128;100,1;100,128","DT_BF16;INT32;DT_BF16","ND;ND;ND","1000,128","DT_BF16","ND",10.0
"1000,128;200,1;200,128","DT_BF16;INT32;DT_BF16","ND;ND;ND","1000,128","DT_BF16","ND",20.0
""",
    )
    ds = InterpolatingDataSource(ProfilingDataSource(data_dir))

    result = ds._interpolate_scatter_nd_update_by_shapes(
        ["ScatterNdUpdate"],
        [(2000, 128), (150, 1), (150, 128)],
        ["DT_BF16", "INT32", "DT_BF16"],
    )

    assert result is None
    assert ds.last_miss_reason == "scatter_cache_write_interpolation_failed"
    assert ds.last_miss_details["attempts"][0]["status"] == "regime_key_unmatched"


def test_native_layer_norm_maps_to_layer_norm_v3_with_metadata_ignored(tmp_path):
    data_dir = tmp_path / "native_layer_norm"
    data_dir.mkdir()
    _write_text(
        data_dir / "op_mapping.yaml",
        """
version: "test"
operator_mappings:
  "aten.native_layer_norm.default":
    kernel_type: LayerNormV3
    alternate_kernel_types: [LayerNormV3WithImplMode]
    tc_input_count: 3
""",
    )
    _write_text(
        data_dir / "LayerNormV3.csv",
        "\n".join(
            [
                "Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Duration(us)",
                '"100,128;128;128","FLOAT;FLOAT;FLOAT","ND;ND;ND","100,128;100,1;100,1","FLOAT;FLOAT;FLOAT","ND;ND;ND",10.0',
                '"200,128;128;128","FLOAT;FLOAT;FLOAT","ND;ND;ND","200,128;200,1;200,1","FLOAT;FLOAT;FLOAT","ND;ND;ND",20.0',
            ]
        ),
    )
    ds = InterpolatingDataSource(ProfilingDataSource(data_dir))
    op = _make_op_info(
        "aten.native_layer_norm.default",
        [
            torch.empty(150, 128, device="meta", dtype=torch.float32),
            (128,),
            torch.empty(128, device="meta", dtype=torch.float32),
            torch.empty(128, device="meta", dtype=torch.float32),
            1e-5,
        ],
        (
            torch.empty(150, 128, device="meta", dtype=torch.float32),
            torch.empty(150, 1, device="meta", dtype=torch.float32),
            torch.empty(150, 1, device="meta", dtype=torch.float32),
        ),
    )

    result = ds.lookup(op)

    assert result is not None
    assert result.source == QuerySource.INTERPOLATED
    assert result.latency_us == pytest.approx(15.0)
    assert result.details["kernel_type"] == "LayerNormV3"


def test_dynamic_quant_compute_scale_interpolates_guarded_2d(tmp_path, monkeypatch):
    data_dir = tmp_path / "dynamic_quant_compute_scale"
    data_dir.mkdir()
    _write_text(
        data_dir / "op_mapping.yaml",
        """
version: "test"
operator_mappings:
  "tensor_cast.dynamic_quantize_asymmetric.default":
    kernel_type: DynamicQuant
    compute_subcategory: compute_scale
""",
    )
    rows = []
    for tokens in (100, 200):
        for channels in (64, 128):
            latency = tokens / 10 + channels / 64
            rows.append(
                f'"{tokens},{channels}","DT_BF16","ND",'
                f'"{tokens},{channels};{tokens};{tokens}","INT8;FLOAT;INT32","ND;ND;ND",{latency}'
            )
    _write_text(
        data_dir / "DynamicQuant.csv",
        "\n".join(
            [
                "Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Duration(us)",
                *rows,
            ]
        ),
    )
    ds = InterpolatingDataSource(ProfilingDataSource(data_dir))
    monkeypatch.setattr(ds.base, "lookup", lambda _op_invoke_info: None)
    op = _make_op_info(
        "tensor_cast.dynamic_quantize_asymmetric.default",
        [torch.empty((150, 96), device="meta", dtype=torch.bfloat16), [-1]],
        (
            torch.empty((150, 96), device="meta", dtype=torch.int8),
            torch.empty((150, 1), device="meta", dtype=torch.float32),
            torch.empty((150, 1), device="meta", dtype=torch.int32),
        ),
    )

    result = ds.lookup(op)

    assert result is not None
    assert result.source == QuerySource.INTERPOLATED
    assert result.latency_us == pytest.approx(16.5)
    assert result.details["compute_subcategory"] == "compute_scale"
    assert result.details["interpolation_path"] == "compute_scale_2d"
    assert result.details["interpolation_dim"] == 2
    assert result.details["axes"] == ["M", "K"]
    assert result.details["scale_mode"] == "per_token"
    assert result.details["auxiliary_modes"] == [["per_token", None], ["per_token", None]]


def test_compute_scale_mode_keeps_block_quant_kernel_specific():
    assert InterpolatingDataSource._compute_scale_mode((8, 64), (), "DynamicQuant") == ("per_tensor", None)
    assert InterpolatingDataSource._compute_scale_mode((8, 64), (8, 1), "DynamicQuant") == ("per_token", None)
    assert InterpolatingDataSource._compute_scale_mode((8, 64), (1, 64), "DynamicQuant") == (
        "per_channel",
        None,
    )
    assert InterpolatingDataSource._compute_scale_mode((8, 64), (4,), "DynamicQuant") is None
    assert InterpolatingDataSource._compute_scale_mode((8, 64), (4,), "DynamicBlockQuant") == (
        "per_block",
        16,
    )


def test_dynamic_quant_compute_scale_matches_float16_profiling_dtype(tmp_path, monkeypatch):
    data_dir = tmp_path / "dynamic_quant_float16"
    data_dir.mkdir()
    _write_text(
        data_dir / "op_mapping.yaml",
        """
version: "test"
operator_mappings:
  "tensor_cast.dynamic_quantize_symmetric.default":
    kernel_type: DynamicQuant
    compute_subcategory: compute_scale
""",
    )
    _write_text(
        data_dir / "DynamicQuant.csv",
        """
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Duration(us)
"100,64","DT_FLOAT16","ND","100,64;100","INT8;FLOAT","ND;ND",11.0
"200,64","DT_FLOAT16","ND","200,64;200","INT8;FLOAT","ND;ND",21.0
""",
    )
    ds = InterpolatingDataSource(ProfilingDataSource(data_dir))
    monkeypatch.setattr(ds.base, "lookup", lambda _op_invoke_info: None)
    op = _make_op_info(
        "tensor_cast.dynamic_quantize_symmetric.default",
        [torch.empty((150, 64), device="meta", dtype=torch.float16), [-1]],
        (
            torch.empty((150, 64), device="meta", dtype=torch.int8),
            torch.empty((150, 1), device="meta", dtype=torch.float32),
        ),
    )

    result = ds.lookup(op)

    assert result is not None
    assert result.source == QuerySource.INTERPOLATED
    assert result.latency_us == pytest.approx(16.0)
    assert result.details["interpolation_path"] == "compute_scale_1d"


def test_dynamic_quant_compute_scale_preserves_base_exact_result(tmp_path, monkeypatch):
    data_dir = tmp_path / "dynamic_quant_output_regime"
    data_dir.mkdir()
    _write_text(
        data_dir / "op_mapping.yaml",
        """
version: "test"
operator_mappings:
  "tensor_cast.dynamic_quantize_asymmetric.default":
    kernel_type: DynamicQuant
    compute_subcategory: compute_scale
""",
    )
    _write_text(
        data_dir / "DynamicQuant.csv",
        """
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Duration(us)
"100,64","DT_BF16","ND","100,64;100","INT8;FLOAT","ND;ND",11.0
"200,64","DT_BF16","ND","200,64;200","INT8;FLOAT","ND;ND",21.0
"100,128","DT_BF16","ND","100,128;100","INT8;FLOAT","ND;ND",12.0
"200,128","DT_BF16","ND","200,128;200","INT8;FLOAT","ND;ND",22.0
""",
    )
    ds = InterpolatingDataSource(ProfilingDataSource(data_dir))
    monkeypatch.setattr(
        ds.base,
        "lookup",
        lambda _op_invoke_info: QueryResult(
            latency_us=1.0,
            confidence=1.0,
            source=QuerySource.MEASURED,
            details={"kernel_type": "DynamicQuant"},
        ),
    )
    op = _make_op_info(
        "tensor_cast.dynamic_quantize_asymmetric.default",
        [torch.empty((150, 96), device="meta", dtype=torch.bfloat16), [-1]],
        (
            torch.empty((150, 96), device="meta", dtype=torch.int8),
            torch.empty((150, 1), device="meta", dtype=torch.float32),
            torch.empty((150, 1), device="meta", dtype=torch.int32),
        ),
    )

    result = ds.lookup(op)

    assert result is not None
    assert result.source == QuerySource.MEASURED
    assert result.latency_us == pytest.approx(1.0)


def test_dynamic_quant_compute_scale_does_not_override_base_exact_result(tmp_path, monkeypatch):
    data_dir = tmp_path / "dynamic_quant_exact_coordinate"
    data_dir.mkdir()
    _write_text(
        data_dir / "op_mapping.yaml",
        """
version: "test"
operator_mappings:
  "tensor_cast.dynamic_quantize_symmetric.default":
    kernel_type: DynamicQuant
    compute_subcategory: compute_scale
""",
    )
    _write_text(
        data_dir / "DynamicQuant.csv",
        """
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Duration(us)
"100,64","DT_BF16","ND","100,64;100","INT8;FLOAT","ND;ND",11.0
"200,64","DT_BF16","ND","200,64;200","INT8;FLOAT","ND;ND",21.0
""",
    )
    ds = InterpolatingDataSource(ProfilingDataSource(data_dir))
    monkeypatch.setattr(
        ds.base,
        "lookup",
        lambda _op_invoke_info: QueryResult(
            latency_us=999.0,
            confidence=1.0,
            source=QuerySource.MEASURED,
            details={"kernel_type": "DynamicQuant"},
        ),
    )
    op = _make_op_info(
        "tensor_cast.dynamic_quantize_symmetric.default",
        [torch.empty((100, 64), device="meta", dtype=torch.bfloat16), [-1]],
        (
            torch.empty((100, 64), device="meta", dtype=torch.int8),
            torch.empty((100, 1), device="meta", dtype=torch.float32),
        ),
    )

    result = ds.lookup(op)

    assert result is not None
    assert result.source == QuerySource.MEASURED
    assert result.latency_us == pytest.approx(999.0)


def test_dynamic_quant_compute_scale_honors_max_interpolation_dim(tmp_path, monkeypatch):
    data_dir = tmp_path / "dynamic_quant_max_dim"
    data_dir.mkdir()
    _write_text(
        data_dir / "op_mapping.yaml",
        """
version: "test"
interpolation_policy:
  kernel_overrides:
    DynamicQuant:
      max_interpolation_dim: 1
operator_mappings:
  "tensor_cast.dynamic_quantize_symmetric.default":
    kernel_type: DynamicQuant
    compute_subcategory: compute_scale
""",
    )
    rows = []
    for tokens in (100, 200):
        for channels in (64, 128):
            rows.append(
                f'"{tokens},{channels}","DT_BF16","ND",'
                f'"{tokens},{channels};{tokens}","INT8;FLOAT","ND;ND",{tokens + channels}'
            )
    _write_text(
        data_dir / "DynamicQuant.csv",
        "\n".join(
            [
                "Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Duration(us)",
                *rows,
            ]
        ),
    )
    ds = InterpolatingDataSource(ProfilingDataSource(data_dir))
    monkeypatch.setattr(ds.base, "lookup", lambda _op_invoke_info: None)
    op = _make_op_info(
        "tensor_cast.dynamic_quantize_symmetric.default",
        [torch.empty((150, 96), device="meta", dtype=torch.bfloat16), [-1]],
        (
            torch.empty((150, 96), device="meta", dtype=torch.int8),
            torch.empty((150, 1), device="meta", dtype=torch.float32),
        ),
    )

    result = ds.lookup(op)

    assert result is None
    attempts = ds.last_miss_details["attempts"][0]["diagnostics"]["attempts"]
    assert any(attempt["status"] == "interpolation_dim_disabled" for attempt in attempts)


def test_quantized_matmul_interpolates_guarded_3d(tmp_path, monkeypatch):
    data_dir = tmp_path / "quantized_matmul"
    data_dir.mkdir()
    _write_text(
        data_dir / "op_mapping.yaml",
        """
version: "test"
operator_mappings:
  "tensor_cast.static_quant_linear.default":
    kernel_type: QuantBatchMatmulV3
    tc_input_count: 2
    compute_subcategory: quantized_matmul
    expected_input_formats: [ND, ND]
""",
    )
    rows = []
    for m_dim in (10, 20):
        for k_dim in (32, 64):
            for n_dim in (64, 96):
                latency = m_dim + k_dim + n_dim
                rows.append(
                    f'"{m_dim},{k_dim};{k_dim},{n_dim}","INT8;INT8","ND;ND","{m_dim},{n_dim}","DT_BF16","ND",{latency}'
                )
    _write_text(
        data_dir / "QuantBatchMatmulV3.csv",
        "\n".join(
            [
                "Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Duration(us)",
                *rows,
            ]
        ),
    )
    ds = InterpolatingDataSource(ProfilingDataSource(data_dir))
    monkeypatch.setattr(ds.base, "lookup", lambda _op_invoke_info: None)
    op = _make_op_info(
        "tensor_cast.static_quant_linear.default",
        [
            torch.empty((15, 48), device="meta", dtype=torch.int8),
            torch.empty((48, 80), device="meta", dtype=torch.int8),
        ],
        torch.empty((15, 80), device="meta", dtype=torch.bfloat16),
    )

    result = ds.lookup(op)

    assert result is not None
    assert result.source == QuerySource.INTERPOLATED
    assert result.latency_us == pytest.approx(143.0)
    assert result.details["compute_subcategory"] == "quantized_matmul"
    assert result.details["interpolation_path"] == "quantized_matmul_3d"
    assert result.details["interpolation_dim"] == 3
    assert result.details["axes"] == ["M", "K", "N"]


def test_quantized_matmul_base_exact_owns_output_dtype_compatibility(tmp_path):
    data_dir = tmp_path / "quantized_matmul_output_dtype"
    data_dir.mkdir()
    _write_text(
        data_dir / "op_mapping.yaml",
        """
version: "test"
operator_mappings:
  "tensor_cast.static_quant_linear.default":
    kernel_type: QuantBatchMatmulV3
    tc_input_count: 2
    compute_subcategory: quantized_matmul
    expected_input_formats: [ND, ND]
""",
    )
    _write_text(
        data_dir / "QuantBatchMatmulV3.csv",
        """
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Duration(us)
"10,32;32,64","INT8;INT8","ND;ND","10,64","DT_BF16","ND",106.0
""",
    )
    ds = InterpolatingDataSource(ProfilingDataSource(data_dir))
    op = _make_op_info(
        "tensor_cast.static_quant_linear.default",
        [
            torch.empty((10, 32), device="meta", dtype=torch.int8),
            torch.empty((32, 64), device="meta", dtype=torch.int8),
        ],
        torch.empty((10, 64), device="meta", dtype=torch.float16),
    )

    result = ds.lookup(op)

    assert result is not None
    assert result.source == QuerySource.MEASURED
    assert result.details["kernel_type"] == "QuantBatchMatmulV3"


def test_quantized_matmul_preserves_base_exact_coordinate(tmp_path):
    data_dir = tmp_path / "quantized_matmul_exact_output_regime"
    data_dir.mkdir()
    _write_text(
        data_dir / "op_mapping.yaml",
        """
version: "test"
operator_mappings:
  "tensor_cast.static_quant_linear.default":
    kernel_type: QuantBatchMatmulV3
    tc_input_count: 2
    compute_subcategory: quantized_matmul
    expected_input_formats: [ND, ND]
""",
    )
    _write_text(
        data_dir / "QuantBatchMatmulV3.csv",
        """
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Duration(us)
"10,32;32,64","INT8;INT8","ND;ND","10,64","DT_BF16","ND",106.0
""",
    )
    ds = InterpolatingDataSource(ProfilingDataSource(data_dir))
    op = _make_op_info(
        "tensor_cast.static_quant_linear.default",
        [
            torch.empty((10, 32), device="meta", dtype=torch.int8),
            torch.empty((32, 64), device="meta", dtype=torch.int8),
        ],
        torch.empty((10, 64), device="meta", dtype=torch.bfloat16),
    )

    result = ds.lookup(op)

    assert result is not None
    assert result.source == QuerySource.MEASURED
    assert result.details["kernel_type"] == "QuantBatchMatmulV3"


def test_quantized_matmul_does_not_override_base_format_selection(tmp_path):
    data_dir = tmp_path / "quantized_matmul_ambiguous_input_formats"
    data_dir.mkdir()
    _write_text(
        data_dir / "op_mapping.yaml",
        """
version: "test"
operator_mappings:
  "tensor_cast.static_quant_linear.default":
    kernel_type: QuantBatchMatmulV3
    tc_input_count: 2
    compute_subcategory: quantized_matmul
    expected_input_formats: [ND, FRACTAL_NZ]
""",
    )
    _write_text(
        data_dir / "QuantBatchMatmulV3.csv",
        """
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Duration(us)
"10,32;32,64","INT8;INT8","ND;ND","10,64","DT_BF16","ND",106.0
"10,32;2,2,16,32","INT8;INT8","ND;FRACTAL_NZ","10,64","DT_BF16","ND",206.0
""",
    )
    ds = InterpolatingDataSource(ProfilingDataSource(data_dir))
    op = _make_op_info(
        "tensor_cast.static_quant_linear.default",
        [
            torch.empty((10, 32), device="meta", dtype=torch.int8),
            torch.empty((32, 64), device="meta", dtype=torch.int8),
        ],
        torch.empty((10, 64), device="meta", dtype=torch.bfloat16),
    )

    result = ds.lookup(op)

    assert result is not None
    assert result.source == QuerySource.MEASURED
    assert result.latency_us == pytest.approx(106.0)


def test_quantized_matmul_returns_base_exact_before_interpolation_format_filter(tmp_path):
    data_dir = tmp_path / "quantized_matmul_undeclared_input_format"
    data_dir.mkdir()
    _write_text(
        data_dir / "op_mapping.yaml",
        """
version: "test"
operator_mappings:
  "tensor_cast.static_quant_linear.default":
    kernel_type: QuantBatchMatmulV3
    tc_input_count: 2
    compute_subcategory: quantized_matmul
    expected_input_formats: [ND, ND]
""",
    )
    _write_text(
        data_dir / "QuantBatchMatmulV3.csv",
        """
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Duration(us)
"10,32;2,2,16,32","INT8;INT8","ND;FRACTAL_NZ","10,64","DT_BF16","ND",206.0
""",
    )
    ds = InterpolatingDataSource(ProfilingDataSource(data_dir))
    op = _make_op_info(
        "tensor_cast.static_quant_linear.default",
        [
            torch.empty((10, 32), device="meta", dtype=torch.int8),
            torch.empty((32, 64), device="meta", dtype=torch.int8),
        ],
        torch.empty((10, 64), device="meta", dtype=torch.bfloat16),
    )

    result = ds.lookup(op)

    assert result is not None
    assert result.source == QuerySource.MEASURED
    assert result.latency_us == pytest.approx(206.0)


def test_int4_quantized_matmul_restores_logical_packed_weight_shape(tmp_path):
    data_dir = tmp_path / "int4_quantized_matmul_target"
    data_dir.mkdir()
    _write_text(
        data_dir / "op_mapping.yaml",
        """
version: "test"
operator_mappings:
  "tensor_cast.static_quant_linear_int4.default":
    kernel_type: QuantBatchMatmulV3
    tc_input_count: 2
    compute_subcategory: quantized_matmul
    expected_input_formats: [ND, ND]
""",
    )
    ds = InterpolatingDataSource(ProfilingDataSource(data_dir))
    op = _make_op_info(
        "tensor_cast.static_quant_linear_int4.default",
        [
            torch.empty((10, 32), device="meta", dtype=torch.int8),
            torch.empty((4, 16), device="meta", dtype=torch.int32),
            torch.empty((16,), device="meta", dtype=torch.float32),
            None,
            None,
            None,
            None,
            torch.bfloat16,
        ],
        torch.empty((10, 16), device="meta", dtype=torch.bfloat16),
    )
    mapping = ds.base._op_mapping["operator_mappings"]["tensor_cast.static_quant_linear_int4.default"]

    target = ds._build_compute_target(op, mapping, "QuantBatchMatmulV3")

    assert target is not None
    assert target.axes == {"M": 10.0, "K": 32.0, "N": 16.0}


def test_fp8_quantized_matmul_does_not_reuse_int8_candidates(tmp_path, monkeypatch):
    fp8_dtype = getattr(torch, "float8_e4m3fn", None)
    if fp8_dtype is None:
        pytest.skip("float8 dtype is unavailable in this PyTorch build")

    data_dir = tmp_path / "fp8_quantized_matmul_regime"
    data_dir.mkdir()
    _write_text(
        data_dir / "op_mapping.yaml",
        """
version: "test"
operator_mappings:
  "tensor_cast.fp8_linear.default":
    kernel_type: QuantBatchMatmulV3
    tc_input_count: 2
    compute_subcategory: quantized_matmul
    expected_input_formats: [ND, ND]
""",
    )
    _write_text(
        data_dir / "QuantBatchMatmulV3.csv",
        """
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Duration(us)
"10,32;32,64","INT8;INT8","ND;ND","10,64","DT_BF16","ND",106.0
"20,32;32,64","INT8;INT8","ND;ND","20,64","DT_BF16","ND",116.0
""",
    )
    ds = InterpolatingDataSource(ProfilingDataSource(data_dir))
    monkeypatch.setattr(ds.base, "lookup", lambda _op_invoke_info: None)
    op = _make_op_info(
        "tensor_cast.fp8_linear.default",
        [
            torch.empty((15, 32), device="meta", dtype=fp8_dtype),
            torch.empty((32, 64), device="meta", dtype=fp8_dtype),
        ],
        torch.empty((15, 64), device="meta", dtype=torch.bfloat16),
    )

    result = ds.lookup(op)

    assert result is None
    assert ds.last_miss_reason == "compute_multidim_interpolation_failed"
    assert ds.last_miss_details["attempts"][0]["status"] == "target_unavailable"
