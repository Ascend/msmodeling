"""Tests for FIA enriched CSV lookup (spec: 2026-03-23-fia-enriched-csv-redesign.md)."""

import shutil
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from tensor_cast.performance_model.profiling_database.data_source import QuerySource
from tensor_cast.performance_model.profiling_database.profiling_data_source import (
    _normalize_fia_q_shape,
    _parse_fia_q_shape,
    ProfilingDataSource,
)


# ---- Unit tests: _parse_fia_q_shape ----


class TestParseFiaQShape:
    def test_3d_tnd(self):
        assert _parse_fia_q_shape("128,4,128;12307,128,128;stuff") == (128, 4, 128)

    def test_4d_bnsd(self):
        assert _parse_fia_q_shape("4,16,1,512;other;stuff") == (4, 16, 1, 512)

    def test_empty(self):
        assert _parse_fia_q_shape("") is None

    def test_empty_slot0(self):
        assert _parse_fia_q_shape(";12307,128,128") is None

    def test_fewer_slots(self):
        assert _parse_fia_q_shape("128,4,128") == (128, 4, 128)


# ---- Unit tests: _normalize_fia_q_shape ----


class TestNormalizeFiaQShape:
    def test_n1_3d_identity(self):
        assert _normalize_fia_q_shape((128, 4, 128)) == (128, 4, 128)

    def test_n2_4d_bnsd_squeeze(self):
        assert _normalize_fia_q_shape((4, 16, 1, 512)) == (4, 16, 512)

    def test_n3_2d_reshape(self):
        assert _normalize_fia_q_shape((128, 512), head_dim=128) == (128, 4, 128)

    def test_n4_4d_s_not_1(self):
        assert _normalize_fia_q_shape((4, 16, 32, 512)) is None

    def test_n5_1d(self):
        assert _normalize_fia_q_shape((512,)) is None

    def test_2d_no_head_dim(self):
        assert _normalize_fia_q_shape((128, 512), head_dim=0) is None

    def test_2d_indivisible(self):
        assert _normalize_fia_q_shape((128, 513), head_dim=128) is None


# ---- Integration tests ----

_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "fia_raw_test"


def _make_mock_device_profile():
    dp = MagicMock()
    dp.name = "ATLAS_800_A3_752T_128G_DIE"
    dp.comm_grid = None
    return dp


def _make_attention_op_info(query_shape, key_shape, seq_lens, dtype):
    """Build a mock OpInvokeInfo for tensor_cast.attention.default."""
    query = torch.zeros(query_shape, dtype=dtype)
    key = torch.zeros(key_shape, dtype=dtype)
    value = torch.zeros(key_shape, dtype=dtype)
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int64)
    args = (query, key, value, None, None, None, seq_lens_t, None)
    op_info = MagicMock()
    op_info.func.__str__ = lambda self: "torch.ops.tensor_cast.attention.default"
    op_info.func.__repr__ = lambda self: "torch.ops.tensor_cast.attention.default"
    op_info.args = args
    op_info.kwargs = {}
    op_info.out = None
    return op_info


class TestLookupAttentionEnriched:
    """Tests for _lookup_attention() enriched CSV path."""

    @pytest.fixture
    def ds(self, tmp_path):
        dst = tmp_path / "db"
        shutil.copytree(_FIXTURE_DIR, dst)
        return ProfilingDataSource(str(dst), _make_mock_device_profile())

    def test_e1_qwen3_exact_match(self, ds):
        """Qwen3 GQA 3D Q, avg_seq_len=4096 → HIT."""
        op = _make_attention_op_info(
            query_shape=(128, 4, 128),
            key_shape=(12307, 128, 128),
            seq_lens=[4096] * 128,
            dtype=torch.bfloat16,
        )
        result = ds.lookup(op)
        assert result is not None
        assert abs(result.latency_us - 58.2) < 1.0
        assert result.source == QuerySource.MEASURED

    def test_e2_avg_seq_len_mismatch(self, ds):
        """avg_seq_len=999 not in CSV → MISS."""
        op = _make_attention_op_info(
            query_shape=(128, 4, 128),
            key_shape=(12307, 128, 128),
            seq_lens=[999] * 128,
            dtype=torch.bfloat16,
        )
        assert ds.lookup(op) is None

    def test_e3_nd_mismatch(self, ds):
        """N=8 vs CSV N=4 → MISS."""
        op = _make_attention_op_info(
            query_shape=(128, 8, 128),
            key_shape=(12307, 128, 128),
            seq_lens=[4096] * 128,
            dtype=torch.bfloat16,
        )
        assert ds.lookup(op) is None

    def test_e4_dtype_mismatch(self, ds):
        """INT8 vs BF16 → MISS."""
        op = _make_attention_op_info(
            query_shape=(128, 4, 128),
            key_shape=(12307, 128, 128),
            seq_lens=[4096] * 128,
            dtype=torch.int8,
        )
        assert ds.lookup(op) is None

    def test_e6_dsv3_mla_4d_bnsd(self, ds):
        """DSV3 MLA 4D BNSD CSV Q=(4,16,1,512) → TC 3D (4,16,512) match."""
        op = _make_attention_op_info(
            query_shape=(4, 16, 512),
            key_shape=(1135, 1, 128, 512),
            seq_lens=[2048] * 4,
            dtype=torch.bfloat16,
        )
        result = ds.lookup(op)
        assert result is not None
        assert abs(result.latency_us - 28.0) < 1.0

    def test_e7_block_padding_tolerance(self, ds):
        """TC T=160, CSV T=160 with avg_seq_len=2048 → HIT at 59.0."""
        # CSV row 4: Q=(160,4,128), avg_seq_len=2048, Duration=59.0
        # TC T=176 is block-padded from 160 (ceil(160/16)*16=160, ceil(160/32)*32=160,
        # ceil(160/64)*64=192). Actually 160 is already aligned.
        # Use TC T=192 which is ceil(160/64)*64 — but that's not 160.
        # Better test: CSV has T=128, TC T=128 (exact) — already tested.
        # For block-padding: CSV T=128, TC T=128 is exact match.
        # Let's test TC T=144 vs CSV T=128: ceil(128/16)*16=128, not 144.
        # Actually _is_block_padded(144, 128) checks if 144 == ceil(128/bs)*bs:
        #   bs=16: ceil(128/16)*16=128 ≠ 144
        #   bs=32: ceil(128/32)*32=128 ≠ 144
        #   bs=64: ceil(128/64)*64=128 ≠ 144
        # So 144 is NOT a block-padded version of 128.
        # _is_block_padded(tc=336, csv=330): ceil(330/16)*16=336 → True!
        # CSV row 2: Q=(336,4,128), avg_seq_len=4096
        _make_attention_op_info(
            query_shape=(336, 4, 128),
            key_shape=(12307, 128, 128),
            seq_lens=[4096] * 330,  # avg=4096, but batch_size=330 ≠ 336
            dtype=torch.bfloat16,
        )
        # This tests T dim tolerance: TC T=336 matches CSV T=336 exactly
        # For a true block-padding test, we need TC T ≠ CSV T
        # CSV has T=496, so TC T=512 should match: ceil(496/16)*16=496, not 512
        # Actually block-padding goes the other way: TC pads, CSV is raw
        # _is_block_padded(tc_T=512, csv_T=496): ceil(496/16)*16=496≠512,
        #   ceil(496/32)*32=512 → True!
        op2 = _make_attention_op_info(
            query_shape=(512, 4, 128),
            key_shape=(12307, 128, 128),
            seq_lens=[4096] * 512,
            dtype=torch.bfloat16,
        )
        result = ds.lookup(op2)
        assert result is not None
        assert abs(result.latency_us - 64.2) < 1.0  # CSV row 3: T=496, 64.2us

    def test_e8_tc_2d_query(self, ds):
        """TC 2D Q=(128, 512) → normalize to (128, 4, 128) → HIT."""
        op = _make_attention_op_info(
            query_shape=(128, 512),
            key_shape=(12307, 128, 128),
            seq_lens=[4096] * 128,
            dtype=torch.bfloat16,
        )
        result = ds.lookup(op)
        assert result is not None
        assert abs(result.latency_us - 58.2) < 1.0

    def test_e9_avg_seq_len_minus1_skipped(self, ds):
        """Row with avg_seq_len=-1 is skipped, no false match."""
        # The -1 row has Duration=55.0 — if it matched we'd get 55.0
        op = _make_attention_op_info(
            query_shape=(128, 4, 128),
            key_shape=(12307, 128, 128),
            seq_lens=[4096] * 128,
            dtype=torch.bfloat16,
        )
        result = ds.lookup(op)
        assert result is not None
        # Should match the avg_seq_len=4096 row (58.2), not the -1 row (55.0)
        assert abs(result.latency_us - 58.2) < 1.0

    def test_no_match_returns_none(self, ds):
        """Shape not in CSV → None."""
        op = _make_attention_op_info(
            query_shape=(999, 4, 128),
            key_shape=(12307, 128, 128),
            seq_lens=[4096] * 999,
            dtype=torch.bfloat16,
        )
        assert ds.lookup(op) is None


class TestQueryByAttnParams:
    """Tests for _query_by_attn_params() shared attention query core."""

    @pytest.fixture
    def ds(self, tmp_path):
        dst = tmp_path / "db"
        shutil.copytree(_FIXTURE_DIR, dst)
        return ProfilingDataSource(str(dst), _make_mock_device_profile())

    def test_exact_match(self, ds):
        """Primary kernel_type matches → returns (latency, kernel_type)."""
        params = {"q_shape_3d": (4, 16, 512), "avg_seq_len": 2048}
        result = ds._query_by_attn_params(
            ["FusedInferAttentionScore"], params, "DT_BF16"
        )
        assert result is not None
        lat, kernel = result
        assert abs(lat - 28.0) < 1.0
        assert kernel == "FusedInferAttentionScore"

    def test_miss(self, ds):
        """Shape not in CSV → None."""
        params = {"q_shape_3d": (99, 16, 512), "avg_seq_len": 2048}
        result = ds._query_by_attn_params(
            ["FusedInferAttentionScore"], params, "DT_BF16"
        )
        assert result is None

    def test_alternate_kernel_fallback(self, ds):
        """Primary misses, alternate kernel hits."""
        params = {"q_shape_3d": (4, 16, 512), "avg_seq_len": 2048}
        # "NoSuchKernel" will miss, "FusedInferAttentionScore" should hit
        result = ds._query_by_attn_params(
            ["NoSuchKernel", "FusedInferAttentionScore"], params, "DT_BF16"
        )
        assert result is not None
        lat, kernel = result
        assert abs(lat - 28.0) < 1.0
        assert kernel == "FusedInferAttentionScore"

    def test_missing_params(self, ds):
        """Missing q_shape_3d → None."""
        result = ds._query_by_attn_params(
            ["FusedInferAttentionScore"], {"avg_seq_len": 2048}, "DT_BF16"
        )
        assert result is None

    def test_block_padding_tolerance(self, ds):
        """TC T=512, CSV T=496 → block-padding match."""
        params = {"q_shape_3d": (512, 4, 128), "avg_seq_len": 4096}
        result = ds._query_by_attn_params(
            ["FusedInferAttentionScore"], params, "DT_BF16"
        )
        assert result is not None
        lat, kernel = result
        assert abs(lat - 64.2) < 1.0
        assert kernel == "FusedInferAttentionScore"


# ---- Helper: build enriched CSV with Runtime columns in tmp_path ----

_ENRICHED_HEADER = (
    "OP State,Accelerator Core,Input Shapes,Input Data Types,Input Formats,"
    "Output Shapes,Output Data Types,Output Formats,Average Duration(us),"
    "Median Duration(us),Std Duration(us),Average aicore_time(us),"
    "Average aic_total_cycles,Average aic_mac_time(us),Average aic_mac_ratio,"
    "Average aic_scalar_time(us),Average aic_scalar_ratio,"
    "Average aic_mte1_time(us),Average aic_mte1_ratio,"
    "Average aic_mte2_time(us),Average aic_mte2_ratio,"
    "Average aic_fixpipe_time(us),Average aic_fixpipe_ratio,"
    "Average aic_icache_miss_rate,Average aiv_time(us),"
    "Average aiv_total_cycles,Average aiv_vec_time(us),"
    "Average aiv_vec_ratio,Average aiv_scalar_time(us),"
    "Average aiv_scalar_ratio,Average aiv_mte2_time(us),"
    "Average aiv_mte2_ratio,Average aiv_mte3_time(us),"
    "Average aiv_mte3_ratio,Average aiv_icache_miss_rate,"
    "Average cube_utilization(%),"
    "avg_seq_len,Runtime sparse_mode,Runtime num_key_value_heads"
)

# Minimal profiling stats placeholder (35 empty fields after Output Formats)
_STATS = ",".join([""] * 27)


def _fia_row(
    q_shape_str, dtype_str, out_shape_str, duration, avg_seq, sparse, kv_heads
):
    """Build one enriched FIA CSV row."""
    return (
        f'dynamic,MIX_AIC,"""{q_shape_str}""",'
        f"{dtype_str},"
        f"ND;ND;ND,"
        f'"""{out_shape_str}""",DT_BF16;FLOAT,ND;ND,'
        f"{duration},{_STATS},"
        f"{avg_seq},{sparse},{kv_heads}"
    )


def _build_enriched_db(tmp_path, rows):
    """Create a tmp db dir with enriched FIA CSV + minimal op_mapping."""
    db = tmp_path / "enriched_db"
    db.mkdir()
    csv_lines = [_ENRICHED_HEADER] + rows
    (db / "FusedInferAttentionScore.csv").write_text(
        "\n".join(csv_lines), encoding="utf-8"
    )
    # Minimal op_mapping
    (db / "op_mapping.yaml").write_text(
        "operator_mappings:\n"
        '  "tensor_cast.attention.default":\n'
        "    kernel_type: FusedInferAttentionScore\n"
        "    query_mode: attention_special\n",
        encoding="utf-8",
    )
    return db


def _make_attention_op_with_query_lens(
    query_shape, key_shape, seq_lens, query_lens, dtype
):
    """Build mock OpInvokeInfo with query_lens for sparse_mode inference."""
    query = torch.zeros(query_shape, dtype=dtype)
    key = torch.zeros(key_shape, dtype=dtype)
    value = torch.zeros(key_shape, dtype=dtype)
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int64)
    query_lens_t = torch.tensor(query_lens, dtype=torch.int64) if query_lens else None
    args = (query, key, value, None, None, None, seq_lens_t, query_lens_t)
    op_info = MagicMock()
    op_info.func.__str__ = lambda self: "torch.ops.tensor_cast.attention.default"
    op_info.func.__repr__ = lambda self: "torch.ops.tensor_cast.attention.default"
    op_info.args = args
    op_info.kwargs = {}
    op_info.out = None
    return op_info


class TestSparseModeMismatch:
    """Tests for sparse_mode matching in _lookup_attention()."""

    @pytest.fixture
    def ds(self, tmp_path):
        rows = [
            # sparse_mode=0 (decode, no_mask), kv_heads=8, avg_seq=4096, 120us
            _fia_row(
                "128,4,128;12307,128,128;12307,128,128;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;",
                "DT_BF16;DT_BF16;DT_BF16" + ";DT_UNDEFINED" * 28,
                "128,4,128;",
                120.0,
                4096,
                0,
                8,
            ),
            # sparse_mode=3 (prefill, causal), kv_heads=8, avg_seq=4096, 65us
            _fia_row(
                "128,4,128;12307,128,128;12307,128,128;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;",
                "DT_BF16;DT_BF16;DT_BF16" + ";DT_UNDEFINED" * 28,
                "128,4,128;",
                65.0,
                4096,
                3,
                8,
            ),
        ]
        db = _build_enriched_db(tmp_path, rows)
        return ProfilingDataSource(str(db), _make_mock_device_profile())

    def test_decode_matches_sparse3(self, ds):
        """Decode (query_lens all 1) → sparse_mode=3 (causal) → HIT 65us.

        Both prefill and decode use sparse_mode=3 in vLLM profiling data.
        MLA decode (sparse_mode=0) goes through the decomposer path, not
        _infer_sparse_mode, so this function always returns 3.
        """
        op = _make_attention_op_with_query_lens(
            (128, 4, 128),
            (12307, 8, 128),
            [4096] * 128,
            [1] * 128,
            torch.bfloat16,
        )
        result = ds.lookup(op)
        assert result is not None
        assert abs(result.latency_us - 65.0) < 1.0

    def test_prefill_matches_sparse3(self, ds):
        """Prefill (query_lens > 1) → sparse_mode=3 → HIT 65us."""
        op = _make_attention_op_with_query_lens(
            (128, 4, 128),
            (12307, 8, 128),
            [4096] * 128,
            [128] * 1,
            torch.bfloat16,
        )
        result = ds.lookup(op)
        assert result is not None
        assert abs(result.latency_us - 65.0) < 1.0

    def test_sparse_mode_mismatch_miss(self, tmp_path):
        """CSV only has sparse_mode=0, decode (sparse_mode=3) → MISS."""
        rows = [
            _fia_row(
                "128,4,128;12307,128,128;12307,128,128;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;",
                "DT_BF16;DT_BF16;DT_BF16" + ";DT_UNDEFINED" * 28,
                "128,4,128;",
                65.0,
                4096,
                0,  # sparse_mode=0 (MLA decode), won't match non-MLA decode=3
                8,
            ),
        ]
        db = _build_enriched_db(tmp_path, rows)
        ds = ProfilingDataSource(str(db), _make_mock_device_profile())
        op = _make_attention_op_with_query_lens(
            (128, 4, 128),
            (12307, 8, 128),
            [4096] * 128,
            [1] * 128,
            torch.bfloat16,
        )
        assert ds.lookup(op) is None


class TestNumKvHeadsMatch:
    """Tests for num_kv_heads matching in _lookup_attention()."""

    @pytest.fixture
    def ds(self, tmp_path):
        rows = [
            # kv_heads=1 (MQA), sparse_mode=3 (causal), avg_seq=4096, 55us
            _fia_row(
                "4,16,512;12307,128,512;12307,128,512;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;",
                "DT_BF16;DT_BF16;DT_BF16" + ";DT_UNDEFINED" * 28,
                "4,16,512;",
                55.0,
                4096,
                3,
                1,
            ),
            # kv_heads=8 (GQA), sparse_mode=3 (causal), avg_seq=4096, 90us
            _fia_row(
                "128,4,128;12307,128,128;12307,128,128;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;",
                "DT_BF16;DT_BF16;DT_BF16" + ";DT_UNDEFINED" * 28,
                "128,4,128;",
                90.0,
                4096,
                3,
                8,
            ),
        ]
        db = _build_enriched_db(tmp_path, rows)
        return ProfilingDataSource(str(db), _make_mock_device_profile())

    def test_mqa_kv_heads_1(self, ds):
        """key shape[-2]=1 → num_kv_heads=1 → HIT 55us."""
        op = _make_attention_op_with_query_lens(
            (4, 16, 512),
            (12307, 1, 512),
            [4096] * 4,
            [1] * 4,
            torch.bfloat16,
        )
        result = ds.lookup(op)
        assert result is not None
        assert abs(result.latency_us - 55.0) < 1.0

    def test_gqa_kv_heads_8(self, ds):
        """key shape[-2]=8 → num_kv_heads=8 → HIT 90us (not 55us)."""
        op = _make_attention_op_with_query_lens(
            (128, 4, 128),
            (12307, 8, 128),
            [4096] * 128,
            [1] * 128,
            torch.bfloat16,
        )
        result = ds.lookup(op)
        assert result is not None
        assert abs(result.latency_us - 90.0) < 1.0

    def test_kv_heads_mismatch(self, tmp_path):
        """CSV only has kv_heads=8, query with kv_heads=1 → MISS."""
        rows = [
            _fia_row(
                "128,4,128;12307,128,128;12307,128,128;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;",
                "DT_BF16;DT_BF16;DT_BF16" + ";DT_UNDEFINED" * 28,
                "128,4,128;",
                90.0,
                4096,
                0,
                8,
            ),
        ]
        db = _build_enriched_db(tmp_path, rows)
        ds = ProfilingDataSource(str(db), _make_mock_device_profile())
        # key with kv_heads=1: 3D (*, 1, head_dim)
        op = _make_attention_op_with_query_lens(
            (128, 4, 128),
            (12307, 1, 128),
            [4096] * 128,
            [1] * 128,
            torch.bfloat16,
        )
        assert ds.lookup(op) is None


class TestBackwardCompatNoRuntimeCols:
    """Old CSV without Runtime columns → skip sparse_mode/kv_heads matching."""

    @pytest.fixture
    def ds(self, tmp_path):
        """Use existing fixture (no Runtime columns)."""
        dst = tmp_path / "db"
        shutil.copytree(_FIXTURE_DIR, dst)
        return ProfilingDataSource(str(dst), _make_mock_device_profile())

    def test_old_csv_still_matches(self, ds):
        """Old CSV without Runtime cols → (N, D, dtype, avg_seq) match only."""
        op = _make_attention_op_with_query_lens(
            (128, 4, 128),
            (12307, 128, 128),
            [4096] * 128,
            [1] * 128,
            torch.bfloat16,
        )
        result = ds.lookup(op)
        assert result is not None
        assert abs(result.latency_us - 58.2) < 1.0


class TestLatencyColPriority:
    """Test _latency_col() priority order.

    Priority: Average Duration (microbench / parse_kernel_details output)
    > Profiling Average Duration (enriched CSV) > Duration (fallback).
    """

    def test_average_duration_first(self):
        """Average Duration wins when all columns present."""
        import pandas as pd

        df = pd.DataFrame(
            {
                "Profiling Average Duration(us)": [2.0],
                "Average Duration(us)": [3.0],
                "Duration(us)": [4.0],
            }
        )
        assert ProfilingDataSource._latency_col(df) == "Average Duration(us)"

    def test_profiling_average_second(self):
        """Profiling Average Duration used when Average Duration absent."""
        import pandas as pd

        df = pd.DataFrame(
            {
                "Profiling Average Duration(us)": [2.0],
                "Duration(us)": [4.0],
            }
        )
        assert ProfilingDataSource._latency_col(df) == "Profiling Average Duration(us)"

    def test_average_only(self):
        import pandas as pd

        df = pd.DataFrame({"Average Duration(us)": [3.0]})
        assert ProfilingDataSource._latency_col(df) == "Average Duration(us)"

    def test_duration_fallback(self):
        import pandas as pd

        df = pd.DataFrame({"Duration(us)": [4.0]})
        assert ProfilingDataSource._latency_col(df) == "Duration(us)"

    def test_no_col_returns_duration(self):
        import pandas as pd

        df = pd.DataFrame({"other": [1.0]})
        assert ProfilingDataSource._latency_col(df) == "Duration(us)"


# ---- input_layout tie-breaker tests ----

_ENRICHED_HEADER_WITH_LAYOUT = (
    "OP State,Accelerator Core,Input Shapes,Input Data Types,Input Formats,"
    "Output Shapes,Output Data Types,Output Formats,Average Duration(us),"
    "Median Duration(us),Std Duration(us),Average aicore_time(us),"
    "Average aic_total_cycles,Average aic_mac_time(us),Average aic_mac_ratio,"
    "Average aic_scalar_time(us),Average aic_scalar_ratio,"
    "Average aic_mte1_time(us),Average aic_mte1_ratio,"
    "Average aic_mte2_time(us),Average aic_mte2_ratio,"
    "Average aic_fixpipe_time(us),Average aic_fixpipe_ratio,"
    "Average aic_icache_miss_rate,Average aiv_time(us),"
    "Average aiv_total_cycles,Average aiv_vec_time(us),"
    "Average aiv_vec_ratio,Average aiv_scalar_time(us),"
    "Average aiv_scalar_ratio,Average aiv_mte2_time(us),"
    "Average aiv_mte2_ratio,Average aiv_mte3_time(us),"
    "Average aiv_mte3_ratio,Average aiv_icache_miss_rate,"
    "Average cube_utilization(%),"
    "avg_seq_len,Runtime sparse_mode,Runtime num_key_value_heads,"
    "Runtime input_layout"
)


def _fia_row_with_layout(
    q_shape_str, dtype_str, out_shape_str, duration, avg_seq, sparse, kv_heads, layout
):
    """Build one enriched FIA CSV row with input_layout column."""
    return (
        f'dynamic,MIX_AIC,"""{q_shape_str}""",'
        f"{dtype_str},"
        f"ND;ND;ND,"
        f'"""{out_shape_str}""",DT_BF16;FLOAT,ND;ND,'
        f"{duration},{_STATS},"
        f"{avg_seq},{sparse},{kv_heads},{layout}"
    )


def _build_enriched_db_with_layout(tmp_path, rows, subdir="enriched_layout_db"):
    """Create a tmp db dir with enriched FIA CSV (with layout col) + minimal op_mapping."""
    db = tmp_path / subdir
    db.mkdir()
    csv_lines = [_ENRICHED_HEADER_WITH_LAYOUT] + rows
    (db / "FusedInferAttentionScore.csv").write_text(
        "\n".join(csv_lines), encoding="utf-8"
    )
    (db / "op_mapping.yaml").write_text(
        "operator_mappings:\n"
        '  "tensor_cast.attention.default":\n'
        "    kernel_type: FusedInferAttentionScore\n"
        "    query_mode: attention_special\n",
        encoding="utf-8",
    )
    return db


class TestInputLayoutTieBreaker:
    """Tests for input_layout tie-breaker in _query_by_attn_params."""

    @pytest.fixture
    def ds(self, tmp_path):
        rows = [
            # TND (prefill), kv_heads=4, sparse=3, avg_seq=4096, 70us
            _fia_row_with_layout(
                "128,4,128;12307,128,128;12307,128,128;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;",
                "DT_BF16;DT_BF16;DT_BF16" + ";DT_UNDEFINED" * 28,
                "128,4,128;",
                70.0,
                4096,
                3,
                4,
                "TND",
            ),
            # BNSD_NBSD (decode), kv_heads=4, sparse=0, avg_seq=4096, 30us
            _fia_row_with_layout(
                "128,4,128;12307,128,128;12307,128,128;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;",
                "DT_BF16;DT_BF16;DT_BF16" + ";DT_UNDEFINED" * 28,
                "128,4,128;",
                30.0,
                4096,
                0,
                4,
                "BNSD_NBSD",
            ),
        ]
        db = _build_enriched_db_with_layout(tmp_path, rows)
        return ProfilingDataSource(str(db), _make_mock_device_profile())

    def test_layout_tnd_selects_prefill(self, ds):
        """input_layout=TND matches TND row (70us), not BNSD_NBSD (30us)."""
        params = {
            "q_shape_3d": (128, 4, 128),
            "avg_seq_len": 4096,
            "sparse_mode": 3,
            "num_kv_heads": 4,
            "input_layout": "TND",
        }
        result = ds._query_by_attn_params(
            ["FusedInferAttentionScore"], params, "DT_BF16"
        )
        assert result is not None
        lat, kernel = result
        assert abs(lat - 70.0) < 1.0

    def test_layout_bnsd_selects_decode(self, ds):
        """input_layout=BNSD_NBSD matches decode row (30us)."""
        params = {
            "q_shape_3d": (128, 4, 128),
            "avg_seq_len": 4096,
            "sparse_mode": 0,
            "num_kv_heads": 4,
            "input_layout": "BNSD_NBSD",
        }
        result = ds._query_by_attn_params(
            ["FusedInferAttentionScore"], params, "DT_BF16"
        )
        assert result is not None
        lat, kernel = result
        assert abs(lat - 30.0) < 1.0

    def test_layout_none_still_matches(self, tmp_path):
        """input_layout=None skips layout filtering and uses other match signals."""
        rows = [
            _fia_row_with_layout(
                "128,4,128;12307,128,128;12307,128,128;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;",
                "DT_BF16;DT_BF16;DT_BF16" + ";DT_UNDEFINED" * 28,
                "128,4,128;",
                70.0,
                4096,
                3,
                4,
                "TND",
            ),
            _fia_row_with_layout(
                "128,4,128;12307,128,128;12307,128,128;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;",
                "DT_BF16;DT_BF16;DT_BF16" + ";DT_UNDEFINED" * 28,
                "128,4,128;",
                30.0,
                8192,
                3,
                4,
                "BNSD_NBSD",
            ),
        ]
        db = _build_enriched_db_with_layout(tmp_path, rows, "layout_none_db")
        ds = ProfilingDataSource(str(db), _make_mock_device_profile())
        params = {
            "q_shape_3d": (128, 4, 128),
            "avg_seq_len": 8192,
            "sparse_mode": 3,
            "num_kv_heads": 4,
            "input_layout": None,
        }
        result = ds._query_by_attn_params(
            ["FusedInferAttentionScore"], params, "DT_BF16"
        )
        assert result is not None
        lat, kernel = result
        assert abs(lat - 30.0) < 1.0
        assert kernel == "FusedInferAttentionScore"

    def test_layout_mismatch_miss(self, tmp_path):
        """CSV only has TND, query with BNSD_NBSD + sparse=3 → MISS."""
        rows = [
            _fia_row_with_layout(
                "128,4,128;12307,128,128;12307,128,128;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;",
                "DT_BF16;DT_BF16;DT_BF16" + ";DT_UNDEFINED" * 28,
                "128,4,128;",
                70.0,
                4096,
                3,
                4,
                "TND",
            ),
        ]
        db = _build_enriched_db_with_layout(tmp_path, rows, "layout_miss_db")
        ds = ProfilingDataSource(str(db), _make_mock_device_profile())
        params = {
            "q_shape_3d": (128, 4, 128),
            "avg_seq_len": 4096,
            "sparse_mode": 3,
            "num_kv_heads": 4,
            "input_layout": "BNSD_NBSD",
        }
        result = ds._query_by_attn_params(
            ["FusedInferAttentionScore"], params, "DT_BF16"
        )
        assert result is None


class TestInputLayoutFromLookupAttention:
    """Tests that _lookup_attention derives input_layout from query ndim."""

    @pytest.fixture
    def ds_with_layout(self, tmp_path):
        # Two rows same shape but different layout
        rows = [
            # TND (prefill), kv_heads=8, sparse=3, avg_seq=4096, 70us
            _fia_row_with_layout(
                "128,4,128;12307,128,128;12307,128,128;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;",
                "DT_BF16;DT_BF16;DT_BF16" + ";DT_UNDEFINED" * 28,
                "128,4,128;",
                70.0,
                4096,
                3,
                8,
                "TND",
            ),
            # BNSD_NBSD (decode), kv_heads=8, sparse=3 (causal), avg_seq=4096, 30us
            _fia_row_with_layout(
                "128,4,128;12307,128,128;12307,128,128;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;",
                "DT_BF16;DT_BF16;DT_BF16" + ";DT_UNDEFINED" * 28,
                "128,4,128;",
                30.0,
                4096,
                3,
                8,
                "BNSD_NBSD",
            ),
        ]
        db = _build_enriched_db_with_layout(tmp_path, rows, "layout_e2e_db")
        return ProfilingDataSource(str(db), _make_mock_device_profile())

    def test_3d_query_derives_tnd(self, ds_with_layout):
        """3D query shape → input_layout=TND → matches TND row."""
        op = _make_attention_op_with_query_lens(
            (128, 4, 128),
            (12307, 8, 128),
            [4096] * 128,
            [128] * 1,  # prefill → sparse_mode=3
            torch.bfloat16,
        )
        result = ds_with_layout.lookup(op)
        assert result is not None
        assert abs(result.latency_us - 70.0) < 1.0

    def test_4d_query_derives_bnsd(self, ds_with_layout):
        """4D query shape → input_layout=BNSD_NBSD → matches BNSD row."""
        # 4D query: (B, N, S, D) → normalize to 3D (B, N, D) since S=1
        op = _make_attention_op_with_query_lens(
            (128, 4, 1, 128),
            (12307, 8, 128),
            [4096] * 128,
            [1] * 128,  # decode → sparse_mode=0
            torch.bfloat16,
        )
        result = ds_with_layout.lookup(op)
        assert result is not None
        assert abs(result.latency_us - 30.0) < 1.0


# ---- MISS reason granularity tests ----


class TestAttentionMissReason:
    """Tests for fine-grained miss reasons in attention lookup."""

    def test_csv_not_found_reason(self, tmp_path):
        """No CSV file for kernel → csv_not_found."""
        db = tmp_path / "empty_db"
        db.mkdir()
        (db / "op_mapping.yaml").write_text(
            "operator_mappings:\n"
            '  "tensor_cast.attention.default":\n'
            "    kernel_type: NonExistentKernel\n"
            "    query_mode: attention_special\n",
            encoding="utf-8",
        )
        ds = ProfilingDataSource(str(db), _make_mock_device_profile())
        op = _make_attention_op_with_query_lens(
            (128, 4, 128),
            (12307, 8, 128),
            [4096] * 128,
            [1] * 128,
            torch.bfloat16,
        )
        result = ds.lookup(op)
        assert result is None
        assert ds.last_miss_reason == "csv_not_found"

    def test_shape_mismatch_reason(self, tmp_path):
        """CSV exists but no matching row → shape_mismatch."""
        rows = [
            _fia_row(
                "128,4,128;12307,128,128;12307,128,128;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;",
                "DT_BF16;DT_BF16;DT_BF16" + ";DT_UNDEFINED" * 28,
                "128,4,128;",
                58.2,
                4096,
                0,
                8,
            ),
        ]
        db = _build_enriched_db(tmp_path, rows)
        ds = ProfilingDataSource(str(db), _make_mock_device_profile())
        # Query with N=99 — won't match
        op = _make_attention_op_with_query_lens(
            (128, 99, 128),
            (12307, 8, 128),
            [4096] * 128,
            [1] * 128,
            torch.bfloat16,
        )
        result = ds.lookup(op)
        assert result is None
        assert ds.last_miss_reason == "shape_mismatch"

    def test_insufficient_args_reason(self, tmp_path):
        """Too few args → insufficient_args."""
        rows = [
            _fia_row(
                "128,4,128;12307,128,128;12307,128,128;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;",
                "DT_BF16;DT_BF16;DT_BF16" + ";DT_UNDEFINED" * 28,
                "128,4,128;",
                58.2,
                4096,
                0,
                8,
            ),
        ]
        db = _build_enriched_db(tmp_path, rows)
        ds = ProfilingDataSource(str(db), _make_mock_device_profile())
        op = MagicMock()
        op.func.__str__ = lambda self: "torch.ops.tensor_cast.attention.default"
        op.func.__repr__ = lambda self: "torch.ops.tensor_cast.attention.default"
        op.args = (torch.zeros(128, 4, 128),)  # only 1 arg
        op.kwargs = {}
        op.out = None
        result = ds.lookup(op)
        assert result is None
        assert ds.last_miss_reason == "insufficient_args"

    def test_missing_seq_lens_reason(self, tmp_path):
        """No seq_lens tensor → missing_seq_lens."""
        rows = [
            _fia_row(
                "128,4,128;12307,128,128;12307,128,128;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;",
                "DT_BF16;DT_BF16;DT_BF16" + ";DT_UNDEFINED" * 28,
                "128,4,128;",
                58.2,
                4096,
                0,
                8,
            ),
        ]
        db = _build_enriched_db(tmp_path, rows)
        ds = ProfilingDataSource(str(db), _make_mock_device_profile())
        query = torch.zeros(128, 4, 128, dtype=torch.bfloat16)
        key = torch.zeros(12307, 8, 128, dtype=torch.bfloat16)
        value = torch.zeros(12307, 8, 128, dtype=torch.bfloat16)
        # seq_lens is None (args[6] = None)
        args = (query, key, value, None, None, None, None, None)
        op = MagicMock()
        op.func.__str__ = lambda self: "torch.ops.tensor_cast.attention.default"
        op.func.__repr__ = lambda self: "torch.ops.tensor_cast.attention.default"
        op.args = args
        op.kwargs = {}
        op.out = None
        result = ds.lookup(op)
        assert result is None
        assert ds.last_miss_reason == "missing_seq_lens"
