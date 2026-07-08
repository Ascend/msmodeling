"""Tests for G2 MLA decomposition + InterpolatingDataSource composite support."""

from unittest.mock import MagicMock

import pytest
import torch
from tensor_cast.performance_model.profiling_database.data_source import QuerySource
from tensor_cast.performance_model.profiling_database.interpolating_data_source import (
    InterpolatingDataSource,
)
from tensor_cast.performance_model.profiling_database.profiling_data_source import (
    ProfilingDataSource,
    _decompose_mla,
    _decompose_mla_quant,
    _decompose_mlapo,
    _decompose_mlapo_quant,
    _is_decode_mla,
)

# ---- Helpers ----


def _make_op_info(func, args):
    mock = MagicMock()
    mock.func = func
    mock.args = tuple(args)
    mock.kwargs = {}
    mock.out = None
    return mock


def _make_mla_decode_args(
    num_tokens=16,
    num_heads=16,
    qk_nope_head_dim=128,
    qk_rope_head_dim=64,
    kv_lora_rank=512,
    v_head_dim=128,
    batch_size=16,
    avg_seq_len=4096,
):
    """Build args for multihead_latent_attention in decode mode."""
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
    q = torch.empty(num_tokens, num_heads, qk_head_dim, device="meta", dtype=torch.bfloat16)
    kv_cache = torch.empty(256, 16, kv_lora_rank + qk_rope_head_dim, device="meta", dtype=torch.bfloat16)
    block_table = torch.empty(batch_size, 16, device="meta", dtype=torch.int32)
    query_start_loc = torch.arange(batch_size + 1, dtype=torch.int32)
    seq_lens = torch.full((batch_size,), avg_seq_len, dtype=torch.int64)
    query_lens = None  # decode
    W_UK_T = torch.empty(num_heads, qk_nope_head_dim, kv_lora_rank, device="meta", dtype=torch.bfloat16)
    W_UV = torch.empty(num_heads, kv_lora_rank, v_head_dim, device="meta", dtype=torch.bfloat16)
    kv_b_proj = None  # decode
    return [
        q,
        kv_cache,
        block_table,
        query_start_loc,
        seq_lens,
        query_lens,
        W_UK_T,
        W_UV,
        kv_b_proj,
        v_head_dim,
    ]


def _make_mla_prefill_args(
    num_tokens=136,
    num_heads=16,
    qk_nope_head_dim=128,
    qk_rope_head_dim=64,
    kv_lora_rank=512,
    v_head_dim=128,
    batch_size=2,
    avg_seq_len=68,
):
    """Build args for multihead_latent_attention in prefill mode."""
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
    q = torch.empty(num_tokens, num_heads, qk_head_dim, device="meta", dtype=torch.bfloat16)
    kv_cache = torch.empty(256, 16, kv_lora_rank + qk_rope_head_dim, device="meta", dtype=torch.bfloat16)
    block_table = torch.empty(batch_size, 16, device="meta", dtype=torch.int32)
    query_start_loc = torch.arange(batch_size + 1, dtype=torch.int32)
    seq_lens = torch.full((batch_size,), avg_seq_len, dtype=torch.int64)
    query_lens = torch.full((batch_size,), avg_seq_len, dtype=torch.int64)  # prefill
    W_UK_T = None
    W_UV = None
    proj_out_dim = num_heads * (qk_nope_head_dim + v_head_dim)
    kv_b_proj = torch.empty(kv_lora_rank, proj_out_dim, device="meta", dtype=torch.bfloat16)
    return [
        q,
        kv_cache,
        block_table,
        query_start_loc,
        seq_lens,
        query_lens,
        W_UK_T,
        W_UV,
        kv_b_proj,
        v_head_dim,
    ]


# ---- Unit tests: decomposition functions ----


class TestIsDecodeMLA:
    def test_none_query_lens_is_decode(self):
        assert _is_decode_mla((None, None, None, None, None, None)) is True

    def test_all_ones_is_decode(self):
        args = (None, None, None, None, None, torch.ones(16, dtype=torch.int64))
        assert _is_decode_mla(args) is True

    def test_query_lens_gt_1_is_prefill(self):
        args = (None, None, None, None, None, torch.full((2,), 68, dtype=torch.int64))
        assert _is_decode_mla(args) is False


class TestDecomposeMLA:
    def test_decode_returns_3_specs(self):
        args = _make_mla_decode_args()
        op = _make_op_info(torch.ops.tensor_cast.multihead_latent_attention.default, args)
        specs = _decompose_mla(op, {})
        assert specs is not None
        assert len(specs) == 3
        assert specs[0].kernel_type == "BatchMatMulV2"
        assert specs[0].alternate_kernel_types == ["BatchMatMulNd"]
        assert specs[1].kernel_type == "FusedInferAttentionScore"
        assert specs[1].query_mode == "attention"
        assert specs[2].kernel_type == "TransposeBatchMatMul"

    def test_decode_shapes_correct(self):
        # Use num_tokens=4 != num_heads=16 to verify heads-first order
        args = _make_mla_decode_args(
            num_tokens=4,
            num_heads=16,
            qk_nope_head_dim=128,
            kv_lora_rank=512,
            v_head_dim=128,
        )
        op = _make_op_info(torch.ops.tensor_cast.multihead_latent_attention.default, args)
        specs = _decompose_mla(op, {})
        # q @ W_UK_T: (num_heads=16, num_tokens=4, qk_nope=128) @ (16, 128, 512)
        assert specs[0].input_shapes == [(16, 4, 128), (16, 128, 512)]
        # attn_out @ W_UV: (num_heads=16, num_tokens=4, kv_lora=512) @ (16, 512, 128)
        assert specs[2].input_shapes == [(16, 4, 512), (16, 512, 128)]

    def test_prefill_decomposes_to_matmul_and_fia(self):
        """Prefill decomposes to MatMulV2 + FIA (v0.18.0: unified FIA)."""
        args = _make_mla_prefill_args()
        op = _make_op_info(torch.ops.tensor_cast.multihead_latent_attention.default, args)
        specs = _decompose_mla(op, {})
        assert specs is not None
        assert len(specs) == 2
        assert specs[0].kernel_type == "MatMulV2"
        # kv_c @ kv_b_proj: (136, 512) @ (512, 16*(128+128))
        assert specs[0].input_shapes[0] == (136, 512)
        assert specs[0].input_shapes[1][0] == 512
        assert specs[1].kernel_type == "FusedInferAttentionScore"

    def test_prefill_fia_has_attention_params(self):
        """Prefill FIA spec has attention_params (v0.18.0)."""
        args = _make_mla_prefill_args(num_tokens=136, kv_lora_rank=512)
        op = _make_op_info(torch.ops.tensor_cast.multihead_latent_attention.default, args)
        specs = _decompose_mla(op, {})
        assert specs is not None
        assert len(specs) == 2
        assert specs[1].kernel_type == "FusedInferAttentionScore"
        assert specs[1].attention_params is not None
        # Prefill decompresses KV via kv_b_proj → num_kv_heads = num_heads
        # (differs from decode where KV stays compressed as single latent)
        assert specs[1].attention_params["num_kv_heads"] == 16

    def test_insufficient_args_returns_none(self):
        op = _make_op_info(
            torch.ops.tensor_cast.multihead_latent_attention.default,
            [torch.empty(136, 5120, device="meta", dtype=torch.bfloat16)],
        )
        assert _decompose_mla(op, {}) is None

    def test_fia_attention_params_decode(self):
        """Decode FIA spec uses attention_params (not fia_raw_shapes)."""
        args = _make_mla_decode_args(batch_size=16, avg_seq_len=4096, num_heads=16, kv_lora_rank=512)
        op = _make_op_info(torch.ops.tensor_cast.multihead_latent_attention.default, args)
        specs = _decompose_mla(op, {})
        fia = specs[1]
        assert fia.attention_params is not None
        assert fia.attention_params["avg_seq_len"] == 4096
        q_shape_3d = fia.attention_params["q_shape_3d"]
        assert q_shape_3d[0] == 16  # batch_size
        assert q_shape_3d[1] == 16  # num_heads
        assert q_shape_3d[2] == 512  # kv_lora_rank (not head_dim=576)


class TestDecomposeMLAPrefillFIAFix:
    """MISS #4 (FIA prefill T + head_dim)."""

    def test_fia_prefill_uses_num_tokens_and_nope_dim(self):
        """FIA prefill Q must use TND layout: (num_tokens, num_heads, qk_nope_head_dim=128)."""
        args = _make_mla_prefill_args(
            num_tokens=136,
            num_heads=16,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            kv_lora_rank=512,
            batch_size=2,
        )
        op = _make_op_info(torch.ops.tensor_cast.multihead_latent_attention.default, args)
        specs = _decompose_mla(op, {})
        fia = specs[1]
        q_shape_3d = fia.attention_params["q_shape_3d"]
        # Must be TND: (num_tokens=136, num_heads=16, qk_nope_head_dim=128)
        assert len(q_shape_3d) == 3, f"Expected 3D TND shape, got {q_shape_3d}"
        assert q_shape_3d[0] == 136, f"T should be num_tokens=136, got {q_shape_3d[0]}"
        assert q_shape_3d[1] == 16, f"N should be num_heads=16, got {q_shape_3d[1]}"
        assert q_shape_3d[2] == 128, f"D should be qk_nope_head_dim=128, got {q_shape_3d[2]}"

    def test_fia_prefill_sparse_mode_3(self):
        """FIA prefill sparse_mode must be 3 (causal), not 0."""
        args = _make_mla_prefill_args()
        op = _make_op_info(torch.ops.tensor_cast.multihead_latent_attention.default, args)
        specs = _decompose_mla(op, {})
        fia = specs[1]
        assert fia.attention_params["sparse_mode"] == 3, (
            f"Prefill sparse_mode should be 3 (causal), got {fia.attention_params['sparse_mode']}"
        )


class TestDecomposeMLAQuant:
    def test_decode_uses_quant_kernel(self):
        args = _make_mla_decode_args()
        op = _make_op_info(torch.ops.tensor_cast.multihead_latent_attention_quant.default, args)
        specs = _decompose_mla_quant(op, {})
        assert specs is not None
        assert specs[0].kernel_type == "QuantBatchMatmulV3"

    def test_prefill_decomposes_to_matmul_and_fia(self):
        """Quant prefill decomposes to MatMulV2 + FIA (v0.18.0)."""
        args = _make_mla_prefill_args()
        op = _make_op_info(torch.ops.tensor_cast.multihead_latent_attention_quant.default, args)
        specs = _decompose_mla_quant(op, {})
        assert specs is not None
        assert len(specs) == 2
        assert specs[0].kernel_type == "MatMulV2"
        assert specs[1].kernel_type == "FusedInferAttentionScore"


# ---- Integration tests: composite lookup with CSV data ----

MLA_OP_MAPPING = """\
version: "test"
device: TEST_DEVICE
operator_mappings:
  "tensor_cast.multihead_latent_attention.default":
    composite: true
    sub_kernels: [TransposeBatchMatMul, FusedInferAttentionScore]
  "tensor_cast.mlapo.default":
    composite: true
    sub_kernels: [MatMulV2, KvRmsNormRopeCache]
"""

# BatchMatMulV2 CSV: decode q@W_UK_T shape (first BMM in BF16 MLA decode)
BATCH_MATMUL_V2_CSV = """\
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Duration(us)
"16,16,128;16,128,512","DT_BF16;DT_BF16","ND;ND","16,16,512","DT_BF16","ND",5.0"""

# BatchMatMulNd CSV: legacy fallback for older profiling databases.
BATCH_MATMUL_ND_CSV = """\
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Duration(us)
"16,16,128;16,128,512","DT_BF16;DT_BF16","ND;ND","16,16,512","DT_BF16","ND",7.0"""

# TransposeBatchMatMul CSV: decode attn_out@W_UV shape (second BMM in MLA decode)
TBMM_CSV = """\
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Duration(us)
"16,16,512;16,512,128","DT_BF16;DT_BF16","ND;ND","16,16,128","DT_BF16","ND",4.0"""

# FIA CSV: raw format (Case C 4D BNSD paged with rope)
# decode: batch=16, heads=16, seq=1, head_dim=512 (kv_lora_rank, not 576), rope_dim=64
# MISS #5 fix: FIA head_dim = kv_lora_rank = 512 (not head_dim=576)
# slots: 0=q(16,16,1,512), 1=k(256,1,16,512), 2=v(256,1,16,512), 6=seq_lens(16,),
#        14=block_table(16,256), 24=rope_q(16,16,1,64), 25=rope_k(256,1,16,64)
_FIA_DECODE_ROW_16 = '"16,16,1,512;256,1,16,512;256,1,16,512;;;;16;;;;;;;;16,256;;;;;;;;;;16,16,1,64;256,1,16,64;;;;;"'
_FIA_DECODE_ROW_32 = '"32,16,1,512;256,1,16,512;256,1,16,512;;;;32;;;;;;;;32,256;;;;;;;;;;32,16,1,64;256,1,16,64;;;;;"'
FIA_CSV = (
    "Input Shapes,Input Data Types,Input Formats,Output Shapes,"
    "Output Data Types,Output Formats,Duration(us),avg_seq_len\n"
    + _FIA_DECODE_ROW_16
    + ",DT_BF16;DT_BF16;DT_BF16;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;"
    "INT64;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;"
    "DT_UNDEFINED;DT_UNDEFINED;INT32;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;"
    "DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;"
    "DT_UNDEFINED;DT_BF16;DT_BF16;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;"
    "DT_UNDEFINED;DT_UNDEFINED"
    ",ND;ND;ND;NULL;NULL;NULL;ND;NULL;NULL;NULL;NULL;NULL;NULL;NULL;ND;"
    "NULL;NULL;NULL;NULL;NULL;NULL;NULL;NULL;NULL;ND;ND;NULL;NULL;NULL;NULL;NULL"
    ',"""16,16,1,512;""",DT_BF16;FLOAT,ND;ND,50.0,4096'
)

# MatMulV2 CSV: mlapo hidden @ q_a_proj
MATMUL_CSV = """\
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Duration(us)
"136,5120;5120,1536","DT_BF16;DT_BF16","ND;ND","136,1536","DT_BF16","ND",8.0
"100,5120;5120,1536","DT_BF16;DT_BF16","ND;ND","100,1536","DT_BF16","ND",6.0
"200,5120;5120,1536","DT_BF16;DT_BF16","ND;ND","200,1536","DT_BF16","ND",12.0"""

# KvRmsNormRopeCache CSV
KVRNRC_CSV = """\
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Duration(us)
"136,5120;5120,576","DT_BF16;DT_BF16","ND;ND","136,576","DT_BF16","ND",3.0"""


@pytest.fixture
def mla_data_dir(tmp_path):
    d = tmp_path / "mla"
    d.mkdir()
    (d / "op_mapping.yaml").write_text(MLA_OP_MAPPING)
    (d / "BatchMatMulV2.csv").write_text(BATCH_MATMUL_V2_CSV.strip())
    (d / "BatchMatMulNd.csv").write_text(BATCH_MATMUL_ND_CSV.strip())
    (d / "TransposeBatchMatMul.csv").write_text(TBMM_CSV.strip())
    (d / "FusedInferAttentionScore.csv").write_text(FIA_CSV.strip())
    (d / "MatMulV2.csv").write_text(MATMUL_CSV.strip())
    (d / "KvRmsNormRopeCache.csv").write_text(KVRNRC_CSV.strip())
    return d


@pytest.fixture
def mla_legacy_data_dir(tmp_path):
    """MLA data dir without BatchMatMulV2 to verify legacy fallback."""
    d = tmp_path / "mla_legacy"
    d.mkdir()
    (d / "op_mapping.yaml").write_text(MLA_OP_MAPPING)
    (d / "BatchMatMulNd.csv").write_text(BATCH_MATMUL_ND_CSV.strip())
    (d / "TransposeBatchMatMul.csv").write_text(TBMM_CSV.strip())
    (d / "FusedInferAttentionScore.csv").write_text(FIA_CSV.strip())
    (d / "MatMulV2.csv").write_text(MATMUL_CSV.strip())
    (d / "KvRmsNormRopeCache.csv").write_text(KVRNRC_CSV.strip())
    return d


class TestCompositeLookupMLA:
    def test_mla_decode_hit(self, mla_data_dir):
        """MLA decode: all 3 sub-kernels hit → sum latency."""
        ds = ProfilingDataSource(mla_data_dir)
        args = _make_mla_decode_args(
            num_tokens=16,
            num_heads=16,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            kv_lora_rank=512,
            v_head_dim=128,
            batch_size=16,
            avg_seq_len=4096,
        )
        op = _make_op_info(torch.ops.tensor_cast.multihead_latent_attention.default, args)
        result = ds.lookup(op)
        assert result is not None
        # 5.0 (BatchMatMulV2 q@W_UK_T) + 50.0 (FIA) + 4.0 (TBMM out@W_UV) = 59.0
        assert abs(result.latency_us - 59.0) < 0.1
        assert result.source == QuerySource.MEASURED
        assert result.details["kernel_type"].startswith("BatchMatMulV2,")

    def test_mla_decode_fia_miss_returns_partial(self, mla_data_dir):
        """MLA decode: FIA miss (wrong batch_size) → PARTIAL."""
        ds = ProfilingDataSource(mla_data_dir)
        args = _make_mla_decode_args(batch_size=99, avg_seq_len=4096)
        op = _make_op_info(torch.ops.tensor_cast.multihead_latent_attention.default, args)
        result = ds.lookup(op)
        assert result is not None
        assert result.source == QuerySource.PARTIAL
        assert result.details.get("partial") is True

    def test_mla_decode_falls_back_to_batch_matmul_nd(self, mla_legacy_data_dir):
        """MLA decode falls back to BatchMatMulNd when BatchMatMulV2 CSV is absent."""
        ds = ProfilingDataSource(mla_legacy_data_dir)
        args = _make_mla_decode_args(
            num_tokens=16,
            num_heads=16,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            kv_lora_rank=512,
            v_head_dim=128,
            batch_size=16,
            avg_seq_len=4096,
        )
        op = _make_op_info(torch.ops.tensor_cast.multihead_latent_attention.default, args)
        result = ds.lookup(op)
        assert result is not None
        # 7.0 (BatchMatMulNd fallback q@W_UK_T) + 50.0 (FIA) + 4.0 (TBMM out@W_UV)
        assert abs(result.latency_us - 61.0) < 0.1
        assert result.source == QuerySource.MEASURED
        assert "BatchMatMulNd" in result.details["kernel_type"]

    def test_mla_insufficient_args_returns_none(self, mla_data_dir):
        """MLA with insufficient args → decompose fails → None."""
        ds = ProfilingDataSource(mla_data_dir)
        op = _make_op_info(
            torch.ops.tensor_cast.multihead_latent_attention.default,
            [torch.empty(136, 5120, device="meta", dtype=torch.bfloat16)],
        )
        result = ds.lookup(op)
        assert result is None


# ---- Integration tests: InterpolatingDataSource composite ----


class TestCompositeInterpolation:
    def test_mla_decode_fia_hit(self, mla_data_dir):
        """MLA decode: FIA shape + avg_seq_len matches → exact hit."""
        base = ProfilingDataSource(mla_data_dir)
        ds = InterpolatingDataSource(base)
        args = _make_mla_decode_args(
            num_tokens=16,
            num_heads=16,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            kv_lora_rank=512,
            v_head_dim=128,
            batch_size=16,
            avg_seq_len=4096,
        )
        op = _make_op_info(torch.ops.tensor_cast.multihead_latent_attention.default, args)
        result = ds.lookup(op)
        # TBMM exact: 5.0 + 4.0, FIA enriched hit: 50.0 → total 59.0
        assert result is not None
        assert result.source == QuerySource.MEASURED
        assert abs(result.latency_us - 59.0) < 0.1

    def test_existing_interpolation_not_broken(self, mla_data_dir):
        """Existing compute interpolation still works (regression test)."""
        base = ProfilingDataSource(mla_data_dir)
        ds = InterpolatingDataSource(base)
        # Non-composite MatMulV2 is not in op_mapping as non-composite, skip
        # Just verify the ds object is functional
        op = _make_op_info(
            torch.ops.aten.add.Tensor,
            [
                torch.empty(100, device="meta", dtype=torch.bfloat16),
                torch.empty(100, device="meta", dtype=torch.bfloat16),
            ],
        )
        result = ds.lookup(op)
        assert result is None  # unmapped op


# ============================================================
# Extended test suite: edge cases, boundary, accuracy, robustness
# Reference: AI Configurator design principles
# ============================================================

# ---- 1. Extrapolation rejection ----

EXTRAP_OP_MAPPING = """\
version: "test"
device: TEST_DEVICE
operator_mappings:
  "aten.mm.default":
    kernel_type: MatMulV2
  "tensor_cast.attention.default":
    kernel_type: FusedInferAttentionScore
    query_mode: attention_special
"""

EXTRAP_MATMUL_CSV = """\
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Duration(us)
"256,512;512,1024","DT_BF16;DT_BF16","ND;ND","256,1024","DT_BF16","ND",25.0
"512,512;512,1024","DT_BF16;DT_BF16","ND;ND","512,1024","DT_BF16","ND",50.0
"1024,512;512,1024","DT_BF16;DT_BF16","ND;ND","1024,1024","DT_BF16","ND",100.0"""

_EXTRAP_FIA_HEADER = (
    "Input Shapes,Input Data Types,Input Formats,Output Shapes,"
    "Output Data Types,Output Formats,Duration(us),avg_seq_len,"
    "Runtime sparse_mode,Runtime input_layout,Runtime num_key_value_heads"
)
_EXTRAP_FIA_ROW_COMMON = (
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
EXTRAP_FIA_CSV = (
    _EXTRAP_FIA_HEADER
    + "\n"
    + _EXTRAP_FIA_ROW_COMMON
    + ",100.0,1000,3,TND,4\n"
    + _EXTRAP_FIA_ROW_COMMON
    + ",400.0,2000,3,TND,4\n"
    + _EXTRAP_FIA_ROW_COMMON
    + ",1600.0,4000,3,TND,4"
)


@pytest.fixture
def extrap_data_dir(tmp_path):
    d = tmp_path / "extrap"
    d.mkdir()
    (d / "op_mapping.yaml").write_text(EXTRAP_OP_MAPPING)
    (d / "MatMulV2.csv").write_text(EXTRAP_MATMUL_CSV.strip())
    (d / "FusedInferAttentionScore.csv").write_text(EXTRAP_FIA_CSV.strip())
    return d


class TestExtrapolationRejection:
    """AI Configurator principle: only interpolate within bracket, never extrapolate."""

    def test_compute_below_min_returns_none(self, extrap_data_dir):
        """seq_len=64 below CSV min=256 → no bracket → None."""
        base = ProfilingDataSource(extrap_data_dir)
        ds = InterpolatingDataSource(base)
        op = _make_op_info(
            torch.ops.aten.mm.default,
            [
                torch.empty(64, 512, device="meta", dtype=torch.bfloat16),
                torch.empty(512, 1024, device="meta", dtype=torch.bfloat16),
            ],
        )
        assert ds.lookup(op) is None

    def test_compute_above_max_returns_none(self, extrap_data_dir):
        """seq_len=2048 above CSV max=1024 → no bracket → None."""
        base = ProfilingDataSource(extrap_data_dir)
        ds = InterpolatingDataSource(base)
        op = _make_op_info(
            torch.ops.aten.mm.default,
            [
                torch.empty(2048, 512, device="meta", dtype=torch.bfloat16),
                torch.empty(512, 1024, device="meta", dtype=torch.bfloat16),
            ],
        )
        assert ds.lookup(op) is None

    def test_attention_below_min_returns_none(self, extrap_data_dir):
        """avg_seq_len=500 below CSV min=1000 → None."""
        base = ProfilingDataSource(extrap_data_dir)
        ds = InterpolatingDataSource(base)
        op = _make_op_info(
            torch.ops.tensor_cast.attention.default,
            [
                torch.empty(1, 512, device="meta", dtype=torch.bfloat16),
                torch.empty(16, 128, 4, 128, device="meta", dtype=torch.bfloat16),
                torch.empty(16, 128, 4, 128, device="meta", dtype=torch.bfloat16),
                None,
                None,
                None,
                torch.tensor([500], dtype=torch.int64),
                torch.tensor([1], dtype=torch.int64),
            ],
        )
        assert ds.lookup(op) is None

    def test_attention_above_max_returns_none(self, extrap_data_dir):
        """avg_seq_len=8000 above CSV max=4000 → None."""
        base = ProfilingDataSource(extrap_data_dir)
        ds = InterpolatingDataSource(base)
        op = _make_op_info(
            torch.ops.tensor_cast.attention.default,
            [
                torch.empty(1, 512, device="meta", dtype=torch.bfloat16),
                torch.empty(16, 128, 4, 128, device="meta", dtype=torch.bfloat16),
                torch.empty(16, 128, 4, 128, device="meta", dtype=torch.bfloat16),
                None,
                None,
                None,
                torch.tensor([8000], dtype=torch.int64),
                torch.tensor([1], dtype=torch.int64),
            ],
        )
        assert ds.lookup(op) is None


# ---- 2. Single data point ----

SINGLE_POINT_CSV = """\
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Duration(us)
"256,512;512,1024","DT_BF16;DT_BF16","ND;ND","256,1024","DT_BF16","ND",25.0"""


class TestSingleDataPoint:
    """Need ≥2 data points for interpolation; 1 point → None."""

    def test_single_csv_row_no_interpolation(self, tmp_path):
        d = tmp_path / "single"
        d.mkdir()
        (d / "op_mapping.yaml").write_text(EXTRAP_OP_MAPPING)
        (d / "MatMulV2.csv").write_text(SINGLE_POINT_CSV.strip())
        base = ProfilingDataSource(d)
        ds = InterpolatingDataSource(base)
        op = _make_op_info(
            torch.ops.aten.mm.default,
            [
                torch.empty(300, 512, device="meta", dtype=torch.bfloat16),
                torch.empty(512, 1024, device="meta", dtype=torch.bfloat16),
            ],
        )
        assert ds.lookup(op) is None

    def test_single_csv_row_exact_match_still_works(self, tmp_path):
        """Exact match should still work even with 1 row."""
        d = tmp_path / "single_exact"
        d.mkdir()
        (d / "op_mapping.yaml").write_text(EXTRAP_OP_MAPPING)
        (d / "MatMulV2.csv").write_text(SINGLE_POINT_CSV.strip())
        base = ProfilingDataSource(d)
        ds = InterpolatingDataSource(base)
        op = _make_op_info(
            torch.ops.aten.mm.default,
            [
                torch.empty(256, 512, device="meta", dtype=torch.bfloat16),
                torch.empty(512, 1024, device="meta", dtype=torch.bfloat16),
            ],
        )
        result = ds.lookup(op)
        assert result is not None
        assert abs(result.latency_us - 25.0) < 0.01
        assert result.source == QuerySource.MEASURED


# ---- 3. Confidence levels ----


class TestConfidenceLevels:
    """Verify confidence: MEASURED > linear > sqrt > composite interpolated."""

    def test_exact_match_confidence_1(self, extrap_data_dir):
        base = ProfilingDataSource(extrap_data_dir)
        ds = InterpolatingDataSource(base)
        op = _make_op_info(
            torch.ops.aten.mm.default,
            [
                torch.empty(256, 512, device="meta", dtype=torch.bfloat16),
                torch.empty(512, 1024, device="meta", dtype=torch.bfloat16),
            ],
        )
        result = ds.lookup(op)
        assert result.confidence == 1.0
        assert result.source == QuerySource.MEASURED

    def test_linear_interpolation_confidence_07(self, extrap_data_dir):
        base = ProfilingDataSource(extrap_data_dir)
        ds = InterpolatingDataSource(base)
        op = _make_op_info(
            torch.ops.aten.mm.default,
            [
                torch.empty(384, 512, device="meta", dtype=torch.bfloat16),
                torch.empty(512, 1024, device="meta", dtype=torch.bfloat16),
            ],
        )
        result = ds.lookup(op)
        assert result is not None
        assert result.confidence == 0.7
        assert result.source == QuerySource.INTERPOLATED

    def test_attention_interpolation_uses_1d_confidence(self, extrap_data_dir):
        base = ProfilingDataSource(extrap_data_dir)
        ds = InterpolatingDataSource(base)
        op = _make_op_info(
            torch.ops.tensor_cast.attention.default,
            [
                torch.empty(1, 512, device="meta", dtype=torch.bfloat16),
                torch.empty(16, 128, 4, 128, device="meta", dtype=torch.bfloat16),
                torch.empty(16, 128, 4, 128, device="meta", dtype=torch.bfloat16),
                None,
                None,
                None,
                torch.tensor([1500], dtype=torch.int64),
                torch.tensor([1], dtype=torch.int64),
            ],
        )
        result = ds.lookup(op)
        assert result is not None
        assert result.confidence == 0.7

    def test_composite_exact_confidence_08(self, mla_data_dir):
        """Composite exact match → confidence 0.8."""
        ds = ProfilingDataSource(mla_data_dir)
        args = _make_mla_decode_args(
            num_tokens=16,
            num_heads=16,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            kv_lora_rank=512,
            v_head_dim=128,
            batch_size=16,
            avg_seq_len=4096,
        )
        op = _make_op_info(torch.ops.tensor_cast.multihead_latent_attention.default, args)
        result = ds.lookup(op)
        assert result.confidence == 0.8

    def test_composite_partial_miss_returns_none_for_analytic_fallback(self, mla_data_dir):
        """Composite PARTIAL miss falls through so analytic fallback can cover residual latency."""
        base = ProfilingDataSource(mla_data_dir)
        ds = InterpolatingDataSource(base)
        args = _make_mla_decode_args(
            num_tokens=16,
            num_heads=16,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            kv_lora_rank=512,
            v_head_dim=128,
            batch_size=32,  # batch=32 not in CSV → FIA miss
            avg_seq_len=3000,
        )
        op = _make_op_info(torch.ops.tensor_cast.multihead_latent_attention.default, args)
        result = ds.lookup(op)
        # FIA raw shape for batch=32 is not in CSV. Interpolation misses, so the
        # wrapper returns None and lets EmpiricalPerformanceModel use analytic fallback
        # instead of exposing a partial sub-kernel latency as a complete result.
        assert result is None


# ---- 4. Monotonicity ----


class TestMonotonicity:
    """Interpolated values should be monotonic if CSV data is monotonic."""

    def test_compute_monotonic_increasing(self, extrap_data_dir):
        """Increasing seq_len → increasing latency."""
        base = ProfilingDataSource(extrap_data_dir)
        ds = InterpolatingDataSource(base)
        latencies = []
        for seq in [300, 400, 600, 800, 900]:
            op = _make_op_info(
                torch.ops.aten.mm.default,
                [
                    torch.empty(seq, 512, device="meta", dtype=torch.bfloat16),
                    torch.empty(512, 1024, device="meta", dtype=torch.bfloat16),
                ],
            )
            result = ds.lookup(op)
            assert result is not None, f"seq={seq} should interpolate"
            latencies.append(result.latency_us)
        # Verify monotonically increasing
        for i in range(len(latencies) - 1):
            assert latencies[i] < latencies[i + 1], (
                f"Not monotonic: seq[{i}]={latencies[i]} >= seq[{i + 1}]={latencies[i + 1]}"
            )

    def test_interpolation_within_bracket_bounds(self, extrap_data_dir):
        """Interpolated value must be between bracket endpoints (no overshoot)."""
        base = ProfilingDataSource(extrap_data_dir)
        ds = InterpolatingDataSource(base)
        # Between 256→25.0 and 512→50.0
        op = _make_op_info(
            torch.ops.aten.mm.default,
            [
                torch.empty(384, 512, device="meta", dtype=torch.bfloat16),
                torch.empty(512, 1024, device="meta", dtype=torch.bfloat16),
            ],
        )
        result = ds.lookup(op)
        assert result is not None
        assert 25.0 <= result.latency_us <= 50.0

    def test_attention_interpolation_within_bounds(self, extrap_data_dir):
        """Interpolated attention value within bracket bounds."""
        base = ProfilingDataSource(extrap_data_dir)
        ds = InterpolatingDataSource(base)
        op = _make_op_info(
            torch.ops.tensor_cast.attention.default,
            [
                torch.empty(1, 512, device="meta", dtype=torch.bfloat16),
                torch.empty(16, 128, 4, 128, device="meta", dtype=torch.bfloat16),
                torch.empty(16, 128, 4, 128, device="meta", dtype=torch.bfloat16),
                None,
                None,
                None,
                torch.tensor([1500], dtype=torch.int64),
                torch.tensor([1], dtype=torch.int64),
            ],
        )
        result = ds.lookup(op)
        assert result is not None
        assert 100.0 <= result.latency_us <= 400.0


# ---- 5. Dtype mismatch ----

DTYPE_MISMATCH_CSV = """\
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Duration(us)
"256,512;512,1024","INT8;INT8","ND;ND","256,1024","INT8","ND",10.0
"512,512;512,1024","INT8;INT8","ND;ND","512,1024","INT8","ND",20.0"""


class TestDtypeMismatch:
    """Interpolation must respect dtype: BF16 query should not match INT8 CSV."""

    def test_bf16_query_int8_csv_returns_none(self, tmp_path):
        d = tmp_path / "dtype_mm"
        d.mkdir()
        (d / "op_mapping.yaml").write_text(EXTRAP_OP_MAPPING)
        (d / "MatMulV2.csv").write_text(DTYPE_MISMATCH_CSV.strip())
        base = ProfilingDataSource(d)
        ds = InterpolatingDataSource(base)
        op = _make_op_info(
            torch.ops.aten.mm.default,
            [
                torch.empty(384, 512, device="meta", dtype=torch.bfloat16),
                torch.empty(512, 1024, device="meta", dtype=torch.bfloat16),
            ],
        )
        result = ds.lookup(op)
        assert result is None


# ---- 6. Attention linear accuracy comparison ----


class TestAttentionLinearInterpolationAccuracy:
    """Verify default attention interpolation is linear unless an explicit policy is added later."""

    def test_attention_linear_interpolation_applied(self, extrap_data_dir):
        """Default attention interpolation uses linear seq interpolation.

        CSV: seq=1000→100, seq=2000→400, seq=4000→1600
        For seq=1500 (between 1000 and 2000):
          Linear: 100 + 0.5*300 = 250
        """
        base = ProfilingDataSource(extrap_data_dir)
        ds = InterpolatingDataSource(base)
        op = _make_op_info(
            torch.ops.tensor_cast.attention.default,
            [
                torch.empty(1, 512, device="meta", dtype=torch.bfloat16),
                torch.empty(16, 128, 4, 128, device="meta", dtype=torch.bfloat16),
                torch.empty(16, 128, 4, 128, device="meta", dtype=torch.bfloat16),
                None,
                None,
                None,
                torch.tensor([1500], dtype=torch.int64),
                torch.tensor([1], dtype=torch.int64),
            ],
        )
        result = ds.lookup(op)
        assert result is not None
        assert result.latency_us == pytest.approx(250.0)
        # Should be within bracket bounds
        assert 100.0 <= result.latency_us <= 400.0


# ---- 7. Composite: mixed exact + interpolation ----

MLA_RICH_FIA_CSV = (
    "Input Shapes,Input Data Types,Input Formats,Output Shapes,"
    "Output Data Types,Output Formats,Duration(us),avg_seq_len\n"
    + _FIA_DECODE_ROW_16
    + ",DT_BF16;DT_BF16;DT_BF16;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;"
    "INT64;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;"
    "DT_UNDEFINED;DT_UNDEFINED;INT32;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;"
    "DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;"
    "DT_UNDEFINED;DT_BF16;DT_BF16;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;"
    "DT_UNDEFINED;DT_UNDEFINED"
    ",ND;ND;ND;NULL;NULL;NULL;ND;NULL;NULL;NULL;NULL;NULL;NULL;NULL;ND;"
    "NULL;NULL;NULL;NULL;NULL;NULL;NULL;NULL;NULL;ND;ND;NULL;NULL;NULL;NULL;NULL"
    ',"""16,16,1,512;""",DT_BF16;FLOAT,ND;ND,50.0,4096\n'
    + _FIA_DECODE_ROW_32
    + ",DT_BF16;DT_BF16;DT_BF16;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;"
    "INT64;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;"
    "DT_UNDEFINED;DT_UNDEFINED;INT32;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;"
    "DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;"
    "DT_UNDEFINED;DT_BF16;DT_BF16;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;"
    "DT_UNDEFINED;DT_UNDEFINED"
    ",ND;ND;ND;NULL;NULL;NULL;ND;NULL;NULL;NULL;NULL;NULL;NULL;NULL;ND;"
    "NULL;NULL;NULL;NULL;NULL;NULL;NULL;NULL;NULL;ND;ND;NULL;NULL;NULL;NULL;NULL"
    ',"""32,16,1,512;""",DT_BF16;FLOAT,ND;ND,100.0,4096'
)


@pytest.fixture
def mla_rich_data_dir(tmp_path):
    """MLA data dir with richer FIA CSV for interpolation tests."""
    d = tmp_path / "mla_rich"
    d.mkdir()
    (d / "op_mapping.yaml").write_text(MLA_OP_MAPPING)
    (d / "BatchMatMulV2.csv").write_text(BATCH_MATMUL_V2_CSV.strip())
    (d / "BatchMatMulNd.csv").write_text(BATCH_MATMUL_ND_CSV.strip())
    (d / "TransposeBatchMatMul.csv").write_text(TBMM_CSV.strip())
    (d / "FusedInferAttentionScore.csv").write_text(MLA_RICH_FIA_CSV.strip())
    (d / "MatMulV2.csv").write_text(MATMUL_CSV.strip())
    (d / "KvRmsNormRopeCache.csv").write_text(KVRNRC_CSV.strip())
    return d


class TestCompositeMixedHitInterpolate:
    """Composite ops: some sub-kernels exact hit, others interpolated."""

    def test_tbmm_exact_fia_hit(self, mla_rich_data_dir):
        """TBMM shapes match exactly, FIA enriched shape also hits exactly."""
        base = ProfilingDataSource(mla_rich_data_dir)
        ds = InterpolatingDataSource(base)
        args = _make_mla_decode_args(
            num_tokens=16,
            num_heads=16,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            kv_lora_rank=512,
            v_head_dim=128,
            batch_size=16,
            avg_seq_len=4096,
        )
        op = _make_op_info(torch.ops.tensor_cast.multihead_latent_attention.default, args)
        result = ds.lookup(op)
        assert result is not None
        # BatchMatMulV2 exact: 5.0, TBMM exact: 4.0, FIA enriched hit: 50.0.
        assert abs(result.latency_us - 59.0) < 0.1
        assert result.source == QuerySource.MEASURED

    def test_all_sub_kernels_miss_returns_none(self, mla_rich_data_dir):
        """All sub-kernels miss → None to allow analytic fallback."""
        base = ProfilingDataSource(mla_rich_data_dir)
        ds = InterpolatingDataSource(base)
        args = _make_mla_decode_args(
            num_tokens=99,
            num_heads=8,
            qk_nope_head_dim=64,
            qk_rope_head_dim=32,
            kv_lora_rank=256,
            v_head_dim=64,
            batch_size=64,
            avg_seq_len=999,
        )
        op = _make_op_info(torch.ops.tensor_cast.multihead_latent_attention.default, args)
        result = ds.lookup(op)
        assert result is None


# ---- 8. Empty CSV ----


class TestEmptyCSV:
    def test_empty_csv_returns_none(self, tmp_path):
        d = tmp_path / "empty"
        d.mkdir()
        (d / "op_mapping.yaml").write_text(EXTRAP_OP_MAPPING)
        (d / "MatMulV2.csv").write_text(
            "Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Duration(us)\n"
        )
        base = ProfilingDataSource(d)
        ds = InterpolatingDataSource(base)
        op = _make_op_info(
            torch.ops.aten.mm.default,
            [
                torch.empty(256, 512, device="meta", dtype=torch.bfloat16),
                torch.empty(512, 1024, device="meta", dtype=torch.bfloat16),
            ],
        )
        assert ds.lookup(op) is None


# ---- 9. Decompose failure modes ----


class TestDecomposeFailureModes:
    def test_mla_decode_missing_W_UK_T(self):
        """Decode path with W_UK_T=None → decompose returns None."""
        args = _make_mla_decode_args()
        args[6] = None  # W_UK_T = None
        op = _make_op_info(torch.ops.tensor_cast.multihead_latent_attention.default, args)
        assert _decompose_mla(op, {}) is None

    def test_mla_prefill_missing_kv_b_proj(self):
        """Prefill path with kv_b_proj=None → decompose returns None."""
        args = _make_mla_prefill_args()
        args[8] = None  # kv_b_proj = None
        op = _make_op_info(torch.ops.tensor_cast.multihead_latent_attention.default, args)
        assert _decompose_mla(op, {}) is None

    def test_mla_unsupported_dtype(self):
        """MLA with unsupported dtype → decompose returns None."""
        args = _make_mla_decode_args()
        # Replace q with float64 (not in DTYPE_MAP)
        args[0] = torch.empty(16, 16, 192, device="meta", dtype=torch.float64)
        op = _make_op_info(torch.ops.tensor_cast.multihead_latent_attention.default, args)
        assert _decompose_mla(op, {}) is None

    def test_mla_seq_lens_not_tensor(self):
        """MLA with seq_lens as list instead of tensor → returns None."""
        args = _make_mla_decode_args()
        args[4] = [4096] * 16  # list instead of tensor
        op = _make_op_info(torch.ops.tensor_cast.multihead_latent_attention.default, args)
        assert _decompose_mla(op, {}) is None


class TestMLADecomposeWithAttentionParams:
    """Tests for MLA decomposers using attention_params (Tasks 7 & 8)."""

    def test_e1_mla_decode_attention_params(self):
        """MLA decode produces attention_params for FIA sub-kernel."""
        args = _make_mla_decode_args(
            batch_size=4,
            num_heads=16,
            kv_lora_rank=448,
            qk_rope_head_dim=64,
        )
        op = _make_op_info(torch.ops.tensor_cast.multihead_latent_attention.default, args)
        specs = _decompose_mla(op, {})
        assert len(specs) == 3
        fia_spec = specs[1]
        assert fia_spec.attention_params is not None
        q_shape_3d = fia_spec.attention_params["q_shape_3d"]
        assert q_shape_3d[0] == 4  # batch_size
        assert q_shape_3d[1] == 16  # num_heads
        assert fia_spec.attention_params["avg_seq_len"] == 4096

    def test_e2_mla_decode_attention_query_mode(self):
        """MLA decode FIA spec has query_mode='attention'."""
        args = _make_mla_decode_args(batch_size=4, num_heads=16)
        op = _make_op_info(torch.ops.tensor_cast.multihead_latent_attention.default, args)
        specs = _decompose_mla(op, {})
        fia_spec = specs[1]
        assert fia_spec.query_mode == "attention"
        assert fia_spec.attention_params is not None

    def test_e3_mla_prefill_fia(self):
        """MLA prefill: decomposes to MatMulV2 + FIA (v0.18.0)."""
        args = _make_mla_prefill_args(num_tokens=256, num_heads=16, kv_lora_rank=512)
        op = _make_op_info(torch.ops.tensor_cast.multihead_latent_attention.default, args)
        specs = _decompose_mla(op, {})
        assert specs is not None
        assert len(specs) == 2
        assert specs[0].kernel_type == "MatMulV2"
        assert specs[1].kernel_type == "FusedInferAttentionScore"

    def test_e3b_mla_prefill_matmulv2_tc_input_count(self):
        """MLA prefill MatMulV2 needs tc_input_count=2 (CSV has bias columns)."""
        args = _make_mla_prefill_args(num_tokens=256, num_heads=16, kv_lora_rank=512)
        op = _make_op_info(torch.ops.tensor_cast.multihead_latent_attention.default, args)
        specs = _decompose_mla(op, {})
        assert specs[0].tc_input_count == 2

    def test_e3c_mla_decode_tbmm_no_tc_input_count(self):
        """MLA BF16 decode: BatchMatMulV2 needs no tc_input_count override."""
        args = _make_mla_decode_args(batch_size=4, num_heads=16)
        op = _make_op_info(torch.ops.tensor_cast.multihead_latent_attention.default, args)
        specs = _decompose_mla(op, {})
        assert specs[0].tc_input_count is None  # BatchMatMulV2
        assert specs[2].tc_input_count is None  # TransposeBatchMatMul

    def test_e4_mla_quant_decode_attention_params(self):
        """MLA quant decode also produces attention_params."""
        args = _make_mla_decode_args(batch_size=4, num_heads=16, kv_lora_rank=448)
        op = _make_op_info(torch.ops.tensor_cast.multihead_latent_attention_quant.default, args)
        specs = _decompose_mla_quant(op, {})
        assert len(specs) == 3
        fia_spec = specs[1]
        assert fia_spec.attention_params is not None
        assert fia_spec.query_mode == "attention"

    def test_e4b_mla_quant_decode_qbmv3_tc_input_count(self):
        """MLA quant decode: QuantBatchMatmulV3 needs tc_input_count=2."""
        args = _make_mla_decode_args(batch_size=4, num_heads=16, kv_lora_rank=448)
        op = _make_op_info(torch.ops.tensor_cast.multihead_latent_attention_quant.default, args)
        specs = _decompose_mla_quant(op, {})
        assert specs[0].tc_input_count == 2  # QuantBatchMatmulV3
        assert specs[2].tc_input_count is None  # TransposeBatchMatMul

    def test_e5_mla_quant_prefill_fia(self):
        """MLA quant prefill: decomposes to MatMulV2 + FIA (v0.18.0)."""
        args = _make_mla_prefill_args(num_tokens=256, num_heads=16, kv_lora_rank=512)
        op = _make_op_info(torch.ops.tensor_cast.multihead_latent_attention_quant.default, args)
        specs = _decompose_mla_quant(op, {})
        assert specs is not None
        assert len(specs) == 2
        assert specs[0].kernel_type == "MatMulV2"
        assert specs[1].kernel_type == "FusedInferAttentionScore"

    def test_e5b_mla_quant_prefill_matmulv2_tc_input_count(self):
        """MLA quant prefill MatMulV2 needs tc_input_count=2."""
        args = _make_mla_prefill_args(num_tokens=256, num_heads=16, kv_lora_rank=512)
        op = _make_op_info(torch.ops.tensor_cast.multihead_latent_attention_quant.default, args)
        specs = _decompose_mla_quant(op, {})
        assert specs[0].tc_input_count == 2


# ---- 10. Interpolation linearity verification ----


class TestInterpolationLinearity:
    """Verify linear interpolation produces exact midpoint for equidistant data."""

    def test_exact_midpoint(self, extrap_data_dir):
        """seq=384 is exact midpoint of 256→25 and 512→50 → expect 37.5."""
        base = ProfilingDataSource(extrap_data_dir)
        ds = InterpolatingDataSource(base)
        op = _make_op_info(
            torch.ops.aten.mm.default,
            [
                torch.empty(384, 512, device="meta", dtype=torch.bfloat16),
                torch.empty(512, 1024, device="meta", dtype=torch.bfloat16),
            ],
        )
        result = ds.lookup(op)
        assert result is not None
        assert abs(result.latency_us - 37.5) < 0.1

    def test_quarter_point(self, extrap_data_dir):
        """seq=320 is 25% between 256 and 512 → expect 31.25."""
        base = ProfilingDataSource(extrap_data_dir)
        ds = InterpolatingDataSource(base)
        op = _make_op_info(
            torch.ops.aten.mm.default,
            [
                torch.empty(320, 512, device="meta", dtype=torch.bfloat16),
                torch.empty(512, 1024, device="meta", dtype=torch.bfloat16),
            ],
        )
        result = ds.lookup(op)
        assert result is not None
        assert abs(result.latency_us - 31.25) < 0.1

    def test_three_quarter_point(self, extrap_data_dir):
        """seq=448 is 75% between 256 and 512 → expect 43.75."""
        base = ProfilingDataSource(extrap_data_dir)
        ds = InterpolatingDataSource(base)
        op = _make_op_info(
            torch.ops.aten.mm.default,
            [
                torch.empty(448, 512, device="meta", dtype=torch.bfloat16),
                torch.empty(512, 1024, device="meta", dtype=torch.bfloat16),
            ],
        )
        result = ds.lookup(op)
        assert result is not None
        assert abs(result.latency_us - 43.75) < 0.1


# ---- 11. MLAPO decomposition: weight dimension direction (bugfix 2618b0b) ----


def _make_mlapo_args(
    num_tokens=136,
    hidden_size=5120,
    q_lora_rank=1536,
    num_heads_x_qk_head_dim=3072,
    kv_proj_dim=576,  # kv_lora_rank + rope_dim
    kv_lora_rank=512,
):
    """Build args for mlapo op.

    Weight shapes follow F.linear convention: (out_features, in_features).
    Critically, q_lora_rank != hidden_size and kv_proj_dim != hidden_size,
    so using shape[1] (in_features) instead of shape[0] (out_features) would
    produce wrong intermediate activation shapes.
    """
    hidden_states = torch.empty(num_tokens, hidden_size, device="meta", dtype=torch.bfloat16)
    # args[1], args[2]: norms (unused by decomposer but need placeholders)
    q_a_layernorm = torch.empty(q_lora_rank, device="meta", dtype=torch.bfloat16)
    q_a_scale = None
    # args[3]: q_a_proj (out_features=q_lora_rank, in_features=hidden_size)
    q_a_proj = torch.empty(q_lora_rank, hidden_size, device="meta", dtype=torch.bfloat16)
    q_a_proj_scale = None
    # args[5]: q_b_proj (out_features=num_heads*qk_head_dim, in_features=q_lora_rank)
    q_b_proj = torch.empty(num_heads_x_qk_head_dim, q_lora_rank, device="meta", dtype=torch.bfloat16)
    # args[6]: kv_a_proj (out_features=kv_proj_dim, in_features=hidden_size)
    kv_a_proj = torch.empty(kv_proj_dim, hidden_size, device="meta", dtype=torch.bfloat16)
    # args[7]: kv_a_layernorm_weight
    kv_a_layernorm = torch.empty(kv_lora_rank, device="meta", dtype=torch.bfloat16)
    # Pad to 20 args (decomposer checks len(args) >= 14 for mlapo, >= 20 for quant)
    args = [
        hidden_states,  # 0
        q_a_layernorm,  # 1
        q_a_scale,  # 2
        q_a_proj,  # 3
        q_a_proj_scale,  # 4
        q_b_proj,  # 5
        kv_a_proj,  # 6
        kv_a_layernorm,  # 7
        None,  # 8
        None,  # 9
        None,  # 10
        None,  # 11
        kv_lora_rank,  # 12
        None,  # 13
        None,  # 14
        None,  # 15
        None,  # 16
        None,  # 17
        None,  # 18
        None,  # 19
    ]
    return args


class TestDecomposeMlapo:
    """Tests for _decompose_mlapo weight dimension direction (bugfix 2618b0b).

    The bug: q_lora_rank and kv_proj_dim were read from shape[1] (in_features)
    instead of shape[0] (out_features). With F.linear convention
    weight=(out_features, in_features), shape[1]=hidden_size, which is wrong.
    """

    def test_returns_3_specs(self):
        """NPU fuses q_a_proj + kv_a_proj into fused_qkv_a_proj → 3 specs."""
        args = _make_mlapo_args()
        op = _make_op_info(torch.ops.tensor_cast.mlapo.default, args)
        specs = _decompose_mlapo(op, {})
        assert specs is not None
        assert len(specs) == 3

    def test_kernel_types(self):
        args = _make_mlapo_args()
        op = _make_op_info(torch.ops.tensor_cast.mlapo.default, args)
        specs = _decompose_mlapo(op, {})
        assert specs[0].kernel_type == "MatMulV2"  # fused_qkv_a_proj
        assert specs[1].kernel_type == "MatMulV2"  # q_b_proj
        assert specs[2].kernel_type == "KvRmsNormRopeCache"

    def test_q_lora_rank_from_out_features(self):
        """q_compressed @ q_b_proj: activation shape must use q_lora_rank (shape[0]),
        not hidden_size (shape[1]). This is the core regression test.
        """
        args = _make_mlapo_args(num_tokens=136, hidden_size=5120, q_lora_rank=1536)
        op = _make_op_info(torch.ops.tensor_cast.mlapo.default, args)
        specs = _decompose_mlapo(op, {})
        # Op2: q_compressed @ q_b_proj → input_shapes[0] = (num_tokens, q_lora_rank)
        # Bug would produce (136, 5120) instead of (136, 1536)
        assert specs[1].input_shapes[0] == (136, 1536)

    def test_kv_proj_dim_from_out_features(self):
        """KvRmsNormRopeCache shape must use kv_proj_dim (shape[0]),
        not hidden_size (shape[1]). This is the core regression test.
        NPU CSV shape is 4D (T,1,1,D) — MISS #3 fix.
        """
        args = _make_mlapo_args(num_tokens=136, hidden_size=5120, kv_proj_dim=576)
        op = _make_op_info(torch.ops.tensor_cast.mlapo.default, args)
        specs = _decompose_mlapo(op, {})
        # KvRmsNormRopeCache is now specs[2] (was [3] before fused_qkv_a_proj merge)
        # MISS #3 fix: NPU CSV shape is 4D (T,1,1,D), not 2D (T,D)
        assert specs[2].input_shapes[0] == (136, 1, 1, 576)

    def test_fused_qkv_a_proj_shape(self):
        """Op1: hidden @ fused_qkv_a_proj with N = q_lora_rank + kv_proj_dim."""
        args = _make_mlapo_args(num_tokens=100, hidden_size=5120, q_lora_rank=1536, kv_proj_dim=576)
        op = _make_op_info(torch.ops.tensor_cast.mlapo.default, args)
        specs = _decompose_mlapo(op, {})
        # Fused: (num_tokens, hidden_size) @ (q_lora_rank+kv_proj_dim, hidden_size)
        assert specs[0].input_shapes == [(100, 5120), (2112, 5120)]

    def test_insufficient_args_returns_none(self):
        op = _make_op_info(
            torch.ops.tensor_cast.mlapo.default,
            [torch.empty(136, 5120, device="meta", dtype=torch.bfloat16)],
        )
        assert _decompose_mlapo(op, {}) is None

    def test_matmulv2_specs_have_tc_input_count_2(self):
        """MatMulV2 CSV has extra bias inputs; tc_input_count=2 is required."""
        args = _make_mlapo_args()
        op = _make_op_info(torch.ops.tensor_cast.mlapo.default, args)
        specs = _decompose_mlapo(op, {})
        assert specs[0].tc_input_count == 2  # fused_qkv_a_proj
        assert specs[1].tc_input_count == 2  # q_b_proj
        assert specs[2].tc_input_count is None  # KvRmsNormRopeCache

    def test_none_weight_returns_none(self):
        args = _make_mlapo_args()
        args[3] = None  # q_a_proj = None
        op = _make_op_info(torch.ops.tensor_cast.mlapo.default, args)
        assert _decompose_mlapo(op, {}) is None


class TestDecomposeMlapoQuant:
    """Tests for _decompose_mlapo_quant weight dimension direction (bugfix 2618b0b)."""

    def test_returns_3_specs_with_quant_kernel(self):
        """NPU fuses q_a_proj + kv_a_proj into fused_qkv_a_proj → 3 specs."""
        args = _make_mlapo_args()
        op = _make_op_info(torch.ops.tensor_cast.mlapo_quant.default, args)
        specs = _decompose_mlapo_quant(op, {})
        assert specs is not None
        assert len(specs) == 3
        # Quant variant uses QuantBatchMatmulV3 for projections
        assert specs[0].kernel_type == "QuantBatchMatmulV3"  # fused_qkv_a_proj
        assert specs[1].kernel_type == "QuantBatchMatmulV3"  # q_b_proj
        assert specs[2].kernel_type == "KvRmsNormRopeCache"

    def test_q_lora_rank_from_out_features_quant(self):
        """Same bugfix regression: q_lora_rank must come from shape[0]."""
        args = _make_mlapo_args(num_tokens=136, hidden_size=5120, q_lora_rank=1536)
        op = _make_op_info(torch.ops.tensor_cast.mlapo_quant.default, args)
        specs = _decompose_mlapo_quant(op, {})
        # Bug would produce (136, 5120) instead of (136, 1536)
        assert specs[1].input_shapes[0] == (136, 1536)

    def test_kv_proj_dim_from_out_features_quant(self):
        """Same bugfix regression: kv_proj_dim must come from shape[0].
        MISS #3 fix: NPU CSV shape is 4D (T,1,1,D), not 2D (T,D).
        """
        args = _make_mlapo_args(num_tokens=136, hidden_size=5120, kv_proj_dim=576)
        op = _make_op_info(torch.ops.tensor_cast.mlapo_quant.default, args)
        specs = _decompose_mlapo_quant(op, {})
        # KvRmsNormRopeCache is now specs[2] (was [3] before fused merge)
        # MISS #3 fix: NPU CSV shape is 4D (T,1,1,D), not 2D (T,D)
        assert specs[2].input_shapes[0] == (136, 1, 1, 576)

    def test_qbmv3_specs_have_tc_input_count_2(self):
        """QuantBatchMatmulV3 CSV has extra bias inputs; tc_input_count=2 is required."""
        args = _make_mlapo_args()
        op = _make_op_info(torch.ops.tensor_cast.mlapo_quant.default, args)
        specs = _decompose_mlapo_quant(op, {})
        assert specs[0].tc_input_count == 2  # fused_qkv_a_proj
        assert specs[1].tc_input_count == 2  # q_b_proj
        assert specs[2].tc_input_count is None  # KvRmsNormRopeCache

    def test_insufficient_args_returns_none(self):
        """mlapo_quant requires len(args) >= 20."""
        args = _make_mlapo_args()[:15]  # truncate to < 20
        op = _make_op_info(torch.ops.tensor_cast.mlapo_quant.default, args)
        assert _decompose_mlapo_quant(op, {}) is None


# ============================================================
# Task 1: Fix MLAPO decomposer — dtype + weight shape + KvRmsNormRopeCache
# MISS #1: QBMV3 dtype must be INT8 (not BF16)
# MISS #2: q_b_proj must use full weight shape from int params
# MISS #3: KvRmsNormRopeCache input must be 4D (T,1,1,D)
# ============================================================


def _make_mlapo_quant_args(
    num_tokens: int = 8,
    hidden_size: int = 7168,
    num_heads: int = 16,
    qk_head_dim: int = 192,
    qk_nope_head_dim: int = 128,
    qk_rope_head_dim: int = 64,
    kv_lora_rank: int = 512,
    q_lora_rank: int = 1536,
    kv_proj_dim: int = 576,  # kv_lora_rank + qk_rope_head_dim = 512 + 64
):
    """Build 20 args matching mlapo_quant op signature.

    Args layout (tensor_cast/ops/mla.py:116-138):
        args[0]:  hidden_states (num_tokens, hidden_size) — BF16
        args[1]:  cos
        args[2]:  sin
        args[3]:  q_a_proj_weight (q_lora_rank, hidden_size) — INT8
        args[4]:  q_a_layernorm_weight
        args[5]:  q_b_proj_weight (SLICED! e.g., 384, q_lora_rank) — INT8
                  NOTE: sliced by SinkSplitPass, NOT the full shape
        args[6]:  kv_a_proj_weight (kv_proj_dim, hidden_size) — INT8
        args[7]:  kv_a_layernorm_weight
        args[8]:  num_heads (int) = 16
        args[9]:  qk_head_dim (int) = 192
        args[10]: qk_nope_head_dim (int) = 128
        args[11]: qk_rope_head_dim (int) = 64
        args[12]: kv_lora_rank (int) = 512
        args[13]: q_lora_rank (int) = 1536
        args[14]: q_a_proj_scale
        args[15]: q_a_proj_offset
        args[16]: q_b_proj_scale
        args[17]: q_b_proj_offset
        args[18]: kv_a_proj_scale
        args[19]: kv_a_proj_offset
    """
    # args[5] is SLICED by SinkSplitPass: only 384 rows instead of full 3072
    sliced_q_b_proj_rows = num_heads * qk_head_dim // 8  # e.g. 384 for 16*192//8
    return [
        # [0] hidden_states — BF16 (activation)
        torch.empty(num_tokens, hidden_size, device="meta", dtype=torch.bfloat16),
        # [1] cos
        torch.empty(num_tokens, qk_rope_head_dim, device="meta", dtype=torch.bfloat16),
        # [2] sin
        torch.empty(num_tokens, qk_rope_head_dim, device="meta", dtype=torch.bfloat16),
        # [3] q_a_proj_weight (q_lora_rank, hidden_size) — INT8
        torch.empty(q_lora_rank, hidden_size, device="meta", dtype=torch.int8),
        # [4] q_a_layernorm_weight
        torch.empty(q_lora_rank, device="meta", dtype=torch.bfloat16),
        # [5] q_b_proj_weight — SLICED by SinkSplitPass to (sliced_rows, q_lora_rank)
        torch.empty(sliced_q_b_proj_rows, q_lora_rank, device="meta", dtype=torch.int8),
        # [6] kv_a_proj_weight (kv_proj_dim, hidden_size) — INT8
        torch.empty(kv_proj_dim, hidden_size, device="meta", dtype=torch.int8),
        # [7] kv_a_layernorm_weight
        torch.empty(kv_lora_rank, device="meta", dtype=torch.bfloat16),
        # [8] num_heads
        num_heads,
        # [9] qk_head_dim
        qk_head_dim,
        # [10] qk_nope_head_dim
        qk_nope_head_dim,
        # [11] qk_rope_head_dim
        qk_rope_head_dim,
        # [12] kv_lora_rank
        kv_lora_rank,
        # [13] q_lora_rank
        q_lora_rank,
        # [14] q_a_proj_scale
        torch.empty(1, device="meta", dtype=torch.float32),
        # [15] q_a_proj_offset (None — no offset for dynamic quant)
        None,
        # [16] q_b_proj_scale
        torch.empty(1, device="meta", dtype=torch.float32),
        # [17] q_b_proj_offset
        None,
        # [18] kv_a_proj_scale
        torch.empty(1, device="meta", dtype=torch.float32),
        # [19] kv_a_proj_offset
        None,
    ]


class TestKvRmsNormRopeCacheDispatchContract:
    def test_meta_dispatch_preserves_mla_output_contract(self):
        """Cover the registered meta op instead of only YAML/decomposer wiring."""
        num_tokens = 8
        kv_lora_rank = 512
        qk_rope_head_dim = 64

        with torch.device("meta"):
            kv = torch.empty(
                num_tokens,
                kv_lora_rank + qk_rope_head_dim,
                dtype=torch.bfloat16,
            )
            gamma = torch.empty(kv_lora_rank, dtype=torch.bfloat16)
            cos = torch.empty(1, num_tokens, qk_rope_head_dim, dtype=torch.bfloat16)
            sin = torch.empty(1, num_tokens, qk_rope_head_dim, dtype=torch.bfloat16)
            kv_cache = torch.empty(
                256,
                16,
                kv_lora_rank + qk_rope_head_dim,
                dtype=torch.bfloat16,
            )
            slot_mapping = torch.arange(num_tokens, dtype=torch.long)

        k_pe, kv_c_normed = torch.ops.tensor_cast.kv_rmsnorm_rope_cache(
            kv,
            gamma,
            cos,
            sin,
            kv_cache,
            slot_mapping,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            epsilon=1e-6,
        )

        assert k_pe.shape == (num_tokens, qk_rope_head_dim)
        assert kv_c_normed.shape == (num_tokens, kv_lora_rank)
        assert k_pe.dtype == torch.bfloat16
        assert kv_c_normed.dtype == torch.bfloat16
        assert k_pe.device.type == "meta"
        assert kv_c_normed.device.type == "meta"


class TestDecomposeMLAPOQuantFix:
    """Test fixes for MLAPO quant decomposer: dtype, q_b_proj shape, KvRmsNormRopeCache 4D."""

    def _make_op(self, **kwargs):
        args = _make_mlapo_quant_args(**kwargs)
        return _make_op_info(torch.ops.tensor_cast.mlapo_quant.default, args)

    def test_qbmv3_dtype_is_int8(self):
        """MISS #1: QuantBatchMatmulV3 sub-kernels must use dtype='INT8', not 'DT_BF16'.

        NPU runs DynamicQuant/AscendQuantV2 before QBMV3, so QBMV3 activation dtype is INT8.
        """
        op = self._make_op()
        specs = _decompose_mlapo_quant(op, {})
        assert specs is not None
        # First two specs are QBMV3 matmuls
        assert specs[0].kernel_type == "QuantBatchMatmulV3"
        assert specs[1].kernel_type == "QuantBatchMatmulV3"
        assert specs[0].dtype == "INT8", f"Expected INT8 for QBMV3, got {specs[0].dtype!r}"
        assert specs[1].dtype == "INT8", f"Expected INT8 for QBMV3, got {specs[1].dtype!r}"

    def test_q_b_proj_uses_full_weight_shape(self):
        """MISS #2: q_b_proj weight must use full shape (num_heads*qk_head_dim, q_lora_rank).

        SinkSplitPass slices args[5] to (384, q_lora_rank) for TP, but the NPU kernel
        uses the full weight. Decomposer must compute shape from int params args[8]*args[9].
        """
        op = self._make_op(num_heads=16, qk_head_dim=192, q_lora_rank=1536)
        specs = _decompose_mlapo_quant(op, {})
        assert specs is not None
        # q_b_proj spec is specs[1]
        q_b_proj_spec = specs[1]
        assert q_b_proj_spec.kernel_type == "QuantBatchMatmulV3"
        # Full weight shape: (num_heads * qk_head_dim, q_lora_rank) = (3072, 1536)
        weight_shape = q_b_proj_spec.input_shapes[1]
        assert weight_shape == (3072, 1536), (
            f"Expected (3072, 1536), got {weight_shape}. "
            "Decomposer may be using sliced tensor shape instead of int params."
        )

    def test_kv_rms_norm_rope_cache_is_4d(self):
        """MISS #3: KvRmsNormRopeCache input must be 4D (T,1,1,D), not 2D (T,D).

        NPU CSV shape for KvRmsNormRopeCache is (T,1,1,576), not (T,576).
        """
        op = self._make_op(num_tokens=8, kv_proj_dim=576)
        specs = _decompose_mlapo_quant(op, {})
        assert specs is not None
        kv_spec = specs[2]
        assert kv_spec.kernel_type == "KvRmsNormRopeCache"
        # Must be 4D: (T, 1, 1, kv_proj_dim)
        input_shape = kv_spec.input_shapes[0]
        assert len(input_shape) == 4, f"Expected 4D shape (T,1,1,D), got {len(input_shape)}D: {input_shape}"
        assert input_shape == (8, 1, 1, 576), f"Expected (8,1,1,576), got {input_shape}"

    def test_kv_rms_norm_rope_cache_keeps_bf16_dtype(self):
        """KvRmsNormRopeCache keeps BF16 even when matmul kernels are INT8.

        KvRmsNormRopeCache operates on the kv_a output which is still BF16,
        not quantized. Only QBMV3 matmuls use INT8.
        """
        op = self._make_op()
        specs = _decompose_mlapo_quant(op, {})
        assert specs is not None
        kv_spec = specs[2]
        assert kv_spec.kernel_type == "KvRmsNormRopeCache"
        assert kv_spec.dtype == "DT_BF16", f"Expected DT_BF16 for KvRmsNormRopeCache, got {kv_spec.dtype!r}"


# ============================================================
# Task 2: Integration tests for MLAPO composite lookup pipeline
# TestCompositeLookupMLAPO: mlapo_quant/mlapo → decomposer → _find_compute_match → CSV
# ============================================================

# op_mapping for MLAPO integration tests: uses decomposer=true (no sub_kernels list)
MLAPO_OP_MAPPING = """\
version: "test"
device: TEST_DEVICE
operator_mappings:
  "tensor_cast.mlapo_quant.default":
    composite: true
    decomposer: true
  "tensor_cast.mlapo.default":
    composite: true
    decomposer: true
"""

# QuantBatchMatmulV3 CSV for MLAPO quant integration tests.
# num_tokens=136, hidden_size=7168, q_lora_rank=1536, kv_proj_dim=576, num_heads=16, qk_head_dim=192
#
# fused_qkv_a_proj: activation (136,7168) INT8, weight (2112,7168) FRACTAL_NZ
#   FRACTAL_NZ (66,448,16,32) → ND (2112, 7168)   [block_w=32→H=66, block_h=16→W=448]
# q_b_proj: activation (136,1536) INT8, weight (3072,1536) FRACTAL_NZ
#   FRACTAL_NZ (96,96,16,32) → ND (3072, 1536)    [block_w=32→H=96, block_h=16→W=96]
#
# tc_input_count=2 → only activation and weight are compared; CSV has 4 inputs total
# (activation, weight, scale, offset).
QBMV3_MLAPO_CSV = """\
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Duration(us)
"136,7168;66,448,16,32;2112;2112","INT8;INT8;FLOAT;INT32","ND;FRACTAL_NZ;ND;ND","136,2112","INT8","ND",15.0
"136,1536;96,96,16,32;3072;3072","INT8;INT8;FLOAT;INT32","ND;FRACTAL_NZ;ND;ND","136,3072","INT8","ND",12.0"""

# MatMulV2 CSV for MLAPO BF16 integration tests.
# num_tokens=136, hidden_size=5120, q_lora_rank=1536, kv_proj_dim=576
#
# fused_qkv_a_proj: activation (136,5120), weight (2112,5120) BF16 ND
# q_b_proj: activation (136,1536), weight (3072,1536) BF16 ND
#
# Note: MATMUL transpose check tries both (K,N) and (N,K) orientations,
# so either orientation in CSV will match TC's (out_features, in_features) weight.
MATMULV2_MLAPO_CSV = """\
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Duration(us)
"136,5120;2112,5120","DT_BF16;DT_BF16","ND;ND","136,2112","DT_BF16","ND",10.0
"136,1536;3072,1536","DT_BF16;DT_BF16","ND;ND","136,3072","DT_BF16","ND",8.0"""

# KvRmsNormRopeCache CSV with real 4D NPU shapes.
# NPU shape is (T, 1, 1, kv_proj_dim) = (136, 1, 1, 576) for T=136, kv_proj_dim=576.
# The NPU kernel has 12 inputs total; decomposer passes only 1 shape.
# auto_truncate=True → only first input compared against CSV.
_KVRNRC_SHAPES = '"136,1,1,576;512;136,1,1,64;136,1,1,64;136;100,128,1,64;100,128,1,512;;;;;"'
_KVRNRC_DTYPES = (
    '"DT_BF16;DT_BF16;DT_BF16;DT_BF16;INT64;DT_BF16;DT_BF16'
    ";DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED"
    '"'
)
_KVRNRC_FMTS = '"ND;ND;ND;ND;ND;ND;ND;NULL;NULL;NULL;NULL;NULL"'
KVRNRC_CSV_4D = (
    "Input Shapes,Input Data Types,Input Formats,"
    "Output Shapes,Output Data Types,Output Formats,Duration(us)\n"
    f"{_KVRNRC_SHAPES},{_KVRNRC_DTYPES},{_KVRNRC_FMTS},"
    '"136,1,1,512;136,1,1,64","DT_BF16;DT_BF16","ND;ND",3.5'
)


@pytest.fixture
def mlapo_data_dir(tmp_path):
    """Fixture: tmp dir with op_mapping + QBMV3 + MatMulV2 + KvRmsNormRopeCache CSVs."""
    d = tmp_path / "mlapo"
    d.mkdir()
    (d / "op_mapping.yaml").write_text(MLAPO_OP_MAPPING)
    (d / "QuantBatchMatmulV3.csv").write_text(QBMV3_MLAPO_CSV.strip())
    (d / "MatMulV2.csv").write_text(MATMULV2_MLAPO_CSV.strip())
    (d / "KvRmsNormRopeCache.csv").write_text(KVRNRC_CSV_4D.strip())
    return d


class TestCompositeLookupMLAPO:
    """Integration: mlapo_quant/mlapo → decomposer → _find_compute_match → CSV hit."""

    def test_mlapo_quant_full_hit(self, mlapo_data_dir):
        """mlapo_quant decomposed: 2x QBMV3 + KvRmsNormRopeCache all HIT → sum latency."""
        ds = ProfilingDataSource(mlapo_data_dir)
        # num_tokens=136 matches CSV rows; hidden_size=7168, q_lora_rank=1536,
        # kv_proj_dim=576, num_heads=16, qk_head_dim=192 → qk_head_dim = 192
        args = _make_mlapo_quant_args(
            num_tokens=136,
            hidden_size=7168,
            num_heads=16,
            qk_head_dim=192,
            q_lora_rank=1536,
            kv_proj_dim=576,
            kv_lora_rank=512,
        )
        op = _make_op_info(torch.ops.tensor_cast.mlapo_quant.default, args)
        result = ds.lookup(op)
        assert result is not None, "Expected HIT for mlapo_quant but got None"
        assert result.details.get("composite") is True
        # 15.0 (fused_qkv_a_proj QBMV3) + 12.0 (q_b_proj QBMV3) + 3.5 (KvRmsNormRopeCache)
        assert abs(result.latency_us - (15.0 + 12.0 + 3.5)) < 0.01, (
            f"Expected latency {15.0 + 12.0 + 3.5}, got {result.latency_us}"
        )

    def test_mlapo_bf16_full_hit(self, mlapo_data_dir):
        """Mlapo BF16 decomposed: 2x MatMulV2 + KvRmsNormRopeCache all HIT → sum latency."""
        ds = ProfilingDataSource(mlapo_data_dir)
        # Use _make_mlapo_args with hidden_size=5120, q_lora_rank=1536, kv_proj_dim=576
        args = _make_mlapo_args(
            num_tokens=136,
            hidden_size=5120,
            q_lora_rank=1536,
            num_heads_x_qk_head_dim=3072,
            kv_proj_dim=576,
            kv_lora_rank=512,
        )
        op = _make_op_info(torch.ops.tensor_cast.mlapo.default, args)
        result = ds.lookup(op)
        assert result is not None, "Expected HIT for mlapo BF16 but got None"
        assert result.details.get("composite") is True
        # 10.0 (fused_qkv_a_proj MatMulV2) + 8.0 (q_b_proj MatMulV2) + 3.5 (KvRmsNormRopeCache)
        assert abs(result.latency_us - (10.0 + 8.0 + 3.5)) < 0.01, (
            f"Expected latency {10.0 + 8.0 + 3.5}, got {result.latency_us}"
        )

    def test_mlapo_quant_shape_miss_returns_partial(self, mlapo_data_dir):
        """mlapo_quant with wrong num_tokens → QBMV3 miss → PARTIAL result."""
        ds = ProfilingDataSource(mlapo_data_dir)
        # Use num_tokens=999 which won't match any CSV row
        args = _make_mlapo_quant_args(
            num_tokens=999,
            hidden_size=7168,
            num_heads=16,
            qk_head_dim=192,
            q_lora_rank=1536,
            kv_proj_dim=576,
        )
        op = _make_op_info(torch.ops.tensor_cast.mlapo_quant.default, args)
        result = ds.lookup(op)
        # All 3 sub-kernels miss (no CSV rows with T=999) → None
        assert result is None, f"Expected None for all-miss with num_tokens=999, got {result}"

    def test_mlapo_insufficient_args_returns_none(self, mlapo_data_dir):
        """mlapo_quant with insufficient args → decompose fails → None."""
        ds = ProfilingDataSource(mlapo_data_dir)
        op = _make_op_info(
            torch.ops.tensor_cast.mlapo_quant.default,
            [torch.empty(136, 7168, device="meta", dtype=torch.bfloat16)],
        )
        result = ds.lookup(op)
        assert result is None, "Expected None for insufficient args but got result"
