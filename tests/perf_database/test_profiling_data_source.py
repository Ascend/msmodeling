from unittest.mock import MagicMock

import pytest
import torch

from tensor_cast.device import CommGrid, InterconnectTopology
from tensor_cast.model_config import ParallelConfig
from tensor_cast.performance_model.profiling_database.data_source import QuerySource

from tensor_cast.performance_model.profiling_database.profiling_data_source import (
    _dtype_byte_size,
    _is_block_padded,
    COMPOSITE_DECOMPOSERS,
    DTYPE_MAP,
    fractal_nz_to_nd,
    get_topology_tier,
    ProfilingDataSource,
    SubKernelSpec,
)


def _make_parallel_config(ep_size=1, world_size=16, tp_size=8):
    return ParallelConfig(
        world_size=world_size,
        tensor_parallel_size=tp_size,
        expert_parallel_size=ep_size,
    )


# --- fractal_nz_to_nd tests ---


def test_fractal_nz_to_nd_bf16():
    # BF16 MatMulV2: [320,48,16,16] -> K=320*16=5120, N=48*16=768
    assert fractal_nz_to_nd((320, 48, 16, 16)) == (5120, 768)


def test_fractal_nz_to_nd_int8():
    # INT8 QuantBatchMatmulV3: [N/32, K/16, 16, 32]
    # H=48, W=448, block_h=16, block_w=32 -> (H*block_w, W*block_h) = (1536, 7168)
    assert fractal_nz_to_nd((48, 448, 16, 32)) == (1536, 7168)


def test_fractal_nz_to_nd_batched():
    # GroupedMatmul INT8: [E, N/32, K/16, 16, 32]
    # batch=64, H=48, W=448, block_h=16, block_w=32 -> (64, 1536, 7168)
    assert fractal_nz_to_nd((64, 48, 448, 16, 32)) == (64, 1536, 7168)


def test_dtype_map():
    assert DTYPE_MAP[torch.bfloat16] == "DT_BF16"
    assert DTYPE_MAP[torch.float16] == "DT_BF16"
    assert DTYPE_MAP[torch.int8] == "INT8"
    assert DTYPE_MAP[torch.float32] == "FLOAT"


def test_dtype_byte_size():
    assert _dtype_byte_size("DT_BF16") == 2
    assert _dtype_byte_size("FLOAT") == 4
    assert _dtype_byte_size("INT8") == 1
    assert _dtype_byte_size("INT32") == 4
    assert _dtype_byte_size("INT64") == 8
    assert _dtype_byte_size("UNKNOWN") == 0


# --- ProfilingDataSource tests ---

SPIKE_OP_MAPPING_YAML = """
version: "0.14.0"
device: TEST_DEVICE

operator_mappings:
  "aten.mm.default":
    kernel_type: MatMulV2
  "aten.bmm.default":
    kernel_type: TransposeBatchMatMul
  "tensor_cast.attention.default":
    kernel_type: FusedInferAttentionScore
    query_mode: attention_special
  "tensor_cast.multihead_latent_attention.default":
    composite: true
    sub_kernels: [TransposeBatchMatMul, FusedInferAttentionScore]
  "tensor_cast.all_reduce.default":
    kernel_type: hcom_allReduce_
    category: communication
"""

# CSV with FRACTAL_NZ weight
SPIKE_MATMUL_CSV = """\
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Average Duration(us)
"136,5120;320,48,16,16","DT_BF16;DT_BF16","ND;FRACTAL_NZ","136,768","DT_BF16","ND",45.3
"1,5120;320,48,16,16","DT_BF16;DT_BF16","ND;FRACTAL_NZ","1,768","DT_BF16","ND",12.1
"""


@pytest.fixture
def sample_data_dir(tmp_path):
    data_dir = tmp_path / "spike"
    data_dir.mkdir()
    (data_dir / "op_mapping.yaml").write_text(SPIKE_OP_MAPPING_YAML)
    (data_dir / "MatMulV2.csv").write_text(SPIKE_MATMUL_CSV.strip())
    return data_dir


def _make_op_info(func, input_tensors, output_tensors=None):
    """Create a mock OpInvokeInfo with real torch.ops func and meta tensors."""
    mock = MagicMock()
    mock.func = func
    mock.args = tuple(input_tensors)
    mock.kwargs = {}
    if output_tensors:
        mock.out = (
            output_tensors[0] if len(output_tensors) == 1 else tuple(output_tensors)
        )
    else:
        mock.out = None
    return mock


class _FakeTorchOp:
    def __init__(self, qualname: str):
        self.qualname = qualname

    def __str__(self) -> str:
        return f"torch.ops.{self.qualname}"


def test_exact_match_with_fractal_nz(sample_data_dir):
    """aten.mm(A[136,5120], B[5120,768]) matches
    CSV row with FRACTAL_NZ weight [320,48,16,16] after restoration."""
    ds = ProfilingDataSource(sample_data_dir)
    op = _make_op_info(
        torch.ops.aten.mm.default,
        [
            torch.empty(136, 5120, device="meta", dtype=torch.bfloat16),
            torch.empty(5120, 768, device="meta", dtype=torch.bfloat16),
        ],
    )
    result = ds.lookup(op)
    assert result is not None
    assert abs(result.latency_us - 45.3) < 0.01
    assert result.confidence == 1.0
    assert result.source == QuerySource.MEASURED
    assert result.details.get("kernel_type") == "MatMulV2"
    assert result.shape_match_info is not None
    assert result.shape_match_info.shape_match_rule == "identity"


def test_miss_wrong_shape(sample_data_dir):
    ds = ProfilingDataSource(sample_data_dir)
    op = _make_op_info(
        torch.ops.aten.mm.default,
        [
            torch.empty(256, 5120, device="meta", dtype=torch.bfloat16),
            torch.empty(5120, 768, device="meta", dtype=torch.bfloat16),
        ],
    )
    result = ds.lookup(op)
    assert result is None
    assert ds.last_shape_match_info is not None
    assert ds.last_shape_match_info.simulation_shapes == [[256, 5120], [5120, 768]]
    assert ds.last_shape_match_info.kernel_shapes == []
    assert ds.last_shape_match_info.shape_match_rule == "shape_mismatch"


def test_miss_unmapped_op(sample_data_dir):
    ds = ProfilingDataSource(sample_data_dir)
    op = _make_op_info(
        torch.ops.aten.add.Tensor,
        [
            torch.empty(136, 5120, device="meta", dtype=torch.bfloat16),
            torch.empty(136, 5120, device="meta", dtype=torch.bfloat16),
        ],
    )
    result = ds.lookup(op)
    assert result is None
    assert ds.last_miss_reason == "unmapped"
    assert ds.last_shape_match_info is not None
    assert ds.last_shape_match_info.simulation_shapes == [[136, 5120], [136, 5120]]
    assert ds.last_shape_match_info.kernel_shapes == []
    assert ds.last_shape_match_info.shape_match_rule == "unmapped"


def test_compute_csv_not_found_records_shape_debug(tmp_path):
    data_dir = tmp_path / "csv_not_found"
    data_dir.mkdir()
    (data_dir / "op_mapping.yaml").write_text(
        'version: "test"\n'
        "device: TEST_DEVICE\n\n"
        "operator_mappings:\n"
        '  "aten.mm.default":\n'
        "    kernel_type: MissingKernel\n"
    )
    ds = ProfilingDataSource(data_dir)
    op = _make_op_info(
        torch.ops.aten.mm.default,
        [
            torch.empty(136, 5120, device="meta", dtype=torch.bfloat16),
            torch.empty(5120, 768, device="meta", dtype=torch.bfloat16),
        ],
    )

    result = ds.lookup(op)

    assert result is None
    assert ds.last_miss_reason == "csv_not_found"
    assert ds.last_shape_match_info is not None
    assert ds.last_shape_match_info.simulation_shapes == [[136, 5120], [5120, 768]]
    assert ds.last_shape_match_info.kernel_shapes == []
    assert ds.last_shape_match_info.shape_match_rule == "csv_not_found"


def test_compute_transpose_hit_attaches_shape_debug(tmp_path):
    data_dir = tmp_path / "transpose_hit"
    data_dir.mkdir()
    (data_dir / "op_mapping.yaml").write_text(
        'version: "test"\n'
        "device: TEST_DEVICE\n\n"
        "operator_mappings:\n"
        '  "aten.mm.default":\n'
        "    kernel_type: MatMulV2\n"
    )
    (data_dir / "MatMulV2.csv").write_text(
        "Input Shapes,Input Data Types,Input Formats,Output Shapes,"
        "Output Data Types,Output Formats,Average Duration(us)\n"
        '"136,5120;768,5120","DT_BF16;DT_BF16","ND;ND","136,768","DT_BF16","ND",47.4\n'
    )
    ds = ProfilingDataSource(data_dir)
    op = _make_op_info(
        torch.ops.aten.mm.default,
        [
            torch.empty(136, 5120, device="meta", dtype=torch.bfloat16),
            torch.empty(5120, 768, device="meta", dtype=torch.bfloat16),
        ],
    )

    result = ds.lookup(op)

    assert result is not None
    assert abs(result.latency_us - 47.4) < 0.01
    assert result.shape_match_info is not None
    assert result.shape_match_info.simulation_shapes == [[136, 5120], [5120, 768]]
    assert result.shape_match_info.kernel_shapes == [[136, 5120], [768, 5120]]
    assert result.shape_match_info.shape_match_rule == "transpose"
    assert ds.last_shape_match_info == result.shape_match_info


def test_composite_returns_none(sample_data_dir):
    """Composite ops (MLA) return None in spike, fallback to analytic."""
    ds = ProfilingDataSource(sample_data_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.multihead_latent_attention.default,
        [torch.empty(136, 5120, device="meta", dtype=torch.bfloat16)],
    )
    result = ds.lookup(op)
    assert result is None


def test_communication_returns_none(sample_data_dir):
    """Communication ops return None in spike, fallback to analytic."""
    ds = ProfilingDataSource(sample_data_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.all_reduce.default,
        [torch.empty(136, 5120, device="meta", dtype=torch.bfloat16), 0, [0, 1]],
    )
    result = ds.lookup(op)
    assert result is None


# --- Weight transpose matching tests ---

LMHEAD_CSV = """\
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Average Duration(us)
"1,5120;9496,5120","DT_BF16;DT_BF16","ND;ND","1,9496","DT_BF16","ND",91.753
"""


@pytest.fixture
def lmhead_data_dir(tmp_path):
    data_dir = tmp_path / "lmhead"
    data_dir.mkdir()
    op_mapping = (
        'version: "test"\n'
        "operator_mappings:\n"
        '  "aten.mm.default":\n'
        "    kernel_type: MatMulV2\n"
    )
    (data_dir / "op_mapping.yaml").write_text(op_mapping)
    (data_dir / "MatMulV2.csv").write_text(LMHEAD_CSV.strip())
    return data_dir


def test_nd_weight_transpose_match(lmhead_data_dir):
    """ND-format matmul weight stored as (N,K) should match TC's (K,N).
    CSV has weight (9496,5120) = (N,K). TC mm receives (5120,9496) = (K,N)
    because F.linear transposes before dispatch."""
    ds = ProfilingDataSource(lmhead_data_dir)
    op = _make_op_info(
        torch.ops.aten.mm.default,
        [
            torch.empty(1, 5120, device="meta", dtype=torch.bfloat16),
            torch.empty(5120, 9496, device="meta", dtype=torch.bfloat16),
        ],
    )
    result = ds.lookup(op)
    assert result is not None, "Should match with ND weight transpose"
    assert abs(result.latency_us - 91.753) < 0.01


def test_nd_weight_no_false_positive(lmhead_data_dir):
    """Non-transpose shape mismatches should NOT match."""
    ds = ProfilingDataSource(lmhead_data_dir)
    op = _make_op_info(
        torch.ops.aten.mm.default,
        [
            torch.empty(1, 5120, device="meta", dtype=torch.bfloat16),
            torch.empty(5120, 1234, device="meta", dtype=torch.bfloat16),
        ],
    )
    result = ds.lookup(op)
    assert result is None, "Completely different N should not match"


# --- Block-padding tolerance tests ---

ADD_CSV = """\
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Average Duration(us)
"136,5120;136,5120","DT_BF16;DT_BF16","ND;ND","136,5120","DT_BF16","ND",16.238
"""


@pytest.fixture
def add_data_dir(tmp_path):
    data_dir = tmp_path / "add"
    data_dir.mkdir()
    op_mapping = (
        'version: "test"\n'
        "operator_mappings:\n"
        '  "aten.add.Tensor":\n'
        "    kernel_type: Add\n"
    )
    (data_dir / "op_mapping.yaml").write_text(op_mapping)
    (data_dir / "Add.csv").write_text(ADD_CSV.strip())
    return data_dir


def test_block_padding_tolerance(add_data_dir):
    """TC seq=144 (padded from 136 via ceil(136/16)*16) should match CSV seq=136."""
    ds = ProfilingDataSource(add_data_dir)
    op = _make_op_info(
        torch.ops.aten.add.Tensor,
        [
            torch.empty(144, 5120, device="meta", dtype=torch.bfloat16),
            torch.empty(144, 5120, device="meta", dtype=torch.bfloat16),
        ],
    )
    result = ds.lookup(op)
    assert result is not None, "Should match with block-padding tolerance (144 ≈ 136)"
    assert abs(result.latency_us - 16.238) < 0.01


def test_block_padding_no_false_positive(add_data_dir):
    """Shapes that aren't block-padding should NOT match."""
    ds = ProfilingDataSource(add_data_dir)
    op = _make_op_info(
        torch.ops.aten.add.Tensor,
        [
            torch.empty(256, 5120, device="meta", dtype=torch.bfloat16),
            torch.empty(256, 5120, device="meta", dtype=torch.bfloat16),
        ],
    )
    result = ds.lookup(op)
    assert result is None, "256 is not a block-padding of 136"


def test_block_padding_32_alignment(add_data_dir):
    """INT8 uses 32-alignment: ceil(136/32)*32=160 should also match."""
    ds = ProfilingDataSource(add_data_dir)
    op = _make_op_info(
        torch.ops.aten.add.Tensor,
        [
            torch.empty(160, 5120, device="meta", dtype=torch.bfloat16),
            torch.empty(160, 5120, device="meta", dtype=torch.bfloat16),
        ],
    )
    result = ds.lookup(op)
    assert result is not None, "Should match with 32-alignment padding (160 ≈ 136)"


# --- Batch-dim stripping tests ---

RMSNORM_CSV = """\
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Average Duration(us)
"136,5120;5120","DT_BF16;DT_BF16","ND;ND","136,5120;136,1","DT_BF16;FLOAT","ND;ND",21.660000
"""

ADD_RMSNORM_CSV = """\
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Average Duration(us)
"136,5120;136,5120;5120","DT_BF16;DT_BF16;DT_BF16","ND;ND;ND","136,5120;136,1;136,5120","DT_BF16;FLOAT;DT_BF16","ND;ND;ND",33.150000
"""


@pytest.fixture
def rmsnorm_data_dir(tmp_path):
    data_dir = tmp_path / "rmsnorm"
    data_dir.mkdir()
    op_mapping = (
        'version: "test"\n'
        "operator_mappings:\n"
        '  "tensor_cast.rms_norm.default":\n'
        "    kernel_type: RmsNorm\n"
    )
    (data_dir / "op_mapping.yaml").write_text(op_mapping)
    (data_dir / "RmsNorm.csv").write_text(RMSNORM_CSV.strip())
    return data_dir


@pytest.fixture
def add_rmsnorm_data_dir(tmp_path):
    data_dir = tmp_path / "add_rmsnorm"
    data_dir.mkdir()
    op_mapping = (
        'version: "test"\n'
        "operator_mappings:\n"
        '  "tensor_cast.add_rms_norm2.default":\n'
        "    kernel_type: AddRmsNorm\n"
    )
    (data_dir / "op_mapping.yaml").write_text(op_mapping)
    (data_dir / "AddRmsNorm.csv").write_text(ADD_RMSNORM_CSV.strip())
    return data_dir


def test_batch_dim_stripping_rmsnorm(rmsnorm_data_dir):
    """TC RmsNorm sends (1,144,5120),(5120,) — match CSV (136,5120),(5120) after batch strip + padding."""
    ds = ProfilingDataSource(rmsnorm_data_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.rms_norm.default,
        [
            torch.empty(1, 144, 5120, device="meta", dtype=torch.bfloat16),
            torch.empty(5120, device="meta", dtype=torch.bfloat16),
        ],
    )
    result = ds.lookup(op)
    assert result is not None, (
        "Should match after stripping batch dim=1 + padding tolerance"
    )
    assert abs(result.latency_us - 21.66) < 0.01
    assert result.shape_match_info is not None
    assert result.shape_match_info.shape_match_rule == "padding"


def test_batch_dim_stripping_add(add_data_dir):
    """TC Add sends (1,144,5120),(1,144,5120) — match CSV (136,5120),(136,5120)."""
    ds = ProfilingDataSource(add_data_dir)
    op = _make_op_info(
        torch.ops.aten.add.Tensor,
        [
            torch.empty(1, 144, 5120, device="meta", dtype=torch.bfloat16),
            torch.empty(1, 144, 5120, device="meta", dtype=torch.bfloat16),
        ],
    )
    result = ds.lookup(op)
    assert result is not None, "Should match after stripping batch dim=1 + padding"


def test_batch_dim_stripping_add_rmsnorm(add_rmsnorm_data_dir):
    """TC AddRmsNorm sends (1,144,5120),(144,5120),(5120,) — match CSV (136,5120),(136,5120),(5120)."""
    ds = ProfilingDataSource(add_rmsnorm_data_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.add_rms_norm2.default,
        [
            torch.empty(1, 144, 5120, device="meta", dtype=torch.bfloat16),
            torch.empty(144, 5120, device="meta", dtype=torch.bfloat16),
            torch.empty(5120, device="meta", dtype=torch.bfloat16),
        ],
    )
    result = ds.lookup(op)
    assert result is not None, "Should match after stripping batch dim + padding"


def test_batch_dim_no_false_positive(add_data_dir):
    """Batch dim > 1 should NOT be stripped."""
    ds = ProfilingDataSource(add_data_dir)
    op = _make_op_info(
        torch.ops.aten.add.Tensor,
        [
            torch.empty(2, 144, 5120, device="meta", dtype=torch.bfloat16),
            torch.empty(2, 144, 5120, device="meta", dtype=torch.bfloat16),
        ],
    )
    result = ds.lookup(op)
    assert result is None, "Batch dim > 1 should not match"


# --- SwiGlu input concatenation tests ---

SWIGLU_CSV = """\
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Average Duration(us)
"136,3200","DT_BF16","ND","136,1600","DT_BF16","ND",14.871969
"""


@pytest.fixture
def swiglu_data_dir(tmp_path):
    data_dir = tmp_path / "swiglu"
    data_dir.mkdir()
    op_mapping = (
        'version: "test"\n'
        "operator_mappings:\n"
        '  "tensor_cast.swiglu.default":\n'
        "    kernel_type: SwiGlu\n"
    )
    (data_dir / "op_mapping.yaml").write_text(op_mapping)
    (data_dir / "SwiGlu.csv").write_text(SWIGLU_CSV.strip())
    return data_dir


def test_swiglu_input_concat(swiglu_data_dir):
    """TC SwiGlu sends 2 inputs (1,144,1600),(1,144,1600) -> CSV has 1 input (136,3200)."""
    ds = ProfilingDataSource(swiglu_data_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.swiglu.default,
        [
            torch.empty(1, 144, 1600, device="meta", dtype=torch.bfloat16),
            torch.empty(1, 144, 1600, device="meta", dtype=torch.bfloat16),
        ],
    )
    result = ds.lookup(op)
    assert result is not None, "Should match SwiGlu after concatenating 2 inputs into 1"
    assert abs(result.latency_us - 14.871969) < 0.01


def test_swiglu_no_false_positive(swiglu_data_dir):
    """Wrong shape should not match."""
    ds = ProfilingDataSource(swiglu_data_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.swiglu.default,
        [
            torch.empty(1, 256, 1600, device="meta", dtype=torch.bfloat16),
            torch.empty(1, 256, 1600, device="meta", dtype=torch.bfloat16),
        ],
    )
    result = ds.lookup(op)
    assert result is None, "Wrong seq dim should not match"


def test_attention_special_returns_none(sample_data_dir):
    """attention_special ops return None in spike, fallback to analytic."""
    ds = ProfilingDataSource(sample_data_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.attention.default,
        [torch.empty(136, 5120, device="meta", dtype=torch.bfloat16)],
    )
    result = ds.lookup(op)
    assert result is None


# --- RoPE shape normalization tests ---

ROPE_CSV = """\
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Average Duration(us)
"1,136,4,128;1,136,1,128;1,136,1,128;1,136,1,128","DT_BF16;DT_BF16;DT_BF16;DT_BF16","ND;ND;ND;ND","1,136,4,128;1,136,1,128","DT_BF16;DT_BF16","ND;ND",12.500000
"""


@pytest.fixture
def rope_data_dir(tmp_path):
    data_dir = tmp_path / "rope"
    data_dir.mkdir()
    op_mapping = (
        'version: "test"\n'
        "operator_mappings:\n"
        '  "tensor_cast.apply_rope.default":\n'
        "    kernel_type: InterleaveRope\n"
        "    alternate_kernel_types: [ApplyRotaryPosEmb]\n"
    )
    (data_dir / "op_mapping.yaml").write_text(op_mapping)
    (data_dir / "ApplyRotaryPosEmb.csv").write_text(ROPE_CSV.strip())
    return data_dir


def test_rope_shape_normalization(rope_data_dir):
    """TC RoPE sends [Q(1,1,144,128), K(1,4,144,128), cos(1,144,128), sin(1,144,128)]
    CSV expects [K(1,136,4,128), Q(1,136,1,128), cos(1,136,1,128), sin(1,136,1,128)].
    Should match after: reorder Q/K, transpose (B,H,S,D)->(B,S,H,D), insert head dim in cos/sin.
    """
    ds = ProfilingDataSource(rope_data_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.apply_rope.default,
        [
            torch.empty(1, 1, 144, 128, device="meta", dtype=torch.bfloat16),  # Q
            torch.empty(1, 4, 144, 128, device="meta", dtype=torch.bfloat16),  # K
            torch.empty(1, 144, 128, device="meta", dtype=torch.bfloat16),  # cos
            torch.empty(1, 144, 128, device="meta", dtype=torch.bfloat16),  # sin
        ],
    )
    result = ds.lookup(op)
    assert result is not None, "Should match RoPE after shape normalization + padding"
    assert abs(result.latency_us - 12.5) < 0.01
    assert result.details.get("kernel_type") == "ApplyRotaryPosEmb"


def test_rope_no_false_positive(rope_data_dir):
    """Wrong head count should not match."""
    ds = ProfilingDataSource(rope_data_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.apply_rope.default,
        [
            torch.empty(
                1, 8, 144, 128, device="meta", dtype=torch.bfloat16
            ),  # Q with wrong heads
            torch.empty(
                1, 8, 144, 128, device="meta", dtype=torch.bfloat16
            ),  # K with wrong heads
            torch.empty(1, 144, 128, device="meta", dtype=torch.bfloat16),
            torch.empty(1, 144, 128, device="meta", dtype=torch.bfloat16),
        ],
    )
    result = ds.lookup(op)
    assert result is None, "Wrong head count should not match"


# --- RoPE with _triton_rope + tc_input_count=2 ---

TRITON_ROPE_CSV = """\
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Average Duration(us)
"41040,4,128;41040,1,128;81920,128","DT_BF16;DT_BF16;DT_BF16","ND;ND;ND","41040,4,128;41040,1,128","DT_BF16;DT_BF16","ND;ND",55.0
"336,4,128;336,1,128;81920,128","DT_BF16;DT_BF16;DT_BF16","ND;ND;ND","336,4,128;336,1,128","DT_BF16;DT_BF16","ND;ND",8.5
"""


@pytest.fixture
def triton_rope_data_dir(tmp_path):
    data_dir = tmp_path / "triton_rope"
    data_dir.mkdir()
    op_mapping = (
        'version: "test"\n'
        "operator_mappings:\n"
        '  "tensor_cast.apply_rope.default":\n'
        "    kernel_type: _triton_rope\n"
        "    tc_input_count: 2\n"
    )
    (data_dir / "op_mapping.yaml").write_text(op_mapping)
    (data_dir / "_triton_rope.csv").write_text(TRITON_ROPE_CSV.strip())
    return data_dir


def test_triton_rope_tc_input_count_2_prefill(triton_rope_data_dir):
    """TC RoPE with _triton_rope + tc_input_count=2: Qwen3 Prefill.
    TC sends [Q(1,1,41040,128), K(1,4,41040,128), cos, sin] — tc_input_count=2 truncates to [Q, K].
    Normalize: swap Q↔K, transpose (B,H,S,D)→(B,S,H,D), strip batch=1.
    Result: [K(41040,4,128), Q(41040,1,128)] should match CSV first 2 inputs."""
    ds = ProfilingDataSource(triton_rope_data_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.apply_rope.default,
        [
            torch.empty(1, 1, 41040, 128, device="meta", dtype=torch.bfloat16),  # Q
            torch.empty(1, 4, 41040, 128, device="meta", dtype=torch.bfloat16),  # K
            torch.empty(1, 41040, 128, device="meta", dtype=torch.bfloat16),  # cos
            torch.empty(1, 41040, 128, device="meta", dtype=torch.bfloat16),  # sin
        ],
    )
    result = ds.lookup(op)
    assert result is not None, (
        "Should match _triton_rope with tc_input_count=2 after normalize"
    )
    assert abs(result.latency_us - 55.0) < 0.01


def test_triton_rope_tc_input_count_2_decode_miss(triton_rope_data_dir):
    """TC RoPE Decode: M=16 not in CSV (CSV has M=336, M=41040) — shape_coverage_gap."""
    ds = ProfilingDataSource(triton_rope_data_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.apply_rope.default,
        [
            torch.empty(1, 1, 16, 128, device="meta", dtype=torch.bfloat16),
            torch.empty(1, 4, 16, 128, device="meta", dtype=torch.bfloat16),
            torch.empty(1, 16, 128, device="meta", dtype=torch.bfloat16),
            torch.empty(1, 16, 128, device="meta", dtype=torch.bfloat16),
        ],
    )
    result = ds.lookup(op)
    assert result is None, "M=16 not in CSV — should miss (shape_coverage_gap)"


# --- RoPE dtype relaxed matching (P-E2E-1: NPU FLOAT vs TC BF16) ---

TRITON_ROPE_FLOAT_CSV = """\
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Average Duration(us)
"336,4,128;336,1,128;81920,128","DT_BF16;FLOAT;DT_BF16","ND;ND;ND","336,4,128;336,1,128","DT_BF16;FLOAT","ND;ND",12.3
"""


@pytest.fixture
def triton_rope_float_dir(tmp_path):
    """_triton_rope CSV with FLOAT dtype for K (real production data pattern)."""
    data_dir = tmp_path / "triton_rope_float"
    data_dir.mkdir()
    op_mapping = (
        'version: "test"\n'
        "operator_mappings:\n"
        '  "tensor_cast.apply_rope.default":\n'
        "    kernel_type: _triton_rope\n"
        "    tc_input_count: 2\n"
    )
    (data_dir / "op_mapping.yaml").write_text(op_mapping)
    (data_dir / "_triton_rope.csv").write_text(TRITON_ROPE_FLOAT_CSV.strip())
    return data_dir


def test_triton_rope_dtype_relaxed_hit(triton_rope_float_dir):
    """CSV K dtype=FLOAT, TC K dtype=BF16 → relaxed match → HIT.

    This is the P-E2E-1 fix: NPU _triton_rope profiling records K as FLOAT
    (FP32) while TC dispatches BF16. Performance is identical.
    """
    ds = ProfilingDataSource(triton_rope_float_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.apply_rope.default,
        [
            torch.empty(1, 1, 336, 128, device="meta", dtype=torch.bfloat16),  # Q
            torch.empty(1, 4, 336, 128, device="meta", dtype=torch.bfloat16),  # K
            torch.empty(1, 336, 128, device="meta", dtype=torch.bfloat16),  # cos
            torch.empty(1, 336, 128, device="meta", dtype=torch.bfloat16),  # sin
        ],
    )
    result = ds.lookup(op)
    assert result is not None, (
        "RoPE dtype relaxed: BF16 vs FLOAT should match for _ROPE_KERNELS"
    )
    assert abs(result.latency_us - 12.3) < 0.01


def test_matmul_dtype_relaxed_and_transpose_absorbed(tmp_path):
    """MatMul allows FLOAT<->BF16 and absorbs ND weight transpose from F.linear."""
    data_dir = tmp_path / "matmul_dtype_relaxed"
    data_dir.mkdir()
    (data_dir / "op_mapping.yaml").write_text(
        'version: "test"\n'
        "operator_mappings:\n"
        '  "aten.mm.default":\n'
        "    kernel_type: MatMulV2\n"
    )
    (data_dir / "MatMulV2.csv").write_text(
        "Input Shapes,Input Data Types,Input Formats,Output Shapes,"
        "Output Data Types,Output Formats,Average Duration(us)\n"
        '"2048,7168;256,7168","DT_BF16;DT_BF16","ND;ND",'
        '"2048,256","DT_BF16","ND",47.4\n'
    )
    ds = ProfilingDataSource(data_dir)
    op = _make_op_info(
        torch.ops.aten.mm.default,
        [
            torch.empty(2048, 7168, device="meta", dtype=torch.float32),
            torch.empty(7168, 256, device="meta", dtype=torch.float32),
        ],
    )
    result = ds.lookup(op)
    assert result is not None, (
        "MatMul should match profiling row when dtype relaxes FLOAT->DT_BF16 "
        "and the transposed ND weight is absorbed"
    )
    assert abs(result.latency_us - 47.4) < 0.01
    assert result.details.get("kernel_type") == "MatMulV2"


COMPOSITE_MATMUL_CSV = """\
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Average Duration(us)
"136,512;32,320,16,16","DT_BF16;DT_BF16","ND;FRACTAL_NZ","136,5120","DT_BF16","ND",14.156
"""

# mat1[144,512] @ mat2[512,5120] -> output[144,5120]
# message_bytes = 144 * 5120 * 2 (bfloat16) = 1474560, num_devices = 2
COMPOSITE_COMM_CSV = """\
message_bytes,num_devices,Duration(us)
1474560,2,200.00
"""


@pytest.fixture
def composite_data_dir(tmp_path):
    data_dir = tmp_path / "composite"
    data_dir.mkdir()
    op_mapping = (
        'version: "test"\n'
        "operator_mappings:\n"
        '  "tensor_cast.matmul_all_reduce.default":\n'
        "    composite: true\n"
        "    sub_kernels: [MatMulV2, hcom_allReduce_]\n"
    )
    (data_dir / "op_mapping.yaml").write_text(op_mapping)
    (data_dir / "MatMulV2.csv").write_text(COMPOSITE_MATMUL_CSV.strip())
    (data_dir / "hcom_allReduce_.csv").write_text(COMPOSITE_COMM_CSV.strip())
    return data_dir


def test_composite_decomposition_matmul(composite_data_dir):
    """matmul_all_reduce decomposes to MatMulV2 + hcom_allReduce_; latency is summed."""
    ds = ProfilingDataSource(composite_data_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.matmul_all_reduce.default,
        [
            torch.empty(144, 512, device="meta", dtype=torch.bfloat16),  # mat1
            torch.empty(512, 5120, device="meta", dtype=torch.bfloat16),  # mat2
            None,  # bias
            0,  # rank
            [0, 1],  # rank_group
        ],
    )
    result = ds.lookup(op)
    assert result is not None, (
        "Should match both MatMulV2 and hcom_allReduce_ sub-kernels"
    )
    assert abs(result.latency_us - (14.156 + 200.00)) < 0.01
    assert result.details.get("composite") is True
    assert result.confidence == 0.9
    assert result.sub_kernel_shapes is not None
    assert len(result.sub_kernel_shapes) == 2
    assert result.sub_kernel_shapes[0].kernel_type == "MatMulV2"
    assert result.sub_kernel_shapes[0].simulation_shapes == [[144, 512], [512, 5120]]
    assert result.sub_kernel_shapes[0].kernel_shapes == [[136, 512], [32, 320, 16, 16]]
    assert result.sub_kernel_shapes[0].shape_match_rule == "padding"
    assert result.sub_kernel_shapes[1].kernel_type == "hcom_allReduce_"
    assert result.sub_kernel_shapes[1].simulation_shapes == []
    assert result.sub_kernel_shapes[1].kernel_shapes == []
    assert result.sub_kernel_shapes[1].shape_match_rule == "comm"


def test_composite_mc2_has_sub_kernel_durations(composite_data_dir):
    """MC2 composite should include per-sub-kernel durations in details."""
    ds = ProfilingDataSource(composite_data_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.matmul_all_reduce.default,
        [
            torch.empty(144, 512, device="meta", dtype=torch.bfloat16),
            torch.empty(512, 5120, device="meta", dtype=torch.bfloat16),
            None,
            0,
            [0, 1],
        ],
    )
    result = ds.lookup(op)
    assert result is not None
    skd = result.details.get("sub_kernel_durations")
    assert skd is not None, "sub_kernel_durations missing from composite details"
    assert len(skd) == 2
    assert skd[0] == ("MatMulV2", 14.16)
    assert skd[1][0] == "hcom_allReduce_"
    assert abs(skd[1][1] - 200.0) < 0.01


def test_composite_no_sub_kernels(sample_data_dir):
    """Composite op without matching sub-kernel CSVs returns None."""
    ds = ProfilingDataSource(sample_data_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.multihead_latent_attention.default,
        [torch.empty(136, 5120, device="meta", dtype=torch.bfloat16)],
    )
    result = ds.lookup(op)
    assert result is None


# --- B2: composite sub-kernel sum tests ---


# Fixture: compute CSV only (no comm CSV) — for comm-miss scenario
@pytest.fixture
def mc2_compute_only_dir(tmp_path):
    data_dir = tmp_path / "mc2_compute_only"
    data_dir.mkdir()
    op_mapping = (
        'version: "test"\n'
        "operator_mappings:\n"
        '  "tensor_cast.matmul_all_reduce.default":\n'
        "    composite: true\n"
        "    sub_kernels: [MatMulV2, hcom_allReduce_]\n"
    )
    (data_dir / "op_mapping.yaml").write_text(op_mapping)
    (data_dir / "MatMulV2.csv").write_text(COMPOSITE_MATMUL_CSV.strip())
    # No hcom_allReduce_.csv
    return data_dir


# Fixture: compute CSV with wrong shapes — for shape-mismatch scenario
@pytest.fixture
def mc2_wrong_shape_dir(tmp_path):
    data_dir = tmp_path / "mc2_wrong_shape"
    data_dir.mkdir()
    op_mapping = (
        'version: "test"\n'
        "operator_mappings:\n"
        '  "tensor_cast.matmul_all_reduce.default":\n'
        "    composite: true\n"
        "    sub_kernels: [MatMulV2, hcom_allReduce_]\n"
    )
    (data_dir / "op_mapping.yaml").write_text(op_mapping)
    # CSV has different shapes — won't match mat1[144,512] @ mat2[512,5120]
    wrong_csv = (
        "Input Shapes,Input Data Types,Input Formats,Output Shapes,"
        "Output Data Types,Output Formats,Average Duration(us)\n"
        '"1,256;16,160,16,16","DT_BF16;DT_BF16","ND;FRACTAL_NZ",'
        '"1,2560","DT_BF16","ND",99.0\n'
    )
    (data_dir / "MatMulV2.csv").write_text(wrong_csv.strip())
    (data_dir / "hcom_allReduce_.csv").write_text(COMPOSITE_COMM_CSV.strip())
    return data_dir


def test_composite_mc2_compute_hit_comm_miss_returns_none(mc2_compute_only_dir):
    """Compute sub-kernel hits but comm CSV absent → None + comm_sub_kernel_miss."""
    ds = ProfilingDataSource(mc2_compute_only_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.matmul_all_reduce.default,
        [
            torch.empty(144, 512, device="meta", dtype=torch.bfloat16),
            torch.empty(512, 5120, device="meta", dtype=torch.bfloat16),
            None,
            0,
            [0, 1],
        ],
    )
    result = ds.lookup(op)
    assert result is None
    assert ds.last_miss_reason == "comm_sub_kernel_miss"


def test_composite_mc2_compute_miss_returns_none(mc2_wrong_shape_dir):
    """Compute CSV exists but shapes don't match → None + shape_mismatch."""
    ds = ProfilingDataSource(mc2_wrong_shape_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.matmul_all_reduce.default,
        [
            torch.empty(144, 512, device="meta", dtype=torch.bfloat16),
            torch.empty(512, 5120, device="meta", dtype=torch.bfloat16),
            None,
            0,
            [0, 1],
        ],
    )
    result = ds.lookup(op)
    assert result is None
    assert ds.last_miss_reason == "shape_mismatch"


def test_composite_mla_csv_not_found_returns_none(sample_data_dir):
    """MLA composite: attempts composite lookup, returns csv_not_found (sub-kernel CSV missing)."""
    ds = ProfilingDataSource(sample_data_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.multihead_latent_attention.default,
        [torch.empty(136, 5120, device="meta", dtype=torch.bfloat16)],
    )
    result = ds.lookup(op)
    assert result is None
    # After C1 fix: composite lookup attempted, sub-kernel CSV missing
    assert ds.last_miss_reason != "mla_not_implemented"


def test_composite_no_sub_kernels_miss_reason(tmp_path):
    """Composite op with empty sub_kernels list → None + no_sub_kernels."""
    data_dir = tmp_path / "empty_sub"
    data_dir.mkdir()
    op_mapping = (
        'version: "test"\n'
        "operator_mappings:\n"
        '  "tensor_cast.matmul_all_reduce.default":\n'
        "    composite: true\n"
        "    sub_kernels: []\n"
    )
    (data_dir / "op_mapping.yaml").write_text(op_mapping)
    ds = ProfilingDataSource(data_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.matmul_all_reduce.default,
        [
            torch.empty(144, 512, device="meta", dtype=torch.bfloat16),
            torch.empty(512, 5120, device="meta", dtype=torch.bfloat16),
            None,
            0,
            [0, 1],
        ],
    )
    result = ds.lookup(op)
    assert result is None
    assert ds.last_miss_reason == "no_sub_kernels"


# --- B2: quant MC2 + MLA placeholder tests ---

# QuantBatchMatmulV3 CSV: INT8 inputs, ND format
# x[144,512] INT8, w[512,5120] INT8 → output[144,5120] BF16
QUANT_MATMUL_CSV = """\
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Average Duration(us)
"144,512;512,5120","INT8;INT8","ND;ND","144,5120","DT_BF16","ND",22.5
"""

# Comm CSV: message_bytes = 144 * 5120 * 2 (BF16 output) = 1474560
QUANT_COMM_CSV = """\
message_bytes,num_devices,Duration(us)
1474560,2,200.00
"""

# Wrong comm CSV: message_bytes = 144 * 5120 * 1 (INT8 input) = 737280
QUANT_COMM_WRONG_CSV = """\
message_bytes,num_devices,Duration(us)
737280,2,150.00
"""


@pytest.fixture
def quant_mc2_data_dir(tmp_path):
    """Quant MC2 fixture: static_quant_linear_all_reduce with tc_input_count=2."""
    data_dir = tmp_path / "quant_mc2"
    data_dir.mkdir()
    op_mapping = (
        'version: "test"\n'
        "operator_mappings:\n"
        '  "tensor_cast.static_quant_linear_all_reduce.default":\n'
        "    composite: true\n"
        "    sub_kernels: [QuantBatchMatmulV3, hcom_allReduce_]\n"
        "    tc_input_count: 2\n"
    )
    (data_dir / "op_mapping.yaml").write_text(op_mapping)
    (data_dir / "QuantBatchMatmulV3.csv").write_text(QUANT_MATMUL_CSV.strip())
    (data_dir / "hcom_allReduce_.csv").write_text(QUANT_COMM_CSV.strip())
    return data_dir


def test_composite_quant_mc2_hit(quant_mc2_data_dir):
    """static_quant_linear_all_reduce: tc_input_count=2 truncates 6 tensor args to x+w,
    matches QuantBatchMatmulV3 + hcom_allReduce_, latency summed."""
    ds = ProfilingDataSource(quant_mc2_data_dir)
    # 6 tensor args: x, w, scale, zero_point, bias, per_token_scale
    op = _make_op_info(
        torch.ops.tensor_cast.static_quant_linear_all_reduce.default,
        [
            torch.empty(144, 512, device="meta", dtype=torch.int8),  # x
            torch.empty(512, 5120, device="meta", dtype=torch.int8),  # w
            torch.empty(5120, device="meta", dtype=torch.bfloat16),  # scale
            torch.empty(5120, device="meta", dtype=torch.int8),  # zero_point
            torch.empty(5120, device="meta", dtype=torch.bfloat16),  # bias
            torch.empty(144, device="meta", dtype=torch.bfloat16),  # per_token_scale
            0,  # rank
            [0, 1],  # rank_group
        ],
        output_tensors=[torch.empty(144, 5120, device="meta", dtype=torch.bfloat16)],
    )
    result = ds.lookup(op)
    assert result is not None, "Should match with tc_input_count=2 truncation"
    assert abs(result.latency_us - (22.5 + 200.00)) < 0.01
    assert result.details.get("composite") is True


def test_composite_quant_mc2_message_bytes_uses_output_dtype(tmp_path):
    """message_bytes should use BF16 output (2B) not INT8 input (1B).
    With INT8 input: 144*5120*1=737280. With BF16 output: 144*5120*2=1474560.
    Only the BF16-sized comm CSV should match."""
    data_dir = tmp_path / "quant_mc2_dtype"
    data_dir.mkdir()
    op_mapping = (
        'version: "test"\n'
        "operator_mappings:\n"
        '  "tensor_cast.static_quant_linear_all_reduce.default":\n'
        "    composite: true\n"
        "    sub_kernels: [QuantBatchMatmulV3, hcom_allReduce_]\n"
        "    tc_input_count: 2\n"
    )
    (data_dir / "op_mapping.yaml").write_text(op_mapping)
    (data_dir / "QuantBatchMatmulV3.csv").write_text(QUANT_MATMUL_CSV.strip())
    # Only provide INT8-sized comm CSV (737280) — should NOT match
    (data_dir / "hcom_allReduce_.csv").write_text(QUANT_COMM_WRONG_CSV.strip())

    ds = ProfilingDataSource(data_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.static_quant_linear_all_reduce.default,
        [
            torch.empty(144, 512, device="meta", dtype=torch.int8),
            torch.empty(512, 5120, device="meta", dtype=torch.int8),
            torch.empty(5120, device="meta", dtype=torch.bfloat16),
            torch.empty(5120, device="meta", dtype=torch.int8),
            torch.empty(5120, device="meta", dtype=torch.bfloat16),
            torch.empty(144, device="meta", dtype=torch.bfloat16),
            0,
            [0, 1],
        ],
        output_tensors=[torch.empty(144, 5120, device="meta", dtype=torch.bfloat16)],
    )
    result = ds.lookup(op)
    # BF16 output → message_bytes=1474560, but CSV only has 737280 → comm miss
    assert result is None
    assert ds.last_miss_reason == "comm_sub_kernel_miss"


def test_composite_mla_attempts_lookup(tmp_path):
    """After C1: MLA attempts composite lookup instead of rejecting."""
    data_dir = tmp_path / "mla_composite"
    data_dir.mkdir()
    op_mapping = (
        'version: "test"\n'
        "operator_mappings:\n"
        '  "tensor_cast.multihead_latent_attention.default":\n'
        "    composite: true\n"
        "    sub_kernels: [BatchMatMulV2, FusedInferAttentionScore]\n"
    )
    (data_dir / "op_mapping.yaml").write_text(op_mapping)
    ds = ProfilingDataSource(data_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.multihead_latent_attention.default,
        [torch.empty(136, 5120, device="meta", dtype=torch.bfloat16)],
    )
    result = ds.lookup(op)
    assert result is None  # CSVs missing, but composite lookup attempted
    assert ds.last_miss_reason != "mla_not_implemented"


def test_composite_mlapo_attempts_lookup(tmp_path):
    """After C1: MLAPO attempts composite lookup instead of rejecting."""
    data_dir = tmp_path / "mlapo_composite"
    data_dir.mkdir()
    op_mapping = (
        'version: "test"\n'
        "operator_mappings:\n"
        '  "tensor_cast.mlapo.default":\n'
        "    composite: true\n"
        "    sub_kernels: [MatMulV2, KvRmsNormRopeCache]\n"
    )
    (data_dir / "op_mapping.yaml").write_text(op_mapping)
    ds = ProfilingDataSource(data_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.mlapo.default,
        [torch.empty(136, 5120, device="meta", dtype=torch.bfloat16)],
    )
    result = ds.lookup(op)
    assert result is None  # CSVs missing, but composite lookup attempted
    assert ds.last_miss_reason != "mla_not_implemented"


def test_composite_tc_input_count_truncation(tmp_path):
    """tc_input_count truncation: 4 tensor args truncated to 2, matches CSV with 2 inputs."""
    data_dir = tmp_path / "tc_input_trunc"
    data_dir.mkdir()
    # Generic composite with tc_input_count=2 and compute-only sub_kernels
    op_mapping = (
        'version: "test"\n'
        "operator_mappings:\n"
        '  "tensor_cast.static_quant_linear_all_reduce.default":\n'
        "    composite: true\n"
        "    sub_kernels: [QuantBatchMatmulV3]\n"
        "    tc_input_count: 2\n"
    )
    (data_dir / "op_mapping.yaml").write_text(op_mapping)
    (data_dir / "QuantBatchMatmulV3.csv").write_text(QUANT_MATMUL_CSV.strip())

    ds = ProfilingDataSource(data_dir)
    # 4 tensor args — without truncation, len(tc_inputs)=4 != len(csv_shapes)=2 → miss
    op = _make_op_info(
        torch.ops.tensor_cast.static_quant_linear_all_reduce.default,
        [
            torch.empty(144, 512, device="meta", dtype=torch.int8),
            torch.empty(512, 5120, device="meta", dtype=torch.int8),
            torch.empty(5120, device="meta", dtype=torch.bfloat16),  # extra: scale
            torch.empty(144, device="meta", dtype=torch.bfloat16),  # extra: per_token
            0,
            [0, 1],
        ],
    )
    result = ds.lookup(op)
    assert result is not None, "tc_input_count=2 should truncate to x+w and match"
    assert abs(result.latency_us - 22.5) < 0.01
    assert result.details.get("composite") is True


# --- Communication query tests ---

COMM_OP_MAPPING_YAML = """
version: "test"
device: TEST_DEVICE

operator_mappings:
  "tensor_cast.all_reduce.default":
    kernel_type: hcom_allReduce_
    category: communication
  "tensor_cast.all_gather.default":
    kernel_type: hcom_allGather_
    category: communication
  "tensor_cast.all_to_all.default":
    kernel_type: hcom_alltoallv_
    category: communication
  "aten.mm.default":
    kernel_type: MatMulV2
"""

# Comm CSV format: message_bytes, num_devices, dtype, topology_tier, Duration(us)
COMM_ALLREDUCE_CSV = """\
message_bytes,num_devices,dtype,topology_tier,Duration(us)
1310720,16,DT_BF16,0,689.96
655360,16,DT_BF16,0,412.50
1310720,4,DT_BF16,2,125.30
"""

COMM_ALLGATHER_CSV = """\
message_bytes,num_devices,dtype,topology_tier,Duration(us)
655360,16,DT_BF16,0,167.62
"""


@pytest.fixture
def comm_data_dir(tmp_path):
    data_dir = tmp_path / "comm"
    data_dir.mkdir()
    (data_dir / "op_mapping.yaml").write_text(COMM_OP_MAPPING_YAML)
    (data_dir / "hcom_allReduce_.csv").write_text(COMM_ALLREDUCE_CSV.strip())
    (data_dir / "hcom_allGather_.csv").write_text(COMM_ALLGATHER_CSV.strip())
    return data_dir


def test_comm_allreduce_exact_match(comm_data_dir):
    """all_reduce with matching message_bytes + num_devices should return latency."""
    ds = ProfilingDataSource(comm_data_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.all_reduce.default,
        [
            torch.empty(1, 640, 1024, device="meta", dtype=torch.bfloat16),
            0,
            list(range(16)),
        ],
    )
    result = ds.lookup(op)
    assert result is not None, "Should match comm CSV by message_bytes + num_devices"
    assert abs(result.latency_us - 689.96) < 0.01
    assert result.source == QuerySource.MEASURED
    assert result.details.get("kernel_type") == "hcom_allReduce_"
    assert result.shape_match_info is not None
    assert result.shape_match_info.simulation_shapes == [[1310720]]
    assert result.shape_match_info.kernel_shapes == [[1310720]]
    assert result.shape_match_info.shape_match_rule == "comm"


def test_comm_allreduce_different_shape_same_bytes(comm_data_dir):
    """Different tensor shape but same message_bytes should still match."""
    ds = ProfilingDataSource(comm_data_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.all_reduce.default,
        [
            torch.empty(640, 1024, device="meta", dtype=torch.bfloat16),
            0,
            list(range(16)),
        ],
    )
    result = ds.lookup(op)
    assert result is not None
    assert abs(result.latency_us - 689.96) < 0.01


def test_comm_allreduce_miss_wrong_bytes(comm_data_dir):
    """Non-matching message_bytes should return None."""
    ds = ProfilingDataSource(comm_data_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.all_reduce.default,
        [
            torch.empty(100, 100, device="meta", dtype=torch.bfloat16),
            0,
            list(range(16)),
        ],
    )
    result = ds.lookup(op)
    assert result is None


def test_comm_allgather_match(comm_data_dir):
    """all_gather(x, dim, rank, rank_group) should match by message_bytes."""
    ds = ProfilingDataSource(comm_data_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.all_gather.default,
        [
            torch.empty(1, 640, 512, device="meta", dtype=torch.bfloat16),
            0,
            0,
            list(range(16)),
        ],
    )
    result = ds.lookup(op)
    assert result is not None
    assert abs(result.latency_us - 167.62) < 0.01


def test_comm_no_csv_returns_none(comm_data_dir):
    """Communication op without CSV file should return None."""
    ds = ProfilingDataSource(comm_data_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.all_to_all.default,
        [
            torch.empty(100, 512, device="meta", dtype=torch.bfloat16),
            [25] * 4,
            [25] * 4,
            0,
            [0, 1, 2, 3],
        ],
    )
    result = ds.lookup(op)
    assert result is None


# --- _comm_data_dir fallback tests ---

COMM_DATA_REF_OP_MAPPING_YAML = """
version: "test"
device: TEST_DEVICE

communication_data_ref: "../hccl_ref/"

operator_mappings:
  "tensor_cast.all_reduce.default":
    kernel_type: hcom_allReduce_
    category: communication
"""


@pytest.fixture
def comm_data_ref_dir(tmp_path):
    """Data dir with communication_data_ref pointing to a sibling hccl dir."""
    data_dir = tmp_path / "main"
    data_dir.mkdir()
    (data_dir / "op_mapping.yaml").write_text(COMM_DATA_REF_OP_MAPPING_YAML)
    # CSV lives in the referenced dir, not in data_dir
    hccl_dir = tmp_path / "hccl_ref"
    hccl_dir.mkdir()
    (hccl_dir / "hcom_allReduce_.csv").write_text(COMM_ALLREDUCE_CSV.strip())
    return data_dir


def test_comm_data_ref_fallback(comm_data_ref_dir):
    """_load_csv should find CSV via communication_data_ref when not in data_dir."""
    ds = ProfilingDataSource(comm_data_ref_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.all_reduce.default,
        [
            torch.empty(1, 640, 1024, device="meta", dtype=torch.bfloat16),
            0,
            list(range(16)),
        ],
    )
    result = ds.lookup(op)
    assert result is not None
    assert abs(result.latency_us - 689.96) < 0.01


# --- Attention special query tests ---

ATTN_OP_MAPPING_YAML = """
version: "test"
device: TEST_DEVICE

operator_mappings:
  "tensor_cast.attention.default":
    kernel_type: FusedInferAttentionScore
    query_mode: attention_special
  "tensor_cast.attention_quant.default":
    kernel_type: FusedInferAttentionScore
    query_mode: attention_special
"""

# Enriched FIA CSV format with avg_seq_len + Input Shapes
_ATTN_FIA_HEADER = (
    "Input Shapes,Input Data Types,Input Formats,Output Shapes,"
    "Output Data Types,Output Formats,Duration(us),avg_seq_len"
)


def _make_fia_row(q_shape_str, out_shape_str, duration, avg_seq_len):
    """Build one enriched FIA CSV row with minimal slot data."""
    return (
        f'"{q_shape_str}"'
        ',"DT_BF16;DT_BF16;DT_BF16;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;'
        "INT64;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;"
        "DT_UNDEFINED;DT_UNDEFINED;INT32;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;"
        "DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;"
        "DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;"
        'DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED"'
        ',"ND;ND;ND;NULL;NULL;NULL;ND;NULL;NULL;NULL;NULL;NULL;NULL;NULL;ND;'
        'NULL;NULL;NULL;NULL;NULL;NULL;NULL;NULL;NULL;NULL;NULL;NULL;NULL;NULL;NULL;NULL"'
        f',"""{out_shape_str}""","DT_BF16;FLOAT","ND;ND",{duration},{avg_seq_len}'
    )


ATTN_FIA_CSV = (
    _ATTN_FIA_HEADER
    + "\n"
    + _make_fia_row(
        "7000,4,128;56,128,4,128;56,128,4,128;;;;7000;;;;;;;;7000,56;;;;;;;;;;;;;;",
        "7000,4,128;",
        98.50,
        3500,
    )
    + "\n"
    + _make_fia_row(
        "10,4,128;360,128,4,128;360,128,4,128;;;;10;;;;;;;;10,36;;;;;;;;;;;;;;",
        "10,4,128;",
        890.70,
        4500,
    )
    + "\n"
    + _make_fia_row(
        "1,8,128;32,128,8,128;32,128,8,128;;;;1;;;;;;;;1,32;;;;;;;;;;;;;;",
        "1,8,128;",
        112.36,
        4096,
    )
)


@pytest.fixture
def attn_data_dir(tmp_path):
    data_dir = tmp_path / "attn"
    data_dir.mkdir()
    (data_dir / "op_mapping.yaml").write_text(ATTN_OP_MAPPING_YAML)
    (data_dir / "FusedInferAttentionScore.csv").write_text(ATTN_FIA_CSV.strip())
    return data_dir


def test_attention_prefill_match(attn_data_dir):
    """Prefill: batch=2, seq_lens=[3500,3500], 4 heads, head_dim=128."""
    ds = ProfilingDataSource(attn_data_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.attention.default,
        [
            torch.empty(
                7000, 512, device="meta", dtype=torch.bfloat16
            ),  # query: hidden=4*128=512
            torch.empty(
                56, 128, 4, 128, device="meta", dtype=torch.bfloat16
            ),  # key (paged)
            torch.empty(56, 128, 4, 128, device="meta", dtype=torch.bfloat16),  # value
            None,  # attention_mask
            torch.empty(2, 28, device="meta", dtype=torch.int32),  # block_table
            torch.empty(3, device="meta", dtype=torch.int64),  # query_start_loc
            torch.tensor([3500, 3500], dtype=torch.int64),  # seq_lens
            torch.tensor([3500, 3500], dtype=torch.int64),  # query_lens
        ],
    )
    result = ds.lookup(op)
    assert result is not None, "Should match FIA by batch_size=2, avg_seq_len=3500"
    assert abs(result.latency_us - 98.50) < 0.01
    assert result.details.get("kernel_type") == "FusedInferAttentionScore"
    assert result.details.get("avg_seq_len") == 3500
    assert result.details.get("sparse_mode") == 3
    assert result.details.get("num_kv_heads") == 4
    assert result.shape_match_info is not None
    assert result.shape_match_info.simulation_shapes == [
        [7000, 512],
        [56, 128, 4, 128],
        [56, 128, 4, 128],
        [2, 28],
        [3],
        [2],
        [2],
    ]
    assert result.shape_match_info.kernel_shapes == []
    assert result.shape_match_info.shape_match_rule == "attention"


def test_attention_decode_match(attn_data_dir):
    """Decode: batch=10, seq_lens=[4500]*10, 4 heads, head_dim=128."""
    ds = ProfilingDataSource(attn_data_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.attention.default,
        [
            torch.empty(10, 512, device="meta", dtype=torch.bfloat16),  # query
            torch.empty(360, 128, 4, 128, device="meta", dtype=torch.bfloat16),  # key
            torch.empty(360, 128, 4, 128, device="meta", dtype=torch.bfloat16),  # value
            None,
            torch.empty(10, 36, device="meta", dtype=torch.int32),
            torch.empty(11, device="meta", dtype=torch.int64),
            torch.tensor([4500] * 10, dtype=torch.int64),  # seq_lens
            torch.tensor([1] * 10, dtype=torch.int64),  # query_lens
        ],
    )
    result = ds.lookup(op)
    assert result is not None, "Should match FIA by batch_size=10, avg_seq_len=4500"
    assert abs(result.latency_us - 890.70) < 0.01


def test_attention_miss_wrong_heads(attn_data_dir):
    """Wrong num_heads should not match."""
    ds = ProfilingDataSource(attn_data_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.attention.default,
        [
            torch.empty(1, 2048, device="meta", dtype=torch.bfloat16),  # 16 heads * 128
            torch.empty(
                32, 128, 16, 128, device="meta", dtype=torch.bfloat16
            ),  # 16 kv heads
            torch.empty(32, 128, 16, 128, device="meta", dtype=torch.bfloat16),
            None,
            torch.empty(1, 32, device="meta", dtype=torch.int32),
            torch.empty(2, device="meta", dtype=torch.int64),
            torch.tensor([4096], dtype=torch.int64),
            torch.tensor([4096], dtype=torch.int64),
        ],
    )
    result = ds.lookup(op)
    assert result is None, "16 heads not in CSV, should miss"


def test_attention_miss_no_seq_lens(attn_data_dir):
    """If seq_lens is None, should return None gracefully."""
    ds = ProfilingDataSource(attn_data_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.attention.default,
        [
            torch.empty(100, 512, device="meta", dtype=torch.bfloat16),
            torch.empty(10, 128, 4, 128, device="meta", dtype=torch.bfloat16),
            torch.empty(10, 128, 4, 128, device="meta", dtype=torch.bfloat16),
            None,
            None,
            None,
            None,
            None,
        ],
    )
    result = ds.lookup(op)
    assert result is None, "No seq_lens -> can't compute batch/seq, return None"


# --- topology_tier matching tests ---
#
# Test grid: [2, 4] — 2 pods, 4 devices per pod
#   tier 0 (inter_pod): ranks spanning different pods  (e.g. [0, 4])
#   tier 1 (intra_pod): ranks within same pod          (e.g. [0, 1])
#
# Rank → coord mapping:
#   rank 0 → [0, 0],  rank 1 → [0, 1],  rank 2 → [0, 2],  rank 3 → [0, 3]
#   rank 4 → [1, 0],  rank 5 → [1, 1],  rank 6 → [1, 2],  rank 7 → [1, 3]


def _make_test_comm_grid() -> CommGrid:
    """2-tier grid [2, 4]: tier 0 = inter-pod, tier 1 = intra-pod."""
    return CommGrid(
        grid=torch.zeros([2, 4], dtype=torch.int32),
        topologies={
            0: InterconnectTopology(bandwidth_bytes_ps=196e9, latency_s=5.5e-6),
            1: InterconnectTopology(bandwidth_bytes_ps=224e9, latency_s=0.2e-6),
        },
    )


# CSV with two rows: same message_bytes+num_devices=2, different topology_tier
# num_devices=2 matches rank_group size ([0,4] or [0,1] both have 2 elements)
# message_bytes = torch.empty(4,1024,160,bfloat16).nelement()*2 = 655360*2 = 1310720
COMM_TIERED_CSV = """\
message_bytes,num_devices,dtype,topology_tier,Duration(us)
1310720,2,DT_BF16,0,689.96
1310720,2,DT_BF16,1,125.30
"""

COMM_TIER0_ONLY_CSV = """\
message_bytes,num_devices,dtype,topology_tier,Duration(us)
1310720,2,DT_BF16,0,689.96
"""

COMM_NO_TIER_COL_CSV = """\
message_bytes,num_devices,dtype,Duration(us)
1310720,2,DT_BF16,350.00
"""


@pytest.fixture
def tiered_comm_dir(tmp_path):
    data_dir = tmp_path / "tiered_comm"
    data_dir.mkdir()
    op_mapping = (
        'version: "test"\n'
        "operator_mappings:\n"
        '  "tensor_cast.all_reduce.default":\n'
        "    kernel_type: hcom_allReduce_\n"
        "    category: communication\n"
    )
    (data_dir / "op_mapping.yaml").write_text(op_mapping)
    (data_dir / "hcom_allReduce_.csv").write_text(COMM_TIERED_CSV.strip())
    return data_dir


@pytest.fixture
def tier0_only_comm_dir(tmp_path):
    data_dir = tmp_path / "tier0_only"
    data_dir.mkdir()
    op_mapping = (
        'version: "test"\n'
        "operator_mappings:\n"
        '  "tensor_cast.all_reduce.default":\n'
        "    kernel_type: hcom_allReduce_\n"
        "    category: communication\n"
    )
    (data_dir / "op_mapping.yaml").write_text(op_mapping)
    (data_dir / "hcom_allReduce_.csv").write_text(COMM_TIER0_ONLY_CSV.strip())
    return data_dir


@pytest.fixture
def no_tier_col_comm_dir(tmp_path):
    data_dir = tmp_path / "no_tier_col"
    data_dir.mkdir()
    op_mapping = (
        'version: "test"\n'
        "operator_mappings:\n"
        '  "tensor_cast.all_reduce.default":\n'
        "    kernel_type: hcom_allReduce_\n"
        "    category: communication\n"
    )
    (data_dir / "op_mapping.yaml").write_text(op_mapping)
    (data_dir / "hcom_allReduce_.csv").write_text(COMM_NO_TIER_COL_CSV.strip())
    return data_dir


# --- get_topology_tier unit tests ---


def test_get_topology_tier_inter_pod():
    """Ranks spanning different pods → tier 0 (inter_pod)."""
    comm_grid = _make_test_comm_grid()
    # rank 0 → [0,0], rank 4 → [1,0]: differ at dim 0 → tier 0
    assert get_topology_tier(comm_grid, [0, 4]) == 0


def test_get_topology_tier_intra_pod():
    """Ranks within same pod → tier 1 (intra_pod)."""
    comm_grid = _make_test_comm_grid()
    # rank 0 → [0,0], rank 1 → [0,1]: differ at dim 1 → tier 1
    assert get_topology_tier(comm_grid, [0, 1]) == 1


def test_get_topology_tier_multi_rank_intra():
    """All ranks in same pod → tier 1."""
    comm_grid = _make_test_comm_grid()
    assert get_topology_tier(comm_grid, [0, 1, 2, 3]) == 1


def test_get_topology_tier_multi_rank_inter():
    """Ranks spanning pods → tier 0."""
    comm_grid = _make_test_comm_grid()
    assert get_topology_tier(comm_grid, [0, 1, 4, 5]) == 0


def _make_device_profile_with_comm_grid(comm_grid):
    """Wrap a CommGrid in a mock DeviceProfile for ProfilingDataSource."""
    mock_dp = MagicMock()
    mock_dp.comm_grid = comm_grid
    return mock_dp


# --- _lookup_comm topology_tier integration tests ---


def test_comm_topology_tier_selects_correct_row(tiered_comm_dir):
    """With comm_grid, inter-pod group (tier 0) should match the tier=0 row (689.96 us)."""
    comm_grid = _make_test_comm_grid()
    ds = ProfilingDataSource(
        tiered_comm_dir, _make_device_profile_with_comm_grid(comm_grid)
    )
    # rank_group [0,4] spans pods → tier 0
    # tensor: 4 devices, message_bytes = 4 * 1024 * 160 * 2 = 1310720
    op = _make_op_info(
        torch.ops.tensor_cast.all_reduce.default,
        [
            torch.empty(4, 1024, 160, device="meta", dtype=torch.bfloat16),
            0,
            [0, 4],
        ],
    )
    result = ds.lookup(op)
    assert result is not None
    assert abs(result.latency_us - 689.96) < 0.01
    assert result.details.get("topology_tier") == 0


def test_comm_topology_tier_intra_pod_row(tiered_comm_dir):
    """Intra-pod group (tier 1) should match the tier=1 row (125.30 us)."""
    comm_grid = _make_test_comm_grid()
    ds = ProfilingDataSource(
        tiered_comm_dir, _make_device_profile_with_comm_grid(comm_grid)
    )
    # rank_group [0,1] within pod → tier 1
    op = _make_op_info(
        torch.ops.tensor_cast.all_reduce.default,
        [
            torch.empty(4, 1024, 160, device="meta", dtype=torch.bfloat16),
            0,
            [0, 1],
        ],
    )
    result = ds.lookup(op)
    assert result is not None
    assert abs(result.latency_us - 125.30) < 0.01
    assert result.details.get("topology_tier") == 1


def test_comm_topology_tier_miss_when_tier_absent(tier0_only_comm_dir):
    """Intra-pod group (tier 1) should MISS when CSV only has tier=0 data."""
    comm_grid = _make_test_comm_grid()
    ds = ProfilingDataSource(
        tier0_only_comm_dir, _make_device_profile_with_comm_grid(comm_grid)
    )
    op = _make_op_info(
        torch.ops.tensor_cast.all_reduce.default,
        [
            torch.empty(4, 1024, 160, device="meta", dtype=torch.bfloat16),
            0,
            [0, 1],  # intra-pod → tier 1, not in CSV
        ],
    )
    result = ds.lookup(op)
    assert result is None, "tier=1 not in CSV, should miss"


def test_comm_no_comm_grid_ignores_topology_tier(tiered_comm_dir):
    """Without comm_grid, topology_tier column is ignored; first matching row returned."""
    ds = ProfilingDataSource(tiered_comm_dir)  # no comm_grid
    op = _make_op_info(
        torch.ops.tensor_cast.all_reduce.default,
        [
            torch.empty(4, 1024, 160, device="meta", dtype=torch.bfloat16),
            0,
            [0, 1],
        ],
    )
    result = ds.lookup(op)
    # Both rows match on message_bytes+num_devices; first row (tier=0, 689.96) returned
    assert result is not None
    assert abs(result.latency_us - 689.96) < 0.01
    assert result.details.get("topology_tier") is None


def test_comm_csv_without_topology_tier_col(no_tier_col_comm_dir):
    """CSV without topology_tier column works fine even when comm_grid is provided."""
    comm_grid = _make_test_comm_grid()
    ds = ProfilingDataSource(
        no_tier_col_comm_dir, _make_device_profile_with_comm_grid(comm_grid)
    )
    op = _make_op_info(
        torch.ops.tensor_cast.all_reduce.default,
        [
            torch.empty(4, 1024, 160, device="meta", dtype=torch.bfloat16),
            0,
            [0, 1],
        ],
    )
    result = ds.lookup(op)
    assert result is not None
    assert abs(result.latency_us - 350.00) < 0.01


# --- communication_data_ref path redirection tests ---
#
# Directory layout mirrors production:
#   vllm_ascend/v0.13.0/op_mapping.yaml  ← communication_data_ref: "../../hccl/v8.1.RC1/"
#   hccl/v8.1.RC1/hcom_allReduce_.csv
#
# message_bytes = torch.empty(4,1024,160,bfloat16).nelement()*2 = 1310720

_COMM_REF_OP_MAPPING_WITH_REF = """\
version: "test"
communication_data_ref: "../../hccl/v8.1.RC1/"
operator_mappings:
  "tensor_cast.all_reduce.default":
    kernel_type: hcom_allReduce_
    category: communication
"""

_COMM_REF_OP_MAPPING_NO_REF = """\
version: "test"
operator_mappings:
  "tensor_cast.all_reduce.default":
    kernel_type: hcom_allReduce_
    category: communication
"""

_COMM_REF_CSV = """\
message_bytes,num_devices,dtype,topology_tier,Duration(us)
1310720,16,DT_BF16,0,512.00
"""


@pytest.fixture
def comm_ref_dir(tmp_path):
    """Separate hccl dir; op_mapping.yaml points to it via communication_data_ref."""
    vllm_dir = tmp_path / "vllm_ascend" / "v0.13.0"
    vllm_dir.mkdir(parents=True)
    hccl_dir = tmp_path / "hccl" / "v8.1.RC1"
    hccl_dir.mkdir(parents=True)
    (vllm_dir / "op_mapping.yaml").write_text(_COMM_REF_OP_MAPPING_WITH_REF)
    (hccl_dir / "hcom_allReduce_.csv").write_text(_COMM_REF_CSV.strip())
    return vllm_dir


@pytest.fixture
def comm_no_ref_dir(tmp_path):
    """Legacy layout: CSV and op_mapping.yaml in the same directory, no communication_data_ref."""
    data_dir = tmp_path / "legacy"
    data_dir.mkdir()
    (data_dir / "op_mapping.yaml").write_text(_COMM_REF_OP_MAPPING_NO_REF)
    (data_dir / "hcom_allReduce_.csv").write_text(_COMM_REF_CSV.strip())
    return data_dir


def test_comm_data_ref_resolves_csv_from_separate_dir(comm_ref_dir):
    """communication_data_ref points to a separate hccl dir; CSV should be found and hit."""
    ds = ProfilingDataSource(comm_ref_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.all_reduce.default,
        [
            torch.empty(4, 1024, 160, device="meta", dtype=torch.bfloat16),
            0,
            list(range(16)),
        ],
    )
    result = ds.lookup(op)
    assert result is not None, f"Expected hit, got miss: {ds.last_miss_reason}"
    assert abs(result.latency_us - 512.00) < 0.01
    assert result.details.get("kernel_type") == "hcom_allReduce_"


def test_comm_data_ref_missing_falls_back_to_data_dir(comm_no_ref_dir):
    """Without communication_data_ref, _comm_data_dir falls back to data_dir (legacy layout)."""
    ds = ProfilingDataSource(comm_no_ref_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.all_reduce.default,
        [
            torch.empty(4, 1024, 160, device="meta", dtype=torch.bfloat16),
            0,
            list(range(16)),
        ],
    )
    result = ds.lookup(op)
    assert result is not None, (
        f"Expected hit in legacy layout, got: {ds.last_miss_reason}"
    )
    assert abs(result.latency_us - 512.00) < 0.01


def test_comm_data_ref_csv_not_found_returns_none(comm_ref_dir):
    """communication_data_ref dir exists but CSV is absent → None + csv_not_found."""
    ds = ProfilingDataSource(comm_ref_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.all_gather.default,  # no hcom_allGather_.csv in hccl dir
        [
            torch.empty(4, 1024, 160, device="meta", dtype=torch.bfloat16),
            0,
            0,
            list(range(16)),
        ],
    )
    # Need all_gather in op_mapping — patch the loaded mapping directly
    ds._op_mapping.setdefault("operator_mappings", {})[
        "tensor_cast.all_gather.default"
    ] = {"kernel_type": "hcom_allGather_", "category": "communication"}
    result = ds.lookup(op)
    assert result is None
    assert ds.last_miss_reason == "csv_not_found"


# --- Comm hccl-preferred priority tests ---

# CSV with different latency to distinguish which directory was used
_COMM_HCCL_CSV = """\
message_bytes,num_devices,dtype,topology_tier,Duration(us)
1310720,16,DT_BF16,0,100.00
"""

_COMM_VLLM_CSV = """\
message_bytes,num_devices,dtype,topology_tier,Duration(us)
1310720,16,DT_BF16,0,999.99
"""

_COMM_PRIORITY_OP_MAPPING = """\
version: "test"
device: TEST_DEVICE
communication_data_ref: "../hccl/"
operator_mappings:
  "tensor_cast.all_reduce.default":
    kernel_type: hcom_allReduce_
    category: communication
  "aten.mm.default":
    kernel_type: MatMulV2
"""

_COMPUTE_CSV = """\
input_shapes,output_shapes,dtype,Average Duration(us)
"[128, 5120];[5120, 5120]","[128, 5120]",DT_BF16,50.0
"""


def _allreduce_op_16dev():
    return _make_op_info(
        torch.ops.tensor_cast.all_reduce.default,
        [
            torch.empty(1310720 // 2, device="meta", dtype=torch.bfloat16),
            0,
            list(range(16)),
        ],
    )


@pytest.fixture
def comm_priority_both_dir(tmp_path):
    """Both data_dir and _comm_data_dir have hcom_allReduce_.csv with different latency."""
    data_dir = tmp_path / "vllm"
    data_dir.mkdir()
    (data_dir / "op_mapping.yaml").write_text(_COMM_PRIORITY_OP_MAPPING)
    (data_dir / "hcom_allReduce_.csv").write_text(_COMM_VLLM_CSV.strip())
    hccl_dir = tmp_path / "hccl"
    hccl_dir.mkdir()
    (hccl_dir / "hcom_allReduce_.csv").write_text(_COMM_HCCL_CSV.strip())
    return data_dir


@pytest.fixture
def comm_priority_hccl_only(tmp_path):
    """Only _comm_data_dir has the CSV."""
    data_dir = tmp_path / "vllm"
    data_dir.mkdir()
    (data_dir / "op_mapping.yaml").write_text(_COMM_PRIORITY_OP_MAPPING)
    hccl_dir = tmp_path / "hccl"
    hccl_dir.mkdir()
    (hccl_dir / "hcom_allReduce_.csv").write_text(_COMM_HCCL_CSV.strip())
    return data_dir


@pytest.fixture
def comm_priority_vllm_only(tmp_path):
    """Only data_dir has the CSV; _comm_data_dir exists but is empty."""
    data_dir = tmp_path / "vllm"
    data_dir.mkdir()
    (data_dir / "op_mapping.yaml").write_text(_COMM_PRIORITY_OP_MAPPING)
    (data_dir / "hcom_allReduce_.csv").write_text(_COMM_VLLM_CSV.strip())
    hccl_dir = tmp_path / "hccl"
    hccl_dir.mkdir()
    return data_dir


@pytest.fixture
def comm_priority_none_dir(tmp_path):
    """Neither directory has the CSV."""
    data_dir = tmp_path / "vllm"
    data_dir.mkdir()
    (data_dir / "op_mapping.yaml").write_text(_COMM_PRIORITY_OP_MAPPING)
    hccl_dir = tmp_path / "hccl"
    hccl_dir.mkdir()
    return data_dir


@pytest.fixture
def comm_priority_compute_dir(tmp_path):
    """data_dir has compute CSV + hccl dir has comm CSV; verify non-comm unaffected."""
    data_dir = tmp_path / "vllm"
    data_dir.mkdir()
    (data_dir / "op_mapping.yaml").write_text(_COMM_PRIORITY_OP_MAPPING)
    (data_dir / "MatMulV2.csv").write_text(_COMPUTE_CSV.strip())
    hccl_dir = tmp_path / "hccl"
    hccl_dir.mkdir()
    return data_dir


def test_comm_hccl_preferred_over_data_dir(comm_priority_both_dir):
    """T1: When both dirs have the CSV, _comm_data_dir (hccl) takes precedence."""
    ds = ProfilingDataSource(comm_priority_both_dir)
    result = ds.lookup(_allreduce_op_16dev())
    assert result is not None
    assert abs(result.latency_us - 100.00) < 0.01, (
        f"Should use hccl CSV (100.00), got {result.latency_us}"
    )


def test_comm_hccl_only_hit(comm_priority_hccl_only):
    """T2: CSV only in _comm_data_dir -> normal HIT."""
    ds = ProfilingDataSource(comm_priority_hccl_only)
    result = ds.lookup(_allreduce_op_16dev())
    assert result is not None
    assert abs(result.latency_us - 100.00) < 0.01


def test_comm_fallback_to_data_dir(comm_priority_vllm_only):
    """T3: CSV only in data_dir, _comm_data_dir empty -> fallback HIT."""
    ds = ProfilingDataSource(comm_priority_vllm_only)
    result = ds.lookup(_allreduce_op_16dev())
    assert result is not None
    assert abs(result.latency_us - 999.99) < 0.01, "Should fallback to data_dir CSV"


def test_comm_no_csv_anywhere(comm_priority_none_dir):
    """T4: No CSV in either directory -> None."""
    ds = ProfilingDataSource(comm_priority_none_dir)
    result = ds.lookup(_allreduce_op_16dev())
    assert result is None


def test_non_comm_kernel_ignores_comm_dir(comm_priority_compute_dir):
    """T5: Non-hcom_ kernel (MatMulV2) only queries data_dir, not _comm_data_dir."""
    ds = ProfilingDataSource(comm_priority_compute_dir)
    df = ds._load_csv("MatMulV2")
    assert df is not None, "Non-comm kernel should load from data_dir"
    df2 = ds._load_csv("SomeNonExistent")
    assert df2 is None


def test_comm_allreduce_interpolates_message_bytes(comm_data_dir):
    """When exact message_bytes misses, interpolate between bracketing rows."""
    ds = ProfilingDataSource(comm_data_dir)
    # comm_data_dir has allReduce: 655360→412.50us, 1310720→689.96us (num_devices=16, tier=0)
    # Query 983040 bytes (midpoint): 412.50 + 277.46 * 0.5 = 551.23
    op = _make_op_info(
        torch.ops.tensor_cast.all_reduce.default,
        [
            torch.empty(
                983040 // 2, device="meta", dtype=torch.bfloat16
            ),  # 983040 bytes
            0,
            list(range(16)),
        ],
    )
    result = ds.lookup(op)
    assert result is not None, (
        "Should interpolate comm between bracketing message_bytes"
    )
    assert abs(result.latency_us - 551.23) < 1.0
    assert result.source == QuerySource.INTERPOLATED


def test_comm_allreduce_no_extrapolation(comm_data_dir):
    """When message_bytes is outside CSV range, return None (no extrapolation)."""
    ds = ProfilingDataSource(comm_data_dir)
    # CSV max for num_devices=16, tier=0 is 1310720. Query 2x max → can't bracket
    op = _make_op_info(
        torch.ops.tensor_cast.all_reduce.default,
        [
            torch.empty(2621440 // 2, device="meta", dtype=torch.bfloat16),
            0,
            list(range(16)),
        ],
    )
    result = ds.lookup(op)
    assert result is None, "Should not extrapolate beyond CSV range"


def test_comm_interpolation_latency_dominated_region(tmp_path):
    """In latency-dominated region (small messages), interpolation should
    stay near alpha (startup latency), not linearly ramp toward the next point.

    Real HCCL pattern: latency ≈ 120us for 1KB-4MB, then ramps.
    Naive linear between 1MB (120us) and 16MB (300us) would predict
    166us at 4MB, but actual is ~107us (still latency-dominated).
    """
    data_dir = tmp_path / "alpha_beta"
    data_dir.mkdir()
    (data_dir / "op_mapping.yaml").write_text(COMM_OP_MAPPING_YAML)

    # Mimic real HCCL data with alpha-beta behavior (powers-of-4 spacing)
    csv_content = """\
message_bytes,num_devices,dtype,topology_tier,Duration(us)
1024,16,DT_BF16,1,120.0
4096,16,DT_BF16,1,120.0
16384,16,DT_BF16,1,120.0
65536,16,DT_BF16,1,120.5
1048576,16,DT_BF16,1,130.0
16777216,16,DT_BF16,1,288.0
67108864,16,DT_BF16,1,791.0
268435456,16,DT_BF16,1,2804.0
"""
    (data_dir / "hcom_allReduce_.csv").write_text(csv_content.strip())

    ds = ProfilingDataSource(data_dir)

    # Query 4MB (4194304) — between 1MB (130.0us) and 16MB (288.0us)
    # Alpha-beta model: alpha≈120, beta≈100GB/s → 120 + 4194304/100000 ≈ 162us
    # Naive linear: 130.0 + (288.0-130.0) * (4194304-1048576)/(16777216-1048576) = 161.6us
    # Both happen to give ~162us here (acceptable)
    #
    # Better test: 160KB (163840) — between 64KB (120.5us) and 1MB (130.0us)
    # This is the actual Qwen3 allReduce message size!
    # Naive linear: 120.5 + (130.0-120.5) * (163840-65536)/(1048576-65536) = 121.5us
    # Alpha-beta:   120 + 163840/100000 = 121.6us (close, because bracket is tight)
    op = _make_op_info(
        torch.ops.tensor_cast.all_reduce.default,
        [
            torch.empty(
                163840 // 2, device="meta", dtype=torch.bfloat16
            ),  # 163840 bytes
            0,
            list(range(16)),
        ],
    )
    result = ds.lookup(op)
    assert result is not None
    assert result.source == QuerySource.INTERPOLATED
    # Should be in [120.5, 130.0] range and close to alpha (~120-122us)
    assert 120.0 <= result.latency_us <= 125.0, (
        f"Interpolated {result.latency_us:.1f}us: latency-dominated region "
        f"should stay near alpha (~120us), not ramp toward 130us"
    )


def test_comm_allreduce_exact_still_measured(comm_data_dir):
    """Exact message_bytes match should return MEASURED, not INTERPOLATED."""
    ds = ProfilingDataSource(comm_data_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.all_reduce.default,
        [
            torch.empty(
                1, 640, 1024, device="meta", dtype=torch.bfloat16
            ),  # 1310720 bytes
            0,
            list(range(16)),
        ],
    )
    result = ds.lookup(op)
    assert result is not None
    assert result.source == QuerySource.MEASURED
    assert abs(result.latency_us - 689.96) < 0.01


# --- MoE tc_input_count tests ---

MOE_OP_MAPPING_YAML = """\
version: "0.14.0"
device: TEST_DEVICE

operator_mappings:
  "tensor_cast.init_routing_v2.default":
    kernel_type: MoeTokenPermute
    tc_input_count: 2
  "tensor_cast.unpermute_tokens.default":
    kernel_type: MoeTokenUnpermute
    tc_input_count: 1
"""

# MoE permute CSV (simulates MoeTokenPermute.csv raw profiling format)
MOE_PERMUTE_CSV = """\
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Average Duration(us)
"4,7168;4,8","DT_BF16;INT32","ND;ND","32,7168","DT_BF16","ND",6.12
"19,1;19","FLOAT;INT32","ND;ND","19,1;19","FLOAT","ND;ND",11.39
"""

# MoE unpermute CSV (simulates MoeTokenUnpermute.csv, contains NPU internal params)
MOE_UNPERMUTE_CSV = """\
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Average Duration(us)
"128,7168;128;","DT_BF16;INT32;DT_UNDEFINED","ND;ND;NULL","128,7168","DT_BF16","ND",7.01
"256,7168;256;4,8","DT_BF16;INT32;DT_BF16","ND;ND;ND","4,7168","DT_BF16","ND",6.02
"""


@pytest.fixture
def moe_data_dir(tmp_path):
    d = tmp_path / "moe"
    d.mkdir()
    (d / "op_mapping.yaml").write_text(MOE_OP_MAPPING_YAML)
    (d / "MoeTokenPermute.csv").write_text(MOE_PERMUTE_CSV.strip())
    (d / "MoeTokenUnpermute.csv").write_text(MOE_UNPERMUTE_CSV.strip())
    return d


def test_moe_permute_hit(moe_data_dir):
    """init_routing_v2 (4,7168)+(4,8) matches first CSV row -> 6.12 us."""
    ds = ProfilingDataSource(moe_data_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.init_routing_v2.default,
        [
            torch.empty(4, 7168, device="meta", dtype=torch.bfloat16),
            torch.empty(4, 8, device="meta", dtype=torch.int32),
        ],
    )
    result = ds.lookup(op)
    assert result is not None
    assert abs(result.latency_us - 6.12) < 0.01


def test_moe_permute_dtype_filter(moe_data_dir):
    """BF16 inputs should not match the FLOAT row."""
    ds = ProfilingDataSource(moe_data_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.init_routing_v2.default,
        [
            torch.empty(19, 1, device="meta", dtype=torch.bfloat16),
            torch.empty(19, device="meta", dtype=torch.int32),
        ],
    )
    result = ds.lookup(op)
    assert result is None


def test_moe_permute_shape_miss(moe_data_dir):
    """(5,7168)+(5,8) has no matching row -> None."""
    ds = ProfilingDataSource(moe_data_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.init_routing_v2.default,
        [
            torch.empty(5, 7168, device="meta", dtype=torch.bfloat16),
            torch.empty(5, 8, device="meta", dtype=torch.int32),
        ],
    )
    result = ds.lookup(op)
    assert result is None


def test_moe_unpermute_hit_tc_input_count_1(moe_data_dir):
    """tc_input_count=1: only first TC input (128,7168) compared -> 7.01 us."""
    ds = ProfilingDataSource(moe_data_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.unpermute_tokens.default,
        [
            torch.empty(128, 7168, device="meta", dtype=torch.bfloat16),
        ],
    )
    result = ds.lookup(op)
    assert result is not None
    assert abs(result.latency_us - 7.01) < 0.01


def test_moe_unpermute_hit_with_extra_csv_inputs(moe_data_dir):
    """tc_input_count=1: (256,7168) matches second row (which has 3 CSV inputs) -> 6.02 us."""
    ds = ProfilingDataSource(moe_data_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.unpermute_tokens.default,
        [
            torch.empty(256, 7168, device="meta", dtype=torch.bfloat16),
        ],
    )
    result = ds.lookup(op)
    assert result is not None
    assert abs(result.latency_us - 6.02) < 0.01


def test_moe_unpermute_shape_miss(moe_data_dir):
    """(512,7168) has no matching row -> None."""
    ds = ProfilingDataSource(moe_data_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.unpermute_tokens.default,
        [
            torch.empty(512, 7168, device="meta", dtype=torch.bfloat16),
        ],
    )
    result = ds.lookup(op)
    assert result is None


def test_kernel_type_equals_csv_filename(moe_data_dir):
    """kernel_type is used directly as CSV filename (convention: kernel_type == CSV filename)."""
    ds = ProfilingDataSource(moe_data_dir)
    assert (moe_data_dir / "MoeTokenPermute.csv").exists()
    op = _make_op_info(
        torch.ops.tensor_cast.init_routing_v2.default,
        [
            torch.empty(4, 7168, device="meta", dtype=torch.bfloat16),
            torch.empty(4, 8, device="meta", dtype=torch.int32),
        ],
    )
    result = ds.lookup(op)
    assert result is not None
    assert abs(result.latency_us - 6.12) < 0.01


# --- Integration tests: real CANN 8.3 / 8.5 data directories ---

from pathlib import Path  # noqa: E402

_CANN83_DATA_DIR = Path(__file__).resolve().parents[2] / (
    "tensor_cast/performance_model/profiling_database/data/"
    "ATLAS_800_A3_752T_128G_DIE/vllm_ascend/vllm0.13.0_torch2.8.0_cann8.3"
)
_CANN85_DATA_DIR = Path(__file__).resolve().parents[2] / (
    "tensor_cast/performance_model/profiling_database/data/"
    "ATLAS_800_A3_752T_128G_DIE/vllm_ascend/vllm0.15.0_torch2.9.0_cann8.5"
)

_skip_no_cann83 = pytest.mark.skipif(
    not _CANN83_DATA_DIR.exists(), reason="CANN 8.3 data dir not present"
)
_skip_no_cann85 = pytest.mark.skipif(
    not _CANN85_DATA_DIR.exists(), reason="CANN 8.5 data dir not present"
)


# MoE real CANN tests (init_routing_v2, unpermute_tokens, moe_gating_top_k_softmax)
# moved to G1 PR — they depend on tensor_cast.ops.fused_moe which is G1 scope.


MOE_GATING_TOPK_OP_MAPPING = """
version: "test"
device: TEST_DEVICE

operator_mappings:
  "tensor_cast.moe_gating_top_k_softmax.default":
    kernel_type: MoeGatingTopK
"""

MOE_GATING_TOPK_CSV = """\
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Average Duration(us)
"8,256","FLOAT","ND","8,8;8,8","FLOAT;INT64","ND;ND",12.34
"""


@pytest.fixture
def moe_gating_topk_data_dir(tmp_path):
    data_dir = tmp_path / "moe_gating_topk"
    data_dir.mkdir()
    (data_dir / "op_mapping.yaml").write_text(MOE_GATING_TOPK_OP_MAPPING)
    (data_dir / "MoeGatingTopK.csv").write_text(MOE_GATING_TOPK_CSV.strip())
    return data_dir


def test_moe_gating_top_k_softmax_lookup_exact_match(moe_gating_topk_data_dir):
    ds = ProfilingDataSource(moe_gating_topk_data_dir)
    op = _make_op_info(
        _FakeTorchOp("tensor_cast.moe_gating_top_k_softmax.default"),
        [
            torch.empty(8, 256, device="meta", dtype=torch.float32),
            8,
        ],
    )
    result = ds.lookup(op)
    assert result is not None
    assert abs(result.latency_us - 12.34) < 0.01
    assert result.source == QuerySource.MEASURED
    assert result.details.get("kernel_type") == "MoeGatingTopK"


def test_moe_gating_top_k_softmax_lookup_shape_miss(moe_gating_topk_data_dir):
    ds = ProfilingDataSource(moe_gating_topk_data_dir)
    op = _make_op_info(
        _FakeTorchOp("tensor_cast.moe_gating_top_k_softmax.default"),
        [
            torch.empty(17, 256, device="meta", dtype=torch.float32),
            8,
        ],
    )
    result = ds.lookup(op)
    assert result is None
    assert ds.last_miss_reason == "shape_mismatch"


# --- C1: MLA/MLAPO unblock tests ---


def test_mlapo_composite_not_rejected():
    """After C1 fix, MLAPO ops should attempt composite lookup, not return mla_not_implemented."""
    import os
    import tempfile

    import yaml

    op_mapping = {
        "version": "test",
        "device": "TEST",
        "operator_mappings": {
            "tensor_cast.mlapo.default": {
                "composite": True,
                "sub_kernels": ["MatMulV2", "KvRmsNormRopeCache"],
            }
        },
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "op_mapping.yaml"), "w") as f:
            yaml.dump(op_mapping, f)

        ds = ProfilingDataSource(tmpdir, device_profile=MagicMock())

        mock_op = MagicMock()
        mock_op.func = "torch.ops.tensor_cast.mlapo.default"
        mock_op.args = [
            torch.randn(8, 576),
            torch.randn(576, 512),
        ]

        ds.lookup(mock_op)

        # Result may be None (CSV missing), but reason should NOT be mla_not_implemented
        assert ds.last_miss_reason != "mla_not_implemented", (
            f"Expected composite lookup attempt, got {ds.last_miss_reason}"
        )


# --- C4: MISS reason reclassification tests ---


def test_miss_reason_respects_tc_input_count():
    """With tc_input_count, miss reason should compare truncated counts."""
    import os
    import tempfile

    import pandas as pd
    import yaml

    op_mapping = {
        "version": "test",
        "device": "TEST",
        "operator_mappings": {
            "tensor_cast.quantize.default": {
                "kernel_type": "AscendQuantV2",
                "tc_input_count": 1,
            }
        },
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "op_mapping.yaml"), "w") as f:
            yaml.dump(op_mapping, f)

        # CSV with 1-input shape that doesn't match TC shape
        csv_data = pd.DataFrame(
            {
                "Input Shapes": ['"999,888"'],
                "Input Data Types": ["DT_BF16"],
                "Input Formats": ["ND"],
                "Output Shapes": ['"999,888"'],
                "Output Data Types": ["DT_BF16"],
                "Output Formats": ["ND"],
                "AVG_DURATION_US": [10.0],
            }
        )
        csv_data.to_csv(os.path.join(tmpdir, "AscendQuantV2.csv"), index=False)

        ds = ProfilingDataSource(tmpdir, device_profile=MagicMock())

        mock_op = MagicMock()
        mock_op.func = "torch.ops.tensor_cast.quantize.default"
        # TC has 3 inputs but tc_input_count=1, so only first is compared
        mock_op.args = [
            torch.randn(128, 5120),  # tensor (different from CSV 999,888)
            torch.randn(5120),  # scale (ignored by tc_input_count)
            torch.randn(5120),  # zero_point (ignored by tc_input_count)
        ]

        result = ds.lookup(mock_op)
        assert result is None  # should miss
        # Key assertion: reason should be shape_mismatch, NOT input_count_mismatch
        assert ds.last_miss_reason == "shape_mismatch", (
            f"Expected shape_mismatch, got {ds.last_miss_reason}"
        )


# --- Flatten batch 3D→2D tests (quantize / norm kernels) ---

QUANT_2D_CSV = """\
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Average Duration(us)
"16,7168","DT_BF16","ND","16,7168","INT8","ND",5.5
"256,5120","DT_BF16","ND","256,5120","INT8","ND",18.2
"""


@pytest.fixture
def quant_flatten_data_dir(tmp_path):
    data_dir = tmp_path / "quant_flatten"
    data_dir.mkdir()
    op_mapping = (
        'version: "test"\n'
        "operator_mappings:\n"
        '  "tensor_cast.quantize.default":\n'
        "    kernel_type: AscendQuantV2\n"
        "    tc_input_count: 1\n"
    )
    (data_dir / "op_mapping.yaml").write_text(op_mapping)
    (data_dir / "AscendQuantV2.csv").write_text(QUANT_2D_CSV.strip())
    return data_dir


def test_flatten_batch_quantize_3d_to_2d(quant_flatten_data_dir):
    """TC quantize sends (1,16,7168) 3D — should match CSV (16,7168) 2D
    via flatten batch rule for AscendQuantV2."""
    ds = ProfilingDataSource(quant_flatten_data_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.quantize.default,
        [
            torch.empty(1, 16, 7168, device="meta", dtype=torch.bfloat16),
            torch.empty(7168, device="meta", dtype=torch.bfloat16),  # scale
            torch.empty(7168, device="meta", dtype=torch.bfloat16),  # zp
        ],
    )
    result = ds.lookup(op)
    assert result is not None, (
        "Should match 3D (1,16,7168) → 2D (16,7168) via flatten batch"
    )
    assert abs(result.latency_us - 5.5) < 0.01
    assert result.shape_match_info is not None
    assert result.shape_match_info.shape_match_rule == "batch_strip"


def test_flatten_batch_quantize_batch_gt_1(quant_flatten_data_dir):
    """TC quantize sends (4,64,5120) 3D — should match CSV (256,5120) 2D
    via flatten: 4*64=256."""
    ds = ProfilingDataSource(quant_flatten_data_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.quantize.default,
        [
            torch.empty(4, 64, 5120, device="meta", dtype=torch.bfloat16),
            torch.empty(5120, device="meta", dtype=torch.bfloat16),
            torch.empty(5120, device="meta", dtype=torch.bfloat16),
        ],
    )
    result = ds.lookup(op)
    assert result is not None, (
        "Should match 3D (4,64,5120) → 2D (256,5120) via flatten batch"
    )
    assert abs(result.latency_us - 18.2) < 0.01
    assert result.shape_match_info is not None
    assert result.shape_match_info.shape_match_rule == "flatten_3d"


def test_flatten_batch_quantize_with_padding(quant_flatten_data_dir):
    """TC quantize sends (1,272,5120) 3D — should match CSV (256,5120) 2D
    via flatten + block padding: flatten→(272,5120), 272 ≈ 256 via ceil(256/16)*16=256? No.
    Actually 272 = ceil(256/16)*16 = 256? No, ceil(256/16)*16 = 256. 272 = ceil(268/16)*16.
    Use (1,256,5120) instead for exact flatten match."""
    ds = ProfilingDataSource(quant_flatten_data_dir)
    # 3D exact flatten (no padding needed)
    op = _make_op_info(
        torch.ops.tensor_cast.quantize.default,
        [
            torch.empty(1, 256, 5120, device="meta", dtype=torch.bfloat16),
            torch.empty(5120, device="meta", dtype=torch.bfloat16),
            torch.empty(5120, device="meta", dtype=torch.bfloat16),
        ],
    )
    result = ds.lookup(op)
    assert result is not None, (
        "Should match 3D (1,256,5120) → 2D (256,5120) via flatten"
    )


def test_flatten_batch_rmsnorm_3d_to_2d(rmsnorm_data_dir):
    """TC RmsNorm sends (2,68,5120),(5120,) 3D — should match CSV (136,5120),(5120)
    via flatten batch: 2*68=136."""
    ds = ProfilingDataSource(rmsnorm_data_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.rms_norm.default,
        [
            torch.empty(2, 68, 5120, device="meta", dtype=torch.bfloat16),
            torch.empty(5120, device="meta", dtype=torch.bfloat16),
        ],
    )
    result = ds.lookup(op)
    assert result is not None, (
        "Should match 3D (2,68,5120) → 2D (136,5120) via flatten batch"
    )
    assert abs(result.latency_us - 21.66) < 0.01


def test_flatten_batch_not_applied_to_matmul(sample_data_dir):
    """MatMulV2 is NOT in _FLATTEN_BATCH_KERNELS — 3D should NOT match 2D."""
    ds = ProfilingDataSource(sample_data_dir)
    # CSV has (136,5120) as first input. Try 3D (2,68,5120) — should NOT match.
    op = _make_op_info(
        torch.ops.aten.mm.default,
        [
            torch.empty(2, 68, 5120, device="meta", dtype=torch.bfloat16),
            torch.empty(5120, 768, device="meta", dtype=torch.bfloat16),
        ],
    )
    result = ds.lookup(op)
    assert result is None, "Flatten batch should NOT apply to MatMulV2"


def test_flatten_batch_2d_still_works(quant_flatten_data_dir):
    """2D TC shape should still match 2D CSV directly (no flatten needed)."""
    ds = ProfilingDataSource(quant_flatten_data_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.quantize.default,
        [
            torch.empty(16, 7168, device="meta", dtype=torch.bfloat16),
            torch.empty(7168, device="meta", dtype=torch.bfloat16),
            torch.empty(7168, device="meta", dtype=torch.bfloat16),
        ],
    )
    result = ds.lookup(op)
    assert result is not None, "2D exact match should still work"
    assert abs(result.latency_us - 5.5) < 0.01


# --- Merge-last-dims tests (MLA quantize 3D→2D) ---

QUANT_MLA_CSV = """\
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Average Duration(us)
"8,2048","DT_BF16","ND","8,2048","INT8","ND",9.8
"256,2048","DT_BF16","ND","256,2048","INT8","ND",20.5
"""


@pytest.fixture
def quant_mla_data_dir(tmp_path):
    data_dir = tmp_path / "quant_mla"
    data_dir.mkdir()
    op_mapping = (
        'version: "test"\n'
        "operator_mappings:\n"
        '  "tensor_cast.quantize.default":\n'
        "    kernel_type: AscendQuantV2\n"
        "    tc_input_count: 1\n"
    )
    (data_dir / "op_mapping.yaml").write_text(op_mapping)
    (data_dir / "AscendQuantV2.csv").write_text(QUANT_MLA_CSV.strip())
    return data_dir


def test_merge_last_dims_quantize_mla_decode(quant_mla_data_dir):
    """MLA quantize: TC (8, 16, 128) 3D → should match CSV (8, 2048) 2D
    by merging last two dims: 16*128=2048."""
    ds = ProfilingDataSource(quant_mla_data_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.quantize.default,
        [
            torch.empty(8, 16, 128, device="meta", dtype=torch.bfloat16),
            torch.tensor(1.0),
            None,
        ],
    )
    result = ds.lookup(op)
    assert result is not None, "Should match via last-two-dims merge"
    assert abs(result.latency_us - 9.8) < 0.01


def test_merge_last_dims_quantize_mla_prefill(quant_mla_data_dir):
    """MLA quantize: TC (256, 16, 128) 3D → should match CSV (256, 2048) 2D."""
    ds = ProfilingDataSource(quant_mla_data_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.quantize.default,
        [
            torch.empty(256, 16, 128, device="meta", dtype=torch.bfloat16),
            torch.tensor(1.0),
            None,
        ],
    )
    result = ds.lookup(op)
    assert result is not None, "Should match via last-two-dims merge"
    assert abs(result.latency_us - 20.5) < 0.01


def test_merge_last_dims_quantize_mla_batch1(quant_mla_data_dir):
    """MLA quantize batch=1: TC (1, 16, 128) 3D → should match CSV (1, 2048) 2D.
    _strip_batch_dim collapses (1,16,128)→(16,128), so merge must use original shape."""
    # Add a batch=1 row to CSV
    csv_path = quant_mla_data_dir / "AscendQuantV2.csv"
    with open(csv_path, "a") as f:
        f.write('\n"1,2048","DT_BF16","ND","1,2048","INT8","ND",4.2\n')
    ds = ProfilingDataSource(quant_mla_data_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.quantize.default,
        [
            torch.empty(1, 16, 128, device="meta", dtype=torch.bfloat16),
            torch.tensor(1.0),
            None,
        ],
    )
    result = ds.lookup(op)
    assert result is not None, (
        "Should match (1,16,128) → (1,2048) via merge-last-dims on original shape"
    )
    assert abs(result.latency_us - 4.2) < 0.01


def test_merge_last_dims_not_applied_to_matmul(sample_data_dir):
    """MatMulV2 is NOT in _FLATTEN_BATCH_KERNELS — merge should NOT apply.
    Uses sample_data_dir which has MatMulV2 mapped via aten.mm.default."""
    ds = ProfilingDataSource(sample_data_dir)
    # CSV has (136,5120) for MatMulV2. Try 3D (2,68,5120) — should NOT match
    # via merge-last-dims (68*5120=348160 ≠ 5120).
    op = _make_op_info(
        torch.ops.aten.mm.default,
        [
            torch.empty(2, 68, 5120, device="meta", dtype=torch.bfloat16),
            torch.empty(5120, 768, device="meta", dtype=torch.bfloat16),
        ],
    )
    result = ds.lookup(op)
    assert result is None, "Merge last dims should NOT apply to MatMulV2"


# --- Elementwise output-shape matching tests ---

ELEMENTWISE_OP_MAPPING_YAML = """
version: "test"
operator_mappings:
  "aten.mul.Tensor":
    kernel_type: Mul
    query_mode: elementwise
"""

# CSV where input shapes differ from typical TC broadcast patterns,
# but output shapes are the matching key for elementwise lookup.
ELEMENTWISE_MUL_CSV = """\
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Average Duration(us)
"256,7168;256,7168","DT_BF16;DT_BF16","ND;ND","256,7168","DT_BF16","ND",10.5
"512,7168;512,7168","DT_BF16;DT_BF16","ND;ND","512,7168","DT_BF16","ND",20.0
"""


@pytest.fixture
def elementwise_data_dir(tmp_path):
    data_dir = tmp_path / "elementwise"
    data_dir.mkdir()
    (data_dir / "op_mapping.yaml").write_text(ELEMENTWISE_OP_MAPPING_YAML)
    (data_dir / "Mul.csv").write_text(ELEMENTWISE_MUL_CSV.strip())
    return data_dir


def test_elementwise_exact_match_same_dtype(elementwise_data_dir):
    """BF16 tensor out=(256,7168), CSV has BF16 output (256,7168) -> exact HIT."""
    ds = ProfilingDataSource(elementwise_data_dir)
    out_tensor = torch.empty(256, 7168, device="meta", dtype=torch.bfloat16)
    op = _make_op_info(
        torch.ops.aten.mul.Tensor,
        [
            torch.empty(256, 7168, device="meta", dtype=torch.bfloat16),
            torch.empty(256, 7168, device="meta", dtype=torch.bfloat16),
        ],
        [out_tensor],
    )
    result = ds.lookup(op)
    assert result is not None, "Should match elementwise on output shape"
    assert abs(result.latency_us - 10.5) < 0.01
    assert result.confidence == 1.0
    assert result.source == QuerySource.MEASURED


def test_elementwise_dtype_scaled_match(elementwise_data_dir):
    """FP32 tensor out=(256,7168), CSV has BF16 output (256,7168) -> HIT with latency * 2.0."""
    ds = ProfilingDataSource(elementwise_data_dir)
    out_tensor = torch.empty(256, 7168, device="meta", dtype=torch.float32)
    op = _make_op_info(
        torch.ops.aten.mul.Tensor,
        [
            torch.empty(256, 7168, device="meta", dtype=torch.float32),
            torch.empty(256, 7168, device="meta", dtype=torch.float32),
        ],
        [out_tensor],
    )
    result = ds.lookup(op)
    assert result is not None, "Should match elementwise with dtype scaling"
    # FP32=4 bytes, BF16=2 bytes -> scale = 4/2 = 2.0; 10.5 * 2.0 = 21.0
    assert abs(result.latency_us - 21.0) < 0.01
    assert result.confidence == 0.9
    assert result.details == {
        "kernel_type": "Mul",
        "query_mode": "elementwise",
        "dtype_scale": 2.0,
    }
    assert result.shape_match_info is not None
    assert result.shape_match_info.simulation_shapes == [[256, 7168]]
    assert result.shape_match_info.kernel_shapes == [[256, 7168]]
    assert result.shape_match_info.shape_match_rule == "elementwise"


def test_elementwise_broadcast_ignored(elementwise_data_dir):
    """Scalar mul where TC has 1 tensor + scalar, out=(256,7168) -> HIT on output shape.
    Elementwise lookup matches on output shape, not input pattern."""
    ds = ProfilingDataSource(elementwise_data_dir)
    out_tensor = torch.empty(256, 7168, device="meta", dtype=torch.bfloat16)
    op = _make_op_info(
        torch.ops.aten.mul.Tensor,
        [
            torch.empty(256, 7168, device="meta", dtype=torch.bfloat16),
            # Second arg is a scalar (not a tensor) — broadcast
            3.14,
        ],
        [out_tensor],
    )
    result = ds.lookup(op)
    assert result is not None, (
        "Should match elementwise on output shape regardless of inputs"
    )
    assert abs(result.latency_us - 10.5) < 0.01


def test_elementwise_miss_no_shape(elementwise_data_dir):
    """out=(512,4096) not in CSV -> MISS with elementwise_output_shape_mismatch."""
    ds = ProfilingDataSource(elementwise_data_dir)
    out_tensor = torch.empty(512, 4096, device="meta", dtype=torch.bfloat16)
    op = _make_op_info(
        torch.ops.aten.mul.Tensor,
        [
            torch.empty(512, 4096, device="meta", dtype=torch.bfloat16),
            torch.empty(512, 4096, device="meta", dtype=torch.bfloat16),
        ],
        [out_tensor],
    )
    result = ds.lookup(op)
    assert result is None, "No CSV row with output (512,4096)"
    assert ds.last_miss_reason == "elementwise_output_shape_mismatch"


def test_elementwise_fallback_no_output(elementwise_data_dir):
    """op_invoke_info.out = None -> falls back to _lookup_compute (returns None
    since _lookup_compute uses input-shape matching and the CSV input shapes
    happen to match, but the key test is the fallback path, not the result)."""
    ds = ProfilingDataSource(elementwise_data_dir)
    op = _make_op_info(
        torch.ops.aten.mul.Tensor,
        [
            # Use shapes that DON'T match any CSV input shapes to ensure
            # _lookup_compute also returns None, proving the fallback happened.
            torch.empty(999, 7168, device="meta", dtype=torch.bfloat16),
            torch.empty(999, 7168, device="meta", dtype=torch.bfloat16),
        ],
    )
    # out is None (no output_tensors passed)
    assert op.out is None
    result = ds.lookup(op)
    # Falls back to _lookup_compute which returns None (no input shape match)
    assert result is None
    # Miss reason should be from _lookup_compute, NOT elementwise
    assert ds.last_miss_reason == "shape_mismatch"


def test_elementwise_fallback_tuple_output(elementwise_data_dir):
    """op_invoke_info.out = (tensor, tensor2) -> unwraps to first element, matches."""
    ds = ProfilingDataSource(elementwise_data_dir)
    out_tensor1 = torch.empty(256, 7168, device="meta", dtype=torch.bfloat16)
    out_tensor2 = torch.empty(10, device="meta", dtype=torch.bfloat16)
    op = _make_op_info(
        torch.ops.aten.mul.Tensor,
        [
            torch.empty(256, 7168, device="meta", dtype=torch.bfloat16),
            torch.empty(256, 7168, device="meta", dtype=torch.bfloat16),
        ],
        [out_tensor1, out_tensor2],
    )
    # _make_op_info with 2 outputs creates a tuple
    assert isinstance(op.out, tuple)
    result = ds.lookup(op)
    assert result is not None, "Should unwrap tuple output and match on first element"
    assert abs(result.latency_us - 10.5) < 0.01


# --- _find_compute_match tests (replaces _query_by_shapes) ---

QUERY_BY_SHAPES_OP_MAPPING = """
version: "test"
device: TEST_DEVICE

operator_mappings:
  "aten.mm.default":
    kernel_type: MatMulV2
    alternate_kernel_types: [MatMulV3]
"""

QUERY_BY_SHAPES_MATMULV2_CSV = """\
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Average Duration(us)
"136,5120;5120,768","DT_BF16;DT_BF16","ND;ND","136,768","DT_BF16","ND",45.3
"""

QUERY_BY_SHAPES_MATMULV3_CSV = """\
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Average Duration(us)
"256,5120;5120,768","DT_BF16;DT_BF16","ND;ND","256,768","DT_BF16","ND",55.0
"""

# CSV with 4 inputs (like QuantBatchMatmulV3: activation, FRACTAL_NZ weight, bias, bias)
QUERY_BY_SHAPES_QBMV3_CSV = """\
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Average Duration(us)
"4099,7168;66,448,16,32;2112;2112","DT_BF16;DT_BF16;DT_BF16;DT_BF16","ND;FRACTAL_NZ;ND;ND","4099,2112","DT_BF16","ND",100.5
"""


@pytest.fixture
def query_shapes_data_dir(tmp_path):
    d = tmp_path / "qbs"
    d.mkdir()
    (d / "op_mapping.yaml").write_text(QUERY_BY_SHAPES_OP_MAPPING)
    (d / "MatMulV2.csv").write_text(QUERY_BY_SHAPES_MATMULV2_CSV.strip())
    (d / "MatMulV3.csv").write_text(QUERY_BY_SHAPES_MATMULV3_CSV.strip())
    (d / "QuantBatchMatmulV3.csv").write_text(QUERY_BY_SHAPES_QBMV3_CSV.strip())
    return d


class TestFindComputeMatch:
    def test_primary_kernel_hit(self, query_shapes_data_dir):
        ds = ProfilingDataSource(query_shapes_data_dir)
        tc_inputs = [
            ((136, 5120), torch.bfloat16),
            ((5120, 768), torch.bfloat16),
        ]
        hit = ds._find_compute_match(["MatMulV2"], tc_inputs)
        assert hit is not None
        assert abs(hit.latency_us - 45.3) < 0.01
        assert hit.kernel_type == "MatMulV2"

    def test_alternate_kernel_fallback(self, query_shapes_data_dir):
        """Primary misses, alternate hits."""
        ds = ProfilingDataSource(query_shapes_data_dir)
        tc_inputs = [
            ((256, 5120), torch.bfloat16),
            ((5120, 768), torch.bfloat16),
        ]
        hit = ds._find_compute_match(["MatMulV2", "MatMulV3"], tc_inputs)
        assert hit is not None
        assert abs(hit.latency_us - 55.0) < 0.01
        assert hit.kernel_type == "MatMulV3"

    def test_all_miss_returns_none(self, query_shapes_data_dir):
        ds = ProfilingDataSource(query_shapes_data_dir)
        tc_inputs = [
            ((999, 5120), torch.bfloat16),
            ((5120, 768), torch.bfloat16),
        ]
        hit = ds._find_compute_match(["MatMulV2", "MatMulV3"], tc_inputs)
        assert hit is None

    def test_tc_input_count_truncates_csv(self, query_shapes_data_dir):
        """tc_input_count=2 allows matching CSV with 4 inputs using only first 2."""
        ds = ProfilingDataSource(query_shapes_data_dir)
        tc_inputs = [
            ((4099, 7168), torch.bfloat16),
            ((2112, 7168), torch.bfloat16),
        ]
        hit = ds._find_compute_match(
            ["QuantBatchMatmulV3"], tc_inputs, tc_input_count=2
        )
        assert hit is not None
        assert abs(hit.latency_us - 100.5) < 0.01
        assert hit.kernel_type == "QuantBatchMatmulV3"

    def test_without_tc_input_count_auto_truncates_csv(self, query_shapes_data_dir):
        """Without tc_input_count, auto_truncate=True truncates CSV to len(tc_inputs).

        Fix R1: _find_compute_match auto-truncates CSV comparison to len(tc_inputs)
        when tc_input_count is None and auto_truncate=True.
        """
        ds = ProfilingDataSource(query_shapes_data_dir)
        tc_inputs = [
            ((4099, 7168), torch.bfloat16),
            ((2112, 7168), torch.bfloat16),
        ]
        hit = ds._find_compute_match(
            ["QuantBatchMatmulV3"], tc_inputs, auto_truncate=True
        )
        assert hit is not None
        assert abs(hit.latency_us - 100.5) < 0.01
        assert hit.kernel_type == "QuantBatchMatmulV3"


# --- Composite partial match tests ---

PARTIAL_MATCH_OP_MAPPING = """
version: "test"
device: TEST_DEVICE

operator_mappings:
  "tensor_cast.mlapo_quant.default":
    composite: true
    sub_kernels: [QuantBatchMatmulV3, KvRmsNormRopeCache]
"""

PARTIAL_MATCH_QBMV3_CSV = """\
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Average Duration(us)
"4099,7168;2112,7168","DT_BF16;DT_BF16","ND;ND","4099,2112","DT_BF16","ND",80.0
"""


@pytest.fixture
def partial_match_data_dir(tmp_path):
    d = tmp_path / "partial"
    d.mkdir()
    (d / "op_mapping.yaml").write_text(PARTIAL_MATCH_OP_MAPPING)
    (d / "QuantBatchMatmulV3.csv").write_text(PARTIAL_MATCH_QBMV3_CSV.strip())
    # No KvRmsNormRopeCache.csv — will MISS
    return d


class TestCompositePartialMatch:
    def test_public_lookup_partial_preserves_shape_debug(self, tmp_path, monkeypatch):
        data_dir = tmp_path / "partial_public"
        data_dir.mkdir()
        (data_dir / "op_mapping.yaml").write_text(
            'version: "test"\n'
            "device: TEST_DEVICE\n\n"
            "operator_mappings:\n"
            '  "tensor_cast.fake_partial_debug":\n'
            "    composite: true\n"
        )
        (data_dir / "PadKernel.csv").write_text(
            "Input Shapes,Input Data Types,Input Formats,Output Shapes,"
            "Output Data Types,Output Formats,Average Duration(us)\n"
            '"120,5120","DT_BF16","ND","120,5120","DT_BF16","ND",80.0\n'
        )
        ds = ProfilingDataSource(data_dir)

        monkeypatch.setitem(
            COMPOSITE_DECOMPOSERS,
            "tensor_cast.fake_partial_debug",
            lambda _op, _mapping: [
                SubKernelSpec(
                    kernel_type="PadKernel",
                    input_shapes=[(128, 5120)],
                    dtype="DT_BF16",
                ),
                SubKernelSpec(
                    kernel_type="MissingKernel",
                    input_shapes=[(64, 5120)],
                    dtype="DT_BF16",
                ),
            ],
        )

        op = _make_op_info(
            "tensor_cast.fake_partial_debug",
            [torch.empty(128, 5120, device="meta", dtype=torch.bfloat16)],
        )
        result = ds.lookup(op)

        assert result is not None
        assert result.source == QuerySource.PARTIAL
        assert result.shape_match_info is None
        assert result.confidence == pytest.approx(0.5)
        assert result.details["partial"] is True
        assert result.details["missed_kernels"] == ["MissingKernel"]
        assert ds.last_miss_reason == "sub_kernel_miss:MissingKernel"
        assert result.sub_kernel_shapes is not None
        assert len(result.sub_kernel_shapes) == 1
        assert result.sub_kernel_shapes[0].kernel_type == "PadKernel"
        assert result.sub_kernel_shapes[0].kernel_shapes == [[120, 5120]]
        assert result.sub_kernel_shapes[0].shape_match_rule == "padding"

    def test_partial_returns_partial_source(self, partial_match_data_dir):
        """When some sub-kernels hit and others miss, return PARTIAL."""
        ds = ProfilingDataSource(partial_match_data_dir)
        specs = [
            SubKernelSpec(
                kernel_type="QuantBatchMatmulV3",
                input_shapes=[(4099, 7168), (2112, 7168)],
                dtype="DT_BF16",
            ),
            SubKernelSpec(
                kernel_type="KvRmsNormRopeCache",
                input_shapes=[(4099, 576)],
                dtype="DT_BF16",
            ),
        ]

        op = _make_op_info(
            torch.ops.tensor_cast.mlapo_quant.default,
            [torch.empty(4099, 7168, device="meta", dtype=torch.bfloat16)],
        )
        result = ds._lookup_composite_decomposed(op, {}, lambda op, m: specs)
        assert result is not None
        assert result.source == QuerySource.PARTIAL
        assert result.latency_us == 80.0
        assert "hit_kernels" in result.details
        assert "missed_kernels" in result.details
        assert "KvRmsNormRopeCache" in result.details["missed_kernels"]
        assert result.confidence == pytest.approx(0.5)
        assert result.sub_kernel_shapes is not None
        assert len(result.sub_kernel_shapes) == 1
        assert result.sub_kernel_shapes[0].kernel_type == "QuantBatchMatmulV3"
        assert result.sub_kernel_shapes[0].simulation_shapes == [
            [4099, 7168],
            [2112, 7168],
        ]
        assert result.sub_kernel_shapes[0].kernel_shapes == [[4099, 7168], [2112, 7168]]
        assert result.sub_kernel_shapes[0].shape_match_rule == "identity"

    def test_all_hit_returns_measured(self, partial_match_data_dir):
        """When all sub-kernels hit, return MEASURED."""
        ds = ProfilingDataSource(partial_match_data_dir)
        specs = [
            SubKernelSpec(
                kernel_type="QuantBatchMatmulV3",
                input_shapes=[(4099, 7168), (2112, 7168)],
                dtype="DT_BF16",
            ),
        ]

        op = _make_op_info(
            torch.ops.tensor_cast.mlapo_quant.default,
            [torch.empty(4099, 7168, device="meta", dtype=torch.bfloat16)],
        )
        result = ds._lookup_composite_decomposed(op, {}, lambda op, m: specs)
        assert result is not None
        assert result.source == QuerySource.MEASURED
        assert result.confidence == 0.8
        # kernel_type must be a comma-separated string, not a list
        assert isinstance(result.details["kernel_type"], str)
        assert "QuantBatchMatmulV3" in result.details["kernel_type"]

    def test_all_miss_returns_none(self, partial_match_data_dir):
        """When all sub-kernels miss, return None to allow analytic fallback."""
        ds = ProfilingDataSource(partial_match_data_dir)
        specs = [
            SubKernelSpec(
                kernel_type="KvRmsNormRopeCache",
                input_shapes=[(4099, 576)],
                dtype="DT_BF16",
            ),
        ]

        op = _make_op_info(
            torch.ops.tensor_cast.mlapo_quant.default,
            [torch.empty(4099, 7168, device="meta", dtype=torch.bfloat16)],
        )
        result = ds._lookup_composite_decomposed(op, {}, lambda op, m: specs)
        assert result is None


# --- DFC EP Size matching tests ---

MOE_OP_MAPPING = """
version: "test"
device: TEST_DEVICE

operator_mappings:
  "tensor_cast.dispatch_ffn_combine.default":
    kernel_type: DispatchFFNCombine
    query_mode: moe_fused
    tc_input_count: 1
"""

MOE_DFC_CSV = """\
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,EP Size,Average Duration(us)
"513,7168","DT_BF16","ND","513,7168","DT_BF16","ND",16,235.0
"513,7168","DT_BF16","ND","513,7168","DT_BF16","ND",8,180.0
"1024,7168","DT_BF16","ND","1024,7168","DT_BF16","ND",16,400.0
"""


@pytest.fixture
def dfc_data_dir(tmp_path):
    d = tmp_path / "dfc"
    d.mkdir()
    (d / "op_mapping.yaml").write_text(MOE_OP_MAPPING)
    (d / "DispatchFFNCombine.csv").write_text(MOE_DFC_CSV.strip())
    return d


class TestLookupMoe:
    def test_ep_size_exact_match(self, dfc_data_dir):
        """Same shape, different EP sizes → match the right one."""
        ds = ProfilingDataSource(
            dfc_data_dir, parallel_config=_make_parallel_config(ep_size=16)
        )
        op = _make_op_info(
            torch.ops.tensor_cast.dispatch_ffn_combine.default,
            [
                torch.empty(513, 7168, device="meta", dtype=torch.bfloat16),
                torch.empty(513, dtype=torch.int64, device="meta"),
            ],
        )
        result = ds.lookup(op)
        assert result is not None
        assert abs(result.latency_us - 235.0) < 0.01

    def test_ep_size_8_matches_different_row(self, dfc_data_dir):
        ds = ProfilingDataSource(
            dfc_data_dir, parallel_config=_make_parallel_config(ep_size=8)
        )
        op = _make_op_info(
            torch.ops.tensor_cast.dispatch_ffn_combine.default,
            [
                torch.empty(513, 7168, device="meta", dtype=torch.bfloat16),
                torch.empty(513, dtype=torch.int64, device="meta"),
            ],
        )
        result = ds.lookup(op)
        assert result is not None
        assert abs(result.latency_us - 180.0) < 0.01

    def test_ep_size_not_configured_misses(self, dfc_data_dir):
        """CSV has EP Size column but ProfilingDataSource has no ep_size → MISS."""
        ds = ProfilingDataSource(dfc_data_dir)  # no ep_size
        op = _make_op_info(
            torch.ops.tensor_cast.dispatch_ffn_combine.default,
            [
                torch.empty(513, 7168, device="meta", dtype=torch.bfloat16),
                torch.empty(513, dtype=torch.int64, device="meta"),
            ],
        )
        result = ds.lookup(op)
        assert result is None
        assert ds.last_miss_reason == "ep_size_not_configured"

    def test_shape_miss(self, dfc_data_dir):
        """Shape doesn't match any CSV row.

        NOTE: This is the last test in TestLookupMoe.
        """
        ds = ProfilingDataSource(
            dfc_data_dir, parallel_config=_make_parallel_config(ep_size=16)
        )
        op = _make_op_info(
            torch.ops.tensor_cast.dispatch_ffn_combine.default,
            [
                torch.empty(999, 7168, device="meta", dtype=torch.bfloat16),
                torch.empty(999, dtype=torch.int64, device="meta"),
            ],
        )
        result = ds.lookup(op)
        assert result is None


class TestBlockSizes:
    def test_block_size_8(self):
        """MISS #8: 4104 = ceil(4099/8)*8 must be recognized as block-padded."""
        assert _is_block_padded(4104, 4099) is True

    def test_block_size_16(self):
        assert _is_block_padded(4112, 4099) is True

    def test_not_padded(self):
        assert _is_block_padded(4100, 4099) is False


# --- accepted_miss tests ---

ACCEPTED_MISS_OP_MAPPING_YAML = """
version: "test"
device: TEST_DEVICE

operator_mappings:
  "tensor_cast.concat_and_cache_mla.default":
    kernel_type: ReshapeAndCacheNdKernel
    tc_input_count: 2
    accepted_miss: "MLA cache write absorbed by KvRmsNormRopeCache."
  "aten.mm.default":
    kernel_type: MatMulV2
"""


@pytest.fixture
def accepted_miss_data_dir(tmp_path):
    data_dir = tmp_path / "accepted_miss"
    data_dir.mkdir()
    (data_dir / "op_mapping.yaml").write_text(ACCEPTED_MISS_OP_MAPPING_YAML)
    return data_dir


def test_accepted_miss_returns_zero_latency_hit(accepted_miss_data_dir):
    """accepted_miss ops return QueryResult with latency=0 and note."""
    ds = ProfilingDataSource(accepted_miss_data_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.concat_and_cache_mla.default,
        [
            torch.empty(4099, 512, device="meta", dtype=torch.bfloat16),
            torch.empty(4099, 64, device="meta", dtype=torch.bfloat16),
        ],
    )
    result = ds.lookup(op)
    assert result is not None
    assert result.latency_us == 0.0
    assert result.confidence == 1.0
    assert result.details["kernel_type"] == "accepted_miss"
    assert result.details["zero_cost"] is True
    assert "KvRmsNormRopeCache" in result.details["note"]
    assert result.shape_match_info is not None
    assert result.shape_match_info.simulation_shapes == [[4099, 512], [4099, 64]]
    assert result.shape_match_info.kernel_shapes == []
    assert result.shape_match_info.shape_match_rule == "accepted_miss"


def test_zero_cost_returns_shape_debug_info(tmp_path):
    data_dir = tmp_path / "zero_cost"
    data_dir.mkdir()
    (data_dir / "op_mapping.yaml").write_text(
        'version: "test"\n'
        "device: TEST_DEVICE\n\n"
        "operator_mappings:\n"
        '  "aten.view.default":\n'
        "    kernel_type: View\n"
        "    zero_cost: true\n"
    )
    ds = ProfilingDataSource(data_dir)
    op = _make_op_info(
        torch.ops.aten.view.default,
        [
            torch.empty(1, 136, 5120, device="meta", dtype=torch.bfloat16),
            [136, 5120],
        ],
    )

    result = ds.lookup(op)

    assert result is not None
    assert result.latency_us == 0.0
    assert result.details == {"kernel_type": "View", "zero_cost": True}
    assert result.shape_match_info is not None
    assert result.shape_match_info.simulation_shapes == [[1, 136, 5120]]
    assert result.shape_match_info.kernel_shapes == []
    assert result.shape_match_info.shape_match_rule == "zero_cost"


def test_accepted_miss_does_not_affect_normal_ops(accepted_miss_data_dir):
    """Ops without accepted_miss still go through normal lookup (MISS if no CSV)."""
    ds = ProfilingDataSource(accepted_miss_data_dir)
    op = _make_op_info(
        torch.ops.aten.mm.default,
        [
            torch.empty(136, 5120, device="meta", dtype=torch.bfloat16),
            torch.empty(5120, 768, device="meta", dtype=torch.bfloat16),
        ],
    )
    result = ds.lookup(op)
    assert result is None  # no CSV → MISS
