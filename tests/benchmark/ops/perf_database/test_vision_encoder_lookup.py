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

"""PR-8 vision encoder kernel routing tests (Bailing V3 VL).

Covers the three PR-8 op_mapping/lookup changes:
- Encoder varlen attention (vision tower) routes to ``UnpadFlashAttention``
  via ``varlen_encoder_kernel_type`` instead of failing on missing seq_lens.
- Paged 2-D queries that carry seq_lens keep the FIA enriched-CSV path.
- ``aten.convolution`` routes by first-input rank: rank 3 -> Conv1D,
  rank 4 -> Conv2D, rank 5 -> Conv3D (vision patch-embed).
- ``aten.clone`` materialization (merger reshape) resolves to a measured
  ``TensorMove`` row instead of the retired zero_cost shortcut.
"""

import shutil
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from tensor_cast.performance_model.profiling_database.data_source import QuerySource
from tensor_cast.performance_model.profiling_database.interpolating_data_source import (
    InterpolatingDataSource,
)
from tensor_cast.performance_model.profiling_database.profiling_data_source import (
    ProfilingDataSource,
)

_FIA_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "fia_raw_test"
_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "varlen_encoder_test"


def _make_mock_device_profile():
    dp = MagicMock()
    dp.name = "ATLAS_800_A3_752T_128G_DIE"
    dp.comm_grid = None
    return dp


def _make_op_info(func_name, args):
    """Build a mock OpInvokeInfo whose str(func) matches an op_mapping key."""
    op_info = MagicMock()
    op_info.func.__str__ = lambda self: func_name  # type: ignore[method-assign]
    op_info.func.__repr__ = lambda self: func_name  # type: ignore[method-assign]
    op_info.args = args
    op_info.kwargs = {}
    op_info.out = None
    return op_info


def _vision_attention_op(func_name="torch.ops.tensor_cast.attention.default"):
    """Encoder varlen attention call: flattened 2-D query, no paged metadata.

    Mirrors the Bailing V3 VL vision tower: query [total_tokens, hidden],
    key/value [1, total_tokens, heads, head_dim] and attention_meta=None,
    so block_table / query_start_loc / seq_lens / query_lens are all None.
    """
    query = torch.zeros((8160, 1152), dtype=torch.bfloat16)
    key = torch.zeros((1, 8160, 16, 72), dtype=torch.bfloat16)
    value = torch.zeros((1, 8160, 16, 72), dtype=torch.bfloat16)
    return _make_op_info(func_name, (query, key, value, None, None, None, None, None))


@pytest.fixture
def ds(tmp_path):
    dst = tmp_path / "db"
    shutil.copytree(_FIA_FIXTURE_DIR, dst)
    shutil.copytree(_FIXTURE_DIR, dst, dirs_exist_ok=True)
    return ProfilingDataSource(str(dst), _make_mock_device_profile())


class TestVarlenEncoderAttention:
    def test_v1_encoder_hits_are_segment_gated(self, ds):
        """Exact UnpadFlashAttention hits stay disabled until the segment
        signature is carried end-to-end (PR #1101 review): the op contract
        cannot distinguish segmentations, so shape matches must not be
        reported as MEASURED. Kernel label + demand are still recorded.
        """
        result = ds.lookup(_vision_attention_op())
        assert result is None
        assert ds.last_miss_reason == "varlen_encoder_segment_signature_unavailable"

    def test_v1b_segment_ambiguity_never_reuses_a_measured_row(self, ds):
        """cu_seqlens [0, 4080, 8160] and [0, 2040, 8160] share the same
        (shape, dtype) op call (query=[8160, 1152], K/V=[1, 8160, 16, 72])
        although their full-attention ΣL² differs (33,292,800 vs 41,616,000).
        Both must miss with the segment reason — a measured row must never be
        reused across segmentations.
        """
        first = _vision_attention_op()
        second = _vision_attention_op()  # identical op call: cu_seqlens is absent
        assert ds.lookup(first) is None
        assert ds.last_miss_reason == "varlen_encoder_segment_signature_unavailable"
        assert ds.lookup(second) is None
        assert ds.last_miss_reason == "varlen_encoder_segment_signature_unavailable"

    def test_v1c_exact_hit_requires_validated_segment_signature(self, ds, monkeypatch):
        """Opening the segment gate (mocking a validated signature) restores
        the measured lookup: the gate, not the matcher, is what blocks hits.
        """
        monkeypatch.setattr(ds, "_varlen_encoder_segment_signature", lambda info: (0, 8160))
        result = ds.lookup(_vision_attention_op())
        assert result is not None
        assert result.details["kernel_type"] == "UnpadFlashAttention"
        assert abs(result.latency_us - 321.0) < 1.0
        assert result.source == QuerySource.MEASURED

    def test_v2_paged_2d_query_still_uses_fia(self, ds):
        """A 2-D query WITH seq_lens keeps the paged FIA projection."""
        op = _make_op_info(
            "torch.ops.tensor_cast.attention.default",
            (
                torch.zeros((128, 512), dtype=torch.bfloat16),
                torch.zeros((12307, 128, 128), dtype=torch.bfloat16),
                torch.zeros((12307, 128, 128), dtype=torch.bfloat16),
                None,
                None,
                None,
                torch.full((128,), 4096, dtype=torch.int64),
                None,
            ),
        )
        result = ds.lookup(op)
        assert result is not None
        assert result.details["kernel_type"] == "FusedInferAttentionScore"
        assert abs(result.latency_us - 58.2) < 1.0

    def test_v3_missing_encoder_kernel_field_reports_dedicated_reason(self, ds):
        """attention_quant has no varlen_encoder_kernel_type: the varlen call
        must miss with varlen_encoder_kernel_type_missing, not missing_seq_lens.
        """
        op = _vision_attention_op("torch.ops.tensor_cast.attention_quant.default")
        # attention_quant carries extra scale/offset scalars after query_lens.
        op.args = op.args + (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, torch.bfloat16)
        assert ds.lookup(op) is None
        assert ds.last_miss_reason == "varlen_encoder_kernel_type_missing"

    def test_v4_encoder_without_csv_reports_csv_not_found(self, ds, monkeypatch):
        """With the segment gate open (validated signature) and the CSV
        removed, the exact lookup must report csv_not_found.
        """
        (Path(ds.data_dir) / "UnpadFlashAttention.csv").unlink()
        monkeypatch.setattr(ds, "_varlen_encoder_segment_signature", lambda info: (0, 8160))
        result = ds.lookup(_vision_attention_op())
        assert result is None
        assert ds.last_miss_reason == "csv_not_found"

    def test_v5_paged_3d_tnd_query_unaffected(self, ds):
        """A regular 3-D TND paged query still resolves through FIA."""
        op = _make_op_info(
            "torch.ops.tensor_cast.attention.default",
            (
                torch.zeros((128, 4, 128), dtype=torch.bfloat16),
                torch.zeros((12307, 128, 128), dtype=torch.bfloat16),
                torch.zeros((12307, 128, 128), dtype=torch.bfloat16),
                None,
                None,
                None,
                torch.full((128,), 4096, dtype=torch.int64),
                None,
            ),
        )
        result = ds.lookup(op)
        assert result is not None
        assert result.details["kernel_type"] == "FusedInferAttentionScore"

    def test_v6_interpolating_wrapper_skips_fia_for_varlen(self, tmp_path):
        """The interpolation wrapper must not run the paged FIA projection on
        encoder varlen calls; it records a dedicated miss instead and keeps
        the analytic fallback.
        """
        dst = tmp_path / "db"
        shutil.copytree(_FIA_FIXTURE_DIR, dst)
        shutil.copytree(_FIXTURE_DIR, dst, dirs_exist_ok=True)
        base = ProfilingDataSource(str(dst), _make_mock_device_profile())
        wrapper = InterpolatingDataSource(base)
        result = wrapper.lookup(_vision_attention_op())
        assert result is None
        # Base exact routing reported the segment-gated miss; the wrapper
        # classified the varlen call itself (not an attention-index problem).
        assert base.last_miss_reason == "varlen_encoder_segment_signature_unavailable"
        assert wrapper._last_miss_reason == "varlen_encoder_not_interpolated"


class TestConvolutionRankRouting:
    @staticmethod
    def _convolution_op(input_shape, weight_shape, bias_size, stride, padding):
        args = (
            torch.zeros(input_shape, dtype=torch.bfloat16),
            torch.zeros(weight_shape, dtype=torch.bfloat16),
            torch.zeros((bias_size,), dtype=torch.bfloat16),
            stride,
            padding,
            (1,) * len(stride),
            False,
            (0,) * len(stride),
            1,
        )
        return _make_op_info("torch.ops.aten.convolution.default", args)

    def test_c1_conv3d_patch_embed(self, ds):
        """Bailing V3 VL patch-embed: rank-5 input routes to Conv3D."""
        op = self._convolution_op((8160, 3, 2, 16, 16), (1152, 3, 2, 16, 16), 1152, (2, 16, 16), (0, 0, 0))
        result = ds.lookup(op)
        assert result is not None
        assert result.details["kernel_type"] == "Conv3D"
        assert abs(result.latency_us - 410.5) < 1.0

    def test_c2_conv2d_default_rank(self, ds):
        """Rank-4 input keeps Conv2D first."""
        op = self._convolution_op((8, 3, 224, 224), (64, 3, 7, 7), 64, (7, 7), (3, 3))
        result = ds.lookup(op)
        assert result is not None
        assert result.details["kernel_type"] == "Conv2D"
        assert abs(result.latency_us - 55.6) < 1.0

    def test_c3_conv1d_rank(self, ds):
        """Rank-3 input routes to Conv1D."""
        op = self._convolution_op((2, 64, 512), (128, 64, 4), 128, (4,), (0,))
        result = ds.lookup(op)
        assert result is not None
        assert result.details["kernel_type"] == "Conv1D"
        assert abs(result.latency_us - 12.3) < 1.0

    def test_c4_conv3d_without_csv_misses_on_rank_mismatched_conv2d(self, ds):
        """Without Conv3D.csv the rank-5 call must miss (analytic fallback),
        not silently hit a rank-4 Conv2D row.
        """
        (Path(ds.data_dir) / "Conv3D.csv").unlink()
        result = ds.lookup(
            self._convolution_op((8160, 3, 2, 16, 16), (1152, 3, 2, 16, 16), 1152, (2, 16, 16), (0, 0, 0))
        )
        assert result is None


class TestCloneMaterialization:
    def test_t1_interpolate_merge_copy_resolves_measured_tensor_move(self, ds):
        """The Bailing V3 VL pos-embed interpolation's window-merge reorder
        (permute(0, 2, 1, 3, 4).reshape of the [h/m, m, w/m, m, C] grid, e.g.
        [34, 60, 2, 2, 1152] at 1080p) is anchored on
        tensor_cast.materialize by the wrapper patch and resolves a measured
        TensorMove row; the shape list is host-side metadata and carries no
        bytes.
        """
        op = _make_op_info(
            "torch.ops.tensor_cast.materialize.default",
            (torch.zeros((34, 60, 2, 2, 1152), dtype=torch.bfloat16), [1, -1, 1152]),
        )
        result = ds.lookup(op)
        assert result is not None
        assert result.details["kernel_type"] == "TensorMove"
        assert abs(result.latency_us - 17.8) < 1.0
        assert not result.details.get("zero_cost")

    def test_t2_clone_keeps_tensor_move_where_not_zero_cost(self, ds):
        """On stacks where aten.clone is NOT zero_cost (v0.13/v0.15 keep the
        original TensorMove mapping), clone lookups still resolve measured
        TensorMove rows. The v0.18+/v0.23 stacks keep clone zero_cost (fused
        MoE routing glue) and route the merger copy through
        tensor_cast.materialize instead — see test_t1.
        """
        op = _make_op_info(
            "torch.ops.aten.clone.default",
            (torch.zeros((4096, 4096), dtype=torch.bfloat16),),
        )
        result = ds.lookup(op)
        assert result is not None
        assert result.details["kernel_type"] == "TensorMove"
        assert abs(result.latency_us - 6.5) < 1.0
