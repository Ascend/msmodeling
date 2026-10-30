# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the
# Mulan PSL v2.
# You may obtain a copy of the Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2/
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE. See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Bailing V3 VL official modeling contract (PR-2, issue #536).

Loads the official self-contained remote-code modeling vendored in the repo
asset (the modeling ships its own ImportError fallback to pure-PyTorch KDA
implementations, so no fla dependency is required) and checks:

- the meta-built structure matches the Ling-3.0-flash-VL checkpoint layout
  (62,563-tensor index verified during design);
- the official M-RoPE position semantics (SGLang ``bailing_3drope`` layout):
  text segments share one incremental axis, image segments use the
  body-diagonal (t, h - (h-1)//2 + t, w - (w-1)//2 + t) grid, and the next
  text segment resumes after the segment's ``t``-row maximum;
- the M-RoPE frequency layout via ``apply_video_rope``: alternating H/W
  dims inside each spatial section, T dims block-copied.

The remote-code import is expensive (~1 min), so every test shares one
module-scoped fixture. The official implementation (rather than the earlier
vLLM-PR-55921 thin shell) is the Phase-3 numerical B2B reference.
"""

import pytest
import torch
from transformers import AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module

_ASSET = "tests/assets/model_config/ling_3_0_flash_vl"
_IMAGE_TOKEN = 157157
_VISION_START = 157158
_VISION_END = 157159
_MERGE_SIZE = 2


@pytest.fixture(scope="module")
def vl_model():
    """Meta-build the official VL model once for the whole module."""
    from tensor_cast.transformers.builtin_model.bailing_v3_vl import (
        _patch_hf_config_for_bailing_moe_v3_vl,
    )

    config = AutoConfig.from_pretrained(_ASSET, trust_remote_code=True)
    _patch_hf_config_for_bailing_moe_v3_vl(config)

    model_cls = get_class_from_dynamic_module(
        "modeling_bailing_moe_v3_vl.BailingMoeV3VLForConditionalGeneration",
        _ASSET,
        trust_remote_code=True,
    )
    with torch.device("meta"):
        model = model_cls(config)
    return model


def test_meta_structure_matches_checkpoint(vl_model):
    inner = vl_model.model
    assert type(inner).__name__ == "BailingMoeV3VLModel"
    assert len(inner.layers) == 42
    mla_layers = [
        idx
        for idx, layer in enumerate(inner.layers)
        if type(layer.attention).__name__ == "BailingMoeV3MultiLatentAttention"
    ]
    assert mla_layers == [5, 11, 17, 23, 29, 35, 41]
    kda_count = sum(1 for layer in inner.layers if type(layer.attention).__name__ == "BailingMoeV3KimiDeltaAttention")
    assert kda_count == 35

    visual = inner.visual
    assert type(visual).__name__ == "BailingMoeV3VLVisionModel"
    assert len(visual.blocks) == 27
    # The official merger class is the shared Qwen3-VL-MoE merger; this
    # checkpoint sets disable_merger_proj=True so it degenerates to
    # Norm + 2x2 reshape with no projection linears.
    assert type(visual.merger).__name__ == "Qwen3VLMoeVisionPatchMerger"
    assert visual.disable_merger_proj is True
    assert not any("linear" in name for name, _ in visual.merger.named_parameters())
    assert tuple(visual.merger.norm.weight.shape) == (1152,)
    assert visual.use_deepstack is False
    assert list(getattr(visual, "deepstack_visual_indexes", []) or []) == []
    assert getattr(visual, "deepstack_merger_list", []) == []
    assert tuple(visual.patch_embed.proj.weight.shape) == (1152, 3, 2, 16, 16)
    assert tuple(visual.pos_embed.weight.shape) == (2304, 1152)

    assert tuple(vl_model.linear_proj[0].weight.shape) == (2560, 4608)
    assert tuple(vl_model.linear_proj[0].bias.shape) == (2560,)
    assert tuple(vl_model.linear_proj[2].weight.shape) == (2560, 2560)
    assert tuple(vl_model.lm_head.weight.shape) == (157184, 2560)
    assert tuple(inner.word_embeddings.weight.shape) == (157184, 2560)

    moe = inner.layers[5].mlp
    assert type(moe).__name__ == "BailingMoeV3SparseMoeBlock"
    assert len(moe.experts) == 512
    assert tuple(moe.gate.weight.shape) == (512, 2560)
    assert type(inner.layers[0].mlp).__name__ == "BailingMoeV3MLP"


def test_checkpoint_namespace_roots_align(vl_model):
    names = dict(vl_model.named_parameters())
    expected_roots = [
        "model.visual.pos_embed.weight",
        "model.visual.patch_embed.proj.weight",
        "model.visual.merger.norm.weight",
        "model.visual.blocks.0.attn.qkv.weight",
        "model.visual.blocks.0.mlp.linear_fc1.weight",
        "model.word_embeddings.weight",
        "model.layers.0.attention.q_proj.weight",
        "model.layers.0.attention.q_conv1d.weight",
        "model.layers.5.attention.q_proj.weight",
        "model.layers.5.attention.kv_b_proj.weight",
        "model.layers.5.attention.dense.weight",
        "model.layers.5.mlp.gate.weight",
        "model.layers.5.mlp.experts.0.gate_proj.weight",
        "model.layers.5.mlp.shared_experts.gate_proj.weight",
        "model.norm.weight",
        "linear_proj.0.weight",
        "linear_proj.0.bias",
        "linear_proj.2.weight",
        "linear_proj.2.bias",
        "lm_head.weight",
    ]
    missing = [key for key in expected_roots if key not in names]
    assert missing == []
    assert not [key for key in names if "deepstack" in key and key.startswith("model.visual.")]


def test_official_vl_modeling_strips_native_mtp_tail():
    """Internal SFT configs (e.g. ling_vl_flash_sft_rc2) declare 1 native MTP
    layer via ``text_config.num_nextn_predict_layers`` while the open
    Ling-3.0-flash-VL config ships 0. The official modeling then appends a
    ``BailingMoeV3MTPLayer`` to ``model.layers`` (43 = 42 main + 1 MTP), and
    the TensorCast VL patch must strip the validated tail instead of
    rejecting the model at initialization (PR-2 review finding).
    """
    from tensor_cast.transformers.builtin_model.bailing_v3 import _strip_bailing_v3_native_mtp
    from tensor_cast.transformers.builtin_model.bailing_v3_vl import (
        _patch_hf_config_for_bailing_moe_v3_vl,
    )

    config = AutoConfig.from_pretrained(_ASSET, trust_remote_code=True)
    _patch_hf_config_for_bailing_moe_v3_vl(config)
    config.text_config.num_nextn_predict_layers = 1

    model_cls = get_class_from_dynamic_module(
        "modeling_bailing_moe_v3_vl.BailingMoeV3VLForConditionalGeneration",
        _ASSET,
        trust_remote_code=True,
    )
    with torch.device("meta"):
        mtp_model = model_cls(config)
    inner = mtp_model.model

    # The official modeling builds 42 main layers + the MTP tail.
    assert len(inner.layers) == 43
    assert type(inner.layers[-1]).__name__ == "BailingMoeV3MTPLayer"
    assert inner.num_nextn_predict_layers == 1

    stripped = _strip_bailing_v3_native_mtp(inner)
    assert stripped == 1
    assert len(inner.layers) == 42
    assert inner.num_nextn_predict_layers == 0
    # The main stack layout is unchanged from the open config: the last main
    # layer (idx 41) is an MLA layer and no MTP class remains.
    assert type(inner.layers[-1].attention).__name__ == "BailingMoeV3MultiLatentAttention"
    assert not any(type(layer).__name__ == "BailingMoeV3MTPLayer" for layer in inner.layers)


def test_config_patch_bridges_top_level_mrope_section(vl_model):
    """The real config.json declares mrope_section=[8, 12, 12] at top level
    only; the official backbone initializes from config.text_config and the
    rotary falls back to [12, 10, 10] when the bridge is missing — the T-axis
    frequency span would silently shift 8 -> 12 (review: zhenghaojie). The
    module fixture loads through the patched config, so the loaded rotary
    must resolve the declared section, not the default.
    """
    from tensor_cast.transformers.builtin_model.bailing_v3_vl import (
        _patch_hf_config_for_bailing_moe_v3_vl,
    )

    config = AutoConfig.from_pretrained(_ASSET, trust_remote_code=True)
    assert list(config.mrope_section) == [8, 12, 12]
    assert getattr(config.text_config, "mrope_section", None) is None
    assert _patch_hf_config_for_bailing_moe_v3_vl(config) is True
    assert list(config.text_config.mrope_section) == [8, 12, 12]

    # The module fixture was built through the same patched config chain.
    assert list(vl_model.model.rotary_emb.mrope_section) == [8, 12, 12]


def test_official_runtime_placeholder_mask_returns_concrete_count_mask(vl_model):
    """meta-mode placeholder-mask contract (review: zhenghaojie, patch 3 of
    _patch_bailing_v3_vl_official_runtime): with meta inputs the patched
    get_placeholder_mask returns a CPU-concrete mask whose True count equals
    image_features rows x hidden so the official ``mask.sum().item()`` check
    passes while masked_scatter stays shape-only.
    """
    from tensor_cast.transformers.builtin_model.bailing_v3_vl import (
        _patch_bailing_v3_vl_official_runtime,
    )

    _patch_bailing_v3_vl_official_runtime(vl_model.model)
    inner = vl_model.model
    hidden = inner.config.hidden_size
    embeds = torch.zeros((1, 23, hidden), device="meta")
    image_features = torch.zeros((16, hidden), device="meta")
    input_ids = torch.zeros((1, 23), dtype=torch.long, device="meta")

    image_mask, video_mask = inner.get_placeholder_mask(input_ids, embeds, image_features=image_features)

    assert image_mask.shape == embeds.shape
    assert image_mask.dtype == torch.bool
    assert image_mask.device.type == "cpu"
    assert int(image_mask.sum()) == image_features.shape[0] * hidden
    assert video_mask is not None


def test_official_runtime_positions_short_circuit_on_meta(vl_model):
    """meta-mode M-RoPE positions contract (review: zhenghaojie, patch 4):
    compute_3d_position_ids short-circuits on meta tensors to the 3-axis
    text baseline [3, B, L] with shared incremental values, replacing the
    token-value-driven get_rope_index that cannot run structurally.
    """
    from tensor_cast.transformers.builtin_model.bailing_v3_vl import (
        _patch_bailing_v3_vl_official_runtime,
    )

    _patch_bailing_v3_vl_official_runtime(vl_model.model)
    inner = vl_model.model
    input_ids = torch.zeros((1, 23), dtype=torch.long, device="meta")
    embeds = torch.zeros((1, 23, inner.config.hidden_size), device="meta")
    image_grid_thw = torch.tensor([[1, 8, 8]], dtype=torch.long)

    positions = inner.compute_3d_position_ids(input_ids, embeds, image_grid_thw=image_grid_thw)

    assert positions.shape == (3, 1, 23)
    axis_rows = positions[:, 0, :].tolist()
    assert axis_rows == [list(range(23))] * 3
    # The meta short-circuit also resets rope_deltas to the text baseline.
    assert tuple(inner.rope_deltas.shape) == (1, 1)


def test_official_runtime_patch_embed_routes_through_conv3d(vl_model):
    """patch-embed runtime patch (review: zhenghaojie, patch 2): the class
    forward routes through the Conv3d module (the single-Conv3D anchor of
    PR-8 P1-06) instead of the official unfold+F.linear SGLang-alignment
    path, keeping shape propagation on meta tensors.
    """
    from tensor_cast.transformers.builtin_model.bailing_v3_vl import (
        _patch_bailing_v3_vl_official_runtime,
    )

    _patch_bailing_v3_vl_official_runtime(vl_model.model)
    patch_embed = vl_model.model.visual.patch_embed
    assert getattr(type(patch_embed), "_bailing_vl_tc_patch_embed_patched", False)

    rows = 8160
    flat = torch.zeros(
        (rows, patch_embed.in_channels * patch_embed.temporal_patch_size * patch_embed.patch_size**2),
        device="meta",
    )
    out = patch_embed(flat)
    assert out.shape == (rows, patch_embed.embed_dim)


def _single_request_ids(num_prefix_text, grid_h, grid_w, num_suffix_text):
    placeholder = grid_h * grid_w // (_MERGE_SIZE**2)
    tokens = (
        [10] * num_prefix_text + [_VISION_START] + [_IMAGE_TOKEN] * placeholder + [_VISION_END] + [11] * num_suffix_text
    )
    return torch.tensor([tokens], dtype=torch.long)


def test_mrope_positions_text_image_text(vl_model):
    # 3 prefix text + vision_start + 16 placeholders (8x8 grid) + vision_end
    # + 2 text = 23 tokens; official SGLang body-diagonal layout.
    input_ids = _single_request_ids(3, 8, 8, 2)
    grid_thw = torch.tensor([[1, 8, 8]], dtype=torch.long)

    position_ids, deltas = vl_model.model.get_rope_index(input_ids, image_grid_thw=grid_thw)
    assert position_ids.shape == (3, 1, 23)

    rows = position_ids[:, 0].tolist()
    # Prefix text (including the vision_start boundary token): shared axes 0..3.
    assert [row[:4] for row in rows] == [[0, 1, 2, 3]] * 3
    # Image segment: t stays at the segment base 4 (t axis = base + 2*frame
    # with a single frame); h spans 3..6 by grid row (- (4-1)//2 = -1);
    # w spans 3..6 by grid column.
    image = [row[4:20] for row in rows]
    assert image[0] == [4] * 16
    assert image[1] == [3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6]
    assert image[2] == [3, 4, 5, 6] * 4
    # Suffix text (vision_end + 2 tokens) resumes after the segment's t-row
    # maximum + 1 = 5 (the body-diagonal keeps h/w up to 6, but the cursor
    # contract follows the t row, matching SGLang bailing_3drope).
    assert [row[20:] for row in rows] == [[5, 6, 7]] * 3
    # mrope delta = t-row max + 1 - seq_len = 8 - 23 = -15.
    assert deltas.flatten().tolist() == [-15]


def test_mrope_positions_multi_image_segments(vl_model):
    # Two 4-placeholder images (2x2 grids) with one text token in between.
    tokens = (
        [10]
        + [_VISION_START]
        + [_IMAGE_TOKEN] * 4
        + [_VISION_END]
        + [12]
        + [_VISION_START]
        + [_IMAGE_TOKEN] * 4
        + [_VISION_END]
        + [11, 11]
    )
    input_ids = torch.tensor([tokens], dtype=torch.long)
    grid_thw = torch.tensor([[1, 4, 4], [1, 4, 4]], dtype=torch.long)

    position_ids, _ = vl_model.model.get_rope_index(input_ids, image_grid_thw=grid_thw)
    assert position_ids.shape == (3, 1, 16)
    rows = position_ids[:, 0].tolist()
    # First image starts at token 2 with the segment base 2 (1 text +
    # vision_start); the 2x2 merged grid has a zero center offset.
    assert [row[2:6] for row in rows] == [
        [2, 2, 2, 2],
        [2, 2, 3, 3],
        [2, 3, 2, 3],
    ]
    # Text between the images (vision_end + text + vision_start, 3 tokens)
    # resumes after the first segment's t-row max + 1 = 3.
    assert [row[6:9] for row in rows] == [[3, 4, 5]] * 3
    # The second image re-anchors on the running t row: base = 3 + 3 = 6.
    assert [row[9:13] for row in rows] == [
        [6, 6, 6, 6],
        [6, 6, 7, 7],
        [6, 7, 6, 7],
    ]
    # Final text (vision_end + 2 tokens) resumes at 6 + 1 = 7.
    assert [row[13:] for row in rows] == [[7, 8, 9]] * 3


def test_mrope_positions_non_square_grid_text_resumes_after_image_span(vl_model):
    # A non-square grid (1, 8, 16) merges to (t=1, h=4, w=8): the W axis
    # spans 1..8 (- (8-1)//2 = -3 offset from the segment base 4) while the
    # trailing text still resumes after the t-row maximum (5).
    input_ids = _single_request_ids(3, 8, 16, 2)
    grid_thw = torch.tensor([[1, 8, 16]], dtype=torch.long)

    position_ids, _ = vl_model.model.get_rope_index(input_ids, image_grid_thw=grid_thw)
    assert position_ids.shape == (3, 1, 39)
    rows = position_ids[:, 0].tolist()

    image = [row[4:36] for row in rows]
    assert image[0] == [4] * 32
    assert image[1] == [3] * 8 + [4] * 8 + [5] * 8 + [6] * 8
    assert image[2] == [1, 2, 3, 4, 5, 6, 7, 8] * 4
    # Suffix text (vision_end + 2 tokens) resumes at the t-row max + 1 = 5.
    assert [row[36:] for row in rows] == [[5, 6, 7]] * 3


def test_mrope_positions_pure_text_is_shared_baseline(vl_model):
    input_ids = torch.tensor([[10, 11, 12, 13]], dtype=torch.long)
    position_ids, deltas = vl_model.model.get_rope_index(input_ids)
    assert position_ids.shape == (3, 1, 4)
    expected = torch.arange(4).view(1, 1, 4).expand(3, 1, 4)
    assert torch.equal(position_ids, expected)
    assert deltas.flatten().tolist() == [0]


def test_mrope_positions_supports_temporal_frames(vl_model):
    # Video grids with t>1 use the t axis as ``base + 2 * frame``: the second
    # frame's positions anchor at 4 (base 2 + 2*1) while h/w re-run the
    # body-diagonal offsets from the new base. Placeholder count for (2, 8, 8)
    # is 2 * 8 * 8 / 4 = 32.
    tokens = [10, _VISION_START] + [_IMAGE_TOKEN] * 32 + [_VISION_END]
    input_ids = torch.tensor([tokens], dtype=torch.long)
    grid_thw = torch.tensor([[2, 8, 8]], dtype=torch.long)

    position_ids, _ = vl_model.model.get_rope_index(input_ids, image_grid_thw=grid_thw)
    assert position_ids.shape == (3, 1, 35)
    rows = position_ids[:, 0].tolist()
    # Prefix: [text, vision_start] shared 0, 1. First frame: t = 2, h/w span
    # 1..4 (-(4-1)//2 = -1 body-diagonal offset).
    assert [row[:18] for row in rows] == [
        [0, 1] + [2] * 16,
        [0, 1] + [1] * 4 + [2] * 4 + [3] * 4 + [4] * 4,
        [0, 1] + [1, 2, 3, 4] * 4,
    ]
    # Second frame: t jumps to 2 + 2*1 = 4; h/w re-run 3..6 from the new base.
    assert [row[18:34] for row in rows] == [
        [4] * 16,
        [3] * 4 + [4] * 4 + [5] * 4 + [6] * 4,
        [3, 4, 5, 6] * 4,
    ]
    # Suffix vision_end resumes at the t-row max + 1 = 5.
    assert [row[34:] for row in rows] == [[5], [5], [5]]


def test_mrope_positions_rejects_missing_grid_entries(vl_model):
    # Two image starts but a single grid entry: the official implementation
    # raises instead of silently truncating.
    tokens = (
        [10]
        + [_VISION_START]
        + [_IMAGE_TOKEN] * 4
        + [_VISION_END]
        + [_VISION_START]
        + [_IMAGE_TOKEN] * 4
        + [_VISION_END]
    )
    input_ids = torch.tensor([tokens], dtype=torch.long)
    grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.long)
    with pytest.raises(ValueError, match="Not enough image_grid_thw entries"):
        vl_model.model.get_rope_index(input_ids, image_grid_thw=grid_thw)


@pytest.fixture(scope="module")
def small_mrope_rope(vl_model):
    """Build a tiny real-value M-RoPE with section [2, 2, 2]."""
    config_cls = get_class_from_dynamic_module(
        "configuration_bailing_moe_v3_vl.BailingMoeV3Config",
        _ASSET,
        trust_remote_code=True,
    )
    rope_cls = get_class_from_dynamic_module(
        "modeling_bailing_moe_v3_vl.BailingMoeV3VLRotaryEmbedding",
        _ASSET,
        trust_remote_code=True,
    )
    config = config_cls(
        qk_rope_head_dim=12,
        rope_theta=10_000.0,
        rope_scaling=None,
        max_position_embeddings=32,
        mrope_section=[2, 2, 2],
    )
    return rope_cls(config)


def test_mrope_section_validation(small_mrope_rope):
    assert list(small_mrope_rope.mrope_section) == [2, 2, 2]
    assert tuple(small_mrope_rope.inv_freq.shape) == (6,)
    # qk_rope_head_dim = 12 with partial_rotary_factor forced to 1.0.
    assert float(small_mrope_rope.config.partial_rotary_factor) == 1.0
    x = torch.empty(1, 2, 12)
    positions = torch.tensor([[[1, 2]], [[3, 4]], [[5, 6]]], dtype=torch.long)
    cos, sin = small_mrope_rope(x, positions)
    assert cos.shape == (1, 2, 12)
    assert sin.shape == (1, 2, 12)


def test_apply_video_rope_interleaves_hw_and_blocks_t(small_mrope_rope):
    # The layout for section [T=2, H=2, W=2] over a 12-dim head is the
    # reversed [H+W, T, H+W, T] section sequence: inside each spatial section
    # H and W alternate dim-by-dim (even j -> H row, odd j -> W row), and each
    # T section block-copies its dims from the t row.
    base = torch.arange(12, dtype=torch.float32)
    cos = torch.stack([base + 0.0, base + 100.0, base + 200.0]).view(3, 1, 1, 12)
    sin = torch.stack([base + 300.0, base + 400.0, base + 500.0]).view(3, 1, 1, 12)

    cos_flat, sin_flat = small_mrope_rope.apply_video_rope(cos, sin, [2, 2, 2])
    assert cos_flat.shape == (1, 1, 12)
    # Columns: [h0, w1, h2, w3, t4, t5, h6, w7, h8, w9, t10, t11]
    assert cos_flat[0, 0].tolist() == [
        100.0,
        201.0,
        102.0,
        203.0,
        4.0,
        5.0,
        106.0,
        207.0,
        108.0,
        209.0,
        10.0,
        11.0,
    ]
    assert sin_flat[0, 0].tolist() == [
        400.0,
        501.0,
        402.0,
        503.0,
        304.0,
        305.0,
        406.0,
        507.0,
        408.0,
        509.0,
        310.0,
        311.0,
    ]


def test_mrope_rope_forward_expands_2d_positions(small_mrope_rope):
    # A 2D position_ids input (text baseline) is expanded to the 3-axis
    # shared layout; the official rope consumes 3D positions directly.
    x = torch.empty(1, 2, 12)
    positions = torch.tensor([[1, 2]], dtype=torch.long)
    cos, sin = small_mrope_rope(x, positions)
    assert cos.shape == (1, 2, 12)
    cos_text, sin_text = small_mrope_rope(x, positions.expand(3, 1, 2))
    assert torch.equal(cos, cos_text)
    assert torch.equal(sin, sin_text)
