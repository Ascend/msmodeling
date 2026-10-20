"""Regression tests for DeepSeek V4.1 model support."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode
from transformers import AutoConfig, PretrainedConfig

import tensor_cast.ops  # noqa: F401 (register operators and performance handlers)
import tensor_cast.performance_model.builtin_model.deepseek_v41  # noqa: F401 (register operators and performance handlers)
import tensor_cast.performance_model.hyper_connection  # noqa: F401 (register operators and performance handlers)
from tensor_cast.core.input_generator import (
    _generate_deepseek_v41_image_inputs,
    _deepseek_v41_image_grid,
    _get_deepseek_v41_cache_ownership,
    _get_deepseek_v41_compressed_cache_info,
    _get_deepseek_v41_engram_state,
    _get_deepseek_v41_state,
    _is_deepseek_v41_config,
    _seed_deepseek_v41_draft_windows,
    get_sparse_attention_indexer_cache_info,
    reset_deepseek_v41_state,
    generate_inputs,
    RequestInfo,
)
from tensor_cast.layers.deepseek_v4 import DeepseekV4SparseAttention
from tensor_cast.layers.deepseek_v41 import (
    DeepseekV41SharedAttentionState,
    DeepseekV41SparseAttention as DeepseekV41AttentionWrapper,
    EngramHashState,
    deepseek_v41_kv_owner,
    resolve_csa2_layer_role,
)
from tensor_cast.layers.deepseek_v41_dspark import (
    DeepseekV41DsparkAttention,
    DeepseekV41DsparkBlock,
    DeepseekV41NativeDsparkDraft,
    build_deepseek_v41_native_dspark,
)
from tensor_cast.layers.mla import MultiheadLatentAttentionBase
from tensor_cast.layers.moe_layer import MoELayer
from tensor_cast.layers.quant_linear import QuantLinearBase, TensorCastQuantLinear
from tensor_cast.layers.sampler import SamplingMetadata
from tensor_cast.layers.utils import ModelWrapperBase
from tensor_cast.model_config import DsparkConfig, MlaConfig, MoEConfig, ModelConfig, ParallelConfig, QuantConfig
from tensor_cast.performance_model.op_invoke_info import OpInvokeInfo
from tensor_cast.quantize_utils import LinearQuantType
from tensor_cast.transformers.builtin_model.deepseek_v4 import (
    make_v41_identity_pre_mix,
    validate_precomputed_engram_hash_ids,
)
from tensor_cast.transformers.builtin_model.deepseek_v41 import (
    DeepseekV41Aligner,
    DeepseekV41Config,
    DeepseekV41DecoderLayer,
    DeepseekV41Model,
    DeepseekV41SparseAttention,
    DeepseekV41TextConfig,
    DeepseekV41VisionConfig,
    DeepseekV41VisionModel,
    route_deepseek_v41_gate,
)
from tensor_cast.transformers.custom_model_registry import MoeExpertMLP, get_model_profile
from tensor_cast.transformers.transformations import _patch_moe_expert_helper, _quantize_deepseek_v41_native_dspark
from tensor_cast.transformers.utils import AutoModelConfigLoader
from tests.helpers.model_assets import vendored_model_config_path


CHECKPOINT = Path(vendored_model_config_path("deepseek-ai/DeepSeek-V4.1-Flash"))


def _tiny_config():
    config = DeepseekV41Config.from_pretrained(
        vendored_model_config_path("deepseek-ai/DeepSeek-V4.1-Flash")
    ).get_text_config()
    for name, value in {
        "hidden_size": 32,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 32,
        "qk_rope_head_dim": 8,
        "q_lora_rank": 16,
        "o_groups": 1,
        "o_lora_rank": 8,
        "vocab_size": 64,
        "intermediate_size": 64,
        "moe_intermediate_size": 16,
        "n_routed_experts": 4,
        "num_experts_per_tok": 2,
        "n_group": 1,
        "topk_group": 1,
        "dspark_n_routed_experts": 4,
        "dspark_num_experts_per_tok": 2,
        "index_head_dim": 32,
        "index_n_heads": 2,
        "index_topk": 4,
        "sliding_window": 8,
    }.items():
        setattr(config, name, value)
    return config


class _Target(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.calls = []

    def forward(self, input_ids, position_ids, inputs_embeds=None, **kwargs):
        self.calls.append(kwargs)
        hidden = self.embed_tokens(input_ids)
        return self.lm_head(hidden), None, [hidden] * 3


def _local_mla_init(self, mla_config, inner, tp_group, decode_only=False, parallel_group_manager=None):
    # Projection sharding belongs to the MLA tests; retain the real V4/V4.1 forwards here.
    MultiheadLatentAttentionBase.__init__(self, mla_config, inner, decode_only)


def _index_cost(batch, start_pos, uses_candidates=False, dtype=torch.int4):
    q = torch.empty(batch, 1, 2, 32, device="meta")
    key = torch.empty(batch, 128, 32, dtype=dtype, device="meta")
    args = (
        q,
        None,
        torch.empty(batch, 1, 2, device="meta"),
        key,
        None,
        4,
        2,
        start_pos,
        torch.empty(batch, 1, 1, dtype=torch.int32, device="meta") if uses_candidates else None,
        8,
        1,
        False,
        uses_candidates,
    )
    output = torch.ops.tensor_cast.csa2_index_select(*args)
    return OpInvokeInfo(torch.ops.tensor_cast.csa2_index_select.default, args, {}, output).get_perf_properties()


class _IdentityAttention(nn.Module):
    def forward(self, hidden_states, **kwargs):
        self.attention_input = hidden_states
        return hidden_states, None


class _IdentityMlp(nn.Module):
    def forward(self, hidden_states, input_ids=None):
        return hidden_states


class _RecordOps(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.ops = []
        self.mean_dimensions = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.ops.append(str(func))
        if str(func) == "aten.mean.dim":
            self.mean_dimensions.append(tuple(args[1]))
        return func(*args, **(kwargs or {}))


# Configuration and loading


def test_profile_is_v41_specific_and_does_not_fallback_to_v4() -> None:
    profile = get_model_profile("deepseek_v41")
    assert profile is not None
    assert profile.model_type == "deepseek_v41"
    assert profile.mla_module_name == "DeepseekV41SparseAttention"
    assert profile.mla_module_class_type.__name__ == "DeepseekV41SparseAttention"
    assert profile.visual_module_path == "vision"
    assert profile.visual_layers_path_str == "vision.blocks"
    assert profile.visual_merger_linear_mapping["aligner.w1"] == "colwise"
    assert get_model_profile("deepseek_v4") is not profile


def test_v41_root_config_normalizes_nested_dicts() -> None:
    raw = json.loads((CHECKPOINT / "config.json").read_text(encoding="utf-8"))
    config = DeepseekV41Config.from_dict(raw)
    assert isinstance(config.get_text_config(), DeepseekV41TextConfig)
    assert config.get_text_config().rms_norm_eps == pytest.approx(1e-20)
    assert config.get_text_config().index_topk == 512


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("compress_ratios", [0] * 39, "at least one entry"),
        ("compress_ratios", [0, 3] + [1] * 38, "unsupported values"),
        ("kv_source_layer_ids", [2, 8, 14], "exactly 4"),
        ("index_source_layer_ids", [2] * 8, "duplicate"),
        ("candidate_source_layer_id", 40, "outside"),
    ],
)
def test_v41_config_rejects_invalid_structure(field, value, message) -> None:
    kwargs = {
        "hidden_size": 128,
        "num_hidden_layers": 40,
        "num_attention_heads": 8,
        "num_key_value_heads": 1,
        "compress_ratios": [0] * 40,
        field: value,
    }
    with pytest.raises(ValueError, match=message):
        DeepseekV41TextConfig(**kwargs)


def test_missing_ratios_in_real_nested_config_fails_early() -> None:
    with pytest.raises(ValueError, match="requires compress_ratios"):
        DeepseekV41Config(
            text_config={
                "model_type": "deepseek_v41_text",
                "hidden_size": 128,
                "num_hidden_layers": 40,
                "num_attention_heads": 8,
                "num_key_value_heads": 1,
            }
        )


def test_model_shell_builds_exact_active_layer_count() -> None:
    config = DeepseekV41Config(
        hidden_size=64,
        num_hidden_layers=40,
        num_attention_heads=4,
        num_key_value_heads=1,
        vocab_size=128,
        intermediate_size=128,
        moe_intermediate_size=16,
        n_routed_experts=2,
        n_shared_experts=1,
        num_experts_per_tok=1,
        q_lora_rank=16,
        head_dim=16,
        qk_rope_head_dim=4,
        o_lora_rank=8,
        o_groups=1,
        compress_ratios=[0, 1] + [2] * 18 + [1] * 20,
    )
    model = DeepseekV41Model(config)

    assert len(model.layers) == 40
    assert all(isinstance(layer.self_attn, DeepseekV41SparseAttention) for layer in model.layers)
    assert [layer.self_attn.compress_ratio for layer in model.layers] == config.compress_ratios
    assert all(layer.mlp.__class__.__name__ == "DeepseekV41MoE" for layer in model.layers)


def test_text_config_is_recognized_as_deepseek_v41() -> None:
    assert _is_deepseek_v41_config(SimpleNamespace(model_type="deepseek_v41"))
    assert _is_deepseek_v41_config(SimpleNamespace(model_type="deepseek_v41_text"))
    assert not _is_deepseek_v41_config(SimpleNamespace(model_type="glm"))


def test_v41_remote_raw_config_fallback_avoids_remote_code():
    raw = DeepseekV41Config.from_dict({"text_config": {"compress_ratios": [0] * 40}}).to_dict()
    with patch.object(AutoConfig, "from_pretrained", side_effect=ValueError("unknown model type")) as probe:
        with patch.object(PretrainedConfig, "get_config_dict", return_value=(raw, {})) as reader:
            config = AutoModelConfigLoader().load_config("review-fixture/DeepSeek-V4.1-Flash")
    assert config.model_type == "deepseek_v41"
    assert config.num_hidden_layers == 40
    reader.assert_called_once()
    probe.assert_called_once_with("review-fixture/DeepSeek-V4.1-Flash", trust_remote_code=False)


# CSA2 state, attention and performance


def test_csa2_query_quant_uses_fixed_mxfp4_payload_and_scale() -> None:
    query = torch.empty(2, 3, 4, 64, device="meta", dtype=torch.bfloat16)
    payload, scale = torch.ops.tensor_cast.csa2_query_quant(query, 32)

    assert payload.shape == query.shape
    assert payload.dtype == torch.int4
    assert scale.shape == (2, 3, 4, 2)
    assert scale.dtype == torch.float8_e8m0fnu

    with pytest.raises(ValueError, match="fixed MXFP4 group size 32"):
        torch.ops.tensor_cast.csa2_query_quant(query, 16)


def test_csa2_roles_preserve_physical_ownership() -> None:
    raw = json.loads((CHECKPOINT / "config.json").read_text(encoding="utf-8"))
    config = DeepseekV41Config.from_dict(raw).get_text_config()

    assert resolve_csa2_layer_role(config, 2).role == "source"
    assert resolve_csa2_layer_role(config, 3).role == "reuse"
    assert resolve_csa2_layer_role(config, 3).compressed_kv_owner == 2
    assert resolve_csa2_layer_role(config, 3).topk_owner == 2
    assert resolve_csa2_layer_role(config, 20).produces_candidate_pool
    assert resolve_csa2_layer_role(config, 24).role == "reindex"
    assert resolve_csa2_layer_role(config, 24).compressed_kv_owner == 20
    assert resolve_csa2_layer_role(config, 24).index_k_owner == 20


def test_shared_state_is_request_local_and_requires_publish() -> None:
    state = DeepseekV41SharedAttentionState()
    with pytest.raises(RuntimeError, match="has not published"):
        state.require_compressed_kv(2)
    value = object()
    state.compressed_kv_by_source[2] = value
    assert state.require_compressed_kv(2) is value
    state.reset()
    assert state.compressed_kv_by_source == {}


def test_cache_ownership_metadata_uses_absolute_layer_ids() -> None:
    raw = json.loads((CHECKPOINT / "config.json").read_text(encoding="utf-8"))
    config = DeepseekV41Config.from_dict(raw)

    class Model:
        text_config = config.get_text_config()

    ownership = _get_deepseek_v41_cache_ownership(Model())
    assert ownership["compressed_kv_owner_by_layer"][3] == 2
    assert ownership["compressed_kv_owner_by_layer"][24] == 20
    assert ownership["index_k_owner_by_layer"][24] == 20


def test_compressed_cache_uses_packed_batch_token_total_once() -> None:
    config = DeepseekV41Config.from_dict(
        json.loads((CHECKPOINT / "config.json").read_text(encoding="utf-8"))
    ).get_text_config()
    model = SimpleNamespace(text_config=config, head_dim=config.head_dim)
    caches, scales, _ = _get_deepseek_v41_compressed_cache_info(
        model, block_size=16, batch_size=4, total_kv_tokens=4096
    )
    source = int(config.kv_source_layer_ids[0])
    ratio = int(config.compress_ratios[source])
    expected_blocks = (4096 // ratio + 15) // 16
    assert caches[source].shape[0] == max(expected_blocks, 1)
    assert scales[source].shape[:2] == caches[source].shape[:2]


@pytest.mark.parametrize("op_name", ["csa2_compressor", "csa2_compressor_v2"])
@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("token_local", [False, True])
def test_compressor_projection_cost_matches_standalone_gemms(op_name, ratio, token_local):
    batch, seq_len, hidden_dim, head_dim = 2, 6, 32, 16
    hidden_shape = (batch * seq_len, hidden_dim) if token_local else (batch, seq_len, hidden_dim)
    hidden = torch.empty(hidden_shape, dtype=torch.float16, device="meta")
    cache = torch.empty(batch, seq_len, head_dim, device="meta")
    lengths = torch.full((batch,), seq_len, dtype=torch.int64)
    args = (hidden, cache, ratio, head_dim, 4, False, lengths, lengths)
    op = getattr(torch.ops.tensor_cast, op_name).default
    props = OpInvokeInfo(op, args, {}, op(*args)).get_perf_properties()

    # Compare the semantic compressor with its one/two actual FP32 GEMMs.
    matrix = torch.empty(batch * seq_len, hidden_dim, dtype=torch.float32, device="meta")
    weight = torch.empty(hidden_dim, head_dim, dtype=torch.float32, device="meta")
    projection = torch.mm(matrix, weight)
    gemm_props = OpInvokeInfo(torch.ops.aten.mm.default, (matrix, weight), {}, projection).get_perf_properties()
    projection_count = 2 if ratio == 2 else 1
    assert props.compute_ops[torch.float32].mma_ops == (
        projection_count * gemm_props.compute_ops[torch.float32].mma_ops
    )
    expected_pooling = batch * (seq_len // ratio) * ratio * head_dim * 3 if ratio == 2 else 0
    assert props.compute_ops[torch.float32].gp_ops == expected_pooling


def test_r2_decode_tail_shape_follows_absolute_position() -> None:
    import torch

    hidden = torch.empty(1, 1, 16, device="meta")
    cache = torch.empty(1, 4, 16, device="meta")
    pending = torch.empty(1, 1, 16, device="meta")
    scores = torch.empty(1, 1, device="meta")

    odd = torch.ops.tensor_cast.csa2_compressor(hidden, cache, 2, 16, 4, False, None, None, pending, scores, 128)
    even = torch.ops.tensor_cast.csa2_compressor(hidden, cache, 2, 16, 4, False, None, None, pending, scores, 129)
    assert odd[0].shape[1] == 0 and odd[2].shape[1] == 1
    assert even[0].shape[1] == 1 and even[2].shape[1] == 0


def test_r2_decode_tail_carries_values_between_calls() -> None:
    import torch

    cache = torch.empty(1, 4, 4)
    first = torch.ones(1, 1, 4)
    pending = torch.empty(1, 0, 4)
    scores = torch.empty(1, 0)
    odd = torch.ops.tensor_cast.csa2_compressor(first, cache, 2, 4, 2, False, None, None, pending, scores, 128)
    even = torch.ops.tensor_cast.csa2_compressor(
        torch.full((1, 1, 4), 3.0), cache, 2, 4, 2, False, None, None, odd[2], odd[3], 129
    )
    assert odd[0].shape[1] == 0 and even[0].shape[1] == 1
    assert torch.isfinite(even[0]).all()
    torch.testing.assert_close(odd[2], first)
    combined = torch.ops.tensor_cast.csa2_compressor(
        torch.cat((first, torch.full_like(first, 3.0)), dim=1), cache, 2, 4, 2, False, None, None, pending, scores, 128
    )
    torch.testing.assert_close(even[0], combined[0])


def test_gather_attention_keeps_window_and_compressed_streams_separate() -> None:
    import torch

    q = torch.empty(1, 2, 4, 8, device="meta")
    window = torch.empty(1, 8, 8, device="meta")
    compressed = torch.empty(1, 4, 8, device="meta")
    topk = torch.empty(1, 2, 6, dtype=torch.int32, device="meta")
    output = torch.ops.tensor_cast.sparse_index_gather_attention(
        q, window, compressed, topk, torch.empty(4, device="meta"), 0.1, 8
    )
    assert output.shape == (1, 2, 4, 8)
    assert output.dtype == torch.bfloat16

    fp16_output = torch.ops.tensor_cast.sparse_index_gather_attention(
        q,
        window,
        compressed,
        topk,
        torch.empty(4, device="meta"),
        0.1,
        8,
        None,
        None,
        None,
        None,
        None,
        16,
        -1,
        torch.float16,
    )
    assert fp16_output.dtype == torch.float16


def test_fp4_cache_payload_and_scale_contract() -> None:
    import torch

    x = torch.empty(2, 3, 34, device="meta")
    payload, scale = torch.ops.tensor_cast.fp4_kv_quantize(x, 16)
    assert payload.shape == x.shape
    assert payload.dtype == torch.int4
    assert scale.shape == (2, 3, 3)
    assert scale.dtype == torch.float8_e4m3fn
    restored = torch.ops.tensor_cast.fp4_kv_dequantize(payload, scale, 16)
    assert restored.shape == x.shape


def test_indexer_accepts_fp4_index_scale_stream() -> None:
    import torch

    q = torch.empty(1, 2, 4, 32, device="meta")
    index_k = torch.empty(1, 8, 32, dtype=torch.int4, device="meta")
    index_scale = torch.empty(1, 8, 1, dtype=torch.float8_e8m0fnu, device="meta")
    topk = torch.ops.tensor_cast.csa2_indexer(q, index_k, 4, None, 6, index_scale, 32)
    assert topk.shape == (1, 2, 4)
    assert topk.dtype == torch.int32


def test_compressed_rope_preserves_shape_and_charges_gp_work() -> None:
    from tensor_cast.performance_model.op_invoke_info import OpInvokeInfo

    latent = torch.empty(2, 4, 512, dtype=torch.bfloat16, device="meta")
    rotated = torch.ops.tensor_cast.csa2_compressed_rope(latent, 64, 2, 0, 160000.0)
    assert rotated.shape == latent.shape
    info = OpInvokeInfo(
        torch.ops.tensor_cast.csa2_compressed_rope.default,
        (latent, 64, 2, 0, 160000.0),
        {},
        rotated,
    )
    assert info.get_perf_properties().compute_ops[torch.bfloat16].gp_ops == 2 * 4 * 64 * 6


def test_indexer_properties_include_query_fp4_and_owned_index_k_work() -> None:
    from tensor_cast.performance_model.op_invoke_info import OpInvokeInfo

    q = torch.empty(1, 4, 2, 128, dtype=torch.bfloat16, device="meta")
    index_k = torch.empty(1, 8, 128, dtype=torch.int4, device="meta")
    index_scale = torch.empty(1, 8, 4, dtype=torch.float8_e8m0fnu, device="meta")
    output = torch.empty(1, 4, 4, dtype=torch.int32, device="meta")
    info = OpInvokeInfo(
        torch.ops.tensor_cast.csa2_indexer.default,
        (q, index_k, 4, None, 8, index_scale, 32, 32, 2, 64, 160000.0, 0, True, 512),
        {},
        output,
    )
    props = info.get_perf_properties()
    assert props.memory_readwrite_bytes > 0
    assert props.memory_write_bytes > 0
    assert props.compute_ops[torch.float32].gp_ops > 0
    assert props.compute_ops[torch.bfloat16].mma_ops > 0


def test_fp4_analytic_properties_include_scale_and_dequant_cost() -> None:
    import torch

    from tensor_cast.performance_model.op_invoke_info import OpInvokeInfo

    x = torch.empty(2, 4, 32, device="meta")
    payload, scale = torch.ops.tensor_cast.fp4_kv_quantize(x, 16)
    info = OpInvokeInfo(torch.ops.tensor_cast.fp4_kv_quantize.default, (x, 16), {}, (payload, scale))
    props = info.get_perf_properties()
    assert props.compute_ops[torch.float32].gp_ops > 0

    q = torch.empty(1, 1, 2, 32, device="meta")
    window = torch.empty(1, 8, 32, device="meta")
    indices = torch.empty(1, 1, 6, dtype=torch.int32, device="meta")
    gather_out = torch.empty(1, 1, 2, 32, device="meta")
    gather = OpInvokeInfo(
        torch.ops.tensor_cast.sparse_index_gather_attention.default,
        (q, window, payload, indices, torch.empty(2, device="meta"), 0.1, 32, 4, 2, None, None, scale, 16),
        {},
        gather_out,
    )
    gather_props = gather.get_perf_properties()
    assert gather_props.compute_ops[torch.float32].gp_ops > 0


def test_fp4_gather_reads_packed_half_byte_payload() -> None:
    from tensor_cast.performance_model.op_invoke_info import OpInvokeInfo

    q = torch.empty(1, 1, 1, 32, device="meta")
    window = torch.empty(1, 1, 32, device="meta")
    payload = torch.empty(1, 4, 32, dtype=torch.int4, device="meta")
    indices = torch.empty(1, 1, 4, dtype=torch.int32, device="meta")
    output = torch.empty(1, 1, 1, 32, device="meta")
    gather = OpInvokeInfo(
        torch.ops.tensor_cast.sparse_index_gather_attention.default,
        (q, window, payload, indices, torch.empty(1, device="meta"), 0.1, 32, 0, 4),
        {},
        output,
    )

    props = gather.get_perf_properties()
    # Four selected 32-wide FP4 rows occupy 4 * 32 / 2 = 64 bytes.
    baseline = OpInvokeInfo(
        torch.ops.tensor_cast.sparse_index_gather_attention.default,
        (q, window, payload, indices, torch.empty(1, device="meta"), 0.1, 32, 0, 0),
        {},
        output,
    ).get_perf_properties()
    assert props.memory_read_bytes - baseline.memory_read_bytes == 64


def test_l39_reuse_gather_accounts_for_compressed_reads_without_payload_handle() -> None:
    import torch

    from tensor_cast.performance_model.op_invoke_info import OpInvokeInfo

    q = torch.empty(1, 1, 2, 32, device="meta")
    window = torch.empty(1, 128, 32, device="meta")
    indices = torch.empty(1, 1, 640, dtype=torch.int32, device="meta")
    output = torch.empty(1, 1, 2, 32, device="meta")
    gather = OpInvokeInfo(
        torch.ops.tensor_cast.sparse_index_gather_attention.default,
        (
            q,
            window,
            None,
            indices,
            torch.empty(2, device="meta"),
            0.1,
            32,
            128,
            None,
            None,
            None,
            None,
            16,
            39,
        ),
        {},
        output,
    )
    props = gather.get_perf_properties()
    compressed_slots = 512
    payload_and_scale_bytes = compressed_slots * (16 + 2)
    assert props.memory_read_bytes >= payload_and_scale_bytes
    assert props.compute_ops[torch.float32].gp_ops > 0


def test_csa2_state_publish_append_read_and_reset():
    state = DeepseekV41SharedAttentionState()
    other = DeepseekV41SharedAttentionState()
    kv = torch.ones(1, 2, 32)
    state.publish_compressed_kv(2, kv)
    state.publish_compressed_kv(2, kv)
    assert state.require_compressed_kv(2) is kv
    state.publish_compressed_kv(2, torch.full((1, 1, 32), 3.0))
    torch.testing.assert_close(state.require_compressed_kv(2)[:, -1], torch.full((1, 32), 3.0))
    assert state.require_compressed_kv(2).shape == (1, 3, 32)
    state.publish_compressed_kv_scale(2, torch.ones(1, 2, 2))
    state.publish_compressed_kv_scale(2, torch.ones(1, 1, 2))
    assert state.compressed_kv_scale_by_source[2].shape == (1, 3, 2)
    slots = torch.tensor([0, 2])
    topk = torch.tensor([[[0, 1]]])
    state.publish_compressed_slot_mapping(2, slots)
    state.publish_topk(2, topk)
    assert state.require_compressed_slot_mapping(2) is slots
    assert state.require_topk(2) is topk
    tail = state.get_compressor_tail(2)
    assert tail.pending_tokens is None
    pending, scores = torch.ones(1, 1, 32), torch.ones(1, 1)
    state.update_compressor_tail(2, pending, scores)
    assert state.get_compressor_tail(2).pending_tokens is pending
    assert state.get_compressor_tail(2).pending_scores is scores
    with TestCase().assertRaisesRegex(RuntimeError, "missing"):
        other.require_compressed_slot_mapping(2)
    with TestCase().assertRaisesRegex(RuntimeError, "has not published"):
        other.require_topk(2)
    state.reset()
    assert not state.compressed_kv_by_source
    assert not state.compressed_kv_scale_by_source
    assert not state.compressed_slot_by_source
    assert not state.topk_indices_by_source
    assert not state.compressor_tail_by_source


def test_csa2_meta_publication_preserves_packed_dimensions():
    state = DeepseekV41SharedAttentionState()
    state.publish_compressed_kv(8, torch.empty(2, 3, 32, device="meta"))
    state.publish_compressed_kv(8, torch.empty(2, 2, 32, device="meta"))
    assert state.require_compressed_kv(8).shape == (2, 5, 32)
    config = _tiny_config()
    assert resolve_csa2_layer_role(config, 2).owns_index_k
    assert not resolve_csa2_layer_role(config, 3).owns_index_k
    assert deepseek_v41_kv_owner(config, 0) is None
    assert deepseek_v41_kv_owner(config, 3) == 2
    assert deepseek_v41_kv_owner(config, 24) == 20


def test_request_state_retains_prefill_and_decode_until_reset():
    requests = [SimpleNamespace(csa2_state=None, is_decode=False) for _ in range(2)]
    state = _get_deepseek_v41_state(None, requests)
    assert all(request.csa2_state is state for request in requests)
    state.publish_topk(2, torch.tensor([0]))
    for request in requests:
        request.is_decode = True
    assert _get_deepseek_v41_state(None, requests) is state
    requests[1].csa2_state = DeepseekV41SharedAttentionState()
    with TestCase().assertRaisesRegex(ValueError, "share one workload state"):
        _get_deepseek_v41_state(None, requests)
    requests[1].csa2_state = state
    reset_deepseek_v41_state(requests=requests)
    assert all(request.csa2_state is None for request in requests)
    assert not state.topk_indices_by_source
    replacement = _get_deepseek_v41_state(None, requests)
    assert replacement is not state
    requests[0].is_decode = False
    assert _get_deepseek_v41_state(None, requests) is replacement
    assert _get_deepseek_v41_state(None, []) is not _get_deepseek_v41_state(None, [])
    history = torch.tensor([[1, 2, 3]])
    engram = EngramHashState()
    engram.update(history)
    assert engram.history is history
    engram.update(history + 1)
    torch.testing.assert_close(engram.history, torch.tensor([[2, 3, 4]]))


def test_csa2_cache_update_respects_valid_slots_without_mutating_input():
    cache = torch.zeros(1, 4, 2)
    values = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
    result = torch.ops.tensor_cast.csa2_cache_update(cache, values, torch.tensor([2, -1, 9]))
    torch.testing.assert_close(result[0, 2], values[0, 0])
    assert result.sum() == 3
    assert cache.count_nonzero() == 0
    contiguous = torch.ops.tensor_cast.csa2_cache_update(cache, values, None)
    torch.testing.assert_close(contiguous[0, :3], values[0])
    empty = torch.ops.tensor_cast.csa2_cache_update(cache, values[:, :0], None)
    torch.testing.assert_close(empty, cache)


def test_csa2_selection_obeys_causality_candidates_and_invalid_positions():
    q = torch.ones(1, 1, 1, 2)
    weights = torch.ones(1, 1, 1)
    keys = torch.tensor([[[1.0, 1.0], [4.0, 4.0], [2.0, 2.0], [9.0, 9.0]]])
    select = torch.ops.tensor_cast.csa2_index_select
    selected, candidates = select(q, None, weights, keys, None, 2, 1, 2, None, 2, 2, True, False)
    assert set(selected.flatten().tolist()) == {1, 2}
    assert set(candidates.flatten().tolist()) == {0, 1}
    masked, _ = select(q, None, weights, keys, None, 2, 1, 2, torch.tensor([[[0]]]), 2, 2, False, True)
    assert set(masked.flatten().tolist()) == {0, 1}
    invalid, _ = select(q, None, weights, keys, None, 2, 2, 0, None, 2, 2, False, False)
    assert (invalid == -1).all()
    with TestCase().assertRaisesRegex(ValueError, "candidate block IDs"):
        select(q, None, weights, keys, None, 2, 1, 2, None, 2, 2, False, True)
    shifted = torch.ops.tensor_cast.csa2_prepare_indices(torch.tensor([[-1, 0, 2]]), torch.tensor([2]), 8)
    torch.testing.assert_close(shifted, torch.tensor([[-1, 8, 10]], dtype=torch.int32))
    mask = torch.ops.tensor_cast.csa2_select_candidate_blocks(
        torch.tensor([[[10.0], [9.0], [1.0], [1.0], [2.0]]]), 2, 1
    )
    torch.testing.assert_close(mask, torch.tensor([[True, True, False, False, True]]))


def test_v41_attention_wrapper_executes_window_and_guards_missing_owner():
    config = _tiny_config()
    with torch.device("meta"):
        inner = DeepseekV41SparseAttention(config, 0)
        with patch("tensor_cast.layers.mla.MultiheadLatentAttentionTensorCast.__init__", _local_mla_init):
            wrapper = DeepseekV41AttentionWrapper(
                MlaConfig(module_name="DeepseekV41SparseAttention"), inner, SimpleNamespace(world_size=1)
            )
        hidden = torch.empty(1, 8, config.hidden_size)
        rope = (torch.empty(8, 8), torch.empty(8, 8))
        metadata = SimpleNamespace(query_lens=torch.tensor([8]), seq_lens=torch.tensor([8]), slot_mapping=None)
        output, _ = wrapper(hidden, rope, None, attention_meta=metadata, csa2_state=DeepseekV41SharedAttentionState())
        assert output.shape == hidden.shape
        wrapper.csa2_role = resolve_csa2_layer_role(config, 3)
        with TestCase().assertRaisesRegex(RuntimeError, "owner L2 has not published"):
            wrapper(hidden, rope, None, csa2_state=DeepseekV41SharedAttentionState())


def test_v41_source_publishes_both_quantized_cache_planes_and_tail():
    config = _tiny_config()
    with torch.device("meta"):
        inner = DeepseekV41SparseAttention(config, 2)
        attention = SimpleNamespace(_inner=inner, compress_ratio=2, _head_dim=32, _qk_rope_head_dim=8)
        state = DeepseekV41SharedAttentionState()
        kv = torch.empty(1, 4, 32, dtype=torch.int4)
        scale = torch.empty(1, 4, 2, dtype=torch.float8_e4m3fn)
        index = torch.empty(1, 4, 32, dtype=torch.int4)
        index_scale = torch.empty(1, 4, 1, dtype=torch.float8_e8m0fnu)
        result = DeepseekV4SparseAttention._publish_v41_source(
            attention,
            torch.empty(1, 2, 32),
            kv,
            scale,
            index,
            index_scale,
            SimpleNamespace(start_pos=0, compressed_slot_mapping=torch.tensor([0])),
            state,
            2,
        )
        assert tuple(value.shape for value in result) == (kv.shape, scale.shape, index.shape, index_scale.shape)
        assert state.require_compressed_kv(2) is result[0]
        assert state.index_k_by_source[2] is result[2]
        assert state.compressed_kv_scale_by_source[2] is result[1]
        assert state.index_k_scale_by_source[2] is result[3]
        assert state.get_compressor_tail(2).pending_tokens.shape[1] == 0


def test_v41_default_layers_aliases_and_explicit_precedence():
    config = DeepseekV41TextConfig(compress_ratios=[0] * 40, score_func="sigmoid", route_scale=2.0)
    assert config.num_hidden_layers == len(config.active_backbone_layer_ids) == len(config.compress_ratios) == 40
    assert config.scoring_func == config.score_func == "sigmoid"
    assert config.route_scale == config.routed_scaling_factor == 2.0
    override = DeepseekV41TextConfig(
        compress_ratios=[0] * 40,
        score_func="sigmoid",
        scoring_func="sqrtsoftplus",
        route_scale=2.0,
        routed_scaling_factor=1.5,
    )
    assert override.score_func == "sqrtsoftplus" and override.route_scale == 1.5
    for root in (DeepseekV41Config(text_config=config), DeepseekV41Config(text_config=config.to_dict())):
        assert root.num_hidden_layers == root.get_text_config().num_hidden_layers == 40
        assert root.route_scale == root.get_text_config().route_scale == 2.0


def test_fp4_eager_multigroup_and_padding_preserve_shape():
    for width in (32, 64, 65):
        x = torch.arange(2 * 3 * width, dtype=torch.float32).reshape(2, 3, width)
        payload, scales = torch.ops.tensor_cast.fp4_kv_quantize(x, 32)
        assert payload.shape == x.shape and payload.dtype == torch.int4
        assert scales.shape == (2, 3, (width + 31) // 32)
        assert torch.isfinite(scales.float()).all()


def test_legacy_indexer_uses_absolute_causal_positions():
    keys = torch.arange(1, 5, dtype=torch.float32).reshape(1, 4, 1)
    decode = torch.ops.tensor_cast.csa2_indexer(
        torch.ones(1, 1, 1, 1), keys, 4, None, 4, None, 32, 32, 2, 0, 160000.0, 7
    )
    assert set(decode.flatten().tolist()) == {0, 1, 2, 3}
    prefill = torch.ops.tensor_cast.csa2_indexer(
        torch.ones(1, 4, 1, 1), keys, 4, None, 4, None, 32, 32, 2, 0, 160000.0, 0
    )
    assert (prefill[0, 0] == -1).all()
    assert set(prefill[0, 1].tolist()) == {-1, 0}
    assert set(prefill[0, 3].tolist()) == {-1, 0, 1}


def test_gather_with_empty_compressed_cache_keeps_window_result():
    q = torch.ones(1, 1, 1, 2)
    window = torch.tensor([[[2.0, 3.0]]])
    sink = torch.tensor([0.0])
    indices = torch.tensor([[[0, -1]]], dtype=torch.int32)
    actual = torch.ops.tensor_cast.sparse_index_gather_attention(
        q, window, torch.empty(1, 0, 2), indices, sink, 0.5, 2, out_dtype=torch.float32
    )
    expected = torch.ops.tensor_cast.sparse_index_gather_attention(
        q, window, None, indices, sink, 0.5, 2, out_dtype=torch.float32
    )
    torch.testing.assert_close(actual, expected)
    masked = torch.ops.tensor_cast.sparse_index_gather_attention(
        q,
        window,
        torch.ones(1, 1, 2),
        indices,
        sink,
        0.5,
        2,
        candidate_mask=torch.empty(1, 0, dtype=torch.bool),
        out_dtype=torch.float32,
    )
    torch.testing.assert_close(masked, expected)


def test_index_cost_tracks_context_candidates_batch_and_fp4_bytes():
    short = _index_cost(1, 1)
    long = _index_cost(1, 63)
    limited = _index_cost(1, 63, True)
    assert short.compute_ops[torch.float32].mma_ops < limited.compute_ops[torch.float32].mma_ops
    assert limited.compute_ops[torch.float32].mma_ops < long.compute_ops[torch.float32].mma_ops
    assert _index_cost(2, 63).compute_ops[torch.float32].mma_ops == 2 * long.compute_ops[torch.float32].mma_ops
    fp8 = _index_cost(1, 63, dtype=torch.float8_e4m3fn)
    assert fp8.memory_read_bytes - long.memory_read_bytes == 32 * 32 / 2
    writes = []
    for dtype in (torch.int4, torch.float8_e4m3fn):
        cache = torch.empty(1, 128, 32, dtype=dtype, device="meta")
        update = torch.empty(1, 2, 32, dtype=dtype, device="meta")
        output = torch.ops.tensor_cast.csa2_cache_update(cache, update)
        props = OpInvokeInfo(
            torch.ops.tensor_cast.csa2_cache_update.default, (cache, update), {}, output
        ).get_perf_properties()
        writes.append(props.memory_write_bytes)
    assert writes == [32, 64]


def test_gather_scalar_lengths_match_tensor_lengths():
    q = torch.empty(1, 1, 1, 32, device="meta")
    window = torch.empty(1, 128, 32, device="meta")
    compressed = torch.empty(1, 128, 32, dtype=torch.int4, device="meta")
    indices = torch.empty(1, 1, 128, dtype=torch.int32, device="meta")
    base = (q, window, compressed, indices, torch.empty(1, device="meta"), 0.1, 32)
    output = torch.empty_like(q)

    def props(lengths):
        return OpInvokeInfo(
            torch.ops.tensor_cast.sparse_index_gather_attention.default, base + lengths, {}, output
        ).get_perf_properties()

    scalar = props((1, 2))
    tensor = props((torch.tensor([1]), torch.tensor([2])))
    full = props((None, None))
    # Tensor lengths add two int64 metadata reads; the KV work must be identical.
    assert scalar.memory_read_bytes + 16 == tensor.memory_read_bytes < full.memory_read_bytes
    assert scalar.compute_ops[torch.float32].mma_ops == tensor.compute_ops[torch.float32].mma_ops
    assert scalar.compute_ops[torch.float32].mma_ops < full.compute_ops[torch.float32].mma_ops


def test_chunked_prefill_preserves_compressor_tail_until_explicit_reset():
    request = SimpleNamespace(csa2_state=None, is_decode=False)
    state = _get_deepseek_v41_state(None, [request])
    first = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    second = torch.tensor([[[4.0, 3.0, 2.0, 1.0]]])
    cache = torch.zeros(1, 4, 4)
    odd = torch.ops.tensor_cast.csa2_compressor(first, cache, 2, 4, 2, False)
    state.update_compressor_tail(2, odd[2], odd[3])
    continued = _get_deepseek_v41_state(None, [request])
    assert continued is state
    tail = continued.get_compressor_tail(2)
    even = torch.ops.tensor_cast.csa2_compressor(
        second, cache, 2, 4, 2, False, None, None, tail.pending_tokens, tail.pending_scores, 1
    )
    whole = torch.ops.tensor_cast.csa2_compressor(torch.cat((first, second), 1), cache, 2, 4, 2, False)
    torch.testing.assert_close(even[0], whole[0])


def test_v4_indexer_bytes_are_per_original_token_not_compressed_slot():
    model = MagicMock()
    model.num_hidden_layers = 1
    model.model_config.hf_config = SimpleNamespace(model_type="deepseek_v4")
    model.model_config.dtype = torch.bfloat16
    model.model_config.mla_config = SimpleNamespace(mla_cls=DeepseekV4SparseAttention)
    attention = SimpleNamespace(use_indexer=True, _index_head_dim=32, indexer=None, compress_ratio=4)
    model.unwrap.return_value = SimpleNamespace(layers=[SimpleNamespace(self_attn=attention)])
    with patch("tensor_cast.core.input_generator.get_attention_quant_config", return_value=None):
        info = get_sparse_attention_indexer_cache_info(model, 16, 16, batch_size=1, total_kv_tokens=256)
    assert info["indexer_cache_by_layers"][0].shape == (4, 16, 32)
    assert info["indexer_cache_per_token"] == 32 * 2 / 4


def test_standalone_meta_decode_seeds_once_and_retains_request_windows():
    from tensor_cast.layers.deepseek_v41 import DeepseekV41SharedAttentionState

    model = SimpleNamespace(
        _inner=SimpleNamespace(
            draft=SimpleNamespace(
                native_dspark=True,
                layers=[SimpleNamespace(attn=SimpleNamespace(window_size=8, head_dim=32))],
            )
        )
    )
    state = DeepseekV41SharedAttentionState()
    _seed_deepseek_v41_draft_windows(model, state, 2)
    window = state.draft_kv_by_layer[0]
    assert window.shape == (2, 8, 32) and window.dtype == torch.float8_e4m3fn
    _seed_deepseek_v41_draft_windows(model, state, 2)
    assert state.draft_kv_by_layer[0] is window
    state.reset()
    assert not state.draft_kv_by_layer


# Hyper-Connection and attention inputs


def test_shared_hyper_connection_ops_keep_v41_shape_contracts() -> None:
    hidden = torch.empty(2, 8, 5, 64, device="meta")
    mixes = torch.empty(2, 8, 35, dtype=torch.float32, device="meta")
    scale = torch.empty(3, 5, dtype=torch.float32, device="meta")
    base = torch.empty(3, 5, dtype=torch.float32, device="meta")

    pre, post, comb = torch.ops.tensor_cast.hc_split_sinkhorn(mixes, scale, base, 5, 20, 1e-6)
    reduced = torch.ops.tensor_cast.hc_pre_reduce(hidden, pre)

    assert pre.shape == (2, 8, 5)
    assert post.shape == (2, 8, 5)
    assert comb.shape == (2, 8, 5, 5)
    assert reduced.shape == (2, 8, 64)


def test_v41_hc_pre_v3_collapses_incoming_mix_and_returns_next_mix() -> None:
    from tensor_cast.performance_model.op_invoke_info import OpInvokeInfo

    hidden = torch.empty(2, 8, 4, 64, device="meta")
    fn = torch.empty(24, 256, dtype=torch.float32, device="meta")
    scale = torch.empty(3, dtype=torch.float32, device="meta")
    base = torch.empty(24, dtype=torch.float32, device="meta")
    incoming = torch.empty(2, 8, 4, dtype=torch.float32, device="meta")
    args = (hidden, fn, scale, base, incoming, 4, 20, 1e-20, 1e-6)

    collapsed, post, comb, next_pre = torch.ops.tensor_cast.hc_pre_v3(*args)

    assert collapsed.shape == (2, 8, 64)
    assert post.shape == next_pre.shape == incoming.shape
    assert comb.shape == (2, 8, 4, 4)
    props = OpInvokeInfo(torch.ops.tensor_cast.hc_pre_v3.default, args, {}, (collapsed, post, comb, next_pre))
    assert props.get_perf_properties().compute_ops[torch.float32].mma_ops > 0


@pytest.mark.parametrize("block_type", [DeepseekV41DecoderLayer, DeepseekV41DsparkBlock])
def test_v41_hc_pre_uses_incoming_mix_in_one_op(block_type, monkeypatch) -> None:
    block = block_type.__new__(block_type)
    torch.nn.Module.__init__(block)
    block.hc_mult = 4
    block.hc_sinkhorn_iters = 20
    block.hc_eps = 1e-6
    block.config = SimpleNamespace(rms_norm_eps=1e-20)
    leading = (1, 2) if block_type is DeepseekV41DecoderLayer else (2,)
    hidden = torch.empty(*leading, 4, 8, device="meta")
    incoming = torch.empty(*leading, 4, dtype=torch.float32, device="meta")
    fn = torch.empty(24, 32, dtype=torch.float32, device="meta")
    scale = torch.empty(3, dtype=torch.float32, device="meta")
    base = torch.empty(24, dtype=torch.float32, device="meta")
    calls = []
    generated = []
    for name in ("hc_attn_fn", "hc_ffn_fn"):
        setattr(block, name, fn)
    for name in ("hc_attn_scale", "hc_ffn_scale"):
        setattr(block, name, scale)
    for name in ("hc_attn_base", "hc_ffn_base"):
        setattr(block, name, base)
    block.input_layernorm = SimpleNamespace(weight=torch.empty(8, device="meta"))
    block.post_attention_layernorm = SimpleNamespace(weight=torch.empty(8, device="meta"))
    monkeypatch.setattr(torch.ops.tensor_cast, "rms_norm", lambda x, *args: x)
    monkeypatch.setattr(torch.ops.tensor_cast, "hc_post", lambda x, residual, *args: residual)

    def fake_hc_pre(x, weight, hc_scale, hc_base, pre_mix, *settings):
        calls.append(pre_mix)
        next_pre = torch.empty_like(pre_mix)
        generated.append(next_pre)
        return (
            torch.empty(*x.shape[:-2], 8, device="meta"),
            torch.empty_like(pre_mix),
            torch.empty(*pre_mix.shape, 4, dtype=torch.float32, device="meta"),
            next_pre,
        )

    monkeypatch.setattr(torch.ops.tensor_cast, "hc_pre_v3", fake_hc_pre)
    if block_type is DeepseekV41DecoderLayer:
        block.engram_layer_pos = None
        block.self_attn = lambda **kwargs: (kwargs["hidden_states"], None)
        block.mlp = lambda x, **kwargs: x
        output, outgoing = block(hidden, pre_mix=incoming)
    else:
        kv_state = {1: torch.empty(1, 8, 8, device="meta")}
        main_hidden = torch.empty(2, 8, device="meta")
        attention_calls = []

        def attention(x, target_hidden, start_pos, state, layer_id):
            attention_calls.append((target_hidden, start_pos, state, layer_id))
            return x

        block.attn = attention
        block.mlp = lambda x: x
        output, outgoing = block(hidden, incoming, main_hidden, 0, kv_state, 1)
        assert len(attention_calls) == 1
        assert attention_calls[0][0] is main_hidden
        assert attention_calls[0][1] == 0
        assert attention_calls[0][2] is kv_state
        assert attention_calls[0][3] == 1

    assert output.shape == hidden.shape
    assert outgoing.shape == incoming.shape
    assert len(calls) == 2
    assert calls[0] is incoming
    assert calls[1] is generated[0]
    assert outgoing is generated[1]


def test_initial_hc_mix_selects_stream_zero() -> None:
    import torch

    hidden = torch.empty(2, 3, 4, 16)
    pre_mix = make_v41_identity_pre_mix(hidden, 4)

    assert pre_mix.shape == (2, 3, 4)
    assert torch.equal(pre_mix[..., 0], torch.ones(2, 3))
    assert torch.count_nonzero(pre_mix[..., 1:]) == 0


def test_hyper_connection_costs_count_fp32_reduction_and_sinkhorn_iterations():
    hidden = torch.empty(2, 3, 5, 32, dtype=torch.bfloat16, device="meta")
    pre = torch.empty(2, 3, 5, device="meta")
    reduced = torch.ops.tensor_cast.hc_pre_reduce(hidden, pre)
    props = OpInvokeInfo(torch.ops.tensor_cast.hc_pre_reduce.default, (hidden, pre), {}, reduced).get_perf_properties()
    assert props.compute_ops[torch.float32].gp_ops == 6 * 5 * 32 * 2
    assert props.memory_read_bytes > 0 and props.memory_write_bytes > 0
    mixes = torch.empty(2, 3, 35, device="meta")
    scale, base = torch.empty(3, device="meta"), torch.empty(35, device="meta")
    costs = []
    for iterations in (1, 4):
        args = (mixes, scale, base, 5, iterations, 1e-6)
        output = torch.ops.tensor_cast.hc_split_sinkhorn(*args)
        props = OpInvokeInfo(torch.ops.tensor_cast.hc_split_sinkhorn.default, args, {}, output).get_perf_properties()
        assert props.memory_read_bytes > 0 and props.memory_write_bytes > 0
        costs.append(props.compute_ops[torch.float32].gp_ops)
    assert costs[1] - costs[0] == 6 * 3 * (6 * 25 + 4 * 5)


def test_model_aux_states_are_actual_attention_inputs_without_extra_mean():
    config = _tiny_config()
    config.dspark_noise_token_id = 0
    with torch.device("meta"):
        model = DeepseekV41Model(DeepseekV41Config(text_config=config))
        for layer in model.layers:
            layer.self_attn = _IdentityAttention()
            layer.mlp = _IdentityMlp()
        embeds = torch.empty(2, 3, 32)
        hashes = torch.empty(2, 3, len(config.engram_layer_ids), 24, dtype=torch.long)
        with _RecordOps() as record:
            result = model(inputs_embeds=embeds, engram_hash_ids=hashes, output_hidden_states=True, return_dict=True)
        assert len(result.hidden_states) == 40
        assert all(result.hidden_states[i] is model.layers[i].self_attn.attention_input for i in (37, 38, 39))
        assert (-2,) not in record.mean_dimensions


# Engram


@pytest.mark.parametrize("model_type", ["deepseek_v41", "deepseek_v41_text"])
@pytest.mark.parametrize("offload_flag", ["enable_decode_kv_offload", "enable_prefill_kv_offload"])
def test_v41_rejects_kv_offload_before_input_generation(model_type, offload_flag):
    from tensor_cast.core.model_runner import ModelRunner

    runner = object.__new__(ModelRunner)
    runner.user_input = SimpleNamespace(**{offload_flag: True})
    runner.model = SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type=model_type)))
    generate = MagicMock()
    with pytest.raises(ValueError, match="DeepSeek V4.1 KV offload is not supported"):
        runner.run_inference(generate_inputs_func=generate)
    generate.assert_not_called()


def test_v41_offload_disabled_preserves_cache_accounting():
    from tensor_cast.core.model_runner import ModelRunner

    runner = object.__new__(ModelRunner)
    runner.user_input = SimpleNamespace()
    runner.model = SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="deepseek_v41")))
    assert runner._apply_kv_offload_accounting(10.0, 1024, 128, 32, 1, False, 2) == (
        10.0,
        1152 / 1024**3,
        "none",
        0.0,
        0.0,
    )


@pytest.mark.parametrize("device_name", ["A2_376T_64G", "A3_752T_128G_DIE"])
def test_engram_uses_shared_host_link_for_serial_transfer(device_name):
    from tensor_cast.device import ATLAS_800
    from tensor_cast.performance_model import _estimate_default_without_static_cost

    hashes = torch.empty(2, 4, 24, dtype=torch.long, device="meta")
    values = torch.ops.tensor_cast.engram_table_lookup(hashes, 384006168, 256, 264, 16)
    info = OpInvokeInfo(
        torch.ops.tensor_cast.engram_table_lookup.default, (hashes, 384006168, 256, 264, 16), {}, values
    )
    profile = getattr(ATLAS_800, device_name)
    result = _estimate_default_without_static_cost(info, profile)
    transfer = profile.host_latency_s + info.get_perf_properties().host_to_device_bytes / (
        profile.host_bandwidth_bytes_ps * profile.host_transfer_efficiency
    )
    assert result.statistics["host_to_device_time_s"] == pytest.approx(transfer)
    assert result.execution_time_s == pytest.approx(
        transfer + max(result.statistics["compute_time_s"], result.statistics["memory_access_time_s"])
    )


def test_engram_state_accepts_an_empty_request_batch() -> None:
    class Model:
        text_config = type("TextConfig", (), {"model_type": "deepseek_v41"})()

    assert isinstance(_get_deepseek_v41_engram_state(Model(), []), EngramHashState)


def test_engram_semantic_ops_and_h2d_traffic() -> None:
    import torch

    from tensor_cast.performance_model.op_invoke_info import OpInvokeInfo

    input_ids = torch.empty(2, 4, dtype=torch.long, device="meta")
    history = torch.empty(2, 3, dtype=torch.long, device="meta")
    all_hashes, next_history = torch.ops.tensor_cast.engram_hash_lookup(
        input_ids, history, 24, 4, 2, input_ids.shape[0]
    )
    assert all_hashes.shape == (2, 4, 2, 24)
    assert next_history.shape == (2, 3)
    hashes = all_hashes[..., 0, :]
    hidden = torch.empty(2, 4, 4, 4096, dtype=torch.bfloat16, device="meta")
    values = torch.ops.tensor_cast.engram_table_lookup(hashes, 384006168, 256, 264, 16)
    assert values.shape == (2, 4, 24, 256)
    wkv = torch.empty(24 * 256, 4096 * 5, dtype=torch.float8_e4m3fn, device="meta")
    q_weight = torch.empty(4, 4096, dtype=torch.float32, device="meta")
    k_weight = torch.empty(4, 4096, dtype=torch.float32, device="meta")
    injected = torch.ops.tensor_cast.engram_inject(hidden, values, wkv, q_weight, k_weight, 4)
    assert injected.shape == hidden.shape
    info = OpInvokeInfo(
        torch.ops.tensor_cast.engram_table_lookup.default,
        (hashes, 384006168, 256, 264, 16),
        {},
        values,
    )
    props = info.get_perf_properties()
    assert props.host_to_device_bytes == ((2 * 4 * 24 + 15) // 16) * 264

    inject_info = OpInvokeInfo(
        torch.ops.tensor_cast.engram_inject.default,
        (hidden, values, wkv, q_weight, k_weight, 4),
        {},
        injected,
    )
    inject_props = inject_info.get_perf_properties()
    assert inject_props.memory_read_bytes >= hidden.numel() * hidden.element_size()
    assert inject_props.memory_write_bytes >= injected.numel() * injected.element_size()


def test_precomputed_engram_hash_contract() -> None:
    input_ids = torch.empty(2, 4, dtype=torch.long, device="meta")
    hashes = torch.empty(2, 4, 2, 24, dtype=torch.long, device="meta")
    validate_precomputed_engram_hash_ids(hashes, input_ids, 2, 24)
    with pytest.raises(ValueError, match="batch and sequence"):
        validate_precomputed_engram_hash_ids(hashes[:, :3], input_ids, 2, 24)
    with pytest.raises(ValueError, match="layer/hash-column"):
        validate_precomputed_engram_hash_ids(hashes, input_ids, 2, 16)


def test_decoder_precomputed_hashes_without_ids_and_actual_attention_tap():
    config = _tiny_config()
    with torch.device("meta"):
        layer = DeepseekV41DecoderLayer(config, 1)
        layer.self_attn = _IdentityAttention()
        layer.mlp = _IdentityMlp()
        hidden = torch.empty(2, 3, config.hc_mult, config.hidden_size)
        hashes = torch.empty(2, 3, len(config.engram_layer_ids), 24, dtype=torch.long)
        with _RecordOps() as record:
            result = layer(hidden, engram_hash_ids=hashes, output_attention_input=True)
        assert "tensor_cast.engram_table_lookup.default" in record.ops
        assert "tensor_cast.engram_inject.default" in record.ops
        assert len(result) == 3 and result[2].shape == (2, 3, 32)
        assert result[2].ndim == 3
        assert result[2] is layer.self_attn.attention_input


def test_model_validates_hashes_against_embeddings_without_token_ids():
    config = _tiny_config()
    with torch.device("meta"):
        model = DeepseekV41Model(DeepseekV41Config(text_config=config))
        embeds = torch.empty(2, 3, 32)
        hashes = torch.empty(1, 3, len(config.engram_layer_ids), 24, dtype=torch.long)
        with TestCase().assertRaisesRegex(ValueError, "batch and sequence"):
            model(inputs_embeds=embeds, engram_hash_ids=hashes)


# Vision and multimodal inputs


def test_vision_patch_aligner_and_merge_shapes() -> None:
    import torch

    vision_config = DeepseekV41VisionConfig(
        num_hidden_layers=2,
        hidden_size=32,
        num_attention_heads=4,
        intermediate_size=48,
        patch_size=2,
        downsample_ratio=3,
        min_pixels=0,
        max_image_tokens=64,
    )
    vision = DeepseekV41VisionModel(vision_config).to(device="meta", dtype=torch.bfloat16)
    aligner = DeepseekV41Aligner(vision_config, 64).to(device="meta", dtype=torch.bfloat16)
    pixels = torch.empty(1, 12, 3, 2, 2, device="meta", dtype=torch.bfloat16)

    vision_hidden = vision(pixels, 3, 4)
    image_features = aligner(vision_hidden, 3, 4)
    text_hidden = torch.empty(1, 7, 64, device="meta", dtype=torch.bfloat16)
    token_types = torch.empty(1, 7, device="meta", dtype=torch.int8)
    boundary = torch.empty(64, device="meta", dtype=torch.bfloat16)
    merged = torch.ops.tensor_cast.deepseek_v41_image_merge(
        text_hidden, image_features, token_types, boundary, boundary, boundary
    )

    assert vision_hidden.shape == (1, 12, 32)
    assert image_features.shape == (1, 2, 64)
    assert merged.shape == text_hidden.shape


def test_vl_gate_and_vision_roofline_properties() -> None:
    import torch

    from tensor_cast.performance_model.op_invoke_info import OpInvokeInfo

    scores = torch.empty(8, 384, device="meta", dtype=torch.float32)
    text_bias = torch.empty(384, device="meta", dtype=torch.float32)
    image_bias = torch.empty(384, device="meta", dtype=torch.float32)
    image_mask = torch.empty(8, device="meta", dtype=torch.bool)
    weights, indices = torch.ops.tensor_cast.moe_gating_top_k_vl(
        scores, 6, True, 1.5, text_bias, image_bias, image_mask
    )
    gate_info = OpInvokeInfo(
        torch.ops.tensor_cast.moe_gating_top_k_vl.default,
        (scores, 6, True, 1.5, text_bias, image_bias, image_mask),
        {},
        (weights, indices),
    )
    gate_props = gate_info.get_perf_properties()

    query = torch.empty(1, 12, 4, 8, device="meta", dtype=torch.bfloat16)
    key = torch.empty_like(query)
    rope_out = torch.ops.tensor_cast.deepseek_v41_vision_rope(query, key, 3, 4, 10000.0)
    rope_info = OpInvokeInfo(
        torch.ops.tensor_cast.deepseek_v41_vision_rope.default,
        (query, key, 3, 4, 10000.0),
        {},
        rope_out,
    )
    rope_props = rope_info.get_perf_properties()

    assert weights.shape == indices.shape == (8, 6)
    assert gate_props.compute_ops[torch.float32].gp_ops > 0
    assert rope_props.compute_ops[torch.bfloat16].gp_ops > 0


def test_reference_image_geometry() -> None:
    from tensor_cast.core.input_generator import (
        _deepseek_v41_image_grid,
        _validate_deepseek_v41_image_chunk,
    )

    config = DeepseekV41VisionConfig()
    assert _deepseek_v41_image_grid(544, 544, config) == (39, 39, 13, 13)
    with pytest.raises(ValueError, match="first prefill chunk"):
        _validate_deepseek_v41_image_chunk("deepseek_v41", True, False, 128)
    _validate_deepseek_v41_image_chunk("deepseek_v41", True, False, 0)


def test_image_inputs_account_for_data_parallel_batch_and_boundary_tokens():
    vision = SimpleNamespace(patch_size=16, downsample_ratio=2, in_channels=3, max_image_tokens=1024)
    model = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(vision_config=vision),
            parallel_config=SimpleNamespace(data_parallel_size=2),
            dtype=torch.bfloat16,
        )
    )
    result = _generate_deepseek_v41_image_inputs(model, 2, 64, 96, concurrency=3)
    assert result["pixel_values"].shape == (4, 24, 3, 16, 16)
    assert result["pixel_values"].dtype == torch.bfloat16
    assert result["vision_grid_height"] == 4
    assert result["vision_grid_width"] == 6
    assert result["num_image_tokens"] == 2 * (2 * (3 + 1) + 2)


# Native DSpark and parallelism


def test_local_checkpoint_uses_builtin_loader_and_excludes_dspark() -> None:
    loader = AutoModelConfigLoader()
    config = loader.load_config(str(CHECKPOINT))
    text = config.get_text_config()

    assert config.model_type == "deepseek_v41"
    assert text.model_type == "deepseek_v41_text"
    assert config.vision_config.model_type == "deepseek_v41_vision"
    assert text.num_hidden_layers == 40
    assert text.active_backbone_layer_ids == tuple(range(40))
    assert len(text.compress_ratios) == 40
    assert text.excluded_dspark_layer_ids == (37, 38, 39)
    assert text.kv_source_layer_ids == (2, 8, 14, 20)
    assert text.index_source_layer_ids == (2, 8, 14, 20, 24, 28, 32, 36)
    assert loader.is_transformers_natively_supported is False


def test_native_dspark_uses_checkpoint_structure() -> None:
    config = DeepseekV41Config.from_pretrained(CHECKPOINT).get_text_config()
    scfg = DsparkConfig(dspark_block_size=8, num_draft_layers=3)

    with torch.device("meta"):
        draft = DeepseekV41NativeDsparkDraft(config, scfg)

    assert draft.target_layer_ids == [37, 38, 39]
    assert len(draft.layers) == 3
    assert draft.verify_window_size == 8
    assert draft.proposal_count == 7
    assert all(layer.attn.__class__.__name__ == "DeepseekV41DsparkAttention" for layer in draft.layers)
    assert all(layer.mlp.gate.top_k == 3 for layer in draft.layers)
    assert all(layer.mlp.gate.weight.shape[0] == 128 for layer in draft.layers)


def test_native_dspark_attention_does_not_consume_backbone_csa2_state() -> None:
    config = DeepseekV41Config.from_pretrained(CHECKPOINT).get_text_config()
    scfg = DsparkConfig(dspark_block_size=8, num_draft_layers=3)
    with torch.device("meta"):
        draft = DeepseekV41NativeDsparkDraft(config, scfg)
        hidden = torch.empty(1, 7, config.hidden_size, device="meta")
        main_hidden = torch.empty(1, 1, config.hidden_size, device="meta")
        window = draft.layers[0].attn.prefill(main_hidden)
        output = draft.layers[0].attn(hidden, main_hidden, 4096, {0: window})

    assert output.shape == hidden.shape


def test_native_dspark_prefill_only_seeds_three_fp8_windows() -> None:
    config = DeepseekV41Config.from_pretrained(CHECKPOINT).get_text_config()
    with torch.device("meta"):
        draft = DeepseekV41NativeDsparkDraft(config, DsparkConfig(dspark_block_size=5, num_draft_layers=3))
        aux = [torch.empty(1, 32, config.hidden_size, device="meta") for _ in range(3)]
        caches = draft.prefill(aux)

    assert len(caches) == 3
    assert all(cache.shape == (1, 128, config.head_dim) for cache in caches)
    assert all(cache.dtype == torch.float8_e4m3fn for cache in caches)


def test_native_dspark_uses_checkpoint_precision_independent_of_target_quantization() -> None:
    config = DeepseekV41Config.from_pretrained(CHECKPOINT).get_text_config()
    with torch.device("meta"):
        draft = DeepseekV41NativeDsparkDraft(config, DsparkConfig(dspark_block_size=5, num_draft_layers=3))
        # patch_moe performs this fused-3D-weight to per-expert conversion in
        # the real build before native draft quantization runs.
        original_experts = draft.layers[0].mlp.experts
        draft.layers[0].mlp.experts = torch.nn.ModuleList([MoeExpertMLP(original_experts, 0)])

    class Host:
        def __init__(self):
            self._inner = torch.nn.Module()
            self._inner.draft = draft
            self.model_config = ModelConfig(
                parallel_config=ParallelConfig(),
                quant_config=QuantConfig(),
                quant_linear_cls=TensorCastQuantLinear,
            )

    host = Host()
    _quantize_deepseek_v41_native_dspark(host)

    assert isinstance(draft.main_proj, QuantLinearBase)
    assert draft.main_proj.quant_config.quant_type == LinearQuantType.FP8
    assert isinstance(draft.layers[0].attn.q_a_proj, QuantLinearBase)
    assert draft.layers[0].attn.q_a_proj.quant_config.quant_type == LinearQuantType.FP8
    assert isinstance(draft.layers[0].mlp.experts[0].gate_proj, QuantLinearBase)
    assert draft.layers[0].mlp.experts[0].gate_proj.quant_config.quant_type == LinearQuantType.MXFP4
    assert isinstance(draft.layers[0].mlp.shared_experts.gate_proj, QuantLinearBase)
    assert draft.layers[0].mlp.shared_experts.gate_proj.quant_config.quant_type == LinearQuantType.FP8


def test_native_dspark_build_shares_weights_and_executes_prefill_decode():
    config = _tiny_config()
    scfg = DsparkConfig(dspark_block_size=6, num_draft_layers=3, markov_rank=8)
    with torch.device("meta"):
        target = _Target(config)
        wrapper = build_deepseek_v41_native_dspark(ModelWrapperBase(target), scfg, config)
        assert wrapper.draft.embed_tokens is target.embed_tokens
        assert wrapper.draft.lm_head is target.lm_head
        assert wrapper.draft.proposal_count == 5
        # Production applies the same expert conversion before executing meta MoE.
        moe_config = MoEConfig(module_name="DeepseekV41MoE", gate_router=route_deepseek_v41_gate)
        for layer in wrapper.draft.layers:
            _patch_moe_expert_helper(SimpleNamespace(hf_config=SimpleNamespace(model_type="deepseek_v41")), layer.mlp)
            layer.mlp = MoELayer(moe_config, layer.mlp)
        ids = torch.empty(2, 4, dtype=torch.long)
        positions = torch.empty_like(ids)
        state = DeepseekV41SharedAttentionState()
        logits, caches = wrapper(ids, positions, sampling_metadata=SamplingMetadata(), csa2_state=state)
        assert logits.shape == (2, 4, 64)
        assert len(caches) == 3
        assert all(cache.shape == (2, 8, 32) and cache.dtype == torch.float8_e4m3fn for cache in caches)
        proposals, verified, confidence = wrapper(
            ids[:, :1],
            positions[:, :1],
            sampling_metadata=SamplingMetadata(selected_token_indices=None),
            attention_meta=SimpleNamespace(start_pos=32),
            csa2_state=state,
        )
        assert proposals.shape == confidence.shape == (2, 5)
        assert verified.shape[0] == 2
        assert proposals.dtype == torch.long
        assert target.calls[-1]["output_aux_hidden_state_layer_ids"] == [37, 38, 39]
        with TestCase().assertRaisesRegex(ValueError, "sampling_metadata"):
            wrapper(ids, positions)
        with TestCase().assertRaisesRegex(ValueError, "num_draft_layers"):
            build_deepseek_v41_native_dspark(ModelWrapperBase(target), DsparkConfig(num_draft_layers=2), config)


def test_dspark_ring_window_consumes_prefill_and_two_decode_updates():
    config = _tiny_config()
    config.sliding_window = 4
    attention = DeepseekV41DsparkAttention(config)
    # Isolate projected KV values; exercise the real ring writer and attention forward.
    projected = torch.full((1, 3, 32), 1.0, dtype=torch.float8_e4m3fn)
    with patch.object(attention, "_project_main_kv", return_value=projected):
        state = {0: attention.prefill(torch.ones(1, 3, 32))}
    torch.testing.assert_close(state[0][:, :3].float(), projected.float())
    for position, value in ((3, 2.0), (4, 3.0)):
        previous = state[0]
        with torch.device("meta"):
            hidden = torch.empty(1, 2, 32)
        # Use meta projections/attention but retain materialized cache writes for checking the ring.
        attention.to("meta")
        kv = torch.full((1, 1, 32), value, dtype=torch.float8_e4m3fn)
        with patch.object(attention, "_project_main_kv", return_value=kv):
            output = attention(hidden, hidden[:, :1], position, state)
        assert output.shape == hidden.shape
        assert state[0] is not previous
        torch.testing.assert_close(state[0][:, position % 4].float(), torch.full((1, 32), value))
    assert state[0][0, 1, 0].float() == 1.0
    assert state[0][0, 3, 0].float() == 2.0


def test_text_config_native_dspark_tp_shards_projections_experts_and_markov():
    from tensor_cast.layers.deepseek_v41_dspark import build_deepseek_v41_native_dspark
    from tensor_cast.layers.moe_layer import MoELayer
    from tensor_cast.layers.utils import ModelWrapperBase
    from tensor_cast.model_config import DsparkConfig, MlaConfig, ModelConfig, MoEConfig, ParallelConfig, QuantConfig
    from tensor_cast.parallel_group import ParallelGroupManager
    from tensor_cast.transformers.builtin_model.deepseek_v41 import route_deepseek_v41_gate
    from tensor_cast.transformers.transformations import _patch_moe_expert_helper, shard_model_by_tp

    config = _tiny_config()
    config.o_groups = 2
    scfg = DsparkConfig(dspark_block_size=6, num_draft_layers=3, markov_rank=8)
    parallel = ParallelConfig(world_size=2, rank=0, tensor_parallel_size=2, moe_tensor_parallel_size=2)
    with torch.device("meta"):
        target = _Target(config)
        wrapper = build_deepseek_v41_native_dspark(ModelWrapperBase(target), scfg, config)
        moe_config = MoEConfig(module_name="DeepseekV41MoE", gate_router=route_deepseek_v41_gate)
        for layer in wrapper.draft.layers:
            _patch_moe_expert_helper(SimpleNamespace(hf_config=SimpleNamespace(model_type="deepseek_v41")), layer.mlp)
            layer.mlp = MoELayer(moe_config, layer.mlp)

        host = SimpleNamespace(
            _inner=wrapper,
            hf_config=config,
            text_config=config,
            is_vl_model=False,
            model_config=ModelConfig(
                hf_config=config,
                parallel_config=parallel,
                quant_config=QuantConfig(),
                dspark_config=scfg,
                moe_config=moe_config,
                mla_config=MlaConfig(module_name="DeepseekV41SparseAttention", mla_cls=DeepseekV4SparseAttention),
            ),
            parallel_group_manager=ParallelGroupManager(parallel),
        )

        def replace(name, module):
            parts = name.split(".")
            parent = wrapper.get_submodule(".".join(parts[:-1])) if len(parts) > 1 else wrapper
            setattr(parent, parts[-1], module)

        host._replace_module = replace
        shard_model_by_tp(host)
        attention = wrapper.draft.layers[0].attn
        assert attention.q_b_proj.weight.shape == (32, 16)
        assert attention.wo_a.weight.shape[0] == 8
        assert attention.o_proj.weight.shape[1] == 8
        assert wrapper.draft.markov_head.markov_bias.weight.shape[0] == 32
        experts = wrapper.draft.layers[0].mlp.fused_moe.experts
        assert hasattr(experts.experts[0].gate_proj, "tp_size")
        assert experts.experts[0].gate_proj.tp_size == 2


@pytest.mark.parametrize("mode", ["prefill", "decode", "image"])
def test_fixed_inputs_wire_request_states_and_compressed_cache(mode, monkeypatch):
    from tensor_cast.core import input_generator as module

    text = _tiny_config()
    text.num_hidden_layers = 2
    text.compress_ratios = [0, 4]
    text.kv_source_layer_ids = [1]
    root = SimpleNamespace(
        model_type="deepseek_v41",
        vision_config=SimpleNamespace(
            patch_size=16,
            downsample_ratio=2,
            in_channels=3,
            max_image_tokens=1024,
        ),
    )
    config = ModelConfig(ParallelConfig(), QuantConfig(), hf_config=root)
    model = SimpleNamespace(
        model_config=config,
        text_config=text,
        is_vl_model=mode == "image",
        _inner=SimpleNamespace(
            draft=SimpleNamespace(
                native_dspark=True,
                layers=[
                    SimpleNamespace(attn=SimpleNamespace(window_size=8, head_dim=32)),
                ],
            )
        ),
    )
    # Isolate the ordinary paged-cache allocator; exercise the real V4.1 cache/state paths.
    monkeypatch.setattr(module, "dcp_sharded_num_blocks", lambda model, blocks: blocks)
    monkeypatch.setattr(module, "_get_kv_cache_info", lambda *args: ({}, 0))
    monkeypatch.setattr(module, "kv_cache_excluded_layer_indices", lambda model: set())
    monkeypatch.setattr(module, "get_sparse_attention_indexer_cache_info", lambda *args: {})
    request = RequestInfo(
        query_len=1 if mode == "decode" else 8,
        seq_len=16 if mode == "decode" else 8,
        is_decode=mode == "decode",
        concurrency=2,
    )
    if mode == "image":
        request.image_batch_size, request.image_height, request.image_width = 1, 64, 96
    result = generate_inputs(model, [request], block_size=16)
    assert result["csa2_state"] is request.csa2_state
    assert result["engram_hash_state"] is request.engram_hash_state
    assert result["engram_token_ids"].shape == result["input_ids"].shape
    assert result["compressed_kv_cache_by_source"].keys() == {1}
    assert result["compressed_kv_scale_cache_by_source"].keys() == {1}
    assert result["compressed_kv_cache_per_token"] > 0
    assert result["csa2_cache_ownership"]["compressed_kv_owner_by_layer"][1] == 1
    if mode == "image":
        assert result["input_ids"].shape == (1, 36)  # 2 requests * (8 text + 10 image tokens)
        assert result["pixel_values"].shape == (2, 24, 3, 16, 16)
        for key, dtype in (
            ("image_token_types", torch.int8),
            ("image_token_mask", torch.bool),
            ("engram_token_mask", torch.bool),
        ):
            assert result[key].shape == result["input_ids"].shape
            assert result[key].dtype == dtype
    else:
        assert "image_token_mask" not in result
    state = result["csa2_state"]
    if mode == "decode":
        window = state.draft_kv_by_layer[0]
        assert window.shape == (2, 8, 32)
        assert window.dtype == torch.float8_e4m3fn
        again = generate_inputs(model, [request], block_size=16)
        assert again["csa2_state"].draft_kv_by_layer[0] is window
        assert again["engram_hash_state"] is request.engram_hash_state
    else:
        assert not state.draft_kv_by_layer


def test_engram_request_state_is_shared_reused_and_rejects_mixed_batches():
    model = SimpleNamespace(text_config=SimpleNamespace(model_type="deepseek_v41_text"))
    requests = [RequestInfo(query_len=1, seq_len=8) for _ in range(2)]
    state = _get_deepseek_v41_engram_state(model, requests)
    assert all(request.engram_hash_state is state for request in requests)
    history = torch.tensor([[1, 2], [3, 4]])
    state.update(history)
    assert _get_deepseek_v41_engram_state(model, requests) is state
    assert state.history is history
    requests[1].engram_hash_state = EngramHashState()
    with pytest.raises(ValueError, match="share one Engram hash state"):
        _get_deepseek_v41_engram_state(model, requests)


@pytest.mark.parametrize(
    "height,width,options,expected",
    [
        (64, 1024, {"max_wh_ratio": 2}, (4, 8, 2, 4)),
        (16, 16, {"min_pixels": 4096}, (4, 4, 2, 2)),
        (1024, 1024, {"max_image_tokens": 32}, (10, 10, 5, 5)),
        (4096, 16, {"max_image_tokens": 10}, (8, 2, 4, 1)),
        (16, 4096, {"max_image_tokens": 10}, (2, 14, 1, 7)),
    ],
)
def test_image_grid_resize_boundaries_respect_token_budget(height, width, options, expected):
    vision = SimpleNamespace(patch_size=16, downsample_ratio=2, max_image_tokens=1024)
    for name, value in options.items():
        setattr(vision, name, value)
    grid = _deepseek_v41_image_grid(height, width, vision)
    assert grid == expected
    assert grid[2] * (grid[3] + 1) + 2 <= vision.max_image_tokens
