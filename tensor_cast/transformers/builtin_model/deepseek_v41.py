"""DeepSeek V4.1-Flash configuration, text backbone, Engram and Vision shells."""

from __future__ import annotations

import logging
from typing import Any, Mapping, Optional

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, DeepseekV3Config, PretrainedConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

from ...layers.deepseek_v41 import (
    DeepseekV41SparseAttention as DeepseekV41SparseAttentionWrapper,
    resolve_csa2_layer_role,
)
from ...layers import COLWISE_LINEAR, ROWWISE_LINEAR
from ...layers.deepseek_v4 import compute_v4_gate_scores, route_v4_gate_tail
from .deepseek_v4 import (
    DeepseekV4Compressor,
    DeepseekV4DecoderLayer,
    DeepseekV4Indexer,
    DeepseekV4MLP,
    DeepseekV4MoE,
    DeepseekV4Model,
    DeepseekV4RMSNorm,
    DeepseekV4RotaryEmbedding,
    DeepseekV4SparseAttention,
    _apply_v4_swiglu_limit,
)
from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
    DeepseekV3ForCausalLM,
    DeepseekV3PreTrainedModel,
)
from ..custom_model_registry import ModelProfile, register_model_profile

logger = logging.getLogger(__name__)

_VALID_COMPRESS_RATIOS = frozenset({0, 1, 2})
_BACKBONE_LAYERS = 40
_KV_SOURCE_COUNT = 4
_INDEX_SOURCE_COUNT = 8


def _as_int_tuple(value: Optional[list[int]], name: str) -> tuple[int, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"DeepSeek V4.1 {name} must be a list of integers")
    try:
        result = tuple(int(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"DeepSeek V4.1 {name} must be a list of integers") from exc
    if any(item < 0 for item in result):
        raise ValueError(f"DeepSeek V4.1 {name} must contain non-negative layer ids")
    return result


class DeepseekV41TextConfig(DeepseekV3Config):
    """Normalized text configuration for the V4.1 checkpoint.

    The checkpoint has 43 compression entries because the final three entries
    belong to DSpark.  Only the first ``num_hidden_layers`` entries are active
    backbone policy and are exposed as ``compress_ratios``.
    """

    model_type = "deepseek_v41_text"

    def __init__(
        self,
        compress_ratios: Optional[list[int]] = None,
        kv_source_layer_ids: Optional[list[int]] = None,
        index_source_layer_ids: Optional[list[int]] = None,
        candidate_source_layer_id: Optional[int] = None,
        index_topk: Optional[int] = None,
        index_n_heads: int = 32,
        index_head_dim: int = 128,
        candidate_topk_blocks: int = 2048,
        candidate_block_size: int = 8,
        compress_rope_theta: Optional[float] = None,
        hc_mult: int = 4,
        hc_sinkhorn_iters: int = 20,
        hc_eps: float = 1e-6,
        o_groups: int = 8,
        o_lora_rank: Optional[int] = 1024,
        swiglu_limit: Optional[float] = None,
        expert_dtype: Optional[str] = None,
        scoring_func: Optional[str] = None,
        topk_method: str = "noaux_tc",
        routed_scaling_factor: Optional[float] = None,
        engram_layer_ids: Optional[list[int]] = None,
        engram_num_embeddings: Optional[list[int]] = None,
        engram_max_ngram_size: int = 1,
        engram_vocab_size: int = 0,
        engram_n_heads: int = 0,
        engram_head_dim: int = 0,
        engram_compressed_vocab_size: int = 0,
        engram_pad_token_id: int = 2,
        num_nextn_predict_layers: int = 0,
        dspark_target_layer_ids: Optional[list[int]] = None,
        **kwargs: Any,
    ) -> None:
        # Reference inference_config.json uses these aliases.
        alias_score = kwargs.pop("score_func", "sqrtsoftplus")
        alias_scale = kwargs.pop("route_scale", 1.5)
        scoring_func = alias_score if scoring_func is None else scoring_func
        routed_scaling_factor = alias_scale if routed_scaling_factor is None else routed_scaling_factor
        aliases = {
            "dim": "hidden_size",
            "n_layers": "num_hidden_layers",
            "n_heads": "num_attention_heads",
            "n_kv_heads": "num_key_value_heads",
            "moe_inter_dim": "moe_intermediate_size",
            "n_routed_experts": "n_routed_experts",
            "n_shared_experts": "n_shared_experts",
            "n_activated_experts": "num_experts_per_tok",
            "window_size": "sliding_window",
            "score_func": "scoring_func",
            "route_scale": "routed_scaling_factor",
            "norm_eps": "rms_norm_eps",
            "rope_head_dim": "qk_rope_head_dim",
        }
        for source, target in aliases.items():
            if source in kwargs and target not in kwargs:
                kwargs[target] = kwargs.pop(source)

        # V4.1 has MoE in every active backbone layer.  Do not inherit the V3
        # default which silently creates dense layers at the beginning.
        kwargs.setdefault("first_k_dense_replace", 0)
        num_hidden_layers = int(kwargs.get("num_hidden_layers", _BACKBONE_LAYERS))
        kwargs["num_hidden_layers"] = num_hidden_layers
        kwargs["scoring_func"] = scoring_func
        kwargs["routed_scaling_factor"] = routed_scaling_factor
        if num_hidden_layers < 1:
            raise ValueError("DeepSeek V4.1 num_hidden_layers must be positive")

        super().__init__(**kwargs)

        ratios_missing = compress_ratios is None
        if ratios_missing:
            # Transformers constructs a no-argument instance while rendering
            # diff JSON.  Keep that schema probe valid; real checkpoint loads
            # are rejected by DeepseekV41Config before reaching this branch.
            compress_ratios = []
        if not ratios_missing and len(compress_ratios) < num_hidden_layers:
            raise ValueError(
                "DeepSeek V4.1 compress_ratios must contain at least one entry per active layer: "
                f"expected {num_hidden_layers}, got {len(compress_ratios)}"
            )
        invalid = sorted(set(compress_ratios) - _VALID_COMPRESS_RATIOS)
        if invalid:
            raise ValueError(
                f"DeepSeek V4.1 compress_ratios contains unsupported values: {invalid}; supported values are 0, 1, 2"
            )
        self.compress_ratios = [int(ratio) for ratio in compress_ratios[:num_hidden_layers]]
        self.active_backbone_layer_ids = tuple(range(num_hidden_layers))
        self.excluded_dspark_layer_ids = _as_int_tuple(dspark_target_layer_ids, "dspark_target_layer_ids")
        self.num_nextn_predict_layers = int(num_nextn_predict_layers)

        self.kv_source_layer_ids = _as_int_tuple(kv_source_layer_ids, "kv_source_layer_ids")
        self.index_source_layer_ids = _as_int_tuple(index_source_layer_ids, "index_source_layer_ids")
        self._validate_source_ids(num_hidden_layers)
        self.candidate_source_layer_id = None if candidate_source_layer_id is None else int(candidate_source_layer_id)
        if self.candidate_source_layer_id is not None and not 0 <= self.candidate_source_layer_id < num_hidden_layers:
            raise ValueError("DeepSeek V4.1 candidate_source_layer_id is outside the active backbone")

        self.index_topk = None if index_topk is None else int(index_topk)
        self.topk_limit = self.index_topk
        self.index_n_heads = int(index_n_heads)
        self.index_head_dim = int(index_head_dim)
        self.candidate_topk_blocks = int(candidate_topk_blocks)
        self.candidate_block_size = int(candidate_block_size)
        self.compress_rope_theta = compress_rope_theta
        self.hc_mult = int(hc_mult)
        self.hc_sinkhorn_iters = int(hc_sinkhorn_iters)
        self.hc_eps = float(hc_eps)
        self.o_groups = int(o_groups)
        self.o_lora_rank = None if o_lora_rank is None else int(o_lora_rank)
        self.swiglu_limit = swiglu_limit
        self.expert_dtype = expert_dtype
        self.scoring_func = str(scoring_func)
        self.score_func = self.scoring_func
        self.topk_method = str(topk_method)
        self.routed_scaling_factor = float(routed_scaling_factor)
        self.route_scale = self.routed_scaling_factor
        self.engram_layer_ids = _as_int_tuple(engram_layer_ids, "engram_layer_ids")
        self.engram_num_embeddings = _as_int_tuple(engram_num_embeddings, "engram_num_embeddings")
        self.engram_max_ngram_size = int(engram_max_ngram_size)
        self.engram_vocab_size = int(engram_vocab_size)
        self.engram_n_heads = int(engram_n_heads)
        self.engram_head_dim = int(engram_head_dim)
        self.engram_compressed_vocab_size = int(engram_compressed_vocab_size)
        self.engram_pad_token_id = int(engram_pad_token_id)
        if self.engram_num_embeddings and len(self.engram_num_embeddings) != len(self.engram_layer_ids):
            raise ValueError("DeepSeek V4.1 Engram rows must match engram_layer_ids")

    def _validate_source_ids(self, num_hidden_layers: int) -> None:
        for name, ids, expected_count in (
            ("kv_source_layer_ids", self.kv_source_layer_ids, _KV_SOURCE_COUNT),
            ("index_source_layer_ids", self.index_source_layer_ids, _INDEX_SOURCE_COUNT),
        ):
            if not ids:
                continue
            if len(ids) != expected_count:
                raise ValueError(f"DeepSeek V4.1 {name} must contain exactly {expected_count} layer ids")
            if len(set(ids)) != len(ids):
                raise ValueError(f"DeepSeek V4.1 {name} must not contain duplicate layer ids")
            if any(layer_id >= num_hidden_layers for layer_id in ids):
                raise ValueError(f"DeepSeek V4.1 {name} contains a layer outside the active backbone")


class DeepseekV41VisionConfig(PretrainedConfig):
    model_type = "deepseek_v41_vision"

    def __init__(
        self,
        num_hidden_layers: int = 32,
        hidden_size: int = 1024,
        num_attention_heads: int = 16,
        intermediate_size: int = 2816,
        patch_size: int = 14,
        rope_theta: float = 10000.0,
        downsample_ratio: int = 3,
        max_image_tokens: int = 1024,
        min_pixels: int = 544 * 544,
        max_wh_ratio: Optional[float] = None,
        in_channels: int = 3,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.num_hidden_layers = int(num_hidden_layers)
        self.hidden_size = int(hidden_size)
        self.num_attention_heads = int(num_attention_heads)
        self.intermediate_size = int(intermediate_size)
        self.patch_size = int(patch_size)
        self.rope_theta = float(rope_theta)
        self.downsample_ratio = int(downsample_ratio)
        self.max_image_tokens = int(max_image_tokens)
        self.min_pixels = int(min_pixels)
        self.max_wh_ratio = max_wh_ratio
        self.in_channels = int(in_channels)


class DeepseekV41Config(DeepseekV3Config):
    """Root V4.1 config that normalizes nested text and vision sections."""

    model_type = "deepseek_v41"

    def __init__(
        self,
        text_config: Optional[Mapping[str, Any] | DeepseekV41TextConfig] = None,
        vision_config: Optional[Mapping[str, Any] | DeepseekV41VisionConfig] = None,
        **kwargs: Any,
    ) -> None:
        raw_text = dict(text_config) if isinstance(text_config, Mapping) else None
        if isinstance(text_config, DeepseekV41TextConfig):
            text = text_config
        else:
            # Accept both checkpoint-style nested configs and the flat form
            # used by TensorCast unit fixtures and config overrides.
            text_kwargs = raw_text if raw_text is not None else dict(kwargs)
            if raw_text is not None and "compress_ratios" not in raw_text:
                raise ValueError("DeepSeek V4.1 text_config requires compress_ratios")
            text_kwargs.pop("model_type", None)
            text = DeepseekV41TextConfig(**text_kwargs)

        # Keep the text fields available on the root object for existing
        # TensorCast code, while get_text_config() returns the nested object.
        for key, value in text.to_dict().items():
            kwargs[key] = value
        # Transformers validates token ids from inside PretrainedConfig.__init__
        # and calls get_text_config() during that validation.  Publish the
        # nested config before entering the base constructor.
        object.__setattr__(self, "text_config", text)
        super().__init__(**kwargs)
        self.model_type = type(self).model_type
        self.text_config = text

        if isinstance(vision_config, DeepseekV41VisionConfig):
            self.vision_config = vision_config
        elif isinstance(vision_config, Mapping):
            vision_kwargs = dict(vision_config)
            vision_kwargs.pop("model_type", None)
            self.vision_config = DeepseekV41VisionConfig(**vision_kwargs)
        else:
            self.vision_config = None

        # The active layer count is defined by text_config, never by DSpark or
        # MTP metadata in the checkpoint.
        self.num_hidden_layers = text.num_hidden_layers
        self.active_backbone_layer_ids = text.active_backbone_layer_ids
        self.excluded_dspark_layer_ids = text.excluded_dspark_layer_ids

    def get_text_config(self, decoder: bool = False, encoder: bool = False) -> DeepseekV41TextConfig:
        return self.text_config


class DeepseekV41RMSNorm(DeepseekV4RMSNorm):
    pass


class DeepseekV41RotaryEmbedding(DeepseekV4RotaryEmbedding):
    pass


class DeepseekV41MoE(DeepseekV4MoE):
    def __init__(self, config):
        super().__init__(config)
        num_experts = int(getattr(config, "n_routed_experts", getattr(config, "num_local_experts", 0)))
        self.gate.bias_vl = nn.Parameter(torch.empty(num_experts, dtype=torch.float32))
        self.gate.image_token_id = int(getattr(config, "image_token_id", 129264))


class DeepseekV41MLP(DeepseekV4MLP):
    pass


class DeepseekV41Compressor(DeepseekV4Compressor):
    def __init__(self, config: DeepseekV41Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.wkv = nn.Linear(config.hidden_size, config.head_dim, bias=False)
        self.wgate = nn.Linear(config.hidden_size, config.head_dim, bias=False) if self.compress_ratio == 2 else None
        self.norm = DeepseekV41RMSNorm(config.head_dim, eps=config.rms_norm_eps)


class DeepseekV41Indexer(DeepseekV4Indexer):
    def __init__(self, config: DeepseekV41Config, index_layer_idx: int):
        super().__init__(config, index_layer_idx)
        if index_layer_idx in config.kv_source_layer_ids:
            self.wk = nn.Linear(config.head_dim, config.index_head_dim, bias=False)
            self.k_norm = DeepseekV41RMSNorm(config.index_head_dim, eps=config.rms_norm_eps)


def route_deepseek_v41_gate(
    gate: nn.Module,
    hidden_states: torch.Tensor,
    top_k: int,
    input_ids: Optional[torch.Tensor] = None,
    moe_layer_idx: Optional[int] = None,
    *,
    tp_size: int = 1,
    tp_rank: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """V4.1 score routing with the image-only correction-bias branch."""
    del moe_layer_idx
    scores, route_scale, normalize_weights = compute_v4_gate_scores(gate, hidden_states)
    if input_ids is None:
        weights, indices = route_v4_gate_tail(
            gate,
            top_k,
            False,
            scores,
            normalize_weights,
            route_scale,
            None,
            tp_size=tp_size,
            tp_rank=tp_rank,
        )
        return indices, weights.to(hidden_states.dtype)

    image_mask = input_ids.reshape(-1).eq(int(gate.image_token_id))
    if tp_size > 1:
        num_tokens = scores.shape[0]
        padding = (-num_tokens) % tp_size
        if padding:
            scores = F.pad(scores, (0, 0, 0, padding))
            image_mask = F.pad(image_mask, (0, padding), value=False)
        scores = torch.tensor_split(scores, tp_size, dim=0)[tp_rank]
        image_mask = torch.tensor_split(image_mask, tp_size, dim=0)[tp_rank]
    text_bias = gate.e_score_correction_bias
    weights, indices = torch.ops.tensor_cast.moe_gating_top_k_vl(
        scores,
        top_k,
        normalize_weights,
        route_scale,
        text_bias,
        gate.bias_vl,
        image_mask,
    )
    return indices, weights.to(hidden_states.dtype)


class DeepseekV41VisionAttention(nn.Module):
    def __init__(self, config: DeepseekV41VisionConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.rope_theta = config.rope_theta
        self.wqkv = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.wo = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor, grid_height: int, grid_width: int) -> torch.Tensor:
        images, patches, _ = hidden_states.shape
        q, k, v = self.wqkv(hidden_states).chunk(3, dim=-1)
        shape = (images, patches, self.num_heads, self.head_dim)
        q, k, v = q.view(shape), k.view(shape), v.view(shape)
        q, k = torch.ops.tensor_cast.deepseek_v41_vision_rope(q, k, grid_height, grid_width, self.rope_theta)
        output = torch.ops.tensor_cast.attention(q, k, v, None, None, None, None, None)
        return self.wo(output.reshape(images, patches, -1))


class DeepseekV41VisionMLP(nn.Module):
    def __init__(self, config: DeepseekV41VisionConfig):
        super().__init__()
        self.w1 = nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate, up = self.w1(hidden_states).chunk(2, dim=-1)
        return self.w2(F.silu(gate) * up)


class DeepseekV41VisionBlock(nn.Module):
    def __init__(self, config: DeepseekV41VisionConfig):
        super().__init__()
        self.norm1 = DeepseekV41RMSNorm(config.hidden_size)
        self.attn = DeepseekV41VisionAttention(config)
        self.norm2 = DeepseekV41RMSNorm(config.hidden_size)
        self.mlp = DeepseekV41VisionMLP(config)

    def forward(self, hidden_states: torch.Tensor, grid_height: int, grid_width: int) -> torch.Tensor:
        normed = torch.ops.tensor_cast.rms_norm(hidden_states, self.norm1.weight.data, self.norm1.variance_epsilon)
        hidden_states = hidden_states + self.attn(normed, grid_height, grid_width)
        normed = torch.ops.tensor_cast.rms_norm(hidden_states, self.norm2.weight.data, self.norm2.variance_epsilon)
        return hidden_states + self.mlp(normed)


class DeepseekV41VisionModel(nn.Module):
    def __init__(self, config: DeepseekV41VisionConfig):
        super().__init__()
        self.config = config
        patch_width = config.in_channels * config.patch_size * config.patch_size
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Linear(patch_width, config.hidden_size)
        self.blocks = nn.ModuleList(DeepseekV41VisionBlock(config) for _ in range(config.num_hidden_layers))
        self.norm = DeepseekV41RMSNorm(config.hidden_size)

    def forward(self, pixel_values: torch.Tensor, grid_height: int, grid_width: int) -> torch.Tensor:
        if pixel_values.ndim != 5:
            raise ValueError("DeepSeek V4.1 pixel_values must be [images, patches, 3, patch, patch]")
        hidden_states = self.patch_embed.proj(pixel_values.flatten(2))
        for block in self.blocks:
            hidden_states = block(hidden_states, grid_height, grid_width)
        return torch.ops.tensor_cast.rms_norm(hidden_states, self.norm.weight.data, self.norm.variance_epsilon)


class DeepseekV41Aligner(nn.Module):
    def __init__(self, vision_config: DeepseekV41VisionConfig, text_hidden_size: int):
        super().__init__()
        self.downsample_ratio = vision_config.downsample_ratio
        in_dim = vision_config.hidden_size * self.downsample_ratio**2
        self.w1 = nn.Linear(in_dim, text_hidden_size)
        self.w2 = nn.Linear(text_hidden_size, text_hidden_size)

    def forward(self, hidden_states: torch.Tensor, grid_height: int, grid_width: int) -> torch.Tensor:
        hidden_states = torch.ops.tensor_cast.deepseek_v41_vision_downsample(
            hidden_states, grid_height, grid_width, self.downsample_ratio
        )
        return self.w2(F.gelu(self.w1(hidden_states)))


class DeepseekV41SparseAttention(DeepseekV4SparseAttention):
    """V4.1 attention shell consumed by the V4 TensorCast wrapper.

    Phase 1 models ratio 1 and ratio 2 as compressed paths.  The CSA2 source,
    reuse and reindex distinctions are added in Phase 2; no V4 ratio values
    (4/128) or DSpark layers are introduced here.
    """

    def __init__(self, config: DeepseekV41Config, layer_idx: int):
        # Reimplement the V4 shell initialization so the compressor/indexer
        # classes and ratio policy are unambiguously V4.1-specific.
        nn.Module.__init__(self)
        self.config = config
        self.layer_idx = layer_idx
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.max_position_embeddings = config.max_position_embeddings
        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_nope_head_dim = getattr(config, "qk_nope_head_dim", 128)
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.kv_lora_rank = getattr(config, "kv_lora_rank", 512)
        self.v_head_dim = getattr(config, "v_head_dim", 128)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.is_causal = True
        self.compress_ratio = config.compress_ratios[layer_idx]
        self.csa2_role = resolve_csa2_layer_role(config, layer_idx)
        self.layer_role = self.csa2_role.role
        self.kv_owner_layer_id = self.csa2_role.compressed_kv_owner
        self.index_owner_layer_id = self.csa2_role.index_k_owner
        self.topk_owner_layer_id = self.csa2_role.topk_owner
        self.produces_candidate_pool = self.csa2_role.produces_candidate_pool
        self.use_indexer = self.csa2_role.produces_topk
        self.use_compressor = self.csa2_role.owns_compressed_kv
        self.n_groups = int(getattr(config, "o_groups", 1))
        self.o_lora_rank = int(getattr(config, "o_lora_rank", None) or self.hidden_size)

        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.qk_head_dim, bias=False)
        else:
            self.q_a_proj = nn.Linear(self.hidden_size, self.q_lora_rank, bias=config.attention_bias)
            self.q_a_layernorm = DeepseekV41RMSNorm(self.q_lora_rank)
            self.q_b_proj = nn.Linear(self.q_lora_rank, self.num_heads * self.head_dim, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(self.hidden_size, self.head_dim, bias=config.attention_bias)
        self.kv_a_layernorm = DeepseekV41RMSNorm(self.head_dim)
        per_group_in_dim = (self.num_heads * self.head_dim) // self.n_groups
        self.wo_a = nn.Linear(per_group_in_dim, self.n_groups * self.o_lora_rank, bias=False)
        self.o_proj = nn.Linear(self.n_groups * self.o_lora_rank, self.hidden_size, bias=config.attention_bias)
        self.scaling = 1.0 / (self.head_dim**0.5)
        self.softmax_scale = self.scaling
        self.attn_sink = nn.Parameter(torch.empty(self.num_heads, dtype=torch.float32))
        self.attention_sink = self.attn_sink
        self.compressor = DeepseekV41Compressor(config, layer_idx) if self.use_compressor else None
        self.indexer = DeepseekV41Indexer(config, layer_idx) if self.use_indexer else None


class DeepseekV41DecoderLayer(DeepseekV4DecoderLayer):
    """40-layer V4.1 block shell; both phases use the same ordered blocks."""

    def __init__(self, config: DeepseekV41Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = DeepseekV41SparseAttention(config, layer_idx)
        self.mlp = DeepseekV41MoE(config)
        self.input_layernorm = DeepseekV41RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = DeepseekV41RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        gate = getattr(self.mlp, "gate", None)
        if gate is not None:
            gate.score_func = str(getattr(config, "score_func", "sqrtsoftplus"))
            gate.route_scale = float(getattr(config, "route_scale", 1.5))
            gate.hash = False
            self.mlp.moe_layer_idx = layer_idx
            self.mlp.use_hash_routing = False
            swiglu_limit = float(getattr(config, "swiglu_limit", 0.0) or 0.0)
            experts = getattr(self.mlp, "experts", None)
            expert_iter = experts if isinstance(experts, nn.ModuleList) else getattr(experts, "experts", ())
            for expert in expert_iter:
                _apply_v4_swiglu_limit(expert, swiglu_limit)
            shared_experts = getattr(self.mlp, "shared_experts", None)
            if shared_experts is not None:
                _apply_v4_swiglu_limit(shared_experts, swiglu_limit)
        engram_layers = tuple(getattr(config, "engram_layer_ids", ()))
        self.engram_layer_pos = engram_layers.index(layer_idx) if layer_idx in engram_layers else None

    def _hc_pre(self, hidden_states, hc_fn, hc_scale, hc_base, pre_mix):
        return torch.ops.tensor_cast.hc_pre_v3(
            hidden_states,
            hc_fn,
            hc_scale,
            hc_base,
            pre_mix,
            self.hc_mult,
            self.hc_sinkhorn_iters,
            self.config.rms_norm_eps,
            self.hc_eps,
            "incoming",
        )

    def forward(self, hidden_states, pre_mix=None, input_ids=None, output_attention_input=False, **kwargs):
        if hidden_states.ndim == 3 and hidden_states.shape[-2] == self.hc_mult:
            hidden_states = hidden_states.unsqueeze(0)
        if pre_mix is not None and pre_mix.ndim == 2:
            pre_mix = pre_mix.unsqueeze(0)
        if self.engram_layer_pos is not None:
            layer_pos = self.engram_layer_pos
            all_hash_ids = kwargs.get("engram_hash_ids")
            if isinstance(all_hash_ids, torch.Tensor):
                hash_ids = all_hash_ids[..., layer_pos, :]
                table_rows = int(self.config.engram_num_embeddings[layer_pos])
                head_dim = int(self.config.engram_head_dim)
                row_bytes = head_dim + max(head_dim // 32, 1)
                rank_group = list(getattr(self.config, "engram_rank_group", (0,)))
                table_values = torch.ops.tensor_cast.engram_table_lookup(
                    hash_ids,
                    table_rows,
                    head_dim,
                    row_bytes,
                    len(rank_group),
                )
                if len(rank_group) > 1:
                    table_values = torch.ops.tensor_cast.all_reduce(
                        table_values, int(getattr(self.config, "engram_rank", 0)), rank_group
                    )
                hidden_states = torch.ops.tensor_cast.engram_inject(
                    hidden_states,
                    table_values,
                    self.engram_wkv_weight,
                    self.engram_q_weight,
                    self.engram_k_weight,
                    self.hc_mult,
                    kwargs.get("engram_token_mask"),
                )
        if pre_mix is None:
            pre_mix = torch.zeros(
                *hidden_states.shape[:-2], self.hc_mult, dtype=torch.float32, device=hidden_states.device
            )
        residual = hidden_states
        hidden_states, attn_post, attn_comb, attn_pre = self._hc_pre(
            hidden_states, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base, pre_mix
        )
        hidden_states = torch.ops.tensor_cast.rms_norm(
            hidden_states, self.input_layernorm.weight.data, self.config.rms_norm_eps
        )
        attention_input = hidden_states
        hidden_states, _ = self.self_attn(hidden_states=hidden_states, **kwargs)
        hidden_states = torch.ops.tensor_cast.hc_post(hidden_states, residual, attn_post, attn_comb, self.hc_mult)

        residual = hidden_states
        hidden_states, ffn_post, ffn_comb, ffn_pre = self._hc_pre(
            hidden_states, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base, attn_pre
        )
        hidden_states = torch.ops.tensor_cast.rms_norm(
            hidden_states, self.post_attention_layernorm.weight.data, self.config.rms_norm_eps
        )
        routing_ids = input_ids if kwargs.get("image_token_mask") is not None else None
        hidden_states = self.mlp(hidden_states, input_ids=routing_ids)
        hidden_states = torch.ops.tensor_cast.hc_post(hidden_states, residual, ffn_post, ffn_comb, self.hc_mult)
        if output_attention_input:
            return hidden_states, ffn_pre, attention_input
        return hidden_states, ffn_pre


class DeepseekV41Model(DeepseekV4Model):
    config_class = DeepseekV41Config
    config: DeepseekV41Config

    def __init__(self, config: DeepseekV41Config):
        super().__init__(config)
        self.layers = nn.ModuleList(
            DeepseekV41DecoderLayer(config, layer_idx) for layer_idx in config.active_backbone_layer_ids
        )
        self.rotary_emb = DeepseekV41RotaryEmbedding(config=config)
        self.vision = None
        self.aligner = None
        if config.vision_config is not None:
            self.vision = DeepseekV41VisionModel(config.vision_config)
            self.aligner = DeepseekV41Aligner(config.vision_config, config.hidden_size)
            self.image_start = nn.Parameter(torch.empty(config.hidden_size))
            self.image_end = nn.Parameter(torch.empty(config.hidden_size))
            self.image_newline = nn.Parameter(torch.empty(config.hidden_size))

    def prepare_multimodal_embeddings(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        pixel_values = kwargs.get("pixel_values")
        if pixel_values is None:
            return inputs_embeds
        if self.vision is None or self.aligner is None:
            raise ValueError("DeepSeek V4.1 image input requires a vision_config")
        attention_meta = kwargs.get("attention_meta")
        if int(getattr(attention_meta, "start_pos", 0) or 0) != 0:
            raise ValueError("DeepSeek V4.1 image spans must be fully contained in the first prefill chunk")
        image_token_types = kwargs.get("image_token_types")
        if image_token_types is None:
            raise ValueError("DeepSeek V4.1 image input requires image_token_types")
        grid_height = int(kwargs["vision_grid_height"])
        grid_width = int(kwargs["vision_grid_width"])
        image_features = self.aligner(
            self.vision(pixel_values, grid_height, grid_width),
            grid_height,
            grid_width,
        )
        return torch.ops.tensor_cast.deepseek_v41_image_merge(
            inputs_embeds,
            image_features,
            image_token_types,
            self.image_start,
            self.image_end,
            self.image_newline,
        )


class DeepseekV41ForCausalLM(DeepseekV3ForCausalLM):
    """Causal-LM shell backed by the V4.1 40-layer model."""

    config_class = DeepseekV41Config

    def __init__(self, config: DeepseekV41Config):
        DeepseekV3PreTrainedModel.__init__(self, config)
        self.model = DeepseekV41Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()


def _safe_register_auto_model() -> None:
    mapping = getattr(AutoModel, "_model_mapping", None)
    extra_content = getattr(mapping, "_extra_content", {})
    existing = extra_content.get(DeepseekV41Config) if isinstance(extra_content, dict) else None
    if existing is DeepseekV41Model:
        return
    if existing is not None and existing is not DeepseekV41Model:
        raise ValueError("deepseek_v41 is already registered to an incompatible AutoModel class")
    AutoModel.register(DeepseekV41Config, DeepseekV41Model, exist_ok=True)
    AutoModelForCausalLM.register(DeepseekV41Config, DeepseekV41ForCausalLM, exist_ok=True)


def _safe_register_auto_config() -> None:
    model_type = DeepseekV41Config.model_type
    mapping = getattr(CONFIG_MAPPING, "_extra_content", {})
    existing = mapping.get(model_type) if isinstance(mapping, dict) else None
    if existing is DeepseekV41Config or model_type in CONFIG_MAPPING:
        return
    if existing is not None:
        raise ValueError(
            f"{model_type} is already registered to an incompatible AutoConfig class: "
            f"{existing.__module__}.{existing.__name__}"
        )
    AutoConfig.register(model_type, DeepseekV41Config)


# Phase 1 profile maps the V4.1 shell classes to the existing TensorCast V4
# attention wrapper.  CSA2 state ownership is intentionally deferred to Phase 2.
register_model_profile(
    ModelProfile(
        model_type="deepseek_v41",
        moe_module_name="DeepseekV41MoE",
        moe_num_experts_key="n_routed_experts",
        moe_gate_returns_raw_logits=False,
        mla_module_name="DeepseekV41SparseAttention",
        mla_field_names_override={"kv_b_proj": None},
        mla_module_class_type=DeepseekV41SparseAttentionWrapper,
        moe_gate_router=route_deepseek_v41_gate,
        visual_module_path="vision",
        visual_layers_module_path="vision.blocks",
        visual_layers_path_str="vision.blocks",
        language_layers_path_str="layers",
        visual_merger_linear_mapping={
            "aligner.w1": COLWISE_LINEAR,
            "aligner.w2": ROWWISE_LINEAR,
        },
        visual_mlp_linear_mapping={
            "vision.blocks.*.mlp.w1": COLWISE_LINEAR,
            "vision.blocks.*.mlp.w2": ROWWISE_LINEAR,
        },
    )
)
_safe_register_auto_config()
_safe_register_auto_model()
