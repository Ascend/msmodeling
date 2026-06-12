# Copyright (C) 2025 HuggingFace Inc. team.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
import logging
import math
from pathlib import Path
from typing import Optional, Tuple

import torch

from torch import nn
from transformers import AutoConfig, AutoModel, DeepseekV3Config
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.cache_utils import Cache

from ...layers.deepseek_v4 import route_deepseek_v4_gate, DeepseekV4SparseAttention
from ...model_config import MlaFieldNames
from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
    DeepseekV3MLP,
    DeepseekV3Model,
    DeepseekV3MoE,
    DeepseekV3RMSNorm,
    DeepseekV3RotaryEmbedding,
)
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
from ..custom_model_registry import ModelProfile, register_model_profile


_VALID_COMPRESS_RATIOS = {0, 4, 128}
_RATIO_TO_LAYER_TYPE = {
    0: "sliding_attention",
    4: "compressed_sparse_attention",
    128: "heavily_compressed_attention",
}
_LAYER_TYPE_TO_RATIO = {value: key for key, value in _RATIO_TO_LAYER_TYPE.items()}

logger = logging.getLogger(__name__)


def _safe_register_auto_config() -> None:
    registry = getattr(AutoConfig, "register", None)
    if registry is None:
        raise RuntimeError("transformers AutoConfig.register is unavailable for deepseek_v4")
    existing = None
    mapping = getattr(CONFIG_MAPPING, "_extra_content", None)
    if isinstance(mapping, dict):
        existing = mapping.get("deepseek_v4")
    if existing is not None and existing is not DeepseekV4Config:
        raise ValueError(
            "deepseek_v4 is already registered to an incompatible AutoConfig class: "
            f"{existing.__module__}.{existing.__name__}"
        )
    AutoConfig.register("deepseek_v4", DeepseekV4Config)


def _safe_register_auto_model() -> None:
    mapping = getattr(AutoModel, "_model_mapping", None)
    extra_content = getattr(mapping, "_extra_content", None)
    existing = extra_content.get(DeepseekV4Config) if isinstance(extra_content, dict) else None
    if existing is not None and existing is not DeepseekV4Model:
        raise ValueError(
            "deepseek_v4 is already registered to an incompatible AutoModel class: "
            f"{existing.__module__}.{existing.__name__}"
        )
    AutoModel.register(DeepseekV4Config, DeepseekV4Model)


def _register_deepseek_v4_family() -> None:
    _safe_register_auto_config()
    _safe_register_auto_model()


def patch_method_for_deepseek_v4(model):
    return model


# DeepSeek V4 (Flash/Pro) reuses the standard FusedMoETensorCast (V3-style)
# implementation for the post-gate MoE flow.  V4's NPU runtime fuses dispatch
# + experts + combine into a single `DispatchFFNCombine` kernel, but on the
# msmodeling side we model the same flow as the existing chain of
# init_routing_v2 -> all_to_all -> grouped expert FFN -> all_to_all ->
# unpermute_tokens -> weighted reduction.  As long as each sub-op's perf
# estimator is accurate, the SUM of those costs matches NPU's fused kernel.
# What V4 specifically adds on top of V3 is hash routing (handled by the
# `moe_gating_top_k_hash` op in MoELayer) and the HC pre/post wrapping at the
# decoder-layer level — both orthogonal to the FusedMoE class.
register_model_profile(
    ModelProfile(
        model_type="deepseek_v4",
        moe_module_name="DeepseekV4MoE",
        moe_num_experts_key="n_routed_experts",
        moe_gate_returns_raw_logits=False,
        mtp_block_module_name="DeepseekV4DecoderLayer",
        mla_module_name="DeepseekV4SparseAttention",
        mla_field_names_override=MlaFieldNames(kv_b_proj=None),
        mla_module_class_type=DeepseekV4SparseAttention,
        patch_method=patch_method_for_deepseek_v4,
        moe_gate_router=route_deepseek_v4_gate,
    )
)


class DeepseekV4Config(DeepseekV3Config):
    """DeepSeek V4 config fields consumed by msmodeling.

    This class still subclasses `DeepseekV3Config` to reuse the common HF MoE
    and rotary fields, but V4-specific fields that affect modeling are surfaced
    explicitly here:
        - `compress_ratios` / `layer_types`: per-layer attention policy
        - `topk_limit` / `index_topk`: Lightning indexer top-k
        - `num_hash_layers`: leading MoE layers using hash routing
        - `hc_mult`, `hc_sinkhorn_iters`, `hc_eps`: mHC shape and Sinkhorn cost
        - `o_groups`, `o_lora_rank`: grouped output projection shape
        - `score_func` / `scoring_func`, `route_scale` / `routed_scaling_factor`: V4 routing semantics
        - `expert_dtype`: drives FP4 expert quant-cost selection in ConfigResolver

    V4 fields that are accepted but not currently modeled directly, such as
    `swiglu_limit` and `compress_rope_theta`, are stored for schema visibility
    instead of being silently dropped.
    """

    model_type = "deepseek_v4"

    @staticmethod
    def _normalize_rope_params(rope_params: Optional[dict]) -> Optional[dict]:
        """Coerce `rope_scaling` / `rope_parameters` to the format expected by
        transformers' rope_utils. config.json may carry e.g. `"type": "yarn"`
        and integer-typed `factor` / `beta_fast` / `beta_slow`; transformers
        expects float-typed scalars and a `rope_type` mirror, otherwise the
        rope cache initializer raises.
        """
        if rope_params is None:
            return None
        normalized = dict(rope_params)
        rope_type = normalized.get("rope_type", normalized.get("type"))
        if rope_type is not None:
            normalized["type"] = rope_type
            normalized["rope_type"] = rope_type
        for key in ("factor", "beta_fast", "beta_slow"):
            value = normalized.get(key)
            if value is not None:
                normalized[key] = float(value)
        return normalized

    @staticmethod
    def _normalize_compress_ratios(
        compress_ratios: Optional[list[int]],
        *,
        num_hidden_layers: int,
        config_path: Optional[str],
    ) -> list[int]:
        location = f" in {config_path}" if config_path else ""
        if compress_ratios is None:
            if config_path is None:
                # Transformers may instantiate the config class with no arguments
                # when computing default diffs for logging/serialization.
                # Treat that internal no-source construction as a schema probe,
                # not as a real model config load.
                return []
            raise ValueError(
                f"DeepSeek V4 requires compress_ratios to be defined{location}; expected one entry per decoder layer."
            )
        if len(compress_ratios) < num_hidden_layers:
            raise ValueError(
                "DeepSeek V4 compress_ratios must provide at least one entry per decoder layer"
                f"{location}: expected at least {num_hidden_layers}, got {len(compress_ratios)}."
            )
        if len(compress_ratios) > num_hidden_layers:
            logger.warning(
                "DeepSeek V4 compress_ratios has %d entries%s but the model has %d decoder layers; "
                "ignoring trailing entries: %s.",
                len(compress_ratios),
                location,
                num_hidden_layers,
                compress_ratios[num_hidden_layers:],
            )
            compress_ratios = compress_ratios[:num_hidden_layers]
        invalid = [ratio for ratio in compress_ratios if ratio not in _VALID_COMPRESS_RATIOS]
        if invalid:
            invalid_values = ", ".join(str(ratio) for ratio in sorted(set(invalid)))
            raise ValueError(
                "DeepSeek V4 compress_ratios contains unsupported values"
                f"{location}: {invalid_values}. Supported values: 0, 4, 128."
            )
        return [int(ratio) for ratio in compress_ratios]

    @staticmethod
    def _normalize_layer_policy(
        compress_ratios: Optional[list[int]],
        layer_types: Optional[list[str]],
        *,
        num_hidden_layers: int,
        config_path: Optional[str],
    ) -> tuple[list[int], list[str]]:
        normalized_ratios = DeepseekV4Config._normalize_compress_ratios(
            compress_ratios,
            num_hidden_layers=num_hidden_layers,
            config_path=config_path,
        )
        if not normalized_ratios:
            return [], []

        location = f" in {config_path}" if config_path else ""
        expected_layer_types = [_RATIO_TO_LAYER_TYPE[ratio] for ratio in normalized_ratios]
        if layer_types is None:
            return normalized_ratios, expected_layer_types
        if len(layer_types) < num_hidden_layers:
            raise ValueError(
                "DeepSeek V4 layer_types must provide at least one entry per decoder layer"
                f"{location}: expected at least {num_hidden_layers}, got {len(layer_types)}."
            )
        if len(layer_types) > num_hidden_layers:
            logger.warning(
                "DeepSeek V4 layer_types has %d entries%s but the model has %d decoder layers; "
                "ignoring trailing entries: %s.",
                len(layer_types),
                location,
                num_hidden_layers,
                layer_types[num_hidden_layers:],
            )
            layer_types = layer_types[:num_hidden_layers]
        invalid = [layer_type for layer_type in layer_types if layer_type not in _LAYER_TYPE_TO_RATIO]
        if invalid:
            invalid_values = ", ".join(sorted(set(invalid)))
            raise ValueError(
                "DeepSeek V4 layer_types contains unsupported values"
                f"{location}: {invalid_values}. Supported values: "
                "sliding_attention, compressed_sparse_attention, heavily_compressed_attention."
            )
        provided_ratios = [_LAYER_TYPE_TO_RATIO[layer_type] for layer_type in layer_types]
        for layer_idx, (provided, expected) in enumerate(zip(provided_ratios, normalized_ratios)):
            if provided != expected:
                raise ValueError(
                    "DeepSeek V4 layer_types must match compress_ratios"
                    f"{location}: layer {layer_idx} maps to ratio {provided}, "
                    f"but compress_ratios has {expected}."
                )
        return normalized_ratios, list(layer_types)

    @staticmethod
    def _resolve_config_path(kwargs: dict) -> Optional[str]:
        name_or_path = kwargs.get("_name_or_path")
        if not name_or_path:
            return None
        config_path = Path(name_or_path) / "config.json"
        return str(config_path) if config_path.exists() else str(name_or_path)

    def __init__(
        self,
        topk_limit: Optional[int] = None,
        compress_ratios: Optional[list[int]] = None,
        num_hash_layers: int = 0,
        # V4 default; matches the reference checkpoint's config.json.
        hc_mult: int = 4,
        hc_sinkhorn_iters: int = 20,
        hc_eps: float = 1e-6,
        head_dim: Optional[int] = None,
        o_groups: int = 1,
        o_lora_rank: Optional[int] = None,
        expert_dtype: Optional[str] = None,
        swiglu_limit: Optional[float] = None,
        compress_rope_theta: Optional[float] = None,
        # Reference Gate.forward score function and route scaling. Surfaced on
        # Config so MoELayer.route() can read them via `gate.score_func` /
        # `gate.route_scale` and emit the matmul + score path explicitly.
        score_func: str = "sqrtsoftplus",
        route_scale: float = 1.0,
        layer_types: Optional[list[str]] = None,
        **kwargs,
    ):
        raw_aliases = {
            "dim": "hidden_size",
            "n_layers": "num_hidden_layers",
            "n_heads": "num_attention_heads",
            "n_kv_heads": "num_key_value_heads",
            "n_hash_layers": "num_hash_layers",
            "n_routed_experts": "n_routed_experts",
            "n_shared_experts": "n_shared_experts",
            "n_activated_experts": "num_experts_per_tok",
            "moe_inter_dim": "moe_intermediate_size",
            "rope_head_dim": "qk_rope_head_dim",
            "window_size": "sliding_window",
            "score_func": "scoring_func",
            "route_scale": "routed_scaling_factor",
        }
        for raw_key, hf_key in raw_aliases.items():
            if raw_key in kwargs and hf_key not in kwargs:
                kwargs[hf_key] = kwargs.pop(raw_key)

        if "num_hash_layers" in kwargs:
            num_hash_layers = kwargs.pop("num_hash_layers")
        scoring_func = kwargs.pop("scoring_func", None)
        routed_scaling_factor = kwargs.pop("routed_scaling_factor", None)

        # In the V4 reference (deepseek-ai/DeepSeek-V4-Flash/inference/model.py; Pro shares
        # the same model.py), every Block uses MoE — there is no
        # The HF DeepseekV3Config defaults to 3, which would force layers 0..2
        # into dense MLPs and break the hash-routing layout (where the first
        # `num_hash_layers` layers must be MoE so their gates can carry the
        # `hash=True` flag). Default V4 to 0 unless overridden in config.json.
        kwargs.setdefault("first_k_dense_replace", 0)
        index_topk = kwargs.pop("index_topk", None)
        rope_scaling = self._normalize_rope_params(kwargs.get("rope_scaling"))
        rope_parameters = self._normalize_rope_params(kwargs.get("rope_parameters"))
        config_path = self._resolve_config_path(kwargs)
        if rope_scaling is not None:
            kwargs["rope_scaling"] = rope_scaling
        if rope_parameters is not None:
            kwargs["rope_parameters"] = rope_parameters
        super().__init__(**kwargs)
        normalized_rope = self._normalize_rope_params(
            getattr(self, "rope_parameters", None) or getattr(self, "rope_scaling", None)
        )
        if normalized_rope is not None:
            self.rope_scaling = normalized_rope
            self.rope_parameters = normalized_rope
        self.topk_limit = index_topk if index_topk is not None else topk_limit
        self.compress_ratios, self.layer_types = self._normalize_layer_policy(
            compress_ratios,
            layer_types,
            num_hidden_layers=self.num_hidden_layers,
            config_path=config_path,
        )
        self.num_hash_layers = num_hash_layers
        self.hc_mult = int(hc_mult)
        self.hc_sinkhorn_iters = int(hc_sinkhorn_iters)
        self.hc_eps = float(hc_eps)
        self.o_groups = int(o_groups)
        self.o_lora_rank = None if o_lora_rank is None else int(o_lora_rank)
        self.expert_dtype = expert_dtype
        self.swiglu_limit = swiglu_limit
        self.compress_rope_theta = compress_rope_theta
        self.score_func = str(scoring_func if scoring_func is not None else score_func)
        self.route_scale = float(routed_scaling_factor if routed_scaling_factor is not None else route_scale)
        self.routed_scaling_factor = self.route_scale
        self.head_dim = (
            head_dim
            if head_dim is not None
            else getattr(self, "qk_head_dim", self.hidden_size // self.num_attention_heads)
        )


class DeepseekV4RMSNorm(DeepseekV3RMSNorm):
    pass


class DeepseekV4RotaryEmbedding(DeepseekV3RotaryEmbedding):
    pass


class DeepseekV4MoE(DeepseekV3MoE):
    pass


class DeepseekV4MLP(DeepseekV3MLP):
    def __init__(self, config):
        super().__init__(config)
        self.swiglu_limit = float(getattr(config, "swiglu_limit", 0.0) or 0.0)

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        if self.swiglu_limit > 0:
            hidden_states = torch.ops.tensor_cast.v4_clamped_swiglu(
                gate,
                up,
                self.swiglu_limit,
            )
        else:
            hidden_states = self.act_fn(gate) * up
        return self.down_proj(hidden_states)


def _apply_v4_swiglu_limit(module: nn.Module, swiglu_limit: float) -> None:
    module.swiglu_limit = float(swiglu_limit or 0.0)
    if module.swiglu_limit <= 0:
        return
    if getattr(module, "_v4_swiglu_patched", False):
        return
    original_forward = module.forward

    def forward(x, *args, **kwargs):
        if all(hasattr(module, name) for name in ("gate_proj", "up_proj", "down_proj")):
            gate = module.gate_proj(x)
            up = module.up_proj(x)
            hidden_states = torch.ops.tensor_cast.v4_clamped_swiglu(
                gate,
                up,
                module.swiglu_limit,
            )
            return module.down_proj(hidden_states)
        return original_forward(x, *args, **kwargs)

    module.forward = forward
    module._v4_swiglu_patched = True


class DeepseekV4Indexer(nn.Module):
    def __init__(self, config: "DeepseekV4Config", index_layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = index_layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.topk_limit = config.topk_limit
        self.q_lora_rank = config.q_lora_rank

        self.wq_b = nn.Linear(self.q_lora_rank, self.num_heads * self.head_dim, bias=False)
        self.weights_proj = nn.Linear(
            self.hidden_size,
            self.num_heads,
            dtype=torch.get_default_dtype(),
            bias=False,
        )
        self.softmax_scale = 1.0 / math.sqrt(self.head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_resid: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values_index: "Cache",
        cache_position: torch.LongTensor | None,
    ) -> torch.LongTensor:
        raise NotImplementedError(
            "DeepseekV4Indexer is a builtin shell module and must be replaced by tensor_cast wrappers before execution."
        )


class DeepseekV4Compressor(nn.Module):
    def __init__(self, config: "DeepseekV4Config", layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.compress_ratio = config.compress_ratios[layer_idx] if layer_idx < len(config.compress_ratios) else 0
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
    ) -> torch.Tensor | None:
        raise NotImplementedError(
            "DeepseekV4Compressor is a builtin shell module and must be replaced by tensor_cast wrappers before execution."
        )


class DeepseekV4SparseAttention(nn.Module):
    """V4 sparse attention module (HF-style structure, covers Flash and Pro).

    Mirrors `deepseek-ai/DeepSeek-V4-Flash/inference/model.py:Attention` directly so the
    cost-modeling forward (in `tensor_cast/layers/mla.py`) can emit ops with
    V4-correct shapes:

    * Q path uses `wq_a -> q_norm -> wq_b` with full per-head `head_dim` (512),
      not the standard MLA `qk_nope_head_dim + qk_rope_head_dim` (192).
    * KV path uses a single `wkv` projection of width `head_dim` (shared K/V),
      not the standard MLA `kv_a_proj_with_mqa` (kv_lora_rank + qk_rope_head_dim).
    * O path is grouped: `wo_a` does a per-group projection from
      `n_heads*head_dim/n_groups` to `o_lora_rank`, then `wo_b` collapses the
      stacked groups back to `hidden_size`.

    The builtin module is a structural parameter shell consumed by the V4 MLA
    wrapper in `tensor_cast/layers/mla.py`.
    We therefore keep only the attributes the wrapper's real V4 path reads:
        - `q_a_proj` / `q_a_layernorm` / `q_b_proj`  ↔  reference `wq_a` / `q_norm` / `wq_b`
        - `kv_a_proj_with_mqa` / `kv_a_layernorm`     ↔  reference `wkv` / `kv_norm`
        - `wo_a` / `o_proj`                           ↔  reference `wo_a` / `wo_b`
    There is intentionally no placeholder `kv_b_proj`: V4 shared-KV attention
    does not use the standard MLA kv-b decomposition path.
    """

    def __init__(self, config: DeepseekV4Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.max_position_embeddings = config.max_position_embeddings

        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        # `qk_nope_head_dim` / `v_head_dim` / `kv_lora_rank` are still surfaced
        # because the generic MLA wrappers and cost-model metadata read them,
        # even though V4's real attention path does not perform standard
        # `kv_b_proj` decomposition. We keep the structural scalars, but the
        # actual V4 wrapper path consumes `wkv` + shared-KV sparse attention
        # directly.
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.is_causal = True
        self.compress_ratio = config.compress_ratios[layer_idx] if layer_idx < len(config.compress_ratios) else 0
        self.use_indexer = self.compress_ratio == 4
        self.use_compressor = self.compress_ratio > 0

        # O-projection layout (reference Attention.__init__):
        #   wo_a: ColumnParallelLinear(n_heads*head_dim/n_groups,
        #                              n_groups*o_lora_rank)
        #   wo_b: RowParallelLinear (n_groups*o_lora_rank, hidden_size)
        # With config defaults: per-group in = 64*512/8 = 4096, total out =
        # 8*1024 = 8192, then wo_b projects 8192 -> 4096.
        self.n_groups = int(getattr(config, "o_groups", 1))
        self.o_lora_rank = int(getattr(config, "o_lora_rank") or self.hidden_size)

        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.qk_head_dim, bias=False)
        else:
            # `wq_a` (D -> q_lora_rank)
            self.q_a_proj = nn.Linear(self.hidden_size, config.q_lora_rank, bias=config.attention_bias)
            # `q_norm`
            self.q_a_layernorm = DeepseekV4RMSNorm(config.q_lora_rank)
            # `wq_b` (q_lora_rank -> n_heads * head_dim) — V4 uses the full
            # per-head dim, not the (nope+rope) split.
            self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.head_dim, bias=False)

        # `wkv` (D -> head_dim). V4 produces a single shared 512-wide KV
        # vector per token (last `qk_rope_head_dim` carry RoPE). Stored
        # under the standard MLA `kv_a_proj_with_mqa` attribute so the MLA
        # wrapper's field-name lookup keeps working.
        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size,
            self.head_dim,
            bias=config.attention_bias,
        )
        # `kv_norm`: V4 normalizes the full shared KV (including the RoPE
        # dims) over `head_dim`, unlike standard MLA which only normalizes
        # the latent kv_lora_rank slice.
        self.kv_a_layernorm = DeepseekV4RMSNorm(self.head_dim)

        # `wo_a` (per-group): ColumnParallel-style projection from the
        # per-group head-dim slice to `o_lora_rank`. Modeled here as a single
        # `nn.Linear(per_group_dim, n_groups * o_lora_rank)` so the V4 wrapper
        # forward can reshape its weight into per-group blocks and apply a
        # grouped einsum.
        per_group_in_dim = (self.num_heads * self.head_dim) // self.n_groups
        self.wo_a = nn.Linear(
            per_group_in_dim,
            self.n_groups * self.o_lora_rank,
            bias=False,
        )
        # `wo_b` (n_groups * o_lora_rank -> hidden_size).
        self.o_proj = nn.Linear(
            self.n_groups * self.o_lora_rank,
            self.hidden_size,
            bias=config.attention_bias,
        )
        self.scaling = 1.0 / math.sqrt(self.head_dim)
        self.softmax_scale = self.scaling
        self.attn_sink = nn.Parameter(torch.empty(self.num_heads, dtype=torch.float32))
        self.attention_sink = self.attn_sink
        self.compressor = DeepseekV4Compressor(config, layer_idx) if self.use_compressor else None
        self.indexer = DeepseekV4Indexer(config, layer_idx) if self.use_indexer else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor | None, Tuple[torch.Tensor] | None]:
        raise NotImplementedError(
            "DeepseekV4SparseAttention is a builtin shell module and must be replaced by tensor_cast wrappers before execution."
        )


class DeepseekV4DecoderLayer(nn.Module):
    def __init__(self, config: DeepseekV4Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.hc_mult = int(getattr(config, "hc_mult", 1) or 1)
        self.hc_sinkhorn_iters = int(getattr(config, "hc_sinkhorn_iters", 1) or 1)
        self.hc_eps = float(getattr(config, "hc_eps", 1e-6))
        self.self_attn = DeepseekV4SparseAttention(config=config, layer_idx=layer_idx)

        is_moe_layer = layer_idx >= config.first_k_dense_replace
        if is_moe_layer:
            self.mlp = DeepseekV4MoE(config)
        else:
            self.mlp = DeepseekV4MLP(config)

        # All V4 MoE gates use the reference Gate.forward chain (matmul +
        # score function + indices + gather/normalize/route_scale). The
        # model-specific routing callback is registered on the V4 ModelProfile,
        # so common MoELayer only performs generic dispatch while this builtin
        # module carries the V4 gate metadata consumed by that callback.
        # The post-score tail differs by indices source:
        #   - hash layers (first num_hash_layers MoE layers): tid2eid[input_ids]
        #     -> moe_gating_top_k_hash
        #   - non-hash layers: topk(scores+bias) -> moe_gating_top_k
        # NOTE: This policy is counted in MoE-layer order, not absolute decoder
        # layers. DeepSeek V4 can start with dense layers.
        if is_moe_layer and getattr(self.mlp, "gate", None) is not None:
            self.mlp.gate.score_func = str(getattr(config, "score_func", "sqrtsoftplus"))
            self.mlp.gate.route_scale = float(getattr(config, "route_scale", 1.0))
            moe_layer_idx = layer_idx - int(getattr(config, "first_k_dense_replace", 0) or 0)
            use_hash_routing = moe_layer_idx < int(getattr(config, "num_hash_layers", 0) or 0)
            self.mlp.moe_layer_idx = moe_layer_idx
            self.mlp.use_hash_routing = use_hash_routing
            self.mlp.gate.hash = use_hash_routing
            if use_hash_routing and not hasattr(self.mlp.gate, "tid2eid"):
                self.mlp.gate.register_buffer(
                    "tid2eid",
                    torch.empty(
                        config.vocab_size,
                        config.num_experts_per_tok,
                        dtype=torch.int32,
                    ),
                    persistent=True,
                )
            swiglu_limit = float(getattr(config, "swiglu_limit", 0.0) or 0.0)
            experts = getattr(self.mlp, "experts", None)
            if experts is not None:
                expert_iter = experts if isinstance(experts, nn.ModuleList) else getattr(experts, "experts", [])
                for expert in expert_iter:
                    if expert is not None:
                        _apply_v4_swiglu_limit(expert, swiglu_limit)
            shared_experts = getattr(self.mlp, "shared_experts", None)
            if shared_experts is not None:
                _apply_v4_swiglu_limit(shared_experts, swiglu_limit)

        self.input_layernorm = DeepseekV4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = DeepseekV4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        mix_hc = (2 + self.hc_mult) * self.hc_mult
        hc_dim = self.hc_mult * self.hidden_size
        self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_attn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_attn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
        self.hc_ffn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))

    def _emit_hc_pre(
        self,
        hidden_states: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ):
        """Emit a trace-faithful HC-pre sequence matching reference `hc_pre`.

        Reference (`deepseek-ai/DeepSeek-V4-Flash/inference/model.py:673-681`):
          1. x: [B,S,Hc,D]
          2. x_flat = x.flatten(2).float()
          3. rsqrt = rsqrt(mean(x_flat^2) + eps)
          4. mixes = linear(x_flat, hc_fn) * rsqrt
          5. pre, post, comb = hc_split_sinkhorn(mixes, ...)
          6. y = sum(pre.unsqueeze(-1) * x, dim=2)
          7. return y.to(dtype), post, comb

        Steps 5-7 are folded into the single `hc_pre_sinkhorn` semantic op so
        the cost model can account for the sinkhorn iterations together with
        the weighted reduction back to the original hidden width.
        """
        x_flat = hidden_states.float().flatten(-2)
        rsqrt = torch.ops.tensor_cast.hc_pre_inv_rms(hidden_states, self.hc_mult)
        hc_mixes = torch.matmul(x_flat, hc_fn.transpose(0, 1)) * rsqrt
        reduced_hidden_states, post, comb = torch.ops.tensor_cast.hc_pre_sinkhorn(
            hc_mixes,
            hidden_states,
            hc_scale,
            hc_base,
            self.hc_mult,
            self.hc_sinkhorn_iters,
            self.hc_eps,
        )
        return reduced_hidden_states, post, comb

    @staticmethod
    def _emit_hc_post(
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
        hc_mult: int,
    ) -> torch.Tensor:
        """Emit the HcPost semantic op (model.py 683-686).

        Reference computes `y = post*x + sum(comb*residual, dim=hc)` so the
        residual is FOLDED INTO the op output. The caller MUST NOT add an
        extra `residual + y` on top, otherwise the residual contribution
        gets double-counted (compared to the reference NPU kernel).
        """
        return torch.ops.tensor_cast.hc_post(x, residual, post, comb, hc_mult)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor] | None = None,
        input_ids: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        # Reference Block.forward (deepseek-ai/DeepSeek-V4-Flash/inference/model.py 688-699)
        # wraps BOTH attention and FFN with HC pre/post. The HC post op itself
        # mixes the residual back in, so we deliberately omit a follow-up
        # `+ residual` (unlike the V3 / V32 decoder layers).
        residual = hidden_states
        hidden_states, hc_post_attn, hc_comb_attn = self._emit_hc_pre(
            hidden_states, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
        )
        hidden_states = torch.ops.tensor_cast.rms_norm(
            hidden_states,
            self.input_layernorm.weight.data,
            getattr(self.input_layernorm, "variance_epsilon", self.config.rms_norm_eps),
        )
        # Attention consumes the reduced HC-pre tensor shaped [B, S, 4096].
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = self._emit_hc_post(hidden_states, residual, hc_post_attn, hc_comb_attn, self.hc_mult)
        # HcPost restores the HC-expanded state shaped [B, S, 4, 4096].

        residual = hidden_states
        hidden_states, hc_post_ffn, hc_comb_ffn = self._emit_hc_pre(
            hidden_states, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base
        )
        hidden_states = torch.ops.tensor_cast.rms_norm(
            hidden_states,
            self.post_attention_layernorm.weight.data,
            getattr(
                self.post_attention_layernorm,
                "variance_epsilon",
                self.config.rms_norm_eps,
            ),
        )
        hidden_states = self.mlp(hidden_states, input_ids=input_ids)
        hidden_states = self._emit_hc_post(hidden_states, residual, hc_post_ffn, hc_comb_ffn, self.hc_mult)
        return hidden_states


class DeepseekV4Model(DeepseekV3Model):
    config: DeepseekV4Config

    def __init__(self, config: DeepseekV4Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        layers = []
        for layer_idx in range(config.num_hidden_layers):
            layers.append(DeepseekV4DecoderLayer(config, layer_idx))
        self.layers = nn.ModuleList(layers)

        self.norm = DeepseekV4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = DeepseekV4RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.post_init()
        hc_mult = int(getattr(config, "hc_mult", 1) or 1)
        hc_dim = hc_mult * config.hidden_size
        self.hc_head_fn = nn.Parameter(torch.empty(hc_mult, hc_dim, dtype=torch.float32))
        self.hc_head_base = nn.Parameter(torch.empty(hc_mult, dtype=torch.float32))
        self.hc_head_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds
        hc_mult = int(getattr(self.config, "hc_mult", 1) or 1)
        hidden_states = hidden_states.unsqueeze(2).repeat(1, 1, hc_mult, 1)

        if position_embeddings is None:
            if position_ids is None:
                seq_length = hidden_states.shape[1]
                position_ids = torch.arange(
                    seq_length,
                    dtype=torch.long,
                    device=hidden_states.device,
                ).unsqueeze(0)
            position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                input_ids=input_ids,
                **kwargs,
            )

        # Reference output path applies hc_head before the final norm/logits
        # (deepseek-ai/DeepSeek-V4-Flash/inference/model.py:718-721, 728-735).
        hc_eps = float(getattr(self.config, "hc_eps", 1e-6))
        reduced_hidden_states = torch.ops.tensor_cast.hc_head(
            hidden_states,
            self.hc_head_fn,
            self.hc_head_scale,
            self.hc_head_base,
            hc_mult,
            hc_eps,
        )
        reduced_hidden_states = self.norm(reduced_hidden_states)

        if return_dict is False:
            return (reduced_hidden_states,)

        from transformers.modeling_outputs import BaseModelOutputWithPast

        return BaseModelOutputWithPast(
            last_hidden_state=reduced_hidden_states,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )


_register_deepseek_v4_family()
