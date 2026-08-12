import logging
from types import MethodType

import torch
import torch.nn.functional as F
from tensor_cast.transformers.transformations import patch_moe

from ..custom_model_registry import (
    ModelProfile,
    register_model_profile,
)
from ..model import TransformerModel
from ...layers.internal import CopyLayerWrapper
from ...layers.minimax_m3_attention import (
    GemmaRMSNormFusedWrapper,
    MiniMaxM3AttentionWrapper,
    RMSNormFusedWrapper,
    _fused_decoder_layer_forward,
)
from ...layers.quant_linear import TensorCastQuantLinear

logger = logging.getLogger(__name__)

_M3_MXFP8_GROUP_SIZE = 128
_MINIMAX_M3_DECODER_LAYER_NAME = "MiniMaxM3VLDecoderLayer"
_MINIMAX_M3_LANGUAGE_MODULE_PATH = "language_model"


class MiniMaxM3MoeExpertMLP(torch.nn.Module):
    """Per-expert MLP for routed experts in MoE layers (layer 3-59).

    Adapts M3's fused ``gate_up_proj`` weight layout to the split
    ``gate_proj`` / ``up_proj`` structure that ``SinkSplitPass`` and
    ``GroupedMatmulSwigluPass`` require.  The fused weight is split once
    at construction (``chunk(2, dim=0)``); forward never emits a fused
    matmul + ``chunk`` so the FX graph matches the DeepSeek expert pattern.

    Registered through ``ModelProfile.custom_expert_module_type`` as the
    MiniMax-M3-specific expert adapter.

    When ``down_proj`` is quantized, ``m3_swiglu_quant`` produces the int8
    activation and scale for the quantized down projection.
    """

    def __init__(
        self,
        original_experts_module: torch.nn.Module,
        expert_idx: int,
        group_size: int = _M3_MXFP8_GROUP_SIZE,
    ):
        super().__init__()
        self.expert_idx = expert_idx
        self.swiglu_alpha = original_experts_module.swiglu_alpha
        self.swiglu_limit = original_experts_module.swiglu_limit
        self.group_size = group_size

        hidden_dim = original_experts_module.hidden_dim
        intermediate_dim = original_experts_module.intermediate_dim
        self.gate_proj = torch.nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj = torch.nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = torch.nn.Linear(intermediate_dim, hidden_dim, bias=False)

        with torch.no_grad():
            gate_up_weight = original_experts_module.gate_up_proj.data[expert_idx]
            gate_weight, up_weight = gate_up_weight.chunk(2, dim=0)
            self.gate_proj.weight.copy_(gate_weight)
            self.up_proj.weight.copy_(up_weight)
            self.down_proj.weight.copy_(original_experts_module.down_proj.data[expert_idx])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(hidden_states)
        up = self.up_proj(hidden_states)
        if isinstance(self.down_proj, TensorCastQuantLinear):
            hidden_states, activation_scale = torch.ops.tensor_cast.m3_swiglu_quant(
                gate,
                up,
                self.swiglu_alpha,
                self.swiglu_limit,
                self.group_size,
            )
            return self.down_proj(hidden_states, external_activation_scale=activation_scale)
        hidden_states = torch.ops.tensor_cast.m3_swiglu(gate, up, self.swiglu_alpha, self.swiglu_limit)
        return self.down_proj(hidden_states)


class MiniMaxM3DenseMLPWrapper(torch.nn.Module):
    """Dense FFN wrapper for non-MoE layers (layer 0-2 and MTP blocks).

    M3's upstream ``MiniMaxM3VLDenseMLP`` stores a fused ``gate_up_proj``
    (``nn.Linear`` with ``out_features = 2 * intermediate``) and runs a
    single matmul followed by ``chunk(2, dim=-1)`` in forward.  This wrapper
    splits the weight into separate ``gate_proj`` / ``up_proj`` linears so the
    FX graph is structurally identical to ``MiniMaxM3MoeExpertMLP`` and the
    same compile passes (``SinkSplitPass``, ``GroupedMatmulSwigluPass``) apply.

    Installed by ``patch_minimax_m3_dense_mlp`` which replaces every
    ``MiniMaxM3VLDenseMLP`` module with this wrapper.  ``down_proj`` is reused
    from the original module (not rebuilt) to preserve any quantization state.

    Quantization behavior matches ``MiniMaxM3MoeExpertMLP``.
    """

    def __init__(self, mlp, group_size: int = _M3_MXFP8_GROUP_SIZE):
        super().__init__()
        self.swiglu_alpha = mlp.swiglu_alpha
        self.swiglu_limit = mlp.swiglu_limit
        self.group_size = group_size

        hidden_dim = mlp.gate_up_proj.in_features
        intermediate_dim = mlp.gate_up_proj.out_features // 2
        self.gate_proj = torch.nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj = torch.nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = mlp.down_proj

        with torch.no_grad():
            gate_weight, up_weight = mlp.gate_up_proj.weight.chunk(2, dim=0)
            self.gate_proj.weight.copy_(gate_weight)
            self.up_proj.weight.copy_(up_weight)

    def forward(self, hidden_states):
        gate = self.gate_proj(hidden_states)
        up = self.up_proj(hidden_states)
        if isinstance(self.down_proj, TensorCastQuantLinear):
            hidden_states, activation_scale = torch.ops.tensor_cast.m3_swiglu_quant(
                gate,
                up,
                self.swiglu_alpha,
                self.swiglu_limit,
                self.group_size,
            )
            return self.down_proj(hidden_states, external_activation_scale=activation_scale)
        hidden_states = torch.ops.tensor_cast.m3_swiglu(gate, up, self.swiglu_alpha, self.swiglu_limit)
        return self.down_proj(hidden_states)


def route_minimax_m3_gate(
    gate: torch.nn.Module,
    hidden_states: torch.Tensor,
    top_k: int,
    input_ids=None,
    moe_layer_idx=None,
    *,
    tp_size: int = 1,
    tp_rank: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    gate_weight = gate.weight
    router_input = torch.ops.aten.alias.default(hidden_states)
    router_logits = F.linear(router_input.to(gate_weight.dtype), gate_weight).float()

    if tp_size > 1:
        num_tokens = router_logits.shape[0]
        pad = (-num_tokens) % tp_size
        if pad > 0:
            router_logits = F.pad(router_logits, (0, 0, 0, pad))
        router_logits = torch.tensor_split(router_logits, tp_size, dim=0)[tp_rank]

    topk_weights, topk_indices = torch.ops.tensor_cast.moe_gating_top_k_sigmoid(
        router_logits,
        top_k,
        getattr(gate, "routed_scaling_factor", 1.0),
        getattr(gate, "e_score_correction_bias", None),
    )
    return topk_indices, topk_weights.to(hidden_states.dtype)


def patch_minimax_m3_dense_mlp(model: TransformerModel) -> TransformerModel:
    for name, module in list(model._inner.named_modules()):
        if isinstance(module, MiniMaxM3DenseMLPWrapper):
            continue
        if type(module).__name__ == "MiniMaxM3VLDenseMLP":
            model._replace_module(name, MiniMaxM3DenseMLPWrapper(module))
    return model


def _get_minimax_m3_effective_text_config(model: TransformerModel):
    candidates = (
        getattr(model, "text_config", None),
        getattr(getattr(model, "_inner", None), "hf_config", None),
        getattr(model, "hf_config", None),
    )
    for config in candidates:
        if config is None:
            continue
        text_config = getattr(config, "text_config", None)
        if text_config is not None and hasattr(text_config, "hidden_size"):
            return text_config
        if hasattr(config, "hidden_size"):
            return config
    return model.text_config


def _get_config_value(config, key: str, default=None):
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _resolve_minimax_m3_sparse_attention_config(text_config):
    layer_types = getattr(text_config, "layer_types", None)
    sparse_config = getattr(text_config, "sparse_attention_config", None)

    if layer_types:
        sparse_attention_freq = [1 if layer_type == "minimax_m3_sparse" else 0 for layer_type in layer_types]
    else:
        sparse_attention_freq = _get_config_value(sparse_config, "sparse_attention_freq")
        if sparse_attention_freq is None:
            sparse_attention_freq = _get_config_value(sparse_config, "sparse_disable_index_value", [])
        sparse_attention_freq = [int(value) for value in sparse_attention_freq]

    num_indexer_heads = getattr(
        text_config,
        "index_n_heads",
        _get_config_value(sparse_config, "sparse_num_index_heads"),
    )
    indexer_head_dim = getattr(
        text_config,
        "index_head_dim",
        _get_config_value(sparse_config, "sparse_index_dim"),
    )
    topk_blocks = getattr(
        text_config,
        "index_topk_blocks",
        _get_config_value(sparse_config, "sparse_topk_blocks"),
    )
    block_size = getattr(
        text_config,
        "index_block_size",
        _get_config_value(sparse_config, "sparse_block_size"),
    )
    local_blocks = getattr(
        text_config,
        "index_local_blocks",
        _get_config_value(sparse_config, "sparse_local_block"),
    )

    return sparse_attention_freq, num_indexer_heads, indexer_head_dim, topk_blocks, block_size, local_blocks


def _iter_minimax_m3_decoder_layers(model: TransformerModel):
    for _, module in list(model._inner.named_modules()):
        if type(module).__name__ == _MINIMAX_M3_DECODER_LAYER_NAME:
            yield module


def patch_minimax_m3_attention(model: TransformerModel) -> TransformerModel:
    text_config = _get_minimax_m3_effective_text_config(model)
    (
        sparse_attention_freq,
        num_indexer_heads,
        indexer_head_dim,
        topk_blocks,
        block_size,
        local_blocks,
    ) = _resolve_minimax_m3_sparse_attention_config(text_config)
    if not sparse_attention_freq:
        logger.info("No MiniMax-M3 sparse attention config found, skipping M3 attention patch")
        return model

    if None in (num_indexer_heads, indexer_head_dim, topk_blocks, block_size, local_blocks):
        raise ValueError("MiniMax-M3 sparse attention config is incomplete")

    hidden_size = text_config.hidden_size
    num_q_heads = text_config.num_attention_heads
    num_kv_heads = text_config.num_key_value_heads
    head_dim = getattr(text_config, "head_dim", hidden_size // num_q_heads)

    tp_size = 1
    tp_rank = 0
    if model.parallel_group_manager is not None and model.parallel_group_manager.tp_group is not None:
        tp_size = model.parallel_group_manager.tp_group.world_size
        tp_rank = model.parallel_group_manager.tp_group.rank_in_group
    per_rank_q_heads = num_q_heads // tp_size
    per_rank_kv_heads = num_kv_heads // tp_size if num_kv_heads >= tp_size else 1
    per_rank_indexer_heads = num_indexer_heads // tp_size if num_indexer_heads >= tp_size else 1
    if num_indexer_heads >= tp_size:
        indexer_head_start = tp_rank * per_rank_indexer_heads
    else:
        replicas_per_indexer_head = tp_size // num_indexer_heads
        indexer_head_start = tp_rank // replicas_per_indexer_head

    patched_count = 0
    for layer in _iter_minimax_m3_decoder_layers(model):
        if isinstance(layer, CopyLayerWrapper):
            continue
        inner = layer
        while hasattr(inner, "_inner"):
            inner = inner._inner
        self_attn = getattr(inner, "self_attn", None)
        if self_attn is None or isinstance(self_attn, MiniMaxM3AttentionWrapper):
            continue

        layer_idx = getattr(self_attn, "layer_idx", patched_count)
        if layer_idx < len(sparse_attention_freq):
            is_sparse = sparse_attention_freq[layer_idx] == 1
        else:
            is_sparse = sparse_attention_freq[-1] == 1

        rotary_dim = getattr(text_config, "rotary_dim", head_dim)
        wrapper = MiniMaxM3AttentionWrapper(
            original_module=self_attn,
            is_sparse_layer=is_sparse,
            hidden_size=hidden_size,
            num_q_heads=per_rank_q_heads,
            num_kv_heads=per_rank_kv_heads,
            head_dim=head_dim,
            num_indexer_heads=per_rank_indexer_heads,
            indexer_head_dim=indexer_head_dim,
            topk_blocks=topk_blocks,
            block_size=block_size,
            local_blocks=local_blocks,
            rotary_dim=rotary_dim,
            indexer_head_start=indexer_head_start,
        )

        inner.self_attn = wrapper
        patched_count += 1

    return model


def patch_minimax_m3_layernorm(model: TransformerModel) -> TransformerModel:
    """Replace RMSNorm modules with fused tensor_cast ops and patch DecoderLayer.forward.

    Two-level patching:
    1. DecoderLayer.forward: monkey-patch to fuse residual+norm into add_rms_norm2.
    2. RMSNorm modules: replace with GemmaRMSNormFusedWrapper / RMSNormFusedWrapper
       to use fused rms_norm op for Q/K/index norms and add_rms_norm2 for layer norms.
    """
    text_config = _get_minimax_m3_effective_text_config(model)
    use_gemma_norm = getattr(text_config, "use_gemma_norm", True)

    norm_count = 0
    for layer in _iter_minimax_m3_decoder_layers(model):
        # Skip CopyLayerWrapper: it only copies a region result and never
        # executes the real decoder forward, so patching it is unnecessary
        # (and would fail because it lacks input_layernorm etc.).
        if isinstance(layer, CopyLayerWrapper):
            continue
        inner = layer
        while hasattr(inner, "_inner"):
            inner = inner._inner

        # Patch input_layernorm and post_attention_layernorm with fused wrapper
        for norm_name in ["input_layernorm", "post_attention_layernorm"]:
            original_norm = getattr(inner, norm_name, None)
            if original_norm is not None and not isinstance(
                original_norm, (GemmaRMSNormFusedWrapper, RMSNormFusedWrapper)
            ):
                wrapper = GemmaRMSNormFusedWrapper(original_norm)
                setattr(inner, norm_name, wrapper)
                norm_count += 1

        # Patch q_norm, k_norm (all layers have these on self_attn)
        self_attn = getattr(inner, "self_attn", None)
        if self_attn is not None:
            attn_inner = self_attn
            while hasattr(attn_inner, "_inner"):
                attn_inner = attn_inner._inner
            for norm_name in ["q_norm", "k_norm"]:
                original_norm = getattr(attn_inner, norm_name, None)
                if original_norm is not None and not isinstance(original_norm, RMSNormFusedWrapper):
                    wrapper = RMSNormFusedWrapper(original_norm, is_gemma=use_gemma_norm)
                    setattr(attn_inner, norm_name, wrapper)
                    norm_count += 1

            # Patch indexer.q_norm, indexer.k_norm (sparse layers only)
            indexer = getattr(attn_inner, "indexer", None)
            if indexer is not None:
                for norm_name in ["q_norm", "k_norm"]:
                    original_norm = getattr(indexer, norm_name, None)
                    if original_norm is not None and not isinstance(original_norm, RMSNormFusedWrapper):
                        wrapper = RMSNormFusedWrapper(original_norm, is_gemma=use_gemma_norm)
                        setattr(indexer, norm_name, wrapper)
                        norm_count += 1

        # Monkey-patch DecoderLayer.forward to use add_rms_norm2
        inner.forward = MethodType(_fused_decoder_layer_forward, inner)

    # Note: model.norm (final norm) is NOT patched to rms_norm because
    # profiling counts only per-layer norms (234 = 60*2 q/k + 57*2 indexer q/k).
    # The final norm stays as original to match profiling call counts.

    logger.info("Patched %d RMSNorm modules with fused ops (including add_rms_norm2)", norm_count)
    return model


def patch_minimax_m3_moe_gate_attrs(model: TransformerModel) -> TransformerModel:
    for module in model._inner.modules():
        if type(module).__name__ != "MiniMaxM3VLSparseMoeBlock":
            continue
        gate = getattr(module, "gate", None)
        if gate is not None and hasattr(module, "routed_scaling_factor"):
            gate.routed_scaling_factor = module.routed_scaling_factor
    return model


def patch_method_for_minimax_m3(model: TransformerModel) -> TransformerModel:
    # MiniMax-M3 uses partial/3D RoPE; the generic rotary cache rewrites
    # position_embeddings to a shape that is incompatible with its attention.
    model.model_config.cache_rotary_embedding = False
    model = patch_minimax_m3_attention(model)
    model = patch_minimax_m3_layernorm(model)
    model = patch_minimax_m3_dense_mlp(model)
    model = patch_minimax_m3_moe_gate_attrs(model)
    model = patch_moe(model)
    return model


register_model_profile(
    ModelProfile(
        model_type="minimax_m3_vl",
        moe_module_name="MiniMaxM3VLSparseMoeBlock",
        moe_num_experts_key="num_local_experts",
        moe_gate_router=route_minimax_m3_gate,
        patch_method=patch_method_for_minimax_m3,
        mtp_block_module_name=_MINIMAX_M3_DECODER_LAYER_NAME,
        language_module_path=_MINIMAX_M3_LANGUAGE_MODULE_PATH,
        language_layers_path_str=f"{_MINIMAX_M3_LANGUAGE_MODULE_PATH}.layers",
        custom_expert_module_type=MiniMaxM3MoeExpertMLP,
    )
)
