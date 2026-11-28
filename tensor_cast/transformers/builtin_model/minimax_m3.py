import logging
import math
from types import MethodType

import torch
import torch.nn.functional as F
from tensor_cast.layers.moe_layer import FusedMoETensorCast, MoELayer
from tensor_cast.performance_model.utils import bytes_of_tensor
from tensor_cast.transformers.transformations import (
    maybe_enable_mtp,
    maybe_reuse_layers,
    patch_attention,
    patch_moe,
    quantize_model,
    shard_model,
    wrap_model,
)

from ..custom_model_registry import (
    ModelProfile,
    register_custom_model,
    register_model_profile,
)
from ..model import TransformerModel
from ... import config
from ...layers.internal import CopyLayerWrapper
from ...layers.minimax_m3_attention import (
    GemmaRMSNormFusedWrapper,
    MiniMaxM3AttentionWrapper,
    RMSNormFusedWrapper,
    _fused_decoder_layer_forward,
)
from ...layers.quant_linear import TensorCastQuantLinear

logger = logging.getLogger(__name__)

_EMPTY_VISUAL_LAYERS_ATTR = "_tensor_cast_empty_visual_layers"
_M3_MXFP8_GROUP_SIZE = 128


class MiniMaxM3MoeExpertMLP(torch.nn.Module):
    """Per-expert MLP for routed experts in MoE layers (layer 3-59).

    Adapts M3's fused ``gate_up_proj`` weight layout to the split
    ``gate_proj`` / ``up_proj`` structure that ``SinkSplitPass`` and
    ``GroupedMatmulSwigluPass`` require.  The fused weight is split once
    at construction (``chunk(2, dim=0)``); forward never emits a fused
    matmul + ``chunk`` so the FX graph matches the DeepSeek expert pattern.

    Registered via ``ModelProfile.custom_expert_module_type`` and instantiated
    per expert by ``transformations._patch_moe_expert_helper``.

    When ``down_proj`` is quantized (``TensorCastQuantLinear``), the fused
    ``m3_swiglu_quant`` op (SwiGLU + post-activation quant) is used so the
    activation scale is produced in-line and fed directly to ``down_proj``,
    avoiding a redundant ``dynamic_quantize_symmetric`` call.
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
            return self.down_proj(hidden_states, activation_scale=activation_scale)
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

    Quantization behavior matches ``MiniMaxM3MoeExpertMLP``: when ``down_proj``
    is a ``TensorCastQuantLinear``, forward uses ``m3_swiglu_quant`` to produce
    the int8 activation and scale in one fused op.
    """

    def __init__(self, mlp, group_size: int = 128):
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
            return self.down_proj(hidden_states, activation_scale=activation_scale)
        hidden_states = torch.ops.tensor_cast.m3_swiglu(gate, up, self.swiglu_alpha, self.swiglu_limit)
        return self.down_proj(hidden_states)


class MiniMaxM3MoELayer(MoELayer):
    def __init__(self, moe_config, module):
        super().__init__(moe_config, module)
        self.fused_moe = MiniMaxM3FusedMoETensorCast(
            self.moe_config,
            self.get_attr(module, "experts", None),
            self.get_attr(module, "shared_experts", None),
            self.get_attr(module, "shared_experts_gate", None),
            self.top_k,
        )
        self.fused_moe.routed_scaling_factor = module.routed_scaling_factor

        self.routed_scaling_factor = module.routed_scaling_factor
        e_score_correction_bias = getattr(module, "e_score_correction_bias", None)
        self.correction_bias = e_score_correction_bias

    def route(self, hidden_states, tp_size=1, tp_rank=0, input_ids=None):
        """Override route() to use fused gate_gatingsigmoid op.

        MiniMax-M3 uses scoring_func="sigmoid" with correction_bias.
        The HF TopKRouter.forward() internally does F.sigmoid + torch.topk +
        gather + div as separate ops. We bypass it and use the fused
        moe_gating_top_k_sigmoid op instead, matching the NPU profiling
        operator npu_moe_gating_top_k(norm_type=1) / gate_gatingsigmoid.
        """
        # Compute raw logits directly from gate weight, bypassing
        # TopKRouter.forward() which contains decomposed sigmoid+topk
        gate_weight = self.gate.weight
        router_logits = F.linear(hidden_states.to(gate_weight.dtype), gate_weight)
        router_logits = router_logits.float()

        if tp_size > 1:
            num_tokens = router_logits.shape[0]
            pad = (-num_tokens) % tp_size
            if pad > 0:
                router_logits = F.pad(router_logits, (0, 0, 0, pad))
            router_logits = torch.tensor_split(router_logits, tp_size, dim=0)[tp_rank]

        topk_weights, topk_indices = torch.ops.tensor_cast.moe_gating_top_k_sigmoid(
            router_logits,
            self.top_k,
            self.routed_scaling_factor,
            self.correction_bias,
        )
        topk_weights = topk_weights.to(hidden_states.dtype)
        return topk_indices, topk_weights


class MiniMaxM3FusedMoETensorCast(FusedMoETensorCast):
    routed_scaling_factor = 1.0

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
        skip_shared_experts: bool = False,
    ) -> torch.Tensor:
        original_shape = hidden_states.shape
        num_tokens = topk_indices.numel()
        split_sizes = self.get_split_sizes(num_tokens, self.top_k)

        expert_indices = topk_indices
        expert_weights = topk_weights
        dispatched_hidden_states = self.dispatch_tokens(
            hidden_states,
            expert_indices,
            split_sizes[0],
            split_sizes[1],
            split_sizes[3],
        )

        experts_hidden_states = []
        if self.ep_group.rank_in_group < self.num_external_shared_experts:
            assert len(dispatched_hidden_states) == 1
            experts_hidden_states.append(self._run_shared_experts(dispatched_hidden_states[0]))
        else:
            for expert_idx, x in enumerate(dispatched_hidden_states):
                num_expert_tokens = x.shape[0]
                pad_size = (-num_expert_tokens) % self._global_tp_size
                if pad_size > 0:
                    x = torch.nn.functional.pad(x, (0, 0, 0, pad_size))
                out = self.experts.call_expert(expert_idx, x, topk_indices, topk_weights)
                experts_hidden_states.append(out[:num_expert_tokens])

        combined_hidden_states = self.combine_tokens(
            experts_hidden_states,
            expert_indices,
            split_sizes[0],
            split_sizes[1],
            split_sizes[3],
        )
        final_hidden_states = (combined_hidden_states * expert_weights.unsqueeze(-1)).sum(dim=-2)
        final_hidden_states = final_hidden_states * self.routed_scaling_factor

        final_hidden_states = final_hidden_states.view(original_shape)

        if self.shared_experts and self.num_external_shared_experts == 0 and not skip_shared_experts:
            shared_output = self._run_shared_experts(hidden_states)
            final_hidden_states = final_hidden_states + shared_output

        return final_hidden_states.to(hidden_states.dtype)


def _patch_minimax_m3_hf_config(hf_config, model_id):
    if hasattr(hf_config, "text_config") and not hasattr(hf_config, "num_hidden_layers"):
        tc = hf_config.text_config
        for attr in [
            "num_hidden_layers",
            "hidden_size",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "intermediate_size",
            "vocab_size",
            "rms_norm_eps",
            "max_position_embeddings",
        ]:
            if hasattr(tc, attr) and not hasattr(hf_config, attr):
                setattr(hf_config, attr, getattr(tc, attr))


def patch_minimax_m3_dense_mlp(model: TransformerModel) -> TransformerModel:
    for name, module in list(model._inner.named_modules()):
        if isinstance(module, MiniMaxM3DenseMLPWrapper):
            continue
        if type(module).__name__ == "MiniMaxM3VLDenseMLP":
            model._replace_module(name, MiniMaxM3DenseMLPWrapper(module))
    return model


def _ensure_empty_visual_layers_for_reuse(model: TransformerModel):
    # MiniMax-M3 text layers live under model.language_model.layers, but the current VL reuse path only
    # reaches language layers when visual layers are present. M3 simulations here do not compile visual
    # layers, so provide an empty visual-layer container to trigger language-layer reuse.
    unwrapped = model.unwrap()
    if not hasattr(unwrapped, _EMPTY_VISUAL_LAYERS_ATTR):
        setattr(unwrapped, _EMPTY_VISUAL_LAYERS_ATTR, torch.nn.ModuleList())


def _get_quantization_config_value(config, key, default=None):
    quantization_config = getattr(config, "quantization_config", None)
    if quantization_config is None and hasattr(config, "text_config"):
        quantization_config = getattr(config.text_config, "quantization_config", None)
    if isinstance(quantization_config, dict):
        return quantization_config.get(key, default)
    return getattr(quantization_config, key, default)


def _is_mxfp8_config(config) -> bool:
    quant_method = _get_quantization_config_value(config, "quant_method")
    return quant_method == "mxfp8"


def _get_weight_block_size(config):
    block_size = _get_quantization_config_value(config, "weight_block_size", (1, 32))
    return tuple(block_size)


def _mxfp8_tensor_weight_size(tensor: torch.Tensor, weight_block_size) -> int:
    # MiniMax-M3 MXFP8 expert weights are stored as one byte per weight element plus one byte per
    # scale block. The expert dimension is a batch dimension, so blocks cover the last two dims.
    rows_per_block, cols_per_block = weight_block_size
    leading_numel = math.prod(tensor.shape[:-2]) if tensor.ndim > 2 else 1
    out_dim, in_dim = tensor.shape[-2:]
    num_scale_blocks = leading_numel * math.ceil(out_dim / rows_per_block) * math.ceil(in_dim / cols_per_block)
    return tensor.numel() + num_scale_blocks


def _mxfp8_weight_size_for_module(module, weight_block_size) -> int:
    """Compute MXFP8-aware weight size for a single module's params/buffers."""
    total_size = 0
    for name, param in module.named_parameters():
        if param.ndim == 3 and (name.endswith(".gate_up_proj") or name.endswith(".down_proj")):
            total_size += _mxfp8_tensor_weight_size(param, weight_block_size)
        else:
            total_size += int(bytes_of_tensor(param))
    for _, buffer in module.named_buffers():
        total_size += int(bytes_of_tensor(buffer))
    return total_size


def estimate_minimax_m3_weight_size(model: TransformerModel) -> int:
    if not _is_mxfp8_config(model.hf_config):
        return model.get_weight_size_nested([model])

    from ...layers.internal import RegionMarkerWrapper

    weight_block_size = _get_weight_block_size(model.hf_config)
    total_size = _mxfp8_weight_size_for_module(model, weight_block_size)

    # Add represented extra weight from RegionMarkerWrapper (layer reuse).
    # CopyLayerWrapper has no _inner, so named_parameters() only sees
    # representative layers. We must account for the repeated layers manually.
    for submodule in model.modules():
        if not isinstance(submodule, RegionMarkerWrapper):
            continue
        repeat_count = submodule.repeat_count
        if repeat_count <= 1:
            continue
        total_size += (repeat_count - 1) * _mxfp8_weight_size_for_module(submodule._inner, weight_block_size)
    return total_size


def patch_minimax_m3_attention(model: TransformerModel) -> TransformerModel:
    sparse_cfg = None
    text_config = model.text_config

    if hasattr(text_config, "sparse_attention_config"):
        sparse_cfg = text_config.sparse_attention_config
    layer_types = getattr(text_config, "layer_types", None)
    has_native_sparse_config = (
        layer_types is not None
        and any(layer_type == "minimax_m3_sparse" for layer_type in layer_types)
        and hasattr(text_config, "index_n_heads")
    )
    if sparse_cfg is None and has_native_sparse_config:
        sparse_cfg = {
            "use_sparse_attention": True,
            "sparse_attention_freq": [1 if layer_type == "minimax_m3_sparse" else 0 for layer_type in layer_types],
            "sparse_num_index_heads": text_config.index_n_heads,
            "sparse_index_dim": text_config.index_head_dim,
            "sparse_topk_blocks": text_config.index_topk_blocks,
            "sparse_block_size": text_config.index_block_size,
            "sparse_local_block": text_config.index_local_blocks,
        }
    if sparse_cfg is None or not sparse_cfg.get("use_sparse_attention", False):
        logger.info("No sparse_attention_config found, skipping M3 attention patch")
        return model

    sparse_attention_freq = sparse_cfg.get("sparse_attention_freq", [])
    num_indexer_heads = sparse_cfg.get("sparse_num_index_heads", 4)
    indexer_head_dim = sparse_cfg.get("sparse_index_dim", 128)
    topk_blocks = sparse_cfg.get("sparse_topk_blocks", 16)
    block_size = sparse_cfg.get("sparse_block_size", 128)
    local_blocks = sparse_cfg.get("sparse_local_block", 1)

    hidden_size = text_config.hidden_size
    num_q_heads = text_config.num_attention_heads
    num_kv_heads = text_config.num_key_value_heads
    head_dim = getattr(text_config, "head_dim", hidden_size // num_q_heads)

    tp_size = 1
    if model.parallel_group_manager is not None and model.parallel_group_manager.tp_group is not None:
        tp_size = model.parallel_group_manager.tp_group.world_size
    per_rank_q_heads = num_q_heads // tp_size
    per_rank_kv_heads = num_kv_heads // tp_size if num_kv_heads >= tp_size else 1
    per_rank_indexer_heads = num_indexer_heads // tp_size if num_indexer_heads >= tp_size else 1

    unwrapped = model.unwrap()
    if not hasattr(unwrapped, "layers"):
        candidates = [
            getattr(unwrapped, "language_model", None),
            getattr(getattr(unwrapped, "model", None), "language_model", None),
            getattr(getattr(unwrapped, "language_model", None), "model", None),
        ]
        for candidate in candidates:
            if candidate is not None and hasattr(candidate, "layers"):
                unwrapped = candidate
                break
        else:
            model_attr = getattr(unwrapped, "model", None)
            if hasattr(model_attr, "layers"):
                unwrapped = model_attr
            else:
                logger.warning("Cannot find layers for M3 attention patch")
                return model

    num_hidden = getattr(model.text_config, "num_hidden_layers", len(unwrapped.layers))
    for layer_idx, layer in enumerate(unwrapped.layers):
        if layer_idx >= num_hidden:
            continue  # skip MTP layers
        self_attn = layer
        while hasattr(self_attn, "_inner"):
            self_attn = self_attn._inner
        if hasattr(self_attn, "self_attn"):
            self_attn = self_attn.self_attn

        is_sparse = layer_idx < len(sparse_attention_freq) and sparse_attention_freq[layer_idx] == 1

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
        )

        parent = layer
        while hasattr(parent, "_inner") and hasattr(parent._inner, "self_attn"):
            parent = parent._inner
        if hasattr(parent, "self_attn"):
            parent.self_attn = wrapper
        else:
            logger.warning("Could not replace self_attn for layer %d", layer_idx)

    return model


def patch_minimax_m3_layernorm(model: TransformerModel) -> TransformerModel:
    """Replace RMSNorm modules with fused tensor_cast ops and patch DecoderLayer.forward.

    Two-level patching:
    1. DecoderLayer.forward: monkey-patch to fuse residual+norm into add_rms_norm2.
    2. RMSNorm modules: replace with GemmaRMSNormFusedWrapper / RMSNormFusedWrapper
       to use fused rms_norm op for Q/K/index norms and add_rms_norm2 for layer norms.
    """
    use_gemma_norm = getattr(model.text_config, "use_gemma_norm", True)

    unwrapped = model.unwrap()
    if not hasattr(unwrapped, "layers"):
        candidates = [
            getattr(unwrapped, "language_model", None),
            getattr(getattr(unwrapped, "model", None), "language_model", None),
            getattr(getattr(unwrapped, "language_model", None), "model", None),
        ]
        for candidate in candidates:
            if candidate is not None and hasattr(candidate, "layers"):
                unwrapped = candidate
                break
        else:
            model_attr = getattr(unwrapped, "model", None)
            if hasattr(model_attr, "layers"):
                unwrapped = model_attr
            else:
                logger.warning("Cannot find layers for M3 layernorm patch")
                return model

    norm_count = 0
    num_hidden = getattr(model.text_config, "num_hidden_layers", len(unwrapped.layers))
    for layer_idx, layer in enumerate(unwrapped.layers):
        if layer_idx >= num_hidden:
            continue  # skip MTP layers
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


def patch_apply_rotary_pos_emb_for_fused_rope(model: TransformerModel) -> TransformerModel:
    """Monkey-patch the HF module-level ``apply_rotary_pos_emb`` to emit ``fused_rope``.

    M3 uses partial RoPE (``rotary_dim`` < ``head_dim``): the upstream function
    slices ``q[..., :rotary_dim]`` / ``q[..., rotary_dim:]``, rotates the first
    slice, then concatenates the unrotated tail back — producing a
    ``slice → rotate_half → mul/add → cat`` subgraph.

    The standard RoPE fusion pipeline (``NormalRopePattern`` → ``apply_rope`` →
    ``FusedRopePass`` → ``fused_rope``) cannot match this pattern because
    ``NormalRopePattern`` only matches ``(q * cos) + (rotate_half(q) * sin)``
    applied to the *entire* tensor, with no slice/cat.  Extending the pattern to
    handle partial RoPE would require new pattern code; until then this
    monkey-patch is the only way to make dense layers (layer 0-2, whose forward
    delegates to ``self._inner`` and thus calls the HF function) emit
    ``fused_rope`` instead of decomposed aten ops.

    Sparse layers are unaffected: ``MiniMaxM3AttentionWrapper`` calls
    ``fused_rope`` directly and never reaches the HF function.


    """
    import importlib

    hf_modeling = importlib.import_module("transformers.models.minimax_m3_vl.modeling_minimax_m3_vl")

    rotary_dim = getattr(model.text_config, "rotary_dim", 64)

    def fused_apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
        """Replace apply_rotary_pos_emb with fused_rope op for NPU profiling.

        Input format (same as original apply_rotary_pos_emb):
            q: (batch, num_heads, seq_len, head_dim) BHSD
            k: (batch, num_kv_heads, seq_len, head_dim) BHSD
            cos: (batch, seq_len, rotary_dim)
            sin: (batch, seq_len, rotary_dim)
        """
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)

        cos_sin = torch.cat([cos, sin], dim=-1)

        batch, num_q_heads, seq_len, head_dim = q.shape
        _, num_kv_heads, _, _ = k.shape

        q_3d = q.transpose(1, 2).reshape(batch * seq_len, num_q_heads, head_dim)
        k_3d = k.transpose(1, 2).reshape(batch * seq_len, num_kv_heads, head_dim)
        cos_sin_3d = cos_sin.transpose(1, 2).reshape(batch * seq_len, -1)

        q_out_3d, k_out_3d = torch.ops.tensor_cast.fused_rope(q_3d, k_3d, cos_sin_3d, rotary_dim, True)

        q_embed = q_out_3d.reshape(batch, seq_len, num_q_heads, head_dim).transpose(1, 2)
        k_embed = k_out_3d.reshape(batch, seq_len, num_kv_heads, head_dim).transpose(1, 2)

        if head_dim > rotary_dim:
            q_pass = q[..., rotary_dim:]
            k_pass = k[..., rotary_dim:]
            q_embed = torch.cat([q_embed, q_pass], dim=-1)
            k_embed = torch.cat([k_embed, k_pass], dim=-1)

        return q_embed, k_embed

    hf_modeling.apply_rotary_pos_emb = fused_apply_rotary_pos_emb
    logger.info("Patched apply_rotary_pos_emb to use fused_rope for MiniMax-M3")
    return model


@register_custom_model("minimax_m3_vl")
def _(model: TransformerModel):
    # M3's MoE performance model relies on dispatch_ffn_combine fusion;
    # enable it here so the global default stays off for other models.
    config.compilation.fusion_patterns.enable_dispatch_ffn_combine = True
    model = wrap_model(model)
    model = maybe_enable_mtp(model)
    _ensure_empty_visual_layers_for_reuse(model)
    model = maybe_reuse_layers(model)
    model = patch_minimax_m3_attention(model)
    model = patch_minimax_m3_layernorm(model)
    model = patch_apply_rotary_pos_emb_for_fused_rope(model)
    model = patch_attention(model)
    model = patch_minimax_m3_dense_mlp(model)
    model = patch_moe(
        model,
        lambda moe_config, module: MiniMaxM3MoELayer(moe_config, module),
    )
    model = quantize_model(model)
    model = shard_model(model)
    return model


register_model_profile(
    ModelProfile(
        model_type="minimax_m3_vl",
        moe_module_name="MiniMaxM3VLSparseMoeBlock",
        moe_gate_returns_raw_logits=False,
        moe_num_experts_key="num_local_experts",
        mtp_block_module_name="MiniMaxM3DecoderLayer",
        hf_config_patch_method=_patch_minimax_m3_hf_config,
        weight_size_estimator=estimate_minimax_m3_weight_size,
        language_layers_path_str="language_model.layers",
        language_module_path="language_model",
        visual_layers_module_path=_EMPTY_VISUAL_LAYERS_ATTR,
        visual_layers_path_str=_EMPTY_VISUAL_LAYERS_ATTR,
        moe_field_names_override={
            "shared_experts": "shared_experts",
        },
        custom_expert_module_type=MiniMaxM3MoeExpertMLP,
    )
)
