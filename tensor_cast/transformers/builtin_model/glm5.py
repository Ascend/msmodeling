import inspect

import torch
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast

from ...layers.glm5 import Glm5SparseAttention
from ...layers.internal import CopyLayerWrapper, RegionMarkerWrapper
from ..custom_model_registry import ModelProfile, MoeExpertMLP, register_model_profile


_GLM5_ATTENTION_OUTPUT_HIDDEN_STATES_INDEX = 0
_GLM5_ATTENTION_OUTPUT_TOPK_INDICES_INDEX = 2


class Glm5DecoderLayerCompat(torch.nn.Module):
    """Keep the GLM-5 attention auxiliary output across old HF decoder layers."""

    def __init__(self, layer: torch.nn.Module):
        super().__init__()
        self._inner = layer

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values=None,
        use_cache: bool | None = False,
        cache_position: torch.Tensor | None = None,
        position_embeddings=None,
        prev_topk_indices: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        residual = hidden_states
        hidden_states = self._inner.input_layernorm(hidden_states)
        attention_outputs = self._inner.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            prev_topk_indices=prev_topk_indices,
            **kwargs,
        )
        if len(attention_outputs) <= _GLM5_ATTENTION_OUTPUT_TOPK_INDICES_INDEX:
            raise ValueError("GLM-5 attention must return (attention_output, attention_weights, topk_indices)")
        # GLM-MoE-DSA returns attention output at 0, attention weights at 1,
        # and the GLM-5-specific top-k index tensor at 2.
        hidden_states = residual + attention_outputs[_GLM5_ATTENTION_OUTPUT_HIDDEN_STATES_INDEX]
        topk_indices = attention_outputs[_GLM5_ATTENTION_OUTPUT_TOPK_INDICES_INDEX]

        residual = hidden_states
        hidden_states = self._inner.post_attention_layernorm(hidden_states)
        hidden_states = self._inner.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, topk_indices


def _prepare_glm5_decoder_layer(layer: torch.nn.Module) -> torch.nn.Module:
    """Patch real decoder layers while preserving repetition wrappers."""
    if isinstance(layer, CopyLayerWrapper):
        # CopyLayerWrapper replays its representative and has no executable
        # decoder layer of its own.
        return layer
    if isinstance(layer, RegionMarkerWrapper):
        if not isinstance(layer._inner, Glm5DecoderLayerCompat):
            layer._inner = Glm5DecoderLayerCompat(layer._inner)
        return layer
    if isinstance(layer, Glm5DecoderLayerCompat):
        return layer
    return Glm5DecoderLayerCompat(layer)


class Glm5ModelCompat(torch.nn.Module):
    """Run IndexShare explicitly when the installed HF model lacks that loop."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        prepared_layers = torch.nn.ModuleList([_prepare_glm5_decoder_layer(layer) for layer in model.layers])
        model.layers = prepared_layers
        self._inner = model

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            inner = self._modules.get("_inner")
            if inner is not None and hasattr(inner, name):
                return getattr(inner, name)
            raise

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values=None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.Tensor | None = None,
        **kwargs,
    ):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self._inner.embed_tokens(input_ids)

        if use_cache is None:
            use_cache = getattr(self._inner.config, "use_cache", False)
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self._inner.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self._inner.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)
        topk_indices = None
        for decoder_layer in self._inner.layers[: self._inner.config.num_hidden_layers]:
            hidden_states, topk_indices = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                prev_topk_indices=topk_indices,
                **kwargs,
            )

        hidden_states = self._inner.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


def _decoder_supports_prev_topk(layer: torch.nn.Module) -> bool:
    while isinstance(layer, RegionMarkerWrapper):
        layer = layer._inner
    try:
        return "prev_topk_indices" in inspect.signature(type(layer).forward).parameters
    except (TypeError, ValueError):
        return False


def _replace_child_module(root: torch.nn.Module, target: torch.nn.Module, replacement: torch.nn.Module) -> bool:
    for parent in root.modules():
        for name, child in parent._modules.items():
            if child is target:
                setattr(parent, name, replacement)
                return True
    return False


def _resolve_glm5_mtp_block_owner(mtp_layer: torch.nn.Module) -> torch.nn.Module | None:
    """Find the real MTP layer behind repetition wrappers.

    Copy layers replay the representative region and do not own an executable
    ``mtp_block``. A region marker owns the original MTP layer through ``_inner``.
    """
    if isinstance(mtp_layer, CopyLayerWrapper):
        return None
    while isinstance(mtp_layer, RegionMarkerWrapper):
        mtp_layer = mtp_layer._inner
    return mtp_layer


def patch_glm5_model(model) -> None:
    """Bridge Transformers releases before the GLM-5 IndexShare decoder update."""
    root = model.unwrap()
    base_model = root
    if not hasattr(base_model, "layers") and hasattr(base_model, "model"):
        base_model = base_model.model
    if not hasattr(base_model, "layers") or not base_model.layers:
        return

    if _decoder_supports_prev_topk(base_model.layers[0]):
        return

    if not isinstance(base_model, Glm5ModelCompat):
        replacement = Glm5ModelCompat(base_model)
        if not _replace_child_module(model, base_model, replacement):
            raise RuntimeError("Unable to install the GLM-5 Transformers compatibility wrapper")

    from ...layers.mtp import MtpWrapper

    for module in model.modules():
        if not isinstance(module, MtpWrapper):
            continue
        for mtp_layer in module.mtp.layers:
            mtp_block_owner = _resolve_glm5_mtp_block_owner(mtp_layer)
            if mtp_block_owner is None:
                continue
            if not isinstance(mtp_block_owner.mtp_block, Glm5DecoderLayerCompat):
                mtp_block_owner.mtp_block = Glm5DecoderLayerCompat(mtp_block_owner.mtp_block)


register_model_profile(
    ModelProfile(
        model_type="glm_moe_dsa",
        moe_module_name="GlmMoeDsaMoE",
        moe_num_experts_key="n_routed_experts",
        moe_gate_returns_raw_logits=True,
        mla_module_name="GlmMoeDsaAttention",
        mla_module_class_type=Glm5SparseAttention,
        mtp_block_module_name="GlmMoeDsaDecoderLayer",
        custom_expert_module_type=MoeExpertMLP,
        patch_method=patch_glm5_model,
    )
)
