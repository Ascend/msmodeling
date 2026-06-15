import logging

import torch

from ...model_config import MoEFieldNames

from ...utils import exact_division

from ..custom_model_registry import (
    ModelProfile,
    register_model_profile,
    resolve_visual_config,
)

logger = logging.getLogger(__name__)

QWEN3_5_VISUAL_CONFIG = resolve_visual_config({})


def _set_qwen3_5_linear_attn_tp_size(model):
    tp_size = model.parallel_group_manager.tp_group.world_size
    if tp_size <= 1:
        return

    for module in model._inner.modules():
        if hasattr(module, "num_k_heads") and hasattr(module, "num_v_heads"):
            if module.num_k_heads % tp_size != 0:
                raise ValueError(
                    "Qwen3.5 linear attention requires tp_size to divide "
                    f"num_k_heads exactly, but got num_k_heads={module.num_k_heads} "
                    f"and tp_size={tp_size}."
                )
            if module.num_v_heads % tp_size != 0:
                raise ValueError(
                    "Qwen3.5 linear attention requires tp_size to divide "
                    f"num_v_heads exactly, but got num_v_heads={module.num_v_heads} "
                    f"and tp_size={tp_size}."
                )
            if module.head_k_dim != module.head_v_dim:
                raise ValueError(
                    "Qwen3.5 linear attention TP sharding requires head_k_dim to equal "
                    f"head_v_dim, but got head_k_dim={module.head_k_dim} and "
                    f"head_v_dim={module.head_v_dim}."
                )
            module.tensor_cast_tp_size = tp_size


def patch_method_for_qwen3_5(model):
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5GatedDeltaNet,
        Qwen3_5DecoderLayer,
        Qwen3_5Model,
        Qwen3_5TextModel,
    )
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeGatedDeltaNet,
        Qwen3_5MoeDecoderLayer,
        Qwen3_5MoeModel,
        Qwen3_5MoeTextModel,
    )

    def _get_local_linear_attn_heads(self):
        tp_size = getattr(self, "tensor_cast_tp_size", 1)
        return (
            exact_division(self.num_k_heads, tp_size),
            exact_division(self.num_v_heads, tp_size),
        )

    def _has_previous_state(cache_params, cache_position, layer_idx):
        if cache_position is not None and hasattr(cache_position, "numel") and cache_position.numel() > 0:
            has_previous_state = getattr(cache_position, "tensor_cast_has_previous_state", None)
            if has_previous_state is not None:
                return bool(has_previous_state)
            is_meta = hasattr(cache_position, "is_meta") and cache_position.is_meta
            if not is_meta:
                try:
                    return cache_position[0].item() > 0
                except RuntimeError:
                    return False

        if cache_params is None or torch.compiler.is_compiling():
            return False
        try:
            return cache_params.has_previous_state(layer_idx)
        except TypeError:
            try:
                return cache_params.has_previous_state()
            except (AttributeError, RuntimeError):
                return False
        except (AttributeError, RuntimeError):
            return False

    def _is_recurrent_decode_batch(seq_len, cache_position):
        if seq_len == 1:
            return True

        query_lens = getattr(cache_position, "tensor_cast_query_lens", None)
        is_decode = getattr(cache_position, "tensor_cast_is_decode", None)
        if query_lens is None or is_decode is None:
            logger.debug(
                "Missing metadata for recurrent decode detection: "
                "query_lens=%s, is_decode=%s. Falling back to chunk path.",
                query_lens,
                is_decode,
            )
            return False
        if sum(query_lens) != seq_len or not all(is_decode):
            return False

        num_mtp_tokens = int(getattr(cache_position, "tensor_cast_num_mtp_tokens", 0) or 0)
        recurrent_query_lens = {1}
        if num_mtp_tokens > 0:
            recurrent_query_lens.add(1 + num_mtp_tokens)
        return all(query_len in recurrent_query_lens for query_len in query_lens)

    def _patched_update_linear_attn_mask(self, attention_mask, cache_position):
        # Qwen3.5 linear-attention mask path has tensor-value-based branches that
        # are compile-unfriendly under TensorCast tracing; return None in compile
        # mode to keep a stable graph and align with decode behavior.
        if torch.compiler.is_compiling():
            return None

        if cache_position is not None and hasattr(cache_position, "device") and cache_position.device.type == "meta":
            return attention_mask

        linear_attn_mask = attention_mask

        is_meta_tensor = (hasattr(cache_position, "is_meta") and cache_position.is_meta) or (
            attention_mask is not None and hasattr(attention_mask, "is_meta") and attention_mask.is_meta
        )

        if is_meta_tensor:
            return None

        try:
            if cache_position is None:
                cache_condition = False
            elif hasattr(cache_position, "has_previous_state"):
                cache_condition = cache_position.has_previous_state()
            else:
                cache_condition = cache_position[0] > 0 if cache_position.numel() > 0 else False
            mask_condition = (
                torch.all(attention_mask == 1).item()
                if attention_mask is not None and attention_mask.numel() > 0
                else False
            )

            if cache_condition or mask_condition:
                linear_attn_mask = None
        except RuntimeError:
            return None

        return linear_attn_mask

    def _patched_linear_attn_forward(
        self,
        hidden_states: torch.Tensor,
        cache_params=None,
        cache_position=None,
        attention_mask=None,
        **kwargs,
    ):
        local_num_k_heads, local_num_v_heads = _get_local_linear_attn_heads(self)
        batch_size, seq_len, _ = hidden_states.shape
        has_previous_state = _has_previous_state(
            cache_params,
            cache_position,
            self.layer_idx,
        )
        use_recurrent = has_previous_state and _is_recurrent_decode_batch(seq_len, cache_position)
        flatten_decode_batch = use_recurrent and seq_len != 1

        if attention_mask is not None:
            hidden_states = torch.ops.tensor_cast.linear_attn_apply_padding_mask(hidden_states, attention_mask)

        mixed_qkv = self.in_proj_qkv(hidden_states)
        z = self.in_proj_z(hidden_states).reshape(batch_size, seq_len, local_num_v_heads, self.head_v_dim)
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        core_batch_size = batch_size
        core_seq_len = seq_len
        if flatten_decode_batch:
            core_batch_size = batch_size * seq_len
            core_seq_len = 1
            mixed_qkv = mixed_qkv.reshape(core_batch_size, core_seq_len, -1)
            z = z.reshape(core_batch_size, core_seq_len, local_num_v_heads, self.head_v_dim)
            b = b.reshape(core_batch_size, core_seq_len, -1)
            a = a.reshape(core_batch_size, core_seq_len, -1)

        conv_op = (
            torch.ops.tensor_cast.linear_attn_causal_conv_update
            if use_recurrent
            else torch.ops.tensor_cast.linear_attn_causal_conv
        )
        mixed_qkv = mixed_qkv.transpose(1, 2)
        mixed_qkv = conv_op(
            mixed_qkv,
            self.conv_kernel_size,
        )
        mixed_qkv = mixed_qkv.transpose(1, 2)

        key_dim = local_num_k_heads * self.head_k_dim
        value_dim = local_num_v_heads * self.head_v_dim
        query, key, value = torch.split(mixed_qkv, [key_dim, key_dim, value_dim], dim=-1)
        query = query.reshape(core_batch_size, core_seq_len, local_num_k_heads, self.head_k_dim)
        key = key.reshape(core_batch_size, core_seq_len, local_num_k_heads, self.head_k_dim)
        value = value.reshape(core_batch_size, core_seq_len, local_num_v_heads, self.head_v_dim)

        query, key, beta, g = torch.ops.tensor_cast.linear_attn_fused_gdn_gating(
            query,
            key,
            b,
            a,
            self.A_log,
            self.dt_bias,
            local_num_v_heads,
        )

        if use_recurrent:
            core_attn_out = torch.ops.tensor_cast.linear_attn_recurrent_gated_delta_rule(
                query, key, value, beta, g, 1, 1
            )
        else:
            chunk_size = kwargs.get("chunk_size", 64)
            state_read_passes = 1 if has_previous_state else 0
            state_write_passes = 1
            core_attn_out = torch.ops.tensor_cast.linear_attn_chunk_gated_delta_rule(
                query,
                key,
                value,
                beta,
                g,
                chunk_size,
                state_read_passes,
                state_write_passes,
            )

        norm_weight = getattr(self.norm, "weight", None)
        core_attn_out = torch.ops.tensor_cast.linear_attn_gated_rmsnorm(
            core_attn_out,
            z,
            norm_weight,
            self.layer_norm_epsilon,
        )
        if flatten_decode_batch:
            core_attn_out = core_attn_out.reshape(batch_size, seq_len, local_num_v_heads, self.head_v_dim)
        core_attn_out = core_attn_out.reshape(batch_size * seq_len, -1)
        output = self.out_proj(core_attn_out)
        return output.reshape(batch_size, seq_len, -1)

    def _patched_decoder_layer_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        cache_position = kwargs.get("cache_position")

        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(
                cache_params=past_key_values,
                cache_position=cache_position,
                attention_mask=attention_mask,
                hidden_states=hidden_states,
            )
        elif self.layer_type == "full_attention":
            hidden_states, _ = self.self_attn(
                position_ids=position_ids,
                past_key_values=past_key_values,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                hidden_states=hidden_states,
                **kwargs,
            )
        else:
            raise ValueError(f"Unknown layer_type: {self.layer_type}")

        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if isinstance(hidden_states, tuple):
            hidden_states, _ = hidden_states
        return residual + hidden_states

    target_classes = [Qwen3_5Model, Qwen3_5MoeModel]
    original_methods = {cls: cls.get_placeholder_mask for cls in target_classes}

    def _patched_get_placeholder_mask(self, *args, **kwargs):
        # In meta/simulation runs, HF strict image_features-vs-token-count checks
        # can fail even when we only need shape simulation; skip this check.
        kwargs["image_features"] = None
        return original_methods[type(self)](self, *args, **kwargs)

    # Model-specific monkey patches required for TensorCast simulation:
    # - linear attention op routing and mask behavior for compile stability
    # - VL placeholder-mask relaxation for meta execution
    Qwen3_5TextModel._update_linear_attn_mask = _patched_update_linear_attn_mask
    Qwen3_5MoeTextModel._update_linear_attn_mask = _patched_update_linear_attn_mask
    Qwen3_5GatedDeltaNet.forward = _patched_linear_attn_forward
    Qwen3_5MoeGatedDeltaNet.forward = _patched_linear_attn_forward
    Qwen3_5DecoderLayer.forward = _patched_decoder_layer_forward
    Qwen3_5MoeDecoderLayer.forward = _patched_decoder_layer_forward
    for cls in target_classes:
        cls.get_placeholder_mask = _patched_get_placeholder_mask

    _set_qwen3_5_linear_attn_tp_size(model)


register_model_profile(
    ModelProfile(
        model_type="qwen3_5_moe",
        moe_module_name="Qwen3_5MoeSparseMoeBlock",
        moe_gate_returns_raw_logits=False,
        moe_num_experts_key=["text_config", "num_experts"],
        moe_field_names_override=MoEFieldNames(
            shared_experts="shared_expert",
            shared_experts_gate="shared_expert_gate",
        ),
        mtp_block_module_name="Qwen3_5MoeDecoderLayer",
        model_family="qwen3_5",
        patch_method=patch_method_for_qwen3_5,
        **QWEN3_5_VISUAL_CONFIG,
    )
)
