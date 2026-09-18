import logging

import torch

from ...model_config import MoEFieldNames
from ...utils import exact_division
from ..custom_model_registry import ModelProfile, register_model_profile
from ..utils import has_previous_linear_attention_state, is_recurrent_linear_attention_decode_batch

logger = logging.getLogger(__name__)


def patch_method_for_qwen3_next(model):
    from transformers.models.qwen3_next import modeling_qwen3_next

    tp_size = model.parallel_group_manager.tp_group.world_size
    if tp_size > 1:
        for module in model._inner.modules():
            if not isinstance(module, modeling_qwen3_next.Qwen3NextGatedDeltaNet):
                continue
            if module.num_k_heads % tp_size != 0 or module.num_v_heads % tp_size != 0:
                raise ValueError(
                    "Qwen3-Next linear attention requires tp_size to divide both "
                    f"head counts, but got num_k_heads={module.num_k_heads}, "
                    f"num_v_heads={module.num_v_heads}, and tp_size={tp_size}."
                )
            if module.head_k_dim != module.head_v_dim:
                raise ValueError(
                    "Qwen3-Next fused linear-attention TP sharding requires equal "
                    f"key/value head dimensions, but got {module.head_k_dim} and {module.head_v_dim}."
                )
            module.tensor_cast_tp_size = tp_size

    def _patched_update_linear_attn_mask(self, attention_mask, cache_position):
        """
        Core Conflict:
        During PyTorch's symbolic tracing (e.g., torch.compile or torch.fx),
        input tensors (like cache_position) are Meta Tensors.
        Meta Tensors contain only shape and dtype metadata, no actual data values.

        Error Trigger:
        The original code if cache_position[0] > 0:
        attempts to use the result of a tensor comparison directly in a Python if control flow statement.
        Python's if requires a concrete boolean value (True or False).
        To obtain this, PyTorch implicitly calls .item() to extract the scalar value from the tensor.
        Since Meta Tensors hold no data, calling .item() fails, raising Tensor.item() cannot be called on meta tensors.
        Conclusion:
        In dynamic graph compilation modes,
        you cannot use specific tensor values to dictate Python code execution branches.
        """
        # Currently, this is the only feasible modification. However, the drawback is that
        # it still passes an attention mask to the linear attention mechanism during decoding, where it is unnecessary.
        # Check if it's a meta tensor

        is_meta = (hasattr(cache_position, "is_meta") and cache_position.is_meta) or (
            attention_mask is not None and hasattr(attention_mask, "is_meta") and attention_mask.is_meta
        )
        if is_meta:
            return attention_mask

        try:
            if cache_position is None:
                cache_condition = False
            else:
                cache_condition = cache_position[0] > 0 if cache_position.numel() > 0 else False
            mask_condition = (
                torch.all(attention_mask == 1).item()
                if attention_mask is not None and attention_mask.numel() > 0
                else False
            )

            if cache_condition or mask_condition:
                return None
        except RuntimeError:
            logger.warning(
                "_update_linear_attn_mask fallback due to runtime error",
                exc_info=True,
            )
        return attention_mask

    def _patched_linear_attn_forward(
        self,
        hidden_states,
        cache_params=None,
        cache_position=None,
        attention_mask=None,
        **kwargs,
    ):
        tp_size = getattr(self, "tensor_cast_tp_size", 1)
        if tp_size <= 0:
            raise ValueError(f"Qwen3-Next linear attention TP size must be positive, but got {tp_size}.")
        local_num_k_heads = exact_division(self.num_k_heads, tp_size)
        local_num_v_heads = exact_division(self.num_v_heads, tp_size)
        batch_size, seq_len, _ = hidden_states.shape

        has_previous_state = has_previous_linear_attention_state(cache_params, cache_position, self.layer_idx)
        use_recurrent = has_previous_state and is_recurrent_linear_attention_decode_batch(seq_len, cache_position)
        flatten_decode_batch = use_recurrent and seq_len != 1

        if attention_mask is not None:
            hidden_states = torch.ops.tensor_cast.linear_attn_apply_padding_mask(hidden_states, attention_mask)

        projected_states_qkvz = self.in_proj_qkvz(hidden_states)
        projected_states_ba = self.in_proj_ba(hidden_states)
        query, key, value, z = torch.split(
            projected_states_qkvz,
            [
                local_num_k_heads * self.head_k_dim,
                local_num_k_heads * self.head_k_dim,
                local_num_v_heads * self.head_v_dim,
                local_num_v_heads * self.head_v_dim,
            ],
            dim=-1,
        )
        b, a = torch.split(projected_states_ba, [local_num_v_heads, local_num_v_heads], dim=-1)

        query = query.reshape(batch_size, seq_len, local_num_k_heads, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, local_num_k_heads, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, local_num_v_heads, self.head_v_dim)
        z = z.reshape(batch_size, seq_len, local_num_v_heads, self.head_v_dim)

        core_batch_size = batch_size
        core_seq_len = seq_len
        if flatten_decode_batch:
            core_batch_size = batch_size * seq_len
            core_seq_len = 1
            query = query.reshape(core_batch_size, core_seq_len, local_num_k_heads, self.head_k_dim)
            key = key.reshape(core_batch_size, core_seq_len, local_num_k_heads, self.head_k_dim)
            value = value.reshape(core_batch_size, core_seq_len, local_num_v_heads, self.head_v_dim)
            z = z.reshape(core_batch_size, core_seq_len, local_num_v_heads, self.head_v_dim)
            b = b.reshape(core_batch_size, core_seq_len, local_num_v_heads)
            a = a.reshape(core_batch_size, core_seq_len, local_num_v_heads)

        mixed_qkv = torch.cat(
            (
                query.reshape(core_batch_size, core_seq_len, -1),
                key.reshape(core_batch_size, core_seq_len, -1),
                value.reshape(core_batch_size, core_seq_len, -1),
            ),
            dim=-1,
        ).transpose(1, 2)
        conv_op = (
            torch.ops.tensor_cast.linear_attn_causal_conv_update
            if use_recurrent
            else torch.ops.tensor_cast.linear_attn_causal_conv
        )
        mixed_qkv = conv_op(mixed_qkv, self.conv_kernel_size).transpose(1, 2)
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
            core_attn_out = torch.ops.tensor_cast.linear_attn_chunk_gated_delta_rule(
                query,
                key,
                value,
                beta,
                g,
                chunk_size,
                1 if has_previous_state else 0,
                1,
            )
        core_attn_out = torch.ops.tensor_cast.linear_attn_gated_rmsnorm(
            core_attn_out,
            z,
            getattr(self.norm, "weight", None),
            self.layer_norm_epsilon,
        )
        if flatten_decode_batch:
            core_attn_out = core_attn_out.reshape(batch_size, seq_len, local_num_v_heads, self.head_v_dim)
        core_attn_out = core_attn_out.reshape(batch_size * seq_len, -1)
        return self.out_proj(core_attn_out).reshape(batch_size, seq_len, -1)

    modeling_qwen3_next.Qwen3NextModel._update_linear_attn_mask = _patched_update_linear_attn_mask
    modeling_qwen3_next.Qwen3NextGatedDeltaNet.forward = _patched_linear_attn_forward


register_model_profile(
    ModelProfile(
        model_type="qwen3_next",
        moe_module_name="Qwen3NextSparseMoeBlock",
        moe_gate_returns_raw_logits=False,
        moe_num_experts_key=["text_config", "num_experts"],
        moe_field_names_override=MoEFieldNames(
            shared_experts="shared_expert",
            shared_experts_gate="shared_expert_gate",
        ),
        model_family="qwen3_next",
        patch_method=patch_method_for_qwen3_next,
    )
)
