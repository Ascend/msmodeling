import logging

import torch

from ...model_config import MoEFieldNames
from ..custom_model_registry import ModelProfile, register_model_profile

logger = logging.getLogger(__name__)


def _get_local_linear_attn_heads(self):
    from transformers.models.qwen3_next import modeling_qwen3_next

    if isinstance(self, modeling_qwen3_next.Qwen3NextGatedDeltaNet):
        return self.num_k_heads, self.num_v_heads
    return 0, 0


def patch_method_for_qwen3_next(_model):
    from transformers.models.qwen3_next import modeling_qwen3_next

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
            attention_mask is not None
            and hasattr(attention_mask, "is_meta")
            and attention_mask.is_meta
        )
        if is_meta:
            return attention_mask

        try:
            cache_condition = (
                cache_position[0] > 0 if cache_position.numel() > 0 else False
            )
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
    ):
        # Route Qwen3Next GatedDeltaNet through tensor_cast.linear_attention so
        # TensorCast can model mixed full/linear attention explicitly.
        del cache_params
        local_num_k_heads, local_num_v_heads = _get_local_linear_attn_heads(self)

        return torch.ops.tensor_cast.linear_attention(
            hidden_states,
            attention_mask,
            cache_position,
            local_num_k_heads,
            local_num_v_heads,
            self.head_k_dim,
            self.head_v_dim,
            self.conv_kernel_size,
        )

    modeling_qwen3_next.Qwen3NextModel._update_linear_attn_mask = (
        _patched_update_linear_attn_mask
    )
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
        patch_method=patch_method_for_qwen3_next,
    )
)
