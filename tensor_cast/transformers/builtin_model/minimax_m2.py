import torch
from typing import Optional

from tensor_cast.transformers.transformations import (
    maybe_enable_mtp,
    maybe_reuse_layers,
    patch_attention,
    patch_mla,
    patch_moe,
    patch_rotary_emb,
    quantize_model,
    shard_model,
    wrap_model,
)
from ...layers.moe_layer import MoELayer
from ...layers.parallel_linear import replace_with_sharded_tensor

from ..custom_model_registry import (
    ModelProfile,
    register_custom_model,
    register_model_profile,
)
from ..model import TransformerModel


def shard_qk_norm(model: TransformerModel) -> TransformerModel:
    """
    Shard q_norm and k_norm weights for MiniMax-M2.5 model with tensor parallelism.

    MiniMax-M2.5 uses QK normalization where:
    - q_proj output: num_attention_heads * head_dim (e.g., 48 * 128 = 6144)
    - q_norm weight: num_attention_heads * head_dim

    When TP is applied, q_proj is sharded column-wise, so q_norm must also be sharded
    to match the reduced dimension.
    """
    parallel_group_manager = getattr(model, "parallel_group_manager", None)
    if parallel_group_manager is None:
        return model

    tp_group = getattr(parallel_group_manager, "tp_group", None)
    if tp_group is None:
        return model

    tp_size = tp_group.world_size
    if tp_size <= 1:
        return model

    if not getattr(model.hf_config, "use_qk_norm", False):
        return model

    unwrapped = model.unwrap()
    if not hasattr(unwrapped, "layers"):
        return model

    tp_rank = tp_group.rank_in_group
    num_attention_heads = model.hf_config.num_attention_heads
    num_key_value_heads = model.hf_config.num_key_value_heads

    for layer in unwrapped.layers:
        # Get the self_attn module, handling wrapper layers
        self_attn = layer
        while hasattr(self_attn, "_inner"):
            self_attn = self_attn._inner
        if hasattr(self_attn, "self_attn"):
            self_attn = self_attn.self_attn

        # Shard q_norm if it exists
        if hasattr(self_attn, "q_norm") and hasattr(self_attn.q_norm, "weight"):
            replace_with_sharded_tensor(
                self_attn.q_norm,
                "weight",
                tp_size,
                tp_rank,
                dim=0,
                head_num=num_attention_heads,
            )

        # Shard k_norm if it exists
        if hasattr(self_attn, "k_norm") and hasattr(self_attn.k_norm, "weight"):
            replace_with_sharded_tensor(
                self_attn.k_norm,
                "weight",
                tp_size,
                tp_rank,
                dim=0,
                head_num=num_key_value_heads,
            )

    return model


class MoELayerWithBias(MoELayer):
    def forward(self, hidden_states: torch.Tensor, input_ids: Optional[torch.Tensor] = None):
        num_experts = getattr(self.gate, "num_experts", None)
        if num_experts is None and hasattr(self.gate, "weight") and len(self.gate.weight.shape) == 2:
            num_experts = self.gate.weight.shape[0]
        if num_experts is None:
            num_experts = getattr(self.moe_config, "num_experts", None) or getattr(self.fused_moe, "num_experts", None)

        e_score_correction_bias = torch.zeros(num_experts, device=hidden_states.device, dtype=hidden_states.dtype)

        if self.moe_config.gate_returns_raw_logits:
            if self.top_k is None:
                raise ValueError("top_k must be specified if gate_returns_raw_logits is True")

            gate_output = self.gate(hidden_states, e_score_correction_bias=e_score_correction_bias)

            if isinstance(gate_output, tuple):
                router_logits = gate_output[0]
            else:
                router_logits = gate_output

            topk_weights, topk_indices = torch.ops.tensor_cast.moe_gating_top_k_softmax(router_logits, self.top_k)

            if self.norm_topk_prob:
                topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
            topk_weights = topk_weights.to(hidden_states.dtype)
        else:
            gate_output = self.gate(hidden_states, e_score_correction_bias=e_score_correction_bias)

            if isinstance(gate_output, tuple) and len(gate_output) >= 2:
                if len(gate_output) == 3:
                    router_logits, topk_weights, topk_indices = gate_output
                else:
                    topk_indices, topk_weights = gate_output[0], gate_output[1]
            elif isinstance(gate_output, torch.Tensor):
                top_k = self.top_k
                topk_weights, topk_indices = torch.topk(gate_output, top_k, dim=-1)
            else:
                raise ValueError(f"Expected gate to return tuple with at least 2 elements, got {type(gate_output)}")

            if hidden_states.dim() > 2:
                target_shape = list(hidden_states.shape[:-1]) + [topk_indices.shape[-1]]
                topk_indices = topk_indices.view(*target_shape)
                topk_weights = topk_weights.view(*target_shape)

        return self.fused_moe(hidden_states, topk_indices, topk_weights)


@register_custom_model("minimax_m2")
def _(model: TransformerModel):
    model = wrap_model(model)
    model = maybe_enable_mtp(model)
    model = maybe_reuse_layers(model)
    model = patch_rotary_emb(model)
    model = patch_attention(model)
    model = patch_mla(model)
    model = patch_moe(model, MoELayerWithBias)
    model = quantize_model(model)
    model = shard_model(model)
    model = shard_qk_norm(model)
    return model


register_model_profile(
    ModelProfile(
        model_type="minimax_m2",
        moe_module_name="MiniMaxM2SparseMoeBlock",
        moe_gate_returns_raw_logits=False,
        moe_num_experts_key="num_local_experts",
    )
)
