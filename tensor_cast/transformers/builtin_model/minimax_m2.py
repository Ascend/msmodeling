import torch

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

from ..custom_model_registry import (
    ModelProfile,
    register_custom_model,
    register_model_profile,
)
from ..model import TransformerModel


class MoELayerWithBias(MoELayer):
    def forward(self, hidden_states: torch.Tensor):
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
    return model


register_model_profile(
    ModelProfile(
        model_type="minimax_m2",
        moe_module_name="MiniMaxM2SparseMoeBlock",
        moe_gate_returns_raw_logits=False,
        moe_num_experts_key="num_local_experts",
    )
)
