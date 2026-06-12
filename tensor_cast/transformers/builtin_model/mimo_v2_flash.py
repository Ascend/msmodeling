import torch
from transformers import AutoConfig, AutoModel

from ..custom_model_registry import ModelProfile, register_model_profile
from .mimo_v2_flash_hf.configuration_mimo_v2_flash import MiMoV2FlashConfig
from .mimo_v2_flash_hf.modeling_mimo_v2_flash import MiMoV2Model

AutoConfig.register("mimo_v2_flash", MiMoV2FlashConfig)
AutoModel.register(MiMoV2FlashConfig, MiMoV2Model)


class MiMoV2MoeExpertMLP(torch.nn.Module):
    def __init__(self, expert_module_list, expert_idx=None):
        super().__init__()
        if expert_idx is not None:
            self.expert = expert_module_list[expert_idx]
        else:
            self.expert = expert_module_list[0]

    def forward(self, hidden_states):
        return self.expert(hidden_states)


register_model_profile(
    ModelProfile(
        model_type="mimo_v2_flash",
        moe_module_name="MiMoV2MoE",
        moe_gate_returns_raw_logits=False,
        moe_num_experts_key="n_routed_experts",
        mtp_block_module_name="MiMoV2DecoderLayer",
        custom_expert_module_type=MiMoV2MoeExpertMLP,
    )
)
