from ...layers.mla import DeepseekSparseAttention
from ..custom_model_registry import ModelProfile, MoeExpertMLP, register_model_profile


register_model_profile(
    ModelProfile(
        model_type="glm_moe_dsa",
        moe_module_name="GlmMoeDsaMoE",
        moe_num_experts_key="n_routed_experts",
        moe_gate_returns_raw_logits=True,
        mla_module_name="GlmMoeDsaAttention",
        mla_module_class_type=DeepseekSparseAttention,
        mtp_block_module_name="GlmMoeDsaDecoderLayer",
        custom_expert_module_type=MoeExpertMLP,
    )
)
