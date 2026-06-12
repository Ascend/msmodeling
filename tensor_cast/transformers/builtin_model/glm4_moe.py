from ..custom_model_registry import ModelProfile, register_model_profile

register_model_profile(
    ModelProfile(
        model_type="glm4_moe",
        moe_module_name="Glm4MoeMoE",
        moe_gate_returns_raw_logits=True,
        moe_num_experts_key="n_routed_experts",
        mtp_block_module_name="Glm4MoeDecoderLayer",
    )
)
