from ..custom_model_registry import ModelProfile, register_model_profile


register_model_profile(
    ModelProfile(
        model_type="deepseek_v3",
        moe_module_name="DeepseekV3MoE",
        moe_num_experts_key="n_routed_experts",
        moe_gate_returns_raw_logits=True,
        mtp_block_module_name="DeepseekV3DecoderLayer",
        mla_module_name="DeepseekV3Attention",
    )
)
