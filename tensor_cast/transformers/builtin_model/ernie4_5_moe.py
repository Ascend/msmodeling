from ..custom_model_registry import ModelProfile, register_model_profile

register_model_profile(
    ModelProfile(
        model_type="ernie4_5_moe",
        moe_module_name="Ernie4_5_MoeSparseMoeBlock",
        moe_gate_returns_raw_logits=False,
        moe_num_experts_key="moe_num_experts",
    )
)
