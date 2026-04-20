from ..custom_model_registry import ModelProfile, register_model_profile

register_model_profile(
    ModelProfile(
        model_type="qwen3_moe",
        moe_module_name="Qwen3MoeSparseMoeBlock",
        moe_gate_returns_raw_logits=False,
    )
)
