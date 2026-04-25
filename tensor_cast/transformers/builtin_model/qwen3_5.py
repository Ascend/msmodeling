from ..custom_model_registry import ModelProfile, register_model_profile
from .qwen3_5_moe import patch_method_for_qwen3_5, QWEN3_5_VISUAL_CONFIG

register_model_profile(
    ModelProfile(
        model_type="qwen3_5",
        model_family="qwen3_5",
        patch_method=patch_method_for_qwen3_5,
        **QWEN3_5_VISUAL_CONFIG,
    )
)
