from tensor_cast.layers import COLWISE_LINEAR, ROWWISE_LINEAR
from tensor_cast.transformers.custom_model_registry import (
    ModelProfile,
    register_model_profile,
)

INTERNVL_VISUAL_CONFIG = {
    "visual_module_path": "visual",
    "language_module_path": "language_model",
    "visual_layers_module_path": "vision_tower.encoder.layer",
    "visual_layers_path_str": "vision_tower.encoder.layer",
    "language_layers_path_str": "language_model.layers",
    "visual_merger_linear_mapping": {},
    "visual_mlp_linear_mapping": {
        "vision_tower.encoder.layer.*.mlp.fc1": COLWISE_LINEAR,
        "vision_tower.encoder.layer.*.mlp.fc2": ROWWISE_LINEAR,
    },
}

register_model_profile(
    ModelProfile(
        model_type="internvl",
        model_family="internvl",
        **INTERNVL_VISUAL_CONFIG,
    )
)
