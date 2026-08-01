from ..model_config import DiffusersPipelineMetadata
from .model_resolver import DiffusersPipelineManifest

_DIFFUSERS_HUNYUANVIDEO15_VISION_NUM_SEMANTIC_TOKENS = 729
_TENCENT_HUNYUANVIDEO15_TRANSFORMER_COMPONENT = [
    "hyvideo.models.transformers.hunyuanvideo_1_5_transformer",
    "HunyuanVideo_1_5_DiffusionTransformer",
]
_TENCENT_HUNYUANVIDEO15_TRANSFORMER_CLASS = "HunyuanVideo_1_5_DiffusionTransformer"
_TENCENT_HUNYUANVIDEO15_TRANSFORMER_PROFILE = {
    "concat_condition": True,
    "glyph_byT5_v2": True,
    "guidance_embed": False,
    "heads_num": 16,
    "hidden_size": 2048,
    "in_channels": 32,
    "is_reshape_temporal_channels": False,
    "mlp_act_type": "gelu_tanh",
    "mlp_width_ratio": 4,
    "mm_double_blocks_depth": 54,
    "mm_single_blocks_depth": 0,
    "out_channels": 32,
    "patch_size": [1, 1, 1],
    "qk_norm": True,
    "qk_norm_type": "rms",
    "qkv_bias": True,
    "rope_dim_list": [16, 56, 56],
    "rope_theta": 256,
    "text_pool_type": None,
    "text_projection": "single_refiner",
    "text_states_dim": 3584,
    "text_states_dim_2": None,
    "use_attention_mask": True,
    "use_cond_type_embedding": True,
    "use_meanflow": False,
    "vision_projection": "linear",
    "vision_states_dim": 1152,
}
_TENCENT_HUNYUANVIDEO15_TARGET_SIZE = {"480p": 640, "720p": 960}
_TENCENT_HUNYUANVIDEO15_BYT5_DIM = 1472
_TENCENT_HUNYUANVIDEO15_REFINER_LAYERS = 2


def adapt_tencent_hunyuanvideo15_t2v_config(
    manifest: DiffusersPipelineManifest,
    transformer_config: dict,
) -> tuple[dict, DiffusersPipelineMetadata]:
    """Validate and adapt the native Tencent T2V profile to the built-in Diffusers model."""
    transformer_class = transformer_config.get("_class_name")
    if transformer_class != _TENCENT_HUNYUANVIDEO15_TRANSFORMER_CLASS:
        raise ValueError(
            "Raw Tencent HunyuanVideo1.5 selected transformer must declare "
            f"{_TENCENT_HUNYUANVIDEO15_TRANSFORMER_CLASS!r}; got {transformer_class!r}."
        )
    if manifest.format != "tencent" or manifest.config.get("_class_name") != "HunyuanVideo_1_5_Pipeline":
        raise ValueError(f"Unsupported raw Tencent HunyuanVideo1.5 pipeline contract from {manifest.config_path!r}.")
    if manifest.config.get("transformer") != _TENCENT_HUNYUANVIDEO15_TRANSFORMER_COMPONENT:
        raise ValueError("Raw Tencent HunyuanVideo1.5 pipeline has an unsupported transformer component.")
    if manifest.config.get("vision_num_semantic_tokens") != _DIFFUSERS_HUNYUANVIDEO15_VISION_NUM_SEMANTIC_TOKENS:
        raise ValueError("Raw Tencent HunyuanVideo1.5 pipeline has unsupported vision_num_semantic_tokens.")
    if manifest.config.get("vision_states_dim") != _TENCENT_HUNYUANVIDEO15_TRANSFORMER_PROFILE["vision_states_dim"]:
        raise ValueError("Raw Tencent HunyuanVideo1.5 pipeline has unsupported vision_states_dim.")

    if transformer_config.get("ideal_task") != "t2v":
        raise ValueError("Raw Tencent HunyuanVideo1.5 transformer ideal_task must be 't2v'.")
    resolution = transformer_config.get("ideal_resolution")
    if resolution not in _TENCENT_HUNYUANVIDEO15_TARGET_SIZE:
        accepted = ", ".join(_TENCENT_HUNYUANVIDEO15_TARGET_SIZE)
        raise ValueError(
            f"Raw Tencent HunyuanVideo1.5 transformer ideal_resolution must be one of: {accepted}; got {resolution!r}."
        )
    for field, expected in _TENCENT_HUNYUANVIDEO15_TRANSFORMER_PROFILE.items():
        actual = transformer_config.get(field)
        if actual != expected:
            raise ValueError(
                f"Raw Tencent HunyuanVideo1.5 transformer field {field!r} must be {expected!r}; got {actual!r}."
            )

    hidden_size = transformer_config["hidden_size"]
    heads_num = transformer_config["heads_num"]
    patch_size_t, patch_size, patch_size_w = transformer_config["patch_size"]
    if patch_size != patch_size_w:
        raise ValueError("Raw Tencent HunyuanVideo1.5 transformer requires equal spatial patch dimensions.")

    native_latent_channels = transformer_config["in_channels"]
    condition_latent_channels = native_latent_channels
    mask_channels = 1
    # Diffusers concatenates the noisy latent, condition latent, and one-channel mask.
    diffusers_input_channels = native_latent_channels + condition_latent_channels + mask_channels

    adapted_config = {
        "_class_name": "HunyuanVideo15Transformer3DModel",
        "attention_head_dim": hidden_size // heads_num,
        "image_embed_dim": transformer_config["vision_states_dim"],
        "in_channels": diffusers_input_channels,
        "mlp_ratio": transformer_config["mlp_width_ratio"],
        "num_attention_heads": heads_num,
        "num_layers": transformer_config["mm_double_blocks_depth"],
        "num_refiner_layers": _TENCENT_HUNYUANVIDEO15_REFINER_LAYERS,
        "out_channels": transformer_config["out_channels"],
        "patch_size": patch_size,
        "patch_size_t": patch_size_t,
        "qk_norm": "rms_norm",
        "rope_axes_dim": transformer_config["rope_dim_list"],
        "rope_theta": float(transformer_config["rope_theta"]),
        "target_size": _TENCENT_HUNYUANVIDEO15_TARGET_SIZE[resolution],
        "task_type": "t2v",
        "text_embed_2_dim": _TENCENT_HUNYUANVIDEO15_BYT5_DIM,
        "text_embed_dim": transformer_config["text_states_dim"],
        "use_meanflow": transformer_config["use_meanflow"],
    }
    metadata = DiffusersPipelineMetadata(
        pipeline_class=manifest.config["_class_name"],
        contract_version="tencent-hunyuanvideo15-t2v-v1",
        vision_num_semantic_tokens=manifest.config["vision_num_semantic_tokens"],
        vision_states_dim=manifest.config["vision_states_dim"],
    )
    return adapted_config, metadata


def resolve_hunyuanvideo15_pipeline_metadata(
    manifest: DiffusersPipelineManifest,
    transformer_config: dict,
) -> DiffusersPipelineMetadata:
    """Resolve the validated T2V vision-input contract for HunyuanVideo1.5."""
    if transformer_config.get("_class_name") != "HunyuanVideo15Transformer3DModel":
        raise ValueError("HunyuanVideo1.5 metadata requires HunyuanVideo15Transformer3DModel.")
    if transformer_config.get("task_type") != "t2v":
        raise ValueError("HunyuanVideo1.5 I2V variants are not supported; select an explicit T2V variant.")

    vision_states_dim = transformer_config.get("image_embed_dim")
    if not isinstance(vision_states_dim, int):
        raise ValueError("HunyuanVideo1.5 Transformer variant must define integer image_embed_dim.")

    pipeline_class = manifest.config.get("_class_name")
    if manifest.format != "diffusers" or pipeline_class != "HunyuanVideo15Pipeline":
        raise ValueError(
            f"Unsupported HunyuanVideo1.5 pipeline contract {pipeline_class!r} from {manifest.config_path!r}."
        )

    transformer_component = manifest.config.get("transformer")
    if transformer_component != ["diffusers", "HunyuanVideo15Transformer3DModel"]:
        raise ValueError(
            "HunyuanVideo1.5 pipeline manifest must declare the canonical transformer component "
            "['diffusers', 'HunyuanVideo15Transformer3DModel']."
        )

    # HunyuanVideo15Pipeline defines this canonical T2V contract when its manifest omits it.
    vision_num_semantic_tokens = _DIFFUSERS_HUNYUANVIDEO15_VISION_NUM_SEMANTIC_TOKENS
    variant_vision_num_semantic_tokens = transformer_config.get("vision_num_semantic_tokens")
    if (
        variant_vision_num_semantic_tokens is not None
        and variant_vision_num_semantic_tokens != vision_num_semantic_tokens
    ):
        raise ValueError(
            "HunyuanVideo1.5 local pipeline profile conflicts with selected Transformer variant "
            "vision_num_semantic_tokens."
        )
    contract_version = "diffusers-hunyuanvideo15-v1"

    return DiffusersPipelineMetadata(
        pipeline_class=pipeline_class,
        contract_version=contract_version,
        vision_num_semantic_tokens=vision_num_semantic_tokens,
        vision_states_dim=vision_states_dim,
    )
