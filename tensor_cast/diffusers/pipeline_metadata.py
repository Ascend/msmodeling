from ..model_config import DiffusersPipelineMetadata
from .model_resolver import DiffusersPipelineManifest

_DIFFUSERS_HUNYUANVIDEO15_VISION_NUM_SEMANTIC_TOKENS = 729


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
