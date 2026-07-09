# _*_coding:utf-8_*_
"""
model_builder
"""

import logging

from .. import config
from ..compilation import get_backend
from ..core.config_resolver import ConfigResolver
from ..core.user_config import UserInputConfig
from ..pipeline_parallel import (
    apply_stage_boundaries,
    build_pipeline_plan,
    build_stage_model_config,
    PipelineModel,
    PipelineStageModel,
)
from ..transformers.custom_model_registry import get_visual
from ..transformers.model import TransformerModel

logger = logging.getLogger(__name__)


def _requires_glm5_compile_overrides(user_input: UserInputConfig) -> bool:
    model_name = user_input.model_id.rstrip("/").split("/")[-1]
    return user_input.do_compile and model_name in {"GLM-5", "GLM-5.1"}


def _prepare_vl_compile(model: TransformerModel) -> bool:
    # We intentionally skip compiling the visual encoder (ViT-like) by wrapping
    # visual.forward with torch._dynamo.disable and disabling full-graph:
    # 1) The visual path contributes a relatively small portion of end-to-end time (~20%),
    #    so the optimization headroom is limited.
    # 2) Vision blocks have few profitable fusion opportunities; even if fused,
    #    the expected gains are small compared to the language path.
    # 3) The current implementation causes compile errors and requires substantial
    #    adaptation effort (it is largely Python-level and not torch-native).
    # This introduces a deliberate graph break to improve stability with negligible
    # impact on overall performance analysis.
    logger.warning(
        "Skipping compile for visual encoder: wrap visual.forward with torch._dynamo.disable "
        "(small share ~20%, limited fusion benefit, current compile errors; introduces graph break)."
    )
    visual = get_visual(model)
    if visual is not None and hasattr(visual, "forward"):
        import torch._dynamo

        orig_forward = visual.forward

        def _wrapped_forward(*args, **kwargs):
            @torch._dynamo.disable
            def _call(*a, **k):
                return orig_forward(*a, **k)

            return _call(*args, **kwargs)

        visual.forward = _wrapped_forward
    return False


def _build_pipeline_model(user_input: UserInputConfig, model_config) -> PipelineModel:
    if getattr(model_config.hf_config, "vision_config", None) is not None:
        raise ValueError("Pipeline parallel model construction only supports text-only decoder models for now.")
    if model_config.mtp_config is not None:
        raise ValueError("Pipeline parallel model construction does not support MTP models yet.")

    pp_size = model_config.parallel_config.pipeline_parallel_size
    plan = build_pipeline_plan(model_config, pp_size)
    logger.info("Building pipeline model with %d stages", pp_size)
    stages = []
    for stage_index, stage_spec in enumerate(plan.stages, start=1):
        logger.info(
            "Building pipeline stage %d/%d with layers [%d, %d)",
            stage_index,
            pp_size,
            stage_spec.layer_start,
            stage_spec.layer_end,
        )
        stage_model_config = build_stage_model_config(model_config, stage_spec)
        stage_model = TransformerModel(user_input.model_id, stage_model_config)
        apply_stage_boundaries(stage_model, stage_spec)
        if user_input.do_compile:
            import torch

            use_glm5_overrides = _requires_glm5_compile_overrides(user_input)
            config.compilation.fusion_patterns.enable_matmul_allreduce = not use_glm5_overrides
            config.compilation.fusion_patterns.enable_dispatch_ffn_combine = bool(
                user_input.enable_dispatch_ffn_combine
            )
            stage_model = torch.compile(
                stage_model,
                backend=get_backend(device_name=user_input.device),
                dynamic=user_input.dynamic_shapes,
                fullgraph=not user_input.allow_graph_break,
            )
        stages.append(PipelineStageModel(stage_spec=stage_spec, model=stage_model))
    logger.info("Pipeline model construction completed with %d stages", len(stages))
    return PipelineModel(model_config=model_config, plan=plan, stages=stages)


def build_model(
    user_input: UserInputConfig | None = None,
) -> TransformerModel | PipelineModel:
    """
    Build a transformer model based on the given args

    :param user_input: user_input
    :return: The loaded (and possibly compiled) Transformer or Pipeline model.
    """
    config_resolver = ConfigResolver(user_input=user_input)
    model_config = config_resolver.resolve()
    if model_config.parallel_config.pipeline_parallel_size > 1:
        return _build_pipeline_model(user_input, model_config)

    model = TransformerModel(user_input.model_id, model_config)
    use_full_graph = not user_input.allow_graph_break
    if user_input.do_compile and getattr(model, "is_vl_model", False):
        use_full_graph = _prepare_vl_compile(model)
    if user_input.do_compile:
        import torch

        use_glm5_overrides = _requires_glm5_compile_overrides(user_input)
        config.compilation.fusion_patterns.enable_matmul_allreduce = not use_glm5_overrides
        config.compilation.fusion_patterns.enable_dispatch_ffn_combine = bool(user_input.enable_dispatch_ffn_combine)
        model = torch.compile(
            model,
            backend=get_backend(device_name=user_input.device),
            dynamic=user_input.dynamic_shapes,
            fullgraph=use_full_graph,
        )
    return model
