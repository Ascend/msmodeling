# _*_coding:utf-8_*_
"""
model_builder
"""

import contextlib
import logging

import torch
from transformers.initialization import no_init_weights

from .. import config
from ..compilation import get_backend
from ..core.config_resolver import ConfigResolver
from ..core.user_config import UserInputConfig
from ..layers.mtp import MtpWrapper
from ..pipeline_parallel import (
    apply_stage_boundaries,
    build_pipeline_plan,
    build_stage_model_config,
    PipelineModel,
    PipelineStageModel,
)
from ..transformers.custom_model_registry import (
    get_model_profile,
    get_visual,
    get_vl_language_model,
)
from ..transformers.model import CausalLmWrapper, TransformerModel
from ..transformers.utils import init_on_device_without_buffers

logger = logging.getLogger(__name__)


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


def _supports_pipeline_text_path(model_config) -> bool:
    """Return whether a VL profile exposes a language-only layer path for PP."""
    hf_config = getattr(model_config, "hf_config", None)
    if getattr(hf_config, "vision_config", None) is None:
        return True
    profile = get_model_profile(getattr(hf_config, "model_type", ""))
    return bool(
        profile
        and getattr(profile, "language_module_path", None)
        and getattr(profile, "language_layers_path_str", None)
    )


def _narrow_pipeline_vl_stage_to_language_model(
    stage_model: TransformerModel,
) -> TransformerModel:
    """Run pipeline stages through the VL profile's language module only."""
    if not getattr(stage_model, "is_vl_model", False) or not hasattr(stage_model, "unwrap"):
        return stage_model
    existing_inner = getattr(stage_model, "_inner", None)
    existing_mtp_wrapper = existing_inner if isinstance(existing_inner, MtpWrapper) else None
    existing_lm_head = getattr(existing_inner, "lm_head", None)
    try:
        language_model = get_vl_language_model(stage_model)
    except AttributeError:
        return stage_model
    if language_model is None:
        return stage_model

    dtype_context = (
        stage_model.set_default_dtype() if hasattr(stage_model, "set_default_dtype") else contextlib.nullcontext()
    )
    with dtype_context, init_on_device_without_buffers("meta"), no_init_weights():
        language_wrapper = CausalLmWrapper(stage_model.text_config, language_model)
    if isinstance(existing_lm_head, torch.nn.Module):
        language_wrapper.lm_head = existing_lm_head
    if existing_mtp_wrapper is not None:
        existing_mtp_wrapper._inner = language_wrapper
        stage_model._inner = existing_mtp_wrapper
    else:
        stage_model._inner = language_wrapper
    stage_model.is_vl_model = False
    return stage_model


def _build_pipeline_model(user_input: UserInputConfig, model_config) -> PipelineModel:
    if not _supports_pipeline_text_path(model_config):
        raise ValueError(
            "Pipeline parallel model construction only supports text-only decoder models "
            "or VL profiles with an explicit language module path for now."
        )
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
        stage_model = _narrow_pipeline_vl_stage_to_language_model(stage_model)
        apply_stage_boundaries(stage_model, stage_spec)
        if user_input.do_compile:
            import torch

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

        config.compilation.fusion_patterns.enable_dispatch_ffn_combine = bool(user_input.enable_dispatch_ffn_combine)
        model = torch.compile(
            model,
            backend=get_backend(device_name=user_input.device),
            dynamic=user_input.dynamic_shapes,
            fullgraph=use_full_graph,
        )
    return model
