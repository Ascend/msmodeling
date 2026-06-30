# _*_coding:utf-8_*_
"""
model_builder
"""

import logging

from .. import config
from ..compilation import get_backend
from ..core.config_resolver import ConfigResolver
from ..core.user_config import UserInputConfig
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


def build_model(user_input: UserInputConfig = None) -> TransformerModel:
    """
    Build a transformer model based on the given args

    :param user_input: user_input
    :return: The loaded (and possibly compiled) Transformer model.
    """
    config_resolver = ConfigResolver(user_input=user_input)
    model_config = config_resolver.resolve()
    model = TransformerModel(user_input.model_id, model_config)
    use_full_graph = not user_input.allow_graph_break
    if user_input.do_compile and getattr(model, "is_vl_model", False):
        use_full_graph = _prepare_vl_compile(model)
    if user_input.do_compile:
        import torch

        use_glm5_overrides = _requires_glm5_compile_overrides(user_input)
        config.compilation.fusion_patterns.enable_matmul_allreduce = not use_glm5_overrides
        config.compilation.multistream.enable = bool(user_input.enable_multistream)
        config.compilation.fusion_patterns.enable_dispatch_ffn_combine = bool(user_input.enable_dispatch_ffn_combine)
        model = torch.compile(
            model,
            backend=get_backend(device_name=user_input.device),
            dynamic=False,
            fullgraph=use_full_graph,
        )
    return model
