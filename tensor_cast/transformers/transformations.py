import copy
import dataclasses
import fnmatch
import logging
import math
import operator
import typing
from typing import TYPE_CHECKING, Optional, Union

import torch

if TYPE_CHECKING:
    from .model import ModelWrapperBase

from ..layers import (
    COLWISE_LINEAR,
    PARALLEL_EMBEDDING,
    PARALLEL_MODULE_CLS,
    ROWWISE_LINEAR,
)
from ..layers.glm5 import (
    Glm5SparseAttention,
    extend_glm5_indexer_types_for_mtp,
    get_glm5_indexer_flow_flags,
    get_glm5_indexer_types,
    glm5_uses_indexshare,
    resolve_glm5_indexer_source_layer,
)
from ..layers.internal import CopyLayerWrapper, RegionMarkerWrapper
from ..layers.mla import MultiheadLatentAttentionBase, tp_plan_module_path, tp_plan_nested_module_path
from ..layers.moe_layer import MoELayer, ParallelMoELayer
from ..layers.quant_linear import QuantLinearBase
from ..layers.rotary_embedding import CachingRotaryEmb
from ..model_config import config_has_draft_spec
from ..quantize_utils import quantize_linear_modules
from .custom_model_registry import (
    get_language_layers,
    get_model_profile,
    get_visual,
    get_visual_layers,
    get_visual_layers_path,
    get_visual_merger_linear,
    get_visual_mlp_linear,
    get_vl_language_model,
)
from .utils import strip_module_name
from ..adapter.patch_report import PatchReport, attach_patch_report

logger = logging.getLogger(__name__)

_MINIMAX_M3_MODEL_TYPE = "minimax_m3_vl"


def wrap_model(model: "ModelWrapperBase") -> "ModelWrapperBase":
    """
    Normalize the forward interface so that we don't have to adapt to transformers specifics outside:
    1. We already return torch.Tensor or a tuple of tensors when intermediates are needed
    2. We don't need to pass transformers specific args like `use_cache` or `return_dict` etc. outside.
    This makes other wrappers' life simpler.
    """
    from ..diffusers.diffusers_model import DiffusersTransformerModel

    if isinstance(model, DiffusersTransformerModel):
        model._inner.set_attention_backend("tensor_cast")
    else:
        if not model._inner.get_output_embeddings():
            if model.is_vl_model:
                from .model import VLModelWrapper

                model._inner = VLModelWrapper(
                    hf_config=model.hf_config,
                    model=model._inner,
                )
            else:
                from .model import CausalLmWrapper

                model._inner = CausalLmWrapper(
                    hf_config=model.hf_config,
                    model=model._inner,
                )
        else:
            from .model import ModelWrapper

            model._inner = ModelWrapper(model._inner)
    return model


def _resolve_target_decoder_layers(model: "ModelWrapperBase") -> Optional[torch.nn.ModuleList]:
    """Locate target text decoder ModuleList (CausalLM / VL nested layouts)."""
    unwrapped = model.unwrap()
    candidates = []
    if hasattr(unwrapped, "layers"):
        candidates.append(unwrapped.layers)
    nested = getattr(unwrapped, "model", None)
    if nested is not None and hasattr(nested, "layers"):
        candidates.append(nested.layers)
    language_model = get_vl_language_model(model)
    if language_model is not None:
        if hasattr(language_model, "layers"):
            candidates.append(language_model.layers)
        lm_nested = getattr(language_model, "model", None)
        if lm_nested is not None and hasattr(lm_nested, "layers"):
            candidates.append(lm_nested.layers)
    model_type = getattr(getattr(model, "hf_config", None), "model_type", None)
    if model_type:
        try:
            path = get_language_layers(model_type)
            if path:
                candidates.append(operator.attrgetter(path)(unwrapped))
        except (AttributeError, TypeError, ValueError):
            pass
    for layers in candidates:
        if layers is not None and len(layers) > 0:
            return layers
    return None


def maybe_enable_mtp(model: "ModelWrapperBase") -> "ModelWrapperBase":
    if not model.model_config.mtp_config:
        return model

    mtp_config = copy.deepcopy(model.model_config.mtp_config)
    if model.is_vl_model:
        hf_config_source = model.text_config
        if hf_config_source is None:
            raise ValueError("VL model detected but text_config is None; cannot enable MTP")
    else:
        hf_config_source = model.hf_config
    hf_config = copy.deepcopy(hf_config_source)

    if mtp_config.mtp_block_module_name is None:
        layers = _resolve_target_decoder_layers(model)
        if layers is not None:
            mtp_config.mtp_block_module_name = type(layers[-1]).__name__

    if hasattr(hf_config, "layer_types") and isinstance(hf_config.layer_types, list) and hf_config.layer_types:
        hf_config.layer_types.extend([hf_config.layer_types[-1]] * mtp_config.num_mtp_layers)
    if (
        hasattr(hf_config, "mlp_layer_types")
        and isinstance(hf_config.mlp_layer_types, list)
        and hf_config.mlp_layer_types
    ):
        hf_config.mlp_layer_types.extend([hf_config.mlp_layer_types[-1]] * mtp_config.num_mtp_layers)
    if hasattr(hf_config, "indexer_types") and isinstance(hf_config.indexer_types, list) and hf_config.indexer_types:
        if getattr(hf_config, "model_type", None) == "glm_moe_dsa" and glm5_uses_indexshare(hf_config):
            extend_glm5_indexer_types_for_mtp(hf_config.indexer_types, mtp_config.num_mtp_layers)
        else:
            hf_config.indexer_types.extend([hf_config.indexer_types[-1]] * mtp_config.num_mtp_layers)

    orig_dtype = torch.get_default_dtype()
    torch.set_default_dtype(model.model_config.dtype)
    try:
        from tensor_cast.layers.mtp import MtpWrapper

        model._inner = MtpWrapper(mtp_config, hf_config, model._inner)
    finally:
        torch.set_default_dtype(orig_dtype)
    return model


def maybe_enable_dflash(model: "ModelWrapperBase") -> "ModelWrapperBase":
    """Attach unified DflashWrapper (Qwen3 draft + TC attention + KV injection)."""
    if not model.model_config.dflash_config:
        return model
    if model.model_config.mtp_config:
        raise ValueError("Dflash and MTP are mutually exclusive")
    if model.model_config.dspark_config:
        raise ValueError("DSpark and Dflash are mutually exclusive")

    dcfg = copy.deepcopy(model.model_config.dflash_config)
    if model.is_vl_model:
        hf_config_source = model.text_config
        if hf_config_source is None:
            raise ValueError("VL model detected but text_config is None; cannot enable Dflash")
    else:
        hf_config_source = model.hf_config
    hf_config = copy.deepcopy(hf_config_source)

    layers = _resolve_target_decoder_layers(model)
    if layers is None:
        raise ValueError(f"Unable to resolve decoder layers for Dflash from {model}")

    num_hidden_layers = int(getattr(hf_config, "num_hidden_layers", len(layers)))
    target_hidden_size = int(getattr(hf_config, "hidden_size"))
    target_vocab_size = int(getattr(hf_config, "vocab_size"))
    target_max_pos = getattr(hf_config, "max_position_embeddings", None)

    orig_dtype = torch.get_default_dtype()
    torch.set_default_dtype(model.model_config.dtype)
    try:
        from tensor_cast.layers.dflash import build_dflash_draft_and_wrapper

        model._inner = build_dflash_draft_and_wrapper(
            model,
            dcfg,
            hf_config,
            num_target_hidden_layers=num_hidden_layers,
            target_hidden_size=target_hidden_size,
            target_vocab_size=target_vocab_size,
            target_max_position_embeddings=int(target_max_pos) if target_max_pos is not None else None,
            dtype=model.model_config.dtype,
            target_layers=layers,
        )
        model.model_config.dflash_config = dcfg
    finally:
        torch.set_default_dtype(orig_dtype)
    return model


def maybe_enable_dspark(model: "ModelWrapperBase") -> "ModelWrapperBase":
    """Attach DsparkWrapper (DFlash backbone + Markov/Confidence) from dspark.py."""
    if not model.model_config.dspark_config:
        return model
    if model.model_config.mtp_config:
        raise ValueError("DSpark and MTP are mutually exclusive")
    if model.model_config.dflash_config:
        raise ValueError("DSpark and Dflash are mutually exclusive")

    scfg = copy.deepcopy(model.model_config.dspark_config)
    if model.is_vl_model:
        hf_config_source = model.text_config
        if hf_config_source is None:
            raise ValueError("VL model detected but text_config is None; cannot enable DSpark")
    else:
        hf_config_source = model.hf_config
    hf_config = copy.deepcopy(hf_config_source)

    layers = _resolve_target_decoder_layers(model)
    if layers is None:
        raise ValueError(f"Unable to resolve decoder layers for DSpark from {model}")

    num_hidden_layers = int(getattr(hf_config, "num_hidden_layers", len(layers)))
    target_hidden_size = int(getattr(hf_config, "hidden_size"))
    target_vocab_size = int(getattr(hf_config, "vocab_size"))
    target_max_pos = getattr(hf_config, "max_position_embeddings", None)

    orig_dtype = torch.get_default_dtype()
    torch.set_default_dtype(model.model_config.dtype)
    try:
        from tensor_cast.layers.dspark import build_dspark_draft_and_wrapper

        model._inner = build_dspark_draft_and_wrapper(
            model,
            scfg,
            hf_config,
            num_target_hidden_layers=num_hidden_layers,
            target_hidden_size=target_hidden_size,
            target_vocab_size=target_vocab_size,
            target_max_position_embeddings=int(target_max_pos) if target_max_pos is not None else None,
            dtype=model.model_config.dtype,
            target_layers=layers,
        )
        model.model_config.dspark_config = scfg
    finally:
        torch.set_default_dtype(orig_dtype)
    return model


def maybe_reuse_layers(model: "ModelWrapperBase") -> "ModelWrapperBase":
    if not model.model_config.enable_repetition:
        return model

    effective_hf_config = getattr(getattr(model, "_inner", None), "hf_config", getattr(model, "hf_config", None))
    glm5_indexer_types = (
        get_glm5_indexer_types(effective_hf_config)
        if effective_hf_config is not None and glm5_uses_indexshare(effective_hf_config)
        else None
    )

    def get_submodule_structure_key(module: torch.nn.Module) -> str:
        submodule_types = []
        self_attn = getattr(module, "self_attn", None)
        if self_attn is not None:
            layer_idx = getattr(self_attn, "layer_idx", None)
            skip_topk = getattr(self_attn, "skip_topk", False)
            next_skip_topk = getattr(self_attn, "next_skip_topk", False)
            if glm5_indexer_types is not None and isinstance(layer_idx, int):
                skip_topk, next_skip_topk = get_glm5_indexer_flow_flags(glm5_indexer_types, layer_idx)
            if skip_topk or next_skip_topk:
                submodule_types.append(f"glm5_indexer_flow:{layer_idx if layer_idx is not None else id(module)}")
        for name, sub_module in module.named_modules():
            submodule_types.append(name)
            submodule_types.append(".".join([type(sub_module).__module__, type(sub_module).__name__]))
            submodule_types.extend(
                f"buffer:{buffer_name}" for buffer_name, _ in sub_module.named_buffers(recurse=False)
            )
        return ",".join(submodule_types)

    def reuse_modules(modules):
        """Wrap structurally repeated modules with region replay wrappers."""
        seen_keys: dict[str, RegionMarkerWrapper] = {}
        for i, module in enumerate(modules):
            key = get_submodule_structure_key(module)
            if key not in seen_keys:
                modules[i] = RegionMarkerWrapper(region_id=id(module), layer=module)
                seen_keys[key] = modules[i]
            else:
                region_wrapper = seen_keys[key]
                region_wrapper.repeat_count += 1
                modules[i] = CopyLayerWrapper(
                    region_id=region_wrapper.region_id,
                    layer=module,
                    representative=region_wrapper,
                )

    def reuse_layers(layers):
        # We analyze the structure of sub-modules of each layer to detect repetition patterns.
        # For the first layer of the repetition, we wrap it with RegionMarkerWrapper and then
        # wrap the rest layers of the same pattern with CopyLayerWrapper. CopyLayerWrapper is a
        # synthetic module with no children, so later transformations only process representative layers.
        reuse_modules(layers)

    def reuse_glm5_stateless_submodules(layers):
        """Reuse GLM-5.2 MLPs while keeping IndexShare decoder data flow real.

        A complete decoder-layer copy would also need to copy the auxiliary
        ``topk_indices`` output used by shared Indexer layers. The current
        region-copy primitive only replays the first tensor output, so limit
        this pass to MLPs, which have no cross-layer state.
        """
        mlps = []
        for layer in layers:
            mlp = getattr(layer, "mlp", None)
            if mlp is None:
                # Keep the generic behavior for synthetic or unsupported layer
                # containers that do not expose the standard decoder MLP.
                reuse_layers(layers)
                return
            mlps.append(mlp)

        reuse_modules(mlps)
        for layer, mlp in zip(layers, mlps):
            layer.mlp = mlp

    unwrapped = model.unwrap()
    if hasattr(unwrapped, "layers"):
        if glm5_indexer_types is not None:
            reuse_glm5_stateless_submodules(unwrapped.layers)
        else:
            reuse_layers(unwrapped.layers)

    visual_layers = get_visual_layers(model)
    if visual_layers is not None:
        reuse_layers(visual_layers)

    if model.is_vl_model:
        # Some VL models run text-only simulations where the visual module is
        # absent or disabled, but their reusable decoder layers still live
        # under a language path such as language_model.layers.
        import operator

        language_layers_path = get_language_layers(model.hf_config.model_type)
        try:
            language_layers = operator.attrgetter(language_layers_path)(model.unwrap())
            reuse_layers(language_layers)
        except AttributeError:
            logger.debug(
                f"Could not access language layers via path '{language_layers_path}' "
                f"for model type '{model.hf_config.model_type}'. Skipping layer reuse."
            )
    from tensor_cast.layers.mtp import MtpWrapper

    if isinstance(model._inner, MtpWrapper):
        reuse_layers(model._inner.mtp.layers)
    # DFlash/DSpark draft: do NOT wrap with RegionMarker/CopyLayer.
    # Under torch.compile, draft mark_region_begin / copy_region are frequently DCE'd
    # (identity custom ops) while mark_region_end remains — breaking Runtime pairing and
    # silently dropping draft-layer FLOPs. Draft depth is small (typically ≤6); run real
    # layers. Target-side enable_repetition is unchanged.
    return model


def patch_model(model: "ModelWrapperBase"):
    profile = get_model_profile(model.hf_config.model_type)
    if profile and profile.patch_method:
        profile.patch_method(model)


def patch_rotary_emb(model: "ModelWrapperBase") -> "ModelWrapperBase":
    unwrapped = model.unwrap()
    vl_language_model = get_vl_language_model(model)
    if vl_language_model is not None:
        unwrapped = vl_language_model
    if model.model_config.cache_rotary_embedding and hasattr(unwrapped, "rotary_emb"):
        unwrapped.rotary_emb = CachingRotaryEmb(
            unwrapped.rotary_emb,
            act_dtype=model.model_config.dtype,
            max_position_embeddings=model.text_config.max_position_embeddings,
            expand_to_3d_position_ids=vl_language_model is not None,
        )

    # Cache draft-owned rotary for Dflash/DSpark (never share target RoPE).
    inner = getattr(model, "_inner", None)
    if (
        config_has_draft_spec(model.model_config)
        and model.model_config.cache_rotary_embedding
        and inner is not None
        and hasattr(inner, "rotary_emb")
        and getattr(inner, "rotary_emb", None) is not None
    ):
        from tensor_cast.layers.dflash import DflashWrapper

        if isinstance(inner, DflashWrapper):
            inner.rotary_emb = CachingRotaryEmb(
                inner.rotary_emb,
                act_dtype=model.model_config.dtype,
                max_position_embeddings=getattr(
                    inner.draft_hf_config, "max_position_embeddings", model.text_config.max_position_embeddings
                ),
                expand_to_3d_position_ids=False,
            )
            inner.draft.rotary_emb = inner.rotary_emb
    return model


def _validate_gqa_dcp_kv_heads(model: "ModelWrapperBase", dcp_group) -> None:
    """Enforce the GQA-only DCP constraint ``num_key_value_heads >= tp / dcp``.

    DCP re-partitions KV heads across the TP domain, so each dcp rank holds
    ``h_kv * dcp / tp`` KV heads; this must be >= 1, otherwise the rank has no KV
    head to read and the configuration is illegal on vllm-ascend. We raise here so
    such a config errors at model-build time instead of being silently modeled with
    a degenerate single KV head and reported as optimal. No-op when DCP is disabled
    or ``num_key_value_heads`` is unavailable (e.g. MLA-style configs).
    """
    dcp_size = getattr(dcp_group, "world_size", 1) or 1
    if dcp_size <= 1:
        return
    parallel_config = model.model_config.parallel_config
    tp_size = parallel_config.tensor_parallel_size
    text_config = getattr(model, "text_config", None)
    num_kv_heads = getattr(text_config, "num_key_value_heads", None)
    if num_kv_heads is None:
        return
    # tp % dcp == 0 is already guaranteed by ParallelConfig, so tp / dcp is exact.
    min_kv_heads = tp_size // dcp_size
    if num_kv_heads < min_kv_heads:
        raise ValueError(
            f"Illegal GQA + DCP configuration: num_key_value_heads ({num_kv_heads}) must be >= "
            f"tensor_parallel_size / decode_context_parallel_size ({tp_size} / {dcp_size} = "
            f"{min_kv_heads}); otherwise a DCP rank holds no KV head."
        )


def patch_attention(model: "ModelWrapperBase") -> "ModelWrapperBase":
    # Assign a depth_layer_idx to each attention layer in the vision model
    # and append them sequentially to attention_by_layers.
    # This allows:
    # 1) vision attention and text attention to use the same attention_by_layers registry
    # 2) each vision attention layer to have a corresponding index
    # 3) during the subsequent flash_attention_forward invocation,
    #    the corresponding attention instance can be retrieved via depth_layer_idx
    if model.model_config.attention_cls is None:
        return model

    # DCP reuses the TP communication domain; attach the DCP group so the text
    # attention layers can gather Q / shard the KV context on the decode path.
    # Vision attention layers below intentionally keep the default no-op group.
    dcp_group = getattr(getattr(model, "parallel_group_manager", None), "dcp_group", None)

    # GQA-only DCP constraint (rfc_context_parallel_dcp §2.1.3): each dcp rank holds
    # ``h_kv * dcp / tp`` KV heads, which must be >= 1, i.e. ``h_kv >= tp / dcp``.
    # This depends on the concrete model's ``num_key_value_heads`` and so is validated
    # here -- after ModelConfig is loaded and before any GQA attention layer runs --
    # rather than in the model-agnostic ParallelConfig. MLA models use ``patch_mla``
    # and are exempt (latent KV is not partitioned across TP heads).
    _validate_gqa_dcp_kv_heads(model, dcp_group)

    model.attention_by_layers = {}
    for i in range(model.num_hidden_layers):
        attention_layer = model.model_config.attention_cls()
        if dcp_group is not None:
            attention_layer.dcp_group = dcp_group
        model.attention_by_layers[i] = attention_layer

    visual_model = get_visual(model)
    if visual_model is not None:
        pattern = "blocks.*.attn"
        depth_layer_idx = len(model.attention_by_layers)
        for name, module in visual_model.named_modules():
            if fnmatch.fnmatchcase(strip_module_name(name), pattern):
                module._tensor_cast_context = {
                    "attention_by_layers": model.attention_by_layers,
                    "depth_layer_idx": depth_layer_idx,
                }
                model.attention_by_layers[depth_layer_idx] = model.model_config.attention_cls()
                depth_layer_idx += 1
    return model


def _missing_required_fields(module: torch.nn.Module, field_names) -> tuple[str, ...]:
    """Return required configured attributes that are absent from module."""

    def is_optional(annotation):
        if typing.get_origin(annotation) is Union:
            return type(None) in typing.get_args(annotation)
        return False

    if not dataclasses.is_dataclass(field_names):
        if hasattr(field_names, "__dataclass_fields__"):
            fields_obj = field_names
        else:
            return tuple()
    else:
        fields_obj = field_names

    missing = []
    for field in dataclasses.fields(fields_obj):
        field_name = field.name
        target_attr = getattr(fields_obj, field_name, field_name)
        if target_attr is None or is_optional(type(fields_obj).__annotations__.get(field_name)):
            continue
        if not hasattr(module, target_attr):
            missing.append(target_attr)
    return tuple(missing)


def _all_required_fields_exist(module: torch.nn.Module, field_names) -> bool:
    """Helper for MLA/MoE checks."""
    return not _missing_required_fields(module, field_names)


def _candidate_aliases(module: torch.nn.Module, missing_fields: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    fields = set(vars(module).keys())
    fields.update(getattr(module, "_modules", {}).keys())
    fields.update(getattr(module, "_parameters", {}).keys())
    fields.update(getattr(module, "_buffers", {}).keys())
    fields = sorted(fields)
    aliases = {}
    for missing in missing_fields:
        compact_missing = missing.replace("_", "")
        matches = []
        for field in fields:
            compact_field = field.replace("_", "")
            if missing in field or compact_missing in compact_field or compact_field in compact_missing:
                matches.append(field)
        aliases[missing] = tuple(matches)
    return aliases


def _expected_replacements_from_layers(model: "ModelWrapperBase") -> int | None:
    num_layers = getattr(model, "num_hidden_layers", None)
    if num_layers is None:
        return None
    # Draft layers are Qwen3 GQA and are skipped by MLA/MoE patches.
    if config_has_draft_spec(model.model_config):
        num_layers = num_layers - int(model.model_config.draft_num_layers())
    return num_layers


def patch_mla(
    model: "ModelWrapperBase",
    report: PatchReport | None = None,
    strict: bool = False,
) -> "ModelWrapperBase":
    mla_config = model.model_config.mla_config
    if mla_config is None:
        return model

    report = report or PatchReport(
        pass_name="MLA",  # nosec B106
        target_module_name=mla_config.module_name,
        expected_replacements=_expected_replacements_from_layers(model),
    )

    # Pass `parallel_group_manager` only to MLA classes whose __init__ accepts
    # it. V4 (Flash/Pro) needs it to pick up `o_proj_tp_group`; V3/V3.2 don't
    # declare the parameter and should receive the legacy 3-arg call.
    extra_kwargs = {}
    mla_cls = mla_config.mla_cls
    if mla_cls is not None and getattr(mla_cls, "supports_parallel_group_manager", False) is True:
        extra_kwargs["parallel_group_manager"] = model.parallel_group_manager

    named_modules = list(model._inner.named_modules())
    for name, module in named_modules:
        # Dflash/DSpark draft is Qwen3 GQA; never apply target MLA patches under draft.*.
        if config_has_draft_spec(model.model_config) and (name == "draft" or name.startswith("draft.")):
            continue
        if type(module).__name__ == mla_config.module_name:
            report.matched_modules.append(name)
            missing_fields = _missing_required_fields(module, mla_config.field_names)
            if missing_fields:
                report.add_skip(
                    name,
                    type(module).__name__,
                    "missing_required_fields",
                    missing_fields,
                    _candidate_aliases(module, missing_fields),
                )
                continue
            mla_tp_group = model.parallel_group_manager.tp_group
            if mla_config.enable_dsa_cp:
                mla_tp_group = copy.copy(mla_tp_group)
                mla_tp_group.rank_group = [mla_tp_group.rank]
                mla_tp_group.rank_in_group = 0
                mla_tp_group.world_size = 1
            mla = mla_config.mla_cls(
                mla_config,
                module,
                mla_tp_group,
                **extra_kwargs,
            )
            effective_hf_config = getattr(model._inner, "hf_config", model.hf_config)
            if isinstance(mla, Glm5SparseAttention) and glm5_uses_indexshare(effective_hf_config):
                indexer_types = get_glm5_indexer_types(effective_hf_config)
                layer_idx = mla.layer_idx
                mla.indexer_source_layer_idx = resolve_glm5_indexer_source_layer(indexer_types, layer_idx)
                mla.indexer_type = indexer_types[layer_idx]
                mla.skip_topk, mla.next_skip_topk = get_glm5_indexer_flow_flags(indexer_types, layer_idx)
            old_type = type(module).__name__
            model._replace_module(name, mla)
            report.add_replacement(name, old_type, type(mla).__name__)
    attach_patch_report(model, report)
    report.validate(strict=strict)
    return model


def _is_3d_tensor_experts(experts_module, expected_num_experts):
    if experts_module is None:
        return False

    if isinstance(experts_module, torch.nn.ModuleList):
        return False

    if isinstance(experts_module, torch.nn.Module):
        for _, param in experts_module.named_parameters():
            if param.ndim == 3 and param.shape[0] == expected_num_experts:
                return True
    return False


def _patch_moe_expert_helper(model: "ModelWrapperBase", module):
    """Helper for MoE patching."""
    profile = get_model_profile(model.hf_config.model_type)
    if not profile or not profile.custom_expert_module_type:
        return

    experts = module.experts
    expert_num = len(experts) if isinstance(experts, torch.nn.ModuleList) else getattr(experts, "num_experts", 0)
    assert isinstance(expert_num, int) and expert_num > 0

    adapter = profile.custom_expert_module_type
    module.experts = torch.nn.ModuleList(
        [
            adapter(experts, i) if _is_3d_tensor_experts(experts, expert_num) else adapter(experts)
            for i in range(expert_num)
        ]
    )


def patch_moe(
    model: "ModelWrapperBase",
    custom_moe_layer=None,
    report: PatchReport | None = None,
    strict: bool = False,
) -> "ModelWrapperBase":
    # replace the vanilla mixture-of-expert (MOE) module with the fused one
    # so that it can be "meta" and torch.compile traced and easily optimized
    # by the backend.
    #
    # NOTE: Why we have to replace the vanilla moe module with the fused one:
    # 1. MOE is data-dependent and the vanilla MOE module usually uses the
    #    data-dependent ops like torch.nonzero or torch.where to route the
    #    experts. This makes it impossible to trace with the "meta" device and
    #    torch.compile based on which we conduct the analysis and graph optimizations.
    # 2. The vanilla MOE usually uses a naive python-based for-loop to distribute
    #    the tokens to the experts, which is slow.
    # 3. The vanilla MOE is not written in a way that can be easily scaled up/out
    #    with expert-parallelism (EP).
    moe_config = model.model_config.moe_config
    if not moe_config:
        return model

    report = report or PatchReport(
        pass_name="MoE",  # nosec B106
        target_module_name=moe_config.module_name,
        expected_replacements=_expected_replacements_from_layers(model),
    )
    reset_moe_metadata = False
    for name, module in model._inner.named_modules():
        # Dflash/DSpark draft is dense Qwen3; skip MoE patches under draft.*.
        if config_has_draft_spec(model.model_config) and (name == "draft" or name.startswith("draft.")):
            continue
        if type(module).__name__ == moe_config.module_name:
            if not reset_moe_metadata:
                model.top_k = None
                model.num_routing_experts = None
                reset_moe_metadata = True
            report.matched_modules.append(name)
            missing_fields = _missing_required_fields(module, moe_config.field_names)
            if missing_fields:
                report.add_skip(
                    name,
                    type(module).__name__,
                    "missing_required_fields",
                    missing_fields,
                    _candidate_aliases(module, missing_fields),
                )
                continue
            _patch_moe_expert_helper(model, module)
            if custom_moe_layer is not None:
                moe_layer = custom_moe_layer(moe_config, module)
            else:
                moe_layer = MoELayer(moe_config, module)

            expert_num = moe_layer.fused_moe.experts.num_experts
            if model.top_k is None:
                model.top_k = moe_layer.top_k
                model.num_routing_experts = expert_num

            old_type = type(module).__name__
            model._replace_module(name, moe_layer)
            report.add_replacement(name, old_type, type(moe_layer).__name__)
    attach_patch_report(model, report)
    report.validate(strict=strict)
    return model


def _shard_model_visual_by_tp_helper(model: "ModelWrapperBase"):
    """Helper for visual sharding."""
    vision_tp_group = getattr(model.parallel_group_manager, "vision_tp_group", model.parallel_group_manager.tp_group)
    tp_size = vision_tp_group.world_size
    visual_layers_path = get_visual_layers_path(model.hf_config.model_type)
    if tp_size <= 1 or visual_layers_path is None:
        return
    pattern = f"{visual_layers_path}.*.attn"
    for name, module in model._inner.named_modules():
        if fnmatch.fnmatchcase(strip_module_name(name), pattern) and hasattr(module, "qkv"):
            if module.num_heads % tp_size != 0:
                raise ValueError(
                    "Vision attention TP requires vision_tp_size to divide num_heads exactly, "
                    f"but got module={strip_module_name(name)}, num_heads={module.num_heads}, "
                    f"vision_tp_size={tp_size}."
                )
            module.num_heads = module.num_heads // tp_size


def shard_model_by_tp(
    model: "ModelWrapperBase",
    report: PatchReport | None = None,
) -> "ModelWrapperBase":
    """
    Replaces all nn.Linear and nn.Embedding modules with Parallel modules based on the
    parallel configuration stored in self.model_config.
    """

    def get_shard_plan(self):
        tp_group = self.parallel_group_manager.tp_group
        o_proj_tp_group = self.parallel_group_manager.o_proj_tp_group
        mlp_tp_group = self.parallel_group_manager.mlp_tp_group
        lmhead_tp_group = self.parallel_group_manager.lmhead_tp_group
        vision_tp_group = self.parallel_group_manager.vision_tp_group
        moe_tp_group = self.parallel_group_manager.moe_tp_group
        is_pipeline_stage_local = (
            self.model_config.parallel_config.pipeline_parallel_size == 1
            and self.model_config.parallel_config.source_pipeline_parallel_size > 1
        )

        def get_tp_plan():
            # TODO:
            # 1. the name of modules should be configured;
            # 2. we can define a class to represent the data with clearer semantics
            tp_plan = {}

            embedding_parallel = self.model_config.parallel_config.embedding_parallel
            if embedding_parallel:
                params = {
                    "tp_group": tp_group,
                    "shard_mode": embedding_parallel,
                }
                tp_plan.update({"embed_tokens": (PARALLEL_EMBEDDING, params)})

            params = {
                "tp_group": tp_group,
                "global_tp_group": tp_group,
            }
            config_info = self.hf_config if not self.is_vl_model else self.text_config
            language_layers = get_language_layers(self.hf_config.model_type)
            layer_prefixes = [f"{language_layers}"]
            if self.model_config.mtp_config is not None:
                layer_prefixes.append("mtp.layers.*.mtp_block")
            # Dflash/DSpark draft uses a separate Qwen3 GQA plan (never MLA prefixes).
            if self.model_config.mla_config:
                params.update({"head_num": config_info.num_attention_heads})
                mla_cls = self.model_config.mla_config.mla_cls
                enable_dsa_cp = self.model_config.mla_config.enable_dsa_cp
                for prefix in layer_prefixes:
                    q_b_kv_b_params = dict(params)
                    if enable_dsa_cp:
                        q_b_kv_b_params["disable_tp"] = True
                    tp_plan.update(
                        {
                            tp_plan_module_path(prefix, "self_attn.q_proj"): (COLWISE_LINEAR, params),
                            tp_plan_module_path(prefix, "self_attn.q_b_proj"): (COLWISE_LINEAR, q_b_kv_b_params),
                            tp_plan_module_path(prefix, "self_attn.kv_b_proj"): (COLWISE_LINEAR, q_b_kv_b_params),
                        }
                    )
                    tp_plan.update(mla_cls.build_tp_plan_extras(prefix, params, config_info))
            else:
                params.update({"head_num": config_info.num_attention_heads})
                tp_plan.update({f"{language_layers}.*.q_proj": (COLWISE_LINEAR, params)})
                if self.hf_config.model_type == _MINIMAX_M3_MODEL_TYPE and self.model_config.mtp_config is not None:
                    tp_plan.update(
                        {
                            tp_plan_nested_module_path("mtp.layers.*.mtp_block", "q_proj"): (
                                COLWISE_LINEAR,
                                params,
                            ),
                        }
                    )
                params = params.copy()
                params.update(
                    {
                        "head_num": config_info.num_key_value_heads,
                        "is_replicable": True,
                    }
                )
                tp_plan.update(
                    {
                        f"{language_layers}.*.k_proj": (
                            COLWISE_LINEAR,
                            params,
                        ),
                        f"{language_layers}.*.v_proj": (
                            COLWISE_LINEAR,
                            params,
                        ),
                    }
                )
                if self.hf_config.model_type == _MINIMAX_M3_MODEL_TYPE and self.model_config.mtp_config is not None:
                    tp_plan.update(
                        {
                            tp_plan_nested_module_path("mtp.layers.*.mtp_block", "k_proj"): (
                                COLWISE_LINEAR,
                                params,
                            ),
                            tp_plan_nested_module_path("mtp.layers.*.mtp_block", "v_proj"): (
                                COLWISE_LINEAR,
                                params,
                            ),
                        }
                    )

            params = {
                "tp_group": o_proj_tp_group,
                "global_tp_group": tp_group,
                "head_num": config_info.num_attention_heads,
            }
            mla_cls = self.model_config.mla_config.mla_cls if self.model_config.mla_config else None
            if self.model_config.mla_config is not None and self.model_config.mla_config.enable_dsa_cp:
                params["disable_tp"] = True
            for prefix in layer_prefixes:
                tp_plan.update({tp_plan_nested_module_path(prefix, "o_proj"): (ROWWISE_LINEAR, params)})
                if mla_cls is not None:
                    tp_plan.update(mla_cls.build_o_proj_tp_plan_extras(prefix, params, config_info))

            model_profile = get_model_profile(self.hf_config.model_type)
            if model_profile is not None and model_profile.model_family == "qwen3_5":
                linear_num_key_heads = getattr(config_info, "linear_num_key_heads", None)
                linear_num_value_heads = getattr(config_info, "linear_num_value_heads", None)
                linear_key_head_dim = getattr(config_info, "linear_key_head_dim", None)
                linear_value_head_dim = getattr(config_info, "linear_value_head_dim", None)
                if None in (
                    linear_num_key_heads,
                    linear_num_value_heads,
                    linear_key_head_dim,
                    linear_value_head_dim,
                ):
                    raise ValueError("Qwen3.5 linear attention TP plan requires linear attention config fields.")
                if linear_key_head_dim != linear_value_head_dim:
                    raise ValueError(
                        "Qwen3.5 linear attention TP plan requires linear_key_head_dim to equal "
                        f"linear_value_head_dim, but got {linear_key_head_dim} and {linear_value_head_dim}."
                    )
                if linear_num_key_heads % tp_group.world_size != 0:
                    raise ValueError(
                        "Qwen3.5 linear attention TP plan requires tp_size to divide "
                        f"linear_num_key_heads, but got {linear_num_key_heads} and {tp_group.world_size}."
                    )
                if linear_num_value_heads % tp_group.world_size != 0:
                    raise ValueError(
                        "Qwen3.5 linear attention TP plan requires tp_size to divide "
                        f"linear_num_value_heads, but got {linear_num_value_heads} and {tp_group.world_size}."
                    )

                linear_attn_col_params = {
                    "tp_group": tp_group,
                    "global_tp_group": tp_group,
                }
                qkv_head_num = 2 * linear_num_key_heads + linear_num_value_heads
                for prefix in layer_prefixes:
                    tp_plan.update(
                        {
                            tp_plan_module_path(prefix, "linear_attn.in_proj_qkv"): (
                                COLWISE_LINEAR,
                                {**linear_attn_col_params, "head_num": qkv_head_num},
                            ),
                            tp_plan_module_path(prefix, "linear_attn.in_proj_z"): (
                                COLWISE_LINEAR,
                                {**linear_attn_col_params, "head_num": linear_num_value_heads},
                            ),
                            tp_plan_module_path(prefix, "linear_attn.in_proj_b"): (
                                COLWISE_LINEAR,
                                {**linear_attn_col_params, "head_num": linear_num_value_heads},
                            ),
                            tp_plan_module_path(prefix, "linear_attn.in_proj_a"): (
                                COLWISE_LINEAR,
                                {**linear_attn_col_params, "head_num": linear_num_value_heads},
                            ),
                            tp_plan_module_path(prefix, "linear_attn.out_proj"): (
                                ROWWISE_LINEAR,
                                {
                                    "tp_group": o_proj_tp_group,
                                    "global_tp_group": tp_group,
                                    "head_num": linear_num_value_heads,
                                    "reduce_output": True,
                                },
                            ),
                        }
                    )

            params = {
                "tp_group": mlp_tp_group,
                "global_tp_group": tp_group,
            }
            for prefix in layer_prefixes:
                tp_plan.update(
                    {
                        tp_plan_module_path(prefix, "mlp.gate_proj"): (COLWISE_LINEAR, params),
                        tp_plan_module_path(prefix, "mlp.up_proj"): (COLWISE_LINEAR, params),
                        tp_plan_module_path(prefix, "mlp.down_proj"): (ROWWISE_LINEAR, params),
                    }
                )
            visual_layers_path = get_visual_layers_path(self.hf_config.model_type)
            if visual_layers_path is not None and vision_tp_group.world_size > 1:
                params = {
                    "tp_group": vision_tp_group,
                    "global_tp_group": vision_tp_group,
                }
                tp_plan.update(
                    {
                        f"{visual_layers_path}.*.attn.qkv": (COLWISE_LINEAR, params),
                        f"{visual_layers_path}.*.attn.proj": (ROWWISE_LINEAR, params),
                    }
                )
                visual_merger_linear = get_visual_merger_linear(self.hf_config.model_type)
                for key, parallel_type in visual_merger_linear.items():
                    tp_plan[key] = (parallel_type, params)

                params = {
                    "tp_group": vision_tp_group,
                    "global_tp_group": vision_tp_group,
                }
                visual_mlp_linear = get_visual_mlp_linear(self.hf_config.model_type)
                for key, parallel_type in visual_mlp_linear.items():
                    tp_plan[key] = (parallel_type, params)
            if not self.model_config.parallel_config.has_ep():
                params = {
                    "tp_group": moe_tp_group,
                    "global_tp_group": moe_tp_group,
                }
                for prefix in layer_prefixes:
                    tp_plan.update(
                        {
                            f"{prefix}.*.experts.*.gate_proj": (COLWISE_LINEAR, params),
                            f"{prefix}.*.experts.*.up_proj": (COLWISE_LINEAR, params),
                            f"{prefix}.*.experts.*.down_proj": (ROWWISE_LINEAR, params),
                        }
                    )
            else:
                params = {
                    "tp_group": moe_tp_group,
                    "global_tp_group": tp_group,
                }
                if is_pipeline_stage_local:
                    params["gather_slice_data"] = False
                for prefix in layer_prefixes:
                    tp_plan.update(
                        {
                            f"{prefix}.*.experts.*.gate_proj": (COLWISE_LINEAR, params),
                            f"{prefix}.*.experts.*.up_proj": (COLWISE_LINEAR, params),
                            f"{prefix}.*.experts.*.down_proj": (ROWWISE_LINEAR, params),
                        }
                    )
                    if (
                        self.model_config.moe_config is not None
                        and self.model_config.moe_config.enable_shared_expert_tp
                    ):
                        shared_expert_params = {
                            "tp_group": mlp_tp_group,
                            "global_tp_group": mlp_tp_group,
                        }
                        shared_expert_down_proj_params = {
                            "tp_group": mlp_tp_group,
                            "global_tp_group": mlp_tp_group,
                            "reduce_output": False,
                        }
                        tp_plan.update(
                            {
                                f"{prefix}.*.shared_expert*.gate_proj": (
                                    COLWISE_LINEAR,
                                    shared_expert_params,
                                ),
                                f"{prefix}.*.shared_expert*.up_proj": (
                                    COLWISE_LINEAR,
                                    shared_expert_params,
                                ),
                                f"{prefix}.*.shared_expert*.down_proj": (
                                    ROWWISE_LINEAR,
                                    shared_expert_down_proj_params,
                                ),
                            }
                        )
                    else:
                        tp_plan.update(
                            {
                                f"{prefix}.*.shared_expert.*.gate_proj": (
                                    COLWISE_LINEAR,
                                    params,
                                ),
                                f"{prefix}.*.shared_expert.*.up_proj": (
                                    COLWISE_LINEAR,
                                    params,
                                ),
                                f"{prefix}.*.shared_expert.*.down_proj": (
                                    ROWWISE_LINEAR,
                                    params,
                                ),
                            }
                        )

            params = {
                "tp_group": lmhead_tp_group,
                "global_tp_group": tp_group,
                "gather_output": True,
            }
            tp_plan.update({"lm_head": (COLWISE_LINEAR, params)})
            if self.model_config.mtp_config is not None:
                tp_plan.update({"mtp.lm_head": (COLWISE_LINEAR, params)})

            # Dflash/DSpark draft: Qwen3 GQA shard; shared lm_head/embed follow target plan only.
            # DSparkWrapper subclasses DflashWrapper, so isinstance covers both.
            draft_enabled = config_has_draft_spec(self.model_config)
            if draft_enabled:
                from tensor_cast.layers.dflash import DflashWrapper

                draft_hf = self._inner.draft_hf_config if isinstance(self._inner, DflashWrapper) else None
                if draft_hf is not None:
                    draft_prefix = "draft.layers.*.dflash_block"
                    q_params = {
                        "tp_group": tp_group,
                        "global_tp_group": tp_group,
                        "head_num": draft_hf.num_attention_heads,
                    }
                    kv_params = {
                        "tp_group": tp_group,
                        "global_tp_group": tp_group,
                        "head_num": draft_hf.num_key_value_heads,
                        "is_replicable": True,
                    }
                    o_params = {
                        "tp_group": o_proj_tp_group,
                        "global_tp_group": tp_group,
                        "head_num": draft_hf.num_attention_heads,
                    }
                    mlp_params = {"tp_group": mlp_tp_group, "global_tp_group": tp_group}
                    tp_plan.update(
                        {
                            # Head-major fused context KV; same head_num as k/v_proj.
                            "draft.context_kv_proj": (COLWISE_LINEAR, kv_params),
                            f"{draft_prefix}.self_attn.q_proj": (COLWISE_LINEAR, q_params),
                            f"{draft_prefix}.self_attn.k_proj": (COLWISE_LINEAR, kv_params),
                            f"{draft_prefix}.self_attn.v_proj": (COLWISE_LINEAR, kv_params),
                            f"{draft_prefix}.self_attn.o_proj": (ROWWISE_LINEAR, o_params),
                            f"{draft_prefix}.mlp.gate_proj": (COLWISE_LINEAR, mlp_params),
                            f"{draft_prefix}.mlp.up_proj": (COLWISE_LINEAR, mlp_params),
                            f"{draft_prefix}.mlp.down_proj": (ROWWISE_LINEAR, mlp_params),
                        }
                    )
                    # DSpark: markov_bias must use the same ColumnParallel policy as lm_head
                    # (same tp_group / gather_output). Do not give it a divergent vocab layout.
                    if getattr(getattr(self._inner, "draft", None), "markov_head", None) is not None:
                        tp_plan.update(
                            {
                                "draft.markov_head.markov_bias": (COLWISE_LINEAR, params),
                            }
                        )
            return tp_plan

        return {"tp_plan": get_tp_plan()}

    shard_plan = get_shard_plan(model)
    tp_plan = shard_plan["tp_plan"]

    modules = {}
    module_stripped_to_names = {}
    for name, module in model._inner.named_modules():
        if isinstance(module, (torch.nn.Embedding, torch.nn.Linear, QuantLinearBase)):
            modules[name] = module
            module_stripped_to_names[strip_module_name(name)] = name

    report = report or PatchReport(pass_name="Shard", target_module_name="tp_plan")  # nosec B106
    for pattern, tp_config in tp_plan.items():
        matches = fnmatch.filter(module_stripped_to_names.keys(), pattern)
        if not matches:
            report.unmatched_patterns.append(pattern)
        for stripped_name in matches:
            name = module_stripped_to_names[stripped_name]
            module = modules[name]
            parallel_module = PARALLEL_MODULE_CLS[tp_config[0]](module, **tp_config[1])
            model._replace_module(name, parallel_module)
            report.add_replacement(name, type(module).__name__, type(parallel_module).__name__, {"pattern": pattern})

    _shard_model_visual_by_tp_helper(model)
    attach_patch_report(model, report)
    return model


def _find_dflash_wrapper(model: "ModelWrapperBase"):
    """Return nested ``DflashWrapper`` / ``DsparkWrapper`` if present."""
    from tensor_cast.layers.dflash import DflashWrapper

    node = getattr(model, "_inner", model)
    seen: set[int] = set()
    while node is not None and id(node) not in seen:
        seen.add(id(node))
        if isinstance(node, DflashWrapper):
            return node
        node = getattr(node, "_inner", None)
    return None


def _resync_dflash_shared_vocab(model: "ModelWrapperBase") -> None:
    """Rebind draft shared embed/lm_head after TP replaces target modules.

    ``draft.set_shared`` runs before ``shard_model``. Replacing ``embed_tokens`` /
    ``lm_head`` with ParallelEmbedding / ColumnParallelLinear leaves draft aliases
    on the raw modules: draft ``lm_head`` then emits local ``V/TP`` logits (no
    gather) while ``markov_bias`` may still be full ``V``, breaking
    ``logits + bias`` under DSpark.
    """
    from tensor_cast.layers.dflash import resolve_target_embed_and_lm_head
    from tensor_cast.layers.utils import ModelWrapperBase

    wrapper = _find_dflash_wrapper(model)
    if wrapper is None or not getattr(wrapper.draft, "_shared_vocab", False):
        return

    embed = None
    lm_head = None
    if isinstance(model, ModelWrapperBase):
        try:
            embed, lm_head = resolve_target_embed_and_lm_head(model)
        except ValueError:
            embed, lm_head = None, None
    if embed is None or lm_head is None:
        unwrapped = model.unwrap() if callable(getattr(model, "unwrap", None)) else None
        if unwrapped is None:
            _unify_dspark_markov_tp_with_lm_head(model)
            return
        embed = getattr(unwrapped, "embed_tokens", None)
        if embed is None and hasattr(unwrapped, "get_input_embeddings"):
            embed = unwrapped.get_input_embeddings()
        lm_head = getattr(unwrapped, "lm_head", None)
        if lm_head is None and hasattr(unwrapped, "get_output_embeddings"):
            lm_head = unwrapped.get_output_embeddings()
    if embed is not None and lm_head is not None:
        wrapper.draft.set_shared(embed, lm_head)
    # After rebind (or if rebind skipped), force Markov vocab TP == lm_head TP.
    _unify_dspark_markov_tp_with_lm_head(model)


def _unify_dspark_markov_tp_with_lm_head(model: "ModelWrapperBase") -> None:
    """Align DSpark ``markov_bias`` ColumnParallel policy with draft ``lm_head``.

    Main/DFlash ``lm_head`` is ColumnParallel (``gather_output=True`` → full V).
    If draft still aliases the sharded inner Linear (local ``V/TP``), Markov must
    also emit local logits — not full-V bias — so ``logits + bias`` is valid.
    """
    from tensor_cast.layers.parallel_linear import ColumnParallelLinear

    wrapper = _find_dflash_wrapper(model)
    if wrapper is None:
        return
    draft = getattr(wrapper, "draft", None)
    markov = getattr(draft, "markov_head", None) if draft is not None else None
    if markov is None:
        return

    lm_head = draft.lm_head
    markov_bias = markov.markov_bias

    if isinstance(lm_head, ColumnParallelLinear):
        gather = bool(lm_head.gather_output)
        if isinstance(markov_bias, ColumnParallelLinear):
            markov_bias.gather_output = gather
            return
        if isinstance(markov_bias, (torch.nn.Linear, QuantLinearBase)):
            markov.markov_bias = ColumnParallelLinear(
                markov_bias,
                tp_group=lm_head.tp_group,
                global_tp_group=lm_head.global_tp_group,
                gather_output=gather,
            )
        return

    # draft.lm_head is not the parallel wrapper (local V/TP weights). Keep Markov local.
    if isinstance(markov_bias, ColumnParallelLinear):
        markov_bias.gather_output = False
        return
    if not isinstance(markov_bias, (torch.nn.Linear, QuantLinearBase)):
        return
    local_v = int(getattr(lm_head, "out_features", 0) or 0)
    full_v = int(getattr(markov_bias, "out_features", 0) or 0)
    if local_v <= 0 or full_v <= local_v or full_v % local_v != 0:
        return
    pgm = getattr(model, "parallel_group_manager", None)
    tp_group = getattr(pgm, "tp_group", None) if pgm is not None else None
    lmhead_tp = getattr(pgm, "lmhead_tp_group", None) if pgm is not None else None
    shard_group = lmhead_tp if lmhead_tp is not None else tp_group
    if shard_group is None or int(shard_group.world_size) <= 1:
        return
    if int(shard_group.world_size) != full_v // local_v:
        # Prefer lmhead TP group size when it matches V/(V/TP).
        if tp_group is not None and int(tp_group.world_size) == full_v // local_v:
            shard_group = tp_group
        else:
            return
    markov.markov_bias = ColumnParallelLinear(
        markov_bias,
        tp_group=shard_group,
        global_tp_group=tp_group if tp_group is not None else shard_group,
        gather_output=False,
    )


def shard_model_by_ep(model: "ModelWrapperBase") -> "ModelWrapperBase":
    moe_config = model.model_config.moe_config
    if not moe_config or not getattr(model, "top_k", None) or not getattr(model, "num_routing_experts", None):
        return model

    ep_group = model.parallel_group_manager.ep_group
    model.num_external_shared_experts = 0
    model.num_redundant_experts = 0
    if not model.model_config.parallel_config.has_ep():
        assert not moe_config.enable_redundant_experts and not moe_config.enable_external_shared_experts
    else:
        if moe_config.enable_external_shared_experts:
            assert ep_group.world_size >= 2
            if model.top_k + 1 > ep_group.world_size:
                model.num_external_shared_experts = 1
            else:
                model.num_external_shared_experts = math.ceil(ep_group.world_size / (model.top_k + 1))

            num_routing_experts_device = ep_group.world_size - model.num_external_shared_experts
            model.num_redundant_experts = (
                num_routing_experts_device - model.num_routing_experts % num_routing_experts_device
            )
            if not moe_config.enable_redundant_experts and model.num_redundant_experts == num_routing_experts_device:
                model.num_redundant_experts = 0

            if not moe_config.host_external_shared_experts:
                if model.model_config.parallel_config.rank == -1:
                    model.parallel_group_manager.set_rank(model.num_external_shared_experts)
                else:
                    raise ValueError(
                        "If you want to check the performance of the device with external shared experts, "
                        f"set the rank to -1 or {model.num_external_shared_experts}."
                    )
        else:
            if moe_config.enable_redundant_experts:
                model.num_redundant_experts = ep_group.world_size

    dp_group = model.parallel_group_manager.dp_group
    tp_group = model.parallel_group_manager.tp_group
    moe_tp_group = model.parallel_group_manager.moe_tp_group
    mlp_tp_group = model.parallel_group_manager.mlp_tp_group
    routed_expert_global_tp_group = tp_group if model.model_config.parallel_config.has_ep() else moe_tp_group
    for name, module in model._inner.named_modules():
        if isinstance(module, MoELayer):
            model._replace_module(
                name,
                ParallelMoELayer(
                    module,
                    dp_group,
                    routed_expert_global_tp_group,
                    mlp_tp_group,
                    ep_group,
                    model.num_external_shared_experts,
                    model.num_redundant_experts,
                ),
            )
    return model


def shard_model(model: "ModelWrapperBase") -> "ModelWrapperBase":
    shard_model_by_ep(model)
    shard_model_by_tp(model)
    # shard_model_by_tp already resyncs; keep a second call cheap/no-op if missing.
    _resync_dflash_shared_vocab(model)
    return model


def _exclude_unquantized_dsa_linears(model_config) -> None:
    modules_to_not_convert = model_config.quant_config.modules_to_not_convert
    if modules_to_not_convert is None:
        modules_to_not_convert = []
        model_config.quant_config.modules_to_not_convert = modules_to_not_convert

    for pattern in (
        "*.kv_b_proj",
        "*indexer*.wk",
        "*indexer*.weights_proj",
    ):
        if pattern not in modules_to_not_convert:
            modules_to_not_convert.append(pattern)


def _ensure_draft_excluded_from_linear_quant(model: "ModelWrapperBase") -> None:
    """RFC: draft-owned Linear must not share target ``--quantize-linear-action``.

    ``create_quant_config`` patterns like ``*.layers.*`` / ``*.mlp.*`` otherwise match
    ``draft.layers.*.dflash_block.mlp.*`` and quantize draft FFN/Attn projections.
    """
    if not config_has_draft_spec(model.model_config):
        return
    quant_config = getattr(model.model_config, "quant_config", None)
    if quant_config is None:
        return
    exclude = list(quant_config.modules_to_not_convert or [])
    if "draft" not in exclude:
        exclude.append("draft")
    quant_config.modules_to_not_convert = exclude


def _draft_attention_layer_indices(model: "ModelWrapperBase") -> set[int]:
    """Attention registry indices owned by DFlash/DSpark draft (skip attention quant)."""
    inner = getattr(model, "_inner", None)
    draft = getattr(inner, "draft", None)
    if draft is None:
        return set()
    idxs = getattr(draft, "_draft_attn_layer_indices", None)
    if idxs:
        return {int(i) for i in idxs}
    return set()


def quantize_linear(
    model: "ModelWrapperBase",
    report: PatchReport | None = None,
) -> "ModelWrapperBase":
    """
    Replaces all nn.Linear modules with QuantLinear modules based on the
    quantization configuration stored in self.model_config.
    """
    from ..diffusers.diffusers_model import DiffusersTransformerModel

    if isinstance(model, DiffusersTransformerModel):
        if not model.model_config.quant_linear_cls:
            return model
        root = (
            model._inner.transformer_blocks
            if hasattr(model._inner, "transformer_blocks")
            else model._inner.blocks
            if hasattr(model._inner, "blocks")
            else None
        )
        before = {}
        if root is not None:
            before = {
                name: type(module).__name__
                for name, module in root.named_modules()
                if isinstance(module, torch.nn.Linear)
            }
        quantize_linear_modules(
            root,
            model.model_config.quant_linear_cls,
            model.model_config.quant_config,
            default_config_name="default_dit",
            strip_module_fn=None,
        )
        after_root = root
    else:
        if not model.model_config.quant_linear_cls:
            return model
        mla_config = model.model_config.mla_config
        if mla_config is not None and mla_config.enable_dsa_cp:
            _exclude_unquantized_dsa_linears(model.model_config)
        _ensure_draft_excluded_from_linear_quant(model)
        before = {
            name: type(module).__name__
            for name, module in model._inner.named_modules()
            if isinstance(module, torch.nn.Linear)
        }
        quantize_linear_modules(
            model._inner,
            model.model_config.quant_linear_cls,
            model.model_config.quant_config,
            default_config_name=None,
            strip_module_fn=lambda n: n.replace("_inner.", "") if "_inner." in n else n,
        )
        after_root = model._inner

    if report is not None and after_root is not None:
        for name, module in after_root.named_modules():
            if name in before and isinstance(module, QuantLinearBase):
                report.add_replacement(name, before[name], type(module).__name__)
    return model


def quantize_attention(
    model: "ModelWrapperBase",
    report: PatchReport | None = None,
) -> "ModelWrapperBase":
    if not hasattr(model.model_config, "quant_config"):
        return model

    attention_configs = model.model_config.quant_config.attention_configs
    default_attention_config = attention_configs.get(-1)
    draft_attn_idxs = _draft_attention_layer_indices(model)

    if model.model_config.mla_config:
        for name, module in model._inner.named_modules():
            # Draft is Qwen3 GQA — never apply target MLA attention quant under draft.*.
            if name == "draft" or name.startswith("draft."):
                continue
            if isinstance(module, MultiheadLatentAttentionBase):
                if hasattr(module, "layer_idx") and module.layer_idx in attention_configs:
                    module.quant_config = attention_configs[module.layer_idx]
                else:
                    module.quant_config = default_attention_config
                if module.quant_config is not None:
                    module.quantize_params()
                    if report is not None:
                        report.add_replacement(
                            name,
                            type(module).__name__,
                            type(module).__name__,
                            {"attention_quantized": True},
                        )

    if hasattr(model, "attention_by_layers"):
        for i in range(model.num_hidden_layers):
            if i in draft_attn_idxs:
                # Keep draft attention unquantized (does not share target quant strategy).
                continue
            if i not in model.attention_by_layers:
                continue
            model.attention_by_layers[i].quant_config = attention_configs.get(i, default_attention_config)
            if report is not None and model.attention_by_layers[i].quant_config is not None:
                report.add_replacement(
                    f"attention_by_layers.{i}",
                    type(model.attention_by_layers[i]).__name__,
                    type(model.attention_by_layers[i]).__name__,
                    {"attention_quantized": True},
                )
    return model


def quantize_model(
    model: "ModelWrapperBase",
    report: PatchReport | None = None,
) -> "ModelWrapperBase":
    from ..diffusers.diffusers_model import DiffusersTransformerModel

    report = report or PatchReport(pass_name="Quant", target_module_name="quantizable modules")  # nosec B106
    if isinstance(model, DiffusersTransformerModel):
        # TODO quantization on cuda: github NVIDIA/Model-Optimizer/tree/main/examples/diffusers
        # TODO whether linears outside blocks should be quant?
        quantize_linear(model, report=report)
    else:
        quantize_linear(model, report=report)
        quantize_attention(model, report=report)
    attach_patch_report(model, report)
    return model
