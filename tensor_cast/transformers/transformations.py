import copy
import dataclasses
import fnmatch
import math
import typing
from typing import TYPE_CHECKING, Union

import torch

if TYPE_CHECKING:
    from .model import ModelWrapperBase

from ..layers import (
    COLWISE_LINEAR,
    PARALLEL_EMBEDDING,
    PARALLEL_MODULE_CLS,
    ROWWISE_LINEAR,
)
from ..layers.internal import CopyLayerWrapper, RegionMarkerWrapper
from ..layers.mla import MultiheadLatentAttentionBase
from ..layers.moe_layer import MoELayer, ParallelMoELayer
from ..layers.quant_linear import QuantLinearBase
from ..layers.rotary_embedding import CachingRotaryEmb
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


def maybe_enable_mtp(model: "ModelWrapperBase") -> "ModelWrapperBase":
    if not model.model_config.mtp_config:
        return model

    mtp_config = copy.deepcopy(model.model_config.mtp_config)
    unwrapped = model.unwrap()

    if mtp_config.mtp_block_module_name is None and hasattr(unwrapped, "layers"):
        decoder_cls_name = type(unwrapped.layers[-1]).__name__
        mtp_config.mtp_block_module_name = decoder_cls_name
        if hasattr(model.hf_config, "layer_types") and isinstance(
            model.hf_config.layer_types, list
        ):
            model.hf_config.layer_types.extend(
                [model.hf_config.layer_types] * mtp_config.num_mtp_layers
            )

    orig_dtype = torch.get_default_dtype()
    torch.set_default_dtype(model.model_config.dtype)
    from tensor_cast.layers.mtp import MtpWrapper

    model._inner = MtpWrapper(mtp_config, model.hf_config, model._inner)
    torch.set_default_dtype(orig_dtype)
    return model


def maybe_reuse_layers(model: "ModelWrapperBase") -> "ModelWrapperBase":
    if not model.model_config.enable_repetition:
        return model

    def get_submodule_structure_key(module: torch.nn.Module) -> str:
        submodule_types = []
        for name, sub_module in module.named_modules():
            submodule_types.append(name)
            submodule_types.append(
                ".".join([type(sub_module).__module__, type(sub_module).__name__])
            )
        return ",".join(submodule_types)

    def reuse_layers(layers):
        # We analyze the structure of sub-modules of each layer to detect repetition patterns.
        # For the first layer of the repetition, we wrap it with RegionMarkerWrapper and then
        # wrap the rest layers of the same pattern with CopyLayerWrapper. This tells the runtime
        # that we can copy the execution of the first layer to the rest layers.
        seen_keys = {}
        for i, layer in enumerate(layers):
            key = get_submodule_structure_key(layer)
            if key not in seen_keys:
                seen_keys[key] = id(layer)
                layers[i] = RegionMarkerWrapper(region_id=seen_keys[key], layer=layer)
            else:
                layers[i] = CopyLayerWrapper(region_id=seen_keys[key], layer=layer)

    unwrapped = model.unwrap()
    if hasattr(unwrapped, "layers"):
        reuse_layers(unwrapped.layers)

    visual_layers = get_visual_layers(model)
    if visual_layers is not None:
        reuse_layers(visual_layers)
        language_model = get_vl_language_model(model)
        if hasattr(language_model, "layers"):
            reuse_layers(language_model.layers)
    from tensor_cast.layers.mtp import MtpWrapper

    if isinstance(model._inner, MtpWrapper):
        reuse_layers(model._inner.mtp.layers)

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
    return model


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

    model.attention_by_layers = {}
    for i in range(model.num_hidden_layers):
        model.attention_by_layers[i] = model.model_config.attention_cls()

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
                model.attention_by_layers[depth_layer_idx] = (
                    model.model_config.attention_cls()
                )
                depth_layer_idx += 1
    return model


def _all_required_fields_exist(module: torch.nn.Module, field_names) -> bool:
    """Helper for MLA/MoE checks."""

    def is_optional(annotation):
        if typing.get_origin(annotation) is Union:
            return type(None) in typing.get_args(annotation)
        return False

    if not dataclasses.is_dataclass(field_names):
        if hasattr(field_names, "__dataclass_fields__"):
            fields_obj = field_names
        else:
            return False
    else:
        fields_obj = field_names

    for field in dataclasses.fields(fields_obj):
        field_name = field.name
        target_attr = getattr(fields_obj, field_name, field_name)
        if not is_optional(
            type(fields_obj).__annotations__.get(field_name)
        ) and not hasattr(module, target_attr):
            return False
    return True


def patch_mla(model: "ModelWrapperBase") -> "ModelWrapperBase":
    mla_config = model.model_config.mla_config
    if mla_config is None:
        return model

    named_modules = list(model._inner.named_modules())
    for name, module in named_modules:
        if type(module).__name__ == mla_config.module_name:
            if not _all_required_fields_exist(module, mla_config.field_names):
                continue
            mla = mla_config.mla_cls(
                mla_config, module, model.parallel_group_manager.tp_group
            )
            model._replace_module(name, mla)
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
    expert_num = (
        len(experts)
        if isinstance(experts, torch.nn.ModuleList)
        else getattr(experts, "num_experts", 0)
    )
    assert isinstance(expert_num, int) and expert_num > 0

    adapter = profile.custom_expert_module_type
    module.experts = torch.nn.ModuleList(
        [
            adapter(experts, i)
            if _is_3d_tensor_experts(experts, expert_num)
            else adapter(experts)
            for i in range(expert_num)
        ]
    )


def patch_moe(model: "ModelWrapperBase", custom_moe_layer=None) -> "ModelWrapperBase":
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

    model.top_k = None
    model.num_routing_experts = None
    for name, module in model._inner.named_modules():
        if type(module).__name__ == moe_config.module_name:
            if not _all_required_fields_exist(module, moe_config.field_names):
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

            model._replace_module(name, moe_layer)
    return model


def _shard_model_visual_by_tp_helper(model: "ModelWrapperBase"):
    """Helper for visual sharding."""
    tp_size = model.parallel_group_manager.tp_group.world_size
    visual_layers_path = get_visual_layers_path(model.hf_config.model_type)
    if tp_size <= 1 or visual_layers_path is None:
        return
    pattern = f"{visual_layers_path}.*.attn"
    for name, module in model._inner.named_modules():
        if fnmatch.fnmatchcase(strip_module_name(name), pattern) and hasattr(
            module, "qkv"
        ):
            assert module.num_heads % tp_size == 0
            module.num_heads = module.num_heads // tp_size


def shard_model_by_tp(model: "ModelWrapperBase") -> "ModelWrapperBase":
    """
    Replaces all nn.Linear and nn.Embedding modules with Parallel modules based on the
    parallel configuration stored in self.model_config.
    """

    def get_shard_plan(self):
        tp_group = self.parallel_group_manager.tp_group
        o_proj_tp_group = self.parallel_group_manager.o_proj_tp_group
        mlp_tp_group = self.parallel_group_manager.mlp_tp_group
        lmhead_tp_group = self.parallel_group_manager.lmhead_tp_group
        all_rank_group = self.parallel_group_manager.all_rank_group
        moe_tp_group = self.parallel_group_manager.moe_tp_group

        def get_tp_plan():
            # TODO:
            # 1. the name of modules should be configured;
            # 2. we can define a class to represent the data with clearer semantics
            tp_plan = {}

            if self.model_config.parallel_config.embedding_parallel:
                params = {
                    "tp_group": tp_group,
                    "shard_mode": self.model_config.parallel_config.embedding_parallel_mode,
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
            if self.model_config.mla_config:
                params.update({"head_num": config_info.num_attention_heads})
                for prefix in layer_prefixes:
                    tp_plan.update(
                        {
                            f"{prefix}.*.self_attn.q_proj": (COLWISE_LINEAR, params),
                            f"{prefix}.*.self_attn.q_b_proj": (COLWISE_LINEAR, params),
                            f"{prefix}.*.self_attn.kv_b_proj": (COLWISE_LINEAR, params),
                        }
                    )
                if self.model_config.mtp_config is not None:
                    tp_plan.update(
                        {
                            "mtp.layers.*.mtp_block.self_attn.q_proj": (
                                COLWISE_LINEAR,
                                params,
                            ),
                            "mtp.layers.*.mtp_block.self_attn.q_b_proj": (
                                COLWISE_LINEAR,
                                params,
                            ),
                            "mtp.layers.*.mtp_block.self_attn.kv_b_proj": (
                                COLWISE_LINEAR,
                                params,
                            ),
                        }
                    )
            else:
                params.update({"head_num": config_info.num_attention_heads})
                tp_plan.update(
                    {f"{language_layers}.*.q_proj": (COLWISE_LINEAR, params)}
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

            params = {
                "tp_group": o_proj_tp_group,
                "global_tp_group": tp_group,
                "head_num": config_info.num_attention_heads,
            }
            for prefix in layer_prefixes:
                tp_plan.update({f"{prefix}.*.o_proj": (ROWWISE_LINEAR, params)})

            params = {
                "tp_group": mlp_tp_group,
                "global_tp_group": tp_group,
            }
            for prefix in layer_prefixes:
                tp_plan.update(
                    {
                        f"{prefix}.*.mlp.gate_proj": (COLWISE_LINEAR, params),
                        f"{prefix}.*.mlp.up_proj": (COLWISE_LINEAR, params),
                        f"{prefix}.*.mlp.down_proj": (ROWWISE_LINEAR, params),
                    }
                )
            visual_layers_path = get_visual_layers_path(self.hf_config.model_type)
            if visual_layers_path is not None:
                params = {
                    "tp_group": tp_group,
                    "global_tp_group": tp_group,
                }
                tp_plan.update(
                    {
                        f"{visual_layers_path}.*.attn.qkv": (COLWISE_LINEAR, params),
                        f"{visual_layers_path}.*.attn.proj": (ROWWISE_LINEAR, params),
                    }
                )
                visual_merger_linear = get_visual_merger_linear(
                    self.hf_config.model_type
                )
                for key, parallel_type in visual_merger_linear.items():
                    tp_plan[key] = (parallel_type, params)

                params = {
                    "tp_group": mlp_tp_group,
                    "global_tp_group": tp_group,
                }
                visual_mlp_linear = get_visual_mlp_linear(self.hf_config.model_type)
                for key, parallel_type in visual_mlp_linear.items():
                    tp_plan[key] = (parallel_type, params)
            if not self.model_config.parallel_config.has_ep():
                params = {
                    "tp_group": all_rank_group,
                    "global_tp_group": all_rank_group,
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
                for prefix in layer_prefixes:
                    tp_plan.update(
                        {
                            f"{prefix}.*.experts.*.gate_proj": (COLWISE_LINEAR, params),
                            f"{prefix}.*.experts.*.up_proj": (COLWISE_LINEAR, params),
                            f"{prefix}.*.experts.*.down_proj": (ROWWISE_LINEAR, params),
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

    for pattern, tp_config in tp_plan.items():
        matches = fnmatch.filter(module_stripped_to_names.keys(), pattern)
        for stripped_name in matches:
            name = module_stripped_to_names[stripped_name]
            module = modules[name]
            parallel_module = PARALLEL_MODULE_CLS[tp_config[0]](module, **tp_config[1])
            model._replace_module(name, parallel_module)

    _shard_model_visual_by_tp_helper(model)
    return model


def shard_model_by_ep(model: "ModelWrapperBase") -> "ModelWrapperBase":
    moe_config = model.model_config.moe_config
    if (
        not moe_config
        or not getattr(model, "top_k", None)
        or not getattr(model, "num_routing_experts", None)
    ):
        return model

    ep_group = model.parallel_group_manager.ep_group
    model.num_external_shared_experts = 0
    model.num_redundant_experts = 0
    if not model.model_config.parallel_config.has_ep():
        assert (
            not moe_config.enable_redundant_experts
            and not moe_config.enable_external_shared_experts
        )
    else:
        if moe_config.enable_external_shared_experts:
            assert ep_group.world_size >= 2
            if model.top_k + 1 > ep_group.world_size:
                model.num_external_shared_experts = 1
            else:
                model.num_external_shared_experts = math.ceil(
                    ep_group.world_size / (model.top_k + 1)
                )

            num_routing_experts_device = (
                ep_group.world_size - model.num_external_shared_experts
            )
            model.num_redundant_experts = (
                num_routing_experts_device
                - model.num_routing_experts % num_routing_experts_device
            )
            if (
                not moe_config.enable_redundant_experts
                and model.num_redundant_experts == num_routing_experts_device
            ):
                model.num_redundant_experts = 0

            if not moe_config.host_external_shared_experts:
                if model.model_config.parallel_config.rank == -1:
                    model.parallel_group_manager.set_rank(
                        model.num_external_shared_experts
                    )
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
    for name, module in model._inner.named_modules():
        if isinstance(module, MoELayer):
            model._replace_module(
                name,
                ParallelMoELayer(
                    module,
                    dp_group,
                    tp_group,
                    ep_group,
                    model.num_external_shared_experts,
                    model.num_redundant_experts,
                ),
            )
    return model


def shard_model(model: "ModelWrapperBase") -> "ModelWrapperBase":
    shard_model_by_ep(model)
    shard_model_by_tp(model)
    return model


def quantize_linear(model: "ModelWrapperBase") -> "ModelWrapperBase":
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
        quantize_linear_modules(
            root,
            model.model_config.quant_linear_cls,
            model.model_config.quant_config,
            default_config_name="default_dit",
            strip_module_fn=None,
        )
    else:
        if not model.model_config.quant_linear_cls:
            return model
        quantize_linear_modules(
            model._inner,
            model.model_config.quant_linear_cls,
            model.model_config.quant_config,
            default_config_name=None,
            strip_module_fn=lambda n: n.replace("_inner.", "") if "_inner." in n else n,
        )
    return model


def quantize_attention(model: "ModelWrapperBase") -> "ModelWrapperBase":
    if not hasattr(model.model_config, "quant_config"):
        return model

    attention_configs = model.model_config.quant_config.attention_configs
    default_attention_config = attention_configs.get(-1)

    if model.model_config.mla_config:
        for _, module in model._inner.named_modules():
            if isinstance(module, MultiheadLatentAttentionBase):
                if (
                    hasattr(module, "layer_idx")
                    and module.layer_idx in attention_configs
                ):
                    module.quant_config = attention_configs[module.layer_idx]
                else:
                    module.quant_config = default_attention_config
                if module.quant_config is not None:
                    module.quantize_params()

    if hasattr(model, "attention_by_layers"):
        for i in range(model.num_hidden_layers):
            model.attention_by_layers[i].quant_config = attention_configs.get(
                i, default_attention_config
            )
    return model


def quantize_model(model: "ModelWrapperBase") -> "ModelWrapperBase":
    from ..diffusers.diffusers_model import DiffusersTransformerModel

    if isinstance(model, DiffusersTransformerModel):
        # TODO quantization on cuda: github NVIDIA/Model-Optimizer/tree/main/examples/diffusers
        # TODO whether linears outside blocks should be quant?
        pass
    else:
        quantize_linear(model)
        quantize_attention(model)
    return model
