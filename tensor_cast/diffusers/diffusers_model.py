import json
import logging
import os

import numpy as np
import torch
from transformers.initialization import no_init_weights

from ..core.model_source_security import normalize_model_source
from ..layers.attention import AttentionTensorCast
from ..layers.quant_linear import TensorCastQuantLinear
from ..model_config import (
    DiffusersConfig,
    DiffusersTransformerConfig,
    DiffusersVaeConfig,
    RemoteSource,
)
from ..ops.static_mask import static_false_mask
from ..parallel_group import ParallelGroup
from ..transformers.model import ModelWrapperBase
from ..transformers.transformations import quantize_linear, quantize_model, wrap_model
from ..transformers.utils import init_on_device_without_buffers
from . import diffusers_attention  # noqa: F401
from .cache_agent import CacheConfig, CacheState
from .cache_agent.dit_block_cache import DiTBlockCache
from .diffusers_utils import get_diffusers_transformer_module
from .dit_cache_registry import (
    DiTBlockCacheSpec,
    get_dit_block_cache_spec,
    replace_blocks_in_range,
)
from .model_resolver import (
    DiffusersModelSelection,
    resolve_diffusers_model_selection,
    resolve_diffusers_pipeline_manifest,
    try_resolve_diffusers_pipeline_manifest,
)
from .pipeline_metadata import (
    adapt_tencent_hunyuanvideo15_t2v_config,
    resolve_hunyuanvideo15_pipeline_metadata,
)

logger = logging.getLogger(__name__)


def build_diffusers_transformer_model(
    model_id: str,
    parallel_config: None,
    quant_config: None,
    dtype: torch.dtype,
    remote_source: str = RemoteSource.huggingface,
    model_selection: DiffusersModelSelection | None = None,
    resolved_model_path: str | None = None,
):
    if model_selection is None:
        if resolved_model_path is None:
            model_selection = resolve_diffusers_model_selection(model_id, remote_source)
        else:
            is_local_path = os.path.isdir(resolved_model_path)
            model_selection = DiffusersModelSelection(
                repository_root=resolved_model_path,
                variant_path=resolved_model_path,
                variant_id=None,
                source=None,
                is_remote=not is_local_path,
            )
    validate_local_path = not model_selection.is_remote
    model_config = load_config_from_file(
        model_path=model_selection.variant_path,
        parallel_config=parallel_config,
        quant_config=quant_config,
        quant_linear_cls=TensorCastQuantLinear,
        attention_cls=AttentionTensorCast,
        dtype=dtype,
        validate_local_path=validate_local_path,
        model_selection=model_selection,
    )
    model = DiffusersTransformerModel(model_id, model_config.transformer_config)
    return model, model_config


def _normalize_diffusers_transformer_config(config: dict) -> dict:
    config = dict(config)
    if config.get("_class_name") == "WanModel":
        config["_class_name"] = "WanTransformer3DModel"
        config["in_channels"] = config.get("in_channels", config.get("in_dim", 16))
        config["out_channels"] = config.get("out_channels", config.get("out_dim", config["in_channels"]))
        head_count_field = "num_attention_heads" if "num_attention_heads" in config else "num_heads"
        num_attention_heads = config.get("num_attention_heads", config.get("num_heads", 40))
        if head_count_field in config and num_attention_heads <= 0:
            raise ValueError(f"Wan {head_count_field} must be positive, got {num_attention_heads}")
        config["num_attention_heads"] = num_attention_heads
        if "attention_head_dim" not in config and config.get("dim") is not None:
            config["attention_head_dim"] = config["dim"] // num_attention_heads
        config["text_dim"] = config.get("text_dim", 4096)
        config["patch_size"] = config.get("patch_size") or [1, 2, 2]
        config["cross_attn_norm"] = config.get("cross_attn_norm", True)
        config["qk_norm"] = config.get("qk_norm", "rms_norm_across_heads")
        config["rope_max_seq_len"] = config.get("rope_max_seq_len") or 1024
        config.setdefault("image_dim", None)
        config.setdefault("added_kv_proj_dim", None)
        config.setdefault("pos_embed_seq_len", None)
        for legacy_key in (
            "dim",
            "in_dim",
            "model_type",
            "num_heads",
            "out_dim",
            "text_len",
        ):
            config.pop(legacy_key, None)
    return config


def load_config_from_file(
    model_path: str,
    parallel_config: None,
    quant_config: None,
    quant_linear_cls: None,
    attention_cls: None,
    dtype: torch.dtype,
    validate_local_path: bool = True,
    model_selection: DiffusersModelSelection | None = None,
):
    # TODO add seperate parallel_config and quant_config(atten_cls is needed?) for vae and text
    source_info = normalize_model_source(
        model_path,
        RemoteSource.huggingface,
        validate_local=validate_local_path,
    )
    resolved_model_path = source_info.model_id
    if not source_info.is_local_path or not os.path.isdir(resolved_model_path):
        raise ValueError(f"Input args.model_id should be dir, but got {resolved_model_path}")

    config_path_dict: dict[str, str] = {}
    for root, _, files in os.walk(resolved_model_path):
        if "config.json" in files:
            folder_name = os.path.basename(root)
            config_path = os.path.join(root, "config.json")
            config_path = os.path.abspath(config_path)
            config_path_dict[folder_name] = config_path

    def _load_config(config_path: str) -> dict:
        with open(config_path, encoding="utf-8") as f:
            return json.load(f)

    transformer_config_json_path = config_path_dict.get("transformer")
    transformer_config = None
    if transformer_config_json_path is not None:
        transformer_config = _load_config(transformer_config_json_path)
    else:

        def _looks_like_transformer_config(cfg: dict) -> bool:
            class_name = cfg.get("_class_name")
            return isinstance(class_name, str) and ("Transformer" in class_name or class_name == "WanModel")

        transformer_candidates: dict[str, tuple[str, dict]] = {}
        for folder_name, config_path in config_path_dict.items():
            config = _load_config(config_path)
            if _looks_like_transformer_config(config):
                transformer_candidates[folder_name] = (config_path, config)

        if "high_noise_model" in transformer_candidates:
            transformer_config_json_path, transformer_config = transformer_candidates["high_noise_model"]
        elif len(transformer_candidates) == 1:
            transformer_config_json_path, transformer_config = next(iter(transformer_candidates.values()))
        else:
            raise ValueError(
                "No transformer/config.json found in input model path. "
                "Expect a Diffusers-style model directory that contains transformer/config.json."
            )
    transformer_config = _normalize_diffusers_transformer_config(transformer_config)
    pipeline_metadata = None
    transformer_class = transformer_config.get("_class_name")
    if model_selection is None:
        model_selection = DiffusersModelSelection(
            repository_root=resolved_model_path,
            variant_path=resolved_model_path,
            variant_id=None,
            source=None,
            is_remote=False,
        )

    manifest = try_resolve_diffusers_pipeline_manifest(model_selection)
    is_raw_tencent_contract = (manifest is not None and manifest.format == "tencent") or (
        transformer_class == "HunyuanVideo_1_5_DiffusionTransformer"
    )
    if is_raw_tencent_contract:
        if not model_selection.is_remote:
            raise ValueError(
                "Raw Tencent HunyuanVideo-1.5 local Transformer paths are not supported; use an official remote "
                "repository with an explicit transformer/<t2v_variant> selector."
            )
        if manifest is None:
            manifest = resolve_diffusers_pipeline_manifest(model_selection)
        transformer_config, pipeline_metadata = adapt_tencent_hunyuanvideo15_t2v_config(
            manifest,
            transformer_config,
        )
        transformer_class = transformer_config["_class_name"]
    elif transformer_class == "HunyuanVideo15Transformer3DModel":
        if manifest is None:
            manifest = resolve_diffusers_pipeline_manifest(model_selection)
        pipeline_metadata = resolve_hunyuanvideo15_pipeline_metadata(manifest, transformer_config)
        transformer_config.pop("vision_num_semantic_tokens", None)
    elif manifest is not None and manifest.config.get("_class_name") == "HunyuanVideo15Pipeline":
        raise ValueError("HunyuanVideo15Pipeline requires HunyuanVideo15Transformer3DModel.")

    vae_config_json_path = config_path_dict.get("vae")

    model_config = DiffusersConfig()
    model_config.model_path = resolved_model_path
    model_config.pipeline_metadata = pipeline_metadata
    model_config.transformer_config = DiffusersTransformerConfig(
        parallel_config=parallel_config,
        quant_config=quant_config,
        config_json=transformer_config_json_path,
        model_config=transformer_config,
        quant_linear_cls=quant_linear_cls,
        attention_cls=attention_cls,
        dtype=dtype,
    )
    if vae_config_json_path is not None and os.path.isfile(vae_config_json_path):
        with open(vae_config_json_path, encoding="utf-8") as f:
            vae_config = json.load(f)
        model_config.vae_config = DiffusersVaeConfig(
            parallel_config=parallel_config,
            quant_config=quant_config,
            config_json=vae_config_json_path,
            model_config=vae_config,
            dtype=dtype,
        )
    return model_config


class DiffusersTransformerModel(ModelWrapperBase):
    def __init__(
        self,
        model_id: str,
        model_config: DiffusersTransformerConfig,
    ):
        super().__init__(None)
        self.model_id = model_id
        self.model_config = model_config

        hf_config_json = self.model_config.config_json
        self.sp_group = get_sp_group(
            world_size=self.model_config.parallel_config.world_size,
            ulysses_size=self.model_config.parallel_config.ulysses_size,
        )

        if hf_config_json is None:
            raise ValueError("hf_config_json should not be None.")
        hf_config = self.model_config.model_config
        if hf_config is None:
            raise ValueError("transformer model_config should not be None.")
        self._static_boolean_mask_names = ()
        if hf_config.get("_class_name") == "HunyuanVideo15Transformer3DModel" and hf_config.get("task_type") == "t2v":
            self._static_boolean_mask_names = (
                "encoder_attention_mask",
                "encoder_attention_mask_2",
            )
        model_class = get_diffusers_transformer_module(hf_config)

        with init_on_device_without_buffers("meta"), no_init_weights():
            self._inner = model_class.from_config(hf_config).to(model_config.dtype)
        self._inner.eval()
        wrap_model(self)
        quantize_model(self)
        quantize_linear(self)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_images: torch.Tensor | None = None,
        return_dict=False,
        **kwargs: object,
    ):
        if self._static_boolean_mask_names:
            kwargs = dict(kwargs)
            for name in self._static_boolean_mask_names:
                mask = kwargs.get(name)
                if isinstance(mask, torch.Tensor):
                    kwargs[name] = static_false_mask(mask)
        hidden_states = self._inner(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )[0]
        return hidden_states

    def enable_dit_block_cache(
        self,
        cache_config: CacheConfig,
        spec: DiTBlockCacheSpec | None = None,
    ) -> CacheState | None:
        """
        Enable DiT block cache (dit_block_cache).

        Replace blocks in the configured cache range with cache-aware wrappers.
        Step scheduling (update/reuse/bypass) is driven externally by the caller.
        """
        model_config = self.model_config.model_config or {}
        class_name = model_config.get("_class_name")
        explicit_spec = spec is not None
        if spec is None:
            spec = get_dit_block_cache_spec(class_name)
        if spec is None:
            logger.warning("dit_block_cache is not implemented for model %r.", class_name)
            return None
        if explicit_spec and spec.class_name != class_name:
            raise ValueError(
                f"DiT cache spec class identity mismatch: expected {class_name!r}, got {spec.class_name!r}."
            )

        blocks_with_setters = list(spec.get_blocks_with_setters(self._inner))
        if not blocks_with_setters:
            if explicit_spec:
                raise ValueError(f"DiT cache spec {spec.class_name!r} resolved no blocks.")
            return None
        blocks_count = len(blocks_with_setters)

        if explicit_spec and (cache_config.block_start < 0 or cache_config.block_end < 0):
            raise ValueError("DiT cache block range must be non-negative.")
        bounded_block_start = cache_config.block_start
        # end is clamped to the actual block count; start is intentionally not
        # clamped (start < end is validated after clamping for explicit specs).
        bounded_block_end = min(cache_config.block_end, blocks_count)
        if explicit_spec and bounded_block_start >= bounded_block_end:
            raise ValueError(
                f"DiT cache range must be nonempty after clamp: [{bounded_block_start}, {bounded_block_end})."
            )

        cache_state = CacheState()
        replaced = replace_blocks_in_range(
            blocks_with_setters,
            bounded_block_start,
            bounded_block_end,
            lambda block, flat_idx: DiTBlockCache(
                block=block,
                state=cache_state,
                block_index=flat_idx,
                block_start=bounded_block_start,
                block_end=bounded_block_end,
                make_wrapped_forward=spec.make_wrapped_forward,
            ),
        )

        logger.info(
            "Enabled dit_block_cache for %s: replaced %d blocks in range [%d, %d) out of %d.",
            spec.model_type,
            replaced,
            bounded_block_start,
            bounded_block_end,
            blocks_count,
        )
        if explicit_spec and replaced == 0:
            raise ValueError("DiT cache spec replaced no blocks.")
        return cache_state if replaced > 0 else None


def get_sp_group(world_size: int, ulysses_size: int) -> ParallelGroup:
    all_ranks = np.arange(world_size)
    rank = 0
    if ulysses_size > 0:
        rank_groups = all_ranks.reshape(-1, ulysses_size)
    else:
        rank_groups = all_ranks.reshape(1, -1)
    sp_group = ParallelGroup(
        rank=rank,
        rank_groups=[x.tolist() for x in rank_groups],
        global_world_size=world_size,
    )
    return sp_group
