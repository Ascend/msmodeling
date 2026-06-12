import json
import logging
import os
from typing import Dict, Optional

import numpy as np
import torch
from transformers.initialization import no_init_weights

from ..layers.attention import AttentionTensorCast
from ..layers.quant_linear import TensorCastQuantLinear
from ..model_config import (
    DiffusersConfig,
    DiffusersTransformerConfig,
    DiffusersVaeConfig,
)
from ..parallel_group import ParallelGroup
from ..transformers.model import ModelWrapperBase
from ..transformers.transformations import quantize_linear, quantize_model, wrap_model
from ..transformers.utils import init_on_device_without_buffers
from .cache_agent import CacheConfig, CacheState
from .cache_agent.dit_block_cache import DiTBlockCache
from .diffusers_utils import get_diffusers_transformer_module
from .dit_cache_registry import get_dit_block_cache_spec, replace_blocks_in_range

logger = logging.getLogger(__name__)


def build_diffusers_transformer_model(
    model_id: str,
    parallel_config: None,
    quant_config: None,
    dtype: torch.dtype,
):
    model_config = load_config_from_file(
        model_path=model_id,
        parallel_config=parallel_config,
        quant_config=quant_config,
        quant_linear_cls=TensorCastQuantLinear,
        attention_cls=AttentionTensorCast,
        dtype=dtype,
    )
    model = DiffusersTransformerModel(model_id, model_config.transformer_config)
    return model, model_config


def load_config_from_file(
    model_path: str,
    parallel_config: None,
    quant_config: None,
    quant_linear_cls: None,
    attention_cls: None,
    dtype: torch.dtype,
):
    # TODO add seperate parallel_config and quant_config(atten_cls is needed?) for vae and text
    if not os.path.isdir(model_path):
        raise ValueError(f"Input args.model_id should be dir, but got {model_path}")

    config_path_dict: Dict[str, str] = {}
    model_path = os.path.abspath(model_path)
    for root, _, files in os.walk(model_path):
        if "config.json" in files:
            folder_name = os.path.basename(root)
            config_path = os.path.join(root, "config.json")
            config_path = os.path.abspath(config_path)
            config_path_dict[folder_name] = config_path

    config_dict: Dict[str, Dict] = {}
    for key, config_path in config_path_dict.items():
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
        config_dict[key] = config

    transformer_config_json_path = config_path_dict.get("transformer")
    transformer_config = config_dict.get("transformer")
    if transformer_config_json_path is None or transformer_config is None:
        # Fall back to a single candidate that looks like a Diffusers Transformer config.
        def _looks_like_transformer_config(cfg: Dict) -> bool:
            class_name = cfg.get("_class_name")
            return isinstance(class_name, str) and "Transformer" in class_name

        transformer_candidates: Dict[str, str] = {}
        for folder_name, cfg in config_dict.items():
            if _looks_like_transformer_config(cfg):
                transformer_candidates[folder_name] = config_path_dict[folder_name]

        if len(transformer_candidates) == 1:
            folder_name, path = next(iter(transformer_candidates.items()))
            transformer_config_json_path = path
            transformer_config = config_dict[folder_name]
        else:
            raise ValueError(
                "No transformer/config.json found in input model path. "
                "Expect a Diffusers-style model directory that contains transformer/config.json."
            )

    vae_config_json_path = config_path_dict.get("vae")

    model_config = DiffusersConfig()
    model_config.model_path = model_path
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
        encoder_hidden_states_images: Optional[torch.Tensor] = None,
        return_dict=False,
        **kwargs: object,
    ):
        hidden_states = self._inner(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )[0]
        return hidden_states

    def enable_dit_block_cache(self, cache_config: CacheConfig) -> Optional[CacheState]:
        """
        Enable DiT block cache (dit_block_cache).

        Replace blocks in the configured cache range with cache-aware wrappers.
        Step scheduling (update/reuse/bypass) is driven externally by the caller.
        """
        model_config = self.model_config.model_config or {}
        class_name = model_config.get("_class_name")
        spec = get_dit_block_cache_spec(class_name)
        if spec is None:
            logger.warning("dit_block_cache is not implemented for model %r.", class_name)
            return None

        blocks_with_setters = list(spec.get_blocks_with_setters(self._inner))
        if not blocks_with_setters:
            return None
        blocks_count = len(blocks_with_setters)

        bounded_block_end = min(cache_config.block_end, blocks_count)

        cache_state = CacheState()
        replaced = replace_blocks_in_range(
            blocks_with_setters,
            cache_config.block_start,
            bounded_block_end,
            lambda block, flat_idx: DiTBlockCache(
                block=block,
                state=cache_state,
                block_index=flat_idx,
                block_start=cache_config.block_start,
                block_end=bounded_block_end,
                make_wrapped_forward=spec.make_wrapped_forward,
            ),
        )

        logger.info(
            "Enabled dit_block_cache for %s: replaced %d blocks in range [%d, %d) out of %d.",
            spec.model_type,
            replaced,
            cache_config.block_start,
            bounded_block_end,
            blocks_count,
        )
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
