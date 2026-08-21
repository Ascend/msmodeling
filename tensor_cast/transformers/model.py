import contextlib
import logging
import typing
from typing import Dict, Optional, Sequence, Union

import torch
from transformers import PreTrainedModel
from transformers.initialization import no_init_weights
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from tensor_cast.transformers.transformations import (
    maybe_enable_dflash,
    maybe_enable_dspark,
    maybe_enable_mtp,
    maybe_reuse_layers,
    patch_attention,
    patch_mla,
    patch_moe,
    patch_rotary_emb,
    quantize_model,
    shard_model,
    wrap_model,
)
from ..layers.attention import flash_attention_forward
from ..layers.sampler import select_lm_head_hidden_states
from ..layers.utils import ModelWrapperBase
from ..model_config import ModelConfig, config_has_draft_spec
from ..parallel_group import ParallelGroupManager
from ..performance_model.utils import bytes_of_tensor
from .custom_model_registry import get_custom_model
from .transformations import patch_model
from .utils import (
    AutoModelConfigLoader,
    init_on_device_without_buffers,
    patch_find_packed_sequence_indices_for_meta,
)

if typing.TYPE_CHECKING:
    from ..layers.sampler import SamplingMetadata

logger = logging.getLogger(__name__)

ALL_ATTENTION_FUNCTIONS["tensor_cast"] = flash_attention_forward

# Keys that ModelRunner injects into each attention layer's
# _extra_forward_kwargs side-channel.  The same set of keys also
# appears in the kwargs dicts returned by generate_inputs() /
# generate_inputs_varlen().
_EXTRA_TC_KWARGS_KEYS = (
    "attention_meta",
    "kv_cache_by_layers",
    "kv_cache_per_token",
    "sampling_metadata",
    "attention_by_layers",
)


def resolve_decoder_num_layers(hf_config: object) -> int:
    """Return decoder ``num_hidden_layers`` from HF / text config."""
    if hf_config is None:
        raise ValueError("hf_config is required to resolve decoder num_layers")
    text_cfg = hf_config
    getter = getattr(hf_config, "get_text_config", None)
    if callable(getter):
        text_cfg = getter()
    elif getattr(hf_config, "text_config", None) is not None:
        text_cfg = hf_config.text_config
    n = getattr(text_cfg, "num_hidden_layers", None)
    if n is None:
        raise ValueError(f"config has no num_hidden_layers: {type(hf_config)!r}")
    return int(n)


def select_aux_hidden_states(
    all_hidden_states: Sequence[torch.Tensor],
    layer_ids: Sequence[int],
    *,
    num_layers: int,
) -> list[torch.Tensor]:
    """Pick decoder-layer outputs from HF ``output_hidden_states`` tuple.

    Layout is determined only by ``num_layers``:

    - ``len == num_layers + 1`` → ``(embed_out, layer0, …, layerN-1)``, offset 1
    - ``len == num_layers`` → layers only, offset 0
    - otherwise → error (do not guess from max layer id)
    """
    if not layer_ids:
        raise ValueError("layer_ids must be non-empty")
    if not all_hidden_states:
        raise ValueError("all_hidden_states must be non-empty")
    num_layers = int(num_layers)
    if num_layers < 1:
        raise ValueError(f"num_layers must be >= 1, got {num_layers}")
    n = len(all_hidden_states)
    if n == num_layers + 1:
        offset = 1
    elif n == num_layers:
        offset = 0
    else:
        raise ValueError(
            f"hidden_states length={n} does not match num_layers={num_layers} "
            f"or num_layers+1 (embed+layers); cannot determine layout "
            f"(layer_ids={list(layer_ids)})"
        )
    max_id = max(int(i) for i in layer_ids)
    if max_id < 0:
        raise ValueError(f"layer_ids must be non-negative, got {layer_ids}")
    if max_id >= num_layers:
        raise ValueError(
            f"aux layer id {max_id} out of range for num_layers={num_layers} (layer_ids={list(layer_ids)})"
        )
    return [all_hidden_states[int(i) + offset] for i in layer_ids]


def _unpack_model_hidden_outputs(outputs: object) -> tuple[torch.Tensor, Sequence[torch.Tensor]]:
    """Normalize HF return_dict / tuple outputs to ``(last_hidden, all_hidden)``."""
    if hasattr(outputs, "last_hidden_state") and getattr(outputs, "hidden_states", None) is not None:
        return outputs.last_hidden_state, outputs.hidden_states
    if isinstance(outputs, (tuple, list)):
        # Common: (last_hidden, ..., all_hidden_states) with all_hidden near the end.
        last = outputs[0]
        all_hidden = None
        for item in reversed(outputs[1:]):
            if isinstance(item, (tuple, list)) and item and isinstance(item[0], torch.Tensor):
                all_hidden = item
                break
        if all_hidden is None:
            raise RuntimeError(f"Unable to locate hidden_states in model outputs: {type(outputs)}")
        return last, all_hidden
    raise RuntimeError(f"Unexpected model output type for aux collection: {type(outputs)}")


class TensorDict:
    def __init__(self, tensors: Dict[str, torch.Tensor]):
        self.tensors = tensors


class CausalLmWrapper(ModelWrapperBase):
    def __init__(self, hf_config, model: torch.nn.Module):
        super().__init__(model)
        self.hf_config = hf_config
        self.lm_head = torch.nn.Linear(
            self.hf_config.hidden_size,
            self.hf_config.vocab_size,
            bias=False,
        )

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        position_ids: torch.Tensor,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_intermediate_hidden_states: bool = False,  # output hidden_states before lm_head
        output_aux_hidden_state_layer_ids: Optional[Sequence[int]] = None,
        **kwargs: object,  # NOTE: extra args should be torch.compile compatible
    ) -> Union[torch.Tensor, TensorDict, tuple]:
        want_aux = output_aux_hidden_state_layer_ids is not None
        if want_aux:
            outputs = self._inner(
                input_ids=input_ids,
                use_cache=False,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                output_hidden_states=True,
                return_dict=True,
                **kwargs,
            )
            last_hidden, all_hidden = _unpack_model_hidden_outputs(outputs)
            aux_hiddens = select_aux_hidden_states(
                all_hidden,
                output_aux_hidden_state_layer_ids,
                num_layers=resolve_decoder_num_layers(self.hf_config),
            )
            intermediate_hidden_states = last_hidden
            sampling_metadata: Optional[SamplingMetadata] = kwargs.get("sampling_metadata")
            logits = self.lm_head(select_lm_head_hidden_states(last_hidden, sampling_metadata, mode="target"))
            if output_intermediate_hidden_states:
                return logits, intermediate_hidden_states, aux_hiddens
            return logits, aux_hiddens

        hidden_states = self._inner(
            input_ids=input_ids,
            use_cache=False,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            return_dict=False,
            **kwargs,
        )[0]
        intermediate_hidden_states = hidden_states
        sampling_metadata: Optional[SamplingMetadata] = kwargs.get("sampling_metadata")
        hidden_states = select_lm_head_hidden_states(hidden_states, sampling_metadata, mode="target")
        hidden_states = self.lm_head(hidden_states)
        if output_intermediate_hidden_states:
            return hidden_states, intermediate_hidden_states
        else:
            return hidden_states


class VLModelWrapper(ModelWrapperBase):
    """
    Vision-Language model wrapper, for Qwen3 VL multimodal models
    """

    def __init__(self, hf_config, model: torch.nn.Module):
        super().__init__(model)
        self.hf_config = hf_config
        hidden_size = hf_config.text_config.hidden_size
        vocab_size = hf_config.text_config.vocab_size
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        position_ids: torch.Tensor,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_intermediate_hidden_states: bool = False,
        output_aux_hidden_state_layer_ids: Optional[Sequence[int]] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, TensorDict, tuple]:
        want_aux = output_aux_hidden_state_layer_ids is not None
        outputs = self._inner(
            input_ids=input_ids,
            use_cache=False,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=want_aux,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        sampling_metadata: Optional[SamplingMetadata] = kwargs.get("sampling_metadata")
        logits = self.lm_head(select_lm_head_hidden_states(hidden_states, sampling_metadata, mode="target"))

        if want_aux:
            all_hidden = getattr(outputs, "hidden_states", None)
            if all_hidden is None:
                raise RuntimeError("VL model did not return hidden_states for Dflash aux")
            aux_hiddens = select_aux_hidden_states(
                all_hidden,
                output_aux_hidden_state_layer_ids,
                num_layers=resolve_decoder_num_layers(self.hf_config),
            )
            if output_intermediate_hidden_states:
                return logits, outputs.last_hidden_state, aux_hiddens
            return logits, aux_hiddens

        if output_intermediate_hidden_states:
            return logits, outputs.last_hidden_state
        return logits


class ModelWrapper(ModelWrapperBase):
    def __init__(self, model: torch.nn.Module):
        super().__init__(model)

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        position_ids: torch.Tensor,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_intermediate_hidden_states: bool = False,
        output_aux_hidden_state_layer_ids: Optional[Sequence[int]] = None,
        **kwargs: object,  # NOTE: extra args should be torch.compile compatible
    ) -> Union[torch.Tensor, TensorDict, tuple]:
        if output_aux_hidden_state_layer_ids is not None:
            outputs = self._inner(
                input_ids=input_ids,
                use_cache=False,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                output_hidden_states=True,
                return_dict=True,
                **kwargs,
            )
            sampling_metadata: Optional[SamplingMetadata] = kwargs.get("sampling_metadata")
            if hasattr(outputs, "logits"):
                logits = outputs.logits
                all_hidden = outputs.hidden_states
                last_hidden = all_hidden[-1] if all_hidden is not None else getattr(outputs, "last_hidden_state", None)
            else:
                last_hidden, all_hidden = _unpack_model_hidden_outputs(outputs)
                logits = last_hidden
            if all_hidden is None:
                raise RuntimeError("Model did not return hidden_states for Dflash aux")
            cfg = getattr(self._inner, "config", None)
            aux_hiddens = select_aux_hidden_states(
                all_hidden,
                output_aux_hidden_state_layer_ids,
                num_layers=resolve_decoder_num_layers(cfg),
            )
            logits = select_lm_head_hidden_states(logits, sampling_metadata, mode="target")
            if output_intermediate_hidden_states:
                if last_hidden is None:
                    last_hidden = all_hidden[-1]
                return logits, last_hidden, aux_hiddens
            return logits, aux_hiddens

        hidden_states = self._inner(
            input_ids=input_ids,
            use_cache=False,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            return_dict=False,
            **kwargs,
        )[0]
        if output_intermediate_hidden_states:
            return hidden_states, hidden_states
        return hidden_states


class TransformerModel(ModelWrapperBase):
    def __init__(
        self,
        model_id: str,
        model_config: ModelConfig,
        hf_model: PreTrainedModel = None,
    ):
        """
        Construct a transformer model wrapper that auto-loads a transformer model and converts
        it into a model according to the given model configuration.

        Args:
            model_id: transformer model id, (`str` or `os.PathLike`)
            model_config: specify how we should load and convert the transformer model
            hf_model: native model
            #TODO: native model + running config(model_config) = running model,do not need model_id
        """
        super().__init__(None)

        self.model_id = model_id
        self.model_config = model_config

        logger.info("Initializing 'TransformerModel' for model_id: %s", model_id)
        with init_on_device_without_buffers("meta"), no_init_weights():
            auto_loader = AutoModelConfigLoader()
            if self.model_config.hf_config is not None:
                logger.info("Using provided HuggingFace configuration")
                self.hf_config = self.model_config.hf_config

                # Apply patches for specific models before loading them
                auto_loader._apply_hf_config_patches(self.hf_config, model_id)

                if self.model_config.num_hidden_layers_override:
                    logger.info(
                        "Overriding num_hidden_layers to %s",
                        model_config.num_hidden_layers_override,
                    )
                    self.hf_config.get_text_config().num_hidden_layers = model_config.num_hidden_layers_override
                self._inner = auto_loader.load_model(
                    self.hf_config,
                    self.model_config.dtype,
                    trust_remote_code=self.model_config.trust_remote_code,
                )
            else:
                logger.info("Auto-loading model and configuration for: %s", model_id)
                self.hf_config, self._inner = auto_loader.auto_load_model_and_config(self.model_id, self.model_config)
            logger.info("origin model and config are loaded successfully")

            self.text_config = self.hf_config.get_text_config()
            self.is_vl_model = hasattr(self.hf_config, "vision_config")
            logger.info("Model type: %s", "Vision-Language" if self.is_vl_model else "Text-only")

            if self.model_config.attention_cls and self.model_config.attention_cls.attn_implmentation:
                attn_impl = self.model_config.attention_cls.attn_implmentation
                logger.info("Setting attention implementation to: %s", attn_impl)
                self.text_config._attn_implementation = attn_impl
                if self.is_vl_model:
                    self.hf_config.vision_config._attn_implementation = attn_impl

            logger.info("Initializing parallel groups")
            self.parallel_group_manager = ParallelGroupManager(self.model_config.parallel_config)
            # the order of these functions matters!
            logger.info("Applying model transformations")
            model_type = self.hf_config.model_type
            with self.set_default_dtype():
                custom_fn = get_custom_model(model_type)
                if custom_fn:
                    custom_fn(self)
                else:
                    wrap_model(self)
                    maybe_enable_mtp(self)
                    maybe_enable_dflash(self)
                    maybe_enable_dspark(self)
                    maybe_reuse_layers(self)
                    patch_model(self)
                    patch_rotary_emb(self)
                    patch_attention(self)
                    patch_mla(self)
                    patch_moe(self)
                    quantize_model(self)
                    shard_model(self)

        logger.info("Loading model weights")
        self.load_weights()

    @contextlib.contextmanager
    def set_default_dtype(self):
        orig_dtype = torch.get_default_dtype()
        torch.set_default_dtype(self.model_config.dtype)
        try:
            yield
        finally:
            torch.set_default_dtype(orig_dtype)

    def load_weights(self):
        """TODO: load real weights"""

    def _replace_module(self, name: str, new_module: torch.nn.Module):
        # Split module path to get parent and child name
        path = name.split(".")
        parent_name = ".".join(path[:-1])
        child_name = path[-1]
        # Find the parent module
        parent_module = self._inner
        if parent_name:
            parent_module = self._inner.get_submodule(parent_name)
        setattr(parent_module, child_name, new_module)

    @staticmethod
    def get_weight_size_nested(modules):
        total_size = 0
        for mod in modules:
            for _, param in mod.named_parameters():
                total_size += bytes_of_tensor(param)
            for _, buffer in mod.named_buffers():
                total_size += bytes_of_tensor(buffer)
            total_size += TransformerModel.get_represented_extra_weight_size(mod)
        return total_size

    @staticmethod
    def get_represented_extra_weight_size(module):
        from ..layers.internal import RegionMarkerWrapper

        total_size = 0
        for submodule in module.modules():
            if not isinstance(submodule, RegionMarkerWrapper):
                continue
            repeat_count = submodule.repeat_count
            if repeat_count <= 1:
                continue
            # RegionMarkerWrapper wraps the original representative layer. Its inner layer should not contain another
            # RegionMarkerWrapper, which keeps represented-weight accounting finite and avoids double counting.
            total_size += (repeat_count - 1) * TransformerModel.get_weight_size_nested([submodule._inner])
        return total_size

    @property
    def num_hidden_layers(self):
        num_hidden_layers = self.text_config.num_hidden_layers
        if self.model_config.mtp_config:
            num_hidden_layers += self.model_config.mtp_config.num_mtp_layers
        if config_has_draft_spec(self.model_config):
            num_hidden_layers += self.model_config.draft_num_layers()
        return num_hidden_layers

    @property
    def hidden_size(self):
        return self.text_config.hidden_size

    @property
    def intermediate_size(self):
        return self.text_config.intermediate_size

    @property
    def vocab_size(self):
        return self.text_config.vocab_size

    @property
    def head_dim(self):
        return getattr(
            self.text_config,
            "head_dim",
            self.hidden_size // self.text_config.num_attention_heads,
        )

    @property
    def weight_size(self):
        return self.get_weight_size_nested([self])

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        position_ids: torch.Tensor,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,  # NOTE: extra args should be torch.compile compatible
    ) -> Union[torch.Tensor, TensorDict]:
        """
        Tensors will be migrated to fake tensor in follow-up work; this patch will be removed.
        """

        # Store tc_kwargs in the instance variable and explicitly inject full_kwargs
        tc_kwargs = {key: kwargs.get(key) for key in _EXTRA_TC_KWARGS_KEYS}
        # attention_by_layers may also be set as an instance attribute on the wrapper
        attention_by_layers = getattr(self, "attention_by_layers", None)
        if attention_by_layers is not None:
            tc_kwargs["attention_by_layers"] = attention_by_layers
        full_kwargs = {**kwargs, **tc_kwargs}

        context = contextlib.nullcontext()
        if not torch.compiler.is_compiling():
            context = patch_find_packed_sequence_indices_for_meta()
        with context:
            return self._inner(
                input_ids=input_ids,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                **full_kwargs,
            )
