import contextlib
import logging
import typing
from typing import Dict, Optional, Union

import torch
from transformers import PreTrainedModel
from transformers.initialization import no_init_weights
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from tensor_cast.transformers.transformations import (
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
from ..layers.utils import ModelWrapperBase
from ..model_config import ModelConfig
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
        **kwargs: object,  # NOTE: extra args should be torch.compile compatible
    ) -> Union[torch.Tensor, TensorDict]:
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
        if sampling_metadata and sampling_metadata.selected_token_indices is not None:
            hidden_states = hidden_states.index_select(1, sampling_metadata.selected_token_indices)
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
        **kwargs: object,
    ) -> Union[torch.Tensor, TensorDict]:
        outputs = self._inner(
            input_ids=input_ids,
            use_cache=False,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        sampling_metadata: Optional[SamplingMetadata] = kwargs.get("sampling_metadata")
        if sampling_metadata and sampling_metadata.selected_token_indices is not None:
            hidden_states = hidden_states.index_select(1, sampling_metadata.selected_token_indices)
        logits = self.lm_head(hidden_states)

        return logits


class ModelWrapper(ModelWrapperBase):
    def __init__(self, model: torch.nn.Module):
        super().__init__(model)

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        position_ids: torch.Tensor,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,  # NOTE: extra args should be torch.compile compatible
    ) -> Union[torch.Tensor, TensorDict]:
        hidden_states = self._inner(
            input_ids=input_ids,
            use_cache=False,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            return_dict=False,
            **kwargs,
        )[0]
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
        return total_size

    @property
    def num_hidden_layers(self):
        num_hidden_layers = self.text_config.num_hidden_layers
        if self.model_config.mtp_config:
            num_hidden_layers += self.model_config.mtp_config.num_mtp_layers
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
        context = contextlib.nullcontext()
        if not torch.compiler.is_compiling():
            context = patch_find_packed_sequence_indices_for_meta()
        with context:
            return self._inner(
                input_ids=input_ids,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                attention_by_layers=getattr(self, "attention_by_layers", None),
                **kwargs,
            )
