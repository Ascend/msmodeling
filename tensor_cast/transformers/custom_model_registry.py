import dataclasses
import fnmatch
import importlib
import logging
import operator
import os
from typing import Any, Callable, Dict, List, Optional, Type, TYPE_CHECKING, Union

import torch

from ..layers.mla import MultiheadLatentAttentionTensorCast
from ..model_config import MlaConfig, MlaFieldNames, MoEConfig, MoEFieldNames, MtpConfig

if TYPE_CHECKING:
    from ..layers.utils import ModelWrapperBase
    from .model import TransformerModel


logger = logging.getLogger(__name__)


_CUSTOM_MODEL_REGISTRY: Dict[str, Callable] = {}
_USER_CUSTOM_MODEL_LOADED = False


"""
This dictionary defines the access paths for model components and their
structural mapping during weight conversion or parallelization.

Key Descriptions:
visual:
    - Meaning: Retrieves the Vision Encoder instance.
    - Purpose: Points to the root module responsible for image feature extraction.

language_model:
    - Meaning: Retrieves the Language Model (LLM) instance.
    - Purpose: Points to the core LLM responsible for text processing and multi-modal fusion.

visual.layers:
    - Meaning: Points to the list of layers (Transformer Layers) within the vision module.
    - Distinction: This is an [Object Accessor]. It tells the program how to retrieve the
      actual Layer objects from the model instance.
    - Mapping: Internally usually corresponds to `visual.blocks` (e.g., Qwen2-VL or GLM).

path.visual.layers:
    - Meaning: The [String Path Representation] of vision layers inside the model.
    - Distinction: This is a [Path Mapping]. It returns a string "visual.blocks" rather than an object.
    - Purpose: Used for distributed strategies or logging to identify weight namespaces in state_dict.

path.language_model.layers:
    - Meaning: The [String Path Representation] of language model layers.
    - Purpose: Same as above, mapping to "language_model.layers".

visual_merger_linear:
    - Meaning: Configuration for linear layers in the vision feature fusion layer (Merger/Projector).
    - Purpose: Targets linear mapping layers that merge or transform multiple visual tokens.
      Returning an empty dict typically indicates using the default parallel strategy.

visual_mlp_linear:
    - Meaning: Configuration for linear layers within the MLP blocks of the vision module.
    - Purpose: Points to the Feed-Forward Network (FFN) inside each Vision Transformer layer.
"""
COMMON_VISUAL_CONFIG = {
    "visual_module_path": "visual",
    "language_module_path": "language_model",
    "visual_layers_module_path": "visual.blocks",
    "visual_layers_path_str": "visual.blocks",
    "language_layers_path_str": "language_model.layers",
    "visual_merger_linear_mapping": {},
    "visual_mlp_linear_mapping": {},
}


def resolve_visual_config(
    custom_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Resolve visual configuration by merging custom config with common defaults.
    Used to generate arguments for ModelProfile's visual configuration fields.
    """
    config = COMMON_VISUAL_CONFIG.copy()
    if custom_config:
        config.update(custom_config)
    return config


class MoeExpertMLP(torch.nn.Module):
    def __init__(self, original_experts_module: torch.nn.Module, expert_idx: int):
        super().__init__()
        self.expert_idx = expert_idx
        self.hidden_size = original_experts_module.hidden_dim
        self.intermediate_size = original_experts_module.intermediate_dim
        self.act_fn = original_experts_module.act_fn

        intermediate_dim = original_experts_module.intermediate_dim
        hidden_dim = original_experts_module.hidden_dim

        self.gate_proj = torch.nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj = torch.nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = torch.nn.Linear(intermediate_dim, hidden_dim, bias=False)

        with torch.no_grad():
            gate_up_weight = original_experts_module.gate_up_proj.data[expert_idx]
            gate_weight, up_weight = gate_up_weight.chunk(2, dim=0)
            self.gate_proj.weight.copy_(gate_weight)
            self.up_proj.weight.copy_(up_weight)
            self.down_proj.weight.copy_(original_experts_module.down_proj.data[expert_idx])

    def forward(self, hidden_states):
        gate = self.gate_proj(hidden_states)
        up = self.up_proj(hidden_states)
        hidden_states = self.down_proj(up * self.act_fn(gate))
        return hidden_states


@dataclasses.dataclass
class ModelProfile:
    """Model Profile containing static metadata and factory methods to build runtime configurations.

    Supported configurations:
    - MoE (Mixture-of-Experts)
    - MTP (Multi-Task Processing)
    - MLA (Multihead Latent Attention)
    - Custom expert module
    - Model family mapping
    - Visual language model patching

    Model families group related model types for unified processing.
    """

    # Corresponds to the `model_type` field in the model's `config.json` (HuggingFace/ModelScope).
    # Used as the primary key for model identification.
    # Example: "llama", "qwen2", "deepseek_v3"
    model_type: str

    # --- MoE (Mixture-of-Experts) Configuration ---

    # Fully-qualified class name of the MoE module defined in `modeling_{model_type}.py`.
    # Typically follows the naming pattern `{ModelType}SparseMoeBlock` or `{ModelType}MoE`.
    # Leave as None for dense (non-MoE) models.
    moe_module_name: Optional[str] = None

    # The key in `config.json` that specifies the total number of experts.
    # Example: If config has {"num_local_experts": 64}, set this to "num_local_experts".
    moe_num_experts_key: Union[str, List[str]] = "num_experts"

    # Indicates if the MoE gate/router returns *raw, unprocessed logits*.
    #
    # Logic for setting this field:
    # 1. [Default/False] If the gate class (in modeling_{model_type}.py) internally performs
    #    softmax/sigmoid AND top-k selection (e.g., Ernie4_5_MoeTopKRouter), leave this as False
    #    (or omit it). The framework expects ready-to-use weights.
    # 2. [True] Only set to True if the gate returns raw logits (e.g., output of a Linear layer)
    #    without any probability conversion or token routing. The framework will then handle softmax.
    moe_gate_returns_raw_logits: bool = False

    # Configuration object to map standard MoE attribute names to the model's specific variable names.
    #
    # 【What is this class?】
    # It defines the standard variable names (fields) that the framework expects to find in the MoE module.
    # - gate: The router/gate network.
    # - experts: The list of routed experts.
    # - shared_experts: The shared expert layer.
    # - shared_experts_gate: The gate for shared experts.
    #
    # 【How to locate the MoE Class?】
    # 1. Open `transformers/models/{model_type}/modeling_{model_type}.py`.
    # 2. Search for a class named `{ModelType}SparseMoeBlock` or `{ModelType}MoE`.
    #    - This class acts as the "container" that holds the gate, experts, and shared experts.
    #    - Example: `Qwen3NextSparseMoeBlock`, `MixtralSparseMoeBlock`.
    # 3. Inspect its `__init__` method to find the specific attribute names (e.g., `self.shared_expert`).
    #
    # 【Filling Rule】
    # - The field name (e.g., `shared_experts`) represents the Standard Name.
    # - The value you assign (e.g., `"shared_expert"`) represents the Model's Actual Name.
    # - **Only override fields where the names differ.** If the model uses the standard name, ignore it.
    #
    # 【Example: Qwen3Next】
    # Source code: `self.shared_expert = Qwen3NextMLP(...)` (Note the singular 'expert')
    # Assignment:   moe_field_names_override=MoEFieldNames(shared_experts="shared_expert")
    moe_field_names_override: Optional[MoEFieldNames] = None

    # --- MTP (Multi-Token Prediction) Configuration ---
    # Class name implementing the Multi-Token Prediction (or Speculative Decoding) logic.
    # Used to identify blocks responsible for predicting future tokens (e.g., DeepSeekV3DecoderLayer).
    # Leave as None if the model does not support MTP.
    mtp_block_module_name: Optional[str] = None

    # --- MLA (Multi-head Latent Attention) Configuration ---

    # Fully-qualified class name implementing the MLA mechanism.
    # Used to apply specific MLA optimizations (e.g., FlashMLA, KV cache compression).
    # Example: "transformers.models.deepseek_v3.modeling_deepseek.DeepseekV3Attention"
    mla_module_name: Optional[str] = None

    # Dictionary to map standard MLA field names to the model's specific attribute names.
    # Used when the model's internal naming for MLA components (like compressed KV projections)
    # differs from the standard.
    # Example: {"q_proj": "q_a_proj", "kv_a_proj_with_mqa": "kv_a_layernorm"}
    mla_field_names_override: Optional[Dict[str, Any]] = None

    # Python class type used for the MLA module implementation.
    # Defaults to the built-in tensor casting class. Override only for custom MLA implementations.
    mla_module_class_type: Optional[Type["torch.nn.Module"]] = MultiheadLatentAttentionTensorCast

    # Python type used to dynamically instantiate a custom expert module.
    # Use this if the standard MLP structure does not fit the model's expert architecture.
    custom_expert_module_type: Optional[Type["torch.nn.Module"]] = MoeExpertMLP

    # Identifier for the model family, grouping related architectures for unified processing.
    # Example: "llama" family might include "llama", "baichuan", "yi".
    model_family: Optional[str] = None

    # Callable method for dynamic model patching during loading.
    # Used to structurally modify the model (e.g., operator replacement) at runtime.
    patch_method: Optional[Callable] = None

    # Attribute path to the Vision Encoder instance within the root model.
    # Example: "model.vision_tower"
    visual_module_path: Optional[str] = None

    # Attribute path to the Language Model (LLM) instance within the root model.
    # Example: "model.language_model"
    language_module_path: Optional[str] = None

    # Python module path where vision layer classes are defined.
    # Used for dynamic imports during model parsing.
    # Example: "transformers.models.clip.modeling_clip"
    visual_layers_module_path: Optional[str] = None

    # Dot-separated attribute path to access vision transformer layers in the model instance.
    # Example: "vision_model.encoder.layers"
    visual_layers_path_str: Optional[str] = None

    # Dot-separated attribute path to access language model layers.
    # Example: "language_model.layers"
    language_layers_path_str: Optional[str] = None

    # Mapping for linear layers in the visual feature merger/projector.
    # Defines how visual features are projected to the language space.
    visual_merger_linear_mapping: Optional[Dict[str, Any]] = dataclasses.field(default_factory=dict)

    # Mapping for linear layers (MLP/FFN) inside the vision encoder blocks.
    # Used to locate specific weights like fc1/fc2 within vision transformer layers.
    visual_mlp_linear_mapping: Optional[Dict[str, Any]] = dataclasses.field(default_factory=dict)

    def _build_field_names(self, base_class: Type, override_dict: Optional[Dict[str, Any]]) -> Any:
        if not override_dict:
            return base_class()

        existing_fields = {f.name: getattr(base_class(), f.name) for f in dataclasses.fields(base_class())}
        existing_fields.update(override_dict)
        return base_class(**existing_fields)

    def build_moe_config(
        self,
        enable_redundant: bool = False,
        enable_external_shared: bool = False,
        host_external_shared: bool = False,
        fused_moe_cls: Optional[Type] = None,
    ) -> Optional[MoEConfig]:
        if not self.moe_module_name:
            return None

        return MoEConfig(
            module_name=self.moe_module_name,
            fused_moe_cls=fused_moe_cls,
            field_names=self.moe_field_names_override or MoEFieldNames(),
            gate_returns_raw_logits=self.moe_gate_returns_raw_logits,
            enable_redundant_experts=enable_redundant,
            enable_external_shared_experts=enable_external_shared,
            host_external_shared_experts=host_external_shared,
            num_experts_key=self.moe_num_experts_key,
        )

    def build_mtp_config(self, num_mtp_layers: int) -> Optional[MtpConfig]:
        if not self.mtp_block_module_name or num_mtp_layers <= 0:
            return None

        return MtpConfig(
            num_mtp_layers=num_mtp_layers,
            mtp_block_module_name=self.mtp_block_module_name,
        )

    def build_mla_config(self) -> Optional[MlaConfig]:
        if not self.mla_module_name:
            return None

        field_names = self._build_field_names(MlaFieldNames, self.mla_field_names_override)

        return MlaConfig(
            module_name=self.mla_module_name,
            field_names=field_names,
        )


def get_model_family(model_type: str) -> Optional[str]:
    profile = get_model_profile(model_type)
    if profile is None:
        return None
    return profile.model_family


def get_mla_module(model_type: str) -> Type["torch.nn.Module"]:
    profile = get_model_profile(model_type)
    if profile is None:
        return MultiheadLatentAttentionTensorCast
    return profile.mla_module_class_type


_MODEL_PROFILE_REGISTRY: Dict[str, ModelProfile] = {}


def register_model_profile(profile: ModelProfile):
    """
    Registers a ModelProfile instance.
    Should be used as a decorator or called directly after defining the profile.
    """
    if profile.model_type in _MODEL_PROFILE_REGISTRY:
        raise ValueError(f"ModelProfile for '{profile.model_type}' is already registered.")

    _MODEL_PROFILE_REGISTRY[profile.model_type] = profile
    return profile


def get_model_profile(model_type: str) -> Optional[ModelProfile]:
    """
    Retrieves the ModelProfile for a given model type.
    Returns None if the model type is not registered.
    """
    return _MODEL_PROFILE_REGISTRY.get(model_type)


def get_moe_config(model_type: str = "") -> Optional[MoEConfig]:
    if not model_type:
        return None

    profile = get_model_profile(model_type)
    if profile is None:
        return None

    return profile.build_moe_config(
        enable_redundant=False,
        enable_external_shared=False,
        host_external_shared=False,
        fused_moe_cls=None,
    )


def get_vl_model_module(model: "ModelWrapperBase", profile_attr: str, default_key: str):
    profile = get_model_profile(model.hf_config.model_type)
    path = getattr(profile, profile_attr, None)
    if not path and profile and profile.model_family == "default":
        path = COMMON_VISUAL_CONFIG[default_key]
    return operator.attrgetter(path)(model.unwrap()) if path else None


def get_visual(model: "ModelWrapperBase"):
    return get_vl_model_module(model, "visual_module_path", "visual_module_path")


def get_vl_language_model(model: "ModelWrapperBase"):
    return get_vl_model_module(model, "language_module_path", "language_module_path")


def get_visual_layers(model: "ModelWrapperBase"):
    return get_vl_model_module(model, "visual_layers_module_path", "visual_layers_module_path")


def get_vl_model_profile_attr(model_type: str, profile_attr: str, default_key: str, fallback_value=None):
    profile = get_model_profile(model_type)
    if profile and getattr(profile, profile_attr, None):
        return getattr(profile, profile_attr)

    if profile and profile.model_family == "default":
        return COMMON_VISUAL_CONFIG[default_key]
    return fallback_value


def get_visual_merger_linear(model_type: str):
    return get_vl_model_profile_attr(model_type, "visual_merger_linear_mapping", "visual_merger_linear_mapping", {})


def get_visual_mlp_linear(model_type: str):
    return get_vl_model_profile_attr(model_type, "visual_mlp_linear_mapping", "visual_visual_mlp_linear_mapping", {})


def get_visual_layers_path(model_type: str) -> Optional[str]:
    return get_vl_model_profile_attr(model_type, "visual_layers_path_str", "visual_layers_path_str", None)


def get_language_layers(model_type: str) -> str:
    return get_vl_model_profile_attr(model_type, "language_layers_path_str", "language_layers_path_str", "layers")


def get_mla_module_name(model_type: str = "") -> str:
    if not model_type:
        return None
    profile = get_model_profile(model_type)
    return profile.mla_module_name if profile else None


def get_mtp_block_module_name(model_type: str = "") -> str:
    if not model_type:
        return None
    profile = get_model_profile(model_type)
    return profile.mtp_block_module_name if profile else None


def find_matching_key(registry: Dict[str, Any], key: str) -> Optional[str]:
    if not key:
        return None
    for pattern in registry.keys():
        if fnmatch.fnmatchcase(key, pattern) or fnmatch.fnmatch(key, pattern):
            return pattern
    return None


def register_custom_model(model_type: str):
    def decorator(
        fn: Callable[["TransformerModel"], "TransformerModel"],
    ) -> Callable[["TransformerModel"], "TransformerModel"]:
        _CUSTOM_MODEL_REGISTRY[model_type] = fn
        return fn

    return decorator


def get_custom_model(model_type: str) -> Optional[Callable]:
    if not _USER_CUSTOM_MODEL_LOADED:
        import_custom_model_modules()

    match_key = find_matching_key(_CUSTOM_MODEL_REGISTRY, model_type)
    return _CUSTOM_MODEL_REGISTRY.get(match_key) if match_key else None


def import_custom_model_modules():
    global _USER_CUSTOM_MODEL_LOADED
    if _USER_CUSTOM_MODEL_LOADED:
        return

    _PACKAGE_ROOT = os.path.dirname(importlib.util.find_spec("tensor_cast").origin)
    custom_model_path = os.path.join(_PACKAGE_ROOT, "custom_model")
    if not os.path.exists(custom_model_path):
        return
    from tensor_cast import custom_model  # noqa: F401

    _USER_CUSTOM_MODEL_LOADED = True
