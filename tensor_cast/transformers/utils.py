# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2024 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import contextlib
import logging
import os
import torch

from typing import List, Optional, Tuple
from transformers import AutoModelForCausalLM, PretrainedConfig, PreTrainedModel
from transformers.quantizers.auto import AutoQuantizationConfig
from transformers.utils.quantization_config import (
    CompressedTensorsConfig,
    FineGrainedFP8Config,
    QuantizationConfigMixin,
)

from .custom_model_registry import get_model_profile
from ..core.model_source_security import normalize_model_source
from ..layers.mla import MultiheadLatentAttentionBase
from ..model_config import AttentionQuantConfig, ModelConfig, RemoteSource
from ..model_hub import (
    MODELSCOPE_WEIGHT_IGNORE_PATTERNS as _MODELSCOPE_WEIGHT_IGNORE_PATTERNS,  # noqa: F401
    snapshot_modelscope_without_weights,
)

logger = logging.getLogger(__name__)


def _modelscope_snapshot_config_only(model_id: str) -> str:
    """
    Materialize a local Hub directory with config and code files only (no weight tensors).

    ModelScope ``AutoConfig.from_pretrained`` may otherwise sync the full repository.
    """
    return snapshot_modelscope_without_weights(model_id)


def replace_module(model, name: str, new_module: torch.nn.Module):
    path = name.split(".")
    parent_name = ".".join(path[:-1])
    child_name = path[-1]
    parent_module = model
    if parent_name:
        parent_module = model.get_submodule(parent_name)
    setattr(parent_module, child_name, new_module)


def strip_module_name(name: str) -> str:
    """Strip `_inner` module name from the given module path name"""
    stripped = name.removeprefix("_inner.")
    stripped_before = name
    while stripped != stripped_before:
        stripped_before = stripped
        stripped = stripped_before.removeprefix("_inner.")
    stripped = stripped.replace("._inner.", ".")
    stripped_before = stripped
    stripped = stripped_before.removesuffix("._inner")
    while stripped != stripped_before:
        stripped_before = stripped
        stripped = stripped_before.removesuffix("._inner")
    return stripped


def get_attention_quant_config(model, layer_idx) -> Optional[AttentionQuantConfig]:
    if model.model_config.mla_config is not None:
        for _, module in model._inner.named_modules():
            if (
                isinstance(module, MultiheadLatentAttentionBase)
                and hasattr(module, "layer_idx")
                and module.layer_idx == layer_idx
                and (attn_quant_config := module.quant_config) is not None
            ):
                return attn_quant_config
    if hasattr(model, "attention_by_layers") and layer_idx in model.attention_by_layers:
        return model.attention_by_layers[layer_idx].quant_config
    return None


_INIT_ON_DEVICE_FACTORY_NAMES = (
    "empty",
    "zeros",
    "ones",
    "arange",
    "randn",
    "rand",
    "randint",
)


def _make_factory_use_device(factory, device: torch.device):
    def factory_with_device(*args, **kwargs):
        kwargs["device"] = device
        return factory(*args, **kwargs)

    return factory_with_device


def _move_registered_parameter(module: torch.nn.Module, name: str, device: torch.device) -> None:
    parameter = module._parameters.get(name)
    if parameter is None:
        return

    parameter_type = type(parameter)
    parameter_data = parameter.to(device)
    attributes = dict(getattr(parameter, "__dict__", {}))
    try:
        moved_parameter = parameter_type(parameter_data, requires_grad=parameter.requires_grad)
    except TypeError:
        attributes["requires_grad"] = parameter.requires_grad
        moved_parameter = parameter_type(parameter_data, **attributes)
    else:
        moved_parameter.__dict__.update(attributes)
    module._parameters[name] = moved_parameter


@contextlib.contextmanager
def init_on_device_without_buffers(device: torch.device):
    """Initialize newly registered parameters on ``device`` while leaving buffers unhooked."""

    target_device = torch.device(device)
    original_register_parameter = torch.nn.Module.register_parameter
    original_factories = {}

    def register_parameter_on_device(module, name, parameter):
        original_register_parameter(module, name, parameter)
        _move_registered_parameter(module, name, target_device)

    try:
        torch.nn.Module.register_parameter = register_parameter_on_device
        for factory_name in _INIT_ON_DEVICE_FACTORY_NAMES:
            original_factory = getattr(torch, factory_name)
            original_factories[factory_name] = original_factory
            setattr(torch, factory_name, _make_factory_use_device(original_factory, target_device))
        yield
    finally:
        torch.nn.Module.register_parameter = original_register_parameter
        for factory_name, original_factory in original_factories.items():
            setattr(torch, factory_name, original_factory)


@contextlib.contextmanager
def patch_find_packed_sequence_indices_for_meta():
    """
    This function tells the model which tokens belong to the same sentence
    when multiple sentences are packed into one batch.
    But during performance modeling (e.g., estimating memory or compute),
    we don’t care about how sequences are packed—we only need the model’s structure (like top_k=2, num_experts=64).
    Returning None simply means “assume no packing,” which is a safe and reasonable default for modeling.
    Even if real inference uses packing, it doesn’t change the model’s architecture, parameters,
    or compute graph—so performance estimates remain accurate.
    """
    from transformers import masking_utils

    original_func = masking_utils.find_packed_sequence_indices

    def safe_find_packed_sequence_indices(position_ids: torch.Tensor):
        if position_ids.device.type == "meta":
            return None
        return original_func(position_ids)

    masking_utils.find_packed_sequence_indices = safe_find_packed_sequence_indices
    try:
        yield
    finally:
        masking_utils.find_packed_sequence_indices = original_func


class AutoModelConfigLoader:
    modules_to_not_convert_map = {
        # The list of modules to not quantize, useful for quantizing models that explicitly require to have
        #   some modules left in their original precision.
        "fp8": "modules_to_not_convert",
        "fp_quant": "modules_to_not_convert",
        # layer names or types to not quantize, supports regex prefixed by 're:'
        "compressed-tensors": "ignore",
    }

    def __init__(self):
        self.is_transformers_natively_supported: bool = False
        self.resolved_model_id: Optional[str] = None

    @staticmethod
    def is_model_type_different(config: PretrainedConfig) -> Tuple[bool, str]:
        """
        Check whether the model type has changed.
        for example: kimi_k2's real model_type is deepseek_v3

        Args:
            config: hf_config.

        Returns:
            tuple: (is_different, type)
                - (False, original_type) if the types are the same
                - (True, current_type) if the types are different
        """
        # Some model config instances do not have a model_type, for example, mimo_v2_flash
        maybe_real_type = config.to_dict()["model_type"]
        if maybe_real_type and config.model_type != maybe_real_type:
            return True, maybe_real_type
        return False, config.model_type

    @staticmethod
    def check_model_path(path):
        """
        Check whether a config.json file and Python files starting with 'configuration' exist in the specified path.

        Args:
            path (str): The directory path to check.

        Returns:
            dict: A dictionary containing the check results:
                - has_config_json (bool): Whether config.json exists.
                - has_configuration_py (bool): Whether any Python file starting with 'configuration' exists.
                - configuration_py_files (list[str]): List of Python files starting with 'configuration'.
        """

        result = {
            "has_config_json": False,
            "has_configuration_py": False,
            "configuration_py_files": [],
        }

        if not os.path.exists(path) or not os.path.isdir(path):
            return result

        for file in os.listdir(path):
            if file == "config.json":
                result["has_config_json"] = True
            elif file.startswith("configuration") and file.endswith(".py"):
                result["has_configuration_py"] = True
                result["configuration_py_files"].append(file)

        return result

    def load_config(self, model_id: str, remote_source: str = RemoteSource.huggingface) -> Optional[PretrainedConfig]:
        """
        load config
        """
        source_info = normalize_model_source(model_id, remote_source)
        model_id = source_info.model_id
        self.resolved_model_id = model_id

        if remote_source == RemoteSource.modelscope:
            from modelscope import AutoConfig
        else:
            from transformers import AutoConfig

        if remote_source == RemoteSource.modelscope and not source_info.is_local_path:
            resolved = _modelscope_snapshot_config_only(model_id)
            logger.info(
                "ModelScope Hub id %s resolved to config-only snapshot at %s",
                model_id,
                resolved,
            )
            model_id = resolved
            self.resolved_model_id = resolved

        check_model_path_res = self.check_model_path(model_id)
        if check_model_path_res["has_config_json"] and not check_model_path_res["has_configuration_py"]:
            model_id = os.path.join(
                model_id, "config.json"
            )  # When there's only one configuration file, you should pass the path to the configuration file itself.

        # First, probe whether the model is natively supported by Transformers.
        # We pass trust_remote_code=False explicitly (rather than None) so the
        # probe never enters the interactive prompt branch — it returns
        # immediately if the config requires remote code, letting the except
        # branch fall back to trust_remote_code=True.  Passing None would
        # trigger transformers' interactive y/N prompt (or signal.SIGALRM on
        # platforms that support it), which blocks headless simulation runs.
        try:
            hf_config = AutoConfig.from_pretrained(model_id, trust_remote_code=False)
            self.is_transformers_natively_supported = True
        except Exception:
            hf_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

            # TODO: Maybe add a config for user to set model_type
            is_diff, real_type = self.is_model_type_different(hf_config)
            if is_diff:
                # Using the real config class to load again
                # for example: use native deepseek_v3 to load kimi-k2`s config.json
                logger.warning("Using a model of type %s to instantiate again.", real_type)
                hf_config = AutoConfig.for_model(real_type).from_dict(hf_config.to_dict())
                self.is_transformers_natively_supported = True

        logger.info(
            "is_transformers_natively_supported = %s",
            self.is_transformers_natively_supported,
        )
        return hf_config

    def _apply_hf_config_patches(self, hf_config: PretrainedConfig, model_id: str):
        model_type = getattr(hf_config, "model_type", None)
        if model_type is None:
            return
        profile = get_model_profile(model_type)
        if profile is not None and profile.hf_config_patch_method is not None:
            try:
                profile.hf_config_patch_method(hf_config, model_id)
            except Exception as e:
                logger.warning(f"Failed to apply HF config patches for {model_type}: {e}")

    def load_model(
        self,
        hf_config: PretrainedConfig,
        dtype: torch.dtype,
        remote_source: str = RemoteSource.huggingface,
        **kwargs,
    ) -> Optional[PreTrainedModel]:
        trust_remote_code = not self.is_transformers_natively_supported
        if "trust_remote_code" in kwargs:
            trust_remote_code = kwargs.pop("trust_remote_code")

        return self.try_to_load_model(
            hf_config,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            remote_source=remote_source,
        )

    @staticmethod
    def load_quant_config(hf_config: PretrainedConfig) -> QuantizationConfigMixin:
        quant_config = AutoQuantizationConfig.from_dict(hf_config.quantization_config)
        return quant_config

    @staticmethod
    def get_modules_to_not_convert(quant_config) -> List[Optional[str]]:
        modules_to_not_convert = []
        if isinstance(quant_config, FineGrainedFP8Config):
            modules_to_not_convert = quant_config.modules_to_not_convert
        elif isinstance(quant_config, CompressedTensorsConfig):
            modules_to_not_convert = quant_config.quantization_config.ignore
        return modules_to_not_convert

    def auto_load_model_and_config(
        self, model_id: str, model_config: ModelConfig
    ) -> Tuple[PretrainedConfig, PreTrainedModel]:
        """
        Load the model and config using model_id and model_config.
        """
        hf_config = self.load_config(model_id, remote_source=model_config.remote_source)
        model_id = self.resolved_model_id or model_id
        if model_config.num_hidden_layers_override:
            hf_config.num_hidden_layers = model_config.num_hidden_layers_override

        # Apply patches for specific models before loading them
        self._apply_hf_config_patches(hf_config, model_id)

        hf_model = self.load_model(hf_config, model_config.dtype, remote_source=model_config.remote_source)
        return hf_config, hf_model

    @staticmethod
    def try_to_load_model(*args, remote_source: str = RemoteSource.huggingface, **kwarg):
        if remote_source == RemoteSource.modelscope:
            from modelscope import AutoModel
        else:
            from transformers import AutoModel
        try:
            hf_model = AutoModel.from_config(*args, **kwarg)
        except Exception:
            hf_model = AutoModelForCausalLM.from_config(*args, **kwarg)
        return hf_model
