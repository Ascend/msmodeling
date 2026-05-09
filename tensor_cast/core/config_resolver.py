# _*_coding:utf-8_*_
"""
Resolves and configures model settings for tensor cast operations.
"""

import logging

from ..core.user_config import UserInputConfig
from ..layers.attention import AttentionTensorCast
from ..layers.quant_linear import TensorCastQuantLinear
from ..model_config import (
    MlaConfig,
    ModelConfig,
    MtpConfig,
    ParallelConfig,
    QuantConfig,
)
from ..transformers.custom_model_registry import (
    get_mla_module,
    get_mla_module_name,
    get_moe_config,
    get_mtp_block_module_name,
)
from ..transformers.utils import AutoModelConfigLoader

logger = logging.getLogger(__name__)


class ConfigResolver:
    """
    Resolves and configures model settings for tensor cast operations.

    This class handles the configuration of various model components including
    parallelization, quantization, MoE (Mixture of Experts), MLA (Multihead Latent Attention),
    and MTP (Multi-Token Prediction) settings. It loads the HuggingFace model configuration
    and applies user-specified overrides.

    Attributes:
        model_id: The identifier of the model to configure.
        hf_config: The loaded HuggingFace model configuration.
        user_input: User-provided input configuration.
        model_config: The resolved model configuration containing all settings.
    """

    def __init__(
        self,
        model_id: str = "",
        user_input: UserInputConfig = None,
        parallel_config: ParallelConfig = None,
        quant_config: QuantConfig = None,
    ):
        """
        Initialize the ConfigResolver.

        Args:
            model_id: The identifier of the model to configure. If empty, will use user_input.model_id.
            user_input: User-provided input configuration. If None, parallel_config and quant_config must be provided.
            parallel_config: Parallelization configuration. Required if user_input is None.
            quant_config: Quantization configuration. Required if user_input is None.

        Raises:
            ValueError: If user_input is None and either parallel_config or quant_config is None.
        """
        self.model_id = model_id or user_input.model_id
        auto_loader = AutoModelConfigLoader()
        self.hf_config = auto_loader.load_config(
            self.model_id, remote_source=user_input.remote_source
        )
        self.user_input = user_input

        if user_input is not None:
            quant_config = user_input.get_quant_config()
            parallel_config = user_input.get_parallel_config()
        else:
            if parallel_config is None or quant_config is None:
                raise ValueError(
                    "When the user input is None,quant_config and parallel_config can not be None"
                )

        self.model_config = ModelConfig(
            parallel_config,
            quant_config,
            attention_cls=AttentionTensorCast,
            quant_linear_cls=TensorCastQuantLinear,
        )
        self.model_config.hf_config = self.hf_config
        self.model_config.trust_remote_code = (
            not auto_loader.is_transformers_natively_supported
        )

    def resolve(self) -> ModelConfig:
        """
        Resolve and apply all configuration updates.

        Updates the model configuration with user-specified settings for
        repetition, hidden layers, MoE, MLA, and MTP features.

        Returns:
            ModelConfig: The fully resolved model configuration.
        """
        self.update_hf_config(
            enable_repetition=not self.user_input.disable_repetition,
            num_hidden_layers_override=self.user_input.num_hidden_layers_override,
        )
        self.update_moe_config(
            enable_redundant_experts=self.user_input.enable_redundant_experts,
            enable_shared_expert_tp=self.user_input.enable_shared_expert_tp,
            enable_external_shared_experts=self.user_input.enable_external_shared_experts,
            host_external_shared_experts=self.user_input.host_external_shared_experts,
        )
        self.update_mla_config()
        self.update_mtp_config(num_mtp_tokens=self.user_input.num_mtp_tokens)
        self.update_parallel_config()
        self.validate_moe_parallel_config()
        # Apply remote source configuration
        self.model_config.remote_source = self.user_input.remote_source
        return self.model_config

    def update_moe_config(
        self,
        model_type: str = "",
        enable_redundant_experts: bool = False,
        enable_shared_expert_tp: bool = False,
        enable_external_shared_experts: bool = False,
        host_external_shared_experts: bool = False,
    ):
        """Update the Mixture of Experts (MoE) configuration.

        Args:
            model_type: The type of the model. If empty, uses the loaded model's type.
            enable_redundant_experts: Pad routing-expert count to a multiple of EP
                size so that every rank hosts the same number of experts.
            enable_shared_expert_tp: Apply tensor-parallelism to shared experts
                across the EP group.  Requires ``expert_parallel_size > 1``.
                Mutually exclusive with ``host_external_shared_experts``.
            enable_external_shared_experts: Allocate dedicated ranks within the EP
                group to run shared experts, separating them from routing experts.
            host_external_shared_experts: Place external shared experts on the host
                (CPU) side.  Mutually exclusive with ``enable_shared_expert_tp``.

        Legal combinations (when both relate to shared experts)::

            enable_shared_expert_tp  | host_external_shared_experts | OK?
            -------------------------+------------------------------+-----
            False                    | False                        | Yes
            True                     | False                        | Yes (needs EP > 1)
            False                    | True                         | Yes
            True                     | True                         | NO — ValueError
        """
        if not model_type:
            model_type = self.hf_config.model_type
        moe_config = get_moe_config(model_type)
        if moe_config is not None:
            moe_config.enable_redundant_experts = enable_redundant_experts
            moe_config.enable_shared_expert_tp = enable_shared_expert_tp
            moe_config.enable_external_shared_experts = enable_external_shared_experts
            moe_config.host_external_shared_experts = host_external_shared_experts
        self.model_config.moe_config = moe_config

    def update_mla_config(self, model_type: str = ""):
        """
        Update the Multihead Latent Attention (MLA) configuration.

        Args:
            model_type: The type of the model. If empty, uses the loaded model's type.
        """
        if not model_type:
            model_type = self.hf_config.model_type
        mla_module_name = get_mla_module_name(model_type)
        if mla_module_name is not None:
            mla_config = MlaConfig(
                module_name=mla_module_name,
                mla_cls=get_mla_module(model_type),
            )
            self.model_config.mla_config = mla_config

    def update_mtp_config(self, model_type: str = "", num_mtp_tokens: int = 0):
        """
        Update the Multi-Token Prediction (MTP) configuration.

        Args:
            model_type: The type of the model. If empty, uses the loaded model's type.
            num_mtp_tokens: Number of MTP tokens to enable.
        """
        if not model_type:
            model_type = self.hf_config.model_type
        mtp_block_module_name = get_mtp_block_module_name(model_type)
        if num_mtp_tokens > 0:
            mtp_config = MtpConfig(
                num_mtp_layers=num_mtp_tokens,
                mtp_block_module_name=mtp_block_module_name,
            )
            self.model_config.mtp_config = mtp_config

    def update_hf_config(
        self, enable_repetition: bool = False, num_hidden_layers_override: int = 0
    ):
        """
        Update the HuggingFace configuration settings.

        Args:
            enable_repetition: Whether to enable repetition in the model.
            num_hidden_layers_override: Override the number of hidden layers.
        """
        self.model_config.enable_repetition = enable_repetition
        self.model_config.num_hidden_layers_override = num_hidden_layers_override
        if hasattr(self.hf_config, "vision_config"):
            # This update of the dtype in the configuration ensures that the visual and language modules
            # use the same dtype during both initialization and execution,
            # thereby avoiding dtype mismatches in subsequent execution paths.
            dtype = self.model_config.dtype
            for sub_config_key in self.hf_config.sub_configs:
                sub_config = getattr(self.hf_config, sub_config_key)
                sub_config.dtype = dtype

    def update_parallel_config(self):
        if self.model_config.moe_config is None:
            self.model_config.parallel_config.expert_parallel_size = 1
            self.model_config.parallel_config.moe_tensor_parallel_size = 1
            self.model_config.parallel_config.moe_data_parallel_size = (
                self.model_config.parallel_config.world_size
            )

    def validate_moe_parallel_config(self):
        """Validate MoE-related parallel configuration constraints.

        Raises:
            ValueError: If enable_shared_expert_tp is True but EP size <= 1.
            ValueError: If enable_shared_expert_tp and host_external_shared_experts
                are both True (mutually exclusive).
        """
        moe_config = self.model_config.moe_config
        if moe_config is None:
            return

        expert_parallel_size = self.model_config.parallel_config.expert_parallel_size
        if moe_config.enable_shared_expert_tp and expert_parallel_size <= 1:
            raise ValueError(
                "When enable_shared_expert_tp=True, expert_parallel_size must be "
                "greater than 1. Either set enable_shared_expert_tp=False or "
                "set expert_parallel_size > 1."
            )

        if (
            moe_config.enable_shared_expert_tp
            and moe_config.host_external_shared_experts
        ):
            raise ValueError(
                "enable_shared_expert_tp and host_external_shared_experts are "
                "mutually exclusive. enable_shared_expert_tp applies TP to shared "
                "experts across the EP group, while host_external_shared_experts "
                "dedicates separate ranks to run shared experts. Set at most one."
            )
