import dataclasses

try:
    # Native in Python 3.11+
    from enum import StrEnum
except ImportError:
    # Fallback for Python 3.10
    from strenum import StrEnum
from typing import Dict, List, Optional, Type, Union

import torch
from transformers import PretrainedConfig
from transformers.utils.quantization_config import QuantizationConfigMixin

from .quantize_utils import (
    AttentionQuantType,
    get_torch_dtype_from_quant_type,
    LinearQuantType,
    QuantGranularity,
    QuantScheme,
)
from .utils import get_modules_to_not_convert


@dataclasses.dataclass
class LinearQuantConfig:
    """
    Quantization configuration for PyTorch Linear op.

    The shape of the scale/offset decides the granularity, i.e.
    per-tensor, per-channel or per-group.

    For symmetric quantization, the offset tensor is None.

    For dynamic quantization, the activation scale/offset is None.
    """

    # weight config
    # The scale and offset are computed according to `weight_quant_granularity`
    # and `weight_quant_scheme` if they are not explicitly provided.
    weight_scale: Optional[torch.Tensor] = None
    weight_offset: Optional[torch.Tensor] = None
    weight_transposed: bool = False
    """
    Weight's shape is always in (N, K) following PyTorch's Linear semantics.
    This field sets the contiguous layout: True: (N, K) contiguous, False: (K, N) contiguous
    """
    weight_int4_pack_dim: int = 1
    """The dim to pack two int4 into an int8"""
    weight_group_size: Optional[int] = None
    """Group size for weight quantization along k-dim. For MXFP4, it also implies
    the channel group size for activation quantization. The shape of weight_scale
    should be aligned with this setting."""
    weight_quant_granularity: Optional[QuantGranularity] = None
    """Quantization granularity for weight. If None, it is inferred from the shape
    of weight_scale and weight_offset."""
    weight_quant_scheme: QuantScheme = QuantScheme.SYMMETRIC
    """Quantization scheme for weight. If None, it is inferred from whether
    weight_offset is None or not."""

    quant_type: LinearQuantType = LinearQuantType.W8A16

    # activation config
    dynamic_quant_granularity: Optional[QuantGranularity] = None
    dynamic_quant_scheme: QuantScheme = QuantScheme.SYMMETRIC
    activation_scale: Optional[torch.Tensor] = None
    """Scale for static quantization, None for dynamic quantization"""
    activation_offset: Optional[torch.Tensor] = None
    """Offset for static quantization, None for symmetric quantization or dynamic quantization"""

    # output config
    out_dtype: Optional[torch.dtype] = None
    """We deliberately not support output scales and only support out_dtype as
    high-precision dtype (fp16, bf16, fp32) for simplicity. Use-case for quantized
    output is not common."""

    def __post_init__(self):
        if (
            self.weight_quant_granularity is not None
            and self.dynamic_quant_granularity is None
        ):
            self.dynamic_quant_granularity = self.weight_quant_granularity
        if self.weight_scale is None and self.weight_offset is not None:
            raise ValueError(
                "weight_offset is provided but weight_scale is None, which is invalid"
            )
        if self.weight_scale is None:
            if self.weight_quant_granularity is None:
                self.weight_quant_granularity = QuantGranularity.PER_TENSOR
            if self.weight_quant_scheme is None:
                self.weight_quant_scheme = QuantScheme.SYMMETRIC
        if (
            self.weight_scale is None
            and self.weight_quant_granularity == QuantGranularity.PER_GROUP
            and self.weight_group_size is None
        ):
            raise ValueError(
                "weight_group_size must be provided when weight_quant_granularity is PER_GROUP and "
                "weight_scale is not provided"
            )
        if self.activation_scale is None:
            if self.dynamic_quant_granularity is None:
                self.dynamic_quant_granularity = QuantGranularity.PER_TENSOR
            if self.dynamic_quant_scheme is None:
                self.dynamic_quant_scheme = QuantScheme.SYMMETRIC
        # Validate FP8 configuration
        if self.quant_type == LinearQuantType.FP8:
            if (
                self.dynamic_quant_scheme is not None
                and self.dynamic_quant_scheme != QuantScheme.SYMMETRIC
            ):
                raise ValueError(
                    "FP8 quantization only supports symmetric scheme for activations"
                )
            if self.activation_scale is not None or self.activation_offset is not None:
                raise ValueError(
                    "FP8 quantization does not support static activation quantization"
                )

        # Validate MXFP4 configuration
        if self.quant_type == LinearQuantType.MXFP4:
            if self.dynamic_quant_granularity != QuantGranularity.PER_GROUP:
                raise ValueError(
                    "MXFP4 quantization only supports PER_GROUP granularity"
                )
            if (
                self.dynamic_quant_scheme is not None
                and self.dynamic_quant_scheme != QuantScheme.SYMMETRIC
            ):
                raise ValueError("MXFP4 quantization only supports symmetric scheme")
            if self.activation_scale is not None or self.activation_offset is not None:
                raise ValueError(
                    "MXFP4 quantization does not support static activation quantization"
                )


@dataclasses.dataclass
class AttentionQuantConfig:
    """
    Quantization configuration for an attention layer, specifying
    how KV cache is quantized and how the intermediate activation
    tensors are quantized and computed for attention scoring, normalization
    and aggregation.

    For a normal attention implementation, we would have something like below,
    where Q and KV cache are quantized and quant dtype of Q and attention prob
    is aligned with that of KV:

    `out = dequant(quant(softmax(dequant(Q @ K^T)), attention_prob_scale/offset) @ V)`

    TODO: support dynamic quant of query, kv, attention_prob?
    TODO: support different quant dtype of Q and attention_prob from KV
    TODO: support int4 quant
    """

    quant_type: AttentionQuantType = AttentionQuantType.INT8
    kv_scale: Optional[torch.Tensor] = None
    query_scale: Optional[torch.Tensor] = None
    attention_prob_scale: Optional[torch.Tensor] = None

    query_offset: Optional[torch.Tensor] = None
    kv_offset: Optional[torch.Tensor] = None
    attention_prob_offset: Optional[torch.Tensor] = None

    def get_quant_dtype(self) -> torch.dtype:
        return get_torch_dtype_from_quant_type(self.quant_type)


@dataclasses.dataclass
class MultiheadLatentAttentionQuantConfig(AttentionQuantConfig):
    """
    Quantization configuration for multihead latent attention (MLA) layer.

    Similar to `AttentionQuantConfig`, but with additional quant params
    for the kv projection.

    Check `tensor_cast.multihead_latent_attention_quant` op for more details.
    """

    kv_projected_scale: Optional[torch.Tensor] = None
    kv_projected_offset: Optional[torch.Tensor] = None
    qk_scale: Optional[torch.Tensor] = None
    qk_offset: Optional[torch.Tensor] = None
    v_scale: Optional[torch.Tensor] = None
    v_offset: Optional[torch.Tensor] = None
    out_scale: Optional[torch.Tensor] = None
    out_offset: Optional[torch.Tensor] = None


# TODO: Implement a quantizer that is compatible with
#  both open-source ecosystems and Ascend, referencing the quantizer in Transformers.


@dataclasses.dataclass
class QuantConfig:
    linear_configs: Dict[str, LinearQuantConfig] = dataclasses.field(
        default_factory=dict
    )
    """Per-layer configs: full module path -> LinearQuantConfig"""

    attention_configs: Dict[int, AttentionQuantConfig] = dataclasses.field(
        default_factory=dict
    )
    """Per-layer configs: attn_layer_id -> AttentionQuantConfig"""

    modules_to_not_convert: Optional[List[str]] = None

    # TODO: our quant config should be compatible with multiple scenarios.
    #  Get some attr from the quant config instance from config.json
    ori_quant_config: QuantizationConfigMixin = None

    def __post_init__(self):
        if self.modules_to_not_convert is None:
            self.modules_to_not_convert = ["lm_head"]

    def update_modules_to_not_convert(self, quant_config: QuantizationConfigMixin):
        self.modules_to_not_convert = get_modules_to_not_convert(quant_config)


class WordEmbeddingTPMode(StrEnum):
    col = "col"
    row = "row"


@dataclasses.dataclass
class ParallelConfig:
    world_size: int = 1
    rank: int = -1
    tensor_parallel_size: int = 1
    data_parallel_size: Optional[int] = None
    pipeline_parallel_size: int = 1
    o_proj_tensor_parallel_size: Optional[int] = None
    o_proj_data_parallel_size: Optional[int] = None
    mlp_tensor_parallel_size: Optional[int] = None
    mlp_data_parallel_size: Optional[int] = None
    lmhead_tensor_parallel_size: Optional[int] = None
    lmhead_data_parallel_size: Optional[int] = None
    embedding_parallel: bool = False
    embedding_parallel_mode: WordEmbeddingTPMode = WordEmbeddingTPMode.col
    expert_parallel_size: int = 1
    moe_tensor_parallel_size: Optional[int] = None
    moe_data_parallel_size: int = 1
    ulysses_size: int = 1

    def has_attn_tp(self) -> bool:
        return self.tensor_parallel_size > 1

    def has_o_proj_tp(self) -> bool:
        return self.o_proj_tensor_parallel_size > 1

    def has_mlp_tp(self) -> bool:
        return self.mlp_tensor_parallel_size > 1

    def has_lmhead_tp(self) -> bool:
        return self.lmhead_tensor_parallel_size > 1

    def has_ep(self) -> bool:
        return self.expert_parallel_size > 1

    def __post_init__(self) -> None:
        if self.data_parallel_size is None:
            self.data_parallel_size = (
                self.world_size
                // self.tensor_parallel_size
                // self.pipeline_parallel_size
            )

        if (
            self.tensor_parallel_size
            * self.data_parallel_size
            * self.pipeline_parallel_size
            != self.world_size
        ):
            raise ValueError(
                f"tensor_parallel_size ({self.tensor_parallel_size}) * "
                f"data_parallel_size ({self.data_parallel_size}) * "
                f"pipeline_parallel_size ({self.pipeline_parallel_size}) "
                f"must equal world_size ({self.world_size})"
            )

        if self.moe_tensor_parallel_size is None:
            self.moe_tensor_parallel_size = (
                self.world_size
                // self.moe_data_parallel_size
                // self.expert_parallel_size
            )
        if (
            self.moe_data_parallel_size
            * self.moe_tensor_parallel_size
            * self.expert_parallel_size
            != self.world_size
        ):
            raise ValueError(
                f"moe_tensor_parallel_size ({self.moe_tensor_parallel_size}) * "
                f"moe_data_parallel_size ({self.moe_data_parallel_size}) * "
                f"expert_parallel_size ({self.expert_parallel_size}) "
                f"must equal to world_size ({self.world_size})"
            )
        if self.o_proj_tensor_parallel_size is None:
            self.o_proj_tensor_parallel_size = self.tensor_parallel_size
        if self.o_proj_data_parallel_size is None:
            self.o_proj_data_parallel_size = (
                self.world_size
                // self.o_proj_tensor_parallel_size
                // self.pipeline_parallel_size
            )
        if (
            self.o_proj_tensor_parallel_size
            * self.o_proj_data_parallel_size
            * self.pipeline_parallel_size
            != self.world_size
        ):
            raise ValueError(
                f"o_proj_tensor_parallel_size ({self.o_proj_tensor_parallel_size}) * "
                f"o_proj_data_parallel_size ({self.o_proj_data_parallel_size}) * "
                f"pipeline_parallel_size ({self.pipeline_parallel_size}) "
                f"must equal world_size ({self.world_size})"
            )

        if self.mlp_tensor_parallel_size is None:
            self.mlp_tensor_parallel_size = self.tensor_parallel_size
        if self.mlp_data_parallel_size is None:
            self.mlp_data_parallel_size = (
                self.world_size
                // self.mlp_tensor_parallel_size
                // self.pipeline_parallel_size
            )
        if (
            self.mlp_tensor_parallel_size
            * self.mlp_data_parallel_size
            * self.pipeline_parallel_size
            != self.world_size
        ):
            raise ValueError(
                f"mlp_tensor_parallel_size ({self.mlp_tensor_parallel_size}) * "
                f"mlp_data_parallel_size ({self.mlp_data_parallel_size}) * "
                f"pipeline_parallel_size ({self.pipeline_parallel_size}) "
                f"must equal world_size ({self.world_size})"
            )

        if self.lmhead_tensor_parallel_size is None:
            self.lmhead_tensor_parallel_size = self.tensor_parallel_size
        if self.lmhead_data_parallel_size is None:
            self.lmhead_data_parallel_size = (
                self.world_size
                // self.lmhead_tensor_parallel_size
                // self.pipeline_parallel_size
            )
        if (
            self.lmhead_tensor_parallel_size
            * self.lmhead_data_parallel_size
            * self.pipeline_parallel_size
            != self.world_size
        ):
            raise ValueError(
                f"lmhead_tensor_parallel_size ({self.lmhead_tensor_parallel_size}) * "
                f"lmhead_data_parallel_size ({self.lmhead_data_parallel_size}) * "
                f"pipeline_parallel_size ({self.pipeline_parallel_size}) "
                f"must equal world_size ({self.world_size})"
            )


@dataclasses.dataclass(frozen=True)
class MoEFieldNames:
    gate: str = "gate"
    experts: str = "experts"
    shared_experts: Optional[str] = "shared_experts"
    shared_experts_gate: Optional[str] = "shared_experts_gate"
    top_k: Optional[str] = "top_k"
    norm_topk_prob: Optional[str] = "norm_topk_prob"


@dataclasses.dataclass
class MoEConfig:
    module_name: str
    fused_moe_cls: Optional[Type["FusedMoEBase"]] = None  # noqa: F821
    field_names: MoEFieldNames = MoEFieldNames()
    gate_returns_raw_logits: bool = False
    """whether the gate module returns raw logits or (topk_indices, topk_weights) tuple"""
    # TODO: add expert-parallel configuration
    enable_redundant_experts: bool = False
    enable_shared_expert_tp: bool = False
    enable_external_shared_experts: bool = False
    host_external_shared_experts: bool = False
    num_experts_key: Union[str, List[str]] = "num_experts"


@dataclasses.dataclass(frozen=True)
class MlaFieldNames:
    config: str = "config"
    layer_idx: str = "layer_idx"
    num_heads: str = "num_heads"
    q_lora_rank: str = "q_lora_rank"
    qk_nope_head_dim: str = "qk_nope_head_dim"
    qk_rope_head_dim: str = "qk_rope_head_dim"
    qk_head_dim: str = "qk_head_dim"
    kv_lora_rank: str = "kv_lora_rank"
    v_head_dim: str = "v_head_dim"
    q_proj: Optional[str] = "q_proj"
    q_a_proj: Optional[str] = "q_a_proj"
    q_b_proj: Optional[str] = "q_b_proj"
    kv_a_proj_with_mqa: str = "kv_a_proj_with_mqa"
    kv_b_proj: str = "kv_b_proj"
    o_proj: str = "o_proj"
    q_a_layernorm: Optional[str] = "q_a_layernorm"
    kv_a_layernorm: str = "kv_a_layernorm"

    def __post_init__(self):
        if self.q_proj is None and (
            self.q_a_proj is None or self.q_b_proj is None or self.q_a_layernorm is None
        ):
            raise ValueError(
                "Either q_proj or all of q_a_proj/q_b_proj/q_a_layernorm must be specified"
            )


@dataclasses.dataclass
class MlaConfig:
    module_name: str
    mla_cls: Optional[Type["MultiheadLatentAttentionBase"]] = None  # noqa: F821
    field_names: MlaFieldNames = MlaFieldNames()


@dataclasses.dataclass
class MtpConfig:
    num_mtp_layers: int
    # None for auto mode, we would use the last decoder layer
    # class name as the mtp block module name
    mtp_block_module_name: Optional[str] = None


class RemoteSource(StrEnum):
    huggingface = "huggingface"
    modelscope = "modelscope"


@dataclasses.dataclass
class ModelConfig:
    parallel_config: ParallelConfig
    quant_config: QuantConfig
    dtype: torch.dtype = torch.half
    cache_rotary_embedding: bool = True
    moe_config: Optional[MoEConfig] = None
    mla_config: Optional[MlaConfig] = None
    mtp_config: Optional[MtpConfig] = None
    attention_cls: Optional[Type["AttentionBase"]] = None  # noqa: F821
    quant_linear_cls: Optional[Type["QuantLinearBase"]] = None  # noqa: F821
    hf_config: Optional[PretrainedConfig] = None
    trust_remote_code: bool = True
    remote_source: str = RemoteSource.huggingface

    num_hidden_layers_override: int = 0
    """Override hf_config.num_hidden_layers, useful for speeding up sanity tests
    with small overrides for very large models."""
    enable_repetition: bool = False
    """Transformer models have repetitive patterns. This configuration flag tells TensorCast
    whether to automatically detect and leverage the repetition patterns to reduce the
    performance estimation cost. This is especially helpful for large models."""

    def __post_init__(self):
        # TODO: Use Pydantic to add data validation.
        if self.num_hidden_layers_override < 0:
            self.num_hidden_layers_override = 0


@dataclasses.dataclass
class DiffusersConfig:
    model_path: Optional[str] = None
    text_config: Optional[Type["DiffusersTextConfig"]] = None
    transformer_config: Optional[Type["DiffusersTransformerConfig"]] = None
    vae_config: Optional[Type["DiffusersVaeConfig"]] = None


@dataclasses.dataclass
class DiffusersTextConfig:
    parallel_config: ParallelConfig
    quant_config: QuantConfig
    dtype: torch.dtype = torch.float16
    config_json: Optional[str] = None
    model_config: Optional[dict] = None


@dataclasses.dataclass
class DiffusersTransformerConfig:
    parallel_config: ParallelConfig
    quant_config: QuantConfig
    dtype: torch.dtype = torch.float16
    config_json: Optional[str] = None
    model_config: Optional[dict] = None
    attention_cls: Optional[Type["AttentionBase"]] = None  # noqa: F821
    quant_linear_cls: Optional[Type["QuantLinearBase"]] = None  # noqa: F821


@dataclasses.dataclass
class DiffusersVaeConfig:
    parallel_config: ParallelConfig
    quant_config: QuantConfig
    dtype: torch.dtype = torch.float16
    config_json: Optional[str] = None
    model_config: Optional[dict] = None
