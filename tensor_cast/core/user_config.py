# _*_coding:utf-8_*_
"""
user_config
"""

import logging
import math
from dataclasses import dataclass, field, fields
from typing import List, Optional, Union

from ..core.input_generator import RequestInfo
from ..core.quantization.config import create_quant_config
from ..core.quantization.datatypes import QuantizeAttentionAction, QuantizeLinearAction
from ..device import DeviceProfile
from ..model_config import (
    ParallelConfig,
    QuantConfig,
    RemoteSource,
    WordEmbeddingTPMode,
)


logger = logging.getLogger(__name__)


@dataclass
class UserInputConfig:
    device: str = "TEST_DEVICE"
    model_id: str = ""
    num_queries: int = 0
    query_len: int = 0
    context_length: int = 0
    prefix_cache_hit_rate: float = 0.0
    do_compile: bool = False
    allow_graph_break: bool = False
    enable_multistream: bool = True
    dump_input_shapes: bool = False
    dump_op_bound_results: bool = False
    chrome_trace: Optional[str] = None
    graph_log_url: Optional[str] = None
    log_level: Optional[str] = None
    quantize_linear_action: QuantizeLinearAction = QuantizeLinearAction.W8A8_DYNAMIC
    quantize_non_expert_linear_action: QuantizeLinearAction = QuantizeLinearAction.DISABLED
    quantize_lmhead: bool = False
    mxfp4_group_size: int = 32
    quantize_attention_action: QuantizeAttentionAction = QuantizeAttentionAction.DISABLED
    enable_sequence_parallel: bool = False
    decode: bool = False
    num_mtp_tokens: int = 0
    mtp_acceptance_rate: List[float] = field(default_factory=lambda: [0.9, 0.6, 0.4, 0.2])
    num_hidden_layers_override: int = 0
    disable_repetition: bool = False
    reserved_memory_gb: float = 0
    world_size: int = 1
    tp_size: int = 1
    pp_size: int = 1
    dp_size: Optional[int] = None
    o_proj_tp_size: Optional[int] = None
    o_proj_dp_size: Optional[int] = None
    mlp_tp_size: Optional[int] = None
    mlp_dp_size: Optional[int] = None
    lmhead_tp_size: Optional[int] = None
    lmhead_dp_size: Optional[int] = None
    ep_size: int = 1
    moe_dp_size: int = 1
    moe_tp_size: Optional[int] = None
    word_embedding_tp: Optional[WordEmbeddingTPMode] = None
    enable_redundant_experts: bool = False
    """Pad routing-expert count to a multiple of EP size for load balancing."""
    enable_shared_expert_tp: bool = False
    enable_dispatch_ffn_combine: bool = False
    """Apply tensor-parallelism to shared experts across the EP group.
    Requires expert_parallel_size > 1.
    Mutually exclusive with ``host_external_shared_experts``.
    """
    enable_external_shared_experts: bool = False
    """Allocate dedicated ranks within the EP group to run shared experts."""
    host_external_shared_experts: bool = False
    """Place external shared experts on the host (CPU) side instead of device.
    Mutually exclusive with ``enable_shared_expert_tp``.
    """
    vision_tp_size: int = 1
    block_size: int = 128
    remote_source: str = RemoteSource.huggingface
    image_batch_size: Optional[int] = None
    image_height: Optional[int] = None
    image_width: Optional[int] = None
    performance_model: Union[str, List[str]] = "analytic"
    """Performance model type(s): 'analytic' | 'profiling'.
    Can be a single string or a list of strings to run multiple models.
    """
    profiling_database: Optional[str] = None
    """Path to the performance database directory (required for 'profiling' mode)."""

    def __post_init__(self):
        self._validate_device()
        self._validate_vision_parallelism()
        self._normalize_performance_model()
        self._normalize_word_embedding_tp()

    def _normalize_performance_model(self):
        """Normalize performance_model to a list of model type strings."""
        pm = self.performance_model
        if isinstance(pm, str):
            self.performance_model = [pm]

    def _validate_device(self):
        if self.device not in DeviceProfile.all_device_profiles:
            raise ValueError(f"Device '{self.device}' not recognized.")

    def _validate_vision_parallelism(self):
        if self.vision_tp_size < 1:
            raise ValueError(f"vision_tp_size must be at least 1, got {self.vision_tp_size}")
        if self.vision_tp_size > self.world_size:
            raise ValueError(
                f"vision_tp_size ({self.vision_tp_size}) must not exceed num_devices/world_size ({self.world_size})"
            )
        if self.world_size % self.vision_tp_size != 0:
            raise ValueError(
                f"num_devices/world_size ({self.world_size}) must be divisible by vision_tp_size "
                f"({self.vision_tp_size})"
            )

    def _normalize_word_embedding_tp(self):
        if self.word_embedding_tp is None or self.word_embedding_tp == "":
            self.word_embedding_tp = None
            return
        if isinstance(self.word_embedding_tp, bool):
            self.word_embedding_tp = WordEmbeddingTPMode.col if self.word_embedding_tp else None
            return
        try:
            self.word_embedding_tp = WordEmbeddingTPMode(self.word_embedding_tp)
        except ValueError as err:
            raise ValueError(
                f"word_embedding_tp must be one of {{'col', 'row'}} or None, got {self.word_embedding_tp!r}."
            ) from err

    def _print_info(self):
        print("--- Configuration ---")
        print(f"Device: {self.device}")
        print(f"Model ID: {self.model_id}")
        print(f"Number of Queries: {self.num_queries}")
        print(f"Input Length (per query): {self.query_len}")
        print(f"Context Length (per query): {self.context_length}")
        print(f"Is Decode: {self.decode}")
        print(f"Enable repetition: {not self.disable_repetition}")
        if self.num_mtp_tokens > 0:
            print(f"Number of MTP layers: {self.num_mtp_tokens}")
        if self.quantize_linear_action != QuantizeLinearAction.DISABLED:
            print(f"Quantization Linear: {self.quantize_linear_action}, quantize LM Head: {self.quantize_lmhead}")
            if self.quantize_linear_action == QuantizeLinearAction.MXFP4:
                print(f"  MXFP4 group size: {self.mxfp4_group_size}")
        else:
            print("Quantization Linear: Disabled")
        if self.quantize_non_expert_linear_action != QuantizeLinearAction.DISABLED:
            print(f"Quantization Non-Expert Linear (override): {self.quantize_non_expert_linear_action}")
        if self.quantize_attention_action != QuantizeAttentionAction.DISABLED:
            print(f"Quantization Attention: {self.quantize_attention_action}")
        else:
            print("Quantization Attention: Disabled")
        print(f"Use torch.compile: {self.do_compile}")
        if self.do_compile:
            print(f"  allow graph break: {self.allow_graph_break}")
            print(f"  enable multistream: {self.enable_multistream}")
        print(f"Group table averages by input shapes: {self.dump_input_shapes}")
        print(f"Dump operator bound ratios: {self.dump_op_bound_results}")
        if self.chrome_trace:
            print(f"Chrome trace output file: {self.chrome_trace}")
        if self.image_batch_size:
            print(f"image_batch_size: {self.image_batch_size}")
            print(f"image_height: {self.image_height}")
            print(f"image_width: {self.image_width}")
            print(f"vision_tp_size: {self.vision_tp_size}")
        print("---------------------\n")

    def get_parallel_config(self) -> ParallelConfig:
        return ParallelConfig(
            world_size=self.world_size,
            tensor_parallel_size=self.tp_size,
            data_parallel_size=self.dp_size,
            o_proj_tensor_parallel_size=self.o_proj_tp_size,
            o_proj_data_parallel_size=self.o_proj_dp_size,
            mlp_tensor_parallel_size=self.mlp_tp_size,
            mlp_data_parallel_size=self.mlp_dp_size,
            lmhead_tensor_parallel_size=self.lmhead_tp_size,
            lmhead_data_parallel_size=self.lmhead_dp_size,
            expert_parallel_size=self.ep_size,
            moe_tensor_parallel_size=self.moe_tp_size,
            moe_data_parallel_size=self.moe_dp_size,
            embedding_parallel=self.word_embedding_tp,
            vision_tensor_parallel_size=self.vision_tp_size,
            pipeline_parallel_size=self.pp_size,
        )

    def get_quant_config(self) -> QuantConfig:
        if (
            self.quantize_linear_action == QuantizeLinearAction.DISABLED
            and self.quantize_non_expert_linear_action == QuantizeLinearAction.DISABLED
            and self.quantize_attention_action == QuantizeAttentionAction.DISABLED
        ):
            return QuantConfig()
        extra_kwargs = {}
        linear_actions = [
            self.quantize_linear_action,
            self.quantize_non_expert_linear_action,
        ]
        if QuantizeLinearAction.MXFP4 in linear_actions:
            from ..quantize_utils import QuantGranularity

            extra_kwargs.update(
                weight_group_size=self.mxfp4_group_size,
                weight_quant_granularity=QuantGranularity.PER_GROUP,
            )
        return create_quant_config(
            self.quantize_linear_action,
            quantize_non_expert_linear_action=self.quantize_non_expert_linear_action,
            quantize_lmhead=self.quantize_lmhead,
            quantize_attention_action=self.quantize_attention_action,
            **extra_kwargs,
        )

    def get_request_info(self) -> RequestInfo:
        effective_hit_rate = self.get_effective_prefix_cache_hit_rate()
        cached_prefix_tokens = math.floor(self.query_len * effective_hit_rate)
        effective_query_len = self.query_len - cached_prefix_tokens
        if effective_query_len < 1:
            raise ValueError(
                "Effective query length must be at least 1 after applying prefix cache hit rate. "
                f"Got query_len={self.query_len}, prefix_cache_hit_rate={self.prefix_cache_hit_rate}."
            )
        effective_context_len = self.context_length + cached_prefix_tokens
        return RequestInfo(
            query_len=effective_query_len,
            seq_len=effective_context_len + effective_query_len,
            concurrency=self.num_queries,
            is_decode=self.decode,
            image_batch_size=self.image_batch_size,
            image_height=self.image_height,
            image_width=self.image_width,
            context_length=self.context_length,
        )

    def get_effective_prefix_cache_hit_rate(self, is_decode: Optional[bool] = None):
        if is_decode is None:
            is_decode = self.decode
        if is_decode and self.prefix_cache_hit_rate > 0:
            logger.warning(
                "Ignoring prefix_cache_hit_rate=%.4f in decode mode.",
                self.prefix_cache_hit_rate,
            )
            return 0.0
        return self.prefix_cache_hit_rate

    @classmethod
    def from_args(cls, args) -> "UserInputConfig":
        # get all names of cls
        field_names = {_field.name for _field in fields(cls)}
        logger.debug(
            "Initializing %s from command-line arguments. Class has %d defined fields: %s",
            cls.__name__,
            len(field_names),
            sorted(field_names),
        )

        # Extract only the fields that exist in the cls from args.

        # Handle the special case where the input arguments differ
        # from the command-line arguments by implementing backward compatibility first.
        # input_key:target_key
        special_input_key_map = {
            "compile": "do_compile",
            "compile_allow_graph_break": "allow_graph_break",
            "query_length": "query_len",
            "num_devices": "world_size",
        }
        logger.debug(
            "Using special input key mapping for backward compatibility: %s",
            special_input_key_map,
        )
        filtered_kwargs = {}
        for field_name, field_value in vars(args).items():
            if field_name in special_input_key_map:
                filtered_kwargs[special_input_key_map[field_name]] = field_value
            elif field_name in field_names:
                filtered_kwargs[field_name] = field_value
        return cls(**filtered_kwargs)
