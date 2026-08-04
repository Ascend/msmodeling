# Copyright (c) 2026-2026 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Build constrained Theory evaluation environments from ModelRunContext."""

from __future__ import annotations

import math
from typing import Mapping

from tools.model_diagnostics.domain.models import ExecutionPhase, ModelRunContext
from tools.model_diagnostics.specification.errors import SpecificationLoadError
from tools.model_diagnostics.specification.mtp_window import (
    parse_num_mtp_tokens,
    validate_mtp_decode_window,
)


def _require_positive(context: ModelRunContext, field_name: str) -> int:
    value = getattr(context, field_name)
    if value is None:
        raise SpecificationLoadError(f"ModelRunContext.{field_name} is required for Theory evaluation")
    return value


def _config_int(config: Mapping[str, object], key: str) -> int:
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SpecificationLoadError(f"model_config.{key} must be a positive integer")
    return value


def _optional_config_int(config: Mapping[str, object], key: str, *, default: int) -> int:
    value = config.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise SpecificationLoadError(f"model_config.{key} must be an integer")
    if value <= 0:
        raise SpecificationLoadError(f"model_config.{key} must be positive")
    return value


def _dtype_binding(
    config: Mapping[str, object],
    key: str,
    *,
    default: str,
) -> str:
    value = config.get(key, default)
    if not isinstance(value, str) or not value.strip():
        raise SpecificationLoadError(f"quantization_config.{key} must be a non-empty string")
    return value


def build_theory_env(context: ModelRunContext) -> dict[str, object]:
    """Derive B/Q/S/T/H/TP and related bindings for shape/dtype expressions."""

    batch = _require_positive(context, "batch_size")
    query = _require_positive(context, "query_length")
    context_length = context.context_length if context.context_length is not None else 0
    if isinstance(context_length, bool) or not isinstance(context_length, int) or context_length < 0:
        raise SpecificationLoadError("context_length must be a non-negative integer")

    config = context.model_config
    hidden = _config_int(config, "hidden_size")
    vocab = _config_int(config, "vocab_size")
    intermediate = _config_int(config, "intermediate_size")
    num_heads = _config_int(config, "num_attention_heads")
    num_kv_heads = _config_int(config, "num_key_value_heads")
    head_dim = config.get("head_dim")
    if head_dim is None:
        if hidden % num_heads != 0:
            raise SpecificationLoadError("hidden_size must be divisible by num_attention_heads")
        head_dim = hidden // num_heads
    elif isinstance(head_dim, bool) or not isinstance(head_dim, int) or head_dim <= 0:
        raise SpecificationLoadError("model_config.head_dim must be a positive integer")

    tp = context.parallel.tensor_parallel_size
    dp = context.parallel.data_parallel_size
    ep = context.parallel.expert_parallel_size
    mlp_tp = _optional_config_int(config, "mlp_tp_size", default=tp)
    o_proj_tp = _optional_config_int(config, "o_proj_tp_size", default=tp)
    lmhead_tp = _optional_config_int(config, "lmhead_tp_size", default=1)

    dtype = config.get("torch_dtype") or config.get("dtype") or "float16"
    if not isinstance(dtype, str) or not dtype.strip():
        raise SpecificationLoadError("model dtype must be a non-empty string")
    quantization = context.quantization_config
    activation_dtype = _dtype_binding(
        quantization,
        "activation_dtype",
        default=dtype,
    )
    linear_input_dtype = _dtype_binding(
        quantization,
        "linear_input_dtype",
        default=activation_dtype,
    )
    weight_dtype = _dtype_binding(
        quantization,
        "weight_dtype",
        default=dtype,
    )
    scale_dtype = _dtype_binding(
        quantization,
        "scale_dtype",
        default="float32",
    )
    accumulation_dtype = _dtype_binding(
        quantization,
        "accumulation_dtype",
        default=dtype,
    )
    output_dtype = _dtype_binding(
        quantization,
        "output_dtype",
        default=activation_dtype,
    )

    # Local rank batch after DP split, matching the alignment table B=ceil(num_queries/DP).
    local_batch = int(math.ceil(batch / dp))
    seq = context_length + query
    tokens = local_batch * query
    local_heads = num_heads // tp
    if num_heads % tp != 0:
        raise SpecificationLoadError("num_attention_heads must be divisible by TP")
    local_kv = max(num_kv_heads // tp, 1)
    ftp = intermediate // mlp_tp
    if intermediate % mlp_tp != 0:
        raise SpecificationLoadError("intermediate_size must be divisible by MLP_TP")
    vtp = vocab // lmhead_tp
    if vocab % lmhead_tp != 0:
        raise SpecificationLoadError("vocab_size must be divisible by lmhead_tp_size")

    embedding_tp_mode = config.get("word_embedding_tp")
    if embedding_tp_mode not in {None, "col", "row"}:
        raise SpecificationLoadError("model_config.word_embedding_tp must be 'col', 'row', or null")
    embedding_vocab = int(math.ceil(vocab / tp)) if embedding_tp_mode == "row" else vocab
    embedding_hidden = int(math.ceil(hidden / tp)) if embedding_tp_mode == "col" else hidden
    embedding_output_hidden = embedding_hidden if embedding_tp_mode == "col" else hidden

    validate_mtp_decode_window(context)
    mtp = parse_num_mtp_tokens(context)
    rtgt = local_batch * (mtp + 1)
    rprop = local_batch
    if context.phase is ExecutionPhase.PREFILL:
        rout = local_batch
    elif context.phase is ExecutionPhase.DECODE:
        rout = tokens
    else:
        rout = local_batch

    # Paged-attention pool sizing mirrors tensor_cast.core.input_generator.generate_inputs.
    block_size = _optional_config_int(config, "block_size", default=128)
    max_context_length = seq + mtp + 1
    num_blocks = (max_context_length * local_batch + block_size - 1) // block_size
    max_blocks_per_seq = (seq + block_size - 1) // block_size

    return {
        "B": local_batch,
        "Q": query,
        "C": context_length,
        "S": seq,
        "T": tokens,
        "H": hidden,
        "V": vocab,
        "F": intermediate,
        "Nh": num_heads,
        "Nkv": num_kv_heads,
        "Dh": head_dim,
        "TP": tp,
        "DP": dp,
        "EP": ep,
        "MLP_TP": mlp_tp,
        "OTP": o_proj_tp,
        "LMTP": lmhead_tp,
        "MTP": mtp,
        "Rtgt": rtgt,
        "Rprop": rprop,
        "Lh": local_heads,
        "Lkv": local_kv,
        "Ftp": ftp,
        "Vtp": vtp,
        "EV": embedding_vocab,
        "EH": embedding_hidden,
        "EO": embedding_output_hidden,
        "Rout": rout,
        "Bs": block_size,
        "Nblk": num_blocks,
        "Mb": max_blocks_per_seq,
        "D": dtype,
        "ACT": activation_dtype,
        "LINEAR_IN": linear_input_dtype,
        "WEIGHT": weight_dtype,
        "SCALE": scale_dtype,
        "ACC": accumulation_dtype,
        "OUT": output_dtype,
        "int64": "int64",
        "unknown": None,
    }
